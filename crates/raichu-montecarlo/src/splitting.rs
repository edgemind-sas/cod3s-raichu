//! Generalized adaptive multilevel splitting (Bréhier et al., 2016,
//! sections 2.3-2.5). All particles tied at the minimum are replaced,
//! branching at the first strictly higher completed-instant score.
//! Independent batches provide a Student interval with explicit extinction
//! diagnostics. A finite iteration cap produces an error, never an estimate.

use crate::confidence::{batch_interval, is_valid_level, BatchInterval, DEFAULT_CONFIDENCE};
use raichu_core::{
    CompiledModel, Engine, EngineConfig, EngineError, FlowConfig, Snapshot, SolverParams,
};
use raichu_expr::Value;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Default particles in each independent batch, calibrated on the 2oo3 witness.
pub const DEFAULT_SPLITTING_PARTICLES: u64 = 16_000;
/// Default independent batches.
pub const DEFAULT_SPLITTING_BATCHES: u64 = 20;
/// Maximum resampling iterations per batch.
pub const DEFAULT_SPLITTING_MAX_ITERATIONS: u64 = 10_000;
/// No continuous score sampling unless a grid is declared.
pub const DEFAULT_SPLITTING_SCORE_GRID: &[f64] = &[];

/// Source of the importance score, read only after a complete instant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ImportanceSource {
    /// Numeric model attribute, qualified as `component.attribute`.
    Attribute {
        /// Qualified attribute name.
        name: String,
    },
}
/// Settings for a splitting campaign. Particles is the population per batch.
#[derive(Debug, Clone)]
pub struct SplittingSettings {
    /// First-hit target whose probability is estimated.
    pub target: String,
    /// Finite, nonnegative trajectory horizon in model time units.
    pub t_max: f64,
    /// Master seed, shared by disjoint deterministic streams.
    pub seed: u64,
    /// Population in each independent batch, at least two.
    pub particles: u64,
    /// Independent batches, at least two.
    pub batches: u64,
    /// Numeric importance source.
    pub importance: ImportanceSource,
    /// Strictly increasing finite times in `[0, t_max]` for continuous scores.
    pub score_grid: Vec<f64>,
    /// Maximum replacement iterations per batch; zero allows no replacement.
    pub max_iterations: u64,
    /// Two-sided confidence level in `(0, 1)`.
    pub confidence: f64,
    /// Rayon worker count, independent of the result.
    pub threads: Option<usize>,
    /// Continuous integration parameters.
    pub ode: SolverParams,
    /// Flow resolution policy.
    pub flow: FlowConfig,
}
impl Default for SplittingSettings {
    fn default() -> Self {
        Self {
            target: String::new(),
            t_max: 1.0,
            seed: 0,
            particles: DEFAULT_SPLITTING_PARTICLES,
            batches: DEFAULT_SPLITTING_BATCHES,
            importance: ImportanceSource::Attribute {
                name: String::new(),
            },
            score_grid: DEFAULT_SPLITTING_SCORE_GRID.to_vec(),
            max_iterations: DEFAULT_SPLITTING_MAX_ITERATIONS,
            confidence: DEFAULT_CONFIDENCE,
            threads: None,
            ode: SolverParams::default(),
            flow: FlowConfig::default(),
        }
    }
}
/// One finite selection level and the exact survival factor it contributed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplittingLevel {
    /// Lowest running maximum before replacement, always finite.
    pub level: f64,
    /// Number of particles at or below the selection level.
    pub killed: u64,
    /// Surviving fraction, `1-killed/particles`.
    pub survival_fraction: f64,
}
/// One independent batch, including an explicit zero for extinction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplittingBatch {
    /// Product of survival fractions times the final hit fraction.
    pub estimate: f64,
    /// All particles tied below the target at termination.
    pub extinct: bool,
    /// Finite replacement levels in increasing order.
    pub levels: Vec<SplittingLevel>,
    /// Number of initial and restarted trajectories.
    pub trajectories: u64,
}
/// Ordered batch estimates, interval and conservative confidence verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplittingEstimate {
    /// Requested first-hit target.
    pub target: String,
    /// Explicit master seed.
    pub seed: u64,
    /// Student interval over the independent batch estimates.
    pub interval: BatchInterval,
    /// Batches in deterministic index order.
    pub batches: Vec<SplittingBatch>,
    /// Number of extinct batches included as zero in the mean.
    pub extinct_batches: u64,
    /// Any extinct batch, nonpositive lower bound or relative half-width > 1.
    pub estimate_inconclusive: bool,
    /// Total initial and restarted trajectories.
    pub trajectories: u64,
}
/// Invalid settings, engine failures or a campaign with no completed estimate.
#[derive(Debug, thiserror::Error)]
pub enum SplittingError {
    /// A setting is invalid before any trajectory is sampled.
    #[error("invalid splitting setting `{name}`: {detail}")]
    InvalidSetting {
        /// Setting name.
        name: String,
        /// Explanation.
        detail: String,
    },
    /// A numerical or model failure during a trajectory.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// No finite estimate is valid for a truncated splitting batch.
    #[error("splitting inconclusive: batch {batch} reached the iteration cap ({iterations}); no estimate is returned")]
    IterationLimit {
        /// Zero-based batch index.
        batch: u64,
        /// Completed replacement iterations.
        iterations: u64,
    },
}
fn invalid(name: &str, detail: impl Into<String>) -> SplittingError {
    SplittingError::InvalidSetting {
        name: name.into(),
        detail: detail.into(),
    }
}
const STREAM_BLOCK: u64 = 1 << 32;
fn stream(batch: u64, rank: u64) -> Result<u64, SplittingError> {
    if rank >= STREAM_BLOCK {
        return Err(invalid(
            "particles/max_iterations",
            "per-batch stream space exhausted",
        ));
    }
    let index = batch
        .checked_mul(STREAM_BLOCK)
        .and_then(|b| b.checked_add(rank))
        .ok_or_else(|| invalid("batches", "stream index overflow"))?;
    raichu_rng::final_stream(index).map_err(|e| invalid("batches", e.to_string()))
}
fn validate(m: &CompiledModel, s: &SplittingSettings) -> Result<(), SplittingError> {
    if !m.targets.iter().any(|t| t.name == s.target) {
        return Err(invalid("target", format!("unknown target `{}`", s.target)));
    }
    if !s.t_max.is_finite() || s.t_max < 0.0 {
        return Err(invalid("t_max", "expected a finite nonnegative horizon"));
    }
    if s.particles < 2 || s.particles >= STREAM_BLOCK || usize::try_from(s.particles).is_err() {
        return Err(invalid("particles", "expected 2..2^32 particles"));
    }
    if s.batches < 2 || s.batches >= (1 << 31) || usize::try_from(s.batches).is_err() {
        return Err(invalid("batches", "expected 2..2^31 independent batches"));
    }
    if !is_valid_level(s.confidence) {
        return Err(invalid(
            "confidence",
            "expected a value strictly inside (0, 1)",
        ));
    }
    if s.threads == Some(0) {
        return Err(invalid("threads", "expected a positive count"));
    }
    if s.score_grid
        .iter()
        .any(|t| !t.is_finite() || *t < 0.0 || *t > s.t_max)
        || s.score_grid.windows(2).any(|p| p[0] >= p[1])
    {
        return Err(invalid(
            "score_grid",
            "expected strictly increasing finite dates inside the horizon",
        ));
    }
    let ImportanceSource::Attribute { name } = &s.importance;
    let idx = m
        .var_names
        .iter()
        .position(|n| n == name)
        .ok_or_else(|| invalid("importance", format!("unknown attribute `{name}`")))?;
    if !matches!(m.var_init[idx], Value::Int(_) | Value::Float(_)) {
        return Err(invalid(
            "importance",
            format!("attribute `{name}` is not numeric"),
        ));
    }
    if let Some(unit) = m.fmu_units.first() {
        return Err(EngineError::FmuSnapshotApi {
            unit: unit.name.clone(),
            operation: "splitting",
            alternative: "a native model without imported FMU units",
        }
        .into());
    }
    Ok(())
}
fn config(s: &SplittingSettings, stream: u64) -> EngineConfig {
    EngineConfig {
        t_max: s.t_max,
        seed: s.seed,
        rng_stream: stream,
        stop_at_targets: true,
        ode: s.ode.clone(),
        flow: s.flow.clone(),
        ..Default::default()
    }
}
#[derive(Clone)]
struct Crossing {
    level: f64,
    snapshot: Arc<Snapshot>,
}
struct Particle {
    crossings: Vec<Crossing>,
    level: f64,
}
fn score(e: &Engine<'_>, s: &SplittingSettings) -> Result<f64, SplittingError> {
    if e.reached_target()
        .is_some_and(|(target, _)| target == s.target)
    {
        return Ok(f64::INFINITY);
    }
    let ImportanceSource::Attribute { name } = &s.importance;
    let v = match e.attribute(name) {
        Some(Value::Float(v)) => v,
        Some(Value::Int(v)) => v as f64,
        _ => {
            return Err(invalid(
                "importance",
                format!("attribute `{name}` is not numeric"),
            ))
        }
    };
    if !v.is_finite() {
        return Err(invalid(
            "importance",
            format!(
                "attribute `{name}` has non-finite score at time {}",
                e.current_time()
            ),
        ));
    }
    Ok(v)
}
fn trajectory(
    m: &CompiledModel,
    s: &SplittingSettings,
    id: u64,
    branch: Option<Crossing>,
) -> Result<Particle, SplittingError> {
    let mut e = if let Some(c) = &branch {
        Engine::restart_from_snapshot(m, config(s, id), &c.snapshot, id)?
    } else {
        Engine::new(m, config(s, id))?
    };
    e.advance_to(e.current_time())?;
    let mut crossings = branch.into_iter().collect::<Vec<_>>();
    let mut level = crossings.last().map_or(f64::NEG_INFINITY, |c| c.level);
    let mut grid = s.score_grid.partition_point(|t| *t <= e.current_time());
    loop {
        let value = score(&e, s)?;
        if value > level {
            level = value;
            e.forget_history();
            crossings.push(Crossing {
                level,
                snapshot: Arc::new(e.snapshot()?),
            });
        }
        e.forget_history();
        if e.reached_target().is_some() || e.current_time() >= s.t_max {
            break;
        }
        let until = s.score_grid.get(grid).copied().unwrap_or(s.t_max);
        let event = e.step_until(until)?;
        // This completes all ties and cascades before observing the score.
        e.advance_to(e.current_time())?;
        if e.current_time() >= until {
            grid += usize::from(grid < s.score_grid.len());
        }
        if event.is_none() && until == s.t_max {
            let value = score(&e, s)?;
            if value > level {
                level = value;
                e.forget_history();
                crossings.push(Crossing {
                    level,
                    snapshot: Arc::new(e.snapshot()?),
                });
            }
            break;
        }
    }
    Ok(Particle { crossings, level })
}
fn batch(
    m: &CompiledModel,
    s: &SplittingSettings,
    b: u64,
) -> Result<SplittingBatch, SplittingError> {
    // Rank zero belongs to parent selection. Every trajectory launch uses
    // a new positive rank; batches occupy disjoint blocks in the final range.
    let mut selection = raichu_rng::replica_rng(s.seed, stream(b, 0)?);
    let initial: Vec<_> = (1..=s.particles)
        .into_par_iter()
        .map(|rank| trajectory(m, s, stream(b, rank)?, None))
        .collect();
    let mut particles = initial.into_iter().collect::<Result<Vec<_>, _>>()?;
    let mut rank = s.particles + 1;
    let mut weight = 1.0;
    let mut levels = Vec::new();
    loop {
        let minimum = particles
            .iter()
            .map(|p| p.level)
            .fold(f64::INFINITY, f64::min);
        if minimum == f64::INFINITY {
            return Ok(SplittingBatch {
                estimate: weight,
                extinct: false,
                levels,
                trajectories: rank - 1,
            });
        }
        let survivors: Vec<_> = particles
            .iter()
            .enumerate()
            .filter_map(|(i, p)| (p.level > minimum).then_some(i))
            .collect();
        if survivors.is_empty() {
            return Ok(SplittingBatch {
                estimate: 0.0,
                extinct: true,
                levels,
                trajectories: rank - 1,
            });
        }
        if levels.len() as u64 >= s.max_iterations {
            return Err(SplittingError::IterationLimit {
                batch: b,
                iterations: s.max_iterations,
            });
        }
        let killed: Vec<_> = particles
            .iter()
            .enumerate()
            .filter_map(|(i, p)| (p.level <= minimum).then_some(i))
            .collect();
        let survival_fraction = 1.0 - killed.len() as f64 / s.particles as f64;
        weight *= survival_fraction;
        levels.push(SplittingLevel {
            level: minimum,
            killed: killed.len() as u64,
            survival_fraction,
        });
        for i in killed {
            let parent = survivors[selection.random_range(0..survivors.len())];
            let crossing = particles[parent]
                .crossings
                .iter()
                .find(|c| c.level > minimum)
                .cloned()
                .ok_or_else(|| invalid("importance", "survivor has no strict crossing"))?;
            particles[i] = trajectory(m, s, stream(b, rank)?, Some(crossing))?;
            rank += 1;
        }
    }
}
/// Estimate a first-hit target probability by generalized AMS.
///
/// Initial waves and independent batches run in parallel; reductions and clone
/// launch ranks are serial. Equal seeds therefore give byte-identical results
/// at every thread count. Invalid settings fail before any random draw.
///
/// # Errors
/// Invalid settings, native engine errors, stream exhaustion, or an iteration
/// cap. The latter is inconclusive and deliberately contains no numeric estimate.
pub fn run_splitting(
    m: &CompiledModel,
    s: &SplittingSettings,
) -> Result<SplittingEstimate, SplittingError> {
    validate(m, s)?;
    let compute = || {
        let batches: Vec<_> = (0..s.batches)
            .into_par_iter()
            .map(|b| batch(m, s, b))
            .collect();
        batches.into_iter().collect::<Result<Vec<_>, _>>()
    };
    let batches = match s.threads {
        None => compute(),
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| invalid("threads", e.to_string()))?
            .install(compute),
    }?;
    let values: Vec<_> = batches.iter().map(|b| b.estimate).collect();
    let interval = batch_interval(&values, s.confidence);
    if !interval.low.is_finite() || !interval.high.is_finite() {
        return Err(invalid(
            "confidence",
            "Student interval could not be computed",
        ));
    }
    let extinct_batches = batches.iter().filter(|b| b.extinct).count() as u64;
    let estimate_inconclusive = extinct_batches > 0
        || interval.low <= 0.0
        || (interval.high - interval.low) / 2.0 > interval.estimate;
    let trajectories = batches.iter().map(|b| b.trajectories).sum();
    Ok(SplittingEstimate {
        target: s.target.clone(),
        seed: s.seed,
        interval,
        batches,
        extinct_batches,
        estimate_inconclusive,
        trajectories,
    })
}
