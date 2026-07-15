//! # raichu-montecarlo — parallel replica driver
//!
//! Runs `nb_runs` independent trajectories of a compiled model and
//! estimates indicator statistics at a sampling schedule.
//!
//! Reproducibility contract:
//!
//! - replica `r` uses the RNG substream `r` of the master seed
//!   (`raichu-rng` policy) — replicas are independent by construction
//!   and each one replays bit-identically;
//! - replica results are collected **in replica order** and reduced by
//!   a **serial, index-ordered fold**: floating-point addition is not
//!   associative, so this is what makes 1-thread and N-thread runs
//!   produce *identical bytes*, not just statistically equal numbers;
//! - `rayon` only parallelises the embarrassingly parallel trajectory
//!   loop — the single-trajectory engine stays single-threaded.
//!
//! Estimators per indicator and schedule instant: mean and sample
//! standard deviation of the sampled value, plus the **sojourn time**
//! (time-integral of the indicator value up to the instant — for 0/1
//! state indicators this is the classic cumulated-sojourn estimator,
//! the sojourn-time measure).

use raichu_core::{
    CompiledModel, Engine, EngineConfig, EngineError, IndicatorSeries, Sequence, SolverParams,
};
use raichu_expr::Value;
use serde::Serialize;

/// Monte-Carlo run parameters.
#[derive(Debug, Clone)]
pub struct McConfig {
    /// Number of replicas.
    pub nb_runs: u64,
    /// Master seed (replica `r` uses substream `r`).
    pub seed: u64,
    /// Horizon of each trajectory.
    pub t_max: f64,
    /// Ascending sampling instants (also the estimator support).
    pub samples: Vec<f64>,
    /// Thread count (`None` = rayon default). The result is
    /// byte-identical whatever the value — see the crate docs.
    pub threads: Option<usize>,
    /// Quantile orders to estimate (e.g. `[0.25, 0.75]`), on both the
    /// sampled value and the cumulated sojourn, nearest-rank across
    /// replicas (M4; quantile stats, e.g. P25/P75).
    pub quantiles: Vec<f64>,
    /// Numerical parameters of the ODE backend for every replica
    /// (engine defaults unless overridden — the knob of the
    /// tolerance-parity experiments; recorded as provenance upstream).
    pub ode: SolverParams,
    /// Early-stop each trajectory at the first sequence target (feared
    /// event) and hold the frozen state through the remaining sample
    /// instants — the latch semantics of target-stopped studies (the
    /// reference regime of recorded sequence campaigns). Ignored when the
    /// model declares no target.
    pub stop_at_targets: bool,
}

/// A quantile series over the schedule instants.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QuantileSeries {
    /// Quantile order in (0, 1).
    pub q: f64,
    /// Nearest-rank quantile at each schedule instant.
    pub values: Vec<f64>,
}

/// Estimates of one indicator over the schedule.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IndicatorEstimate {
    /// Indicator name.
    pub name: String,
    /// Schedule instants.
    pub instants: Vec<f64>,
    /// Mean of the sampled value at each instant.
    pub mean: Vec<f64>,
    /// Sample standard deviation (ddof = 1) at each instant.
    pub std: Vec<f64>,
    /// Mean cumulated sojourn (time-integral of the value) up to each
    /// instant.
    pub sojourn_mean: Vec<f64>,
    /// Sample standard deviation of the cumulated sojourn.
    pub sojourn_std: Vec<f64>,
    /// Mean number of occurrences (state entries / rising edges) up to each
    /// instant — the RAMS `nb-occurrences` measure.
    pub nb_occurrences_mean: Vec<f64>,
    /// Sample standard deviation of the occurrence count.
    pub nb_occurrences_std: Vec<f64>,
    /// Requested quantiles of the sampled value.
    pub quantiles: Vec<QuantileSeries>,
    /// Requested quantiles of the cumulated sojourn.
    pub sojourn_quantiles: Vec<QuantileSeries>,
}

/// Full Monte-Carlo result with provenance.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct McEstimates {
    /// Per-indicator estimates.
    pub indicators: Vec<IndicatorEstimate>,
    /// Number of replicas.
    pub nb_runs: u64,
    /// Master seed.
    pub seed: u64,
    /// Engine version.
    pub engine_version: String,
}

fn value_as_f64(value: Value) -> f64 {
    match value {
        Value::Bool(b) => f64::from(u8::from(b)),
        Value::Int(i) => i as f64,
        Value::Float(f) => f,
    }
}

/// Cumulated time-integral of a change-point series at `instant`.
fn sojourn_at(points: &[(f64, Value)], instant: f64) -> f64 {
    let mut acc = 0.0;
    for (idx, (t_start, value)) in points.iter().enumerate() {
        if *t_start >= instant {
            break;
        }
        let t_end = points
            .get(idx + 1)
            .map_or(instant, |(t_next, _)| t_next.min(instant));
        acc += value_as_f64(*value) * (t_end - t_start);
    }
    acc
}

/// Number of **occurrences** (state entries) up to `instant`: the count of
/// rising edges in the change-point series — a value going from ≤ 0 to > 0,
/// including an active initial value. This is the RAMS `nb-occurrences`
/// measure (how many times the feared event happened by `instant`).
fn nb_occurrences_at(points: &[(f64, Value)], instant: f64) -> f64 {
    let mut count = 0.0;
    let mut prev = 0.0;
    for (t_start, value) in points {
        // Inclusive bound: an entry AT the sample instant counts, matching
        // the sampled value at that instant (which reflects the post-event
        // state) — the two measures stay mutually consistent.
        if *t_start > instant {
            break;
        }
        let cur = value_as_f64(*value);
        if prev <= 0.0 && cur > 0.0 {
            count += 1.0;
        }
        prev = cur;
    }
    count
}

/// One replica's samples: `[indicator][instant] → (value, sojourn, nb_occ)`.
type ReplicaSamples = Vec<Vec<(f64, f64, f64)>>;

fn run_replica(
    model: &CompiledModel,
    config: &McConfig,
    replica: u64,
) -> Result<ReplicaSamples, EngineError> {
    let engine_config = EngineConfig {
        t_max: config.t_max,
        samples: config.samples.clone(),
        seed: config.seed,
        rng_stream: replica,
        ode: config.ode.clone(),
        sequences: config.stop_at_targets,
        ..EngineConfig::default()
    };
    let result = Engine::new(model, engine_config)?.run()?;
    let per_indicator = result
        .samples
        .iter()
        .zip(&result.indicators)
        .map(
            |(sampled, change_points): (&IndicatorSeries, &IndicatorSeries)| {
                config
                    .samples
                    .iter()
                    .zip(&sampled.points)
                    .map(|(instant, (_, value))| {
                        (
                            value_as_f64(*value),
                            sojourn_at(&change_points.points, *instant),
                            nb_occurrences_at(&change_points.points, *instant),
                        )
                    })
                    .collect()
            },
        )
        .collect();
    Ok(per_indicator)
}

/// Run the Monte-Carlo estimation.
///
/// Replicas run in parallel; the reduction is a serial fold in replica
/// order, so the estimates are bit-identical for any thread count.
pub fn run(model: &CompiledModel, config: &McConfig) -> Result<McEstimates, EngineError> {
    use rayon::prelude::*;

    let compute = || -> Result<Vec<ReplicaSamples>, EngineError> {
        (0..config.nb_runs)
            .into_par_iter()
            .map(|replica| run_replica(model, config, replica))
            .collect()
    };
    let replicas = match config.threads {
        None => compute()?,
        Some(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| EngineError::TypeError {
                time: 0.0,
                detail: format!("thread-pool construction failed: {e}"),
            })?
            .install(compute)?,
    };

    let n_indicators = model.indicators.len();
    let n_instants = config.samples.len();
    let n = config.nb_runs as f64;

    let mut indicators = Vec::with_capacity(n_indicators);
    for (idx, indicator) in model.indicators.iter().enumerate() {
        let mut mean = vec![0.0; n_instants];
        let mut std = vec![0.0; n_instants];
        let mut sojourn_mean = vec![0.0; n_instants];
        let mut sojourn_std = vec![0.0; n_instants];
        let mut nb_occurrences_mean = vec![0.0; n_instants];
        let mut nb_occurrences_std = vec![0.0; n_instants];
        for k in 0..n_instants {
            // Serial, replica-ordered accumulation (determinism).
            let (mut sum, mut sum_sq, mut sj_sum, mut sj_sum_sq) = (0.0, 0.0, 0.0, 0.0);
            let (mut oc_sum, mut oc_sum_sq) = (0.0, 0.0);
            for replica in &replicas {
                let (value, sojourn, nb_occ) = replica[idx][k];
                sum += value;
                sum_sq += value * value;
                sj_sum += sojourn;
                sj_sum_sq += sojourn * sojourn;
                oc_sum += nb_occ;
                oc_sum_sq += nb_occ * nb_occ;
            }
            mean[k] = sum / n;
            sojourn_mean[k] = sj_sum / n;
            nb_occurrences_mean[k] = oc_sum / n;
            if config.nb_runs > 1 {
                std[k] = ((sum_sq - n * mean[k] * mean[k]) / (n - 1.0))
                    .max(0.0)
                    .sqrt();
                sojourn_std[k] = ((sj_sum_sq - n * sojourn_mean[k] * sojourn_mean[k]) / (n - 1.0))
                    .max(0.0)
                    .sqrt();
                nb_occurrences_std[k] =
                    ((oc_sum_sq - n * nb_occurrences_mean[k] * nb_occurrences_mean[k]) / (n - 1.0))
                        .max(0.0)
                        .sqrt();
            }
        }
        // Nearest-rank quantiles (deterministic: total_cmp sort over the
        // replica-ordered column).
        let mut quantiles = Vec::new();
        let mut sojourn_quantiles = Vec::new();
        for &q in &config.quantiles {
            let mut value_rows = vec![0.0; n_instants];
            let mut sojourn_rows = vec![0.0; n_instants];
            for k in 0..n_instants {
                let mut column: Vec<f64> =
                    replicas.iter().map(|replica| replica[idx][k].0).collect();
                let mut sj_column: Vec<f64> =
                    replicas.iter().map(|replica| replica[idx][k].1).collect();
                column.sort_unstable_by(f64::total_cmp);
                sj_column.sort_unstable_by(f64::total_cmp);
                let rank = ((q * column.len() as f64).ceil() as usize)
                    .saturating_sub(1)
                    .min(column.len().saturating_sub(1));
                value_rows[k] = column[rank];
                sojourn_rows[k] = sj_column[rank];
            }
            quantiles.push(QuantileSeries {
                q,
                values: value_rows,
            });
            sojourn_quantiles.push(QuantileSeries {
                q,
                values: sojourn_rows,
            });
        }
        indicators.push(IndicatorEstimate {
            name: indicator.name.clone(),
            instants: config.samples.clone(),
            mean,
            std,
            sojourn_mean,
            sojourn_std,
            nb_occurrences_mean,
            nb_occurrences_std,
            quantiles,
            sojourn_quantiles,
        });
    }

    Ok(McEstimates {
        indicators,
        nb_runs: config.nb_runs,
        seed: config.seed,
        engine_version: env!("CARGO_PKG_VERSION").to_owned(),
    })
}

/// Run `nb_runs` **sequence-recording** replicas and collect their raw
/// per-trajectory sequences, in replica order (deterministic). Each replica
/// runs with sequence recording on and target early-stop; a trajectory that
/// reaches no target still contributes its (target-less) sequence. Feed the
/// result to [`raichu_core::analyse`] for the minimal-sequence corpus.
pub fn run_sequences(
    model: &CompiledModel,
    config: &McConfig,
) -> Result<Vec<Sequence>, EngineError> {
    use rayon::prelude::*;

    let compute = || -> Result<Vec<Option<Sequence>>, EngineError> {
        (0..config.nb_runs)
            .into_par_iter()
            .map(|replica| {
                let engine_config = EngineConfig {
                    t_max: config.t_max,
                    sequences: true,
                    seed: config.seed,
                    rng_stream: replica,
                    ode: config.ode.clone(),
                    ..EngineConfig::default()
                };
                Ok(Engine::new(model, engine_config)?.run()?.sequence)
            })
            .collect()
    };
    let per = match config.threads {
        None => compute()?,
        Some(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| EngineError::TypeError {
                time: 0.0,
                detail: format!("thread-pool construction failed: {e}"),
            })?
            .install(compute)?,
    };
    Ok(per.into_iter().flatten().collect())
}
