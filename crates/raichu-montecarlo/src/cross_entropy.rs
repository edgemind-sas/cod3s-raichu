//! Biased Monte-Carlo for rare feared events, with bias factors fitted by
//! cross-entropy.
//!
//! A plain campaign needs on the order of `100 / p` replicas to estimate a
//! probability `p` to about 10 %. This driver draws the dates of
//! constant-rate exponential transitions under **multiplied rates**, so the
//! feared event is reached often, and weights each replica by the
//! **likelihood ratio** of the nominal law against the biased one, so the
//! weighted proportion is still an unbiased estimate of `p` (de Boer,
//! Kroese, Mannor and Rubinstein 2005, *A Tutorial on the Cross-Entropy
//! Method*, Annals of Operations Research 134; Ridder 2005, same volume, for
//! Markovian reliability systems).
//!
//! # Families
//!
//! One factor is fitted per **family** of eligible transitions. A
//! transition is eligible when its law is a constant-rate exponential
//! (`CLaw::Exp`); every other transition runs unbiased and contributes a
//! factor of 1. Families are keyed by the transition's structure and law:
//! local automaton name, local transition name, source and target state
//! names, and the rate. Components instantiated from one class with one
//! rate therefore share a factor. An override map names, per transition,
//! the family it belongs to.
//!
//! # Likelihood ratio and the fit
//!
//! For a constant-rate exponential of nominal rate `λ` drawn at `f·λ`, the
//! ratio over a trajectory is `f^(-n) · exp((f - 1)·H)`, with `n` its firing
//! count and `H` its nominal exposure (`λ` times the time it was armed and
//! running), both reported by the engine
//! ([`raichu_core::TransitionExposure`]). Per family, the cross-entropy
//! update is the maximum-likelihood rate under the weighted hits: the new
//! factor is `Σ W·n / Σ W·H` over the pilot replicas that reached the
//! target, smoothed against the previous factor and clamped to
//! `[factor_min, factor_max]`. A family with zero weighted count or zero
//! weighted exposure keeps its previous factor: without that, a family
//! that never fires in a hitting replica (a repair, a component unrelated
//! to the target) would slide toward 0 and a later nominal firing of it
//! would carry a weight of about `1/f`.
//!
//! When a pilot sees no hit, every family not declared a repair is
//! multiplied by `escalation_ratio` (clamped to the ceiling). When no pilot
//! ever hits, or when the final campaign records no hit, the driver
//! returns [`CrossEntropyError::NoHit`]: a zero is never returned as an
//! estimate.
//!
//! # Reproducibility
//!
//! The final campaign draws replica `i` on stream `i`
//! ([`raichu_rng::final_stream`]), exactly as a plain campaign does, so with
//! every factor at 1 and no fit it reproduces [`crate::run_to_targets`] bit
//! for bit. Pilot iteration `k` draws on its own block of streams
//! ([`raichu_rng::pilot_stream`]), never reused by the estimate. Replicas run
//! in parallel and are folded serially in replica order: the result is
//! byte-identical whatever the thread count.

use std::collections::BTreeMap;

use raichu_core::compile::CLaw;
use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError, FlowConfig, SolverParams};
use raichu_model::TransitionKind;
use serde::{Deserialize, Serialize};

use crate::confidence::{weighted_interval, WeightedInterval, DEFAULT_CONFIDENCE};
use crate::{in_pool, is_valid_level, replica_end, ReplicaEnd};

/// Settings of a cross-entropy campaign. Every field has a documented
/// default ([`Default`]); `target` and `t_max` have none that makes sense
/// and must be set.
#[derive(Debug, Clone)]
pub struct CrossEntropySettings {
    /// The feared event whose probability of being the **first** target
    /// reached by `t_max` is estimated.
    pub target: String,
    /// Horizon of every trajectory.
    pub t_max: f64,
    /// Master seed of the campaign (final replica `i` uses stream `i`).
    pub seed: u64,
    /// Sampling instants handed to the engine, as a plain campaign would
    /// be; they change no trajectory of a discrete model. Default: none.
    pub samples: Vec<f64>,
    /// Replicas of the final campaign. Default 10 000.
    pub nb_runs: u64,
    /// Replicas of each pilot iteration. Default 1 000.
    pub pilot_runs: u64,
    /// Cap on pilot iterations, escalations included. Default 20.
    pub max_iterations: u64,
    /// Weight of the new factor against the previous one, in `(0, 1]`.
    /// Default 0.7.
    pub smoothing: f64,
    /// The fit stops when no factor moves by more than this relative
    /// amount. Default 0.02.
    pub tolerance: f64,
    /// Confidence level of the final interval. Default 0.95.
    pub confidence: f64,
    /// Starting factor of every family not declared a repair (repairs
    /// start at 1). Default 10.
    pub initial_factor: f64,
    /// Multiplier applied to non-repair families after a pilot with no
    /// hit, `> 1`. Default 10.
    pub escalation_ratio: f64,
    /// Lowest admissible factor, `> 0`. Default 1e-2.
    pub factor_min: f64,
    /// Highest admissible factor. Default 1e4.
    pub factor_max: f64,
    /// Whether factors are fitted. `false` runs the final campaign at the
    /// starting factors (the structural check at factor 1). Default true.
    pub fit: bool,
    /// Effective sample size below which the hits of a campaign are too
    /// few, or too unevenly weighted, to be trusted: a pilot under it
    /// cannot confirm a fit, and a final campaign under it is flagged
    /// [`CrossEntropyEstimate::estimate_inconclusive`]. Default 50.
    pub min_effective_sample_size: f64,
    /// Override of the automatic families: qualified transition name
    /// (`component.automaton.transition`) to family label.
    pub families: BTreeMap<String, String>,
    /// Worker threads (`None` = rayon default); no result depends on it.
    pub threads: Option<usize>,
    /// ODE solver parameters of every trajectory.
    pub ode: SolverParams,
    /// Flow-resolution policy of every trajectory.
    pub flow: FlowConfig,
}

/// Default final replicas ([`CrossEntropySettings::nb_runs`]).
pub const DEFAULT_CE_NB_RUNS: u64 = 10_000;
/// Default replicas per pilot iteration ([`CrossEntropySettings::pilot_runs`]).
pub const DEFAULT_CE_PILOT_RUNS: u64 = 1_000;
/// Default cap on pilot iterations ([`CrossEntropySettings::max_iterations`]).
pub const DEFAULT_CE_MAX_ITERATIONS: u64 = 20;
/// Default smoothing weight ([`CrossEntropySettings::smoothing`]).
pub const DEFAULT_CE_SMOOTHING: f64 = 0.7;
/// Default relative tolerance of the fit ([`CrossEntropySettings::tolerance`]).
pub const DEFAULT_CE_TOLERANCE: f64 = 0.02;
/// Default starting factor ([`CrossEntropySettings::initial_factor`]).
pub const DEFAULT_CE_INITIAL_FACTOR: f64 = 10.0;
/// Default escalation ratio ([`CrossEntropySettings::escalation_ratio`]).
pub const DEFAULT_CE_ESCALATION_RATIO: f64 = 10.0;
/// Default factor floor ([`CrossEntropySettings::factor_min`]).
pub const DEFAULT_CE_FACTOR_MIN: f64 = 1e-2;
/// Default factor ceiling ([`CrossEntropySettings::factor_max`]).
pub const DEFAULT_CE_FACTOR_MAX: f64 = 1e4;
/// Default trust threshold on the effective sample size
/// ([`CrossEntropySettings::min_effective_sample_size`]). Measured on
/// 2026-09-28: every estimate recovered within its interval had an
/// effective sample size of 48 or more; the estimates found 30 % to three
/// orders of magnitude off had 35 or less.
pub const DEFAULT_CE_MIN_EFFECTIVE_SAMPLE_SIZE: f64 = 50.0;

impl Default for CrossEntropySettings {
    fn default() -> Self {
        CrossEntropySettings {
            target: String::new(),
            t_max: 0.0,
            seed: 0,
            samples: Vec::new(),
            nb_runs: DEFAULT_CE_NB_RUNS,
            pilot_runs: DEFAULT_CE_PILOT_RUNS,
            max_iterations: DEFAULT_CE_MAX_ITERATIONS,
            smoothing: DEFAULT_CE_SMOOTHING,
            tolerance: DEFAULT_CE_TOLERANCE,
            confidence: DEFAULT_CONFIDENCE,
            initial_factor: DEFAULT_CE_INITIAL_FACTOR,
            escalation_ratio: DEFAULT_CE_ESCALATION_RATIO,
            factor_min: DEFAULT_CE_FACTOR_MIN,
            factor_max: DEFAULT_CE_FACTOR_MAX,
            fit: true,
            min_effective_sample_size: DEFAULT_CE_MIN_EFFECTIVE_SAMPLE_SIZE,
            families: BTreeMap::new(),
            threads: None,
            ode: SolverParams::default(),
            flow: FlowConfig::default(),
        }
    }
}

/// A family of eligible transitions sharing one bias factor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Family {
    /// Readable label: the override label, or
    /// `automaton.transition[source->targets]@rate`.
    pub label: String,
    /// Qualified names of the member transitions, in compile order.
    pub transitions: Vec<String>,
    /// Whether every member is declared a repair (it then starts at 1 and
    /// is never escalated).
    pub repair: bool,
    /// Compiled transition indices of the members.
    #[serde(skip)]
    pub indices: Vec<usize>,
}

/// A family and the factor the campaign used for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FittedFamily {
    /// The family's label.
    pub label: String,
    /// Qualified names of the member transitions.
    pub transitions: Vec<String>,
    /// Whether it is a repair family.
    pub repair: bool,
    /// The factor the final campaign drew its rates with.
    pub factor: f64,
}

/// One pilot iteration of the fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PilotIteration {
    /// Factors the pilot drew with, in family order.
    pub factors: Vec<f64>,
    /// Pilot replicas that reached the target first.
    pub hits: u64,
    /// Whether the pilot saw no hit and escalated the factors.
    pub escalated: bool,
    /// Effective sample size of the pilot's hit weights (0 with no hit):
    /// the fit is confirmed only from a pilot at or above
    /// [`CrossEntropySettings::min_effective_sample_size`].
    pub effective_sample_size: f64,
}

/// The answer of a cross-entropy campaign.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossEntropyEstimate {
    /// The target estimated.
    pub target: String,
    /// Weighted estimate, its interval and diagnostics (effective sample
    /// size, relative error).
    pub estimate: WeightedInterval,
    /// Final replicas that reached the target first (unweighted).
    pub reached: u64,
    /// Final replicas.
    pub replicas: u64,
    /// Families and the factors the final campaign used, in family order.
    pub families: Vec<FittedFamily>,
    /// The pilot iterations, in order (empty when not fitted).
    pub history: Vec<PilotIteration>,
    /// Whether the fit met its tolerance, on a pilot whose effective sample
    /// size reached the threshold, before the iteration cap (true when
    /// there was nothing to fit).
    pub converged: bool,
    /// Whether the final hits are too few or too unevenly weighted to trust
    /// the estimate and its interval: their effective sample size is below
    /// [`CrossEntropySettings::min_effective_sample_size`]. Measured on a
    /// repairable pair with fast repairs over a long horizon, such an
    /// estimate was from 30 % to three orders of magnitude off, with an
    /// interval that did not contain the truth.
    pub estimate_inconclusive: bool,
    /// How each final replica ended, in replica order. Omitted from the
    /// serialized form when empty: a quantification envelope clears it,
    /// one entry per replica being too large for a result document.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ends: Vec<ReplicaEnd>,
}

/// Why a cross-entropy campaign produced no estimate.
#[derive(Debug, thiserror::Error)]
pub enum CrossEntropyError {
    /// The engine refused a trajectory.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// The target names no declared target of the model.
    #[error(
        "the model declares no target named `{target}`; it declares {}",
        if declared.is_empty() { "none".to_owned() } else { declared.iter().map(|d| format!("`{d}`")).collect::<Vec<_>>().join(", ") }
    )]
    UnknownTarget {
        /// The name asked for.
        target: String,
        /// The targets the model declares, in declaration order.
        declared: Vec<String>,
    },
    /// The family override names a transition the model does not compile.
    #[error("the family override names `{transition}`, which is not a transition of the model")]
    UnknownTransition {
        /// The name given.
        transition: String,
    },
    /// The family override names a transition whose law cannot be biased.
    #[error(
        "the family override names `{transition}`, whose law ({law}) is not a \
         constant-rate exponential: it runs unbiased and belongs to no family"
    )]
    IneligibleTransition {
        /// The name given.
        transition: String,
        /// Its law.
        law: String,
    },
    /// A setting outside its domain.
    #[error("invalid cross-entropy setting `{setting}`: {detail}")]
    InvalidSetting {
        /// The setting.
        setting: String,
        /// What is wrong.
        detail: String,
    },
    /// No replica reached the target: no estimate is returned in place of
    /// a zero.
    #[error(
        "no {stage} replica reached `{target}` ({replicas} replicas, factors \
         {factors:?}): the target may be unreachable, or rarer than the \
         factor ceiling can reach; raise factor_max, nb_runs or \
         max_iterations"
    )]
    NoHit {
        /// The target.
        target: String,
        /// `pilot` (every pilot iteration) or `final`.
        stage: &'static str,
        /// Replicas of the stage that found nothing (per iteration for
        /// the pilots).
        replicas: u64,
        /// The factors in force, in family order.
        factors: Vec<f64>,
    },
}

/// The families of eligible transitions of `model`, with the override
/// `names` (qualified transition name to label) applied.
///
/// # Errors
/// [`CrossEntropyError::UnknownTransition`] or
/// [`CrossEntropyError::IneligibleTransition`] for an override entry.
pub fn cross_entropy_families(
    model: &CompiledModel,
    names: &BTreeMap<String, String>,
) -> Result<Vec<Family>, CrossEntropyError> {
    for name in names.keys() {
        let Some(t) = model.transitions.iter().find(|t| &t.name == name) else {
            return Err(CrossEntropyError::UnknownTransition {
                transition: name.clone(),
            });
        };
        if !matches!(t.distrib, CLaw::Exp(_)) {
            return Err(CrossEntropyError::IneligibleTransition {
                transition: name.clone(),
                law: law_label(&t.distrib).to_owned(),
            });
        }
    }
    // Families in first-member compile order, members in compile order.
    let mut order: Vec<String> = Vec::new();
    let mut members: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, t) in model.transitions.iter().enumerate() {
        let CLaw::Exp(rate) = t.distrib else {
            continue;
        };
        let label = match names.get(&t.name) {
            Some(label) => label.clone(),
            None => automatic_label(model, idx, rate),
        };
        let entry = members.entry(label.clone()).or_default();
        if entry.is_empty() {
            order.push(label);
        }
        entry.push(idx);
    }
    Ok(order
        .into_iter()
        .map(|label| {
            let indices = members.remove(&label).unwrap_or_default();
            Family {
                transitions: indices
                    .iter()
                    .map(|&i| model.transitions[i].name.clone())
                    .collect(),
                repair: indices
                    .iter()
                    .all(|&i| model.transitions[i].kind == Some(TransitionKind::Repair)),
                label,
                indices,
            }
        })
        .collect())
}

/// `automaton.transition[source->targets]@rate`, from local names: the
/// automatic family key.
fn automatic_label(model: &CompiledModel, idx: usize, rate: f64) -> String {
    let t = &model.transitions[idx];
    let automaton = &model.automata[t.automaton];
    let local = |qualified: &str| {
        qualified
            .split_once('.')
            .map_or(qualified, |(_, rest)| rest)
            .to_owned()
    };
    let transition = t.name.rsplit('.').next().unwrap_or(&t.name);
    let targets: Vec<&str> = t
        .targets
        .iter()
        .map(|&s| automaton.states[s].as_str())
        .collect();
    format!(
        "{}.{}[{}->{}]@{:e}",
        local(&automaton.name),
        transition,
        automaton.states[t.source],
        targets.join("|"),
        rate
    )
}

fn law_label(law: &CLaw) -> &'static str {
    match law {
        CLaw::Delay(_) => "delay",
        CLaw::Inst(_) => "instantaneous",
        CLaw::Watched { .. } => "watched",
        CLaw::Exp(_) => "exponential",
        CLaw::ExpVar { .. } => "exponential with a rate expression",
        CLaw::Weibull(..) => "weibull",
        CLaw::Lognormal(..) => "lognormal",
        CLaw::Gamma(..) => "gamma",
        CLaw::Uniform(..) => "uniform",
        CLaw::Empirical(..) => "empirical",
    }
}

/// One replica's contribution: whether it hit, its log-likelihood ratio,
/// per family (firings, nominal exposure), and how it ended.
struct ReplicaOutcome {
    hit: bool,
    log_weight: f64,
    per_family: Vec<(u64, f64)>,
    end: ReplicaEnd,
}

fn setting(setting: &str, detail: String) -> CrossEntropyError {
    CrossEntropyError::InvalidSetting {
        setting: setting.to_owned(),
        detail,
    }
}

fn validate(model: &CompiledModel, s: &CrossEntropySettings) -> Result<(), CrossEntropyError> {
    if !model.targets.iter().any(|t| t.name == s.target) {
        return Err(CrossEntropyError::UnknownTarget {
            target: s.target.clone(),
            declared: model.targets.iter().map(|t| t.name.clone()).collect(),
        });
    }
    if !(s.t_max.is_finite() && s.t_max > 0.0) {
        return Err(setting(
            "t_max",
            format!("a positive finite horizon, got {}", s.t_max),
        ));
    }
    if s.nb_runs == 0 {
        return Err(setting("nb_runs", "at least one final replica".to_owned()));
    }
    raichu_rng::final_stream(s.nb_runs - 1).map_err(|e| setting("nb_runs", e.to_string()))?;
    if s.fit {
        if s.pilot_runs == 0 {
            return Err(setting(
                "pilot_runs",
                "at least one pilot replica".to_owned(),
            ));
        }
        if s.max_iterations == 0 {
            return Err(setting(
                "max_iterations",
                "at least one pilot iteration".to_owned(),
            ));
        }
        raichu_rng::pilot_stream(s.max_iterations - 1, s.pilot_runs - 1, s.pilot_runs).map_err(
            |e| match e {
                raichu_rng::StreamError::PilotIterationOutOfRange { .. } => {
                    setting("max_iterations", e.to_string())
                }
                other => setting("pilot_runs", other.to_string()),
            },
        )?;
    }
    if !(s.smoothing > 0.0 && s.smoothing <= 1.0) {
        return Err(setting(
            "smoothing",
            format!("inside (0, 1], got {}", s.smoothing),
        ));
    }
    if !(s.tolerance.is_finite() && s.tolerance >= 0.0) {
        return Err(setting(
            "tolerance",
            format!("a non-negative number, got {}", s.tolerance),
        ));
    }
    if !is_valid_level(s.confidence) {
        return Err(setting(
            "confidence",
            format!("a probability strictly inside (0, 1), got {}", s.confidence),
        ));
    }
    if !(s.factor_min.is_finite() && s.factor_min > 0.0) {
        return Err(setting(
            "factor_min",
            format!("a positive finite factor, got {}", s.factor_min),
        ));
    }
    if !(s.factor_max.is_finite() && s.factor_max >= s.factor_min) {
        return Err(setting(
            "factor_max",
            format!(
                "a finite factor at least factor_min ({}), got {}",
                s.factor_min, s.factor_max
            ),
        ));
    }
    if !(s.initial_factor >= s.factor_min && s.initial_factor <= s.factor_max) {
        return Err(setting(
            "initial_factor",
            format!(
                "inside [factor_min, factor_max] = [{}, {}], got {}",
                s.factor_min, s.factor_max, s.initial_factor
            ),
        ));
    }
    if !(s.escalation_ratio.is_finite() && s.escalation_ratio > 1.0) {
        return Err(setting(
            "escalation_ratio",
            format!("a finite ratio above 1, got {}", s.escalation_ratio),
        ));
    }
    if !(s.min_effective_sample_size.is_finite() && s.min_effective_sample_size >= 1.0) {
        return Err(setting(
            "min_effective_sample_size",
            format!(
                "a finite effective sample size of at least 1, got {}",
                s.min_effective_sample_size
            ),
        ));
    }
    Ok(())
}

/// Effective sample size `(Σw)² / Σw²` of a stage's hit weights, computed on
/// weights relative to the largest (the ratio is scale-free), 0 with no hit.
fn hit_effective_sample_size(outcomes: &[ReplicaOutcome]) -> f64 {
    let shift = outcomes
        .iter()
        .filter(|o| o.hit)
        .map(|o| o.log_weight)
        .fold(f64::NEG_INFINITY, f64::max);
    let (sum, sum_sq) = outcomes
        .iter()
        .filter(|o| o.hit)
        .map(|o| (o.log_weight - shift).exp())
        .fold((0.0_f64, 0.0_f64), |(s, q), w| (s + w, q + w * w));
    if sum_sq > 0.0 {
        sum * sum / sum_sq
    } else {
        0.0
    }
}

/// Run one replica on `stream` with the per-family `factors`.
fn replica(
    model: &CompiledModel,
    s: &CrossEntropySettings,
    families: &[Family],
    dense: &[f64],
    factors: &[f64],
    stream: u64,
) -> Result<ReplicaOutcome, EngineError> {
    let config = EngineConfig {
        t_max: s.t_max,
        samples: s.samples.clone(),
        seed: s.seed,
        rng_stream: stream,
        ode: s.ode.clone(),
        stop_at_targets: true,
        sequences: true,
        flow: s.flow.clone(),
        rate_factors: dense.to_vec(),
        ..EngineConfig::default()
    };
    let result = Engine::new(model, config)?.run()?;
    let end = replica_end(result.sequence, s.t_max);
    let stats = result.rate_statistics.unwrap_or_default();
    let mut log_weight = 0.0;
    let per_family = families
        .iter()
        .zip(factors)
        .map(|(family, &f)| {
            let (n, h) = family.indices.iter().fold((0_u64, 0.0_f64), |(n, h), &i| {
                stats
                    .get(i)
                    .map_or((n, h), |x| (n + x.firings, h + x.exposure))
            });
            // log of f^(-n) exp((f - 1) H): the nominal law over the biased one.
            log_weight += -(n as f64) * f.ln() + (f - 1.0) * h;
            (n, h)
        })
        .collect();
    let hit = end.end_cause.as_deref() == Some(s.target.as_str());
    Ok(ReplicaOutcome {
        hit,
        log_weight,
        per_family,
        end,
    })
}

/// A stage: `count` replicas on the streams `stream(i)`, in parallel,
/// returned in replica order.
fn stage(
    model: &CompiledModel,
    s: &CrossEntropySettings,
    families: &[Family],
    factors: &[f64],
    count: u64,
    stream: impl Fn(u64) -> u64 + Sync,
) -> Result<Vec<ReplicaOutcome>, CrossEntropyError> {
    use rayon::prelude::*;
    let mut dense = vec![1.0; model.transitions.len()];
    for (family, &f) in families.iter().zip(factors) {
        for &i in &family.indices {
            dense[i] = f;
        }
    }
    let dense = &dense;
    let compute = || -> Result<Vec<ReplicaOutcome>, EngineError> {
        (0..count)
            .into_par_iter()
            .map(|i| replica(model, s, families, dense, factors, stream(i)))
            .collect()
    };
    Ok(in_pool(s.threads, compute)?)
}

/// The cross-entropy update over the hits of one pilot: per family, the
/// weighted firing count over the weighted exposure, smoothed and clamped;
/// a family with no weighted count or exposure keeps its factor.
fn update(outcomes: &[ReplicaOutcome], factors: &[f64], s: &CrossEntropySettings) -> Vec<f64> {
    // Weights are relative: both sums of a family scale by the same
    // constant, so subtracting the largest log-weight among the hits
    // avoids overflow without changing the ratio.
    let shift = outcomes
        .iter()
        .filter(|o| o.hit)
        .map(|o| o.log_weight)
        .fold(f64::NEG_INFINITY, f64::max);
    // One relative weight per hit, in replica order, shared by every family.
    let hits: Vec<(f64, &[(u64, f64)])> = outcomes
        .iter()
        .filter(|o| o.hit)
        .map(|o| ((o.log_weight - shift).exp(), o.per_family.as_slice()))
        .collect();
    factors
        .iter()
        .enumerate()
        .map(|(k, &old)| {
            let (num, den) =
                hits.iter()
                    .fold((0.0_f64, 0.0_f64), |(num, den), &(w, per_family)| {
                        let (n, h) = per_family[k];
                        (num + w * n as f64, den + w * h)
                    });
            if !(num > 0.0 && den > 0.0 && (num / den).is_finite()) {
                return old;
            }
            let fitted = s.smoothing * (num / den) + (1.0 - s.smoothing) * old;
            fitted.clamp(s.factor_min, s.factor_max)
        })
        .collect()
}

/// Estimate the probability that `settings.target` is the first target
/// reached by `settings.t_max`, by a biased campaign whose factors are
/// fitted by cross-entropy (see the module docs).
///
/// # Errors
/// [`CrossEntropyError`]: an unknown target, an invalid setting or
/// override, an engine refusal, or no hit ([`CrossEntropyError::NoHit`]).
pub fn run_cross_entropy(
    model: &CompiledModel,
    settings: &CrossEntropySettings,
) -> Result<CrossEntropyEstimate, CrossEntropyError> {
    let s = settings;
    validate(model, s)?;
    let families = cross_entropy_families(model, &s.families)?;
    let mut factors: Vec<f64> = families
        .iter()
        .map(|f| if f.repair { 1.0 } else { s.initial_factor })
        .collect();

    let mut history = Vec::new();
    let mut converged = true;
    if s.fit && !families.is_empty() {
        converged = false;
        let mut ever_hit = false;
        for k in 0..s.max_iterations {
            let outcomes = stage(model, s, &families, &factors, s.pilot_runs, |i| {
                // Validated above for every iteration below the cap.
                raichu_rng::pilot_stream(k, i, s.pilot_runs).unwrap_or(u64::MAX)
            })?;
            let hits = outcomes.iter().filter(|o| o.hit).count() as u64;
            if hits == 0 {
                history.push(PilotIteration {
                    factors: factors.clone(),
                    hits,
                    escalated: true,
                    effective_sample_size: 0.0,
                });
                for (f, family) in factors.iter_mut().zip(&families) {
                    if !family.repair {
                        *f = (*f * s.escalation_ratio).min(s.factor_max);
                    }
                }
                continue;
            }
            ever_hit = true;
            let pilot_ess = hit_effective_sample_size(&outcomes);
            // A fit is confirmed only by a pilot informative enough to
            // judge it: one whose hits are carried by a few replicas returns
            // those replicas' own rates, so its factors barely move whatever
            // the truth, and never on the pilot right after an escalation.
            let informative = pilot_ess >= s.min_effective_sample_size
                && history.last().is_none_or(|previous| !previous.escalated);
            history.push(PilotIteration {
                factors: factors.clone(),
                hits,
                escalated: false,
                effective_sample_size: pilot_ess,
            });
            let next = update(&outcomes, &factors, s);
            let moved = next
                .iter()
                .zip(&factors)
                .map(|(new, old)| ((new - old) / old).abs())
                .fold(0.0_f64, f64::max);
            factors = next;
            if moved <= s.tolerance && informative {
                converged = true;
                break;
            }
        }
        if !ever_hit {
            return Err(CrossEntropyError::NoHit {
                target: s.target.clone(),
                stage: "pilot",
                replicas: s.pilot_runs,
                factors,
            });
        }
    }

    // Validated above: every final replica has its stream, equal to its
    // index as in a plain campaign.
    let outcomes = stage(model, s, &families, &factors, s.nb_runs, |i| {
        raichu_rng::final_stream(i).unwrap_or(i)
    })?;
    let reached = outcomes.iter().filter(|o| o.hit).count() as u64;
    if reached == 0 {
        return Err(CrossEntropyError::NoHit {
            target: s.target.clone(),
            stage: "final",
            replicas: s.nb_runs,
            factors,
        });
    }
    let values: Vec<f64> = outcomes
        .iter()
        .map(|o| if o.hit { o.log_weight.exp() } else { 0.0 })
        .collect();
    let estimate = weighted_interval(&values, s.confidence);
    // An estimate of 0 with hits (every hit weight underflowing) has an
    // effective sample size of 0, so it is flagged here too.
    let estimate_inconclusive = estimate.effective_sample_size < s.min_effective_sample_size;
    Ok(CrossEntropyEstimate {
        target: s.target.clone(),
        estimate,
        reached,
        replicas: s.nb_runs,
        families: families
            .iter()
            .zip(&factors)
            .map(|(family, &factor)| FittedFamily {
                label: family.label.clone(),
                transitions: family.transitions.clone(),
                repair: family.repair,
                factor,
            })
            .collect(),
        history,
        converged,
        estimate_inconclusive,
        ends: outcomes.into_iter().map(|o| o.end).collect(),
    })
}
