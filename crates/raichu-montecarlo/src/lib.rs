//! # raichu-montecarlo: parallel replica driver
//!
//! Runs `nb_runs` independent trajectories of a compiled model and
//! estimates indicator statistics at a sampling schedule.
//!
//! Reproducibility contract:
//!
//! - replica `r` uses the RNG substream `r` of the master seed
//!   (`raichu-rng` policy): replicas are independent by construction
//!   and each one replays bit-identically;
//! - replica results are reduced by a **serial, index-ordered fold**:
//!   floating-point addition is not associative, so this is what makes
//!   1-thread and N-thread runs produce *identical bytes*, not just
//!   statistically equal numbers;
//! - the replicas run in parallel **chunks** whose samples are folded,
//!   in replica order, as each chunk completes: a campaign's memory does
//!   not grow with its number of replicas (quantiles excepted, which need
//!   the whole column), and the fold is the same sequence of additions as
//!   over the whole campaign at once;
//! - `rayon` only parallelises the embarrassingly parallel trajectory
//!   loop: the single-trajectory engine stays single-threaded.
//!
//! Estimators per indicator and schedule instant: mean and sample
//! standard deviation of the sampled value, plus the **sojourn time**
//! (time-integral of the indicator value up to the instant: for 0/1
//! state indicators this is the classic cumulated-sojourn estimator,
//! the sojourn-time measure).
//!
//! Every one of those means also carries a **confidence interval** at
//! the level the study declares ([`McConfig::confidence`]): see
//! [`confidence`] for what is computed and why.

pub mod confidence;
pub mod cross_entropy;
pub mod splitting;
pub use splitting::*;

pub use confidence::{
    batch_interval, constant_sample_bounds, is_valid_level, normal_bounds, normal_quantile,
    student_critical, unobserved_frequency_bound, weighted_interval, wilson_bounds, z_of,
    BatchInterval, ConfidenceInterval, Departure, IntervalMethod, WeightedInterval,
    DEFAULT_CONFIDENCE,
};
pub use cross_entropy::{
    cross_entropy_families, run_cross_entropy, CrossEntropyError, CrossEntropyEstimate,
    CrossEntropySettings, Family, FittedFamily, PilotIteration, DEFAULT_CE_ESCALATION_RATIO,
    DEFAULT_CE_FACTOR_MAX, DEFAULT_CE_FACTOR_MIN, DEFAULT_CE_INITIAL_FACTOR,
    DEFAULT_CE_MAX_ITERATIONS, DEFAULT_CE_MIN_EFFECTIVE_SAMPLE_SIZE, DEFAULT_CE_NB_RUNS,
    DEFAULT_CE_PILOT_RUNS, DEFAULT_CE_SMOOTHING, DEFAULT_CE_TOLERANCE,
};

use raichu_analysis::{importance, target_events, ImportanceAnalysis};
use raichu_core::{
    CIndicator, CIndicatorTarget, CoSimulationHost, CompiledModel, Engine, EngineConfig,
    EngineError, FlowConfig, FmuProvenance, IndicatorSeries, PreparedCoSimulation, Sequence,
    SolverParams,
};
use raichu_expr::Value;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

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
    /// byte-identical whatever the value: see the crate docs.
    pub threads: Option<usize>,
    /// Quantile orders to estimate (e.g. `[0.25, 0.75]`), on both the
    /// sampled value and the cumulated sojourn, nearest-rank across
    /// replicas (M4; quantile stats, e.g. P25/P75).
    pub quantiles: Vec<f64>,
    /// Confidence level of the intervals reported on every estimator,
    /// strictly inside `(0, 1)`.
    ///
    /// A **study parameter**: a campaign that has to answer to a
    /// regulator at 99 % states 0.99 here and the whole result is
    /// reported at 99 %, level included ([`ConfidenceInterval::level`]).
    /// [`DEFAULT_CONFIDENCE`] is the conventional value, offered as a
    /// starting point and never assumed by a reader. A level outside
    /// `(0, 1)` is refused before the campaign runs.
    pub confidence: f64,
    /// Numerical parameters of the ODE backend for every replica
    /// (engine defaults unless overridden: the knob of the
    /// tolerance-parity experiments; recorded as provenance upstream).
    pub ode: SolverParams,
    /// Early-stop each trajectory at the first sequence target (feared
    /// event) and hold the frozen state through the remaining sample
    /// instants: the latch semantics of target-stopped studies (the
    /// reference regime of recorded sequence campaigns). Ignored when the
    /// model declares no target.
    pub stop_at_targets: bool,
    /// Convergence policy of the continuous flow resolution, applied to
    /// every replica ([`FlowConfig::default`] is the documented policy).
    /// It is per-run configuration and not per-replica state: the
    /// resolution carries nothing across a segment, so sharing one policy
    /// across replicas keeps the estimates invariant in the thread count.
    pub flow: FlowConfig,
}

/// A quantile series over the schedule instants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantileSeries {
    /// Quantile order in (0, 1).
    pub q: f64,
    /// Nearest-rank quantile at each schedule instant.
    pub values: Vec<f64>,
}

/// The smallest and the largest value a measure took across the replicas,
/// at each schedule instant: what a study asks with the `min` and `max`
/// statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Extremes {
    /// Smallest value over the replicas at each instant.
    pub min: Vec<f64>,
    /// Largest value over the replicas at each instant.
    pub max: Vec<f64>,
}

impl Extremes {
    fn empty(n_instants: usize) -> Self {
        Self {
            min: vec![f64::INFINITY; n_instants],
            max: vec![f64::NEG_INFINITY; n_instants],
        }
    }

    fn fold(&mut self, k: usize, value: f64) {
        self.min[k] = self.min[k].min(value);
        self.max[k] = self.max[k].max(value);
    }
}

/// Estimates of one indicator over the schedule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndicatorEstimate {
    /// Indicator name.
    pub name: String,
    /// Schedule instants.
    pub instants: Vec<f64>,
    /// Mean of the sampled value at each instant.
    pub mean: Vec<f64>,
    /// Sample standard deviation (ddof = 1) at each instant.
    pub std: Vec<f64>,
    /// Confidence interval on [`Self::mean`], at the level the study
    /// declared. Wilson when the indicator is a probability by
    /// declaration (a state, or a boolean attribute), normal otherwise.
    /// Where every replica returned the same value it is the frequency
    /// bound instead of a point: see [`confidence`].
    pub ci: ConfidenceInterval,
    /// Mean cumulated sojourn (time-integral of the value) up to each
    /// instant.
    pub sojourn_mean: Vec<f64>,
    /// Sample standard deviation of the cumulated sojourn.
    pub sojourn_std: Vec<f64>,
    /// Confidence interval on [`Self::sojourn_mean`].
    pub sojourn_ci: ConfidenceInterval,
    /// Mean number of occurrences (state entries / rising edges) up to each
    /// instant: the RAMS `nb-occurrences` measure.
    pub nb_occurrences_mean: Vec<f64>,
    /// Sample standard deviation of the occurrence count.
    pub nb_occurrences_std: Vec<f64>,
    /// Confidence interval on [`Self::nb_occurrences_mean`].
    pub nb_occurrences_ci: ConfidenceInterval,
    /// Probability that the indicator has been active at least once by each
    /// instant: the RAMS `had-value` measure. Per trajectory it is 1 from
    /// the first occurrence on (an active initial value included) and stays
    /// 1 after the indicator falls back, so it is the first-entry
    /// distribution, not the value's mean.
    pub reached_mean: Vec<f64>,
    /// Sample standard deviation of the reached indicator.
    pub reached_std: Vec<f64>,
    /// Confidence interval on [`Self::reached_mean`]: a proportion, so
    /// Wilson whatever the indicator's own kind.
    pub reached_ci: ConfidenceInterval,
    /// Extremes of the sampled value over the replicas.
    pub extremes: Extremes,
    /// Extremes of the cumulated sojourn.
    pub sojourn_extremes: Extremes,
    /// Extremes of the occurrence count.
    pub nb_occurrences_extremes: Extremes,
    /// Extremes of the reached indicator (0 and 1, or one of them when
    /// every replica agrees).
    pub reached_extremes: Extremes,
    // The departure measures below default when absent, so a result
    // document written before they existed (a Monte-Carlo detail of a
    // `raichu.quantification` envelope, for instance) still reads, with
    // empty series that no instant can be read off.
    /// Mean number of **departures from zero** up to each instant: moves
    /// of the value from exactly `0` to any non-zero value, whatever its
    /// sign, the initial value never counted. This is the reference
    /// engine's `nb_visits` computation, the one a cod3s/muscadet
    /// `nb-occurrences` indicator is defined against. It differs from
    /// [`Self::nb_occurrences_mean`] on an active initial value (not
    /// counted here), on a negative value (a departure here, not a rising
    /// edge there) and on a move between two non-zero values (counted by
    /// neither).
    #[serde(default)]
    pub zero_departures_mean: Vec<f64>,
    /// Sample standard deviation of the departure count.
    #[serde(default)]
    pub zero_departures_std: Vec<f64>,
    /// Confidence interval on [`Self::zero_departures_mean`], built as the
    /// one on [`Self::nb_occurrences_mean`] (a count).
    #[serde(default = "absent_interval")]
    pub zero_departures_ci: ConfidenceInterval,
    /// Extremes of the departure count.
    #[serde(default = "absent_extremes")]
    pub zero_departures_extremes: Extremes,
    /// Probability that the value has been **non-zero** at least once by
    /// each instant, the initial value included: the reference engine's
    /// `realized` computation, the one a cod3s/muscadet `had_value`
    /// indicator is defined against. It differs from
    /// [`Self::reached_mean`] on a value that is only ever negative
    /// (non-zero, so reached here; never above zero, so not reached
    /// there).
    #[serde(default)]
    pub nonzero_reached_mean: Vec<f64>,
    /// Sample standard deviation of the non-zero-reached indicator.
    #[serde(default)]
    pub nonzero_reached_std: Vec<f64>,
    /// Confidence interval on [`Self::nonzero_reached_mean`]: a
    /// proportion, so Wilson whatever the indicator's own kind.
    #[serde(default = "absent_interval")]
    pub nonzero_reached_ci: ConfidenceInterval,
    /// Extremes of the non-zero-reached indicator.
    #[serde(default = "absent_extremes")]
    pub nonzero_reached_extremes: Extremes,
    /// Requested quantiles of the sampled value.
    pub quantiles: Vec<QuantileSeries>,
    /// Requested quantiles of the cumulated sojourn.
    pub sojourn_quantiles: Vec<QuantileSeries>,
}

/// The interval a document written before the departure measures existed
/// reads back with: no instant and no construction ([`IntervalMethod::Undefined`]),
/// so it can never be mistaken for a computed one. Its level is the
/// conventional one only because a level has to be a probability.
fn absent_interval() -> ConfidenceInterval {
    ConfidenceInterval {
        level: DEFAULT_CONFIDENCE,
        method: IntervalMethod::Undefined,
        low: Vec::new(),
        high: Vec::new(),
        constant_sample: Vec::new(),
    }
}

/// The extremes a document written before the departure measures existed
/// reads back with: no instant.
fn absent_extremes() -> Extremes {
    Extremes::empty(0)
}

/// Full Monte-Carlo result with provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McEstimates {
    /// Per-indicator estimates.
    pub indicators: Vec<IndicatorEstimate>,
    /// Number of replicas.
    pub nb_runs: u64,
    /// Master seed.
    pub seed: u64,
    /// Confidence level every interval of this result was computed at:
    /// provenance, on the same footing as the seed and the replica
    /// count. A result states its precision *and* the level that
    /// precision is claimed at.
    pub confidence: f64,
    /// Engine version.
    pub engine_version: String,
    /// Imported co-simulation units, omitted for native-only campaigns.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fmu_units: Vec<FmuProvenance>,
    /// Unit that required serial replica execution, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serial_fallback_unit: Option<String>,
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
/// rising edges in the change-point series: a value going from ≤ 0 to > 0,
/// including an active initial value. This is the RAMS `nb-occurrences`
/// measure (how many times the feared event happened by `instant`).
fn nb_occurrences_at(points: &[(f64, Value)], instant: f64) -> f64 {
    let mut count = 0.0;
    let mut prev = 0.0;
    for (t_start, value) in points {
        // Inclusive bound: an entry AT the sample instant counts, matching
        // the sampled value at that instant (which reflects the post-event
        // state): the two measures stay mutually consistent.
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

/// The value a change-point series **starts** from: the last entry dated
/// at the series' first date.
///
/// The initial propagation may record more than one entry at that date
/// (the declared value, then what the start-up fixpoint settles on); the
/// trajectory starts from the settled one, so the moves between them are
/// part of the initial value and not departures from it. `None` for an
/// empty series.
fn initial_index(points: &[(f64, Value)]) -> Option<usize> {
    let first = points.first()?.0;
    Some(points.iter().take_while(|(t, _)| *t == first).count() - 1)
}

/// Number of **departures from zero** up to `instant`: moves of the value
/// from exactly `0` to any non-zero value, sign ignored, the initial value
/// (see [`initial_index`]) never counted. The reference engine's
/// `nb_visits` computation, measured on PyCATSHOO 1.3.8.0 (2026-10-02).
/// The bound is inclusive, as in [`nb_occurrences_at`].
fn zero_departures_at(points: &[(f64, Value)], instant: f64) -> f64 {
    let Some(start) = initial_index(points) else {
        return 0.0;
    };
    if points[start].0 > instant {
        return 0.0;
    }
    let mut count = 0.0;
    let mut prev = value_as_f64(points[start].1);
    for (t_start, value) in &points[start + 1..] {
        if *t_start > instant {
            break;
        }
        let cur = value_as_f64(*value);
        if prev == 0.0 && cur != 0.0 {
            count += 1.0;
        }
        prev = cur;
    }
    count
}

/// Whether the value has been **non-zero** at or before `instant`, the
/// initial value (see [`initial_index`]) included: `1.0` or `0.0`. The
/// reference engine's `realized` computation, measured on PyCATSHOO
/// 1.3.8.0 (2026-10-02). The bound is inclusive, as in
/// [`nb_occurrences_at`].
fn nonzero_reached_at(points: &[(f64, Value)], instant: f64) -> f64 {
    let Some(start) = initial_index(points) else {
        return 0.0;
    };
    let reached = points[start..]
        .iter()
        .take_while(|(t_start, _)| *t_start <= instant)
        .any(|(_, value)| value_as_f64(*value) != 0.0);
    if reached {
        1.0
    } else {
        0.0
    }
}

/// What one replica records of one indicator at one schedule instant.
#[derive(Debug, Clone, Copy)]
struct Sample {
    /// The sampled value.
    value: f64,
    /// Its time-integral up to the instant.
    sojourn: f64,
    /// Rising edges up to the instant ([`nb_occurrences_at`]).
    nb_occurrences: f64,
    /// Departures from zero up to the instant ([`zero_departures_at`]).
    zero_departures: f64,
    /// Non-zero at or before the instant ([`nonzero_reached_at`]).
    nonzero_reached: f64,
}

/// One replica's samples, flat: indicator `i` at instant `k` is
/// `samples[i * per_indicator + k]`. One allocation per replica rather
/// than one per indicator: a chunk holds thousands of replicas, and a model
/// can declare hundreds of indicators.
struct ReplicaSamples {
    samples: Vec<Sample>,
    /// The instants recorded per indicator; the engine records every
    /// indicator at every instant it reaches, so they all share it.
    per_indicator: usize,
}

/// Whether the sampled value of this indicator is a `{0, 1}` draw **by
/// declaration**, which makes its mean a probability and its interval a
/// binomial one.
///
/// Read off the model, never off the observed sample: an integer
/// indicator that happened to stay in `{0, 1}` over one campaign is not
/// a probability, and calling it one would make the reported precision
/// depend on the draw.
fn is_declared_binary(model: &CompiledModel, indicator: &CIndicator) -> bool {
    match indicator.target {
        // 1.0 in the state, 0.0 out of it: binary by construction.
        CIndicatorTarget::State(_, _) => true,
        // An attribute is binary exactly when it is declared `bool`;
        // the initial value carries that declared kind (validation
        // refuses an initial value of another kind).
        CIndicatorTarget::Var(var) => model
            .var_init
            .get(var)
            .is_some_and(|value| matches!(value, Value::Bool(_))),
        // The truth of a threshold, whatever the kind of what it
        // compares: binary by declaration, and the reason the variant
        // exists -- its mean is a probability and its sojourn a duration,
        // where the raw attribute's sojourn is an area.
        CIndicatorTarget::Predicate(_, _, _) => true,
    }
}

/// One replica's samples, plus how it ended when `record_end` is set.
///
/// `record_end` switches the engine's sequence record on, which is the
/// only place a finished run states its end cause; the record observes
/// the trajectory and never changes it, so the samples are the same with
/// or without it.
fn replica_samples(
    model: &CompiledModel,
    config: &McConfig,
    replica: u64,
    record_end: bool,
    prepared: Option<&PreparedCoSimulation>,
    reusable_host: Option<&mut CoSimulationHost>,
) -> Result<(ReplicaSamples, Option<ReplicaEnd>), EngineError> {
    let engine_config = EngineConfig {
        t_max: config.t_max,
        samples: config.samples.clone(),
        seed: config.seed,
        rng_stream: replica,
        ode: config.ode.clone(),
        // Early stop only: this driver reads `samples` / `indicators` and
        // never the per-trajectory trace, so it must not pay for recording it
        // unless the caller asked how each replica ended.
        stop_at_targets: config.stop_at_targets,
        sequences: record_end,
        flow: config.flow.clone(),
        allow_fmu_import: prepared.is_some(),
        ..EngineConfig::default()
    };
    let mut owned_host = prepared.map(PreparedCoSimulation::spawn_host);
    let result = match reusable_host.or(owned_host.as_mut()) {
        Some(host) => Engine::new_with_host(model, engine_config, host)?.run()?,
        _ => Engine::new(model, engine_config)?.run()?,
    };
    let end = result
        .sequence
        .map(|sequence| replica_end(Some(sequence), config.t_max));
    let recorded = |sampled: &IndicatorSeries| sampled.points.len().min(config.samples.len());
    let per_indicator = result.samples.first().map_or(0, recorded);
    // The flat layout needs every indicator to carry as many samples as the
    // first: the engine records them all at each instant it reaches, and a
    // series that did not would be folded into its neighbour's sums.
    if let Some(i) = result
        .samples
        .iter()
        .position(|sampled| recorded(sampled) != per_indicator)
    {
        return Err(EngineError::TypeError {
            time: config.t_max,
            detail: format!(
                "replica {replica} recorded {} samples of `{}` where the first indicator has \
                 {per_indicator}",
                recorded(&result.samples[i]),
                model
                    .indicators
                    .get(i)
                    .map_or("an unnamed indicator", |indicator| indicator.name.as_str())
            ),
        });
    }
    let mut samples = Vec::with_capacity(result.samples.len() * per_indicator);
    for (sampled, change_points) in result.samples.iter().zip(&result.indicators) {
        samples.extend(
            config
                .samples
                .iter()
                .zip(&sampled.points)
                .map(|(instant, (_, value))| Sample {
                    value: value_as_f64(*value),
                    sojourn: sojourn_at(&change_points.points, *instant),
                    nb_occurrences: nb_occurrences_at(&change_points.points, *instant),
                    zero_departures: zero_departures_at(&change_points.points, *instant),
                    nonzero_reached: nonzero_reached_at(&change_points.points, *instant),
                }),
        );
    }
    Ok((
        ReplicaSamples {
            samples,
            per_indicator,
        },
        end,
    ))
}

/// Run the Monte-Carlo estimation.
///
/// Replicas run in parallel; the reduction is a serial fold in replica
/// order, so the estimates are bit-identical for any thread count.
pub fn run(model: &CompiledModel, config: &McConfig) -> Result<McEstimates, EngineError> {
    run_internal(model, config, None, false)
}

/// Run an explicitly authorized campaign with one isolated FMU instance per replica.
///
/// A unit that forbids multiple instances makes the campaign serial unless
/// `require_parallel` is true, in which case it is rejected before loading
/// native code. `base_dir` resolves relative paths declared by the model.
///
/// # Errors
/// Returns a typed permission, capability, FMU or engine error.
pub fn run_with_fmu(
    model: &CompiledModel,
    config: &McConfig,
    base_dir: &Path,
    allow_fmu_import: bool,
    require_parallel: bool,
) -> Result<McEstimates, EngineError> {
    if !allow_fmu_import {
        return run(model, config);
    }
    run_internal(model, config, Some(base_dir), require_parallel)
}

fn run_internal(
    model: &CompiledModel,
    config: &McConfig,
    base_dir: Option<&Path>,
    require_parallel: bool,
) -> Result<McEstimates, EngineError> {
    check_level(config)?;

    let PreparedCampaign {
        serial_fallback_unit,
        fmu_units,
        prepared,
    } = prepare_campaign(model, base_dir, require_parallel)?;

    let mut reducer = Reducer::new(model, config);
    stream_replicas(
        model,
        config,
        prepared.as_ref(),
        serial_fallback_unit.is_some(),
        false,
        |samples, _| reducer.push(&samples),
    )?;
    Ok(reducer.finish(model, config, fmu_units, serial_fallback_unit))
}

/// Simulate every replica of a campaign and hand each one to `sink`, in
/// replica order: in parallel chunks of [`chunk_len`] replicas, or one at a
/// time when an FMU that allows a single instance makes the campaign
/// serial. Only one chunk of samples is alive at a time.
fn stream_replicas<F>(
    model: &CompiledModel,
    config: &McConfig,
    prepared: Option<&PreparedCoSimulation>,
    serial: bool,
    record_end: bool,
    mut sink: F,
) -> Result<(), EngineError>
where
    F: FnMut(ReplicaSamples, Option<ReplicaEnd>) + Send,
{
    use rayon::prelude::*;

    if serial {
        let mut host = prepared.map(PreparedCoSimulation::spawn_host);
        for replica in 0..config.nb_runs {
            let (samples, end) =
                replica_samples(model, config, replica, record_end, prepared, host.as_mut())?;
            sink(samples, end);
        }
        return Ok(());
    }
    let chunk = chunk_len(model.indicators.len() * config.samples.len());
    in_pool(config.threads, || {
        let mut first = 0;
        while first < config.nb_runs {
            let last = (first + chunk).min(config.nb_runs);
            let chunk = (first..last)
                .into_par_iter()
                .map(|replica| replica_samples(model, config, replica, record_end, prepared, None))
                .collect::<Result<Vec<_>, _>>()?;
            for (samples, end) in chunk {
                sink(samples, end);
            }
            first = last;
        }
        Ok(())
    })
}

struct PreparedCampaign {
    serial_fallback_unit: Option<String>,
    fmu_units: Vec<FmuProvenance>,
    prepared: Option<PreparedCoSimulation>,
}

fn prepare_campaign(
    model: &CompiledModel,
    base_dir: Option<&Path>,
    require_parallel: bool,
) -> Result<PreparedCampaign, EngineError> {
    let Some(base_dir) = base_dir else {
        if let Some(unit) = model.fmu_units.first() {
            return Err(EngineError::FmuPermission {
                unit: unit.name.clone(),
            });
        }
        return Ok(PreparedCampaign {
            serial_fallback_unit: None,
            fmu_units: Vec::new(),
            prepared: None,
        });
    };
    if model.fmu_units.is_empty() {
        return Ok(PreparedCampaign {
            serial_fallback_unit: None,
            fmu_units: Vec::new(),
            prepared: None,
        });
    }
    let prepared = PreparedCoSimulation::prepare_authorized(model, base_dir, true)?;
    let single_instance = prepared.single_instance_unit();
    if require_parallel {
        if let Some(unit) = single_instance {
            return Err(EngineError::FmuCapability {
                unit: unit.to_owned(),
                capability: "multiple instances per process, required by parallel execution",
            });
        }
    }
    Ok(PreparedCampaign {
        serial_fallback_unit: single_instance.map(str::to_owned),
        fmu_units: prepared.provenance(),
        prepared: Some(prepared),
    })
}

/// Run `compute` on the default rayon pool, or on a pool of `threads`
/// workers when one is asked for.
fn in_pool<T: Send, E: Send + From<EngineError>>(
    threads: Option<usize>,
    compute: impl FnOnce() -> Result<T, E> + Send,
) -> Result<T, E> {
    match threads {
        None => compute(),
        Some(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| EngineError::TypeError {
                time: 0.0,
                detail: format!("thread-pool construction failed: {e}"),
            })?
            .install(compute),
    }
}

/// Refuse a confidence level that cannot produce an interval, before the
/// replicas: it must not cost a campaign first.
fn check_level(config: &McConfig) -> Result<(), EngineError> {
    if is_valid_level(config.confidence) {
        return Ok(());
    }
    Err(EngineError::InvalidStudyParameter {
        parameter: "confidence".to_owned(),
        detail: format!(
            "a confidence level is a probability strictly inside (0, 1), got {}",
            config.confidence
        ),
    })
}

/// How one replica ended: the first declared target it reached, or none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicaEnd {
    /// Name of the first target (feared event) the replica reached,
    /// `None` when it reached none by the horizon.
    pub end_cause: Option<String>,
    /// Instant the target was reached, or the horizon when none was.
    pub end_time: f64,
}

/// How a replica ended, read off its sequence record: the first target
/// it reached, or none by `horizon` when it recorded no end.
pub(crate) fn replica_end(sequence: Option<Sequence>, horizon: f64) -> ReplicaEnd {
    sequence.map_or(
        ReplicaEnd {
            end_cause: None,
            end_time: horizon,
        },
        |sequence| ReplicaEnd {
            end_cause: sequence.end_cause,
            end_time: sequence.end_time,
        },
    )
}

/// A stop-at-targets campaign: its estimates, and how each replica ended.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetCampaign {
    /// The estimates, equal to [`run`] with
    /// [`McConfig::stop_at_targets`] on and the same configuration
    /// otherwise.
    pub estimates: McEstimates,
    /// One entry per replica, in replica order.
    pub ends: Vec<ReplicaEnd>,
}

impl TargetCampaign {
    /// Number of replicas whose first target reached is `target`.
    #[must_use]
    pub fn count_reached(&self, target: &str) -> u64 {
        self.ends
            .iter()
            .filter(|end| end.end_cause.as_deref() == Some(target))
            .count() as u64
    }
}

/// Run one **stop-at-targets** campaign and report, beside its estimates,
/// how each replica ended.
///
/// Every trajectory stops at the first declared target it reaches, as
/// [`run`] does with [`McConfig::stop_at_targets`] on, whatever that
/// field says: the estimates are byte-identical to that call, and the end
/// causes are what counting "which feared event came first, by the
/// horizon" needs, without inferring it from an indicator or running a
/// second campaign. A model declaring no target yields no end cause.
///
/// # Errors
/// As [`run`].
pub fn run_to_targets(
    model: &CompiledModel,
    config: &McConfig,
) -> Result<TargetCampaign, EngineError> {
    run_to_targets_internal(model, config, None, false)
}

/// Run a target-stopped campaign with explicitly authorized FMU imports.
///
/// # Errors
/// Returns a typed permission, capability, FMU or engine error.
pub fn run_to_targets_with_fmu(
    model: &CompiledModel,
    config: &McConfig,
    base_dir: &Path,
    allow_fmu_import: bool,
    require_parallel: bool,
) -> Result<TargetCampaign, EngineError> {
    if !allow_fmu_import {
        return run_to_targets(model, config);
    }
    run_to_targets_internal(model, config, Some(base_dir), require_parallel)
}

fn run_to_targets_internal(
    model: &CompiledModel,
    config: &McConfig,
    base_dir: Option<&Path>,
    require_parallel: bool,
) -> Result<TargetCampaign, EngineError> {
    let mut ends = Vec::new();
    let estimates = targets_campaign(model, config, base_dir, require_parallel, |end| {
        ends.push(end);
    })?;
    Ok(TargetCampaign { estimates, ends })
}

/// A stop-at-targets campaign reduced to counts: its estimates, and how
/// many replicas ended at each target.
///
/// The same campaign as [`run_to_targets`], byte for byte, without the
/// one [`ReplicaEnd`] per replica: what a probability of reaching a target
/// needs is a count, and a count keeps the memory of a campaign
/// independent of its number of replicas.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetCounts {
    /// The estimates, equal to those of [`run_to_targets`].
    pub estimates: McEstimates,
    /// Number of replicas whose first target reached is the key; a target
    /// no replica reached is absent.
    pub reached: BTreeMap<String, u64>,
}

impl TargetCounts {
    /// Number of replicas whose first target reached is `target`.
    #[must_use]
    pub fn count_reached(&self, target: &str) -> u64 {
        self.reached.get(target).copied().unwrap_or(0)
    }
}

/// Run one **stop-at-targets** campaign and count, beside its estimates,
/// the replicas that ended at each target: [`run_to_targets`] without the
/// per-replica ends, for campaigns too large to keep one per replica.
///
/// # Errors
/// As [`run`].
pub fn count_to_targets(
    model: &CompiledModel,
    config: &McConfig,
) -> Result<TargetCounts, EngineError> {
    count_to_targets_internal(model, config, None, false)
}

/// Count a target-stopped campaign with explicitly authorized FMU imports.
///
/// # Errors
/// Returns a typed permission, capability, FMU or engine error.
pub fn count_to_targets_with_fmu(
    model: &CompiledModel,
    config: &McConfig,
    base_dir: &Path,
    allow_fmu_import: bool,
    require_parallel: bool,
) -> Result<TargetCounts, EngineError> {
    if !allow_fmu_import {
        return count_to_targets(model, config);
    }
    count_to_targets_internal(model, config, Some(base_dir), require_parallel)
}

fn count_to_targets_internal(
    model: &CompiledModel,
    config: &McConfig,
    base_dir: Option<&Path>,
    require_parallel: bool,
) -> Result<TargetCounts, EngineError> {
    let mut reached = BTreeMap::new();
    let estimates = targets_campaign(model, config, base_dir, require_parallel, |end| {
        if let Some(target) = end.end_cause {
            *reached.entry(target).or_insert(0) += 1;
        }
    })?;
    Ok(TargetCounts { estimates, reached })
}

/// The campaign behind [`run_to_targets`] and [`count_to_targets`]: every
/// trajectory stops at its first target, and `on_end` receives how each
/// replica ended, in replica order.
fn targets_campaign<F>(
    model: &CompiledModel,
    config: &McConfig,
    base_dir: Option<&Path>,
    require_parallel: bool,
    mut on_end: F,
) -> Result<McEstimates, EngineError>
where
    F: FnMut(ReplicaEnd) + Send,
{
    let config = McConfig {
        stop_at_targets: true,
        ..config.clone()
    };
    check_level(&config)?;
    let PreparedCampaign {
        serial_fallback_unit,
        fmu_units,
        prepared,
    } = prepare_campaign(model, base_dir, require_parallel)?;
    let mut reducer = Reducer::new(model, &config);
    let horizon = config.t_max;
    stream_replicas(
        model,
        &config,
        prepared.as_ref(),
        serial_fallback_unit.is_some(),
        true,
        |samples, end| {
            reducer.push(&samples);
            on_end(end.unwrap_or_else(|| replica_end(None, horizon)));
        },
    )?;
    Ok(reducer.finish(model, &config, fmu_units, serial_fallback_unit))
}

/// The bytes of samples one chunk of replicas may hold before it is folded
/// into the estimates. A campaign holds at most one chunk at a time, so its
/// memory does not grow with `nb_runs`; only the quantile columns, when
/// quantiles are asked, keep one value per replica.
const CHUNK_BYTES: usize = 64 << 20;

/// The most replicas in one chunk, reached by models with few indicators.
const MAX_CHUNK: u64 = 1 << 16;

/// The fewest replicas in one chunk, whatever a replica weighs: enough to
/// keep every thread of a large machine busy between two folds.
const MIN_CHUNK: u64 = 1 << 10;

/// The replicas of one chunk, for replicas recording `samples_per_replica`
/// samples each: as many as [`CHUNK_BYTES`] holds, between [`MIN_CHUNK`]
/// and [`MAX_CHUNK`]. The chunking never changes a result, since the fold
/// takes the replicas in replica order whatever their grouping.
fn chunk_len(samples_per_replica: usize) -> u64 {
    let replica_bytes = samples_per_replica
        .saturating_mul(size_of::<Sample>())
        .max(1);
    let fits = u64::try_from(CHUNK_BYTES / replica_bytes).unwrap_or(MAX_CHUNK);
    fits.clamp(MIN_CHUNK, MAX_CHUNK)
}

/// Running sums of one indicator at one instant, folded in replica order.
#[derive(Clone, Copy, Default)]
struct InstantSums {
    sum: f64,
    sum_sq: f64,
    sj_sum: f64,
    sj_sum_sq: f64,
    oc_sum: f64,
    oc_sum_sq: f64,
    /// A 0/1 draw, so its sum is also the sum of its squares.
    reached_sum: f64,
    zd_sum: f64,
    zd_sum_sq: f64,
    /// A 0/1 draw as well.
    nonzero_sum: f64,
}

/// What the reduction keeps of one indicator while the replicas stream in.
struct IndicatorFold {
    sums: Vec<InstantSums>,
    extremes: Extremes,
    sojourn_extremes: Extremes,
    nb_occurrences_extremes: Extremes,
    reached_extremes: Extremes,
    zero_departures_extremes: Extremes,
    nonzero_reached_extremes: Extremes,
    /// `[instant][replica]` sampled values and sojourns, in replica order,
    /// kept only when quantiles are asked: a quantile needs the whole
    /// column, every other estimator a running sum.
    value_columns: Vec<Vec<f64>>,
    sojourn_columns: Vec<Vec<f64>>,
}

/// The serial, replica-ordered reduction of a campaign's samples into its
/// estimates (the determinism contract of the crate docs), fed one replica
/// at a time.
///
/// Each running sum receives the replicas in replica order, whatever the
/// chunking and the thread count, so it performs the same floating-point
/// additions in the same order as a fold over the whole collected campaign:
/// the estimates are the same bytes, at a memory that no longer depends on
/// the number of replicas.
struct Reducer {
    folds: Vec<IndicatorFold>,
    keep_columns: bool,
    folded: u64,
}

impl Reducer {
    fn new(model: &CompiledModel, config: &McConfig) -> Self {
        let n_instants = config.samples.len();
        let keep_columns = !config.quantiles.is_empty();
        let folds = model
            .indicators
            .iter()
            .map(|_| IndicatorFold {
                sums: vec![InstantSums::default(); n_instants],
                extremes: Extremes::empty(n_instants),
                sojourn_extremes: Extremes::empty(n_instants),
                nb_occurrences_extremes: Extremes::empty(n_instants),
                reached_extremes: Extremes::empty(n_instants),
                zero_departures_extremes: Extremes::empty(n_instants),
                nonzero_reached_extremes: Extremes::empty(n_instants),
                value_columns: vec![Vec::new(); if keep_columns { n_instants } else { 0 }],
                sojourn_columns: vec![Vec::new(); if keep_columns { n_instants } else { 0 }],
            })
            .collect();
        Self {
            folds,
            keep_columns,
            folded: 0,
        }
    }

    /// Fold the next replica, in replica order.
    fn push(&mut self, replica: &ReplicaSamples) {
        let per_indicator = replica.per_indicator;
        if per_indicator == 0 {
            self.folded += 1;
            return;
        }
        for (fold, samples) in self
            .folds
            .iter_mut()
            .zip(replica.samples.chunks_exact(per_indicator))
        {
            for (k, sample) in samples.iter().enumerate() {
                let Sample {
                    value,
                    sojourn,
                    nb_occurrences: nb_occ,
                    zero_departures,
                    nonzero_reached,
                } = *sample;
                let s = &mut fold.sums[k];
                s.sum += value;
                s.sum_sq += value * value;
                s.sj_sum += sojourn;
                s.sj_sum_sq += sojourn * sojourn;
                s.oc_sum += nb_occ;
                s.oc_sum_sq += nb_occ * nb_occ;
                // Reached by `instant` exactly when it occurred by then: the
                // occurrence count takes an active initial value as its first
                // entry, and its bound is the sample's own.
                let reached = if nb_occ > 0.0 { 1.0 } else { 0.0 };
                s.reached_sum += reached;
                s.zd_sum += zero_departures;
                s.zd_sum_sq += zero_departures * zero_departures;
                s.nonzero_sum += nonzero_reached;
                // Extremes are order-independent, but folded here, in the
                // same replica-ordered pass, so a NaN or a signed zero
                // resolves the same way whatever the thread count.
                fold.extremes.fold(k, value);
                fold.sojourn_extremes.fold(k, sojourn);
                fold.nb_occurrences_extremes.fold(k, nb_occ);
                fold.reached_extremes.fold(k, reached);
                fold.zero_departures_extremes.fold(k, zero_departures);
                fold.nonzero_reached_extremes.fold(k, nonzero_reached);
                if self.keep_columns {
                    fold.value_columns[k].push(value);
                    fold.sojourn_columns[k].push(sojourn);
                }
            }
        }
        self.folded += 1;
    }

    /// The estimates of the folded campaign.
    fn finish(
        self,
        model: &CompiledModel,
        config: &McConfig,
        fmu_units: Vec<FmuProvenance>,
        serial_fallback_unit: Option<String>,
    ) -> McEstimates {
        debug_assert_eq!(self.folded, config.nb_runs, "every replica is folded once");
        let n_instants = config.samples.len();
        let n = config.nb_runs as f64;

        let mut indicators = Vec::with_capacity(model.indicators.len());
        for (indicator, fold) in model.indicators.iter().zip(self.folds) {
            let IndicatorFold {
                sums,
                extremes,
                sojourn_extremes,
                nb_occurrences_extremes,
                reached_extremes,
                zero_departures_extremes,
                nonzero_reached_extremes,
                mut value_columns,
                mut sojourn_columns,
            } = fold;
            let mut mean = vec![0.0; n_instants];
            let mut std = vec![0.0; n_instants];
            let mut sojourn_mean = vec![0.0; n_instants];
            let mut sojourn_std = vec![0.0; n_instants];
            let mut nb_occurrences_mean = vec![0.0; n_instants];
            let mut nb_occurrences_std = vec![0.0; n_instants];
            let mut reached_mean = vec![0.0; n_instants];
            let mut reached_std = vec![0.0; n_instants];
            let mut zero_departures_mean = vec![0.0; n_instants];
            let mut zero_departures_std = vec![0.0; n_instants];
            let mut nonzero_reached_mean = vec![0.0; n_instants];
            let mut nonzero_reached_std = vec![0.0; n_instants];
            for (k, s) in sums.iter().enumerate() {
                mean[k] = s.sum / n;
                sojourn_mean[k] = s.sj_sum / n;
                nb_occurrences_mean[k] = s.oc_sum / n;
                reached_mean[k] = s.reached_sum / n;
                zero_departures_mean[k] = s.zd_sum / n;
                nonzero_reached_mean[k] = s.nonzero_sum / n;
                if config.nb_runs > 1 {
                    std[k] = ((s.sum_sq - n * mean[k] * mean[k]) / (n - 1.0))
                        .max(0.0)
                        .sqrt();
                    sojourn_std[k] = ((s.sj_sum_sq - n * sojourn_mean[k] * sojourn_mean[k])
                        / (n - 1.0))
                        .max(0.0)
                        .sqrt();
                    nb_occurrences_std[k] = ((s.oc_sum_sq
                        - n * nb_occurrences_mean[k] * nb_occurrences_mean[k])
                        / (n - 1.0))
                        .max(0.0)
                        .sqrt();
                    reached_std[k] = ((s.reached_sum - n * reached_mean[k] * reached_mean[k])
                        / (n - 1.0))
                        .max(0.0)
                        .sqrt();
                    zero_departures_std[k] = ((s.zd_sum_sq
                        - n * zero_departures_mean[k] * zero_departures_mean[k])
                        / (n - 1.0))
                        .max(0.0)
                        .sqrt();
                    nonzero_reached_std[k] = ((s.nonzero_sum
                        - n * nonzero_reached_mean[k] * nonzero_reached_mean[k])
                        / (n - 1.0))
                        .max(0.0)
                        .sqrt();
                }
            }
            // Nearest-rank quantiles (deterministic: total_cmp sort over the
            // replica-ordered column).
            let mut quantiles = Vec::new();
            let mut sojourn_quantiles = Vec::new();
            for column in value_columns.iter_mut().chain(sojourn_columns.iter_mut()) {
                column.sort_unstable_by(f64::total_cmp);
            }
            for &q in &config.quantiles {
                let mut value_rows = vec![0.0; n_instants];
                let mut sojourn_rows = vec![0.0; n_instants];
                for k in 0..n_instants {
                    let column = &value_columns[k];
                    let sj_column = &sojourn_columns[k];
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
            // Confidence intervals: closed forms over the sums already
            // reduced above, so they cost O(instants) arithmetic and never
            // revisit a replica.
            let level = config.confidence;
            // The declared kind decides both the construction and, where a
            // campaign observed no dispersion at all, the size of the
            // departure it failed to observe: a 0/1 indicator keeps its
            // sojourn inside `[0, t]`, a free-valued one does not.
            let binary = is_declared_binary(model, indicator);
            let ci = if binary {
                ConfidenceInterval::on_proportion(level, config.nb_runs, &mean)
            } else {
                ConfidenceInterval::on_mean(
                    level,
                    config.nb_runs,
                    &mean,
                    &std,
                    Departure::attribute(),
                )
            };
            let sojourn_departure = if binary {
                Departure::sojourn(&config.samples)
            } else {
                Departure::integral(&config.samples)
            };
            let sojourn_ci = ConfidenceInterval::on_mean(
                level,
                config.nb_runs,
                &sojourn_mean,
                &sojourn_std,
                sojourn_departure,
            );
            let nb_occurrences_ci = ConfidenceInterval::on_mean(
                level,
                config.nb_runs,
                &nb_occurrences_mean,
                &nb_occurrences_std,
                Departure::count(),
            );
            let reached_ci =
                ConfidenceInterval::on_proportion(level, config.nb_runs, &reached_mean);
            let zero_departures_ci = ConfidenceInterval::on_mean(
                level,
                config.nb_runs,
                &zero_departures_mean,
                &zero_departures_std,
                Departure::count(),
            );
            let nonzero_reached_ci =
                ConfidenceInterval::on_proportion(level, config.nb_runs, &nonzero_reached_mean);
            indicators.push(IndicatorEstimate {
                name: indicator.name.clone(),
                instants: config.samples.clone(),
                mean,
                std,
                ci,
                sojourn_mean,
                sojourn_std,
                sojourn_ci,
                nb_occurrences_mean,
                nb_occurrences_std,
                nb_occurrences_ci,
                reached_mean,
                reached_std,
                reached_ci,
                extremes,
                sojourn_extremes,
                nb_occurrences_extremes,
                reached_extremes,
                zero_departures_mean,
                zero_departures_std,
                zero_departures_ci,
                zero_departures_extremes,
                nonzero_reached_mean,
                nonzero_reached_std,
                nonzero_reached_ci,
                nonzero_reached_extremes,
                quantiles,
                sojourn_quantiles,
            });
        }

        McEstimates {
            indicators,
            nb_runs: config.nb_runs,
            seed: config.seed,
            confidence: config.confidence,
            engine_version: env!("CARGO_PKG_VERSION").to_owned(),
            fmu_units,
            serial_fallback_unit,
        }
    }
}

/// Replicas of a sequence campaign simulated together before they are
/// handed on, in replica order. Smaller than [`MAX_CHUNK`] because a recorded
/// trajectory weighs kilobytes on a repairable system where a sample row
/// weighs bytes: 4 096 of them stay in the tens of megabytes, and still
/// give every thread of a large machine a hundred replicas per chunk.
const SEQUENCE_CHUNK: u64 = 1 << 12;

/// The observations of a sequence campaign, resolved against the model;
/// the default plan observes nothing.
#[derive(Default)]
struct ObservationPlan {
    /// The indicator each observation reads.
    columns: Vec<usize>,
    /// The instant each observation is read at, in the order asked.
    times: Vec<f64>,
    /// The distinct instants the engine samples, sorted.
    samples: Vec<f64>,
}

impl ObservationPlan {
    fn new(
        model: &CompiledModel,
        config: &McConfig,
        observations: &[SequenceObservation],
    ) -> Result<Self, EngineError> {
        let mut columns = Vec::with_capacity(observations.len());
        let mut times = Vec::with_capacity(observations.len());
        for observation in observations {
            let column = model
                .indicators
                .iter()
                .position(|indicator| indicator.name == observation.indicator)
                .ok_or_else(|| EngineError::TypeError {
                    time: 0.0,
                    detail: format!(
                        "the observation reads `{}`, which is no indicator of the model",
                        observation.indicator
                    ),
                })?;
            if !observation.time.is_finite() || observation.time < 0.0 {
                return Err(EngineError::TypeError {
                    time: 0.0,
                    detail: format!(
                        "the observation of `{}` is read at {}, which is not an instant of a \
                         trajectory",
                        observation.indicator, observation.time
                    ),
                });
            }
            columns.push(column);
            times.push(observation.read_at(config.t_max));
        }
        let mut samples = times.clone();
        samples.sort_by(f64::total_cmp);
        samples.dedup();
        Ok(ObservationPlan {
            columns,
            times,
            samples,
        })
    }

    /// Simulate one replica: its sequence and its observed values.
    fn replica(
        &self,
        model: &CompiledModel,
        config: &McConfig,
        stop_at_targets: bool,
        replica: u64,
    ) -> Result<(Sequence, Vec<f64>), EngineError> {
        let engine_config = EngineConfig {
            t_max: config.t_max,
            sequences: true,
            stop_at_targets,
            seed: config.seed,
            rng_stream: replica,
            ode: config.ode.clone(),
            flow: config.flow.clone(),
            samples: self.samples.clone(),
            ..EngineConfig::default()
        };
        let result = Engine::new(model, engine_config)?.run()?;
        let row = self
            .columns
            .iter()
            .zip(&self.times)
            .map(|(&column, &time)| {
                result.samples[column]
                    .points
                    .iter()
                    .find(|(at, _)| *at == time)
                    .map(|(_, value)| observed_number(*value))
                    .ok_or_else(|| EngineError::TypeError {
                        time,
                        detail: format!(
                            "replica {replica} recorded no value of `{}` at {time}",
                            model.indicators[column].name
                        ),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sequence = result.sequence.ok_or_else(|| EngineError::TypeError {
            time: 0.0,
            detail: format!("replica {replica} recorded no sequence"),
        })?;
        Ok((sequence, row))
    }
}

/// Simulate every replica of a sequence campaign and hand each one to
/// `sink`, in replica order, in parallel chunks of [`SEQUENCE_CHUNK`]:
/// only one chunk of trajectories is alive at a time.
///
/// `stop_at_targets` decides the regime: on, each trajectory ends at the
/// first feared event, which is the corpus minimal-sequence analysis is
/// defined on; off, it keeps evolving to the horizon, which is what a
/// measure read at an instant *beyond* the feared event needs.
fn stream_sequences<E, F>(
    model: &CompiledModel,
    config: &McConfig,
    stop_at_targets: bool,
    plan: &ObservationPlan,
    mut sink: F,
) -> Result<(), E>
where
    E: Send + From<EngineError>,
    F: FnMut(u64, Sequence, Vec<f64>) -> Result<(), E> + Send,
{
    use rayon::prelude::*;

    in_pool(config.threads, || {
        let mut first = 0;
        while first < config.nb_runs {
            let last = (first + SEQUENCE_CHUNK).min(config.nb_runs);
            let chunk = (first..last)
                .into_par_iter()
                .map(|replica| plan.replica(model, config, stop_at_targets, replica))
                .collect::<Result<Vec<_>, EngineError>>()?;
            for (replica, (sequence, row)) in (first..last).zip(chunk) {
                sink(replica, sequence, row)?;
            }
            first = last;
        }
        Ok(())
    })
}

/// Run `nb_runs` **sequence-recording** replicas and collect their raw
/// per-trajectory sequences, in replica order (deterministic). Each replica
/// runs with sequence recording on and target early-stop; a trajectory that
/// reaches no target still contributes its (target-less) sequence. Feed the
/// result to [`raichu_analysis::analyse`] for the minimal-sequence corpus.
///
/// Every trajectory is held until the campaign ends, so memory grows with
/// `nb_runs`: a large campaign goes through [`run_sequences_streamed`].
///
/// Reports sequences rather than estimators, so it produces no interval
/// and ignores [`McConfig::confidence`].
pub fn run_sequences(
    model: &CompiledModel,
    config: &McConfig,
) -> Result<Vec<Sequence>, EngineError> {
    collect_sequences(model, config, true)
}

/// Every trajectory of a campaign without observations, in replica order.
fn collect_sequences(
    model: &CompiledModel,
    config: &McConfig,
    stop_at_targets: bool,
) -> Result<Vec<Sequence>, EngineError> {
    let mut sequences = Vec::new();
    stream_sequences(
        model,
        config,
        stop_at_targets,
        &ObservationPlan::default(),
        |_, sequence, _| {
            sequences.push(sequence);
            Ok::<_, EngineError>(())
        },
    )?;
    Ok(sequences)
}

/// One quantity a sequence campaign reads on every trajectory: the value of
/// the model indicator named `indicator` at `time`.
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceObservation {
    /// The model indicator whose value is read.
    pub indicator: String,
    /// The instant it is read at. An instant past the horizon reads the
    /// value at the horizon, which is the trajectory's last state.
    pub time: f64,
}

impl SequenceObservation {
    /// The instant it is actually read at on a campaign of horizon
    /// `t_max`: its own, brought back to the horizon when it lies beyond.
    #[must_use]
    pub fn read_at(&self, t_max: f64) -> f64 {
        self.time.min(t_max)
    }
}

/// A sequence campaign's trajectories with the values observed on each.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservedSequences {
    /// The raw trajectories, in replica order.
    pub sequences: Vec<Sequence>,
    /// One row per trajectory, one value per observation in the order they
    /// were asked for; a boolean reads `0` or `1`.
    pub observed: Vec<Vec<f64>>,
    /// The instant each observation was actually read at: its requested
    /// instant, brought back to the horizon when it lay beyond.
    pub times: Vec<f64>,
}

/// [`run_sequences`], reading `observations` on every trajectory as it
/// runs: the same seed gives the same trajectories, plus their values.
///
/// A trajectory that stops at a feared event holds its final state through
/// the later instants, and an instant at an event date reads the state
/// after the event, so the value is the state the trajectory was in at
/// that instant, or ended in.
///
/// Every trajectory is held until the campaign ends; a large campaign goes
/// through [`run_sequences_streamed`].
///
/// # Errors
/// [`EngineError::TypeError`] for an observation naming no model
/// indicator or read at an instant that is negative or not finite; any
/// error a replica raises.
pub fn run_sequences_observed(
    model: &CompiledModel,
    config: &McConfig,
    observations: &[SequenceObservation],
) -> Result<ObservedSequences, EngineError> {
    let plan = ObservationPlan::new(model, config, observations)?;
    let mut sequences = Vec::new();
    let mut observed = Vec::new();
    stream_sequences(model, config, true, &plan, |_, sequence, row| {
        sequences.push(sequence);
        observed.push(row);
        Ok::<_, EngineError>(())
    })?;
    Ok(ObservedSequences {
        sequences,
        observed,
        times: plan.times,
    })
}

/// [`run_sequences_observed`] without holding the campaign: each
/// trajectory is handed to `sink` as `(replica, sequence, observed)`, in
/// replica order, and dropped once `sink` returns. The replicas run in
/// parallel chunks, so memory is bounded by one chunk of trajectories
/// whatever `nb_runs` is; what `sink` keeps is its own affair (a
/// [`raichu_analysis::SequenceReducer`] keeps the distinct cleaned paths,
/// a [`raichu_analysis::RawCorpusWriter`] keeps nothing).
///
/// The same seed gives the same trajectories in the same order as
/// [`run_sequences_observed`], whatever the thread count. Each observation
/// is read at [`SequenceObservation::read_at`].
///
/// # Errors
/// What [`run_sequences_observed`] raises, before any replica for an
/// observation it refuses; the first error `sink` returns, which stops the
/// campaign.
pub fn run_sequences_streamed<E, F>(
    model: &CompiledModel,
    config: &McConfig,
    observations: &[SequenceObservation],
    sink: F,
) -> Result<(), E>
where
    E: Send + From<EngineError>,
    F: FnMut(u64, Sequence, Vec<f64>) -> Result<(), E> + Send,
{
    let plan = ObservationPlan::new(model, config, observations)?;
    stream_sequences(model, config, true, &plan, sink)
}

/// An indicator value as the number a raw corpus records.
fn observed_number(value: Value) -> f64 {
    match value {
        Value::Bool(b) => f64::from(u8::from(b)),
        Value::Int(i) => i as f64,
        Value::Float(x) => x,
    }
}

/// Native **importance measures** of one feared event: run a
/// sequence-recording campaign and reduce it to the per-component
/// Birnbaum, Fussell-Vesely and criticality series over
/// [`McConfig::samples`].
///
/// `target` names the feared event among the model's declared targets; a
/// model with exactly one may leave it out. The campaign runs
/// **free-running** (no target early-stop), because a measure read at an
/// instant is a statement about the state of the system at that instant,
/// and a trajectory frozen at its first feared event stops producing one.
/// The minimal-sequence corpus the cut structure comes from is recovered
/// by truncating each trajectory at that first occurrence, which is the
/// same thing the early stop would have recorded: one campaign, both
/// halves of the answer.
///
/// See [`mod@raichu_analysis::importance`] for what the measures mean and what
/// they assume.
pub fn run_importance(
    model: &CompiledModel,
    config: &McConfig,
    target: Option<&str>,
) -> Result<ImportanceAnalysis, EngineError> {
    // `target_events` keeps the declaration order, so the position of the
    // chosen event is also the index of its `CTarget`.
    let declared = target_events(model);
    let chosen = match target {
        Some(name) => declared
            .iter()
            .position(|(declared_name, _)| declared_name == name)
            .ok_or_else(|| EngineError::TypeError {
                time: 0.0,
                detail: format!(
                    "no feared event named `{name}`; the model declares {}",
                    named_targets(&declared)
                ),
            })?,
        None if declared.len() == 1 => 0,
        None => {
            return Err(EngineError::TypeError {
                time: 0.0,
                detail: format!(
                    "the model declares {} feared events ({}): name the one to measure",
                    declared.len(),
                    named_targets(&declared)
                ),
            })
        }
    };
    let event = &declared[chosen].1;
    // A target whose entry is not recorded never appears in a trajectory,
    // so the analysis would find no cut and report an empty ranking with
    // nothing saying why. Refuse instead, naming what is missing.
    let declared_target = &model.targets[chosen];
    let recorded = model.automata[declared_target.automaton]
        .transitions
        .iter()
        .any(|&index| {
            let transition = &model.transitions[index];
            transition.monitored && transition.targets.contains(&declared_target.state)
        });
    if !recorded {
        return Err(EngineError::TypeError {
            time: 0.0,
            detail: format!(
                "the feared event `{}` is never recorded in a sequence: the transition \
                 entering its state needs `\"monitored\": true`",
                event.name()
            ),
        });
    }
    let raw = collect_sequences(model, config, false)?;
    Ok(importance(&raw, event, &config.samples))
}

/// The declared feared events, for an error message that names the
/// choices instead of only refusing.
fn named_targets(declared: &[(String, raichu_analysis::BasicEvent)]) -> String {
    if declared.is_empty() {
        return "none".to_owned();
    }
    declared
        .iter()
        .map(|(name, _)| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod departure_tests {
    use super::*;

    fn series(points: &[(f64, f64)]) -> Vec<(f64, Value)> {
        points.iter().map(|&(t, x)| (t, Value::Float(x))).collect()
    }

    #[test]
    fn moves_at_the_first_date_are_part_of_the_initial_value() {
        // The declared 0 then what the start-up fixpoint settles on, both at
        // t = 0: the trajectory starts from 1, so no departure, but reached.
        let points = series(&[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (2.0, -1.0)]);
        assert_eq!(zero_departures_at(&points, 0.0), 0.0);
        assert_eq!(zero_departures_at(&points, 1.5), 0.0);
        assert_eq!(zero_departures_at(&points, 2.0), 1.0);
        assert_eq!(nonzero_reached_at(&points, 0.0), 1.0);
        // The other way round: a transient non-zero value the start-up
        // fixpoint settles back to 0 was never the trajectory's value.
        let points = series(&[(0.0, 1.0), (0.0, 0.0), (3.0, 2.0)]);
        assert_eq!(nonzero_reached_at(&points, 2.0), 0.0);
        assert_eq!(nonzero_reached_at(&points, 3.0), 1.0);
        assert_eq!(zero_departures_at(&points, 3.0), 1.0);
    }

    #[test]
    fn the_bound_is_inclusive_and_an_empty_series_reads_zero() {
        let points = series(&[(0.0, 0.0), (2.0, 4.0)]);
        assert_eq!(zero_departures_at(&points, 2.0), 1.0);
        assert_eq!(nonzero_reached_at(&points, 2.0), 1.0);
        assert_eq!(zero_departures_at(&points, 1.999), 0.0);
        assert_eq!(nonzero_reached_at(&points, 1.999), 0.0);
        assert_eq!(zero_departures_at(&[], 1.0), 0.0);
        assert_eq!(nonzero_reached_at(&[], 1.0), 0.0);
    }

    #[test]
    fn booleans_read_as_zero_and_one() {
        let points = vec![
            (0.0, Value::Bool(true)),
            (1.0, Value::Bool(false)),
            (2.0, Value::Bool(true)),
        ];
        assert_eq!(zero_departures_at(&points, 2.0), 1.0);
        assert_eq!(nb_occurrences_at(&points, 2.0), 2.0);
        assert_eq!(nonzero_reached_at(&points, 0.0), 1.0);
    }
}

#[cfg(test)]
mod chunk_tests {
    use super::*;

    #[test]
    fn a_chunk_holds_a_bounded_number_of_bytes_of_samples() {
        let bytes =
            |n_samples: usize| chunk_len(n_samples) as usize * n_samples * size_of::<Sample>();
        // The 165 indicators of the EPS showcase, sampled once: a chunk of
        // 65 536 replicas held about 1 GB of samples.
        assert!(bytes(165) <= CHUNK_BYTES, "{} bytes", bytes(165));
        // A small model keeps the largest chunk.
        assert_eq!(chunk_len(3), MAX_CHUNK);
        // A huge one keeps enough replicas per chunk to feed every thread.
        assert_eq!(chunk_len(10_000_000), MIN_CHUNK);
        assert_eq!(chunk_len(0), MAX_CHUNK);
    }
}
