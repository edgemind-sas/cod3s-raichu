//! PyO3 glue for the `pyraichu` Python package.
//!
//! This crate contains **only** binding code: all engine logic lives in
//! the pure-Rust `crates/*` (unit-testable, benchmarkable, reusable
//! without Python). The extension module is exposed as
//! `pyraichu._pyraichu` and wrapped by the pure-Python package in
//! `python/pyraichu/`.
//!
//! M0 surface: `validate_model`, `simulate_json`. Results cross the FFI
//! as JSON strings (fixture-scale data); zero-copy numpy arrays arrive
//! with the Monte-Carlo milestone where volumes justify them.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::import_exception;
use pyo3::prelude::*;
use raichu::raichu_analysis::analyse as analyse_sequences;
use raichu::raichu_analysis::{
    clean as clean_sequences, minimal_sequences, read_raw_corpus, write_raw_corpus,
    ObservedCondition, RawCorpus, RawHeader, RawObservation,
};
use raichu::raichu_core::{
    fault_tree as generate_fault_tree, fault_tree_for_targets, FaultTree, FaultTreeSettings,
};
use raichu::raichu_core::{
    CoSimulationHost, CompiledModel, Engine, EngineConfig, EngineError,
    FlowConfig as CoreFlowConfig, Snapshot as CoreSnapshot, SolverParams, StochasticDates,
};
use raichu::raichu_explore::{
    exact_domain_report, explore_discretised, explore_discretised_with_fmu, explore_exact,
    read_exploration as read_exploration_result, Algorithm, Cutoffs, DiscretisedSettings,
    ExactSettings, ExplorationResult, Precision,
};
use raichu::raichu_expr::{AttrRef, CmpOp};
use raichu::raichu_fta::{
    fault_tree_envelope, quantify as quantify_tree, read_fault_tree_envelope, read_open_psa,
    Engine as FtaEngine, EnvelopeTop, QuantifySettings,
};
use raichu::raichu_model::Model;
use raichu::raichu_model::{Indicator, IndicatorTarget};
use raichu::raichu_montecarlo::{
    run as mc_run, run_importance as mc_run_importance, run_sequences as mc_run_sequences,
    run_sequences_observed as mc_run_sequences_observed, run_with_fmu as mc_run_with_fmu, McConfig,
    SequenceObservation, DEFAULT_CONFIDENCE,
};
use raichu::raichu_quantify::{
    quantify as quantify_study, quantify_with_fmu as quantify_study_with_fmu, read_quantification,
    Method, QuantifyError, Study,
};
use std::path::Path;

create_exception!(
    _pyraichu,
    ModelError,
    PyException,
    "The model is invalid (typed build-time validation failure)."
);
create_exception!(
    _pyraichu,
    SimulationError,
    PyException,
    "The simulation failed (typed engine error)."
);
// Defined in Python (`pyraichu.UnboundedRateError`, a `SimulationError`)
// so it can carry its fields as attributes while `str()` stays the
// engine's message; imported lazily, when the error is materialised.
import_exception!(pyraichu, UnboundedRateError);

/// The Python exception for an engine error: the typed
/// `UnboundedRateError` when the run met the model's reserved
/// "unbounded" magnitude (so a caller can name the construct to bound),
/// `SimulationError` for every other failure.
fn engine_error(error: EngineError) -> PyErr {
    match &error {
        EngineError::UnboundedRate {
            time,
            variable,
            rate,
            unbounded,
        } => UnboundedRateError::new_err((
            error.to_string(),
            variable.clone(),
            *time,
            *rate,
            *unbounded,
        )),
        _ => SimulationError::new_err(error.to_string()),
    }
}

/// Convergence policy of the continuous flow resolution, as **one
/// object** every entry point accepts under a single `flow=` keyword.
///
/// The four knobs are read together and are the only ones a caller
/// overrides as a group, so grouping them keeps four more positional
/// arguments off signatures that are already wide: two of them suppress
/// the too-many-arguments lint as it is. Every knob left unset (`None`)
/// takes the engine's documented default, so `FlowConfig()` and passing
/// nothing at all are the same run.
// `from_py_object` is what lets an entry point take `Option<FlowConfig>`
// directly: PyO3 extracts it by cloning the held policy, which is four
// scalars. `frozen` because the object is read-only once built, so no
// caller can mutate the policy a running engine was handed.
#[pyclass(frozen, from_py_object)]
#[derive(Clone, Default)]
struct FlowConfig {
    inner: CoreFlowConfig,
}

#[pymethods]
impl FlowConfig {
    /// Build a policy, each unset knob keeping the engine default.
    #[new]
    #[pyo3(signature = (sweep_budget = None, active_set_budget = None, relaxation = None, tolerance = None))]
    fn new(
        sweep_budget: Option<usize>,
        active_set_budget: Option<usize>,
        relaxation: Option<f64>,
        tolerance: Option<f64>,
    ) -> Self {
        let default = CoreFlowConfig::default();
        FlowConfig {
            inner: CoreFlowConfig {
                sweep_budget: sweep_budget.unwrap_or(default.sweep_budget),
                // `None` is the engine default *and* the meaning of the
                // default (derive the budget from the compiled network),
                // so the two coincide and there is nothing to unwrap.
                active_set_budget,
                relaxation: relaxation.unwrap_or(default.relaxation),
                tolerance: tolerance.unwrap_or(default.tolerance),
            },
        }
    }

    /// Sweeps the numeric level of one resolution may spend.
    #[getter]
    fn sweep_budget(&self) -> usize {
        self.inner.sweep_budget
    }

    /// Sweeps the combinatorial level may spend; `None` derives it from
    /// the compiled network.
    #[getter]
    fn active_set_budget(&self) -> Option<usize> {
        self.inner.active_set_budget
    }

    /// Under-relaxation weight latched on a detected two-cycle.
    #[getter]
    fn relaxation(&self) -> f64 {
        self.inner.relaxation
    }

    /// Per-edge convergence tolerance, and the dead band of every
    /// active-set margin.
    #[getter]
    fn tolerance(&self) -> f64 {
        self.inner.tolerance
    }

    // Floats through `{:?}`, which renders 1e-9 as `1e-9` where `{}`
    // would spell out nine zeros: the repr is meant to be read back as
    // the call that would rebuild it.
    fn __repr__(&self) -> String {
        let derived = "None".to_owned();
        format!(
            "FlowConfig(sweep_budget={}, active_set_budget={}, relaxation={:?}, tolerance={:?})",
            self.inner.sweep_budget,
            self.inner
                .active_set_budget
                .map_or(derived, |b| b.to_string()),
            self.inner.relaxation,
            self.inner.tolerance
        )
    }
}

/// The policy an entry point runs under: the caller's, or the engine
/// default when the keyword was omitted.
fn flow_policy(flow: Option<FlowConfig>) -> CoreFlowConfig {
    flow.map_or_else(CoreFlowConfig::default, |f| f.inner)
}

fn parse_and_compile(model_json: &str) -> PyResult<CompiledModel> {
    let model: Model = Model::from_json(model_json)
        .map_err(|e| ModelError::new_err(format!("invalid model JSON: {e}")))?;
    CompiledModel::compile(&model).map_err(|e| ModelError::new_err(e.to_string()))
}

/// Parse and validate a model; raise `ModelError` when invalid.
#[pyfunction]
fn validate_model(model_json: &str) -> PyResult<()> {
    parse_and_compile(model_json).map(|_| ())
}

/// Validate an export manifest and return the resource document and FMI description.
#[pyfunction]
fn prepare_fmu_export(model_json: &str, manifest_json: &str) -> PyResult<(String, String)> {
    let manifest: raichu_fmu::ExportManifest = serde_json::from_str(manifest_json)
        .map_err(|error| ModelError::new_err(error.to_string()))?;
    let document = raichu_fmu::prepare_document(model_json, &manifest)
        .map_err(|error| ModelError::new_err(error.to_string()))?;
    let description = raichu_fmu::model_description(&document)
        .map_err(|error| ModelError::new_err(error.to_string()))?;
    Ok((document, description))
}

/// Switching loops of a model, as JSON: a cycle automaton to variable to
/// automaton where some automaton switches on a single threshold.
///
/// A warning and never a refusal. The loop itself is legitimate, a
/// thermostat is one; what makes it pathological is a switch with no
/// band, so a loop whose every switch has one is not reported.
#[pyfunction]
fn switching_loops_json(model_json: &str) -> PyResult<String> {
    let compiled = parse_and_compile(model_json)?;
    // Read off the compiled model: the walk already ran once, at compile
    // time, and running it again here would answer the same question
    // twice.
    let found: Vec<_> = compiled
        .switching_loops
        .iter()
        .map(|loop_| {
            serde_json::json!({
                "automata": loop_.automata,
                "bandless": loop_.bandless,
                "through": loop_.through,
                "message": loop_.describe(),
            })
        })
        .collect();
    serde_json::to_string(&found).map_err(|e| ModelError::new_err(e.to_string()))
}

/// Unfed triggers of a model, as JSON: a mode sealed for the whole run
/// by an in port no connection reaches.
///
/// A warning and never a refusal. The model is valid and it runs; an
/// aggregation over an empty port answers, so the mode is decided by the
/// missing wire rather than by the model, and what it arms, a cold
/// standby or a rule that produces, delivers from the initial instant
/// and for ever.
#[pyfunction]
fn unfed_triggers_json(model_json: &str) -> PyResult<String> {
    let compiled = parse_and_compile(model_json)?;
    // Read off the compiled model, like the loops next door: the search
    // already ran once, at compile time.
    let found: Vec<_> = compiled
        .unfed_triggers
        .iter()
        .map(|trigger| {
            serde_json::json!({
                "ports": trigger.ports,
                "automaton": trigger.automaton,
                "state": trigger.state,
                "message": trigger.describe(),
            })
        })
        .collect();
    serde_json::to_string(&found).map_err(|e| ModelError::new_err(e.to_string()))
}

/// Feature names an authored model document requires, derived from the
/// constructs it contains (never from a declaration).
///
/// The authoring layer calls this to compose the format envelope, so the
/// derivation has a single home, in Rust, and a Python-side list can
/// never lag what the body holds.
#[pyfunction]
fn required_features(model_json: &str) -> PyResult<Vec<String>> {
    let model = Model::seal_json(model_json)
        .and_then(|sealed| Model::from_json(&sealed))
        .map_err(|e| ModelError::new_err(format!("invalid model JSON: {e}")))?;
    Ok(model
        .required_features()
        .into_iter()
        .map(|feature| feature.name().to_owned())
        .collect())
}

/// Rewrite an authored model document (bare or already enveloped) as a
/// sealed one, with the required-feature list derived from the body.
#[pyfunction]
fn seal_model(model_json: &str) -> PyResult<String> {
    Model::seal_json(model_json)
        .map_err(|e| ModelError::new_err(format!("invalid model JSON: {e}")))
}

/// Run a deterministic simulation and return the full result
/// (events, indicator series, dense samples, causal journal,
/// provenance) as JSON.
///
/// `flow` is an optional [`FlowConfig`] overriding the convergence
/// policy of the continuous flow resolution (engine defaults when
/// omitted).
///
/// The GIL is released while the engine runs.
#[pyfunction]
#[pyo3(signature = (model_json, t_max, journal = false, confluence_check = false, samples = None, seed = 0, rng_stream = 0, flow = None, max_transition_firings = None, max_flow_restarts = None, allow_fmu_import = false, fmu_base_dir = None))]
#[allow(clippy::too_many_arguments)] // mirrors the Python keyword signature
fn simulate_json(
    py: Python<'_>,
    model_json: &str,
    t_max: f64,
    journal: bool,
    confluence_check: bool,
    samples: Option<Vec<f64>>,
    seed: u64,
    rng_stream: u64,
    flow: Option<FlowConfig>,
    max_transition_firings: Option<u64>,
    max_flow_restarts: Option<u64>,
    allow_fmu_import: bool,
    fmu_base_dir: Option<&str>,
) -> PyResult<String> {
    let compiled = parse_and_compile(model_json)?;
    let flow = flow_policy(flow);
    py.detach(|| {
        let defaults = EngineConfig::default();
        let config = EngineConfig {
            t_max,
            journal,
            confluence_check,
            samples: samples.unwrap_or_default(),
            seed,
            rng_stream,
            flow,
            max_transition_firings: max_transition_firings
                .unwrap_or(defaults.max_transition_firings),
            max_flow_restarts: max_flow_restarts.unwrap_or(defaults.max_flow_restarts),
            allow_fmu_import,
            ..defaults
        };
        let result = if compiled.fmu_units.is_empty() {
            Engine::new(&compiled, config)
                .map_err(engine_error)?
                .run()
                .map_err(engine_error)?
        } else {
            let mut host = CoSimulationHost::prepare_authorized(
                &compiled,
                Path::new(fmu_base_dir.unwrap_or(".")),
                allow_fmu_import,
            )
            .map_err(engine_error)?;
            Engine::new_with_host(&compiled, config, &mut host)
                .map_err(engine_error)?
                .run()
                .map_err(engine_error)?
        };
        serde_json::to_string(&result).map_err(|e| SimulationError::new_err(e.to_string()))
    })
}

/// Run a Monte-Carlo estimation (M2): `nb_runs` replicas on independent
/// RNG substreams of `seed`, estimates (mean/std/sojourn) at `samples`.
///
/// The GIL is released; replicas run in parallel with an index-ordered
/// reduction (identical bytes whatever the thread count).
///
/// The `rtol`/`atol`/`max_step`/`tol_event`/`sub_samples` keywords
/// override the corresponding ODE-backend parameters (engine defaults
/// when omitted): the knobs of the tolerance-parity experiments. `flow`
/// is an optional [`FlowConfig`] overriding the convergence policy of
/// the continuous flow resolution, applied to every replica.
///
/// `confidence` is the confidence level of the interval every estimator
/// carries, a study parameter in `(0, 1)`; omitted, it is
/// `DEFAULT_CONFIDENCE` (0.95). The level applied is reported back with
/// the result, so an artefact never leaves it to be assumed.
#[pyfunction]
#[pyo3(signature = (model_json, nb_runs, t_max, samples, seed = 0, threads = None, quantiles = None, confidence = None, rtol = None, atol = None, max_step = None, tol_event = None, sub_samples = None, stop_at_targets = false, flow = None, event_resolution = None, allow_fmu_import = false, fmu_base_dir = None, require_parallel = false))]
#[allow(clippy::too_many_arguments)] // mirrors the Python keyword signature
fn monte_carlo_json(
    py: Python<'_>,
    model_json: &str,
    nb_runs: u64,
    t_max: f64,
    samples: Vec<f64>,
    seed: u64,
    threads: Option<usize>,
    quantiles: Option<Vec<f64>>,
    confidence: Option<f64>,
    rtol: Option<f64>,
    atol: Option<f64>,
    max_step: Option<f64>,
    tol_event: Option<f64>,
    sub_samples: Option<usize>,
    stop_at_targets: bool,
    flow: Option<FlowConfig>,
    event_resolution: Option<f64>,
    allow_fmu_import: bool,
    fmu_base_dir: Option<&str>,
    require_parallel: bool,
) -> PyResult<String> {
    if let Some(v) = event_resolution {
        if !(v.is_finite() && v > 0.0) {
            return Err(SimulationError::new_err(format!(
                "event_resolution is the widest spacing accepted between two \
                 crossing scan points, a finite positive time, got {v}"
            )));
        }
    }
    let compiled = parse_and_compile(model_json)?;
    let flow = flow_policy(flow);
    py.detach(|| {
        let mut ode = SolverParams::default();
        if let Some(v) = rtol {
            ode.rtol = v;
        }
        if let Some(v) = atol {
            ode.atol = v;
        }
        if let Some(v) = max_step {
            ode.max_step = v;
        }
        if let Some(v) = tol_event {
            ode.tol_event = v;
        }
        if let Some(v) = sub_samples {
            ode.sub_samples = v;
        }
        ode.event_resolution = event_resolution;
        let config = McConfig {
            nb_runs,
            seed,
            t_max,
            samples,
            threads,
            quantiles: quantiles.unwrap_or_default(),
            confidence: confidence.unwrap_or(DEFAULT_CONFIDENCE),
            ode,
            stop_at_targets,
            flow,
        };
        let estimates = if compiled.fmu_units.is_empty() {
            mc_run(&compiled, &config)
        } else {
            mc_run_with_fmu(
                &compiled,
                &config,
                Path::new(fmu_base_dir.unwrap_or(".")),
                allow_fmu_import,
                require_parallel,
            )
        }
        .map_err(engine_error)?;
        serde_json::to_string(&estimates).map_err(|e| SimulationError::new_err(e.to_string()))
    })
}

/// Native minimal-sequence analysis: run `nb_runs` sequence-recording
/// replicas (target early-stop) and return the JSON of the minimal sequences
/// (group → filter-cycles → minimal), the RAMS output cod3s produces via its
/// `SequenceAnalyser`. Each minimal sequence is `{events, end_cause,
/// end_time, weight}` (weight = the trajectory count that collapsed into it).
///
/// `flow` is an optional [`FlowConfig`] overriding the convergence
/// policy of the continuous flow resolution, applied to every replica.
#[pyfunction]
#[pyo3(signature = (model_json, nb_runs, t_max, seed = 0, threads = None, flow = None))]
fn analyse_sequences_json(
    py: Python<'_>,
    model_json: &str,
    nb_runs: u64,
    t_max: f64,
    seed: u64,
    threads: Option<usize>,
    flow: Option<FlowConfig>,
) -> PyResult<String> {
    let compiled = parse_and_compile(model_json)?;
    let flow = flow_policy(flow);
    py.detach(|| {
        let config = McConfig {
            nb_runs,
            seed,
            t_max,
            samples: Vec::new(),
            threads,
            quantiles: Vec::new(),
            // Reports sequences, not estimators: no interval is produced.
            confidence: DEFAULT_CONFIDENCE,
            ode: SolverParams::default(),
            stop_at_targets: false,
            flow,
        };
        let raw = mc_run_sequences(&compiled, &config).map_err(engine_error)?;
        let minimal = analyse_sequences(raw);
        serde_json::to_string(&minimal).map_err(|e| SimulationError::new_err(e.to_string()))
    })
}

/// The two reduced levels of a raw corpus, as the JSON object
/// `{"cleaned": [...], "minimal": [...]}`.
fn reduced_levels(raw: Vec<raichu::raichu_core::engine::Sequence>) -> PyResult<serde_json::Value> {
    let cleaned = clean_sequences(raw);
    let minimal = minimal_sequences(cleaned.clone());
    Ok(serde_json::json!({
        "cleaned": serde_json::to_value(&cleaned).map_err(|e| SimulationError::new_err(e.to_string()))?,
        "minimal": serde_json::to_value(&minimal).map_err(|e| SimulationError::new_err(e.to_string()))?,
    }))
}

/// A condition as the binding receives it: `(observation, op, value)`, with
/// `op` one of `==`, `!=`, `<`, `<=`, `>`, `>=`.
type ConditionArg = (String, String, f64);

fn observed_condition(condition: ConditionArg) -> PyResult<ObservedCondition> {
    let (observation, op, value) = condition;
    let op = match op.as_str() {
        "==" => CmpOp::Eq,
        "!=" => CmpOp::Ne,
        "<" => CmpOp::Lt,
        "<=" => CmpOp::Le,
        ">" => CmpOp::Gt,
        ">=" => CmpOp::Ge,
        other => {
            return Err(SimulationError::new_err(format!(
                "`{other}` is not a comparison; use one of ==, !=, <, <=, >, >="
            )))
        }
    };
    Ok(ObservedCondition {
        observation,
        op,
        value,
    })
}

/// Reduce a corpus to its two levels, first keeping only the trajectories
/// `condition` holds on when one is given, and report what it kept.
fn reduce_corpus(
    corpus: &RawCorpus,
    condition: Option<ConditionArg>,
) -> PyResult<serde_json::Value> {
    let Some(condition) = condition else {
        return reduced_levels(corpus.sequences.clone());
    };
    let symbol = condition.1.clone();
    let condition = observed_condition(condition)?;
    let kept = corpus
        .filtered(&condition)
        .map_err(|e| SimulationError::new_err(e.to_string()))?;
    let kept_count = kept.len();
    let mut levels = reduced_levels(kept)?;
    levels["condition"] = serde_json::json!({
        "observation": condition.observation,
        "op": symbol,
        "value": condition.value,
        "total_trajectories": corpus.sequences.len(),
        "kept_trajectories": kept_count,
    });
    Ok(levels)
}

/// A sequence campaign kept whole: run `nb_runs` sequence-recording replicas
/// (target early-stop), write the RAW corpus to `raw_path` in the
/// `raichu.sequences` format when one is given, and return the JSON of the
/// two reduced levels, `{"cleaned": [...], "minimal": [...]}`.
///
/// `observations` lists `(name, component, attribute, time)`: the value of
/// that attribute at that instant is read on every trajectory and carried by
/// the raw corpus under `name`. `condition` is `(observation, op, value)`:
/// only the trajectories whose observed value satisfies it are reduced, and
/// the returned JSON then holds a `condition` report. The raw corpus always
/// holds every trajectory.
///
/// The raw corpus is written from Rust and never materialised as Python
/// objects: a large campaign holds one line per replica.
#[pyfunction]
#[pyo3(signature = (model_json, nb_runs, t_max, seed = 0, threads = None, flow = None, raw_path = None, observations = None, condition = None))]
#[allow(clippy::too_many_arguments)]
fn run_sequences_json(
    py: Python<'_>,
    model_json: &str,
    nb_runs: u64,
    t_max: f64,
    seed: u64,
    threads: Option<usize>,
    flow: Option<FlowConfig>,
    raw_path: Option<std::path::PathBuf>,
    observations: Option<Vec<(String, String, String, f64)>>,
    condition: Option<ConditionArg>,
) -> PyResult<String> {
    let mut model: Model = Model::from_json(model_json)
        .map_err(|e| ModelError::new_err(format!("invalid model JSON: {e}")))?;
    let observations = observations.unwrap_or_default();
    // Each observation reads an attribute through an indicator added for
    // it: the engine samples indicators, and a name already taken would
    // make the reading ambiguous.
    for (name, component, attribute, _) in &observations {
        if model
            .indicators
            .iter()
            .any(|indicator| &indicator.name == name)
        {
            return Err(ModelError::new_err(format!(
                "the observation `{name}` has the name of an indicator the model already declares"
            )));
        }
        model.indicators.push(Indicator {
            name: name.clone(),
            target: IndicatorTarget::Attribute {
                attr: AttrRef {
                    component: component.clone(),
                    attribute: attribute.clone(),
                },
            },
        });
    }
    let compiled =
        CompiledModel::compile(&model).map_err(|e| ModelError::new_err(e.to_string()))?;
    let flow = flow_policy(flow);
    py.detach(|| {
        let config = McConfig {
            nb_runs,
            seed,
            t_max,
            samples: Vec::new(),
            threads,
            quantiles: Vec::new(),
            confidence: DEFAULT_CONFIDENCE,
            ode: SolverParams::default(),
            stop_at_targets: false,
            flow,
        };
        let asked: Vec<SequenceObservation> = observations
            .iter()
            .map(|(name, _, _, time)| SequenceObservation {
                indicator: name.clone(),
                time: *time,
            })
            .collect();
        let campaign =
            mc_run_sequences_observed(&compiled, &config, &asked).map_err(engine_error)?;
        let header = RawHeader::new(
            raichu::VERSION,
            &model.name,
            seed,
            nb_runs,
            t_max,
            compiled.targets.iter().map(|t| t.name.clone()).collect(),
        )
        .with_observations(
            observations
                .iter()
                .zip(&campaign.times)
                .map(|((name, _, _, _), time)| RawObservation {
                    name: name.clone(),
                    time: *time,
                })
                .collect(),
        );
        let observed = if observations.is_empty() {
            Vec::new()
        } else {
            campaign.observed
        };
        if let Some(path) = raw_path {
            let file = std::fs::File::create(&path).map_err(|e| {
                SimulationError::new_err(format!("cannot create {}: {e}", path.display()))
            })?;
            write_raw_corpus(
                std::io::BufWriter::new(file),
                &header,
                &campaign.sequences,
                &observed,
            )
            .map_err(|e| SimulationError::new_err(e.to_string()))?;
        }
        let corpus = RawCorpus {
            header,
            sequences: campaign.sequences,
            observed,
        };
        Ok(reduce_corpus(&corpus, condition)?.to_string())
    })
}

/// Read a raw corpus written by [`run_sequences_json`] and reduce it again:
/// the JSON `{"header": {...}, "cleaned": [...], "minimal": [...]}`, reduced
/// from the trajectories `condition` holds on when one is given (plus a
/// `condition` report).
#[pyfunction]
#[pyo3(signature = (raw_path, condition = None))]
fn analyse_raw_sequences_json(
    py: Python<'_>,
    raw_path: std::path::PathBuf,
    condition: Option<ConditionArg>,
) -> PyResult<String> {
    py.detach(|| {
        let file = std::fs::File::open(&raw_path).map_err(|e| {
            SimulationError::new_err(format!("cannot open {}: {e}", raw_path.display()))
        })?;
        let corpus = read_raw_corpus(std::io::BufReader::new(file))
            .map_err(|e| SimulationError::new_err(e.to_string()))?;
        let mut levels = reduce_corpus(&corpus, condition)?;
        levels["header"] = serde_json::to_value(&corpus.header)
            .map_err(|e| SimulationError::new_err(e.to_string()))?;
        Ok(levels.to_string())
    })
}

/// Native **importance measures**: run `nb_runs` sequence-recording
/// replicas and return the JSON of the per-component Birnbaum,
/// Fussell-Vesely and criticality series at `instants`, plus the minimal
/// cut sets they were computed on.
///
/// `target` names the feared event among the model's declared targets; a
/// model with exactly one may leave it out. One campaign answers both
/// halves: the cut structure comes from the trajectories truncated at the
/// feared event, the probabilities from the same trajectories left to run
/// to the horizon.
///
/// `flow` is an optional [`FlowConfig`] overriding the convergence
/// policy of the continuous flow resolution, applied to every replica.
#[pyfunction]
#[pyo3(signature = (model_json, nb_runs, t_max, instants, target = None, seed = 0, threads = None, flow = None))]
#[allow(clippy::too_many_arguments)] // mirrors the Python keyword signature
fn importance_json(
    py: Python<'_>,
    model_json: &str,
    nb_runs: u64,
    t_max: f64,
    instants: Vec<f64>,
    target: Option<String>,
    seed: u64,
    threads: Option<usize>,
    flow: Option<FlowConfig>,
) -> PyResult<String> {
    let compiled = parse_and_compile(model_json)?;
    let flow = flow_policy(flow);
    py.detach(|| {
        let config = McConfig {
            nb_runs,
            seed,
            t_max,
            samples: instants,
            threads,
            quantiles: Vec::new(),
            // Reports importance measures, not estimators: no interval is produced.
            confidence: DEFAULT_CONFIDENCE,
            ode: SolverParams::default(),
            stop_at_targets: false,
            flow,
        };
        let analysis =
            mc_run_importance(&compiled, &config, target.as_deref()).map_err(engine_error)?;
        serde_json::to_string(&analysis).map_err(|e| SimulationError::new_err(e.to_string()))
    })
}

/// Explore the sequence tree of a model to the target `target` and return
/// the result in its open format, `raichu.exploration` (version 1 for an
/// exact result, version 2 for a discretised one), as JSON.
///
/// Each retained sequence carries its probability (dimensionless) of
/// reaching the target by `horizon` (in the model's time unit), and the
/// result carries a lower bound (the sum of the retained probabilities)
/// and an upper bound (plus the mass every cut-off discarded). Every
/// cut-off is optional.
///
/// `algorithm` selects the driver:
///
/// - `"exact"` (default): the Markov family, probabilities in closed
///   form. `rel_precision` and `max_terms` override the numerical
///   precision of the sequence probabilities (engine defaults when
///   omitted); `level` must be omitted and `refine` left true.
/// - `"discretised"`: every law the engine carries and continuous
///   evolution. The distribution of the next event is cut into `level`
///   equal-mass cells (default `DEFAULT_LEVEL`, 8); with `refine` the run
///   is repeated at `2 x level`, which is the reported one, and the
///   difference is the error estimate. `max_branches` defaults to
///   `DEFAULT_MAX_BRANCHES` per pass when omitted; `rel_precision` and
///   `max_terms` must be omitted (they do not apply).
///
/// An unknown algorithm, a setting that does not apply to the selected
/// one, an invalid setting, a model outside the algorithm's domain, and a
/// law outside it armed along an explored sequence raise
/// `SimulationError` with the engine's message, the last one naming the
/// transition and the sequence.
///
/// The GIL is released while the exploration runs.
#[pyfunction]
#[pyo3(signature = (model_json, target, horizon, algorithm = "exact", min_probability = None, max_length = None, max_failures = None, max_branches = None, gap_tolerance = None, rel_precision = None, max_terms = None, threads = None, level = None, refine = true, allow_fmu_import = false, fmu_base_dir = None))]
#[allow(clippy::too_many_arguments)] // mirrors the Python keyword signature
fn explore_json(
    py: Python<'_>,
    model_json: &str,
    target: String,
    horizon: f64,
    algorithm: &str,
    min_probability: Option<f64>,
    max_length: Option<usize>,
    max_failures: Option<u64>,
    max_branches: Option<u64>,
    gap_tolerance: Option<f64>,
    rel_precision: Option<f64>,
    max_terms: Option<usize>,
    threads: Option<usize>,
    level: Option<u32>,
    refine: bool,
    allow_fmu_import: bool,
    fmu_base_dir: Option<&str>,
) -> PyResult<String> {
    let parsed: Algorithm = algorithm.parse().map_err(|e| {
        SimulationError::new_err(format!(
            "exploration algorithm `{algorithm}` is not provided by this engine: {e}"
        ))
    })?;
    let not_applicable = |setting: &str, only: &str| {
        SimulationError::new_err(format!(
            "`{setting}` does not apply to the `{algorithm}` exploration algorithm \
             (only to `{only}`)"
        ))
    };
    let compiled = parse_and_compile(model_json)?;
    match parsed {
        Algorithm::Exact => {
            if level.is_some() {
                return Err(not_applicable("level", "discretised"));
            }
            if !refine {
                return Err(not_applicable("refine", "discretised"));
            }
            let mut settings = ExactSettings::new(target, horizon);
            settings.cutoffs = Cutoffs {
                min_probability,
                max_length,
                max_failures,
                max_branches,
            };
            if let Some(tolerance) = gap_tolerance {
                settings.gap_tolerance = tolerance;
            }
            let defaults = Precision::default();
            settings.precision = Precision {
                rel_precision: rel_precision.unwrap_or(defaults.rel_precision),
                max_terms: max_terms.unwrap_or(defaults.max_terms),
            };
            settings.threads = threads;
            py.detach(|| {
                let result = explore_exact(&compiled, &settings).map_err(engine_error)?;
                serde_json::to_string(&result).map_err(|e| SimulationError::new_err(e.to_string()))
            })
        }
        Algorithm::Discretised => {
            if rel_precision.is_some() {
                return Err(not_applicable("rel_precision", "exact"));
            }
            if max_terms.is_some() {
                return Err(not_applicable("max_terms", "exact"));
            }
            let mut settings = DiscretisedSettings::new(target, horizon);
            settings.cutoffs = Cutoffs {
                min_probability,
                max_length,
                max_failures,
                max_branches: max_branches.or(settings.cutoffs.max_branches),
            };
            if let Some(tolerance) = gap_tolerance {
                settings.gap_tolerance = tolerance;
            }
            if let Some(level) = level {
                settings.level = level;
            }
            settings.refine = refine;
            settings.threads = threads;
            py.detach(|| {
                let result = if compiled.fmu_units.is_empty() {
                    explore_discretised(&compiled, &settings)
                } else {
                    explore_discretised_with_fmu(
                        &compiled,
                        &settings,
                        Path::new(fmu_base_dir.unwrap_or(".")),
                        allow_fmu_import,
                    )
                }
                .map_err(engine_error)?;
                serde_json::to_string(&result).map_err(|e| SimulationError::new_err(e.to_string()))
            })
        }
    }
}

/// The static exact-domain report of a model, as a JSON array: every
/// reason, found without exploring, that the model is outside the exact
/// exploration domain (an ODE, a reachable watched transition, an
/// expression reading time). Each entry is the tagged violation
/// (`kind` plus its fields) with a readable `message`; empty when none
/// is found. A law outside the domain is only found during a run.
#[pyfunction]
fn exploration_domain_json(model_json: &str) -> PyResult<String> {
    let compiled = parse_and_compile(model_json)?;
    let report = exact_domain_report(&compiled)
        .into_iter()
        .map(|violation| {
            let mut entry = serde_json::to_value(&violation)
                .map_err(|e| SimulationError::new_err(e.to_string()))?;
            entry["message"] = serde_json::Value::String(violation.to_string());
            Ok(entry)
        })
        .collect::<PyResult<Vec<_>>>()?;
    serde_json::to_string(&report).map_err(|e| SimulationError::new_err(e.to_string()))
}

/// Generate a fault tree from `model_json`: explaining `top_json` (an
/// expression over the model's states and attributes) or, when
/// `targets_json` is given instead, the model's declared targets of those
/// names (`["name", ...]`). Attributes are unrolled into the states that
/// compute them; `profile_json` (`[[qualified name, value], ...]`) holds
/// some at a value instead.
fn generate_tree(
    model_json: &str,
    top_json: Option<&str>,
    targets_json: Option<&str>,
    profile_json: Option<&str>,
    max_nodes: Option<usize>,
) -> PyResult<(FaultTree, EnvelopeTop)> {
    let compiled = parse_and_compile(model_json)?;
    let profile = match profile_json {
        None => Vec::new(),
        Some(text) => serde_json::from_str(text)
            .map_err(|e| SimulationError::new_err(format!("fault tree: the profile: {e}")))?,
    };
    let settings = FaultTreeSettings { profile, max_nodes };
    let (tree, top) = match (top_json, targets_json) {
        (Some(top_json), None) => {
            let top = serde_json::from_str(top_json).map_err(|e| {
                SimulationError::new_err(format!("fault tree: the top expression: {e}"))
            })?;
            let expression = serde_json::from_str(top_json).map_err(|e| {
                SimulationError::new_err(format!("fault tree: the top expression: {e}"))
            })?;
            (
                generate_fault_tree(&compiled, &top, &settings),
                EnvelopeTop::Expression { expression },
            )
        }
        (None, Some(targets_json)) => {
            let targets: Vec<String> = serde_json::from_str(targets_json)
                .map_err(|e| SimulationError::new_err(format!("fault tree: the targets: {e}")))?;
            (
                fault_tree_for_targets(&compiled, &targets, &settings),
                EnvelopeTop::Targets { targets },
            )
        }
        _ => {
            return Err(SimulationError::new_err(
                "fault tree: give either a top expression or targets, not both and not neither",
            ))
        }
    };
    let tree = tree.map_err(|e| SimulationError::new_err(e.to_string()))?;
    Ok((tree, top))
}

/// Generate the fault tree explaining `top_json`, or the targets named by
/// `targets_json` (see `generate_tree`). Answers `{"tree": {top,
/// basic_events, warnings}, "minimal_cut_sets": [[event index, ...]],
/// "open_psa": "<document>"}`.
#[pyfunction]
#[pyo3(signature = (model_json, top_json = None, profile_json = None, max_nodes = None, cut_set_limit = 100_000, name = "fault_tree", targets_json = None))]
fn fault_tree_json(
    model_json: &str,
    top_json: Option<&str>,
    profile_json: Option<&str>,
    max_nodes: Option<usize>,
    cut_set_limit: usize,
    name: &str,
    targets_json: Option<&str>,
) -> PyResult<String> {
    let (tree, _) = generate_tree(model_json, top_json, targets_json, profile_json, max_nodes)?;
    let cuts = tree
        .minimal_cut_sets(cut_set_limit)
        .map_err(|e| SimulationError::new_err(e.to_string()))?;
    let answer = serde_json::json!({
        "tree": tree,
        "minimal_cut_sets": cuts,
        "open_psa": tree.to_open_psa(name),
    });
    serde_json::to_string(&answer).map_err(|e| SimulationError::new_err(e.to_string()))
}

fn fta_engine(engine: &str) -> PyResult<FtaEngine> {
    match engine {
        "auto" => Ok(FtaEngine::Auto),
        "exact" => Ok(FtaEngine::Exact),
        "cut_sets" => Ok(FtaEngine::CutSets),
        other => Err(SimulationError::new_err(format!(
            "fault tree: engine `{other}` is not one of auto, exact, cut_sets"
        ))),
    }
}

/// Generate the fault tree of `top_json` or `targets_json` (see
/// `generate_tree`) and quantify it at each of `mission_times` (at most
/// 20, strictly increasing; the last is the horizon, where the minimal
/// cut sets and the importance measures are computed). Answers the
/// `raichu.fault_tree` envelope as JSON. The GIL is released while the
/// tree is quantified.
#[pyfunction]
#[pyo3(signature = (model_json, mission_times, top_json = None, targets_json = None, profile_json = None, max_nodes = None, name = "fault_tree", max_bdd_nodes = 10_000_000, cut_set_limit = 100_000, cut_sets = true, engine = "auto", max_order = None, min_cut_probability = 0.0, max_cut_sets = 1_000_000, max_expansions = 100_000_000))]
#[allow(clippy::too_many_arguments)] // mirrors the Python keyword signature
fn fault_tree_envelope_json(
    py: Python<'_>,
    model_json: &str,
    mission_times: Vec<f64>,
    top_json: Option<&str>,
    targets_json: Option<&str>,
    profile_json: Option<&str>,
    max_nodes: Option<usize>,
    name: &str,
    max_bdd_nodes: usize,
    cut_set_limit: usize,
    cut_sets: bool,
    engine: &str,
    max_order: Option<usize>,
    min_cut_probability: f64,
    max_cut_sets: usize,
    max_expansions: u64,
) -> PyResult<String> {
    let engine = fta_engine(engine)?;
    if !(0.0..=1.0).contains(&min_cut_probability) {
        return Err(SimulationError::new_err(format!(
            "fault tree: min_cut_probability {min_cut_probability} is outside [0, 1]"
        )));
    }
    let (tree, top) = generate_tree(model_json, top_json, targets_json, profile_json, max_nodes)?;
    let settings = QuantifySettings {
        mission_time: None,
        max_bdd_nodes,
        cut_set_limit,
        cut_sets,
        engine,
        max_order,
        min_cut_probability,
        max_cut_sets,
        max_expansions,
    };
    let envelope = py
        .detach(|| fault_tree_envelope(&tree, name, top, &mission_times, &settings))
        .map_err(|e| SimulationError::new_err(e.to_string()))?;
    envelope
        .to_json()
        .map_err(|e| SimulationError::new_err(e.to_string()))
}

/// Validate a `raichu.fault_tree` envelope (its `format`, its `version`
/// and its fields); raise `SimulationError` when it is not one this
/// engine reads.
#[pyfunction]
fn validate_fault_tree_envelope(envelope_json: &str) -> PyResult<()> {
    read_fault_tree_envelope(envelope_json)
        .map(|_| ())
        .map_err(|e| SimulationError::new_err(e.to_string()))
}

/// Quantify the fault tree an OpenPSA document defines, exactly: the
/// top-event probability at `mission_time`, every basic event's
/// importance, and the minimal cut sets when the tree is coherent and
/// they number at most `cut_set_limit` (not extracted at all when
/// `cut_sets` is false). `top` names the top gate when the
/// document has several unreferenced ones. Answers the quantification
/// with `"tree"` (the document's name) and `"events"` (the basic-event
/// names, which the indices of the result refer to).
#[pyfunction]
#[pyo3(signature = (open_psa, top = None, mission_time = None, max_bdd_nodes = 10_000_000, cut_set_limit = 100_000, cut_sets = true, engine = "auto", max_order = None, min_cut_probability = 0.0, max_cut_sets = 1_000_000, max_expansions = 100_000_000))]
#[allow(clippy::too_many_arguments)]
fn fault_tree_quantify_json(
    py: Python<'_>,
    open_psa: &str,
    top: Option<&str>,
    mission_time: Option<f64>,
    max_bdd_nodes: usize,
    cut_set_limit: usize,
    cut_sets: bool,
    engine: &str,
    max_order: Option<usize>,
    min_cut_probability: f64,
    max_cut_sets: usize,
    max_expansions: u64,
) -> PyResult<String> {
    let engine = fta_engine(engine)?;
    if !(0.0..=1.0).contains(&min_cut_probability) {
        return Err(SimulationError::new_err(format!(
            "fault tree: min_cut_probability {min_cut_probability} is outside [0, 1]"
        )));
    }
    let settings = QuantifySettings {
        mission_time,
        max_bdd_nodes,
        cut_set_limit,
        cut_sets,
        engine,
        max_order,
        min_cut_probability,
        max_cut_sets,
        max_expansions,
    };
    // The GIL is released while the tree is read and quantified.
    let (tree, result) = py
        .detach(|| {
            let tree = read_open_psa(open_psa, top)?;
            let result = quantify_tree(&tree, &settings)?;
            Ok::<_, raichu::raichu_fta::FtaError>((tree, result))
        })
        .map_err(|e| SimulationError::new_err(e.to_string()))?;
    // Serialised straight to text: the cut-set count is a u128, which a
    // `serde_json::Value` cannot hold past u64 (`json!` would panic).
    #[derive(serde::Serialize)]
    struct Answer<'a> {
        tree: &'a str,
        events: Vec<&'a str>,
        result: &'a raichu::raichu_fta::Quantification,
    }
    let answer = Answer {
        tree: &tree.name,
        events: tree.events.iter().map(|e| e.name.as_str()).collect(),
        result: &result,
    };
    serde_json::to_string(&answer).map_err(|e| SimulationError::new_err(e.to_string()))
}

/// Parse an exploration result through the format's own reader.
fn parse_exploration(result_json: &str) -> PyResult<ExplorationResult> {
    read_exploration_result(result_json).map_err(|e| SimulationError::new_err(e.to_string()))
}

/// Validate an exploration result document (its `format`, its `version`
/// and its fields); raise `SimulationError` when it is not one this
/// engine reads. The validating half of `pyraichu.read_exploration`,
/// which builds the Python object from the text itself: Python parses a
/// float to the nearest double, so the round trip stays exact.
#[pyfunction]
fn validate_exploration(result_json: &str) -> PyResult<()> {
    parse_exploration(result_json).map(|_| ())
}

/// Reduce an exploration result to its minimal sequences through the
/// engine's own reduction (group, filter cycles, absorb super-sequences),
/// as a JSON array of `{events, end_cause, end_time, weight}`.
///
/// `weight` is a **probability** here, not a replica count: the retained
/// sequences are disjoint events, so the weights of the sequences that
/// collapse into one minimal sequence add up, and the total weight is the
/// result's lower bound. Dates are 0, since the exact driver never moves
/// the clock.
#[pyfunction]
fn exploration_minimal_sequences_json(result_json: &str) -> PyResult<String> {
    let result = parse_exploration(result_json)?;
    let minimal = analyse_sequences(result.to_sequences());
    serde_json::to_string(&minimal).map_err(|e| SimulationError::new_err(e.to_string()))
}

/// Quantify one study with one method and return the
/// `raichu.quantification` envelope as JSON (see `raichu_quantify`).
///
/// `study_json` is the study (`target`, `horizon`, optional `instants`,
/// `seed`, `threads`); `method` names the engine (`monte_carlo`, `exact`,
/// `discretised`, `cross_entropy`, `splitting`) and `settings_json` holds the settings that belong to it
/// (an object, `None` for every default). An unknown method, or a setting
/// of another method, is refused naming the valid ones before anything
/// runs. The GIL is released while the engine runs.
#[pyfunction]
#[pyo3(signature = (model_json, study_json, method, settings_json = None, allow_fmu_import = false, fmu_base_dir = None, require_parallel = false))]
#[allow(clippy::too_many_arguments)] // mirrors the Python keyword signature
fn quantify_json(
    py: Python<'_>,
    model_json: &str,
    study_json: &str,
    method: &str,
    settings_json: Option<&str>,
    allow_fmu_import: bool,
    fmu_base_dir: Option<&str>,
    require_parallel: bool,
) -> PyResult<String> {
    let model = Model::from_json(model_json)
        .map_err(|e| ModelError::new_err(format!("invalid model JSON: {e}")))?;
    let study: Study = serde_json::from_str(study_json)
        .map_err(|e| SimulationError::new_err(format!("invalid study: {e}")))?;
    let settings: serde_json::Value = match settings_json {
        None => serde_json::Value::Null,
        Some(text) => serde_json::from_str(text)
            .map_err(|e| SimulationError::new_err(format!("invalid method settings: {e}")))?,
    };
    let method = Method::from_parts(method, &settings)
        .map_err(|e| SimulationError::new_err(e.to_string()))?;
    py.detach(|| {
        let answer = if model.fmu_units.is_empty() {
            quantify_study(&model, &study, &method)
        } else {
            quantify_study_with_fmu(
                &model,
                &study,
                &method,
                Path::new(fmu_base_dir.unwrap_or(".")),
                allow_fmu_import,
                require_parallel,
            )
        };
        let envelope = answer.map_err(|e| match e {
            QuantifyError::Compile(e) => ModelError::new_err(e.to_string()),
            QuantifyError::Engine(e) => engine_error(e),
            other => SimulationError::new_err(other.to_string()),
        })?;
        envelope
            .to_json()
            .map_err(|e| SimulationError::new_err(e.to_string()))
    })
}

/// Check that `envelope_json` is a quantification envelope this engine
/// reads (format, version, and a method, probability kind and detail that
/// belong together); raise `SimulationError` naming what is wrong. The
/// validating half of `pyraichu.read_quantification`, which builds the
/// Python object from the text itself.
#[pyfunction]
fn validate_quantification(envelope_json: &str) -> PyResult<()> {
    read_quantification(envelope_json)
        .map(|_| ())
        .map_err(|e| SimulationError::new_err(e.to_string()))
}

/// Opaque checkpoint of an [`Interactive`] session's full trajectory
/// state (see `raichu_core::Snapshot`): produced by `Interactive.snapshot`
/// and reinstated by `Interactive.restore`. Held as a Python object; its
/// contents are engine-internal.
#[pyclass]
struct Snapshot {
    inner: CoreSnapshot,
}

/// A stateful, step-by-step interactive simulation over a compiled
/// model. Unlike the one-shot `simulate_json`, it advances one event at
/// a time under the caller's control (fire a *chosen* transition, force
/// its outcome branch, reschedule, snapshot / undo) and inspects the
/// state between events.
///
/// The borrowing `raichu_core::Engine` cannot outlive a single call, so
/// this object keeps the owned `CompiledModel` + a `Snapshot` and
/// rebuilds a throwaway engine (`Engine::from_snapshot`) per method:
/// exact restores make this identical to driving one persistent engine.
#[pyclass(unsendable)]
struct Interactive {
    model: CompiledModel,
    config: EngineConfig,
    snap: CoreSnapshot,
    host: Option<CoSimulationHost>,
}

impl Interactive {
    /// Rebuild the engine positioned at the current snapshot.
    ///
    /// This clones the whole trajectory state, so it is reserved for the
    /// **mutating** methods; the read-only accessors answer straight from
    /// `self.snap` (polling the state between steps must not cost a state
    /// clone per read).
    fn engine(&mut self) -> PyResult<Engine<'_>> {
        match self.host.as_mut() {
            Some(host) => {
                Engine::from_snapshot_with_host(&self.model, self.config.clone(), &self.snap, host)
                    .map_err(engine_error)
            }
            None => Engine::from_snapshot(&self.model, self.config.clone(), &self.snap)
                .map_err(engine_error),
        }
    }

    fn json<T: serde::Serialize + ?Sized>(value: &T) -> PyResult<String> {
        serde_json::to_string(value).map_err(|e| SimulationError::new_err(e.to_string()))
    }
}

#[pymethods]
impl Interactive {
    #[new]
    #[pyo3(signature = (model_json, t_max, journal = false, confluence_check = false, seed = 0, rng_stream = 0, flow = None, allow_fmu_import = false, fmu_base_dir = None, operator_control = false))]
    #[allow(clippy::too_many_arguments)] // mirrors the Python keyword signature
    fn new(
        model_json: &str,
        t_max: f64,
        journal: bool,
        confluence_check: bool,
        seed: u64,
        rng_stream: u64,
        flow: Option<FlowConfig>,
        allow_fmu_import: bool,
        fmu_base_dir: Option<&str>,
        operator_control: bool,
    ) -> PyResult<Self> {
        let model = parse_and_compile(model_json)?;
        let config = EngineConfig {
            t_max,
            journal,
            confluence_check,
            seed,
            rng_stream,
            flow: flow_policy(flow),
            allow_fmu_import,
            stochastic_dates: if operator_control {
                StochasticDates::Operator
            } else {
                StochasticDates::Drawn
            },
            ..EngineConfig::default()
        };
        let (snap, host) = if model.fmu_units.is_empty() {
            (
                Engine::new(&model, config.clone())
                    .map_err(engine_error)?
                    .snapshot()
                    .map_err(engine_error)?,
                None,
            )
        } else {
            let mut host = CoSimulationHost::prepare_authorized(
                &model,
                Path::new(fmu_base_dir.unwrap_or(".")),
                allow_fmu_import,
            )
            .map_err(engine_error)?;
            host.require_serializable_state().map_err(engine_error)?;
            let snap = Engine::new_with_host(&model, config.clone(), &mut host)
                .map_err(engine_error)?
                .try_snapshot()
                .map_err(engine_error)?;
            (snap, Some(host))
        };
        Ok(Interactive {
            model,
            config,
            snap,
            host,
        })
    }

    /// Current simulation time.
    #[getter]
    fn time(&self) -> f64 {
        self.snap.time()
    }

    /// JSON array of the currently-armed transitions
    /// (`{index, transition, kind, date}`), earliest first.
    ///
    /// The only reader that needs a rebuilt engine: locating an armed
    /// watched transition evaluates its guard against the live state.
    fn fireable(&mut self) -> PyResult<String> {
        Self::json(&self.engine()?.fireable())
    }

    /// Value of an attribute by qualified name (`component.attribute`),
    /// tagged-JSON encoded; `None` if unknown.
    fn attribute(&self, qualified: &str) -> PyResult<Option<String>> {
        self.snap
            .attribute(&self.model, qualified)
            .map(|v| Self::json(&v))
            .transpose()
    }

    /// Current state name of an automaton (`component.automaton`);
    /// `None` if unknown.
    fn state(&self, qualified: &str) -> Option<String> {
        self.snap.state(&self.model, qualified).map(str::to_owned)
    }

    /// JSON array of the events fired so far, chronological.
    fn history(&self) -> PyResult<String> {
        Self::json(self.snap.history())
    }

    /// Fire the armed transition `name`, optionally **forcing** its
    /// destination branch to the state `to` (bypassing the RNG /
    /// deterministic resolution). Returns the fired event as JSON.
    #[pyo3(signature = (name, to = None))]
    fn fire(&mut self, name: &str, to: Option<&str>) -> PyResult<String> {
        let (event, snap) = {
            let mut engine = self.engine()?;
            let event = match to {
                Some(to) => engine.fire_named_to(name, to),
                None => engine.fire_named(name),
            }
            .map_err(engine_error)?;
            let snap = engine.try_snapshot().map_err(engine_error)?;
            (event, snap)
        };
        self.snap = snap;
        Self::json(&event)
    }

    /// Advance to the next scheduled event (earliest-first, as a plain
    /// run would). Returns the fired event as JSON, or `None` at the
    /// horizon.
    fn step(&mut self) -> PyResult<Option<String>> {
        let (event, snap) = {
            let mut engine = self.engine()?;
            let event = engine.step().map_err(engine_error)?;
            let snap = engine.try_snapshot().map_err(engine_error)?;
            (event, snap)
        };
        self.snap = snap;
        event.map(|e| Self::json(&e)).transpose()
    }

    /// Advance through all events at or before `date` and stop at that date.
    fn advance_to(&mut self, date: f64) -> PyResult<()> {
        let snap = {
            let mut engine = self.engine()?;
            engine.advance_to(date).map_err(engine_error)?;
            engine.try_snapshot().map_err(engine_error)?
        };
        self.snap = snap;
        Ok(())
    }

    /// Advance without an unsolicited stochastic draw or branch choice.
    #[pyo3(signature = (date, max_events = 10000))]
    fn advance_operator_to(&mut self, date: f64, max_events: usize) -> PyResult<String> {
        let (outcome, snap) = {
            let mut engine = self.engine()?;
            let outcome = engine
                .advance_operator_to(date, max_events)
                .map_err(engine_error)?;
            let snap = engine.try_snapshot().map_err(engine_error)?;
            (outcome, snap)
        };
        self.snap = snap;
        Self::json(&outcome)
    }

    /// Set an attribute named in the caller's explicit input manifest.
    fn set_input(
        &mut self,
        qualified: &str,
        value_json: &str,
        allowed_inputs: Vec<String>,
    ) -> PyResult<()> {
        let value: raichu::raichu_expr::Value = serde_json::from_str(value_json)
            .map_err(|error| SimulationError::new_err(error.to_string()))?;
        let snap = {
            let mut engine = self.engine()?;
            engine
                .set_input(qualified, value, &allowed_inputs)
                .map_err(engine_error)?;
            engine.try_snapshot().map_err(engine_error)?
        };
        self.snap = snap;
        Ok(())
    }

    /// Override an armed transition's scheduled firing date (must be
    /// `>=` the current time).
    fn set_date(&mut self, name: &str, date: f64) -> PyResult<()> {
        let snap = {
            let mut engine = self.engine()?;
            engine.set_date(name, date).map_err(engine_error)?;
            engine.try_snapshot().map_err(engine_error)?
        };
        self.snap = snap;
        Ok(())
    }

    /// Reset the session to its initial state (`t = 0`, fresh RNG).
    fn reset(&mut self) -> PyResult<()> {
        let snap = {
            let mut engine = self.engine()?;
            engine.reset().map_err(engine_error)?;
            engine.try_snapshot().map_err(engine_error)?
        };
        self.snap = snap;
        Ok(())
    }

    /// Capture the full trajectory state as an opaque checkpoint.
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            inner: self.snap.clone(),
        }
    }

    /// Reinstate a previously captured checkpoint (undo).
    fn restore(&mut self, snap: &Snapshot) -> PyResult<()> {
        {
            let mut engine = self.engine()?;
            engine.try_restore(&snap.inner).map_err(engine_error)?;
        }
        self.snap = snap.inner.clone();
        Ok(())
    }
}

/// RAICHU engine bindings (private extension module).
#[pymodule]
fn _pyraichu(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", raichu::VERSION)?;
    module.add("MODEL_ENVELOPE_KEY", raichu::raichu_model::ENVELOPE_KEY)?;
    module.add("DEFAULT_CONFIDENCE", DEFAULT_CONFIDENCE)?;
    module.add(
        "MODEL_FORMAT_REVISION",
        raichu::raichu_model::FORMAT_REVISION,
    )?;
    module.add("ModelError", py.get_type::<ModelError>())?;
    module.add("SimulationError", py.get_type::<SimulationError>())?;
    module.add_function(wrap_pyfunction!(validate_model, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_fmu_export, module)?)?;
    module.add_function(wrap_pyfunction!(switching_loops_json, module)?)?;
    module.add_function(wrap_pyfunction!(unfed_triggers_json, module)?)?;
    module.add_function(wrap_pyfunction!(required_features, module)?)?;
    module.add_function(wrap_pyfunction!(seal_model, module)?)?;
    module.add_function(wrap_pyfunction!(simulate_json, module)?)?;
    module.add_function(wrap_pyfunction!(monte_carlo_json, module)?)?;
    module.add_function(wrap_pyfunction!(analyse_sequences_json, module)?)?;
    module.add_function(wrap_pyfunction!(run_sequences_json, module)?)?;
    module.add_function(wrap_pyfunction!(analyse_raw_sequences_json, module)?)?;
    module.add_function(wrap_pyfunction!(importance_json, module)?)?;
    module.add_function(wrap_pyfunction!(explore_json, module)?)?;
    module.add_function(wrap_pyfunction!(exploration_domain_json, module)?)?;
    module.add_function(wrap_pyfunction!(fault_tree_json, module)?)?;
    module.add_function(wrap_pyfunction!(fault_tree_quantify_json, module)?)?;
    module.add_function(wrap_pyfunction!(fault_tree_envelope_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_fault_tree_envelope, module)?)?;
    module.add_function(wrap_pyfunction!(validate_exploration, module)?)?;
    module.add_function(wrap_pyfunction!(quantify_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_quantification, module)?)?;
    module.add_function(wrap_pyfunction!(
        exploration_minimal_sequences_json,
        module
    )?)?;
    module.add_class::<Interactive>()?;
    module.add_class::<Snapshot>()?;
    module.add_class::<FlowConfig>()?;
    Ok(())
}
