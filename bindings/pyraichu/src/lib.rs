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
use raichu::raichu_analysis::{
    minimal_sequences, BasicEvent, BasicEventSelection, ImportanceGroup, ObservedCondition,
    RawCorpusReader, RawCorpusWriter, RawHeader, RawObservation, RawSequencesError,
    SequenceReducer,
};
use raichu::raichu_core::{
    fault_tree as generate_fault_tree, fault_tree_for_targets, FaultTree, FaultTreeSettings,
};
use raichu::raichu_core::{
    CoSimulationHost, CompiledModel, Engine, EngineConfig, EngineError,
    FlowConfig as CoreFlowConfig, Sequence, Snapshot as CoreSnapshot, SolverParams,
    StochasticDates,
};
use raichu::raichu_explore::{
    exact_domain_report, explore_discretised, explore_discretised_with_fmu, explore_exact,
    read_exploration as read_exploration_result, Algorithm, Cutoffs, DiscretisedSettings,
    ExactSettings, ExplorationResult, Precision,
};
use raichu::raichu_expr::{AttrRef, CmpOp};
use raichu::raichu_fta::{
    fault_tree_envelope, fault_tree_structure, quantify as quantify_tree, read_fault_tree_envelope,
    read_fault_tree_structure, read_open_psa, read_open_psa_measure, Engine as FtaEngine,
    EnvelopeTop, FaultTreeMeasure, QuantifySettings, StructureGeneration, StructureSettings,
    StructureSource,
};
use raichu::raichu_model::Model;
use raichu::raichu_model::{Indicator, IndicatorTarget};
use raichu::raichu_montecarlo::{
    run as mc_run, run_importance as mc_run_importance,
    run_sequences_streamed as mc_run_sequences_streamed, run_with_fmu as mc_run_with_fmu, McConfig,
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
///
/// A campaign's replica failure keeps the class of what the replica
/// raised, so a caller catching `UnboundedRateError` catches it from a
/// campaign as from a single run; the message is the campaign's, and the
/// exception carries `replica`, `nb_runs` and `seed` as attributes, the
/// coordinates that replay the replica with `simulate`.
fn engine_error(error: EngineError) -> PyErr {
    let message = error.to_string();
    let raised = match error.cause() {
        EngineError::UnboundedRate {
            time,
            variable,
            rate,
            unbounded,
        } => UnboundedRateError::new_err((message, variable.clone(), *time, *rate, *unbounded)),
        _ => SimulationError::new_err(message),
    };
    if let EngineError::ReplicaFailed {
        replica,
        nb_runs,
        seed,
        ..
    } = &error
    {
        Python::attach(|py| {
            let value = raised.value(py);
            // Setting an attribute on a fresh exception instance cannot
            // fail short of memory exhaustion; the exception is raised
            // either way, so a failure here only loses the coordinates,
            // which the message also states.
            let _ = value.setattr("replica", *replica);
            let _ = value.setattr("nb_runs", *nb_runs);
            let _ = value.setattr("seed", *seed);
        });
    }
    raised
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

/// Per-group decisions of a model, as JSON: whether the minimal-sequence
/// reduction may cancel each cycle group, per target, with the path that
/// keeps it (see `raichu_core::cycle_groups`). Each entry gains a
/// `messages` list, one line per target the group is kept for.
#[pyfunction]
fn cycle_groups_json(model_json: &str) -> PyResult<String> {
    /// A decision with its one-line messages.
    #[derive(serde::Serialize)]
    struct Reported<'a> {
        #[serde(flatten)]
        decision: &'a raichu::raichu_core::CycleGroupDecision,
        messages: Vec<String>,
    }
    let compiled = parse_and_compile(model_json)?;
    // Read off the compiled model, where the analysis already ran.
    let found: Vec<Reported<'_>> = compiled
        .cycle_groups
        .iter()
        .map(|decision| Reported {
            decision,
            messages: decision.describe(),
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

/// A refused library, system or study document as `ModelError`.
fn refuse_document(error: raichu::raichu_library::ExpandError) -> PyErr {
    ModelError::new_err(error.to_string())
}

/// Parse library documents, refusing the first malformed one.
fn parse_libraries(texts: &[String]) -> PyResult<Vec<raichu::raichu_library::Library>> {
    texts
        .iter()
        .map(|text| raichu::raichu_library::parse_library(text))
        .collect::<Result<Vec<_>, _>>()
        .map_err(refuse_document)
}

/// Expand a component type library system (and optionally a study) into
/// a flat core model, returned as JSON:
/// `{"model": <sealed model document>, "provenance": {...},
/// "simulation": {...} | null, "warnings": [...]}`.
///
/// `libraries_json` holds the library documents the system may draw on;
/// the standard library is added when one of them imports it. A refused
/// document raises `ModelError` naming the document, the JSON Pointer of
/// the fault and what is wrong.
#[pyfunction]
#[pyo3(signature = (system_json, libraries_json, study_json = None))]
fn expand_documents_json(
    system_json: &str,
    libraries_json: Vec<String>,
    study_json: Option<&str>,
) -> PyResult<String> {
    use raichu::raichu_library as library;
    let refuse = refuse_document;
    let libraries = parse_libraries(&libraries_json)?;
    let system = library::parse_system(system_json).map_err(refuse)?;
    let study = study_json
        .map(library::parse_study)
        .transpose()
        .map_err(refuse)?;
    let expansion = library::expand(&libraries, &system, study.as_ref()).map_err(refuse)?;
    let model_json = expansion
        .model
        .to_json()
        .map_err(|e| ModelError::new_err(e.to_string()))?;
    let model: serde_json::Value =
        serde_json::from_str(&model_json).map_err(|e| ModelError::new_err(e.to_string()))?;
    let result = serde_json::json!({
        "model": model,
        "provenance": expansion.provenance.elements,
        "simulation": expansion.simulation,
        "warnings": expansion.warnings,
    });
    Ok(result.to_string())
}

/// Check every type of a library without a system, returned as JSON: one
/// entry per type with its status (`valid`, `invalid`, `unchecked`), the
/// errors of an invalid type and the required parameters of an unchecked
/// one.
#[pyfunction]
#[pyo3(signature = (library_json, libraries_json = Vec::new()))]
fn check_library_json(library_json: &str, libraries_json: Vec<String>) -> PyResult<String> {
    use raichu::raichu_library as library;
    let checked = library::parse_library(library_json).map_err(refuse_document)?;
    let others = parse_libraries(&libraries_json)?;
    let report: Vec<serde_json::Value> = library::check_library(&checked, &others)
        .into_iter()
        .map(|(name, outcome)| match outcome {
            library::TypeCheck::Valid => serde_json::json!({"type": name, "status": "valid"}),
            library::TypeCheck::Invalid(errors) => serde_json::json!({
                "type": name,
                "status": "invalid",
                "errors": errors.iter().map(ToString::to_string).collect::<Vec<_>>(),
            }),
            library::TypeCheck::Unchecked(parameters) => serde_json::json!({
                "type": name,
                "status": "unchecked",
                "parameters": parameters,
            }),
        })
        .collect();
    Ok(serde_json::Value::Array(report).to_string())
}

/// The evaluation order derived from what each explicit step reads, as a
/// JSON list of `{"component", "attribute"}` (or `null` for a model with
/// no explicit step): the order the expander gives every expanded model.
///
/// `model_json` is a model document, enveloped or bare; a bare body may
/// use any construct, since only its read relation is looked at.
#[pyfunction]
fn derive_evaluation_order_json(model_json: &str) -> PyResult<String> {
    let sealed = Model::seal_json(model_json)
        .map_err(|e| ModelError::new_err(format!("invalid model JSON: {e}")))?;
    let model = Model::from_json(&sealed)
        .map_err(|e| ModelError::new_err(format!("invalid model JSON: {e}")))?;
    let order = raichu::raichu_library::derive_evaluation_order(&model)
        .map_err(|e| ModelError::new_err(e.to_string()))?;
    serde_json::to_string(&order).map_err(|e| ModelError::new_err(e.to_string()))
}

/// The standard library `std` shipped with this engine, as a library
/// document.
#[pyfunction]
fn standard_library_json() -> PyResult<String> {
    use raichu::raichu_library as library;
    let std = library::standard_library().map_err(|e| ModelError::new_err(e.to_string()))?;
    Ok(library::document::library_document(&std).to_string())
}

/// Run a deterministic simulation and return the full result
/// (events, indicator series, dense samples, causal journal,
/// provenance) as JSON.
///
/// `flow` is an optional [`FlowConfig`] overriding the convergence
/// policy of the continuous flow resolution (engine defaults when
/// omitted). The `rtol`/`atol`/`max_step`/`tol_event`/`sub_samples`/
/// `event_resolution` keywords and `stop_at_targets` are the ones
/// [`monte_carlo_json`] takes, so one replica of a campaign replays here
/// under the campaign's own settings.
///
/// The GIL is released while the engine runs.
#[pyfunction]
#[pyo3(signature = (model_json, t_max, journal = false, confluence_check = false, samples = None, seed = 0, rng_stream = 0, flow = None, max_transition_firings = None, max_flow_restarts = None, allow_fmu_import = false, fmu_base_dir = None, rtol = None, atol = None, max_step = None, tol_event = None, sub_samples = None, event_resolution = None, stop_at_targets = false))]
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
    rtol: Option<f64>,
    atol: Option<f64>,
    max_step: Option<f64>,
    tol_event: Option<f64>,
    sub_samples: Option<usize>,
    event_resolution: Option<f64>,
    stop_at_targets: bool,
) -> PyResult<String> {
    let ode = solver_params(
        rtol,
        atol,
        max_step,
        tol_event,
        sub_samples,
        event_resolution,
    )?;
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
            ode,
            stop_at_targets,
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

/// The ODE-backend parameters a caller overrides by keyword, over the
/// engine defaults. `simulate` and `monte_carlo` take the same set, so a
/// campaign's replica replays alone under the campaign's own solver
/// settings.
fn solver_params(
    rtol: Option<f64>,
    atol: Option<f64>,
    max_step: Option<f64>,
    tol_event: Option<f64>,
    sub_samples: Option<usize>,
    event_resolution: Option<f64>,
) -> PyResult<SolverParams> {
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
    // Refused here, before the model is compiled, rather than per
    // replica: the engine checks the same domains when it is built.
    ode.validate()
        .map_err(|error| SimulationError::new_err(error.to_string()))?;
    Ok(ode)
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
    let ode = solver_params(
        rtol,
        atol,
        max_step,
        tol_event,
        sub_samples,
        event_resolution,
    )?;
    let compiled = parse_and_compile(model_json)?;
    let flow = flow_policy(flow);
    py.detach(|| {
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
        // JSON has no spelling for a NaN or an infinity: refused here
        // rather than written as a `null` no reader takes back.
        estimates
            .check_finite()
            .map_err(|e| SimulationError::new_err(e.to_string()))?;
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
///
/// With `report`, the JSON is `{"minimal": [...], "reduction": {...}}`
/// instead, the reduction report saying which trajectories kept their
/// cycles because cancelling them would leave a feared event without a
/// cause.
#[pyfunction]
#[pyo3(signature = (model_json, nb_runs, t_max, seed = 0, threads = None, flow = None, report = false))]
#[allow(clippy::too_many_arguments)] // mirrors the Python keyword signature
fn analyse_sequences_json(
    py: Python<'_>,
    model_json: &str,
    nb_runs: u64,
    t_max: f64,
    seed: u64,
    threads: Option<usize>,
    flow: Option<FlowConfig>,
    report: bool,
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
        let mut reducer = SequenceReducer::with_cycle_groups(compiled.cycle_groups.clone());
        mc_run_sequences_streamed(&compiled, &config, &[], |replica, sequence, _| {
            reducer.push_indexed(replica, sequence);
            Ok::<_, EngineError>(())
        })
        .map_err(engine_error)?;
        let reduction = report
            .then(|| serde_json::to_value(reducer.report()))
            .transpose()
            .map_err(|e| SimulationError::new_err(e.to_string()))?;
        minimal_json(&minimal_sequences(reducer.cleaned()), reduction)
    })
}

/// Minimal sequences as JSON: the bare array, or, given a reduction report,
/// the envelope `{"minimal": [...], "reduction": {...}}`.
fn minimal_json(minimal: &[Sequence], reduction: Option<serde_json::Value>) -> PyResult<String> {
    let json = match reduction {
        Some(reduction) => serde_json::to_string(&serde_json::json!({
            "minimal": minimal,
            "reduction": reduction,
        })),
        None => serde_json::to_string(minimal),
    };
    json.map_err(|e| SimulationError::new_err(e.to_string()))
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

/// What a sequence campaign or a corpus read can fail with, while the
/// trajectories go by.
enum CorpusError {
    Engine(EngineError),
    Raw(RawSequencesError),
}

impl From<EngineError> for CorpusError {
    fn from(error: EngineError) -> Self {
        CorpusError::Engine(error)
    }
}

impl From<RawSequencesError> for CorpusError {
    fn from(error: RawSequencesError) -> Self {
        CorpusError::Raw(error)
    }
}

fn corpus_error(error: CorpusError) -> PyErr {
    match error {
        CorpusError::Engine(error) => engine_error(error),
        CorpusError::Raw(error) => SimulationError::new_err(error.to_string()),
    }
}

/// A condition resolved against a corpus header.
struct ResolvedCondition {
    condition: ObservedCondition,
    /// The position of the observation it reads in a trajectory's
    /// observed values.
    column: usize,
    /// The comparison as it was spelled, for the report.
    op_symbol: String,
}

/// The reduction of a corpus to its two levels, fed one trajectory at a
/// time, keeping only the trajectories a condition holds on when one is
/// given. Holds the distinct cleaned paths and two counts, never the
/// trajectories.
struct ConditionalReduction {
    reducer: SequenceReducer,
    condition: Option<ResolvedCondition>,
    total: u64,
    kept: u64,
}

impl ConditionalReduction {
    /// A reduction of the corpus `header` describes; refuses a condition
    /// on an observation it does not carry, before any trajectory.
    fn new(header: &RawHeader, condition: Option<ConditionArg>) -> PyResult<Self> {
        let condition = match condition {
            None => None,
            Some(condition) => {
                let symbol = condition.1.clone();
                let condition = observed_condition(condition)?;
                let position = header
                    .observation_index(&condition.observation)
                    .map_err(|e| SimulationError::new_err(e.to_string()))?;
                Some(ResolvedCondition {
                    condition,
                    column: position,
                    op_symbol: symbol,
                })
            }
        };
        Ok(ConditionalReduction {
            reducer: header.reducer(),
            condition,
            total: 0,
            kept: 0,
        })
    }

    fn push(&mut self, sequence: Sequence, observed: &[f64]) {
        // Trajectories arrive in replica order, so the count so far is the
        // replica index, which the reduction report names.
        let replica = self.total;
        self.total += 1;
        let holds = self
            .condition
            .as_ref()
            .is_none_or(|resolved| resolved.condition.holds(observed[resolved.column]));
        if holds {
            self.kept += 1;
            self.reducer.push_indexed(replica, sequence);
        }
    }

    /// The two levels as `{"cleaned": [...], "minimal": [...]}`, the
    /// `reduction` report, plus a `condition` report when one was given.
    fn finish(self) -> PyResult<serde_json::Value> {
        let to_value = |v: &[Sequence]| {
            serde_json::to_value(v).map_err(|e| SimulationError::new_err(e.to_string()))
        };
        let reduction = serde_json::to_value(self.reducer.report())
            .map_err(|e| SimulationError::new_err(e.to_string()))?;
        let cleaned = self.reducer.cleaned();
        // Serialised before the minimal level consumes it: no copy is made.
        let cleaned_value = to_value(&cleaned)?;
        let minimal = minimal_sequences(cleaned);
        let mut levels = serde_json::json!({
            "cleaned": cleaned_value,
            "minimal": to_value(&minimal)?,
            "reduction": reduction,
        });
        if let Some(ResolvedCondition {
            condition,
            op_symbol: symbol,
            ..
        }) = self.condition
        {
            levels["condition"] = serde_json::json!({
                "observation": condition.observation,
                "op": symbol,
                "value": condition.value,
                "total_trajectories": self.total,
                "kept_trajectories": self.kept,
            });
        }
        Ok(levels)
    }
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
/// Each trajectory is written and reduced as the campaign produces it, and
/// dropped: memory holds one chunk of trajectories and the distinct
/// cleaned paths, whatever `nb_runs` is, and the raw corpus never becomes
/// Python objects.
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
        let header = RawHeader::new(
            raichu::VERSION,
            &model.name,
            seed,
            nb_runs,
            t_max,
            compiled.targets.iter().map(|t| t.name.clone()).collect(),
        )
        .with_cycle_groups(compiled.cycle_groups.clone())
        .with_observations(
            asked
                .iter()
                .map(|observation| RawObservation {
                    name: observation.indicator.clone(),
                    time: observation.read_at(t_max),
                })
                .collect(),
        );
        let mut reduction = ConditionalReduction::new(&header, condition)?;
        // The corpus is written beside `raw_path` and moved onto it once the
        // campaign succeeded: a campaign that fails leaves whatever
        // `raw_path` held untouched, and no partial corpus behind.
        let mut writer = match &raw_path {
            None => None,
            Some(path) => {
                let partial = partial_path(path);
                let file = std::fs::File::create(&partial).map_err(|e| {
                    SimulationError::new_err(format!("cannot create {}: {e}", partial.display()))
                })?;
                let writer = RawCorpusWriter::new(std::io::BufWriter::new(file), &header)
                    .map_err(|e| SimulationError::new_err(e.to_string()));
                match writer {
                    Ok(writer) => Some((writer, partial)),
                    Err(e) => {
                        let _ = std::fs::remove_file(&partial);
                        return Err(e);
                    }
                }
            }
        };
        let campaign =
            mc_run_sequences_streamed(&compiled, &config, &asked, |_, sequence, observed| {
                if let Some((writer, _)) = writer.as_mut() {
                    writer.push(&sequence, &observed)?;
                }
                reduction.push(sequence, &observed);
                Ok::<_, CorpusError>(())
            })
            .map_err(corpus_error);
        if let (Some((writer, partial)), Some(path)) = (writer, &raw_path) {
            let written = campaign.and_then(|()| {
                writer
                    .finish()
                    .map_err(|e| SimulationError::new_err(e.to_string()))?;
                std::fs::rename(&partial, path).map_err(|e| {
                    SimulationError::new_err(format!(
                        "cannot move the corpus to {}: {e}",
                        path.display()
                    ))
                })
            });
            if written.is_err() {
                let _ = std::fs::remove_file(&partial);
            }
            written?;
        } else {
            campaign?;
        }
        Ok(reduction.finish()?.to_string())
    })
}

/// The file a raw corpus is written to before it is moved onto `path`: the
/// same name with `.partial` appended, in the same directory, so the move
/// is a rename within one file system.
fn partial_path(path: &std::path::Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".partial");
    std::path::PathBuf::from(name)
}

/// Read a raw corpus written by [`run_sequences_json`] and reduce it again:
/// the JSON `{"header": {...}, "cleaned": [...], "minimal": [...]}`, reduced
/// from the trajectories `condition` holds on when one is given (plus a
/// `condition` report). The corpus is read one line at a time, so a corpus
/// of any size reduces in the memory of its distinct cleaned paths.
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
        let corpus = RawCorpusReader::new(std::io::BufReader::new(file))
            .map_err(|e| SimulationError::new_err(e.to_string()))?;
        let header = corpus.header().clone();
        let mut reduction = ConditionalReduction::new(&header, condition)?;
        for trajectory in corpus {
            let (sequence, observed) =
                trajectory.map_err(|e| SimulationError::new_err(e.to_string()))?;
            reduction.push(sequence, &observed);
        }
        let mut levels = reduction.finish()?;
        levels["header"] =
            serde_json::to_value(&header).map_err(|e| SimulationError::new_err(e.to_string()))?;
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
///
/// `options_json` carries the basic-event selection and the groups:
/// `{"basic_events": "failures" | "monitored" | [{obj, automaton, attr}],
/// "groups": [{name, events: [{obj, automaton, attr}]}]}`, both optional
/// (the declared failure states, no group).
#[pyfunction]
#[pyo3(signature = (model_json, nb_runs, t_max, instants, target = None, seed = 0, threads = None, flow = None, options_json = None))]
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
    options_json: Option<&str>,
) -> PyResult<String> {
    let (selection, groups) = importance_options(options_json)?;
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
        let analysis = mc_run_importance(&compiled, &config, target.as_deref(), &selection, groups)
            .map_err(engine_error)?;
        serde_json::to_string(&analysis).map_err(|e| SimulationError::new_err(e.to_string()))
    })
}

/// The basic-event selection and the groups of an importance analysis,
/// from the options document [`importance_json`] documents.
fn importance_options(
    options_json: Option<&str>,
) -> PyResult<(BasicEventSelection, Vec<ImportanceGroup>)> {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum Selection {
        Named(String),
        Listed(Vec<BasicEvent>),
    }
    #[derive(serde::Deserialize, Default)]
    #[serde(deny_unknown_fields)]
    struct Options {
        #[serde(default)]
        basic_events: Option<Selection>,
        #[serde(default)]
        groups: Vec<ImportanceGroup>,
    }
    let refused =
        |detail: String| SimulationError::new_err(format!("importance options: {detail}"));
    let options: Options = match options_json {
        Some(json) => serde_json::from_str(json).map_err(|e| refused(e.to_string()))?,
        None => Options::default(),
    };
    let selection = match options.basic_events {
        None => BasicEventSelection::Failures,
        Some(Selection::Named(name)) => match name.as_str() {
            "failures" => BasicEventSelection::Failures,
            "monitored" => BasicEventSelection::Monitored,
            other => {
                return Err(refused(format!(
                    "`basic_events` is `{other}`: expected `failures`, `monitored` or a list"
                )))
            }
        },
        Some(Selection::Listed(events)) => BasicEventSelection::Listed(events),
    };
    Ok((selection, options.groups))
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
) -> PyResult<(FaultTree, EnvelopeTop, FaultTreeSettings)> {
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
    Ok((tree, top, settings))
}

/// Generate the fault tree explaining `top_json`, or the targets named by
/// `targets_json` (see `generate_tree`). Answers `{"tree": {top,
/// basic_events, warnings}, "minimal_cut_sets": [[event index, ...]],
/// "open_psa": "<document>"}`, the document naming `measure`
/// (`probability_without_repair` or `unavailability`), the one it is
/// meant to be quantified in.
#[pyfunction]
#[pyo3(signature = (model_json, top_json = None, profile_json = None, max_nodes = None, cut_set_limit = 100_000, name = "fault_tree", targets_json = None, cut_sets = true, measure = "probability_without_repair"))]
#[allow(clippy::too_many_arguments)]
fn fault_tree_json(
    model_json: &str,
    top_json: Option<&str>,
    profile_json: Option<&str>,
    max_nodes: Option<usize>,
    cut_set_limit: usize,
    name: &str,
    targets_json: Option<&str>,
    cut_sets: bool,
    measure: &str,
) -> PyResult<String> {
    let measure = fta_measure(measure)?;
    let (tree, top, generation) =
        generate_tree(model_json, top_json, targets_json, profile_json, max_nodes)?;
    let source_model =
        Model::from_json(model_json).map_err(|e| SimulationError::new_err(e.to_string()))?;
    let model_hash = raichu::raichu_quantify::model_content_hash(&source_model)
        .map_err(|e| SimulationError::new_err(e.to_string()))?;
    let profile = generation
        .profile
        .iter()
        .map(|(name, value)| serde_json::to_value(value).map(|value| (name.clone(), value)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| SimulationError::new_err(e.to_string()))?;
    let structure = fault_tree_structure(
        &tree,
        name,
        top,
        &StructureSettings {
            cut_sets,
            cut_set_limit,
        },
        StructureSource::Model {
            model_hash,
            generation: StructureGeneration {
                max_nodes: generation
                    .max_nodes
                    .unwrap_or(raichu::raichu_core::DEFAULT_MAX_NODES),
                profile,
            },
        },
    )
    .map_err(|e| SimulationError::new_err(e.to_string()))?;
    let cuts = structure.minimal_cut_sets.as_ref().map(|sets| {
        // Keep the existing index-based binding representation without a
        // repeated scan of the event table for every cut member.
        let indices: std::collections::HashMap<&str, usize> = tree
            .basic_events
            .iter()
            .enumerate()
            .map(|(i, event)| (event.name.as_str(), i))
            .collect();
        let mut indexed: Vec<Vec<usize>> = sets
            .iter()
            .map(|cut| {
                let mut members: Vec<usize> = cut
                    .iter()
                    .filter_map(|name| indices.get(name.as_str()).copied())
                    .collect();
                members.sort();
                members
            })
            .collect();
        indexed.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        indexed
    });
    let answer = serde_json::json!({
        "tree": tree,
        "minimal_cut_sets": cuts,
        "open_psa": tree.to_open_psa(name, measure),
        "structure": structure,
    });
    serde_json::to_string(&answer).map_err(|e| SimulationError::new_err(e.to_string()))
}

/// Validate the versioned structural result without numerical quantification.
#[pyfunction]
fn validate_fault_tree_structure(structure_json: &str) -> PyResult<()> {
    read_fault_tree_structure(structure_json)
        .map(|_| ())
        .map_err(|e| SimulationError::new_err(e.to_string()))
}

fn fta_measure(measure: &str) -> PyResult<FaultTreeMeasure> {
    FaultTreeMeasure::from_name(measure).ok_or_else(|| {
        SimulationError::new_err(format!(
            "fault tree: measure `{measure}` is not one of probability_without_repair, \
             unavailability"
        ))
    })
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
/// cut sets and the importance measures are computed), in `measure`.
/// Answers the `raichu.fault_tree` envelope as JSON. The GIL is released
/// while the tree is quantified.
#[pyfunction]
#[pyo3(signature = (model_json, mission_times, top_json = None, targets_json = None, profile_json = None, max_nodes = None, name = "fault_tree", max_bdd_nodes = 10_000_000, cut_set_limit = 100_000, cut_sets = true, engine = "auto", max_order = None, min_cut_probability = 0.0, max_cut_sets = 1_000_000, max_expansions = 100_000_000, measure = "probability_without_repair"))]
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
    measure: &str,
) -> PyResult<String> {
    let engine = fta_engine(engine)?;
    let measure = fta_measure(measure)?;
    if !(0.0..=1.0).contains(&min_cut_probability) {
        return Err(SimulationError::new_err(format!(
            "fault tree: min_cut_probability {min_cut_probability} is outside [0, 1]"
        )));
    }
    let (tree, top, _) =
        generate_tree(model_json, top_json, targets_json, profile_json, max_nodes)?;
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
        measure,
        // The envelope values every gate itself.
        gate_probabilities: true,
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
/// document has several unreferenced ones. `measure` is what each basic
/// event's probability measures; without it, the measure the document
/// names, else the probability without repair. Answers the quantification
/// with `"tree"` (the document's name), `"events"` (the basic-event
/// names, which the indices of the result refer to) and `"measure"`.
#[pyfunction]
#[pyo3(signature = (open_psa, top = None, mission_time = None, max_bdd_nodes = 10_000_000, cut_set_limit = 100_000, cut_sets = true, engine = "auto", max_order = None, min_cut_probability = 0.0, max_cut_sets = 1_000_000, max_expansions = 100_000_000, measure = None))]
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
    measure: Option<&str>,
) -> PyResult<String> {
    let engine = fta_engine(engine)?;
    if !(0.0..=1.0).contains(&min_cut_probability) {
        return Err(SimulationError::new_err(format!(
            "fault tree: min_cut_probability {min_cut_probability} is outside [0, 1]"
        )));
    }
    let measure = match measure {
        Some(name) => fta_measure(name)?,
        None => read_open_psa_measure(open_psa)
            .map_err(|e| SimulationError::new_err(e.to_string()))?
            .unwrap_or_default(),
    };
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
        measure,
        gate_probabilities: false,
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
        measure: &'static str,
        result: &'a raichu::raichu_fta::Quantification,
    }
    let answer = Answer {
        measure: measure.as_str(),
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
///
/// With `report`, the JSON is `{"minimal": [...], "reduction": {...}}`:
/// the reduction report names the retained sequences whose cycles were
/// kept by their index, and adds `contradicted_share`, their probability
/// mass as a share of the lower bound.
#[pyfunction]
#[pyo3(signature = (result_json, report = false))]
fn exploration_minimal_sequences_json(result_json: &str, report: bool) -> PyResult<String> {
    let result = parse_exploration(result_json)?;
    let (minimal, reduction) = result.minimal_sequences();
    if !report {
        return minimal_json(&minimal, None);
    }
    let share = if result.lower > 0.0 {
        reduction.contradicted_weight / result.lower
    } else {
        0.0
    };
    let mut reduction =
        serde_json::to_value(&reduction).map_err(|e| SimulationError::new_err(e.to_string()))?;
    reduction["contradicted_share"] = serde_json::json!(share);
    minimal_json(&minimal, Some(reduction))
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

/// A compiled occurrence law as the model schema spells it, for reading a
/// deferred transition back.
fn law_json(law: &raichu::raichu_core::compile::CLaw) -> serde_json::Value {
    use raichu::raichu_core::compile::CLaw;
    match law {
        CLaw::Delay(time) => serde_json::json!({"distrib": "delay", "time": time}),
        CLaw::Inst(probs) => serde_json::json!({"distrib": "inst", "probs": probs}),
        CLaw::Watched { .. } => serde_json::json!({"distrib": "watched"}),
        CLaw::Exp(rate) => serde_json::json!({"distrib": "exp", "rate": rate}),
        CLaw::ExpVar { .. } => serde_json::json!({"distrib": "exp", "rate_expr": true}),
        CLaw::Weibull(shape, scale) => {
            serde_json::json!({"distrib": "weibull", "shape": shape, "scale": scale})
        }
        CLaw::Lognormal(mu, sigma) => {
            serde_json::json!({"distrib": "lognormal", "mu": mu, "sigma": sigma})
        }
        CLaw::Gamma(shape, scale) => {
            serde_json::json!({"distrib": "gamma", "shape": shape, "scale": scale})
        }
        CLaw::Uniform(low, high) => {
            serde_json::json!({"distrib": "uniform", "low": low, "high": high})
        }
        CLaw::Empirical(points) => serde_json::json!({"distrib": "empirical", "points": points}),
    }
}

#[pymethods]
impl Interactive {
    #[new]
    #[pyo3(signature = (model_json, t_max, journal = false, confluence_check = false, seed = 0, rng_stream = 0, flow = None, allow_fmu_import = false, fmu_base_dir = None, operator_control = false, deferred_draws = false))]
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
        deferred_draws: bool,
    ) -> PyResult<Self> {
        if operator_control && deferred_draws {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "deferred_draws and operator_control exclude each other: a deferred \
                 session arms stochastic transitions without a date for the caller to \
                 fire, an operator session lets the operator choose among armed ones",
            ));
        }
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
            } else if deferred_draws {
                StochasticDates::Deferred
            } else {
                StochasticDates::Drawn
            },
            // A deferred session leaves every random outcome to the
            // caller, the branch of an instantaneous choice included.
            deferred_choices: deferred_draws,
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

    /// JSON array of the causal journal recorded so far, in the record
    /// shape `simulate(..., journal=True)` returns. A session opened
    /// without `journal=True` records none, and says so.
    fn journal(&self) -> PyResult<String> {
        if !self.config.journal {
            return Err(SimulationError::new_err(
                "this session records no causal journal: open it with journal=True",
            ));
        }
        Self::json(self.snap.journal())
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

    /// JSON array of the stochastic transitions armed without a date in a
    /// session opened with `deferred_draws=True`
    /// (`{index, transition, law, armed_at, age, paused, hazard}`); empty in
    /// any other session. `law` restates the compiled occurrence law with
    /// its parameters, so a caller can check the law it is about to fire.
    fn deferred(&mut self) -> PyResult<String> {
        let armed: Vec<serde_json::Value> = self
            .engine()?
            .deferred()
            .into_iter()
            .map(|d| {
                serde_json::json!({
                    "index": d.index,
                    "transition": d.transition,
                    "law": law_json(&d.law),
                    "armed_at": d.armed_at,
                    "age": d.age,
                    "paused": d.paused,
                    "hazard": d.hazard,
                })
            })
            .collect();
        Self::json(&armed)
    }

    /// Fire the deferred transition `name` at `date` (`>=` the current
    /// time), the deterministic events before it firing on the way;
    /// `branch` forces the destination branch of a multi-target law.
    /// Returns the fired event as JSON.
    #[pyo3(signature = (name, date, branch = None))]
    fn fire_deferred_at(
        &mut self,
        name: &str,
        date: f64,
        branch: Option<usize>,
    ) -> PyResult<String> {
        let (event, snap) = {
            let mut engine = self.engine()?;
            let event = engine
                .fire_deferred_named_at(name, date, branch)
                .map_err(engine_error)?;
            let snap = engine.try_snapshot().map_err(engine_error)?;
            (event, snap)
        };
        self.snap = snap;
        Self::json(&event)
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
    module.add_function(wrap_pyfunction!(cycle_groups_json, module)?)?;
    module.add_function(wrap_pyfunction!(unfed_triggers_json, module)?)?;
    module.add_function(wrap_pyfunction!(required_features, module)?)?;
    module.add_function(wrap_pyfunction!(seal_model, module)?)?;
    module.add_function(wrap_pyfunction!(expand_documents_json, module)?)?;
    module.add_function(wrap_pyfunction!(check_library_json, module)?)?;
    module.add_function(wrap_pyfunction!(derive_evaluation_order_json, module)?)?;
    module.add_function(wrap_pyfunction!(standard_library_json, module)?)?;
    module.add_function(wrap_pyfunction!(simulate_json, module)?)?;
    module.add_function(wrap_pyfunction!(monte_carlo_json, module)?)?;
    module.add_function(wrap_pyfunction!(analyse_sequences_json, module)?)?;
    module.add_function(wrap_pyfunction!(run_sequences_json, module)?)?;
    module.add_function(wrap_pyfunction!(analyse_raw_sequences_json, module)?)?;
    module.add_function(wrap_pyfunction!(importance_json, module)?)?;
    module.add_function(wrap_pyfunction!(explore_json, module)?)?;
    module.add_function(wrap_pyfunction!(exploration_domain_json, module)?)?;
    module.add_function(wrap_pyfunction!(fault_tree_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_fault_tree_structure, module)?)?;
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
