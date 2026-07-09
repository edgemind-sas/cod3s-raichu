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
use pyo3::prelude::*;
use raichu::raichu_core::{CompiledModel, Engine, EngineConfig, SolverParams};
use raichu::raichu_model::Model;
use raichu::raichu_montecarlo::{run as mc_run, McConfig};

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

/// Run a deterministic simulation and return the full result
/// (events, indicator series, dense samples, causal journal,
/// provenance) as JSON.
///
/// The GIL is released while the engine runs.
#[pyfunction]
#[pyo3(signature = (model_json, t_max, journal = false, confluence_check = false, samples = None, seed = 0, rng_stream = 0))]
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
) -> PyResult<String> {
    let compiled = parse_and_compile(model_json)?;
    py.detach(|| {
        let config = EngineConfig {
            t_max,
            journal,
            confluence_check,
            samples: samples.unwrap_or_default(),
            seed,
            rng_stream,
            ..EngineConfig::default()
        };
        let engine =
            Engine::new(&compiled, config).map_err(|e| SimulationError::new_err(e.to_string()))?;
        let result = engine
            .run()
            .map_err(|e| SimulationError::new_err(e.to_string()))?;
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
/// when omitted) — the knobs of the tolerance-parity experiments.
#[pyfunction]
#[pyo3(signature = (model_json, nb_runs, t_max, samples, seed = 0, threads = None, quantiles = None, rtol = None, atol = None, max_step = None, tol_event = None, sub_samples = None))]
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
    rtol: Option<f64>,
    atol: Option<f64>,
    max_step: Option<f64>,
    tol_event: Option<f64>,
    sub_samples: Option<usize>,
) -> PyResult<String> {
    let compiled = parse_and_compile(model_json)?;
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
        let config = McConfig {
            nb_runs,
            seed,
            t_max,
            samples,
            threads,
            quantiles: quantiles.unwrap_or_default(),
            ode,
        };
        let estimates =
            mc_run(&compiled, &config).map_err(|e| SimulationError::new_err(e.to_string()))?;
        serde_json::to_string(&estimates).map_err(|e| SimulationError::new_err(e.to_string()))
    })
}

/// RAICHU engine bindings (private extension module).
#[pymodule]
fn _pyraichu(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", raichu::VERSION)?;
    module.add("ModelError", py.get_type::<ModelError>())?;
    module.add("SimulationError", py.get_type::<SimulationError>())?;
    module.add_function(wrap_pyfunction!(validate_model, module)?)?;
    module.add_function(wrap_pyfunction!(simulate_json, module)?)?;
    module.add_function(wrap_pyfunction!(monte_carlo_json, module)?)?;
    Ok(())
}
