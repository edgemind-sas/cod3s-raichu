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
use raichu::raichu_core::analyse as analyse_sequences;
use raichu::raichu_core::{
    CompiledModel, Engine, EngineConfig, Snapshot as CoreSnapshot, SolverParams,
};
use raichu::raichu_model::Model;
use raichu::raichu_montecarlo::{run as mc_run, run_sequences as mc_run_sequences, McConfig};

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
#[pyo3(signature = (model_json, nb_runs, t_max, samples, seed = 0, threads = None, quantiles = None, rtol = None, atol = None, max_step = None, tol_event = None, sub_samples = None, stop_at_targets = false))]
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
    stop_at_targets: bool,
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
            stop_at_targets,
        };
        let estimates =
            mc_run(&compiled, &config).map_err(|e| SimulationError::new_err(e.to_string()))?;
        serde_json::to_string(&estimates).map_err(|e| SimulationError::new_err(e.to_string()))
    })
}

/// Native minimal-sequence analysis: run `nb_runs` sequence-recording
/// replicas (target early-stop) and return the JSON of the minimal sequences
/// (group → filter-cycles → minimal), the RAMS output cod3s produces via its
/// `SequenceAnalyser`. Each minimal sequence is `{events, end_cause,
/// end_time, weight}` (weight = the trajectory count that collapsed into it).
#[pyfunction]
#[pyo3(signature = (model_json, nb_runs, t_max, seed = 0, threads = None))]
fn analyse_sequences_json(
    py: Python<'_>,
    model_json: &str,
    nb_runs: u64,
    t_max: f64,
    seed: u64,
    threads: Option<usize>,
) -> PyResult<String> {
    let compiled = parse_and_compile(model_json)?;
    py.detach(|| {
        let config = McConfig {
            nb_runs,
            seed,
            t_max,
            samples: Vec::new(),
            threads,
            quantiles: Vec::new(),
            ode: SolverParams::default(),
            stop_at_targets: false,
        };
        let raw = mc_run_sequences(&compiled, &config)
            .map_err(|e| SimulationError::new_err(e.to_string()))?;
        let minimal = analyse_sequences(raw);
        serde_json::to_string(&minimal).map_err(|e| SimulationError::new_err(e.to_string()))
    })
}

/// Opaque checkpoint of an [`Interactive`] session's full trajectory
/// state (see `raichu_core::Snapshot`) — produced by `Interactive.snapshot`
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
/// rebuilds a throwaway engine (`Engine::from_snapshot`) per method —
/// exact restores make this identical to driving one persistent engine.
#[pyclass]
struct Interactive {
    model: CompiledModel,
    config: EngineConfig,
    snap: CoreSnapshot,
}

impl Interactive {
    /// Rebuild the engine positioned at the current snapshot.
    fn engine(&self) -> Engine<'_> {
        Engine::from_snapshot(&self.model, self.config.clone(), &self.snap)
    }

    fn json<T: serde::Serialize + ?Sized>(value: &T) -> PyResult<String> {
        serde_json::to_string(value).map_err(|e| SimulationError::new_err(e.to_string()))
    }
}

#[pymethods]
impl Interactive {
    #[new]
    #[pyo3(signature = (model_json, t_max, journal = false, confluence_check = false, seed = 0, rng_stream = 0))]
    fn new(
        model_json: &str,
        t_max: f64,
        journal: bool,
        confluence_check: bool,
        seed: u64,
        rng_stream: u64,
    ) -> PyResult<Self> {
        let model = parse_and_compile(model_json)?;
        let config = EngineConfig {
            t_max,
            journal,
            confluence_check,
            seed,
            rng_stream,
            ..EngineConfig::default()
        };
        let snap = Engine::new(&model, config.clone())
            .map_err(|e| SimulationError::new_err(e.to_string()))?
            .snapshot();
        Ok(Interactive {
            model,
            config,
            snap,
        })
    }

    /// Current simulation time.
    #[getter]
    fn time(&self) -> f64 {
        self.engine().current_time()
    }

    /// JSON array of the currently-armed transitions
    /// (`{index, transition, kind, date}`), earliest first.
    fn fireable(&self) -> PyResult<String> {
        Self::json(&self.engine().fireable())
    }

    /// Value of an attribute by qualified name (`component.attribute`),
    /// tagged-JSON encoded; `None` if unknown.
    fn attribute(&self, qualified: &str) -> PyResult<Option<String>> {
        self.engine()
            .attribute(qualified)
            .map(|v| Self::json(&v))
            .transpose()
    }

    /// Current state name of an automaton (`component.automaton`);
    /// `None` if unknown.
    fn state(&self, qualified: &str) -> Option<String> {
        self.engine().state(qualified).map(str::to_owned)
    }

    /// JSON array of the events fired so far, chronological.
    fn history(&self) -> PyResult<String> {
        Self::json(self.engine().history())
    }

    /// Fire the armed transition `name`, optionally **forcing** its
    /// destination branch to the state `to` (bypassing the RNG /
    /// deterministic resolution). Returns the fired event as JSON.
    #[pyo3(signature = (name, to = None))]
    fn fire(&mut self, name: &str, to: Option<&str>) -> PyResult<String> {
        let mut engine = self.engine();
        let event = match to {
            Some(to) => engine.fire_named_to(name, to),
            None => engine.fire_named(name),
        }
        .map_err(|e| SimulationError::new_err(e.to_string()))?;
        self.snap = engine.snapshot();
        Self::json(&event)
    }

    /// Advance to the next scheduled event (earliest-first, as a plain
    /// run would). Returns the fired event as JSON, or `None` at the
    /// horizon.
    fn step(&mut self) -> PyResult<Option<String>> {
        let mut engine = self.engine();
        let event = engine
            .step()
            .map_err(|e| SimulationError::new_err(e.to_string()))?;
        self.snap = engine.snapshot();
        event.map(|e| Self::json(&e)).transpose()
    }

    /// Override an armed transition's scheduled firing date (must be
    /// `>=` the current time).
    fn set_date(&mut self, name: &str, date: f64) -> PyResult<()> {
        let mut engine = self.engine();
        engine
            .set_date(name, date)
            .map_err(|e| SimulationError::new_err(e.to_string()))?;
        self.snap = engine.snapshot();
        Ok(())
    }

    /// Reset the session to its initial state (`t = 0`, fresh RNG).
    fn reset(&mut self) -> PyResult<()> {
        let mut engine = self.engine();
        engine
            .reset()
            .map_err(|e| SimulationError::new_err(e.to_string()))?;
        self.snap = engine.snapshot();
        Ok(())
    }

    /// Capture the full trajectory state as an opaque checkpoint.
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            inner: self.snap.clone(),
        }
    }

    /// Reinstate a previously captured checkpoint (undo).
    fn restore(&mut self, snap: &Snapshot) {
        self.snap = snap.inner.clone();
    }
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
    module.add_function(wrap_pyfunction!(analyse_sequences_json, module)?)?;
    module.add_class::<Interactive>()?;
    module.add_class::<Snapshot>()?;
    Ok(())
}
