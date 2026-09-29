use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CString};

use raichu_core::{CompiledModel, Engine, EngineConfig, Snapshot};
use raichu_expr::Value;
use self_cell::self_cell;

use crate::description::{inspect, token, ExportManifest, Variable};

self_cell!(
    struct EngineCell {
        owner: CompiledModel,
        #[not_covariant]
        dependent: Engine,
    }
);

pub(crate) type Logger = unsafe extern "C" fn(*mut c_void, c_int, *const c_char, *const c_char);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    Instantiated,
    Initialization,
    Step,
    Terminated,
    Error,
}

pub(crate) struct Instance {
    pub(crate) token: String,
    pub(crate) manifest: ExportManifest,
    pub(crate) variables: Vec<Variable>,
    compiled: Option<CompiledModel>,
    engine: Option<EngineCell>,
    pub(crate) phase: Phase,
    pub(crate) seed: u64,
    pub(crate) rng_stream: u64,
    start_time: f64,
    stop_time: Option<f64>,
    pending: HashMap<String, Value>,
    pub(crate) logger: Option<Logger>,
    pub(crate) environment: *mut c_void,
    pub(crate) logging_on: bool,
    states: HashMap<usize, Box<FmuState>>,
}

struct FmuState {
    trajectory: Snapshot,
    phase: Phase,
    seed: u64,
    rng_stream: u64,
    start_time: f64,
    stop_time: Option<f64>,
    pending: HashMap<String, Value>,
}

impl Instance {
    pub(crate) fn new(
        document: &str,
        logger: Option<Logger>,
        environment: *mut c_void,
        logging_on: bool,
    ) -> Result<Self, String> {
        let (model, manifest, variables) = inspect(document).map_err(|e| e.to_string())?;
        let compiled = CompiledModel::compile(&model).map_err(|e| e.to_string())?;
        Ok(Self {
            token: token(document.as_bytes()),
            manifest,
            variables,
            compiled: Some(compiled),
            engine: None,
            phase: Phase::Instantiated,
            seed: 0,
            rng_stream: 0,
            start_time: 0.0,
            stop_time: None,
            pending: HashMap::new(),
            logger,
            environment,
            logging_on,
            states: HashMap::new(),
        })
    }

    pub(crate) fn log(&self, message: &str) {
        if let Some(callback) = self.logger {
            let clean = message.replace('\0', " ");
            if let (Ok(category), Ok(message)) = (CString::new("error"), CString::new(clean)) {
                // SAFETY: callback and environment are provided by the importer for this instance.
                unsafe { callback(self.environment, 3, category.as_ptr(), message.as_ptr()) }
            }
        }
    }

    pub(crate) fn variable(&self, vr: u32, kind: &str) -> Result<&Variable, String> {
        vr.checked_sub(1)
            .and_then(|index| self.variables.get(index as usize))
            .filter(|variable| variable.value_reference == vr && variable.kind == kind)
            .ok_or_else(|| format!("unknown or wrong-type value reference {vr}"))
    }

    pub(crate) fn get(&self, vr: u32, kind: &str) -> Result<Value, String> {
        if self.phase == Phase::Error || self.phase == Phase::Terminated {
            return Err("get in invalid lifecycle state".into());
        }
        let variable = self.variable(vr, kind)?;
        if vr == 1 {
            return Ok(Value::Int(self.seed as i64));
        }
        if vr == 2 {
            return Ok(Value::Int(self.rng_stream as i64));
        }
        if vr == 3 {
            return Ok(Value::Float(self.engine.as_ref().map_or(0.0, |engine| {
                engine.with_dependent(|_, e| e.current_time())
            })));
        }
        if let Some(engine) = &self.engine {
            return engine
                .with_dependent(|_, e| e.attribute(&variable.name))
                .ok_or_else(|| format!("attribute `{}` unavailable", variable.name));
        }
        if let Some(value) = self.pending.get(&variable.name) {
            return Ok(*value);
        }
        match kind {
            "Float64" => variable
                .start
                .parse::<f64>()
                .map(Value::Float)
                .map_err(|e| e.to_string()),
            "Int64" => variable
                .start
                .parse::<i64>()
                .map(Value::Int)
                .map_err(|e| e.to_string()),
            "Boolean" => variable
                .start
                .parse::<bool>()
                .map(Value::Bool)
                .map_err(|e| e.to_string()),
            _ => Err("unavailable value type".into()),
        }
    }

    pub(crate) fn set(&mut self, vr: u32, kind: &str, value: Value) -> Result<(), String> {
        if self.phase == Phase::Error || self.phase == Phase::Terminated {
            return Err("set in invalid lifecycle state".into());
        }
        let variable = self.variable(vr, kind)?;
        let name = variable.name.clone();
        let causality = variable.causality;
        if causality == "output" || causality == "independent" {
            return Err(format!("variable `{name}` is read-only"));
        }
        if causality == "parameter"
            && !matches!(self.phase, Phase::Instantiated | Phase::Initialization)
        {
            return Err(format!(
                "parameter `{name}` cannot change after initialization"
            ));
        }
        if vr == 1 || vr == 2 {
            return Err("use UInt64 setter for seed and rng_stream".into());
        }
        if causality == "parameter" && self.phase == Phase::Initialization {
            self.pending.insert(name, value);
            return self.rebuild_initialization();
        }
        if let Some(engine) = &mut self.engine {
            let allowed = if causality == "input" {
                &self.manifest.inputs
            } else {
                &self.manifest.parameters
            };
            let result = engine.with_dependent_mut(|_, e| e.set_input(&name, value, allowed));
            if let Err(error) = result {
                self.phase = Phase::Error;
                return Err(format!("engine: {error}"));
            }
        }
        if self.phase != Phase::Step {
            self.pending.insert(name, value);
        }
        Ok(())
    }

    pub(crate) fn start(&mut self, start: f64, stop: Option<f64>) -> Result<(), String> {
        if self.phase != Phase::Instantiated {
            return Err("enter initialization in invalid lifecycle state".into());
        }
        if !start.is_finite() || start < 0.0 || stop.is_some_and(|s| !s.is_finite() || s <= start) {
            return Err("invalid start or stop time".into());
        }
        let engine = self.build_engine(start, stop, self.seed, self.rng_stream, &self.pending)?;
        self.engine = Some(engine);
        self.start_time = start;
        self.stop_time = stop;
        self.phase = Phase::Initialization;
        Ok(())
    }

    fn build_engine(
        &self,
        start: f64,
        stop: Option<f64>,
        seed: u64,
        rng_stream: u64,
        pending: &HashMap<String, Value>,
    ) -> Result<EngineCell, String> {
        let mut compiled = self
            .compiled
            .as_ref()
            .ok_or("compiled model unavailable")?
            .clone();
        for (name, value) in pending {
            let index = compiled
                .var_index
                .get(name)
                .copied()
                .ok_or_else(|| format!("unknown initial attribute `{name}`"))?;
            compiled.var_init[index] = *value;
        }
        let mut config = EngineConfig {
            seed,
            rng_stream,
            ..EngineConfig::default()
        };
        config.t_max = stop.unwrap_or(f64::INFINITY);
        let mut engine = EngineCell::try_new(compiled, |model| Engine::new(model, config))
            .map_err(|e| format!("engine: {e}"))?;
        let mut allowed = self.manifest.inputs.clone();
        allowed.extend(self.manifest.parameters.iter().cloned());
        for (name, value) in pending {
            engine
                .with_dependent_mut(|_, e| e.set_input(name, *value, &allowed))
                .map_err(|e| format!("engine: {e}"))?;
        }
        if start > 0.0 {
            engine
                .with_dependent_mut(|_, e| e.advance_to(start))
                .map_err(|e| format!("engine: {e}"))?;
        }
        Ok(engine)
    }

    pub(crate) fn set_seed_parameter(&mut self, vr: u32, value: u64) -> Result<(), String> {
        if !matches!(self.phase, Phase::Instantiated | Phase::Initialization) {
            return Err("seed and rng_stream cannot change after initialization".into());
        }
        if vr == 1 {
            self.seed = value;
        } else {
            self.rng_stream = value;
        }
        if self.phase == Phase::Initialization {
            self.rebuild_initialization()?;
        }
        Ok(())
    }

    fn rebuild_initialization(&mut self) -> Result<(), String> {
        let start = self.start_time;
        let stop = self.stop_time;
        self.engine = None;
        self.phase = Phase::Instantiated;
        if let Err(error) = self.start(start, stop) {
            self.phase = Phase::Error;
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn step(&mut self, current: f64, size: f64) -> Result<f64, String> {
        if self.phase != Phase::Step {
            return Err("step in invalid lifecycle state".into());
        }
        if !current.is_finite()
            || !size.is_finite()
            || size <= 0.0
            || !((current + size).is_finite())
        {
            return Err("invalid communication step".into());
        }
        let engine = self.engine.as_mut().ok_or("engine unavailable")?;
        let actual = engine.with_dependent(|_, e| e.current_time());
        if (actual - current).abs() > 1e-12 * actual.abs().max(current.abs()).max(1.0) {
            return Err(format!(
                "communication point {current} differs from engine time {actual}"
            ));
        }
        let result = engine.with_dependent_mut(|_, e| e.advance_to(current + size));
        if let Err(error) = result {
            self.phase = Phase::Error;
            return Err(format!("engine: {error}"));
        }
        Ok(current + size)
    }

    pub(crate) fn snapshot(&mut self) -> Result<*mut c_void, String> {
        if !matches!(self.phase, Phase::Initialization | Phase::Step) {
            return Err("state capture in invalid lifecycle state".into());
        }
        let state = self
            .engine
            .as_ref()
            .ok_or("engine unavailable")?
            .with_dependent(|_, e| e.snapshot())
            .map_err(|e| e.to_string())?;
        let mut boxed = Box::new(FmuState {
            trajectory: state,
            phase: self.phase,
            seed: self.seed,
            rng_stream: self.rng_stream,
            start_time: self.start_time,
            stop_time: self.stop_time,
            pending: self.pending.clone(),
        });
        let pointer = (&mut *boxed as *mut FmuState).cast::<c_void>();
        self.states.insert(pointer as usize, boxed);
        Ok(pointer)
    }

    pub(crate) fn restore(&mut self, pointer: *mut c_void) -> Result<(), String> {
        if !matches!(
            self.phase,
            Phase::Initialization | Phase::Step | Phase::Error
        ) {
            return Err("state restore in invalid lifecycle state".into());
        }
        let state = self
            .states
            .get(&(pointer as usize))
            .ok_or("unknown FMU state pointer")?;
        let mut engine = self.build_engine(
            state.start_time,
            state.stop_time,
            state.seed,
            state.rng_stream,
            &state.pending,
        )?;
        if let Err(error) = engine.with_dependent_mut(|_, e| e.restore(&state.trajectory)) {
            self.phase = Phase::Error;
            return Err(format!("engine: {error}"));
        }
        self.engine = Some(engine);
        self.phase = state.phase;
        self.seed = state.seed;
        self.rng_stream = state.rng_stream;
        self.start_time = state.start_time;
        self.stop_time = state.stop_time;
        self.pending = state.pending.clone();
        Ok(())
    }

    pub(crate) fn reset(&mut self) -> Result<(), String> {
        self.engine = None;
        self.seed = 0;
        self.rng_stream = 0;
        self.pending.clear();
        self.start_time = 0.0;
        self.stop_time = None;
        self.phase = Phase::Instantiated;
        Ok(())
    }

    pub(crate) fn free_state(&mut self, pointer: *mut c_void) -> bool {
        self.states.remove(&(pointer as usize)).is_some()
    }

    pub(crate) fn owns_state(&self, pointer: *mut c_void) -> bool {
        self.states.contains_key(&(pointer as usize))
    }

    #[cfg(test)]
    pub(crate) fn checkpoint_count(&self) -> usize {
        self.states.len()
    }
}
