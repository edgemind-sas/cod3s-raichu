//! FMI co-simulation host, owned by the caller and borrowed by an engine.
//!
//! Archive preparation resolves bindings without loading native code. A run
//! then instantiates the prepared units after explicit permission is checked.

use std::path::Path;

use raichu_expr::Value;
use raichu_fmi::{Archive, Instance, Variable, VariableKind};

use super::{EngineError, FmuProvenance};
use crate::compile::{CompiledModel, VarIdx};

#[derive(Debug, Clone)]
struct Binding {
    attribute: VarIdx,
    variable: Variable,
}

#[derive(Clone)]
struct PreparedUnit {
    name: String,
    step: f64,
    archive: Archive,
    inputs: Vec<Binding>,
    outputs: Vec<Binding>,
    parameters: Vec<(Variable, Value)>,
}

struct Unit {
    name: String,
    step: f64,
    archive: Archive,
    inputs: Vec<Binding>,
    outputs: Vec<Binding>,
    parameters: Vec<(Variable, Value)>,
    instance: Option<Instance>,
    sampled: Vec<Value>,
    last_point: f64,
    grid_index: u64,
    extra_due: Option<f64>,
}

impl From<PreparedUnit> for Unit {
    fn from(prepared: PreparedUnit) -> Self {
        Self {
            name: prepared.name,
            step: prepared.step,
            archive: prepared.archive,
            inputs: prepared.inputs,
            outputs: prepared.outputs,
            parameters: prepared.parameters,
            instance: None,
            sampled: Vec::new(),
            last_point: 0.0,
            grid_index: 0,
            extra_due: None,
        }
    }
}

/// Immutable, validated FMU archives and bindings shared by independent runs.
///
/// Preparation does not load native code. Each spawned host owns its own FMU
/// instances and trajectory state while sharing the unpacked archive contents.
pub struct PreparedCoSimulation {
    units: Vec<PreparedUnit>,
    declarations: Vec<raichu_model::FmuUnit>,
    var_names: Vec<String>,
}

impl PreparedCoSimulation {
    /// Unpack and validate the FMU declarations once, after run permission.
    ///
    /// # Errors
    /// Returns a typed permission, archive or binding error.
    pub fn prepare_authorized(
        model: &CompiledModel,
        base_dir: &Path,
        allow_fmu_import: bool,
    ) -> Result<Self, EngineError> {
        if !allow_fmu_import {
            if let Some(unit) = model.fmu_units.first() {
                return Err(EngineError::FmuPermission {
                    unit: unit.name.clone(),
                });
            }
        }
        let mut units = Vec::with_capacity(model.fmu_units.len());
        for declared in &model.fmu_units {
            let path = Path::new(&declared.path);
            let path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                base_dir.join(path)
            };
            let archive = Archive::open(path).map_err(|source| EngineError::FmuRuntime {
                unit: declared.name.clone(),
                time: 0.0,
                source,
            })?;
            let inputs = declared
                .inputs
                .iter()
                .map(|binding| resolve_binding(model, &archive, &declared.name, binding, "input"))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = declared
                .outputs
                .iter()
                .map(|binding| resolve_binding(model, &archive, &declared.name, binding, "output"))
                .collect::<Result<Vec<_>, _>>()?;
            let parameters = declared
                .parameters
                .iter()
                .map(|parameter| {
                    let variable = archive
                        .description()
                        .variable(&parameter.variable)
                        .ok_or_else(|| EngineError::FmuBinding {
                            unit: declared.name.clone(),
                            attribute: "<parameter>".to_owned(),
                            variable: parameter.variable.clone(),
                            reason: "unknown FMU variable".to_owned(),
                        })?;
                    if variable.causality != "parameter"
                        || !same_kind(parameter.value, &variable.kind)
                    {
                        return Err(EngineError::FmuBinding {
                            unit: declared.name.clone(),
                            attribute: "<parameter>".to_owned(),
                            variable: parameter.variable.clone(),
                            reason: "parameter causality or type mismatch".to_owned(),
                        });
                    }
                    Ok((variable.clone(), parameter.value))
                })
                .collect::<Result<Vec<_>, EngineError>>()?;
            units.push(PreparedUnit {
                name: declared.name.clone(),
                step: declared.step,
                archive,
                inputs,
                outputs,
                parameters,
            });
        }
        Ok(Self {
            units,
            declarations: model.fmu_units.clone(),
            var_names: model.var_names.clone(),
        })
    }

    /// Create an independent, uninitialized trajectory host without unpacking.
    #[must_use]
    pub fn spawn_host(&self) -> CoSimulationHost {
        CoSimulationHost {
            units: self.units.iter().cloned().map(Unit::from).collect(),
            declarations: self.declarations.clone(),
            var_names: self.var_names.clone(),
        }
    }

    fn into_host(self) -> CoSimulationHost {
        CoSimulationHost {
            units: self.units.into_iter().map(Unit::from).collect(),
            declarations: self.declarations,
            var_names: self.var_names,
        }
    }

    /// Identity and content hashes of the prepared units.
    #[must_use]
    pub fn provenance(&self) -> Vec<FmuProvenance> {
        self.units
            .iter()
            .map(|unit| provenance(&unit.name, &unit.archive))
            .collect()
    }

    /// Name the first unit that forbids concurrent instances, if any.
    #[must_use]
    pub fn single_instance_unit(&self) -> Option<&str> {
        self.units
            .iter()
            .find(|unit| {
                unit.archive
                    .description()
                    .can_be_instantiated_only_once_per_process
            })
            .map(|unit| unit.name.as_str())
    }

    /// Check serialized state capability before any native library is loaded.
    ///
    /// # Errors
    /// Names the first unit missing the capability.
    pub fn require_serializable_state(&self) -> Result<(), EngineError> {
        for unit in &self.units {
            let description = unit.archive.description();
            if !description.can_get_and_set_fmu_state || !description.can_serialize_fmu_state {
                return Err(EngineError::FmuCapability {
                    unit: unit.name.clone(),
                    capability: "serialized FMU state get/set",
                });
            }
        }
        Ok(())
    }
}

/// Prepared FMU archives and live instances for one simulation trajectory.
///
/// The caller retains this host across engine reconstruction so an interactive
/// facade never resets native state between calls.
pub struct CoSimulationHost {
    units: Vec<Unit>,
    declarations: Vec<raichu_model::FmuUnit>,
    var_names: Vec<String>,
}

impl CoSimulationHost {
    /// Unpack units and resolve bindings without loading a native library.
    /// Permission is checked before opening any archive.
    ///
    /// # Errors
    /// Returns a typed permission, archive or binding error.
    pub fn prepare_authorized(
        model: &CompiledModel,
        base_dir: &Path,
        allow_fmu_import: bool,
    ) -> Result<Self, EngineError> {
        PreparedCoSimulation::prepare_authorized(model, base_dir, allow_fmu_import)
            .map(PreparedCoSimulation::into_host)
    }

    pub(super) fn matches_model(&self, model: &CompiledModel) -> bool {
        self.declarations == model.fmu_units && self.var_names == model.var_names
    }

    /// Identity and content hashes of every prepared unit.
    #[must_use]
    pub fn provenance(&self) -> Vec<FmuProvenance> {
        self.units
            .iter()
            .map(|unit| provenance(&unit.name, &unit.archive))
            .collect()
    }

    /// Whether any unit forbids multiple instances in one process.
    #[must_use]
    pub fn single_instance_unit(&self) -> Option<&str> {
        self.units
            .iter()
            .find(|unit| {
                unit.archive
                    .description()
                    .can_be_instantiated_only_once_per_process
            })
            .map(|unit| unit.name.as_str())
    }

    /// Check that every unit can be serialized for snapshots and restoration.
    ///
    /// # Errors
    /// Names the first unit missing either native-state capability.
    pub fn require_serializable_state(&self) -> Result<(), EngineError> {
        for unit in &self.units {
            let description = unit.archive.description();
            if !description.can_get_and_set_fmu_state || !description.can_serialize_fmu_state {
                return Err(EngineError::FmuCapability {
                    unit: unit.name.clone(),
                    capability: "serialized FMU state get/set",
                });
            }
        }
        Ok(())
    }

    pub(super) fn start(&mut self, vars: &[Value]) -> Result<(), EngineError> {
        for unit in &mut self.units {
            let reset_error = unit
                .instance
                .as_mut()
                .and_then(|instance| instance.reset_uninitialized().err());
            if let Some(source) = reset_error {
                if unit
                    .archive
                    .description()
                    .can_be_instantiated_only_once_per_process
                {
                    return Err(runtime_error(&unit.name, 0.0, source));
                }
                drop(unit.instance.take());
            }
            if unit.instance.is_none() {
                unit.instance = Some(
                    unit.archive
                        .instantiate_uninitialized(&unit.name)
                        .map_err(|source| runtime_error(&unit.name, 0.0, source))?,
                );
            }
            let instance = unit
                .instance
                .as_mut()
                .ok_or_else(|| EngineError::FmuHostRequired {
                    unit: unit.name.clone(),
                })?;
            for (variable, value) in &unit.parameters {
                instance
                    .set_value(variable, to_fmi(*value))
                    .map_err(|source| runtime_error(&unit.name, 0.0, source))?;
            }
            for binding in &unit.inputs {
                instance
                    .set_value(&binding.variable, to_fmi(vars[binding.attribute]))
                    .map_err(|source| runtime_error(&unit.name, 0.0, source))?;
            }
            instance
                .initialize(0.0)
                .map_err(|source| runtime_error(&unit.name, 0.0, source))?;
            unit.sampled = unit
                .inputs
                .iter()
                .map(|binding| vars[binding.attribute])
                .collect();
            unit.last_point = 0.0;
            unit.grid_index = 0;
            unit.extra_due = None;
        }
        Ok(())
    }

    pub(super) fn next_point(&self) -> Option<f64> {
        self.units
            .iter()
            .map(|unit| {
                unit.extra_due
                    .unwrap_or_else(|| (unit.grid_index + 1) as f64 * unit.step)
            })
            .min_by(f64::total_cmp)
    }

    pub(super) fn initial_outputs(&mut self) -> Result<Vec<(VarIdx, Value, String)>, EngineError> {
        let mut outputs = Vec::new();
        for unit in &mut self.units {
            let instance = unit
                .instance
                .as_ref()
                .ok_or_else(|| EngineError::FmuHostRequired {
                    unit: unit.name.clone(),
                })?;
            for binding in &unit.outputs {
                let value = instance
                    .get_value(&binding.variable)
                    .map_err(|source| runtime_error(&unit.name, 0.0, source))?;
                outputs.push((binding.attribute, from_fmi(value), unit.name.clone()));
            }
        }
        Ok(outputs)
    }

    pub(super) fn input_values(&self, vars: &[Value]) -> Vec<Value> {
        self.units
            .iter()
            .flat_map(|unit| unit.inputs.iter().map(|binding| vars[binding.attribute]))
            .collect()
    }

    pub(super) fn mark_input_changes(
        &mut self,
        before: &[Value],
        after: &[Value],
        time: f64,
    ) -> Vec<(String, VarIdx, Value)> {
        let mut held = Vec::new();
        let mut offset = 0;
        for unit in &mut self.units {
            let previous = &before[offset..offset + unit.inputs.len()];
            offset += unit.inputs.len();
            if time <= unit.last_point {
                continue;
            }
            let regular = (unit.grid_index + 1) as f64 * unit.step;
            if time >= regular {
                continue;
            }
            for ((binding, sampled), old) in unit.inputs.iter().zip(&unit.sampled).zip(previous) {
                if *old != after[binding.attribute] {
                    if unit
                        .archive
                        .description()
                        .can_handle_variable_communication_step_size
                    {
                        unit.extra_due = Some(time);
                    } else {
                        held.push((unit.name.clone(), binding.attribute, *sampled));
                    }
                }
            }
        }
        held
    }

    pub(super) fn step_due(
        &mut self,
        time: f64,
    ) -> Result<Vec<(VarIdx, Value, String)>, EngineError> {
        let mut changed = Vec::new();
        for unit in &mut self.units {
            let regular = (unit.grid_index + 1) as f64 * unit.step;
            let due_extra = unit.extra_due.is_some_and(|point| same_time(point, time));
            let due_regular = same_time(regular, time);
            if !due_extra && !due_regular {
                continue;
            }
            let duration = time - unit.last_point;
            if duration <= 0.0 {
                continue;
            }
            let instance = unit
                .instance
                .as_mut()
                .ok_or_else(|| EngineError::FmuHostRequired {
                    unit: unit.name.clone(),
                })?;
            for (binding, value) in unit.inputs.iter().zip(&unit.sampled) {
                instance
                    .set_value(&binding.variable, to_fmi(*value))
                    .map_err(|source| runtime_error(&unit.name, time, source))?;
            }
            instance
                .do_step(unit.last_point, duration)
                .map_err(|source| runtime_error(&unit.name, time, source))?;
            for binding in &unit.outputs {
                let value = instance
                    .get_value(&binding.variable)
                    .map_err(|source| runtime_error(&unit.name, time, source))?;
                changed.push((binding.attribute, from_fmi(value), unit.name.clone()));
            }
            unit.last_point = time;
            if due_regular {
                unit.grid_index += 1;
            }
            unit.extra_due = None;
        }
        Ok(changed)
    }

    pub(super) fn sample_inputs(&mut self, vars: &[Value], time: f64) {
        for unit in &mut self.units {
            if same_time(unit.last_point, time) {
                unit.sampled = unit
                    .inputs
                    .iter()
                    .map(|binding| vars[binding.attribute])
                    .collect();
            }
        }
    }

    pub(super) fn save_states(&mut self) -> Result<Vec<Vec<u8>>, EngineError> {
        self.require_serializable_state()?;
        self.units
            .iter_mut()
            .map(|unit| {
                unit.instance
                    .as_mut()
                    .ok_or_else(|| EngineError::FmuHostRequired {
                        unit: unit.name.clone(),
                    })?
                    .get_state()
                    .map_err(|source| runtime_error(&unit.name, unit.last_point, source))
            })
            .collect()
    }

    pub(super) fn restore_states(&mut self, states: &[Vec<u8>]) -> Result<(), EngineError> {
        self.require_serializable_state()?;
        for (unit, state) in self.units.iter_mut().zip(states) {
            unit.instance
                .as_mut()
                .ok_or_else(|| EngineError::FmuHostRequired {
                    unit: unit.name.clone(),
                })?
                .set_state(state)
                .map_err(|source| runtime_error(&unit.name, unit.last_point, source))?;
        }
        Ok(())
    }

    pub(super) fn positions(&self) -> Vec<(f64, u64, Option<f64>, Vec<Value>)> {
        self.units
            .iter()
            .map(|unit| {
                (
                    unit.last_point,
                    unit.grid_index,
                    unit.extra_due,
                    unit.sampled.clone(),
                )
            })
            .collect()
    }

    pub(super) fn restore_positions(&mut self, positions: &[(f64, u64, Option<f64>, Vec<Value>)]) {
        for (unit, (last, index, extra, sampled)) in self.units.iter_mut().zip(positions) {
            unit.last_point = *last;
            unit.grid_index = *index;
            unit.extra_due = *extra;
            unit.sampled = sampled.clone();
        }
    }
}

fn provenance(name: &str, archive: &Archive) -> FmuProvenance {
    let description = archive.description();
    FmuProvenance {
        name: name.to_owned(),
        fmi_version: match description.fmi_version {
            raichu_fmi::FmiVersion::V2 => "2.0",
            raichu_fmi::FmiVersion::V3 => "3.0",
        }
        .to_owned(),
        generating_tool: description.generation_tool.clone(),
        generating_tool_version: description.generation_tool_version.clone(),
        instantiation_token: description.token.clone(),
        content_hash: archive.sha256().to_owned(),
    }
}

fn resolve_binding(
    model: &CompiledModel,
    archive: &Archive,
    unit: &str,
    binding: &raichu_model::FmuBinding,
    causality: &str,
) -> Result<Binding, EngineError> {
    let attribute = format!(
        "{}.{}",
        binding.attribute.component, binding.attribute.attribute
    );
    let failure = |reason: &str| EngineError::FmuBinding {
        unit: unit.to_owned(),
        attribute: attribute.clone(),
        variable: binding.variable.clone(),
        reason: reason.to_owned(),
    };
    let index = *model
        .var_index
        .get(&attribute)
        .ok_or_else(|| failure("unknown model attribute"))?;
    let variable = archive
        .description()
        .variable(&binding.variable)
        .ok_or_else(|| failure("unknown FMU variable"))?;
    let accepted = variable.causality == causality
        || (causality == "input"
            && variable.causality == "parameter"
            && variable.variability == "tunable");
    if !accepted {
        return Err(failure("wrong FMU variable causality"));
    }
    if !same_kind(model.var_init[index], &variable.kind) {
        return Err(failure("incompatible attribute and FMU variable types"));
    }
    Ok(Binding {
        attribute: index,
        variable: variable.clone(),
    })
}

fn same_kind(value: Value, kind: &VariableKind) -> bool {
    matches!(
        (value, kind),
        (Value::Float(_), VariableKind::Float64)
            | (Value::Int(_), VariableKind::Int64)
            | (Value::Bool(_), VariableKind::Boolean)
    )
}

fn to_fmi(value: Value) -> raichu_fmi::Value {
    match value {
        Value::Float(v) => raichu_fmi::Value::Float(v),
        Value::Int(v) => raichu_fmi::Value::Int(v),
        Value::Bool(v) => raichu_fmi::Value::Bool(v),
    }
}

fn from_fmi(value: raichu_fmi::Value) -> Value {
    match value {
        raichu_fmi::Value::Float(v) => Value::Float(v),
        raichu_fmi::Value::Int(v) => Value::Int(v),
        raichu_fmi::Value::Bool(v) => Value::Bool(v),
    }
}

fn runtime_error(unit: &str, time: f64, source: raichu_fmi::FmiError) -> EngineError {
    EngineError::FmuRuntime {
        unit: unit.to_owned(),
        time,
        source,
    }
}

fn same_time(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
}
