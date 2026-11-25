//! Validation of the FMU units and of the allocation operators.

use crate::validate::Scope;
use crate::{
    channel_attribute_name, Allocation, AllocationPolicy, Component, Connection, EquationKind,
    Model, ModelError,
};
use raichu_expr::PortRef;
use std::collections::{HashMap, HashSet};

impl Model {
    /// Check declarations independent of any FMU archive or binary.
    pub(crate) fn check_fmu_units(
        &self,
        scopes: &HashMap<&str, Scope<'_>>,
    ) -> Result<(), ModelError> {
        let writers = self.attribute_writers();
        let mut unit_names = HashSet::new();
        let mut outputs: HashMap<(&str, &str), String> = HashMap::new();
        for unit in &self.fmu_units {
            if !unit_names.insert(unit.name.as_str()) {
                return Err(ModelError::FmuDuplicateUnit {
                    unit: unit.name.clone(),
                });
            }
            if unit.path.trim().is_empty() {
                return Err(ModelError::FmuEmptyPath {
                    unit: unit.name.clone(),
                });
            }
            if !(unit.step.is_finite() && unit.step > 0.0) {
                return Err(ModelError::FmuInvalidStep {
                    unit: unit.name.clone(),
                    step: unit.step,
                });
            }
            for binding in unit.inputs.iter().chain(&unit.outputs) {
                let attribute = &binding.attribute;
                if scopes
                    .get(attribute.component.as_str())
                    .and_then(|scope| scope.attribute_kind(&attribute.attribute))
                    .is_none()
                {
                    return Err(ModelError::FmuBindingUnknownAttribute {
                        unit: unit.name.clone(),
                        attribute: format!("{}.{}", attribute.component, attribute.attribute),
                        variable: binding.variable.clone(),
                    });
                }
            }
            let mut output_variables = HashSet::new();
            for binding in &unit.outputs {
                if !output_variables.insert(binding.variable.as_str()) {
                    return Err(ModelError::FmuOutputDuplicated {
                        unit: unit.name.clone(),
                        variable: binding.variable.clone(),
                    });
                }
                let attribute = &binding.attribute;
                let key = (attribute.component.as_str(), attribute.attribute.as_str());
                let writer = writers
                    .get(&key)
                    .cloned()
                    .or_else(|| {
                        self.components.iter().find_map(|component| {
                            component.automata.iter().find_map(|automaton| {
                                automaton.transitions.iter().find_map(|transition| {
                                    transition
                                        .effects
                                        .iter()
                                        .any(|effect| effect.target == *attribute)
                                        .then(|| {
                                            format!(
                                                "transition `{}.{}.{}`",
                                                component.name, automaton.name, transition.name
                                            )
                                        })
                                })
                            })
                        })
                    })
                    .or_else(|| {
                        self.components.iter().find_map(|component| {
                            if component.name != attribute.component {
                                return None;
                            }
                            component.allocations.iter().find_map(|allocation| {
                                self.connections
                                    .iter()
                                    .any(|connection| {
                                        connection.from.component == component.name
                                            && connection.from.port == allocation.port
                                            && channel_attribute_name(
                                                connection,
                                                &allocation.allocated,
                                            ) == attribute.attribute
                                    })
                                    .then(|| {
                                        format!(
                                            "distribution operator `{}.{}`",
                                            component.name, allocation.name
                                        )
                                    })
                            })
                        })
                    })
                    .or_else(|| outputs.get(&key).cloned());
                if let Some(writer) = writer {
                    return Err(ModelError::FmuOutputWritten {
                        unit: unit.name.clone(),
                        attribute: format!("{}.{}", key.0, key.1),
                        variable: binding.variable.clone(),
                        writer,
                    });
                }
                outputs.insert(
                    key,
                    format!("FMU unit `{}` output `{}`", unit.name, binding.variable),
                );
            }
        }
        Ok(())
    }

    /// Structural soundness of the conservative distribution operators
    /// against the connection list: the policy parameters cover exactly
    /// the port's connections, fixed shares sum to 1, and nothing else
    /// writes an allocated quantity.
    ///
    /// Runs after [`Model::check_connections`], so every endpoint has
    /// already been resolved: what is checked here is the *operator*, not
    /// the wiring.
    pub(crate) fn check_allocations(&self) -> Result<(), ModelError> {
        let writers = self.attribute_writers();
        // materialised allocated attribute → the operator that claims it,
        // as `component.allocation` (two operators writing one quantity
        // is the same defect as an equation writing it).
        let mut claimed: HashMap<(&str, String), String> = HashMap::new();
        for component in &self.components {
            for allocation in &component.allocations {
                let edges: Vec<&Connection> = self
                    .connections
                    .iter()
                    .filter(|connection| {
                        connection.from.component == component.name
                            && connection.from.port == allocation.port
                    })
                    .collect();
                Self::check_allocation_policy(component, allocation, &edges)?;
                for connection in &edges {
                    let attribute = channel_attribute_name(connection, &allocation.allocated);
                    let owner = format!("{}.{}", component.name, allocation.name);
                    let writer = writers
                        .get(&(component.name.as_str(), attribute.as_str()))
                        .cloned();
                    if let Some(writer) = writer {
                        return Err(ModelError::AllocationTargetWritten {
                            component: component.name.clone(),
                            allocation: allocation.name.clone(),
                            attribute,
                            writer,
                        });
                    }
                    if let Some(other) =
                        claimed.insert((component.name.as_str(), attribute.clone()), owner)
                    {
                        return Err(ModelError::AllocationTargetWritten {
                            component: component.name.clone(),
                            allocation: allocation.name.clone(),
                            attribute,
                            writer: format!("distribution operator `{other}`"),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Everything that assigns to an attribute outside the distribution
    /// operators: `(component, attribute) → a description of the writer`,
    /// keeping the *first* writer in declaration order so the diagnostic
    /// is stable.
    pub(crate) fn attribute_writers(&self) -> HashMap<(&str, &str), String> {
        let mut writers: HashMap<(&str, &str), String> = HashMap::new();
        for component in &self.components {
            for equation in &component.equations {
                writers
                    .entry((component.name.as_str(), equation.target.as_str()))
                    .or_insert_with(|| {
                        format!(
                            "the {} equation on `{}.{}`",
                            match equation.kind {
                                EquationKind::Explicit => "explicit",
                                EquationKind::Ode => "ODE",
                            },
                            component.name,
                            equation.target
                        )
                    });
            }
            for function in &component.sensitive_functions {
                for assignment in &function.effects {
                    writers
                        .entry((
                            assignment.target.component.as_str(),
                            assignment.target.attribute.as_str(),
                        ))
                        .or_insert_with(|| {
                            format!("sensitive function `{}.{}`", component.name, function.name)
                        });
                }
            }
        }
        writers
    }

    /// The policy parameters of one operator against the connections its
    /// port carries: one value per connection, no extra, no repeat, in
    /// domain, and summing to 1 for fixed shares.
    fn check_allocation_policy(
        component: &Component,
        allocation: &Allocation,
        edges: &[&Connection],
    ) -> Result<(), ModelError> {
        let (params, share) = match &allocation.policy {
            // Proportional needs no parameter: adding a consumer to the
            // port needs no edit anywhere.
            AllocationPolicy::Proportional => return Ok(()),
            AllocationPolicy::Shares { shares } => (shares, true),
            AllocationPolicy::Priority { priorities } => (priorities, false),
        };
        let named = |port: &PortRef| format!("{}.{}", port.component, port.port);
        let mut seen: HashSet<(&str, &str)> = HashSet::new();
        for param in params {
            let key = (param.to.component.as_str(), param.to.port.as_str());
            if !edges
                .iter()
                .any(|edge| (edge.to.component.as_str(), edge.to.port.as_str()) == key)
            {
                return Err(ModelError::AllocationParamUnknown {
                    component: component.name.clone(),
                    allocation: allocation.name.clone(),
                    consumer: named(&param.to),
                });
            }
            if !seen.insert(key) {
                return Err(ModelError::AllocationParamDuplicate {
                    component: component.name.clone(),
                    allocation: allocation.name.clone(),
                    consumer: named(&param.to),
                });
            }
            let in_domain =
                param.value.is_finite() && (!share || (0.0..=1.0).contains(&param.value));
            if !in_domain {
                return Err(ModelError::AllocationParamInvalid {
                    component: component.name.clone(),
                    allocation: allocation.name.clone(),
                    consumer: named(&param.to),
                    value: param.value,
                });
            }
        }
        // Declaration order out: the first uncovered connection a model
        // has is always the same one, whatever order the values are in.
        let mut sum = 0.0;
        for edge in edges {
            let Some(param) = params.iter().find(|param| {
                param.to.component == edge.to.component && param.to.port == edge.to.port
            }) else {
                return Err(ModelError::AllocationParamMissing {
                    component: component.name.clone(),
                    allocation: allocation.name.clone(),
                    consumer: named(&edge.to),
                });
            };
            sum += param.value;
        }
        // Fixed shares are a partition of the available quantity: a sum
        // below 1 leaves a slice nobody can claim, a sum above 1 hands
        // out what does not exist. The tolerance is the rounding of a
        // hand-written decomposition (0.1 + 0.2 is not 0.3 in binary),
        // not a licence to be approximately conservative.
        if share && !edges.is_empty() && (sum - 1.0).abs() > 1e-9 {
            return Err(ModelError::AllocationSharesNotUnit {
                component: component.name.clone(),
                allocation: allocation.name.clone(),
                port: allocation.port.clone(),
                sum,
            });
        }
        Ok(())
    }
}
