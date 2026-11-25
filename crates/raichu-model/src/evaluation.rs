//! Evaluation order of the allocation operators and of the equations they feed.

use crate::algebraic::EquationNode;
use crate::{
    channel_attribute_name, ChannelAttribute, Connection, EquationKind, EvaluationStepReads, Model,
    ModelError, PortDir,
};
use raichu_expr::AttrRef;
use std::collections::{HashMap, HashSet};

impl Model {
    /// The declared evaluation order must cover **exactly** the
    /// explicit equations: one entry each, no omission, no repetition,
    /// nothing that is not one.
    ///
    /// A partial order is refused rather than completed, because every
    /// way of completing it is a guess the modeller did not make: an
    /// omitted equation would either never be swept or be swept at a
    /// position nobody chose, and both are silent wrong answers.
    ///
    /// Runs after cycle classification: a nonlinear or singular cycle
    /// cannot be repaired by an evaluation order.
    pub(crate) fn check_evaluation_order(&self) -> Result<(), ModelError> {
        let Some(order) = &self.evaluation_order else {
            return Ok(());
        };
        let declared: Vec<(&str, &str)> = self.evaluation_steps();
        let known: HashSet<(&str, &str)> = declared.iter().copied().collect();

        let mut listed: HashSet<(&str, &str)> = HashSet::new();
        for entry in order {
            let key = (entry.component.as_str(), entry.attribute.as_str());
            if !known.contains(&key) {
                return Err(ModelError::EvaluationOrderUnknown {
                    component: entry.component.clone(),
                    attribute: entry.attribute.clone(),
                });
            }
            if !listed.insert(key) {
                return Err(ModelError::EvaluationOrderDuplicate {
                    component: entry.component.clone(),
                    attribute: entry.attribute.clone(),
                });
            }
        }
        // Declaration order out: the first omission a model has is
        // always the same one, whatever the order lists.
        for (component, target) in declared {
            if !listed.contains(&(component, target)) {
                return Err(ModelError::EvaluationOrderMissing {
                    component: component.to_owned(),
                    attribute: target.to_owned(),
                });
            }
        }
        Ok(())
    }

    /// The steps of the explicit sweep in **positional** order: for each
    /// component in declaration order, its explicit equations (named by
    /// their target attribute) and then its distribution operators (named
    /// by their own name).
    ///
    /// This is both the default sweep order and the exact set
    /// [`Model::evaluation_order`] must cover. ODE targets are absent by
    /// construction: the integrator carries them, the sweep does not.
    #[must_use]
    pub fn evaluation_steps(&self) -> Vec<(&str, &str)> {
        self.components
            .iter()
            .flat_map(|component| {
                component
                    .equations
                    .iter()
                    .filter(|equation| equation.kind == EquationKind::Explicit)
                    .map(move |equation| (component.name.as_str(), equation.target.as_str()))
                    .chain(
                        component.allocations.iter().map(move |allocation| {
                            (component.name.as_str(), allocation.name.as_str())
                        }),
                    )
            })
            .collect()
    }

    /// The read relation of the explicit sweep: every step of
    /// [`Model::evaluation_steps`], in that positional order, with what it
    /// writes and what it reads.
    ///
    /// This is the relation cycle detection and the priority-surplus check
    /// walk, exposed so an authoring layer can derive an evaluation order
    /// from it rather than from a convention of its own. A distribution
    /// operator reads its available quantity and the demand channel of each
    /// of its connections. Interface connections are resolved first.
    ///
    /// # Errors
    /// The structural errors of the component checks (an unknown
    /// component, port or attribute named by an expression), and of
    /// interface-connection resolution.
    pub fn evaluation_step_reads(&self) -> Result<Vec<EvaluationStepReads>, ModelError> {
        if self.needs_resolution() {
            return self.with_resolved_connections()?.evaluation_step_reads();
        }
        let scopes = self.check_components()?;
        let mut feeds: HashMap<(&str, &str), Vec<&Connection>> = HashMap::new();
        for connection in &self.connections {
            feeds
                .entry((
                    connection.to.component.as_str(),
                    connection.to.port.as_str(),
                ))
                .or_default()
                .push(connection);
        }
        let to_refs = |nodes: Vec<EquationNode>| -> Vec<AttrRef> {
            nodes
                .into_iter()
                .map(|(component, attribute)| AttrRef {
                    component,
                    attribute,
                })
                .collect()
        };
        let mut steps = Vec::new();
        for component in &self.components {
            for equation in &component.equations {
                if equation.kind != EquationKind::Explicit {
                    continue;
                }
                let mut reads = Vec::new();
                Self::collect_reads(&scopes, &feeds, &equation.expr, &mut reads);
                steps.push(EvaluationStepReads {
                    component: component.name.clone(),
                    step: equation.target.clone(),
                    writes: vec![AttrRef {
                        component: component.name.clone(),
                        attribute: equation.target.clone(),
                    }],
                    reads: to_refs(reads),
                });
            }
            for allocation in &component.allocations {
                let mut reads = Vec::new();
                Self::collect_reads(&scopes, &feeds, &allocation.available, &mut reads);
                let edges = self.allocation_edges(component, allocation);
                for connection in &edges {
                    reads.push((
                        component.name.clone(),
                        channel_attribute_name(connection, &allocation.demand),
                    ));
                }
                let writes = edges
                    .iter()
                    .map(|connection| AttrRef {
                        component: component.name.clone(),
                        attribute: channel_attribute_name(connection, &allocation.allocated),
                    })
                    .collect();
                steps.push(EvaluationStepReads {
                    component: component.name.clone(),
                    step: allocation.name.clone(),
                    writes,
                    reads: to_refs(reads),
                });
            }
        }
        Ok(steps)
    }

    /// Every per-connection channel attribute the compiler materialises,
    /// in **connection declaration order**, and within a connection in
    /// the source port's channel declaration order.
    ///
    /// Deliberately lenient on unresolvable endpoints: a connection whose
    /// source port does not exist, or is not an out port, materialises
    /// nothing here and is reported by [`Model::validate`] itself. This
    /// keeps the derivation usable *during* validation, before the
    /// connection checks have run.
    #[must_use]
    pub fn channel_attributes(&self) -> Vec<ChannelAttribute> {
        if self.needs_resolution() {
            if let Ok(resolved) = self.with_resolved_connections() {
                return resolved.channel_attributes();
            }
        }
        let mut out = Vec::new();
        for (index, connection) in self.connections.iter().enumerate() {
            let Some(port) = self
                .components
                .iter()
                .find(|c| c.name == connection.from.component)
                .and_then(|c| c.ports.iter().find(|p| p.name == connection.from.port))
                .filter(|p| p.dir == PortDir::Out)
            else {
                continue;
            };
            for channel in &port.channels {
                out.push(ChannelAttribute {
                    connection: index,
                    component: connection.from.component.clone(),
                    port: port.name.clone(),
                    channel: channel.name.clone(),
                    attribute: channel_attribute_name(connection, &channel.name),
                    init: channel.init,
                });
            }
        }
        out
    }
}
