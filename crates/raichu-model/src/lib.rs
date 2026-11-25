//! # raichu-model: the native RAICHU formalism
//!
//! Data model of the Distributed Stochastic Hybrid Automata (DSHA)
//! formalism realising Piecewise-Deterministic Markov Processes (PDMP),
//! as formalised in Desgeorges et al. 2021 (RESS).
//!
//! Design decisions:
//!
//! - **In/out ports** are the fundamental connection notion; *interfaces*
//!   group ports so an [`InterfaceConnection`] joins two groups at once.
//! - The model layer is pure data (serde), side-effect-free, validated at
//!   build time with **typed errors: never a crash on bad input**.
//! - Behaviour (guards, sensitive-function effects) is expressed as
//!   serializable expression trees from `raichu-expr`. Sensitivity sets
//!   (which attribute change re-triggers which function) are *derived*
//!   from the expressions, not declared by hand: one modeller error
//!   class removed.
//!
//! Validation ([`Model::validate`]) is the single gate: a model that
//! passes is structurally sound (all references resolve, distributions are
//! well-formed, initial states exist). The engine only consumes validated
//! models.

mod algebraic;
mod connections;
mod error;
mod evaluation;
mod format;
mod graph;
mod schema;
mod validate;

pub use algebraic::{AlgebraicBlock, BlockRow, SingularReason};
pub use error::ModelError;
pub use format::{Feature, FormatHeader, LoadError, ENVELOPE_KEY, FORMAT_REVISION};
pub use graph::strongly_connected_components;
pub use schema::{
    channel_attribute_name, Allocation, AllocationPolicy, AttrKind, Attribute, Automaton, Channel,
    ChannelAttribute, Component, Connection, ConsumerParam, CycleCancellation, Distrib, Equation,
    EquationKind, EvaluationStepReads, FmuBinding, FmuParameter, FmuUnit, Indicator,
    IndicatorTarget, Interface, InterfaceConnection, InterfaceRef, InterruptionPolicy, Model,
    ModelSource, Port, PortDir, PortEquation, Program, ProgramConstraint, ProgramSense,
    ProgramTieBreak, ProgramTieObjective, ProgramVariable, SensitiveFunction, Target, Transition,
    TransitionKind, EDGE, PORT,
};

use std::collections::{HashMap, HashSet};

/// Component-scope lookup tables used during validation.
struct Scope<'m> {
    attributes: HashMap<&'m str, &'m Attribute>,
    /// Names of the compiler-materialised channel attributes of this
    /// component. They are float attributes like any other: readable,
    /// writable, journalled; they simply have no declaration to borrow.
    channel_attributes: HashSet<String>,
    ports: HashMap<&'m str, &'m Port>,
    /// automaton name → set of its state names.
    automata: HashMap<&'m str, HashSet<&'m str>>,
}

impl Scope<'_> {
    /// Kind of an attribute of this component, declared or materialised.
    fn attribute_kind(&self, name: &str) -> Option<AttrKind> {
        if let Some(attribute) = self.attributes.get(name) {
            return Some(attribute.kind);
        }
        // Materialised channel attributes are float by construction.
        self.channel_attributes
            .contains(name)
            .then_some(AttrKind::Float)
    }
}

impl Model {
    /// Validate structural soundness: unique names, resolvable
    /// references, well-formed distributions, matching kinds. Returns the *first*
    /// error encountered in deterministic (declaration) order.
    ///
    /// The engine (`raichu-core`) only accepts validated models; this is
    /// the fail-fast gate (typed errors at build
    /// time, never mid-simulation surprises).
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.needs_resolution() {
            return self.with_resolved_connections()?.validate();
        }
        let scopes = self.check_components()?;
        self.check_fmu_units(&scopes)?;
        self.check_connections(&scopes)?;
        self.check_allocations()?;
        self.check_expressions(&scopes)?;
        self.check_programs(&scopes)?;
        self.classify_algebraic(&scopes)?;
        // The compiler checks declared ordering after simultaneous blocks form.
        self.check_priority_surplus_return(&scopes)?;
        self.check_evaluation_order()?;
        self.check_indicators(&scopes)?;
        self.check_transition_effects()?;
        if let Some(value) = self.unbounded_rate {
            if !(value.is_finite() && value > 0.0) {
                return Err(ModelError::UnboundedRateInvalid { value });
            }
        }
        Ok(())
    }

    fn check_components(&self) -> Result<HashMap<&str, Scope<'_>>, ModelError> {
        let materialised = self.channel_attributes();
        let mut scopes: HashMap<&str, Scope<'_>> = HashMap::new();
        for component in &self.components {
            if scopes.contains_key(component.name.as_str()) {
                return Err(ModelError::DuplicateComponent {
                    name: component.name.clone(),
                });
            }
            let owned: Vec<&ChannelAttribute> = materialised
                .iter()
                .filter(|entry| entry.component == component.name)
                .collect();
            let scope = Self::check_component(component, &owned)?;
            scopes.insert(component.name.as_str(), scope);
        }
        Ok(scopes)
    }
}

#[cfg(test)]
mod tests;
