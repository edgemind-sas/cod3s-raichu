//! Structural validation of a model, the fail-fast gate the engine relies on.

mod components;
mod expressions;
mod programs;
mod resources;

use crate::{AttrKind, Attribute, ChannelAttribute, Model, ModelError, Port};
use std::collections::{HashMap, HashSet};

/// Component-scope lookup tables used during validation.
pub(crate) struct Scope<'m> {
    pub(crate) attributes: HashMap<&'m str, &'m Attribute>,
    /// Names of the compiler-materialised channel attributes of this
    /// component. They are float attributes like any other: readable,
    /// writable, journalled; they simply have no declaration to borrow.
    pub(crate) channel_attributes: HashSet<String>,
    pub(crate) ports: HashMap<&'m str, &'m Port>,
    /// automaton name → set of its state names.
    pub(crate) automata: HashMap<&'m str, HashSet<&'m str>>,
}

impl Scope<'_> {
    /// Kind of an attribute of this component, declared or materialised.
    pub(crate) fn attribute_kind(&self, name: &str) -> Option<AttrKind> {
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

    pub(crate) fn check_components(&self) -> Result<HashMap<&str, Scope<'_>>, ModelError> {
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
