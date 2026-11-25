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
mod format;
mod graph;
mod schema;

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

use algebraic::EquationNode;
use raichu_expr::{AggOp, AttrRef, CmpOp, Expr, PortRef, Value};
use std::collections::{HashMap, HashSet};

/// Flow family of a value kind: a boolean carries a **discrete** flow
/// (present or absent), an integer or a float a **continuous** quantity.
/// The two never share one aggregation (see
/// [`ModelError::ConnectionFamilyMismatch`]).
fn is_discrete(kind: AttrKind) -> bool {
    matches!(kind, AttrKind::Bool)
}

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

    /// Check declarations independent of any FMU archive or binary.
    fn check_fmu_units(&self, scopes: &HashMap<&str, Scope<'_>>) -> Result<(), ModelError> {
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
    fn check_allocations(&self) -> Result<(), ModelError> {
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

    /// An edge effect's target must have no other writer that would erase
    /// the one-shot write at the next evaluation: an explicit equation or a
    /// sensitive function. The target of an ODE equation is not such a
    /// writer: integration starts from the attribute's current value, so an
    /// edge effect on it is a reset map, the jump of a piecewise-
    /// deterministic Markov process.
    fn check_transition_effects(&self) -> Result<(), ModelError> {
        let writers = self.level_writers();
        for component in &self.components {
            for automaton in &component.automata {
                for transition in &automaton.transitions {
                    for assignment in &transition.effects {
                        let key = (
                            assignment.target.component.as_str(),
                            assignment.target.attribute.as_str(),
                        );
                        if let Some(writer) = writers.get(&key) {
                            return Err(ModelError::TransitionEffectOverwritten {
                                transition: format!(
                                    "{}.{}.{}",
                                    component.name, automaton.name, transition.name
                                ),
                                attribute: format!("{}.{}", key.0, key.1),
                                writer: writer.clone(),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The writers that set an attribute as a *level*, re-evaluated
    /// whenever what they read changes: explicit equations and sensitive
    /// functions, first writer in declaration order. ODE equations are left
    /// out: they move their target from its current value instead of
    /// recomputing it.
    fn level_writers(&self) -> HashMap<(&str, &str), String> {
        let mut writers: HashMap<(&str, &str), String> = HashMap::new();
        for component in &self.components {
            for equation in &component.equations {
                if matches!(equation.kind, EquationKind::Ode) {
                    continue;
                }
                writers
                    .entry((component.name.as_str(), equation.target.as_str()))
                    .or_insert_with(|| {
                        format!(
                            "the explicit equation on `{}.{}`",
                            component.name, equation.target
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

    /// Validate the discrete-only, affine and single-writer contract of all
    /// model-level programs before compilation can consume them.
    fn check_programs(&self, scopes: &HashMap<&str, Scope<'_>>) -> Result<(), ModelError> {
        let mut names = HashSet::new();
        let mut outputs: HashMap<AttrRef, usize> = HashMap::new();
        let invalid =
            |program: &Program, term: String, reason: String| ModelError::ProgramInvalid {
                program: program.name.clone(),
                term,
                reason,
            };
        for (index, program) in self.programs.iter().enumerate() {
            if !names.insert(program.name.as_str()) {
                return Err(invalid(
                    program,
                    "name".into(),
                    "duplicate program name".into(),
                ));
            }
            if program.variables.is_empty() {
                return Err(invalid(
                    program,
                    "variables".into(),
                    "at least one decision variable is required".into(),
                ));
            }
            if program
                .node_limit
                .is_some_and(|limit| limit <= 0 || limit > i32::MAX as i64)
            {
                return Err(invalid(
                    program,
                    "node_limit".into(),
                    "must be in 1..=2147483647".into(),
                ));
            }
            let mut local = HashSet::new();
            for variable in &program.variables {
                let attr = &variable.attribute;
                let term = format!("{}.{}", attr.component, attr.attribute);
                let kind = scopes
                    .get(attr.component.as_str())
                    .and_then(|scope| scope.attribute_kind(&attr.attribute))
                    .ok_or_else(|| {
                        invalid(program, term.clone(), "unknown decision attribute".into())
                    })?;
                if !kind.matches(&variable.on_infeasible) {
                    return Err(invalid(
                        program,
                        term,
                        "fallback kind does not match the decision attribute".into(),
                    ));
                }
                if !local.insert(attr.clone()) {
                    return Err(invalid(
                        program,
                        term,
                        "decision attribute is repeated".into(),
                    ));
                }
                if let Some(other) = outputs.insert(attr.clone(), index) {
                    return Err(invalid(
                        program,
                        term,
                        format!("also written by program `{}`", self.programs[other].name),
                    ));
                }
            }
            for (attr, expected) in [(&program.feasible, AttrKind::Bool)].into_iter().chain(
                program
                    .objective_value
                    .iter()
                    .map(|attr| (attr, AttrKind::Float)),
            ) {
                let term = format!("{}.{}", attr.component, attr.attribute);
                let kind = scopes
                    .get(attr.component.as_str())
                    .and_then(|scope| scope.attribute_kind(&attr.attribute))
                    .ok_or_else(|| {
                        invalid(program, term.clone(), "unknown output attribute".into())
                    })?;
                if kind != expected {
                    return Err(invalid(
                        program,
                        term,
                        format!("output must have kind {expected:?}"),
                    ));
                }
                if let Some(other) = outputs.insert(attr.clone(), index) {
                    return Err(invalid(
                        program,
                        term,
                        format!("also written by program `{}`", self.programs[other].name),
                    ));
                }
            }
        }

        let mut allocated = HashSet::new();
        for component in &self.components {
            for allocation in &component.allocations {
                for edge in self.connections.iter().filter(|edge| {
                    edge.from.component == component.name && edge.from.port == allocation.port
                }) {
                    allocated.insert(AttrRef {
                        component: component.name.clone(),
                        attribute: channel_attribute_name(edge, &allocation.allocated),
                    });
                }
            }
        }
        let equation_targets: HashSet<_> = self
            .components
            .iter()
            .flat_map(|component| {
                component.equations.iter().map(|equation| AttrRef {
                    component: component.name.clone(),
                    attribute: equation.target.clone(),
                })
            })
            .collect();
        let regular_writers = self.attribute_writers();
        let sources = self.in_port_sources();
        let mut dependencies = vec![HashSet::new(); self.programs.len()];
        for (index, program) in self.programs.iter().enumerate() {
            let decisions: HashSet<_> = program
                .variables
                .iter()
                .map(|variable| variable.attribute.clone())
                .collect();
            for output in decisions
                .iter()
                .chain(std::iter::once(&program.feasible))
                .chain(program.objective_value.iter())
            {
                let term = format!("{}.{}", output.component, output.attribute);
                if let Some(writer) =
                    regular_writers.get(&(output.component.as_str(), output.attribute.as_str()))
                {
                    return Err(invalid(program, term, format!("also written by {writer}")));
                }
                if allocated.contains(output) {
                    return Err(invalid(
                        program,
                        term,
                        "also written by a distribution operator".into(),
                    ));
                }
                if self
                    .components
                    .iter()
                    .flat_map(|component| &component.automata)
                    .flat_map(|automaton| &automaton.transitions)
                    .flat_map(|transition| &transition.effects)
                    .any(|effect| &effect.target == output)
                {
                    return Err(invalid(
                        program,
                        term,
                        "also written by a transition effect".into(),
                    ));
                }
            }
            let mut expressions: Vec<(&str, &Expr, bool)> =
                vec![("objective", &program.objective, true)];
            for variable in &program.variables {
                if let Some(expr) = &variable.lower {
                    expressions.push(("variable lower", expr, false));
                }
                if let Some(expr) = &variable.upper {
                    expressions.push(("variable upper", expr, false));
                }
            }
            let mut constraint_names = HashSet::new();
            for constraint in &program.constraints {
                if !constraint_names.insert(constraint.name.as_str()) {
                    return Err(invalid(
                        program,
                        constraint.name.clone(),
                        "duplicate constraint name".into(),
                    ));
                }
                if constraint.lower.is_none() && constraint.upper.is_none() {
                    return Err(invalid(
                        program,
                        constraint.name.clone(),
                        "constraint requires at least one bound".into(),
                    ));
                }
                expressions.push((&constraint.name, &constraint.expr, true));
                if let Some(expr) = &constraint.lower {
                    expressions.push((&constraint.name, expr, false));
                }
                if let Some(expr) = &constraint.upper {
                    expressions.push((&constraint.name, expr, false));
                }
            }
            if let Some(ProgramTieBreak::Objectives(objectives)) = &program.tie_break {
                for objective in objectives {
                    expressions.push(("tie_break", &objective.expr, true));
                }
            }
            for (term, expr, affine) in expressions {
                Self::check_expr(scopes, &sources, expr, term)
                    .map_err(|error| invalid(program, term.into(), error.to_string()))?;
                if !Self::program_expr_numeric(expr, scopes, &decisions) {
                    return Err(invalid(
                        program,
                        term.into(),
                        "expected a numeric expression".into(),
                    ));
                }
                let form = expr.affine_in(&decisions).map_err(|error| {
                    invalid(
                        program,
                        term.into(),
                        format!("nonlinear decision term: {:?}", error.expression),
                    )
                })?;
                if !affine && !form.terms.is_empty() {
                    return Err(invalid(
                        program,
                        term.into(),
                        "bound reads this program's decision variable".into(),
                    ));
                }
                let free_parts = std::iter::once(&form.constant)
                    .chain(form.terms.iter().map(|(_, coefficient)| coefficient));
                for free in free_parts {
                    if Self::expr_contains_time(free) {
                        return Err(invalid(
                            program,
                            term.into(),
                            "time input is continuously varying".into(),
                        ));
                    }
                    let mut refs = Vec::new();
                    free.for_each_attr_ref(&mut |attr| refs.push(attr.clone()));
                    for attr in refs {
                        let name = format!("{}.{}", attr.component, attr.attribute);
                        if equation_targets.contains(&attr) || allocated.contains(&attr) {
                            return Err(invalid(
                                program,
                                name,
                                "input is computed by the explicit sweep or integration".into(),
                            ));
                        }
                        if let Some(&producer) = outputs.get(&attr) {
                            if producer == index {
                                return Err(invalid(
                                    program,
                                    name,
                                    "program reads its own output".into(),
                                ));
                            }
                            dependencies[index].insert(producer);
                        }
                    }
                    let mut ports = Vec::new();
                    free.for_each_port_agg(&mut |port, channel| {
                        ports.push((port.clone(), channel.map(str::to_owned)))
                    });
                    for (port, channel) in ports {
                        for edge in self.connections.iter().filter(|edge| edge.to == port) {
                            let attribute = channel
                                .as_deref()
                                .map(|name| channel_attribute_name(edge, name))
                                .or_else(|| {
                                    scopes
                                        .get(edge.from.component.as_str())
                                        .and_then(|scope| scope.ports.get(edge.from.port.as_str()))
                                        .and_then(|source| source.attr.clone())
                                });
                            if let Some(attribute) = attribute {
                                let source = AttrRef {
                                    component: edge.from.component.clone(),
                                    attribute,
                                };
                                if equation_targets.contains(&source) || allocated.contains(&source)
                                {
                                    return Err(invalid(
                                        program,
                                        format!("{}.{}", port.component, port.port),
                                        "port aggregate reads a sweep or integrated quantity"
                                            .into(),
                                    ));
                                }
                                if let Some(&producer) = outputs.get(&source) {
                                    if producer == index {
                                        return Err(invalid(
                                            program,
                                            format!("{}.{}", port.component, port.port),
                                            "program reads its own output through a port".into(),
                                        ));
                                    }
                                    dependencies[index].insert(producer);
                                }
                            }
                        }
                    }
                }
            }
        }
        fn reaches(
            index: usize,
            next: usize,
            graph: &[HashSet<usize>],
            seen: &mut HashSet<usize>,
        ) -> bool {
            if next == index {
                return true;
            }
            if !seen.insert(next) {
                return false;
            }
            graph[next]
                .iter()
                .any(|&node| reaches(index, node, graph, seen))
        }
        for (index, edges) in dependencies.iter().enumerate() {
            if edges
                .iter()
                .any(|&next| reaches(index, next, &dependencies, &mut HashSet::new()))
            {
                return Err(invalid(
                    &self.programs[index],
                    "inputs".into(),
                    "cycle between programs".into(),
                ));
            }
        }
        Ok(())
    }

    fn expr_contains_time(expr: &Expr) -> bool {
        if matches!(expr, Expr::Time) {
            return true;
        }
        let mut found = false;
        expr.for_each_child(&mut |child| found |= Self::expr_contains_time(child));
        found
    }

    fn program_expr_numeric(
        expr: &Expr,
        scopes: &HashMap<&str, Scope<'_>>,
        decisions: &HashSet<AttrRef>,
    ) -> bool {
        match expr {
            Expr::Const { value } => !matches!(value, Value::Bool(_)),
            Expr::Attr { attr } => {
                decisions.contains(attr)
                    || scopes
                        .get(attr.component.as_str())
                        .and_then(|scope| scope.attribute_kind(&attr.attribute))
                        .is_some_and(|kind| kind != AttrKind::Bool)
            }
            Expr::PortAgg { agg, .. } => !matches!(agg, AggOp::All | AggOp::Any),
            Expr::Add { args } | Expr::Mul { args } | Expr::Min { args } | Expr::Max { args } => {
                args.iter()
                    .all(|arg| Self::program_expr_numeric(arg, scopes, decisions))
            }
            Expr::Sub { lhs, rhs } | Expr::Div { lhs, rhs } => {
                Self::program_expr_numeric(lhs, scopes, decisions)
                    && Self::program_expr_numeric(rhs, scopes, decisions)
            }
            Expr::Sin { arg }
            | Expr::Exp { arg }
            | Expr::PiecewiseLinear { arg, .. }
            | Expr::SmoothNoise { arg, .. } => Self::program_expr_numeric(arg, scopes, decisions),
            Expr::If {
                cond,
                then,
                otherwise,
            } => {
                Self::program_expr_boolean(cond, scopes, decisions)
                    && Self::program_expr_numeric(then, scopes, decisions)
                    && Self::program_expr_numeric(otherwise, scopes, decisions)
            }
            Expr::Time => true,
            Expr::StateActive { .. } | Expr::Cmp { .. } | Expr::Bool { .. } => false,
        }
    }

    fn program_expr_boolean(
        expr: &Expr,
        scopes: &HashMap<&str, Scope<'_>>,
        decisions: &HashSet<AttrRef>,
    ) -> bool {
        match expr {
            Expr::Const { value } => matches!(value, Value::Bool(_)),
            Expr::Attr { attr } => {
                !decisions.contains(attr)
                    && scopes
                        .get(attr.component.as_str())
                        .and_then(|scope| scope.attribute_kind(&attr.attribute))
                        .is_some_and(|kind| kind == AttrKind::Bool)
            }
            Expr::StateActive { .. } => true,
            Expr::PortAgg { agg, .. } => matches!(agg, AggOp::All | AggOp::Any),
            Expr::Cmp { lhs, rhs, .. } => {
                (Self::program_expr_numeric(lhs, scopes, decisions)
                    && Self::program_expr_numeric(rhs, scopes, decisions))
                    || (Self::program_expr_boolean(lhs, scopes, decisions)
                        && Self::program_expr_boolean(rhs, scopes, decisions))
            }
            Expr::Bool { args, .. } => args
                .iter()
                .all(|arg| Self::program_expr_boolean(arg, scopes, decisions)),
            Expr::If {
                cond,
                then,
                otherwise,
            } => {
                Self::program_expr_boolean(cond, scopes, decisions)
                    && Self::program_expr_boolean(then, scopes, decisions)
                    && Self::program_expr_boolean(otherwise, scopes, decisions)
            }
            _ => false,
        }
    }

    /// Everything that assigns to an attribute outside the distribution
    /// operators: `(component, attribute) → a description of the writer`,
    /// keeping the *first* writer in declaration order so the diagnostic
    /// is stable.
    fn attribute_writers(&self) -> HashMap<(&str, &str), String> {
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
    fn check_evaluation_order(&self) -> Result<(), ModelError> {
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

    fn check_component<'m>(
        component: &'m Component,
        materialised: &[&ChannelAttribute],
    ) -> Result<Scope<'m>, ModelError> {
        let mut attributes = HashMap::new();
        for attribute in &component.attributes {
            if attributes
                .insert(attribute.name.as_str(), attribute)
                .is_some()
            {
                return Err(ModelError::DuplicateInComponent {
                    kind: "attribute",
                    name: attribute.name.clone(),
                    component: component.name.clone(),
                });
            }
            if !attribute.kind.matches(&attribute.init) {
                return Err(ModelError::InitKindMismatch {
                    component: component.name.clone(),
                    attribute: attribute.name.clone(),
                    kind: attribute.kind,
                });
            }
        }

        let mut ports = HashMap::new();
        for port in &component.ports {
            if ports.insert(port.name.as_str(), port).is_some() {
                return Err(ModelError::DuplicateInComponent {
                    kind: "port",
                    name: port.name.clone(),
                    component: component.name.clone(),
                });
            }
            match (port.dir, &port.attr) {
                (PortDir::Out, None) => {
                    return Err(ModelError::OutPortWithoutVariable {
                        component: component.name.clone(),
                        port: port.name.clone(),
                    });
                }
                (PortDir::Out, Some(var)) if !attributes.contains_key(var.as_str()) => {
                    return Err(ModelError::PortUnknownVariable {
                        component: component.name.clone(),
                        port: port.name.clone(),
                        attribute: var.clone(),
                    });
                }
                (PortDir::In, Some(_)) => {
                    return Err(ModelError::InPortWithVariable {
                        component: component.name.clone(),
                        port: port.name.clone(),
                    });
                }
                _ => {}
            }
            if port.dir == PortDir::In && !port.channels.is_empty() {
                return Err(ModelError::InPortWithChannels {
                    component: component.name.clone(),
                    port: port.name.clone(),
                });
            }
            let mut channel_names = HashSet::new();
            for channel in &port.channels {
                if !channel_names.insert(channel.name.as_str()) {
                    return Err(ModelError::DuplicatePortChannel {
                        component: component.name.clone(),
                        port: port.name.clone(),
                        channel: channel.name.clone(),
                    });
                }
            }
        }

        // The compiler materialises one float attribute per (connection,
        // channel): refuse a name already taken by a declared attribute,
        // or claimed twice by two unnamed parallel edges. Both are silent
        // wrong answers otherwise, the second attribute quietly winning.
        let mut channel_attributes: HashSet<String> = HashSet::new();
        let mut origin: HashMap<&str, usize> = HashMap::new();
        for entry in materialised {
            if attributes.contains_key(entry.attribute.as_str()) {
                return Err(ModelError::ChannelAttributeCollision {
                    component: component.name.clone(),
                    port: entry.port.clone(),
                    channel: entry.channel.clone(),
                    attribute: entry.attribute.clone(),
                });
            }
            if let Some(first) = origin.insert(entry.attribute.as_str(), entry.connection) {
                return Err(ModelError::DuplicateChannelAttribute {
                    component: component.name.clone(),
                    attribute: entry.attribute.clone(),
                    channel: entry.channel.clone(),
                    first,
                    second: entry.connection,
                });
            }
            channel_attributes.insert(entry.attribute.clone());
        }

        let mut interface_names = HashSet::new();
        for interface in &component.interfaces {
            if !interface_names.insert(interface.name.as_str()) {
                return Err(ModelError::DuplicateInComponent {
                    kind: "interface",
                    name: interface.name.clone(),
                    component: component.name.clone(),
                });
            }
            for port in &interface.ports {
                if !ports.contains_key(port.as_str()) {
                    return Err(ModelError::InterfaceUnknownPort {
                        component: component.name.clone(),
                        interface: interface.name.clone(),
                        port: port.clone(),
                    });
                }
            }
        }

        let mut automata: HashMap<&str, HashSet<&str>> = HashMap::new();
        for automaton in &component.automata {
            if automata.contains_key(automaton.name.as_str()) {
                return Err(ModelError::DuplicateInComponent {
                    kind: "automaton",
                    name: automaton.name.clone(),
                    component: component.name.clone(),
                });
            }
            Self::check_automaton(component, automaton)?;
            automata.insert(
                automaton.name.as_str(),
                automaton.states.iter().map(String::as_str).collect(),
            );
        }
        Self::check_cycle_cancellation(component)?;

        let mut function_names = HashSet::new();
        for function in &component.sensitive_functions {
            if !function_names.insert(function.name.as_str()) {
                return Err(ModelError::DuplicateInComponent {
                    kind: "sensitive function",
                    name: function.name.clone(),
                    component: component.name.clone(),
                });
            }
        }

        let mut equation_targets = HashSet::new();
        let mut explicit_targets = HashSet::new();
        for equation in &component.equations {
            let kind = attributes
                .get(equation.target.as_str())
                .map(|attribute| attribute.kind)
                .or_else(|| {
                    channel_attributes
                        .contains(equation.target.as_str())
                        .then_some(AttrKind::Float)
                });
            let Some(kind) = kind else {
                return Err(ModelError::EquationUnknownTarget {
                    component: component.name.clone(),
                    target: equation.target.clone(),
                });
            };
            if kind != AttrKind::Float {
                return Err(ModelError::EquationTargetNotFloat {
                    component: component.name.clone(),
                    target: equation.target.clone(),
                });
            }
            if !equation_targets.insert(equation.target.as_str()) {
                return Err(ModelError::DuplicateEquation {
                    component: component.name.clone(),
                    target: equation.target.clone(),
                });
            }
            if equation.kind == EquationKind::Explicit {
                explicit_targets.insert(equation.target.as_str());
            }
        }

        // Conservative distribution operators: everything checkable
        // inside the component. Their consumer-keyed policy parameters
        // need the connection list and are checked by
        // [`Model::check_allocations`].
        let mut allocation_names = HashSet::new();
        for allocation in &component.allocations {
            if !allocation_names.insert(allocation.name.as_str()) {
                return Err(ModelError::DuplicateInComponent {
                    kind: "distribution operator",
                    name: allocation.name.clone(),
                    component: component.name.clone(),
                });
            }
            // An operator and an explicit equation are both steps of the
            // sweep, designated in the evaluation order by their name: two
            // steps of one component may not share one.
            if explicit_targets.contains(allocation.name.as_str()) {
                return Err(ModelError::EvaluationStepAmbiguous {
                    component: component.name.clone(),
                    name: allocation.name.clone(),
                });
            }
            let port = ports
                .get(allocation.port.as_str())
                .filter(|port| port.dir == PortDir::Out)
                .ok_or_else(|| ModelError::AllocationPortInvalid {
                    component: component.name.clone(),
                    allocation: allocation.name.clone(),
                    port: allocation.port.clone(),
                })?;
            for (role, channel) in [
                ("demand", &allocation.demand),
                ("allocated", &allocation.allocated),
            ] {
                if !port.channels.iter().any(|c| &c.name == channel) {
                    return Err(ModelError::AllocationUnknownChannel {
                        component: component.name.clone(),
                        allocation: allocation.name.clone(),
                        channel: channel.clone(),
                        role,
                    });
                }
            }
            if allocation.demand == allocation.allocated {
                return Err(ModelError::AllocationChannelReused {
                    component: component.name.clone(),
                    allocation: allocation.name.clone(),
                    channel: allocation.demand.clone(),
                });
            }
        }

        Ok(Scope {
            attributes,
            channel_attributes,
            ports,
            automata,
        })
    }

    /// A [`Transition::cycle_cancellation`] other than `auto` sits on a
    /// transition of a cycle group, and every transition of one
    /// `(component, cycle_group)` declares the same value: the decision is
    /// the group's, so two halves of it cannot disagree.
    fn check_cycle_cancellation(component: &Component) -> Result<(), ModelError> {
        let mut declared: HashMap<&str, (CycleCancellation, String)> = HashMap::new();
        for automaton in &component.automata {
            for transition in &automaton.transitions {
                let qualified =
                    || format!("{}.{}.{}", component.name, automaton.name, transition.name);
                let Some(group) = transition.cycle_group.as_deref() else {
                    if !transition.cycle_cancellation.is_auto() {
                        return Err(ModelError::CycleCancellationInvalid {
                            transition: qualified(),
                            reason: "the transition has no `cycle_group`, so there is no \
                                     cancellation to decide"
                                .to_owned(),
                        });
                    }
                    continue;
                };
                match declared.get(group) {
                    Some((value, first)) if *value != transition.cycle_cancellation => {
                        return Err(ModelError::CycleCancellationInvalid {
                            transition: qualified(),
                            reason: format!(
                                "`{first}` of the same cycle group `{group}` declares \
                                 another value; every transition of a group declares \
                                 the same one"
                            ),
                        });
                    }
                    Some(_) => {}
                    None => {
                        declared.insert(group, (transition.cycle_cancellation, qualified()));
                    }
                }
            }
        }
        Ok(())
    }

    fn check_automaton(component: &Component, automaton: &Automaton) -> Result<(), ModelError> {
        let mut states = HashSet::new();
        for state in &automaton.states {
            if !states.insert(state.as_str()) {
                return Err(ModelError::DuplicateInComponent {
                    kind: "state",
                    name: state.clone(),
                    component: component.name.clone(),
                });
            }
        }
        if !states.contains(automaton.init.as_str()) {
            return Err(ModelError::UnknownInitState {
                component: component.name.clone(),
                automaton: automaton.name.clone(),
                state: automaton.init.clone(),
            });
        }

        let mut transition_names = HashSet::new();
        for transition in &automaton.transitions {
            if !transition_names.insert(transition.name.as_str()) {
                return Err(ModelError::DuplicateInComponent {
                    kind: "transition",
                    name: transition.name.clone(),
                    component: component.name.clone(),
                });
            }
            if transition.guard.is_none() && transition.on_interruption != InterruptionPolicy::Reset
            {
                return Err(ModelError::InterruptionPolicyWithoutGuard {
                    component: component.name.clone(),
                    automaton: automaton.name.clone(),
                    transition: transition.name.clone(),
                    policy: match transition.on_interruption {
                        InterruptionPolicy::Resume => "resume",
                        _ => "continue",
                    },
                });
            }
            for state in std::iter::once(&transition.source).chain(&transition.targets) {
                if !states.contains(state.as_str()) {
                    return Err(ModelError::UnknownTransitionState {
                        component: component.name.clone(),
                        automaton: automaton.name.clone(),
                        transition: transition.name.clone(),
                        state: state.clone(),
                    });
                }
            }
            if let Some(recorded) = &transition.monitored_states {
                let reason = if !transition.monitored {
                    Some("the transition is not `monitored`, so nothing is recorded".to_owned())
                } else if recorded.is_empty() {
                    Some(
                        "the list is empty: a transition that records no target is                          declared `\"monitored\": false`"
                            .to_owned(),
                    )
                } else {
                    recorded
                        .iter()
                        .find(|state| !transition.targets.contains(state))
                        .map(|state| format!("`{state}` is not a target of the transition"))
                };
                if let Some(reason) = reason {
                    return Err(ModelError::MonitoredStatesInvalid {
                        transition: format!(
                            "{}.{}.{}",
                            component.name, automaton.name, transition.name
                        ),
                        reason,
                    });
                }
            }
            match &transition.distrib {
                Distrib::Delay { time } => {
                    if transition.targets.len() != 1 {
                        return Err(ModelError::DelayTargetCount {
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            targets: transition.targets.len(),
                        });
                    }
                    if !time.is_finite() || *time < 0.0 {
                        return Err(ModelError::InvalidDelay {
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            time: *time,
                        });
                    }
                }
                Distrib::Inst { probs } => {
                    if probs.len() + 1 != transition.targets.len() {
                        return Err(ModelError::InstArityMismatch {
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            targets: transition.targets.len(),
                            probs: probs.len(),
                        });
                    }
                    let sum: f64 = probs.iter().sum();
                    let each_valid = probs
                        .iter()
                        .all(|p| p.is_finite() && (0.0..=1.0).contains(p));
                    if !each_valid || sum > 1.0 {
                        return Err(ModelError::InvalidInstProbs {
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            sum,
                        });
                    }
                }
                Distrib::Exp { rate, rate_expr } => {
                    Self::check_single_target("exp", component, automaton, transition)?;
                    match (rate, rate_expr) {
                        (Some(rate), None) => {
                            // A rate of 0 is a transition that cannot
                            // fire, not an error.
                            if !rate.is_finite() || *rate < 0.0 {
                                return Err(ModelError::InvalidExpRate {
                                    component: component.name.clone(),
                                    automaton: automaton.name.clone(),
                                    transition: transition.name.clone(),
                                    rate: *rate,
                                });
                            }
                        }
                        (None, Some(_)) => {}
                        _ => {
                            return Err(ModelError::ExpRateSpec {
                                component: component.name.clone(),
                                automaton: automaton.name.clone(),
                                transition: transition.name.clone(),
                            });
                        }
                    }
                }
                Distrib::Weibull { shape, scale } => {
                    Self::check_single_target("weibull", component, automaton, transition)?;
                    Self::check_positive(
                        "weibull", "shape", *shape, component, automaton, transition,
                    )?;
                    Self::check_positive(
                        "weibull", "scale", *scale, component, automaton, transition,
                    )?;
                }
                Distrib::Lognormal { mu, sigma } => {
                    Self::check_single_target("lognormal", component, automaton, transition)?;
                    if !mu.is_finite() {
                        return Err(ModelError::InvalidLawParameter {
                            distribution: "lognormal",
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            parameter: "mu",
                            value: *mu,
                        });
                    }
                    Self::check_positive(
                        "lognormal",
                        "sigma",
                        *sigma,
                        component,
                        automaton,
                        transition,
                    )?;
                }
                Distrib::Normal {
                    mu,
                    sigma,
                    low,
                    high,
                } => {
                    Self::check_single_target("normal", component, automaton, transition)?;
                    let invalid =
                        |parameter: &'static str, value: f64| ModelError::InvalidLawParameter {
                            distribution: "normal",
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            parameter,
                            value,
                        };
                    if !mu.is_finite() {
                        return Err(invalid("mu", *mu));
                    }
                    Self::check_positive(
                        "normal", "sigma", *sigma, component, automaton, transition,
                    )?;
                    if !low.is_finite() || *low < 0.0 {
                        return Err(invalid("low", *low));
                    }
                    if let Some(high) = high {
                        if !high.is_finite() || *high <= *low {
                            return Err(invalid("high", *high));
                        }
                    }
                }
                Distrib::Gamma { shape, scale } => {
                    Self::check_single_target("gamma", component, automaton, transition)?;
                    Self::check_positive(
                        "gamma", "shape", *shape, component, automaton, transition,
                    )?;
                    Self::check_positive(
                        "gamma", "scale", *scale, component, automaton, transition,
                    )?;
                }
                Distrib::Uniform { low, high } => {
                    Self::check_single_target("uniform", component, automaton, transition)?;
                    if !low.is_finite() || *low < 0.0 {
                        return Err(ModelError::InvalidLawParameter {
                            distribution: "uniform",
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            parameter: "low",
                            value: *low,
                        });
                    }
                    if !high.is_finite() || *high <= *low {
                        return Err(ModelError::InvalidLawParameter {
                            distribution: "uniform",
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            parameter: "high",
                            value: *high,
                        });
                    }
                }
                Distrib::Empirical { points } => {
                    Self::check_single_target("empirical", component, automaton, transition)?;
                    let table_error = |detail: String| ModelError::InvalidEmpiricalTable {
                        component: component.name.clone(),
                        automaton: automaton.name.clone(),
                        transition: transition.name.clone(),
                        detail,
                    };
                    if points.is_empty() {
                        return Err(table_error("empty table".to_owned()));
                    }
                    let mut prev: Option<(f64, f64)> = None;
                    for &(t, c) in points {
                        if !t.is_finite() || t < 0.0 || !c.is_finite() || !(0.0..=1.0).contains(&c)
                        {
                            return Err(table_error(format!("invalid point ({t}, {c})")));
                        }
                        if let Some((pt, pc)) = prev {
                            if t < pt || c < pc {
                                return Err(table_error(format!(
                                    "non-monotone point ({t}, {c}) after ({pt}, {pc})"
                                )));
                            }
                        }
                        prev = Some((t, c));
                    }
                    if let Some((_, last)) = prev {
                        if last != 1.0 {
                            return Err(table_error(format!(
                                "last cumulative probability is {last}, expected 1"
                            )));
                        }
                    }
                }
                Distrib::Watched => {
                    if transition.targets.len() != 1 {
                        return Err(ModelError::WatchedTargetCount {
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            targets: transition.targets.len(),
                        });
                    }
                    match &transition.guard {
                        None => {
                            return Err(ModelError::WatchedGuardMissing {
                                component: component.name.clone(),
                                automaton: automaton.name.clone(),
                                transition: transition.name.clone(),
                            });
                        }
                        Some(guard) if Self::is_watched_guard(guard) => {}
                        Some(_) => {
                            return Err(ModelError::WatchedGuardNotComparison {
                                component: component.name.clone(),
                                automaton: automaton.name.clone(),
                                transition: transition.name.clone(),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// A watched guard must expose exactly one *ordering comparison*
    /// (the located boundary), optionally conjoined/disjoined with
    /// discrete gate expressions (which only change at discrete events):
    /// `cmp`, `and(gates…, cmp)` or `or(gates…, cmp)`.
    /// A watched guard must contain at least one ordering comparison
    /// (the continuous boundary) anywhere under `and`/`or`/`not`
    /// connectives (the other operands act as discrete gates). The
    /// margin compiler maps `and` to the `min` of the operand margins,
    /// `or` to the `max`, `not` to the negation, and any other boolean
    /// operand to ±1.
    fn is_watched_guard(guard: &Expr) -> bool {
        match guard {
            Expr::Cmp {
                cmp:
                    raichu_expr::CmpOp::Lt
                    | raichu_expr::CmpOp::Le
                    | raichu_expr::CmpOp::Gt
                    | raichu_expr::CmpOp::Ge,
                ..
            } => true,
            Expr::Bool {
                bool_op: raichu_expr::BoolOp::And | raichu_expr::BoolOp::Or,
                args,
            } => args.iter().any(Self::is_watched_guard),
            Expr::Bool {
                bool_op: raichu_expr::BoolOp::Not,
                args,
            } => args.iter().any(Self::is_watched_guard),
            _ => false,
        }
    }

    fn check_single_target(
        distribution: &'static str,
        component: &Component,
        automaton: &Automaton,
        transition: &Transition,
    ) -> Result<(), ModelError> {
        if transition.targets.len() != 1 {
            return Err(ModelError::StochasticTargetCount {
                distribution,
                component: component.name.clone(),
                automaton: automaton.name.clone(),
                transition: transition.name.clone(),
                targets: transition.targets.len(),
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)] // flat validation helper
    fn check_positive(
        distribution: &'static str,
        parameter: &'static str,
        value: f64,
        component: &Component,
        automaton: &Automaton,
        transition: &Transition,
    ) -> Result<(), ModelError> {
        if !value.is_finite() || value <= 0.0 {
            return Err(ModelError::InvalidLawParameter {
                distribution,
                component: component.name.clone(),
                automaton: automaton.name.clone(),
                transition: transition.name.clone(),
                parameter,
                value,
            });
        }
        Ok(())
    }

    fn check_connections(&self, scopes: &HashMap<&str, Scope<'_>>) -> Result<(), ModelError> {
        for connection in &self.connections {
            let from = Self::resolve_port(scopes, &connection.from, "from")?;
            let to = Self::resolve_port(scopes, &connection.to, "to")?;
            if from.dir != PortDir::Out || to.dir != PortDir::In {
                return Err(ModelError::ConnectionDirectionMismatch {
                    from_component: connection.from.component.clone(),
                    from_port: connection.from.port.clone(),
                    to_component: connection.to.component.clone(),
                    to_port: connection.to.port.clone(),
                });
            }
        }
        self.check_flow_families(scopes)
    }

    /// Refuse an in port fed by both a **discrete** flow and a
    /// **continuous** one (see [`ModelError::ConnectionFamilyMismatch`]).
    ///
    /// The family is read off the kind of the attribute each producer
    /// exports, which is the only family the model layer carries: `Bool`
    /// is discrete, `Int` and `Float` are quantities. A single producer
    /// is never refused, whatever its kind: what has no defensible
    /// answer is the *mixture*, because one aggregation then has to give
    /// one meaning to two families.
    ///
    /// Reports the first offending connection in declaration order,
    /// against the first producer of the same in port.
    fn check_flow_families(&self, scopes: &HashMap<&str, Scope<'_>>) -> Result<(), ModelError> {
        // in port → the first connection feeding it and the kind it carries.
        let mut first: HashMap<(&str, &str), (&Connection, AttrKind)> = HashMap::new();
        for connection in &self.connections {
            let Some(kind) = Self::exported_kind(scopes, &connection.from) else {
                continue;
            };
            let key = (
                connection.to.component.as_str(),
                connection.to.port.as_str(),
            );
            if let Some(&(other, other_kind)) = first.get(&key) {
                if is_discrete(other_kind) != is_discrete(kind) {
                    return Err(ModelError::ConnectionFamilyMismatch {
                        port: format!("{}.{}", connection.to.component, connection.to.port),
                        other: format!("{}.{}", other.from.component, other.from.port),
                        other_kind,
                        producer: format!("{}.{}", connection.from.component, connection.from.port),
                        kind,
                    });
                }
            } else {
                first.insert(key, (connection, kind));
            }
        }
        Ok(())
    }

    /// Kind of the attribute an out port exports, `None` when the port
    /// or its attribute does not resolve (reported by the other passes).
    fn exported_kind(scopes: &HashMap<&str, Scope<'_>>, port: &PortRef) -> Option<AttrKind> {
        let scope = scopes.get(port.component.as_str())?;
        let attribute = scope.ports.get(port.port.as_str())?.attr.as_ref()?;
        scope.attribute_kind(attribute)
    }

    fn resolve_port<'m>(
        scopes: &'m HashMap<&str, Scope<'m>>,
        port_ref: &PortRef,
        side: &'static str,
    ) -> Result<&'m Port, ModelError> {
        scopes
            .get(port_ref.component.as_str())
            .and_then(|scope| scope.ports.get(port_ref.port.as_str()))
            .copied()
            .ok_or_else(|| ModelError::ConnectionUnknownPort {
                component: port_ref.component.clone(),
                port: port_ref.port.clone(),
                side,
            })
    }

    fn check_expressions(&self, scopes: &HashMap<&str, Scope<'_>>) -> Result<(), ModelError> {
        let sources = self.in_port_sources();
        for component in &self.components {
            for automaton in &component.automata {
                for transition in &automaton.transitions {
                    if let Some(guard) = &transition.guard {
                        let context = format!(
                            "guard of transition `{}` in `{}.{}`",
                            transition.name, component.name, automaton.name
                        );
                        Self::check_expr(scopes, &sources, guard, &context)?;
                    }
                    if let Distrib::Exp {
                        rate_expr: Some(rate),
                        ..
                    } = &transition.distrib
                    {
                        let context = format!(
                            "rate of transition `{}` in `{}.{}`",
                            transition.name, component.name, automaton.name
                        );
                        Self::check_expr(scopes, &sources, rate, &context)?;
                    }
                    for (index, assignment) in transition.effects.iter().enumerate() {
                        let context = format!(
                            "effect #{index} of transition `{}` in `{}.{}`",
                            transition.name, component.name, automaton.name
                        );
                        Self::check_attr_ref(scopes, &assignment.target, &context)?;
                        Self::check_expr(scopes, &sources, &assignment.value, &context)?;
                    }
                }
            }
            for function in &component.sensitive_functions {
                for (index, assignment) in function.effects.iter().enumerate() {
                    let context = format!(
                        "effect #{index} of sensitive function `{}.{}`",
                        component.name, function.name
                    );
                    Self::check_attr_ref(scopes, &assignment.target, &context)?;
                    Self::check_expr(scopes, &sources, &assignment.value, &context)?;
                }
            }
            for equation in &component.equations {
                let context = format!("equation of `{}.{}`", component.name, equation.target);
                Self::check_expr(scopes, &sources, &equation.expr, &context)?;
            }
            for allocation in &component.allocations {
                let context = format!(
                    "available quantity of distribution operator `{}.{}`",
                    component.name, allocation.name
                );
                Self::check_expr(scopes, &sources, &allocation.available, &context)?;
            }
        }
        Ok(())
    }

    /// (in-port) → out-ports feeding it, in connection declaration order.
    /// Used to check a `port_agg` channel selector against what the
    /// producers on the other end of the edges actually declare.
    fn in_port_sources(&self) -> HashMap<(String, String), Vec<(String, String)>> {
        let mut sources: HashMap<(String, String), Vec<(String, String)>> = HashMap::new();
        for connection in &self.connections {
            sources
                .entry((connection.to.component.clone(), connection.to.port.clone()))
                .or_default()
                .push((
                    connection.from.component.clone(),
                    connection.from.port.clone(),
                ));
        }
        sources
    }

    fn check_expr(
        scopes: &HashMap<&str, Scope<'_>>,
        sources: &HashMap<(String, String), Vec<(String, String)>>,
        expr: &Expr,
        context: &str,
    ) -> Result<(), ModelError> {
        if let Expr::Bool { bool_op, args } = expr {
            if matches!(bool_op, raichu_expr::BoolOp::Not) && args.len() != 1 {
                return Err(ModelError::NotArity {
                    context: context.to_owned(),
                    args: args.len(),
                });
            }
        }
        match expr {
            Expr::Const { .. } => Ok(()),
            Expr::Attr { attr } => Self::check_attr_ref(scopes, attr, context),
            Expr::PortAgg { port, channel, .. } => {
                let resolved = scopes
                    .get(port.component.as_str())
                    .and_then(|scope| scope.ports.get(port.port.as_str()));
                match resolved {
                    Some(p) if p.dir == PortDir::In => {}
                    _ => {
                        return Err(ModelError::ExprBadPortAgg {
                            context: context.to_owned(),
                            component: port.component.clone(),
                            port: port.port.clone(),
                        });
                    }
                }
                let Some(channel) = channel else {
                    return Ok(());
                };
                // A channel selector reads the per-connection attributes
                // materialised for it, so every producer on the other end
                // of an edge must declare it. An in port with *no*
                // connection aggregates the empty set and stays legal:
                // no-connection defaults are relied on throughout.
                let edges = sources.get(&(port.component.clone(), port.port.clone()));
                for (from_component, from_port) in edges.into_iter().flatten() {
                    let declares = scopes
                        .get(from_component.as_str())
                        .and_then(|scope| scope.ports.get(from_port.as_str()))
                        .is_some_and(|source| source.channels.iter().any(|c| &c.name == channel));
                    if !declares {
                        return Err(ModelError::ExprUnknownPortChannel {
                            context: context.to_owned(),
                            port: format!("{}.{}", port.component, port.port),
                            channel: channel.clone(),
                            producer: format!("{from_component}.{from_port}"),
                        });
                    }
                }
                Ok(())
            }
            Expr::StateActive { state } => {
                let known = scopes
                    .get(state.component.as_str())
                    .and_then(|scope| scope.automata.get(state.automaton.as_str()))
                    .is_some_and(|states| states.contains(state.state.as_str()));
                if known {
                    Ok(())
                } else {
                    Err(ModelError::ExprUnknownState {
                        context: context.to_owned(),
                        component: state.component.clone(),
                        automaton: state.automaton.clone(),
                        state: state.state.clone(),
                    })
                }
            }
            Expr::Cmp { lhs, rhs, .. } | Expr::Sub { lhs, rhs } | Expr::Div { lhs, rhs } => {
                Self::check_expr(scopes, sources, lhs, context)?;
                Self::check_expr(scopes, sources, rhs, context)
            }
            Expr::Bool { args, .. } => {
                for arg in args {
                    Self::check_expr(scopes, sources, arg, context)?;
                }
                Ok(())
            }
            Expr::Add { args } | Expr::Mul { args } | Expr::Min { args } | Expr::Max { args } => {
                if args.is_empty() {
                    return Err(ModelError::EmptyExprArgs {
                        context: context.to_owned(),
                        op: match expr {
                            Expr::Add { .. } => "add",
                            Expr::Mul { .. } => "mul",
                            Expr::Min { .. } => "min",
                            _ => "max",
                        },
                    });
                }
                for arg in args {
                    Self::check_expr(scopes, sources, arg, context)?;
                }
                Ok(())
            }
            Expr::If {
                cond,
                then,
                otherwise,
            } => {
                Self::check_expr(scopes, sources, cond, context)?;
                Self::check_expr(scopes, sources, then, context)?;
                Self::check_expr(scopes, sources, otherwise, context)
            }
            Expr::Sin { arg } | Expr::Exp { arg } => {
                Self::check_expr(scopes, sources, arg, context)
            }
            Expr::PiecewiseLinear {
                arg,
                points,
                period,
                phase_shift,
            } => {
                raichu_expr::signal::check_periodic_piecewise_linear(points, *period, *phase_shift)
                    .map_err(|detail| ModelError::InvalidTimeSignal {
                        context: context.to_owned(),
                        op: "piecewise_linear",
                        detail,
                    })?;
                Self::check_expr(scopes, sources, arg, context)
            }
            Expr::SmoothNoise {
                arg,
                amplitude,
                time_scale,
                ..
            } => {
                raichu_expr::signal::check_smooth_noise(*amplitude, *time_scale).map_err(
                    |detail| ModelError::InvalidTimeSignal {
                        context: context.to_owned(),
                        op: "smooth_noise",
                        detail,
                    },
                )?;
                Self::check_expr(scopes, sources, arg, context)
            }
            Expr::Time => Ok(()),
        }
    }

    fn check_attr_ref(
        scopes: &HashMap<&str, Scope<'_>>,
        var_ref: &AttrRef,
        context: &str,
    ) -> Result<(), ModelError> {
        let known = scopes
            .get(var_ref.component.as_str())
            .is_some_and(|scope| scope.attribute_kind(&var_ref.attribute).is_some());
        if known {
            Ok(())
        } else {
            Err(ModelError::ExprUnknownVariable {
                context: context.to_owned(),
                component: var_ref.component.clone(),
                attribute: var_ref.attribute.clone(),
            })
        }
    }

    fn check_indicators(&self, scopes: &HashMap<&str, Scope<'_>>) -> Result<(), ModelError> {
        let mut names = HashSet::new();
        for indicator in &self.indicators {
            if !names.insert(indicator.name.as_str()) {
                return Err(ModelError::DuplicateIndicator {
                    name: indicator.name.clone(),
                });
            }
            match &indicator.target {
                IndicatorTarget::Attribute { attr } => {
                    let context = format!("target of indicator `{}`", indicator.name);
                    Self::check_attr_ref(scopes, attr, &context).map_err(|_| {
                        ModelError::IndicatorUnresolved {
                            indicator: indicator.name.clone(),
                            detail: format!(
                                "unknown attribute `{}.{}`",
                                attr.component, attr.attribute
                            ),
                        }
                    })?;
                }
                IndicatorTarget::State {
                    component,
                    automaton,
                    state,
                } => {
                    let found = self
                        .components
                        .iter()
                        .find(|c| &c.name == component)
                        .and_then(|c| c.automata.iter().find(|a| &a.name == automaton))
                        .is_some_and(|a| a.states.contains(state));
                    if !found {
                        return Err(ModelError::IndicatorUnresolved {
                            indicator: indicator.name.clone(),
                            detail: format!("unknown state `{component}.{automaton}.{state}`"),
                        });
                    }
                }
                IndicatorTarget::Predicate { attr, cmp, value } => {
                    let kind = scopes
                        .get(attr.component.as_str())
                        .and_then(|scope| scope.attribute_kind(&attr.attribute))
                        .ok_or_else(|| ModelError::IndicatorUnresolved {
                            indicator: indicator.name.clone(),
                            detail: format!(
                                "unknown attribute `{}.{}`",
                                attr.component, attr.attribute
                            ),
                        })?;
                    Self::check_predicate_kinds(&indicator.name, attr, kind, *cmp, value)?;
                }
            }
        }
        // scopes is only used through the helpers above; keep the
        // signature symmetric with the other passes.
        let _ = scopes;
        Ok(())
    }

    /// Refuse an indicator threshold whose two sides do not compare.
    ///
    /// The engine's own [`raichu_expr::CmpOp`] semantics are the rule
    /// being enforced ahead of time: booleans compare for equality only,
    /// numbers compare across `int`/`float`, and the two families do not
    /// mix. Doing it here rather than at record time is what lets the
    /// three recording sites stay infallible, and it is also the more
    /// useful moment: a threshold that can never be evaluated is a
    /// modelling mistake, not a run-time event.
    fn check_predicate_kinds(
        indicator: &str,
        attr: &AttrRef,
        kind: AttrKind,
        cmp: CmpOp,
        value: &Value,
    ) -> Result<(), ModelError> {
        let numeric_attr = matches!(kind, AttrKind::Int | AttrKind::Float);
        let numeric_value = matches!(value, Value::Int(_) | Value::Float(_));
        if numeric_attr != numeric_value {
            return Err(ModelError::IndicatorPredicateKind {
                indicator: indicator.to_owned(),
                detail: format!(
                    "the threshold compares `{}.{}`, declared {kind:?}, with {value:?}: a \
                     boolean and a number have no comparison",
                    attr.component, attr.attribute
                ),
            });
        }
        if !numeric_attr && !matches!(cmp, CmpOp::Eq | CmpOp::Ne) {
            return Err(ModelError::IndicatorPredicateKind {
                indicator: indicator.to_owned(),
                detail: format!(
                    "the threshold orders `{}.{}` with {cmp:?}, and it is declared {kind:?}: a \
                     boolean is tested for equality (`eq`, `ne`) and never ordered",
                    attr.component, attr.attribute
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
