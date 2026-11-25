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

mod error;
mod format;
mod graph;
mod schema;

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

use graph::has_perfect_matching;
use raichu_expr::{AggOp, Assignment, AttrRef, CmpOp, Expr, PortRef, Value};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

/// Flow family of a value kind: a boolean carries a **discrete** flow
/// (present or absent), an integer or a float a **continuous** quantity.
/// The two never share one aggregation (see
/// [`ModelError::ConnectionFamilyMismatch`]).
fn is_discrete(kind: AttrKind) -> bool {
    matches!(kind, AttrKind::Bool)
}

/// One node of the instantaneous dependency graph: the qualified name
/// `(component, attribute)` of an attribute an explicit equation
/// computes.
type EquationNode = (String, String);

/// Which level of the singularity analysis refused an algebraic cycle
/// (see [`ModelError::AlgebraicSingular`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingularReason {
    /// No assignment of equations to unknowns exists on the block's
    /// sparsity pattern: some attribute nothing determines (or two
    /// equations determine the same one). Refused by the model layer,
    /// which sees the pattern.
    Structural,
    /// The sparsity pattern admits an assignment, but with every
    /// coefficient constant at build time the factorisation of the
    /// system is rank-deficient (`x = y` with `y = x`, or `x = x`).
    /// Refused by the compiler, which runs the factorisation.
    RankDeficient,
}

impl std::fmt::Display for SingularReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SingularReason::Structural => write!(
                f,
                "no assignment of equations to unknowns exists on the \
                 sparsity pattern"
            ),
            SingularReason::RankDeficient => write!(
                f,
                "every coefficient is constant and the constant system is \
                 rank-deficient"
            ),
        }
    }
}

/// Whether `expr` is the literal zero: the one coefficient the sparsity
/// pattern counts as absent. Anything else : a computed expression, an
/// `if`, an attribute read : counts as non-zero, because whether it can
/// vanish is a question about the state, and structure is decided
/// without one.
fn is_literal_zero(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Const {
            value: Value::Float(0.0) | Value::Int(0)
        }
    )
}

/// Render `component.attribute` names, comma-joined, for a diagnostic
/// that names a whole block.
fn render_variables(variables: &[AttrRef]) -> String {
    variables
        .iter()
        .map(|attr| format!("{}.{}", attr.component, attr.attribute))
        .collect::<Vec<_>>()
        .join(", ")
}

/// One equation of a linear algebraic block: its right-hand side
/// separated into an unknown-free part and one coefficient per block
/// unknown it reads.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockRow {
    /// The unknown-free part of the right-hand side.
    pub constant: Expr,
    /// One `(unknown, coefficient)` pair per block unknown this row
    /// reads, in first-occurrence order, with repeated coefficients merged.
    /// Coefficients retain their expression form, including literal zeros.
    pub terms: Vec<(AttrRef, Expr)>,
}

/// A cycle of explicit equations that is **linear in its unknowns**: the
/// unit of simultaneous evaluation the engine solves in place.
///
/// The `n` equations write the `n` attributes of [`AlgebraicBlock::unknowns`]
/// (same order), so the block is the system `x = c + A·x`: row `i` of
/// [`AlgebraicBlock::rows`] gives the unknown-free part `c_i` and the
/// non-zero coefficients of `A`'s row `i`. The engine solves
/// `(I − A)·x = c` wherever the sweep runs.
#[derive(Debug, Clone, PartialEq)]
pub struct AlgebraicBlock {
    /// The block's unknowns: the target attribute of each member
    /// equation, in the member equations' declaration order.
    pub unknowns: Vec<AttrRef>,
    /// One row per member equation, in the same order as `unknowns`.
    pub rows: Vec<BlockRow>,
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

    /// Every port connection the model runs on: the explicit
    /// [`Model::connections`] in declaration order, followed by the
    /// expansion of each [`InterfaceConnection`] in declaration order.
    ///
    /// Borrowed, at no cost, when the model declares no interface
    /// connection. An interface connection is expanded port by port in
    /// the order the `from` interface lists its ports, then the `to`
    /// interface's ports that point back: an out port of `from` becomes
    /// `from → to`, an in port of `from` becomes `to → from`.
    ///
    /// # Errors
    ///
    /// The interface does not exist ([`ModelError::InterfaceConnectionUnknown`]),
    /// or the two interfaces do not pair exactly by port name
    /// ([`ModelError::InterfacePortUnmatched`],
    /// [`ModelError::InterfacePortDirection`]). A port an interface lists
    /// but its component lacks is reported by [`Model::validate`] as
    /// [`ModelError::InterfaceUnknownPort`].
    pub fn resolved_connections(&self) -> Result<Cow<'_, [Connection]>, ModelError> {
        if self.interface_connections.is_empty() {
            return Ok(Cow::Borrowed(&self.connections));
        }
        let mut out = self.connections.clone();
        for link in &self.interface_connections {
            out.extend(self.expand_interface_connection(link)?);
        }
        Ok(Cow::Owned(out))
    }

    /// This model with its authoring forms resolved: interface connections
    /// expanded into [`Model::connections`] and the list emptied, then port
    /// equations and out-port aggregations lowered to explicit equations
    /// ([`PortEquation`]). The model the engine actually runs. Borrowed
    /// when there is nothing to resolve.
    ///
    /// # Errors
    ///
    /// As [`Model::resolved_connections`], and the port-equation refusals
    /// ([`ModelError::PortEquationUnknownChannel`],
    /// [`ModelError::PortEquationUnknownEdgeChannel`],
    /// [`ModelError::OutPortAggregation`]).
    pub fn with_resolved_connections(&self) -> Result<Cow<'_, Model>, ModelError> {
        if !self.needs_resolution() {
            return Ok(Cow::Borrowed(self));
        }
        let connections = self.resolved_connections()?.into_owned();
        let mut model = Model {
            connections,
            interface_connections: Vec::new(),
            ..self.clone()
        };
        if model.has_port_constructs() {
            model.lower_port_constructs()?;
        }
        if model.has_port_effects() {
            model.lower_port_effects()?;
        }
        Ok(Cow::Owned(model))
    }

    /// Whether [`Model::with_resolved_connections`] has anything to do.
    fn needs_resolution(&self) -> bool {
        !self.interface_connections.is_empty()
            || self.has_port_constructs()
            || self.has_port_effects()
    }

    /// Whether an edge effect or a sensitive function writes through a
    /// port ([`PORT`]).
    fn has_port_effects(&self) -> bool {
        let through = |effects: &[Assignment]| effects.iter().any(|a| a.target.component == PORT);
        self.components.iter().any(|c| {
            c.automata
                .iter()
                .any(|a| a.transitions.iter().any(|t| through(&t.effects)))
                || c.sensitive_functions.iter().any(|f| through(&f.effects))
        })
    }

    /// Lower every edge effect written through a port to one ordinary edge
    /// effect per connection feeding that port, on the attribute the
    /// connected out port exports, in connection declaration order. The
    /// lowered effects take the place of the one they come from, so the
    /// order of a transition's effects is kept.
    fn lower_port_effects(&mut self) -> Result<(), ModelError> {
        let exported: HashMap<(&str, &str), &str> = self
            .components
            .iter()
            .flat_map(|c| {
                c.ports
                    .iter()
                    .filter(|p| p.dir == PortDir::Out)
                    .filter_map(move |p| {
                        p.attr
                            .as_deref()
                            .map(|attr| ((c.name.as_str(), p.name.as_str()), attr))
                    })
            })
            .collect();
        let mut sources: HashMap<(String, String), Vec<AttrRef>> = HashMap::new();
        for connection in &self.connections {
            let from = (
                connection.from.component.as_str(),
                connection.from.port.as_str(),
            );
            if let Some(attribute) = exported.get(&from) {
                sources
                    .entry((connection.to.component.clone(), connection.to.port.clone()))
                    .or_default()
                    .push(AttrRef {
                        component: connection.from.component.clone(),
                        attribute: (*attribute).to_owned(),
                    });
            }
        }
        for component in &mut self.components {
            if let Some(function) = component
                .sensitive_functions
                .iter()
                .find(|f| f.effects.iter().any(|a| a.target.component == PORT))
            {
                return Err(ModelError::PortEffectInSensitiveFunction {
                    function: format!("{}.{}", component.name, function.name),
                });
            }
            let in_ports: HashSet<&str> = component
                .ports
                .iter()
                .filter(|p| p.dir == PortDir::In)
                .map(|p| p.name.as_str())
                .collect();
            for automaton in &mut component.automata {
                for transition in &mut automaton.transitions {
                    if !transition
                        .effects
                        .iter()
                        .any(|a| a.target.component == PORT)
                    {
                        continue;
                    }
                    let mut lowered = Vec::with_capacity(transition.effects.len());
                    for assignment in transition.effects.drain(..) {
                        if assignment.target.component != PORT {
                            lowered.push(assignment);
                            continue;
                        }
                        let port = assignment.target.attribute;
                        if !in_ports.contains(port.as_str()) {
                            return Err(ModelError::PortEffectNotAnInPort {
                                transition: format!(
                                    "transition `{}` in `{}.{}`",
                                    transition.name, component.name, automaton.name
                                ),
                                port,
                            });
                        }
                        let key = (component.name.clone(), port);
                        for source in sources.get(&key).map(Vec::as_slice).unwrap_or_default() {
                            lowered.push(Assignment {
                                target: source.clone(),
                                value: assignment.value.clone(),
                            });
                        }
                    }
                    transition.effects = lowered;
                }
            }
        }
        Ok(())
    }

    /// Whether the model uses a port equation or aggregates an out port.
    fn has_port_constructs(&self) -> bool {
        if self
            .components
            .iter()
            .any(|c| c.ports.iter().any(|p| !p.equations.is_empty()))
        {
            return true;
        }
        let out_ports = self.out_ports();
        let mut found = false;
        self.for_each_expr(&mut |_, expr| {
            // Only a channel-named aggregation is the construct; one without
            // a channel stays the malformed read the validation refuses.
            expr.for_each_port_agg(&mut |port, channel| {
                found |= channel.is_some()
                    && out_ports.contains(&(port.component.as_str(), port.port.as_str()));
            });
        });
        found
    }

    fn out_ports(&self) -> HashSet<(&str, &str)> {
        self.components
            .iter()
            .flat_map(|c| {
                c.ports
                    .iter()
                    .filter(|p| p.dir == PortDir::Out)
                    .map(move |p| (c.name.as_str(), p.name.as_str()))
            })
            .collect()
    }

    /// Visit every expression an out-port aggregation may appear in, with
    /// a description of where it is.
    fn for_each_expr(&self, f: &mut impl FnMut(&dyn Fn() -> String, &Expr)) {
        for component in &self.components {
            for automaton in &component.automata {
                for transition in &automaton.transitions {
                    let here = || {
                        format!(
                            "transition `{}` in `{}.{}`",
                            transition.name, component.name, automaton.name
                        )
                    };
                    if let Some(guard) = &transition.guard {
                        f(&here, guard);
                    }
                    if let Distrib::Exp {
                        rate_expr: Some(rate),
                        ..
                    } = &transition.distrib
                    {
                        f(&here, rate);
                    }
                    for assignment in &transition.effects {
                        f(&here, &assignment.value);
                    }
                }
            }
            for function in &component.sensitive_functions {
                let here = || format!("sensitive function `{}.{}`", component.name, function.name);
                for assignment in &function.effects {
                    f(&here, &assignment.value);
                }
            }
            for equation in &component.equations {
                let here = || format!("equation of `{}.{}`", component.name, equation.target);
                f(&here, &equation.expr);
            }
            for allocation in &component.allocations {
                let here = || {
                    format!(
                        "available quantity of distribution operator `{}.{}`",
                        component.name, allocation.name
                    )
                };
                f(&here, &allocation.available);
            }
            for port in &component.ports {
                for equation in &port.equations {
                    let here = || {
                        format!(
                            "port equation `{}` of `{}.{}`",
                            equation.channel, component.name, port.name
                        )
                    };
                    f(&here, &equation.expr);
                }
            }
        }
    }

    /// Visit mutably the same expressions as [`Model::for_each_expr`]: one
    /// slot list, kept beside it so the two cannot drift apart.
    fn for_each_expr_mut(&mut self, f: &mut impl FnMut(&mut Expr)) {
        for component in &mut self.components {
            for automaton in &mut component.automata {
                for transition in &mut automaton.transitions {
                    if let Some(guard) = &mut transition.guard {
                        f(guard);
                    }
                    if let Distrib::Exp {
                        rate_expr: Some(rate),
                        ..
                    } = &mut transition.distrib
                    {
                        f(rate);
                    }
                    for assignment in &mut transition.effects {
                        f(&mut assignment.value);
                    }
                }
            }
            for function in &mut component.sensitive_functions {
                for assignment in &mut function.effects {
                    f(&mut assignment.value);
                }
            }
            for equation in &mut component.equations {
                f(&mut equation.expr);
            }
            for allocation in &mut component.allocations {
                f(&mut allocation.available);
            }
            for port in &mut component.ports {
                for equation in &mut port.equations {
                    f(&mut equation.expr);
                }
            }
        }
    }

    /// The connections through each port, by `(component, port)` of either
    /// end, in declaration order.
    fn connections_by_port(&self) -> HashMap<(&str, &str), Vec<&Connection>> {
        let mut through: HashMap<(&str, &str), Vec<&Connection>> = HashMap::new();
        for connection in &self.connections {
            through
                .entry((
                    connection.from.component.as_str(),
                    connection.from.port.as_str(),
                ))
                .or_default()
                .push(connection);
            through
                .entry((
                    connection.to.component.as_str(),
                    connection.to.port.as_str(),
                ))
                .or_default()
                .push(connection);
        }
        through
    }

    /// Lower every port equation to one explicit equation per connection,
    /// then every out-port aggregation to an explicit sum or count, in
    /// place. Port equations lower in component, port and equation
    /// declaration order, connections in declaration order; each lands on
    /// the producer, after its own equations.
    fn lower_port_constructs(&mut self) -> Result<(), ModelError> {
        let declared: HashMap<(&str, &str), Vec<&str>> = self
            .components
            .iter()
            .flat_map(|c| {
                c.ports
                    .iter()
                    .filter(|p| p.dir == PortDir::Out)
                    .map(move |p| {
                        (
                            (c.name.as_str(), p.name.as_str()),
                            p.channels.iter().map(|ch| ch.name.as_str()).collect(),
                        )
                    })
            })
            .collect();
        let through = self.connections_by_port();
        let index: HashMap<&str, usize> = self
            .components
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.as_str(), i))
            .collect();
        let mut lowered: Vec<(usize, Equation)> = Vec::new();
        for component in &self.components {
            for port in &component.ports {
                let here = format!("{}.{}", component.name, port.name);
                let connections = through
                    .get(&(component.name.as_str(), port.name.as_str()))
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .filter(|k| match port.dir {
                        PortDir::Out => {
                            k.from.component == component.name && k.from.port == port.name
                        }
                        PortDir::In => k.to.component == component.name && k.to.port == port.name,
                    });
                let connections: Vec<&Connection> = connections.copied().collect();
                for equation in &port.equations {
                    // An out port checks its own channels even when nothing
                    // is connected to it yet.
                    if port.dir == PortDir::Out
                        && !declared[&(component.name.as_str(), port.name.as_str())]
                            .contains(&equation.channel.as_str())
                    {
                        return Err(ModelError::PortEquationUnknownChannel {
                            port: here.clone(),
                            channel: equation.channel.clone(),
                            producer_port: here,
                        });
                    }
                    for connection in &connections {
                        let producer = (
                            connection.from.component.as_str(),
                            connection.from.port.as_str(),
                        );
                        let channels: &[&str] = declared.get(&producer).map_or(&[], Vec::as_slice);
                        let unknown = |channel: &str| !channels.contains(&channel);
                        let producer_port = || format!("{}.{}", producer.0, producer.1);
                        if unknown(&equation.channel) {
                            return Err(ModelError::PortEquationUnknownChannel {
                                port: here,
                                channel: equation.channel.clone(),
                                producer_port: producer_port(),
                            });
                        }
                        let mut expr = equation.expr.clone();
                        let mut missing: Option<String> = None;
                        expr.rewrite(&mut |node| {
                            if let Expr::Attr { attr } = node {
                                if attr.component == EDGE {
                                    if unknown(&attr.attribute) {
                                        missing.get_or_insert_with(|| attr.attribute.clone());
                                    }
                                    *attr = AttrRef {
                                        component: connection.from.component.clone(),
                                        attribute: channel_attribute_name(
                                            connection,
                                            &attr.attribute,
                                        ),
                                    };
                                }
                            }
                        });
                        if let Some(channel) = missing {
                            return Err(ModelError::PortEquationUnknownEdgeChannel {
                                port: here,
                                channel,
                                producer_port: producer_port(),
                            });
                        }
                        if let Some(&target) = index.get(producer.0) {
                            lowered.push((
                                target,
                                Equation {
                                    target: channel_attribute_name(connection, &equation.channel),
                                    kind: EquationKind::Explicit,
                                    expr,
                                },
                            ));
                        }
                    }
                }
            }
        }
        for component in &mut self.components {
            for port in &mut component.ports {
                port.equations.clear();
            }
        }
        for (target, equation) in lowered {
            self.components[target].equations.push(equation);
        }
        self.lower_out_port_aggregations()
    }

    /// Rewrite every `port_agg` over an out port as the explicit sum (or
    /// count) of the named channel over the port's connections.
    fn lower_out_port_aggregations(&mut self) -> Result<(), ModelError> {
        // Each out port with the connections leaving it, in declaration
        // order; owned, so the rewrite below can borrow it while it mutates
        // the components.
        let mut leaving: HashMap<(String, String), Vec<Connection>> = self
            .out_ports()
            .into_iter()
            .map(|(c, p)| ((c.to_owned(), p.to_owned()), Vec::new()))
            .collect();
        if leaving.is_empty() {
            return Ok(());
        }
        for connection in &self.connections {
            let key = (
                connection.from.component.clone(),
                connection.from.port.clone(),
            );
            if let Some(edges) = leaving.get_mut(&key) {
                edges.push(connection.clone());
            }
        }
        let is_out =
            |port: &PortRef| leaving.contains_key(&(port.component.clone(), port.port.clone()));
        // Refuse first, over the unmodified model, so the error names
        // where the aggregation was written.
        let mut refusal: Option<ModelError> = None;
        self.for_each_expr(&mut |here, expr| {
            Self::find_out_aggregation(expr, &is_out, &mut |port, agg| {
                if refusal.is_none() && !matches!(agg, AggOp::Sum | AggOp::Count) {
                    refusal = Some(ModelError::OutPortAggregation {
                        context: here(),
                        component: port.component.clone(),
                        port: port.port.clone(),
                        agg: agg.name().to_owned(),
                    });
                }
            });
        });
        if let Some(error) = refusal {
            return Err(error);
        }
        self.for_each_expr_mut(&mut |expr| {
            expr.rewrite(&mut |node| {
                let Expr::PortAgg {
                    port,
                    agg,
                    channel: Some(channel),
                } = node
                else {
                    return;
                };
                let Some(edges) = leaving.get(&(port.component.clone(), port.port.clone())) else {
                    return;
                };
                *node = match agg {
                    AggOp::Count => Expr::Const {
                        value: Value::Int(i64::try_from(edges.len()).unwrap_or(i64::MAX)),
                    },
                    _ if edges.is_empty() => Expr::Const {
                        value: Value::Int(0),
                    },
                    _ => Expr::Add {
                        args: edges
                            .iter()
                            .map(|k| Expr::Attr {
                                attr: AttrRef {
                                    component: k.from.component.clone(),
                                    attribute: channel_attribute_name(k, channel),
                                },
                            })
                            .collect(),
                    },
                };
            });
        });
        Ok(())
    }

    /// Calls `f` on every channel-named aggregation of an out port in `expr`.
    fn find_out_aggregation(
        expr: &Expr,
        is_out: &impl Fn(&PortRef) -> bool,
        f: &mut impl FnMut(&PortRef, AggOp),
    ) {
        if let Expr::PortAgg {
            port,
            agg,
            channel: Some(_),
        } = expr
        {
            if is_out(port) {
                f(port, *agg);
            }
        }
        expr.for_each_child(&mut |child| Self::find_out_aggregation(child, is_out, f));
    }

    fn expand_interface_connection(
        &self,
        link: &InterfaceConnection,
    ) -> Result<Vec<Connection>, ModelError> {
        let side = |reference: &InterfaceRef, side: &'static str| {
            let unknown = || ModelError::InterfaceConnectionUnknown {
                component: reference.component.clone(),
                interface: reference.interface.clone(),
                side,
            };
            let component = self
                .components
                .iter()
                .find(|c| c.name == reference.component)
                .ok_or_else(unknown)?;
            let interface = component
                .interfaces
                .iter()
                .find(|i| i.name == reference.interface)
                .ok_or_else(unknown)?;
            let ports: Vec<(&str, Option<PortDir>)> = interface
                .ports
                .iter()
                .map(|name| {
                    let dir = component
                        .ports
                        .iter()
                        .find(|p| &p.name == name)
                        .map(|p| p.dir);
                    (name.as_str(), dir)
                })
                .collect();
            Ok::<_, ModelError>(ports)
        };
        let from = side(&link.from, "from")?;
        let to = side(&link.to, "to")?;
        let unmatched = |component: &str, port: &str| ModelError::InterfacePortUnmatched {
            from: Box::new(link.from.clone()),
            to: Box::new(link.to.clone()),
            component: component.to_owned(),
            port: port.to_owned(),
        };
        for (port, _) in &to {
            if !from.iter().any(|(name, _)| name == port) {
                return Err(unmatched(&link.to.component, port));
            }
        }
        let mut out = Vec::with_capacity(from.len());
        for (port, dir) in &from {
            let Some((_, other)) = to.iter().find(|(name, _)| name == port) else {
                return Err(unmatched(&link.from.component, port));
            };
            let (source, target) = match (dir, other) {
                (Some(PortDir::Out), Some(PortDir::In)) => (&link.from, &link.to),
                (Some(PortDir::In), Some(PortDir::Out)) => (&link.to, &link.from),
                (Some(same), Some(_)) => {
                    return Err(ModelError::InterfacePortDirection {
                        from: Box::new(link.from.clone()),
                        to: Box::new(link.to.clone()),
                        port: (*port).to_owned(),
                        dir: match same {
                            PortDir::In => "in",
                            PortDir::Out => "out",
                        },
                    });
                }
                // A listed port the component lacks: validation reports
                // it as `InterfaceUnknownPort`, so nothing is expanded.
                _ => continue,
            };
            out.push(Connection {
                name: link.name.clone(),
                from: PortRef {
                    component: source.component.clone(),
                    port: (*port).to_owned(),
                },
                to: PortRef {
                    component: target.component.clone(),
                    port: (*port).to_owned(),
                },
            });
        }
        Ok(out)
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

    /// Classify every cycle of the explicit equations (see
    /// [`Model::algebraic_blocks`]).
    ///
    /// The graph has **one node per explicit-equation target** and an
    /// edge from a target to every explicit-equation target its
    /// right-hand side reads, in-port aggregations resolved through the
    /// connections feeding them. An ODE target is deliberately *not* a
    /// node: the integrator carries its value, so it breaks any cycle it
    /// sits on, which is exactly the capacity carve-out. An **allocated
    /// quantity** ([`Allocation`]) is not a node either, and that
    /// carve-out is equally deliberate: a conservative flow network is
    /// cyclic by nature (what a consumer asks for depends on what it was
    /// given), and refusing it here would refuse the very shape the
    /// operator exists for.
    ///
    /// Strongly connected components of this graph are the cycles.
    /// Tarjan's algorithm partitions them in one depth-first pass, in a
    /// deterministic order for a fixed declaration order. Each component
    /// is then classified: nonlinear in its unknowns (refused, naming
    /// the term), structurally singular (refused, naming the
    /// variables), or a linear block handed back to the compiler.
    fn classify_algebraic(
        &self,
        scopes: &HashMap<&str, Scope<'_>>,
    ) -> Result<Vec<AlgebraicBlock>, ModelError> {
        let mut nodes: Vec<EquationNode> = Vec::new();
        let mut index: HashMap<EquationNode, usize> = HashMap::new();
        let mut equations: HashMap<EquationNode, &Equation> = HashMap::new();
        for component in &self.components {
            for equation in &component.equations {
                if equation.kind == EquationKind::Explicit {
                    let node = (component.name.clone(), equation.target.clone());
                    index.insert(node.clone(), nodes.len());
                    equations.insert(node.clone(), equation);
                    nodes.push(node);
                }
            }
        }
        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Connections indexed by destination in port, once: a resolved
        // flow network has as many connections as equations, and walking
        // the whole list per aggregation would make validation quadratic.
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
        // The resolution every read of an in port goes through, shared
        // by this classification and `collect_reads`.
        let port_sources = |port: &PortRef, channel: Option<&str>| {
            Self::resolve_port_sources(scopes, &feeds, port, channel)
        };

        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
        for component in &self.components {
            for equation in &component.equations {
                if equation.kind != EquationKind::Explicit {
                    continue;
                }
                let key = (component.name.clone(), equation.target.clone());
                let Some(&node) = index.get(&key) else {
                    continue;
                };
                let mut reads = Vec::new();
                Self::collect_reads(scopes, &feeds, &equation.expr, &mut reads);
                for read in reads {
                    if let Some(&dependency) = index.get(&read) {
                        if !edges[node].contains(&dependency) {
                            edges[node].push(dependency);
                        }
                    }
                }
            }
        }

        // Declaration order in, declaration order out: Tarjan emits the
        // components in a fixed order for a fixed declaration order, and
        // the first refused one is always the same.
        let mut blocks = Vec::new();
        for members in strongly_connected_components(&edges) {
            let self_loop = members.len() == 1 && edges[members[0]].contains(&members[0]);
            if members.len() == 1 && !self_loop {
                continue;
            }
            // Members in declaration order: the block's unknown order.
            let mut members = members;
            members.sort_unstable();
            let unknowns: Vec<AttrRef> = members
                .iter()
                .map(|&node| AttrRef {
                    component: nodes[node].0.clone(),
                    attribute: nodes[node].1.clone(),
                })
                .collect();
            let selected: HashSet<AttrRef> = unknowns.iter().cloned().collect();

            // Affinity of every member, coefficients merged per unknown:
            // one entry per (row, unknown), which is the sparsity
            // pattern and, past this point, the compiled block.
            let mut rows: Vec<BlockRow> = Vec::with_capacity(members.len());
            for &node in &members {
                let equation = equations[&nodes[node]];
                let form = equation
                    .expr
                    .affine_in_resolved(&selected, &port_sources)
                    .map_err(|term| ModelError::AlgebraicNonlinear {
                        component: nodes[node].0.clone(),
                        attribute: nodes[node].1.clone(),
                        term: term.expression.render(),
                    })?;
                let mut merged: Vec<(AttrRef, Expr)> = Vec::with_capacity(form.terms.len());
                for (unknown, coefficient) in form.terms {
                    if let Some(seen) = merged.iter_mut().find(|(seen, _)| *seen == unknown) {
                        let previous = std::mem::replace(
                            &mut seen.1,
                            Expr::Const {
                                value: Value::Float(0.0),
                            },
                        );
                        seen.1 = Expr::Add {
                            args: vec![previous, coefficient],
                        };
                    } else {
                        merged.push((unknown, coefficient));
                    }
                }
                rows.push(BlockRow {
                    constant: form.constant,
                    terms: merged,
                });
            }

            // Structural singularity: no perfect matching on the
            // pattern, where any coefficient that is not the literal 0
            // counts as non-zero. A column is an unknown's position in
            // the block, not a term's position in its row.
            let columns: HashMap<&AttrRef, usize> = unknowns
                .iter()
                .enumerate()
                .map(|(column, unknown)| (unknown, column))
                .collect();
            // The solved matrix is I-A. Keep the identity diagonal unless
            // a literal coefficient 1 cancels it. State-dependent or
            // computed coefficients stay conservative; the LU rank test
            // handles their cancellations on the actual matrix.
            let pattern: Vec<Vec<usize>> = rows
                .iter()
                .enumerate()
                .map(|(row_index, row)| {
                    let diagonal_cancels = row.terms.iter().any(|(unknown, coefficient)| {
                        columns[unknown] == row_index
                            && matches!(
                                coefficient,
                                Expr::Const {
                                    value: Value::Float(1.0) | Value::Int(1)
                                }
                            )
                    });
                    let mut entries = if diagonal_cancels {
                        Vec::new()
                    } else {
                        vec![row_index]
                    };
                    entries.extend(
                        row.terms
                            .iter()
                            .filter(|(unknown, coefficient)| {
                                columns[unknown] != row_index && !is_literal_zero(coefficient)
                            })
                            .map(|(unknown, _)| columns[unknown]),
                    );
                    entries.sort_unstable();
                    entries.dedup();
                    entries
                })
                .collect();
            if !has_perfect_matching(&pattern, unknowns.len()) {
                return Err(ModelError::AlgebraicSingular {
                    variables: render_variables(&unknowns),
                    reason: SingularReason::Structural,
                });
            }
            blocks.push(AlgebraicBlock { unknowns, rows });
        }
        Ok(blocks)
    }

    /// The linear blocks of this model: every cycle of the explicit
    /// equations, classified and solved-in-place ready.
    ///
    /// A cycle **linear in its unknowns** comes back as one
    /// [`AlgebraicBlock`]; the engine evaluates its coefficients and
    /// right-hand sides and solves the system `(I − A)·x = c` wherever
    /// the sweep runs. A cycle that is not linear (a `min`, a
    /// threshold, a `median`, an `if` condition reading an unknown) is
    /// refused with the term that makes it so, and a cycle whose system
    /// is structurally singular is refused with the variables it
    /// involves ([`ModelError::AlgebraicSingular`] at the
    /// [`SingularReason::Structural`] level; the rank-deficient level
    /// needs a factorisation and is raised by the compiler).
    ///
    /// An ODE target and an allocated quantity are deliberately absent
    /// from the graph: the integrator and the distribution operator
    /// carry them, and a cycle they sit on is well posed.
    ///
    /// # Errors
    /// [`ModelError::AlgebraicNonlinear`] and
    /// [`ModelError::AlgebraicSingular`], first refusal in a
    /// deterministic order.
    pub fn algebraic_blocks(&self) -> Result<Vec<AlgebraicBlock>, ModelError> {
        let scopes = self.check_components()?;
        self.classify_algebraic(&scopes)
    }

    /// Refuse a **priority split combined with surplus return** (see
    /// [`ModelError::AllocationPrioritySurplusReturn`]).
    ///
    /// *Surplus return* is a structural property, not a declared one: a
    /// consumer returns surplus when the demand it publishes on an edge
    /// depends, through the instantaneous reads of the explicit sweep, on
    /// a quantity that same operator allocated. That is precisely the
    /// cycle [`Model::algebraic_blocks`] deliberately allows, which
    /// is why it has to be recognised here instead.
    ///
    /// The search runs over the same read relation the loop detection
    /// uses, extended with one edge family: an allocated quantity reads
    /// the available quantity and every demand of its operator. It starts
    /// at each demand of a priority operator and looks for any allocated
    /// quantity of that same operator. A demand reaching *another*
    /// operator's output is ordinary network coupling and is left alone.
    fn check_priority_surplus_return(
        &self,
        scopes: &HashMap<&str, Scope<'_>>,
    ) -> Result<(), ModelError> {
        let priority_operators: Vec<(&Component, &Allocation)> = self
            .components
            .iter()
            .flat_map(|component| {
                component
                    .allocations
                    .iter()
                    .map(move |allocation| (component, allocation))
            })
            .filter(|(_, allocation)| {
                matches!(allocation.policy, AllocationPolicy::Priority { .. })
            })
            .collect();
        if priority_operators.is_empty() {
            return Ok(());
        }

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

        // The read relation: which attributes each computed attribute
        // reads at the instant it is evaluated.
        let mut reads: HashMap<EquationNode, Vec<EquationNode>> = HashMap::new();
        for component in &self.components {
            for equation in &component.equations {
                if equation.kind != EquationKind::Explicit {
                    continue;
                }
                let mut into = Vec::new();
                Self::collect_reads(scopes, &feeds, &equation.expr, &mut into);
                reads.insert((component.name.clone(), equation.target.clone()), into);
            }
            for allocation in &component.allocations {
                let mut into = Vec::new();
                Self::collect_reads(scopes, &feeds, &allocation.available, &mut into);
                for connection in self.allocation_edges(component, allocation) {
                    into.push((
                        component.name.clone(),
                        channel_attribute_name(connection, &allocation.demand),
                    ));
                }
                for connection in self.allocation_edges(component, allocation) {
                    let allocated = (
                        component.name.clone(),
                        channel_attribute_name(connection, &allocation.allocated),
                    );
                    reads.insert(allocated, into.clone());
                }
            }
        }

        for (component, allocation) in priority_operators {
            let edges = self.allocation_edges(component, allocation);
            let allocated: HashSet<EquationNode> = edges
                .iter()
                .map(|connection| {
                    (
                        component.name.clone(),
                        channel_attribute_name(connection, &allocation.allocated),
                    )
                })
                .collect();
            for connection in &edges {
                let start = (
                    component.name.clone(),
                    channel_attribute_name(connection, &allocation.demand),
                );
                if let Some(reached) = Self::reaches(&reads, &start, &allocated) {
                    return Err(ModelError::AllocationPrioritySurplusReturn {
                        component: component.name.clone(),
                        allocation: allocation.name.clone(),
                        demand: start.1,
                        allocated: reached.1,
                    });
                }
            }
        }
        Ok(())
    }

    /// The connections one distribution operator's out port carries, in
    /// declaration order (the order every per-edge vector shares).
    fn allocation_edges(&self, component: &Component, allocation: &Allocation) -> Vec<&Connection> {
        self.connections
            .iter()
            .filter(|connection| {
                connection.from.component == component.name
                    && connection.from.port == allocation.port
            })
            .collect()
    }

    /// A member of `wanted` reachable from `start` through the read
    /// relation, or `None`.
    ///
    /// A depth-first walk whose candidates are sorted before they are
    /// pushed, so the attribute that ends up in the diagnostic is a
    /// property of the model rather than of a hash order.
    fn reaches(
        reads: &HashMap<EquationNode, Vec<EquationNode>>,
        start: &EquationNode,
        wanted: &HashSet<EquationNode>,
    ) -> Option<EquationNode> {
        let mut seen: HashSet<EquationNode> = HashSet::new();
        let mut frontier = vec![start.clone()];
        while let Some(node) = frontier.pop() {
            if !seen.insert(node.clone()) {
                continue;
            }
            let Some(next) = reads.get(&node) else {
                continue;
            };
            let mut sorted: Vec<EquationNode> = next.clone();
            sorted.sort();
            sorted.dedup();
            for candidate in sorted.into_iter().rev() {
                if wanted.contains(&candidate) {
                    return Some(candidate);
                }
                frontier.push(candidate);
            }
        }
        None
    }

    /// Attributes an expression reads at the instant it is evaluated,
    /// with in-port aggregations resolved through the connections that
    /// feed them (a named channel reads the per-connection attributes
    /// materialised for it).
    ///
    /// `count` reads the *topology* and never a value, so it creates no
    /// data dependency: counting the connections of an in port cannot
    /// close an algebraic loop.
    fn collect_reads(
        scopes: &HashMap<&str, Scope<'_>>,
        feeds: &HashMap<(&str, &str), Vec<&Connection>>,
        expr: &Expr,
        into: &mut Vec<EquationNode>,
    ) {
        match expr {
            Expr::Const { .. } | Expr::StateActive { .. } | Expr::Time => {}
            Expr::Attr { attr } => into.push((attr.component.clone(), attr.attribute.clone())),
            Expr::PortAgg { port, agg, channel } => {
                if *agg == AggOp::Count {
                    return;
                }
                for AttrRef {
                    component,
                    attribute,
                } in Self::resolve_port_sources(scopes, feeds, port, channel.as_deref())
                {
                    into.push((component, attribute));
                }
            }
            Expr::Cmp { lhs, rhs, .. } | Expr::Sub { lhs, rhs } | Expr::Div { lhs, rhs } => {
                Self::collect_reads(scopes, feeds, lhs, into);
                Self::collect_reads(scopes, feeds, rhs, into);
            }
            Expr::Bool { args, .. }
            | Expr::Add { args }
            | Expr::Mul { args }
            | Expr::Min { args }
            | Expr::Max { args } => {
                for arg in args {
                    Self::collect_reads(scopes, feeds, arg, into);
                }
            }
            Expr::If {
                cond,
                then,
                otherwise,
            } => {
                Self::collect_reads(scopes, feeds, cond, into);
                Self::collect_reads(scopes, feeds, then, into);
                Self::collect_reads(scopes, feeds, otherwise, into);
            }
            Expr::Sin { arg }
            | Expr::Exp { arg }
            | Expr::PiecewiseLinear { arg, .. }
            | Expr::SmoothNoise { arg, .. } => {
                Self::collect_reads(scopes, feeds, arg, into);
            }
        }
    }

    /// The attributes one in-port aggregation reads: one per incoming
    /// connection, in connection declaration order : the exported
    /// attribute of the producer, or the per-connection attribute
    /// materialised for the named channel.
    fn resolve_port_sources(
        scopes: &HashMap<&str, Scope<'_>>,
        feeds: &HashMap<(&str, &str), Vec<&Connection>>,
        port: &PortRef,
        channel: Option<&str>,
    ) -> Vec<AttrRef> {
        let key = (port.component.as_str(), port.port.as_str());
        feeds
            .get(&key)
            .into_iter()
            .flatten()
            .filter_map(|connection| match channel {
                Some(channel) => Some(AttrRef {
                    component: connection.from.component.clone(),
                    attribute: channel_attribute_name(connection, channel),
                }),
                None => scopes
                    .get(connection.from.component.as_str())
                    .and_then(|scope| scope.ports.get(connection.from.port.as_str()))
                    .and_then(|source| source.attr.as_ref())
                    .map(|attribute| AttrRef {
                        component: connection.from.component.clone(),
                        attribute: attribute.clone(),
                    }),
            })
            .collect()
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
