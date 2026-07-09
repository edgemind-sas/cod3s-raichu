//! # raichu-model — the native RAICHU formalism
//!
//! Data model of the Distributed Stochastic Hybrid Automata (DSHA)
//! formalism realising Piecewise-Deterministic Markov Processes (PDMP),
//! as formalised in Desgeorges et al. 2021 (RESS).
//!
//! Design decisions:
//!
//! - **In/out ports** are the fundamental connection notion; *interfaces*
//!   group ports for batch connection.
//! - The model layer is pure data (serde), side-effect-free, validated at
//!   build time with **typed errors — never a crash on bad input**.
//! - Behaviour (guards, sensitive-function effects) is expressed as
//!   serializable expression trees from `raichu-expr`. Sensitivity sets
//!   (which attribute change re-triggers which function) are *derived*
//!   from the expressions, not declared by hand — one modeller error
//!   class removed.
//!
//! Validation ([`Model::validate`]) is the single gate: a model that
//! passes is structurally sound (all references resolve, distributions are
//! well-formed, initial states exist). The engine only consumes validated
//! models.

use raichu_expr::{Assignment, AttrRef, Expr, PortRef, Value};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// The type of an attribute (M0 kinds; `String` discrete state is reserved
/// and will extend this enum without breaking serialized models).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttrKind {
    /// Boolean state.
    Bool,
    /// 64-bit signed integer state.
    Int,
    /// 64-bit floating-point state.
    Float,
}

impl AttrKind {
    /// Whether `value` is an instance of this kind.
    #[must_use]
    pub fn matches(self, value: &Value) -> bool {
        matches!(
            (self, value),
            (AttrKind::Bool, Value::Bool(_))
                | (AttrKind::Int, Value::Int(_))
                | (AttrKind::Float, Value::Float(_))
        )
    }
}

/// An intrinsic, typed state attribute of a component.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attribute {
    /// Attribute name, unique inside its component.
    pub name: String,
    /// Type of the attribute.
    pub kind: AttrKind,
    /// Initial value (must match `kind`; checked at validation).
    pub init: Value,
}

/// Direction of a port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortDir {
    /// The port receives values from connected out-ports.
    In,
    /// The port exposes a local attribute to connected in-ports.
    Out,
}

/// A connection endpoint on a component boundary — the fundamental
/// interconnection notion of RAICHU (the model representation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Port {
    /// Port name, unique inside its component.
    pub name: String,
    /// Direction.
    pub dir: PortDir,
    /// For an **out** port: the local attribute it exposes (required).
    /// For an **in** port: must be absent — in-port values are read
    /// through aggregation expressions (`Expr::PortAgg`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attr: Option<String>,
}

/// A named group of ports, used to connect components in batch
/// (grouping only — the ports stay the fundamental notion).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interface {
    /// Interface name, unique inside its component.
    pub name: String,
    /// Names of the grouped ports (must exist on the component).
    pub ports: Vec<String>,
}

/// A directed connection from an out-port to an in-port.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Connection {
    /// Source (must be an out-port).
    pub from: PortRef,
    /// Destination (must be an in-port).
    pub to: PortRef,
}

/// Occurrence distribution of a transition (M0 deterministic subset; `Exp` and the
/// wider distribution library arrive in milestone M2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "distrib", rename_all = "snake_case")]
pub enum Distrib {
    /// Deterministic delay: fires `time` after the source state is
    /// entered. Exactly one target state.
    Delay {
        /// Delay duration (simulation time units, ≥ 0).
        time: f64,
    },
    /// Instantaneous branching: fires immediately;
    /// the destination is drawn among `targets` with the given
    /// probabilities. `probs` holds N−1 values for N targets, the last
    /// probability being the complement `1 − Σ probs`.
    Inst {
        /// First N−1 branch probabilities, each in [0, 1], Σ ≤ 1.
        probs: Vec<f64>,
    },
    /// Watched transition (paper rule `schedule_boundary`, M1): fires exactly when
    /// the continuous trajectory makes its guard become true. The guard
    /// is the boundary predicate — it must be a single ordering
    /// comparison (`<`, `≤`, `>`, `≥`) between float expressions, so
    /// the engine can locate the crossing by root-finding on the signed
    /// margin. Exactly one target state.
    Watched,
    /// Exponential distribution (paper rules `schedule_stochastic` and `reschedule_modifiable`): a
    /// spontaneous PDMP jump with survival
    /// `P(T > t) = exp(−∫₀ᵗ λ(x(u)) du)`. Exactly one target state,
    /// and exactly one of the two rate forms:
    ///
    /// - `rate` — fixed positive λ; the firing date is sampled as
    ///   `t + Exp(λ)` at source-state entry (`schedule_stochastic`). Memoryless — a
    ///   guard turning true re-arms with a fresh draw.
    /// - `rate_expr` — **state-dependent** λ(x) as an expression. The
    ///   engine realises the survival integral with a cumulative
    ///   hazard: a threshold `E ~ Exp(1)` is drawn at source-state
    ///   entry and the transition fires when `∫ λ dt` reaches `E`.
    ///   When λ depends only on discretely-updated state the hazard is
    ///   piecewise-constant and the firing date is rescheduled at each
    ///   discrete change (`reschedule_modifiable`); when it depends on
    ///   ODE-integrated attributes (or time), the hazard is integrated
    ///   alongside the continuous state and the firing time is located
    ///   like a watched boundary crossing. λ must evaluate ≥ 0.
    Exp {
        /// Fixed occurrence rate λ > 0 (events per time unit).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rate: Option<f64>,
        /// State-dependent occurrence rate λ(x) ≥ 0 (events per time
        /// unit), evaluated on the current model state.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rate_expr: Option<Expr>,
    },
    /// Weibull distribution (M4): delay sampled with CDF `1 − e^{−(t/scale)^shape}`.
    Weibull {
        /// Shape parameter k > 0.
        shape: f64,
        /// Scale parameter λ > 0 (time units).
        scale: f64,
    },
    /// Log-normal distribution (M4): `ln(delay) ~ N(mu, sigma²)`.
    Lognormal {
        /// Mean of the underlying normal.
        mu: f64,
        /// Standard deviation of the underlying normal (> 0).
        sigma: f64,
    },
    /// Gamma distribution (M4): shape/scale parametrisation
    /// (mean = shape·scale).
    Gamma {
        /// Shape parameter k > 0.
        shape: f64,
        /// Scale parameter θ > 0 (time units).
        scale: f64,
    },
    /// Uniform distribution (M4): delay drawn uniformly in `[low, high)`.
    Uniform {
        /// Lower bound (≥ 0).
        low: f64,
        /// Upper bound (> low).
        high: f64,
    },
    /// Empirical / user-defined distribution (M4): inverse-CDF sampling from a
    /// table of `(time, cumulative probability)` points — any
    /// distribution supplied as a sampled CDF. `u < points[0].1` maps
    /// to `points[0].0` (probability mass at the first time); between
    /// points the CDF is linearly interpolated; the last cumulative
    /// probability must be 1.
    Empirical {
        /// Non-decreasing `(time, cumulative probability)` table.
        points: Vec<(f64, f64)>,
    },
}

/// The kind of a continuous-evolution equation (CEvol of the paper).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EquationKind {
    /// Algebraic assignment `V = expr`, recomputed whenever its inputs
    /// change during continuous evolution (solved in declaration order).
    Explicit,
    /// First-order ODE `dV/dt = expr`, integrated by `raichu-numeric`.
    Ode,
}

/// A continuous-evolution equation attached to a component. Targets a
/// *local* `Float` attribute; declaration order across the component is
/// the solving `Order` of the formalism.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Equation {
    /// Local attribute receiving the value (must be `Float`).
    pub target: String,
    /// Explicit assignment or ODE.
    pub kind: EquationKind,
    /// Right-hand side.
    pub expr: Expr,
}

/// What happens to a pending occurrence countdown when the transition
/// guard turns false (naming set by Roland, 2026-07-04). Only
/// meaningful for duration distributions (delay, exp, …) with a guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterruptionPolicy {
    /// The occurrence duration is cancelled; a fresh duration is drawn
    /// as soon as the guard holds again (interruptible transition).
    /// RAICHU's default.
    #[default]
    Reset,
    /// The countdown is paused and resumes where it stopped when the
    /// guard holds again — a RAICHU extension, e.g. suspended repair
    /// work.
    Resume,
    /// The countdown never stops: the transition fires at the drawn
    /// date even if the guard dropped meanwhile
    /// (pinned by the `interrupt_01` cross-validation).
    Continue,
}

/// A transition of an automaton.
///
/// Maps to the paper's deterministic transitions ⟨q_src, guard, delay,
/// dest-distribution⟩; stochastic (`schedule_stochastic`) and watched (`schedule_boundary`)
/// transitions arrive in M2/M1 respectively.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transition {
    /// Transition name, unique inside its automaton.
    pub name: String,
    /// Source state (must exist in the automaton).
    pub source: String,
    /// Guard: the transition is armed while the guard holds.
    /// Absent means "always true".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard: Option<Expr>,
    /// Target states (exactly one for `Delay`; one per branch for
    /// `Inst`, with `probs.len() == targets.len() - 1`).
    pub targets: Vec<String>,
    /// What happens to a pending countdown when the guard turns false
    /// (paper rule `drop_disabled`; see [`InterruptionPolicy`]).
    #[serde(default)]
    pub on_interruption: InterruptionPolicy,
    /// Occurrence distribution.
    #[serde(flatten)]
    pub distrib: Distrib,
}

/// A finite automaton owned by a component. The global state space is
/// the Cartesian product of component automata but is **never
/// materialised** (the key to avoiding combinatorial explosion — keep it
/// that way).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Automaton {
    /// Automaton name, unique inside its component.
    pub name: String,
    /// State names (unique).
    pub states: Vec<String>,
    /// Initial state (must be one of `states`).
    pub init: String,
    /// Transitions.
    pub transitions: Vec<Transition>,
}

/// A sensitive function: declarative effects re-evaluated whenever one of
/// the attributes read by its expressions changes (the sensitivity set is
/// *derived* from the expressions — no manual `addSensitiveMethod`
/// bookkeeping). Effects run to a fixpoint during the discrete-evolution
/// phase (paper rule `propagate_effects`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensitiveFunction {
    /// Function name, unique inside its component.
    pub name: String,
    /// Ordered assignments applied when the function fires.
    pub effects: Vec<Assignment>,
}

/// What an indicator observes — a **typed, validated reference**
/// (rather than a stringly-typed `"comp.attr"` path).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "target", rename_all = "snake_case")]
pub enum IndicatorTarget {
    /// Observe an attribute's value over time.
    Attribute {
        /// The observed attribute.
        attr: AttrRef,
    },
    /// Observe whether an automaton is in a given state (0/1 over time).
    State {
        /// Component owning the automaton.
        component: String,
        /// The automaton.
        automaton: String,
        /// The observed state.
        state: String,
    },
}

/// A quantity recorded during simulation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Indicator {
    /// Indicator name (unique in the model; any length is accepted).
    pub name: String,
    /// Observed target.
    #[serde(flatten)]
    pub target: IndicatorTarget,
}

/// A component: typed state, ports/interfaces, automata and sensitive
/// functions (the 7-tuple of Desgeorges et al., with the paper's
/// connection receptacles expressed through the native port notion).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Component {
    /// Component name, unique in the model.
    pub name: String,
    /// Typed state attributes.
    #[serde(default)]
    pub attributes: Vec<Attribute>,
    /// Connection endpoints.
    #[serde(default)]
    pub ports: Vec<Port>,
    /// Port groupings for batch connection.
    #[serde(default)]
    pub interfaces: Vec<Interface>,
    /// Component-level automata.
    #[serde(default)]
    pub automata: Vec<Automaton>,
    /// Declarative sensitive functions.
    #[serde(default)]
    pub sensitive_functions: Vec<SensitiveFunction>,
    /// Continuous evolution: explicit equations and ODEs (M1;
    /// declaration order = solving order).
    #[serde(default)]
    pub equations: Vec<Equation>,
}

/// A complete model: a graph ⟨Cpt, cnx⟩ of components plus observed
/// indicators.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    /// Model name (provenance metadata).
    pub name: String,
    /// Components.
    pub components: Vec<Component>,
    /// Out-port → in-port connections.
    #[serde(default)]
    pub connections: Vec<Connection>,
    /// Recorded indicators.
    #[serde(default)]
    pub indicators: Vec<Indicator>,
}

/// Typed model-validation errors. Every invalid model is reported with a
/// precise, contextual error — never a panic, never a crash.
#[derive(Debug, Error, PartialEq)]
pub enum ModelError {
    /// Two components share a name.
    #[error("duplicate component name `{name}`")]
    DuplicateComponent {
        /// The duplicated name.
        name: String,
    },
    /// Two items of the same kind share a name inside one component.
    #[error("duplicate {kind} `{name}` in component `{component}`")]
    DuplicateInComponent {
        /// Item kind (attribute, port, interface, automaton, …).
        kind: &'static str,
        /// The duplicated name.
        name: String,
        /// The owning component.
        component: String,
    },
    /// An initial value does not match its attribute's declared kind.
    #[error("initial value of `{component}.{attribute}` does not match kind {kind:?}")]
    InitKindMismatch {
        /// The owning component.
        component: String,
        /// The attribute.
        attribute: String,
        /// The declared kind.
        kind: AttrKind,
    },
    /// An out-port must expose a local attribute.
    #[error("out-port `{component}.{port}` must reference a local attribute")]
    OutPortWithoutVariable {
        /// The owning component.
        component: String,
        /// The port.
        port: String,
    },
    /// An in-port must not carry a backing attribute (values are read via
    /// aggregation expressions).
    #[error("in-port `{component}.{port}` must not reference an attribute")]
    InPortWithVariable {
        /// The owning component.
        component: String,
        /// The port.
        port: String,
    },
    /// A port references an attribute that does not exist.
    #[error("port `{component}.{port}` references unknown attribute `{attribute}`")]
    PortUnknownVariable {
        /// The owning component.
        component: String,
        /// The port.
        port: String,
        /// The missing attribute.
        attribute: String,
    },
    /// An interface lists a port that does not exist.
    #[error("interface `{interface}` of component `{component}` lists unknown port `{port}`")]
    InterfaceUnknownPort {
        /// The owning component.
        component: String,
        /// The interface.
        interface: String,
        /// The missing port.
        port: String,
    },
    /// A connection endpoint does not resolve.
    #[error("connection endpoint `{component}.{port}` does not exist ({side})")]
    ConnectionUnknownPort {
        /// The referenced component.
        component: String,
        /// The referenced port.
        port: String,
        /// Which side of the connection (`from` / `to`).
        side: &'static str,
    },
    /// A connection must go from an out-port to an in-port.
    #[error(
        "connection `{from_component}.{from_port}` → `{to_component}.{to_port}` \
         must link an out-port to an in-port"
    )]
    ConnectionDirectionMismatch {
        /// Source component.
        from_component: String,
        /// Source port.
        from_port: String,
        /// Destination component.
        to_component: String,
        /// Destination port.
        to_port: String,
    },
    /// An automaton's initial state is not among its states.
    #[error("automaton `{component}.{automaton}` has unknown initial state `{state}`")]
    UnknownInitState {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The missing state.
        state: String,
    },
    /// A transition endpoint state does not exist.
    #[error(
        "transition `{transition}` of `{component}.{automaton}` references \
         unknown state `{state}`"
    )]
    UnknownTransitionState {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
        /// The missing state.
        state: String,
    },
    /// A delay transition must have exactly one target.
    #[error(
        "delay transition `{transition}` of `{component}.{automaton}` must have \
         exactly one target (got {targets})"
    )]
    DelayTargetCount {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
        /// Number of targets found.
        targets: usize,
    },
    /// A delay must be non-negative and finite.
    #[error(
        "delay transition `{transition}` of `{component}.{automaton}` has invalid \
         time {time} (must be finite and ≥ 0)"
    )]
    InvalidDelay {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
        /// The offending value.
        time: f64,
    },
    /// Instantaneous branching: `probs.len()` must equal
    /// `targets.len() − 1` (last probability is the complement).
    #[error(
        "inst transition `{transition}` of `{component}.{automaton}` has \
         {targets} target(s) but {probs} probabilities (expected targets − 1)"
    )]
    InstArityMismatch {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
        /// Number of targets.
        targets: usize,
        /// Number of probabilities.
        probs: usize,
    },
    /// A branching probability is outside [0, 1] or the sum exceeds 1.
    #[error(
        "inst transition `{transition}` of `{component}.{automaton}` has invalid \
         probabilities (each must be in [0,1], sum ≤ 1; got sum = {sum})"
    )]
    InvalidInstProbs {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
        /// The (offending) probability sum.
        sum: f64,
    },
    /// `Not` takes exactly one argument.
    #[error("boolean `not` in {context} must have exactly one argument (got {args})")]
    NotArity {
        /// Where the expression appears.
        context: String,
        /// Number of arguments found.
        args: usize,
    },
    /// An expression references an attribute that does not exist.
    #[error("{context} references unknown attribute `{component}.{attribute}`")]
    ExprUnknownVariable {
        /// Where the expression appears.
        context: String,
        /// The referenced component.
        component: String,
        /// The missing attribute.
        attribute: String,
    },
    /// An expression aggregates over a port that does not exist or is
    /// not an in-port.
    #[error("{context} aggregates over unknown or non-in port `{component}.{port}`")]
    ExprBadPortAgg {
        /// Where the expression appears.
        context: String,
        /// The referenced component.
        component: String,
        /// The offending port.
        port: String,
    },
    /// An expression references an automaton state that does not exist.
    #[error("{context} references unknown state `{component}.{automaton}.{state}`")]
    ExprUnknownState {
        /// Where the expression appears.
        context: String,
        /// The referenced component.
        component: String,
        /// The referenced automaton.
        automaton: String,
        /// The missing state.
        state: String,
    },
    /// An interruption policy needs a guard to ever trigger.
    #[error(
        "transition `{transition}` of `{component}.{automaton}` declares \
         on_interruption = {policy} but has no guard (nothing can \
         interrupt it)"
    )]
    InterruptionPolicyWithoutGuard {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
        /// The declared (ineffective) policy.
        policy: &'static str,
    },
    /// An n-ary operator needs at least one operand.
    #[error("{op} in {context} needs at least one operand")]
    EmptyExprArgs {
        /// Where the expression appears.
        context: String,
        /// The offending operator.
        op: &'static str,
    },
    /// An equation targets an attribute that does not exist locally.
    #[error("equation in `{component}` targets unknown local attribute `{target}`")]
    EquationUnknownTarget {
        /// The owning component.
        component: String,
        /// The missing attribute.
        target: String,
    },
    /// Continuous equations may only target `Float` attributes.
    #[error("equation target `{component}.{target}` must be a float attribute")]
    EquationTargetNotFloat {
        /// The owning component.
        component: String,
        /// The offending attribute.
        target: String,
    },
    /// A attribute may carry at most one equation.
    #[error("attribute `{component}.{target}` has more than one equation")]
    DuplicateEquation {
        /// The owning component.
        component: String,
        /// The doubly-defined attribute.
        target: String,
    },
    /// A watched transition needs a guard (its boundary predicate).
    #[error(
        "watched transition `{transition}` of `{component}.{automaton}` has no \
         guard (the guard is the boundary predicate)"
    )]
    WatchedGuardMissing {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
    },
    /// A watched guard needs a locatable continuous boundary: at least
    /// one ordering comparison, possibly composed with and/or/not and
    /// discrete gates (margins combine as min/max/negation).
    #[error(
        "watched transition `{transition}` of `{component}.{automaton}`: the \
         guard must contain at least one ordering comparison (<, <=, >, >=), \
         possibly composed with and/or/not and discrete gates"
    )]
    WatchedGuardNotComparison {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
    },
    /// A stochastic-distribution parameter is out of its domain.
    #[error(
        "{distribution} transition `{transition}` of `{component}.{automaton}` has invalid \
         parameter {parameter} = {value}"
    )]
    InvalidLawParameter {
        /// The distribution kind.
        distribution: &'static str,
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
        /// The offending parameter name.
        parameter: &'static str,
        /// The offending value.
        value: f64,
    },
    /// An empirical table is malformed.
    #[error("empirical transition `{transition}` of `{component}.{automaton}`: {detail}")]
    InvalidEmpiricalTable {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
        /// What is wrong with the table.
        detail: String,
    },
    /// An exponential rate must be finite and strictly positive.
    #[error(
        "exp transition `{transition}` of `{component}.{automaton}` has invalid \
         rate {rate} (must be finite and > 0)"
    )]
    InvalidExpRate {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
        /// The offending rate.
        rate: f64,
    },
    /// An exp transition must carry exactly one of `rate` / `rate_expr`.
    #[error(
        "exp transition `{transition}` of `{component}.{automaton}` must set \
         exactly one of `rate` (fixed) or `rate_expr` (state-dependent)"
    )]
    ExpRateSpec {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
    },
    /// A stochastic-distribution transition must have exactly one target.
    #[error(
        "{distribution} transition `{transition}` of `{component}.{automaton}` must have \
         exactly one target (got {targets})"
    )]
    StochasticTargetCount {
        /// The distribution kind.
        distribution: &'static str,
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
        /// Number of targets found.
        targets: usize,
    },
    /// A watched transition must have exactly one target.
    #[error(
        "watched transition `{transition}` of `{component}.{automaton}` must \
         have exactly one target (got {targets})"
    )]
    WatchedTargetCount {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
        /// Number of targets found.
        targets: usize,
    },
    /// Two indicators share a name.
    #[error("duplicate indicator name `{name}`")]
    DuplicateIndicator {
        /// The duplicated name.
        name: String,
    },
    /// An indicator references something that does not exist.
    #[error("indicator `{indicator}`: {detail}")]
    IndicatorUnresolved {
        /// The indicator.
        indicator: String,
        /// What failed to resolve.
        detail: String,
    },
}

/// Component-scope lookup tables used during validation.
struct Scope<'m> {
    attributes: HashMap<&'m str, &'m Attribute>,
    ports: HashMap<&'m str, &'m Port>,
    /// automaton name → set of its state names.
    automata: HashMap<&'m str, HashSet<&'m str>>,
}

impl Model {
    /// Load a model from its JSON representation.
    ///
    /// This only checks JSON well-formedness against the schema; call
    /// [`Model::validate`] afterwards for structural soundness.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize the model to pretty JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Validate structural soundness: unique names, resolvable
    /// references, well-formed distributions, matching kinds. Returns the *first*
    /// error encountered in deterministic (declaration) order.
    ///
    /// The engine (`raichu-core`) only accepts validated models; this is
    /// the fail-fast gate (typed errors at build
    /// time, never mid-simulation surprises).
    pub fn validate(&self) -> Result<(), ModelError> {
        let scopes = self.check_components()?;
        self.check_connections(&scopes)?;
        self.check_expressions(&scopes)?;
        self.check_indicators(&scopes)?;
        Ok(())
    }

    fn check_components(&self) -> Result<HashMap<&str, Scope<'_>>, ModelError> {
        let mut scopes: HashMap<&str, Scope<'_>> = HashMap::new();
        for component in &self.components {
            if scopes.contains_key(component.name.as_str()) {
                return Err(ModelError::DuplicateComponent {
                    name: component.name.clone(),
                });
            }
            let scope = Self::check_component(component)?;
            scopes.insert(component.name.as_str(), scope);
        }
        Ok(scopes)
    }

    fn check_component(component: &Component) -> Result<Scope<'_>, ModelError> {
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
        for equation in &component.equations {
            let Some(attribute) = attributes.get(equation.target.as_str()) else {
                return Err(ModelError::EquationUnknownTarget {
                    component: component.name.clone(),
                    target: equation.target.clone(),
                });
            };
            if attribute.kind != AttrKind::Float {
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
        }

        Ok(Scope {
            attributes,
            ports,
            automata,
        })
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
                            if !rate.is_finite() || *rate <= 0.0 {
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
    /// A watched guard must contain at least one ordering comparison —
    /// the continuous boundary — anywhere under `and`/`or`/`not`
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
        Ok(())
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
        for component in &self.components {
            for automaton in &component.automata {
                for transition in &automaton.transitions {
                    if let Some(guard) = &transition.guard {
                        let context = format!(
                            "guard of transition `{}` in `{}.{}`",
                            transition.name, component.name, automaton.name
                        );
                        Self::check_expr(scopes, guard, &context)?;
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
                    Self::check_expr(scopes, &assignment.value, &context)?;
                }
            }
            for equation in &component.equations {
                let context = format!("equation of `{}.{}`", component.name, equation.target);
                Self::check_expr(scopes, &equation.expr, &context)?;
            }
        }
        Ok(())
    }

    fn check_expr(
        scopes: &HashMap<&str, Scope<'_>>,
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
            Expr::PortAgg { port, .. } => {
                let resolved = scopes
                    .get(port.component.as_str())
                    .and_then(|scope| scope.ports.get(port.port.as_str()));
                match resolved {
                    Some(p) if p.dir == PortDir::In => Ok(()),
                    _ => Err(ModelError::ExprBadPortAgg {
                        context: context.to_owned(),
                        component: port.component.clone(),
                        port: port.port.clone(),
                    }),
                }
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
                Self::check_expr(scopes, lhs, context)?;
                Self::check_expr(scopes, rhs, context)
            }
            Expr::Bool { args, .. } => {
                for arg in args {
                    Self::check_expr(scopes, arg, context)?;
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
                    Self::check_expr(scopes, arg, context)?;
                }
                Ok(())
            }
            Expr::If {
                cond,
                then,
                otherwise,
            } => {
                Self::check_expr(scopes, cond, context)?;
                Self::check_expr(scopes, then, context)?;
                Self::check_expr(scopes, otherwise, context)
            }
            Expr::Sin { arg } | Expr::Exp { arg } => Self::check_expr(scopes, arg, context),
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
            .is_some_and(|scope| scope.attributes.contains_key(var_ref.attribute.as_str()));
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
            }
        }
        // scopes is only used through the helpers above; keep the
        // signature symmetric with the other passes.
        let _ = scopes;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
