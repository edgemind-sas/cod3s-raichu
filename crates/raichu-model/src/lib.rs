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

use raichu_expr::{AggOp, Assignment, AttrRef, CmpOp, Expr, PortRef, Value};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};
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

/// A **per-connection channel** declared on an out port.
///
/// An out port exposes one attribute, so every in port connected to it
/// reads the same number. A channel adds the opposite affordance: one
/// quantity *per connection*, so a producer can hand a different share to
/// each consumer over the same port (the conservative-flow shape).
///
/// The channel is declared once, here, on the port; the **compiler
/// materialises** one float attribute per (connection, channel) on the
/// producing component, named by [`channel_attribute_name`]. Declaring
/// those attributes by hand would name the topology twice and let a model
/// declare a channel with no matching connection.
///
/// Materialised attributes are ordinary float attributes: equations and
/// sensitive functions write them, the causal journal records them, the
/// snapshot carries them, and indicators observe them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Channel {
    /// Channel name, unique inside its port.
    pub name: String,
    /// Initial value of every attribute materialised for this channel:
    /// the value a consumer reads before anything writes the channel.
    #[serde(default)]
    pub init: f64,
}

/// A connection endpoint on a component boundary: the fundamental
/// interconnection notion of RAICHU (the model representation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Port {
    /// Port name, unique inside its component.
    pub name: String,
    /// Direction.
    pub dir: PortDir,
    /// For an **out** port: the local attribute it exposes (required).
    /// For an **in** port: must be absent, in-port values are read
    /// through aggregation expressions (`Expr::PortAgg`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attr: Option<String>,
    /// **Out ports only**: per-connection channels (see [`Channel`]).
    /// An in port reads channels through `Expr::PortAgg`, it never
    /// declares them; validation refuses a channel list on an in port.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels: Vec<Channel>,
}

/// A named group of ports, validated at build time (each port must
/// exist), that an [`InterfaceConnection`] joins to another at once. The
/// ports stay the fundamental notion: the engine runs on the port
/// connections an interface connection stands for.
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
    /// Optional connection name. It only names the *edge*: the naming of
    /// the per-connection attributes materialised for the source port's
    /// [`Channel`]s (see [`channel_attribute_name`]). Absent, the edge is
    /// named after its destination, which is unambiguous unless two
    /// connections join the same pair of ports: name those explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Source (must be an out-port).
    pub from: PortRef,
    /// Destination (must be an in-port).
    pub to: PortRef,
}

/// Reference to an interface by hierarchical name (authoring form).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InterfaceRef {
    /// Name of the component owning the interface.
    pub component: String,
    /// Name of the interface inside the component.
    pub interface: String,
}

/// A connection between two [`Interface`]s: the batch form of
/// [`Connection`], the counterpart of connecting two message boxes.
///
/// It stands for one port-to-port connection per port of the two
/// interfaces, **paired by port name**: an out port of either side is
/// connected to the in port of the same name on the other side, so one
/// interface connection can carry flows in both directions. The pairing
/// must be exact: every port of each interface finds its partner, of the
/// opposite direction ([`ModelError::InterfacePortUnmatched`],
/// [`ModelError::InterfacePortDirection`]).
///
/// The model never runs on interface connections themselves:
/// [`Model::resolved_connections`] expands them, after the explicit
/// [`Model::connections`] and in declaration order, and every consumer
/// (validation, compilation, diagnostics) reads that one list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterfaceConnection {
    /// Optional name, given to every port connection it expands to (see
    /// [`Connection::name`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// One side.
    pub from: InterfaceRef,
    /// The other side.
    pub to: InterfaceRef,
}

/// Name of the attribute materialised on the producing component for one
/// (connection, channel) pair: `<out port>__<channel>__<edge>`, the edge
/// being the connection's name when it carries one and
/// `<destination component>__<destination port>` otherwise.
///
/// This is the **single** naming function: the compiler, the causal
/// journal, indicators and the collision refusal all read the same name,
/// so the name a model author sees in a diagnostic is the name the
/// journal prints. It is derived from the topology, not from a
/// declaration index, so inserting a connection does not rename the
/// attributes of the others.
#[must_use]
pub fn channel_attribute_name(connection: &Connection, channel: &str) -> String {
    let edge = match &connection.name {
        Some(name) => name.clone(),
        None => format!("{}__{}", connection.to.component, connection.to.port),
    };
    format!("{}__{channel}__{edge}", connection.from.port)
}

/// One per-connection channel attribute the compiler materialises: the
/// shared description of the model's derived state, produced by
/// [`Model::channel_attributes`] and consumed by both validation and the
/// compiler so the two never disagree on what exists.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelAttribute {
    /// Index of the connection in [`Model::connections`].
    pub connection: usize,
    /// Component owning the materialised attribute (the producer).
    pub component: String,
    /// Out port declaring the channel.
    pub port: String,
    /// Channel name.
    pub channel: String,
    /// Name of the materialised attribute inside `component`.
    pub attribute: String,
    /// Initial value (the channel's declared default).
    pub init: f64,
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
    /// is the boundary predicate: it must contain at least one ordering
    /// comparison (`<`, `≤`, `>`, `≥`) between float expressions,
    /// possibly composed with and/or/not and discrete gates, so the
    /// engine can locate the crossing by root-finding on the signed
    /// margin. Exactly one target state.
    Watched,
    /// Exponential distribution (paper rules `schedule_stochastic` and `reschedule_modifiable`): a
    /// spontaneous PDMP jump with survival
    /// `P(T > t) = exp(−∫₀ᵗ λ(x(u)) du)`. Exactly one target state,
    /// and exactly one of the two rate forms:
    ///
    /// - `rate`: fixed positive λ; the firing date is sampled as
    ///   `t + Exp(λ)` at source-state entry (`schedule_stochastic`). Memoryless: a
    ///   guard turning true re-arms with a fresh draw.
    /// - `rate_expr`: **state-dependent** λ(x) as an expression. The
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
    /// table of `(time, cumulative probability)` points: any
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

/// A binding between one model attribute and one variable in an imported FMU.
/// The FMU description resolves the variable name and type before execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FmuBinding {
    /// Attribute read for an input, or written by an output.
    pub attribute: AttrRef,
    /// Variable name in the FMU description.
    pub variable: String,
}

/// An initial value assigned to an FMU parameter before simulation starts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FmuParameter {
    /// Parameter name in the FMU description.
    pub variable: String,
    /// Typed initial value, checked against the FMU description at preparation.
    pub value: Value,
}

/// One imported FMI co-simulation unit and its model-facing bindings.
/// Loading its native library requires a separate run-side permission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FmuUnit {
    /// Name unique among the model's units.
    pub name: String,
    /// FMU archive path, resolved by the loader against its base directory.
    pub path: String,
    /// Communication interval in model time units; finite and positive.
    pub step: f64,
    /// Values sampled from model attributes and written to FMU inputs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<FmuBinding>,
    /// FMU outputs written to model attributes at communication points.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<FmuBinding>,
    /// Start values for FMU parameters.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parameters: Vec<FmuParameter>,
}

/// One **consumer-keyed** parameter of an allocation policy: the value
/// that applies to the connection ending at `to`.
///
/// Keyed by the destination port rather than by a position in a list, so
/// inserting or reordering a connection cannot silently re-attach a share
/// to a different consumer. Parallel edges to the same destination share
/// one entry, which is coherent: each of them then carries that value,
/// and the share sum is checked over the *connections*.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsumerParam {
    /// The in port at the far end of the connection this value applies to.
    pub to: PortRef,
    /// The value: a share of the available quantity under
    /// [`AllocationPolicy::Shares`] (in [0, 1], summing to 1 over the
    /// port's connections), a priority rank under
    /// [`AllocationPolicy::Priority`] (lowest served first).
    pub value: f64,
}

/// How an [`Allocation`] splits a shortage among the consumers of one out
/// port. Three policies, no default: how a scarce quantity is shared is a
/// modelling decision, never an engine convention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
pub enum AllocationPolicy {
    /// Proportional to demand: each consumer receives
    /// `available x demand / Σ demands`, capped at its own demand.
    /// Needs no parameter, so a consumer added to the port needs no edit.
    Proportional,
    /// Fixed shares: each consumer receives `available x share`, capped at
    /// its own demand, and the surplus of a consumer that asked for less
    /// is redistributed to the others. The shares must cover every
    /// connection of the port and sum to 1.
    Shares {
        /// One share per consumer of the port.
        shares: Vec<ConsumerParam>,
    },
    /// Strict priority: consumers are served in full, in ascending rank,
    /// until the available quantity runs out. Equal ranks break by
    /// **connection declaration index**, so the order is a property of
    /// the model file and never of the engine's sweep.
    Priority {
        /// One rank per consumer of the port.
        priorities: Vec<ConsumerParam>,
    },
}

/// The **conservative distribution operator**: it reads one available
/// quantity and one demand per outgoing connection of an out port, and
/// writes one allocated quantity per outgoing connection under a declared
/// policy.
///
/// # Why it is not a sensitive function
///
/// A [`SensitiveFunction`] effect writes **one** target. A split cannot be
/// written that way: the share handed to one consumer depends on what
/// every other consumer asked for, so the operator writes the whole vector
/// at once, as a single indivisible step.
///
/// # Why it sits on the explicit-equation path
///
/// The engine runs only the explicit pass inside its solver callbacks; a
/// trigger-driven entity never runs during integration. Quantities written
/// from a trigger would stay frozen across an integration segment, every
/// watched margin reading them would freeze with them, and a boundary
/// crossing would degrade silently from *located* to *polled*. The
/// operator is therefore an ordinary step of the explicit sweep, named in
/// [`Model::evaluation_order`] like an explicit equation, and evaluated at
/// every evaluation point.
///
/// Both channels live on the producing port: `demand` carries what each
/// consumer asks for (written upstream of this step), `allocated` carries
/// what it gets. Nothing else may write the `allocated` quantities: a
/// second writer is refused at build time
/// ([`ModelError::AllocationTargetWritten`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Allocation {
    /// Operator name, unique inside its component, and the name that
    /// designates this step in [`Model::evaluation_order`].
    pub name: String,
    /// The **out** port whose connections receive the quantity.
    pub port: String,
    /// The quantity available for distribution. Evaluated on the current
    /// state; a negative value distributes nothing.
    pub available: Expr,
    /// Channel of `port` carrying each consumer's demand (read).
    pub demand: String,
    /// Channel of `port` receiving each consumer's share (written).
    pub allocated: String,
    /// The split policy.
    #[serde(flatten)]
    pub policy: AllocationPolicy,
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
    /// guard holds again: a RAICHU extension, e.g. suspended repair
    /// work.
    Resume,
    /// The countdown never stops: the transition fires at the drawn
    /// date even if the guard dropped meanwhile
    /// (pinned by the `interrupt_01` cross-validation).
    Continue,
}

/// The declared **reliability role** of a transition: whether firing it
/// fails something or repairs something.
///
/// A driver that bounds the number of failures along a sequence (the
/// failure-count cut-off of the sequence-tree exploration) reads this
/// declaration; it never infers the role from state names, a convention the
/// engine could not check. The kind changes no simulated trajectory, so it
/// is a baseline construct and requires no feature.
///
/// For a transition with several targets (an instantaneous branching, such
/// as an on-demand failure draw towards `[failed, parked]`), the declared
/// role applies to its **first declared target** only: firing into any
/// other branch is not a failure (respectively a repair).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionKind {
    /// Firing into the first declared target is a failure occurrence.
    Failure,
    /// Firing into the first declared target is a repair (a return to
    /// service).
    Repair,
    /// The transition belongs to an **observer**: an automaton that watches
    /// the model (a feared event, a counted alarm) and writes nothing the
    /// rest of the model reads. Unlike the two roles above it changes the
    /// trajectory, in one place only: among transitions due at the same
    /// date, an observer's fires first, so a state the model reaches and
    /// leaves within one instant is still observed, as the reference engine
    /// observes it. Requires [`Feature::ObserverPriority`].
    Observation,
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
    /// **Sequence analysis**: when true, firing this transition records a
    /// `SeqEvent` (component, target state, time) into the per-trajectory
    /// sequence trace (zero cost unless sequence recording is enabled). The
    /// muscadet plugin sets it on ObjFM occ/rep and ObjEvent occ transitions.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub monitored: bool,
    /// **Sequence analysis**: cycle-pair group id: occ/rep (occ/not_occ)
    /// partner transitions of one failure mode share it, so the
    /// cycle-filtering step can drop transient failure→repair pairs that net
    /// out before the feared event. `None` = not part of a cycle pair.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cycle_group: Option<String>,
    /// **Declared reliability role** (see [`TransitionKind`]): `failure`,
    /// `repair`, or absent (the default, and the reading of every model
    /// written before the field existed). Applies to the first declared
    /// target only. The muscadet plugin sets it on the failure-mode edges
    /// it emits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<TransitionKind>,
    /// **Edge effects**: assignments evaluated ONCE, when the transition
    /// fires, after its state change, in declaration order. Unlike a
    /// sensitive function's effects (a level, re-evaluated whenever what it
    /// reads changes), nothing re-applies them and nothing restores the
    /// value: the attribute keeps what was written until something else
    /// writes it. That is a one-shot effect, and on an attribute nothing
    /// else writes, a variable that memorises (a latched detection, an
    /// alarm). An interrupted transition writes nothing.
    ///
    /// On the target of an ODE equation, an edge effect is a **reset
    /// map**, the jump of a piecewise-deterministic Markov process: the
    /// continuous state takes the written value at the firing instant and
    /// the integration restarts from it (the value must be a finite float).
    ///
    /// Non-baseline construct: a document carrying it must declare
    /// [`Feature::TransitionEffects`]. An effect on an attribute an explicit
    /// equation or a sensitive function writes is refused
    /// ([`ModelError::TransitionEffectOverwritten`]): the next evaluation
    /// would erase it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<Assignment>,
    /// Occurrence distribution.
    #[serde(flatten)]
    pub distrib: Distrib,
}

/// A finite automaton owned by a component. The global state space is
/// the Cartesian product of component automata but is **never
/// materialised** (the key to avoiding combinatorial explosion: keep it
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
/// *derived* from the expressions: no manual `addSensitiveMethod`
/// bookkeeping). Effects run to a fixpoint during the discrete-evolution
/// phase (paper rule `propagate_effects`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensitiveFunction {
    /// Function name, unique inside its component.
    pub name: String,
    /// Ordered assignments applied when the function fires.
    pub effects: Vec<Assignment>,
}

/// What an indicator observes: a **typed, validated reference**
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
    /// Observe the **truth of a threshold** on an attribute: `true` while
    /// `attr cmp value` holds, `false` otherwise.
    ///
    /// The difference with [`IndicatorTarget::Attribute`] is the whole
    /// point of the variant, and it is a difference of *quantity*, not of
    /// spelling: on a free-valued attribute the sojourn of
    /// [`IndicatorTarget::Attribute`] is the time-integral of the value
    /// (an area), while the sojourn here is the **time spent under the
    /// condition** (a duration, bounded by the horizon). A study that
    /// asks "how long was there hydrogen in the tank" asks for the
    /// second, and reading it off the first returns the tank's content
    /// integrated over time, which is plausible, larger than the horizon,
    /// and silently the wrong question's answer.
    ///
    /// `value` is a constant: this carries the threshold a study
    /// declares (cod3s's `operator` / `value_test` pair), not a general
    /// expression. Kinds are checked at build time, so a boolean
    /// attribute may only be tested for (in)equality and a numeric one
    /// may not be compared to a boolean.
    Predicate {
        /// The compared attribute.
        attr: AttrRef,
        /// The comparison operator.
        cmp: CmpOp,
        /// The constant the attribute is compared against.
        value: Value,
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
    /// Conservative distribution operators owned by this component (see
    /// [`Allocation`]). They are steps of the explicit sweep, placed after
    /// the component's explicit equations unless
    /// [`Model::evaluation_order`] says otherwise.
    #[serde(default)]
    pub allocations: Vec<Allocation>,
}

/// A **sequence-analysis target** (feared event / événement redouté): a
/// named automaton state whose activation ends and labels a trajectory's
/// recorded sequence (mirrors cod3s `endCause`). Empty `targets` = no early
/// stop and no `end_cause` (the trajectory runs to `t_max`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Target {
    /// Target name (the `end_cause` label, e.g. `doors_unsecured`).
    pub name: String,
    /// Component owning the automaton.
    pub component: String,
    /// The automaton.
    pub automaton: String,
    /// The state whose activation reaches the target.
    pub state: String,
}

/// Top-level key of the **format envelope** of a serialized model
/// document (see [`FormatHeader`]).
pub const ENVELOPE_KEY: &str = "raichu_model";

/// Revision of the envelope itself, not of the engine: it changes only
/// if the envelope's own shape changes, which is why it is a small
/// integer and not the crate version.
pub const FORMAT_REVISION: u32 = 1;

/// A named construct of the model format that an engine either
/// implements or does not.
///
/// The registry opens with this enum: everything the format could
/// express *before* it is the **baseline** and carries no name, so
/// every model written until now needs no feature at all. From here on,
/// a construct an older engine would silently ignore gets a variant,
/// and a model using it must say so in its [`FormatHeader`].
///
/// The list a document carries is never taken on trust: [`Model::from_json`]
/// derives the truth from the parsed content ([`Model::required_features`])
/// and refuses a document whose declaration does not cover it. A
/// hand-written model therefore cannot use a construct and stay silent
/// about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Feature {
    /// Model-level [`Model::evaluation_order`]: the explicit equations
    /// are swept in a declared order instead of the positional one. An
    /// engine that ignored the field would sweep a different order and
    /// return a different number without a word.
    EvaluationOrder,
    /// Component-level [`Component::allocations`]: the conservative
    /// distribution operator. An engine that ignored the field would
    /// leave every allocated quantity at its channel default, and report
    /// a network that delivers nothing without a word.
    Allocation,
    /// Model-level [`Model::unbounded_rate`]: the magnitude an authoring
    /// layer writes for "no ceiling", which no integrated rate may reach.
    /// An engine that ignored the field would integrate that magnitude as
    /// a rate and report a stock of -1e19 without a word.
    UnboundedRate,
    /// Transition-level [`Transition::effects`]: assignments made once, on
    /// the firing edge. An engine that ignored the field would never make
    /// the write, and a latched detection would read its initial value for
    /// the whole trajectory without a word.
    TransitionEffects,
    /// Model-level [`Model::fmu_units`]: imported native FMI co-simulation
    /// units. An older engine must refuse instead of ignoring their physics.
    Fmi,
    /// Model-level mixed-integer programs solved at discrete fixpoints.
    MixedIntegerProgram,
    /// Transition-level [`TransitionKind::Observation`]: observers fire first
    /// among transitions due at the same date. An engine that ignored it
    /// would fire them in positional order and miss every state the model
    /// reaches and leaves within one instant, without a word.
    ObserverPriority,
    /// Model-level [`Model::interface_connections`]. An engine that
    /// ignored the field would leave the ports they connect unfed, and
    /// run a model whose components never hear from each other without
    /// a word.
    InterfaceConnections,
}

impl Feature {
    /// Every feature this engine implements, in declaration order.
    pub const ALL: &'static [Feature] = &[
        Feature::EvaluationOrder,
        Feature::Allocation,
        Feature::UnboundedRate,
        Feature::TransitionEffects,
        Feature::Fmi,
        Feature::MixedIntegerProgram,
        Feature::ObserverPriority,
        Feature::InterfaceConnections,
    ];

    /// Serialized name of the feature.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Feature::EvaluationOrder => "evaluation_order",
            Feature::Allocation => "allocation",
            Feature::UnboundedRate => "unbounded_rate",
            Feature::TransitionEffects => "transition_effects",
            Feature::Fmi => "fmi",
            Feature::MixedIntegerProgram => "mixed_integer_program",
            Feature::ObserverPriority => "observer_priority",
            Feature::InterfaceConnections => "interface_connections",
        }
    }

    /// The feature this name designates, or `None` when this engine
    /// does not implement it (a newer construct, or a typo: both are
    /// refused, and refusing is the point).
    #[must_use]
    pub fn parse(name: &str) -> Option<Feature> {
        Feature::ALL.iter().copied().find(|f| f.name() == name)
    }

    /// Comma-separated list of the implemented feature names, for
    /// diagnostics.
    #[must_use]
    pub fn known() -> String {
        Feature::ALL
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Header of the **format envelope**: the mandatory, reader-visible
/// preamble of a serialized model document.
///
/// # Why a wrapper and not a field
///
/// The model schema accepts unknown fields (no `deny_unknown_fields`
/// anywhere) and carried no version. An optional `requires` field
/// alongside `name` and `components` would therefore be *invisible* to
/// an engine that predates it: that engine would parse the document,
/// ignore the field **and the new construct it announces**, and return a
/// different number instead of a refusal. Silence is exactly the
/// failure to avoid.
///
/// The envelope instead **displaces** the fields a legacy reader
/// requires: the model body moves under `model`, so a reader expecting
/// `name` and `components` at the top level fails on a missing field
/// rather than succeeding on a misread. The refusal is structural, so
/// it works even against engines released before the envelope existed.
/// Readers from this version on know the shape, so *their* refusal is
/// the precise, named [`LoadError::UnsupportedFeature`].
///
/// A serialized document looks like:
///
/// ```json
/// {
///   "raichu_model": {"format": 1, "requires": ["evaluation_order"]},
///   "model": {"name": "…", "components": []}
/// }
/// ```
///
/// The bare (envelope-less) body stays readable, so the whole existing
/// corpus loads unchanged; it declares no feature, so it may only use
/// baseline constructs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormatHeader {
    /// Envelope revision ([`FORMAT_REVISION`]).
    pub format: u32,
    /// Names of the features the document requires. Derived from the
    /// body by the writer, verified against the body by the reader.
    #[serde(default)]
    pub requires: Vec<String>,
}

/// A serialized model document: envelope header plus body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ModelDocument {
    #[serde(rename = "raichu_model")]
    header: FormatHeader,
    model: Model,
}

/// Borrowed counterpart of [`ModelDocument`] for writing (serializing a
/// model must not clone it).
#[derive(Debug, Serialize)]
struct ModelDocumentRef<'m> {
    #[serde(rename = "raichu_model")]
    header: FormatHeader,
    model: &'m Model,
}

/// Failure to load a serialized model **document**: a malformed
/// document, or one this engine must refuse rather than misread.
///
/// Distinct from [`ModelError`], which reports an unsound *model*: this
/// one reports a document the engine cannot honour as written.
#[derive(Debug, Error)]
pub enum LoadError {
    /// The document is not valid JSON, or does not match the schema.
    #[error("malformed model document: {0}")]
    Json(#[from] serde_json::Error),
    /// The document declares a feature this engine does not implement:
    /// it was written by a newer engine (or names a construct that does
    /// not exist). Refused, never loaded with the construct ignored.
    #[error(
        "model requires the `{feature}` feature, which this engine \
         (RAICHU {version}) does not implement; implemented: {known}"
    )]
    UnsupportedFeature {
        /// The feature name the document declares.
        feature: String,
        /// Version of the engine that refuses it.
        version: &'static str,
        /// The implemented feature names.
        known: String,
    },
    /// The document *uses* a non-baseline construct without declaring
    /// it. Refused so the declaration can never lag the content: an
    /// undeclared construct would be ignored in silence by any older
    /// engine reading the same file.
    #[error(
        "model uses the `{feature}` construct but does not declare it: \
         wrap the model body in the `{envelope}` envelope and list \
         `{feature}` under `requires`, so an engine that predates the \
         construct refuses the model instead of ignoring it"
    )]
    FeatureNotDeclared {
        /// The construct's feature name.
        feature: &'static str,
        /// The envelope key to add ([`ENVELOPE_KEY`]).
        envelope: &'static str,
    },
}

/// Objective direction of a model-level program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramSense {
    /// Minimise the objective.
    Minimize,
    /// Maximise the objective.
    Maximize,
}

/// One decision attribute and its optional numeric bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramVariable {
    /// Attribute whose value receives the optimal decision.
    pub attribute: AttrRef,
    /// Lower bound, evaluated at each discrete solve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lower: Option<Expr>,
    /// Upper bound, evaluated at each discrete solve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upper: Option<Expr>,
    /// Value published when the program is infeasible.
    pub on_infeasible: Value,
}

/// One ranged linear constraint on a program's decisions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramConstraint {
    /// Name unique within the program.
    pub name: String,
    /// Affine expression in the program's decisions.
    pub expr: Expr,
    /// Optional lower bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lower: Option<Expr>,
    /// Optional upper bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upper: Option<Expr>,
}

/// An optional secondary objective for deterministic lexicographic solving.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramTieObjective {
    /// Optimisation direction.
    pub sense: ProgramSense,
    /// Affine expression in the decisions.
    pub expr: Expr,
}

/// How equal primary optima are separated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramTieBreak {
    /// Use the solver's arbitrary optimum without a uniqueness guarantee.
    None,
    /// Optimise the declared secondary objectives in order.
    Objectives(Vec<ProgramTieObjective>),
}

/// A mixed-integer program solved when its discrete inputs change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    /// Name unique within the model.
    pub name: String,
    /// Decision attributes in lexicographic default order.
    pub variables: Vec<ProgramVariable>,
    /// Primary objective direction.
    pub sense: ProgramSense,
    /// Primary affine objective.
    pub objective: Expr,
    /// Ranged linear constraints.
    #[serde(default)]
    pub constraints: Vec<ProgramConstraint>,
    /// Boolean attribute receiving solve feasibility.
    pub feasible: AttrRef,
    /// Optional float attribute receiving the optimal objective.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_value: Option<AttrRef>,
    /// Absent uses decision-order lexicographic minimisation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tie_break: Option<ProgramTieBreak>,
    /// Maximum branch-and-bound nodes, defaulted by the solver.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_limit: Option<i64>,
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
    /// Interface-to-interface connections, each expanded into port
    /// connections by [`Model::resolved_connections`].
    ///
    /// Non-baseline construct: a document carrying it must declare
    /// [`Feature::InterfaceConnections`] in its [`FormatHeader`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interface_connections: Vec<InterfaceConnection>,
    /// Recorded indicators.
    #[serde(default)]
    pub indicators: Vec<Indicator>,
    /// Sequence-analysis targets (feared events): see [`Target`].
    #[serde(default)]
    pub targets: Vec<Target>,
    /// Programs solved at discrete fixpoints. Requires the
    /// `mixed_integer_program` format feature when nonempty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub programs: Vec<Program>,
    /// **Declared evaluation order** of the explicit sweep: the order in
    /// which the engine runs its steps at every evaluation point.
    ///
    /// A step is either an explicit equation, designated by its target
    /// attribute, or a conservative distribution operator
    /// ([`Allocation`]), designated by its name: an operator writes many
    /// attributes, so it is named rather than targeted, and a component
    /// may not give one name to both ([`ModelError::EvaluationStepAmbiguous`]).
    /// [`Model::evaluation_steps`] lists exactly what must be covered.
    ///
    /// Absent (the default), the order is positional: components in
    /// declaration order, and inside each the explicit equations in
    /// declaration order followed by the distribution operators, which is
    /// what every model written before this field compiled to.
    ///
    /// Present, it must cover **exactly** those steps: one entry each,
    /// no omission, no repetition, no entry
    /// that is not one ([`ModelError::EvaluationOrderUnknown`],
    /// [`ModelError::EvaluationOrderMissing`],
    /// [`ModelError::EvaluationOrderDuplicate`]). ODE targets are not
    /// listed: the integrator carries them, the sweep does not.
    ///
    /// The order is part of the answer, not a detail: a sweep reads
    /// whatever the previous evaluation point left in an attribute it
    /// has not reached yet, so a resolved flow network needs its own
    /// order (capability along the flow, demand against it, production
    /// along it again). Writing it into the model file makes it
    /// readable and comparable from one version to the next.
    ///
    /// Non-baseline construct: a document carrying it must declare
    /// [`Feature::EvaluationOrder`] in its [`FormatHeader`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_order: Option<Vec<AttrRef>>,
    /// **Declared unbounded magnitude**: the value an authoring layer
    /// writes where it means "no ceiling" (a volume that puts no limit on
    /// the rate it is drawn at, a demand that takes whatever comes). A
    /// document has no literal for an infinity, so such a quantity
    /// crosses it as a finite stand-in, harmless as the ceiling of a
    /// `min` against a finite demand.
    ///
    /// It is NOT harmless as a rate. Where an unbounded demand meets an
    /// unbounded ceiling, the stand-in becomes the derivative of a
    /// stock, and the integrator steps it over the stock's empty bound
    /// before any event can be located: a volume of 5 read -9.3e19. The
    /// physics is an impulse (the stock moves at once), which a rate
    /// cannot express. Declared here, the engine refuses any ODE
    /// right-hand side whose magnitude reaches this value, with a typed
    /// error naming the variable and the date, instead of integrating it.
    ///
    /// Absent (the default), no magnitude is reserved and nothing is
    /// checked. Present, it must be finite and positive
    /// ([`ModelError::UnboundedRateInvalid`]).
    ///
    /// Non-baseline construct: a document carrying it must declare
    /// [`Feature::UnboundedRate`] in its [`FormatHeader`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unbounded_rate: Option<f64>,
    /// Imported FMI co-simulation units. The feature declaration records
    /// their use; it does not grant permission to execute their native code.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fmu_units: Vec<FmuUnit>,
}

/// Typed model-validation errors. Every invalid model is reported with a
/// precise, contextual error: never a panic, never a crash.
#[derive(Debug, Error, PartialEq)]
pub enum ModelError {
    /// Two imported FMU units have the same name.
    #[error("duplicate FMU unit `{unit}`")]
    FmuDuplicateUnit {
        /// Duplicated unit name.
        unit: String,
    },
    /// An FMU unit has no usable archive path.
    #[error("FMU unit `{unit}` has an empty archive path")]
    FmuEmptyPath {
        /// Unit name.
        unit: String,
    },
    /// A communication interval must be finite and positive.
    #[error(
        "FMU unit `{unit}` has invalid communication step {step}; expected a finite positive value"
    )]
    FmuInvalidStep {
        /// Unit name.
        unit: String,
        /// Invalid interval in model time units.
        step: f64,
    },
    /// A binding refers to an attribute absent from the model.
    #[error("FMU unit `{unit}` binds variable `{variable}` to unknown attribute `{attribute}`")]
    FmuBindingUnknownAttribute {
        /// Unit name.
        unit: String,
        /// Qualified model attribute.
        attribute: String,
        /// FMU variable name.
        variable: String,
    },
    /// An FMU output variable is mapped more than once.
    #[error("FMU unit `{unit}` output variable `{variable}` is bound more than once")]
    FmuOutputDuplicated {
        /// Unit name.
        unit: String,
        /// Duplicated FMU variable.
        variable: String,
    },
    /// An FMU output and another source write the same attribute.
    #[error(
        "FMU unit `{unit}` output `{variable}` writes `{attribute}`, which {writer} also writes"
    )]
    FmuOutputWritten {
        /// Unit name.
        unit: String,
        /// Qualified model attribute.
        attribute: String,
        /// FMU output variable name.
        variable: String,
        /// Other writer.
        writer: String,
    },
    /// A program declaration violates its structural or linearity contract.
    #[error("program `{program}`: {term}: {reason}")]
    ProgramInvalid {
        /// Program name.
        program: String,
        /// Offending field, attribute or expression.
        term: String,
        /// Why the declaration is unsound.
        reason: String,
    },
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
    /// An interface connection names an interface that does not exist.
    #[error("interface connection endpoint `{component}.{interface}` does not exist ({side})")]
    InterfaceConnectionUnknown {
        /// The referenced component.
        component: String,
        /// The referenced interface.
        interface: String,
        /// Which side of the connection (`from` / `to`).
        side: &'static str,
    },
    /// A port of one interface has no port of the same name in the
    /// interface it is connected to.
    #[error(
        "interface connection `{}.{}` ↔ `{}.{}`: port `{port}` of `{component}` has no \
         port of the same name on the other side (interfaces are paired by port name)",
        from.component, from.interface, to.component, to.interface
    )]
    InterfacePortUnmatched {
        /// The `from` side.
        from: Box<InterfaceRef>,
        /// The `to` side.
        to: Box<InterfaceRef>,
        /// The component whose port is left unpaired.
        component: String,
        /// The unpaired port.
        port: String,
    },
    /// Two ports paired by name have the same direction.
    #[error(
        "interface connection `{}.{}` ↔ `{}.{}`: port `{port}` is an {dir} port on both \
         sides, and a pair must join an out port to an in port",
        from.component, from.interface, to.component, to.interface
    )]
    InterfacePortDirection {
        /// The `from` side.
        from: Box<InterfaceRef>,
        /// The `to` side.
        to: Box<InterfaceRef>,
        /// The port name both sides carry.
        port: String,
        /// The direction both carry (`in` / `out`).
        dir: &'static str,
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
    /// An indicator's threshold ([`IndicatorTarget::Predicate`]) compares
    /// two kinds that have no comparison: a boolean attribute ordered
    /// against anything, or a numeric attribute tested against a boolean.
    ///
    /// Refused at build time rather than answered `false` at every
    /// instant, because a threshold nobody can evaluate is exactly the
    /// silence the variant exists to end.
    #[error("indicator `{indicator}`: {detail}")]
    IndicatorPredicateKind {
        /// The indicator.
        indicator: String,
        /// Which two kinds, and why they do not compare.
        detail: String,
    },
    /// A channel list was declared on an **in** port. Channels are
    /// per-connection quantities a producer hands out; a consumer reads
    /// them through `port_agg` and never declares them.
    #[error("in-port `{component}.{port}` declares channels (out-ports only)")]
    InPortWithChannels {
        /// The owning component.
        component: String,
        /// The in port.
        port: String,
    },
    /// Two channels of one port share a name.
    #[error("duplicate channel `{channel}` on port `{component}.{port}`")]
    DuplicatePortChannel {
        /// The owning component.
        component: String,
        /// The port.
        port: String,
        /// The duplicated channel name.
        channel: String,
    },
    /// The attribute the compiler would materialise for a (connection,
    /// channel) pair is already declared by hand on the same component.
    #[error(
        "channel `{channel}` of port `{component}.{port}` materialises the \
         attribute `{component}.{attribute}`, which collides with the \
         attribute `{attribute}` already declared on component \
         `{component}`: rename the declared attribute, rename the channel, \
         or name the connection"
    )]
    ChannelAttributeCollision {
        /// The producing component.
        component: String,
        /// The out port declaring the channel.
        port: String,
        /// The channel.
        channel: String,
        /// The materialised (and already declared) attribute name.
        attribute: String,
    },
    /// Two connections materialise the same attribute name (parallel
    /// edges between the same pair of ports, none of them named).
    #[error(
        "connections #{first} and #{second} both materialise \
         `{component}.{attribute}` for channel `{channel}`: give at least \
         one of them a `name`"
    )]
    DuplicateChannelAttribute {
        /// The producing component.
        component: String,
        /// The materialised attribute name.
        attribute: String,
        /// The channel.
        channel: String,
        /// Index of the first connection producing the name.
        first: usize,
        /// Index of the colliding connection.
        second: usize,
    },
    /// A `port_agg` names a channel that a connected out port does not
    /// declare.
    #[error(
        "{context}: aggregation on in-port `{port}` names channel \
         `{channel}`, which the connected out-port `{producer}` does not \
         declare"
    )]
    ExprUnknownPortChannel {
        /// Where the expression sits.
        context: String,
        /// The in port, as `component.port`.
        port: String,
        /// The named channel.
        channel: String,
        /// The connected out port lacking the channel, as
        /// `component.port`.
        producer: String,
    },
    /// **Nonlinear algebraic cycle**: explicit equations depend on one
    /// another in a cycle, and at least one right-hand side is not
    /// linear in the cycle's unknowns.
    ///
    /// A cycle of explicit equations is a fixpoint equation; a *linear*
    /// one is solved in place by the engine as one block, a nonlinear
    /// one has no iteration this engine vouches for (a Newton scheme
    /// would answer without a convergence argument, and a threshold or
    /// a `min` inside the cycle turns its answer into a switching
    /// sequence the sweep cannot settle). The refusal names the term
    /// that makes the cycle nonlinear. An **ODE-integrated** attribute
    /// breaks the cycle, because the integrator carries its value
    /// instead of the algebra recomputing it: it is the *capacity* of
    /// the flow vocabulary.
    #[error(
        "explicit equations form a cycle that is not linear in its \
         unknowns: the right-hand side of `{component}.{attribute}` \
         reads `{term}`, which no linear solve can carry. Integrate one \
         of the cycle's attributes (an ODE target, that is a capacity) \
         or cut the dependency"
    )]
    AlgebraicNonlinear {
        /// The component of the offending equation.
        component: String,
        /// The target attribute of the offending equation.
        attribute: String,
        /// The offending term, rendered as written.
        term: String,
    },
    /// **Singular algebraic cycle**: the cycle's linear system does not
    /// determine one value per attribute.
    ///
    /// Structural (no assignment of equations to unknowns exists on the
    /// sparsity pattern: some attribute is never determined), or
    /// rank-deficient with every coefficient constant at build time
    /// (`x = y`, `y = x`, `x = x`: the rows are linearly dependent). In
    /// both cases the equations do not determine a unique solution.
    #[error(
        "explicit equations form a cycle whose linear system is singular \
         ({reason}): {variables}. No unique instantaneous solution is determined; \
         integrate one of the cycle's attributes (an ODE target, that is \
         a capacity) or cut the dependency"
    )]
    AlgebraicSingular {
        /// The cycle's unknowns, `component.attribute`, comma-joined.
        variables: String,
        /// Which of the two levels refused.
        reason: SingularReason,
    },
    /// A declared evaluation order places a reader before a producer
    /// required by the algebraic-block condensation.
    #[error(
        "evaluation order runs `{first}` before `{second}`, but the two \
         first depends on the second after algebraic blocks form: \
         reorder those steps, omit the declared order, or cut the dependency"
    )]
    AlgebraicOrderConflict {
        /// The reader placed before its producer.
        first: String,
        /// The producer that must run first.
        second: String,
    },
    /// A connection joins a **discrete** flow (a boolean: present or
    /// absent) and a **continuous** one (a numeric quantity) into the
    /// same in port.
    ///
    /// One aggregation cannot mix the two families: `sum` counts a
    /// boolean as one unit, so a boolean joining a numeric balance
    /// injects one unit of quantity that nothing produced, and it turns
    /// the aggregation's own type from a count into a quantity. The
    /// boolean aggregations (`all`, `any`) reject the numeric side
    /// outright, and today they do so mid-simulation. A *homogeneous*
    /// fan-in stays legal in either family: booleans summed into a
    /// balance are the modeller's deliberate unit count (the spent-fuel
    /// pool counts its running trains that way).
    #[error(
        "in-port `{port}` is fed both by `{other}` ({other_kind:?}, a \
         discrete flow) and by `{producer}` ({kind:?}, a continuous \
         quantity): one aggregation cannot mix the two families, a \
         boolean counting as one unit of the quantity; feed them to two \
         separate in ports"
    )]
    ConnectionFamilyMismatch {
        /// The destination in port, as `component.port`.
        port: String,
        /// The producer feeding it first, as `component.port`.
        other: String,
        /// The kind that producer exports.
        other_kind: AttrKind,
        /// The producer of the offending connection, as `component.port`.
        producer: String,
        /// The kind it exports.
        kind: AttrKind,
    },
    /// The declared evaluation order names something that is not an
    /// explicit equation of this model.
    #[error(
        "evaluation order names `{component}.{attribute}`, which is \
         neither an explicit equation nor a distribution operator of the \
         model (only those are swept; an ODE target is carried by the \
         integrator)"
    )]
    EvaluationOrderUnknown {
        /// The named component.
        component: String,
        /// The named attribute.
        attribute: String,
    },
    /// A declared explicit equation is absent from the evaluation
    /// order. A partial order has no meaning: the missing equation
    /// would have no evaluation point at all.
    #[error(
        "sweep step `{component}.{attribute}` is missing from the \
         evaluation order, which must list every explicit equation and \
         every distribution operator exactly once"
    )]
    EvaluationOrderMissing {
        /// The omitted component.
        component: String,
        /// The omitted attribute.
        attribute: String,
    },
    /// The declared evaluation order lists one equation twice.
    #[error(
        "evaluation order lists `{component}.{attribute}` more than \
         once; each sweep step runs exactly once"
    )]
    EvaluationOrderDuplicate {
        /// The repeated component.
        component: String,
        /// The repeated attribute.
        attribute: String,
    },
    /// A distribution operator names the same step as an explicit
    /// equation of its component, so an evaluation-order entry could not
    /// tell the two apart.
    #[error(
        "component `{component}` declares both an explicit equation on \
         `{name}` and a distribution operator named `{name}`: an \
         evaluation-order entry could designate either, so one of them \
         must be renamed"
    )]
    EvaluationStepAmbiguous {
        /// The owning component.
        component: String,
        /// The doubly-used step name.
        name: String,
    },
    /// The declared unbounded magnitude is not a finite positive number.
    #[error(
        "the declared unbounded rate is {value}: it must be finite and \
         positive, the magnitude no integrated rate may reach"
    )]
    UnboundedRateInvalid {
        /// The declared value.
        value: f64,
    },
    /// A transition's edge effect writes an attribute that an explicit
    /// equation or a sensitive function also writes: the next evaluation
    /// would erase the one-shot write. (On an ODE target an edge effect is
    /// a reset map, and accepted.)
    #[error(
        "transition `{transition}` writes `{attribute}` on its firing edge, \
         and {writer} writes it too: the next evaluation would erase the \
         one-shot write. An edge effect needs an attribute nothing else \
         writes, or the target of an ODE equation, which it resets"
    )]
    TransitionEffectOverwritten {
        /// `component.automaton.transition`.
        transition: String,
        /// `component.attribute`.
        attribute: String,
        /// The other writer.
        writer: String,
    },
    /// A distribution operator does not sit on an out port of its
    /// component.
    #[error(
        "distribution operator `{component}.{allocation}` names `{port}`, \
         which is not an out port of `{component}` (the quantity is \
         handed out over the port's connections)"
    )]
    AllocationPortInvalid {
        /// The owning component.
        component: String,
        /// The operator.
        allocation: String,
        /// The named port.
        port: String,
    },
    /// A distribution operator names a channel its port does not declare.
    #[error(
        "distribution operator `{component}.{allocation}` reads/writes its \
         {role} on channel `{channel}`, which its port does not declare"
    )]
    AllocationUnknownChannel {
        /// The owning component.
        component: String,
        /// The operator.
        allocation: String,
        /// The missing channel.
        channel: String,
        /// Which side names it (`demand` / `allocated`).
        role: &'static str,
    },
    /// A distribution operator reads and writes the same channel: every
    /// evaluation would overwrite the demands with the allocations.
    #[error(
        "distribution operator `{component}.{allocation}` uses channel \
         `{channel}` for both its demand and its allocated quantity: \
         writing the allocation would destroy the demand it was computed \
         from"
    )]
    AllocationChannelReused {
        /// The owning component.
        component: String,
        /// The operator.
        allocation: String,
        /// The doubly-used channel.
        channel: String,
    },
    /// Fixed shares that do not sum to 1 over the port's connections:
    /// the operator would systematically hand out more or less than what
    /// is available.
    #[error(
        "the fixed shares of distribution operator \
         `{component}.{allocation}` on port `{port}` sum to {sum}, not 1"
    )]
    AllocationSharesNotUnit {
        /// The owning component.
        component: String,
        /// The operator.
        allocation: String,
        /// The out port carrying the flow.
        port: String,
        /// The offending sum.
        sum: f64,
    },
    /// A policy parameter names a consumer the operator's port does not
    /// feed.
    #[error(
        "distribution operator `{component}.{allocation}` declares a \
         policy value for `{consumer}`, which its port does not feed"
    )]
    AllocationParamUnknown {
        /// The owning component.
        component: String,
        /// The operator.
        allocation: String,
        /// The unconnected consumer, as `component.port`.
        consumer: String,
    },
    /// A connection of the operator's port carries no policy parameter.
    #[error(
        "distribution operator `{component}.{allocation}` declares no \
         policy value for `{consumer}`, which its port feeds; a keyed \
         policy must cover every connection"
    )]
    AllocationParamMissing {
        /// The owning component.
        component: String,
        /// The operator.
        allocation: String,
        /// The uncovered consumer, as `component.port`.
        consumer: String,
    },
    /// Two policy parameters name the same consumer.
    #[error(
        "distribution operator `{component}.{allocation}` declares two \
         policy values for `{consumer}`"
    )]
    AllocationParamDuplicate {
        /// The owning component.
        component: String,
        /// The operator.
        allocation: String,
        /// The repeated consumer, as `component.port`.
        consumer: String,
    },
    /// A policy parameter is outside its domain (a share must be a
    /// finite number in [0, 1]; a priority rank must be finite).
    #[error(
        "distribution operator `{component}.{allocation}` declares the \
         invalid policy value {value} for `{consumer}`"
    )]
    AllocationParamInvalid {
        /// The owning component.
        component: String,
        /// The operator.
        allocation: String,
        /// The consumer, as `component.port`.
        consumer: String,
        /// The offending value.
        value: f64,
    },
    /// Something other than the operator writes one of the quantities it
    /// allocates. Two writers of one attribute mean the last one silently
    /// wins, which is exactly the order dependence the operator exists to
    /// remove.
    #[error(
        "`{writer}` also writes `{component}.{attribute}`, which the \
         distribution operator `{component}.{allocation}` allocates: an \
         allocated quantity has exactly one writer"
    )]
    AllocationTargetWritten {
        /// The producing component.
        component: String,
        /// The operator.
        allocation: String,
        /// The doubly-written attribute.
        attribute: String,
        /// The other writer (an equation, a sensitive function or
        /// another operator).
        writer: String,
    },
    /// **A priority split whose consumers return their surplus.** The two
    /// constructs are each sound on their own; their composition is not
    /// covered.
    ///
    /// The engine resolves a flow network by iterating the ordered sweep
    /// *downward* from a cold start in which nothing has been delivered
    /// yet. That start over-estimates every delivery, so the sequence is
    /// non-increasing and bounded below, and therefore converges. The
    /// over-estimate is what carries the argument, and nobody has shown
    /// it for a strict priority order: a consumer whose demand shrinks
    /// because it was served can move the point where the supply runs
    /// out, which *raises* a later consumer's delivery instead of
    /// lowering it, and the sequence stops being monotone.
    ///
    /// Refused here rather than accepted and then reported as
    /// non-convergent: a resolution that does not settle would either
    /// stop at a cap with an arbitrary iterate or oscillate, and both
    /// answers look like numbers. Use `proportional` or `shares`, whose
    /// weighted split keeps the over-estimate, or cut the dependency so
    /// the demand no longer reads what the operator handed out.
    #[error(
        "distribution operator `{component}.{allocation}` splits by \
         priority while its consumers return surplus: the demand on \
         `{demand}` depends on the quantity this operator allocates on \
         `{allocated}`. The descending resolution carries no convergence \
         argument for that composition; use the `proportional` or \
         `shares` policy, or cut the dependency"
    )]
    AllocationPrioritySurplusReturn {
        /// The producing component.
        component: String,
        /// The operator.
        allocation: String,
        /// The materialised demand attribute the loop starts from.
        demand: String,
        /// The materialised allocated attribute it reaches.
        allocated: String,
    },
}

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

/// Strongly connected components of `edges` (one node per index, edge
/// `i → j` when `edges[i]` contains `j`), Tarjan's algorithm, iterative
/// so a deep chain of equations cannot exhaust the stack.
///
/// Components come out in reverse topological order (a component before
/// any component that reaches it), each as its nodes in unspecified
/// order; a singleton without a self-edge is a node of its own and is
/// still emitted, letting the caller decide what a component means.
fn tarjan_components(edges: &[Vec<usize>]) -> Vec<Vec<usize>> {
    #[derive(Clone, Copy)]
    struct Frame {
        node: usize,
        child: usize,
    }
    let mut next_index = 0usize;
    let mut index_of = vec![usize::MAX; edges.len()];
    let mut lowlink = vec![0usize; edges.len()];
    let mut on_stack = vec![false; edges.len()];
    let mut stack: Vec<usize> = Vec::new();
    let mut components: Vec<Vec<usize>> = Vec::new();
    for root in 0..edges.len() {
        if index_of[root] != usize::MAX {
            continue;
        }
        let mut frames = vec![Frame {
            node: root,
            child: 0,
        }];
        while let Some(frame) = frames.pop() {
            if frame.child == 0 {
                index_of[frame.node] = next_index;
                next_index += 1;
                lowlink[frame.node] = index_of[frame.node];
                stack.push(frame.node);
                on_stack[frame.node] = true;
            }
            let mut child = frame.child;
            let mut descended = false;
            while child < edges[frame.node].len() {
                let successor = edges[frame.node][child];
                if index_of[successor] == usize::MAX {
                    frames.push(Frame {
                        node: frame.node,
                        child: child + 1,
                    });
                    frames.push(Frame {
                        node: successor,
                        child: 0,
                    });
                    descended = true;
                    break;
                }
                if on_stack[successor] {
                    lowlink[frame.node] = lowlink[frame.node].min(index_of[successor]);
                }
                child += 1;
            }
            if descended {
                continue;
            }
            if lowlink[frame.node] == index_of[frame.node] {
                let mut component = Vec::new();
                while let Some(member) = stack.pop() {
                    on_stack[member] = false;
                    component.push(member);
                    if member == frame.node {
                        break;
                    }
                }
                components.push(component);
            }
            if let Some(parent) = frames.last_mut() {
                lowlink[parent.node] = lowlink[parent.node].min(lowlink[frame.node]);
            }
        }
    }
    components
}

/// Whether the `rows × columns` sparsity pattern admits an assignment of
/// one distinct column to every row (a perfect matching), Kuhn's
/// augmenting paths, rows in order and candidates ascending, so the
/// answer : and the refusal that a `false` produces : is a property of
/// the pattern alone.
fn has_perfect_matching(pattern: &[Vec<usize>], columns: usize) -> bool {
    if pattern.len() != columns {
        return false;
    }
    // Column → row currently assigned to it.
    let mut owner: Vec<Option<usize>> = vec![None; columns];
    for row in 0..pattern.len() {
        let mut seen = vec![false; columns];
        if !assign(row, pattern, &mut owner, &mut seen) {
            return false;
        }
    }
    true
}

/// Augmenting-path search for `row` (see [`has_perfect_matching`]).
fn assign(
    row: usize,
    pattern: &[Vec<usize>],
    owner: &mut [Option<usize>],
    seen: &mut [bool],
) -> bool {
    for &column in &pattern[row] {
        if seen[column] {
            continue;
        }
        seen[column] = true;
        match owner[column] {
            None => {
                owner[column] = Some(row);
                return true;
            }
            Some(previous) => {
                if assign(previous, pattern, owner, seen) {
                    owner[column] = Some(row);
                    return true;
                }
            }
        }
    }
    false
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
    /// Load a model from its JSON representation, in either accepted
    /// shape: the **format envelope** (see [`FormatHeader`]) or the
    /// bare, envelope-less body kept readable for the existing corpus.
    ///
    /// The feature gate runs here: the document's declared features
    /// must all be implemented by this engine, and must cover every
    /// non-baseline construct the body actually contains
    /// ([`Model::required_features`]). A bare body declares nothing, so
    /// it may only use baseline constructs.
    ///
    /// This checks the *document*; call [`Model::validate`] afterwards
    /// for the model's structural soundness.
    pub fn from_json(json: &str) -> Result<Self, LoadError> {
        let (declared, model) = Self::parse_document(json)?;
        model.check_features(declared.as_deref())?;
        Ok(model)
    }

    /// Serialize the model as a complete document: the mandatory format
    /// envelope plus the body. The envelope's `requires` list is
    /// **derived** from the model ([`Model::required_features`]), never
    /// hand-written, so it cannot lag what the body contains.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&ModelDocumentRef {
            header: self.format_header(),
            model: self,
        })
    }

    /// Rewrite an authored document (bare or already enveloped) as a
    /// sealed one, deriving the feature list from the body.
    ///
    /// This is the **writer** side of the gate, for an authoring layer
    /// that produces a body and must not compose the envelope itself.
    /// It tolerates a body that uses a construct without declaring it:
    /// that is precisely what this call exists to fix. It does *not*
    /// tolerate a declaration naming a feature this engine does not
    /// implement, which sealing cannot fix and re-deriving would
    /// silently drop: refused as [`Model::from_json`] refuses it.
    pub fn seal_json(document_json: &str) -> Result<String, LoadError> {
        let (declared, model) = Self::parse_document(document_json)?;
        Self::declared_features(declared.as_deref())?;
        Ok(model.to_json()?)
    }

    /// The envelope header this model would be written with.
    #[must_use]
    pub fn format_header(&self) -> FormatHeader {
        FormatHeader {
            format: FORMAT_REVISION,
            requires: self
                .required_features()
                .into_iter()
                .map(|feature| feature.name().to_owned())
                .collect(),
        }
    }

    /// The features this model **actually** requires, derived from the
    /// constructs it contains.
    ///
    /// This is the authority the declared list is checked against: a
    /// declaration is a mirror the reader verifies, never a claim it
    /// trusts, so a hand-written model cannot use a construct and omit
    /// it from the envelope.
    #[must_use]
    pub fn required_features(&self) -> BTreeSet<Feature> {
        let mut features = BTreeSet::new();
        if !self.fmu_units.is_empty() {
            features.insert(Feature::Fmi);
        }
        if !self.programs.is_empty() {
            features.insert(Feature::MixedIntegerProgram);
        }
        if !self.interface_connections.is_empty() {
            features.insert(Feature::InterfaceConnections);
        }
        if self.evaluation_order.is_some() {
            features.insert(Feature::EvaluationOrder);
        }
        if self.unbounded_rate.is_some() {
            features.insert(Feature::UnboundedRate);
        }
        if self.components.iter().any(|component| {
            component
                .automata
                .iter()
                .any(|automaton| automaton.transitions.iter().any(|t| !t.effects.is_empty()))
        }) {
            features.insert(Feature::TransitionEffects);
        }
        if self.components.iter().any(|component| {
            component.automata.iter().any(|automaton| {
                automaton
                    .transitions
                    .iter()
                    .any(|t| t.kind == Some(TransitionKind::Observation))
            })
        }) {
            features.insert(Feature::ObserverPriority);
        }
        if self
            .components
            .iter()
            .any(|component| !component.allocations.is_empty())
        {
            features.insert(Feature::Allocation);
        }
        features
    }

    /// Split a document into its declared feature list (`None` for a
    /// bare, envelope-less body) and its model body. No feature gate:
    /// the callers apply the part of it they need.
    fn parse_document(json: &str) -> Result<(Option<Vec<String>>, Self), LoadError> {
        let value: serde_json::Value = serde_json::from_str(json)?;
        if value.get(ENVELOPE_KEY).is_some() {
            let document: ModelDocument = serde_json::from_value(value)?;
            Ok((Some(document.header.requires), document.model))
        } else {
            Ok((None, serde_json::from_value(value)?))
        }
    }

    /// Resolve a declared feature list, refusing any name this engine
    /// does not implement. `None` is a bare body: it declares nothing.
    fn declared_features(declared: Option<&[String]>) -> Result<BTreeSet<Feature>, LoadError> {
        let mut allowed: BTreeSet<Feature> = BTreeSet::new();
        for name in declared.unwrap_or(&[]) {
            let Some(feature) = Feature::parse(name) else {
                return Err(LoadError::UnsupportedFeature {
                    feature: name.clone(),
                    version: env!("CARGO_PKG_VERSION"),
                    known: Feature::known(),
                });
            };
            allowed.insert(feature);
        }
        Ok(allowed)
    }

    /// Feature gate: every declared name must be implemented, and the
    /// declaration must cover every non-baseline construct the body
    /// contains. `declared` is `None` for a bare body, which declares
    /// nothing and may therefore only use baseline constructs.
    fn check_features(&self, declared: Option<&[String]>) -> Result<(), LoadError> {
        let allowed = Self::declared_features(declared)?;
        for feature in self.required_features() {
            if !allowed.contains(&feature) {
                return Err(LoadError::FeatureNotDeclared {
                    feature: feature.name(),
                    envelope: ENVELOPE_KEY,
                });
            }
        }
        Ok(())
    }

    /// Validate structural soundness: unique names, resolvable
    /// references, well-formed distributions, matching kinds. Returns the *first*
    /// error encountered in deterministic (declaration) order.
    ///
    /// The engine (`raichu-core`) only accepts validated models; this is
    /// the fail-fast gate (typed errors at build
    /// time, never mid-simulation surprises).
    pub fn validate(&self) -> Result<(), ModelError> {
        if !self.interface_connections.is_empty() {
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
            Expr::Sin { arg } | Expr::Exp { arg } => {
                Self::program_expr_numeric(arg, scopes, decisions)
            }
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
        if !self.interface_connections.is_empty() {
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

    /// This model with its interface connections expanded into
    /// [`Model::connections`] and the list emptied: the model the engine
    /// actually runs. Borrowed when there is nothing to expand.
    ///
    /// # Errors
    ///
    /// As [`Model::resolved_connections`].
    pub fn with_resolved_connections(&self) -> Result<Cow<'_, Model>, ModelError> {
        if self.interface_connections.is_empty() {
            return Ok(Cow::Borrowed(self));
        }
        let connections = self.resolved_connections()?.into_owned();
        Ok(Cow::Owned(Model {
            connections,
            interface_connections: Vec::new(),
            ..self.clone()
        }))
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
        for members in tarjan_components(&edges) {
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
            Expr::Sin { arg } | Expr::Exp { arg } => {
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
