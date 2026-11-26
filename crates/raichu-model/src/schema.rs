//! The data types of the native model: what a model document declares.

#[cfg(doc)]
use crate::Feature;
#[cfg(doc)]
use crate::FormatHeader;
#[cfg(doc)]
use crate::ModelError;
use raichu_expr::{Assignment, AttrRef, CmpOp, Expr, PortRef, Value};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The type of an attribute. A `String` discrete state is reserved: it
/// would extend this enum without breaking serialized models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
/// propagations write them, the causal journal records them, the
/// snapshot carries them, and indicators observe them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
    /// **Port equations**: per-connection channel equations (see
    /// [`PortEquation`]). Non-baseline construct: a document carrying one
    /// must declare [`Feature::PortEquations`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub equations: Vec<PortEquation>,
}

/// An equation declared on a port and evaluated **once per connection**
/// through it, writing that connection's `channel`.
///
/// A component cannot name its connections, so this is how it states a
/// per-connection quantity locally. On an **in** port it writes the
/// channel of every connection that feeds the port (a consumer publishing
/// its demand on each supplier's edge); on an **out** port, the channel of
/// every connection leaving it (a producer's claim per consumer). Channels
/// live on the producer's out port, which must declare `channel`.
///
/// Inside `expr`, an `attr` leaf whose component is the reserved name
/// [`EDGE`] reads another channel of the same connection
/// (`{"component": "$edge", "attribute": "demand"}`); every other leaf
/// reads as anywhere else. A total over an out port's connections is read
/// back with `port_agg` (`sum` or `count`) on that out port.
///
/// The construct is an authoring form: [`Model::with_resolved_connections`]
/// lowers it to one ordinary explicit equation per connection, named by
/// the channel attribute it writes, so the engine never sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PortEquation {
    /// The channel written, on each connection through the port.
    pub channel: String,
    /// Right-hand side, evaluated per connection.
    pub expr: Expr,
}

/// Reserved component name reading a channel of the connection a
/// [`PortEquation`] is evaluated for.
pub const EDGE: &str = "$edge";

/// Reserved component name of an edge effect written **through a port**:
/// `{"component": "$port", "attribute": "<in port>"}` as the target of a
/// [`Transition::effects`] assignment names an in port of the transition's
/// own component, and the effect writes the attribute exported by every
/// out port connected to it.
///
/// A component cannot name its neighbours, so this is how a transition
/// acts on one (a unit whose repair refills the buffer it feeds). The
/// construct is an authoring form: [`Model::with_resolved_connections`]
/// lowers it to one ordinary edge effect per connection, in connection
/// declaration order, so the engine never sees it. An in port nothing is
/// connected to writes nothing. Only edge effects may write through a
/// port: a propagation is a level re-evaluated to a fixpoint, and
/// one writing a neighbour is refused. Non-baseline construct: a document
/// carrying it must declare [`Feature::PortEffects`].
pub const PORT: &str = "$port";

/// A named group of ports, validated at build time (each port must
/// exist), that an [`InterfaceConnection`] joins to another at once. The
/// ports stay the fundamental notion: the engine runs on the port
/// connections an interface connection stands for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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

/// One step of the explicit sweep with the attributes it writes and the
/// attributes it reads at the instant it is evaluated, produced by
/// [`Model::evaluation_step_reads`].
///
/// A step is named as in [`Model::evaluation_order`]: an explicit
/// equation by its target attribute, a distribution operator by its own
/// name. In-port aggregations are resolved through the connections that
/// feed them, so `reads` names the producer-side attributes, and `count`
/// reads nothing (it reads the topology, never a value).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationStepReads {
    /// Component owning the step.
    pub component: String,
    /// The step's name in [`Model::evaluation_order`].
    pub step: String,
    /// Attributes the step writes: the equation's target, or one
    /// allocated channel attribute per outgoing connection of an operator.
    pub writes: Vec<AttrRef>,
    /// Attributes the step reads, in expression order, possibly repeated.
    pub reads: Vec<AttrRef>,
}

/// Occurrence distribution of a transition: when it fires once armed, and,
/// for an instantaneous branching, which target it reaches.
///
/// Four families: a deterministic delay (`Delay`), an instantaneous
/// branching (`Inst`), a boundary crossing (`Boundary`), and seven
/// laws of a random delay (`Exp`, `Weibull`, `Lognormal`, `Gamma`,
/// `Uniform`, `Normal`, `Empirical`). Every law except `Inst` has exactly
/// one target state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
    /// Boundary transition (paper rule `schedule_boundary`): fires exactly when
    /// the continuous trajectory makes its guard become true. The guard
    /// is the boundary predicate: it must contain at least one ordering
    /// comparison (`<`, `≤`, `>`, `≥`) between float expressions,
    /// possibly composed with and/or/not and discrete gates, so the
    /// engine can locate the crossing by root-finding on the signed
    /// margin. Exactly one target state.
    #[serde(alias = "watched")]
    Boundary,
    /// Exponential distribution (paper rules `schedule_stochastic` and `reschedule_modifiable`): a
    /// spontaneous PDMP jump with survival
    /// `P(T > t) = exp(−∫₀ᵗ λ(x(u)) du)`. Exactly one target state,
    /// and exactly one of the two rate forms:
    ///
    /// - `rate`: fixed λ ≥ 0; the firing date is sampled as
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
    ///   like a boundary crossing. λ must evaluate ≥ 0.
    Exp {
        /// Fixed occurrence rate λ ≥ 0 (events per time unit). At 0 the
        /// transition is armed and never fires.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rate: Option<f64>,
        /// State-dependent occurrence rate λ(x) ≥ 0 (events per time
        /// unit), evaluated on the current model state.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rate_expr: Option<Expr>,
    },
    /// Weibull distribution: delay sampled with CDF `1 − e^{−(t/scale)^shape}`.
    Weibull {
        /// Shape parameter k > 0.
        shape: f64,
        /// Scale parameter λ > 0 (time units).
        scale: f64,
    },
    /// Log-normal distribution: `ln(delay) ~ N(mu, sigma²)`.
    Lognormal {
        /// Mean of the underlying normal.
        mu: f64,
        /// Standard deviation of the underlying normal (> 0).
        sigma: f64,
    },
    /// Gamma distribution: shape/scale parametrisation
    /// (mean = shape·scale).
    Gamma {
        /// Shape parameter k > 0.
        shape: f64,
        /// Scale parameter θ > 0 (time units).
        scale: f64,
    },
    /// Uniform distribution: delay drawn uniformly in `[low, high)`.
    Uniform {
        /// Lower bound (≥ 0).
        low: f64,
        /// Upper bound (> low).
        high: f64,
    },
    /// Normal distribution truncated to `[low, high]`: the delay follows
    /// `N(mu, sigma²)` conditioned on the interval, its density
    /// renormalised by the probability the untruncated law gives to it.
    /// A draw is never moved to a bound, so no probability sits on one.
    /// A delay cannot be negative, hence the lower bound, 0 unless
    /// stated; the upper bound is optional.
    Normal {
        /// Mean of the untruncated law (time units).
        mu: f64,
        /// Standard deviation of the untruncated law (> 0, time units).
        sigma: f64,
        /// Lower bound of the delay (≥ 0); 0 when absent.
        #[serde(default, skip_serializing_if = "is_zero")]
        low: f64,
        /// Upper bound of the delay (> low); unbounded when absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        high: Option<f64>,
    },
    /// Empirical / user-defined distribution: inverse-CDF sampling from a
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

/// Whether a bound holds its default of 0 and is left out when written.
#[allow(clippy::trivially_copy_pass_by_ref)] // the signature serde calls
fn is_zero(value: &f64) -> bool {
    *value == 0.0
}

/// The kind of a continuous-evolution equation (CEvol of the paper).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct FmuBinding {
    /// Attribute read for an input, or written by an output.
    pub attribute: AttrRef,
    /// Variable name in the FMU description.
    pub variable: String,
}

/// An initial value assigned to an FMU parameter before simulation starts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct FmuParameter {
    /// Parameter name in the FMU description.
    pub variable: String,
    /// Typed initial value, checked against the FMU description at preparation.
    pub value: Value,
}

/// One imported FMI co-simulation unit and its model-facing bindings.
/// Loading its native library requires a separate run-side permission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
/// # Why it is not a propagation
///
/// A [`Propagation`] effect writes **one** target. A split cannot be
/// written that way: the share handed to one consumer depends on what
/// every other consumer asked for, so the operator writes the whole vector
/// at once, as a single indivisible step.
///
/// # Why it sits on the explicit-equation path
///
/// The engine runs only the explicit pass inside its solver callbacks; a
/// trigger-driven entity never runs during integration. Quantities written
/// from a trigger would stay frozen across an integration segment, every
/// boundary margin reading them would freeze with them, and a boundary
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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

/// Whether the minimal-sequence reduction may cancel a cycle group, that
/// is drop a failure repaired before the feared event on the hypothesis
/// that a repaired component no longer contributes
/// ([`Transition::cycle_cancellation`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum CycleCancellation {
    /// The engine decides from the compiled model: the group is not
    /// cancelled when its effect reaches a feared event through a
    /// quantity that remembers it.
    #[default]
    Auto,
    /// The group is never cancelled: its failures stay in the sequences,
    /// repaired or not. For an effect the analysis cannot see, such as a
    /// counter a propagation keeps.
    Never,
    /// The group is always cancelled, whatever the analysis finds: the
    /// analyst asserts that the effect of a repaired failure is gone.
    Always,
}

impl CycleCancellation {
    /// Whether this is the default, [`CycleCancellation::Auto`].
    #[must_use]
    pub fn is_auto(&self) -> bool {
        *self == CycleCancellation::Auto
    }
}

/// A transition of an automaton.
///
/// Maps to the paper's transitions ⟨q_src, guard, delay,
/// dest-distribution⟩. The occurrence law ([`Distrib`]) selects the
/// scheduling rule: a deterministic delay or an instantaneous branching
/// (`schedule_deterministic`), a random delay (`schedule_stochastic`), or a
/// boundary crossing (`schedule_boundary`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Transition {
    /// Transition name, unique inside its automaton.
    pub name: String,
    /// Source state (must exist in the automaton).
    pub source: String,
    /// Guard: the transition is armed while the guard holds.
    /// Absent means "always true".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard: Option<Expr>,
    /// Target states: one per branch for `Inst`, with
    /// `probs.len() == targets.len() - 1`; exactly one for every other law.
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
    /// **Sequence analysis**: the targets whose entry a `monitored`
    /// transition records, when not all of them. `None` (the default)
    /// records every target; a list records the entry into those targets
    /// only, and a firing that realises another branch records nothing.
    ///
    /// This is how a Bernoulli draw is monitored: its winning branch is
    /// the event of the mission, while its losing branch only parks the
    /// automaton until the solicitation falls, and recording that park
    /// would break the strict failure/repair alternation the cycle filter
    /// relies on. Each entry must name a target of the transition, the
    /// list must not be empty, and the transition must be `monitored`.
    ///
    /// Non-baseline construct: a document carrying it must declare
    /// [`Feature::MonitoredStates`]. An engine that ignored the field
    /// would record every branch, and the sequence and importance analyses
    /// would read a lost draw as an event without a word.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monitored_states: Option<Vec<String>>,
    /// **Sequence analysis**: cycle-pair group id: occ/rep (occ/not_occ)
    /// partner transitions of one failure mode share it, so the
    /// cycle-filtering step can drop transient failure→repair pairs that net
    /// out before the feared event. `None` = not part of a cycle pair.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cycle_group: Option<String>,
    /// **Sequence analysis**: whether the reduction may cancel this
    /// transition's [`Transition::cycle_group`]
    /// ([`CycleCancellation`]): `auto` (the default) lets the engine decide
    /// from the model, `never` and `always` override that decision. Every
    /// transition of one `(component, cycle_group)` declares the same
    /// value, and a transition with no `cycle_group` declares none.
    ///
    /// Non-baseline construct when not `auto`: a document carrying it must
    /// declare [`Feature::CycleCancellation`]. An engine that ignored the
    /// field would cancel a group the analyst declared `never`, which is
    /// exactly the silent collapse the declaration exists to prevent.
    #[serde(default, skip_serializing_if = "CycleCancellation::is_auto")]
    pub cycle_cancellation: CycleCancellation,
    /// **Declared reliability role** (see [`TransitionKind`]): `failure`,
    /// `repair`, or absent (the default, and the reading of every model
    /// written before the field existed). Applies to the first declared
    /// target only. The muscadet plugin sets it on the failure-mode edges
    /// it emits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<TransitionKind>,
    /// **Edge effects**: assignments evaluated ONCE, when the transition
    /// fires, after its state change, in declaration order. Unlike a
    /// propagation's effects (a level, re-evaluated whenever what it
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
    /// equation or a propagation writes is refused
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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

/// A propagation: declarative effects re-evaluated whenever one of
/// the attributes read by its expressions changes (the sensitivity set is
/// *derived* from the expressions: no trigger is registered by hand).
/// Effects run to a fixpoint during the discrete-evolution phase (paper rule `propagate_effects`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Propagation {
    /// Function name, unique inside its component.
    pub name: String,
    /// Ordered assignments applied when the function fires.
    pub effects: Vec<Assignment>,
}

/// What an indicator observes: a **typed, validated reference**
/// (rather than a stringly-typed `"comp.attr"` path).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Indicator {
    /// Indicator name (unique in the model; any length is accepted).
    pub name: String,
    /// Observed target.
    #[serde(flatten)]
    pub target: IndicatorTarget,
}

/// A component: typed state, ports/interfaces, automata and
/// propagations (the 7-tuple of Desgeorges et al., with the paper's
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
    /// Declarative propagations.
    #[serde(default, alias = "sensitive_functions")]
    pub propagations: Vec<Propagation>,
    /// Continuous evolution: explicit equations and ODEs (declaration
    /// order = solving order).
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
/// recorded sequence (the end cause a cod3s study reads). Empty `targets` = no early
/// stop and no `end_cause` (the trajectory runs to `t_max`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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

/// Objective direction of a model-level program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ProgramSense {
    /// Minimise the objective.
    Minimize,
    /// Maximise the objective.
    Maximize,
}

/// One decision attribute and its optional numeric bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ProgramTieObjective {
    /// Optimisation direction.
    pub sense: ProgramSense,
    /// Affine expression in the decisions.
    pub expr: Expr,
}

/// How equal primary optima are separated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ProgramTieBreak {
    /// Use the solver's arbitrary optimum without a uniqueness guarantee.
    None,
    /// Optimise the declared secondary objectives in order.
    Objectives(Vec<ProgramTieObjective>),
}

/// A mixed-integer program solved when its discrete inputs change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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
    /// **Source documents** of a model expanded from a component type
    /// library: the library and its version, the system, the study and the
    /// engine that supplied the standard library. Absent (the default) for
    /// a model written directly. Provenance only: it changes no result,
    /// needs no feature, and an absent field is not written, so a model
    /// written before it serializes, and hashes, as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ModelSource>,
}

/// The documents a model was expanded from (see [`Model::source`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSource {
    /// Library name.
    pub library: String,
    /// Library version, matched exactly by the system.
    pub library_version: String,
    /// System name.
    pub system: String,
    /// Study name, when a study was applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub study: Option<String>,
    /// The libraries shipped with the engine that the library reaches
    /// through its imports, each with its version (the engine's: an
    /// embedded fragment changes with the engine).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub embedded: BTreeMap<String, String>,
}
