//! The typed errors of model validation.

#[cfg(doc)]
use crate::IndicatorTarget;
#[cfg(doc)]
use crate::Transition;
#[cfg(doc)]
use crate::PORT;
use crate::{AttrKind, InterfaceRef, SingularReason};
use thiserror::Error;

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
    /// A time signal (`piecewise_linear`, `smooth_noise`) whose literal
    /// parameters do not describe one.
    #[error("{op} in {context}: {detail}")]
    InvalidTimeSignal {
        /// Where the expression appears.
        context: String,
        /// The offending operator.
        op: &'static str,
        /// What is wrong, naming the offending value.
        detail: String,
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
    /// A port equation writes a channel the producer's out port does not
    /// declare.
    #[error(
        "port equation of `{port}` writes channel `{channel}`, which \
         `{producer_port}` does not declare"
    )]
    PortEquationUnknownChannel {
        /// The port declaring the equation, as `component.port`.
        port: String,
        /// The channel written.
        channel: String,
        /// The producer's out port at the other end of a connection (the
        /// port itself for an out port), as `component.port`.
        producer_port: String,
    },
    /// A port equation reads `$edge.<channel>` for a channel the
    /// connection's producer does not declare.
    #[error(
        "port equation of `{port}` reads `$edge.{channel}`, which \
         `{producer_port}` does not declare"
    )]
    PortEquationUnknownEdgeChannel {
        /// The port declaring the equation, as `component.port`.
        port: String,
        /// The channel read.
        channel: String,
        /// The producer's out port at the other end of the connection, as
        /// `component.port`.
        producer_port: String,
    },
    /// An aggregation of an out port's channel is not a `sum` or a `count`.
    #[error(
        "{context} aggregates a channel of out port `{component}.{port}` with `{agg}`: an \
         out port aggregates one of its channels with `sum` or `count`"
    )]
    OutPortAggregation {
        /// Where the expression appears.
        context: String,
        /// The component.
        component: String,
        /// The out port.
        port: String,
        /// The aggregation written.
        agg: String,
    },
    /// An edge effect written through a port ([`PORT`]) names a port that
    /// is not an in port of the transition's component.
    #[error(
        "{transition} writes through port `{port}`, which is not an in port of its \
         component: an effect through a port names one of the component's in ports"
    )]
    PortEffectNotAnInPort {
        /// The transition, as `transition `name` in `component.automaton``.
        transition: String,
        /// The port named.
        port: String,
    },
    /// A propagation writes through a port ([`PORT`]).
    #[error(
        "propagation `{function}` writes through a port: only a transition's edge \
         effects may write through a port"
    )]
    PortEffectInPropagation {
        /// The function, as `component.function`.
        function: String,
    },
    /// A attribute may carry at most one equation.
    #[error("attribute `{component}.{target}` has more than one equation")]
    DuplicateEquation {
        /// The owning component.
        component: String,
        /// The doubly-defined attribute.
        target: String,
    },
    /// A boundary transition needs a guard (its boundary predicate).
    #[error(
        "boundary transition `{transition}` of `{component}.{automaton}` has no \
         guard (the guard is the boundary predicate)"
    )]
    BoundaryGuardMissing {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
    },
    /// A boundary guard needs a locatable continuous boundary: at least
    /// one ordering comparison, possibly composed with and/or/not and
    /// discrete gates (margins combine as min/max/negation).
    #[error(
        "boundary transition `{transition}` of `{component}.{automaton}`: the \
         guard must contain at least one ordering comparison (<, <=, >, >=), \
         possibly composed with and/or/not and discrete gates"
    )]
    BoundaryGuardNotComparison {
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
    /// A truncated normal law whose interval the law gives no probability
    /// to: there is nothing to renormalise and no delay to draw.
    #[error(
        "normal transition `{transition}` of `{component}.{automaton}`: the law \
         N({mu}, {sigma}²) gives no probability to the interval from {low} to {high}, \
         so it cannot be truncated to it"
    )]
    NormalIntervalWithoutMass {
        /// The owning component.
        component: String,
        /// The automaton.
        automaton: String,
        /// The transition.
        transition: String,
        /// Mean of the untruncated law.
        mu: f64,
        /// Standard deviation of the untruncated law.
        sigma: f64,
        /// Lower bound.
        low: f64,
        /// Upper bound (`inf` when unbounded).
        high: f64,
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
         rate {rate} (must be finite and >= 0)"
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
    /// A boundary transition must have exactly one target.
    #[error(
        "boundary transition `{transition}` of `{component}.{automaton}` must \
         have exactly one target (got {targets})"
    )]
    BoundaryTargetCount {
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
    /// equation or a propagation also writes: the next evaluation
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
    /// A transition's [`Transition::monitored_states`] is not a non-empty
    /// subset of its targets on a monitored transition.
    #[error("transition `{transition}` declares `monitored_states`, but {reason}")]
    MonitoredStatesInvalid {
        /// `component.automaton.transition`.
        transition: String,
        /// What is wrong with the declaration.
        reason: String,
    },
    /// A transition's [`Transition::cycle_cancellation`] is declared where
    /// it cannot apply, or disagrees with another transition of its group.
    #[error("transition `{transition}` declares `cycle_cancellation`, but {reason}")]
    CycleCancellationInvalid {
        /// `component.automaton.transition`.
        transition: String,
        /// What is wrong with the declaration.
        reason: String,
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
        /// The other writer (an equation, a propagation or
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
