//! Name→index resolution: turns a validated [`Model`] into dense tables
//! the engine consumes.
//!
//! Performance contract: all names are resolved here, once, at
//! build time: the simulation hot path only touches vector indices,
//! never string lookups, and never allocates.

use crate::flow::CPolicy;
use raichu_expr::{AggOp, AttrRef, BoolOp, CmpOp, Expr, PortRef, Value};
use raichu_model::{
    AlgebraicBlock, Allocation, AllocationPolicy, Distrib, EquationKind, IndicatorTarget,
    InterruptionPolicy, Model, ModelError, PortDir, ProgramSense, ProgramTieBreak, SingularReason,
    TransitionKind,
};
use raichu_numeric::LuFactorization;
use std::collections::{BTreeSet, HashMap, HashSet};
use thiserror::Error;

/// Margin tightening applied to *strict* watched comparisons (`<`,
/// `>`): the engine fires at margin ≥ 0, so a strict boundary is
/// shifted inward by this amount: a trajectory resting exactly on it
/// does not fire, and a genuine crossing date shifts by ε/slope. Must
/// sit *below* the event-location tolerance (`tol_event` = 1e-10):
/// when two watched guards share one crossing, the located state
/// overshoots by ~tol_event and the sibling must still read a
/// non-negative margin to fire immediately.
pub const STRICT_MARGIN_EPS: f64 = 1e-12;

/// Dense index of an attribute in the engine state vector.
pub type VarIdx = usize;
/// Dense index of an automaton.
pub type AutIdx = usize;
/// Index of a state *within its automaton*.
pub type StateIdx = usize;
/// Dense index of a transition (global).
pub type TransIdx = usize;
/// Dense index of a sensitive function (global).
pub type FnIdx = usize;
/// Position of a watched transition *within* [`CompiledModel::watched`],
/// which is the order the engine scans the watched population in.
pub type WatchedIdx = usize;

/// Errors raised while compiling a model.
#[derive(Debug, Error)]
pub enum CompileError {
    /// The model failed its structural validation.
    #[error(transparent)]
    Invalid(#[from] ModelError),
    /// Internal resolution failure: indicates a validator/compiler
    /// mismatch, reported as a typed error rather than a panic.
    #[error("internal resolution failure: {what} `{name}` not found")]
    Unresolved {
        /// Kind of entity that failed to resolve.
        what: &'static str,
        /// Qualified name.
        name: String,
    },
}

/// A compiled expression: same shape as [`Expr`] but with every
/// reference resolved to dense indices.
#[derive(Debug, Clone)]
pub enum CExpr {
    /// Literal constant.
    Const(Value),
    /// Read the attribute at this index.
    Var(VarIdx),
    /// True while `automaton` is in `state`.
    StateActive {
        /// Automaton index.
        automaton: AutIdx,
        /// State index within the automaton.
        state: StateIdx,
    },
    /// Aggregate the out-attributes connected to an in-port.
    PortAgg {
        /// Indices of the connected out-port attributes (connection
        /// declaration order: deterministic).
        sources: Vec<VarIdx>,
        /// Aggregation operator.
        agg: AggOp,
    },
    /// Comparison.
    Cmp {
        /// Operator.
        op: CmpOp,
        /// Left operand.
        lhs: Box<CExpr>,
        /// Right operand.
        rhs: Box<CExpr>,
    },
    /// Boolean connective.
    Bool {
        /// Operator.
        op: BoolOp,
        /// Operands.
        args: Vec<CExpr>,
    },
    /// N-ary sum.
    Add {
        /// Operands.
        args: Vec<CExpr>,
    },
    /// Binary subtraction.
    Sub {
        /// Minuend.
        lhs: Box<CExpr>,
        /// Subtrahend.
        rhs: Box<CExpr>,
    },
    /// N-ary product.
    Mul {
        /// Operands.
        args: Vec<CExpr>,
    },
    /// Binary division (float semantics).
    Div {
        /// Dividend.
        lhs: Box<CExpr>,
        /// Divisor.
        rhs: Box<CExpr>,
    },
    /// N-ary minimum.
    Min {
        /// Operands.
        args: Vec<CExpr>,
    },
    /// N-ary maximum.
    Max {
        /// Operands.
        args: Vec<CExpr>,
    },
    /// Conditional.
    If {
        /// Boolean condition.
        cond: Box<CExpr>,
        /// Value when true.
        then: Box<CExpr>,
        /// Value when false.
        otherwise: Box<CExpr>,
    },
    /// Sine.
    Sin(Box<CExpr>),
    /// Natural exponential.
    Exp(Box<CExpr>),
    /// Current simulation time.
    Time,
}

/// Compiled occurrence distribution.
#[derive(Debug, Clone)]
pub enum CLaw {
    /// Deterministic delay.
    Delay(f64),
    /// Instantaneous branching; probabilities include the reconstructed
    /// complement (length == number of targets).
    Inst(Vec<f64>),
    /// Watched transition (M1): fires when `margin` crosses from
    /// negative to non-negative during continuous evolution (`schedule_boundary`).
    /// The margin is the signed boundary distance derived from the
    /// guard comparison: guard true ⇔ margin ≥ 0.
    Watched {
        /// Signed boundary margin.
        margin: CExpr,
    },
    /// Exponential distribution (M2, `schedule_stochastic`): firing date sampled at
    /// source-state entry.
    Exp(f64),
    /// Exponential distribution with a state-dependent rate λ(x) (`reschedule_modifiable`):
    /// realised by a cumulative hazard against an `Exp(1)` threshold
    /// (`P(T > t) = exp(−∫λ)`: the PDMP survival function, exactly).
    ExpVar {
        /// Rate expression λ(x) ≥ 0.
        rate: CExpr,
        /// Whether λ varies during continuous evolution (depends,
        /// transitively through explicit equations, on an
        /// ODE-integrated attribute or on time). If so the hazard is
        /// integrated alongside the continuous state and the firing
        /// time located like a boundary crossing; otherwise λ is
        /// piecewise-constant and the firing date is rescheduled at
        /// each discrete change (`reschedule_modifiable` proper).
        continuous: bool,
    },
    /// Weibull distribution (M4): shape k, scale λ.
    Weibull(f64, f64),
    /// Log-normal distribution (M4): μ, σ of the underlying normal.
    Lognormal(f64, f64),
    /// Gamma distribution (M4): shape k, scale θ.
    Gamma(f64, f64),
    /// Uniform distribution (M4): [low, high).
    Uniform(f64, f64),
    /// Empirical inverse-CDF table (M4): (time, cumulative prob).
    Empirical(Vec<(f64, f64)>),
}

/// A compiled transition.
#[derive(Debug, Clone)]
pub struct CTransition {
    /// Qualified name `component.automaton.transition` (journal only).
    pub name: String,
    /// Owning component name (the `obj` of a recorded `SeqEvent`).
    pub component: String,
    /// Owning automaton.
    pub automaton: AutIdx,
    /// Source state.
    pub source: StateIdx,
    /// Guard (absent = always true).
    pub guard: Option<CExpr>,
    /// Target states (one per branch).
    pub targets: Vec<StateIdx>,
    /// What happens to a pending countdown when the guard turns false
    /// (paper rule `drop_disabled`).
    pub on_interruption: InterruptionPolicy,
    /// Firing this transition records a `SeqEvent` (sequence analysis).
    pub monitored: bool,
    /// Cycle-pair group id (occ/rep partners share it; sequence analysis).
    pub cycle_group: Option<String>,
    /// Declared reliability role, if any
    /// ([`raichu_model::Transition::kind`]): it applies to the first entry
    /// of [`CTransition::targets`] only, the declared order being kept.
    pub kind: Option<TransitionKind>,
    /// Edge effects `target := value`, applied once when the transition
    /// fires, after its state change ([`raichu_model::Transition::effects`]).
    pub effects: Vec<(VarIdx, CExpr)>,
    /// Occurrence distribution.
    pub distrib: CLaw,
}

/// A compiled sequence-analysis target (feared event).
#[derive(Debug, Clone)]
pub struct CTarget {
    /// Target name (the `end_cause` label).
    pub name: String,
    /// The automaton whose state activation reaches the target.
    pub automaton: AutIdx,
    /// The state whose activation reaches the target.
    pub state: StateIdx,
}

/// A compiled automaton.
#[derive(Debug, Clone)]
pub struct CAutomaton {
    /// Qualified name `component.automaton`.
    pub name: String,
    /// State names, indexed by [`StateIdx`].
    pub states: Vec<String>,
    /// Initial state.
    pub init: StateIdx,
    /// Indices of the transitions owned by this automaton.
    pub transitions: Vec<TransIdx>,
}

/// A compiled **conservative distribution operator**: the available
/// quantity, the per-connection demands it reads, the per-connection
/// quantities it writes, and the policy that splits one among the other.
///
/// The three vectors share one order, the connection declaration order of
/// the operator's out port, which is the order [`CPolicy::Priority`]
/// breaks its ties by.
#[derive(Debug, Clone)]
pub struct CAllocation {
    /// Qualified name `component.allocation` (journal, diagnostics).
    pub name: String,
    /// The quantity available for distribution.
    pub available: CExpr,
    /// Attributes carrying each consumer's demand (read).
    pub demands: Vec<VarIdx>,
    /// Attributes receiving each consumer's share (written).
    pub allocated: Vec<VarIdx>,
    /// The split policy, resolved to the same order.
    pub policy: CPolicy,
}

/// The **active-set margins** of one conservative distribution operator:
/// the compiled watched guards that tell the engine when the frozen
/// saturation pattern of that operator is about to change.
///
/// One margin per outgoing connection, evaluated inside the solver
/// callbacks exactly like the margin of a watched transition. What each
/// margin *measures* depends on the class the edge currently sits in, so
/// the expression is not fixed at compile time; what is fixed here is the
/// operator it belongs to and the attributes it reads.
///
/// The per-edge margins of one operator share **one** dependency set:
/// the quantity offered to a consumer under a weighted split is a
/// function of the available quantity and of *every* demand, so narrowing
/// the registration per edge would claim an independence that does not
/// exist. The registration is therefore per operator, which is the
/// granularity a variable-to-margin index can honestly index.
///
/// Minima are deliberately **not** given a margin. A limiting reagent
/// written as a minimum keeps a kink rather than a jump, so the
/// integrator handles it without help, and one watched guard per input
/// pair would add a quadratic population for accuracy the kink already
/// provides. The branch a minimum takes still enters the *termination
/// test* of the resolution, where it is free.
#[derive(Debug, Clone)]
pub struct CFlowMargins {
    /// Index of the operator's step in [`CompiledModel::explicit`].
    pub step: usize,
    /// Qualified operator name `component.allocation`.
    pub name: String,
    /// Qualified name of the attribute each edge is allocated, in
    /// connection declaration order (diagnostics and causal journal).
    pub consumers: Vec<String>,
    /// Attributes every margin of this operator reads: the available
    /// quantity's own reads plus the demand of each edge. Sorted and
    /// deduplicated, so an index inverting it iterates a stable sequence.
    pub deps: Vec<VarIdx>,
}

/// The **inverted dependency index of the margins**: which of them a
/// change to a given attribute, automaton state or to the clock can move.
///
/// It is the same inversion [`CompiledModel::var_triggers`] performs for
/// the sensitive functions, applied to the watched population instead:
/// the compiler already knows what every guard reads, so the engine can
/// be told which guards a change reaches rather than re-deriving it by
/// scanning all of them. Every list is **ascending and deduplicated**, so
/// an engine walking one visits the watched transitions in the order it
/// would have scanned them, and a run replays identically. That
/// ordering is a requirement, not a convenience: a hash-set iteration
/// would reorder simultaneous firings from one process to the next,
/// because the default hasher is seeded per process.
///
/// **What it narrows and what it does not.** It narrows the two sites
/// that scan the *whole* watched population: the immediate-guard check
/// after every discrete fixpoint, and the active-margin set the engine
/// hands the solver at every integration segment. It does **not** narrow
/// the event evaluation inside the solver callbacks: that one is already
/// bounded by the margins of the segment, evaluated at every scan point
/// and every bisection step because their values are what the root
/// finder brackets.
#[derive(Debug, Clone, Default)]
pub struct MarginIndex {
    /// attribute index → positions in [`CompiledModel::watched`] whose
    /// guard (hence whose margin, compiled from that same guard) reads
    /// it.
    pub watched_by_var: Vec<Vec<WatchedIdx>>,
    /// automaton index → positions in [`CompiledModel::watched`] whose
    /// guard reads that automaton's current state.
    pub watched_by_state: Vec<Vec<WatchedIdx>>,
    /// Positions in [`CompiledModel::watched`] whose guard reads the
    /// clock, and which therefore move whenever time does. Usually
    /// empty: a boundary is normally a predicate on the continuous
    /// state.
    pub watched_by_time: Vec<WatchedIdx>,
    /// automaton index → positions in [`CompiledModel::watched`] that
    /// automaton **owns**. This is the dependency of the *arming*
    /// filter (a watched transition is monitored only while its
    /// automaton sits in its source state), which is what lets the
    /// per-segment margin set be maintained instead of rebuilt.
    pub watched_by_owner: Vec<Vec<WatchedIdx>>,
    /// attribute index → indices into [`CompiledModel::flow_margins`]
    /// whose margins read it, inverting [`CFlowMargins::deps`].
    ///
    /// Registered so the index is a *complete* answer to "which margins
    /// read this attribute" for every margin family the engine carries.
    /// The active-set margins of the distribution operators have no
    /// arming filter to narrow (every operator of the sweep contributes
    /// its edges to every segment) and their evaluation happens inside
    /// the solver, which this index deliberately leaves alone: they are
    /// therefore indexed and not consumed by a scan site.
    pub flow_by_var: Vec<Vec<usize>>,
}

/// A compiled **algebraic block**: the member equations of one linear
/// cycle, solved simultaneously.
///
/// Row `i` is the i-th member equation arranged as
/// `targets[i] = constants[i] + Σ coefficients[i][k] · targets[column k]`,
/// so the block is the system `(I − A)·x = c` with `A` the coefficient
/// matrix. The engine evaluates the (unknown-free) constants and
/// coefficients, factorises `I − A` when a coefficient input moved, and
/// solves: one step of the sweep, wherever the sweep runs.
#[derive(Debug, Clone)]
pub struct CBlock {
    /// Qualified block name, `component.attribute` of the first member
    /// (journal, diagnostics).
    pub name: String,
    /// Member targets in the block's unknown order (one per row).
    pub targets: Vec<VarIdx>,
    /// Qualified member names, same order (diagnostics, errors).
    pub members: Vec<String>,
    /// Unknown-free part of each right-hand side, one per row.
    pub constants: Vec<CExpr>,
    /// Per row, the sparse non-zero coefficients of `A`:
    /// `(column, coefficient)` with `column` an index into `targets`,
    /// ascending in column. Coefficients are unknown-free by
    /// construction.
    pub coefficients: Vec<Vec<(usize, CExpr)>>,
    /// Immutable LU of I-A when all coefficients are constant, validated
    /// during compilation and shared by every trajectory and solver stage.
    pub constant_factorization: Option<LuFactorization>,
}

/// One step of the **explicit sweep**, run at every evaluation point in
/// table order.
///
/// An equation writes one attribute; a distribution operator writes one
/// per outgoing connection, which is why it is a step of its own rather
/// than a sensitive-function effect. Both live in one table because both
/// must run inside the solver callbacks: a quantity written outside that
/// table stays frozen through an integration segment, and a watched
/// margin reading it would be polled rather than located. A block
/// writes all of its members at once (it is one step, not `n`).
#[derive(Debug, Clone)]
pub enum CStep {
    /// Explicit assignment `target = expr`.
    Equation {
        /// The attribute receiving the value.
        target: VarIdx,
        /// The right-hand side.
        expr: CExpr,
    },
    /// Conservative distribution of one quantity over the connections of
    /// an out port.
    Allocate(CAllocation),
    /// A linear algebraic block, solved simultaneously (see [`CBlock`]).
    Block(CBlock),
}

/// A compiled sensitive function.
#[derive(Debug, Clone)]
pub struct CFunction {
    /// Qualified name `component.function`.
    pub name: String,
    /// Ordered effects `target := value`.
    pub effects: Vec<(VarIdx, CExpr)>,
}

/// Compiled affine expression over one program's decision columns.
#[derive(Debug, Clone)]
pub struct CProgramAffine {
    /// Decision-free offset.
    pub constant: CExpr,
    /// Sparse terms `(column index, coefficient expression)`.
    pub terms: Vec<(usize, CExpr)>,
}

impl CProgramAffine {
    fn collect_sensitivity(&self, vars: &mut Vec<VarIdx>, auts: &mut Vec<AutIdx>) {
        self.constant.collect_sensitivity(vars, auts);
        for (_, coefficient) in &self.terms {
            coefficient.collect_sensitivity(vars, auts);
        }
    }
}

/// One compiled decision attribute.
#[derive(Debug, Clone)]
pub struct CProgramVariable {
    /// Published attribute.
    pub target: VarIdx,
    /// Declared kind.
    pub kind: raichu_model::AttrKind,
    /// Lower and upper numeric expressions.
    pub lower: Option<CExpr>,
    /// Upper numeric expression.
    pub upper: Option<CExpr>,
    /// Infeasible fallback.
    pub fallback: Value,
}

/// One compiled ranged constraint.
#[derive(Debug, Clone)]
pub struct CProgramConstraint {
    /// Affine left-hand side.
    pub expr: CProgramAffine,
    /// Optional lower bound.
    pub lower: Option<CExpr>,
    /// Optional upper bound.
    pub upper: Option<CExpr>,
}

/// A discrete fixpoint program with all references resolved.
#[derive(Debug, Clone)]
pub struct CProgram {
    /// Program name.
    pub name: String,
    /// Decision columns.
    pub variables: Vec<CProgramVariable>,
    /// Primary direction.
    pub sense: raichu_milp::Sense,
    /// Primary objective.
    pub objective: CProgramAffine,
    /// Constraints.
    pub constraints: Vec<CProgramConstraint>,
    /// Feasibility output.
    pub feasible: VarIdx,
    /// Optional objective-value output.
    pub objective_value: Option<VarIdx>,
    /// Lexicographic rule.
    pub tie_break: CProgramTieBreak,
    /// Node cap.
    pub node_limit: Option<u64>,
}

/// Compiled secondary-objective rule.
#[derive(Debug, Clone)]
pub enum CProgramTieBreak {
    /// Default decision-order tie-break.
    Default,
    /// No tie-break.
    None,
    /// Secondary objectives, followed by decision-order tie-break.
    Objectives(Vec<(raichu_milp::Sense, CProgramAffine)>),
}

fn program_sense(sense: ProgramSense) -> raichu_milp::Sense {
    match sense {
        ProgramSense::Minimize => raichu_milp::Sense::Minimize,
        ProgramSense::Maximize => raichu_milp::Sense::Maximize,
    }
}

/// A compiled indicator.
#[derive(Debug, Clone)]
pub struct CIndicator {
    /// Indicator name.
    pub name: String,
    /// What it observes.
    pub target: CIndicatorTarget,
}

/// Compiled indicator target.
#[derive(Debug, Clone)]
pub enum CIndicatorTarget {
    /// A attribute's value.
    Var(VarIdx),
    /// 1.0 while the automaton is in the state, else 0.0.
    State(AutIdx, StateIdx),
    /// `true` while the attribute satisfies the threshold, else `false`.
    /// Kind compatibility is settled by model validation, so evaluating
    /// this cannot fail.
    Predicate(VarIdx, CmpOp, Value),
}

/// A validated model resolved to dense tables.
#[derive(Debug, Clone)]
pub struct CompiledModel {
    /// Unique identity for thread-local caches shared by replicas of this
    /// compiled model, including clones.
    pub(crate) cache_id: u64,
    /// Model name (provenance).
    pub name: String,
    /// Imported co-simulation units, separate from authored transitions.
    pub fmu_units: Vec<raichu_model::FmuUnit>,
    /// Qualified attribute names `component.attribute` (journal, results).
    pub var_names: Vec<String>,
    /// Initial attribute values.
    pub var_init: Vec<Value>,
    /// Automata.
    pub automata: Vec<CAutomaton>,
    /// All transitions (global order = declaration order).
    pub transitions: Vec<CTransition>,
    /// All sensitive functions (global order = declaration order: this
    /// *is* the documented deterministic fixpoint order).
    pub functions: Vec<CFunction>,
    /// Model-level programs, executed after sensitive functions in the
    /// same discrete fixpoint worklist.
    pub programs: Vec<CProgram>,
    /// var index → functions to re-evaluate when it changes.
    pub var_triggers: Vec<Vec<FnIdx>>,
    /// automaton index → functions to re-evaluate when its state changes.
    pub state_triggers: Vec<Vec<FnIdx>>,
    /// Indicators.
    pub indicators: Vec<CIndicator>,
    /// Sequence-analysis targets (feared events), resolved to indices.
    pub targets: Vec<CTarget>,
    /// ODE attributes and right-hand sides, declaration order (CEvol).
    pub ode: Vec<(VarIdx, CExpr)>,
    /// The model's declared unbounded magnitude
    /// ([`raichu_model::Model::unbounded_rate`]): no ODE right-hand side
    /// may reach it (`evolC` refuses with
    /// [`crate::EngineError::UnboundedRate`]). `None`: nothing reserved.
    pub unbounded_rate: Option<f64>,
    /// The explicit sweep: equations and distribution operators in
    /// evaluation order (run before ODE right-hand sides at every
    /// evaluation point). Positional unless the model declares an
    /// `evaluation_order`.
    pub explicit: Vec<CStep>,
    /// Whether any step of the explicit sweep reads the simulation
    /// clock.
    ///
    /// Such a step varies continuously with **no ODE attribute behind
    /// it**: a declared time profile is one, and nothing else in the
    /// model need move for it to. The engine reads this to decide that
    /// continuous evolution must run at all, so that the sweep is
    /// re-evaluated as the clock advances rather than once at the
    /// initial instant and never again.
    pub explicit_reads_time: bool,
    /// Whether some step of the explicit sweep reads an attribute that the
    /// same step or a LATER one writes: a ring the evaluation order had to
    /// tear somewhere.
    ///
    /// One pass of such a sweep reads, at the tear, the value the previous
    /// evaluation left behind, so its result depends on what was evaluated
    /// before rather than on the state alone. The flow resolution iterates
    /// to the fixpoint and does not care; the solver's right-hand side,
    /// which runs one pass per stage, does: it becomes a function of the
    /// evaluation history, including rejected trial steps, and an active-set
    /// margin sitting on an exact balance crosses its band on that noise
    /// and restarts the segment over and over. The engine reads this flag
    /// to repeat the pass inside the right-hand side until it settles
    /// (`evolC`); a sweep without a tear keeps its single pass, bit for bit.
    pub(crate) sweep_reads_ahead: bool,
    /// Switching loops found in this model: an automaton whose guard
    /// reads a quantity its own decision moves, where some automaton on
    /// the cycle switches on a single threshold.
    ///
    /// Computed once here rather than on demand, so a caller that wants
    /// the diagnosis pays nothing for it and one that does not is warned
    /// anyway (see [`crate::loops`]). Never a refusal: the loop is
    /// legitimate, the missing band is what is not.
    pub switching_loops: Vec<crate::loops::SwitchingLoop>,
    /// Unfed triggers found in this model: a mode sealed for the whole
    /// run by an in port no connection reaches.
    ///
    /// Carried and computed on the same terms as `switching_loops`, and
    /// kept apart from it because the two answer different questions:
    /// a loop is a mode with no fixpoint, this is a mode with no choice
    /// (see [`crate::triggers`]). Never a refusal either: the model is
    /// valid and runs, it is only almost always an oversight.
    pub unfed_triggers: Vec<crate::triggers::UnfedTrigger>,
    /// Indices of watched transitions (monitored during continuous
    /// evolution, never date-scheduled).
    pub watched: Vec<TransIdx>,
    /// Active-set margins, one entry per conservative distribution
    /// operator of the explicit sweep, in sweep order. Empty for a model
    /// carrying no operator, which is what lets such a model skip the
    /// flow resolution entirely and keep the counted-work profile it had
    /// before the resolution existed.
    pub flow_margins: Vec<CFlowMargins>,
    /// Inverted dependency index of the margins: which of them a change
    /// reaches. See [`MarginIndex`].
    pub margin_index: MarginIndex,
    /// Lookup: qualified attribute name → index (API convenience).
    pub var_index: HashMap<String, VarIdx>,
    /// Lookup: qualified automaton name → index (API convenience).
    pub automaton_index: HashMap<String, AutIdx>,
}

struct Resolver {
    vars: HashMap<(String, String), VarIdx>,
    states: HashMap<(String, String, String), (AutIdx, StateIdx)>,
    automata: HashMap<(String, String), AutIdx>,
    /// (component, in-port) → connected source attribute indices.
    port_sources: HashMap<(String, String), Vec<VarIdx>>,
    /// (component, in-port, channel) → the *per-connection* attributes
    /// materialised for that channel, one per incoming connection, in the
    /// same connection declaration order as `port_sources`.
    port_channel_sources: HashMap<(String, String, String), Vec<VarIdx>>,
}

impl Resolver {
    fn compile_program_affine(
        &self,
        expr: &Expr,
        decisions: &HashSet<AttrRef>,
        columns: &HashMap<AttrRef, usize>,
    ) -> Result<CProgramAffine, CompileError> {
        let form = expr
            .affine_in(decisions)
            .map_err(|error| CompileError::Unresolved {
                what: "affine program expression",
                name: format!("{:?}", error.expression),
            })?;
        let constant = self.compile_expr(&form.constant)?;
        let terms = form
            .terms
            .into_iter()
            .map(|(attr, coefficient)| {
                let column =
                    columns
                        .get(&attr)
                        .copied()
                        .ok_or_else(|| CompileError::Unresolved {
                            what: "program decision",
                            name: format!("{}.{}", attr.component, attr.attribute),
                        })?;
                Ok((column, self.compile_expr(&coefficient)?))
            })
            .collect::<Result<Vec<_>, CompileError>>()?;
        Ok(CProgramAffine { constant, terms })
    }

    fn var(&self, component: &str, attribute: &str) -> Result<VarIdx, CompileError> {
        self.vars
            .get(&(component.to_owned(), attribute.to_owned()))
            .copied()
            .ok_or_else(|| CompileError::Unresolved {
                what: "attribute",
                name: format!("{component}.{attribute}"),
            })
    }

    fn state(
        &self,
        component: &str,
        automaton: &str,
        state: &str,
    ) -> Result<(AutIdx, StateIdx), CompileError> {
        self.states
            .get(&(component.to_owned(), automaton.to_owned(), state.to_owned()))
            .copied()
            .ok_or_else(|| CompileError::Unresolved {
                what: "state",
                name: format!("{component}.{automaton}.{state}"),
            })
    }

    fn port(&self, component: &str, port: &str) -> Vec<VarIdx> {
        // An in-port with no connection aggregates over the empty set:
        // legal (muscadet relies on no-connection defaults).
        self.port_sources
            .get(&(component.to_owned(), port.to_owned()))
            .cloned()
            .unwrap_or_default()
    }

    /// Sources of an in-port aggregation that names a channel: the
    /// per-connection attributes materialised for it. Same no-connection
    /// default as [`Resolver::port`], same declaration order.
    fn port_channel(&self, component: &str, port: &str, channel: &str) -> Vec<VarIdx> {
        self.port_channel_sources
            .get(&(component.to_owned(), port.to_owned(), channel.to_owned()))
            .cloned()
            .unwrap_or_default()
    }

    fn compile_expr(&self, expr: &Expr) -> Result<CExpr, CompileError> {
        Ok(match expr {
            Expr::Const { value } => CExpr::Const(*value),
            Expr::Attr { attr } => CExpr::Var(self.var(&attr.component, &attr.attribute)?),
            Expr::StateActive { state } => {
                let (automaton, state) =
                    self.state(&state.component, &state.automaton, &state.state)?;
                CExpr::StateActive { automaton, state }
            }
            Expr::PortAgg { port, agg, channel } => CExpr::PortAgg {
                sources: match channel {
                    // No channel named: the producer's single exported
                    // attribute, exactly as before this affordance.
                    None => self.port(&port.component, &port.port),
                    Some(channel) => self.port_channel(&port.component, &port.port, channel),
                },
                agg: *agg,
            },
            Expr::Cmp { cmp, lhs, rhs } => CExpr::Cmp {
                op: *cmp,
                lhs: Box::new(self.compile_expr(lhs)?),
                rhs: Box::new(self.compile_expr(rhs)?),
            },
            Expr::Bool { bool_op, args } => CExpr::Bool {
                op: *bool_op,
                args: self.compile_args(args)?,
            },
            Expr::Add { args } => CExpr::Add {
                args: self.compile_args(args)?,
            },
            Expr::Mul { args } => CExpr::Mul {
                args: self.compile_args(args)?,
            },
            Expr::Min { args } => CExpr::Min {
                args: self.compile_args(args)?,
            },
            Expr::Max { args } => CExpr::Max {
                args: self.compile_args(args)?,
            },
            Expr::Sub { lhs, rhs } => CExpr::Sub {
                lhs: Box::new(self.compile_expr(lhs)?),
                rhs: Box::new(self.compile_expr(rhs)?),
            },
            Expr::Div { lhs, rhs } => CExpr::Div {
                lhs: Box::new(self.compile_expr(lhs)?),
                rhs: Box::new(self.compile_expr(rhs)?),
            },
            Expr::If {
                cond,
                then,
                otherwise,
            } => CExpr::If {
                cond: Box::new(self.compile_expr(cond)?),
                then: Box::new(self.compile_expr(then)?),
                otherwise: Box::new(self.compile_expr(otherwise)?),
            },
            Expr::Sin { arg } => CExpr::Sin(Box::new(self.compile_expr(arg)?)),
            Expr::Exp { arg } => CExpr::Exp(Box::new(self.compile_expr(arg)?)),
            Expr::Time => CExpr::Time,
        })
    }

    fn compile_args(&self, args: &[Expr]) -> Result<Vec<CExpr>, CompileError> {
        args.iter().map(|a| self.compile_expr(a)).collect()
    }

    /// Signed boundary margin of a watched guard (guard true ⇔ margin
    /// ≥ 0). Validation guarantees one of three shapes:
    ///
    /// - a single ordering comparison → `lhs − rhs` (or reversed);
    /// - `and(gates…, cmp)` → `if(and(gates), margin(cmp), −1)`;
    /// - `or(gates…, cmp)`  → `if(or(gates), +1, margin(cmp))`.
    ///
    /// The gate expressions are *discrete* (they only change at
    /// discrete events), so the margin stays continuous within every
    /// integration segment; a gate flip is caught by the
    /// immediate-watched check right after the discrete fixpoint.
    fn compile_watched_margin(&self, guard: &Expr) -> Result<CExpr, CompileError> {
        match guard {
            Expr::Cmp {
                cmp: cmp @ (CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge),
                lhs,
                rhs,
            } => {
                let left = self.compile_expr(lhs)?;
                let right = self.compile_expr(rhs)?;
                let raw = match cmp {
                    CmpOp::Ge | CmpOp::Gt => CExpr::Sub {
                        lhs: Box::new(left),
                        rhs: Box::new(right),
                    },
                    _ => CExpr::Sub {
                        lhs: Box::new(right),
                        rhs: Box::new(left),
                    },
                };
                // The engine fires at margin ≥ 0. For *strict*
                // comparisons a trajectory resting exactly on the
                // boundary (e.g. a ternary signal at 0 against a
                // `< 0` guard) must NOT fire: tighten the margin by
                // STRICT_MARGIN_EPS. The induced crossing-date shift
                // (ε / slope) sits below the documented
                // event tolerances (~1e-9).
                Ok(match cmp {
                    CmpOp::Gt | CmpOp::Lt => CExpr::Sub {
                        lhs: Box::new(raw),
                        rhs: Box::new(CExpr::Const(Value::Float(STRICT_MARGIN_EPS))),
                    },
                    _ => raw,
                })
            }
            // AND: every boundary must hold, the binding one is the
            // *minimum* margin. OR: any suffices, the maximum.
            Expr::Bool {
                bool_op: bool_op @ (BoolOp::And | BoolOp::Or),
                args,
            } => {
                let margins = args
                    .iter()
                    .map(|arg| self.compile_watched_margin(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(match bool_op {
                    BoolOp::And => CExpr::Min { args: margins },
                    _ => CExpr::Max { args: margins },
                })
            }
            // NOT: the guard flips exactly where the margin changes
            // sign: negate it.
            Expr::Bool {
                bool_op: BoolOp::Not,
                args,
            } if args.len() == 1 => {
                let inner = self.compile_watched_margin(&args[0])?;
                Ok(CExpr::Sub {
                    lhs: Box::new(CExpr::Const(Value::Float(0.0))),
                    rhs: Box::new(inner),
                })
            }
            // Any other boolean operand is a *discrete gate*: constant
            // between discrete events, mapped to ±1 so it composes
            // through min/max without hiding the continuous boundary.
            other => {
                let gate = self.compile_expr(other)?;
                Ok(CExpr::If {
                    cond: Box::new(gate),
                    then: Box::new(CExpr::Const(Value::Float(1.0))),
                    otherwise: Box::new(CExpr::Const(Value::Float(-1.0))),
                })
            }
        }
    }
}

/// Whether a step of `explicit` reads an attribute written by itself or by
/// a later step (see [`CompiledModel::sweep_reads_ahead`]).
fn sweep_reads_ahead(explicit: &[CStep]) -> bool {
    // The last step writing each attribute: a read is ahead of its writer
    // when that writer sits at or after the reading step.
    let mut last_writer: HashMap<VarIdx, usize> = HashMap::new();
    for (index, step) in explicit.iter().enumerate() {
        match step {
            CStep::Equation { target, .. } => {
                last_writer.insert(*target, index);
            }
            CStep::Allocate(allocation) => {
                for &var in &allocation.allocated {
                    last_writer.insert(var, index);
                }
            }
            // The block is one step writing all of its members.
            CStep::Block(block) => {
                for &var in &block.targets {
                    last_writer.insert(var, index);
                }
            }
        }
    }
    let (mut vars, mut auts) = (Vec::new(), Vec::new());
    explicit.iter().enumerate().any(|(index, step)| {
        vars.clear();
        match step {
            CStep::Equation { expr, .. } => expr.collect_sensitivity(&mut vars, &mut auts),
            CStep::Allocate(allocation) => {
                allocation
                    .available
                    .collect_sensitivity(&mut vars, &mut auts);
                vars.extend_from_slice(&allocation.demands);
            }
            // A block reads what its constants and coefficients read;
            // its members appear only as solved columns.
            CStep::Block(block) => {
                for expr in &block.constants {
                    expr.collect_sensitivity(&mut vars, &mut auts);
                }
                for row in &block.coefficients {
                    for (_, coefficient) in row {
                        coefficient.collect_sensitivity(&mut vars, &mut auts);
                    }
                }
            }
        }
        vars.iter()
            .any(|var| last_writer.get(var).is_some_and(|&writer| writer >= index))
    })
}

/// One node of the sweep-ordering graph: a surviving step (an equation
/// or a distribution operator) or a whole algebraic block. `position`
/// is the index the node inherits from the table the declared order
/// left (a block inherits the earliest position its members held) and
/// doubles as the tie-break key, so a step the dependencies leave free
/// stays where the model put it.
struct OrderNode {
    position: usize,
    /// Attributes whose writers this node waits for.
    reads: Vec<VarIdx>,
    /// Attributes this node writes and other steps may wait for. Empty
    /// for an operator: an allocated quantity sets no ordering
    /// constraint on its readers (the carve-out of the classification,
    /// mirrored here).
    registered_writes: Vec<VarIdx>,
    step: CStep,
}

/// Form the classified algebraic blocks into single steps and reorder
/// the sweep around them.
///
/// With no block the table comes back untouched: a model without a
/// cycle keeps the step table it compiled to before blocks existed,
/// entry for entry. With at least one, the table is the **stable
/// topological order of the condensation**: blocks are nodes beside the
/// remaining equations and operators, an equation's in-edges come from
/// the writers of what it reads, a block's from the writers of its
/// coefficient inputs, an operator's from the writers of its available
/// quantity and its demands : and nothing else. An **allocated
/// quantity sets no ordering constraint on its readers** (the
/// carve-out mirrored: the resolution iterates such reads to a
/// fixpoint), so operators register no writer edges, and declaration
/// order keeps the steps the graph leaves free where they were.
///
/// A member equation is *removed* from the table and folded into its
/// block: the block takes the earliest position its members held.
fn form_blocks(
    resolver: &Resolver,
    explicit: Vec<CStep>,
    blocks: Vec<AlgebraicBlock>,
    declared_order: bool,
) -> Result<Vec<CStep>, CompileError> {
    if blocks.is_empty() {
        return Ok(explicit);
    }

    // Where each member equation sat in the table the declared order
    // left: the block inherits the earliest of those positions, which
    // is what keeps it where its members were whenever the dependencies
    // leave it free.
    let mut equation_positions: HashMap<VarIdx, usize> = HashMap::new();
    for (position, step) in explicit.iter().enumerate() {
        if let CStep::Equation { target, .. } = step {
            equation_positions.insert(*target, position);
        }
    }

    let mut nodes: Vec<OrderNode> = Vec::new();
    let mut block_of_member: HashMap<VarIdx, usize> = HashMap::new();
    for block in &blocks {
        let mut targets = Vec::with_capacity(block.unknowns.len());
        for unknown in &block.unknowns {
            targets.push(resolver.var(&unknown.component, &unknown.attribute)?);
        }
        let columns: HashMap<&AttrRef, usize> = block
            .unknowns
            .iter()
            .enumerate()
            .map(|(column, unknown)| (unknown, column))
            .collect();
        let mut members = Vec::with_capacity(block.rows.len());
        let mut constants = Vec::with_capacity(block.rows.len());
        let mut coefficients = Vec::with_capacity(block.rows.len());
        let mut position = usize::MAX;
        for (row, unknown, &target) in block
            .rows
            .iter()
            .zip(&block.unknowns)
            .zip(&targets)
            .map(|((row, unknown), target)| (row, unknown, target))
        {
            members.push(format!("{}.{}", unknown.component, unknown.attribute));
            constants.push(resolver.compile_expr(&row.constant)?);
            let mut compiled: Vec<(usize, CExpr)> = row
                .terms
                .iter()
                .map(|(term_unknown, coefficient)| {
                    Ok((columns[term_unknown], resolver.compile_expr(coefficient)?))
                })
                .collect::<Result<Vec<_>, CompileError>>()?;
            compiled.sort_by_key(|(column, _)| *column);
            coefficients.push(compiled);
            position = position.min(equation_positions[&target]);
            block_of_member.insert(target, nodes.len());
        }
        let compiled = CBlock {
            name: members
                .first()
                .cloned()
                .unwrap_or_else(|| "block".to_owned()),
            targets,
            members,
            constants,
            coefficients,
            constant_factorization: None,
        };
        nodes.push(OrderNode {
            position,
            reads: compiled_reads(&compiled),
            registered_writes: compiled.targets.clone(),
            step: CStep::Block(compiled),
        });
    }
    for (surviving_position, step) in explicit.into_iter().enumerate() {
        let (reads, registered_writes) = match &step {
            CStep::Equation { target, expr } => {
                // A member has been folded into its block already; the
                // block carries its row.
                if block_of_member.contains_key(target) {
                    continue;
                }
                let mut reads = Vec::new();
                expr.collect_value_reads(&mut reads);
                (reads, vec![*target])
            }
            CStep::Allocate(allocation) => {
                let mut reads = Vec::new();
                allocation.available.collect_value_reads(&mut reads);
                reads.extend_from_slice(&allocation.demands);
                (reads, Vec::new())
            }
            CStep::Block(_) => {
                return Err(CompileError::Unresolved {
                    what: "individual step before algebraic block formation",
                    name: "an algebraic block".to_owned(),
                })
            }
        };
        nodes.push(OrderNode {
            position: surviving_position,
            reads,
            registered_writes,
            step,
        });
    }
    order_condensation(nodes, declared_order, resolver)
}

/// The numeric value of every coefficient of a block when they are all
/// **constant** (read no attribute, no state, no clock): one dense row of
/// values per block row, `None` the moment any coefficient depends on
/// the trajectory (that block's singularities are the run-time pivot
/// test's business).
fn constant_coefficients(block: &CBlock) -> Option<Vec<Vec<f64>>> {
    block
        .coefficients
        .iter()
        .map(|row| {
            let mut values = vec![0.0; block.targets.len()];
            for (column, coefficient) in row {
                values[*column] = coefficient.const_value()?;
            }
            Some(values)
        })
        .collect()
}

/// The variables a compiled block reads outside itself: everything its
/// constants and coefficients read (its members appear only as solved
/// columns).
fn compiled_reads(block: &CBlock) -> Vec<VarIdx> {
    let mut vars = Vec::new();
    for expr in &block.constants {
        expr.collect_value_reads(&mut vars);
    }
    for row in &block.coefficients {
        for (_, coefficient) in row {
            coefficient.collect_value_reads(&mut vars);
        }
    }
    vars
}

/// Stable topological order of the condensation: Kahn's algorithm over
/// the nodes' in-edges, the ready set kept by the position each node
/// inherits from the table the declared order left, so a step the graph
/// leaves free stays exactly where the model put it.
///
/// The condensation of a classified model is acyclic by construction:
/// every cycle of the read relation is inside a block, and an operator
/// registers no writer edges. Should a cycle survive anyway, that is a
/// validator/compiler disagreement and surfaces as a typed error, never
/// as a half-ordered table.
fn order_condensation(
    nodes: Vec<OrderNode>,
    declared_order: bool,
    resolver: &Resolver,
) -> Result<Vec<CStep>, CompileError> {
    let count = nodes.len();
    let mut writers: HashMap<VarIdx, Vec<usize>> = HashMap::new();
    for (id, node) in nodes.iter().enumerate() {
        for &var in &node.registered_writes {
            writers.entry(var).or_default().push(id);
        }
    }
    let mut remaining: Vec<usize> = vec![0; count];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (id, node) in nodes.iter().enumerate() {
        let mut waiting: Vec<usize> = Vec::new();
        for var in &node.reads {
            for &writer in writers.get(var).into_iter().flatten() {
                if writer != id {
                    if declared_order && nodes[writer].position > node.position {
                        let name = |node: &OrderNode| match &node.step {
                            CStep::Equation { target, .. } => resolver
                                .vars
                                .iter()
                                .find_map(|((component, attribute), index)| {
                                    (*index == *target).then(|| format!("{component}.{attribute}"))
                                })
                                .unwrap_or_else(|| format!("variable {target}")),
                            CStep::Block(block) => block.members.join(", "),
                            CStep::Allocate(allocation) => allocation.name.clone(),
                        };
                        return Err(CompileError::Invalid(ModelError::AlgebraicOrderConflict {
                            first: name(node),
                            second: name(&nodes[writer]),
                        }));
                    }
                    waiting.push(writer);
                }
            }
        }
        waiting.sort_unstable();
        waiting.dedup();
        remaining[id] = waiting.len();
        for writer in waiting {
            dependents[writer].push(id);
        }
    }
    let mut ready: BTreeSet<(usize, usize)> = nodes
        .iter()
        .enumerate()
        .filter(|(id, _)| remaining[*id] == 0)
        .map(|(id, node)| (node.position, id))
        .collect();
    let mut ordered = Vec::with_capacity(count);
    while let Some((_, id)) = ready.pop_first() {
        ordered.push(nodes[id].step.clone());
        for &dependent in &dependents[id] {
            remaining[dependent] -= 1;
            if remaining[dependent] == 0 {
                ready.insert((nodes[dependent].position, dependent));
            }
        }
    }
    if ordered.len() != count {
        return Err(CompileError::Unresolved {
            what: "acyclic sweep order (a cycle survived the block classification)",
            name: "the explicit sweep".to_owned(),
        });
    }
    Ok(ordered)
}
impl CExpr {
    /// Collect the attribute and automaton sensitivity sets of this
    /// expression (which changes must re-trigger a function reading it).
    pub(crate) fn collect_sensitivity(&self, vars: &mut Vec<VarIdx>, auts: &mut Vec<AutIdx>) {
        self.collect_dependencies(vars, auts, true);
    }

    /// Sweep ordering follows values, whereas the historical sensitivity
    /// relation also subscribes to sources of a topology-only `count`.
    fn collect_value_reads(&self, vars: &mut Vec<VarIdx>) {
        self.collect_dependencies(vars, &mut Vec::new(), false);
    }

    fn collect_dependencies(
        &self,
        vars: &mut Vec<VarIdx>,
        auts: &mut Vec<AutIdx>,
        count_sources: bool,
    ) {
        match self {
            CExpr::Const(_) => {}
            CExpr::Var(idx) => vars.push(*idx),
            CExpr::StateActive { automaton, .. } => auts.push(*automaton),
            CExpr::PortAgg { sources, agg } => {
                if count_sources || *agg != AggOp::Count {
                    vars.extend_from_slice(sources);
                }
            }
            CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
                lhs.collect_dependencies(vars, auts, count_sources);
                rhs.collect_dependencies(vars, auts, count_sources);
            }
            CExpr::Bool { args, .. }
            | CExpr::Add { args }
            | CExpr::Mul { args }
            | CExpr::Min { args }
            | CExpr::Max { args } => {
                for a in args {
                    a.collect_dependencies(vars, auts, count_sources);
                }
            }
            CExpr::If {
                cond,
                then,
                otherwise,
            } => {
                cond.collect_dependencies(vars, auts, count_sources);
                then.collect_dependencies(vars, auts, count_sources);
                otherwise.collect_dependencies(vars, auts, count_sources);
            }
            CExpr::Sin(arg) | CExpr::Exp(arg) => {
                arg.collect_dependencies(vars, auts, count_sources);
            }
            CExpr::Time => {}
        }
    }

    /// Whether this expression reads the simulation time (which makes
    /// it continuously varying even without ODE attributes).
    pub(crate) fn reads_time(&self) -> bool {
        match self {
            CExpr::Time => true,
            CExpr::Const(_) | CExpr::Var(_) | CExpr::StateActive { .. } | CExpr::PortAgg { .. } => {
                false
            }
            CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
                lhs.reads_time() || rhs.reads_time()
            }
            CExpr::Bool { args, .. }
            | CExpr::Add { args }
            | CExpr::Mul { args }
            | CExpr::Min { args }
            | CExpr::Max { args } => args.iter().any(CExpr::reads_time),
            CExpr::If {
                cond,
                then,
                otherwise,
            } => cond.reads_time() || then.reads_time() || otherwise.reads_time(),
            CExpr::Sin(arg) | CExpr::Exp(arg) => arg.reads_time(),
        }
    }

    /// The value of this expression when it reads no state at all: only
    /// literals, arithmetic, the clock-independent branches of a
    /// conditional and the like answer. `None` means "depends on the
    /// trajectory" and is the compiler's test for *constant* (level-2
    /// singularity is decided on constants alone).
    pub(crate) fn const_value(&self) -> Option<f64> {
        match self {
            CExpr::Const(Value::Float(value)) => Some(*value),
            CExpr::Const(Value::Int(value)) => Some(*value as f64),
            CExpr::Const(Value::Bool(_)) | CExpr::Var(_) => None,
            CExpr::StateActive { .. } | CExpr::PortAgg { .. } | CExpr::Time => None,
            CExpr::Cmp { .. } | CExpr::Bool { .. } => None,
            CExpr::Add { args } => args
                .iter()
                .try_fold(0.0, |sum, arg| Some(sum + arg.const_value()?)),
            CExpr::Sub { lhs, rhs } => Some(lhs.const_value()? - rhs.const_value()?),
            CExpr::Mul { args } => args
                .iter()
                .try_fold(1.0, |product, arg| Some(product * arg.const_value()?)),
            CExpr::Div { lhs, rhs } => Some(lhs.const_value()? / rhs.const_value()?),
            CExpr::Min { args } => args
                .iter()
                .map(CExpr::const_value)
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .reduce(f64::min),
            CExpr::Max { args } => args
                .iter()
                .map(CExpr::const_value)
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .reduce(f64::max),
            CExpr::If {
                cond,
                then,
                otherwise,
            } => {
                // Sound and simple: a conditional is constant only when
                // both branches agree (a condition that cannot decide
                // between two constants leaves this `None`, and the
                // run-time pivot test answers instead).
                let _ = cond;
                match (then.const_value(), otherwise.const_value()) {
                    (Some(same), Some(other)) if same == other => Some(same),
                    _ => None,
                }
            }
            CExpr::Sin(arg) => Some(arg.const_value()?.sin()),
            CExpr::Exp(arg) => Some(arg.const_value()?.exp()),
        }
    }
}

/// Whether `expr` varies during continuous evolution: it reads the
/// simulation time or one of `continuous_vars` (ODE-integrated
/// attributes and their explicit-equation closure).
fn expr_is_continuous(expr: &CExpr, continuous_vars: &BTreeSet<VarIdx>) -> bool {
    if expr.reads_time() {
        return true;
    }
    let mut vars = Vec::new();
    let mut auts = Vec::new();
    expr.collect_sensitivity(&mut vars, &mut auts);
    vars.iter().any(|var| continuous_vars.contains(var))
}

/// Resolve one conservative distribution operator against the
/// connections its out port carries.
///
/// The demand and allocated vectors are built from the **same** iteration
/// over `model.connections` as the per-connection channel attributes
/// themselves, so the three orders (demands, allocations, policy
/// parameters) are one order: the connection declaration order.
fn compile_allocation(
    resolver: &Resolver,
    model: &Model,
    component: &raichu_model::Component,
    allocation: &Allocation,
) -> Result<CAllocation, CompileError> {
    let edges: Vec<&raichu_model::Connection> = model
        .connections
        .iter()
        .filter(|connection| {
            connection.from.component == component.name && connection.from.port == allocation.port
        })
        .collect();
    let channel_var = |channel: &str, connection: &raichu_model::Connection| {
        let attribute = raichu_model::channel_attribute_name(connection, channel);
        resolver.var(&component.name, &attribute)
    };
    let demands = edges
        .iter()
        .map(|connection| channel_var(&allocation.demand, connection))
        .collect::<Result<Vec<_>, _>>()?;
    let allocated = edges
        .iter()
        .map(|connection| channel_var(&allocation.allocated, connection))
        .collect::<Result<Vec<_>, _>>()?;

    // A consumer-keyed parameter resolved to the connection order.
    // `Model::validate` has established the bijection between the two, so
    // a missing entry here is a validator/compiler disagreement and is
    // reported as a typed error rather than defaulted.
    let value_for = |params: &[raichu_model::ConsumerParam], to: &PortRef| {
        params
            .iter()
            .find(|param| param.to.component == to.component && param.to.port == to.port)
            .map(|param| param.value)
            .ok_or_else(|| CompileError::Unresolved {
                what: "allocation policy value for connection",
                name: format!("{}.{}", to.component, to.port),
            })
    };
    let policy = match &allocation.policy {
        AllocationPolicy::Proportional => CPolicy::Proportional,
        AllocationPolicy::Shares { shares } => CPolicy::Shares(
            edges
                .iter()
                .map(|connection| value_for(shares, &connection.to))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        AllocationPolicy::Priority { priorities } => {
            let ranks = edges
                .iter()
                .map(|connection| value_for(priorities, &connection.to))
                .collect::<Result<Vec<f64>, _>>()?;
            // The serving order is settled **here**, once, by (rank,
            // declaration index): equal ranks therefore break by
            // declaration index, and the engine never sorts at run time.
            let mut order: Vec<usize> = (0..ranks.len()).collect();
            order.sort_by(|a, b| ranks[*a].total_cmp(&ranks[*b]).then(a.cmp(b)));
            CPolicy::Priority(order)
        }
    };

    Ok(CAllocation {
        name: format!("{}.{}", component.name, allocation.name),
        available: resolver.compile_expr(&allocation.available)?,
        demands,
        allocated,
        policy,
    })
}

/// Permute the explicit sweep into the model's declared evaluation order.
///
/// `Model::validate` has already established that the order and the sweep
/// steps are in bijection, so every lookup below resolves and nothing is
/// left over; the two failure paths are kept as typed errors rather than
/// assertions, because a validator/compiler disagreement must surface as
/// a diagnostic and never as a panic.
fn order_steps(
    resolver: &Resolver,
    order: &[AttrRef],
    steps: Vec<CStep>,
) -> Result<Vec<CStep>, CompileError> {
    // An entry names a step: an explicit equation by its target
    // attribute, or a distribution operator by its qualified name. The
    // two namespaces cannot collide inside one component
    // (`ModelError::EvaluationStepAmbiguous`).
    let mut equations: HashMap<VarIdx, CStep> = HashMap::new();
    let mut operators: HashMap<String, CStep> = HashMap::new();
    for step in steps {
        match &step {
            CStep::Equation { target, .. } => {
                equations.insert(*target, step);
            }
            CStep::Allocate(allocation) => {
                operators.insert(allocation.name.clone(), step);
            }
            // Blocks form after this permutation; a Block here is a
            // validator/compiler disagreement, refused as typed.
            CStep::Block(_) => {
                return Err(CompileError::Unresolved {
                    what: "sweep step named by the evaluation order",
                    name: "an algebraic block".to_owned(),
                });
            }
        }
    }

    let mut ordered = Vec::with_capacity(order.len());
    for entry in order {
        let qualified = format!("{}.{}", entry.component, entry.attribute);
        let named = resolver
            .vars
            .get(&(entry.component.clone(), entry.attribute.clone()))
            .and_then(|target| equations.remove(target));
        let Some(step) = named.or_else(|| operators.remove(&qualified)) else {
            return Err(CompileError::Unresolved {
                what: "sweep step named by the evaluation order",
                name: qualified,
            });
        };
        ordered.push(step);
    }
    if let Some(step) = equations
        .values()
        .next()
        .or_else(|| operators.values().next())
    {
        return Err(CompileError::Unresolved {
            what: "sweep step missing from the evaluation order",
            name: match step {
                CStep::Equation { target, .. } => format!("attribute #{target}"),
                CStep::Allocate(allocation) => allocation.name.clone(),
                CStep::Block(_) => "an algebraic block".to_owned(),
            },
        });
    }
    Ok(ordered)
}

impl CompiledModel {
    /// Validate `model` then resolve every name to dense indices.
    pub fn compile(model: &Model) -> Result<Self, CompileError> {
        model.validate()?;

        // Per-connection channel attributes: derived once, from the model
        // itself, so validation and this pass agree on what exists
        // (`Model::channel_attributes` is the single derivation).
        let channel_attributes = model.channel_attributes();

        // Pass 1: index attributes, automata, states.
        let mut resolver = Resolver {
            vars: HashMap::new(),
            states: HashMap::new(),
            automata: HashMap::new(),
            port_sources: HashMap::new(),
            port_channel_sources: HashMap::new(),
        };
        let mut var_names = Vec::new();
        let mut var_init = Vec::new();
        let mut automata = Vec::new();
        // (connection index, channel) → materialised attribute index.
        let mut channel_vars: HashMap<(usize, &str), VarIdx> = HashMap::new();
        for component in &model.components {
            for attribute in &component.attributes {
                resolver.vars.insert(
                    (component.name.clone(), attribute.name.clone()),
                    var_names.len(),
                );
                var_names.push(format!("{}.{}", component.name, attribute.name));
                var_init.push(attribute.init);
            }
            // Materialised channel attributes sit right after the
            // component's declared ones: ordinary float attributes, so
            // sensitivity triggers, the journal, the snapshot, indicators
            // and the estimators need no special case for them.
            for entry in channel_attributes
                .iter()
                .filter(|entry| entry.component == component.name)
            {
                let idx = var_names.len();
                resolver
                    .vars
                    .insert((component.name.clone(), entry.attribute.clone()), idx);
                var_names.push(format!("{}.{}", component.name, entry.attribute));
                var_init.push(Value::Float(entry.init));
                channel_vars.insert((entry.connection, entry.channel.as_str()), idx);
            }
            for automaton in &component.automata {
                let aut_idx = automata.len();
                resolver
                    .automata
                    .insert((component.name.clone(), automaton.name.clone()), aut_idx);
                let mut init = 0;
                for (state_idx, state) in automaton.states.iter().enumerate() {
                    resolver.states.insert(
                        (
                            component.name.clone(),
                            automaton.name.clone(),
                            state.clone(),
                        ),
                        (aut_idx, state_idx),
                    );
                    if state == &automaton.init {
                        init = state_idx;
                    }
                }
                automata.push(CAutomaton {
                    name: format!("{}.{}", component.name, automaton.name),
                    states: automaton.states.clone(),
                    init,
                    transitions: Vec::new(),
                });
            }
        }

        // Pass 2: connections → in-port source lists (declaration order).
        //
        // The declaration order of these lists is load-bearing: an
        // aggregation is an ordered floating-point fold, so a reordering
        // shifts existing results in their last bits and breaks the
        // strict comparison level of the validation contract.
        for (index, connection) in model.connections.iter().enumerate() {
            let source_port = model
                .components
                .iter()
                .find(|c| c.name == connection.from.component)
                .and_then(|c| c.ports.iter().find(|p| p.name == connection.from.port))
                .filter(|p| p.dir == PortDir::Out)
                .ok_or_else(|| CompileError::Unresolved {
                    what: "out-port",
                    name: format!("{}.{}", connection.from.component, connection.from.port),
                })?;
            let source_var = source_port
                .attr
                .as_ref()
                .ok_or_else(|| CompileError::Unresolved {
                    what: "out-port",
                    name: format!("{}.{}", connection.from.component, connection.from.port),
                })?;
            let var_idx = resolver.var(&connection.from.component, source_var)?;
            resolver
                .port_sources
                .entry((connection.to.component.clone(), connection.to.port.clone()))
                .or_default()
                .push(var_idx);
            for channel in &source_port.channels {
                let materialised = channel_vars
                    .get(&(index, channel.name.as_str()))
                    .copied()
                    .ok_or_else(|| CompileError::Unresolved {
                        what: "materialised channel attribute",
                        name: format!(
                            "{}.{}",
                            connection.from.component,
                            raichu_model::channel_attribute_name(connection, &channel.name)
                        ),
                    })?;
                resolver
                    .port_channel_sources
                    .entry((
                        connection.to.component.clone(),
                        connection.to.port.clone(),
                        channel.name.clone(),
                    ))
                    .or_default()
                    .push(materialised);
            }
        }

        // Pass 3: transitions, functions, indicators.
        let mut transitions = Vec::new();
        let mut explicit: Vec<CStep> = Vec::new();
        let mut ode: Vec<(VarIdx, CExpr)> = Vec::new();
        let mut functions = Vec::new();
        for component in &model.components {
            for automaton in &component.automata {
                let aut_idx = resolver.automata[&(component.name.clone(), automaton.name.clone())];
                for transition in &automaton.transitions {
                    let (_, source) =
                        resolver.state(&component.name, &automaton.name, &transition.source)?;
                    let targets = transition
                        .targets
                        .iter()
                        .map(|t| {
                            resolver
                                .state(&component.name, &automaton.name, t)
                                .map(|(_, s)| s)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let guard = transition
                        .guard
                        .as_ref()
                        .map(|g| resolver.compile_expr(g))
                        .transpose()?;
                    let distribution = match &transition.distrib {
                        Distrib::Delay { time } => CLaw::Delay(*time),
                        Distrib::Exp {
                            rate: Some(rate),
                            rate_expr: None,
                        } => CLaw::Exp(*rate),
                        Distrib::Exp {
                            rate: None,
                            rate_expr: Some(expr),
                        } => CLaw::ExpVar {
                            rate: resolver.compile_expr(expr)?,
                            // Continuity is resolved in pass 3-bis,
                            // once every equation has been collected.
                            continuous: false,
                        },
                        Distrib::Exp { .. } => {
                            // Unreachable after `Model::validate`
                            // (ExpRateSpec), kept as a typed error.
                            return Err(CompileError::Unresolved {
                                what: "exp rate (exactly one of rate/rate_expr)",
                                name: format!(
                                    "{}.{}.{}",
                                    component.name, automaton.name, transition.name
                                ),
                            });
                        }
                        Distrib::Weibull { shape, scale } => CLaw::Weibull(*shape, *scale),
                        Distrib::Lognormal { mu, sigma } => CLaw::Lognormal(*mu, *sigma),
                        Distrib::Gamma { shape, scale } => CLaw::Gamma(*shape, *scale),
                        Distrib::Uniform { low, high } => CLaw::Uniform(*low, *high),
                        Distrib::Empirical { points } => CLaw::Empirical(points.clone()),
                        Distrib::Inst { probs } => {
                            let complement = 1.0 - probs.iter().sum::<f64>();
                            let mut full = probs.clone();
                            full.push(complement);
                            CLaw::Inst(full)
                        }
                        Distrib::Watched => {
                            let Some(guard) = &transition.guard else {
                                return Err(CompileError::Unresolved {
                                    what: "watched guard",
                                    name: format!(
                                        "{}.{}.{}",
                                        component.name, automaton.name, transition.name
                                    ),
                                });
                            };
                            let margin = resolver.compile_watched_margin(guard)?;
                            CLaw::Watched { margin }
                        }
                    };
                    let effects = transition
                        .effects
                        .iter()
                        .map(|assignment| {
                            let target = resolver
                                .var(&assignment.target.component, &assignment.target.attribute)?;
                            let value = resolver.compile_expr(&assignment.value)?;
                            Ok((target, value))
                        })
                        .collect::<Result<Vec<_>, CompileError>>()?;
                    let trans_idx = transitions.len();
                    automata[aut_idx].transitions.push(trans_idx);
                    transitions.push(CTransition {
                        name: format!("{}.{}.{}", component.name, automaton.name, transition.name),
                        component: component.name.clone(),
                        automaton: aut_idx,
                        source,
                        guard,
                        targets,
                        on_interruption: transition.on_interruption,
                        monitored: transition.monitored,
                        cycle_group: transition.cycle_group.clone(),
                        kind: transition.kind,
                        effects,
                        distrib: distribution,
                    });
                }
            }
            for function in &component.sensitive_functions {
                let effects = function
                    .effects
                    .iter()
                    .map(|assignment| {
                        let target = resolver
                            .var(&assignment.target.component, &assignment.target.attribute)?;
                        let value = resolver.compile_expr(&assignment.value)?;
                        Ok((target, value))
                    })
                    .collect::<Result<Vec<_>, CompileError>>()?;
                functions.push(CFunction {
                    name: format!("{}.{}", component.name, function.name),
                    effects,
                });
            }
            for equation in &component.equations {
                let target = resolver.var(&component.name, &equation.target)?;
                let expr = resolver.compile_expr(&equation.expr)?;
                match equation.kind {
                    EquationKind::Explicit => explicit.push(CStep::Equation { target, expr }),
                    EquationKind::Ode => ode.push((target, expr)),
                }
            }
            // Distribution operators come after the component's explicit
            // equations, which is the positional order documented for the
            // sweep; a model that needs another one declares it.
            for allocation in &component.allocations {
                let compiled = compile_allocation(&resolver, model, component, allocation)?;
                explicit.push(CStep::Allocate(compiled));
            }
        }
        let watched: Vec<TransIdx> = transitions
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t.distrib, CLaw::Watched { .. }))
            .map(|(i, _)| i)
            .collect();

        // Pass 3-ter: the **declared evaluation order**, applied once,
        // here, as a permutation of the explicit table.
        //
        // The order is a compile-time property of the table, not a
        // runtime indirection: `recompute_explicit` still walks
        // `model.explicit` from 0, so the ~23 sweeps an accepted solver
        // step performs are the same code over the same layout. A model
        // that declares no order does not enter this branch at all, so
        // its table keeps the positional order it had before the field
        // existed, entry for entry.
        //
        // Applied to the *individual* equations, before the algebraic
        // blocks form: members become one simultaneous step. The condensation
        // check refuses a declared reader-before-producer dependency.
        //
        // Placed before 3-bis, whose fixpoint over the explicit table
        // is order-independent by construction: it iterates to closure.
        if let Some(order) = &model.evaluation_order {
            explicit = order_steps(&resolver, order, explicit)?;
        }

        // Pass 3-quinquies: the **algebraic blocks**. The model
        // layer has classified every cycle of the explicit equations;
        // linear ones come back as blocks, which become single steps of
        // the sweep at the stable topological order of the condensation.
        // A model without a cycle takes none of this: its table is the
        // one the passes above produced, byte for byte.
        let blocks = model.algebraic_blocks()?;
        if !blocks.is_empty() {
            explicit = form_blocks(
                &resolver,
                explicit,
                blocks,
                model.evaluation_order.is_some(),
            )?;
            // Level 2 of the singularity analysis: a block whose
            // coefficients are **constant at build time** is factorised
            // here, and a rank-deficient factorisation is refused with
            // the variables it involves (`x = y` with `y = x`, or
            // `x = x`: the rows are linearly dependent). A block with
            // state-dependent coefficients keeps its singularities for
            // the run-time pivot test, which sees the actual state.
            for step in &mut explicit {
                let CStep::Block(block) = step else {
                    continue;
                };
                let Some(coefficients) = constant_coefficients(block) else {
                    continue;
                };
                let n = block.targets.len();
                let mut matrix = vec![0.0; n * n];
                for (row, values) in matrix.chunks_mut(n).zip(&coefficients) {
                    for (entry, value) in row.iter_mut().zip(values) {
                        *entry -= value;
                    }
                }
                for (diagonal, entry) in matrix.iter_mut().enumerate() {
                    if diagonal % (n + 1) == 0 {
                        *entry += 1.0;
                    }
                }
                block.constant_factorization = Some(
                    LuFactorization::decompose(&matrix, n).map_err(|error| match error {
                        raichu_numeric::LinearSolveError::Singular(_) => {
                            CompileError::Invalid(ModelError::AlgebraicSingular {
                                variables: block.members.join(", "),
                                reason: SingularReason::RankDeficient,
                            })
                        }
                        error => CompileError::Unresolved {
                            what: "finite algebraic block coefficients",
                            name: format!("{}: {error}", block.members.join(", ")),
                        },
                    })?,
                );
            }
        }

        // Pass 3-bis: continuity of state-dependent rates (`reschedule_modifiable`
        // routing). A attribute is *continuous* if the ODE integrates it
        // or an explicit equation ties it (transitively) to one.
        let mut continuous_vars: BTreeSet<VarIdx> = ode.iter().map(|(var, _)| *var).collect();
        loop {
            let mut changed = false;
            for step in &explicit {
                match step {
                    CStep::Equation { target, expr } => {
                        if !continuous_vars.contains(target)
                            && expr_is_continuous(expr, &continuous_vars)
                        {
                            continuous_vars.insert(*target);
                            changed = true;
                        }
                    }
                    // An allocated quantity moves whenever the available
                    // quantity or any demand moves: the operator is a
                    // function of them, evaluated at every solver stage
                    // like any other step of the sweep.
                    CStep::Allocate(allocation) => {
                        let moving = expr_is_continuous(&allocation.available, &continuous_vars)
                            || allocation
                                .demands
                                .iter()
                                .any(|var| continuous_vars.contains(var));
                        if moving {
                            for var in &allocation.allocated {
                                changed |= continuous_vars.insert(*var);
                            }
                        }
                    }
                    // A block's members move together: all of them are
                    // functions of the block's coefficient inputs, so
                    // any input that moves moves the whole block. The
                    // members appear only as solved columns, never as
                    // reads of their own rows.
                    CStep::Block(block) => {
                        let moving = block
                            .constants
                            .iter()
                            .any(|expr| expr_is_continuous(expr, &continuous_vars))
                            || block.coefficients.iter().flat_map(|row| row.iter()).any(
                                |(_, coefficient)| {
                                    expr_is_continuous(coefficient, &continuous_vars)
                                },
                            );
                        if moving {
                            for var in &block.targets {
                                changed |= continuous_vars.insert(*var);
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        for transition in &mut transitions {
            let is_continuous = match &transition.distrib {
                CLaw::ExpVar { rate, .. } => Some(expr_is_continuous(rate, &continuous_vars)),
                _ => None,
            };
            if let (Some(flag), CLaw::ExpVar { continuous, .. }) =
                (is_continuous, &mut transition.distrib)
            {
                *continuous = flag;
            }
        }

        // Pass 3-quater: the **active-set margins**, one entry per
        // distribution operator, indexed against the *final* sweep table
        // (hence after 3-ter, which permutes it).
        //
        // Compiling them here rather than deriving them at each segment
        // start is what registers their variable dependencies once: a
        // later index inverting `deps` sees every margin the operators
        // contribute, exactly as it sees the margins of the watched
        // transitions.
        let flow_margins: Vec<CFlowMargins> = explicit
            .iter()
            .enumerate()
            .filter_map(|(step, item)| match item {
                CStep::Allocate(allocation) => Some((step, allocation)),
                CStep::Equation { .. } | CStep::Block(_) => None,
            })
            .map(|(step, allocation)| {
                let mut deps = Vec::new();
                let mut auts = Vec::new();
                allocation
                    .available
                    .collect_sensitivity(&mut deps, &mut auts);
                deps.extend_from_slice(&allocation.demands);
                deps.sort_unstable();
                deps.dedup();
                CFlowMargins {
                    step,
                    name: allocation.name.clone(),
                    consumers: allocation
                        .allocated
                        .iter()
                        .map(|&var| var_names[var].clone())
                        .collect(),
                    deps,
                }
            })
            .collect();

        // Pass 3-quinquies: **invert** the margin dependencies, exactly
        // as pass 4 below inverts the sensitive functions'. Both scan
        // sites of the watched population then cost what moved instead of
        // what exists.
        //
        // The guard is the only expression consulted: the margin is
        // compiled *from* the guard (`compile_watched_margin`), so the
        // two read the same attributes and one dependency set answers for
        // both the immediate-guard check and the located crossing.
        let mut margin_index = MarginIndex {
            watched_by_var: vec![Vec::new(); var_names.len()],
            watched_by_state: vec![Vec::new(); automata.len()],
            watched_by_time: Vec::new(),
            watched_by_owner: vec![Vec::new(); automata.len()],
            flow_by_var: vec![Vec::new(); var_names.len()],
        };
        for (position, &trans_idx) in watched.iter().enumerate() {
            let transition = &transitions[trans_idx];
            margin_index.watched_by_owner[transition.automaton].push(position);
            let Some(guard) = &transition.guard else {
                continue;
            };
            let mut vars = Vec::new();
            let mut auts = Vec::new();
            guard.collect_sensitivity(&mut vars, &mut auts);
            vars.sort_unstable();
            vars.dedup();
            auts.sort_unstable();
            auts.dedup();
            for var in vars {
                margin_index.watched_by_var[var].push(position);
            }
            for aut in auts {
                margin_index.watched_by_state[aut].push(position);
            }
            if guard.reads_time() {
                margin_index.watched_by_time.push(position);
            }
        }
        for (operator, margins) in flow_margins.iter().enumerate() {
            for &var in &margins.deps {
                margin_index.flow_by_var[var].push(operator);
            }
        }

        let programs = model
            .programs
            .iter()
            .map(|program| {
                let decisions: HashSet<_> = program
                    .variables
                    .iter()
                    .map(|variable| variable.attribute.clone())
                    .collect();
                let columns: HashMap<_, _> = program
                    .variables
                    .iter()
                    .enumerate()
                    .map(|(index, variable)| (variable.attribute.clone(), index))
                    .collect();
                let variables = program
                    .variables
                    .iter()
                    .map(|variable| {
                        let target = resolver
                            .var(&variable.attribute.component, &variable.attribute.attribute)?;
                        let kind = match variable.on_infeasible {
                            Value::Bool(_) => raichu_model::AttrKind::Bool,
                            Value::Int(_) => raichu_model::AttrKind::Int,
                            Value::Float(_) => raichu_model::AttrKind::Float,
                        };
                        Ok(CProgramVariable {
                            target,
                            kind,
                            lower: variable
                                .lower
                                .as_ref()
                                .map(|expr| resolver.compile_expr(expr))
                                .transpose()?,
                            upper: variable
                                .upper
                                .as_ref()
                                .map(|expr| resolver.compile_expr(expr))
                                .transpose()?,
                            fallback: variable.on_infeasible,
                        })
                    })
                    .collect::<Result<Vec<_>, CompileError>>()?;
                let objective =
                    resolver.compile_program_affine(&program.objective, &decisions, &columns)?;
                let constraints = program
                    .constraints
                    .iter()
                    .map(|constraint| {
                        Ok(CProgramConstraint {
                            expr: resolver.compile_program_affine(
                                &constraint.expr,
                                &decisions,
                                &columns,
                            )?,
                            lower: constraint
                                .lower
                                .as_ref()
                                .map(|expr| resolver.compile_expr(expr))
                                .transpose()?,
                            upper: constraint
                                .upper
                                .as_ref()
                                .map(|expr| resolver.compile_expr(expr))
                                .transpose()?,
                        })
                    })
                    .collect::<Result<Vec<_>, CompileError>>()?;
                let tie_break = match &program.tie_break {
                    None => CProgramTieBreak::Default,
                    Some(ProgramTieBreak::None) => CProgramTieBreak::None,
                    Some(ProgramTieBreak::Objectives(objectives)) => CProgramTieBreak::Objectives(
                        objectives
                            .iter()
                            .map(|objective| {
                                Ok((
                                    program_sense(objective.sense),
                                    resolver.compile_program_affine(
                                        &objective.expr,
                                        &decisions,
                                        &columns,
                                    )?,
                                ))
                            })
                            .collect::<Result<Vec<_>, CompileError>>()?,
                    ),
                };
                Ok(CProgram {
                    name: program.name.clone(),
                    variables,
                    sense: program_sense(program.sense),
                    objective,
                    constraints,
                    feasible: resolver
                        .var(&program.feasible.component, &program.feasible.attribute)?,
                    objective_value: program
                        .objective_value
                        .as_ref()
                        .map(|attr| resolver.var(&attr.component, &attr.attribute))
                        .transpose()?,
                    tie_break,
                    node_limit: program.node_limit.map(|limit| limit as u64),
                })
            })
            .collect::<Result<Vec<CProgram>, CompileError>>()?;

        // Pass 4: sensitivity sets → trigger tables.
        let mut var_triggers = vec![Vec::new(); var_names.len()];
        let mut state_triggers = vec![Vec::new(); automata.len()];
        for (fn_idx, function) in functions.iter().enumerate() {
            let mut vars = Vec::new();
            let mut auts = Vec::new();
            for (_, value) in &function.effects {
                value.collect_sensitivity(&mut vars, &mut auts);
            }
            vars.sort_unstable();
            vars.dedup();
            auts.sort_unstable();
            auts.dedup();
            for var in vars {
                var_triggers[var].push(fn_idx);
            }
            for aut in auts {
                state_triggers[aut].push(fn_idx);
            }
        }
        for (program_idx, program) in programs.iter().enumerate() {
            let mut vars = Vec::new();
            let mut auts = Vec::new();
            program.objective.collect_sensitivity(&mut vars, &mut auts);
            for variable in &program.variables {
                for expr in [variable.lower.as_ref(), variable.upper.as_ref()]
                    .into_iter()
                    .flatten()
                {
                    expr.collect_sensitivity(&mut vars, &mut auts);
                }
            }
            for constraint in &program.constraints {
                constraint.expr.collect_sensitivity(&mut vars, &mut auts);
                for expr in [constraint.lower.as_ref(), constraint.upper.as_ref()]
                    .into_iter()
                    .flatten()
                {
                    expr.collect_sensitivity(&mut vars, &mut auts);
                }
            }
            if let CProgramTieBreak::Objectives(objectives) = &program.tie_break {
                for (_, objective) in objectives {
                    objective.collect_sensitivity(&mut vars, &mut auts);
                }
            }
            vars.sort_unstable();
            vars.dedup();
            auts.sort_unstable();
            auts.dedup();
            let step = functions.len() + program_idx;
            for var in vars {
                var_triggers[var].push(step);
            }
            for aut in auts {
                state_triggers[aut].push(step);
            }
        }

        // Pass 5: indicators.
        let indicators = model
            .indicators
            .iter()
            .map(|indicator| {
                let target = match &indicator.target {
                    IndicatorTarget::Attribute { attr } => {
                        CIndicatorTarget::Var(resolver.var(&attr.component, &attr.attribute)?)
                    }
                    IndicatorTarget::State {
                        component,
                        automaton,
                        state,
                    } => {
                        let (aut, st) = resolver.state(component, automaton, state)?;
                        CIndicatorTarget::State(aut, st)
                    }
                    IndicatorTarget::Predicate { attr, cmp, value } => CIndicatorTarget::Predicate(
                        resolver.var(&attr.component, &attr.attribute)?,
                        *cmp,
                        *value,
                    ),
                };
                Ok(CIndicator {
                    name: indicator.name.clone(),
                    target,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;

        let targets = model
            .targets
            .iter()
            .map(|target| {
                let (aut, st) =
                    resolver.state(&target.component, &target.automaton, &target.state)?;
                Ok(CTarget {
                    name: target.name.clone(),
                    automaton: aut,
                    state: st,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;

        let var_index = var_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let automaton_index = automata
            .iter()
            .enumerate()
            .map(|(i, a)| (a.name.clone(), i))
            .collect();

        let mut compiled = CompiledModel {
            cache_id: {
                static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
            name: model.name.clone(),
            fmu_units: model.fmu_units.clone(),
            var_names,
            var_init,
            automata,
            transitions,
            functions,
            programs,
            var_triggers,
            state_triggers,
            indicators,
            targets,
            ode,
            unbounded_rate: model.unbounded_rate,
            explicit_reads_time: explicit.iter().any(|step| match step {
                CStep::Equation { expr, .. } => expr.reads_time(),
                // An allocation distributes a quantity it is handed; it
                // reads no clock of its own.
                CStep::Allocate(_) => false,
                // A block whose constants or coefficients read the
                // clock moves with it : and its factorisation is
                // redone every sweep, the coefficient bits having
                // changed.
                CStep::Block(block) => {
                    block.constants.iter().any(CExpr::reads_time)
                        || block
                            .coefficients
                            .iter()
                            .flat_map(|row| row.iter())
                            .any(|(_, coefficient)| coefficient.reads_time())
                }
            }),
            sweep_reads_ahead: sweep_reads_ahead(&explicit),
            explicit,
            watched,
            flow_margins,
            margin_index,
            var_index,
            automaton_index,
            switching_loops: Vec::new(),
            unfed_triggers: Vec::new(),
        };
        // Structural diagnostics come last, on the finished tables. A
        // warning and never a refusal: the loop itself is legitimate, and
        // `tracing` costs nothing where no subscriber is installed, so a
        // library caller pays nothing and an application sees it.
        compiled.switching_loops = crate::loops::switching_loops(&compiled);
        for found in &compiled.switching_loops {
            tracing::warn!(model = %compiled.name, "{}", found.describe());
        }
        // Same route, same rule, and the authored model rather than the
        // tables: the wire that is missing has a name only up there, the
        // resolution to indices having replaced every port by the list of
        // attributes behind it.
        compiled.unfed_triggers = crate::triggers::unfed_triggers(model);
        for found in &compiled.unfed_triggers {
            tracing::warn!(model = %compiled.name, "{}", found.describe());
        }
        Ok(compiled)
    }
}
