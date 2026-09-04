//! Name→index resolution: turns a validated [`Model`] into dense tables
//! the engine consumes.
//!
//! Performance contract: all names are resolved here, once, at
//! build time: the simulation hot path only touches vector indices,
//! never string lookups, and never allocates.

use crate::flow::CPolicy;
use raichu_expr::{AggOp, AttrRef, BoolOp, CmpOp, Expr, PortRef, Value};
use raichu_model::{
    Allocation, AllocationPolicy, Distrib, EquationKind, IndicatorTarget, InterruptionPolicy,
    Model, ModelError, PortDir,
};
use std::collections::{BTreeSet, HashMap};
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

/// One step of the **explicit sweep**, run at every evaluation point in
/// table order.
///
/// An equation writes one attribute; a distribution operator writes one
/// per outgoing connection, which is why it is a step of its own rather
/// than a sensitive-function effect. Both live in one table because both
/// must run inside the solver callbacks: a quantity written outside that
/// table stays frozen through an integration segment, and a watched
/// margin reading it would be polled rather than located.
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
}

/// A compiled sensitive function.
#[derive(Debug, Clone)]
pub struct CFunction {
    /// Qualified name `component.function`.
    pub name: String,
    /// Ordered effects `target := value`.
    pub effects: Vec<(VarIdx, CExpr)>,
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
}

/// A validated model resolved to dense tables.
#[derive(Debug, Clone)]
pub struct CompiledModel {
    /// Model name (provenance).
    pub name: String,
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

impl CExpr {
    /// Collect the attribute and automaton sensitivity sets of this
    /// expression (which changes must re-trigger a function reading it).
    fn collect_sensitivity(&self, vars: &mut Vec<VarIdx>, auts: &mut Vec<AutIdx>) {
        match self {
            CExpr::Const(_) => {}
            CExpr::Var(idx) => vars.push(*idx),
            CExpr::StateActive { automaton, .. } => auts.push(*automaton),
            CExpr::PortAgg { sources, .. } => vars.extend_from_slice(sources),
            CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
                lhs.collect_sensitivity(vars, auts);
                rhs.collect_sensitivity(vars, auts);
            }
            CExpr::Bool { args, .. }
            | CExpr::Add { args }
            | CExpr::Mul { args }
            | CExpr::Min { args }
            | CExpr::Max { args } => {
                for a in args {
                    a.collect_sensitivity(vars, auts);
                }
            }
            CExpr::If {
                cond,
                then,
                otherwise,
            } => {
                cond.collect_sensitivity(vars, auts);
                then.collect_sensitivity(vars, auts);
                otherwise.collect_sensitivity(vars, auts);
            }
            CExpr::Sin(arg) | CExpr::Exp(arg) => arg.collect_sensitivity(vars, auts),
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
        // Placed after 3-bis, whose fixpoint over the explicit table is
        // order-independent by construction: it iterates to closure.
        if let Some(order) = &model.evaluation_order {
            explicit = order_steps(&resolver, order, explicit)?;
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
                CStep::Equation { .. } => None,
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

        Ok(CompiledModel {
            name: model.name.clone(),
            var_names,
            var_init,
            automata,
            transitions,
            functions,
            var_triggers,
            state_triggers,
            indicators,
            targets,
            ode,
            explicit_reads_time: explicit.iter().any(|step| match step {
                CStep::Equation { expr, .. } => expr.reads_time(),
                // An allocation distributes a quantity it is handed; it
                // reads no clock of its own.
                CStep::Allocate(_) => false,
            }),
            explicit,
            watched,
            flow_margins,
            margin_index,
            var_index,
            automaton_index,
        })
    }
}
