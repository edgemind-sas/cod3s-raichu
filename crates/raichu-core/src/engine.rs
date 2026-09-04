//! The deterministic simulation engine (M0 discrete subset + M1
//! continuous evolution).
//!
//! Implements the cycle `init → schedule → continuous → discrete →
//! update` of Desgeorges et al. 2021. Rule mapping:
//!
//! - scheduling of deterministic transitions: `schedule_deterministic`
//!   ([`Engine::refresh_schedule`]);
//! - continuous evolution up to the next scheduled date: `integrate_continuous`
//!   ([`Engine::integrate_to`]);
//! - watched transitions fired at located boundary crossings: `schedule_boundary`
//!   (margin monitoring inside [`Engine::integrate_to`]);
//! - firing of the earliest transition: `fire_transition` ([`Engine::step`]);
//! - sensitive-function propagation to fixpoint: `propagate_effects`
//!   ([`Engine::run_fixpoint`]);
//! - dropping interruptible transitions whose guard turned false:
//!   `drop_disabled` ([`Engine::refresh_schedule`]).
//!
//! `schedule_stochastic` is implemented (M2, exponential distribution). `reschedule_modifiable` is
//! implemented through the cumulative-hazard realisation of
//! state-dependent rates (`CLaw::ExpVar`): a piecewise-constant rate
//! is rescheduled at each discrete change ([`Engine::refresh_schedule`]),
//! a continuously-varying rate is integrated alongside the ODE state and
//! its firing located like a boundary crossing ([`Engine::integrate_to`]).
//!
//! **Continuous/discrete coupling semantics:** sensitive functions
//! react to *discrete* changes (transition firings and effect
//! cascades); the continuous flow influences the discrete side only
//! through **watched transitions** (the paper's mechanism) and through
//! guards re-evaluated at discrete epochs. Explicit equations are
//! recomputed at every continuous evaluation point, in declaration
//! order, before ODE right-hand sides.
//!
//! **Equality semantics** (validation contract): the engine
//! guarantees a *deterministic* fixpoint order (global function
//! declaration order) but cross-validation only compares the *converged*
//! state and the event dates. The optional confluence check re-runs each
//! fixpoint in reverse order and reports divergence as a diagnostic
//! (rather than silently returning an order-dependent result).

use crate::compile::{
    AutIdx, CAllocation, CExpr, CFlowMargins, CIndicatorTarget, CLaw, CStep, CompiledModel, FnIdx,
    StateIdx, TransIdx, VarIdx, WatchedIdx,
};
use crate::flow::{allocate, classify, edge_margin, flow_band, EdgeClass, FLOW_TOLERANCE};
use raichu_expr::{AggOp, BoolOp, CmpOp, Value};
use raichu_numeric::{DormandPrince45, OdeSolver, OdeSystem, Outcome, SolverParams};
use rand_chacha::ChaCha8Rng;
use rand_distr::Distribution;
use serde::Serialize;
use std::collections::BTreeSet;
use thiserror::Error;

/// Convergence policy of the continuous **flow resolution**: the two
/// budgets that bound one resolution, the damping it applies to a
/// two-cycle, and the tolerance its quantities are settled to.
///
/// One object rather than four more fields on [`EngineConfig`] and four
/// more keywords on every entry point. The four are read *together*: a
/// tolerance loosened without a budget to match is a half-measure, and a
/// budget raised without knowing the tolerance it is spent against says
/// nothing. Grouping them also keeps the binding's already-wide
/// signatures from growing another four positional arguments, which is
/// what the surface exists to avoid.
///
/// [`Default`] reproduces the documented policy exactly, so a config
/// built without touching this group behaves as it did before the group
/// existed. Each field names the constant that documents *why* its
/// default is what it is; those constants stay the single place the
/// policy is argued.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowConfig {
    /// Sweeps the **numeric** level of one resolution may spend once its
    /// active set has settled. Default: [`FLOW_SWEEP_BUDGET`].
    pub sweep_budget: usize,
    /// Sweeps the **combinatorial** level of one resolution may spend,
    /// and segment restarts one instant may absorb.
    ///
    /// `None` derives it from the compiled network
    /// ([`active_set_budget`]), which is the default and was the only
    /// source before this knob existed. `Some(n)` overrides that
    /// derivation for every model this configuration runs: the
    /// derivation describes a *model*, so an override is a deliberate
    /// departure from what the model says about itself, not a tuning.
    pub active_set_budget: Option<usize>,
    /// Under-relaxation weight latched on the first detected two-cycle
    /// (`x ← (1 − w)·x + w·F(x)`). Default: [`FLOW_RELAXATION`]. A
    /// weight of one is no damping at all.
    pub relaxation: f64,
    /// Per-edge convergence tolerance of the numeric level, and the dead
    /// band of every active-set margin. One value serves both on
    /// purpose: see [`FLOW_TOLERANCE`], the default, whose documentation
    /// carries the ordering against the event-location tolerance that
    /// keeps a freshly resolved network from re-crossing its own
    /// boundary on the spot. Loosening it past that ordering trades that
    /// guarantee away.
    pub tolerance: f64,
}

impl Default for FlowConfig {
    fn default() -> Self {
        FlowConfig {
            sweep_budget: FLOW_SWEEP_BUDGET,
            active_set_budget: None,
            relaxation: FLOW_RELAXATION,
            tolerance: FLOW_TOLERANCE,
        }
    }
}

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Simulation horizon (events strictly after it are not fired).
    pub t_max: f64,
    /// Record the structured causal journal (zero cost when `false`).
    pub journal: bool,
    /// Record the per-trajectory sequence trace: the ordered `SeqEvent`s of
    /// fired *monitored* transitions plus the end cause when a target
    /// (feared event) is reached (zero cost when `false`). Recording only:
    /// the early stop is [`EngineConfig::stop_at_targets`], so a driver
    /// that wants the latch without the trace does not pay for the trace.
    pub sequences: bool,
    /// End the trajectory at the first target (feared event) reached,
    /// holding the latched state through the remaining sample instants
    /// (mirroring cod3s sequence runs). Independent of `sequences`:
    /// sequence *analysis* needs both, a Monte-Carlo run that only wants
    /// the early stop needs this one alone (zero cost when `false`).
    pub stop_at_targets: bool,
    /// Re-run every fixpoint in reverse order and fail on divergence
    /// (non-confluence diagnostic; ~2× fixpoint cost when enabled).
    pub confluence_check: bool,
    /// Safety cap on fixpoint iterations: beyond it the model is
    /// declared to have an instantaneous loop (typed error, not a hang).
    pub max_fixpoint_iterations: usize,
    /// Numerical parameters of the default ODE backend (explicit,
    /// recorded as provenance: validation-contract level 3).
    pub ode: SolverParams,
    /// Ascending instants at which every indicator is sampled (dense
    /// output for continuous attributes, piecewise-constant hold for
    /// discrete ones). Empty = no sampling.
    pub samples: Vec<f64>,
    /// Master seed of the RNG policy (M2). Only consumed by stochastic
    /// distributions; deterministic models ignore it.
    pub seed: u64,
    /// Substream index (`ChaCha8Rng::set_stream`): the Monte-Carlo
    /// driver assigns one stream per replica.
    pub rng_stream: u64,
    /// Convergence policy of the continuous flow resolution (budgets,
    /// damping, tolerance). [`FlowConfig::default`] is the documented
    /// policy; a model with no distribution operator runs no resolution
    /// and is untouched by any of it.
    pub flow: FlowConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            t_max: f64::INFINITY,
            journal: false,
            sequences: false,
            stop_at_targets: false,
            confluence_check: false,
            max_fixpoint_iterations: 10_000,
            ode: SolverParams::default(),
            samples: Vec::new(),
            seed: 0,
            rng_stream: 0,
            flow: FlowConfig::default(),
        }
    }
}

/// Typed runtime errors. The engine never panics on a library path.
#[derive(Debug, Error)]
pub enum EngineError {
    /// An expression combined values of incompatible kinds.
    #[error("type error at t={time}: {detail}")]
    TypeError {
        /// Simulation time of the failure.
        time: f64,
        /// Human-readable detail.
        detail: String,
    },
    /// The sensitive-function propagation did not reach a fixpoint.
    #[error(
        "no fixpoint after {iterations} iterations at t={time}: \
         probable instantaneous loop (functions keep rewriting state)"
    )]
    InstantaneousLoop {
        /// Simulation time of the failure.
        time: f64,
        /// Iteration cap that was hit.
        iterations: usize,
    },
    /// The converged state depends on the function evaluation order.
    #[error(
        "non-confluent model at t={time}: sensitive functions `{first}` and \
         `{second}` write conflicting values (converged state depends on \
         evaluation order)"
    )]
    NonConfluent {
        /// Simulation time of the diagnostic.
        time: f64,
        /// A function involved in the conflict (declaration order).
        first: String,
        /// The other function involved.
        second: String,
    },
    /// Interactive control: no transition carries this qualified name.
    #[error("unknown transition `{transition}`")]
    UnknownTransition {
        /// The requested (unresolved) transition name.
        transition: String,
    },
    /// Interactive control: the requested transition is not currently
    /// armed. It is neither date-scheduled (`pending`) nor a watched
    /// transition whose guard already holds, so there is nothing to fire.
    #[error("transition `{transition}` is not fireable at t={time} (not armed)")]
    NotFireable {
        /// The transition that could not be fired.
        transition: String,
        /// Simulation time of the attempt.
        time: f64,
    },
    /// Interactive control: a forced destination (`fire_*_to`) named a
    /// state that is not one of the transition's declared target
    /// branches.
    #[error("`{state}` is not a target branch of transition `{transition}`")]
    ForcedTargetInvalid {
        /// The transition whose branch was forced.
        transition: String,
        /// The invalid (unknown or non-target) state name.
        state: String,
    },
    /// Interactive control: a manual firing date (`set_date`) was in the
    /// past (before the current time) or non-finite.
    #[error(
        "cannot schedule transition `{transition}` at t={date} \
         (before the current time t={time})"
    )]
    DateInPast {
        /// The transition being re-dated.
        transition: String,
        /// The rejected date.
        date: f64,
        /// The current simulation time.
        time: f64,
    },
    /// The ODE backend failed (stiffness, non-finite derivatives, …).
    #[error("continuous evolution failed: {0}")]
    Ode(#[from] raichu_numeric::OdeError),
    /// Watched transitions kept firing at the same instant (Zeno-like
    /// loop on a boundary).
    #[error(
        "watched transitions keep firing at t={time} without time \
         advancing (boundary loop)"
    )]
    WatchedLoop {
        /// The stuck instant.
        time: f64,
    },
    /// The conservative flow network did not settle within its sweep
    /// budget (see [`Engine::resolve_flows`]).
    #[error(
        "the continuous flow network did not settle after {sweeps} sweeps \
         at t={time}: {cause}; still moving: {moving}"
    )]
    FlowNotConverged {
        /// Simulation time of the resolution that would not settle.
        time: f64,
        /// Sweeps the resolution spent before its budget ran out.
        sweeps: usize,
        /// Which budget ran out, and what the iteration was doing.
        cause: FlowStall,
        /// Qualified `operator[consumer]` name of every edge that moved
        /// in the final sweep, comma separated.
        moving: String,
    },
    /// A located active-set boundary was crossed again and again without
    /// time advancing: the continuous analogue of [`Self::WatchedLoop`],
    /// for a crossing that fires no transition.
    #[error(
        "the active set of the continuous flow network keeps changing at \
         t={time} without time advancing ({restarts} segment restarts at \
         the same instant); edges crossing: {edges}"
    )]
    FlowChattering {
        /// The stuck instant.
        time: f64,
        /// Segment restarts spent at that instant.
        restarts: usize,
        /// Qualified `operator[consumer]` name of every edge that
        /// crossed there, comma separated.
        edges: String,
    },
}

/// Why a continuous-flow resolution stopped without settling.
///
/// The three variants share one payload (the moving edges): a slow
/// monotone sequence and a long cycle exhaust a budget without matching
/// the two-cycle test, and a diagnostic that named the edges only in the
/// cycle case would leave the two commonest stalls unexplained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowStall {
    /// The saturation pattern kept changing: the combinatorial search
    /// spent the budget derived from the compiled edge count without the
    /// pattern ever holding still for two consecutive sweeps.
    ActiveSet,
    /// The saturation pattern held still but the quantities kept moving
    /// by more than [`FLOW_TOLERANCE`]: the numeric level spent its
    /// constant budget.
    Quantities,
    /// The iteration returned to a state it held two sweeps earlier:
    /// two allocations, each of which justifies the other. Under-relaxation
    /// was engaged in response and did not absorb it, so this is a policy
    /// conflict in the model, not a numerical accident.
    TwoCycle,
}

impl std::fmt::Display for FlowStall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            FlowStall::ActiveSet => {
                "the saturation pattern kept changing (active-set budget exhausted)"
            }
            FlowStall::Quantities => {
                "the saturation pattern held but the quantities kept moving \
                 (flow budget exhausted)"
            }
            FlowStall::TwoCycle => {
                "two allocations alternate, each justifying the other \
                 (two-cycle persisting under under-relaxation)"
            }
        };
        f.write_str(text)
    }
}

/// One record of the structured causal journal.
///
/// Covers two structured trace levels: attribute
/// modifications during the fixpoint phase, and transition firings.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "record", rename_all = "snake_case")]
pub enum JournalRecord {
    /// A transition fired (`fire_transition` / `schedule_boundary`).
    TransitionFired {
        /// Simulation time.
        time: f64,
        /// Qualified transition name.
        transition: String,
        /// Source state name.
        from: String,
        /// Target state name.
        to: String,
    },
    /// A sensitive function was triggered (`propagate_effects`).
    FunctionTriggered {
        /// Simulation time.
        time: f64,
        /// Qualified function name.
        function: String,
    },
    /// An attribute changed value (cause = the enclosing record above).
    AttributeChanged {
        /// Simulation time.
        time: f64,
        /// Qualified attribute name.
        attribute: String,
        /// Previous value.
        old: Value,
        /// New value.
        new: Value,
        /// Qualified name of the function that wrote it.
        cause: String,
    },
    /// A transition was scheduled (`schedule_deterministic`).
    TransitionScheduled {
        /// Simulation time.
        time: f64,
        /// Qualified transition name.
        transition: String,
        /// Planned firing date.
        firing_at: f64,
    },
    /// A pending stochastic transition was rescheduled because its
    /// state-dependent rate changed at a discrete step (`reschedule_modifiable`).
    TransitionRescheduled {
        /// Simulation time.
        time: f64,
        /// Qualified transition name.
        transition: String,
        /// New planned firing date (`+∞` serialises as `null`: the
        /// rate dropped to zero and the countdown is on hold).
        firing_at: f64,
    },
    /// The **active set** of a distribution operator changed at a located
    /// boundary crossing: the integration segment ended there, at the
    /// crossing instant rather than at the next scheduled date, and the
    /// network was resolved again from that state.
    ///
    /// This is the record that makes the located-crossing claim
    /// observable: a resolution that only happened at discrete dates
    /// would leave the journal without it, or with it at the wrong
    /// instant.
    ActiveSetCrossed {
        /// Located crossing instant.
        time: f64,
        /// Qualified operator name `component.allocation`.
        operator: String,
        /// Qualified name of the allocated attribute whose saturation
        /// changed.
        consumer: String,
        /// Saturation class the edge held during the segment.
        from: EdgeClass,
        /// Class it holds after the network was resolved again.
        to: EdgeClass,
    },
    /// A pending transition was dropped (`drop_disabled` or source left).
    TransitionDropped {
        /// Simulation time.
        time: f64,
        /// Qualified transition name.
        transition: String,
        /// Why it was dropped.
        reason: DropReason,
    },
}

/// Why a pending transition was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DropReason {
    /// Its guard turned false (`drop_disabled`) under the `reset` policy:
    /// a fresh duration is redrawn when the guard returns
    /// (interruptible transition).
    GuardFalse,
    /// Its guard turned false under the `resume` policy (RAICHU
    /// extension): the countdown is *paused* and the remaining time
    /// resumes when the guard returns.
    GuardPaused,
    /// Its automaton left the source state.
    SourceLeft,
}

/// Cumulative-hazard state of an armed state-dependent-rate transition
/// (`CLaw::ExpVar`): the transition fires when `accumulated` reaches
/// `threshold`, realising the PDMP survival `P(T > t) = exp(−∫λ dt)`
/// exactly. The threshold is drawn `Exp(1)` at arming (`schedule_stochastic`).
#[derive(Debug, Clone, Copy)]
struct Hazard {
    /// `Exp(1)` firing threshold `E`.
    threshold: f64,
    /// Hazard accumulated so far, `H = ∫ λ dt ≤ E`.
    accumulated: f64,
    /// Rate λ at `since`: supports the lazy piecewise-constant
    /// accumulation of non-continuous rates between discrete steps.
    rate: f64,
    /// Time of the last accumulation point.
    since: f64,
}

/// A fired event (the discrete structure compared at validation level 1).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Event {
    /// Firing date.
    pub time: f64,
    /// Qualified transition name.
    pub transition: String,
    /// Source state name.
    pub from: String,
    /// Target state name.
    pub to: String,
}

/// One recorded event of a trajectory's **sequence** (mirrors cod3s
/// `SeqEvent`): the entry into a monitored state. `name()` is `obj.attr`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SeqEvent {
    /// Owning component (cod3s `elt.parent().name()`).
    pub obj: String,
    /// The monitored state entered (cod3s `elt.basename()`, e.g. `occ__cc_12`).
    pub attr: String,
    /// Firing date.
    pub time: f64,
    /// Cycle-pair group id of the firing transition (internal to the
    /// cycle-filtering step; not part of the compared/serialized sequence).
    #[serde(skip)]
    pub cycle_group: Option<String>,
}

/// One trajectory's recorded sequence: the ordered monitored-transition
/// firings plus the end cause/time (the reached target, or `None` if the
/// trajectory ran to `t_max` without reaching one). Weight 1 per raw
/// trajectory; the Monte-Carlo pipeline groups and re-weights them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sequence {
    /// Ordered monitored-state entries.
    pub events: Vec<SeqEvent>,
    /// Reached target's name (`end_cause`), or `None` if none was reached.
    pub end_cause: Option<String>,
    /// Time the target was reached, or the horizon when none was.
    pub end_time: f64,
    /// Statistical weight (1 for a raw trajectory).
    pub weight: f64,
}

/// Kind of an armed transition, for interactive inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FireableKind {
    /// Deterministic delay.
    Delay,
    /// A sampled stochastic law (exponential, Weibull, lognormal, …).
    Stochastic,
    /// Instantaneous branching (fires at the current instant).
    Inst,
    /// Watched boundary transition (fires when its margin is crossed
    /// during continuous evolution).
    Watched,
}

/// One armed transition offered to interactive control
/// ([`Engine::fireable`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Fireable {
    /// Transition index: the stable handle for [`Engine::fire_idx`].
    pub index: usize,
    /// Qualified transition name (`component.automaton.transition`).
    pub transition: String,
    /// Kind of occurrence law.
    pub kind: FireableKind,
    /// Scheduled firing date; `None` for a watched transition whose
    /// boundary has not been located yet (its guard is not yet true:
    /// the crossing is found only during continuous evolution).
    pub date: Option<f64>,
}

/// An opaque checkpoint of the engine's full mutable trajectory state
/// (time, discrete + continuous attributes, schedule, RNG, recorded
/// history), produced by [`Engine::snapshot`] and reinstated by
/// [`Engine::restore`]. Cloning the RNG makes any continuation after a
/// restore bit-for-bit reproducible.
#[derive(Debug, Clone)]
pub struct Snapshot {
    time: f64,
    vars: Vec<Value>,
    states: Vec<StateIdx>,
    pending: Vec<Option<f64>>,
    frozen: Vec<Option<f64>>,
    hazards: Vec<Option<Hazard>>,
    events: Vec<Event>,
    journal: Vec<JournalRecord>,
    seq_events: Vec<SeqEvent>,
    seq_end: Option<(String, f64)>,
    indicator_series: Vec<IndicatorSeries>,
    sampled: Vec<IndicatorSeries>,
    sample_cursor: usize,
    watched_streak: (f64, usize),
    rng: ChaCha8Rng,
    worklist: BTreeSet<FnIdx>,
}

/// Value of an attribute by qualified name (`component.attribute`):
/// shared by [`Engine`] and [`Snapshot`] so the two never drift.
fn attribute_of(model: &CompiledModel, vars: &[Value], qualified: &str) -> Option<Value> {
    model.var_index.get(qualified).map(|&idx| vars[idx])
}

/// Current state name of an automaton by qualified name
/// (`component.automaton`): shared by [`Engine`] and [`Snapshot`].
fn state_of<'m>(model: &'m CompiledModel, states: &[StateIdx], qualified: &str) -> Option<&'m str> {
    model.automaton_index.get(qualified).map(|&idx| {
        let automaton = &model.automata[idx];
        automaton.states[states[idx]].as_str()
    })
}

impl Snapshot {
    /// Simulation time captured in this snapshot.
    ///
    /// Read directly, without rebuilding an [`Engine`]: a facade that
    /// polls the state between steps (the Python `interactive` object)
    /// would otherwise pay a full state clone per read.
    #[must_use]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Events fired up to this snapshot, chronological. See
    /// [`Snapshot::time`] on why this bypasses the engine rebuild.
    #[must_use]
    pub fn history(&self) -> &[Event] {
        &self.events
    }

    /// Value of an attribute by qualified name (`component.attribute`),
    /// resolved against the model this snapshot was taken from.
    #[must_use]
    pub fn attribute(&self, model: &CompiledModel, qualified: &str) -> Option<Value> {
        attribute_of(model, &self.vars, qualified)
    }

    /// Current state name of an automaton by qualified name
    /// (`component.automaton`), resolved against the model this snapshot
    /// was taken from.
    #[must_use]
    pub fn state<'m>(&self, model: &'m CompiledModel, qualified: &str) -> Option<&'m str> {
        state_of(model, &self.states, qualified)
    }
}

/// An indicator's recorded change-points `(time, value)`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IndicatorSeries {
    /// Indicator name.
    pub name: String,
    /// Change-points: value observed from each time onward (first entry
    /// is the initial value at t = 0).
    pub points: Vec<(f64, Value)>,
}

/// **Counted work** of a run: the machine-independent units a
/// performance comparison is expressed in.
///
/// Wall-clock moves with the machine, the allocator and the load; these
/// counts do not, which is what lets a third party reproduce a
/// measurement and what makes a self-regression visible even when a
/// wall-clock gate with slack stays green.
///
/// Two counters report zero for a model that declares no conservative
/// distribution operator: `flow_sweeps` and `allocation_capping_passes`.
/// That is not an absent producer but an absent network, and it is the
/// property that keeps such a model's profile identical to the one it had
/// before the flow resolution existed.
///
/// The counters are **cumulative instrumentation over the engine's
/// life**, not trajectory state: [`Engine::restore`] rewinds the
/// trajectory, never the work already done.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct WorkCounters {
    /// Calls to the explicit-equation pass. It runs on every solver
    /// stage, every interior event-scan sample and every bisection step,
    /// so this dominates the continuous cost of any model.
    pub explicit_evaluations: u64,
    /// ODE steps the error controller accepted.
    pub solver_steps_accepted: u64,
    /// ODE trial steps the error controller rejected and retried.
    pub solver_steps_rejected: u64,
    /// Integration segments started. A segment restart discards the
    /// adapted step size, so segments per unit simulated time is the
    /// cost driver a chattering boundary shows up in.
    pub segments: u64,
    /// Sweeps of the continuous-flow resolution: one per ordered pass of
    /// the descending iteration that settles the active set and then the
    /// flows, counted at every discrete epoch and at every located
    /// active-set crossing. Zero for a model with no distribution
    /// operator, which skips the resolution entirely.
    pub flow_sweeps: u64,
    /// Passes of the capping loop of the conservative distribution
    /// operators. One pass per operator per explicit sweep when no
    /// consumer is over-served, one more per consumer that is: the
    /// counter is therefore how much the *policies* cost on top of the
    /// sweep itself.
    pub allocation_capping_passes: u64,
    /// Watched-margin expressions evaluated inside solver callbacks.
    pub margin_evaluations: u64,
    /// Watched guards **evaluated** by the immediate-guard scan that runs
    /// after every discrete fixpoint.
    ///
    /// Evaluated, not visited: the scan walks the armed positions and
    /// answers from a cached verdict for every guard whose inputs have
    /// not moved since it was last evaluated (see [`crate::MarginIndex`]), so
    /// this counts the work the model's *changes* justify rather than the
    /// size of its watched population. A model whose network never moves
    /// pays one cold pass and nothing after it.
    pub immediate_guard_scans: u64,
}

/// Provenance metadata attached to every result (reproducibility by
/// construction).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Provenance {
    /// Engine version (workspace version).
    pub engine_version: String,
    /// Model name.
    pub model: String,
    /// Simulation horizon.
    pub t_max: f64,
    /// RNG seed (`None` in the deterministic engine; the field exists
    /// so M2 introduces no schema change).
    pub seed: Option<u64>,
    /// Relative tolerance of the ODE controller (level-3 provenance).
    pub ode_rtol: f64,
    /// Event-location time tolerance (level-3 provenance).
    pub ode_tol_event: f64,
}

/// Full result of a simulation run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SimulationResult {
    /// Fired events in order (validation level 1).
    pub events: Vec<Event>,
    /// Indicator change-point series.
    pub indicators: Vec<IndicatorSeries>,
    /// Indicator values at the requested sample instants (level-3
    /// trajectory comparison; empty when no schedule was given).
    pub samples: Vec<IndicatorSeries>,
    /// Causal journal (empty when disabled).
    pub journal: Vec<JournalRecord>,
    /// Recorded sequence (`None` when sequence recording is disabled).
    pub sequence: Option<Sequence>,
    /// Provenance metadata.
    pub provenance: Provenance,
    /// Counted work (machine-independent performance units).
    pub work: WorkCounters,
    /// Final simulation time.
    pub final_time: f64,
}

/// Evaluate a compiled expression against an explicit state (usable
/// both by the engine and by the continuous-system adapter).
fn eval_expr(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
) -> Result<Value, EngineError> {
    match expr {
        CExpr::Const(value) => Ok(*value),
        CExpr::Var(idx) => Ok(vars[*idx]),
        CExpr::StateActive { automaton, state } => Ok(Value::Bool(states[*automaton] == *state)),
        CExpr::PortAgg { sources, agg } => eval_agg(model, vars, time, sources, *agg),
        CExpr::Cmp { op, lhs, rhs } => {
            let lhs = eval_expr(model, vars, states, time, lhs)?;
            let rhs = eval_expr(model, vars, states, time, rhs)?;
            eval_cmp(time, *op, lhs, rhs)
        }
        CExpr::Bool { op, args } => match op {
            BoolOp::And => {
                for arg in args {
                    if !eval_bool(model, vars, states, time, arg)? {
                        return Ok(Value::Bool(false));
                    }
                }
                Ok(Value::Bool(true))
            }
            BoolOp::Or => {
                for arg in args {
                    if eval_bool(model, vars, states, time, arg)? {
                        return Ok(Value::Bool(true));
                    }
                }
                Ok(Value::Bool(false))
            }
            // Arity validated at model build (exactly one).
            BoolOp::Not => Ok(Value::Bool(!eval_bool(
                model, vars, states, time, &args[0],
            )?)),
        },
        CExpr::Add { args } | CExpr::Mul { args } => {
            let product = matches!(expr, CExpr::Mul { .. });
            let mut acc_i: i64 = if product { 1 } else { 0 };
            let mut acc_f: f64 = if product { 1.0 } else { 0.0 };
            let mut any_float = false;
            for arg in args {
                match eval_num(model, vars, states, time, arg)? {
                    Num::Int(i) => {
                        if product {
                            acc_i *= i;
                            acc_f *= i as f64;
                        } else {
                            acc_i += i;
                            acc_f += i as f64;
                        }
                    }
                    Num::Float(f) => {
                        any_float = true;
                        if product {
                            acc_f *= f;
                        } else {
                            acc_f += f;
                        }
                    }
                }
            }
            if any_float {
                Ok(Value::Float(acc_f))
            } else {
                Ok(Value::Int(acc_i))
            }
        }
        CExpr::Sub { lhs, rhs } => {
            let lhs = eval_num(model, vars, states, time, lhs)?;
            let rhs = eval_num(model, vars, states, time, rhs)?;
            Ok(match (lhs, rhs) {
                (Num::Int(a), Num::Int(b)) => Value::Int(a - b),
                (a, b) => Value::Float(a.as_f64() - b.as_f64()),
            })
        }
        CExpr::Div { lhs, rhs } => {
            let lhs = eval_num(model, vars, states, time, lhs)?.as_f64();
            let rhs = eval_num(model, vars, states, time, rhs)?.as_f64();
            // IEEE semantics (±inf on zero divisor); NaN is caught by
            // comparisons and the integrator's finiteness checks.
            Ok(Value::Float(lhs / rhs))
        }
        CExpr::Min { args } | CExpr::Max { args } => {
            let take_min = matches!(expr, CExpr::Min { .. });
            let mut best: Option<Num> = None;
            for arg in args {
                let value = eval_num(model, vars, states, time, arg)?;
                best = Some(match best {
                    None => value,
                    Some(current) => {
                        let replace = if take_min {
                            value.as_f64() < current.as_f64()
                        } else {
                            value.as_f64() > current.as_f64()
                        };
                        if replace {
                            value
                        } else {
                            current
                        }
                    }
                });
            }
            // Arity ≥ 1 validated at model build.
            Ok(best.map_or(Value::Int(0), Num::into_value))
        }
        CExpr::If {
            cond,
            then,
            otherwise,
        } => {
            if eval_bool(model, vars, states, time, cond)? {
                eval_expr(model, vars, states, time, then)
            } else {
                eval_expr(model, vars, states, time, otherwise)
            }
        }
        CExpr::Sin(arg) => Ok(Value::Float(
            eval_num(model, vars, states, time, arg)?.as_f64().sin(),
        )),
        CExpr::Exp(arg) => Ok(Value::Float(
            eval_num(model, vars, states, time, arg)?.as_f64().exp(),
        )),
        CExpr::Time => Ok(Value::Float(time)),
    }
}

/// Numeric intermediate for arithmetic evaluation.
#[derive(Debug, Clone, Copy)]
enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    fn as_f64(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }
    fn into_value(self) -> Value {
        match self {
            Num::Int(i) => Value::Int(i),
            Num::Float(f) => Value::Float(f),
        }
    }
}

fn eval_num(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
) -> Result<Num, EngineError> {
    match eval_expr(model, vars, states, time, expr)? {
        Value::Int(i) => Ok(Num::Int(i)),
        Value::Float(f) => Ok(Num::Float(f)),
        Value::Bool(_) => Err(EngineError::TypeError {
            time,
            detail: "arithmetic on a boolean value".to_owned(),
        }),
    }
}

fn eval_bool(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
) -> Result<bool, EngineError> {
    match eval_expr(model, vars, states, time, expr)? {
        Value::Bool(b) => Ok(b),
        other => Err(EngineError::TypeError {
            time,
            detail: format!("expected a boolean, got {other:?}"),
        }),
    }
}

fn eval_f64(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
) -> Result<f64, EngineError> {
    Ok(eval_num(model, vars, states, time, expr)?.as_f64())
}

fn eval_agg(
    model: &CompiledModel,
    vars: &[Value],
    time: f64,
    sources: &[VarIdx],
    agg: AggOp,
) -> Result<Value, EngineError> {
    match agg {
        AggOp::Count => Ok(Value::Int(sources.len() as i64)),
        AggOp::Sum => {
            let mut int_sum = 0i64;
            let mut float_sum = 0.0f64;
            let mut any_float = false;
            for &idx in sources {
                match vars[idx] {
                    Value::Int(i) => int_sum += i,
                    Value::Float(f) => {
                        any_float = true;
                        float_sum += f;
                    }
                    Value::Bool(b) => int_sum += i64::from(b),
                }
            }
            if any_float {
                Ok(Value::Float(float_sum + int_sum as f64))
            } else {
                Ok(Value::Int(int_sum))
            }
        }
        AggOp::All | AggOp::Any => {
            let mut all = true;
            let mut any = false;
            for &idx in sources {
                match vars[idx] {
                    Value::Bool(b) => {
                        all &= b;
                        any |= b;
                    }
                    other => {
                        return Err(EngineError::TypeError {
                            time,
                            detail: format!(
                                "boolean aggregation over non-boolean value {other:?} \
                                 (attribute `{}`)",
                                model.var_names[idx]
                            ),
                        });
                    }
                }
            }
            Ok(Value::Bool(if agg == AggOp::All { all } else { any }))
        }
        AggOp::Mean | AggOp::Median => {
            let mut values = Vec::with_capacity(sources.len());
            for &idx in sources {
                values.push(match vars[idx] {
                    Value::Int(i) => i as f64,
                    Value::Float(f) => f,
                    Value::Bool(b) => f64::from(u8::from(b)),
                });
            }
            if values.is_empty() {
                return Ok(Value::Float(0.0));
            }
            if agg == AggOp::Mean {
                let n = values.len() as f64;
                Ok(Value::Float(values.iter().sum::<f64>() / n))
            } else {
                values.sort_by(f64::total_cmp);
                let mid = values.len() / 2;
                Ok(Value::Float(if values.len() % 2 == 1 {
                    values[mid]
                } else {
                    0.5 * (values[mid - 1] + values[mid])
                }))
            }
        }
    }
}

fn eval_cmp(time: f64, op: CmpOp, lhs: Value, rhs: Value) -> Result<Value, EngineError> {
    let ordering = match (lhs, rhs) {
        (Value::Bool(a), Value::Bool(b)) => {
            return match op {
                CmpOp::Eq => Ok(Value::Bool(a == b)),
                CmpOp::Ne => Ok(Value::Bool(a != b)),
                _ => Err(EngineError::TypeError {
                    time,
                    detail: format!("ordering comparison {op:?} on booleans"),
                }),
            };
        }
        (Value::Int(a), Value::Int(b)) => a.partial_cmp(&b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(&b),
        (Value::Int(a), Value::Float(b)) => (a as f64).partial_cmp(&b),
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(b as f64)),
        (a, b) => {
            return Err(EngineError::TypeError {
                time,
                detail: format!("comparison between incompatible kinds {a:?} and {b:?}"),
            });
        }
    };
    let Some(ordering) = ordering else {
        return Err(EngineError::TypeError {
            time,
            detail: "comparison involving NaN".to_owned(),
        });
    };
    let result = match op {
        CmpOp::Eq => ordering.is_eq(),
        CmpOp::Ne => !ordering.is_eq(),
        CmpOp::Lt => ordering.is_lt(),
        CmpOp::Le => ordering.is_le(),
        CmpOp::Gt => ordering.is_gt(),
        CmpOp::Ge => ordering.is_ge(),
    };
    Ok(Value::Bool(result))
}

/// Reusable scratch of the explicit sweep: the demand and allocation
/// vectors of the conservative distribution operators, and their capping
/// marks.
///
/// The sweep runs on every solver stage, so the buffers are owned by the
/// caller and reused: after the first pass the operator allocates nothing.
/// This is scratch, not trajectory state, so it stays out of the snapshot.
#[derive(Debug, Clone, Default)]
struct FlowScratch {
    demands: Vec<f64>,
    allocated: Vec<f64>,
    capped: Vec<bool>,
    /// When present, every distribution operator run by the sweep appends
    /// its edge classes here, in sweep order: this is how the active set
    /// is read off **the operator's own capping outcome** instead of being
    /// recomputed by a second, parallel rule that could disagree with it.
    ///
    /// Set only by the boundary resolution. The per-stage path leaves it
    /// `None` and pays nothing.
    classes: Option<Vec<EdgeClass>>,
}

/// Read an attribute as a number, refusing a boolean where a quantity is
/// expected (a distribution operator moves quantities, not flags).
fn quantity(
    model: &CompiledModel,
    vars: &[Value],
    time: f64,
    var: VarIdx,
) -> Result<f64, EngineError> {
    match vars[var] {
        Value::Float(value) => Ok(value),
        Value::Int(value) => Ok(value as f64),
        Value::Bool(_) => Err(EngineError::TypeError {
            time,
            detail: format!(
                "`{}` carries a boolean where a distributed quantity is expected",
                model.var_names[var]
            ),
        }),
    }
}

/// Guard one input of a distribution operator: a non-finite quantity is a
/// loud failure, a negative one is not a quantity and distributes nothing.
///
/// The clamp is deliberate and narrow. A level or a rate can pass through
/// zero during an integration segment and land a few ulps below it; that
/// is rounding, not a negative demand, and aborting a run on it would make
/// every conservative flow fragile at exactly the operating point where
/// it matters. A NaN or an infinity, on the other hand, means the model
/// itself produced no number, and is reported.
fn flow_input(value: f64, time: f64, operator: &str, role: &str) -> Result<f64, EngineError> {
    if !value.is_finite() {
        return Err(EngineError::TypeError {
            time,
            detail: format!("the {role} of distribution operator `{operator}` is {value}"),
        });
    }
    Ok(value.max(0.0))
}

/// The mutable side-channels of one explicit sweep, carried together
/// because every caller holds all three and none of them belongs to the
/// state the sweep computes: what the pass **counts**
/// ([`WorkCounters`]), what it **scribbles on** ([`FlowScratch`]), and
/// what it **reports as moved** ([`ChangeLog`]).
struct PassContext<'a> {
    /// Counted work of the pass.
    work: &'a mut WorkCounters,
    /// Reused buffers of the distribution operators.
    scratch: &'a mut FlowScratch,
    /// Change detection of the pass's writes.
    changed: &'a mut ChangeLog,
}

/// Run one conservative distribution operator: read the available
/// quantity and the per-connection demands, split them under the compiled
/// policy, and write one quantity per outgoing connection.
fn run_allocation(
    model: &CompiledModel,
    vars: &mut [Value],
    states: &[StateIdx],
    time: f64,
    allocation: &CAllocation,
    ctx: &mut PassContext<'_>,
) -> Result<(), EngineError> {
    let PassContext {
        work,
        scratch,
        changed,
    } = ctx;
    let raw = eval_f64(model, vars, states, time, &allocation.available)?;
    let available = flow_input(raw, time, &allocation.name, "available quantity")?;
    scratch.demands.clear();
    for &var in &allocation.demands {
        let demand = quantity(model, vars, time, var)?;
        scratch
            .demands
            .push(flow_input(demand, time, &allocation.name, "demand")?);
    }
    scratch.allocated.clear();
    scratch.allocated.resize(scratch.demands.len(), 0.0);
    work.allocation_capping_passes += allocate(
        &allocation.policy,
        available,
        &scratch.demands,
        &mut scratch.allocated,
        &mut scratch.capped,
    );
    if scratch.classes.is_some() {
        // Split the borrow: `classify` reads the three buffers the
        // operator just wrote and appends to the fourth.
        let FlowScratch {
            demands,
            allocated,
            capped,
            classes,
        } = scratch;
        if let Some(sink) = classes.as_mut() {
            classify(&allocation.policy, demands, allocated, capped, sink);
        }
    }
    for (&target, &value) in allocation.allocated.iter().zip(&scratch.allocated) {
        changed.write(vars, target, Value::Float(value));
    }
    Ok(())
}

/// Read one operator's inputs on the current state: the available
/// quantity (returned) and one demand per edge (into `demands`), both
/// through the same guards the operator itself applies, so a margin never
/// sees a quantity the operator would have refused or clamped.
fn read_flow_inputs(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    allocation: &CAllocation,
    demands: &mut Vec<f64>,
) -> Result<f64, EngineError> {
    let raw = eval_f64(model, vars, states, time, &allocation.available)?;
    let available = flow_input(raw, time, &allocation.name, "available quantity")?;
    demands.clear();
    for &var in &allocation.demands {
        let demand = quantity(model, vars, time, var)?;
        demands.push(flow_input(demand, time, &allocation.name, "demand")?);
    }
    Ok(available)
}

/// **Change detection of the explicit sweep**: every write goes through
/// here, and the ones that actually move a value are reported.
///
/// The sweep rewrites each of its targets at every evaluation point,
/// whether or not the new value differs from the one already there. That
/// is fine as an *assignment*, and useless as a *signal*: an index that
/// tells the engine which margins a change reaches is worth nothing if
/// every attribute is announced as changed on every pass. Comparing
/// before writing is what turns the rewrite into a signal.
///
/// **The comparison is unconditional**, paid by every model, including
/// the discrete-only ones that carry no watched transition and receive
/// nothing in exchange. Only the *recording* is optional: the passes
/// whose result is thrown away (the solver callbacks, the dense-sample
/// callback, the active-set probe) run against a copied attribute vector
/// and have no state to invalidate.
///
/// Floats are compared at the **flow tolerance**, the same number the
/// flow resolution settles to, so a quantity still creeping inside the
/// band the resolution has already declared settled is not announced as a
/// move. Everything else compares exactly.
struct ChangeLog {
    /// The band a float write must leave to count as a move
    /// ([`FlowConfig::tolerance`]).
    tolerance: f64,
    /// Targets that moved, in write order, or `None` on a pass whose
    /// result is discarded.
    moved: Option<Vec<VarIdx>>,
}

impl ChangeLog {
    /// A log that compares but records nothing: the passes run on a copy
    /// of the attribute vector.
    fn discarding(tolerance: f64) -> Self {
        ChangeLog {
            tolerance,
            moved: None,
        }
    }

    /// Write one target, reporting it when the value moved.
    ///
    /// Two tests, cheapest first. **Identity** settles the common case:
    /// a sweep of a settled network recomputes most of its targets to the
    /// bits they already held, and that is one comparison. Only a genuine
    /// difference is worth the banded test, which costs a handful of
    /// operations and is where the tolerance enters.
    #[inline]
    fn write(&mut self, vars: &mut [Value], target: VarIdx, value: Value) {
        let old = vars[target];
        vars[target] = value;
        if old == value {
            return;
        }
        if !value_settled(&old, &value, self.tolerance) {
            if let Some(sink) = self.moved.as_mut() {
                sink.push(target);
            }
        }
    }
}

/// Run the explicit sweep (equations and distribution operators, in table
/// order) into `vars`, counting the pass into `work` (see
/// [`WorkCounters`]) and reporting the targets that moved into `changed`
/// (see [`ChangeLog`]).
fn recompute_explicit(
    model: &CompiledModel,
    vars: &mut [Value],
    states: &[StateIdx],
    time: f64,
    ctx: &mut PassContext<'_>,
) -> Result<(), EngineError> {
    ctx.work.explicit_evaluations += 1;
    for step in &model.explicit {
        match step {
            CStep::Equation { target, expr } => {
                let value = eval_f64(model, vars, states, time, expr)?;
                ctx.changed.write(vars, *target, Value::Float(value));
            }
            CStep::Allocate(allocation) => {
                run_allocation(model, vars, states, time, allocation, ctx)?;
            }
        }
    }
    Ok(())
}

/// Record which branch of every comparison the sweep resolves, into a
/// vector whose *equality* between two sweeps is half the active-set
/// termination test (the other half being the operators' capping
/// outcome).
///
/// Minima and maxima contribute the index of the argument they select,
/// conditionals the branch they take. Both are finite, categorical
/// answers: settling them to exact equality turns most of a resolution
/// from an asymptotic question into a combinatorial one, which is why
/// they are tested before the flows are tested to a tolerance.
///
/// Only the **taken** branch of a conditional is walked. The other one is
/// not part of the answer, and evaluating it could fail on a state the
/// model never intended it to see.
fn record_branches(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
    into: &mut Vec<u32>,
) -> Result<(), EngineError> {
    match expr {
        CExpr::Min { args } | CExpr::Max { args } => {
            let wants_min = matches!(expr, CExpr::Min { .. });
            let mut best: Option<(usize, f64)> = None;
            for (index, arg) in args.iter().enumerate() {
                let value = eval_f64(model, vars, states, time, arg)?;
                let better = match best {
                    None => true,
                    Some((_, incumbent)) => {
                        if wants_min {
                            value < incumbent
                        } else {
                            value > incumbent
                        }
                    }
                };
                if better {
                    best = Some((index, value));
                }
            }
            // An empty min/max selects nothing; `u32::MAX` is that
            // answer, and it is as stable as any other.
            into.push(best.map_or(u32::MAX, |(index, _)| index as u32));
            for arg in args {
                record_branches(model, vars, states, time, arg, into)?;
            }
        }
        CExpr::If {
            cond,
            then,
            otherwise,
        } => {
            let taken = eval_bool(model, vars, states, time, cond)?;
            into.push(u32::from(taken));
            record_branches(model, vars, states, time, cond, into)?;
            record_branches(
                model,
                vars,
                states,
                time,
                if taken { then } else { otherwise },
                into,
            )?;
        }
        CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
            record_branches(model, vars, states, time, lhs, into)?;
            record_branches(model, vars, states, time, rhs, into)?;
        }
        CExpr::Bool { args, .. } | CExpr::Add { args } | CExpr::Mul { args } => {
            for arg in args {
                record_branches(model, vars, states, time, arg, into)?;
            }
        }
        CExpr::Sin(arg) | CExpr::Exp(arg) => {
            record_branches(model, vars, states, time, arg, into)?;
        }
        CExpr::Const(_)
        | CExpr::Var(_)
        | CExpr::StateActive { .. }
        | CExpr::PortAgg { .. }
        | CExpr::Time => {}
    }
    Ok(())
}

/// The **active set** of a resolved network: which edges are saturated
/// (read off each operator's capping outcome) and which branch of each
/// comparison the sweep took.
///
/// Two sweeps that produce equal signatures have settled the
/// combinatorial half of the resolution; what remains is numeric and is
/// settled against [`FLOW_TOLERANCE`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ActiveSet {
    edges: Vec<EdgeClass>,
    branches: Vec<u32>,
}

/// How one integration segment ended.
///
/// Three outcomes, not two: besides reaching the requested date and
/// locating a watched transition, a segment can end because the **active
/// set** it froze stopped holding. That third outcome fires no
/// transition; it re-resolves the network at the crossing instant and
/// integration continues from there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Segment {
    /// The requested date was reached.
    Reached,
    /// A watched boundary (or a continuously-varying hazard) was
    /// located: this transition must fire.
    Watched(TransIdx),
    /// An active-set boundary was located and the network was resolved
    /// again from the crossing state. Carries the global edge index that
    /// crossed, so a chattering boundary can be *named* rather than
    /// merely counted.
    Resolved(usize),
}

/// Sweeps the **numeric** level of one resolution may spend once its
/// active set has settled: the constant half of the two-level budget of
/// [`Engine::resolve_flows`].
///
/// It is a constant because nothing in the compiled network sizes it: the
/// active set is finite and its budget counts edges, whereas the
/// quantities converge at a rate that is a property of the model's
/// arithmetic, not of its size.
///
/// **Why 64.** The undamped iteration reaches the per-edge tolerance in a
/// handful of sweeps on every settling network measured here (three on
/// the contested supply). The budget is not sized for those: it is sized
/// for the damped iteration that [`FLOW_RELAXATION`] substitutes after a
/// two-cycle, whose residual then falls by a factor `q` per sweep. From
/// an initial residual of order one, reaching [`FLOW_TOLERANCE`] costs
/// `ln(1e-9) / ln(q)` sweeps: 30 at `q = 1/2`, 58 at `q = 0.7`. Sixty-four
/// therefore covers every damped contraction up to about `q = 0.72` and
/// refuses the rest with a diagnostic rather than with silence.
///
/// This is the default of [`FlowConfig::sweep_budget`], which is where a
/// caller overrides it.
pub const FLOW_SWEEP_BUDGET: usize = 64;

/// Under-relaxation factor of the flow resolution: the weight given to
/// the sweep's raw answer when blending it with the iterate it started
/// from (`x ← (1 − w)·x + w·F(x)`).
///
/// **Why one half, and not the 0.9 of general practice.** The two figures
/// answer different failure modes. A factor near 0.9 damps a *diverging
/// monotone* iteration, whose linearised multiplier `μ` is above 1: there
/// the aim is to shave the overshoot while keeping most of the step, and
/// taking `w` far below 1 would only make a convergent case crawl. Here
/// the relaxation is engaged **only** in response to an observed
/// two-cycle, whose multiplier sits near −1. The damped multiplier is
/// `|1 − w(1 − μ)|`, which at `μ = −1` is `|1 − 2w|`: minimal, and zero,
/// at `w = 1/2`. The same point under `w = 0.9` leaves `0.8` per sweep,
/// about 95 sweeps to [`FLOW_TOLERANCE`], which no constant budget of the
/// size above could hold.
///
/// The relaxation is **not** applied from the first sweep. The cold-start
/// sequence of [`Engine::resolve_flows`] descends, and damping a
/// descending sequence buys nothing while costing every well-behaved
/// network a factor on its sweep count. It is latched on the first
/// two-cycle detection and never released within a resolution; it is a
/// local of the resolution, so nothing carries it across a segment.
///
/// This is the default of [`FlowConfig::relaxation`], which is where a
/// caller overrides it.
pub const FLOW_RELAXATION: f64 = 0.5;

/// Sweeps the **combinatorial** level of one resolution may spend, and
/// segment restarts one instant may absorb: the derived half of the
/// two-level budget.
///
/// Derived, not chosen. The descending cold-start sequence saturates at
/// least one more edge per round, so a search that is still changing the
/// saturation pattern after one round per compiled edge is no longer
/// searching. **Two** rounds per edge rather than one, because an edge
/// has three classes under a priority order
/// ([`EdgeClass::Unserved`] → [`EdgeClass::Partial`] → [`EdgeClass::Full`])
/// and therefore two class changes to spend along that descent. Branch
/// decisions (a limiting minimum, a conditional) enter the same pattern
/// and are counted alongside the edges, since a resolution can equally be
/// held up by a minimum that keeps swapping its limiting argument. Two
/// more rounds cover the sweep that produces the first candidate and the
/// sweep that confirms it.
///
/// The same figure bounds the segment restarts of
/// [`Engine::advance_continuous`] at one instant, for the same reason: at
/// most that many distinct class changes can be located there before the
/// boundary is chattering rather than moving.
///
/// This derivation is the source of the combinatorial budget unless
/// [`FlowConfig::active_set_budget`] overrides it.
#[must_use]
pub fn active_set_budget(model: &CompiledModel) -> usize {
    let mut budget = 2usize;
    for step in &model.explicit {
        match step {
            CStep::Equation { expr, .. } => budget += decision_sites(expr),
            CStep::Allocate(allocation) => {
                budget += 2 * allocation.allocated.len() + decision_sites(&allocation.available);
            }
        }
    }
    budget
}

/// Number of **branch decisions** an expression can contribute to the
/// active set: one per minimum, maximum and conditional, counted over
/// every branch (the sweep walks only the taken one, so this is an upper
/// bound, which is what a budget wants).
fn decision_sites(expr: &CExpr) -> usize {
    match expr {
        CExpr::Min { args } | CExpr::Max { args } => {
            1 + args.iter().map(decision_sites).sum::<usize>()
        }
        CExpr::If {
            cond,
            then,
            otherwise,
        } => 1 + decision_sites(cond) + decision_sites(then) + decision_sites(otherwise),
        CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
            decision_sites(lhs) + decision_sites(rhs)
        }
        CExpr::Bool { args, .. } | CExpr::Add { args } | CExpr::Mul { args } => {
            args.iter().map(decision_sites).sum()
        }
        CExpr::Sin(arg) | CExpr::Exp(arg) => decision_sites(arg),
        CExpr::Const(_)
        | CExpr::Var(_)
        | CExpr::StateActive { .. }
        | CExpr::PortAgg { .. }
        | CExpr::Time => 0,
    }
}

/// Whether one sweep moved every quantity by less than the per-edge flow
/// tolerance: the numeric half of the resolution's stopping test, applied
/// only once the combinatorial half has settled.
///
/// Mixed relative-and-absolute, so a network carrying large quantities is
/// held to the same meaning as one carrying small ones. A non-float
/// attribute compares exactly: nothing in the sweep writes one, so an
/// inequality there is a change, not a residual.
///
/// `tolerance` is [`FlowConfig::tolerance`], passed rather than read from
/// the constant so the stopping test and the margin dead band of the same
/// run are always the same number.
fn flows_settled(before: &[Value], after: &[Value], tolerance: f64) -> bool {
    before
        .iter()
        .zip(after)
        .all(|(old, new)| value_settled(old, new, tolerance))
}

/// One attribute's half of [`flows_settled`]: also the test that decides
/// whether an edge "moved" in the sweep the diagnostic reports on, so
/// that what stops the resolution and what the diagnostic names are the
/// same measure.
fn value_settled(old: &Value, new: &Value, tolerance: f64) -> bool {
    match (old, new) {
        (Value::Float(old), Value::Float(new)) => {
            (old - new).abs() <= tolerance * old.abs().max(new.abs()).max(1.0)
        }
        _ => old == new,
    }
}

/// One operator's active set, frozen for the duration of a segment.
///
/// The combinatorial search that produced `classes` ran once, at the
/// segment boundary. Inside the segment only the *quantities* move, which
/// is what keeps the solver's right-hand side a function of the state and
/// keeps an accepted and a rejected trial of the same step from doing
/// different amounts of work.
struct FrozenFlow<'m> {
    /// The operator, borrowed from the compiled sweep.
    allocation: &'m CAllocation,
    /// Where its margins are registered (naming, dependencies).
    margins: &'m CFlowMargins,
    /// Saturation class per edge, in connection declaration order.
    classes: Vec<EdgeClass>,
}

/// Adapter exposing the compiled continuous section to `raichu-numeric`.
/// Errors raised inside the solver callbacks are stashed and re-raised
/// after integration (the trait is infallible by design).
struct ContinuousSystem<'m> {
    model: &'m CompiledModel,
    vars: Vec<Value>,
    states: Vec<StateIdx>,
    /// Active watched transitions monitored this segment.
    margins: Vec<TransIdx>,
    /// Active continuously-varying hazards monitored this segment:
    /// `(transition, remaining threshold E − H)`. Each occupies one
    /// auxiliary state slot after the ODE attributes, integrating
    /// `dH/dt = λ(x)`; the firing is the event `H − (E − H₀) = 0`,
    /// located exactly like a watched boundary crossing (`reschedule_modifiable`
    /// under continuous evolution).
    hazards: Vec<(TransIdx, f64)>,
    /// The **frozen active set** of this segment: one entry per
    /// distribution operator, each carrying the saturation class of every
    /// edge as it was settled at the segment boundary. Their margins ride
    /// alongside the watched ones, so the instant the frozen pattern
    /// stops holding is *located*, not noticed at the next discrete date.
    flows: Vec<FrozenFlow<'m>>,
    error: Option<EngineError>,
    /// Work done inside the solver callbacks, merged back into the
    /// engine's counters when the segment returns.
    work: WorkCounters,
    /// Scratch of the distribution operators run by the explicit sweep
    /// inside those callbacks.
    scratch: FlowScratch,
    /// Scratch holding one operator's demands while its active-set
    /// margins are evaluated (reused: the event callback runs on every
    /// interior scan point and every bisection step).
    flow_demands: Vec<f64>,
    /// The run's flow tolerance ([`FlowConfig::tolerance`]): the dead
    /// band of every active-set margin evaluated here. Copied from the
    /// config at segment start, so the band this segment applies and the
    /// tolerance the resolution that opened it settled to are the same
    /// number.
    flow_tolerance: f64,
}

impl ContinuousSystem<'_> {
    fn load(&mut self, t: f64, y: &[f64]) {
        for (slot, (var, _)) in self.model.ode.iter().enumerate() {
            self.vars[*var] = Value::Float(y[slot]);
        }
        if self.error.is_none() {
            // Nothing outside this callback survives the segment, so the
            // pass compares but records nothing.
            let mut changed = ChangeLog::discarding(self.flow_tolerance);
            let mut ctx = PassContext {
                work: &mut self.work,
                scratch: &mut self.scratch,
                changed: &mut changed,
            };
            if let Err(error) =
                recompute_explicit(self.model, &mut self.vars, &self.states, t, &mut ctx)
            {
                self.error = Some(error);
            }
        }
    }
}

impl OdeSystem for ContinuousSystem<'_> {
    fn dim(&self) -> usize {
        self.model.ode.len() + self.hazards.len()
    }

    fn rhs(&mut self, t: f64, y: &[f64], dydt: &mut [f64]) {
        self.load(t, y);
        for (slot, (_, expr)) in self.model.ode.iter().enumerate() {
            match eval_f64(self.model, &self.vars, &self.states, t, expr) {
                Ok(value) => dydt[slot] = value,
                Err(error) => {
                    self.error.get_or_insert(error);
                    dydt[slot] = 0.0;
                }
            }
        }
        let ode_len = self.model.ode.len();
        for (slot, (trans_idx, _)) in self.hazards.iter().enumerate() {
            let CLaw::ExpVar { rate, .. } = &self.model.transitions[*trans_idx].distrib else {
                dydt[ode_len + slot] = 0.0;
                continue;
            };
            match eval_f64(self.model, &self.vars, &self.states, t, rate) {
                Ok(lambda) if lambda.is_finite() && lambda >= 0.0 => {
                    dydt[ode_len + slot] = lambda;
                }
                Ok(lambda) => {
                    self.error.get_or_insert(EngineError::TypeError {
                        time: t,
                        detail: format!(
                            "state-dependent rate of `{}` evaluated to {lambda} \
                             (must be finite and >= 0)",
                            self.model.transitions[*trans_idx].name
                        ),
                    });
                    dydt[ode_len + slot] = 0.0;
                }
                Err(error) => {
                    self.error.get_or_insert(error);
                    dydt[ode_len + slot] = 0.0;
                }
            }
        }
    }

    fn n_events(&self) -> usize {
        self.margins.len()
            + self.hazards.len()
            + self.flows.iter().map(|f| f.classes.len()).sum::<usize>()
    }

    fn events(&mut self, t: f64, y: &[f64], out: &mut [f64]) {
        self.load(t, y);
        self.work.margin_evaluations += self.margins.len() as u64;
        for (slot, trans_idx) in self.margins.iter().enumerate() {
            let CLaw::Watched { margin } = &self.model.transitions[*trans_idx].distrib else {
                out[slot] = -1.0;
                continue;
            };
            match eval_f64(self.model, &self.vars, &self.states, t, margin) {
                Ok(value) => out[slot] = value,
                Err(error) => {
                    self.error.get_or_insert(error);
                    out[slot] = -1.0;
                }
            }
        }
        let (n_margins, ode_len) = (self.margins.len(), self.model.ode.len());
        for (slot, (_, remaining)) in self.hazards.iter().enumerate() {
            out[n_margins + slot] = y[ode_len + slot] - remaining;
        }
        // Active-set margins. They are watched guards like the ones
        // above: the segment ends where the frozen saturation pattern
        // stops holding, and the crossing instant is bisected to the same
        // event tolerance.
        let mut slot = n_margins + self.hazards.len();
        for flow in &self.flows {
            self.work.margin_evaluations += flow.classes.len() as u64;
            let inputs = read_flow_inputs(
                self.model,
                &self.vars,
                &self.states,
                t,
                flow.allocation,
                &mut self.flow_demands,
            );
            let available = match inputs {
                Ok(available) => available,
                Err(error) => {
                    self.error.get_or_insert(error);
                    for edge in 0..flow.classes.len() {
                        out[slot + edge] = -1.0;
                    }
                    slot += flow.classes.len();
                    continue;
                }
            };
            let band = flow_band(
                self.flow_demands
                    .iter()
                    .fold(available.abs(), |scale, d| scale.max(d.abs())),
                self.flow_tolerance,
            );
            for edge in 0..flow.classes.len() {
                out[slot + edge] = edge_margin(
                    &flow.allocation.policy,
                    available,
                    &self.flow_demands,
                    &flow.classes,
                    edge,
                    band,
                );
            }
            slot += flow.classes.len();
        }
    }
}

/// A simulation engine over a compiled model.
///
/// **Not a singleton**: any number
/// of engines can coexist in one process; every piece of state lives in
/// this struct.
pub struct Engine<'m> {
    model: &'m CompiledModel,
    config: EngineConfig,
    solver: Box<dyn OdeSolver>,
    time: f64,
    vars: Vec<Value>,
    states: Vec<StateIdx>,
    /// Pending firing date per transition (`None` = not scheduled;
    /// watched transitions are monitored, never date-scheduled).
    pending: Vec<Option<f64>>,
    /// Remaining countdown of paused transitions
    /// (`on_interruption: resume` only).
    frozen: Vec<Option<f64>>,
    /// Cumulative-hazard state per transition (`CLaw::ExpVar` only;
    /// survives a `resume` pause, cleared by `reset`/firing/exit).
    hazards: Vec<Option<Hazard>>,
    /// Transitions whose state-dependent rate varies continuously
    /// (monitored during `integrate_to`, like watched boundaries).
    continuous_rates: Vec<TransIdx>,
    events: Vec<Event>,
    journal: Vec<JournalRecord>,
    /// Ordered monitored-state entries recorded this trajectory (sequence
    /// analysis; empty unless `config.sequences`).
    seq_events: Vec<SeqEvent>,
    /// The reached target `(end_cause, end_time)` once one activates: set
    /// once, triggers the trajectory early-stop.
    seq_end: Option<(String, f64)>,
    indicator_series: Vec<IndicatorSeries>,
    sampled: Vec<IndicatorSeries>,
    sample_cursor: usize,
    /// Consecutive watched firings without time advancing (Zeno guard).
    watched_streak: (f64, usize),
    /// Replica generator (master seed + substream; `schedule_stochastic` draws).
    rng: ChaCha8Rng,
    /// Whether the model carries any stochastic distribution (provenance).
    stochastic: bool,
    /// Scratch worklist for the fixpoint (reused across steps: no
    /// allocation in the hot loop once warmed up).
    worklist: BTreeSet<FnIdx>,
    /// Counted work (see [`WorkCounters`]): cumulative instrumentation
    /// over this engine's life, deliberately outside the snapshot.
    work: WorkCounters,
    /// Scratch of the explicit sweep's distribution operators (see
    /// [`FlowScratch`]): reused, never part of the trajectory.
    flow_scratch: FlowScratch,
    /// Combinatorial half of the flow convergence policy, resolved once
    /// at construction: the config's override
    /// ([`FlowConfig::active_set_budget`]) when it carries one, else the
    /// derivation from the compiled network ([`active_set_budget`]).
    /// Constant for the engine's life: it describes the model and the
    /// configuration, not the trajectory, so it stays out of the
    /// snapshot.
    active_set_budget: usize,
    /// Where the engine's own explicit passes report the attributes they
    /// moved (see [`ChangeLog`]). Drained into the stale marks of the
    /// watched population; the buffer itself is scratch, reused.
    changed: ChangeLog,
    /// **Indexed watched set**, arming half: whether each position of
    /// `model.watched` sits in its source state. Maintained on state
    /// changes through [`crate::MarginIndex::watched_by_owner`], never rebuilt
    /// by a full scan.
    watched_armed: Vec<bool>,
    /// **Indexed watched set**, ascending list of the armed positions:
    /// the per-segment margin set, and the population the immediate-guard
    /// scan walks. Ascending by construction, which is what preserves the
    /// documented firing order of simultaneous crossings.
    watched_active: Vec<WatchedIdx>,
    /// **Indexed watched set**, cache half: the last evaluated verdict of
    /// each watched guard.
    watched_guard: Vec<bool>,
    /// **Indexed watched set**, invalidation half: whether the cached
    /// verdict of each position still reflects the state. Set through
    /// [`crate::MarginIndex`] whenever an attribute, an automaton state or the
    /// clock a guard reads moves; cleared when the guard is re-evaluated.
    watched_stale: Vec<bool>,
}

impl<'m> Engine<'m> {
    /// Build and initialise an engine with the default ODE backend
    /// (Dormand-Prince 4(5), parameters from the config).
    pub fn new(model: &'m CompiledModel, config: EngineConfig) -> Result<Self, EngineError> {
        let solver = Box::new(DormandPrince45::new(config.ode.clone()));
        Self::with_solver(model, config, solver)
    }

    /// Build an engine with an explicit ODE backend (the trait is the
    /// swap point: see `raichu-numeric`).
    pub fn with_solver(
        model: &'m CompiledModel,
        config: EngineConfig,
        solver: Box<dyn OdeSolver>,
    ) -> Result<Self, EngineError> {
        let mut engine = Self::bare(model, config, solver);
        engine.initialize()?;
        Ok(engine)
    }

    /// Rebuild an engine positioned at a previously captured
    /// [`Snapshot`], **skipping** the initialization axiom (the snapshot
    /// already carries a valid, possibly-advanced state).
    ///
    /// This is the seam a stateful facade uses when it cannot hold the
    /// borrowing [`Engine`] across calls (e.g. the Python `interactive`
    /// object): it keeps the owned model + a `Snapshot`, and rebuilds a
    /// throwaway engine on each call. Restores are exact, so a run
    /// driven this way is identical to one driven on a persistent engine.
    pub fn from_snapshot(
        model: &'m CompiledModel,
        config: EngineConfig,
        snapshot: &Snapshot,
    ) -> Self {
        let solver = Box::new(DormandPrince45::new(config.ode.clone()));
        let mut engine = Self::bare(model, config, solver);
        engine.restore(snapshot);
        engine
    }

    /// Construct the engine struct with its pristine pre-initialization
    /// field values (no fixpoint, no schedule yet). Shared by
    /// [`Engine::with_solver`] and [`Engine::from_snapshot`].
    fn bare(model: &'m CompiledModel, config: EngineConfig, solver: Box<dyn OdeSolver>) -> Self {
        let stochastic = model.transitions.iter().any(|t| {
            matches!(
                t.distrib,
                CLaw::Exp(_)
                    | CLaw::ExpVar { .. }
                    | CLaw::Weibull(..)
                    | CLaw::Lognormal(..)
                    | CLaw::Gamma(..)
                    | CLaw::Uniform(..)
                    | CLaw::Empirical(_)
            )
            // A genuinely-branching instantaneous transition (≥ 2 positive
            // branches) draws its destination from the RNG.
            || matches!(&t.distrib, CLaw::Inst(probs)
                if probs.iter().filter(|p| **p > 0.0).count() >= 2)
        });
        let continuous_rates: Vec<TransIdx> = model
            .transitions
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                matches!(
                    t.distrib,
                    CLaw::ExpVar {
                        continuous: true,
                        ..
                    }
                )
            })
            .map(|(i, _)| i)
            .collect();
        let rng = raichu_rng::replica_rng(config.seed, config.rng_stream);
        Engine {
            time: 0.0,
            vars: model.var_init.clone(),
            states: model.automata.iter().map(|a| a.init).collect(),
            pending: vec![None; model.transitions.len()],
            frozen: vec![None; model.transitions.len()],
            hazards: vec![None; model.transitions.len()],
            continuous_rates,
            events: Vec::new(),
            journal: Vec::new(),
            seq_events: Vec::new(),
            seq_end: None,
            indicator_series: model
                .indicators
                .iter()
                .map(|i| IndicatorSeries {
                    name: i.name.clone(),
                    points: Vec::new(),
                })
                .collect(),
            sampled: model
                .indicators
                .iter()
                .map(|i| IndicatorSeries {
                    name: i.name.clone(),
                    points: Vec::new(),
                })
                .collect(),
            sample_cursor: 0,
            watched_streak: (0.0, 0),
            rng,
            stochastic,
            worklist: BTreeSet::new(),
            work: WorkCounters::default(),
            flow_scratch: FlowScratch::default(),
            active_set_budget: config
                .flow
                .active_set_budget
                .unwrap_or_else(|| active_set_budget(model)),
            changed: ChangeLog {
                tolerance: config.flow.tolerance,
                moved: Some(Vec::new()),
            },
            // Pristine: nothing armed, nothing cached, everything stale.
            // `initialize` (or `restore`) derives the real arming.
            watched_armed: vec![false; model.watched.len()],
            watched_active: Vec::new(),
            watched_guard: vec![false; model.watched.len()],
            watched_stale: vec![true; model.watched.len()],
            solver,
            model,
            config,
        }
    }

    /// Initialization axiom (Desgeorges et al. 2021): run every
    /// sensitive function once in declaration order to a fixpoint, solve
    /// the explicit equations, build the initial schedule, and record
    /// the t = 0 indicator/sample values. Shared by [`Engine::new`] and
    /// [`Engine::reset`] so a reset state is identical to a fresh build.
    fn initialize(&mut self) -> Result<(), EngineError> {
        // The state vectors have just been set wholesale (fresh build or
        // reset): derive the arming and discard every cached verdict.
        self.rebuild_watched_index();
        self.worklist.extend(0..self.model.functions.len());
        self.run_fixpoint()?;
        self.resolve_flows()?;
        self.refresh_schedule()?;
        self.record_indicators();
        // Sample instants at or before t = 0 use the initial state.
        self.flush_samples_through(0.0);
        // Sequence analysis: a target already active at initialization
        // (declared init state) ends the trajectory at t = 0.
        self.check_targets();
        Ok(())
    }

    /// Sequence analysis: label the trajectory with the first target
    /// (feared event) whose state is active: sets `seq_end` once.
    fn check_targets(&mut self) {
        if !self.config.stop_at_targets || self.seq_end.is_some() {
            return;
        }
        for target in &self.model.targets {
            if self.states[target.automaton] == target.state {
                self.seq_end = Some((target.name.clone(), self.time));
                break;
            }
        }
    }

    /// Current simulation time.
    #[must_use]
    pub fn current_time(&self) -> f64 {
        self.time
    }

    /// Counted work done so far (see [`WorkCounters`]): the
    /// machine-independent performance units of this run.
    ///
    /// Cumulative over the engine's life. [`Engine::restore`] rewinds the
    /// trajectory, not the work already done, so a rewound-and-replayed
    /// engine reports *more* work than a straight run of the same
    /// trajectory: compare counters between fresh runs.
    #[must_use]
    pub fn work(&self) -> WorkCounters {
        let solver = self.solver.stats();
        WorkCounters {
            solver_steps_accepted: solver.accepted,
            solver_steps_rejected: solver.rejected,
            ..self.work
        }
    }

    /// Value of an attribute by qualified name (`component.attribute`).
    #[must_use]
    pub fn attribute(&self, qualified: &str) -> Option<Value> {
        attribute_of(self.model, &self.vars, qualified)
    }

    /// Current state name of an automaton by qualified name
    /// (`component.automaton`).
    #[must_use]
    pub fn state(&self, qualified: &str) -> Option<&str> {
        state_of(self.model, &self.states, qualified)
    }

    /// **Interactive control**: every currently-armed transition.
    ///
    /// Lists the date-scheduled transitions (delay / inst / stochastic)
    /// with their firing date, plus the watched transitions armed in
    /// their source state (date = the current instant when their guard
    /// already holds, else `None`: the boundary being located only
    /// during continuous evolution).
    ///
    /// Sorted by date (unlocated watched last), then transition index,
    /// so the first entry is exactly what [`Engine::step`] would fire
    /// next.
    #[must_use]
    pub fn fireable(&self) -> Vec<Fireable> {
        let mut out: Vec<Fireable> = Vec::new();
        for (idx, pending) in self.pending.iter().enumerate() {
            if let Some(date) = *pending {
                out.push(Fireable {
                    index: idx,
                    transition: self.model.transitions[idx].name.clone(),
                    kind: fireable_kind(&self.model.transitions[idx].distrib),
                    date: Some(date),
                });
            }
        }
        for &idx in &self.model.watched {
            let transition = &self.model.transitions[idx];
            if self.states[transition.automaton] != transition.source {
                continue;
            }
            // Guard already true ⇒ fireable at the current instant; else
            // its boundary has not been located yet (date unknown). A
            // guard type error is treated as "not fireable now" here; it
            // resurfaces when the transition is actually stepped/fired.
            let date = self
                .is_immediate_watched(idx)
                .unwrap_or(false)
                .then_some(self.time);
            out.push(Fireable {
                index: idx,
                transition: transition.name.clone(),
                kind: FireableKind::Watched,
                date,
            });
        }
        out.sort_by(|a, b| match (a.date, b.date) {
            (Some(x), Some(y)) => x
                .partial_cmp(&y)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.index.cmp(&b.index)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.index.cmp(&b.index),
        });
        out
    }

    /// **Interactive control**: fire the armed transition carrying this
    /// qualified name (see [`Engine::fire_idx`] for the semantics).
    ///
    /// Errors with [`EngineError::UnknownTransition`] if no transition
    /// bears the name, or [`EngineError::NotFireable`] if it is not armed.
    pub fn fire_named(&mut self, name: &str) -> Result<Event, EngineError> {
        let idx = self.transition_index(name)?;
        self.fire_idx_inner(idx, None)
    }

    /// **Interactive control**: fire the armed transition `name`,
    /// **forcing** its destination branch to the state named `to`
    /// (bypassing the RNG / deterministic-branch resolution). This is
    /// what makes a non-deterministic instantaneous branching (or any
    /// stochastic branch) reproducibly testable: the outcome is chosen,
    /// not drawn.
    ///
    /// Errors with [`EngineError::ForcedTargetInvalid`] if `to` is not
    /// one of the transition's declared target states.
    pub fn fire_named_to(&mut self, name: &str, to: &str) -> Result<Event, EngineError> {
        let idx = self.transition_index(name)?;
        let forced = self.resolve_forced(idx, to)?;
        self.fire_idx_inner(idx, Some(forced))
    }

    /// **Interactive control**: fire a *chosen* armed transition by its
    /// index (the stable handle from [`Engine::fireable`]), resolving the
    /// destination the normal way. See [`Engine::fire_idx_to`] to force
    /// the branch.
    pub fn fire_idx(&mut self, trans_idx: TransIdx) -> Result<Event, EngineError> {
        self.fire_idx_inner(trans_idx, None)
    }

    /// **Interactive control**: fire a chosen armed transition by index,
    /// **forcing** its destination to the state named `to`.
    pub fn fire_idx_to(&mut self, trans_idx: TransIdx, to: &str) -> Result<Event, EngineError> {
        let forced = self.resolve_forced(trans_idx, to)?;
        self.fire_idx_inner(trans_idx, Some(forced))
    }

    /// **Interactive control**: override the scheduled firing date of an
    /// armed transition (manual date-setting). The transition must be
    /// date-scheduled (`pending`, i.e. not a watched boundary), and the
    /// new date must not lie in the past (`>=` the current time).
    ///
    /// The override sticks for delay / inst / fixed-rate transitions
    /// until they fire or leave their source state; a *state-dependent
    /// rate* transition may have its date recomputed at the next
    /// discrete step (`reschedule_modifiable`).
    pub fn set_date(&mut self, name: &str, date: f64) -> Result<(), EngineError> {
        let idx = self.transition_index(name)?;
        self.set_date_idx(idx, date)
    }

    /// **Interactive control**: override an armed transition's firing
    /// date by index (see [`Engine::set_date`]).
    pub fn set_date_idx(&mut self, trans_idx: TransIdx, date: f64) -> Result<(), EngineError> {
        let Some(transition) = self.model.transitions.get(trans_idx) else {
            return Err(EngineError::UnknownTransition {
                transition: format!("<index {trans_idx}>"),
            });
        };
        let name = transition.name.clone();
        if !date.is_finite() || date < self.time {
            return Err(EngineError::DateInPast {
                transition: name,
                date,
                time: self.time,
            });
        }
        match self.pending.get_mut(trans_idx) {
            Some(slot) if slot.is_some() => *slot = Some(date),
            _ => {
                return Err(EngineError::NotFireable {
                    transition: name,
                    time: self.time,
                })
            }
        }
        if self.config.journal {
            self.journal.push(JournalRecord::TransitionRescheduled {
                time: self.time,
                transition: name,
                firing_at: date,
            });
        }
        Ok(())
    }

    /// **Interactive control**: capture the full mutable trajectory
    /// state as an opaque [`Snapshot`] (checkpoint / undo point). Costs
    /// one clone of the state vectors; the immutable model is untouched.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            time: self.time,
            vars: self.vars.clone(),
            states: self.states.clone(),
            pending: self.pending.clone(),
            frozen: self.frozen.clone(),
            hazards: self.hazards.clone(),
            events: self.events.clone(),
            journal: self.journal.clone(),
            seq_events: self.seq_events.clone(),
            seq_end: self.seq_end.clone(),
            indicator_series: self.indicator_series.clone(),
            sampled: self.sampled.clone(),
            sample_cursor: self.sample_cursor,
            watched_streak: self.watched_streak,
            rng: self.rng.clone(),
            worklist: self.worklist.clone(),
        }
    }

    /// **Interactive control**: reinstate a previously captured
    /// [`Snapshot`] (undo). The RNG is restored too, so any continuation
    /// is bit-for-bit identical to continuing from the original point.
    pub fn restore(&mut self, snap: &Snapshot) {
        self.time = snap.time;
        self.vars = snap.vars.clone();
        self.states = snap.states.clone();
        self.pending = snap.pending.clone();
        self.frozen = snap.frozen.clone();
        self.hazards = snap.hazards.clone();
        self.events = snap.events.clone();
        self.journal = snap.journal.clone();
        self.seq_events = snap.seq_events.clone();
        self.seq_end = snap.seq_end.clone();
        self.indicator_series = snap.indicator_series.clone();
        self.sampled = snap.sampled.clone();
        self.sample_cursor = snap.sample_cursor;
        self.watched_streak = snap.watched_streak;
        self.rng = snap.rng.clone();
        self.worklist = snap.worklist.clone();
        // The indexed watched set is *derived*, never carried: rewinding
        // the state rewinds the arming and discards every cached verdict,
        // which is what keeps a replay from a restored snapshot exact.
        self.rebuild_watched_index();
    }

    /// **Interactive control**: the events fired so far, in
    /// chronological order (the same data a finished [`SimulationResult`]
    /// reports in its `events`).
    #[must_use]
    pub fn history(&self) -> &[Event] {
        &self.events
    }

    /// **Interactive control**: reset the engine to its initial state
    /// (`t = 0`), as freshly built: clears the trajectory and recorded
    /// history, re-seeds the RNG to `(seed, stream)`, and re-runs the
    /// initialization axiom. A run restarted from here is identical to a
    /// fresh [`Engine::new`].
    pub fn reset(&mut self) -> Result<(), EngineError> {
        let n = self.model.transitions.len();
        self.time = 0.0;
        self.vars = self.model.var_init.clone();
        self.states = self.model.automata.iter().map(|a| a.init).collect();
        self.pending = vec![None; n];
        self.frozen = vec![None; n];
        self.hazards = vec![None; n];
        self.events.clear();
        self.journal.clear();
        self.seq_events.clear();
        self.seq_end = None;
        for series in &mut self.indicator_series {
            series.points.clear();
        }
        for series in &mut self.sampled {
            series.points.clear();
        }
        self.sample_cursor = 0;
        self.watched_streak = (0.0, 0);
        self.rng = raichu_rng::replica_rng(self.config.seed, self.config.rng_stream);
        self.worklist.clear();
        self.initialize()
    }

    /// Fire a *chosen* armed transition (rather than the earliest one, as
    /// [`Engine::step`] does), advancing time to its scheduled date and
    /// running the discrete fixpoint. `forced` overrides the destination
    /// branch when set.
    ///
    /// - A date-scheduled transition (delay / inst / stochastic) fires at
    ///   its `pending` date; with continuous evolution the state is
    ///   integrated up to that date first, and a **watched boundary**
    ///   crossed en route fires *instead* (a forced jump cannot be
    ///   skipped: the returned event is that boundary transition, whose
    ///   branch is never forced).
    /// - A watched transition may be fired only while its guard already
    ///   holds (at the current instant).
    ///
    /// Choosing a non-earliest transition deliberately overrides the
    /// schedule: the interactive counterpart of a manually driven run.
    fn fire_idx_inner(
        &mut self,
        trans_idx: TransIdx,
        forced: Option<StateIdx>,
    ) -> Result<Event, EngineError> {
        let Some(transition) = self.model.transitions.get(trans_idx) else {
            return Err(EngineError::UnknownTransition {
                transition: format!("<index {trans_idx}>"),
            });
        };
        let name = transition.name.clone();
        if let Some(date) = self.pending.get(trans_idx).copied().flatten() {
            if !date.is_finite() {
                return Err(EngineError::NotFireable {
                    transition: name,
                    time: self.time,
                });
            }
            // Advance to the scheduled date, but never move the clock
            // backwards: an *overdue* transition (date already passed
            // because an earlier `fire_idx` skipped ahead) fires at the
            // current instant.
            let t_new = date.max(self.time);
            if self.needs_integration() && t_new > self.time {
                if let Some(watched_idx) = self.advance_continuous(t_new)? {
                    self.note_watched_firing()?;
                    return self.fire(watched_idx, None);
                }
            }
            if t_new > self.time {
                self.flush_samples_before(t_new);
            }
            self.time = t_new;
            self.note_time_change();
            self.watched_streak = (t_new, 0);
            self.fire(trans_idx, forced)
        } else if self.is_immediate_watched(trans_idx)? {
            self.note_watched_firing()?;
            self.fire(trans_idx, forced)
        } else {
            Err(EngineError::NotFireable {
                transition: name,
                time: self.time,
            })
        }
    }

    /// Resolve a qualified transition name to its index, or
    /// [`EngineError::UnknownTransition`].
    fn transition_index(&self, name: &str) -> Result<TransIdx, EngineError> {
        self.model
            .transitions
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| EngineError::UnknownTransition {
                transition: name.to_owned(),
            })
    }

    /// Resolve a forced destination *state name* to a valid branch of
    /// `trans_idx`, or [`EngineError::ForcedTargetInvalid`] if the name
    /// is unknown or not one of the transition's declared target states.
    fn resolve_forced(&self, trans_idx: TransIdx, to: &str) -> Result<StateIdx, EngineError> {
        let Some(transition) = self.model.transitions.get(trans_idx) else {
            return Err(EngineError::UnknownTransition {
                transition: format!("<index {trans_idx}>"),
            });
        };
        let automaton = &self.model.automata[transition.automaton];
        match automaton.states.iter().position(|s| s == to) {
            Some(state) if transition.targets.contains(&state) => Ok(state),
            _ => Err(EngineError::ForcedTargetInvalid {
                transition: transition.name.clone(),
                state: to.to_owned(),
            }),
        }
    }

    /// Whether `trans_idx` is a watched transition sitting in its source
    /// state with its guard already true: i.e. fireable at the current
    /// instant.
    fn is_immediate_watched(&self, trans_idx: TransIdx) -> Result<bool, EngineError> {
        let transition = &self.model.transitions[trans_idx];
        if !matches!(transition.distrib, CLaw::Watched { .. }) {
            return Ok(false);
        }
        if self.states[transition.automaton] != transition.source {
            return Ok(false);
        }
        let Some(guard) = &transition.guard else {
            return Ok(false);
        };
        eval_bool(self.model, &self.vars, &self.states, self.time, guard)
    }

    /// Fire the next transition: discrete (`fire_transition`) or watched at a
    /// located boundary crossing (`schedule_boundary`): if one occurs within the
    /// horizon.
    ///
    /// Returns the fired event, or `None` when nothing remains before
    /// `t_max`. Tie-break: earliest date first, then lowest transition
    /// index (documented deterministic order; the converged state does
    /// not depend on it for confluent models).
    pub fn step(&mut self) -> Result<Option<Event>, EngineError> {
        // Watched transition already past its boundary (initial
        // conditions or post-jump state): fires immediately.
        if let Some(trans_idx) = self.immediate_watched()? {
            self.note_watched_firing()?;
            return self.fire(trans_idx, None).map(Some);
        }

        let next_discrete = self.next_pending();
        let t_target =
            next_discrete.map_or(self.config.t_max, |(_, date)| date.min(self.config.t_max));

        if self.needs_integration() && t_target > self.time && t_target.is_finite() {
            if let Some(trans_idx) = self.advance_continuous(t_target)? {
                self.note_watched_firing()?;
                return self.fire(trans_idx, None).map(Some);
            }
        }

        match next_discrete {
            Some((trans_idx, date)) if date <= self.config.t_max => {
                // The clock never runs backwards: in a step-driven run
                // every scheduled date is ≥ the current time, so `max`
                // is a no-op; it only guards an *overdue* transition left
                // behind after an interactive `fire_idx` skipped ahead.
                let t_new = date.max(self.time);
                self.flush_samples_before(t_new);
                self.time = t_new;
                self.note_time_change();
                self.watched_streak = (t_new, 0);
                self.fire(trans_idx, None).map(Some)
            }
            _ => Ok(None),
        }
    }

    /// Run until the schedule drains or the horizon is reached, then
    /// return the full result with provenance. With
    /// [`EngineConfig::stop_at_targets`] on, a run **early-stops** at the
    /// first target (feared event) reached.
    pub fn run(mut self) -> Result<SimulationResult, EngineError> {
        loop {
            if let Some((_, t_hit)) = &self.seq_end {
                // The target is reached: FINISH the hit instant first,
                // fire every transition still due at it, so the latched
                // state is the completed instant, not a half-propagated
                // one (PyCATSHOO completes the step before stopping).
                let t_hit = *t_hit;
                let still_due = self.pending.iter().flatten().any(|d| *d <= t_hit);
                if !still_due {
                    break;
                }
            }
            if self.step()?.is_none() {
                break;
            }
        }
        // Advance the clock (and the continuous state) to the horizon:
        // unless a target early-stopped the trajectory.
        let final_time = if let Some((_, t)) = &self.seq_end {
            *t
        } else if self.config.t_max.is_finite() {
            if self.needs_integration() && self.config.t_max > self.time {
                self.advance_continuous(self.config.t_max)?;
            }
            self.config.t_max
        } else {
            self.time
        };
        self.time = final_time;
        self.note_time_change();
        // A target-stopped trajectory holds its frozen state through the
        // remaining sample instants (the latch semantics of a
        // target-stopped study: the feared-event state stays active from
        // the hit to the horizon in every sampled measure). With an
        // infinite horizon the latch extends through the last *requested*
        // sample instant, so every replica's series covers the schedule.
        let flush_to = if self.seq_end.is_some() {
            if self.config.t_max.is_finite() {
                self.config.t_max
            } else {
                self.config
                    .samples
                    .last()
                    .copied()
                    .unwrap_or(final_time)
                    .max(final_time)
            }
        } else {
            final_time
        };
        self.flush_samples_through(flush_to);
        let sequence = self.config.sequences.then(|| {
            let (end_cause, end_time) = match self.seq_end.take() {
                Some((cause, t)) => (Some(cause), t),
                None => (None, final_time),
            };
            Sequence {
                events: std::mem::take(&mut self.seq_events),
                end_cause,
                end_time,
                weight: 1.0,
            }
        });
        let work = self.work();
        Ok(SimulationResult {
            events: self.events,
            indicators: self.indicator_series,
            samples: self.sampled,
            journal: self.journal,
            sequence,
            provenance: Provenance {
                engine_version: env!("CARGO_PKG_VERSION").to_owned(),
                model: self.model.name.clone(),
                t_max: self.config.t_max,
                seed: self.stochastic.then_some(self.config.seed),
                ode_rtol: self.config.ode.rtol,
                ode_tol_event: self.config.ode.tol_event,
            },
            work,
            final_time,
        })
    }

    // ---- internals ----------------------------------------------------

    /// Fire `trans_idx` at the current time: state change, journal,
    /// discrete evolution to fixpoint, schedule update, indicators.
    ///
    /// `forced` overrides the destination branch (interactive control,
    /// bypassing the RNG / deterministic-branch resolution); `None`
    /// resolves the destination the normal way ([`Engine::resolve_target`]).
    fn fire(
        &mut self,
        trans_idx: TransIdx,
        forced: Option<StateIdx>,
    ) -> Result<Event, EngineError> {
        self.pending[trans_idx] = None;
        self.frozen[trans_idx] = None;
        self.hazards[trans_idx] = None;
        let target = match forced {
            Some(state) => state,
            None => self.resolve_target(trans_idx)?,
        };
        let transition = &self.model.transitions[trans_idx];
        let automaton = &self.model.automata[transition.automaton];
        let event = Event {
            time: self.time,
            transition: transition.name.clone(),
            from: automaton.states[transition.source].clone(),
            to: automaton.states[target].clone(),
        };
        let owner = transition.automaton;
        self.states[owner] = target;
        // Indexed watched set: re-arm what this automaton owns and
        // invalidate the guards that read its state.
        self.note_state_change(owner);
        let transition = &self.model.transitions[trans_idx];
        if self.config.journal {
            self.journal.push(JournalRecord::TransitionFired {
                time: self.time,
                transition: event.transition.clone(),
                from: event.from.clone(),
                to: event.to.clone(),
            });
        }
        self.events.push(event.clone());
        // Sequence analysis: record the entry into a monitored state.
        if self.config.sequences && transition.monitored {
            self.seq_events.push(SeqEvent {
                obj: transition.component.clone(),
                attr: event.to.clone(),
                time: self.time,
                cycle_group: transition.cycle_group.clone(),
            });
        }

        // Discrete evolution: functions sensitive to this automaton.
        self.worklist.extend(
            self.model.state_triggers[transition.automaton]
                .iter()
                .copied(),
        );
        self.run_fixpoint()?;
        self.resolve_flows()?;
        self.refresh_schedule()?;
        self.record_indicators();
        // Sequence analysis: the first target (feared event) whose state is
        // now active labels the trajectory's end cause (and ends it: see
        // `run`). States change only through transitions (or the declared
        // init, checked in `initialize`), so this catches every activation.
        self.check_targets();
        Ok(event)
    }

    /// Settle the conservative flow network at a boundary: **the active
    /// set first, to exact equality, then the flows, to the per-edge
    /// tolerance**.
    ///
    /// Which consumers are saturated and which branch of each comparison
    /// is taken is a finite, combinatorial question; how much each edge
    /// carries is not. Settling the finite half first turns most of the
    /// problem from asymptotic into combinatorial, and it is what lets
    /// the rest of the segment run with the answer frozen.
    ///
    /// The sequence descends. Its first term is the ordered sweep run
    /// from a **cold start**, in which every allocated quantity is zero
    /// and every consumer therefore sizes itself as though it held
    /// nothing: that pass over-estimates every delivery, which is what
    /// makes it a post-fixpoint and the sequence that follows
    /// non-increasing. Iterating up from zero would carry no such
    /// argument.
    ///
    /// The cold start is recomputed here rather than carried from the
    /// previous resolution. A warm start would be state living outside
    /// the attribute vector, and it would have to enter the snapshot for
    /// a replay to reproduce it.
    ///
    /// A model with no distribution operator has no active set to settle
    /// and takes none of this: it runs the single ordered pass it ran
    /// before the resolution existed, at the same cost, which is why its
    /// counted-work profile is unchanged.
    ///
    /// # Convergence policy
    ///
    /// The two levels carry **two budgets, both counted in sweeps**, and
    /// the whole policy is [`EngineConfig::flow`]: the figures below are
    /// its defaults, not the only values it can take. A sweep that
    /// changes the saturation pattern is charged to the combinatorial
    /// budget ([`FlowConfig::active_set_budget`], by default derived from
    /// the compiled edge and decision count by [`active_set_budget`]); a
    /// sweep that leaves the pattern alone and only moves quantities is
    /// charged to the numeric one ([`FlowConfig::sweep_budget`], by
    /// default [`FLOW_SWEEP_BUDGET`]). Every sweep is charged to exactly
    /// one of them, so the whole resolution is bounded by their sum, plus
    /// the single regrant described below. Neither is the engine's
    /// [`EngineConfig::max_fixpoint_iterations`], which counts worklist
    /// pops of the *discrete* fixpoint and would bound a different thing.
    ///
    /// **Under-relaxation** ([`FlowConfig::relaxation`], by default
    /// [`FLOW_RELAXATION`]) is latched the first time, and only the first
    /// time, the iterate returns to where it stood two sweeps earlier, in
    /// its saturation pattern or in its quantities. That observation says
    /// the descending argument behind the combinatorial budget has
    /// failed, so the combinatorial budget is granted **once more** at
    /// that point: what follows is a damped numeric iteration, not a
    /// descent, and charging its first class flips against a spent budget
    /// would refuse networks that damping is about to settle.
    ///
    /// Exhausting either budget raises [`EngineError::FlowNotConverged`],
    /// naming every edge that moved in the final sweep. The cause
    /// distinguishes a two-cycle that survived damping from a pattern
    /// that never settled and from quantities that never stopped
    /// creeping, but all three carry the same payload: a long cycle and a
    /// slow monotone sequence exhaust a budget without ever matching the
    /// two-cycle test, and they are the commonest stalls.
    ///
    /// Whatever the resolution moves is drained into the stale marks of
    /// the indexed watched set on every exit path, the failing ones
    /// included: a resolution that raised still left the attribute vector
    /// where its last sweep put it.
    fn resolve_flows(&mut self) -> Result<(), EngineError> {
        let outcome = self.resolve_flows_inner();
        self.drain_changes();
        outcome
    }

    /// The resolution proper (see [`Engine::resolve_flows`], which wraps
    /// it to drain the change log).
    fn resolve_flows_inner(&mut self) -> Result<(), EngineError> {
        if self.model.flow_margins.is_empty() {
            let mut ctx = PassContext {
                work: &mut self.work,
                scratch: &mut self.flow_scratch,
                changed: &mut self.changed,
            };
            return recompute_explicit(
                self.model,
                &mut self.vars,
                &self.states,
                self.time,
                &mut ctx,
            );
        }
        let model = self.model;
        for margins in &model.flow_margins {
            if let CStep::Allocate(allocation) = &model.explicit[margins.step] {
                for &target in &allocation.allocated {
                    // The cold start is a *write* like any other: an edge
                    // whose settled quantity is zero would otherwise be
                    // reported as unchanged by the sweep that follows,
                    // and its move from the previous value would go
                    // unannounced.
                    self.changed
                        .write(&mut self.vars, target, Value::Float(0.0));
                }
            }
        }
        // History of the iteration, two sweeps deep: the two-cycle test
        // compares the fresh iterate with the one two steps back, which
        // is what an alternation between two allocations looks like.
        let mut previous: Option<ActiveSet> = None;
        let mut two_back: Option<ActiveSet> = None;
        let mut two_back_values: Option<Vec<Value>> = None;
        let mut before: Vec<Value> = Vec::with_capacity(self.vars.len());
        // Budgets, and the relaxation state. All three are locals: a
        // relaxation that survived a resolution would make the answer
        // depend on what the engine resolved before, and Monte-Carlo
        // results would stop being invariant in the thread count.
        let mut set_spent = 0usize;
        let mut flow_spent = 0usize;
        let mut relaxation = 1.0f64;
        let mut cycled = false;
        let mut sweeps = 0usize;
        // Read once: the policy is fixed for the engine's life, and
        // taking a copy keeps the loop from re-borrowing `self.config`
        // while it mutates `self.vars`.
        let policy = self.config.flow.clone();
        loop {
            before.clear();
            before.extend_from_slice(&self.vars);
            let current = self.sweep_once()?;
            if relaxation < 1.0 {
                self.relax_allocations(&before, relaxation);
            }
            sweeps += 1;
            let pattern_held = previous.as_ref() == Some(&current);
            if pattern_held && flows_settled(&before, &self.vars, policy.tolerance) {
                return Ok(());
            }
            // Two-cycle: the iterate is back where it stood two sweeps
            // ago, in its pattern or in its quantities, while differing
            // from the sweep just before it.
            let cycles = (two_back.as_ref() == Some(&current) && !pattern_held)
                || two_back_values
                    .as_ref()
                    .is_some_and(|old| flows_settled(old, &self.vars, policy.tolerance));
            // Latched on the *first* cycle only, and latched on `cycled`
            // rather than on the weight: a configuration that asks for no
            // damping at all (a weight of one) would otherwise re-grant
            // the combinatorial budget at every detection and never
            // exhaust it, turning a stall into a spin.
            if cycles && !cycled {
                cycled = true;
                relaxation = policy.relaxation;
                set_spent = 0;
            }
            if pattern_held {
                flow_spent += 1;
            } else {
                set_spent += 1;
            }
            if set_spent > self.active_set_budget || flow_spent > policy.sweep_budget {
                let cause = if cycled {
                    FlowStall::TwoCycle
                } else if pattern_held {
                    FlowStall::Quantities
                } else {
                    FlowStall::ActiveSet
                };
                return Err(self.flow_stalled(cause, sweeps, &before, previous.as_ref(), &current));
            }
            two_back_values = Some(before.clone());
            two_back = previous;
            previous = Some(current);
        }
    }

    /// Blend the allocated quantities of every distribution operator
    /// toward the answer the sweep just wrote, leaving the state at
    /// `(1 − w)·before + w·raw` (see [`FLOW_RELAXATION`]).
    ///
    /// Only the **allocated** quantities are blended, because they are
    /// the iteration's variables: every demand and every downstream
    /// equation is a function of them and is recomputed from the blended
    /// values by the next sweep. The state written by this sweep is
    /// therefore momentarily inconsistent with the blend, by exactly the
    /// amount the blend moved; the resolution only returns once that
    /// amount is below [`FLOW_TOLERANCE`], which is the level it promises
    /// anyway.
    fn relax_allocations(&mut self, before: &[Value], weight: f64) {
        let model = self.model;
        for margins in &model.flow_margins {
            let CStep::Allocate(allocation) = &model.explicit[margins.step] else {
                continue;
            };
            for &target in &allocation.allocated {
                let (Value::Float(old), Value::Float(raw)) = (before[target], self.vars[target])
                else {
                    continue;
                };
                // Through the change log like every other write of the
                // resolution: the blend moves the same targets the sweep
                // wrote, and the margins reading them must hear about it.
                self.changed.write(
                    &mut self.vars,
                    target,
                    Value::Float((1.0 - weight) * old + weight * raw),
                );
            }
        }
    }

    /// Build the non-convergence diagnostic: name the component and flow
    /// of every edge that moved in the final sweep, in the shape the
    /// instantaneous-loop and non-confluence diagnostics use.
    ///
    /// An edge "moved" if its saturation class changed or its allocated
    /// quantity moved by more than the flow tolerance: the same measure
    /// the stopping test applies, so the diagnostic can never name an
    /// empty set while the resolution claims something is still moving.
    /// It names one anyway when the movement was a branch decision (a
    /// conditional, a limiting minimum) rather than an edge, since a
    /// silent empty list would read as a defect.
    fn flow_stalled(
        &self,
        cause: FlowStall,
        sweeps: usize,
        before: &[Value],
        previous: Option<&ActiveSet>,
        current: &ActiveSet,
    ) -> EngineError {
        let mut moving: Vec<String> = Vec::new();
        let mut base = 0usize;
        for margins in &self.model.flow_margins {
            let CStep::Allocate(allocation) = &self.model.explicit[margins.step] else {
                continue;
            };
            for (edge, &target) in allocation.allocated.iter().enumerate() {
                let quantity_moved = !value_settled(
                    &before[target],
                    &self.vars[target],
                    self.config.flow.tolerance,
                );
                let class_moved = previous.is_some_and(|old| {
                    old.edges.get(base + edge) != current.edges.get(base + edge)
                });
                if quantity_moved || class_moved {
                    moving.push(self.edge_name(margins, edge));
                }
            }
            base += allocation.allocated.len();
        }
        let moving = if moving.is_empty() {
            "no edge (a conditional or a limiting minimum kept flipping)".to_owned()
        } else {
            moving.join(", ")
        };
        EngineError::FlowNotConverged {
            time: self.time,
            sweeps,
            cause,
            moving,
        }
    }

    /// Qualified `operator[consumer]` name of one edge: the operator is
    /// `component.allocation`, the consumer the allocated attribute, so
    /// the pair names the component and the flow.
    fn edge_name(&self, margins: &CFlowMargins, edge: usize) -> String {
        let consumer = margins
            .consumers
            .get(edge)
            .cloned()
            .unwrap_or_else(|| format!("edge #{edge}"));
        format!("{}[{}]", margins.name, consumer)
    }

    /// Name a set of **global** edge indices (the index space the frozen
    /// active set and [`Segment::Resolved`] use), for the chattering
    /// diagnostic.
    fn edge_names(&self, indices: &[usize]) -> String {
        let mut names: Vec<String> = Vec::new();
        let mut base = 0usize;
        for margins in &self.model.flow_margins {
            let CStep::Allocate(allocation) = &self.model.explicit[margins.step] else {
                continue;
            };
            for edge in 0..allocation.allocated.len() {
                if indices.contains(&(base + edge)) {
                    names.push(self.edge_name(margins, edge));
                }
            }
            base += allocation.allocated.len();
        }
        if names.is_empty() {
            "no edge".to_owned()
        } else {
            names.join(", ")
        }
    }

    /// One sweep of the resolution: the ordered explicit pass, plus the
    /// active set it produced.
    fn sweep_once(&mut self) -> Result<ActiveSet, EngineError> {
        self.flow_scratch.classes = Some(Vec::new());
        let mut ctx = PassContext {
            work: &mut self.work,
            scratch: &mut self.flow_scratch,
            changed: &mut self.changed,
        };
        let outcome = recompute_explicit(
            self.model,
            &mut self.vars,
            &self.states,
            self.time,
            &mut ctx,
        );
        let edges = self.flow_scratch.classes.take().unwrap_or_default();
        outcome?;
        self.work.flow_sweeps += 1;
        let mut branches = Vec::new();
        for step in &self.model.explicit {
            match step {
                CStep::Equation { expr, .. } => record_branches(
                    self.model,
                    &self.vars,
                    &self.states,
                    self.time,
                    expr,
                    &mut branches,
                )?,
                CStep::Allocate(allocation) => record_branches(
                    self.model,
                    &self.vars,
                    &self.states,
                    self.time,
                    &allocation.available,
                    &mut branches,
                )?,
            }
        }
        Ok(ActiveSet { edges, branches })
    }

    /// The active set of the **current** state, read off a sweep run on a
    /// copy of the attribute vector so the committed state does not move.
    ///
    /// Derived, never carried: this is what keeps the frozen active set
    /// out of the snapshot and a replay exact.
    fn frozen_classes(&mut self) -> Result<Vec<EdgeClass>, EngineError> {
        if self.model.flow_margins.is_empty() {
            return Ok(Vec::new());
        }
        let mut vars = self.vars.clone();
        let mut scratch = FlowScratch {
            classes: Some(Vec::new()),
            ..FlowScratch::default()
        };
        // The copy is thrown away with the pass: nothing to invalidate.
        let mut changed = ChangeLog::discarding(self.config.flow.tolerance);
        let mut ctx = PassContext {
            work: &mut self.work,
            scratch: &mut scratch,
            changed: &mut changed,
        };
        recompute_explicit(self.model, &mut vars, &self.states, self.time, &mut ctx)?;
        Ok(scratch.classes.unwrap_or_default())
    }

    /// Advance continuous evolution to `t_target`, restarting a segment
    /// at every located active-set crossing until either the date is
    /// reached or a watched transition must fire.
    ///
    /// The loop is guarded the way [`Engine::note_watched_firing`] guards
    /// watched transitions, and for the same reason: a crossing that
    /// fires no transition is invisible to that guard, so a boundary the
    /// network re-crosses on the spot would spin here forever. Restarts
    /// are counted **per instant** rather than per advance: an instant
    /// admits at most one class change per compiled edge
    /// ([`active_set_budget`]) before the boundary is chattering
    /// rather than moving, whereas a long horizon legitimately crosses
    /// any number of boundaries. Progress is measured against the
    /// event-location tolerance, so a crossing located a hair after the
    /// previous one does not pass for progress.
    fn advance_continuous(&mut self, t_target: f64) -> Result<Option<TransIdx>, EngineError> {
        let mut stuck_at = f64::NEG_INFINITY;
        let mut stuck: Vec<usize> = Vec::new();
        loop {
            match self.integrate_to(t_target)? {
                Segment::Reached => return Ok(None),
                Segment::Watched(trans_idx) => return Ok(Some(trans_idx)),
                Segment::Resolved(edge) => {
                    if self.time > stuck_at + self.config.ode.tol_event {
                        stuck_at = self.time;
                        stuck.clear();
                    }
                    stuck.push(edge);
                    if stuck.len() > self.active_set_budget {
                        return Err(EngineError::FlowChattering {
                            time: self.time,
                            restarts: stuck.len(),
                            edges: self.edge_names(&stuck),
                        });
                    }
                    if self.time >= t_target {
                        return Ok(None);
                    }
                }
            }
        }
    }

    /// Whether continuous evolution must run: the model has ODE
    /// attributes, an armed hazard varies continuously
    /// (`reschedule_modifiable` under `integrate_continuous`: possibly
    /// with no ODE at all, e.g. a time-dependent rate), or a step of the
    /// explicit sweep reads the clock.
    ///
    /// The third case has no state behind it at all. A declared time
    /// profile is an explicit equation over `time` and nothing else, so
    /// without this the engine would find nothing to advance, evaluate
    /// the sweep once at the initial instant, and report that value at
    /// every sample instant and to every watched guard for the rest of
    /// the run: a curve reported as a constant, with nothing to signal
    /// it.
    fn needs_integration(&self) -> bool {
        !self.model.ode.is_empty()
            || self.model.explicit_reads_time
            || self
                .continuous_rates
                .iter()
                .any(|&idx| self.pending[idx].is_some())
    }

    /// Zeno guard: watched transitions must not keep firing without
    /// time advancing.
    fn note_watched_firing(&mut self) -> Result<(), EngineError> {
        if self.watched_streak.0 == self.time {
            self.watched_streak.1 += 1;
            if self.watched_streak.1 > 1_000 {
                return Err(EngineError::WatchedLoop { time: self.time });
            }
        } else {
            self.watched_streak = (self.time, 1);
        }
        Ok(())
    }

    // ---- the indexed watched set -------------------------------------
    //
    // Three pieces of bookkeeping keep the two scan sites proportional to
    // what moved: which positions are *armed* (their automaton sits in
    // the source state), what each guard last *evaluated to*, and which
    // of those verdicts a change has *invalidated*. Every mutation of the
    // attribute vector, of an automaton state or of the clock passes
    // through one of the `note_*` methods below; forgetting one would
    // leave a verdict cached against a state that no longer holds, which
    // is why the full rebuild is the reinstatement path of every
    // wholesale state change (build, reset, restore, confluence probe).

    /// Rebuild the whole index from the current state: arming derived
    /// from the automaton states, every cached verdict discarded.
    ///
    /// The reinstatement path of a wholesale state change. It is the only
    /// place the watched population is walked in full, and it is
    /// deliberately not on the simulation cycle.
    fn rebuild_watched_index(&mut self) {
        let model = self.model;
        self.watched_active.clear();
        for (position, &trans_idx) in model.watched.iter().enumerate() {
            let transition = &model.transitions[trans_idx];
            let armed = self.states[transition.automaton] == transition.source;
            self.watched_armed[position] = armed;
            self.watched_stale[position] = true;
            if armed {
                self.watched_active.push(position);
            }
        }
    }

    /// An attribute moved: invalidate the guards that read it.
    #[inline]
    fn note_var_change(&mut self, var: VarIdx) {
        let model = self.model;
        for &position in &model.margin_index.watched_by_var[var] {
            self.watched_stale[position] = true;
        }
    }

    /// The clock moved: invalidate the guards that read it. Normally a
    /// no-op, a boundary being a predicate on the continuous state.
    #[inline]
    fn note_time_change(&mut self) {
        let model = self.model;
        for &position in &model.margin_index.watched_by_time {
            self.watched_stale[position] = true;
        }
    }

    /// An automaton changed state: invalidate the guards that read that
    /// state, and re-arm the transitions the automaton owns.
    ///
    /// The arming update keeps `watched_active` **ascending**, inserting
    /// and removing by binary search rather than re-deriving the list, so
    /// the per-segment margin set and the guard scan visit the watched
    /// transitions in the order the full scan visited them.
    fn note_state_change(&mut self, automaton: AutIdx) {
        let model = self.model;
        for &position in &model.margin_index.watched_by_state[automaton] {
            self.watched_stale[position] = true;
        }
        let state = self.states[automaton];
        for &position in &model.margin_index.watched_by_owner[automaton] {
            let armed = model.transitions[model.watched[position]].source == state;
            if armed == self.watched_armed[position] {
                continue;
            }
            self.watched_armed[position] = armed;
            match self.watched_active.binary_search(&position) {
                Ok(at) if !armed => {
                    self.watched_active.remove(at);
                }
                Err(at) if armed => {
                    self.watched_active.insert(at, position);
                }
                _ => {}
            }
        }
    }

    /// Drain what the engine's own explicit passes reported (see
    /// [`ChangeLog`]) into the stale marks, keeping the buffer for reuse.
    fn drain_changes(&mut self) {
        let Some(mut moved) = self.changed.moved.take() else {
            return;
        };
        let model = self.model;
        for &var in &moved {
            for &position in &model.margin_index.watched_by_var[var] {
                self.watched_stale[position] = true;
            }
        }
        moved.clear();
        self.changed.moved = Some(moved);
    }

    /// A watched transition whose *guard* already holds while its
    /// automaton sits in the source state fires immediately.
    ///
    /// The guard is evaluated exactly (boolean), not through the
    /// margin: after a located crossing, a sibling transition sharing
    /// the boundary may sit within round-off of it: its strict guard
    /// is already true while its ε-tightened margin is still negative.
    /// Conversely a trajectory *resting* exactly on a strict boundary
    /// keeps a false guard and does not fire (no Zeno).
    ///
    /// **Indexed.** The scan walks the armed positions in ascending
    /// order, exactly as the full scan did, but re-evaluates only the
    /// guards a change has invalidated; the rest answer from their cached
    /// verdict. The short-circuit is preserved with the order: the scan
    /// still stops at the first armed guard that holds, so a guard past
    /// it is neither evaluated nor cleared, and a guard that would raise
    /// on a state the model never reaches stays unevaluated exactly as
    /// before.
    fn immediate_watched(&mut self) -> Result<Option<TransIdx>, EngineError> {
        let model = self.model;
        for index in 0..self.watched_active.len() {
            let position = self.watched_active[index];
            let trans_idx = model.watched[position];
            let Some(guard) = &model.transitions[trans_idx].guard else {
                continue;
            };
            if self.watched_stale[position] {
                // Counted work: this scan is one of the two sites the
                // index narrows, so its cost is measured here.
                self.work.immediate_guard_scans += 1;
                self.watched_guard[position] =
                    eval_bool(model, &self.vars, &self.states, self.time, guard)?;
                self.watched_stale[position] = false;
            }
            if self.watched_guard[position] {
                return Ok(Some(trans_idx));
            }
        }
        Ok(None)
    }

    /// `integrate_continuous`: integrate the continuous state to `t_target`, monitoring
    /// active watched boundaries. Returns the watched transition to fire
    /// if a crossing was located first (time/state already advanced).
    fn integrate_to(&mut self, t_target: f64) -> Result<Segment, EngineError> {
        // **Indexed.** The armed set is maintained on state changes, not
        // re-derived here: a segment restarted at a located active-set
        // crossing changes no automaton state, so the set it monitors is
        // the one the previous segment monitored, and rebuilding it would
        // cost the model's whole watched population per restart.
        let margins: Vec<TransIdx> = self
            .watched_active
            .iter()
            .map(|&position| self.model.watched[position])
            .collect();

        // Armed continuously-varying hazards ride along as auxiliary
        // state (`dH/dt = λ`), their firing located as an event.
        let hazard_monitors: Vec<(TransIdx, f64)> = self
            .continuous_rates
            .iter()
            .copied()
            .filter(|&idx| self.pending[idx].is_some())
            .filter_map(|idx| {
                self.hazards[idx].map(|hazard| (idx, hazard.threshold - hazard.accumulated))
            })
            .collect();

        let mut y: Vec<f64> = Vec::with_capacity(self.model.ode.len() + hazard_monitors.len());
        for (var, _) in &self.model.ode {
            match self.vars[*var] {
                Value::Float(f) => y.push(f),
                other => {
                    return Err(EngineError::TypeError {
                        time: self.time,
                        detail: format!(
                            "ODE attribute `{}` holds non-float value {other:?}",
                            self.model.var_names[*var]
                        ),
                    });
                }
            }
        }

        y.resize(self.model.ode.len() + hazard_monitors.len(), 0.0);

        // Freeze the active set for this segment. It is *derived* from
        // the resolved state, here, rather than carried from the
        // resolution that produced it: nothing about the search survives
        // outside the attribute vector, so a restored snapshot replays
        // identically.
        let classes = self.frozen_classes()?;
        let mut taken = 0usize;
        let mut flows: Vec<FrozenFlow<'_>> = Vec::with_capacity(self.model.flow_margins.len());
        for margins in &self.model.flow_margins {
            let CStep::Allocate(allocation) = &self.model.explicit[margins.step] else {
                continue;
            };
            let width = allocation.allocated.len();
            let slice = classes
                .get(taken..taken + width)
                .map(<[EdgeClass]>::to_vec)
                .unwrap_or_default();
            taken += width;
            if slice.len() == width {
                flows.push(FrozenFlow {
                    allocation,
                    margins,
                    classes: slice,
                });
            }
        }

        let mut system = ContinuousSystem {
            model: self.model,
            vars: self.vars.clone(),
            states: self.states.clone(),
            margins,
            hazards: hazard_monitors,
            flows,
            error: None,
            work: WorkCounters::default(),
            scratch: FlowScratch::default(),
            flow_demands: Vec::new(),
            flow_tolerance: self.config.flow.tolerance,
        };
        self.work.segments += 1;

        // Dense sampling: indicator values recorded from the interpolant.
        let segment_samples: Vec<f64> = self.config.samples[self.sample_cursor..]
            .iter()
            .copied()
            .take_while(|s| *s <= t_target)
            .collect();
        let mut recorded: Vec<(f64, Vec<Value>)> = Vec::new();
        // The dense-sample callback runs its own explicit pass, and the
        // solver holds `system` mutably meanwhile: it counts into its own
        // tally, merged with the system's when the segment returns.
        let mut sample_work = WorkCounters::default();
        let mut sample_scratch = FlowScratch::default();
        // Second half of the error stash. The solver's sample callback is
        // as infallible as its system, and it runs while the solver holds
        // `system` mutably, so it cannot reach `system.error`: without a
        // stash of its own an explicit pass failing at a dense-sample
        // instant would be dropped on the floor, and a sample instant is
        // not necessarily an instant any other callback visits.
        let mut sample_error: Option<EngineError> = None;
        // Recorded from a copy of the attribute vector: nothing to
        // invalidate, so the pass compares and reports nothing.
        let mut sample_changed = ChangeLog::discarding(self.config.flow.tolerance);
        {
            let model = self.model;
            let base_vars = self.vars.clone();
            let states = self.states.clone();
            let mut on_sample = |t: f64, y_at: &[f64]| {
                let mut vars = base_vars.clone();
                for (slot, (var, _)) in model.ode.iter().enumerate() {
                    vars[*var] = Value::Float(y_at[slot]);
                }
                let mut ctx = PassContext {
                    work: &mut sample_work,
                    scratch: &mut sample_scratch,
                    changed: &mut sample_changed,
                };
                if let Err(error) = recompute_explicit(model, &mut vars, &states, t, &mut ctx) {
                    // Stashed, then re-raised below: the callback has no
                    // way to report it and this instant's values are not
                    // recordable.
                    sample_error.get_or_insert(error);
                    return;
                }
                let values = model
                    .indicators
                    .iter()
                    .map(|indicator| match indicator.target {
                        CIndicatorTarget::Var(idx) => vars[idx],
                        CIndicatorTarget::State(aut, state) => {
                            Value::Float(if states[aut] == state { 1.0 } else { 0.0 })
                        }
                    })
                    .collect();
                recorded.push((t, values));
            };

            let outcome = self.solver.integrate(
                &mut system,
                self.time,
                &mut y,
                t_target,
                &segment_samples,
                &mut on_sample,
            )?;

            self.work.explicit_evaluations +=
                system.work.explicit_evaluations + sample_work.explicit_evaluations;
            self.work.allocation_capping_passes +=
                system.work.allocation_capping_passes + sample_work.allocation_capping_passes;
            self.work.margin_evaluations += system.work.margin_evaluations;

            // Re-raise whatever the callbacks stashed. The time it
            // carries is the *evaluation point* at which the failure was
            // detected: inside a bisection that is a probe of the
            // interval, not a located instant, and it is reported as it
            // stands rather than being rewritten to the segment's
            // committed time, which would name a state that never failed.
            if let Some(error) = system.error.take().or(sample_error) {
                return Err(error);
            }

            // Commit the reached continuous state. Three families of
            // event share one index space: watched transitions, then
            // continuously-varying hazards, then the active-set margins
            // of the frozen flows. The flows come last on purpose, so a
            // transition crossing in the same sub-interval wins the tie.
            let watched_and_hazards = system.margins.len() + system.hazards.len();
            let (t_reached, fired, crossed) = match outcome {
                Outcome::Reached { t } => (t, None, None),
                Outcome::Event { index, t } if index < watched_and_hazards => {
                    let fired = if index < system.margins.len() {
                        system.margins[index]
                    } else {
                        system.hazards[index - system.margins.len()].0
                    };
                    (t, Some(fired), None)
                }
                Outcome::Event { index, t } => (t, None, Some(index - watched_and_hazards)),
            };
            self.time = t_reached;
            self.note_time_change();
            // Committing the reached continuous state. Compared
            // **exactly** here, unlike the explicit pass: an integrated
            // attribute is the very thing a watched boundary is a
            // predicate on, and a tolerance band on it would let a
            // trajectory cross a boundary while the guard reading it kept
            // a cached verdict.
            let model = self.model;
            for (slot, (var, _)) in model.ode.iter().enumerate() {
                let value = Value::Float(y[slot]);
                if self.vars[*var] != value {
                    self.vars[*var] = value;
                    self.note_var_change(*var);
                }
            }
            // Bank the hazard accrued over this segment (`reschedule_modifiable`
            // bookkeeping; the fired transition's slot, if any, is
            // cleared by `fire`).
            let ode_len = self.model.ode.len();
            for (slot, (trans_idx, _)) in system.hazards.iter().enumerate() {
                if let Some(hazard) = self.hazards[*trans_idx].as_mut() {
                    hazard.accumulated += y[ode_len + slot].max(0.0);
                    hazard.since = t_reached;
                }
            }
            self.resolve_flows()?;

            // Commit dense samples (strictly before the reached time:
            // a sample at exactly an event date is recorded post-event
            // by the flush in the next advance).
            for (t, values) in recorded {
                if t < t_reached || (fired.is_none() && crossed.is_none()) {
                    for (series, value) in self.sampled.iter_mut().zip(values) {
                        series.points.push((t, value));
                    }
                    self.sample_cursor += 1;
                }
            }
            match crossed {
                None => Ok(fired.map_or(Segment::Reached, Segment::Watched)),
                Some(index) => {
                    // A located active-set crossing is a change point of
                    // the network like a fired transition is one of the
                    // discrete state: the resolved quantities take a new
                    // shape there. `record_indicators` appends only on a
                    // change, so a model with no operator (and therefore
                    // no crossing) never reaches this and keeps its
                    // series exactly as before.
                    self.record_indicators();
                    self.journal_active_set_crossing(&system.flows, index)?;
                    Ok(Segment::Resolved(index))
                }
            }
        }
    }

    /// Record a located active-set crossing in the causal journal, naming
    /// the operator, the edge, and the two saturation classes it moved
    /// between. The network has already been resolved again, so the
    /// destination class is read from the fresh resolution.
    ///
    /// Zero cost when the journal is off, which is the default.
    fn journal_active_set_crossing(
        &mut self,
        flows: &[FrozenFlow<'_>],
        index: usize,
    ) -> Result<(), EngineError> {
        if !self.config.journal {
            return Ok(());
        }
        let mut offset = 0usize;
        let mut located: Option<(&FrozenFlow<'_>, usize, usize)> = None;
        for flow in flows {
            if index < offset + flow.classes.len() {
                located = Some((flow, index - offset, offset));
                break;
            }
            offset += flow.classes.len();
        }
        let Some((flow, edge, base)) = located else {
            return Ok(());
        };
        let after = self.frozen_classes()?;
        let record = JournalRecord::ActiveSetCrossed {
            time: self.time,
            operator: flow.margins.name.clone(),
            consumer: flow
                .margins
                .consumers
                .get(edge)
                .cloned()
                .unwrap_or_else(|| format!("edge #{edge}")),
            from: flow.classes[edge],
            to: after
                .get(base + edge)
                .copied()
                .unwrap_or(flow.classes[edge]),
        };
        self.journal.push(record);
        Ok(())
    }

    fn next_pending(&self) -> Option<(TransIdx, f64)> {
        let mut best: Option<(TransIdx, f64)> = None;
        for (idx, pending) in self.pending.iter().enumerate() {
            if let Some(date) = pending {
                let better = match best {
                    None => true,
                    Some((_, best_date)) => *date < best_date,
                };
                if better {
                    best = Some((idx, *date));
                }
            }
        }
        best
    }

    fn resolve_target(&mut self, trans_idx: TransIdx) -> Result<StateIdx, EngineError> {
        // Copy the `'m` model reference out so the transition borrow is
        // independent of `&mut self` (frees `self.rng` for the draw).
        let model = self.model;
        let transition = &model.transitions[trans_idx];
        match &transition.distrib {
            CLaw::Delay(_)
            | CLaw::Watched { .. }
            | CLaw::Exp(_)
            | CLaw::ExpVar { .. }
            | CLaw::Weibull(..)
            | CLaw::Lognormal(..)
            | CLaw::Gamma(..)
            | CLaw::Uniform(..)
            | CLaw::Empirical(_) => Ok(transition.targets[0]),
            CLaw::Inst(probs) => {
                // Deterministic fast path: exactly one branch with
                // probability 1: resolved without touching the RNG, so a
                // deterministic model stays RNG-free and bit-identical on
                // replay.
                if let Some(branch) = probs
                    .iter()
                    .position(|p| (*p - 1.0).abs() <= f64::EPSILON)
                    .filter(|_| probs.iter().filter(|p| **p > 0.0).count() == 1)
                {
                    return Ok(transition.targets[branch]);
                }
                // Stochastic instantaneous branching (`schedule_stochastic`
                // realised on demand): draw the destination from the
                // categorical distribution over `probs` (Σ = 1, validated at
                // model build) by inverse-CDF on one uniform. The draw
                // happens at fire time in the deterministic firing order, so
                // replay stays bit-identical for a fixed (seed, stream).
                let u: f64 = rand::Rng::random(&mut self.rng);
                let mut cumulative = 0.0;
                for (branch, p) in probs.iter().enumerate() {
                    cumulative += *p;
                    if u < cumulative {
                        return Ok(transition.targets[branch]);
                    }
                }
                // `u` within rounding of 1.0: the last branch.
                Ok(transition.targets[probs.len() - 1])
            }
        }
    }

    /// `propagate_effects`: propagate sensitive functions to a fixpoint in the
    /// documented deterministic order (ascending function index).
    fn run_fixpoint(&mut self) -> Result<(), EngineError> {
        if self.config.confluence_check {
            self.confluence_probe()
        } else {
            self.converge(false)
        }
    }

    /// Apply one function's effects; when `trigger` is set, attribute
    /// changes enqueue their dependent functions.
    fn apply_function(&mut self, fn_idx: FnIdx, trigger: bool) -> Result<(), EngineError> {
        let function = &self.model.functions[fn_idx];
        for (target, value_expr) in &function.effects {
            let new = eval_expr(self.model, &self.vars, &self.states, self.time, value_expr)?;
            let old = self.vars[*target];
            if old != new {
                self.vars[*target] = new;
                // Indexed watched set: invalidate the guards reading it.
                // The comparison above is the sensitive functions' own,
                // older than the explicit pass's: this site was already a
                // change *detector*, it only lacked a consumer.
                self.note_var_change(*target);
                if self.config.journal {
                    self.journal.push(JournalRecord::AttributeChanged {
                        time: self.time,
                        attribute: self.model.var_names[*target].clone(),
                        old,
                        new,
                        cause: function.name.clone(),
                    });
                }
                if trigger {
                    self.worklist
                        .extend(self.model.var_triggers[*target].iter().copied());
                }
            }
        }
        Ok(())
    }

    /// Non-confluence diagnostic: converge a *copy* of the state with
    /// the worklist processed in reverse order and compare. Divergence
    /// means the model's result depends on evaluation order: reported
    /// as a typed error (rather than silently picking an arbitrary order).
    fn confluence_probe(&mut self) -> Result<(), EngineError> {
        let saved_vars = self.vars.clone();
        let saved_worklist = self.worklist.clone();

        // Canonical forward pass (journaled normally).
        self.converge(false)?;
        let forward_vars = std::mem::replace(&mut self.vars, saved_vars);
        // The attribute vector has just been rewound wholesale, and it is
        // rewound again below whichever way the probe ends: discard every
        // cached watched verdict here, once, rather than reason about the
        // union of two passes on every exit path (the diverging one
        // included, which an interactive caller can catch and continue
        // from).
        for stale in &mut self.watched_stale {
            *stale = true;
        }

        // Silent reverse pass on the saved state.
        self.worklist = saved_worklist;
        let journal_flag = std::mem::replace(&mut self.config.journal, false);
        let reverse = self.converge(true);
        self.config.journal = journal_flag;
        reverse?;

        if forward_vars != self.vars {
            // Name the first diverging attribute's writers for the
            // diagnostic message.
            let diverging = forward_vars
                .iter()
                .zip(&self.vars)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            // The distribution operators are named alongside the
            // sensitive functions, though they cannot themselves diverge:
            // they run in the explicit sweep, after the fixpoint, from
            // whatever state it converged to, so both replays recompute
            // them identically. A diverging allocated quantity therefore
            // means a *second* writer, which the model layer refuses
            // (`AllocationTargetWritten`), or a defect in this engine: in
            // either case naming the operator is what makes the report
            // actionable, and leaving it out is the "divergence with no
            // writer name" this probe must never produce.
            let operators = self.model.explicit.iter().filter_map(|step| match step {
                CStep::Allocate(allocation) if allocation.allocated.contains(&diverging) => {
                    Some(allocation.name.clone())
                }
                _ => None,
            });
            let mut writers = self
                .model
                .functions
                .iter()
                .filter(|f| f.effects.iter().any(|(target, _)| *target == diverging))
                .map(|f| f.name.clone())
                .chain(operators);
            let first = writers.next().unwrap_or_else(|| "<unknown>".to_owned());
            let second = writers.next().unwrap_or_else(|| first.clone());
            return Err(EngineError::NonConfluent {
                time: self.time,
                first,
                second,
            });
        }
        // Both orders agree: keep the forward result as canonical (the
        // cached verdicts were already discarded above).
        self.vars = forward_vars;
        Ok(())
    }

    /// Converge the current worklist, ascending or descending order.
    fn converge(&mut self, reverse: bool) -> Result<(), EngineError> {
        let mut iterations = 0usize;
        loop {
            let next = if reverse {
                self.worklist.pop_last()
            } else {
                self.worklist.pop_first()
            };
            let Some(fn_idx) = next else { break };
            iterations += 1;
            if iterations > self.config.max_fixpoint_iterations {
                self.worklist.clear();
                return Err(EngineError::InstantaneousLoop {
                    time: self.time,
                    iterations: self.config.max_fixpoint_iterations,
                });
            }
            if self.config.journal {
                self.journal.push(JournalRecord::FunctionTriggered {
                    time: self.time,
                    function: self.model.functions[fn_idx].name.clone(),
                });
            }
            self.apply_function(fn_idx, true)?;
        }
        Ok(())
    }

    /// `schedule_deterministic` + `drop_disabled`: (re)schedule fireable transitions and drop
    /// stale ones. Deterministic full scan (fine at fixture model
    /// sizes; the scan is index-ordered so the schedule is
    /// reproducible). Watched transitions are never date-scheduled.
    /// Evaluate a state-dependent rate λ(x) on the current state and
    /// reject non-finite or negative values with a typed error.
    fn eval_rate(&self, trans_idx: TransIdx, rate: &CExpr) -> Result<f64, EngineError> {
        let lambda = eval_f64(self.model, &self.vars, &self.states, self.time, rate)?;
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(EngineError::TypeError {
                time: self.time,
                detail: format!(
                    "state-dependent rate of `{}` evaluated to {lambda} \
                     (must be finite and >= 0)",
                    self.model.transitions[trans_idx].name
                ),
            });
        }
        Ok(lambda)
    }

    fn refresh_schedule(&mut self) -> Result<(), EngineError> {
        for trans_idx in 0..self.model.transitions.len() {
            let transition = &self.model.transitions[trans_idx];
            if matches!(transition.distrib, CLaw::Watched { .. }) {
                continue;
            }
            let in_source = self.states[transition.automaton] == transition.source;
            if !in_source {
                // Leaving the source cancels any paused countdown and
                // discards the banked hazard.
                self.frozen[trans_idx] = None;
                self.hazards[trans_idx] = None;
            }
            let guard_ok = match &transition.guard {
                None => true,
                Some(guard) => eval_bool(self.model, &self.vars, &self.states, self.time, guard)?,
            };
            match self.pending[trans_idx] {
                Some(_) if !in_source => {
                    self.pending[trans_idx] = None;
                    if self.config.journal {
                        self.journal.push(JournalRecord::TransitionDropped {
                            time: self.time,
                            transition: transition.name.clone(),
                            reason: DropReason::SourceLeft,
                        });
                    }
                }
                Some(date)
                    if !guard_ok
                        && transition.on_interruption
                            != raichu_model::InterruptionPolicy::Continue =>
                {
                    // `drop_disabled`: reset cancels the occurrence duration
                    // (interruptible transition); resume
                    // pauses it (RAICHU extension); continue never
                    // reaches this arm.
                    let reason = match transition.on_interruption {
                        raichu_model::InterruptionPolicy::Resume => {
                            if let Some(hazard) = self.hazards[trans_idx].as_mut() {
                                // Pause the hazard clock: bank what has
                                // accrued (continuous hazards are already
                                // committed by the integrator); the
                                // re-arm recomputes λ at resumption.
                                if !matches!(
                                    transition.distrib,
                                    CLaw::ExpVar {
                                        continuous: true,
                                        ..
                                    }
                                ) {
                                    hazard.accumulated += hazard.rate * (self.time - hazard.since);
                                }
                                hazard.since = self.time;
                            } else {
                                self.frozen[trans_idx] = Some(date - self.time);
                            }
                            DropReason::GuardPaused
                        }
                        _ => {
                            self.hazards[trans_idx] = None;
                            DropReason::GuardFalse
                        }
                    };
                    self.pending[trans_idx] = None;
                    if self.config.journal {
                        self.journal.push(JournalRecord::TransitionDropped {
                            time: self.time,
                            transition: transition.name.clone(),
                            reason,
                        });
                    }
                }
                #[allow(clippy::float_cmp)] // λ is re-evaluated exactly
                Some(previous) if in_source => {
                    // `reschedule_modifiable`: a pending state-dependent rate whose
                    // inputs changed at this discrete step is
                    // rescheduled against the same `Exp(1)` threshold
                    // (reached with the guard still true, or with the
                    // `continue` policy riding through a false guard).
                    // Continuously-varying rates need no rescheduling:
                    // their hazard is integrated by `integrate_continuous` directly.
                    let CLaw::ExpVar {
                        rate,
                        continuous: false,
                    } = &self.model.transitions[trans_idx].distrib
                    else {
                        continue;
                    };
                    let lambda = self.eval_rate(trans_idx, rate)?;
                    let Some(hazard) = self.hazards[trans_idx].as_mut() else {
                        continue;
                    };
                    if lambda != hazard.rate {
                        hazard.accumulated += hazard.rate * (self.time - hazard.since);
                        hazard.since = self.time;
                        hazard.rate = lambda;
                        let firing_at = if lambda > 0.0 {
                            self.time + (hazard.threshold - hazard.accumulated) / lambda
                        } else {
                            f64::INFINITY
                        };
                        self.pending[trans_idx] = Some(firing_at);
                        if self.config.journal && firing_at != previous {
                            self.journal.push(JournalRecord::TransitionRescheduled {
                                time: self.time,
                                transition: self.model.transitions[trans_idx].name.clone(),
                                firing_at,
                            });
                        }
                    }
                }
                None if in_source && guard_ok => {
                    // `schedule_stochastic` for a state-dependent rate: draw the
                    // `Exp(1)` threshold (fresh arming) or keep the
                    // banked hazard (resume re-arm), then schedule
                    // against the current λ: `+∞` while λ = 0 or while
                    // the hazard is integrated continuously (`integrate_continuous`
                    // locates the firing like a boundary crossing).
                    if let CLaw::ExpVar { rate, continuous } =
                        &self.model.transitions[trans_idx].distrib
                    {
                        let lambda = self.eval_rate(trans_idx, rate)?;
                        let mut hazard = match self.hazards[trans_idx] {
                            Some(banked) => banked,
                            None => Hazard {
                                threshold: rand_distr::Exp1.sample(&mut self.rng),
                                accumulated: 0.0,
                                rate: lambda,
                                since: self.time,
                            },
                        };
                        hazard.rate = lambda;
                        hazard.since = self.time;
                        let firing_at = if *continuous || lambda <= 0.0 {
                            f64::INFINITY
                        } else {
                            self.time + (hazard.threshold - hazard.accumulated) / lambda
                        };
                        self.hazards[trans_idx] = Some(hazard);
                        self.pending[trans_idx] = Some(firing_at);
                        if self.config.journal {
                            self.journal.push(JournalRecord::TransitionScheduled {
                                time: self.time,
                                transition: self.model.transitions[trans_idx].name.clone(),
                                firing_at,
                            });
                        }
                        continue;
                    }
                    // A paused countdown resumes where it stopped.
                    if let Some(remaining) = self.frozen[trans_idx].take() {
                        let firing_at = self.time + remaining;
                        self.pending[trans_idx] = Some(firing_at);
                        if self.config.journal {
                            self.journal.push(JournalRecord::TransitionScheduled {
                                time: self.time,
                                transition: self.model.transitions[trans_idx].name.clone(),
                                firing_at,
                            });
                        }
                        continue;
                    }
                    // `schedule_stochastic`: stochastic firing dates are sampled at
                    // source-state entry. Draws happen here, in
                    // transition-index order: replay is bit-identical
                    // for a fixed (seed, stream).
                    let time_now = self.time;
                    let bad_law = move |detail: String| EngineError::TypeError {
                        time: time_now,
                        detail,
                    };
                    let firing_at = match &self.model.transitions[trans_idx].distrib {
                        CLaw::Delay(delay) => self.time + delay,
                        CLaw::Inst(_) => self.time,
                        CLaw::Watched { .. } => continue,
                        // Armed by the dedicated block above.
                        CLaw::ExpVar { .. } => continue,
                        CLaw::Exp(rate) => {
                            let distribution = rand_distr::Exp::new(*rate)
                                .map_err(|e| bad_law(format!("exp({rate}): {e}")))?;
                            self.time + distribution.sample(&mut self.rng)
                        }
                        CLaw::Weibull(shape, scale) => {
                            let distribution = rand_distr::Weibull::new(*scale, *shape)
                                .map_err(|e| bad_law(format!("weibull: {e}")))?;
                            self.time + distribution.sample(&mut self.rng)
                        }
                        CLaw::Lognormal(mu, sigma) => {
                            let distribution = rand_distr::LogNormal::new(*mu, *sigma)
                                .map_err(|e| bad_law(format!("lognormal: {e}")))?;
                            self.time + distribution.sample(&mut self.rng)
                        }
                        CLaw::Gamma(shape, scale) => {
                            let distribution = rand_distr::Gamma::new(*shape, *scale)
                                .map_err(|e| bad_law(format!("gamma: {e}")))?;
                            self.time + distribution.sample(&mut self.rng)
                        }
                        CLaw::Uniform(low, high) => {
                            let distribution = rand_distr::Uniform::new(*low, *high)
                                .map_err(|e| bad_law(format!("uniform: {e}")))?;
                            self.time + distribution.sample(&mut self.rng)
                        }
                        CLaw::Empirical(points) => {
                            let u: f64 = rand::Rng::random(&mut self.rng);
                            self.time + sample_empirical(points, u)
                        }
                    };
                    self.pending[trans_idx] = Some(firing_at);
                    if self.config.journal {
                        self.journal.push(JournalRecord::TransitionScheduled {
                            time: self.time,
                            transition: self.model.transitions[trans_idx].name.clone(),
                            firing_at,
                        });
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn record_indicators(&mut self) {
        for (indicator, series) in self
            .model
            .indicators
            .iter()
            .zip(self.indicator_series.iter_mut())
        {
            let value = match indicator.target {
                CIndicatorTarget::Var(idx) => self.vars[idx],
                CIndicatorTarget::State(aut, state) => {
                    Value::Float(if self.states[aut] == state { 1.0 } else { 0.0 })
                }
            };
            let changed = series.points.last().is_none_or(|(_, last)| *last != value);
            if changed {
                series.points.push((self.time, value));
            }
        }
    }

    /// Record pending sample instants strictly before `t` with the
    /// *current* (pre-jump) state: piecewise-constant hold for the
    /// discrete-only case.
    fn flush_samples_before(&mut self, t: f64) {
        self.flush_samples(t, false);
    }

    /// Record pending sample instants up to and including `t` (end of
    /// run).
    fn flush_samples_through(&mut self, t: f64) {
        self.flush_samples(t, true);
    }

    fn flush_samples(&mut self, t: f64, inclusive: bool) {
        while self.sample_cursor < self.config.samples.len() {
            let s = self.config.samples[self.sample_cursor];
            let due = if inclusive { s <= t } else { s < t };
            if !due {
                break;
            }
            for (indicator, series) in self.model.indicators.iter().zip(self.sampled.iter_mut()) {
                let value = match indicator.target {
                    CIndicatorTarget::Var(idx) => self.vars[idx],
                    CIndicatorTarget::State(aut, state) => {
                        Value::Float(if self.states[aut] == state { 1.0 } else { 0.0 })
                    }
                };
                series.points.push((s, value));
            }
            self.sample_cursor += 1;
        }
    }
}

/// Classify a compiled occurrence law for interactive inspection
/// ([`Engine::fireable`]).
fn fireable_kind(distrib: &CLaw) -> FireableKind {
    match distrib {
        CLaw::Delay(_) => FireableKind::Delay,
        CLaw::Inst(_) => FireableKind::Inst,
        CLaw::Watched { .. } => FireableKind::Watched,
        CLaw::Exp(_)
        | CLaw::ExpVar { .. }
        | CLaw::Weibull(..)
        | CLaw::Lognormal(..)
        | CLaw::Gamma(..)
        | CLaw::Uniform(..)
        | CLaw::Empirical(_) => FireableKind::Stochastic,
    }
}

/// Inverse-CDF sampling from a validated empirical table: `u` below the
/// first cumulative probability maps to the first time (probability
/// mass); between points the CDF is linearly interpolated.
fn sample_empirical(points: &[(f64, f64)], u: f64) -> f64 {
    let (first_t, first_c) = points[0];
    if u <= first_c {
        return first_t;
    }
    for window in points.windows(2) {
        let (t0, c0) = window[0];
        let (t1, c1) = window[1];
        if u <= c1 {
            if c1 == c0 {
                return t1;
            }
            return t0 + (t1 - t0) * (u - c0) / (c1 - c0);
        }
    }
    // u ≤ 1 and the validated table ends at cumulative 1.
    points[points.len() - 1].0
}
