//! Engine configuration, errors, and the public types a run reads and
//! returns (events, sequences, snapshots, indicator series, work counters,
//! provenance, the interactive and deferred-draw views).

use super::*;

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
    /// Explicit permission to execute imported FMU native code, off by default.
    pub allow_fmu_import: bool,
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
    /// Safety cap on how many times ONE transition may fire in a single
    /// trajectory. Beyond it the model is declared to be chattering
    /// (typed error, not a run that never ends). `0` disables the cap.
    ///
    /// The two Zeno guards this sits beside catch a loop that does not
    /// advance time: [`EngineError::WatchedLoop`] counts firings at one
    /// instant, [`EngineError::FlowChattering`] counts segment restarts
    /// at one instant. Neither sees a **limit cycle**, where time does
    /// advance, by a little, every turn. A volume oscillating across its
    /// bound at the scale of its hysteresis is the case this was built
    /// for: it fires hundreds of thousands of times, produces a
    /// trajectory that looks right at every sample instant, and never
    /// finishes.
    pub max_transition_firings: u64,
    /// Safety cap on how many times the continuous flow network may
    /// restart an integration segment in a single trajectory. Beyond it
    /// the active set is declared to be chattering (typed error, not a
    /// run that never ends). `0` disables the cap.
    ///
    /// The companion of [`Self::max_transition_firings`] on the flow
    /// side, and it exists for the same blind spot.
    /// [`EngineError::FlowChattering`] resets its count as soon as the
    /// clock moves by one event-location tolerance, so a cycle that
    /// advances time by a little every turn escapes it: the run does not
    /// fail, it grinds. A plant restarting a segment on every accepted
    /// solver step is the case this was built for.
    pub max_flow_restarts: u64,
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
    /// How the firing dates of stochastic transitions are handled:
    /// drawn when the transition is armed (the default, the Monte-Carlo
    /// semantics), or deferred and left to the caller
    /// ([`StochasticDates::Deferred`], the seam of the discretised
    /// sequence-tree explorer).
    pub stochastic_dates: StochasticDates,
    /// Per-transition multiplicative factors on the rate of constant-rate
    /// exponential laws, indexed like `CompiledModel::transitions`: the
    /// seam of the biased Monte-Carlo sampler fitted by cross-entropy.
    ///
    /// Empty (the default) means no factor at all: nothing is biased and
    /// no statistics are collected, so the default path allocates nothing
    /// new. A non-empty vector must hold one factor per transition. A
    /// factor multiplies the rate when the date of a `CLaw::Exp`
    /// transition is drawn (`schedule_stochastic`, the paper's `schST`);
    /// the draw consumes exactly the random numbers of the nominal one,
    /// so a factor of exactly 1 leaves the trajectory bit-identical.
    /// Each factor must be finite and positive, and a factor other than
    /// 1 is accepted only on a constant-rate exponential law: every
    /// other law runs unbiased ([`EngineError::InvalidRateFactor`]).
    /// Drawn mode only: a non-empty vector under
    /// [`StochasticDates::Deferred`] is refused.
    ///
    /// When the vector is present the run reports, per transition, its
    /// firing count and nominal exposure
    /// ([`SimulationResult::rate_statistics`], [`TransitionExposure`]).
    pub rate_factors: Vec<f64>,
}

/// Sufficient statistics of one transition over a trajectory, for the
/// likelihood ratio of a biased run and for the cross-entropy update.
///
/// Collected only when [`EngineConfig::rate_factors`] is non-empty. For a
/// constant-rate exponential transition of nominal rate `λ` biased by a
/// factor `f`, the likelihood ratio of the nominal law against the biased
/// one over a trajectory is `f^(-firings) · exp((f − 1) · exposure)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct TransitionExposure {
    /// How many times the transition fired over the trajectory, whatever
    /// its law.
    pub firings: u64,
    /// Nominal cumulated hazard `λ · (time armed and running)`, with `λ`
    /// the model's rate, not the biased one. "Running" follows the
    /// interruption policy (`drop_disabled`, the paper's `updateIT`): a
    /// `reset` transition stops at the drop, a `resume` one excludes its
    /// paused stretches, a `continue` one runs through false-guard
    /// stretches; every one stops at firing and at source exit, and the
    /// last stretch is censored at the trajectory's final time (the
    /// target instant on an early stop, the horizon otherwise). Zero for
    /// every law other than a constant-rate exponential.
    pub exposure: f64,
}

/// How the engine handles the firing date of a **stochastic** transition
/// (exponential, state-dependent exponential, Weibull, lognormal, gamma,
/// uniform, empirical). Deterministic delays, instantaneous branchings and
/// watched transitions are unaffected by this choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StochasticDates {
    /// The date is sampled from the RNG when the transition is armed
    /// (`schedule_stochastic`), and a state-dependent rate is realised
    /// against an `Exp(1)` hazard threshold drawn then. The engine fires
    /// the earliest date. This is the Monte-Carlo semantics.
    #[default]
    Drawn,
    /// Arming consumes **no** random number and schedules **no** date. The
    /// engine records what a later resolution of the law needs: the
    /// arming instant, the age (time spent armed, net of the pauses of a
    /// `resume` interruption) and, for a state-dependent exponential, the
    /// cumulative hazard `H = ∫ λ dt` accumulated so far (integrated with
    /// the ODE against an infinite threshold when λ varies continuously,
    /// banked at each rate change when it is piecewise constant). The
    /// engine never fires such a transition by itself: the caller fires it
    /// at a chosen instant with [`Engine::fire_deferred_at`], and reads the
    /// bookkeeping with [`Engine::deferred`] and
    /// [`Engine::probe_deferred`].
    ///
    /// Interruption policies act on the age: `reset` drops the armed
    /// transition when its guard turns false (the age restarts from zero
    /// at the next arming), `resume` freezes the age while the guard is
    /// false, `continue` keeps the transition armed and aging, exactly as
    /// drawn mode keeps its date and fires it whatever the guard.
    Deferred,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            allow_fmu_import: false,
            t_max: f64::INFINITY,
            journal: false,
            sequences: false,
            stop_at_targets: false,
            confluence_check: false,
            max_fixpoint_iterations: 10_000,
            max_transition_firings: 100_000,
            max_flow_restarts: 100_000,
            ode: SolverParams::default(),
            samples: Vec::new(),
            seed: 0,
            rng_stream: 0,
            flow: FlowConfig::default(),
            stochastic_dates: StochasticDates::Drawn,
            rate_factors: Vec::new(),
        }
    }
}

/// Typed runtime errors. The engine never panics on a library path.
#[derive(Debug, Error)]
pub enum EngineError {
    /// The native-only snapshot API cannot capture or restore FMU state.
    #[error("FMU unit `{unit}` requires `{alternative}` for {operation}")]
    FmuSnapshotApi {
        /// Name of the imported unit.
        unit: String,
        /// Operation that was requested.
        operation: &'static str,
        /// FMU-aware API to use instead.
        alternative: &'static str,
    },
    /// Import requires the runner's explicit permission.
    #[error("FMU unit `{unit}` requires allow_fmu_import before unpacking or execution")]
    FmuPermission {
        /// Name of the unit requiring permission.
        unit: String,
    },
    /// A model importing an FMU needs a prepared host.
    #[error("FMU unit `{unit}` requires a co-simulation host")]
    FmuHostRequired {
        /// Name of the unit.
        unit: String,
    },
    /// The supplied host was prepared from another compiled model.
    #[error("co-simulation host does not match compiled model `{model}`")]
    FmuHostMismatch {
        /// Name of the requested model.
        model: String,
    },
    /// An FMU binding does not match the unit's description.
    #[error("FMU unit `{unit}` binding `{attribute}` to `{variable}` is invalid: {reason}")]
    FmuBinding {
        /// Name of the unit.
        unit: String,
        /// Qualified model attribute.
        attribute: String,
        /// FMU variable name.
        variable: String,
        /// Detailed mismatch.
        reason: String,
    },
    /// FMU I/O or stepping failed at the given engine date.
    #[error("FMU unit `{unit}` failed at t={time}: {source}")]
    FmuRuntime {
        /// Name of the unit.
        unit: String,
        /// Date of the failed operation.
        time: f64,
        /// Underlying FMI error.
        #[source]
        source: raichu_fmi::FmiError,
    },
    /// A driver requested a capability the FMU does not declare.
    #[error("FMU unit `{unit}` lacks capability `{capability}`")]
    FmuCapability {
        /// Name of the unit.
        unit: String,
        /// Missing capability.
        capability: &'static str,
    },
    /// A model-level program did not prove a usable optimum.
    #[error("program `{program}` failed at t={time}: {reason}")]
    ProgramFailed {
        /// Program name.
        program: String,
        /// Simulation time.
        time: f64,
        /// Solver outcome or invalid numeric input.
        reason: String,
    },
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
    /// Interactive control: a forced destination given by *branch index*
    /// ([`Engine::fire_now`], [`Engine::fire_idx_to_branch`]) is not a
    /// position in the transition's compiled target list.
    #[error(
        "branch {branch} is out of range for transition `{transition}` \
         ({branches} branches)"
    )]
    ForcedBranchOutOfRange {
        /// The transition whose branch was forced.
        transition: String,
        /// The rejected branch index.
        branch: usize,
        /// Number of branches the transition declares.
        branches: usize,
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
    /// An ODE right-hand side reached the magnitude the model declares
    /// as "unbounded" ([`raichu_model::Model::unbounded_rate`]): a stand-in
    /// for "no ceiling" became the rate of a stock. Integrated, it steps
    /// the stock far past its bound before any event is located, so the
    /// run is refused instead of answering a non-physical number.
    #[error(
        "`{variable}` would change at {rate:e} per unit time at t={time}, \
         the magnitude this model reserves for \"unbounded\" ({unbounded:e}): \
         an unbounded draw met an unbounded supply, whose physics is an \
         instantaneous transfer that a rate cannot express; bound one of \
         the two (a finite fill or serve rate, or a finite demand)"
    )]
    UnboundedRate {
        /// Simulation time of the evaluation.
        time: f64,
        /// The integrated variable, `component.attribute`.
        variable: String,
        /// The right-hand side value.
        rate: f64,
        /// The declared unbounded magnitude.
        unbounded: f64,
    },
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
    /// budget (see `Engine::resolve_flows`).
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
    /// The continuous flow network restarted segments past its budget:
    /// a limit cycle in the active set, where time advances a little
    /// every turn and [`Self::FlowChattering`] cannot see it.
    #[error(
        "the active set of the continuous flow network restarted \
         {restarts} segments between t={since} and t={time} (average step \
         {step:e}): the network is chattering, not evolving; edges \
         crossing most recently: {edges}. Raise `max_flow_restarts` if \
         this is genuinely a fast-switching model"
    )]
    FlowLimitCycle {
        /// Restarts counted, which is the budget.
        restarts: u64,
        /// When the first restart happened.
        since: f64,
        /// When the budget ran out.
        time: f64,
        /// Mean simulated time between two restarts: the number that
        /// says this is a cycle at a numerical scale, not a physical one.
        step: f64,
        /// Qualified `operator[consumer]` name of the edges that crossed
        /// at the last restart, comma separated.
        edges: String,
    },
    /// One transition fired past its budget: a limit cycle, where time
    /// advances a little every turn and neither Zeno guard can see it.
    #[error(
        "transition `{transition}` fired {firings} times between t={since} \
         and t={time} (average step {step:e}): the model is chattering, not \
         evolving. Raise `max_transition_firings` if this is genuinely a \
         high-frequency model"
    )]
    TransitionChattering {
        /// Qualified name of the transition that ran away.
        transition: String,
        /// Firings counted, which is the budget.
        firings: u64,
        /// When it first fired.
        since: f64,
        /// When the budget ran out.
        time: f64,
        /// Mean simulated time between two of its firings: the number
        /// that says this is a cycle at a numerical scale rather than a
        /// physical one.
        step: f64,
    },
    /// A study parameter was outside the domain the estimator is defined
    /// on (a confidence level outside `(0, 1)`, say). Raised before the
    /// campaign starts: a parameter that cannot produce a result must
    /// not cost one.
    #[error("invalid study parameter `{parameter}`: {detail}")]
    InvalidStudyParameter {
        /// Name of the offending parameter, as the caller spells it.
        parameter: String,
        /// What was expected, and what arrived.
        detail: String,
    },
    /// Exact sequence-tree exploration: the model is outside the exact
    /// (Markov) domain **before** anything is explored. It declares
    /// continuous evolution, a watched transition its automaton can
    /// reach, or an expression reading the simulation time. Raised by
    /// the static domain check, so a model that cannot be explored
    /// exactly costs nothing.
    #[error(
        "the model is outside the exact exploration domain: {}",
        .reasons.join("; ")
    )]
    OutsideExactDomain {
        /// Every reason found, in model order (one line each).
        reasons: Vec<String>,
    },
    /// Exact sequence-tree exploration: a transition whose law is
    /// neither instantaneous, zero delay, constant exponential nor
    /// piecewise-constant state-dependent exponential became armed
    /// along an explored sequence. A law that is never armed does not
    /// block; this one was, and the sequence that armed it is named.
    #[error(
        "transition `{transition}` ({law}) is armed after the sequence [{}]: \
         its law is outside the exact exploration domain (instantaneous, \
         zero delay, or exponential with a rate constant between jumps)",
        .sequence.join(", ")
    )]
    LawOutsideExactDomain {
        /// Qualified name of the armed transition.
        transition: String,
        /// Its law, as a short readable label.
        law: String,
        /// Qualified names of the transitions fired from the initial
        /// state to the node where it became armed, in firing order.
        sequence: Vec<String>,
    },
    /// Exact sequence-tree exploration: more than `firings`
    /// instantaneous transitions (instantaneous law or zero delay) fire
    /// in a row along one explored sequence, with no timed step between
    /// them. Their mass never decreases, so no probability cut-off stops
    /// the chain: it is diagnosed as a cycle of instantaneous
    /// transitions. Distinct from [`EngineError::InstantaneousLoop`],
    /// which is about sensitive functions within one fixpoint; the cap
    /// is the same, [`EngineConfig::max_fixpoint_iterations`].
    #[error(
        "more than {firings} instantaneous transitions fire in a row after the \
         sequence [{}] (the last one `{transition}`): probable cycle of \
         instantaneous or zero-delay transitions. Raise `max_fixpoint_iterations` \
         if the chain is genuinely this long",
        .sequence.join(", ")
    )]
    InstantaneousCycle {
        /// Qualified name of the instantaneous transition that would have
        /// fired past the cap.
        transition: String,
        /// Consecutive instantaneous firings allowed, which is the cap.
        firings: usize,
        /// Qualified names of the transitions fired from the initial
        /// state up to the start of the instantaneous chain (the chain
        /// itself excluded), in firing order.
        sequence: Vec<String>,
    },
    /// [`EngineConfig::rate_factors`] holds a factor the engine cannot
    /// apply: zero, negative or non-finite on any transition, or other
    /// than 1 on a transition whose law is not a constant-rate
    /// exponential (such a transition runs unbiased). Raised when the
    /// engine is built, naming the transition.
    #[error("rate factor {factor} on transition `{transition}` is refused: {reason}")]
    InvalidRateFactor {
        /// Qualified name of the transition.
        transition: String,
        /// The refused factor.
        factor: f64,
        /// Why it is refused, in words.
        reason: String,
    },
    /// [`EngineConfig::rate_factors`] is neither empty nor one factor per
    /// compiled transition. Raised when the engine is built.
    #[error(
        "rate factors: the model compiles {expected} transitions, the \
         factor vector holds {found}"
    )]
    RateFactorCount {
        /// Number of compiled transitions.
        expected: usize,
        /// Length of the factor vector received.
        found: usize,
    },
    /// An operation of the deferred-draw mode was called on an engine
    /// running with drawn dates ([`StochasticDates::Drawn`]).
    #[error("`{operation}` needs deferred stochastic dates (`StochasticDates::Deferred`)")]
    NotDeferredMode {
        /// The operation that was refused.
        operation: String,
    },
    /// Interactive control: `set_date` on a transition armed in deferred
    /// mode. Such a transition has no date to override; it is fired at a
    /// chosen instant with [`Engine::fire_deferred_at`].
    #[error(
        "transition `{transition}` is armed without a date (deferred stochastic \
         dates): fire it with `fire_deferred_at` instead of dating it"
    )]
    DeferredDate {
        /// The transition whose date was to be set.
        transition: String,
    },
    /// [`Engine::fire_deferred_at`] asked for an instant after an event
    /// the engine would have to process first: the next deterministic
    /// date, a watched boundary crossing located on the way, or the
    /// horizon. Nothing is changed.
    #[error(
        "cannot fire deferred transition `{transition}` at t={date}: {cause} \
         comes first, at t={limit}"
    )]
    DeferredFiringTooLate {
        /// The deferred transition.
        transition: String,
        /// The requested firing instant.
        date: f64,
        /// The instant of the event that comes first.
        limit: f64,
        /// What that event is, in words.
        cause: String,
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
    /// A program was solved or its exact numeric input was found in cache.
    ProgramSolved {
        /// Simulation time.
        time: f64,
        /// Program name.
        program: String,
        /// Proven status, either `optimal` or `infeasible`.
        status: &'static str,
        /// Optimal primary objective, absent when infeasible.
        objective: Option<f64>,
        /// Whether an exact-input cache entry served the outcome.
        cached: bool,
    },
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
    /// A fixed-step unit keeps its previously sampled input until its next point.
    FmuInputHeld {
        /// Date of the model event that changed the bound attribute.
        time: f64,
        /// FMU unit name.
        unit: String,
        /// Qualified model attribute supplying the input.
        attribute: String,
        /// Value already sampled for the current interval.
        held: Value,
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

/// One stochastic transition armed in deferred mode, as read by
/// [`Engine::deferred`].
#[derive(Debug, Clone)]
pub struct DeferredTransition {
    /// Transition index (the handle of [`Engine::fire_deferred_at`]).
    pub index: usize,
    /// Qualified transition name.
    pub transition: String,
    /// Its compiled occurrence law.
    pub law: CLaw,
    /// Instant it was armed (time units). A `resume` pause keeps it.
    pub armed_at: f64,
    /// Age `a` at the current instant (time units): the time spent armed
    /// with the countdown running, net of `resume` pauses. The law's
    /// conditional survival `S(a + s) / S(a)` is what resolves it.
    pub age: f64,
    /// `true` while a `resume` interruption holds it paused: it cannot
    /// fire and does not age until its guard holds again.
    pub paused: bool,
    /// For a state-dependent exponential (`CLaw::ExpVar`) only: the
    /// cumulative hazard `H = ∫ λ(x(u)) du` accumulated over the running
    /// stretches since arming (dimensionless); the survival of the
    /// remaining countdown from now is `exp(−(H(t) − H(now)))`. `None`
    /// for every other law, whose hazard is a function of the age.
    pub hazard: Option<f64>,
}

/// Why [`Engine::probe_deferred`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeStop {
    /// The next date-scheduled (deterministic) transition is due at the
    /// stop instant. It is not fired.
    Deterministic {
        /// Its transition index.
        index: usize,
    },
    /// A watched boundary crossing was located (or already holds) at the
    /// stop instant. It is not fired.
    Watched {
        /// Its transition index.
        index: usize,
    },
    /// The requested limit was reached.
    Limit,
    /// The simulation horizon `t_max` came before the requested limit.
    Horizon,
}

/// One dense sample of the cumulative hazards of the continuously
/// varying deferred transitions (see [`DeferredProbe`]).
#[derive(Debug, Clone, PartialEq)]
pub struct HazardSample {
    /// Sample instant (time units).
    pub time: f64,
    /// Cumulative hazard `H` of each transition of
    /// [`DeferredProbe::transitions`] at `time`, in that order.
    pub hazards: Vec<f64>,
}

/// Result of [`Engine::probe_deferred`].
#[derive(Debug, Clone, PartialEq)]
pub struct DeferredProbe {
    /// Instant the probe stopped at (the engine clock is there).
    pub stop: f64,
    /// Why it stopped there.
    pub reason: ProbeStop,
    /// Indices of the running deferred transitions whose rate varies
    /// continuously (`CLaw::ExpVar { continuous: true, .. }`), ascending.
    pub transitions: Vec<usize>,
    /// Their cumulative hazards along the probe, in strictly increasing
    /// time, from the start instant to `stop` inclusive: one sample at
    /// every accepted solver step end and at the interior points the
    /// solver scans for crossings (dense output). Empty when
    /// `transitions` is.
    pub samples: Vec<HazardSample>,
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
    pub(super) fmu_states: Option<Vec<Vec<u8>>>,
    pub(super) fmu_positions: Vec<(f64, u64, Option<f64>, Vec<Value>)>,
    pub(super) time: f64,
    pub(super) vars: Vec<Value>,
    pub(super) states: Vec<StateIdx>,
    pub(super) pending: Vec<Option<f64>>,
    pub(super) frozen: Vec<Option<f64>>,
    pub(super) hazards: Vec<Option<Hazard>>,
    pub(super) deferred: Vec<Option<DeferredAge>>,
    pub(super) events: Vec<Event>,
    pub(super) journal: Vec<JournalRecord>,
    pub(super) seq_events: Vec<SeqEvent>,
    pub(super) seq_end: Option<(String, f64)>,
    pub(super) indicator_series: Vec<IndicatorSeries>,
    pub(super) sampled: Vec<IndicatorSeries>,
    pub(super) sample_cursor: usize,
    pub(super) watched_streak: (f64, usize),
    pub(super) firings: Vec<u64>,
    pub(super) first_firing: Vec<f64>,
    pub(super) flow_restarts: u64,
    pub(super) first_flow_restart: f64,
    pub(super) rng: ChaCha8Rng,
    pub(super) worklist: BTreeSet<FnIdx>,
    pub(super) exposure: Option<ExposureTally>,
}

/// Value of an attribute by qualified name (`component.attribute`):
/// shared by [`Engine`] and [`Snapshot`] so the two never drift.
pub(super) fn attribute_of(
    model: &CompiledModel,
    vars: &[Value],
    qualified: &str,
) -> Option<Value> {
    model.var_index.get(qualified).map(|&idx| vars[idx])
}

/// Current state name of an automaton by qualified name
/// (`component.automaton`): shared by [`Engine`] and [`Snapshot`].
pub(super) fn state_of<'m>(
    model: &'m CompiledModel,
    states: &[StateIdx],
    qualified: &str,
) -> Option<&'m str> {
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

/// Identity of one imported FMU attached to result provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct FmuProvenance {
    /// Declared unit name.
    pub name: String,
    /// FMI major and minor version.
    pub fmi_version: String,
    /// Tool that generated the FMU, when declared.
    pub generating_tool: Option<String>,
    /// Version of the generating tool, when declared.
    pub generating_tool_version: Option<String>,
    /// FMI 2 GUID or FMI 3 instantiation token.
    pub instantiation_token: String,
    /// SHA-256 digest of the original archive, prefixed by `sha256:`.
    pub content_hash: String,
}

/// Provenance metadata attached to every result (reproducibility by construction).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Provenance {
    /// Programs that requested no tie-break, so their dispatch is not
    /// guaranteed unique across solver builds or platforms.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub non_unique_programs: Vec<String>,
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
    /// Imported co-simulation units, omitted for native-only runs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fmu_units: Vec<FmuProvenance>,
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
    /// Per-transition firing count and nominal exposure, indexed like
    /// `CompiledModel::transitions`: `Some` exactly when
    /// [`EngineConfig::rate_factors`] is non-empty. Omitted from the
    /// serialized result when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_statistics: Option<Vec<TransitionExposure>>,
}
