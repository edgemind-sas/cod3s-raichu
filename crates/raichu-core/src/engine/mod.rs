//! The deterministic simulation engine: one trajectory of a compiled
//! model, discrete and continuous evolution together.
//!
//! Implements the cycle `init → schedule → continuous → discrete →
//! update` of Desgeorges et al. (2021), whose operational semantics is
//! one initialisation axiom and eight inference rules. Each rule has a
//! descriptive name in this crate (a name for the rule, not a function);
//! the paper's own name follows it, then the functions that implement it,
//! so the mapping to the paper stays auditable:
//!
//! - initialisation (the axiom): `Engine::initialize`, called by
//!   [`Engine::new`];
//! - scheduling of deterministic transitions, `schedule_deterministic`
//!   (the paper's `schDT`): `Engine::refresh_schedule`;
//! - scheduling of stochastic transitions, `schedule_stochastic`
//!   (`schST`): `Engine::refresh_schedule`, which draws a firing date
//!   from the transition's law (an exponential hazard threshold for a
//!   state-dependent rate; no date in deferred-draw mode, where
//!   `Engine::refresh_deferred` arms the transition without a draw);
//! - watched transitions fired at located boundary crossings,
//!   `schedule_boundary` (`schWT`): margin monitoring inside
//!   `Engine::integrate_to`;
//! - continuous evolution up to the next scheduled date,
//!   `integrate_continuous` (`evolC`): `Engine::advance_continuous` and
//!   `Engine::integrate_to`;
//! - firing of the earliest transition, `fire_transition` (`evolT`):
//!   [`Engine::step`];
//! - sensitive-function propagation to fixpoint, `propagate_effects`
//!   (`evolA`): `Engine::run_fixpoint`;
//! - rescheduling of modifiable transitions whose rate changed,
//!   `reschedule_modifiable` (`updateMT`): through the cumulative-hazard
//!   realisation of state-dependent rates (`CLaw::ExpVar`), a
//!   piecewise-constant rate is rescheduled at each discrete change
//!   (`Engine::refresh_schedule`) and a continuously-varying rate is
//!   integrated alongside the ODE state, its firing located like a
//!   boundary crossing (`Engine::integrate_to`);
//! - dropping interruptible transitions whose guard turned false,
//!   `drop_disabled` (`updateIT`): `Engine::refresh_schedule`.
//!
//! The functions written without a link are internal to this module.
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
//!
//! **Layout.** This module holds the [`Engine`] struct and its
//! constructors; the rest lives in submodules, each adding its own
//! `impl Engine` block: `config` (configuration, errors, public types),
//! `eval` (expression evaluation), `flow` (flow resolution), `ode` (the
//! continuous system handed to the solver), `continuous` (integration and
//! event location), `schedule` (scheduling), `run` (run loop, firing,
//! fixpoint), `interactive` (interactive surface), `deferred`
//! (deferred-draw mode), `notes` (change log, watched index, journal
//! notes) and `sampling` (indicators). Every public item is re-exported
//! here, so its path is `raichu_core::engine::<name>`.

use crate::compile::{
    AutIdx, CAllocation, CExpr, CFlowMargins, CIndicatorTarget, CLaw, CProgramAffine,
    CProgramTieBreak, CStep, CompiledModel, FnIdx, StateIdx, TransIdx, VarIdx, WatchedIdx,
};
use crate::flow::{allocate, classify, edge_margin, flow_band, EdgeClass, FLOW_TOLERANCE};
use raichu_expr::{AggOp, BoolOp, CmpOp, Value};
use raichu_numeric::{DormandPrince45, OdeSolver, OdeSystem, Outcome, SolverParams};
use rand_chacha::ChaCha8Rng;
use rand_distr::Distribution;
use serde::Serialize;
use std::collections::BTreeSet;
use thiserror::Error;

mod config;
mod continuous;
mod deferred;
mod eval;
mod flow;
mod interactive;
mod notes;
mod ode;
mod run;
mod sampling;
mod schedule;

pub use config::{
    DeferredProbe, DeferredTransition, DropReason, EngineConfig, EngineError, Event, Fireable,
    FireableKind, FlowConfig, FlowStall, HazardSample, IndicatorSeries, JournalRecord, ProbeStop,
    Provenance, SeqEvent, Sequence, SimulationResult, Snapshot, StochasticDates,
    TransitionExposure, WorkCounters,
};
pub use flow::{active_set_budget, FLOW_RELAXATION, FLOW_SWEEP_BUDGET};

pub(crate) use eval::eval_frozen;

// Items shared across the submodules, which reach them through
// `use super::*`.
use config::{attribute_of, state_of};
use deferred::DeferredAge;
use eval::{eval_bool, eval_expr, eval_f64, predicate_holds};
use flow::{
    flows_settled, read_flow_inputs, recompute_explicit, value_settled, FlowScratch, PassContext,
};
use notes::ChangeLog;
use ode::{ContinuousSystem, FrozenFlow};
use sampling::indicator_value;
use schedule::{fireable_kind, validate_rate_factors, ExposureTally, Hazard};

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
    /// Age bookkeeping per transition armed in deferred mode
    /// ([`StochasticDates::Deferred`]); always `None` in drawn mode. A
    /// deferred state-dependent rate keeps its cumulative hazard in
    /// `hazards`, against an infinite threshold.
    deferred: Vec<Option<DeferredAge>>,
    /// Dense hazard samples being recorded by [`Engine::probe_deferred`]
    /// (`None` outside a probe): scratch, never part of the trajectory.
    hazard_trace: Option<Vec<HazardSample>>,
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
    /// Per transition: how many times it has fired, and when it first
    /// did. The pair is what turns a runaway count into a diagnosis, the
    /// span being what says a cycle is numerical rather than physical.
    firings: Vec<u64>,
    first_firing: Vec<f64>,
    /// Segment restarts of the flow network over the whole run, and when
    /// the first one happened: the flow-side counterpart of `firings`.
    flow_restarts: u64,
    first_flow_restart: f64,
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
    /// Per-transition firing counts and nominal exposure, present exactly
    /// when [`EngineConfig::rate_factors`] is non-empty (see
    /// [`TransitionExposure`]). Trajectory state: part of the snapshot.
    exposure: Option<ExposureTally>,
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
        validate_rate_factors(model, &config)?;
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
    ///
    /// This constructor cannot fail, so it does not check
    /// [`EngineConfig::rate_factors`]: the caller passes the configuration
    /// the snapshot was taken under, which [`Engine::new`] validated. The
    /// exposure statistics travel with the snapshot.
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
            deferred: vec![None; model.transitions.len()],
            hazard_trace: None,
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
            firings: vec![0; model.transitions.len()],
            first_firing: vec![f64::NAN; model.transitions.len()],
            flow_restarts: 0,
            first_flow_restart: f64::NAN,
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
            exposure: (!config.rate_factors.is_empty()).then(|| ExposureTally::new(model)),
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
        self.worklist
            .extend(0..self.model.functions.len() + self.model.programs.len());
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
}
