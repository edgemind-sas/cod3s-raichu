//! # raichu-core: scheduler and simulation cycle
//!
//! Implements the operational semantics of Desgeorges et al. 2021:
//! `init → schedule → continuous → discrete → update`, mapped to the
//! paper's rules (`schedule_deterministic`, `schedule_stochastic`, `schedule_boundary`, `integrate_continuous`, `fire_transition`, `propagate_effects`,
//! `reschedule_modifiable`, `drop_disabled`) so the correspondence stays auditable: see
//! the [`engine`] module documentation for the per-rule mapping of the
//! M0 deterministic subset.
//!
//! Standing requirements:
//!
//! - **Fixpoint semantics:** sensitive-function propagation runs to a
//!   fixpoint in a documented deterministic order (global declaration
//!   order); cross-validation compares the *converged state*, never the
//!   propagation path. The confluence probe flags models whose result
//!   would depend on the order, and an
//!   iteration cap turns instantaneous loops into typed errors.
//! - **Performance:** names are resolved to dense indices at compile
//!   time ([`compile::CompiledModel`]); the propagation loop reuses its
//!   worklist and touches only vector indices.
//! - **Causal journal:** structured, toggleable records
//!   (event → triggered functions → attribute changes → rescheduling),
//!   zero-cost when off.
//! - The engine is **not** a process singleton: any number of
//!   [`engine::Engine`]s per process; single-trajectory runs are
//!   deterministic and single-threaded so replays are exact.

pub mod compile;
pub mod engine;
pub mod fault_tree;
pub mod flow;
pub mod loops;
pub mod triggers;

pub use compile::{
    CIndicator, CIndicatorTarget, CompileError, CompiledModel, MarginIndex, WatchedIdx,
};
pub use engine::{
    DeferredProbe, DeferredTransition, DropReason, Engine, EngineConfig, EngineError, Event,
    Fireable, FireableKind, FlowConfig, FlowStall, HazardSample, IndicatorSeries, JournalRecord,
    ProbeStop, Provenance, SeqEvent, Sequence, SimulationResult, Snapshot, StochasticDates,
    TransitionExposure, WorkCounters,
};
pub use fault_tree::{
    fault_tree, BasicLaw, FaultTree, FaultTreeError, FaultTreeSettings, FtNode, GateOp, TreeEvent,
};
pub use flow::{CPolicy, EdgeClass, FLOW_TOLERANCE};
pub use loops::{switching_loops, SwitchingLoop};
pub use raichu_numeric::{SolverParams, SolverStats};
pub use triggers::{unfed_triggers, UnfedTrigger};
