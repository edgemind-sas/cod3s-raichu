//! # raichu-core — scheduler and simulation cycle
//!
//! Implements the operational semantics of Desgeorges et al. 2021:
//! `init → schedule → continuous → discrete → update`, mapped to the
//! paper's rules (`schedule_deterministic`, `schedule_stochastic`, `schedule_boundary`, `integrate_continuous`, `fire_transition`, `propagate_effects`,
//! `reschedule_modifiable`, `drop_disabled`) so the correspondence stays auditable — see
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

pub use compile::{CompileError, CompiledModel};
pub use engine::{
    DropReason, Engine, EngineConfig, EngineError, Event, Fireable, FireableKind, IndicatorSeries,
    JournalRecord, Provenance, SimulationResult, Snapshot,
};
pub use raichu_numeric::SolverParams;
