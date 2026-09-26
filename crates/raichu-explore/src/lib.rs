//! # raichu-explore: sequence-tree exploration drivers
//!
//! A driver beside the Monte-Carlo one (`raichu-montecarlo`): instead of
//! drawing trajectories, it explores the tree of the sequences a model can
//! follow from its initial state to a declared feared event (a model
//! [`raichu_model::Target`]), and reports each retained sequence with its
//! probability at a horizon, plus guaranteed bounds on what the declared
//! cut-offs left out. This is the FIGSEQ / PyCATSHOO sequence-tree
//! lineage.
//!
//! - [`exact`]: the **exact** algorithm on the Markov family
//!   (instantaneous branchings and exponential laws whose rate is
//!   constant between jumps, no continuous evolution). Each sequence
//!   probability is computed in closed form, in nonnegative arithmetic.
//! - [`result`]: the result type and its open format,
//!   `raichu.exploration` v1.
//!
//! The engine stays the single source of semantics: the explorer drives
//! one [`raichu_core::Engine`] per worker, never re-implementing guards,
//! effects, sensitive functions or interruption policies.

pub mod exact;
pub mod result;

pub use exact::{exact_domain_report, explore_exact, DomainViolation, ExactSettings};
pub use result::{
    read_exploration, Algorithm, CutoffTallies, CutoffTally, Cutoffs, ExplorationResult,
    ExploredEvent, ExploredSequence, ExploredStep, Precision, ReadExplorationError,
    DEFAULT_GAP_TOLERANCE, EXPLORATION_FORMAT, EXPLORATION_VERSION,
};
