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
//! - [`discretised`]: the **discretised** algorithm, for every law the
//!   engine carries and continuous evolution: the distribution of the
//!   random next event is cut into equal-mass cells, each a branch, with
//!   an error estimate by refinement.
//! - [`result`]: the result type and its open format,
//!   `raichu.exploration` (version 1 for an exact result, 2 for a
//!   discretised one).
//!
//! The engine stays the single source of semantics: the explorer drives
//! one [`raichu_core::Engine`] per worker, never re-implementing guards,
//! effects, sensitive functions or interruption policies.

pub mod discretised;
pub mod exact;
pub mod result;
mod walk;

pub use discretised::{
    explore_discretised, DiscretisedSettings, DEFAULT_LEVEL, DEFAULT_MAX_BRANCHES,
};
pub use exact::{exact_domain_report, explore_exact, DomainViolation, ExactSettings};
pub use result::{
    read_exploration, Algorithm, CutoffTallies, CutoffTally, Cutoffs, Discretisation,
    ExplorationResult, ExploredEvent, ExploredSequence, ExploredStep, Precision,
    ReadExplorationError, Refinement, DEFAULT_GAP_TOLERANCE, EXPLORATION_FORMAT,
    EXPLORATION_VERSION,
};
