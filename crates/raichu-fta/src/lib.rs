//! # raichu-fta: fault-tree quantification
//!
//! A fault tree, generated from a model by [`raichu_core::fault_tree`] or
//! read from the OpenPSA model-exchange format, gets its top-event
//! probability, its minimal cut sets and the importance of each basic
//! event.
//!
//! This crate is the **exact** engine: reduced ordered binary decision
//! diagrams (Rauzy 1993), one per independent module (Dutuit and Rauzy
//! 1996), with the importance measures read off the diagrams (Dutuit and
//! Rauzy 2001) and the minimal cut sets extracted into a zero-suppressed
//! diagram (Rauzy 1993). The methods are implemented from these
//! publications. A result always says which method produced it and
//! records the variable order the diagrams were built in, since their size
//! depends on it.
//!
//! What the exact engine does not do, and says so rather than
//! approximating: a diagram that outgrows its declared budget is an error
//! naming the budget; the minimal cut sets of a non-coherent tree (one
//! whose top is not monotone in its events) are not computed, although
//! its probability and importance measures are, exactly.
//!
//! ```
//! use raichu_fta::{quantify, BasicEvent, Formula, Law, QuantifySettings, Tree};
//!
//! let event = |name: &str, probability: f64| BasicEvent {
//!     name: name.to_owned(),
//!     law: Law::Constant { probability },
//! };
//! // Two pumps in parallel, behind one valve in series.
//! let tree = Tree {
//!     name: "cooling".to_owned(),
//!     events: vec![event("P1", 0.1), event("P2", 0.1), event("V", 0.01)],
//!     gates: Vec::new(),
//!     top: Formula::or(vec![
//!         Formula::and(vec![Formula::event(0), Formula::event(1)]),
//!         Formula::event(2),
//!     ]),
//! };
//! let result = quantify(&tree, &QuantifySettings::default())?;
//! let exact = 1.0 - (1.0 - 0.1 * 0.1) * (1.0 - 0.01);
//! assert!((result.probability - exact).abs() < 1e-15);
//! assert_eq!(result.minimal_cut_sets, Some(vec![vec![2], vec![0, 1]]));
//! # Ok::<(), raichu_fta::FtaError>(())
//! ```

mod dd;
mod from_core;
mod law;
mod open_psa;
mod quantify;
mod tree;

pub use law::Law;
pub use open_psa::{read_open_psa, write_open_psa};
pub use quantify::{
    quantify, EventImportance, ModuleRecord, Provenance, Quantification, QuantifySettings,
};
pub use tree::{BasicEvent, Formula, GateDef, Tree};

/// The stack given to [`quantify`] and [`read_open_psa`]: reserved
/// address space, committed only as it is used.
const STACK_BYTES: usize = 1 << 30;

/// Run `work` on a scoped thread with [`STACK_BYTES`] of stack. A panic in
/// `work` is a bug and is propagated as one; a thread that cannot be
/// spawned is a [`FtaError::Resource`].
fn with_large_stack<T: Send>(
    work: impl FnOnce() -> Result<T, FtaError> + Send,
) -> Result<T, FtaError> {
    std::thread::scope(|scope| {
        match std::thread::Builder::new()
            .name("raichu-fta".to_owned())
            .stack_size(STACK_BYTES)
            .spawn_scoped(scope, work)
        {
            Ok(handle) => match handle.join() {
                Ok(value) => value,
                Err(payload) => std::panic::resume_unwind(payload),
            },
            Err(e) => Err(FtaError::Resource(format!(
                "cannot spawn the worker thread: {e}"
            ))),
        }
    })
}

/// Why a tree could not be quantified.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FtaError {
    /// The tree is malformed: a dangling index, a duplicate name, a gate
    /// reaching itself.
    #[error("fault tree: {0}")]
    Invalid(String),
    /// An OpenPSA document could not be read.
    #[error("OpenPSA: {0}")]
    OpenPsa(String),
    /// A basic event's law has parameters it cannot take.
    #[error("fault tree: basic event `{event}`: {detail}")]
    BadLaw {
        /// The event.
        event: String,
        /// What is wrong.
        detail: String,
    },
    /// A basic event's law depends on time and no mission time was given.
    #[error(
        "fault tree: basic event `{event}` follows a {law} law, so its \
         probability depends on the mission time; give one"
    )]
    MissionTimeRequired {
        /// The event.
        event: String,
        /// Its law.
        law: &'static str,
    },
    /// The mission time is not a finite non-negative number.
    #[error("fault tree: mission time {0} is not a finite non-negative number")]
    BadMissionTime(f64),
    /// A decision diagram outgrew the declared budget.
    #[error("fault tree: {0}")]
    TooLarge(String),
    /// The system refused a resource the computation needs.
    #[error("fault tree: {0}")]
    Resource(String),
}
