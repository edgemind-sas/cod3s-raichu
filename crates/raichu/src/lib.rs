//! # RAICHU — Rust Automata InCredibly Hybrid Unleashed
//!
//! Native, open-source Rust engine for the **hybrid simulation of complex
//! systems**: discrete stochastic behaviour (random failures, repairs,
//! reconfigurations) tightly coupled with continuous multiphysics
//! evolution (ODEs). Grounded in Piecewise-Deterministic Markov Processes
//! (PDMP) realised as Distributed Stochastic Hybrid Automata (DSHA).
//!
//! This umbrella crate re-exports the public API of the engine crates:
//!
//! - [`raichu_model`] — the native formalism (components, ports,
//!   interfaces, automata, transitions).
//! - [`raichu_expr`] — serializable expression trees (guards, effects,
//!   ODE right-hand sides).
//! - [`raichu_core`] — scheduler and simulation cycle.
//! - [`raichu_io`] — model I/O and cod3s interop.
//! - [`raichu_rng`] — reproducible RNG streams.
//! - [`raichu_numeric`] — continuous evolution (milestone M1).
//!
//! The engine upholds standing validation / performance / observability
//! contracts across its milestone sequence.

pub use raichu_core;
pub use raichu_expr;
pub use raichu_io;
pub use raichu_model;
pub use raichu_montecarlo;
pub use raichu_numeric;
pub use raichu_rng;

/// Engine version (single source of truth: the workspace `Cargo.toml`,
/// mirrored into the Python wheel by maturin as `pyraichu.__version__`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_matches_workspace_package() {
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
        assert!(!VERSION.is_empty());
    }
}
