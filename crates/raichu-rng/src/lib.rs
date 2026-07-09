//! # raichu-rng — reproducible randomness
//!
//! Seeding and substream policy (reproducibility by construction):
//!
//! - one explicit **master seed** per study;
//! - one independent **substream per Monte-Carlo replica**
//!   (`ChaCha8Rng::set_stream` — 2⁶⁴ independent streams by
//!   construction, bit-reproducible across platforms);
//! - no global RNG anywhere: the engine receives its generator
//!   explicitly and draws only at scheduling points, in deterministic
//!   (transition-index) order, so a single trajectory replays
//!   bit-identically.
//!
//! `rand_distr` is used with `std_math` **off** so sampled values are
//! identical across platforms (deliberate stack decision).

use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Build the generator of one replica: master seed + substream index.
///
/// Streams are independent for distinct `stream` values under the same
/// master seed; the same `(master, stream)` pair always yields the same
/// sequence (provenance: record both).
#[must_use]
pub fn replica_rng(master: u64, stream: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(master);
    rng.set_stream(stream);
    rng
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::RngCore;

    #[test]
    fn same_seed_same_stream_is_identical() {
        let mut a = replica_rng(42, 7);
        let mut b = replica_rng(42, 7);
        let seq_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn distinct_streams_diverge() {
        let mut a = replica_rng(42, 0);
        let mut b = replica_rng(42, 1);
        let seq_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_ne!(seq_a, seq_b);
    }
}
