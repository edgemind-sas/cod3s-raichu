//! # raichu-rng: reproducible randomness
//!
//! Seeding and substream policy (reproducibility by construction):
//!
//! - one explicit **master seed** per study;
//! - one independent **substream per Monte-Carlo replica**
//!   (`ChaCha8Rng::set_stream`: 2⁶⁴ independent streams by
//!   construction, bit-reproducible across platforms);
//! - no global RNG anywhere: the engine receives its generator
//!   explicitly and draws only at scheduling points, in deterministic
//!   (transition-index) order, so a single trajectory replays
//!   bit-identically.
//!
//! `rand_distr` is used with `std_math` **off** so sampled values are
//! identical across platforms (deliberate stack decision).
//!
//! # Stream partition of a biased campaign
//!
//! A biased (cross-entropy) study draws from one master seed in two
//! kinds of campaign: pilot iterations that fit the rate factors, then
//! one final campaign that makes the estimate. The 2⁶⁴ streams of the
//! seed are split so that neither ever reuses a draw of the other:
//!
//! - the **final campaign** draws replica `i` on stream `i`
//!   ([`final_stream`]), exactly as plain Monte-Carlo does, so with
//!   every factor at 1 it *is* the Monte-Carlo campaign of the same
//!   seed, bit for bit. Its range is `[0, 2⁶³)`;
//! - **pilot iteration `k`** draws replica `i` on stream
//!   `2⁶³ + k·B + i` ([`pilot_stream`]), with a block `B = 2³²`
//!   ([`PILOT_BLOCK`]) strictly larger than the largest admissible pilot
//!   ([`MAX_PILOT_SIZE`] = 2³¹ replicas), so two iterations never share
//!   a stream.
//!
//! **No overflow.** With `k < 2³¹` ([`MAX_PILOT_ITERATIONS`]) and
//! `i < 2³¹`, the largest stream is
//! `2⁶³ + (2³¹ − 1)·2³² + 2³¹ − 1 = 2⁶⁴ − 2³¹ − 1 < 2⁶⁴`, so the sum
//! fits a `u64` over the whole admissible range; it is computed with
//! checked arithmetic all the same, and anything outside that range is
//! refused with a [`StreamError`] rather than wrapped onto another
//! campaign's streams.

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

/// First stream of the pilot range: every pilot stream is at or above
/// it, every final-campaign stream below it (`2⁶³`).
pub const PILOT_STREAM_BASE: u64 = 1 << 63;

/// Number of streams reserved per pilot iteration (`2³²`), strictly
/// larger than [`MAX_PILOT_SIZE`] so consecutive iterations never meet.
pub const PILOT_BLOCK: u64 = 1 << 32;

/// Largest admissible number of replicas in one pilot iteration
/// (`2³¹`, about 2.1 billion): far beyond any affordable pilot, and
/// below [`PILOT_BLOCK`].
pub const MAX_PILOT_SIZE: u64 = 1 << 31;

/// Number of admissible pilot iterations (`2³¹`): iteration indices run
/// over `0..MAX_PILOT_ITERATIONS`, which keeps the last block inside
/// the `u64` stream space (see the crate docs for the arithmetic).
pub const MAX_PILOT_ITERATIONS: u64 = 1 << 31;

/// Number of admissible replicas in a final campaign (`2⁶³`): its
/// streams are `0..MAX_FINAL_REPLICAS`, all below [`PILOT_STREAM_BASE`].
pub const MAX_FINAL_REPLICAS: u64 = PILOT_STREAM_BASE;

// Compile-time proof of the partition: a block holds a whole pilot,
// and the last admissible pilot stream fits a `u64` (overflow in const
// evaluation is a build error, so changing a constant carelessly fails
// the build instead of wrapping at run time).
const _: () = assert!(PILOT_BLOCK > MAX_PILOT_SIZE);
const _: () = assert!(MAX_FINAL_REPLICAS <= PILOT_STREAM_BASE);
const _: u64 = PILOT_STREAM_BASE + (MAX_PILOT_ITERATIONS - 1) * PILOT_BLOCK + (MAX_PILOT_SIZE - 1);

/// A stream request outside the partition of a biased campaign.
///
/// Every variant is a request that would otherwise land on a stream
/// another campaign owns (or overflow the stream index), so it is
/// refused instead of wrapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamError {
    /// A final-campaign replica index at or above
    /// [`MAX_FINAL_REPLICAS`], which would enter the pilot range.
    FinalReplicaOutOfRange {
        /// Replica index requested.
        replica: u64,
        /// Exclusive upper bound on the replica index.
        max: u64,
    },
    /// A pilot larger than [`MAX_PILOT_SIZE`] replicas.
    PilotSizeTooLarge {
        /// Pilot size requested.
        size: u64,
        /// Largest admissible pilot size.
        max: u64,
    },
    /// A pilot replica index not below the pilot size.
    PilotReplicaOutOfRange {
        /// Replica index requested.
        replica: u64,
        /// Pilot size it was requested in.
        size: u64,
    },
    /// A pilot iteration index at or above [`MAX_PILOT_ITERATIONS`].
    PilotIterationOutOfRange {
        /// Iteration index requested.
        iteration: u64,
        /// Exclusive upper bound on the iteration index.
        max: u64,
    },
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::FinalReplicaOutOfRange { replica, max } => write!(
                f,
                "final-campaign replica {replica} is out of range: at most {max} replicas \
                 (the streams above belong to the pilot iterations)"
            ),
            Self::PilotSizeTooLarge { size, max } => write!(
                f,
                "pilot size {size} exceeds the maximum of {max} replicas per pilot iteration"
            ),
            Self::PilotReplicaOutOfRange { replica, size } => write!(
                f,
                "pilot replica {replica} is out of range for a pilot of {size} replicas"
            ),
            Self::PilotIterationOutOfRange { iteration, max } => write!(
                f,
                "pilot iteration {iteration} is out of range: at most {max} pilot iterations"
            ),
        }
    }
}

impl std::error::Error for StreamError {}

/// Stream of replica `replica` in the **final campaign** of a biased
/// study: the replica index itself, as in plain Monte-Carlo, so the
/// final campaign with every factor at 1 reproduces the Monte-Carlo
/// campaign of the same seed.
///
/// # Errors
///
/// [`StreamError::FinalReplicaOutOfRange`] when `replica` is at or
/// above [`MAX_FINAL_REPLICAS`] (it would enter the pilot range).
pub fn final_stream(replica: u64) -> Result<u64, StreamError> {
    if replica >= MAX_FINAL_REPLICAS {
        return Err(StreamError::FinalReplicaOutOfRange {
            replica,
            max: MAX_FINAL_REPLICAS,
        });
    }
    Ok(replica)
}

/// Stream of replica `replica` in **pilot iteration `iteration`** of a
/// pilot of `pilot_size` replicas: `2⁶³ + iteration·2³² + replica`.
///
/// Disjoint from every final-campaign stream and from every other
/// iteration's streams, over the whole admissible range (see the crate
/// docs, *Stream partition of a biased campaign*).
///
/// # Errors
///
/// [`StreamError::PilotSizeTooLarge`] when `pilot_size` exceeds
/// [`MAX_PILOT_SIZE`], [`StreamError::PilotReplicaOutOfRange`] when
/// `replica >= pilot_size`, [`StreamError::PilotIterationOutOfRange`]
/// when `iteration >= MAX_PILOT_ITERATIONS`.
pub fn pilot_stream(iteration: u64, replica: u64, pilot_size: u64) -> Result<u64, StreamError> {
    if pilot_size > MAX_PILOT_SIZE {
        return Err(StreamError::PilotSizeTooLarge {
            size: pilot_size,
            max: MAX_PILOT_SIZE,
        });
    }
    if replica >= pilot_size {
        return Err(StreamError::PilotReplicaOutOfRange {
            replica,
            size: pilot_size,
        });
    }
    let out_of_range = StreamError::PilotIterationOutOfRange {
        iteration,
        max: MAX_PILOT_ITERATIONS,
    };
    if iteration >= MAX_PILOT_ITERATIONS {
        return Err(out_of_range);
    }
    // Cannot fail over the admissible range (crate docs); checked so a
    // future change of the constants fails loudly instead of wrapping.
    iteration
        .checked_mul(PILOT_BLOCK)
        .and_then(|offset| PILOT_STREAM_BASE.checked_add(offset))
        .and_then(|block| block.checked_add(replica))
        .ok_or(out_of_range)
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
