//! The stream partition between the final campaign of a biased
//! Monte-Carlo study and its pilot iterations.
//!
//! Two properties are checked, both of which an estimate rests on:
//!
//! - the final campaign draws replica `i` on stream `i`, so with every
//!   rate factor at 1 it is the plain Monte-Carlo campaign of the same
//!   seed, bit for bit;
//! - no pilot draw is ever reused: the pilot blocks do not intersect
//!   each other nor the final range, up to the declared maximum pilot
//!   size and iteration count, and anything beyond them is refused
//!   rather than wrapped around.

#![allow(clippy::unwrap_used, clippy::panic, clippy::expect_used)]

use raichu_rng::{
    final_stream, pilot_stream, replica_rng, StreamError, MAX_FINAL_REPLICAS, MAX_PILOT_ITERATIONS,
    MAX_PILOT_SIZE, PILOT_BLOCK, PILOT_STREAM_BASE,
};
use rand_chacha::rand_core::RngCore;

#[test]
fn final_campaign_stream_of_replica_i_is_i() {
    for i in [0, 1, 2, 999, 1 << 40, MAX_FINAL_REPLICAS - 1] {
        assert_eq!(final_stream(i), Ok(i));
    }
}

#[test]
fn final_campaign_generator_is_the_monte_carlo_one() {
    // Monte-Carlo builds replica `i` on `replica_rng(seed, i)`: the final
    // campaign must yield the very same draws.
    let mut mc = replica_rng(42, 17);
    let mut fin = replica_rng(42, final_stream(17).unwrap());
    let a: Vec<u64> = (0..8).map(|_| mc.next_u64()).collect();
    let b: Vec<u64> = (0..8).map(|_| fin.next_u64()).collect();
    assert_eq!(a, b);
}

#[test]
fn the_final_range_stops_below_the_pilot_base() {
    assert_eq!(MAX_FINAL_REPLICAS, PILOT_STREAM_BASE);
    assert_eq!(
        final_stream(MAX_FINAL_REPLICAS),
        Err(StreamError::FinalReplicaOutOfRange {
            replica: MAX_FINAL_REPLICAS,
            max: MAX_FINAL_REPLICAS,
        })
    );
}

#[test]
fn the_pilot_block_is_larger_than_any_admissible_pilot() {
    // The library also proves this at compile time; read through
    // `black_box` here so the check is a run-time one too.
    let (block, max, base) = std::hint::black_box((PILOT_BLOCK, MAX_PILOT_SIZE, PILOT_STREAM_BASE));
    assert!(block > max);
    assert_eq!(base, 1 << 63);
}

#[test]
fn pilot_streams_of_iterations_0_and_1_do_not_intersect_nor_the_final_range() {
    for size in [1, 2, 1000, MAX_PILOT_SIZE] {
        // The two blocks are contiguous intervals: comparing their ends
        // is the whole of the disjointness check.
        let first_0 = pilot_stream(0, 0, size).unwrap();
        let last_0 = pilot_stream(0, size - 1, size).unwrap();
        let first_1 = pilot_stream(1, 0, size).unwrap();
        let last_1 = pilot_stream(1, size - 1, size).unwrap();
        assert!(first_0 >= PILOT_STREAM_BASE, "size {size}");
        assert!(last_0 < first_1, "size {size}");
        assert!(first_1 <= last_1, "size {size}");
        // Every final-campaign stream lies below the pilot base.
        assert!(final_stream(MAX_FINAL_REPLICAS - 1).unwrap() < first_0);
    }
    // And replica by replica on a small pilot.
    let size = 64;
    let block_0: Vec<u64> = (0..size)
        .map(|i| pilot_stream(0, i, size).unwrap())
        .collect();
    let block_1: Vec<u64> = (0..size)
        .map(|i| pilot_stream(1, i, size).unwrap())
        .collect();
    assert!(block_0.iter().all(|s| !block_1.contains(s)));
    assert!(block_0
        .iter()
        .chain(&block_1)
        .all(|&s| s >= PILOT_STREAM_BASE));
}

#[test]
fn the_last_admissible_pilot_stream_does_not_overflow() {
    let last = pilot_stream(MAX_PILOT_ITERATIONS - 1, MAX_PILOT_SIZE - 1, MAX_PILOT_SIZE).unwrap();
    assert!(last > pilot_stream(MAX_PILOT_ITERATIONS - 2, 0, MAX_PILOT_SIZE).unwrap());
}

#[test]
fn a_pilot_size_above_the_maximum_is_refused() {
    assert_eq!(
        pilot_stream(0, 0, MAX_PILOT_SIZE + 1),
        Err(StreamError::PilotSizeTooLarge {
            size: MAX_PILOT_SIZE + 1,
            max: MAX_PILOT_SIZE,
        })
    );
}

#[test]
fn a_replica_outside_its_pilot_is_refused() {
    assert_eq!(
        pilot_stream(0, 10, 10),
        Err(StreamError::PilotReplicaOutOfRange {
            replica: 10,
            size: 10
        })
    );
}

#[test]
fn an_iteration_beyond_the_maximum_is_refused() {
    assert_eq!(
        pilot_stream(MAX_PILOT_ITERATIONS, 0, 10),
        Err(StreamError::PilotIterationOutOfRange {
            iteration: MAX_PILOT_ITERATIONS,
            max: MAX_PILOT_ITERATIONS,
        })
    );
}

#[test]
fn refusals_read_as_sentences() {
    let message = pilot_stream(0, 0, MAX_PILOT_SIZE + 1)
        .unwrap_err()
        .to_string();
    assert!(message.contains("pilot size"), "{message}");
    let _: &dyn std::error::Error = &StreamError::PilotSizeTooLarge { size: 1, max: 0 };
}
