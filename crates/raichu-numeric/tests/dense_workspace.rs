//! Allocation profile of one Dormand-Prince integration.
//!
//! The dense interpolant and the scan scratch are workspace sized once per
//! `integrate` call: the number of heap allocations must not grow with the
//! number of accepted steps, whatever reads the interpolant (events,
//! observations, samples, or nothing at all). Counted with a per-thread
//! allocator counter, so tests running in parallel do not pollute each other.
#![allow(clippy::unwrap_used, clippy::panic)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use raichu_numeric::{DormandPrince45, OdeSolver, OdeSystem, Outcome, SolverParams};

struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

// SAFETY: every call is forwarded unchanged to the system allocator; the
// counter is a const-initialised thread-local `Cell`, which never allocates.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations() -> u64 {
    ALLOCATIONS.with(Cell::get)
}

/// `y' = 1`, `y(0) = 0`, with optional consumers of the interpolant.
///
/// The event threshold is never reached on the tested spans, and the
/// observation flips exactly once, at `t = 1`, on every span: both
/// consumers are scanned on every step without changing what happens.
struct Ramp {
    events: bool,
    observations: bool,
    flips: Vec<(usize, f64, bool)>,
}

impl OdeSystem for Ramp {
    fn dim(&self) -> usize {
        1
    }
    fn rhs(&mut self, _t: f64, _y: &[f64], out: &mut [f64]) {
        out[0] = 1.0;
    }
    fn n_events(&self) -> usize {
        usize::from(self.events)
    }
    fn events(&mut self, _t: f64, y: &[f64], out: &mut [f64]) {
        out[0] = y[0] - 1.0e6;
    }
    fn n_observations(&self) -> usize {
        usize::from(self.observations)
    }
    fn observations(&mut self, _t: f64, y: &[f64], out: &mut [bool]) {
        out[0] = y[0] >= 1.0;
    }
    fn observed(&mut self, index: usize, t: f64, now: bool) {
        self.flips.push((index, t, now));
    }
}

#[derive(Clone, Copy, Debug)]
struct Consumers {
    events: bool,
    observations: bool,
    samples: bool,
}

/// Allocations made by `integrate` alone over `[0, t_end]`, with the step
/// pinned to `max_step = 0.125` (the constant field never moves the
/// controller off the cap), so `t_end = 128` takes 1 024 accepted steps.
fn integrate_counting(consumers: Consumers, t_end: f64) -> (u64, u64, Vec<f64>) {
    let mut system = Ramp {
        events: consumers.events,
        observations: consumers.observations,
        flips: Vec::with_capacity(4),
    };
    // One sample per unit of time, the last one on the endpoint.
    let samples: Vec<f64> = if consumers.samples {
        (1..=t_end as usize).map(|s| s as f64).collect()
    } else {
        Vec::new()
    };
    let mut seen: Vec<f64> = Vec::with_capacity(samples.len());
    let mut solver = DormandPrince45::new(SolverParams {
        max_step: 0.125,
        ..SolverParams::default()
    });
    let mut y = vec![0.0];
    let before = allocations();
    let outcome = solver
        .integrate(&mut system, 0.0, &mut y, t_end, &samples, &mut |_, ys| {
            seen.push(ys[0]);
        })
        .unwrap();
    let spent = allocations() - before;
    assert_eq!(outcome, Outcome::Reached { t: t_end });
    if consumers.observations {
        assert_eq!(system.flips.len(), 1, "{consumers:?}");
    }
    (spent, solver.stats().accepted, seen)
}

fn every_combination() -> Vec<Consumers> {
    let mut all = Vec::new();
    for bits in 0..8u8 {
        all.push(Consumers {
            events: bits & 1 != 0,
            observations: bits & 2 != 0,
            samples: bits & 4 != 0,
        });
    }
    all
}

#[test]
fn allocations_do_not_grow_with_the_number_of_accepted_steps() {
    for consumers in every_combination() {
        let (short, short_steps, _) = integrate_counting(consumers, 64.0);
        let (long, long_steps, _) = integrate_counting(consumers, 128.0);
        assert_eq!((short_steps, long_steps), (512, 1024), "{consumers:?}");
        assert_eq!(
            short, long,
            "{consumers:?}: {short} allocations over {short_steps} steps, \
             {long} over {long_steps}"
        );
    }
}

#[test]
fn every_sample_is_delivered_including_the_endpoint() {
    for consumers in every_combination().into_iter().filter(|c| c.samples) {
        let (_, _, seen) = integrate_counting(consumers, 128.0);
        let expected: Vec<f64> = (1..=128).map(f64::from).collect();
        assert_eq!(seen.len(), expected.len(), "{consumers:?}");
        for (got, want) in seen.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-9, "{consumers:?}: {got} vs {want}");
        }
    }
}

/// `y' = 1`, `y(0) = 0`, with one boundary at `y = threshold`, observed too.
struct Truncated {
    threshold: f64,
    flips: Vec<(usize, f64, bool)>,
}

impl OdeSystem for Truncated {
    fn dim(&self) -> usize {
        1
    }
    fn rhs(&mut self, _t: f64, _y: &[f64], out: &mut [f64]) {
        out[0] = 1.0;
    }
    fn n_events(&self) -> usize {
        1
    }
    fn events(&mut self, _t: f64, y: &[f64], out: &mut [f64]) {
        out[0] = y[0] - self.threshold;
    }
    fn n_observations(&self) -> usize {
        1
    }
    fn observations(&mut self, _t: f64, y: &[f64], out: &mut [bool]) {
        out[0] = y[0] >= 1.0;
    }
    fn observed(&mut self, index: usize, t: f64, now: bool) {
        self.flips.push((index, t, now));
    }
}

/// Integrates until the boundary at `threshold` (strictly inside a step,
/// away from the step grid) ends the segment, with a sample every unit of
/// time up to 128. Returns the allocations, the outcome and the samples.
fn truncate_counting(threshold: f64) -> (u64, Outcome, Vec<(f64, f64)>) {
    let mut system = Truncated {
        threshold,
        flips: Vec::with_capacity(4),
    };
    let samples: Vec<f64> = (1..=128).map(f64::from).collect();
    let mut seen: Vec<(f64, f64)> = Vec::with_capacity(samples.len());
    let mut solver = DormandPrince45::new(SolverParams {
        max_step: 0.125,
        ..SolverParams::default()
    });
    let mut y = vec![0.0];
    let before = allocations();
    let outcome = solver
        .integrate(&mut system, 0.0, &mut y, 128.0, &samples, &mut |t, ys| {
            seen.push((t, ys[0]));
        })
        .unwrap();
    let spent = allocations() - before;
    assert_eq!(system.flips.len(), 1);
    (spent, outcome, seen)
}

#[test]
fn an_event_truncating_the_integration_keeps_the_workspace_fixed() {
    let (early, early_outcome, early_seen) = truncate_counting(40.06);
    let (late, late_outcome, late_seen) = truncate_counting(100.06);
    assert_eq!(
        early, late,
        "allocations grew with the steps before the event"
    );
    for (outcome, seen, threshold) in [
        (early_outcome, early_seen, 40.06),
        (late_outcome, late_seen, 100.06),
    ] {
        let Outcome::Event { index, t } = outcome else {
            panic!("expected the boundary at {threshold} to end the segment");
        };
        assert_eq!(index, 0);
        assert!((t - threshold).abs() < 1e-9, "{t} vs {threshold}");
        // Every sample strictly before the event, and none after it.
        let expected = threshold.floor() as usize;
        assert_eq!(seen.len(), expected, "threshold {threshold}");
        for (ts, value) in seen {
            assert!(ts < threshold && (value - ts).abs() < 1e-9, "{ts}: {value}");
        }
    }
}
