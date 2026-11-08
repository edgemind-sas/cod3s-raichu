//! A right-hand side that jumps inside a step (an `if`, `min` or `max`)
//! makes the local error shrink like `h`, not `h^5`: the controller needs
//! more retries than on a smooth flow, and must be allowed them. It gives
//! up only when the shortest representable step is itself rejected.
#![allow(clippy::unwrap_used, clippy::panic)]

use raichu_numeric::{DormandPrince45, OdeError, OdeSolver, OdeSystem, Outcome, SolverParams};

/// `y' = 0` up to `at`, `y' = jump` after it.
struct Jump {
    at: f64,
    jump: f64,
}

impl OdeSystem for Jump {
    fn dim(&self) -> usize {
        1
    }
    fn rhs(&mut self, t: f64, _: &[f64], out: &mut [f64]) {
        out[0] = if t > self.at { self.jump } else { 0.0 };
    }
    fn n_events(&self) -> usize {
        0
    }
    fn events(&mut self, _: f64, _: &[f64], _: &mut [f64]) {}
}

fn integrate_jump(at: f64, jump: f64, t_end: f64, params: SolverParams) -> (Outcome, f64) {
    let mut solver = DormandPrince45::new(params);
    let mut y = [1.0];
    let outcome = solver
        .integrate(
            &mut Jump { at, jump },
            at,
            &mut y,
            t_end,
            &[],
            &mut |_, _| {},
        )
        .unwrap_or_else(|error| panic!("jump {jump:e} at {at}: {error}"));
    (outcome, y[0])
}

#[test]
fn a_jump_at_the_start_of_the_span_reaches_the_horizon() {
    // The reproducer of the issue: every trial from t = 0 straddles the
    // jump, and a step of a few 1e-11 is the first one accepted.
    for jump in [1e2, 1e4, 1e8] {
        let (outcome, y) = integrate_jump(0.0, jump, 1.0, SolverParams::default());
        assert_eq!(outcome, Outcome::Reached { t: 1.0 });
        let expected = 1.0 + jump;
        assert!(
            ((y - expected) / expected).abs() < 1e-6,
            "jump {jump:e}: y(1) = {y}, expected {expected}"
        );
    }
}

#[test]
fn a_jump_away_from_zero_reaches_the_horizon_too() {
    // Away from t = 0 the time axis is coarser: the accepted step must
    // still be representable above the jump.
    let (outcome, y) = integrate_jump(0.5, 1e4, 1.5, SolverParams::default());
    assert_eq!(outcome, Outcome::Reached { t: 1.5 });
    let expected = 1.0 + 1e4;
    assert!(((y - expected) / expected).abs() < 1e-6, "y = {y}");
}

#[test]
fn a_jump_without_a_step_cap_reaches_the_horizon() {
    // An infinite `max_step` sets no work window: the representable-step
    // criterion alone bounds the retries.
    let params = SolverParams {
        max_step: f64::INFINITY,
        ..SolverParams::default()
    };
    let (outcome, y) = integrate_jump(0.0, 1e4, 1.0, params);
    assert_eq!(outcome, Outcome::Reached { t: 1.0 });
    assert!(((y - 1.0 - 1e4) / 1e4).abs() < 1e-6, "y = {y}");
}

#[test]
fn a_rejected_shortest_representable_step_is_an_underflow() {
    // At t = 0.5 one time increment is 1.1e-16; a jump of 1e30 changes
    // the state by 1e14 over it, far past the tolerance, so no step can
    // be accepted. The search ends there, in a bounded number of trials,
    // with or without a work window.
    for max_step in [0.1, f64::INFINITY] {
        let mut solver = DormandPrince45::new(SolverParams {
            max_step,
            ..SolverParams::default()
        });
        let mut y = [1.0];
        let outcome = solver.integrate(
            &mut Jump {
                at: 0.5,
                jump: 1e30,
            },
            0.5,
            &mut y,
            1.0,
            &[],
            &mut |_, _| {},
        );
        assert!(
            matches!(outcome, Err(OdeError::StepSizeUnderflow { t, h }) if t == 0.5 && h > 0.0),
            "max_step {max_step}: {outcome:?}"
        );
        assert_eq!(y, [1.0], "the state stays at the last accepted point");
        let stats = solver.stats();
        assert_eq!(stats.accepted, 0);
        assert!(
            stats.rejected > 20 && stats.rejected < 2_000,
            "max_step {max_step}: {} rejected trials",
            stats.rejected
        );
    }
}

#[test]
fn a_solution_blowing_up_ends_on_a_typed_error_without_a_step_cap() {
    // `y' = y^2`, `y(0) = 1` has a pole at `t = 1`. Without a work window
    // the representable-step criterion alone must end the search there.
    struct Blowup;
    impl OdeSystem for Blowup {
        fn dim(&self) -> usize {
            1
        }
        fn rhs(&mut self, _: f64, y: &[f64], out: &mut [f64]) {
            out[0] = y[0] * y[0];
        }
        fn n_events(&self) -> usize {
            0
        }
        fn events(&mut self, _: f64, _: &[f64], _: &mut [f64]) {}
    }
    let mut solver = DormandPrince45::new(SolverParams {
        max_step: f64::INFINITY,
        ..SolverParams::default()
    });
    let mut y = [1.0];
    let outcome = solver.integrate(&mut Blowup, 0.0, &mut y, 2.0, &[], &mut |_, _| {});
    match outcome {
        Err(OdeError::StepSizeUnderflow { t, .. } | OdeError::NonFinite { t }) => {
            assert!(t > 0.99 && t <= 1.0, "stopped at the pole, got t = {t}");
            assert!(
                solver.stats().rejected < 2_000,
                "{} rejected trials",
                solver.stats().rejected
            );
        }
        other => panic!("expected a typed error at the pole, got {other:?}"),
    }
}
