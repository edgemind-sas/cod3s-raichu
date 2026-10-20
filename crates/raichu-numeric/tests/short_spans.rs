//! Short durations still require state evolution, error control and callbacks.
#![allow(clippy::unwrap_used, clippy::panic)]

use raichu_numeric::{DormandPrince45, OdeError, OdeSolver, OdeSystem, Outcome, SolverParams};

fn next_time(t: f64) -> f64 {
    f64::from_bits(if t >= 0.0 {
        t.to_bits() + 1
    } else {
        t.to_bits() - 1
    })
}

struct Flow {
    rate: f64,
    exponential: bool,
    event: Option<f64>,
    observation: Option<f64>,
    flips: Vec<(f64, bool)>,
}

impl Flow {
    fn constant(rate: f64) -> Self {
        Self {
            rate,
            exponential: false,
            event: None,
            observation: None,
            flips: vec![],
        }
    }
}

impl OdeSystem for Flow {
    fn dim(&self) -> usize {
        1
    }
    fn rhs(&mut self, _: f64, y: &[f64], out: &mut [f64]) {
        out[0] = self.rate * if self.exponential { y[0] } else { 1.0 };
    }
    fn n_events(&self) -> usize {
        usize::from(self.event.is_some())
    }
    fn events(&mut self, _: f64, y: &[f64], out: &mut [f64]) {
        if let Some(threshold) = self.event {
            out[0] = y[0] - threshold;
        }
    }
    fn n_observations(&self) -> usize {
        usize::from(self.observation.is_some())
    }
    fn observations(&mut self, _: f64, y: &[f64], out: &mut [bool]) {
        if let Some(threshold) = self.observation {
            out[0] = y[0] >= threshold;
        }
    }
    fn observed(&mut self, _: usize, t: f64, now: bool) {
        self.flips.push((t, now));
    }
}

#[test]
fn short_spans_and_remainders_evolve_and_deliver_the_final_sample() {
    let far: f64 = 1e12;
    let ulp = next_time(far) - far;
    for (start, end, max_step) in [
        (0.0, 5e-15, 0.1),
        (far, next_time(far), 0.1),
        (-far, next_time(-far), 0.1),
        (0.0, 0.125 + 5e-15, 0.125),
        (far, far + 0.125 + ulp, 0.125),
        (far, far + 3.0 * ulp, 1.5 * ulp),
        (-far, -far + 3.0 * ulp, 1.5 * ulp),
    ] {
        for rate in [-1e15, 0.0, 1e15] {
            let mut solver = DormandPrince45::new(SolverParams {
                max_step,
                ..SolverParams::default()
            });
            let mut y = [0.0];
            let mut samples = vec![];
            let outcome = solver
                .integrate(
                    &mut Flow::constant(rate),
                    start,
                    &mut y,
                    end,
                    &[end],
                    &mut |t, y| samples.push((t, y[0])),
                )
                .unwrap();
            let expected = rate * (end - start);
            // Round-off allowance stays well below the missing remainder.
            let tolerance = (8.0 * f64::EPSILON * expected.abs()).max(1e-12);
            assert_eq!(outcome, Outcome::Reached { t: end });
            assert!(
                (y[0] - expected).abs() <= tolerance,
                "{start}..{end}: {y:?} != {expected}"
            );
            assert_eq!(samples.len(), 1, "{samples:?}");
            assert_eq!(samples[0].0, end);
            assert!((samples[0].1 - expected).abs() <= tolerance);
        }
    }
}

#[test]
fn short_intervals_keep_events_observations_and_sample_ordering() {
    for prefix in [0.0, 0.125] {
        let rate = 1e15;
        let mut flow = Flow::constant(rate);
        flow.event = Some(rate * prefix + 3.0);
        flow.observation = Some(rate * prefix + 1.0);
        let mut solver = DormandPrince45::new(SolverParams {
            max_step: 0.125,
            tol_event: 1e-18,
            ..SolverParams::default()
        });
        let end = prefix + 5e-15;
        let mut y = [0.0];
        let mut samples = vec![];
        let outcome = solver
            .integrate(
                &mut flow,
                0.0,
                &mut y,
                end,
                &[prefix + 0.5e-15, prefix + 4e-15, end],
                &mut |t, y| samples.push((t, y[0])),
            )
            .unwrap();
        let Outcome::Event { index, t } = outcome else {
            panic!("expected a boundary, got {outcome:?}");
        };
        let time_tolerance = 1e-18_f64.max(8.0 * (next_time(prefix) - prefix));
        assert_eq!(index, 0);
        assert!((t - (prefix + 3e-15)).abs() <= time_tolerance, "{t}");
        assert!((y[0] - (rate * prefix + 3.0)).abs() <= rate * time_tolerance);
        assert_eq!(flow.flips.len(), 1, "{:?}", flow.flips);
        assert!(flow.flips[0].1);
        assert!((flow.flips[0].0 - (prefix + 1e-15)).abs() <= time_tolerance);
        assert_eq!(samples.len(), 1, "{samples:?}");
        assert!(samples[0].0 < t && flow.flips[0].0 <= t);
        assert!((samples[0].1 - rate * samples[0].0).abs() <= rate * time_tolerance);
    }
}

#[test]
fn short_span_still_checks_error_and_non_finite_derivatives() {
    let mut solver = DormandPrince45::new(SolverParams::default());
    let mut flow = Flow::constant(1e15);
    flow.exponential = true;
    let mut y = [1.0];
    solver
        .integrate(&mut flow, 0.0, &mut y, 5e-15, &[], &mut |_, _| {})
        .unwrap();
    assert!((y[0] - 5.0_f64.exp()).abs() < 1e-6, "{y:?}");
    assert!(solver.stats().rejected > 0);

    for rate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let outcome = solver.integrate(
            &mut Flow::constant(rate),
            0.0,
            &mut [0.0],
            5e-15,
            &[],
            &mut |_, _| {},
        );
        assert!(
            matches!(outcome, Err(OdeError::NonFinite { .. })),
            "{outcome:?}"
        );
    }
}

#[test]
fn unresolvable_error_budget_or_step_cap_fails_explicitly() {
    for start in [1e12, next_time(1e12), -1e12, next_time(-1e12)] {
        let end = next_time(start);
        let span = end - start;
        for (exponential, max_step) in [(true, span), (false, span / 2.0), (false, 0.75 * span)] {
            let mut flow = Flow::constant(1.0 / span);
            flow.exponential = exponential;
            let mut solver = DormandPrince45::new(SolverParams {
                max_step,
                ..SolverParams::default()
            });
            let mut y = [1.0];
            let outcome = solver.integrate(&mut flow, start, &mut y, end, &[end], &mut |_, _| {
                panic!("unaccepted sample")
            });
            assert!(
                matches!(outcome, Err(OdeError::StepSizeUnderflow { .. })),
                "{outcome:?}"
            );
            assert_eq!(y, [1.0]);
        }
    }
}

#[test]
fn zero_length_span_keeps_state_and_initial_event_semantics() {
    let mut solver = DormandPrince45::new(SolverParams::default());
    let mut flow = Flow::constant(f64::NAN);
    let mut y = [2.0];
    let outcome = solver.integrate(&mut flow, 1.0, &mut y, 1.0, &[1.0], &mut |_, _| {
        panic!("initial sample")
    });
    assert_eq!(outcome.unwrap(), Outcome::Reached { t: 1.0 });
    assert_eq!(y, [2.0]);
    flow.event = Some(2.0);
    let outcome = solver.integrate(&mut flow, 1.0, &mut y, 1.0, &[], &mut |_, _| {});
    assert_eq!(outcome.unwrap(), Outcome::Event { index: 0, t: 1.0 });
}

#[test]
fn adaptive_work_is_bounded_without_limiting_long_resolved_flows() {
    struct StiffDrain;
    impl OdeSystem for StiffDrain {
        fn dim(&self) -> usize {
            1
        }
        fn rhs(&mut self, _: f64, y: &[f64], out: &mut [f64]) {
            out[0] = 1.0 - 1e16 * y[0] / (y[0] + 100.0);
        }
        fn n_events(&self) -> usize {
            0
        }
        fn events(&mut self, _: f64, _: &[f64], _: &mut [f64]) {}
    }
    let mut solver = DormandPrince45::new(SolverParams::default());
    let result = solver.integrate(&mut StiffDrain, 0.0, &mut [2.0], 1e-4, &[], &mut |_, _| {});
    assert!(matches!(result, Err(OdeError::StepSizeUnderflow { t, .. }) if t > 0.0 && t < 1e-4));
    assert!(solver.stats().accepted > 0);

    let mut solver = DormandPrince45::new(SolverParams {
        max_step: 1.0,
        ..SolverParams::default()
    });
    let mut y = [0.0];
    let result = solver.integrate(
        &mut Flow::constant(1.0),
        0.0,
        &mut y,
        10_001.0,
        &[],
        &mut |_, _| {},
    );
    assert_eq!(result.unwrap(), Outcome::Reached { t: 10_001.0 });
    assert!((y[0] - 10_001.0).abs() <= 8.0 * f64::EPSILON * 10_001.0);
    assert!(solver.stats().accepted > 10_000);
}
