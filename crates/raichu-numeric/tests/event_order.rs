//! Chronological selection of competing boundaries on both solver backends.
#![allow(clippy::unwrap_used, clippy::panic)]

use raichu_numeric::{DormandPrince45, FixedEuler, OdeSolver, OdeSystem, Outcome, SolverParams};

const TOL: f64 = 1e-10;

struct Ramp {
    thresholds: Vec<f64>,
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
        self.thresholds.len()
    }
    fn events(&mut self, _t: f64, y: &[f64], out: &mut [f64]) {
        for (margin, threshold) in out.iter_mut().zip(&self.thresholds) {
            *margin = y[0] - threshold;
        }
    }
    fn n_observations(&self) -> usize {
        2
    }
    fn observations(&mut self, _t: f64, y: &[f64], out: &mut [bool]) {
        out[0] = y[0] >= 0.001;
        out[1] = y[0] >= 0.003;
    }
    fn observed(&mut self, index: usize, t: f64, now: bool) {
        self.flips.push((index, t, now));
    }
}

fn solvers() -> [Box<dyn OdeSolver>; 2] {
    [
        Box::new(DormandPrince45::new(SolverParams::default())),
        Box::new(FixedEuler::new(0.1, TOL)),
    ]
}

#[test]
fn competing_margins_choose_time_before_index() {
    // All crossings lie inside the first scan interval (0, 0.00625].
    for thresholds in [
        vec![0.004, 0.002],
        vec![0.002, 0.004],
        vec![0.002 + 4.0 * TOL, 0.002],
        vec![0.005, 0.004, 0.002],
        vec![0.005, 0.002, 0.004],
        vec![0.002, 0.005, 0.004],
    ] {
        for mut solver in solvers() {
            let expected_index = thresholds.iter().position(|&x| x == 0.002).unwrap();
            let mut system = Ramp {
                thresholds: thresholds.clone(),
                flips: vec![],
            };
            let mut y = [0.0];
            let outcome = solver
                .integrate(&mut system, 0.0, &mut y, 0.1, &[], &mut |_, _| {})
                .unwrap();
            let Outcome::Event { index, t } = outcome else {
                panic!("expected a boundary, got {outcome:?}");
            };
            assert_eq!(index, expected_index, "thresholds={thresholds:?}");
            assert!((t - 0.002).abs() <= TOL, "t={t}");
            assert!((y[0] - 0.002).abs() <= TOL, "y={y:?}");
        }
    }
}

#[test]
fn simultaneous_and_initial_crossings_keep_the_first_index() {
    for (thresholds, expected_index, expected_time) in [
        (vec![0.004, 0.002, 0.002], 1, 0.002),
        (vec![0.004, 0.0, -0.001], 1, 0.0),
    ] {
        for mut solver in solvers() {
            let mut system = Ramp {
                thresholds: thresholds.clone(),
                flips: vec![],
            };
            let mut y = [0.0];
            let outcome = solver
                .integrate(&mut system, 0.0, &mut y, 0.1, &[], &mut |_, _| {})
                .unwrap();
            let Outcome::Event { index, t } = outcome else {
                panic!("expected a boundary, got {outcome:?}");
            };
            assert_eq!(index, expected_index);
            assert!((t - expected_time).abs() <= TOL, "t={t}");
            assert!((y[0] - expected_time).abs() <= TOL);
        }
    }
}

#[test]
fn samples_and_observations_stop_at_the_earliest_crossing() {
    for mut solver in solvers() {
        let mut system = Ramp {
            thresholds: vec![0.004, 0.002],
            flips: vec![],
        };
        let mut y = [0.0];
        let mut samples = Vec::new();
        let outcome = solver
            .integrate(
                &mut system,
                0.0,
                &mut y,
                0.1,
                &[0.0005, 0.0015, 0.003, 0.005],
                &mut |t, y| samples.push((t, y[0])),
            )
            .unwrap();
        let Outcome::Event { t, .. } = outcome else {
            panic!("expected a boundary, got {outcome:?}");
        };
        assert_eq!(samples.len(), 2, "{samples:?}");
        for (time, value) in samples {
            assert!(time < t);
            assert!((value - time).abs() <= TOL);
        }
        assert_eq!(system.flips.len(), 1, "{:?}", system.flips);
        let (index, time, now) = system.flips[0];
        assert_eq!(index, 0);
        assert!(now && time <= t);
        assert!((time - 0.001).abs() <= TOL);
        assert!((y[0] - t).abs() <= TOL);
    }
}

#[test]
fn competing_crossings_at_large_times_terminate_at_float_resolution() {
    let t0 = 6e5;
    let resolution = t0 * f64::EPSILON;
    assert!(resolution > TOL);
    for mut solver in solvers() {
        let mut system = Ramp {
            thresholds: vec![0.004, 0.002],
            flips: vec![],
        };
        let mut y = [0.0];
        let outcome = solver
            .integrate(&mut system, t0, &mut y, t0 + 0.1, &[], &mut |_, _| {})
            .unwrap();
        let Outcome::Event { index, t } = outcome else {
            panic!("expected a boundary, got {outcome:?}");
        };
        assert_eq!(index, 1);
        assert!((t - (t0 + 0.002)).abs() <= resolution, "t={t}");
        assert!((y[0] - 0.002).abs() <= 2.0 * resolution, "y={y:?}");
    }
}
