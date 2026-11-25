//! The absorption profile of a phase chain at several instants: `P(S_k <=
//! t)` and its integral `∫_0^t P(S_k <= s) ds = E[(t - S_k)^+]`, each with
//! a guaranteed error bound, validated against closed forms.
//!
//! The sequence-tree explorer reads it to place the mass of a node in
//! time: a node entered after `k` sojourns and left after `k + 1` is
//! occupied at `t` with probability `P(S_k <= t) - P(S_{k+1} <= t)`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_numeric::{PhaseTypeAccumulator, PhaseTypeError, PhaseTypeSettings};

fn accumulator(horizon: f64, rates: &[f64]) -> PhaseTypeAccumulator {
    PhaseTypeAccumulator::from_rates(horizon, rates, PhaseTypeSettings::default()).unwrap()
}

/// Composite Simpson rule of `f` over `[0, t]`.
fn simpson(f: impl Fn(f64) -> f64, t: f64) -> f64 {
    let n = 20_000;
    let h = t / f64::from(n);
    let mut sum = f(0.0) + f(t);
    for i in 1..n {
        sum += f(f64::from(i) * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

fn assert_covered(value: f64, error: f64, expected: f64, tol: f64, what: &str) {
    let diff = (value - expected).abs();
    assert!(
        diff <= tol * expected.abs().max(1e-300),
        "{what}: {value:e} vs {expected:e} (relative {:e})",
        diff / expected.abs()
    );
    assert!(
        error >= 0.0 && error.is_finite(),
        "{what}: error bound {error}"
    );
    assert!(
        error <= 1e-6 * expected.abs().max(1e-12),
        "{what}: loose bound {error:e}"
    );
}

#[test]
fn an_empty_path_is_entered_at_zero() {
    let acc = PhaseTypeAccumulator::new(10.0, PhaseTypeSettings::default()).unwrap();
    let profile = acc.absorption_profile(&[0.0, 2.5, 10.0]).unwrap();
    for (point, t) in profile.iter().zip([0.0, 2.5, 10.0]) {
        assert_eq!(point.cdf, 1.0);
        assert_eq!(point.integral, t);
        assert_eq!(point.cdf_error, 0.0);
        assert_eq!(point.integral_error, 0.0);
    }
}

#[test]
fn one_exponential_phase_matches_its_closed_forms() {
    let lambda = 0.3;
    let acc = accumulator(10.0, &[lambda]);
    let instants = [0.0, 0.5, 3.0, 10.0];
    let profile = acc.absorption_profile(&instants).unwrap();
    for (point, &t) in profile.iter().zip(&instants) {
        let cdf = -(-lambda * t).exp_m1();
        let integral = t - cdf / lambda;
        if t == 0.0 {
            assert_eq!(point.cdf, 0.0);
            assert_eq!(point.integral, 0.0);
            continue;
        }
        assert_covered(
            point.cdf,
            point.cdf_error,
            cdf,
            1e-11,
            &format!("cdf at {t}"),
        );
        assert_covered(
            point.integral,
            point.integral_error,
            integral,
            1e-10,
            &format!("integral at {t}"),
        );
    }
}

#[test]
fn a_hypoexponential_path_matches_its_closed_forms() {
    let (a, b) = (0.7, 0.2);
    let cdf = |t: f64| 1.0 - (b * (-a * t).exp() - a * (-b * t).exp()) / (b - a);
    let acc = accumulator(8.0, &[a, b]);
    let instants = [1.0, 4.0, 8.0];
    let profile = acc.absorption_profile(&instants).unwrap();
    for (point, &t) in profile.iter().zip(&instants) {
        assert_covered(
            point.cdf,
            point.cdf_error,
            cdf(t),
            1e-10,
            &format!("cdf at {t}"),
        );
        assert_covered(
            point.integral,
            point.integral_error,
            simpson(cdf, t),
            1e-9,
            &format!("integral at {t}"),
        );
    }
    // The value at the horizon is the absorption the accumulator keeps.
    assert_eq!(profile[2].cdf.to_bits(), acc.absorption().value.to_bits());
}

/// Erlang(10, q) distribution function by its upper Poisson series,
/// every term positive: exact to rounding for tiny values.
fn erlang10(q: f64, t: f64) -> f64 {
    let x = q * t;
    let mut term = (-x).exp();
    let mut sum = 0.0;
    for j in 1..=80 {
        term *= x / f64::from(j);
        if j >= 10 {
            sum += term;
        }
    }
    sum
}

#[test]
fn a_long_deep_path_keeps_tiny_probabilities_with_a_relative_bound() {
    // Ten phases of rate 0.05 at t = 2: P is about 2.5e-17.
    let q = 0.05;
    let acc = accumulator(2.0, &[q; 10]);
    let profile = acc.absorption_profile(&[1.0, 2.0]).unwrap();
    for (point, t) in profile.iter().zip([1.0f64, 2.0]) {
        let cdf = erlang10(q, t);
        let integral = simpson(|s| erlang10(q, s), t);
        assert!(
            (point.cdf / cdf - 1.0).abs() < 1e-10,
            "cdf at {t}: {:e} vs {cdf:e}",
            point.cdf
        );
        assert!(point.cdf_error <= 1e-6 * point.cdf, "cdf bound at {t}");
        assert!(
            (point.integral / integral - 1.0).abs() < 1e-8,
            "integral at {t}: {:e} vs {integral:e}",
            point.integral
        );
        assert!(
            point.integral_error <= 1e-6 * point.integral,
            "integral bound at {t}"
        );
    }
}

#[test]
fn a_blocked_path_is_never_entered() {
    let acc = accumulator(5.0, &[0.4, 0.0]);
    let profile = acc.absorption_profile(&[1.0, 5.0]).unwrap();
    assert!(profile.iter().all(|p| p.cdf == 0.0 && p.integral == 0.0));
}

#[test]
fn instants_outside_the_horizon_are_refused() {
    let acc = accumulator(5.0, &[0.4]);
    for instants in [vec![6.0], vec![-1.0], vec![f64::NAN]] {
        assert!(matches!(
            acc.absorption_profile(&instants),
            Err(PhaseTypeError::InvalidInstant { .. })
        ));
    }
}

#[test]
fn a_path_past_the_term_cap_is_refused_by_name() {
    let settings = PhaseTypeSettings {
        max_terms: 50,
        ..PhaseTypeSettings::default()
    };
    let acc = PhaseTypeAccumulator::from_rates(100.0, &[5.0, 3.0], settings).unwrap();
    assert!(matches!(
        acc.absorption_profile(&[100.0]),
        Err(PhaseTypeError::ProfileTermCap { max_terms: 50 })
    ));
}
