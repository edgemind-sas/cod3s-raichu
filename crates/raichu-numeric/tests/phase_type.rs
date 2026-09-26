//! Phase-type absorption probabilities, validated against **closed forms**.
//!
//! The quantity under test is `P(S_k <= t)`, where `S_k` is the sum of
//! `k` independent exponential sojourns of rates `q1..qk`: the probability
//! that an acyclic chain of phases is absorbed by the horizon `t`. The
//! accumulator computes it by uniformization (nonnegative terms, explicit
//! Poisson-tail truncation bound) and falls back to scaling and squaring
//! when the term count exceeds its cap.
//!
//! Witnesses, all kept in this file only:
//!
//! - the elementary closed forms (exponential, two-rate hypoexponential,
//!   Erlang), evaluated with `exp_m1` where it matters;
//! - the Harrison (1990) partial-fraction closed form for sums of
//!   exponentials with repeated rates, valid where its alternating sum
//!   does not cancel (moderate `q t`);
//! - the power series `P = prod(q) * sum_m (-1)^m h_m(q) t^(k+m) / (k+m)!`
//!   (`h_m` the complete homogeneous symmetric polynomial), whose terms
//!   decrease geometrically when `max(q) t` is small: the witness for
//!   probabilities far below 1e-12.
//!
//! Every comparison states its relative tolerance, and every test checks
//! that the returned error bound covers the observed error.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_numeric::{AbsorptionMethod, PhaseTypeAccumulator, PhaseTypeError, PhaseTypeSettings};

// --- witnesses --------------------------------------------------------------

/// Harrison (1990) closed form of `P(S <= t)` for a sum of exponentials
/// with possibly repeated rates, by partial fractions of
/// `G(s) = prod_i (l_i / (s + l_i))^{r_i} / s`.
///
/// `F(t) = 1 + sum_i sum_{l=1}^{r_i} A_{i,l} t^{l-1} e^{-l_i t} / (l-1)!`
/// where `A_{i,l}` is the coefficient of `e^{r_i - l}` in the Taylor
/// expansion of `(s + l_i)^{r_i} G(s)` at `s = -l_i`.
fn harrison_cdf(rates: &[f64], t: f64) -> f64 {
    // Group rates by exact value, keeping multiplicities.
    let mut distinct: Vec<(f64, usize)> = Vec::new();
    for &q in rates {
        match distinct.iter_mut().find(|(l, _)| *l == q) {
            Some(entry) => entry.1 += 1,
            None => distinct.push((q, 1)),
        }
    }
    let constant: f64 = rates.iter().product();
    let mut total = 1.0;
    for (i, &(li, ri)) in distinct.iter().enumerate() {
        let s0 = -li;
        // Taylor series in e = s - s0, truncated at order ri - 1.
        let order = ri;
        // 1 / s = 1/(s0 + e) = (1/s0) sum (-e/s0)^n
        let mut series: Vec<f64> = (0..order)
            .map(|n| (1.0 / s0) * (-1.0 / s0).powi(n as i32))
            .collect();
        for (j, &(lj, rj)) in distinct.iter().enumerate() {
            if j == i {
                continue;
            }
            // (s + lj)^{-rj} = (lj - li + e)^{-rj} = c^{-rj} (1 + e/c)^{-rj}
            let c = lj - li;
            let mut factor = vec![0.0; order];
            let mut binom = 1.0; // generalized binomial (-rj choose n)
            for (n, slot) in factor.iter_mut().enumerate() {
                if n > 0 {
                    binom *= (-(rj as f64) - (n as f64 - 1.0)) / n as f64;
                }
                *slot = c.powi(-(rj as i32)) * binom * c.powi(-(n as i32));
            }
            let mut product = vec![0.0; order];
            for a in 0..order {
                for b in 0..order - a {
                    product[a + b] += series[a] * factor[b];
                }
            }
            series = product;
        }
        for l in 1..=ri {
            let coefficient = constant * series[ri - l];
            let mut fact = 1.0;
            for m in 1..l {
                fact *= m as f64;
            }
            total += coefficient * t.powi(l as i32 - 1) * (-li * t).exp() / fact;
        }
    }
    total
}

/// Power series `P = prod(q) sum_m (-1)^m h_m(q) t^{k+m} / (k+m)!`,
/// summed until the terms no longer move the sum.
fn series_cdf(rates: &[f64], t: f64) -> f64 {
    let k = rates.len();
    let terms = 60;
    // h[m] = complete homogeneous symmetric polynomial of degree m.
    let mut h = vec![0.0; terms];
    h[0] = 1.0;
    for &q in rates {
        for m in 1..terms {
            h[m] += q * h[m - 1];
        }
    }
    let prod: f64 = rates.iter().product();
    let mut sum = 0.0;
    for (m, hm) in h.iter().enumerate() {
        let n = k + m;
        let mut term = hm * prod;
        for i in 1..=n {
            term *= t / i as f64;
        }
        if m % 2 == 1 {
            term = -term;
        }
        sum += term;
    }
    sum
}

fn hypo2_cdf(q1: f64, q2: f64, t: f64) -> f64 {
    1.0 - (q2 * (-q1 * t).exp() - q1 * (-q2 * t).exp()) / (q2 - q1)
}

fn erlang_cdf(k: usize, q: f64, t: f64) -> f64 {
    let x = q * t;
    let mut sum = 0.0;
    let mut term = 1.0;
    for n in 0..k {
        if n > 0 {
            term *= x / n as f64;
        }
        sum += term;
    }
    1.0 - (-x).exp() * sum
}

fn accumulate(rates: &[f64], horizon: f64, settings: PhaseTypeSettings) -> PhaseTypeAccumulator {
    let mut acc = PhaseTypeAccumulator::new(horizon, settings).unwrap();
    for &q in rates {
        acc.push(q).unwrap();
    }
    acc
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// The value agrees with the witness to `tol`, and the returned bound
/// covers the observed error (up to the witness's own rounding, `slack`
/// relative).
fn assert_witness(acc: &PhaseTypeAccumulator, witness: f64, tol: f64, slack: f64, what: &str) {
    let a = acc.absorption();
    let err = (a.value - witness).abs();
    assert!(
        err <= tol * witness,
        "{what}: value {:e} vs witness {:e}, relative error {:e} > {tol:e}",
        a.value,
        witness,
        rel(a.value, witness)
    );
    assert!(
        err <= a.error_bound + slack * witness,
        "{what}: observed error {err:e} exceeds the returned bound {:e}",
        a.error_bound
    );
}

// --- closed forms -------------------------------------------------------------

#[test]
fn one_rate_reproduces_the_exponential_cdf() {
    for &(q, t) in &[(0.7, 2.0), (1e-3, 0.5), (3.0, 10.0), (1.0, 1.0)] {
        let acc = accumulate(&[q], t, PhaseTypeSettings::default());
        let witness = -(-q * t).exp_m1();
        assert_witness(&acc, witness, 1e-14, 1e-16, "exponential");
        assert!(!acc.absorption().flagged);
    }
}

#[test]
fn two_distinct_rates_reproduce_the_hypoexponential_cdf() {
    for &(q1, q2, t) in &[(1.0, 3.0, 0.8), (0.2, 5.0, 3.0), (4.0, 0.5, 1.7)] {
        let acc = accumulate(&[q1, q2], t, PhaseTypeSettings::default());
        assert_witness(&acc, hypo2_cdf(q1, q2, t), 1e-13, 1e-15, "hypoexponential");
    }
}

#[test]
fn three_equal_rates_reproduce_the_erlang_cdf() {
    for &(q, t) in &[(2.0, 1.5), (0.3, 4.0), (5.0, 0.1)] {
        let acc = accumulate(&[q, q, q], t, PhaseTypeSettings::default());
        assert_witness(&acc, erlang_cdf(3, q, t), 1e-13, 1e-15, "Erlang(3)");
    }
}

#[test]
fn repeated_and_distinct_rates_reproduce_the_harrison_witness() {
    let rates = [1.0, 2.0, 1.0, 3.0, 2.0, 1.0];
    for &t in &[0.7, 1.3, 4.0] {
        let acc = accumulate(&rates, t, PhaseTypeSettings::default());
        // The Harrison witness is an alternating sum: its own rounding
        // is a few 1e-14 here, so compare at 1e-11.
        let witness = harrison_cdf(&rates, t);
        assert_witness(&acc, witness, 1e-11, 1e-12, "Harrison mix");
        // The series witness agrees with Harrison at small t.
        if t < 1.0 {
            assert!(rel(series_cdf(&rates, t), witness) < 1e-11);
        }
    }
}

#[test]
fn harrison_witness_agrees_with_the_elementary_forms() {
    // Guards the witness itself before it is trusted.
    assert!(rel(harrison_cdf(&[1.0, 3.0], 0.8), hypo2_cdf(1.0, 3.0, 0.8)) < 1e-14);
    assert!(rel(harrison_cdf(&[2.0, 2.0, 2.0], 1.5), erlang_cdf(3, 2.0, 1.5)) < 1e-13);
}

#[test]
fn the_error_bound_covers_the_observed_error_over_a_grid() {
    let pool = [0.3, 1.1, 2.9, 7.0];
    for &t in &[0.01, 0.5, 2.0, 10.0] {
        for a in 0..pool.len() {
            for b in 0..pool.len() {
                for c in 0..pool.len() {
                    let rates = [pool[a], pool[b], pool[c]];
                    let acc = accumulate(&rates, t, PhaseTypeSettings::default());
                    let witness = if t < 1.0 {
                        series_cdf(&rates, t)
                    } else {
                        harrison_cdf(&rates, t)
                    };
                    assert_witness(&acc, witness, 1e-10, 1e-12, "grid");
                }
            }
        }
    }
}

// --- tiny probabilities -------------------------------------------------------

#[test]
fn a_probability_of_order_1e_18_keeps_its_relative_precision() {
    let rates = [1e-6, 2e-6, 3e-6];
    let t = 1.0;
    let acc = accumulate(&rates, t, PhaseTypeSettings::default());
    let witness = series_cdf(&rates, t);
    assert!(witness > 5e-19 && witness < 2e-18, "witness {witness:e}");
    assert_witness(&acc, witness, 1e-10, 1e-15, "tiny");
    assert!(!acc.absorption().flagged);

    // One minus a survival loses everything here: the complement of the
    // closed form returns noise at the 1e-16 level, not 1e-18.
    let naive = harrison_cdf(&rates, t);
    assert!(
        rel(naive, witness) > 1e-3,
        "naive {naive:e} unexpectedly accurate"
    );
}

#[test]
fn a_probability_far_below_the_double_epsilon_is_still_positive() {
    // Five phases at 1e-5 over t = 1: about 1e-25 / 120.
    let rates = [1e-5; 5];
    let acc = accumulate(&rates, 1.0, PhaseTypeSettings::default());
    let witness = series_cdf(&rates, 1.0);
    assert_witness(&acc, witness, 1e-10, 1e-15, "far below epsilon");
}

// --- large and stiff q t --------------------------------------------------------

#[test]
fn a_large_q_t_below_the_cap_does_not_underflow_the_poisson_weights() {
    // e^{-5e4} underflows to 0: a naive Poisson weight would lose all mass.
    let settings = PhaseTypeSettings::default();
    let acc = accumulate(&[5e4, 1.0], 1.0, settings.clone());
    let a = acc.absorption();
    assert!(matches!(a.method, AbsorptionMethod::Uniformization { .. }));
    assert_witness(&acc, hypo2_cdf(5e4, 1.0, 1.0), 1e-9, 1e-15, "large q t");

    // A long Erlang: mass concentrated far from the origin.
    let acc = accumulate(&[100.0; 40], 0.5, settings);
    let witness = erlang_cdf(40, 100.0, 0.5);
    assert_witness(&acc, witness, 1e-9, 1e-14, "Erlang(40)");
}

#[test]
fn a_stiff_path_goes_through_the_fallback_within_precision() {
    let settings = PhaseTypeSettings {
        rel_precision: 1e-6,
        ..PhaseTypeSettings::default()
    };
    let acc = accumulate(&[1e7], 1.0, settings.clone());
    let a = acc.absorption();
    assert!(
        matches!(a.method, AbsorptionMethod::ScalingAndSquaring { .. }),
        "{:?}",
        a.method
    );
    assert!(!a.flagged, "bound {:e}", a.error_bound);
    assert_witness(&acc, 1.0, 1e-6, 1e-16, "stiff single phase");

    let acc = accumulate(&[1e7, 1.0], 1.0, settings);
    let a = acc.absorption();
    assert!(matches!(
        a.method,
        AbsorptionMethod::ScalingAndSquaring { .. }
    ));
    assert!(!a.flagged);
    assert_witness(
        &acc,
        hypo2_cdf(1e7, 1.0, 1.0),
        1e-6,
        1e-15,
        "stiff two phases",
    );
}

#[test]
fn a_stiff_path_flags_itself_when_the_precision_is_out_of_reach() {
    let settings = PhaseTypeSettings {
        rel_precision: 1e-14,
        ..PhaseTypeSettings::default()
    };
    let acc = accumulate(&[1e7, 1.0], 1.0, settings);
    let a = acc.absorption();
    assert!(a.flagged, "bound {:e} should exceed 1e-14", a.error_bound);
    // Flagged, yet the bound still covers the error.
    let witness = hypo2_cdf(1e7, 1.0, 1.0);
    assert!((a.value - witness).abs() <= a.error_bound + 1e-15);
}

#[test]
fn a_lowered_term_cap_moves_a_moderate_path_to_the_fallback() {
    let settings = PhaseTypeSettings {
        max_terms: 50,
        rel_precision: 1e-8,
    };
    let rates = [3.0, 40.0, 7.0, 3.0];
    let acc = accumulate(&rates, 2.0, settings);
    assert!(matches!(
        acc.absorption().method,
        AbsorptionMethod::ScalingAndSquaring { .. }
    ));
    assert_witness(&acc, harrison_cdf(&rates, 2.0), 1e-9, 1e-12, "capped");
}

// --- incremental use ------------------------------------------------------------

#[test]
fn appending_a_rate_equals_recomputing_from_scratch() {
    // Rates that raise the uniformization rate midway, and rates that do not.
    let rates = [0.5, 2.0, 2.0, 9.0, 0.1, 9.0, 3.0];
    let settings = PhaseTypeSettings::default();
    let mut acc = PhaseTypeAccumulator::new(1.5, settings.clone()).unwrap();
    for (i, &q) in rates.iter().enumerate() {
        acc.push(q).unwrap();
        let scratch =
            PhaseTypeAccumulator::from_rates(1.5, &rates[..=i], settings.clone()).unwrap();
        let (a, b) = (acc.absorption(), scratch.absorption());
        assert!(
            (a.value - b.value).abs() <= 1e-14 * b.value,
            "prefix {i}: {:e} vs {:e}",
            a.value,
            b.value
        );
        assert_eq!(acc.rates(), scratch.rates());
    }
}

#[test]
fn a_cloned_accumulator_is_independent_of_its_parent() {
    let mut parent = accumulate(&[1.0, 2.0], 1.0, PhaseTypeSettings::default());
    let before = parent.absorption();
    let mut child = parent.clone();
    child.push(3.0).unwrap();
    assert_eq!(parent.absorption(), before);
    parent.push(5.0).unwrap();
    assert!(
        rel(
            child.absorption().value,
            harrison_cdf(&[1.0, 2.0, 3.0], 1.0)
        ) < 1e-12
    );
    assert!(
        rel(
            parent.absorption().value,
            harrison_cdf(&[1.0, 2.0, 5.0], 1.0)
        ) < 1e-12
    );
}

// --- edges ------------------------------------------------------------------------

#[test]
fn instantaneous_phases_are_no_ops() {
    let mut acc = accumulate(&[1.0, 2.0], 1.0, PhaseTypeSettings::default());
    let before = acc.absorption();
    acc.push_instantaneous();
    acc.push_instantaneous();
    assert_eq!(acc.absorption(), before);
    assert_eq!(acc.phase_count(), 2);
}

#[test]
fn an_empty_path_is_completed_at_once() {
    let acc = PhaseTypeAccumulator::new(3.0, PhaseTypeSettings::default()).unwrap();
    let a = acc.absorption();
    assert_eq!(a.value, 1.0);
    assert_eq!(a.error_bound, 0.0);
    assert!(!a.flagged);
    let mut only_instant = acc.clone();
    only_instant.push_instantaneous();
    assert_eq!(only_instant.absorption().value, 1.0);
}

#[test]
fn a_zero_horizon_completes_no_timed_phase() {
    let acc = accumulate(&[1.0], 0.0, PhaseTypeSettings::default());
    let a = acc.absorption();
    assert_eq!(a.value, 0.0);
    assert_eq!(a.error_bound, 0.0);
    assert!(!a.flagged);
    let empty = PhaseTypeAccumulator::new(0.0, PhaseTypeSettings::default()).unwrap();
    assert_eq!(empty.absorption().value, 1.0);
}

#[test]
fn a_zero_rate_phase_is_never_left() {
    let mut acc = accumulate(&[1.0, 0.0], 5.0, PhaseTypeSettings::default());
    assert_eq!(acc.absorption().value, 0.0);
    assert_eq!(acc.absorption().error_bound, 0.0);
    acc.push(3.0).unwrap();
    assert_eq!(acc.absorption().value, 0.0);
}

#[test]
fn invalid_inputs_are_typed_errors() {
    let settings = PhaseTypeSettings::default();
    for h in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            PhaseTypeAccumulator::new(h, settings.clone()),
            Err(PhaseTypeError::InvalidHorizon { .. })
        ));
    }
    let mut acc = PhaseTypeAccumulator::new(1.0, settings.clone()).unwrap();
    for q in [-0.5, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            acc.push(q),
            Err(PhaseTypeError::InvalidRate { .. })
        ));
    }
    assert_eq!(acc.phase_count(), 0, "a refused rate is not appended");
    for p in [0.0, -1e-9, 1.0, f64::NAN] {
        let bad = PhaseTypeSettings {
            rel_precision: p,
            ..settings.clone()
        };
        assert!(matches!(
            PhaseTypeAccumulator::new(1.0, bad),
            Err(PhaseTypeError::InvalidSettings { .. })
        ));
    }
    let bad = PhaseTypeSettings {
        max_terms: 0,
        ..settings
    };
    assert!(matches!(
        PhaseTypeAccumulator::new(1.0, bad),
        Err(PhaseTypeError::InvalidSettings { .. })
    ));
}
