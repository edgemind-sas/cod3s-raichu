//! Law hazards and quantiles, validated against tabulated values.
//!
//! Reference numbers were generated ONCE with mpmath 1.4.1 at 50
//! significant digits (`mp.erf`, `mp.erfc`, `mp.erfinv`, `mp.ncdf`,
//! `mp.gammainc(..., regularized=True)`, `mp.findroot` for quantiles,
//! solved on the logarithm of the tail with a bracketing solver),
//! cross-checked against scipy 1.18.1 (`stats.lognorm.sf`,
//! `stats.gamma.sf`), then rounded to the nearest double and hardcoded
//! here. The tests never call Python. Closed forms (exponential, Weibull,
//! uniform, delay) are checked against their formula directly.
//!
//! Every comparison states its relative tolerance.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_numeric::laws::{
    erf, erfc, regularized_gamma_p, regularized_gamma_q, standard_normal_cdf,
    standard_normal_quantile, Law, LawError,
};

fn assert_rel(actual: f64, expected: f64, tol: f64, what: &str) {
    if expected == 0.0 {
        assert!(actual.abs() <= tol, "{what}: got {actual}, expected 0");
        return;
    }
    if expected.is_infinite() {
        assert_eq!(actual, expected, "{what}");
        return;
    }
    let rel = ((actual - expected) / expected).abs();
    assert!(
        rel <= tol,
        "{what}: got {actual:e}, expected {expected:e}, relative error {rel:e} > {tol:e}"
    );
}

// --- special functions ------------------------------------------------------

#[test]
fn erf_and_erfc_match_mpmath() {
    // (x, erf(x), erfc(x)) from mpmath at 50 digits.
    let table = [
        (0.0, 0.0, 1.0),
        (1e-8, 1.1283791670955126e-08, 0.9999999887162083),
        (0.3, 0.3286267594591274, 0.6713732405408726),
        (0.5, 0.5204998778130465, 0.4795001221869535),
        (1.0, 0.8427007929497149, 0.15729920705028513),
        (1.3, 0.9340079449406524, 0.06599205505934755),
        (2.0, 0.9953222650189527, 0.004677734981047266),
        (3.5, 0.9999992569016276, 7.430983723414128e-07),
        (6.0, 1.0, 2.1519736712498913e-17),
        (10.0, 1.0, 2.088487583762545e-45),
        (26.0, 1.0, 5.663192408856143e-296),
        (-0.7, -0.6778011938374184, 1.6778011938374184),
        (-3.0, -0.9999779095030014, 1.9999779095030015),
    ];
    for (x, e, c) in table {
        assert_rel(erf(x), e, 1e-14, &format!("erf({x})"));
        // Tail precision degrades like x^2 * eps (rounding of x^2).
        let tol = 1e-14 * (1.0 + x * x);
        assert_rel(erfc(x), c, tol, &format!("erfc({x})"));
    }
}

#[test]
fn regularized_incomplete_gamma_matches_mpmath() {
    // (a, x, P(a, x), Q(a, x)) from mpmath at 50 digits.
    let table = [
        (0.5, 0.25, 0.5204998778130465, 0.4795001221869535),
        (0.3, 1e-5, 0.03523536061556258, 0.9647646393844375),
        (0.3, 0.5, 0.8138118046743926, 0.18618819532560735),
        (0.3, 5.0, 0.9993486812492816, 0.0006513187507184515),
        (2.5, 1e-4, 3.008796191247676e-11, 0.9999999999699121),
        (2.5, 2.5, 0.5841198130044921, 0.41588018699550794),
        (2.5, 3.5, 0.7793596920632893, 0.2206403079367108),
        (2.5, 10.0, 0.9987502694369687, 0.0012497305630313753),
        (2.5, 100.0, 1.0, 2.8406228986415315e-41),
        (150.0, 100.0, 1.88421046603867e-06, 0.999998115789534),
        (150.0, 150.0, 0.5108582297493597, 0.4891417702506403),
        (150.0, 200.0, 0.9999032137800506, 9.678621994933577e-05),
        (1000.0, 900.0, 0.0005499022657117829, 0.9994500977342882),
        (1000.0, 1000.0, 0.5042052441802155, 0.4957947558197845),
        (1000.0, 1100.0, 0.99894067674607, 0.0010593232539299773),
        (5.0, 700.0, 1.0, 9.920391479800145e-295),
    ];
    for (a, x, p, q) in table {
        assert_rel(regularized_gamma_p(a, x), p, 1e-12, &format!("P({a}, {x})"));
        assert_rel(regularized_gamma_q(a, x), q, 1e-12, &format!("Q({a}, {x})"));
    }
}

#[test]
fn standard_normal_quantile_matches_mpmath_and_inverts_the_cdf() {
    // (p, Phi^-1(p)) from mpmath (erfinv, findroot for 1e-300).
    let table = [
        (1e-300, -37.0470962993612),
        (1e-20, -9.262340089798407),
        (1e-5, -4.264890793922825),
        (0.02, -2.053748910631823),
        (0.3, -0.5244005127080408),
        (0.5, 0.0),
        (0.9, 1.2815515655446006),
        (0.999999, 4.753424308817087),
    ];
    for (p, z) in table {
        assert_rel(standard_normal_quantile(p), z, 1e-13, &format!("ppf({p})"));
        if p < 0.5 {
            assert_rel(standard_normal_cdf(z), p, 1e-12, &format!("cdf({z})"));
        }
    }
    assert_eq!(standard_normal_quantile(0.0), f64::NEG_INFINITY);
    assert_eq!(standard_normal_quantile(1.0), f64::INFINITY);
}

// --- parameter validation ---------------------------------------------------

#[test]
fn invalid_parameters_are_typed_errors() {
    assert!(matches!(
        Law::delay(-1.0),
        Err(LawError::InvalidParameter { law: "delay", .. })
    ));
    assert!(matches!(
        Law::exponential(f64::NAN),
        Err(LawError::InvalidParameter {
            law: "exponential",
            ..
        })
    ));
    assert!(Law::weibull(0.0, 1.0).is_err());
    assert!(Law::weibull(1.0, -1.0).is_err());
    assert!(Law::lognormal(f64::INFINITY, 1.0).is_err());
    assert!(Law::lognormal(0.0, 0.0).is_err());
    assert!(Law::gamma(-2.0, 1.0).is_err());
    assert!(Law::uniform(-1.0, 1.0).is_err());
    assert!(Law::uniform(2.0, 2.0).is_err());
    assert!(matches!(
        Law::empirical(vec![]),
        Err(LawError::InvalidEmpiricalTable { .. })
    ));
    assert!(Law::empirical(vec![(1.0, 0.5), (2.0, 0.4), (3.0, 1.0)]).is_err());
    assert!(Law::empirical(vec![(1.0, 0.5), (2.0, 0.9)]).is_err());
    let law = Law::exponential(1.0).unwrap();
    assert!(matches!(
        law.quantile(1.5),
        Err(LawError::ProbabilityOutOfRange { .. })
    ));
    assert!(matches!(
        law.inverse_cumulative_hazard(-1.0),
        Err(LawError::InvalidHazard { .. })
    ));
    assert!(matches!(
        law.conditional_survival(-1.0, 1.0),
        Err(LawError::InvalidAge { .. })
    ));
    assert!(matches!(
        law.conditional_survival(1.0, -1.0),
        Err(LawError::InvalidDuration { .. })
    ));
}

// --- per-law reference values -----------------------------------------------

#[test]
fn exponential_closed_form_and_tails() {
    let law = Law::exponential(0.5).unwrap();
    for t in [0.0, 1.0, 10.0, 100.0, 1000.0] {
        assert_rel(law.survival(t), (-0.5 * t).exp(), 1e-15, "exp S");
        assert_rel(law.cumulative_hazard(t), 0.5 * t, 1e-15, "exp H");
    }
    // Small-t hazard and CDF keep full relative precision.
    assert_rel(law.cumulative_hazard(1e-20), 5e-21, 1e-15, "exp H small");
    assert_rel(law.cdf(1e-20), 5e-21, 1e-15, "exp F small");
    assert_rel(law.quantile(1e-18).unwrap(), 2e-18, 1e-15, "exp Q small");
    assert_rel(
        law.inverse_survival(1e-300).unwrap(),
        -(1e-300f64).ln() / 0.5,
        1e-15,
        "exp isf",
    );
}

#[test]
fn weibull_matches_mpmath() {
    // Weibull(k = 1.5, lambda = 1000): (t, S, F, H) from mpmath.
    let law = Law::weibull(1.5, 1000.0).unwrap();
    let table = [
        (10.0, 0.999000499833375, 0.0009995001666250082, 0.001),
        (
            500.0,
            0.7021885013265596,
            0.2978114986734404,
            0.3535533905932738,
        ),
        (1000.0, 0.36787944117144233, 0.6321205588285577, 1.0),
        (
            5000.0,
            1.394569237787393e-05,
            0.9999860543076221,
            11.180339887498949,
        ),
    ];
    for (t, s, f, h) in table {
        assert_rel(law.survival(t), s, 1e-14, "weibull S");
        assert_rel(law.cdf(t), f, 1e-14, "weibull F");
        assert_rel(law.cumulative_hazard(t), h, 1e-14, "weibull H");
    }
    // H = (t / lambda)^k exactly, even far below 1e-12.
    let tiny = 1e-9f64.powf(1.5);
    assert_rel(law.cumulative_hazard(1e-6), tiny, 1e-14, "H tiny");
    assert_rel(law.cdf(1e-6), tiny - tiny * tiny / 2.0, 1e-15, "F tiny");
}

#[test]
fn lognormal_matches_mpmath() {
    // Lognormal(mu = 2, sigma = 0.8): (t, F, S, H) from mpmath.
    let law = Law::lognormal(2.0, 0.8).unwrap();
    let table = [
        (1e-3, 4.2518228216791845e-29, 1.0, 4.2518228216791845e-29),
        (
            0.5,
            0.0003807340412645056,
            0.9996192659587355,
            0.00038080653887171637,
        ),
        (
            3.0,
            0.12992734617106338,
            0.8700726538289366,
            0.13917856067208295,
        ),
        (7.38905609893065, 0.5, 0.5, std::f64::consts::LN_2),
        (
            20.0,
            0.8933726046214501,
            0.10662739537854997,
            2.2384148079731663,
        ),
        (
            1000.0,
            0.9999999995733839,
            4.266160621580387e-10,
            21.57513665903414,
        ),
        (1e6, 1.0, 1.1539923681837482e-49, 112.68344200200318),
    ];
    for (t, f, s, h) in table {
        assert_rel(law.cdf(t), f, 1e-13, &format!("lognormal F({t})"));
        assert_rel(law.survival(t), s, 1e-12, &format!("lognormal S({t})"));
        assert_rel(
            law.cumulative_hazard(t),
            h,
            1e-13,
            &format!("lognormal H({t})"),
        );
    }
    // (u, quantile) from mpmath.
    for (u, q) in [
        (1e-10, 0.04554416693774406),
        (0.1, 2.650527687581952),
        (0.5, 7.38905609893065),
        (0.9, 20.598973664354947),
    ] {
        assert_rel(law.quantile(u).unwrap(), q, 1e-13, "lognormal Q");
    }
}

#[test]
fn gamma_matches_mpmath() {
    // Gamma(k = 2.5, theta = 3): (t, F, S, H) from mpmath.
    let law = Law::gamma(2.5, 3.0).unwrap();
    let table = [
        (
            3e-4,
            3.0087961912476746e-11,
            0.9999999999699121,
            3.008796191292939e-11,
        ),
        (
            3.0,
            0.15085496391539036,
            0.8491450360846097,
            0.16352527559465035,
        ),
        (
            7.5,
            0.5841198130044921,
            0.41588018699550794,
            0.8773580722343328,
        ),
        (
            30.0,
            0.9987502694369687,
            0.0012497305630313753,
            6.684827300476975,
        ),
        (300.0, 1.0, 2.8406228986415315e-41, 93.3619654541246),
    ];
    for (t, f, s, h) in table {
        assert_rel(law.cdf(t), f, 1e-12, &format!("gamma F({t})"));
        assert_rel(law.survival(t), s, 1e-12, &format!("gamma S({t})"));
        assert_rel(law.cumulative_hazard(t), h, 1e-12, &format!("gamma H({t})"));
    }
    for (u, q) in [
        (1e-12, 7.686965245245804e-05),
        (0.01, 0.8314471150924158),
        (0.5, 6.527190286643291),
        (0.99, 22.62940870408348),
    ] {
        assert_rel(law.quantile(u).unwrap(), q, 1e-12, "gamma Q");
    }
    assert_rel(
        law.inverse_survival(1e-100).unwrap(),
        714.5691555962442,
        1e-12,
        "gamma isf",
    );
}

#[test]
fn uniform_closed_form() {
    let law = Law::uniform(2.0, 10.0).unwrap();
    assert_eq!(law.survival(1.0), 1.0);
    assert_eq!(law.cumulative_hazard(2.0), 0.0);
    assert_rel(law.survival(4.0), 0.75, 1e-15, "uniform S");
    assert_rel(law.cumulative_hazard(4.0), -(0.75f64).ln(), 1e-15, "H");
    let t = 2.0 + 8e-15;
    assert_rel(law.cumulative_hazard(t), (t - 2.0) / 8.0, 1e-14, "H small");
    assert_eq!(law.survival(10.0), 0.0);
    assert_eq!(law.cumulative_hazard(10.0), f64::INFINITY);
    assert_eq!(law.quantile(0.0).unwrap(), 2.0);
    assert_eq!(law.quantile(1.0).unwrap(), 10.0);
    assert_rel(law.quantile(0.25).unwrap(), 4.0, 1e-15, "uniform Q");
}

#[test]
fn delay_has_zero_hazard_before_its_date_and_certain_firing_at_it() {
    let law = Law::delay(5.0).unwrap();
    for t in [0.0, 1.0, 4.999999] {
        assert_eq!(law.cumulative_hazard(t), 0.0);
        assert_eq!(law.survival(t), 1.0);
        assert_eq!(law.cdf(t), 0.0);
    }
    for t in [5.0, 6.0, f64::INFINITY] {
        assert_eq!(law.cumulative_hazard(t), f64::INFINITY);
        assert_eq!(law.survival(t), 0.0);
        assert_eq!(law.cdf(t), 1.0);
    }
    for u in [0.0, 0.3, 1.0] {
        assert_eq!(law.quantile(u).unwrap(), 5.0);
    }
    // Conditional on age 2: fires exactly 3 later.
    assert_eq!(law.conditional_survival(2.0, 2.9).unwrap(), 1.0);
    assert_eq!(law.conditional_survival(2.0, 3.0).unwrap(), 0.0);
    assert_eq!(law.conditional_quantile(2.0, 0.5).unwrap(), 3.0);
    // At or beyond the date the conditional law does not exist.
    assert!(matches!(
        law.conditional_survival(5.0, 1.0),
        Err(LawError::AgeBeyondSupport { .. })
    ));
    // A zero delay fires at once.
    let zero = Law::delay(0.0).unwrap();
    assert_eq!(zero.survival(0.0), 0.0);
}

// --- empirical: the exact inverse of the engine sampler ---------------------

/// The engine's inverse-CDF sampler, `sample_empirical` in
/// `crates/raichu-core/src/engine.rs`, reproduced verbatim.
fn engine_sample_empirical(points: &[(f64, f64)], u: f64) -> f64 {
    let (first_t, first_c) = points[0];
    if u <= first_c {
        return first_t;
    }
    for window in points.windows(2) {
        let (t0, c0) = window[0];
        let (t1, c1) = window[1];
        if u <= c1 {
            if c1 == c0 {
                return t1;
            }
            return t0 + (t1 - t0) * (u - c0) / (c1 - c0);
        }
    }
    points[points.len() - 1].0
}

/// Atom at 1 (mass 0.2), flat CDF on [1, 2], linear to 4, atom at 4
/// (mass 0.2), linear to 6.
fn table() -> Vec<(f64, f64)> {
    vec![(1.0, 0.2), (2.0, 0.2), (4.0, 0.7), (4.0, 0.9), (6.0, 1.0)]
}

#[test]
fn empirical_quantile_is_the_engine_sampler_and_the_cdf_its_inverse() {
    let points = table();
    let law = Law::empirical(points.clone()).unwrap();
    for i in 0..=1000 {
        let u = i as f64 / 1000.0;
        let t = law.quantile(u).unwrap();
        assert_eq!(t, engine_sample_empirical(&points, u), "Q({u})");
        // Generalised inverse: F(Q(u)) >= u, and Q(F(t)) <= t.
        assert!(law.cdf(t) >= u - 1e-15, "F(Q({u})) = {} < u", law.cdf(t));
    }
    for i in 0..=700 {
        let t = i as f64 / 100.0;
        let f = law.cdf(t);
        if f == 0.0 {
            // Q(0) is the lower end of the support, by convention.
            continue;
        }
        assert!(law.quantile(f).unwrap() <= t + 1e-14, "Q(F({t})) > t");
    }
    // Right-continuous CDF with the two atoms.
    assert_eq!(law.cdf(0.999), 0.0);
    assert_eq!(law.cdf(1.0), 0.2);
    assert_eq!(law.cdf(1.5), 0.2);
    assert_rel(law.cdf(3.0), 0.45, 1e-15, "linear part");
    assert_rel(law.cdf(4.0), 0.9, 1e-15, "atom at 4");
    assert_rel(law.cdf(5.0), 0.95, 1e-15, "last segment");
    assert_eq!(law.cdf(6.0), 1.0);
    assert_eq!(law.cumulative_hazard(6.0), f64::INFINITY);
    assert_rel(law.cumulative_hazard(3.0), -(0.55f64).ln(), 1e-15, "H");
    // On a flat stretch the conditional law starts where the CDF rises.
    assert_rel(
        law.conditional_quantile(1.5, 0.0).unwrap(),
        0.5,
        1e-15,
        "flat",
    );
    assert_eq!(law.conditional_quantile(3.0, 0.0).unwrap(), 0.0);
}

// --- generic properties, every law ------------------------------------------

fn all_laws() -> Vec<(&'static str, Law)> {
    vec![
        ("exponential", Law::exponential(0.02).unwrap()),
        ("weibull", Law::weibull(2.3, 40.0).unwrap()),
        ("weibull<1", Law::weibull(0.6, 40.0).unwrap()),
        ("lognormal", Law::lognormal(3.0, 0.5).unwrap()),
        ("gamma", Law::gamma(2.5, 3.0).unwrap()),
        ("gamma<1", Law::gamma(0.4, 20.0).unwrap()),
        ("uniform", Law::uniform(5.0, 60.0).unwrap()),
        ("empirical", Law::empirical(table()).unwrap()),
    ]
}

#[test]
fn quantile_inverts_the_cdf_on_a_grid() {
    for (name, law) in all_laws() {
        for i in 1..100 {
            let u = i as f64 / 100.0;
            let t = law.quantile(u).unwrap();
            if name != "empirical" {
                assert_rel(law.cdf(t), u, 1e-12, &format!("{name}: F(Q({u}))"));
                let p = 1.0 - u;
                let ts = law.inverse_survival(p).unwrap();
                assert_rel(ts, t, 1e-12, &format!("{name}: isf({p})"));
                let h = law.cumulative_hazard(t);
                assert_rel(
                    law.inverse_cumulative_hazard(h).unwrap(),
                    t,
                    1e-12,
                    &format!("{name}: H^-1(H(t))"),
                );
            }
        }
    }
}

#[test]
fn conditional_survival_is_the_survival_ratio() {
    for (name, law) in all_laws() {
        for age in [0.0, 1.0, 4.5, 20.0] {
            let sa = law.survival(age);
            if sa == 0.0 {
                continue;
            }
            for s in [0.0, 0.5, 3.0, 10.0, 50.0] {
                let expected = law.survival(age + s) / sa;
                let got = law.conditional_survival(age, s).unwrap();
                assert!(
                    (got - expected).abs() <= 1e-12 * expected.max(1e-300),
                    "{name}: S({age}+{s})/S({age}) = {expected}, got {got}"
                );
                let dh = law.conditional_cumulative_hazard(age, s).unwrap();
                let direct = law.cumulative_hazard(age + s) - law.cumulative_hazard(age);
                if direct.is_finite() {
                    assert!(
                        (dh - direct).abs() <= 1e-12 * (1.0 + direct.abs()),
                        "{name}: dH({age}, {s}) = {dh}, direct {direct}"
                    );
                } else {
                    assert_eq!(dh, f64::INFINITY);
                }
            }
            // Conditional quantile inverts the conditional CDF.
            if name != "empirical" {
                for u in [0.1, 0.5, 0.9] {
                    let s = law.conditional_quantile(age, u).unwrap();
                    let fc = 1.0 - law.conditional_survival(age, s).unwrap();
                    assert_rel(fc, u, 1e-10, &format!("{name}: cond Q({age}, {u})"));
                }
            }
        }
    }
}

#[test]
fn conditional_hazard_keeps_precision_for_short_windows() {
    // Weibull: H(a + s) - H(a) for s << a, closed form with expm1/log1p.
    let law = Law::weibull(2.0, 100.0).unwrap();
    let (a, s) = (50.0, 1e-9);
    // ((a+s)^2 - a^2) / 100^2 = (2 a s + s^2) / 1e4.
    let expected = (2.0 * a * s + s * s) / 1e4;
    assert_rel(
        law.conditional_cumulative_hazard(a, s).unwrap(),
        expected,
        1e-14,
        "weibull dH short window",
    );
    // Exponential is memoryless.
    let law = Law::exponential(3.0).unwrap();
    assert_rel(
        law.conditional_cumulative_hazard(1e6, 1e-12).unwrap(),
        3e-12,
        1e-15,
        "exp dH",
    );
    assert_rel(
        law.conditional_quantile(1e6, 0.5).unwrap(),
        (2.0f64).ln() / 3.0,
        1e-15,
        "exp memoryless quantile",
    );
}

#[test]
fn large_time_survival_keeps_relative_precision() {
    // S far below 1e-300 would underflow; H keeps it.
    // Reference values from mpmath at 50 digits.
    let law = Law::lognormal(2.0, 0.8).unwrap();
    assert_eq!(law.survival(1e30), 0.0);
    assert_rel(
        law.cumulative_hazard(1e30),
        3520.502830191478,
        1e-13,
        "lognormal H(1e30)",
    );
    let gamma = Law::gamma(2.5, 3.0).unwrap();
    assert_eq!(gamma.survival(3000.0), 0.0);
    assert_rel(
        gamma.cumulative_hazard(3000.0),
        989.9215503273734,
        1e-13,
        "gamma H(3000)",
    );
}

#[test]
fn gamma_and_lognormal_quantiles_hold_in_both_tails_across_shapes() {
    for shape in [0.05, 0.5, 1.0, 7.3, 1e4] {
        let law = Law::gamma(shape, 2.0).unwrap();
        let smallest = if shape < 0.1 { 1e-10 } else { 1e-100 };
        for target in [smallest, 1e-12, 1e-3, 0.2, 0.5] {
            let t = law.quantile(target).unwrap();
            assert_rel(
                law.cdf(t),
                target,
                1e-11,
                &format!("gamma({shape}) F(Q({target}))"),
            );
            let t = law.inverse_survival(target).unwrap();
            assert_rel(
                law.survival(t),
                target,
                1e-11,
                &format!("gamma({shape}) S(isf({target}))"),
            );
        }
    }
    let law = Law::lognormal(-1.0, 2.5).unwrap();
    for target in [1e-250, 1e-12, 0.2, 0.5] {
        let t = law.quantile(target).unwrap();
        assert_rel(
            law.cdf(t),
            target,
            1e-12,
            &format!("lognormal F(Q({target}))"),
        );
        let t = law.inverse_survival(target).unwrap();
        assert_rel(
            law.survival(t),
            target,
            1e-12,
            &format!("lognormal S(isf({target}))"),
        );
    }
}
