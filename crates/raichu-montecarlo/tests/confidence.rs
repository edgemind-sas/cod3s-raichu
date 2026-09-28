//! Confidence intervals, validated against **closed forms**.
//!
//! No engine is a reference here: neither RAICHU nor any neighbouring
//! tool reports intervals, so there is nothing to compare against. What
//! there is instead is analysis. The model under test is a single
//! `ok → nok` exponential transition, whose firing probability is known
//! exactly, `P(T ≤ t) = 1 − e^{−λt}`, and every claim below is checked
//! against that closed form or against the closed form of the interval
//! itself:
//!
//! - the normal quantile against published values of `Φ⁻¹`;
//! - the Wilson and normal bounds against their algebraic definition,
//!   special cases included (`p = 0`, `p = 1`);
//! - **the coverage**, which is what a confidence interval actually
//!   promises: over 200 independent campaigns, the share that brackets
//!   the exact probability must match the declared level, and must fall
//!   when the declared level falls;
//! - **the two degenerate ends**, an event no replica reached and one
//!   every replica reached, where the same probe is driven by a rate
//!   small enough and large enough for the sample to be constant: the
//!   bound must be the frequency closed form `z²/(n + z²)` and not a
//!   point;
//! - the `1/√n` narrowing, which is the property the whole thing is
//!   about: 200 replicas and 20 000 must not report the same precision.
//!
//! Seeds are fixed throughout, so every verdict here is reproducible.

#![allow(clippy::unwrap_used, clippy::panic, clippy::expect_used)]

use std::hint::black_box;
use std::time::Instant;

use raichu_core::{CompiledModel, EngineError, FlowConfig};
use raichu_model::{Automaton, Component, Distrib, Indicator, IndicatorTarget, Model, Transition};
use raichu_montecarlo::{
    constant_sample_bounds, normal_bounds, normal_quantile, run, unobserved_frequency_bound,
    weighted_interval, wilson_bounds, z_of, ConfidenceInterval, Departure, IntervalMethod,
    McConfig, WeightedInterval, DEFAULT_CONFIDENCE,
};

/// A single exponential `ok → nok` transition at rate `rate`, observed
/// through a state indicator: `E[1{nok at t}] = 1 − e^{−rate·t}`.
fn exponential_model(rate: f64) -> Model {
    Model {
        name: "ci_probe".into(),
        components: vec![Component {
            name: "C".into(),
            attributes: vec![],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "aut".into(),
                states: vec!["ok".into(), "nok".into()],
                init: "ok".into(),
                transitions: vec![Transition {
                    name: "fire".into(),
                    source: "ok".into(),
                    guard: None,
                    targets: vec!["nok".into()],
                    on_interruption: Default::default(),
                    monitored: false,
                    cycle_group: None,
                    kind: None,
                    effects: vec![],
                    distrib: Distrib::Exp {
                        rate: Some(rate),
                        rate_expr: None,
                    },
                }],
            }],
            allocations: vec![],
            equations: vec![],
            sensitive_functions: vec![],
        }],
        connections: vec![],
        indicators: vec![Indicator {
            name: "fired".into(),
            target: IndicatorTarget::State {
                component: "C".into(),
                automaton: "aut".into(),
                state: "nok".into(),
            },
        }],
        targets: vec![],
        evaluation_order: None,
        unbounded_rate: None,
    }
}

fn config(nb_runs: u64, seed: u64, instants: &[f64], confidence: f64) -> McConfig {
    McConfig {
        nb_runs,
        seed,
        t_max: *instants.last().unwrap(),
        samples: instants.to_vec(),
        threads: None,
        quantiles: vec![],
        confidence,
        ode: Default::default(),
        stop_at_targets: false,
        flow: FlowConfig::default(),
    }
}

/// Exact firing probability of the probe model at `t`.
fn exact_probability(rate: f64, t: f64) -> f64 {
    1.0 - (-rate * t).exp()
}

// ---------------------------------------------------------------------
// The quantile, against published values of Φ⁻¹
// ---------------------------------------------------------------------

#[test]
fn normal_quantile_matches_published_values() {
    // Reference values of Φ⁻¹ to full double precision. The central
    // branch, both tail branches (|r| ≤ 5 and beyond) and the midpoint
    // are all exercised.
    let reference: [(f64, f64); 9] = [
        (0.5, 0.0),
        (0.75, 0.674_489_750_196_081_7),
        (0.9, 1.281_551_565_544_600_8),
        (0.95, 1.644_853_626_951_471_5),
        (0.975, 1.959_963_984_540_053_6),
        (0.995, 2.575_829_303_548_9),
        (0.999, 3.090_232_306_167_813),
        (0.999_999_9, 5.199_337_582_290_662),
        (1e-12, -7.034_483_825_301_132),
    ];
    for (p, expected) in reference {
        let got = normal_quantile(p);
        let tolerance = 1e-12 * expected.abs().max(1.0);
        assert!(
            (got - expected).abs() < tolerance,
            "Phi^-1({p}): got {got}, published {expected}"
        );
    }
}

#[test]
fn normal_quantile_is_antisymmetric_about_one_half() {
    // Only orders whose complement is an exact double are used: for
    // `p = 1e-8`, `1 − p` is not representable and the round trip loses
    // the eight digits the comparison would be about, which says
    // something about binary arithmetic and nothing about the quantile.
    // Both the central branch (0.25) and the tail branch (0.0625) are
    // covered.
    for p in [0.062_5, 0.125, 0.25, 0.375] {
        let left = normal_quantile(p);
        let right = normal_quantile(1.0 - p);
        assert!(
            (left + right).abs() < 1e-15 * right.abs().max(1.0),
            "Phi^-1({p}) = {left} is not the opposite of Phi^-1({}) = {right}",
            1.0 - p
        );
    }
}

#[test]
fn the_two_sided_deviate_is_the_one_the_tables_print() {
    // The deviates a safety report cites, from the confidence level and
    // not from a hard-coded 1.96.
    for (level, expected) in [
        (0.80, 1.281_551_565_544_600_8),
        (0.90, 1.644_853_626_951_471_5),
        (0.95, 1.959_963_984_540_053_6),
        (0.99, 2.575_829_303_548_9),
    ] {
        let got = z_of(level);
        assert!(
            (got - expected).abs() < 1e-12,
            "z({level}): got {got}, expected {expected}"
        );
    }
}

// ---------------------------------------------------------------------
// The bounds, against their algebraic definition
// ---------------------------------------------------------------------

#[test]
fn wilson_bounds_match_their_algebraic_definition() {
    let (n, level) = (500_u64, 0.95);
    let z = 1.959_963_984_540_053_6_f64;
    for p in [0.0, 0.001, 0.0483, 0.5, 0.9, 1.0] {
        let nf = n as f64;
        let denom = 1.0 + z * z / nf;
        let center = (p + z * z / (2.0 * nf)) / denom;
        let half = z / denom * (p * (1.0 - p) / nf + z * z / (4.0 * nf * nf)).sqrt();
        let (low, high) = wilson_bounds(p, n, level);
        assert!((low - (center - half).max(0.0)).abs() < 1e-15, "p = {p}");
        assert!((high - (center + half).min(1.0)).abs() < 1e-15, "p = {p}");
        assert!((0.0..=1.0).contains(&low) && (0.0..=1.0).contains(&high));
    }
}

#[test]
fn wilson_bounds_have_the_known_closed_form_at_the_ends() {
    // The two cases the textbook interval gets wrong: an event no
    // replica reached, and one every replica reached. Wilson has an
    // exact closed form at both, and neither interval is degenerate.
    let (n, level) = (500_u64, 0.95);
    let z2 = z_of(level).powi(2);
    let nf = n as f64;

    let (low, high) = wilson_bounds(0.0, n, level);
    assert_eq!(low, 0.0);
    assert!((high - z2 / (nf + z2)).abs() < 1e-15);
    assert!(
        high > 0.0076 && high < 0.0077,
        "0 of 500 gives {low}..{high}"
    );

    let (low, high) = wilson_bounds(1.0, n, level);
    assert!((low - nf / (nf + z2)).abs() < 1e-15);
    assert_eq!(high, 1.0);
}

#[test]
fn normal_bounds_match_their_algebraic_definition() {
    for (mean, std, n, level) in [
        (0.0483, 0.2144, 500_u64, 0.95),
        (12.5, 3.25, 1000, 0.99),
        (-4.0, 1.0, 30, 0.80),
    ] {
        let half = z_of(level) * std / (n as f64).sqrt();
        let (low, high) = normal_bounds(mean, std, n, level);
        assert!((low - (mean - half)).abs() < 1e-15);
        assert!((high - (mean + half)).abs() < 1e-15);
    }
}

// ---------------------------------------------------------------------
// Coverage: the property an interval actually promises
// ---------------------------------------------------------------------

/// Share of `campaigns` independent campaigns whose interval on the
/// sampled value at the single instant brackets `truth`.
fn covered_campaigns(
    model: &CompiledModel,
    campaigns: u64,
    nb_runs: u64,
    instant: f64,
    level: f64,
    truth: f64,
) -> u64 {
    (0..campaigns)
        .filter(|campaign| {
            // One master seed per campaign: substreams of distinct
            // seeds, so the campaigns are independent draws.
            let estimates = run(model, &config(nb_runs, 1_000 + campaign, &[instant], level))
                .expect("the probe model always runs");
            let ci = &estimates.indicators[0].ci;
            ci.low[0] <= truth && truth <= ci.high[0]
        })
        .count() as u64
}

#[test]
fn the_interval_covers_the_closed_form_at_the_declared_rate() {
    // λ = 0.3 at t = 2 gives the exact P = 1 − e^{−0.6} = 0.451188…,
    // and 200 campaigns of 200 replicas each say how often a 95 %
    // interval brackets it. Expected 190; the count has standard
    // deviation √(200 · 0.95 · 0.05) = 3.08, so the ±4σ acceptance
    // window is [178, 200]: wide enough never to flake on a correct
    // implementation, narrow enough that a one-sided or mis-scaled
    // interval falls out of it (a 90 % interval would land near 180,
    // a 99 % one near 198).
    let (rate, instant, campaigns, nb_runs) = (0.3, 2.0, 200_u64, 200_u64);
    let truth = exact_probability(rate, instant);
    let compiled = CompiledModel::compile(&exponential_model(rate)).unwrap();

    let covered = covered_campaigns(&compiled, campaigns, nb_runs, instant, 0.95, truth);
    assert!(
        (178..=200).contains(&covered),
        "95 % interval covered the closed form {covered}/{campaigns} times, \
         expected 190 ± 4σ (σ = 3.08)"
    );
}

#[test]
fn lowering_the_declared_level_lowers_the_coverage() {
    // The level is a study parameter, and this is what makes it one: on
    // the *same* campaigns, the 80 % interval is nested inside the 95 %
    // one, so it can never cover more; and its coverage lands on 160,
    // not on 190. Expected 160 with σ = √(200 · 0.8 · 0.2) = 5.66, so
    // the ±4σ window is [137, 183] and the two levels cannot be
    // confused.
    let (rate, instant, campaigns, nb_runs) = (0.3, 2.0, 200_u64, 200_u64);
    let truth = exact_probability(rate, instant);
    let compiled = CompiledModel::compile(&exponential_model(rate)).unwrap();

    let at_95 = covered_campaigns(&compiled, campaigns, nb_runs, instant, 0.95, truth);
    let at_80 = covered_campaigns(&compiled, campaigns, nb_runs, instant, 0.80, truth);

    assert!(
        (137..=183).contains(&at_80),
        "80 % interval covered {at_80}/{campaigns} times, expected 160 ± 4σ (σ = 5.66)"
    );
    assert!(
        at_80 < at_95,
        "nested intervals on identical campaigns: 80 % covered {at_80}, 95 % covered {at_95}"
    );
}

// ---------------------------------------------------------------------
// The whole point: 500 replicas must not read like 100 000
// ---------------------------------------------------------------------

#[test]
fn the_interval_narrows_as_one_over_root_n() {
    // Hundredfold the replicas, and the half-width must fall by ten:
    // the point estimate alone cannot tell a 200-replica campaign from
    // a 20 000-replica one, the interval can.
    let (rate, instant) = (0.3, 2.0);
    let compiled = CompiledModel::compile(&exponential_model(rate)).unwrap();

    let half_width = |nb_runs: u64| {
        let est = run(&compiled, &config(nb_runs, 7, &[instant], 0.95)).unwrap();
        let ci = &est.indicators[0].ci;
        0.5 * (ci.high[0] - ci.low[0])
    };
    let small = half_width(200);
    let large = half_width(20_000);
    let ratio = small / large;

    assert!(
        (9.0..=11.0).contains(&ratio),
        "half-width fell from {small} to {large}: ratio {ratio}, expected ≈ 10"
    );
}

#[test]
fn a_campaign_that_saw_nothing_still_bounds_the_probability() {
    // The failure mode that makes the textbook interval unusable in a
    // safety deliverable: a feared event so rare that no replica reached
    // it. `p ± z·√(p(1−p)/n)` answers [0, 0], i.e. "impossible", from
    // 500 trajectories. Wilson answers what 500 trajectories establish,
    // the closed form z²/(n + z²) = 0.00762…
    let rate = 1e-12;
    let compiled = CompiledModel::compile(&exponential_model(rate)).unwrap();
    let estimates = run(&compiled, &config(500, 3, &[1.0], 0.95)).unwrap();
    let indicator = &estimates.indicators[0];

    assert_eq!(indicator.mean[0], 0.0, "the probe must fire in no replica");
    assert_eq!(indicator.std[0], 0.0);
    assert_eq!(indicator.ci.method, IntervalMethod::Wilson);
    assert_eq!(indicator.ci.low[0], 0.0);

    let z2 = z_of(0.95).powi(2);
    let expected_high = z2 / (500.0 + z2);
    assert!(
        (indicator.ci.high[0] - expected_high).abs() < 1e-15,
        "upper bound {} vs closed form {expected_high}",
        indicator.ci.high[0]
    );
    assert!(
        indicator.ci.high[0] > 0.0,
        "the interval must not claim certainty"
    );
}

// ---------------------------------------------------------------------
// What the estimates carry, and what the study declares
// ---------------------------------------------------------------------

#[test]
fn every_estimator_carries_an_interval_at_the_declared_level() {
    let compiled = CompiledModel::compile(&exponential_model(0.3)).unwrap();
    let instants = [0.5, 1.0, 2.0, 4.0];
    let estimates = run(&compiled, &config(400, 11, &instants, 0.99)).unwrap();

    assert_eq!(estimates.confidence, 0.99, "the result states its level");
    let indicator = &estimates.indicators[0];
    for ci in [
        &indicator.ci,
        &indicator.sojourn_ci,
        &indicator.nb_occurrences_ci,
    ] {
        assert_eq!(ci.level, 0.99);
        assert_eq!(ci.low.len(), instants.len());
        assert_eq!(ci.high.len(), instants.len());
    }
    // A state indicator is a probability by declaration; a sojourn and
    // an occurrence count are not.
    assert_eq!(indicator.ci.method, IntervalMethod::Wilson);
    assert_eq!(indicator.sojourn_ci.method, IntervalMethod::Normal);
    assert_eq!(indicator.nb_occurrences_ci.method, IntervalMethod::Normal);

    for k in 0..instants.len() {
        assert!(indicator.ci.low[k] <= indicator.mean[k]);
        assert!(indicator.mean[k] <= indicator.ci.high[k]);
        assert!(indicator.sojourn_ci.low[k] <= indicator.sojourn_mean[k]);
        assert!(indicator.sojourn_mean[k] <= indicator.sojourn_ci.high[k]);
    }
}

#[test]
fn raising_the_level_widens_every_interval_by_the_deviate_ratio() {
    // On identical campaigns the normal half-width is exactly
    // proportional to z, so the widths of the 99 % and the 95 % sojourn
    // intervals stand in the closed-form ratio 2.5758 / 1.9600.
    let compiled = CompiledModel::compile(&exponential_model(0.3)).unwrap();
    let instants = [1.0, 3.0];
    let at_95 = run(&compiled, &config(800, 5, &instants, 0.95)).unwrap();
    let at_99 = run(&compiled, &config(800, 5, &instants, 0.99)).unwrap();

    let expected = z_of(0.99) / z_of(0.95);
    for k in 0..instants.len() {
        let wide = at_99.indicators[0].sojourn_ci.high[k] - at_99.indicators[0].sojourn_ci.low[k];
        let narrow = at_95.indicators[0].sojourn_ci.high[k] - at_95.indicators[0].sojourn_ci.low[k];
        assert!(
            (wide / narrow - expected).abs() < 1e-12,
            "width ratio {} vs closed form {expected}",
            wide / narrow
        );
    }
}

#[test]
fn a_level_outside_zero_one_is_refused_before_the_campaign_runs() {
    let compiled = CompiledModel::compile(&exponential_model(0.3)).unwrap();
    for level in [0.0, 1.0, 1.5, -0.1, f64::NAN] {
        let error = run(&compiled, &config(10, 1, &[1.0], level)).unwrap_err();
        assert!(
            matches!(error, EngineError::InvalidStudyParameter { ref parameter, .. }
                     if parameter == "confidence"),
            "level {level} was accepted: {error:?}"
        );
    }
}

#[test]
fn one_replica_yields_no_interval_rather_than_a_perfect_one() {
    // A single trajectory shows no dispersion. Reporting [mean, mean]
    // would read as an exact answer; the estimate says instead that no
    // interval exists.
    let compiled = CompiledModel::compile(&exponential_model(0.3)).unwrap();
    let estimates = run(&compiled, &config(1, 2, &[2.0], DEFAULT_CONFIDENCE)).unwrap();
    let indicator = &estimates.indicators[0];

    assert_eq!(indicator.sojourn_ci.method, IntervalMethod::Undefined);
    assert_eq!(indicator.sojourn_ci.low, indicator.sojourn_mean);
    assert_eq!(indicator.sojourn_ci.high, indicator.sojourn_mean);
    // Wilson needs no dispersion estimate, so the probability still gets
    // a (very wide) interval from one draw.
    assert_eq!(indicator.ci.method, IntervalMethod::Wilson);
    assert!(indicator.ci.high[0] - indicator.ci.low[0] > 0.5);
}

#[test]
fn intervals_are_serialised_with_their_level_and_method() {
    let compiled = CompiledModel::compile(&exponential_model(0.3)).unwrap();
    let estimates = run(&compiled, &config(100, 9, &[2.0], 0.9)).unwrap();
    let json = serde_json::to_value(&estimates).unwrap();

    assert_eq!(json["confidence"], 0.9);
    let ci = &json["indicators"][0]["ci"];
    assert_eq!(ci["level"], 0.9);
    assert_eq!(ci["method"], "wilson");
    assert!(ci["low"].is_array() && ci["high"].is_array());
    assert!(ci["constant_sample"].is_array());
    assert_eq!(json["indicators"][0]["sojourn_ci"]["method"], "normal");

    // And on a campaign that saw no dispersion at all, the flag travels
    // with the bounds: the artefact says the bound is the frequency one
    // without the reader having to recompute a standard deviation.
    let flat = CompiledModel::compile(&exponential_model(1e-12)).unwrap();
    let degenerate = run(&flat, &config(100, 9, &[2.0], 0.9)).unwrap();
    let json = serde_json::to_value(&degenerate).unwrap();
    for series in ["ci", "sojourn_ci", "nb_occurrences_ci"] {
        assert_eq!(json["indicators"][0][series]["constant_sample"][0], true);
    }
}

// ---------------------------------------------------------------------
// A constant sample: the interval that must not close on a point
// ---------------------------------------------------------------------

/// A rate large enough that `1 − e^{−rate·t}` is 1 to the last bit over
/// the schedules below: the transition has fired in every replica, so
/// the sampled value and the occurrence count are constant at one.
const CERTAIN: f64 = 1e6;

/// A rate small enough that no replica fires: the sampled value, the
/// cumulated sojourn and the occurrence count are constant at zero.
const IMPOSSIBLE: f64 = 1e-12;

#[test]
fn the_frequency_bound_is_the_rule_of_three_in_exact_form() {
    // What a campaign that observed no departure at all establishes,
    // against its algebra: it is Wilson's own upper bound at p = 0, and
    // it is the textbook 3/n once n is large enough for the z² in the
    // denominator to stop mattering.
    let level = 0.95;
    let z2 = z_of(level).powi(2);
    for n in [2_u64, 50, 500, 1_000, 100_000] {
        let bound = unobserved_frequency_bound(n, level);
        assert!(
            (bound - z2 / (n as f64 + z2)).abs() < 1e-18,
            "n = {n}: {bound}"
        );
        assert_eq!(bound, wilson_bounds(0.0, n, level).1, "n = {n}");
    }
    // The two campaign sizes this ticket was written on: 500 replicas
    // bound the frequency at 0.76 %, 1000 at 0.38 %.
    assert!((unobserved_frequency_bound(500, level) - 0.007_624_340).abs() < 1e-9);
    assert!((unobserved_frequency_bound(1_000, level) - 0.003_826_758).abs() < 1e-9);
    // 3.84/n against the 3/n of the textbooks, once n is large.
    let ratio = unobserved_frequency_bound(100_000, level) * 100_000.0;
    assert!((ratio - 3.8415).abs() < 0.001, "3.84/n rule: got {ratio}/n");
}

#[test]
fn the_constant_sample_rule_and_wilson_agree_where_both_apply() {
    // The consistency the construction rests on: a 0/1 quantity whose
    // sample is constant gets the same bounds from the frequency rule
    // as from Wilson, at both ends. They part only above one, where a
    // count is free to go and a proportion is not.
    let (n, level) = (500_u64, 0.95);
    let epsilon = unobserved_frequency_bound(n, level);

    assert_eq!(
        constant_sample_bounds(0.0, 1.0, Some(0.0), n, level),
        wilson_bounds(0.0, n, level),
        "an outcome no draw showed"
    );
    let (low, high) = constant_sample_bounds(1.0, 1.0, Some(0.0), n, level);
    assert!(
        (low - wilson_bounds(1.0, n, level).0).abs() < 1e-15,
        "an outcome every draw showed: {low}"
    );
    assert!((high - (1.0 + epsilon)).abs() < 1e-15);

    // Without a declared floor the interval straddles the value: a
    // quantity that declares no support gets no one-sided bound.
    assert_eq!(
        constant_sample_bounds(0.0, 1.0, None, n, level),
        (-epsilon, epsilon)
    );
}

#[test]
fn a_constant_sample_reports_what_the_campaign_leaves_open_and_not_a_point() {
    // The defect, at the end where no replica reached the event. The
    // three estimators are constant at zero, so the three sample
    // standard deviations are zero and the two normal half-widths with
    // them: the intervals closed on [0, 0], which declares the event
    // impossible on the strength of 500 trajectories. What 500
    // trajectories establish is z²/(n + z²) — and at t = 1 the elapsed
    // time is 1, so the sojourn's own scale is 1 and the three closed
    // forms coincide on the same number.
    let compiled = CompiledModel::compile(&exponential_model(IMPOSSIBLE)).unwrap();
    let (n, level) = (500_u64, 0.95);
    let estimates = run(&compiled, &config(n, 3, &[1.0], level)).unwrap();
    let indicator = &estimates.indicators[0];
    let epsilon = unobserved_frequency_bound(n, level);

    for (name, mean, std) in [
        ("value", &indicator.mean, &indicator.std),
        ("sojourn", &indicator.sojourn_mean, &indicator.sojourn_std),
        (
            "occurrences",
            &indicator.nb_occurrences_mean,
            &indicator.nb_occurrences_std,
        ),
    ] {
        assert_eq!(mean[0], 0.0, "{name}: the probe must fire in no replica");
        assert_eq!(std[0], 0.0, "{name}: the sample must be constant");
    }

    for (name, ci) in [
        ("ci", &indicator.ci),
        ("sojourn_ci", &indicator.sojourn_ci),
        ("nb_occurrences_ci", &indicator.nb_occurrences_ci),
    ] {
        assert!(
            ci.constant_sample[0],
            "{name}: the constant sample is not marked"
        );
        assert_eq!(ci.low[0], 0.0, "{name}: a bound below zero claims nothing");
        assert!(
            (ci.high[0] - epsilon).abs() < 1e-15,
            "{name}: upper bound {} against the closed form {epsilon}",
            ci.high[0]
        );
        assert!(
            ci.high[0] > 0.0,
            "{name}: the interval must not claim impossibility"
        );
    }

    // The construction did not change: a draw with no dispersion is not
    // a declaration of type, so a sojourn does not become a proportion.
    assert_eq!(indicator.ci.method, IntervalMethod::Wilson);
    assert_eq!(indicator.sojourn_ci.method, IntervalMethod::Normal);
    assert_eq!(indicator.nb_occurrences_ci.method, IntervalMethod::Normal);
}

#[test]
fn an_event_every_replica_reached_keeps_a_bound_below_its_mean() {
    // The other end, and the one the real campaign showed: the
    // occurrence count of the feared event read `1.0000 ± 0.0000` with
    // the interval `[1.000000, 1.000000]`, i.e. the mean number of
    // occurrences is exactly one, established on 500 replicas. It could
    // be 0.999. The bound below it is 1 − z²/(n + z²), which is Wilson's
    // own closed form at p = 1.
    let compiled = CompiledModel::compile(&exponential_model(CERTAIN)).unwrap();
    let (n, level) = (500_u64, 0.95);
    let estimates = run(&compiled, &config(n, 21, &[1.0], level)).unwrap();
    let indicator = &estimates.indicators[0];
    let epsilon = unobserved_frequency_bound(n, level);

    assert_eq!(indicator.mean[0], 1.0, "every replica must have fired");
    assert_eq!(indicator.nb_occurrences_mean[0], 1.0);
    assert_eq!(indicator.nb_occurrences_std[0], 0.0);

    assert!((indicator.ci.low[0] - (1.0 - epsilon)).abs() < 1e-15);
    assert_eq!(indicator.ci.high[0], 1.0);
    assert!(indicator.ci.low[0] < indicator.mean[0]);

    let occurrences = &indicator.nb_occurrences_ci;
    assert!(occurrences.constant_sample[0]);
    assert!(
        (occurrences.low[0] - (1.0 - epsilon)).abs() < 1e-15,
        "lower bound {} against the closed form {}",
        occurrences.low[0],
        1.0 - epsilon
    );
    assert!((occurrences.high[0] - (1.0 + epsilon)).abs() < 1e-15);
    assert!(occurrences.low[0] < indicator.nb_occurrences_mean[0]);

    // The sojourn is not degenerate here, and must keep its own
    // interval: the replicas fire at different instants, so the time
    // spent in the state differs and the central-limit construction has
    // something to describe.
    assert!(indicator.sojourn_std[0] > 0.0);
    assert!(!indicator.sojourn_ci.constant_sample[0]);
    assert!(indicator.sojourn_ci.low[0] < indicator.sojourn_mean[0]);
}

#[test]
fn no_interval_closes_on_a_point_above_two_replicas() {
    // The property itself, over the three estimators, the two
    // degenerate ends, an ordinary campaign and the smallest campaign
    // that gets an interval at all. The schedule starts after the
    // origin: at the origin a cumulated sojourn is the integral over an
    // empty interval, zero for every trajectory of every model, and a
    // width of zero there is an identity rather than a claim (next
    // test).
    let instants = [0.5, 1.0, 4.0];
    for (label, rate, nb_runs) in [
        ("an event no replica reached", IMPOSSIBLE, 500_u64),
        ("an event every replica reached", CERTAIN, 500),
        ("an ordinary campaign", 0.3, 500),
        ("the two replicas that get an interval", IMPOSSIBLE, 2),
    ] {
        let compiled = CompiledModel::compile(&exponential_model(rate)).unwrap();
        let estimates = run(&compiled, &config(nb_runs, 31, &instants, 0.95)).unwrap();
        let indicator = &estimates.indicators[0];
        for (name, ci) in [
            ("ci", &indicator.ci),
            ("sojourn_ci", &indicator.sojourn_ci),
            ("nb_occurrences_ci", &indicator.nb_occurrences_ci),
        ] {
            for (k, instant) in instants.iter().enumerate() {
                assert!(
                    ci.high[k] > ci.low[k],
                    "{label}: {name} closed on the point {} at t = {instant} \
                     ({nb_runs} replicas)",
                    ci.low[k]
                );
            }
        }
    }
}

#[test]
fn the_sojourn_bound_is_the_time_the_campaign_leaves_open() {
    // A sojourn is a time, so what it charges a departure is a time:
    // what the replicas in ε could have spent in the state by t, which
    // is t. Charging "one unit" instead would answer differently on a
    // model written in hours and on the same model written in seconds,
    // which is not a property a bound may have. An occurrence count, a
    // pure number, does not scale with the horizon at all.
    let compiled = CompiledModel::compile(&exponential_model(IMPOSSIBLE)).unwrap();
    let instants = [1.0, 10.0, 100.0];
    let (n, level) = (500_u64, 0.95);
    let estimates = run(&compiled, &config(n, 41, &instants, level)).unwrap();
    let indicator = &estimates.indicators[0];
    let epsilon = unobserved_frequency_bound(n, level);

    for (k, instant) in instants.iter().enumerate() {
        assert_eq!(indicator.sojourn_mean[k], 0.0);
        assert_eq!(
            indicator.sojourn_ci.low[k], 0.0,
            "a cumulated sojourn is never negative"
        );
        assert!(
            (indicator.sojourn_ci.high[k] - epsilon * instant).abs() < 1e-15,
            "sojourn bound {} at t = {instant}, closed form {}",
            indicator.sojourn_ci.high[k],
            epsilon * instant
        );
        assert!((indicator.nb_occurrences_ci.high[k] - epsilon).abs() < 1e-18);
    }
}

#[test]
fn the_sojourn_over_an_empty_interval_is_zero_and_says_so() {
    // The one place a width of zero survives, and the one place it
    // claims nothing: at the origin the cumulated sojourn is the
    // integral over an empty interval, so it is zero for every
    // trajectory of every model, and the elapsed time it would charge a
    // departure is zero too. One instant later the bound is back.
    let compiled = CompiledModel::compile(&exponential_model(IMPOSSIBLE)).unwrap();
    let (n, level) = (500_u64, 0.95);
    let estimates = run(&compiled, &config(n, 43, &[0.0, 1.0], level)).unwrap();
    let sojourn = &estimates.indicators[0].sojourn_ci;

    assert!(sojourn.constant_sample[0]);
    assert_eq!((sojourn.low[0], sojourn.high[0]), (0.0, 0.0));
    assert!((sojourn.high[1] - unobserved_frequency_bound(n, level)).abs() < 1e-15);
}

#[test]
fn a_campaign_that_moved_reports_exactly_what_it_reported_before() {
    // The rule reaches the degenerate case and nothing else. Wherever
    // the sample has a dispersion, the bounds are the central-limit
    // ones term by term, to the last bit, and nothing is marked: the
    // coverage measured above is measured on those, and it may not move
    // because a degenerate end was repaired.
    let compiled = CompiledModel::compile(&exponential_model(0.3)).unwrap();
    let instants = [0.5, 1.0, 2.0, 4.0];
    let (n, level) = (400_u64, 0.95);
    let estimates = run(&compiled, &config(n, 11, &instants, level)).unwrap();
    let indicator = &estimates.indicators[0];

    for (name, ci, mean, std) in [
        (
            "sojourn_ci",
            &indicator.sojourn_ci,
            &indicator.sojourn_mean,
            &indicator.sojourn_std,
        ),
        (
            "nb_occurrences_ci",
            &indicator.nb_occurrences_ci,
            &indicator.nb_occurrences_mean,
            &indicator.nb_occurrences_std,
        ),
    ] {
        for k in 0..instants.len() {
            assert!(
                std[k] > 0.0,
                "{name}: the probe must disperse at instant {k}"
            );
            assert!(
                !ci.constant_sample[k],
                "{name}: nothing to mark at instant {k}"
            );
            assert_eq!(
                (ci.low[k], ci.high[k]),
                normal_bounds(mean[k], std[k], n, level),
                "{name} at instant {k}"
            );
        }
    }
    assert!(!indicator.ci.constant_sample.iter().any(|&flat| flat));
}

// ---------------------------------------------------------------------
// Cost
// ---------------------------------------------------------------------

#[test]
fn the_intervals_cost_nothing_beside_the_campaign() {
    // The intervals are closed forms over sums the reduction already
    // holds: their cost is O(indicators × instants) and does not grow
    // with the replica count, while the campaign is O(replicas ×
    // trajectory). The measurement below is a ratio, not a threshold on
    // an absolute duration, and the assertion leaves two orders of
    // magnitude of margin over the observed share, so it reports a
    // regression rather than machine load.
    let instants: Vec<f64> = (1..=20).map(f64::from).collect();
    let compiled = CompiledModel::compile(&exponential_model(0.05)).unwrap();
    let nb_runs = 2_000;

    let start = Instant::now();
    let estimates = run(&compiled, &config(nb_runs, 13, &instants, 0.95)).unwrap();
    let campaign = start.elapsed().as_secs_f64();

    // Replay exactly the three series the reduction builds per
    // indicator, on the series it built, allocation included.
    let indicator = &estimates.indicators[0];
    let repeats = 2_000;
    let start = Instant::now();
    let mut sink = 0.0;
    for _ in 0..repeats {
        let value = ConfidenceInterval::on_proportion(0.95, nb_runs, black_box(&indicator.mean));
        let sojourn = ConfidenceInterval::on_mean(
            0.95,
            nb_runs,
            black_box(&indicator.sojourn_mean),
            black_box(&indicator.sojourn_std),
            Departure::sojourn(black_box(&indicator.instants)),
        );
        let occurrences = ConfidenceInterval::on_mean(
            0.95,
            nb_runs,
            black_box(&indicator.nb_occurrences_mean),
            black_box(&indicator.nb_occurrences_std),
            Departure::count(),
        );
        sink += value.high[0] + sojourn.high[0] + occurrences.high[0];
    }
    let intervals =
        start.elapsed().as_secs_f64() / f64::from(repeats) * estimates.indicators.len() as f64;
    assert!(sink.is_finite());

    let share = intervals / campaign;
    assert!(
        share < 0.01,
        "interval arithmetic took {intervals:.3e} s beside a {campaign:.3e} s campaign \
         ({:.4} % of it), expected far below 1 %",
        100.0 * share
    );
}

#[test]
fn the_interval_helpers_are_usable_on_their_own() {
    // The reduction builds a whole series at once; the same closed forms
    // are reachable for a single figure, which is what a report writer
    // reaches for.
    let ci = ConfidenceInterval::on_proportion(0.95, 500, &[0.0483]);
    let (low, high) = wilson_bounds(0.0483, 500, 0.95);
    assert_eq!(ci.low, vec![low]);
    assert_eq!(ci.high, vec![high]);
    assert_eq!(ci.method, IntervalMethod::Wilson);
    assert_eq!(ci.level, 0.95);
}

// ---------------------------------------------------------------------
// Weighted indicator (likelihood-ratio weighted hits)
// ---------------------------------------------------------------------

/// Sample mean and standard deviation (ddof = 1), written out from
/// their definition, independently of the crate.
fn mean_and_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

#[test]
fn unit_weights_give_the_normal_interval_of_the_plain_proportion_under_another_name() {
    // 37 hits among 500 replicas, every weight 1: the weighted indicator
    // is the 0/1 indicator itself.
    let values: Vec<f64> = (0..500)
        .map(|i| if i % 13 == 0 { 1.0 } else { 0.0 })
        .collect();
    let hits = values.iter().filter(|&&v| v > 0.0).count();
    let (p, std) = mean_and_std(&values);
    let (low, high) = normal_bounds(p, std, 500, 0.95);

    let w: WeightedInterval = weighted_interval(&values, 0.95);
    assert_eq!(w.method, IntervalMethod::WeightedNormal);
    assert_ne!(w.method, IntervalMethod::Normal);
    assert_ne!(w.method, IntervalMethod::Wilson);
    assert_eq!(w.level, 0.95);
    assert_eq!(w.replicas, 500);
    assert!((w.estimate - hits as f64 / 500.0).abs() < 1e-15);
    assert!((w.low - low).abs() < 1e-15, "{} vs {low}", w.low);
    assert!((w.high - high).abs() < 1e-15, "{} vs {high}", w.high);
    // Every hit weighs the same: the effective sample size is the hit
    // count.
    assert!((w.effective_sample_size - hits as f64).abs() < 1e-9);
    let se = std / 500f64.sqrt();
    assert!((w.standard_error - se).abs() < 1e-15);
    assert!((w.relative_error.unwrap() - se / p).abs() < 1e-12);
}

#[test]
fn one_small_hit_among_a_thousand_gives_its_weight_over_n_and_an_ess_of_one() {
    let mut values = vec![0.0; 1000];
    values[417] = 1e-6;
    let w = weighted_interval(&values, 0.95);
    assert_eq!(w.method, IntervalMethod::WeightedNormal);
    assert!((w.estimate - 1e-9).abs() < 1e-24, "{}", w.estimate);
    assert_eq!(w.effective_sample_size, 1.0);
    // Not clamped: the lower bound falls below zero, which is the
    // interval saying the normal approximation is out of its range.
    assert!(w.low < 0.0);
    assert!(w.high > 1e-9);
    // The relative error, computed from its definition (standard error
    // over the estimate) rather than from a derived closed form.
    let (mean, std) = mean_and_std(&values);
    let rel = std / 1000f64.sqrt() / mean;
    assert!((w.relative_error.unwrap() - rel).abs() < 1e-12);
}

#[test]
fn unequal_weights_lower_the_effective_sample_size() {
    // Hits weighing 1, 1, 1 and 10: (13)² / (3 + 100) = 169/103.
    let mut values = vec![0.0; 100];
    values[3] = 1.0;
    values[20] = 1.0;
    values[55] = 1.0;
    values[90] = 10.0;
    let w = weighted_interval(&values, 0.9);
    assert!((w.effective_sample_size - 169.0 / 103.0).abs() < 1e-12);
    assert_eq!(w.level, 0.9);
}

#[test]
fn no_hit_gives_an_undefined_interval_an_ess_of_zero_and_no_relative_error() {
    let values = vec![0.0; 1000];
    let w = weighted_interval(&values, 0.95);
    assert_eq!(w.method, IntervalMethod::Undefined);
    assert_eq!(w.estimate, 0.0);
    assert_eq!((w.low, w.high), (0.0, 0.0));
    assert_eq!(w.effective_sample_size, 0.0);
    assert_eq!(w.relative_error, None);
    assert_eq!(w.replicas, 1000);
}

#[test]
fn fewer_than_two_replicas_give_an_undefined_interval() {
    for values in [vec![], vec![0.5]] {
        let w = weighted_interval(&values, 0.95);
        assert_eq!(w.method, IntervalMethod::Undefined, "{values:?}");
        assert_eq!(w.low, w.estimate);
        assert_eq!(w.high, w.estimate);
        assert_eq!(w.relative_error, None);
    }
    assert_eq!(weighted_interval(&[0.5], 0.95).estimate, 0.5);
    assert_eq!(weighted_interval(&[], 0.95).estimate, 0.0);
}

#[test]
fn the_weighted_method_serialises_under_its_own_name() {
    assert_eq!(
        serde_json::to_value(IntervalMethod::WeightedNormal).unwrap(),
        "weighted_normal"
    );
    let mut values = vec![0.0; 10];
    values[2] = 0.25;
    let json = serde_json::to_value(weighted_interval(&values, 0.95)).unwrap();
    assert_eq!(json["method"], "weighted_normal");
    assert!(json["effective_sample_size"].is_number());
    assert!(json["relative_error"].is_number());
    let none = serde_json::to_value(weighted_interval(&[0.0; 10], 0.95)).unwrap();
    assert!(none["relative_error"].is_null());
}
