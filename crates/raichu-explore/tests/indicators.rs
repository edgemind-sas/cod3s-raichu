//! Indicators of a discretised exploration: the mean value of a state
//! indicator at declared instants and its sojourn time, each with a lower
//! and an upper bound, and the mean time to the target truncated at the
//! horizon.
//!
//! Closed forms first (a non-repairable pair, a Weibull unit, a repairable
//! unit frozen when the target is reached), then the bounds under
//! cut-offs, a Monte-Carlo witness, determinism across thread counts, the
//! format version and the refusals.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError};
use raichu_explore::{
    explore_discretised, read_exploration, BoundedValue, DiscretisedSettings, ExplorationResult,
    ExploredIndicators, IndicatorRequest, ReadExplorationError,
};
use raichu_model::Model;
use rayon::prelude::*;
use serde_json::{json, Value as Json};

// ---- model builders -------------------------------------------------------

fn active(component: &str, automaton: &str, state: &str) -> Json {
    json!({"op": "state_active",
           "state": {"component": component, "automaton": automaton, "state": state}})
}

fn down(component: &str) -> Json {
    active(component, "fail", "nok")
}

fn and(args: Vec<Json>) -> Json {
    json!({"op": "bool", "bool_op": "and", "args": args})
}

fn tr(name: &str, source: &str, targets: &[&str], law: Json) -> Json {
    let mut t = json!({"name": name, "source": source, "targets": targets});
    for (k, v) in law.as_object().unwrap() {
        t[k] = v.clone();
    }
    t
}

fn with(mut t: Json, key: &str, value: Json) -> Json {
    t[key] = value;
    t
}

fn automaton(name: &str, states: &[&str], transitions: Vec<Json>) -> Json {
    json!({"name": name, "states": states, "init": states[0], "transitions": transitions})
}

fn component(name: &str, automata: Vec<Json>, attributes: Json) -> Json {
    json!({"name": name, "ports": [], "attributes": attributes, "automata": automata,
           "equations": []})
}

fn exp(rate: f64) -> Json {
    json!({"distrib": "exp", "rate": rate})
}

fn weibull(shape: f64, scale: f64) -> Json {
    json!({"distrib": "weibull", "shape": shape, "scale": scale})
}

fn inst() -> Json {
    json!({"distrib": "inst", "probs": []})
}

/// A unit `name.fail` (`ok`, `nok`) failing under `law`, repaired under
/// `repair` when given.
fn unit(name: &str, law: Json, repair: Option<Json>) -> Json {
    let mut transitions = vec![with(
        with(tr("occ", "ok", &["nok"], law), "monitored", json!(true)),
        "kind",
        json!("failure"),
    )];
    if let Some(repair) = repair {
        transitions.push(tr("rep", "nok", &["ok"], repair));
    }
    component(
        name,
        vec![automaton("fail", &["ok", "nok"], transitions)],
        json!([]),
    )
}

/// The feared-event watcher `sys.watch`, entering `down` as soon as
/// `guard` holds.
fn watcher(guard: Json) -> Json {
    component(
        "sys",
        vec![automaton(
            "watch",
            &["ok", "down"],
            vec![with(tr("down", "ok", &["down"], inst()), "guard", guard)],
        )],
        json!([]),
    )
}

fn state_indicator(name: &str, component: &str) -> Json {
    json!({"name": name, "target": "state", "component": component, "automaton": "fail",
           "state": "nok"})
}

fn build(name: &str, components: Vec<Json>, targets: Json, indicators: Vec<Json>) -> CompiledModel {
    let document = json!({"name": name, "components": components, "targets": targets,
                          "indicators": indicators});
    let model = Model::from_json(&document.to_string()).unwrap();
    CompiledModel::compile(&model).unwrap()
}

fn sys_down() -> Json {
    json!([{"name": "sys_down", "component": "sys", "automaton": "watch", "state": "down"}])
}

const A: f64 = 0.01;
const B: f64 = 0.02;
const HORIZON: f64 = 100.0;
const INSTANTS: [f64; 3] = [10.0, 50.0, 100.0];

/// A (rate `A`) and B (rate `B`), non repairable, target "both failed",
/// one state indicator per unit.
fn pair() -> CompiledModel {
    build(
        "pair",
        vec![
            unit("A", exp(A), None),
            unit("B", exp(B), None),
            watcher(and(vec![down("A"), down("B")])),
        ],
        sys_down(),
        vec![state_indicator("A_ko", "A"), state_indicator("B_ko", "B")],
    )
}

/// A fails under a Weibull law, B under an exponential one; target "both
/// failed". Outside the exact domain.
fn weibull_pair() -> CompiledModel {
    build(
        "weibull_pair",
        vec![
            unit("A", weibull(2.0, 60.0), None),
            unit("B", exp(B), None),
            watcher(and(vec![down("A"), down("B")])),
        ],
        sys_down(),
        vec![state_indicator("A_ko", "A")],
    )
}

const LAMBDA: f64 = 0.05;
const MU: f64 = 0.5;

/// A repairable (`LAMBDA`, `MU`), B non repairable (rate `B`); the target
/// is B's failure, so the process stops, A frozen, when B fails.
fn repairable_until_b() -> CompiledModel {
    build(
        "repairable_until_b",
        vec![
            unit("A", exp(LAMBDA), Some(exp(MU))),
            unit("B", exp(B), None),
        ],
        json!([{"name": "b_down", "component": "B", "automaton": "fail", "state": "nok"}]),
        vec![state_indicator("A_ko", "A")],
    )
}

// ---- helpers ----------------------------------------------------------------

fn settings(
    target: &str,
    horizon: f64,
    level: u32,
    names: &[&str],
    instants: &[f64],
) -> DiscretisedSettings {
    let mut s = DiscretisedSettings::new(target, horizon);
    s.level = level;
    s.indicators = Some(IndicatorRequest {
        names: names.iter().map(|&n| n.to_owned()).collect(),
        instants: instants.to_vec(),
    });
    s
}

fn indicators(result: &ExplorationResult) -> &ExploredIndicators {
    result
        .indicators
        .as_ref()
        .expect("indicators were requested")
}

fn mid(value: &BoundedValue) -> f64 {
    0.5 * (value.lower + value.upper)
}

/// Composite Simpson rule of `f` over `[0, t]`.
fn simpson(f: impl Fn(f64) -> f64, t: f64) -> f64 {
    let n = 2000;
    let h = t / f64::from(n);
    let mut sum = f(0.0) + f(t);
    for i in 1..n {
        sum += f(f64::from(i) * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{what}: {actual} vs {expected} (tolerance {tol})"
    );
}

fn assert_within(value: &BoundedValue, expected: f64, slack: f64, what: &str) {
    assert!(
        value.lower <= value.upper,
        "{what}: lower {} above upper {}",
        value.lower,
        value.upper
    );
    assert!(
        value.lower - slack <= expected && expected <= value.upper + slack,
        "{what}: {expected} outside [{}, {}] (slack {slack})",
        value.lower,
        value.upper
    );
}

/// Unavailability of A alone at `s`: `LAMBDA / (LAMBDA + MU) (1 - exp(-(LAMBDA + MU) s))`.
fn unavailability(s: f64) -> f64 {
    LAMBDA / (LAMBDA + MU) * -(-(LAMBDA + MU) * s).exp_m1()
}

/// `P(A down at min(t, T_B))`: A's unavailability at `t` when B has not
/// failed, at B's failure date otherwise.
fn frozen_unavailability(t: f64) -> f64 {
    (-B * t).exp() * unavailability(t) + simpson(|s| B * (-B * s).exp() * unavailability(s), t)
}

// ---- closed forms -----------------------------------------------------------

#[test]
fn the_pair_indicators_converge_to_their_closed_forms() {
    let result = explore_discretised(
        &pair(),
        &settings("sys_down", HORIZON, 32, &["A_ko", "B_ko"], &INSTANTS),
    )
    .unwrap();
    let found = indicators(&result);
    assert_eq!(found.instants, INSTANTS);
    assert_eq!(found.values.len(), 2);
    for (indicator, rate) in found.values.iter().zip([A, B]) {
        assert!(
            !indicator.inconclusive,
            "{}: no cut-off fired",
            indicator.name
        );
        for (k, &t) in INSTANTS.iter().enumerate() {
            let mean = -(-rate * t).exp_m1();
            let sojourn = t - mean / rate;
            let what = format!("{} at {t}", indicator.name);
            // No cut-off fired: the two bounds coincide.
            assert_close(
                indicator.mean[k].lower,
                indicator.mean[k].upper,
                1e-12,
                &what,
            );
            assert_close(
                indicator.sojourn[k].lower,
                indicator.sojourn[k].upper,
                1e-9,
                &what,
            );
            assert_close(
                mid(&indicator.mean[k]),
                mean,
                0.02,
                &format!("mean of {what}"),
            );
            assert_close(
                mid(&indicator.sojourn[k]),
                sojourn,
                0.03 * sojourn.max(1.0),
                &format!("sojourn of {what}"),
            );
            assert!(indicator.mean[k].error_estimate.is_some_and(f64::is_finite));
            assert!(indicator.sojourn[k]
                .error_estimate
                .is_some_and(f64::is_finite));
        }
    }
    assert_eq!(found.values[0].name, "A_ko");
    assert_eq!(found.values[1].name, "B_ko");
}

#[test]
fn the_sojourn_time_counts_the_time_before_another_component_moves() {
    // A's sojourn in `nok` runs from A's failure, whatever B does
    // afterwards: counting only the time since the last firing of the
    // sequence would give E[t - max(T_A, T_B)] given A failed, about 25.2
    // here, instead of 36.8.
    let result = explore_discretised(
        &pair(),
        &settings("sys_down", HORIZON, 64, &["A_ko"], &[HORIZON]),
    )
    .unwrap();
    let sojourn = mid(&indicators(&result).values[0].sojourn[0]);
    let expected = HORIZON - -(-A * HORIZON).exp_m1() / A;
    assert_close(sojourn, expected, 0.5, "sojourn of A_ko at the horizon");
    assert!((sojourn - 25.2).abs() > 5.0);
}

#[test]
fn the_mean_time_to_the_target_is_truncated_at_the_horizon() {
    let result =
        explore_discretised(&pair(), &settings("sys_down", HORIZON, 32, &[], &[HORIZON])).unwrap();
    let found = indicators(&result);
    assert!(found.values.is_empty());
    let expected = simpson(
        |t| 1.0 - (-(-A * t).exp_m1()) * (-(-B * t).exp_m1()),
        HORIZON,
    );
    assert_close(
        found.time_to_target.lower,
        found.time_to_target.upper,
        1e-9,
        "time to target bounds",
    );
    assert_close(
        mid(&found.time_to_target),
        expected,
        0.01 * expected,
        "time to target",
    );
}

#[test]
fn a_weibull_unit_matches_its_distribution_function() {
    let result = explore_discretised(
        &weibull_pair(),
        &settings("sys_down", HORIZON, 32, &["A_ko"], &INSTANTS),
    )
    .unwrap();
    let cdf = |t: f64| -(-(t / 60.0).powi(2)).exp_m1();
    let indicator = &indicators(&result).values[0];
    for (k, &t) in INSTANTS.iter().enumerate() {
        assert_close(
            mid(&indicator.mean[k]),
            cdf(t),
            0.02,
            &format!("mean at {t}"),
        );
        let sojourn = simpson(cdf, t);
        assert_close(
            mid(&indicator.sojourn[k]),
            sojourn,
            0.03 * sojourn.max(1.0),
            &format!("sojourn at {t}"),
        );
    }
}

#[test]
fn the_error_shrinks_when_the_level_grows() {
    let error = |level: u32| {
        let mut s = settings("sys_down", HORIZON, level, &["A_ko"], &INSTANTS);
        s.refine = false;
        let result = explore_discretised(&pair(), &s).unwrap();
        let indicator = &indicators(&result).values[0];
        assert!(indicator.mean[0].error_estimate.is_none(), "no refinement");
        INSTANTS
            .iter()
            .enumerate()
            .map(|(k, &t)| (mid(&indicator.mean[k]) + (-A * t).exp_m1()).abs())
            .fold(0.0, f64::max)
    };
    let (coarse, fine) = (error(4), error(64));
    assert!(
        fine < 0.25 * coarse,
        "error {coarse} at level 4, {fine} at level 64"
    );
}

// ---- stopped at the target, bounds under cut-offs -----------------------------

/// The repairable model explored to 10 with at most three firings per
/// sequence, refined from level 8 to 16. A repair is fast against the
/// window, so the mean at an instant carries a discretisation error of a
/// few hundredths at these levels: the checks allow twice the refinement
/// estimate, as the probability checks of the discretised driver do.
fn repairable_settings(instants: &[f64]) -> DiscretisedSettings {
    let mut s = settings("b_down", 10.0, 8, &["A_ko"], instants);
    s.cutoffs.max_length = Some(3);
    s
}

/// Twice the refinement estimate of `value`, plus a floor.
fn slack(value: &BoundedValue, floor: f64) -> f64 {
    2.0 * value.error_estimate.expect("a refined result") + floor
}

#[test]
fn a_repairable_unit_is_frozen_when_the_target_is_reached() {
    let instants = [2.0, 5.0, 10.0];
    let result =
        explore_discretised(&repairable_until_b(), &repairable_settings(&instants)).unwrap();
    let indicator = &indicators(&result).values[0];
    for (k, &t) in instants.iter().enumerate() {
        let expected = frozen_unavailability(t);
        eprintln!("t={t}: mean {:?} expected {expected}", indicator.mean[k]);
        let what = format!("mean at {t}");
        assert_within(
            &indicator.mean[k],
            expected,
            slack(&indicator.mean[k], 1e-3),
            &what,
        );
        // Not the free process: A alone would be at its own unavailability.
        let sojourn = simpson(frozen_unavailability, t);
        eprintln!(
            "t={t}: sojourn {:?} expected {sojourn}",
            indicator.sojourn[k]
        );
        let what = format!("sojourn at {t}");
        assert_within(
            &indicator.sojourn[k],
            sojourn,
            slack(&indicator.sojourn[k], 1e-3),
            &what,
        );
        assert!(indicator.mean[k].upper <= 1.0 + 1e-12);
        assert!(indicator.sojourn[k].upper <= t + 1e-9);
    }
}

#[test]
fn a_cut_off_widens_the_bounds_by_the_mass_it_discarded() {
    // One firing at most: every node after the first failure is pruned.
    let mut s = settings("sys_down", HORIZON, 16, &["A_ko", "B_ko"], &INSTANTS);
    s.cutoffs.max_length = Some(1);
    let result = explore_discretised(&pair(), &s).unwrap();
    assert!(result.cutoff_tallies.max_length.fired());
    let found = indicators(&result);
    for (indicator, rate) in found.values.iter().zip([A, B]) {
        assert!(
            indicator.inconclusive,
            "{}: the bounds are far apart",
            indicator.name
        );
        for (k, &t) in INSTANTS.iter().enumerate() {
            let mean = -(-rate * t).exp_m1();
            let what = format!("{} at {t}", indicator.name);
            assert_within(&indicator.mean[k], mean, 0.03, &what);
            assert_within(&indicator.sojourn[k], t - mean / rate, 0.03 * t, &what);
            assert!(
                indicator.mean[k].upper - indicator.mean[k].lower > 1e-3,
                "{what}"
            );
        }
        // At the horizon the unknown mass is at most what the cut-off
        // discarded for the target probability.
        let gap = indicator.mean[2].upper - indicator.mean[2].lower;
        assert!(gap <= result.upper - result.lower + 1e-12, "gap {gap}");
    }
    let expected = simpson(
        |t| 1.0 - (-(-A * t).exp_m1()) * (-(-B * t).exp_m1()),
        HORIZON,
    );
    assert_within(&found.time_to_target, expected, 0.5, "time to target");
    assert!(found.time_to_target.upper - found.time_to_target.lower > 1.0);
}

/// The 99 % normal interval of `P(A down at min(t, T_target))` over `n`
/// drawn-date replicas stopped at the target.
fn monte_carlo_down(model: &CompiledModel, t: f64, n: u64) -> (f64, f64) {
    let hits: u64 = (0..n)
        .into_par_iter()
        .map(|replica| {
            let config = EngineConfig {
                t_max: t,
                seed: 20_261_125,
                rng_stream: replica,
                stop_at_targets: true,
                ..EngineConfig::default()
            };
            let mut engine = Engine::new(model, config).unwrap();
            while engine.reached_target().is_none() && engine.step_until(t).unwrap().is_some() {}
            u64::from(engine.state("A.fail") == Some("nok"))
        })
        .sum();
    let p = hits as f64 / n as f64;
    let half = 2.576 * (p * (1.0 - p) / n as f64).sqrt();
    (p - half, p + half)
}

#[test]
fn the_frozen_mean_meets_monte_carlo() {
    let instants = [5.0, 10.0];
    let model = repairable_until_b();
    let result = explore_discretised(&model, &repairable_settings(&instants)).unwrap();
    let indicator = &indicators(&result).values[0];
    for (k, &t) in instants.iter().enumerate() {
        let (low, high) = monte_carlo_down(&model, t, 40_000);
        let value = &indicator.mean[k];
        let slack = slack(value, 0.0);
        eprintln!(
            "t={t}: MC [{low}, {high}], explored [{}, {}] +- {slack}",
            value.lower, value.upper
        );
        assert!(
            low <= value.upper + slack && high >= value.lower - slack,
            "t={t}: MC [{low}, {high}] vs [{}, {}]",
            value.lower,
            value.upper
        );
    }
}

// ---- determinism, format, refusals --------------------------------------------

#[test]
fn indicators_are_identical_with_one_and_four_threads() {
    let run = |threads: usize| {
        let mut s = settings("sys_down", HORIZON, 8, &["A_ko", "B_ko"], &INSTANTS);
        s.threads = Some(threads);
        s.cutoffs.min_probability = Some(1e-3);
        explore_discretised(&pair(), &s).unwrap()
    };
    assert_eq!(run(1), run(4));
}

#[test]
fn a_result_with_indicators_is_written_at_format_version_3() {
    let model = pair();
    let with_indicators = explore_discretised(
        &model,
        &settings("sys_down", HORIZON, 4, &["A_ko"], &INSTANTS),
    )
    .unwrap();
    assert_eq!(with_indicators.version, 3);
    let json = serde_json::to_string(&with_indicators).unwrap();
    assert_eq!(read_exploration(&json).unwrap(), with_indicators);

    // Declared at an earlier version, the document is inconsistent.
    let tampered = json.replacen("\"version\":3", "\"version\":2", 1);
    assert!(matches!(
        read_exploration(&tampered),
        Err(ReadExplorationError::IndicatorsVersion { version: 2 })
    ));

    // Without a request, nothing changes: version 2, no `indicators` key,
    // and the same numbers.
    let mut plain = DiscretisedSettings::new("sys_down", HORIZON);
    plain.level = 4;
    let without = explore_discretised(&model, &plain).unwrap();
    assert_eq!(without.version, 2);
    assert!(without.indicators.is_none());
    let document: Json = serde_json::from_str(&serde_json::to_string(&without).unwrap()).unwrap();
    assert!(document.get("indicators").is_none());
    assert_eq!(without.lower.to_bits(), with_indicators.lower.to_bits());
    assert_eq!(without.sequences, with_indicators.sequences);
}

fn refused(model: &CompiledModel, s: &DiscretisedSettings) -> (String, String) {
    match explore_discretised(model, s) {
        Err(EngineError::InvalidStudyParameter { parameter, detail }) => (parameter, detail),
        other => panic!("expected a refused parameter, got {other:?}"),
    }
}

#[test]
fn invalid_indicator_requests_are_refused_before_exploring() {
    let model = pair();
    let (parameter, detail) = refused(
        &model,
        &settings("sys_down", HORIZON, 4, &["nope"], &INSTANTS),
    );
    assert_eq!(parameter, "indicators");
    assert!(
        detail.contains("nope") && detail.contains("A_ko"),
        "{detail}"
    );

    let (parameter, _) = refused(
        &model,
        &settings("sys_down", HORIZON, 4, &["A_ko", "A_ko"], &INSTANTS),
    );
    assert_eq!(parameter, "indicators");

    for instants in [
        vec![],
        vec![-1.0],
        vec![50.0, 150.0],
        vec![50.0, 10.0],
        vec![f64::NAN],
    ] {
        let (parameter, detail) = refused(
            &model,
            &settings("sys_down", HORIZON, 4, &["A_ko"], &instants),
        );
        assert_eq!(parameter, "instants", "{instants:?}: {detail}");
    }
}

#[test]
fn an_indicator_that_is_not_a_state_is_refused_by_name() {
    let model = build(
        "with_attribute",
        vec![component(
            "A",
            vec![automaton(
                "fail",
                &["ok", "nok"],
                vec![tr("occ", "ok", &["nok"], exp(A))],
            )],
            json!([{"name": "level", "kind": "float",
                        "init": {"kind": "float", "value": 0.0}}]),
        )],
        json!([{"name": "a_down", "component": "A", "automaton": "fail", "state": "nok"}]),
        vec![json!({"name": "level", "target": "attribute",
                    "attr": {"component": "A", "attribute": "level"}})],
    );
    let (parameter, detail) = refused(&model, &settings("a_down", 10.0, 4, &["level"], &[10.0]));
    assert_eq!(parameter, "indicators");
    assert!(
        detail.contains("level") && detail.contains("state"),
        "{detail}"
    );
}
