//! The **zero-departure** pair: `zero_departures` and `nonzero_reached`.
//!
//! The reference engine's `nb_visits` and `realized` computations, measured
//! on PyCATSHOO 1.3.8.0 (2026-10-02) on a float attribute stepping through
//! five levels, one per time unit, sampled at 0.5, 1.5, 2.5, 3.5, 4.5 and 6:
//!
//! - `nb_visits` counts the moves from exactly 0 to any non-zero value,
//!   sign ignored, a non-zero initial value not counted;
//! - `realized` is 1 once the value has been non-zero, the initial value
//!   included.
//!
//! Every expected figure below is the reference engine's own output on
//! these level sequences. The existing `nb_occurrences` / `reached` pair
//! (rising edges from `<= 0` to `> 0`, an active initial value included)
//! is asserted alongside, unchanged.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, FlowConfig};
use raichu_model::Model;
use raichu_montecarlo::{run, IndicatorEstimate, McConfig, DEFAULT_CONFIDENCE};

const INSTANTS: [f64; 6] = [0.5, 1.5, 2.5, 3.5, 4.5, 6.0];

/// A float attribute `v` holding `levels[i]` on `[i, i + 1)`, the last level
/// held to the horizon: an automaton steps through one state per level on
/// unit delays, and one sensitive function writes the level of the active
/// state.
fn stepping(levels: &[f64]) -> Model {
    let states: Vec<String> = (0..levels.len()).map(|i| format!("\"s{i}\"")).collect();
    let transitions: Vec<String> = (0..levels.len() - 1)
        .map(|i| {
            format!(
                r#"{{"name": "t{i}", "source": "s{i}", "targets": ["s{}"], "distrib": "delay", "time": 1.0}}"#,
                i + 1
            )
        })
        .collect();
    let float =
        |x: f64| format!(r#"{{"op": "const", "value": {{"kind": "float", "value": {x:?}}}}}"#);
    let mut level = float(*levels.last().unwrap());
    for (i, x) in levels.iter().enumerate().rev().skip(1) {
        level = format!(
            r#"{{"op": "if", "cond": {{"op": "state_active", "state": {{"component": "c", "automaton": "aut", "state": "s{i}"}}}}, "then": {}, "otherwise": {level}}}"#,
            float(*x)
        );
    }
    Model::from_json(&format!(
        r#"{{
  "name": "stepping",
  "components": [{{
    "name": "c",
    "attributes": [{{"name": "v", "kind": "float", "init": {{"kind": "float", "value": {init:?}}}}}],
    "ports": [],
    "automata": [{{"name": "aut", "states": [{states}], "init": "s0", "transitions": [{transitions}]}}],
    "sensitive_functions": [{{"name": "upd", "effects": [
      {{"target": {{"component": "c", "attribute": "v"}}, "value": {level}}}]}}]}}],
  "indicators": [{{"name": "v", "target": "attribute", "attr": {{"component": "c", "attribute": "v"}}}}]
}}"#,
        init = levels[0],
        states = states.join(", "),
        transitions = transitions.join(", "),
    ))
    .unwrap()
}

fn estimate(model: &Model, samples: &[f64]) -> IndicatorEstimate {
    let compiled = CompiledModel::compile(model).unwrap();
    run(
        &compiled,
        &McConfig {
            nb_runs: 1,
            seed: 7,
            t_max: *samples.last().unwrap(),
            samples: samples.to_vec(),
            threads: Some(1),
            quantiles: vec![],
            confidence: DEFAULT_CONFIDENCE,
            ode: Default::default(),
            stop_at_targets: false,
            flow: FlowConfig::default(),
        },
    )
    .unwrap()
    .indicators
    .remove(0)
}

/// One measured case: the level sequence, then the expected
/// `zero_departures`, `nonzero_reached`, `nb_occurrences` and `reached`
/// series at [`INSTANTS`].
struct Case {
    levels: [f64; 5],
    zero_departures: [f64; 6],
    nonzero_reached: [f64; 6],
    nb_occurrences: [f64; 6],
    reached: [f64; 6],
}

fn check(case: &Case) {
    let est = estimate(&stepping(&case.levels), &INSTANTS);
    let levels = case.levels;
    assert_eq!(
        est.zero_departures_mean, case.zero_departures,
        "zero_departures on {levels:?}"
    );
    assert_eq!(
        est.nonzero_reached_mean, case.nonzero_reached,
        "nonzero_reached on {levels:?}"
    );
    assert_eq!(
        est.nb_occurrences_mean, case.nb_occurrences,
        "nb_occurrences on {levels:?}"
    );
    assert_eq!(est.reached_mean, case.reached, "reached on {levels:?}");
    // One replica: the extremes are the draw itself.
    assert_eq!(est.zero_departures_extremes.min, case.zero_departures);
    assert_eq!(est.zero_departures_extremes.max, case.zero_departures);
    assert_eq!(est.nonzero_reached_extremes.max, case.nonzero_reached);
}

#[test]
fn a_negative_level_departs_from_zero_and_a_nonzero_to_nonzero_move_does_not() {
    // Measured: res_time [0, 1, 0.5, -0.75, -0.5, -0.5],
    // nb_visits [0, 1, 1, 1, 1, 1], realized [0, 1, 1, 1, 1, 1].
    let case = Case {
        levels: [0.0, 2.0, -3.0, 0.5, 0.0],
        zero_departures: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        nonzero_reached: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        // -3 -> 0.5 is a rising edge, 2 -> -3 is not.
        nb_occurrences: [0.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        reached: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    };
    check(&case);
    // The signed time-integral is the reference engine's `res_time`.
    let est = estimate(&stepping(&case.levels), &INSTANTS);
    assert_eq!(est.sojourn_mean, [0.0, 1.0, 0.5, -0.75, -0.5, -0.5]);
}

#[test]
fn a_departure_to_a_negative_value_counts() {
    // Measured: nb_visits [0, 1, 1, 2, 2, 2], realized [0, 1, 1, 1, 1, 1].
    check(&Case {
        levels: [0.0, -3.0, 0.0, 2.0, 0.0],
        zero_departures: [0.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        nonzero_reached: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        // Never above zero before t = 3.
        nb_occurrences: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        reached: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    });
}

#[test]
fn a_return_to_zero_then_a_new_departure_counts_twice() {
    // Measured: nb_visits [0, 1, 1, 1, 2, 2].
    check(&Case {
        levels: [0.0, 2.0, 5.0, 0.0, 1.0],
        zero_departures: [0.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        nonzero_reached: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        nb_occurrences: [0.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        reached: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    });
}

#[test]
fn a_nonzero_initial_value_is_reached_but_not_a_departure() {
    // Measured: nb_visits [0, 0, 1, 1, 1, 1], realized [1, 1, 1, 1, 1, 1].
    check(&Case {
        levels: [1.0, 0.0, -1.0, 1.0, 0.0],
        zero_departures: [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        nonzero_reached: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        // The active initial value is the first occurrence; -1 -> 1 the second.
        nb_occurrences: [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        reached: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    });
}

#[test]
fn an_initially_active_state_is_one_occurrence_and_no_departure() {
    // Initially `on`, left at 1, entered again at 2: the occurrence count
    // takes the initial value as its first entry, the departure count does
    // not.
    let model = Model::from_json(
        r#"{
  "name": "initially_on",
  "components": [{
    "name": "c", "attributes": [], "ports": [],
    "automata": [{
      "name": "aut", "states": ["off", "on"], "init": "on",
      "transitions": [
        {"name": "enter", "source": "off", "targets": ["on"], "distrib": "delay", "time": 1.0},
        {"name": "leave", "source": "on", "targets": ["off"], "distrib": "delay", "time": 1.0}]}]}],
  "indicators": [
    {"name": "c_on", "target": "state", "component": "c", "automaton": "aut", "state": "on"}]
}"#,
    )
    .unwrap();
    let est = estimate(&model, &[0.5, 1.5, 2.5]);
    assert_eq!(est.mean, [1.0, 0.0, 1.0]);
    assert_eq!(est.nb_occurrences_mean, [1.0, 1.0, 2.0]);
    assert_eq!(est.zero_departures_mean, [0.0, 0.0, 1.0]);
    assert_eq!(est.reached_mean, [1.0, 1.0, 1.0]);
    assert_eq!(est.nonzero_reached_mean, [1.0, 1.0, 1.0]);
}

#[test]
fn the_pair_is_bit_identical_whatever_the_thread_count_and_carries_intervals() {
    let model = Model::from_json(
        r#"{
  "name": "on_off",
  "components": [{
    "name": "c", "attributes": [], "ports": [],
    "automata": [{
      "name": "aut", "states": ["off", "on"], "init": "off",
      "transitions": [
        {"name": "enter", "source": "off", "targets": ["on"], "distrib": "exp", "rate": 0.5},
        {"name": "leave", "source": "on", "targets": ["off"], "distrib": "exp", "rate": 4.0}]}]}],
  "indicators": [
    {"name": "c_on", "target": "state", "component": "c", "automaton": "aut", "state": "on"}]
}"#,
    )
    .unwrap();
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = |threads| McConfig {
        nb_runs: 2_000,
        seed: 11,
        t_max: 4.0,
        samples: vec![1.0, 2.0, 4.0],
        threads: Some(threads),
        quantiles: vec![],
        confidence: DEFAULT_CONFIDENCE,
        ode: Default::default(),
        stop_at_targets: false,
        flow: FlowConfig::default(),
    };
    let one = run(&compiled, &config(1)).unwrap().indicators.remove(0);
    let many = run(&compiled, &config(8)).unwrap().indicators.remove(0);
    assert_eq!(one, many);
    // Initially off, so on a state indicator the two pairs coincide.
    assert_eq!(one.zero_departures_mean, one.nb_occurrences_mean);
    assert_eq!(one.nonzero_reached_mean, one.reached_mean);
    assert_eq!(one.nonzero_reached_ci, one.reached_ci);
    assert_eq!(one.zero_departures_ci, one.nb_occurrences_ci);
    for k in 0..3 {
        let p = one.nonzero_reached_mean[k];
        assert!(one.nonzero_reached_ci.low[k] <= p && p <= one.nonzero_reached_ci.high[k]);
    }
}

#[test]
fn a_result_written_before_the_pair_existed_still_reads() {
    let est = estimate(&stepping(&[0.0, 2.0, -3.0, 0.5, 0.0]), &INSTANTS);
    let mut document = serde_json::to_value(&est).unwrap();
    let object = document.as_object_mut().unwrap();
    for field in [
        "zero_departures_mean",
        "zero_departures_std",
        "zero_departures_ci",
        "zero_departures_extremes",
        "nonzero_reached_mean",
        "nonzero_reached_std",
        "nonzero_reached_ci",
        "nonzero_reached_extremes",
    ] {
        assert!(object.remove(field).is_some(), "{field} is serialized");
    }
    let read: IndicatorEstimate = serde_json::from_value(document).unwrap();
    assert!(read.zero_departures_mean.is_empty());
    assert!(read.nonzero_reached_ci.low.is_empty());
    assert_eq!(read.nb_occurrences_mean, est.nb_occurrences_mean);
}
