//! A threshold on a continuous attribute flips at its located date.
//!
//! A threshold indicator is piecewise constant, so its change points are
//! an exact representation of it, provided each one sits at the date the
//! threshold is crossed. They used to sit at the next SAMPLE instant: the
//! time a ramp `x = t` spends above 1.5 read 0.0 by t = 2 on a one-hour
//! grid (0.25 with an extra sample at 1.75), where the truth is 0.5, and
//! the sojourn depended on the grid the study happened to sample on. The
//! reference engine locates the crossing (measured 2026-09-25: a room's
//! hydrogen above 0.85 for 5.64 h over 60 h there, 3.85 h here).
//!
//! Every expected figure is a closed form of the ramp or of the triangle.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::Value;
use raichu_model::Model;

/// `x` integrated at the rate `rate`, which flips sign at `turn` (a
/// triangle when both are given), observed by `x cmp threshold`.
fn ramp(cmp: &str, threshold: f64, turn: Option<f64>) -> Model {
    let rate = match turn {
        None => r#"{"op": "const", "value": {"kind": "float", "value": 1.0}}"#.to_owned(),
        Some(turn) => format!(
            r#"{{"op": "if",
                "cond": {{"op": "cmp", "cmp": "lt", "lhs": {{"op": "time"}},
                         "rhs": {{"op": "const", "value": {{"kind": "float", "value": {turn:?}}}}}}},
                "then": {{"op": "const", "value": {{"kind": "float", "value": 1.0}}}},
                "otherwise": {{"op": "const", "value": {{"kind": "float", "value": -1.0}}}}}}"#
        ),
    };
    Model::from_json(&format!(
        r#"{{
  "name": "ramp",
  "components": [{{
    "name": "c", "ports": [],
    "attributes": [{{"name": "x", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}],
    "equations": [{{"target": "x", "kind": "ode", "expr": {rate}}}]}}],
  "indicators": [{{
    "name": "above", "target": "predicate", "attr": {{"component": "c", "attribute": "x"}},
    "cmp": "{cmp}", "value": {{"kind": "float", "value": {threshold:?}}}}}]
}}"#
    ))
    .unwrap()
}

/// The indicator's change points over a run to `t_max` sampled on `samples`.
fn flips(model: &Model, t_max: f64, samples: Vec<f64>) -> Vec<(f64, bool)> {
    let compiled = CompiledModel::compile(model).unwrap();
    let result = Engine::new(
        &compiled,
        EngineConfig {
            t_max,
            samples,
            ..EngineConfig::default()
        },
    )
    .unwrap()
    .run()
    .unwrap();
    result.indicators[0]
        .points
        .iter()
        .map(|(t, value)| (*t, matches!(value, Value::Bool(true))))
        .collect()
}

fn assert_flips(got: &[(f64, bool)], expected: &[(f64, bool)]) {
    assert_eq!(got.len(), expected.len(), "{got:?} against {expected:?}");
    for ((t, v), (te, ve)) in got.iter().zip(expected) {
        assert_eq!(v, ve, "{got:?}");
        assert!((t - te).abs() < 1e-6, "flip at {t}, expected {te}: {got:?}");
    }
}

#[test]
fn a_rising_threshold_flips_at_its_crossing_whatever_the_grid() {
    let model = ramp("ge", 1.5, None);
    for samples in [vec![1.0, 2.0, 3.0], vec![1.0, 1.75, 2.0, 3.0], vec![]] {
        assert_flips(&flips(&model, 3.0, samples), &[(0.0, false), (1.5, true)]);
    }
}

#[test]
fn a_threshold_entered_and_left_between_two_samples_is_seen_both_ways() {
    // x rises to 1 at t = 1 and falls back: above 0.8 on [0.8, 1.2], strictly
    // between the two samples, which alone would see nothing at all.
    let model = ramp("gt", 0.8, Some(1.0));
    assert_flips(
        &flips(&model, 2.0, vec![0.5, 1.5]),
        &[(0.0, false), (0.8, true), (1.2, false)],
    );
}

#[test]
fn a_falling_comparison_is_located_too() {
    // x <= 0.5 holds from the start and stops holding at 0.5.
    let model = ramp("le", 0.5, None);
    assert_flips(&flips(&model, 1.0, vec![1.0]), &[(0.0, true), (0.5, false)]);
}

/// `x' = sin(x t)` from 0.1, a watched transition at `x >= 0.9`, and one
/// threshold indicator `x cmp 0.3` crossed on the way.
fn watched_with_threshold(cmp: &str) -> Model {
    Model::from_json(&format!(
        r#"{{"name": "w", "components": [{{"name": "c", "ports": [],
  "attributes": [{{"name": "x", "kind": "float", "init": {{"kind": "float", "value": 0.1}}}}],
  "equations": [{{"target": "x", "kind": "ode", "expr": {{"op": "sin", "arg": {{"op": "mul", "args": [
      {{"op": "attr", "attr": {{"component": "c", "attribute": "x"}}}}, {{"op": "time"}}]}}}}}}],
  "automata": [{{"name": "watch", "states": ["run", "done"], "init": "run", "transitions": [
      {{"name": "hit", "source": "run", "targets": ["done"], "distrib": "watched",
        "guard": {{"op": "cmp", "cmp": "ge",
                  "lhs": {{"op": "attr", "attr": {{"component": "c", "attribute": "x"}}}},
                  "rhs": {{"op": "const", "value": {{"kind": "float", "value": 0.9}}}}}}}}]}}]}}],
  "indicators": [{{"name": "s", "target": "predicate", "attr": {{"component": "c", "attribute": "x"}},
                   "cmp": "{cmp}", "value": {{"kind": "float", "value": 0.3}}}}]}}"#
    ))
    .unwrap()
}

#[test]
fn locating_a_threshold_does_not_move_the_trajectory() {
    // An observation reads the trajectory and must not move it. A located
    // `ge` and a sampled `eq` on the same attribute must leave the watched
    // transition at the SAME date, to the bit: the solver takes the same
    // steps whatever is observed.
    let date = |cmp: &str| {
        let compiled = CompiledModel::compile(&watched_with_threshold(cmp)).unwrap();
        let result = Engine::new(
            &compiled,
            EngineConfig {
                t_max: 10.0,
                ..EngineConfig::default()
            },
        )
        .unwrap()
        .run()
        .unwrap();
        result
            .events
            .iter()
            .map(|event| event.time)
            .collect::<Vec<f64>>()
    };
    let (located, sampled) = (date("ge"), date("eq"));
    assert!(!located.is_empty());
    assert_eq!(located, sampled);
}

#[test]
fn a_flip_on_a_sample_instant_keeps_every_sample_once() {
    // The ramp crosses 4.5 exactly on a sample of a 0.1 grid: every sample
    // is recorded once, in order, and the flip sits at 4.5.
    let samples: Vec<f64> = (0..=100).map(|k| f64::from(k) / 10.0).collect();
    let compiled = CompiledModel::compile(&ramp("ge", 4.5, None)).unwrap();
    let result = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 10.0,
            samples: samples.clone(),
            ..EngineConfig::default()
        },
    )
    .unwrap()
    .run()
    .unwrap();
    let recorded: Vec<f64> = result.samples[0].points.iter().map(|(t, _)| *t).collect();
    assert_eq!(recorded, samples);
    let flips: Vec<(f64, bool)> = result.indicators[0]
        .points
        .iter()
        .map(|(t, value)| (*t, matches!(value, Value::Bool(true))))
        .collect();
    assert_flips(&flips, &[(0.0, false), (4.5, true)]);
}
