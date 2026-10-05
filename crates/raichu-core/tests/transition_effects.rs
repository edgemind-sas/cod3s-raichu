//! Edge effects: a transition writes once, when it fires.
//!
//! A sensitive function's effects are a LEVEL, re-evaluated whenever what
//! they read changes. A failure mode on the reference engine can also write
//! on the firing EDGE only (`occ_effects_trans` / `not_occ_effects_trans` in
//! cod3s, a sensitive method on the transition): the value is written once
//! and stays until something else writes it, which is how a detection is
//! made to latch on a gate nothing resets. Every expected figure is a date
//! written in the test.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::Value;
use raichu_model::{Model, ModelError};

/// A rail cracking at 1 and repaired at 3. The crack latches `detected`
/// on its firing edge and the repair clears it; `alarm` follows `detected`
/// through a sensitive function. `extra` is appended to the component.
fn rail(fail_guard: &str, extra: &str) -> String {
    format!(
        r#"{{
  "name": "rail",
  "components": [{{
    "name": "r", "ports": [],
    "attributes": [
      {{"name": "detected", "kind": "bool", "init": {{"kind": "bool", "value": false}}}},
      {{"name": "alarm", "kind": "bool", "init": {{"kind": "bool", "value": false}}}},
      {{"name": "armed", "kind": "bool", "init": {{"kind": "bool", "value": true}}}}],
    "automata": [{{
      "name": "crack", "states": ["ok", "ko"], "init": "ok",
      "transitions": [
        {{"name": "fail", "source": "ok", "targets": ["ko"], {fail_guard}
         "distrib": "delay", "time": 1.0,
         "effects": [{{"target": {{"component": "r", "attribute": "detected"}},
                      "value": {{"op": "const", "value": {{"kind": "bool", "value": true}}}}}}]}},
        {{"name": "repair", "source": "ko", "targets": ["ok"], "distrib": "delay", "time": 2.0,
         "effects": [{{"target": {{"component": "r", "attribute": "detected"}},
                      "value": {{"op": "const", "value": {{"kind": "bool", "value": false}}}}}}]}}]}},
      {{"name": "disarm", "states": ["on", "off"], "init": "on",
       "transitions": [{{"name": "cut", "source": "on", "targets": ["off"], "distrib": "delay", "time": 0.5}}]}}],
    "sensitive_functions": [{{"name": "follow", "effects": [
      {{"target": {{"component": "r", "attribute": "alarm"}},
       "value": {{"op": "attr", "attr": {{"component": "r", "attribute": "detected"}}}}}}]}}]
    {extra}
  }}],
  "indicators": [
    {{"name": "detected", "target": "attribute", "attr": {{"component": "r", "attribute": "detected"}}}},
    {{"name": "alarm", "target": "attribute", "attr": {{"component": "r", "attribute": "alarm"}}}}]
}}"#
    )
}

fn sealed(body: &str) -> String {
    format!(
        r#"{{"raichu_model": {{"format": 1, "requires": ["transition_effects"]}}, "model": {body}}}"#
    )
}

fn series(json: &str, name: &str) -> Vec<(f64, bool)> {
    let model = Model::from_json(json).unwrap();
    let compiled = CompiledModel::compile(&model).unwrap();
    let result = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 5.0,
            ..EngineConfig::default()
        },
    )
    .unwrap()
    .run()
    .unwrap();
    let indicator = result.indicators.iter().find(|s| s.name == name).unwrap();
    indicator
        .points
        .iter()
        .map(|(t, v)| (*t, matches!(v, Value::Bool(true))))
        .collect()
}

#[test]
fn a_detection_latches_on_the_failure_edge_and_clears_on_the_repair_edge() {
    let json = sealed(&rail("", ""));
    // Cracked at 1, repaired at 3, cracked again at 4: the latch is made
    // and cleared by each edge, every time it fires.
    let expected = vec![(0.0, false), (1.0, true), (3.0, false), (4.0, true)];
    assert_eq!(series(&json, "detected"), expected);
    // What reads the latched value follows it, as it would a level.
    assert_eq!(series(&json, "alarm"), expected);
}

#[test]
fn an_interrupted_transition_writes_nothing() {
    // The crack is only armed while `disarm` is `on`, which ends at 0.5:
    // the countdown is dropped before it fires, and nothing is written.
    let guard = r#""guard": {"op": "state_active", "state": {"component": "r", "automaton": "disarm", "state": "on"}},"#;
    let json = sealed(&rail(guard, ""));
    assert_eq!(series(&json, "detected"), vec![(0.0, false)]);
}

#[test]
fn an_edge_write_on_an_attribute_a_function_rewrites_is_refused() {
    // The crack's edge writes `alarm`, which the `follow` function also
    // writes: its next evaluation would erase the one-shot write.
    let json = rail("", "").replacen(
        r#""effects": [{"target": {"component": "r", "attribute": "detected"}"#,
        r#""effects": [{"target": {"component": "r", "attribute": "alarm"}"#,
        1,
    );
    let model: Model = serde_json::from_str(&json).unwrap();
    let outcome = model.validate();
    assert!(
        matches!(outcome, Err(ModelError::TransitionEffectOverwritten { .. })),
        "{outcome:?}"
    );
}

#[test]
fn the_edge_effects_are_a_feature_a_bare_body_cannot_carry() {
    assert!(Model::from_json(&rail("", "")).is_err());
    assert!(Model::from_json(&sealed(&rail("", ""))).is_ok());
}

// ---- reset maps: an edge effect on the target of an ODE ------------------

/// A tank draining at 1 per unit of time from `level = 10`. Its `refill`
/// automaton fires `refill` (law `refill_law`, guard `refill_guard`) with
/// the edge effect `level := value`; a watched `dry` transition records
/// when the level reaches 0. The level is an ODE target: the edge effect
/// is a reset map, and integration restarts from the written value.
fn tank(refill_law: &str, refill_source: &str, value: &str, extra_equation: &str) -> String {
    format!(
        r#"{{"raichu_model": {{"format": 1, "requires": ["transition_effects"]}}, "model": {{
  "name": "tank",
  "components": [{{
    "name": "t", "ports": [],
    "attributes": [
      {{"name": "level", "kind": "float", "init": {{"kind": "float", "value": 10.0}}}},
      {{"name": "shadow", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}],
    "equations": [
      {{"target": "level", "kind": "ode", "expr": {{"op": "const", "value": {{"kind": "float", "value": -1.0}}}}}}
      {extra_equation}],
    "automata": [
      {{"name": "refill", "states": ["wait", "done"], "init": "wait",
       "transitions": [{{"name": "refill", "source": "wait", "targets": ["{refill_source}"], {refill_law},
         "effects": [{{"target": {{"component": "t", "attribute": "level"}}, "value": {value}}}]}}]}},
      {{"name": "status", "states": ["wet", "dry"], "init": "wet",
       "transitions": [{{"name": "dry", "source": "wet", "targets": ["dry"], "distrib": "watched",
         "guard": {{"op": "cmp", "cmp": "le",
                   "lhs": {{"op": "attr", "attr": {{"component": "t", "attribute": "level"}}}},
                   "rhs": {{"op": "const", "value": {{"kind": "float", "value": 0.0}}}}}}}}]}}]
  }}],
  "indicators": [
    {{"name": "level", "target": "attribute", "attr": {{"component": "t", "attribute": "level"}}}},
    {{"name": "dry", "target": "state", "component": "t", "automaton": "status", "state": "dry"}}]
}}}}"#
    )
}

const TEN: &str = r#"{"op": "const", "value": {"kind": "float", "value": 10.0}}"#;

/// The level sampled on `samples`, and the date the tank ran dry.
fn run_tank(json: &str, t_max: f64, samples: Vec<f64>) -> (Vec<f64>, Option<f64>) {
    let model = Model::from_json(json).unwrap();
    let compiled = CompiledModel::compile(&model).unwrap();
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
    let level = result
        .samples
        .iter()
        .find(|s| s.name == "level")
        .unwrap()
        .points
        .iter()
        .map(|(_, v)| match v {
            Value::Float(x) => *x,
            other => panic!("level is {other:?}"),
        })
        .collect();
    let dry = result
        .indicators
        .iter()
        .find(|s| s.name == "dry")
        .unwrap()
        .points
        .iter()
        // A state indicator reads 1.0 while the state is active.
        .find(|(_, v)| matches!(v, Value::Float(x) if *x == 1.0))
        .map(|(t, _)| *t);
    (level, dry)
}

fn assert_close(got: f64, expected: f64) {
    assert!((got - expected).abs() < 1e-6, "{got} against {expected}");
}

#[test]
fn an_edge_effect_on_an_ode_target_resets_it_and_integration_restarts_from_there() {
    // Drained to 6 by 4, reset to 10 at 4: the level reads 7, 9.5, 9 around
    // the jump, and the tank runs dry at 14 instead of 10.
    let json = tank(r#""distrib": "delay", "time": 4.0"#, "done", TEN, "");
    let (level, dry) = run_tank(&json, 20.0, vec![3.0, 4.5, 5.0, 13.0]);
    for (got, expected) in level.iter().zip([7.0, 9.5, 9.0, 1.0]) {
        assert_close(*got, expected);
    }
    assert_close(dry.expect("the tank runs dry"), 14.0);
}

#[test]
fn a_watched_transition_can_reset_the_variable_its_guard_reads() {
    // The refill is watched at level <= 2 and puts it back to 10, looping
    // on itself: 10 -> 2 in 8, then every 8 units. The tank never runs dry.
    let guard = r#""distrib": "watched", "guard": {"op": "cmp", "cmp": "le",
        "lhs": {"op": "attr", "attr": {"component": "t", "attribute": "level"}},
        "rhs": {"op": "const", "value": {"kind": "float", "value": 2.0}}}"#;
    let json = tank(guard, "wait", TEN, "");
    let (level, dry) = run_tank(&json, 30.0, vec![7.0, 9.0, 17.0, 25.0]);
    for (got, expected) in level.iter().zip([3.0, 9.0, 9.0, 9.0]) {
        assert_close(*got, expected);
    }
    assert_eq!(dry, None);
}

#[test]
fn an_edge_write_on_an_explicit_equation_target_is_still_refused() {
    let explicit = r#", {"target": "shadow", "kind": "explicit",
        "expr": {"op": "const", "value": {"kind": "float", "value": 1.0}}}"#;
    let json = tank(r#""distrib": "delay", "time": 4.0"#, "done", TEN, explicit).replace(
        r#""effects": [{"target": {"component": "t", "attribute": "level"}"#,
        r#""effects": [{"target": {"component": "t", "attribute": "shadow"}"#,
    );
    let outcome = Model::from_json(&json).unwrap().validate();
    assert!(
        matches!(
            &outcome,
            Err(ModelError::TransitionEffectOverwritten { attribute, writer, .. })
                if attribute == "t.shadow" && writer.contains("explicit equation")
        ),
        "{outcome:?}"
    );
}

#[test]
fn a_reset_to_a_value_that_is_not_a_finite_float_is_an_error() {
    // 0/0 is NaN: integrating from it would spread it silently.
    let nan = r#"{"op": "div", "lhs": {"op": "const", "value": {"kind": "float", "value": 0.0}},
                 "rhs": {"op": "const", "value": {"kind": "float", "value": 0.0}}}"#;
    let json = tank(r#""distrib": "delay", "time": 4.0"#, "done", nan, "");
    let model = Model::from_json(&json).unwrap();
    let compiled = CompiledModel::compile(&model).unwrap();
    let outcome = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 20.0,
            ..EngineConfig::default()
        },
    )
    .unwrap()
    .run();
    let error = outcome.expect_err("a NaN reset is refused");
    assert!(error.to_string().contains("finite float"), "{error}");
}
