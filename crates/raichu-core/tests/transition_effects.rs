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
