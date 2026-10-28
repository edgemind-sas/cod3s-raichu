//! One-shot writes a catalogued mode makes on the edges of its lifecycle
//! (`edge_effects`): written once when the mode occurs or is repaired, and
//! kept until something else writes the attribute.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::Value as V;
use raichu_library::{expand, parse_library, parse_study, parse_system, ExpandError, Library};
use raichu_model::{Model, Transition};
use serde_json::{json, Value};

fn on(attribute: &str, value: Value, face: &str) -> Value {
    json!({"attribute": attribute, "value": value, "on": face})
}

const DETECTED: &str = "detected_fed_available_out";

fn library(extra_modes: Value) -> Library {
    let mut modes = vec![
        json!({"name": "detect", "nature": "failure", "edge_effects": [on(DETECTED, json!(true), "occurrence")]}),
        json!({"name": "fix", "nature": "failure", "edge_effects": [on(DETECTED, json!(false), "repair")]}),
        json!({"name": "toggle", "nature": "failure", "edge_effects": [
            on(DETECTED, json!(true), "occurrence"), on(DETECTED, json!(false), "repair")
        ]}),
        json!({"name": "clamp", "nature": "failure", "effects": [{"attribute": DETECTED, "value": true}]}),
        json!({"name": "tick", "nature": "failure"}),
    ];
    modes.extend(extra_modes.as_array().cloned().unwrap_or_default());
    parse_library(
        &json!({"raichu_library": {"format": 1}, "library": {
            "name": "rails", "version": "1.0.0", "imports": ["std"],
            "types": [{
                "name": "Rail",
                "uses": [{"fragment": "std.bool_source_out", "with": {"name": "detected", "fed_available_init": false}}],
                "body": {"attributes": [{"name": "note", "kind": "bool", "init": {"kind": "bool", "value": false}}]},
                "modes": modes
            }]
        }})
        .to_string(),
    )
    .unwrap()
}

fn system(members: usize) -> raichu_library::SystemModel {
    let instances: Vec<Value> = (1..=members)
        .map(|i| json!({"name": format!("R{i}"), "type": "Rail"}))
        .collect();
    let indicators: Vec<Value> = (1..=members)
        .map(|i| json!({"name": format!("r{i}"), "target": "attribute", "attr": {"component": format!("R{i}"), "attribute": DETECTED}}))
        .collect();
    parse_system(
        &json!({"raichu_system": {"format": 1}, "system": {
            "name": "line", "library": {"name": "rails", "version": "1.0.0"},
            "instances": instances, "indicators": indicators
        }})
        .to_string(),
    )
    .unwrap()
}

fn study(body: Value) -> raichu_library::StudyDocument {
    let mut body = body;
    body["name"] = json!("st");
    body["system"] = json!("line");
    parse_study(&json!({"raichu_study": {"format": 1}, "study": body}).to_string()).unwrap()
}

fn delay(t: f64) -> Value {
    json!({"distrib": "delay", "time": t})
}

fn expanded(members: usize, extra_modes: Value, st: &Value) -> Result<Model, ExpandError> {
    expand(
        &[library(extra_modes)],
        &system(members),
        Some(&study(st.clone())),
    )
    .map(|e| e.model)
}

fn run(model: &Model, t_max: f64) -> raichu_core::SimulationResult {
    let compiled = CompiledModel::compile(model).unwrap();
    let config = EngineConfig {
        t_max,
        sequences: true,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config).unwrap().run().unwrap()
}

fn series(result: &raichu_core::SimulationResult, name: &str) -> Vec<(f64, V)> {
    result
        .indicators
        .iter()
        .find(|s| s.name == name)
        .unwrap()
        .points
        .clone()
}

fn transition<'m>(
    model: &'m Model,
    component: &str,
    automaton: &str,
    name: &str,
) -> &'m Transition {
    model
        .components
        .iter()
        .find(|c| c.name == component)
        .unwrap()
        .automata
        .iter()
        .find(|a| a.name == automaton)
        .unwrap()
        .transitions
        .iter()
        .find(|t| t.name == name)
        .unwrap()
}

fn written_components(t: &Transition) -> Vec<String> {
    t.effects
        .iter()
        .map(|e| e.target.component.clone())
        .collect()
}

fn refusal(extra_modes: Value, members: usize, st: Value) -> String {
    expanded(members, extra_modes, &st).unwrap_err().to_string()
}

#[test]
fn an_occurrence_write_latches_past_the_repair() {
    let model = expanded(
        1,
        json!([]),
        &json!({"modes": [{"mode": "detect", "instances": ["R1"], "occurrence": delay(1.0), "repair": delay(2.0)}]}),
    )
    .unwrap();
    let result = run(&model, 10.0);
    assert_eq!(
        series(&result, "r1"),
        vec![(0.0, V::Bool(false)), (1.0, V::Bool(true))]
    );
    // Written by the edge itself, by no level function.
    assert_eq!(
        written_components(transition(&model, "R1", "detect", "occ")),
        vec!["R1"]
    );
    let rail = model.components.iter().find(|c| c.name == "R1").unwrap();
    assert!(rail
        .sensitive_functions
        .iter()
        .all(|f| !f.name.starts_with("modes__")));
}

#[test]
fn a_repair_write_lands_at_the_repair_date() {
    let model = expanded(
        1,
        json!([]),
        &json!({"modes": [
            {"mode": "detect", "instances": ["R1"], "occurrence": delay(1.0)},
            {"mode": "fix", "instances": ["R1"], "occurrence": delay(2.0), "repair": delay(3.0)}
        ]}),
    )
    .unwrap();
    let result = run(&model, 10.0);
    assert_eq!(
        series(&result, "r1"),
        vec![
            (0.0, V::Bool(false)),
            (1.0, V::Bool(true)),
            (5.0, V::Bool(false))
        ]
    );
}

fn on_demand(p: f64) -> Value {
    // Solicited from 1 on, when `tick` occurs.
    json!({"modes": [
        {"mode": "tick", "instances": ["R1"], "occurrence": delay(1.0)},
        {"mode": "detect", "instances": ["R1"],
         "occurrence": {"distrib": "inst", "probs": [p]}, "condition": {"op": "mode_active", "mode": "R1.tick"}}
    ]})
}

#[test]
fn an_on_demand_draw_writes_only_when_it_lands_on_occurred() {
    let lost = run(&expanded(1, json!([]), &on_demand(0.0)).unwrap(), 5.0);
    assert_eq!(series(&lost, "r1"), vec![(0.0, V::Bool(false))]);
    let won = run(&expanded(1, json!([]), &on_demand(1.0)).unwrap(), 5.0);
    assert_eq!(
        series(&won, "r1"),
        vec![(0.0, V::Bool(false)), (1.0, V::Bool(true))]
    );
}

#[test]
fn a_stepped_mode_writes_on_its_effect_step() {
    let model = expanded(
        1,
        json!([]),
        &json!({"modes": [
            {"mode": "detect", "instances": ["R1"], "occurrence": delay(1.0)},
            {"mode": "fix", "instances": ["R1"], "occurrence": delay(2.0), "repair": delay(3.0),
             "effect_step": "zero_delay"}
        ]}),
    )
    .unwrap();
    assert_eq!(
        written_components(transition(&model, "R1", "fix__effect", "rep")),
        vec!["R1"]
    );
    assert!(transition(&model, "R1", "fix", "rep").effects.is_empty());
    let result = run(&model, 10.0);
    assert_eq!(
        series(&result, "r1"),
        vec![
            (0.0, V::Bool(false)),
            (1.0, V::Bool(true)),
            (5.0, V::Bool(false))
        ]
    );
}

fn group(step: Option<&str>, orders: Value, mode: &str) -> Value {
    let mut g = json!({"name": "EX", "mode": mode, "instances": ["R1", "R2"], "orders": orders});
    if let Some(step) = step {
        g["effect_step"] = json!(step);
    }
    json!({"common_cause_groups": [g]})
}

#[test]
fn a_combination_writes_every_member_it_covers() {
    let model = expanded(
        2,
        json!([]),
        &group(
            None,
            json!([{"order": 2, "occurrence": delay(1.0)}]),
            "detect",
        ),
    )
    .unwrap();
    assert_eq!(
        written_components(transition(&model, "EX", "cc_1_2", "occ")),
        vec!["R1", "R2"]
    );
    let result = run(&model, 5.0);
    for name in ["r1", "r2"] {
        assert_eq!(
            series(&result, name),
            vec![(0.0, V::Bool(false)), (1.0, V::Bool(true))]
        );
    }
}

#[test]
fn a_stepped_group_writes_on_each_member_step() {
    let model = expanded(
        2,
        json!([]),
        &group(
            Some("zero_delay"),
            json!([{"order": 1, "occurrence": delay(3.0)}, {"order": 2, "occurrence": delay(1.0)}]),
            "detect",
        ),
    )
    .unwrap();
    assert!(transition(&model, "EX", "cc_1_2", "occ").effects.is_empty());
    for member in ["R1", "R2"] {
        assert_eq!(
            written_components(transition(&model, member, "EX__effect", "occ")),
            vec![member]
        );
    }
    let result = run(&model, 5.0);
    assert_eq!(
        series(&result, "r1"),
        vec![(0.0, V::Bool(false)), (1.0, V::Bool(true))]
    );
}

#[test]
fn an_edge_write_on_an_attribute_a_mode_holds_is_refused() {
    let message = refusal(
        json!([]),
        1,
        json!({"modes": [
            {"mode": "detect", "instances": ["R1"], "occurrence": delay(1.0)},
            {"mode": "clamp", "instances": ["R1"], "occurrence": delay(2.0)}
        ]}),
    );
    assert!(
        message.contains(DETECTED) && message.contains("held"),
        "{message}"
    );
}

#[test]
fn an_edge_write_on_an_attribute_without_a_rule_is_refused() {
    let message = refusal(
        json!([{"name": "note_it", "nature": "failure", "edge_effects": [on("note", json!(true), "occurrence")]}]),
        1,
        json!({"modes": [{"mode": "note_it", "instances": ["R1"], "occurrence": delay(1.0)}]}),
    );
    assert!(
        message.contains("note") && message.contains("targetable"),
        "{message}"
    );
}

#[test]
fn a_repair_write_on_a_mode_without_repair_is_refused() {
    let message = refusal(
        json!([]),
        1,
        json!({"modes": [{"mode": "fix", "instances": ["R1"], "occurrence": delay(1.0)}]}),
    );
    assert!(message.contains("repair"), "{message}");
}

#[test]
fn an_edge_write_of_the_wrong_kind_is_refused() {
    let message = refusal(
        json!([{"name": "odd", "nature": "failure", "edge_effects": [on(DETECTED, json!(1.5), "occurrence")]}]),
        1,
        json!({"modes": [{"mode": "odd", "instances": ["R1"], "occurrence": delay(1.0)}]}),
    );
    assert!(
        message.contains(DETECTED) && message.contains("1.5"),
        "{message}"
    );
}

#[test]
fn writes_on_both_faces_over_overlapping_combinations_are_refused() {
    let message = refusal(
        json!([]),
        2,
        group(
            None,
            json!([{"order": 1, "occurrence": delay(3.0), "repair": delay(1.0)},
                   {"order": 2, "occurrence": delay(1.0), "repair": delay(1.0)}]),
            "toggle",
        ),
    );
    assert!(
        message.contains("both") && message.contains(DETECTED),
        "{message}"
    );
}
