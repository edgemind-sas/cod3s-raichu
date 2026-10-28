//! Modes, their combination and the study (unit U3), run on the engine.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::Value as V;
use raichu_library::{
    expand, parse_library, parse_study, parse_system, Library, StudyDocument, SystemModel,
};
use raichu_model::Model;
use serde_json::{json, Value};

fn library(body: Value) -> Library {
    parse_library(&json!({"raichu_library": {"format": 1}, "library": body}).to_string()).unwrap()
}

fn system(body: Value) -> SystemModel {
    parse_system(&json!({"raichu_system": {"format": 1}, "system": body}).to_string()).unwrap()
}

fn study(body: Value) -> StudyDocument {
    parse_study(&json!({"raichu_study": {"format": 1}, "study": body}).to_string()).unwrap()
}

fn boolean(b: bool) -> Value {
    json!({"kind": "bool", "value": b})
}

fn attr(component: &str, attribute: &str) -> Value {
    json!({"op": "attr", "attr": {"component": component, "attribute": attribute}})
}

fn eq_true(component: &str, attribute: &str) -> Value {
    json!({"op": "cmp", "cmp": "eq", "lhs": attr(component, attribute), "rhs": {"op": "const", "value": boolean(true)}})
}

fn delay(t: f64) -> Value {
    json!({"distrib": "delay", "time": t})
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

fn events(result: &raichu_core::SimulationResult) -> Vec<(String, f64)> {
    result
        .sequence
        .as_ref()
        .unwrap()
        .events
        .iter()
        .map(|e| (format!("{}.{}.{}", e.obj, e.automaton, e.attr), e.time))
        .collect()
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

fn actuator_library() -> Library {
    let cut = json!([{"attribute": "COCOFE_fed_available_out", "value": false}]);
    library(json!({
        "name": "stra", "version": "1.0.0",
        "types": [{
            "name": "Actionneur",
            "body": {"attributes": [
                {"name": "COCOFE_fed_available_out", "kind": "bool", "init": boolean(true)},
                {"name": "level", "kind": "float", "init": {"kind": "float", "value": 1.0}}
            ]},
            "targetable": [
                {"attribute": "COCOFE_fed_available_out", "rule": "any_cuts"},
                {"attribute": "level", "rule": "min"}
            ],
            "modes": [
                {"name": "mdd_actionneur_led", "nature": "failure", "effects": cut,
                 "condition": {"op": "cmp", "cmp": "lt", "lhs": {"op": "time"}, "rhs": {"op": "const", "value": {"kind": "float", "value": 3.0}}}},
                {"name": "mdd_actionneur_alim", "nature": "failure", "effects": cut},
                {"name": "mdc_actionneur_override", "nature": "compromise", "effects": cut,
                 "condition": {"op": "cmp", "cmp": "lt", "lhs": {"op": "time"}, "rhs": {"op": "const", "value": {"kind": "float", "value": 3.0}}}},
                {"name": "low", "nature": "failure", "effects": [{"attribute": "level", "value": 0.5}]},
                {"name": "lower", "nature": "failure", "effects": [{"attribute": "level", "value": 0.2}]}
            ]
        }]
    }))
}

fn actuator_system() -> SystemModel {
    system(
        json!({"name": "fire", "library": {"name": "stra", "version": "1.0.0"},
        "instances": [{"name": "Act", "type": "Actionneur"}],
        "indicators": [
            {"name": "available", "target": "attribute", "attr": {"component": "Act", "attribute": "COCOFE_fed_available_out"}},
            {"name": "level", "target": "attribute", "attr": {"component": "Act", "attribute": "level"}}
        ]}),
    )
}

/// Covers AE1: three modes cut one output; the output stays cut while any
/// of them is occurred and returns to its nominal value after the last
/// repair.
#[test]
fn several_modes_cutting_one_output_release_it_after_the_last_repair() {
    let s = study(json!({"name": "s", "system": "fire", "modes": [
        {"mode": "mdd_actionneur_led", "instances": ["Act"], "occurrence": delay(1.0), "repair": delay(3.0)},
        {"mode": "mdc_actionneur_override", "instances": ["Act"], "occurrence": delay(2.0), "repair": delay(5.0)},
        {"mode": "mdd_actionneur_alim", "instances": ["Act"], "occurrence": delay(100.0)}
    ]}));
    let expansion = expand(&[actuator_library()], &actuator_system(), Some(&s)).unwrap();
    let result = run(&expansion.model, 20.0);
    // led occurs at 1 and is repaired at 4; override occurs at 2 and is
    // repaired at 7; neither re-occurs (their condition is `time < 3`).
    assert_eq!(
        series(&result, "available"),
        vec![
            (0.0, V::Bool(true)),
            (1.0, V::Bool(false)),
            (7.0, V::Bool(true))
        ]
    );
    assert_eq!(
        events(&result),
        vec![
            ("Act.mdd_actionneur_led.occ".to_owned(), 1.0),
            ("Act.mdc_actionneur_override.occ".to_owned(), 2.0),
            ("Act.mdd_actionneur_led.rep".to_owned(), 4.0),
            ("Act.mdc_actionneur_override.rep".to_owned(), 7.0),
        ]
    );
}

/// Covers AE2.
#[test]
fn a_mode_targeting_an_attribute_without_a_rule_is_refused() {
    let lib = library(json!({"name": "stra", "version": "1.0.0", "types": [{
        "name": "Actionneur",
        "body": {"attributes": [{"name": "COCOFE_prod_available", "kind": "bool", "init": boolean(true)}]},
        "modes": [{"name": "crash", "nature": "failure", "effects": [{"attribute": "COCOFE_prod_available", "value": false}]}]
    }]}));
    let s = study(
        json!({"name": "s", "system": "fire", "modes": [{"mode": "crash", "instances": ["Act"], "occurrence": delay(1.0)}]}),
    );
    let err = expand(&[lib], &actuator_system_without_indicators(), Some(&s)).unwrap_err();
    assert!(
        err.message.contains("`COCOFE_prod_available`"),
        "{}",
        err.message
    );
    assert!(err.message.contains("type `Actionneur`"), "{}", err.message);
    assert!(err.message.contains("mode `crash`"), "{}", err.message);
    assert!(err.message.contains("combination rule"), "{}", err.message);
}

fn actuator_system_without_indicators() -> SystemModel {
    system(
        json!({"name": "fire", "library": {"name": "stra", "version": "1.0.0"},
        "instances": [{"name": "Act", "type": "Actionneur"}]}),
    )
}

#[test]
fn min_holds_the_smallest_active_value_and_returns_to_the_init() {
    let s = study(json!({"name": "s", "system": "fire", "modes": [
        {"mode": "low", "instances": ["Act"], "occurrence": delay(1.0), "repair": delay(4.0)},
        {"mode": "lower", "instances": ["Act"], "occurrence": delay(2.0), "repair": delay(1.0)}
    ]}));
    let expansion = expand(&[actuator_library()], &actuator_system(), Some(&s)).unwrap();
    let result = run(&expansion.model, 4.5);
    // low occurs at 1 (0.5); lower occurs at 2 (0.2) and is repaired at 3,
    // then occurs again at 5 (beyond the horizon); low is repaired at 5.
    assert_eq!(
        series(&result, "level"),
        vec![
            (0.0, V::Float(1.0)),
            (1.0, V::Float(0.5)),
            (2.0, V::Float(0.2)),
            (3.0, V::Float(0.5))
        ]
    );
}

#[test]
fn any_cuts_with_a_true_effect_is_refused() {
    let lib = library(json!({"name": "stra", "version": "1.0.0", "types": [{
        "name": "Actionneur",
        "body": {"attributes": [{"name": "out", "kind": "bool", "init": boolean(true)}]},
        "targetable": [{"attribute": "out", "rule": "any_cuts"}],
        "modes": [{"name": "m", "nature": "failure", "effects": [{"attribute": "out", "value": true}]}]
    }]}));
    let s = study(
        json!({"name": "s", "system": "fire", "modes": [{"mode": "m", "instances": ["Act"], "occurrence": delay(1.0)}]}),
    );
    let err = expand(&[lib], &actuator_system_without_indicators(), Some(&s)).unwrap_err();
    assert!(err.message.contains("`any_cuts`"), "{}", err.message);
}

#[test]
fn a_targeted_attribute_also_written_by_the_type_is_refused() {
    let lib = library(json!({"name": "stra", "version": "1.0.0", "types": [{
        "name": "Actionneur",
        "body": {"attributes": [{"name": "x", "kind": "float", "init": {"kind": "float", "value": 1.0}}],
                 "equations": [{"target": "x", "kind": "explicit", "expr": {"op": "const", "value": {"kind": "float", "value": 2.0}}}]},
        "targetable": [{"attribute": "x", "rule": "max"}],
        "modes": [{"name": "m", "nature": "failure", "effects": [{"attribute": "x", "value": 5.0}]}]
    }]}));
    let s = study(
        json!({"name": "s", "system": "fire", "modes": [{"mode": "m", "instances": ["Act"], "occurrence": delay(1.0)}]}),
    );
    let err = expand(&[lib], &actuator_system_without_indicators(), Some(&s)).unwrap_err();
    assert!(err.message.contains("equation of `x`"), "{}", err.message);
}

#[test]
fn a_mode_without_repair_stays_occurred() {
    let s = study(json!({"name": "s", "system": "fire", "modes": [
        {"mode": "mdd_actionneur_alim", "instances": ["Act"], "occurrence": delay(1.0)}
    ]}));
    let expansion = expand(&[actuator_library()], &actuator_system(), Some(&s)).unwrap();
    let result = run(&expansion.model, 50.0);
    assert_eq!(
        series(&result, "available"),
        vec![(0.0, V::Bool(true)), (1.0, V::Bool(false))]
    );
}

fn sensor_library() -> Library {
    library(json!({"name": "imdr", "version": "1.0.0", "types": [{
        "name": "Sensor",
        "body": {"attributes": [{"name": "measure_blocked", "kind": "bool", "init": boolean(false)}]},
        "targetable": [{"attribute": "measure_blocked", "rule": "any_activates"}],
        "modes": [{"name": "blocked", "nature": "failure", "effects": [{"attribute": "measure_blocked", "value": true}]}]
    }]}))
}

fn nine_sensors() -> SystemModel {
    let instances: Vec<Value> = (1..=9)
        .map(|i| json!({"name": format!("CT_{i}"), "type": "Sensor"}))
        .collect();
    system(
        json!({"name": "plant", "library": {"name": "imdr", "version": "1.0.0"}, "instances": instances}),
    )
}

/// Covers AE3.
#[test]
fn a_type_level_activation_gives_every_instance_its_own_mode() {
    let s = study(json!({"name": "s", "system": "plant", "modes": [
        {"mode": "blocked", "type": "Sensor", "occurrence": delay(2.0)}
    ]}));
    let expansion = expand(&[sensor_library()], &nine_sensors(), Some(&s)).unwrap();
    let model = &expansion.model;
    assert_eq!(model.name, "plant/s");
    for component in &model.components {
        assert!(
            component.automata.iter().any(|a| a.name == "blocked"),
            "{}",
            component.name
        );
    }
    let result = run(model, 5.0);
    let fired = events(&result);
    assert_eq!(fired.len(), 9);
    assert_eq!(fired[0], ("CT_1.blocked.occ".to_owned(), 2.0));
}

#[test]
fn an_instance_activation_overrides_the_type_activation_for_that_instance() {
    let s = study(json!({"name": "s", "system": "plant", "modes": [
        {"mode": "blocked", "type": "Sensor", "occurrence": delay(2.0)},
        {"mode": "blocked", "instances": ["CT_2"], "occurrence": delay(3.0)}
    ]}));
    let expansion = expand(&[sensor_library()], &nine_sensors(), Some(&s)).unwrap();
    let fired = events(&run(&expansion.model, 5.0));
    assert!(
        fired.contains(&("CT_2.blocked.occ".to_owned(), 3.0)),
        "{fired:?}"
    );
    assert!(
        fired.contains(&("CT_3.blocked.occ".to_owned(), 2.0)),
        "{fired:?}"
    );
}

#[test]
fn two_activations_at_the_same_level_are_refused() {
    let s = study(json!({"name": "s", "system": "plant", "modes": [
        {"mode": "blocked", "instances": ["CT_2"], "occurrence": delay(2.0)},
        {"mode": "blocked", "instances": ["CT_2"], "occurrence": delay(3.0)}
    ]}));
    let err = expand(&[sensor_library()], &nine_sensors(), Some(&s)).unwrap_err();
    assert!(
        err.message.contains("`CT_2.blocked` is activated twice"),
        "{}",
        err.message
    );
}

#[test]
fn a_mode_absent_from_the_catalogue_is_refused_naming_the_catalogue() {
    let s = study(json!({"name": "s", "system": "plant", "modes": [
        {"mode": "drift", "type": "Sensor", "occurrence": delay(2.0)}
    ]}));
    let err = expand(&[sensor_library()], &nine_sensors(), Some(&s)).unwrap_err();
    assert!(err.message.contains("no mode `drift`"), "{}", err.message);
    assert!(err.message.contains("`blocked`"), "{}", err.message);
}

fn cyber_library() -> Library {
    library(json!({"name": "cyber", "version": "1.0.0", "types": [
        {"name": "Serveur",
         "body": {"attributes": [{"name": "f_service_prod_available", "kind": "bool", "init": boolean(false)}],
                  "ports": [{"name": "f_service_out", "dir": "out", "attr": "f_service_prod_available"}]},
         "targetable": [{"attribute": "f_service_prod_available", "rule": "any_activates"}],
         "modes": [
            {"name": "mdc_a", "nature": "compromise"},
            {"name": "mdc_b", "nature": "compromise", "effects": [{"attribute": "f_service_prod_available", "value": true}]}
         ]},
        {"name": "ProcessIndustriel",
         "body": {"attributes": [
                    {"name": "f_service_fed_in", "kind": "bool", "init": boolean(false)},
                    {"name": "F1_fed_available_out", "kind": "bool", "init": boolean(true)}],
                  "ports": [{"name": "f_service_in", "dir": "in"}],
                  "sensitive_functions": [{"name": "fed", "effects": [{
                    "target": {"component": "$self", "attribute": "f_service_fed_in"},
                    "value": {"op": "port_agg", "port": {"component": "$self", "port": "f_service_in"}, "agg": "any"}}]}]},
         "targetable": [{"attribute": "F1_fed_available_out", "rule": "any_cuts"}],
         "modes": [{"name": "mdc_proc", "nature": "compromise",
                    "condition": eq_true("$self", "f_service_fed_in"),
                    "effects": [{"attribute": "F1_fed_available_out", "value": false}]}]}
    ]}))
}

fn cyber_system() -> SystemModel {
    system(
        json!({"name": "cyber_3comp", "library": {"name": "cyber", "version": "1.0.0"},
        "instances": [{"name": "Srv", "type": "Serveur"}, {"name": "Proc", "type": "ProcessIndustriel"}],
        "connections": [{"from": "Srv.f_service_out", "to": "Proc.f_service_in"}]}),
    )
}

/// Covers AE4: an escalation chain across modes and instances, named by
/// public names.
#[test]
fn an_attack_chain_fires_in_order_under_public_names() {
    let s = study(json!({"name": "s", "system": "cyber_3comp",
        "modes": [
            {"mode": "mdc_a", "instances": ["Srv"], "occurrence": delay(10.0)},
            {"mode": "mdc_b", "instances": ["Srv"], "occurrence": delay(5.0),
             "condition": {"op": "mode_active", "mode": "Srv.mdc_a"}},
            {"mode": "mdc_proc", "instances": ["Proc"], "occurrence": delay(8.0)}
        ],
        "feared_events": [{"name": "process_F1_lost", "target": true, "tempo_occ": 0.01,
            "condition": {"op": "cmp", "cmp": "eq", "lhs": attr("Proc", "F1_fed_available_out"),
                          "rhs": {"op": "const", "value": boolean(false)}}}]}));
    let expansion = expand(&[cyber_library()], &cyber_system(), Some(&s)).unwrap();
    let result = run(&expansion.model, 30.0);
    assert_eq!(
        events(&result),
        vec![
            ("Srv.mdc_a.occ".to_owned(), 10.0),
            ("Srv.mdc_b.occ".to_owned(), 15.0),
            ("Proc.mdc_proc.occ".to_owned(), 23.0),
            ("process_F1_lost.ev.occ".to_owned(), 23.01),
        ]
    );
    assert_eq!(expansion.model.targets[0].name, "process_F1_lost");
    let origin = expansion
        .provenance
        .elements
        .get("automaton:Srv.mdc_b")
        .unwrap();
    assert_eq!(origin.nature.as_deref(), Some("compromise"));
}

#[test]
fn a_condition_on_a_mode_the_study_does_not_activate_is_refused() {
    let s = study(json!({"name": "s", "system": "cyber_3comp", "modes": [
        {"mode": "mdc_b", "instances": ["Srv"], "occurrence": delay(5.0),
         "condition": {"op": "mode_active", "mode": "Srv.mdc_a"}}
    ]}));
    let err = expand(&[cyber_library()], &cyber_system(), Some(&s)).unwrap_err();
    assert!(
        err.message
            .contains("`Srv.mdc_a`, which the study does not activate"),
        "{}",
        err.message
    );
}

#[test]
fn a_feared_event_named_like_an_instance_is_refused() {
    let s = study(
        json!({"name": "s", "system": "cyber_3comp", "feared_events": [
            {"name": "Srv", "condition": {"op": "const", "value": boolean(true)}}
        ]}),
    );
    let err = expand(&[cyber_library()], &cyber_system(), Some(&s)).unwrap_err();
    assert!(
        err.message.contains("feared event `Srv`"),
        "{}",
        err.message
    );
}

#[test]
fn a_study_indicator_reusing_a_system_indicator_name_is_refused() {
    let s = study(json!({"name": "s", "system": "fire", "indicators": [
        {"name": "available", "target": "attribute", "attr": {"component": "Act", "attribute": "level"}}
    ]}));
    let err = expand(&[actuator_library()], &actuator_system(), Some(&s)).unwrap_err();
    assert!(
        err.message.contains("indicator `available`"),
        "{}",
        err.message
    );
}

#[test]
fn a_study_naming_another_system_is_refused() {
    let s = study(json!({"name": "s", "system": "elsewhere"}));
    let err = expand(&[actuator_library()], &actuator_system(), Some(&s)).unwrap_err();
    assert!(
        err.message.contains("`elsewhere`") && err.message.contains("`fire`"),
        "{}",
        err.message
    );
}

#[test]
fn a_one_off_mode_colliding_with_a_catalogued_mode_is_refused() {
    let s = study(json!({"name": "s", "system": "plant", "one_off_modes": [
        {"instance": "CT_1", "name": "blocked", "nature": "failure", "occurrence": delay(1.0)}
    ]}));
    let err = expand(&[sensor_library()], &nine_sensors(), Some(&s)).unwrap_err();
    assert!(err.message.contains("collides"), "{}", err.message);
}

#[test]
fn a_one_off_mode_runs_on_its_instance() {
    let s = study(json!({"name": "s", "system": "plant", "one_off_modes": [
        {"instance": "CT_4", "name": "stuck", "nature": "failure", "occurrence": delay(1.5),
         "effects": [{"attribute": "measure_blocked", "value": true}]}
    ]}));
    let expansion = expand(&[sensor_library()], &nine_sensors(), Some(&s)).unwrap();
    assert_eq!(
        events(&run(&expansion.model, 5.0)),
        vec![("CT_4.stuck.occ".to_owned(), 1.5)]
    );
}

#[test]
fn two_catalogued_modes_with_one_name_are_refused() {
    let lib = library(json!({"name": "imdr", "version": "1.0.0", "types": [{
        "name": "Sensor",
        "body": {"attributes": [{"name": "measure_blocked", "kind": "bool", "init": boolean(false)}]},
        "targetable": [{"attribute": "measure_blocked", "rule": "any_activates"}],
        "modes": [{"name": "blocked", "nature": "failure"}, {"name": "blocked", "nature": "failure"}]}]}));
    let err = expand(&[lib], &nine_sensors(), None).unwrap_err();
    assert_eq!(err.pointer, "/library/types/0/modes/1/name");
    assert!(
        err.message
            .contains("mode `blocked` is declared twice on type `Sensor`"),
        "{}",
        err.message
    );
}

// ---- on-demand laws -------------------------------------------------------

fn backup_library() -> Library {
    library(
        json!({"name": "standby", "version": "1.0.0", "imports": ["std"], "types": [
            {"name": "Command",
             "uses": [{"fragment": "std.bool_source_out", "with": {"name": "ctrl"}}],
             "modes": [{"name": "lost", "nature": "failure", "effects": [{"attribute": "ctrl_fed_available_out", "value": false}]}]},
            {"name": "Backup",
             "uses": [
                {"fragment": "std.bool_flow_in", "with": {"name": "ctrl"}},
                {"fragment": "std.bool_source_out", "with": {"name": "elec"}}
             ],
             "modes": [{"name": "refusal", "nature": "failure", "effects": [{"attribute": "elec_fed_available_out", "value": false}]}]}
        ]}),
    )
}

fn backup_system() -> SystemModel {
    system(
        json!({"name": "standby", "library": {"name": "standby", "version": "1.0.0"},
        "instances": [{"name": "C", "type": "Command"}, {"name": "B", "type": "Backup"}],
        "connections": [{"from": "C.ctrl_out", "to": "B.ctrl_in"}],
        "indicators": [{"name": "elec", "target": "attribute", "attr": {"component": "B", "attribute": "elec_fed_out"}}]}),
    )
}

/// The backup is solicited when its command is lost (at 2), and refuses to
/// start with probability `p`; a refusal is repaired after 5.
fn on_demand_study(p: f64) -> StudyDocument {
    let solicited = json!({"op": "cmp", "cmp": "eq", "lhs": attr("B", "ctrl_fed_in"),
                           "rhs": {"op": "const", "value": boolean(false)}});
    study(json!({"name": "s", "system": "standby", "modes": [
        {"mode": "lost", "instances": ["C"], "occurrence": delay(2.0)},
        {"mode": "refusal", "instances": ["B"], "occurrence": {"distrib": "inst", "probs": [p]},
         "repair": delay(5.0), "condition": solicited}
    ]}))
}

#[test]
fn an_on_demand_mode_draws_once_on_the_rising_edge_of_its_solicitation() {
    let model = expand(
        &[backup_library()],
        &backup_system(),
        Some(&on_demand_study(0.3)),
    )
    .unwrap()
    .model;
    let backup = model.components.iter().find(|c| c.name == "B").unwrap();
    let mode = backup
        .automata
        .iter()
        .find(|a| a.name == "refusal")
        .unwrap();
    assert_eq!(mode.states, vec!["rep", "occ", "parked"]);
    let names: Vec<_> = mode.transitions.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["occ", "parked", "rep"]);
    let draw = &mode.transitions[0];
    assert_eq!(draw.targets, vec!["occ", "parked"]);
    assert_eq!(
        draw.distrib,
        raichu_model::Distrib::Inst { probs: vec![0.3] }
    );
    assert_eq!(draw.monitored_states, Some(vec!["occ".to_owned()]));
    assert!(draw.monitored);
    assert_eq!(draw.kind, Some(raichu_model::TransitionKind::Failure));
    assert_eq!(draw.cycle_group.as_deref(), Some("refusal"));
    // The re-arm waits for the solicitation to fall, and is no event.
    let rearm = &mode.transitions[1];
    assert_eq!(
        (rearm.source.as_str(), rearm.targets.clone()),
        ("parked", vec!["rep".to_owned()])
    );
    assert!(!rearm.monitored);
    assert_eq!((rearm.kind, rearm.cycle_group.clone()), (None, None));
}

#[test]
fn a_certain_refusal_cuts_the_backup_when_it_is_solicited() {
    let model = expand(
        &[backup_library()],
        &backup_system(),
        Some(&on_demand_study(1.0)),
    )
    .unwrap()
    .model;
    let result = run(&model, 6.0);
    assert_eq!(
        events(&result),
        vec![
            ("C.lost.occ".to_owned(), 2.0),
            ("B.refusal.occ".to_owned(), 2.0)
        ]
    );
    let model = expand(
        &[backup_library()],
        &backup_system(),
        Some(&on_demand_study(0.0)),
    )
    .unwrap()
    .model;
    let result = run(&model, 6.0);
    // A start that never fails parks the draw: no refusal event.
    assert_eq!(events(&result), vec![("C.lost.occ".to_owned(), 2.0)]);
}

#[test]
fn an_on_demand_repair_law_is_refused() {
    let mut st = on_demand_study(0.3);
    st.modes[1].repair = Some(json!({"distrib": "inst", "probs": [0.5]}));
    let err = expand(&[backup_library()], &backup_system(), Some(&st)).unwrap_err();
    assert!(err.message.contains("repair"), "{}", err.message);
}

#[test]
fn a_lost_draw_parks_for_the_solicitation_and_re_arms_when_it_falls() {
    // The command is lost on [2, 3), [5, 6), ...: the backup is solicited
    // then. A start that never fails parks each draw and re-arms between.
    let mut st = on_demand_study(0.0);
    st.modes[0].repair = Some(json!({"distrib": "delay", "time": 1.0}));
    // An activation asking for another interruption policy keeps `reset`
    // on the draw: a solicitation that falls within its instant cancels it.
    st.modes[1].on_interruption = Some(raichu_model::InterruptionPolicy::Continue);
    st.modes[1].repair = None;
    let model = expand(&[backup_library()], &backup_system(), Some(&st))
        .unwrap()
        .model;
    let backup = model.components.iter().find(|c| c.name == "B").unwrap();
    let mode = backup
        .automata
        .iter()
        .find(|a| a.name == "refusal")
        .unwrap();
    assert!(mode.transitions[..2]
        .iter()
        .all(|t| t.on_interruption == raichu_model::InterruptionPolicy::Reset));
    let compiled = CompiledModel::compile(&model).unwrap();
    let mut engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 7.0,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let mut seen = Vec::new();
    for t in [1.0, 2.5, 4.0, 5.5] {
        engine.advance_to(t).unwrap();
        seen.push(engine.state("B.refusal").unwrap().to_owned());
    }
    assert_eq!(seen, vec!["rep", "parked", "rep", "parked"]);
}

fn not_active(mode: &str) -> Value {
    json!({"op": "bool", "bool_op": "not", "args": [{"op": "mode_active", "mode": mode}]})
}

/// A repair condition holds the repair back while it is false: `alim`
/// would repair at 2, but `low` occurs at 1.5 and blocks it until its own
/// repair at 4.5; the repair delay then runs again (reset policy).
#[test]
fn a_repair_condition_holds_the_repair_until_it_becomes_true() {
    let s = study(json!({"name": "s", "system": "fire", "modes": [
        {"mode": "mdd_actionneur_alim", "instances": ["Act"], "occurrence": delay(1.0),
         "repair": delay(1.0), "repair_condition": not_active("Act.low")},
        {"mode": "low", "instances": ["Act"], "occurrence": delay(1.5), "repair": delay(3.0)}
    ]}));
    let expansion = expand(&[actuator_library()], &actuator_system(), Some(&s)).unwrap();
    let result = run(&expansion.model, 5.9);
    assert_eq!(
        events(&result),
        vec![
            ("Act.mdd_actionneur_alim.occ".to_owned(), 1.0),
            ("Act.low.occ".to_owned(), 1.5),
            ("Act.low.rep".to_owned(), 4.5),
            ("Act.mdd_actionneur_alim.rep".to_owned(), 5.5),
        ]
    );
    let alim = expansion.model.components[0]
        .automata
        .iter()
        .find(|a| a.name == "mdd_actionneur_alim")
        .unwrap();
    let occurrence = alim.transitions.iter().find(|t| t.name == "occ").unwrap();
    assert_eq!(occurrence.guard, None, "the occurrence edge is unchanged");
}

#[test]
fn a_repair_condition_never_reaches_the_on_demand_rearm() {
    let mut s = on_demand_study(0.3);
    s.modes[1].repair_condition = Some(not_active("C.lost"));
    let model = expand(&[backup_library()], &backup_system(), Some(&s))
        .unwrap()
        .model;
    let backup = model.components.iter().find(|c| c.name == "B").unwrap();
    let mode = backup
        .automata
        .iter()
        .find(|a| a.name == "refusal")
        .unwrap();
    let draw = &mode.transitions[0];
    let rearm = &mode.transitions[1];
    let repair = &mode.transitions[2];
    assert_eq!(
        rearm.guard,
        Some(raichu_expr::Expr::Bool {
            bool_op: raichu_expr::BoolOp::Not,
            args: vec![draw.guard.clone().unwrap()],
        }),
        "the re-arm is guarded by the negated occurrence condition only"
    );
    assert!(repair.guard.is_some());
}

#[test]
fn a_repair_condition_naming_an_unactivated_mode_is_refused() {
    let s = study(json!({"name": "s", "system": "fire", "modes": [
        {"mode": "mdd_actionneur_alim", "instances": ["Act"], "occurrence": delay(1.0),
         "repair": delay(1.0), "repair_condition": not_active("Act.low")}
    ]}));
    let error = expand(&[actuator_library()], &actuator_system(), Some(&s)).unwrap_err();
    assert!(error.to_string().contains("Act.low"), "{error}");
    assert!(
        error.pointer.contains("/repair_condition"),
        "{}",
        error.pointer
    );
}

#[test]
fn a_repair_condition_without_a_repair_law_is_refused() {
    let s = study(json!({"name": "s", "system": "fire", "modes": [
        {"mode": "mdd_actionneur_alim", "instances": ["Act"], "occurrence": delay(1.0),
         "repair_condition": not_active("Act.mdd_actionneur_alim")}
    ]}));
    let error = expand(&[actuator_library()], &actuator_system(), Some(&s)).unwrap_err();
    assert!(error.to_string().contains("repair condition"), "{error}");
}

/// A pulse: occurs at 1 and is repaired at once (delay 0). With its effects
/// carried through a zero-delay step, the cut is held for that step, which
/// an on-demand detector reads; without it, the zero-delay repair withdraws
/// the cut before any draw.
fn pulse_library() -> Library {
    library(json!({
        "name": "pulse", "version": "1.0.0", "imports": ["std"],
        "types": [
            {"name": "Train",
             "uses": [{"fragment": "std.bool_source_out", "with": {"name": "present", "fed_available_init": false}}],
             "modes": [{"name": "passage", "nature": "failure", "effects": [{"attribute": "present_fed_available_out", "value": true}]}]},
            {"name": "Detector",
             "uses": [
                {"fragment": "std.bool_flow_in", "with": {"name": "present"}},
                {"fragment": "std.bool_source_out", "with": {"name": "seen", "fed_available_init": false}}
             ],
             "modes": [{"name": "detection", "nature": "failure", "effects": [{"attribute": "seen_fed_available_out", "value": true}]}]}
        ]
    }))
}

fn pulse_run(step: Option<&str>) -> (Model, raichu_core::SimulationResult) {
    let s = system(
        json!({"name": "s", "library": {"name": "pulse", "version": "1.0.0"},
        "instances": [{"name": "T", "type": "Train"}, {"name": "D", "type": "Detector"}],
        "connections": [{"from": "T.present_out", "to": "D.present_in"}],
        "indicators": [{"name": "seen", "target": "attribute", "attr": {"component": "D", "attribute": "seen_fed_out"}}]}),
    );
    let mut passage = json!({"mode": "passage", "instances": ["T"], "occurrence": delay(1.0), "repair": delay(0.0)});
    if let Some(step) = step {
        passage["effect_step"] = json!(step);
    }
    let solicited = json!({"op": "cmp", "cmp": "eq", "lhs": attr("D", "present_fed_in"),
                           "rhs": {"op": "const", "value": boolean(true)}});
    let st = study(json!({"name": "st", "system": "s", "modes": [
        passage,
        {"mode": "detection", "instances": ["D"], "occurrence": {"distrib": "inst", "probs": [1.0]},
         "condition": solicited}
    ]}));
    let model = expand(&[pulse_library()], &s, Some(&st)).unwrap().model;
    let result = run(&model, 1.5);
    (model, result)
}

#[test]
fn an_effect_carried_through_a_zero_delay_step_is_seen_for_that_step() {
    let (_, immediate) = pulse_run(None);
    assert_eq!(series(&immediate, "seen"), vec![(0.0, V::Bool(false))]);
    let (model, stepped) = pulse_run(Some("zero_delay"));
    assert_eq!(
        series(&stepped, "seen"),
        vec![(0.0, V::Bool(false)), (1.0, V::Bool(true))]
    );
    let train = model.components.iter().find(|c| c.name == "T").unwrap();
    let step = train
        .automata
        .iter()
        .find(|a| a.name == "passage__effect")
        .unwrap();
    assert_eq!(step.states, vec!["rep", "occ"]);
    // As a PyCATSHOO mirror, the step carries the sequence events and the
    // kinds, in the mode's cycle group; the mode's own automaton none.
    assert!(step
        .transitions
        .iter()
        .all(|t| t.monitored && t.kind.is_some() && t.cycle_group.as_deref() == Some("passage")));
    assert_eq!(
        step.transitions[0].distrib,
        raichu_model::Distrib::Delay { time: 0.0 }
    );
    let passage = train.automata.iter().find(|a| a.name == "passage").unwrap();
    assert!(passage
        .transitions
        .iter()
        .all(|t| !t.monitored && t.kind.is_none() && t.cycle_group.is_none()));
    let fired: Vec<String> = events(&stepped).into_iter().map(|(n, _)| n).collect();
    assert!(
        fired.contains(&"T.passage__effect.occ".to_owned()),
        "{fired:?}"
    );
    assert!(
        fired.iter().all(|n| !n.starts_with("T.passage.")),
        "{fired:?}"
    );
    // The mutual lock of a mirror: the mode occurs again only with its
    // effects at rest, and is repaired only once they have taken hold.
    let guard = |name: &str| {
        let transition = passage.transitions.iter().find(|t| t.name == name).unwrap();
        serde_json::to_string(&transition.guard).unwrap()
    };
    assert!(guard("occ").contains("passage__effect") && guard("occ").contains("\"rep\""));
    assert!(guard("rep").contains("passage__effect") && guard("rep").contains("\"occ\""));
}

#[test]
fn an_instant_step_is_an_instantaneous_transition_of_probability_one() {
    let (model, stepped) = pulse_run(Some("instant"));
    let train = model.components.iter().find(|c| c.name == "T").unwrap();
    let step = train
        .automata
        .iter()
        .find(|a| a.name == "passage__effect")
        .unwrap();
    assert_eq!(
        step.transitions[0].distrib,
        raichu_model::Distrib::Inst { probs: Vec::new() }
    );
    assert_eq!(series(&stepped, "seen")[0], (0.0, V::Bool(false)));
}
