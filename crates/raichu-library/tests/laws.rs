//! Laws in library documents: a law as a type parameter spread onto a
//! transition (`$law`), every law a two-state mode can carry, and timed
//! laws for the confirmation times of feared events.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_library::{expand, parse_library, parse_study, parse_system, Library, SystemModel};
use raichu_model::Distrib;
use serde_json::{json, Value};

fn float(x: f64) -> Value {
    json!({"op": "const", "value": {"kind": "float", "value": x}})
}

fn temp_above(x: f64) -> Value {
    json!({"op": "cmp", "cmp": "gt",
           "lhs": {"op": "attr", "attr": {"component": "$self", "attribute": "temp"}},
           "rhs": float(x)})
}

/// A heater whose temperature rises at 2 per hour, with a failure law as
/// a parameter and a catalogued `overheat` mode whose condition is a
/// continuous boundary.
fn library() -> Library {
    parse_library(
        &json!({"raichu_library": {"format": 1}, "library": {
        "name": "heat", "version": "1.0.0",
        "types": [{
            "name": "Heater",
            "parameters": [{"name": "failure", "kind": "law", "default": {"distrib": "exp", "rate": 0.01}}],
            "body": {
                "attributes": [
                    {"name": "temp", "kind": "float", "init": {"kind": "float", "value": 0.0}},
                    {"name": "on", "kind": "bool", "init": {"kind": "bool", "value": true}}
                ],
                "equations": [{"target": "temp", "kind": "ode", "expr": float(2.0)}],
                "automata": [{"name": "health", "states": ["ok", "ko"], "init": "ok",
                    "transitions": [{"name": "fail", "source": "ok", "targets": ["ko"], "$law": "failure"}]}]
            },
            "targetable": [{"attribute": "on", "rule": "any_cuts"}],
            "modes": [{"name": "overheat", "nature": "failure",
                       "effects": [{"attribute": "on", "value": false}],
                       "condition": temp_above(10.0)},
                      {"name": "stuck", "nature": "failure",
                       "effects": [{"attribute": "on", "value": false}]}]
        }]}})
        .to_string(),
    )
    .unwrap()
}

fn system(failure: Option<Value>) -> SystemModel {
    let mut instance = json!({"name": "H", "type": "Heater"});
    if let Some(law) = failure {
        instance["parameters"] = json!({"failure": law});
    }
    parse_system(
        &json!({"raichu_system": {"format": 1}, "system": {
        "name": "room", "library": {"name": "heat", "version": "1.0.0"}, "instances": [instance]}})
        .to_string(),
    )
    .unwrap()
}

fn expand_with(sections: Value) -> Result<raichu_model::Model, String> {
    let mut body = json!({"name": "s", "system": "room"});
    for (key, value) in sections.as_object().unwrap() {
        body[key] = value.clone();
    }
    let study = parse_study(&json!({"raichu_study": {"format": 1}, "study": body}).to_string())
        .map_err(|e| e.to_string())?;
    expand(&[library()], &system(None), Some(&study))
        .map(|e| e.model)
        .map_err(|e| e.to_string())
}

fn transition<'a>(
    model: &'a raichu_model::Model,
    automaton: &str,
    name: &str,
) -> &'a raichu_model::Transition {
    model
        .components
        .iter()
        .flat_map(|c| &c.automata)
        .find(|a| a.name == automaton)
        .unwrap()
        .transitions
        .iter()
        .find(|t| t.name == name)
        .unwrap()
}

#[test]
fn a_law_parameter_lands_on_the_transition_as_written() {
    for law in [
        json!({"distrib": "exp", "rate": 0.5}),
        json!({"distrib": "weibull", "shape": 2.0, "scale": 100.0}),
    ] {
        let model = expand(&[library()], &system(Some(law.clone())), None)
            .unwrap()
            .model;
        let written = serde_json::to_value(transition(&model, "health", "fail")).unwrap();
        for (key, value) in law.as_object().unwrap() {
            assert_eq!(&written[key], value, "{key}");
        }
    }
}

#[test]
fn a_law_parameter_that_is_no_law_is_refused_with_its_names() {
    let error = expand(&[library()], &system(Some(json!({"rate": 0.5}))), None)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("failure") && error.contains("law"),
        "{error}"
    );
}

#[test]
fn a_key_both_the_law_and_the_transition_write_is_refused() {
    let mut lib = serde_json::to_value(library()).unwrap();
    lib["types"][0]["body"]["automata"][0]["transitions"][0]["rate"] = json!(1.0);
    let lib: Library = serde_json::from_value(lib).unwrap();
    let error = expand(&[lib], &system(None), None).unwrap_err().to_string();
    assert!(
        error.contains("`rate`") && error.contains("already writes"),
        "{error}"
    );
}

fn run(model: &raichu_model::Model, t_max: f64) -> Vec<(String, f64)> {
    let compiled = CompiledModel::compile(model).unwrap();
    let config = EngineConfig {
        t_max,
        sequences: true,
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    result
        .sequence
        .unwrap()
        .events
        .iter()
        .map(|e| (format!("{}.{}.{}", e.obj, e.automaton, e.attr), e.time))
        .collect()
}

/// The temperature crosses 10 at t = 5: a watched occurrence fires on that
/// boundary.
#[test]
fn a_watched_occurrence_fires_on_its_boundary() {
    let model = expand_with(json!({"modes": [{
        "mode": "overheat", "instances": ["H"], "occurrence": {"distrib": "watched"}
    }]}))
    .unwrap();
    let events = run(&model, 12.0);
    let overheat: Vec<_> = events
        .iter()
        .filter(|(n, _)| n.starts_with("H.overheat"))
        .collect();
    assert_eq!(overheat.len(), 1, "{events:?}");
    assert_eq!(overheat[0].0, "H.overheat.occ");
    assert!((overheat[0].1 - 5.0).abs() < 1e-6, "{events:?}");
}

/// Occurred at t = 1 by a delay, the mode is repaired when the temperature
/// crosses 20, at t = 10.
#[test]
fn a_watched_repair_fires_on_its_repair_condition_boundary() {
    let model = expand_with(json!({"modes": [{
        "mode": "stuck", "instances": ["H"],
        "occurrence": {"distrib": "delay", "time": 1.0},
        "repair": {"distrib": "watched"},
        "repair_condition": {"op": "cmp", "cmp": "gt",
            "lhs": {"op": "attr", "attr": {"component": "H", "attribute": "temp"}}, "rhs": float(20.0)}
    }]}))
    .unwrap();
    let events = run(&model, 10.5);
    let overheat: Vec<_> = events
        .iter()
        .filter(|(n, _)| n.starts_with("H.stuck"))
        .collect();
    assert_eq!(overheat[0].0, "H.stuck.occ");
    assert!((overheat[0].1 - 1.0).abs() < 1e-6, "{events:?}");
    assert_eq!(overheat[1].0, "H.stuck.rep");
    assert!((overheat[1].1 - 10.0).abs() < 1e-6, "{events:?}");
}

#[test]
fn a_watched_repair_without_a_repair_condition_is_refused() {
    let error = expand_with(json!({"modes": [{
        "mode": "overheat", "instances": ["H"],
        "occurrence": {"distrib": "exp", "rate": 0.1},
        "repair": {"distrib": "watched"}
    }], "feared_events": []}))
    .unwrap_err();
    assert!(
        error.contains("watched repair") && error.contains("repair_condition"),
        "{error}"
    );
}

/// Occurred at t = 1, the `stuck` mode is repaired at once when the
/// heater's health fails over (a discrete condition), at t = 3.
#[test]
fn a_certain_repair_is_taken_when_its_discrete_condition_holds() {
    let model = expand_with(json!({"modes": [{
        "mode": "stuck", "instances": ["H"],
        "occurrence": {"distrib": "delay", "time": 1.0},
        "repair": {"distrib": "inst", "probs": []},
        "repair_condition": {"op": "state_active",
            "state": {"component": "H", "automaton": "health", "state": "ko"}}
    }]}))
    .unwrap();
    let repair = transition(&model, "stuck", "rep");
    assert!(matches!(repair.distrib, Distrib::Inst { ref probs } if probs.is_empty()));
    let mut lib = serde_json::to_value(library()).unwrap();
    lib["types"][0]["parameters"][0]["default"] = json!({"distrib": "delay", "time": 3.0});
    let lib: Library = serde_json::from_value(lib).unwrap();
    let study = parse_study(
        &json!({"raichu_study": {"format": 1}, "study": {"name": "s", "system": "room", "modes": [{
            "mode": "stuck", "instances": ["H"],
            "occurrence": {"distrib": "delay", "time": 1.0},
            "repair": {"distrib": "inst", "probs": []},
            "repair_condition": {"op": "state_active",
                "state": {"component": "H", "automaton": "health", "state": "ko"}}}]}})
        .to_string(),
    )
    .unwrap();
    let model = expand(&[lib], &system(None), Some(&study)).unwrap().model;
    let events = run(&model, 5.0);
    let stuck: Vec<_> = events
        .iter()
        .filter(|(n, _)| n.starts_with("H.stuck"))
        .collect();
    // Repaired at 3 once health is down, then re-armed: it occurs again a
    // delay later and is repaired at once, the condition still holding.
    assert_eq!(
        stuck[..4],
        [
            &("H.stuck.occ".to_owned(), 1.0),
            &("H.stuck.rep".to_owned(), 3.0),
            &("H.stuck.occ".to_owned(), 4.0),
            &("H.stuck.rep".to_owned(), 4.0)
        ],
        "{events:?}"
    );
}

#[test]
fn a_certain_repair_on_a_continuous_condition_is_refused_pointing_to_watched() {
    let error = expand_with(json!({"modes": [{
        "mode": "stuck", "instances": ["H"],
        "occurrence": {"distrib": "exp", "rate": 0.1},
        "repair": {"distrib": "inst", "probs": []},
        "repair_condition": {"op": "cmp", "cmp": "gt",
            "lhs": {"op": "attr", "attr": {"component": "H", "attribute": "temp"}}, "rhs": float(20.0)}
    }]}))
    .unwrap_err();
    assert!(
        error.contains("`H.temp`") && error.contains("watched"),
        "{error}"
    );
}

#[test]
fn a_certain_repair_without_a_repair_condition_is_refused() {
    let error = expand_with(json!({"modes": [{
        "mode": "overheat", "instances": ["H"],
        "occurrence": {"distrib": "exp", "rate": 0.1},
        "repair": {"distrib": "inst", "probs": []}
    }], "feared_events": []}))
    .unwrap_err();
    assert!(
        error.contains("certain repair") && error.contains("repair_condition"),
        "{error}"
    );
}

#[test]
fn an_occurrence_with_several_branches_is_refused_naming_the_type_body_route() {
    let error = expand_with(json!({"modes": [{
        "mode": "overheat", "instances": ["H"],
        "occurrence": {"distrib": "inst", "probs": [0.2, 0.3]}
    }], "feared_events": []}))
    .unwrap_err();
    assert!(
        error.contains("several branches") && error.contains("type-body automaton"),
        "{error}"
    );
}

#[test]
fn a_feared_event_confirms_with_a_timed_law_and_keeps_a_number_as_a_delay() {
    let model = expand_with(json!({"modes": [], "feared_events": [{
        "name": "Hot", "condition": {"op": "cmp", "cmp": "gt",
            "lhs": {"op": "attr", "attr": {"component": "H", "attribute": "temp"}}, "rhs": float(10.0)},
        "tempo_occ": {"distrib": "weibull", "shape": 2.0, "scale": 3.0},
        "tempo_not_occ": 4.0
    }]}))
    .unwrap();
    assert!(matches!(
        transition(&model, "ev", "occ").distrib,
        Distrib::Weibull { .. }
    ));
    assert!(
        matches!(transition(&model, "ev", "not_occ").distrib, Distrib::Delay { time } if time == 4.0)
    );
}

#[test]
fn a_feared_event_refuses_an_untimed_confirmation_law() {
    let error = expand_with(json!({"modes": [], "feared_events": [{
        "name": "Hot", "condition": {"op": "cmp", "cmp": "gt",
            "lhs": {"op": "attr", "attr": {"component": "H", "attribute": "temp"}}, "rhs": float(10.0)},
        "tempo_occ": {"distrib": "watched"}
    }]}))
    .unwrap_err();
    assert!(
        error.contains("tempo_occ") && error.contains("timed law"),
        "{error}"
    );
}
