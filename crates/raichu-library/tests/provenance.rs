//! Provenance and located errors (unit U4).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_library::{
    expand, parse_library, parse_study, parse_system, DocumentKind, Library, StudyDocument,
    SystemModel,
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

fn plant_library() -> Library {
    library(json!({
        "name": "home", "version": "2.0.0",
        "fragments": [{
            "name": "lit_from_port",
            "body": {
                "attributes": [{"name": "lit", "kind": "bool", "init": boolean(false)}],
                "ports": [{"name": "power_in", "dir": "in"}],
                "sensitive_functions": [{"name": "light", "effects": [{
                    "target": {"component": "$self", "attribute": "lit"},
                    "value": {"op": "port_agg", "port": {"component": "$self", "port": "power_in"}, "agg": "any"}}]}]
            }
        }, {
            "name": "broken",
            "body": {
                "attributes": [{"name": "y", "kind": "float", "init": {"kind": "float", "value": 0.0}}],
                "equations": [{"target": "y", "kind": "explicit",
                    "expr": {"op": "attr", "attr": {"component": "$self", "attribute": "missing"}}}]
            }
        }],
        "types": [
            {"name": "Switch", "body": {
                "attributes": [{"name": "on", "kind": "bool", "init": boolean(true)}],
                "ports": [{"name": "power_out", "dir": "out", "attr": "on"}]},
             "targetable": [{"attribute": "on", "rule": "any_cuts"}],
             "modes": [{"name": "stuck_off", "nature": "failure", "effects": [{"attribute": "on", "value": false}]}]},
            {"name": "Lamp", "uses": [{"fragment": "lit_from_port"}]},
            {"name": "Faulty", "uses": [{"fragment": "broken"}]},
            {"name": "Clash",
             "body": {"attributes": [{"name": "on", "kind": "bool", "init": boolean(true)}],
                      "sensitive_functions": [{"name": "modes__on", "effects": []}]},
             "targetable": [{"attribute": "on", "rule": "any_cuts"}],
             "modes": [{"name": "m", "nature": "failure", "effects": [{"attribute": "on", "value": false}]}]}
        ]
    }))
}

fn room() -> SystemModel {
    system(
        json!({"name": "room", "library": {"name": "home", "version": "2.0.0"},
        "instances": [{"name": "S", "type": "Switch"}, {"name": "L1", "type": "Lamp"}],
        "connections": [{"from": "S.power_out", "to": "L1.power_in"}],
        "indicators": [{"name": "lit", "target": "attribute", "attr": {"component": "L1", "attribute": "lit"}}]}),
    )
}

fn room_study(name: &str) -> StudyDocument {
    study(json!({"name": name, "system": "room",
        "modes": [{"mode": "stuck_off", "type": "Switch", "occurrence": {"distrib": "exp", "rate": 0.1}}],
        "feared_events": [{"name": "dark", "target": true, "condition": {"op": "cmp", "cmp": "eq",
            "lhs": {"op": "attr", "attr": {"component": "L1", "attribute": "lit"}},
            "rhs": {"op": "const", "value": boolean(false)}}}]}))
}

#[test]
fn every_element_of_the_expansion_has_one_origin() {
    let expansion = expand(&[plant_library()], &room(), Some(&room_study("s1"))).unwrap();
    let p = &expansion.provenance.elements;
    let model = &expansion.model;
    for c in &model.components {
        assert!(
            p.contains_key(&format!("component:{}", c.name)),
            "component {}",
            c.name
        );
        for a in &c.attributes {
            assert!(
                p.contains_key(&format!("attribute:{}.{}", c.name, a.name)),
                "{}.{}",
                c.name,
                a.name
            );
        }
        for port in &c.ports {
            assert!(
                p.contains_key(&format!("port:{}.{}", c.name, port.name)),
                "{}.{}",
                c.name,
                port.name
            );
        }
        for f in &c.sensitive_functions {
            assert!(
                p.contains_key(&format!("function:{}.{}", c.name, f.name)),
                "{}.{}",
                c.name,
                f.name
            );
        }
        for a in &c.automata {
            assert!(
                p.contains_key(&format!("automaton:{}.{}", c.name, a.name)),
                "{}.{}",
                c.name,
                a.name
            );
            for t in &a.transitions {
                assert!(p.contains_key(&format!("transition:{}.{}.{}", c.name, a.name, t.name)));
            }
        }
    }
    assert!(p.contains_key("connection:S.power_out->L1.power_in"));
    assert!(p.contains_key("indicator:lit"));
    assert!(p.contains_key("target:dark"));
    let lit = &p["attribute:L1.lit"];
    assert_eq!(lit.fragment.as_deref(), Some("home.lit_from_port"));
    assert_eq!(lit.instance.as_deref(), Some("L1"));
    assert_eq!(lit.type_name.as_deref(), Some("Lamp"));
    assert_eq!(
        p["automaton:S.stuck_off"].mode.as_deref(),
        Some("stuck_off")
    );
}

#[test]
fn a_core_error_is_restated_with_the_fragment_and_instance_it_comes_from() {
    let s = system(
        json!({"name": "room", "library": {"name": "home", "version": "2.0.0"},
        "instances": [{"name": "F", "type": "Faulty"}]}),
    );
    let err = expand(&[plant_library()], &s, None).unwrap_err();
    assert_eq!(err.document, DocumentKind::Library);
    assert!(
        err.pointer.starts_with("/library/fragments/1"),
        "{}",
        err.pointer
    );
    assert!(err.message.contains("missing"), "{}", err.message);
    assert!(
        err.message.contains("fragment `home.broken`"),
        "{}",
        err.message
    );
    assert!(err.message.contains("instance `F`"), "{}", err.message);
}

#[test]
fn a_generated_function_name_already_taken_by_the_type_is_refused() {
    let s = system(
        json!({"name": "room", "library": {"name": "home", "version": "2.0.0"},
        "instances": [{"name": "C", "type": "Clash"}]}),
    );
    let st = study(json!({"name": "s", "system": "room",
        "modes": [{"mode": "m", "instances": ["C"], "occurrence": {"distrib": "delay", "time": 1.0}}]}));
    let err = expand(&[plant_library()], &s, Some(&st)).unwrap_err();
    assert!(err.message.contains("`modes__on`"), "{}", err.message);
}

#[test]
fn two_studies_of_one_system_give_two_distinct_models() {
    let a = expand(&[plant_library()], &room(), Some(&room_study("s1")))
        .unwrap()
        .model;
    let b = expand(&[plant_library()], &room(), Some(&room_study("s2")))
        .unwrap()
        .model;
    assert_eq!(a.name, "room/s1");
    assert_eq!(b.name, "room/s2");
    assert_ne!(a.to_json().unwrap(), b.to_json().unwrap());
}

#[test]
fn the_expanded_model_records_its_source_through_a_save_and_reload() {
    let model = expand(&[plant_library()], &room(), Some(&room_study("s1")))
        .unwrap()
        .model;
    let source = model.source.clone().unwrap();
    assert_eq!(source.library, "home");
    assert_eq!(source.library_version, "2.0.0");
    assert_eq!(source.system, "room");
    assert_eq!(source.study.as_deref(), Some("s1"));
    assert!(source.embedded.is_empty());
    let reloaded = Model::from_json(&model.to_json().unwrap()).unwrap();
    assert_eq!(reloaded.source, Some(source));
}

#[test]
fn a_model_written_directly_serializes_without_a_source() {
    let mut model = expand(&[plant_library()], &room(), None).unwrap().model;
    model.source = None;
    let text = model.to_json().unwrap();
    assert!(!text.contains("\"source\""), "{text}");
    let again = Model::from_json(&text).unwrap().to_json().unwrap();
    assert_eq!(again, text);
}
