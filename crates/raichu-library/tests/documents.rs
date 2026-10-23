//! Parsing of the three documents: envelopes, strictness, located errors.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_library::document::{library_document, study_document, system_document};
use raichu_library::{parse_library, parse_study, parse_system, DocumentKind};
use serde_json::json;

fn minimal_library() -> serde_json::Value {
    json!({
        "raichu_library": {"format": 1},
        "library": {
            "name": "demo",
            "version": "1.0.0",
            "types": [{
                "name": "Lamp",
                "parameters": [{"name": "power", "kind": "number", "default": 1.0}],
                "body": {"attributes": [
                    {"name": "on", "kind": "bool", "init": {"kind": "bool", "value": true}}
                ]}
            }]
        }
    })
}

fn minimal_system() -> serde_json::Value {
    json!({
        "raichu_system": {"format": 1},
        "system": {
            "name": "room",
            "library": {"name": "demo", "version": "1.0.0"},
            "instances": [{"name": "L1", "type": "Lamp", "parameters": {"power": 2.0}}]
        }
    })
}

fn minimal_study() -> serde_json::Value {
    json!({
        "raichu_study": {"format": 1},
        "study": {
            "name": "s1",
            "system": "room",
            "modes": [{"mode": "burnt", "instances": ["L1"],
                       "occurrence": {"distrib": "exp", "rate": 0.1}}],
            "simulation": {"method": "monte_carlo", "nb_runs": 10, "horizon": 5.0}
        }
    })
}

#[test]
fn the_three_documents_parse_and_round_trip() {
    let library = parse_library(&minimal_library().to_string()).unwrap();
    assert_eq!(
        parse_library(&library_document(&library).to_string()).unwrap(),
        library
    );
    let system = parse_system(&minimal_system().to_string()).unwrap();
    assert_eq!(
        parse_system(&system_document(&system).to_string()).unwrap(),
        system
    );
    let study = parse_study(&minimal_study().to_string()).unwrap();
    assert_eq!(
        parse_study(&study_document(&study).to_string()).unwrap(),
        study
    );
}

#[test]
fn an_unknown_library_key_is_refused_with_its_pointer() {
    let mut doc = minimal_library();
    doc["library"]["typez"] = json!([]);
    let err = parse_library(&doc.to_string()).unwrap_err();
    assert_eq!(err.document, DocumentKind::Library);
    assert_eq!(err.pointer, "/library/typez");
    assert!(err.message.contains("typez"), "{}", err.message);
}

#[test]
fn an_unknown_key_deep_in_a_document_is_located() {
    let mut doc = minimal_library();
    doc["library"]["types"][0]["parameters"][0]["defualt"] = json!(2.0);
    let err = parse_library(&doc.to_string()).unwrap_err();
    assert_eq!(err.pointer, "/library/types/0/parameters/0/defualt");
}

#[test]
fn a_flat_core_model_handed_as_a_system_names_the_expected_envelope() {
    let doc = json!({"raichu_model": {"format": 1, "requires": []}, "model": {}});
    let err = parse_system(&doc.to_string()).unwrap_err();
    assert!(err.message.contains("raichu_model"), "{}", err.message);
    assert!(err.message.contains("raichu_system"), "{}", err.message);
}

#[test]
fn a_library_handed_as_a_study_is_refused() {
    let err = parse_study(&minimal_library().to_string()).unwrap_err();
    assert!(err.message.contains("raichu_library"), "{}", err.message);
}

#[test]
fn another_envelope_revision_is_refused() {
    let mut doc = minimal_system();
    doc["raichu_system"]["format"] = json!(2);
    let err = parse_system(&doc.to_string()).unwrap_err();
    assert_eq!(err.pointer, "/raichu_system/format");
    assert!(err.message.contains("revision 1"), "{}", err.message);
}

#[test]
fn metadata_is_accepted_everywhere_it_is_declared() {
    let mut library = minimal_library();
    library["library"]["metadata"] = json!({"editor": {"x": 1, "nested": [1, 2]}});
    library["library"]["types"][0]["metadata"] = json!({"colour": "red"});
    assert!(parse_library(&library.to_string()).is_ok());
    let mut system = minimal_system();
    system["system"]["instances"][0]["metadata"] = json!({"x": 10, "y": 20});
    assert!(parse_system(&system.to_string()).is_ok());
    let mut study = minimal_study();
    study["study"]["modes"][0]["metadata"] = json!("free text");
    assert!(parse_study(&study.to_string()).is_ok());
}

#[test]
fn instance_names_follow_the_grammar() {
    let mut system = minimal_system();
    system["system"]["instances"][0]["name"] = json!("Algo Feu");
    assert!(parse_system(&system.to_string()).is_ok());
    system["system"]["instances"][0]["name"] = json!("Véhicule");
    assert!(parse_system(&system.to_string()).is_ok());
    system["system"]["instances"][0]["name"] = json!("Srv.main");
    let err = parse_system(&system.to_string()).unwrap_err();
    assert_eq!(err.pointer, "/system/instances/0/name");
    assert!(err.message.contains("`.`"), "{}", err.message);
    system["system"]["instances"][0]["name"] = json!("$x");
    assert!(parse_system(&system.to_string()).is_err());
}

#[test]
fn a_document_with_an_extra_top_level_key_is_refused() {
    let mut doc = minimal_study();
    doc["model"] = json!({});
    let err = parse_study(&doc.to_string()).unwrap_err();
    assert_eq!(err.pointer, "/model");
}
