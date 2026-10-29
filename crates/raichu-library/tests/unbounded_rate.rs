//! The unbounded-rate guard of an expanded model.
//!
//! The muscadet plugin declares `unbounded_rate` on every model whose system
//! declares a continuous flow, so the engine refuses by name any integrated
//! rate reaching the magnitude the layer writes for "unbounded". A model
//! expanded from documents applying the standard continuous flow fragments
//! is the same network and must carry the same guard; a hand-written system
//! may state its own magnitude.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_library::{expand, parse_library, parse_system, standard_library, STD_UNBOUNDED_RATE};
use serde_json::{json, Value};

fn library() -> raichu_library::Library {
    parse_library(
        &json!({"raichu_library": {"format": 1}, "library": {
        "name": "plant", "version": "1.0.0", "imports": ["std"],
        "types": [
            {"name": "Source", "uses": [
                {"fragment": "std.continuous_flow_out", "with": {"name": "Q"}},
                {"fragment": "std.continuous_source", "with": {"name": "Q", "rate": 2.0}}]},
            {"name": "Sink", "uses": [
                {"fragment": "std.continuous_flow_in", "with": {"name": "Q", "demand": 2.0}},
                {"fragment": "std.continuous_sink", "with": {"name": "Q", "demand": 2.0}}]},
            {"name": "Switch", "uses": [
                {"fragment": "std.bool_source_out", "with": {"name": "power"}}]}
        ]}})
        .to_string(),
    )
    .unwrap()
}

fn system(instances: Value, connections: Value, extra: Value) -> raichu_library::SystemModel {
    let mut body = json!({
        "name": "net", "library": {"name": "plant", "version": "1.0.0"},
        "instances": instances, "connections": connections,
    });
    for (key, value) in extra.as_object().unwrap() {
        body[key] = value.clone();
    }
    parse_system(&json!({"raichu_system": {"format": 1}, "system": body}).to_string()).unwrap()
}

#[test]
fn a_continuous_network_declares_the_standard_magnitude() {
    let system = system(
        json!([{"name": "S", "type": "Source"}, {"name": "C", "type": "Sink"}]),
        json!([{"from": "S.Q_out", "to": "C.Q_in"}]),
        json!({}),
    );
    let model = expand(&[library()], &system, None).unwrap().model;
    assert_eq!(model.unbounded_rate, Some(STD_UNBOUNDED_RATE));
}

/// A consumer alone is a continuous network for the plugin too: its system
/// declares a continuous flow, supplier or not.
#[test]
fn a_lone_continuous_input_declares_it_too() {
    let system = system(json!([{"name": "C", "type": "Sink"}]), json!([]), json!({}));
    let model = expand(&[library()], &system, None).unwrap().model;
    assert_eq!(model.unbounded_rate, Some(STD_UNBOUNDED_RATE));
}

#[test]
fn a_lone_continuous_output_declares_it_too() {
    let system = system(
        json!([{"name": "S", "type": "Source"}]),
        json!([]),
        json!({}),
    );
    let model = expand(&[library()], &system, None).unwrap().model;
    assert_eq!(model.unbounded_rate, Some(STD_UNBOUNDED_RATE));
}

#[test]
fn a_discrete_system_declares_nothing() {
    let system = system(
        json!([{"name": "P", "type": "Switch"}]),
        json!([]),
        json!({}),
    );
    let model = expand(&[library()], &system, None).unwrap().model;
    assert_eq!(model.unbounded_rate, None);
}

#[test]
fn a_hand_written_system_states_its_own_magnitude() {
    let system = system(
        json!([{"name": "P", "type": "Switch"}]),
        json!([]),
        json!({"unbounded_rate": 1.0e20}),
    );
    let model = expand(&[library()], &system, None).unwrap().model;
    assert_eq!(model.unbounded_rate, Some(1.0e20));
}

#[test]
fn a_continuous_network_keeps_the_magnitude_it_states_when_it_is_the_standard_one() {
    let system = system(
        json!([{"name": "C", "type": "Sink"}]),
        json!([]),
        json!({"unbounded_rate": STD_UNBOUNDED_RATE}),
    );
    let model = expand(&[library()], &system, None).unwrap().model;
    assert_eq!(model.unbounded_rate, Some(STD_UNBOUNDED_RATE));
}

#[test]
fn a_magnitude_contradicting_the_standard_fragments_is_refused() {
    let system = system(
        json!([{"name": "C", "type": "Sink"}]),
        json!([]),
        json!({"unbounded_rate": 1.0e20}),
    );
    let error = expand(&[library()], &system, None).unwrap_err().to_string();
    assert!(error.contains("unbounded_rate"), "{error}");
    assert!(error.contains("1e20") && error.contains("1e30"), "{error}");
}

/// The magnitude the expander declares is the one the standard volume
/// writes for a ceiling nobody set: two spellings of one number would let
/// the guard pass a rate the volume reports as unbounded.
#[test]
fn the_declared_magnitude_is_the_one_std_writes() {
    let std = serde_json::to_string(&standard_library().unwrap()).unwrap();
    let written: Vec<f64> = std
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == 'e' || c == '+' || c == '-'))
        .filter_map(|token| token.parse::<f64>().ok())
        .filter(|value| *value >= 1.0e15)
        .collect();
    assert!(!written.is_empty(), "std writes no unbounded magnitude");
    assert!(
        written.iter().all(|v| *v == STD_UNBOUNDED_RATE),
        "{written:?}"
    );
}
