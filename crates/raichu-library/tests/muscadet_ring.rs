//! A recirculation ring through a transmitting volume (`rule_set` `torn`,
//! `volume_flow` and `volume_outlet` `transmits`), and the expander's
//! message when the ring is left untorn.
//!
//! The agreement with the muscadet plugin is characterised in the Python
//! unit suite (`ventilation_ring` and `transmitting_buffer` benches); this
//! file pins the refusal and that a torn ring expands and runs.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_library::{expand, parse_library, parse_system, Library};
use serde_json::{json, Value};

fn mu(fragment: &str, with: Value) -> Value {
    json!({"fragment": format!("muscadet.{fragment}"), "with": with})
}

/// A room of 100 holding 90 of air with a through-path, a fan renewing
/// the air (it draws 25, exhausts 25 and blows 25 back), an atmosphere
/// asking for 50 of exhaust.
fn library(torn: &[&str]) -> Library {
    let room = vec![
        mu(
            "volume_total",
            json!({"volume": "room", "flows": ["AIR"], "capacity": 100.0,
            "initial": "partial"}),
        ),
        mu("continuous_flow_in", json!({"name": "AIR"})),
        mu(
            "volume_flow",
            json!({"volume": "room", "flow": "AIR", "capacity": 100.0,
            "content_init": 90.0, "fill_unbounded": true, "transmits": true}),
        ),
        mu(
            "volume_outlet",
            json!({"volume": "room", "flow": "AIR", "rate": 0.0,
            "empty_automaton": "room_bounds", "transmits": true}),
        ),
        mu(
            "continuous_flow_out",
            json!({"name": "AIR", "held": true,
            "empty_automaton": "room_bounds"}),
        ),
    ];
    let fan = vec![
        mu("continuous_flow_in", json!({"name": "AIR"})),
        mu(
            "rule_set",
            json!({"name": "renewal", "active": "go", "idle": "none",
            "consumes": ["AIR"], "produces": ["exhaust", "AIR"], "always_on": true, "torn": torn}),
        ),
        mu(
            "rule_consumes",
            json!({"rule_set": "renewal", "active": "go", "flow": "AIR",
            "rate": 25.0, "always_on": true}),
        ),
        mu(
            "rule_produces",
            json!({"rule_set": "renewal", "active": "go", "flow": "exhaust",
            "rate": 25.0, "always_on": true}),
        ),
        mu(
            "rule_produces",
            json!({"rule_set": "renewal", "active": "go", "flow": "AIR",
            "rate": 25.0, "always_on": true}),
        ),
        mu(
            "continuous_flow_out",
            json!({"name": "exhaust", "available": "produced"}),
        ),
        mu(
            "continuous_flow_out",
            json!({"name": "AIR", "available": "produced"}),
        ),
    ];
    let atmosphere = vec![
        mu(
            "continuous_flow_in",
            json!({"name": "exhaust", "demand": 50.0}),
        ),
        mu(
            "continuous_sink",
            json!({"name": "exhaust", "demand": 50.0}),
        ),
    ];
    parse_library(
        &json!({"raichu_library": {"format": 1}, "library": {
            "name": "plant", "version": "1.0.0", "imports": ["muscadet"], "types": [
                {"name": "Room", "uses": room},
                {"name": "Fan", "uses": fan},
                {"name": "Atmosphere", "uses": atmosphere}]}})
        .to_string(),
    )
    .unwrap()
}

fn system() -> raichu_library::SystemModel {
    parse_system(
        &json!({"raichu_system": {"format": 1}, "system": {
            "name": "ring", "library": {"name": "plant", "version": "1.0.0"},
            "instances": [{"name": "ROOM", "type": "Room"}, {"name": "FAN", "type": "Fan"},
                          {"name": "SKY", "type": "Atmosphere"}],
            "connections": [{"from": "ROOM.AIR_out", "to": "FAN.AIR_in"},
                            {"from": "FAN.AIR_out", "to": "ROOM.AIR_in"},
                            {"from": "FAN.exhaust_out", "to": "SKY.exhaust_in"}]}})
        .to_string(),
    )
    .unwrap()
}

#[test]
fn an_untorn_ring_is_refused_naming_its_instances_and_the_parameter() {
    let error = expand(&[library(&[])], &system(), None)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    for name in [
        "`ROOM`",
        "`FAN`",
        "`FAN.renewal`",
        "`torn`",
        "muscadet.rule_set",
    ] {
        assert!(error.contains(name), "{name} missing from: {error}");
    }
    // The atmosphere only reads the ring's exhaust.
    assert!(!error.contains("`SKY`"), "{error}");
}

#[test]
fn a_torn_ring_expands_and_runs() {
    let model = expand(&[library(&["AIR"])], &system(), None).unwrap().model;
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max: 10.0,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config).unwrap().run().unwrap();
}
