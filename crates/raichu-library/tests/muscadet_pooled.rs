//! Pooled serving out of a mixed volume (`volume_pooled`,
//! `volume_pooled_flow`, `continuous_flow_out` with `available: pooled`).
//!
//! Its agreement with the muscadet plugin is characterised in the Python
//! unit suite (`test_library_std_continuous.py`, the `pooled_*` benches);
//! this file pins the closed form the fragments write, with no plugin.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::Value;
use raichu_library::{expand, parse_library, parse_system};
use serde_json::json;

fn held(flow: &str, content: f64) -> Vec<serde_json::Value> {
    let stock = format!("room_stock_{flow}");
    vec![
        json!({"fragment": "muscadet.continuous_flow_in", "with": {"name": flow}}),
        json!({"fragment": "muscadet.volume_flow", "with": {"volume": "room", "flow": flow,
            "capacity": 100.0, "content_init": content}}),
        json!({"fragment": "muscadet.volume_stock", "with": {"volume": "room", "flow": flow,
            "capacity": 100.0, "initial": "stocked"}}),
        json!({"fragment": "muscadet.volume_outlet", "with": {"volume": "room", "flow": flow,
            "rate": 50.0, "empty_automaton": stock}}),
        json!({"fragment": "muscadet.volume_pooled_flow", "with": {"volume": "room", "flow": flow}}),
        json!({"fragment": "muscadet.continuous_flow_out", "with": {"name": flow,
            "available": "pooled", "held": true, "empty_automaton": stock}}),
    ]
}

/// A full room of 1 hydrogen in 99 of air, hydrogen asked 2 and air 3.
/// Nothing transits it, so the excess is the whole request, 5: hydrogen
/// leaves at 5 h/(h+a), air at its request of 3 (its share, 5 a/(h+a),
/// stays above it), and both run out together.
#[test]
fn the_excess_is_shared_in_proportion_to_the_raw_contents() {
    let mut room = vec![
        json!({"fragment": "muscadet.volume_total", "with": {"volume": "room",
        "flows": ["H2", "AIR"], "capacity": 100.0, "initial": "full",
        "ratio_flows": ["H2", "AIR"]}}),
    ];
    room.extend(held("H2", 1.0));
    room.extend(held("AIR", 99.0));
    room.push(json!({"fragment": "muscadet.volume_pooled",
        "with": {"volume": "room", "flows": ["H2", "AIR"]}}));
    let sink = |flow: &str, demand: f64| {
        json!({"name": format!("Sink{flow}"), "uses": [
            {"fragment": "muscadet.continuous_flow_in", "with": {"name": flow, "demand": demand}},
            {"fragment": "muscadet.continuous_sink", "with": {"name": flow, "demand": demand}}]})
    };
    let library = parse_library(
        &json!({"raichu_library": {"format": 1}, "library": {
            "name": "plant", "version": "1.0.0", "imports": ["muscadet"],
            "types": [{"name": "Room", "uses": room}, sink("H2", 2.0), sink("AIR", 3.0)]}})
        .to_string(),
    )
    .unwrap();
    let attribute = |name: &str, component: &str, attribute: &str| {
        json!({"name": name, "target": "attribute",
               "attr": {"component": component, "attribute": attribute}})
    };
    let system = parse_system(
        &json!({"raichu_system": {"format": 1}, "system": {
            "name": "mixed", "library": {"name": "plant", "version": "1.0.0"},
            "instances": [{"name": "R", "type": "Room"}, {"name": "CH", "type": "SinkH2"},
                          {"name": "CA", "type": "SinkAIR"}],
            "connections": [{"from": "R.H2_out", "to": "CH.H2_in"},
                            {"from": "R.AIR_out", "to": "CA.AIR_in"}],
            "indicators": [attribute("h", "R", "room_content_H2"),
                           attribute("a", "R", "room_content_AIR"),
                           attribute("served_h", "CH", "H2_fed_in"),
                           attribute("served_a", "CA", "AIR_fed_in")]}})
        .to_string(),
    )
    .unwrap();
    let model = expand(&[library], &system, None).unwrap().model;
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max: 30.0,
        samples: vec![1.0, 5.0, 10.0, 20.0, 30.0],
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    let series = |name: &str| -> Vec<(f64, f64)> {
        result
            .samples
            .iter()
            .find(|s| s.name == name)
            .unwrap()
            .points
            .iter()
            .map(|(t, v)| {
                let Value::Float(v) = v else {
                    panic!("float {name}")
                };
                (*t, *v)
            })
            .collect()
    };
    let (h, a, served_h, served_a) = (
        series("h"),
        series("a"),
        series("served_h"),
        series("served_a"),
    );
    assert_eq!(h.len(), 5, "{h:?}");
    for i in 0..h.len() {
        let (t, h) = h[i];
        let a = a[i].1;
        let expected = 5.0 * h / (h + a);
        assert!(
            (served_h[i].1 - expected).abs() < 1e-9,
            "hydrogen {} at {t}, expected {expected}",
            served_h[i].1
        );
        assert!((served_a[i].1 - 3.0).abs() < 1e-9, "air at {t}");
    }
}

/// A mixture group drawing 5 of volume per unit of time out of a full room
/// of 1 hydrogen in 99 of air that nothing feeds: each constituent leaves
/// at 5 times its share of the content, so the shares stay put and the
/// room loses exactly 5 per unit of time.
#[test]
fn a_mixture_group_draws_its_rate_at_the_composition() {
    let drawn = |flow: &str, content: f64| {
        let stock = format!("room_stock_{flow}");
        vec![
            json!({"fragment": "muscadet.continuous_flow_in", "with": {"name": flow}}),
            json!({"fragment": "muscadet.volume_flow", "with": {"volume": "room", "flow": flow,
                "capacity": 100.0, "content_init": content}}),
            json!({"fragment": "muscadet.volume_stock", "with": {"volume": "room", "flow": flow,
                "capacity": 100.0, "initial": "stocked"}}),
            json!({"fragment": "muscadet.volume_outlet", "with": {"volume": "room", "flow": flow,
                "rate": 50.0, "empty_automaton": stock}}),
            json!({"fragment": "muscadet.volume_pooled_flow", "with": {"volume": "room",
                "flow": flow, "drawn": true}}),
            json!({"fragment": "muscadet.continuous_flow_out", "with": {"name": flow,
                "available": "mixture", "held": true, "empty_automaton": stock}}),
        ]
    };
    let mut room = vec![
        json!({"fragment": "muscadet.volume_total", "with": {"volume": "room",
        "flows": ["H2", "AIR"], "capacity": 100.0, "initial": "full",
        "ratio_flows": ["H2", "AIR"]}}),
    ];
    room.extend(drawn("H2", 1.0));
    room.extend(drawn("AIR", 99.0));
    let fan = vec![
        json!({"fragment": "muscadet.continuous_flow_in", "with": {"name": "H2"}}),
        json!({"fragment": "muscadet.continuous_flow_in", "with": {"name": "AIR"}}),
        json!({"fragment": "muscadet.mixture_in", "with": {"name": "extraction",
            "flows": ["H2", "AIR"], "flow_rate": 5.0}}),
    ];
    let library = parse_library(
        &json!({"raichu_library": {"format": 1}, "library": {
            "name": "plant", "version": "1.0.0", "imports": ["muscadet"],
            "types": [{"name": "Room", "uses": room}, {"name": "Fan", "uses": fan}]}})
        .to_string(),
    )
    .unwrap();
    let attribute = |name: &str, component: &str, attribute: &str| {
        json!({"name": name, "target": "attribute",
               "attr": {"component": component, "attribute": attribute}})
    };
    let system = parse_system(
        &json!({"raichu_system": {"format": 1}, "system": {
            "name": "ventilated", "library": {"name": "plant", "version": "1.0.0"},
            "instances": [{"name": "R", "type": "Room"}, {"name": "F", "type": "Fan"}],
            "connections": [{"from": "R.H2_out", "to": "F.H2_in"},
                            {"from": "R.AIR_out", "to": "F.AIR_in"}],
            "indicators": [attribute("h", "R", "room_content_H2"),
                           attribute("a", "R", "room_content_AIR"),
                           attribute("out_h", "F", "H2_fed_in"),
                           attribute("out_a", "F", "AIR_fed_in")]}})
        .to_string(),
    )
    .unwrap();
    let model = expand(&[library], &system, None).unwrap().model;
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max: 10.0,
        samples: vec![1.0, 5.0, 10.0],
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    let at = |name: &str| -> Vec<f64> {
        result
            .samples
            .iter()
            .find(|s| s.name == name)
            .unwrap()
            .points
            .iter()
            .map(|(_, v)| {
                let Value::Float(v) = v else {
                    panic!("float {name}")
                };
                *v
            })
            .collect()
    };
    let (h, a, out_h, out_a) = (at("h"), at("a"), at("out_h"), at("out_a"));
    for (i, t) in [1.0, 5.0, 10.0].into_iter().enumerate() {
        assert!(
            (h[i] + a[i] - (100.0 - 5.0 * t)).abs() < 1e-9,
            "total at {t}"
        );
        assert!((h[i] / (h[i] + a[i]) - 0.01).abs() < 1e-12, "share at {t}");
        assert!((out_h[i] - 0.05).abs() < 1e-12, "hydrogen at {t}");
        assert!((out_a[i] - 4.95).abs() < 1e-12, "air at {t}");
    }
}
