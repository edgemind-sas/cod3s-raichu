//! The continuous fragments of the standard library (unit U14).
//!
//! Their agreement with the muscadet plugin is characterised bench by bench
//! in the Python unit suite (`test_library_std_continuous.py`); this file
//! pins what needs no plugin: the fragments exist and describe themselves,
//! and a plant built from them meets a closed form.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::Value;
use raichu_library::{expand, parse_library, parse_system, standard_library};
use serde_json::json;

const CONTINUOUS: [&str; 18] = [
    "continuous_flow_out",
    "continuous_flow_in",
    "surplus_release",
    "continuous_source",
    "continuous_sink",
    "time_profile",
    "rate_publisher",
    "volume_flow",
    "volume_outlet",
    "volume_stock",
    "volume_total",
    "rule_set",
    "rule_consumes",
    "rule_produces",
    "tap",
    "aggregate_measure",
    "threshold_compare",
    "threshold_band",
];

#[test]
fn every_continuous_fragment_is_in_std_and_says_what_it_does() {
    let std = standard_library().unwrap();
    for name in CONTINUOUS {
        let fragment = std
            .fragments
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("std has no fragment `{name}`"));
        let description = fragment.description.as_deref().unwrap_or_default();
        assert!(description.len() > 40, "`{name}` is not described");
    }
}

/// A source of 2 per hour feeding a store of capacity 10 that nobody draws
/// from: the store asks its nominal demand of 2, receives it, and its
/// content is 2 t until it is full, at t = 5 exactly.
#[test]
fn a_store_fills_at_its_feed_rate_and_reaches_its_bound_on_time() {
    let library = parse_library(
        &json!({"raichu_library": {"format": 1}, "library": {
        "name": "plant", "version": "1.0.0", "imports": ["std"],
        "types": [
            {"name": "Source", "uses": [
                {"fragment": "std.continuous_flow_out", "with": {"name": "Q"}},
                {"fragment": "std.continuous_source", "with": {"name": "Q", "rate": 2.0}}]},
            {"name": "Store", "uses": [
                {"fragment": "std.continuous_flow_in", "with": {"name": "Q", "demand": 2.0}},
                {"fragment": "std.volume_flow", "with": {"volume": "tank", "flow": "Q",
                    "capacity": 10.0, "demand": 2.0}},
                {"fragment": "std.volume_outlet", "with": {"volume": "tank", "flow": "Q",
                    "rate": 2.0, "empty_automaton": "tank_bounds"}},
                {"fragment": "std.continuous_flow_out", "with": {"name": "Q", "held": true,
                    "empty_automaton": "tank_bounds"}},
                {"fragment": "std.volume_total", "with": {"volume": "tank", "flows": ["Q"],
                    "capacity": 10.0}}]}
        ]}})
        .to_string(),
    )
    .unwrap();
    let system = parse_system(
        &json!({"raichu_system": {"format": 1}, "system": {
            "name": "fill", "library": {"name": "plant", "version": "1.0.0"},
            "instances": [{"name": "S", "type": "Source"}, {"name": "T", "type": "Store"}],
            "connections": [{"from": "S.Q_out", "to": "T.Q_in"}],
            "indicators": [{"name": "content", "target": "attribute",
                            "attr": {"component": "T", "attribute": "tank_content"}}]}})
        .to_string(),
    )
    .unwrap();
    let model = expand(&[library], &system, None).unwrap().model;
    assert!(
        model.evaluation_order.is_some(),
        "a continuous plant gets an order"
    );
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max: 8.0,
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();

    let full: Vec<f64> = result
        .events
        .iter()
        .filter(|e| e.transition.ends_with("tank_reach_full"))
        .map(|e| e.time)
        .collect();
    assert_eq!(full.len(), 1, "{:?}", result.events);
    assert!((full[0] - 5.0).abs() < 1e-9, "full at {}", full[0]);

    let content = result
        .indicators
        .iter()
        .find(|s| s.name == "content")
        .unwrap();
    for (t, value) in &content.points {
        let Value::Float(v) = value else {
            panic!("float content")
        };
        let expected = (2.0 * t).min(10.0);
        assert!(
            (v - expected).abs() < 1e-9,
            "content {v} at {t}, expected {expected}"
        );
    }
}
