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

const CONTINUOUS: [&str; 19] = [
    "continuous_flow_out",
    "continuous_flow_in",
    "surplus_release",
    "continuous_source",
    "continuous_sink",
    "time_profile",
    "rate_publisher",
    "pass_through",
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

fn run_plant(
    types: serde_json::Value,
    system: serde_json::Value,
    t_max: f64,
) -> raichu_core::SimulationResult {
    let library = parse_library(
        &json!({"raichu_library": {"format": 1}, "library": {
            "name": "plant", "version": "1.0.0", "imports": ["std"], "types": types}})
        .to_string(),
    )
    .unwrap();
    let system = parse_system(&system.to_string()).unwrap();
    let model = expand(&[library], &system, None).unwrap().model;
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config).unwrap().run().unwrap()
}

fn float_points(result: &raichu_core::SimulationResult, name: &str) -> Vec<(f64, f64)> {
    result
        .indicators
        .iter()
        .find(|s| s.name == name)
        .unwrap()
        .points
        .iter()
        .map(|(t, value)| {
            let Value::Float(v) = value else {
                panic!("float indicator")
            };
            (*t, *v)
        })
        .collect()
}

/// A source of 2 feeding a sink asking 5 through a bus: the bus asks the
/// sink's 5, receives the 2 there is and hands them on.
#[test]
fn a_bus_hands_on_what_arrives_and_asks_what_it_is_asked() {
    let result = run_plant(
        json!([
            {"name": "Source", "uses": [
                {"fragment": "std.continuous_flow_out", "with": {"name": "Q"}},
                {"fragment": "std.continuous_source", "with": {"name": "Q", "rate": 2.0}}]},
            {"name": "Bus", "uses": [
                {"fragment": "std.continuous_flow_in", "with": {"name": "Q"}},
                {"fragment": "std.pass_through", "with": {"name": "Q"}},
                {"fragment": "std.continuous_flow_out", "with": {"name": "Q", "available": "produced"}}]},
            {"name": "Sink", "uses": [
                {"fragment": "std.continuous_flow_in", "with": {"name": "Q", "demand": 5.0}},
                {"fragment": "std.continuous_sink", "with": {"name": "Q", "demand": 5.0}}]}
        ]),
        json!({"raichu_system": {"format": 1}, "system": {
            "name": "bus", "library": {"name": "plant", "version": "1.0.0"},
            "instances": [{"name": "S", "type": "Source"}, {"name": "B", "type": "Bus"},
                          {"name": "K", "type": "Sink"}],
            "connections": [{"from": "S.Q_out", "to": "B.Q_in"}, {"from": "B.Q_out", "to": "K.Q_in"}],
            "indicators": [
                {"name": "asked", "target": "attribute", "attr": {"component": "B", "attribute": "Q_demand_in"}},
                {"name": "received", "target": "attribute", "attr": {"component": "K", "attribute": "Q_fed_in"}}]}}),
        1.0,
    );
    for (_, v) in float_points(&result, "asked") {
        assert_eq!(v, 5.0);
    }
    for (_, v) in float_points(&result, "received") {
        assert_eq!(v, 2.0);
    }
}

/// A source of 4 feeding a store of capacity 10 with an unbounded fill and
/// no consumer: the store takes all 4 and is full at t = 2.5.
#[test]
fn an_unbounded_fill_takes_the_whole_supply() {
    let result = run_plant(
        json!([
            {"name": "Source", "uses": [
                {"fragment": "std.continuous_flow_out", "with": {"name": "Q"}},
                {"fragment": "std.continuous_source", "with": {"name": "Q", "rate": 4.0}}]},
            {"name": "Store", "uses": [
                {"fragment": "std.continuous_flow_in", "with": {"name": "Q"}},
                {"fragment": "std.volume_flow", "with": {"volume": "tank", "flow": "Q",
                    "capacity": 10.0, "fill_unbounded": true, "outlet": false}},
                {"fragment": "std.volume_total", "with": {"volume": "tank", "flows": ["Q"],
                    "capacity": 10.0}}]}
        ]),
        json!({"raichu_system": {"format": 1}, "system": {
            "name": "fill", "library": {"name": "plant", "version": "1.0.0"},
            "instances": [{"name": "S", "type": "Source"}, {"name": "T", "type": "Store"}],
            "connections": [{"from": "S.Q_out", "to": "T.Q_in"}],
            "indicators": [{"name": "content", "target": "attribute",
                            "attr": {"component": "T", "attribute": "tank_content"}}]}}),
        4.0,
    );
    let full: Vec<f64> = result
        .events
        .iter()
        .filter(|e| e.transition.ends_with("tank_reach_full"))
        .map(|e| e.time)
        .collect();
    assert_eq!(full.len(), 1, "{:?}", result.events);
    assert!((full[0] - 2.5).abs() < 1e-9, "full at {}", full[0]);
    for (t, v) in float_points(&result, "content") {
        let expected = (4.0 * t).min(10.0);
        assert!(
            (v - expected).abs() < 1e-9,
            "content {v} at {t}, expected {expected}"
        );
    }
}
