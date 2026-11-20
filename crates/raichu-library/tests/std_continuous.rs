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

const CONTINUOUS: [&str; 22] = [
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
    "shared_output",
    "rule_cascade",
    "tap",
    "aggregate_measure",
    "threshold_compare",
    "threshold_band",
    "republish",
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

/// A source of 4 feeding a rule always on that turns 2 of water into 3 of
/// gas, for a consumer asking 9: the rule runs at the supply's scale, 2,
/// and delivers 6, with no mode automaton.
#[test]
fn an_always_on_rule_runs_at_the_scale_its_supply_allows() {
    let result = run_plant(
        json!([
            {"name": "Source", "uses": [
                {"fragment": "std.continuous_flow_out", "with": {"name": "W"}},
                {"fragment": "std.continuous_source", "with": {"name": "W", "rate": 4.0}}]},
            {"name": "Machine", "uses": [
                {"fragment": "std.continuous_flow_in", "with": {"name": "W"}},
                {"fragment": "std.rule_set", "with": {"name": "work", "active": "go", "idle": "none",
                    "consumes": ["W"], "produces": ["G"], "always_on": true}},
                {"fragment": "std.rule_consumes", "with": {"rule_set": "work", "active": "go",
                    "flow": "W", "rate": 2.0, "always_on": true}},
                {"fragment": "std.rule_produces", "with": {"rule_set": "work", "active": "go",
                    "flow": "G", "rate": 3.0, "always_on": true}},
                {"fragment": "std.continuous_flow_out", "with": {"name": "G", "available": "produced"}}]},
            {"name": "Sink", "uses": [
                {"fragment": "std.continuous_flow_in", "with": {"name": "G", "demand": 9.0}},
                {"fragment": "std.continuous_sink", "with": {"name": "G", "demand": 9.0}}]}
        ]),
        json!({"raichu_system": {"format": 1}, "system": {
            "name": "steady", "library": {"name": "plant", "version": "1.0.0"},
            "instances": [{"name": "S", "type": "Source"}, {"name": "M", "type": "Machine"},
                          {"name": "K", "type": "Sink"}],
            "connections": [{"from": "S.W_out", "to": "M.W_in"}, {"from": "M.G_out", "to": "K.G_in"}],
            "indicators": [{"name": "gas", "target": "attribute",
                            "attr": {"component": "K", "attribute": "G_fed_in"}}]}}),
        2.0,
    );
    assert!(result.events.is_empty(), "{:?}", result.events);
    for (_, v) in float_points(&result, "gas") {
        assert_eq!(v, 6.0);
    }
}

/// A source of 4 feeding a hopper of 10 upstream of a rule always on that
/// draws 1 for a consumer asking 1: the hopper fills without bound, so it
/// takes all 4 and fills at 3 until it is full at t = 10/3, then asks the
/// rule's need alone.
#[test]
fn an_input_side_volume_takes_the_supply_until_full_then_the_need() {
    let result = run_plant(
        json!([
            {"name": "Source", "uses": [
                {"fragment": "std.continuous_flow_out", "with": {"name": "F"}},
                {"fragment": "std.continuous_source", "with": {"name": "F", "rate": 4.0}}]},
            {"name": "Feeder", "uses": [
                {"fragment": "std.continuous_flow_in", "with": {"name": "F"}},
                {"fragment": "std.volume_flow", "with": {"volume": "hopper", "flow": "F",
                    "capacity": 10.0, "fill_unbounded": true, "side": "in",
                    "empty_automaton": "hopper_bounds"}},
                {"fragment": "std.volume_total", "with": {"volume": "hopper", "flows": ["F"],
                    "capacity": 10.0}},
                {"fragment": "std.rule_set", "with": {"name": "burn", "active": "go", "idle": "none",
                    "consumes": [], "from_volume": ["F"], "produces": ["H"], "always_on": true}},
                {"fragment": "std.rule_consumes", "with": {"rule_set": "burn", "active": "go",
                    "flow": "F", "rate": 1.0, "always_on": true, "from_volume": true}},
                {"fragment": "std.rule_produces", "with": {"rule_set": "burn", "active": "go",
                    "flow": "H", "rate": 1.0, "always_on": true}},
                {"fragment": "std.continuous_flow_out", "with": {"name": "H", "available": "produced"}}]},
            {"name": "Sink", "uses": [
                {"fragment": "std.continuous_flow_in", "with": {"name": "H", "demand": 1.0}},
                {"fragment": "std.continuous_sink", "with": {"name": "H", "demand": 1.0}}]}
        ]),
        json!({"raichu_system": {"format": 1}, "system": {
            "name": "hopper", "library": {"name": "plant", "version": "1.0.0"},
            "instances": [{"name": "S", "type": "Source"}, {"name": "F", "type": "Feeder"},
                          {"name": "K", "type": "Sink"}],
            "connections": [{"from": "S.F_out", "to": "F.F_in"}, {"from": "F.H_out", "to": "K.H_in"}],
            "indicators": [
                {"name": "content", "target": "attribute",
                 "attr": {"component": "F", "attribute": "hopper_content"}},
                {"name": "heat", "target": "attribute",
                 "attr": {"component": "K", "attribute": "H_fed_in"}}]}}),
        5.0,
    );
    let full: Vec<f64> = result
        .events
        .iter()
        .filter(|e| e.transition.ends_with("hopper_reach_full"))
        .map(|e| e.time)
        .collect();
    assert!(!full.is_empty(), "{:?}", result.events);
    assert!((full[0] - 10.0 / 3.0).abs() < 1e-9, "full at {}", full[0]);
    for (t, v) in float_points(&result, "content") {
        let expected = (3.0 * t).min(10.0);
        assert!(
            (v - expected).abs() < 1e-6,
            "content {v} at {t}, expected {expected}"
        );
    }
    for (_, v) in float_points(&result, "heat") {
        assert_eq!(v, 1.0);
    }
}

/// Two rules always on produce one exhaust for a consumer asking 3: the
/// first, fed 2, is asked the whole demand and makes 2; the second, fed 5,
/// is asked only the 1 left, so it draws 1 and the consumer receives 3.
#[test]
fn a_cascade_asks_the_second_rule_set_what_the_first_leaves() {
    let source = |name: &str, flow: &str, rate: f64| {
        json!({"name": name, "uses": [
            {"fragment": "std.continuous_flow_out", "with": {"name": flow}},
            {"fragment": "std.continuous_source", "with": {"name": flow, "rate": rate}}]})
    };
    let rule = |set: &str, flow: &str, cascaded: bool| {
        let (produces, cascade) = if cascaded {
            (json!([]), json!(["X"]))
        } else {
            (json!(["X"]), json!([]))
        };
        vec![
            json!({"fragment": "std.continuous_flow_in", "with": {"name": flow}}),
            json!({"fragment": "std.rule_set", "with": {"name": set, "active": "go", "idle": "none",
                "consumes": [flow], "produces": produces, "cascaded": cascade, "always_on": true}}),
            json!({"fragment": "std.rule_consumes", "with": {"rule_set": set, "active": "go",
                "flow": flow, "rate": 1.0, "always_on": true}}),
            json!({"fragment": "std.rule_produces", "with": {"rule_set": set, "active": "go",
                "flow": "X", "rate": 1.0, "always_on": true, "shared": true}}),
        ]
    };
    let mut uses = rule("first", "W1", false);
    uses.extend(rule("second", "W2", true));
    uses.extend([
        json!({"fragment": "std.rule_cascade", "with": {"rule_set": "second", "flow": "X",
            "before": ["first"]}}),
        json!({"fragment": "std.shared_output", "with": {"flow": "X",
            "producers": ["first", "second"]}}),
        json!({"fragment": "std.continuous_flow_out", "with": {"name": "X", "available": "produced"}}),
    ]);
    let result = run_plant(
        json!([
            source("Source1", "W1", 2.0),
            source("Source2", "W2", 5.0),
            {"name": "Vent", "uses": uses},
            {"name": "Sink", "uses": [
                {"fragment": "std.continuous_flow_in", "with": {"name": "X", "demand": 3.0}},
                {"fragment": "std.continuous_sink", "with": {"name": "X", "demand": 3.0}}]}
        ]),
        json!({"raichu_system": {"format": 1}, "system": {
            "name": "cascade", "library": {"name": "plant", "version": "1.0.0"},
            "instances": [{"name": "S1", "type": "Source1"}, {"name": "S2", "type": "Source2"},
                          {"name": "V", "type": "Vent"}, {"name": "K", "type": "Sink"}],
            "connections": [{"from": "S1.W1_out", "to": "V.W1_in"},
                            {"from": "S2.W2_out", "to": "V.W2_in"},
                            {"from": "V.X_out", "to": "K.X_in"}],
            "indicators": [
                {"name": "exhaust", "target": "attribute",
                 "attr": {"component": "K", "attribute": "X_fed_in"}},
                {"name": "second_draw", "target": "attribute",
                 "attr": {"component": "V", "attribute": "W2_demand_in"}}]}}),
        2.0,
    );
    for (_, v) in float_points(&result, "exhaust") {
        assert_eq!(v, 3.0);
    }
    for (_, v) in float_points(&result, "second_draw") {
        assert_eq!(v, 1.0);
    }
}

/// Three sources of 1, 4 and 9 each read by an instrument republishing it,
/// the last with a gain of 2, and a voter taking the median of the three
/// readings, 1, 4 and 18: it reads 4, and the instrument publishes 18.
#[test]
fn a_voter_reads_the_median_of_what_instruments_republish() {
    let instances: Vec<serde_json::Value> = [(1.0, 1.0), (4.0, 1.0), (9.0, 2.0)]
        .iter()
        .enumerate()
        .flat_map(|(i, (rate, gain))| {
            vec![
                json!({"name": format!("S{i}"), "type": "Source", "parameters": {"rate": rate}}),
                json!({"name": format!("CT{i}"), "type": "Instrument", "parameters": {"gain": gain}}),
            ]
        })
        .chain([json!({"name": "AP", "type": "Voter"})])
        .collect();
    let connections: Vec<serde_json::Value> = (0..3)
        .flat_map(|i| {
            vec![
                json!({"from": format!("S{i}.Q_rate_out"), "to": format!("CT{i}.Q_rate_in")}),
                json!({"from": format!("CT{i}.reading_level_out"), "to": "AP.reading_level_in"}),
            ]
        })
        .collect();
    let result = run_plant(
        json!([
            {"name": "Source", "parameters": [{"name": "rate", "kind": "number"}], "uses": [
                {"fragment": "std.continuous_flow_out", "with": {"name": "Q"}},
                {"fragment": "std.continuous_source", "with": {"name": "Q", "rate": {"$param": "rate"}}},
                {"fragment": "std.rate_publisher", "with": {"name": "Q", "of": "capability"}}]},
            {"name": "Instrument", "parameters": [{"name": "gain", "kind": "number"}], "uses": [
                {"fragment": "std.aggregate_measure", "with": {"name": "Q_rate"}},
                {"fragment": "std.republish", "with": {"name": "reading", "input": "Q_rate", "gain": {"$param": "gain"}}}]},
            {"name": "Voter", "uses": [
                {"fragment": "std.aggregate_measure", "with": {"name": "reading_level", "agg": "median"}}]}
        ]),
        json!({"raichu_system": {"format": 1}, "system": {
            "name": "voter", "library": {"name": "plant", "version": "1.0.0"},
            "instances": instances,
            "connections": connections,
            "indicators": [
                {"name": "median", "target": "attribute", "attr": {"component": "AP", "attribute": "reading_level"}},
                {"name": "doubled", "target": "attribute", "attr": {"component": "CT2", "attribute": "reading_level"}}]}}),
        1.0,
    );
    let median = float_points(&result, "median");
    assert!(!median.is_empty());
    for (_, v) in median {
        assert_eq!(v, 4.0);
    }
    for (_, v) in float_points(&result, "doubled") {
        assert_eq!(v, 18.0);
    }
}

/// A component `P` whose attribute `f` follows `std.time_profile` with
/// the parameters `with` (`target` and `initial` given here), plus, when
/// `failing`, a repairable failure automaton drawing exponential dates.
fn profile_model(with: serde_json::Value, failing: bool) -> raichu_model::Model {
    try_profile_model(with, failing).unwrap()
}

/// [`profile_model`], the expansion error as text.
fn try_profile_model(
    with: serde_json::Value,
    failing: bool,
) -> Result<raichu_model::Model, String> {
    let mut with = with;
    with["target"] = json!("f");
    let mut uses = vec![json!({"fragment": "std.time_profile", "with": with})];
    if with.get("period").is_none() {
        uses.clear();
    }
    let body = if failing {
        json!({"automata": [{"name": "life", "states": ["ok", "ko"], "init": "ok",
            "transitions": [
                {"name": "fail", "source": "ok", "targets": ["ko"], "distrib": "exp", "rate": 0.3},
                {"name": "repair", "source": "ko", "targets": ["ok"], "distrib": "exp", "rate": 0.7}]}]})
    } else {
        json!({})
    };
    let library = parse_library(
        &json!({"raichu_library": {"format": 1}, "library": {
            "name": "plant", "version": "1.0.0", "imports": ["std"],
            "types": [{"name": "Profiled", "uses": uses, "body": body}]}})
        .to_string(),
    )
    .unwrap();
    let indicators = if uses.is_empty() {
        json!([])
    } else {
        json!([{"name": "f", "target": "attribute", "attr": {"component": "P", "attribute": "f"}}])
    };
    let system = parse_system(
        &json!({"raichu_system": {"format": 1}, "system": {
            "name": "profiled", "library": {"name": "plant", "version": "1.0.0"},
            "instances": [{"name": "P", "type": "Profiled"}],
            "indicators": indicators}})
        .to_string(),
    )
    .unwrap();
    expand(&[library], &system, None)
        .map(|expanded| expanded.model)
        .map_err(|error| error.to_string())
}

/// The profile `f` sampled at `samples` on replica stream `stream`.
fn sample_profile(with: serde_json::Value, samples: &[f64], seed: u64, stream: u64) -> Vec<f64> {
    let model = profile_model(with, false);
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max: samples.last().copied().unwrap_or(1.0) + 1.0,
        samples: samples.to_vec(),
        seed,
        rng_stream: stream,
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    let series = result.samples.iter().find(|s| s.name == "f").unwrap();
    assert_eq!(series.points.len(), samples.len());
    series
        .points
        .iter()
        .map(|(_, value)| match value {
            Value::Float(v) => *v,
            other => panic!("float profile, got {other:?}"),
        })
        .collect()
}

/// The ramped square wave of the reference profile: low 2, high 8 for 8
/// hours, 5-minute ramps, a 24-hour period shifted by 8 hours, written as
/// the four points of one period.
fn square_profile() -> serde_json::Value {
    let ramp = 5.0 / 60.0;
    json!({"shape": "piecewise_linear", "period": 24.0, "phase_shift": 8.0,
        "points": [[0.0, 2.0], [ramp, 8.0], [ramp + 8.0, 8.0], [2.0 * ramp + 8.0, 2.0]],
        "value_min": 0.0, "initial": 2.0})
}

#[test]
fn a_square_profile_ramps_holds_its_plateau_and_falls_back() {
    let ramp = 5.0 / 60.0;
    let samples = [
        2.0,
        8.0 + 2.5 / 60.0,
        8.0 + ramp + 4.0,
        20.0,
        32.0 + 2.5 / 60.0,
    ];
    let values = sample_profile(square_profile(), &samples, 1, 0);
    for (got, expected) in values.iter().zip([2.0, 5.0, 8.0, 2.0, 5.0]) {
        assert!((got - expected).abs() < 1e-9, "{values:?}");
    }
}

#[test]
fn a_drawn_profile_is_continuous_across_its_period_and_periodic() {
    let drawn = json!({"shape": "piecewise_linear", "period": 24.0, "phase_shift": 0.0,
        "points": [[0.0, 1.0], [6.0, 3.0], [15.0, 0.5]], "value_min": 0.0, "initial": 1.0});
    let samples = [3.0, 19.5, 24.0 - 1e-6, 24.0, 24.0 + 1e-6, 27.0, 43.5];
    let v = sample_profile(drawn, &samples, 1, 0);
    assert!((v[0] - 2.0).abs() < 1e-9, "{v:?}");
    assert!((v[1] - 0.75).abs() < 1e-9, "{v:?}");
    for edge in &v[2..5] {
        assert!((edge - 1.0).abs() < 1e-6, "{v:?}");
    }
    assert!((v[5] - v[0]).abs() < 1e-9, "{v:?}");
    assert!((v[6] - v[1]).abs() < 1e-9, "{v:?}");
}

#[test]
fn an_ill_formed_profile_is_refused_naming_what_is_wrong() {
    let drawn = |points: serde_json::Value| {
        json!({"shape": "piecewise_linear", "period": 24.0, "phase_shift": 0.0,
            "points": points, "value_min": 0.0})
    };
    let refused = |with: serde_json::Value| try_profile_model(with, false).unwrap_err();
    let error = refused(drawn(json!([[0.0, 1.0], [30.0, 3.0]])));
    assert!(
        error.contains("piecewise_linear in equation of `P.f`"),
        "{error}"
    );
    assert!(error.contains("not below the period"), "{error}");
    let error = refused(drawn(json!([[1.0, 1.0]])));
    assert!(error.contains("first instant must be 0"), "{error}");
    let error = refused(drawn(json!([[0.0, 1.0], [5.0, 2.0], [5.0, 3.0]])));
    assert!(error.contains("strictly increasing"), "{error}");
    let error = refused(drawn(json!([])));
    assert!(error.contains("at least one"), "{error}");
    let error = refused(drawn(json!([[0.0, 1.0, 2.0]])));
    assert!(error.contains("expected 2 elements"), "{error}");
    let mut noisy = noisy_profile(1);
    noisy["noise_time_scale"] = json!(0.0);
    let error = refused(noisy);
    assert!(
        error.contains("smooth_noise") && error.contains("time_scale"),
        "{error}"
    );
}

#[test]
fn a_sine_profile_with_a_ceiling_is_clipped_on_both_sides() {
    let sine = json!({"amplitude": 5.0, "offset": 5.0, "period": 24.0, "phase_shift": 0.0,
        "value_min": 1.0, "value_max": 8.0, "initial": 5.0});
    let samples: Vec<f64> = (0..48).map(|i| f64::from(i) * 0.5).collect();
    let values = sample_profile(sine, &samples, 1, 0);
    let (mut floor, mut ceiling) = (0, 0);
    for (t, got) in samples.iter().zip(&values) {
        let raw = 5.0 * (std::f64::consts::TAU * t / 24.0).sin() + 5.0;
        let expected = raw.clamp(1.0, 8.0);
        assert!(
            (got - expected).abs() < 1e-9,
            "{got} at {t}, expected {expected}"
        );
        floor += usize::from(raw < 1.0);
        ceiling += usize::from(raw > 8.0);
    }
    assert!(floor > 0 && ceiling > 0, "both bounds bite");
}

/// The noise of the reference profile: amplitude 0.5 around the square
/// wave, nodes every 2 hours.
fn noisy_profile(key: u64) -> serde_json::Value {
    let mut profile = square_profile();
    profile["noise"] = json!("smooth");
    profile["noise_amplitude"] = json!(0.5);
    profile["noise_time_scale"] = json!(2.0);
    profile["noise_key"] = json!(key);
    profile
}

#[test]
fn a_noisy_profile_is_reproducible_per_replica_and_differs_between_replicas() {
    // A key above the signed 64-bit range, as half the flow hashes are.
    let key = 0xCBF2_9CE4_8422_2325;
    let samples: Vec<f64> = (0..40).map(|i| 0.75 + f64::from(i) * 1.3).collect();
    let first = sample_profile(noisy_profile(key), &samples, 7, 3);
    assert_eq!(first, sample_profile(noisy_profile(key), &samples, 7, 3));
    assert_ne!(first, sample_profile(noisy_profile(key), &samples, 7, 4));
    assert_ne!(first, sample_profile(noisy_profile(key), &samples, 8, 3));
    assert_ne!(
        first,
        sample_profile(noisy_profile(key + 1), &samples, 7, 3)
    );
    let quiet = sample_profile(square_profile(), &samples, 7, 3);
    assert_ne!(first, quiet);
    // The noise is null at its first node, so every replica starts from
    // the shape without noise.
    let start = [0.0, 1e-6];
    for stream in 0..5 {
        let noisy = sample_profile(noisy_profile(key), &start, 7, stream);
        assert!((noisy[0] - 2.0).abs() < 1e-12, "{noisy:?}");
        assert!((noisy[1] - 2.0).abs() < 1e-6, "{noisy:?}");
    }
}

#[test]
fn a_noisy_profile_is_continuous() {
    let samples: Vec<f64> = (1..200)
        .flat_map(|i| {
            let t = f64::from(i) * 0.37;
            [t, t + 1e-9]
        })
        .collect();
    let values = sample_profile(noisy_profile(17), &samples, 3, 0);
    for pair in values.chunks(2) {
        assert!((pair[1] - pair[0]).abs() < 1e-6, "{pair:?}");
    }
}

/// The failure dates of the repairable component, with or without a
/// profile beside it.
fn failure_dates(profile: Option<serde_json::Value>, stream: u64) -> Vec<(String, f64)> {
    let with = profile.unwrap_or_else(|| json!({}));
    let model = profile_model(with, true);
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max: 200.0,
        seed: 11,
        rng_stream: stream,
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    result
        .events
        .iter()
        .map(|e| (e.transition.clone(), e.time))
        .collect()
}

#[test]
fn a_noise_draws_nothing_from_the_transition_dates() {
    for stream in 0..4 {
        let bare = failure_dates(None, stream);
        assert!(bare.len() > 10, "{bare:?}");
        assert_eq!(bare, failure_dates(Some(square_profile()), stream));
        assert_eq!(bare, failure_dates(Some(noisy_profile(5)), stream));
    }
}

#[test]
fn a_restarted_trajectory_keeps_the_noise_of_its_replica() {
    let model = profile_model(noisy_profile(23), false);
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max: 100.0,
        seed: 5,
        rng_stream: 2,
        ..EngineConfig::default()
    };
    let mut original = Engine::new(&compiled, config.clone()).unwrap();
    original.advance_to(10.0).unwrap();
    let snapshot = original.snapshot().unwrap();
    original.advance_to(17.3).unwrap();
    let mut clone =
        Engine::restart_from_snapshot(&compiled, config.clone(), &snapshot, 99).unwrap();
    clone.advance_to(17.3).unwrap();
    assert_eq!(original.attribute("P.f"), clone.attribute("P.f"));
    // While a fresh replica on the clone's stream has another noise.
    let mut other = Engine::new(
        &compiled,
        EngineConfig {
            rng_stream: 99,
            ..config
        },
    )
    .unwrap();
    other.advance_to(17.3).unwrap();
    assert_ne!(original.attribute("P.f"), other.attribute("P.f"));
}
