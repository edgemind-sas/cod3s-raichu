//! Port effects: a transition's edge effect written through one of its
//! component's in ports.
//!
//! A component cannot name its neighbours, so before this construct a
//! transition could not act on one: a production unit whose repair refills
//! the buffer it feeds had to be written with a direct reference to the
//! buffer, which a component type library refuses. An effect whose target
//! names the reserved component `$port` and an in port of its own
//! component writes the attribute every connection feeding that port
//! exports. The construct is lowered to ordinary edge effects when the
//! model is resolved, so the engine loop is untouched.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::Value;
use raichu_model::Model;
use serde_json::{json, Value as Json};

fn float(x: f64) -> Json {
    json!({"kind": "float", "value": x})
}

fn load(body: &Json) -> Result<Model, String> {
    let sealed = Model::seal_json(&body.to_string()).map_err(|e| e.to_string())?;
    Model::from_json(&sealed).map_err(|e| e.to_string())
}

/// A buffer draining at one unit per hour, fed by `units`, each of which
/// fails after one hour and is repaired after two more; its repair refills
/// the buffer to 10. `through_port` writes the refill through the unit's in
/// port; otherwise the unit names the buffer directly (the flat form).
fn plant(units: &[&str], through_port: bool) -> Json {
    let unit = |name: &str| {
        let target = if through_port {
            json!({"component": "$port", "attribute": "buffer_in"})
        } else {
            json!({"component": "buffer", "attribute": "level"})
        };
        json!({
            "name": name,
            "attributes": [],
            "ports": [{"name": "buffer_in", "dir": "in"}],
            "automata": [{
                "name": "health", "states": ["up", "down"], "init": "up",
                "transitions": [
                    {"name": "fail", "source": "up", "targets": ["down"],
                     "distrib": "delay", "time": 1.0},
                    {"name": "repair", "source": "down", "targets": ["up"],
                     "distrib": "delay", "time": 2.0,
                     "effects": [{"target": target,
                                  "value": {"op": "const", "value": float(10.0)}}]}
                ]
            }]
        })
    };
    let mut components = vec![json!({
        "name": "buffer",
        "attributes": [{"name": "level", "kind": "float", "init": float(5.0)}],
        "ports": [{"name": "level_out", "dir": "out", "attr": "level"}],
        "equations": [{"target": "level", "kind": "ode",
                       "expr": {"op": "const", "value": float(-1.0)}}]
    })];
    components.extend(units.iter().map(|name| unit(name)));
    let connections: Vec<Json> = units
        .iter()
        .map(|name| {
            json!({"name": format!("to_{name}"),
                   "from": {"component": "buffer", "port": "level_out"},
                   "to": {"component": name, "port": "buffer_in"}})
        })
        .collect();
    json!({
        "name": "port_effects",
        "components": components,
        "connections": connections,
        "indicators": [{"name": "level", "target": "attribute",
                        "attr": {"component": "buffer", "attribute": "level"}}]
    })
}

/// What a run did: its events (date, transition) and the level at each
/// sample date.
type Run = (Vec<(f64, String)>, Vec<(f64, f64)>);

fn run(model: &Model) -> Run {
    let compiled = CompiledModel::compile(model).expect("model compiles");
    let config = EngineConfig {
        t_max: 6.0,
        samples: vec![0.0, 1.0, 2.5, 3.0, 3.5, 6.0],
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    let events = result
        .events
        .iter()
        .map(|e| (e.time, e.transition.clone()))
        .collect();
    let level = result
        .samples
        .iter()
        .find(|s| s.name == "level")
        .unwrap()
        .points
        .iter()
        .map(|(t, v)| match v {
            Value::Float(f) => (*t, *f),
            other => panic!("not a float: {other:?}"),
        })
        .collect();
    (events, level)
}

#[test]
fn a_repair_through_a_port_refills_like_the_direct_write() {
    let direct = load(&plant(&["unit"], false)).unwrap();
    let through = load(&plant(&["unit"], true)).unwrap();
    let (direct_events, direct_level) = run(&direct);
    let (through_events, through_level) = run(&through);
    assert_eq!(through_events, direct_events);
    assert_eq!(through_level, direct_level);
    // The refill lands at the repair date, t = 3: half an hour later the
    // buffer holds 9.5 where it would hold 1.5 without it.
    let (t, level) = through_level[4];
    assert_eq!(t, 3.5);
    assert!((level - 9.5).abs() < 1e-9, "{level}");
}

#[test]
fn two_connections_write_both_sources() {
    // Two buffers feed one unit: its repair refills both.
    let mut body = plant(&["unit"], true);
    let second = json!({
        "name": "spare",
        "attributes": [{"name": "level", "kind": "float", "init": float(1.0)}],
        "ports": [{"name": "level_out", "dir": "out", "attr": "level"}]
    });
    body["components"].as_array_mut().unwrap().push(second);
    body["connections"].as_array_mut().unwrap().push(json!({
        "name": "spare_to_unit",
        "from": {"component": "spare", "port": "level_out"},
        "to": {"component": "unit", "port": "buffer_in"}
    }));
    body["indicators"].as_array_mut().unwrap().push(json!({
        "name": "spare", "target": "attribute",
        "attr": {"component": "spare", "attribute": "level"}}));
    let model = load(&body).unwrap();
    let resolved = model.with_resolved_connections().unwrap();
    let unit = resolved
        .components
        .iter()
        .find(|c| c.name == "unit")
        .unwrap();
    let targets: Vec<(&str, &str)> = unit.automata[0].transitions[1]
        .effects
        .iter()
        .map(|a| (a.target.component.as_str(), a.target.attribute.as_str()))
        .collect();
    assert_eq!(targets, vec![("buffer", "level"), ("spare", "level")]);
}

#[test]
fn an_unconnected_port_writes_nothing() {
    let mut body = plant(&["unit"], true);
    body["connections"] = json!([]);
    let model = load(&body).unwrap();
    model.validate().unwrap();
    let (_, level) = run(&model);
    // No refill: the buffer drains from 5 at one unit per hour.
    let (t, level) = level[4];
    assert_eq!(t, 3.5);
    assert!((level - 1.5).abs() < 1e-9, "{level}");
}

#[test]
fn the_model_declares_the_feature_and_a_bare_body_is_refused() {
    let body = plant(&["unit"], true);
    let model = load(&body).unwrap();
    let features: Vec<&str> = model.required_features().iter().map(|f| f.name()).collect();
    assert!(features.contains(&"port_effects"), "{features:?}");
    // A body declaring the edge effects but not the port that carries one
    // is what an engine predating the construct would accept: refused,
    // naming the feature.
    let enveloped = json!({
        "raichu_model": {"format": 1, "requires": ["transition_effects"]},
        "model": body
    });
    let refused = Model::from_json(&enveloped.to_string())
        .unwrap_err()
        .to_string();
    assert!(refused.contains("port_effects"), "{refused}");
}

fn refusal(edit: impl FnOnce(&mut Json)) -> String {
    let mut body = plant(&["unit"], true);
    edit(&mut body);
    match load(&body) {
        Err(message) => message,
        Ok(model) => model.validate().unwrap_err().to_string(),
    }
}

#[test]
fn an_effect_through_an_unknown_port_is_refused() {
    let message = refusal(|body| {
        body["components"][1]["automata"][0]["transitions"][1]["effects"][0]["target"]
            ["attribute"] = json!("nowhere");
    });
    assert!(message.contains("nowhere"), "{message}");
    assert!(message.contains("unit"), "{message}");
}

#[test]
fn an_effect_through_an_out_port_is_refused() {
    let message = refusal(|body| {
        body["components"][1]["ports"]
            .as_array_mut()
            .unwrap()
            .push(json!({"name": "flag_out", "dir": "out", "attr": "flag"}));
        body["components"][1]["attributes"] =
            json!([{"name": "flag", "kind": "float", "init": float(0.0)}]);
        body["components"][1]["automata"][0]["transitions"][1]["effects"][0]["target"]
            ["attribute"] = json!("flag_out");
    });
    assert!(message.contains("flag_out"), "{message}");
}

#[test]
fn a_sensitive_function_cannot_write_through_a_port() {
    let message = refusal(|body| {
        body["components"][1]["sensitive_functions"] = json!([{
            "name": "push",
            "effects": [{"target": {"component": "$port", "attribute": "buffer_in"},
                         "value": {"op": "const", "value": float(1.0)}}]
        }]);
    });
    assert!(message.contains("push"), "{message}");
}
