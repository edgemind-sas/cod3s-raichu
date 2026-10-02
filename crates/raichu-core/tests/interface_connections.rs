//! Interface connections: two interfaces joined at once, paired by port
//! name, flows running in both directions.
//!
//! An interface connection is an authoring form. The engine runs on the
//! port connections it stands for, so the decisive property is that a
//! model written with one runs exactly as the same model written port by
//! port, and that a pairing that does not close is refused when the model
//! is built rather than leaving a port unfed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, SimulationResult};
use raichu_expr::Value;
use raichu_model::{
    Feature, Interface, InterfaceConnection, InterfaceRef, LoadError, Model, ModelError,
};
use serde_json::{json, Value as Json};

#[path = "support/unfed_standby.rs"]
mod unfed_standby;

fn float(name: &str, value: f64) -> Json {
    json!({"name": name, "kind": "float", "init": {"kind": "float", "value": value}})
}

fn received(target: &str, port: &str, component: &str) -> Json {
    json!({"target": target, "kind": "explicit", "expr": {
        "op": "port_agg", "agg": "sum",
        "port": {"component": component, "port": port}}})
}

/// `A` sends `a` on `p` and receives on `q`; `B` the other way round. The
/// two interfaces list their ports in different orders on purpose: the
/// pairing is by name, never by position.
fn body(wiring: Json) -> Json {
    let mut body = json!({
        "name": "two_way",
        "components": [
            {"name": "A",
             "attributes": [float("a", 3.0), float("from_b", 0.0)],
             "ports": [{"name": "p", "dir": "out", "attr": "a"},
                       {"name": "q", "dir": "in"}],
             "interfaces": [{"name": "bus", "ports": ["p", "q"]}],
             "equations": [received("from_b", "q", "A")]},
            {"name": "B",
             "attributes": [float("b", 5.0), float("from_a", 0.0)],
             "ports": [{"name": "q", "dir": "out", "attr": "b"},
                       {"name": "p", "dir": "in"}],
             "interfaces": [{"name": "bus", "ports": ["q", "p"]}],
             "equations": [received("from_a", "p", "B")]}
        ],
        "indicators": [
            {"name": "a_received", "target": "attribute",
             "attr": {"component": "A", "attribute": "from_b"}},
            {"name": "b_received", "target": "attribute",
             "attr": {"component": "B", "attribute": "from_a"}}
        ]
    });
    for (key, value) in wiring.as_object().unwrap() {
        body[key] = value.clone();
    }
    body
}

fn by_ports() -> Json {
    body(json!({"connections": [
        {"from": {"component": "A", "port": "p"}, "to": {"component": "B", "port": "p"}},
        {"from": {"component": "B", "port": "q"}, "to": {"component": "A", "port": "q"}}
    ]}))
}

fn by_interface(from: &str, to: &str) -> Json {
    body(json!({"interface_connections": [
        {"from": {"component": from, "interface": "bus"},
         "to": {"component": to, "interface": "bus"}}
    ]}))
}

fn sealed(body: &Json) -> String {
    json!({"raichu_model": {"format": 1, "requires": ["interface_connections"]},
           "model": body})
    .to_string()
}

fn load(body: &Json) -> Model {
    Model::from_json(&sealed(body)).expect("model document loads")
}

fn run(model: &Model) -> SimulationResult {
    let compiled = CompiledModel::compile(model).expect("model compiles");
    let config = EngineConfig {
        t_max: 1.0,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config).unwrap().run().unwrap()
}

fn last(result: &SimulationResult, indicator: &str) -> f64 {
    let series = result
        .indicators
        .iter()
        .find(|s| s.name == indicator)
        .unwrap_or_else(|| panic!("indicator `{indicator}` recorded"));
    match series.points.last().expect("at least one point").1 {
        Value::Float(f) => f,
        other => panic!("`{indicator}` is not a float: {other:?}"),
    }
}

#[test]
fn an_interface_connection_carries_both_directions() {
    let result = run(&load(&by_interface("A", "B")));
    assert_eq!(last(&result, "a_received"), 5.0);
    assert_eq!(last(&result, "b_received"), 3.0);
}

#[test]
fn either_side_may_be_named_first() {
    let result = run(&load(&by_interface("B", "A")));
    assert_eq!(last(&result, "a_received"), 5.0);
    assert_eq!(last(&result, "b_received"), 3.0);
}

#[test]
fn it_runs_exactly_as_the_port_connections_it_stands_for() {
    let explicit = run(&load(&by_ports()));
    let grouped = run(&load(&by_interface("A", "B")));
    assert_eq!(explicit.indicators, grouped.indicators);
}

#[test]
fn the_expansion_follows_the_from_interface_order() {
    let model = load(&by_interface("A", "B"));
    let resolved = model.resolved_connections().unwrap();
    let pairs: Vec<_> = resolved
        .iter()
        .map(|c| {
            (
                c.from.component.as_str(),
                c.from.port.as_str(),
                c.to.component.as_str(),
            )
        })
        .collect();
    assert_eq!(pairs, [("A", "p", "B"), ("B", "q", "A")]);
}

#[test]
fn explicit_connections_come_first_and_stay_as_written() {
    let mut body = by_interface("A", "B");
    body["components"][1]["ports"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name": "extra", "dir": "in"}));
    body["connections"] = json!([
        {"from": {"component": "A", "port": "p"},
         "to": {"component": "B", "port": "extra"}}
    ]);
    let model = load(&body);
    let resolved = model.resolved_connections().unwrap();
    assert_eq!(resolved.len(), 3);
    assert_eq!(resolved[0].to.port, "extra");
    assert_eq!(model.connections.len(), 1, "the document is not rewritten");
}

#[test]
fn a_model_without_interface_connections_borrows_its_list() {
    let model = load(&by_ports());
    assert!(matches!(
        model.resolved_connections().unwrap(),
        std::borrow::Cow::Borrowed(_)
    ));
}

#[test]
fn the_construct_must_be_declared_in_the_envelope() {
    let bare = by_interface("A", "B").to_string();
    let refused = Model::from_json(&bare).expect_err("an undeclared construct is refused");
    assert!(matches!(
        refused,
        LoadError::FeatureNotDeclared {
            feature: "interface_connections",
            ..
        }
    ));
    assert!(load(&by_interface("A", "B"))
        .required_features()
        .contains(&Feature::InterfaceConnections));
    assert!(!load(&by_ports())
        .required_features()
        .contains(&Feature::InterfaceConnections));
}

#[test]
fn a_document_round_trips_in_its_authored_form() {
    let model = load(&by_interface("A", "B"));
    let again = Model::from_json(&model.to_json().unwrap()).unwrap();
    assert_eq!(again, model);
    assert!(again.connections.is_empty());
}

#[test]
fn an_unknown_interface_is_refused() {
    let mut body = by_interface("A", "B");
    body["interface_connections"][0]["to"]["interface"] = json!("nope");
    let refused = load(&body).validate().unwrap_err();
    assert!(
        matches!(&refused, ModelError::InterfaceConnectionUnknown { component, interface, side: "to" }
            if component == "B" && interface == "nope"),
        "{refused}"
    );
}

#[test]
fn a_port_with_no_partner_of_the_same_name_is_refused() {
    let mut body = by_interface("A", "B");
    body["components"][1]["ports"][1]["name"] = json!("p2");
    body["components"][1]["interfaces"][0]["ports"] = json!(["q", "p2"]);
    body["components"][1]["equations"][0]["expr"]["port"]["port"] = json!("p2");
    let refused = load(&body).validate().unwrap_err();
    assert!(
        matches!(&refused, ModelError::InterfacePortUnmatched { component, port, .. }
            if component == "B" && port == "p2"),
        "{refused}"
    );
}

#[test]
fn a_pair_of_the_same_direction_is_refused() {
    let mut body = by_interface("A", "B");
    body["components"][1]["ports"][1] = json!({"name": "p", "dir": "out", "attr": "b"});
    let refused = load(&body).validate().unwrap_err();
    assert!(
        matches!(&refused, ModelError::InterfacePortDirection { port, dir: "out", .. }
            if port == "p"),
        "{refused}"
    );
}

/// The cold standby of the unfed-trigger tests, its trigger left
/// unwired, with an interface on each side whose ports share a name.
fn standby_with_interfaces() -> Model {
    let mut model = unfed_standby::standby("or", false, "up");
    model.components[0].ports[0].name = "cooling_trigger_in".into();
    for component in &mut model.components {
        component.interfaces = vec![Interface {
            name: "cooling".into(),
            ports: vec!["cooling_trigger_in".into()],
        }];
    }
    model
}

#[test]
fn an_interface_connection_feeds_its_ports_for_the_unfed_diagnostic() {
    // Negative control first: the same model with nothing joining the
    // interfaces is reported, so the silence below is the wiring's.
    let unwired = standby_with_interfaces();
    assert_eq!(raichu_core::unfed_triggers(&unwired).len(), 1);

    let mut grouped = standby_with_interfaces();
    grouped.interface_connections = vec![InterfaceConnection {
        name: None,
        from: InterfaceRef {
            component: "Main".into(),
            interface: "cooling".into(),
        },
        to: InterfaceRef {
            component: "Backup".into(),
            interface: "cooling".into(),
        },
    }];
    assert!(raichu_core::unfed_triggers(&grouped).is_empty());
}
