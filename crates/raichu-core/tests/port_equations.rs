//! Port equations: an equation declared on a port, evaluated once per
//! connection through it, writing that connection's channel.
//!
//! A component cannot name its connections, so before this construct
//! every per-connection quantity of a flow network (what each consumer
//! asks, what each is offered, what each receives) had to be written by
//! an authoring layer that knew the whole connection list. A port
//! equation lets a component state it locally: the consumer writes its
//! own demand onto every connection it is fed through, the producer
//! writes each connection's claim from that connection's demand, and an
//! aggregation over the producer's own out port reads the total back.
//!
//! The construct is lowered to ordinary explicit equations when the
//! model is resolved, so the engine loop is untouched.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::Value;
use raichu_model::Model;
use serde_json::{json, Value as Json};

fn float(x: f64) -> Json {
    json!({"kind": "float", "value": x})
}

fn attr(component: &str, attribute: &str) -> Json {
    json!({"op": "attr", "attr": {"component": component, "attribute": attribute}})
}

fn load(body: &Json) -> Result<Model, String> {
    let sealed = Model::seal_json(&body.to_string()).map_err(|e| e.to_string())?;
    Model::from_json(&sealed).map_err(|e| e.to_string())
}

/// One supply of `available` and two consumers asking `need_a` and
/// `need_b`. Each consumer writes its need onto the connection that feeds
/// it; the supply claims, per connection, the need truncated at what it
/// has, and publishes the total it is asked.
fn network(available: f64, need_a: f64, need_b: f64) -> Json {
    let consumer = |name: &str, need: f64| {
        json!({
            "name": name,
            "attributes": [
                {"name": "need", "kind": "float", "init": float(need)},
                {"name": "claimed", "kind": "float", "init": float(0.0)}
            ],
            "ports": [{"name": "input", "dir": "in", "equations": [
                {"channel": "demand", "expr": attr(name, "need")}
            ]}],
            "equations": [{"target": "claimed", "kind": "explicit",
                "expr": {"op": "port_agg", "agg": "sum", "channel": "claim",
                         "port": {"component": name, "port": "input"}}}]
        })
    };
    json!({
        "name": "port_equations",
        "components": [
            {
                "name": "supply",
                "attributes": [
                    {"name": "available", "kind": "float", "init": float(available)},
                    {"name": "asked", "kind": "float", "init": float(0.0)},
                    {"name": "consumers", "kind": "float", "init": float(0.0)}
                ],
                "ports": [{"name": "out", "dir": "out", "attr": "available",
                    "channels": [{"name": "demand"}, {"name": "claim"}],
                    "equations": [
                        {"channel": "claim", "expr": {"op": "min", "args": [
                            attr("$edge", "demand"), attr("supply", "available")]}}
                    ]}],
                "equations": [
                    {"target": "asked", "kind": "explicit",
                     "expr": {"op": "port_agg", "agg": "sum", "channel": "demand",
                              "port": {"component": "supply", "port": "out"}}},
                    {"target": "consumers", "kind": "explicit",
                     "expr": {"op": "port_agg", "agg": "count", "channel": "demand",
                              "port": {"component": "supply", "port": "out"}}}
                ]
            },
            consumer("a", need_a),
            consumer("b", need_b)
        ],
        "connections": [
            {"name": "to_a", "from": {"component": "supply", "port": "out"},
             "to": {"component": "a", "port": "input"}},
            {"name": "to_b", "from": {"component": "supply", "port": "out"},
             "to": {"component": "b", "port": "input"}}
        ],
        "indicators": [
            {"name": "a_claimed", "target": "attribute", "attr": {"component": "a", "attribute": "claimed"}},
            {"name": "b_claimed", "target": "attribute", "attr": {"component": "b", "attribute": "claimed"}},
            {"name": "asked", "target": "attribute", "attr": {"component": "supply", "attribute": "asked"}},
            {"name": "consumers", "target": "attribute", "attr": {"component": "supply", "attribute": "consumers"}}
        ]
    })
}

fn final_values(model: &Model) -> Vec<(String, f64)> {
    let compiled = CompiledModel::compile(model).expect("model compiles");
    let config = EngineConfig {
        t_max: 1.0,
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    result
        .indicators
        .iter()
        .map(|series| match series.points.last().unwrap().1 {
            Value::Float(f) => (series.name.clone(), f),
            #[allow(clippy::cast_precision_loss)] // small counts
            Value::Int(i) => (series.name.clone(), i as f64),
            ref other => panic!("not a number: {other:?}"),
        })
        .collect()
}

fn value(values: &[(String, f64)], name: &str) -> f64 {
    values.iter().find(|(n, _)| n == name).unwrap().1
}

#[test]
fn each_connection_gets_its_own_claim_from_its_own_demand() {
    let mut model = load(&network(5.0, 3.0, 7.0)).unwrap();
    model.validate().unwrap();
    // The lowered steps are named like any explicit step, so an order can
    // list them: demands first, then the claims and totals that read them.
    let order: Vec<raichu_expr::AttrRef> = [
        ("supply", "out__demand__to_a"),
        ("supply", "out__demand__to_b"),
        ("supply", "out__claim__to_a"),
        ("supply", "out__claim__to_b"),
        ("supply", "asked"),
        ("supply", "consumers"),
        ("a", "claimed"),
        ("b", "claimed"),
    ]
    .into_iter()
    .map(|(component, attribute)| raichu_expr::AttrRef {
        component: component.to_owned(),
        attribute: attribute.to_owned(),
    })
    .collect();
    model.evaluation_order = Some(order);
    model.validate().unwrap();
    let values = final_values(&model);
    assert_eq!(value(&values, "a_claimed"), 3.0);
    assert_eq!(value(&values, "b_claimed"), 5.0);
    assert_eq!(value(&values, "asked"), 10.0);
    assert_eq!(value(&values, "consumers"), 2.0);
}

#[test]
fn the_model_declares_the_feature_and_a_bare_body_is_refused() {
    let body = network(5.0, 3.0, 7.0);
    let model = load(&body).unwrap();
    let features: Vec<&str> = model.required_features().iter().map(|f| f.name()).collect();
    assert!(features.contains(&"port_equations"), "{features:?}");
    // A body that does not declare it is what an engine predating the
    // construct would see: refused, naming the feature.
    let refused = Model::from_json(&body.to_string()).unwrap_err().to_string();
    assert!(refused.contains("port_equations"), "{refused}");
}

#[test]
fn a_model_without_port_equations_serialises_unchanged() {
    let body = json!({
        "name": "plain",
        "components": [{"name": "C", "attributes": [], "ports": [
            {"name": "out", "dir": "out", "attr": "x", "channels": [{"name": "demand"}]}]}]
    });
    let model = Model::from_json(&body.to_string()).unwrap();
    let written: Json = serde_json::from_str(&model.to_json().unwrap()).unwrap();
    let port = &written["model"]["components"][0]["ports"][0];
    assert!(port.get("equations").is_none(), "{port}");
    assert!(matches!(
        model.with_resolved_connections().unwrap(),
        std::borrow::Cow::Borrowed(_)
    ));
}

fn refusal(edit: impl FnOnce(&mut Json)) -> String {
    let mut body = network(5.0, 3.0, 7.0);
    edit(&mut body);
    match load(&body) {
        Err(message) => message,
        Ok(model) => model.validate().unwrap_err().to_string(),
    }
}

#[test]
fn a_channel_the_producer_does_not_declare_is_refused() {
    let message = refusal(|body| {
        body["components"][1]["ports"][0]["equations"][0]["channel"] = json!("wish");
    });
    assert!(message.contains("wish"), "{message}");
    assert!(message.contains("supply.out"), "{message}");
}

#[test]
fn an_edge_read_of_an_undeclared_channel_is_refused() {
    let message = refusal(|body| {
        body["components"][0]["ports"][0]["equations"][0]["expr"]["args"][0] =
            attr("$edge", "nothing");
    });
    assert!(message.contains("nothing"), "{message}");
}

#[test]
fn an_edge_read_outside_a_port_equation_is_refused() {
    let message = refusal(|body| {
        body["components"][0]["equations"][0]["expr"] = attr("$edge", "demand");
    });
    assert!(message.contains("$edge"), "{message}");
}

#[test]
fn two_writers_of_one_channel_are_refused() {
    let message = refusal(|body| {
        body["components"][0]["ports"][0]["equations"]
            .as_array_mut()
            .unwrap()
            .push(json!({"channel": "demand", "expr": float_expr(1.0)}));
    });
    assert!(message.contains("demand"), "{message}");
    assert!(message.contains("more than one equation"), "{message}");
}

#[test]
fn an_out_port_aggregation_other_than_sum_or_count_is_refused() {
    let message = refusal(|body| {
        body["components"][0]["equations"][0]["expr"]["agg"] = json!("median");
    });
    assert!(message.contains("median"), "{message}");
    assert!(message.contains("supply.out"), "{message}");
}

fn float_expr(x: f64) -> Json {
    json!({"op": "const", "value": float(x)})
}

#[test]
fn port_equations_on_an_unconnected_port_lower_to_nothing() {
    let mut body = network(5.0, 3.0, 7.0);
    body["connections"] = json!([]);
    let model = load(&body).unwrap();
    model.validate().unwrap();
    let values = final_values(&model);
    assert_eq!(value(&values, "a_claimed"), 0.0);
    assert_eq!(value(&values, "asked"), 0.0);
    assert_eq!(value(&values, "consumers"), 0.0);
}

#[test]
fn the_read_relation_sees_the_lowered_steps() {
    let model = load(&network(5.0, 3.0, 7.0)).unwrap();
    let steps: Vec<String> = model
        .evaluation_step_reads()
        .unwrap()
        .into_iter()
        .map(|s| format!("{}.{}", s.component, s.step))
        .collect();
    assert!(
        steps.contains(&"supply.out__claim__to_a".to_owned()),
        "{steps:?}"
    );
    assert!(
        steps.contains(&"supply.out__demand__to_b".to_owned()),
        "{steps:?}"
    );
}

#[test]
fn an_out_port_aggregation_without_a_channel_is_the_old_refusal() {
    // Not the construct: no feature is required, and validation refuses it
    // as it always has.
    let mut body = network(5.0, 3.0, 7.0);
    body["components"][0]["equations"][0]["expr"]
        .as_object_mut()
        .unwrap()
        .remove("channel");
    body["components"][0]["ports"][0]
        .as_object_mut()
        .unwrap()
        .remove("equations");
    for consumer in [1, 2] {
        body["components"][consumer]["ports"][0]
            .as_object_mut()
            .unwrap()
            .remove("equations");
    }
    body["components"][0]["equations"]
        .as_array_mut()
        .unwrap()
        .truncate(1);
    let model = Model::from_json(&body.to_string()).expect("a bare body: no feature needed");
    let refused = model.validate().unwrap_err();
    assert!(
        matches!(refused, raichu_model::ModelError::ExprBadPortAgg { .. }),
        "{refused}"
    );
}

#[test]
fn an_out_port_total_drives_a_state_dependent_rate() {
    // A failure rate proportional to what the supply is asked: the total is
    // read in the transition's rate expression, a slot like any other.
    let mut body = network(5.0, 3.0, 7.0);
    body["components"][0]["automata"] = json!([{
        "name": "life", "states": ["ok", "failed"], "init": "ok",
        "transitions": [{"name": "fail", "source": "ok", "targets": ["failed"],
            "distrib": "exp",
            "rate_expr": {"op": "port_agg", "agg": "sum", "channel": "demand",
                          "port": {"component": "supply", "port": "out"}}}]
    }]);
    let model = load(&body).unwrap();
    assert!(
        model
            .required_features()
            .iter()
            .any(|f| f.name() == "port_equations"),
        "the construct is detected in a rate"
    );
    let compiled = CompiledModel::compile(&model).expect("model compiles");
    let config = EngineConfig {
        t_max: 100.0,
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    // A rate of 10 per unit time fires well before t = 100; a rate read as
    // zero never does.
    assert!(
        result
            .events
            .iter()
            .any(|e| e.transition.ends_with("life.fail")),
        "{:?}",
        result.events
    );
}
