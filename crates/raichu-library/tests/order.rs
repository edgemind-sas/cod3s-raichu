//! The evaluation order derived from an expanded model (unit U12).
//!
//! A type author writes each component's equations in whatever order reads
//! naturally, and instances are laid out in system order, so the positional
//! sweep of an expanded model is an accident of authoring. The expander
//! derives the order instead: a step follows every step whose result it
//! reads, independent steps keep declaration order, and the members of a
//! cycle stay contiguous for the engine's condensation.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::{AttrRef, Value};
use raichu_library::{derive_evaluation_order, expand, parse_library, parse_system};
use raichu_model::Model;
use serde_json::{json, Value as Json};

fn model(body: Json) -> Model {
    let sealed = Model::seal_json(&body.to_string()).expect("model body seals");
    Model::from_json(&sealed).expect("model document loads")
}

fn float(x: f64) -> Json {
    json!({"kind": "float", "value": x})
}

fn attr(component: &str, attribute: &str) -> Json {
    json!({"op": "attr", "attr": {"component": component, "attribute": attribute}})
}

fn constant(x: f64) -> Json {
    json!({"op": "const", "value": float(x)})
}

fn names(order: &[AttrRef]) -> Vec<String> {
    order
        .iter()
        .map(|a| format!("{}.{}", a.component, a.attribute))
        .collect()
}

fn floats(names: &[&str]) -> Json {
    Json::Array(
        names
            .iter()
            .map(|n| json!({"name": n, "kind": "float", "init": float(0.0)}))
            .collect(),
    )
}

/// Final value of an indicator after a run of `model` to t = 1.
fn final_value(model: &Model, indicator: &str) -> f64 {
    let compiled = CompiledModel::compile(model).expect("model compiles");
    let config = EngineConfig {
        t_max: 1.0,
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    let series = result
        .indicators
        .iter()
        .find(|s| s.name == indicator)
        .unwrap();
    match series.points.last().unwrap().1 {
        Value::Float(f) => f,
        ref other => panic!("not a float: {other:?}"),
    }
}

#[test]
fn a_chain_declared_backwards_is_swept_forwards() {
    let m = model(json!({
        "name": "chain",
        "components": [{
            "name": "C",
            "attributes": floats(&["x", "y", "z"]),
            "equations": [
                {"target": "z", "kind": "explicit", "expr": {"op": "add", "args": [attr("C", "y"), constant(1.0)]}},
                {"target": "y", "kind": "explicit", "expr": {"op": "add", "args": [attr("C", "x"), constant(1.0)]}},
                {"target": "x", "kind": "explicit", "expr": constant(5.0)}
            ]
        }],
        "indicators": [{"name": "z", "target": "attribute", "attr": {"component": "C", "attribute": "z"}}]
    }));
    let order = derive_evaluation_order(&m).unwrap().unwrap();
    assert_eq!(names(&order), ["C.x", "C.y", "C.z"]);

    // The order is what makes the sweep right on its first pass.
    let mut ordered = m.clone();
    ordered.evaluation_order = Some(order);
    assert_eq!(final_value(&ordered, "z"), 7.0);
}

#[test]
fn independent_steps_keep_their_declaration_order() {
    let m = model(json!({
        "name": "independent",
        "components": [
            {"name": "B", "attributes": floats(&["b2", "b1"]),
             "equations": [
                {"target": "b2", "kind": "explicit", "expr": constant(2.0)},
                {"target": "b1", "kind": "explicit", "expr": constant(1.0)}]},
            {"name": "A", "attributes": floats(&["a"]),
             "equations": [{"target": "a", "kind": "explicit", "expr": constant(3.0)}]}
        ]
    }));
    let order = derive_evaluation_order(&m).unwrap().unwrap();
    assert_eq!(names(&order), ["B.b2", "B.b1", "A.a"]);
}

#[test]
fn a_linear_loop_stays_contiguous_and_compiles() {
    // w reads x and is declared first; x and y read each other.
    let m = model(json!({
        "name": "loop",
        "components": [{
            "name": "C",
            "attributes": floats(&["w", "x", "y"]),
            "equations": [
                {"target": "w", "kind": "explicit", "expr": {"op": "add", "args": [attr("C", "x"), constant(0.0)]}},
                {"target": "x", "kind": "explicit", "expr": {"op": "add", "args": [
                    {"op": "mul", "args": [constant(0.5), attr("C", "y")]}, constant(1.0)]}},
                {"target": "y", "kind": "explicit", "expr": {"op": "mul", "args": [constant(0.5), attr("C", "x")]}}
            ]
        }],
        "indicators": [{"name": "w", "target": "attribute", "attr": {"component": "C", "attribute": "w"}}]
    }));
    let order = derive_evaluation_order(&m).unwrap().unwrap();
    // The loop is one block, kept together, and its reader comes after it.
    assert_eq!(names(&order)[2], "C.w");
    let mut block = names(&order)[..2].to_vec();
    block.sort();
    assert_eq!(block, ["C.x", "C.y"]);
    let mut ordered = m.clone();
    ordered.evaluation_order = Some(order);
    // x = 0.5 (0.5 x) + 1, so x = 4/3, solved by the engine's block.
    assert!((final_value(&ordered, "w") - 4.0 / 3.0).abs() < 1e-12);
}

#[test]
fn a_distribution_operator_follows_the_demands_and_precedes_its_readers() {
    // The consumer is declared first and reads what the operator allocates;
    // the operator reads the demand channel written by an equation.
    let m = model(json!({
        "name": "split",
        "components": [
            {"name": "a", "attributes": floats(&["got"]),
             "ports": [{"name": "input", "dir": "in"}],
             "equations": [{"target": "got", "kind": "explicit",
                "expr": {"op": "port_agg", "agg": "sum", "channel": "alloc",
                         "port": {"component": "a", "port": "input"}}}]},
            {"name": "supply",
             "attributes": [{"name": "available", "kind": "float", "init": float(5.0)}],
             "ports": [{"name": "out", "dir": "out", "attr": "available",
                        "channels": [{"name": "demand"}, {"name": "alloc"}]}],
             "equations": [{"target": "out__demand__a", "kind": "explicit", "expr": constant(3.0)}],
             "allocations": [{"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
                              "available": attr("supply", "available"), "policy": "proportional"}]}
        ],
        "connections": [{"name": "a", "from": {"component": "supply", "port": "out"},
                         "to": {"component": "a", "port": "input"}}]
    }));
    let order = derive_evaluation_order(&m).unwrap().unwrap();
    assert_eq!(
        names(&order),
        ["supply.out__demand__a", "supply.split", "a.got"]
    );
}

#[test]
fn a_model_without_explicit_steps_gets_no_order() {
    let m = model(json!({
        "name": "discrete",
        "components": [{"name": "C", "attributes": [{"name": "on", "kind": "bool", "init": {"kind": "bool", "value": true}}]}]
    }));
    assert_eq!(derive_evaluation_order(&m).unwrap(), None);
}

#[test]
fn the_expander_emits_the_derived_order() {
    let library = parse_library(
        &json!({"raichu_library": {"format": 1}, "library": {
            "name": "plant", "version": "1.0.0",
            "types": [{
                "name": "Chain",
                "body": {
                    "attributes": floats(&["x", "y"]),
                    "equations": [
                        {"target": "y", "kind": "explicit", "expr": {"op": "add", "args": [attr("$self", "x"), constant(1.0)]}},
                        {"target": "x", "kind": "explicit", "expr": constant(5.0)}
                    ]
                }
            }, {
                "name": "Switch",
                "body": {"attributes": [{"name": "on", "kind": "bool", "init": {"kind": "bool", "value": true}}]}
            }]
        }})
        .to_string(),
    )
    .unwrap();
    let continuous = parse_system(
        &json!({"raichu_system": {"format": 1}, "system": {
            "name": "s", "library": {"name": "plant", "version": "1.0.0"},
            "instances": [{"name": "C1", "type": "Chain"}]}})
        .to_string(),
    )
    .unwrap();
    let expanded = expand(std::slice::from_ref(&library), &continuous, None)
        .unwrap()
        .model;
    assert_eq!(
        names(expanded.evaluation_order.as_deref().unwrap()),
        ["C1.x", "C1.y"]
    );

    let discrete = parse_system(
        &json!({"raichu_system": {"format": 1}, "system": {
            "name": "s", "library": {"name": "plant", "version": "1.0.0"},
            "instances": [{"name": "S1", "type": "Switch"}]}})
        .to_string(),
    )
    .unwrap();
    let expanded = expand(&[library], &discrete, None).unwrap().model;
    assert_eq!(expanded.evaluation_order, None);
}

#[test]
fn inside_a_cycle_a_step_follows_what_it_reads_but_for_the_closing_edge() {
    // A flow network is cyclic: the demand reads what was delivered, which
    // reads the allocation, which reads the demand. Declared positionally, the
    // delivery (an equation) would sweep before the allocation (an operator),
    // and read the previous point's allocation. Inside the cycle only the one
    // edge that closes it may be read stale.
    let m = model(json!({
        "name": "cycle",
        "components": [
            {"name": "supply",
             "attributes": [{"name": "available", "kind": "float", "init": float(5.0)}, float_attr("fed")],
             "ports": [{"name": "out", "dir": "out", "attr": "available",
                        "channels": [{"name": "demand"}, {"name": "alloc"}, {"name": "delivered"}]}],
             "equations": [
                {"target": "out__delivered__a", "kind": "explicit",
                 "expr": {"op": "mul", "args": [attr("supply", "out__alloc__a"), constant(1.0)]}},
                {"target": "fed", "kind": "explicit", "expr": attr("supply", "out__delivered__a")},
                {"target": "out__demand__a", "kind": "explicit",
                 "expr": {"op": "add", "args": [attr("a", "need"),
                     {"op": "mul", "args": [constant(0.0), attr("supply", "fed")]}]}}
             ],
             "allocations": [{"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
                              "available": attr("supply", "available"), "policy": "proportional"}]},
            {"name": "a", "attributes": [{"name": "need", "kind": "float", "init": float(3.0)}],
             "ports": [{"name": "input", "dir": "in"}]}
        ],
        "connections": [{"name": "a", "from": {"component": "supply", "port": "out"},
                         "to": {"component": "a", "port": "input"}}]
    }));
    // Walked from the positionally first member (the delivery): the
    // allocation now precedes its reader, and the one stale read left is the
    // total `fed` reading the delivery, the edge that closes the cycle.
    let order = names(&derive_evaluation_order(&m).unwrap().unwrap());
    assert_eq!(
        order,
        [
            "supply.fed",
            "supply.out__demand__a",
            "supply.split",
            "supply.out__delivered__a"
        ]
    );
}

fn float_attr(name: &str) -> Json {
    json!({"name": name, "kind": "float", "init": float(0.0)})
}
