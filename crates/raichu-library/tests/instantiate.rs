//! Instantiation of types, fragments and parameters (unit U2).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::CompiledModel;
use raichu_library::{expand, parse_library, parse_system, Library, SystemModel};
use serde_json::{json, Value};

fn library(body: Value) -> Library {
    parse_library(&json!({"raichu_library": {"format": 1}, "library": body}).to_string()).unwrap()
}

fn system(body: Value) -> SystemModel {
    parse_system(&json!({"raichu_system": {"format": 1}, "system": body}).to_string()).unwrap()
}

fn float(x: f64) -> Value {
    json!({"kind": "float", "value": x})
}

#[test]
fn a_required_parameter_omitted_is_refused_and_a_default_fills_in() {
    let lib = library(json!({
        "name": "plant", "version": "1.0.0",
        "types": [{
            "name": "Battery",
            "parameters": [{"name": "capacity", "kind": "number"},
                           {"name": "content_ini", "kind": "number", "default": 0.0}],
            "body": {"attributes": [
                {"name": "capacity", "kind": "float", "init": {"kind": "float", "value": {"$param": "capacity"}}},
                {"name": "content", "kind": "float", "init": {"kind": "float", "value": {"$param": "content_ini"}}}
            ]}
        }]
    }));
    let missing = system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "B1", "type": "Battery"}]}),
    );
    let err = expand(std::slice::from_ref(&lib), &missing, None).unwrap_err();
    assert!(err.message.contains("instance `B1`"), "{}", err.message);
    assert!(err.message.contains("`capacity`"), "{}", err.message);
    let given = system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "B1", "type": "Battery", "parameters": {"capacity": 100}}]}),
    );
    let model = expand(&[lib], &given, None).unwrap().model;
    let attrs = &model.components[0].attributes;
    assert_eq!(attrs[0].init, raichu_expr::Value::Float(100.0));
    assert_eq!(attrs[1].init, raichu_expr::Value::Float(0.0));
}

fn flows_library() -> Library {
    library(json!({
        "name": "plant", "version": "1.0.0",
        "types": [{
            "name": "Tank",
            "parameters": [
                {"name": "flow_types", "kind": "list", "items": "string"},
                {"name": "method", "kind": "enum", "values": ["mean", "median", "and", "or"], "default": "and"},
                {"name": "lambda", "kind": "number", "default": 0.5}
            ],
            "body": {
                "attributes": [
                    {"$for_each": "flow_types", "as": "f", "items": [
                        {"name": "${f}_content", "kind": "float", "init": float(0.0)}
                    ]},
                    {"name": "votes", "kind": "int", "init": {"$choose": "method", "cases": {
                        "mean": {"kind": "int", "value": 1}, "median": {"kind": "int", "value": 2},
                        "and": {"kind": "int", "value": 3}, "or": {"kind": "int", "value": 4}}}}
                ],
                "ports": [{"$for_each": "flow_types", "as": "f", "items": [
                    {"name": "${f}_out", "dir": "out", "attr": "${f}_content"}
                ]}],
                "automata": [{"name": "life", "states": ["ok", "ko"], "init": "ok",
                    "transitions": [{"name": "break", "source": "ok", "targets": ["ko"],
                                     "distrib": "exp", "rate": {"$param": "lambda"}}]}]
            }
        }]
    }))
}

#[test]
fn for_each_repeats_attributes_and_ports_per_list_element() {
    let s = system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "T", "type": "Tank", "parameters": {"flow_types": ["H2", "O2"]}}]}),
    );
    let model = expand(&[flows_library()], &s, None).unwrap().model;
    let c = &model.components[0];
    let names: Vec<&str> = c.attributes.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names, ["H2_content", "O2_content", "votes"]);
    let ports: Vec<&str> = c.ports.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(ports, ["H2_out", "O2_out"]);
}

#[test]
fn an_empty_list_expands_cleanly() {
    let s = system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "T", "type": "Tank", "parameters": {"flow_types": []}}]}),
    );
    let model = expand(&[flows_library()], &s, None).unwrap().model;
    assert!(model.components[0].ports.is_empty());
    model.validate().unwrap();
}

#[test]
fn choose_selects_the_case_and_an_unknown_value_lists_the_allowed_ones() {
    let s = system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "T", "type": "Tank", "parameters": {"flow_types": [], "method": "median"}}]}),
    );
    let model = expand(&[flows_library()], &s, None).unwrap().model;
    assert_eq!(
        model.components[0].attributes[0].init,
        raichu_expr::Value::Int(2)
    );
    let bad = system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "T", "type": "Tank", "parameters": {"flow_types": [], "method": "average"}}]}),
    );
    let err = expand(&[flows_library()], &bad, None).unwrap_err();
    assert!(
        err.message.contains("`mean`, `median`, `and`, `or`"),
        "{}",
        err.message
    );
}

#[test]
fn a_law_parameter_reduces_to_a_literal_rate() {
    let s = system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "T", "type": "Tank", "parameters": {"flow_types": [], "lambda": 2}}]}),
    );
    let model = expand(&[flows_library()], &s, None).unwrap().model;
    let transition = &model.components[0].automata[0].transitions[0];
    assert_eq!(
        transition.distrib,
        raichu_model::Distrib::Exp {
            rate: Some(2.0),
            rate_expr: None
        }
    );
}

#[test]
fn a_string_given_to_a_number_is_refused_at_its_pointer() {
    let s = system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "T", "type": "Tank", "parameters": {"flow_types": [], "lambda": "fast"}}]}),
    );
    let err = expand(&[flows_library()], &s, None).unwrap_err();
    assert_eq!(err.pointer, "/system/instances/0/parameters/lambda");
    assert!(err.message.contains("instance `T`"), "{}", err.message);
}

#[test]
fn a_misspelled_key_in_a_type_body_is_refused() {
    let mut lib = flows_library();
    lib.types[0].body["automata"][0]["transitions"][0]["gaurd"] =
        json!({"op": "const", "value": {"kind": "bool", "value": true}});
    let s = system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "T", "type": "Tank", "parameters": {"flow_types": []}}]}),
    );
    let err = expand(&[lib], &s, None).unwrap_err();
    assert!(
        err.pointer.ends_with("/transitions/0/gaurd"),
        "{}",
        err.pointer
    );
    assert!(err.message.contains("instance `T`"), "{}", err.message);
}

fn gate_library(extra_fragment: Value) -> Library {
    let mut fragments = vec![
        json!({"name": "ready", "body": {"attributes": [{"name": "ready", "kind": "bool", "init": {"kind": "bool", "value": true}}]}}),
        json!({"name": "ready_too", "body": {"attributes": [{"name": "ready", "kind": "bool", "init": {"kind": "bool", "value": false}}]}}),
    ];
    fragments.push(extra_fragment);
    library(json!({
        "name": "plant", "version": "1.0.0",
        "fragments": fragments,
        "types": [
            {"name": "Twice", "uses": [{"fragment": "ready"}, {"fragment": "ready_too"}]},
            {"name": "Loop", "uses": [{"fragment": "a"}]},
            {"name": "Nosy", "body": {"attributes": [{"name": "x", "kind": "bool", "init": {"kind": "bool", "value": true}}],
                "sensitive_functions": [{"name": "peek", "effects": [{
                    "target": {"component": "$self", "attribute": "x"},
                    "value": {"op": "attr", "attr": {"component": "Other", "attribute": "y"}}}]}]}}
        ]
    }))
}

fn one(type_name: &str) -> SystemModel {
    system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "X", "type": type_name}]}),
    )
}

#[test]
fn two_fragments_declaring_one_attribute_are_refused_naming_both() {
    let err = expand(&[gate_library(json!({"name": "a"}))], &one("Twice"), None).unwrap_err();
    assert!(err.message.contains("attribute `ready`"), "{}", err.message);
    assert!(
        err.message.contains("fragment `plant.ready`"),
        "{}",
        err.message
    );
    assert!(
        err.message.contains("fragment `plant.ready_too`"),
        "{}",
        err.message
    );
}

#[test]
fn a_fragment_cycle_is_refused_naming_the_chain() {
    let lib = gate_library(json!({"name": "a", "uses": [{"fragment": "a"}]}));
    let err = expand(&[lib], &one("Loop"), None).unwrap_err();
    assert!(
        err.message.contains("cycle: plant.a -> plant.a"),
        "{}",
        err.message
    );
}

#[test]
fn a_type_body_reading_another_instance_is_refused() {
    let err = expand(&[gate_library(json!({"name": "a"}))], &one("Nosy"), None).unwrap_err();
    assert!(err.message.contains("`Other.y`"), "{}", err.message);
    assert!(err.message.contains("$self"), "{}", err.message);
}

fn lamp_library() -> Library {
    library(json!({
        "name": "home", "version": "2.0.0",
        "types": [
            {"name": "Switch", "body": {
                "attributes": [{"name": "on", "kind": "bool", "init": {"kind": "bool", "value": true}}],
                "ports": [{"name": "power_out", "dir": "out", "attr": "on"}]}},
            {"name": "Lamp", "body": {
                "attributes": [{"name": "lit", "kind": "bool", "init": {"kind": "bool", "value": false}}],
                "ports": [{"name": "power_in", "dir": "in"}],
                "sensitive_functions": [{"name": "light", "effects": [{
                    "target": {"component": "$self", "attribute": "lit"},
                    "value": {"op": "port_agg", "port": {"component": "$self", "port": "power_in"}, "agg": "any"}}]}]}}
        ]
    }))
}

fn room(connections: Value) -> SystemModel {
    system(
        json!({"name": "room", "library": {"name": "home", "version": "2.0.0"},
        "instances": [{"name": "S", "type": "Switch"}, {"name": "L1", "type": "Lamp"}, {"name": "L2", "type": "Lamp"}],
        "connections": connections}),
    )
}

#[test]
fn three_instances_expand_connect_validate_and_compile() {
    let s = room(
        json!([{"from": "S.power_out", "to": "L1.power_in"}, {"from": "S.power_out", "to": "L2.power_in"}]),
    );
    let expansion = expand(&[lamp_library()], &s, None).unwrap();
    let model = &expansion.model;
    assert_eq!(model.components.len(), 3);
    for lamp in &model.components[1..] {
        let effect = &lamp.sensitive_functions[0].effects[0];
        assert_eq!(effect.target.component, lamp.name);
    }
    model.validate().unwrap();
    CompiledModel::compile(model).unwrap();
    assert!(expansion.provenance.component("L2").is_some());
}

#[test]
fn a_connection_against_the_port_direction_names_both_ends() {
    let s = room(json!([{"from": "L1.power_in", "to": "S.power_out"}]));
    let err = expand(&[lamp_library()], &s, None).unwrap_err();
    assert_eq!(err.pointer, "/system/connections/0/from");
    assert!(
        err.message.contains("`L1.power_in` is an in port"),
        "{}",
        err.message
    );
}

#[test]
fn an_unknown_port_lists_the_ports_of_the_instance() {
    let s = room(json!([{"from": "S.power", "to": "L1.power_in"}]));
    let err = expand(&[lamp_library()], &s, None).unwrap_err();
    assert!(err.message.contains("`power_out`"), "{}", err.message);
}

#[test]
fn a_library_version_mismatch_names_both_versions() {
    let s = system(json!({"name": "room", "library": {"name": "home", "version": "1.0.0"}}));
    let err = expand(&[lamp_library()], &s, None).unwrap_err();
    assert_eq!(err.pointer, "/system/library");
    assert!(
        err.message.contains("`1.0.0`") && err.message.contains("`2.0.0`"),
        "{}",
        err.message
    );
}

#[test]
fn two_fragments_with_one_name_are_refused() {
    let doc = json!({"raichu_library": {"format": 1}, "library": {
        "name": "plant", "version": "1.0.0",
        "fragments": [{"name": "f"}, {"name": "f"}]}});
    let err = parse_library(&doc.to_string()).unwrap_err();
    assert_eq!(err.pointer, "/library/fragments/1/name");
    assert!(
        err.message.contains("fragment `f` is declared twice"),
        "{}",
        err.message
    );
}

#[test]
fn an_integral_number_interpolates_without_a_fraction() {
    let lib = library(json!({"name": "plant", "version": "1.0.0", "types": [{
        "name": "T",
        "parameters": [{"name": "n", "kind": "number", "default": 2}],
        "body": {"attributes": [{"name": "a_${n}", "kind": "bool", "init": {"kind": "bool", "value": true}}]}}]}));
    let s = system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "X", "type": "T"}, {"name": "Y", "type": "T", "parameters": {"n": 2.5}}]}),
    );
    let model = expand(&[lib], &s, None).unwrap().model;
    assert_eq!(model.components[0].attributes[0].name, "a_2");
    assert_eq!(model.components[1].attributes[0].name, "a_2.5");
}

fn shares_library() -> Library {
    library(json!({
        "name": "plant", "version": "1.0.0",
        "types": [{
            "name": "Splitter",
            "parameters": [{"name": "shares", "kind": "list", "items": "object", "default": []}],
            "body": {
                "attributes": [{"name": "stock", "kind": "float", "init": float(5.0)}],
                "ports": [{"name": "out", "dir": "out", "attr": "stock",
                           "channels": [{"name": "demand"}, {"name": "alloc"}]}],
                "allocations": [{"name": "split", "port": "out", "demand": "demand",
                                 "allocated": "alloc", "policy": "shares", "shares": {"$param": "shares"},
                                 "available": {"op": "attr", "attr": {"component": "$self", "attribute": "stock"}}}]
            }
        }, {
            "name": "Sink",
            "body": {"ports": [{"name": "in", "dir": "in"}]}
        }]
    }))
}

#[test]
fn a_list_of_objects_is_inserted_as_written() {
    let s = system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [
            {"name": "S", "type": "Splitter", "parameters": {"shares": [
                {"to": {"component": "A", "port": "in"}, "value": 1.0}]}},
            {"name": "A", "type": "Sink"}],
        "connections": [{"from": "S.out", "to": "A.in"}]}),
    );
    let model = expand(&[shares_library()], &s, None).unwrap().model;
    let written = serde_json::to_value(&model.components[0].allocations[0]).unwrap();
    assert_eq!(
        written["shares"],
        json!([{"to": {"component": "A", "port": "in"}, "value": 1.0}])
    );
}

#[test]
fn a_list_of_objects_refuses_an_item_that_is_not_one() {
    let s = system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "S", "type": "Splitter", "parameters": {"shares": [0.25]}}]}),
    );
    let err = expand(&[shares_library()], &s, None).unwrap_err();
    assert!(err.message.contains("object"), "{}", err.message);
    assert!(err.message.contains("`shares`"), "{}", err.message);
}

fn edge_library(read: Value) -> Library {
    library(json!({
        "name": "plant", "version": "1.0.0",
        "types": [{
            "name": "Supply",
            "body": {
                "attributes": [{"name": "stock", "kind": "float", "init": float(5.0)}],
                "ports": [{"name": "out", "dir": "out", "attr": "stock",
                           "channels": [{"name": "demand"}, {"name": "claim"}],
                           "equations": [{"channel": "claim", "expr": {"op": "min", "args": [
                               read,
                               {"op": "attr", "attr": {"component": "$self", "attribute": "stock"}}]}}]}]
            }
        }, {
            "name": "Need",
            "body": {
                "attributes": [{"name": "need", "kind": "float", "init": float(3.0)}],
                "ports": [{"name": "in", "dir": "in", "equations": [
                    {"channel": "demand", "expr": {"op": "attr", "attr": {"component": "$self", "attribute": "need"}}}]}]
            }
        }]
    }))
}

fn edge_system() -> SystemModel {
    system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "S", "type": "Supply"}, {"name": "N", "type": "Need"}],
        "connections": [{"from": "S.out", "to": "N.in"}]}),
    )
}

#[test]
fn a_port_equation_in_a_type_body_reads_its_edge() {
    let lib =
        edge_library(json!({"op": "attr", "attr": {"component": "$edge", "attribute": "demand"}}));
    let model = expand(&[lib], &edge_system(), None).unwrap().model;
    let resolved = model.with_resolved_connections().unwrap();
    let targets: Vec<&str> = resolved.components[0]
        .equations
        .iter()
        .map(|e| e.target.as_str())
        .collect();
    assert!(targets.contains(&"out__claim__N__in"), "{targets:?}");
    assert!(targets.contains(&"out__demand__N__in"), "{targets:?}");
    CompiledModel::compile(&model).unwrap();
}

#[test]
fn a_port_equation_in_a_type_body_cannot_reach_another_instance() {
    let lib = edge_library(json!({"op": "attr", "attr": {"component": "N", "attribute": "need"}}));
    let err = expand(&[lib], &edge_system(), None).unwrap_err();
    assert!(
        err.message.contains("outside its own instance"),
        "{}",
        err.message
    );
    assert!(err.message.contains("`N.need`"), "{}", err.message);
}

#[test]
fn a_misspelled_port_equation_key_is_refused() {
    let lib = library(json!({
        "name": "plant", "version": "1.0.0",
        "types": [{"name": "T", "body": {
            "attributes": [{"name": "x", "kind": "float", "init": float(0.0)}],
            "ports": [{"name": "out", "dir": "out", "attr": "x", "channels": [{"name": "claim"}],
                       "equations": [{"chanel": "claim", "expr": {"op": "const", "value": float(1.0)}}]}]
        }}]
    }));
    let s = system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "T1", "type": "T"}]}),
    );
    // The required `channel` is missing, so the reader refuses at the
    // equation before any key is probed, naming the field it expected.
    let err = expand(&[lib], &s, None).unwrap_err();
    assert!(err.message.contains("`channel`"), "{}", err.message);
    assert!(
        err.pointer.contains("/ports/0/equations/0"),
        "{}",
        err.pointer
    );
}

/// A buffer type and a unit type whose repair refills the buffer it is
/// connected to, written with `target` as the repair's effect target.
fn refill_library(target: Value) -> Library {
    library(json!({
        "name": "plant", "version": "1.0.0",
        "types": [{
            "name": "Buffer",
            "body": {
                "attributes": [{"name": "level", "kind": "float", "init": float(5.0)}],
                "ports": [{"name": "level_out", "dir": "out", "attr": "level"}]
            }
        }, {
            "name": "Unit",
            "body": {
                "ports": [{"name": "buffer_in", "dir": "in"}],
                "automata": [{"name": "health", "states": ["up", "down"], "init": "up",
                    "transitions": [
                        {"name": "fail", "source": "up", "targets": ["down"],
                         "distrib": "delay", "time": 1.0},
                        {"name": "repair", "source": "down", "targets": ["up"],
                         "distrib": "delay", "time": 2.0,
                         "effects": [{"target": target,
                                      "value": {"op": "const", "value": float(10.0)}}]}]}]
            }
        }]
    }))
}

fn refill_system() -> SystemModel {
    system(
        json!({"name": "s", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "B", "type": "Buffer"}, {"name": "U", "type": "Unit"}],
        "connections": [{"from": "B.level_out", "to": "U.buffer_in"}]}),
    )
}

#[test]
fn a_type_body_writes_a_neighbour_through_a_port() {
    let lib = refill_library(json!({"component": "$port", "attribute": "buffer_in"}));
    let model = expand(&[lib], &refill_system(), None).unwrap().model;
    let resolved = model.with_resolved_connections().unwrap();
    let effect = &resolved.components[1].automata[0].transitions[1].effects[0];
    assert_eq!(
        (
            effect.target.component.as_str(),
            effect.target.attribute.as_str()
        ),
        ("B", "level")
    );
    CompiledModel::compile(&model).unwrap();
}

#[test]
fn a_type_body_still_cannot_write_a_neighbour_directly() {
    let lib = refill_library(json!({"component": "B", "attribute": "level"}));
    let err = expand(&[lib], &refill_system(), None).unwrap_err();
    assert!(err.message.contains("`B.level`"), "{}", err.message);
}
