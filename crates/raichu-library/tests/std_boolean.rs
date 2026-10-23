//! The boolean fragments of the standard library (unit U6), run on the
//! engine.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::Value as V;
use raichu_library::{
    check_library, expand, parse_library, parse_study, parse_system, standard_library, Library,
    StudyDocument, SystemModel, TypeCheck,
};
use raichu_model::Model;
use serde_json::{json, Value};

fn library(body: Value) -> Library {
    parse_library(&json!({"raichu_library": {"format": 1}, "library": body}).to_string()).unwrap()
}

fn system(body: Value) -> SystemModel {
    parse_system(&json!({"raichu_system": {"format": 1}, "system": body}).to_string()).unwrap()
}

fn study(body: Value) -> StudyDocument {
    parse_study(&json!({"raichu_study": {"format": 1}, "study": body}).to_string()).unwrap()
}

fn delay(t: f64) -> Value {
    json!({"distrib": "delay", "time": t})
}

/// Indicator change-points over a run to `t_max`.
fn run(model: &Model, t_max: f64) -> raichu_core::SimulationResult {
    let compiled = CompiledModel::compile(model).unwrap();
    let config = EngineConfig {
        t_max,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config).unwrap().run().unwrap()
}

fn series(result: &raichu_core::SimulationResult, name: &str) -> Vec<(f64, V)> {
    result
        .indicators
        .iter()
        .find(|s| s.name == name)
        .unwrap()
        .points
        .clone()
}

fn watch(name: &str, component: &str, attribute: &str) -> Value {
    json!({"name": name, "target": "attribute", "attr": {"component": component, "attribute": attribute}})
}

/// Sources a, b, c, a consumer whose output `out` is produced under
/// `[[a_fed_in, b_fed_in], [c_fed_in]]`, a k-of-n gate and a server with a
/// dormant service and a dormant availability gate.
fn witness_library() -> Library {
    library(json!({
        "name": "demo", "version": "1.0.0", "imports": ["std"],
        "types": [
            {"name": "Source",
             "parameters": [{"name": "on", "kind": "bool", "default": true}],
             "uses": [{"fragment": "std.bool_source_out", "with": {"name": "x", "prod_init": {"$param": "on"}}}],
             "modes": [{"name": "fail", "nature": "failure", "effects": [{"attribute": "x_fed_available_out", "value": false}]}]},
            {"name": "Consumer",
             "uses": [
                {"fragment": "std.bool_flow_in", "with": {"name": "a"}},
                {"fragment": "std.bool_flow_in", "with": {"name": "b"}},
                {"fragment": "std.bool_flow_in", "with": {"name": "c"}},
                {"fragment": "std.bool_flow_out", "with": {"name": "out", "prod_cond": [["a_fed_in", "b_fed_in"], ["c_fed_in"]]}}
             ]},
            {"name": "Vote",
             "uses": [{"fragment": "std.logic_gate", "with": {"name": "two_of", "logic": "k", "k": 2}}]},
            {"name": "Lonely",
             "uses": [{"fragment": "std.bool_flow_in", "with": {"name": "p", "logic": "and"}},
                      {"fragment": "std.bool_flow_in", "with": {"name": "q", "logic": "or", "default": true}}]},
            {"name": "Server",
             "uses": [
                {"fragment": "std.bool_source_out", "with": {"name": "f_service", "prod_init": false}},
                {"fragment": "std.bool_source_out", "with": {"name": "srv_veh", "fed_available_init": false}}
             ],
             "modes": [
                {"name": "mdc_b", "nature": "compromise", "effects": [{"attribute": "f_service_prod_available", "value": true}]},
                {"name": "mc_priv2", "nature": "compromise", "effects": [{"attribute": "srv_veh_fed_available_out", "value": true}]}
             ]}
        ]
    }))
}

fn binding() -> Value {
    json!({"name": "demo", "version": "1.0.0"})
}

#[test]
fn an_input_reads_its_default_while_unconnected_and_follows_its_source_otherwise() {
    let s = system(json!({"name": "s", "library": binding(),
        "instances": [{"name": "L", "type": "Lonely"}, {"name": "A", "type": "Source"}, {"name": "C", "type": "Consumer"}],
        "connections": [{"from": "A.x_out", "to": "C.a_in"}],
        "indicators": [watch("p", "L", "p_fed_in"), watch("q", "L", "q_fed_in"), watch("a", "C", "a_fed_in")]}));
    let result = run(&expand(&[witness_library()], &s, None).unwrap().model, 1.0);
    assert_eq!(series(&result, "p"), vec![(0.0, V::Bool(false))]);
    assert_eq!(series(&result, "q"), vec![(0.0, V::Bool(true))]);
    assert_eq!(series(&result, "a"), vec![(0.0, V::Bool(true))]);
}

#[test]
fn an_output_is_produced_when_any_group_of_its_condition_holds() {
    let s = system(json!({"name": "s", "library": binding(),
        "instances": [
            {"name": "A", "type": "Source", "parameters": {"on": false}},
            {"name": "B", "type": "Source"},
            {"name": "Cs", "type": "Source"},
            {"name": "C", "type": "Consumer"}],
        "connections": [
            {"from": "A.x_out", "to": "C.a_in"}, {"from": "B.x_out", "to": "C.b_in"}, {"from": "Cs.x_out", "to": "C.c_in"}],
        "indicators": [watch("out", "C", "out_fed_out")]}));
    let st = study(json!({"name": "st", "system": "s",
        "modes": [{"mode": "fail", "instances": ["Cs"], "occurrence": delay(2.0)}]}));
    let result = run(
        &expand(&[witness_library()], &s, Some(&st)).unwrap().model,
        5.0,
    );
    // a is off, so only the group [c] produces; c fails at 2.
    assert_eq!(
        series(&result, "out"),
        vec![(0.0, V::Bool(true)), (2.0, V::Bool(false))]
    );
}

#[test]
fn a_k_of_n_gate_switches_at_the_kth_fed_input() {
    let s = system(json!({"name": "s", "library": binding(),
        "instances": [
            {"name": "S1", "type": "Source"}, {"name": "S2", "type": "Source"}, {"name": "S3", "type": "Source"},
            {"name": "V", "type": "Vote"}],
        "connections": [
            {"from": "S1.x_out", "to": "V.two_of_in"}, {"from": "S2.x_out", "to": "V.two_of_in"},
            {"from": "S3.x_out", "to": "V.two_of_in"}],
        "indicators": [watch("vote", "V", "two_of_fed_in")]}));
    let st = study(json!({"name": "st", "system": "s", "modes": [
        {"mode": "fail", "instances": ["S1"], "occurrence": delay(1.0)},
        {"mode": "fail", "instances": ["S2"], "occurrence": delay(2.0)}]}));
    let result = run(
        &expand(&[witness_library()], &s, Some(&st)).unwrap().model,
        5.0,
    );
    assert_eq!(
        series(&result, "vote"),
        vec![(0.0, V::Bool(true)), (2.0, V::Bool(false))]
    );
}

#[test]
fn a_dormant_service_is_fed_only_while_its_compromise_mode_is_occurred() {
    let s = system(json!({"name": "s", "library": binding(),
        "instances": [{"name": "Srv", "type": "Server"}],
        "indicators": [watch("service", "Srv", "f_service_fed_out"), watch("veh", "Srv", "srv_veh_fed_out")]}));
    let st = study(json!({"name": "st", "system": "s", "modes": [
        {"mode": "mdc_b", "instances": ["Srv"], "occurrence": delay(1.0), "repair": delay(2.0)},
        {"mode": "mc_priv2", "instances": ["Srv"], "occurrence": delay(4.0), "repair": delay(10.0)}]}));
    let result = run(
        &expand(&[witness_library()], &s, Some(&st)).unwrap().model,
        3.5,
    );
    assert_eq!(
        series(&result, "service"),
        vec![
            (0.0, V::Bool(false)),
            (1.0, V::Bool(true)),
            (3.0, V::Bool(false))
        ]
    );
    let result = run(
        &expand(&[witness_library()], &s, Some(&st)).unwrap().model,
        6.0,
    );
    assert_eq!(
        series(&result, "veh"),
        vec![(0.0, V::Bool(false)), (4.0, V::Bool(true))]
    );
}

#[test]
fn a_gate_starting_unavailable_resolves_to_any_activates() {
    // A mode cutting a gate that starts unavailable is refused: the rule
    // resolved from `fed_available_init: false` is `any_activates`.
    let lib = library(
        json!({"name": "demo", "version": "1.0.0", "imports": ["std"], "types": [{
        "name": "Server",
        "uses": [{"fragment": "std.bool_source_out", "with": {"name": "srv", "fed_available_init": false}}],
        "modes": [{"name": "cut", "nature": "failure", "effects": [{"attribute": "srv_fed_available_out", "value": false}]}]}]}),
    );
    let s = system(
        json!({"name": "s", "library": binding(), "instances": [{"name": "Srv", "type": "Server"}]}),
    );
    let st = study(
        json!({"name": "st", "system": "s", "modes": [{"mode": "cut", "instances": ["Srv"], "occurrence": delay(1.0)}]}),
    );
    let err = expand(&[lib], &s, Some(&st)).unwrap_err();
    assert!(err.message.contains("`any_activates`"), "{}", err.message);
}

#[test]
fn the_standard_library_parses_checks_and_carries_the_engine_version() {
    let std = standard_library().unwrap();
    assert_eq!(std.version, env!("CARGO_PKG_VERSION"));
    let names: Vec<&str> = std.fragments.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "bool_flow_in",
            "bool_flow_out",
            "bool_source_out",
            "logic_gate"
        ]
    );
    let report = check_library(&witness_library(), &[]);
    assert!(
        report.iter().all(|(_, c)| *c == TypeCheck::Valid),
        "{report:?}"
    );
}

#[test]
fn an_expansion_using_std_stamps_the_engine_version() {
    let s = system(
        json!({"name": "s", "library": binding(), "instances": [{"name": "A", "type": "Source"}]}),
    );
    let model = expand(&[witness_library()], &s, None).unwrap().model;
    assert_eq!(
        model.source.unwrap().std_version.as_deref(),
        Some(env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn check_library_reports_the_failing_choice_and_the_unchecked_type() {
    let lib = library(json!({"name": "demo", "version": "1.0.0", "types": [
        {"name": "Picky",
         "parameters": [{"name": "method", "kind": "enum", "values": ["mean", "median"], "default": "mean"}],
         "body": {
            "attributes": [{"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}],
            "equations": [{"target": "x", "kind": "explicit", "expr": {"$choose": "method", "cases": {
                "mean": {"op": "const", "value": {"kind": "float", "value": 1.0}},
                "median": {"op": "attr", "attr": {"component": "$self", "attribute": "nope"}}}}}]}},
        {"name": "Needy", "parameters": [{"name": "capacity", "kind": "number"}]}
    ]}));
    let report = check_library(&lib, &[]);
    match &report[0].1 {
        TypeCheck::Invalid(errors) => {
            assert_eq!(errors.len(), 1);
            assert!(errors[0].message.contains("nope"), "{}", errors[0].message);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        report[1].1,
        TypeCheck::Unchecked(vec!["capacity".to_owned()])
    );
}
