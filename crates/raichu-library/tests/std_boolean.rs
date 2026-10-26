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
    // The boolean fragments come first; the continuous ones follow them
    // (`std_continuous.rs`).
    let names: Vec<&str> = std.fragments.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names[..5],
        [
            "bool_flow_in",
            "bool_flow_out",
            "bool_source_out",
            "bool_flow_out_tempo",
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

/// Three pumps that fail at `fail_at` and are repaired 5 later, read by a
/// voter that loses service while two or more are failed. `standby` puts
/// each pump through a `standby` state before it runs, and the exported
/// set of states is then `failed` and `standby`.
fn pump_body(fail_at: f64, standby: bool) -> Value {
    let (states, init, first) = if standby {
        (
            json!(["standby", "ok", "failed"]),
            "standby",
            json!({"name": "start", "source": "standby", "targets": ["ok"], "distrib": "delay", "time": 0.5}),
        )
    } else {
        (json!(["ok", "failed"]), "ok", Value::Null)
    };
    let mut transitions = vec![
        json!({"name": "fail", "source": "ok", "targets": ["failed"], "distrib": "delay", "time": fail_at}),
        json!({"name": "repair", "source": "failed", "targets": ["ok"], "distrib": "delay", "time": 5.0}),
    ];
    if !first.is_null() {
        transitions.insert(0, first);
    }
    json!({"automata": [{"name": "health", "states": states, "init": init, "transitions": transitions}]})
}

fn voter_body(lost: Value, held: Value) -> Value {
    json!({"automata": [{"name": "service", "states": ["up", "down"], "init": "up",
    "transitions": [
        {"name": "lose", "source": "up", "targets": ["down"], "distrib": "inst", "probs": [], "guard": lost},
        {"name": "recover", "source": "down", "targets": ["up"], "distrib": "inst", "probs": [], "guard": held}
    ]}]})
}

fn count_at_least(sum: Value, k: i64) -> Value {
    json!({"op": "cmp", "cmp": "ge", "lhs": sum, "rhs": {"op": "const", "value": {"kind": "int", "value": k}}})
}

fn count_below(sum: Value, k: i64) -> Value {
    json!({"op": "cmp", "cmp": "lt", "lhs": sum, "rhs": {"op": "const", "value": {"kind": "int", "value": k}}})
}

/// The flat reference: the voter counts failed pumps by reading their
/// states directly.
fn flat_voting(fail_at: [f64; 3], standby: bool) -> Model {
    let exported: &[&str] = if standby {
        &["failed", "standby"]
    } else {
        &["failed"]
    };
    let term = |pump: &str| {
        let any = json!({"op": "bool", "bool_op": "or", "args": exported.iter().map(|s|
            json!({"op": "state_active", "state": {"component": pump, "automaton": "health", "state": s}})
        ).collect::<Vec<_>>()});
        json!({"op": "if", "cond": any, "then": {"op": "const", "value": {"kind": "int", "value": 1}},
               "otherwise": {"op": "const", "value": {"kind": "int", "value": 0}}})
    };
    let sum = json!({"op": "add", "args": [term("P1"), term("P2"), term("P3")]});
    let mut components: Vec<Value> = ["P1", "P2", "P3"]
        .iter()
        .zip(fail_at)
        .map(|(name, at)| {
            let mut body = pump_body(at, standby);
            body["name"] = json!(name);
            body["attributes"] = json!([]);
            body
        })
        .collect();
    let mut voter = voter_body(count_at_least(sum.clone(), 2), count_below(sum, 2));
    voter["name"] = json!("system");
    voter["attributes"] = json!([]);
    components.push(voter);
    Model::from_json(&json!({"name": "voting", "components": components}).to_string()).unwrap()
}

/// The same plant as a library: each pump exports its set of states with
/// `std.state_out`, and the voter sums what its in port receives.
fn library_voting(fail_at: [f64; 3], standby: bool) -> Model {
    let exported: Vec<&str> = if standby {
        vec!["failed", "standby"]
    } else {
        vec!["failed"]
    };
    let sum =
        json!({"op": "port_agg", "agg": "sum", "port": {"component": "$self", "port": "down_in"}});
    let pump = |name: &str, at: f64| {
        let mut ty = json!({"name": name,
            "uses": [{"fragment": "std.state_out", "with": {"name": "down", "automaton": "health", "states": exported}}]});
        ty["body"] = pump_body(at, standby);
        ty
    };
    let mut voter = json!({"name": "Voter"});
    let mut body = voter_body(count_at_least(sum.clone(), 2), count_below(sum, 2));
    body["ports"] = json!([{"name": "down_in", "dir": "in"}]);
    voter["body"] = body;
    let lib = library(json!({
        "name": "voting", "version": "1.0.0", "imports": ["std"],
        "types": [pump("Pump1", fail_at[0]), pump("Pump2", fail_at[1]), pump("Pump3", fail_at[2]), voter]
    }));
    let sys = system(
        json!({"name": "voting", "library": {"name": "voting", "version": "1.0.0"},
        "instances": [{"name": "P1", "type": "Pump1"}, {"name": "P2", "type": "Pump2"},
                      {"name": "P3", "type": "Pump3"}, {"name": "system", "type": "Voter"}],
        "connections": [{"from": "P1.down_out", "to": "system.down_in"},
                        {"from": "P2.down_out", "to": "system.down_in"},
                        {"from": "P3.down_out", "to": "system.down_in"}]}),
    );
    expand(&[lib, standard_library().unwrap()], &sys, None)
        .unwrap()
        .model
}

fn events(model: &Model, t_max: f64) -> Vec<(f64, String)> {
    run(model, t_max)
        .events
        .iter()
        .map(|e| (e.time, e.transition.clone()))
        .collect()
}

#[test]
fn a_voter_reading_state_mirrors_fires_as_one_reading_the_states() {
    let flat = events(&flat_voting([1.0, 2.0, 3.0], false), 20.0);
    let lib = events(&library_voting([1.0, 2.0, 3.0], false), 20.0);
    assert_eq!(lib, flat);
    assert!(
        flat.contains(&(2.0, "system.service.lose".to_owned())),
        "{flat:?}"
    );
}

#[test]
fn a_set_of_two_states_exports_true_while_either_is_active() {
    // The pumps sit in standby for half an hour: two of three in the
    // exported set from the start, so the service is lost at once.
    let flat = events(&flat_voting([1.0, 2.0, 3.0], true), 20.0);
    let lib = events(&library_voting([1.0, 2.0, 3.0], true), 20.0);
    assert_eq!(lib, flat);
    assert_eq!(flat[0], (0.0, "system.service.lose".to_owned()));
}

#[test]
fn same_date_failures_and_the_vote_keep_their_order() {
    // Two pumps fail at the same date: both failures fire before the vote
    // they cause, in the same order on both routes.
    let flat = events(&flat_voting([2.0, 2.0, 9.0], false), 20.0);
    let lib = events(&library_voting([2.0, 2.0, 9.0], false), 20.0);
    assert_eq!(lib, flat);
}

fn tempo_library(with: Value) -> Library {
    library(
        json!({"name": "demo", "version": "1.0.0", "imports": ["std"], "types": [
            {"name": "Source",
             "uses": [{"fragment": "std.bool_source_out", "with": {"name": "x"}}],
             "modes": [{"name": "fail", "nature": "failure", "effects": [{"attribute": "x_fed_available_out", "value": false}]}]},
            {"name": "Backup",
             "uses": [
                {"fragment": "std.bool_flow_in", "with": {"name": "x"}},
                {"fragment": "std.bool_flow_out_tempo", "with": with}
             ]}
        ]}),
    )
}

#[test]
fn a_temporised_output_enables_and_disables_after_its_delays() {
    let lib = tempo_library(json!({"name": "y", "prod_cond": [["x_fed_in"]],
        "enable_value": 3.0, "disable_value": 2.0}));
    let s = system(json!({"name": "s", "library": binding(),
        "instances": [{"name": "S", "type": "Source"}, {"name": "B", "type": "Backup"}],
        "connections": [{"from": "S.x_out", "to": "B.x_in"}],
        "indicators": [watch("y", "B", "y_fed_out")]}));
    let st = study(json!({"name": "st", "system": "s",
        "modes": [{"mode": "fail", "instances": ["S"], "occurrence": delay(10.0)}]}));
    let result = run(&expand(&[lib], &s, Some(&st)).unwrap().model, 20.0);
    // Produced from 0, enabled after 3; the source fails at 10 and the
    // output keeps feeding until the disable delay ends at 12.
    assert_eq!(
        series(&result, "y"),
        vec![
            (0.0, V::Bool(false)),
            (3.0, V::Bool(true)),
            (12.0, V::Bool(false))
        ]
    );
}

#[test]
fn a_temporised_output_may_start_enabled_and_draw_its_waits() {
    let lib = tempo_library(
        json!({"name": "y", "prod_cond": [["x_fed_in"]], "init_enable": true,
        "disable_law": "exp", "disable_value": 0.5}),
    );
    let s = system(json!({"name": "s", "library": binding(),
        "instances": [{"name": "S", "type": "Source"}, {"name": "B", "type": "Backup"}],
        "connections": [{"from": "S.x_out", "to": "B.x_in"}]}));
    let model = expand(&[lib], &s, None).unwrap().model;
    let backup = model.components.iter().find(|c| c.name == "B").unwrap();
    let tempo = backup
        .automata
        .iter()
        .find(|a| a.name == "y_tempo")
        .unwrap();
    assert_eq!(tempo.init, "enabled");
    let names: Vec<_> = tempo.transitions.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["y_enable", "y_disable"]);
    assert!(matches!(
        tempo.transitions[1].distrib,
        raichu_model::Distrib::Exp { rate: Some(rate), .. } if rate == 0.5
    ));
}

#[test]
fn a_negation_feeds_a_production_condition_while_its_source_is_false() {
    // A backup produces only while the command does NOT arrive.
    let lib = library(
        json!({"name": "demo", "version": "1.0.0", "imports": ["std"], "types": [
            {"name": "Command",
             "uses": [{"fragment": "std.bool_source_out", "with": {"name": "ctrl"}}],
             "modes": [{"name": "fail", "nature": "failure", "effects": [{"attribute": "ctrl_fed_available_out", "value": false}]}]},
            {"name": "Backup",
             "uses": [
                {"fragment": "std.bool_flow_in", "with": {"name": "ctrl"}},
                {"fragment": "std.bool_not", "with": {"of": "ctrl_fed_in"}},
                {"fragment": "std.bool_flow_out", "with": {"name": "elec", "prod_cond": [["ctrl_fed_in_not"]]}}
             ]}
        ]}),
    );
    let s = system(json!({"name": "s", "library": binding(),
        "instances": [{"name": "C", "type": "Command"}, {"name": "B", "type": "Backup"}],
        "connections": [{"from": "C.ctrl_out", "to": "B.ctrl_in"}],
        "indicators": [watch("elec", "B", "elec_fed_out")]}));
    let st = study(json!({"name": "st", "system": "s",
        "modes": [{"mode": "fail", "instances": ["C"], "occurrence": delay(4.0), "repair": delay(2.0)}]}));
    // The command fails at 4, is repaired at 6 (and would fail again at 10).
    let result = run(&expand(&[lib], &s, Some(&st)).unwrap().model, 9.0);
    assert_eq!(
        series(&result, "elec"),
        vec![
            (0.0, V::Bool(false)),
            (4.0, V::Bool(true)),
            (6.0, V::Bool(false))
        ]
    );
}

#[test]
fn a_temporised_wait_restarts_when_the_condition_fails_before_it_ends() {
    // The source is up on [0, 2), [3, 5), [6, 8): windows of 2, shorter
    // than the 3-unit enable wait. A wait that restarts on each
    // interruption never ends; one that kept its age would end at 4.
    let lib = tempo_library(json!({"name": "y", "prod_cond": [["x_fed_in"]], "enable_value": 3.0}));
    let s = system(json!({"name": "s", "library": binding(),
        "instances": [{"name": "S", "type": "Source"}, {"name": "B", "type": "Backup"}],
        "connections": [{"from": "S.x_out", "to": "B.x_in"}],
        "indicators": [watch("y", "B", "y_fed_out")]}));
    let st = study(json!({"name": "st", "system": "s",
        "modes": [{"mode": "fail", "instances": ["S"], "occurrence": delay(2.0), "repair": delay(1.0)}]}));
    let result = run(&expand(&[lib], &s, Some(&st)).unwrap().model, 8.0);
    assert_eq!(series(&result, "y"), vec![(0.0, V::Bool(false))]);
}

#[test]
fn a_never_ending_wait_builds_no_transition() {
    let lib = tempo_library(
        json!({"name": "y", "prod_cond": [["x_fed_in"]], "init_enable": true,
        "disable_law": "never"}),
    );
    let s = system(json!({"name": "s", "library": binding(),
        "instances": [{"name": "S", "type": "Source"}, {"name": "B", "type": "Backup"}],
        "connections": [{"from": "S.x_out", "to": "B.x_in"}]}));
    let model = expand(&[lib], &s, None).unwrap().model;
    let backup = model.components.iter().find(|c| c.name == "B").unwrap();
    let tempo = backup
        .automata
        .iter()
        .find(|a| a.name == "y_tempo")
        .unwrap();
    let names: Vec<_> = tempo.transitions.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["y_enable"]);
}
