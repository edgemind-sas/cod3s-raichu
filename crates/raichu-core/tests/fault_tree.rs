//! Fault-tree generation against block diagrams whose minimal cut sets are
//! known by construction.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{fault_tree, CompiledModel, FaultTreeError, FaultTreeSettings};
use raichu_expr::{Expr, Value};
use raichu_model::Model;
use serde_json::{json, Value as Json};

/// A component `name` failing `ok -> nok` at `rate`, optionally repairing,
/// optionally guarded.
fn unit(name: &str, rate: f64, repair: Option<f64>, guard: Option<Json>) -> Json {
    let mut fail = json!({
        "name": "fail", "source": "ok", "targets": ["nok"],
        "distrib": "exp", "rate": rate
    });
    if let Some(guard) = guard {
        fail["guard"] = guard;
    }
    let mut transitions = vec![fail];
    if let Some(mu) = repair {
        transitions.push(json!({
            "name": "repair", "source": "nok", "targets": ["ok"],
            "distrib": "exp", "rate": mu
        }));
    }
    json!({
        "name": name,
        "attributes": [
            {"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}
        ],
        "automata": [{
            "name": "health", "states": ["ok", "nok"], "init": "ok",
            "transitions": transitions
        }]
    })
}

fn nok(name: &str) -> Json {
    json!({"op": "state_active", "state": {"component": name, "automaton": "health", "state": "nok"}})
}

fn model(components: Vec<Json>) -> CompiledModel {
    let document = json!({"name": "ft", "components": components});
    let model: Model = serde_json::from_value(document).expect("model");
    CompiledModel::compile(&model).expect("compile")
}

fn expr(value: Json) -> Expr {
    serde_json::from_value(value).expect("expr")
}

/// The minimal cut sets as sorted lists of event names.
fn cuts(model: &CompiledModel, top: Json, settings: &FaultTreeSettings) -> Vec<Vec<String>> {
    let tree = fault_tree(model, &expr(top), settings).expect("tree");
    tree.minimal_cut_sets(10_000)
        .expect("cut sets")
        .into_iter()
        .map(|set| {
            let mut names: Vec<String> = set
                .into_iter()
                .map(|i| tree.basic_events[i].name.clone())
                .collect();
            names.sort();
            names
        })
        .collect()
}

fn names(list: &[&[&str]]) -> Vec<Vec<String>> {
    list.iter()
        .map(|set| set.iter().map(|s| (*s).to_owned()).collect())
        .collect()
}

#[test]
fn a_parallel_pair_fails_on_both() {
    let m = model(vec![
        unit("A", 1e-3, None, None),
        unit("B", 2e-3, None, None),
    ]);
    let top = json!({"op": "bool", "bool_op": "and", "args": [nok("A"), nok("B")]});
    assert_eq!(
        cuts(&m, top, &FaultTreeSettings::default()),
        names(&[&["A.health.fail", "B.health.fail"]])
    );
}

#[test]
fn a_series_pair_fails_on_either() {
    let m = model(vec![
        unit("A", 1e-3, None, None),
        unit("B", 2e-3, None, None),
    ]);
    let top = json!({"op": "bool", "bool_op": "or", "args": [nok("A"), nok("B")]});
    assert_eq!(
        cuts(&m, top, &FaultTreeSettings::default()),
        names(&[&["A.health.fail"], &["B.health.fail"]])
    );
}

#[test]
fn a_two_out_of_three_vote_has_three_pairs() {
    let m = model(vec![
        unit("A", 1e-3, None, None),
        unit("B", 1e-3, None, None),
        unit("C", 1e-3, None, None),
    ]);
    let vote = |name| {
        json!({"op": "if", "cond": nok(name),
        "then": {"op": "const", "value": {"kind": "int", "value": 1}},
        "otherwise": {"op": "const", "value": {"kind": "int", "value": 0}}})
    };
    let top = json!({"op": "cmp", "cmp": "ge",
        "lhs": {"op": "add", "args": [vote("A"), vote("B"), vote("C")]},
        "rhs": {"op": "const", "value": {"kind": "int", "value": 2}}});
    assert_eq!(
        cuts(&m, top, &FaultTreeSettings::default()),
        names(&[
            &["A.health.fail", "B.health.fail"],
            &["A.health.fail", "C.health.fail"],
            &["B.health.fail", "C.health.fail"],
        ])
    );
}

#[test]
fn a_repair_loop_is_handled_and_not_needed() {
    let m = model(vec![unit("A", 1e-3, Some(0.1), None)]);
    let tree = fault_tree(&m, &expr(nok("A")), &FaultTreeSettings::default()).expect("tree");
    assert_eq!(tree.basic_events.len(), 1, "the repair explains nothing");
    assert_eq!(
        cuts(&m, nok("A"), &FaultTreeSettings::default()),
        names(&[&["A.health.fail"]])
    );
}

#[test]
fn a_guard_on_another_state_is_a_cascade() {
    let m = model(vec![
        unit("A", 1e-3, None, None),
        unit("B", 2e-3, None, Some(nok("A"))),
    ]);
    assert_eq!(
        cuts(&m, nok("B"), &FaultTreeSettings::default()),
        names(&[&["A.health.fail", "B.health.fail"]])
    );
}

#[test]
fn a_guard_on_a_frozen_attribute_is_a_constant_a_profile_moves() {
    let guard = json!({"op": "cmp", "cmp": "gt",
        "lhs": {"op": "attr", "attr": {"component": "B", "attribute": "x"}},
        "rhs": {"op": "const", "value": {"kind": "float", "value": 5.0}}});
    let m = model(vec![unit("B", 2e-3, None, Some(guard))]);
    // Frozen at its initial 0: the transition can never fire.
    assert!(cuts(&m, nok("B"), &FaultTreeSettings::default()).is_empty());
    // Held at 10 by a profile: it can.
    let profile = FaultTreeSettings {
        profile: vec![("B.x".to_owned(), Value::Float(10.0))],
        ..FaultTreeSettings::default()
    };
    assert_eq!(cuts(&m, nok("B"), &profile), names(&[&["B.health.fail"]]));
}

#[test]
fn a_negated_state_is_refused() {
    let m = model(vec![unit("A", 1e-3, None, None)]);
    let ok = json!({"op": "state_active", "state": {"component": "A", "automaton": "health", "state": "ok"}});
    let top = json!({"op": "bool", "bool_op": "not", "args": [ok]});
    let error = fault_tree(&m, &expr(top), &FaultTreeSettings::default()).unwrap_err();
    assert!(matches!(error, FaultTreeError::NonCoherent(_)), "{error}");
}

#[test]
fn an_unknown_state_is_refused_by_name() {
    let m = model(vec![unit("A", 1e-3, None, None)]);
    let error = fault_tree(&m, &expr(nok("Z")), &FaultTreeSettings::default()).unwrap_err();
    assert_eq!(error, FaultTreeError::Unresolved("Z.health".to_owned()));
}

#[test]
fn the_tree_exports_to_open_psa() {
    let m = model(vec![
        unit("A", 1e-3, None, None),
        unit("B", 2e-3, None, None),
    ]);
    let top = json!({"op": "bool", "bool_op": "and", "args": [nok("A"), nok("B")]});
    let tree = fault_tree(&m, &expr(top), &FaultTreeSettings::default()).expect("tree");
    let xml = tree.to_open_psa("parallel");
    assert!(xml.contains("<define-fault-tree name=\"parallel\">"));
    assert!(xml.contains(
        "<and><basic-event name=\"A.health.fail\"/><basic-event name=\"B.health.fail\"/></and>"
    ));
    assert!(
        xml.contains("<exponential><float value=\"0.001\"/><system-mission-time/></exponential>")
    );
}
