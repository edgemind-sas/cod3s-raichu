//! Fault-tree generation over computed attributes: what a sensitive
//! function writes is unrolled into the states that govern it, an observer
//! is crossed as its condition, and a top over the model's declared
//! targets is explained down to the failures. Each case is the smallest
//! model in the shape a flow layer emits (a flow variable computed from a
//! failure state, a feared event observing it).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{
    fault_tree, fault_tree_for_targets, CompiledModel, FaultTree, FaultTreeError, FaultTreeSettings,
};
use raichu_expr::Expr;
use raichu_model::Model;
use serde_json::{json, Value as Json};

fn boolean(value: bool) -> Json {
    json!({"kind": "bool", "value": value})
}

fn attr(component: &str, attribute: &str) -> Json {
    json!({"op": "attr", "attr": {"component": component, "attribute": attribute}})
}

fn state(component: &str, automaton: &str, state: &str) -> Json {
    json!({"op": "state_active", "state": {"component": component, "automaton": automaton, "state": state}})
}

/// A block whose `up` attribute a function computes from its failure
/// automaton: `up = (health == ok)`. Optionally repairable.
fn block(name: &str, rate: f64, repair: Option<f64>) -> Json {
    let mut transitions = vec![json!({
        "name": "fail", "source": "ok", "targets": ["nok"], "distrib": "exp", "rate": rate
    })];
    if let Some(mu) = repair {
        transitions.push(json!({
            "name": "repair", "source": "nok", "targets": ["ok"], "distrib": "exp", "rate": mu
        }));
    }
    json!({
        "name": name,
        "attributes": [{"name": "up", "kind": "bool", "init": boolean(false)}],
        "automata": [{"name": "health", "states": ["ok", "nok"], "init": "ok",
                      "transitions": transitions}],
        "sensitive_functions": [{"name": "update_up", "effects": [
            {"target": {"component": name, "attribute": "up"},
             "value": state(name, "health", "ok")}
        ]}]
    })
}

/// A sink whose `fed` attribute is `logic` over the blocks' `up`, through
/// an in-port: `any` for a parallel system, `all` for a series one.
fn sink(logic: &str) -> Json {
    json!({
        "name": "T",
        "attributes": [{"name": "fed", "kind": "bool", "init": boolean(false)}],
        "ports": [{"name": "in", "dir": "in"}],
        "sensitive_functions": [{"name": "update_fed", "effects": [
            {"target": {"component": "T", "attribute": "fed"},
             "value": {"op": "port_agg", "port": {"component": "T", "port": "in"}, "agg": logic}}
        ]}]
    })
}

/// The feared event: an observer entering `occ` as soon as `cond` holds,
/// delayed by `delay`.
fn observer(cond: Json, delay: f64) -> Json {
    json!({
        "name": "lost",
        "automata": [{"name": "ev", "states": ["not_occ", "occ"], "init": "not_occ",
            "transitions": [
                {"name": "occ", "source": "not_occ", "targets": ["occ"], "guard": cond,
                 "kind": "observation", "distrib": "delay", "time": delay},
                {"name": "not_occ", "source": "occ", "targets": ["not_occ"],
                 "guard": {"op": "bool", "bool_op": "not", "args": [cond]},
                 "kind": "observation", "distrib": "delay", "time": 0.0}
            ]}]
    })
}

fn with_ports(mut component: Json) -> Json {
    component["ports"] = json!([{"name": "out", "dir": "out", "attr": "up"}]);
    component
}

/// Blocks `names` wired into the sink, the feared event "the sink is not
/// fed" declared as the target `lost`.
fn system(blocks: Vec<Json>, logic: &str, delay: f64) -> CompiledModel {
    let names: Vec<String> = blocks
        .iter()
        .map(|b| b["name"].as_str().unwrap().to_owned())
        .collect();
    let mut components: Vec<Json> = blocks.into_iter().map(with_ports).collect();
    components.push(sink(logic));
    let cond = json!({"op": "cmp", "cmp": "eq", "lhs": attr("T", "fed"),
                      "rhs": {"op": "const", "value": boolean(false)}});
    components.push(observer(cond, delay));
    let connections: Vec<Json> = names
        .iter()
        .map(|n| json!({"from": {"component": n, "port": "out"}, "to": {"component": "T", "port": "in"}}))
        .collect();
    compile(json!({
        "name": "rbd",
        "components": components,
        "connections": connections,
        "targets": [{"name": "lost", "component": "lost", "automaton": "ev", "state": "occ"}]
    }))
}

fn compile(document: Json) -> CompiledModel {
    let model: Model = serde_json::from_value(document).expect("model");
    CompiledModel::compile(&model).expect("compile")
}

fn targets(model: &CompiledModel) -> Result<FaultTree, FaultTreeError> {
    fault_tree_for_targets(model, &["lost".to_owned()], &FaultTreeSettings::default())
}

fn cut_names(tree: &FaultTree) -> Vec<Vec<String>> {
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
fn a_parallel_target_is_explained_down_to_both_failures() {
    let m = system(
        vec![block("B1", 1e-3, None), block("B2", 2e-3, None)],
        "any",
        0.0,
    );
    let tree = targets(&m).expect("tree");
    assert_eq!(
        cut_names(&tree),
        names(&[&["B1.health.fail", "B2.health.fail"]])
    );
    // The observer is crossed, never a basic event of its own.
    assert!(tree.basic_events.iter().all(|e| e.component != "lost"));
    assert!(tree.warnings.is_empty(), "{:?}", tree.warnings);
}

#[test]
fn a_series_target_fails_on_either() {
    let m = system(
        vec![block("B1", 1e-3, None), block("B2", 2e-3, None)],
        "all",
        0.0,
    );
    assert_eq!(
        cut_names(&targets(&m).expect("tree")),
        names(&[&["B1.health.fail"], &["B2.health.fail"]])
    );
}

#[test]
fn the_observer_state_as_an_explicit_top_is_crossed_too() {
    let m = system(
        vec![block("B1", 1e-3, None), block("B2", 2e-3, None)],
        "any",
        0.0,
    );
    let top: Expr = serde_json::from_value(state("lost", "ev", "occ")).unwrap();
    let tree = fault_tree(&m, &top, &FaultTreeSettings::default()).expect("tree");
    assert_eq!(
        cut_names(&tree),
        names(&[&["B1.health.fail", "B2.health.fail"]])
    );
}

#[test]
fn a_flow_attribute_as_an_explicit_top_is_unrolled_not_frozen() {
    let m = system(
        vec![block("B1", 1e-3, None), block("B2", 2e-3, None)],
        "any",
        0.0,
    );
    // `T.fed` starts false (before the first fixpoint): frozen, `not fed`
    // would be the constant true.
    let top: Expr =
        serde_json::from_value(json!({"op": "bool", "bool_op": "not", "args": [attr("T", "fed")]}))
            .unwrap();
    let tree = fault_tree(&m, &top, &FaultTreeSettings::default()).expect("tree");
    assert_eq!(
        cut_names(&tree),
        names(&[&["B1.health.fail", "B2.health.fail"]])
    );
}

#[test]
fn a_count_below_a_threshold_is_a_vote_on_the_failures() {
    // Lost when fewer than two of three blocks are up: at least two down.
    let mut components: Vec<Json> = ["B1", "B2", "B3"]
        .iter()
        .map(|n| with_ports(block(n, 1e-3, None)))
        .collect();
    let sum = json!({"op": "port_agg", "port": {"component": "T", "port": "in"}, "agg": "sum"});
    let mut t = sink("any");
    t["sensitive_functions"][0]["effects"][0]["value"] = json!({"op": "cmp", "cmp": "ge",
        "lhs": sum, "rhs": {"op": "const", "value": {"kind": "int", "value": 2}}});
    components.push(t);
    let cond = json!({"op": "cmp", "cmp": "eq", "lhs": attr("T", "fed"),
                      "rhs": {"op": "const", "value": boolean(false)}});
    components.push(observer(cond, 0.0));
    let connections: Vec<Json> = ["B1", "B2", "B3"]
        .iter()
        .map(|n| json!({"from": {"component": n, "port": "out"}, "to": {"component": "T", "port": "in"}}))
        .collect();
    let m = compile(json!({
        "name": "two_of_three", "components": components,
        "connections": connections,
        "targets": [{"name": "lost", "component": "lost", "automaton": "ev", "state": "occ"}]
    }));
    assert_eq!(
        cut_names(&targets(&m).expect("tree")),
        names(&[
            &["B1.health.fail", "B2.health.fail"],
            &["B1.health.fail", "B3.health.fail"],
            &["B2.health.fail", "B3.health.fail"],
        ])
    );
}

#[test]
fn an_ungoverned_condition_is_a_degenerate_tree_refused_with_its_cause() {
    // The sink is fed by a constant nothing computes from a state: the
    // event can never occur.
    let constant = json!({
        "name": "C",
        "attributes": [{"name": "up", "kind": "bool", "init": boolean(true)}],
        "ports": [{"name": "out", "dir": "out", "attr": "up"}]
    });
    let cond = json!({"op": "cmp", "cmp": "eq", "lhs": attr("T", "fed"),
                      "rhs": {"op": "const", "value": boolean(false)}});
    let m = compile(json!({
        "name": "ungoverned",
        "components": [constant, sink("any"), observer(cond, 0.0)],
        "connections": [{"from": {"component": "C", "port": "out"}, "to": {"component": "T", "port": "in"}}],
        "targets": [{"name": "lost", "component": "lost", "automaton": "ev", "state": "occ"}]
    }));
    let error = targets(&m).unwrap_err();
    match &error {
        FaultTreeError::Degenerate { value, cause } => {
            assert!(!value);
            assert!(cause.contains("lost.ev.occ"), "{cause}");
            assert!(cause.contains("reads no state"), "{cause}");
        }
        other => panic!("expected a degenerate tree, got {other}"),
    }
}

#[test]
fn a_waiting_observer_is_refused_rather_than_taken_for_a_failure() {
    let m = system(vec![block("B1", 1e-3, None)], "any", 5.0);
    let error = targets(&m).unwrap_err();
    assert!(
        matches!(&error, FaultTreeError::Unsupported(d) if d.contains("observer") && d.contains("lost.ev.occ")),
        "{error}"
    );
}

#[test]
fn an_attribute_a_transition_effect_writes_is_refused_by_name() {
    // `count` is bumped by the failure's edge effect: no expression of the
    // states says what it is.
    let mut b = block("B1", 1e-3, None);
    b["attributes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name": "count", "kind": "int", "init": {"kind": "int", "value": 0}}));
    b["automata"][0]["transitions"][0]["effects"] = json!([{
        "target": {"component": "B1", "attribute": "count"},
        "value": {"op": "const", "value": {"kind": "int", "value": 1}}
    }]);
    let m = compile(json!({"name": "m", "components": [b]}));
    let top: Expr = serde_json::from_value(json!({"op": "cmp", "cmp": "ge",
        "lhs": attr("B1", "count"), "rhs": {"op": "const", "value": {"kind": "int", "value": 1}}}))
    .unwrap();
    let error = fault_tree(&m, &top, &FaultTreeSettings::default()).unwrap_err();
    assert!(
        matches!(&error, FaultTreeError::Unsupported(d)
            if d.contains("B1.count") && d.contains("effect of transition")),
        "{error}"
    );
}

#[test]
fn a_ring_of_attributes_is_refused_by_name() {
    let ring = json!({
        "name": "R",
        "attributes": [
            {"name": "a", "kind": "bool", "init": boolean(false)},
            {"name": "b", "kind": "bool", "init": boolean(false)}
        ],
        "automata": [{"name": "health", "states": ["ok", "nok"], "init": "ok",
            "transitions": [{"name": "fail", "source": "ok", "targets": ["nok"],
                             "distrib": "exp", "rate": 1e-3}]}],
        "sensitive_functions": [
            {"name": "set_a", "effects": [{"target": {"component": "R", "attribute": "a"},
                "value": {"op": "bool", "bool_op": "or", "args": [attr("R", "b"), state("R", "health", "nok")]}}]},
            {"name": "set_b", "effects": [{"target": {"component": "R", "attribute": "b"},
                "value": attr("R", "a")}]}
        ]
    });
    let m = compile(json!({"name": "m", "components": [ring]}));
    let top: Expr = serde_json::from_value(attr("R", "a")).unwrap();
    let error = fault_tree(&m, &top, &FaultTreeSettings::default()).unwrap_err();
    assert!(
        matches!(&error, FaultTreeError::Unsupported(d) if d.contains("R.a") && d.contains("ring")),
        "{error}"
    );
}

#[test]
fn an_unknown_target_is_refused_naming_the_declared_ones() {
    let m = system(vec![block("B1", 1e-3, None)], "any", 0.0);
    let error = fault_tree_for_targets(&m, &["nope".to_owned()], &FaultTreeSettings::default())
        .unwrap_err();
    assert_eq!(
        error,
        FaultTreeError::UnknownTarget {
            name: "nope".to_owned(),
            declared: vec!["lost".to_owned()],
        }
    );
}

#[test]
fn what_makes_the_tree_an_upper_bound_is_said_per_transition() {
    // A repair the tree ignores.
    let m = system(
        vec![block("B1", 1e-3, Some(0.1)), block("B2", 2e-3, None)],
        "any",
        0.0,
    );
    let tree = targets(&m).expect("tree");
    assert_eq!(tree.warnings.len(), 1, "{:?}", tree.warnings);
    assert!(
        tree.warnings[0].contains("B1.health.repair"),
        "{:?}",
        tree.warnings
    );

    // A failure that can only happen while another block is down.
    let mut b2 = block("B2", 2e-3, None);
    b2["automata"][0]["transitions"][0]["guard"] = state("B1", "health", "nok");
    let m = system(vec![block("B1", 1e-3, None), b2], "any", 0.0);
    let tree = targets(&m).expect("tree");
    assert!(
        tree.warnings
            .iter()
            .any(|w| w.contains("B2.health.fail") && w.contains("guarded")),
        "{:?}",
        tree.warnings
    );
}
