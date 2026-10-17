//! A generated tree's gates keep their identity into the quantifier and
//! the OpenPSA document: every gate is named by its occurrence
//! identifier, an attribute read in two places is two gates of one
//! origin, and naming them changes no cut set.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashSet;

use raichu_core::{
    fault_tree, CompiledModel, FaultTree, FaultTreeMeasure, FaultTreeSettings, FtNode, GateOrigin,
    OriginKind,
};
use raichu_expr::Expr;
use raichu_fta::{quantify, read_open_psa, Formula, QuantifySettings, Tree};
use raichu_model::Model;
use serde_json::{json, Value as Json};

fn boolean(value: bool) -> Json {
    json!({"kind": "bool", "value": value})
}

fn state(component: &str, automaton: &str, state: &str) -> Json {
    json!({"op": "state_active", "state": {"component": component, "automaton": automaton, "state": state}})
}

fn not(arg: Json) -> Json {
    json!({"op": "bool", "bool_op": "not", "args": [arg]})
}

fn exp(name: &str, source: &str, target: &str, rate: f64) -> Json {
    json!({"name": name, "source": source, "targets": [target], "distrib": "exp", "rate": rate})
}

/// `A.up` needs both its automata in `ok`, one of them repairable. `S`
/// goes `a -> b`, comes back `b -> a` (a repair loop), and trips
/// `b -> c` only while `A` is down. The top reads `A.up` twice: directly,
/// and through the guard of `S`'s trip, beyond the loop.
fn model() -> CompiledModel {
    let up = json!({"op": "attr", "attr": {"component": "A", "attribute": "up"}});
    let document = json!({"name": "m", "components": [
        {
            "name": "A",
            "attributes": [{"name": "up", "kind": "bool", "init": boolean(true)}],
            "automata": [
                {"name": "h1", "states": ["ok", "nok"], "init": "ok",
                 "transitions": [exp("fail", "ok", "nok", 1e-3), exp("repair", "nok", "ok", 0.1)]},
                {"name": "h2", "states": ["ok", "nok"], "init": "ok",
                 "transitions": [exp("fail", "ok", "nok", 2e-3)]}
            ],
            "sensitive_functions": [{"name": "update", "effects": [
                {"target": {"component": "A", "attribute": "up"},
                 "value": {"op": "bool", "bool_op": "and",
                           "args": [state("A", "h1", "ok"), state("A", "h2", "ok")]}}
            ]}]
        },
        {
            "name": "S",
            "automata": [{"name": "mode", "states": ["a", "b", "c"], "init": "a",
                "transitions": [
                    exp("start", "a", "b", 1e-2),
                    exp("back", "b", "a", 0.5),
                    {"name": "trip", "source": "b", "targets": ["c"], "distrib": "exp",
                     "rate": 1e-2, "guard": not(up.clone())}
                ]}]
        },
        {
            "name": "B",
            "automata": [{"name": "health", "states": ["ok", "nok"], "init": "ok",
                "transitions": [exp("fail", "ok", "nok", 3e-3)]}]
        }
    ]});
    let model: Model = serde_json::from_value(document).expect("model");
    CompiledModel::compile(&model).expect("compile")
}

fn tree() -> FaultTree {
    let up = json!({"op": "attr", "attr": {"component": "A", "attribute": "up"}});
    let top: Expr = serde_json::from_value(json!({"op": "bool", "bool_op": "or", "args": [
        state("S", "mode", "c"),
        {"op": "bool", "bool_op": "and", "args": [not(up), state("B", "health", "nok")]}
    ]}))
    .unwrap();
    fault_tree(&model(), &top, &FaultTreeSettings::default()).expect("tree")
}

fn gates(node: &FtNode) -> Vec<(Option<&str>, Option<&GateOrigin>)> {
    let mut out = Vec::new();
    let mut pending = vec![node];
    while let Some(node) = pending.pop() {
        if let FtNode::Gate {
            id,
            origin,
            children,
            ..
        } = node
        {
            out.push((id.as_deref(), origin.as_ref()));
            pending.extend(children.iter().rev());
        }
    }
    out
}

fn named_cuts(sets: &[Vec<usize>], names: &[String]) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = sets
        .iter()
        .map(|set| {
            let mut n: Vec<String> = set.iter().map(|&i| names[i].clone()).collect();
            n.sort();
            n
        })
        .collect();
    out.sort();
    out
}

fn mcs(tree: &Tree) -> Vec<Vec<String>> {
    let quantified = quantify(
        tree,
        &QuantifySettings {
            mission_time: Some(100.0),
            ..QuantifySettings::default()
        },
    )
    .expect("quantify");
    let names: Vec<String> = tree.events.iter().map(|e| e.name.clone()).collect();
    named_cuts(&quantified.minimal_cut_sets.expect("cut sets"), &names)
}

#[test]
fn an_attribute_read_on_two_branches_is_two_gates_of_one_origin() {
    let tree = tree();
    let a_up = GateOrigin {
        kind: OriginKind::Attribute,
        component: "A".to_owned(),
        name: "up".to_owned(),
        negated: true,
    };
    let all = gates(&tree.top);
    let ids: Vec<&str> = all
        .iter()
        .map(|(id, _)| id.expect("an identifier"))
        .collect();
    assert_eq!(
        ids.iter().collect::<HashSet<_>>().len(),
        ids.len(),
        "{ids:?}"
    );
    let reads: Vec<&str> = all
        .iter()
        .filter(|(_, origin)| *origin == Some(&a_up))
        .map(|(id, _)| id.unwrap())
        .collect();
    assert_eq!(reads.len(), 2, "{all:?}");
    assert_ne!(reads[0], reads[1]);
}

#[test]
fn the_open_psa_document_reads_back_with_its_gates_and_the_same_cut_sets() {
    let generated = tree();
    let names: Vec<String> = generated
        .basic_events
        .iter()
        .map(|e| e.name.clone())
        .collect();
    let expected = named_cuts(&generated.minimal_cut_sets(10_000).unwrap(), &names);
    assert_eq!(expected.len(), 4, "{expected:?}");
    let xml = generated.to_open_psa("t", FaultTreeMeasure::default());
    // `read_open_psa` refuses a gate defined twice.
    let back = read_open_psa(&xml, None).expect("read back");
    let gate_names: HashSet<&str> = back.gates.iter().map(|g| g.name.as_str()).collect();
    let ids: HashSet<&str> = gates(&generated.top)
        .into_iter()
        .map(|(id, _)| id.unwrap())
        .collect();
    assert_eq!(gate_names, ids);
    assert_eq!(mcs(&back), expected);
}

#[test]
fn a_generated_tree_reaches_the_quantifier_with_its_gates_named() {
    let generated = tree();
    let tree = Tree::from_generated(&generated, "t");
    let ids: Vec<&str> = gates(&generated.top)
        .into_iter()
        .map(|(id, _)| id.unwrap())
        .collect();
    // The top stays the tree's formula; every gate below it is a `GateDef`
    // named by its identifier.
    let defined: Vec<&str> = tree.gates.iter().map(|g| g.name.as_str()).collect();
    let mut nested: Vec<&str> = ids[1..].to_vec();
    nested.sort_unstable();
    let mut sorted = defined.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, nested);
    assert!(!matches!(tree.top, Formula::Gate { .. }));
    let names: Vec<String> = generated
        .basic_events
        .iter()
        .map(|e| e.name.clone())
        .collect();
    assert_eq!(
        mcs(&tree),
        named_cuts(&generated.minimal_cut_sets(10_000).unwrap(), &names)
    );
}
