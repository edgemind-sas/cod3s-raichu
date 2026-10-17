//! Fault-tree generation over computed attributes: what a sensitive
//! function writes is unrolled into the states that govern it, an observer
//! is crossed as its condition, and a top over the model's declared
//! targets is explained down to the failures. Each case is the smallest
//! model in the shape a flow layer emits (a flow variable computed from a
//! failure state, a feared event observing it).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{
    fault_tree, fault_tree_for_targets, BasicLaw, CompiledModel, FaultTree, FaultTreeError,
    FaultTreeSettings, FtNode, GateOp, GateOrigin, GenerationWarning, OriginKind, TreeRepair,
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
fn a_repair_is_attached_to_its_event_not_warned_at_generation() {
    // The repair back to the event's source is a fact of structure: the
    // event carries it with its law; whether it is ignored depends on the
    // measure the tree is quantified in, so generation says nothing.
    let m = system(
        vec![block("B1", 1e-3, Some(0.1)), block("B2", 2e-3, None)],
        "any",
        0.0,
    );
    let tree = targets(&m).expect("tree");
    assert!(tree.warnings.is_empty(), "{:?}", tree.warnings);
    let b1 = tree
        .basic_events
        .iter()
        .find(|e| e.component == "B1")
        .expect("B1");
    assert_eq!(
        b1.repairs,
        vec![TreeRepair {
            transition: "B1.health.repair".to_owned(),
            law: Some(BasicLaw::Exponential { rate: 0.1 }),
            guarded: false,
        }]
    );
    let b2 = tree
        .basic_events
        .iter()
        .find(|e| e.component == "B2")
        .expect("B2");
    assert!(b2.repairs.is_empty());
    // The structure document carries the repair, with its law nested.
    let coded = serde_json::to_value(b1).expect("json");
    assert_eq!(
        coded["repairs"],
        json!([{"transition": "B1.health.repair",
                "law": {"law": "exponential", "rate": 0.1},
                "guarded": false}])
    );
    // A tree written before repairs existed reads back with none.
    let mut old = coded.clone();
    old.as_object_mut().unwrap().remove("repairs");
    let read: raichu_core::TreeEvent = serde_json::from_value(old).expect("read");
    assert!(read.repairs.is_empty());
}

#[test]
fn a_guarded_repair_is_attached_as_guarded() {
    // The repair waits for B2 to be up: a guard on states (an ObjMode2S
    // `not_occ_cond`), counted as available and said to be guarded.
    let mut b1 = block("B1", 1e-3, Some(0.1));
    b1["automata"][0]["transitions"][1]["guard"] = state("B2", "health", "ok");
    let m = system(vec![b1, block("B2", 2e-3, None)], "any", 0.0);
    let tree = targets(&m).expect("tree");
    let b1 = tree
        .basic_events
        .iter()
        .find(|e| e.component == "B1")
        .expect("B1");
    assert_eq!(b1.repairs.len(), 1);
    assert!(b1.repairs[0].guarded);
}

/// A block failed by an external mode, in the shape the muscadet layer
/// emits: the mode automaton `fm` locks its failure on the target's mirror
/// being at rest and its repair on the mirror being failed; the mirror
/// follows the control attribute `ctrl` (true while any impacting mode is
/// in `occ`) through two transitions no draw governs; `up` reads the
/// mirror. `others` are further modes impacting the same target, each an
/// automaton of its own whose `occ` also raises the control.
fn external_block(name: &str, rate: f64, repair: f64, others: &[&str]) -> Json {
    let mode = |automaton: &str| {
        json!({"name": automaton, "states": ["rep", "occ"], "init": "rep",
        "transitions": [
            {"name": "fail", "source": "rep", "targets": ["occ"],
             "guard": state(name, "mirror", "rep"), "distrib": "exp", "rate": rate},
            {"name": "repair", "source": "occ", "targets": ["rep"],
             "guard": state(name, "mirror", "occ"), "distrib": "exp", "rate": repair}
        ]})
    };
    let ctrl_is = |value: bool| {
        json!({"op": "cmp", "cmp": "eq", "lhs": attr(name, "ctrl"),
               "rhs": {"op": "const", "value": boolean(value)}})
    };
    let mut automata = vec![mode("fm")];
    let mut raised = vec![state(name, "fm", "occ")];
    for other in others {
        automata.push(mode(other));
        raised.push(state(name, other, "occ"));
    }
    automata.push(
        json!({"name": "mirror", "states": ["rep", "occ"], "init": "rep",
        "transitions": [
            {"name": "occ", "source": "rep", "targets": ["occ"], "guard": ctrl_is(true),
             "distrib": "delay", "time": 0.0},
            {"name": "rep", "source": "occ", "targets": ["rep"], "guard": ctrl_is(false),
             "distrib": "delay", "time": 0.0}
        ]}),
    );
    json!({
        "name": name,
        "attributes": [{"name": "up", "kind": "bool", "init": boolean(true)},
                       {"name": "ctrl", "kind": "bool", "init": boolean(false)}],
        "automata": automata,
        "sensitive_functions": [
            {"name": "update_ctrl", "effects": [
                {"target": {"component": name, "attribute": "ctrl"},
                 "value": {"op": "bool", "bool_op": "or", "args": raised}}]},
            {"name": "update_up", "effects": [
                {"target": {"component": name, "attribute": "up"},
                 "value": state(name, "mirror", "rep")}]}
        ]
    })
}

#[test]
fn a_lock_the_source_state_implies_is_no_approximation() {
    // The default guards of a muscadet external mode: the failure waits for
    // the mirror at rest, the repair for the mirror failed, and the mirror
    // is at rest exactly when the mode is. Neither guard holds a draw back,
    // and the mirror's own transitions draw nothing.
    let m = system(
        vec![
            external_block("A", 1e-3, 0.1, &[]),
            external_block("B", 2e-3, 0.1, &[]),
        ],
        "any",
        0.0,
    );
    let tree = targets(&m).expect("tree");
    assert_eq!(
        tree.minimal_cut_sets(10).expect("cuts").len(),
        1,
        "{:?}",
        tree.basic_events
    );
    assert_eq!(
        tree.basic_events
            .iter()
            .map(|e| e.name.as_str())
            .collect::<Vec<_>>(),
        ["A.fm.fail", "B.fm.fail"]
    );
    assert!(tree.warnings.is_empty(), "{:?}", tree.warnings);
    for event in &tree.basic_events {
        assert_eq!(event.repairs.len(), 1, "{event:?}");
        assert!(!event.repairs[0].guarded, "{event:?}");
    }
}

#[test]
fn a_lock_shared_with_another_mode_still_conditions_the_draw() {
    // Two modes on one target: `fm` failing waits for the mirror at rest,
    // which `other` failing first prevents. That lock is a real condition
    // and keeps its warning; `fm`'s repair still waits for nothing, since
    // the mirror is failed whenever `fm` is.
    let m = system(
        vec![
            external_block("A", 1e-3, 0.1, &["other"]),
            external_block("B", 2e-3, 0.1, &[]),
        ],
        "any",
        0.0,
    );
    let tree = targets(&m).expect("tree");
    for transition in ["A.fm.fail", "A.other.fail"] {
        assert!(
            tree.warnings.contains(&GenerationWarning::GuardReadsState {
                transition: transition.to_owned(),
            }),
            "{transition}: {:?}",
            tree.warnings
        );
    }
    assert!(
        !tree.warnings.iter().any(
            |w| matches!(w, GenerationWarning::GuardReadsState { transition }
                              if transition.starts_with("B.") || transition.contains("mirror"))
        ),
        "{:?}",
        tree.warnings
    );
    let fail = tree
        .basic_events
        .iter()
        .find(|e| e.name == "A.fm.fail")
        .expect("A fail");
    assert!(!fail.repairs[0].guarded, "{fail:?}");
}

#[test]
fn a_mirror_that_does_not_follow_its_condition_keeps_the_lock_warned() {
    // The same block, its mirror no longer a function of the mode: back
    // to rest under a guard that is not the negation of the one raising
    // it, or raised after a delay. The lock then reads a state that can
    // lag the mode, and stays warned.
    let ctrl_is = |value: bool| {
        json!({"op": "cmp", "cmp": "eq", "lhs": attr("A", "ctrl"),
               "rhs": {"op": "const", "value": boolean(value)}})
    };
    let hysteresis = |mut a: Json| {
        a["automata"][1]["transitions"][1]["guard"] = json!({"op": "bool", "bool_op": "and",
            "args": [ctrl_is(false), state("B", "fm", "rep")]});
        a
    };
    let delayed = |mut a: Json| {
        a["automata"][1]["transitions"][0]["time"] = json!(1.0);
        a
    };
    let variants: [&dyn Fn(Json) -> Json; 2] = [&hysteresis, &delayed];
    for variant in variants {
        let m = system(
            vec![
                variant(external_block("A", 1e-3, 0.1, &[])),
                external_block("B", 2e-3, 0.1, &[]),
            ],
            "any",
            0.0,
        );
        let tree = targets(&m).expect("tree");
        assert!(
            tree.warnings.contains(&GenerationWarning::GuardReadsState {
                transition: "A.fm.fail".to_owned(),
            }),
            "{:?}",
            tree.warnings
        );
        let fail = tree
            .basic_events
            .iter()
            .find(|e| e.name == "A.fm.fail")
            .expect("A fail");
        assert!(fail.repairs[0].guarded, "{fail:?}");
    }
}

#[test]
fn a_zero_delay_latch_keeps_its_guard_warned() {
    // A trip that fires the instant B is down and never resets: tripped
    // outlives the guard once B is repaired, so the tree, reading the
    // guard at the mission time, is not exact, and says so.
    let m = compile(json!({
        "name": "latch",
        "components": [
            block("B", 1e-3, Some(0.1)),
            {"name": "P", "automata": [{"name": "prot", "states": ["armed", "tripped"],
                "init": "armed", "transitions": [
                    {"name": "trip", "source": "armed", "targets": ["tripped"],
                     "guard": state("B", "health", "nok"), "distrib": "delay", "time": 0.0}
                ]}]}
        ]
    }));
    let top: Expr = serde_json::from_value(state("P", "prot", "tripped")).unwrap();
    let tree = fault_tree(&m, &top, &FaultTreeSettings::default()).expect("tree");
    assert!(
        tree.warnings.contains(&GenerationWarning::GuardReadsState {
            transition: "P.prot.trip".to_owned(),
        }),
        "{:?}",
        tree.warnings
    );
}

#[test]
fn a_follower_probe_that_cannot_unroll_leaves_no_false_ring() {
    // Z's repair waits for M's mirror, whose guards read `A = B`, and B is
    // written by a transition's effect, which no tree unrolls. Reading the
    // repair's guard probes M and fails half way through `A`; the next
    // unrolling of `A` (W's guard) must name B's writer, not a ring.
    let a_is = |value: bool| {
        json!({"op": "cmp", "cmp": "eq", "lhs": attr("M", "a"),
               "rhs": {"op": "const", "value": boolean(value)}})
    };
    let mut z = block("Z", 1e-3, Some(0.1));
    z["automata"][0]["transitions"][1]["guard"] = state("M", "mirror", "occ");
    let mut w = block("W", 2e-3, None);
    w["automata"][0]["transitions"][0]["guard"] = a_is(true);
    let m = compile(json!({
        "name": "probe",
        "components": [
            z,
            w,
            {"name": "M",
             "attributes": [{"name": "a", "kind": "bool", "init": boolean(false)},
                            {"name": "b", "kind": "bool", "init": boolean(false)}],
             "automata": [
                 {"name": "mirror", "states": ["rep", "occ"], "init": "rep", "transitions": [
                     {"name": "occ", "source": "rep", "targets": ["occ"], "guard": a_is(true),
                      "distrib": "delay", "time": 0.0},
                     {"name": "rep", "source": "occ", "targets": ["rep"], "guard": a_is(false),
                      "distrib": "delay", "time": 0.0}]},
                 {"name": "k", "states": ["x", "y"], "init": "x", "transitions": [
                     {"name": "set", "source": "x", "targets": ["y"], "distrib": "exp", "rate": 1e-3,
                      "effects": [{"target": {"component": "M", "attribute": "b"},
                                   "value": {"op": "const", "value": boolean(true)}}]}]}
             ],
             "sensitive_functions": [{"name": "update_a", "effects": [
                 {"target": {"component": "M", "attribute": "a"}, "value": attr("M", "b")}]}]}
        ]
    }));
    let top: Expr = serde_json::from_value(json!({"op": "bool", "bool_op": "or",
        "args": [state("Z", "health", "nok"), state("W", "health", "nok")]}))
    .unwrap();
    let error = fault_tree(&m, &top, &FaultTreeSettings::default()).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("effect of transition"), "{message}");
    assert!(!message.contains("ring"), "{message}");
}

#[test]
fn leaving_the_target_for_a_third_state_is_no_repair() {
    // nok is left for `dead`, not for `ok`: that exit is not a repair of
    // the failure; it keeps a generation warning of its own.
    let mut b1 = block("B1", 1e-3, None);
    b1["automata"][0]["states"] = json!(["ok", "nok", "dead"]);
    b1["automata"][0]["transitions"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name": "die", "source": "nok", "targets": ["dead"],
                     "distrib": "exp", "rate": 0.5}));
    let m = system(vec![b1, block("B2", 2e-3, None)], "any", 0.0);
    let tree = targets(&m).expect("tree");
    let fail = tree
        .basic_events
        .iter()
        .find(|e| e.name == "B1.health.fail")
        .expect("B1 fail");
    assert!(fail.repairs.is_empty(), "{:?}", fail.repairs);
    assert!(
        tree.warnings.contains(&GenerationWarning::TargetLeft {
            automaton: "B1.health".to_owned(),
            state: "nok".to_owned(),
            transitions: vec!["B1.health.die".to_owned()],
        }),
        "{:?}",
        tree.warnings
    );
}

#[test]
fn what_makes_the_tree_an_upper_bound_is_said_per_transition() {
    // A failure that can only happen while another block is down: coded
    // with the guarded transition's name.
    let mut b2 = block("B2", 2e-3, None);
    b2["automata"][0]["transitions"][0]["guard"] = state("B1", "health", "nok");
    let m = system(vec![block("B1", 1e-3, None), b2], "any", 0.0);
    let tree = targets(&m).expect("tree");
    let guarded = GenerationWarning::GuardReadsState {
        transition: "B2.health.fail".to_owned(),
    };
    assert!(tree.warnings.contains(&guarded), "{:?}", tree.warnings);
    assert!(guarded.to_string().contains("guarded"), "{guarded}");
    let coded = serde_json::to_value(&guarded).expect("json");
    assert_eq!(
        coded,
        json!({"kind": "guard_reads_state", "transition": "B2.health.fail"})
    );
}

#[test]
fn a_warning_met_twice_is_listed_once() {
    // B1's failure state is left for a third state; B1 is explained on
    // both branches of the top, so that warning is met twice; equal
    // warnings are one.
    let mut b1 = block("B1", 1e-3, None);
    b1["automata"][0]["states"] = json!(["ok", "nok", "dead"]);
    b1["automata"][0]["transitions"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name": "die", "source": "nok", "targets": ["dead"],
                     "distrib": "exp", "rate": 0.5}));
    let m = system(
        vec![b1, block("B2", 2e-3, None), block("B3", 3e-3, None)],
        "any",
        0.0,
    );
    let both = |a: &str, b: &str| {
        json!({"op": "bool", "bool_op": "and",
               "args": [state(a, "health", "nok"), state(b, "health", "nok")]})
    };
    let top: Expr = serde_json::from_value(
        json!({"op": "bool", "bool_op": "or", "args": [both("B1", "B2"), both("B1", "B3")]}),
    )
    .unwrap();
    let tree = fault_tree(&m, &top, &FaultTreeSettings::default()).expect("tree");
    let left: Vec<&GenerationWarning> = tree
        .warnings
        .iter()
        .filter(|w| matches!(w, GenerationWarning::TargetLeft { .. }))
        .collect();
    assert_eq!(left.len(), 1, "{:?}", tree.warnings);
}

// --- gate identity and origin -------------------------------------------

/// Every gate of `node`, in preorder.
fn gates(node: &FtNode) -> Vec<&FtNode> {
    let mut out = Vec::new();
    let mut pending = vec![node];
    while let Some(node) = pending.pop() {
        if let FtNode::Gate { children, .. } = node {
            out.push(node);
            pending.extend(children.iter().rev());
        }
    }
    out
}

fn origin_of(node: &FtNode) -> Option<&GateOrigin> {
    match node {
        FtNode::Gate { origin, .. } => origin.as_ref(),
        _ => None,
    }
}

fn attribute_origin(component: &str, name: &str, negated: bool) -> GateOrigin {
    GateOrigin {
        kind: OriginKind::Attribute,
        component: component.to_owned(),
        name: name.to_owned(),
        negated,
    }
}

/// A battery of two cells whose `out_elec` a function computes as
/// `value`, over the cells' states.
fn batterie(value: Json) -> CompiledModel {
    let cell = |name: &str| {
        json!({"name": name, "states": ["ok", "nok"], "init": "ok",
               "transitions": [{"name": "fail", "source": "ok", "targets": ["nok"],
                                "distrib": "exp", "rate": 1e-3}]})
    };
    compile(json!({"name": "m", "components": [{
        "name": "Batterie",
        "attributes": [{"name": "out_elec", "kind": "bool", "init": boolean(true)}],
        "automata": [cell("cell1"), cell("cell2")],
        "sensitive_functions": [{"name": "update", "effects": [
            {"target": {"component": "Batterie", "attribute": "out_elec"}, "value": value}
        ]}]
    }]}))
}

#[test]
fn an_unrolled_attribute_labels_the_gate_it_becomes() {
    // An attribute true on a cell's failure, read positively by the top.
    let m = batterie(json!({"op": "bool", "bool_op": "or", "args": [
        state("Batterie", "cell1", "nok"), state("Batterie", "cell2", "nok")]}));
    let top: Expr = serde_json::from_value(attr("Batterie", "out_elec")).unwrap();
    let tree = fault_tree(&m, &top, &FaultTreeSettings::default()).expect("tree");
    assert_eq!(
        origin_of(&tree.top),
        Some(&attribute_origin("Batterie", "out_elec", false)),
        "{:?}",
        tree.top
    );
    let coded = serde_json::to_value(&tree.top).expect("json");
    assert_eq!(
        coded["origin"],
        json!({"kind": "attribute", "component": "Batterie", "name": "out_elec",
               "negated": false})
    );
    assert!(coded["id"].is_string(), "{coded}");
}

#[test]
fn a_negation_carried_across_a_label_marks_it_negated() {
    let m = batterie(json!({"op": "bool", "bool_op": "and", "args": [
        state("Batterie", "cell1", "ok"), state("Batterie", "cell2", "ok")]}));
    let top: Expr = serde_json::from_value(
        json!({"op": "bool", "bool_op": "not", "args": [attr("Batterie", "out_elec")]}),
    )
    .unwrap();
    let tree = fault_tree(&m, &top, &FaultTreeSettings::default()).expect("tree");
    assert_eq!(
        origin_of(&tree.top),
        Some(&attribute_origin("Batterie", "out_elec", true))
    );
    assert!(matches!(&tree.top, FtNode::Gate { op: GateOp::Or, .. }));
}

#[test]
fn a_two_state_mode_sits_directly_under_its_attribute_gate() {
    // The state and transition gates over each failure reduce to its
    // basic event: the attribute gate holds the events directly.
    let m = system(
        vec![block("B1", 1e-3, None), block("B2", 2e-3, None)],
        "any",
        0.0,
    );
    let tree = targets(&m).expect("tree");
    let FtNode::Gate {
        op,
        origin,
        children,
        ..
    } = &tree.top
    else {
        panic!("a gate: {:?}", tree.top)
    };
    assert_eq!(*op, GateOp::And);
    assert_eq!(origin.as_ref(), Some(&attribute_origin("T", "fed", true)));
    assert!(
        children.iter().all(|c| matches!(c, FtNode::Basic { .. })),
        "{children:?}"
    );
}

/// A block whose `up` needs two automata in `ok`.
fn twin_block(name: &str) -> Json {
    let health = |automaton: &str| {
        json!({"name": automaton, "states": ["ok", "nok"], "init": "ok",
               "transitions": [{"name": "fail", "source": "ok", "targets": ["nok"],
                                "distrib": "exp", "rate": 1e-3}]})
    };
    json!({
        "name": name,
        "attributes": [{"name": "up", "kind": "bool", "init": boolean(false)}],
        "automata": [health("h1"), health("h2")],
        "sensitive_functions": [{"name": "update_up", "effects": [
            {"target": {"component": name, "attribute": "up"},
             "value": {"op": "bool", "bool_op": "and",
                       "args": [state(name, "h1", "ok"), state(name, "h2", "ok")]}}
        ]}]
    })
}

#[test]
fn a_labelled_gate_is_not_merged_into_a_parent_of_its_connective() {
    // Series: lost when either block is down, each down when either of
    // its automata failed. Unlabelled, the two disjunctions would flatten
    // into the top's.
    let m = system(vec![twin_block("B1"), twin_block("B2")], "all", 0.0);
    let tree = targets(&m).expect("tree");
    let FtNode::Gate {
        op,
        origin,
        children,
        ..
    } = &tree.top
    else {
        panic!("a gate: {:?}", tree.top)
    };
    assert_eq!(*op, GateOp::Or);
    assert_eq!(origin.as_ref(), Some(&attribute_origin("T", "fed", true)));
    let inner: Vec<Option<&GateOrigin>> = children.iter().map(origin_of).collect();
    assert_eq!(
        inner,
        vec![
            Some(&attribute_origin("B1", "up", true)),
            Some(&attribute_origin("B2", "up", true)),
        ]
    );
    for child in children {
        assert!(matches!(child, FtNode::Gate { op: GateOp::Or, .. }));
    }
    // The Boolean function is the flattened one's.
    assert_eq!(
        cut_names(&tree),
        names(&[
            &["B1.h1.fail"],
            &["B1.h2.fail"],
            &["B2.h1.fail"],
            &["B2.h2.fail"]
        ])
    );
}

#[test]
fn every_gate_has_an_identifier_unique_and_stable_across_generations() {
    let m = system(vec![twin_block("B1"), twin_block("B2")], "all", 0.0);
    let first = targets(&m).expect("tree");
    let second = targets(&m).expect("tree");
    let ids: Vec<String> = gates(&first.top)
        .into_iter()
        .map(|g| match g {
            FtNode::Gate { id, .. } => id.clone().expect("an identifier"),
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(ids.len(), 3);
    let unique: std::collections::HashSet<&String> = ids.iter().collect();
    assert_eq!(unique.len(), ids.len(), "{ids:?}");
    assert_eq!(
        serde_json::to_value(&first).unwrap(),
        serde_json::to_value(&second).unwrap()
    );
    // A tree written before gates carried them reads back without.
    let mut old = serde_json::to_value(&first.top).unwrap();
    old.as_object_mut().unwrap().remove("id");
    old.as_object_mut().unwrap().remove("origin");
    let read: FtNode = serde_json::from_value(old).expect("read");
    assert!(matches!(
        read,
        FtNode::Gate {
            id: None,
            origin: None,
            ..
        }
    ));
}

#[test]
fn the_open_psa_gates_are_named_by_their_identifiers() {
    let m = system(vec![twin_block("B1"), twin_block("B2")], "all", 0.0);
    let tree = targets(&m).expect("tree");
    let xml = tree.to_open_psa("t", raichu_core::FaultTreeMeasure::default());
    // The top is the gate named `top`; the others are referenced by name.
    for (position, gate) in gates(&tree.top).into_iter().enumerate() {
        let FtNode::Gate { id: Some(id), .. } = gate else {
            panic!("an identifier")
        };
        assert_eq!(position == 0, id == "top", "{id}");
        assert!(
            xml.contains(&format!("<define-gate name=\"{id}\">")),
            "{id}: {xml}"
        );
        if position > 0 {
            assert!(xml.contains(&format!("<gate name=\"{id}\"/>")), "{id}");
        }
    }
}
