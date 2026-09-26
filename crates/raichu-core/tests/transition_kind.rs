//! Declared transition kind reaches the compiled transition.
//!
//! The exploration driver counts fired `failure` transitions (KTD6 of the
//! sequence-tree plan), and it reads the compiled model, not the document:
//! the declaration must survive compilation, and its absence must too.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::CompiledModel;
use raichu_model::{Model, TransitionKind};

fn model_json(fail_kind: &str, repair_kind: &str) -> String {
    format!(
        r#"{{
  "name": "roles",
  "components": [{{
    "name": "c", "ports": [], "attributes": [],
    "automata": [{{
      "name": "life", "states": ["ok", "ko", "parked"], "init": "ok",
      "transitions": [
        {{"name": "fail", "source": "ok", "targets": ["ko", "parked"], {fail_kind} "distrib": "inst", "probs": [0.3]}},
        {{"name": "rearm", "source": "parked", "targets": ["ok"], "distrib": "delay", "time": 1.0}},
        {{"name": "fix", "source": "ko", "targets": ["ok"], {repair_kind} "distrib": "exp", "rate": 1.0}}]}}]
  }}]
}}"#
    )
}

fn compile(json: &str) -> CompiledModel {
    let model = Model::from_json(json).unwrap();
    model.validate().unwrap();
    CompiledModel::compile(&model).unwrap()
}

#[test]
fn undeclared_kinds_compile_to_none() {
    let compiled = compile(&model_json("", ""));
    assert!(compiled.transitions.iter().all(|t| t.kind.is_none()));
}

#[test]
fn declared_kinds_reach_the_compiled_transition() {
    let compiled = compile(&model_json(r#""kind": "failure","#, r#""kind": "repair","#));
    let kind_of = |name: &str| {
        compiled
            .transitions
            .iter()
            .find(|t| t.name == name)
            .unwrap()
            .kind
    };
    assert_eq!(kind_of("c.life.fail"), Some(TransitionKind::Failure));
    assert_eq!(kind_of("c.life.rearm"), None);
    assert_eq!(kind_of("c.life.fix"), Some(TransitionKind::Repair));
}

#[test]
fn a_failure_draw_keeps_its_failure_state_first() {
    // The failure count credits a `failure` transition only when it enters
    // its FIRST declared target: the compiled target order is the declared
    // one, so the parked branch is never mistaken for the failure.
    let compiled = compile(&model_json(r#""kind": "failure","#, ""));
    let fail = compiled
        .transitions
        .iter()
        .find(|t| t.name == "c.life.fail")
        .unwrap();
    let ko = compiled.automata[fail.automaton]
        .states
        .iter()
        .position(|s| s == "ko")
        .unwrap();
    assert_eq!(fail.targets.len(), 2);
    assert_eq!(fail.targets[0], ko);
}
