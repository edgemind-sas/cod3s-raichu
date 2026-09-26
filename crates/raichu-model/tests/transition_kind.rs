//! Declared transition kind: a transition may be declared a `failure` or a
//! `repair`, so that a driver counting failures along a sequence reads a
//! declaration instead of guessing from state names.
//!
//! The field is optional and serde-defaulted to absent, so every model
//! written before it keeps loading unchanged.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_model::{LoadError, Model, TransitionKind};

/// One component with a failure and a repair edge. `fail_kind` and
/// `repair_kind` are spliced verbatim before `"distrib"` (empty = absent).
fn model_json(fail_kind: &str, repair_kind: &str) -> String {
    format!(
        r#"{{
  "name": "roles",
  "components": [{{
    "name": "c", "ports": [], "attributes": [],
    "automata": [{{
      "name": "life", "states": ["ok", "ko"], "init": "ok",
      "transitions": [
        {{"name": "fail", "source": "ok", "targets": ["ko"], {fail_kind} "distrib": "exp", "rate": 0.1}},
        {{"name": "fix", "source": "ko", "targets": ["ok"], {repair_kind} "distrib": "exp", "rate": 1.0}}]}}]
  }}]
}}"#
    )
}

fn transitions(model: &Model) -> Vec<&raichu_model::Transition> {
    model.components[0].automata[0].transitions.iter().collect()
}

#[test]
fn a_model_without_kind_loads_with_no_kind() {
    let model = Model::from_json(&model_json("", "")).unwrap();
    model.validate().unwrap();
    assert!(transitions(&model).iter().all(|t| t.kind.is_none()));
}

#[test]
fn declared_kinds_load_and_round_trip() {
    let json = model_json(r#""kind": "failure","#, r#""kind": "repair","#);
    let model = Model::from_json(&json).unwrap();
    model.validate().unwrap();
    let ts = transitions(&model);
    assert_eq!(ts[0].kind, Some(TransitionKind::Failure));
    assert_eq!(ts[1].kind, Some(TransitionKind::Repair));

    let written = model.to_json().unwrap();
    let reread = Model::from_json(&written).unwrap();
    assert_eq!(reread, model);
}

#[test]
fn an_absent_kind_is_not_serialized() {
    let model = Model::from_json(&model_json("", "")).unwrap();
    let written = model.to_json().unwrap();
    assert!(
        !written.contains("\"kind\""),
        "an undeclared kind must stay absent from the written document: {written}"
    );
}

#[test]
fn a_declared_kind_requires_no_feature() {
    // The kind changes no simulated number, so a bare body may carry it.
    let json = model_json(r#""kind": "failure","#, "");
    let model = Model::from_json(&json).unwrap();
    assert!(model.required_features().is_empty());
}

#[test]
fn an_unknown_kind_is_refused_naming_the_value() {
    let json = model_json(r#""kind": "breakdown","#, "");
    let err = Model::from_json(&json).expect_err("an unknown kind must be refused");
    assert!(
        matches!(err, LoadError::Json(_)),
        "unexpected error: {err:?}"
    );
    let message = err.to_string();
    assert!(
        message.contains("breakdown"),
        "the refusal must name the offending value: {message}"
    );
}
