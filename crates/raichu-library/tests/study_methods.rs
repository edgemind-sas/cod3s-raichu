//! The quantification section of a study.
//!
//! A study names one quantification: a target, a horizon, and a method with
//! its settings, written as the quantification envelope records it. The
//! answers are compared with direct quantifications in the Python unit
//! suite (`test_library_study_quantification.py`); this file pins what the
//! expander owns.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_library::{expand, parse_library, parse_study, parse_system, QuantificationMethodName};
use serde_json::{json, Value};

/// The method names a study accepts are the quantification contract's,
/// in its order: a method the contract adds must reach the format.
#[test]
fn the_study_names_every_quantification_method() {
    assert_eq!(
        QuantificationMethodName::NAMES,
        raichu_quantify::Method::NAMES
    );
}

fn documents(quantification: Value) -> Result<raichu_library::Expansion, String> {
    let library = parse_library(
        &json!({"raichu_library": {"format": 1}, "library": {
        "name": "units", "version": "1.0.0",
        "types": [{"name": "Unit", "body": {"automata": [{
            "name": "health", "states": ["ok", "ko"], "init": "ok",
            "transitions": [{"name": "fail", "source": "ok", "targets": ["ko"],
                             "distrib": "exp", "rate": 0.001}]}]}}]}})
        .to_string(),
    )
    .unwrap();
    let system = parse_system(
        &json!({"raichu_system": {"format": 1}, "system": {
        "name": "single", "library": {"name": "units", "version": "1.0.0"},
        "instances": [{"name": "U", "type": "Unit"}],
        "targets": [{"name": "down", "component": "U", "automaton": "health", "state": "ko"}]}})
        .to_string(),
    )
    .unwrap();
    let study = parse_study(
        &json!({"raichu_study": {"format": 1}, "study": {
        "name": "q", "system": "single", "quantification": quantification}})
        .to_string(),
    )
    .map_err(|e| e.to_string())?;
    expand(&[library], &system, Some(&study)).map_err(|e| e.to_string())
}

#[test]
fn a_quantification_reaches_the_expansion_as_written() {
    let section = json!({"target": "down", "horizon": 100.0, "seed": 4,
        "method": {"name": "exact", "settings": {"max_length": 3}}});
    let expansion = documents(section.clone()).unwrap();
    assert_eq!(
        serde_json::to_value(expansion.quantification.unwrap()).unwrap(),
        section
    );
}

#[test]
fn a_target_nobody_declares_is_refused_with_the_declared_ones() {
    let error = documents(json!({"target": "nowhere", "horizon": 100.0,
        "method": {"name": "exact"}}))
    .unwrap_err();
    assert!(error.contains("/study/quantification/target"), "{error}");
    assert!(
        error.contains("`nowhere`") && error.contains("`down`"),
        "{error}"
    );
}

#[test]
fn a_negative_horizon_is_refused() {
    let error = documents(json!({"target": "down", "horizon": -1.0,
        "method": {"name": "exact"}}))
    .unwrap_err();
    assert!(error.contains("/study/quantification/horizon"), "{error}");
}

#[test]
fn reporting_instants_outside_the_horizon_are_refused_at_their_pointer() {
    let error = documents(
        json!({"target": "down", "horizon": 10.0, "instants": [5.0, 12.0],
        "method": {"name": "monte_carlo", "settings": {"nb_runs": 10}}}),
    )
    .unwrap_err();
    assert!(
        error.contains("/study/quantification/instants/1"),
        "{error}"
    );
    assert!(error.contains("[0, horizon = 10]"), "{error}");
}

#[test]
fn reporting_instants_out_of_order_are_refused_at_their_pointer() {
    let error = documents(
        json!({"target": "down", "horizon": 10.0, "instants": [5.0, 5.0],
        "method": {"name": "monte_carlo", "settings": {"nb_runs": 10}}}),
    )
    .unwrap_err();
    assert!(
        error.contains("/study/quantification/instants/1"),
        "{error}"
    );
    assert!(error.contains("strictly ascending"), "{error}");
}

#[test]
fn ascending_instants_inside_the_horizon_are_kept() {
    let expansion = documents(
        json!({"target": "down", "horizon": 10.0, "instants": [0.0, 5.0, 10.0],
        "method": {"name": "monte_carlo", "settings": {"nb_runs": 10}}}),
    )
    .unwrap();
    assert_eq!(
        expansion.quantification.unwrap().instants,
        Some(vec![0.0, 5.0, 10.0])
    );
}

#[test]
fn an_unknown_method_is_refused_when_the_study_is_read() {
    let error = documents(json!({"target": "down", "horizon": 1.0,
        "method": {"name": "annealing"}}))
    .unwrap_err();
    assert!(error.contains("annealing"), "{error}");
}

#[test]
fn a_study_without_quantification_carries_none() {
    let study = parse_study(
        &json!({"raichu_study": {"format": 1}, "study": {"name": "q", "system": "single"}})
            .to_string(),
    )
    .unwrap();
    assert!(study.quantification.is_none());
}
