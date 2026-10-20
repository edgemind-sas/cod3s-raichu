//! Automatic exploration preserves the engine's arming waves and observer rank.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::CompiledModel;
use raichu_explore::{
    explore_discretised, explore_exact, DiscretisedSettings, ExactSettings, ExplorationResult,
};
use raichu_model::Model;
use serde_json::{json, Value};

fn active(component: &str, state: &str) -> Value {
    json!({"op": "state_active", "state": {
        "component": component, "automaton": "a", "state": state
    }})
}

fn compile(components: Vec<Value>, targets: Value) -> CompiledModel {
    let model: Model = serde_json::from_value(json!({
        "name": "firing_order", "components": components, "targets": targets
    }))
    .unwrap();
    CompiledModel::compile(&model).unwrap()
}

fn observer(name: &str, guard: Value) -> Value {
    json!({"name": name, "automata": [{"name": "a", "states": ["off", "on"],
        "init": "off", "transitions": [{"name": "detect", "source": "off",
            "targets": ["on"], "distrib": "delay", "time": 0.0,
            "kind": "observation", "guard": guard}]}]})
}

fn target(name: &str, component: &str) -> Value {
    json!({"name": name, "component": component, "automaton": "a", "state": "on"})
}

fn repaired_failure(observer_first: bool) -> CompiledModel {
    let plant = json!({"name": "plant", "automata": [{"name": "a",
        "states": ["ok", "failed", "repaired"], "init": "ok", "transitions": [
            {"name": "fail", "source": "ok", "targets": ["failed"],
             "distrib": "exp", "rate": 1.0},
            {"name": "repair", "source": "failed", "targets": ["repaired"],
             "distrib": "delay", "time": 0.0}]}]});
    let watch = observer("observer", active("plant", "failed"));
    let components = if observer_first {
        vec![watch, plant]
    } else {
        vec![plant, watch]
    };
    compile(components, json!([target("lost", "observer")]))
}

fn assert_probability(result: &ExplorationResult, expected: f64) {
    assert!(
        (result.lower - expected).abs() < 1e-12,
        "{} != {expected}",
        result.lower
    );
    assert!(
        (result.upper - expected).abs() < 1e-12,
        "{} != {expected}",
        result.upper
    );
    assert!(!result.inconclusive);
}

#[test]
fn exact_observes_failure_before_zero_delay_repair_in_either_declaration_order() {
    for observer_first in [false, true] {
        let result = explore_exact(
            &repaired_failure(observer_first),
            &ExactSettings::new("lost", 1.0),
        )
        .unwrap();
        assert_probability(&result, 1.0 - (-1.0_f64).exp());
        let path: Vec<_> = result
            .sequence_steps(0)
            .unwrap()
            .map(|s| s.transition.as_str())
            .collect();
        assert_eq!(
            path,
            ["plant.a.fail", "observer.a.detect", "plant.a.repair"]
        );
    }
}

#[test]
fn discretised_observes_failure_before_zero_delay_repair_under_refinement() {
    for observer_first in [false, true] {
        for level in [2, 4, 8] {
            let mut settings = DiscretisedSettings::new("lost", 1.0);
            settings.level = level;
            let result = explore_discretised(&repaired_failure(observer_first), &settings).unwrap();
            // One exponential's whole mass is integrated exactly at every level.
            assert_probability(&result, 1.0 - (-1.0_f64).exp());
            assert!(result.error_estimate().unwrap() < 1e-12);
        }
    }
}

/// A and B are due in wave zero. A arms its repair and two observers in
/// wave one. The transient observer must wait for B, which cancels it;
/// the persistent observer must then beat A's repair, despite its index.
/// Repair arms recovery in a later wave. A, B and the observer branch,
/// exercising both root forks and sibling snapshot restoration.
fn branching_waves() -> CompiledModel {
    let a = json!({"name": "A", "automata": [{"name": "a",
        "states": ["up", "failed", "missed", "repaired"], "init": "up", "transitions": [
            {"name": "fail", "source": "up", "targets": ["failed", "missed"],
             "distrib": "inst", "probs": [0.25]},
            {"name": "repair", "source": "failed", "targets": ["repaired"],
             "distrib": "delay", "time": 0.0}]}]});
    let b = json!({"name": "B", "automata": [{"name": "a",
        "states": ["up", "failed", "missed"], "init": "up", "transitions": [
            {"name": "fail", "source": "up", "targets": ["failed", "missed"],
             "distrib": "inst", "probs": [0.4]}]}]});
    let transient = observer(
        "transient",
        json!({"op": "bool", "bool_op": "and",
        "args": [active("A", "failed"), active("B", "up")]}),
    );
    let mut persistent = observer("persistent", active("A", "failed"));
    persistent["automata"][0]["states"] = json!(["off", "on", "missed"]);
    let transition = &mut persistent["automata"][0]["transitions"][0];
    transition["targets"] = json!(["on", "missed"]);
    transition["distrib"] = json!("inst");
    transition.as_object_mut().unwrap().remove("time");
    transition["probs"] = json!([0.3]);
    let recovery = observer(
        "recovery",
        json!({"op": "bool", "bool_op": "and",
        "args": [active("A", "repaired"), active("persistent", "on")]}),
    );
    compile(
        vec![a, transient, persistent, recovery, b],
        json!([
            target("lost", "recovery"),
            target("wrong_wave", "transient")
        ]),
    )
}

fn assert_waves(result: &ExplorationResult) {
    assert_probability(result, 0.25 * 0.3);
    assert_eq!(result.sequences.len(), 2);
    for (k, probability) in [0.25 * 0.6 * 0.3, 0.25 * 0.4 * 0.3].into_iter().enumerate() {
        assert!((result.sequences[k].probability - probability).abs() < 1e-12);
        let path: Vec<_> = result
            .sequence_steps(k)
            .unwrap()
            .map(|s| s.transition.as_str())
            .collect();
        assert_eq!(
            path,
            [
                "A.a.fail",
                "B.a.fail",
                "persistent.a.detect",
                "A.a.repair",
                "recovery.a.detect"
            ]
        );
    }
}

#[test]
fn exact_preserves_old_waves_and_observer_priority_across_branch_forks() {
    for threads in [1, 2] {
        let mut settings = ExactSettings::new("lost", 1.0);
        settings.threads = Some(threads);
        assert_waves(&explore_exact(&branching_waves(), &settings).unwrap());
    }
}

#[test]
fn discretised_preserves_old_waves_and_observer_priority_across_branch_forks() {
    for threads in [1, 2] {
        let mut settings = DiscretisedSettings::new("lost", 1.0);
        settings.threads = Some(threads);
        settings.level = 2;
        assert_waves(&explore_discretised(&branching_waves(), &settings).unwrap());
    }
}
