//! Common-cause groups declared in the study document.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::Value as V;
use raichu_library::{expand, parse_library, parse_study, parse_system, Library};
use raichu_model::Model;
use serde_json::{json, Value};

fn library() -> Library {
    parse_library(
        &json!({"raichu_library": {"format": 1}, "library": {
            "name": "pumps", "version": "1.0.0", "imports": ["std"],
            "types": [
                {"name": "Pump",
                 "uses": [{"fragment": "std.bool_source_out", "with": {"name": "water"}}],
                 "modes": [
                    {"name": "fail", "nature": "failure", "effects": [{"attribute": "water_fed_available_out", "value": false}]},
                    {"name": "stuck", "nature": "failure", "effects": [{"attribute": "water_fed_available_out", "value": false}],
                     "condition": {"op": "const", "value": {"kind": "bool", "value": true}}}
                 ]},
                {"name": "Valve", "body": {"attributes": [{"name": "open", "kind": "bool", "init": {"kind": "bool", "value": true}}]}}
            ]
        }})
        .to_string(),
    )
    .unwrap()
}

fn system(members: usize) -> raichu_library::SystemModel {
    let instances: Vec<Value> = (1..=members)
        .map(|i| json!({"name": format!("E{i}"), "type": "Pump"}))
        .chain(std::iter::once(json!({"name": "V", "type": "Valve"})))
        .collect();
    let indicators: Vec<Value> = (1..=members.min(3))
        .map(|i| json!({"name": format!("e{i}"), "target": "attribute", "attr": {"component": format!("E{i}"), "attribute": "water_fed_out"}}))
        .collect();
    parse_system(
        &json!({"raichu_system": {"format": 1}, "system": {
            "name": "plant", "library": {"name": "pumps", "version": "1.0.0"},
            "instances": instances, "indicators": indicators
        }})
        .to_string(),
    )
    .unwrap()
}

fn delay(t: f64) -> Value {
    json!({"distrib": "delay", "time": t})
}

fn study(body: Value) -> raichu_library::StudyDocument {
    let mut body = body;
    body["name"] = json!("st");
    body["system"] = json!("plant");
    parse_study(&json!({"raichu_study": {"format": 1}, "study": body}).to_string()).unwrap()
}

fn group(orders: Value) -> Value {
    json!({"name": "EX", "mode": "fail", "instances": ["E1", "E2", "E3"], "orders": orders})
}

fn expanded(members: usize, st: &raichu_library::StudyDocument) -> Model {
    expand(&[library()], &system(members), Some(st))
        .unwrap()
        .model
}

fn run(model: &Model, t_max: f64) -> raichu_core::SimulationResult {
    let compiled = CompiledModel::compile(model).unwrap();
    let config = EngineConfig {
        t_max,
        sequences: true,
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

fn events(result: &raichu_core::SimulationResult) -> Vec<(String, f64)> {
    result
        .sequence
        .as_ref()
        .unwrap()
        .events
        .iter()
        .map(|e| (format!("{}.{}.{}", e.obj, e.automaton, e.attr), e.time))
        .collect()
}

fn automata(model: &Model, component: &str) -> Vec<String> {
    model
        .components
        .iter()
        .find(|c| c.name == component)
        .unwrap()
        .automata
        .iter()
        .map(|a| a.name.clone())
        .collect()
}

#[test]
fn a_group_has_one_automaton_per_combination_named_by_member_indices() {
    let st = study(json!({"common_cause_groups": [group(json!([
        {"order": 1, "occurrence": delay(100.0)},
        {"order": 2, "occurrence": delay(1.0), "repair": delay(2.0)},
        {"order": 3, "occurrence": delay(100.0)}
    ]))]}));
    let model = expanded(3, &st);
    assert_eq!(
        automata(&model, "EX"),
        ["cc_1", "cc_2", "cc_3", "cc_1_2", "cc_1_3", "cc_2_3", "cc_1_2_3"]
    );
    // cc_1_2, cc_1_3 and cc_2_3 all occur at 1: every member is cut.
    let result = run(&model, 2.5);
    for e in ["e1", "e2", "e3"] {
        assert_eq!(
            series(&result, e),
            vec![(0.0, V::Bool(true)), (1.0, V::Bool(false))],
            "{e}"
        );
    }
    let fired: Vec<String> = events(&result).into_iter().map(|(n, _)| n).collect();
    assert!(fired.contains(&"EX.cc_1_2.occ".to_owned()), "{fired:?}");
}

#[test]
fn an_inactive_order_builds_no_combination() {
    let st = study(json!({"common_cause_groups": [group(json!([
        {"order": 2, "occurrence": delay(1.0)}
    ]))]}));
    assert_eq!(
        automata(&expanded(3, &st), "EX"),
        ["cc_1_2", "cc_1_3", "cc_2_3"]
    );
}

#[test]
fn a_member_is_held_by_the_combinations_that_cover_it_only() {
    let st = study(json!({"common_cause_groups": [{
        "name": "EX", "mode": "fail", "instances": ["E1", "E2"],
        "orders": [{"order": 2, "occurrence": delay(1.0), "repair": delay(1.0)}]}]}));
    let result = run(&expanded(3, &st), 1.5);
    assert_eq!(series(&result, "e2")[1], (1.0, V::Bool(false)));
    assert_eq!(series(&result, "e3"), vec![(0.0, V::Bool(true))]);
}

#[test]
fn mode_active_on_a_member_reads_its_covering_combinations() {
    let st = study(json!({
        "common_cause_groups": [{"name": "EX", "mode": "fail", "instances": ["E1", "E2"],
            "orders": [{"order": 2, "occurrence": delay(1.0)}]}],
        "feared_events": [{"name": "lost", "condition": {"op": "mode_active", "mode": "E1.fail"}}]
    }));
    let result = run(&expanded(3, &st), 3.0);
    assert!(events(&result).contains(&("lost.ev.occ".to_owned(), 1.0)));
}

#[test]
fn two_groups_sharing_a_member_both_hold_it() {
    let st = study(json!({"common_cause_groups": [
        {"name": "GA", "mode": "fail", "instances": ["E1", "E2"],
         "orders": [{"order": 2, "occurrence": delay(1.0), "repair": delay(1.0)}]},
        {"name": "GB", "mode": "fail", "instances": ["E1", "E3"],
         "orders": [{"order": 2, "occurrence": delay(1.5), "repair": delay(1.0)}]}
    ]}));
    let result = run(&expanded(3, &st), 2.2);
    assert_eq!(
        series(&result, "e1"),
        vec![(0.0, V::Bool(true)), (1.0, V::Bool(false))]
    );
}

#[test]
fn an_on_demand_order_parks_its_lost_draws() {
    let st = study(json!({"common_cause_groups": [group(json!([
        {"order": 2, "occurrence": {"distrib": "inst", "probs": [0.3]}, "repair": delay(1.0)}
    ]))]}));
    let model = expanded(3, &st);
    let ex = model.components.iter().find(|c| c.name == "EX").unwrap();
    assert_eq!(ex.automata[0].states, vec!["rep", "occ", "parked"]);
}

#[test]
fn a_locked_group_carries_its_effects_through_member_steps() {
    // Order 1 would fail E1 at 2, but cc_1_2 holds E1 from 1 to 6: with the
    // lock, cc_1 waits for E1's step to return to rest.
    let st = study(json!({"common_cause_groups": [{
    "name": "EX", "mode": "fail", "instances": ["E1", "E2"], "effect_step": "zero_delay",
    "orders": [
        {"order": 1, "occurrence": delay(2.0), "repair": delay(100.0)},
        {"order": 2, "occurrence": delay(1.0), "repair": delay(5.0)}
    ]}]}));
    let model = expanded(3, &st);
    assert!(automata(&model, "E1").contains(&"EX__effect".to_owned()));
    let result = run(&model, 5.5);
    let fired = events(&result);
    // As on the declaration route, the members' steps carry the sequence
    // events (one per member), the combinations none.
    assert_eq!(
        fired,
        vec![
            ("E1.EX__effect.occ".to_owned(), 1.0),
            ("E2.EX__effect.occ".to_owned(), 1.0)
        ],
        "cc_1 is locked while cc_1_2 holds E1"
    );
    let e1 = model.components.iter().find(|c| c.name == "E1").unwrap();
    let step = e1.automata.iter().find(|a| a.name == "EX__effect").unwrap();
    for t in &step.transitions {
        assert!(t.monitored, "{}", t.name);
        assert_eq!(t.cycle_group.as_deref(), Some("EX"), "{}", t.name);
        assert!(t.kind.is_some(), "{}", t.name);
    }
    // An unlocked group lets cc_1 fail at 2 while cc_1_2 holds, and its
    // combinations carry the events.
    let mut st = st;
    st.common_cause_groups[0].effect_step = None;
    let fired = events(&run(&expanded(3, &st), 3.0));
    assert!(
        fired.contains(&("EX.cc_1.occ".to_owned(), 2.0)),
        "{fired:?}"
    );
    assert!(
        fired.iter().all(|(n, _)| !n.contains("__effect")),
        "{fired:?}"
    );
}

#[test]
fn mode_active_reads_every_group_holding_the_member() {
    // GA holds E1 from 1, GB from 5: the feared event reads both.
    let st = study(json!({
        "common_cause_groups": [
            {"name": "GA", "mode": "fail", "instances": ["E1", "E2"],
             "orders": [{"order": 2, "occurrence": delay(1.0)}]},
            {"name": "GB", "mode": "fail", "instances": ["E1", "E3"],
             "orders": [{"order": 2, "occurrence": delay(5.0)}]}
        ],
        "feared_events": [{"name": "lost", "condition": {"op": "mode_active", "mode": "E1.fail"}}]
    }));
    let fired = events(&run(&expanded(3, &st), 6.0));
    assert!(
        fired.contains(&("lost.ev.occ".to_owned(), 1.0)),
        "{fired:?}"
    );
}

#[test]
fn a_nine_member_group_with_every_order_expands_and_compiles() {
    let orders: Vec<Value> = (1..=9)
        .map(|k| json!({"order": k, "occurrence": {"distrib": "exp", "rate": 1e-3}}))
        .collect();
    let members: Vec<String> = (1..=9).map(|i| format!("E{i}")).collect();
    let st = study(json!({"common_cause_groups": [
        {"name": "EX", "mode": "fail", "instances": members, "orders": orders}]}));
    let model = expanded(9, &st);
    assert_eq!(automata(&model, "EX").len(), 511);
    CompiledModel::compile(&model).unwrap();
}

fn refused(body: Value, members: usize) -> String {
    expand(&[library()], &system(members), Some(&study(body)))
        .unwrap_err()
        .to_string()
}

#[test]
fn malformed_groups_are_refused_by_name() {
    let two = |extra: Value| {
        let mut g = json!({"name": "EX", "mode": "fail", "instances": ["E1", "E2"],
                           "orders": [{"order": 2, "occurrence": delay(1.0)}]});
        for (k, v) in extra.as_object().unwrap() {
            g[k] = v.clone();
        }
        json!({"common_cause_groups": [g]})
    };
    assert!(refused(two(json!({"mode": "leak"})), 3).contains("not in the catalogue"));
    assert!(refused(two(json!({"instances": ["E1", "V"]})), 3).contains("not in the catalogue"));
    assert!(refused(two(json!({"instances": ["E1", "E1"]})), 3).contains("twice"));
    assert!(refused(two(json!({"mode": "stuck"})), 3).contains("condition"));
    assert!(refused(two(json!({"orders": []})), 3).contains("no order"));
    assert!(refused(
        two(json!({"orders": [{"order": 3, "occurrence": delay(1.0)}]})),
        3
    )
    .contains("order 3"));
    assert!(refused(
        two(json!({"orders": [{"order": 2, "occurrence": delay(1.0)}, {"order": 2, "occurrence": delay(2.0)}]})),
        3
    )
    .contains("twice"));
    assert!(refused(two(json!({"name": "E3"})), 3).contains("already uses"));
    let mut both = two(json!({}));
    both["modes"] = json!([{"mode": "fail", "instances": ["E1"], "occurrence": delay(1.0)}]);
    assert!(refused(both, 3).contains("individual activation"));
    let mut target = two(json!({}));
    target["targets"] =
        json!([{"name": "t", "component": "E1", "automaton": "fail", "state": "occ"}]);
    assert!(refused(target, 3).contains("E1.fail"));
    let mut indicator = two(json!({}));
    indicator["indicators"] = json!([{"name": "i", "target": "state", "component": "E1",
                                      "automaton": "fail", "state": "occ"}]);
    let message = refused(indicator, 3);
    assert!(
        message.contains("indicator `i`") && message.contains("common-cause group holds"),
        "{message}"
    );
}
