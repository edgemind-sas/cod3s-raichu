//! Automatic ordering queries preserve clocks, explicit choices and snapshot waves.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, StochasticDates};
use raichu_model::Model;
use serde_json::json;

fn waves() -> CompiledModel {
    let model: Model = serde_json::from_value(json!({
        "name": "automatic_order", "components": [
            {"name": "A", "automata": [{"name": "a", "states": ["up", "down", "repaired"],
                "init": "up", "transitions": [
                    {"name": "fail", "source": "up", "targets": ["down"], "distrib": "delay", "time": 2.0},
                    {"name": "repair", "source": "down", "targets": ["repaired"], "distrib": "delay", "time": 0.0}]}]},
            {"name": "observer", "automata": [{"name": "a", "states": ["off", "on", "missed"],
                "init": "off", "transitions": [
                    {"name": "detect", "source": "off", "targets": ["on", "missed"], "distrib": "inst", "probs": [0.3],
                     "kind": "observation", "guard": {"op": "state_active",
                        "state": {"component": "A", "automaton": "a", "state": "down"}}}]}]},
            {"name": "B", "automata": [{"name": "a", "states": ["up", "down"], "init": "up", "transitions": [
                {"name": "fail", "source": "up", "targets": ["down"], "distrib": "delay", "time": 2.0}]}]}
        ]
    })).unwrap();
    CompiledModel::compile(&model).unwrap()
}

fn next_name(engine: &mut Engine<'_>, model: &CompiledModel) -> Option<String> {
    engine
        .next_immediate_transition()
        .unwrap()
        .map(|idx| model.transitions[idx].name.clone())
}

#[test]
fn restored_forks_keep_old_waves_before_new_observers_and_forced_branches() {
    let model = waves();
    let config = EngineConfig {
        t_max: 3.0,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new(&model, config.clone()).unwrap();
    assert_eq!(
        next_name(&mut engine, &model),
        None,
        "the query does not advance to a future date"
    );
    assert_eq!(engine.step().unwrap().unwrap().transition, "A.a.fail");
    let old_wave = engine.try_snapshot().unwrap();
    assert_eq!(next_name(&mut engine, &model).as_deref(), Some("B.a.fail"));
    assert_eq!(
        engine.fireable()[0].transition,
        "A.a.repair",
        "display order remains date/index"
    );

    // An explicit human choice can skip B and the observer; it is not
    // silently replaced by the automatic choice.
    assert_eq!(
        engine.fire_named("A.a.repair").unwrap().transition,
        "A.a.repair"
    );
    assert_eq!(engine.state("observer.a").unwrap(), "off");
    engine.try_restore(&old_wave).unwrap();
    assert_eq!(engine.step().unwrap().unwrap().transition, "B.a.fail");
    let observer_wave = engine.snapshot().unwrap();

    for (branch, state) in [(0, "on"), (1, "missed")] {
        let mut fork = Engine::new(&model, config.clone()).unwrap();
        fork.restore(&observer_wave).unwrap();
        let idx = fork.next_immediate_transition().unwrap().unwrap();
        assert_eq!(model.transitions[idx].name, "observer.a.detect");
        fork.fire_now(idx, Some(branch)).unwrap();
        assert_eq!(fork.state("observer.a").unwrap(), state);
        assert_eq!(next_name(&mut fork, &model).as_deref(), Some("A.a.repair"));
        assert_eq!(fork.step().unwrap().unwrap().transition, "A.a.repair");
        assert_eq!(next_name(&mut fork, &model), None);
    }
}

#[test]
fn repeated_order_queries_do_not_draw_or_resample_stochastic_clocks() {
    let model = waves();
    let config = EngineConfig {
        t_max: 3.0,
        seed: 7,
        ..EngineConfig::default()
    };
    let mut queried = Engine::new(&model, config.clone()).unwrap();
    let mut control = Engine::new(&model, config).unwrap();
    while let Some(expected) = control.step().unwrap() {
        let before = queried.fireable();
        for _ in 0..5 {
            queried.next_immediate_transition().unwrap();
        }
        assert_eq!(queried.fireable(), before);
        assert_eq!(queried.step().unwrap().unwrap(), expected);
    }

    let stochastic: Model = serde_json::from_value(json!({
        "name": "stochastic", "components": [{"name": "c", "automata": [{
            "name": "a", "states": ["up", "down"], "init": "up", "transitions": [{
                "name": "fail", "source": "up", "targets": ["down"], "distrib": "exp", "rate": 1.0
            }]}]}]
    }))
    .unwrap();
    let stochastic = CompiledModel::compile(&stochastic).unwrap();
    for policy in [StochasticDates::Drawn, StochasticDates::Deferred] {
        let mut engine = Engine::new(
            &stochastic,
            EngineConfig {
                stochastic_dates: policy,
                ..EngineConfig::default()
            },
        )
        .unwrap();
        let before = engine.fireable();
        for _ in 0..5 {
            assert_eq!(engine.next_immediate_transition().unwrap(), None);
        }
        assert_eq!(engine.fireable(), before);
    }
}

#[test]
fn an_immediate_watched_boundary_keeps_its_automatic_precedence() {
    let model: Model = serde_json::from_value(json!({
        "name": "watched_order", "components": [{"name": "c", "automata": [
            {"name": "scheduled", "states": ["off", "on"], "init": "off", "transitions": [
                {"name": "go", "source": "off", "targets": ["on"], "distrib": "delay", "time": 0.0}]},
            {"name": "boundary", "states": ["off", "on"], "init": "off", "transitions": [
                {"name": "go", "source": "off", "targets": ["on"], "distrib": "watched",
                 "guard": {"op": "cmp", "cmp": "ge", "lhs": {"op": "time"},
                    "rhs": {"op": "const", "value": {"kind": "float", "value": 0.0}}}}]}
        ]}]
    })).unwrap();
    let model = CompiledModel::compile(&model).unwrap();
    let mut engine = Engine::new(&model, EngineConfig::default()).unwrap();
    assert_eq!(engine.fireable()[0].transition, "c.scheduled.go");
    assert_eq!(
        next_name(&mut engine, &model).as_deref(),
        Some("c.boundary.go")
    );
    assert_eq!(engine.step().unwrap().unwrap().transition, "c.boundary.go");
    assert_eq!(
        next_name(&mut engine, &model).as_deref(),
        Some("c.scheduled.go")
    );
}
