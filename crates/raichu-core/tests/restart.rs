//! Independent continuations of an age-conditioned trajectory checkpoint.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_model::Model;
use serde_json::{json, Value};

fn model(law: Value) -> CompiledModel {
    let mut transition = json!({"name": "fail", "source": "up", "targets": ["down"]});
    transition
        .as_object_mut()
        .unwrap()
        .extend(law.as_object().unwrap().clone());
    let model: Model = serde_json::from_value(json!({
        "name": "restart", "components": [{"name": "unit", "automata": [{
            "name": "life", "states": ["up", "down"], "init": "up", "transitions": [transition]
        }]}]
    }))
    .unwrap();
    CompiledModel::compile(&model).unwrap()
}

#[test]
fn different_streams_redraw_pending_dates() {
    let model = model(json!({"distrib": "exp", "rate": 0.1}));
    let config = EngineConfig {
        t_max: 1000.0,
        ..Default::default()
    };
    let original = Engine::new(&model, config.clone()).unwrap();
    let snapshot = original.snapshot().unwrap();
    let next = |stream| {
        let mut engine =
            Engine::restart_from_snapshot(&model, config.clone(), &snapshot, stream).unwrap();
        engine.step().unwrap().unwrap().time
    };
    assert_ne!(
        next(1),
        next(2),
        "clones must not replay their parent's drawn date"
    );
}

fn config() -> EngineConfig {
    EngineConfig {
        t_max: 1000.0,
        seed: 7201,
        ..Default::default()
    }
}

fn pending(engine: &Engine<'_>, name: &str) -> f64 {
    engine
        .fireable()
        .iter()
        .find(|item| item.transition == name)
        .unwrap()
        .date
        .unwrap()
}

fn checkpoint_at(model: &CompiledModel, config: EngineConfig, date: f64) -> raichu_core::Snapshot {
    // Choose a naturally surviving parent; no forced date changes its law.
    for seed in 0..1000 {
        let mut engine = Engine::new(
            model,
            EngineConfig {
                seed,
                ..config.clone()
            },
        )
        .unwrap();
        engine.advance_to(date).unwrap();
        if engine.state("unit.life") == Some("up") {
            return engine.snapshot().unwrap();
        }
    }
    panic!("no surviving parent found");
}

fn ks(samples: &mut [f64], cdf: impl Fn(f64) -> f64) {
    samples.sort_by(f64::total_cmp);
    let n = samples.len() as f64;
    let mut distance: f64 = 0.0;
    for (i, &sample) in samples.iter().enumerate() {
        let probability = cdf(sample);
        distance = distance.max((probability - i as f64 / n).abs());
        distance = distance.max(((i + 1) as f64 / n - probability).abs());
    }
    // DKW bound, alpha = 1e-6. Fixed seeds make this a reproducible gate.
    let critical = (-(0.000001_f64 / 2.0).ln() / (2.0 * n)).sqrt();
    assert!(
        distance <= critical,
        "KS distance {distance}, bound {critical}"
    );
}

#[test]
fn every_fixed_law_has_the_age_conditioned_residual_distribution() {
    use raichu_numeric::laws::Law;
    let cases = [
        (
            json!({"distrib":"exp","rate":0.1}),
            Law::exponential(0.1).unwrap(),
            3.0,
        ),
        (
            json!({"distrib":"weibull","shape":2.0,"scale":8.0}),
            Law::weibull(2.0, 8.0).unwrap(),
            4.0,
        ),
        (
            json!({"distrib":"lognormal","mu":2.0,"sigma":0.7}),
            Law::lognormal(2.0, 0.7).unwrap(),
            5.0,
        ),
        (
            json!({"distrib":"gamma","shape":3.0,"scale":2.0}),
            Law::gamma(3.0, 2.0).unwrap(),
            4.0,
        ),
        (
            json!({"distrib":"uniform","low":2.0,"high":8.0}),
            Law::uniform(2.0, 8.0).unwrap(),
            4.0,
        ),
        (
            json!({"distrib":"empirical","points":[[0.0,0.0],[3.0,0.4],[9.0,1.0]]}),
            Law::empirical(vec![(0.0, 0.0), (3.0, 0.4), (9.0, 1.0)]).unwrap(),
            2.0,
        ),
    ];
    for (description, law, age) in cases {
        let model = model(description);
        let snapshot = checkpoint_at(&model, config(), age);
        let mut samples: Vec<f64> = (0..10_000)
            .map(|stream| {
                let engine =
                    Engine::restart_from_snapshot(&model, config(), &snapshot, stream).unwrap();
                let residual = pending(&engine, "unit.life.fail") - age;
                assert!(residual >= 0.0);
                residual
            })
            .collect();
        ks(&mut samples, |residual| {
            law.conditional_cdf(age, residual).unwrap()
        });
    }
}

#[test]
fn restarts_replay_exactly_on_the_same_stream_and_restore_remains_exact() {
    let model = model(json!({"distrib":"weibull","shape":2.0,"scale":8.0}));
    let mut parent = Engine::new(&model, config()).unwrap();
    parent.advance_to(1.0).unwrap();
    let snapshot = parent.snapshot().unwrap();
    let old_date = pending(&parent, "unit.life.fail");
    let rebuilt = Engine::from_snapshot(&model, config(), &snapshot).unwrap();
    assert_eq!(
        old_date.to_bits(),
        pending(&rebuilt, "unit.life.fail").to_bits()
    );
    let mut first = Engine::restart_from_snapshot(&model, config(), &snapshot, 24).unwrap();
    let mut second = Engine::restart_from_snapshot(&model, config(), &snapshot, 24).unwrap();
    assert_eq!(first.step().unwrap(), second.step().unwrap());
    let checkpoint = first.snapshot().unwrap();
    first.restore(&checkpoint).unwrap();
    assert_eq!(first.history(), second.history());
}

fn gated(law: Value, policy: &str, start: f64) -> CompiledModel {
    let mut failure = json!({"name":"fail", "source":"up", "targets":["down"],
        "guard":{"op":"bool", "bool_op":"not", "args":[{"op":"state_active", "state":{
            "component":"gate","automaton":"phase","state":"closed"}}]},
        "on_interruption":policy});
    failure
        .as_object_mut()
        .unwrap()
        .extend(law.as_object().unwrap().clone());
    let document = json!({"name":"gated", "components":[
        {"name":"gate", "automata":[{"name":"phase", "states":["open","closed","reopened"], "init":"open", "transitions":[
            {"name":"close","source":"open","targets":["closed"],"distrib":"delay","time":4.0},
            {"name":"reopen","source":"closed","targets":["reopened"],"distrib":"delay","time":3.0}
        ]}]},
        {"name":"unit", "automata":[{"name":"life", "states":["idle","up","down"], "init":"idle", "transitions":[
            {"name":"start","source":"idle","targets":["up"],"distrib":"delay","time":start}, failure
        ]}]}
    ]});
    CompiledModel::compile(&serde_json::from_value(document).unwrap()).unwrap()
}

#[test]
fn paused_weibull_redraws_at_rearm_using_active_age_only() {
    use raichu_numeric::laws::Law;
    let model = gated(
        json!({"distrib":"weibull","shape":2.0,"scale":8.0}),
        "resume",
        1.0,
    );
    let snapshot = checkpoint_at(&model, config(), 5.0);
    let mut residuals = Vec::new();
    for stream in 0..10_000 {
        let mut engine =
            Engine::restart_from_snapshot(&model, config(), &snapshot, stream).unwrap();
        assert!(engine
            .fireable()
            .iter()
            .all(|item| item.transition != "unit.life.fail"));
        engine.advance_to(7.0).unwrap();
        residuals.push(pending(&engine, "unit.life.fail") - 7.0);
    }
    let law = Law::weibull(2.0, 8.0).unwrap();
    ks(&mut residuals, |r| law.conditional_cdf(3.0, r).unwrap());
}

#[test]
fn continue_ages_through_a_false_guard_and_can_fire_there() {
    use raichu_numeric::laws::Law;
    let model = gated(
        json!({"distrib":"weibull","shape":2.0,"scale":8.0}),
        "continue",
        1.0,
    );
    let snapshot = checkpoint_at(&model, config(), 5.0);
    let mut residuals = Vec::new();
    let mut fired_while_closed = 0;
    for stream in 0..10_000 {
        let mut engine =
            Engine::restart_from_snapshot(&model, config(), &snapshot, stream).unwrap();
        residuals.push(pending(&engine, "unit.life.fail") - 5.0);
        if engine.step().unwrap().unwrap().transition == "unit.life.fail" {
            assert_eq!(engine.state("gate.phase"), Some("closed"));
            fired_while_closed += 1;
        }
    }
    assert!(fired_while_closed > 0);
    let law = Law::weibull(2.0, 8.0).unwrap();
    ks(&mut residuals, |r| law.conditional_cdf(4.0, r).unwrap());
}

#[test]
fn reset_rearms_from_age_zero_after_a_restart() {
    use raichu_numeric::laws::Law;
    let model = gated(
        json!({"distrib":"weibull","shape":2.0,"scale":8.0}),
        "reset",
        1.0,
    );
    let snapshot = checkpoint_at(&model, config(), 5.0);
    let mut residuals = Vec::new();
    for stream in 0..10_000 {
        let mut engine =
            Engine::restart_from_snapshot(&model, config(), &snapshot, stream).unwrap();
        engine.advance_to(7.0).unwrap();
        residuals.push(pending(&engine, "unit.life.fail") - 7.0);
    }
    let law = Law::weibull(2.0, 8.0).unwrap();
    ks(&mut residuals, |r| law.cdf(r));
}

#[test]
fn deterministic_pending_and_paused_dates_are_preserved() {
    let model = gated(json!({"distrib":"delay","time":8.0}), "resume", 1.0);
    for time in [2.0, 5.0, 8.0] {
        let snapshot = checkpoint_at(&model, config(), time);
        let mut exact = Engine::from_snapshot(&model, config(), &snapshot).unwrap();
        let restarted = Engine::restart_from_snapshot(&model, config(), &snapshot, 123).unwrap();
        exact.forget_history();
        assert_eq!(exact.run().unwrap().events, restarted.run().unwrap().events);
    }
}

#[test]
fn zero_rate_stays_dormant() {
    let model = model(
        json!({"distrib":"exp","rate_expr":{"op":"const","value":{"kind":"float","value":0.0}}}),
    );
    let snapshot = checkpoint_at(&model, config(), 4.0);
    let engine = Engine::restart_from_snapshot(&model, config(), &snapshot, 2).unwrap();
    assert_eq!(pending(&engine, "unit.life.fail"), f64::INFINITY);
}

fn constant(value: f64) -> Value {
    json!({"op":"const","value":{"kind":"float","value":value}})
}

fn first_failure(mut engine: Engine<'_>) -> f64 {
    while let Some(event) = engine.step().unwrap() {
        if event.transition == "unit.life.fail" {
            return event.time;
        }
    }
    panic!("failure was not reached");
}

#[test]
fn discrete_variable_rate_restarts_above_the_accrued_hazard() {
    let rate = json!({"op":"if","cond":{"op":"state_active","state":{
        "component":"gate","automaton":"phase","state":"closed"}},
        "then":constant(0.3),"otherwise":constant(0.1)});
    let model = gated(json!({"distrib":"exp","rate_expr":rate}), "continue", 1.0);
    let snapshot = checkpoint_at(&model, config(), 5.0);
    let mut samples = Vec::new();
    for stream in 0..10_000 {
        let engine = Engine::restart_from_snapshot(&model, config(), &snapshot, stream).unwrap();
        assert!(pending(&engine, "unit.life.fail") > 5.0);
        samples.push(first_failure(engine) - 5.0);
    }
    ks(&mut samples, |r| {
        -(-0.3 * r.min(2.0) - 0.1 * (r - 2.0).max(0.0)).exp_m1()
    });
}

#[test]
fn continuous_hazard_and_paused_hazard_have_fresh_future_thresholds() {
    let law =
        json!({"distrib":"exp","rate_expr":{"op":"mul","args":[constant(0.02),{"op":"time"}]}});
    for (policy, snapshot_time, restart_time) in [("continue", 3.0, 3.0), ("resume", 5.0, 7.0)] {
        let model = gated(law.clone(), policy, 1.0);
        let snapshot = checkpoint_at(&model, config(), snapshot_time);
        let mut samples = Vec::new();
        for stream in 0..10_000 {
            let mut engine =
                Engine::restart_from_snapshot(&model, config(), &snapshot, stream).unwrap();
            if policy == "resume" {
                engine.advance_to(restart_time).unwrap();
            }
            let date = first_failure(engine);
            assert!(date > restart_time);
            samples.push(date - restart_time);
        }
        ks(&mut samples, |r| {
            -(-0.01 * (r * r + 2.0 * restart_time * r)).exp_m1()
        });
    }
}

#[test]
fn invalid_modes_are_refused_in_configs_and_snapshots() {
    use raichu_core::{EngineError, StochasticDates};
    let model = model(json!({"distrib":"exp","rate":0.1}));
    let snapshot = Engine::new(&model, config()).unwrap().snapshot().unwrap();
    for (bad, parameter) in [
        (
            EngineConfig {
                stochastic_dates: StochasticDates::Deferred,
                ..config()
            },
            "stochastic_dates",
        ),
        (
            EngineConfig {
                rate_factors: vec![1.0],
                ..config()
            },
            "rate_factors",
        ),
    ] {
        assert!(
            matches!(Engine::restart_from_snapshot(&model,bad.clone(),&snapshot,0),
            Err(EngineError::InvalidStudyParameter { parameter: p, .. }) if p == parameter)
        );
        let captured = Engine::new(&model, bad).unwrap().snapshot().unwrap();
        assert!(
            matches!(Engine::restart_from_snapshot(&model,config(),&captured,0),
            Err(EngineError::InvalidStudyParameter { parameter: p, .. }) if p == parameter)
        );
    }
    let fmu: Model = serde_json::from_value(json!({"name":"fmu", "components":[],
        "fmu_units":[{"name":"external", "path":"unknown.fmu", "step":1.0}]}))
    .unwrap();
    let fmu = CompiledModel::compile(&fmu).unwrap();
    assert!(
        matches!(Engine::restart_from_snapshot(&fmu,config(),&snapshot,0),
        Err(EngineError::FmuSnapshotApi { unit, .. }) if unit == "external")
    );
}

#[test]
fn restarting_after_resuming_uses_total_active_age() {
    use raichu_numeric::laws::Law;
    let model = gated(
        json!({"distrib":"weibull","shape":2.0,"scale":8.0}),
        "resume",
        1.0,
    );
    let snapshot = checkpoint_at(&model, config(), 9.0);
    let mut samples = Vec::new();
    for stream in 0..10_000 {
        let engine = Engine::restart_from_snapshot(&model, config(), &snapshot, stream).unwrap();
        samples.push(pending(&engine, "unit.life.fail") - 9.0);
    }
    let law = Law::weibull(2.0, 8.0).unwrap();
    ks(&mut samples, |r| law.conditional_cdf(5.0, r).unwrap());
}

#[test]
fn extreme_log_tail_uses_hazard_inversion_instead_of_underflow() {
    // Keep a tiny law scale relative to elapsed age, but choose the parent
    // through the existing interactive forced-date API to reach that age.
    // Its conditional law is still mathematically defined in the far tail.
    let model = model(json!({"distrib":"gamma","shape":2.0,"scale":1.0}));
    let mut parent = Engine::new(&model, config()).unwrap();
    parent.set_date("unit.life.fail", 950.0).unwrap();
    parent.advance_to(800.0).unwrap();
    let snapshot = parent.snapshot().unwrap();
    let law = raichu_numeric::laws::Law::gamma(2.0, 1.0).unwrap();
    let mut samples = Vec::new();
    for stream in 0..1000 {
        let engine = Engine::restart_from_snapshot(&model, config(), &snapshot, stream).unwrap();
        let residual = pending(&engine, "unit.life.fail") - 800.0;
        assert!(residual.is_finite() && residual > 0.0);
        samples.push(residual);
    }
    ks(&mut samples, |r| law.conditional_cdf(800.0, r).unwrap());
}

#[test]
fn step_until_returns_one_event_and_leaves_later_dates_untouched() {
    let model = gated(json!({"distrib":"delay","time":8.0}), "resume", 1.0);
    let mut engine = Engine::new(&model, config()).unwrap();
    assert_eq!(engine.step_until(3.0).unwrap().unwrap().time, 1.0);
    assert!(engine.step_until(3.0).unwrap().is_none());
    assert_eq!(engine.current_time(), 3.0);
    assert_eq!(pending(&engine, "gate.phase.close"), 4.0);
    assert_eq!(engine.step().unwrap().unwrap().time, 4.0);
    assert_eq!(engine.step().unwrap().unwrap().time, 7.0);
    assert_eq!(engine.step_until(12.0).unwrap().unwrap().time, 12.0);
}

#[test]
fn step_until_integrates_and_locates_watched_guards() {
    let model: Model = serde_json::from_value(json!({"name":"ramp", "components":[{
        "name":"ramp", "attributes":[{"name":"x","kind":"float","init":{"kind":"float","value":0.0}}],
        "equations":[{"target":"x","kind":"ode","expr":constant(1.0)}],
        "automata":[{"name":"phase","states":["before","after"],"init":"before","transitions":[{
            "name":"cross","source":"before","targets":["after"],"distrib":"watched",
            "guard":{"op":"cmp","cmp":"ge","lhs":{"op":"attr","attr":{"component":"ramp","attribute":"x"}},"rhs":constant(2.0)}
        }]}]
    }]})).unwrap();
    let model = CompiledModel::compile(&model).unwrap();
    let mut engine = Engine::new(&model, config()).unwrap();
    assert!(engine.step_until(1.0).unwrap().is_none());
    assert_eq!(engine.current_time(), 1.0);
    assert!(
        matches!(engine.attribute("ramp.x"),Some(raichu_expr::Value::Float(x)) if (x-1.0).abs()<1e-10)
    );
    let event = engine.step_until(3.0).unwrap().unwrap();
    assert!((event.time - 2.0).abs() < 1e-8);
    assert!(engine.step_until(4.0).unwrap().is_none());
    assert_eq!(engine.current_time(), 4.0);
    assert!(matches!(
        engine.step_until(f64::NAN),
        Err(raichu_core::EngineError::AdvanceTimeInvalid { .. })
    ));
    assert!(engine.step_until(5.0).unwrap().is_none());
}

#[test]
fn a_new_source_episode_forgets_the_previous_clock_age() {
    // Build the repair in model data rather than mutating a compiled law.
    let source: Model = serde_json::from_value(json!({"name":"renewal", "components":[{
        "name":"unit", "automata":[{"name":"life","states":["up","down"],"init":"up","transitions":[
            {"name":"fail","source":"up","targets":["down"],"distrib":"weibull","shape":2.0,"scale":8.0},
            {"name":"repair","source":"down","targets":["up"],"distrib":"delay","time":1.0}
        ]}]
    }]})).unwrap();
    let model = CompiledModel::compile(&source).unwrap();
    let mut parent = Engine::new(&model, config()).unwrap();
    parent.step().unwrap();
    let reentry = parent.step().unwrap().unwrap().time;
    parent.advance_to(reentry + 0.5).unwrap();
    assert_eq!(parent.state("unit.life"), Some("up"));
    let snapshot = parent.snapshot().unwrap();
    let law = raichu_numeric::laws::Law::weibull(2.0, 8.0).unwrap();
    let mut samples = Vec::new();
    for stream in 0..10_000 {
        let restarted = Engine::restart_from_snapshot(&model, config(), &snapshot, stream).unwrap();
        samples.push(pending(&restarted, "unit.life.fail") - snapshot.time());
    }
    ks(&mut samples, |r| law.conditional_cdf(0.5, r).unwrap());
}

#[test]
fn an_exponential_armed_after_zero_has_a_fresh_residual() {
    let model = gated(json!({"distrib":"exp","rate":0.1}), "continue", 2.0);
    let snapshot = checkpoint_at(&model, config(), 3.0);
    let mut samples = Vec::new();
    for stream in 0..10_000 {
        let restarted = Engine::restart_from_snapshot(&model, config(), &snapshot, stream).unwrap();
        samples.push(pending(&restarted, "unit.life.fail") - 3.0);
    }
    ks(&mut samples, |r| -(-0.1 * r).exp_m1());
}

#[test]
fn empirical_atoms_preserve_conditional_mass() {
    let model = model(
        json!({"distrib":"empirical","points":[[1.0,0.3],[1.0,0.3],[4.0,0.3],[4.0,0.7],[8.0,1.0]]}),
    );
    let snapshot = checkpoint_at(&model, config(), 2.0);
    let mut hits = 0;
    for stream in 0..10_000 {
        let restarted = Engine::restart_from_snapshot(&model, config(), &snapshot, stream).unwrap();
        let date = pending(&restarted, "unit.life.fail");
        assert!((4.0..=8.0).contains(&date));
        hits += usize::from(date == 4.0);
    }
    let expected = 0.4 / 0.7;
    let frequency = hits as f64 / 10_000.0;
    assert!(
        (frequency - expected).abs() < 0.025,
        "atom frequency {frequency}, expected {expected}"
    );
}

#[test]
fn step_until_restores_the_horizon_after_a_stepping_error() {
    use raichu_core::EngineError;
    let model = model(json!({"distrib":"exp","rate_expr":{"op":"if",
        "cond":{"op":"cmp","cmp":"lt","lhs":{"op":"time"},"rhs":constant(2.0)},
        "then":constant(0.0),"otherwise":constant(-1.0)}}));
    let mut engine = Engine::new(&model, config()).unwrap();
    assert!(matches!(
        engine.step_until(3.0),
        Err(EngineError::TypeError { .. })
    ));
    // Reaching the rate evaluator again proves the temporary horizon 3
    // was restored: a retained horizon would reject the bound 4 first.
    assert!(matches!(
        engine.step_until(4.0),
        Err(EngineError::TypeError { .. })
    ));
    assert!(matches!(
        engine.step_until(1001.0),
        Err(EngineError::AdvanceTimeInvalid { .. })
    ));
}
