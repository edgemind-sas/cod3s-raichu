//! A timed transition whose guard reads a continuously-evolving quantity
//! is armed at the date the guard starts to hold, and interrupted at the
//! date it stops holding, whatever its law and whatever its role.
//!
//! The crossing of such a guard used to be located for one shape only, an
//! observer with a deterministic delay. Any other guarded transition (a
//! failure drawn from an exponential law while a temperature is above its
//! limit, a repair delayed while a level is low, an instantaneous
//! branching) had its guard re-read only when some discrete transition
//! fired: it was armed late, at the next discrete event whatever it was,
//! or never when none followed, with nothing to signal it.
//!
//! Every expected date is a closed form of the ramp or of the triangle.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, FireableKind, StochasticDates};
use raichu_model::Model;
use serde_json::{json, Value as Json};

fn float(value: f64) -> Json {
    json!({"kind": "float", "value": value})
}

fn x() -> Json {
    json!({"op": "attr", "attr": {"component": "c", "attribute": "x"}})
}

fn x_cmp(cmp: &str, threshold: f64) -> Json {
    json!({"op": "cmp", "cmp": cmp, "lhs": x(),
           "rhs": {"op": "const", "value": float(threshold)}})
}

/// `x' = rate`, the rate flipping from +1 to -1 at `turn` when given.
fn ramp_rate(turn: Option<f64>) -> Json {
    match turn {
        None => json!({"op": "const", "value": float(1.0)}),
        Some(turn) => json!({"op": "if",
            "cond": {"op": "cmp", "cmp": "lt", "lhs": {"op": "time"},
                     "rhs": {"op": "const", "value": float(turn)}},
            "then": {"op": "const", "value": float(1.0)},
            "otherwise": {"op": "const", "value": float(-1.0)}}),
    }
}

/// A ramp `x` and a unit `p` whose transition `fail` leaves `ok` under
/// `guard`, with the law and the extra fields `law` carries. No other
/// discrete event exists.
fn guarded(guard: &Json, turn: Option<f64>, law: &Json, targets: &[&str]) -> Model {
    let mut transition = json!({"name": "fail", "source": "ok", "targets": targets,
                                "guard": guard});
    for (key, value) in law.as_object().expect("law fields") {
        transition[key] = value.clone();
    }
    serde_json::from_value(json!({
        "name": "guarded",
        "components": [
            {"name": "c",
             "attributes": [{"name": "x", "kind": "float", "init": float(0.0)}],
             "equations": [{"target": "x", "kind": "ode", "expr": ramp_rate(turn)}]},
            {"name": "p",
             "automata": [{"name": "health", "states": ["ok", "ko", "spared"], "init": "ok",
                 "transitions": [transition]}]}
        ]
    }))
    .expect("model")
}

fn run_events(model: &Model, t_max: f64, seed: u64) -> Vec<(f64, String)> {
    let compiled = CompiledModel::compile(model).unwrap();
    let result = Engine::new(
        &compiled,
        EngineConfig {
            t_max,
            seed,
            ..EngineConfig::default()
        },
    )
    .unwrap()
    .run()
    .unwrap();
    result
        .events
        .iter()
        .map(|e| (e.time, e.transition.clone()))
        .collect()
}

fn assert_events(got: &[(f64, String)], expected: &[(f64, &str)]) {
    assert_eq!(got.len(), expected.len(), "{got:?} against {expected:?}");
    for ((t, name), (te, ne)) in got.iter().zip(expected) {
        assert_eq!(name, ne, "{got:?}");
        assert!(
            (t - te).abs() < 1e-6,
            "{name} at {t}, expected {te}: {got:?}"
        );
    }
}

#[test]
fn a_delay_counts_from_the_date_its_guard_starts_to_hold() {
    // x = t reaches 1.5 at t = 1.5; the delay of 2 ends at 3.5. Nothing
    // discrete happens on the way: the transition used to never fire.
    let model = guarded(
        &x_cmp("ge", 1.5),
        None,
        &json!({"distrib": "delay", "time": 2.0}),
        &["ko"],
    );
    assert_events(&run_events(&model, 10.0, 0), &[(3.5, "p.health.fail")]);
}

#[test]
fn a_role_changes_nothing_to_the_location() {
    for kind in ["failure", "repair"] {
        let model = guarded(
            &x_cmp("ge", 1.5),
            None,
            &json!({"distrib": "delay", "time": 2.0, "kind": kind}),
            &["ko"],
        );
        assert_events(&run_events(&model, 10.0, 0), &[(3.5, "p.health.fail")]);
    }
}

#[test]
fn a_delay_is_interrupted_at_the_date_its_guard_stops_holding() {
    // x rises to 1 at t = 1 and falls back: above 0.8 on [0.8, 1.2]. A
    // delay of 1 armed at 0.8 would end at 1.8, after the guard fell.
    let law = |policy: &str| json!({"distrib": "delay", "time": 1.0, "on_interruption": policy});
    let model = |policy: &str| guarded(&x_cmp("gt", 0.8), Some(1.0), &law(policy), &["ko"]);
    assert_events(&run_events(&model("reset"), 5.0, 0), &[]);
    // `continue` rides through the fall: nothing interrupts the delay.
    assert_events(
        &run_events(&model("continue"), 5.0, 0),
        &[(1.8, "p.health.fail")],
    );
}

#[test]
fn an_instantaneous_branching_fires_at_the_crossing() {
    let model = guarded(
        &x_cmp("ge", 1.5),
        None,
        &json!({"distrib": "inst", "probs": [1.0]}),
        &["ko", "spared"],
    );
    assert_events(&run_events(&model, 10.0, 0), &[(1.5, "p.health.fail")]);
}

#[test]
fn an_exponential_law_is_drawn_from_the_crossing() {
    // The firing date is 1.5 + Exp(1): its mean over independent seeds is
    // 2.5, with a standard error of 1 / sqrt(n). No date precedes 1.5.
    let model = guarded(
        &x_cmp("ge", 1.5),
        None,
        &json!({"distrib": "exp", "rate": 1.0}),
        &["ko"],
    );
    let n = 2_000_u32;
    let mut sum = 0.0;
    for seed in 0..u64::from(n) {
        let events = run_events(&model, 1.0e3, seed);
        assert_eq!(events.len(), 1, "seed {seed}: {events:?}");
        assert!(events[0].0 >= 1.5, "seed {seed}: {events:?}");
        sum += events[0].0;
    }
    let mean = sum / f64::from(n);
    let tolerance = 4.0 / f64::from(n).sqrt();
    assert!((mean - 2.5).abs() < tolerance, "mean firing date {mean}");
}

#[test]
fn a_state_dependent_rate_accrues_from_the_crossing() {
    // A constant rate written as an expression follows the hazard route:
    // a rate large enough to fire at once shows where the hazard starts.
    let model = guarded(
        &x_cmp("ge", 1.5),
        None,
        &json!({"distrib": "exp",
                "rate_expr": {"op": "const", "value": float(1.0e9)}}),
        &["ko"],
    );
    let events = run_events(&model, 10.0, 0);
    assert_eq!(events.len(), 1, "{events:?}");
    assert!((events[0].0 - 1.5).abs() < 1e-6, "{events:?}");
}

#[test]
fn a_guard_moving_only_at_discrete_events_is_not_located() {
    let guard = json!({"op": "state_active",
        "state": {"component": "p", "automaton": "health", "state": "ok"}});
    let model = guarded(
        &guard,
        None,
        &json!({"distrib": "delay", "time": 2.0}),
        &["ko"],
    );
    let compiled = CompiledModel::compile(&model).unwrap();
    assert!(compiled.watched.is_empty(), "{:?}", compiled.watched);
    assert_events(&run_events(&model, 10.0, 0), &[(2.0, "p.health.fail")]);
}

#[test]
fn a_deferred_stochastic_transition_is_armed_at_the_crossing() {
    // With no date drawn, the transition carries no date before or after
    // the crossing; what the crossing changes is that it is armed, so it
    // may be fired by hand once the clock has passed 1.5.
    let model = guarded(
        &x_cmp("ge", 1.5),
        None,
        &json!({"distrib": "exp", "rate": 1.0}),
        &["ko"],
    );
    let compiled = CompiledModel::compile(&model).unwrap();
    let mut engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 10.0,
            stochastic_dates: StochasticDates::Deferred,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let fail = compiled
        .transitions
        .iter()
        .position(|t| t.name == "p.health.fail")
        .unwrap();
    assert!(
        engine.fire_deferred_at(fail, 1.0, None).is_err(),
        "the guard does not hold at 1.0"
    );
    assert_eq!(engine.current_time(), 0.0);
    let listed = engine
        .fireable()
        .into_iter()
        .find(|f| f.transition == "p.health.fail")
        .expect("the transition is listed");
    assert_eq!(listed.kind, FireableKind::Stochastic);
    // Advancing past the crossing arms it, with nothing discrete on the way.
    engine.advance_to(2.0).unwrap();
    assert!((engine.current_time() - 2.0).abs() < 1e-9);
    engine.fire_deferred_at(fail, 4.0, None).unwrap();
    assert!((engine.current_time() - 4.0).abs() < 1e-9);
}
