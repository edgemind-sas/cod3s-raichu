//! An observer fires its delay after the date its condition starts to hold
//! (at that date with no delay), including when that condition reads a
//! continuously-evolving quantity.
//!
//! A feared event is an observer: a `delay` transition of `kind`
//! `observation` whose guard is the event's condition, its tempo being the
//! delay. Its guard used to be re-evaluated only when some discrete
//! transition fired, so a room filling up between two discrete events never
//! raised the event on its threshold: the event came late (at the next
//! discrete event, whatever it was) or never, while a threshold indicator on
//! the same predicate, in the same run, flipped at the crossing
//! (edgemind-sas/cod3s-raichu#162). The engine now locates the crossings of
//! such an observer's condition, with or without a delay: the countdown is
//! armed at a located rise and interrupted at a located fall.
//!
//! Every expected date is a closed form of the ramp or of the triangle.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{
    fault_tree_for_targets, CompiledModel, Engine, EngineConfig, EngineError, FaultTreeError,
    FaultTreeSettings, FireableKind, StochasticDates,
};
use raichu_model::Model;
use serde_json::{json, Value as Json};

fn float(value: f64) -> Json {
    json!({"kind": "float", "value": value})
}

fn x() -> Json {
    json!({"op": "attr", "attr": {"component": "c", "attribute": "x"}})
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

/// The feared event as the muscadet plugin builds an `ObjEvent`: `occ` when
/// `cond` holds, back to `not_occ` when it stops holding, both observers.
fn event(cond: Json, delay: f64) -> Json {
    json!({
        "name": "ev_c",
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

/// A ramp `x` observed by `x cmp threshold`, with no other discrete event.
fn ramp_with_event(cmp: &str, threshold: f64, turn: Option<f64>, delay: f64) -> Model {
    let cond = json!({"op": "cmp", "cmp": cmp, "lhs": x(),
                      "rhs": {"op": "const", "value": float(threshold)}});
    serde_json::from_value(json!({
        "name": "ramp",
        "components": [
            {"name": "c",
             "attributes": [{"name": "x", "kind": "float", "init": float(0.0)}],
             "equations": [{"target": "x", "kind": "ode", "expr": ramp_rate(turn)}]},
            event(cond, delay)
        ],
        "targets": [{"name": "over", "component": "ev_c", "automaton": "ev", "state": "occ"}]
    }))
    .expect("model")
}

/// The `(date, transition)` of every event of a run to `t_max`.
fn events(model: &Model, t_max: f64) -> Vec<(f64, String)> {
    let compiled = CompiledModel::compile(model).unwrap();
    let result = Engine::new(
        &compiled,
        EngineConfig {
            t_max,
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
fn an_event_on_a_rising_quantity_fires_at_the_crossing() {
    // x = t crosses 1.5 at t = 1.5, and nothing discrete happens before
    // the horizon: the event used to never fire.
    let model = ramp_with_event("ge", 1.5, None, 0.0);
    assert_events(&events(&model, 3.0), &[(1.5, "ev_c.ev.occ")]);
}

#[test]
fn an_event_entered_and_left_between_discrete_events_is_seen_both_ways() {
    // x rises to 1 at t = 1 and falls back: above 0.8 on [0.8, 1.2].
    let model = ramp_with_event("gt", 0.8, Some(1.0), 0.0);
    assert_events(
        &events(&model, 2.0),
        &[(0.8, "ev_c.ev.occ"), (1.2, "ev_c.ev.not_occ")],
    );
}

#[test]
fn an_event_holding_from_the_start_fires_at_the_start() {
    // x <= 0.5 holds at t = 0 and stops holding at 0.5.
    let model = ramp_with_event("le", 0.5, None, 0.0);
    assert_events(
        &events(&model, 1.0),
        &[(0.0, "ev_c.ev.occ"), (0.5, "ev_c.ev.not_occ")],
    );
}

#[test]
fn an_event_on_a_discrete_threshold_is_not_located() {
    // `n` is an integer a delay transition writes at t = 2: the observer's
    // guard moves only at discrete events, so nothing is located and the
    // observer stays an ordinary zero-delay transition.
    let model: Model = serde_json::from_value(json!({
        "name": "count",
        "components": [
            {"name": "c",
             "attributes": [{"name": "n", "kind": "int", "init": {"kind": "int", "value": 0}}],
             "automata": [{"name": "a", "states": ["s0", "s1"], "init": "s0",
                 "transitions": [{"name": "go", "source": "s0", "targets": ["s1"],
                                  "distrib": "delay", "time": 2.0,
                                  "effects": [{"target": {"component": "c", "attribute": "n"},
                                               "value": {"op": "const", "value": {"kind": "int", "value": 3}}}]}]}]},
            event(json!({"op": "cmp", "cmp": "ge",
                         "lhs": {"op": "attr", "attr": {"component": "c", "attribute": "n"}},
                         "rhs": {"op": "const", "value": {"kind": "int", "value": 2}}}), 0.0)
        ]
    }))
    .expect("model");
    let compiled = CompiledModel::compile(&model).unwrap();
    assert!(compiled.watched.is_empty(), "{:?}", compiled.watched);
    assert_events(
        &events(&model, 5.0),
        &[(2.0, "c.a.go"), (2.0, "ev_c.ev.occ")],
    );
}

#[test]
fn a_waiting_event_counts_its_tempo_from_the_crossing() {
    // x = t crosses 1.5 at t = 1.5 and nothing discrete happens after it:
    // the tempo of 0.5 runs from the crossing, so the event fires at 2.0.
    // It used to wait for a discrete event to start its countdown, and
    // with none before the horizon it never fired.
    let model = ramp_with_event("ge", 1.5, None, 0.5);
    assert_events(&events(&model, 3.0), &[(2.0, "ev_c.ev.occ")]);
}

#[test]
fn a_waiting_event_whose_condition_falls_back_first_never_fires() {
    // x rises to 1 at t = 1 and falls back: above 0.8 on [0.8, 1.2] only,
    // 0.4 in all. A tempo of 0.5 is never served, and the countdown the
    // fall interrupts is cancelled, as a guard turning false cancels any
    // countdown (the default `reset` interruption policy).
    let model = ramp_with_event("gt", 0.8, Some(1.0), 0.5);
    assert_events(&events(&model, 2.0), &[]);
}

#[test]
fn a_waiting_event_shorter_than_its_condition_fires_then_returns() {
    // The same window of 0.4, a tempo of 0.3: the event fires at
    // 0.8 + 0.3 = 1.1 and returns, with no tempo, at the fall at 1.2.
    let model = ramp_with_event("gt", 0.8, Some(1.0), 0.3);
    assert_events(
        &events(&model, 2.0),
        &[(1.1, "ev_c.ev.occ"), (1.2, "ev_c.ev.not_occ")],
    );
}

#[test]
fn a_located_event_raises_no_new_fault_tree_refusal() {
    // The fault tree of a located observer refuses for the reason its
    // condition gives (a continuous quantity has no state to unroll into),
    // never because the observer would wait before it fires.
    let model = ramp_with_event("ge", 1.5, None, 0.0);
    let compiled = CompiledModel::compile(&model).unwrap();
    let error = fault_tree_for_targets(
        &compiled,
        &["over".to_owned()],
        &FaultTreeSettings::default(),
    )
    .expect_err("a continuous condition has no tree");
    let message = match error {
        FaultTreeError::Unsupported(detail) | FaultTreeError::NonCoherent(detail) => detail,
        other => format!("{other:?}"),
    };
    assert!(!message.contains("waits before it fires"), "{message}");
}

#[test]
fn a_located_event_keeps_its_place_in_the_same_date_waves() {
    // Two failures `a` and `b` are both due at t = 2, armed in the same
    // wave. The event reads `a` down, `b` still up, and the ramp past 1:
    // `a` makes it hold, `b` makes it stop holding, within one instant.
    // The reference fires every transition armed before a wave ahead of
    // what that wave arms, so `b` fires before the event and the event is
    // never raised. Located for the ramp, the event must still wait its
    // turn at a discrete instant rather than fire the moment `a` lands.
    let failure = |name: &str| {
        json!({"name": name, "states": ["up", "down"], "init": "up",
               "transitions": [{"name": "fail", "source": "up", "targets": ["down"],
                                "distrib": "delay", "time": 2.0}]})
    };
    let down = |automaton: &str, state: &str| {
        json!({"op": "state_active",
               "state": {"component": "c", "automaton": automaton, "state": state}})
    };
    let cond = json!({"op": "bool", "bool_op": "and", "args": [
        down("a", "down"),
        down("b", "up"),
        {"op": "cmp", "cmp": "ge", "lhs": x(), "rhs": {"op": "const", "value": float(1.0)}}
    ]});
    let model: Model = serde_json::from_value(json!({
        "name": "waves",
        "components": [
            {"name": "c",
             "attributes": [{"name": "x", "kind": "float", "init": float(0.0)}],
             "equations": [{"target": "x", "kind": "ode", "expr": ramp_rate(None)}],
             "automata": [failure("a"), failure("b")]},
            event(cond, 0.0)
        ]
    }))
    .expect("model");
    let compiled = CompiledModel::compile(&model).unwrap();
    assert_eq!(
        compiled.watched.len(),
        2,
        "both edges of the event are located"
    );
    assert_events(
        &events(&model, 3.0),
        &[(2.0, "c.a.fail"), (2.0, "c.b.fail")],
    );
}

#[test]
fn an_event_on_the_clock_alone_fires_at_its_date_with_nothing_to_integrate() {
    // No ODE, no explicit equation: the only thing that moves is time, and
    // only the event reads it. The engine must still advance continuously
    // to locate `time >= 1.5`, or the event never fires.
    let cond = json!({"op": "cmp", "cmp": "ge", "lhs": {"op": "time"},
                      "rhs": {"op": "const", "value": float(1.5)}});
    let model: Model = serde_json::from_value(json!({
        "name": "clock", "components": [event(cond, 0.0)]
    }))
    .expect("model");
    assert_events(&events(&model, 3.0), &[(1.5, "ev_c.ev.occ")]);
}

/// `x = 1 - t`, a switch turning `on` at t = 0.1, and an event on `on`
/// with `x > 0.5` and a tempo of 0.5: armed at 0.1, due at 0.6, while its
/// condition stops holding at t = 0.5.
fn countdown_cut_short() -> Model {
    let on = json!({"op": "state_active",
                    "state": {"component": "c", "automaton": "sw", "state": "on"}});
    let cond = json!({"op": "bool", "bool_op": "and", "args": [
        on,
        {"op": "cmp", "cmp": "gt", "lhs": x(), "rhs": {"op": "const", "value": float(0.5)}}
    ]});
    serde_json::from_value(json!({
        "name": "interrupted",
        "components": [
            {"name": "c",
             "attributes": [{"name": "x", "kind": "float", "init": float(1.0)}],
             "equations": [{"target": "x", "kind": "ode",
                            "expr": {"op": "const", "value": float(-1.0)}}],
             "automata": [{"name": "sw", "states": ["off", "on"], "init": "off",
                 "transitions": [{"name": "switch", "source": "off", "targets": ["on"],
                                  "distrib": "delay", "time": 0.1}]}]},
            event(cond, 0.5)
        ]
    }))
    .expect("model")
}

#[test]
fn a_countdown_armed_at_a_discrete_instant_is_interrupted_by_a_located_fall() {
    // The event is armed by a discrete event, so its date is the next one
    // on the schedule when the integration towards it starts; the fall at
    // 0.5 interrupts it on the way and it never fires.
    assert_events(
        &events(&countdown_cut_short(), 2.0),
        &[(0.1, "c.sw.switch")],
    );
}

#[test]
fn firing_an_interrupted_countdown_by_hand_is_refused() {
    // Driven interactively: once armed, the event is listed as fireable at
    // 0.6, but asking for it integrates through the fall at 0.5, so it is
    // no longer due when its date comes.
    let compiled = CompiledModel::compile(&countdown_cut_short()).unwrap();
    let mut engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 2.0,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    engine.step().unwrap().expect("the switch at 0.1");
    let armed = engine
        .fireable()
        .into_iter()
        .find(|f| f.transition == "ev_c.ev.occ")
        .expect("the event is armed");
    assert!((armed.date.unwrap() - 0.6).abs() < 1e-9, "{armed:?}");
    let refused = engine.fire_idx(armed.index);
    assert!(
        matches!(refused, Err(EngineError::NotFireable { .. })),
        "{refused:?}"
    );
}

#[test]
fn firing_a_countdown_restarted_on_the_way_by_hand_is_refused() {
    // Armed at 0.1, due at 0.6; its condition stops holding on [0.2, 0.3)
    // and holds again from 0.3, so the reset countdown is re-armed there,
    // due at 0.8. Asking for it at its old date must not fire it early.
    let on = json!({"op": "state_active",
                    "state": {"component": "c", "automaton": "sw", "state": "on"}});
    let clock = |cmp: &str, value: f64| {
        json!({"op": "cmp", "cmp": cmp, "lhs": {"op": "time"},
               "rhs": {"op": "const", "value": float(value)}})
    };
    let cond = json!({"op": "bool", "bool_op": "and", "args": [
        on,
        {"op": "bool", "bool_op": "or", "args": [clock("lt", 0.2), clock("ge", 0.3)]}
    ]});
    let model: Model = serde_json::from_value(json!({
        "name": "restarted",
        "components": [
            {"name": "c",
             "attributes": [{"name": "x", "kind": "float", "init": float(0.0)}],
             "equations": [{"target": "x", "kind": "ode", "expr": ramp_rate(None)}],
             "automata": [{"name": "sw", "states": ["off", "on"], "init": "off",
                 "transitions": [{"name": "switch", "source": "off", "targets": ["on"],
                                  "distrib": "delay", "time": 0.1}]}]},
            event(cond, 0.5)
        ]
    }))
    .expect("model");
    assert_events(
        &events(&model, 0.9),
        &[(0.1, "c.sw.switch"), (0.8, "ev_c.ev.occ")],
    );
    let compiled = CompiledModel::compile(&model).unwrap();
    let mut engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 0.9,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    engine.step().unwrap().expect("the switch at 0.1");
    let armed = engine
        .fireable()
        .into_iter()
        .find(|f| f.transition == "ev_c.ev.occ")
        .expect("the event is armed");
    let refused = engine.fire_idx(armed.index);
    assert!(
        matches!(refused, Err(EngineError::NotFireable { .. })),
        "{refused:?}"
    );
}

/// The event `time >= 0.5` (gated by `gate`, when given) with a tempo of
/// 1.0, due at 1.5, beside a component `b` failing at `b_fail`.
fn countdown_due_at(b_fail: f64, gated: bool) -> Model {
    let clock = json!({"op": "cmp", "cmp": "ge", "lhs": {"op": "time"},
                       "rhs": {"op": "const", "value": float(0.5)}});
    let cond = if gated {
        let up = json!({"op": "state_active",
                        "state": {"component": "b", "automaton": "h", "state": "up"}});
        json!({"op": "bool", "bool_op": "and", "args": [up, clock]})
    } else {
        clock
    };
    serde_json::from_value(json!({
        "name": "tie",
        "components": [
            {"name": "b",
             "automata": [{"name": "h", "states": ["up", "down"], "init": "up",
                 "transitions": [{"name": "fail", "source": "up", "targets": ["down"],
                                  "distrib": "delay", "time": b_fail}]}]},
            event(cond, 1.0)
        ]
    }))
    .expect("model")
}

#[test]
fn a_countdown_due_with_a_failure_keeps_its_observer_rank() {
    // Both are armed before the instant at 1.5 and both are due then: the
    // observer fires first, as it does when neither is located, and only
    // then does the failure end its condition.
    let got = events(&countdown_due_at(1.5, true), 3.0);
    assert_events(
        &got,
        &[
            (1.5, "ev_c.ev.occ"),
            (1.5, "b.h.fail"),
            (1.5, "ev_c.ev.not_occ"),
        ],
    );
}

#[test]
fn a_countdown_due_at_the_horizon_fires_there() {
    // Due at 1.5 with the horizon at 1.5, like a delay transition dated
    // exactly at the horizon.
    let got = events(&countdown_due_at(10.0, false), 1.5);
    assert_events(&got, &[(1.5, "ev_c.ev.occ")]);
}

/// The cut-short countdown of [`countdown_cut_short`] with its condition
/// back from t = 0.7 (`x < 0.3`), under the interruption policy `policy`.
fn countdown_paused(policy: &str) -> Model {
    let on = json!({"op": "state_active",
                    "state": {"component": "c", "automaton": "sw", "state": "on"}});
    let level = |cmp: &str, value: f64| json!({"op": "cmp", "cmp": cmp, "lhs": x(), "rhs": {"op": "const", "value": float(value)}});
    let cond = json!({"op": "bool", "bool_op": "and", "args": [
        on,
        {"op": "bool", "bool_op": "or", "args": [level("gt", 0.5), level("lt", 0.3)]}
    ]});
    let mut event = event(cond, 0.5);
    event["automata"][0]["transitions"][0]["on_interruption"] = json!(policy);
    serde_json::from_value(json!({
        "name": "paused",
        "components": [
            {"name": "c",
             "attributes": [{"name": "x", "kind": "float", "init": float(1.0)}],
             "equations": [{"target": "x", "kind": "ode",
                            "expr": {"op": "const", "value": float(-1.0)}}],
             "automata": [{"name": "sw", "states": ["off", "on"], "init": "off",
                 "transitions": [{"name": "switch", "source": "off", "targets": ["on"],
                                  "distrib": "delay", "time": 0.1}]}]},
            event
        ]
    }))
    .expect("model")
}

#[test]
fn an_interrupted_countdown_follows_its_interruption_policy() {
    // Armed at 0.1, the condition holds on [0.1, 0.5) and again from 0.7.
    // `reset` restarts the 0.5 at 0.7: due 1.2. `resume` keeps the 0.4
    // already served: due 0.8. `continue` never stops: due 0.6, the
    // return edge fires at once, since the condition is false then, and the
    // next occurrence counts its own tempo from 0.7: due 1.2.
    let at = |policy: &str| events(&countdown_paused(policy), 2.0);
    assert_events(&at("reset"), &[(0.1, "c.sw.switch"), (1.2, "ev_c.ev.occ")]);
    assert_events(&at("resume"), &[(0.1, "c.sw.switch"), (0.8, "ev_c.ev.occ")]);
    assert_events(
        &at("continue"),
        &[
            (0.1, "c.sw.switch"),
            (0.6, "ev_c.ev.occ"),
            (0.6, "ev_c.ev.not_occ"),
            (1.2, "ev_c.ev.occ"),
        ],
    );
}

#[test]
fn many_countdowns_armed_at_one_crossing_all_fire() {
    // 1100 events on `time >= 1` with a tempo of 0.5 share one crossing:
    // they are armed together, without tripping the guard against a
    // boundary that fires again and again at one instant.
    let cond = json!({"op": "cmp", "cmp": "ge", "lhs": {"op": "time"},
                      "rhs": {"op": "const", "value": float(1.0)}});
    let components: Vec<Json> = (0..1100)
        .map(|i| {
            let mut event = event(cond.clone(), 0.5);
            event["name"] = json!(format!("ev_{i}"));
            event
        })
        .collect();
    let model: Model =
        serde_json::from_value(json!({"name": "many", "components": components})).expect("model");
    let got = events(&model, 2.0);
    assert_eq!(got.len(), 1100);
    assert!(
        got.iter().all(|(t, _)| (t - 1.5).abs() < 1e-6),
        "{:?}",
        &got[..3]
    );
}

#[test]
fn a_refused_firing_leaves_the_engine_where_it_was() {
    let compiled = CompiledModel::compile(&countdown_cut_short()).unwrap();
    let mut engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 2.0,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    engine.step().unwrap().expect("the switch at 0.1");
    let before = engine.current_time();
    let armed = engine
        .fireable()
        .into_iter()
        .find(|f| f.transition == "ev_c.ev.occ")
        .expect("the event is armed");
    assert!(engine.fire_idx(armed.index).is_err());
    assert_eq!(engine.current_time(), before);
    // And the run then goes on as if nothing had been asked.
    let rest: Vec<_> = std::iter::from_fn(|| engine.step().unwrap()).collect();
    assert!(rest.is_empty(), "{rest:?}");
}

#[test]
fn an_idle_located_observer_is_listed_with_the_kind_of_its_own_law() {
    // Before x reaches 1 the event waits for a crossing, with no date; it
    // is a delay after that crossing, not a watched boundary.
    let compiled = CompiledModel::compile(&ramp_with_event("ge", 1.0, None, 0.5)).unwrap();
    let engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 3.0,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let idle = engine
        .fireable()
        .into_iter()
        .find(|f| f.transition == "ev_c.ev.occ")
        .expect("the event is listed");
    assert_eq!((idle.kind, idle.date), (FireableKind::Delay, None));
}

#[test]
fn a_deferred_firing_past_a_located_countdown_names_the_observer() {
    // x = t reaches 1 at t = 1 and the event falls due 0.5 later: a
    // deferred failure asked for at 2.0 is refused at 1.5, and the cause
    // names the observer as such, not as a watched transition.
    let cond = json!({"op": "cmp", "cmp": "ge", "lhs": x(),
                      "rhs": {"op": "const", "value": float(1.0)}});
    let model: Model = serde_json::from_value(json!({
        "name": "deferred",
        "components": [
            {"name": "c",
             "attributes": [{"name": "x", "kind": "float", "init": float(0.0)}],
             "equations": [{"target": "x", "kind": "ode", "expr": ramp_rate(None)}]},
            {"name": "f",
             "automata": [{"name": "fail", "states": ["ok", "ko"], "init": "ok",
                 "transitions": [{"name": "occ", "source": "ok", "targets": ["ko"],
                                  "distrib": "exp", "rate": 0.1}]}]},
            event(cond, 0.5)
        ]
    }))
    .expect("model");
    let compiled = CompiledModel::compile(&model).unwrap();
    let mut engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 3.0,
            stochastic_dates: StochasticDates::Deferred,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let failure = compiled
        .transitions
        .iter()
        .position(|t| t.name == "f.fail.occ")
        .unwrap();
    let err = engine.fire_deferred_at(failure, 2.0, None).unwrap_err();
    let EngineError::DeferredFiringTooLate { limit, cause, .. } = &err else {
        panic!("{err:?}");
    };
    assert!((limit - 1.5).abs() < 1e-6, "{err:?}");
    assert_eq!(cause, "the located observer `ev_c.ev.occ`");
    assert_eq!(engine.current_time(), 0.0);
}

#[test]
fn a_countdown_nothing_interrupts_is_not_integrated_towards() {
    // A clock event on `time >= 1` with a long tempo and the `continue`
    // policy: once armed at 1, no crossing can change its fate, so the
    // engine jumps to its due date instead of integrating the whole
    // countdown. The solver work is the same whatever the tempo, which
    // it was not while the armed observer still held an event slot.
    let run = |delay: f64| {
        let cond = json!({"op": "cmp", "cmp": "ge", "lhs": {"op": "time"},
                          "rhs": {"op": "const", "value": float(1.0)}});
        let mut event = event(cond, delay);
        event["automata"][0]["transitions"][0]["on_interruption"] = json!("continue");
        let model: Model = serde_json::from_value(json!({
            "name": "clock", "components": [event]
        }))
        .expect("model");
        let compiled = CompiledModel::compile(&model).unwrap();
        let result = Engine::new(
            &compiled,
            EngineConfig {
                t_max: delay + 2.0,
                ..EngineConfig::default()
            },
        )
        .unwrap()
        .run()
        .unwrap();
        let fired: Vec<(f64, String)> = result
            .events
            .iter()
            .map(|e| (e.time, e.transition.clone()))
            .collect();
        assert_events(&fired, &[(1.0 + delay, "ev_c.ev.occ")]);
        result.work.solver_steps_accepted
    };
    assert_eq!(run(10.0), run(10_000.0));
}

#[test]
fn firing_a_chosen_transition_passes_over_a_countdown_armed_on_the_way() {
    // `x' = 1` from 0, a repair dated 10, and an event on `x > 5` with a
    // tempo of 2: its countdown is armed at the crossing (5) and due at 7,
    // on the way to the repair. Asking for the repair fires the repair, at
    // 10, as asking for any non-earliest transition passes over the dated
    // ones before it; the countdown is left overdue, and fires next.
    let cond = json!({"op": "cmp", "cmp": "gt", "lhs": x(),
                      "rhs": {"op": "const", "value": float(5.0)}});
    let model: Model = serde_json::from_value(json!({
        "name": "passed_over",
        "components": [
            {"name": "c",
             "attributes": [{"name": "x", "kind": "float", "init": float(0.0)}],
             "equations": [{"target": "x", "kind": "ode", "expr": ramp_rate(None)}],
             "automata": [{"name": "pump", "states": ["off", "on"], "init": "off",
                 "transitions": [{"name": "repair", "source": "off", "targets": ["on"],
                                  "distrib": "delay", "time": 10.0}]}]},
            event(cond, 2.0)
        ]
    }))
    .expect("model");
    let compiled = CompiledModel::compile(&model).unwrap();
    let mut engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 20.0,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let repair = engine
        .fireable()
        .into_iter()
        .find(|f| f.transition == "c.pump.repair")
        .expect("the repair is armed");
    let fired = engine.fire_idx(repair.index).expect("the repair fires");
    assert_eq!(fired.transition, "c.pump.repair");
    assert!((fired.time - 10.0).abs() < 1e-9, "{fired:?}");
    let next = engine.step().unwrap().expect("the overdue event");
    assert_eq!(next.transition, "ev_c.ev.occ");
    assert!((next.time - 10.0).abs() < 1e-9, "{next:?}");
}

/// A clock event on `time >= 5` with a tempo of 2 that nothing
/// interrupts, beside a failure whose date is drawn on demand: once the
/// countdown is armed at 5, nothing moves continuously, and the
/// countdown is due at 7.
fn clock_countdown_with_failure(
    stochastic_dates: StochasticDates,
) -> (CompiledModel, EngineConfig) {
    let cond = json!({"op": "cmp", "cmp": "ge", "lhs": {"op": "time"},
                      "rhs": {"op": "const", "value": float(5.0)}});
    let mut event = event(cond, 2.0);
    event["automata"][0]["transitions"][0]["on_interruption"] = json!("continue");
    let model: Model = serde_json::from_value(json!({
        "name": "clock_countdown",
        "components": [
            {"name": "f",
             "automata": [{"name": "fail", "states": ["ok", "ko"], "init": "ok",
                 "transitions": [{"name": "occ", "source": "ok", "targets": ["ko"],
                                  "distrib": "exp", "rate": 0.1}]}]},
            event
        ]
    }))
    .expect("model");
    let config = EngineConfig {
        t_max: 20.0,
        stochastic_dates,
        ..EngineConfig::default()
    };
    (CompiledModel::compile(&model).unwrap(), config)
}

#[test]
fn a_deferred_firing_past_a_clock_countdown_nothing_interrupts_is_refused() {
    // The crossing at 5 leaves nothing to integrate; the advance towards
    // 10 must still report the countdown due at 7 rather than read as
    // having reached 10, or the failure fires at 10 past an event that
    // was due first.
    let (compiled, config) = clock_countdown_with_failure(StochasticDates::Deferred);
    let mut engine = Engine::new(&compiled, config).unwrap();
    let failure = compiled
        .transitions
        .iter()
        .position(|t| t.name == "f.fail.occ")
        .unwrap();
    let err = engine.fire_deferred_at(failure, 10.0, None).unwrap_err();
    let EngineError::DeferredFiringTooLate { limit, cause, .. } = &err else {
        panic!("{err:?}");
    };
    assert!((limit - 7.0).abs() < 1e-9, "{err:?}");
    assert_eq!(cause, "the located observer `ev_c.ev.occ`");
    assert_eq!(engine.current_time(), 0.0);
    assert!(engine.history().is_empty(), "{:?}", engine.history());
}

#[test]
fn a_deferred_firing_before_a_clock_countdown_falls_due_fires() {
    // Asked for at 6, between the crossing and the due date: the failure
    // fires at 6 and the countdown still falls due at 7.
    let (compiled, config) = clock_countdown_with_failure(StochasticDates::Deferred);
    let mut engine = Engine::new(&compiled, config).unwrap();
    let failure = compiled
        .transitions
        .iter()
        .position(|t| t.name == "f.fail.occ")
        .unwrap();
    let fired = engine.fire_deferred_at(failure, 6.0, None).unwrap();
    assert_eq!(fired.transition, "f.fail.occ");
    assert!((fired.time - 6.0).abs() < 1e-9, "{fired:?}");
    let next = engine.step().unwrap().expect("the countdown");
    assert_eq!(next.transition, "ev_c.ev.occ");
    assert!((next.time - 7.0).abs() < 1e-9, "{next:?}");
}

#[test]
fn advancing_past_a_clock_countdown_fires_it_at_its_date() {
    // advance_to(10) must fire the countdown at 7 on the way, not leave
    // the clock at 10 with the event still pending.
    let (compiled, config) = clock_countdown_with_failure(StochasticDates::Deferred);
    let mut engine = Engine::new(&compiled, config).unwrap();
    engine.advance_to(10.0).unwrap();
    assert_eq!(engine.current_time(), 10.0);
    let fired: Vec<(f64, String)> = engine
        .history()
        .iter()
        .map(|e| (e.time, e.transition.clone()))
        .collect();
    assert_events(&fired, &[(7.0, "ev_c.ev.occ")]);
}

#[test]
fn an_idle_located_observer_is_listed_without_a_date_and_refused_by_fire() {
    // Listed (it is armed in its source state) with no date: nothing can
    // fire it before its condition starts to hold, and asking leaves the
    // engine where it was.
    let compiled = CompiledModel::compile(&ramp_with_event("ge", 1.0, None, 0.5)).unwrap();
    let mut engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 3.0,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let idle = engine
        .fireable()
        .into_iter()
        .find(|f| f.transition == "ev_c.ev.occ")
        .expect("the event is listed");
    assert_eq!(idle.date, None);
    let err = engine.fire_idx(idle.index).unwrap_err();
    assert!(matches!(err, EngineError::NotFireable { .. }), "{err:?}");
    assert_eq!(engine.current_time(), 0.0);
    assert!(engine.history().is_empty());
}

#[test]
fn firing_a_chosen_transition_passes_over_every_countdown_armed_on_the_way() {
    // `x' = 1` from 0, a repair dated 10, and two events with a tempo of
    // 1: one on `x > 2` (due at 3), one on `x > 4` (due at 5). Asking for
    // the repair passes over both in one advance; both are left overdue
    // and fire next, in the order of their dates.
    let on = |threshold: f64| {
        json!({"op": "cmp", "cmp": "gt", "lhs": x(),
               "rhs": {"op": "const", "value": float(threshold)}})
    };
    let mut early = event(on(2.0), 1.0);
    early["name"] = json!("ev_a");
    let mut late = event(on(4.0), 1.0);
    late["name"] = json!("ev_b");
    let model: Model = serde_json::from_value(json!({
        "name": "passed_over_twice",
        "components": [
            {"name": "c",
             "attributes": [{"name": "x", "kind": "float", "init": float(0.0)}],
             "equations": [{"target": "x", "kind": "ode", "expr": ramp_rate(None)}],
             "automata": [{"name": "pump", "states": ["off", "on"], "init": "off",
                 "transitions": [{"name": "repair", "source": "off", "targets": ["on"],
                                  "distrib": "delay", "time": 10.0}]}]},
            late,
            early
        ]
    }))
    .expect("model");
    let compiled = CompiledModel::compile(&model).unwrap();
    let mut engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 20.0,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let repair = compiled
        .transitions
        .iter()
        .position(|t| t.name == "c.pump.repair")
        .unwrap();
    let fired = engine.fire_idx(repair).expect("the repair fires");
    assert!((fired.time - 10.0).abs() < 1e-9, "{fired:?}");
    let next: Vec<String> = (0..2)
        .map(|_| engine.step().unwrap().expect("an overdue event").transition)
        .collect();
    assert_eq!(next, ["ev_a.ev.occ", "ev_b.ev.occ"]);
}
