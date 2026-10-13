//! An observer with no wait fires at the date its condition starts to hold,
//! including when that condition reads a continuously-evolving quantity.
//!
//! A feared event is an observer: a `delay` 0 transition of `kind`
//! `observation` whose guard is the event's condition. Its guard used to be
//! re-evaluated only when some discrete transition fired, so a room filling
//! up between two discrete events never raised the event on its threshold:
//! the event came late (at the next discrete event, whatever it was) or
//! never, while a threshold indicator on the same predicate, in the same
//! run, flipped at the crossing (edgemind-sas/cod3s-raichu#162). The
//! compiler now locates such an observer like a watched transition.
//!
//! Every expected date is a closed form of the ramp or of the triangle.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{
    fault_tree_for_targets, CompiledModel, Engine, EngineConfig, FaultTreeError, FaultTreeSettings,
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
fn a_waiting_event_is_not_located() {
    // A tempo starts its countdown when the condition holds: locating the
    // condition is not enough to honour it, so a waiting observer is left
    // as it was declared.
    let model = ramp_with_event("ge", 1.5, None, 0.5);
    let compiled = CompiledModel::compile(&model).unwrap();
    assert_eq!(
        compiled.watched.len(),
        1,
        "only the return edge, which has no tempo"
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
