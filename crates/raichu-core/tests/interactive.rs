//! Interactive-simulation control surface (isimu):
//!
//! - Phase A — `fireable()` + firing a *chosen* armed transition
//!   (`fire_named` / `fire_idx`) rather than only the earliest, then
//!   inspecting the resulting state through `attribute` / `state`.
//! - Phase B — forcing a chosen transition's destination branch
//!   (`fire_named_to` / `fire_idx_to`), bypassing the RNG /
//!   deterministic-branch resolution — the reproducible outcome control
//!   that makes stochastic mechanics testable.
//!
//! The single-trajectory engine stays deterministic; these methods add
//! only *control + observation* over the same cycle.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError, FireableKind};
use raichu_expr::{Assignment, AttrRef, Expr, StateRef, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Distrib, Model, SensitiveFunction, Transition,
};

/// A component with a single two-state failure automaton `fail`
/// (`ok → nok` after `ttf`, `nok → ok` after `ttr`), with a boolean
/// attribute `up` mirrored from the automaton state by a sensitive
/// function so state changes are observable through `attribute`.
fn failing_component(name: &str, ttf: f64, ttr: f64) -> Component {
    Component {
        name: name.into(),
        attributes: vec![Attribute {
            name: "up".into(),
            kind: AttrKind::Bool,
            init: Value::Bool(true),
        }],
        ports: vec![],
        interfaces: vec![],
        automata: vec![Automaton {
            name: "fail".into(),
            states: vec!["ok".into(), "nok".into()],
            init: "ok".into(),
            transitions: vec![
                Transition {
                    name: "occ".into(),
                    source: "ok".into(),
                    guard: None,
                    targets: vec!["nok".into()],
                    on_interruption: Default::default(),
                    distrib: Distrib::Delay { time: ttf },
                },
                Transition {
                    name: "rep".into(),
                    source: "nok".into(),
                    guard: None,
                    targets: vec!["ok".into()],
                    on_interruption: Default::default(),
                    distrib: Distrib::Delay { time: ttr },
                },
            ],
        }],
        // `up` reflects the automaton: true iff `fail` sits in `ok`.
        equations: vec![],
        sensitive_functions: vec![SensitiveFunction {
            name: "reflect".into(),
            effects: vec![raichu_expr::Assignment {
                target: raichu_expr::AttrRef {
                    component: name.into(),
                    attribute: "up".into(),
                },
                value: raichu_expr::Expr::StateActive {
                    state: raichu_expr::StateRef {
                        component: name.into(),
                        automaton: "fail".into(),
                        state: "ok".into(),
                    },
                },
            }],
        }],
    }
}

/// Two independent failing components with distinct time-to-failure
/// (A@5, B@8): at t = 0 both `occ` transitions are armed, A strictly
/// earlier than B.
fn two_component_model() -> Model {
    Model {
        name: "isimu_two".into(),
        components: vec![
            failing_component("A", 5.0, 10.0),
            failing_component("B", 8.0, 10.0),
        ],
        connections: vec![],
        indicators: vec![],
    }
}

fn compile(model: &Model) -> CompiledModel {
    CompiledModel::compile(model).unwrap()
}

#[test]
fn fireable_lists_armed_transitions_earliest_first() {
    let model = two_component_model();
    let compiled = compile(&model);
    let engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    let fireable = engine.fireable();
    // Only the two `occ` transitions are armed at t = 0 (the `rep`
    // transitions sit in a non-source state).
    let names: Vec<&str> = fireable.iter().map(|f| f.transition.as_str()).collect();
    assert_eq!(names, vec!["A.fail.occ", "B.fail.occ"]);
    // Sorted earliest date first.
    assert_eq!(fireable[0].date, Some(5.0));
    assert_eq!(fireable[1].date, Some(8.0));
    // Delay transitions are classified as such.
    assert!(fireable.iter().all(|f| f.kind == FireableKind::Delay));
}

#[test]
fn fire_named_fires_a_non_earliest_transition() {
    let model = two_component_model();
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    // Deliberately fire B (the *later* transition, date 8) before A
    // (date 5) — the interactive override the plain `step()` cannot do.
    let event = engine.fire_named("B.fail.occ").unwrap();
    assert_eq!(event.time, 8.0);
    assert_eq!(event.transition, "B.fail.occ");
    assert_eq!(event.from, "ok");
    assert_eq!(event.to, "nok");

    // Time advanced to B's date; B is down, A is still up (skipped).
    assert_eq!(engine.current_time(), 8.0);
    assert_eq!(engine.state("B.fail"), Some("nok"));
    assert_eq!(engine.attribute("B.up"), Some(Value::Bool(false)));
    assert_eq!(engine.state("A.fail"), Some("ok"));
    assert_eq!(engine.attribute("A.up"), Some(Value::Bool(true)));
}

#[test]
fn fire_idx_matches_fireable_index() {
    let model = two_component_model();
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    // Pick B by its index in the fireable list and fire through the
    // stable transition handle.
    let target = engine
        .fireable()
        .into_iter()
        .find(|f| f.transition == "B.fail.occ")
        .unwrap();
    let event = engine.fire_idx(target.index).unwrap();
    assert_eq!(event.transition, "B.fail.occ");
    assert_eq!(engine.state("B.fail"), Some("nok"));
}

#[test]
fn overdue_skipped_transition_fires_at_current_time_not_in_the_past() {
    // After skipping A (armed @5) by firing B @8, A is overdue; the
    // clock must not run backwards when it finally fires.
    let model = two_component_model();
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    engine.fire_named("B.fail.occ").unwrap(); // t → 8, A still pending @5
    assert_eq!(engine.current_time(), 8.0);

    // A is still armed; firing it does not move time back to 5.
    let a = engine
        .fireable()
        .into_iter()
        .find(|f| f.transition == "A.fail.occ")
        .unwrap();
    assert_eq!(a.date, Some(5.0)); // still recorded at its stale date
    let event = engine.fire_idx(a.index).unwrap();
    assert_eq!(event.time, 8.0); // fires *now*, not in the past
    assert_eq!(engine.current_time(), 8.0);
    assert_eq!(engine.state("A.fail"), Some("nok"));
}

#[test]
fn fire_named_unknown_transition_is_typed_error() {
    let model = two_component_model();
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    let err = engine.fire_named("A.fail.nope").unwrap_err();
    assert!(
        matches!(err, EngineError::UnknownTransition { .. }),
        "{err:?}"
    );
}

#[test]
fn fire_named_unarmed_transition_is_not_fireable() {
    // `A.fail.rep` sits in the `nok` state which is not active at t = 0:
    // it is not armed, so firing it is a typed `NotFireable` error.
    let model = two_component_model();
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    let err = engine.fire_named("A.fail.rep").unwrap_err();
    assert!(matches!(err, EngineError::NotFireable { .. }), "{err:?}");
}

#[test]
fn interactive_firing_reaches_the_same_state_as_stepping() {
    // Firing the earliest transition explicitly is equivalent to a
    // plain `step()` (same event, same resulting state).
    let model = two_component_model();
    let compiled = compile(&model);

    let mut a = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let stepped = a.step().unwrap().unwrap();

    let mut b = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let fired = b.fire_named("A.fail.occ").unwrap();

    assert_eq!(stepped, fired);
    assert_eq!(a.current_time(), b.current_time());
    assert_eq!(a.state("A.fail"), b.state("A.fail"));
}

// ---- Phase B — forced branch (`fire_*_to`) ----------------------------

/// A one-shot demand: an instantaneous branching `resolve` from
/// `pending` to either `ok` or `ko`, with boolean attributes `success`
/// / `failure` mirrored from the outcome state. `ok_prob` is the first
/// branch probability (the `ko` complement is reconstructed at compile).
fn demand_model(ok_prob: f64) -> Model {
    let reflect = |attr: &str, state: &str| SensitiveFunction {
        name: format!("reflect_{attr}"),
        effects: vec![Assignment {
            target: AttrRef {
                component: "d".into(),
                attribute: attr.into(),
            },
            value: Expr::StateActive {
                state: StateRef {
                    component: "d".into(),
                    automaton: "req".into(),
                    state: state.into(),
                },
            },
        }],
    };
    Model {
        name: "isimu_demand".into(),
        components: vec![Component {
            name: "d".into(),
            attributes: vec![
                Attribute {
                    name: "success".into(),
                    kind: AttrKind::Bool,
                    init: Value::Bool(false),
                },
                Attribute {
                    name: "failure".into(),
                    kind: AttrKind::Bool,
                    init: Value::Bool(false),
                },
            ],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "req".into(),
                states: vec!["pending".into(), "ok".into(), "ko".into()],
                init: "pending".into(),
                transitions: vec![Transition {
                    name: "resolve".into(),
                    source: "pending".into(),
                    guard: None,
                    targets: vec!["ok".into(), "ko".into()],
                    on_interruption: Default::default(),
                    distrib: Distrib::Inst {
                        probs: vec![ok_prob],
                    },
                }],
            }],
            equations: vec![],
            sensitive_functions: vec![reflect("success", "ok"), reflect("failure", "ko")],
        }],
        connections: vec![],
        indicators: vec![],
    }
}

#[test]
fn fire_named_to_forces_the_ok_branch() {
    let model = demand_model(0.7); // non-deterministic: 0.7 / 0.3
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    // The instantaneous branching is armed at t = 0.
    assert!(engine
        .fireable()
        .iter()
        .any(|f| f.transition == "d.req.resolve" && f.kind == FireableKind::Inst));

    let event = engine.fire_named_to("d.req.resolve", "ok").unwrap();
    assert_eq!(event.to, "ok");
    assert_eq!(engine.state("d.req"), Some("ok"));
    assert_eq!(engine.attribute("d.success"), Some(Value::Bool(true)));
    assert_eq!(engine.attribute("d.failure"), Some(Value::Bool(false)));
}

#[test]
fn fire_named_to_forces_the_ko_branch() {
    let model = demand_model(0.7);
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    let event = engine.fire_named_to("d.req.resolve", "ko").unwrap();
    assert_eq!(event.to, "ko");
    assert_eq!(engine.state("d.req"), Some("ko"));
    assert_eq!(engine.attribute("d.success"), Some(Value::Bool(false)));
    assert_eq!(engine.attribute("d.failure"), Some(Value::Bool(true)));
}

#[test]
fn natural_fire_of_stochastic_inst_draws_a_branch() {
    // Brique 2: a non-deterministic instantaneous branching now *draws*
    // its destination from the RNG (it no longer errors). Forcing (above)
    // overrides that draw; a plain fire takes it.
    let model = demand_model(0.7);
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    let event = engine.fire_named("d.req.resolve").unwrap();
    assert!(matches!(event.to.as_str(), "ok" | "ko"), "{}", event.to);
    assert_eq!(engine.state("d.req"), Some(event.to.as_str()));
}

#[test]
fn forcing_overrides_the_deterministic_branch() {
    // Even the branch the engine would *not* pick can be forced:
    // probs = [1.0] ⇒ natural outcome is `ok`, but forced `ko` wins.
    let model = demand_model(1.0);
    let compiled = compile(&model);

    let mut natural = Engine::new(&compiled, EngineConfig::default()).unwrap();
    assert_eq!(natural.fire_named("d.req.resolve").unwrap().to, "ok");

    let mut forced = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let event = forced.fire_named_to("d.req.resolve", "ko").unwrap();
    assert_eq!(event.to, "ko");
    assert_eq!(forced.attribute("d.failure"), Some(Value::Bool(true)));
}

#[test]
fn forcing_a_non_target_state_is_typed_error() {
    let model = demand_model(0.7);
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    // `pending` is the source, not a declared target branch.
    let err = engine
        .fire_named_to("d.req.resolve", "pending")
        .unwrap_err();
    assert!(
        matches!(err, EngineError::ForcedTargetInvalid { .. }),
        "{err:?}"
    );

    // An unknown state name likewise.
    let err = engine.fire_named_to("d.req.resolve", "nope").unwrap_err();
    assert!(
        matches!(err, EngineError::ForcedTargetInvalid { .. }),
        "{err:?}"
    );
}

#[test]
fn fire_idx_to_forces_by_index() {
    let model = demand_model(0.7);
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    let target = engine
        .fireable()
        .into_iter()
        .find(|f| f.transition == "d.req.resolve")
        .unwrap();
    let event = engine.fire_idx_to(target.index, "ko").unwrap();
    assert_eq!(event.to, "ko");
    assert_eq!(engine.state("d.req"), Some("ko"));
}

// ---- Phase C — manual date-setting (`set_date`) -----------------------

#[test]
fn set_date_reschedules_a_pending_transition() {
    let model = two_component_model(); // A@5, B@8
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    // Push A from 5 to 10 → B (8) becomes the earliest.
    engine.set_date("A.fail.occ", 10.0).unwrap();
    let fireable = engine.fireable();
    assert_eq!(fireable[0].transition, "B.fail.occ");
    assert_eq!(fireable[0].date, Some(8.0));
    let a = fireable
        .iter()
        .find(|f| f.transition == "A.fail.occ")
        .unwrap();
    assert_eq!(a.date, Some(10.0));

    // A plain step now fires B first — the rescheduling took effect.
    let event = engine.step().unwrap().unwrap();
    assert_eq!(event.transition, "B.fail.occ");
    assert_eq!(event.time, 8.0);
}

#[test]
fn set_date_can_bring_a_transition_earlier() {
    let model = two_component_model();
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    // Pull B from 8 to 3 → it becomes the earliest and fires at 3.
    engine.set_date("B.fail.occ", 3.0).unwrap();
    let event = engine.step().unwrap().unwrap();
    assert_eq!(event.transition, "B.fail.occ");
    assert_eq!(event.time, 3.0);
    assert_eq!(engine.current_time(), 3.0);
}

#[test]
fn set_date_in_the_past_is_rejected() {
    let model = two_component_model();
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    engine.fire_named("B.fail.occ").unwrap(); // t → 8
    let err = engine.set_date("A.fail.occ", 5.0).unwrap_err(); // 5 < 8
    assert!(matches!(err, EngineError::DateInPast { .. }), "{err:?}");

    // Re-dating at or after the current time is accepted, and the
    // transition then fires exactly there.
    engine.set_date("A.fail.occ", 9.0).unwrap();
    let event = engine.fire_named("A.fail.occ").unwrap();
    assert_eq!(event.time, 9.0);
    assert_eq!(engine.current_time(), 9.0);
}

#[test]
fn set_date_on_unarmed_or_unknown_transition_is_rejected() {
    let model = two_component_model();
    let compiled = compile(&model);
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();

    // `rep` is not armed at t = 0 (its source `nok` is inactive).
    let err = engine.set_date("A.fail.rep", 10.0).unwrap_err();
    assert!(matches!(err, EngineError::NotFireable { .. }), "{err:?}");

    // Unknown transition name.
    let err = engine.set_date("A.fail.nope", 10.0).unwrap_err();
    assert!(
        matches!(err, EngineError::UnknownTransition { .. }),
        "{err:?}"
    );
}

// ---- Phase D — snapshot / restore + history + reset -------------------

fn bounded(compiled: &CompiledModel, t_max: f64) -> Engine<'_> {
    Engine::new(
        compiled,
        EngineConfig {
            t_max,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn snapshot_restore_round_trips_state_and_history() {
    let model = two_component_model();
    let compiled = compile(&model);
    let mut engine = bounded(&compiled, 30.0);

    engine.step().unwrap(); // fire A.occ @5
    let snap = engine.snapshot();
    let t_at_snap = engine.current_time();
    let hist_at_snap: Vec<_> = engine.history().to_vec();

    // Continue past the snapshot.
    engine.step().unwrap();
    engine.step().unwrap();
    assert!(engine.current_time() > t_at_snap);
    assert!(engine.history().len() > hist_at_snap.len());

    // Restore undoes everything: time, discrete state, and history.
    engine.restore(&snap);
    assert_eq!(engine.current_time(), t_at_snap);
    assert_eq!(engine.history(), hist_at_snap.as_slice());
    assert_eq!(engine.state("A.fail"), Some("nok"));
    assert_eq!(engine.state("B.fail"), Some("ok"));
}

#[test]
fn continuation_after_restore_is_reproducible() {
    // Fire to a point, snapshot, run to the horizon, restore, run
    // again: the two continuations produce identical event sequences.
    let model = two_component_model();
    let compiled = compile(&model);
    let mut engine = bounded(&compiled, 30.0);

    engine.step().unwrap();
    let snap = engine.snapshot();

    let mut first = Vec::new();
    while let Some(event) = engine.step().unwrap() {
        first.push(event);
    }

    engine.restore(&snap);
    let mut second = Vec::new();
    while let Some(event) = engine.step().unwrap() {
        second.push(event);
    }

    assert_eq!(first, second);
    assert!(!first.is_empty());
}

#[test]
fn reset_returns_to_a_fresh_initial_state() {
    let model = two_component_model();
    let compiled = compile(&model);
    let mut engine = bounded(&compiled, 30.0);

    let fresh_fireable = engine.fireable();
    engine.step().unwrap();
    engine.step().unwrap();
    assert!(!engine.history().is_empty());

    engine.reset().unwrap();
    assert_eq!(engine.current_time(), 0.0);
    assert!(engine.history().is_empty());
    assert_eq!(engine.state("A.fail"), Some("ok"));
    assert_eq!(engine.state("B.fail"), Some("ok"));
    // The schedule is regenerated identically to a fresh engine.
    assert_eq!(engine.fireable(), fresh_fireable);
}

#[test]
fn reset_then_run_matches_a_fresh_run() {
    let model = two_component_model();
    let compiled = compile(&model);

    let fresh = bounded(&compiled, 30.0).run().unwrap();

    let mut engine = bounded(&compiled, 30.0);
    engine.step().unwrap();
    engine.reset().unwrap();
    while engine.step().unwrap().is_some() {}

    assert_eq!(engine.history(), fresh.events.as_slice());
}
