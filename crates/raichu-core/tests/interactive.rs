//! Interactive-simulation control surface (isimu):
//!
//! - Phase A: `fireable()` + firing a *chosen* armed transition
//!   (`fire_named` / `fire_idx`) rather than only the earliest, then
//!   inspecting the resulting state through `attribute` / `state`.
//! - Phase B: forcing a chosen transition's destination branch
//!   (`fire_named_to` / `fire_idx_to`), bypassing the RNG /
//!   deterministic-branch resolution: the reproducible outcome control
//!   that makes stochastic mechanics testable.
//!
//! The single-trajectory engine stays deterministic; these methods add
//! only *control + observation* over the same cycle.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError, FireableKind};
use raichu_expr::{Assignment, AttrRef, Expr, StateRef, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Distrib, Equation, EquationKind, Model,
    SensitiveFunction, Transition,
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
                    monitored: false,
                    monitored_states: None,
                    cycle_group: None,
                    kind: None,
                    effects: vec![],
                    distrib: Distrib::Delay { time: ttf },
                },
                Transition {
                    name: "rep".into(),
                    source: "nok".into(),
                    guard: None,
                    targets: vec!["ok".into()],
                    on_interruption: Default::default(),
                    monitored: false,
                    monitored_states: None,
                    cycle_group: None,
                    kind: None,
                    effects: vec![],
                    distrib: Distrib::Delay { time: ttr },
                },
            ],
        }],
        // `up` reflects the automaton: true iff `fail` sits in `ok`.
        allocations: vec![],
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
        programs: vec![],
        name: "isimu_two".into(),
        components: vec![
            failing_component("A", 5.0, 10.0),
            failing_component("B", 8.0, 10.0),
        ],
        connections: vec![],
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
        unbounded_rate: None,
        fmu_units: vec![],
        interface_connections: vec![],
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
    // (date 5): the interactive override the plain `step()` cannot do.
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

// ---- Phase B: forced branch (`fire_*_to`) ----------------------------

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
        programs: vec![],
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
                    monitored: false,
                    monitored_states: None,
                    cycle_group: None,
                    kind: None,
                    effects: vec![],
                    distrib: Distrib::Inst {
                        probs: vec![ok_prob],
                    },
                }],
            }],
            allocations: vec![],
            equations: vec![],
            sensitive_functions: vec![reflect("success", "ok"), reflect("failure", "ko")],
        }],
        connections: vec![],
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
        unbounded_rate: None,
        fmu_units: vec![],
        interface_connections: vec![],
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

// ---- Phase C: manual date-setting (`set_date`) -----------------------

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

    // A plain step now fires B first: the rescheduling took effect.
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

// ---- Phase D: snapshot / restore + history + reset -------------------

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
fn advance_to_completes_events_and_moves_clock() {
    let compiled = compile(&two_component_model());
    let expected = bounded(&compiled, 18.0).run().unwrap();
    let mut engine = bounded(&compiled, 18.0);
    engine.advance_to(3.0).unwrap();
    assert_eq!(engine.current_time(), 3.0);
    engine.advance_to(8.0).unwrap();
    assert_eq!(engine.current_time(), 8.0);
    engine.advance_to(18.0).unwrap();
    assert_eq!(engine.history(), expected.events.as_slice());
    assert!(engine.advance_to(17.0).is_err());
}

#[test]
fn input_requires_explicit_permission_and_replays_from_snapshot() {
    let compiled = compile(&two_component_model());
    let mut engine = bounded(&compiled, 20.0);
    assert!(engine.set_input("A.up", Value::Bool(false), &[]).is_err());
    assert!(engine
        .set_input("A.up", Value::Int(1), &["A.up".to_owned()])
        .is_err());
    engine
        .set_input("A.up", Value::Bool(false), &["A.up".to_owned()])
        .unwrap();
    assert_eq!(engine.attribute("A.up"), Some(Value::Bool(false)));
    let snap = engine.snapshot().unwrap();
    engine.advance_to(20.0).unwrap();
    let first = engine.history().to_vec();
    engine.restore(&snap).unwrap();
    engine.advance_to(20.0).unwrap();
    assert_eq!(engine.history(), first.as_slice());
}

#[test]
fn input_change_fires_watched_guard_at_current_point() {
    let mut model = two_component_model();
    model.components[0].attributes.push(Attribute {
        name: "trigger".into(),
        kind: AttrKind::Float,
        init: Value::Float(0.0),
    });
    let transition = &mut model.components[0].automata[0].transitions[0];
    transition.distrib = Distrib::Watched;
    transition.guard = Some(Expr::Cmp {
        cmp: raichu_expr::CmpOp::Ge,
        lhs: Box::new(Expr::attr("A", "trigger")),
        rhs: Box::new(Expr::Const {
            value: Value::Float(0.5),
        }),
    });
    let compiled = compile(&model);
    let mut engine = bounded(&compiled, 10.0);
    engine
        .set_input("A.trigger", Value::Float(1.0), &["A.trigger".into()])
        .unwrap();
    assert_eq!(engine.current_time(), 0.0);
    assert_eq!(engine.history().len(), 1);
    assert_eq!(engine.history()[0].time, 0.0);
    assert_eq!(engine.state("A.fail"), Some("nok"));
}

#[test]
fn stepped_ode_matches_native_horizon() {
    let mut model = two_component_model();
    model.components[0].attributes.push(Attribute {
        name: "stock".into(),
        kind: AttrKind::Float,
        init: Value::Float(0.0),
    });
    model.components[0].equations.push(Equation {
        target: "stock".into(),
        kind: EquationKind::Ode,
        expr: Expr::Const {
            value: Value::Float(1.0),
        },
    });
    let compiled = compile(&model);
    let native = bounded(&compiled, 3.0).run().unwrap();
    let mut engine = bounded(&compiled, 3.0);
    for point in 1..=30 {
        engine.advance_to(f64::from(point) / 10.0).unwrap();
    }
    assert_eq!(engine.current_time(), 3.0);
    assert_eq!(engine.history(), native.events.as_slice());
    let Some(Value::Float(stock)) = engine.attribute("A.stock") else {
        panic!("missing ODE stock");
    };
    assert!((stock - 3.0).abs() < 1e-9);
}

#[test]
fn seeded_advance_replays_identically_after_restore() {
    let mut model = two_component_model();
    model.components[0].automata[0].transitions[0].distrib = Distrib::Exp {
        rate: Some(0.8),
        rate_expr: None,
    };
    let compiled = compile(&model);
    let mut engine = Engine::new(
        &compiled,
        EngineConfig {
            seed: 42,
            t_max: 10.0,
            ..Default::default()
        },
    )
    .unwrap();
    engine.advance_to(1.0).unwrap();
    let snap = engine.snapshot().unwrap();
    engine.advance_to(10.0).unwrap();
    let first = engine.history().to_vec();
    engine.restore(&snap).unwrap();
    engine.advance_to(10.0).unwrap();
    assert_eq!(engine.history(), first.as_slice());
}

#[test]
fn snapshot_restore_round_trips_state_and_history() {
    let model = two_component_model();
    let compiled = compile(&model);
    let mut engine = bounded(&compiled, 30.0);

    engine.step().unwrap(); // fire A.occ @5
    let snap = engine.snapshot().unwrap();
    let t_at_snap = engine.current_time();
    let hist_at_snap: Vec<_> = engine.history().to_vec();

    // Continue past the snapshot.
    engine.step().unwrap();
    engine.step().unwrap();
    assert!(engine.current_time() > t_at_snap);
    assert!(engine.history().len() > hist_at_snap.len());

    // Restore undoes everything: time, discrete state, and history.
    engine.restore(&snap).unwrap();
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
    let snap = engine.snapshot().unwrap();

    let mut first = Vec::new();
    while let Some(event) = engine.step().unwrap() {
        first.push(event);
    }

    engine.restore(&snap).unwrap();
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

#[test]
fn operator_reproducer_stops_before_unresolved_choice() {
    let compiled = compile(&demand_model(0.7));
    let mut engine = Engine::new(
        &compiled,
        EngineConfig {
            stochastic_dates: raichu_core::StochasticDates::Operator,
            t_max: 10.0,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let result = engine.advance_operator_to(5.0, 100).unwrap();
    assert_eq!(result.stop, raichu_core::OperatorStop::Choice);
    assert_eq!(engine.current_time(), 0.0);
    assert!(engine.history().is_empty());
    assert_eq!(engine.state("d.req"), Some("pending"));
}

fn operator_engine(compiled: &CompiledModel) -> Engine<'_> {
    Engine::new(
        compiled,
        EngineConfig {
            t_max: 20.0,
            stochastic_dates: raichu_core::StochasticDates::Operator,
            ..EngineConfig::default()
        },
    )
    .unwrap()
}

fn operator_stock_model() -> Model {
    Model::from_json(r#"{"name":"operator_stock","components":[
      {"name":"stock","attributes":[{"name":"level","kind":"float","init":{"kind":"float","value":4.0}}],
       "equations":[{"target":"level","kind":"ode","expr":{"op":"if","cond":{"op":"state_active","state":{"component":"stock","automaton":"bound","state":"full"}},"then":{"op":"const","value":{"kind":"float","value":-1.0}},"otherwise":{"op":"const","value":{"kind":"float","value":0.0}}}}],
       "automata":[{"name":"bound","states":["full","empty"],"init":"full","transitions":[{"name":"empty","source":"full","targets":["empty"],"distrib":"watched","guard":{"op":"cmp","cmp":"le","lhs":{"op":"attr","attr":{"component":"stock","attribute":"level"}},"rhs":{"op":"const","value":{"kind":"float","value":0.0}}}}]}]},
      {"name":"failure","automata":[{"name":"mode","states":["ok","failed"],"init":"ok","transitions":[{"name":"fail","source":"ok","targets":["failed"],"distrib":"exp","rate":0.5}]}]}
    ]}"#).unwrap()
}

#[test]
fn operator_boundary_keeps_programmed_failure_and_restores_exactly() {
    let compiled = compile(&operator_stock_model());
    let mut engine = operator_engine(&compiled);
    assert_eq!(
        engine
            .fireable()
            .iter()
            .find(|f| f.kind == FireableKind::Stochastic)
            .unwrap()
            .date,
        None
    );
    engine.set_date("failure.mode.fail", 8.0).unwrap();
    let before = engine.snapshot().unwrap();
    let boundary = engine.advance_operator_to(10.0, 100).unwrap();
    assert_eq!(boundary.stop, raichu_core::OperatorStop::Event);
    assert!((boundary.reached_time - 4.0).abs() < 1e-8);
    assert_eq!(boundary.events[0].transition, "stock.bound.empty");
    assert_eq!(engine.state("failure.mode"), Some("ok"));
    assert_eq!(
        engine
            .fireable()
            .iter()
            .find(|f| f.transition == "failure.mode.fail")
            .unwrap()
            .date,
        Some(8.0)
    );
    let result = engine.advance_operator_to(10.0, 100).unwrap();
    assert_eq!(result.reached_time, 8.0);
    assert_eq!(result.events[0].transition, "failure.mode.fail");
    let values = engine.attribute("stock.level");
    engine.restore(&before).unwrap();
    assert_eq!(engine.advance_operator_to(10.0, 100).unwrap(), boundary);
    assert_eq!(engine.advance_operator_to(10.0, 100).unwrap(), result);
    assert_eq!(engine.attribute("stock.level"), values);
    assert_eq!(
        engine.advance_operator_to(10.0, 100).unwrap().stop,
        raichu_core::OperatorStop::Target
    );
    assert_eq!(engine.current_time(), 10.0);
    engine.reset().unwrap();
    assert_eq!(engine.current_time(), 0.0);
    assert_eq!(
        engine
            .fireable()
            .iter()
            .find(|f| f.transition == "failure.mode.fail")
            .unwrap()
            .date,
        None
    );
}

#[test]
fn operator_unplanned_clock_and_invalid_commands_preserve_state() {
    let compiled = compile(&operator_stock_model());
    let mut engine = operator_engine(&compiled);
    engine.advance_operator_to(10.0, 100).unwrap();
    engine.advance_operator_to(10.0, 100).unwrap();
    assert_eq!(engine.state("failure.mode"), Some("ok"));
    let history = engine.history().to_vec();
    let level = engine.attribute("stock.level");
    for invalid in [f64::NAN, f64::INFINITY, -1.0, 9.0, 21.0] {
        assert!(engine.advance_operator_to(invalid, 100).is_err());
        assert!(engine.set_date("failure.mode.fail", invalid).is_err());
        assert_eq!(engine.current_time(), 10.0);
        assert_eq!(engine.history(), history);
        assert_eq!(engine.attribute("stock.level"), level);
    }
    assert!(engine.advance_to(12.0).is_err());
    assert!(engine.step().is_err());
}

#[test]
fn operator_choice_requires_explicit_destination_and_budget_commits_progress() {
    let compiled = compile(&demand_model(0.7));
    let mut engine = operator_engine(&compiled);
    assert!(engine.fire_named("d.req.resolve").is_err());
    assert!(engine.history().is_empty());
    let choice = engine.advance_operator_to(5.0, 100).unwrap();
    assert_eq!(choice.choice.as_deref(), Some("d.req.resolve"));
    assert_eq!(
        engine.fire_named_to("d.req.resolve", "ko").unwrap().to,
        "ko"
    );
    assert_eq!(
        engine.advance_operator_to(5.0, 100).unwrap().stop,
        raichu_core::OperatorStop::Target
    );
    let compiled = compile(&two_component_model());
    let mut engine = operator_engine(&compiled);
    let bounded = engine.advance_operator_to(10.0, 1).unwrap();
    assert_eq!(bounded.stop, raichu_core::OperatorStop::Incomplete);
    assert_eq!(bounded.reached_time, 5.0);
    assert_eq!(engine.history().len(), 1);
    assert_eq!(
        engine.advance_operator_to(10.0, 100).unwrap().reached_time,
        8.0
    );
}

#[test]
fn operator_countdowns_follow_each_interruption_policy() {
    use raichu_model::InterruptionPolicy;
    for (policy, expected) in [
        (InterruptionPolicy::Reset, None),
        (InterruptionPolicy::Resume, Some(10.0)),
        (InterruptionPolicy::Continue, Some(8.0)),
    ] {
        let mut model = two_component_model();
        model.components[0] = failing_component("G", 2.0, 2.0);
        model.components[1] = failing_component("E", 20.0, 20.0);
        let failure = &mut model.components[1].automata[0].transitions[0];
        failure.distrib = Distrib::Exp {
            rate: Some(0.5),
            rate_expr: None,
        };
        failure.guard = Some(Expr::attr("G", "up"));
        failure.on_interruption = policy;
        let compiled = compile(&model);
        let mut engine = operator_engine(&compiled);
        engine.set_date("E.fail.occ", 8.0).unwrap();
        engine.advance_operator_to(5.0, 100).unwrap(); // guard false at 2
        engine.advance_operator_to(5.0, 100).unwrap(); // guard true at 4
        engine.advance_operator_to(5.0, 100).unwrap();
        let armed = engine
            .fireable()
            .into_iter()
            .find(|f| f.transition == "E.fail.occ")
            .unwrap();
        assert_eq!(armed.date, expected, "{policy:?}");
        let deferred = engine
            .deferred()
            .into_iter()
            .find(|f| f.transition == "E.fail.occ")
            .unwrap();
        let expected_age = match policy {
            InterruptionPolicy::Reset => 1.0,
            InterruptionPolicy::Resume => 3.0,
            InterruptionPolicy::Continue => 5.0,
        };
        assert_eq!(deferred.age, expected_age, "{policy:?}");
    }
}

#[test]
fn operator_firing_error_rolls_back_native_mutations() {
    let mut model = two_component_model();
    // Initialization is stable; the fired state enables a genuine
    // sensitive-action oscillation, after native state/time/history changed.
    model.components[0].attributes.push(Attribute {
        name: "loop".into(),
        kind: AttrKind::Bool,
        init: Value::Bool(false),
    });
    model.components[0]
        .sensitive_functions
        .push(SensitiveFunction {
            name: "oscillate".into(),
            effects: vec![Assignment {
                target: AttrRef {
                    component: "A".into(),
                    attribute: "loop".into(),
                },
                value: Expr::If {
                    cond: Box::new(Expr::StateActive {
                        state: StateRef {
                            component: "A".into(),
                            automaton: "fail".into(),
                            state: "nok".into(),
                        },
                    }),
                    then: Box::new(Expr::Bool {
                        bool_op: raichu_expr::BoolOp::Not,
                        args: vec![Expr::attr("A", "loop")],
                    }),
                    otherwise: Box::new(Expr::Const {
                        value: Value::Bool(false),
                    }),
                },
            }],
        });
    let compiled = compile(&model);
    let mut engine = operator_engine(&compiled);
    assert!(engine.advance_operator_to(10.0, 100).is_err());
    assert_eq!(engine.current_time(), 0.0);
    assert_eq!(engine.state("A.fail"), Some("ok"));
    assert!(engine.history().is_empty());
    assert_eq!(engine.fireable()[0].date, Some(5.0));
    assert!(engine.fire_named("A.fail.occ").is_err());
    assert_eq!(engine.current_time(), 0.0);
    assert!(engine.history().is_empty());
}

#[test]
fn operator_snapshot_rejects_other_policy_and_model_without_mutation() {
    let compiled = compile(&two_component_model());
    let mut engine = operator_engine(&compiled);
    engine.advance_operator_to(3.0, 100).unwrap();
    let automatic = bounded(&compiled, 20.0).snapshot().unwrap();
    assert!(matches!(
        engine.restore(&automatic),
        Err(EngineError::OperatorSnapshotPolicy)
    ));
    let other_compiled = compile(&two_component_model());
    let foreign = operator_engine(&other_compiled).snapshot().unwrap();
    assert!(matches!(
        engine.restore(&foreign),
        Err(EngineError::OperatorSnapshotModel)
    ));
    assert_eq!(engine.current_time(), 3.0);
    assert!(engine.history().is_empty());
}

#[test]
fn operator_numerical_failure_restores_continuous_projection() {
    let mut model = operator_stock_model();
    model.components[0].equations[0].expr = Expr::If {
        cond: Box::new(Expr::Cmp {
            cmp: raichu_expr::CmpOp::Ge,
            lhs: Box::new(Expr::Time),
            rhs: Box::new(Expr::Const {
                value: Value::Float(0.5),
            }),
        }),
        then: Box::new(Expr::Div {
            lhs: Box::new(Expr::Const {
                value: Value::Float(1.0),
            }),
            rhs: Box::new(Expr::Const {
                value: Value::Float(0.0),
            }),
        }),
        otherwise: Box::new(Expr::Const {
            value: Value::Float(-1.0),
        }),
    };
    let compiled = compile(&model);
    let mut engine = operator_engine(&compiled);
    engine.advance_operator_to(0.25, 100).unwrap();
    let level = engine.attribute("stock.level");
    assert!(engine.advance_operator_to(2.0, 100).is_err());
    assert_eq!(engine.current_time(), 0.25);
    assert_eq!(engine.attribute("stock.level"), level);
    assert!(engine.history().is_empty());
    assert_eq!(engine.state("stock.bound"), Some("full"));
}

#[test]
fn operator_simultaneous_boundary_delay_and_explicit_date_converge() {
    let mut model = operator_stock_model();
    model.components.push(failing_component("D", 4.0, 20.0));
    let compiled = compile(&model);
    let mut engine = operator_engine(&compiled);
    engine.set_date("failure.mode.fail", 4.0).unwrap();
    let result = engine.advance_operator_to(10.0, 100).unwrap();
    assert_eq!(result.stop, raichu_core::OperatorStop::Event);
    assert_eq!(result.events.len(), 3);
    assert_eq!(result.events[0].transition, "stock.bound.empty");
    for event in &result.events {
        assert!((event.time - 4.0).abs() < 1e-8);
    }
    assert_eq!(engine.state("stock.bound"), Some("empty"));
    assert_eq!(engine.state("failure.mode"), Some("failed"));
    assert_eq!(engine.state("D.fail"), Some("nok"));
    assert_eq!(engine.attribute("D.up"), Some(Value::Bool(false)));
}

#[test]
fn operator_selected_firing_keeps_its_date_when_boundary_fires_instead() {
    let compiled = compile(&operator_stock_model());
    let mut engine = operator_engine(&compiled);
    engine.set_date("failure.mode.fail", 8.0).unwrap();
    let event = engine.fire_named("failure.mode.fail").unwrap();
    assert_eq!(event.transition, "stock.bound.empty");
    assert_eq!(engine.fireable()[0].date, Some(8.0));
    assert_eq!(engine.fire_named("failure.mode.fail").unwrap().time, 8.0);
}

#[test]
fn operator_reset_restores_transition_safety_budget() {
    let compiled = compile(&two_component_model());
    let mut engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 20.0,
            stochastic_dates: raichu_core::StochasticDates::Operator,
            max_transition_firings: 2,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    engine.advance_operator_to(5.0, 100).unwrap();
    engine.reset().unwrap();
    assert_eq!(
        engine.advance_operator_to(5.0, 100).unwrap().reached_time,
        5.0
    );
}

#[test]
fn operator_single_positive_branch_does_not_require_choice() {
    for probability in [0.0, 1.0] {
        let compiled = compile(&demand_model(probability));
        let mut engine = operator_engine(&compiled);
        let outcome = engine.advance_operator_to(5.0, 100).unwrap();
        assert_eq!(outcome.stop, raichu_core::OperatorStop::Event);
        assert_eq!(outcome.choice, None);
        assert_eq!(outcome.reached_time, 0.0);
        assert_eq!(
            outcome.events[0].to,
            if probability == 1.0 { "ok" } else { "ko" }
        );
    }
}
