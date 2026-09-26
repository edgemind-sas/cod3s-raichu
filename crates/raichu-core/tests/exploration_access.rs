//! Exploration accessors: what a sequence-tree explorer reads from, and
//! does to, an engine without sampling.
//!
//! - `armed_rate`: the current exponential rate of an armed transition
//!   (constant, piecewise-constant state-dependent, or the instantaneous
//!   value of a continuously varying one), `None` for other laws.
//! - `reached_target`: the latched feared event, if any.
//! - `fire_now`: fire an armed transition at the current instant, the
//!   clock unmoved, optionally forcing the destination by branch index.
//! - `fire_idx_to_branch`: the scheduled-date counterpart of
//!   `fire_idx_to`, forcing the destination by branch index.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::compile::CLaw;
use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError, Event};
use raichu_expr::{CmpOp, Expr, StateRef, Value};
use raichu_model::{AttrKind, Attribute, Automaton, Component, Distrib, Model, Target, Transition};

/// A transition `name` from `source` to `targets` under `distrib`.
fn transition(name: &str, source: &str, targets: &[&str], distrib: Distrib) -> Transition {
    Transition {
        name: name.into(),
        source: source.into(),
        guard: None,
        targets: targets.iter().map(|t| (*t).into()).collect(),
        on_interruption: Default::default(),
        monitored: false,
        cycle_group: None,
        kind: None,
        effects: vec![],
        distrib,
    }
}

/// A component with one automaton `fail` (`ok`, `nok`) whose failure
/// `occ` follows `occ_law`, repaired after a fixed delay.
fn component(name: &str, occ_law: Distrib) -> Component {
    Component {
        name: name.into(),
        attributes: vec![],
        ports: vec![],
        interfaces: vec![],
        automata: vec![Automaton {
            name: "fail".into(),
            states: vec!["ok".into(), "nok".into()],
            init: "ok".into(),
            transitions: vec![
                transition("occ", "ok", &["nok"], occ_law),
                transition("rep", "nok", &["ok"], Distrib::Delay { time: 10.0 }),
            ],
        }],
        allocations: vec![],
        equations: vec![],
        sensitive_functions: vec![],
    }
}

fn model(components: Vec<Component>, targets: Vec<Target>) -> Model {
    Model {
        name: "exploration_access".into(),
        components,
        connections: vec![],
        indicators: vec![],
        targets,
        evaluation_order: None,
        unbounded_rate: None,
    }
}

fn exp(rate: f64) -> Distrib {
    Distrib::Exp {
        rate: Some(rate),
        rate_expr: None,
    }
}

fn exp_expr(rate_expr: Expr) -> Distrib {
    Distrib::Exp {
        rate: None,
        rate_expr: Some(rate_expr),
    }
}

fn constant(value: f64) -> Expr {
    Expr::Const {
        value: Value::Float(value),
    }
}

/// `if A.fail.nok then when_down else when_up`: a rate that changes with
/// the discrete state of `A` only (piecewise constant between jumps).
fn rate_on_a(when_down: f64, when_up: f64) -> Expr {
    Expr::If {
        cond: Box::new(Expr::StateActive {
            state: StateRef {
                component: "A".into(),
                automaton: "fail".into(),
                state: "nok".into(),
            },
        }),
        then: Box::new(constant(when_down)),
        otherwise: Box::new(constant(when_up)),
    }
}

fn compile(model: &Model) -> CompiledModel {
    CompiledModel::compile(model).unwrap()
}

fn index(compiled: &CompiledModel, name: &str) -> usize {
    compiled
        .transitions
        .iter()
        .position(|t| t.name == name)
        .unwrap()
}

fn seeded(seed: u64) -> EngineConfig {
    EngineConfig {
        seed,
        ..EngineConfig::default()
    }
}

// ---- armed_rate --------------------------------------------------------

#[test]
fn armed_rate_of_a_constant_exponential_is_its_rate() {
    let compiled = compile(&model(vec![component("A", exp(0.3))], vec![]));
    let engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let occ = index(&compiled, "A.fail.occ");
    assert_eq!(engine.armed_rate(occ).unwrap(), Some(0.3));
}

#[test]
fn armed_rate_of_an_unarmed_transition_is_none() {
    // `rep` sits in `nok`, not the active state: nothing is armed.
    let compiled = compile(&model(vec![component("A", exp(0.3))], vec![]));
    let engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let rep = index(&compiled, "A.fail.rep");
    assert_eq!(engine.armed_rate(rep).unwrap(), None);
}

#[test]
fn armed_rate_of_an_unknown_index_is_a_typed_error() {
    let compiled = compile(&model(vec![component("A", exp(0.3))], vec![]));
    let engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let err = engine.armed_rate(99).unwrap_err();
    assert!(
        matches!(err, EngineError::UnknownTransition { .. }),
        "{err:?}"
    );
}

#[test]
fn armed_rate_follows_a_piecewise_constant_state_dependent_rate() {
    let compiled = compile(&model(
        vec![
            component("A", exp(0.3)),
            component("B", exp_expr(rate_on_a(2.0, 0.5))),
        ],
        vec![],
    ));
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let b_occ = index(&compiled, "B.fail.occ");
    assert_eq!(engine.armed_rate(b_occ).unwrap(), Some(0.5));

    engine
        .fire_now(index(&compiled, "A.fail.occ"), None)
        .unwrap();
    assert_eq!(engine.armed_rate(b_occ).unwrap(), Some(2.0));
}

#[test]
fn armed_rate_reports_a_zero_rate_as_zero_not_none() {
    // A dormant spare: armed, but its rate is 0 until A fails. The
    // explorer must see `Some(0.0)` to skip it, not mistake it for an
    // unarmed transition or a non-exponential law.
    let compiled = compile(&model(
        vec![
            component("A", exp(0.3)),
            component("B", exp_expr(rate_on_a(1.0, 0.0))),
        ],
        vec![],
    ));
    let engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let b_occ = index(&compiled, "B.fail.occ");
    assert_eq!(engine.armed_rate(b_occ).unwrap(), Some(0.0));
}

#[test]
fn armed_rate_of_a_continuously_varying_rate_is_its_current_value() {
    // λ(t) = t + 1 varies between jumps: `armed_rate` returns its value
    // now; the law family (`continuous: true`) is read from the model.
    let rate = Expr::Add {
        args: vec![Expr::Time, constant(1.0)],
    };
    let compiled = compile(&model(vec![component("A", exp_expr(rate))], vec![]));
    let engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let occ = index(&compiled, "A.fail.occ");
    assert!(matches!(
        compiled.transitions[occ].distrib,
        CLaw::ExpVar {
            continuous: true,
            ..
        }
    ));
    assert_eq!(engine.armed_rate(occ).unwrap(), Some(1.0));
}

#[test]
fn armed_rate_of_weibull_and_delay_is_none() {
    let compiled = compile(&model(
        vec![
            component(
                "A",
                Distrib::Weibull {
                    shape: 2.0,
                    scale: 5.0,
                },
            ),
            component("B", Distrib::Delay { time: 3.0 }),
        ],
        vec![],
    ));
    let engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    assert_eq!(
        engine.armed_rate(index(&compiled, "A.fail.occ")).unwrap(),
        None
    );
    assert_eq!(
        engine.armed_rate(index(&compiled, "B.fail.occ")).unwrap(),
        None
    );
}

// ---- fire_now ------------------------------------------------------------

#[test]
fn fire_now_leaves_the_clock_unmoved() {
    let compiled = compile(&model(
        vec![component("A", exp(0.3)), component("B", exp(0.1))],
        vec![],
    ));
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let event = engine
        .fire_now(index(&compiled, "B.fail.occ"), None)
        .unwrap();
    assert_eq!(event.time, 0.0);
    assert_eq!(event.to, "nok");
    assert_eq!(engine.current_time(), 0.0);
    assert_eq!(engine.state("B.fail"), Some("nok"));
    assert_eq!(engine.state("A.fail"), Some("ok"));
}

#[test]
fn fire_now_of_an_unarmed_transition_is_not_fireable() {
    let compiled = compile(&model(vec![component("A", exp(0.3))], vec![]));
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let err = engine
        .fire_now(index(&compiled, "A.fail.rep"), None)
        .unwrap_err();
    assert!(matches!(err, EngineError::NotFireable { .. }), "{err:?}");
}

/// A protection `P.protection` (`closed`, `open`) opened by a watched
/// transition when the constant attribute `P.level` is at most 0.5.
fn protection(level: f64) -> Component {
    let mut open = transition("open", "closed", &["open"], Distrib::Watched);
    open.guard = Some(Expr::Cmp {
        cmp: CmpOp::Le,
        lhs: Box::new(Expr::attr("P", "level")),
        rhs: Box::new(constant(0.5)),
    });
    Component {
        name: "P".into(),
        attributes: vec![Attribute {
            name: "level".into(),
            kind: AttrKind::Float,
            init: Value::Float(level),
        }],
        ports: vec![],
        interfaces: vec![],
        automata: vec![Automaton {
            name: "protection".into(),
            states: vec!["closed".into(), "open".into()],
            init: "closed".into(),
            transitions: vec![open],
        }],
        allocations: vec![],
        equations: vec![],
        sensitive_functions: vec![],
    }
}

#[test]
fn fire_now_fires_a_watched_transition_whose_guard_already_holds() {
    // A watched transition is never scheduled: `fire_now` takes its own
    // branch for one whose boundary is already crossed at t = 0.
    let compiled = compile(&model(vec![protection(0.0)], vec![]));
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let event = engine
        .fire_now(index(&compiled, "P.protection.open"), None)
        .unwrap();
    assert_eq!(event.time, 0.0);
    assert_eq!(event.transition, "P.protection.open");
    assert_eq!(event.from, "closed");
    assert_eq!(event.to, "open");
    assert_eq!(engine.current_time(), 0.0);
    assert_eq!(engine.state("P.protection"), Some("open"));
}

#[test]
fn fire_now_of_a_watched_transition_whose_guard_fails_is_not_fireable() {
    let compiled = compile(&model(vec![protection(1.0)], vec![]));
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let err = engine
        .fire_now(index(&compiled, "P.protection.open"), None)
        .unwrap_err();
    assert!(matches!(err, EngineError::NotFireable { .. }), "{err:?}");
    assert_eq!(engine.state("P.protection"), Some("closed"));
}

/// Exploring-style run: repeatedly fire the lowest-index armed
/// transition at the current instant, through a fixed path, and return
/// the fired events and final discrete states.
fn explore_path(compiled: &CompiledModel, seed: u64) -> (Vec<Event>, Vec<String>) {
    let mut engine = Engine::new(compiled, seeded(seed)).unwrap();
    let mut events = Vec::new();
    for _ in 0..6 {
        let Some(next) = engine.fireable().into_iter().map(|f| f.index).min() else {
            break;
        };
        events.push(engine.fire_now(next, None).unwrap());
    }
    let states = ["A.fail", "B.fail", "C.fail"]
        .iter()
        .map(|a| engine.state(a).unwrap().to_owned())
        .collect();
    (events, states)
}

#[test]
fn exploring_with_two_different_seeds_yields_the_same_discrete_path() {
    // Firing at the current instant ignores every drawn date, so the
    // discrete path cannot depend on the seed.
    let compiled = compile(&model(
        vec![
            component("A", exp(0.3)),
            component("B", exp_expr(rate_on_a(2.0, 0.5))),
            component(
                "C",
                Distrib::Weibull {
                    shape: 2.0,
                    scale: 5.0,
                },
            ),
        ],
        vec![],
    ));
    let first = explore_path(&compiled, 1);
    let second = explore_path(&compiled, 987_654_321);
    assert_eq!(first, second);
    assert!(first.0.iter().all(|e| e.time == 0.0));
    assert!(!first.0.is_empty());
}

// ---- forced destination by branch index --------------------------------

/// A one-shot demand: instantaneous `resolve` from `pending` to `ok`
/// (probability `ok_prob`) or `ko` (the complement).
fn demand_model(ok_prob: f64) -> Model {
    model(
        vec![Component {
            name: "d".into(),
            attributes: vec![],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "req".into(),
                states: vec!["pending".into(), "ok".into(), "ko".into()],
                init: "pending".into(),
                transitions: vec![transition(
                    "resolve",
                    "pending",
                    &["ok", "ko"],
                    Distrib::Inst {
                        probs: vec![ok_prob],
                    },
                )],
            }],
            allocations: vec![],
            equations: vec![],
            sensitive_functions: vec![],
        }],
        vec![],
    )
}

#[test]
fn fire_now_forces_the_destination_by_branch_index() {
    let compiled = compile(&demand_model(0.9));
    let resolve = index(&compiled, "d.req.resolve");
    // The branch probabilities are read from the compiled law.
    let CLaw::Inst(probs) = &compiled.transitions[resolve].distrib else {
        panic!("expected an instantaneous law");
    };
    assert_eq!(probs.len(), 2);
    assert!((probs[1] - 0.1).abs() < 1e-15);

    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let event = engine.fire_now(resolve, Some(1)).unwrap();
    assert_eq!(event.to, "ko");
    assert_eq!(engine.state("d.req"), Some("ko"));
}

#[test]
fn fire_idx_to_branch_forces_the_destination_by_branch_index() {
    let compiled = compile(&demand_model(0.9));
    let resolve = index(&compiled, "d.req.resolve");
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let event = engine.fire_idx_to_branch(resolve, 0).unwrap();
    assert_eq!(event.to, "ok");
}

#[test]
fn a_branch_index_out_of_range_is_a_typed_error() {
    let compiled = compile(&demand_model(0.9));
    let resolve = index(&compiled, "d.req.resolve");
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let before = engine.snapshot();

    let err = engine.fire_now(resolve, Some(2)).unwrap_err();
    assert!(
        matches!(
            err,
            EngineError::ForcedBranchOutOfRange {
                branch: 2,
                branches: 2,
                ..
            }
        ),
        "{err:?}"
    );
    let err = engine.fire_idx_to_branch(resolve, 5).unwrap_err();
    assert!(
        matches!(err, EngineError::ForcedBranchOutOfRange { .. }),
        "{err:?}"
    );
    // A refused firing changes nothing.
    assert_eq!(engine.state("d.req"), Some("pending"));
    assert_eq!(engine.snapshot().time(), before.time());
}

// ---- reached_target ----------------------------------------------------

#[test]
fn reached_target_is_none_before_and_named_after_the_firing() {
    let compiled = compile(&model(
        vec![component("A", exp(0.3)), component("B", exp(0.1))],
        vec![Target {
            name: "a_down".into(),
            component: "A".into(),
            automaton: "fail".into(),
            state: "nok".into(),
        }],
    ));
    let mut engine = Engine::new(&compiled, config_with_targets()).unwrap();
    assert_eq!(engine.reached_target(), None);

    engine
        .fire_now(index(&compiled, "B.fail.occ"), None)
        .unwrap();
    assert_eq!(engine.reached_target(), None);

    engine
        .fire_now(index(&compiled, "A.fail.occ"), None)
        .unwrap();
    assert_eq!(engine.reached_target(), Some(("a_down", 0.0)));

    // Restoring a snapshot taken before the firing rewinds the latch.
    let compiled_again = compile(&model(
        vec![component("A", exp(0.3))],
        vec![Target {
            name: "a_down".into(),
            component: "A".into(),
            automaton: "fail".into(),
            state: "nok".into(),
        }],
    ));
    let mut engine = Engine::new(&compiled_again, config_with_targets()).unwrap();
    let root = engine.snapshot();
    engine
        .fire_now(index(&compiled_again, "A.fail.occ"), None)
        .unwrap();
    assert!(engine.reached_target().is_some());
    engine.restore(&root);
    assert_eq!(engine.reached_target(), None);
}

fn config_with_targets() -> EngineConfig {
    EngineConfig {
        stop_at_targets: true,
        ..EngineConfig::default()
    }
}
