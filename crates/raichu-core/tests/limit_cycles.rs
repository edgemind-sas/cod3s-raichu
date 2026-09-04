//! Limit cycles: a model that advances time by a little every turn and
//! never gets anywhere.
//!
//! The engine had two Zeno guards and both look at ONE instant.
//! `WatchedLoop` counts watched firings at the same instant;
//! `FlowChattering` counts active-set restarts at the same instant. A
//! cycle that advances the clock by a hysteresis width, or by a solver
//! step, escapes both: the run does not fail, it grinds, and the
//! trajectory reads correctly at every sample instant while it does.
//!
//! Two budgets close that. They do not decide whether a model is right;
//! they enforce that a simulation terminates and say what was spinning.
//! Both are configurable, because a genuinely fast-switching model is a
//! legitimate thing to want and a cap that could not be raised would be
//! a limit on what may be modelled.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError};
use raichu_expr::{CmpOp, Expr, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Distrib, Equation, EquationKind, Model, Transition,
};

/// A saw-tooth: `x` rises at 1, a watched pair sends it back down at 1
/// and up again at 0, so the two transitions fire forever, a hundred
/// times per unit of time. Time advances every turn, so neither
/// same-instant guard sees it.
fn sawtooth() -> Model {
    Model {
        name: "sawtooth".into(),
        components: vec![Component {
            name: "C".into(),
            attributes: vec![
                Attribute {
                    name: "x".into(),
                    kind: AttrKind::Float,
                    init: Value::Float(0.0),
                },
                Attribute {
                    name: "rate".into(),
                    kind: AttrKind::Float,
                    init: Value::Float(1.0),
                },
            ],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "cycle".into(),
                states: vec!["up".into(), "down".into()],
                init: "up".into(),
                transitions: vec![
                    Transition {
                        name: "top".into(),
                        source: "up".into(),
                        guard: Some(Expr::Cmp {
                            cmp: CmpOp::Ge,
                            lhs: Box::new(Expr::attr("C", "x")),
                            rhs: Box::new(Expr::Const {
                                value: Value::Float(0.01),
                            }),
                        }),
                        targets: vec!["down".into()],
                        on_interruption: Default::default(),
                        monitored: false,
                        cycle_group: None,
                        distrib: Distrib::Watched,
                    },
                    Transition {
                        name: "bottom".into(),
                        source: "down".into(),
                        guard: Some(Expr::Cmp {
                            cmp: CmpOp::Le,
                            lhs: Box::new(Expr::attr("C", "x")),
                            rhs: Box::new(Expr::Const {
                                value: Value::Float(0.0),
                            }),
                        }),
                        targets: vec!["up".into()],
                        on_interruption: Default::default(),
                        monitored: false,
                        cycle_group: None,
                        distrib: Distrib::Watched,
                    },
                ],
            }],
            allocations: vec![],
            equations: vec![
                Equation {
                    target: "rate".into(),
                    kind: EquationKind::Explicit,
                    expr: Expr::If {
                        cond: Box::new(Expr::StateActive {
                            state: raichu_expr::StateRef {
                                component: "C".into(),
                                automaton: "cycle".into(),
                                state: "up".into(),
                            },
                        }),
                        then: Box::new(Expr::Const {
                            value: Value::Float(1.0),
                        }),
                        otherwise: Box::new(Expr::Const {
                            value: Value::Float(-1.0),
                        }),
                    },
                },
                Equation {
                    target: "x".into(),
                    kind: EquationKind::Ode,
                    expr: Expr::attr("C", "rate"),
                },
            ],
            sensitive_functions: vec![],
        }],
        connections: vec![],
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
    }
}

fn run(model: &Model, budget: u64, t_max: f64) -> Result<usize, EngineError> {
    let compiled = CompiledModel::compile(model).unwrap();
    let config = EngineConfig {
        t_max,
        max_transition_firings: budget,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config)
        .unwrap()
        .run()
        .map(|result| result.events.len())
}

#[test]
fn a_transition_limit_cycle_fails_with_a_diagnosis() {
    let error = run(&sawtooth(), 500, 1_000.0).expect_err("the cycle should be caught");
    match error {
        EngineError::TransitionChattering {
            transition,
            firings,
            step,
            since,
            time,
        } => {
            assert!(
                transition.ends_with("top") || transition.ends_with("bottom"),
                "{transition}"
            );
            assert_eq!(firings, 500);
            // The number that makes the diagnosis land: a hundredth of a
            // unit per turn, against a horizon of a thousand.
            assert!(step > 0.0 && step < 1.0, "{step}");
            assert!(time > since, "{since} .. {time}");
        }
        other => panic!("wrong error: {other}"),
    }
}

#[test]
fn the_budget_is_per_transition_and_a_normal_model_is_untouched() {
    // The same model on a horizon it settles within: two firings, far
    // below a budget that the cycle above blew through.
    let events = run(&sawtooth(), 500, 0.015).expect("no cycle in this horizon");
    assert!(events < 10, "{events}");
}

#[test]
fn a_zero_budget_lifts_the_guard() {
    // A cap that could not be raised would be a limit on what may be
    // modelled rather than a diagnostic.
    let events = run(&sawtooth(), 0, 1.0).expect("the guard is off");
    // Two turns of the saw per 0.02 of time, so a hundred turns per unit:
    // far past the budget of 500 the previous test blew through on a
    // longer horizon, and allowed here because the cap is lifted.
    assert!(events > 90, "{events}");
}

#[test]
fn the_default_budget_is_high_enough_for_an_ordinary_model() {
    assert_eq!(EngineConfig::default().max_transition_firings, 100_000);
    assert_eq!(EngineConfig::default().max_flow_restarts, 100_000);
}
