//! Engine benchmarks (M1 goal condition): expression-evaluation /
//! fixpoint throughput on the discrete cycle, and hybrid ODE
//! integration with watched-transition location.
//!
//! Run with `cargo bench`. These establish the baseline demanded by the
//! performance contract: regressions become visible; the
//! blocking side-by-side thresholds arrive with the Monte-Carlo
//! milestone (M2), where volumes are realistic.

#![allow(clippy::unwrap_used, missing_docs)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::{Assignment, AttrRef, CmpOp, Expr, PortRef, StateRef, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Connection, Distrib, Equation, EquationKind, Model,
    Port, PortDir, SensitiveFunction, Transition,
};

/// delay_001-class model: discrete cycle + sensitive-function fixpoint
/// (expression evaluation dominates).
fn delay_model() -> Model {
    Model {
        name: "bench_delay".into(),
        components: vec![
            Component {
                name: "source".into(),
                attributes: vec![Attribute {
                    name: "flow_out".into(),
                    kind: AttrKind::Bool,
                    init: Value::Bool(true),
                }],
                ports: vec![Port {
                    name: "out".into(),
                    dir: PortDir::Out,
                    attr: Some("flow_out".into()),
                }],
                interfaces: vec![],
                automata: vec![Automaton {
                    name: "failure".into(),
                    states: vec!["ok".into(), "nok".into()],
                    init: "ok".into(),
                    transitions: vec![
                        Transition {
                            name: "fail".into(),
                            source: "ok".into(),
                            guard: None,
                            targets: vec!["nok".into()],
                            on_interruption: Default::default(),
                            distrib: Distrib::Delay { time: 5.0 },
                        },
                        Transition {
                            name: "repair".into(),
                            source: "nok".into(),
                            guard: None,
                            targets: vec!["ok".into()],
                            on_interruption: Default::default(),
                            distrib: Distrib::Delay { time: 10.0 },
                        },
                    ],
                }],
                equations: vec![],
                sensitive_functions: vec![SensitiveFunction {
                    name: "update_flow_out".into(),
                    effects: vec![Assignment {
                        target: AttrRef {
                            component: "source".into(),
                            attribute: "flow_out".into(),
                        },
                        value: Expr::StateActive {
                            state: StateRef {
                                component: "source".into(),
                                automaton: "failure".into(),
                                state: "ok".into(),
                            },
                        },
                    }],
                }],
            },
            Component {
                name: "target".into(),
                attributes: vec![Attribute {
                    name: "fed".into(),
                    kind: AttrKind::Bool,
                    init: Value::Bool(false),
                }],
                ports: vec![Port {
                    name: "input".into(),
                    dir: PortDir::In,
                    attr: None,
                }],
                interfaces: vec![],
                automata: vec![],
                equations: vec![],
                sensitive_functions: vec![SensitiveFunction {
                    name: "update_fed".into(),
                    effects: vec![Assignment {
                        target: AttrRef {
                            component: "target".into(),
                            attribute: "fed".into(),
                        },
                        value: Expr::PortAgg {
                            port: PortRef {
                                component: "target".into(),
                                port: "input".into(),
                            },
                            agg: raichu_expr::AggOp::Any,
                        },
                    }],
                }],
            },
        ],
        connections: vec![Connection {
            from: PortRef {
                component: "source".into(),
                port: "out".into(),
            },
            to: PortRef {
                component: "target".into(),
                port: "input".into(),
            },
        }],
        indicators: vec![],
    }
}

/// tank_01-class model: ODE integration + watched-transition location.
fn tank_model() -> Model {
    let content = || Expr::attr("tank", "content");
    Model {
        name: "bench_tank".into(),
        components: vec![Component {
            name: "tank".into(),
            attributes: vec![Attribute {
                name: "content".into(),
                kind: AttrKind::Float,
                init: Value::Float(0.0),
            }],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "pump".into(),
                states: vec!["off".into(), "on".into()],
                init: "off".into(),
                transitions: vec![
                    Transition {
                        name: "start".into(),
                        source: "off".into(),
                        guard: Some(Expr::Cmp {
                            cmp: CmpOp::Ge,
                            lhs: Box::new(content()),
                            rhs: Box::new(Expr::Const {
                                value: Value::Float(8.0),
                            }),
                        }),
                        targets: vec!["on".into()],
                        on_interruption: Default::default(),
                        distrib: Distrib::Watched,
                    },
                    Transition {
                        name: "stop".into(),
                        source: "on".into(),
                        guard: Some(Expr::Cmp {
                            cmp: CmpOp::Le,
                            lhs: Box::new(content()),
                            rhs: Box::new(Expr::Const {
                                value: Value::Float(2.0),
                            }),
                        }),
                        targets: vec!["off".into()],
                        on_interruption: Default::default(),
                        distrib: Distrib::Watched,
                    },
                ],
            }],
            equations: vec![Equation {
                target: "content".into(),
                kind: EquationKind::Ode,
                expr: Expr::If {
                    cond: Box::new(Expr::StateActive {
                        state: StateRef {
                            component: "tank".into(),
                            automaton: "pump".into(),
                            state: "on".into(),
                        },
                    }),
                    then: Box::new(Expr::Sub {
                        lhs: Box::new(Expr::Const {
                            value: Value::Float(1.5),
                        }),
                        rhs: Box::new(Expr::Const {
                            value: Value::Float(2.0),
                        }),
                    }),
                    otherwise: Box::new(Expr::Const {
                        value: Value::Float(1.5),
                    }),
                },
            }],
            sensitive_functions: vec![],
        }],
        connections: vec![],
        indicators: vec![],
    }
}

/// Discrete cycle: 1000 time units ≈ 134 events, each with fixpoint
/// propagation through two components (expression evaluation).
#[divan::bench]
fn discrete_fixpoint_cycle(bencher: divan::Bencher) {
    let model = delay_model();
    let compiled = CompiledModel::compile(&model).unwrap();
    bencher.bench(|| {
        let config = EngineConfig {
            t_max: 1_000.0,
            ..EngineConfig::default()
        };
        let result = Engine::new(&compiled, config).unwrap().run().unwrap();
        divan::black_box(result.events.len())
    });
}

/// Hybrid cycle: 160 time units ≈ 20 watched crossings, each located by
/// dense-output scan + bisection over the adaptive DP45 integration.
#[divan::bench]
fn ode_integration_with_watched_events(bencher: divan::Bencher) {
    let model = tank_model();
    let compiled = CompiledModel::compile(&model).unwrap();
    bencher.bench(|| {
        let config = EngineConfig {
            t_max: 160.0,
            ..EngineConfig::default()
        };
        let result = Engine::new(&compiled, config).unwrap().run().unwrap();
        divan::black_box(result.events.len())
    });
}

fn main() {
    divan::main();
}
