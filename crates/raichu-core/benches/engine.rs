//! Engine benchmarks (M1 goal condition): expression-evaluation /
//! fixpoint throughput on the discrete cycle, and hybrid ODE
//! integration with watched-transition location.
//!
//! Run with `cargo bench`. These establish the baseline demanded by the
//! performance contract: regressions become visible; the
//! blocking side-by-side thresholds arrive with the Monte-Carlo
//! milestone (M2), where volumes are realistic.
//!
//! Two of them come as a **pair**, on the same synthetic network of a
//! few hundred edges. `wide_network_watched_scan` carries a watched
//! protection on every branch, which is the population the scan sites
//! walk; `wide_network_no_watched` is the same network without them,
//! which pays whatever the explicit pass costs per written attribute and
//! receives none of the narrowing. Reading them together is what
//! separates a scan that got cheaper from a pass that got dearer: a
//! change that improves the first while moving the second is trading,
//! not winning, and the pair says so.

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
                    channels: vec![],
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
                            monitored: false,
                            cycle_group: None,
                            distrib: Distrib::Delay { time: 5.0 },
                        },
                        Transition {
                            name: "repair".into(),
                            source: "nok".into(),
                            guard: None,
                            targets: vec!["ok".into()],
                            on_interruption: Default::default(),
                            monitored: false,
                            cycle_group: None,
                            distrib: Distrib::Delay { time: 10.0 },
                        },
                    ],
                }],
                allocations: vec![],
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
                    channels: vec![],
                }],
                interfaces: vec![],
                automata: vec![],
                allocations: vec![],
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
                            channel: None,
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
            name: None,
        }],
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
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
                        monitored: false,
                        cycle_group: None,
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
                        monitored: false,
                        cycle_group: None,
                        distrib: Distrib::Watched,
                    },
                ],
            }],
            allocations: vec![],
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
        targets: vec![],
        evaluation_order: None,
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

/// Width of the synthetic network: how many independent branches hang off
/// the shared bus, hence how many connections (edges), explicit equations
/// and watched trip guards the model carries.
///
/// A few hundred is the size at which the *scan* sites stop being noise:
/// below it a full scan is short enough to hide inside the fixpoint,
/// above it the model is no longer synthetic but slow.
const NETWORK_WIDTH: usize = 200;

/// Horizon of the synthetic-network benches: long enough that every
/// branch fails and is repaired several times, short enough that one
/// sample stays in the tens of milliseconds.
const NETWORK_HORIZON: f64 = 60.0;

/// A synthetic network of `NETWORK_WIDTH` edges, shaped like the discrete
/// flow models the platform actually runs: one shared bus, one connection
/// per branch, one explicit equation per branch, and one **local**
/// failure automaton per branch.
///
/// The locality is the point. A branch failing moves two attributes of
/// that branch and nothing else, so a scan whose cost tracks the whole
/// model does `NETWORK_WIDTH` times the work the change justifies. With
/// `watched` set, every branch also carries a protection pair whose
/// guards read only that branch's supplied quantity: that is the
/// population the immediate-guard scan walks.
fn wide_network(watched: bool) -> Model {
    let n = NETWORK_WIDTH;
    let mut components = Vec::with_capacity(n + 1);
    let mut connections = Vec::with_capacity(n);

    components.push(Component {
        name: "bus".into(),
        attributes: vec![Attribute {
            name: "power".into(),
            kind: AttrKind::Float,
            init: Value::Float(1.0),
        }],
        ports: vec![Port {
            name: "out".into(),
            dir: PortDir::Out,
            attr: Some("power".into()),
            channels: vec![],
        }],
        interfaces: vec![],
        automata: vec![Automaton {
            name: "outage".into(),
            states: vec!["ok".into(), "ko".into()],
            init: "ok".into(),
            transitions: vec![
                Transition {
                    name: "trip".into(),
                    source: "ok".into(),
                    guard: None,
                    targets: vec!["ko".into()],
                    on_interruption: Default::default(),
                    monitored: false,
                    cycle_group: None,
                    distrib: Distrib::Delay { time: 61.0 },
                },
                Transition {
                    name: "restore".into(),
                    source: "ko".into(),
                    guard: None,
                    targets: vec!["ok".into()],
                    on_interruption: Default::default(),
                    monitored: false,
                    cycle_group: None,
                    distrib: Distrib::Delay { time: 13.0 },
                },
            ],
        }],
        allocations: vec![],
        equations: vec![],
        sensitive_functions: vec![SensitiveFunction {
            name: "update_power".into(),
            effects: vec![Assignment {
                target: AttrRef {
                    component: "bus".into(),
                    attribute: "power".into(),
                },
                value: Expr::If {
                    cond: Box::new(Expr::StateActive {
                        state: StateRef {
                            component: "bus".into(),
                            automaton: "outage".into(),
                            state: "ok".into(),
                        },
                    }),
                    then: Box::new(Expr::Const {
                        value: Value::Float(1.0),
                    }),
                    otherwise: Box::new(Expr::Const {
                        value: Value::Float(0.0),
                    }),
                },
            }],
        }],
    });

    for i in 0..n {
        let load = format!("load{i}");
        let mut automata = vec![Automaton {
            name: "failure".into(),
            states: vec!["ok".into(), "ko".into()],
            init: "ok".into(),
            transitions: vec![
                Transition {
                    name: "fail".into(),
                    source: "ok".into(),
                    guard: None,
                    targets: vec!["ko".into()],
                    on_interruption: Default::default(),
                    monitored: false,
                    cycle_group: None,
                    distrib: Distrib::Delay {
                        time: 7.0 + (i % 29) as f64,
                    },
                },
                Transition {
                    name: "repair".into(),
                    source: "ko".into(),
                    guard: None,
                    targets: vec!["ok".into()],
                    on_interruption: Default::default(),
                    monitored: false,
                    cycle_group: None,
                    distrib: Distrib::Delay {
                        time: 3.0 + (i % 17) as f64,
                    },
                },
            ],
        }];
        if watched {
            let supplied = || Expr::attr(&load, "supplied");
            automata.push(Automaton {
                name: "protection".into(),
                states: vec!["closed".into(), "open".into()],
                init: "closed".into(),
                transitions: vec![
                    Transition {
                        name: "open".into(),
                        source: "closed".into(),
                        guard: Some(Expr::Cmp {
                            cmp: CmpOp::Le,
                            lhs: Box::new(supplied()),
                            rhs: Box::new(Expr::Const {
                                value: Value::Float(0.25),
                            }),
                        }),
                        targets: vec!["open".into()],
                        on_interruption: Default::default(),
                        monitored: false,
                        cycle_group: None,
                        distrib: Distrib::Watched,
                    },
                    Transition {
                        name: "close".into(),
                        source: "open".into(),
                        guard: Some(Expr::Cmp {
                            cmp: CmpOp::Ge,
                            lhs: Box::new(supplied()),
                            rhs: Box::new(Expr::Const {
                                value: Value::Float(0.75),
                            }),
                        }),
                        targets: vec!["closed".into()],
                        on_interruption: Default::default(),
                        monitored: false,
                        cycle_group: None,
                        distrib: Distrib::Watched,
                    },
                ],
            });
        }
        components.push(Component {
            name: load.clone(),
            attributes: vec![
                Attribute {
                    name: "available".into(),
                    kind: AttrKind::Float,
                    init: Value::Float(1.0),
                },
                Attribute {
                    name: "supplied".into(),
                    kind: AttrKind::Float,
                    init: Value::Float(1.0),
                },
            ],
            ports: vec![Port {
                name: "input".into(),
                dir: PortDir::In,
                attr: None,
                channels: vec![],
            }],
            interfaces: vec![],
            automata,
            allocations: vec![],
            equations: vec![Equation {
                target: "supplied".into(),
                kind: EquationKind::Explicit,
                expr: Expr::Mul {
                    args: vec![
                        Expr::PortAgg {
                            port: PortRef {
                                component: load.clone(),
                                port: "input".into(),
                            },
                            agg: raichu_expr::AggOp::Sum,
                            channel: None,
                        },
                        Expr::attr(&load, "available"),
                    ],
                },
            }],
            sensitive_functions: vec![SensitiveFunction {
                name: "update_available".into(),
                effects: vec![Assignment {
                    target: AttrRef {
                        component: load.clone(),
                        attribute: "available".into(),
                    },
                    value: Expr::If {
                        cond: Box::new(Expr::StateActive {
                            state: StateRef {
                                component: load.clone(),
                                automaton: "failure".into(),
                                state: "ok".into(),
                            },
                        }),
                        then: Box::new(Expr::Const {
                            value: Value::Float(1.0),
                        }),
                        otherwise: Box::new(Expr::Const {
                            value: Value::Float(0.0),
                        }),
                    },
                }],
            }],
        });
        connections.push(Connection {
            from: PortRef {
                component: "bus".into(),
                port: "out".into(),
            },
            to: PortRef {
                component: load,
                port: "input".into(),
            },
            name: None,
        });
    }

    Model {
        name: if watched {
            "bench_wide_network".into()
        } else {
            "bench_wide_network_no_watched".into()
        },
        components,
        connections,
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
    }
}

/// **Scan site.** A few hundred edges, each with a watched protection
/// pair, driven by *local* failures: the immediate-guard scan runs after
/// every discrete fixpoint over the whole watched population, while at
/// most one branch's quantities have moved.
#[divan::bench]
fn wide_network_watched_scan(bencher: divan::Bencher) {
    let model = wide_network(true);
    let compiled = CompiledModel::compile(&model).unwrap();
    bencher.bench(|| {
        let config = EngineConfig {
            t_max: NETWORK_HORIZON,
            ..EngineConfig::default()
        };
        let result = Engine::new(&compiled, config).unwrap().run().unwrap();
        divan::black_box(result.events.len())
    });
}

/// **Control.** The same network with the watched protection pairs
/// removed: it pays whatever the explicit pass costs per written
/// attribute and receives none of the scan narrowing, which is what
/// prices change detection on a discrete-only model.
#[divan::bench]
fn wide_network_no_watched(bencher: divan::Bencher) {
    let model = wide_network(false);
    let compiled = CompiledModel::compile(&model).unwrap();
    bencher.bench(|| {
        let config = EngineConfig {
            t_max: NETWORK_HORIZON,
            ..EngineConfig::default()
        };
        let result = Engine::new(&compiled, config).unwrap().run().unwrap();
        divan::black_box(result.events.len())
    });
}

fn main() {
    divan::main();
}
