//! Integration tests of the deterministic engine — Phase 2 exit
//! criteria of the M0 plan.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError, JournalRecord};
use raichu_expr::{AggOp, Assignment, AttrRef, Expr, PortRef, StateRef, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Connection, Distrib, Indicator, IndicatorTarget,
    Interface, Model, Port, PortDir, SensitiveFunction, Transition,
};

/// The `delay_001` fixture shape (muscadet
/// `test_comp_failure_cod3s_delay_001`): a source with a failure
/// automaton (ttf = 5, ttr = 10) feeding a target through a boolean
/// flow. Expected: flow drops at t=5, recovers at t=15, drops at t=20…
fn delay_model() -> Model {
    Model {
        name: "delay_001".into(),
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
                interfaces: vec![Interface {
                    name: "outputs".into(),
                    ports: vec!["out".into()],
                }],
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
                            agg: AggOp::Any,
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
        indicators: vec![
            Indicator {
                name: "target_fed".into(),
                target: IndicatorTarget::Attribute {
                    attr: AttrRef {
                        component: "target".into(),
                        attribute: "fed".into(),
                    },
                },
            },
            Indicator {
                name: "source_nok".into(),
                target: IndicatorTarget::State {
                    component: "source".into(),
                    automaton: "failure".into(),
                    state: "nok".into(),
                },
            },
        ],
        targets: vec![],
    }
}

fn run(model: &Model, t_max: f64, journal: bool) -> raichu_core::SimulationResult {
    let compiled = CompiledModel::compile(model).unwrap();
    let config = EngineConfig {
        t_max,
        journal,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config).unwrap().run().unwrap()
}

#[test]
fn delay_model_fires_at_5_and_repairs_at_15() {
    let result = run(&delay_model(), 22.0, false);
    let summary: Vec<(f64, &str, &str, &str)> = result
        .events
        .iter()
        .map(|e| {
            (
                e.time,
                e.transition.as_str(),
                e.from.as_str(),
                e.to.as_str(),
            )
        })
        .collect();
    assert_eq!(
        summary,
        vec![
            (5.0, "source.failure.fail", "ok", "nok"),
            (15.0, "source.failure.repair", "nok", "ok"),
            (20.0, "source.failure.fail", "ok", "nok"),
        ]
    );
}

#[test]
fn flow_propagates_through_the_port_at_init_and_on_failure() {
    let result = run(&delay_model(), 22.0, false);
    // target_fed: false at t=0 pre-fixpoint → true after init fixpoint,
    // false at 5, true at 15, false at 20.
    let fed = &result.indicators[0];
    assert_eq!(fed.name, "target_fed");
    assert_eq!(
        fed.points,
        vec![
            (0.0, Value::Bool(true)),
            (5.0, Value::Bool(false)),
            (15.0, Value::Bool(true)),
            (20.0, Value::Bool(false)),
        ]
    );
    // source_nok state indicator: 0 → 1 → 0 → 1.
    let nok = &result.indicators[1];
    assert_eq!(
        nok.points,
        vec![
            (0.0, Value::Float(0.0)),
            (5.0, Value::Float(1.0)),
            (15.0, Value::Float(0.0)),
            (20.0, Value::Float(1.0)),
        ]
    );
}

#[test]
fn replay_is_byte_identical() {
    let model = delay_model();
    let a = serde_json::to_vec(&run(&model, 100.0, true)).unwrap();
    let b = serde_json::to_vec(&run(&model, 100.0, true)).unwrap();
    assert_eq!(a, b);
}

#[test]
fn journal_records_the_causality_chain() {
    let result = run(&delay_model(), 6.0, true);
    // At t=5: transition fired → update_flow_out triggered → flow_out
    // changed → update_fed triggered → fed changed.
    let at_5: Vec<&JournalRecord> = result
        .journal
        .iter()
        .filter(|r| match r {
            JournalRecord::TransitionFired { time, .. }
            | JournalRecord::FunctionTriggered { time, .. }
            | JournalRecord::AttributeChanged { time, .. }
            | JournalRecord::TransitionScheduled { time, .. }
            | JournalRecord::TransitionRescheduled { time, .. }
            | JournalRecord::TransitionDropped { time, .. } => *time == 5.0,
        })
        .collect();
    let kinds: Vec<String> = at_5
        .iter()
        .map(|r| match r {
            JournalRecord::TransitionFired { transition, .. } => format!("fired:{transition}"),
            JournalRecord::FunctionTriggered { function, .. } => format!("fn:{function}"),
            JournalRecord::AttributeChanged { attribute, .. } => format!("attr:{attribute}"),
            JournalRecord::TransitionScheduled { transition, .. } => {
                format!("sched:{transition}")
            }
            JournalRecord::TransitionRescheduled { transition, .. } => {
                format!("resched:{transition}")
            }
            JournalRecord::TransitionDropped { transition, .. } => format!("drop:{transition}"),
        })
        .collect();
    assert_eq!(
        kinds,
        vec![
            "fired:source.failure.fail",
            "fn:source.update_flow_out",
            "attr:source.flow_out",
            "fn:target.update_fed",
            "attr:target.fed",
            "sched:source.failure.repair",
        ]
    );
}

#[test]
fn journal_off_means_no_records() {
    let result = run(&delay_model(), 22.0, false);
    assert!(result.journal.is_empty());
}

#[test]
fn two_engines_coexist_in_one_process() {
    // Engines are not process singletons: many coexist per process.
    let model_a = delay_model();
    let mut model_b = delay_model();
    model_b.components[0].automata[0].transitions[0].distrib = Distrib::Delay { time: 7.0 };

    let compiled_a = CompiledModel::compile(&model_a).unwrap();
    let compiled_b = CompiledModel::compile(&model_b).unwrap();
    let mut engine_a = Engine::new(&compiled_a, EngineConfig::default()).unwrap();
    let mut engine_b = Engine::new(&compiled_b, EngineConfig::default()).unwrap();

    // Interleave stepping: each engine keeps its own clock and state.
    let event_a = engine_a.step().unwrap().unwrap();
    let event_b = engine_b.step().unwrap().unwrap();
    assert_eq!(event_a.time, 5.0);
    assert_eq!(event_b.time, 7.0);
    assert_eq!(engine_a.state("source.failure"), Some("nok"));
    assert_eq!(engine_b.state("source.failure"), Some("nok"));
    assert_eq!(engine_a.current_time(), 5.0);
    assert_eq!(engine_b.current_time(), 7.0);
}

#[test]
fn step_api_exposes_state_and_variables() {
    let model = delay_model();
    let compiled = CompiledModel::compile(&model).unwrap();
    let mut engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    assert_eq!(engine.attribute("source.flow_out"), Some(Value::Bool(true)));
    assert_eq!(engine.attribute("target.fed"), Some(Value::Bool(true)));
    engine.step().unwrap();
    assert_eq!(
        engine.attribute("source.flow_out"),
        Some(Value::Bool(false))
    );
    assert_eq!(engine.attribute("target.fed"), Some(Value::Bool(false)));
}

/// An instantaneous loop: a function that rewrites its own input never
/// converges — the engine reports a typed error instead of hanging
/// (Design Goal #6).
#[test]
fn instantaneous_loop_is_detected() {
    let model = Model {
        name: "loop".into(),
        components: vec![Component {
            name: "c".into(),
            attributes: vec![Attribute {
                name: "x".into(),
                kind: AttrKind::Bool,
                init: Value::Bool(false),
            }],
            ports: vec![],
            interfaces: vec![],
            automata: vec![],
            equations: vec![],
            sensitive_functions: vec![SensitiveFunction {
                name: "flip".into(),
                effects: vec![Assignment {
                    target: AttrRef {
                        component: "c".into(),
                        attribute: "x".into(),
                    },
                    value: Expr::Bool {
                        bool_op: raichu_expr::BoolOp::Not,
                        args: vec![Expr::attr("c", "x")],
                    },
                }],
            }],
        }],
        connections: vec![],
        indicators: vec![],
        targets: vec![],
    };
    let compiled = CompiledModel::compile(&model).unwrap();
    let result = Engine::new(&compiled, EngineConfig::default());
    assert!(matches!(
        result.map(|_| ()),
        Err(EngineError::InstantaneousLoop { .. })
    ));
}

/// A non-confluent model: two functions write conflicting values to the
/// same attribute from the same trigger. The confluence probe reports it
/// (rather than silently returning an order-dependent result).
#[test]
fn non_confluence_is_diagnosed() {
    let model = Model {
        name: "non_confluent".into(),
        components: vec![Component {
            name: "c".into(),
            attributes: vec![
                Attribute {
                    name: "a".into(),
                    kind: AttrKind::Bool,
                    init: Value::Bool(false),
                },
                Attribute {
                    name: "x".into(),
                    kind: AttrKind::Bool,
                    init: Value::Bool(false),
                },
            ],
            ports: vec![],
            interfaces: vec![],
            automata: vec![],
            equations: vec![],
            sensitive_functions: vec![
                SensitiveFunction {
                    name: "writer_identity".into(),
                    effects: vec![Assignment {
                        target: AttrRef {
                            component: "c".into(),
                            attribute: "x".into(),
                        },
                        value: Expr::attr("c", "a"),
                    }],
                },
                SensitiveFunction {
                    name: "writer_negation".into(),
                    effects: vec![Assignment {
                        target: AttrRef {
                            component: "c".into(),
                            attribute: "x".into(),
                        },
                        value: Expr::Bool {
                            bool_op: raichu_expr::BoolOp::Not,
                            args: vec![Expr::attr("c", "a")],
                        },
                    }],
                },
            ],
        }],
        connections: vec![],
        indicators: vec![],
        targets: vec![],
    };
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        confluence_check: true,
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config);
    assert!(matches!(
        result.map(|_| ()),
        Err(EngineError::NonConfluent { .. })
    ));
}

/// The same non-confluent model passes silently without the probe —
/// with the documented deterministic order (declaration order), the
/// last writer wins. This documents *why* the probe exists.
#[test]
fn non_confluent_model_is_order_deterministic_without_probe() {
    // Same model as above.
    let model = Model {
        name: "non_confluent".into(),
        components: vec![Component {
            name: "c".into(),
            attributes: vec![
                Attribute {
                    name: "a".into(),
                    kind: AttrKind::Bool,
                    init: Value::Bool(false),
                },
                Attribute {
                    name: "x".into(),
                    kind: AttrKind::Bool,
                    init: Value::Bool(false),
                },
            ],
            ports: vec![],
            interfaces: vec![],
            automata: vec![],
            equations: vec![],
            sensitive_functions: vec![
                SensitiveFunction {
                    name: "writer_identity".into(),
                    effects: vec![Assignment {
                        target: AttrRef {
                            component: "c".into(),
                            attribute: "x".into(),
                        },
                        value: Expr::attr("c", "a"),
                    }],
                },
                SensitiveFunction {
                    name: "writer_negation".into(),
                    effects: vec![Assignment {
                        target: AttrRef {
                            component: "c".into(),
                            attribute: "x".into(),
                        },
                        value: Expr::Bool {
                            bool_op: raichu_expr::BoolOp::Not,
                            args: vec![Expr::attr("c", "a")],
                        },
                    }],
                },
            ],
        }],
        connections: vec![],
        indicators: vec![],
        targets: vec![],
    };
    let compiled = CompiledModel::compile(&model).unwrap();
    let engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    // Declaration order: writer_negation runs last → x = !a = true.
    assert_eq!(engine.attribute("c.x"), Some(Value::Bool(true)));
}

/// Instantaneous branching with a certain branch fires immediately and
/// deterministically.
#[test]
fn inst_transition_fires_immediately_on_certain_branch() {
    let model = Model {
        name: "inst".into(),
        components: vec![Component {
            name: "c".into(),
            attributes: vec![],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "a".into(),
                states: vec!["wait".into(), "go".into(), "end".into()],
                init: "wait".into(),
                transitions: vec![
                    Transition {
                        name: "arm".into(),
                        source: "wait".into(),
                        guard: None,
                        targets: vec!["go".into()],
                        on_interruption: Default::default(),
                        monitored: false,
                        cycle_group: None,
                        distrib: Distrib::Delay { time: 3.0 },
                    },
                    Transition {
                        name: "jump".into(),
                        source: "go".into(),
                        guard: None,
                        // probs = [] → single target with complement 1.
                        targets: vec!["end".into()],
                        on_interruption: Default::default(),
                        monitored: false,
                        cycle_group: None,
                        distrib: Distrib::Inst { probs: vec![] },
                    },
                ],
            }],
            equations: vec![],
            sensitive_functions: vec![],
        }],
        connections: vec![],
        indicators: vec![],
        targets: vec![],
    };
    let result = run(&model, 10.0, false);
    let times: Vec<(f64, &str)> = result
        .events
        .iter()
        .map(|e| (e.time, e.to.as_str()))
        .collect();
    assert_eq!(times, vec![(3.0, "go"), (3.0, "end")]);
}

/// Provenance is emitted with every result.
#[test]
fn provenance_is_recorded() {
    let result = run(&delay_model(), 22.0, false);
    assert_eq!(result.provenance.model, "delay_001");
    assert_eq!(result.provenance.t_max, 22.0);
    assert_eq!(result.provenance.seed, None);
    assert!(!result.provenance.engine_version.is_empty());
}

// --- M1: hybrid tank (ODE + watched transitions) ---------------------------

/// Reduced heated-tank-class model (M1 plan): constant inflow 1.5,
/// pump automaton toggling outflow 2.0 through watched thresholds
/// content ≥ 8 (start) and content ≤ 2 (stop), content₀ = 0.
/// Closed-form event dates: 16/3, 52/3, 64/3, 100/3, 112/3 …
fn tank_model() -> Model {
    use raichu_expr::{CmpOp, StateRef};
    let content = || Expr::attr("tank", "content");
    Model {
        name: "tank_01".into(),
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
            equations: vec![raichu_model::Equation {
                target: "content".into(),
                kind: raichu_model::EquationKind::Ode,
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
        indicators: vec![
            Indicator {
                name: "content".into(),
                target: IndicatorTarget::Attribute {
                    attr: AttrRef {
                        component: "tank".into(),
                        attribute: "content".into(),
                    },
                },
            },
            Indicator {
                name: "pump_on".into(),
                target: IndicatorTarget::State {
                    component: "tank".into(),
                    automaton: "pump".into(),
                    state: "on".into(),
                },
            },
        ],
        targets: vec![],
    }
}

#[test]
fn tank_watched_events_match_closed_form() {
    let model = tank_model();
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max: 38.0,
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    let expected = [
        (16.0 / 3.0, "start"),
        (52.0 / 3.0, "stop"),
        (64.0 / 3.0, "start"),
        (100.0 / 3.0, "stop"),
        (112.0 / 3.0, "start"),
    ];
    assert_eq!(result.events.len(), expected.len(), "{:#?}", result.events);
    for (event, (date, name)) in result.events.iter().zip(expected) {
        assert!(
            (event.time - date).abs() < 1e-8,
            "expected {name} at {date}, got {} at {}",
            event.transition,
            event.time
        );
        assert!(event.transition.ends_with(name));
    }
    // Boundary values at the located crossings: content ≈ 8 then ≈ 2.
    let content = &result.indicators[0];
    assert!((content.points[1].0 - 16.0 / 3.0).abs() < 1e-8);
    match content.points[1].1 {
        Value::Float(v) => assert!((v - 8.0).abs() < 1e-6, "content at start = {v}"),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn tank_dense_samples_match_closed_form() {
    let model = tank_model();
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max: 24.0,
        samples: vec![5.0, 10.0, 20.0],
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    let content: Vec<(f64, f64)> = result.samples[0]
        .points
        .iter()
        .map(|(t, v)| match v {
            Value::Float(f) => (*t, *f),
            other => panic!("unexpected {other:?}"),
        })
        .collect();
    // Closed form: rise at 1.5 until 16/3; fall at -0.5 until 52/3;
    // rise again at 1.5.
    let expected = [
        (5.0, 7.5),
        (10.0, 8.0 - 0.5 * (10.0 - 16.0 / 3.0)),
        (20.0, 2.0 + 1.5 * (20.0 - 52.0 / 3.0)),
    ];
    assert_eq!(content.len(), 3, "{content:?}");
    for ((t, v), (et, ev)) in content.iter().zip(expected) {
        assert!((t - et).abs() < 1e-12);
        assert!((v - ev).abs() < 1e-6, "content({t}) = {v}, expected {ev}");
    }
}

#[test]
fn tank_runs_on_the_euler_backend_too() {
    // The OdeSolver trait is the swap point (M1 goal condition).
    let model = tank_model();
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max: 10.0,
        ..EngineConfig::default()
    };
    let solver = Box::new(raichu_numeric::FixedEuler {
        step: 1e-3,
        tol_event: 1e-9,
    });
    let mut engine = Engine::with_solver(&compiled, config, solver).unwrap();
    let event = engine.step().unwrap().unwrap();
    assert!(
        (event.time - 16.0 / 3.0).abs() < 1e-2,
        "Euler-located crossing at {}",
        event.time
    );
}

// --- interruption policies: reset / resume / continue ----------------------

/// Gate opens on [0,4) then [7,∞); a worker needs a 6 h occurrence
/// duration gated by the open condition. The three policies diverge
/// precisely here (naming set by Roland):
/// `continue` → fires at 6 (countdown never stops);
/// `reset` → 13 (duration cancelled, fresh draw at 7 —
/// RAICHU default);
/// `resume` → 9 (4 h done + remaining 2 h after 7, RAICHU extension).
fn gate_worker_model(on_interruption: raichu_model::InterruptionPolicy) -> Model {
    use raichu_expr::BoolOp;
    Model {
        name: "interrupt_probe".into(),
        components: vec![
            Component {
                name: "gate".into(),
                attributes: vec![Attribute {
                    name: "open".into(),
                    kind: AttrKind::Bool,
                    init: Value::Bool(true),
                }],
                ports: vec![],
                interfaces: vec![],
                automata: vec![Automaton {
                    name: "phase".into(),
                    states: vec!["up1".into(), "down".into(), "up2".into()],
                    init: "up1".into(),
                    transitions: vec![
                        Transition {
                            name: "close".into(),
                            source: "up1".into(),
                            guard: None,
                            targets: vec!["down".into()],
                            on_interruption: Default::default(),
                            monitored: false,
                            cycle_group: None,
                            distrib: Distrib::Delay { time: 4.0 },
                        },
                        Transition {
                            name: "reopen".into(),
                            source: "down".into(),
                            guard: None,
                            targets: vec!["up2".into()],
                            on_interruption: Default::default(),
                            monitored: false,
                            cycle_group: None,
                            distrib: Distrib::Delay { time: 3.0 },
                        },
                    ],
                }],
                equations: vec![],
                sensitive_functions: vec![SensitiveFunction {
                    name: "update_open".into(),
                    effects: vec![Assignment {
                        target: AttrRef {
                            component: "gate".into(),
                            attribute: "open".into(),
                        },
                        value: Expr::Bool {
                            bool_op: BoolOp::Or,
                            args: vec![
                                Expr::StateActive {
                                    state: StateRef {
                                        component: "gate".into(),
                                        automaton: "phase".into(),
                                        state: "up1".into(),
                                    },
                                },
                                Expr::StateActive {
                                    state: StateRef {
                                        component: "gate".into(),
                                        automaton: "phase".into(),
                                        state: "up2".into(),
                                    },
                                },
                            ],
                        },
                    }],
                }],
            },
            Component {
                name: "worker".into(),
                attributes: vec![],
                ports: vec![],
                interfaces: vec![],
                automata: vec![Automaton {
                    name: "job".into(),
                    states: vec!["wait".into(), "done".into()],
                    init: "wait".into(),
                    transitions: vec![Transition {
                        name: "finish".into(),
                        source: "wait".into(),
                        guard: Some(Expr::attr("gate", "open")),
                        targets: vec!["done".into()],
                        on_interruption,
                        monitored: false,
                        cycle_group: None,
                        distrib: Distrib::Delay { time: 6.0 },
                    }],
                }],
                equations: vec![],
                sensitive_functions: vec![],
            },
        ],
        connections: vec![],
        indicators: vec![],
        targets: vec![],
    }
}

fn finish_date(on_interruption: raichu_model::InterruptionPolicy) -> f64 {
    let model = gate_worker_model(on_interruption);
    let compiled = CompiledModel::compile(&model).unwrap();
    let result = Engine::new(&compiled, EngineConfig::default())
        .unwrap()
        .run()
        .unwrap();
    result
        .events
        .iter()
        .find(|e| e.transition == "worker.job.finish")
        .map(|e| e.time)
        .unwrap()
}

#[test]
fn continue_policy_countdown_never_stops() {
    // Continue policy: fires at 6 even though the gate closed at 4.
    assert_eq!(finish_date(raichu_model::InterruptionPolicy::Continue), 6.0);
}

#[test]
fn reset_policy_draws_a_fresh_duration() {
    // Reset policy (RAICHU default): re-armed at 7.
    assert_eq!(finish_date(raichu_model::InterruptionPolicy::Reset), 13.0);
}

#[test]
fn resume_policy_continues_the_countdown() {
    // RAICHU extension: 4 h elapsed + remaining 2 h after 7 → 9.
    assert_eq!(finish_date(raichu_model::InterruptionPolicy::Resume), 9.0);
}

#[test]
fn interruption_policy_without_guard_is_rejected() {
    let mut model = gate_worker_model(raichu_model::InterruptionPolicy::Resume);
    model.components[1].automata[0].transitions[0].guard = None;
    assert!(matches!(
        model.validate(),
        Err(raichu_model::ModelError::InterruptionPolicyWithoutGuard { .. })
    ));
}

/// `reschedule_modifiable`: a state-dependent rate held at λ = 0 (firing date on
/// hold at +∞) is rescheduled — with a causal-journal record — when a
/// discrete step switches its inputs, and the banked `Exp(1)` threshold
/// then produces the firing.
#[test]
fn expvar_rate_change_is_rescheduled_and_journaled() {
    let model = Model {
        name: "expvar_updatemt".into(),
        components: vec![Component {
            name: "C".into(),
            attributes: vec![],
            ports: vec![],
            interfaces: vec![],
            automata: vec![
                Automaton {
                    name: "env".into(),
                    states: vec!["cold".into(), "hot".into()],
                    init: "cold".into(),
                    transitions: vec![Transition {
                        name: "heat".into(),
                        source: "cold".into(),
                        guard: None,
                        targets: vec!["hot".into()],
                        on_interruption: Default::default(),
                        monitored: false,
                        cycle_group: None,
                        distrib: Distrib::Delay { time: 5.0 },
                    }],
                },
                Automaton {
                    name: "aut".into(),
                    states: vec!["ok".into(), "nok".into()],
                    init: "ok".into(),
                    transitions: vec![Transition {
                        name: "fire".into(),
                        source: "ok".into(),
                        guard: None,
                        targets: vec!["nok".into()],
                        on_interruption: Default::default(),
                        monitored: false,
                        cycle_group: None,
                        distrib: Distrib::Exp {
                            rate: None,
                            rate_expr: Some(Expr::If {
                                cond: Box::new(Expr::StateActive {
                                    state: StateRef {
                                        component: "C".into(),
                                        automaton: "env".into(),
                                        state: "hot".into(),
                                    },
                                }),
                                then: Box::new(Expr::Const {
                                    value: Value::Float(10.0),
                                }),
                                otherwise: Box::new(Expr::Const {
                                    value: Value::Float(0.0),
                                }),
                            }),
                        },
                    }],
                },
            ],
            equations: vec![],
            sensitive_functions: vec![],
        }],
        connections: vec![],
        indicators: vec![],
        targets: vec![],
    };
    let result = run(&model, 30.0, true);

    let rescheduled: Vec<f64> = result
        .journal
        .iter()
        .filter_map(|record| match record {
            JournalRecord::TransitionRescheduled {
                time, transition, ..
            } if transition == "C.aut.fire" => Some(*time),
            _ => None,
        })
        .collect();
    assert_eq!(
        rescheduled,
        vec![5.0],
        "one reschedule_modifiable record at the switch"
    );

    let fired = result
        .events
        .iter()
        .find(|e| e.transition == "C.aut.fire")
        .expect("the transition fires once the rate becomes positive");
    assert!(
        fired.time > 5.0,
        "no firing while λ = 0 (got {})",
        fired.time
    );
}

/// M3 aggregation extensions: `mean` and `median` over an in-port
/// (audit M1 list — sensor averaging, redundant-sensor median).
#[test]
fn port_mean_and_median_aggregations() {
    let sensor = |name: &str, value: f64| Component {
        name: name.into(),
        attributes: vec![Attribute {
            name: "v".into(),
            kind: AttrKind::Float,
            init: Value::Float(value),
        }],
        ports: vec![Port {
            name: "out".into(),
            dir: PortDir::Out,
            attr: Some("v".into()),
        }],
        interfaces: vec![],
        automata: vec![],
        equations: vec![],
        sensitive_functions: vec![],
    };
    let model = Model {
        name: "agg_probe".into(),
        components: vec![
            sensor("s1", 10.0),
            sensor("s2", 99.0),
            sensor("s3", 12.0),
            Component {
                name: "voter".into(),
                attributes: vec![
                    Attribute {
                        name: "mean".into(),
                        kind: AttrKind::Float,
                        init: Value::Float(0.0),
                    },
                    Attribute {
                        name: "median".into(),
                        kind: AttrKind::Float,
                        init: Value::Float(0.0),
                    },
                ],
                ports: vec![Port {
                    name: "in".into(),
                    dir: PortDir::In,
                    attr: None,
                }],
                interfaces: vec![],
                automata: vec![],
                equations: vec![],
                sensitive_functions: vec![SensitiveFunction {
                    name: "aggregate".into(),
                    effects: vec![
                        Assignment {
                            target: AttrRef {
                                component: "voter".into(),
                                attribute: "mean".into(),
                            },
                            value: Expr::PortAgg {
                                port: PortRef {
                                    component: "voter".into(),
                                    port: "in".into(),
                                },
                                agg: AggOp::Mean,
                            },
                        },
                        Assignment {
                            target: AttrRef {
                                component: "voter".into(),
                                attribute: "median".into(),
                            },
                            value: Expr::PortAgg {
                                port: PortRef {
                                    component: "voter".into(),
                                    port: "in".into(),
                                },
                                agg: AggOp::Median,
                            },
                        },
                    ],
                }],
            },
        ],
        connections: (1..=3)
            .map(|k| Connection {
                from: PortRef {
                    component: format!("s{k}"),
                    port: "out".into(),
                },
                to: PortRef {
                    component: "voter".into(),
                    port: "in".into(),
                },
            })
            .collect(),
        indicators: vec![],
        targets: vec![],
    };
    let compiled = CompiledModel::compile(&model).unwrap();
    let engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    // Mean pulled toward the outlier; median votes it out.
    assert_eq!(
        engine.attribute("voter.mean"),
        Some(Value::Float((10.0 + 99.0 + 12.0) / 3.0))
    );
    assert_eq!(engine.attribute("voter.median"), Some(Value::Float(12.0)));
}
