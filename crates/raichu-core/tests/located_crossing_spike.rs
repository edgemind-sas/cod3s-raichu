//! Entry-spike gate for the continuous-flow layer.
//!
//! The continuous-flow plan hoists the combinatorial active-set search to
//! segment boundaries and freezes it during integration, then emits the
//! resolved flows as *explicit* equations so a margin reading one still
//! moves with the state. That only works if the engine locates a boundary
//! crossing whose margin reads a value produced by the explicit pass,
//! rather than noticing it at the next scheduled discrete date.
//!
//! `ContinuousSystem::load` runs `recompute_explicit` before every `rhs`
//! and `events` callback, so an explicit variable is a genuine function of
//! `(t, x)` inside a segment. These tests pin that behaviour, because a
//! silent regression to polled boundaries would leave the flow layer
//! reporting crossings at the wrong instant with no failing assertion.
//!
//! Both models are deliberately minimal and carry a far-away scheduled
//! transition (`t = 100`), so a polled implementation would report the
//! crossing at that date instead of the analytic one.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::{AggOp, AttrRef, CmpOp, Expr, PortRef, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Connection, Distrib, Equation, EquationKind,
    Indicator, IndicatorTarget, Model, Port, PortDir, Transition,
};

/// Analytic crossing instant of both models: `y = 2x`, `dx/dt = 1`,
/// `x(0) = 0`, boundary `y >= 5`.
const EXPECTED_CROSSING: f64 = 2.5;

/// The far-away scheduled date a polled implementation would report instead.
const FAR_SCHEDULED_DATE: f64 = 100.0;

fn run(model: &Model, t_max: f64) -> raichu_core::SimulationResult {
    let compiled = CompiledModel::compile(model).unwrap();
    let config = EngineConfig {
        t_max,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config).unwrap().run().unwrap()
}

/// A two-state automaton whose only transition is a delay far past the
/// horizon, so the schedule holds no date near the boundary crossing.
fn far_scheduled_automaton() -> Automaton {
    Automaton {
        name: "far".into(),
        states: vec!["a".into(), "b".into()],
        init: "a".into(),
        transitions: vec![Transition {
            name: "later".into(),
            source: "a".into(),
            guard: None,
            targets: vec!["b".into()],
            on_interruption: Default::default(),
            monitored: false,
            cycle_group: None,
            distrib: Distrib::Delay {
                time: FAR_SCHEDULED_DATE,
            },
        }],
    }
}

/// A watched transition firing when `y` reaches 5.
fn boundary_automaton(component: &str) -> Automaton {
    Automaton {
        name: "gate".into(),
        states: vec!["low".into(), "high".into()],
        init: "low".into(),
        transitions: vec![Transition {
            name: "cross".into(),
            source: "low".into(),
            guard: Some(Expr::Cmp {
                cmp: CmpOp::Ge,
                lhs: Box::new(Expr::attr(component, "y")),
                rhs: Box::new(Expr::Const {
                    value: Value::Float(5.0),
                }),
            }),
            targets: vec!["high".into()],
            on_interruption: Default::default(),
            monitored: false,
            cycle_group: None,
            distrib: Distrib::Watched,
        }],
    }
}

fn float_attr(name: &str) -> Attribute {
    Attribute {
        name: name.into(),
        kind: AttrKind::Float,
        init: Value::Float(0.0),
    }
}

/// Case A: the margin reads an explicit variable of the same component.
///
/// `x` integrates at rate 1; `y = 2x` is explicit; the boundary is `y >= 5`.
fn same_component_model() -> Model {
    Model {
        name: "spike_explicit_same_component".into(),
        components: vec![Component {
            name: "src".into(),
            attributes: vec![float_attr("x"), float_attr("y")],
            ports: vec![],
            interfaces: vec![],
            automata: vec![boundary_automaton("src"), far_scheduled_automaton()],
            allocations: vec![],
            equations: vec![
                Equation {
                    target: "x".into(),
                    kind: EquationKind::Ode,
                    expr: Expr::Const {
                        value: Value::Float(1.0),
                    },
                },
                Equation {
                    target: "y".into(),
                    kind: EquationKind::Explicit,
                    expr: Expr::Mul {
                        args: vec![
                            Expr::Const {
                                value: Value::Float(2.0),
                            },
                            Expr::attr("src", "x"),
                        ],
                    },
                },
            ],
            sensitive_functions: vec![],
        }],
        connections: vec![],
        indicators: vec![Indicator {
            name: "y".into(),
            target: IndicatorTarget::Attribute {
                attr: AttrRef {
                    component: "src".into(),
                    attribute: "y".into(),
                },
            },
        }],
        targets: vec![],
        evaluation_order: None,
    }
}

/// Case B: the margin reads an explicit variable computed from a port
/// aggregation, which is the shape a resolved continuous flow takes.
///
/// `src.x` integrates at rate 1 and is exported; `sink.y = 2 * sum(in)`
/// is explicit; the boundary is `sink.y >= 5`.
fn across_connection_model() -> Model {
    Model {
        name: "spike_explicit_across_connection".into(),
        components: vec![
            Component {
                name: "src".into(),
                attributes: vec![float_attr("x")],
                ports: vec![Port {
                    name: "out".into(),
                    dir: PortDir::Out,
                    attr: Some("x".into()),
                    channels: vec![],
                }],
                interfaces: vec![],
                automata: vec![],
                allocations: vec![],
                equations: vec![Equation {
                    target: "x".into(),
                    kind: EquationKind::Ode,
                    expr: Expr::Const {
                        value: Value::Float(1.0),
                    },
                }],
                sensitive_functions: vec![],
            },
            Component {
                name: "sink".into(),
                attributes: vec![float_attr("y")],
                ports: vec![Port {
                    name: "input".into(),
                    dir: PortDir::In,
                    attr: None,
                    channels: vec![],
                }],
                interfaces: vec![],
                automata: vec![boundary_automaton("sink"), far_scheduled_automaton()],
                allocations: vec![],
                equations: vec![Equation {
                    target: "y".into(),
                    kind: EquationKind::Explicit,
                    expr: Expr::Mul {
                        args: vec![
                            Expr::Const {
                                value: Value::Float(2.0),
                            },
                            Expr::PortAgg {
                                port: PortRef {
                                    component: "sink".into(),
                                    port: "input".into(),
                                },
                                agg: AggOp::Sum,
                                channel: None,
                            },
                        ],
                    },
                }],
                sensitive_functions: vec![],
            },
        ],
        connections: vec![Connection {
            from: PortRef {
                component: "src".into(),
                port: "out".into(),
            },
            to: PortRef {
                component: "sink".into(),
                port: "input".into(),
            },
            name: None,
        }],
        indicators: vec![Indicator {
            name: "y".into(),
            target: IndicatorTarget::Attribute {
                attr: AttrRef {
                    component: "sink".into(),
                    attribute: "y".into(),
                },
            },
        }],
        targets: vec![],
        evaluation_order: None,
    }
}

/// The instant the `cross` transition was reported, if it fired at all.
fn crossing_instant(result: &raichu_core::SimulationResult) -> Option<f64> {
    result
        .events
        .iter()
        .find(|event| event.transition.split('.').next_back() == Some("cross"))
        .map(|event| event.time)
}

#[test]
fn explicit_variable_boundary_is_located_not_polled() {
    let result = run(&same_component_model(), 10.0);
    let instant = crossing_instant(&result).unwrap_or_else(|| {
        panic!(
            "the boundary transition never fired; events were {:?}",
            result.events
        )
    });
    assert!(
        (instant - EXPECTED_CROSSING).abs() < 1e-6,
        "boundary crossing reported at t={instant}, expected the analytic instant \
         t={EXPECTED_CROSSING}. A value near t={FAR_SCHEDULED_DATE} or at the horizon \
         means the margin was polled at the next scheduled date rather than located."
    );
}

#[test]
fn boundary_across_a_connection_is_located_not_polled() {
    let result = run(&across_connection_model(), 10.0);
    let instant = crossing_instant(&result).unwrap_or_else(|| {
        panic!(
            "the boundary transition never fired; events were {:?}",
            result.events
        )
    });
    assert!(
        (instant - EXPECTED_CROSSING).abs() < 1e-6,
        "boundary crossing reported at t={instant}, expected the analytic instant \
         t={EXPECTED_CROSSING}. A value near t={FAR_SCHEDULED_DATE} or at the horizon \
         means the margin was polled at the next scheduled date rather than located."
    );
}
