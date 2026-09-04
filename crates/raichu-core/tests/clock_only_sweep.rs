//! A model whose only time dependence is an explicit equation over the
//! clock, with no ODE attribute and no armed hazard.
//!
//! The engine decides whether continuous evolution runs at all, and it
//! used to decide it on the ODE attributes and the continuously-varying
//! hazards alone. An explicit equation over `time` is neither: it varies
//! with nothing behind it. A model carrying one and nothing else found
//! nothing to advance, evaluated its sweep once at the initial instant
//! and reported that value at every sample instant for the rest of the
//! run, and to every watched guard: a curve reported as a constant, with
//! nothing to signal it.
//!
//! Both halves are pinned here, because the second is the one that
//! silently changes an answer rather than a recording: a boundary a
//! guard watches on such a quantity was never crossed at all.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::{AttrRef, CmpOp, Expr, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Distrib, Equation, EquationKind, Indicator,
    IndicatorTarget, Model, Transition,
};

/// The instant the ramp reaches the watched boundary.
const CROSSING: f64 = 2.5;

/// `ramp = time`: the simplest equation that reads the clock and nothing
/// else, so a failure cannot be blamed on the rest of the expression.
fn clock_model(watched: bool) -> Model {
    let mut component = Component {
        name: "C".into(),
        attributes: vec![Attribute {
            name: "ramp".into(),
            kind: AttrKind::Float,
            init: Value::Float(0.0),
        }],
        ports: vec![],
        interfaces: vec![],
        automata: vec![],
        allocations: vec![],
        equations: vec![Equation {
            target: "ramp".into(),
            kind: EquationKind::Explicit,
            expr: Expr::Time,
        }],
        sensitive_functions: vec![],
    };
    if watched {
        // Fires when the ramp passes 2.5, which is the instant t = 2.5
        // and nothing else. Read against a ramp frozen at 0, the guard
        // never holds and the transition never fires at all.
        component.automata = vec![Automaton {
            name: "alarm".into(),
            states: vec!["quiet".into(), "tripped".into()],
            init: "quiet".into(),
            transitions: vec![Transition {
                name: "trip".into(),
                source: "quiet".into(),
                guard: Some(Expr::Cmp {
                    cmp: CmpOp::Ge,
                    lhs: Box::new(Expr::attr("C", "ramp")),
                    rhs: Box::new(Expr::Const {
                        value: Value::Float(CROSSING),
                    }),
                }),
                targets: vec!["tripped".into()],
                on_interruption: Default::default(),
                monitored: false,
                cycle_group: None,
                distrib: Distrib::Watched,
            }],
        }];
    }
    Model {
        name: "clock_only".into(),
        components: vec![component],
        indicators: vec![Indicator {
            name: "ramp".into(),
            target: IndicatorTarget::Attribute {
                attr: AttrRef {
                    component: "C".into(),
                    attribute: "ramp".into(),
                },
            },
        }],
        connections: vec![],
        targets: vec![],
        evaluation_order: None,
    }
}

fn run(model: &Model, t_max: f64, samples: Vec<f64>) -> raichu_core::SimulationResult {
    let compiled = CompiledModel::compile(model).unwrap();
    let config = EngineConfig {
        t_max,
        samples,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config).unwrap().run().unwrap()
}

#[test]
fn an_explicit_equation_over_the_clock_is_swept_as_time_advances() {
    let result = run(&clock_model(false), 10.0, vec![0.0, 2.5, 7.5, 10.0]);
    let sampled: Vec<(f64, f64)> = result.samples[0]
        .points
        .iter()
        .map(|(t, v)| match v {
            Value::Float(f) => (*t, *f),
            other => panic!("unexpected {other:?}"),
        })
        .collect();
    assert_eq!(sampled.len(), 4, "{sampled:?}");
    for (instant, value) in sampled {
        let error: f64 = value - instant;
        assert!(error.abs() < 1e-9, "ramp at t={instant} read {value}");
    }
}

#[test]
fn a_watched_guard_on_a_clock_reading_is_located() {
    let result = run(&clock_model(true), 10.0, vec![]);
    assert_eq!(result.events.len(), 1, "{:?}", result.events);
    let error: f64 = result.events[0].time - CROSSING;
    assert!(error.abs() < 1e-6, "fired at {}", result.events[0].time);
}

#[test]
fn the_compiled_model_says_whether_its_sweep_reads_the_clock() {
    // Pinned on the flag as well as on the behaviour: the flag is what
    // the engine branches on, and a model reading no clock must keep the
    // event-only path it had.
    assert!(
        CompiledModel::compile(&clock_model(false))
            .unwrap()
            .explicit_reads_time
    );

    let mut still = clock_model(false);
    still.components[0].equations[0].expr = Expr::Const {
        value: Value::Float(1.0),
    };
    assert!(!CompiledModel::compile(&still).unwrap().explicit_reads_time);
}
