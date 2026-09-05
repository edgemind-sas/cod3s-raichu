//! A guard that reads what its own decision moves, caught before any
//! simulation.
//!
//! The run-time budgets of 0.21.x name such a loop after the fact and
//! after the wait. This is the same finding taken from the compiled
//! tables alone: a cycle automaton to variable to automaton, reported
//! only when some automaton on it switches on a single threshold, since
//! a loop whose every switch has a band is a legitimate oscillation and
//! not a fault.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{switching_loops, CompiledModel};
use raichu_expr::{CmpOp, Expr, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Distrib, Equation, EquationKind, Model, Transition,
};

/// A pump whose start threshold reads a supply its own running consumes.
///
/// `supply` is 10 while the pump is off and 1 while it runs, which is
/// the whole loop: the guard decides the value the guard reads. `release`
/// is the threshold the pump stops at, so passing something other than
/// the start threshold gives the switch a band.
fn pump(start: f64, release: f64) -> Model {
    Model {
        name: "pump".into(),
        components: vec![Component {
            name: "P".into(),
            attributes: vec![Attribute {
                name: "supply".into(),
                kind: AttrKind::Float,
                init: Value::Float(10.0),
            }],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "duty".into(),
                states: vec!["off".into(), "on".into()],
                init: "off".into(),
                transitions: vec![
                    Transition {
                        name: "start".into(),
                        source: "off".into(),
                        guard: Some(Expr::Cmp {
                            cmp: CmpOp::Ge,
                            lhs: Box::new(Expr::attr("P", "supply")),
                            rhs: Box::new(Expr::Const {
                                value: Value::Float(start),
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
                            cmp: CmpOp::Lt,
                            lhs: Box::new(Expr::attr("P", "supply")),
                            rhs: Box::new(Expr::Const {
                                value: Value::Float(release),
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
            // The supply the guard reads is decided by the mode the
            // guard selects: that is the loop, in one equation.
            equations: vec![Equation {
                target: "supply".into(),
                kind: EquationKind::Explicit,
                expr: Expr::If {
                    cond: Box::new(Expr::StateActive {
                        state: raichu_expr::StateRef {
                            component: "P".into(),
                            automaton: "duty".into(),
                            state: "on".into(),
                        },
                    }),
                    then: Box::new(Expr::Const {
                        value: Value::Float(1.0),
                    }),
                    otherwise: Box::new(Expr::Const {
                        value: Value::Float(10.0),
                    }),
                },
            }],
            allocations: vec![],
            sensitive_functions: vec![],
        }],
        connections: vec![],
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
    }
}

#[test]
fn a_single_threshold_on_a_value_the_mode_decides_is_reported() {
    let model = CompiledModel::compile(&pump(4.0, 4.0)).unwrap();
    let loops = switching_loops(&model);
    assert_eq!(loops.len(), 1, "{loops:?}");
    assert_eq!(loops[0].automata, vec!["P.duty".to_string()]);
    assert_eq!(loops[0].bandless, vec!["P.duty".to_string()]);
    assert_eq!(loops[0].through, vec!["P.supply".to_string()]);
}

#[test]
fn the_same_loop_with_a_band_is_not_reported() {
    // The dependency is unchanged and still a cycle. What changed is
    // that crossing the band costs time, so the loop is an oscillation
    // and not a fault, and a diagnostic that fired here would be noise.
    let model = CompiledModel::compile(&pump(4.0, 1.0)).unwrap();
    assert!(switching_loops(&model).is_empty());
}

#[test]
fn the_diagnostic_leads_with_the_culprit_and_the_cure() {
    // It names the automaton to change, sizes the cycle, and says what
    // to do. The cycle itself stays available as a field: a loop through
    // a flow network passes through dozens of attributes, and listing
    // them would bury the one line the reader can act on.
    let model = CompiledModel::compile(&pump(4.0, 4.0)).unwrap();
    let found = &switching_loops(&model)[0];
    let message = found.describe();
    assert!(
        message.starts_with("switching loop: P.duty switches"),
        "{message}"
    );
    assert!(message.contains("band"), "{message}");
    assert_eq!(found.through, vec!["P.supply".to_string()]);
}

#[test]
fn a_model_with_no_loop_at_all_is_silent() {
    // Same automaton, but reading a constant nothing writes: there is no
    // cycle, so nothing to say however the threshold is stated.
    let mut model = pump(4.0, 4.0);
    model.components[0].equations.clear();
    let compiled = CompiledModel::compile(&model).unwrap();
    assert!(switching_loops(&compiled).is_empty());
}

#[test]
fn the_loops_ride_on_the_compiled_model() {
    // Computed once, at compile time, so a caller that wants the
    // diagnosis pays nothing for it and one that never asks is warned
    // through `tracing` anyway. Recomputing on demand would answer the
    // same question twice.
    let model = CompiledModel::compile(&pump(4.0, 4.0)).unwrap();
    assert_eq!(model.switching_loops, switching_loops(&model));
    assert_eq!(model.switching_loops.len(), 1);

    let banded = CompiledModel::compile(&pump(4.0, 1.0)).unwrap();
    assert!(banded.switching_loops.is_empty());
}
