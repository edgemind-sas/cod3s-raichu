//! A watched boundary reached at an exactly-representable instant must
//! still be located.
//!
//! Regression gate for a silent miss: the event scan brackets a crossing
//! on `prev_margin < 0 && margin >= 0`, so it can only see a margin that
//! turns non-negative *strictly inside* the interval it scans. The margin
//! vector the scan starts from is refreshed at every accepted step, from
//! the committed step solution rather than from the interpolant the scan
//! reads. When a boundary is reached within the round-off separating the
//! two, the refreshed margin is already non-negative: the scan that
//! follows starts from a value that is not `< 0` and, the margin only
//! growing from there, the bracket can never form again. The transition
//! never fired, the trajectory ran past its own boundary, and nothing was
//! reported.
//!
//! The models below are the smallest shape that exhibits it: a constant
//! slope whose crossing lands on a step boundary of the integrator, and a
//! single watched transition, with no scheduled date to end the segment
//! early and hand the crossing to the discrete-fixpoint guard scan.
//!
//! Both orientations are covered on purpose. `le` and `ge` compile to
//! opposite margins (`rhs − lhs` against `lhs − rhs`), and which of the
//! two the interpolant round-off saves is pure luck: the defect showed
//! only on `le`, while `ge` on the very same trajectory was located
//! normally.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::{CmpOp, Expr, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Distrib, Equation, EquationKind, Model, Transition,
};

/// Horizon of every model here: far past every expected crossing, so a
/// missed crossing shows up as "no event at all" rather than a late one.
const T_MAX: f64 = 40.0;

/// `x(0) = x0`, `dx/dt = slope`, one watched transition on `x <cmp> rhs`.
fn ramp_model(x0: f64, slope: f64, cmp: CmpOp, rhs: f64) -> Model {
    Model {
        name: "exact_boundary_ramp".into(),
        components: vec![Component {
            name: "ramp".into(),
            attributes: vec![Attribute {
                name: "x".into(),
                kind: AttrKind::Float,
                init: Value::Float(x0),
            }],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "gate".into(),
                states: vec!["armed".into(), "fired".into()],
                init: "armed".into(),
                transitions: vec![Transition {
                    name: "cross".into(),
                    source: "armed".into(),
                    guard: Some(Expr::Cmp {
                        cmp,
                        lhs: Box::new(Expr::attr("ramp", "x")),
                        rhs: Box::new(Expr::Const {
                            value: Value::Float(rhs),
                        }),
                    }),
                    targets: vec!["fired".into()],
                    on_interruption: Default::default(),
                    monitored: false,
                    cycle_group: None,
                    distrib: Distrib::Watched,
                }],
            }],
            allocations: vec![],
            equations: vec![Equation {
                target: "x".into(),
                kind: EquationKind::Ode,
                expr: Expr::Const {
                    value: Value::Float(slope),
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

/// Firing date of the single watched transition, or `None` if it never
/// fired within the horizon.
fn crossing_date(x0: f64, slope: f64, cmp: CmpOp, rhs: f64) -> Option<f64> {
    let model = ramp_model(x0, slope, cmp, rhs);
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max: T_MAX,
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    assert!(
        result.events.len() <= 1,
        "one-shot model fired more than once: {:?}",
        result.events
    );
    result.events.first().map(|event| event.time)
}

/// Located dates carry the integrator's event tolerance (`tol_event`,
/// 1e-10), so they are compared against the analytic instant at that
/// scale, never for equality.
fn assert_located(got: Option<f64>, expected: f64, what: &str) {
    let Some(time) = got else {
        panic!("{what}: the boundary was never located (expected t = {expected})");
    };
    assert!(
        (time - expected).abs() < 1e-9,
        "{what}: located at t = {time}, expected {expected}"
    );
}

/// `x <= 0` reached at t = 10 exactly, the case that silently never fired.
#[test]
fn non_strict_le_locates_an_exactly_representable_crossing() {
    assert_located(
        crossing_date(50.0, -5.0, CmpOp::Le, 0.0),
        10.0,
        "x <= 0 from 50 at slope -5",
    );
}

/// Same defect away from zero: a non-zero right-hand side is not what
/// saves the margin, so `x <= 10` reached at t = 8 must locate too.
#[test]
fn non_strict_le_locates_an_exact_crossing_of_a_non_zero_boundary() {
    assert_located(
        crossing_date(50.0, -5.0, CmpOp::Le, 10.0),
        8.0,
        "x <= 10 from 50 at slope -5",
    );
}

/// The opposite orientation on the mirrored trajectory: `ge` compiles to
/// the opposite margin and must locate the same instant.
#[test]
fn non_strict_ge_locates_an_exactly_representable_crossing() {
    assert_located(
        crossing_date(0.0, 5.0, CmpOp::Ge, 50.0),
        10.0,
        "x >= 50 from 0 at slope +5",
    );
}

/// Inexact crossings were never affected; they are pinned so a future
/// change to the scan cannot trade one case for the other.
#[test]
fn inexact_crossings_stay_located() {
    assert_located(
        crossing_date(50.0, -3.0, CmpOp::Le, 0.0),
        50.0 / 3.0,
        "x <= 0 from 50 at slope -3",
    );
    assert_located(
        crossing_date(0.0, 3.0, CmpOp::Ge, 50.0),
        50.0 / 3.0,
        "x >= 50 from 0 at slope +3",
    );
}

/// A *strict* guard keeps its inward shift (`STRICT_MARGIN_EPS`): it
/// fires just past its boundary, never on it. Reaching the boundary at an
/// exactly-representable instant must not turn that into a firing at the
/// boundary itself.
#[test]
fn strict_guard_fires_past_an_exactly_representable_boundary() {
    let time = crossing_date(50.0, -5.0, CmpOp::Lt, 0.0)
        .expect("a strict guard on a crossing trajectory must still fire");
    assert!(
        time > 10.0,
        "x < 0 fired at t = {time}, not strictly past the boundary instant 10.0"
    );
    assert!(
        time - 10.0 < 1e-9,
        "x < 0 fired at t = {time}, too far past the boundary instant 10.0"
    );
}

/// The semantic the whole fix must not blur: a trajectory *resting*
/// exactly on a boundary satisfies a non-strict guard and not a strict
/// one. With a zero slope the margin sits at exactly 0.0 for the whole
/// horizon.
#[test]
fn a_trajectory_resting_on_its_boundary_fires_only_the_non_strict_guard() {
    assert_eq!(
        crossing_date(0.0, 0.0, CmpOp::Le, 0.0),
        Some(0.0),
        "x <= 0 with x resting at 0 must fire"
    );
    assert_eq!(
        crossing_date(0.0, 0.0, CmpOp::Ge, 0.0),
        Some(0.0),
        "x >= 0 with x resting at 0 must fire"
    );
    assert_eq!(
        crossing_date(0.0, 0.0, CmpOp::Lt, 0.0),
        None,
        "x < 0 with x resting at 0 must not fire"
    );
    assert_eq!(
        crossing_date(0.0, 0.0, CmpOp::Gt, 0.0),
        None,
        "x > 0 with x resting at 0 must not fire"
    );
}
