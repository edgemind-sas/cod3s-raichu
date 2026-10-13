//! Declared unbounded magnitude: no integrated rate may reach it.
//!
//! A document has no literal for an infinity, so an authoring layer
//! writes "no ceiling" as a finite stand-in. As the ceiling of a `min`
//! against a finite demand it is harmless; as the derivative of a stock
//! it is not: a stock of 5 drained at 1e30 is stepped to -9.3e19 before
//! its empty bound can be located. These tests pin the contract that
//! turns that silent number into a typed refusal.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError};
use raichu_model::{Model, ModelError};

/// A stock `x` of 5 drained at the constant `rate`: `dx/dt = -rate`.
/// `unbounded` is the model-level fragment (empty for none).
fn drain_body(rate: f64, unbounded: &str) -> String {
    format!(
        r#"{{
  {unbounded}
  "name": "unbounded_rate",
  "components": [{{
    "name": "C",
    "attributes": [
      {{"name": "x", "kind": "float", "init": {{"kind": "float", "value": 5.0}}}},
      {{"name": "rate", "kind": "float", "init": {{"kind": "float", "value": {rate:e}}}}}
    ],
    "equations": [
      {{"target": "x", "kind": "ode",
       "expr": {{"op": "sub",
          "lhs": {{"op": "const", "value": {{"kind": "float", "value": 0.0}}}},
          "rhs": {{"op": "attr", "attr": {{"component": "C", "attribute": "rate"}}}}}}}}
    ]
  }}],
  "indicators": [
    {{"name": "x", "target": "attribute", "attr": {{"component": "C", "attribute": "x"}}}}
  ]
}}"#
    )
}

fn sealed(body: &str) -> String {
    format!(
        r#"{{"raichu_model": {{"format": 1, "requires": ["unbounded_rate"]}}, "model": {body}}}"#
    )
}

fn run(json: &str) -> Result<raichu_core::SimulationResult, EngineError> {
    let model = Model::from_json(json).expect("fixture parses");
    let compiled = CompiledModel::compile(&model).expect("model compiles");
    let config = EngineConfig {
        t_max: 1.0,
        samples: vec![1.0],
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config).expect("engine builds").run()
}

fn final_x(json: &str) -> f64 {
    let result = run(json).expect("simulation runs");
    let (time, value) = result.samples[0].points.last().expect("a sample");
    assert_eq!(*time, 1.0);
    let raichu_expr::Value::Float(value) = value else {
        panic!("float indicator");
    };
    *value
}

#[test]
fn a_rate_at_the_declared_unbounded_magnitude_is_refused_by_name() {
    let error = run(&sealed(&drain_body(1e30, r#""unbounded_rate": 1e30,"#)))
        .expect_err("an unbounded rate must not be integrated");
    let EngineError::UnboundedRate {
        variable,
        rate,
        unbounded,
        time,
    } = error
    else {
        panic!("expected UnboundedRate, got {error:?}");
    };
    assert_eq!(variable, "C.x");
    assert_eq!(rate, -1e30);
    assert_eq!(unbounded, 1e30);
    assert_eq!(time, 0.0);
}

#[test]
fn a_finite_rate_under_the_declaration_integrates_as_before() {
    let x = final_x(&sealed(&drain_body(2.0, r#""unbounded_rate": 1e30,"#)));
    assert!((x - 3.0).abs() < 1e-9, "5 - 2 x 1 = 3, got {x}");
}

#[test]
fn without_a_declaration_nothing_is_reserved() {
    // The check is the model's own contract: a model that reserves no
    // magnitude is integrated as written, whatever it says.
    let x = final_x(&drain_body(1e30, ""));
    assert!(x < -1e20, "integrated as written, got {x}");
}

#[test]
fn the_declaration_is_a_feature_a_bare_body_cannot_carry() {
    assert!(Model::from_json(&drain_body(2.0, r#""unbounded_rate": 1e30,"#)).is_err());
    let model = Model::from_json(&sealed(&drain_body(2.0, r#""unbounded_rate": 1e30,"#))).unwrap();
    assert_eq!(model.unbounded_rate, Some(1e30));
}

#[test]
fn a_declared_magnitude_must_be_finite_and_positive() {
    for value in ["0.0", "-1.0"] {
        let fragment = format!(r#""unbounded_rate": {value},"#);
        let model: Model = serde_json::from_str(&drain_body(2.0, &fragment)).unwrap();
        assert!(matches!(
            model.validate(),
            Err(ModelError::UnboundedRateInvalid { .. })
        ));
    }
}

/// A stock `x` of 2, fed at 1 and drained through a mixture share:
/// `dx/dt = 1 - u * x / (x + 100)`, `u` an attribute of value `u`.
///
/// The shape of a volume holding a mixture whose ceiling is left at the
/// stand-in: what leaves it is the stand-in scaled by the share of the
/// drawn constituent, so the rate sits orders of magnitude below the
/// stand-in while being set by it.
fn mixture_drain_body(u: f64) -> String {
    format!(
        r#"{{
  "unbounded_rate": 1e30,
  "name": "mixture_drain",
  "components": [{{
    "name": "C",
    "attributes": [
      {{"name": "x", "kind": "float", "init": {{"kind": "float", "value": 2.0}}}},
      {{"name": "u", "kind": "float", "init": {{"kind": "float", "value": {u:e}}}}}
    ],
    "equations": [
      {{"target": "x", "kind": "ode",
       "expr": {{"op": "sub",
          "lhs": {{"op": "const", "value": {{"kind": "float", "value": 1.0}}}},
          "rhs": {{"op": "div",
             "lhs": {{"op": "mul", "args": [
                {{"op": "attr", "attr": {{"component": "C", "attribute": "u"}}}},
                {{"op": "attr", "attr": {{"component": "C", "attribute": "x"}}}}]}},
             "rhs": {{"op": "add", "args": [
                {{"op": "attr", "attr": {{"component": "C", "attribute": "x"}}}},
                {{"op": "const", "value": {{"kind": "float", "value": 100.0}}}}]}}}}}}}}
    ]
  }}],
  "indicators": [
    {{"name": "x", "target": "attribute", "attr": {{"component": "C", "attribute": "x"}}}}
  ]
}}"#
    )
}

#[test]
fn a_rate_the_stand_in_sets_below_its_magnitude_is_refused_not_reported_as_stiffness() {
    // 1e30 x 2 / 102 = 1.96e28: under the declared magnitude, so the
    // right-hand side check lets it through and the step controller
    // collapses on it. That collapse is the same refusal, and is reported
    // as one.
    let error = run(&sealed(&mixture_drain_body(1e30)))
        .expect_err("a rate set by the stand-in must not be integrated");
    let EngineError::UnboundedRate {
        variable,
        rate,
        unbounded,
        time,
    } = error
    else {
        panic!("expected UnboundedRate, got {error:?}");
    };
    assert_eq!(variable, "C.x");
    assert_eq!(unbounded, 1e30);
    assert_eq!(time, 0.0);
    assert!(rate < -1e28 && rate > -1e30, "the diluted rate, got {rate}");
}

#[test]
fn genuine_stiffness_stays_a_solver_error() {
    // Same shape, a large but declared coefficient: the rate does not
    // move with the stand-in, so the failure is the solver's to report.
    let error = run(&sealed(&mixture_drain_body(1e20))).expect_err("too stiff");
    assert!(
        matches!(
            error,
            EngineError::Ode(raichu_numeric::OdeError::StepSizeUnderflow { .. })
        ),
        "expected the solver's underflow, got {error:?}"
    );
}

#[test]
fn the_stand_in_written_as_a_literal_is_found_too() {
    // The authoring layer also writes the stand-in inline (a fill claim
    // with no ceiling), not only as an attribute's value.
    let body = mixture_drain_body(1e30).replace(
        r#"{"op": "attr", "attr": {"component": "C", "attribute": "u"}}"#,
        r#"{"op": "const", "value": {"kind": "float", "value": 1e30}}"#,
    );
    assert!(
        body.contains(r#""value": 1e30}}"#),
        "the literal replaced the read"
    );
    let error = run(&sealed(&body)).expect_err("refused");
    assert!(
        matches!(error, EngineError::UnboundedRate { ref variable, .. } if variable == "C.x"),
        "expected UnboundedRate on C.x, got {error:?}"
    );
}
