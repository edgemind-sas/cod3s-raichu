//! The engine refuses solver parameters outside their domain when it is
//! built, by name, whatever the entry point. A NaN `max_step` used to be
//! dropped silently by the `min` it entered (no step cap at all), and a
//! zero one surfaced as a step-size underflow that read as stiffness.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError, SolverParams};
use raichu_model::Model;
use serde_json::json;

fn ramp() -> CompiledModel {
    let model: Model = serde_json::from_value(json!({
        "name": "ramp",
        "components": [
            {"name": "c",
             "attributes": [{"name": "x", "kind": "float",
                             "init": {"kind": "float", "value": 0.0}}],
             "equations": [{"target": "x", "kind": "ode",
                            "expr": {"op": "const", "value": {"kind": "float", "value": 1.0}}}]}
        ]
    }))
    .expect("model");
    CompiledModel::compile(&model).unwrap()
}

fn refused_parameter(ode: SolverParams) -> String {
    let compiled = ramp();
    let config = EngineConfig {
        t_max: 1.0,
        ode,
        ..EngineConfig::default()
    };
    match Engine::new(&compiled, config) {
        Err(EngineError::InvalidStudyParameter { parameter, .. }) => parameter,
        Err(other) => panic!("expected InvalidStudyParameter, got {other:?}"),
        Ok(_) => panic!("the engine was built"),
    }
}

#[test]
fn a_nan_step_cap_is_refused_instead_of_dropped() {
    let ode = SolverParams {
        max_step: f64::NAN,
        ..SolverParams::default()
    };
    assert_eq!(refused_parameter(ode), "max_step");
}

#[test]
fn a_zero_step_cap_is_refused_by_name_not_as_stiffness() {
    let ode = SolverParams {
        max_step: 0.0,
        ..SolverParams::default()
    };
    assert_eq!(refused_parameter(ode), "max_step");
}

#[test]
fn a_negative_tolerance_is_refused() {
    let ode = SolverParams {
        rtol: -1e-9,
        ..SolverParams::default()
    };
    assert_eq!(refused_parameter(ode), "rtol");
    let ode = SolverParams {
        atol: -1e-12,
        ..SolverParams::default()
    };
    assert_eq!(refused_parameter(ode), "atol");
}

#[test]
fn an_engine_restored_from_a_snapshot_checks_its_solver_too() {
    let compiled = ramp();
    let engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 1.0,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let snapshot = engine.snapshot().unwrap();
    let config = EngineConfig {
        t_max: 1.0,
        ode: SolverParams {
            tol_event: 0.0,
            ..SolverParams::default()
        },
        ..EngineConfig::default()
    };
    let err = Engine::from_snapshot(&compiled, config, &snapshot)
        .err()
        .expect("refused");
    assert!(
        matches!(&err, EngineError::InvalidStudyParameter { parameter, .. } if parameter == "tol_event"),
        "{err:?}"
    );
}

#[test]
fn an_unbounded_step_cap_runs() {
    let compiled = ramp();
    let config = EngineConfig {
        t_max: 1.0,
        ode: SolverParams {
            max_step: f64::INFINITY,
            ..SolverParams::default()
        },
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    assert_eq!(result.final_time, 1.0);
}
