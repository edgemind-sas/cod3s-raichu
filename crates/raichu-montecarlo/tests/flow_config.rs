//! **The driver honours the engine's flow policy.** `McConfig` carries
//! its own copy of the convergence policy and hands it to two distinct
//! engine-config literals (the estimating replicas and the
//! sequence-recording ones). Both are wired by hand and either one can
//! be wired wrong, and neither failure is visible from the core crate:
//! a policy dropped on the driver's side would leave every replica
//! silently running the engine defaults.
//!
//! Each property is asserted by a *behaviour* that flips against the
//! single-trajectory path on the same model, never by reading a field
//! back.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError, FlowConfig};
use raichu_model::Model;
use raichu_montecarlo::{run, run_sequences, McConfig};

/// One supply, one consumer, one proportional split, and a demand that
/// **collapses when it is served** (`d = 6 - 1.5*a`).
///
/// Undamped the iteration alternates between 6 and 0 forever. Damped at
/// one half it contracts by 1/4 per sweep onto 2.4, which takes about
/// twenty sweeps: inside the default numeric budget of 64, outside a
/// budget of two. That gap is what makes the policy observable from
/// here.
const DAMPED: &str = r#"
{
  "name": "damped",
  "components": [
    {
      "name": "supply",
      "attributes": [
        {"name": "capacity", "kind": "float",
         "init": {"kind": "float", "value": 100.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "capacity",
         "channels": [{"name": "demand"}, {"name": "alloc"}]}
      ],
      "equations": [
        {"target": "out__demand__ea", "kind": "explicit",
         "expr": {"op": "sub",
           "lhs": {"op": "const", "value": {"kind": "float", "value": 6.0}},
           "rhs": {"op": "mul", "args": [
             {"op": "const", "value": {"kind": "float", "value": 1.5}},
             {"op": "attr", "attr": {"component": "supply",
                                     "attribute": "out__alloc__ea"}}]}}}
      ],
      "allocations": [
        {"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
         "available": {"op": "attr",
           "attr": {"component": "supply", "attribute": "capacity"}},
         "policy": "proportional"}
      ]
    },
    {
      "name": "a",
      "attributes": [
        {"name": "got", "kind": "float", "init": {"kind": "float", "value": 0.0}}
      ],
      "ports": [{"name": "input", "dir": "in"}],
      "equations": [
        {"target": "got", "kind": "explicit",
         "expr": {"op": "port_agg", "agg": "sum", "channel": "alloc",
                  "port": {"component": "a", "port": "input"}}}
      ]
    }
  ],
  "connections": [
    {"name": "ea", "from": {"component": "supply", "port": "out"},
     "to": {"component": "a", "port": "input"}}
  ],
  "indicators": [
    {"name": "a_got", "target": "attribute",
     "attr": {"component": "a", "attribute": "got"}}
  ]
}
"#;

fn compiled() -> CompiledModel {
    let document = format!(
        r#"{{"raichu_model": {{"format": 1,
            "requires": ["evaluation_order", "allocation"]}},
            "model": {DAMPED}}}"#
    );
    let model = Model::from_json(&document).expect("model document loads");
    CompiledModel::compile(&model).expect("model compiles")
}

fn mc_config(flow: FlowConfig) -> McConfig {
    McConfig {
        nb_runs: 2,
        seed: 0,
        t_max: 1.0,
        samples: vec![0.0, 1.0],
        threads: Some(1),
        quantiles: Vec::new(),
        ode: raichu_core::SolverParams::default(),
        stop_at_targets: false,
        flow,
    }
}

/// The single-trajectory answer for the same model and policy: the
/// reference the driver has to reproduce.
fn single(model: &CompiledModel, flow: FlowConfig) -> Result<f64, EngineError> {
    let result = Engine::new(
        model,
        EngineConfig {
            t_max: 1.0,
            flow,
            ..EngineConfig::default()
        },
    )?
    .run()?;
    let series = result
        .indicators
        .iter()
        .find(|s| s.name == "a_got")
        .expect("the indicator is recorded");
    match series.points.last().expect("at least one point").1 {
        raichu_expr::Value::Float(f) => Ok(f),
        other => panic!("`a_got` is not a float: {other:?}"),
    }
}

/// The estimating replicas settle on the answer the single trajectory
/// settles on, under the documented default.
#[test]
fn the_estimating_replicas_reproduce_the_single_trajectory_answer() {
    let model = compiled();
    let estimates = run(&model, &mc_config(FlowConfig::default())).expect("the default settles it");
    let mean = estimates
        .indicators
        .iter()
        .find(|i| i.name == "a_got")
        .expect("the indicator is estimated")
        .mean
        .last()
        .copied()
        .expect("at the last sample instant");
    let reference = single(&model, FlowConfig::default()).expect("and so does the trajectory");
    assert!(
        (mean - reference).abs() <= 1e-9,
        "the driver settled on {mean} where the single trajectory settled \
         on {reference}"
    );
}

/// A tightened budget refuses the same network on **both** driver
/// entry points, exactly as it refuses it on the single-trajectory path.
/// Each entry point builds its own engine config, so this is two
/// separate wirings under one assertion.
#[test]
fn both_driver_entry_points_honour_a_tightened_budget() {
    let model = compiled();
    let tightened = FlowConfig {
        sweep_budget: 2,
        ..FlowConfig::default()
    };
    assert!(
        matches!(
            single(&model, tightened.clone()),
            Err(EngineError::FlowNotConverged { .. })
        ),
        "the single-trajectory path no longer refuses a budget of two: \
         the comparison below would be vacuous"
    );
    assert!(
        matches!(
            run(&model, &mc_config(tightened.clone())),
            Err(EngineError::FlowNotConverged { .. })
        ),
        "the estimating replicas ignored the tightened budget"
    );
    let sequences = McConfig {
        stop_at_targets: true,
        ..mc_config(tightened)
    };
    assert!(
        matches!(
            run_sequences(&model, &sequences),
            Err(EngineError::FlowNotConverged { .. })
        ),
        "the sequence-recording replicas ignored the tightened budget: \
         that engine config is wired separately from the estimating one"
    );
}

/// Omitting the policy is the previous behaviour: a config carrying the
/// default and one spelling it out produce identical estimates, byte for
/// byte through their serialization.
#[test]
fn spelling_out_the_default_leaves_the_estimates_identical() {
    let model = compiled();
    let implicit = run(&model, &mc_config(FlowConfig::default())).expect("settles");
    let explicit = run(
        &model,
        &mc_config(FlowConfig {
            sweep_budget: raichu_core::engine::FLOW_SWEEP_BUDGET,
            active_set_budget: None,
            relaxation: raichu_core::engine::FLOW_RELAXATION,
            tolerance: raichu_core::FLOW_TOLERANCE,
        }),
    )
    .expect("settles too");
    assert_eq!(
        implicit, explicit,
        "spelling out the documented policy moved the estimates"
    );
}
