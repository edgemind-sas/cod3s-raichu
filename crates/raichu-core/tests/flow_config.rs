//! **Engine configuration surface of the flow resolution**: the two
//! budgets, the damping and the tolerance of the continuous-flow
//! resolution are settings, not constants compiled into the engine.
//!
//! What is pinned here is the *reach* of each knob, not the policy
//! itself (`flow_convergence.rs` owns that). Three properties:
//!
//! - **The default is the documented policy.** A config built without
//!   touching the flow field carries exactly the figures the constants
//!   declare, so nothing that omits the knobs changes behaviour.
//! - **Each knob reaches the resolution.** Every one is asserted by a
//!   *behaviour* that flips, never by reading the field back: a setting
//!   that is stored and ignored is the failure this file exists to
//!   catch, and a round-trip assertion cannot see it.
//! - **Omitting the knobs changes nothing**, down to the counted work.
//!
//! The Monte-Carlo half of the surface is pinned on the driver's own
//! side (`raichu-montecarlo/tests/flow_config.rs`), which is where its
//! configuration object lives.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::engine::{active_set_budget, FLOW_RELAXATION, FLOW_SWEEP_BUDGET};
use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError, FlowConfig, FLOW_TOLERANCE};
use raichu_expr::Value;
use raichu_model::Model;

// ---------------------------------------------------------------------
// Models. The same shapes `flow_convergence.rs` uses, kept local so a
// change to that file's fixtures cannot silently change what is
// asserted here.
// ---------------------------------------------------------------------

fn sealed(body: &str) -> String {
    format!(
        r#"{{"raichu_model": {{"format": 1,
            "requires": ["evaluation_order", "allocation"]}},
            "model": {body}}}"#
    )
}

fn compile(body: &str) -> CompiledModel {
    let model = Model::from_json(&sealed(body)).expect("model document loads");
    CompiledModel::compile(&model).expect("model compiles")
}

/// The allocated quantity of the single consumer: the per-connection
/// channel attribute the operator writes, which is what makes these
/// networks feed back on themselves.
const ALLOCATED: &str = r#"{"op": "attr",
    "attr": {"component": "supply", "attribute": "out__alloc__ea"}}"#;

/// One supply, one consumer, one proportional split, and a demand the
/// caller writes.
fn one_consumer(name: &str, capacity: f64, demand: &str) -> String {
    format!(
        r#"
{{
  "name": "{name}",
  "components": [
    {{
      "name": "supply",
      "attributes": [
        {{"name": "capacity", "kind": "float",
          "init": {{"kind": "float", "value": {capacity}}}}}
      ],
      "ports": [
        {{"name": "out", "dir": "out", "attr": "capacity",
          "channels": [{{"name": "demand"}}, {{"name": "alloc"}}]}}
      ],
      "equations": [
        {{"target": "out__demand__ea", "kind": "explicit", "expr": {demand}}}
      ],
      "allocations": [
        {{"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
          "available": {{"op": "attr",
            "attr": {{"component": "supply", "attribute": "capacity"}}}},
          "policy": "proportional"}}
      ]
    }},
    {{
      "name": "a",
      "attributes": [
        {{"name": "got", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "ports": [{{"name": "input", "dir": "in"}}],
      "equations": [
        {{"target": "got", "kind": "explicit",
          "expr": {{"op": "port_agg", "agg": "sum", "channel": "alloc",
                    "port": {{"component": "a", "port": "input"}}}}}}
      ]
    }}
  ],
  "connections": [
    {{"name": "ea", "from": {{"component": "supply", "port": "out"}},
      "to": {{"component": "a", "port": "input"}}}}
  ],
  "indicators": [
    {{"name": "a_got", "target": "attribute",
      "attr": {{"component": "a", "attribute": "got"}}}}
  ]
}}
"#
    )
}

/// A demand that **collapses when it is served**: `d = 6 - 1.5*a`.
///
/// Undamped the iteration alternates between 6 and 0 forever. Damped at
/// one half it contracts by 1/4 per sweep onto the fixpoint 2.4, which
/// takes about twenty sweeps to reach the flow tolerance: comfortably
/// inside the default numeric budget of 64, and comfortably outside a
/// budget of two. That gap is what makes the budget knob observable.
fn collapsing_demand() -> String {
    format!(
        r#"{{"op": "sub",
             "lhs": {{"op": "const", "value": {{"kind": "float", "value": 6.0}}}},
             "rhs": {{"op": "mul", "args": [
               {{"op": "const", "value": {{"kind": "float", "value": 1.5}}}},
               {ALLOCATED}]}}}}"#
    )
}

/// A demand that **jumps** when it is served: 9 while the consumer holds
/// at most 3, 1 once it holds more. No fixed point exists, so the
/// resolution always stalls: what a knob can change is *which* budget
/// stops it and after how many sweeps.
fn jumping_demand() -> String {
    format!(
        r#"{{"op": "if",
             "cond": {{"op": "cmp", "cmp": "gt", "lhs": {ALLOCATED},
                      "rhs": {{"op": "const", "value": {{"kind": "float", "value": 3.0}}}}}},
             "then": {{"op": "const", "value": {{"kind": "float", "value": 1.0}}}},
             "otherwise": {{"op": "const", "value": {{"kind": "float", "value": 9.0}}}}}}"#
    )
}

/// A demand that creeps: `d = 0.01 + 0.999*a`, monotone onto 10 at a
/// rate of 0.999 per sweep. Nothing here is combinatorial and nothing
/// cycles: the numeric budget alone stops it, and the number of sweeps
/// it spends before it does is a direct reading of that budget.
fn creeping_demand() -> String {
    format!(
        r#"{{"op": "add", "args": [
             {{"op": "const", "value": {{"kind": "float", "value": 0.01}}}},
             {{"op": "mul", "args": [
               {{"op": "const", "value": {{"kind": "float", "value": 0.999}}}},
               {ALLOCATED}]}}]}}"#
    )
}

fn last_value(result: &raichu_core::SimulationResult, indicator: &str) -> f64 {
    let series = result
        .indicators
        .iter()
        .find(|s| s.name == indicator)
        .unwrap_or_else(|| panic!("indicator `{indicator}` recorded"));
    match series.points.last().expect("at least one point").1 {
        Value::Float(f) => f,
        other => panic!("indicator `{indicator}` is not a float: {other:?}"),
    }
}

fn run(body: &str, flow: FlowConfig) -> Result<raichu_core::SimulationResult, EngineError> {
    let compiled = compile(body);
    Engine::new(
        &compiled,
        EngineConfig {
            t_max: 1.0,
            flow,
            ..EngineConfig::default()
        },
    )?
    .run()
}

// ---------------------------------------------------------------------
// The default is the documented policy.
// ---------------------------------------------------------------------

/// A default engine config carries the figures the convergence-policy
/// constants declare. Pinned against the constants rather than against
/// literals so the two cannot drift apart: the constants remain the
/// single place the policy is stated and argued.
#[test]
fn the_default_flow_config_is_the_documented_policy() {
    let config = EngineConfig::default();
    assert_eq!(
        config.flow.sweep_budget, FLOW_SWEEP_BUDGET,
        "the numeric budget defaults to the constant that documents it"
    );
    assert_eq!(
        config.flow.active_set_budget, None,
        "the combinatorial budget is derived from the compiled network \
         unless a caller overrides it"
    );
    assert_eq!(
        config.flow.relaxation, FLOW_RELAXATION,
        "the damping defaults to the constant that documents it"
    );
    assert_eq!(
        config.flow.tolerance, FLOW_TOLERANCE,
        "the per-edge tolerance defaults to the constant that documents it"
    );
    assert_eq!(
        config.flow,
        FlowConfig::default(),
        "the engine config's flow field is the flow config's own default"
    );
}

// ---------------------------------------------------------------------
// Each knob reaches the resolution.
// ---------------------------------------------------------------------

/// The numeric budget: a damped network that settles under the default
/// budget is refused under a budget of two, and refused with the
/// resolution's own stall diagnostic rather than with a hang or a
/// silently-standing last iterate.
#[test]
fn the_numeric_budget_stops_a_network_the_default_settles() {
    let body = one_consumer("damped", 100.0, &collapsing_demand());

    let settled = run(&body, FlowConfig::default()).expect("the default budget settles it");
    let got = last_value(&settled, "a_got");
    assert!(
        (got - 2.4).abs() <= 1e-8,
        "under the default budget the network settles on 2.4, got {got}"
    );

    let error = run(
        &body,
        FlowConfig {
            sweep_budget: 2,
            ..FlowConfig::default()
        },
    )
    .expect_err("a budget of two cannot cover a contraction of about twenty sweeps");
    let EngineError::FlowNotConverged { ref moving, .. } = error else {
        panic!("expected the flow non-convergence diagnostic, got {error:?}");
    };
    assert_eq!(
        moving, "supply.split[supply.out__alloc__ea]",
        "the shortened budget reports through the same diagnostic, naming \
         the edge that was still moving"
    );
}

/// The numeric budget is *the* thing bounding a monotone stall, so the
/// sweeps it spends move with it. Two budgets, two sweep counts: a knob
/// that were stored and ignored would produce the same count twice.
#[test]
fn the_numeric_budget_sizes_the_work_a_monotone_stall_spends() {
    let body = one_consumer("creeping", 1000.0, &creeping_demand());
    let spent = |budget: usize| -> usize {
        let error = run(
            &body,
            FlowConfig {
                sweep_budget: budget,
                ..FlowConfig::default()
            },
        )
        .expect_err("the creeping network never settles");
        match error {
            EngineError::FlowNotConverged { sweeps, .. } => sweeps,
            other => panic!("expected the flow non-convergence diagnostic, got {other:?}"),
        }
    };
    let (small, large) = (spent(8), spent(40));
    assert!(
        small < large,
        "the sweep count did not follow the budget ({small} at 8, {large} \
         at 40): the setting is not reaching the resolution"
    );
    assert!(
        small <= 8 + 2 && large <= 40 + 2,
        "each run stopped just past its own budget ({small} at 8, {large} \
         at 40)"
    );
}

/// The combinatorial budget: an override replaces the derivation from
/// the compiled network. The network has no fixed point, so the
/// derivation would spend its own (larger) budget; forcing a smaller one
/// stops it sooner, which is only visible because the *number* of sweeps
/// moved.
#[test]
fn the_active_set_budget_override_replaces_the_derivation() {
    let body = one_consumer("oscillating", 5.0, &jumping_demand());
    let derived = active_set_budget(&compile(&body));
    let spent = |budget: Option<usize>| -> usize {
        let error = run(
            &body,
            FlowConfig {
                active_set_budget: budget,
                ..FlowConfig::default()
            },
        )
        .expect_err("the jumping network has no fixed point");
        match error {
            EngineError::FlowNotConverged { sweeps, .. } => sweeps,
            other => panic!("expected the flow non-convergence diagnostic, got {other:?}"),
        }
    };
    let default = spent(None);
    let forced = spent(Some(1));
    assert!(
        forced < default,
        "forcing the combinatorial budget to 1 spent {forced} sweeps \
         against {default} under the derived budget of {derived}: the \
         override is not replacing the derivation"
    );
}

/// The damping: the alternation of `d = 6 - 1.5*a` only settles because
/// under-relaxation is applied at one half. At a weight of one there is
/// no damping left and the same network is refused, which is the state
/// the engine was in before the relaxation was latched.
#[test]
fn the_relaxation_weight_decides_whether_an_alternation_settles() {
    let body = one_consumer("damped", 100.0, &collapsing_demand());
    run(&body, FlowConfig::default()).expect("damped at one half, it settles");
    let error = run(
        &body,
        FlowConfig {
            relaxation: 1.0,
            ..FlowConfig::default()
        },
    )
    .expect_err("undamped, the alternation between 6 and 0 never settles");
    assert!(
        matches!(error, EngineError::FlowNotConverged { .. }),
        "expected the flow non-convergence diagnostic, got {error:?}"
    );
}

/// The tolerance: a monotone sequence creeping at 0.999 per sweep needs
/// about 13 800 sweeps to reach 1e-9 and a couple of dozen to reach
/// 1e-2. The default refuses it on the numeric budget; a tolerance loose
/// enough for the same budget accepts it. Nothing else about the model
/// changes, so only the tolerance can account for the difference.
#[test]
fn the_tolerance_decides_when_the_quantities_have_settled() {
    let body = one_consumer("creeping", 1000.0, &creeping_demand());
    let error = run(&body, FlowConfig::default())
        .expect_err("at 1e-9 the creeping sequence exhausts the numeric budget");
    assert!(
        matches!(error, EngineError::FlowNotConverged { .. }),
        "expected the flow non-convergence diagnostic, got {error:?}"
    );
    let loose = run(
        &body,
        FlowConfig {
            tolerance: 1e-2,
            ..FlowConfig::default()
        },
    )
    .expect("at 1e-2 the same sequence settles inside the same budget");
    let got = last_value(&loose, "a_got");
    assert!(
        got > 0.0,
        "the loosened resolution produced a served quantity, got {got}"
    );
}

// ---------------------------------------------------------------------
// Omitting the knobs reproduces the previous behaviour exactly.
// ---------------------------------------------------------------------

/// A config that never mentions the flow field and one that spells out
/// the documented policy run the same model to the same counted work,
/// down to the sweep. This is the assertion a default gone adrift would
/// break, and counted work is the measure that sees it: an answer can be
/// right while the work behind it moved.
#[test]
fn spelling_out_the_default_changes_nothing() {
    let body = one_consumer("damped", 100.0, &collapsing_demand());
    let implicit = run(&body, FlowConfig::default()).expect("settles");
    let explicit = run(
        &body,
        FlowConfig {
            sweep_budget: FLOW_SWEEP_BUDGET,
            active_set_budget: None,
            relaxation: FLOW_RELAXATION,
            tolerance: FLOW_TOLERANCE,
        },
    )
    .expect("settles too");
    assert_eq!(
        implicit.work, explicit.work,
        "the omitted flow config did not reproduce the documented policy"
    );
    assert_eq!(
        last_value(&implicit, "a_got"),
        last_value(&explicit, "a_got"),
        "the two configurations settled on different answers"
    );
}
