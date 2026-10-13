//! **A slow geometric flow resolution is carried to its limit**, at the
//! segment boundary and inside the right-hand side alike.
//!
//! Under a held active set the quantities of a torn sweep follow an affine
//! map, and its rate is a property of the state, not of the network: a
//! full volume that admits only what leaves it, holding a share `s` of
//! what it admits, feeds back `1 − s` of its intake at every pass (issue
//! #160, a room holding 10 % of hydrogen, contracting at 0.9 and refused
//! after 83 sweeps). No constant budget holds a rate arbitrarily close to
//! one, so the resolution measures the rate and jumps to the limit
//! (Aitken's Δ² on the dominant mode).
//!
//! Pinned here:
//!
//! - **The boundary resolution settles a slow creep on its exact limit**,
//!   in a handful of sweeps, where the plain iteration needed thousands.
//! - **The right-hand side does too.** A torn ring repeated inside one
//!   evaluation used to keep its last pass after the budget, silently: at
//!   0.999 per pass, 64 passes leave 94 % of each stage's move unresolved,
//!   and the integral of the flow comes out wrong with no diagnostic.
//! - **A jump never crosses a breakpoint.** On a map that is smooth
//!   rather than affine the jump can overshoot, and past a conditional or
//!   a capping it could settle the resolution on another equilibrium; it
//!   is taken only as far as it keeps the active set it was measured on.
//! - **What is not geometric is not extrapolated**: a drift (rate exactly
//!   one) still exhausts the budget and is reported as such
//!   (`flow_convergence.rs`, `flow_config.rs`).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, SimulationResult};
use raichu_expr::Value;
use raichu_model::Model;

/// The allocated quantity of the consumer: the channel attribute the
/// operator writes, read back by the demand that sizes it.
const ALLOCATED: &str = r#"{"op": "attr",
    "attr": {"component": "supply", "attribute": "out__alloc__ea"}}"#;

/// One supply of capacity 1e6 and one consumer whose demand reads what it
/// was allocated: `d = c·x + 0.999·a`, where `x` is a clock (`x' = 1`).
///
/// Its fixed point is `a = 1000·c·x`: the creep `a ← c·x + 0.999·a`
/// contracts at 0.999 per sweep, about 20 000 sweeps to the flow
/// tolerance from a cold start. The consumer integrates what it receives
/// (`z' = a`), so `z(t) = 500·c·t²` exactly when every evaluation of the
/// right-hand side reads the settled flow.
///
/// `c = 0` freezes the fixed point at the constant `a = 10` reached by
/// `d = 0.01 + 0.999·a` (the `offset`), the case the boundary resolution
/// alone has to settle.
fn creep(clock_gain: f64, offset: f64) -> String {
    creep_capped(clock_gain, offset, 1_000_000.0)
}

/// [`creep`] on a supply of capacity `capacity`: the operator caps the
/// allocation there, a breakpoint of the active set.
fn creep_capped(clock_gain: f64, offset: f64, capacity: f64) -> String {
    let body = format!(
        r#"
{{
  "name": "creep",
  "components": [
    {{
      "name": "supply",
      "attributes": [
        {{"name": "capacity", "kind": "float",
          "init": {{"kind": "float", "value": {capacity:?}}}}},
        {{"name": "x", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "ports": [
        {{"name": "out", "dir": "out", "attr": "capacity",
          "channels": [{{"name": "demand"}}, {{"name": "alloc"}}]}}
      ],
      "equations": [
        {{"target": "x", "kind": "ode",
          "expr": {{"op": "const", "value": {{"kind": "float", "value": 1.0}}}}}},
        {{"target": "out__demand__ea", "kind": "explicit",
          "expr": {{"op": "add", "args": [
            {{"op": "const", "value": {{"kind": "float", "value": {offset}}}}},
            {{"op": "mul", "args": [
              {{"op": "const", "value": {{"kind": "float", "value": {clock_gain}}}}},
              {{"op": "attr", "attr": {{"component": "supply", "attribute": "x"}}}}]}},
            {{"op": "mul", "args": [
              {{"op": "const", "value": {{"kind": "float", "value": 0.999}}}},
              {ALLOCATED}]}}]}}}}
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
        {{"name": "got", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}},
        {{"name": "z", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "ports": [{{"name": "input", "dir": "in"}}],
      "equations": [
        {{"target": "got", "kind": "explicit",
          "expr": {{"op": "port_agg", "agg": "sum", "channel": "alloc",
                    "port": {{"component": "a", "port": "input"}}}}}},
        {{"target": "z", "kind": "ode",
          "expr": {{"op": "attr", "attr": {{"component": "a", "attribute": "got"}}}}}}
      ]
    }}
  ],
  "connections": [
    {{"name": "ea", "from": {{"component": "supply", "port": "out"}},
      "to": {{"component": "a", "port": "input"}}}}
  ],
  "indicators": [
    {{"name": "got", "target": "attribute",
      "attr": {{"component": "a", "attribute": "got"}}}},
    {{"name": "z", "target": "attribute",
      "attr": {{"component": "a", "attribute": "z"}}}}
  ]
}}
"#
    );
    format!(
        r#"{{"raichu_model": {{"format": 1,
            "requires": ["evaluation_order", "allocation"]}},
            "model": {body}}}"#
    )
}

fn run(document: &str, t_max: f64, samples: Vec<f64>) -> SimulationResult {
    let model = Model::from_json(document).expect("model document loads");
    let compiled = CompiledModel::compile(&model).expect("model compiles");
    Engine::new(
        &compiled,
        EngineConfig {
            t_max,
            samples,
            ..EngineConfig::default()
        },
    )
    .expect("the initial resolution settles")
    .run()
    .expect("the run completes")
}

fn sampled(result: &SimulationResult, index: usize) -> Vec<(f64, f64)> {
    result.samples[index]
        .points
        .iter()
        .map(|(t, value)| match value {
            Value::Float(f) => (*t, *f),
            other => panic!("indicator {index} is not a float: {other:?}"),
        })
        .collect()
}

/// The creep onto 10 settles **on 10**, and in a handful of sweeps: the
/// plain iteration needs about 20 000 to reach the tolerance, which no
/// budget holds, and before the extrapolation this network was refused.
#[test]
fn a_slow_creep_settles_on_its_limit_at_the_boundary() {
    let result = run(&creep(0.0, 0.01), 1.0, vec![1.0]);
    let got = sampled(&result, 0);
    let (_, value) = got[0];
    assert!(
        (value - 10.0).abs() <= 1e-8,
        "the resolution settled on {value}, not on the limit 0.01/0.001 = 10"
    );
    let sweeps = result.work.flow_sweeps;
    assert!(
        sweeps <= 16,
        "{sweeps} sweeps: the creep was iterated rather than extrapolated"
    );
}

/// Inside the right-hand side the fixed point moves with the clock, so
/// every stage starts a fresh creep from the previous stage's answer. The
/// flow must read `1000·c·x` at every sample, and its integral must be the
/// closed form `500·c·t²`: a ring left unsettled at each stage would lag
/// the clock and integrate visibly less.
#[test]
fn a_slow_creep_settles_inside_the_right_hand_side() {
    let gain = 0.001;
    let result = run(&creep(gain, 0.0), 2.0, vec![0.5, 1.0, 1.5, 2.0]);
    for (t, value) in sampled(&result, 0) {
        let expected = 1000.0 * gain * t;
        assert!(
            (value - expected).abs() <= 1e-6 * expected.max(1.0),
            "at t={t} the flow reads {value}, not the settled {expected}"
        );
    }
    for (t, value) in sampled(&result, 1) {
        let expected = 500.0 * gain * t * t;
        assert!(
            (value - expected).abs() <= 1e-6 * expected.max(1.0),
            "at t={t} the integral reads {value}, not the closed form {expected}"
        );
    }
}

/// One supply of capacity 1000 and one consumer whose demand is
/// `demand`, an expression of the allocated quantity: the smallest torn
/// ring, for maps the [`creep`] fixture cannot write.
fn ring(demand: &str) -> String {
    let body = format!(
        r#"
{{
  "name": "ring",
  "components": [
    {{
      "name": "supply",
      "attributes": [
        {{"name": "capacity", "kind": "float",
          "init": {{"kind": "float", "value": 1000.0}}}}
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
    {{"name": "got", "target": "attribute",
      "attr": {{"component": "a", "attribute": "got"}}}}
  ]
}}
"#
    );
    format!(
        r#"{{"raichu_model": {{"format": 1,
            "requires": ["evaluation_order", "allocation"]}},
            "model": {body}}}"#
    )
}

fn constant(value: f64) -> String {
    format!(r#"{{"op": "const", "value": {{"kind": "float", "value": {value:?}}}}}"#)
}

/// `d = 0.1 + 0.99·a − 0.0001·a²` below 9.5, and 20 above it: two
/// equilibria, 9.1608 on the near branch and 20 on the far one.
///
/// From the cold start the plain sequence climbs to 9.1608 and never
/// reaches the threshold. The three-step test agrees long before the
/// jump is accurate on this curved map, and the full jump lands at 9.97
/// (measured in review): past the threshold, where the next sweep would
/// have settled the resolution on 20. The jump must be held to the branch
/// it was measured on.
#[test]
fn a_jump_does_not_carry_the_resolution_onto_another_branch() {
    let allocated = r#"{"op": "attr",
        "attr": {"component": "supply", "attribute": "out__alloc__ea"}}"#;
    let curved = format!(
        r#"{{"op": "sub",
             "lhs": {{"op": "add", "args": [{c01}, {{"op": "mul", "args": [{c099}, {allocated}]}}]}},
             "rhs": {{"op": "mul", "args": [{c1e4}, {allocated}, {allocated}]}}}}"#,
        c01 = constant(0.1),
        c099 = constant(0.99),
        c1e4 = constant(0.0001),
    );
    let demand = format!(
        r#"{{"op": "if",
             "cond": {{"op": "cmp", "cmp": "gt", "lhs": {allocated}, "rhs": {c95}}},
             "then": {c20},
             "otherwise": {curved}}}"#,
        c95 = constant(9.5),
        c20 = constant(20.0),
    );
    let near = (-0.01 + (0.0001f64 + 4.0 * 0.0001 * 0.1).sqrt()) / (2.0 * 0.0001);
    let result = run(&ring(&demand), 1.0, vec![1.0]);
    let (_, value) = sampled(&result, 0)[0];
    assert!(
        (value - near).abs() <= 1e-6,
        "the resolution settled on {value}, not on the near equilibrium {near} \
         the plain sequence converges to"
    );
}

/// The creep onto 10 on a supply capped at 5: the limit of the held
/// active set lies past the capping, so the full jump crosses it. Taken
/// whole it would still land on the cap here, but only because the far
/// side has no fixed point of its own; held to its active set, the jump
/// brings the sequence up to the cap, which a plain sweep then crosses,
/// and the resolution settles on 5.
#[test]
fn a_creep_past_a_capping_settles_on_the_cap() {
    let result = run(&creep_capped(0.0, 0.01, 5.0), 1.0, vec![1.0]);
    let (_, value) = sampled(&result, 0)[0];
    assert!(
        (value - 5.0).abs() <= 1e-9,
        "the resolution settled on {value}, not on the capacity 5"
    );
}

/// Inside the right-hand side the settled flow `1000·c·x = t` crosses the
/// capacity 1.5 at t = 1.5: before it the flow follows the clock, after it
/// the cap holds, and the integral is `t²/2` then `1.125 + 1.5·(t − 1.5)`.
#[test]
fn a_creep_crossing_a_capping_mid_segment_integrates_its_closed_form() {
    let result = run(&creep_capped(0.001, 0.0, 1.5), 2.0, vec![0.5, 1.0, 2.0]);
    for (t, value) in sampled(&result, 0) {
        let expected = t.min(1.5);
        assert!(
            (value - expected).abs() <= 1e-6,
            "at t={t} the flow reads {value}, not {expected}"
        );
    }
    for (t, value) in sampled(&result, 1) {
        let expected = if t <= 1.5 {
            0.5 * t * t
        } else {
            1.125 + 1.5 * (t - 1.5)
        };
        assert!(
            (value - expected).abs() <= 1e-6,
            "at t={t} the integral reads {value}, not the closed form {expected}"
        );
    }
}
