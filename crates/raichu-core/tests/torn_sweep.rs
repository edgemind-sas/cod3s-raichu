//! **A torn sweep inside the solver**: the right-hand side is a function
//! of the state, not of the evaluation that ran before it.
//!
//! An explicit sweep whose evaluation order reads an attribute before the
//! step that writes it (a ring the order had to tear) gives, in one pass,
//! the value the PREVIOUS evaluation left at the tear. The flow resolution
//! iterates to the fixpoint at every segment boundary; the solver's
//! right-hand side used to run a single pass per stage. Inside a segment
//! its result therefore depended on the stage evaluated before, rejected
//! trial steps included.
//!
//! On an active-set margin sitting on an EXACT balance that noise is
//! enough: it crosses the band, the resolution finds nothing changed
//! (`open` to `open`), and the segment restarts, over and over. Measured
//! 2026-09-25 on the H2 showcase, whose ventilation exhausts exactly what
//! the room hands it: 1084 restarts in 0.9 h of ventilation, 8750
//! segments and 43 656 rejected steps over 100 h, a one-year replica
//! beyond seven minutes.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::flow::FLOW_TOLERANCE;
use raichu_core::{CompiledModel, Engine, EngineConfig, SimulationResult};
use raichu_expr::Value;
use raichu_model::Model;

/// A source handing `avail` to one consumer asking for 5.
///
/// `avail = 5 + x - v` with `v = x` written AFTER it: at the fixpoint the
/// two cancel and the source offers exactly what is asked, the balance
/// the ventilation of the showcase holds. `x` is a clock (`x' = 1`), so a
/// single pass reads at `v` the clock of the previous evaluation and
/// offers 5 plus the distance between two solver stages.
///
/// `torn = false` declares `v` before `avail`: the same model, no tear.
fn balanced_source(torn: bool) -> String {
    let avail = r#"{"target": "avail", "kind": "explicit",
          "expr": {"op": "add", "args": [
            {"op": "const", "value": {"kind": "float", "value": 5.0}},
            {"op": "sub",
              "lhs": {"op": "attr", "attr": {"component": "src", "attribute": "x"}},
              "rhs": {"op": "attr", "attr": {"component": "src", "attribute": "v"}}}]}}"#;
    let v = r#"{"target": "v", "kind": "explicit",
          "expr": {"op": "attr", "attr": {"component": "src", "attribute": "x"}}}"#;
    let (first, second) = if torn { (avail, v) } else { (v, avail) };
    let body = format!(
        r#"
{{
  "name": "balanced",
  "components": [
    {{
      "name": "src",
      "attributes": [
        {{"name": "x", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}},
        {{"name": "v", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}},
        {{"name": "avail", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "ports": [
        {{"name": "out", "dir": "out", "attr": "avail",
          "channels": [{{"name": "demand"}}, {{"name": "alloc"}}]}}
      ],
      "equations": [
        {{"target": "x", "kind": "ode",
          "expr": {{"op": "const", "value": {{"kind": "float", "value": 1.0}}}}}},
        {{"target": "out__demand__e", "kind": "explicit",
          "expr": {{"op": "const", "value": {{"kind": "float", "value": 5.0}}}}}},
        {first},
        {second}
      ],
      "allocations": [
        {{"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
          "available": {{"op": "attr", "attr": {{"component": "src", "attribute": "avail"}}}},
          "policy": "proportional"}}
      ]
    }},
    {{
      "name": "sink",
      "attributes": [
        {{"name": "got", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "ports": [{{"name": "input", "dir": "in"}}],
      "equations": [
        {{"target": "got", "kind": "explicit",
          "expr": {{"op": "port_agg", "agg": "sum", "channel": "alloc",
                    "port": {{"component": "sink", "port": "input"}}}}}}
      ]
    }}
  ],
  "connections": [
    {{"name": "e", "from": {{"component": "src", "port": "out"}},
      "to": {{"component": "sink", "port": "input"}}}}
  ],
  "indicators": [
    {{"name": "got", "target": "attribute",
      "attr": {{"component": "sink", "attribute": "got"}}}}
  ]
}}
"#
    );
    format!(r#"{{"raichu_model": {{"format": 1, "requires": ["allocation"]}}, "model": {body}}}"#)
}

fn run(torn: bool) -> SimulationResult {
    let model = Model::from_json(&balanced_source(torn)).expect("model document loads");
    let compiled = CompiledModel::compile(&model).expect("model compiles");
    Engine::new(
        &compiled,
        EngineConfig {
            t_max: 10.0,
            samples: vec![2.5, 5.0, 7.5, 10.0],
            ..EngineConfig::default()
        },
    )
    .unwrap()
    .run()
    .expect("the run completes")
}

fn got(result: &SimulationResult) -> Vec<f64> {
    result.samples[0]
        .points
        .iter()
        .map(|(_, value)| match value {
            Value::Float(f) => *f,
            other => panic!("`got` is not a float: {other:?}"),
        })
        .collect()
}

#[test]
fn a_torn_balance_does_not_restart_the_segment() {
    let result = run(true);
    // One segment for the whole horizon, the sample instants included: the
    // balance never moves, so nothing is ever located.
    assert!(
        result.work.segments <= 2,
        "{} segments over 10 time units on a balance that never moves; a \
         right-hand side reading the previous stage at the tear re-crosses \
         the band at every step",
        result.work.segments
    );
    for value in got(&result) {
        assert!(
            (value - 5.0).abs() <= FLOW_TOLERANCE * 5.0,
            "the consumer receives {value}, where the fixpoint hands it 5"
        );
    }
}

#[test]
fn the_torn_and_the_untorn_order_integrate_alike() {
    let (torn, untorn) = (run(true), run(false));
    assert_eq!(torn.work.segments, untorn.work.segments);
    assert_eq!(
        torn.work.solver_steps_accepted,
        untorn.work.solver_steps_accepted
    );
    assert_eq!(
        torn.work.solver_steps_rejected,
        untorn.work.solver_steps_rejected
    );
    for (a, b) in got(&torn).iter().zip(got(&untorn)) {
        assert!((a - b).abs() <= FLOW_TOLERANCE * 5.0, "{a} against {b}");
    }
}
