//! The conservative distribution operator: one available quantity, one
//! demand per outgoing connection, one allocated quantity per outgoing
//! connection, under a declared policy.
//!
//! A sensitive-function effect writes **one** target, so it cannot express
//! a split: the share handed to one consumer depends on what every other
//! consumer asked for. The operator is therefore a construct of its own,
//! declared on the producing component and evaluated on the
//! **explicit-equation path**.
//!
//! That placement is load-bearing, not incidental. The engine runs only
//! the explicit pass inside its solver callbacks
//! (`crates/raichu-core/tests/located_crossing_spike.rs` pins it), and a
//! trigger-driven entity never runs during integration. Quantities written
//! from a trigger would stay frozen across an integration segment, every
//! watched margin reading them would stay frozen too, and a boundary
//! crossing would silently degrade from *located* to *polled*.
//!
//! The second load-bearing property is order-independence. Each consumer's
//! share is a function of its own demand and of the totals, never of its
//! position in a sweep; the two genuine ties (equal demands under a
//! proportional split, equal priorities under a priority split) break by
//! **compiled declaration index**, which is a property of the model file
//! rather than of the engine's worklist.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::flow::{allocate, CPolicy};
use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::Value;
use raichu_model::{Model, ModelError};

/// Wrap a model body in the format envelope, declaring the features the
/// bodies below use. The declaration is verified against the parsed body,
/// so an over-declaration is harmless and an omission is refused.
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

/// Textual edit of a model body, refusing a pattern that is not there:
/// a silent no-op would turn a refusal test into a test of the original
/// model, which passes for the wrong reason.
fn edited(body: &str, from: &str, to: &str) -> String {
    assert!(
        body.contains(from),
        "the edit pattern is absent from the model body: {from}"
    );
    body.replace(from, to)
}

fn refusal(body: &str) -> ModelError {
    let model = Model::from_json(&sealed(body)).expect("model document loads");
    model.validate().expect_err("the model is refused")
}

/// Run to `t_max` and return the final value of every indicator.
fn run(body: &str, t_max: f64) -> raichu_core::SimulationResult {
    let compiled = compile(body);
    let config = EngineConfig {
        t_max,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config).unwrap().run().unwrap()
}

fn value(result: &raichu_core::SimulationResult, indicator: &str) -> f64 {
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

/// Assert an allocated quantity to the tolerance the validation contract
/// applies to deterministic float arithmetic.
fn assert_close(got: f64, expected: f64, what: &str) {
    assert!(
        (got - expected).abs() <= 1e-12 * expected.abs().max(1.0),
        "{what}: got {got}, expected {expected}"
    );
}

// ---------------------------------------------------------------------
// One producer, two consumers: the shape every scenario below reuses.
// ---------------------------------------------------------------------

/// A supply of `available` units splitting between two consumers `a` and
/// `b`, which demand `demand_a` and `demand_b`.
///
/// The demands are written by explicit equations on the producing
/// component, which is the shape the resolved flow network emits; the
/// operator reads them from the `demand` channel and writes each
/// consumer's share into the `alloc` channel of the same port.
fn split_model(available: f64, demand_a: f64, demand_b: f64, policy: &str) -> String {
    format!(
        r#"
{{
  "name": "split",
  "components": [
    {{
      "name": "supply",
      "attributes": [
        {{"name": "available", "kind": "float",
          "init": {{"kind": "float", "value": {available}}}}}
      ],
      "ports": [
        {{"name": "out", "dir": "out", "attr": "available",
          "channels": [{{"name": "demand"}}, {{"name": "alloc"}}]}}
      ],
      "equations": [
        {{"target": "out__demand__a", "kind": "explicit",
          "expr": {{"op": "const", "value": {{"kind": "float", "value": {demand_a}}}}}}},
        {{"target": "out__demand__b", "kind": "explicit",
          "expr": {{"op": "const", "value": {{"kind": "float", "value": {demand_b}}}}}}}
      ],
      "allocations": [
        {{"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
          "available": {{"op": "attr",
            "attr": {{"component": "supply", "attribute": "available"}}}},
          {policy}}}
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
    }},
    {{
      "name": "b",
      "attributes": [
        {{"name": "got", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "ports": [{{"name": "input", "dir": "in"}}],
      "equations": [
        {{"target": "got", "kind": "explicit",
          "expr": {{"op": "port_agg", "agg": "sum", "channel": "alloc",
                    "port": {{"component": "b", "port": "input"}}}}}}
      ]
    }}
  ],
  "connections": [
    {{"name": "a", "from": {{"component": "supply", "port": "out"}},
      "to": {{"component": "a", "port": "input"}}}},
    {{"name": "b", "from": {{"component": "supply", "port": "out"}},
      "to": {{"component": "b", "port": "input"}}}}
  ],
  "indicators": [
    {{"name": "a_got", "target": "attribute",
      "attr": {{"component": "a", "attribute": "got"}}}},
    {{"name": "b_got", "target": "attribute",
      "attr": {{"component": "b", "attribute": "got"}}}}
  ]
}}
"#
    )
}

/// The `shares` policy fragment, keyed by the consumer each share applies
/// to (the destination of the connection, not a position in a list).
fn shares(share_a: f64, share_b: f64) -> String {
    format!(
        r#""policy": "shares", "shares": [
            {{"to": {{"component": "a", "port": "input"}}, "value": {share_a}}},
            {{"to": {{"component": "b", "port": "input"}}, "value": {share_b}}}]"#
    )
}

/// The `priority` policy fragment: the lowest rank is served first.
fn priorities(rank_a: f64, rank_b: f64) -> String {
    format!(
        r#""policy": "priority", "priorities": [
            {{"to": {{"component": "a", "port": "input"}}, "value": {rank_a}}},
            {{"to": {{"component": "b", "port": "input"}}, "value": {rank_b}}}]"#
    )
}

#[test]
fn proportional_split_serves_three_and_two() {
    let result = run(
        &split_model(5.0, 6.0, 4.0, r#""policy": "proportional""#),
        1.0,
    );
    assert_close(value(&result, "a_got"), 3.0, "a receives 5 * 6/10");
    assert_close(value(&result, "b_got"), 2.0, "b receives 5 * 4/10");
}

#[test]
fn a_surplus_demand_is_never_over_served() {
    // Demands below the available quantity are served in full, and the
    // operator keeps the surplus rather than inventing a consumer for it.
    let result = run(
        &split_model(10.0, 6.0, 1.0, r#""policy": "proportional""#),
        1.0,
    );
    assert_close(value(&result, "a_got"), 6.0, "a receives its whole demand");
    assert_close(value(&result, "b_got"), 1.0, "b receives its whole demand");
}

#[test]
fn priority_serves_the_lowest_rank_first() {
    // `a` ranks 2 and `b` ranks 1: `b` is served in full and `a` takes
    // the remainder, whatever the declaration order of the shares.
    let result = run(&split_model(5.0, 3.0, 4.0, &priorities(2.0, 1.0)), 1.0);
    assert_close(value(&result, "b_got"), 4.0, "b is served first, in full");
    assert_close(value(&result, "a_got"), 1.0, "a takes what is left");
}

#[test]
fn equal_priority_and_demand_break_by_declaration_index() {
    // Both consumers rank 1 and both demand 3, against an available 5.
    // The tie breaks by the compiled declaration index of the connection:
    // the first-declared edge is served first.
    let result = run(&split_model(5.0, 3.0, 3.0, &priorities(1.0, 1.0)), 1.0);
    assert_close(value(&result, "a_got"), 3.0, "the first edge is served");
    assert_close(value(&result, "b_got"), 2.0, "the second takes the rest");

    // Swap the two connections in the model file and the allocation
    // swaps with them: the tie-break is the declaration index and
    // nothing else (never a hash order, never an arrival order).
    let swapped = edited(
        &split_model(5.0, 3.0, 3.0, &priorities(1.0, 1.0)),
        r#"{"name": "a", "from": {"component": "supply", "port": "out"},
      "to": {"component": "a", "port": "input"}},
    {"name": "b", "from": {"component": "supply", "port": "out"},
      "to": {"component": "b", "port": "input"}}"#,
        r#"{"name": "b", "from": {"component": "supply", "port": "out"},
      "to": {"component": "b", "port": "input"}},
    {"name": "a", "from": {"component": "supply", "port": "out"},
      "to": {"component": "a", "port": "input"}}"#,
    );
    let result = run(&swapped, 1.0);
    assert_close(value(&result, "b_got"), 3.0, "the first edge is served");
    assert_close(value(&result, "a_got"), 2.0, "the second takes the rest");
}

#[test]
fn a_short_demand_does_not_absorb_the_surplus() {
    // Fixed halves of 10 would hand 5 to each, but `a` only asks for 2.
    // The capping loop must give `a` its 2 and redistribute the surplus
    // to `b` on the next pass, rather than leaving 3 units stranded.
    let body = split_model(10.0, 2.0, 10.0, &shares(0.5, 0.5));
    let result = run(&body, 1.0);
    assert_close(value(&result, "a_got"), 2.0, "a is capped at its demand");
    assert_close(value(&result, "b_got"), 8.0, "b absorbs the surplus");
    assert!(
        result.work.allocation_capping_passes >= 2,
        "the capping loop ran more than once: {:?}",
        result.work
    );
}

#[test]
fn the_counter_measures_the_capping_loop_and_not_the_sweep() {
    // Two models with the same number of explicit sweeps: one where no
    // consumer is over-served, one where the capping loop has work to do.
    // What separates their counts is the loop, which is what the counter
    // must report for a later performance gate to mean anything.
    let uncapped = run(
        &split_model(5.0, 6.0, 4.0, r#""policy": "proportional""#),
        1.0,
    )
    .work;
    let capped = run(&split_model(10.0, 2.0, 10.0, &shares(0.5, 0.5)), 1.0).work;
    assert_eq!(
        uncapped.explicit_evaluations, capped.explicit_evaluations,
        "the two models sweep the same number of times"
    );
    assert!(
        uncapped.allocation_capping_passes > 0
            && capped.allocation_capping_passes > uncapped.allocation_capping_passes,
        "capping costs passes: {uncapped:?} against {capped:?}"
    );
}

// ---------------------------------------------------------------------
// Order-independence: the sweep order of the discrete fixpoint must not
// reach the allocated quantities.
// ---------------------------------------------------------------------

/// The demand-writing sensitive function of one consumer: `a` always
/// asks for 6 and `b` for 4, whatever order the two are declared in.
fn demand_function(consumer: &str, demand: f64) -> String {
    format!(
        r#"{{"name": "ask_{consumer}", "effects": [
          {{"target": {{"component": "supply", "attribute": "out__demand__{consumer}"}},
            "value": {{"op": "const", "value": {{"kind": "float", "value": {demand}}}}}}}]}}"#
    )
}

/// The same split, with the demands written by **sensitive functions**
/// instead of explicit equations, so the discrete fixpoint has something
/// to converge and its visit order is observable. The two models differ
/// only in the declaration order of those functions.
fn function_driven_model(reversed: bool) -> String {
    let (first, second) = if reversed {
        (demand_function("b", 4.0), demand_function("a", 6.0))
    } else {
        (demand_function("a", 6.0), demand_function("b", 4.0))
    };
    format!(
        r#"
{{
  "name": "function_driven_split",
  "components": [
    {{
      "name": "supply",
      "attributes": [
        {{"name": "available", "kind": "float", "init": {{"kind": "float", "value": 5.0}}}}
      ],
      "ports": [
        {{"name": "out", "dir": "out", "attr": "available",
          "channels": [{{"name": "demand"}}, {{"name": "alloc"}}]}}
      ],
      "sensitive_functions": [{first}, {second}],
      "allocations": [
        {{"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
          "available": {{"op": "attr",
            "attr": {{"component": "supply", "attribute": "available"}}}},
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
    }},
    {{
      "name": "b",
      "attributes": [
        {{"name": "got", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "ports": [{{"name": "input", "dir": "in"}}],
      "equations": [
        {{"target": "got", "kind": "explicit",
          "expr": {{"op": "port_agg", "agg": "sum", "channel": "alloc",
                    "port": {{"component": "b", "port": "input"}}}}}}
      ]
    }}
  ],
  "connections": [
    {{"name": "a", "from": {{"component": "supply", "port": "out"}},
      "to": {{"component": "a", "port": "input"}}}},
    {{"name": "b", "from": {{"component": "supply", "port": "out"}},
      "to": {{"component": "b", "port": "input"}}}}
  ],
  "indicators": [
    {{"name": "a_got", "target": "attribute",
      "attr": {{"component": "a", "attribute": "got"}}}},
    {{"name": "b_got", "target": "attribute",
      "attr": {{"component": "b", "attribute": "got"}}}}
  ]
}}
"#
    )
}

/// Run with the confluence probe on: the probe converges the discrete
/// fixpoint forward, then again in reverse worklist order, and compares
/// the two states for exact equality.
fn run_probed(body: &str) -> raichu_core::SimulationResult {
    let compiled = compile(body);
    let config = EngineConfig {
        t_max: 1.0,
        confluence_check: true,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config)
        .expect("the probe accepts an allocation model")
        .run()
        .expect("the probe reports no divergence")
}

#[test]
fn the_confluence_probe_accepts_an_allocation_model() {
    // The operator sits on the explicit path, outside the sensitive-function
    // table the probe replays, and its outputs are recomputed from the
    // converged state afterwards. Forward and reverse must therefore agree,
    // and the probe must not report a divergence it cannot name a writer for.
    let result = run_probed(&function_driven_model(false));
    assert_close(value(&result, "a_got"), 3.0, "a under the probe");
    assert_close(value(&result, "b_got"), 2.0, "b under the probe");
}

#[test]
fn the_allocation_ignores_the_function_declaration_order() {
    // The same two demands, written by the same two functions, declared
    // the other way round: the fixpoint visits them in the opposite order
    // and the allocated quantities are identical, bit for bit.
    let forward = run(&function_driven_model(false), 1.0);
    let reverse = run(&function_driven_model(true), 1.0);
    assert_eq!(
        value(&forward, "a_got"),
        value(&reverse, "a_got"),
        "a receives the same share under either visit order"
    );
    assert_eq!(
        value(&forward, "b_got"),
        value(&reverse, "b_got"),
        "b receives the same share under either visit order"
    );
    assert_close(value(&forward, "a_got"), 3.0, "a receives 5 * 6/10");
    assert_close(value(&forward, "b_got"), 2.0, "b receives 5 * 4/10");
}

// ---------------------------------------------------------------------
// The allocated quantities move during an integration segment.
// ---------------------------------------------------------------------

/// The available quantity is integrated (`d available/dt = 1`), the split
/// is proportional to two constant demands of 6 and 4, and consumer `a`
/// watches its own share against 3.
///
/// `a` therefore crosses at `0.6 * t = 3`, i.e. `t = 5`. The model also
/// carries a delay transition at `t = 100`: an implementation that let the
/// allocated quantities stay frozen between discrete dates would report
/// the crossing there (or never), not at 5.
const INTEGRATED_SPLIT: &str = r#"
{
  "name": "integrated_split",
  "components": [
    {
      "name": "supply",
      "attributes": [
        {"name": "available", "kind": "float", "init": {"kind": "float", "value": 0.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "available",
         "channels": [{"name": "demand"}, {"name": "alloc"}]}
      ],
      "equations": [
        {"target": "available", "kind": "ode",
         "expr": {"op": "const", "value": {"kind": "float", "value": 1.0}}},
        {"target": "out__demand__a", "kind": "explicit",
         "expr": {"op": "const", "value": {"kind": "float", "value": 6.0}}},
        {"target": "out__demand__b", "kind": "explicit",
         "expr": {"op": "const", "value": {"kind": "float", "value": 4.0}}}
      ],
      "allocations": [
        {"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
         "available": {"op": "attr",
           "attr": {"component": "supply", "attribute": "available"}},
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
      ],
      "automata": [
        {"name": "gate", "states": ["low", "high"], "init": "low",
         "transitions": [
           {"name": "cross", "source": "low", "targets": ["high"], "distrib": "watched",
            "guard": {"op": "cmp", "cmp": "ge",
              "lhs": {"op": "attr", "attr": {"component": "a", "attribute": "got"}},
              "rhs": {"op": "const", "value": {"kind": "float", "value": 3.0}}}}]},
        {"name": "far", "states": ["p", "q"], "init": "p",
         "transitions": [
           {"name": "later", "source": "p", "targets": ["q"],
            "distrib": "delay", "time": 100.0}]}
      ]
    },
    {
      "name": "b",
      "attributes": [
        {"name": "got", "kind": "float", "init": {"kind": "float", "value": 0.0}}
      ],
      "ports": [{"name": "input", "dir": "in"}],
      "equations": [
        {"target": "got", "kind": "explicit",
         "expr": {"op": "port_agg", "agg": "sum", "channel": "alloc",
                  "port": {"component": "b", "port": "input"}}}
      ]
    }
  ],
  "connections": [
    {"name": "a", "from": {"component": "supply", "port": "out"},
     "to": {"component": "a", "port": "input"}},
    {"name": "b", "from": {"component": "supply", "port": "out"},
     "to": {"component": "b", "port": "input"}}
  ],
  "indicators": [
    {"name": "a_got", "target": "attribute",
     "attr": {"component": "a", "attribute": "got"}}
  ]
}
"#;

#[test]
fn an_allocated_quantity_moves_inside_an_integration_segment() {
    let result = run(INTEGRATED_SPLIT, 20.0);
    let crossing = result
        .events
        .iter()
        .find(|event| event.transition.ends_with(".cross"))
        .unwrap_or_else(|| {
            panic!(
                "the watched boundary on an allocated quantity never fired; \
                 events were {:?}",
                result.events
            )
        });
    assert!(
        (crossing.time - 5.0).abs() < 1e-6,
        "the crossing was reported at t={}, expected the analytic t=5. A value \
         near t=100 (the scheduled delay) or at the horizon means the allocated \
         quantities stayed frozen through the segment and the boundary was \
         polled rather than located.",
        crossing.time
    );
}

// ---------------------------------------------------------------------
// A multi-producer, multi-consumer network with feedback.
// ---------------------------------------------------------------------

/// Two producers and two consumers, wired so that what `c1` asks of `p2`
/// depends on what `p1` allocated it: the feedback a one-to-one edge
/// cannot exhibit.
///
/// ```text
///   p1 (5 available) ──► c1.in1        p2 (2 available) ──► c1.in2
///                    └─► c2.in1                         └─► c2.in2
/// ```
///
/// `c1` needs 6 in total: it asks `p1` for 6, then asks `p2` for whatever
/// `p1` did not give it. `c2` asks `p1` for 4 and `p2` for 1. The declared
/// evaluation order makes one sweep exact: demands into `p1`, `p1`'s
/// split, what `c1` received, `c1`'s residual demand on `p2`, `p2`'s
/// split.
fn network_model(policy1: &str, policy2: &str) -> String {
    format!(
        r#"
{{
  "name": "network",
  "components": [
    {{
      "name": "p1",
      "attributes": [
        {{"name": "available", "kind": "float", "init": {{"kind": "float", "value": 5.0}}}}
      ],
      "ports": [
        {{"name": "out", "dir": "out", "attr": "available",
          "channels": [{{"name": "demand"}}, {{"name": "alloc"}}]}}
      ],
      "equations": [
        {{"target": "out__demand__e1", "kind": "explicit",
          "expr": {{"op": "const", "value": {{"kind": "float", "value": 6.0}}}}}},
        {{"target": "out__demand__e2", "kind": "explicit",
          "expr": {{"op": "const", "value": {{"kind": "float", "value": 4.0}}}}}}
      ],
      "allocations": [
        {{"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
          "available": {{"op": "attr",
            "attr": {{"component": "p1", "attribute": "available"}}}},
          {policy1}}}
      ]
    }},
    {{
      "name": "p2",
      "attributes": [
        {{"name": "available", "kind": "float", "init": {{"kind": "float", "value": 2.0}}}}
      ],
      "ports": [
        {{"name": "out", "dir": "out", "attr": "available",
          "channels": [{{"name": "demand"}}, {{"name": "alloc"}}]}}
      ],
      "equations": [
        {{"target": "out__demand__e3", "kind": "explicit",
          "expr": {{"op": "max", "args": [
            {{"op": "const", "value": {{"kind": "float", "value": 0.0}}}},
            {{"op": "sub",
              "lhs": {{"op": "const", "value": {{"kind": "float", "value": 6.0}}}},
              "rhs": {{"op": "attr", "attr": {{"component": "c1", "attribute": "from_p1"}}}}}}]}}}},
        {{"target": "out__demand__e4", "kind": "explicit",
          "expr": {{"op": "const", "value": {{"kind": "float", "value": 1.0}}}}}}
      ],
      "allocations": [
        {{"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
          "available": {{"op": "attr",
            "attr": {{"component": "p2", "attribute": "available"}}}},
          {policy2}}}
      ]
    }},
    {{
      "name": "c1",
      "attributes": [
        {{"name": "from_p1", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}},
        {{"name": "from_p2", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "ports": [
        {{"name": "in1", "dir": "in"}},
        {{"name": "in2", "dir": "in"}}
      ],
      "equations": [
        {{"target": "from_p1", "kind": "explicit",
          "expr": {{"op": "port_agg", "agg": "sum", "channel": "alloc",
                    "port": {{"component": "c1", "port": "in1"}}}}}},
        {{"target": "from_p2", "kind": "explicit",
          "expr": {{"op": "port_agg", "agg": "sum", "channel": "alloc",
                    "port": {{"component": "c1", "port": "in2"}}}}}}
      ]
    }},
    {{
      "name": "c2",
      "attributes": [
        {{"name": "from_p1", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}},
        {{"name": "from_p2", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "ports": [
        {{"name": "in1", "dir": "in"}},
        {{"name": "in2", "dir": "in"}}
      ],
      "equations": [
        {{"target": "from_p1", "kind": "explicit",
          "expr": {{"op": "port_agg", "agg": "sum", "channel": "alloc",
                    "port": {{"component": "c2", "port": "in1"}}}}}},
        {{"target": "from_p2", "kind": "explicit",
          "expr": {{"op": "port_agg", "agg": "sum", "channel": "alloc",
                    "port": {{"component": "c2", "port": "in2"}}}}}}
      ]
    }}
  ],
  "connections": [
    {{"name": "e1", "from": {{"component": "p1", "port": "out"}},
      "to": {{"component": "c1", "port": "in1"}}}},
    {{"name": "e2", "from": {{"component": "p1", "port": "out"}},
      "to": {{"component": "c2", "port": "in1"}}}},
    {{"name": "e3", "from": {{"component": "p2", "port": "out"}},
      "to": {{"component": "c1", "port": "in2"}}}},
    {{"name": "e4", "from": {{"component": "p2", "port": "out"}},
      "to": {{"component": "c2", "port": "in2"}}}}
  ],
  "evaluation_order": [
    {{"component": "p1", "attribute": "out__demand__e1"}},
    {{"component": "p1", "attribute": "out__demand__e2"}},
    {{"component": "p1", "attribute": "split"}},
    {{"component": "c1", "attribute": "from_p1"}},
    {{"component": "c2", "attribute": "from_p1"}},
    {{"component": "p2", "attribute": "out__demand__e3"}},
    {{"component": "p2", "attribute": "out__demand__e4"}},
    {{"component": "p2", "attribute": "split"}},
    {{"component": "c1", "attribute": "from_p2"}},
    {{"component": "c2", "attribute": "from_p2"}}
  ],
  "indicators": [
    {{"name": "c1_p1", "target": "attribute",
      "attr": {{"component": "c1", "attribute": "from_p1"}}}},
    {{"name": "c1_p2", "target": "attribute",
      "attr": {{"component": "c1", "attribute": "from_p2"}}}},
    {{"name": "c2_p1", "target": "attribute",
      "attr": {{"component": "c2", "attribute": "from_p1"}}}},
    {{"name": "c2_p2", "target": "attribute",
      "attr": {{"component": "c2", "attribute": "from_p2"}}}}
  ]
}}
"#
    )
}

/// Consumer-keyed policy parameters for the network's two producers.
fn network_params(policy: &str, c1: f64, c2: f64, key: &str, port2: &str) -> String {
    format!(
        r#""policy": "{policy}", "{key}": [
            {{"to": {{"component": "c1", "port": "{port2}"}}, "value": {c1}}},
            {{"to": {{"component": "c2", "port": "{port2}"}}, "value": {c2}}}]"#
    )
}

#[test]
fn a_network_with_feedback_matches_the_hand_calculation() {
    // Proportional. p1: 5 among demands 6 and 4 -> 3 and 2. c1 then asks
    // p2 for 6 - 3 = 3, c2 asks for 1; p2 splits 2 among 3 and 1 -> 1.5
    // and 0.5.
    let body = network_model(r#""policy": "proportional""#, r#""policy": "proportional""#);
    let result = run(&body, 1.0);
    assert_close(value(&result, "c1_p1"), 3.0, "proportional c1 from p1");
    assert_close(value(&result, "c2_p1"), 2.0, "proportional c2 from p1");
    assert_close(value(&result, "c1_p2"), 1.5, "proportional c1 from p2");
    assert_close(value(&result, "c2_p2"), 0.5, "proportional c2 from p2");

    // Fixed shares. p1 hands 0.8/0.2 of 5 -> 4 and 1, both under demand.
    // c1 then asks p2 for 6 - 4 = 2, c2 for 1; p2's 0.25/0.75 of 2 would
    // be 0.5 and 1.5, but c2 only wants 1: it is capped there and the
    // surplus goes back to c1 on the next pass -> 1 and 1.
    let body = network_model(
        &network_params("shares", 0.8, 0.2, "shares", "in1"),
        &network_params("shares", 0.25, 0.75, "shares", "in2"),
    );
    let result = run(&body, 1.0);
    assert_close(value(&result, "c1_p1"), 4.0, "shares c1 from p1");
    assert_close(value(&result, "c2_p1"), 1.0, "shares c2 from p1");
    assert_close(value(&result, "c1_p2"), 1.0, "shares c1 from p2");
    assert_close(value(&result, "c2_p2"), 1.0, "shares c2 from p2");

    // Priority. p1 serves c2 first (rank 1): 4 of 5, leaving 1 for c1.
    // c1 then asks p2 for 6 - 1 = 5 and is served first there (rank 1),
    // taking all 2 and leaving c2 nothing.
    let body = network_model(
        &network_params("priority", 2.0, 1.0, "priorities", "in1"),
        &network_params("priority", 1.0, 2.0, "priorities", "in2"),
    );
    let result = run(&body, 1.0);
    assert_close(value(&result, "c1_p1"), 1.0, "priority c1 from p1");
    assert_close(value(&result, "c2_p1"), 4.0, "priority c2 from p1");
    assert_close(value(&result, "c1_p2"), 2.0, "priority c1 from p2");
    assert_close(value(&result, "c2_p2"), 0.0, "priority c2 from p2");
}

#[test]
fn a_cyclic_flow_network_is_not_refused_as_an_algebraic_loop() {
    // What a consumer asks for depends on what it was given: a
    // conservative flow network is cyclic by nature. The build-time
    // algebraic-loop refusal must therefore treat an allocated quantity
    // the way it treats an integrated one, as a value the cycle does not
    // recompute, or it would refuse the very shape the operator exists
    // for.
    //
    // The contrast is with `flow_diagnostics.rs`, where the *same* loop
    // closed through a channel written by an explicit equation is refused
    // as an algebraic loop: it is the operator, not the channel, that
    // breaks the cycle, exactly as an integrated attribute does.
    //
    // What one sweep computes here is one iteration from the previous
    // evaluation point, not the network's fixpoint: converging it is the
    // network resolution, which this unit does not carry.
    let body = edited(
        &split_model(5.0, 6.0, 4.0, r#""policy": "proportional""#),
        r#"{"target": "out__demand__a", "kind": "explicit",
          "expr": {"op": "const", "value": {"kind": "float", "value": 6}}}"#,
        r#"{"target": "out__demand__a", "kind": "explicit",
          "expr": {"op": "add", "args": [
            {"op": "const", "value": {"kind": "float", "value": 6.0}},
            {"op": "attr", "attr": {"component": "a", "attribute": "got"}}]}}"#,
    );
    let model = Model::from_json(&sealed(&body)).expect("model document loads");
    model
        .validate()
        .expect("a demand that reads its own allocation is accepted");
}

// ---------------------------------------------------------------------
// Build-time refusals.
// ---------------------------------------------------------------------

#[test]
fn fixed_shares_must_sum_to_one() {
    let error = refusal(&split_model(5.0, 6.0, 4.0, &shares(0.5, 0.3)));
    let message = error.to_string();
    assert!(
        matches!(error, ModelError::AllocationSharesNotUnit { .. }),
        "expected a share-sum refusal, got {error:?}"
    );
    assert!(
        message.contains("supply") && message.contains("split") && message.contains("out"),
        "the refusal names the component and the flow: {message}"
    );
}

#[test]
fn every_consumer_of_the_port_needs_a_parameter() {
    // `b` is connected to the port but carries no share.
    let body = split_model(
        5.0,
        6.0,
        4.0,
        r#""policy": "shares", "shares": [
        {"to": {"component": "a", "port": "input"}, "value": 1.0}]"#,
    );
    assert!(
        matches!(refusal(&body), ModelError::AllocationParamMissing { .. }),
        "a connection with no declared share is refused"
    );
}

#[test]
fn a_parameter_must_name_a_connected_consumer() {
    let body = split_model(
        5.0,
        6.0,
        4.0,
        r#""policy": "shares", "shares": [
            {"to": {"component": "a", "port": "input"}, "value": 0.5},
            {"to": {"component": "b", "port": "input"}, "value": 0.4},
            {"to": {"component": "b", "port": "elsewhere"}, "value": 0.1}]"#,
    );
    assert!(
        matches!(refusal(&body), ModelError::AllocationParamUnknown { .. }),
        "a share for an edge the port does not carry is refused"
    );
}

#[test]
fn the_demand_and_allocated_channels_must_differ() {
    let body = edited(
        &split_model(5.0, 6.0, 4.0, r#""policy": "proportional""#),
        r#""allocated": "alloc""#,
        r#""allocated": "demand""#,
    );
    assert!(
        matches!(refusal(&body), ModelError::AllocationChannelReused { .. }),
        "reading and writing the same channel is refused"
    );
}

#[test]
fn the_operator_must_name_declared_channels() {
    let body = edited(
        &split_model(5.0, 6.0, 4.0, r#""policy": "proportional""#),
        r#""demand": "demand""#,
        r#""demand": "wanted""#,
    );
    assert!(
        matches!(refusal(&body), ModelError::AllocationUnknownChannel { .. }),
        "a channel the port does not declare is refused"
    );
}

#[test]
fn the_operator_must_sit_on_an_out_port() {
    let body = edited(
        &split_model(5.0, 6.0, 4.0, r#""policy": "proportional""#),
        r#""port": "out", "demand""#,
        r#""port": "nowhere", "demand""#,
    );
    assert!(
        matches!(refusal(&body), ModelError::AllocationPortInvalid { .. }),
        "an operator on an unknown port is refused"
    );
}

#[test]
fn nothing_else_may_write_an_allocated_quantity() {
    // An explicit equation on the producing component targeting one of the
    // quantities the operator writes: last writer wins silently otherwise.
    let body = edited(
        &split_model(5.0, 6.0, 4.0, r#""policy": "proportional""#),
        r#""equations": [
        {"target": "out__demand__a""#,
        r#""equations": [
        {"target": "out__alloc__b", "kind": "explicit",
          "expr": {"op": "const", "value": {"kind": "float", "value": 9.0}}},
        {"target": "out__demand__a""#,
    );
    assert!(
        matches!(refusal(&body), ModelError::AllocationTargetWritten { .. }),
        "a second writer of an allocated quantity is refused"
    );
}

#[test]
fn two_operators_may_not_allocate_the_same_quantity() {
    let body = edited(
        &split_model(5.0, 6.0, 4.0, r#""policy": "proportional""#),
        r#""allocations": [
        {"name": "split", "port": "out","#,
        r#""allocations": [
        {"name": "twin", "port": "out", "demand": "demand", "allocated": "alloc",
          "available": {"op": "const", "value": {"kind": "float", "value": 1.0}},
          "policy": "proportional"},
        {"name": "split", "port": "out","#,
    );
    assert!(
        matches!(refusal(&body), ModelError::AllocationTargetWritten { .. }),
        "a second operator on the same channel is refused"
    );
}

#[test]
fn an_operator_may_not_take_the_name_of_an_equation_of_its_component() {
    // Both are steps of the sweep, designated in the evaluation order by
    // their name: one name, one step.
    let body = edited(
        &split_model(5.0, 6.0, 4.0, r#""policy": "proportional""#),
        r#"{"name": "split", "port": "out","#,
        r#"{"name": "out__demand__a", "port": "out","#,
    );
    assert!(
        matches!(refusal(&body), ModelError::EvaluationStepAmbiguous { .. }),
        "an operator named after an explicit equation is refused"
    );
}

#[test]
fn an_undeclared_operator_is_refused_by_the_envelope() {
    // The bare body uses a non-baseline construct without declaring it:
    // an engine that predates the operator would ignore it in silence.
    let body = split_model(5.0, 6.0, 4.0, r#""policy": "proportional""#);
    let error = Model::from_json(&body).expect_err("the bare body is refused");
    assert!(
        error.to_string().contains("allocation"),
        "the refusal names the construct: {error}"
    );
}

// ---------------------------------------------------------------------
// Conservation, as a property of the operator itself.
// ---------------------------------------------------------------------

/// A deterministic 64-bit LCG: the demand vectors below must be the same
/// on every machine and every run, so the property test is reproducible
/// rather than merely random.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self, high: f64) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * high
    }
}

/// The policy and the demand vector of one random case.
fn random_case(rng: &mut Lcg, case: u32, n: usize) -> (CPolicy, Vec<f64>) {
    let demands: Vec<f64> = (0..n).map(|_| rng.next_f64(12.0)).collect();
    let policy = match case % 3 {
        0 => CPolicy::Proportional,
        1 => {
            // Random shares, normalised to sum to 1 as validation requires.
            let raw: Vec<f64> = (0..n).map(|_| rng.next_f64(1.0) + 0.05).collect();
            let total: f64 = raw.iter().sum();
            CPolicy::Shares(raw.iter().map(|s| s / total).collect())
        }
        _ => {
            // A serving order is a permutation of the consumers; the
            // compiler derives it from (rank, declaration index).
            let mut order: Vec<usize> = (0..n).collect();
            order.rotate_left(case as usize % n.max(1));
            CPolicy::Priority(order)
        }
    };
    (policy, demands)
}

/// The same policy with its consumers reversed, so the reversed problem
/// is the same problem read backwards.
fn reversed(policy: &CPolicy, n: usize) -> CPolicy {
    match policy {
        CPolicy::Proportional => CPolicy::Proportional,
        CPolicy::Shares(shares) => CPolicy::Shares(shares.iter().rev().copied().collect()),
        CPolicy::Priority(order) => {
            CPolicy::Priority(order.iter().map(|index| n - 1 - index).collect())
        }
    }
}

#[test]
fn allocations_never_exceed_the_available_quantity() {
    // The properties below hold for every demand vector, not for the
    // handful of hand-picked ones above. The generator is a deterministic
    // LCG rather than a seeded random one so a failure is reproducible on
    // any machine and in any order, which is the same reason the engine
    // has no hidden RNG.
    let mut rng = Lcg(0x5eed);
    let mut capped = Vec::new();
    let mut allocated = Vec::new();
    let mut mirrored = Vec::new();
    for case in 0..4000u32 {
        let n = 1 + (case as usize % 7);
        let available = rng.next_f64(20.0);
        let (policy, demands) = random_case(&mut rng, case, n);
        allocated.clear();
        allocated.resize(n, f64::NAN);
        let passes = allocate(&policy, available, &demands, &mut allocated, &mut capped);

        // The capping loop is bounded by the consumer count: each pass
        // caps at least one consumer, and one more pass finds nothing to
        // cap. This is a finite search, never an iteration to a tolerance.
        assert!(
            passes >= 1 && passes as usize <= n + 1,
            "case {case}: {passes} passes for {n} consumers"
        );

        // Conservation. The tolerance is the rounding of the sum itself:
        // the operator hands out at most `available`, but adding n doubles
        // back up in a test can land an ulp above it.
        let sum: f64 = allocated.iter().sum();
        assert!(
            sum <= available + 1e-12 * available.max(1.0),
            "case {case}: allocated {sum} of an available {available} \
             (demands {demands:?})"
        );

        // Nobody is over-served, and nobody receives a negative quantity.
        for (index, &given) in allocated.iter().enumerate() {
            assert!(
                given >= 0.0 && given <= demands[index] + 1e-12 * demands[index].max(1.0),
                "case {case}: consumer {index} received {given} against a \
                 demand of {}",
                demands[index]
            );
        }

        // No shortage means no split: everybody is served in full.
        let asked: f64 = demands.iter().sum();
        if asked < available * (1.0 - 1e-9) {
            for (index, &given) in allocated.iter().enumerate() {
                assert!(
                    (given - demands[index]).abs() <= 1e-12 * demands[index].max(1.0),
                    "case {case}: consumer {index} received {given} of a \
                     demand of {} that the available {available} covers",
                    demands[index]
                );
            }
        }

        // Order independence: reverse the consumers and the allocation
        // reverses with them. A share that depended on a position in the
        // sweep, or on the order the loop happened to visit, would not
        // survive this. It is asserted to a tolerance rather than bit for
        // bit because the *totals* are ordered floating-point sums, which
        // the reversal reassociates.
        let mirror_demands: Vec<f64> = demands.iter().rev().copied().collect();
        mirrored.clear();
        mirrored.resize(n, f64::NAN);
        allocate(
            &reversed(&policy, n),
            available,
            &mirror_demands,
            &mut mirrored,
            &mut capped,
        );
        for index in 0..n {
            let (forward, backward) = (allocated[index], mirrored[n - 1 - index]);
            assert!(
                (forward - backward).abs() <= 1e-9 * forward.abs().max(1.0),
                "case {case}: consumer {index} received {forward} forwards \
                 and {backward} backwards"
            );
        }
    }
}
