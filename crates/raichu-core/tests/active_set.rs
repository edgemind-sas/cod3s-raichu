//! **Active-set resolution**: the combinatorial part of a conservative
//! flow network is settled once, at the segment boundary, and the
//! resolved flows then move with the state through the integration.
//!
//! Two things are separated here, and keeping them apart is the whole
//! design:
//!
//! - **Which** consumers are saturated, and which branch of each minimum
//!   is taken, is a *finite* question. It is answered by iterating the
//!   ordered sweep downward from its cold start until the answer stops
//!   changing, and it is answered only at a discrete epoch or right
//!   after a located crossing.
//! - **How much** each edge carries is a continuous function of the
//!   state. It is recomputed by the ordinary explicit pass at every
//!   solver stage, which costs what the engine already pays and is what
//!   lets a watched guard reading a flow be *located* rather than polled.
//!
//! The bridge between the two is a **watched guard per active-set
//! margin**: while the segment runs, the engine monitors the distance to
//! every saturation boundary of the frozen active set. Crossing one ends
//! the segment at the crossing instant and resolves the network again.
//! Without that generation step nothing would discover the crossing.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::compile::CStep;
use raichu_core::flow::{edge_margin, flow_band, CPolicy, EdgeClass, FLOW_TOLERANCE};
use raichu_core::{CompiledModel, Engine, EngineConfig, JournalRecord};
use raichu_expr::Value;
use raichu_model::{Model, ModelError};

/// Wrap a model body in the format envelope. The declaration is verified
/// against the parsed body, so over-declaring is harmless and omitting is
/// refused.
fn sealed(body: &str) -> String {
    format!(
        r#"{{"raichu_model": {{"format": 1,
            "requires": ["evaluation_order", "allocation"]}},
            "model": {body}}}"#
    )
}

fn parse(body: &str) -> Model {
    Model::from_json(&sealed(body)).expect("model document loads")
}

fn compile(body: &str) -> CompiledModel {
    CompiledModel::compile(&parse(body)).expect("model compiles")
}

fn refusal(body: &str) -> ModelError {
    parse(body).validate().expect_err("the model is refused")
}

fn run_with(body: &str, config: EngineConfig) -> raichu_core::SimulationResult {
    let compiled = compile(body);
    Engine::new(&compiled, config).unwrap().run().unwrap()
}

fn run(body: &str, t_max: f64) -> raichu_core::SimulationResult {
    run_with(
        body,
        EngineConfig {
            t_max,
            ..EngineConfig::default()
        },
    )
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

/// Assert a resolved flow to the per-edge flow tolerance (the level the
/// resolution itself converges to: a fixpoint reached by iteration is not
/// a bit-exact quantity).
fn assert_flow(got: f64, expected: f64, what: &str) {
    let tolerance = FLOW_TOLERANCE * expected.abs().max(1.0);
    assert!(
        (got - expected).abs() <= tolerance,
        "{what}: got {got}, expected {expected} (flow tolerance {tolerance})"
    );
}

// ---------------------------------------------------------------------
// A contested supply: two consumers, one supply, each able to absorb
// more than half of it.
// ---------------------------------------------------------------------

/// One supply of 10 and two consumers `a` and `b`, each able to absorb 8.
///
/// Each consumer sizes its demand **net of what the other already
/// holds**, which is the shape of a capability computed against a shared
/// supply: `demand = min(8, capacity - what the other was allocated)`.
///
/// One ordered pass cannot answer this. It starts from the cold state
/// where neither consumer holds anything, so both size themselves as
/// though the other were absent and ask for 8: the published demands sum
/// to 16 against a supply of 10. Iterating the pass downward is what
/// brings the demands back to 5 and 5.
fn contested_supply(policy: &str) -> String {
    format!(
        r#"
{{
  "name": "contested",
  "components": [
    {{
      "name": "supply",
      "attributes": [
        {{"name": "capacity", "kind": "float",
          "init": {{"kind": "float", "value": 10.0}}}}
      ],
      "ports": [
        {{"name": "out", "dir": "out", "attr": "capacity",
          "channels": [{{"name": "demand"}}, {{"name": "alloc"}}]}}
      ],
      "equations": [
        {{"target": "out__demand__ea", "kind": "explicit",
          "expr": {{"op": "min", "args": [
            {{"op": "const", "value": {{"kind": "float", "value": 8.0}}}},
            {{"op": "max", "args": [
              {{"op": "const", "value": {{"kind": "float", "value": 0.0}}}},
              {{"op": "sub",
                "lhs": {{"op": "attr",
                  "attr": {{"component": "supply", "attribute": "capacity"}}}},
                "rhs": {{"op": "attr",
                  "attr": {{"component": "supply", "attribute": "out__alloc__eb"}}}}}}]}}]}}}},
        {{"target": "out__demand__eb", "kind": "explicit",
          "expr": {{"op": "min", "args": [
            {{"op": "const", "value": {{"kind": "float", "value": 8.0}}}},
            {{"op": "max", "args": [
              {{"op": "const", "value": {{"kind": "float", "value": 0.0}}}},
              {{"op": "sub",
                "lhs": {{"op": "attr",
                  "attr": {{"component": "supply", "attribute": "capacity"}}}},
                "rhs": {{"op": "attr",
                  "attr": {{"component": "supply", "attribute": "out__alloc__ea"}}}}}}]}}]}}}}
      ],
      "allocations": [
        {{"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
          "available": {{"op": "attr",
            "attr": {{"component": "supply", "attribute": "capacity"}}}},
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
    {{"name": "ea", "from": {{"component": "supply", "port": "out"}},
      "to": {{"component": "a", "port": "input"}}}},
    {{"name": "eb", "from": {{"component": "supply", "port": "out"}},
      "to": {{"component": "b", "port": "input"}}}}
  ],
  "indicators": [
    {{"name": "demand_a", "target": "attribute",
      "attr": {{"component": "supply", "attribute": "out__demand__ea"}}}},
    {{"name": "demand_b", "target": "attribute",
      "attr": {{"component": "supply", "attribute": "out__demand__eb"}}}},
    {{"name": "a_got", "target": "attribute",
      "attr": {{"component": "a", "attribute": "got"}}}},
    {{"name": "b_got", "target": "attribute",
      "attr": {{"component": "b", "attribute": "got"}}}}
  ]
}}
"#
    )
}

#[test]
fn the_resolved_demands_sum_to_at_most_the_supply() {
    let result = run(&contested_supply(r#""policy": "proportional""#), 1.0);
    let (demand_a, demand_b) = (value(&result, "demand_a"), value(&result, "demand_b"));
    assert!(
        demand_a + demand_b <= 10.0 + FLOW_TOLERANCE * 10.0,
        "the resolved demands sum to {} against a supply of 10; the single \
         ordered pass leaves them at 8 + 8 = 16, each consumer sized as \
         though the other were absent",
        demand_a + demand_b
    );
    assert_flow(demand_a, 5.0, "a's resolved demand");
    assert_flow(demand_b, 5.0, "b's resolved demand");
    assert_flow(value(&result, "a_got"), 5.0, "a's share");
    assert_flow(value(&result, "b_got"), 5.0, "b's share");
    assert!(
        result.work.flow_sweeps >= 2,
        "the resolution took {} sweeps; settling an active set needs at \
         least two, one to produce a candidate and one to confirm it",
        result.work.flow_sweeps
    );
}

// ---------------------------------------------------------------------
// Forward and reverse sweep orders.
// ---------------------------------------------------------------------

/// The declared evaluation order of the contested supply, forward
/// (demands, then the split, then what each consumer received) and
/// reversed.
fn ordered(body: &str, reverse: bool) -> String {
    let steps = [
        r#"{"component": "supply", "attribute": "out__demand__ea"}"#,
        r#"{"component": "supply", "attribute": "out__demand__eb"}"#,
        r#"{"component": "supply", "attribute": "split"}"#,
        r#"{"component": "a", "attribute": "got"}"#,
        r#"{"component": "b", "attribute": "got"}"#,
    ];
    let mut order: Vec<&str> = steps.to_vec();
    if reverse {
        order.reverse();
    }
    let order = order.join(",\n    ");
    let marker = "\"indicators\":";
    let at = body.find(marker).expect("the body declares indicators");
    format!(
        "{}\"evaluation_order\": [\n    {}\n  ],\n  {}",
        &body[..at],
        order,
        &body[at..]
    )
}

/// **What this test establishes, and what it does not.**
///
/// It is the replacement for the confluence probe on a continuous
/// fixture: the probe compares converged states for *exact* equality,
/// which a flow loop converged to a tolerance cannot satisfy between two
/// sweep orders even when both reach the same answer.
///
/// Where the policy carries a uniqueness result, agreeing within the flow
/// tolerance is genuine determinism: the network has one consistent flow
/// and both orders find it. Where more than one fixpoint is admissible,
/// this test establishes the *weaker* property, and that is deliberate:
/// see `a_multi_fixpoint_network_is_pinned_by_the_compiled_order`, which
/// asserts only that the compiled order pins which admissible fixpoint is
/// reached, never that every order reaches the same one.
#[test]
fn a_forward_and_a_reverse_order_resolve_to_the_same_flows() {
    let body = contested_supply(r#""policy": "proportional""#);
    let forward = run(&ordered(&body, false), 1.0);
    let reverse = run(&ordered(&body, true), 1.0);
    for indicator in ["a_got", "b_got", "demand_a", "demand_b"] {
        assert_flow(
            value(&reverse, indicator),
            value(&forward, indicator),
            &format!("`{indicator}` under the reverse order"),
        );
    }
    assert_flow(value(&forward, "a_got"), 5.0, "a's share, forward order");
}

// ---------------------------------------------------------------------
// A network admitting more than one fixpoint.
// ---------------------------------------------------------------------

/// A supply of 10 split proportionally between two consumers that each
/// ask for exactly what they were last given.
///
/// Every split summing to 10 satisfies the sweep, and so does the empty
/// one: the network has a continuum of fixpoints and no uniqueness result
/// to appeal to. What pins the answer is the **cold start** of the
/// resolution (every allocated quantity at zero) followed by the compiled
/// order, which is a property of the model file rather than of the run.
const MULTI_FIXPOINT: &str = r#"
{
  "name": "multi_fixpoint",
  "components": [
    {
      "name": "supply",
      "attributes": [
        {"name": "capacity", "kind": "float", "init": {"kind": "float", "value": 10.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "capacity",
         "channels": [{"name": "demand"}, {"name": "alloc"}]}
      ],
      "equations": [
        {"target": "out__demand__ea", "kind": "explicit",
         "expr": {"op": "attr",
           "attr": {"component": "supply", "attribute": "out__alloc__ea"}}},
        {"target": "out__demand__eb", "kind": "explicit",
         "expr": {"op": "attr",
           "attr": {"component": "supply", "attribute": "out__alloc__eb"}}}
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
    {"name": "ea", "from": {"component": "supply", "port": "out"},
     "to": {"component": "a", "port": "input"}},
    {"name": "eb", "from": {"component": "supply", "port": "out"},
     "to": {"component": "b", "port": "input"}}
  ],
  "indicators": [
    {"name": "a_got", "target": "attribute",
     "attr": {"component": "a", "attribute": "got"}},
    {"name": "b_got", "target": "attribute",
     "attr": {"component": "b", "attribute": "got"}}
  ]
}
"#;

#[test]
fn a_multi_fixpoint_network_is_pinned_by_the_compiled_order() {
    let first = run(MULTI_FIXPOINT, 1.0);
    let second = run(MULTI_FIXPOINT, 1.0);
    // The cold start hands the sweep two zero demands, so the split has
    // no weight to distribute and the resolution settles there. Any
    // split summing to 10 would satisfy the sweep equally well: what
    // rules them out is the cold start and the compiled order, not a
    // uniqueness result.
    assert_flow(value(&first, "a_got"), 0.0, "a's pinned share");
    assert_flow(value(&first, "b_got"), 0.0, "b's pinned share");
    assert_eq!(
        value(&first, "a_got"),
        value(&second, "a_got"),
        "the pinned fixpoint reproduces run to run"
    );
    assert_eq!(
        value(&first, "b_got"),
        value(&second, "b_got"),
        "the pinned fixpoint reproduces run to run"
    );
}

// ---------------------------------------------------------------------
// The compiled registration of the margins.
// ---------------------------------------------------------------------

/// Emitting the resolved flows is not enough on its own: an active-set
/// margin has to be a compiled, *registered* guard, or nothing downstream
/// can find it.
///
/// This pins what the registration promises: which operator each margin
/// belongs to, which edge each one watches, and which attributes it
/// reads. The last is what a variable-to-margin index inverts, so getting
/// it wrong would leave that index silently blind to a moving boundary.
#[test]
fn every_active_set_margin_registers_the_attributes_it_reads() {
    let compiled = compile(&contested_supply(r#""policy": "proportional""#));
    assert_eq!(
        compiled.flow_margins.len(),
        1,
        "one registration per distribution operator"
    );
    let margins = &compiled.flow_margins[0];
    assert_eq!(margins.name, "supply.split");
    assert_eq!(
        margins.consumers,
        ["supply.out__alloc__ea", "supply.out__alloc__eb"],
        "one margin per outgoing connection, in declaration order"
    );
    assert!(
        matches!(
            compiled.explicit.get(margins.step),
            Some(CStep::Allocate(_))
        ),
        "the registered step index points at the operator in the swept table"
    );

    // The available quantity and *both* demands. A weighted split offers
    // one consumer a share of what the others left, so every margin of
    // the operator moves when any demand moves: registering per edge
    // would claim an independence that is not there.
    let names: Vec<&str> = margins
        .deps
        .iter()
        .map(|&var| compiled.var_names[var].as_str())
        .collect();
    for expected in [
        "supply.capacity",
        "supply.out__demand__ea",
        "supply.out__demand__eb",
    ] {
        assert!(
            names.contains(&expected),
            "`{expected}` is read by the margins but not registered: {names:?}"
        );
    }
    assert!(
        margins.deps.windows(2).all(|pair| pair[0] < pair[1]),
        "the dependencies are sorted and deduplicated, so an inversion \
         iterates a stable sequence: {names:?}"
    );

    // A model with no operator registers nothing, which is what lets it
    // skip the resolution entirely.
    let Some(json) = fixture("tank_01") else {
        return;
    };
    let plain = CompiledModel::compile(&Model::from_json(&json).expect("fixture JSON parses"))
        .expect("fixture compiles");
    assert!(plain.flow_margins.is_empty());
}

// ---------------------------------------------------------------------
// An active-set boundary crossed inside an integration segment.
// ---------------------------------------------------------------------

/// The available quantity is integrated (`d capacity/dt = 1`) and split
/// proportionally between two constant demands of 6 and 4.
///
/// While `capacity < 10` neither consumer is saturated: each receives a
/// fraction of a growing supply. At `capacity = 10` both reach their
/// demand at once and the active set flips to *both saturated*. That
/// instant, `t = 10`, is inside an integration segment: the only
/// scheduled date in the model is a delay at `t = 100`.
const CROSSING: &str = r#"
{
  "name": "active_set_crossing",
  "components": [
    {
      "name": "supply",
      "attributes": [
        {"name": "capacity", "kind": "float", "init": {"kind": "float", "value": 0.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "capacity",
         "channels": [{"name": "demand"}, {"name": "alloc"}]}
      ],
      "equations": [
        {"target": "capacity", "kind": "ode",
         "expr": {"op": "const", "value": {"kind": "float", "value": 1.0}}},
        {"target": "out__demand__ea", "kind": "explicit",
         "expr": {"op": "const", "value": {"kind": "float", "value": 6.0}}},
        {"target": "out__demand__eb", "kind": "explicit",
         "expr": {"op": "const", "value": {"kind": "float", "value": 4.0}}}
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
      ],
      "automata": [
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
    {"name": "ea", "from": {"component": "supply", "port": "out"},
     "to": {"component": "a", "port": "input"}},
    {"name": "eb", "from": {"component": "supply", "port": "out"},
     "to": {"component": "b", "port": "input"}}
  ],
  "indicators": [
    {"name": "a_got", "target": "attribute",
     "attr": {"component": "a", "attribute": "got"}},
    {"name": "b_got", "target": "attribute",
     "attr": {"component": "b", "attribute": "got"}}
  ]
}
"#;

#[test]
fn an_active_set_crossing_is_located_at_its_instant() {
    let result = run_with(
        CROSSING,
        EngineConfig {
            t_max: 20.0,
            journal: true,
            ..EngineConfig::default()
        },
    );
    let crossings: Vec<&JournalRecord> = result
        .journal
        .iter()
        .filter(|record| matches!(record, JournalRecord::ActiveSetCrossed { .. }))
        .collect();
    let JournalRecord::ActiveSetCrossed {
        time,
        operator,
        consumer,
        from,
        to,
    } = crossings.first().unwrap_or_else(|| {
        panic!(
            "no active-set crossing was located; the journal was {:?}",
            result.journal
        )
    })
    else {
        unreachable!("filtered above")
    };
    assert_eq!(operator, "supply.split", "the crossing names its operator");
    assert_eq!(consumer, "supply.out__alloc__ea", "and the edge that moved");
    assert_eq!(
        (*from, *to),
        (EdgeClass::Open, EdgeClass::Capped),
        "the edge left the split and was fixed at its demand"
    );
    assert!(
        (time - 10.0).abs() < 1e-6,
        "the active-set crossing was reported at t={time}, expected the \
         analytic t=10. A value at t=100 (the scheduled delay) or at the \
         horizon means the boundary was polled at the next discrete date \
         rather than located inside the segment"
    );
    // Past the crossing both consumers hold their demand and nothing
    // more: the resolution took the saturated branch.
    assert_flow(value(&result, "a_got"), 6.0, "a's saturated share");
    assert_flow(value(&result, "b_got"), 4.0, "b's saturated share");
    // And the network resolved *on* that boundary does not re-cross it:
    // the dead band is what keeps the engine from chattering on the spot.
    // One crossing means two segments, and no more.
    assert_eq!(
        (crossings.len(), result.work.segments),
        (1, 2),
        "the boundary was crossed once and the segment restarted once; \
         a larger count is the network re-crossing its own resolved \
         boundary (work was {:?})",
        result.work
    );
}

/// The same growing supply, split by **strict priority**: `b` is served
/// first (rank 1, demand 4), `a` second (rank 2, demand 6). The supply
/// starts at 1 and grows at 1 per unit time.
///
/// A priority order has three classes per edge, not two, and both of its
/// boundaries lie inside the segment: `b` fills up at `capacity = 4`
/// (`t = 3`), which is the same instant `a` starts being served at all,
/// and `a` fills up at `capacity = 10` (`t = 9`).
fn priority_crossing() -> String {
    CROSSING
        .replace("\"name\": \"active_set_crossing\"", "\"name\": \"priority_crossing\"")
        .replace(
            "{\"name\": \"capacity\", \"kind\": \"float\", \"init\": {\"kind\": \"float\", \"value\": 0.0}}",
            "{\"name\": \"capacity\", \"kind\": \"float\", \"init\": {\"kind\": \"float\", \"value\": 1.0}}",
        )
        .replace(
            "\"policy\": \"proportional\"",
            "\"policy\": \"priority\",\n         \"priorities\": [\n                        {\"to\": {\"component\": \"a\", \"port\": \"input\"}, \"value\": 2.0},\n                        {\"to\": {\"component\": \"b\", \"port\": \"input\"}, \"value\": 1.0}]",
        )
}

#[test]
fn a_priority_active_set_is_located_across_its_three_classes() {
    let result = run_with(
        &priority_crossing(),
        EngineConfig {
            t_max: 20.0,
            journal: true,
            ..EngineConfig::default()
        },
    );
    let crossings: Vec<(f64, EdgeClass, EdgeClass)> = result
        .journal
        .iter()
        .filter_map(|record| match record {
            JournalRecord::ActiveSetCrossed { time, from, to, .. } => Some((*time, *from, *to)),
            _ => None,
        })
        .collect();
    let instants: Vec<f64> = crossings.iter().map(|(t, _, _)| *t).collect();
    assert_eq!(
        instants.len(),
        2,
        "expected the two priority boundaries, got {crossings:?}"
    );
    assert!(
        (instants[0] - 3.0).abs() < 1e-6,
        "the supply fills `b` at t=3, reported at {}",
        instants[0]
    );
    assert!(
        (instants[1] - 9.0).abs() < 1e-6,
        "the supply fills `a` at t=9, reported at {}",
        instants[1]
    );
    // `a` is the lower-priority consumer: it goes from being served
    // nothing, to being served part of what it asked, to being full.
    assert_eq!(
        (crossings[0].1, crossings[0].2),
        (EdgeClass::Unserved, EdgeClass::Partial)
    );
    assert_eq!(
        (crossings[1].1, crossings[1].2),
        (EdgeClass::Partial, EdgeClass::Full)
    );
    assert_flow(value(&result, "a_got"), 6.0, "a is served in full at last");
    assert_flow(
        value(&result, "b_got"),
        4.0,
        "b was served first throughout",
    );
}

// ---------------------------------------------------------------------
// The dead band of an active-set margin.
// ---------------------------------------------------------------------

/// A margin located at a crossing leaves the state *on* its boundary, so
/// the margin of the freshly frozen active set starts at zero. Without a
/// dead band the next segment would re-cross it at once and the engine
/// would chatter on the spot.
///
/// The band is the **flow tolerance**, deliberately, and it must sit
/// above the event-location tolerance: the resolution only promises the
/// flows to within the flow tolerance, so a residual smaller than that is
/// not a crossing. This test pins the ordering of the two constants by
/// evaluating the same margin under both.
#[test]
fn a_residual_below_the_flow_tolerance_does_not_re_cross() {
    // Two consumers demanding 6 and 4 out of an available quantity a
    // hair past 10: `a` is offered 0.6 x available, so it overshoots its
    // demand of 6 by 5e-10, which is above the event-location tolerance
    // (1e-10) and below the flow tolerance (1e-9).
    let residual = 5e-10;
    let available = 10.0 + residual / 0.6;
    let demands = [6.0, 4.0];
    let policy = CPolicy::Proportional;
    // Neither consumer is saturated below the boundary: that is the
    // frozen active set whose margin the crossing would end.
    let classes = [EdgeClass::Open, EdgeClass::Open];

    let event_tolerance = 1e-10;
    let at_event_tolerance =
        edge_margin(&policy, available, &demands, &classes, 0, event_tolerance);
    assert!(
        at_event_tolerance >= 0.0,
        "banded only by the event-location tolerance, a residual of \
         {residual} re-crosses at once (margin {at_event_tolerance})"
    );

    let at_flow_tolerance = edge_margin(
        &policy,
        available,
        &demands,
        &classes,
        0,
        flow_band(available, FLOW_TOLERANCE),
    );
    assert!(
        at_flow_tolerance < 0.0,
        "at the chosen flow tolerance the same residual is not a crossing \
         (margin {at_flow_tolerance})"
    );
}

// ---------------------------------------------------------------------
// The margin index covers the operators too.
// ---------------------------------------------------------------------

/// The compiled margin index inverts [`CFlowMargins::deps`] entry for
/// entry, so it is a complete answer to "which margins read this
/// attribute" and not merely the watched half of one.
///
/// It has no scan site of its own to narrow: every operator of the sweep
/// contributes its edges to every segment (there is no arming filter to
/// apply), and the margins themselves are evaluated inside the solver
/// callbacks, which the index deliberately leaves alone because a root
/// finder needs every bracketed value at every probe. What the inversion
/// buys is the guarantee that a *complete* dependency answer exists,
/// which is what makes it sound to skip work on the strength of "nothing
/// this margin reads has moved".
#[test]
fn the_margin_index_inverts_the_operator_dependencies() {
    let compiled = compile(CROSSING);
    let index = &compiled.margin_index;
    assert!(
        !compiled.flow_margins.is_empty(),
        "the model must carry an operator for this to test anything"
    );
    for (operator, margins) in compiled.flow_margins.iter().enumerate() {
        assert!(
            !margins.deps.is_empty(),
            "operator `{}` reads nothing, which cannot be right",
            margins.name
        );
        for &var in &margins.deps {
            assert!(
                index.flow_by_var[var].contains(&operator),
                "operator `{}` reads `{}` but is not indexed under it",
                margins.name,
                compiled.var_names[var]
            );
        }
    }
    for (var, operators) in index.flow_by_var.iter().enumerate() {
        assert!(
            operators.windows(2).all(|w| w[0] < w[1]),
            "the operator index must be ascending and deduplicated"
        );
        for &operator in operators {
            assert!(
                compiled.flow_margins[operator].deps.contains(&var),
                "operator `{}` is indexed under `{}` without reading it",
                compiled.flow_margins[operator].name,
                compiled.var_names[var]
            );
        }
    }
}

// ---------------------------------------------------------------------
// Models with no continuous flow are untouched.
// ---------------------------------------------------------------------

/// The cross-validation corpus is absent from some checkouts of this
/// project. A test that pins a fixture's behaviour reports that it did
/// not run rather than failing, so the suite stays green either way.
fn fixture(name: &str) -> Option<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/tests/validation/fixtures")
        .join(format!("{name}.json"));
    match std::fs::read_to_string(&path) {
        Ok(json) => Some(json),
        Err(_) => {
            eprintln!("skipped: no cross-validation corpus at {}", path.display());
            None
        }
    }
}

/// A model carrying no distribution operator has no active set to settle,
/// so it never enters the resolution: it runs the single ordered pass it
/// ran before this unit, at every one of the sites it ran at.
///
/// The counted work below was recorded on the commit that precedes this
/// unit. It is pinned rather than merely compared to itself, because
/// "unchanged" is a claim about the previous engine and a self-comparison
/// cannot make it.
#[test]
fn a_model_with_no_continuous_flow_is_byte_identical() {
    let expected = [
        // (fixture, t_max, explicit_evaluations, segments, margin_evaluations)
        ("tank_01", 40.0, 9394u64, 6u64, 6946u64),
        ("pdmp_001", 1000.0, 233_376, 90, 1_034_946),
        ("heated_room_s3", 100.0, 23552, 16, 34852),
        ("tank_02", 1000.0, 142_437, 6, 412_795),
        ("pool_02", 1000.0, 242_800, 471, 886_355),
    ];
    for (name, t_max, explicit, segments, margins) in expected {
        let Some(json) = fixture(name) else { continue };
        let model = Model::from_json(&json).expect("fixture JSON parses");
        let compiled = CompiledModel::compile(&model).expect("fixture compiles");
        let config = EngineConfig {
            t_max,
            ..EngineConfig::default()
        };
        let work = Engine::new(&compiled, config).unwrap().run().unwrap().work;
        assert_eq!(work.flow_sweeps, 0, "{name} runs no flow resolution");
        assert_eq!(
            work.allocation_capping_passes, 0,
            "{name} runs no distribution operator"
        );
        assert_eq!(
            (
                work.explicit_evaluations,
                work.segments,
                work.margin_evaluations
            ),
            (explicit, segments, margins),
            "{name}: counted work moved against the pre-unit baseline"
        );
    }
}

// ---------------------------------------------------------------------
// Snapshot and restore.
// ---------------------------------------------------------------------

/// The resolution keeps nothing outside the attribute vector: the frozen
/// active set is *derived* from the resolved state at each segment start,
/// never carried across one. A restored snapshot therefore replays
/// identically, which is the property a warm start would have broken.
///
/// The model is the one whose active set changes mid-segment, so the
/// replay has to reproduce a located crossing and the resolution that
/// follows it, not merely a static split.
#[test]
fn the_resolved_state_survives_snapshot_and_restore() {
    let compiled = compile(CROSSING);
    let config = || EngineConfig {
        t_max: 20.0,
        ..EngineConfig::default()
    };
    let read = |engine: &Engine<'_>| {
        (
            engine.current_time().to_bits(),
            engine
                .attribute("supply.capacity")
                .map(|v| format!("{v:?}")),
            engine.attribute("a.got").map(|v| format!("{v:?}")),
            engine.attribute("b.got").map(|v| format!("{v:?}")),
        )
    };

    let mut engine = Engine::new(&compiled, config()).unwrap();
    let snapshot = engine.snapshot();
    let at_snapshot = read(&engine);

    // Run once from the snapshot state, through the crossing at t = 10.
    while engine.step().unwrap().is_some() {}
    let reference = read(&engine);
    assert_ne!(
        reference, at_snapshot,
        "the run must actually move, or the replay proves nothing"
    );

    // Rewind and replay: bit-for-bit the same, crossing included.
    engine.restore(&snapshot);
    assert_eq!(
        read(&engine),
        at_snapshot,
        "a restore reinstates the resolved flows exactly"
    );
    while engine.step().unwrap().is_some() {}
    assert_eq!(
        read(&engine),
        reference,
        "the replay from a restored snapshot reproduces the run exactly"
    );

    // And an engine rebuilt on the snapshot, the seam a stateful facade
    // uses, reaches the same place.
    let mut rebuilt = Engine::from_snapshot(&compiled, config(), &snapshot);
    while rebuilt.step().unwrap().is_some() {}
    assert_eq!(
        read(&rebuilt),
        reference,
        "an engine rebuilt on the snapshot replays the same trajectory"
    );
}

// ---------------------------------------------------------------------
// Priority allocation combined with surplus return: refused.
// ---------------------------------------------------------------------

/// A supply whose consumers return the surplus they did not use: each
/// consumer's demand is what it still needs *net of what this very
/// operator already allocated it*.
///
/// The descending resolution needs the ordered pass to over-estimate
/// every delivery. That holds for the weighted policies; nobody has shown
/// it for a strict priority order, where a consumer moving up the served
/// prefix can raise a later delivery rather than lower it. The
/// composition is therefore refused at build time.
fn surplus_return(policy: &str) -> String {
    contested_supply(policy)
        .replace("\"name\": \"contested\"", "\"name\": \"surplus_return\"")
        .replace("out__alloc__eb", "out__alloc__ea__PLACEHOLDER")
        .replace("out__alloc__ea\"", "out__alloc__eb\"")
        .replace("out__alloc__ea__PLACEHOLDER", "out__alloc__ea")
}

#[test]
fn priority_with_surplus_return_is_refused_at_build_time() {
    // `contested_supply` already returns surplus: each demand reads an
    // allocated quantity of the same operator. Under a weighted policy
    // it resolves; under a strict priority it is refused.
    let error = refusal(&contested_supply(&priorities(1.0, 2.0)));
    assert!(
        matches!(error, ModelError::AllocationPrioritySurplusReturn { .. }),
        "expected the priority/surplus-return refusal, got {error:?}"
    );
    let message = error.to_string();
    for fragment in ["supply", "split", "priority", "surplus"] {
        assert!(
            message.contains(fragment),
            "the refusal names the composition (`{fragment}` missing): {message}"
        );
    }
    // The symmetric wiring (each demand reading the *other* edge's
    // allocation) is the same composition and is refused the same way.
    let error = refusal(&surplus_return(&priorities(1.0, 2.0)));
    assert!(
        matches!(error, ModelError::AllocationPrioritySurplusReturn { .. }),
        "expected the priority/surplus-return refusal, got {error:?}"
    );
}

/// The `priority` policy fragment: the lowest rank is served first.
fn priorities(rank_a: f64, rank_b: f64) -> String {
    format!(
        r#""policy": "priority", "priorities": [
            {{"to": {{"component": "a", "port": "input"}}, "value": {rank_a}}},
            {{"to": {{"component": "b", "port": "input"}}, "value": {rank_b}}}]"#
    )
}

/// A priority supply with **no** surplus return still resolves: the
/// refusal is on the composition, not on the policy.
const PRIORITY_NO_RETURN: &str = r#"
{
  "name": "priority_plain",
  "components": [
    {
      "name": "supply",
      "attributes": [
        {"name": "capacity", "kind": "float", "init": {"kind": "float", "value": 5.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "capacity",
         "channels": [{"name": "demand"}, {"name": "alloc"}]}
      ],
      "equations": [
        {"target": "out__demand__ea", "kind": "explicit",
         "expr": {"op": "const", "value": {"kind": "float", "value": 6.0}}},
        {"target": "out__demand__eb", "kind": "explicit",
         "expr": {"op": "const", "value": {"kind": "float", "value": 4.0}}}
      ],
      "allocations": [
        {"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
         "available": {"op": "attr",
           "attr": {"component": "supply", "attribute": "capacity"}},
         "policy": "priority",
         "priorities": [
           {"to": {"component": "a", "port": "input"}, "value": 2.0},
           {"to": {"component": "b", "port": "input"}, "value": 1.0}]}
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
    {"name": "ea", "from": {"component": "supply", "port": "out"},
     "to": {"component": "a", "port": "input"}},
    {"name": "eb", "from": {"component": "supply", "port": "out"},
     "to": {"component": "b", "port": "input"}}
  ],
  "indicators": [
    {"name": "a_got", "target": "attribute",
     "attr": {"component": "a", "attribute": "got"}},
    {"name": "b_got", "target": "attribute",
     "attr": {"component": "b", "attribute": "got"}}
  ]
}
"#;

#[test]
fn a_priority_supply_without_surplus_return_still_resolves() {
    parse(PRIORITY_NO_RETURN)
        .validate()
        .expect("a priority split with no surplus return is accepted");
    let result = run(PRIORITY_NO_RETURN, 1.0);
    // `b` has the lower rank: it is served in full (4), `a` takes the
    // remaining 1.
    assert_flow(value(&result, "b_got"), 4.0, "b is served first");
    assert_flow(value(&result, "a_got"), 1.0, "a takes the remainder");
}
