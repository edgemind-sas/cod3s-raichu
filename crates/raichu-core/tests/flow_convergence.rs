//! **Convergence policy of the continuous-flow resolution**: a network
//! that will not settle stops in bounded work and says why.
//!
//! The resolution converges on two levels and carries one budget per
//! level, both counted in sweeps. The combinatorial budget is *derived*
//! from the compiled network (`active_set_budget`), because the
//! descending cold-start sequence changes at most two classes per edge;
//! the numeric budget is the constant `FLOW_SWEEP_BUDGET`, because
//! nothing in the model's size sizes a rate of convergence.
//!
//! Three properties are pinned here, and they are separable:
//!
//! - **Bounded work.** Every stall path ends in a diagnostic, never in a
//!   last iterate silently standing and never in a spin. The bound is
//!   asserted as a *round count* derived from the model, not as a
//!   wall-clock timeout, so a slower machine cannot turn the assertion
//!   into a flake and a regression cannot hide behind a generous clock.
//! - **One payload for every path.** A slow monotone sequence and a long
//!   cycle exhaust a budget without ever matching the two-cycle test. If
//!   only the detected two-cycle named the moving edges, the two
//!   commonest stalls would report nothing actionable.
//! - **Nothing carries across a resolution.** Under-relaxation is a
//!   local of the resolution that latches it. A relaxation surviving a
//!   segment would make the answer depend on what the engine resolved
//!   before, and Monte-Carlo results would stop being invariant in the
//!   thread count (pinned on the driver side, in `raichu-montecarlo`).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::engine::{active_set_budget, FLOW_SWEEP_BUDGET};
use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError, FlowStall};
use raichu_expr::Value;
use raichu_model::Model;

/// Wrap a model body in the format envelope: both features these models
/// use are declared, and the declaration is verified against the body.
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

/// The engine's own reference to the allocated quantity of the single
/// consumer `a`: the per-connection channel attribute the operator
/// writes, which is what makes these networks feed back on themselves.
const ALLOCATED: &str = r#"{"op": "attr",
    "attr": {"component": "supply", "attribute": "out__alloc__ea"}}"#;

/// One supply, one consumer, one proportional split, and a demand the
/// caller writes: the smallest shape in which a flow resolution can fail
/// to settle.
///
/// `extra` is spliced into the consumer component (an automaton, when a
/// test needs discrete epochs to re-resolve at).
fn one_consumer(name: &str, capacity: f64, demand: &str, extra: &str) -> String {
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
      ]{extra}
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

/// Two chained delays, firing at t = 1 and t = 2. Neither touches the
/// flow network: they exist only to make the engine resolve the *same*
/// network again, at two later instants.
const TWO_EPOCHS: &str = r#",
      "automata": [
        {"name": "clock", "states": ["s0", "s1", "s2"], "init": "s0",
         "transitions": [
           {"name": "tick", "source": "s0", "targets": ["s1"],
            "distrib": "delay", "time": 1.0},
           {"name": "tock", "source": "s1", "targets": ["s2"],
            "distrib": "delay", "time": 1.0}]}
      ]"#;

/// A demand that **collapses when it is served**: `d = 6 − 1.5·a`, with
/// the operator clamping a negative demand to zero.
///
/// Undamped the iteration is a pure alternation. From the cold start the
/// consumer asks for 6 and gets it; holding 6 it asks for −3, which the
/// operator reads as 0 and serves nothing; holding nothing it asks for 6
/// again. The multiplier of the linearised map is −1.5, so no amount of
/// patience settles it.
///
/// Damped at one half the same map contracts by |1 − w(1 − μ)| = 0.25 per
/// sweep onto the fixpoint `6/2.5 = 2.4`.
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
/// at most 3, 1 once it holds more.
///
/// The map has no fixed point at all: a state below the threshold is
/// answered by an allocation above it and conversely, and the jump means
/// no convex combination sits still either. Under-relaxation halves the
/// swing and changes nothing about the alternation, which is the case the
/// diagnostic exists for.
fn jumping_demand() -> String {
    format!(
        r#"{{"op": "if",
             "cond": {{"op": "cmp", "cmp": "gt", "lhs": {ALLOCATED},
                      "rhs": {{"op": "const", "value": {{"kind": "float", "value": 3.0}}}}}},
             "then": {{"op": "const", "value": {{"kind": "float", "value": 1.0}}}},
             "otherwise": {{"op": "const", "value": {{"kind": "float", "value": 9.0}}}}}}"#
    )
}

/// A demand that creeps: `d = 0.01 + 0.999·a`, converging monotonically
/// on 10 at a rate of 0.999 per sweep, so about 13 800 sweeps to the flow
/// tolerance. The active set settles on the first sweep and never moves
/// again: nothing here is combinatorial, and nothing cycles.
fn creeping_demand() -> String {
    format!(
        r#"{{"op": "add", "args": [
             {{"op": "const", "value": {{"kind": "float", "value": 0.01}}}},
             {{"op": "mul", "args": [
               {{"op": "const", "value": {{"kind": "float", "value": 0.999}}}},
               {ALLOCATED}]}}]}}"#
    )
}

fn config(t_max: f64) -> EngineConfig {
    EngineConfig {
        t_max,
        ..EngineConfig::default()
    }
}

/// The error a model raises when it is built (its initial resolution is
/// the one that stalls).
fn build_failure(body: &str) -> EngineError {
    let compiled = compile(body);
    match Engine::new(&compiled, config(1.0)) {
        Err(error) => error,
        Ok(_) => panic!("the engine built: the resolution settled where it must not"),
    }
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

// ---------------------------------------------------------------------
// A network that oscillates between two allocations.
// ---------------------------------------------------------------------

/// The alternation is reported as what it is: a conflict between two
/// allocations, each of which justifies the other, not a numerical
/// accident. The run stops on a budget, and the diagnostic names the
/// component and the flow of the edge that was oscillating.
#[test]
fn an_oscillating_network_stops_on_a_budget_and_names_its_edges() {
    let body = one_consumer("oscillating", 5.0, &jumping_demand(), "");
    let error = build_failure(&body);
    let EngineError::FlowNotConverged {
        sweeps,
        cause,
        ref moving,
        ..
    } = error
    else {
        panic!("expected a flow non-convergence diagnostic, got {error:?}");
    };
    assert_eq!(
        cause,
        FlowStall::TwoCycle,
        "the alternation was seen for what it is: {error}"
    );
    assert_eq!(
        moving, "supply.split[supply.out__alloc__ea]",
        "the diagnostic names the operator and the flow that oscillates"
    );
    // Bounded, and bounded by the *derived* budget: the combinatorial
    // budget twice (it is granted once more when under-relaxation
    // latches) plus the numeric one, plus the sweep each budget is
    // exceeded on.
    let budget = active_set_budget(&compile(&body));
    assert!(
        sweeps <= 2 * (budget + 1) + FLOW_SWEEP_BUDGET + 1,
        "the resolution spent {sweeps} sweeps against a declared bound of \
         {}",
        2 * (budget + 1) + FLOW_SWEEP_BUDGET + 1
    );
    // And the message reads as a diagnostic, not as an internal state
    // dump: it says where, what, and which flow.
    let message = error.to_string();
    for fragment in ["did not settle", "supply.split", "out__alloc__ea"] {
        assert!(
            message.contains(fragment),
            "the message is missing `{fragment}`: {message}"
        );
    }
}

// ---------------------------------------------------------------------
// The same pathology at scale: a few hundred edges.
// ---------------------------------------------------------------------

/// `n` consumers on one supply, each with the jumping demand above,
/// gated so that the network is well posed at t = 0 and pathological from
/// t = 1: the engine must therefore *build*, and it is the resolution at
/// the discrete epoch that stalls. That is what lets the test read the
/// sweep counter, which a failure during the build would take with it.
fn wide_network(n: usize) -> String {
    let mut demands = String::new();
    let mut consumers = String::new();
    let mut connections = String::new();
    for edge in 0..n {
        demands.push_str(&format!(
            r#"        {{"target": "out__demand__e{edge}", "kind": "explicit",
          "expr": {{"op": "if",
            "cond": {{"op": "cmp", "cmp": "gt",
              "lhs": {{"op": "attr", "attr": {{"component": "supply",
                        "attribute": "out__alloc__e{edge}"}}}},
              "rhs": {{"op": "attr", "attr": {{"component": "supply",
                        "attribute": "gate"}}}}}},
            "then": {{"op": "const", "value": {{"kind": "float", "value": 1.0}}}},
            "otherwise": {{"op": "const", "value": {{"kind": "float", "value": 9.0}}}}}}}},
"#
        ));
        consumers.push_str(&format!(
            r#"    {{
      "name": "c{edge}",
      "attributes": [
        {{"name": "got", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "ports": [{{"name": "input", "dir": "in"}}],
      "equations": [
        {{"target": "got", "kind": "explicit",
          "expr": {{"op": "port_agg", "agg": "sum", "channel": "alloc",
                    "port": {{"component": "c{edge}", "port": "input"}}}}}}
      ]
    }},
"#
        ));
        connections.push_str(&format!(
            r#"    {{"name": "e{edge}", "from": {{"component": "supply", "port": "out"}},
      "to": {{"component": "c{edge}", "port": "input"}}}},
"#
        ));
    }
    let capacity = 5.0 * n as f64;
    format!(
        r#"
{{
  "name": "wide",
  "components": [
    {{
      "name": "supply",
      "attributes": [
        {{"name": "capacity", "kind": "float",
          "init": {{"kind": "float", "value": {capacity}}}}},
        {{"name": "gate", "kind": "float",
          "init": {{"kind": "float", "value": 1000000000.0}}}}
      ],
      "ports": [
        {{"name": "out", "dir": "out", "attr": "capacity",
          "channels": [{{"name": "demand"}}, {{"name": "alloc"}}]}}
      ],
      "equations": [
        {{"target": "gate", "kind": "explicit",
          "expr": {{"op": "if",
            "cond": {{"op": "cmp", "cmp": "gt", "lhs": {{"op": "time"}},
              "rhs": {{"op": "const", "value": {{"kind": "float", "value": 0.5}}}}}},
            "then": {{"op": "const", "value": {{"kind": "float", "value": 3.0}}}},
            "otherwise": {{"op": "const",
              "value": {{"kind": "float", "value": 1000000000.0}}}}}}}},
{demands}        {{"target": "capacity", "kind": "explicit",
          "expr": {{"op": "const", "value": {{"kind": "float", "value": {capacity}}}}}}}
      ],
      "allocations": [
        {{"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
          "available": {{"op": "attr",
            "attr": {{"component": "supply", "attribute": "capacity"}}}},
          "policy": "proportional"}}
      ]
    }},
{consumers}    {{
      "name": "ctrl",
      "attributes": [],
      "ports": [],
      "automata": [
        {{"name": "clock", "states": ["s0", "s1"], "init": "s0",
         "transitions": [
           {{"name": "tick", "source": "s0", "targets": ["s1"],
            "distrib": "delay", "time": 1.0}}]}}
      ]
    }}
  ],
  "connections": [
{connections}    {{"name": "keep", "from": {{"component": "supply", "port": "out"}},
      "to": {{"component": "c0", "port": "input"}}}}
  ],
  "indicators": []
}}
"#
    )
}

/// A network of a few hundred edges that will not settle raises its
/// diagnostic **within its declared round count**, and the count is
/// declared by the model, not by the clock: a wall-clock timeout would
/// pass on a fast machine and flake on a loaded one, and would say
/// nothing about whether the budget is the thing doing the bounding.
#[test]
fn a_wide_non_converging_network_stops_within_its_declared_round_count() {
    let n = 300;
    let body = wide_network(n);
    let compiled = compile(&body);
    let budget = active_set_budget(&compiled);
    // 2 per edge for the saturation classes, 1 per conditional (one per
    // demand, one for the gate), 2 for the candidate and its
    // confirmation. The extra edge is the duplicate connection `keep`.
    assert_eq!(
        budget,
        2 * (n + 1) + (n + 1) + 2,
        "the budget is derived from the compiled network, not chosen"
    );
    let mut engine = Engine::new(&compiled, config(5.0)).expect("the network is well posed at t=0");
    let before = engine.work().flow_sweeps;
    let error = loop {
        match engine.step() {
            Err(error) => break error,
            Ok(Some(_)) => {}
            Ok(None) => panic!("the run finished: the pathological epoch never happened"),
        }
    };
    assert!(
        matches!(error, EngineError::FlowNotConverged { .. }),
        "expected a flow non-convergence diagnostic, got {error:?}"
    );
    let spent = engine.work().flow_sweeps - before;
    let declared = 2 * (budget as u64 + 1) + FLOW_SWEEP_BUDGET as u64 + 1;
    assert!(
        spent <= declared,
        "the stalling resolution spent {spent} sweeps against a declared \
         bound of {declared}"
    );
    // The bound is not vacuous: a resolution that gave up after a
    // handful of sweeps would satisfy it too.
    assert!(
        spent > budget as u64,
        "only {spent} sweeps were spent: the combinatorial budget of \
         {budget} was never actually explored"
    );
}

// ---------------------------------------------------------------------
// A stall with no cycle in it at all.
// ---------------------------------------------------------------------

/// A monotone sequence creeping toward its limit exhausts the numeric
/// budget without ever matching the two-cycle test, and it must still say
/// which flows were moving. This is the path that a diagnostic built
/// around cycle detection alone would leave mute.
#[test]
fn a_slow_monotone_sequence_exhausts_the_flow_budget_and_names_its_edges() {
    let body = one_consumer("creeping", 1000.0, &creeping_demand(), "");
    let error = build_failure(&body);
    let EngineError::FlowNotConverged {
        sweeps,
        cause,
        ref moving,
        ..
    } = error
    else {
        panic!("expected a flow non-convergence diagnostic, got {error:?}");
    };
    assert_eq!(
        cause,
        FlowStall::Quantities,
        "nothing cycles here and the saturation pattern never moves: \
         {error}"
    );
    assert_eq!(
        moving, "supply.split[supply.out__alloc__ea]",
        "the monotone stall names its moving edge exactly as a cycle does"
    );
    // The numeric budget did the bounding, so the count sits just past
    // it rather than at the combinatorial budget.
    assert!(
        sweeps <= FLOW_SWEEP_BUDGET + 2,
        "{sweeps} sweeps were spent against a numeric budget of \
         {FLOW_SWEEP_BUDGET}"
    );
}

// ---------------------------------------------------------------------
// Under-relaxation.
// ---------------------------------------------------------------------

/// The alternation of `d = 6 − 1.5·a` is absorbed: damping is latched on
/// the first return to a state held two sweeps earlier, and the sequence
/// then contracts onto 2.4. Without it the resolution would spend both
/// budgets and refuse a network that has a perfectly good answer.
#[test]
fn under_relaxation_settles_a_network_that_would_two_cycle() {
    let body = one_consumer("damped", 100.0, &collapsing_demand(), "");
    let compiled = compile(&body);
    let result = Engine::new(&compiled, config(1.0))
        .expect("the damped resolution settles")
        .run()
        .expect("and the run completes");
    let got = last_value(&result, "a_got");
    assert!(
        (got - 2.4).abs() <= 1e-8,
        "the resolution settled on {got}, not on the fixpoint 6/2.5 = 2.4 \
         of the damped map"
    );
    // The undamped sequence alternates between 6 and 0 forever, so
    // reaching 2.4 at all is the damping. It costs sweeps: an
    // alternation-free network of this shape settles in three, this one
    // takes 21 (three to enter the alternation and detect it, then the
    // damped contraction at 1/4 per sweep down to the flow tolerance).
    // The bound below is the numeric budget itself: a damped iteration
    // that needed more than that would be refused rather than settled.
    let sweeps = result.work.flow_sweeps;
    assert!(
        sweeps > 3 && sweeps <= FLOW_SWEEP_BUDGET as u64,
        "{sweeps} sweeps: either the alternation was never entered (so \
         this model no longer exercises the relaxation) or the damped \
         contraction no longer fits the numeric budget"
    );
}

/// The relaxation is a local of the resolution that latched it. If it
/// were engine state, the second resolution of an unchanged network would
/// start damped and cost a different number of sweeps than the first.
///
/// The model resolves three times on the same network (at the
/// initialization axiom, and at the two delays that fire at t = 1 and
/// t = 2) and the three resolutions must cost exactly the same.
#[test]
fn the_relaxation_state_does_not_survive_a_resolution() {
    let body = one_consumer("damped_epochs", 100.0, &collapsing_demand(), TWO_EPOCHS);
    let compiled = compile(&body);
    let mut engine = Engine::new(&compiled, config(3.0)).expect("the damped resolution settles");
    let initial = engine.work().flow_sweeps;
    let mut per_epoch = Vec::new();
    let mut previous = initial;
    while engine.step().expect("the run does not stall").is_some() {
        let now = engine.work().flow_sweeps;
        per_epoch.push(now - previous);
        previous = now;
    }
    assert_eq!(
        per_epoch.len(),
        2,
        "expected the two delayed epochs, got {per_epoch:?}"
    );
    // 21 as measured: three sweeps to enter the alternation and detect
    // it, then the damped contraction. What matters is only that the
    // count is above the three an alternation-free network takes, so
    // that the equality below is comparing damped resolutions.
    assert!(
        initial > 3,
        "the initial resolution took {initial} sweeps: it never entered \
         the alternation, so this model no longer exercises the relaxation"
    );
    assert_eq!(
        per_epoch,
        vec![initial, initial],
        "the same network resolved again cost {per_epoch:?} sweeps against \
         {initial} the first time: a relaxation that survived the \
         resolution would start the next one damped"
    );
}

// ---------------------------------------------------------------------
// A failure raised inside the solver, where the system type is
// infallible.
// ---------------------------------------------------------------------

/// A flow whose available quantity is unusable only inside a window
/// narrower than the event scan grid: `1/0` while the level sits within
/// 1e-7 of the boundary the watched transition is located at, and a
/// perfectly ordinary 5 everywhere else.
///
/// Nothing on the interior scan grid (a sixteenth of a step of at most
/// 0.1, so 6.25e-3 apart on a level falling at 1 per unit time) can land
/// in a window of 2e-7. The bisection, on the other hand, must: it
/// halves its bracket down to 1e-10 around the crossing, so its last
/// probes are inside the window by construction.
const MID_BISECTION: &str = r#"
{
  "name": "mid_bisection",
  "components": [
    {
      "name": "tank",
      "attributes": [
        {"name": "level", "kind": "float", "init": {"kind": "float", "value": 2.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "level",
         "channels": [{"name": "demand"}, {"name": "alloc"}]}
      ],
      "equations": [
        {"target": "level", "kind": "ode",
         "expr": {"op": "const", "value": {"kind": "float", "value": -1.0}}},
        {"target": "out__demand__ea", "kind": "explicit",
         "expr": {"op": "const", "value": {"kind": "float", "value": 1.0}}}
      ],
      "allocations": [
        {"name": "split", "port": "out", "demand": "demand", "allocated": "alloc",
         "available": {"op": "if",
           "cond": {"op": "bool", "bool_op": "and", "args": [
             {"op": "cmp", "cmp": "lt",
              "lhs": {"op": "attr", "attr": {"component": "tank", "attribute": "level"}},
              "rhs": {"op": "const", "value": {"kind": "float", "value": 0.4730001}}},
             {"op": "cmp", "cmp": "gt",
              "lhs": {"op": "attr", "attr": {"component": "tank", "attribute": "level"}},
              "rhs": {"op": "const", "value": {"kind": "float", "value": 0.4729999}}}]},
           "then": {"op": "div",
             "lhs": {"op": "const", "value": {"kind": "float", "value": 1.0}},
             "rhs": {"op": "const", "value": {"kind": "float", "value": 0.0}}},
           "otherwise": {"op": "const", "value": {"kind": "float", "value": 5.0}}},
         "policy": "proportional"}
      ],
      "automata": [
        {"name": "watch", "states": ["run", "done"], "init": "run",
         "transitions": [
           {"name": "hit", "source": "run", "targets": ["done"], "distrib": "watched",
            "guard": {"op": "cmp", "cmp": "le",
              "lhs": {"op": "attr", "attr": {"component": "tank", "attribute": "level"}},
              "rhs": {"op": "const", "value": {"kind": "float", "value": 0.473}}}}]}
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
    {"name": "ea", "from": {"component": "tank", "port": "out"},
     "to": {"component": "a", "port": "input"}}
  ],
  "indicators": [
    {"name": "a_got", "target": "attribute",
     "attr": {"component": "a", "attribute": "got"}}
  ]
}
"#;

/// The solver's system type is infallible: a callback has no way to
/// return a failure. It stashes it instead, and the stash is re-raised
/// once integration returns. Without that, a flow failing between two
/// bisection probes would be swallowed and the run would carry on from a
/// state nobody could defend.
///
/// **The time the diagnostic carries is the evaluation point**, not a
/// located instant: mid-bisection there is no located instant to name,
/// only a probe of a bracket. It is reported as it stands rather than
/// rewritten to the segment's committed time, which would name a state
/// that never failed. The two are asserted apart below.
#[test]
fn a_flow_failing_mid_bisection_reports_through_the_error_stash() {
    let compiled = compile(MID_BISECTION);
    let mut engine = Engine::new(&compiled, config(5.0)).expect("engine builds");
    let error = loop {
        match engine.step() {
            Err(error) => break error,
            Ok(Some(_)) => {}
            Ok(None) => panic!("the run finished: the failing window was never probed"),
        }
    };
    let EngineError::TypeError { time, ref detail } = error else {
        panic!("expected the flow-input diagnostic, got {error:?}");
    };
    assert!(
        detail.contains("split") && detail.contains("available quantity"),
        "the stashed error kept its own detail: {detail}"
    );
    // The probe sat inside the 2e-7 window, which nothing on the event
    // scan grid can reach: this is a bisection probe.
    assert!(
        (time - 1.527).abs() < 1e-6,
        "the failure is reported at t={time}, outside the window the \
         bisection is the only thing that can probe"
    );
    // And it is not the engine's committed time: that never advanced,
    // because the segment never completed.
    assert!(
        engine.current_time() < time,
        "the committed time is {} and the diagnostic says {time}: a \
         diagnostic rewritten to the committed time would name a state \
         that never failed",
        engine.current_time()
    );
}

/// The same tank with **no watched transition**, so nothing is ever
/// bisected, and a failing window centred on a dense-sample instant that
/// no scan point can reach: `t = 1.2345678` is not a multiple of the
/// interior scan spacing, and the window is 2e-7 wide.
///
/// The dense-sample callback is the second infallible seam of the
/// integration: it runs while the solver holds the system mutably, so it
/// cannot reach the system's stash and needs one of its own. A sample
/// instant is not necessarily an instant any other callback visits, so
/// without that stash the failure is not merely reported late, it is
/// never reported at all.
fn sample_instant_failure() -> String {
    MID_BISECTION
        .replace(
            "\"name\": \"mid_bisection\"",
            "\"name\": \"sample_instant\"",
        )
        .replace("0.4730001", "0.7655322")
        .replace("0.4729999", "0.7653322")
        // Drop the watched automaton: with nothing to locate there is no
        // bisection, and the sample callback is the only thing that can
        // land inside the window.
        .replace(
            r#""automata": [
        {"name": "watch", "states": ["run", "done"], "init": "run",
         "transitions": [
           {"name": "hit", "source": "run", "targets": ["done"], "distrib": "watched",
            "guard": {"op": "cmp", "cmp": "le",
              "lhs": {"op": "attr", "attr": {"component": "tank", "attribute": "level"}},
              "rhs": {"op": "const", "value": {"kind": "float", "value": 0.473}}}}]}
      ]"#,
            r#""automata": []"#,
        )
}

/// A failure at a dense-sample instant reaches the caller. The sample
/// callback cannot return it, so it stashes it, exactly as the system
/// does; without that stash the run would finish on a state nobody
/// evaluated successfully.
#[test]
fn an_explicit_failure_at_a_dense_sample_instant_is_not_swallowed() {
    let compiled = compile(&sample_instant_failure());
    let error = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 2.0,
            samples: vec![1.234_567_8],
            ..EngineConfig::default()
        },
    )
    .expect("engine builds")
    .run()
    .expect_err("the failing sample instant must not be swallowed");
    let EngineError::TypeError { time, ref detail } = error else {
        panic!("expected the flow-input diagnostic, got {error:?}");
    };
    assert!(
        detail.contains("split") && detail.contains("available quantity"),
        "the stashed error kept its own detail: {detail}"
    );
    assert!(
        (time - 1.234_567_8).abs() < 1e-6,
        "the failure is reported at t={time}, not at the sample instant"
    );
}

// ---------------------------------------------------------------------
// The derived budget.
// ---------------------------------------------------------------------

/// The combinatorial budget reads the compiled network rather than a
/// constant, and this pins both halves of the reading: two rounds per
/// edge, because an edge under a priority order has three classes and so
/// two class changes to spend, and one per branch decision, because a
/// minimum or a conditional that keeps swapping holds the resolution up
/// exactly as a saturation that keeps flipping does.
#[test]
fn the_budget_is_derived_from_the_compiled_network() {
    let plain = one_consumer(
        "plain",
        10.0,
        r#"{"op": "const", "value": {"kind": "float", "value": 1.0}}"#,
        "",
    );
    // One edge, no branch decision in either the demand or the available
    // quantity: two class changes plus the two confirmation sweeps.
    assert_eq!(active_set_budget(&compile(&plain)), 4);
    // The same network with a conditional demand admits one more.
    let branching = one_consumer("branching", 10.0, &jumping_demand(), "");
    assert_eq!(active_set_budget(&compile(&branching)), 5);
}
