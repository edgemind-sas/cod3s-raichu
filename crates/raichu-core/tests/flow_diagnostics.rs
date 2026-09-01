//! Flow-graph diagnostics: a model with no defensible answer is refused
//! when it is built, not computed with an arbitrary answer.
//!
//! The refusals live in `Model::validate`, which `CompiledModel::compile`
//! runs first: these tests pin that the diagnostic arrives *before* any
//! component, attribute table or engine exists, so a modeller sees the
//! cause rather than a trajectory built on a value nobody can defend.
//!
//! They also pin the two carve-outs, because an over-refusal would be a
//! defect and not a feature: a continuous cycle broken by an integrated
//! attribute (a capacity) is well posed, and a homogeneous discrete
//! fan-in is the modeller's deliberate unit count.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::compile::CompileError;
use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_model::{Model, ModelError};

fn parse(json: &str) -> Model {
    Model::from_json(json).expect("fixture JSON parses")
}

/// The refusal a model produces at build time, as a `ModelError`.
fn refusal(json: &str) -> ModelError {
    match CompiledModel::compile(&parse(json)) {
        Err(CompileError::Invalid(error)) => error,
        Err(other) => panic!("expected a validation refusal, got {other:?}"),
        Ok(_) => panic!("expected a validation refusal, the model compiled"),
    }
}

/// Compile and run: the accepted models must still simulate.
fn run(json: &str, t_max: f64) {
    let compiled = CompiledModel::compile(&parse(json)).expect("model compiles");
    let config = EngineConfig {
        t_max,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config)
        .expect("engine builds")
        .run()
        .expect("simulation runs");
}

/// Two components whose instantaneous values each need the other's:
/// `link.rate` is computed from what `store` exports and `store.level`
/// from what `link` exports. `store_kind` selects whether the store
/// integrates its level (a capacity) or recomputes it instantaneously.
fn cycle_model(store_kind: &str) -> String {
    format!(
        r#"
{{
  "name": "continuous_cycle",
  "components": [
    {{
      "name": "store",
      "attributes": [
        {{"name": "level", "kind": "float", "init": {{"kind": "float", "value": 1.0}}}}
      ],
      "ports": [
        {{"name": "level_out", "dir": "out", "attr": "level"}},
        {{"name": "rate_in", "dir": "in"}}
      ],
      "equations": [
        {{"target": "level", "kind": "{store_kind}",
         "expr": {{"op": "port_agg", "port": {{"component": "store", "port": "rate_in"}},
                  "agg": "sum"}}}}
      ]
    }},
    {{
      "name": "link",
      "attributes": [
        {{"name": "rate", "kind": "float", "init": {{"kind": "float", "value": 0.0}}}}
      ],
      "ports": [
        {{"name": "rate_out", "dir": "out", "attr": "rate"}},
        {{"name": "level_in", "dir": "in"}}
      ],
      "equations": [
        {{"target": "rate", "kind": "explicit",
         "expr": {{"op": "mul", "args": [
            {{"op": "const", "value": {{"kind": "float", "value": -0.5}}}},
            {{"op": "port_agg", "port": {{"component": "link", "port": "level_in"}},
             "agg": "sum"}}]}}}}
      ]
    }}
  ],
  "connections": [
    {{"from": {{"component": "store", "port": "level_out"}},
     "to": {{"component": "link", "port": "level_in"}}}},
    {{"from": {{"component": "link", "port": "rate_out"}},
     "to": {{"component": "store", "port": "rate_in"}}}}
  ],
  "indicators": [
    {{"name": "level", "target": "attribute",
     "attr": {{"component": "store", "attribute": "level"}}}}
  ]
}}
"#
    )
}

/// Two producers feeding one in port, whose values a balance sums.
/// `second_kind` is the kind the second producer exports.
fn fan_in_model(second_kind: &str, second_init: &str) -> String {
    format!(
        r#"
{{
  "name": "fan_in",
  "components": [
    {{
      "name": "train1",
      "attributes": [
        {{"name": "pumping", "kind": "bool", "init": {{"kind": "bool", "value": true}}}}
      ],
      "ports": [{{"name": "pumping_out", "dir": "out", "attr": "pumping"}}]
    }},
    {{
      "name": "train2",
      "attributes": [
        {{"name": "pumping", "kind": "{second_kind}", "init": {second_init}}}
      ],
      "ports": [{{"name": "pumping_out", "dir": "out", "attr": "pumping"}}]
    }},
    {{
      "name": "pool",
      "attributes": [
        {{"name": "temperature", "kind": "float", "init": {{"kind": "float", "value": 20.0}}}}
      ],
      "ports": [{{"name": "pumping_in", "dir": "in"}}],
      "equations": [
        {{"target": "temperature", "kind": "ode",
         "expr": {{"op": "sub",
                  "lhs": {{"op": "const", "value": {{"kind": "float", "value": 8.0}}}},
                  "rhs": {{"op": "mul", "args": [
                     {{"op": "const", "value": {{"kind": "float", "value": 5.0}}}},
                     {{"op": "port_agg",
                      "port": {{"component": "pool", "port": "pumping_in"}},
                      "agg": "sum"}}]}}}}}}
      ]
    }}
  ],
  "connections": [
    {{"from": {{"component": "train1", "port": "pumping_out"}},
     "to": {{"component": "pool", "port": "pumping_in"}}}},
    {{"from": {{"component": "train2", "port": "pumping_out"}},
     "to": {{"component": "pool", "port": "pumping_in"}}}}
  ],
  "indicators": [
    {{"name": "temperature", "target": "attribute",
     "attr": {{"component": "pool", "attribute": "temperature"}}}}
  ]
}}
"#
    )
}

/// The same loop, closed through a **per-connection channel**: the
/// producer's share of the edge is computed from what the consumer sends
/// back, and the consumer's value from that share. `consumer_kind`
/// selects whether the consumer integrates its value (a capacity) or
/// recomputes it instantaneously.
fn channel_cycle_model(consumer_kind: &str) -> String {
    format!(
        r#"
{{
  "name": "channel_cycle",
  "components": [
    {{
      "name": "producer",
      "attributes": [
        {{"name": "total", "kind": "float", "init": {{"kind": "float", "value": 4.0}}}}
      ],
      "ports": [
        {{"name": "out", "dir": "out", "attr": "total",
         "channels": [{{"name": "share", "init": 0.0}}]}},
        {{"name": "back", "dir": "in"}}
      ],
      "equations": [
        {{"target": "out__share__consumer__feed", "kind": "explicit",
         "expr": {{"op": "port_agg", "port": {{"component": "producer", "port": "back"}},
                  "agg": "sum"}}}}
      ]
    }},
    {{
      "name": "consumer",
      "attributes": [
        {{"name": "used", "kind": "float", "init": {{"kind": "float", "value": 1.0}}}}
      ],
      "ports": [
        {{"name": "feed", "dir": "in"}},
        {{"name": "back_out", "dir": "out", "attr": "used"}}
      ],
      "equations": [
        {{"target": "used", "kind": "{consumer_kind}",
         "expr": {{"op": "port_agg", "port": {{"component": "consumer", "port": "feed"}},
                  "agg": "sum", "channel": "share"}}}}
      ]
    }}
  ],
  "connections": [
    {{"from": {{"component": "producer", "port": "out"}},
     "to": {{"component": "consumer", "port": "feed"}}}},
    {{"from": {{"component": "consumer", "port": "back_out"}},
     "to": {{"component": "producer", "port": "back"}}}}
  ],
  "indicators": [
    {{"name": "used", "target": "attribute",
     "attr": {{"component": "consumer", "attribute": "used"}}}}
  ]
}}
"#
    )
}

#[test]
fn continuous_cycle_with_no_capacity_is_refused_before_the_engine_is_built() {
    assert_eq!(
        refusal(&cycle_model("explicit")),
        ModelError::AlgebraicLoop {
            cycle: "store.level -> link.rate -> store.level".into(),
        }
    );
}

#[test]
fn the_same_cycle_broken_by_a_capacity_compiles_and_runs() {
    run(&cycle_model("ode"), 5.0);
}

#[test]
fn a_cycle_closed_through_a_channel_is_refused_too() {
    // The per-connection quantities are ordinary float attributes, so a
    // loop through them is a loop like any other.
    assert_eq!(
        refusal(&channel_cycle_model("explicit")),
        ModelError::AlgebraicLoop {
            cycle: "producer.out__share__consumer__feed -> consumer.used \
                    -> producer.out__share__consumer__feed"
                .into(),
        }
    );
}

#[test]
fn the_same_channel_cycle_broken_by_a_capacity_compiles_and_runs() {
    run(&channel_cycle_model("ode"), 5.0);
}

#[test]
fn a_discrete_flow_joining_a_continuous_one_is_refused_before_the_engine_is_built() {
    assert_eq!(
        refusal(&fan_in_model("float", r#"{"kind": "float", "value": 0.5}"#)),
        ModelError::ConnectionFamilyMismatch {
            port: "pool.pumping_in".into(),
            other: "train1.pumping_out".into(),
            other_kind: raichu_model::AttrKind::Bool,
            producer: "train2.pumping_out".into(),
            kind: raichu_model::AttrKind::Float,
        }
    );
}

#[test]
fn a_homogeneous_discrete_fan_in_compiles_and_runs() {
    // The spent-fuel-pool shape of `pool.json`: each running train
    // removes a fixed load, so the balance sums booleans on purpose.
    run(
        &fan_in_model("bool", r#"{"kind": "bool", "value": false}"#),
        5.0,
    );
}
