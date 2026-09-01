//! Per-connection channel quantities, and the counted-work counters.
//!
//! A producer exposes one attribute per out port, and every in port
//! connected to it reads that same number. A conservative flow needs the
//! opposite: each consumer must receive the share allocated to *it*.
//!
//! The channel is declared once, on the producing port; the compiler
//! materialises one float attribute per (connection, channel) on the
//! producing component. Materialised attributes are ordinary float
//! attributes: they are written by equations and sensitive functions,
//! they appear in the causal journal, and they ride in the snapshot.
//!
//! These tests also pin the counted-work counters, the machine-independent
//! units every later performance gate of the continuous-flow plan is
//! measured in.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::compile::{CExpr, CStep};
use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_expr::Value;
use raichu_model::Model;

fn compile(json: &str) -> CompiledModel {
    let model = Model::from_json(json).expect("fixture JSON parses");
    CompiledModel::compile(&model).expect("fixture compiles")
}

fn run(json: &str, t_max: f64) -> raichu_core::SimulationResult {
    let compiled = compile(json);
    let config = EngineConfig {
        t_max,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config).unwrap().run().unwrap()
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

/// One producer, two consumers, one channel: the producer publishes a
/// total of 5 on its out port and allocates 3 to one consumer and 2 to
/// the other over the *same* port. Each consumer reads its own share.
///
/// The producer writes the materialised attributes from explicit
/// equations, which is what the resolved flow network of the plan emits.
const SPLIT_MODEL: &str = r#"
{
  "name": "channel_split",
  "components": [
    {
      "name": "producer",
      "attributes": [
        {"name": "total", "kind": "float", "init": {"kind": "float", "value": 5.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "total",
         "channels": [{"name": "share", "init": 0.0}]}
      ],
      "equations": [
        {"target": "out__share__hi", "kind": "explicit",
         "expr": {"op": "const", "value": {"kind": "float", "value": 3.0}}},
        {"target": "out__share__lo", "kind": "explicit",
         "expr": {"op": "const", "value": {"kind": "float", "value": 2.0}}}
      ]
    },
    {
      "name": "hi",
      "attributes": [
        {"name": "received", "kind": "float", "init": {"kind": "float", "value": 0.0}}
      ],
      "ports": [{"name": "input", "dir": "in"}],
      "equations": [
        {"target": "received", "kind": "explicit",
         "expr": {"op": "port_agg", "agg": "sum", "channel": "share",
                  "port": {"component": "hi", "port": "input"}}}
      ]
    },
    {
      "name": "lo",
      "attributes": [
        {"name": "received", "kind": "float", "init": {"kind": "float", "value": 0.0}},
        {"name": "total_seen", "kind": "float", "init": {"kind": "float", "value": 0.0}}
      ],
      "ports": [{"name": "input", "dir": "in"}],
      "equations": [
        {"target": "received", "kind": "explicit",
         "expr": {"op": "port_agg", "agg": "sum", "channel": "share",
                  "port": {"component": "lo", "port": "input"}}},
        {"target": "total_seen", "kind": "explicit",
         "expr": {"op": "port_agg", "agg": "sum",
                  "port": {"component": "lo", "port": "input"}}}
      ]
    }
  ],
  "connections": [
    {"name": "hi", "from": {"component": "producer", "port": "out"},
     "to": {"component": "hi", "port": "input"}},
    {"name": "lo", "from": {"component": "producer", "port": "out"},
     "to": {"component": "lo", "port": "input"}}
  ],
  "indicators": [
    {"name": "hi_received", "target": "attribute",
     "attr": {"component": "hi", "attribute": "received"}},
    {"name": "lo_received", "target": "attribute",
     "attr": {"component": "lo", "attribute": "received"}},
    {"name": "lo_total_seen", "target": "attribute",
     "attr": {"component": "lo", "attribute": "total_seen"}}
  ]
}
"#;

#[test]
fn each_consumer_reads_its_own_per_connection_quantity() {
    let result = run(SPLIT_MODEL, 1.0);
    assert_eq!(
        last_value(&result, "hi_received"),
        3.0,
        "the consumer on the `hi` connection reads its own share"
    );
    assert_eq!(
        last_value(&result, "lo_received"),
        2.0,
        "the consumer on the `lo` connection reads its own share"
    );
    // The producer's exported attribute stays readable: an aggregation
    // that names no channel still reads the total, unchanged.
    assert_eq!(
        last_value(&result, "lo_total_seen"),
        5.0,
        "an aggregation naming no channel still reads the exported total"
    );
}

#[test]
fn materialised_attributes_carry_the_topology_derived_name() {
    let compiled = compile(SPLIT_MODEL);
    assert!(
        compiled.var_index.contains_key("producer.out__share__hi"),
        "materialised names: {:?}",
        compiled.var_names
    );
    assert!(
        compiled.var_index.contains_key("producer.out__share__lo"),
        "materialised names: {:?}",
        compiled.var_names
    );
}

/// A channel declared on a port that no connection uses materialises
/// nothing, and a consumer aggregating over it reads the empty set:
/// the declared default of the aggregation, not a build error. muscadet
/// relies on unconnected in-ports defaulting rather than failing.
const UNCONNECTED_MODEL: &str = r#"
{
  "name": "channel_unconnected",
  "components": [
    {
      "name": "producer",
      "attributes": [
        {"name": "total", "kind": "float", "init": {"kind": "float", "value": 5.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "total",
         "channels": [{"name": "share", "init": 1.5}]}
      ]
    },
    {
      "name": "consumer",
      "attributes": [
        {"name": "received", "kind": "float", "init": {"kind": "float", "value": 9.0}}
      ],
      "ports": [{"name": "input", "dir": "in"}],
      "equations": [
        {"target": "received", "kind": "explicit",
         "expr": {"op": "port_agg", "agg": "sum", "channel": "share",
                  "port": {"component": "consumer", "port": "input"}}}
      ]
    }
  ],
  "connections": [],
  "indicators": [
    {"name": "received", "target": "attribute",
     "attr": {"component": "consumer", "attribute": "received"}}
  ]
}
"#;

#[test]
fn a_channel_without_connections_yields_the_aggregation_default() {
    let compiled = compile(UNCONNECTED_MODEL);
    assert!(
        !compiled
            .var_names
            .iter()
            .any(|name| name.contains("out__share")),
        "no connection, no materialised attribute: {:?}",
        compiled.var_names
    );
    let result = run(UNCONNECTED_MODEL, 1.0);
    assert_eq!(
        last_value(&result, "received"),
        0.0,
        "an aggregation over no connection yields its default, not an error"
    );
}

/// A materialised attribute initialises to its channel's declared
/// default, so a connected consumer reads that value before anything
/// writes the channel.
const DEFAULT_MODEL: &str = r#"
{
  "name": "channel_default",
  "components": [
    {
      "name": "producer",
      "attributes": [
        {"name": "total", "kind": "float", "init": {"kind": "float", "value": 5.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "total",
         "channels": [{"name": "share", "init": 1.5}]}
      ]
    },
    {
      "name": "consumer",
      "attributes": [
        {"name": "received", "kind": "float", "init": {"kind": "float", "value": 9.0}}
      ],
      "ports": [{"name": "input", "dir": "in"}],
      "equations": [
        {"target": "received", "kind": "explicit",
         "expr": {"op": "port_agg", "agg": "sum", "channel": "share",
                  "port": {"component": "consumer", "port": "input"}}}
      ]
    }
  ],
  "connections": [
    {"from": {"component": "producer", "port": "out"},
     "to": {"component": "consumer", "port": "input"}}
  ],
  "indicators": [
    {"name": "received", "target": "attribute",
     "attr": {"component": "consumer", "attribute": "received"}},
    {"name": "allocated", "target": "attribute",
     "attr": {"component": "producer", "attribute": "out__share__consumer__input"}}
  ]
}
"#;

#[test]
fn a_materialised_attribute_starts_at_the_declared_channel_default() {
    let compiled = compile(DEFAULT_MODEL);
    // No connection name given: the edge is named from the destination.
    let idx = compiled.var_index["producer.out__share__consumer__input"];
    assert_eq!(compiled.var_init[idx], Value::Float(1.5));
    let result = run(DEFAULT_MODEL, 1.0);
    assert_eq!(last_value(&result, "received"), 1.5);
}

/// Three producers feeding one in port, with values chosen so the sum is
/// order-sensitive in binary floating point: `1e16 + 1 + 1` folds to
/// `1e16` left to right and to `1.0000000000000002e16` if the two small
/// terms are added first. The fold order *is* the declaration order, and
/// the strict comparison level of the validation contract depends on it.
const ORDER_MODEL: &str = r#"
{
  "name": "aggregate_order",
  "components": [
    {
      "name": "big",
      "attributes": [
        {"name": "v", "kind": "float", "init": {"kind": "float", "value": 1e16}}
      ],
      "ports": [{"name": "out", "dir": "out", "attr": "v"}]
    },
    {
      "name": "small_a",
      "attributes": [
        {"name": "v", "kind": "float", "init": {"kind": "float", "value": 1.0}}
      ],
      "ports": [{"name": "out", "dir": "out", "attr": "v"}]
    },
    {
      "name": "small_b",
      "attributes": [
        {"name": "v", "kind": "float", "init": {"kind": "float", "value": 1.0}}
      ],
      "ports": [{"name": "out", "dir": "out", "attr": "v"}]
    },
    {
      "name": "sink",
      "attributes": [
        {"name": "total", "kind": "float", "init": {"kind": "float", "value": 0.0}}
      ],
      "ports": [{"name": "input", "dir": "in"}],
      "equations": [
        {"target": "total", "kind": "explicit",
         "expr": {"op": "port_agg", "agg": "sum",
                  "port": {"component": "sink", "port": "input"}}}
      ]
    }
  ],
  "connections": [
    {"from": {"component": "big", "port": "out"},
     "to": {"component": "sink", "port": "input"}},
    {"from": {"component": "small_a", "port": "out"},
     "to": {"component": "sink", "port": "input"}},
    {"from": {"component": "small_b", "port": "out"},
     "to": {"component": "sink", "port": "input"}}
  ],
  "indicators": [
    {"name": "total", "target": "attribute",
     "attr": {"component": "sink", "attribute": "total"}}
  ]
}
"#;

#[test]
fn an_existing_multi_connection_aggregate_stays_bit_identical() {
    let compiled = compile(ORDER_MODEL);
    let sources = match &compiled.explicit[0] {
        CStep::Equation {
            expr: CExpr::PortAgg { sources, .. },
            ..
        } => sources.clone(),
        other => panic!("expected a port aggregation, got {other:?}"),
    };
    let names: Vec<&str> = sources
        .iter()
        .map(|idx| compiled.var_names[*idx].as_str())
        .collect();
    assert_eq!(
        names,
        vec!["big.v", "small_a.v", "small_b.v"],
        "the per-destination source list keeps connection declaration order"
    );

    let result = run(ORDER_MODEL, 1.0);
    let expected = (1e16f64 + 1.0) + 1.0;
    assert_eq!(
        last_value(&result, "total").to_bits(),
        expected.to_bits(),
        "the ordered fold is bit-identical to the left-to-right sum"
    );
}

/// A channel whose materialised name is already taken by a user-declared
/// attribute is refused at build time, naming both.
const COLLISION_MODEL: &str = r#"
{
  "name": "channel_collision",
  "components": [
    {
      "name": "producer",
      "attributes": [
        {"name": "total", "kind": "float", "init": {"kind": "float", "value": 5.0}},
        {"name": "out__share__consumer__input", "kind": "float",
         "init": {"kind": "float", "value": 0.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "total",
         "channels": [{"name": "share", "init": 0.0}]}
      ]
    },
    {
      "name": "consumer",
      "attributes": [],
      "ports": [{"name": "input", "dir": "in"}]
    }
  ],
  "connections": [
    {"from": {"component": "producer", "port": "out"},
     "to": {"component": "consumer", "port": "input"}}
  ]
}
"#;

#[test]
fn a_materialised_name_colliding_with_a_declared_attribute_is_refused() {
    let model = Model::from_json(COLLISION_MODEL).expect("fixture JSON parses");
    let error = model
        .validate()
        .expect_err("the collision must be refused at build time");
    let message = error.to_string();
    assert!(
        message.contains("out__share__consumer__input"),
        "the message names the materialised attribute: {message}"
    );
    assert!(
        message.contains("share") && message.contains("out"),
        "the message names the channel and the port it is declared on: {message}"
    );
    assert!(
        message.contains("producer"),
        "the message names the owning component: {message}"
    );
}

/// A channel list belongs to an out port: an in port reads channels, it
/// does not declare them.
const IN_PORT_CHANNEL_MODEL: &str = r#"
{
  "name": "channel_on_in_port",
  "components": [
    {
      "name": "consumer",
      "attributes": [],
      "ports": [
        {"name": "input", "dir": "in", "channels": [{"name": "share", "init": 0.0}]}
      ]
    }
  ],
  "connections": []
}
"#;

#[test]
fn a_channel_list_on_an_in_port_is_rejected() {
    let model = Model::from_json(IN_PORT_CHANNEL_MODEL).expect("fixture JSON parses");
    let error = model
        .validate()
        .expect_err("an in port must not declare channels");
    let message = error.to_string();
    assert!(
        message.contains("consumer") && message.contains("input"),
        "the message names the component and the port: {message}"
    );
}

/// A materialised attribute is an ordinary float attribute: a sensitive
/// function writes it, the causal journal records the write under the
/// stable topology-derived name, and the value rides in the snapshot.
const JOURNAL_MODEL: &str = r#"
{
  "name": "channel_journal",
  "components": [
    {
      "name": "producer",
      "attributes": [
        {"name": "total", "kind": "float", "init": {"kind": "float", "value": 5.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "total",
         "channels": [{"name": "share", "init": 0.0}]}
      ],
      "automata": [
        {"name": "mode", "states": ["idle", "serving"], "init": "idle",
         "transitions": [
           {"name": "start", "source": "idle", "targets": ["serving"],
            "distrib": "delay", "time": 2.0}
         ]}
      ],
      "sensitive_functions": [
        {"name": "allocate",
         "effects": [
           {"target": {"component": "producer", "attribute": "out__share__consumer__input"},
            "value": {"op": "if",
                      "cond": {"op": "state_active",
                               "state": {"component": "producer", "automaton": "mode",
                                         "state": "serving"}},
                      "then": {"op": "const", "value": {"kind": "float", "value": 4.0}},
                      "otherwise": {"op": "const", "value": {"kind": "float", "value": 0.0}}}}
         ]}
      ]
    },
    {
      "name": "consumer",
      "attributes": [
        {"name": "received", "kind": "float", "init": {"kind": "float", "value": 0.0}}
      ],
      "ports": [{"name": "input", "dir": "in"}],
      "equations": [
        {"target": "received", "kind": "explicit",
         "expr": {"op": "port_agg", "agg": "sum", "channel": "share",
                  "port": {"component": "consumer", "port": "input"}}}
      ]
    }
  ],
  "connections": [
    {"from": {"component": "producer", "port": "out"},
     "to": {"component": "consumer", "port": "input"}}
  ],
  "indicators": [
    {"name": "received", "target": "attribute",
     "attr": {"component": "consumer", "attribute": "received"}}
  ]
}
"#;

#[test]
fn a_materialised_attribute_is_journalled_under_a_stable_name() {
    let compiled = compile(JOURNAL_MODEL);
    let config = EngineConfig {
        t_max: 5.0,
        journal: true,
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    let journalled = result.journal.iter().any(|record| {
        matches!(
            record,
            raichu_core::JournalRecord::AttributeChanged { attribute, new, .. }
                if attribute == "producer.out__share__consumer__input"
                    && *new == Value::Float(4.0)
        )
    });
    assert!(
        journalled,
        "the materialised attribute appears in the journal: {:#?}",
        result.journal
    );
}

#[test]
fn a_materialised_attribute_survives_snapshot_and_restore() {
    let compiled = compile(JOURNAL_MODEL);
    let config = EngineConfig {
        t_max: 5.0,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new(&compiled, config).unwrap();
    let name = "producer.out__share__consumer__input";
    let before = engine.snapshot();
    assert_eq!(engine.attribute(name), Some(Value::Float(0.0)));

    while engine.step().unwrap().is_some() {}
    assert_eq!(
        engine.attribute(name),
        Some(Value::Float(4.0)),
        "the allocation was written"
    );
    let after = engine.snapshot();
    assert_eq!(after.attribute(&compiled, name), Some(Value::Float(4.0)));

    engine.restore(&before);
    assert_eq!(
        engine.attribute(name),
        Some(Value::Float(0.0)),
        "restore rewinds the materialised attribute like any other"
    );
    engine.restore(&after);
    assert_eq!(engine.attribute(name), Some(Value::Float(4.0)));
}

#[test]
fn an_indicator_can_observe_a_materialised_attribute() {
    // Materialised attributes are ordinary attributes: the indicator
    // machinery needs no special case for them.
    let result = run(DEFAULT_MODEL, 1.0);
    assert_eq!(last_value(&result, "allocated"), 1.5);
}

/// A `port_agg` naming a channel that the producer on the other end of
/// the edge does not declare is refused at build time, naming both ends.
const UNKNOWN_CHANNEL_MODEL: &str = r#"
{
  "name": "channel_unknown",
  "components": [
    {
      "name": "producer",
      "attributes": [
        {"name": "total", "kind": "float", "init": {"kind": "float", "value": 5.0}}
      ],
      "ports": [{"name": "out", "dir": "out", "attr": "total"}]
    },
    {
      "name": "consumer",
      "attributes": [
        {"name": "received", "kind": "float", "init": {"kind": "float", "value": 0.0}}
      ],
      "ports": [{"name": "input", "dir": "in"}],
      "equations": [
        {"target": "received", "kind": "explicit",
         "expr": {"op": "port_agg", "agg": "sum", "channel": "share",
                  "port": {"component": "consumer", "port": "input"}}}
      ]
    }
  ],
  "connections": [
    {"from": {"component": "producer", "port": "out"},
     "to": {"component": "consumer", "port": "input"}}
  ]
}
"#;

#[test]
fn an_aggregation_naming_an_undeclared_channel_is_refused() {
    let model = Model::from_json(UNKNOWN_CHANNEL_MODEL).expect("fixture JSON parses");
    let message = model
        .validate()
        .expect_err("the channel is not declared by the producer")
        .to_string();
    assert!(
        message.contains("share")
            && message.contains("consumer.input")
            && message.contains("producer.out"),
        "the message names the channel and both ends of the edge: {message}"
    );
}

/// Two unnamed connections between the same pair of ports would
/// materialise the same attribute twice, the second silently winning.
const PARALLEL_EDGES_MODEL: &str = r#"
{
  "name": "channel_parallel_edges",
  "components": [
    {
      "name": "producer",
      "attributes": [
        {"name": "total", "kind": "float", "init": {"kind": "float", "value": 5.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "total",
         "channels": [{"name": "share", "init": 0.0}]}
      ]
    },
    {
      "name": "consumer",
      "attributes": [],
      "ports": [{"name": "input", "dir": "in"}]
    }
  ],
  "connections": [
    {"from": {"component": "producer", "port": "out"},
     "to": {"component": "consumer", "port": "input"}},
    {"from": {"component": "producer", "port": "out"},
     "to": {"component": "consumer", "port": "input"}}
  ]
}
"#;

#[test]
fn two_unnamed_parallel_edges_are_refused() {
    let model = Model::from_json(PARALLEL_EDGES_MODEL).expect("fixture JSON parses");
    let message = model
        .validate()
        .expect_err("two edges cannot claim one materialised name")
        .to_string();
    assert!(
        message.contains("out__share__consumer__input") && message.contains("name"),
        "the message names the attribute and the way out: {message}"
    );
}

#[test]
fn naming_the_parallel_edges_resolves_the_clash() {
    let named = PARALLEL_EDGES_MODEL.replacen(
        r#"{"from": {"component": "producer", "port": "out"},"#,
        r#"{"name": "first", "from": {"component": "producer", "port": "out"},"#,
        1,
    );
    let compiled = compile(&named);
    assert!(compiled
        .var_index
        .contains_key("producer.out__share__first"));
    assert!(compiled
        .var_index
        .contains_key("producer.out__share__consumer__input"));
}
