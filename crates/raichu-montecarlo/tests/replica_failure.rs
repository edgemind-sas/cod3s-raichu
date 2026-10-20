//! **A replica that fails refuses its campaign, and says which one it
//! was** (issue #160, observation 3).
//!
//! A campaign whose replica raised used to return the replica's own
//! error, bare: nothing said which replica it was, so it could not be
//! replayed alone, and when several replicas of one parallel chunk failed
//! the one reported was whichever a worker returned first. Pinned here:
//!
//! - **The replica is named, with its replay coordinates.** The error is
//!   [`EngineError::ReplicaFailed`]; a single run at its `seed` and
//!   `rng_stream = replica` raises the same cause.
//! - **It is the lowest-numbered failure, whatever the thread count.**
//!   Every replica below it completed, which is checked replica by
//!   replica on the single-trajectory path.
//! - **No estimate comes out.** The failing replicas are the ones whose
//!   trajectory reached the failing state, so an estimate over the others
//!   would be biased; the campaign is refused on every entry point.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError, SolverParams};
use raichu_model::Model;
use raichu_montecarlo::{
    run, run_cross_entropy, run_sequences, CrossEntropyError, CrossEntropySettings, McConfig,
    DEFAULT_CONFIDENCE,
};

/// A supply feeding one consumer whose demand is a constant 1 until a
/// switch fails, at an exponential date of rate 1, and then **jumps when
/// it is served** (9 while the consumer holds at most 3, 1 above): a
/// network with no fixed point, which the flow resolution refuses at the
/// failure date. Over a horizon of 0.7 about half of the replicas fail.
const SWITCHED: &str = r#"
{
  "name": "switched",
  "components": [
    {
      "name": "supply",
      "attributes": [
        {"name": "capacity", "kind": "float",
         "init": {"kind": "float", "value": 5.0}}
      ],
      "ports": [
        {"name": "out", "dir": "out", "attr": "capacity",
         "channels": [{"name": "demand"}, {"name": "alloc"}]}
      ],
      "equations": [
        {"target": "out__demand__ea", "kind": "explicit",
         "expr": {"op": "if",
           "cond": {"op": "state_active",
                    "state": {"component": "switch", "automaton": "life", "state": "ko"}},
           "then": {"op": "if",
             "cond": {"op": "cmp", "cmp": "gt",
               "lhs": {"op": "attr", "attr": {"component": "supply",
                                              "attribute": "out__alloc__ea"}},
               "rhs": {"op": "const", "value": {"kind": "float", "value": 3.0}}},
             "then": {"op": "const", "value": {"kind": "float", "value": 1.0}},
             "otherwise": {"op": "const", "value": {"kind": "float", "value": 9.0}}},
           "otherwise": {"op": "const", "value": {"kind": "float", "value": 1.0}}}}
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
      "name": "switch",
      "ports": [], "attributes": [],
      "automata": [{"name": "life", "states": ["ok", "ko"], "init": "ok",
        "transitions": [{"name": "fail", "source": "ok", "targets": ["ko"],
                         "distrib": "exp", "rate": 1.0}]}]
    }
  ],
  "connections": [
    {"name": "ea", "from": {"component": "supply", "port": "out"},
     "to": {"component": "a", "port": "input"}}
  ],
  "targets": [{"name": "switched", "component": "switch", "automaton": "life", "state": "ko"}],
  "indicators": [
    {"name": "got", "target": "attribute",
     "attr": {"component": "a", "attribute": "got"}}
  ]
}
"#;

const SEED: u64 = 11;
const NB_RUNS: u64 = 64;
const T_MAX: f64 = 0.7;

fn compiled() -> CompiledModel {
    let document = format!(
        r#"{{"raichu_model": {{"format": 1, "requires": ["evaluation_order", "allocation"]}},
            "model": {SWITCHED}}}"#
    );
    CompiledModel::compile(&Model::from_json(&document).expect("model loads")).expect("compiles")
}

fn config(threads: Option<usize>) -> McConfig {
    McConfig {
        nb_runs: NB_RUNS,
        seed: SEED,
        t_max: T_MAX,
        samples: vec![T_MAX],
        threads,
        quantiles: Vec::new(),
        confidence: DEFAULT_CONFIDENCE,
        ode: SolverParams::default(),
        stop_at_targets: false,
        flow: Default::default(),
    }
}

/// One replica on the single-trajectory path, at the coordinates a
/// campaign gives it.
fn replay(model: &CompiledModel, replica: u64) -> Result<(), EngineError> {
    Engine::new(
        model,
        EngineConfig {
            t_max: T_MAX,
            samples: vec![T_MAX],
            seed: SEED,
            rng_stream: replica,
            ..EngineConfig::default()
        },
    )?
    .run()
    .map(|_| ())
}

/// The replica a campaign must report: the first one, in replica order,
/// whose single run fails.
fn first_failing(model: &CompiledModel) -> u64 {
    (0..NB_RUNS)
        .find(|&replica| replay(model, replica).is_err())
        .expect("some replica of the campaign fails, or this test exercises nothing")
}

fn replica_of(error: &EngineError) -> (u64, u64, u64) {
    match error {
        EngineError::ReplicaFailed {
            replica,
            nb_runs,
            seed,
            ..
        } => (*replica, *nb_runs, *seed),
        other => panic!("expected the campaign's replica failure, got {other:?}"),
    }
}

#[test]
fn a_failing_replica_is_named_and_replays_alone() {
    let model = compiled();
    let expected = first_failing(&model);
    // Not vacuous: some replicas complete, so "the lowest failure" is a
    // statement about order and not about replica 0.
    assert!(
        (0..NB_RUNS).any(|replica| replay(&model, replica).is_ok()),
        "every replica fails: the model no longer separates them"
    );

    let error = run(&model, &config(None)).expect_err("a failing replica refuses the campaign");
    assert_eq!(replica_of(&error), (expected, NB_RUNS, SEED));
    assert!(
        matches!(error.cause(), EngineError::FlowNotConverged { .. }),
        "the cause is the replica's own refusal: {error}"
    );
    let replayed = replay(&model, expected).expect_err("its replay fails the same way");
    assert_eq!(replayed.to_string(), error.cause().to_string());

    let message = error.to_string();
    for fragment in [
        format!("replica {expected} of {NB_RUNS} failed"),
        format!("seed={SEED}, rng_stream={expected} and the campaign's settings"),
        "did not settle".to_owned(),
    ] {
        assert!(
            message.contains(&fragment),
            "the message is missing `{fragment}`: {message}"
        );
    }
}

#[test]
fn the_reported_replica_does_not_depend_on_the_thread_count() {
    let model = compiled();
    let reported: Vec<_> = [Some(1), Some(2), Some(8)]
        .into_iter()
        .map(|threads| {
            replica_of(&run(&model, &config(threads)).expect_err("refused at every thread count"))
        })
        .collect();
    assert!(
        reported.windows(2).all(|pair| pair[0] == pair[1]),
        "the reported replica moved with the thread count: {reported:?}"
    );
}

#[test]
fn a_sequence_campaign_reports_the_same_replica() {
    let model = compiled();
    let expected = first_failing(&model);
    let error = run_sequences(&model, &config(None)).expect_err("refused as well");
    assert_eq!(replica_of(&error).0, expected);
}

/// A cross-entropy campaign reports its own lowest-numbered failing
/// replica too, at every thread count. With no fitting and every factor
/// at 1 its final replicas are the plain campaign's, stream for stream,
/// so the failure reported is the one the plain campaign names; the
/// error stays the replica's own, because a biased replica would not
/// replay as a plain simulation.
#[test]
fn a_cross_entropy_campaign_reports_its_lowest_failing_replica() {
    let model = compiled();
    let expected = replay(&model, first_failing(&model)).expect_err("fails alone");
    let settings = |threads| CrossEntropySettings {
        target: "switched".into(),
        t_max: T_MAX,
        seed: SEED,
        nb_runs: NB_RUNS,
        fit: false,
        initial_factor: 1.0,
        threads,
        ..CrossEntropySettings::default()
    };
    for threads in [Some(1), Some(2), Some(8)] {
        match run_cross_entropy(&model, &settings(threads)) {
            Err(CrossEntropyError::Engine(error)) => assert_eq!(
                error.to_string(),
                expected.to_string(),
                "with {threads:?} threads"
            ),
            other => {
                panic!("expected the replica's refusal with {threads:?} threads, got {other:?}")
            }
        }
    }
}

/// A wide grid reduces chunks to 100 replicas. Rare failures let at least
/// one whole chunk finish before the lowest failing replica is reported.
#[test]
fn failures_after_a_reduced_chunk_boundary_keep_replica_order() {
    let document = format!(
        r#"{{"raichu_model": {{"format": 1, "requires": ["evaluation_order", "allocation"]}},
            "model": {SWITCHED}}}"#
    );
    let mut source = Model::from_json(&document).unwrap();
    source.components[2].automata[0].transitions[0].distrib = raichu_model::Distrib::Exp {
        rate: Some(0.001),
        rate_expr: None,
    };
    source.indicators = (0..165)
        .map(|i| {
            let mut indicator = source.indicators[0].clone();
            indicator.name = format!("got_{i}");
            indicator
        })
        .collect();
    let model = CompiledModel::compile(&source).unwrap();
    let expected = (0..4096)
        .find(|&replica| replay(&model, replica).is_err())
        .expect("the seeded campaign has a failing replica");
    assert!(
        expected >= 100,
        "replica {expected} must follow a full chunk"
    );
    for threads in [Some(1), Some(4)] {
        let cfg = McConfig {
            nb_runs: 4096,
            samples: (0..=100).map(|i| T_MAX * f64::from(i) / 100.0).collect(),
            ..config(threads)
        };
        let error = run(&model, &cfg).expect_err("the campaign is refused");
        assert_eq!(replica_of(&error), (expected, cfg.nb_runs, SEED));
        assert!(matches!(
            error.cause(),
            EngineError::FlowNotConverged { .. }
        ));
    }
}
