//! Campaigns folded chunk by chunk.
//!
//! The replicas run in parallel chunks whose samples are folded in replica
//! order as each chunk completes, so a campaign's memory does not grow
//! with its number of replicas. Both campaigns below span more than one
//! chunk (65 536 replicas for a model this small): the thread count must
//! not change a byte, and
//! counting the ends must give what tallying the per-replica ends gives.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use raichu_core::{CompiledModel, SolverParams};
use raichu_expr::{BoolOp, Expr, StateRef};
use raichu_model::{
    Automaton, Component, Distrib, Indicator, IndicatorTarget, Model, Target, Transition,
    TransitionKind,
};
use raichu_montecarlo::{count_to_targets, run, run_to_targets, McConfig, DEFAULT_CONFIDENCE};

/// More replicas than one chunk holds.
const NB_RUNS: u64 = 70_000;
const HORIZON: f64 = 10.0;

fn transition(name: &str, source: &str, target: &str, rate: f64) -> Transition {
    Transition {
        name: name.into(),
        source: source.into(),
        guard: None,
        targets: vec![target.into()],
        on_interruption: Default::default(),
        monitored: false,
        cycle_group: None,
        kind: None,
        effects: vec![],
        distrib: Distrib::Exp {
            rate: Some(rate),
            rate_expr: None,
        },
    }
}

fn component(name: &str, automaton: Automaton) -> Component {
    Component {
        name: name.into(),
        attributes: vec![],
        ports: vec![],
        interfaces: vec![],
        automata: vec![automaton],
        allocations: vec![],
        equations: vec![],
        sensitive_functions: vec![],
    }
}

/// A repairable unit: `ok` -> `nok` at `fail`, back at `repair`.
fn unit(name: &str, fail: f64, repair: f64) -> Component {
    let mut occ = transition("occ", "ok", "nok", fail);
    occ.kind = Some(TransitionKind::Failure);
    let mut rep = transition("rep", "nok", "ok", repair);
    rep.kind = Some(TransitionKind::Repair);
    component(
        name,
        Automaton {
            name: "fail".into(),
            states: vec!["ok".into(), "nok".into()],
            init: "ok".into(),
            transitions: vec![occ, rep],
        },
    )
}

fn down(c: &str) -> Expr {
    Expr::StateActive {
        state: StateRef {
            component: c.into(),
            automaton: "fail".into(),
            state: "nok".into(),
        },
    }
}

/// A repairable pair whose feared event is both units down at once,
/// observed through the state of the first unit.
fn repairable_pair() -> CompiledModel {
    CompiledModel::compile(&repairable_pair_model()).unwrap()
}

/// The source model of [`repairable_pair`].
fn repairable_pair_model() -> Model {
    let mut lose = Transition {
        distrib: Distrib::Inst { probs: vec![] },
        ..transition("lose", "ok", "down", 1.0)
    };
    lose.guard = Some(Expr::Bool {
        bool_op: BoolOp::And,
        args: vec![down("A"), down("B")],
    });
    let watcher = component(
        "sys",
        Automaton {
            name: "watch".into(),
            states: vec!["ok".into(), "down".into()],
            init: "ok".into(),
            transitions: vec![lose],
        },
    );
    Model {
        programs: vec![],
        name: "repairable_pair".into(),
        components: vec![unit("A", 0.2, 1.0), unit("B", 0.3, 1.5), watcher],
        connections: vec![],
        indicators: vec![Indicator {
            name: "A_down".into(),
            target: IndicatorTarget::State {
                component: "A".into(),
                automaton: "fail".into(),
                state: "nok".into(),
            },
        }],
        targets: vec![Target {
            name: "feared".into(),
            component: "sys".into(),
            automaton: "watch".into(),
            state: "down".into(),
        }],
        evaluation_order: None,
        unbounded_rate: None,
        fmu_units: vec![],
        interface_connections: vec![],
    }
}

fn config(threads: Option<usize>) -> McConfig {
    McConfig {
        nb_runs: NB_RUNS,
        seed: 2017,
        t_max: HORIZON,
        samples: vec![0.0, HORIZON / 2.0, HORIZON],
        threads,
        quantiles: vec![0.25, 0.5, 0.75],
        confidence: DEFAULT_CONFIDENCE,
        ode: SolverParams::default(),
        stop_at_targets: false,
        flow: Default::default(),
    }
}

#[test]
fn a_campaign_spanning_several_chunks_does_not_depend_on_the_thread_count() {
    let model = repairable_pair();
    let one = run(&model, &config(Some(1))).unwrap();
    let many = run(&model, &config(Some(4))).unwrap();
    assert_eq!(one, many);
    // The quantile columns hold every replica, not only the last chunk.
    let estimate = &one.indicators[0];
    assert!(estimate.mean[2] > 0.0 && estimate.mean[2] < 1.0);
    assert_eq!(estimate.quantiles.len(), 3);
}

#[test]
fn counting_the_ends_gives_the_tally_of_the_per_replica_ends() {
    let model = repairable_pair();
    let cfg = config(None);
    let listed = run_to_targets(&model, &cfg).unwrap();
    let counted = count_to_targets(&model, &cfg).unwrap();
    assert_eq!(listed.ends.len() as u64, NB_RUNS);
    let mut tally = BTreeMap::new();
    for end in &listed.ends {
        if let Some(target) = &end.end_cause {
            *tally.entry(target.clone()).or_insert(0_u64) += 1;
        }
    }
    assert!(
        tally["feared"] > 0,
        "the feared event is reached in some replicas"
    );
    assert_eq!(counted.reached, tally);
    assert_eq!(
        counted.count_reached("feared"),
        listed.count_reached("feared")
    );
    assert_eq!(counted.count_reached("unknown"), 0);
    assert_eq!(counted.estimates, listed.estimates);
}

/// The repairable pair observed through 600 indicators, alternately on `A`
/// and on `B`: each replica records 1 800 samples, so a chunk holds the
/// fewest replicas the driver allows and 2 500 replicas span three chunks.
fn heavily_observed_pair() -> CompiledModel {
    let mut model = repairable_pair_model();
    model.indicators = (0..600)
        .map(|i| {
            let unit = if i % 2 == 0 { "A" } else { "B" };
            Indicator {
                name: format!("{unit}_down_{i}"),
                target: IndicatorTarget::State {
                    component: unit.into(),
                    automaton: "fail".into(),
                    state: "nok".into(),
                },
            }
        })
        .collect();
    CompiledModel::compile(&model).unwrap()
}

#[test]
fn small_chunks_of_heavily_observed_replicas_keep_every_indicator_in_place() {
    let model = heavily_observed_pair();
    let config = |threads| McConfig {
        nb_runs: 2_500,
        quantiles: vec![],
        ..config(threads)
    };
    let one = run(&model, &config(Some(1))).unwrap();
    let many = run(&model, &config(Some(4))).unwrap();
    assert_eq!(one, many);
    // Every indicator on `A` reads the trajectories the single-indicator
    // pair reads, with the same seed: a sample folded into another
    // indicator's sums would no longer reproduce its estimate.
    let alone = run(&repairable_pair(), &config(Some(1))).unwrap();
    let a = &alone.indicators[0];
    for (i, estimate) in one.indicators.iter().enumerate() {
        assert_eq!(
            estimate.name,
            format!("{}_down_{i}", if i % 2 == 0 { "A" } else { "B" })
        );
        if i % 2 == 0 {
            assert_eq!(estimate.mean, a.mean, "indicator {i}");
            assert_eq!(estimate.sojourn_mean, a.sojourn_mean, "indicator {i}");
        } else {
            assert_ne!(estimate.mean, a.mean, "indicator {i} reads `B`, not `A`");
        }
    }
}
