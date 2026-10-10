//! What the confidence intervals cost, next to what a campaign costs.
//!
//! Run with `cargo bench -p raichu-montecarlo`. The pair is the point:
//! `campaign_*` measures replicas of a two-state model, the cheapest
//! campaign there is and therefore the least favourable comparison
//! available; `intervals_*` measures the reduction step that turns the
//! accumulated sums into bounds, allocation included. The interval step
//! does not read the replicas, so its cost grows with the schedule and
//! not with the campaign: doubling the replica count moves one number of
//! the pair and not the other.

#![allow(clippy::unwrap_used, missing_docs)]

use raichu_core::{CompiledModel, FlowConfig};
use raichu_model::{Automaton, Component, Distrib, Indicator, IndicatorTarget, Model, Transition};
use raichu_montecarlo::{run, ConfidenceInterval, Departure, McConfig, DEFAULT_CONFIDENCE};

fn main() {
    divan::main();
}

fn probe_model() -> Model {
    Model {
        programs: vec![],
        name: "ci_bench".into(),
        components: vec![Component {
            name: "C".into(),
            attributes: vec![],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "aut".into(),
                states: vec!["ok".into(), "nok".into()],
                init: "ok".into(),
                transitions: vec![Transition {
                    name: "fire".into(),
                    source: "ok".into(),
                    guard: None,
                    targets: vec!["nok".into()],
                    on_interruption: Default::default(),
                    monitored: false,
                    monitored_states: None,
                    cycle_group: None,
                    kind: None,
                    effects: vec![],
                    distrib: Distrib::Exp {
                        rate: Some(0.05),
                        rate_expr: None,
                    },
                }],
            }],
            allocations: vec![],
            equations: vec![],
            sensitive_functions: vec![],
        }],
        connections: vec![],
        indicators: vec![Indicator {
            name: "fired".into(),
            target: IndicatorTarget::State {
                component: "C".into(),
                automaton: "aut".into(),
                state: "nok".into(),
            },
        }],
        targets: vec![],
        evaluation_order: None,
        unbounded_rate: None,
        fmu_units: vec![],
        interface_connections: vec![],
    }
}

const INSTANTS: usize = 20;

fn schedule() -> Vec<f64> {
    (1..=INSTANTS).map(|k| k as f64).collect()
}

/// A campaign of `nb_runs` replicas over a 20-instant schedule.
#[divan::bench(args = [500, 2_000])]
fn campaign(bencher: divan::Bencher, nb_runs: u64) {
    let compiled = CompiledModel::compile(&probe_model()).unwrap();
    let samples = schedule();
    bencher.bench_local(|| {
        run(
            &compiled,
            &McConfig {
                nb_runs,
                seed: 1,
                t_max: INSTANTS as f64,
                samples: samples.clone(),
                threads: None,
                quantiles: vec![],
                confidence: DEFAULT_CONFIDENCE,
                ode: Default::default(),
                stop_at_targets: false,
                flow: FlowConfig::default(),
            },
        )
        .unwrap()
    });
}

/// The three interval series one indicator carries, over the same
/// schedule: the whole of what the campaign above pays for them.
#[divan::bench(args = [500, 2_000])]
fn intervals(bencher: divan::Bencher, nb_runs: u64) {
    let means: Vec<f64> = (0..INSTANTS).map(|k| 0.05 * (k as f64 + 1.0)).collect();
    let stds: Vec<f64> = means.iter().map(|m| (m * (1.0 - m)).sqrt()).collect();
    let instants: Vec<f64> = (0..INSTANTS).map(|k| k as f64 + 1.0).collect();
    bencher.bench_local(|| {
        (
            ConfidenceInterval::on_proportion(DEFAULT_CONFIDENCE, nb_runs, &means),
            ConfidenceInterval::on_mean(
                DEFAULT_CONFIDENCE,
                nb_runs,
                &means,
                &stds,
                Departure::sojourn(&instants),
            ),
            ConfidenceInterval::on_mean(
                DEFAULT_CONFIDENCE,
                nb_runs,
                &means,
                &stds,
                Departure::count(),
            ),
        )
    });
}
