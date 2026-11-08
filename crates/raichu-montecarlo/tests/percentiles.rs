//! Percentiles (P5, P95, ...) of every indicator measure.
//!
//! A study asks the `P<N>` statistics of an indicator measure: the
//! nearest-rank quantile of order `N/100` of the per-replica values at each
//! instant. The campaign serves them on all six measures, from a summary
//! whose memory does not grow with the replica count. The stated tolerance
//! is the rank error `ε` of the result: exact below `1/ε` replicas, and
//! beyond it a replica value whose rank is within `⌊ε · nb_runs⌋` of the
//! requested one. The exact reference is the same campaign at `ε = 0`,
//! which keeps every replica value.

#![allow(clippy::unwrap_used, clippy::panic)]

use raichu_core::{CompiledModel, FlowConfig};
use raichu_model::{Automaton, Component, Distrib, Indicator, IndicatorTarget, Model, Transition};
use raichu_montecarlo::{
    run, IndicatorEstimate, McConfig, McEstimates, QuantileSeries, DEFAULT_CONFIDENCE,
    DEFAULT_QUANTILE_RANK_ERROR,
};

const HORIZON: f64 = 40.0;
const INSTANTS: [f64; 4] = [0.0, 5.0, 20.0, HORIZON];
const PLATFORM_ORDERS: [f64; 6] = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99];

fn transition(name: &str, source: &str, target: &str, rate: f64) -> Transition {
    Transition {
        name: name.into(),
        source: source.into(),
        guard: None,
        targets: vec![target.into()],
        on_interruption: Default::default(),
        monitored: false,
        monitored_states: None,
        cycle_group: None,
        cycle_cancellation: Default::default(),
        kind: None,
        effects: vec![],
        distrib: Distrib::Exp {
            rate: Some(rate),
            rate_expr: None,
        },
    }
}

/// A repairable component: failure at rate 0.1, repair at rate 0.5. Its
/// `nok` state spreads every measure: a 0/1 value, a continuous sojourn,
/// occurrence and departure counts of a few units, and a reached
/// indicator that is not yet certain at the horizon's start.
fn repairable() -> CompiledModel {
    let model = Model {
        programs: vec![],
        name: "percentiles".into(),
        components: vec![Component {
            name: "C".into(),
            attributes: vec![],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "aut".into(),
                states: vec!["ok".into(), "nok".into()],
                init: "ok".into(),
                transitions: vec![
                    transition("fail", "ok", "nok", 0.1),
                    transition("repair", "nok", "ok", 0.5),
                ],
            }],
            allocations: vec![],
            equations: vec![],
            sensitive_functions: vec![],
        }],
        connections: vec![],
        indicators: vec![Indicator {
            name: "down".into(),
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
        source: None,
    };
    CompiledModel::compile(&model).unwrap()
}

fn config(nb_runs: u64, quantiles: Vec<f64>, rank_error: f64, threads: Option<usize>) -> McConfig {
    McConfig {
        nb_runs,
        seed: 2026,
        t_max: HORIZON,
        samples: INSTANTS.to_vec(),
        threads,
        quantiles,
        quantile_rank_error: rank_error,
        confidence: DEFAULT_CONFIDENCE,
        ode: Default::default(),
        stop_at_targets: false,
        flow: FlowConfig::default(),
    }
}

/// The six quantile families of an estimate, named by their member.
fn families(e: &IndicatorEstimate) -> [(&'static str, &Vec<QuantileSeries>); 6] {
    [
        ("quantiles", &e.quantiles),
        ("sojourn_quantiles", &e.sojourn_quantiles),
        ("nb_occurrences_quantiles", &e.nb_occurrences_quantiles),
        ("reached_quantiles", &e.reached_quantiles),
        ("zero_departures_quantiles", &e.zero_departures_quantiles),
        ("nonzero_reached_quantiles", &e.nonzero_reached_quantiles),
    ]
}

fn campaign(nb_runs: u64, quantiles: Vec<f64>, rank_error: f64) -> McEstimates {
    run(&repairable(), &config(nb_runs, quantiles, rank_error, None)).unwrap()
}

#[test]
fn every_measure_serves_the_requested_percentiles() {
    let estimates = campaign(2_000, PLATFORM_ORDERS.to_vec(), DEFAULT_QUANTILE_RANK_ERROR);
    assert_eq!(estimates.quantile_rank_error, DEFAULT_QUANTILE_RANK_ERROR);
    let e = &estimates.indicators[0];
    for (family, series) in families(e) {
        let orders: Vec<f64> = series.iter().map(|s| s.q).collect();
        assert_eq!(orders, PLATFORM_ORDERS, "{family}");
        for s in series {
            assert_eq!(s.values.len(), INSTANTS.len(), "{family}[q={}]", s.q);
        }
    }
    // The measures spread: the P95 of the sojourn and of the occurrence
    // count at the horizon sit strictly above their P5.
    let at_horizon = INSTANTS.len() - 1;
    for (family, series) in [
        ("sojourn_quantiles", &e.sojourn_quantiles),
        ("nb_occurrences_quantiles", &e.nb_occurrences_quantiles),
        ("zero_departures_quantiles", &e.zero_departures_quantiles),
    ] {
        assert!(
            series[4].values[at_horizon] > series[0].values[at_horizon],
            "{family}: P95 above P5"
        );
    }
}

#[test]
fn below_one_over_the_rank_error_the_percentiles_are_exact() {
    let orders = PLATFORM_ORDERS.to_vec();
    let summarised = campaign(900, orders.clone(), DEFAULT_QUANTILE_RANK_ERROR);
    let exact = campaign(900, orders, 0.0);
    for ((family, s), (_, x)) in families(&summarised.indicators[0])
        .into_iter()
        .zip(families(&exact.indicators[0]))
    {
        assert_eq!(s, x, "{family}");
    }
}

/// Beyond `1/ε` replicas, each percentile lies between the exact
/// percentiles read off the exact campaign at the shifted orders
/// `q ∓ (ε + 1/n)`: the `⌊εn⌋` rank bound, plus one rank of slack for the
/// rounding of `⌈qn⌉`.
#[test]
fn beyond_one_over_the_rank_error_each_percentile_stays_within_its_rank_band() {
    let nb_runs = 20_000u64;
    let rank_error = 0.01;
    let shift = rank_error + 1.0 / nb_runs as f64;
    let summarised = campaign(nb_runs, PLATFORM_ORDERS.to_vec(), rank_error);
    let bands: Vec<f64> = PLATFORM_ORDERS
        .iter()
        .flat_map(|q| [(q - shift).max(0.0), (q + shift).min(1.0)])
        .collect();
    let exact = campaign(nb_runs, bands, 0.0);
    let (s, x) = (&summarised.indicators[0], &exact.indicators[0]);
    for ((family, series), (_, reference)) in families(s).into_iter().zip(families(x)) {
        for (i, q_series) in series.iter().enumerate() {
            let (low, high) = (&reference[2 * i].values, &reference[2 * i + 1].values);
            for (k, value) in q_series.values.iter().enumerate() {
                assert!(
                    low[k] <= *value && *value <= high[k],
                    "{family}[q={}] at t = {}: {value} outside [{}, {}]",
                    q_series.q,
                    INSTANTS[k],
                    low[k],
                    high[k]
                );
            }
        }
    }
}

#[test]
fn the_zero_one_measures_are_exact_whatever_the_rank_error() {
    let nb_runs = 5_000u64;
    let e = &campaign(nb_runs, PLATFORM_ORDERS.to_vec(), 0.05).indicators[0];
    for (means, series) in [
        (&e.reached_mean, &e.reached_quantiles),
        (&e.nonzero_reached_mean, &e.nonzero_reached_quantiles),
    ] {
        for s in series {
            for (k, value) in s.values.iter().enumerate() {
                // The sorted column is the zeros, then the ones.
                let ones = (means[k] * nb_runs as f64).round() as u64;
                let rank = ((s.q * nb_runs as f64).ceil() as u64).max(1);
                let expected = if rank > nb_runs - ones { 1.0 } else { 0.0 };
                assert_eq!(*value, expected, "q={} at t = {}", s.q, INSTANTS[k]);
            }
        }
    }
}

#[test]
fn a_summarised_campaign_does_not_depend_on_the_thread_count() {
    let model = repairable();
    let one = run(&model, &config(20_000, vec![0.05, 0.95], 0.01, Some(1))).unwrap();
    let many = run(&model, &config(20_000, vec![0.05, 0.95], 0.01, Some(4))).unwrap();
    assert_eq!(one, many);
}

#[test]
fn a_result_written_before_the_measure_percentiles_reads_back_exact() {
    let estimates = campaign(50, vec![0.5], DEFAULT_QUANTILE_RANK_ERROR);
    let mut document = serde_json::to_value(&estimates).unwrap();
    document
        .as_object_mut()
        .unwrap()
        .remove("quantile_rank_error");
    let indicator = document["indicators"][0].as_object_mut().unwrap();
    for member in [
        "nb_occurrences_quantiles",
        "reached_quantiles",
        "zero_departures_quantiles",
        "nonzero_reached_quantiles",
    ] {
        indicator.remove(member);
    }
    let read: McEstimates = serde_json::from_value(document).unwrap();
    assert_eq!(read.quantile_rank_error, 0.0, "older quantiles were exact");
    assert!(read.indicators[0].nb_occurrences_quantiles.is_empty());
    assert_eq!(
        read.indicators[0].quantiles,
        estimates.indicators[0].quantiles
    );
}
