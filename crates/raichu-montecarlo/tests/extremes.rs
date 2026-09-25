//! The extremes over replicas: the smallest and the largest value each
//! measure took, at each instant, across the campaign.
//!
//! A study asks for them beside the mean and the deviation. Every expected
//! figure is a bound of an order statistic: a state entered at a date drawn
//! uniformly on [1, 3] has, by t = 4, spent between 1 and 3 in it, and over
//! thousands of replicas the smallest and largest sojourns sit within a few
//! thousandths of those bounds.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, FlowConfig};
use raichu_model::Model;
use raichu_montecarlo::{run, McConfig, McEstimates};

/// A state entered at a date drawn uniformly on `[1, 3]`, never left.
fn uniform_entry() -> Model {
    Model::from_json(
        r#"{
  "name": "extremes",
  "components": [{
    "name": "c", "attributes": [], "ports": [],
    "automata": [{
      "name": "aut", "states": ["off", "on"], "init": "off",
      "transitions": [
        {"name": "enter", "source": "off", "targets": ["on"],
         "distrib": "uniform", "low": 1.0, "high": 3.0}]}]}],
  "indicators": [
    {"name": "c_on", "target": "state", "component": "c", "automaton": "aut", "state": "on"}]
}"#,
    )
    .unwrap()
}

fn estimate(nb_runs: u64, threads: usize) -> McEstimates {
    let compiled = CompiledModel::compile(&uniform_entry()).unwrap();
    run(
        &compiled,
        &McConfig {
            nb_runs,
            seed: 3,
            t_max: 4.0,
            samples: vec![0.5, 2.0, 4.0],
            threads: Some(threads),
            quantiles: vec![],
            ode: Default::default(),
            stop_at_targets: false,
            flow: FlowConfig::default(),
        },
    )
    .unwrap()
}

#[test]
fn the_extremes_are_the_order_statistics_of_each_measure() {
    let estimates = estimate(5_000, 4);
    let indicator = &estimates.indicators[0];
    // Before any entry every replica reads 0 on every measure.
    for extremes in [
        &indicator.extremes,
        &indicator.sojourn_extremes,
        &indicator.nb_occurrences_extremes,
        &indicator.reached_extremes,
    ] {
        assert_eq!((extremes.min[0], extremes.max[0]), (0.0, 0.0));
    }
    // At t = 2 some replicas have entered and some not.
    assert_eq!(
        (indicator.extremes.min[1], indicator.extremes.max[1]),
        (0.0, 1.0)
    );
    assert_eq!(
        (
            indicator.nb_occurrences_extremes.min[1],
            indicator.nb_occurrences_extremes.max[1]
        ),
        (0.0, 1.0)
    );
    // By t = 4 every replica has entered, and spent 4 - U in the state with
    // U uniform on [1, 3]: between 1 and 3, the extremes within a few
    // thousandths of those bounds over 5 000 replicas.
    assert_eq!(
        (indicator.extremes.min[2], indicator.extremes.max[2]),
        (1.0, 1.0)
    );
    assert_eq!(
        (
            indicator.reached_extremes.min[2],
            indicator.reached_extremes.max[2]
        ),
        (1.0, 1.0)
    );
    let (low, high) = (
        indicator.sojourn_extremes.min[2],
        indicator.sojourn_extremes.max[2],
    );
    assert!((1.0..1.01).contains(&low), "smallest sojourn {low}");
    assert!(high > 2.99 && high <= 3.0, "largest sojourn {high}");
}

#[test]
fn the_extremes_are_bit_identical_whatever_the_thread_count() {
    let (one, many) = (estimate(1_000, 1), estimate(1_000, 8));
    assert_eq!(
        one.indicators[0].sojourn_extremes,
        many.indicators[0].sojourn_extremes
    );
    assert_eq!(one.indicators[0].extremes, many.indicators[0].extremes);
}
