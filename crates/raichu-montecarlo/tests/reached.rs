//! The **reached** measure: was the indicator ever active by each instant?
//!
//! A property of the trajectory up to the instant, not of the instant: once
//! the indicator has been active it stays reached, even after it falls back.
//! Its mean over replicas is the probability of having reached it by `t`,
//! which is what a RAMS study asks with "had value" (the reference engine's
//! `realized` computation, measured 2026-09-25 on a room whose hydrogen
//! crosses a threshold and falls back: 0 before the first crossing, 1 at every
//! instant after it).
//!
//! Every expected figure is a closed form: a delay puts the entry at a
//! declared date, and an exponential entry at rate `lambda` is reached by `t`
//! with probability `1 - exp(-lambda t)`, whatever the repairs do after.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, FlowConfig};
use raichu_model::Model;
use raichu_montecarlo::{run, McConfig, McEstimates};

/// One automaton `off -> on -> off`, the entry and the exit on the given
/// laws, observed on its `on` state.
fn on_off(init: &str, enter: &str, leave: &str) -> Model {
    Model::from_json(&format!(
        r#"{{
  "name": "reached",
  "components": [{{
    "name": "c", "attributes": [], "ports": [],
    "automata": [{{
      "name": "aut", "states": ["off", "on"], "init": "{init}",
      "transitions": [
        {{"name": "enter", "source": "off", "targets": ["on"], {enter}}},
        {{"name": "leave", "source": "on", "targets": ["off"], {leave}}}]}}]}}],
  "indicators": [
    {{"name": "c_on", "target": "state", "component": "c", "automaton": "aut", "state": "on"}}]
}}"#
    ))
    .unwrap()
}

fn estimate(model: &Model, nb_runs: u64, samples: Vec<f64>, threads: usize) -> McEstimates {
    let compiled = CompiledModel::compile(model).unwrap();
    let t_max = *samples.last().unwrap();
    run(
        &compiled,
        &McConfig {
            nb_runs,
            seed: 7,
            t_max,
            samples,
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
fn an_indicator_reached_then_left_stays_reached() {
    // Active on [2, 4]: sampled at 1, 3 and 5 the value reads 0, 1, 0 and the
    // reached measure 0, 1, 1.
    let model = on_off(
        "off",
        r#""distrib": "delay", "time": 2.0"#,
        r#""distrib": "delay", "time": 2.0"#,
    );
    let estimates = estimate(&model, 1, vec![1.0, 3.0, 5.0], 1);
    let indicator = &estimates.indicators[0];
    assert_eq!(indicator.mean, vec![0.0, 1.0, 0.0]);
    assert_eq!(indicator.reached_mean, vec![0.0, 1.0, 1.0]);
}

#[test]
fn an_indicator_active_from_the_start_is_reached_at_every_instant() {
    // Initially on, left at 1: reached from t = 0 on, although never
    // entered by a transition.
    let model = on_off(
        "on",
        r#""distrib": "delay", "time": 100.0"#,
        r#""distrib": "delay", "time": 1.0"#,
    );
    let estimates = estimate(&model, 1, vec![0.0, 0.5, 2.0, 5.0], 1);
    let indicator = &estimates.indicators[0];
    assert_eq!(indicator.mean, vec![1.0, 1.0, 0.0, 0.0]);
    assert_eq!(indicator.reached_mean, vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn the_probability_of_having_reached_it_is_the_first_entry_law() {
    // Entered at rate 0.5, left at rate 4: the value's mean is an
    // availability-like quantity, the reached measure the first-entry
    // distribution 1 - exp(-0.5 t), which the repairs do not touch.
    let (lambda, runs) = (0.5_f64, 20_000_u64);
    let samples = vec![0.5, 1.0, 2.0, 4.0];
    let model = on_off(
        "off",
        &format!(r#""distrib": "exp", "rate": {lambda}"#),
        r#""distrib": "exp", "rate": 4.0"#,
    );
    let estimates = estimate(&model, runs, samples.clone(), 4);
    let indicator = &estimates.indicators[0];
    for (k, t) in samples.iter().enumerate() {
        let p = 1.0 - (-lambda * t).exp();
        let error = (p * (1.0 - p) / runs as f64).sqrt();
        let got = indicator.reached_mean[k];
        assert!((got - p).abs() < 5.0 * error, "t={t}: {got} against {p}");
        // A probability's sample deviation.
        assert!((indicator.reached_std[k] - (got * (1.0 - got)).sqrt()).abs() < 1e-3);
        // The value's own mean is far below: the repairs bring it back.
        assert!(indicator.mean[k] < got);
    }
}

#[test]
fn the_reached_measure_is_bit_identical_whatever_the_thread_count() {
    let model = on_off(
        "off",
        r#""distrib": "exp", "rate": 0.5"#,
        r#""distrib": "exp", "rate": 4.0"#,
    );
    let samples = vec![0.5, 1.0, 2.0];
    let one = estimate(&model, 2_000, samples.clone(), 1);
    let many = estimate(&model, 2_000, samples, 8);
    assert_eq!(
        one.indicators[0].reached_mean,
        many.indicators[0].reached_mean
    );
    assert_eq!(
        one.indicators[0].reached_std,
        many.indicators[0].reached_std
    );
}
