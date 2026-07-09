//! Monte-Carlo driver tests — M2 exit criteria: closed-form agreement,
//! thread-count-independent bytes, seed reproducibility.

#![allow(clippy::unwrap_used, clippy::panic)]

use raichu_core::CompiledModel;
use raichu_model::{Automaton, Component, Distrib, Indicator, IndicatorTarget, Model, Transition};
use raichu_montecarlo::{run, McConfig};

/// The `test_pyc_system_003` model: one component, `ok → nok` at rate
/// λ = 1/5, no repair. Closed forms: P(nok at t) = 1 − e^{−λt};
/// E[sojourn_nok](t) = t − (1 − e^{−λt})/λ.
fn exp_ok_nok(rate: f64) -> Model {
    Model {
        name: "exp_ok_nok".into(),
        components: vec![Component {
            name: "C".into(),
            attributes: vec![],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "aut_ok_nok".into(),
                states: vec!["ok".into(), "nok".into()],
                init: "ok".into(),
                transitions: vec![Transition {
                    name: "ok_nok".into(),
                    source: "ok".into(),
                    guard: None,
                    targets: vec!["nok".into()],
                    on_interruption: Default::default(),
                    distrib: Distrib::Exp {
                        rate: Some(rate),
                        rate_expr: None,
                    },
                }],
            }],
            equations: vec![],
            sensitive_functions: vec![],
        }],
        connections: vec![],
        indicators: vec![Indicator {
            name: "C_nok".into(),
            target: IndicatorTarget::State {
                component: "C".into(),
                automaton: "aut_ok_nok".into(),
                state: "nok".into(),
            },
        }],
    }
}

fn schedule() -> Vec<f64> {
    (1..=10).map(f64::from).collect()
}

#[test]
fn estimates_match_closed_forms_within_confidence() {
    let rate = 0.2;
    let model = exp_ok_nok(rate);
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = McConfig {
        nb_runs: 20_000,
        seed: 42,
        t_max: 10.0,
        samples: schedule(),
        threads: None,
        quantiles: vec![],
        ode: Default::default(),
    };
    let estimates = run(&compiled, &config).unwrap();
    let est = &estimates.indicators[0];
    let n = config.nb_runs as f64;
    for (k, t) in est.instants.iter().enumerate() {
        let p = 1.0 - (-rate * t).exp();
        let se = est.std[k] / n.sqrt();
        assert!(
            (est.mean[k] - p).abs() < 4.0 * se.max(1e-6),
            "P(nok at {t}): estimated {} vs closed form {p} (se {se})",
            est.mean[k]
        );
        let sojourn = t - (1.0 - (-rate * t).exp()) / rate;
        let sj_se = est.sojourn_std[k] / n.sqrt();
        assert!(
            (est.sojourn_mean[k] - sojourn).abs() < 4.0 * sj_se.max(1e-6),
            "E[sojourn]({t}): estimated {} vs closed form {sojourn} (se {sj_se})",
            est.sojourn_mean[k]
        );
    }
}

/// FP addition is non-associative: the index-ordered serial reduction
/// is what makes the estimate *bytes* identical across thread counts.
#[test]
fn one_thread_and_many_threads_give_identical_bytes() {
    let model = exp_ok_nok(0.2);
    let compiled = CompiledModel::compile(&model).unwrap();
    let base = McConfig {
        nb_runs: 2_000,
        seed: 7,
        t_max: 10.0,
        samples: schedule(),
        threads: Some(1),
        quantiles: vec![],
        ode: Default::default(),
    };
    let single = run(&compiled, &base).unwrap();
    let multi = run(
        &compiled,
        &McConfig {
            threads: Some(8),
            quantiles: vec![],
            ode: Default::default(),
            ..base
        },
    )
    .unwrap();
    assert_eq!(
        serde_json::to_vec(&single).unwrap(),
        serde_json::to_vec(&multi).unwrap()
    );
}

#[test]
fn same_seed_reproduces_and_other_seed_differs() {
    let model = exp_ok_nok(0.2);
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = McConfig {
        nb_runs: 500,
        seed: 123,
        t_max: 10.0,
        samples: schedule(),
        threads: None,
        quantiles: vec![],
        ode: Default::default(),
    };
    let a = run(&compiled, &config).unwrap();
    let b = run(&compiled, &config).unwrap();
    assert_eq!(
        serde_json::to_vec(&a).unwrap(),
        serde_json::to_vec(&b).unwrap()
    );
    let c = run(
        &compiled,
        &McConfig {
            seed: 124,
            ..config
        },
    )
    .unwrap();
    assert_ne!(
        serde_json::to_vec(&a).unwrap(),
        serde_json::to_vec(&c).unwrap()
    );
}

/// Single-trajectory stochastic replay: same (seed, stream) ⇒ identical
/// event bytes; the mean over replicas differs from any single one.
#[test]
fn single_trajectory_replay_is_bit_identical() {
    use raichu_core::{Engine, EngineConfig};
    let model = exp_ok_nok(0.2);
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max: 100.0,
        seed: 55,
        rng_stream: 3,
        ..EngineConfig::default()
    };
    let a = Engine::new(&compiled, config.clone())
        .unwrap()
        .run()
        .unwrap();
    let b = Engine::new(&compiled, config).unwrap().run().unwrap();
    assert_eq!(
        serde_json::to_vec(&a).unwrap(),
        serde_json::to_vec(&b).unwrap()
    );
    assert_eq!(a.events.len(), 1);
    assert!(a.provenance.seed == Some(55));
}
