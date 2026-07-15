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
                    monitored: false,
                    cycle_group: None,
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
        targets: vec![],
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
        stop_at_targets: false,
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
        // No repair ⇒ nok is entered at most once, so the occurrence count
        // equals the 0/1 state value trajectory-by-trajectory: the means
        // coincide exactly (exp firing times never land on an integer
        // instant, so the < vs ≤ boundary never differs).
        assert_eq!(
            est.nb_occurrences_mean[k], est.mean[k],
            "nb-occurrences vs state prob at {t}"
        );
    }
}

/// A repairable delay model: `ok → nok` at t=3, `nok → ok` at t=2, so nok is
/// (re)entered at t = 3, 8, 13, …. The occurrence count is deterministic and
/// exact — a direct check of the rising-edge counting with repairs.
#[test]
fn nb_occurrences_counts_repeated_entries_exactly() {
    let mut model = exp_ok_nok(0.0);
    // Swap the exponential failure for a delay + add a delay repair.
    let aut = &mut model.components[0].automata[0];
    aut.transitions[0].distrib = Distrib::Delay { time: 3.0 };
    aut.transitions.push(Transition {
        name: "nok_ok".into(),
        source: "nok".into(),
        guard: None,
        targets: vec!["ok".into()],
        on_interruption: Default::default(),
        monitored: false,
        cycle_group: None,
        distrib: Distrib::Delay { time: 2.0 },
    });
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = McConfig {
        nb_runs: 1,
        seed: 0,
        t_max: 15.0,
        samples: vec![2.0, 4.0, 9.0, 14.0],
        threads: None,
        quantiles: vec![],
        ode: Default::default(),
        stop_at_targets: false,
    };
    let est = &run(&compiled, &config).unwrap().indicators[0];
    // Entries into nok before each instant: t<2 → 0; t=4 → 1 (@3); t=9 → 2
    // (@3, @8); t=14 → 3 (@3, @8, @13).
    assert_eq!(est.nb_occurrences_mean, vec![0.0, 1.0, 2.0, 3.0]);
}

/// `stop_at_targets` switches the measures from free-cycling to
/// first-occurrence-latched: the trajectory freezes at the target hit and
/// the frozen state holds through the remaining sample instants (the
/// reference semantics of target-stopped sequence studies).
#[test]
fn stop_at_targets_latches_the_measures() {
    use raichu_model::Target;
    let mut model = exp_ok_nok(0.0);
    let aut = &mut model.components[0].automata[0];
    aut.transitions[0].distrib = Distrib::Delay { time: 3.0 };
    aut.transitions.push(Transition {
        name: "nok_ok".into(),
        source: "nok".into(),
        guard: None,
        targets: vec!["ok".into()],
        on_interruption: Default::default(),
        monitored: false,
        cycle_group: None,
        distrib: Distrib::Delay { time: 2.0 },
    });
    model.targets = vec![Target {
        name: "nok_target".into(),
        component: "C".into(),
        automaton: "aut_ok_nok".into(),
        state: "nok".into(),
    }];
    let compiled = CompiledModel::compile(&model).unwrap();
    let base = McConfig {
        nb_runs: 1,
        seed: 0,
        t_max: 15.0,
        samples: vec![2.0, 4.0, 9.0, 14.0],
        threads: None,
        quantiles: vec![],
        ode: Default::default(),
        stop_at_targets: false,
    };
    // Free-cycling: nok over [3,5)∪[8,10)∪[13,15) → three entries, the
    // cumulated sojourn at 14 is 2 + 2 + 1 = 5.
    let free = &run(&compiled, &base).unwrap().indicators[0];
    assert_eq!(free.nb_occurrences_mean, vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(free.sojourn_mean[3], 5.0);
    // Early-stopped: frozen in nok at the t=3 hit — one occurrence, the
    // state holds through every later instant, sojourn at 14 = 14 − 3 = 11.
    let latched = &run(
        &compiled,
        &McConfig {
            stop_at_targets: true,
            quantiles: vec![],
            ode: Default::default(),
            ..base
        },
    )
    .unwrap()
    .indicators[0];
    assert_eq!(latched.nb_occurrences_mean, vec![0.0, 1.0, 1.0, 1.0]);
    assert_eq!(latched.mean, vec![0.0, 1.0, 1.0, 1.0]);
    assert_eq!(latched.sojourn_mean[3], 11.0);
}

/// Review finding: an early-stopped replica under an INFINITE horizon must
/// still cover every requested sample instant (latch through the last one) —
/// previously the truncated series made the reduction index out of bounds
/// (a panic on the library path).
#[test]
fn stop_at_targets_with_infinite_horizon_covers_the_schedule() {
    use raichu_model::Target;
    let mut model = exp_ok_nok(0.5);
    model.targets = vec![Target {
        name: "nok_target".into(),
        component: "C".into(),
        automaton: "aut_ok_nok".into(),
        state: "nok".into(),
    }];
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = McConfig {
        nb_runs: 50,
        seed: 9,
        t_max: f64::INFINITY,
        samples: vec![10.0, 20.0],
        threads: None,
        quantiles: vec![],
        ode: Default::default(),
        stop_at_targets: true,
    };
    let est = &run(&compiled, &config).unwrap().indicators[0];
    // No panic, both instants estimated; the latch makes the feared state
    // sticky, so the probability is non-decreasing.
    assert_eq!(est.mean.len(), 2);
    assert!(est.mean[0] <= est.mean[1] + 1e-12);
    assert!(est.nb_occurrences_mean.iter().all(|v| *v <= 1.0));
}

/// Review finding: an entry happening EXACTLY at a sample instant counts in
/// the sampled value but was excluded from nb-occurrences — the two measures
/// must agree at the boundary.
#[test]
fn nb_occurrences_includes_an_entry_at_the_sample_instant() {
    let mut model = exp_ok_nok(0.0);
    model.components[0].automata[0].transitions[0].distrib = Distrib::Delay { time: 3.0 };
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = McConfig {
        nb_runs: 1,
        seed: 0,
        t_max: 10.0,
        samples: vec![3.0, 5.0],
        threads: None,
        quantiles: vec![],
        ode: Default::default(),
        stop_at_targets: false,
    };
    let est = &run(&compiled, &config).unwrap().indicators[0];
    // The sampled value at t=3 reflects the post-event state (1.0) — the
    // occurrence count at the same instant must agree.
    assert_eq!(est.mean, vec![1.0, 1.0]);
    assert_eq!(est.nb_occurrences_mean, vec![1.0, 1.0]);
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
        stop_at_targets: false,
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
        stop_at_targets: false,
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
