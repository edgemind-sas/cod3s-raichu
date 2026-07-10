//! Stochastic instantaneous branching (brique 2 — Bernoulli-on-demand
//! foundation): a `CLaw::Inst` whose branching is non-deterministic
//! draws its destination from the categorical distribution over its
//! probabilities, using the engine RNG — seed-reproducible, and matching
//! the target frequencies over many independent draws.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_model::{Automaton, Component, Distrib, Model, Transition};

/// A one-shot instantaneous branching `resolve` from `pending` to either
/// `ok` (probability `ok_prob`) or `ko` (the complement).
fn demand_model(ok_prob: f64) -> Model {
    Model {
        name: "inst_demand".into(),
        components: vec![Component {
            name: "d".into(),
            attributes: vec![],
            ports: vec![],
            interfaces: vec![],
            equations: vec![],
            sensitive_functions: vec![],
            automata: vec![Automaton {
                name: "req".into(),
                states: vec!["pending".into(), "ok".into(), "ko".into()],
                init: "pending".into(),
                transitions: vec![Transition {
                    name: "resolve".into(),
                    source: "pending".into(),
                    guard: None,
                    targets: vec!["ok".into(), "ko".into()],
                    on_interruption: Default::default(),
                    distrib: Distrib::Inst {
                        probs: vec![ok_prob],
                    },
                }],
            }],
        }],
        connections: vec![],
        indicators: vec![],
    }
}

/// Fire the single `resolve` draw on a fresh engine seeded with
/// `(seed, stream)` and return the destination state name.
fn draw(compiled: &CompiledModel, seed: u64, stream: u64) -> String {
    let config = EngineConfig {
        seed,
        rng_stream: stream,
        ..Default::default()
    };
    let mut engine = Engine::new(compiled, config).unwrap();
    let event = engine.step().unwrap().unwrap();
    event.to
}

#[test]
fn deterministic_inst_is_rng_free() {
    // A probability-1 branch resolves to that branch whatever the seed —
    // no RNG dependence, hence bit-identical replay.
    let compiled = CompiledModel::compile(&demand_model(1.0)).unwrap();
    for stream in 0..64 {
        assert_eq!(draw(&compiled, 42, stream), "ok");
    }
}

#[test]
fn stochastic_inst_draw_is_seed_reproducible() {
    let compiled = CompiledModel::compile(&demand_model(0.7)).unwrap();
    // Same (seed, stream) ⇒ identical outcome, however many times.
    let first = draw(&compiled, 7, 0);
    for _ in 0..8 {
        assert_eq!(draw(&compiled, 7, 0), first);
    }
    assert!(matches!(first.as_str(), "ok" | "ko"));
}

#[test]
fn stochastic_inst_draw_follows_the_categorical_distribution() {
    // Over many independent substreams the `ok` frequency matches the
    // branch probability. Seed-fixed ⇒ the test decision is reproducible.
    let gamma = 0.3;
    let compiled = CompiledModel::compile(&demand_model(gamma)).unwrap();
    let n = 5000u64;
    let occ = (0..n)
        .filter(|&s| draw(&compiled, 20260710, s) == "ok")
        .count();
    let freq = occ as f64 / n as f64;
    // 5σ envelope of a Bernoulli(gamma) mean over n draws (~0.032 here),
    // comfortably wide for the fixed-seed empirical frequency.
    let sigma = (gamma * (1.0 - gamma) / n as f64).sqrt();
    assert!(
        (freq - gamma).abs() < 5.0 * sigma,
        "ok frequency {freq} deviates from gamma {gamma} by more than 5σ ({})",
        5.0 * sigma
    );
}

#[test]
fn three_way_branching_covers_every_target() {
    // probs = [0.2, 0.3] ⇒ full [0.2, 0.3, 0.5]: all three branches are
    // reachable across streams (the inverse-CDF walk hits each interval).
    let model = Model {
        name: "inst_three".into(),
        components: vec![Component {
            name: "d".into(),
            attributes: vec![],
            ports: vec![],
            interfaces: vec![],
            equations: vec![],
            sensitive_functions: vec![],
            automata: vec![Automaton {
                name: "req".into(),
                states: vec!["p".into(), "a".into(), "b".into(), "c".into()],
                init: "p".into(),
                transitions: vec![Transition {
                    name: "resolve".into(),
                    source: "p".into(),
                    guard: None,
                    targets: vec!["a".into(), "b".into(), "c".into()],
                    on_interruption: Default::default(),
                    distrib: Distrib::Inst {
                        probs: vec![0.2, 0.3],
                    },
                }],
            }],
        }],
        connections: vec![],
        indicators: vec![],
    };
    let compiled = CompiledModel::compile(&model).unwrap();
    let mut seen = std::collections::BTreeSet::new();
    for stream in 0..200 {
        seen.insert(draw(&compiled, 1, stream));
    }
    assert_eq!(
        seen,
        ["a", "b", "c"].iter().map(|s| s.to_string()).collect()
    );
}
