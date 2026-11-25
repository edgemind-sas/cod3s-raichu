//! An exponential law of rate 0 is a transition that cannot fire.
//!
//! A fixed rate of 0 used to be refused when the model was validated, so
//! a study neutralising a failure mode by setting its rate to 0 did not
//! load at all, while the same rate written as an expression ran. It is
//! now read as what it says: the transition is armed and never fires.
//! The reference engine reads it the same way (measured on its
//! distributed version 1.4.1.0: 0 firing in 2 000 runs to t = 1e6).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompileError, CompiledModel, Engine, EngineConfig};
use raichu_model::{Model, ModelError};
use serde_json::{json, Value as Json};

fn unit(name: &str, rate: f64) -> Json {
    json!({"name": name, "automata": [{
        "name": "health", "states": ["ok", "ko"], "init": "ok",
        "transitions": [{"name": "fail", "source": "ok", "targets": ["ko"],
                         "distrib": "exp", "rate": rate}]}]})
}

fn events(components: Vec<Json>, seed: u64) -> Vec<(f64, String)> {
    let model: Model =
        serde_json::from_value(json!({"name": "m", "components": components})).unwrap();
    let compiled = CompiledModel::compile(&model).unwrap();
    let result = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 1.0e9,
            seed,
            ..EngineConfig::default()
        },
    )
    .unwrap()
    .run()
    .unwrap();
    result
        .events
        .iter()
        .map(|e| (e.time, e.transition.clone()))
        .collect()
}

#[test]
fn a_zero_rate_never_fires() {
    for seed in 0..200 {
        assert!(events(vec![unit("a", 0.0)], seed).is_empty(), "seed {seed}");
    }
}

#[test]
fn a_zero_rate_draws_no_random_number() {
    // The date of `b` is the same with a dormant transition armed before
    // it: a transition that cannot fire consumes nothing of the stream.
    for seed in 0..20 {
        let alone = events(vec![unit("b", 0.5)], seed);
        let beside = events(vec![unit("a", 0.0), unit("b", 0.5)], seed);
        assert_eq!(beside.len(), 1, "seed {seed}: {beside:?}");
        assert_eq!(beside[0].1, "b.health.fail");
        assert_eq!(beside[0].0.to_bits(), alone[0].0.to_bits(), "seed {seed}");
    }
}

#[test]
fn a_negative_or_non_finite_rate_is_still_refused() {
    for rate in [-1.0, f64::NAN, f64::INFINITY] {
        // serde_json writes a non-finite number as null: build the model
        // from its types rather than from text.
        let mut model: Model =
            serde_json::from_value(json!({"name": "m", "components": [unit("a", 1.0)]})).unwrap();
        model.components[0].automata[0].transitions[0].distrib = raichu_model::Distrib::Exp {
            rate: Some(rate),
            rate_expr: None,
        };
        match CompiledModel::compile(&model) {
            Err(CompileError::Invalid(ModelError::InvalidExpRate { .. })) => {}
            other => panic!("rate {rate}: {other:?}"),
        }
    }
}
