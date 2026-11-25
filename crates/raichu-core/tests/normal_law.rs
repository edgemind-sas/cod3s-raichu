//! The normal law of a transition is truncated to its interval, not
//! clamped to it.
//!
//! A delay cannot be negative, so a normal law needs a lower bound, 0
//! unless stated, and may take an upper one. A draw outside the interval
//! is not moved to the nearest bound, which would pile probability on
//! it: with mean 10 and deviation 10, one history in six would fire at
//! the very instant the transition is armed. The delay follows the
//! normal law conditioned on the interval instead.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompileError, CompiledModel, Engine, EngineConfig};
use raichu_model::{Model, ModelError};
use serde_json::{json, Value as Json};

fn model(law: &Json) -> Result<Model, serde_json::Error> {
    let mut transition = json!({"name": "fire", "source": "ok", "targets": ["nok"]});
    for (key, value) in law.as_object().expect("law fields") {
        transition[key] = value.clone();
    }
    serde_json::from_value(json!({
        "name": "normal",
        "components": [{"name": "c", "automata": [{
            "name": "a", "states": ["ok", "nok"], "init": "ok",
            "transitions": [transition]}]}]
    }))
}

fn firing_date(law: &Json, seed: u64) -> f64 {
    let compiled = CompiledModel::compile(&model(law).unwrap()).unwrap();
    let result = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 1.0e6,
            seed,
            ..EngineConfig::default()
        },
    )
    .unwrap()
    .run()
    .unwrap();
    assert_eq!(result.events.len(), 1, "seed {seed}");
    result.events[0].time
}

#[test]
fn no_draw_lands_on_the_lower_bound() {
    // Mean 10, deviation 10, bounds left to their defaults: 0 and none.
    let law = json!({"distrib": "normal", "mu": 10.0, "sigma": 10.0});
    let n = 4_000_u32;
    let (mut sum, mut below_one) = (0.0, 0_u32);
    for seed in 0..u64::from(n) {
        let date = firing_date(&law, seed);
        assert!(date > 0.0, "seed {seed}: a draw at the bound");
        sum += date;
        below_one += u32::from(date < 1.0);
    }
    // The truncated mean is 12.876 (closed form), its deviation 7.91;
    // clamping would give 10.83, eight standard errors away.
    let mean = sum / f64::from(n);
    assert!(
        (mean - 12.876).abs() < 4.0 * 7.91 / f64::from(n).sqrt(),
        "mean {mean}"
    );
    // F(1) = 0.0302: about 121 histories, where clamping would put 16 %.
    assert!(
        (80..170).contains(&below_one),
        "{below_one} histories below 1"
    );
}

#[test]
fn both_bounds_are_respected() {
    let law = json!({"distrib": "normal", "mu": 2.0, "sigma": 3.0, "low": 1.0, "high": 6.0});
    let n = 2_000_u32;
    let mut sum = 0.0;
    for seed in 0..u64::from(n) {
        let date = firing_date(&law, seed);
        assert!(date > 1.0 && date < 6.0, "seed {seed}: {date}");
        sum += date;
    }
    // Truncated mean 3.1868, deviation below the interval's 5 / sqrt(12).
    let mean = sum / f64::from(n);
    assert!(
        (mean - 3.1868).abs() < 4.0 * 1.45 / f64::from(n).sqrt(),
        "mean {mean}"
    );
}

#[test]
fn a_seed_gives_one_date() {
    let law = json!({"distrib": "normal", "mu": 100.0, "sigma": 5.0});
    assert_eq!(
        firing_date(&law, 7).to_bits(),
        firing_date(&law, 7).to_bits()
    );
    assert_ne!(firing_date(&law, 7), firing_date(&law, 8));
}

#[test]
fn the_law_is_written_back_without_its_default_bounds() {
    let model = model(&json!({"distrib": "normal", "mu": 100.0, "sigma": 5.0})).unwrap();
    let written = serde_json::to_value(&model).unwrap();
    let law = &written["components"][0]["automata"][0]["transitions"][0];
    assert_eq!(law["distrib"], "normal");
    assert!(
        law.get("low").is_none() && law.get("high").is_none(),
        "{law}"
    );
}

fn refusal(law: &Json) -> ModelError {
    match CompiledModel::compile(&model(law).unwrap()) {
        Err(CompileError::Invalid(error)) => error,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn invalid_parameters_are_refused_by_name() {
    let normal = |extra: Json| {
        let mut law = json!({"distrib": "normal", "mu": 10.0, "sigma": 2.0});
        for (key, value) in extra.as_object().unwrap() {
            law[key] = value.clone();
        }
        law
    };
    for (law, parameter) in [
        (normal(json!({"sigma": 0.0})), "sigma"),
        (normal(json!({"sigma": -1.0})), "sigma"),
        (normal(json!({"low": -1.0})), "low"),
        (normal(json!({"low": 5.0, "high": 5.0})), "high"),
        (normal(json!({"low": 5.0, "high": 4.0})), "high"),
    ] {
        match refusal(&law) {
            ModelError::InvalidLawParameter {
                distribution,
                parameter: named,
                ..
            } => assert_eq!((distribution, named), ("normal", parameter)),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn an_interval_the_law_gives_nothing_to_is_refused() {
    // 50 deviations above the mean: nothing to renormalise.
    let error = refusal(&json!({"distrib": "normal", "mu": 0.0, "sigma": 1.0, "low": 50.0}));
    assert!(
        matches!(error, ModelError::NormalIntervalWithoutMass { .. }),
        "{error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains("c.a") && message.contains("fire"),
        "{message}"
    );
}
