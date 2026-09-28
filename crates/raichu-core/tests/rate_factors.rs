//! Rate factors and exposure statistics in the drawn mode: the engine
//! seam of the biased Monte-Carlo sampler fitted by cross-entropy.
//!
//! A factor multiplies the rate of a constant-rate exponential
//! transition when its date is drawn (`schedule_stochastic`, the paper's
//! `schST`). With a factor vector present, the run also reports, per
//! transition, its firing count and its nominal exposure: the nominal
//! rate times the time it spent armed and running, following the
//! interruption policy (`drop_disabled`, the paper's `updateIT`). Those
//! two numbers are the sufficient statistics of the likelihood ratio.
//!
//! The first test is the proof everything else rests on: a factor of 1
//! on every transition leaves the trajectory bit-identical to a run
//! with no factor at all.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::compile::CLaw;
use raichu_core::{
    CompiledModel, Engine, EngineConfig, EngineError, SimulationResult, StochasticDates,
    TransitionExposure,
};
use raichu_model::Model;
use serde_json::{json, Value as Json};

// ---- model builders -------------------------------------------------------

fn active(component: &str, automaton: &str, state: &str) -> Json {
    json!({"op": "state_active",
           "state": {"component": component, "automaton": automaton, "state": state}})
}

fn tr(name: &str, source: &str, targets: &[&str], law: Json) -> Json {
    let mut t = json!({"name": name, "source": source, "targets": targets});
    for (k, v) in law.as_object().unwrap() {
        t[k] = v.clone();
    }
    t
}

fn with(mut t: Json, key: &str, value: Json) -> Json {
    t[key] = value;
    t
}

fn automaton(name: &str, states: &[&str], transitions: Vec<Json>) -> Json {
    json!({"name": name, "states": states, "init": states[0], "transitions": transitions})
}

fn component(name: &str, automata: Vec<Json>) -> Json {
    json!({"name": name, "ports": [], "attributes": [], "automata": automata, "equations": []})
}

fn repairable(name: &str, fail: Json, repair: Json) -> Json {
    component(
        name,
        vec![automaton(
            "fail",
            &["ok", "nok"],
            vec![
                tr("occ", "ok", &["nok"], fail),
                tr("rep", "nok", &["ok"], repair),
            ],
        )],
    )
}

fn build(name: &str, components: Vec<Json>, extra: Option<(&str, Json)>) -> Model {
    let mut model = json!({"name": name, "components": components, "indicators": []});
    if let Some((key, value)) = extra {
        model[key] = value;
    }
    Model::from_json(&model.to_string()).unwrap()
}

fn exp(rate: f64) -> Json {
    json!({"distrib": "exp", "rate": rate})
}

fn delay(time: f64) -> Json {
    json!({"distrib": "delay", "time": time})
}

fn weibull(shape: f64, scale: f64) -> Json {
    json!({"distrib": "weibull", "shape": shape, "scale": scale})
}

/// A repairable pair with exponential failures and repairs, plus a
/// Weibull-failing third component and a deterministic repair, so a
/// dense factor vector meets every kind of law.
fn pair_model() -> Model {
    build(
        "pair",
        vec![
            repairable("A", exp(0.3), exp(1.5)),
            repairable("B", exp(0.2), exp(0.8)),
            repairable("C", weibull(2.0, 5.0), delay(1.0)),
        ],
        None,
    )
}

/// One transition `W.job.finish` at rate 0.5, armed from t = 0 while the
/// gate `G.phase` is `up`, under `policy`. The gate goes `down` at
/// `down_at` and back `up` after `down_for`, then stays up.
fn gated_model(policy: &str, down_at: f64, down_for: f64) -> Model {
    let gate = component(
        "G",
        vec![automaton(
            "phase",
            &["up", "down", "final"],
            vec![
                tr("close", "up", &["down"], delay(down_at)),
                tr("open", "down", &["final"], delay(down_for)),
            ],
        )],
    );
    let finish = with(
        with(
            tr("finish", "wait", &["done"], exp(0.5)),
            "guard",
            json!({"op": "bool", "bool_op": "or", "args": [
                active("G", "phase", "up"), active("G", "phase", "final")]}),
        ),
        "on_interruption",
        json!(policy),
    );
    let worker = component("W", vec![automaton("job", &["wait", "done"], vec![finish])]);
    build("gated", vec![gate, worker], None)
}

// ---- helpers --------------------------------------------------------------

fn index(compiled: &CompiledModel, name: &str) -> usize {
    compiled
        .transitions
        .iter()
        .position(|t| t.name == name)
        .unwrap_or_else(|| panic!("no transition `{name}`"))
}

/// A dense factor vector: 1 everywhere, `factor` on the named ones.
fn factors(compiled: &CompiledModel, biased: &[(&str, f64)]) -> Vec<f64> {
    let mut out = vec![1.0; compiled.transitions.len()];
    for (name, factor) in biased {
        out[index(compiled, name)] = *factor;
    }
    out
}

fn run(compiled: &CompiledModel, config: EngineConfig) -> SimulationResult {
    Engine::new(compiled, config).unwrap().run().unwrap()
}

fn stat(result: &SimulationResult, compiled: &CompiledModel, name: &str) -> TransitionExposure {
    result
        .rate_statistics
        .as_ref()
        .expect("statistics requested")[index(compiled, name)]
}

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0)
}

/// A factor small enough that a nominal 0.5 transition never fires
/// before t = 10 at the fixed seed: its exposure is then deterministic.
const NEVER: f64 = 1e-12;

// ---- bit identity ---------------------------------------------------------

#[test]
fn factor_one_everywhere_is_bit_identical_to_no_factor() {
    let compiled = CompiledModel::compile(&pair_model()).unwrap();
    let ones = vec![1.0; compiled.transitions.len()];
    for seed in 0..20_u64 {
        for stream in 0..3_u64 {
            let base = EngineConfig {
                t_max: 200.0,
                seed,
                rng_stream: stream,
                ..EngineConfig::default()
            };
            let plain = run(&compiled, base.clone());
            let unit = run(
                &compiled,
                EngineConfig {
                    rate_factors: ones.clone(),
                    ..base
                },
            );
            assert!(plain.events.len() > 10, "a non-trivial trajectory");
            assert_eq!(plain.events.len(), unit.events.len());
            for (a, b) in plain.events.iter().zip(&unit.events) {
                assert_eq!(a.time.to_bits(), b.time.to_bits(), "date moved");
                assert_eq!(
                    (&a.transition, &a.from, &a.to),
                    (&b.transition, &b.from, &b.to)
                );
            }
            assert_eq!(plain.final_time.to_bits(), unit.final_time.to_bits());
            assert!(plain.rate_statistics.is_none());
            assert!(unit.rate_statistics.is_some());
        }
    }
}

#[test]
fn a_factor_divides_the_drawn_delay_from_the_same_random_number() {
    let model = build(
        "one",
        vec![component(
            "W",
            vec![automaton(
                "job",
                &["wait", "done"],
                vec![tr("finish", "wait", &["done"], exp(0.5))],
            )],
        )],
        None,
    );
    let compiled = CompiledModel::compile(&model).unwrap();
    for seed in 0..10_u64 {
        let nominal = run(
            &compiled,
            EngineConfig {
                seed,
                ..EngineConfig::default()
            },
        );
        let biased = run(
            &compiled,
            EngineConfig {
                seed,
                rate_factors: vec![4.0],
                ..EngineConfig::default()
            },
        );
        let (t1, t4) = (nominal.events[0].time, biased.events[0].time);
        assert!((t4 * 4.0 - t1).abs() <= 1e-12 * t1, "{t4} x 4 != {t1}");
    }
}

// ---- statistics: firing and censoring -------------------------------------

#[test]
fn statistics_are_absent_without_a_factor_vector() {
    let compiled = CompiledModel::compile(&pair_model()).unwrap();
    let engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 10.0,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    assert!(engine.rate_statistics().is_none());
    assert!(engine.run().unwrap().rate_statistics.is_none());
}

#[test]
fn a_fired_transition_reports_one_firing_and_its_nominal_exposure() {
    let model = build(
        "one",
        vec![component(
            "W",
            vec![automaton(
                "job",
                &["wait", "done"],
                vec![tr("finish", "wait", &["done"], exp(0.5))],
            )],
        )],
        None,
    );
    let compiled = CompiledModel::compile(&model).unwrap();
    for seed in 0..10_u64 {
        let result = run(
            &compiled,
            EngineConfig {
                seed,
                rate_factors: vec![4.0],
                ..EngineConfig::default()
            },
        );
        let fired_at = result.events[0].time;
        let s = stat(&result, &compiled, "W.job.finish");
        assert_eq!(s.firings, 1);
        // Nominal, not biased: the rate 0.5, not 4 x 0.5.
        assert!(
            close(s.exposure, 0.5 * fired_at),
            "{} vs {}",
            s.exposure,
            0.5 * fired_at
        );
    }
}

#[test]
fn a_transition_that_never_fires_is_censored_at_the_horizon() {
    let model = gated_model("reset", 100.0, 1.0);
    let compiled = CompiledModel::compile(&model).unwrap();
    let result = run(
        &compiled,
        EngineConfig {
            t_max: 10.0,
            rate_factors: factors(&compiled, &[("W.job.finish", NEVER)]),
            ..EngineConfig::default()
        },
    );
    let s = stat(&result, &compiled, "W.job.finish");
    assert_eq!(s.firings, 0);
    assert!(close(s.exposure, 5.0), "{}", s.exposure);
    // A deterministic transition fires, and is counted, with no exposure.
    let g = stat(&result, &compiled, "G.phase.close");
    assert_eq!((g.firings, g.exposure), (0, 0.0));
}

#[test]
fn leaving_the_source_closes_exposure() {
    // `W.job.quit` leaves `wait` at t = 4: `finish` stops accruing there.
    let quit = tr("quit", "wait", &["gone"], delay(4.0));
    let finish = tr("finish", "wait", &["done"], exp(0.5));
    let model = build(
        "exit",
        vec![component(
            "W",
            vec![automaton(
                "job",
                &["wait", "done", "gone"],
                vec![finish, quit],
            )],
        )],
        None,
    );
    let compiled = CompiledModel::compile(&model).unwrap();
    let result = run(
        &compiled,
        EngineConfig {
            t_max: 10.0,
            rate_factors: factors(&compiled, &[("W.job.finish", NEVER)]),
            ..EngineConfig::default()
        },
    );
    let s = stat(&result, &compiled, "W.job.finish");
    assert_eq!(s.firings, 0);
    assert!(close(s.exposure, 0.5 * 4.0), "{}", s.exposure);
    assert_eq!(stat(&result, &compiled, "W.job.quit").firings, 1);
}

#[test]
fn an_early_stop_at_a_target_censors_exposure_at_the_hit() {
    let quit = tr("quit", "wait", &["gone"], delay(3.0));
    let other = component(
        "V",
        vec![automaton(
            "job",
            &["wait", "done"],
            vec![tr("finish", "wait", &["done"], exp(0.5))],
        )],
    );
    let model = build(
        "stop",
        vec![
            component("W", vec![automaton("job", &["wait", "gone"], vec![quit])]),
            other,
        ],
        Some((
            "targets",
            json!([{"name": "lost", "component": "W", "automaton": "job", "state": "gone"}]),
        )),
    );
    let compiled = CompiledModel::compile(&model).unwrap();
    let result = run(
        &compiled,
        EngineConfig {
            t_max: 10.0,
            stop_at_targets: true,
            rate_factors: factors(&compiled, &[("V.job.finish", NEVER)]),
            ..EngineConfig::default()
        },
    );
    assert_eq!(result.final_time, 3.0);
    let s = stat(&result, &compiled, "V.job.finish");
    assert!(close(s.exposure, 0.5 * 3.0), "{}", s.exposure);
}

// ---- statistics: interruption policies ------------------------------------

fn gated_exposure(policy: &str, down_at: f64, down_for: f64) -> TransitionExposure {
    let compiled = CompiledModel::compile(&gated_model(policy, down_at, down_for)).unwrap();
    let result = run(
        &compiled,
        EngineConfig {
            t_max: 10.0,
            rate_factors: factors(&compiled, &[("W.job.finish", NEVER)]),
            ..EngineConfig::default()
        },
    );
    stat(&result, &compiled, "W.job.finish")
}

#[test]
fn reset_drops_exposure_at_the_guard_and_rearms_a_new_episode() {
    // Guard false from 3 to 5, re-armed at 5 until the horizon 10.
    let s = gated_exposure("reset", 3.0, 2.0);
    assert_eq!(s.firings, 0);
    assert!(close(s.exposure, 0.5 * (3.0 + 5.0)), "{}", s.exposure);
}

#[test]
fn resume_excludes_the_paused_stretch() {
    // Paused from 2 to 6.
    let s = gated_exposure("resume", 2.0, 4.0);
    assert!(close(s.exposure, 0.5 * (2.0 + 4.0)), "{}", s.exposure);
}

#[test]
fn continue_includes_the_false_guard_stretch() {
    let s = gated_exposure("continue", 2.0, 4.0);
    assert!(close(s.exposure, 0.5 * 10.0), "{}", s.exposure);
}

// ---- refusals -------------------------------------------------------------

fn refusal(compiled: &CompiledModel, rate_factors: Vec<f64>) -> EngineError {
    match Engine::new(
        compiled,
        EngineConfig {
            rate_factors,
            ..EngineConfig::default()
        },
    ) {
        Ok(_) => panic!("the factor vector should be refused"),
        Err(e) => e,
    }
}

#[test]
fn a_factor_other_than_one_on_a_non_exponential_law_is_refused_by_name() {
    let compiled = CompiledModel::compile(&pair_model()).unwrap();
    let weibull = index(&compiled, "C.fail.occ");
    assert!(matches!(
        compiled.transitions[weibull].distrib,
        CLaw::Weibull(..)
    ));
    let err = refusal(&compiled, factors(&compiled, &[("C.fail.occ", 2.0)]));
    match &err {
        EngineError::InvalidRateFactor {
            transition, factor, ..
        } => {
            assert_eq!(transition, "C.fail.occ");
            assert_eq!(*factor, 2.0);
        }
        other => panic!("unexpected error {other:?}"),
    }
    assert!(err.to_string().contains("C.fail.occ"));
    // A factor of exactly 1 is accepted on any law (dense vectors).
    let ok = Engine::new(
        &compiled,
        EngineConfig {
            rate_factors: factors(&compiled, &[("C.fail.occ", 1.0)]),
            ..EngineConfig::default()
        },
    );
    assert!(ok.is_ok());
}

#[test]
fn a_zero_negative_or_non_finite_factor_is_refused_by_name() {
    let compiled = CompiledModel::compile(&pair_model()).unwrap();
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let err = refusal(&compiled, factors(&compiled, &[("A.fail.occ", bad)]));
        match err {
            EngineError::InvalidRateFactor { transition, .. } => {
                assert_eq!(transition, "A.fail.occ");
            }
            other => panic!("unexpected error {other:?} for {bad}"),
        }
    }
}

#[test]
fn a_factor_vector_of_the_wrong_length_is_refused() {
    let compiled = CompiledModel::compile(&pair_model()).unwrap();
    let err = refusal(&compiled, vec![1.0; 2]);
    match err {
        EngineError::RateFactorCount { expected, found } => {
            assert_eq!((expected, found), (compiled.transitions.len(), 2));
        }
        other => panic!("unexpected error {other:?}"),
    }
}

#[test]
fn rate_factors_are_refused_in_deferred_mode() {
    let compiled = CompiledModel::compile(&pair_model()).unwrap();
    let err = Engine::new(
        &compiled,
        EngineConfig {
            rate_factors: vec![1.0; compiled.transitions.len()],
            stochastic_dates: StochasticDates::Deferred,
            ..EngineConfig::default()
        },
    )
    .err()
    .expect("refused");
    assert!(
        matches!(&err, EngineError::InvalidStudyParameter { parameter, .. } if parameter == "rate_factors"),
        "{err:?}"
    );
}

// ---- snapshots ------------------------------------------------------------

#[test]
fn a_restored_snapshot_continues_with_the_same_statistics() {
    let compiled = CompiledModel::compile(&pair_model()).unwrap();
    let config = EngineConfig {
        t_max: 50.0,
        seed: 7,
        rate_factors: factors(&compiled, &[("A.fail.occ", 3.0), ("B.fail.occ", 0.5)]),
        ..EngineConfig::default()
    };
    let straight = run(&compiled, config.clone());

    let mut engine = Engine::new(&compiled, config.clone()).unwrap();
    for _ in 0..5 {
        engine.step().unwrap();
    }
    let mid = engine.snapshot().unwrap();
    let at_mid = engine.rate_statistics().unwrap();
    for _ in 0..5 {
        engine.step().unwrap();
    }
    engine.restore(&mid).unwrap();
    assert_eq!(engine.rate_statistics().unwrap(), at_mid);
    let restored = engine.run().unwrap();
    assert_eq!(restored.rate_statistics, straight.rate_statistics);
    assert_eq!(restored.events, straight.events);

    // The facade route: a throwaway engine rebuilt from the snapshot.
    let rebuilt = Engine::from_snapshot(&compiled, config, &mid)
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(rebuilt.rate_statistics, straight.rate_statistics);

    // Firings in the statistics match the recorded events.
    let stats = straight.rate_statistics.as_ref().unwrap();
    for (idx, t) in compiled.transitions.iter().enumerate() {
        let fired = straight
            .events
            .iter()
            .filter(|e| e.transition == t.name)
            .count();
        assert_eq!(stats[idx].firings, fired as u64, "{}", t.name);
    }
}
