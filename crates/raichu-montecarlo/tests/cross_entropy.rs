//! The biased Monte-Carlo campaign fitted by cross-entropy.
//!
//! Every number asserted below is a closed form of a small model: a
//! parallel pair of non-repairable units, whose feared event is both
//! down by the horizon, has `P = (1 - e^{-at})(1 - e^{-bt})`. At failure
//! rates of 1e-3 over a horizon of 10 that is about 1e-4, and at 1e-5
//! about 1e-8: far beyond what a plain campaign of a few thousand
//! replicas can see.
//!
//! The first proof is structural rather than statistical: with every
//! factor fixed at 1 and no fitting, the campaign is the Monte-Carlo
//! one, bit for bit (same streams, same draws, every weight equal to 1).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use raichu_core::{CompiledModel, SolverParams};
use raichu_expr::{BoolOp, Expr, StateRef};
use raichu_model::{Automaton, Component, Distrib, Model, Target, Transition, TransitionKind};
use raichu_montecarlo::{
    cross_entropy_families, run_cross_entropy, run_to_targets, CrossEntropyError,
    CrossEntropySettings, McConfig, DEFAULT_CONFIDENCE,
};

// ---- model literals ------------------------------------------------------

fn transition(name: &str, source: &str, targets: &[&str], distrib: Distrib) -> Transition {
    Transition {
        name: name.into(),
        source: source.into(),
        guard: None,
        targets: targets.iter().map(|t| (*t).into()).collect(),
        on_interruption: Default::default(),
        monitored: false,
        cycle_group: None,
        kind: None,
        effects: vec![],
        distrib,
    }
}

fn component(name: &str, automaton: Automaton) -> Component {
    Component {
        name: name.into(),
        attributes: vec![],
        ports: vec![],
        interfaces: vec![],
        automata: vec![automaton],
        allocations: vec![],
        equations: vec![],
        sensitive_functions: vec![],
    }
}

fn exp(rate: f64) -> Distrib {
    Distrib::Exp {
        rate: Some(rate),
        rate_expr: None,
    }
}

/// A unit `name`: automaton `fail` (`ok` -> `nok` under `law`), declared
/// a failure when `kind` says so, repaired back at `repair` when given.
fn unit(name: &str, law: Distrib, kind: Option<TransitionKind>, repair: Option<f64>) -> Component {
    let mut occ = transition("occ", "ok", &["nok"], law);
    occ.kind = kind;
    let mut transitions = vec![occ];
    if let Some(mu) = repair {
        let mut rep = transition("rep", "nok", &["ok"], exp(mu));
        rep.kind = Some(TransitionKind::Repair);
        transitions.push(rep);
    }
    component(
        name,
        Automaton {
            name: "fail".into(),
            states: vec!["ok".into(), "nok".into()],
            init: "ok".into(),
            transitions,
        },
    )
}

fn down(c: &str) -> Expr {
    Expr::StateActive {
        state: StateRef {
            component: c.into(),
            automaton: "fail".into(),
            state: "nok".into(),
        },
    }
}

/// A watcher entering `down` as soon as `guard` holds: the feared event
/// `feared`.
fn watcher(guard: Expr) -> Component {
    let mut to_down = transition("down", "ok", &["down"], Distrib::Inst { probs: vec![] });
    to_down.guard = Some(guard);
    component(
        "sys",
        Automaton {
            name: "watch".into(),
            states: vec!["ok".into(), "down".into()],
            init: "ok".into(),
            transitions: vec![to_down],
        },
    )
}

fn model(name: &str, mut components: Vec<Component>, guard: Expr) -> Model {
    components.push(watcher(guard));
    Model {
        name: name.into(),
        components,
        connections: vec![],
        indicators: vec![],
        targets: vec![Target {
            name: "feared".into(),
            component: "sys".into(),
            automaton: "watch".into(),
            state: "down".into(),
        }],
        evaluation_order: None,
        unbounded_rate: None,
    }
}

fn both(a: &str, b: &str) -> Expr {
    Expr::Bool {
        bool_op: BoolOp::And,
        args: vec![down(a), down(b)],
    }
}

/// Parallel pair A (rate a), B (rate b), non repairable, failures declared.
fn parallel_pair(a: f64, b: f64) -> Model {
    model(
        "parallel_pair",
        vec![
            unit("A", exp(a), Some(TransitionKind::Failure), None),
            unit("B", exp(b), Some(TransitionKind::Failure), None),
        ],
        both("A", "B"),
    )
}

/// `P(both down by t) = (1 - e^{-at}) (1 - e^{-bt})`.
fn both_down(a: f64, b: f64, t: f64) -> f64 {
    (-(-a * t).exp_m1()) * (-(-b * t).exp_m1())
}

fn compile(model: &Model) -> CompiledModel {
    CompiledModel::compile(model).unwrap()
}

const HORIZON: f64 = 10.0;

fn settings(nb_runs: u64, pilot_runs: u64) -> CrossEntropySettings {
    CrossEntropySettings {
        target: "feared".into(),
        t_max: HORIZON,
        seed: 11,
        nb_runs,
        pilot_runs,
        ..CrossEntropySettings::default()
    }
}

/// `|estimate - truth| <= 4` standard errors: a deterministic check on one
/// seed that a biased or mis-weighted estimator fails by orders of
/// magnitude, and that a correct one fails with probability under 1e-4.
fn assert_close(estimate: f64, standard_error: f64, truth: f64) {
    assert!(
        standard_error > 0.0,
        "a degenerate interval: se = {standard_error}"
    );
    assert!(
        (estimate - truth).abs() <= 4.0 * standard_error,
        "estimate {estimate:e} is {:.1} standard errors from the closed form {truth:e}",
        (estimate - truth).abs() / standard_error
    );
}

// ---- families ------------------------------------------------------------

#[test]
fn families_group_identical_components_and_split_rates_and_roles() {
    let m = model(
        "families",
        vec![
            unit("A", exp(0.01), Some(TransitionKind::Failure), None),
            unit("B", exp(0.01), Some(TransitionKind::Failure), None),
            unit("C", exp(0.02), Some(TransitionKind::Failure), None),
            unit("R", exp(0.01), Some(TransitionKind::Failure), Some(0.5)),
        ],
        both("A", "B"),
    );
    let families = cross_entropy_families(&compile(&m), &BTreeMap::new()).unwrap();
    let members: Vec<Vec<&str>> = families
        .iter()
        .map(|f| f.transitions.iter().map(String::as_str).collect())
        .collect();
    // A, B and R share the failure at 0.01; C fails at another rate; R's
    // repair is its own family; the watcher's instantaneous edge is not
    // eligible and belongs to none.
    assert_eq!(
        members,
        vec![
            vec!["A.fail.occ", "B.fail.occ", "R.fail.occ"],
            vec!["C.fail.occ"],
            vec!["R.fail.rep"],
        ]
    );
    assert_eq!(
        families.iter().map(|f| f.repair).collect::<Vec<_>>(),
        vec![false, false, true]
    );
}

#[test]
fn an_override_merges_transitions_and_refuses_unknown_or_ineligible_names() {
    let m = model(
        "override",
        vec![
            unit("A", exp(0.01), None, None),
            unit("C", exp(0.02), None, None),
            unit(
                "W",
                Distrib::Weibull {
                    shape: 2.0,
                    scale: 10.0,
                },
                None,
                None,
            ),
        ],
        both("A", "C"),
    );
    let compiled = compile(&m);
    let merged: BTreeMap<String, String> = [
        ("A.fail.occ".to_owned(), "all".to_owned()),
        ("C.fail.occ".to_owned(), "all".to_owned()),
    ]
    .into();
    let families = cross_entropy_families(&compiled, &merged).unwrap();
    assert_eq!(families.len(), 1);
    assert_eq!(families[0].label, "all");
    assert_eq!(families[0].transitions, vec!["A.fail.occ", "C.fail.occ"]);

    let unknown: BTreeMap<String, String> = [("Z.fail.occ".to_owned(), "x".to_owned())].into();
    let error = cross_entropy_families(&compiled, &unknown).unwrap_err();
    assert!(
        matches!(&error, CrossEntropyError::UnknownTransition { transition } if transition == "Z.fail.occ"),
        "{error}"
    );

    let weibull: BTreeMap<String, String> = [("W.fail.occ".to_owned(), "x".to_owned())].into();
    let error = cross_entropy_families(&compiled, &weibull).unwrap_err();
    assert!(
        matches!(&error, CrossEntropyError::IneligibleTransition { transition, .. } if transition == "W.fail.occ"),
        "{error}"
    );
}

// ---- the structural proof -------------------------------------------------

#[test]
fn factors_of_one_without_fitting_reproduce_the_monte_carlo_campaign_bit_for_bit() {
    let m = parallel_pair(0.1, 0.03);
    let compiled = compile(&m);
    let s = CrossEntropySettings {
        fit: false,
        initial_factor: 1.0,
        ..settings(2_000, 0)
    };
    let biased = run_cross_entropy(&compiled, &s).unwrap();
    let plain = run_to_targets(
        &compiled,
        &McConfig {
            nb_runs: 2_000,
            seed: 11,
            t_max: HORIZON,
            samples: vec![HORIZON],
            threads: None,
            quantiles: vec![],
            confidence: DEFAULT_CONFIDENCE,
            ode: SolverParams::default(),
            stop_at_targets: true,
            flow: Default::default(),
        },
    )
    .unwrap();
    assert_eq!(biased.reached, plain.count_reached("feared"));
    assert_eq!(biased.ends, plain.ends);
    assert!(biased.history.is_empty());
    // Every weight is exactly 1, so the estimate is the plain proportion.
    assert_eq!(biased.estimate.estimate, biased.reached as f64 / 2_000.0);
}

// ---- estimation against closed forms --------------------------------------

#[test]
fn a_rare_parallel_pair_is_estimated_within_its_interval() {
    let (a, b) = (1e-3, 1e-3);
    let truth = both_down(a, b, HORIZON);
    let result =
        run_cross_entropy(&compile(&parallel_pair(a, b)), &settings(4_000, 1_000)).unwrap();
    assert_close(
        result.estimate.estimate,
        result.estimate.standard_error,
        truth,
    );
    assert!(result.converged, "{:?}", result.history);
    assert!(
        !result.estimate_inconclusive,
        "ESS {}",
        result.estimate.effective_sample_size
    );
    assert!(
        result.families.iter().all(|f| f.factor > 10.0),
        "the fit should push both failure families well above 1: {:?}",
        result.families
    );
}

#[test]
fn escalation_reaches_a_target_no_pilot_sees_at_factor_one() {
    let (a, b) = (1e-5, 1e-5);
    let truth = both_down(a, b, HORIZON);
    let s = CrossEntropySettings {
        initial_factor: 1.0,
        max_iterations: 12,
        ..settings(4_000, 500)
    };
    let result = run_cross_entropy(&compile(&parallel_pair(a, b)), &s).unwrap();
    assert!(
        result
            .history
            .first()
            .is_some_and(|it| it.hits == 0 && it.escalated),
        "the first pilot at factor 1 should see nothing and escalate: {:?}",
        result.history
    );
    assert_close(
        result.estimate.estimate,
        result.estimate.standard_error,
        truth,
    );
}

#[test]
fn an_external_mode_with_no_declared_kind_is_escalated_too() {
    // X is a declared failure unrelated to the target; E, the only road to
    // it, declares no kind (an external failure mode's own edge does not).
    let m = model(
        "external",
        vec![
            unit("X", exp(1e-5), Some(TransitionKind::Failure), None),
            unit("E", exp(1e-5), None, None),
        ],
        down("E"),
    );
    let s = CrossEntropySettings {
        initial_factor: 1.0,
        max_iterations: 12,
        ..settings(2_000, 500)
    };
    let result = run_cross_entropy(&compile(&m), &s).unwrap();
    let truth = -(-1e-5 * HORIZON).exp_m1();
    assert_close(
        result.estimate.estimate,
        result.estimate.standard_error,
        truth,
    );
}

#[test]
fn a_repair_family_keeps_a_finite_factor_inside_the_bounds() {
    let m = model(
        "repairable_pair",
        vec![
            unit("A", exp(1e-3), Some(TransitionKind::Failure), Some(1.0)),
            unit("B", exp(1e-3), Some(TransitionKind::Failure), Some(1.0)),
        ],
        both("A", "B"),
    );
    let s = settings(2_000, 1_000);
    let result = run_cross_entropy(&compile(&m), &s).unwrap();
    for family in &result.families {
        assert!(
            family.factor.is_finite()
                && family.factor >= s.factor_min
                && family.factor <= s.factor_max,
            "{family:?}"
        );
    }
    assert!(result.estimate.estimate > 0.0);
}

// ---- refusals -------------------------------------------------------------

#[test]
fn an_unreachable_target_is_an_error_not_a_zero() {
    // The watcher waits for a unit that never fails (rate 0 is refused by
    // the model, so it waits for itself: a guard that is never true).
    let never = Expr::Bool {
        bool_op: BoolOp::And,
        args: vec![
            down("A"),
            Expr::Bool {
                bool_op: BoolOp::Not,
                args: vec![down("A")],
            },
        ],
    };
    let m = model(
        "unreachable",
        vec![unit("A", exp(0.1), Some(TransitionKind::Failure), None)],
        never,
    );
    let s = CrossEntropySettings {
        max_iterations: 4,
        ..settings(200, 100)
    };
    let error = run_cross_entropy(&compile(&m), &s).unwrap_err();
    assert!(
        matches!(&error, CrossEntropyError::NoHit { target, .. } if target == "feared"),
        "{error}"
    );
    assert!(error.to_string().contains("feared"), "{error}");
}

#[test]
fn a_final_campaign_with_no_hit_is_an_error() {
    let s = CrossEntropySettings {
        fit: false,
        initial_factor: 1.0,
        ..settings(100, 0)
    };
    let error = run_cross_entropy(&compile(&parallel_pair(1e-6, 1e-6)), &s).unwrap_err();
    assert!(
        matches!(&error, CrossEntropyError::NoHit { replicas: 100, .. }),
        "{error}"
    );
}

#[test]
fn an_unknown_target_and_invalid_settings_are_refused_by_name() {
    let compiled = compile(&parallel_pair(0.1, 0.1));
    let unknown = CrossEntropySettings {
        target: "nope".into(),
        ..settings(10, 10)
    };
    assert!(matches!(
        run_cross_entropy(&compiled, &unknown).unwrap_err(),
        CrossEntropyError::UnknownTarget { .. }
    ));
    for (name, s) in [
        (
            "nb_runs",
            CrossEntropySettings {
                nb_runs: 0,
                ..settings(10, 10)
            },
        ),
        (
            "pilot_runs",
            CrossEntropySettings {
                pilot_runs: 0,
                ..settings(10, 10)
            },
        ),
        (
            "smoothing",
            CrossEntropySettings {
                smoothing: 0.0,
                ..settings(10, 10)
            },
        ),
        (
            "factor_min",
            CrossEntropySettings {
                factor_min: 0.0,
                ..settings(10, 10)
            },
        ),
        (
            "escalation_ratio",
            CrossEntropySettings {
                escalation_ratio: 1.0,
                ..settings(10, 10)
            },
        ),
        (
            "t_max",
            CrossEntropySettings {
                t_max: 0.0,
                ..settings(10, 10)
            },
        ),
        (
            "max_iterations",
            CrossEntropySettings {
                max_iterations: 0,
                ..settings(10, 10)
            },
        ),
        (
            "max_iterations",
            CrossEntropySettings {
                max_iterations: (1 << 31) + 1,
                ..settings(10, 10)
            },
        ),
        (
            "tolerance",
            CrossEntropySettings {
                tolerance: -1.0,
                ..settings(10, 10)
            },
        ),
        (
            "confidence",
            CrossEntropySettings {
                confidence: 1.0,
                ..settings(10, 10)
            },
        ),
        (
            "factor_max",
            CrossEntropySettings {
                factor_max: 1e-3,
                ..settings(10, 10)
            },
        ),
        (
            "initial_factor",
            CrossEntropySettings {
                initial_factor: 1e9,
                ..settings(10, 10)
            },
        ),
        (
            "min_effective_sample_size",
            CrossEntropySettings {
                min_effective_sample_size: 0.0,
                ..settings(10, 10)
            },
        ),
    ] {
        let error = run_cross_entropy(&compiled, &s).unwrap_err();
        assert!(
            matches!(&error, CrossEntropyError::InvalidSetting { setting, .. } if setting == name),
            "{name}: {error}"
        );
    }
}

// ---- determinism ----------------------------------------------------------

#[test]
fn the_result_does_not_depend_on_the_thread_count() {
    let compiled = compile(&parallel_pair(1e-3, 2e-3));
    let one = run_cross_entropy(
        &compiled,
        &CrossEntropySettings {
            threads: Some(1),
            ..settings(1_000, 500)
        },
    )
    .unwrap();
    let eight = run_cross_entropy(
        &compiled,
        &CrossEntropySettings {
            threads: Some(8),
            ..settings(1_000, 500)
        },
    )
    .unwrap();
    assert_eq!(
        serde_json::to_string(&one).unwrap(),
        serde_json::to_string(&eight).unwrap()
    );
}

// ---- trustworthiness -------------------------------------------------------

/// A repairable pair with fast repairs over a long horizon: the regime a
/// fixed factor per family cannot bias well. The truth is about 2.0e-6.
fn fast_repair_pair() -> Model {
    model(
        "fast_repair_pair",
        vec![
            unit("A", exp(1e-3), Some(TransitionKind::Failure), Some(1000.0)),
            unit("B", exp(1e-3), Some(TransitionKind::Failure), Some(1000.0)),
        ],
        both("A", "B"),
    )
}

fn fast_repair_settings(nb_runs: u64, pilot_runs: u64) -> CrossEntropySettings {
    CrossEntropySettings {
        t_max: 1_000.0,
        seed: 1,
        ..settings(nb_runs, pilot_runs)
    }
}

#[test]
fn a_degenerate_pilot_is_neither_converged_nor_conclusive() {
    // Measured before the gate: 5 pilot hits carried by one replica were
    // declared converged, and the estimate came out at 4e-75.
    let result = run_cross_entropy(
        &compile(&fast_repair_pair()),
        &fast_repair_settings(300, 300),
    )
    .unwrap();
    assert!(
        !result.converged,
        "a pilot whose hits are carried by one replica cannot confirm a fit: {:?}",
        result.history
    );
    assert!(
        result.estimate_inconclusive,
        "ESS {}",
        result.estimate.effective_sample_size
    );
    for iteration in &result.history {
        assert!(iteration.effective_sample_size >= 0.0);
    }
}

#[test]
fn default_settings_on_the_fast_repair_pair_are_flagged_inconclusive() {
    // Measured before the flag: an interval 1500 times below the truth,
    // returned as an ordinary estimate.
    let result = run_cross_entropy(
        &compile(&fast_repair_pair()),
        &fast_repair_settings(10_000, 1_000),
    )
    .unwrap();
    assert!(
        result.estimate_inconclusive,
        "ESS {} should flag the estimate",
        result.estimate.effective_sample_size
    );
}

#[test]
fn a_fit_stopped_by_its_cap_is_not_converged() {
    let s = CrossEntropySettings {
        max_iterations: 1,
        tolerance: 0.0,
        ..settings(1_000, 500)
    };
    let result = run_cross_entropy(&compile(&parallel_pair(1e-3, 1e-3)), &s).unwrap();
    assert_eq!(result.history.len(), 1);
    assert!(!result.converged);
}
