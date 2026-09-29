//! The quantification contract on models with a closed form: the three
//! methods answer the same study consistently, each envelope carries its
//! engine's own result byte for byte, and the envelope reads back.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, EngineError};
use raichu_explore::{explore_discretised, explore_exact};
use raichu_expr::{AttrRef, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Distrib, FmuBinding, FmuUnit, Indicator,
    IndicatorTarget, Model, Target, Transition,
};
use raichu_montecarlo::IntervalMethod;
use raichu_quantify::{
    model_content_hash, quantify, quantify_with_fmu, read_quantification,
    CrossEntropySamplingSettings, Detail, DiscretisedExplorationSettings, ExactExplorationSettings,
    Method, MonteCarloSettings, Quantification, QuantifyError, ReadQuantificationError, Study,
    TargetProbability, QUANTIFICATION_FORMAT, QUANTIFICATION_VERSION,
};
use std::path::Path;

#[test]
fn imported_unit_propagates_to_quantification_provenance() {
    let mut model = parallel_pair(0.1, 0.1);
    model.components[0].attributes.push(Attribute {
        name: "x".into(),
        kind: AttrKind::Float,
        init: Value::Float(1.0),
    });
    model.fmu_units.push(FmuUnit {
        name: "dahlquist".into(),
        path: concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../raichu-fmi/tests/fixtures/reference-fmus/2.0/Dahlquist.fmu"
        )
        .into(),
        step: 0.1,
        inputs: vec![],
        outputs: vec![FmuBinding {
            attribute: AttrRef {
                component: "A".into(),
                attribute: "x".into(),
            },
            variable: "x".into(),
        }],
        parameters: vec![],
    });
    let study = Study::new("both_down", 0.2);
    let method = monte_carlo(2, 0.95);
    assert!(matches!(
        quantify(&model, &study, &method),
        Err(QuantifyError::Engine(EngineError::FmuPermission { unit })) if unit == "dahlquist"
    ));
    let result = quantify_with_fmu(&model, &study, &method, Path::new("."), true, false).unwrap();
    assert_eq!(result.provenance.fmu_units.len(), 1);
    let Detail::MonteCarlo(estimates) = &result.detail else {
        panic!("expected Monte-Carlo detail");
    };
    assert_eq!(result.provenance.fmu_units, estimates.fmu_units);
    assert_eq!(
        read_quantification(&result.to_json().unwrap())
            .unwrap()
            .provenance
            .fmu_units,
        result.provenance.fmu_units
    );
}

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

/// A non-repairable unit `name`: automaton `fail` (`ok` -> `nok`) under
/// `law`, the failure monitored.
fn unit(name: &str, law: Distrib) -> Component {
    let mut occ = transition("occ", "ok", &["nok"], law);
    occ.monitored = true;
    component(
        name,
        Automaton {
            name: "fail".into(),
            states: vec!["ok".into(), "nok".into()],
            init: "ok".into(),
            transitions: vec![occ],
        },
    )
}

fn down_target(name: &str, component: &str) -> Target {
    Target {
        name: name.into(),
        component: component.into(),
        automaton: "fail".into(),
        state: "nok".into(),
    }
}

fn down_indicator(component: &str) -> Indicator {
    Indicator {
        name: format!("{component}_down"),
        target: IndicatorTarget::State {
            component: component.into(),
            automaton: "fail".into(),
            state: "nok".into(),
        },
    }
}

/// Parallel pair A (rate a), B (rate b), non repairable; the feared event
/// `both_down` is B down while A is down, reached through a watcher that
/// enters `down` as soon as both are.
fn parallel_pair(a: f64, b: f64) -> Model {
    use raichu_expr::{BoolOp, Expr, StateRef};
    let down = |c: &str| Expr::StateActive {
        state: StateRef {
            component: c.into(),
            automaton: "fail".into(),
            state: "nok".into(),
        },
    };
    let mut to_down = transition("down", "ok", &["down"], Distrib::Inst { probs: vec![] });
    to_down.guard = Some(Expr::Bool {
        bool_op: BoolOp::And,
        args: vec![down("A"), down("B")],
    });
    let watcher = component(
        "sys",
        Automaton {
            name: "watch".into(),
            states: vec!["ok".into(), "down".into()],
            init: "ok".into(),
            transitions: vec![to_down],
        },
    );
    Model {
        programs: vec![],
        name: "parallel_pair".into(),
        components: vec![unit("A", exp(a)), unit("B", exp(b)), watcher],
        connections: vec![],
        indicators: vec![down_indicator("A"), down_indicator("B")],
        targets: vec![Target {
            name: "both_down".into(),
            component: "sys".into(),
            automaton: "watch".into(),
            state: "down".into(),
        }],
        evaluation_order: None,
        unbounded_rate: None,
        fmu_units: vec![],
    }
}

/// Two units A (rate a) and B (rate b), each its own feared event: the
/// first one down ends the trajectory.
fn competing_pair(a: f64, b: f64) -> Model {
    Model {
        programs: vec![],
        name: "competing_pair".into(),
        components: vec![unit("A", exp(a)), unit("B", exp(b))],
        connections: vec![],
        indicators: vec![],
        targets: vec![down_target("A_down", "A"), down_target("B_down", "B")],
        evaluation_order: None,
        unbounded_rate: None,
        fmu_units: vec![],
    }
}

/// `P(both down by t) = (1 - e^{-at}) (1 - e^{-bt})`.
fn both_down(a: f64, b: f64, t: f64) -> f64 {
    (-(-a * t).exp_m1()) * (-(-b * t).exp_m1())
}

/// `P(A down first, by t) = a / (a + b) (1 - e^{-(a+b)t})`.
fn first_down(a: f64, b: f64, t: f64) -> f64 {
    a / (a + b) * (-(-(a + b) * t).exp_m1())
}

const A: f64 = 0.1;
const B: f64 = 0.2;
const T: f64 = 5.0;

fn study(target: &str) -> Study {
    Study {
        target: target.into(),
        horizon: T,
        instants: Some(vec![1.0, 2.5, T]),
        seed: 20_260_927,
        threads: None,
    }
}

fn monte_carlo(nb_runs: u64, confidence: f64) -> Method {
    Method::MonteCarlo(MonteCarloSettings {
        confidence,
        ..MonteCarloSettings::new(nb_runs)
    })
}

fn exact() -> Method {
    Method::Exact(ExactExplorationSettings::default())
}

fn discretised() -> Method {
    Method::Discretised(DiscretisedExplorationSettings::default())
}

fn compile(model: &Model) -> CompiledModel {
    CompiledModel::compile(model).unwrap()
}

fn interval(q: &Quantification) -> (f64, f64, f64) {
    match &q.probability {
        TargetProbability::ConfidenceInterval {
            estimate,
            low,
            high,
            ..
        } => (*estimate, *low, *high),
        other => panic!("expected a confidence interval, got {other:?}"),
    }
}

fn bounds(q: &Quantification) -> (f64, f64, Option<f64>) {
    match &q.probability {
        TargetProbability::Bounds {
            lower,
            upper,
            error_estimate,
            ..
        } => (*lower, *upper, *error_estimate),
        other => panic!("expected bounds, got {other:?}"),
    }
}

fn assert_rel(actual: f64, expected: f64, what: &str) {
    let rel = ((actual - expected) / expected).abs();
    assert!(
        rel <= 1e-12,
        "{what}: {actual:e} vs closed form {expected:e} (relative {rel:e})"
    );
}

// ---- the three methods on one study ----------------------------------------

#[test]
fn the_three_methods_agree_with_the_closed_form_of_a_parallel_pair() {
    let model = parallel_pair(A, B);
    let closed = both_down(A, B, T);
    let study = study("both_down");

    let mc = quantify(&model, &study, &monte_carlo(4_000, 0.99)).unwrap();
    let (estimate, low, high) = interval(&mc);
    assert!(
        low <= closed && closed <= high,
        "Monte-Carlo interval [{low}, {high}] (estimate {estimate}) misses {closed}"
    );
    match &mc.probability {
        TargetProbability::ConfidenceInterval {
            reached,
            replicas,
            level,
            method,
            ..
        } => {
            assert_eq!(*replicas, 4_000);
            assert_eq!(*level, 0.99);
            assert_eq!(*method, IntervalMethod::Wilson);
            assert_eq!(estimate, *reached as f64 / 4_000.0);
        }
        _ => unreachable!(),
    }

    let ex = quantify(&model, &study, &exact()).unwrap();
    let (lower, upper, estimate_error) = bounds(&ex);
    assert_rel(lower, closed, "exact lower bound");
    assert_rel(upper, closed, "exact upper bound");
    assert_eq!(estimate_error, None);

    let disc = quantify(&model, &study, &discretised()).unwrap();
    let (d_lower, d_upper, d_error) = bounds(&disc);
    let d_error = d_error.expect("a refined discretised exploration states its error estimate");
    assert!(
        (d_lower - lower).abs() <= d_error && (d_upper - upper).abs() <= d_error,
        "discretised [{d_lower}, {d_upper}] vs exact [{lower}, {upper}], estimate {d_error}"
    );
    // The exact bounds lie inside the Monte-Carlo interval too.
    assert!(low <= lower && upper <= high);
}

#[test]
fn the_first_target_reached_is_counted_the_same_by_monte_carlo_and_exact_exploration() {
    let model = competing_pair(A, B);
    for (target, closed) in [
        ("A_down", first_down(A, B, T)),
        ("B_down", first_down(B, A, T)),
    ] {
        let study = study(target);
        let ex = quantify(&model, &study, &exact()).unwrap();
        let (lower, upper, _) = bounds(&ex);
        assert_rel(lower, closed, target);
        assert_rel(upper, closed, target);
        let mc = quantify(&model, &study, &monte_carlo(4_000, 0.99)).unwrap();
        let (estimate, low, high) = interval(&mc);
        assert!(
            low <= closed && closed <= high,
            "{target}: Monte-Carlo [{low}, {high}] (estimate {estimate}) misses {closed}"
        );
    }
    // The two first-reached probabilities of one campaign partition the
    // replicas that reached a target: the same seed, the same campaign.
    let a = quantify(&model, &study("A_down"), &monte_carlo(4_000, 0.99)).unwrap();
    let b = quantify(&model, &study("B_down"), &monte_carlo(4_000, 0.99)).unwrap();
    let reached = |q: &Quantification| match q.probability {
        TargetProbability::ConfidenceInterval { reached, .. } => reached,
        _ => unreachable!(),
    };
    assert!(reached(&a) + reached(&b) <= 4_000);
    assert_eq!(a.detail, b.detail);
}

// ---- the detailed result is the engine's own ---------------------------------

#[test]
fn the_monte_carlo_detail_is_the_stop_at_targets_campaign_byte_for_byte() {
    let model = parallel_pair(A, B);
    let study = study("both_down");
    let settings = MonteCarloSettings {
        quantiles: vec![0.25, 0.75],
        ..MonteCarloSettings::new(500)
    };
    let q = quantify(&model, &study, &Method::MonteCarlo(settings.clone())).unwrap();
    let config = settings.to_engine(&study);
    assert!(config.stop_at_targets);
    let direct = raichu_montecarlo::run(&compile(&model), &config).unwrap();
    let Detail::MonteCarlo(detail) = &q.detail else {
        panic!("expected a Monte-Carlo detail");
    };
    assert_eq!(detail.as_ref(), &direct);
    assert_eq!(
        serde_json::to_string(detail).unwrap(),
        serde_json::to_string(&direct).unwrap()
    );
    assert_eq!(detail.indicators.len(), 2);
    assert_eq!(detail.indicators[0].instants, vec![1.0, 2.5, T]);
}

#[test]
fn the_exploration_details_are_the_explorers_results_byte_for_byte() {
    let model = parallel_pair(A, B);
    let compiled = compile(&model);
    let study = study("both_down");

    let settings = ExactExplorationSettings {
        min_probability: Some(1e-9),
        ..ExactExplorationSettings::default()
    };
    let q = quantify(&model, &study, &Method::Exact(settings.clone())).unwrap();
    let direct = explore_exact(&compiled, &settings.to_engine(&study)).unwrap();
    let Detail::Exploration(detail) = &q.detail else {
        panic!("expected an exploration detail");
    };
    assert_eq!(
        serde_json::to_string(detail).unwrap(),
        serde_json::to_string(&direct).unwrap()
    );

    let settings = DiscretisedExplorationSettings {
        level: 4,
        ..DiscretisedExplorationSettings::default()
    };
    let q = quantify(&model, &study, &Method::Discretised(settings.clone())).unwrap();
    let direct = explore_discretised(&compiled, &settings.to_engine(&study)).unwrap();
    let Detail::Exploration(detail) = &q.detail else {
        panic!("expected an exploration detail");
    };
    assert_eq!(
        serde_json::to_string(detail).unwrap(),
        serde_json::to_string(&direct).unwrap()
    );
}

// ---- provenance ---------------------------------------------------------------

#[test]
fn a_seed_given_to_an_exploration_changes_nothing_and_is_not_recorded() {
    let model = parallel_pair(A, B);
    for method in [exact(), discretised()] {
        let one = quantify(&model, &study("both_down"), &method).unwrap();
        let other = Study {
            seed: 7,
            instants: Some(vec![T]),
            ..study("both_down")
        };
        let two = quantify(&model, &other, &method).unwrap();
        assert_eq!(one.to_json().unwrap(), two.to_json().unwrap());
        assert_eq!(one.provenance.seed, None);
        assert_eq!(one.provenance.instants, None);
        let value: serde_json::Value = serde_json::from_str(&one.to_json().unwrap()).unwrap();
        assert!(value["provenance"].get("seed").is_none());
        assert!(value["provenance"].get("instants").is_none());
    }
    let mc = quantify(&model, &study("both_down"), &monte_carlo(50, 0.95)).unwrap();
    assert_eq!(mc.provenance.seed, Some(20_260_927));
    assert_eq!(mc.provenance.instants, Some(vec![1.0, 2.5, T]));
}

#[test]
fn the_thread_count_does_not_change_the_envelope() {
    let model = parallel_pair(A, B);
    for method in [monte_carlo(300, 0.95), exact(), discretised()] {
        let one = Study {
            threads: Some(1),
            ..study("both_down")
        };
        let four = Study {
            threads: Some(4),
            ..study("both_down")
        };
        assert_eq!(
            quantify(&model, &one, &method).unwrap().to_json().unwrap(),
            quantify(&model, &four, &method).unwrap().to_json().unwrap(),
            "{}",
            method.name()
        );
    }
}

#[test]
fn the_provenance_names_the_model_its_hash_and_the_study() {
    let model = parallel_pair(A, B);
    let q = quantify(&model, &study("both_down"), &exact()).unwrap();
    assert_eq!(q.format, QUANTIFICATION_FORMAT);
    assert_eq!(q.version, QUANTIFICATION_VERSION);
    assert_eq!(q.provenance.engine_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(q.provenance.model, "parallel_pair");
    assert_eq!(q.provenance.model_hash, model_content_hash(&model).unwrap());
    assert_eq!(q.provenance.target, "both_down");
    assert_eq!(q.provenance.horizon, T);
    assert_eq!(q.method, exact());
}

#[test]
fn the_content_hash_tells_models_apart_and_ignores_the_document_layout() {
    let one = model_content_hash(&parallel_pair(A, B)).unwrap();
    let again = model_content_hash(&parallel_pair(A, B)).unwrap();
    let other = model_content_hash(&parallel_pair(A, 0.25)).unwrap();
    assert_eq!(one, again);
    assert_ne!(one, other);
    assert!(one.starts_with("sha256:"));
    assert_eq!(one.len(), 7 + 64);
    assert!(one[7..]
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    // Read back from a compact document, with its keys in another order:
    // the same model, the same hash.
    let pretty = parallel_pair(A, B).to_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&pretty).unwrap();
    let compact = serde_json::to_string(&value).unwrap();
    let reread = Model::from_json(&compact).unwrap();
    assert_eq!(model_content_hash(&reread).unwrap(), one);
}

// ---- refusals -------------------------------------------------------------------

#[test]
fn an_unknown_target_is_a_typed_error_naming_the_declared_ones() {
    let model = competing_pair(A, B);
    for method in [monte_carlo(10, 0.95), exact(), discretised()] {
        let err = quantify(&model, &study("C_down"), &method).unwrap_err();
        match &err {
            QuantifyError::UnknownTarget { target, declared } => {
                assert_eq!(target, "C_down");
                assert_eq!(declared, &["A_down", "B_down"]);
            }
            other => panic!("expected UnknownTarget, got {other:?}"),
        }
        assert!(err.to_string().contains("`A_down`, `B_down`"), "{err}");
    }
}

#[test]
fn an_exploration_outside_its_domain_surfaces_the_engine_refusal() {
    let mut model = parallel_pair(A, B);
    model.components[0] = unit(
        "A",
        Distrib::Weibull {
            shape: 2.0,
            scale: 10.0,
        },
    );
    let err = quantify(&model, &study("both_down"), &exact()).unwrap_err();
    assert!(
        matches!(
            err,
            QuantifyError::Engine(EngineError::LawOutsideExactDomain { .. })
        ),
        "{err:?}"
    );
    // The same study stays answerable by the two other methods.
    quantify(&model, &study("both_down"), &discretised()).unwrap();
    quantify(&model, &study("both_down"), &monte_carlo(20, 0.95)).unwrap();
}

#[test]
fn invalid_study_parameters_are_refused_before_anything_runs() {
    let model = parallel_pair(A, B);
    let cases = [
        (
            Study {
                horizon: -1.0,
                instants: None,
                ..study("both_down")
            },
            "horizon",
        ),
        (
            Study {
                instants: Some(vec![1.0, 6.0]),
                ..study("both_down")
            },
            "instants",
        ),
        (
            Study {
                instants: Some(vec![2.0, 1.0]),
                ..study("both_down")
            },
            "instants",
        ),
        (
            Study {
                threads: Some(0),
                ..study("both_down")
            },
            "threads",
        ),
    ];
    for (study, parameter) in cases {
        match quantify(&model, &study, &exact()).unwrap_err() {
            QuantifyError::InvalidStudy { parameter: p, .. } => assert_eq!(p, parameter),
            other => panic!("expected InvalidStudy({parameter}), got {other:?}"),
        }
    }
    match quantify(&model, &study("both_down"), &monte_carlo(0, 0.95)).unwrap_err() {
        QuantifyError::InvalidSettings { method, .. } => assert_eq!(method, "monte_carlo"),
        other => panic!("expected InvalidSettings, got {other:?}"),
    }
    assert!(matches!(
        quantify(&model, &study("both_down"), &monte_carlo(10, 1.5)).unwrap_err(),
        QuantifyError::Engine(EngineError::InvalidStudyParameter { .. })
    ));
}

#[test]
fn a_method_is_built_from_its_name_and_refuses_foreign_settings() {
    let method = Method::from_parts("discretised", &serde_json::json!({"level": 4})).unwrap();
    assert_eq!(
        method,
        Method::Discretised(DiscretisedExplorationSettings {
            level: 4,
            ..DiscretisedExplorationSettings::default()
        })
    );
    assert_eq!(
        Method::from_parts("exact", &serde_json::Value::Null).unwrap(),
        exact()
    );

    let err = Method::from_parts("bootstrap", &serde_json::Value::Null).unwrap_err();
    assert!(matches!(err, QuantifyError::UnknownMethod { .. }));
    assert!(
        err.to_string()
            .contains("`monte_carlo`, `exact`, `discretised`"),
        "{err}"
    );

    let err = Method::from_parts("exact", &serde_json::json!({"level": 4})).unwrap_err();
    match &err {
        QuantifyError::SettingNotApplicable {
            method,
            setting,
            applies_to,
        } => {
            assert_eq!(method, "exact");
            assert_eq!(setting, "level");
            assert_eq!(applies_to, &["discretised"]);
        }
        other => panic!("expected SettingNotApplicable, got {other:?}"),
    }
    let err = Method::from_parts(
        "monte_carlo",
        &serde_json::json!({"nb_runs": 10, "max_length": 3}),
    )
    .unwrap_err();
    assert!(err.to_string().contains("`exact`, `discretised`"), "{err}");
    let err = Method::from_parts("monte_carlo", &serde_json::json!({})).unwrap_err();
    assert!(
        matches!(err, QuantifyError::InvalidSettings { .. }),
        "{err:?}"
    );
}

// ---- the envelope reads back ----------------------------------------------------

#[test]
fn every_envelope_reads_back_identically() {
    let model = parallel_pair(A, B);
    for method in [monte_carlo(200, 0.95), exact(), discretised()] {
        let q = quantify(&model, &study("both_down"), &method).unwrap();
        let json = q.to_json().unwrap();
        let back = read_quantification(&json).unwrap();
        assert_eq!(back, q, "{}", method.name());
        assert_eq!(back.to_json().unwrap(), json);
    }
}

#[test]
fn the_reader_refuses_another_format_a_future_version_and_a_mixed_envelope() {
    let model = parallel_pair(A, B);
    let q = quantify(&model, &study("both_down"), &exact()).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&q.to_json().unwrap()).unwrap();

    let mut future = value.clone();
    future["version"] = serde_json::json!(QUANTIFICATION_VERSION + 1);
    assert_eq!(
        read_quantification(&future.to_string()).unwrap_err(),
        ReadQuantificationError::Version(Some(QUANTIFICATION_VERSION + 1))
    );

    let mut other = value.clone();
    other["format"] = serde_json::json!("raichu.exploration");
    assert_eq!(
        read_quantification(&other.to_string()).unwrap_err(),
        ReadQuantificationError::Format(Some("raichu.exploration".into()))
    );

    // An exact envelope claiming the discretised method.
    value["method"] = serde_json::to_value(discretised()).unwrap();
    assert!(matches!(
        read_quantification(&value.to_string()).unwrap_err(),
        ReadQuantificationError::Inconsistent(_)
    ));
}

#[test]
fn the_reader_refuses_what_its_parser_or_the_exploration_reader_refuses() {
    assert!(matches!(
        read_quantification("not json").unwrap_err(),
        ReadQuantificationError::Json(_)
    ));
    let model = parallel_pair(A, B);
    let q = quantify(&model, &study("both_down"), &exact()).unwrap();
    let value: serde_json::Value = serde_json::from_str(&q.to_json().unwrap()).unwrap();

    // Right format and version, a method that does not deserialize.
    let mut broken = value.clone();
    broken["method"] = serde_json::json!({"name": "exact", "settings": {"max_terms": "many"}});
    assert!(matches!(
        read_quantification(&broken.to_string()).unwrap_err(),
        ReadQuantificationError::Json(_)
    ));

    // A version the format never had.
    let mut zero = value.clone();
    zero["version"] = serde_json::json!(0);
    assert_eq!(
        read_quantification(&zero.to_string()).unwrap_err(),
        ReadQuantificationError::Version(Some(0))
    );

    // The envelope is consistent, the exploration inside it is not a
    // document its own reader accepts.
    let mut dangling = value.clone();
    dangling["detail"]["exploration"]["sequences"][0]["steps"][0] = serde_json::json!(999);
    assert!(matches!(
        read_quantification(&dangling.to_string()).unwrap_err(),
        ReadQuantificationError::Inconsistent(_)
    ));
}

#[test]
fn the_reader_refuses_a_probability_that_is_not_its_detail() {
    let model = parallel_pair(A, B);
    let ex = quantify(&model, &study("both_down"), &exact()).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&ex.to_json().unwrap()).unwrap();
    value["probability"]["upper"] = serde_json::json!(0.9);
    assert!(matches!(
        read_quantification(&value.to_string()).unwrap_err(),
        ReadQuantificationError::Inconsistent(_)
    ));

    let mc = quantify(&model, &study("both_down"), &monte_carlo(100, 0.95)).unwrap();
    let base: serde_json::Value = serde_json::from_str(&mc.to_json().unwrap()).unwrap();
    for (path, forged) in [
        ("reached", serde_json::json!(3)),
        ("replicas", serde_json::json!(99)),
        ("level", serde_json::json!(0.9)),
    ] {
        let mut value = base.clone();
        value["probability"][path] = forged;
        assert!(
            matches!(
                read_quantification(&value.to_string()).unwrap_err(),
                ReadQuantificationError::Inconsistent(_)
            ),
            "{path}"
        );
    }
    let mut value = base.clone();
    value["provenance"]["seed"] = serde_json::json!(1);
    assert!(matches!(
        read_quantification(&value.to_string()).unwrap_err(),
        ReadQuantificationError::Inconsistent(_)
    ));
}

#[test]
fn two_models_one_ulp_apart_hash_differently() {
    let rate = 0.1_f64;
    let next = f64::from_bits(rate.to_bits() + 1);
    assert_ne!(
        model_content_hash(&parallel_pair(rate, B)).unwrap(),
        model_content_hash(&parallel_pair(next, B)).unwrap()
    );
}

#[test]
fn a_negative_instant_and_a_non_finite_horizon_are_refused() {
    let model = parallel_pair(A, B);
    for (study, parameter) in [
        (
            Study {
                instants: Some(vec![-1.0, 1.0]),
                ..study("both_down")
            },
            "instants",
        ),
        (
            Study {
                horizon: f64::INFINITY,
                instants: None,
                ..study("both_down")
            },
            "horizon",
        ),
    ] {
        match quantify(&model, &study, &monte_carlo(10, 0.95)).unwrap_err() {
            QuantifyError::InvalidStudy { parameter: p, .. } => assert_eq!(p, parameter),
            other => panic!("expected InvalidStudy({parameter}), got {other:?}"),
        }
    }
}

// ---- the cross-entropy method ------------------------------------------------

fn cross_entropy(nb_runs: u64) -> Method {
    Method::CrossEntropy(CrossEntropySamplingSettings {
        pilot_runs: 500,
        ..CrossEntropySamplingSettings::new(nb_runs)
    })
}

fn weighted(q: &Quantification) -> (f64, f64, f64, f64) {
    match &q.probability {
        TargetProbability::WeightedEstimate {
            estimate,
            standard_error,
            low,
            high,
            ..
        } => (*estimate, *standard_error, *low, *high),
        other => panic!("expected a weighted estimate, got {other:?}"),
    }
}

/// A rare parallel pair: failure rates of 1e-3 over the study horizon.
const RARE: f64 = 1e-3;

#[test]
fn cross_entropy_agrees_with_the_closed_form_and_the_exact_bounds_on_a_rare_pair() {
    let model = parallel_pair(RARE, RARE);
    let closed = both_down(RARE, RARE, T);
    let study = study("both_down");

    let ce = quantify(&model, &study, &cross_entropy(4_000)).unwrap();
    let (estimate, se, low, high) = weighted(&ce);
    assert!(se > 0.0 && low < estimate && estimate < high);
    assert!(
        (estimate - closed).abs() <= 4.0 * se,
        "cross-entropy {estimate:e} is {:.1} standard errors from {closed:e}",
        (estimate - closed).abs() / se
    );
    let ex = quantify(&model, &study, &exact()).unwrap();
    let (lower, upper, _) = bounds(&ex);
    assert_rel(lower, closed, "exact lower bound");
    assert_rel(upper, closed, "exact upper bound");
    match &ce.detail {
        Detail::CrossEntropy(detail) => {
            assert_eq!(detail.replicas, 4_000);
            assert!(!detail.families.is_empty());
            assert!(
                detail.ends.is_empty(),
                "the envelope carries no per-replica ends"
            );
        }
        other => panic!("expected a cross-entropy detail, got {other:?}"),
    }
}

#[test]
fn cross_entropy_at_factor_one_without_fit_counts_what_monte_carlo_counts() {
    let model = parallel_pair(A, B);
    let study = study("both_down");
    let plain = Method::CrossEntropy(CrossEntropySamplingSettings {
        fit: false,
        initial_factor: 1.0,
        ..CrossEntropySamplingSettings::new(2_000)
    });
    let ce = quantify(&model, &study, &plain).unwrap();
    let mc = quantify(&model, &study, &monte_carlo(2_000, 0.95)).unwrap();
    let reached = |q: &Quantification| match q.probability {
        TargetProbability::ConfidenceInterval { reached, .. }
        | TargetProbability::WeightedEstimate { reached, .. } => reached,
        _ => unreachable!(),
    };
    assert_eq!(reached(&ce), reached(&mc));
    let (estimate, _, _, _) = weighted(&ce);
    assert_eq!(estimate, reached(&mc) as f64 / 2_000.0);
}

#[test]
fn cross_entropy_records_the_seed_and_not_the_instants() {
    let q = quantify(
        &parallel_pair(A, B),
        &study("both_down"),
        &cross_entropy(200),
    )
    .unwrap();
    assert_eq!(q.provenance.seed, Some(20_260_927));
    assert_eq!(q.provenance.instants, None);
    assert_eq!(q.method.name(), "cross_entropy");
}

#[test]
fn a_setting_of_another_method_is_refused_by_name_both_ways() {
    let err = Method::from_parts(
        "cross_entropy",
        &serde_json::json!({"nb_runs": 10, "quantiles": [0.5]}),
    )
    .unwrap_err();
    assert!(
        matches!(&err, QuantifyError::SettingNotApplicable { setting, applies_to, .. }
            if setting == "quantiles" && applies_to == &vec!["monte_carlo".to_owned()]),
        "{err:?}"
    );
    let err = Method::from_parts(
        "monte_carlo",
        &serde_json::json!({"nb_runs": 10, "pilot_runs": 5}),
    )
    .unwrap_err();
    assert!(
        matches!(&err, QuantifyError::SettingNotApplicable { setting, applies_to, .. }
            if setting == "pilot_runs" && applies_to == &vec!["cross_entropy".to_owned()]),
        "{err:?}"
    );
    let method = Method::from_parts("cross_entropy", &serde_json::json!({"nb_runs": 10})).unwrap();
    assert_eq!(method.name(), "cross_entropy");
}

#[test]
fn a_cross_entropy_campaign_that_never_hits_is_an_error() {
    let err = quantify(
        &parallel_pair(1e-9, 1e-9),
        &study("both_down"),
        &Method::CrossEntropy(CrossEntropySamplingSettings {
            fit: false,
            initial_factor: 1.0,
            ..CrossEntropySamplingSettings::new(100)
        }),
    )
    .unwrap_err();
    assert!(matches!(err, QuantifyError::NoHit(_)), "{err:?}");
}

#[test]
fn a_cross_entropy_envelope_reads_back_and_refuses_a_forged_estimate() {
    let q = quantify(
        &parallel_pair(RARE, RARE),
        &study("both_down"),
        &cross_entropy(1_000),
    )
    .unwrap();
    assert_eq!(q.version, QUANTIFICATION_VERSION);
    let json = q.to_json().unwrap();
    let back = read_quantification(&json).unwrap();
    assert_eq!(back, q);
    assert_eq!(back.to_json().unwrap(), json);

    let base: serde_json::Value = serde_json::from_str(&json).unwrap();
    let mut v2 = base.clone();
    v2["version"] = serde_json::json!(2);
    assert_eq!(read_quantification(&v2.to_string()).unwrap().version, 2);
    let mut older = base.clone();
    older["version"] = serde_json::json!(1);
    assert!(
        matches!(
            read_quantification(&older.to_string()).unwrap_err(),
            ReadQuantificationError::Inconsistent(_)
        ),
        "a version-1 envelope cannot carry a method version 2 introduced"
    );
    for (path, forged) in [
        ("estimate", serde_json::json!(0.5)),
        ("inconclusive", serde_json::json!(true)),
        ("reached", serde_json::json!(1)),
        ("replicas", serde_json::json!(999)),
        ("level", serde_json::json!(0.9)),
    ] {
        let mut value = base.clone();
        value["probability"][path] = forged;
        assert!(
            matches!(
                read_quantification(&value.to_string()).unwrap_err(),
                ReadQuantificationError::Inconsistent(_)
            ),
            "{path}"
        );
    }
}

#[test]
fn a_version_one_envelope_still_reads() {
    let q = quantify(
        &parallel_pair(A, B),
        &study("both_down"),
        &monte_carlo(100, 0.95),
    )
    .unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&q.to_json().unwrap()).unwrap();
    value["version"] = serde_json::json!(1);
    let back = read_quantification(&value.to_string()).unwrap();
    assert_eq!(back.version, 1);
    assert_eq!(QUANTIFICATION_VERSION, 3);
}
