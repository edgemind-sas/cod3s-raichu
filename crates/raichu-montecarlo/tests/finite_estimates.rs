//! A Monte-Carlo result document carries finite numbers only: JSON has no
//! spelling for a NaN or an infinity, so a result holding one is refused
//! by name rather than written with a `null` no reader takes back.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, EngineError, FlowConfig};
use raichu_expr::{Assignment, AttrRef, Expr, Value};
use raichu_model::Model;
use raichu_montecarlo::{run, McConfig, McEstimates, DEFAULT_CONFIDENCE};

/// A state entered at a date drawn uniformly on `[1, 3]`; on entry, the
/// attribute `x` becomes `numerator / denominator`.
fn entry_setting_x_to(numerator: f64, denominator: f64) -> Model {
    let mut model = Model::from_json(
        r#"{
  "name": "finite",
  "components": [{
    "name": "c",
    "attributes": [{"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}],
    "ports": [],
    "automata": [{
      "name": "aut", "states": ["off", "on"], "init": "off",
      "transitions": [
        {"name": "enter", "source": "off", "targets": ["on"],
         "distrib": "uniform", "low": 1.0, "high": 3.0}]}]}],
  "indicators": [
    {"name": "c_on", "target": "state", "component": "c", "automaton": "aut", "state": "on"},
    {"name": "x", "target": "attribute", "attr": {"component": "c", "attribute": "x"}}]
}"#,
    )
    .unwrap();
    let constant = |v: f64| {
        Box::new(Expr::Const {
            value: Value::Float(v),
        })
    };
    model.components[0].automata[0].transitions[0]
        .effects
        .push(Assignment {
            target: AttrRef {
                component: "c".into(),
                attribute: "x".into(),
            },
            value: Expr::Div {
                lhs: constant(numerator),
                rhs: constant(denominator),
            },
        });
    model
}

fn config() -> McConfig {
    McConfig {
        nb_runs: 200,
        seed: 3,
        t_max: 4.0,
        samples: vec![0.5, 2.0, 4.0],
        threads: None,
        quantiles: vec![0.5],
        confidence: DEFAULT_CONFIDENCE,
        ode: Default::default(),
        stop_at_targets: false,
        flow: FlowConfig::default(),
    }
}

fn estimate(model: &Model) -> McEstimates {
    run(&CompiledModel::compile(model).unwrap(), &config()).unwrap()
}

#[test]
fn a_finite_result_passes() {
    estimate(&entry_setting_x_to(1.0, 4.0))
        .check_finite()
        .expect("the state indicator and a finite x are finite");
}

#[test]
fn the_first_non_finite_value_is_named_by_indicator_measure_and_instant() {
    for numerator in [0.0, 1.0, -1.0] {
        let e = estimate(&entry_setting_x_to(numerator, 0.0))
            .check_finite()
            .unwrap_err();
        // Before t = 1 no replica has entered; at t = 2 some have.
        assert_eq!(
            (e.indicator.as_str(), e.measure.as_str(), e.instant),
            ("x", "mean", 2.0),
            "{numerator}/0"
        );
        assert_eq!(e.value.to_bits(), (numerator / 0.0_f64).to_bits());
    }
}

#[test]
fn a_measure_is_named_by_its_member_in_the_document() {
    let mut estimates = estimate(&entry_setting_x_to(1.0, 4.0));
    // Plant one non-finite value at a time in a copy of a finite
    // indicator, where only that measure carries it.
    let finite = estimates.indicators[1].clone();
    for (plant, measure) in [
        (
            (|i: &mut raichu_montecarlo::IndicatorEstimate| i.quantiles[0].values[1] = f64::NAN)
                as fn(&mut _),
            "quantiles[q=0.5]",
        ),
        (
            |i| i.sojourn_extremes.max[2] = f64::INFINITY,
            "sojourn_extremes.max",
        ),
        (|i| i.ci.low[1] = f64::NEG_INFINITY, "ci.low"),
        (
            |i| i.sojourn_quantiles[0].values[0] = f64::NAN,
            "sojourn_quantiles[q=0.5]",
        ),
    ] {
        let mut indicator = finite.clone();
        indicator.name = "planted".into();
        plant(&mut indicator);
        estimates.indicators = vec![finite.clone(), indicator];
        let e = estimates.check_finite().unwrap_err();
        assert_eq!(
            (e.indicator.as_str(), e.measure.as_str()),
            ("planted", measure)
        );
        assert!(e.to_string().contains(&format!("`{measure}`")), "{e}");
    }
}

/// A campaign input that would put a non-finite number in the document
/// is refused before any replica, naming the input, so the refusal of a
/// non-finite estimate only ever points at the model.
#[test]
fn an_input_that_would_write_a_non_finite_number_is_refused_up_front() {
    let model = CompiledModel::compile(&entry_setting_x_to(1.0, 4.0)).unwrap();
    for (change, parameter) in [
        ((|c: &mut McConfig| c.nb_runs = 0) as fn(&mut _), "nb_runs"),
        (|c| c.samples = vec![1.0, f64::NAN], "samples"),
        (|c| c.samples = vec![f64::INFINITY], "samples"),
        (|c| c.samples = vec![-1.0], "samples"),
        (|c| c.quantiles = vec![f64::NAN], "quantiles"),
        (|c| c.quantiles = vec![1.5], "quantiles"),
        (|c| c.quantiles = vec![-0.1], "quantiles"),
    ] {
        let mut config = config();
        change(&mut config);
        match run(&model, &config) {
            Err(EngineError::InvalidStudyParameter { parameter: p, .. }) => {
                assert_eq!(p, parameter)
            }
            other => panic!("expected InvalidStudyParameter({parameter}), got {other:?}"),
        }
    }
    // The bounds of the quantile orders stay accepted: nearest rank reads
    // the smallest and the largest value there.
    let mut config = config();
    config.quantiles = vec![0.0, 1.0];
    run(&model, &config).unwrap().check_finite().unwrap();
}
