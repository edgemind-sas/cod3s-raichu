//! A threshold indicator answers on the **condition**, not on the value.
//!
//! An observation carrying a threshold (`level > 0`) and the same
//! observation without one are two different quantities, and the
//! difference shows on the `sojourn-time` measure: the sojourn of a raw
//! attribute is the time-integral of its value (an area, in
//! `unit x hours`), the sojourn of a threshold is the time the condition
//! held (a duration, in hours, bounded by the horizon). Reading the
//! second off the first is how a study asking "how long was there
//! hydrogen in the tank" over a 60 h horizon was answered **1199.10 h**:
//! 19.985 units of hydrogen integrated over 60 hours.
//!
//! Every expected figure below is a **closed form**, never a recorded
//! run: the tank holds a declared constant for a declared duration, so
//! the area and the duration are both products of two numbers written in
//! the test. A recording would pass just as happily on the wrong
//! quantity, which is exactly what happened for as long as the threshold
//! was dropped in silence.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, FlowConfig};
use raichu_expr::{AttrRef, CmpOp, Expr, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Distrib, Equation, EquationKind, Indicator,
    IndicatorTarget, Model, ModelError, Transition,
};
use raichu_montecarlo::{run, McConfig};

/// What the tank holds while it holds anything. The H2 showcase's own
/// figure, so the product below is the very number that was reported as
/// a duration.
const CONTENT: f64 = 19.985;

/// When the tank empties, and therefore the exact duration the threshold
/// `content > 0` holds.
const EMPTIED_AT: f64 = 10.0;

/// The campaign's horizon, the showcase's own.
const HORIZON: f64 = 60.0;

/// A tank holding [`CONTENT`] until [`EMPTIED_AT`], then nothing.
///
/// Deterministic on purpose: one replica is the whole distribution, so
/// the estimates below are the trajectory's own figures and no tolerance
/// is a Monte-Carlo tolerance.
fn tank(indicators: Vec<Indicator>) -> Model {
    Model {
        name: "tank".into(),
        components: vec![Component {
            name: "T".into(),
            attributes: vec![Attribute {
                name: "content".into(),
                kind: AttrKind::Float,
                init: Value::Float(CONTENT),
            }],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "supply".into(),
                states: vec!["held".into(), "gone".into()],
                init: "held".into(),
                transitions: vec![Transition {
                    name: "empty".into(),
                    source: "held".into(),
                    guard: None,
                    targets: vec!["gone".into()],
                    on_interruption: Default::default(),
                    monitored: false,
                    cycle_group: None,
                    distrib: Distrib::Delay { time: EMPTIED_AT },
                }],
            }],
            allocations: vec![],
            equations: vec![Equation {
                target: "content".into(),
                kind: EquationKind::Explicit,
                expr: Expr::If {
                    cond: Box::new(Expr::StateActive {
                        state: raichu_expr::StateRef {
                            component: "T".into(),
                            automaton: "supply".into(),
                            state: "held".into(),
                        },
                    }),
                    then: Box::new(Expr::Const {
                        value: Value::Float(CONTENT),
                    }),
                    otherwise: Box::new(Expr::Const {
                        value: Value::Float(0.0),
                    }),
                },
            }],
            sensitive_functions: vec![],
        }],
        indicators,
        connections: vec![],
        targets: vec![],
        evaluation_order: None,
        unbounded_rate: None,
    }
}

fn raw(name: &str) -> Indicator {
    Indicator {
        name: name.into(),
        target: IndicatorTarget::Attribute {
            attr: AttrRef {
                component: "T".into(),
                attribute: "content".into(),
            },
        },
    }
}

fn threshold(name: &str, cmp: CmpOp, value: Value) -> Indicator {
    Indicator {
        name: name.into(),
        target: IndicatorTarget::Predicate {
            attr: AttrRef {
                component: "T".into(),
                attribute: "content".into(),
            },
            cmp,
            value,
        },
    }
}

/// The campaign the showcase runs: sample every 5 h up to the horizon.
fn campaign(model: &Model) -> raichu_montecarlo::McEstimates {
    let compiled = CompiledModel::compile(model).unwrap();
    let samples: Vec<f64> = (0..=12).map(|k| f64::from(k) * 5.0).collect();
    run(
        &compiled,
        &McConfig {
            nb_runs: 1,
            seed: 42,
            t_max: HORIZON,
            samples,
            threads: None,
            quantiles: vec![],
            ode: Default::default(),
            stop_at_targets: false,
            flow: FlowConfig::default(),
        },
    )
    .unwrap()
}

#[test]
fn the_sojourn_of_a_threshold_is_a_duration_and_the_one_of_a_value_is_an_area() {
    let estimates = campaign(&tank(vec![
        raw("content"),
        threshold("has_content", CmpOp::Gt, Value::Float(0.0)),
    ]));
    let at_horizon = estimates.indicators[0].instants.len() - 1;
    assert_eq!(estimates.indicators[0].instants[at_horizon], HORIZON);

    // The area: the tank held 19.985 for 10 h and nothing afterwards.
    let area = estimates.indicators[0].sojourn_mean[at_horizon];
    assert!(
        (area - CONTENT * EMPTIED_AT).abs() < 1e-9,
        "the raw attribute's sojourn is the time-integral of the value: \
         expected {} x {} = {}, got {area}",
        CONTENT,
        EMPTIED_AT,
        CONTENT * EMPTIED_AT
    );

    // The duration: the condition held for exactly the ten hours the
    // tank was not empty. This is the figure a `sojourn-time` measure on
    // a threshold indicator asks for, and it is not the one above.
    let duration = estimates.indicators[1].sojourn_mean[at_horizon];
    assert!(
        (duration - EMPTIED_AT).abs() < 1e-9,
        "the threshold's sojourn is the time the condition held: expected \
         {EMPTIED_AT}, got {duration}"
    );
}

#[test]
fn a_threshold_sojourn_never_exceeds_the_horizon_and_the_raw_value_may() {
    let estimates = campaign(&tank(vec![
        raw("content"),
        threshold("has_content", CmpOp::Gt, Value::Float(0.0)),
    ]));
    // The impossibility that makes the defect recognisable without an
    // oracle: a time spent under a condition cannot outlast the run.
    for (instant, sojourn) in estimates.indicators[1]
        .instants
        .iter()
        .zip(&estimates.indicators[1].sojourn_mean)
    {
        assert!(
            *sojourn <= *instant + 1e-9,
            "a sojourn of {sojourn} at instant {instant} is more time than \
             the run has had"
        );
    }
    // And the control: on the very same attribute, the raw sojourn DOES
    // outlast it, which is what made 1199.10 h plausible enough to be
    // written into a result file.
    let at_horizon = estimates.indicators[0].instants.len() - 1;
    assert!(estimates.indicators[0].sojourn_mean[at_horizon] > HORIZON);
}

#[test]
fn the_mean_of_a_threshold_is_the_probability_the_condition_holds() {
    let estimates = campaign(&tank(vec![threshold(
        "has_content",
        CmpOp::Gt,
        Value::Float(0.0),
    )]));
    // Deterministic trajectory: 1 while held, 0 once gone, and the
    // sample at the emptying instant is post-event.
    let expected: Vec<f64> = estimates.indicators[0]
        .instants
        .iter()
        .map(|t| if *t < EMPTIED_AT { 1.0 } else { 0.0 })
        .collect();
    assert_eq!(estimates.indicators[0].mean, expected);
}

#[test]
fn the_threshold_is_the_one_the_document_wrote() {
    // `>= 19.985` holds while the tank is full, `> 19.985` never does,
    // and `== 0.0` is the complement of the first: three readings of one
    // attribute, three different durations, none of them the area.
    let estimates = campaign(&tank(vec![
        threshold("at_least_full", CmpOp::Ge, Value::Float(CONTENT)),
        threshold("above_full", CmpOp::Gt, Value::Float(CONTENT)),
        threshold("empty", CmpOp::Eq, Value::Float(0.0)),
    ]));
    let at_horizon = estimates.indicators[0].instants.len() - 1;
    let sojourn = |idx: usize| estimates.indicators[idx].sojourn_mean[at_horizon];
    assert!((sojourn(0) - EMPTIED_AT).abs() < 1e-9, "got {}", sojourn(0));
    assert!(sojourn(1).abs() < 1e-9, "got {}", sojourn(1));
    assert!(
        (sojourn(2) - (HORIZON - EMPTIED_AT)).abs() < 1e-9,
        "got {}",
        sojourn(2)
    );
}

#[test]
fn a_threshold_counts_the_times_the_condition_became_true() {
    let estimates = campaign(&tank(vec![threshold(
        "empty",
        CmpOp::Eq,
        Value::Float(0.0),
    )]));
    // One rising edge, at the emptying: `nb-occurrences` on a threshold
    // counts entries into the condition, which is the RAMS measure a
    // study writes beside the sojourn.
    let at_horizon = estimates.indicators[0].instants.len() - 1;
    assert_eq!(estimates.indicators[0].nb_occurrences_mean[at_horizon], 1.0);
    assert_eq!(estimates.indicators[0].nb_occurrences_mean[0], 0.0);
}

#[test]
fn a_threshold_ordering_a_boolean_is_refused_at_build_naming_the_indicator() {
    let mut model = tank(vec![]);
    model.components[0].attributes.push(Attribute {
        name: "online".into(),
        kind: AttrKind::Bool,
        init: Value::Bool(true),
    });
    model.indicators = vec![Indicator {
        name: "half_online".into(),
        target: IndicatorTarget::Predicate {
            attr: AttrRef {
                component: "T".into(),
                attribute: "online".into(),
            },
            cmp: CmpOp::Gt,
            value: Value::Bool(false),
        },
    }];
    let error = model.validate().unwrap_err();
    assert!(
        matches!(error, ModelError::IndicatorPredicateKind { ref indicator, .. }
            if indicator == "half_online"),
        "got {error}"
    );
    assert!(error.to_string().contains("never ordered"), "got {error}");
}

#[test]
fn a_threshold_comparing_a_number_to_a_boolean_is_refused_at_build() {
    let mut model = tank(vec![]);
    model.indicators = vec![threshold("nonsense", CmpOp::Eq, Value::Bool(true))];
    let error = model.validate().unwrap_err();
    assert!(
        matches!(error, ModelError::IndicatorPredicateKind { .. }),
        "got {error}"
    );
    assert!(error.to_string().contains("no comparison"), "got {error}");
}

#[test]
fn a_threshold_on_an_attribute_no_component_holds_is_refused_by_name() {
    let mut model = tank(vec![]);
    model.indicators = vec![Indicator {
        name: "ghost".into(),
        target: IndicatorTarget::Predicate {
            attr: AttrRef {
                component: "T".into(),
                attribute: "no_such_attribute".into(),
            },
            cmp: CmpOp::Gt,
            value: Value::Float(0.0),
        },
    }];
    let error = model.validate().unwrap_err();
    assert!(
        matches!(error, ModelError::IndicatorUnresolved { ref indicator, .. }
            if indicator == "ghost"),
        "got {error}"
    );
}

#[test]
fn a_threshold_survives_the_round_trip_through_the_document() {
    let model = tank(vec![threshold("has_content", CmpOp::Gt, Value::Float(0.0))]);
    let written = serde_json::to_string(&model).unwrap();
    // The serialized spelling is the contract the muscadet seam writes
    // against; pinned here so a rename of the variant is a failing test
    // rather than a document nobody can load.
    assert!(written.contains(r#""target":"predicate""#), "got {written}");
    let back: Model = serde_json::from_str(&written).unwrap();
    assert_eq!(back, model);
}

/// A tank draining at one unit per hour from `initial`, with no discrete
/// event anywhere: the threshold `level > 0` therefore flips in the
/// middle of a continuous segment, where nothing else records anything.
///
/// Closed form: the condition holds for exactly `initial` hours.
fn draining(initial: f64) -> Model {
    Model {
        name: "draining".into(),
        components: vec![Component {
            name: "T".into(),
            attributes: vec![Attribute {
                name: "level".into(),
                kind: AttrKind::Float,
                init: Value::Float(initial),
            }],
            ports: vec![],
            interfaces: vec![],
            automata: vec![],
            allocations: vec![],
            equations: vec![Equation {
                target: "level".into(),
                kind: EquationKind::Ode,
                expr: Expr::Const {
                    value: Value::Float(-1.0),
                },
            }],
            sensitive_functions: vec![],
        }],
        indicators: vec![
            threshold_on("level", "positive", CmpOp::Gt, Value::Float(0.0)),
            Indicator {
                name: "level".into(),
                target: IndicatorTarget::Attribute {
                    attr: AttrRef {
                        component: "T".into(),
                        attribute: "level".into(),
                    },
                },
            },
        ],
        connections: vec![],
        targets: vec![],
        evaluation_order: None,
        unbounded_rate: None,
    }
}

fn threshold_on(attribute: &str, name: &str, cmp: CmpOp, value: Value) -> Indicator {
    Indicator {
        name: name.into(),
        target: IndicatorTarget::Predicate {
            attr: AttrRef {
                component: "T".into(),
                attribute: attribute.into(),
            },
            cmp,
            value,
        },
    }
}

/// The same campaign, on a schedule of the caller's choosing.
fn campaign_on(model: &Model, samples: Vec<f64>) -> raichu_montecarlo::McEstimates {
    let compiled = CompiledModel::compile(model).unwrap();
    run(
        &compiled,
        &McConfig {
            nb_runs: 1,
            seed: 42,
            t_max: HORIZON,
            samples,
            threads: None,
            quantiles: vec![],
            ode: Default::default(),
            stop_at_targets: false,
            flow: FlowConfig::default(),
        },
    )
    .unwrap()
}

#[test]
fn a_threshold_flipping_inside_a_continuous_segment_stops_accumulating() {
    // Twenty hours of hydrogen, sixty hours of run, and NOTHING discrete
    // in between: no transition fires, no active set moves. This is the
    // case where the change-point series has nothing to record the flip
    // and the sojourn used to come back as the whole elapsed time.
    let estimates = campaign_on(
        &draining(20.0),
        (0..=12).map(|k| f64::from(k) * 5.0).collect(),
    );
    let sojourn = &estimates.indicators[0].sojourn_mean;
    let instants = &estimates.indicators[0].instants;
    for (instant, sojourn) in instants.iter().zip(sojourn) {
        let expected = instant.min(20.0);
        assert!(
            (sojourn - expected).abs() < 1e-9,
            "the condition held for 20 h of the run: expected {expected} at \
             instant {instant}, got {sojourn}"
        );
    }
}

#[test]
fn a_threshold_crossed_between_two_samples_is_located_on_the_schedule() {
    // Where the honesty of the measure stops, pinned rather than left to
    // be discovered. The flip is detected at the first sample that sees
    // it, never bisected the way a watched transition's boundary is, so
    // the duration is over-reported by at most one sample interval --
    // and refining the schedule makes it converge, which is the property
    // that distinguishes a RESOLUTION from the freeze it replaced.
    let coarse = campaign_on(
        &draining(22.0),
        (0..=12).map(|k| f64::from(k) * 5.0).collect(),
    );
    let fine = campaign_on(&draining(22.0), (0..=60).map(f64::from).collect());
    let at_end = |e: &raichu_montecarlo::McEstimates| *e.indicators[0].sojourn_mean.last().unwrap();
    // True duration 22 h; the coarse grid can only see the flip at 25.
    assert!(
        (at_end(&coarse) - 25.0).abs() < 1e-9,
        "got {}",
        at_end(&coarse)
    );
    assert!((at_end(&fine) - 22.0).abs() < 1e-9, "got {}", at_end(&fine));
    // Never under-reported, and never beyond one interval.
    assert!(at_end(&coarse) >= 22.0 && at_end(&coarse) <= 22.0 + 5.0);
}

#[test]
fn the_sojourn_of_a_free_valued_attribute_is_left_exactly_as_it_was() {
    // The control on the asymmetry above. A threshold gains change
    // points from the samples; a free-valued attribute does not, and
    // this pins the quantity every recorded result was produced with:
    // the time-integral of a series that only moves at discrete events,
    // which on a purely continuous trajectory is the INITIAL value times
    // the elapsed time (20 x t), not the true integral (20t - t^2/2).
    //
    // That is a defect of the `sojourn` measure on free-valued
    // attributes, and it is not this variant's: it predates it, it is
    // measured here, and closing it means locating the trajectory rather
    // than sampling it.
    let estimates = campaign_on(
        &draining(20.0),
        (0..=12).map(|k| f64::from(k) * 5.0).collect(),
    );
    let raw = &estimates.indicators[1];
    for (instant, sojourn) in raw.instants.iter().zip(&raw.sojourn_mean) {
        assert!(
            (sojourn - 20.0 * instant).abs() < 1e-9,
            "expected the initial value held over the elapsed time \
             ({}), got {sojourn} at {instant}",
            20.0 * instant
        );
    }
}
