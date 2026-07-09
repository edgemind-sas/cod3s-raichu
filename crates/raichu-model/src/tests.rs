//! Unit and property tests for the native model and its validation gate.

#![allow(clippy::unwrap_used)]

use super::*;
use proptest::prelude::*;
use raichu_expr::{AggOp, BoolOp, CmpOp, StateRef};

/// A small but complete valid model: a source exporting a boolean flow to
/// a target, a two-state failure automaton with delay transitions, an
/// instantaneous branching, one sensitive function and two indicators.
fn sample_model() -> Model {
    Model {
        name: "delay_001".into(),
        components: vec![
            Component {
                name: "source".into(),
                attributes: vec![Attribute {
                    name: "flow_out".into(),
                    kind: AttrKind::Bool,
                    init: Value::Bool(true),
                }],
                ports: vec![Port {
                    name: "out".into(),
                    dir: PortDir::Out,
                    attr: Some("flow_out".into()),
                }],
                interfaces: vec![Interface {
                    name: "outputs".into(),
                    ports: vec!["out".into()],
                }],
                automata: vec![Automaton {
                    name: "failure".into(),
                    states: vec!["ok".into(), "nok".into(), "limbo".into()],
                    init: "ok".into(),
                    transitions: vec![
                        Transition {
                            name: "fail".into(),
                            source: "ok".into(),
                            guard: None,
                            targets: vec!["nok".into()],
                            on_interruption: Default::default(),
                            distrib: Distrib::Delay { time: 5.0 },
                        },
                        Transition {
                            name: "repair".into(),
                            source: "nok".into(),
                            guard: None,
                            targets: vec!["ok".into()],
                            on_interruption: Default::default(),
                            distrib: Distrib::Delay { time: 10.0 },
                        },
                        Transition {
                            name: "branch".into(),
                            source: "limbo".into(),
                            guard: None,
                            targets: vec!["ok".into(), "nok".into()],
                            on_interruption: Default::default(),
                            distrib: Distrib::Inst { probs: vec![1.0] },
                        },
                    ],
                }],
                equations: vec![],
                sensitive_functions: vec![SensitiveFunction {
                    // The muscadet pattern: the exported flow follows the
                    // failure automaton (fault ⇒ no flow).
                    name: "update_flow_out".into(),
                    effects: vec![Assignment {
                        target: AttrRef {
                            component: "source".into(),
                            attribute: "flow_out".into(),
                        },
                        value: Expr::StateActive {
                            state: StateRef {
                                component: "source".into(),
                                automaton: "failure".into(),
                                state: "ok".into(),
                            },
                        },
                    }],
                }],
            },
            Component {
                name: "target".into(),
                attributes: vec![Attribute {
                    name: "fed".into(),
                    kind: AttrKind::Bool,
                    init: Value::Bool(false),
                }],
                ports: vec![Port {
                    name: "input".into(),
                    dir: PortDir::In,
                    attr: None,
                }],
                interfaces: vec![],
                automata: vec![],
                equations: vec![],
                sensitive_functions: vec![SensitiveFunction {
                    name: "update_fed".into(),
                    effects: vec![Assignment {
                        target: AttrRef {
                            component: "target".into(),
                            attribute: "fed".into(),
                        },
                        value: Expr::PortAgg {
                            port: PortRef {
                                component: "target".into(),
                                port: "input".into(),
                            },
                            agg: AggOp::Any,
                        },
                    }],
                }],
            },
        ],
        connections: vec![Connection {
            from: PortRef {
                component: "source".into(),
                port: "out".into(),
            },
            to: PortRef {
                component: "target".into(),
                port: "input".into(),
            },
        }],
        indicators: vec![
            Indicator {
                name: "target_fed".into(),
                target: IndicatorTarget::Attribute {
                    attr: AttrRef {
                        component: "target".into(),
                        attribute: "fed".into(),
                    },
                },
            },
            Indicator {
                name: "source_nok".into(),
                target: IndicatorTarget::State {
                    component: "source".into(),
                    automaton: "failure".into(),
                    state: "nok".into(),
                },
            },
        ],
    }
}

#[test]
fn sample_model_is_valid() {
    sample_model().validate().unwrap();
}

#[test]
fn json_round_trip_is_lossless() {
    let model = sample_model();
    let json = model.to_json().unwrap();
    let back = Model::from_json(&json).unwrap();
    assert_eq!(model, back);
    back.validate().unwrap();
}

#[test]
fn rejects_duplicate_component() {
    let mut model = sample_model();
    model.components.push(model.components[0].clone());
    assert!(matches!(
        model.validate(),
        Err(ModelError::DuplicateComponent { name }) if name == "source"
    ));
}

#[test]
fn rejects_init_kind_mismatch() {
    let mut model = sample_model();
    model.components[0].attributes[0].init = Value::Int(1);
    assert!(matches!(
        model.validate(),
        Err(ModelError::InitKindMismatch { attribute, .. }) if attribute == "flow_out"
    ));
}

#[test]
fn rejects_out_port_without_variable() {
    let mut model = sample_model();
    model.components[0].ports[0].attr = None;
    assert!(matches!(
        model.validate(),
        Err(ModelError::OutPortWithoutVariable { .. })
    ));
}

#[test]
fn rejects_in_port_with_variable() {
    let mut model = sample_model();
    model.components[1].ports[0].attr = Some("fed".into());
    assert!(matches!(
        model.validate(),
        Err(ModelError::InPortWithVariable { .. })
    ));
}

#[test]
fn rejects_reversed_connection() {
    let mut model = sample_model();
    let connection = model.connections[0].clone();
    model.connections[0] = Connection {
        from: connection.to,
        to: connection.from,
    };
    assert!(matches!(
        model.validate(),
        Err(ModelError::ConnectionDirectionMismatch { .. })
    ));
}

#[test]
fn rejects_unknown_connection_endpoint() {
    let mut model = sample_model();
    model.connections[0].to.port = "nope".into();
    assert!(matches!(
        model.validate(),
        Err(ModelError::ConnectionUnknownPort { side: "to", .. })
    ));
}

#[test]
fn rejects_unknown_init_state() {
    let mut model = sample_model();
    model.components[0].automata[0].init = "absent".into();
    assert!(matches!(
        model.validate(),
        Err(ModelError::UnknownInitState { state, .. }) if state == "absent"
    ));
}

#[test]
fn rejects_negative_delay() {
    let mut model = sample_model();
    model.components[0].automata[0].transitions[0].distrib = Distrib::Delay { time: -1.0 };
    assert!(matches!(
        model.validate(),
        Err(ModelError::InvalidDelay { time, .. }) if time == -1.0
    ));
}

#[test]
fn rejects_delay_with_two_targets() {
    let mut model = sample_model();
    model.components[0].automata[0].transitions[0]
        .targets
        .push("limbo".into());
    assert!(matches!(
        model.validate(),
        Err(ModelError::DelayTargetCount { targets: 2, .. })
    ));
}

#[test]
fn rejects_inst_arity_mismatch() {
    let mut model = sample_model();
    model.components[0].automata[0].transitions[2].distrib = Distrib::Inst {
        probs: vec![0.5, 0.4],
    };
    assert!(matches!(
        model.validate(),
        Err(ModelError::InstArityMismatch {
            targets: 2,
            probs: 2,
            ..
        })
    ));
}

#[test]
fn rejects_guard_on_unknown_variable() {
    let mut model = sample_model();
    model.components[0].automata[0].transitions[0].guard = Some(Expr::Cmp {
        cmp: CmpOp::Eq,
        lhs: Box::new(Expr::attr("ghost", "var")),
        rhs: Box::new(Expr::bool(true)),
    });
    assert!(matches!(
        model.validate(),
        Err(ModelError::ExprUnknownVariable { component, .. }) if component == "ghost"
    ));
}

#[test]
fn rejects_port_agg_over_out_port() {
    let mut model = sample_model();
    model.components[1].sensitive_functions[0].effects[0].value = Expr::PortAgg {
        port: PortRef {
            component: "source".into(),
            port: "out".into(),
        },
        agg: AggOp::Any,
    };
    assert!(matches!(
        model.validate(),
        Err(ModelError::ExprBadPortAgg { port, .. }) if port == "out"
    ));
}

#[test]
fn rejects_not_with_two_arguments() {
    let mut model = sample_model();
    model.components[0].automata[0].transitions[0].guard = Some(Expr::Bool {
        bool_op: BoolOp::Not,
        args: vec![Expr::bool(true), Expr::bool(false)],
    });
    assert!(matches!(
        model.validate(),
        Err(ModelError::NotArity { args: 2, .. })
    ));
}

#[test]
fn rejects_state_active_on_unknown_state() {
    let mut model = sample_model();
    model.components[0].sensitive_functions[0].effects[0].value = Expr::StateActive {
        state: StateRef {
            component: "source".into(),
            automaton: "failure".into(),
            state: "absent".into(),
        },
    };
    assert!(matches!(
        model.validate(),
        Err(ModelError::ExprUnknownState { state, .. }) if state == "absent"
    ));
}

#[test]
fn rejects_unresolved_indicator() {
    let mut model = sample_model();
    model.indicators[1] = Indicator {
        name: "bad".into(),
        target: IndicatorTarget::State {
            component: "source".into(),
            automaton: "failure".into(),
            state: "absent".into(),
        },
    };
    assert!(matches!(
        model.validate(),
        Err(ModelError::IndicatorUnresolved { indicator, .. }) if indicator == "bad"
    ));
}

#[test]
fn long_indicator_names_are_fine() {
    // Indicator names have no length limit.
    let mut model = sample_model();
    model.indicators[0].name = "x".repeat(10_000);
    model.validate().unwrap();
}

proptest! {
    /// Any probability vector with entries in [0,1] and sum ≤ 1 is
    /// accepted for a matching number of targets (complement rule).
    #[test]
    fn inst_valid_probs_accepted(probs in proptest::collection::vec(0.0f64..=1.0, 0..6)) {
        let sum: f64 = probs.iter().sum();
        prop_assume!(sum <= 1.0);
        let mut model = sample_model();
        let n_targets = probs.len() + 1;
        let states: Vec<String> = (0..n_targets).map(|i| format!("s{i}")).collect();
        let automaton = &mut model.components[0].automata[0];
        automaton.states.extend(states.clone());
        automaton.transitions[2].targets = states;
        automaton.transitions[2].distrib = Distrib::Inst { probs };
        prop_assert!(model.validate().is_ok());
    }

    /// Probability sums beyond 1 are always rejected.
    #[test]
    fn inst_excess_sum_rejected(extra in 0.001f64..10.0) {
        let mut model = sample_model();
        model.components[0].automata[0].transitions[2].distrib = Distrib::Inst {
            probs: vec![(1.0 + extra).min(1.0), (extra / 2.0).min(1.0)],
        };
        // Adjust targets to match arity so the *probability* check fires.
        model.components[0].automata[0].transitions[2].targets =
            vec!["ok".into(), "nok".into(), "limbo".into()];
        let sum: f64 = match &model.components[0].automata[0].transitions[2].distrib {
            Distrib::Inst { probs } => probs.iter().sum(),
            _ => unreachable!(),
        };
        prop_assume!(sum > 1.0);
        let result = model.validate();
        let rejected = matches!(result, Err(ModelError::InvalidInstProbs { .. }));
        prop_assert!(rejected, "expected InvalidInstProbs, got {:?}", result);
    }
}

#[test]
fn exp_law_requires_exactly_one_rate_form() {
    // Neither form.
    let mut model = sample_model();
    model.components[0].automata[0].transitions[0].distrib = Distrib::Exp {
        rate: None,
        rate_expr: None,
    };
    assert!(matches!(
        model.validate(),
        Err(ModelError::ExpRateSpec { .. })
    ));
    // Both forms.
    model.components[0].automata[0].transitions[0].distrib = Distrib::Exp {
        rate: Some(0.1),
        rate_expr: Some(Expr::Const {
            value: Value::Float(0.1),
        }),
    };
    assert!(matches!(
        model.validate(),
        Err(ModelError::ExpRateSpec { .. })
    ));
    // Each single form is accepted.
    model.components[0].automata[0].transitions[0].distrib = Distrib::Exp {
        rate: Some(0.1),
        rate_expr: None,
    };
    model.validate().unwrap();
    model.components[0].automata[0].transitions[0].distrib = Distrib::Exp {
        rate: None,
        rate_expr: Some(Expr::Const {
            value: Value::Float(0.1),
        }),
    };
    model.validate().unwrap();
}
