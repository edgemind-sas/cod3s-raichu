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
                    channels: vec![],
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
                            monitored: false,
                            cycle_group: None,
                            distrib: Distrib::Delay { time: 5.0 },
                        },
                        Transition {
                            name: "repair".into(),
                            source: "nok".into(),
                            guard: None,
                            targets: vec!["ok".into()],
                            on_interruption: Default::default(),
                            monitored: false,
                            cycle_group: None,
                            distrib: Distrib::Delay { time: 10.0 },
                        },
                        Transition {
                            name: "branch".into(),
                            source: "limbo".into(),
                            guard: None,
                            targets: vec!["ok".into(), "nok".into()],
                            on_interruption: Default::default(),
                            monitored: false,
                            cycle_group: None,
                            distrib: Distrib::Inst { probs: vec![1.0] },
                        },
                    ],
                }],
                allocations: vec![],
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
                    channels: vec![],
                }],
                interfaces: vec![],
                automata: vec![],
                allocations: vec![],
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
                            channel: None,
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
            name: None,
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
        targets: vec![],
        evaluation_order: None,
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
        name: None,
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
        channel: None,
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

// ---------------------------------------------------------------------
// Flow-graph diagnostics: a model with no defensible answer is refused
// at build time, naming what is wrong.
// ---------------------------------------------------------------------

/// A float attribute initialised at zero.
fn float_attribute(name: &str) -> Attribute {
    Attribute {
        name: name.into(),
        kind: AttrKind::Float,
        init: Value::Float(0.0),
    }
}

/// An out port exporting `attribute`.
fn out_port(name: &str, attribute: &str) -> Port {
    Port {
        name: name.into(),
        dir: PortDir::Out,
        attr: Some(attribute.into()),
        channels: vec![],
    }
}

/// An in port (values are read through aggregations).
fn in_port(name: &str) -> Port {
    Port {
        name: name.into(),
        dir: PortDir::In,
        attr: None,
        channels: vec![],
    }
}

/// `sum` over everything connected to `component.port`.
fn sum_of(component: &str, port: &str) -> Expr {
    Expr::PortAgg {
        port: PortRef {
            component: component.into(),
            port: port.into(),
        },
        agg: AggOp::Sum,
        channel: None,
    }
}

/// An out-port → in-port connection.
fn connect(from: (&str, &str), to: (&str, &str)) -> Connection {
    Connection {
        name: None,
        from: PortRef {
            component: from.0.into(),
            port: from.1.into(),
        },
        to: PortRef {
            component: to.0.into(),
            port: to.1.into(),
        },
    }
}

/// Two components in a continuous cycle: `a.level` is computed from what
/// `b` exports and `b.rate` from what `a` exports. `first` and `second`
/// choose whether each equation is instantaneous (`Explicit`) or
/// integrated (`Ode`): an integrated attribute is the *capacity* that
/// breaks the loop.
fn continuous_cycle_model(first: EquationKind, second: EquationKind) -> Model {
    Model {
        name: "continuous_cycle".into(),
        components: vec![
            Component {
                name: "a".into(),
                attributes: vec![float_attribute("level")],
                ports: vec![out_port("level_out", "level"), in_port("rate_in")],
                interfaces: vec![],
                automata: vec![],
                sensitive_functions: vec![],
                allocations: vec![],
                equations: vec![Equation {
                    target: "level".into(),
                    kind: first,
                    expr: sum_of("a", "rate_in"),
                }],
            },
            Component {
                name: "b".into(),
                attributes: vec![float_attribute("rate")],
                ports: vec![out_port("rate_out", "rate"), in_port("level_in")],
                interfaces: vec![],
                automata: vec![],
                sensitive_functions: vec![],
                allocations: vec![],
                equations: vec![Equation {
                    target: "rate".into(),
                    kind: second,
                    expr: sum_of("b", "level_in"),
                }],
            },
        ],
        connections: vec![
            connect(("a", "level_out"), ("b", "level_in")),
            connect(("b", "rate_out"), ("a", "rate_in")),
        ],
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
    }
}

/// The spent-fuel-pool shape: several producers feeding one in port,
/// whose values a balance sums. `second` is the kind exported by the
/// second producer: `Bool` keeps the fan-in homogeneous (every producer
/// is a discrete on/off flow), `Float` mixes a discrete flow into a
/// continuous one.
fn fan_in_model(second: AttrKind) -> Model {
    let init = match second {
        AttrKind::Bool => Value::Bool(false),
        AttrKind::Int => Value::Int(0),
        AttrKind::Float => Value::Float(0.0),
    };
    Model {
        name: "fan_in".into(),
        components: vec![
            Component {
                name: "train1".into(),
                attributes: vec![Attribute {
                    name: "pumping".into(),
                    kind: AttrKind::Bool,
                    init: Value::Bool(true),
                }],
                ports: vec![out_port("pumping_out", "pumping")],
                interfaces: vec![],
                automata: vec![],
                sensitive_functions: vec![],
                allocations: vec![],
                equations: vec![],
            },
            Component {
                name: "train2".into(),
                attributes: vec![Attribute {
                    name: "pumping".into(),
                    kind: second,
                    init,
                }],
                ports: vec![out_port("pumping_out", "pumping")],
                interfaces: vec![],
                automata: vec![],
                sensitive_functions: vec![],
                allocations: vec![],
                equations: vec![],
            },
            Component {
                name: "pool".into(),
                attributes: vec![float_attribute("temperature")],
                ports: vec![in_port("pumping_in")],
                interfaces: vec![],
                automata: vec![],
                sensitive_functions: vec![],
                allocations: vec![],
                equations: vec![Equation {
                    target: "temperature".into(),
                    kind: EquationKind::Ode,
                    expr: sum_of("pool", "pumping_in"),
                }],
            },
        ],
        connections: vec![
            connect(("train1", "pumping_out"), ("pool", "pumping_in")),
            connect(("train2", "pumping_out"), ("pool", "pumping_in")),
        ],
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
    }
}

#[test]
fn rejects_continuous_cycle_with_no_capacity() {
    let model = continuous_cycle_model(EquationKind::Explicit, EquationKind::Explicit);
    assert_eq!(
        model.validate(),
        Err(ModelError::AlgebraicLoop {
            cycle: "a.level -> b.rate -> a.level".into(),
        })
    );
}

#[test]
fn accepts_continuous_cycle_broken_by_an_integrated_attribute() {
    // Same topology, one attribute now integrated: the capacity that
    // makes the loop well-posed, whichever side carries it.
    continuous_cycle_model(EquationKind::Ode, EquationKind::Explicit)
        .validate()
        .unwrap();
    continuous_cycle_model(EquationKind::Explicit, EquationKind::Ode)
        .validate()
        .unwrap();
}

#[test]
fn rejects_self_referential_explicit_equation() {
    let mut model = continuous_cycle_model(EquationKind::Explicit, EquationKind::Ode);
    model.components[0].equations[0].expr = Expr::Add {
        args: vec![
            Expr::attr("a", "level"),
            Expr::Const {
                value: Value::Float(1.0),
            },
        ],
    };
    assert_eq!(
        model.validate(),
        Err(ModelError::AlgebraicLoop {
            cycle: "a.level -> a.level".into(),
        })
    );
}

#[test]
fn rejects_connection_joining_a_discrete_flow_to_a_continuous_one() {
    assert_eq!(
        fan_in_model(AttrKind::Float).validate(),
        Err(ModelError::ConnectionFamilyMismatch {
            port: "pool.pumping_in".into(),
            other: "train1.pumping_out".into(),
            other_kind: AttrKind::Bool,
            producer: "train2.pumping_out".into(),
            kind: AttrKind::Float,
        })
    );
}

#[test]
fn accepts_a_homogeneous_discrete_fan_in() {
    // The `pool.json` shape: booleans summed into a balance is the
    // modeller's deliberate unit count (each running train removes a
    // fixed load). It must keep validating.
    fan_in_model(AttrKind::Bool).validate().unwrap();
}

// --- declared evaluation order -------------------------------------

/// A constant-valued equation of the given kind.
fn constant_equation(target: &str, kind: EquationKind, value: f64) -> Equation {
    Equation {
        target: target.into(),
        kind,
        expr: Expr::Const {
            value: Value::Float(value),
        },
    }
}

/// One `evaluation_order` entry.
fn order_entry(component: &str, attribute: &str) -> AttrRef {
    AttrRef {
        component: component.into(),
        attribute: attribute.into(),
    }
}

/// Two explicit equations and one ODE in a single component: the ODE
/// target is carried by the integrator and is therefore *not* part of
/// the evaluation order.
fn ordered_model(order: Option<Vec<AttrRef>>) -> Model {
    Model {
        name: "ordered".into(),
        components: vec![Component {
            name: "c".into(),
            attributes: vec![
                float_attribute("u"),
                float_attribute("v"),
                float_attribute("w"),
            ],
            ports: vec![],
            interfaces: vec![],
            automata: vec![],
            sensitive_functions: vec![],
            allocations: vec![],
            equations: vec![
                constant_equation("u", EquationKind::Explicit, 1.0),
                constant_equation("v", EquationKind::Explicit, 2.0),
                constant_equation("w", EquationKind::Ode, 3.0),
            ],
        }],
        connections: vec![],
        indicators: vec![],
        targets: vec![],
        evaluation_order: order,
    }
}

#[test]
fn accepts_an_order_covering_every_explicit_equation() {
    ordered_model(Some(vec![order_entry("c", "v"), order_entry("c", "u")]))
        .validate()
        .unwrap();
}

#[test]
fn rejects_an_order_omitting_an_equation() {
    assert_eq!(
        ordered_model(Some(vec![order_entry("c", "u")])).validate(),
        Err(ModelError::EvaluationOrderMissing {
            component: "c".into(),
            attribute: "v".into(),
        })
    );
}

#[test]
fn rejects_an_order_naming_a_missing_equation() {
    // `c.z` has no equation at all, `c.w` has an ODE (the integrator
    // carries it, the sweep never touches it): neither is orderable.
    for absent in ["z", "w"] {
        assert_eq!(
            ordered_model(Some(vec![
                order_entry("c", "u"),
                order_entry("c", "v"),
                order_entry("c", absent),
            ]))
            .validate(),
            Err(ModelError::EvaluationOrderUnknown {
                component: "c".into(),
                attribute: absent.into(),
            })
        );
    }
}

#[test]
fn rejects_an_order_listing_an_equation_twice() {
    assert_eq!(
        ordered_model(Some(vec![
            order_entry("c", "u"),
            order_entry("c", "u"),
            order_entry("c", "v"),
        ]))
        .validate(),
        Err(ModelError::EvaluationOrderDuplicate {
            component: "c".into(),
            attribute: "u".into(),
        })
    );
}

// --- format envelope ------------------------------------------------

#[test]
fn a_model_declares_the_features_it_actually_uses() {
    assert!(sample_model().required_features().is_empty());
    let ordered = ordered_model(Some(vec![order_entry("c", "u"), order_entry("c", "v")]));
    assert_eq!(
        ordered.required_features().into_iter().collect::<Vec<_>>(),
        vec![Feature::EvaluationOrder]
    );
    assert_eq!(
        ordered.format_header(),
        FormatHeader {
            format: FORMAT_REVISION,
            requires: vec!["evaluation_order".to_owned()],
        }
    );
}

#[test]
fn sealed_round_trip_preserves_the_order_and_the_feature_list() {
    let model = ordered_model(Some(vec![order_entry("c", "v"), order_entry("c", "u")]));
    let json = model.to_json().unwrap();
    let document: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(document[ENVELOPE_KEY]["format"], FORMAT_REVISION);
    assert_eq!(document[ENVELOPE_KEY]["requires"][0], "evaluation_order");
    assert_eq!(document["model"]["evaluation_order"][0]["attribute"], "v");

    let back = Model::from_json(&json).unwrap();
    assert_eq!(model, back);
    assert_eq!(back.format_header(), model.format_header());
    back.validate().unwrap();
}

#[test]
fn a_bare_body_still_loads() {
    // The whole existing corpus is bare: it declares no feature and
    // uses none, so it keeps loading unchanged.
    let bare = serde_json::to_string(&sample_model()).unwrap();
    assert!(!bare.contains(ENVELOPE_KEY));
    assert_eq!(Model::from_json(&bare).unwrap(), sample_model());
}

#[test]
fn refuses_a_feature_this_engine_does_not_implement() {
    // What a *newer* engine's document looks like to this one: a
    // feature name it has never heard of. Refused by name, never
    // loaded with the construct silently ignored.
    let document = format!(
        r#"{{"{ENVELOPE_KEY}": {{"format": 1, "requires": ["flow_network"]}},
            "model": {{"name": "m", "components": []}}}}"#
    );
    let error = Model::from_json(&document).unwrap_err();
    assert!(
        matches!(
            &error,
            LoadError::UnsupportedFeature { feature, known, .. }
                if feature == "flow_network" && known == &Feature::known()
        ),
        "expected an unsupported-feature refusal, got {error:?}"
    );
    assert!(error.to_string().contains("flow_network"));
}

#[test]
fn refuses_a_construct_the_document_does_not_declare() {
    // A hand-written model cannot use a non-baseline construct and stay
    // silent about it: the reader derives the truth from the body.
    let model = ordered_model(Some(vec![order_entry("c", "u"), order_entry("c", "v")]));
    let bare = serde_json::to_string(&model).unwrap();
    let error = Model::from_json(&bare).unwrap_err();
    assert!(
        matches!(
            error,
            LoadError::FeatureNotDeclared {
                feature: "evaluation_order",
                ..
            }
        ),
        "got {error:?}"
    );

    // Same body, enveloped but declaring nothing: same refusal.
    let under_declared =
        format!(r#"{{"{ENVELOPE_KEY}": {{"format": 1, "requires": []}}, "model": {bare}}}"#);
    assert!(matches!(
        Model::from_json(&under_declared).unwrap_err(),
        LoadError::FeatureNotDeclared { .. }
    ));
}

#[test]
fn a_legacy_reader_cannot_parse_a_sealed_document() {
    // The compatibility direction, proven rather than asserted:
    // `serde_json::from_str::<Model>` *is* what an engine predating the
    // envelope did (its `from_json` was exactly this call). The
    // envelope displaces `name` and `components` from the top level, so
    // that reader fails on a missing field instead of succeeding on a
    // misread. Silence is what would be fatal; a refusal is not.
    let sealed = ordered_model(Some(vec![order_entry("c", "u"), order_entry("c", "v")]))
        .to_json()
        .unwrap();
    let legacy = serde_json::from_str::<Model>(&sealed).unwrap_err();
    assert!(
        legacy.to_string().contains("missing field `name`"),
        "got {legacy}"
    );
}

#[test]
fn sealing_an_authored_body_derives_its_feature_list() {
    // The writer side: an authoring layer produces a bare body and does
    // not compose the envelope itself, so the list cannot lag.
    let model = ordered_model(Some(vec![order_entry("c", "u"), order_entry("c", "v")]));
    let bare = serde_json::to_string(&model).unwrap();
    let sealed = Model::seal_json(&bare).unwrap();
    assert_eq!(Model::from_json(&sealed).unwrap(), model);
    // Idempotent: sealing an already sealed document changes nothing.
    assert_eq!(Model::seal_json(&sealed).unwrap(), sealed);

    // Sealing cannot honour a feature this engine does not implement,
    // and re-deriving the list would silently drop it: refused instead.
    let alien = format!(
        r#"{{"{ENVELOPE_KEY}": {{"format": 1, "requires": ["flow_network"]}}, "model": {bare}}}"#
    );
    assert!(matches!(
        Model::seal_json(&alien).unwrap_err(),
        LoadError::UnsupportedFeature { .. }
    ));
}
