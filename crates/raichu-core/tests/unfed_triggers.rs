//! A trigger input nothing feeds, caught before any simulation.
//!
//! An aggregation over a port with no connection still answers, so a
//! guard reading such a port is pinned for the whole run: the mode it
//! drives is decided by the missing wire rather than by the model. The
//! standby it gates then delivers from the initial instant and for ever,
//! and nothing about the run says so.
//!
//! Reported only when the emptiness is what decides. A guard that also
//! reads something moving does not fold, so the port is not what settles
//! the mode and nothing is said: same compromise as the switching-loop
//! detector next door, where a loop whose every switch has a band is
//! silent.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{unfed_triggers, CompiledModel};
use raichu_expr::{AggOp, Assignment, AttrRef, BoolOp, CmpOp, Expr, PortRef, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Connection, Distrib, Equation, EquationKind, Model,
    Port, PortDir, SensitiveFunction, Transition,
};

#[path = "support/unfed_standby.rs"]
mod unfed_standby;
use unfed_standby::{standby, trigger_condition};

#[test]
fn a_trigger_port_nothing_feeds_is_reported_in_the_three_logics() {
    // `and`, `or` and k-out-of-n answer false over an empty port, each
    // for its own reason, and the standby is armed for ever in all
    // three. A detector that covered `and` alone would leave two thirds
    // of the fault open.
    for logic in ["and", "or", "k"] {
        let found = unfed_triggers(&standby(logic, false, "up"));
        assert_eq!(found.len(), 1, "{logic}: {found:?}");
        assert_eq!(
            found[0].ports,
            vec!["Backup.cooling_trigger_in".to_string()]
        );
        assert_eq!(found[0].automaton, "Backup.cooling_trigger");
        assert_eq!(found[0].state, "up");
    }
}

#[test]
fn the_same_model_with_its_trigger_wired_is_silent() {
    // One connection apart, the two models are identical. What changed
    // is that the aggregate now reads something that moves, so nothing
    // is pinned and there is nothing to report.
    for logic in ["and", "or", "k"] {
        let found = unfed_triggers(&standby(logic, true, "down"));
        assert!(found.is_empty(), "{logic}: {found:?}");
    }
}

#[test]
fn a_trigger_that_rises_after_a_wait_is_reported_where_it_settles() {
    // The automaton starts `down` because the rise is a real delay, so
    // the sealed state is not the initial one: the walk has to reach it
    // through the up transition, which the emptiness pins true.
    let found = unfed_triggers(&standby("and", false, "down"));
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].state, "up");
}

#[test]
fn the_diagnostic_names_the_wire_and_the_cure() {
    let found = unfed_triggers(&standby("and", false, "up"));
    let message = found[0].describe();
    assert!(
        message.starts_with("unfed trigger: nothing is connected to Backup.cooling_trigger_in"),
        "{message}"
    );
    assert!(message.contains("Backup.cooling_trigger"), "{message}");
    // The cost, in one clause: the reason this warning exists at all is
    // that the fault does not look like one, so a reader told only that
    // a mode is sealed has no reason to act on it.
    assert!(
        message.contains("availability the model does not have"),
        "{message}"
    );
    assert!(message.contains("Connect that input"), "{message}");
}

/// How long the message may be, derived from the diagnostic it is
/// modelled on rather than written down.
///
/// Computed from `SwitchingLoop::describe` so the day that wording is
/// rewritten this budget follows it, instead of going stale against a
/// number nobody remembers choosing. Both sides are counted **without
/// their qualified names**: this one interpolates a port, an automaton
/// and a state where a switching loop interpolates one list, and a model
/// whose components have long names would otherwise fail a test about
/// prose.
///
/// The fifth of slack on top is the only judgement here rather than a
/// measurement, and it buys one thing: a switching loop names a pattern
/// the reader already has a word for, while this has to walk them from a
/// port to an aggregate to an automaton to a state. That is one clause
/// more. It is not two.
fn message_budget() -> usize {
    // The shortest honest instance: one bandless automaton, one attribute.
    let sibling = raichu_core::SwitchingLoop {
        automata: vec!["P.duty".to_owned()],
        bandless: vec!["P.duty".to_owned()],
        through: vec!["P.supply".to_owned()],
    };
    let prose = sibling.describe().len() - "P.duty".len();
    prose + prose / 5
}

#[test]
fn the_diagnostic_stays_in_the_register_of_its_sibling() {
    // A warning nobody finishes reading is a warning that does not
    // exist, so the length is a property to hold rather than an accident
    // of whoever last edited the wording.
    let found = unfed_triggers(&standby("and", false, "up"));
    let message = found[0].describe();
    let prose = message.len()
        - "Backup.cooling_trigger_in".len()
        - "Backup.cooling_trigger".len()
        - "up".len();
    let budget = message_budget();

    assert!(
        prose <= budget,
        "the diagnostic has drifted out of the register of the one it is \
         modelled on: {prose} characters of prose against a budget of \
         {budget}. Either shorten it, or make the case in `message_budget` \
         for why this fault needs more words than a switching loop \
         does.\n{message}"
    );
}

#[test]
fn the_register_yardstick_bites() {
    // The guard above is only worth having if it rejects something, and
    // what it has to reject is known: the wording it replaced said the
    // same three things and spent a whole sentence on the middle one. Its
    // prose is kept here, names stripped, so a later rewrite cannot
    // quietly land back where this one started.
    let previous = "unfed trigger: nothing is connected to , so what that in port \
         aggregates is fixed for the whole run and  settles in `` at the \
         initial instant with no transition out of it. A mode decided by a \
         missing connection rather than by the model is the expensive kind of \
         mistake: a standby whose trigger is left unfed delivers from t = 0 \
         and for ever, so the campaign reads an availability the model does \
         not have. Connect that input, or drop the trigger and state the mode \
         unconditionally.";
    let budget = message_budget();
    assert!(
        previous.len() > budget,
        "the yardstick no longer rejects the wording it was written for: \
         {} characters against a budget of {budget}",
        previous.len()
    );
}

/// The standby's trigger condition widened with one more operand, which
/// is the shape every "the emptiness is not the only thing deciding"
/// case takes: `or` of the trigger aggregate and `also`.
fn condition_widened_with(model: &mut Model, also: Expr) {
    let condition = Expr::Bool {
        bool_op: BoolOp::Or,
        args: vec![trigger_condition("or"), also],
    };
    let transitions = &mut model.components[1].automata[0].transitions;
    transitions[0].guard = Some(Expr::Bool {
        bool_op: BoolOp::Not,
        args: vec![condition.clone()],
    });
    transitions[1].guard = Some(condition);
}

#[test]
fn a_guard_that_also_reads_something_moving_is_silent() {
    // The port is still unfed, and the mode may still be wrong. But the
    // emptiness is no longer what decides it: the attribute in the guard
    // can turn the condition either way, so the automaton is not sealed
    // and a diagnostic here would be a guess.
    //
    // "Moving" is a property of the model and not of the word: the
    // attribute is read by the guard AND written by a sensitive
    // function, which is what makes it able to turn the condition. An
    // attribute nobody writes would not move, and the test below is
    // where that case is held.
    let mut model = standby("or", false, "up");
    let backup = &mut model.components[1];
    backup.attributes.push(Attribute {
        name: "inhibited".into(),
        kind: AttrKind::Bool,
        init: Value::Bool(false),
    });
    backup.sensitive_functions.push(SensitiveFunction {
        name: "update_inhibited".into(),
        effects: vec![Assignment {
            target: AttrRef {
                component: "Backup".into(),
                attribute: "inhibited".into(),
            },
            value: Expr::attr("Main", "cooling"),
        }],
    });
    condition_widened_with(&mut model, Expr::attr("Backup", "inhibited"));
    assert!(unfed_triggers(&model).is_empty(), "{model:?}");
}

#[test]
fn a_guard_that_also_reads_the_clock_is_silent() {
    // The other way a condition stops being decided by the emptiness,
    // and the one no writer can be looked up for: a clock moves by
    // itself.
    let mut model = standby("or", false, "up");
    condition_widened_with(
        &mut model,
        Expr::Cmp {
            cmp: CmpOp::Ge,
            lhs: Box::new(Expr::Time),
            rhs: Box::new(Expr::Const {
                value: Value::Float(3.0),
            }),
        },
    );
    assert!(unfed_triggers(&model).is_empty(), "{model:?}");
}

#[test]
fn an_attribute_nothing_writes_is_read_as_the_constant_it_is() {
    // The companion of the test above, and the fact that separates them:
    // "reads an attribute" is not the same as "reads something moving".
    // No equation, no ODE and no sensitive function writes `inhibited`
    // here, so it holds its declared `false` from the initial instant to
    // the end of the run, the disjunction is decided by the empty port
    // alone, and the mode is sealed exactly as it was without the
    // operand. Abstaining here would be abstaining on a constant written
    // the long way.
    let mut model = standby("or", false, "up");
    model.components[1].attributes.push(Attribute {
        name: "inhibited".into(),
        kind: AttrKind::Bool,
        init: Value::Bool(false),
    });
    condition_widened_with(&mut model, Expr::attr("Backup", "inhibited"));
    let found = unfed_triggers(&model);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].state, "up");

    // And it is the VALUE that decides, not the absence of a writer: the
    // same attribute declared `true` settles the disjunction the other
    // way, the automaton leaves `up` at once, and nothing is sealed.
    model.components[1].attributes[1].init = Value::Bool(true);
    assert!(unfed_triggers(&model).is_empty(), "{model:?}");
}

#[test]
fn a_mode_sealed_by_a_constant_rather_than_by_a_port_is_silent() {
    // `false` seals the state just as thoroughly, and it is a
    // declaration: the modeller wrote a mode that is never left. Only a
    // missing wire is an oversight, so only a missing wire is reported.
    let mut model = standby("or", false, "up");
    model.components[1].automata[0].transitions[1].guard = Some(Expr::bool(false));
    let found = unfed_triggers(&model);
    assert!(found.is_empty(), "{found:?}");
}

#[test]
fn a_model_with_no_trigger_at_all_is_silent() {
    // The way to declare a standby that is always on: no trigger port,
    // no condition, nothing for this diagnostic to find. That is the
    // deliberate spelling the warning pushes towards, and it costs
    // nothing to take.
    let mut model = standby("and", false, "up");
    let backup = &mut model.components[1];
    backup.ports.clear();
    backup.automata.clear();
    backup.equations[0].expr = Expr::Const {
        value: Value::Float(1.0),
    };
    assert!(unfed_triggers(&model).is_empty());
}

#[test]
fn the_triggers_ride_on_the_compiled_model() {
    // Same route as the switching loops next door: found once, at
    // compile time, so a caller that asks for the diagnosis pays nothing
    // for it and one that never asks is warned through `tracing` anyway.
    let model = CompiledModel::compile(&standby("and", false, "up")).unwrap();
    assert_eq!(
        model.unfed_triggers,
        unfed_triggers(&standby("and", false, "up"))
    );
    assert_eq!(model.unfed_triggers.len(), 1);

    let wired = CompiledModel::compile(&standby("and", true, "down")).unwrap();
    assert!(wired.unfed_triggers.is_empty());
}

#[test]
fn the_two_diagnostics_do_not_answer_for_each_other() {
    // A trigger nothing feeds is not a cycle of the dependency graph, so
    // the loop detector says nothing about it. Keeping the two apart is
    // what lets each stay readable.
    let model = CompiledModel::compile(&standby("and", false, "up")).unwrap();
    assert!(model.switching_loops.is_empty());
    assert_eq!(model.unfed_triggers.len(), 1);
}

// --- across the definition an authoring layer puts in the way ----------
//
// The shape of a muscadet rule set, at the level the core sees it: the
// guard does not read the port, it reads an attribute an explicit
// equation derives from it. Nothing above is a trigger, and the fault is
// the same one: a threshold stated over an input nobody wired, true from
// the initial instant and for ever.

/// A supply, and a pump whose duty mode thresholds what its input
/// carries. `wired` connects the supply, `draws` makes the mode consume
/// the flow it thresholds, `floor` is the threshold and `below` the side
/// of it the mode is on.
///
/// With `draws`, the mode scales what it produces by what it receives,
/// which is what a `cons` does: sealed on an input nothing feeds, it
/// produces zero and the seal is harmless. Without it, the mode
/// produces at its nominal rate the moment it is entered.
fn duty(wired: bool, draws: bool, floor: f64, below: bool) -> Model {
    let received = Expr::attr("Pump", "received");
    let threshold = Expr::Cmp {
        cmp: if below { CmpOp::Lt } else { CmpOp::Ge },
        lhs: Box::new(received.clone()),
        rhs: Box::new(Expr::Const {
            value: Value::Float(floor),
        }),
    };
    let on = Expr::StateActive {
        state: raichu_expr::StateRef {
            component: "Pump".into(),
            automaton: "duty".into(),
            state: "on".into(),
        },
    };
    let scale = if draws {
        Expr::Div {
            lhs: Box::new(received),
            rhs: Box::new(Expr::Const {
                value: Value::Float(10.0),
            }),
        }
    } else {
        Expr::Const {
            value: Value::Float(1.0),
        }
    };
    Model {
        programs: vec![],
        name: "duty".into(),
        components: vec![
            Component {
                name: "Supply".into(),
                attributes: vec![Attribute {
                    name: "carried".into(),
                    kind: AttrKind::Float,
                    init: Value::Float(10.0),
                }],
                ports: vec![Port {
                    name: "power_out".into(),
                    dir: PortDir::Out,
                    attr: Some("carried".into()),
                    channels: vec![],
                }],
                interfaces: vec![],
                automata: vec![],
                equations: vec![],
                allocations: vec![],
                sensitive_functions: vec![],
            },
            Component {
                name: "Pump".into(),
                attributes: vec![
                    Attribute {
                        name: "received".into(),
                        kind: AttrKind::Float,
                        init: Value::Float(0.0),
                    },
                    Attribute {
                        name: "rate".into(),
                        kind: AttrKind::Float,
                        init: Value::Float(1.0),
                    },
                    Attribute {
                        name: "delivered".into(),
                        kind: AttrKind::Float,
                        init: Value::Float(0.0),
                    },
                ],
                ports: vec![Port {
                    name: "power_in".into(),
                    dir: PortDir::In,
                    attr: None,
                    channels: vec![],
                }],
                interfaces: vec![],
                automata: vec![Automaton {
                    name: "duty".into(),
                    states: vec!["off".into(), "on".into()],
                    init: "off".into(),
                    transitions: vec![
                        Transition {
                            name: "duty_on".into(),
                            source: "off".into(),
                            guard: Some(threshold.clone()),
                            targets: vec!["on".into()],
                            on_interruption: Default::default(),
                            monitored: false,
                            monitored_states: None,
                            cycle_group: None,
                            kind: None,
                            effects: vec![],
                            distrib: Distrib::Delay { time: 0.0 },
                        },
                        Transition {
                            name: "duty_off".into(),
                            source: "on".into(),
                            guard: Some(Expr::Bool {
                                bool_op: BoolOp::Not,
                                args: vec![threshold],
                            }),
                            targets: vec!["off".into()],
                            on_interruption: Default::default(),
                            monitored: false,
                            monitored_states: None,
                            cycle_group: None,
                            kind: None,
                            effects: vec![],
                            distrib: Distrib::Delay { time: 0.0 },
                        },
                    ],
                }],
                // The definition the guard reads through, and the one
                // the seal is judged on.
                equations: vec![
                    Equation {
                        target: "received".into(),
                        kind: EquationKind::Explicit,
                        expr: Expr::PortAgg {
                            port: PortRef {
                                component: "Pump".into(),
                                port: "power_in".into(),
                            },
                            agg: AggOp::Sum,
                            channel: None,
                        },
                    },
                    Equation {
                        target: "delivered".into(),
                        kind: EquationKind::Explicit,
                        expr: Expr::Mul {
                            args: vec![
                                Expr::attr("Pump", "rate"),
                                Expr::If {
                                    cond: Box::new(on),
                                    then: Box::new(scale),
                                    otherwise: Box::new(Expr::Const {
                                        value: Value::Float(0.0),
                                    }),
                                },
                            ],
                        },
                    },
                ],
                allocations: vec![],
                sensitive_functions: vec![],
            },
        ],
        connections: if wired {
            vec![Connection {
                name: None,
                from: PortRef {
                    component: "Supply".into(),
                    port: "power_out".into(),
                },
                to: PortRef {
                    component: "Pump".into(),
                    port: "power_in".into(),
                },
            }]
        } else {
            vec![]
        },
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
        unbounded_rate: None,
        fmu_units: vec![],
        interface_connections: vec![],
    }
}

#[test]
fn a_threshold_reaches_its_port_through_the_equation_that_defines_it() {
    // The case this whole crossing exists for: `received < 4` on an
    // input nobody wired is TRUE, because an empty sum is zero, so the
    // mode is entered at the initial instant and never left. The guard
    // names no port; the equation one step behind it does, and that is
    // the port the report has to name, because it is the one the reader
    // can wire.
    let found = unfed_triggers(&duty(false, false, 4.0, true));
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].ports, vec!["Pump.power_in".to_string()]);
    assert_eq!(found[0].automaton, "Pump.duty");
    assert_eq!(found[0].state, "on");
    assert!(
        found[0]
            .describe()
            .starts_with("unfed trigger: nothing is connected to Pump.power_in"),
        "{}",
        found[0].describe()
    );
}

#[test]
fn the_same_threshold_on_a_wired_input_is_silent() {
    // One connection apart. The equation now aggregates a port that
    // carries something, so it settles nothing, the guard settles
    // nothing, and there is nothing to say.
    let found = unfed_triggers(&duty(true, false, 4.0, true));
    assert!(found.is_empty(), "{found:?}");
}

#[test]
fn a_mode_that_draws_the_flow_nothing_feeds_is_silent() {
    // The same missing wire, the same sealed mode, and no warning: the
    // mode scales what it produces by what it receives, so sealed on an
    // empty input it delivers zero. That is the PESSIMISTIC direction,
    // and it is already in front of the reader in the results as a pump
    // that produces nothing. Reporting it would be crying on a model
    // whose fault already shows.
    let found = unfed_triggers(&duty(false, true, 4.0, true));
    assert!(found.is_empty(), "{found:?}");
}

#[test]
fn a_mode_the_emptiness_keeps_shut_is_silent() {
    // The other pessimistic shape, and the one the corpus has: the
    // threshold is stated the safe way round, `received >= 4`, so the
    // empty input keeps the mode OFF for ever. The automaton is sealed
    // just as thoroughly, in a state nothing reads, so nothing is
    // flattered and nothing is said.
    let found = unfed_triggers(&duty(false, false, 4.0, false));
    assert!(found.is_empty(), "{found:?}");
}

#[test]
fn an_attribute_two_writers_share_stops_the_crossing() {
    // "Exactly one definition" is the whole of the caution. A second
    // writer makes the attribute something the run moves, whatever the
    // first writer says, so the fold stops at it and the port behind is
    // out of reach. Silence rather than a guess.
    let mut model = duty(false, false, 4.0, true);
    model.components[1]
        .sensitive_functions
        .push(SensitiveFunction {
            name: "force_received".into(),
            effects: vec![Assignment {
                target: AttrRef {
                    component: "Pump".into(),
                    attribute: "received".into(),
                },
                value: Expr::attr("Supply", "carried"),
            }],
        });
    assert!(unfed_triggers(&model).is_empty(), "{model:?}");
}

#[test]
fn a_definition_that_reads_itself_back_does_not_spin() {
    // An explicit equation may read its own target: that is the solver's
    // business and not this fold's. What matters here is that it
    // terminates and abstains rather than recursing on itself.
    let mut model = duty(false, false, 4.0, true);
    model.components[1].equations[0].expr = Expr::Max {
        args: vec![
            Expr::attr("Pump", "received"),
            Expr::PortAgg {
                port: PortRef {
                    component: "Pump".into(),
                    port: "power_in".into(),
                },
                agg: AggOp::Sum,
                channel: None,
            },
        ],
    };
    assert!(unfed_triggers(&model).is_empty(), "{model:?}");
}
