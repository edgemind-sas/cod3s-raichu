use raichu_expr::{AggOp, BoolOp, CmpOp, Expr, PortRef, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Connection, Distrib, Equation, EquationKind, Model,
    Port, PortDir, Transition,
};

/// The trigger condition of a cold standby, in the three logics
/// muscadet offers, over the backup's own trigger in port.
///
/// `and` carries the emptiness of the port explicitly, `all(p) AND
/// count(p) >= 1`, because a conjunction over nothing is the vacuous
/// truth while an unconnected input is *absent*; `or` and `sum >= k`
/// already answer false over nothing.
pub fn trigger_condition(logic: &str) -> Expr {
    let port = PortRef {
        component: "Backup".into(),
        port: "cooling_trigger_in".into(),
    };
    let agg = |agg| Expr::PortAgg {
        port: port.clone(),
        agg,
        channel: None,
    };
    match logic {
        "and" => Expr::Bool {
            bool_op: BoolOp::And,
            args: vec![
                agg(AggOp::All),
                Expr::Cmp {
                    cmp: CmpOp::Ge,
                    lhs: Box::new(agg(AggOp::Count)),
                    rhs: Box::new(Expr::Const {
                        value: Value::Int(1),
                    }),
                },
            ],
        },
        "or" => agg(AggOp::Any),
        "k" => Expr::Cmp {
            cmp: CmpOp::Ge,
            lhs: Box::new(agg(AggOp::Sum)),
            rhs: Box::new(Expr::Const {
                value: Value::Int(2),
            }),
        },
        other => panic!("unknown trigger logic `{other}`"),
    }
}

/// A main pump and a backup whose output is armed while its trigger in
/// port is absent, the shape of `add_flow_out_on_trigger`.
///
/// `connected` wires the main pump's output to that trigger port, which
/// is the one line a modeller forgets. `init` is the state the backup's
/// trigger automaton starts in: `up` is what the authoring layer
/// declares when it can see the port is unwired, `down` what it declares
/// when a `time_up` makes the rise a real wait.
pub fn standby(logic: &str, connected: bool, init: &str) -> Model {
    let condition = trigger_condition(logic);
    Model {
        programs: vec![],
        name: "standby".into(),
        components: vec![
            Component {
                name: "Main".into(),
                attributes: vec![Attribute {
                    name: "cooling".into(),
                    kind: AttrKind::Bool,
                    init: Value::Bool(true),
                }],
                ports: vec![Port {
                    name: "cooling_out".into(),
                    dir: PortDir::Out,
                    attr: Some("cooling".into()),
                    channels: vec![],
                }],
                interfaces: vec![],
                automata: vec![],
                equations: vec![],
                allocations: vec![],
                sensitive_functions: vec![],
            },
            Component {
                name: "Backup".into(),
                attributes: vec![Attribute {
                    name: "cooling".into(),
                    kind: AttrKind::Float,
                    init: Value::Float(0.0),
                }],
                ports: vec![Port {
                    name: "cooling_trigger_in".into(),
                    dir: PortDir::In,
                    attr: None,
                    channels: vec![],
                }],
                interfaces: vec![],
                automata: vec![Automaton {
                    name: "cooling_trigger".into(),
                    states: vec!["down".into(), "up".into()],
                    init: init.into(),
                    transitions: vec![
                        Transition {
                            name: "cooling_trigger_up".into(),
                            source: "down".into(),
                            guard: Some(Expr::Bool {
                                bool_op: BoolOp::Not,
                                args: vec![condition.clone()],
                            }),
                            targets: vec!["up".into()],
                            on_interruption: Default::default(),
                            monitored: false,
                            cycle_group: None,
                            kind: None,
                            effects: vec![],
                            distrib: Distrib::Delay { time: 0.0 },
                        },
                        Transition {
                            name: "cooling_trigger_down".into(),
                            source: "up".into(),
                            guard: Some(condition),
                            targets: vec!["down".into()],
                            on_interruption: Default::default(),
                            monitored: false,
                            cycle_group: None,
                            kind: None,
                            effects: vec![],
                            distrib: Distrib::Delay { time: 0.0 },
                        },
                    ],
                }],
                // The backup delivers exactly while the trigger is up:
                // what the diagnostic warns about is this equation being
                // decided once and for all.
                equations: vec![Equation {
                    target: "cooling".into(),
                    kind: EquationKind::Explicit,
                    expr: Expr::If {
                        cond: Box::new(Expr::StateActive {
                            state: raichu_expr::StateRef {
                                component: "Backup".into(),
                                automaton: "cooling_trigger".into(),
                                state: "up".into(),
                            },
                        }),
                        then: Box::new(Expr::Const {
                            value: Value::Float(1.0),
                        }),
                        otherwise: Box::new(Expr::Const {
                            value: Value::Float(0.0),
                        }),
                    },
                }],
                allocations: vec![],
                sensitive_functions: vec![],
            },
        ],
        connections: if connected {
            vec![Connection {
                name: None,
                from: PortRef {
                    component: "Main".into(),
                    port: "cooling_out".into(),
                },
                to: PortRef {
                    component: "Backup".into(),
                    port: "cooling_trigger_in".into(),
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
