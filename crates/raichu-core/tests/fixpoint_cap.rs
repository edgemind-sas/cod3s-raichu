//! The fixpoint's loop guard against the size of the model.
//!
//! [`EngineConfig::max_fixpoint_iterations`] exists to turn an
//! instantaneous loop into a typed error instead of a hang. A loop needs
//! a cycle: functions that rewrite each other's inputs, or their own. A
//! function on no such cycle re-runs only when something upstream of it
//! changed, so it cannot keep a fixpoint from converging however many
//! times it runs. These tests pin both halves: a large model that merely
//! settles is never mistaken for a loop, and a loop is still caught, and
//! named, whatever surrounds it.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError};
use raichu_expr::{Assignment, AttrRef, BoolOp, Expr, StateRef, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Distrib, Model, SensitiveFunction, Transition,
};

fn bool_attr(name: &str, init: bool) -> Attribute {
    Attribute {
        name: name.into(),
        kind: AttrKind::Bool,
        init: Value::Bool(init),
    }
}

fn assign(component: &str, attribute: &str, value: Expr) -> Assignment {
    Assignment {
        target: AttrRef {
            component: component.into(),
            attribute: attribute.into(),
        },
        value,
    }
}

fn component(
    name: &str,
    attributes: Vec<Attribute>,
    functions: Vec<SensitiveFunction>,
) -> Component {
    Component {
        name: name.into(),
        attributes,
        ports: vec![],
        interfaces: vec![],
        automata: vec![],
        allocations: vec![],
        equations: vec![],
        sensitive_functions: functions,
    }
}

fn model(components: Vec<Component>) -> Model {
    Model {
        programs: vec![],
        name: "fixpoint_cap".into(),
        components,
        connections: vec![],
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
        unbounded_rate: None,
        fmu_units: vec![],
        interface_connections: vec![],
        source: None,
    }
}

/// `count` components, each with one function that settles its own
/// attribute once at t = 0 (it reads nothing it writes): the shape of a
/// model of independent components, one failure mode each.
fn settling_once(count: usize) -> Vec<Component> {
    (0..count)
        .map(|i| {
            let name = format!("P{i}");
            component(
                &name,
                vec![bool_attr("ok", true)],
                vec![SensitiveFunction {
                    name: "apply".into(),
                    effects: vec![assign(
                        &name,
                        "ok",
                        Expr::Const {
                            value: Value::Bool(false),
                        },
                    )],
                }],
            )
        })
        .collect()
}

/// More settling functions than the default cap, as in a model at the
/// platform's 10 000-component load target: the initialization axiom
/// runs each of them once, and nothing in the model loops.
#[test]
fn a_model_larger_than_the_cap_settles_at_initialization() {
    let count = EngineConfig::default().max_fixpoint_iterations + 2_000;
    let compiled = CompiledModel::compile(&model(settling_once(count))).unwrap();
    let engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    assert_eq!(engine.attribute("P0.ok"), Some(Value::Bool(false)));
    assert_eq!(
        engine.attribute(&format!("P{}.ok", count - 1)),
        Some(Value::Bool(false))
    );
}

/// A reader declared before its writers re-runs once per writer, since
/// the worklist pops the lowest index first: it runs far more often than
/// the cap here, and still converges, because nothing it writes feeds
/// back into what it reads.
#[test]
fn a_fan_in_reader_re_running_once_per_writer_is_not_a_loop() {
    let writers = 50;
    let mut components = vec![component(
        "all",
        vec![bool_attr("up", true)],
        vec![SensitiveFunction {
            name: "collect".into(),
            effects: vec![assign(
                "all",
                "up",
                Expr::Bool {
                    bool_op: BoolOp::And,
                    args: (0..writers)
                        .map(|i| Expr::attr(format!("P{i}"), "ok"))
                        .collect(),
                },
            )],
        }],
    )];
    // Writers flip `ok` one after the other, from false to true: `up`
    // changes only on the last one, but `collect` runs after each.
    components.extend((0..writers).map(|i| {
        let name = format!("P{i}");
        component(
            &name,
            vec![bool_attr("ok", false)],
            vec![SensitiveFunction {
                name: "apply".into(),
                effects: vec![assign(
                    &name,
                    "ok",
                    Expr::Const {
                        value: Value::Bool(true),
                    },
                )],
            }],
        )
    }));
    let compiled = CompiledModel::compile(&model(components)).unwrap();
    let config = EngineConfig {
        max_fixpoint_iterations: 10,
        ..EngineConfig::default()
    };
    let engine = Engine::new(&compiled, config).unwrap();
    assert_eq!(engine.attribute("all.up"), Some(Value::Bool(true)));
}

/// Two functions rewriting each other's input never converge. Beside a
/// model already larger than the cap, the loop is still the one reported,
/// and the error names a function on it.
#[test]
fn a_loop_beside_a_large_settling_model_is_caught_and_named() {
    let mut components = settling_once(EngineConfig::default().max_fixpoint_iterations + 2_000);
    components.push(component(
        "osc",
        vec![bool_attr("x", false), bool_attr("y", false)],
        vec![
            SensitiveFunction {
                name: "lead".into(),
                effects: vec![assign(
                    "osc",
                    "x",
                    Expr::Bool {
                        bool_op: BoolOp::Not,
                        args: vec![Expr::attr("osc", "y")],
                    },
                )],
            },
            SensitiveFunction {
                name: "follow".into(),
                effects: vec![assign("osc", "y", Expr::attr("osc", "x"))],
            },
        ],
    ));
    let compiled = CompiledModel::compile(&model(components)).unwrap();
    match Engine::new(&compiled, EngineConfig::default()).map(|_| ()) {
        Err(EngineError::InstantaneousLoop {
            function,
            iterations,
            ..
        }) => {
            assert!(
                function == "osc.lead" || function == "osc.follow",
                "named `{function}`, which is not on the loop"
            );
            assert_eq!(iterations, EngineConfig::default().max_fixpoint_iterations);
        }
        other => panic!("expected an instantaneous loop, got {other:?}"),
    }
}

fn int_attr(name: &str) -> Attribute {
    Attribute {
        name: name.into(),
        kind: AttrKind::Int,
        init: Value::Int(0),
    }
}

/// A chain of `depth` diamonds, every reader declared before the
/// functions it reads: `x{k} := y{k+1} + z{k+1}`, `y{k} := x{k}`,
/// `z{k} := x{k}`, fed by a constant at the bottom. No function lies on a
/// cycle, but lowest-index-first propagation re-runs the top once per
/// path, so its runs double with each diamond.
fn diamond_chain(depth: usize) -> Vec<Component> {
    let mut components = Vec::new();
    for k in 0..depth {
        components.push(component(
            &format!("x{k}"),
            vec![int_attr("v")],
            vec![SensitiveFunction {
                name: "sum".into(),
                effects: vec![assign(
                    &format!("x{k}"),
                    "v",
                    Expr::Add {
                        args: vec![
                            Expr::attr(format!("y{}", k + 1), "v"),
                            Expr::attr(format!("z{}", k + 1), "v"),
                        ],
                    },
                )],
            }],
        ));
        for side in ["y", "z"] {
            components.push(component(
                &format!("{side}{}", k + 1),
                vec![int_attr("v")],
                vec![SensitiveFunction {
                    name: "copy".into(),
                    effects: vec![assign(
                        &format!("{side}{}", k + 1),
                        "v",
                        Expr::attr(format!("x{}", k + 1), "v"),
                    )],
                }],
            ));
        }
    }
    components.push(component(
        &format!("x{depth}"),
        vec![int_attr("v")],
        vec![SensitiveFunction {
            name: "source".into(),
            effects: vec![assign(
                &format!("x{depth}"),
                "v",
                Expr::Const {
                    value: Value::Int(1),
                },
            )],
        }],
    ));
    components
}

/// Path re-runs that stay within the work budget converge to the right
/// value however much they exceed the per-step cap: no step is on a cycle.
#[test]
fn a_reconvergent_chain_within_the_work_budget_converges() {
    let compiled = CompiledModel::compile(&model(diamond_chain(8))).unwrap();
    let engine = Engine::new(&compiled, EngineConfig::default()).unwrap();
    assert_eq!(engine.attribute("x0.v"), Some(Value::Int(256)));
}

/// The same shape, deep enough that its path re-runs explode, is stopped
/// by the work budget (twice the cap per step) with an error that
/// says it is not a loop, instead of running for hours.
#[test]
fn an_exploding_reconvergent_chain_is_stopped_by_the_work_budget() {
    let compiled = CompiledModel::compile(&model(diamond_chain(14))).unwrap();
    let config = EngineConfig {
        max_fixpoint_iterations: 4,
        ..EngineConfig::default()
    };
    match Engine::new(&compiled, config).map(|_| ()) {
        Err(EngineError::FixpointWorkExceeded { runs, steps, .. }) => {
            assert_eq!(steps, 3 * 14 + 1);
            assert_eq!(runs, 2 * 4 * steps);
        }
        other => panic!("expected the work budget to stop the fixpoint, got {other:?}"),
    }
}

/// A loop armed by a transition fails its fixpoint; resetting the engine
/// then re-runs the initialization axiom from fresh counts, so the step
/// that hit the cap is not charged again for the runs it made before.
#[test]
fn a_reset_after_a_loop_starts_from_fresh_counts() {
    let armed = Expr::StateActive {
        state: StateRef {
            component: "osc".into(),
            automaton: "arm".into(),
            state: "on".into(),
        },
    };
    let mut osc = component(
        "osc",
        vec![bool_attr("x", false)],
        vec![SensitiveFunction {
            name: "flip".into(),
            effects: vec![assign(
                "osc",
                "x",
                Expr::If {
                    cond: Box::new(armed),
                    then: Box::new(Expr::Bool {
                        bool_op: BoolOp::Not,
                        args: vec![Expr::attr("osc", "x")],
                    }),
                    otherwise: Box::new(Expr::attr("osc", "x")),
                },
            )],
        }],
    );
    osc.automata = vec![Automaton {
        name: "arm".into(),
        states: vec!["off".into(), "on".into()],
        init: "off".into(),
        transitions: vec![Transition {
            name: "arm".into(),
            source: "off".into(),
            guard: None,
            targets: vec!["on".into()],
            on_interruption: Default::default(),
            monitored: false,
            monitored_states: None,
            cycle_group: None,
            cycle_cancellation: Default::default(),
            kind: None,
            effects: vec![],
            distrib: Distrib::Delay { time: 1.0 },
        }],
    }];
    let compiled = CompiledModel::compile(&model(vec![osc])).unwrap();
    let config = EngineConfig {
        max_fixpoint_iterations: 10,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new(&compiled, config).unwrap();
    assert!(matches!(
        engine.step(),
        Err(EngineError::InstantaneousLoop { .. })
    ));
    engine.reset().unwrap();
    assert_eq!(engine.attribute("osc.x"), Some(Value::Bool(false)));
}
