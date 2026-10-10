//! Mixed-integer programs as discrete fixpoint steps.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use raichu_core::{CompiledModel, Engine, EngineConfig, JournalRecord};
use raichu_expr::Value;
use raichu_model::Model;
use serde_json::{json, Value as Json};

fn attr(name: &str) -> Json {
    json!({"component":"dispatch", "attribute":name})
}
fn read(name: &str) -> Json {
    json!({"op":"attr", "attr":attr(name)})
}
fn number(value: f64) -> Json {
    json!({"op":"const", "value":{"kind":"float", "value":value}})
}

fn model() -> Model {
    let body = json!({
        "name":"dispatch",
        "components":[{
            "name":"dispatch",
            "attributes":[
                {"name":"cheap", "kind":"float", "init":{"kind":"float", "value":0.0}},
                {"name":"expensive", "kind":"float", "init":{"kind":"float", "value":0.0}},
                {"name":"feasible", "kind":"bool", "init":{"kind":"bool", "value":false}},
                {"name":"cost", "kind":"float", "init":{"kind":"float", "value":0.0}},
                {"name":"demand", "kind":"float", "init":{"kind":"float", "value":80.0}}
            ],
            "automata":[{
                "name":"failure", "states":["ok", "down"], "init":"ok",
                "transitions":[
                    {"name":"fail", "source":"ok", "targets":["down"], "distrib":"delay", "time":5.0},
                    {"name":"repair", "source":"down", "targets":["ok"], "distrib":"delay", "time":10.0}
                ]
            }]
        }],
        "programs":[{
            "name":"least_cost",
            "variables":[
                {"attribute":attr("cheap"), "lower":number(0.0),
                    "upper":{"op":"if", "cond":{"op":"state_active", "state":{"component":"dispatch", "automaton":"failure", "state":"ok"}}, "then":number(60.0), "otherwise":number(0.0)},
                    "on_infeasible":{"kind":"float", "value":0.0}},
                {"attribute":attr("expensive"), "lower":number(0.0), "upper":number(50.0),
                    "on_infeasible":{"kind":"float", "value":0.0}}
            ],
            "sense":"minimize",
            "objective":{"op":"add", "args":[read("cheap"), {"op":"mul", "args":[number(2.0), read("expensive")]}]},
            "constraints":[{"name":"supply", "expr":{"op":"add", "args":[read("cheap"),read("expensive")]}, "lower":read("demand")}],
            "feasible":attr("feasible"), "objective_value":attr("cost")
        }],
        "indicators":[
            {"name":"cheap", "target":"attribute", "attr":attr("cheap")},
            {"name":"expensive", "target":"attribute", "attr":attr("expensive")},
            {"name":"feasible", "target":"attribute", "attr":attr("feasible")},
            {"name":"cost", "target":"attribute", "attr":attr("cost")}
        ]
    });
    Model::from_json(&Model::seal_json(&body.to_string()).unwrap()).unwrap()
}

#[test]
fn dispatch_recovers_after_infeasible_failure() {
    let compiled = CompiledModel::compile(&model()).unwrap();
    let config = EngineConfig {
        t_max: 16.0,
        journal: true,
        ..Default::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    let points = |name: &str| {
        result
            .indicators
            .iter()
            .find(|series| series.name == name)
            .unwrap()
            .points
            .clone()
    };
    assert_eq!(
        points("cheap"),
        vec![
            (0.0, Value::Float(60.0)),
            (5.0, Value::Float(0.0)),
            (15.0, Value::Float(60.0))
        ]
    );
    assert_eq!(
        points("expensive"),
        vec![
            (0.0, Value::Float(20.0)),
            (5.0, Value::Float(0.0)),
            (15.0, Value::Float(20.0))
        ]
    );
    assert_eq!(
        points("feasible"),
        vec![
            (0.0, Value::Bool(true)),
            (5.0, Value::Bool(false)),
            (15.0, Value::Bool(true))
        ]
    );
    assert_eq!(
        points("cost"),
        vec![
            (0.0, Value::Float(100.0)),
            (5.0, Value::Float(0.0)),
            (15.0, Value::Float(100.0))
        ]
    );
    let solved: Vec<_> = result
        .journal
        .iter()
        .filter_map(|record| match record {
            JournalRecord::ProgramSolved { time, status, .. } => Some((*time, *status)),
            _ => None,
        })
        .collect();
    assert_eq!(
        solved,
        vec![(0.0, "optimal"), (5.0, "infeasible"), (15.0, "optimal")]
    );
}

#[test]
fn repeated_trajectories_reuse_exact_programs() {
    let compiled = CompiledModel::compile(&model()).unwrap();
    let config = EngineConfig {
        t_max: 16.0,
        journal: true,
        ..Default::default()
    };
    let first = Engine::new(&compiled, config.clone())
        .unwrap()
        .run()
        .unwrap();
    let second = Engine::new(&compiled, config).unwrap().run().unwrap();
    assert_eq!(first.events, second.events);
    assert_eq!(first.indicators, second.indicators);
    assert_eq!(first.samples, second.samples);
    let cached: Vec<_> = second
        .journal
        .iter()
        .filter_map(|record| match record {
            JournalRecord::ProgramSolved { cached, .. } => Some(*cached),
            _ => None,
        })
        .collect();
    assert_eq!(cached, vec![true, true, true]);
}

#[test]
fn confluence_probe_and_parallel_runs_keep_the_dispatch() {
    use raichu_expr::{Assignment, AttrRef, Expr};
    use raichu_model::{AttrKind, Attribute, SensitiveFunction};

    let mut model = model();
    model.components[0].attributes.push(Attribute {
        name: "ready".into(),
        kind: AttrKind::Bool,
        init: Value::Bool(false),
    });
    model.components[0]
        .sensitive_functions
        .push(SensitiveFunction {
            name: "observe_dispatch".into(),
            effects: vec![Assignment {
                target: AttrRef {
                    component: "dispatch".into(),
                    attribute: "ready".into(),
                },
                value: Expr::attr("dispatch", "feasible"),
            }],
        });
    let compiled = CompiledModel::compile(&model).unwrap();
    let config = EngineConfig {
        t_max: 16.0,
        confluence_check: true,
        ..Default::default()
    };
    let serial = Engine::new(&compiled, config.clone())
        .unwrap()
        .run()
        .unwrap();
    std::thread::scope(|scope| {
        let tasks: Vec<_> = (0..4)
            .map(|_| {
                let config = config.clone();
                let compiled = &compiled;
                scope.spawn(move || Engine::new(compiled, config).unwrap().run().unwrap())
            })
            .collect();
        for task in tasks {
            let result = task.join().unwrap();
            assert_eq!(result.events, serial.events);
            assert_eq!(result.indicators, serial.indicators);
        }
    });
}

#[test]
fn feasibility_output_can_trigger_a_guard() {
    use raichu_expr::{BoolOp, Expr};
    use raichu_model::{Automaton, Distrib, Transition};

    let mut model = model();
    model.components[0].automata.push(Automaton {
        name: "alarm".into(),
        states: vec!["idle".into(), "active".into()],
        init: "idle".into(),
        transitions: vec![Transition {
            name: "raise".into(),
            source: "idle".into(),
            guard: Some(Expr::Bool {
                bool_op: BoolOp::Not,
                args: vec![Expr::attr("dispatch", "feasible")],
            }),
            targets: vec!["active".into()],
            on_interruption: Default::default(),
            monitored: false,
            monitored_states: None,
            cycle_group: None,
            kind: None,
            effects: vec![],
            distrib: Distrib::Inst { probs: vec![] },
        }],
    });
    let compiled = CompiledModel::compile(&model).unwrap();
    let result = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 6.0,
            ..Default::default()
        },
    )
    .unwrap()
    .run()
    .unwrap();
    assert!(result
        .events
        .iter()
        .any(|event| event.time == 5.0 && event.transition == "dispatch.alarm.raise"));
}

#[test]
fn unrelated_discrete_step_does_not_resolve_program() {
    use raichu_model::{Automaton, Distrib, Transition};

    let mut model = model();
    model.components[0].automata.push(Automaton {
        name: "unrelated".into(),
        states: vec!["before".into(), "after".into()],
        init: "before".into(),
        transitions: vec![Transition {
            name: "tick".into(),
            source: "before".into(),
            guard: None,
            targets: vec!["after".into()],
            on_interruption: Default::default(),
            monitored: false,
            monitored_states: None,
            cycle_group: None,
            kind: None,
            effects: vec![],
            distrib: Distrib::Delay { time: 3.0 },
        }],
    });
    let compiled = CompiledModel::compile(&model).unwrap();
    let result = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 6.0,
            journal: true,
            ..Default::default()
        },
    )
    .unwrap()
    .run()
    .unwrap();
    let solved: Vec<_> = result
        .journal
        .iter()
        .filter_map(|record| match record {
            JournalRecord::ProgramSolved { time, .. } => Some(*time),
            _ => None,
        })
        .collect();
    assert_eq!(solved, vec![0.0, 5.0]);
}

#[test]
fn equal_cost_dispatch_uses_declaration_order_tie_break() -> Result<(), &'static str> {
    let mut model = model();
    model.programs[0].objective = raichu_expr::Expr::Add {
        args: vec![
            raichu_expr::Expr::attr("dispatch", "cheap"),
            raichu_expr::Expr::attr("dispatch", "expensive"),
        ],
    };
    model.components[0]
        .attributes
        .iter_mut()
        .find(|attr| attr.name == "demand")
        .unwrap()
        .init = Value::Float(30.0);
    let compiled = CompiledModel::compile(&model).unwrap();
    let first = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let second = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let Some(Value::Float(cheap)) = first.attribute("dispatch.cheap") else {
        return Err("expected the cheap dispatch");
    };
    let Some(Value::Float(expensive)) = first.attribute("dispatch.expensive") else {
        return Err("expected the expensive dispatch");
    };
    assert!(cheap.abs() <= 1e-8);
    assert!((expensive - 30.0).abs() <= 1e-8);
    assert_eq!(
        first.attribute("dispatch.cheap"),
        second.attribute("dispatch.cheap")
    );
    Ok(())
}

#[test]
fn opting_out_of_the_tie_break_is_recorded_in_provenance() {
    let mut model = model();
    model.programs[0].tie_break = Some(raichu_model::ProgramTieBreak::None);
    let compiled = CompiledModel::compile(&model).unwrap();
    let result = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 20.0,
            ..EngineConfig::default()
        },
    )
    .unwrap()
    .run()
    .unwrap();
    assert_eq!(result.provenance.non_unique_programs, vec!["least_cost"]);
}

#[test]
fn non_finite_objective_is_a_named_program_error() {
    use raichu_expr::Expr;

    let mut model = model();
    model.programs[0].objective = Expr::Add {
        args: vec![
            Expr::attr("dispatch", "cheap"),
            Expr::Div {
                lhs: Box::new(Expr::Const {
                    value: Value::Float(1.0),
                }),
                rhs: Box::new(Expr::Const {
                    value: Value::Float(0.0),
                }),
            },
        ],
    };
    let compiled = CompiledModel::compile(&model).unwrap();
    let error = Engine::new(&compiled, EngineConfig::default())
        .err()
        .unwrap();
    let message = error.to_string();
    assert!(message.contains("least_cost"), "{message}");
    assert!(message.contains("non-finite"), "{message}");
}

#[test]
fn integer_outside_exact_float_range_is_not_published() {
    use raichu_expr::Expr;
    use raichu_model::{AttrKind, ProgramTieBreak};

    let mut model = model();
    model.components[0].attributes[0].kind = AttrKind::Int;
    model.components[0].attributes[0].init = Value::Int(0);
    let variable = &mut model.programs[0].variables[0];
    variable.lower = Some(Expr::Const {
        value: Value::Float(1e16),
    });
    variable.upper = Some(Expr::Const {
        value: Value::Float(1e16),
    });
    variable.on_infeasible = Value::Int(0);
    model.programs[0].objective = Expr::Const {
        value: Value::Float(0.0),
    };
    model.programs[0].tie_break = Some(ProgramTieBreak::None);
    let compiled = CompiledModel::compile(&model).unwrap();
    let error = Engine::new(&compiled, EngineConfig::default())
        .err()
        .unwrap();
    let message = error.to_string();
    assert!(message.contains("least_cost"), "{message}");
    assert!(message.contains("exact f64 range"), "{message}");
}
