//! Algebraic blocks: cycles of explicit equations classified at build
//! time and solved in place.
//!
//! The characterisation test at the bottom pins the *whole* validation
//! corpus against a recorded golden (compiled step table, sweep flags
//! and a run's end-to-end results), so the guarantee that a model that
//! declares no cycle is unchanged, bit for bit, is proven by an
//! unchanged record and not only by unchanged numbers. The golden lives
//! beside the fixtures it characterises, in the private validation
//! corpus; explicit execution requires that corpus to be present.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::compile::{CBlock, CStep};
use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_model::{Model, ModelError};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------
// Characterisation of the corpus
// ---------------------------------------------------------------------

/// Root of the private validation corpus, or `None` where it is absent
/// (a public checkout carries no `python/tests/validation`).
fn corpus_root() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/tests/validation");
    let fixtures = root.join("fixtures");
    if fixtures.is_dir() {
        Some(root)
    } else {
        None
    }
}

/// Every model file of the corpus, as stable keys, in a stable order
/// (subdirectories included, `gen_*` generators excluded). The corpus
/// carries two families: the authored fixtures under `fixtures/` and
/// the **expanded** bodies of the plugin-authored ones (the H2 plant
/// among them), which only the Python layer can produce and which are
/// recorded under `algebraic_characterisation_expanded/` so the
/// compiler they reach can be characterised from Rust alone.
fn corpus_models(root: &Path) -> Vec<(String, PathBuf)> {
    let mut found = Vec::new();
    for (_key, dir) in [
        ("fixtures", root.join("fixtures")),
        ("expanded", root.join("algebraic_characterisation_expanded")),
    ] {
        if !dir.is_dir() {
            continue;
        }
        let mut stack = vec![dir.clone()];
        while let Some(dir) = stack.pop() {
            let mut entries: Vec<_> = fs::read_dir(&dir)
                .expect("fixture directory reads")
                .map(|entry| entry.expect("directory entry reads").path())
                .collect();
            entries.sort();
            for entry in entries {
                if entry.is_dir() {
                    stack.push(entry);
                } else if entry.extension().is_some_and(|ext| ext == "json") {
                    let key = match entry.strip_prefix(root) {
                        Ok(relative) => relative.to_string_lossy().to_string(),
                        Err(_) => continue,
                    };
                    found.push((key, entry));
                }
            }
        }
    }
    found.sort();
    found
}

/// The characterisation record of one model: what the compiler built and
/// what one run of it produced, both serialised so a change in either
/// shows up as a diff of readable text.
fn characterise(json: &str) -> serde_json::Value {
    let mut record = serde_json::Map::new();
    match Model::from_json(json) {
        Err(error) => {
            record.insert("error".into(), serde_json::Value::String(error.to_string()));
        }
        Ok(model) => match CompiledModel::compile(&model) {
            Err(error) => {
                record.insert("error".into(), serde_json::Value::String(error.to_string()));
            }
            Ok(compiled) => {
                let table: Vec<String> = compiled
                    .explicit
                    .iter()
                    .map(|step| match step {
                        CStep::Equation { target, .. } => {
                            format!("equation {}", compiled.var_names[*target])
                        }
                        CStep::Allocate(allocation) => {
                            format!("allocate {}", allocation.name)
                        }
                        CStep::Block(block) => format!("block {}", block.members.join(", ")),
                    })
                    .collect();
                record.insert(
                    "table".into(),
                    serde_json::Value::Array(
                        table.into_iter().map(serde_json::Value::String).collect(),
                    ),
                );
                // The full compiled step expressions, Debug-rendered: the
                // strongest pin this side of a binary dump, and stable
                // for a fixed compiler.
                record.insert(
                    "steps_debug".into(),
                    serde_json::Value::String(format!("{:?}", compiled.explicit)),
                );
                record.insert("reads_time".into(), compiled.explicit_reads_time.into());
                record.insert(
                    "flow_margins".into(),
                    serde_json::Value::Array(
                        compiled
                            .flow_margins
                            .iter()
                            .map(|margins| margins.name.clone().into())
                            .collect(),
                    ),
                );
                // One deterministic run: its full result, events and
                // indicators included, so end-to-end equality is over
                // the bytes of the artefacts, not over a tolerance.
                let config = EngineConfig {
                    t_max: 10.0,
                    seed: 42,
                    ..EngineConfig::default()
                };
                match Engine::new(&compiled, config) {
                    Err(error) => {
                        record.insert("error".into(), serde_json::Value::String(error.to_string()));
                    }
                    Ok(engine) => match engine.run() {
                        Err(error) => {
                            record.insert(
                                "error".into(),
                                serde_json::Value::String(error.to_string()),
                            );
                        }
                        Ok(result) => {
                            let run = serde_json::json!({
                                "events": result.events,
                                "indicators": result.indicators,
                                "samples": result.samples,
                                "final_time": result.final_time,
                            });
                            record.insert("run".into(), serde_json::Value::String(run.to_string()));
                        }
                    },
                }
            }
        },
    }
    serde_json::Value::Object(record)
}

/// Path of the golden record, beside the fixtures it characterises.
fn golden_path(root: &Path) -> PathBuf {
    root.join("algebraic_characterisation.json")
}

/// The corpus compiles to the recorded step tables and runs to the
/// recorded results: the no-cycle characterisation. Explicit private
/// qualification is required; execution without the corpus fails.
#[test]
#[ignore = "requires the private validation corpus"]
fn corpus_characterisation_is_unchanged() {
    let root = corpus_root().expect("private validation corpus is required for characterisation");
    let golden: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(&fs::read_to_string(golden_path(&root)).expect("golden reads"))
            .expect("golden parses");
    let paths = corpus_models(&root);
    assert_eq!(
        paths.len(),
        golden.len(),
        "the corpus gained or lost a fixture: check the frozen baseline inventory"
    );
    let mut failures = Vec::new();
    for (key, path) in paths {
        let json = fs::read_to_string(&path).expect("fixture reads");
        let current = characterise(&json);
        if current != golden[&key] {
            let expected =
                serde_json::to_string_pretty(&golden[&key]).expect("golden entry renders");
            let actual = serde_json::to_string_pretty(&current).expect("record renders");
            failures.push(format!(
                "{key} diverged:\n--- recorded\n{expected}\n--- current\n{actual}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "the corpus changed under a characterised compiler:\n{}",
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------
// The scenario models
// ---------------------------------------------------------------------
/// Parse a model document (the refusal path reports `LoadError`s with
/// their own names).
fn parse(json: &str) -> Model {
    Model::from_json(json).expect("model JSON parses")
}

use raichu_core::{EngineError, FlowConfig};
use raichu_expr::Value;
use serde_json::{json, Value as Json};

fn number(value: f64) -> Json {
    json!({"op":"const", "value":{"kind":"float","value":value}})
}
fn attr(component: &str, attribute: &str) -> Json {
    json!({"op":"attr","attr":{"component":component,"attribute":attribute}})
}
fn add(args: Vec<Json>) -> Json {
    json!({"op":"add","args":args})
}
fn mul(args: Vec<Json>) -> Json {
    json!({"op":"mul","args":args})
}
fn sub(lhs: Json, rhs: Json) -> Json {
    json!({"op":"sub","lhs":lhs,"rhs":rhs})
}
fn div(lhs: Json, rhs: Json) -> Json {
    json!({"op":"div","lhs":lhs,"rhs":rhs})
}
fn choice(cond: Json, then: Json, otherwise: Json) -> Json {
    json!({"op":"if","cond":cond,"then":then,"otherwise":otherwise})
}
fn equation(target: &str, expr: Json) -> Json {
    json!({"target":target,"kind":"explicit","expr":expr})
}
fn component(name: &str, attributes: &[(&str, f64)], equations: Vec<Json>) -> Json {
    json!({"name":name,"attributes":attributes.iter().map(|(name,value)|json!({"name":name,"kind":"float","init":{"kind":"float","value":value}})).collect::<Vec<_>>(),"equations":equations})
}
fn document(components: Vec<Json>, indicators: &[(&str, &str, &str)]) -> Json {
    json!({"name":"algebraic","components":components,"indicators":indicators.iter().map(|(name,component,attribute)|json!({"name":name,"target":"attribute","attr":{"component":component,"attribute":attribute}})).collect::<Vec<_>>()})
}
fn compile_json(model: Json) -> CompiledModel {
    CompiledModel::compile(&parse(&model.to_string())).expect("the model compiles")
}
fn blocks_of(compiled: &CompiledModel) -> Vec<&CBlock> {
    compiled
        .explicit
        .iter()
        .filter_map(|step| {
            if let CStep::Block(block) = step {
                Some(block)
            } else {
                None
            }
        })
        .collect()
}
fn value(result: &raichu_core::SimulationResult, name: &str, time: f64) -> f64 {
    let series = result
        .samples
        .iter()
        .find(|s| s.name == name)
        .or_else(|| result.indicators.iter().find(|s| s.name == name))
        .unwrap();
    match series
        .points
        .iter()
        .find(|(date, _)| (*date - time).abs() < 1e-12)
        .unwrap()
        .1
    {
        Value::Float(v) => v,
        ref v => panic!("expected float, got {v:?}"),
    }
}
fn run(compiled: &CompiledModel, t_max: f64) -> raichu_core::SimulationResult {
    Engine::new(
        compiled,
        EngineConfig {
            t_max,
            samples: vec![0.0, t_max],
            ..EngineConfig::default()
        },
    )
    .unwrap()
    .run()
    .unwrap()
}
fn divider_model(breaker: bool) -> Json {
    let mut source = component("src", &[("v", 10.0)], vec![]);
    let current = if breaker {
        source["automata"] = json!([{"name":"b","states":["closed","open"],"init":"closed","transitions":[{"name":"trip","source":"closed","targets":["open"],"distrib":"delay","time":1.0}]}]);
        choice(
            json!({"op":"state_active","state":{"component":"src","automaton":"b","state":"closed"}}),
            attr("r1", "v1"),
            number(0.0),
        )
    } else {
        attr("r1", "v1")
    };
    document(
        vec![
            source,
            component(
                "r1",
                &[("v1", 0.0)],
                vec![equation("v1", sub(attr("src", "v"), attr("r2", "i")))],
            ),
            component("r2", &[("i", 0.0)], vec![equation("i", current)]),
        ],
        &[("v1", "r1", "v1"), ("i", "r2", "i")],
    )
}
fn self_model(expr: Json) -> Json {
    document(
        vec![component("c", &[("x", 0.0)], vec![equation("x", expr)])],
        &[("x", "c", "x")],
    )
}

#[test]
fn count_reads_topology_without_creating_sweep_dependencies() {
    for count_in_block in [true, false] {
        let count = json!({"op":"port_agg","agg":"count","port":{"component":"a","port":"input"}});
        let mut equations = vec![equation(
            "x",
            add(vec![
                mul(vec![number(0.5), attr("a", "x")]),
                if count_in_block {
                    count.clone()
                } else {
                    attr("a", "n")
                },
            ]),
        )];
        if !count_in_block {
            equations.push(equation("n", mul(vec![number(1.0), count])));
        }
        let mut a = component("a", &[("x", 0.0), ("n", 0.0)], equations);
        a["ports"] = json!([{"name":"input","dir":"in"}]);
        let mut b = component("b", &[("y", 0.0)], vec![equation("y", attr("a", "x"))]);
        b["ports"] = json!([{"name":"out","dir":"out","attr":"y"}]);
        let mut model = document(vec![a, b], &[("x", "a", "x"), ("y", "b", "y")]);
        model["connections"] = json!([{"name":"edge","from":{"component":"b","port":"out"},"to":{"component":"a","port":"input"}}]);
        let compiled = compile_json(model);
        let result = run(&compiled, 0.0);
        assert_eq!(value(&result, "x", 0.0), 2.0);
        assert_eq!(value(&result, "y", 0.0), 2.0);
    }
}
fn rc_model(blocked: bool) -> Json {
    let current = if blocked {
        sub(attr("cap", "v"), attr("res", "vt"))
    } else {
        div(attr("cap", "v"), number(1.5))
    };
    let mut cap = component(
        "cap",
        &[("v", 10.0), ("i", 0.0)],
        vec![
            json!({"target":"v","kind":"ode","expr":mul(vec![number(-1.0),attr("cap","i")])}),
            equation("i", current),
        ],
    );
    // Equal numbers of attributes and explicit steps make the cost comparison fair.
    cap["equations"][0]["kind"] = json!("ode");
    let resistance = component(
        "res",
        &[("vt", 0.0)],
        vec![equation(
            "vt",
            if blocked {
                mul(vec![number(0.5), attr("cap", "i")])
            } else {
                div(attr("cap", "v"), number(3.0))
            },
        )],
    );
    document(vec![cap, resistance], &[("v", "cap", "v")])
}
fn runtime_singular_model(initial: &str) -> Json {
    let mut model = self_model(choice(
        json!({"op":"state_active","state":{"component":"c","automaton":"m","state":"open"}}),
        attr("c", "x"),
        number(1.0),
    ));
    model["components"][0]["automata"] = json!([{"name":"m","states":["closed","open"],"init":initial,"transitions":[{"name":"trip","source":"closed","targets":["open"],"distrib":"delay","time":1.0}]}]);
    model
}

fn singular_trial_model(protected: bool) -> Json {
    let closed =
        json!({"op":"state_active","state":{"component":"c","automaton":"m","state":"closed"}});
    let late = json!({"op":"cmp","cmp":"ge","lhs":attr("c","y"),"rhs":number(0.2)});
    let mut c = component(
        "c",
        &[("x", 0.0), ("y", 0.0)],
        vec![
            equation(
                "x",
                choice(
                    json!({"op":"bool","bool_op":"and","args":[closed,late]}),
                    attr("c", "x"),
                    add(vec![mul(vec![number(0.5), attr("c", "x")]), number(1.0)]),
                ),
            ),
            json!({"target":"y","kind":"ode","expr":number(1.0)}),
        ],
    );
    c["automata"] = json!([{"name":"m","states":["closed","open"],"init":"closed","transitions":if protected {vec![json!({"name":"trip","source":"closed","targets":["open"],"distrib":"watched","guard":{"op":"cmp","cmp":"ge","lhs":attr("c","y"),"rhs":number(0.1)}})]} else {vec![]}}]);
    document(vec![c], &[("x", "c", "x"), ("y", "c", "y")])
}

#[test]
fn a_singular_ode_trial_does_not_block_an_earlier_protective_transition() {
    let compiled = compile_json(singular_trial_model(true));
    for max_step in [0.2, 0.5, 1.0] {
        let result = Engine::new(
            &compiled,
            EngineConfig {
                t_max: 0.5,
                samples: vec![0.0, 0.05, 0.15, 0.5],
                ode: raichu_numeric::SolverParams {
                    max_step,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .and_then(Engine::run)
        .expect("the committed trajectory is never singular");
        assert_eq!(result.events.len(), 1);
        assert!((result.events[0].time - 0.1).abs() <= 1e-10);
        assert_eq!(value(&result, "x", 0.5), 2.0);
        assert!((value(&result, "y", 0.5) - 0.5).abs() < 1e-12);
        for t in [0.05, 0.15] {
            assert_eq!(value(&result, "x", t), 2.0);
            assert!((value(&result, "y", t) - t).abs() < 1e-12);
        }
    }
}

#[test]
fn dynamic_block_coefficients_follow_the_continuous_state() {
    // q'=1, x=q/(1+q)*x+1, y'=x has x=1+t and y=t+t^2/2.
    // Every intermediate ODE evaluation must use its own coefficient,
    // rather than the LU factorisation of the previously committed state.
    let c = component(
        "c",
        &[("q", 0.0), ("x", 0.0), ("y", 0.0)],
        vec![
            json!({"target":"q","kind":"ode","expr":number(1.0)}),
            equation(
                "x",
                add(vec![
                    mul(vec![
                        div(attr("c", "q"), add(vec![number(1.0), attr("c", "q")])),
                        attr("c", "x"),
                    ]),
                    number(1.0),
                ]),
            ),
            json!({"target":"y","kind":"ode","expr":attr("c","x")}),
        ],
    );
    let compiled = compile_json(document(vec![c], &[("x", "c", "x"), ("y", "c", "y")]));
    let result = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 1.0,
            samples: vec![0.0, 0.1, 0.3, 0.7, 1.0],
            ..Default::default()
        },
    )
    .and_then(Engine::run)
    .unwrap();
    for t in [0.0, 0.1, 0.3, 0.7, 1.0] {
        assert!((value(&result, "x", t) - (1.0 + t)).abs() < 1e-9);
        assert!((value(&result, "y", t) - (t + t * t / 2.0)).abs() < 1e-9);
    }
}

#[test]
fn a_singular_continuous_trajectory_still_returns_the_typed_error() {
    let compiled = compile_json(singular_trial_model(false));
    let error = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 0.5,
            ode: raichu_numeric::SolverParams {
                max_step: 0.5,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .and_then(Engine::run)
    .unwrap_err();
    assert!(matches!(error, EngineError::AlgebraicSingular {time, ..} if time >= 0.2));
}

#[test]
fn protective_guards_reading_the_algebraic_block_do_not_spin() {
    let mut model = singular_trial_model(true);
    model["components"][0]["equations"][0]["expr"]["otherwise"]["args"][1] = attr("c", "y");
    model["components"][0]["automata"][0]["transitions"][0]["guard"] =
        json!({"op":"cmp","cmp":"ge","lhs":attr("c","x"),"rhs":number(0.2)});
    let compiled = compile_json(model);
    let result = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 0.5,
            samples: vec![0.5],
            ode: raichu_numeric::SolverParams {
                max_step: 0.5,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .and_then(Engine::run)
    .unwrap();
    assert_eq!(result.events.len(), 1);
    assert!((result.events[0].time - 0.1).abs() <= 1e-10);
    assert!(
        result.work.segments <= 10,
        "retries must make bounded progress"
    );
}

#[test]
fn a_later_transition_does_not_hide_a_real_continuous_singularity() {
    let mut model = singular_trial_model(true);
    model["components"][0]["automata"][0]["transitions"][0]["guard"]["rhs"] = number(0.3);
    let compiled = compile_json(model);
    let error = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 0.5,
            ode: raichu_numeric::SolverParams {
                max_step: 0.5,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .and_then(Engine::run)
    .unwrap_err();
    assert!(matches!(error, EngineError::AlgebraicSingular {time, ..} if time >= 0.2));
}

#[test]
fn a_stale_algebraic_guard_cannot_hide_an_earlier_protective_transition() {
    let mut model = singular_trial_model(true);
    model["components"][0]["equations"][0]["expr"]["otherwise"]["args"][1] =
        mul(vec![number(0.5), sub(number(1.0), attr("c", "y"))]);
    model["components"][0]["automata"][0]["transitions"][0]["guard"] =
        json!({"op":"cmp","cmp":"le","lhs":attr("c","x"),"rhs":number(0.81)});
    let compiled = compile_json(model);
    let result = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 0.5,
            samples: vec![0.5],
            ode: raichu_numeric::SolverParams {
                max_step: 0.5,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .and_then(Engine::run)
    .expect("x=1-y reaches the protective guard at .19");
    assert_eq!(result.events.len(), 1);
    assert!((result.events[0].time - 0.19).abs() <= 1e-10);
    assert!((value(&result, "x", 0.5) - 0.5).abs() < 1e-12);
    assert!(result.work.segments <= 64);
}

#[test]
fn an_isolated_singular_probe_is_not_erased_by_different_retry_dates() {
    let mut model = singular_trial_model(false);
    model["components"][0]["equations"][0]["expr"]["cond"]["args"][1] =
        json!({"op":"cmp","cmp":"eq","lhs":{"op":"time"},"rhs":number(0.4)});
    let compiled = compile_json(model);
    let error = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 0.5,
            ode: raichu_numeric::SolverParams {
                max_step: 0.5,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .and_then(Engine::run)
    .unwrap_err();
    assert!(matches!(error, EngineError::AlgebraicSingular {time, ..} if time == 0.4));
}
#[test]
fn a_voltage_divider_compiles_to_one_block_of_the_expected_variables() {
    let c = compile_json(divider_model(false));
    assert_eq!(blocks_of(&c)[0].members, vec!["r1.v1", "r2.i"]);
    assert_eq!(c.explicit.len(), 1);
    let r = run(&c, 0.0);
    assert!((value(&r, "v1", 0.0) - 5.0).abs() < 1e-12);
}
#[test]
fn copy_cycles_and_bare_self_references_are_refused_naming_the_variables() {
    for (m, variables, expected_reason) in [
        (
            self_model(attr("c", "x")),
            "c.x",
            raichu_model::SingularReason::Structural,
        ),
        (
            document(
                vec![component(
                    "c",
                    &[("x", 0.0), ("y", 0.0)],
                    vec![equation("x", attr("c", "y")), equation("y", attr("c", "x"))],
                )],
                &[],
            ),
            "c.x, c.y",
            raichu_model::SingularReason::RankDeficient,
        ),
    ] {
        match CompiledModel::compile(&parse(&m.to_string())) {
            Err(raichu_core::CompileError::Invalid(ModelError::AlgebraicSingular {
                variables: actual,
                reason,
            })) => {
                assert_eq!(actual, variables);
                assert_eq!(reason, expected_reason);
            }
            other => panic!("expected singular block, got {other:?}"),
        }
    }
}
#[test]
fn a_contracting_self_reference_solves_to_two() {
    let c = compile_json(self_model(add(vec![
        mul(vec![number(0.5), attr("c", "x")]),
        number(1.0),
    ])));
    assert_eq!(blocks_of(&c)[0].targets.len(), 1);
    assert!((value(&run(&c, 0.0), "x", 0.0) - 2.0).abs() < 1e-12);
}
#[test]
fn a_conditional_over_a_breaker_re_solves_on_the_trip() {
    let c = compile_json(divider_model(true));
    let r = run(&c, 2.0);
    assert!((value(&r, "v1", 0.0) - 5.0).abs() < 1e-12);
    assert!((value(&r, "v1", 2.0) - 10.0).abs() < 1e-12);
    assert_eq!(value(&r, "i", 2.0), 0.0);
    assert_eq!(r.events[0].time, 1.0);
}
#[test]
fn a_wheatstone_bridge_solves_to_its_closed_form() {
    // Conductances (source, ground, meter): left (1,1,1), right (2,1,1).
    let c = compile_json(document(
        vec![
            component(
                "a",
                &[("va", 0.0)],
                vec![equation(
                    "va",
                    div(add(vec![number(10.0), attr("b", "vb")]), number(3.0)),
                )],
            ),
            component(
                "b",
                &[("vb", 0.0)],
                vec![equation(
                    "vb",
                    div(add(vec![number(20.0), attr("a", "va")]), number(4.0)),
                )],
            ),
        ],
        &[("va", "a", "va"), ("vb", "b", "vb")],
    ));
    let r = run(&c, 0.0);
    assert!((value(&r, "va", 0.0) - 60.0 / 11.0).abs() < 1e-12);
    assert!((value(&r, "vb", 0.0) - 70.0 / 11.0).abs() < 1e-12);
}
#[test]
fn a_star_cycle_has_a_nonsingular_identity_minus_coefficient_pattern() {
    let c = compile_json(document(
        vec![component(
            "c",
            &[("x", 0.0), ("y", 0.0), ("z", 0.0)],
            vec![
                equation(
                    "x",
                    add(vec![
                        number(1.0),
                        mul(vec![number(0.1), attr("c", "y")]),
                        mul(vec![number(0.1), attr("c", "z")]),
                    ]),
                ),
                equation("y", mul(vec![number(0.1), attr("c", "x")])),
                equation("z", mul(vec![number(0.1), attr("c", "x")])),
            ],
        )],
        &[("x", "c", "x")],
    ));
    assert_eq!(blocks_of(&c)[0].targets.len(), 3);
    assert!((value(&run(&c, 0.0), "x", 0.0) - 1.0 / 0.98).abs() < 1e-12);
}
#[test]
fn a_nonmember_declared_between_members_reads_the_solved_value() {
    let c = compile_json(document(
        vec![component(
            "q",
            &[("a", 0.0), ("c", 0.0), ("b", 0.0)],
            vec![
                equation("b", mul(vec![number(0.5), attr("q", "a")])),
                equation("c", add(vec![attr("q", "b"), number(10.0)])),
                equation(
                    "a",
                    add(vec![mul(vec![number(0.5), attr("q", "b")]), number(2.0)]),
                ),
            ],
        )],
        &[("a", "q", "a"), ("c", "q", "c")],
    ));
    assert!(matches!(c.explicit[0], CStep::Block(_)));
    assert!((value(&run(&c, 0.0), "c", 0.0) - 34.0 / 3.0).abs() < 1e-12);
}
#[test]
fn a_runtime_singular_coefficient_names_the_variables_and_date() {
    for (initial, date) in [("closed", 1.0), ("open", 0.0)] {
        let c = compile_json(runtime_singular_model(initial));
        let error = Engine::new(
            &c,
            EngineConfig {
                t_max: 2.0,
                ..EngineConfig::default()
            },
        )
        .and_then(Engine::run)
        .unwrap_err();
        match error {
            EngineError::AlgebraicSingular {
                variables, time, ..
            } => {
                assert_eq!(variables, "c.x");
                assert_eq!(time, date);
            }
            other => panic!("expected runtime singularity, got {other:?}"),
        }
    }
}
#[test]
fn an_rc_block_inside_the_ode_matches_the_closed_form() {
    for blocked in [false, true] {
        let c = compile_json(rc_model(blocked));
        let r = run(&c, 3.0);
        let expected = 10.0 * (-2.0f64).exp();
        assert!(
            (value(&r, "v", 3.0) - expected).abs() < 2e-7,
            "blocked={blocked}"
        );
    }
}
#[test]
#[ignore = "wall-clock qualification under the explicit memory cap"]
fn rc_block_performance_stays_within_the_ratio_gate() {
    use std::time::Instant;
    let models = [compile_json(rc_model(false)), compile_json(rc_model(true))];
    let measure = |c: &CompiledModel| {
        let start = Instant::now();
        for _ in 0..10_000 {
            std::hint::black_box(run(c, 30.0));
        }
        start.elapsed().as_secs_f64()
    };
    for c in &models {
        for _ in 0..20 {
            std::hint::black_box(run(c, 30.0));
        }
    }
    let mut plain = Vec::new();
    let mut block = Vec::new();
    for round in 0..7 {
        if round % 2 == 0 {
            plain.push(measure(&models[0]));
            block.push(measure(&models[1]));
        } else {
            block.push(measure(&models[1]));
            plain.push(measure(&models[0]));
        }
    }
    plain.sort_by(f64::total_cmp);
    block.sort_by(f64::total_cmp);
    let ratio = block[3] / plain[3];
    eprintln!(
        "RC median seconds: plain={}, block={}, ratio={ratio}; sorted plain={plain:?}, sorted block={block:?}",
        plain[3], block[3]
    );
    assert!(ratio <= 1.5, "RC cost ratio {ratio} exceeds 1.5");
}

fn allocation_block_model(branching: bool, lazy: bool) -> Json {
    let allocated = attr("supply", "out__alloc__edge");
    let condition = json!({"op":"cmp","cmp":"gt","lhs":allocated,"rhs":number(1.0)});
    let constant = if branching {
        choice(condition, number(0.5 - 1e-6), number(0.5 + 1e-6))
    } else if lazy {
        choice(
            json!({"op":"const","value":{"kind":"bool","value":true}}),
            number(0.5),
            json!({"op":"min","args":[div(number(1.0),number(0.0)),number(0.0)]}),
        )
    } else {
        number(0.5)
    };
    let mut supply = component(
        "supply",
        &[("x", 0.0)],
        vec![
            equation(
                "x",
                add(vec![mul(vec![number(0.5), attr("supply", "x")]), constant]),
            ),
            equation("out__demand__edge", number(2.0)),
        ],
    );
    supply["ports"] = json!([{"name":"out","dir":"out","attr":"x","channels":[{"name":"demand"},{"name":"alloc"}]}]);
    supply["allocations"] = json!([{"name":"split","port":"out","demand":"demand","allocated":"alloc","available":attr("supply","x"),"policy":"proportional"}]);
    let mut consumer = component(
        "a",
        &[("got", 0.0)],
        vec![equation(
            "got",
            json!({"op":"port_agg","agg":"sum","channel":"alloc","port":{"component":"a","port":"input"}}),
        )],
    );
    consumer["ports"] = json!([{"name":"input","dir":"in"}]);
    let mut m = document(vec![supply, consumer], &[("got", "a", "got")]);
    m["connections"] = json!([{"name":"edge","from":{"component":"supply","port":"out"},"to":{"component":"a","port":"input"}}]);
    json!({"raichu_model":{"format":1,"requires":["allocation","evaluation_order"]},"model":m})
}
#[test]
fn block_rhs_decisions_enter_the_budget_and_cannot_hide_behind_numeric_tolerance() {
    let plain = compile_json(allocation_block_model(false, false));
    let conditional = compile_json(allocation_block_model(true, false));
    assert!(
        raichu_core::engine::active_set_budget(&conditional)
            > raichu_core::engine::active_set_budget(&plain)
    );
    let error = Engine::new(
        &conditional,
        EngineConfig {
            t_max: 0.0,
            flow: FlowConfig {
                tolerance: 1.0,
                relaxation: 1.0,
                active_set_budget: Some(4),
                ..FlowConfig::default()
            },
            ..EngineConfig::default()
        },
    )
    .and_then(Engine::run)
    .unwrap_err();
    assert!(
        matches!(
            error,
            EngineError::FlowNotConverged {
                cause: raichu_core::engine::FlowStall::TwoCycle,
                ..
            }
        ),
        "got {error:?}"
    );
    let lazy = compile_json(allocation_block_model(false, true));
    let result = run(&lazy, 0.0);
    assert!((value(&result, "got", 0.0) - 1.0).abs() < 1e-9);
}
#[test]
fn compatible_declared_order_keeps_an_allocation_reader_before_its_operator() {
    let mut m = allocation_block_model(false, false);
    m["model"]["evaluation_order"] = json!([{"component":"a","attribute":"got"},{"component":"supply","attribute":"x"},{"component":"supply","attribute":"out__demand__edge"},{"component":"supply","attribute":"split"}]);
    let c = compile_json(m);
    assert!(matches!(c.explicit[0], CStep::Equation { .. }));
    assert!(matches!(c.explicit[3], CStep::Allocate(_)));
    assert!((value(&run(&c, 0.0), "got", 0.0) - 1.0).abs() < 1e-9);
}
#[test]
fn a_declared_reader_before_its_block_is_refused_but_member_order_is_compatible() {
    let mut m = divider_model(false);
    m["components"][0]["attributes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name":"copy","kind":"float","init":{"kind":"float","value":0.0}}));
    m["components"][0]["equations"] = json!([equation("copy", attr("r2", "i"))]);
    for (order, refused) in [
        (vec![("src", "copy"), ("r1", "v1"), ("r2", "i")], true),
        (vec![("r2", "i"), ("r1", "v1"), ("src", "copy")], false),
    ] {
        m["evaluation_order"] = json!(order
            .iter()
            .map(|(c, a)| json!({"component":c,"attribute":a}))
            .collect::<Vec<_>>());
        let sealed = json!({"raichu_model":{"format":1,"requires":["evaluation_order"]},"model":m});
        let result = CompiledModel::compile(&parse(&sealed.to_string()));
        if refused {
            match result {
                Err(raichu_core::CompileError::Invalid(ModelError::AlgebraicOrderConflict {
                    first,
                    second,
                })) => {
                    assert_eq!(first, "src.copy");
                    assert!(second.contains("r2.i"));
                }
                other => panic!("expected order conflict, got {other:?}"),
            }
        } else {
            assert!(result.is_ok(), "{result:?}");
        }
    }
}

#[test]
fn condensation_schedules_late_inputs_before_blocks_and_keeps_independent_ties() {
    let c = compile_json(document(
        vec![component(
            "c",
            &[("first", 0.0), ("x", 0.0), ("last", 0.0), ("input", 0.0)],
            vec![
                equation("first", number(7.0)),
                equation(
                    "x",
                    add(vec![
                        mul(vec![number(0.5), attr("c", "x")]),
                        attr("c", "input"),
                    ]),
                ),
                equation("last", number(9.0)),
                equation("input", number(2.0)),
            ],
        )],
        &[("x", "c", "x")],
    ));
    let order = c
        .explicit
        .iter()
        .map(|step| match step {
            CStep::Equation { target, .. } => c.var_names[*target].as_str(),
            CStep::Block(_) => "block",
            CStep::Allocate(_) => "allocate",
        })
        .collect::<Vec<_>>();
    assert_eq!(order, ["c.first", "c.last", "c.input", "block"]);
    assert_eq!(value(&run(&c, 0.0), "x", 0.0), 4.0);
}
#[test]
fn nonlinear_unknown_terms_are_refused_with_the_offending_term() {
    for expr in [
        json!({"op":"min","args":[attr("c","x"),number(1.0)]}),
        choice(
            json!({"op":"cmp","cmp":"gt","lhs":attr("c","x"),"rhs":number(0.0)}),
            number(1.0),
            number(0.0),
        ),
    ] {
        let error = CompiledModel::compile(&parse(&self_model(expr).to_string())).unwrap_err();
        match error {
            raichu_core::CompileError::Invalid(ModelError::AlgebraicNonlinear {
                component,
                attribute,
                term,
            }) => {
                assert_eq!(component, "c");
                assert_eq!(attribute, "x");
                assert!(term.contains("min") || term.contains("if"), "{term}");
            }
            other => panic!("got {other:?}"),
        }
    }
}
#[test]
fn time_read_in_a_block_marks_the_final_sweep_table() {
    let c = compile_json(self_model(add(vec![
        mul(vec![number(0.5), attr("c", "x")]),
        json!({"op":"time"}),
    ])));
    assert!(c.explicit_reads_time);
    assert!((value(&run(&c, 2.0), "x", 2.0) - 4.0).abs() < 1e-12);
}
