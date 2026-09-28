//! FMI communication points remain separate from authored transition indices.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{Read, Write};
use std::path::PathBuf;

use raichu_core::compile::CLaw;
use raichu_core::{
    fault_tree, CoSimulationHost, CompiledModel, Engine, EngineConfig, EngineError, FaultTreeError,
    FaultTreeSettings, PreparedCoSimulation,
};
use raichu_expr::{Expr, Value};
use raichu_model::Model;
use serde_json::json;

fn base_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../raichu-fmi/tests/fixtures/reference-fmus/3.0")
}

fn model(variable: &str, with_transition: bool) -> CompiledModel {
    let transition = if with_transition {
        vec![json!({
            "name": "switch", "source": "off", "targets": ["on"],
            "distrib": "delay", "time": 0.35,
            "effects": [{
                "target": {"component": "plant", "attribute": "input"},
                "value": {"op": "const", "value": {"kind": "float", "value": 1.0}}
            }]
        })]
    } else {
        Vec::new()
    };
    let document = json!({
        "name": "fmi_feedthrough",
        "components": [{
            "name": "plant",
            "attributes": [
                {"name": "input", "kind": "float", "init": {"kind": "float", "value": 0.0}},
                {"name": "output", "kind": "float", "init": {"kind": "float", "value": 0.0}}
            ],
            "automata": [{"name": "control", "states": ["off", "on"], "init": "off", "transitions": transition}]
        }],
        "fmu_units": [{
            "name": "feed", "path": "Feedthrough.fmu", "step": 0.1,
            "inputs": [{"attribute": {"component": "plant", "attribute": "input"}, "variable": "Float64_continuous_input"}],
            "outputs": [{"attribute": {"component": "plant", "attribute": "output"}, "variable": variable}]
        }]
    });
    let model: Model = serde_json::from_value(document).unwrap();
    CompiledModel::compile(&model).unwrap()
}

#[test]
fn prepared_archive_can_spawn_independent_hosts_after_source_removal() {
    let directory = std::env::temp_dir().join(format!(
        "raichu-prepared-fmu-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    let source = directory.join("Feedthrough.fmu");
    std::fs::copy(base_dir().join("Feedthrough.fmu"), &source).unwrap();
    let compiled = model("Float64_continuous_output", false);
    let prepared = PreparedCoSimulation::prepare_authorized(&compiled, &directory, true).unwrap();
    std::fs::remove_file(&source).unwrap();

    let config = EngineConfig {
        t_max: 0.2,
        allow_fmu_import: true,
        ..EngineConfig::default()
    };
    for _ in 0..2 {
        let mut host = prepared.spawn_host();
        let result = Engine::new_with_host(&compiled, config.clone(), &mut host)
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(result.final_time, 0.2);
    }
    std::fs::remove_dir(&directory).unwrap();
}

#[test]
fn permission_precedes_unpack_and_binding_resolution() {
    let compiled = model("Float64_continuous_output", false);
    let missing = PathBuf::from("/definitely/missing/fmu/path");
    let error = CoSimulationHost::prepare_authorized(&compiled, &missing, false)
        .err()
        .unwrap();
    assert!(matches!(error, EngineError::FmuPermission { unit } if unit == "feed"));
    let error = Engine::new(&compiled, EngineConfig::default())
        .err()
        .unwrap();
    assert!(matches!(error, EngineError::FmuPermission { unit } if unit == "feed"));
}

#[test]
fn fmu_run_needs_a_finite_horizon() {
    let compiled = model("Float64_continuous_output", false);
    let mut host = CoSimulationHost::prepare_authorized(&compiled, &base_dir(), true).unwrap();
    let config = EngineConfig {
        allow_fmu_import: true,
        ..EngineConfig::default()
    };
    let error = Engine::new_with_host(&compiled, config, &mut host)
        .err()
        .unwrap();
    assert!(
        matches!(error, EngineError::InvalidStudyParameter { parameter, .. } if parameter == "t_max")
    );
}

#[test]
fn binding_refuses_unknown_variable_before_library_load() {
    let compiled = model("Float64_continuos_output", false);
    let error = CoSimulationHost::prepare_authorized(&compiled, &base_dir(), true)
        .err()
        .unwrap();
    assert!(
        matches!(error, EngineError::FmuBinding { unit, attribute, variable, .. }
        if unit == "feed" && attribute == "plant.output" && variable == "Float64_continuos_output")
    );
}

#[test]
fn feedthrough_tunable_parameter_can_bind_a_model_input() {
    let mut compiled = model("Float64_continuous_output", false);
    compiled.fmu_units[0].inputs[0].variable = "Float64_tunable_parameter".to_owned();
    let mut host = CoSimulationHost::prepare_authorized(&compiled, &base_dir(), true).unwrap();
    let config = EngineConfig {
        t_max: 0.1,
        allow_fmu_import: true,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new_with_host(&compiled, config, &mut host).unwrap();
    assert_eq!(engine.step().unwrap().unwrap().time, 0.1);
}

#[test]
fn feedthrough_grid_is_separate_from_authored_transitions() {
    let compiled = model("Float64_continuous_output", true);
    assert_eq!(compiled.transitions.len(), 1);
    let mut host = CoSimulationHost::prepare_authorized(&compiled, &base_dir(), true).unwrap();
    let config = EngineConfig {
        t_max: 0.6,
        allow_fmu_import: true,
        rate_factors: vec![1.0],
        ..EngineConfig::default()
    };
    let mut engine = Engine::new_with_host(&compiled, config, &mut host).unwrap();
    let mut points = Vec::new();
    while let Some(event) = engine.step().unwrap() {
        if event.transition == "fmi.communication" {
            points.push((event.time, engine.attribute("plant.output").unwrap()));
        }
    }
    assert!(points
        .iter()
        .any(|(time, output)| (*time - 0.3).abs() < 1e-12 && *output == Value::Float(0.0)));
    assert!(points
        .iter()
        .any(|(time, output)| (*time - 0.4).abs() < 1e-12 && *output == Value::Float(1.0)));
    assert_eq!(compiled.transitions.len(), 1);
}

#[test]
fn fixed_step_feedthrough_holds_an_input_changed_between_points() {
    let directory = tempfile::tempdir().unwrap();
    let source = std::fs::File::open(base_dir().join("Feedthrough.fmu")).unwrap();
    let mut source = zip::ZipArchive::new(source).unwrap();
    let target = std::fs::File::create(directory.path().join("Feedthrough.fmu")).unwrap();
    let mut writer = zip::ZipWriter::new(target);
    for index in 0..source.len() {
        let mut entry = source.by_index(index).unwrap();
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().to_owned();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap();
        if name == "modelDescription.xml" {
            let xml = String::from_utf8(bytes).unwrap();
            assert!(xml.contains("canHandleVariableCommunicationStepSize=\"true\""));
            bytes = xml
                .replace(
                    "canHandleVariableCommunicationStepSize=\"true\"",
                    "canHandleVariableCommunicationStepSize=\"false\"",
                )
                .into_bytes();
        }
        writer
            .start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(&bytes).unwrap();
    }
    writer.finish().unwrap();

    let compiled = model("Float64_continuous_output", true);
    let mut host = CoSimulationHost::prepare_authorized(&compiled, directory.path(), true).unwrap();
    let config = EngineConfig {
        t_max: 0.5,
        allow_fmu_import: true,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new_with_host(&compiled, config, &mut host).unwrap();
    let mut points = Vec::new();
    while let Some(event) = engine.step().unwrap() {
        if event.transition == "fmi.communication" {
            points.push((event.time, engine.attribute("plant.output").unwrap()));
        }
    }
    assert!(points
        .iter()
        .any(|(time, output)| (*time - 0.4).abs() < 1e-12 && *output == Value::Float(0.0)));
    assert!(points
        .iter()
        .any(|(time, output)| (*time - 0.5).abs() < 1e-12 && *output == Value::Float(1.0)));
    assert_eq!(points.len(), 5);
}

#[test]
fn native_state_and_grid_replay_from_snapshot() {
    let compiled = model("Float64_continuous_output", false);
    let mut host = CoSimulationHost::prepare_authorized(&compiled, &base_dir(), true).unwrap();
    host.require_serializable_state().unwrap();
    let config = EngineConfig {
        t_max: 0.5,
        allow_fmu_import: true,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new_with_host(&compiled, config.clone(), &mut host).unwrap();
    assert_eq!(engine.step().unwrap().unwrap().time, 0.1);
    let snapshot = engine.try_snapshot().unwrap();
    let first = engine.step().unwrap().unwrap();
    let first_value = engine.attribute("plant.output");
    drop(engine);
    let mut restored =
        Engine::from_snapshot_with_host(&compiled, config, &snapshot, &mut host).unwrap();
    let replay = restored.step().unwrap().unwrap();
    assert_eq!(first.time, replay.time);
    assert_eq!(first_value, restored.attribute("plant.output"));
}

#[test]
fn long_grid_uses_integer_index_and_never_counts_as_transition_firing() {
    let compiled = model("Float64_continuous_output", false);
    let mut host = CoSimulationHost::prepare_authorized(&compiled, &base_dir(), true).unwrap();
    let config = EngineConfig {
        t_max: 100.0,
        allow_fmu_import: true,
        max_transition_firings: 2,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new_with_host(&compiled, config, &mut host).unwrap();
    for index in 1..=1000 {
        let point = engine.step().unwrap().unwrap();
        assert_eq!(point.time, f64::from(index) * 0.1);
    }
}

#[test]
fn structural_fault_tree_refuses_an_opaque_unit() {
    let compiled = model("Float64_continuous_output", false);
    let top = Expr::Const {
        value: Value::Bool(true),
    };
    let error = fault_tree(&compiled, &top, &FaultTreeSettings::default())
        .err()
        .unwrap();
    assert!(matches!(error, FaultTreeError::OpaqueFmu { unit } if unit == "feed"));
}

#[test]
fn authored_event_at_grid_point_fires_before_the_fmu_step() {
    let mut compiled = model("Float64_continuous_output", true);
    compiled.transitions[0].distrib = CLaw::Delay(0.4);
    let mut host = CoSimulationHost::prepare_authorized(&compiled, &base_dir(), true).unwrap();
    let config = EngineConfig {
        t_max: 0.5,
        allow_fmu_import: true,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new_with_host(&compiled, config, &mut host).unwrap();
    let mut points = Vec::new();
    while let Some(event) = engine.step().unwrap() {
        if event.transition == "fmi.communication" {
            points.push((event.time, engine.attribute("plant.output").unwrap()));
        }
    }
    assert!(points
        .iter()
        .any(|(time, value)| (*time - 0.4).abs() < 1e-12 && *value == Value::Float(0.0)));
    assert!(points
        .iter()
        .any(|(time, value)| (*time - 0.5).abs() < 1e-12 && *value == Value::Float(1.0)));
}

#[test]
fn dahlquist_initial_output_and_first_step_reach_the_model() {
    let document = json!({
        "name": "dahlquist",
        "components": [{"name": "plant", "attributes": [
            {"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}
        ]}],
        "fmu_units": [{
            "name": "ode", "path": "Dahlquist.fmu", "step": 0.1,
            "outputs": [{"attribute": {"component": "plant", "attribute": "x"}, "variable": "x"}]
        }]
    });
    let model: Model = serde_json::from_value(document).unwrap();
    let compiled = CompiledModel::compile(&model).unwrap();
    let mut host = CoSimulationHost::prepare_authorized(&compiled, &base_dir(), true).unwrap();
    let config = EngineConfig {
        t_max: 0.1,
        allow_fmu_import: true,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new_with_host(&compiled, config, &mut host).unwrap();
    assert_eq!(engine.attribute("plant.x"), Some(Value::Float(1.0)));
    engine.step().unwrap().unwrap();
    assert!(
        matches!(engine.attribute("plant.x"), Some(Value::Float(value)) if value > 0.8 && value < 1.0),
        "{:?}",
        engine.attribute("plant.x")
    );
}

#[test]
fn watched_reaction_to_fmu_output_precedes_next_input_sample() {
    let document = json!({
        "name": "fmi_watched_input",
        "components": [{
            "name": "plant",
            "attributes": [
                {"name": "x", "kind": "float", "init": {"kind": "float", "value": 1.0}},
                {"name": "input", "kind": "float", "init": {"kind": "float", "value": 0.0}},
                {"name": "output", "kind": "float", "init": {"kind": "float", "value": 0.0}}
            ],
            "automata": [{
                "name": "watch", "states": ["off", "on"], "init": "off",
                "transitions": [{
                    "name": "switch", "source": "off", "targets": ["on"],
                    "distrib": "watched",
                    "guard": {
                        "op": "cmp", "cmp": "le",
                        "lhs": {"op": "attr", "attr": {"component": "plant", "attribute": "x"}},
                        "rhs": {"op": "const", "value": {"kind": "float", "value": 0.95}}
                    },
                    "effects": [{
                        "target": {"component": "plant", "attribute": "input"},
                        "value": {"op": "const", "value": {"kind": "float", "value": 1.0}}
                    }]
                }]
            }]
        }],
        "fmu_units": [
            {
                "name": "physics", "path": "Dahlquist.fmu", "step": 0.1,
                "outputs": [{"attribute": {"component": "plant", "attribute": "x"}, "variable": "x"}]
            },
            {
                "name": "feed", "path": "Feedthrough.fmu", "step": 0.1,
                "inputs": [{"attribute": {"component": "plant", "attribute": "input"}, "variable": "Float64_continuous_input"}],
                "outputs": [{"attribute": {"component": "plant", "attribute": "output"}, "variable": "Float64_continuous_output"}]
            }
        ]
    });
    let model: Model = serde_json::from_value(document).unwrap();
    let compiled = CompiledModel::compile(&model).unwrap();
    let mut host = CoSimulationHost::prepare_authorized(&compiled, &base_dir(), true).unwrap();
    let config = EngineConfig {
        t_max: 0.2,
        allow_fmu_import: true,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new_with_host(&compiled, config, &mut host).unwrap();
    assert_eq!(engine.step().unwrap().unwrap().time, 0.1);
    assert_eq!(engine.history().len(), 1);
    assert_eq!(engine.history()[0].transition, "plant.watch.switch");
    assert_eq!(engine.attribute("plant.input"), Some(Value::Float(1.0)));
    assert_eq!(engine.step().unwrap().unwrap().time, 0.2);
    assert_eq!(engine.attribute("plant.output"), Some(Value::Float(1.0)));
}

#[test]
fn fmu_output_change_splits_another_units_variable_step() {
    let document = json!({
        "name": "fmi_chained_steps",
        "components": [{
            "name": "plant",
            "attributes": [
                {"name": "source", "kind": "float", "init": {"kind": "float", "value": 0.0}},
                {"name": "middle", "kind": "float", "init": {"kind": "float", "value": 0.0}},
                {"name": "output", "kind": "float", "init": {"kind": "float", "value": 0.0}}
            ],
            "automata": [{
                "name": "control", "states": ["off", "on"], "init": "off",
                "transitions": [{
                    "name": "switch", "source": "off", "targets": ["on"],
                    "distrib": "delay", "time": 0.05,
                    "effects": [{
                        "target": {"component": "plant", "attribute": "source"},
                        "value": {"op": "const", "value": {"kind": "float", "value": 1.0}}
                    }]
                }]
            }]
        }],
        "fmu_units": [
            {
                "name": "first", "path": "Feedthrough.fmu", "step": 0.1,
                "inputs": [{"attribute": {"component": "plant", "attribute": "source"}, "variable": "Float64_continuous_input"}],
                "outputs": [{"attribute": {"component": "plant", "attribute": "middle"}, "variable": "Float64_continuous_output"}]
            },
            {
                "name": "second", "path": "Feedthrough.fmu", "step": 0.2,
                "inputs": [{"attribute": {"component": "plant", "attribute": "middle"}, "variable": "Float64_continuous_input"}],
                "outputs": [{"attribute": {"component": "plant", "attribute": "output"}, "variable": "Float64_continuous_output"}]
            }
        ]
    });
    let model: Model = serde_json::from_value(document).unwrap();
    let compiled = CompiledModel::compile(&model).unwrap();
    let mut host = CoSimulationHost::prepare_authorized(&compiled, &base_dir(), true).unwrap();
    let config = EngineConfig {
        t_max: 0.2,
        allow_fmu_import: true,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new_with_host(&compiled, config, &mut host).unwrap();
    while let Some(event) = engine.step().unwrap() {
        if event.time == 0.1 {
            assert_eq!(engine.attribute("plant.middle"), Some(Value::Float(1.0)));
            assert_eq!(engine.attribute("plant.output"), Some(Value::Float(0.0)));
        }
    }
    assert_eq!(engine.attribute("plant.output"), Some(Value::Float(1.0)));
}

#[test]
fn chained_feedthrough_trajectory_is_independent_of_unit_declaration_order() {
    fn trajectory(reverse: bool) -> Vec<(f64, Value, Value)> {
        let mut document = json!({
            "name": "fmi_jacobi_order",
            "components": [{
                "name": "plant",
                "attributes": [
                    {"name": "source", "kind": "float", "init": {"kind": "float", "value": 0.0}},
                    {"name": "middle", "kind": "float", "init": {"kind": "float", "value": 0.0}},
                    {"name": "output", "kind": "float", "init": {"kind": "float", "value": 0.0}}
                ],
                "automata": [{
                    "name": "control", "states": ["off", "on"], "init": "off",
                    "transitions": [{
                        "name": "switch", "source": "off", "targets": ["on"],
                        "distrib": "delay", "time": 0.15,
                        "effects": [{
                            "target": {"component": "plant", "attribute": "source"},
                            "value": {"op": "const", "value": {"kind": "float", "value": 1.0}}
                        }]
                    }]
                }]
            }],
            "fmu_units": [
                {
                    "name": "first", "path": "Feedthrough.fmu", "step": 0.1,
                    "inputs": [{"attribute": {"component": "plant", "attribute": "source"}, "variable": "Float64_continuous_input"}],
                    "outputs": [{"attribute": {"component": "plant", "attribute": "middle"}, "variable": "Float64_continuous_output"}]
                },
                {
                    "name": "second", "path": "Feedthrough.fmu", "step": 0.1,
                    "inputs": [{"attribute": {"component": "plant", "attribute": "middle"}, "variable": "Float64_continuous_input"}],
                    "outputs": [{"attribute": {"component": "plant", "attribute": "output"}, "variable": "Float64_continuous_output"}]
                }
            ]
        });
        if reverse {
            document["fmu_units"].as_array_mut().unwrap().reverse();
        }
        let model: Model = serde_json::from_value(document).unwrap();
        let compiled = CompiledModel::compile(&model).unwrap();
        let mut host = CoSimulationHost::prepare_authorized(&compiled, &base_dir(), true).unwrap();
        let config = EngineConfig {
            t_max: 0.4,
            allow_fmu_import: true,
            ..EngineConfig::default()
        };
        let mut engine = Engine::new_with_host(&compiled, config, &mut host).unwrap();
        let mut points = Vec::new();
        while let Some(event) = engine.step().unwrap() {
            if event.transition == "fmi.communication" {
                points.push((
                    event.time,
                    engine.attribute("plant.middle").unwrap(),
                    engine.attribute("plant.output").unwrap(),
                ));
            }
        }
        points
    }

    let forward = trajectory(false);
    let backward = trajectory(true);
    assert_eq!(forward, backward);
    assert!(forward.iter().any(|(time, middle, output)| {
        (*time - 0.2).abs() < 1e-12 && *middle == Value::Float(1.0) && *output == Value::Float(0.0)
    }));
    assert!(forward
        .iter()
        .any(|(time, _, output)| { (*time - 0.3).abs() < 1e-12 && *output == Value::Float(1.0) }));
}
