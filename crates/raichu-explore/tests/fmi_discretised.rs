//! Discretised exploration of an imported unit and pre-load capability gates.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use raichu_core::{CompiledModel, EngineError};
use raichu_explore::discretised::explore_discretised_with_fmu;
use raichu_explore::{explore_discretised, DiscretisedSettings};
use raichu_model::Model;
use serde_json::json;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../raichu-fmi/tests/fixtures/reference-fmus/3.0")
}

fn document(path: &str) -> serde_json::Value {
    json!({
        "name": "fmi_exploration",
        "components": [
            {
                "name": "plant",
                "attributes": [{"name": "x", "kind": "float", "init": {"kind": "float", "value": 1.0}}]
            },
            {
                "name": "sys",
                "automata": [{
                    "name": "watch", "states": ["ok", "down"], "init": "ok",
                    "transitions": [{
                        "name": "down", "source": "ok", "targets": ["down"],
                        "distrib": "inst", "probs": [],
                        "guard": {
                            "op": "cmp", "cmp": "le",
                            "lhs": {"op": "attr", "attr": {"component": "plant", "attribute": "x"}},
                            "rhs": {"op": "const", "value": {"kind": "float", "value": 0.7}}
                        }
                    }]
                }]
            }
        ],
        "targets": [{"name": "down", "component": "sys", "automaton": "watch", "state": "down"}],
        "fmu_units": [{
            "name": "physics", "path": path, "step": 0.1,
            "outputs": [{"attribute": {"component": "plant", "attribute": "x"}, "variable": "x"}]
        }]
    })
}

fn model(path: &str) -> CompiledModel {
    let document = document(path);
    let model: Model = serde_json::from_value(document).unwrap();
    CompiledModel::compile(&model).unwrap()
}

fn stochastic_model() -> CompiledModel {
    let mut document = document("Dahlquist.fmu");
    document["components"][1]["automata"][0]["transitions"]
        .as_array_mut()
        .unwrap()
        .insert(
            0,
            json!({
                "name": "failure", "source": "ok", "targets": ["failed"],
                "distrib": "exp", "rate": 1.0
            }),
        );
    document["components"][1]["automata"][0]["states"] = json!(["ok", "failed", "down"]);
    document["components"][1]["automata"][0]["transitions"][1]["source"] = json!("failed");
    let model: Model = serde_json::from_value(document).unwrap();
    CompiledModel::compile(&model).unwrap()
}

fn settings() -> DiscretisedSettings {
    let mut settings = DiscretisedSettings::new("down", 1.0);
    settings.level = 2;
    settings.refine = false;
    settings
}

#[test]
fn a_run_side_grant_is_required_before_unpacking() {
    let compiled = model("Dahlquist.fmu");
    let missing = Path::new("/definitely/missing/fmu/directory");
    let error = explore_discretised(&compiled, &settings()).unwrap_err();
    assert!(matches!(error, EngineError::FmuPermission { unit } if unit == "physics"));
    let error = explore_discretised_with_fmu(&compiled, &settings(), missing, false).unwrap_err();
    assert!(matches!(error, EngineError::FmuPermission { unit } if unit == "physics"));
}

#[test]
fn a_serializable_reference_fmu_reaches_a_watched_target() {
    let compiled = model("Dahlquist.fmu");
    let result =
        explore_discretised_with_fmu(&compiled, &settings(), &fixture_dir(), true).unwrap();
    assert_eq!(result.lower, 1.0);
    assert_eq!(result.upper, 1.0);
    assert_eq!(result.sequences.len(), 1);
    assert_eq!(
        result.sequence_steps(0).unwrap().next().unwrap().transition,
        "sys.watch.down"
    );
}

#[test]
fn stochastic_branches_restore_native_state_independently_of_thread_count() {
    let compiled = stochastic_model();
    let mut one = settings();
    one.cutoffs.max_branches = Some(10_000);
    one.threads = Some(1);
    let mut two = one.clone();
    two.threads = Some(2);
    let base = fixture_dir();
    let first = explore_discretised_with_fmu(&compiled, &one, &base, true).unwrap();
    let second = explore_discretised_with_fmu(&compiled, &two, &base, true).unwrap();
    assert_eq!(first, second);
    assert!(first.lower > 0.0 && first.lower < 1.0);
}

#[test]
fn a_unit_without_state_serialization_is_refused_before_library_load() {
    let source = std::fs::File::open(fixture_dir().join("Dahlquist.fmu")).unwrap();
    let mut source = zip::ZipArchive::new(source).unwrap();
    let mut xml = String::new();
    source
        .by_name("modelDescription.xml")
        .unwrap()
        .read_to_string(&mut xml)
        .unwrap();
    let xml = xml
        .replace(
            "canGetAndSetFMUState=\"true\"",
            "canGetAndSetFMUState=\"false\"",
        )
        .replace(
            "canSerializeFMUState=\"true\"",
            "canSerializeFMUState=\"false\"",
        );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("no-state.fmu");
    let output = std::fs::File::create(&path).unwrap();
    let mut writer = zip::ZipWriter::new(output);
    writer
        .start_file(
            "modelDescription.xml",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
    writer.write_all(xml.as_bytes()).unwrap();
    writer.finish().unwrap();
    let compiled = model("no-state.fmu");
    let error =
        explore_discretised_with_fmu(&compiled, &settings(), directory.path(), true).unwrap_err();
    assert!(
        matches!(error, EngineError::FmuCapability { unit, capability }
        if unit == "physics" && capability == "serialized FMU state get/set")
    );
}
