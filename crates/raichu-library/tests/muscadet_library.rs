//! The embedded `muscadet` library and the `std` names that forward to it.
//!
//! The muscadet-shaped fragments moved from `std` to `muscadet`; `std`
//! keeps the generic ones and, until the next major release, one
//! forwarding fragment per moved name, so a library written against the
//! old names expands to the same model.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_library::{
    expand, muscadet_library, parse_library, parse_study, parse_system, standard_library, Library,
    EMBEDDED,
};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

#[test]
fn muscadet_imports_nothing_and_std_imports_only_muscadet() {
    assert!(muscadet_library().unwrap().imports.is_empty());
    assert_eq!(standard_library().unwrap().imports, ["muscadet"]);
}

/// The registry's own imports form no cycle: no embedded library reaches
/// itself through its imports.
#[test]
fn the_embedded_libraries_import_each_other_without_a_cycle() {
    fn reaches(from: &str, target: &str, seen: &mut BTreeSet<String>) -> bool {
        let library = raichu_library::embedded_library(from).unwrap().unwrap();
        library.imports.iter().any(|import| {
            import == target || (seen.insert(import.clone()) && reaches(import, target, seen))
        })
    }
    for name in EMBEDDED {
        assert!(
            !reaches(name, name, &mut BTreeSet::new()),
            "`{name}` reaches itself through its imports"
        );
    }
}

/// Each moved fragment keeps a `std` name with the same parameters, an
/// empty body, and one application of its `muscadet` counterpart passing
/// every parameter through; `std` holds nothing else but generic
/// fragments, which `muscadet` does not hold.
#[test]
fn every_moved_fragment_keeps_a_forwarding_std_name() {
    let std = standard_library().unwrap();
    let muscadet = muscadet_library().unwrap();
    for moved in &muscadet.fragments {
        let forward = std
            .fragments
            .iter()
            .find(|f| f.name == moved.name)
            .unwrap_or_else(|| panic!("std has no forwarding `{}`", moved.name));
        assert_eq!(forward.parameters, moved.parameters, "{}", moved.name);
        assert_eq!(forward.body, json!({}), "{}", moved.name);
        assert_eq!(forward.uses.len(), 1, "{}", moved.name);
        let application = &forward.uses[0];
        assert_eq!(application.fragment, format!("muscadet.{}", moved.name));
        let passed: Vec<(&String, &Value)> = application.with.iter().collect();
        assert_eq!(passed.len(), moved.parameters.len(), "{}", moved.name);
        for parameter in &moved.parameters {
            assert_eq!(
                application.with[&parameter.name],
                json!({"$param": parameter.name}),
                "{}.{}",
                moved.name,
                parameter.name
            );
        }
    }
    let generic: Vec<&str> = std
        .fragments
        .iter()
        .filter(|f| f.uses.iter().all(|u| !u.fragment.starts_with("muscadet.")))
        .map(|f| f.name.as_str())
        .collect();
    assert_eq!(
        generic,
        [
            "state_out",
            "bool_not",
            "time_profile",
            "aggregate_measure",
            "threshold_compare",
            "threshold_band",
            "republish"
        ]
    );
    for name in generic {
        assert!(muscadet.fragments.iter().all(|f| f.name != name), "{name}");
    }
}

/// The library folders of the published examples and of the unit-test
/// library fixtures.
fn library_folders() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut folders = Vec::new();
    for (dir, nested) in [
        ("examples", "library"),
        ("python/tests/unit/library_fixtures", ""),
    ] {
        for entry in std::fs::read_dir(root.join(dir)).unwrap() {
            let folder = entry.unwrap().path().join(nested);
            if folder.join("library.json").exists() {
                folders.push(folder);
            }
        }
    }
    folders.sort();
    folders
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The library rewritten against the `std` forwarding names: each
/// `muscadet.x` application becomes `std.x`, and the library imports
/// `std` only.
fn through_std(text: &str) -> Library {
    let mut document: Value = serde_json::from_str(&text.replace("\"muscadet.", "\"std.")).unwrap();
    document["library"]["imports"] = json!(["std"]);
    parse_library(&document.to_string()).unwrap()
}

/// The expanded model with its source record set aside: that record
/// names the embedded libraries the model drew on, which is what differs
/// between the two spellings.
fn model_without_source(library: Library, system: &str, study: Option<&str>) -> Value {
    let system = parse_system(system).unwrap();
    let study = study.map(|s| parse_study(s).unwrap());
    let model = expand(&[library], &system, study.as_ref()).unwrap().model;
    let mut value = serde_json::to_value(model).unwrap();
    value.as_object_mut().unwrap().remove("source");
    value
}

/// Every published library importing `muscadet` expands to the same
/// model when written against the `std` forwarding names.
#[test]
fn published_libraries_expand_alike_through_the_forwarding_names() {
    let mut checked = 0;
    for folder in library_folders() {
        let library_path = folder.join("library.json");
        let text = read(&library_path);
        let library = parse_library(&text).unwrap();
        if !library.imports.iter().any(|i| i == "muscadet") {
            continue;
        }
        let study = folder.join("study.json");
        let study = study.exists().then(|| read(&study));
        let mut systems: Vec<PathBuf> = std::fs::read_dir(&folder)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("system") && n.ends_with(".json"))
            })
            .collect();
        systems.sort();
        for system in systems {
            let system = read(&system);
            assert_eq!(
                model_without_source(library.clone(), &system, study.as_deref()),
                model_without_source(through_std(&text), &system, study.as_deref()),
                "{}",
                folder.display()
            );
            checked += 1;
        }
    }
    assert!(checked >= 2, "only {checked} systems checked");
}

/// A continuous plant (a volume filling upstream of a rule always on)
/// written against `muscadet` expands to the same model through the `std`
/// forwarding names: the published libraries are boolean, so this keeps
/// the continuous forwarding under test.
#[test]
fn a_continuous_plant_expands_alike_through_the_forwarding_names() {
    let text = json!({"raichu_library": {"format": 1}, "library": {
    "name": "plant", "version": "1.0.0", "imports": ["muscadet"],
    "types": [
        {"name": "Source", "uses": [
            {"fragment": "muscadet.continuous_flow_out", "with": {"name": "F"}},
            {"fragment": "muscadet.continuous_source", "with": {"name": "F", "rate": 4.0}}]},
        {"name": "Feeder", "uses": [
            {"fragment": "muscadet.continuous_flow_in", "with": {"name": "F"}},
            {"fragment": "muscadet.volume_flow", "with": {"volume": "hopper", "flow": "F",
                "capacity": 10.0, "fill_unbounded": true, "side": "in",
                "empty_automaton": "hopper_bounds"}},
            {"fragment": "muscadet.volume_total", "with": {"volume": "hopper", "flows": ["F"],
                "capacity": 10.0}},
            {"fragment": "muscadet.rule_set", "with": {"name": "burn", "active": "go",
                "idle": "none", "consumes": [], "from_volume": ["F"], "produces": ["H"],
                "always_on": true}},
            {"fragment": "muscadet.rule_consumes", "with": {"rule_set": "burn", "active": "go",
                "flow": "F", "rate": 1.0, "always_on": true, "from_volume": true}},
            {"fragment": "muscadet.rule_produces", "with": {"rule_set": "burn", "active": "go",
                "flow": "H", "rate": 1.0, "always_on": true}},
            {"fragment": "muscadet.continuous_flow_out",
                "with": {"name": "H", "available": "produced"}}]},
        {"name": "Sink", "uses": [
            {"fragment": "muscadet.continuous_flow_in", "with": {"name": "H", "demand": 1.0}},
            {"fragment": "muscadet.continuous_sink", "with": {"name": "H", "demand": 1.0}}]}
    ]}})
    .to_string();
    let system = json!({"raichu_system": {"format": 1}, "system": {
        "name": "hopper", "library": {"name": "plant", "version": "1.0.0"},
        "instances": [{"name": "S", "type": "Source"}, {"name": "F", "type": "Feeder"},
                      {"name": "K", "type": "Sink"}],
        "connections": [{"from": "S.F_out", "to": "F.F_in"}, {"from": "F.H_out", "to": "K.H_in"}]}})
    .to_string();
    let direct = model_without_source(parse_library(&text).unwrap(), &system, None);
    assert!(direct["evaluation_order"].is_array(), "a continuous plant");
    assert_eq!(
        direct,
        model_without_source(through_std(&text), &system, None)
    );
}
