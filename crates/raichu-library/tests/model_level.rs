//! Model-level constructs declared in a system document: interface
//! connections, mixed-integer programs and FMU units.
//!
//! Each binds several instances, so a system declares them. Their run-time
//! behaviour is compared with the flat models in the Python unit suite
//! (`test_library_model_level.py`); this file pins the expansion itself.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_library::{expand, parse_library, parse_system};
use serde_json::{json, Value};

fn library() -> raichu_library::Library {
    parse_library(
        &json!({"raichu_library": {"format": 1}, "library": {
        "name": "bus", "version": "1.0.0",
        "types": [{"name": "Node", "body": {
            "attributes": [{"name": "x", "kind": "float", "init": {"kind": "float", "value": 1.0}}],
            "ports": [{"name": "p", "dir": "out", "attr": "x"}, {"name": "q", "dir": "in"}],
            "interfaces": [{"name": "link", "ports": ["p", "q"]}]
        }}, {"name": "Mirror", "body": {
            "attributes": [{"name": "x", "kind": "float", "init": {"kind": "float", "value": 2.0}}],
            "ports": [{"name": "q", "dir": "out", "attr": "x"}, {"name": "p", "dir": "in"}],
            "interfaces": [{"name": "link", "ports": ["q", "p"]}]
        }}]}})
        .to_string(),
    )
    .unwrap()
}

fn system(extra: Value) -> raichu_library::SystemModel {
    let mut body = json!({
        "name": "pair", "library": {"name": "bus", "version": "1.0.0"},
        "instances": [{"name": "A", "type": "Node"}, {"name": "B", "type": "Mirror"}],
    });
    for (key, value) in extra.as_object().unwrap() {
        body[key] = value.clone();
    }
    parse_system(&json!({"raichu_system": {"format": 1}, "system": body}).to_string()).unwrap()
}

#[test]
fn an_interface_connection_joins_the_two_instances_by_interface() {
    let system = system(json!({"interface_connections": [
        {"from": "A.link", "to": "B.link", "name": "wire"}]}));
    let model = expand(&[library()], &system, None).unwrap().model;
    let written = serde_json::to_value(&model.interface_connections).unwrap();
    assert_eq!(
        written,
        json!([{"name": "wire",
                "from": {"component": "A", "interface": "link"},
                "to": {"component": "B", "interface": "link"}}])
    );
    assert_eq!(model.resolved_connections().unwrap().len(), 2);
}

#[test]
fn an_interface_the_instance_does_not_declare_is_refused_with_the_ones_it_does() {
    let system = system(json!({"interface_connections": [{"from": "A.bus", "to": "B.link"}]}));
    let error = expand(&[library()], &system, None).unwrap_err().to_string();
    assert!(
        error.contains("/system/interface_connections/0/from"),
        "{error}"
    );
    assert!(
        error.contains("no interface `bus`") && error.contains("`link`"),
        "{error}"
    );
}

fn pick(feasible: &str) -> Value {
    json!({
        "name": "pick", "sense": "maximize",
        "variables": [{"attribute": {"component": "A", "attribute": "x"},
                       "lower": {"op": "const", "value": {"kind": "float", "value": 0.0}},
                       "upper": {"op": "const", "value": {"kind": "float", "value": 3.0}},
                       "on_infeasible": {"kind": "float", "value": 0.0}}],
        "objective": {"op": "attr", "attr": {"component": "A", "attribute": "x"}},
        "constraints": [],
        "feasible": {"component": "B", "attribute": feasible}
    })
}

#[test]
fn a_program_is_checked_by_the_core_once_expanded() {
    // The program's `feasible` attribute does not exist: the expander
    // carries the program through, and the core refuses it by name, traced
    // back to the system node that declared it.
    let system = system(json!({"programs": [pick("missing")]}));
    let error = expand(&[library()], &system, None).unwrap_err().to_string();
    assert!(error.contains("program `pick`"), "{error}");
    assert!(
        error.contains("`B.missing`") || error.contains("missing"),
        "{error}"
    );
    assert!(error.contains("/system/programs/0"), "{error}");
}

/// An FMU unit named after the instance it drives (a natural choice) is
/// still traced back to the system node, not to that instance's type.
#[test]
fn an_fmu_unit_named_after_an_instance_is_traced_to_the_system() {
    let unit = json!({"name": "A", "path": "plant.fmu", "step": 0.5,
        "outputs": [{"attribute": {"component": "A", "attribute": "nowhere"}, "variable": "y"}]});
    let system = system(json!({"fmu_units": [unit]}));
    let error = expand(&[library()], &system, None).unwrap_err().to_string();
    assert!(error.contains("/system/fmu_units/0"), "{error}");
}

#[test]
fn a_program_named_after_an_instance_is_traced_to_the_system() {
    let mut program = pick("missing");
    program["name"] = json!("B");
    let system = system(json!({"programs": [program]}));
    let error = expand(&[library()], &system, None).unwrap_err().to_string();
    assert!(error.contains("/system/programs/0"), "{error}");
}

#[test]
fn an_interface_connection_to_an_unknown_instance_is_refused_with_its_name() {
    let system = system(json!({"interface_connections": [{"from": "Ghost.link", "to": "B.link"}]}));
    let error = expand(&[library()], &system, None).unwrap_err().to_string();
    assert!(
        error.contains("/system/interface_connections/0/from"),
        "{error}"
    );
    assert!(error.contains("no instance `Ghost`"), "{error}");
}

#[test]
fn an_unknown_key_in_a_complete_program_is_refused_at_its_pointer() {
    let mut program = pick("x");
    program["costs"] = json!([]);
    let system = system(json!({"programs": [program]}));
    let error = expand(&[library()], &system, None).unwrap_err().to_string();
    assert!(error.contains("/system/programs/0"), "{error}");
    assert!(error.contains("costs"), "{error}");
}

#[test]
fn an_unknown_key_in_an_fmu_unit_is_refused_at_its_pointer() {
    let unit = json!({"name": "plant", "path": "plant.fmu", "step": 0.5, "stride": 2});
    let system = system(json!({"fmu_units": [unit]}));
    let error = expand(&[library()], &system, None).unwrap_err().to_string();
    assert!(error.contains("/system/fmu_units/0"), "{error}");
    assert!(error.contains("stride"), "{error}");
}
