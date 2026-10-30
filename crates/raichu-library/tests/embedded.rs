//! The registry of libraries shipped with the engine.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_library::{
    embedded_library, expand, parse_library, parse_system, standard_library, Library, EMBEDDED, STD,
};
use serde_json::{json, Value};

fn library(body: Value) -> Library {
    parse_library(&json!({"raichu_library": {"format": 1}, "library": body}).to_string()).unwrap()
}

/// `plant` defines its source from a std fragment; `wrap` imports only
/// `plant` and reuses its fragment, which applies std in turn.
fn plant() -> Library {
    library(
        json!({"name": "plant", "version": "1.0.0", "imports": ["std"],
        "fragments": [{"name": "power", "body": {}, "uses": [
            {"fragment": "std.bool_source_out", "with": {"name": "power"}}]}],
        "types": [{"name": "Source", "uses": [{"fragment": "power"}]}]}),
    )
}

fn system(library: &str) -> raichu_library::SystemModel {
    parse_system(
        &json!({"raichu_system": {"format": 1}, "system": {
        "name": "s", "library": {"name": library, "version": "1.0.0"},
        "instances": [{"name": "A", "type": "Source"}]}})
        .to_string(),
    )
    .unwrap()
}

#[test]
fn the_registry_holds_std_at_the_engine_version() {
    assert_eq!(EMBEDDED, [STD]);
    let std = embedded_library("std").unwrap().unwrap();
    assert_eq!(std, standard_library().unwrap());
    assert_eq!(std.version, env!("CARGO_PKG_VERSION"));
    assert!(embedded_library("muscadet").is_none());
}

#[test]
fn a_library_importing_std_records_its_version() {
    let model = expand(&[plant()], &system("plant"), None).unwrap().model;
    let source = model.source.unwrap();
    assert_eq!(source.embedded.len(), 1);
    assert_eq!(source.embedded["std"], env!("CARGO_PKG_VERSION"));
}

#[test]
fn std_reached_through_another_library_is_added_and_recorded() {
    let wrap = library(
        json!({"name": "wrap", "version": "1.0.0", "imports": ["plant"],
        "types": [{"name": "Source", "uses": [{"fragment": "plant.power"}]}]}),
    );
    let model = expand(&[wrap, plant()], &system("wrap"), None)
        .unwrap()
        .model;
    assert_eq!(
        model.source.unwrap().embedded["std"],
        env!("CARGO_PKG_VERSION")
    );
}

#[test]
fn a_user_library_named_after_an_embedded_one_is_refused() {
    let fake = library(json!({"name": "std", "version": "9.9.9", "fragments": [], "types": []}));
    let error = expand(&[plant(), fake], &system("plant"), None)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("`std`") && error.contains("version `9.9.9`"),
        "{error}"
    );
    assert!(error.contains("drop it"), "{error}");
}

#[test]
fn the_embedded_library_itself_may_be_supplied() {
    let model = expand(
        &[plant(), standard_library().unwrap()],
        &system("plant"),
        None,
    )
    .unwrap()
    .model;
    assert_eq!(
        model.source.unwrap().embedded["std"],
        env!("CARGO_PKG_VERSION")
    );
}

#[test]
fn an_import_cycle_is_refused_naming_its_libraries() {
    let a = library(
        json!({"name": "a", "version": "1.0.0", "imports": ["b"], "types": [
        {"name": "Source", "body": {}}]}),
    );
    let b = library(json!({"name": "b", "version": "1.0.0", "imports": ["a"], "types": []}));
    let error = expand(&[a, b], &system("a"), None).unwrap_err().to_string();
    assert!(
        error.contains("cycle") && error.contains("a -> b -> a"),
        "{error}"
    );
}
