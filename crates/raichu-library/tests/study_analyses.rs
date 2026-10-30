//! The analyses a study requests and the options of its simulation.
//!
//! Their parameters are the Python entry points' own, checked by
//! `pyraichu` when it loads the study (`test_library_study_analyses.py`);
//! this file pins what the expander owns: the closed list of analyses and
//! the sections reaching the expansion as written.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_library::parse_study;
use serde_json::json;

fn study(body: serde_json::Value) -> Result<raichu_library::StudyDocument, String> {
    let mut document = json!({"name": "s", "system": "pair"});
    for (key, value) in body.as_object().unwrap() {
        document[key] = value.clone();
    }
    parse_study(&json!({"raichu_study": {"format": 1}, "study": document}).to_string())
        .map_err(|e| e.to_string())
}

#[test]
fn the_three_analyses_are_read_as_written() {
    let analyses = json!({
        "fault_tree": {"targets": ["down"]},
        "importance": {"nb_runs": 10, "t_max": 5.0, "instants": [5.0]},
        "sequence_analysis": {"nb_runs": 10, "t_max": 5.0}});
    let read = study(json!({"analyses": analyses.clone()})).unwrap();
    assert_eq!(
        serde_json::to_value(read.analyses.unwrap()).unwrap(),
        analyses
    );
}

#[test]
fn an_analysis_the_format_does_not_list_is_refused_by_name() {
    let error = study(json!({"analyses": {"fault_trees": {}}})).unwrap_err();
    assert!(error.contains("fault_trees"), "{error}");
}

#[test]
fn simulation_options_are_read_as_written_and_omitted_when_empty() {
    let simulation = json!({"method": "monte_carlo", "nb_runs": 10, "horizon": 5.0,
        "options": {"confidence": 0.9, "flow": {"tolerance": 1e-10}}});
    let read = study(json!({"simulation": simulation.clone()})).unwrap();
    let written = serde_json::to_value(read.simulation.unwrap()).unwrap();
    assert_eq!(written["options"], simulation["options"]);
    let bare =
        study(json!({"simulation": {"method": "monte_carlo", "nb_runs": 10, "horizon": 5.0}}))
            .unwrap();
    assert!(serde_json::to_value(bare.simulation.unwrap())
        .unwrap()
        .get("options")
        .is_none());
}
