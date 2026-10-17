//! The `raichu.fault_tree` envelope: one quantification per mission time,
//! cut sets and importance at the horizon, a format and a version a
//! reader checks first.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{BasicLaw, FaultTree, FtNode, GateOp, GenerationWarning, TreeEvent};
use raichu_fta::{
    fault_tree_envelope, read_fault_tree_envelope, EnvelopeTop, EnvelopeWarning, FtaError,
    QuantifySettings, QuantifyWarning, ReadFaultTreeError, FAULT_TREE_FORMAT, FAULT_TREE_VERSION,
    MAX_MISSION_TIMES, MEASURE_WITHOUT_REPAIR,
};

fn event(component: &str, rate: f64) -> TreeEvent {
    TreeEvent {
        name: format!("{component}.health.fail"),
        component: component.to_owned(),
        automaton: format!("{component}.health"),
        transition: "fail".to_owned(),
        target: "nok".to_owned(),
        law: BasicLaw::Exponential { rate },
        repairs: Vec::new(),
    }
}

/// Two blocks in parallel: lost when both have failed.
fn parallel(warnings: Vec<GenerationWarning>) -> FaultTree {
    FaultTree {
        top: FtNode::Gate {
            op: GateOp::And,
            id: None,
            origin: None,
            children: vec![FtNode::Basic { event: 0 }, FtNode::Basic { event: 1 }],
        },
        basic_events: vec![event("B1", 1e-3), event("B2", 2e-3)],
        warnings,
    }
}

fn top() -> EnvelopeTop {
    EnvelopeTop::Targets {
        targets: vec!["T_lost".to_owned()],
    }
}

fn occurred(rate: f64, t: f64) -> f64 {
    1.0 - (-rate * t).exp()
}

#[test]
fn every_mission_time_is_quantified_and_the_horizon_carries_the_detail() {
    let times = [100.0, 500.0, 1000.0];
    let envelope = fault_tree_envelope(
        &parallel(Vec::new()),
        "rbd",
        top(),
        &times,
        &QuantifySettings::default(),
    )
    .expect("envelope");
    assert_eq!(envelope.format, FAULT_TREE_FORMAT);
    assert_eq!(envelope.version, FAULT_TREE_VERSION);
    assert_eq!(envelope.measure, MEASURE_WITHOUT_REPAIR);
    assert_eq!(envelope.provenance.tree, "rbd");
    assert!(envelope.generation.exact);
    assert_eq!(envelope.instants.len(), 3);
    for (instant, &t) in envelope.instants.iter().zip(&times) {
        let exact = occurred(1e-3, t) * occurred(2e-3, t);
        assert_eq!(instant.mission_time, t);
        assert!((instant.probability - exact).abs() < 1e-15, "{instant:?}");
        assert!(instant.exact);
        assert_eq!(instant.method, "bdd");
    }
    let horizon = &envelope.horizon;
    assert_eq!(horizon.mission_time, 1000.0);
    assert_eq!(horizon.probability, envelope.instants[2].probability);
    let cuts = horizon.minimal_cut_sets.as_ref().expect("cut sets");
    assert_eq!(cuts.len(), 1);
    assert_eq!(cuts[0].events, vec!["B1.health.fail", "B2.health.fail"]);
    assert_eq!(cuts[0].order, 2);
    assert!((cuts[0].probability.unwrap() - horizon.probability).abs() < 1e-15);
    assert_eq!(horizon.importance.len(), 2);
    assert_eq!(horizon.importance[0].event, "B1.health.fail");
    assert!(
        (horizon.importance[0].birnbaum - occurred(2e-3, 1000.0)).abs() < 1e-15,
        "{:?}",
        horizon.importance[0]
    );
}

#[test]
fn the_envelope_reads_back_and_names_its_generation_warnings() {
    let warning = GenerationWarning::TargetLeft {
        automaton: "B1.health".to_owned(),
        state: "nok".to_owned(),
        transitions: vec!["B1.health.die".to_owned()],
    };
    let envelope = fault_tree_envelope(
        &parallel(vec![warning.clone()]),
        "rbd",
        top(),
        &[1000.0],
        &QuantifySettings::default(),
    )
    .expect("envelope");
    assert!(!envelope.generation.exact);
    assert_eq!(
        envelope.generation.warnings,
        vec![EnvelopeWarning {
            warning: warning.clone(),
            message: warning.to_string(),
        }]
    );
    let mut envelope = envelope;
    // A value serde_json's default float parser reads one ulp off, and a
    // cut-set count past `u64`.
    envelope.horizon.probability = 0.248_720_059_264_354_14;
    envelope.horizon.cut_set_count = Some(u128::from(u64::MAX) + 1);
    let text = envelope.to_json().expect("json");
    // The document carries the code, its parameters and the readable
    // message beside them.
    let document: serde_json::Value = serde_json::from_str(&text).expect("json");
    assert_eq!(
        document["generation"]["warnings"],
        serde_json::json!([{
            "kind": "target_left",
            "automaton": "B1.health",
            "state": "nok",
            "transitions": ["B1.health.die"],
            "message": warning.to_string(),
        }])
    );
    let read = read_fault_tree_envelope(&text).expect("read");
    assert_eq!(
        read.horizon.probability.to_bits(),
        envelope.horizon.probability.to_bits()
    );
    assert_eq!(read, envelope);
    // Every other float too: equal bits write equal text.
    assert_eq!(read.to_json().expect("json"), text);
}

#[test]
fn another_format_or_a_later_version_is_refused_first() {
    let envelope = fault_tree_envelope(
        &parallel(Vec::new()),
        "rbd",
        top(),
        &[1000.0],
        &QuantifySettings::default(),
    )
    .expect("envelope");
    let mut document: serde_json::Value =
        serde_json::from_str(&envelope.to_json().unwrap()).unwrap();
    document["version"] = serde_json::json!(FAULT_TREE_VERSION + 1);
    assert_eq!(
        read_fault_tree_envelope(&document.to_string()).unwrap_err(),
        ReadFaultTreeError::Version(Some(FAULT_TREE_VERSION + 1))
    );
    // A version past `u64` is refused by name too, not as malformed JSON.
    let text = document.to_string().replace(
        &format!("\"version\":{}", FAULT_TREE_VERSION + 1),
        "\"version\":18446744073709551616",
    );
    assert_eq!(
        read_fault_tree_envelope(&text).unwrap_err(),
        ReadFaultTreeError::Version(None)
    );
    let error = read_fault_tree_envelope(r#"{"format": "raichu.quantification", "version": 1}"#)
        .unwrap_err();
    assert!(
        matches!(error, ReadFaultTreeError::Format(Some(_))),
        "{error}"
    );
}

#[test]
fn a_bad_list_of_mission_times_is_refused() {
    let tree = parallel(Vec::new());
    let settings = QuantifySettings::default();
    let too_many: Vec<f64> = (1..=MAX_MISSION_TIMES + 1).map(|i| i as f64).collect();
    for times in [vec![], too_many, vec![10.0, 10.0], vec![10.0, 5.0]] {
        let error = fault_tree_envelope(&tree, "rbd", top(), &times, &settings).unwrap_err();
        assert!(matches!(error, FtaError::Invalid(_)), "{times:?}: {error}");
    }
    let error = fault_tree_envelope(&tree, "rbd", top(), &[f64::NAN], &settings).unwrap_err();
    assert!(matches!(error, FtaError::BadMissionTime(_)), "{error}");
}

/// An envelope as the previous release wrote it, version 1: warnings are
/// plain sentences, no gate carries a probability.
const VERSION_1: &str = r#"{
  "format": "raichu.fault_tree",
  "version": 1,
  "provenance": {"engine_version": "0.99.0", "tree": "rbd"},
  "top": {"kind": "targets", "targets": ["T_lost"]},
  "measure": "probability_without_repair",
  "generation": {
    "exact": false,
    "warnings": ["`B1.health.nok` is left for another state by `B1.health.die`; the tree ignores it, so it over-estimates"],
    "basic_events": [
      {"name": "B1.health.fail", "component": "B1", "automaton": "B1.health",
       "transition": "fail", "target": "nok", "law": {"law": "exponential", "rate": 0.001}},
      {"name": "B2.health.fail", "component": "B2", "automaton": "B2.health",
       "transition": "fail", "target": "nok", "law": {"law": "exponential", "rate": 0.002}}
    ]
  },
  "settings": {"max_bdd_nodes": 10000000, "cut_set_limit": 100000, "cut_sets": true,
               "engine": "auto", "max_order": null, "min_cut_probability": 0.0,
               "max_cut_sets": 1000000, "max_expansions": 100000000},
  "instants": [
    {"mission_time": 1000.0, "probability": 0.5465751103, "method": "cut_sets",
     "exact": false, "upper_bound": 0.55,
     "warnings": ["module M0: its exact diagram outgrew max_bdd_nodes (2): quantified from its cut sets under cutoffs"]}
  ],
  "horizon": {
    "mission_time": 1000.0, "probability": 0.5465751103, "method": "cut_sets",
    "exact": false, "upper_bound": 0.55, "coherent": true, "cut_set_count": 1,
    "cut_sets_complete": true, "cut_sets_omitted": null,
    "minimal_cut_sets": [{"events": ["B1.health.fail", "B2.health.fail"], "order": 2,
                          "probability": 0.5465751103}],
    "importance": [],
    "warnings": ["module M0: its exact diagram outgrew max_bdd_nodes (2): quantified from its cut sets under cutoffs"],
    "provenance": {"variable_order": "depth_first_left_most", "modules": []}
  }
}"#;

#[test]
fn an_envelope_of_the_previous_release_reads_with_legacy_warnings() {
    let read = read_fault_tree_envelope(VERSION_1).expect("a version-1 envelope reads");
    assert_eq!(read.version, 1);
    let sentence = "`B1.health.nok` is left for another state by `B1.health.die`; the \
                    tree ignores it, so it over-estimates";
    assert_eq!(
        read.generation.warnings,
        vec![EnvelopeWarning {
            warning: GenerationWarning::Legacy {
                message: sentence.to_owned()
            },
            message: sentence.to_owned(),
        }]
    );
    let module = "module M0: its exact diagram outgrew max_bdd_nodes (2): quantified from \
                  its cut sets under cutoffs";
    let legacy = vec![QuantifyWarning::Legacy {
        message: module.to_owned(),
    }];
    assert_eq!(read.instants[0].warnings, legacy);
    assert_eq!(read.horizon.warnings, legacy);
    assert!(read.instants[0].gates.is_empty());
    assert!(read.horizon.gates.is_empty());
    // Written again, a legacy warning is the variant `{kind: legacy,
    // message}`, which reads back to itself.
    let text = read.to_json().unwrap();
    let document: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        document["generation"]["warnings"][0],
        serde_json::json!({"kind": "legacy", "message": sentence})
    );
    assert_eq!(
        document["horizon"]["warnings"][0],
        serde_json::json!({"kind": "legacy", "message": module})
    );
    assert_eq!(read_fault_tree_envelope(&text).unwrap(), read);
}

#[test]
fn a_version_2_envelope_carries_no_plain_sentence_and_no_legacy_warning() {
    let mut document: serde_json::Value = serde_json::from_str(VERSION_1).unwrap();
    document["version"] = serde_json::json!(2);
    let error = read_fault_tree_envelope(&document.to_string()).unwrap_err();
    assert!(
        matches!(&error, ReadFaultTreeError::Json(why) if why.contains("legacy")),
        "{error}"
    );
}

#[test]
fn version_3_is_refused_by_name() {
    let mut document: serde_json::Value = serde_json::from_str(VERSION_1).unwrap();
    document["version"] = serde_json::json!(3);
    let error = read_fault_tree_envelope(&document.to_string()).unwrap_err();
    assert_eq!(error, ReadFaultTreeError::Version(Some(3)));
    assert!(error.to_string().contains("version Some(3)"), "{error}");
    assert_eq!(FAULT_TREE_VERSION, 2);
}
