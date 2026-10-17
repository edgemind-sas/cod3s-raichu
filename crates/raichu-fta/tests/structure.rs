//! Structural fault-tree output with optional cut-set extraction,
//! producer format validation and explicit omission.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{BasicLaw, FaultTree, FtNode, GateOp, GenerationWarning, TreeEvent};
use raichu_fta::{
    fault_tree_structure as produce_structure, read_fault_tree_structure, EnvelopeTop,
    FaultTreeStructure, FtaError, StructureSettings, StructureSource,
};

fn fault_tree_structure(
    tree: &FaultTree,
    name: &str,
    top: EnvelopeTop,
    settings: &StructureSettings,
) -> Result<FaultTreeStructure, FtaError> {
    produce_structure(tree, name, top, settings, StructureSource::SuppliedTree)
}

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

#[test]
fn structure_does_not_quantify_and_omitted_cuts_do_not_spend_the_budget() {
    let mut tree = parallel(vec![GenerationWarning::GuardReadsState {
        transition: "B2.health.fail".to_owned(),
    }]);
    if let FtNode::Gate { op, .. } = &mut tree.top {
        *op = GateOp::Or;
    }
    let settings = StructureSettings {
        cut_sets: false,
        cut_set_limit: 1,
    };
    let result = fault_tree_structure(&tree, "pair", top(), &settings).unwrap();
    assert_eq!(result.provenance.source, StructureSource::SuppliedTree);
    assert!(result.provenance.generated_at_unix_ms > 0);
    assert!(result
        .provenance
        .dependency_lock_hash
        .starts_with("sha256:"));
    assert_eq!(result.minimal_cut_sets, None);
    assert_eq!(result.cut_sets_omitted.as_deref(), Some("not requested"));
    assert_eq!(result.tree.warnings, tree.warnings);
    assert_eq!(
        read_fault_tree_structure(&result.to_json().unwrap()).unwrap(),
        result
    );
    assert!(fault_tree_structure(
        &tree,
        "pair",
        top(),
        &StructureSettings {
            cut_sets: true,
            ..settings
        }
    )
    .is_err());
}

#[test]
fn structural_sets_are_named_and_unknown_references_are_refused() {
    let result = fault_tree_structure(
        &parallel(vec![]),
        "pair",
        top(),
        &StructureSettings {
            cut_sets: true,
            cut_set_limit: 100,
        },
    )
    .unwrap();
    assert_eq!(
        result.minimal_cut_sets,
        Some(vec![vec![
            "B1.health.fail".to_owned(),
            "B2.health.fail".to_owned()
        ]])
    );
    let mut doc = serde_json::to_value(&result).unwrap();
    doc["tree"]["top"]["children"][0]["event"] = 10.into();
    assert!(read_fault_tree_structure(&doc.to_string())
        .unwrap_err()
        .to_string()
        .contains("unknown"));
    doc["format"] = "other".into();
    assert!(read_fault_tree_structure(&doc.to_string())
        .unwrap_err()
        .to_string()
        .contains("format"));
}

#[test]
fn invalid_public_tree_producer_returns_typed_errors_with_and_without_cuts() {
    let dangling = FaultTree {
        top: FtNode::Basic { event: 0 },
        basic_events: vec![],
        warnings: vec![],
    };
    for cut_sets in [true, false] {
        let result = std::panic::catch_unwind(|| {
            fault_tree_structure(
                &dangling,
                "bad",
                top(),
                &StructureSettings {
                    cut_sets,
                    cut_set_limit: 100,
                },
            )
        });
        assert!(matches!(result, Ok(Err(raichu_fta::FtaError::Invalid(_)))));
    }
}

#[test]
fn reader_refuses_duplicate_constant_and_contradictory_omission_documents() {
    let result = fault_tree_structure(
        &parallel(vec![]),
        "pair",
        top(),
        &StructureSettings {
            cut_sets: true,
            cut_set_limit: 100,
        },
    )
    .unwrap();
    let base = serde_json::to_value(result).unwrap();
    for case in ["duplicate", "constant", "omitted", "listed"] {
        let mut doc = base.clone();
        match case {
            "duplicate" => {
                doc["tree"]["basic_events"][1]["name"] =
                    doc["tree"]["basic_events"][0]["name"].clone()
            }
            "constant" => doc["tree"]["top"] = serde_json::json!({"node":"constant","value":false}),
            "omitted" => doc["settings"]["cut_sets"] = false.into(),
            "listed" => doc["cut_sets_omitted"] = "not requested".into(),
            _ => unreachable!(),
        }
        assert!(
            read_fault_tree_structure(&doc.to_string()).is_err(),
            "{case}"
        );
    }
}

#[test]
fn gate_identifiers_and_origins_are_carried_and_a_repeated_identifier_is_refused() {
    let origin = raichu_core::GateOrigin {
        kind: raichu_core::OriginKind::Attribute,
        component: "T".to_owned(),
        name: "fed".to_owned(),
        negated: true,
    };
    let mut tree = parallel(vec![]);
    tree.top = FtNode::Gate {
        op: GateOp::Or,
        id: Some("top".to_owned()),
        origin: Some(origin.clone()),
        children: vec![
            FtNode::Gate {
                op: GateOp::And,
                id: Some("G1".to_owned()),
                origin: None,
                children: vec![FtNode::Basic { event: 0 }, FtNode::Basic { event: 1 }],
            },
            FtNode::Basic { event: 0 },
        ],
    };
    let settings = StructureSettings {
        cut_sets: false,
        cut_set_limit: 1,
    };
    let result = fault_tree_structure(&tree, "pair", top(), &settings).unwrap();
    let doc = serde_json::to_value(&result).unwrap();
    assert_eq!(doc["tree"]["top"]["id"], "top");
    assert_eq!(
        doc["tree"]["top"]["origin"],
        serde_json::json!({"kind": "attribute", "component": "T", "name": "fed", "negated": true})
    );
    assert_eq!(read_fault_tree_structure(&doc.to_string()).unwrap(), result);
    let mut repeated = doc.clone();
    repeated["tree"]["top"]["children"][0]["id"] = "top".into();
    assert!(read_fault_tree_structure(&repeated.to_string())
        .unwrap_err()
        .to_string()
        .contains("duplicate gate identifiers"));
}

/// A structure as the previous release wrote it, version 1: warnings are
/// plain sentences, gates carry no identifier nor origin.
const VERSION_1: &str = r#"{
  "format": "raichu.fault_tree.structure",
  "version": 1,
  "provenance": {"engine_version": "0.99.0", "tree": "pair",
                 "source": {"kind": "supplied_tree"},
                 "generated_at_unix_ms": 1760000000000,
                 "dependency_lock_hash": "sha256:00"},
  "top": {"kind": "targets", "targets": ["T_lost"]},
  "tree": {
    "top": {"node": "gate", "gate": "and",
            "children": [{"node": "basic", "event": 0}, {"node": "basic", "event": 1}]},
    "basic_events": [
      {"name": "B1.health.fail", "component": "B1", "automaton": "B1.health",
       "transition": "fail", "target": "nok", "law": "exponential", "rate": 0.001},
      {"name": "B2.health.fail", "component": "B2", "automaton": "B2.health",
       "transition": "fail", "target": "nok", "law": "exponential", "rate": 0.002}
    ],
    "warnings": ["transition `B2.health.fail` is guarded by a condition on states; the tree times its draw from the start of the mission, not from when the condition holds, so it over-estimates"]
  },
  "settings": {"cut_sets": true, "cut_set_limit": 100},
  "minimal_cut_sets": [["B1.health.fail", "B2.health.fail"]],
  "cut_sets_omitted": null
}"#;

#[test]
fn a_structure_of_the_previous_release_reads_with_legacy_warnings() {
    let read = read_fault_tree_structure(VERSION_1).expect("a version-1 structure reads");
    assert_eq!(read.version, 1);
    let sentence = "transition `B2.health.fail` is guarded by a condition on states; the \
                    tree times its draw from the start of the mission, not from when the \
                    condition holds, so it over-estimates";
    assert_eq!(
        read.tree.warnings,
        vec![GenerationWarning::Legacy {
            message: sentence.to_owned()
        }]
    );
    assert!(read.tree.basic_events.iter().all(|e| e.repairs.is_empty()));
    let text = read.to_json().unwrap();
    let document: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        document["tree"]["warnings"][0],
        serde_json::json!({"kind": "legacy", "message": sentence})
    );
    assert_eq!(read_fault_tree_structure(&text).unwrap(), read);
}

#[test]
fn a_version_2_structure_refuses_legacy_warnings_and_version_3_is_refused_by_name() {
    assert_eq!(raichu_fta::FAULT_TREE_STRUCTURE_VERSION, 2);
    let written = fault_tree_structure(
        &parallel(vec![]),
        "pair",
        top(),
        &StructureSettings {
            cut_sets: false,
            cut_set_limit: 1,
        },
    )
    .unwrap();
    assert_eq!(written.version, 2);
    let mut document: serde_json::Value = serde_json::from_str(VERSION_1).unwrap();
    document["version"] = 2.into();
    let error = read_fault_tree_structure(&document.to_string()).unwrap_err();
    assert!(error.to_string().contains("legacy"), "{error}");
    document["version"] = 3.into();
    let error = read_fault_tree_structure(&document.to_string()).unwrap_err();
    assert_eq!(
        error,
        raichu_fta::ReadFaultTreeStructureError::Version(Some(3))
    );
    assert!(error.to_string().contains("version Some(3)"), "{error}");
}
