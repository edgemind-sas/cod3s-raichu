//! Structural fault-tree output with optional cut-set extraction,
//! producer format validation and explicit omission.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{BasicLaw, FaultTree, FtNode, GateOp, TreeEvent};
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
    }
}

/// Two blocks in parallel: lost when both have failed.
fn parallel(warnings: Vec<String>) -> FaultTree {
    FaultTree {
        top: FtNode::Gate {
            op: GateOp::And,
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
    let mut tree = parallel(vec!["repair ignored".to_owned()]);
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
