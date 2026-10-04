//! The `raichu.fault_tree` envelope: one quantification per mission time,
//! cut sets and importance at the horizon, a format and a version a
//! reader checks first.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{BasicLaw, FaultTree, FtNode, GateOp, TreeEvent};
use raichu_fta::{
    fault_tree_envelope, read_fault_tree_envelope, EnvelopeTop, FtaError, QuantifySettings,
    ReadFaultTreeError, FAULT_TREE_FORMAT, FAULT_TREE_VERSION, MAX_MISSION_TIMES,
    MEASURE_WITHOUT_REPAIR,
};

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
    let warning = "`B1.health.nok` is left again by `B1.health.repair`".to_owned();
    let envelope = fault_tree_envelope(
        &parallel(vec![warning.clone()]),
        "rbd",
        top(),
        &[1000.0],
        &QuantifySettings::default(),
    )
    .expect("envelope");
    assert!(!envelope.generation.exact);
    assert_eq!(envelope.generation.warnings, vec![warning]);
    let text = envelope.to_json().expect("json");
    let read = read_fault_tree_envelope(&text).expect("read");
    // Numbers are checked for shape here, not re-read to the bit (see
    // `read_fault_tree_envelope`): everything else reads back unchanged.
    assert_eq!(read.generation, envelope.generation);
    assert_eq!(read.top, envelope.top);
    assert_eq!(read.settings, envelope.settings);
    assert_eq!(read.instants.len(), 1);
    assert_eq!(
        read.horizon.minimal_cut_sets.as_ref().map(|c| c.len()),
        Some(1)
    );
    assert!((read.horizon.probability - envelope.horizon.probability).abs() < 1e-15);
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
