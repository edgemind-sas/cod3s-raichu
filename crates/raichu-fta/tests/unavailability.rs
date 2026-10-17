//! The unavailability at a mission time: a repairable basic event's
//! probability of being in its failure state, in closed form for an
//! exponential failure with exponential repairs, asymptotic otherwise, and
//! the warnings the envelope emits once per event according to the
//! measure.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{BasicLaw, FaultTree, FtNode, GateOp, GenerationWarning, TreeEvent, TreeRepair};
use raichu_fta::{
    fault_tree_envelope, quantify, read_fault_tree_envelope, read_open_psa, read_open_psa_measure,
    write_open_psa, BasicEvent, EnvelopeTop, FaultTreeEnvelope, FaultTreeMeasure, Formula, Law,
    QuantifySettings, Repair, Tree, MEASURE_UNAVAILABILITY, MEASURE_WITHOUT_REPAIR,
};

const LAMBDA: f64 = 1e-3;
const MU: f64 = 0.1;

fn repair(component: &str, law: BasicLaw) -> TreeRepair {
    TreeRepair {
        transition: format!("{component}.health.repair"),
        law: Some(law),
        guarded: false,
    }
}

fn event(component: &str, law: BasicLaw, repairs: Vec<TreeRepair>) -> TreeEvent {
    TreeEvent {
        name: format!("{component}.health.fail"),
        component: component.to_owned(),
        automaton: format!("{component}.health"),
        transition: "fail".to_owned(),
        target: "nok".to_owned(),
        law,
        repairs,
    }
}

fn repairable(component: &str, lambda: f64, mu: f64) -> TreeEvent {
    event(
        component,
        BasicLaw::Exponential { rate: lambda },
        vec![repair(component, BasicLaw::Exponential { rate: mu })],
    )
}

fn single(event: TreeEvent) -> FaultTree {
    FaultTree {
        top: FtNode::Basic { event: 0 },
        basic_events: vec![event],
        warnings: Vec::new(),
    }
}

fn gate(op: GateOp, events: Vec<TreeEvent>) -> FaultTree {
    FaultTree {
        top: FtNode::Gate {
            op,
            id: None,
            origin: None,
            children: (0..events.len())
                .map(|event| FtNode::Basic { event })
                .collect(),
        },
        basic_events: events,
        warnings: Vec::new(),
    }
}

fn top() -> EnvelopeTop {
    EnvelopeTop::Targets {
        targets: vec!["T_lost".to_owned()],
    }
}

fn settings(measure: FaultTreeMeasure) -> QuantifySettings {
    QuantifySettings {
        measure,
        ..Default::default()
    }
}

fn envelope(tree: &FaultTree, times: &[f64], measure: FaultTreeMeasure) -> FaultTreeEnvelope {
    fault_tree_envelope(tree, "t", top(), times, &settings(measure)).expect("envelope")
}

/// `λ/(λ+µ) (1 - e^{-(λ+µ)t})`.
fn closed_form(lambda: f64, mu: f64, t: f64) -> f64 {
    lambda / (lambda + mu) * (1.0 - (-(lambda + mu) * t).exp())
}

fn kinds(envelope: &FaultTreeEnvelope) -> Vec<String> {
    envelope
        .generation
        .warnings
        .iter()
        .map(|w| {
            serde_json::to_value(&w.warning).unwrap()["kind"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect()
}

fn warnings(envelope: &FaultTreeEnvelope) -> Vec<GenerationWarning> {
    envelope
        .generation
        .warnings
        .iter()
        .map(|w| w.warning.clone())
        .collect()
}

#[test]
fn an_exponential_pair_is_its_closed_form_and_its_repair_is_not_ignored() {
    // AE1: λ = 1e-3/h, µ = 0.1/h, t = 1000 h.
    let tree = single(repairable("B1", LAMBDA, MU));
    let result = envelope(&tree, &[1000.0], FaultTreeMeasure::Unavailability);
    let expected = closed_form(LAMBDA, MU, 1000.0);
    assert!((expected - 0.009_900_990_099_009_9).abs() < 1e-12);
    assert!(
        (result.horizon.probability - expected).abs() < 1e-12,
        "{} vs {expected}",
        result.horizon.probability
    );
    assert_eq!(result.measure, MEASURE_UNAVAILABILITY);
    assert!(
        result.generation.warnings.is_empty(),
        "{:?}",
        kinds(&result)
    );
    assert!(result.generation.exact);
    assert_eq!(result.horizon.upper_bound, Some(result.horizon.probability));
}

#[test]
fn without_repair_the_same_event_is_its_distribution_and_its_repair_is_ignored() {
    // AE1, the other measure: 1 - e^{-1} and `repair_ignored`.
    let tree = single(repairable("B1", LAMBDA, MU));
    let result = envelope(&tree, &[1000.0], FaultTreeMeasure::ProbabilityWithoutRepair);
    let expected = 1.0 - (-1.0f64).exp();
    assert!((result.horizon.probability - expected).abs() < 1e-15);
    assert!((result.horizon.probability - 0.6321).abs() < 1e-4);
    assert_eq!(result.measure, MEASURE_WITHOUT_REPAIR);
    assert_eq!(
        warnings(&result),
        vec![GenerationWarning::RepairIgnored {
            event: "B1.health.fail".to_owned(),
            automaton: "B1.health".to_owned(),
            state: "nok".to_owned(),
            transitions: vec!["B1.health.repair".to_owned()],
        }]
    );
    assert!(!result.generation.exact);
    // Ignoring a repair over-estimates: the bound stands.
    assert_eq!(result.horizon.upper_bound, Some(result.horizon.probability));
}

#[test]
fn the_default_measure_is_the_probability_without_repair() {
    assert_eq!(
        QuantifySettings::default().measure,
        FaultTreeMeasure::ProbabilityWithoutRepair
    );
}

#[test]
fn a_non_exponential_pair_is_asymptotic_and_said_so_and_has_no_bound() {
    // AE2: failure after 500 h, repair after 24 h: 24 / 524.
    let tree = single(event(
        "B1",
        BasicLaw::Delay { time: 500.0 },
        vec![repair("B1", BasicLaw::Delay { time: 24.0 })],
    ));
    let result = envelope(&tree, &[100.0, 510.0], FaultTreeMeasure::Unavailability);
    for instant in &result.instants {
        assert!(
            (instant.probability - 24.0 / 524.0).abs() < 1e-15,
            "{instant:?}"
        );
        assert_eq!(instant.upper_bound, None, "{instant:?}");
    }
    assert_eq!(result.horizon.upper_bound, None);
    assert_eq!(
        warnings(&result),
        vec![GenerationWarning::AsymptoticUnavailability {
            event: "B1.health.fail".to_owned(),
        }]
    );
    assert!(!result.generation.exact);
    // The module's own bound goes too.
    let modules = result.horizon.provenance["modules"].as_array().unwrap();
    assert!(
        modules.iter().all(|m| m["upper_bound"].is_null()),
        "{modules:?}"
    );
}

#[test]
fn a_guarded_repair_is_counted_and_said_to_be_assumed_available() {
    let mut guarded = repair("B1", BasicLaw::Exponential { rate: MU });
    guarded.guarded = true;
    let tree = single(event(
        "B1",
        BasicLaw::Exponential { rate: LAMBDA },
        vec![guarded],
    ));
    let result = envelope(&tree, &[1000.0], FaultTreeMeasure::Unavailability);
    assert!((result.horizon.probability - closed_form(LAMBDA, MU, 1000.0)).abs() < 1e-12);
    assert_eq!(
        warnings(&result),
        vec![GenerationWarning::GuardedRepairAssumedAvailable {
            event: "B1.health.fail".to_owned(),
            transition: "B1.health.repair".to_owned(),
        }]
    );
    assert_eq!(result.horizon.upper_bound, None);
}

#[test]
fn several_exponential_repairs_sum_into_one_rate() {
    let tree = single(event(
        "B1",
        BasicLaw::Exponential { rate: LAMBDA },
        vec![
            repair("B1", BasicLaw::Exponential { rate: 0.04 }),
            TreeRepair {
                transition: "B1.health.replace".to_owned(),
                law: Some(BasicLaw::Exponential { rate: 0.06 }),
                guarded: false,
            },
        ],
    ));
    let result = envelope(&tree, &[1000.0], FaultTreeMeasure::Unavailability);
    assert!((result.horizon.probability - closed_form(LAMBDA, 0.1, 1000.0)).abs() < 1e-12);
    assert!(result.generation.warnings.is_empty());
}

#[test]
fn an_unrepaired_event_keeps_its_distribution_under_unavailability() {
    let tree = single(event(
        "B1",
        BasicLaw::Exponential { rate: LAMBDA },
        Vec::new(),
    ));
    let result = envelope(&tree, &[1000.0], FaultTreeMeasure::Unavailability);
    assert!((result.horizon.probability - (1.0 - (-1.0f64).exp())).abs() < 1e-15);
    assert!(result.generation.warnings.is_empty());
    assert!(result.generation.exact);
}

#[test]
fn series_and_parallel_of_repairable_blocks_are_their_closed_forms() {
    let t = 50.0;
    let (q1, q2) = (closed_form(1e-2, 0.2, t), closed_form(3e-2, 0.5, t));
    let blocks = || vec![repairable("B1", 1e-2, 0.2), repairable("B2", 3e-2, 0.5)];
    let parallel = envelope(
        &gate(GateOp::And, blocks()),
        &[t],
        FaultTreeMeasure::Unavailability,
    );
    assert!((parallel.horizon.probability - q1 * q2).abs() < 1e-15);
    let series = envelope(
        &gate(GateOp::Or, blocks()),
        &[t],
        FaultTreeMeasure::Unavailability,
    );
    let expected = 1.0 - (1.0 - q1) * (1.0 - q2);
    assert!((series.horizon.probability - expected).abs() < 1e-15);
    // A cut set is the product of its events' unavailabilities.
    let cut = &parallel.horizon.minimal_cut_sets.as_ref().unwrap()[0];
    assert!((cut.probability.unwrap() - q1 * q2).abs() < 1e-15);
}

#[test]
fn twelve_repairable_events_give_twelve_ignored_repairs_whatever_the_instants() {
    let events: Vec<TreeEvent> = (1..=12)
        .map(|i| repairable(&format!("B{i}"), LAMBDA, MU))
        .collect();
    let tree = gate(GateOp::Or, events);
    for times in [vec![1000.0], vec![10.0, 100.0, 500.0, 800.0, 1000.0]] {
        let result = envelope(&tree, &times, FaultTreeMeasure::ProbabilityWithoutRepair);
        let ignored = kinds(&result)
            .iter()
            .filter(|k| *k == "repair_ignored")
            .count();
        assert_eq!(ignored, 12, "{times:?}");
        assert!(result.instants.iter().all(|i| i.warnings.is_empty()));
    }
}

#[test]
fn a_common_cause_combination_with_a_repair_is_an_approximation() {
    let mut cc = repairable("P", LAMBDA, MU);
    cc.name = "P.fm__cc_1_2.occ__cc_1_2".to_owned();
    cc.automaton = "P.fm__cc_1_2".to_owned();
    cc.transition = "occ__cc_1_2".to_owned();
    cc.target = "occ__cc_1_2".to_owned();
    let tree = single(cc);
    let result = envelope(&tree, &[1000.0], FaultTreeMeasure::Unavailability);
    assert!((result.horizon.probability - closed_form(LAMBDA, MU, 1000.0)).abs() < 1e-12);
    assert_eq!(
        warnings(&result),
        vec![GenerationWarning::CommonCauseRepairApproximation {
            event: "P.fm__cc_1_2.occ__cc_1_2".to_owned(),
        }]
    );
    assert_eq!(result.horizon.upper_bound, None);
}

#[test]
fn an_over_estimation_code_reads_neutrally_under_unavailability() {
    let warning = GenerationWarning::GuardReadsState {
        transition: "B1.health.fail".to_owned(),
    };
    let mut tree = single(repairable("B1", LAMBDA, MU));
    tree.warnings = vec![warning.clone()];
    let without = envelope(&tree, &[1000.0], FaultTreeMeasure::ProbabilityWithoutRepair);
    let with = envelope(&tree, &[1000.0], FaultTreeMeasure::Unavailability);
    let message = |e: &FaultTreeEnvelope| {
        e.generation
            .warnings
            .iter()
            .find(|w| w.warning == warning)
            .unwrap()
            .message
            .clone()
    };
    assert!(
        message(&without).contains("over-estimates"),
        "{}",
        message(&without)
    );
    assert!(
        !message(&with).contains("over-estimates"),
        "{}",
        message(&with)
    );
    assert_eq!(message(&without), warning.to_string());
}

#[test]
fn the_envelope_writes_its_measure_and_reads_back() {
    let tree = single(event(
        "B1",
        BasicLaw::Delay { time: 500.0 },
        vec![repair("B1", BasicLaw::Delay { time: 24.0 })],
    ));
    let result = envelope(&tree, &[1000.0], FaultTreeMeasure::Unavailability);
    let text = result.to_json().unwrap();
    let document: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(document["measure"], "unavailability");
    assert_eq!(
        document["generation"]["warnings"][0]["kind"],
        "asymptotic_unavailability"
    );
    assert_eq!(
        document["generation"]["warnings"][0]["event"],
        "B1.health.fail"
    );
    assert!(document["generation"]["warnings"][0]["message"].is_string());
    assert_eq!(
        document["generation"]["basic_events"][0]["repairs"][0]["transition"],
        "B1.health.repair"
    );
    let read = read_fault_tree_envelope(&text).expect("read");
    assert_eq!(read, result);
    // An envelope written before repairs and measure warnings existed
    // still reads: its events carry no repairs.
    let mut old = document.clone();
    old["generation"]["warnings"] = serde_json::json!([]);
    old["generation"]["basic_events"][0]
        .as_object_mut()
        .unwrap()
        .remove("repairs");
    let read = read_fault_tree_envelope(&old.to_string()).expect("read old");
    assert!(read.generation.basic_events[0].repairs.is_empty());
}

#[test]
fn the_open_psa_round_trip_keeps_the_repair_and_the_measure() {
    let mut guarded = repair("B2", BasicLaw::Delay { time: 24.0 });
    guarded.guarded = true;
    let tree = gate(
        GateOp::And,
        vec![
            repairable("B1", 1e-2, 0.2),
            event("B2", BasicLaw::Delay { time: 500.0 }, vec![guarded]),
        ],
    );
    let xml = tree.to_open_psa("t", FaultTreeMeasure::Unavailability);
    assert_eq!(
        read_open_psa_measure(&xml).unwrap(),
        Some(FaultTreeMeasure::Unavailability)
    );
    let read = read_open_psa(&xml, None).expect("read");
    assert_eq!(
        read.events[0].repairs,
        vec![Repair {
            transition: "B1.health.repair".to_owned(),
            law: Some(Law::Exponential { rate: 0.2 }),
            guarded: false,
        }]
    );
    assert_eq!(
        read.events[1].repairs,
        vec![Repair {
            transition: "B2.health.repair".to_owned(),
            law: Some(Law::Delay { time: 24.0 }),
            guarded: true,
        }]
    );
    // Quantifying the document agrees with the envelope of the tree.
    let t = 600.0;
    let from_document = quantify(
        &read,
        &QuantifySettings {
            mission_time: Some(t),
            ..settings(FaultTreeMeasure::Unavailability)
        },
    )
    .unwrap();
    let from_tree = envelope(&tree, &[t], FaultTreeMeasure::Unavailability);
    assert_eq!(
        from_document.probability.to_bits(),
        from_tree.horizon.probability.to_bits()
    );
    // The quantifier's own writer keeps the repairs too.
    let again = read_open_psa(&write_open_psa(&read), None).expect("again");
    assert_eq!(again.events, read.events);
    // A document that names no measure says so.
    assert_eq!(read_open_psa_measure(&write_open_psa(&read)).unwrap(), None);
    let without = tree.to_open_psa("t", FaultTreeMeasure::ProbabilityWithoutRepair);
    assert_eq!(
        read_open_psa_measure(&without).unwrap(),
        Some(FaultTreeMeasure::ProbabilityWithoutRepair)
    );
}

#[test]
fn a_quantified_tree_with_repairs_follows_the_measure_it_is_given() {
    let tree = Tree {
        name: "t".to_owned(),
        events: vec![BasicEvent {
            name: "e".to_owned(),
            law: Law::Exponential { rate: LAMBDA },
            repairs: vec![Repair {
                transition: "r".to_owned(),
                law: Some(Law::Exponential { rate: MU }),
                guarded: false,
            }],
        }],
        gates: Vec::new(),
        top: Formula::event(0),
    };
    let at = |measure| {
        quantify(
            &tree,
            &QuantifySettings {
                mission_time: Some(1000.0),
                ..settings(measure)
            },
        )
        .unwrap()
        .probability
    };
    assert!((at(FaultTreeMeasure::Unavailability) - closed_form(LAMBDA, MU, 1000.0)).abs() < 1e-12);
    assert!(
        (at(FaultTreeMeasure::ProbabilityWithoutRepair) - (1.0 - (-1.0f64).exp())).abs() < 1e-15
    );
}
