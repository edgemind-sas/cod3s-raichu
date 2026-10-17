//! The probability of every gate at every mission time: each identified
//! gate is read off the literal the normalisation made of it (a diagram
//! node, a variable or a constant), across the reductions, the sharing and
//! the coalescing that reshape the gates; a gate in a module quantified
//! from its cut sets, or reading one, has no exact value and says why.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use raichu_core::{BasicLaw, FaultTree, FtNode, GateOp, TreeEvent, TreeRepair};
use raichu_fta::{
    fault_tree_envelope, quantify, BasicEvent, Engine, EnvelopeGate, EnvelopeTop,
    FaultTreeEnvelope, FaultTreeMeasure, Formula, GateDef, GateReason, Law, QuantifySettings, Tree,
};

const TIMES: [f64; 3] = [100.0, 500.0, 1000.0];
const RATES: [f64; 4] = [1e-3, 2e-3, 5e-4, 3e-3];

fn event(index: usize, rate: f64, repairs: Vec<TreeRepair>) -> TreeEvent {
    let component = format!("B{index}");
    TreeEvent {
        name: format!("{component}.health.fail"),
        automaton: format!("{component}.health"),
        component,
        transition: "fail".to_owned(),
        target: "nok".to_owned(),
        law: BasicLaw::Exponential { rate },
        repairs,
    }
}

fn gate(op: GateOp, id: &str, children: Vec<FtNode>) -> FtNode {
    FtNode::Gate {
        op,
        id: Some(id.to_owned()),
        origin: None,
        children,
    }
}

fn basic(event: usize) -> FtNode {
    FtNode::Basic { event }
}

/// Every reshaping the normalisation does to a gate, in one tree:
/// - `G2` is a disjunction under the disjunctive top with no other
///   referrer, so it is coalesced into the top; `G3`, a disjunction under
///   `G2`, into `G2`;
/// - `G4` has a single child and reduces to the event itself;
/// - `G5` is the same conjunction as `G1` and is shared with it.
fn reshaped() -> FaultTree {
    FaultTree {
        top: gate(
            GateOp::Or,
            "top",
            vec![
                gate(GateOp::And, "G1", vec![basic(0), basic(1)]),
                gate(
                    GateOp::Or,
                    "G2",
                    vec![basic(2), gate(GateOp::Or, "G3", vec![basic(3), basic(0)])],
                ),
                gate(GateOp::Or, "G4", vec![basic(2)]),
                gate(GateOp::And, "G5", vec![basic(0), basic(1)]),
            ],
        ),
        basic_events: RATES
            .iter()
            .enumerate()
            .map(|(i, &rate)| event(i, rate, Vec::new()))
            .collect(),
        warnings: Vec::new(),
    }
}

fn top() -> EnvelopeTop {
    EnvelopeTop::Targets {
        targets: vec!["T_lost".to_owned()],
    }
}

fn by_id(gates: &[EnvelopeGate]) -> HashMap<&str, &EnvelopeGate> {
    gates.iter().map(|g| (g.id.as_str(), g)).collect()
}

fn envelope(tree: &FaultTree, settings: &QuantifySettings) -> FaultTreeEnvelope {
    fault_tree_envelope(tree, "reshaped", top(), &TIMES, settings).expect("envelope")
}

/// Within the rounding of the closed forms written naively below.
fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 1e-12 * expected.abs().max(1e-300)
}

#[test]
fn every_gate_has_its_exact_probability_at_every_instant() {
    let envelope = envelope(&reshaped(), &QuantifySettings::default());
    for instant in &envelope.instants {
        let t = instant.mission_time;
        let p: Vec<f64> = RATES.iter().map(|r| 1.0 - (-r * t).exp()).collect();
        let gates = by_id(&instant.gates);
        assert_eq!(gates.len(), 6, "{:?}", instant.gates);
        let expected = [
            ("G1", p[0] * p[1]),
            ("G5", p[0] * p[1]),
            ("G3", 1.0 - (1.0 - p[3]) * (1.0 - p[0])),
            ("G2", 1.0 - (1.0 - p[2]) * (1.0 - p[3]) * (1.0 - p[0])),
            ("G4", p[2]),
            ("top", 1.0 - (1.0 - p[0]) * (1.0 - p[2]) * (1.0 - p[3])),
        ];
        for (id, value) in expected {
            let gate = gates[id];
            assert_eq!(gate.reason, None, "{gate:?}");
            let probability = gate.probability.expect("a probability");
            assert!(
                close(probability, value),
                "{id} at {t}: {probability} vs {value}"
            );
        }
        // The root gate is the top: the same number, to the bit.
        assert_eq!(gates["top"].probability, Some(instant.probability));
    }
    // The horizon repeats the last instant.
    assert_eq!(envelope.horizon.gates, envelope.instants[2].gates);
}

#[test]
fn a_gate_reduced_to_one_event_has_that_event_probability() {
    // Even where the engine has no diagram: the event is exactly known.
    for engine in [Engine::Auto, Engine::CutSets] {
        let settings = QuantifySettings {
            engine,
            ..Default::default()
        };
        let envelope = envelope(&reshaped(), &settings);
        for instant in &envelope.instants {
            let expected = 1.0 - (-RATES[2] * instant.mission_time).exp();
            let gate = by_id(&instant.gates)["G4"];
            assert_eq!(gate.reason, None, "{gate:?}");
            assert!(close(gate.probability.unwrap(), expected), "{gate:?}");
        }
        // The event's own probability, to the bit.
        assert_eq!(
            by_id(&envelope.horizon.gates)["G4"].probability,
            envelope.horizon.importance[2].probability
        );
    }
}

#[test]
fn a_gate_of_a_module_quantified_from_its_cut_sets_has_none_and_says_why() {
    let settings = QuantifySettings {
        engine: Engine::CutSets,
        ..Default::default()
    };
    let envelope = envelope(&reshaped(), &settings);
    let every = envelope.instants.iter().map(|i| &i.gates);
    for gates in every.chain([&envelope.horizon.gates]) {
        assert_eq!(gates.len(), 6);
        // `G4` is an event, not a part of any module.
        for gate in gates.iter().filter(|g| g.id != "G4") {
            assert_eq!(gate.probability, None, "{gate:?}");
            assert_eq!(gate.reason, Some(GateReason::CutSets), "{gate:?}");
        }
    }
    // The top is still quantified: a partial result, not a refusal.
    assert!(envelope.horizon.probability > 0.0);
}

/// A vote of three among eight, too large for the diagram budget, under
/// a disjunction with one more event that fits in it.
fn vote_under_or() -> FaultTree {
    let rates: Vec<f64> = (0..9).map(|i| 1e-3 * (1.0 + i as f64)).collect();
    FaultTree {
        top: gate(
            GateOp::Or,
            "top",
            vec![
                gate(GateOp::AtLeast { k: 3 }, "G1", (0..8).map(basic).collect()),
                basic(8),
            ],
        ),
        basic_events: rates
            .iter()
            .enumerate()
            .map(|(i, &rate)| event(i, rate, Vec::new()))
            .collect(),
        warnings: Vec::new(),
    }
}

#[test]
fn a_gate_reading_a_module_quantified_from_its_cut_sets_has_none_either() {
    let tree = vote_under_or();
    let exact = fault_tree_envelope(&tree, "vote", top(), &[1000.0], &Default::default()).unwrap();
    let vote_nodes = exact.horizon.provenance["modules"][1]["nodes_created"]
        .as_u64()
        .unwrap() as usize;
    let top_nodes = exact.horizon.provenance["modules"][0]["nodes_created"]
        .as_u64()
        .unwrap() as usize;
    assert!(top_nodes < vote_nodes, "{top_nodes} >= {vote_nodes}");
    // A budget the vote outgrows and the top fits in.
    let settings = QuantifySettings {
        max_bdd_nodes: vote_nodes - 1,
        ..Default::default()
    };
    let envelope = fault_tree_envelope(&tree, "vote", top(), &[1000.0], &settings).unwrap();
    assert_eq!(envelope.horizon.method, "bdd+cut_sets");
    let gates = by_id(&envelope.horizon.gates);
    assert_eq!(gates["G1"].probability, None);
    assert_eq!(gates["G1"].reason, Some(GateReason::CutSets));
    assert_eq!(gates["top"].probability, None);
    assert_eq!(gates["top"].reason, Some(GateReason::ReadsCutSets));
}

#[test]
fn under_the_unavailability_every_gate_follows_the_closed_form() {
    const LAMBDA: [f64; 3] = [1e-3, 2e-3, 5e-4];
    const MU: [f64; 3] = [0.1, 0.05, 0.2];
    let events = (0..3)
        .map(|i| {
            event(
                i,
                LAMBDA[i],
                vec![TreeRepair {
                    transition: format!("B{i}.health.repair"),
                    law: Some(BasicLaw::Exponential { rate: MU[i] }),
                    guarded: false,
                }],
            )
        })
        .collect();
    // A parallel pair in series with a third block.
    let tree = FaultTree {
        top: gate(
            GateOp::Or,
            "top",
            vec![gate(GateOp::And, "G1", vec![basic(0), basic(1)]), basic(2)],
        ),
        basic_events: events,
        warnings: Vec::new(),
    };
    let settings = QuantifySettings {
        measure: FaultTreeMeasure::Unavailability,
        ..Default::default()
    };
    let envelope = fault_tree_envelope(&tree, "series", top(), &TIMES, &settings).unwrap();
    for instant in &envelope.instants {
        let t = instant.mission_time;
        let q: Vec<f64> = (0..3)
            .map(|i| LAMBDA[i] / (LAMBDA[i] + MU[i]) * (1.0 - (-(LAMBDA[i] + MU[i]) * t).exp()))
            .collect();
        let gates = by_id(&instant.gates);
        let parallel = gates["G1"].probability.unwrap();
        assert!(close(parallel, q[0] * q[1]), "{parallel} at {t}");
        let series = gates["top"].probability.unwrap();
        let expected = 1.0 - (1.0 - q[0] * q[1]) * (1.0 - q[2]);
        assert!(close(series, expected), "{series} vs {expected} at {t}");
        assert_eq!(series, instant.probability);
    }
}

#[test]
fn a_gate_reached_through_a_negation_is_valued_positively() {
    let constant = |name: &str, probability: f64| BasicEvent {
        name: name.to_owned(),
        law: Law::Constant { probability },
        repairs: Vec::new(),
    };
    let tree = Tree {
        name: "negated".to_owned(),
        events: vec![constant("a", 0.1), constant("b", 0.2), constant("c", 0.3)],
        gates: vec![
            GateDef {
                name: "both".to_owned(),
                formula: Formula::and(vec![Formula::event(0), Formula::event(1)]),
            },
            GateDef {
                name: "unused".to_owned(),
                formula: Formula::event(2),
            },
        ],
        top: Formula::or(vec![
            Formula::not(Formula::Gate { gate: 0 }),
            Formula::event(2),
        ]),
    };
    let quiet = quantify(&tree, &QuantifySettings::default()).unwrap();
    assert!(quiet.gates.is_empty(), "only computed when asked for");
    let settings = QuantifySettings {
        gate_probabilities: true,
        ..Default::default()
    };
    let result = quantify(&tree, &settings).unwrap();
    assert_eq!(
        result.probability, quiet.probability,
        "asking changes nothing else"
    );
    assert_eq!(result.gates.len(), 3);
    assert_eq!(result.gates[0].gate, None);
    assert_eq!(result.gates[0].probability, Some(result.probability));
    assert_eq!(result.gates[1].gate, Some(0));
    assert!(close(result.gates[1].probability.unwrap(), 0.1 * 0.2));
    assert_eq!(result.gates[2].gate, Some(1));
    assert_eq!(result.gates[2].probability, None);
    assert_eq!(result.gates[2].reason, Some(GateReason::NotReached));
}
