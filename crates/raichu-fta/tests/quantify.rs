//! Exact quantification against closed forms and against brute-force
//! enumeration of the truth table.
#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use proptest::prelude::*;
use raichu_fta::{
    quantify, read_open_psa, write_open_psa, BasicEvent, Formula, FtaError, GateDef, Law,
    QuantifySettings, Tree,
};

fn constant(name: &str, probability: f64) -> BasicEvent {
    BasicEvent {
        name: name.to_owned(),
        law: Law::Constant { probability },
    }
}

fn tree(events: Vec<BasicEvent>, top: Formula) -> Tree {
    Tree {
        name: "t".to_owned(),
        events,
        gates: Vec::new(),
        top,
    }
}

fn e(i: usize) -> Formula {
    Formula::event(i)
}

fn run(tree: &Tree) -> raichu_fta::Quantification {
    quantify(tree, &QuantifySettings::default()).unwrap()
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-14 * b.abs().max(1.0), "{a} vs {b}");
}

// ---------------------------------------------------------------------
// Closed forms.
// ---------------------------------------------------------------------

#[test]
fn series_parallel_and_vote_give_their_closed_forms() {
    let p = [0.1, 0.2, 0.3];
    let events = || {
        p.iter()
            .enumerate()
            .map(|(i, q)| constant(&format!("E{i}"), *q))
            .collect::<Vec<_>>()
    };
    let series = run(&tree(events(), Formula::or(vec![e(0), e(1), e(2)])));
    close(series.probability, 1.0 - 0.9 * 0.8 * 0.7);
    let parallel = run(&tree(events(), Formula::and(vec![e(0), e(1), e(2)])));
    close(parallel.probability, 0.1 * 0.2 * 0.3);
    let vote = run(&tree(
        events(),
        Formula::at_least(2, vec![e(0), e(1), e(2)]),
    ));
    let exact = 0.1 * 0.2 * 0.7 + 0.1 * 0.8 * 0.3 + 0.9 * 0.2 * 0.3 + 0.1 * 0.2 * 0.3;
    close(vote.probability, exact);
    assert_eq!(
        vote.minimal_cut_sets,
        Some(vec![vec![0, 1], vec![0, 2], vec![1, 2]])
    );
}

/// The bridge: inputs 1 and 2, bridge 3, outputs 4 and 5. Its failure is
/// the union of its four minimal cut sets, which share events, so no
/// module splits it.
#[test]
fn the_bridge_gives_its_closed_form_and_its_four_cut_sets() {
    let q = 0.1;
    let events = (1..=5).map(|i| constant(&format!("C{i}"), q)).collect();
    let top = Formula::or(vec![
        Formula::and(vec![e(0), e(1)]),
        Formula::and(vec![e(3), e(4)]),
        Formula::and(vec![e(0), e(2), e(4)]),
        Formula::and(vec![e(1), e(2), e(3)]),
    ]);
    let result = run(&tree(events, top));
    let r: f64 = 1.0 - q;
    let reliability = 2.0 * r.powi(2) + 2.0 * r.powi(3) - 5.0 * r.powi(4) + 2.0 * r.powi(5);
    close(result.probability, 1.0 - reliability);
    assert_eq!(
        result.minimal_cut_sets,
        Some(vec![vec![0, 1], vec![3, 4], vec![0, 2, 4], vec![1, 2, 3]])
    );
    assert_eq!(result.cut_set_count, Some(4));
    assert!(result.coherent);
    assert_eq!(result.method, "bdd");
    assert!(result.exact);
    assert_eq!(result.upper_bound, Some(result.probability));
    assert!(result.warnings.is_empty());
}

#[test]
fn the_importance_of_a_series_pair_is_its_closed_form() {
    let (pa, pb) = (0.1, 0.3);
    let result = run(&tree(
        vec![constant("A", pa), constant("B", pb)],
        Formula::or(vec![e(0), e(1)]),
    ));
    let top = 1.0 - (1.0 - pa) * (1.0 - pb);
    let a = &result.importance[0];
    close(a.birnbaum, 1.0 - pb);
    close(a.criticality.unwrap(), pa * (1.0 - pb) / top);
    close(a.fussell_vesely.unwrap(), (top - pb) / top);
    close(a.diagnostic.unwrap(), pa / top);
    close(a.risk_achievement_worth.unwrap(), 1.0 / top);
    close(a.risk_reduction_worth.unwrap(), top / pb);
}

#[test]
fn an_event_that_alone_causes_the_top_has_an_infinite_reduction_worth() {
    let result = run(&tree(vec![constant("A", 0.2)], e(0)));
    close(result.probability, 0.2);
    assert_eq!(result.importance[0].risk_reduction_worth, None);
    assert_eq!(result.minimal_cut_sets, Some(vec![vec![0]]));
}

#[test]
fn independent_subtrees_are_modules_and_are_quantified_apart() {
    // (A.B) + (C.D) with a shared-free structure: every gate is a module.
    let events = ["A", "B", "C", "D"]
        .iter()
        .map(|n| constant(n, 0.25))
        .collect();
    let top = Formula::or(vec![
        Formula::and(vec![e(0), Formula::or(vec![e(1), e(2)])]),
        Formula::and(vec![e(3), Formula::or(vec![e(1), e(2)])]),
    ]);
    let result = run(&tree(events, top));
    // P = P((A + D).(B + C))
    close(
        result.probability,
        (1.0 - 0.75 * 0.75) * (1.0 - 0.75 * 0.75),
    );
    assert!(!result.provenance.modules.is_empty());
    assert_eq!(result.provenance.variable_order, "depth_first_left_most");
}

#[test]
fn a_non_coherent_tree_is_quantified_exactly_and_its_cut_sets_refused() {
    let (pa, pb) = (0.2, 0.7);
    let xor = Formula::or(vec![
        Formula::and(vec![e(0), Formula::not(e(1))]),
        Formula::and(vec![Formula::not(e(0)), e(1)]),
    ]);
    let result = run(&tree(vec![constant("A", pa), constant("B", pb)], xor));
    close(result.probability, pa * (1.0 - pb) + (1.0 - pa) * pb);
    assert!(!result.coherent);
    assert_eq!(result.minimal_cut_sets, None);
    assert_eq!(result.cut_set_count, None);
    assert!(result.cut_sets_omitted.unwrap().contains("not monotone"));
    // Birnbaum on a non-coherent top may be negative: B_A = (1-pb) - pb.
    close(result.importance[0].birnbaum, (1.0 - pb) - pb);
}

#[test]
fn a_negation_that_simplifies_away_leaves_the_tree_coherent() {
    // A + (A . not B) = A: monotone although a negation was written.
    let top = Formula::or(vec![e(0), Formula::and(vec![e(0), Formula::not(e(1))])]);
    let result = run(&tree(vec![constant("A", 0.3), constant("B", 0.5)], top));
    close(result.probability, 0.3);
    assert!(result.coherent);
    assert_eq!(result.minimal_cut_sets, Some(vec![vec![0]]));
}

#[test]
fn timed_laws_are_read_at_the_mission_time() {
    let events = vec![
        BasicEvent {
            name: "pump".to_owned(),
            law: Law::Exponential { rate: 1e-3 },
        },
        BasicEvent {
            name: "valve".to_owned(),
            law: Law::Weibull {
                shape: 2.0,
                scale: 500.0,
            },
        },
    ];
    let t = tree(events, Formula::and(vec![e(0), e(1)]));
    let err = quantify(&t, &QuantifySettings::default()).unwrap_err();
    assert!(matches!(err, FtaError::MissionTimeRequired { .. }));
    let settings = QuantifySettings {
        mission_time: Some(100.0),
        ..Default::default()
    };
    let result = quantify(&t, &settings).unwrap();
    let exact = (1.0 - (-0.1f64).exp()) * (1.0 - (-(0.2f64 * 0.2)).exp());
    close(result.probability, exact);
}

#[test]
fn a_cut_set_count_past_the_limit_is_counted_not_listed() {
    // (A1 + B1) . (A2 + B2) . ... over 10 pairs: 2^10 cut sets.
    let mut events = Vec::new();
    let mut factors = Vec::new();
    for i in 0..10 {
        events.push(constant(&format!("A{i}"), 0.1));
        events.push(constant(&format!("B{i}"), 0.1));
        factors.push(Formula::or(vec![e(2 * i), e(2 * i + 1)]));
    }
    let settings = QuantifySettings {
        cut_set_limit: 1000,
        ..Default::default()
    };
    let result = quantify(&tree(events, Formula::and(factors)), &settings).unwrap();
    assert_eq!(result.cut_set_count, Some(1024));
    assert_eq!(result.minimal_cut_sets, None);
    assert!(result.cut_sets_omitted.unwrap().contains("1024"));
    close(result.probability, (1.0 - 0.81f64).powi(10));
}

#[test]
fn cut_sets_not_requested_leave_the_measures_unchanged() {
    let events = (0..4)
        .map(|i| constant(&format!("E{i}"), 0.1 + 0.1 * i as f64))
        .collect();
    let t = tree(events, Formula::at_least(2, (0..4).map(e).collect()));
    let with = run(&t);
    let without = quantify(
        &t,
        &QuantifySettings {
            cut_sets: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(without.probability, with.probability);
    assert_eq!(without.importance, with.importance);
    assert_eq!(without.minimal_cut_sets, None);
    assert_eq!(without.cut_set_count, None);
    assert_eq!(without.cut_sets_omitted.as_deref(), Some("not requested"));
}

#[test]
fn a_diagram_over_its_budget_is_refused_by_name() {
    let events = (0..6).map(|i| constant(&format!("E{i}"), 0.1)).collect();
    let top = Formula::at_least(3, (0..6).map(e).collect());
    let settings = QuantifySettings {
        max_bdd_nodes: 4,
        engine: raichu_fta::Engine::Exact,
        ..Default::default()
    };
    let err = quantify(&tree(events, top), &settings).unwrap_err();
    assert!(err.to_string().contains("max_bdd_nodes"), "{err}");
}

#[test]
fn constant_tops_are_certain_or_impossible() {
    let t = tree(
        vec![constant("A", 0.4)],
        Formula::or(vec![e(0), Formula::not(e(0))]),
    );
    let result = run(&t);
    assert_eq!(result.probability, 1.0);
    assert_eq!(result.minimal_cut_sets, Some(vec![vec![]]));
}

// ---------------------------------------------------------------------
// OpenPSA.
// ---------------------------------------------------------------------

#[test]
fn an_open_psa_document_is_read_with_its_connectives_and_laws() {
    let text = r#"<?xml version="1.0"?>
<opsa-mef>
  <define-fault-tree name="demo">
    <define-gate name="TOP">
      <or><gate name="G1"/><basic-event name="V"/><house-event name="H"/></or>
    </define-gate>
    <define-gate name="G1">
      <atleast min="2"><basic-event name="P1"/><basic-event name="P2"/><basic-event name="P3"/></atleast>
    </define-gate>
  </define-fault-tree>
  <model-data>
    <define-parameter name="lambda"><float value="1e-3"/></define-parameter>
    <define-basic-event name="P1"><exponential><parameter name="lambda"/><system-mission-time/></exponential></define-basic-event>
    <define-basic-event name="P2"><exponential><parameter name="lambda"/><system-mission-time/></exponential></define-basic-event>
    <define-basic-event name="P3"><exponential><parameter name="lambda"/><system-mission-time/></exponential></define-basic-event>
    <define-basic-event name="V"><mul><float value="2"/><float value="0.005"/></mul></define-basic-event>
    <define-house-event name="H"><constant value="false"/></define-house-event>
  </model-data>
</opsa-mef>"#;
    let tree = read_open_psa(text, None).unwrap();
    assert_eq!(tree.name, "demo");
    let settings = QuantifySettings {
        mission_time: Some(100.0),
        ..Default::default()
    };
    let result = quantify(&tree, &settings).unwrap();
    let p: f64 = 1.0 - (-0.1f64).exp();
    let vote = 3.0 * p * p - 2.0 * p * p * p;
    close(result.probability, 1.0 - (1.0 - vote) * (1.0 - 0.01));

    // Written and read back, it quantifies to the same bits.
    let again = read_open_psa(&write_open_psa(&tree), None).unwrap();
    assert_eq!(
        quantify(&again, &settings).unwrap().probability,
        result.probability
    );
}

#[test]
fn derived_connectives_are_rewritten() {
    let doc = |formula: &str| {
        format!(
            r#"<opsa-mef><define-gate name="T">{formula}</define-gate>
            <define-basic-event name="A"><float value="0.2"/></define-basic-event>
            <define-basic-event name="B"><float value="0.7"/></define-basic-event></opsa-mef>"#
        )
    };
    let p = |formula: &str| run(&read_open_psa(&doc(formula), None).unwrap()).probability;
    let (a, b) = (0.2, 0.7);
    close(
        p(r#"<nand><basic-event name="A"/><basic-event name="B"/></nand>"#),
        1.0 - a * b,
    );
    close(
        p(r#"<nor><basic-event name="A"/><basic-event name="B"/></nor>"#),
        (1.0 - a) * (1.0 - b),
    );
    close(
        p(r#"<xor><basic-event name="A"/><basic-event name="B"/></xor>"#),
        a * (1.0 - b) + (1.0 - a) * b,
    );
    close(
        p(r#"<iff><basic-event name="A"/><basic-event name="B"/></iff>"#),
        a * b + (1.0 - a) * (1.0 - b),
    );
    close(
        p(r#"<imply><basic-event name="A"/><basic-event name="B"/></imply>"#),
        1.0 - a * (1.0 - b),
    );
}

#[test]
fn what_is_not_read_is_refused_by_name() {
    let doc = r#"<opsa-mef><define-gate name="T"><or><basic-event name="A"/></or></define-gate>
        <define-basic-event name="A"><lognormal-deviate><float value="1"/><float value="2"/><float value="0.9"/></lognormal-deviate></define-basic-event></opsa-mef>"#;
    let err = read_open_psa(doc, None).unwrap_err();
    assert!(err.to_string().contains("lognormal-deviate"), "{err}");

    let two_roots = r#"<opsa-mef><define-gate name="T1"><basic-event name="A"/></define-gate>
        <define-gate name="T2"><basic-event name="A"/></define-gate>
        <define-basic-event name="A"><float value="0.1"/></define-basic-event></opsa-mef>"#;
    let err = read_open_psa(two_roots, None).unwrap_err();
    assert!(err.to_string().contains("T1, T2"), "{err}");
    assert!(read_open_psa(two_roots, Some("T2")).is_ok());
}

#[test]
fn a_law_openpsa_cannot_express_round_trips_through_attributes() {
    let events = vec![BasicEvent {
        name: "seal".to_owned(),
        law: Law::Lognormal {
            mu: 4.0,
            sigma: 0.5,
        },
    }];
    let original = Tree {
        name: "t".to_owned(),
        events,
        gates: vec![GateDef {
            name: "G".to_owned(),
            formula: e(0),
        }],
        top: Formula::Gate { gate: 0 },
    };
    let back = read_open_psa(&write_open_psa(&original), None).unwrap();
    assert_eq!(back.events, original.events);
}

// ---------------------------------------------------------------------
// Findings of the review, each pinned by its reproducer.
// ---------------------------------------------------------------------

#[test]
fn a_tautological_module_is_a_constant_not_a_variable() {
    // atleast(2, [c, not c, d, not d]) is always true: the top is certain,
    // and its only minimal cut set is the empty one.
    let events = ["A", "B", "C", "D"]
        .iter()
        .map(|n| constant(n, 0.3))
        .collect();
    let tautology = Formula::at_least(2, vec![e(2), Formula::not(e(2)), e(3), Formula::not(e(3))]);
    let result = run(&tree(events, Formula::or(vec![e(0), e(1), tautology])));
    assert_eq!(result.probability, 1.0);
    assert!(result.coherent);
    assert_eq!(result.cut_set_count, Some(1));
    assert_eq!(result.minimal_cut_sets, Some(vec![vec![]]));
}

#[test]
fn an_absorbed_module_neither_expands_nor_breaks_coherence() {
    // e0 + e0.(e1 + not e2) = e0: coherent, one cut set.
    let events = ["A", "B", "C"].iter().map(|n| constant(n, 0.3)).collect();
    let top = Formula::or(vec![
        e(0),
        Formula::and(vec![e(0), Formula::or(vec![e(1), Formula::not(e(2))])]),
    ]);
    let result = run(&tree(events, top));
    assert!(result.coherent);
    assert_eq!(result.minimal_cut_sets, Some(vec![vec![0]]));

    // e0 + e0.(M + z) with M = AND of 24 (x + y): M alone has 2^24 cut
    // sets, none of which is minimal for the top.
    let mut events = vec![constant("e0", 0.1), constant("z", 0.1)];
    let mut factors = Vec::new();
    for i in 0..24 {
        events.push(constant(&format!("x{i}"), 0.1));
        events.push(constant(&format!("y{i}"), 0.1));
        factors.push(Formula::or(vec![e(2 + 2 * i), e(3 + 2 * i)]));
    }
    let top = Formula::or(vec![
        e(0),
        Formula::and(vec![e(0), Formula::or(vec![Formula::and(factors), e(1)])]),
    ]);
    let result = run(&tree(events, top));
    assert_eq!(result.cut_set_count, Some(1));
    assert_eq!(result.minimal_cut_sets, Some(vec![vec![0]]));
}

#[test]
fn a_parameter_that_reaches_itself_is_refused() {
    let doc = r#"<opsa-mef><define-gate name="T"><basic-event name="A"/></define-gate>
        <define-parameter name="p"><parameter name="q"/></define-parameter>
        <define-parameter name="q"><parameter name="p"/></define-parameter>
        <define-basic-event name="A"><parameter name="p"/></define-basic-event></opsa-mef>"#;
    let err = read_open_psa(doc, None).unwrap_err();
    assert!(
        err.to_string().contains("parameter cycle p -> q -> p"),
        "{err}"
    );
}

#[test]
fn an_n_ary_xor_and_semantics_altering_elements_are_refused() {
    let head = r#"<opsa-mef><define-basic-event name="A"><float value="0.1"/></define-basic-event>
        <define-basic-event name="B"><float value="0.1"/></define-basic-event>
        <define-basic-event name="C"><float value="0.1"/></define-basic-event>"#;
    let xor3 = format!(
        r#"{head}<define-gate name="T"><xor><basic-event name="A"/><basic-event name="B"/><basic-event name="C"/></xor></define-gate></opsa-mef>"#
    );
    assert!(read_open_psa(&xor3, None)
        .unwrap_err()
        .to_string()
        .contains("<xor> takes two"));
    let ccf = format!(
        r#"{head}<define-gate name="T"><or><basic-event name="A"/></or></define-gate><define-CCF-group name="G" model="beta-factor"/></opsa-mef>"#
    );
    assert!(read_open_psa(&ccf, None)
        .unwrap_err()
        .to_string()
        .contains("define-CCF-group"));
    let house = format!(
        r#"{head}<define-gate name="T"><or><house-event name="H"/></or></define-gate><define-house-event name="H"/></opsa-mef>"#
    );
    assert!(read_open_psa(&house, None)
        .unwrap_err()
        .to_string()
        .contains("house event `H`"));
}

#[test]
fn an_empirical_table_is_checked_where_it_is_read() {
    let bad = Law::Empirical {
        points: vec![(1.0, 0.2), (2.0, f64::NAN)],
    };
    assert!(bad.probability("e", Some(1.5)).is_err());
    let short = Law::Empirical {
        points: vec![(1.0, 0.2), (2.0, 0.9)],
    };
    assert!(short.probability("e", Some(1.5)).is_err());
}

#[test]
fn a_flat_gate_of_many_events_and_a_long_gate_chain_do_not_overflow() {
    let n = 100_000;
    let events: Vec<BasicEvent> = (0..n).map(|i| constant(&format!("E{i}"), 1e-6)).collect();
    let flat = tree(events.clone(), Formula::or((0..n).map(e).collect()));
    let settings = QuantifySettings {
        cut_sets: false,
        ..Default::default()
    };
    let result = quantify(&flat, &settings).unwrap();
    let exact = -(n as f64 * (-1e-6f64).ln_1p()).exp_m1();
    assert!(
        (result.probability - exact).abs() < 1e-12,
        "{}",
        result.probability
    );

    // G_i = E_i + G_{i+1}: a chain of gates, each referencing the next.
    let gates: Vec<GateDef> = (0..n)
        .map(|i| GateDef {
            name: format!("G{i}"),
            formula: if i + 1 < n {
                Formula::or(vec![e(i), Formula::Gate { gate: i + 1 }])
            } else {
                e(i)
            },
        })
        .collect();
    let chain = Tree {
        name: "chain".to_owned(),
        events,
        gates,
        top: Formula::Gate { gate: 0 },
    };
    let result = quantify(&chain, &settings).unwrap();
    assert!((result.probability - exact).abs() < 1e-12);
}

#[test]
fn the_reduction_worth_of_a_sole_cause_survives_cancellation() {
    // Top = A . (B + not B) simplifies to A; a residue of P - pB must not
    // turn the infinite reduction worth into a finite one, nor the other
    // way round for B.
    let result = run(&tree(
        vec![constant("A", 0.37), constant("B", 0.11)],
        Formula::and(vec![e(0), Formula::or(vec![e(1), Formula::not(e(1))])]),
    ));
    assert_eq!(result.importance[0].risk_reduction_worth, None);
}

// ---------------------------------------------------------------------
// Brute force.
// ---------------------------------------------------------------------

fn eval(formula: &Formula, tree: &Tree, x: &[bool]) -> bool {
    match formula {
        Formula::Constant { value } => *value,
        Formula::Event { event } => x[*event],
        Formula::Gate { gate } => eval(&tree.gates[*gate].formula, tree, x),
        Formula::And { args } => args.iter().all(|a| eval(a, tree, x)),
        Formula::Or { args } => args.iter().any(|a| eval(a, tree, x)),
        Formula::AtLeast { k, args } => args.iter().filter(|a| eval(a, tree, x)).count() >= *k,
        Formula::Not { arg } => !eval(arg, tree, x),
    }
}

fn brute_probability(tree: &Tree, p: &[f64], fixed: Option<(usize, bool)>) -> f64 {
    let n = p.len();
    let mut total = 0.0;
    for mask in 0u32..(1 << n) {
        let x: Vec<bool> = (0..n).map(|i| mask & (1 << i) != 0).collect();
        if let Some((i, v)) = fixed {
            if x[i] != v {
                continue;
            }
        }
        let weight: f64 = (0..n)
            .filter(|&i| fixed.is_none_or(|(j, _)| j != i))
            .map(|i| if x[i] { p[i] } else { 1.0 - p[i] })
            .product();
        if eval(&tree.top, tree, &x) {
            total += weight;
        }
    }
    total
}

fn brute_cut_sets(tree: &Tree, n: usize) -> Vec<Vec<usize>> {
    let solutions: Vec<u32> = (0u32..(1 << n))
        .filter(|&mask| {
            let x: Vec<bool> = (0..n).map(|i| mask & (1 << i) != 0).collect();
            eval(&tree.top, tree, &x)
        })
        .collect();
    let mut minimal: Vec<Vec<usize>> = solutions
        .iter()
        .filter(|&&s| !solutions.iter().any(|&t| t != s && t & s == t))
        .map(|&s| (0..n).filter(|i| s & (1 << i) != 0).collect())
        .collect();
    minimal.sort_by(|a: &Vec<usize>, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    minimal
}

fn brute_is_monotone(tree: &Tree, n: usize) -> bool {
    let value = |mask: u32| {
        let x: Vec<bool> = (0..n).map(|i| mask & (1 << i) != 0).collect();
        eval(&tree.top, tree, &x)
    };
    (0u32..(1 << n))
        .all(|mask| (0..n).all(|i| mask & (1 << i) != 0 || !value(mask) || value(mask | (1 << i))))
}

fn formula_strategy(n_events: usize, n_gates: usize, negations: bool) -> BoxedStrategy<Formula> {
    let mut leaves = vec![
        (0..n_events).prop_map(Formula::event).boxed(),
        (0..n_events).prop_map(Formula::event).boxed(),
        (0..n_events).prop_map(Formula::event).boxed(),
        any::<bool>()
            .prop_map(|value| Formula::Constant { value })
            .boxed(),
    ];
    if n_gates > 0 {
        leaves.push((0..n_gates).prop_map(|gate| Formula::Gate { gate }).boxed());
    }
    let leaf = proptest::strategy::Union::new(leaves);
    leaf.prop_recursive(3, 24, 4, move |inner| {
        let args = prop::collection::vec(inner.clone(), 1..4);
        let mut options = vec![
            args.clone().prop_map(Formula::and).boxed(),
            args.clone().prop_map(Formula::or).boxed(),
            (args, 0usize..4)
                .prop_map(|(a, k)| Formula::at_least(k.min(a.len() + 1), a))
                .boxed(),
        ];
        if negations {
            options.push(inner.prop_map(Formula::not).boxed());
        }
        proptest::strategy::Union::new(options)
    })
    .boxed()
}

/// A tree whose gate `g` references only gates of lower index, so no cycle.
fn tree_strategy(negations: bool) -> impl Strategy<Value = (Tree, Vec<f64>)> {
    (2usize..7, 0usize..4).prop_flat_map(move |(n_events, n_gates)| {
        let gates: Vec<BoxedStrategy<Formula>> = (0..n_gates)
            .map(|g| formula_strategy(n_events, g, negations))
            .collect();
        (
            gates,
            formula_strategy(n_events, n_gates, negations),
            prop::collection::vec(0.01f64..0.99, n_events),
        )
            .prop_map(move |(gates, top, p)| {
                let tree = Tree {
                    name: "random".to_owned(),
                    events: p
                        .iter()
                        .enumerate()
                        .map(|(i, q)| constant(&format!("E{i}"), *q))
                        .collect(),
                    gates: gates
                        .into_iter()
                        .enumerate()
                        .map(|(i, formula)| GateDef {
                            name: format!("G{i}"),
                            formula,
                        })
                        .collect(),
                    top,
                };
                (tree, p)
            })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn probability_and_birnbaum_match_the_truth_table((tree, p) in tree_strategy(true)) {
        let result = run(&tree);
        let exact = brute_probability(&tree, &p, None);
        prop_assert!((result.probability - exact).abs() < 1e-12, "{} vs {}", result.probability, exact);
        for (i, importance) in result.importance.iter().enumerate() {
            let b = brute_probability(&tree, &p, Some((i, true)))
                - brute_probability(&tree, &p, Some((i, false)));
            prop_assert!((importance.birnbaum - b).abs() < 1e-12, "event {i}: {} vs {b}", importance.birnbaum);
        }
    }

    #[test]
    fn minimal_cut_sets_match_the_truth_table((tree, _p) in tree_strategy(false)) {
        let result = run(&tree);
        let exact = brute_cut_sets(&tree, tree.events.len());
        prop_assert!(result.coherent);
        prop_assert_eq!(result.cut_set_count, Some(exact.len() as u128));
        prop_assert_eq!(result.minimal_cut_sets, Some(exact));
    }

    /// With negations and constants: whenever the result says the top is
    /// coherent, its cut sets must be the truth table's minimal solutions.
    #[test]
    fn a_coherent_verdict_carries_the_true_cut_sets((tree, _p) in tree_strategy(true)) {
        let result = run(&tree);
        let monotone = brute_is_monotone(&tree, tree.events.len());
        prop_assert_eq!(result.coherent, monotone);
        if result.coherent {
            let exact = brute_cut_sets(&tree, tree.events.len());
            prop_assert_eq!(result.cut_set_count, Some(exact.len() as u128));
            prop_assert_eq!(result.minimal_cut_sets, Some(exact));
        }
    }

    #[test]
    fn written_and_read_back_a_tree_quantifies_to_the_same_bits((tree, _p) in tree_strategy(true)) {
        // Unreferenced gates are roots too, so the top is named.
        let back = read_open_psa(&write_open_psa(&tree), Some("top")).unwrap();
        prop_assert_eq!(run(&back).probability, run(&tree).probability);
    }
}
