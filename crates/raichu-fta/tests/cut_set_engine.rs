//! The cut-set engine: minimal cut sets under cutoffs, quantified by the
//! pivotal upper bound, with a guaranteed upper bound on the top event.
#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use proptest::prelude::*;
use raichu_fta::{
    quantify, BasicEvent, Engine, Formula, FtaError, GateDef, Law, QuantifySettings, Tree,
};

fn constant(name: &str, probability: f64) -> BasicEvent {
    BasicEvent {
        name: name.to_owned(),
        law: Law::Constant { probability },
    }
}

fn e(i: usize) -> Formula {
    Formula::event(i)
}

fn tree(events: Vec<BasicEvent>, top: Formula) -> Tree {
    Tree {
        name: "t".to_owned(),
        events,
        gates: Vec::new(),
        top,
    }
}

fn exact(t: &Tree) -> f64 {
    quantify(
        t,
        &QuantifySettings {
            engine: Engine::Exact,
            max_bdd_nodes: 50_000_000,
            ..Default::default()
        },
    )
    .unwrap()
    .probability
}

fn cut_set_engine() -> QuantifySettings {
    QuantifySettings {
        engine: Engine::CutSets,
        ..Default::default()
    }
}

/// `(a.b) + (a.c)` with probabilities near 1: the min-cut upper bound
/// double-counts `a`, the pivotal bound decomposes on it and is exact, as
/// Rauzy (2020) states and the Lambda Mu 2022 measurements show.
#[test]
fn the_three_estimators_separate_on_probabilities_near_one() {
    let (pa, pb, pc) = (0.9, 0.8, 0.7);
    let t = tree(
        vec![constant("a", pa), constant("b", pb), constant("c", pc)],
        Formula::or(vec![
            Formula::and(vec![e(0), e(1)]),
            Formula::and(vec![e(0), e(2)]),
        ]),
    );
    let exact_value = pa * (1.0 - (1.0 - pb) * (1.0 - pc));
    let result = quantify(&t, &cut_set_engine()).unwrap();
    assert!(!result.exact);
    assert_eq!(result.method, "cut_sets");
    let module = &result.provenance.modules[0];
    let pivotal = module.pivotal_upper_bound.unwrap();
    let mcub = module.mincut_upper_bound.unwrap();
    let rare = module.rare_event.unwrap();
    assert!(
        (pivotal - exact_value).abs() < 1e-15,
        "{pivotal} vs {exact_value}"
    );
    let mcub_exact = 1.0 - (1.0 - pa * pb) * (1.0 - pa * pc);
    assert!((mcub - mcub_exact).abs() < 1e-15);
    assert!(mcub > exact_value && rare > mcub);
    assert_eq!(result.probability, pivotal);
    // No cutoff removed anything: the bound is the retained family's.
    assert_eq!(module.neglected, Some(0.0));
    assert!(result.upper_bound.unwrap() >= exact_value - 1e-15);
    assert!(result.cut_sets_complete || !result.exact);
    assert_eq!(result.minimal_cut_sets, Some(vec![vec![0, 1], vec![0, 2]]));
}

/// A flat union of random order-3 cut sets: the worst case of the exact
/// engine. Lowering the probability cutoff keeps more sets and tightens
/// the guaranteed bound towards the exact value, which it never crosses.
#[test]
fn the_guaranteed_bound_tightens_as_the_cutoff_is_lowered() {
    let t = random_dnf(30, 40, 3, 7);
    let exact_value = exact(&t);
    let mut previous_gap = f64::INFINITY;
    for cutoff in [1e-5, 1e-7, 1e-9, 0.0] {
        let settings = QuantifySettings {
            min_cut_probability: cutoff,
            ..cut_set_engine()
        };
        let result = quantify(&t, &settings).unwrap();
        let upper = result.upper_bound.unwrap();
        assert!(
            upper >= exact_value * (1.0 - 1e-12),
            "{cutoff}: {upper} < {exact_value}"
        );
        let gap = upper - exact_value;
        assert!(
            gap <= previous_gap * (1.0 + 1e-12),
            "{cutoff}: {gap} > {previous_gap}"
        );
        previous_gap = gap;
    }
}

#[test]
fn a_module_over_the_budget_falls_back_and_says_so() {
    // Two independent parts: a small vote, and a flat random union the
    // budget cannot hold.
    let mut t = random_dnf(30, 40, 3, 11);
    let n = t.events.len();
    for i in 0..3 {
        t.events.push(constant(&format!("V{i}"), 0.01));
    }
    let big = std::mem::replace(&mut t.top, Formula::Constant { value: false });
    t.top = Formula::and(vec![Formula::at_least(2, (n..n + 3).map(e).collect()), big]);
    let exact_value = exact(&t);

    let settings = QuantifySettings {
        max_bdd_nodes: 2_000,
        ..Default::default()
    };
    let result = quantify(&t, &settings).unwrap();
    assert!(!result.exact);
    assert_eq!(result.method, "bdd+cut_sets");
    let methods: Vec<&str> = result.provenance.modules.iter().map(|m| m.method).collect();
    assert!(
        methods.contains(&"bdd") && methods.contains(&"cut_sets"),
        "{methods:?}"
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("outgrew max_bdd_nodes")),
        "{:?}",
        result.warnings
    );
    assert!(result.upper_bound.unwrap() >= exact_value * (1.0 - 1e-12));
    // The fallen-back module's value is its pivotal upper bound: on the
    // pessimistic side, within a percent here (40 cut sets sharing their
    // events), and tighter than the min-cut upper bound.
    assert!(result.probability >= exact_value * (1.0 - 1e-12));
    assert!(result.probability - exact_value <= 1e-2 * exact_value);
    let fallen = result
        .provenance
        .modules
        .iter()
        .find(|m| m.method == "cut_sets")
        .unwrap();
    let pivotal = fallen.pivotal_upper_bound.unwrap();
    assert!(pivotal <= fallen.mincut_upper_bound.unwrap());
    assert!(fallen.mincut_upper_bound.unwrap() <= fallen.rare_event.unwrap());

    let strict = QuantifySettings {
        engine: Engine::Exact,
        ..settings
    };
    assert!(matches!(quantify(&t, &strict), Err(FtaError::TooLarge(_))));
}

#[test]
fn the_cut_set_engine_refuses_a_negation_by_name() {
    let t = tree(
        vec![constant("a", 0.1), constant("b", 0.2)],
        Formula::or(vec![e(0), Formula::not(e(1))]),
    );
    let err = quantify(&t, &cut_set_engine()).unwrap_err();
    assert!(matches!(err, FtaError::NonCoherent(_)), "{err}");
    assert!(err.to_string().contains("`b`"));
}

#[test]
fn an_exhausted_expansion_budget_keeps_the_bound_valid() {
    let t = random_dnf(30, 40, 3, 5);
    let exact_value = exact(&t);
    let settings = QuantifySettings {
        max_expansions: 50,
        ..cut_set_engine()
    };
    let result = quantify(&t, &settings).unwrap();
    assert!(!result.cut_sets_complete);
    assert!(result.upper_bound.unwrap() >= exact_value * (1.0 - 1e-12));
    assert!(result.warnings.iter().any(|w| w.contains("max_expansions")));
}

#[test]
fn the_count_cutoff_keeps_the_most_probable_sets() {
    let events = (0..6)
        .map(|i| constant(&format!("E{i}"), 0.1 * (i + 1) as f64))
        .collect();
    // Six single-event cut sets; keep the three most probable.
    let t = tree(events, Formula::or((0..6).map(e).collect()));
    let settings = QuantifySettings {
        max_cut_sets: 3,
        ..cut_set_engine()
    };
    let result = quantify(&t, &settings).unwrap();
    assert_eq!(
        result.minimal_cut_sets,
        Some(vec![vec![3], vec![4], vec![5]])
    );
    assert!(!result.cut_sets_complete);
    let neglected = result.provenance.modules[0].neglected.unwrap();
    assert!((neglected - (0.1 + 0.2 + 0.3)).abs() < 1e-12, "{neglected}");
    assert!(result.warnings.iter().any(|w| w.contains("max_cut_sets")));
}

fn random_dnf(n_events: usize, n_cuts: usize, order: usize, seed: u64) -> Tree {
    // A small deterministic generator: the test must not depend on a
    // random crate's stream staying the same across versions.
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let events = (0..n_events)
        .map(|i| constant(&format!("E{i}"), 1e-4 + (next() % 1000) as f64 * 1e-5))
        .collect();
    let cuts = (0..n_cuts)
        .map(|_| {
            let mut members: Vec<usize> = Vec::new();
            while members.len() < order {
                let x = next() % n_events;
                if !members.contains(&x) {
                    members.push(x);
                }
            }
            Formula::and(members.into_iter().map(e).collect())
        })
        .collect();
    tree(events, Formula::or(cuts))
}

// ---------------------------------------------------------------------
// Against the truth table.
// ---------------------------------------------------------------------

fn eval(formula: &Formula, t: &Tree, x: &[bool]) -> bool {
    match formula {
        Formula::Constant { value } => *value,
        Formula::Event { event } => x[*event],
        Formula::Gate { gate } => eval(&t.gates[*gate].formula, t, x),
        Formula::And { args } => args.iter().all(|a| eval(a, t, x)),
        Formula::Or { args } => args.iter().any(|a| eval(a, t, x)),
        Formula::AtLeast { k, args } => args.iter().filter(|a| eval(a, t, x)).count() >= *k,
        Formula::Not { arg } => !eval(arg, t, x),
    }
}

fn brute(t: &Tree, p: &[f64]) -> f64 {
    let n = p.len();
    (0u32..(1 << n))
        .filter(|&mask| {
            let x: Vec<bool> = (0..n).map(|i| mask & (1 << i) != 0).collect();
            eval(&t.top, t, &x)
        })
        .map(|mask| {
            (0..n)
                .map(|i| {
                    if mask & (1 << i) != 0 {
                        p[i]
                    } else {
                        1.0 - p[i]
                    }
                })
                .product::<f64>()
        })
        .sum()
}

fn coherent_formula(n_events: usize, n_gates: usize) -> BoxedStrategy<Formula> {
    let mut leaves = vec![(0..n_events).prop_map(Formula::event).boxed()];
    if n_gates > 0 {
        leaves.push((0..n_gates).prop_map(|gate| Formula::Gate { gate }).boxed());
    }
    proptest::strategy::Union::new(leaves)
        .prop_recursive(3, 24, 4, |inner| {
            let args = prop::collection::vec(inner, 1..4);
            proptest::strategy::Union::new(vec![
                args.clone().prop_map(Formula::and).boxed(),
                args.clone().prop_map(Formula::or).boxed(),
                (args, 1usize..4)
                    .prop_map(|(a, k)| Formula::at_least(k.min(a.len()), a))
                    .boxed(),
            ])
        })
        .boxed()
}

fn coherent_tree() -> impl Strategy<Value = (Tree, Vec<f64>)> {
    (2usize..8, 0usize..4).prop_flat_map(|(n_events, n_gates)| {
        let gates: Vec<BoxedStrategy<Formula>> = (0..n_gates)
            .map(|g| coherent_formula(n_events, g))
            .collect();
        (
            gates,
            coherent_formula(n_events, n_gates),
            prop::collection::vec(0.001f64..0.999, n_events),
        )
            .prop_map(move |(gates, top, p)| {
                let t = Tree {
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
                (t, p)
            })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// Whatever the cutoffs, the certified bound is never below the exact
    /// value; without cutoffs, the pivotal bound is not either and the
    /// cut sets are the exact ones.
    #[test]
    fn the_certified_bound_is_never_below_the_truth(
        (t, p) in coherent_tree(),
        order in prop::option::of(0usize..4),
        cutoff in prop::sample::select(vec![0.0, 1e-3, 1e-2, 0.1, 0.5]),
        count in 1usize..6,
    ) {
        let truth = brute(&t, &p);
        let settings = QuantifySettings {
            max_order: order,
            min_cut_probability: cutoff,
            max_cut_sets: count,
            ..cut_set_engine()
        };
        let result = quantify(&t, &settings).unwrap();
        let upper = result.upper_bound.unwrap();
        prop_assert!(upper >= truth - 1e-12, "upper {} < truth {}", upper, truth);

        let full = quantify(&t, &cut_set_engine()).unwrap();
        prop_assert!(full.probability >= truth - 1e-12, "pivotal {} < truth {}", full.probability, truth);
        let exact_sets = quantify(&t, &QuantifySettings::default()).unwrap().minimal_cut_sets;
        prop_assert_eq!(full.minimal_cut_sets, exact_sets);
    }
}
