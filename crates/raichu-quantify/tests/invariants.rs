//! The quantification contract swept over random Markov models: every
//! envelope, and every exploration document inside one, reads back to the
//! same bits, and each kind of probability
//! keeps the invariants its construction promises.
//!
//! The intervals are not all clamped to `[0, 1]`: the Wilson interval is,
//! the exploration bounds are by construction, while the weighted normal
//! (cross-entropy) and Student (splitting) intervals are left as computed.
//! The invariants are therefore stated per kind, never once for all.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use proptest::prelude::*;
use raichu_explore::read_exploration;
use raichu_expr::{BoolOp, Expr, StateRef};
use raichu_model::{Automaton, Component, Distrib, Model, Target, Transition};
use raichu_montecarlo::ImportanceSource;
use raichu_quantify::{
    quantify, read_quantification, CrossEntropySamplingSettings, Detail,
    DiscretisedExplorationSettings, ExactExplorationSettings, Method, MonteCarloSettings,
    Quantification, SplittingSamplingSettings, Study, TargetProbability,
};

// ---- the random models ----------------------------------------------------

/// One unit of a random model: its failure rate, and its repair rate when
/// it is repairable.
#[derive(Clone, Debug)]
struct UnitSpec {
    failure: f64,
    repair: Option<f64>,
}

/// A random model: `units` in parallel, the feared event being any one of
/// `cut_sets` down at once (a monotone formula in disjunctive normal
/// form), studied over `horizon`.
#[derive(Clone, Debug)]
struct ModelSpec {
    units: Vec<UnitSpec>,
    cut_sets: Vec<Vec<usize>>,
    horizon: f64,
    seed: u64,
}

fn unit_spec() -> impl Strategy<Value = UnitSpec> {
    (0.05_f64..0.5, prop::option::weighted(0.5, 0.5_f64..2.0))
        .prop_map(|(failure, repair)| UnitSpec { failure, repair })
}

fn model_spec() -> impl Strategy<Value = ModelSpec> {
    (2_usize..=4).prop_flat_map(|n| {
        let cut_set = prop::sample::subsequence((0..n).collect::<Vec<_>>(), 1..=n);
        (
            prop::collection::vec(unit_spec(), n),
            prop::collection::vec(cut_set, 1..=3),
            1.0_f64..8.0,
            any::<u64>(),
        )
            .prop_map(|(units, cut_sets, horizon, seed)| ModelSpec {
                units,
                cut_sets,
                horizon,
                seed,
            })
    })
}

fn transition(name: &str, source: &str, target: &str, distrib: Distrib) -> Transition {
    Transition {
        name: name.into(),
        source: source.into(),
        guard: None,
        targets: vec![target.into()],
        on_interruption: Default::default(),
        monitored: false,
        monitored_states: None,
        cycle_group: None,
        cycle_cancellation: Default::default(),
        kind: None,
        effects: vec![],
        distrib,
    }
}

fn component(name: &str, automaton: Automaton) -> Component {
    Component {
        name: name.into(),
        attributes: vec![],
        ports: vec![],
        interfaces: vec![],
        automata: vec![automaton],
        allocations: vec![],
        equations: vec![],
        sensitive_functions: vec![],
    }
}

fn exp(rate: f64) -> Distrib {
    Distrib::Exp {
        rate: Some(rate),
        rate_expr: None,
    }
}

fn unit_name(i: usize) -> String {
    format!("U{i}")
}

fn unit(i: usize, spec: &UnitSpec) -> Component {
    let mut occ = transition("occ", "ok", "nok", exp(spec.failure));
    occ.monitored = true;
    let mut transitions = vec![occ];
    if let Some(rate) = spec.repair {
        transitions.push(transition("rep", "nok", "ok", exp(rate)));
    }
    component(
        &unit_name(i),
        Automaton {
            name: "fail".into(),
            states: vec!["ok".into(), "nok".into()],
            init: "ok".into(),
            transitions,
        },
    )
}

fn down(i: usize) -> Expr {
    Expr::StateActive {
        state: StateRef {
            component: unit_name(i),
            automaton: "fail".into(),
            state: "nok".into(),
        },
    }
}

fn all_of(args: Vec<Expr>, op: BoolOp) -> Expr {
    if args.len() == 1 {
        args.into_iter().next().unwrap()
    } else {
        Expr::Bool { bool_op: op, args }
    }
}

/// The model of `spec`: its units, and a watcher that latches into `down`
/// as soon as every unit of one cut set is down.
fn build(spec: &ModelSpec) -> Model {
    let feared = all_of(
        spec.cut_sets
            .iter()
            .map(|cut| all_of(cut.iter().map(|&i| down(i)).collect(), BoolOp::And))
            .collect(),
        BoolOp::Or,
    );
    let mut to_down = transition("down", "ok", "down", Distrib::Inst { probs: vec![] });
    to_down.guard = Some(feared);
    let watcher = component(
        "sys",
        Automaton {
            name: "watch".into(),
            states: vec!["ok".into(), "down".into()],
            init: "ok".into(),
            transitions: vec![to_down],
        },
    );
    let mut components: Vec<Component> = spec
        .units
        .iter()
        .enumerate()
        .map(|(i, u)| unit(i, u))
        .collect();
    components.push(watcher);
    Model {
        programs: vec![],
        name: "random_markov".into(),
        components,
        connections: vec![],
        indicators: vec![],
        targets: vec![Target {
            name: "feared".into(),
            component: "sys".into(),
            automaton: "watch".into(),
            state: "down".into(),
        }],
        evaluation_order: None,
        unbounded_rate: None,
        fmu_units: vec![],
        interface_connections: vec![],
    }
}

fn study(spec: &ModelSpec) -> Study {
    Study {
        target: "feared".into(),
        horizon: spec.horizon,
        instants: None,
        seed: spec.seed,
        threads: None,
    }
}

// ---- the methods ------------------------------------------------------------

fn monte_carlo() -> Method {
    Method::MonteCarlo(MonteCarloSettings::new(200))
}

/// With cut-offs: on a repairable model each repair reopens the tree, and
/// an exploration without any is not guaranteed to terminate (documented
/// in the sequence-tree exploration guide). The cut-offs also give the
/// bounds a gap, so the discarded mass and the inconclusive flag are
/// exercised.
fn exact() -> Method {
    Method::Exact(ExactExplorationSettings {
        min_probability: Some(1e-7),
        max_length: Some(12),
        max_branches: Some(20_000),
        ..ExactExplorationSettings::default()
    })
}

/// Tighter cut-offs than [`exact`]: the discretised tree grows as
/// `(K x competitors)^depth`, so the length and branch caps are what keep
/// a sweep fast.
fn discretised() -> Method {
    Method::Discretised(DiscretisedExplorationSettings {
        min_probability: Some(1e-7),
        max_length: Some(6),
        max_branches: Some(5_000),
        ..DiscretisedExplorationSettings::default()
    })
}

fn cross_entropy() -> Method {
    Method::CrossEntropy(CrossEntropySamplingSettings {
        pilot_runs: 200,
        ..CrossEntropySamplingSettings::new(400)
    })
}

fn splitting() -> Method {
    Method::Splitting(SplittingSamplingSettings {
        particles: 50,
        batches: 3,
        ..SplittingSamplingSettings::new(ImportanceSource::CutSets {
            max_cut_sets: 1_000,
        })
    })
}

// ---- the invariants -----------------------------------------------------------

/// The envelope reads back equal, and re-serialises to the same bytes.
/// The reader checks the probability against the detail on the way, so
/// this also proves the estimate, counts, levels and bounds consistent.
fn reads_back(q: &Quantification) -> Result<(), TestCaseError> {
    let json = q.to_json().unwrap();
    let back = read_quantification(&json)
        .map_err(|e| TestCaseError::fail(format!("{}: {e}", q.method.name())))?;
    prop_assert_eq!(&back, q);
    prop_assert_eq!(back.to_json().unwrap(), json);
    // An exploration is also a document of its own, read by its own
    // reader: it reads back to the same bits too (#94).
    if let Detail::Exploration(result) = &q.detail {
        let json = serde_json::to_string(result).unwrap();
        let back = read_exploration(&json)
            .map_err(|e| TestCaseError::fail(format!("{}: {e}", q.method.name())))?;
        prop_assert_eq!(&back, result.as_ref());
        prop_assert_eq!(serde_json::to_string(&back).unwrap(), json);
    }
    Ok(())
}

/// `a` and `b` equal up to the rounding of a sum of a few hundred terms.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
}

fn holds(q: &Quantification) -> Result<(), TestCaseError> {
    match (&q.probability, &q.detail) {
        (
            TargetProbability::ConfidenceInterval {
                estimate,
                reached,
                replicas,
                low,
                high,
                ..
            },
            _,
        ) => {
            prop_assert!(*reached <= *replicas);
            prop_assert_eq!(
                estimate.to_bits(),
                (*reached as f64 / *replicas as f64).to_bits()
            );
            prop_assert!(
                0.0 <= *low && low <= estimate && estimate <= high && *high <= 1.0,
                "Wilson [{low}, {high}] around {estimate}"
            );
        }
        (
            TargetProbability::Bounds {
                lower,
                upper,
                inconclusive,
                ..
            },
            Detail::Exploration(result),
        ) => {
            prop_assert!(
                0.0 <= *lower && lower <= upper && *upper <= 1.0,
                "bounds [{lower}, {upper}]"
            );
            let retained: f64 = result.sequences.iter().map(|s| s.probability).sum();
            prop_assert!(
                close(retained, *lower),
                "retained {retained:e} vs lower {lower:e}"
            );
            // The upper bound is the lower bound plus the discarded mass,
            // computed in that order: equal to the bit, whereas
            // `upper - lower` carries the rounding of `lower`.
            let discarded = result.cutoff_tallies.total_mass();
            prop_assert_eq!(upper.to_bits(), (lower + discarded).to_bits());
            let gap = if *upper > 0.0 {
                (upper - lower) / upper
            } else {
                0.0
            };
            prop_assert_eq!(*inconclusive, gap > result.gap_tolerance);
        }
        (
            TargetProbability::WeightedEstimate {
                estimate,
                standard_error,
                reached,
                replicas,
                low,
                high,
                effective_sample_size,
                inconclusive,
                ..
            },
            _,
        ) => {
            prop_assert!(*reached <= *replicas);
            prop_assert!(*estimate >= 0.0 && *standard_error >= 0.0);
            prop_assert!(
                low <= estimate && estimate <= high,
                "weighted [{low}, {high}] around {estimate}"
            );
            // Cauchy-Schwarz: (sum w)^2 / sum w^2 never exceeds the hit count.
            prop_assert!(
                0.0 <= *effective_sample_size
                    && *effective_sample_size <= *reached as f64 * (1.0 + 1e-12),
                "effective sample size {effective_sample_size} over {reached} hits"
            );
            let Method::CrossEntropy(settings) = &q.method else {
                panic!("a weighted estimate from {}", q.method.name());
            };
            prop_assert_eq!(
                *inconclusive,
                *effective_sample_size < settings.min_effective_sample_size
            );
        }
        (
            TargetProbability::SplittingEstimate {
                estimate,
                standard_error,
                low,
                high,
                batches,
                extinct_batches,
                inconclusive,
                ..
            },
            _,
        ) => {
            prop_assert!(*extinct_batches <= *batches);
            prop_assert!(
                (0.0..=1.0).contains(estimate),
                "splitting estimate {estimate}"
            );
            prop_assert!(*standard_error >= 0.0);
            prop_assert!(
                low <= estimate && estimate <= high,
                "Student [{low}, {high}] around {estimate}"
            );
            prop_assert_eq!(
                *inconclusive,
                *extinct_batches > 0 || *low <= 0.0 || (high - low) / 2.0 > *estimate
            );
        }
        (probability, _) => panic!("unexpected probability {probability:?}"),
    }
    Ok(())
}

fn check(spec: &ModelSpec, method: &Method) -> Result<Quantification, TestCaseError> {
    let q = quantify(&build(spec), &study(spec), method)
        .map_err(|e| TestCaseError::fail(format!("{}: {e}", method.name())))?;
    holds(&q)?;
    reads_back(&q)?;
    Ok(q)
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 32,
        ..ProptestConfig::default()
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn monte_carlo_keeps_its_invariants_and_reads_back(spec in model_spec()) {
        check(&spec, &monte_carlo())?;
    }

    #[test]
    fn exact_exploration_keeps_its_invariants_and_reads_back(spec in model_spec()) {
        check(&spec, &exact())?;
    }

    #[test]
    fn discretised_exploration_keeps_its_invariants_and_reads_back(spec in model_spec()) {
        check(&spec, &discretised())?;
    }

    #[test]
    fn cross_entropy_keeps_its_invariants_and_reads_back(spec in model_spec()) {
        check(&spec, &cross_entropy())?;
    }

    #[test]
    fn splitting_keeps_its_invariants_and_reads_back(spec in model_spec()) {
        check(&spec, &splitting())?;
    }
}
