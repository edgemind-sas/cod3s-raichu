//! Exact sequence-tree exploration: closed forms are the primary witness
//! (relative tolerance 1e-12), then cut-offs, bounds, determinism across
//! thread counts, the domain checks, the reduction of the result, and a
//! Monte-Carlo campaign on a repairable model.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{sequence::analyse, CompiledModel, Engine, EngineConfig, EngineError};
use raichu_explore::{
    exact_domain_report, explore_exact, read_exploration, DomainViolation, ExactSettings,
    ExplorationResult, ReadExplorationError, EXPLORATION_FORMAT, EXPLORATION_VERSION,
};
use raichu_expr::{BoolOp, Expr, StateRef, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Distrib, Equation, EquationKind, Model, Target,
    Transition, TransitionKind,
};

// ---- model literals ------------------------------------------------------

const TOL: f64 = 1e-12;

fn transition(name: &str, source: &str, targets: &[&str], distrib: Distrib) -> Transition {
    Transition {
        name: name.into(),
        source: source.into(),
        guard: None,
        targets: targets.iter().map(|t| (*t).into()).collect(),
        on_interruption: Default::default(),
        monitored: false,
        cycle_group: None,
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

fn exp_expr(rate_expr: Expr) -> Distrib {
    Distrib::Exp {
        rate: None,
        rate_expr: Some(rate_expr),
    }
}

fn constant(value: f64) -> Expr {
    Expr::Const {
        value: Value::Float(value),
    }
}

fn active(component: &str, automaton: &str, state: &str) -> Expr {
    Expr::StateActive {
        state: StateRef {
            component: component.into(),
            automaton: automaton.into(),
            state: state.into(),
        },
    }
}

fn down(component: &str) -> Expr {
    active(component, "fail", "nok")
}

fn all(args: Vec<Expr>) -> Expr {
    Expr::Bool {
        bool_op: BoolOp::And,
        args,
    }
}

fn any(args: Vec<Expr>) -> Expr {
    Expr::Bool {
        bool_op: BoolOp::Or,
        args,
    }
}

/// A component `name` with automaton `fail` (`ok`, `nok`): monitored
/// failure `occ` under `occ_law`, optional monitored repair `rep`, the two
/// forming a cycle pair. With `kinds`, `occ` is declared a failure and
/// `rep` a repair.
fn unit(name: &str, occ_law: Distrib, rep_law: Option<Distrib>, kinds: bool) -> Component {
    let mut occ = transition("occ", "ok", &["nok"], occ_law);
    occ.monitored = true;
    occ.cycle_group = Some(name.into());
    let mut transitions = vec![occ];
    if kinds {
        transitions[0].kind = Some(TransitionKind::Failure);
    }
    if let Some(law) = rep_law {
        let mut rep = transition("rep", "nok", &["ok"], law);
        rep.monitored = true;
        rep.cycle_group = Some(name.into());
        if kinds {
            rep.kind = Some(TransitionKind::Repair);
        }
        transitions.push(rep);
    }
    component(
        name,
        Automaton {
            name: "fail".into(),
            states: vec!["ok".into(), "nok".into()],
            init: "ok".into(),
            transitions,
        },
    )
}

/// The feared-event watcher `sys.watch`: an unmonitored instantaneous
/// transition to `down` as soon as `guard` holds.
fn watcher(guard: Expr) -> Component {
    let mut to_down = transition("down", "ok", &["down"], Distrib::Inst { probs: vec![] });
    to_down.guard = Some(guard);
    component(
        "sys",
        Automaton {
            name: "watch".into(),
            states: vec!["ok".into(), "down".into()],
            init: "ok".into(),
            transitions: vec![to_down],
        },
    )
}

fn sys_down() -> Target {
    Target {
        name: "sys_down".into(),
        component: "sys".into(),
        automaton: "watch".into(),
        state: "down".into(),
    }
}

fn model(name: &str, components: Vec<Component>, targets: Vec<Target>) -> Model {
    Model {
        name: name.into(),
        components,
        connections: vec![],
        indicators: vec![],
        targets,
        evaluation_order: None,
        unbounded_rate: None,
    }
}

fn compile(model: &Model) -> CompiledModel {
    CompiledModel::compile(model).unwrap()
}

fn settings(target: &str, horizon: f64) -> ExactSettings {
    ExactSettings::new(target, horizon)
}

fn assert_rel(actual: f64, expected: f64, what: &str) {
    let rel = ((actual - expected) / expected).abs();
    assert!(
        rel <= TOL,
        "{what}: {actual:e} vs closed form {expected:e} (relative {rel:e})"
    );
}

fn names(result: &ExplorationResult, k: usize) -> Vec<String> {
    result
        .sequence_steps(k)
        .unwrap()
        .map(|s| s.transition.clone())
        .collect()
}

/// `P(T_A < T_B <= t)` for independent `T_A ~ Exp(a)`, `T_B ~ Exp(b)`,
/// by direct integration: `a/(a+b) (1 - e^{-(a+b)t}) - e^{-bt} (1 - e^{-at})`.
fn first_then_second(a: f64, b: f64, t: f64) -> f64 {
    a / (a + b) * (-(-(a + b) * t).exp_m1()) - (-b * t).exp() * (-(-a * t).exp_m1())
}

/// Parallel pair A (rate a), B (rate b), non repairable, target "both
/// failed".
fn parallel_pair(a: f64, b: f64) -> Model {
    model(
        "parallel_pair",
        vec![
            unit("A", exp(a), None, false),
            unit("B", exp(b), None, false),
            watcher(all(vec![down("A"), down("B")])),
        ],
        vec![sys_down()],
    )
}

// ---- closed forms ----------------------------------------------------------

/// Covers AE1.
#[test]
fn parallel_pair_yields_both_orderings_with_their_closed_forms() {
    let (a, b, t) = (0.1, 0.2, 5.0);
    let compiled = compile(&parallel_pair(a, b));
    let result = explore_exact(&compiled, &settings("sys_down", t)).unwrap();

    assert_eq!(result.sequences.len(), 2);
    let a_then_b = first_then_second(a, b, t);
    let b_then_a = first_then_second(b, a, t);
    // Ranked by decreasing probability: B first (the faster one).
    assert!(b_then_a > a_then_b);
    assert_eq!(
        names(&result, 0),
        ["B.fail.occ", "A.fail.occ", "sys.watch.down"]
    );
    assert_eq!(
        names(&result, 1),
        ["A.fail.occ", "B.fail.occ", "sys.watch.down"]
    );
    assert_rel(result.sequences[0].probability, b_then_a, "B then A");
    assert_rel(result.sequences[1].probability, a_then_b, "A then B");
    // Monitored events only: the watcher is not monitored.
    let attrs: Vec<(&str, &str)> = result
        .sequence_events(0)
        .unwrap()
        .map(|e| (e.obj.as_str(), e.attr.as_str()))
        .collect();
    assert_eq!(attrs, [("B", "nok"), ("A", "nok")]);
    assert_eq!(result.sequences[0].end_cause, "sys_down");

    let total = (-(-a * t).exp_m1()) * (-(-b * t).exp_m1());
    assert_rel(result.lower, total, "lower bound");
    assert_eq!(result.lower, result.upper);
    assert!(!result.inconclusive);
    assert_eq!(result.imprecise_sequences, 0);
    assert!(!result.cutoff_tallies.min_probability.fired());
    assert_eq!(result.format, EXPLORATION_FORMAT);
    // An exact result stays at format version 1, byte for byte as before
    // the discretised algorithm raised the readable version.
    assert_eq!(result.version, 1);
    assert_eq!(EXPLORATION_VERSION, 2);
}

#[test]
fn series_pair_yields_two_single_failure_sequences() {
    let (a, b, t) = (0.3, 0.05, 2.0);
    let compiled = compile(&model(
        "series_pair",
        vec![
            unit("A", exp(a), None, false),
            unit("B", exp(b), None, false),
            watcher(any(vec![down("A"), down("B")])),
        ],
        vec![sys_down()],
    ));
    let result = explore_exact(&compiled, &settings("sys_down", t)).unwrap();

    assert_eq!(result.sequences.len(), 2);
    for k in 0..result.sequences.len() {
        assert_eq!(result.sequence_events(k).unwrap().count(), 1);
    }
    let one_failed = -(-(a + b) * t).exp_m1();
    assert_rel(
        result.sequences[0].probability,
        a / (a + b) * one_failed,
        "A first",
    );
    assert_rel(
        result.sequences[1].probability,
        b / (a + b) * one_failed,
        "B first",
    );
    assert_rel(result.lower, one_failed, "sum");
    assert_eq!(result.lower, result.upper);
}

/// A demand `D.req` that answers when A fails: `ko` with probability 0.3,
/// `parked` otherwise. With `kinds`, the draw is a declared failure (its
/// first target, `ko`, is the failure branch).
fn demand(kinds: bool) -> Component {
    let mut resolve = transition(
        "resolve",
        "idle",
        &["ko", "parked"],
        Distrib::Inst { probs: vec![0.3] },
    );
    resolve.guard = Some(down("A"));
    resolve.monitored = true;
    if kinds {
        resolve.kind = Some(TransitionKind::Failure);
    }
    component(
        "D",
        Automaton {
            name: "req".into(),
            states: vec!["idle".into(), "ko".into(), "parked".into()],
            init: "idle".into(),
            transitions: vec![resolve],
        },
    )
}

#[test]
fn an_instantaneous_branch_splits_the_sequence_mass() {
    let (a, t) = (0.2, 3.0);
    let compiled = compile(&model(
        "demand",
        vec![unit("A", exp(a), None, false), demand(false)],
        vec![Target {
            name: "demand_failed".into(),
            component: "D".into(),
            automaton: "req".into(),
            state: "ko".into(),
        }],
    ));
    let result = explore_exact(&compiled, &settings("demand_failed", t)).unwrap();

    assert_eq!(result.sequences.len(), 1);
    assert_eq!(names(&result, 0), ["A.fail.occ", "D.req.resolve"]);
    assert_eq!(result.sequence_steps(0).unwrap().nth(1).unwrap().to, "ko");
    assert_rel(
        result.sequences[0].probability,
        0.3 * (-(-a * t).exp_m1()),
        "failure branch",
    );
    // The parked branch is an absorbing leaf: it discards nothing.
    assert_eq!(result.lower, result.upper);
}

#[test]
fn a_rate_that_doubles_after_the_first_failure_gives_the_erlang_closed_form() {
    // Load sharing: each survivor fails at λ, then at 2λ once the other
    // is down. Both orderings see sojourn rates (2λ, 2λ): an Erlang(2, 2λ)
    // path, repeated rate included.
    let (lambda, t) = (0.3, 4.0);
    let shared = |other: &str| {
        exp_expr(Expr::If {
            cond: Box::new(down(other)),
            then: Box::new(constant(2.0 * lambda)),
            otherwise: Box::new(constant(lambda)),
        })
    };
    let compiled = compile(&model(
        "load_sharing",
        vec![
            unit("A", shared("B"), None, false),
            unit("B", shared("A"), None, false),
            watcher(all(vec![down("A"), down("B")])),
        ],
        vec![sys_down()],
    ));
    let result = explore_exact(&compiled, &settings("sys_down", t)).unwrap();

    let x = 2.0 * lambda * t;
    let erlang = -(-x).exp_m1() - x * (-x).exp();
    assert_eq!(result.sequences.len(), 2);
    for sequence in &result.sequences {
        assert_rel(sequence.probability, 0.5 * erlang, "each ordering");
    }
    assert_rel(result.lower, erlang, "total");
}

#[test]
fn a_dormant_spare_is_not_a_branch_and_nothing_is_nan() {
    // Cold standby: the spare S fails at rate 0 while A works, at s once
    // A is down. The only sequence is A then S.
    let (a, s, t) = (0.4, 0.1, 6.0);
    let compiled = compile(&model(
        "cold_standby",
        vec![
            unit("A", exp(a), None, false),
            unit(
                "S",
                exp_expr(Expr::If {
                    cond: Box::new(down("A")),
                    then: Box::new(constant(s)),
                    otherwise: Box::new(constant(0.0)),
                }),
                None,
                false,
            ),
            watcher(all(vec![down("A"), down("S")])),
        ],
        vec![sys_down()],
    ));
    let result = explore_exact(&compiled, &settings("sys_down", t)).unwrap();

    assert_eq!(result.sequences.len(), 1);
    assert_eq!(
        names(&result, 0),
        ["A.fail.occ", "S.fail.occ", "sys.watch.down"]
    );
    let hypo = 1.0 - (s * (-a * t).exp() - a * (-s * t).exp()) / (s - a);
    assert_rel(result.sequences[0].probability, hypo, "A then S");
    assert!(result.lower.is_finite() && result.upper.is_finite());
    assert!(result.relative_gap().is_finite());
    for sequence in &result.sequences {
        assert!(sequence.probability.is_finite() && sequence.error_bound.is_finite());
    }
}

// ---- the exact domain ------------------------------------------------------

fn weibull() -> Distrib {
    Distrib::Weibull {
        shape: 2.0,
        scale: 5.0,
    }
}

/// Covers AE2.
#[test]
fn a_weibull_armed_after_the_first_failure_stops_the_run_naming_it() {
    let mut c = unit("C", weibull(), None, false);
    c.automata[0].transitions[0].guard = Some(down("A"));
    let compiled = compile(&model(
        "weibull_armed",
        vec![
            unit("A", exp(0.1), None, false),
            unit("B", exp(0.2), None, false),
            c,
            watcher(all(vec![down("A"), down("B")])),
        ],
        vec![sys_down()],
    ));
    for threads in [Some(1), Some(4)] {
        let mut s = settings("sys_down", 5.0);
        s.threads = threads;
        let err = explore_exact(&compiled, &s).unwrap_err();
        match err {
            EngineError::LawOutsideExactDomain {
                transition,
                sequence,
                law,
            } => {
                assert_eq!(transition, "C.fail.occ");
                assert_eq!(sequence, ["A.fail.occ"]);
                assert!(law.contains("weibull"), "{law}");
            }
            other => panic!("expected LawOutsideExactDomain, got {other:?}"),
        }
    }
}

#[test]
fn a_weibull_that_is_never_armed_does_not_block() {
    // C's Weibull leaves a state nothing ever enters.
    let c = component(
        "C",
        Automaton {
            name: "fail".into(),
            states: vec!["ok".into(), "spare".into(), "nok".into()],
            init: "ok".into(),
            transitions: vec![transition("occ", "spare", &["nok"], weibull())],
        },
    );
    let (a, b, t) = (0.1, 0.2, 5.0);
    let mut m = parallel_pair(a, b);
    m.components.push(c);
    let result = explore_exact(&compile(&m), &settings("sys_down", t)).unwrap();
    assert_eq!(result.sequences.len(), 2);
    assert_rel(
        result.lower,
        (-(-a * t).exp_m1()) * (-(-b * t).exp_m1()),
        "total",
    );
}

#[test]
fn a_model_declaring_an_ode_is_refused_before_exploration() {
    let mut tank = unit("T", exp(0.1), None, false);
    tank.attributes.push(Attribute {
        name: "level".into(),
        kind: AttrKind::Float,
        init: Value::Float(0.0),
    });
    tank.equations.push(Equation {
        target: "level".into(),
        kind: EquationKind::Ode,
        expr: constant(1.0),
    });
    let compiled = compile(&model(
        "ode",
        vec![tank],
        vec![Target {
            name: "t_down".into(),
            component: "T".into(),
            automaton: "fail".into(),
            state: "nok".into(),
        }],
    ));
    let report = exact_domain_report(&compiled);
    assert_eq!(
        report,
        [DomainViolation::Ode {
            attribute: "T.level".into()
        }]
    );
    let err = explore_exact(&compiled, &settings("t_down", 1.0)).unwrap_err();
    match err {
        EngineError::OutsideExactDomain { reasons } => {
            assert_eq!(reasons.len(), 1);
            assert!(reasons[0].contains("T.level"), "{reasons:?}");
        }
        other => panic!("expected OutsideExactDomain, got {other:?}"),
    }
}

#[test]
fn the_domain_report_names_a_reachable_watched_transition_and_a_time_read() {
    let mut m = parallel_pair(0.1, 0.2);
    let mut w = unit("W", Distrib::Watched, None, false);
    w.attributes.push(Attribute {
        name: "x".into(),
        kind: AttrKind::Float,
        init: Value::Float(0.0),
    });
    w.automata[0].transitions[0].guard = Some(Expr::Cmp {
        cmp: raichu_expr::CmpOp::Ge,
        lhs: Box::new(Expr::attr("W", "x")),
        rhs: Box::new(constant(1.0)),
    });
    m.components.push(w);
    let mut clock = unit("K", exp(0.1), None, false);
    clock.automata[0].transitions[0].guard = Some(Expr::Cmp {
        cmp: raichu_expr::CmpOp::Ge,
        lhs: Box::new(Expr::Time),
        rhs: Box::new(constant(1.0)),
    });
    m.components.push(clock);
    let report = exact_domain_report(&compile(&m));
    assert!(
        report.contains(&DomainViolation::Watched {
            transition: "W.fail.occ".into()
        }),
        "{report:?}"
    );
    assert!(
        report.iter().any(
            |v| matches!(v, DomainViolation::ReadsTime { site } if site.contains("K.fail.occ"))
        ),
        "{report:?}"
    );
    assert!(exact_domain_report(&compile(&parallel_pair(0.1, 0.2))).is_empty());
}

// ---- cut-offs and bounds ---------------------------------------------------

/// Covers AE3.
#[test]
fn raising_the_minimal_probability_retains_fewer_and_widens_the_gap() {
    // A repairable pair: sequences with more failure and repair cycles
    // are less likely, so the retained set shrinks as the threshold rises.
    let compiled = compile(&repairable_pair(0.1, 1.0));
    let run = |threshold: f64| {
        let mut s = settings("sys_down", 10.0);
        s.cutoffs.min_probability = Some(threshold);
        explore_exact(&compiled, &s).unwrap()
    };
    let loose = run(1e-8);
    let tight = run(1e-6);

    assert!(!tight.sequences.is_empty());
    assert!(tight.sequences.len() < loose.sequences.len());
    assert!(tight.upper - tight.lower > loose.upper - loose.lower);
    assert!(tight.cutoff_tallies.min_probability.fired());
    assert!(tight.cutoff_tallies.min_probability.mass > 0.0);
    // The bounds nest: the tighter run brackets the looser one.
    assert!(tight.lower <= loose.lower * (1.0 + TOL));
    assert!(tight.upper >= loose.upper * (1.0 - TOL));
    for result in [&loose, &tight] {
        assert_rel(
            result.upper,
            result.lower + result.cutoff_tallies.total_mass(),
            "upper = lower + discarded",
        );
    }
    for sequence in &tight.sequences {
        assert!(sequence.probability >= 1e-6);
    }
}

/// A repairable pair: failures at `lambda`, repairs at `mu`, target "both
/// failed".
fn repairable_pair(lambda: f64, mu: f64) -> Model {
    model(
        "repairable_pair",
        vec![
            unit("A", exp(lambda), Some(exp(mu)), false),
            unit("B", exp(lambda), Some(exp(mu)), false),
            watcher(all(vec![down("A"), down("B")])),
        ],
        vec![sys_down()],
    )
}

fn capped(threads: Option<usize>) -> ExplorationResult {
    let compiled = compile(&repairable_pair(0.1, 1.0));
    let mut s = settings("sys_down", 1000.0);
    s.cutoffs.max_branches = Some(200);
    s.threads = threads;
    explore_exact(&compiled, &s).unwrap()
}

/// Covers AE4.
#[test]
fn a_capped_repairable_system_over_a_long_horizon_is_inconclusive() {
    let result = capped(None);
    assert!(!result.sequences.is_empty());
    assert!(result.lower > 0.0 && result.lower < result.upper);
    assert!(result.cutoff_tallies.max_branches.fired());
    assert!(result.expanded_nodes <= 200);
    assert!(result.relative_gap() > result.gap_tolerance);
    assert!(result.inconclusive);
}

#[test]
fn the_capped_run_is_bit_identical_with_one_and_four_threads() {
    let one = capped(Some(1));
    let four = capped(Some(4));
    assert!(one.cutoff_tallies.max_branches.fired());
    assert_eq!(one.lower.to_bits(), four.lower.to_bits());
    assert_eq!(one.upper.to_bits(), four.upper.to_bits());
    assert_eq!(
        one.cutoff_tallies.max_branches.mass.to_bits(),
        four.cutoff_tallies.max_branches.mass.to_bits()
    );
    assert_eq!(one.cutoff_tallies, four.cutoff_tallies);
    assert_eq!(one.expanded_nodes, four.expanded_nodes);
    assert_eq!(
        one.steps, four.steps,
        "the step table is thread-independent"
    );
    assert_eq!(one.sequences, four.sequences);
    assert_eq!(one, four);
}

#[test]
fn results_are_identical_with_one_and_four_threads() {
    let compiled = compile(&model(
        "repairable_triple",
        vec![
            unit("A", exp(0.3), Some(exp(1.0)), false),
            unit("B", exp(0.2), Some(exp(1.5)), false),
            unit("C", exp(0.1), Some(exp(2.0)), false),
            watcher(all(vec![down("A"), down("B"), down("C")])),
        ],
        vec![sys_down()],
    ));
    let run = |threads| {
        let mut s = settings("sys_down", 2.0);
        s.cutoffs.min_probability = Some(1e-7);
        s.threads = threads;
        explore_exact(&compiled, &s).unwrap()
    };
    let one = run(Some(1));
    let four = run(Some(4));
    assert!(one.sequences.len() > 3);
    assert_eq!(one.lower.to_bits(), four.lower.to_bits());
    assert_eq!(one.upper.to_bits(), four.upper.to_bits());
    for (x, y) in one.sequences.iter().zip(&four.sequences) {
        assert_eq!(x.probability.to_bits(), y.probability.to_bits());
    }
    assert_eq!(one, four);
}

#[test]
fn the_maximal_failures_cut_off_needs_declared_kinds() {
    let compiled = compile(&parallel_pair(0.1, 0.2));
    let mut s = settings("sys_down", 5.0);
    s.cutoffs.max_failures = Some(1);
    let err = explore_exact(&compiled, &s).unwrap_err();
    assert!(
        matches!(&err, EngineError::InvalidStudyParameter { parameter, .. } if parameter == "max_failures"),
        "{err:?}"
    );
}

#[test]
fn the_maximal_failures_cut_off_does_not_count_a_parked_on_demand_branch() {
    // A fails (a failure), the demand draws `ko` (a failure) or `parked`
    // (not one), B fails (a failure). Target: B down.
    let (a, b, t) = (0.3, 0.1, 4.0);
    let compiled = compile(&model(
        "failures",
        vec![
            unit("A", exp(a), None, true),
            demand(true),
            unit("B", exp(b), None, true),
        ],
        vec![Target {
            name: "b_down".into(),
            component: "B".into(),
            automaton: "fail".into(),
            state: "nok".into(),
        }],
    ));
    let run = |max: u64| {
        let mut s = settings("b_down", t);
        s.cutoffs.max_failures = Some(max);
        explore_exact(&compiled, &s).unwrap()
    };

    let one = run(1);
    assert_eq!(one.sequences.len(), 1);
    assert_eq!(names(&one, 0), ["B.fail.occ"]);
    assert_rel(
        one.sequences[0].probability,
        b / (a + b) * (-(-(a + b) * t).exp_m1()),
        "B first",
    );
    assert!(one.cutoff_tallies.max_failures.fired());

    // A, parked, B: two failures, the parked branch not counting.
    let two = run(2);
    assert_eq!(two.sequences.len(), 2);
    let parked = (0..two.sequences.len())
        .find(|&k| two.sequences[k].steps.len() == 3)
        .expect("the parked path is retained");
    assert_eq!(
        two.sequence_steps(parked).unwrap().nth(1).unwrap().to,
        "parked"
    );
    assert_rel(
        two.sequences[parked].probability,
        0.7 * first_then_second(a, b, t),
        "A, parked, B",
    );
    // A, ko, B carries three failures and is cut.
    assert!(two.cutoff_tallies.max_failures.fired());
    assert!((0..two.sequences.len())
        .all(|k| two.sequence_steps(k).unwrap().all(|step| step.to != "ko")));
}

#[test]
fn the_maximal_length_cut_off_prunes_longer_sequences() {
    let compiled = compile(&parallel_pair(0.1, 0.2));
    let mut s = settings("sys_down", 5.0);
    s.cutoffs.max_length = Some(2);
    let result = explore_exact(&compiled, &s).unwrap();
    // Both sequences need three firings (two failures and the watcher).
    assert!(result.sequences.is_empty());
    assert_eq!(result.lower, 0.0);
    assert!(result.cutoff_tallies.max_length.fired());
    assert!(result.upper > 0.0);
    assert!(result.inconclusive);
}

// ---- settings validation ---------------------------------------------------

#[test]
fn invalid_settings_are_refused_before_exploring() {
    let compiled = compile(&parallel_pair(0.1, 0.2));
    let cases: Vec<(&str, ExactSettings)> = vec![
        ("target", settings("no_such_target", 1.0)),
        ("horizon", settings("sys_down", -1.0)),
        ("horizon", settings("sys_down", f64::INFINITY)),
        ("threads", {
            let mut s = settings("sys_down", 1.0);
            s.threads = Some(0);
            s
        }),
        ("max_branches", {
            let mut s = settings("sys_down", 1.0);
            s.cutoffs.max_branches = Some(0);
            s
        }),
        ("min_probability", {
            let mut s = settings("sys_down", 1.0);
            s.cutoffs.min_probability = Some(0.0);
            s
        }),
        ("min_probability", {
            let mut s = settings("sys_down", 1.0);
            s.cutoffs.min_probability = Some(1.5);
            s
        }),
        ("gap_tolerance", {
            let mut s = settings("sys_down", 1.0);
            s.gap_tolerance = f64::NAN;
            s
        }),
        ("precision", {
            let mut s = settings("sys_down", 1.0);
            s.precision.rel_precision = 0.0;
            s
        }),
    ];
    for (parameter, s) in cases {
        let err = explore_exact(&compiled, &s).unwrap_err();
        assert!(
            matches!(&err, EngineError::InvalidStudyParameter { parameter: p, .. } if p == parameter),
            "{parameter}: {err:?}"
        );
    }
}

// ---- the result as data ------------------------------------------------------

#[test]
fn reducing_the_result_to_minimal_sequences_keeps_the_total_probability() {
    let compiled = compile(&repairable_pair(0.3, 1.0));
    let mut s = settings("sys_down", 3.0);
    s.cutoffs.min_probability = Some(1e-9);
    let result = explore_exact(&compiled, &s).unwrap();
    let sequences = result.to_sequences();
    assert_eq!(sequences.len(), result.sequences.len());
    assert!(sequences
        .iter()
        .all(|s| s.events.iter().all(|e| e.time == 0.0)));
    let reduced = analyse(sequences);
    assert!(reduced.len() < result.sequences.len());
    let total: f64 = reduced.iter().map(|s| s.weight).sum();
    assert!(
        ((total - result.lower) / result.lower).abs() <= TOL,
        "{total:e} vs {:e}",
        result.lower
    );
}

#[test]
fn the_result_round_trips_through_its_open_format() {
    let result = explore_exact(
        &compile(&parallel_pair(0.1, 0.2)),
        &settings("sys_down", 5.0),
    )
    .unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("\"format\":\"raichu.exploration\""));
    // The compact form: a step table, and sequences of indices into it.
    assert!(json.contains("\"steps\":[0,"), "{json}");
    let back = read_exploration(&json).unwrap();
    assert_eq!(back.steps, result.steps);
    assert_eq!(back.sequences.len(), result.sequences.len());
    for (x, y) in back.sequences.iter().zip(&result.sequences) {
        assert_eq!(x.steps, y.steps);
        // serde_json's default float parser is correct to the last ulp or
        // so, not bit-exact.
        assert!(((x.probability - y.probability) / y.probability).abs() <= 1e-15);
    }
}

#[test]
fn the_step_table_holds_each_distinct_step_once_in_model_order() {
    let compiled = compile(&repairable_pair(0.3, 1.0));
    let mut s = settings("sys_down", 3.0);
    s.cutoffs.min_probability = Some(1e-9);
    let result = explore_exact(&compiled, &s).unwrap();
    let fired: usize = result.sequences.iter().map(|q| q.steps.len()).sum();
    // Two units (fail, repair each) and the watcher: at most five
    // distinct steps, however many sequences repeat them.
    assert!(result.steps.len() <= 5, "{:?}", result.steps);
    assert!(fired > 10 * result.steps.len(), "{fired} fired steps");
    let mut seen: Vec<(&str, &str)> = result
        .steps
        .iter()
        .map(|s| (s.transition.as_str(), s.to.as_str()))
        .collect();
    seen.dedup();
    assert_eq!(seen.len(), result.steps.len(), "no duplicate entry");
    // Every step index is used, and the table follows the compiled order.
    let order: Vec<usize> = result
        .steps
        .iter()
        .map(|s| {
            compiled
                .transitions
                .iter()
                .position(|t| t.name == s.transition)
                .unwrap()
        })
        .collect();
    assert!(order.windows(2).all(|w| w[0] <= w[1]), "{order:?}");
    for id in 0..result.steps.len() as u32 {
        assert!(result.sequences.iter().any(|q| q.steps.contains(&id)));
    }
    // The watcher is not monitored: its step carries no event.
    let watch = result
        .steps
        .iter()
        .find(|s| s.transition == "sys.watch.down")
        .unwrap();
    assert!(watch.event.is_none());
}

#[test]
fn a_sequence_referring_outside_the_step_table_is_refused() {
    let result = explore_exact(
        &compile(&parallel_pair(0.1, 0.2)),
        &settings("sys_down", 5.0),
    )
    .unwrap();
    let mut broken = result.clone();
    broken.sequences[1].steps.push(result.steps.len() as u32);
    let err = read_exploration(&serde_json::to_string(&broken).unwrap()).unwrap_err();
    assert_eq!(
        err,
        ReadExplorationError::DanglingStep {
            sequence: 1,
            step: result.steps.len() as u32,
            table: result.steps.len(),
        }
    );
}

// ---- Monte-Carlo -------------------------------------------------------------

#[test]
fn a_monte_carlo_interval_meets_the_exploration_bounds() {
    let (lambda, mu, t) = (0.5, 1.0, 2.0);
    let m = repairable_pair(lambda, mu);
    let compiled_explore = compile(&m);
    let mut s = settings("sys_down", t);
    s.cutoffs.min_probability = Some(1e-10);
    let result = explore_exact(&compiled_explore, &s).unwrap();

    // Replicas driven directly on the engine (the Monte-Carlo driver's
    // configuration differs across releases): the proportion reaching the
    // target by `t`, with its 99 % normal interval.
    let compiled = compile(&m);
    let n: u64 = 100_000;
    let mut hits = 0_u64;
    for replica in 0..n {
        let config = EngineConfig {
            t_max: t,
            seed: 20_260_926,
            rng_stream: replica,
            stop_at_targets: true,
            sequences: true,
            ..EngineConfig::default()
        };
        let run = Engine::new(&compiled, config).unwrap().run().unwrap();
        let reached = run
            .sequence
            .and_then(|sequence| sequence.end_cause)
            .is_some_and(|cause| cause == "sys_down");
        hits += u64::from(reached);
    }
    let p = hits as f64 / n as f64;
    let half = 2.576 * (p * (1.0 - p) / n as f64).sqrt();
    let (low, high) = (p - half, p + half);
    assert!(
        low <= result.upper && high >= result.lower,
        "MC [{low}, {high}] vs exploration [{}, {}]",
        result.lower,
        result.upper
    );
    if result.upper - result.lower < 1e-6 {
        assert!(
            low <= result.lower && result.lower <= high,
            "MC [{low}, {high}] vs exploration lower {}",
            result.lower
        );
    }
}

// ---- deep paths and instantaneous cycles -------------------------------------

/// A repairable unit A (failure and repair at rate 1) beside a unit B whose
/// failure has rate 0 (never a branch): every node has exactly one child,
/// so the tree is a single path alternating A's failure and repair, cut
/// only by `max_length`. Target: B down, never reached.
fn endless_repairable_path() -> CompiledModel {
    compile(&model(
        "endless_repairs",
        vec![
            unit("A", exp(1.0), Some(exp(1.0)), false),
            unit("B", exp_expr(constant(0.0)), None, false),
        ],
        vec![Target {
            name: "b_down".into(),
            component: "B".into(),
            automaton: "fail".into(),
            state: "nok".into(),
        }],
    ))
}

/// Explored to depth `DEEP`: well past what a native recursion over the
/// 2 MiB rayon worker stack survives (it overflowed at about 1 150
/// levels).
const DEEP: usize = 3_001;

fn deep_run(threads: Option<usize>) -> ExplorationResult {
    let mut s = settings("b_down", 5_300.0);
    s.cutoffs.max_length = Some(DEEP);
    s.threads = threads;
    explore_exact(&endless_repairable_path(), &s).unwrap()
}

#[test]
fn a_path_three_thousand_firings_deep_is_explored_without_recursion() {
    for threads in [None, Some(2)] {
        let result = deep_run(threads);
        assert!(result.sequences.is_empty());
        assert_eq!(result.lower, 0.0);
        // Nodes at depths 0..=DEEP are all expanded; the child at depth
        // DEEP + 1 is cut by the length cut-off, with the probability that
        // the whole path completes by the horizon (Erlang(DEEP + 1, 1)).
        assert_eq!(result.expanded_nodes, DEEP as u64 + 1, "{threads:?}");
        assert!(result.cutoff_tallies.max_length.fired());
        assert!(result.upper > 0.5, "{}", result.upper);
    }
}

/// A component cycling `x -> y -> x` through two zero-delay transitions:
/// the cycle has constant mass, so no probability cut-off ever stops it.
fn zero_delay_cycle() -> CompiledModel {
    compile(&model(
        "zero_delay_cycle",
        vec![component(
            "C",
            Automaton {
                name: "loop".into(),
                states: vec!["x".into(), "y".into(), "z".into()],
                init: "x".into(),
                transitions: vec![
                    transition("go", "x", &["y"], Distrib::Delay { time: 0.0 }),
                    transition("back", "y", &["x"], Distrib::Delay { time: 0.0 }),
                ],
            },
        )],
        vec![Target {
            name: "c_z".into(),
            component: "C".into(),
            automaton: "loop".into(),
            state: "z".into(),
        }],
    ))
}

#[test]
fn a_zero_delay_cycle_is_a_typed_error_not_a_crash() {
    let compiled = zero_delay_cycle();
    for threads in [None, Some(2)] {
        let mut s = settings("c_z", 1.0);
        s.threads = threads;
        s.cutoffs.min_probability = Some(1e-12);
        let err = explore_exact(&compiled, &s).unwrap_err();
        match &err {
            EngineError::InstantaneousCycle {
                transition,
                firings,
                sequence,
            } => {
                assert_eq!(
                    *firings,
                    raichu_core::EngineConfig::default().max_fixpoint_iterations
                );
                assert!(transition == "C.loop.go" || transition == "C.loop.back");
                // Nothing timed fired before the cycle started.
                assert!(sequence.is_empty(), "{sequence:?}");
            }
            other => panic!("expected InstantaneousCycle, got {other:?}"),
        }
    }
}

// ---- cut-off precedence and the failure cut-off's domain ----------------------

/// The parallel pair with its failures declared (`kind: failure`).
fn parallel_pair_with_kinds(a: f64, b: f64) -> Model {
    model(
        "parallel_pair_kinds",
        vec![
            unit("A", exp(a), None, true),
            unit("B", exp(b), None, true),
            watcher(all(vec![down("A"), down("B")])),
        ],
        vec![sys_down()],
    )
}

#[test]
fn the_minimal_probability_cut_off_takes_precedence_over_the_length_cut_off() {
    // Every root child has mass below 1 and length 1 > 0: both cut-offs
    // fire on it, and the mass goes to the minimal probability alone.
    let compiled = compile(&parallel_pair_with_kinds(0.1, 0.2));
    let mut s = settings("sys_down", 5.0);
    s.cutoffs.min_probability = Some(1.0);
    s.cutoffs.max_length = Some(0);
    s.cutoffs.max_failures = Some(0);
    let result = explore_exact(&compiled, &s).unwrap();
    let tallies = &result.cutoff_tallies;
    assert_eq!(tallies.min_probability.pruned_nodes, 2);
    assert!(!tallies.max_length.fired());
    assert!(!tallies.max_failures.fired());
    assert!(result.sequences.is_empty());
    assert_eq!(result.upper, tallies.min_probability.mass);
    // The two root children together: A or B fails by the horizon.
    assert_rel(
        tallies.min_probability.mass,
        -(-0.3f64 * 5.0).exp_m1(),
        "pruned mass",
    );
}

#[test]
fn the_length_cut_off_takes_precedence_over_the_failures_cut_off() {
    // Each root child is a failure (count 1 > 0) of length 1 > 0.
    let compiled = compile(&parallel_pair_with_kinds(0.1, 0.2));
    let mut s = settings("sys_down", 5.0);
    s.cutoffs.max_length = Some(0);
    s.cutoffs.max_failures = Some(0);
    let result = explore_exact(&compiled, &s).unwrap();
    let tallies = &result.cutoff_tallies;
    assert_eq!(tallies.max_length.pruned_nodes, 2);
    assert!(!tallies.max_failures.fired());
    assert!(!tallies.min_probability.fired());
    assert_eq!(result.upper, tallies.max_length.mass);
}

#[test]
fn the_maximal_failures_cut_off_is_refused_on_a_repair_only_model() {
    // A repair is declared, no failure: a failure count would count
    // nothing, so the cut-off is refused even though a kind is declared.
    let mut repair_only = unit("A", exp(0.1), Some(exp(1.0)), true);
    repair_only.automata[0].transitions[0].kind = None;
    let compiled = compile(&model(
        "repair_only",
        vec![
            repair_only,
            unit("B", exp(0.2), None, false),
            watcher(all(vec![down("A"), down("B")])),
        ],
        vec![sys_down()],
    ));
    let mut s = settings("sys_down", 5.0);
    s.cutoffs.max_failures = Some(1);
    let err = explore_exact(&compiled, &s).unwrap_err();
    match &err {
        EngineError::InvalidStudyParameter { parameter, detail } => {
            assert_eq!(parameter, "max_failures");
            assert!(detail.contains("`failure`"), "{detail}");
        }
        other => panic!("expected InvalidStudyParameter, got {other:?}"),
    }
}

// ---- a root that is already a leaf ------------------------------------------

#[test]
fn a_target_active_at_the_initial_state_is_one_empty_sequence_of_probability_one() {
    // The target latch is set by the initialization: the root itself is
    // the retained sequence, with no transition and probability 1.
    let compiled = compile(&model(
        "already_there",
        vec![unit("A", exp(0.1), None, false)],
        vec![Target {
            name: "a_ok".into(),
            component: "A".into(),
            automaton: "fail".into(),
            state: "ok".into(),
        }],
    ));
    let result = explore_exact(&compiled, &settings("a_ok", 5.0)).unwrap();
    assert_eq!(result.sequences.len(), 1);
    assert!(result.sequences[0].steps.is_empty());
    assert!(result.steps.is_empty());
    assert_eq!(result.sequences[0].probability, 1.0);
    assert_eq!(result.lower, 1.0);
    assert_eq!(result.upper, 1.0);
    assert_eq!(result.expanded_nodes, 0);
    assert!(!result.inconclusive);
}

#[test]
fn a_root_with_nothing_fireable_yields_an_empty_result() {
    // A's only failure has rate 0: the root is an absorbing leaf.
    let compiled = compile(&model(
        "frozen",
        vec![unit("A", exp_expr(constant(0.0)), None, false)],
        vec![Target {
            name: "a_down".into(),
            component: "A".into(),
            automaton: "fail".into(),
            state: "nok".into(),
        }],
    ));
    for threads in [None, Some(2)] {
        let mut s = settings("a_down", 5.0);
        s.threads = threads;
        let result = explore_exact(&compiled, &s).unwrap();
        assert!(result.sequences.is_empty());
        assert_eq!(result.lower, 0.0);
        assert_eq!(result.upper, 0.0);
        assert_eq!(result.expanded_nodes, 0);
        assert_eq!(result.cutoff_tallies, Default::default());
    }
}
