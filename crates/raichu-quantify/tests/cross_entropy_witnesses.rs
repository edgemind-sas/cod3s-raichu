//! The rare-event witnesses of the cross-entropy method.
//!
//! Two engines agreeing on a biased estimate cannot tell a right answer
//! from a shared mistake, and a rare probability has no reference
//! trajectory to compare. The primary witness is therefore a closed form:
//! the probability of the feared event is known exactly, and the method
//! must recover it inside its own interval at the declared level.
//!
//! - **Coverage** is judged over 100 seeds by a one-sided binomial test
//!   against the declared 95 % at alpha 0.01, with the mean standardised
//!   error checked near 0.
//! - Where the probability times the total replica budget is at most
//!   0.05, a plain Monte-Carlo campaign on the same budget sees no hit.
//! - Repairable systems are checked against the exact absorption
//!   probability of their Markov chain.
//! - In the region where a plain campaign converges too, the two estimates
//!   agree under a two-sample z-test.
//! - One regime is recorded as a limit: a highly reliable pair with fast
//!   repairs over a long horizon, where static per-family factors
//!   degenerate. The diagnostics must say so (a low effective sample
//!   size); the method is not claimed to be right there.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_expr::{BoolOp, Expr, StateRef};
use raichu_model::{Automaton, Component, Distrib, Model, Target, Transition, TransitionKind};
use raichu_quantify::{
    quantify, CrossEntropySamplingSettings, Method, MonteCarloSettings, Quantification, Study,
    TargetProbability,
};

// ---- models ----------------------------------------------------------------

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

fn exp(rate: f64) -> Distrib {
    Distrib::Exp {
        rate: Some(rate),
        rate_expr: None,
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

/// A unit failing at `lambda`, repaired at `mu` when given.
fn unit(name: &str, lambda: f64, mu: Option<f64>) -> Component {
    let mut occ = transition("occ", "ok", &["nok"], exp(lambda));
    occ.kind = Some(TransitionKind::Failure);
    let mut transitions = vec![occ];
    if let Some(mu) = mu {
        let mut rep = transition("rep", "nok", &["ok"], exp(mu));
        rep.kind = Some(TransitionKind::Repair);
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

fn down(c: &str) -> Expr {
    Expr::StateActive {
        state: StateRef {
            component: c.into(),
            automaton: "fail".into(),
            state: "nok".into(),
        },
    }
}

fn and(args: Vec<Expr>) -> Expr {
    Expr::Bool {
        bool_op: BoolOp::And,
        args,
    }
}

fn or(args: Vec<Expr>) -> Expr {
    Expr::Bool {
        bool_op: BoolOp::Or,
        args,
    }
}

fn system(name: &str, mut components: Vec<Component>, lost: Expr) -> Model {
    let mut loss = transition("loss", "ok", &["lost"], Distrib::Inst { probs: vec![] });
    loss.guard = Some(lost);
    components.push(component(
        "sys",
        Automaton {
            name: "watch".into(),
            states: vec!["ok".into(), "lost".into()],
            init: "ok".into(),
            transitions: vec![loss],
        },
    ));
    Model {
        programs: vec![],
        name: name.into(),
        components,
        connections: vec![],
        indicators: vec![],
        targets: vec![Target {
            name: "lost".into(),
            component: "sys".into(),
            automaton: "watch".into(),
            state: "lost".into(),
        }],
        evaluation_order: None,
        unbounded_rate: None,
        fmu_units: vec![],
        interface_connections: vec![],
    }
}

fn pair(lambda: f64, mu: Option<f64>) -> Model {
    system(
        "pair",
        vec![unit("A", lambda, mu), unit("B", lambda, mu)],
        and(vec![down("A"), down("B")]),
    )
}

fn two_out_of_three(lambda: f64) -> Model {
    system(
        "two_out_of_three",
        vec![
            unit("A", lambda, None),
            unit("B", lambda, None),
            unit("C", lambda, None),
        ],
        or(vec![
            and(vec![down("A"), down("B")]),
            and(vec![down("A"), down("C")]),
            and(vec![down("B"), down("C")]),
        ]),
    )
}

// ---- closed forms ------------------------------------------------------------

/// Probability a unit of rate `lambda` has failed by `t`.
fn q(lambda: f64, t: f64) -> f64 {
    -(-lambda * t).exp_m1()
}

/// Two identical repairable units, lost when both are down: the absorption
/// probability by `t` of the chain 0 -> 1 (rate 2λ), 1 -> 0 (μ), 1 -> 2 (λ).
/// `1 - [1, 0] exp(A t) [1, 1]'` with `A` the 2x2 transient generator,
/// whose eigenvalues are real and distinct.
fn repairable_pair_loss(lambda: f64, mu: f64, t: f64) -> f64 {
    let (a11, a12, a21, a22) = (-2.0 * lambda, 2.0 * lambda, mu, -(mu + lambda));
    let tr = a11 + a22;
    let det = a11 * a22 - a12 * a21;
    let disc = (tr * tr / 4.0 - det).sqrt();
    let (r1, r2) = (tr / 2.0 + disc, tr / 2.0 - disc);
    // exp(At) = (e^{r1 t}(A - r2 I) - e^{r2 t}(A - r1 I)) / (r1 - r2); the
    // first row summed over both columns is the survival.
    let (e1, e2) = ((r1 * t).exp(), (r2 * t).exp());
    let row = |r: f64| (a11 - r) + a12;
    let survival = (e1 * row(r2) - e2 * row(r1)) / (r1 - r2);
    1.0 - survival
}

// ---- helpers -------------------------------------------------------------------

fn study(horizon: f64, seed: u64) -> Study {
    Study {
        target: "lost".into(),
        horizon,
        instants: None,
        seed,
        threads: None,
    }
}

fn ce(nb_runs: u64, pilot_runs: u64) -> Method {
    Method::CrossEntropy(CrossEntropySamplingSettings {
        pilot_runs,
        ..CrossEntropySamplingSettings::new(nb_runs)
    })
}

struct Weighted {
    estimate: f64,
    se: f64,
    low: f64,
    high: f64,
    ess: f64,
    inconclusive: bool,
}

fn weighted(q: &Quantification) -> Weighted {
    match &q.probability {
        TargetProbability::WeightedEstimate {
            estimate,
            standard_error,
            low,
            high,
            effective_sample_size,
            inconclusive,
            ..
        } => Weighted {
            estimate: *estimate,
            se: *standard_error,
            low: *low,
            high: *high,
            ess: *effective_sample_size,
            inconclusive: *inconclusive,
        },
        other => panic!("expected a weighted estimate, got {other:?}"),
    }
}

/// Total replicas the campaign drew: every pilot plus the final one.
fn budget(q: &Quantification, pilot_runs: u64, nb_runs: u64) -> u64 {
    match &q.detail {
        raichu_quantify::Detail::CrossEntropy(d) => d.history.len() as u64 * pilot_runs + nb_runs,
        other => panic!("expected a cross-entropy detail, got {other:?}"),
    }
}

fn assert_within(w: &Weighted, truth: f64, what: &str) {
    let z = (w.estimate - truth) / w.se;
    assert!(
        z.abs() <= 4.0,
        "{what}: estimate {:e} is {z:.2} standard errors from {truth:e}",
        w.estimate
    );
}

/// `P(X <= k)` for `X ~ Binomial(n, p)`.
fn binomial_cdf(k: u64, n: u64, p: f64) -> f64 {
    let mut term = (1.0 - p).powi(n as i32); // P(X = 0)
    let mut sum = term;
    for i in 1..=k {
        term *= (n - i + 1) as f64 / i as f64 * p / (1.0 - p);
        sum += term;
    }
    sum
}

// ---- the witnesses ---------------------------------------------------------------

#[test]
fn the_interval_covers_a_one_in_a_million_event_at_its_declared_level() {
    const SEEDS: u64 = 100;
    const NB_RUNS: u64 = 2_000;
    const PILOT: u64 = 500;
    let lambda = 1e-4;
    let horizon = 10.0;
    let truth = q(lambda, horizon).powi(2);
    let model = pair(lambda, None);

    let mut covered = 0_u64;
    let mut z_sum = 0.0;
    let mut largest_budget = 0;
    for seed in 0..SEEDS {
        let result = quantify(&model, &study(horizon, seed), &ce(NB_RUNS, PILOT)).unwrap();
        let w = weighted(&result);
        if w.low <= truth && truth <= w.high {
            covered += 1;
        }
        z_sum += (w.estimate - truth) / w.se;
        largest_budget = largest_budget.max(budget(&result, PILOT, NB_RUNS));
    }
    let mean_z = z_sum / SEEDS as f64;
    println!("coverage {covered}/{SEEDS}, mean standardised error {mean_z:+.3}");
    // One-sided test of "the coverage is at least 95 %" at alpha 0.01.
    let p_value = binomial_cdf(covered, SEEDS, 0.95);
    assert!(
        p_value >= 0.01,
        "coverage {covered}/{SEEDS} rejects the declared 95 % (p = {p_value:.4})"
    );
    // Under a correct estimator the mean of 100 standard normals has a
    // standard deviation of 0.1.
    assert!(
        mean_z.abs() <= 0.4,
        "the estimator is biased: mean z {mean_z:+.3}"
    );

    // The same budget, drawn plainly, sees nothing: p * budget <= 0.05.
    assert!(truth * largest_budget as f64 <= 0.05);
    let plain = quantify(
        &model,
        &study(horizon, 0),
        &Method::MonteCarlo(MonteCarloSettings::new(largest_budget)),
    )
    .unwrap();
    match plain.probability {
        TargetProbability::ConfidenceInterval { reached, .. } => assert_eq!(reached, 0),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_two_out_of_three_system_is_recovered_at_three_rates() {
    let horizon = 10.0;
    for lambda in [1e-3, 1e-4, 1e-5] {
        let x = q(lambda, horizon);
        // At least two of three down.
        let truth = 3.0 * x * x * (1.0 - x) + x * x * x;
        let result = quantify(
            &two_out_of_three(lambda),
            &study(horizon, 1),
            &ce(4_000, 500),
        )
        .unwrap();
        assert_within(&weighted(&result), truth, &format!("2oo3 at {lambda:e}"));
    }
}

#[test]
fn repairable_pairs_match_their_markov_chain_short_and_long() {
    for (lambda, mu, horizon) in [(1e-3, 1.0, 10.0), (1e-4, 0.1, 1_000.0)] {
        let truth = repairable_pair_loss(lambda, mu, horizon);
        let result = quantify(
            &pair(lambda, Some(mu)),
            &study(horizon, 3),
            &ce(4_000, 1_000),
        )
        .unwrap();
        let w = weighted(&result);
        println!(
            "repairable pair λ={lambda:e} μ={mu} T={horizon}: {:e} vs {truth:e}, ESS {:.0}",
            w.estimate, w.ess
        );
        assert_within(
            &w,
            truth,
            &format!("repairable pair λ={lambda:e} μ={mu} T={horizon}"),
        );
        assert!(
            !w.inconclusive,
            "a recovered estimate is conclusive (ESS {:.0})",
            w.ess
        );
    }
}

#[test]
fn where_a_plain_campaign_converges_the_two_agree() {
    let (lambda, horizon) = (0.05, 10.0);
    let model = pair(lambda, None);
    let biased = weighted(&quantify(&model, &study(horizon, 5), &ce(20_000, 1_000)).unwrap());
    let plain = quantify(
        &model,
        &study(horizon, 6),
        &Method::MonteCarlo(MonteCarloSettings::new(20_000)),
    )
    .unwrap();
    let (p, n) = match plain.probability {
        TargetProbability::ConfidenceInterval {
            estimate, replicas, ..
        } => (estimate, replicas as f64),
        other => panic!("{other:?}"),
    };
    let plain_se = (p * (1.0 - p) / n).sqrt();
    let z = (biased.estimate - p) / (biased.se.powi(2) + plain_se.powi(2)).sqrt();
    assert!(
        z.abs() <= 4.0,
        "biased {:e} vs plain {p:e}: z = {z:.2}",
        biased.estimate
    );
    assert_within(&biased, q(lambda, horizon).powi(2), "overlap pair");
}

/// The recorded limit. A highly reliable pair (λ = 5e-5) with fast repairs
/// (μ = 1) over 1000 time units loses both units with a probability of
/// about 5e-6, reached through many repair cycles. Static per-family
/// factors cannot bias such a trajectory well: measured on 2026-09-28, the
/// estimate came 30 to 60 % low with an effective sample size of 8 to 35.
/// What is pinned is that the diagnostics flag it.
#[test]
fn the_regime_static_factors_cannot_handle_is_flagged_by_its_diagnostics() {
    let (lambda, mu, horizon) = (5e-5, 1.0, 1_000.0);
    let truth = repairable_pair_loss(lambda, mu, horizon);
    assert!((4e-6..6e-6).contains(&truth), "{truth:e}");
    let w = weighted(
        &quantify(
            &pair(lambda, Some(mu)),
            &study(horizon, 3),
            &ce(4_000, 1_000),
        )
        .unwrap(),
    );
    println!(
        "limit regime: {:e} vs {truth:e}, ESS {:.0}",
        w.estimate, w.ess
    );
    assert!(
        w.ess < 50.0,
        "the effective sample size ({:.0}) should flag this regime",
        w.ess
    );
    assert!(w.inconclusive, "the estimate must be marked inconclusive");
}

#[test]
fn the_markov_closed_form_matches_a_non_repairable_limit() {
    // With μ = 0 the chain is two independent failures: q².
    let (lambda, t) = (1e-2, 10.0);
    let chain = repairable_pair_loss(lambda, 0.0, t);
    let direct = q(lambda, t).powi(2);
    assert!(
        ((chain - direct) / direct).abs() < 1e-9,
        "{chain:e} vs {direct:e}"
    );
}
