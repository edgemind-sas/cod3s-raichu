//! Discretised sequence-tree exploration: closed forms first (the whole
//! mass of one law falling in its cells, a delay racing an exponential,
//! each interruption policy), then the agreement with the exact driver on
//! the Markov family, Monte-Carlo witnesses (a lognormal repair, a hybrid
//! model with an ODE and a watched threshold, the interruption policies),
//! the refinement estimate, cut-offs and determinism across thread counts.
//!
//! Monte-Carlo witnesses drive replicas on the engine directly (drawn
//! dates), with a 99 % normal interval on the proportion reaching the
//! target by the horizon.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError};
use raichu_explore::{
    explore_discretised, explore_exact, read_exploration, Algorithm, DiscretisedSettings,
    ExactSettings, ExplorationResult, ReadExplorationError, DEFAULT_LEVEL,
};
use raichu_model::Model;
use rayon::prelude::*;
use serde_json::{json, Value as Json};

// ---- model builders -------------------------------------------------------

fn c(value: f64) -> Json {
    json!({"op": "const", "value": {"kind": "float", "value": value}})
}

fn attr(component: &str, attribute: &str) -> Json {
    json!({"op": "attr", "attr": {"component": component, "attribute": attribute}})
}

fn active(component: &str, automaton: &str, state: &str) -> Json {
    json!({"op": "state_active",
           "state": {"component": component, "automaton": automaton, "state": state}})
}

fn down(component: &str) -> Json {
    active(component, "fail", "nok")
}

fn and(args: Vec<Json>) -> Json {
    json!({"op": "bool", "bool_op": "and", "args": args})
}

fn or(args: Vec<Json>) -> Json {
    json!({"op": "bool", "bool_op": "or", "args": args})
}

fn ite(cond: Json, then: Json, otherwise: Json) -> Json {
    json!({"op": "if", "cond": cond, "then": then, "otherwise": otherwise})
}

fn mul(a: Json, b: Json) -> Json {
    json!({"op": "mul", "args": [a, b]})
}

fn ge(lhs: Json, rhs: Json) -> Json {
    json!({"op": "cmp", "cmp": "ge", "lhs": lhs, "rhs": rhs})
}

fn tr(name: &str, source: &str, targets: &[&str], law: Json) -> Json {
    let mut t = json!({"name": name, "source": source, "targets": targets});
    for (k, v) in law.as_object().unwrap() {
        t[k] = v.clone();
    }
    t
}

fn with(mut t: Json, key: &str, value: Json) -> Json {
    t[key] = value;
    t
}

fn automaton(name: &str, states: &[&str], transitions: Vec<Json>) -> Json {
    json!({"name": name, "states": states, "init": states[0], "transitions": transitions})
}

fn component(name: &str, automata: Vec<Json>) -> Json {
    json!({"name": name, "ports": [], "attributes": [], "automata": automata, "equations": []})
}

fn exp(rate: f64) -> Json {
    json!({"distrib": "exp", "rate": rate})
}

fn exp_expr(rate: Json) -> Json {
    json!({"distrib": "exp", "rate_expr": rate})
}

fn delay(time: f64) -> Json {
    json!({"distrib": "delay", "time": time})
}

fn weibull(shape: f64, scale: f64) -> Json {
    json!({"distrib": "weibull", "shape": shape, "scale": scale})
}

fn inst() -> Json {
    json!({"distrib": "inst", "probs": []})
}

/// A non-repairable unit `name.fail` (`ok`, `nok`) failing under `law`,
/// monitored, optionally guarded.
fn unit(name: &str, law: Json, guard: Option<Json>) -> Json {
    let mut occ = with(tr("occ", "ok", &["nok"], law), "monitored", json!(true));
    if let Some(guard) = guard {
        occ = with(occ, "guard", guard);
    }
    component(name, vec![automaton("fail", &["ok", "nok"], vec![occ])])
}

/// The feared-event watcher `sys.watch`, entering `down` as soon as
/// `guard` holds.
fn watcher(guard: Json) -> Json {
    component(
        "sys",
        vec![automaton(
            "watch",
            &["ok", "down"],
            vec![with(tr("down", "ok", &["down"], inst()), "guard", guard)],
        )],
    )
}

fn target(name: &str, component: &str, automaton: &str, state: &str) -> Json {
    json!({"name": name, "component": component, "automaton": automaton, "state": state})
}

fn sys_down() -> Json {
    target("sys_down", "sys", "watch", "down")
}

fn build(name: &str, components: Vec<Json>, targets: Vec<Json>) -> CompiledModel {
    let document = json!({"name": name, "components": components, "targets": targets});
    let model = Model::from_json(&document.to_string()).unwrap();
    CompiledModel::compile(&model).unwrap()
}

// ---- helpers ----------------------------------------------------------------

fn settings(target: &str, horizon: f64, level: u32) -> DiscretisedSettings {
    let mut s = DiscretisedSettings::new(target, horizon);
    s.level = level;
    s
}

fn estimate(result: &ExplorationResult) -> f64 {
    result.error_estimate().expect("a refined result")
}

fn assert_rel(actual: f64, expected: f64, tol: f64, what: &str) {
    let rel = ((actual - expected) / expected).abs();
    assert!(
        rel <= tol,
        "{what}: {actual:e} vs {expected:e} (relative {rel:e}, tolerance {tol:e})"
    );
}

fn names(result: &ExplorationResult, k: usize) -> Vec<String> {
    result
        .sequence_steps(k)
        .unwrap()
        .map(|s| s.transition.clone())
        .collect()
}

/// The 99 % normal interval of the proportion of `n` drawn-date replicas
/// reaching `target` by `t`, replicas driven on the engine directly.
fn monte_carlo(model: &CompiledModel, target: &str, t: f64, n: u64) -> (f64, f64) {
    let hits: u64 = (0..n)
        .into_par_iter()
        .map(|replica| {
            let config = EngineConfig {
                t_max: t,
                seed: 20_260_926,
                rng_stream: replica,
                stop_at_targets: true,
                sequences: true,
                ..EngineConfig::default()
            };
            let run = Engine::new(model, config).unwrap().run().unwrap();
            let reached = run
                .sequence
                .and_then(|sequence| sequence.end_cause)
                .is_some_and(|cause| cause == target);
            u64::from(reached)
        })
        .sum();
    let p = hits as f64 / n as f64;
    let half = 2.576 * (p * (1.0 - p) / n as f64).sqrt();
    (p - half, p + half)
}

/// The Monte-Carlo interval meets the result's bounds widened by its
/// error estimate.
fn assert_meets(interval: (f64, f64), result: &ExplorationResult, what: &str) {
    let (low, high) = interval;
    let err = result.error_estimate().unwrap_or(0.0);
    eprintln!(
        "{what}: MC [{low}, {high}], discretised [{}, {}] +- {err:e}",
        result.lower, result.upper
    );
    assert!(
        low <= result.upper + err && high >= result.lower - err,
        "{what}: MC [{low}, {high}] vs discretised [{}, {}] +- {err}",
        result.lower,
        result.upper
    );
}

// ---- models -----------------------------------------------------------------

/// Parallel pair A (rate a), B (rate b), non repairable, target "both
/// failed".
fn parallel_pair(a: f64, b: f64) -> CompiledModel {
    build(
        "parallel_pair",
        vec![
            unit("A", exp(a), None),
            unit("B", exp(b), None),
            watcher(and(vec![down("A"), down("B")])),
        ],
        vec![sys_down()],
    )
}

/// `P(T_A < T_B <= t)` for independent `T_A ~ Exp(a)`, `T_B ~ Exp(b)`.
fn first_then_second(a: f64, b: f64, t: f64) -> f64 {
    a / (a + b) * (-(-(a + b) * t).exp_m1()) - (-b * t).exp() * (-(-a * t).exp_m1())
}

/// A (exponential) then B, whose Weibull failure is armed only once A has
/// failed; target "both failed". Outside the exact domain.
fn weibull_after_first_failure() -> CompiledModel {
    build(
        "weibull_after_first_failure",
        vec![
            unit("A", exp(0.5), None),
            unit("B", weibull(2.0, 1.5), Some(down("A"))),
            watcher(and(vec![down("A"), down("B")])),
        ],
        vec![sys_down()],
    )
}

/// A chain of four exponential units of rate `lambda`, each armed when the
/// previous one fails; target: the fourth failed. One sequence, of length
/// 4, whose probability is the Erlang(4, lambda) distribution function.
fn chain4(lambda: f64) -> CompiledModel {
    let mut components = vec![unit("C1", exp(lambda), None)];
    for i in 2..=4 {
        components.push(unit(
            &format!("C{i}"),
            exp(lambda),
            Some(down(&format!("C{}", i - 1))),
        ));
    }
    build(
        "chain4",
        components,
        vec![target("c4_down", "C4", "fail", "nok")],
    )
}

fn erlang4_cdf(lambda: f64, t: f64) -> f64 {
    let x = lambda * t;
    // 1 - e^{-x} sum_{k<4} x^k / k!, as the tail sum_{k>=4} (no cancellation).
    let mut term = (-x).exp() * x.powi(4) / 24.0;
    let mut sum = 0.0;
    let mut k = 4.0;
    while term > sum * 1e-18 {
        sum += term;
        k += 1.0;
        term *= x / k;
    }
    sum
}

// ---- agreement with the exact driver (AE1, R11) ----------------------------

/// Covers AE1.
#[test]
fn the_parallel_pair_agrees_with_the_exact_driver_within_twice_the_estimate() {
    let (a, b, t) = (0.5, 0.3, 2.0);
    let model = parallel_pair(a, b);
    let exact = explore_exact(&model, &ExactSettings::new("sys_down", t)).unwrap();
    let result = explore_discretised(&model, &DiscretisedSettings::new("sys_down", t)).unwrap();

    assert_eq!(result.algorithm, Algorithm::Discretised);
    let discretisation = result.discretisation.as_ref().unwrap();
    assert_eq!(discretisation.level, 2 * DEFAULT_LEVEL);
    let err = estimate(&result);
    assert!(
        err > 0.0,
        "the timing matters here, so the estimate is not 0"
    );

    assert!(
        (result.lower - exact.lower).abs() <= 2.0 * err,
        "discretised {} vs exact {} (estimate {err:e})",
        result.lower,
        exact.lower
    );
    // The same two sequences, each within twice the estimate.
    assert_eq!(result.sequences.len(), 2);
    for k in 0..2 {
        let steps = names(&result, k);
        let expected = if steps[0] == "A.fail.occ" {
            first_then_second(a, b, t)
        } else {
            first_then_second(b, a, t)
        };
        assert_eq!(steps.len(), 3, "{steps:?}");
        assert!(
            (result.sequences[k].probability - expected).abs() <= 2.0 * err,
            "{steps:?}: {} vs closed form {expected} (estimate {err:e})",
            result.sequences[k].probability
        );
    }
    // No cut-off fired: the bounds meet.
    assert_eq!(result.lower, result.upper);
    assert!(!result.inconclusive);
}

// ---- laws outside the exact domain (AE2) -----------------------------------

/// Covers AE2.
#[test]
fn a_weibull_armed_after_the_first_failure_is_explored_with_bounds() {
    let model = weibull_after_first_failure();
    let exact = explore_exact(&model, &ExactSettings::new("sys_down", 3.0));
    assert!(matches!(
        exact,
        Err(EngineError::LawOutsideExactDomain { .. })
    ));

    let result = explore_discretised(&model, &settings("sys_down", 3.0, 4)).unwrap();
    assert!(!result.sequences.is_empty());
    assert!(result.lower > 0.0 && result.lower <= result.upper);
    for pair in result.sequences.windows(2) {
        assert!(pair[0].probability >= pair[1].probability);
    }
    assert_eq!(
        names(&result, 0),
        ["A.fail.occ", "B.fail.occ", "sys.watch.down"]
    );
}

// ---- closed forms -------------------------------------------------------------

#[test]
fn a_single_weibull_failure_puts_its_whole_mass_in_its_cells() {
    let (shape, scale, t) = (2.0, 3.0, 2.0);
    let model = build(
        "single_weibull",
        vec![unit("W", weibull(shape, scale), None)],
        vec![target("w_down", "W", "fail", "nok")],
    );
    let result = explore_discretised(&model, &DiscretisedSettings::new("w_down", t)).unwrap();
    let expected = -(-(t / scale).powf(shape)).exp_m1();
    assert_eq!(result.sequences.len(), 1);
    assert_rel(result.lower, expected, 1e-12, "1 - S(t)");
    assert_rel(result.upper, expected, 1e-12, "upper");
    assert!(estimate(&result) <= 1e-12 * expected);
}

#[test]
fn a_delay_racing_an_exponential_failure_matches_its_closed_form() {
    // The mission M ends after `d`, which disarms A's failure (reset).
    let (lambda, d, t) = (0.4, 1.0, 3.0);
    let mission = component(
        "M",
        vec![automaton(
            "mission",
            &["run", "stop"],
            vec![tr("end", "run", &["stop"], delay(d))],
        )],
    );
    let model = build(
        "delay_race",
        vec![
            mission,
            unit("A", exp(lambda), Some(active("M", "mission", "run"))),
        ],
        vec![target("a_down", "A", "fail", "nok")],
    );
    let result = explore_discretised(&model, &DiscretisedSettings::new("a_down", t)).unwrap();
    let expected = -(-lambda * d).exp_m1();
    assert_eq!(result.sequences.len(), 1);
    assert_eq!(names(&result, 0), ["A.fail.occ"]);
    assert_rel(result.lower, expected, 1e-12, "1 - exp(-lambda d)");
}

/// A gate up for 1, down for 0.5, then up again; a worker whose Weibull
/// (shape 2, scale 2) completion is guarded by the gate, under `policy`.
fn policy_model(policy: &str) -> CompiledModel {
    let gate = component(
        "gate",
        vec![automaton(
            "phase",
            &["up1", "down", "up2"],
            vec![
                tr("close", "up1", &["down"], delay(1.0)),
                tr("reopen", "down", &["up2"], delay(0.5)),
            ],
        )],
    );
    let up = or(vec![
        active("gate", "phase", "up1"),
        active("gate", "phase", "up2"),
    ]);
    let finish = with(
        with(
            tr("finish", "wait", &["done"], weibull(2.0, 2.0)),
            "guard",
            up,
        ),
        "on_interruption",
        json!(policy),
    );
    build(
        &format!("policy_{policy}"),
        vec![
            gate,
            component("w", vec![automaton("job", &["wait", "done"], vec![finish])]),
        ],
        vec![target("done", "w", "job", "done")],
    )
}

#[test]
fn each_interruption_policy_matches_its_closed_form_and_monte_carlo() {
    let t = 2.5;
    // Weibull(2, 2) cumulative hazard: (age / 2)^2.
    let h = |age: f64| (age / 2.0).powi(2);
    let cases = [
        // reset: age 1 before the pause, then a fresh age 1 after it.
        ("reset", 1.0 - (-(h(1.0) + h(1.0))).exp()),
        // resume: the age freezes while down: 2 by the horizon.
        ("resume", 1.0 - (-h(2.0)).exp()),
        // continue: armed and aging throughout: 2.5.
        ("continue", 1.0 - (-h(2.5)).exp()),
    ];
    for (policy, expected) in cases {
        let model = policy_model(policy);
        let result = explore_discretised(&model, &settings("done", t, 4)).unwrap();
        assert_rel(result.lower, expected, 1e-12, policy);
        let interval = monte_carlo(&model, "done", t, 100_000);
        assert_meets(interval, &result, policy);
    }
}

// ---- Monte-Carlo witnesses ----------------------------------------------------

#[test]
fn a_lognormal_repair_race_agrees_with_monte_carlo() {
    // A fails (0.8) and is repaired under a lognormal duration into a state
    // where it cannot fail again; B fails (0.6), not repairable. Target:
    // both down, which needs B to fail while A is down (or A after B).
    let t = 2.0;
    let a = component(
        "A",
        vec![automaton(
            "fail",
            &["ok", "nok", "fixed"],
            vec![
                with(
                    tr("occ", "ok", &["nok"], exp(0.8)),
                    "monitored",
                    json!(true),
                ),
                with(
                    tr(
                        "rep",
                        "nok",
                        &["fixed"],
                        json!({"distrib": "lognormal", "mu": (0.3_f64).ln(), "sigma": 0.6}),
                    ),
                    "monitored",
                    json!(true),
                ),
            ],
        )],
    );
    let model = build(
        "lognormal_repair",
        vec![
            a,
            unit("B", exp(0.6), None),
            watcher(and(vec![down("A"), down("B")])),
        ],
        vec![sys_down()],
    );
    let result = explore_discretised(&model, &DiscretisedSettings::new("sys_down", t)).unwrap();
    assert_eq!(result.lower, result.upper, "no cut-off fires");
    let interval = monte_carlo(&model, "sys_down", t, 100_000);
    assert_meets(interval, &result, "lognormal repair");
}

/// `x' = 1` from 0; `W` switches to `hi` when `x >= 1` (watched); `F`
/// fails at the continuously varying rate `k x`, doubled once `W` is
/// `hi`; `G` fails at rate `g`, armed once `W` is `hi`. Target: F or G
/// down.
fn hybrid(k: f64, g: f64) -> CompiledModel {
    let x = json!({"name": "X", "ports": [],
        "attributes": [{"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}],
        "automata": [],
        "equations": [{"target": "x", "kind": "ode", "expr": c(1.0)}]});
    let hi = active("W", "watch", "hi");
    let watch = component(
        "W",
        vec![automaton(
            "watch",
            &["lo", "hi"],
            vec![with(
                tr("hit", "lo", &["hi"], json!({"distrib": "watched"})),
                "guard",
                ge(attr("X", "x"), c(1.0)),
            )],
        )],
    );
    let f_rate = ite(
        hi.clone(),
        mul(c(2.0 * k), attr("X", "x")),
        mul(c(k), attr("X", "x")),
    );
    build(
        "hybrid",
        vec![
            x,
            watch,
            unit("F", exp_expr(f_rate), None),
            unit("G", exp(g), Some(hi)),
            watcher(or(vec![down("F"), down("G")])),
        ],
        vec![sys_down()],
    )
}

/// Covers AE6.
#[test]
fn a_hybrid_model_with_an_ode_and_a_watched_threshold_meets_monte_carlo() {
    let (k, g, t) = (0.3, 0.2, 2.0);
    let model = hybrid(k, g);
    let result = explore_discretised(&model, &settings("sys_down", t, 4)).unwrap();
    // Closed form: H_F(t) = k/2 + k (t^2 - 1), H_G(t) = g (t - 1).
    let expected = -(-(k / 2.0 + k * (t * t - 1.0) + g * (t - 1.0))).exp_m1();
    assert_rel(result.lower, expected, 1e-6, "hybrid closed form");
    // Both failure modes appear, G only after the watched crossing.
    let all: Vec<Vec<String>> = (0..result.sequences.len())
        .map(|k| names(&result, k))
        .collect();
    assert!(all.contains(&vec!["F.fail.occ".to_owned(), "sys.watch.down".to_owned()]));
    assert!(all.contains(&vec![
        "W.watch.hit".to_owned(),
        "G.fail.occ".to_owned(),
        "sys.watch.down".to_owned()
    ]));
    let interval = monte_carlo(&model, "sys_down", t, 100_000);
    assert_meets(interval, &result, "hybrid");
}

// ---- refinement (KTD9) -------------------------------------------------------

/// A (Weibull 2, 1.5) then B (Weibull 2, 1), armed once A has failed;
/// target: B down. The time A fails at decides how long B has.
///
/// B's shape is 2 on purpose: with shape 1.5 the level-8 and level-16
/// values happen to cross the true value (the mass-median rule is not
/// monotone there), so the estimate at `K = 8` understates the error of
/// the reported level-16 value by a factor 5. The estimate is an
/// estimate, not a bound.
fn two_weibulls() -> CompiledModel {
    build(
        "two_weibulls",
        vec![
            unit("A", weibull(2.0, 1.5), None),
            unit("B", weibull(2.0, 1.0), Some(down("A"))),
        ],
        vec![target("b_down", "B", "fail", "nok")],
    )
}

/// `P(T_A + T_B <= t)` by Simpson's rule on `int_0^t f_A(s) F_B(t - s) ds`.
fn two_weibulls_closed_form(t: f64) -> f64 {
    let f_a = |s: f64| (2.0 * s / 2.25) * (-(s / 1.5).powi(2)).exp();
    let cdf_b = |s: f64| -(-(s.max(0.0)).powi(2)).exp_m1();
    let n = 20_000;
    let h = t / n as f64;
    let mut sum = f_a(0.0) * cdf_b(t) + f_a(t) * cdf_b(0.0);
    for i in 1..n {
        let s = i as f64 * h;
        sum += if i % 2 == 1 { 4.0 } else { 2.0 } * f_a(s) * cdf_b(t - s);
    }
    sum * h / 3.0
}

#[test]
fn the_error_estimate_shrinks_when_the_level_doubles() {
    let t = 2.0;
    let model = two_weibulls();
    let coarse = explore_discretised(&model, &settings("b_down", t, 4)).unwrap();
    let fine = explore_discretised(&model, &settings("b_down", t, 8)).unwrap();
    let (e4, e8) = (estimate(&coarse), estimate(&fine));
    assert!(e8 < e4, "estimate at K=8 {e8:e} vs K=4 {e4:e}");
    assert!(
        !fine
            .discretisation
            .unwrap()
            .refinement
            .unwrap()
            .truncation_dominated
    );
    let expected = two_weibulls_closed_form(t);
    assert!(
        (fine.lower - expected).abs() <= 2.0 * e8,
        "K=16 {} vs closed form {expected} (estimate {e8:e})",
        fine.lower
    );
}

#[test]
fn a_refinement_whose_finer_pass_hits_the_branch_cap_is_truncation_dominated() {
    let model = parallel_pair(0.5, 0.3);
    let mut base = settings("sys_down", 2.0, 4);
    base.refine = false;
    base.cutoffs.max_branches = None;
    let nodes = explore_discretised(&model, &base).unwrap().expanded_nodes;

    let mut capped = settings("sys_down", 2.0, 4);
    capped.cutoffs.max_branches = Some(nodes + 1);
    let result = explore_discretised(&model, &capped).unwrap();
    assert!(result.cutoff_tallies.max_branches.fired());
    let refinement = result.discretisation.clone().unwrap().refinement.unwrap();
    assert_eq!(refinement.base_lower, refinement.base_upper);
    assert!(
        refinement.truncation_dominated,
        "{refinement:?} vs lower {} upper {}",
        result.lower, result.upper
    );
}

#[test]
fn switching_the_refinement_off_states_that_no_estimate_was_made() {
    let model = parallel_pair(0.5, 0.3);
    let mut s = settings("sys_down", 2.0, 4);
    s.refine = false;
    let result = explore_discretised(&model, &s).unwrap();
    let discretisation = result.discretisation.clone().unwrap();
    assert_eq!(discretisation.level, 4);
    assert!(discretisation.refinement.is_none());
    assert!(result.error_estimate().is_none());
    let document: Json = serde_json::from_str(&serde_json::to_string(&result).unwrap()).unwrap();
    assert_eq!(document["algorithm"], json!("discretised"));
    assert_eq!(document["discretisation"]["refinement"], Json::Null);
}

// ---- cut-offs, tree size, determinism ----------------------------------------

#[test]
fn a_length_four_sequence_of_probability_1e_4_is_retained_as_one_entry() {
    let (lambda, t) = (0.2318, 1.0);
    let model = chain4(lambda);
    let expected = erlang4_cdf(lambda, t);
    assert!((0.9e-4..1.1e-4).contains(&expected), "{expected}");
    let mut s = settings("c4_down", t, 8);
    s.cutoffs.min_probability = Some(1e-8);
    let result = explore_discretised(&model, &s).unwrap();
    assert_eq!(result.sequences.len(), 1);
    assert_eq!(result.sequences[0].steps.len(), 4);
    let err = estimate(&result);
    eprintln!(
        "length-4: p = {:e}, pruned by min_probability {:?}, estimate {err:e}, nodes {}",
        result.sequences[0].probability,
        result.cutoff_tallies.min_probability,
        result.expanded_nodes
    );
    assert!(
        (result.sequences[0].probability - expected).abs()
            <= 2.0 * err + (result.upper - result.lower),
        "{} vs Erlang {expected} (estimate {err:e}, gap {:e})",
        result.sequences[0].probability,
        result.upper - result.lower
    );
    assert!(
        result.relative_gap() < 0.05,
        "gap {}",
        result.relative_gap()
    );
}

#[test]
fn node_counts_at_levels_8_and_16_on_a_depth_four_model_fit_under_the_default_cap() {
    let model = chain4(0.5);
    let mut counts = Vec::new();
    for level in [8, 16] {
        let mut s = settings("c4_down", 2.0, level);
        s.refine = false;
        let cap = s.cutoffs.max_branches.unwrap();
        let result = explore_discretised(&model, &s).unwrap();
        assert!(!result.cutoff_tallies.max_branches.fired());
        assert!(result.expanded_nodes < cap);
        // One competitor per node: 1 + K + K^2 + K^3 expanded nodes.
        let k = u64::from(level);
        assert_eq!(result.expanded_nodes, 1 + k + k * k + k * k * k);
        counts.push(result.expanded_nodes);
    }
    eprintln!("expanded nodes at K = 8 and 16: {counts:?}");
}

#[test]
fn results_are_identical_with_one_and_four_threads() {
    let model = weibull_after_first_failure();
    let run = |threads, cap| {
        let mut s = settings("sys_down", 3.0, 4);
        s.threads = Some(threads);
        s.cutoffs.max_branches = cap;
        explore_discretised(&model, &s).unwrap()
    };
    assert_eq!(run(1, None), run(4, None));
    let capped = run(1, Some(40));
    assert!(capped.cutoff_tallies.max_branches.fired());
    assert_eq!(capped, run(4, Some(40)));
}

#[test]
fn the_result_round_trips_through_its_open_format() {
    let result = explore_discretised(
        &weibull_after_first_failure(),
        &settings("sys_down", 3.0, 4),
    )
    .unwrap();
    let back = read_exploration(&serde_json::to_string(&result).unwrap()).unwrap();
    assert_eq!(back.algorithm, Algorithm::Discretised);
    assert_eq!(back.steps, result.steps);
    let (d, e) = (back.discretisation.unwrap(), result.discretisation.unwrap());
    assert_eq!(d.level, e.level);
    let (r, s) = (d.refinement.unwrap(), e.refinement.unwrap());
    assert_eq!(r.base_level, s.base_level);
    assert_eq!(r.truncation_dominated, s.truncation_dominated);
    // serde_json's default float parser is correct to the last ulp or so,
    // not bit-exact.
    assert!(((r.error_estimate - s.error_estimate) / s.error_estimate).abs() <= 1e-15);
    assert!(((back.lower - result.lower) / result.lower).abs() <= 1e-15);
}

#[test]
fn an_exact_result_carries_no_discretisation() {
    let exact = explore_exact(
        &parallel_pair(0.5, 0.3),
        &ExactSettings::new("sys_down", 2.0),
    )
    .unwrap();
    assert!(exact.discretisation.is_none());
    let document: Json = serde_json::from_str(&serde_json::to_string(&exact).unwrap()).unwrap();
    assert!(document.get("discretisation").is_none());
}

#[test]
fn invalid_discretised_settings_are_refused_before_exploring() {
    let model = parallel_pair(0.5, 0.3);
    let mut s = settings("sys_down", 2.0, 0);
    let refused = |s: &DiscretisedSettings, parameter: &str| match explore_discretised(&model, s) {
        Err(EngineError::InvalidStudyParameter { parameter: p, .. }) => assert_eq!(p, parameter),
        other => panic!("expected a refusal on {parameter}, got {other:?}"),
    };
    refused(&s, "level");
    s.level = 4;
    s.horizon = -1.0;
    refused(&s, "horizon");
    s.horizon = 2.0;
    s.target = "nothing".into();
    refused(&s, "target");
}

// ---- review findings: format version, fallbacks, bounded laws -------------

#[test]
fn a_discretised_result_is_written_at_format_version_2_and_checked_on_reading() {
    let result = explore_discretised(
        &weibull_after_first_failure(),
        &settings("sys_down", 3.0, 4),
    )
    .unwrap();
    assert_eq!(result.version, 2);
    assert_eq!(Algorithm::Discretised.format_version(), 2);
    assert_eq!(Algorithm::Exact.format_version(), 1);

    let mut document = serde_json::to_value(&result).unwrap();
    // A discretised document at version 1 was not written by this format's
    // writer: refused as inconsistent, not read as a version-1 document.
    document["version"] = json!(1);
    match read_exploration(&document.to_string()) {
        Err(ReadExplorationError::AlgorithmVersion {
            algorithm: Algorithm::Discretised,
            version: 1,
            required: 2,
        }) => {}
        other => panic!("expected an inconsistent-version refusal, got {other:?}"),
    }
    // A version above the highest readable one is refused by version.
    document["version"] = json!(3);
    assert!(matches!(
        read_exploration(&document.to_string()),
        Err(ReadExplorationError::Version(Some(3)))
    ));
    // An exact result read back at version 2 is accepted: version 2 knows
    // the exact algorithm too.
    let exact = explore_exact(
        &parallel_pair(0.5, 0.3),
        &ExactSettings::new("sys_down", 2.0),
    )
    .unwrap();
    assert_eq!(exact.version, 1);
    let mut document = serde_json::to_value(&exact).unwrap();
    document["version"] = json!(2);
    assert!(read_exploration(&document.to_string()).is_ok());
}

/// `x' = 1` from 0; `W` switches to `hi` at the watched crossing `x >= 1`,
/// which disarms `U`, whose failure law is uniform on `[1, 2]`. `U` can
/// only fire inside the event-location bracket of the crossing, so every
/// cell median of the window lies past the true crossing instant.
fn uniform_at_a_watched_crossing() -> CompiledModel {
    let x = json!({"name": "X", "ports": [],
        "attributes": [{"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}],
        "automata": [],
        "equations": [{"target": "x", "kind": "ode", "expr": c(1.0)}]});
    let watch = component(
        "W",
        vec![automaton(
            "watch",
            &["lo", "hi"],
            vec![with(
                tr("hit", "lo", &["hi"], json!({"distrib": "watched"})),
                "guard",
                ge(attr("X", "x"), c(1.0)),
            )],
        )],
    );
    build(
        "uniform_at_crossing",
        vec![
            x,
            watch,
            unit(
                "U",
                json!({"distrib": "uniform", "low": 1.0, "high": 2.0}),
                Some(active("W", "watch", "lo")),
            ),
        ],
        vec![target("u_down", "U", "fail", "nok")],
    )
}

#[test]
fn a_firing_past_a_watched_crossing_falls_back_before_it() {
    let model = uniform_at_a_watched_crossing();
    let mut s = settings("u_down", 1.5, 4);
    s.refine = false;
    let result = explore_discretised(&model, &s).unwrap();
    // The whole mass U can carry lies in the crossing's location bracket,
    // at most the event-location tolerance (1e-10) wide.
    assert!(result.lower.is_finite() && result.upper.is_finite());
    assert!(0.0 <= result.lower && result.lower <= result.upper);
    assert!(result.upper <= 1e-9, "upper {}", result.upper);
}

/// Two clocks whose deterministic dates coincide, exactly or up to
/// round-off, racing a chain of three exponential failures.
fn clocks_and_chain(first: f64, second: f64, other: f64) -> CompiledModel {
    let p = component(
        "P",
        vec![automaton(
            "clock",
            &["p0", "p1", "p2"],
            vec![
                tr("a", "p0", &["p1"], delay(first)),
                tr("b", "p1", &["p2"], delay(second)),
            ],
        )],
    );
    let q = component(
        "Q",
        vec![automaton(
            "clock",
            &["q0", "q1"],
            vec![tr("a", "q0", &["q1"], delay(other))],
        )],
    );
    build(
        "clocks_and_chain",
        vec![
            p,
            q,
            unit("X", exp(1.0), None),
            unit("Y", exp(1.0), Some(down("X"))),
            unit("Z", exp(1.0), Some(down("Y"))),
        ],
        vec![target("z_down", "Z", "fail", "nok")],
    )
}

#[test]
fn near_coincident_deterministic_dates_open_no_extra_window() {
    // P fires at 0.1 then at 0.1 + 0.2 = 0.30000000000000004; Q at 0.3
    // (one ulp earlier), or at 0.1 + 0.2 (the same instant).
    let (first, second) = (0.1, 0.2);
    assert_ne!(first + second, 0.3, "the dates differ by round-off");
    let mut s = settings("z_down", 1.0, 4);
    s.refine = false;
    let near = explore_discretised(&clocks_and_chain(first, second, 0.3), &s).unwrap();
    let exact = explore_discretised(&clocks_and_chain(first, second, first + second), &s).unwrap();
    assert_eq!(near.expanded_nodes, exact.expanded_nodes);
    assert!(
        (near.lower - exact.lower).abs() <= 1e-12 * exact.lower,
        "{} vs {}",
        near.lower,
        exact.lower
    );
    assert_eq!(near.sequences.len(), exact.sequences.len());
}

#[test]
fn a_bounded_uniform_competitor_racing_an_exponential_matches_its_closed_form() {
    // One unit, two competing failure modes from `ok`: `u` uniform on
    // [0.5, 1.5], `e` exponential. The horizon lies past the uniform's
    // support, so its hazard is infinite at the window's end and the
    // last cell's shares come from the fallback.
    let (lambda, t) = (0.7, 2.0);
    let model = build(
        "uniform_race",
        vec![component(
            "M",
            vec![automaton(
                "fail",
                &["ok", "u_down", "e_down"],
                vec![
                    tr(
                        "u",
                        "ok",
                        &["u_down"],
                        json!({"distrib": "uniform", "low": 0.5, "high": 1.5}),
                    ),
                    tr("e", "ok", &["e_down"], exp(lambda)),
                ],
            )],
        )],
        vec![
            target("by_u", "M", "fail", "u_down"),
            target("by_e", "M", "fail", "e_down"),
        ],
    );
    // P(T_u < T_e) = int_{0.5}^{1.5} exp(-lambda s) ds.
    let by_u = ((-0.5 * lambda).exp() - (-1.5 * lambda).exp()) / lambda;
    for (target, expected) in [("by_u", by_u), ("by_e", 1.0 - by_u)] {
        let result = explore_discretised(&model, &DiscretisedSettings::new(target, t)).unwrap();
        let err = estimate(&result);
        for sequence in &result.sequences {
            assert!(sequence.probability.is_finite() && sequence.probability >= 0.0);
        }
        assert!(result.lower.is_finite() && result.lower <= result.upper);
        eprintln!(
            "{target}: {} vs closed form {expected} (estimate {err:e})",
            result.lower
        );
        assert!(
            (result.lower - expected).abs() <= 2.0 * err,
            "{target}: {} vs closed form {expected} (estimate {err:e})",
            result.lower
        );
        // The estimate adapts to the error, so it is pinned too: the last
        // cell holds 1/K of the mass, and sharing it equally rather than
        // by the hazard increments up to its median moves the result by
        // about 3e-2 and the estimate to about 2e-2 (measured).
        assert!(err <= 3e-3, "{target}: estimate {err:e}");
        assert!(
            (result.lower - expected).abs() <= 2e-3,
            "{target}: {} vs closed form {expected}",
            result.lower
        );
    }
}
