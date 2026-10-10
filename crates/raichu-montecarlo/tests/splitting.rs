//! Generalized AMS: closed forms, coverage and pathwise invariants.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use raichu_core::CompiledModel;
use raichu_model::Model;
use raichu_montecarlo::{
    run_splitting, ImportanceSource, SplittingSettings, DEFAULT_SPLITTING_MAX_CUT_SETS,
};
use serde_json::{json, Value};

fn constant(value: f64) -> Value {
    json!({"op":"const","value":{"kind":"float","value":value}})
}
fn system(n: usize, k: usize, law: Value, repair: Option<f64>) -> Model {
    let mut components = Vec::new();
    let mut scores = Vec::new();
    for i in 0..n {
        let name = format!("u{i}");
        let mut transitions = vec![
            json!({"name":"fail","source":"up","targets":["down"],"distrib":"delay","time":0.0}),
        ];
        transitions[0].as_object_mut().unwrap().remove("time");
        transitions[0]
            .as_object_mut()
            .unwrap()
            .extend(law.as_object().unwrap().clone());
        if let Some(rate) = repair {
            transitions.push(json!({"name":"repair","source":"down","targets":["up"],"distrib":"exp","rate":rate}));
        }
        components.push(json!({"name":name,"automata":[{"name":"life","states":["up","down"],"init":"up","transitions":transitions}]}));
        scores.push(json!({"op":"if","cond":{"op":"state_active","state":{"component":name,"automaton":"life","state":"down"}},"then":constant(1.0),"otherwise":constant(0.0)}));
    }
    components.push(json!({"name":"sys","attributes":[{"name":"score","kind":"float","init":{"kind":"float","value":0.0}}],"equations":[{"target":"score","kind":"explicit","expr":{"op":"add","args":scores}}],"automata":[{"name":"target","states":["safe","lost"],"init":"safe","transitions":[{"name":"loss","source":"safe","targets":["lost"],"distrib":"inst","probs":[],"guard":{"op":"cmp","cmp":"ge","lhs":{"op":"attr","attr":{"component":"sys","attribute":"score"}},"rhs":constant(k as f64)}}]}]}));
    serde_json::from_value(json!({"name":"splitting_witness","components":components,"targets":[{"name":"lost","component":"sys","automaton":"target","state":"lost"}]})).unwrap()
}
fn exp(rate: f64) -> Value {
    json!({"distrib":"exp","rate":rate})
}
fn settings(particles: u64) -> SplittingSettings {
    SplittingSettings {
        target: "lost".into(),
        t_max: 1.0,
        particles,
        importance: ImportanceSource::Attribute {
            name: "sys.score".into(),
        },
        ..Default::default()
    }
}
fn cut_settings(particles: u64, max_cut_sets: usize) -> SplittingSettings {
    SplittingSettings {
        target: "lost".into(),
        t_max: 1.0,
        particles,
        importance: ImportanceSource::CutSets { max_cut_sets },
        ..Default::default()
    }
}
/// The same witness whose target guard reads the failed-unit count
/// directly from the states (`sum(if(down, 1, 0)) >= k`), the form
/// fault-tree generation explains: the vote is the top expression, so the
/// minimal cut sets are the failing combinations. The dynamics are the
/// attribute guard's; the tree is what differs.
fn system_vote(n: usize, k: usize, law: Value) -> Model {
    use raichu_expr::{CmpOp, Expr, StateRef, Value as V};
    let mut model = system(n, k, law, None);
    let vote = Expr::Cmp {
        cmp: CmpOp::Ge,
        lhs: Box::new(Expr::Add {
            args: (0..n)
                .map(|i| Expr::If {
                    cond: Box::new(Expr::StateActive {
                        state: StateRef {
                            component: format!("u{i}"),
                            automaton: "life".into(),
                            state: "down".into(),
                        },
                    }),
                    then: Box::new(Expr::Const {
                        value: V::Float(1.0),
                    }),
                    otherwise: Box::new(Expr::Const {
                        value: V::Float(0.0),
                    }),
                })
                .collect(),
        }),
        rhs: Box::new(Expr::Const {
            value: V::Float(k as f64),
        }),
    };
    model.components.last_mut().unwrap().automata[0].transitions[0].guard = Some(vote);
    model
}
fn binomial_cdf(k: u64, n: u64, p: f64) -> f64 {
    let mut term = (1.0 - p).powi(n as i32);
    let mut sum = term;
    for i in 1..=k {
        term *= (n - i + 1) as f64 / i as f64 * p / (1.0 - p);
        sum += term;
    }
    sum
}
#[test]
fn one_in_a_million_interval_coverage() {
    let rate: f64 = 0.0005775;
    let q = -(-rate).exp_m1();
    let truth = 3.0 * q * q - 2.0 * q * q * q;
    let model = CompiledModel::compile(&system(3, 2, exp(rate), None)).unwrap();
    let mut conclusive = 0;
    let mut covered = 0;
    let mut extinct = 0;
    for seed in 0..100 {
        let mut s = settings(16_000);
        s.seed = seed;
        let r = run_splitting(&model, &s).unwrap();
        extinct += u64::from(r.extinct_batches > 0);
        if !r.estimate_inconclusive {
            conclusive += 1;
            covered += u64::from(r.interval.low <= truth && truth <= r.interval.high);
        }
    }
    eprintln!(
        "2oo3 coverage={covered}/{conclusive}, inconclusive={}, extinct={extinct}/100",
        100 - conclusive
    );
    assert!(conclusive >= 90);
    assert!(binomial_cdf(covered, conclusive, 0.95) >= 0.01);
}

#[test]
fn student_quantiles_match_nist_and_cauchy() {
    use raichu_montecarlo::{batch_interval, student_critical, IntervalMethod};
    for (df, expected) in [
        (1, 12.7062047361747),
        (2, 4.30265272969614),
        (19, 2.09302405440826),
        (30, 2.04227245630124),
        (100, 1.98397151844963),
    ] {
        assert!(
            (student_critical(0.95, df) - expected).abs() < 1e-10,
            "df={df}"
        );
    }
    for level in [0.1, 0.5, 0.8, 0.99, 0.9999] {
        let expected = (std::f64::consts::FRAC_PI_2 * level).tan();
        assert!((student_critical(level, 1) / expected - 1.0).abs() < 1e-9);
    }
    let r = batch_interval(&[1.0, 2.0, 3.0], 0.95);
    assert_eq!(r.method, IntervalMethod::BatchStudent);
    assert_eq!(r.estimate, 2.0);
    assert!((r.standard_error - 1.0 / 3f64.sqrt()).abs() < 1e-14);
}
#[test]
fn seeds_threads_ties_and_strict_crossings() {
    let m = CompiledModel::compile(&system(3, 2, exp(0.2), None)).unwrap();
    let mut s = settings(100);
    s.batches = 3;
    s.seed = 17;
    let mut prior = None;
    for threads in [1, 4, 16] {
        s.threads = Some(threads);
        let r = run_splitting(&m, &s).unwrap();
        let bytes = serde_json::to_vec(&r).unwrap();
        if let Some(p) = &prior {
            assert_eq!(p, &bytes);
        }
        prior = Some(bytes);
        for b in r.batches {
            assert!(b.levels.windows(2).all(|p| p[0].level < p[1].level));
            for l in b.levels {
                assert!(l.killed > 0);
                assert_eq!(l.survival_fraction, 1.0 - l.killed as f64 / 100.0);
                assert!([0.0, 1.0].contains(&l.level));
            }
        }
    }
}
#[test]
fn iteration_limit_has_no_estimate_and_extinction_has_zero() {
    use raichu_montecarlo::SplittingError;
    let m = CompiledModel::compile(&system(3, 2, exp(0.2), None)).unwrap();
    let mut s = settings(100);
    s.batches = 2;
    s.max_iterations = 0;
    assert!(matches!(
        run_splitting(&m, &s),
        Err(SplittingError::IterationLimit { iterations: 0, .. })
    ));
    let impossible = CompiledModel::compile(&system(3, 4, exp(0.2), None)).unwrap();
    s.max_iterations = 20;
    let r = run_splitting(&impossible, &s).unwrap();
    assert_eq!(r.interval.estimate, 0.0);
    assert_eq!(r.extinct_batches, 2);
    assert!(r.estimate_inconclusive);
}
#[test]
fn invalid_settings_fail_by_name() {
    let m = CompiledModel::compile(&system(3, 2, exp(0.2), None)).unwrap();
    let mut s = settings(10);
    s.importance = ImportanceSource::Attribute {
        name: "missing.score".into(),
    };
    assert!(run_splitting(&m, &s)
        .unwrap_err()
        .to_string()
        .contains("missing.score"));
    let mut model = system(3, 2, exp(0.2), None);
    model
        .components
        .last_mut()
        .unwrap()
        .attributes
        .push(raichu_model::Attribute {
            name: "bool".into(),
            kind: raichu_model::AttrKind::Bool,
            init: raichu_expr::Value::Bool(false),
        });
    let m = CompiledModel::compile(&model).unwrap();
    s.importance = ImportanceSource::Attribute {
        name: "sys.bool".into(),
    };
    assert!(run_splitting(&m, &s)
        .unwrap_err()
        .to_string()
        .contains("sys.bool"));
    s = settings(10);
    s.score_grid = vec![0.5, 0.2];
    assert!(run_splitting(&m, &s)
        .unwrap_err()
        .to_string()
        .contains("score_grid"));
}

#[test]
fn initial_target_and_nonfinite_scores() {
    let mut model = system(3, 0, exp(0.1), None);
    let m = CompiledModel::compile(&model).unwrap();
    let mut s = settings(20);
    s.batches = 2;
    let r = run_splitting(&m, &s).unwrap();
    assert_eq!(r.interval.estimate, 1.0);
    assert!(r.batches.iter().all(|b| b.levels.is_empty()));
    model = system(3, 2, exp(0.1), None);
    model.components.last_mut().unwrap().equations[0].expr = raichu_expr::Expr::Const {
        value: raichu_expr::Value::Float(f64::INFINITY),
    };
    model.components.last_mut().unwrap().automata[0]
        .transitions
        .clear();
    let m = CompiledModel::compile(&model).unwrap();
    assert!(run_splitting(&m, &s)
        .unwrap_err()
        .to_string()
        .contains("non-finite"));
}
#[test]
fn a_constant_or_overshooting_score_keeps_target_latching() {
    let mut model = system(3, 2, exp(0.3), None);
    // Keep the target's own counting expression, while changing only the score.
    let count = model.components.last().unwrap().equations[0].expr.clone();
    if let Some(raichu_expr::Expr::Cmp { lhs, .. }) =
        &mut model.components.last_mut().unwrap().automata[0].transitions[0].guard
    {
        **lhs = count;
    }
    for value in [0.0, 100.0] {
        model.components.last_mut().unwrap().equations[0].expr = raichu_expr::Expr::Const {
            value: raichu_expr::Value::Float(value),
        };
        let m = CompiledModel::compile(&model).unwrap();
        let mut s = settings(1000);
        s.batches = 20;
        let r = run_splitting(&m, &s).unwrap();
        let q = -(-0.3f64).exp_m1();
        let truth = 3.0 * q * q - 2.0 * q * q * q;
        assert!((r.interval.estimate - truth).abs() < 4.0 * r.interval.standard_error);
        assert!(r.batches.iter().all(|b| b.levels.len() <= 1));
    }
}
#[test]
fn completed_instants_hide_transient_score_peaks() {
    let mut model = system(1, 1, json!({"distrib":"delay","time":0.5}), None);
    // A same-instant repair removes the apparent score peak before observation.
    model.components[0].automata[0]
        .transitions
        .push(raichu_model::Transition {
            name: "repair".into(),
            source: "down".into(),
            targets: vec!["up".into()],
            guard: None,
            on_interruption: Default::default(),
            monitored: false,
            monitored_states: None,
            cycle_group: None,
            kind: None,
            effects: vec![],
            distrib: raichu_model::Distrib::Delay { time: 0.0 },
        });
    // No reachable target: a transient score of one must never split a particle.
    model.components.last_mut().unwrap().automata[0]
        .transitions
        .clear();
    let m = CompiledModel::compile(&model).unwrap();
    let mut s = settings(4);
    s.batches = 2;
    let r = run_splitting(&m, &s).unwrap();
    assert!(r.batches.iter().all(|b| b.extinct && b.levels.is_empty()));
}
#[test]
fn grid_with_state_dependent_hazard_and_competing_target() {
    let mut model = system(0, 1, exp(0.0), None);
    let sys = model.components.last_mut().unwrap();
    sys.attributes.push(raichu_model::Attribute {
        name: "ramp".into(),
        kind: raichu_model::AttrKind::Float,
        init: raichu_expr::Value::Float(0.0),
    });
    sys.equations=serde_json::from_value(json!([
        {"target":"ramp","kind":"ode","expr":constant(1.0)},
        {"target":"score","kind":"explicit","expr":{"op":"attr","attr":{"component":"sys","attribute":"ramp"}}}
    ])).unwrap();
    sys.automata[0].states.push("benign".into());
    let rate_expr = |rate: f64| json!({"op":"mul","args":[constant(rate),{"op":"attr","attr":{"component":"sys","attribute":"ramp"}}]});
    sys.automata[0].transitions=serde_json::from_value(json!([
        {"name":"loss","source":"safe","targets":["lost"],"distrib":"exp","rate_expr":rate_expr(0.3)},
        {"name":"benign","source":"safe","targets":["benign"],"distrib":"exp","rate_expr":rate_expr(0.2)}
    ])).unwrap();
    model.targets.push(raichu_model::Target {
        name: "benign".into(),
        component: "sys".into(),
        automaton: "target".into(),
        state: "benign".into(),
    });
    let m = CompiledModel::compile(&model).unwrap();
    let mut s = settings(100);
    s.batches = 20;
    s.t_max = 2.0;
    s.score_grid = (1..=8).map(|i| i as f64 / 4.0).collect();
    let r = run_splitting(&m, &s).unwrap();
    let truth = 0.6 * (-(-1.0f64).exp_m1());
    assert!(
        (r.interval.estimate - truth).abs() < 4.0 * r.interval.standard_error,
        "{:?}, truth={truth}",
        r.interval
    );
    assert!(r.batches.iter().any(|b| b.levels.len() > 1));
    // Independent plain Monte-Carlo uses the same first-target semantics.
    let mc = raichu_montecarlo::run_to_targets(
        &m,
        &raichu_montecarlo::McConfig {
            nb_runs: 5000,
            t_max: 2.0,
            seed: 991,
            samples: vec![],
            threads: Some(4),
            quantiles: vec![],
            confidence: 0.95,
            ode: Default::default(),
            stop_at_targets: true,
            flow: Default::default(),
        },
    )
    .unwrap();
    let hits = mc
        .ends
        .iter()
        .filter(|e| e.end_cause.as_deref() == Some("lost"))
        .count() as f64;
    let p = hits / 5000.0;
    let se = (p * (1.0 - p) / 5000.0 + r.interval.standard_error.powi(2)).sqrt();
    assert!((p - r.interval.estimate).abs() < 4.0 * se);
}

#[test]
fn fast_repairs_never_claim_a_conclusive_estimate() {
    let m = CompiledModel::compile(&system(2, 2, exp(1e-3), Some(1000.0))).unwrap();
    for seed in 0..10 {
        let mut s = settings(1000);
        s.t_max = 1000.0;
        s.seed = seed;
        s.batches = 20;
        let r = run_splitting(&m, &s).unwrap();
        assert!(
            r.estimate_inconclusive,
            "fast-repair result claims confidence: {:?}",
            r.interval
        );
        eprintln!(
            "fast repair seed={seed} estimate={} extinct={}",
            r.interval.estimate, r.extinct_batches
        );
    }
}
#[test]
fn weibull_rare_event_coverage() {
    let shape = 2.0;
    let scale = (1.0 / 0.0005775f64).sqrt();
    let q = -(-(1.0 / scale).powf(shape)).exp_m1();
    let truth = 3.0 * q * q - 2.0 * q * q * q;
    let m = CompiledModel::compile(&system(
        3,
        2,
        json!({"distrib":"weibull","shape":shape,"scale":scale}),
        None,
    ))
    .unwrap();
    let mut conclusive = 0;
    let mut covered = 0;
    for seed in 200..300 {
        let mut s = settings(16000);
        s.seed = seed;
        let r = run_splitting(&m, &s).unwrap();
        if !r.estimate_inconclusive {
            conclusive += 1;
            covered += u64::from(r.interval.low <= truth && truth <= r.interval.high);
        }
    }
    eprintln!(
        "Weibull coverage={covered}/{conclusive}, inconclusive={}",
        100 - conclusive
    );
    assert!(conclusive >= 90);
    assert!(binomial_cdf(covered, conclusive, 0.95) >= 0.01);
}
// Independent finite-state CTMC exponentiation by uniformization, with scaling
// and squaring. Includes the absorbing state, avoiding 1-survival cancellation.
fn markov_truth(n: usize, k: usize, lambda: f64, mu: f64, t: f64) -> f64 {
    let size = k + 1;
    let nu = (0..k)
        .map(|i| (n - i) as f64 * lambda + i as f64 * mu)
        .fold(0.0, f64::max);
    let scaling = ((nu * t / 0.5).log2().ceil().max(0.0)) as u32;
    let x = nu * t / 2f64.powi(scaling as i32);
    let mut p = vec![vec![0.0; size]; size];
    for (i, row) in p.iter_mut().enumerate().take(k) {
        let up = (n - i) as f64 * lambda / nu;
        let down = i as f64 * mu / nu;
        row[i] = 1.0 - up - down;
        row[i + 1] = up;
        if i > 0 {
            row[i - 1] = down;
        }
    }
    p[k][k] = 1.0;
    let mul = |a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        (0..size)
            .map(|i| {
                (0..size)
                    .map(|j| (0..size).map(|h| a[i][h] * b[h][j]).sum())
                    .collect()
            })
            .collect()
    };
    let mut term = vec![vec![0.0; size]; size];
    for (i, row) in term.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let mut result = vec![vec![0.0; size]; size];
    let mut weight = (-x).exp();
    for j in 0..40 {
        for i in 0..size {
            for h in 0..size {
                result[i][h] += weight * term[i][h];
            }
        }
        term = mul(&term, &p);
        weight *= x / (j + 1) as f64;
    }
    for _ in 0..scaling {
        result = mul(&result, &result);
    }
    result[0][k]
}

#[test]
fn splitting_recovers_a_repairable_regime_cross_entropy_flags() {
    use raichu_montecarlo::{run_cross_entropy, CrossEntropySettings};
    let mut model = system(4, 3, exp(0.01), Some(1.0));
    for c in model.components.iter_mut().take(4) {
        c.automata[0].transitions[0].kind = Some(raichu_model::TransitionKind::Failure);
        c.automata[0].transitions[1].kind = Some(raichu_model::TransitionKind::Repair);
    }
    let m = CompiledModel::compile(&model).unwrap();
    let truth = markov_truth(4, 3, 0.01, 1.0, 100.0);
    // Independent cross-check of the matrix exponential in a closed-form regime.
    let q = -(-0.01f64 * 10.0).exp_m1();
    assert!((markov_truth(3, 2, 0.01, 0.0, 10.0) - (3.0 * q * q - 2.0 * q * q * q)).abs() < 1e-13);
    let mut conclusive = 0;
    let mut covered = 0;
    for seed in 0..100 {
        let mut s = settings(1000);
        s.t_max = 100.0;
        s.seed = seed;
        let r = run_splitting(&m, &s).unwrap();
        if !r.estimate_inconclusive {
            conclusive += 1;
            covered += u64::from(r.interval.low <= truth && truth <= r.interval.high);
        }
        if seed == 44 {
            let mut ce_settings = CrossEntropySettings {
                target: "lost".into(),
                t_max: 100.0,
                seed,
                nb_runs: r.trajectories - 5000,
                pilot_runs: 500,
                max_iterations: 10,
                threads: Some(4),
                ..Default::default()
            };
            // Match actual trajectory launches, including the pilot. The pilot
            // is deterministic and independent of the final campaign's size.
            let preliminary = run_cross_entropy(&m, &ce_settings).unwrap();
            let pilot = preliminary.history.len() as u64 * 500;
            ce_settings.nb_runs = r.trajectories - pilot;
            let ce = run_cross_entropy(&m, &ce_settings).unwrap();
            assert_eq!(ce.replicas + ce.history.len() as u64 * 500, r.trajectories);
            eprintln!("capability gain seed44 truth={truth:e} launches={} splitting={:?} CE={:?} CE_inconclusive={}",r.trajectories,r.interval,ce.estimate,ce.estimate_inconclusive);
            assert!(!r.estimate_inconclusive);
            assert!(ce.estimate_inconclusive);
        }
    }
    eprintln!(
        "repairable coverage={covered}/{conclusive}, inconclusive={}",
        100 - conclusive
    );
    assert!(conclusive >= 90);
    assert!(binomial_cdf(covered, conclusive, 0.95) >= 0.01);
}
#[test]
fn imported_units_are_refused_by_name() {
    let mut model = system(3, 2, exp(0.1), None);
    model.fmu_units =
        serde_json::from_value(json!([{"name":"external","path":"unknown.fmu","step":1.0}]))
            .unwrap();
    let m = CompiledModel::compile(&model).unwrap();
    assert!(run_splitting(&m, &settings(10))
        .unwrap_err()
        .to_string()
        .contains("external"));
}
#[test]
fn weibull_clones_have_distinct_residual_futures() {
    use raichu_core::{Engine, EngineConfig};
    let m = CompiledModel::compile(&system(
        3,
        2,
        json!({"distrib":"weibull","shape":2.0,"scale":10.0}),
        None,
    ))
    .unwrap();
    let config = EngineConfig {
        t_max: 100.0,
        stop_at_targets: true,
        ..Default::default()
    };
    let mut parent = Engine::new(&m, config.clone()).unwrap();
    parent.step().unwrap();
    parent.advance_to(parent.current_time()).unwrap();
    let snapshot = parent.snapshot().unwrap();
    let mut dates = Vec::new();
    for stream in 1..20 {
        let mut child =
            Engine::restart_from_snapshot(&m, config.clone(), &snapshot, stream).unwrap();
        dates.push(child.step().unwrap().unwrap().time);
    }
    assert!(dates.windows(2).all(|pair| pair[0] != pair[1]));
}

/// The cut-set fractions rank completed states like the declared count.
/// Their different numeric levels must yield identical replacement decisions.
#[test]
fn cut_sets_rank_states_like_the_declared_count() {
    let m = CompiledModel::compile(&system_vote(3, 2, exp(0.2))).unwrap();
    let mut attribute = settings(200);
    attribute.seed = 11;
    attribute.batches = 4;
    let mut automatic = attribute.clone();
    automatic.importance = ImportanceSource::CutSets {
        max_cut_sets: DEFAULT_SPLITTING_MAX_CUT_SETS,
    };
    let declared = run_splitting(&m, &attribute).unwrap();
    let derived = run_splitting(&m, &automatic).unwrap();
    assert_eq!(declared.interval, derived.interval);
    for (a, c) in declared.batches.iter().zip(&derived.batches) {
        assert_eq!(a.estimate, c.estimate);
        assert_eq!(a.levels.len(), c.levels.len());
        for (la, lc) in a.levels.iter().zip(&c.levels) {
            assert_eq!(la.killed, lc.killed);
            assert_eq!(la.survival_fraction, lc.survival_fraction);
            assert!([0.0, 1.0].contains(&la.level));
            assert!([0.0, 0.5].contains(&lc.level));
        }
    }
}

/// The automatic source covers the same 1e-6 closed form as the hand
/// declared count, over the same 100-seed witness. The full budget of the
/// declared-score test is kept: the cut-set score costs a few state lookups
/// per completed instant, which moves the wall clock by a small factor only.
#[test]
fn cut_sets_source_covers_the_one_in_a_million_closed_form() {
    let rate: f64 = 0.0005775;
    let q = -(-rate).exp_m1();
    let truth = 3.0 * q * q - 2.0 * q * q * q;
    let model = CompiledModel::compile(&system_vote(3, 2, exp(rate))).unwrap();
    let mut conclusive = 0;
    let mut covered = 0;
    let mut extinct = 0;
    for seed in 0..100 {
        let mut s = cut_settings(16_000, DEFAULT_SPLITTING_MAX_CUT_SETS);
        s.seed = seed;
        let r = run_splitting(&model, &s).unwrap();
        extinct += u64::from(r.extinct_batches > 0);
        if !r.estimate_inconclusive {
            conclusive += 1;
            covered += u64::from(r.interval.low <= truth && truth <= r.interval.high);
        }
    }
    eprintln!(
        "2oo3 cut-set coverage={covered}/{conclusive}, inconclusive={}, extinct={extinct}/100",
        100 - conclusive
    );
    assert!(conclusive >= 90);
    assert!(binomial_cdf(covered, conclusive, 0.95) >= 0.01);
}

/// Fault-tree generation refuses a target guard it cannot explain: here a
/// weighted count of failed units, which is not a vote. The generation's
/// own reason is quoted, and the declared attribute named as the way out.
#[test]
fn an_unexplained_guard_is_refused_with_the_generation_reason() {
    use raichu_expr::Expr;
    let mut model = system(3, 2, exp(0.2), None);
    let weighted: Expr = serde_json::from_value(json!({"op": "cmp", "cmp": "ge",
        "lhs": {"op": "add", "args": [{"op": "if",
            "cond": {"op": "state_active", "state": {"component": "u0", "automaton": "life", "state": "down"}},
            "then": constant(2.0), "otherwise": constant(0.0)}]},
        "rhs": constant(2.0)}))
    .unwrap();
    model.components.last_mut().unwrap().automata[0].transitions[0].guard = Some(weighted);
    let m = CompiledModel::compile(&model).unwrap();
    let err = run_splitting(&m, &cut_settings(10, DEFAULT_SPLITTING_MAX_CUT_SETS))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("fault-tree generation refused the model"),
        "{err}"
    );
    assert!(err.contains("not a vote"), "{err}");
    assert!(err.contains("attribute"), "{err}");
}

/// A target with more minimal cut sets than the cap is refused naming the
/// cap, and runs once the cap is raised.
#[test]
fn more_cut_sets_than_the_cap_is_refused_by_name() {
    // 3-of-5: ten three-event minimal cut sets.
    let m = CompiledModel::compile(&system_vote(5, 3, exp(0.2))).unwrap();
    let err = run_splitting(&m, &cut_settings(10, 4))
        .unwrap_err()
        .to_string();
    assert!(err.contains("max_cut_sets"), "{err}");
    let mut generous = cut_settings(20, 16);
    generous.batches = 2;
    let r = run_splitting(&m, &generous).unwrap();
    assert_eq!(r.batches.len(), 2);
}

#[test]
fn automatic_importance_setting_round_trips_and_defaults_its_cap() {
    let source: ImportanceSource = serde_json::from_value(json!({"kind": "cut_sets"})).unwrap();
    assert_eq!(
        serde_json::to_value(source).unwrap(),
        json!({"kind": "cut_sets", "max_cut_sets": 1000})
    );
    assert!(serde_json::from_value::<ImportanceSource>(
        json!({"kind": "cut_sets", "unexpected": true})
    )
    .is_err());
}

/// Absorption removes the redundant conjunction before the cap is applied.
#[test]
fn cut_set_cap_counts_final_minimal_sets_after_absorption() {
    let mut doc = serde_json::to_value(system_vote(2, 1, exp(1.0))).unwrap();
    let state = |name: &str| json!({"op":"state_active","state":{"component":name,"automaton":"life","state":"down"}});
    doc["components"][2]["automata"][0]["transitions"][0]["guard"] = json!({
        "op":"bool", "bool_op":"or", "args":[state("u0"),
            {"op":"bool", "bool_op":"and", "args":[state("u0"),state("u1")]}]
    });
    let model: Model = serde_json::from_value(doc).unwrap();
    let compiled = CompiledModel::compile(&model).unwrap();
    let mut settings = cut_settings(20, 1);
    settings.batches = 2;
    let capped = run_splitting(&compiled, &settings).unwrap();
    settings.importance = ImportanceSource::CutSets { max_cut_sets: 2 };
    assert_eq!(capped, run_splitting(&compiled, &settings).unwrap());
}

/// Empty cuts represent a certain target, whereas no cuts explain no path.
#[test]
fn cut_set_constant_targets_preserve_first_hit_and_refuse_no_path() {
    for initially_lost in [true, false] {
        let mut doc = serde_json::to_value(system_vote(1, 1, exp(1.0))).unwrap();
        if initially_lost {
            doc["components"][1]["automata"][0]["init"] = json!("lost");
        } else {
            doc["components"][1]["automata"][0]["transitions"][0]["guard"] =
                json!({"op":"const","value":{"kind":"bool","value":true}});
        }
        let model: Model = serde_json::from_value(doc).unwrap();
        let compiled = CompiledModel::compile(&model).unwrap();
        let mut settings = cut_settings(10, 1);
        settings.batches = 2;
        let result = run_splitting(&compiled, &settings).unwrap();
        assert_eq!(result.interval.estimate, 1.0);
        assert_eq!(result.extinct_batches, 0);
        assert!(result
            .batches
            .iter()
            .all(|b| b.estimate == 1.0 && !b.extinct && b.levels.is_empty()));
    }
    // This guard reads an attribute nothing computes (the score's equation
    // removed): it stays at zero, and no failure can reach the target.
    let mut doc = serde_json::to_value(system(1, 1, exp(1.0), None)).unwrap();
    doc["components"][1]["equations"] = json!([]);
    let model: Model = serde_json::from_value(doc).unwrap();
    let model = CompiledModel::compile(&model).unwrap();
    let error = run_splitting(&model, &cut_settings(10, 1))
        .unwrap_err()
        .to_string();
    assert!(error.contains("no way"), "{error}");
    assert!(error.contains("attribute importance"), "{error}");
}

/// Structural extraction imposes no mission-time/CDF condition on rich laws.
#[test]
fn cut_set_structure_accepts_rich_laws_at_zero_horizon() {
    for law in [
        exp(0.2),
        json!({"distrib":"weibull","shape":2.0,"scale":10.0}),
        json!({"distrib":"lognormal","mu":2.0,"sigma":0.7}),
        json!({"distrib":"gamma","shape":3.0,"scale":2.0}),
        json!({"distrib":"uniform","low":2.0,"high":8.0}),
        json!({"distrib":"empirical","points":[[0.0,0.0],[3.0,0.4],[9.0,1.0]]}),
        json!({"distrib":"delay","time":1.0}),
    ] {
        let model = CompiledModel::compile(&system_vote(1, 1, law.clone())).unwrap();
        let mut settings = cut_settings(10, 1);
        settings.t_max = 0.0;
        settings.batches = 2;
        let result = run_splitting(&model, &settings).unwrap();
        assert_eq!(result.interval.estimate, 0.0, "{law}");
        assert_eq!(result.batches.len(), 2);
    }
}
