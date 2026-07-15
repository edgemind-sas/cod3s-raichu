//! M4 distribution library — statistical unit tests against closed-form CDFs.
//!
//! Identity under test: a single `ok → nok` automaton whose transition
//! carries distribution `L` gives `E[1{nok at t}] = P(T ≤ t) = CDF_L(t)`. Each
//! distribution is checked at three instants with a 4·SE tolerance (20 000
//! replicas, fixed seed — the verdict is reproducible).

#![allow(clippy::unwrap_used, clippy::panic)]

use raichu_core::CompiledModel;
use raichu_expr::{AttrRef, Expr, StateRef, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Distrib, Equation, EquationKind, Indicator,
    IndicatorTarget, Model, Transition,
};
use raichu_montecarlo::{run, McConfig};

fn single_law_model(distrib: Distrib) -> Model {
    Model {
        name: "law_probe".into(),
        components: vec![Component {
            name: "C".into(),
            attributes: vec![],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "aut".into(),
                states: vec!["ok".into(), "nok".into()],
                init: "ok".into(),
                transitions: vec![Transition {
                    name: "fire".into(),
                    source: "ok".into(),
                    guard: None,
                    targets: vec!["nok".into()],
                    on_interruption: Default::default(),
                    monitored: false,
                    cycle_group: None,
                    distrib,
                }],
            }],
            equations: vec![],
            sensitive_functions: vec![],
        }],
        connections: vec![],
        indicators: vec![Indicator {
            name: "fired".into(),
            target: IndicatorTarget::State {
                component: "C".into(),
                automaton: "aut".into(),
                state: "nok".into(),
            },
        }],
        targets: vec![],
    }
}

/// Estimate P(T ≤ t) at the given instants and compare to `cdf`.
fn assert_matches_cdf(distrib: Distrib, instants: &[f64], cdf: impl Fn(f64) -> f64) {
    assert_model_matches_cdf(&single_law_model(distrib), instants, cdf);
}

/// Same check for an arbitrary single-indicator model (first indicator
/// = the firing probability).
fn assert_model_matches_cdf(model: &Model, instants: &[f64], cdf: impl Fn(f64) -> f64) {
    let compiled = CompiledModel::compile(model).unwrap();
    let config = McConfig {
        nb_runs: 20_000,
        seed: 4242,
        t_max: *instants.last().unwrap(),
        samples: instants.to_vec(),
        threads: None,
        quantiles: vec![],
        ode: Default::default(),
        stop_at_targets: false,
    };
    let estimates = run(&compiled, &config).unwrap();
    let est = &estimates.indicators[0];
    let n = config.nb_runs as f64;
    for (k, &t) in instants.iter().enumerate() {
        let expected = cdf(t);
        let se = (est.std[k] / n.sqrt()).max(1e-6);
        assert!(
            (est.mean[k] - expected).abs() < 4.0 * se,
            "P(T <= {t}): estimated {} vs closed form {expected} (se {se})",
            est.mean[k]
        );
    }
}

/// Abramowitz–Stegun 7.1.26 erf approximation (|ε| < 1.5e-7 — far below
/// the Monte-Carlo standard error).
fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

#[test]
fn weibull_matches_closed_form() {
    let (shape, scale) = (2.0, 5.0);
    assert_matches_cdf(Distrib::Weibull { shape, scale }, &[2.0, 5.0, 9.0], |t| {
        1.0 - (-(t / scale).powf(shape)).exp()
    });
}

#[test]
fn lognormal_matches_closed_form() {
    let (mu, sigma) = (1.0, 0.5);
    assert_matches_cdf(Distrib::Lognormal { mu, sigma }, &[1.5, 3.0, 6.0], |t| {
        normal_cdf((t.ln() - mu) / sigma)
    });
}

#[test]
fn gamma_erlang2_matches_closed_form() {
    // shape = 2 (Erlang-2): F(t) = 1 − (1 + t/θ)·e^{−t/θ}.
    let (shape, scale) = (2.0, 3.0);
    assert_matches_cdf(Distrib::Gamma { shape, scale }, &[2.0, 6.0, 12.0], |t| {
        1.0 - (1.0 + t / scale) * (-t / scale).exp()
    });
}

#[test]
fn uniform_matches_closed_form() {
    let (low, high) = (2.0, 6.0);
    assert_matches_cdf(Distrib::Uniform { low, high }, &[3.0, 4.0, 5.0], |t| {
        ((t - low) / (high - low)).clamp(0.0, 1.0)
    });
}

#[test]
fn empirical_user_defined_matches_its_table() {
    // Mass 0.2 at t=1, plateau to t=2, linear ramp to 1 at t=4:
    // F(1)=0.2, F(2)=0.2, F(3)=0.6, F(4)=1.
    let points = vec![(1.0, 0.2), (2.0, 0.2), (4.0, 1.0)];
    assert_matches_cdf(Distrib::Empirical { points }, &[1.5, 3.0, 4.0], |t| {
        if t < 2.0 {
            0.2
        } else if t < 4.0 {
            0.2 + 0.8 * (t - 2.0) / 2.0
        } else {
            1.0
        }
    });
}

#[test]
fn quantiles_of_bernoulli_state_follow_the_probability() {
    // exp(λ = 0.2): p(t) = 1 − e^{−0.2 t}. The 0/1 state's nearest-rank
    // quantiles flip exactly where p crosses 1 − q.
    let model = single_law_model(Distrib::Exp {
        rate: Some(0.2),
        rate_expr: None,
    });
    let compiled = CompiledModel::compile(&model).unwrap();
    // p(3.466) ≈ 0.50; p(11.51) ≈ 0.90.
    let config = McConfig {
        nb_runs: 20_000,
        seed: 7,
        t_max: 12.0,
        samples: vec![3.466, 11.513],
        threads: None,
        quantiles: vec![0.25, 0.75],
        ode: Default::default(),
        stop_at_targets: false,
    };
    let estimates = run(&compiled, &config).unwrap();
    let est = &estimates.indicators[0];
    let q25 = &est.quantiles[0].values;
    let q75 = &est.quantiles[1].values;
    // At p ≈ 0.5: a quarter of replicas are surely below (still ok) —
    // q25 = 0 — and q75 = 1.
    assert_eq!((q25[0], q75[0]), (0.0, 1.0));
    // At p ≈ 0.9 > 0.75: even the 25 % rank has fired.
    assert_eq!((q25[1], q75[1]), (1.0, 1.0));
}

/// Shared shape for the state-dependent-rate probes: an `env` automaton
/// turns `hot` at t = 5 (delay distribution); the `aut` transition carries
/// `rate_expr = if hot { hot_rate } else { cold_rate }`.
fn expvar_switch_model(cold_rate: f64, hot_rate: f64) -> Model {
    Model {
        name: "expvar_pc".into(),
        components: vec![Component {
            name: "C".into(),
            attributes: vec![],
            ports: vec![],
            interfaces: vec![],
            automata: vec![
                Automaton {
                    name: "env".into(),
                    states: vec!["cold".into(), "hot".into()],
                    init: "cold".into(),
                    transitions: vec![Transition {
                        name: "heat".into(),
                        source: "cold".into(),
                        guard: None,
                        targets: vec!["hot".into()],
                        on_interruption: Default::default(),
                        monitored: false,
                        cycle_group: None,
                        distrib: Distrib::Delay { time: 5.0 },
                    }],
                },
                Automaton {
                    name: "aut".into(),
                    states: vec!["ok".into(), "nok".into()],
                    init: "ok".into(),
                    transitions: vec![Transition {
                        name: "fire".into(),
                        source: "ok".into(),
                        guard: None,
                        targets: vec!["nok".into()],
                        on_interruption: Default::default(),
                        monitored: false,
                        cycle_group: None,
                        distrib: Distrib::Exp {
                            rate: None,
                            rate_expr: Some(Expr::If {
                                cond: Box::new(Expr::StateActive {
                                    state: StateRef {
                                        component: "C".into(),
                                        automaton: "env".into(),
                                        state: "hot".into(),
                                    },
                                }),
                                then: Box::new(Expr::Const {
                                    value: Value::Float(hot_rate),
                                }),
                                otherwise: Box::new(Expr::Const {
                                    value: Value::Float(cold_rate),
                                }),
                            }),
                        },
                    }],
                },
            ],
            equations: vec![],
            sensitive_functions: vec![],
        }],
        connections: vec![],
        indicators: vec![Indicator {
            name: "fired".into(),
            target: IndicatorTarget::State {
                component: "C".into(),
                automaton: "aut".into(),
                state: "nok".into(),
            },
        }],
        targets: vec![],
    }
}

/// Piecewise-constant state-dependent rate (`reschedule_modifiable` proper): the
/// switch at t = 5 moves λ from 0.05 to 0.2, so the survival integral
/// gives `P(T ≤ t) = 1 − exp(−Λ(t))` with
/// `Λ(t) = 0.05·min(t, 5) + 0.2·max(t − 5, 0)`.
#[test]
fn expvar_piecewise_constant_rate_matches_closed_form() {
    let model = expvar_switch_model(0.05, 0.2);
    let hazard = |t: f64| 0.05 * t.min(5.0) + 0.2 * (t - 5.0).max(0.0);
    assert_model_matches_cdf(&model, &[3.0, 8.0, 10.0], |t| 1.0 - (-hazard(t)).exp());
}

/// Continuously-varying rate integrated along the ODE (`reschedule_modifiable`
/// under `integrate_continuous`): with `dx/dt = 1`, `x(0) = 0` and `λ(x) = 0.02·x`,
/// the cumulative hazard is `0.01·t²`, so
/// `P(T ≤ t) = 1 − exp(−0.01·t²)`.
#[test]
fn expvar_continuous_rate_matches_closed_form() {
    let model = Model {
        name: "expvar_ode".into(),
        components: vec![Component {
            name: "C".into(),
            attributes: vec![Attribute {
                name: "x".into(),
                kind: AttrKind::Float,
                init: Value::Float(0.0),
            }],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "aut".into(),
                states: vec!["ok".into(), "nok".into()],
                init: "ok".into(),
                transitions: vec![Transition {
                    name: "fire".into(),
                    source: "ok".into(),
                    guard: None,
                    targets: vec!["nok".into()],
                    on_interruption: Default::default(),
                    monitored: false,
                    cycle_group: None,
                    distrib: Distrib::Exp {
                        rate: None,
                        rate_expr: Some(Expr::Mul {
                            args: vec![
                                Expr::Const {
                                    value: Value::Float(0.02),
                                },
                                Expr::Attr {
                                    attr: AttrRef {
                                        component: "C".into(),
                                        attribute: "x".into(),
                                    },
                                },
                            ],
                        }),
                    },
                }],
            }],
            equations: vec![Equation {
                target: "x".into(),
                kind: EquationKind::Ode,
                expr: Expr::Const {
                    value: Value::Float(1.0),
                },
            }],
            sensitive_functions: vec![],
        }],
        connections: vec![],
        indicators: vec![Indicator {
            name: "fired".into(),
            target: IndicatorTarget::State {
                component: "C".into(),
                automaton: "aut".into(),
                state: "nok".into(),
            },
        }],
        targets: vec![],
    };
    assert_model_matches_cdf(&model, &[4.0, 8.0, 10.0], |t| 1.0 - (-0.01 * t * t).exp());
}
