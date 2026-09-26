//! Deferred-draw engine mode (the seam of the discretised sequence-tree
//! explorer).
//!
//! In deferred mode a stochastic transition is armed without a firing
//! date: the engine records its arming instant, its age under the
//! interruption policy and, for a state-dependent exponential, its
//! accumulated hazard, and the caller fires it at a chosen instant.
//! Drawn mode, the default, must stay bit-identical: the first section
//! pins a seeded characterisation of three models covering every law,
//! a continuously varying hazard and the three interruption policies.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::compile::CLaw;
use raichu_core::{
    CompiledModel, DeferredTransition, DropReason, Engine, EngineConfig, EngineError,
    JournalRecord, ProbeStop, SimulationResult, StochasticDates,
};
use raichu_expr::Value;
use raichu_model::Model;
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

fn ite(cond: Json, then: Json, otherwise: Json) -> Json {
    json!({"op": "if", "cond": cond, "then": then, "otherwise": otherwise})
}

fn mul(a: Json, b: Json) -> Json {
    json!({"op": "mul", "args": [a, b]})
}

fn ge(lhs: Json, rhs: Json) -> Json {
    json!({"op": "cmp", "cmp": "ge", "lhs": lhs, "rhs": rhs})
}

/// A transition from `source` to `targets` whose law fields are `law`
/// (`{"distrib": ..., ...}`), merged with the optional extras.
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

/// A repairable component `name.fail` (`ok`, `nok`).
fn repairable(name: &str, fail: Json, repair: Json) -> Json {
    component(
        name,
        vec![automaton(
            "fail",
            &["ok", "nok"],
            vec![
                tr("occ", "ok", &["nok"], fail),
                tr("rep", "nok", &["ok"], repair),
            ],
        )],
    )
}

fn state_indicator(name: &str, component: &str, automaton: &str, state: &str) -> Json {
    json!({"name": name, "target": "state", "component": component,
           "automaton": automaton, "state": state})
}

fn build(name: &str, components: Vec<Json>, indicators: Vec<Json>) -> Model {
    let model = json!({"name": name, "components": components, "indicators": indicators});
    Model::from_json(&model.to_string()).unwrap()
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

/// Every fixed law, a piecewise-constant state-dependent rate and a
/// probabilistic instantaneous branching, all repairable.
fn laws_model() -> Model {
    let demand = component(
        "E",
        vec![automaton(
            "demand",
            &["idle", "ask", "ok", "ko"],
            vec![
                tr("call", "idle", &["ask"], delay(3.0)),
                tr(
                    "answer",
                    "ask",
                    &["ok", "ko"],
                    json!({"distrib": "inst", "probs": [0.6]}),
                ),
                tr("done", "ok", &["idle"], delay(1.0)),
                tr("fix", "ko", &["idle"], exp(0.5)),
            ],
        )],
    );
    build(
        "laws",
        vec![
            repairable("A", exp(0.3), delay(2.0)),
            repairable(
                "B",
                weibull(2.0, 5.0),
                json!({"distrib": "uniform", "low": 1.0, "high": 3.0}),
            ),
            repairable(
                "C",
                json!({"distrib": "lognormal", "mu": 1.0, "sigma": 0.5}),
                json!({"distrib": "gamma", "shape": 2.0, "scale": 0.5}),
            ),
            repairable(
                "D",
                exp_expr(ite(active("A", "fail", "nok"), c(2.0), c(0.2))),
                json!({"distrib": "empirical",
                       "points": [[0.5, 0.3], [1.0, 0.7], [2.0, 1.0]]}),
            ),
            demand,
        ],
        vec![
            state_indicator("a_nok", "A", "fail", "nok"),
            state_indicator("b_nok", "B", "fail", "nok"),
            state_indicator("c_nok", "C", "fail", "nok"),
            state_indicator("d_nok", "D", "fail", "nok"),
            state_indicator("e_ko", "E", "demand", "ko"),
        ],
    )
}

/// `x' = slope` from `x0`; `F` fails at the continuously varying rate
/// `k x`; `W` switches when `x >= 6` (watched), which pauses `H`
/// (continuous rate, `resume`) and arms `G`.
fn continuous_model(x0: f64, slope: f64, k: f64) -> Model {
    let x = json!({"name": "X", "ports": [],
        "attributes": [{"name": "x", "kind": "float", "init": {"kind": "float", "value": x0}}],
        "automata": [],
        "equations": [{"target": "x", "kind": "ode", "expr": c(slope)}]});
    let watch = component(
        "W",
        vec![automaton(
            "watch",
            &["lo", "hi"],
            vec![with(
                tr("hit", "lo", &["hi"], json!({"distrib": "watched"})),
                "guard",
                ge(attr("X", "x"), c(6.0)),
            )],
        )],
    );
    let h_fail = with(
        with(
            tr(
                "occ",
                "ok",
                &["nok"],
                exp_expr(mul(c(0.02), attr("X", "x"))),
            ),
            "guard",
            active("W", "watch", "lo"),
        ),
        "on_interruption",
        json!("resume"),
    );
    let h = component(
        "H",
        vec![automaton(
            "fail",
            &["ok", "nok"],
            vec![h_fail, tr("rep", "nok", &["ok"], delay(1.0))],
        )],
    );
    let g_fail = with(
        tr("occ", "ok", &["nok"], exp(0.1)),
        "guard",
        active("W", "watch", "hi"),
    );
    let g = component(
        "G",
        vec![automaton(
            "fail",
            &["ok", "nok"],
            vec![g_fail, tr("rep", "nok", &["ok"], delay(1.0))],
        )],
    );
    build(
        "continuous",
        vec![
            x,
            repairable("F", exp_expr(mul(c(k), attr("X", "x"))), delay(1.0)),
            watch,
            h,
            g,
        ],
        vec![
            json!({"name": "x", "target": "attribute",
                   "attr": {"component": "X", "attribute": "x"}}),
            state_indicator("f_nok", "F", "fail", "nok"),
            state_indicator("h_nok", "H", "fail", "nok"),
            state_indicator("g_nok", "G", "fail", "nok"),
        ],
    )
}

/// A worker `name.job` (`wait`, `done`) whose `finish` follows `law`
/// while the gate is up, under `policy`, and restarts after a delay.
fn worker(name: &str, law: Json, policy: &str) -> Json {
    let finish = with(
        with(
            tr("finish", "wait", &["done"], law),
            "guard",
            active("gate", "phase", "up"),
        ),
        "on_interruption",
        json!(policy),
    );
    component(
        name,
        vec![automaton(
            "job",
            &["wait", "done"],
            vec![finish, tr("restart", "done", &["wait"], delay(1.0))],
        )],
    )
}

/// A gate cycling up (3) / down (2) guarding four workers, one per
/// interruption policy, plus a piecewise state-dependent rate paused
/// under `resume`.
fn interrupt_model() -> Model {
    let gate = component(
        "gate",
        vec![automaton(
            "phase",
            &["up", "down"],
            vec![
                tr("close", "up", &["down"], delay(3.0)),
                tr("open", "down", &["up"], delay(2.0)),
            ],
        )],
    );
    build(
        "interrupt",
        vec![
            gate,
            worker("r", weibull(1.5, 4.0), "reset"),
            worker(
                "s",
                json!({"distrib": "gamma", "shape": 2.0, "scale": 2.0}),
                "resume",
            ),
            worker("k", exp(0.25), "continue"),
            worker(
                "v",
                exp_expr(ite(active("r", "job", "done"), c(0.8), c(0.3))),
                "resume",
            ),
        ],
        vec![
            state_indicator("r_done", "r", "job", "done"),
            state_indicator("s_done", "s", "job", "done"),
            state_indicator("k_done", "k", "job", "done"),
            state_indicator("v_done", "v", "job", "done"),
        ],
    )
}

// ---- drawn-mode characterisation ------------------------------------------

/// FNV-1a over bytes: a hash whose value does not depend on the
/// toolchain, unlike `DefaultHasher`.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn f64(&mut self, x: f64) {
        self.bytes(&x.to_bits().to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
        self.bytes(&[0xff]);
    }
    fn value(&mut self, v: &Value) {
        self.str(&format!("{v:?}"));
    }
}

fn hash_result(h: &mut Fnv, result: &SimulationResult) {
    for e in &result.events {
        h.f64(e.time);
        h.str(&e.transition);
        h.str(&e.from);
        h.str(&e.to);
    }
    for series in result.indicators.iter().chain(&result.samples) {
        h.str(&series.name);
        for (t, v) in &series.points {
            h.f64(*t);
            h.value(v);
        }
    }
    h.f64(result.final_time);
}

/// Hash of 50 seeded trajectories of `model` in drawn mode (the default).
fn characterise(model: &Model, t_max: f64, samples: &[f64]) -> (u64, usize) {
    let compiled = CompiledModel::compile(model).unwrap();
    let mut h = Fnv::new();
    let mut events = 0;
    for seed in 0..50 {
        let config = EngineConfig {
            t_max,
            seed,
            samples: samples.to_vec(),
            ..EngineConfig::default()
        };
        let result = Engine::new(&compiled, config).unwrap().run().unwrap();
        events += result.events.len();
        hash_result(&mut h, &result);
    }
    (h.0, events)
}

/// Compare a characterisation with its pinned hashes.
///
/// A model drawing from a non-exponential law has **two** pinned hashes,
/// one per floating-point math backend of `rand_distr`: `libm` when
/// `raichu-core` is built alone, the platform `std` functions when the
/// build unifies `num-traits/std` in (a workspace build does, through
/// `proptest`). The two differ in the last bits of some drawn dates, not
/// in the event count; both were measured on the tree before the
/// deferred mode existed (2026-09-26), and a change to drawn mode moves
/// both.
fn assert_pinned(got: (u64, usize), pinned: (u64, u64, usize), model: &str) {
    let (libm, std_math, events) = pinned;
    assert_eq!(got.1, events, "{model} model: event count moved");
    assert!(
        got.0 == libm || got.0 == std_math,
        "{model} model characterisation moved: {got:?}"
    );
}

#[test]
fn drawn_mode_is_bit_identical_on_every_law() {
    let got = characterise(&laws_model(), 30.0, &[5.0, 10.0, 20.0]);
    assert_pinned(
        got,
        (16_145_868_882_581_082_605, 805_835_540_276_411_409, 3844),
        "laws",
    );
}

#[test]
fn drawn_mode_is_bit_identical_on_a_continuously_varying_hazard() {
    let samples: Vec<f64> = (1..=15).map(f64::from).collect();
    let got = characterise(&continuous_model(1.0, 0.5, 0.05), 15.0, &samples);
    assert_eq!(
        got,
        (5_608_455_318_325_514_104, 436),
        "continuous model characterisation moved"
    );
}

#[test]
fn drawn_mode_is_bit_identical_under_every_interruption_policy() {
    let got = characterise(&interrupt_model(), 40.0, &[10.0, 20.0, 40.0]);
    assert_pinned(
        got,
        (9_351_900_170_923_758_607, 18_400_255_398_688_170_659, 3151),
        "interrupt",
    );
}

// ---- deferred mode ----------------------------------------------------------

fn deferred_config() -> EngineConfig {
    EngineConfig {
        stochastic_dates: StochasticDates::Deferred,
        ..EngineConfig::default()
    }
}

fn index(compiled: &CompiledModel, name: &str) -> usize {
    compiled
        .transitions
        .iter()
        .position(|t| t.name == name)
        .unwrap()
}

fn find<'a>(deferred: &'a [DeferredTransition], name: &str) -> Option<&'a DeferredTransition> {
    deferred.iter().find(|d| d.transition == name)
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

/// Delays only: a gate cycle, an integrated attribute crossing a watched
/// boundary, and delayed workers under each interruption policy.
fn deterministic_model() -> Model {
    let x = json!({"name": "X", "ports": [],
        "attributes": [{"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}],
        "automata": [],
        "equations": [{"target": "x", "kind": "ode", "expr": c(0.7)}]});
    let gate = component(
        "gate",
        vec![automaton(
            "phase",
            &["up", "down"],
            vec![
                tr("close", "up", &["down"], delay(3.0)),
                tr("open", "down", &["up"], delay(2.0)),
            ],
        )],
    );
    let watch = component(
        "W",
        vec![automaton(
            "watch",
            &["lo", "hi"],
            vec![with(
                tr("hit", "lo", &["hi"], json!({"distrib": "watched"})),
                "guard",
                ge(attr("X", "x"), c(6.0)),
            )],
        )],
    );
    build(
        "deterministic",
        vec![
            x,
            gate,
            watch,
            worker("r", delay(2.5), "reset"),
            worker("s", delay(4.0), "resume"),
            worker("k", delay(3.5), "continue"),
        ],
        vec![
            json!({"name": "x", "target": "attribute",
                   "attr": {"component": "X", "attribute": "x"}}),
            state_indicator("r_done", "r", "job", "done"),
            state_indicator("s_done", "s", "job", "done"),
            state_indicator("k_done", "k", "job", "done"),
            state_indicator("hi", "W", "watch", "hi"),
        ],
    )
}

#[test]
fn a_deterministic_model_runs_identically_in_both_modes() {
    let compiled = CompiledModel::compile(&deterministic_model()).unwrap();
    let run = |dates| {
        let config = EngineConfig {
            t_max: 30.0,
            samples: vec![1.0, 7.5, 12.0, 30.0],
            stochastic_dates: dates,
            ..EngineConfig::default()
        };
        let result = Engine::new(&compiled, config).unwrap().run().unwrap();
        assert!(!result.events.is_empty());
        let mut h = Fnv::new();
        hash_result(&mut h, &result);
        (h.0, result.events)
    };
    assert_eq!(run(StochasticDates::Drawn), run(StochasticDates::Deferred));
}

#[test]
fn deferred_run_fires_no_stochastic_transition() {
    let compiled = CompiledModel::compile(&laws_model()).unwrap();
    let config = EngineConfig {
        t_max: 30.0,
        ..deferred_config()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    // Only the demand cycle moves: its delays and its instantaneous
    // branching, which is not a date and is resolved as in drawn mode.
    assert!(!result.events.is_empty());
    for event in &result.events {
        assert!(event.transition.starts_with("E.demand."), "{event:?}");
        assert_ne!(event.transition, "E.demand.fix", "{event:?}");
    }
}

#[test]
fn deferred_arming_reports_law_arming_instant_and_age() {
    let compiled = CompiledModel::compile(&laws_model()).unwrap();
    let engine = Engine::new(&compiled, deferred_config()).unwrap();
    let deferred = engine.deferred();
    let names: Vec<&str> = deferred.iter().map(|d| d.transition.as_str()).collect();
    assert_eq!(
        names,
        ["A.fail.occ", "B.fail.occ", "C.fail.occ", "D.fail.occ"]
    );
    let b = find(&deferred, "B.fail.occ").unwrap();
    assert_eq!(b.index, index(&compiled, "B.fail.occ"));
    assert!(matches!(b.law, CLaw::Weibull(shape, scale) if shape == 2.0 && scale == 5.0));
    assert_eq!(
        (b.armed_at, b.age, b.paused, b.hazard),
        (0.0, 0.0, false, None)
    );
    // A state-dependent rate carries its accumulated hazard.
    assert_eq!(find(&deferred, "D.fail.occ").unwrap().hazard, Some(0.0));
    // A deferred transition is armed without a date.
    for fireable in engine.fireable() {
        if deferred.iter().any(|d| d.index == fireable.index) {
            assert_eq!(fireable.date, None, "{fireable:?}");
        }
    }
}

/// A gate up on [0, 4) and from 7 on (`up1`, `down`, `up2`), guarding
/// three Weibull workers (one per policy) and driving the piecewise
/// rate of an unguarded worker: 0.5 in `up1`, 0.25 in `down`, 1 in `up2`.
fn policy_model() -> Model {
    let gate = component(
        "gate",
        vec![automaton(
            "phase",
            &["up1", "down", "up2"],
            vec![
                tr("close", "up1", &["down"], delay(4.0)),
                tr("reopen", "down", &["up2"], delay(3.0)),
            ],
        )],
    );
    let up = json!({"op": "bool", "bool_op": "or",
        "args": [active("gate", "phase", "up1"), active("gate", "phase", "up2")]});
    let job = |name: &str, policy: &str| {
        let finish = with(
            with(
                tr("finish", "wait", &["done"], weibull(2.0, 10.0)),
                "guard",
                up.clone(),
            ),
            "on_interruption",
            json!(policy),
        );
        component(
            name,
            vec![automaton("job", &["wait", "done"], vec![finish])],
        )
    };
    let rate = ite(
        active("gate", "phase", "up1"),
        c(0.5),
        ite(active("gate", "phase", "down"), c(0.25), c(1.0)),
    );
    let v = component(
        "v",
        vec![automaton(
            "job",
            &["wait", "done"],
            vec![tr("finish", "wait", &["done"], exp_expr(rate))],
        )],
    );
    build(
        "policies",
        vec![
            gate,
            job("r", "reset"),
            job("s", "resume"),
            job("k", "continue"),
            v,
        ],
        vec![],
    )
}

#[test]
fn age_restarts_pauses_or_runs_on_under_each_policy() {
    let compiled = CompiledModel::compile(&policy_model()).unwrap();
    let mut engine = Engine::new(&compiled, deferred_config()).unwrap();
    let ages = |engine: &Engine<'_>| {
        let deferred = engine.deferred();
        ["r", "s", "k"]
            .map(|w| find(&deferred, &format!("{w}.job.finish")).map(|d| (d.age, d.paused)))
    };
    assert_eq!(
        ages(&engine),
        [Some((0.0, false)), Some((0.0, false)), Some((0.0, false))]
    );

    // t = 4: the gate closes. reset drops the countdown, resume freezes
    // it, continue keeps it running.
    let event = engine.step().unwrap().unwrap();
    assert_eq!(
        (event.time, event.transition.as_str()),
        (4.0, "gate.phase.close")
    );
    assert_eq!(ages(&engine), [None, Some((4.0, true)), Some((4.0, false))]);

    // t = 7: the gate reopens. reset re-arms from zero, resume resumes
    // at the frozen age.
    let event = engine.step().unwrap().unwrap();
    assert_eq!(
        (event.time, event.transition.as_str()),
        (7.0, "gate.phase.reopen")
    );
    assert_eq!(
        ages(&engine),
        [Some((0.0, false)), Some((4.0, false)), Some((7.0, false))]
    );
    let deferred = engine.deferred();
    assert_eq!(find(&deferred, "r.job.finish").unwrap().armed_at, 7.0);
    assert_eq!(find(&deferred, "s.job.finish").unwrap().armed_at, 0.0);

    // Two more units with nothing deterministic left.
    let probe = engine.probe_deferred(9.0).unwrap();
    assert_eq!((probe.stop, probe.reason), (9.0, ProbeStop::Limit));
    assert_eq!(engine.current_time(), 9.0);
    assert_eq!(
        ages(&engine),
        [Some((2.0, false)), Some((6.0, false)), Some((9.0, false))]
    );

    // The piecewise rate: 0.5 x 4 + 0.25 x 3 + 1 x 2.
    let v = engine.deferred();
    let v = find(&v, "v.job.finish").unwrap();
    assert_eq!(v.age, 9.0);
    assert!(close(v.hazard.unwrap(), 4.75, 1e-14), "{v:?}");
}

#[test]
fn a_continuous_hazard_matches_its_analytic_integral() {
    // x' = 1 from 1 and λ = x: H(t) = t + t²/2, the watched boundary x >= 6
    // at t = 5 far away.
    let compiled = CompiledModel::compile(&continuous_model(1.0, 1.0, 1.0)).unwrap();
    let mut engine = Engine::new(&compiled, deferred_config()).unwrap();
    let probe = engine.probe_deferred(3.0).unwrap();
    assert_eq!((probe.stop, probe.reason), (3.0, ProbeStop::Limit));
    let f = index(&compiled, "F.fail.occ");
    let h = index(&compiled, "H.fail.occ");
    assert_eq!(probe.transitions, vec![f, h]);
    assert!(probe.samples.len() > 10, "{}", probe.samples.len());
    assert_eq!(probe.samples[0].time, 0.0);
    assert_eq!(probe.samples.last().unwrap().time, 3.0);
    let mut previous = f64::NEG_INFINITY;
    for sample in &probe.samples {
        assert!(sample.time > previous);
        previous = sample.time;
        let t = sample.time;
        let exact = t + t * t / 2.0;
        assert!(close(sample.hazards[0], exact, 1e-9), "{t}: {sample:?}");
        assert!(
            close(sample.hazards[1], 0.02 * exact, 1e-9),
            "{t}: {sample:?}"
        );
    }
    let deferred = engine.deferred();
    let hazard = find(&deferred, "F.fail.occ").unwrap().hazard.unwrap();
    assert!(close(hazard, 7.5, 1e-9), "{hazard}");
    assert!(close(
        engine.attribute("X.x").map_or(0.0, |v| match v {
            Value::Float(x) => x,
            _ => f64::NAN,
        }),
        4.0,
        1e-12
    ));
}

#[test]
fn a_probe_stops_at_the_next_deterministic_event_without_firing_it() {
    let compiled = CompiledModel::compile(&policy_model()).unwrap();
    let mut engine = Engine::new(&compiled, deferred_config()).unwrap();
    let probe = engine.probe_deferred(100.0).unwrap();
    let close_idx = index(&compiled, "gate.phase.close");
    assert_eq!(
        (probe.stop, probe.reason),
        (4.0, ProbeStop::Deterministic { index: close_idx })
    );
    assert_eq!(engine.current_time(), 4.0);
    assert_eq!(engine.state("gate.phase"), Some("up1"));
    assert!(engine.history().is_empty());
    // No continuous hazard: no dense samples.
    assert!(probe.transitions.is_empty() && probe.samples.is_empty());
}

#[test]
fn a_probe_stops_at_a_located_watched_crossing() {
    // x' = 0.5 from 1 reaches 6 at t = 10.
    let compiled = CompiledModel::compile(&continuous_model(1.0, 0.5, 0.05)).unwrap();
    let mut engine = Engine::new(&compiled, deferred_config()).unwrap();
    let probe = engine.probe_deferred(20.0).unwrap();
    let hit = index(&compiled, "W.watch.hit");
    assert_eq!(probe.reason, ProbeStop::Watched { index: hit });
    assert!((probe.stop - 10.0).abs() < 1e-8, "{}", probe.stop);
    assert_eq!(engine.state("W.watch"), Some("lo"));
    // The hazard trace ends at the crossing.
    assert_eq!(probe.samples.last().unwrap().time, probe.stop);
}

#[test]
fn a_probe_is_capped_by_the_horizon() {
    let compiled = CompiledModel::compile(&continuous_model(1.0, 0.5, 0.05)).unwrap();
    let config = EngineConfig {
        t_max: 2.0,
        ..deferred_config()
    };
    let mut engine = Engine::new(&compiled, config).unwrap();
    let probe = engine.probe_deferred(5.0).unwrap();
    assert_eq!((probe.stop, probe.reason), (2.0, ProbeStop::Horizon));
}

#[test]
fn a_probe_in_drawn_mode_or_into_the_past_is_a_typed_error() {
    let compiled = CompiledModel::compile(&policy_model()).unwrap();
    let mut drawn = Engine::new(&compiled, EngineConfig::default()).unwrap();
    let err = drawn.probe_deferred(1.0).unwrap_err();
    assert!(
        matches!(err, EngineError::NotDeferredMode { .. }),
        "{err:?}"
    );

    let mut engine = Engine::new(&compiled, deferred_config()).unwrap();
    engine.probe_deferred(2.0).unwrap();
    let err = engine.probe_deferred(1.0).unwrap_err();
    assert!(matches!(err, EngineError::DateInPast { .. }), "{err:?}");
    assert_eq!(engine.current_time(), 2.0);
}

/// The discrete state of every automaton, by qualified name.
fn discrete_state(compiled: &CompiledModel, engine: &Engine<'_>) -> Vec<(String, String)> {
    compiled
        .automata
        .iter()
        .map(|a| {
            let qualified = a.name.clone();
            let state = engine.state(&qualified).unwrap().to_owned();
            (qualified, state)
        })
        .collect()
}

#[test]
fn firing_a_deferred_transition_matches_drawn_mode_firing_at_that_date() {
    let compiled = CompiledModel::compile(&laws_model()).unwrap();
    for name in ["A.fail.occ", "B.fail.occ", "C.fail.occ", "D.fail.occ"] {
        let idx = index(&compiled, name);
        let mut deferred = Engine::new(&compiled, deferred_config()).unwrap();
        let fired = deferred.fire_deferred_at(idx, 2.5, None).unwrap();

        let mut drawn = Engine::new(&compiled, EngineConfig::default()).unwrap();
        drawn.set_date_idx(idx, 2.5).unwrap();
        let expected = drawn.fire_idx(idx).unwrap();

        assert_eq!(fired, expected);
        assert_eq!(deferred.current_time(), 2.5);
        assert_eq!(
            discrete_state(&compiled, &deferred),
            discrete_state(&compiled, &drawn)
        );
        // The fired transition is no longer deferred; the others still
        // are, and kept aging; its stochastic repair is armed at 2.5.
        let listed = deferred.deferred();
        assert!(find(&listed, name).is_none());
        let repair = name.replace(".occ", ".rep");
        for other in &listed {
            let expected = if other.transition == repair {
                (2.5, 0.0)
            } else {
                (0.0, 2.5)
            };
            assert_eq!((other.armed_at, other.age), expected, "{other:?}");
        }
        assert_eq!(listed.len(), if name == "A.fail.occ" { 3 } else { 4 });
    }
}

#[test]
fn firing_a_deferred_continuous_hazard_matches_drawn_mode() {
    let compiled = CompiledModel::compile(&continuous_model(1.0, 0.5, 0.05)).unwrap();
    let h = index(&compiled, "H.fail.occ");
    let mut deferred = Engine::new(&compiled, deferred_config()).unwrap();
    let fired = deferred.fire_deferred_at(h, 2.5, Some(0)).unwrap();
    // Drawn mode can fire the same transition at 2.5 only on a seed
    // whose own draws leave both hazards short of their thresholds.
    let expected = (0..50)
        .find_map(|seed| {
            let config = EngineConfig {
                seed,
                ..EngineConfig::default()
            };
            let mut drawn = Engine::new(&compiled, config).unwrap();
            drawn.set_date_idx(h, 2.5).unwrap();
            let event = drawn.fire_idx(h).unwrap();
            (event.transition == "H.fail.occ" && event.time == 2.5).then_some((event, drawn))
        })
        .expect("a seed firing H at 2.5");
    assert_eq!(fired, expected.0);
    assert_eq!(
        discrete_state(&compiled, &deferred),
        discrete_state(&compiled, &expected.1)
    );
    let (Some(Value::Float(a)), Some(Value::Float(b))) =
        (deferred.attribute("X.x"), expected.1.attribute("X.x"))
    else {
        panic!("x is a float");
    };
    assert!(close(a, 2.25, 1e-12) && close(b, 2.25, 1e-12), "{a} {b}");
    // F keeps aging and accruing: 0.05 (2.5 + 0.25 x 2.5²).
    let listed = deferred.deferred();
    let f = find(&listed, "F.fail.occ").unwrap();
    assert_eq!(f.age, 2.5);
    assert!(
        close(f.hazard.unwrap(), 0.05 * (2.5 + 0.25 * 6.25), 1e-9),
        "{f:?}"
    );
}

#[test]
fn firing_in_the_past_is_a_typed_error() {
    let compiled = CompiledModel::compile(&policy_model()).unwrap();
    let mut engine = Engine::new(&compiled, deferred_config()).unwrap();
    engine.probe_deferred(3.0).unwrap();
    let r = index(&compiled, "r.job.finish");
    let err = engine.fire_deferred_at(r, 2.0, None).unwrap_err();
    assert!(matches!(err, EngineError::DateInPast { .. }), "{err:?}");
    let err = engine.fire_deferred_at(r, f64::NAN, None).unwrap_err();
    assert!(matches!(err, EngineError::DateInPast { .. }), "{err:?}");
    assert_eq!(engine.current_time(), 3.0);
    assert_eq!(engine.state("r.job"), Some("wait"));
}

#[test]
fn firing_after_the_next_deterministic_event_is_a_typed_error() {
    let compiled = CompiledModel::compile(&policy_model()).unwrap();
    let mut engine = Engine::new(&compiled, deferred_config()).unwrap();
    let r = index(&compiled, "r.job.finish");
    let err = engine.fire_deferred_at(r, 5.0, None).unwrap_err();
    assert!(
        matches!(&err, EngineError::DeferredFiringTooLate { limit, .. } if *limit == 4.0),
        "{err:?}"
    );
    assert_eq!(engine.current_time(), 0.0);
    // At the event date itself it fires, before the deterministic event.
    let event = engine.fire_deferred_at(r, 4.0, None).unwrap();
    assert_eq!((event.time, event.to.as_str()), (4.0, "done"));
    assert_eq!(engine.state("gate.phase"), Some("up1"));
}

#[test]
fn firing_after_a_watched_crossing_is_a_typed_error_and_changes_nothing() {
    let compiled = CompiledModel::compile(&continuous_model(1.0, 0.5, 0.05)).unwrap();
    let mut engine = Engine::new(&compiled, deferred_config()).unwrap();
    let f = index(&compiled, "F.fail.occ");
    let err = engine.fire_deferred_at(f, 12.0, None).unwrap_err();
    let EngineError::DeferredFiringTooLate { limit, .. } = err else {
        panic!("{err:?}");
    };
    assert!((limit - 10.0).abs() < 1e-8, "{limit}");
    assert_eq!(engine.current_time(), 0.0);
    assert_eq!(engine.attribute("X.x"), Some(Value::Float(1.0)));
    assert_eq!(
        find(&engine.deferred(), "F.fail.occ").unwrap().hazard,
        Some(0.0)
    );
}

#[test]
fn firing_an_unarmed_or_paused_transition_is_not_fireable() {
    let compiled = CompiledModel::compile(&policy_model()).unwrap();
    let mut engine = Engine::new(&compiled, deferred_config()).unwrap();
    engine.step().unwrap(); // the gate closes: s is paused, r dropped
    for name in ["r.job.finish", "s.job.finish", "gate.phase.reopen"] {
        let err = engine
            .fire_deferred_at(index(&compiled, name), 4.5, None)
            .unwrap_err();
        assert!(
            matches!(err, EngineError::NotFireable { .. }),
            "{name}: {err:?}"
        );
    }
    assert_eq!(engine.current_time(), 4.0);
}

#[test]
fn set_date_on_a_deferred_transition_is_a_typed_error() {
    let compiled = CompiledModel::compile(&laws_model()).unwrap();
    let mut engine = Engine::new(&compiled, deferred_config()).unwrap();
    let err = engine.set_date("A.fail.occ", 1.0).unwrap_err();
    assert!(matches!(err, EngineError::DeferredDate { .. }), "{err:?}");
    // A deterministic transition keeps its manual dating.
    engine.set_date("E.demand.call", 1.0).unwrap();
}

#[test]
fn a_snapshot_restores_the_deferred_bookkeeping() {
    let compiled = CompiledModel::compile(&policy_model()).unwrap();
    let mut engine = Engine::new(&compiled, deferred_config()).unwrap();
    let before = engine.snapshot();
    let ages_before: Vec<f64> = engine.deferred().iter().map(|d| d.age).collect();
    engine.probe_deferred(3.0).unwrap();
    engine
        .fire_deferred_at(index(&compiled, "k.job.finish"), 3.5, None)
        .unwrap();
    engine.restore(&before);
    let ages_after: Vec<f64> = engine.deferred().iter().map(|d| d.age).collect();
    assert_eq!(ages_before, ages_after);
    assert_eq!(engine.deferred().len(), 4);
}

/// One unit with two competing failure modes from `ok` (a Weibull wear-out
/// and an exponential shock), each repaired after a delay of 0.5.
fn competing_modes_model() -> Model {
    build(
        "competing_modes",
        vec![component(
            "M",
            vec![automaton(
                "fail",
                &["ok", "worn", "shocked"],
                vec![
                    tr("wear", "ok", &["worn"], weibull(2.0, 3.0)),
                    tr("shock", "ok", &["shocked"], exp(0.4)),
                    tr("fix_wear", "worn", &["ok"], delay(0.5)),
                    tr("fix_shock", "shocked", &["ok"], delay(0.5)),
                ],
            )],
        )],
        vec![],
    )
}

#[test]
fn a_competitor_leaving_its_source_is_dropped_with_its_age() {
    let compiled = CompiledModel::compile(&competing_modes_model()).unwrap();
    let config = EngineConfig {
        journal: true,
        t_max: 3.0,
        ..deferred_config()
    };
    let mut engine = Engine::new(&compiled, config).unwrap();
    let listed = engine.deferred();
    assert!(find(&listed, "M.fail.wear").is_some());
    assert!(find(&listed, "M.fail.shock").is_some());

    // The wear-out fires first: the shock, armed from the same source, is
    // dropped, not merely left unfired.
    engine
        .fire_deferred_at(index(&compiled, "M.fail.wear"), 1.0, None)
        .unwrap();
    assert!(engine.deferred().is_empty(), "{:?}", engine.deferred());

    // Back in `ok` at 1.5: both are armed afresh, with no age carried over
    // from before the drop.
    let repaired = engine.step().unwrap().unwrap();
    assert_eq!(repaired.transition, "M.fail.fix_wear");
    assert_eq!(engine.current_time(), 1.5);
    let listed = engine.deferred();
    for name in ["M.fail.wear", "M.fail.shock"] {
        let armed = find(&listed, name).unwrap();
        assert_eq!((armed.armed_at, armed.age), (1.5, 0.0), "{armed:?}");
    }

    let result = engine.run().unwrap();
    let dropped: Vec<(f64, &str, DropReason)> = result
        .journal
        .iter()
        .filter_map(|record| match record {
            JournalRecord::TransitionDropped {
                time,
                transition,
                reason,
            } => Some((*time, transition.as_str(), *reason)),
            _ => None,
        })
        .collect();
    assert_eq!(dropped, vec![(1.0, "M.fail.shock", DropReason::SourceLeft)]);
}
