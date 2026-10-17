//! The indexed schedule refresh: what rescheduling after a discrete step
//! must cost, and what it must never change.
//!
//! After every firing, `schedule_deterministic` and `drop_disabled` used
//! to walk every transition of the model and re-evaluate its guard. A
//! failure of one pump cannot change whether a transition reading nothing
//! of that pump is armed, so on a model of `n` independent components the
//! walk made each event cost `O(n)` and a trajectory `O(n²)`.
//!
//! Two claims, as for the watched index. The **behavioural** one: the
//! trajectory is bit-identical to the full walk's, random draws included,
//! on a model where guards read attributes and automaton states, a rate
//! depends on the state, and both interruption policies apply. The pinned
//! trace was recorded on the commit that precedes the index, with the full
//! walk in place. The **cost** one lives beside the engine
//! (`engine/schedule.rs`), which reads its crate-internal refresh count:
//! after the initialization axiom, a firing refreshes the transitions that
//! read what it changed, not the model.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, SimulationResult};
use raichu_expr::{Assignment, AttrRef, BoolOp, CmpOp, Expr, StateRef, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Distrib, InterruptionPolicy, Model,
    SensitiveFunction, Transition,
};

fn transition(
    name: &str,
    source: &str,
    target: &str,
    guard: Option<Expr>,
    distrib: Distrib,
) -> Transition {
    Transition {
        name: name.into(),
        source: source.into(),
        guard,
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

fn exp(rate: f64) -> Distrib {
    Distrib::Exp {
        rate: Some(rate),
        rate_expr: None,
    }
}

fn state(component: &str, automaton: &str, state: &str) -> Expr {
    Expr::StateActive {
        state: StateRef {
            component: component.into(),
            automaton: automaton.into(),
            state: state.into(),
        },
    }
}

fn is(component: &str, attribute: &str, value: bool) -> Expr {
    Expr::Cmp {
        cmp: CmpOp::Eq,
        lhs: Box::new(Expr::attr(component, attribute)),
        rhs: Box::new(Expr::Const {
            value: Value::Bool(value),
        }),
    }
}

/// A repairable pump: `ok` follows its failure automaton.
fn pump(index: usize) -> Component {
    let name = format!("P{index}");
    Component {
        name: name.clone(),
        attributes: vec![Attribute {
            name: "ok".into(),
            kind: AttrKind::Bool,
            init: Value::Bool(true),
        }],
        ports: vec![],
        interfaces: vec![],
        automata: vec![Automaton {
            name: "fm".into(),
            states: vec!["ok".into(), "ko".into()],
            init: "ok".into(),
            transitions: vec![
                transition("fail", "ok", "ko", None, exp(0.01 + 0.001 * index as f64)),
                transition("repair", "ko", "ok", None, exp(0.1)),
            ],
        }],
        allocations: vec![],
        equations: vec![],
        sensitive_functions: vec![SensitiveFunction {
            name: "apply".into(),
            effects: vec![Assignment {
                target: AttrRef {
                    component: name.clone(),
                    attribute: "ok".into(),
                },
                value: Expr::Bool {
                    bool_op: BoolOp::Not,
                    args: vec![state(&name, "fm", "ko")],
                },
            }],
        }],
    }
}

/// A component whose transitions read the pumps: a standby that starts
/// after a delay while P0 is down (reset), a wear-out whose rate depends
/// on P1 (state-dependent, rescheduled), and an alarm that a failure of P2
/// arms and a repair pauses (resume).
fn supervisor() -> Component {
    let mut start = transition(
        "start",
        "idle",
        "run",
        Some(is("P0", "ok", false)),
        Distrib::Delay { time: 2.0 },
    );
    start.on_interruption = InterruptionPolicy::Reset;
    let stop = transition(
        "stop",
        "run",
        "idle",
        Some(is("P0", "ok", true)),
        Distrib::Delay { time: 0.5 },
    );
    let wear = transition(
        "wear",
        "new",
        "worn",
        None,
        Distrib::Exp {
            rate: None,
            rate_expr: Some(Expr::If {
                cond: Box::new(is("P1", "ok", true)),
                then: Box::new(Expr::Const {
                    value: Value::Float(0.002),
                }),
                otherwise: Box::new(Expr::Const {
                    value: Value::Float(0.3),
                }),
            }),
        },
    );
    let renew = transition("renew", "worn", "new", None, exp(0.05));
    let mut raise = transition(
        "raise",
        "off",
        "on",
        Some(state("P2", "fm", "ko")),
        Distrib::Weibull {
            shape: 1.5,
            scale: 4.0,
        },
    );
    raise.on_interruption = InterruptionPolicy::Resume;
    let clear = transition("clear", "on", "off", None, exp(0.2));
    Component {
        name: "sup".into(),
        attributes: vec![],
        ports: vec![],
        interfaces: vec![],
        automata: vec![
            Automaton {
                name: "standby".into(),
                states: vec!["idle".into(), "run".into()],
                init: "idle".into(),
                transitions: vec![start, stop],
            },
            Automaton {
                name: "wear".into(),
                states: vec!["new".into(), "worn".into()],
                init: "new".into(),
                transitions: vec![wear, renew],
            },
            Automaton {
                name: "alarm".into(),
                states: vec!["off".into(), "on".into()],
                init: "off".into(),
                transitions: vec![raise, clear],
            },
        ],
        allocations: vec![],
        equations: vec![],
        sensitive_functions: vec![],
    }
}

fn plant(pumps: usize, supervised: bool) -> Model {
    let mut components: Vec<Component> = (0..pumps).map(pump).collect();
    if supervised {
        components.push(supervisor());
    }
    Model {
        programs: vec![],
        name: "schedule_index".into(),
        components,
        connections: vec![],
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
        unbounded_rate: None,
        fmu_units: vec![],
        interface_connections: vec![],
    }
}

fn run(model: &Model, t_max: f64, seed: u64) -> SimulationResult {
    let compiled = CompiledModel::compile(model).expect("the model compiles");
    let config = EngineConfig {
        t_max,
        seed,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config)
        .expect("the engine builds")
        .run()
        .expect("the run completes")
}

/// FNV-1a over the exact event trace: dates by their bits, names and
/// states as text. Any reordering, any draw consumed differently, any
/// date off by one ulp moves it.
fn trace_digest(result: &SimulationResult) -> (usize, u64) {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for event in &result.events {
        feed(&event.time.to_bits().to_le_bytes());
        feed(event.transition.as_bytes());
        feed(event.from.as_bytes());
        feed(event.to.as_bytes());
    }
    (result.events.len(), hash)
}

/// **Unchanged trajectories.** Pinned on the full walk, three seeds: the
/// standby starts and is cancelled, the wear-out is rescheduled when P1
/// changes, the alarm pauses and resumes with P2.
#[test]
fn the_indexed_refresh_reproduces_the_full_walk_bit_for_bit() {
    let model = plant(6, true);
    let pinned: [(u64, (usize, u64)); 3] = [
        (1, (72, 3_868_865_255_109_736_546)),
        (2, (75, 4_203_183_704_242_527_069)),
        (3, (78, 3_726_270_806_483_904_346)),
    ];
    for (seed, expected) in pinned {
        assert_eq!(
            trace_digest(&run(&model, 400.0, seed)),
            expected,
            "seed {seed}"
        );
    }
}

/// A clock-read guard over a deterministic pump: `late` holds from t = 5,
/// but is read only when a step refreshes the schedule, and the pump's
/// first step is at t = 10.
fn clock_watcher() -> Model {
    let mut pump = pump(0);
    pump.automata[0].transitions = vec![
        transition("fail", "ok", "ko", None, Distrib::Delay { time: 10.0 }),
        transition("repair", "ko", "ok", None, Distrib::Delay { time: 10.0 }),
    ];
    let watcher = Component {
        name: "W".into(),
        attributes: vec![],
        ports: vec![],
        interfaces: vec![],
        automata: vec![Automaton {
            name: "late".into(),
            states: vec!["off".into(), "on".into()],
            init: "off".into(),
            transitions: vec![transition(
                "late",
                "off",
                "on",
                Some(Expr::Cmp {
                    cmp: CmpOp::Gt,
                    lhs: Box::new(Expr::Time),
                    rhs: Box::new(Expr::Const {
                        value: Value::Float(5.0),
                    }),
                }),
                Distrib::Delay { time: 1.0 },
            )],
        }],
        allocations: vec![],
        equations: vec![],
        sensitive_functions: vec![],
    };
    let mut model = plant(0, false);
    model.components = vec![pump, watcher];
    model
}

fn continue_to_end(engine: &mut Engine) -> Vec<(u64, String)> {
    while engine.step().expect("the step completes").is_some() {}
    engine
        .history()
        .iter()
        .map(|event| (event.time.to_bits(), event.transition.clone()))
        .collect()
}

/// **A restore refreshes nothing.** The clock passed the guard's
/// threshold with no step in between; the persistent engine arms `late`
/// at its next step (t = 10). A restored engine must do the same, and not
/// read the guard at the restore instant.
#[test]
fn a_restored_engine_continues_like_the_persistent_one() {
    let model = clock_watcher();
    let compiled = CompiledModel::compile(&model).expect("the model compiles");
    let config = EngineConfig {
        t_max: 15.0,
        seed: 1,
        ..EngineConfig::default()
    };
    let mut persistent = Engine::new(&compiled, config.clone()).expect("the engine builds");
    persistent.advance_to(7.0).expect("the clock advances");
    assert!(persistent.history().is_empty(), "no step before t = 10");
    let snapshot = persistent.snapshot().expect("the snapshot is taken");
    let expected = continue_to_end(&mut persistent);
    assert!(
        expected.contains(&(11.0f64.to_bits(), "W.late.late".to_owned())),
        "{expected:?}"
    );

    let mut restored = Engine::new(&compiled, config.clone()).expect("the engine builds");
    restored.restore(&snapshot).expect("the snapshot restores");
    assert_eq!(continue_to_end(&mut restored), expected);

    let mut rebuilt =
        Engine::from_snapshot(&compiled, config, &snapshot).expect("the engine rebuilds");
    assert_eq!(continue_to_end(&mut rebuilt), expected);
}
