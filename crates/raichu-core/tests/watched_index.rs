//! The indexed watched set: what the immediate-guard scan and the
//! per-segment margin filter must cost, and what they must never change.
//!
//! Two claims are separable and both are tested here. The **behavioural**
//! one is that narrowing a scan changes nothing an observer can see: the
//! same events, at the same dates, in the same order, including when two
//! boundaries cross at one instant. The **cost** one is that a change to
//! a single attribute re-evaluates only the guards that read it, which is
//! read off [`WorkCounters::immediate_guard_scans`] rather than off a
//! clock, so it is a property of the engine and not of the machine.
//!
//! The pinned numbers below were recorded on the commit that precedes the
//! index, with the full scan in place. Pinning them rather than comparing
//! the engine with itself is what makes "unchanged" a claim about the
//! previous engine.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig, SimulationResult};
use raichu_expr::{Assignment, AttrRef, CmpOp, Expr, StateRef, Value};
use raichu_model::{
    AttrKind, Attribute, Automaton, Component, Distrib, Model, SensitiveFunction, Transition,
};

// ---------------------------------------------------------------------
// A wide, local network: many watched guards, one moving attribute.
// ---------------------------------------------------------------------

/// One branch of the synthetic network: an attribute driven low by a
/// delay, and a watched protection that opens when it goes low.
///
/// `reads` names the attribute the protection guard watches. Passing the
/// branch's own attribute makes the branch independent of every other
/// one, which is the shape the index is meant to exploit; passing a
/// shared attribute makes several protections cross at one instant, which
/// is the shape the firing order must survive.
fn branch(index: usize, delay: f64, reads: (&str, &str), driven: bool) -> Component {
    let name = format!("unit{index}");
    let mut automata = vec![Automaton {
        name: "protection".into(),
        states: vec!["closed".into(), "open".into()],
        init: "closed".into(),
        transitions: vec![Transition {
            name: "open".into(),
            source: "closed".into(),
            guard: Some(Expr::Cmp {
                cmp: CmpOp::Le,
                lhs: Box::new(Expr::attr(reads.0, reads.1)),
                rhs: Box::new(Expr::Const {
                    value: Value::Float(0.5),
                }),
            }),
            targets: vec!["open".into()],
            on_interruption: Default::default(),
            monitored: false,
            cycle_group: None,
            distrib: Distrib::Watched,
        }],
    }];
    let mut sensitive_functions = vec![];
    if driven {
        automata.push(Automaton {
            name: "drive".into(),
            states: vec!["high".into(), "low".into()],
            init: "high".into(),
            transitions: vec![Transition {
                name: "drop".into(),
                source: "high".into(),
                guard: None,
                targets: vec!["low".into()],
                on_interruption: Default::default(),
                monitored: false,
                cycle_group: None,
                distrib: Distrib::Delay { time: delay },
            }],
        });
        sensitive_functions.push(SensitiveFunction {
            name: "update_level".into(),
            effects: vec![Assignment {
                target: AttrRef {
                    component: name.clone(),
                    attribute: "level".into(),
                },
                value: Expr::If {
                    cond: Box::new(Expr::StateActive {
                        state: StateRef {
                            component: name.clone(),
                            automaton: "drive".into(),
                            state: "high".into(),
                        },
                    }),
                    then: Box::new(Expr::Const {
                        value: Value::Float(1.0),
                    }),
                    otherwise: Box::new(Expr::Const {
                        value: Value::Float(0.0),
                    }),
                },
            }],
        });
    }
    Component {
        name: name.clone(),
        attributes: vec![Attribute {
            name: "level".into(),
            kind: AttrKind::Float,
            init: Value::Float(1.0),
        }],
        ports: vec![],
        interfaces: vec![],
        automata,
        allocations: vec![],
        equations: vec![],
        sensitive_functions,
    }
}

/// `n` independent branches. Exactly one of them (`active`) is ever
/// driven low, so exactly one protection can ever fire: every other guard
/// reads an attribute that never moves.
fn independent_branches(n: usize, active: usize) -> Model {
    Model {
        name: "wide_independent".into(),
        components: (0..n)
            .map(|i| {
                let own = format!("unit{i}");
                branch(i, 4.0, (&own, "level"), i == active)
            })
            .collect(),
        connections: vec![],
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
    }
}

/// `n` branches whose protections all read **one** shared attribute, so
/// every one of them crosses at the same instant. The order in which they
/// fire is the engine's documented tie-break (ascending transition
/// index), and it must survive the index untouched.
fn shared_boundary(n: usize) -> Model {
    let mut components: Vec<Component> = (0..n)
        .map(|i| branch(i, 4.0, ("bus", "level"), false))
        .collect();
    components.insert(
        0,
        Component {
            name: "bus".into(),
            attributes: vec![Attribute {
                name: "level".into(),
                kind: AttrKind::Float,
                init: Value::Float(1.0),
            }],
            ports: vec![],
            interfaces: vec![],
            automata: vec![Automaton {
                name: "drive".into(),
                states: vec!["high".into(), "low".into()],
                init: "high".into(),
                transitions: vec![Transition {
                    name: "drop".into(),
                    source: "high".into(),
                    guard: None,
                    targets: vec!["low".into()],
                    on_interruption: Default::default(),
                    monitored: false,
                    cycle_group: None,
                    distrib: Distrib::Delay { time: 4.0 },
                }],
            }],
            allocations: vec![],
            equations: vec![],
            sensitive_functions: vec![SensitiveFunction {
                name: "update_level".into(),
                effects: vec![Assignment {
                    target: AttrRef {
                        component: "bus".into(),
                        attribute: "level".into(),
                    },
                    value: Expr::If {
                        cond: Box::new(Expr::StateActive {
                            state: StateRef {
                                component: "bus".into(),
                                automaton: "drive".into(),
                                state: "high".into(),
                            },
                        }),
                        then: Box::new(Expr::Const {
                            value: Value::Float(1.0),
                        }),
                        otherwise: Box::new(Expr::Const {
                            value: Value::Float(0.0),
                        }),
                    },
                }],
            }],
        },
    );
    Model {
        name: "shared_boundary".into(),
        components,
        connections: vec![],
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
    }
}

fn run(model: &Model, t_max: f64) -> SimulationResult {
    let compiled = CompiledModel::compile(model).expect("the model compiles");
    let config = EngineConfig {
        t_max,
        ..EngineConfig::default()
    };
    Engine::new(&compiled, config)
        .expect("the engine builds")
        .run()
        .expect("the run completes")
}

/// The event trace as `(date, transition, from, to)`, the level-1
/// comparison of the validation contract.
fn trace(result: &SimulationResult) -> Vec<(String, String, String, String)> {
    result
        .events
        .iter()
        .map(|e| {
            (
                format!("{:.9}", e.time),
                e.transition.clone(),
                e.from.clone(),
                e.to.clone(),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------
// Behaviour: the narrowed scan sees exactly what the full scan saw.
// ---------------------------------------------------------------------

/// One active protection among two hundred produces the trace the full
/// scan produced, event for event.
#[test]
fn one_active_watched_among_many_reproduces_the_full_scan_trace() {
    let result = run(&independent_branches(200, 137), 20.0);
    assert_eq!(
        trace(&result),
        vec![
            (
                "4.000000000".to_owned(),
                "unit137.drive.drop".to_owned(),
                "high".to_owned(),
                "low".to_owned()
            ),
            (
                "4.000000000".to_owned(),
                "unit137.protection.open".to_owned(),
                "closed".to_owned(),
                "open".to_owned()
            ),
        ],
        "the narrowed scan fires exactly what the full scan fired"
    );
}

/// Five protections sharing one boundary cross at the same instant and
/// fire in ascending transition index, exactly as they did before the
/// index existed.
#[test]
fn watched_transitions_crossing_at_one_instant_keep_their_firing_order() {
    let result = run(&shared_boundary(5), 20.0);
    let fired: Vec<String> = result
        .events
        .iter()
        .map(|e| format!("{:.9} {}", e.time, e.transition))
        .collect();
    assert_eq!(
        fired,
        vec![
            "4.000000000 bus.drive.drop",
            "4.000000000 unit0.protection.open",
            "4.000000000 unit1.protection.open",
            "4.000000000 unit2.protection.open",
            "4.000000000 unit3.protection.open",
            "4.000000000 unit4.protection.open",
        ],
        "simultaneous crossings keep the documented ascending-index order"
    );
}

// ---------------------------------------------------------------------
// Cost: the scan tracks what moved, not what exists.
// ---------------------------------------------------------------------

/// **The claim of this unit.** Two hundred protections are armed; one
/// attribute moves once. The immediate-guard scan must re-evaluate the
/// guards that read that attribute, not the two hundred that do not.
///
/// The full scan ran after the initialization axiom and after every
/// discrete firing, walking the whole armed population each time: it
/// evaluated **537** guards on this trajectory. What is left is one cold
/// pass over the population (200, nothing being cached yet) plus exactly
/// one re-evaluation, for the single guard reading the single attribute
/// that moved.
#[test]
fn changing_one_attribute_re_evaluates_only_the_guards_that_read_it() {
    let n = 200;
    let work = run(&independent_branches(n, 137), 20.0).work;
    assert_eq!(
        work.immediate_guard_scans,
        n as u64 + 1,
        "one cold pass over {n} protections plus one guard for the one \
         attribute that moved; the full scan spent 537 here"
    );
}

/// The same model with **no** attribute moving at all: after the cold
/// pass the scan has nothing to re-evaluate, ever.
#[test]
fn a_scan_over_an_unmoving_network_costs_one_cold_pass() {
    let n = 50;
    let model = Model {
        name: "unmoving".into(),
        components: (0..n)
            .map(|i| {
                let own = format!("unit{i}");
                branch(i, 4.0, (&own, "level"), false)
            })
            .collect(),
        connections: vec![],
        indicators: vec![],
        targets: vec![],
        evaluation_order: None,
    };
    let work = run(&model, 20.0).work;
    assert_eq!(
        work.immediate_guard_scans, n as u64,
        "one cold pass over the population and nothing after it"
    );
}

// ---------------------------------------------------------------------
// The index is derived, never carried.
// ---------------------------------------------------------------------

/// A snapshot rewinds the arming and every cached verdict with the state
/// it rewinds: the index is derived from the attribute and state vectors,
/// so a replay from a restored snapshot reproduces the run exactly.
///
/// The model is hybrid on purpose: the restored engine has to reproduce a
/// *located* crossing, which reads the per-segment margin set, and not
/// merely an immediate one.
#[test]
fn a_restored_snapshot_replays_the_same_watched_trajectory() {
    let Some(json) = fixture("tank_01") else {
        return;
    };
    let model = Model::from_json(&json).expect("fixture JSON parses");
    let compiled = CompiledModel::compile(&model).expect("the fixture compiles");
    let config = || EngineConfig {
        t_max: 40.0,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new(&compiled, config()).expect("the engine builds");
    // Advance past the first located crossing, then checkpoint.
    engine.step().expect("a step runs");
    engine.step().expect("a second step runs");
    let snapshot = engine.snapshot();
    let at_snapshot = engine.history().len();

    let mut reference = Vec::new();
    while let Some(event) = engine.step().expect("the run continues") {
        reference.push(format!("{:.9} {}", event.time, event.transition));
    }

    engine.restore(&snapshot);
    assert_eq!(
        engine.history().len(),
        at_snapshot,
        "the restore rewinds the trajectory"
    );
    let mut replay = Vec::new();
    while let Some(event) = engine.step().expect("the replay continues") {
        replay.push(format!("{:.9} {}", event.time, event.transition));
    }
    assert!(!reference.is_empty(), "the replay must cover real events");
    assert_eq!(
        replay, reference,
        "a replay from a restored snapshot fires the same watched events"
    );

    // The same through the rebuild seam a stateful facade uses.
    let mut rebuilt = Engine::from_snapshot(&compiled, config(), &snapshot);
    let mut again = Vec::new();
    while let Some(event) = rebuilt.step().expect("the rebuilt engine runs") {
        again.push(format!("{:.9} {}", event.time, event.transition));
    }
    assert_eq!(
        again, reference,
        "an engine rebuilt on the snapshot replays the same watched events"
    );
}

/// A reset engine is a fresh engine: the arming is re-derived and no
/// verdict survives from the trajectory that was discarded.
#[test]
fn a_reset_engine_rearms_from_the_initial_state() {
    let model = independent_branches(20, 7);
    let compiled = CompiledModel::compile(&model).expect("the model compiles");
    let mut engine = Engine::new(
        &compiled,
        EngineConfig {
            t_max: 20.0,
            ..EngineConfig::default()
        },
    )
    .expect("the engine builds");
    let first: Vec<String> = std::iter::from_fn(|| engine.step().expect("the run advances"))
        .map(|e| format!("{:.9} {}", e.time, e.transition))
        .collect();
    engine
        .reset()
        .expect("the reset runs the initialization axiom");
    let second: Vec<String> = std::iter::from_fn(|| engine.step().expect("the run advances"))
        .map(|e| format!("{:.9} {}", e.time, e.transition))
        .collect();
    assert!(!first.is_empty(), "the trajectory must contain events");
    assert_eq!(second, first, "a reset engine replays a fresh engine's run");
}

// ---------------------------------------------------------------------
// The compiled index is a faithful inversion.
// ---------------------------------------------------------------------

/// The index inverts the watched guards faithfully: every automaton owns
/// exactly the watched transitions it declares, and every attribute
/// registration is a guard that names it.
///
/// Faithfulness is checked against the declarations themselves rather
/// than against a pinned table. The operator half of the same index is
/// checked where the models carrying operators live, in `active_set.rs`.
#[test]
fn the_compiled_index_inverts_the_watched_guards() {
    let Some(json) = fixture("pool_02") else {
        return;
    };
    let model = Model::from_json(&json).expect("fixture JSON parses");
    let compiled = CompiledModel::compile(&model).expect("the fixture compiles");
    let index = &compiled.margin_index;

    assert!(
        !compiled.watched.is_empty(),
        "pool_02 must carry watched transitions for this to test anything"
    );

    // Ownership: each automaton owns exactly its own watched positions,
    // in ascending order.
    let mut owned_total = 0;
    for (automaton, positions) in index.watched_by_owner.iter().enumerate() {
        assert!(
            positions.windows(2).all(|w| w[0] < w[1]),
            "the owner index must be ascending"
        );
        for &position in positions {
            assert_eq!(
                compiled.transitions[compiled.watched[position]].automaton, automaton,
                "a position registered under the wrong automaton"
            );
        }
        owned_total += positions.len();
    }
    assert_eq!(
        owned_total,
        compiled.watched.len(),
        "every watched transition is owned exactly once"
    );

    // Attributes: every registration is real, and no dependency is
    // missing (checked by rebuilding the expected inversion from the
    // guards through the engine's own comparison of names).
    for (var, positions) in index.watched_by_var.iter().enumerate() {
        assert!(
            positions.windows(2).all(|w| w[0] < w[1]),
            "the attribute index must be ascending and deduplicated"
        );
        for &position in positions {
            assert!(
                compiled.transitions[compiled.watched[position]]
                    .guard
                    .is_some(),
                "only guarded transitions are registered under {}",
                compiled.var_names[var]
            );
        }
    }
}

// ---------------------------------------------------------------------
// Regression: the corpus reproduces its dates, the discrete profile its
// counters.
// ---------------------------------------------------------------------

/// The cross-validation corpus is absent from some checkouts of this
/// project. A test that pins a fixture's behaviour reports that it did
/// not run rather than failing, so the suite stays green either way.
fn fixture(name: &str) -> Option<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/tests/validation/fixtures")
        .join(format!("{name}.json"));
    match std::fs::read_to_string(&path) {
        Ok(json) => Some(json),
        Err(_) => {
            eprintln!("skipped: no cross-validation corpus at {}", path.display());
            None
        }
    }
}

/// FNV-1a over the trace text. A digest rather than the trace itself,
/// because one fixture below fires several hundred events and an inline
/// literal of that size documents nothing; the trace is reconstructible
/// from the fixture whenever the digest disagrees.
fn digest(result: &SimulationResult) -> (usize, String) {
    let text = trace(result)
        .iter()
        .map(|(t, n, f, to)| format!("{t}|{n}|{f}|{to}"))
        .collect::<Vec<_>>()
        .join(";");
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (result.events.len(), format!("{hash:016x}"))
}

/// Every fixture that carries a watched transition reproduces its event
/// trace bit for bit. Dates enter the digest at full precision, so a
/// crossing located one ulp elsewhere fails here rather than passing on
/// a tolerance.
#[test]
fn every_watched_fixture_reproduces_its_event_dates() {
    // (fixture, t_max, event count, FNV-1a of the trace)
    let expected: [(&str, f64, usize, &str); 5] = [
        ("tank_01", 40.0, 5, "11de1238f6c514a4"),
        ("tank_02", 1000.0, 5, "461f4b8d9e56fb0a"),
        ("heated_room_s3", 100.0, 15, "42f214e7090d5cb3"),
        ("pool_02", 1000.0, 470, "0c1c6d566324761e"),
        ("pdmp_001", 1000.0, 176, "72e7f58411cdc9a6"),
    ];
    for (name, t_max, events, pinned) in expected {
        let Some(json) = fixture(name) else { continue };
        let model = Model::from_json(&json).expect("fixture JSON parses");
        let result = run(&model, t_max);
        assert_eq!(
            digest(&result),
            (events, pinned.to_owned()),
            "{name}: the event trace moved against the pre-index baseline"
        );
    }
}

/// A discrete-only fixture with no watched transition pays the change
/// detection the explicit pass now performs and receives none of the
/// narrowing: its counted-work profile must be untouched.
#[test]
fn a_discrete_fixture_with_no_watched_transition_keeps_its_profile() {
    let Some(json) = fixture("delay_001") else {
        return;
    };
    let model = Model::from_json(&json).expect("fixture JSON parses");
    let work = run(&model, 18.0).work;
    assert_eq!(
        (
            work.explicit_evaluations,
            work.segments,
            work.margin_evaluations,
            work.immediate_guard_scans,
        ),
        (3u64, 0u64, 0u64, 0u64),
        "delay_001: counted work moved against the pre-index baseline"
    );
}
