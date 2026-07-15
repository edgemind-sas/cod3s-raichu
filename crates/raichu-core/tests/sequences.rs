//! Sequence-analysis recording (F3a-A1): the engine records, per
//! trajectory, the ordered `SeqEvent`s of fired *monitored* transitions
//! and the end cause when a target (feared event) is reached, then
//! early-stops at that target.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{CompiledModel, Engine, EngineConfig};
use raichu_model::{Automaton, Component, Distrib, Model, Target, Transition};

fn mon(name: &str, source: &str, target: &str, time: f64, group: Option<&str>) -> Transition {
    Transition {
        name: name.into(),
        source: source.into(),
        guard: None,
        targets: vec![target.into()],
        on_interruption: Default::default(),
        monitored: true,
        cycle_group: group.map(Into::into),
        distrib: Distrib::Delay { time },
    }
}

fn comp(
    name: &str,
    aut: &str,
    init: &str,
    states: &[&str],
    transitions: Vec<Transition>,
) -> Component {
    Component {
        name: name.into(),
        attributes: vec![],
        ports: vec![],
        interfaces: vec![],
        automata: vec![Automaton {
            name: aut.into(),
            states: states.iter().map(|s| (*s).into()).collect(),
            init: init.into(),
            transitions,
        }],
        equations: vec![],
        sensitive_functions: vec![],
    }
}

/// `A.life` cycles occ@5 / rep@8 / occ@13… (both transitions monitored, a
/// cycle pair). `ER.ev` fires occ@10, which is the feared-event target.
fn model() -> Model {
    Model {
        name: "seq_a1".into(),
        components: vec![
            comp(
                "A",
                "life",
                "rep",
                &["rep", "occ"],
                vec![
                    mon("occ", "rep", "occ", 5.0, Some("A")),
                    mon("rep", "occ", "rep", 3.0, Some("A")),
                ],
            ),
            comp(
                "ER",
                "ev",
                "not_occ",
                &["not_occ", "occ"],
                vec![mon("occ", "not_occ", "occ", 10.0, None)],
            ),
        ],
        connections: vec![],
        indicators: vec![],
        targets: vec![Target {
            name: "feared".into(),
            component: "ER".into(),
            automaton: "ev".into(),
            state: "occ".into(),
        }],
    }
}

fn compiled() -> CompiledModel {
    CompiledModel::compile(&model()).unwrap()
}

#[test]
fn records_monitored_events_and_stops_at_the_target() {
    let m = compiled();
    let config = EngineConfig {
        t_max: 100.0,
        sequences: true,
        ..EngineConfig::default()
    };
    let result = Engine::new(&m, config).unwrap().run().unwrap();
    let seq = result.sequence.expect("sequence recorded when enabled");

    // A.life fires occ@5 and rep@8 (a transient cycle pair); ER.ev reaches
    // the feared event occ@10, which ends the trajectory.
    let events: Vec<_> = seq
        .events
        .iter()
        .map(|e| (e.obj.as_str(), e.attr.as_str(), e.time))
        .collect();
    assert_eq!(
        events,
        vec![("A", "occ", 5.0), ("A", "rep", 8.0), ("ER", "occ", 10.0)]
    );
    assert_eq!(seq.end_cause.as_deref(), Some("feared"));
    assert_eq!(seq.end_time, 10.0);
    assert_eq!(seq.weight, 1.0);
    // Early-stop: the run halts at the target, not at t_max.
    assert_eq!(result.final_time, 10.0);
}

#[test]
fn no_sequence_when_recording_disabled() {
    let m = compiled();
    // Default config: sequences off → zero cost, no sequence, no early stop.
    let result = Engine::new(
        &m,
        EngineConfig {
            t_max: 100.0,
            ..EngineConfig::default()
        },
    )
    .unwrap()
    .run()
    .unwrap();
    assert!(result.sequence.is_none());
    assert_eq!(result.final_time, 100.0);
}

#[test]
fn early_stop_finishes_the_hit_instant() {
    // Review finding: transitions due at the SAME instant as the target
    // activation must still fire before the stop, so the latched state is
    // the completed instant (PyCATSHOO finishes the step before stopping).
    // `T.ev` (the target) and `B.aux` are both due at t=5; T fires first
    // (declaration order) — B must fire too.
    let m = Model {
        name: "same_instant".into(),
        components: vec![
            comp(
                "T",
                "ev",
                "idle",
                &["idle", "occ"],
                vec![mon("occ", "idle", "occ", 5.0, None)],
            ),
            comp(
                "B",
                "aux",
                "off",
                &["off", "on"],
                vec![mon("on", "off", "on", 5.0, None)],
            ),
        ],
        connections: vec![],
        indicators: vec![],
        targets: vec![Target {
            name: "feared".into(),
            component: "T".into(),
            automaton: "ev".into(),
            state: "occ".into(),
        }],
    };
    let compiled = CompiledModel::compile(&m).unwrap();
    let config = EngineConfig {
        t_max: 100.0,
        sequences: true,
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    let seq = result.sequence.unwrap();
    // Both same-instant transitions are recorded; the stop happens after
    // the instant is complete.
    let events: Vec<_> = seq
        .events
        .iter()
        .map(|e| (e.obj.as_str(), e.attr.as_str(), e.time))
        .collect();
    assert_eq!(events, vec![("T", "occ", 5.0), ("B", "on", 5.0)]);
    assert_eq!(seq.end_cause.as_deref(), Some("feared"));
    assert_eq!(result.final_time, 5.0);
}

#[test]
fn initially_active_target_ends_the_trajectory_at_zero() {
    // Review finding: a target naming an automaton's declared INIT state
    // must be detected at initialization (end_cause at t = 0), not run to
    // the horizon uncaused.
    let mut m = model();
    // Make the feared state ER.ev the initial one.
    m.components[1].automata[0].init = "occ".into();
    let compiled = CompiledModel::compile(&m).unwrap();
    let config = EngineConfig {
        t_max: 100.0,
        sequences: true,
        ..EngineConfig::default()
    };
    let result = Engine::new(&compiled, config).unwrap().run().unwrap();
    let seq = result.sequence.unwrap();
    assert_eq!(seq.end_cause.as_deref(), Some("feared"));
    assert_eq!(seq.end_time, 0.0);
    assert_eq!(result.final_time, 0.0);
}

#[test]
fn runs_to_horizon_when_no_target_is_reached() {
    let m = compiled();
    // Horizon 7: A fires occ@5 but its rep@8 and ER.ev's occ@10 are past
    // the horizon, so no target is reached — the sequence records only the
    // one event and ends with no cause at t_max.
    let config = EngineConfig {
        t_max: 7.0,
        sequences: true,
        ..EngineConfig::default()
    };
    let result = Engine::new(&m, config).unwrap().run().unwrap();
    let seq = result.sequence.unwrap();
    let events: Vec<_> = seq
        .events
        .iter()
        .map(|e| (e.obj.as_str(), e.attr.as_str()))
        .collect();
    assert_eq!(events, vec![("A", "occ")]); // occ@5 only
    assert_eq!(seq.end_cause, None);
    assert_eq!(seq.end_time, 7.0);
}
