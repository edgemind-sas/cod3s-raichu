//! Observed sequence campaigns: every trajectory also records the value of
//! chosen indicators at chosen instants, the raw corpus carries them, and a
//! condition on them selects trajectories before the reduction.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{
    read_raw_corpus, read_raw_sequences, write_raw_corpus, write_raw_sequences, CompiledModel,
    ObservedCondition, RawHeader, RawObservation, Sequence, SolverParams,
};
use raichu_expr::CmpOp;
use raichu_model::Model;
use raichu_montecarlo::{run_sequences, run_sequences_observed, McConfig, SequenceObservation};

/// Two repairable components and a feared event reached the instant both
/// are down, plus an indicator on whether `A` is down: the value it reads
/// at an instant is decidable from the recorded events alone, which is
/// what the tests check it against.
const PAIR: &str = r#"
{
  "name": "pair",
  "components": [
    {"name": "A", "ports": [], "attributes": [], "automata": [{"name": "life", "states": ["ok", "ko"], "init": "ok",
      "transitions": [
        {"name": "fail", "source": "ok", "targets": ["ko"], "distrib": "exp", "rate": 0.3, "monitored": true, "cycle_group": "life"},
        {"name": "fix", "source": "ko", "targets": ["ok"], "distrib": "exp", "rate": 1.0, "monitored": true, "cycle_group": "life"}]}]},
    {"name": "B", "ports": [], "attributes": [], "automata": [{"name": "life", "states": ["ok", "ko"], "init": "ok",
      "transitions": [
        {"name": "fail", "source": "ok", "targets": ["ko"], "distrib": "exp", "rate": 0.2, "monitored": true, "cycle_group": "life"},
        {"name": "fix", "source": "ko", "targets": ["ok"], "distrib": "exp", "rate": 1.0, "monitored": true, "cycle_group": "life"}]}]},
    {"name": "ER", "ports": [], "attributes": [], "automata": [{"name": "ev", "states": ["not_occ", "occ"], "init": "not_occ",
      "transitions": [
        {"name": "occ", "source": "not_occ", "targets": ["occ"], "distrib": "delay", "time": 0.0, "monitored": true,
         "guard": {"op": "bool", "bool_op": "and", "args": [
           {"op": "state_active", "state": {"component": "A", "automaton": "life", "state": "ko"}},
           {"op": "state_active", "state": {"component": "B", "automaton": "life", "state": "ko"}}]}}]}]}
  ],
  "targets": [{"name": "both_down", "component": "ER", "automaton": "ev", "state": "occ"}],
  "indicators": [
    {"name": "A_down", "target": "state", "component": "A", "automaton": "life", "state": "ko"}
  ]
}
"#;

const NB_RUNS: u64 = 400;
const T_MAX: f64 = 20.0;

fn compiled() -> CompiledModel {
    CompiledModel::compile(&Model::from_json(PAIR).expect("model loads")).expect("compiles")
}

fn config(threads: Option<usize>) -> McConfig {
    McConfig {
        nb_runs: NB_RUNS,
        seed: 7,
        t_max: T_MAX,
        samples: Vec::new(),
        threads,
        quantiles: Vec::new(),
        ode: SolverParams::default(),
        stop_at_targets: false,
        flow: Default::default(),
    }
}

fn observe(time: f64) -> Vec<SequenceObservation> {
    vec![SequenceObservation {
        indicator: "A_down".into(),
        time,
    }]
}

/// Whether `A` is down at `time` in `sequence`, from its events alone: the
/// state its last event at or before `time` entered, the trajectory's final
/// state past its end.
fn down_at(sequence: &Sequence, time: f64) -> bool {
    sequence
        .events
        .iter()
        .rfind(|e| e.obj == "A" && e.time <= time.min(sequence.end_time))
        .is_some_and(|e| e.attr == "ko")
}

#[test]
fn observing_does_not_change_the_trajectories() {
    let plain = run_sequences(&compiled(), &config(None)).unwrap();
    let observed = run_sequences_observed(&compiled(), &config(None), &observe(5.0)).unwrap();
    assert_eq!(observed.sequences, plain);
}

#[test]
fn the_observed_value_is_the_state_at_the_instant() {
    for time in [0.0, 2.5, 5.0, 12.0, T_MAX] {
        let observed = run_sequences_observed(&compiled(), &config(None), &observe(time)).unwrap();
        let mut down = 0;
        for (sequence, row) in observed.sequences.iter().zip(&observed.observed) {
            let expected = down_at(sequence, time);
            assert_eq!(row, &vec![f64::from(u8::from(expected))], "at {time}");
            down += usize::from(expected);
        }
        // The check is not vacuous: past the start some trajectories are
        // down at the instant and some are not.
        if time > 0.0 {
            assert!(
                down > 0 && down < NB_RUNS as usize,
                "at {time}: {down} down"
            );
        }
    }
}

#[test]
fn an_instant_past_the_horizon_reads_the_last_state() {
    let at_horizon = run_sequences_observed(&compiled(), &config(None), &observe(T_MAX)).unwrap();
    let beyond = run_sequences_observed(&compiled(), &config(None), &observe(1.0e6)).unwrap();
    assert_eq!(beyond.observed, at_horizon.observed);
    assert_eq!(beyond.times, vec![T_MAX]);
}

#[test]
fn the_thread_count_does_not_change_the_observations() {
    let serial = run_sequences_observed(&compiled(), &config(Some(1)), &observe(8.0)).unwrap();
    let parallel = run_sequences_observed(&compiled(), &config(Some(4)), &observe(8.0)).unwrap();
    assert_eq!(serial, parallel);
}

#[test]
fn an_observation_of_nothing_is_refused() {
    let unknown = vec![SequenceObservation {
        indicator: "nope".into(),
        time: 1.0,
    }];
    let err = run_sequences_observed(&compiled(), &config(None), &unknown).unwrap_err();
    assert!(err.to_string().contains("`nope`"), "{err}");
    for time in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(run_sequences_observed(&compiled(), &config(None), &observe(time)).is_err());
    }
}

fn observed_header() -> RawHeader {
    RawHeader::new("test", "pair", 7, NB_RUNS, T_MAX, vec!["both_down".into()]).with_observations(
        vec![RawObservation {
            name: "A_down".into(),
            time: 5.0,
        }],
    )
}

#[test]
fn the_corpus_carries_the_observations_and_filters_on_them() {
    let campaign = run_sequences_observed(&compiled(), &config(None), &observe(5.0)).unwrap();
    let mut bytes = Vec::new();
    write_raw_corpus(
        &mut bytes,
        &observed_header(),
        &campaign.sequences,
        &campaign.observed,
    )
    .unwrap();

    let corpus = read_raw_corpus(bytes.as_slice()).unwrap();
    assert_eq!(corpus.header, observed_header());
    assert_eq!(corpus.sequences, campaign.sequences);
    assert_eq!(corpus.observed, campaign.observed);

    // A reader that does not know observations still reads the corpus.
    let (_, sequences) = read_raw_sequences(bytes.as_slice()).unwrap();
    assert_eq!(sequences, campaign.sequences);

    let down = ObservedCondition {
        observation: "A_down".into(),
        op: CmpOp::Eq,
        value: 1.0,
    };
    let kept = corpus.filtered(&down).unwrap();
    let expected: Vec<Sequence> = campaign
        .sequences
        .iter()
        .filter(|s| down_at(s, 5.0))
        .cloned()
        .collect();
    assert_eq!(kept, expected);
    assert!(!kept.is_empty() && kept.len() < NB_RUNS as usize);

    let up = ObservedCondition {
        op: CmpOp::Lt,
        value: 0.5,
        ..down
    };
    assert_eq!(
        corpus.filtered(&up).unwrap().len() + kept.len(),
        NB_RUNS as usize
    );
}

#[test]
fn a_condition_on_what_was_not_observed_is_refused() {
    let campaign = run_sequences_observed(&compiled(), &config(None), &observe(5.0)).unwrap();
    let mut bytes = Vec::new();
    write_raw_corpus(
        &mut bytes,
        &observed_header(),
        &campaign.sequences,
        &campaign.observed,
    )
    .unwrap();
    let corpus = read_raw_corpus(bytes.as_slice()).unwrap();
    let err = corpus
        .filtered(&ObservedCondition {
            observation: "B_down".into(),
            op: CmpOp::Eq,
            value: 1.0,
        })
        .unwrap_err();
    assert!(err.to_string().contains("`B_down`"), "{err}");
}

#[test]
fn a_corpus_without_observations_is_written_as_before() {
    let raw = run_sequences(&compiled(), &config(None)).unwrap();
    let header = RawHeader::new("test", "pair", 7, NB_RUNS, T_MAX, vec!["both_down".into()]);
    let mut plain = Vec::new();
    write_raw_sequences(&mut plain, &header, &raw).unwrap();
    let text = String::from_utf8(plain).unwrap();
    assert!(!text.contains("observ"), "no observation field leaks in");
}

#[test]
fn observed_rows_must_match_the_header() {
    let campaign = run_sequences_observed(&compiled(), &config(None), &observe(5.0)).unwrap();
    let mut short = campaign.observed.clone();
    short[3].clear();
    let err =
        write_raw_corpus(Vec::new(), &observed_header(), &campaign.sequences, &short).unwrap_err();
    assert!(err.to_string().contains("run 3"), "{err}");

    let plain = RawHeader::new("test", "pair", 7, NB_RUNS, T_MAX, vec!["both_down".into()]);
    assert!(write_raw_corpus(Vec::new(), &plain, &campaign.sequences, &campaign.observed).is_err());
}
