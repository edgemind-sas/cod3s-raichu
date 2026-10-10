//! Sequence campaigns handed on one trajectory at a time (#133): the
//! replicas run in parallel chunks and reach the caller in replica order,
//! so a campaign of any size reduces and writes its raw corpus without
//! holding its trajectories.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_analysis::{
    clean, read_raw_corpus, write_raw_corpus, RawCorpusReader, RawCorpusWriter, RawHeader,
    RawObservation, RawSequencesError, SequenceReducer,
};
use raichu_core::{CompiledModel, EngineError, Sequence, SolverParams};
use raichu_model::Model;
use raichu_montecarlo::{
    run_sequences_observed, run_sequences_streamed, McConfig, SequenceObservation,
    DEFAULT_CONFIDENCE,
};

/// Two repairable components, a feared event the instant both are down,
/// and an indicator on whether `A` is down.
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

/// More than one chunk of replicas (4 096), so the chunk boundary is crossed.
const NB_RUNS: u64 = 9_000;
const T_MAX: f64 = 5.0;

fn compiled() -> CompiledModel {
    CompiledModel::compile(&Model::from_json(PAIR).expect("model loads")).expect("compiles")
}

fn config(threads: Option<usize>) -> McConfig {
    McConfig {
        nb_runs: NB_RUNS,
        seed: 5,
        t_max: T_MAX,
        samples: Vec::new(),
        threads,
        quantiles: Vec::new(),
        confidence: DEFAULT_CONFIDENCE,
        ode: SolverParams::default(),
        stop_at_targets: false,
        flow: Default::default(),
    }
}

fn observe() -> Vec<SequenceObservation> {
    vec![SequenceObservation {
        indicator: "A_down".into(),
        time: 2.5,
    }]
}

/// The streamed campaign, collected back.
fn streamed(threads: Option<usize>) -> Vec<(u64, Sequence, Vec<f64>)> {
    let mut seen = Vec::new();
    run_sequences_streamed(
        &compiled(),
        &config(threads),
        &observe(),
        |run, seq, row| {
            seen.push((run, seq, row));
            Ok::<_, EngineError>(())
        },
    )
    .unwrap();
    seen
}

#[test]
fn the_streamed_campaign_is_the_held_campaign_in_replica_order() {
    let held = run_sequences_observed(&compiled(), &config(None), &observe()).unwrap();
    let seen = streamed(None);
    assert_eq!(seen.len() as u64, NB_RUNS);
    assert!(seen
        .iter()
        .enumerate()
        .all(|(i, (run, _, _))| *run == i as u64));
    let (sequences, observed): (Vec<_>, Vec<_>) =
        seen.into_iter().map(|(_, seq, row)| (seq, row)).unzip();
    assert_eq!(sequences, held.sequences);
    assert_eq!(observed, held.observed);
}

#[test]
fn the_streamed_campaign_does_not_depend_on_the_thread_count() {
    assert_eq!(streamed(Some(1)), streamed(Some(4)));
}

#[test]
fn reducing_and_writing_on_the_fly_gives_what_the_held_campaign_gives() {
    let held = run_sequences_observed(&compiled(), &config(None), &observe()).unwrap();
    let header = RawHeader::new("test", "pair", 5, NB_RUNS, T_MAX, vec!["both_down".into()])
        .with_observations(vec![RawObservation {
            name: "A_down".into(),
            time: observe()[0].read_at(T_MAX),
        }]);

    let mut reducer = SequenceReducer::new();
    let mut writer = RawCorpusWriter::new(Vec::new(), &header).unwrap();
    let mut most_held = 0;
    run_sequences_streamed(&compiled(), &config(None), &observe(), |_, seq, row| {
        writer.push(&seq, &row)?;
        reducer.push(seq);
        most_held = most_held.max(reducer.len());
        Ok::<_, StreamError>(())
    })
    .unwrap();
    let bytes = writer.finish().unwrap();

    let mut expected = Vec::new();
    write_raw_corpus(&mut expected, &header, &held.sequences, &held.observed).unwrap();
    assert_eq!(bytes, expected, "the same file, byte for byte");
    assert_eq!(
        read_raw_corpus(bytes.as_slice()).unwrap().sequences,
        held.sequences
    );
    assert_eq!(reducer.cleaned(), clean(held.sequences));
    assert!(
        most_held < 100,
        "the reduction held {most_held} distinct paths for {NB_RUNS} trajectories"
    );
}

#[test]
fn an_error_from_the_sink_stops_the_campaign() {
    let mut calls = 0;
    let outcome = run_sequences_streamed(&compiled(), &config(None), &observe(), |run, _, _| {
        calls += 1;
        if run == 10 {
            return Err(StreamError::Raw(RawSequencesError::Format("full".into())));
        }
        Ok(())
    });
    assert!(matches!(outcome, Err(StreamError::Raw(_))));
    assert_eq!(calls, 11, "no trajectory is handed on after the error");
}

#[test]
fn a_refused_observation_runs_no_replica() {
    let mut calls = 0;
    let unknown = vec![SequenceObservation {
        indicator: "nothing".into(),
        time: 1.0,
    }];
    let outcome = run_sequences_streamed(&compiled(), &config(None), &unknown, |_, _, _| {
        calls += 1;
        Ok::<_, EngineError>(())
    });
    assert!(outcome.is_err());
    assert_eq!(calls, 0);
}

#[test]
fn the_writer_refuses_a_count_the_header_does_not_state() {
    let header = RawHeader::new("test", "pair", 5, 1, T_MAX, vec![]);
    let seq = Sequence {
        events: Vec::new(),
        end_cause: None,
        end_time: T_MAX,
        weight: 1.0,
    };
    let short = RawCorpusWriter::new(Vec::new(), &header).unwrap();
    assert!(matches!(short.finish(), Err(RawSequencesError::Format(_))));

    let mut long = RawCorpusWriter::new(Vec::new(), &header).unwrap();
    long.push(&seq, &[]).unwrap();
    assert!(matches!(
        long.push(&seq, &[]),
        Err(RawSequencesError::Format(_))
    ));
    assert!(matches!(
        long.push(&seq, &[1.0]),
        Err(RawSequencesError::Format(_))
    ));
}

#[test]
fn the_reader_reports_a_short_corpus_last_then_stops() {
    let seq = Sequence {
        events: Vec::new(),
        end_cause: None,
        end_time: T_MAX,
        weight: 1.0,
    };
    // A two-trajectory corpus whose header then claims three: the writer
    // itself refuses to produce a short corpus.
    let mut writer = RawCorpusWriter::new(
        Vec::new(),
        &RawHeader::new("test", "pair", 5, 2, T_MAX, vec![]),
    )
    .unwrap();
    writer.push(&seq, &[]).unwrap();
    writer.push(&seq, &[]).unwrap();
    let text = String::from_utf8(writer.finish().unwrap()).unwrap();
    let bytes = text
        .replacen("\"nb_runs\":2", "\"nb_runs\":3", 1)
        .into_bytes();
    assert_ne!(bytes, text.as_bytes(), "the header was rewritten");

    let mut reader = RawCorpusReader::new(bytes.as_slice()).unwrap();
    assert_eq!(reader.header().nb_runs, 3);
    assert!(reader.next().unwrap().is_ok());
    assert!(reader.next().unwrap().is_ok());
    assert!(matches!(
        reader.next(),
        Some(Err(RawSequencesError::Format(_)))
    ));
    assert!(reader.next().is_none(), "nothing follows the error");
}

/// What a sink that writes a corpus can fail with; the payloads are read
/// through `Debug` only, when a test fails.
#[derive(Debug)]
#[allow(dead_code)]
enum StreamError {
    Engine(EngineError),
    Raw(RawSequencesError),
}

impl From<EngineError> for StreamError {
    fn from(e: EngineError) -> Self {
        StreamError::Engine(e)
    }
}

impl From<RawSequencesError> for StreamError {
    fn from(e: RawSequencesError) -> Self {
        StreamError::Raw(e)
    }
}
