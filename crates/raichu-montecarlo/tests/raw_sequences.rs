//! The raw sequence corpus (`raichu.sequences` v1): what a campaign records,
//! written whole, read back exactly, and reduced to the same minimal
//! sequences the analysis returns.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::{
    analyse, clean, minimal_sequences, read_raw_sequences, write_raw_sequences, CompiledModel,
    RawHeader, RawSequencesError, SolverParams, RAW_SEQUENCES_FORMAT, RAW_SEQUENCES_VERSION,
};
use raichu_model::Model;
use raichu_montecarlo::{run_sequences, McConfig};

/// Two repairable components on exponential laws, their failure and repair
/// monitored as cycle pairs, and a feared event reached the instant both are
/// down: trajectories carry cycles that net out, and several distinct paths.
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
  "targets": [{"name": "both_down", "component": "ER", "automaton": "ev", "state": "occ"}]
}
"#;

const NB_RUNS: u64 = 300;
const T_MAX: f64 = 20.0;

fn compiled() -> CompiledModel {
    CompiledModel::compile(&Model::from_json(PAIR).expect("model loads")).expect("compiles")
}

fn config(threads: Option<usize>) -> McConfig {
    McConfig {
        nb_runs: NB_RUNS,
        seed: 11,
        t_max: T_MAX,
        samples: Vec::new(),
        threads,
        quantiles: Vec::new(),
        ode: SolverParams::default(),
        stop_at_targets: false,
        flow: Default::default(),
    }
}

fn header() -> RawHeader {
    RawHeader::new("test", "pair", 11, NB_RUNS, T_MAX, vec!["both_down".into()])
}

fn written(threads: Option<usize>) -> Vec<u8> {
    let raw = run_sequences(&compiled(), &config(threads)).expect("campaign runs");
    let mut bytes = Vec::new();
    write_raw_sequences(&mut bytes, &header(), &raw).expect("written");
    bytes
}

#[test]
fn the_corpus_reads_back_exactly_what_the_campaign_recorded() {
    let raw = run_sequences(&compiled(), &config(None)).unwrap();
    let mut bytes = Vec::new();
    write_raw_sequences(&mut bytes, &header(), &raw).unwrap();

    let (read_header, read) = read_raw_sequences(bytes.as_slice()).unwrap();
    assert_eq!(read_header, header());
    assert_eq!(
        read, raw,
        "times, states, cycle groups, end causes, all of it"
    );
    assert!(
        raw.iter()
            .any(|s| s.events.iter().any(|e| e.cycle_group.is_some())),
        "the corpus must carry cycle groups, or the round trip proves nothing about them"
    );
}

#[test]
fn the_file_is_one_header_line_then_one_line_per_replica() {
    let bytes = written(None);
    let text = String::from_utf8(bytes).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len() as u64, NB_RUNS + 1);
    let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(first["format"], RAW_SEQUENCES_FORMAT);
    assert_eq!(first["version"], RAW_SEQUENCES_VERSION);
    assert_eq!(
        first["event_fields"],
        serde_json::json!(["time", "obj", "attr", "cycle_group"])
    );
    let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(second["run"], 0);
}

#[test]
fn reducing_the_read_back_corpus_gives_the_analysis_minimal_sequences() {
    let raw = run_sequences(&compiled(), &config(None)).unwrap();
    let mut bytes = Vec::new();
    write_raw_sequences(&mut bytes, &header(), &raw).unwrap();
    let (_, read) = read_raw_sequences(bytes.as_slice()).unwrap();

    assert_eq!(minimal_sequences(clean(read)), analyse(raw));
}

#[test]
fn the_cleaned_level_accounts_for_every_trajectory() {
    let cleaned = clean(run_sequences(&compiled(), &config(None)).unwrap());
    let total: f64 = cleaned.iter().map(|s| s.weight).sum();
    assert_eq!(total, NB_RUNS as f64);
    assert!(
        cleaned
            .iter()
            .all(|s| s.events.windows(2).all(|w| w[0].time <= w[1].time)),
        "events stay in time order"
    );
}

#[test]
fn the_corpus_does_not_depend_on_the_thread_count() {
    assert_eq!(written(Some(1)), written(Some(4)));
}

fn read_text(text: &str) -> Result<(RawHeader, Vec<raichu_core::Sequence>), RawSequencesError> {
    read_raw_sequences(text.as_bytes())
}

#[test]
fn a_reader_refuses_what_it_cannot_trust() {
    let text = String::from_utf8(written(None)).unwrap();
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();

    let mut other_format = lines.clone();
    other_format[0] = other_format[0].replace(RAW_SEQUENCES_FORMAT, "pycatshoo.xml");
    assert!(matches!(
        read_text(&other_format.join("\n")),
        Err(RawSequencesError::Format(_))
    ));

    let mut newer = lines.clone();
    newer[0] = newer[0].replace("\"version\":1", "\"version\":2");
    assert!(matches!(
        read_text(&newer.join("\n")),
        Err(RawSequencesError::Format(_))
    ));

    let mut short = lines.clone();
    short.pop();
    assert!(matches!(
        read_text(&short.join("\n")),
        Err(RawSequencesError::Format(_))
    ));

    lines.swap(1, 2);
    assert!(matches!(
        read_text(&lines.join("\n")),
        Err(RawSequencesError::Format(_))
    ));

    assert!(matches!(
        read_text("not json\n"),
        Err(RawSequencesError::Line { line: 1, .. })
    ));
    assert!(matches!(read_text(""), Err(RawSequencesError::Format(_))));
}

#[test]
fn writing_refuses_a_count_the_header_does_not_state() {
    let raw = run_sequences(&compiled(), &config(None)).unwrap();
    let mut bytes = Vec::new();
    let wrong = RawHeader::new("test", "pair", 11, NB_RUNS + 1, T_MAX, vec![]);
    assert!(matches!(
        write_raw_sequences(&mut bytes, &wrong, &raw),
        Err(RawSequencesError::Format(_))
    ));
}
