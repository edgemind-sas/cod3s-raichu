//! The raw sequence corpus (`raichu.sequences` v1): what a campaign records,
//! written whole, read back exactly, and reduced to the same minimal
//! sequences the analysis returns.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeSet;

use serde_json::{json, Value};

use raichu_analysis::{
    analyse, clean, minimal_sequences, read_raw_sequences, write_raw_sequences, RawHeader,
    RawObservation, RawSequencesError, RAW_SEQUENCES_FORMAT, RAW_SEQUENCES_VERSION,
};
use raichu_core::{CompiledModel, SolverParams};
use raichu_model::{Model, TransitionKind};
use raichu_montecarlo::{run_sequences, McConfig, DEFAULT_CONFIDENCE};

/// Two repairable components on exponential laws, their failure and repair
/// monitored as cycle pairs, and a feared event reached the instant both are
/// down: trajectories carry cycles that net out, and several distinct paths.
const PAIR: &str = r#"
{
  "name": "pair",
  "components": [
    {"name": "A", "ports": [], "attributes": [], "automata": [{"name": "life", "states": ["ok", "ko"], "init": "ok",
      "transitions": [
        {"name": "fail", "source": "ok", "targets": ["ko"], "distrib": "exp", "rate": 0.3, "monitored": true, "cycle_group": "life", "kind": "failure"},
        {"name": "fix", "source": "ko", "targets": ["ok"], "distrib": "exp", "rate": 1.0, "monitored": true, "cycle_group": "life", "kind": "repair"}]}]},
    {"name": "B", "ports": [], "attributes": [], "automata": [{"name": "life", "states": ["ok", "ko"], "init": "ok",
      "transitions": [
        {"name": "fail", "source": "ok", "targets": ["ko"], "distrib": "exp", "rate": 0.2, "monitored": true, "cycle_group": "life", "kind": "failure"},
        {"name": "fix", "source": "ko", "targets": ["ok"], "distrib": "exp", "rate": 1.0, "monitored": true, "cycle_group": "life", "kind": "repair"}]}]},
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
        confidence: DEFAULT_CONFIDENCE,
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
    assert!(
        raw.iter().any(|s| s
            .events
            .iter()
            .any(|e| e.kind == Some(TransitionKind::Repair))),
        "the corpus must carry kinds, or the round trip proves nothing about them"
    );
}

#[test]
fn the_header_reads_back_to_the_same_bits() {
    // A value serde_json's default float parser reads one ulp off.
    let x = 0.248_720_059_264_354_14;
    let mut header = RawHeader::new("test", "pair", 11, 0, x, vec!["both_down".into()]);
    header.observations.push(RawObservation {
        name: "late".into(),
        time: x,
    });
    let mut bytes = Vec::new();
    write_raw_sequences(&mut bytes, &header, &[]).unwrap();
    let (read, sequences) = read_raw_sequences(bytes.as_slice()).unwrap();
    assert!(sequences.is_empty());
    assert_eq!(read.t_max.to_bits(), x.to_bits());
    assert_eq!(read.observations[0].time.to_bits(), x.to_bits());
    assert_eq!(read, header);
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
    newer[0] = newer[0].replace(
        &format!("\"version\":{RAW_SEQUENCES_VERSION}"),
        &format!("\"version\":{}", RAW_SEQUENCES_VERSION + 1),
    );
    assert_ne!(newer[0], lines[0]);
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

/// A corpus line from before the `automata` field: it reads, every event
/// naming no automaton. A line whose `automata` does not match its events
/// is refused.
#[test]
fn the_automata_field_is_optional_and_checked() {
    let header = r#"{"format":"raichu.sequences","version":1,"engine_version":"0.82.1","model":"m","seed":1,"nb_runs":1,"t_max":1.0,"targets":[],"event_fields":["time","obj","attr","cycle_group"]}"#;
    let old = format!(
        "{header}\n{}\n",
        r#"{"run":0,"end_cause":null,"end_time":1.0,"events":[[0.5,"A","ko","life"]]}"#
    );
    let (_, sequences) = read_raw_sequences(old.as_bytes()).unwrap();
    assert_eq!(sequences[0].events[0].automaton, "");

    let named = format!(
        "{header}\n{}\n",
        r#"{"run":0,"end_cause":null,"end_time":1.0,"events":[[0.5,"A","ko","life"]],"automata":["life"]}"#
    );
    let (_, sequences) = read_raw_sequences(named.as_bytes()).unwrap();
    assert_eq!(sequences[0].events[0].automaton, "life");

    let short = format!(
        "{header}\n{}\n",
        r#"{"run":0,"end_cause":null,"end_time":1.0,"events":[[0.5,"A","ko","life"]],"automata":[]}"#
    );
    assert!(read_raw_sequences(short.as_bytes()).is_ok());
    let wrong = format!(
        "{header}\n{}\n",
        r#"{"run":0,"end_cause":null,"end_time":1.0,"events":[[0.5,"A","ko","life"]],"automata":["a","b"]}"#
    );
    let error = read_raw_sequences(wrong.as_bytes()).unwrap_err();
    assert!(
        matches!(error, RawSequencesError::Format(ref m) if m.contains("names 2 automata for 1 events")),
        "{error}"
    );
}

/// A corpus line from before the `kinds` field reads with every kind
/// absent; a line whose `kinds` does not match its events is refused.
#[test]
fn the_kinds_field_is_optional_and_checked() {
    let header = r#"{"format":"raichu.sequences","version":1,"engine_version":"0.84.0","model":"m","seed":1,"nb_runs":1,"t_max":1.0,"targets":[],"event_fields":["time","obj","attr","cycle_group"]}"#;
    let line = |extra: &str| {
        format!(
            "{header}\n{{\"run\":0,\"end_cause\":null,\"end_time\":1.0,\"events\":[[0.5,\"A\",\"ko\",\"life\"],[0.7,\"A\",\"ok\",\"life\"]]{extra}}}\n"
        )
    };
    let (_, sequences) = read_raw_sequences(line("").as_bytes()).unwrap();
    assert!(sequences[0].events.iter().all(|e| e.kind.is_none()));

    let (_, sequences) =
        read_raw_sequences(line(r#","kinds":["failure",null]"#).as_bytes()).unwrap();
    assert_eq!(
        sequences[0]
            .events
            .iter()
            .map(|e| e.kind)
            .collect::<Vec<_>>(),
        vec![Some(TransitionKind::Failure), None]
    );

    let error = read_raw_sequences(line(r#","kinds":["failure"]"#).as_bytes()).unwrap_err();
    assert!(
        matches!(error, RawSequencesError::Format(ref m) if m.contains("gives 1 kinds for 2 events")),
        "{error}"
    );
}

/// Version 2 carries the per-group decisions the reduction applies; a
/// version 1 corpus carries none and reads with them unknown.
#[test]
fn the_header_carries_the_cycle_group_decisions() {
    let compiled = compiled();
    assert_eq!(compiled.cycle_groups.len(), 2);
    let raw = run_sequences(&compiled, &config(None)).unwrap();
    let header = header().with_cycle_groups(compiled.cycle_groups.clone());
    let mut bytes = Vec::new();
    write_raw_sequences(&mut bytes, &header, &raw).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert!(text.starts_with(&format!(
        "{{\"format\":\"raichu.sequences\",\"version\":{RAW_SEQUENCES_VERSION}"
    )));
    let (read_header, _) = read_raw_sequences(text.as_bytes()).unwrap();
    assert_eq!(
        read_header.cycle_groups,
        Some(compiled.cycle_groups.clone())
    );
    assert!(read_header.reducer().report().cycle_groups.is_some());

    let v1 = r#"{"format":"raichu.sequences","version":1,"engine_version":"0.85.0","model":"m","seed":1,"nb_runs":1,"t_max":1.0,"targets":[],"event_fields":["time","obj","attr","cycle_group"]}
{"run":0,"end_cause":null,"end_time":1.0,"events":[[0.5,"A","ko","life"]]}
"#;
    let (old, _) = read_raw_sequences(v1.as_bytes()).unwrap();
    assert!(old.cycle_groups.is_none());
    assert!(old.reducer().report().cycle_groups.is_none());
}

/// `(automaton, kind)` of every event of a level serialised as the binding
/// returns it, after checking each event carries exactly the five fields
/// of the reduced levels.
fn event_roles(level: &[raichu_core::Sequence]) -> Vec<(String, String, Value, Value)> {
    let json = serde_json::to_value(level).unwrap();
    let mut roles = Vec::new();
    for sequence in json.as_array().unwrap() {
        for event in sequence["events"].as_array().unwrap() {
            let keys: BTreeSet<&str> = event
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            assert_eq!(
                keys,
                BTreeSet::from(["obj", "automaton", "attr", "time", "kind"]),
                "{event}"
            );
            roles.push((
                event["obj"].as_str().unwrap().to_owned(),
                event["attr"].as_str().unwrap().to_owned(),
                event["automaton"].clone(),
                event["kind"].clone(),
            ));
        }
    }
    roles
}

/// The cleaned and minimal levels carry, on each event, the automaton and
/// the declared kind the raw corpus records for it. The states are `ok` and
/// `ko`, not muscadet's `occ` and `rep`: the role comes from the
/// declaration, never from a state name. The feared event's transition
/// declares no kind, and reads `null`.
#[test]
fn the_reduced_levels_carry_each_event_s_automaton_and_kind() {
    let raw = run_sequences(&compiled(), &config(None)).unwrap();
    let cleaned = clean(raw);
    let minimal = minimal_sequences(cleaned.clone());
    for level in [&cleaned, &minimal] {
        let roles = event_roles(level);
        assert!(roles.iter().any(|(obj, ..)| obj == "A"));
        assert!(roles.iter().any(|(obj, ..)| obj == "ER"));
        for (obj, attr, automaton, kind) in roles {
            let expected = match (obj.as_str(), attr.as_str()) {
                ("A" | "B", "ko") => json!(["life", "failure"]),
                ("A" | "B", "ok") => json!(["life", "repair"]),
                ("ER", "occ") => json!(["ev", null]),
                other => panic!("unexpected event {other:?}"),
            };
            assert_eq!(json!([automaton, kind]), expected, "{obj}.{attr}");
        }
    }
}

/// A corpus written before the `automata` and `kinds` arrays reduces as it
/// did, its events reading `null` for both: the corpus does not say, and
/// the levels do not guess.
#[test]
fn a_corpus_without_automata_or_kinds_reduces_with_both_null() {
    let text = String::from_utf8(written(None)).unwrap();
    let mut lines = text.lines();
    let mut old = format!("{}\n", lines.next().unwrap());
    for line in lines {
        let mut record: Value = serde_json::from_str(line).unwrap();
        let fields = record.as_object_mut().unwrap();
        assert!(fields.remove("automata").is_some() || fields["events"] == json!([]));
        fields.remove("kinds");
        old.push_str(&format!("{record}\n"));
    }
    let (_, read) = read_raw_sequences(old.as_bytes()).unwrap();
    let cleaned = clean(read);
    let (_, current) = read_raw_sequences(text.as_bytes()).unwrap();
    assert_eq!(
        cleaned.iter().map(|s| s.weight).collect::<Vec<_>>(),
        clean(current).iter().map(|s| s.weight).collect::<Vec<_>>()
    );
    for level in [cleaned.clone(), minimal_sequences(cleaned)] {
        for (obj, attr, automaton, kind) in event_roles(&level) {
            assert_eq!(
                json!([automaton, kind]),
                json!([null, null]),
                "{obj}.{attr}"
            );
        }
    }
}
