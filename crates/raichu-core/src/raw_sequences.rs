//! The **raw sequence corpus** as an artefact: `raichu.sequences`, version 1.
//!
//! A sequence campaign records one [`Sequence`] per trajectory, in replica
//! order, and the analysis ([`crate::sequence::analyse`]) reduces them to the
//! minimal sequences a study reads. The raw corpus is what the reduction
//! starts from, and keeping it lets anyone recompute, audit or filter the
//! reduction without re-running the campaign.
//!
//! # The format
//!
//! [JSON Lines](https://jsonlines.org): one JSON document per line, UTF-8.
//!
//! - **Line 1, the header** ([`RawHeader`]): `format` (`"raichu.sequences"`),
//!   `version` (`1`), `engine_version`, `model`, `seed`, `nb_runs`, `t_max`,
//!   `targets` (the feared events the campaign stops at), and `event_fields`,
//!   the names of the positions of an event array (`["time", "obj", "attr",
//!   "cycle_group"]`).
//! - **One line per trajectory**, in replica order: `run` (0-based replica
//!   index), `end_cause` (the reached target's name, or `null` when the
//!   trajectory ran to the horizon), `end_time`, and `events`, each an array
//!   ordered as `event_fields` says: the firing date, the component, the
//!   monitored state entered, and the transition's cycle group (`null` when
//!   it has none).
//!
//! Every trajectory weighs one; the file has exactly `nb_runs` trajectory
//! lines. The cycle group is carried because it is what the cycle filter of
//! the reduction reads: without it the reduction could not be recomputed from
//! the file.
//!
//! A reader refuses another `format` and a `version` above the one it knows.
//! A later version may add fields to either kind of line; a v1 reader ignores
//! what it does not know.

use std::io::{BufRead, Write};

use serde::{Deserialize, Serialize};

use crate::engine::{SeqEvent, Sequence};

/// The `format` of a raw sequence corpus.
pub const RAW_SEQUENCES_FORMAT: &str = "raichu.sequences";

/// The version this module writes, and the highest it reads.
pub const RAW_SEQUENCES_VERSION: u32 = 1;

/// The positions of an event array, in order.
const EVENT_FIELDS: [&str; 4] = ["time", "obj", "attr", "cycle_group"];

/// The first line of a raw sequence corpus: what produced it and how.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawHeader {
    /// Always [`RAW_SEQUENCES_FORMAT`].
    pub format: String,
    /// The format version, [`RAW_SEQUENCES_VERSION`] when written here.
    pub version: u32,
    /// The engine version that ran the campaign.
    pub engine_version: String,
    /// The model's name.
    pub model: String,
    /// The campaign's master seed.
    pub seed: u64,
    /// The number of replicas, hence of trajectory lines.
    pub nb_runs: u64,
    /// The horizon.
    pub t_max: f64,
    /// The feared events the campaign stops at, in declaration order.
    pub targets: Vec<String>,
    /// The names of the positions of an event array.
    pub event_fields: Vec<String>,
}

impl RawHeader {
    /// A version-1 header.
    #[must_use]
    pub fn new(
        engine_version: &str,
        model: &str,
        seed: u64,
        nb_runs: u64,
        t_max: f64,
        targets: Vec<String>,
    ) -> Self {
        RawHeader {
            format: RAW_SEQUENCES_FORMAT.to_owned(),
            version: RAW_SEQUENCES_VERSION,
            engine_version: engine_version.to_owned(),
            model: model.to_owned(),
            seed,
            nb_runs,
            t_max,
            targets,
            event_fields: EVENT_FIELDS.iter().map(|f| (*f).to_owned()).collect(),
        }
    }
}

/// One trajectory line, as written.
#[derive(Serialize)]
struct RawRecord<'a> {
    run: u64,
    end_cause: &'a Option<String>,
    end_time: f64,
    events: Vec<(f64, &'a str, &'a str, &'a Option<String>)>,
}

/// One trajectory line, as read. The dates are kept as their JSON lexemes
/// and parsed by [`str::parse`], which rounds correctly: `serde_json`'s own
/// float reading may land one ulp off, and a corpus that does not read back
/// bit for bit could not be re-reduced to the campaign's own answer.
#[derive(Deserialize)]
struct ReadRecord {
    run: u64,
    end_cause: Option<String>,
    end_time: Box<serde_json::value::RawValue>,
    events: Vec<(
        Box<serde_json::value::RawValue>,
        String,
        String,
        Option<String>,
    )>,
}

/// A date read back from its JSON lexeme.
fn date(raw: &serde_json::value::RawValue, line: usize) -> Result<f64, RawSequencesError> {
    raw.get()
        .parse::<f64>()
        .map_err(|e| RawSequencesError::Line {
            line,
            detail: format!("`{}` is not a date: {e}", raw.get()),
        })
}

/// Why a raw sequence corpus could not be written or read.
#[derive(Debug, thiserror::Error)]
pub enum RawSequencesError {
    /// The underlying stream failed.
    #[error("raw sequence corpus: {0}")]
    Io(#[from] std::io::Error),
    /// A line is not the JSON this format expects.
    #[error("raw sequence corpus, line {line}: {detail}")]
    Line {
        /// 1-based line number.
        line: usize,
        /// What was wrong with it.
        detail: String,
    },
    /// The corpus is valid JSON but not this format, or not a version this
    /// reader knows, or not the number of trajectories its header states.
    #[error("raw sequence corpus: {0}")]
    Format(String),
}

/// Write `sequences` as a raw corpus under `header`, one line each.
///
/// `sequences` must be the campaign's raw trajectories in replica order,
/// `header.nb_runs` of them: the file states that count, and a reader checks
/// it.
///
/// # Errors
/// [`RawSequencesError::Format`] when the count differs from the header's;
/// [`RawSequencesError::Io`] when the writer fails.
pub fn write_raw_sequences<W: Write>(
    mut writer: W,
    header: &RawHeader,
    sequences: &[Sequence],
) -> Result<(), RawSequencesError> {
    if sequences.len() as u64 != header.nb_runs {
        return Err(RawSequencesError::Format(format!(
            "the header states {} trajectories and {} are written",
            header.nb_runs,
            sequences.len()
        )));
    }
    writer.write_all(json_line(header)?.as_bytes())?;
    for (run, sequence) in sequences.iter().enumerate() {
        let record = RawRecord {
            run: run as u64,
            end_cause: &sequence.end_cause,
            end_time: sequence.end_time,
            events: sequence
                .events
                .iter()
                .map(|e| (e.time, e.obj.as_str(), e.attr.as_str(), &e.cycle_group))
                .collect(),
        };
        writer.write_all(json_line(&record)?.as_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

/// Read a raw corpus back: its header and its trajectories, in replica
/// order, each weighing one.
///
/// # Errors
/// [`RawSequencesError::Line`] on a line that is not the expected JSON;
/// [`RawSequencesError::Format`] on another format, a version above
/// [`RAW_SEQUENCES_VERSION`], runs out of order, or a trajectory count that
/// is not the header's.
pub fn read_raw_sequences<R: BufRead>(
    reader: R,
) -> Result<(RawHeader, Vec<Sequence>), RawSequencesError> {
    let mut lines = reader.lines().enumerate();
    let header: RawHeader = match lines.next() {
        None => return Err(RawSequencesError::Format("the corpus is empty".to_owned())),
        Some((_, line)) => serde_json::from_str(&line?).map_err(|e| RawSequencesError::Line {
            line: 1,
            detail: e.to_string(),
        })?,
    };
    if header.format != RAW_SEQUENCES_FORMAT {
        return Err(RawSequencesError::Format(format!(
            "the format is `{}`, not `{RAW_SEQUENCES_FORMAT}`",
            header.format
        )));
    }
    if header.version > RAW_SEQUENCES_VERSION {
        return Err(RawSequencesError::Format(format!(
            "version {} is newer than the {RAW_SEQUENCES_VERSION} this reader knows",
            header.version
        )));
    }
    let mut sequences = Vec::new();
    for (index, line) in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: ReadRecord =
            serde_json::from_str(&line).map_err(|e| RawSequencesError::Line {
                line: index + 1,
                detail: e.to_string(),
            })?;
        if record.run != sequences.len() as u64 {
            return Err(RawSequencesError::Format(format!(
                "line {} holds run {} where run {} was expected",
                index + 1,
                record.run,
                sequences.len()
            )));
        }
        let events = record
            .events
            .into_iter()
            .map(|(time, obj, attr, cycle_group)| {
                Ok(SeqEvent {
                    obj,
                    attr,
                    time: date(&time, index + 1)?,
                    cycle_group,
                })
            })
            .collect::<Result<Vec<_>, RawSequencesError>>()?;
        sequences.push(Sequence {
            events,
            end_cause: record.end_cause,
            end_time: date(&record.end_time, index + 1)?,
            weight: 1.0,
        });
    }
    if sequences.len() as u64 != header.nb_runs {
        return Err(RawSequencesError::Format(format!(
            "the header states {} trajectories and the corpus holds {}",
            header.nb_runs,
            sequences.len()
        )));
    }
    Ok((header, sequences))
}

/// One JSON document on one line.
fn json_line<T: Serialize>(value: &T) -> Result<String, RawSequencesError> {
    let mut text = serde_json::to_string(value)
        .map_err(|e| RawSequencesError::Format(format!("cannot encode a line: {e}")))?;
    text.push('\n');
    Ok(text)
}
