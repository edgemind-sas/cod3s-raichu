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
//! # Observations
//!
//! A campaign may also record, for every trajectory, the value of chosen
//! attributes at chosen instants: the header's optional `observations` names
//! them (`[{"name", "time"}]`), and each trajectory line then carries an
//! `observed` array holding their values in that order (a boolean reads `0`
//! or `1`). Both fields are absent when nothing is observed, so a corpus
//! without observations is byte-for-byte what earlier engines wrote, and a
//! reader that predates them ignores them. What they are for is a
//! **condition** on the trajectories ([`ObservedCondition`]): keep only
//! those whose observed value satisfies it, before the reduction.
//!
//! A reader refuses another `format` and a `version` above the one it knows.
//! Fields may be added to either kind of line without a new version as long
//! as a reader that ignores them still reads the corpus right; a v1 reader
//! ignores what it does not know.

use std::io::{BufRead, Write};

use raichu_expr::CmpOp;
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
    /// What every trajectory line's `observed` array holds, in order; empty
    /// (and absent from the file) when the campaign observed nothing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observations: Vec<RawObservation>,
}

/// One observed quantity: an attribute's value at one instant of every
/// trajectory.
///
/// `time` is the instant as the campaign sampled it. A trajectory stopped
/// at a feared event holds its final state through the later instants, and
/// an instant at an event date reads the state after the event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawObservation {
    /// The name a condition refers to it by, unique in the corpus.
    pub name: String,
    /// The instant it was read at.
    pub time: f64,
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
            observations: Vec::new(),
        }
    }

    /// The same header, declaring what each trajectory's `observed` array
    /// holds.
    #[must_use]
    pub fn with_observations(mut self, observations: Vec<RawObservation>) -> Self {
        self.observations = observations;
        self
    }
}

/// A raw corpus read back whole: its header, its trajectories in replica
/// order, and each trajectory's observed values (one row per trajectory,
/// in the order of [`RawHeader::observations`]; rows are empty when the
/// campaign observed nothing).
#[derive(Debug, Clone, PartialEq)]
pub struct RawCorpus {
    /// The first line.
    pub header: RawHeader,
    /// The trajectories, each weighing one.
    pub sequences: Vec<Sequence>,
    /// The observed values, one row per trajectory.
    pub observed: Vec<Vec<f64>>,
}

/// A condition on the trajectories of a corpus: keep a trajectory when its
/// value of `observation` compares to `value` as `op` says.
///
/// The comparison is on the recorded double, exactly: a boolean attribute
/// was recorded as `0` or `1`, so `== 1` keeps the trajectories where it
/// held.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservedCondition {
    /// The name of the observation it reads, among the header's.
    pub observation: String,
    /// The comparison.
    pub op: CmpOp,
    /// The constant the observed value is compared against.
    pub value: f64,
}

impl ObservedCondition {
    /// Whether `observed` satisfies the condition.
    #[must_use]
    pub fn holds(&self, observed: f64) -> bool {
        match self.op {
            CmpOp::Eq => observed == self.value,
            CmpOp::Ne => observed != self.value,
            CmpOp::Lt => observed < self.value,
            CmpOp::Le => observed <= self.value,
            CmpOp::Gt => observed > self.value,
            CmpOp::Ge => observed >= self.value,
        }
    }
}

impl RawCorpus {
    /// The trajectories that satisfy `condition`, in replica order.
    ///
    /// # Errors
    /// [`RawSequencesError::Format`] when the corpus observed nothing by
    /// that name: a condition on a quantity the campaign did not record
    /// cannot be decided, and keeping every trajectory would read as a
    /// filter that passed them all.
    pub fn filtered(
        &self,
        condition: &ObservedCondition,
    ) -> Result<Vec<Sequence>, RawSequencesError> {
        let position = self
            .header
            .observations
            .iter()
            .position(|o| o.name == condition.observation)
            .ok_or_else(|| {
                let known: Vec<&str> = self
                    .header
                    .observations
                    .iter()
                    .map(|o| o.name.as_str())
                    .collect();
                RawSequencesError::Format(format!(
                    "the corpus observed no `{}`; it observed {}",
                    condition.observation,
                    if known.is_empty() {
                        "nothing".to_owned()
                    } else {
                        format!("{known:?}")
                    }
                ))
            })?;
        Ok(self
            .sequences
            .iter()
            .zip(&self.observed)
            .filter(|(_, row)| condition.holds(row[position]))
            .map(|(sequence, _)| sequence.clone())
            .collect())
    }
}

/// One trajectory line, as written.
#[derive(Serialize)]
struct RawRecord<'a> {
    run: u64,
    end_cause: &'a Option<String>,
    end_time: f64,
    events: Vec<(f64, &'a str, &'a str, &'a Option<String>)>,
    #[serde(skip_serializing_if = "<[f64]>::is_empty")]
    observed: &'a [f64],
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
    #[serde(default)]
    observed: Vec<Box<serde_json::value::RawValue>>,
}

/// A number (a date or an observed value) read back from its JSON lexeme.
fn number(raw: &serde_json::value::RawValue, line: usize) -> Result<f64, RawSequencesError> {
    raw.get()
        .parse::<f64>()
        .map_err(|e| RawSequencesError::Line {
            line,
            detail: format!("`{}` is not a number: {e}", raw.get()),
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
/// it. A header that declares observations needs [`write_raw_corpus`].
///
/// # Errors
/// [`RawSequencesError::Format`] when the count differs from the header's,
/// or when the header declares observations;
/// [`RawSequencesError::Io`] when the writer fails.
pub fn write_raw_sequences<W: Write>(
    writer: W,
    header: &RawHeader,
    sequences: &[Sequence],
) -> Result<(), RawSequencesError> {
    write_raw_corpus(writer, header, sequences, &[])
}

/// Write `sequences` and their observed values as a raw corpus under
/// `header`, one line each.
///
/// `observed` holds one row per trajectory, each with one value per
/// observation the header declares, in that order; it is empty when the
/// header declares none.
///
/// # Errors
/// [`RawSequencesError::Format`] when a count differs from what the header
/// states; [`RawSequencesError::Io`] when the writer fails.
pub fn write_raw_corpus<W: Write>(
    mut writer: W,
    header: &RawHeader,
    sequences: &[Sequence],
    observed: &[Vec<f64>],
) -> Result<(), RawSequencesError> {
    if sequences.len() as u64 != header.nb_runs {
        return Err(RawSequencesError::Format(format!(
            "the header states {} trajectories and {} are written",
            header.nb_runs,
            sequences.len()
        )));
    }
    let width = header.observations.len();
    if width == 0 && !observed.is_empty() {
        return Err(RawSequencesError::Format(
            "observed values are written under a header that declares no observation".to_owned(),
        ));
    }
    if width > 0 {
        if observed.len() != sequences.len() {
            return Err(RawSequencesError::Format(format!(
                "{} trajectories are written with {} rows of observed values",
                sequences.len(),
                observed.len()
            )));
        }
        if let Some((run, row)) = observed
            .iter()
            .enumerate()
            .find(|(_, row)| row.len() != width)
        {
            return Err(RawSequencesError::Format(format!(
                "run {run} has {} observed values where the header declares {width}",
                row.len()
            )));
        }
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
            observed: observed.get(run).map_or(&[], Vec::as_slice),
        };
        writer.write_all(json_line(&record)?.as_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

/// Read a raw corpus back: its header and its trajectories, in replica
/// order, each weighing one. Observed values, if any, are dropped: read
/// them with [`read_raw_corpus`].
///
/// # Errors
/// As [`read_raw_corpus`].
pub fn read_raw_sequences<R: BufRead>(
    reader: R,
) -> Result<(RawHeader, Vec<Sequence>), RawSequencesError> {
    let corpus = read_raw_corpus(reader)?;
    Ok((corpus.header, corpus.sequences))
}

/// Read a raw corpus back whole: header, trajectories and observed values.
///
/// # Errors
/// [`RawSequencesError::Line`] on a line that is not the expected JSON;
/// [`RawSequencesError::Format`] on another format, a version above
/// [`RAW_SEQUENCES_VERSION`], runs out of order, a trajectory count that
/// is not the header's, or a trajectory whose observed values are not one
/// per declared observation.
pub fn read_raw_corpus<R: BufRead>(reader: R) -> Result<RawCorpus, RawSequencesError> {
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
    let width = header.observations.len();
    let mut sequences = Vec::new();
    let mut observed = Vec::new();
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
                    time: number(&time, index + 1)?,
                    cycle_group,
                })
            })
            .collect::<Result<Vec<_>, RawSequencesError>>()?;
        if record.observed.len() != width {
            return Err(RawSequencesError::Format(format!(
                "line {} holds {} observed values where the header declares {width}",
                index + 1,
                record.observed.len()
            )));
        }
        if width > 0 {
            observed.push(
                record
                    .observed
                    .iter()
                    .map(|value| number(value, index + 1))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        sequences.push(Sequence {
            events,
            end_cause: record.end_cause,
            end_time: number(&record.end_time, index + 1)?,
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
    Ok(RawCorpus {
        header,
        sequences,
        observed,
    })
}

/// One JSON document on one line.
fn json_line<T: Serialize>(value: &T) -> Result<String, RawSequencesError> {
    let mut text = serde_json::to_string(value)
        .map_err(|e| RawSequencesError::Format(format!("cannot encode a line: {e}")))?;
    text.push('\n');
    Ok(text)
}
