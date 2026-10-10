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
//!   it has none). An optional `automata` array, parallel to `events`, names
//!   the automaton of each event inside its component; it is absent when
//!   the source named none, and a reader that predates it ignores it.
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

use raichu_core::{SeqEvent, Sequence};

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

    /// The position of the observation named `name` in a trajectory's
    /// `observed` array.
    ///
    /// # Errors
    /// [`RawSequencesError::Format`] when the corpus observed nothing by
    /// that name: a condition on a quantity the campaign did not record
    /// cannot be decided, and keeping every trajectory would read as a
    /// filter that passed them all.
    pub fn observation_index(&self, name: &str) -> Result<usize, RawSequencesError> {
        self.observations
            .iter()
            .position(|o| o.name == name)
            .ok_or_else(|| {
                let known: Vec<&str> = self.observations.iter().map(|o| o.name.as_str()).collect();
                RawSequencesError::Format(format!(
                    "the corpus observed no `{name}`; it observed {}",
                    if known.is_empty() {
                        "nothing".to_owned()
                    } else {
                        format!("{known:?}")
                    }
                ))
            })
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
        let position = self.header.observation_index(&condition.observation)?;
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
    #[serde(skip_serializing_if = "<[&str]>::is_empty")]
    automata: Vec<&'a str>,
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
    automata: Vec<String>,
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
/// header declares none. Every count is checked before anything is
/// written; a campaign too large to hold writes through a
/// [`RawCorpusWriter`] instead.
///
/// # Errors
/// [`RawSequencesError::Format`] when a count differs from what the header
/// states; [`RawSequencesError::Io`] when the writer fails.
pub fn write_raw_corpus<W: Write>(
    writer: W,
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
    let mut corpus = RawCorpusWriter::new(writer, header)?;
    for (run, sequence) in sequences.iter().enumerate() {
        corpus.push(sequence, observed.get(run).map_or(&[], Vec::as_slice))?;
    }
    corpus.finish()?;
    Ok(())
}

/// A raw corpus written one trajectory at a time, as a campaign produces
/// them: the header goes out on creation, each [`push`](Self::push) writes
/// one line, and [`finish`](Self::finish) checks that the header's count
/// was met. Nothing but the current line is held, so a campaign of any
/// size writes in bounded memory.
#[derive(Debug)]
pub struct RawCorpusWriter<W: Write> {
    writer: W,
    nb_runs: u64,
    width: usize,
    written: u64,
}

impl<W: Write> RawCorpusWriter<W> {
    /// Write `header` and get ready for its trajectories.
    ///
    /// # Errors
    /// [`RawSequencesError::Io`] when the writer fails.
    pub fn new(mut writer: W, header: &RawHeader) -> Result<Self, RawSequencesError> {
        writer.write_all(json_line(header)?.as_bytes())?;
        Ok(RawCorpusWriter {
            writer,
            nb_runs: header.nb_runs,
            width: header.observations.len(),
            written: 0,
        })
    }

    /// Write the next trajectory, in replica order, with its observed
    /// values (one per observation the header declares; none when it
    /// declares none).
    ///
    /// # Errors
    /// [`RawSequencesError::Format`] when the header's count is already
    /// written or `observed` does not hold one value per declared
    /// observation; [`RawSequencesError::Io`] when the writer fails.
    pub fn push(&mut self, sequence: &Sequence, observed: &[f64]) -> Result<(), RawSequencesError> {
        if self.written == self.nb_runs {
            return Err(RawSequencesError::Format(format!(
                "the header states {} trajectories and one more is written",
                self.nb_runs
            )));
        }
        if observed.len() != self.width {
            return Err(RawSequencesError::Format(format!(
                "run {} has {} observed values where the header declares {}",
                self.written,
                observed.len(),
                self.width
            )));
        }
        let record = RawRecord {
            run: self.written,
            end_cause: &sequence.end_cause,
            end_time: sequence.end_time,
            events: sequence
                .events
                .iter()
                .map(|e| (e.time, e.obj.as_str(), e.attr.as_str(), &e.cycle_group))
                .collect(),
            // Written only when some event names its automaton, so a
            // corpus from a source that names none stays byte for byte
            // what earlier engines wrote.
            automata: if sequence.events.iter().any(|e| !e.automaton.is_empty()) {
                sequence
                    .events
                    .iter()
                    .map(|e| e.automaton.as_str())
                    .collect()
            } else {
                Vec::new()
            },
            observed,
        };
        self.writer.write_all(json_line(&record)?.as_bytes())?;
        self.written += 1;
        Ok(())
    }

    /// Flush the corpus and hand the writer back.
    ///
    /// # Errors
    /// [`RawSequencesError::Format`] when fewer trajectories than the
    /// header states were written; [`RawSequencesError::Io`] when the
    /// flush fails.
    pub fn finish(mut self) -> Result<W, RawSequencesError> {
        if self.written != self.nb_runs {
            return Err(RawSequencesError::Format(format!(
                "the header states {} trajectories and {} are written",
                self.nb_runs, self.written
            )));
        }
        self.writer.flush()?;
        Ok(self.writer)
    }
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
/// A corpus too large to hold reads through a [`RawCorpusReader`] instead.
///
/// # Errors
/// As [`RawCorpusReader`].
pub fn read_raw_corpus<R: BufRead>(reader: R) -> Result<RawCorpus, RawSequencesError> {
    let corpus = RawCorpusReader::new(reader)?;
    let header = corpus.header().clone();
    let width = header.observations.len();
    let mut sequences = Vec::new();
    let mut observed = Vec::new();
    for trajectory in corpus {
        let (sequence, row) = trajectory?;
        sequences.push(sequence);
        if width > 0 {
            observed.push(row);
        }
    }
    Ok(RawCorpus {
        header,
        sequences,
        observed,
    })
}

/// A raw corpus read one trajectory at a time: the header is read and
/// checked on creation, then each item is the next trajectory, in replica
/// order, with its observed values (empty when the corpus observed
/// nothing). Only the current line is held, so a corpus of any size reads
/// in bounded memory.
///
/// Every check [`read_raw_corpus`] makes is made as the lines go by; a
/// trajectory count that falls short of the header's is reported as the
/// last item.
///
/// # Errors
/// [`RawSequencesError::Line`] on a line that is not the expected JSON;
/// [`RawSequencesError::Format`] on another format, a version above
/// [`RAW_SEQUENCES_VERSION`], runs out of order, a trajectory count that
/// is not the header's, or a trajectory whose observed values are not one
/// per declared observation.
#[derive(Debug)]
pub struct RawCorpusReader<R: BufRead> {
    lines: std::iter::Enumerate<std::io::Lines<R>>,
    header: RawHeader,
    read: u64,
    done: bool,
}

impl<R: BufRead> RawCorpusReader<R> {
    /// Read and check the header.
    ///
    /// # Errors
    /// As the type says, for the header line.
    pub fn new(reader: R) -> Result<Self, RawSequencesError> {
        let mut lines = reader.lines().enumerate();
        let header: RawHeader = match lines.next() {
            None => return Err(RawSequencesError::Format("the corpus is empty".to_owned())),
            Some((_, line)) => {
                serde_json::from_str(&line?).map_err(|e| RawSequencesError::Line {
                    line: 1,
                    detail: e.to_string(),
                })?
            }
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
        Ok(RawCorpusReader {
            lines,
            header,
            read: 0,
            done: false,
        })
    }

    /// The corpus header.
    #[must_use]
    pub fn header(&self) -> &RawHeader {
        &self.header
    }

    /// One trajectory line.
    fn trajectory(
        &self,
        index: usize,
        line: &str,
    ) -> Result<(Sequence, Vec<f64>), RawSequencesError> {
        let record: ReadRecord =
            serde_json::from_str(line).map_err(|e| RawSequencesError::Line {
                line: index + 1,
                detail: e.to_string(),
            })?;
        if record.run != self.read {
            return Err(RawSequencesError::Format(format!(
                "line {} holds run {} where run {} was expected",
                index + 1,
                record.run,
                self.read
            )));
        }
        if !record.automata.is_empty() && record.automata.len() != record.events.len() {
            return Err(RawSequencesError::Format(format!(
                "line {} names {} automata for {} events",
                index + 1,
                record.automata.len(),
                record.events.len()
            )));
        }
        let mut automata = record.automata.into_iter();
        let events = record
            .events
            .into_iter()
            .map(|(time, obj, attr, cycle_group)| {
                Ok(SeqEvent {
                    obj,
                    // Absent from a corpus whose source named no automaton
                    // (see `SeqEvent::automaton`: empty when it does not say).
                    automaton: automata.next().unwrap_or_default(),
                    attr,
                    time: number(&time, index + 1)?,
                    cycle_group,
                })
            })
            .collect::<Result<Vec<_>, RawSequencesError>>()?;
        let width = self.header.observations.len();
        if record.observed.len() != width {
            return Err(RawSequencesError::Format(format!(
                "line {} holds {} observed values where the header declares {width}",
                index + 1,
                record.observed.len()
            )));
        }
        let observed = record
            .observed
            .iter()
            .map(|value| number(value, index + 1))
            .collect::<Result<Vec<_>, _>>()?;
        let sequence = Sequence {
            events,
            end_cause: record.end_cause,
            end_time: number(&record.end_time, index + 1)?,
            weight: 1.0,
        };
        Ok((sequence, observed))
    }
}

impl<R: BufRead> Iterator for RawCorpusReader<R> {
    type Item = Result<(Sequence, Vec<f64>), RawSequencesError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        loop {
            let Some((index, line)) = self.lines.next() else {
                self.done = true;
                if self.read != self.header.nb_runs {
                    return Some(Err(RawSequencesError::Format(format!(
                        "the header states {} trajectories and the corpus holds {}",
                        self.header.nb_runs, self.read
                    ))));
                }
                return None;
            };
            let line = match line {
                Ok(line) => line,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e.into()));
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            let item = self.trajectory(index, &line);
            match item {
                Ok(_) => self.read += 1,
                Err(_) => self.done = true,
            }
            return Some(item);
        }
    }
}

/// One JSON document on one line.
fn json_line<T: Serialize>(value: &T) -> Result<String, RawSequencesError> {
    let mut text = serde_json::to_string(value)
        .map_err(|e| RawSequencesError::Format(format!("cannot encode a line: {e}")))?;
    text.push('\n');
    Ok(text)
}
