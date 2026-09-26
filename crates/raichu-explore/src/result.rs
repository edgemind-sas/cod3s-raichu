//! The result of a sequence-tree exploration, in its own open format:
//! `raichu.exploration`, version 1 for an exact result and version 2 for a
//! discretised one ([`Algorithm::format_version`]).
//!
//! An exploration answers "which sequences lead to the feared event by
//! the horizon, and with what probability", and says how much it left
//! out. Its result is therefore not a Monte-Carlo corpus:
//!
//! - each retained sequence carries a **probability**, not a replica
//!   count, and the retained sequences are **disjoint** events (two
//!   different ordered paths of the embedded jump chain cannot both be
//!   the path a trajectory follows), so their probabilities add up;
//! - the result carries a **lower bound** (the sum of the retained
//!   probabilities) and an **upper bound** (the lower bound plus the
//!   mass of every sub-tree a cut-off interrupted), the mass each
//!   cut-off discarded, and an inconclusive flag when the bounds stay
//!   far apart.
//!
//! Each distinct step (a transition fired into one destination) is written
//! once, in the result's `steps` table; a sequence lists indices into it,
//! so the size of a result is the sum of the retained sequence lengths as
//! small integers, plus one table entry per distinct step.
//!
//! The events of a sequence use the vocabulary of the Monte-Carlo
//! sequence corpus (`obj`, the monitored state entered as `attr`, and the
//! cycle group), so [`ExplorationResult::to_sequences`] hands them to the
//! existing reduction ([`raichu_core::sequence::analyse`]).
//!
//! The format is plain serde data. A reader refuses another `format` and a
//! `version` above the one it knows; a later version may add fields, which
//! an earlier reader ignores. Version 2 added the `discretised` algorithm
//! and its `discretisation` field: an exact result is still written as
//! version 1, byte for byte as before, and a version-1 reader refuses a
//! discretised document by its version rather than by an unknown
//! algorithm name. A document declaring an algorithm that its version
//! predates (a `discretised` result at version 1) is refused as
//! inconsistent ([`ReadExplorationError::AlgorithmVersion`]).

use raichu_core::{SeqEvent, Sequence};
use serde::{Deserialize, Serialize};

/// The `format` of an exploration result.
pub const EXPLORATION_FORMAT: &str = "raichu.exploration";

/// The highest format version this crate reads, and the one it writes for
/// the most recent algorithm. Each result is written at its algorithm's
/// own version ([`Algorithm::format_version`]): 1 for [`Algorithm::Exact`],
/// 2 for [`Algorithm::Discretised`].
pub const EXPLORATION_VERSION: u32 = 2;

/// Default relative gap `(upper - lower) / upper` above which a result is
/// flagged inconclusive.
pub const DEFAULT_GAP_TOLERANCE: f64 = 0.01;

/// The exploration algorithm that produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Algorithm {
    /// Exact exploration of the embedded jump chain of a Markov model:
    /// every sequence probability is computed in closed form (path
    /// probability times the probability that its sojourns end by the
    /// horizon).
    Exact,
    /// Discretised exploration: the distribution of the random next event
    /// (which armed transition fires first, and when) is cut into cells of
    /// equal probability mass, each cell a branch fired at its
    /// mass-median instant. Covers every law the engine carries and
    /// continuous evolution; the bounds are bounds on the discretised
    /// model, and the result states the discretisation level and an
    /// estimate of the discretisation error ([`Discretisation`]).
    Discretised,
}

impl Algorithm {
    /// The format version a result of this algorithm is written at, which
    /// is also the first version that knows the algorithm: 1 for
    /// [`Algorithm::Exact`] (so an exact document is unchanged since
    /// version 1), 2 for [`Algorithm::Discretised`].
    #[must_use]
    pub fn format_version(self) -> u32 {
        match self {
            Algorithm::Exact => 1,
            Algorithm::Discretised => 2,
        }
    }
}

impl std::str::FromStr for Algorithm {
    type Err = String;

    /// Parse an algorithm by its serialized name (`exact`,
    /// `discretised`). The accepted
    /// names are the serde spellings, so they cannot drift from the
    /// result format; an unknown name is refused with the list of known
    /// ones.
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        use serde::de::IntoDeserializer;
        Algorithm::deserialize(IntoDeserializer::<serde::de::value::Error>::into_deserializer(name))
            .map_err(|e| e.to_string())
    }
}

/// Why a document is not an exploration result this crate reads.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ReadExplorationError {
    /// The document is not valid JSON, or lacks a field of the format.
    #[error("invalid exploration JSON: {0}")]
    Json(String),
    /// The document declares another format (or none).
    #[error("not an exploration result: format {0:?}, expected `raichu.exploration`")]
    Format(Option<String>),
    /// The document declares a version above [`EXPLORATION_VERSION`] (or
    /// none).
    #[error(
        "exploration format version {0:?} is not readable by this engine (reads up to version {max})",
        max = EXPLORATION_VERSION
    )]
    Version(Option<u32>),
    /// The document declares an algorithm introduced by a later format
    /// version than the one it declares (a `discretised` result at
    /// version 1): it was not written by this format's writer.
    #[error(
        "inconsistent exploration document: algorithm `{algorithm:?}` needs format version \
         {required} or above, but the document declares version {version}"
    )]
    AlgorithmVersion {
        /// The declared algorithm.
        algorithm: Algorithm,
        /// The declared version.
        version: u32,
        /// The first version that knows the algorithm.
        required: u32,
    },
    /// A sequence refers to a step outside the result's step table.
    #[error("sequence {sequence} refers to step {step}, but the step table holds {table} steps")]
    DanglingStep {
        /// Index of the sequence.
        sequence: usize,
        /// The step index it holds.
        step: u32,
        /// Length of the step table.
        table: usize,
    },
}

/// Read an exploration result from its JSON document, refusing another
/// `format`, a `version` above [`EXPLORATION_VERSION`], an algorithm its
/// declared version predates ([`Algorithm::format_version`]), and a
/// sequence referring to a step outside the step table. The envelope
/// is read first, so a document of another format is refused by name
/// rather than by the first field it happens to lack.
pub fn read_exploration(json: &str) -> Result<ExplorationResult, ReadExplorationError> {
    #[derive(Deserialize)]
    struct Envelope {
        format: Option<String>,
        version: Option<u32>,
    }
    let envelope: Envelope =
        serde_json::from_str(json).map_err(|e| ReadExplorationError::Json(e.to_string()))?;
    if envelope.format.as_deref() != Some(EXPLORATION_FORMAT) {
        return Err(ReadExplorationError::Format(envelope.format));
    }
    match envelope.version {
        Some(version) if version <= EXPLORATION_VERSION => {}
        other => return Err(ReadExplorationError::Version(other)),
    }
    let result: ExplorationResult =
        serde_json::from_str(json).map_err(|e| ReadExplorationError::Json(e.to_string()))?;
    let required = result.algorithm.format_version();
    if result.version < required {
        return Err(ReadExplorationError::AlgorithmVersion {
            algorithm: result.algorithm,
            version: result.version,
            required,
        });
    }
    for (sequence, explored) in result.sequences.iter().enumerate() {
        if let Some(&step) = explored
            .steps
            .iter()
            .find(|&&id| id as usize >= result.steps.len())
        {
            return Err(ReadExplorationError::DanglingStep {
                sequence,
                step,
                table: result.steps.len(),
            });
        }
    }
    Ok(result)
}

/// The declared cut-offs of an exploration. Each one is optional; an
/// exploration with none of them explores until every branch ends
/// (target reached, absorbing state, or a prefix probability of exactly
/// zero), which on a repairable model may not happen in practice.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Cutoffs {
    /// Minimal sequence probability (dimensionless, in `(0, 1]`). A node
    /// is pruned when the probability that its prefix is completed by
    /// the horizon falls below it. That quantity bounds the probability
    /// of every sequence extending the prefix, so adding it to the upper
    /// bound is sound.
    pub min_probability: Option<f64>,
    /// Maximal sequence length, counted in fired transitions (every
    /// firing counts, instantaneous ones included).
    pub max_length: Option<usize>,
    /// Maximal number of failures along a sequence: fired transitions of
    /// declared kind `failure` entering their **first** declared target.
    /// Refused on a model that declares no transition kind.
    pub max_failures: Option<u64>,
    /// Maximal number of expanded nodes (nodes whose children are
    /// explored), at least 1. Divided among the root's children before
    /// the exploration starts, so the explored set does not depend on
    /// the thread count.
    pub max_branches: Option<u64>,
}

/// Numerical precision of the sequence probabilities (see
/// [`raichu_numeric::PhaseTypeSettings`], which it mirrors in a
/// serializable form).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Precision {
    /// Requested relative precision of every sequence probability, in
    /// `(0, 1)`. A probability whose error bound exceeds it is flagged
    /// imprecise, never reported silently.
    pub rel_precision: f64,
    /// Cap on the uniformization terms before the computation falls back
    /// to scaling and squaring, at least 1.
    pub max_terms: usize,
}

impl Default for Precision {
    fn default() -> Self {
        let settings = raichu_numeric::PhaseTypeSettings::default();
        Precision {
            rel_precision: settings.rel_precision,
            max_terms: settings.max_terms,
        }
    }
}

impl From<&Precision> for raichu_numeric::PhaseTypeSettings {
    fn from(precision: &Precision) -> Self {
        raichu_numeric::PhaseTypeSettings {
            rel_precision: precision.rel_precision,
            max_terms: precision.max_terms,
        }
    }
}

/// What one cut-off discarded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CutoffTally {
    /// Number of nodes this cut-off pruned.
    pub pruned_nodes: u64,
    /// Sum of the prefix probabilities of those nodes (dimensionless):
    /// the mass this cut-off added to the upper bound.
    pub mass: f64,
}

impl CutoffTally {
    /// Whether this cut-off pruned anything.
    #[must_use]
    pub fn fired(&self) -> bool {
        self.pruned_nodes > 0
    }

    pub(crate) fn add(&mut self, mass: f64) {
        self.pruned_nodes += 1;
        self.mass += mass;
    }

    pub(crate) fn merge(&mut self, other: &CutoffTally) {
        self.pruned_nodes += other.pruned_nodes;
        self.mass += other.mass;
    }
}

/// What each cut-off discarded. When several cut-offs would prune the
/// same node, it is tallied once, under the first of: minimal
/// probability, maximal length, maximal failures (the branch cap is
/// checked separately, when a node is about to be expanded).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CutoffTallies {
    /// The minimal-probability cut-off.
    pub min_probability: CutoffTally,
    /// The maximal-length cut-off.
    pub max_length: CutoffTally,
    /// The maximal-failures cut-off.
    pub max_failures: CutoffTally,
    /// The branch cap.
    pub max_branches: CutoffTally,
}

impl CutoffTallies {
    /// Total mass discarded by every cut-off, summed in the fixed order
    /// of the fields: the difference between the upper and the lower
    /// bound.
    #[must_use]
    pub fn total_mass(&self) -> f64 {
        self.min_probability.mass
            + self.max_length.mass
            + self.max_failures.mass
            + self.max_branches.mass
    }

    pub(crate) fn merge(&mut self, other: &CutoffTallies) {
        self.min_probability.merge(&other.min_probability);
        self.max_length.merge(&other.max_length);
        self.max_failures.merge(&other.max_failures);
        self.max_branches.merge(&other.max_branches);
    }
}

/// One distinct step of an exploration: a transition fired into one of its
/// destinations. The result holds each distinct step once, in its
/// [`ExplorationResult::steps`] table, and every sequence lists indices
/// into that table, so a result grows with the number of fired
/// transitions as small integers rather than as repeated names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExploredStep {
    /// Qualified transition name (`component.automaton.transition`).
    pub transition: String,
    /// Source state.
    pub from: String,
    /// Destination state (the branch taken).
    pub to: String,
    /// The monitored-state entry this step produces, `None` when the
    /// transition is not monitored.
    pub event: Option<ExploredEvent>,
}

/// One monitored-state entry of an explored sequence, in the vocabulary
/// of the Monte-Carlo sequence corpus ([`SeqEvent`]) minus the date: an
/// exact exploration chooses the order of the jumps, not their dates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExploredEvent {
    /// Owning component.
    pub obj: String,
    /// The monitored state entered.
    pub attr: String,
    /// Cycle-pair group of the firing transition, which the cycle filter
    /// of the reduction reads.
    pub cycle_group: Option<String>,
}

impl ExploredEvent {
    /// The corpus event, dated 0: the exact driver never moves the clock,
    /// and a discretised sequence merges many cells whose representative
    /// instants differ, so no date either could give would mean anything.
    #[must_use]
    pub fn to_seq_event(&self) -> SeqEvent {
        SeqEvent {
            obj: self.obj.clone(),
            attr: self.attr.clone(),
            time: 0.0,
            cycle_group: self.cycle_group.clone(),
        }
    }
}

/// One retained sequence: an ordered path from the initial state to the
/// target in which nothing else happens. This is not a minimal cut
/// sequence, which is a reduction over many such paths.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExploredSequence {
    /// Every fired transition, in firing order, as indices into
    /// [`ExplorationResult::steps`]. Resolve them with
    /// [`ExplorationResult::sequence_steps`] and
    /// [`ExplorationResult::sequence_events`].
    pub steps: Vec<u32>,
    /// The target reached.
    pub end_cause: String,
    /// Probability (dimensionless) that a trajectory follows exactly this
    /// path and reaches the target by the horizon.
    pub probability: f64,
    /// Bound on the absolute error of `probability`.
    pub error_bound: f64,
    /// `true` when `error_bound` exceeds the declared relative precision
    /// times `probability`: the value is not guaranteed to that
    /// precision.
    pub imprecise: bool,
}

/// The result of a sequence-tree exploration (`raichu.exploration`, v1 for
/// an exact result, v2 for a discretised one).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExplorationResult {
    /// Always [`EXPLORATION_FORMAT`].
    pub format: String,
    /// The format version: [`Algorithm::format_version`] of `algorithm`
    /// when written here.
    pub version: u32,
    /// The engine version that explored.
    pub engine_version: String,
    /// The model's name.
    pub model: String,
    /// The algorithm.
    pub algorithm: Algorithm,
    /// The target (feared event) explored to.
    pub target: String,
    /// The horizon, in the model's time unit.
    pub horizon: f64,
    /// The declared cut-offs.
    pub cutoffs: Cutoffs,
    /// The declared relative gap tolerance.
    pub gap_tolerance: f64,
    /// The declared numerical precision.
    pub precision: Precision,
    /// Every distinct step (transition and destination) occurring in a
    /// retained sequence, each once, ordered by the transition's index in
    /// the compiled model, then by destination: the order does not depend
    /// on the thread count.
    pub steps: Vec<ExploredStep>,
    /// Retained sequences, by decreasing probability; ties are broken by
    /// their ordered list of resolved steps.
    pub sequences: Vec<ExploredSequence>,
    /// Lower bound on the probability of reaching the target by the
    /// horizon: the sum of the retained probabilities, in exploration
    /// order.
    pub lower: f64,
    /// Upper bound: `lower` plus the mass every cut-off discarded.
    pub upper: f64,
    /// What each cut-off discarded.
    pub cutoff_tallies: CutoffTallies,
    /// `true` when `(upper - lower) / upper` exceeds `gap_tolerance`.
    pub inconclusive: bool,
    /// Number of expanded nodes.
    pub expanded_nodes: u64,
    /// Number of retained sequences flagged imprecise.
    pub imprecise_sequences: usize,
    /// The discretisation of a [`Algorithm::Discretised`] result: its
    /// level and its error estimate. Absent from an exact result (and from
    /// its document, which therefore reads back unchanged).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discretisation: Option<Discretisation>,
}

/// The discretisation a [`Algorithm::Discretised`] result was computed
/// with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Discretisation {
    /// Number of equal-mass cells the next-event distribution is cut into
    /// at every timed node, for the reported numbers (the refined level
    /// `2K` when a refinement was made, `K` otherwise).
    pub level: u32,
    /// The error estimate by refinement, `None` when refinement was
    /// switched off: then **no estimate was made**, and the reported
    /// numbers carry no statement about the discretisation error.
    pub refinement: Option<Refinement>,
}

/// The discretisation error estimate by refinement: the exploration run
/// at the base level `K` and again at `2K`, the reported result being
/// the `2K` one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Refinement {
    /// The base level `K`.
    pub base_level: u32,
    /// Lower bound of the base run (dimensionless probability).
    pub base_lower: f64,
    /// Upper bound of the base run (dimensionless probability).
    pub base_upper: f64,
    /// `max(|lower_2K - lower_K|, |upper_2K - upper_K|)`: an
    /// **estimate** of the discretisation error of the base run, which
    /// also bounds the reported (finer) run's error when the scheme
    /// converges. It is an estimate, not a bound.
    pub error_estimate: f64,
    /// `true` when the two runs' gaps `upper - lower` add up to half of
    /// `error_estimate` or more (in particular whenever either gap exceeds
    /// it): the difference between the two runs may then reflect what
    /// each one truncated as much as the discretisation, since what a run
    /// truncated from its lower bound is at most its gap.
    pub truncation_dominated: bool,
}

impl ExplorationResult {
    /// The relative gap `(upper - lower) / upper`, 0 when `upper` is 0.
    #[must_use]
    pub fn relative_gap(&self) -> f64 {
        relative_gap(self.lower, self.upper)
    }

    /// The discretisation error estimate by refinement, `None` for an
    /// exact result and for a discretised one whose refinement was
    /// switched off.
    #[must_use]
    pub fn error_estimate(&self) -> Option<f64> {
        self.discretisation
            .as_ref()
            .and_then(|d| d.refinement.as_ref())
            .map(|r| r.error_estimate)
    }

    /// The retained sequences as corpus [`Sequence`]s, each weighted by
    /// its probability, dated 0, and ending at the target.
    ///
    /// **Valid input for the minimal-sequence reduction**
    /// ([`raichu_core::sequence::analyse`]): the retained sequences are
    /// disjoint events, so summing their probabilities when sequences are
    /// grouped or absorbed yields the probability of the union, and the
    /// total weight is [`ExplorationResult::lower`].
    ///
    /// **Not a valid input for date-based post-processing**, such as
    /// analyses that read a free-running, dated corpus: an exploration
    /// produces neither dates (the exact clock never moves, and a
    /// discretised sequence merges cells fired at different instants) nor
    /// the trajectories that miss the target.
    #[must_use]
    pub fn to_sequences(&self) -> Vec<Sequence> {
        self.sequences
            .iter()
            .map(|sequence| Sequence {
                events: sequence
                    .steps
                    .iter()
                    .filter_map(|&id| self.steps.get(id as usize))
                    .filter_map(|step| step.event.as_ref())
                    .map(ExploredEvent::to_seq_event)
                    .collect(),
                end_cause: Some(sequence.end_cause.clone()),
                end_time: 0.0,
                weight: sequence.probability,
            })
            .collect()
    }

    /// The fired transitions of sequence `i`, in firing order, resolved
    /// through the [`ExplorationResult::steps`] table; `None` when `i` is
    /// not a sequence index. A step index outside the table, which
    /// [`read_exploration`] refuses and the explorer never writes, is
    /// skipped rather than trusted.
    pub fn sequence_steps(&self, i: usize) -> Option<impl Iterator<Item = &ExploredStep> + '_> {
        let sequence = self.sequences.get(i)?;
        Some(
            sequence
                .steps
                .iter()
                .filter_map(|&id| self.steps.get(id as usize)),
        )
    }

    /// The monitored-state entries of sequence `i`, in firing order: the
    /// events of its monitored steps; `None` when `i` is not a sequence
    /// index.
    pub fn sequence_events(&self, i: usize) -> Option<impl Iterator<Item = &ExploredEvent> + '_> {
        Some(
            self.sequence_steps(i)?
                .filter_map(|step| step.event.as_ref()),
        )
    }
}

/// `(upper - lower) / upper`, 0 when `upper` is 0.
pub(crate) fn relative_gap(lower: f64, upper: f64) -> f64 {
    if upper > 0.0 {
        (upper - lower) / upper
    } else {
        0.0
    }
}
