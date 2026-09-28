//! The result envelope, `raichu.quantification` version 2 (version 1 is
//! still read).

use raichu_explore::{Algorithm, ExplorationResult};
use raichu_montecarlo::{CrossEntropyEstimate, IntervalMethod, McEstimates};
use serde::{Deserialize, Serialize};

use crate::Method;

/// The `format` of a quantification envelope.
pub const QUANTIFICATION_FORMAT: &str = "raichu.quantification";

/// The format version this crate writes and the highest it reads.
///
/// Version 2 added the cross-entropy method, its `weighted_estimate`
/// probability and its `cross_entropy` detail; a version 1 envelope reads
/// unchanged.
pub const QUANTIFICATION_VERSION: u32 = 2;

/// The answer to a study, whatever the engine: `raichu.quantification`
/// version 2.
///
/// The envelope carries the method as applied, the provenance, the
/// probability of the study's target with its uncertainty, and the
/// engine's own detailed result unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quantification {
    /// Always [`QUANTIFICATION_FORMAT`].
    pub format: String,
    /// The format version, [`QUANTIFICATION_VERSION`] when written here.
    pub version: u32,
    /// The method and the settings it applied (defaults resolved).
    pub method: Method,
    /// Where the answer comes from.
    pub provenance: QuantificationProvenance,
    /// The probability that the study's target is the first declared
    /// target reached by the horizon, with its uncertainty.
    pub probability: TargetProbability,
    /// The engine's detailed result, unchanged.
    pub detail: Detail,
}

impl Quantification {
    /// The envelope as compact JSON.
    ///
    /// # Errors
    /// Serialization errors from `serde_json`.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// The provenance of a [`Quantification`].
///
/// The thread count is never recorded: no result depends on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantificationProvenance {
    /// The engine version that quantified.
    pub engine_version: String,
    /// The model's name.
    pub model: String,
    /// The content hash of the sealed model document,
    /// `sha256:<64 hex digits>` (see [`crate::model_content_hash`]).
    /// Comparable only between envelopes of the same `engine_version`.
    pub model_hash: String,
    /// The study's target (feared event).
    pub target: String,
    /// The study's horizon, in the model's time unit.
    pub horizon: f64,
    /// The reporting instants, recorded only for a method that uses them
    /// (Monte-Carlo simulation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instants: Option<Vec<f64>>,
    /// The master seed, recorded only for a method that draws at random
    /// (Monte-Carlo simulation, cross-entropy).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// The probability of the study's target, typed by the kind of
/// uncertainty the engine can state. Serialized with a `kind` member.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TargetProbability {
    /// A statistical estimate with its confidence interval (Monte-Carlo
    /// simulation): `reached` of `replicas` reached the target first by
    /// the horizon.
    ConfidenceInterval {
        /// The point estimate `reached / replicas`.
        estimate: f64,
        /// Replicas whose first target reached was the study's, by the
        /// horizon.
        reached: u64,
        /// Replicas run.
        replicas: u64,
        /// Confidence level of the interval, strictly inside `(0, 1)`.
        level: f64,
        /// The construction: [`IntervalMethod::Wilson`], the estimate
        /// being a binomial proportion.
        method: IntervalMethod,
        /// Lower bound of the interval.
        low: f64,
        /// Upper bound of the interval.
        high: f64,
    },
    /// A weighted estimate with its confidence interval (cross-entropy):
    /// the mean over `replicas` of each replica's likelihood ratio when it
    /// reached the target first, zero otherwise.
    WeightedEstimate {
        /// The point estimate.
        estimate: f64,
        /// Its standard error.
        standard_error: f64,
        /// Replicas that reached the target first (unweighted).
        reached: u64,
        /// Replicas run in the final campaign.
        replicas: u64,
        /// Confidence level of the interval, strictly inside `(0, 1)`.
        level: f64,
        /// The construction: [`IntervalMethod::WeightedNormal`].
        method: IntervalMethod,
        /// Lower bound of the interval.
        low: f64,
        /// Upper bound of the interval.
        high: f64,
        /// Effective sample size of the hit weights: how many equally
        /// weighted hits the estimate is worth. A value near 1 means one
        /// replica carries the estimate.
        effective_sample_size: f64,
        /// Standard error over the estimate.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        relative_error: Option<f64>,
        /// Whether the effective sample size is below the method's
        /// threshold: the estimate and its interval are then not to be
        /// trusted, however narrow the interval.
        inconclusive: bool,
    },
    /// Guaranteed bounds (exact and discretised exploration): the
    /// probability lies in `[lower, upper]`, on the discretised model for
    /// a discretised exploration.
    Bounds {
        /// Lower bound: the sum of the retained sequence probabilities.
        lower: f64,
        /// Upper bound: the lower bound plus the mass the cut-offs
        /// discarded.
        upper: f64,
        /// Whether the relative gap exceeds the declared tolerance.
        inconclusive: bool,
        /// The discretisation error estimate by refinement: present for a
        /// discretised exploration that refined, absent otherwise. An
        /// estimate, not a bound.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_estimate: Option<f64>,
    },
}

impl TargetProbability {
    /// The interval `[low, high]` the probability is stated in: the
    /// confidence interval, or the bounds.
    #[must_use]
    pub fn range(&self) -> (f64, f64) {
        match self {
            TargetProbability::ConfidenceInterval { low, high, .. }
            | TargetProbability::WeightedEstimate { low, high, .. } => (*low, *high),
            TargetProbability::Bounds { lower, upper, .. } => (*lower, *upper),
        }
    }
}

/// The engine's detailed result, unchanged. Serialized with the engine
/// kind as its single key (`{"monte_carlo": {...}}`,
/// `{"exploration": {...}}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Detail {
    /// The Monte-Carlo estimates of the stop-at-targets campaign: equal
    /// to `raichu_montecarlo::run` with `stop_at_targets` on and the same
    /// seed, replicas, level, quantiles, horizon and instants.
    MonteCarlo(Box<McEstimates>),
    /// The exploration result (`raichu.exploration`).
    Exploration(Box<ExplorationResult>),
    /// The cross-entropy campaign: fitted factors per family, pilot
    /// history and diagnostics, without the per-replica ends.
    CrossEntropy(Box<CrossEntropyEstimate>),
}

/// Why a document is not a quantification envelope this crate reads.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ReadQuantificationError {
    /// The document is not valid JSON, or lacks a member of the format.
    #[error("invalid quantification JSON: {0}")]
    Json(String),
    /// The document declares another format (or none).
    #[error("not a quantification envelope: format {0:?}, expected `raichu.quantification`")]
    Format(Option<String>),
    /// The document declares a version above [`QUANTIFICATION_VERSION`]
    /// (or none).
    #[error(
        "quantification format version {0:?} is not readable by this engine (reads up to version {max})",
        max = QUANTIFICATION_VERSION
    )]
    Version(Option<u32>),
    /// The method, the kind of probability and the detail do not belong
    /// together (an exploration detail under a Monte-Carlo method, for
    /// instance): the document was not written by this format's writer.
    #[error("inconsistent quantification envelope: {0}")]
    Inconsistent(String),
}

/// Read a quantification envelope from its JSON document, refusing
/// another `format`, a `version` above [`QUANTIFICATION_VERSION`], and a
/// document whose method, probability kind and detail do not belong
/// together. The format and version are read first, so a document of
/// another format is refused by name rather than by the first member it
/// happens to lack. The detailed result of an exploration is checked as
/// [`raichu_explore::read_exploration`] checks it. Numbers are read
/// correctly rounded, so a written envelope reads back to the same bits.
///
/// # Errors
/// [`ReadQuantificationError`].
pub fn read_quantification(json: &str) -> Result<Quantification, ReadQuantificationError> {
    let document = crate::exact_json::parse(json).map_err(ReadQuantificationError::Json)?;
    let format = document
        .get("format")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    if format.as_deref() != Some(QUANTIFICATION_FORMAT) {
        return Err(ReadQuantificationError::Format(format));
    }
    let version = document
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok());
    match version {
        Some(version) if (1..=QUANTIFICATION_VERSION).contains(&version) => {}
        other => return Err(ReadQuantificationError::Version(other)),
    }
    let envelope: Quantification = serde_json::from_value(document)
        .map_err(|e| ReadQuantificationError::Json(e.to_string()))?;
    check_consistency(&envelope)?;
    if let Detail::Exploration(result) = &envelope.detail {
        // Re-read through the exploration reader: its own format, version
        // and step-table checks apply to the detail as to a standalone
        // document.
        let text = serde_json::to_string(result)
            .map_err(|e| ReadQuantificationError::Json(e.to_string()))?;
        raichu_explore::read_exploration(&text)
            .map_err(|e| ReadQuantificationError::Inconsistent(e.to_string()))?;
    }
    Ok(envelope)
}

fn check_consistency(envelope: &Quantification) -> Result<(), ReadQuantificationError> {
    // One arm per method, without a catch-all: a new method does not
    // compile until its consistency rule is written here.
    match &envelope.method {
        Method::MonteCarlo(settings) => {
            let (
                TargetProbability::ConfidenceInterval {
                    estimate,
                    reached,
                    replicas,
                    level,
                    ..
                },
                Detail::MonteCarlo(detail),
            ) = (&envelope.probability, &envelope.detail)
            else {
                return inconsistent(
                    "monte_carlo",
                    "expects a confidence interval and a Monte-Carlo detail",
                );
            };
            if *replicas != settings.nb_runs || detail.nb_runs != settings.nb_runs {
                return inconsistent("monte_carlo", "the replica counts disagree");
            }
            if !same_bits(*level, settings.confidence)
                || !same_bits(detail.confidence, settings.confidence)
            {
                return inconsistent("monte_carlo", "the confidence levels disagree");
            }
            if *reached > *replicas || !same_bits(*estimate, *reached as f64 / *replicas as f64) {
                return inconsistent("monte_carlo", "the estimate is not `reached / replicas`");
            }
            if envelope.provenance.seed != Some(detail.seed) {
                return inconsistent("monte_carlo", "the seeds disagree");
            }
            Ok(())
        }
        Method::CrossEntropy(settings) => {
            let (
                TargetProbability::WeightedEstimate {
                    estimate,
                    standard_error,
                    reached,
                    replicas,
                    level,
                    method,
                    low,
                    high,
                    effective_sample_size,
                    relative_error,
                    inconclusive,
                },
                Detail::CrossEntropy(detail),
            ) = (&envelope.probability, &envelope.detail)
            else {
                return inconsistent(
                    "cross_entropy",
                    "expects a weighted estimate and a cross-entropy detail",
                );
            };
            let d = &detail.estimate;
            if *replicas != settings.nb_runs || detail.replicas != settings.nb_runs {
                return inconsistent("cross_entropy", "the replica counts disagree");
            }
            if !same_bits(*level, settings.confidence) || !same_bits(d.level, settings.confidence) {
                return inconsistent("cross_entropy", "the confidence levels disagree");
            }
            let same_relative = match (relative_error, d.relative_error) {
                (None, None) => true,
                (Some(a), Some(b)) => same_bits(*a, b),
                _ => false,
            };
            if *reached != detail.reached
                || *reached > *replicas
                || *method != d.method
                || !same_bits(*estimate, d.estimate)
                || !same_bits(*standard_error, d.standard_error)
                || !same_bits(*low, d.low)
                || !same_bits(*high, d.high)
                || !same_bits(*effective_sample_size, d.effective_sample_size)
                || !same_relative
                || *inconclusive != detail.estimate_inconclusive
            {
                return inconsistent("cross_entropy", "the estimate is not the detail's");
            }
            if envelope.version < 2 {
                return inconsistent("cross_entropy", "the method exists from format version 2");
            }
            if detail.target != envelope.provenance.target || envelope.provenance.seed.is_none() {
                return inconsistent("cross_entropy", "the target or seed is not the detail's");
            }
            Ok(())
        }
        Method::Exact(_) => check_bounds(envelope, Algorithm::Exact),
        Method::Discretised(_) => check_bounds(envelope, Algorithm::Discretised),
    }
}

/// The bounds of an exploration envelope must be its detail's own.
fn check_bounds(
    envelope: &Quantification,
    algorithm: Algorithm,
) -> Result<(), ReadQuantificationError> {
    let method = envelope.method.name();
    let (
        TargetProbability::Bounds {
            lower,
            upper,
            inconclusive,
            error_estimate,
        },
        Detail::Exploration(result),
    ) = (&envelope.probability, &envelope.detail)
    else {
        return inconsistent(method, "expects bounds and an exploration detail");
    };
    if result.algorithm != algorithm {
        return inconsistent(method, "the detail was computed by another algorithm");
    }
    let same_estimate = match (error_estimate, result.error_estimate()) {
        (None, None) => true,
        (Some(a), Some(b)) => same_bits(*a, b),
        _ => false,
    };
    if !same_bits(*lower, result.lower)
        || !same_bits(*upper, result.upper)
        || *inconclusive != result.inconclusive
        || !same_estimate
    {
        return inconsistent(method, "the bounds are not the detail's");
    }
    if result.target != envelope.provenance.target
        || !same_bits(result.horizon, envelope.provenance.horizon)
    {
        return inconsistent(method, "the target or horizon is not the detail's");
    }
    Ok(())
}

/// Two numbers are the same when their bits are: the envelope is read
/// correctly rounded, so a value copied from the detail is bit-identical.
fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn inconsistent(method: &str, what: &str) -> Result<(), ReadQuantificationError> {
    Err(ReadQuantificationError::Inconsistent(format!(
        "method `{method}`: {what}"
    )))
}
