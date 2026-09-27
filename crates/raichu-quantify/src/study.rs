//! The study: the question, stated once for every engine.

use raichu_core::CompiledModel;
use serde::{Deserialize, Serialize};

use crate::QuantifyError;

/// The question a quantification answers, independent of the engine:
/// the probability that `target` is the first declared target reached by
/// `horizon`.
///
/// The same study can be handed to every [`crate::Method`]. The seed and
/// the reporting instants matter to Monte-Carlo simulation only: the
/// exploration methods ignore them, and their envelopes do not record
/// them. The thread count never changes a result, so no envelope records
/// it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Study {
    /// Name of the model's declared target (feared event) quantified.
    pub target: String,
    /// Horizon, in the model's time unit: finite and nonnegative.
    pub horizon: f64,
    /// Reporting instants of the Monte-Carlo estimates (its detailed
    /// result), ascending, within `[0, horizon]`. Default: the horizon
    /// alone.
    #[serde(default)]
    pub instants: Option<Vec<f64>>,
    /// Master seed of a Monte-Carlo campaign (replica `r` uses substream
    /// `r`). Default 0.
    #[serde(default)]
    pub seed: u64,
    /// Worker threads (`None` = the default pool), at least 1. Results do
    /// not depend on it.
    #[serde(default)]
    pub threads: Option<usize>,
}

impl Study {
    /// A study of `target` by `horizon`, reported at the horizon alone,
    /// seed 0 and the default thread count.
    #[must_use]
    pub fn new(target: impl Into<String>, horizon: f64) -> Self {
        Study {
            target: target.into(),
            horizon,
            instants: None,
            seed: 0,
            threads: None,
        }
    }

    /// The reporting instants: the declared ones, or the horizon alone.
    #[must_use]
    pub fn reporting_instants(&self) -> Vec<f64> {
        self.instants.clone().unwrap_or_else(|| vec![self.horizon])
    }

    /// Check the study against the compiled `model`, before any engine
    /// runs.
    ///
    /// # Errors
    /// [`QuantifyError::UnknownTarget`] when the target is not declared by
    /// the model, [`QuantifyError::InvalidStudy`] for a horizon that is
    /// negative or not finite, instants that are not ascending inside
    /// `[0, horizon]`, or 0 threads.
    pub fn validate(&self, model: &CompiledModel) -> Result<(), QuantifyError> {
        if !model.targets.iter().any(|t| t.name == self.target) {
            return Err(QuantifyError::UnknownTarget {
                target: self.target.clone(),
                declared: model.targets.iter().map(|t| t.name.clone()).collect(),
            });
        }
        if !(self.horizon.is_finite() && self.horizon >= 0.0) {
            return Err(invalid(
                "horizon",
                format!(
                    "a horizon is a finite nonnegative time, got {}",
                    self.horizon
                ),
            ));
        }
        if let Some(instants) = &self.instants {
            for (k, &instant) in instants.iter().enumerate() {
                if !(instant.is_finite() && (0.0..=self.horizon).contains(&instant)) {
                    return Err(invalid(
                        "instants",
                        format!(
                            "a reporting instant lies in [0, horizon = {}], got {instant}",
                            self.horizon
                        ),
                    ));
                }
                if k > 0 && instant <= instants[k - 1] {
                    return Err(invalid(
                        "instants",
                        format!(
                            "reporting instants are strictly ascending, got {} then {instant}",
                            instants[k - 1]
                        ),
                    ));
                }
            }
        }
        if self.threads == Some(0) {
            return Err(invalid(
                "threads",
                "a thread count is at least 1".to_owned(),
            ));
        }
        Ok(())
    }
}

fn invalid(parameter: &str, detail: String) -> QuantifyError {
    QuantifyError::InvalidStudy {
        parameter: parameter.to_owned(),
        detail,
    }
}
