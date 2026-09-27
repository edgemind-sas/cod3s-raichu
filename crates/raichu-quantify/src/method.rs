//! The methods of quantification and the settings that belong to each.

use raichu_explore::{
    Cutoffs, DiscretisedSettings, ExactSettings, Precision, DEFAULT_GAP_TOLERANCE, DEFAULT_LEVEL,
    DEFAULT_MAX_BRANCHES,
};
use raichu_montecarlo::DEFAULT_CONFIDENCE;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{QuantifyError, Study};

/// A method of quantification: the engine that answers the study, with
/// the settings that belong to it alone (what every engine shares is in
/// the [`Study`]).
///
/// Serialized as `{"name": <method>, "settings": {...}}`, the names being
/// `monte_carlo`, `exact` and `discretised`. Non-exhaustive: a later
/// engine adds a variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "name", content = "settings", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Method {
    /// Monte-Carlo simulation (crate `raichu-montecarlo`).
    MonteCarlo(MonteCarloSettings),
    /// Exact exploration of the sequence tree of a Markov model (crate
    /// `raichu-explore`).
    Exact(ExactExplorationSettings),
    /// Discretised exploration of the sequence tree (crate
    /// `raichu-explore`).
    Discretised(DiscretisedExplorationSettings),
}

impl Method {
    /// The method names this engine provides, in the order the
    /// documentation presents them.
    pub const NAMES: [&'static str; 3] = ["monte_carlo", "exact", "discretised"];

    /// The method's name, as serialized.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Method::MonteCarlo(_) => "monte_carlo",
            Method::Exact(_) => "exact",
            Method::Discretised(_) => "discretised",
        }
    }

    /// The setting names a method takes, `None` for a name this engine
    /// does not provide.
    #[must_use]
    pub fn setting_names(name: &str) -> Option<&'static [&'static str]> {
        match name {
            "monte_carlo" => Some(MonteCarloSettings::NAMES),
            "exact" => Some(ExactExplorationSettings::NAMES),
            "discretised" => Some(DiscretisedExplorationSettings::NAMES),
            _ => None,
        }
    }

    /// Build a method from its name and its settings object, refusing
    /// what does not belong: the entry point of a binding, which receives
    /// the method as text.
    ///
    /// `settings` is a JSON object (or `null`, for every default); a
    /// setting left out takes its default.
    ///
    /// # Errors
    /// [`QuantifyError::UnknownMethod`] naming the methods provided,
    /// [`QuantifyError::SettingNotApplicable`] for a setting of another
    /// method (naming the methods that take it), and
    /// [`QuantifyError::InvalidSettings`] for settings that are not an
    /// object, a value of the wrong type, or a required setting left out.
    pub fn from_parts(name: &str, settings: &Value) -> Result<Method, QuantifyError> {
        let Some(accepted) = Method::setting_names(name) else {
            return Err(QuantifyError::UnknownMethod {
                name: name.to_owned(),
                known: Method::NAMES.iter().map(|&n| n.to_owned()).collect(),
            });
        };
        let object = match settings {
            Value::Null => serde_json::Map::new(),
            Value::Object(object) => object.clone(),
            other => {
                return Err(QuantifyError::InvalidSettings {
                    method: name.to_owned(),
                    detail: format!("settings are a JSON object, got {other}"),
                })
            }
        };
        if let Some(setting) = object.keys().find(|key| !accepted.contains(&key.as_str())) {
            return Err(QuantifyError::SettingNotApplicable {
                method: name.to_owned(),
                setting: setting.clone(),
                applies_to: Method::NAMES
                    .iter()
                    .filter(|&&other| {
                        Method::setting_names(other)
                            .is_some_and(|names| names.contains(&setting.as_str()))
                    })
                    .map(|&other| other.to_owned())
                    .collect(),
            });
        }
        let tagged = serde_json::json!({ "name": name, "settings": Value::Object(object) });
        serde_json::from_value(tagged).map_err(|e| QuantifyError::InvalidSettings {
            method: name.to_owned(),
            detail: e.to_string(),
        })
    }
}

/// Settings of a Monte-Carlo quantification.
///
/// The campaign always stops each trajectory at the first declared
/// target reached (that is what "first target reached by the horizon"
/// counts); its horizon is the study's, its sampling instants the study's
/// reporting instants, and the continuous solver and flow policy are the
/// engine defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonteCarloSettings {
    /// Number of replicas, at least 1.
    pub nb_runs: u64,
    /// Confidence level of every interval, strictly inside `(0, 1)`.
    /// Default [`DEFAULT_CONFIDENCE`].
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    /// Quantile orders estimated in the detailed result. Default none.
    #[serde(default)]
    pub quantiles: Vec<f64>,
}

fn default_confidence() -> f64 {
    DEFAULT_CONFIDENCE
}

impl MonteCarloSettings {
    /// The setting names, as serialized.
    pub const NAMES: &'static [&'static str] = &["nb_runs", "confidence", "quantiles"];

    /// `nb_runs` replicas at the default confidence level, no quantile.
    #[must_use]
    pub fn new(nb_runs: u64) -> Self {
        MonteCarloSettings {
            nb_runs,
            confidence: DEFAULT_CONFIDENCE,
            quantiles: Vec::new(),
        }
    }
}

/// Settings of an exact exploration (see
/// [`raichu_explore::ExactSettings`], whose target, horizon and thread
/// count come from the [`Study`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExactExplorationSettings {
    /// Minimal sequence probability cut-off, in `(0, 1]`.
    pub min_probability: Option<f64>,
    /// Maximal sequence length cut-off, in fired transitions.
    pub max_length: Option<usize>,
    /// Maximal number of failures along a sequence.
    pub max_failures: Option<u64>,
    /// Maximal number of expanded nodes.
    pub max_branches: Option<u64>,
    /// Relative gap above which the result is flagged inconclusive.
    pub gap_tolerance: f64,
    /// Requested relative precision of every sequence probability.
    pub rel_precision: f64,
    /// Cap on the uniformization terms.
    pub max_terms: usize,
}

impl Default for ExactExplorationSettings {
    fn default() -> Self {
        let precision = Precision::default();
        ExactExplorationSettings {
            min_probability: None,
            max_length: None,
            max_failures: None,
            max_branches: None,
            gap_tolerance: DEFAULT_GAP_TOLERANCE,
            rel_precision: precision.rel_precision,
            max_terms: precision.max_terms,
        }
    }
}

impl ExactExplorationSettings {
    /// The setting names, as serialized.
    pub const NAMES: &'static [&'static str] = &[
        "min_probability",
        "max_length",
        "max_failures",
        "max_branches",
        "gap_tolerance",
        "rel_precision",
        "max_terms",
    ];

    /// The engine settings these stand for, on `study`.
    #[must_use]
    pub fn to_engine(&self, study: &Study) -> ExactSettings {
        let mut settings = ExactSettings::new(study.target.clone(), study.horizon);
        settings.cutoffs = Cutoffs {
            min_probability: self.min_probability,
            max_length: self.max_length,
            max_failures: self.max_failures,
            max_branches: self.max_branches,
        };
        settings.gap_tolerance = self.gap_tolerance;
        settings.precision = Precision {
            rel_precision: self.rel_precision,
            max_terms: self.max_terms,
        };
        settings.threads = study.threads;
        settings
    }
}

/// Settings of a discretised exploration (see
/// [`raichu_explore::DiscretisedSettings`], whose target, horizon and
/// thread count come from the [`Study`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DiscretisedExplorationSettings {
    /// Minimal sequence probability cut-off, in `(0, 1]`.
    pub min_probability: Option<f64>,
    /// Maximal sequence length cut-off, in fired transitions.
    pub max_length: Option<usize>,
    /// Maximal number of failures along a sequence.
    pub max_failures: Option<u64>,
    /// Maximal number of expanded nodes per pass. Default
    /// [`DEFAULT_MAX_BRANCHES`].
    pub max_branches: Option<u64>,
    /// Relative gap above which the result is flagged inconclusive.
    pub gap_tolerance: f64,
    /// Discretisation level `K >= 1`. Default [`DEFAULT_LEVEL`].
    pub level: u32,
    /// Whether to estimate the discretisation error by refinement.
    /// Default `true`.
    pub refine: bool,
}

impl Default for DiscretisedExplorationSettings {
    fn default() -> Self {
        DiscretisedExplorationSettings {
            min_probability: None,
            max_length: None,
            max_failures: None,
            max_branches: Some(DEFAULT_MAX_BRANCHES),
            gap_tolerance: DEFAULT_GAP_TOLERANCE,
            level: DEFAULT_LEVEL,
            refine: true,
        }
    }
}

impl DiscretisedExplorationSettings {
    /// The setting names, as serialized.
    pub const NAMES: &'static [&'static str] = &[
        "min_probability",
        "max_length",
        "max_failures",
        "max_branches",
        "gap_tolerance",
        "level",
        "refine",
    ];

    /// The engine settings these stand for, on `study`.
    #[must_use]
    pub fn to_engine(&self, study: &Study) -> DiscretisedSettings {
        let mut settings = DiscretisedSettings::new(study.target.clone(), study.horizon);
        settings.cutoffs = Cutoffs {
            min_probability: self.min_probability,
            max_length: self.max_length,
            max_failures: self.max_failures,
            max_branches: self.max_branches,
        };
        settings.gap_tolerance = self.gap_tolerance;
        settings.level = self.level;
        settings.refine = self.refine;
        settings.threads = study.threads;
        settings
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn keys(value: &Value) -> Vec<String> {
        value.as_object().unwrap().keys().cloned().collect()
    }

    /// The declared name lists are the serialized field names, so the
    /// refusal of a foreign setting cannot drift from the format.
    #[test]
    fn setting_names_are_the_serialized_fields() {
        let mut mc = keys(&serde_json::to_value(MonteCarloSettings::new(1)).unwrap());
        let mut exact = keys(&serde_json::to_value(ExactExplorationSettings::default()).unwrap());
        let mut disc =
            keys(&serde_json::to_value(DiscretisedExplorationSettings::default()).unwrap());
        for (found, declared) in [
            (&mut mc, MonteCarloSettings::NAMES),
            (&mut exact, ExactExplorationSettings::NAMES),
            (&mut disc, DiscretisedExplorationSettings::NAMES),
        ] {
            found.sort();
            let mut declared: Vec<String> = declared.iter().map(|&n| n.to_owned()).collect();
            declared.sort();
            assert_eq!(*found, declared);
        }
    }

    #[test]
    fn names_match_the_serialized_tags() {
        let methods = [
            Method::MonteCarlo(MonteCarloSettings::new(1)),
            Method::Exact(ExactExplorationSettings::default()),
            Method::Discretised(DiscretisedExplorationSettings::default()),
        ];
        for (method, name) in methods.iter().zip(Method::NAMES) {
            assert_eq!(method.name(), name);
            let value = serde_json::to_value(method).unwrap();
            assert_eq!(value["name"], name);
        }
    }
}
