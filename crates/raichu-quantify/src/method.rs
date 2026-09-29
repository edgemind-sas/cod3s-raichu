//! The methods of quantification and the settings that belong to each.

use raichu_explore::{
    Cutoffs, DiscretisedSettings, ExactSettings, Precision, DEFAULT_GAP_TOLERANCE, DEFAULT_LEVEL,
    DEFAULT_MAX_BRANCHES,
};
use std::collections::BTreeMap;

use raichu_montecarlo::{
    CrossEntropySettings, ImportanceSource, SplittingSettings, DEFAULT_CONFIDENCE,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{QuantifyError, Study};

/// A method of quantification: the engine that answers the study, with
/// the settings that belong to it alone (what every engine shares is in
/// the [`Study`]).
///
/// Serialized as `{"name": <method>, "settings": {...}}`, the names being
/// `monte_carlo`, `exact`, `discretised`, `cross_entropy` and `splitting`. Non-exhaustive: a later
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
    /// Biased Monte-Carlo simulation whose rate factors are fitted by
    /// cross-entropy, each replica weighted by its likelihood ratio (crate
    /// `raichu-montecarlo`): the method for feared events too rare for a
    /// plain campaign.
    CrossEntropy(CrossEntropySamplingSettings),
    /// Adaptive splitting with independent batches and a numeric importance attribute.
    Splitting(SplittingSamplingSettings),
}

impl Method {
    /// The method names this engine provides, in the order the
    /// documentation presents them.
    pub const NAMES: [&'static str; 5] = [
        "monte_carlo",
        "exact",
        "discretised",
        "cross_entropy",
        "splitting",
    ];

    /// The method's name, as serialized.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Method::MonteCarlo(_) => "monte_carlo",
            Method::Exact(_) => "exact",
            Method::Discretised(_) => "discretised",
            Method::CrossEntropy(_) => "cross_entropy",
            Method::Splitting(_) => "splitting",
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
            "cross_entropy" => Some(CrossEntropySamplingSettings::NAMES),
            "splitting" => Some(SplittingSamplingSettings::NAMES),
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

/// Settings of a cross-entropy quantification: a biased Monte-Carlo
/// campaign whose per-family rate factors are fitted by cross-entropy
/// (see [`raichu_montecarlo::run_cross_entropy`], whose target, horizon,
/// seed and thread count come from the [`Study`]).
///
/// Every setting but `nb_runs` has the default the driver documents. The
/// campaign uses the study's seed and not its reporting instants: it
/// answers one probability, at the horizon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossEntropySamplingSettings {
    /// Replicas of the final campaign, at least 1.
    pub nb_runs: u64,
    /// Replicas of each pilot iteration.
    #[serde(default = "ce_defaults::pilot_runs")]
    pub pilot_runs: u64,
    /// Cap on pilot iterations, escalations included.
    #[serde(default = "ce_defaults::max_iterations")]
    pub max_iterations: u64,
    /// Weight of a new factor against the previous one, in `(0, 1]`.
    #[serde(default = "ce_defaults::smoothing")]
    pub smoothing: f64,
    /// Relative factor change under which the fit stops.
    #[serde(default = "ce_defaults::tolerance")]
    pub tolerance: f64,
    /// Confidence level of the interval, strictly inside `(0, 1)`.
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    /// Starting factor of every family not declared a repair.
    #[serde(default = "ce_defaults::initial_factor")]
    pub initial_factor: f64,
    /// Multiplier applied to non-repair families after a pilot with no hit.
    #[serde(default = "ce_defaults::escalation_ratio")]
    pub escalation_ratio: f64,
    /// Lowest admissible factor.
    #[serde(default = "ce_defaults::factor_min")]
    pub factor_min: f64,
    /// Highest admissible factor.
    #[serde(default = "ce_defaults::factor_max")]
    pub factor_max: f64,
    /// Whether factors are fitted; `false` runs the final campaign at the
    /// starting factors.
    #[serde(default = "ce_defaults::fit")]
    pub fit: bool,
    /// Effective sample size under which a pilot cannot confirm the fit
    /// and the final estimate is flagged inconclusive.
    #[serde(default = "ce_defaults::min_effective_sample_size")]
    pub min_effective_sample_size: f64,
    /// Override of the automatic families: qualified transition name to
    /// family label.
    #[serde(default)]
    pub families: BTreeMap<String, String>,
}

/// The driver's documented defaults, one function per setting (serde's
/// per-field default form), read from its constants.
mod ce_defaults {
    use raichu_montecarlo as mc;

    pub(super) fn pilot_runs() -> u64 {
        mc::DEFAULT_CE_PILOT_RUNS
    }
    pub(super) fn max_iterations() -> u64 {
        mc::DEFAULT_CE_MAX_ITERATIONS
    }
    pub(super) fn smoothing() -> f64 {
        mc::DEFAULT_CE_SMOOTHING
    }
    pub(super) fn tolerance() -> f64 {
        mc::DEFAULT_CE_TOLERANCE
    }
    pub(super) fn initial_factor() -> f64 {
        mc::DEFAULT_CE_INITIAL_FACTOR
    }
    pub(super) fn escalation_ratio() -> f64 {
        mc::DEFAULT_CE_ESCALATION_RATIO
    }
    pub(super) fn factor_min() -> f64 {
        mc::DEFAULT_CE_FACTOR_MIN
    }
    pub(super) fn factor_max() -> f64 {
        mc::DEFAULT_CE_FACTOR_MAX
    }
    pub(super) fn fit() -> bool {
        true
    }
    pub(super) fn min_effective_sample_size() -> f64 {
        mc::DEFAULT_CE_MIN_EFFECTIVE_SAMPLE_SIZE
    }
}

impl CrossEntropySamplingSettings {
    /// The setting names, as serialized.
    pub const NAMES: &'static [&'static str] = &[
        "nb_runs",
        "pilot_runs",
        "max_iterations",
        "smoothing",
        "tolerance",
        "confidence",
        "initial_factor",
        "escalation_ratio",
        "factor_min",
        "factor_max",
        "fit",
        "min_effective_sample_size",
        "families",
    ];

    /// `nb_runs` final replicas, every other setting at its default.
    #[must_use]
    pub fn new(nb_runs: u64) -> Self {
        CrossEntropySamplingSettings {
            nb_runs,
            pilot_runs: ce_defaults::pilot_runs(),
            max_iterations: ce_defaults::max_iterations(),
            smoothing: ce_defaults::smoothing(),
            tolerance: ce_defaults::tolerance(),
            confidence: DEFAULT_CONFIDENCE,
            initial_factor: ce_defaults::initial_factor(),
            escalation_ratio: ce_defaults::escalation_ratio(),
            factor_min: ce_defaults::factor_min(),
            factor_max: ce_defaults::factor_max(),
            fit: ce_defaults::fit(),
            min_effective_sample_size: ce_defaults::min_effective_sample_size(),
            families: BTreeMap::new(),
        }
    }

    /// The driver settings these stand for, on `study`.
    #[must_use]
    pub fn to_engine(&self, study: &Study) -> CrossEntropySettings {
        CrossEntropySettings {
            target: study.target.clone(),
            t_max: study.horizon,
            seed: study.seed,
            samples: Vec::new(),
            nb_runs: self.nb_runs,
            pilot_runs: self.pilot_runs,
            max_iterations: self.max_iterations,
            smoothing: self.smoothing,
            tolerance: self.tolerance,
            confidence: self.confidence,
            initial_factor: self.initial_factor,
            escalation_ratio: self.escalation_ratio,
            factor_min: self.factor_min,
            factor_max: self.factor_max,
            fit: self.fit,
            min_effective_sample_size: self.min_effective_sample_size,
            families: self.families.clone(),
            threads: study.threads,
            ..CrossEntropySettings::default()
        }
    }
}

/// Adaptive splitting settings; target, horizon, seed and threads come from the study.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplittingSamplingSettings {
    /// Numeric model attribute used as the importance score.
    pub importance: ImportanceSource,
    /// Particles per independent batch, at least two.
    #[serde(default = "splitting_defaults::particles")]
    pub particles: u64,
    /// Independent batches, at least two.
    #[serde(default = "splitting_defaults::batches")]
    pub batches: u64,
    /// Additional observation dates for continuously evolving scores.
    #[serde(default = "splitting_defaults::score_grid")]
    pub score_grid: Vec<f64>,
    /// Replacement iteration cap; reaching it returns an error without an estimate.
    #[serde(default = "splitting_defaults::max_iterations")]
    pub max_iterations: u64,
    /// Two-sided confidence level, strictly inside `(0, 1)`.
    #[serde(default = "default_confidence")]
    pub confidence: f64,
}
mod splitting_defaults {
    pub(super) fn score_grid() -> Vec<f64> {
        raichu_montecarlo::DEFAULT_SPLITTING_SCORE_GRID.to_vec()
    }
    pub(super) fn particles() -> u64 {
        raichu_montecarlo::DEFAULT_SPLITTING_PARTICLES
    }
    pub(super) fn batches() -> u64 {
        raichu_montecarlo::DEFAULT_SPLITTING_BATCHES
    }
    pub(super) fn max_iterations() -> u64 {
        raichu_montecarlo::DEFAULT_SPLITTING_MAX_ITERATIONS
    }
}
impl SplittingSamplingSettings {
    /// Accepted serialized setting names.
    pub const NAMES: &'static [&'static str] = &[
        "importance",
        "particles",
        "batches",
        "score_grid",
        "max_iterations",
        "confidence",
    ];
    /// Construct a campaign with the driver's documented defaults.
    #[must_use]
    pub fn new(importance: ImportanceSource) -> Self {
        Self {
            importance,
            particles: splitting_defaults::particles(),
            batches: splitting_defaults::batches(),
            score_grid: splitting_defaults::score_grid(),
            max_iterations: splitting_defaults::max_iterations(),
            confidence: DEFAULT_CONFIDENCE,
        }
    }
    /// Driver settings for this study, using the default solver and flow policy.
    #[must_use]
    pub fn to_engine(&self, study: &Study) -> SplittingSettings {
        SplittingSettings {
            target: study.target.clone(),
            t_max: study.horizon,
            seed: study.seed,
            threads: study.threads,
            importance: self.importance.clone(),
            particles: self.particles,
            batches: self.batches,
            score_grid: self.score_grid.clone(),
            max_iterations: self.max_iterations,
            confidence: self.confidence,
            ..SplittingSettings::default()
        }
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
        let mut ce = keys(&serde_json::to_value(CrossEntropySamplingSettings::new(1)).unwrap());
        let mut splitting = keys(
            &serde_json::to_value(SplittingSamplingSettings::new(
                ImportanceSource::Attribute {
                    name: "sys.score".into(),
                },
            ))
            .unwrap(),
        );
        for (found, declared) in [
            (&mut mc, MonteCarloSettings::NAMES),
            (&mut exact, ExactExplorationSettings::NAMES),
            (&mut disc, DiscretisedExplorationSettings::NAMES),
            (&mut ce, CrossEntropySamplingSettings::NAMES),
            (&mut splitting, SplittingSamplingSettings::NAMES),
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
            Method::CrossEntropy(CrossEntropySamplingSettings::new(1)),
            Method::Splitting(SplittingSamplingSettings::new(
                ImportanceSource::Attribute {
                    name: "sys.score".into(),
                },
            )),
        ];
        for (method, name) in methods.iter().zip(Method::NAMES) {
            assert_eq!(method.name(), name);
            let value = serde_json::to_value(method).unwrap();
            assert_eq!(value["name"], name);
        }
    }
}
