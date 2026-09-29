//! The typed errors of the quantification contract.

use raichu_core::{CompileError, EngineError};

/// Why a study could not be quantified.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum QuantifyError {
    /// The study's target is not one of the model's declared targets.
    #[error(
        "no feared event named `{target}`; the model declares {}",
        listed(declared)
    )]
    UnknownTarget {
        /// The target the study named.
        target: String,
        /// The targets the model declares, in declaration order.
        declared: Vec<String>,
    },
    /// A study parameter outside its domain.
    #[error("invalid study parameter `{parameter}`: {detail}")]
    InvalidStudy {
        /// The parameter (`horizon`, `instants`, `threads`).
        parameter: String,
        /// What is wrong with it.
        detail: String,
    },
    /// A method name this engine does not provide.
    #[error(
        "quantification method `{name}` is not provided by this engine; the methods are {}",
        listed(known)
    )]
    UnknownMethod {
        /// The name given.
        name: String,
        /// The names this engine provides.
        known: Vec<String>,
    },
    /// A setting given to a method it does not belong to.
    #[error(
        "setting `{setting}` does not apply to the `{method}` method (it applies to {})",
        applies_to_text(applies_to)
    )]
    SettingNotApplicable {
        /// The method the setting was given to.
        method: String,
        /// The setting.
        setting: String,
        /// The methods that do take it (empty for a name no method takes).
        applies_to: Vec<String>,
    },
    /// A method setting outside its domain, or settings that do not
    /// parse.
    #[error("invalid `{method}` settings: {detail}")]
    InvalidSettings {
        /// The method.
        method: String,
        /// What is wrong.
        detail: String,
    },
    /// The model does not compile.
    #[error(transparent)]
    Compile(#[from] CompileError),
    /// The engine refused the study or failed while running it.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// A cross-entropy campaign found no replica reaching the target: no
    /// estimate is returned in place of a zero.
    #[error(transparent)]
    NoHit(raichu_montecarlo::CrossEntropyError),
    /// Splitting refused the campaign or reached a cap without an estimate.
    #[error(transparent)]
    Splitting(#[from] raichu_montecarlo::SplittingError),
    /// The model could not be serialized for its content hash.
    #[error("the model could not be serialized for its content hash: {0}")]
    ModelSerialization(String),
}

fn listed(names: &[String]) -> String {
    if names.is_empty() {
        return "none".to_owned();
    }
    names
        .iter()
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn applies_to_text(methods: &[String]) -> String {
    if methods.is_empty() {
        "no method".to_owned()
    } else {
        listed(methods)
    }
}
