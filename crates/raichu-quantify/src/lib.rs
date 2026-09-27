//! # raichu-quantify: one study, any quantification engine, one envelope
//!
//! RAICHU answers "what is the probability that this feared event happens
//! by this horizon" with three engines:
//!
//! - **Monte-Carlo simulation** ([`Method::MonteCarlo`], crate
//!   `raichu-montecarlo`): replicas drawn at random, the probability
//!   estimated with a confidence interval;
//! - **exact exploration** ([`Method::Exact`], crate `raichu-explore`):
//!   the sequence tree of a Markov model, each sequence probability in
//!   closed form, the answer framed by guaranteed bounds;
//! - **discretised exploration** ([`Method::Discretised`], crate
//!   `raichu-explore`): the sequence tree for every law the engine
//!   carries, bounds on the discretised model plus an estimate of the
//!   discretisation error.
//!
//! Each engine keeps its own settings and its own detailed result. This
//! crate puts one contract above them:
//!
//! - a [`Study`] states the question once (feared event, horizon,
//!   reporting instants, seed, thread count);
//! - a [`Method`] names the engine and carries the settings that belong
//!   to it alone;
//! - [`quantify`] hands the study to the engine and returns a
//!   [`Quantification`], the `raichu.quantification` envelope: the method
//!   as applied, the provenance (engine version, model name and content
//!   hash, horizon, and the seed and instants where the method uses
//!   them), the probability of the feared event with its uncertainty
//!   ([`TargetProbability`]), and the engine's own result unchanged
//!   inside ([`Detail`]).
//!
//! # What the probability is
//!
//! The probability that the study's target is the **first** of the
//! model's declared targets reached, by the horizon. Every engine already
//! behaves this way: a Monte-Carlo trajectory stopped at targets stops at
//! the first one, and an exploration treats a sequence that reaches
//! another target first as a leaf contributing nothing. On a model
//! declaring a single target, it is simply the probability of reaching it
//! by the horizon.
//!
//! # Adding an engine
//!
//! An engine plugs in by implementing [`QuantificationEngine`] in an
//! adapter module of this crate (the engine crates do not know this
//! contract, which is what keeps the dependency graph acyclic) and adding
//! its [`Method`] variant. [`Method`], [`TargetProbability`] and
//! [`Detail`] are `#[non_exhaustive]`, so adding a variant is not a
//! breaking change for Rust code matching on them. A new method also
//! raises the envelope's format version: a reader written before it
//! refuses such an envelope by its version rather than by an unknown name.
//! The bindings reach every method through the single [`quantify`] entry
//! point.

mod envelope;
mod error;
mod exact_json;
mod exploration;
mod hash;
mod method;
mod monte_carlo;
mod study;

pub use envelope::{
    read_quantification, Detail, Quantification, QuantificationProvenance, ReadQuantificationError,
    TargetProbability, QUANTIFICATION_FORMAT, QUANTIFICATION_VERSION,
};
pub use error::QuantifyError;
pub use hash::model_content_hash;
pub use method::{
    DiscretisedExplorationSettings, ExactExplorationSettings, Method, MonteCarloSettings,
};
pub use study::Study;

use raichu_core::CompiledModel;
use raichu_model::Model;

/// The answer one engine gives to a study, before the envelope is put
/// around it: the probability with its uncertainty, and the engine's own
/// result.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// Probability of the study's target, with its uncertainty.
    pub probability: TargetProbability,
    /// The engine's detailed result, unchanged.
    pub detail: Detail,
}

/// A quantification engine: something that answers a [`Study`] on a
/// compiled model.
///
/// Implemented by the settings type of each [`Method`] variant. The
/// envelope, the provenance and the study checks are the dispatcher's
/// ([`quantify_with`]), so an implementation only runs its engine and
/// states its answer.
pub trait QuantificationEngine {
    /// The method, with the settings this engine applies, as the envelope
    /// records it.
    fn method(&self) -> Method;

    /// Whether the study's seed takes part in the answer. When it does
    /// not, the envelope does not record it, since changing it changes
    /// nothing.
    fn uses_seed(&self) -> bool;

    /// Whether the study's reporting instants take part in the answer
    /// (recorded only then, for the same reason as the seed).
    fn uses_instants(&self) -> bool;

    /// Run the engine on `model` for `study`, already validated against
    /// the model.
    ///
    /// # Errors
    /// A setting outside its domain, or any engine error.
    fn answer(&self, model: &CompiledModel, study: &Study) -> Result<Answer, QuantifyError>;
}

/// Quantify `study` on `model` with `method`.
///
/// The model is compiled once, the study is checked against it (the
/// target must be one of the model's declared targets), the engine named
/// by `method` runs, and its answer is returned in the
/// `raichu.quantification` envelope.
///
/// # Errors
/// [`QuantifyError::UnknownTarget`] or [`QuantifyError::InvalidStudy`]
/// before anything runs, [`QuantifyError::InvalidSettings`] for a method
/// setting outside its domain, [`QuantifyError::Compile`] for a model that
/// does not compile, and [`QuantifyError::Engine`] for the engine's own
/// typed refusals (a model outside the exact domain, for instance).
pub fn quantify(
    model: &Model,
    study: &Study,
    method: &Method,
) -> Result<Quantification, QuantifyError> {
    match method {
        Method::MonteCarlo(settings) => quantify_with(settings, model, study),
        Method::Exact(settings) => quantify_with(settings, model, study),
        Method::Discretised(settings) => quantify_with(settings, model, study),
    }
}

/// Quantify `study` on `model` with any [`QuantificationEngine`]: the
/// dispatcher behind [`quantify`], public so that an engine can be driven
/// before it has its [`Method`] variant.
///
/// # Errors
/// As [`quantify`].
pub fn quantify_with(
    engine: &dyn QuantificationEngine,
    model: &Model,
    study: &Study,
) -> Result<Quantification, QuantifyError> {
    let compiled = CompiledModel::compile(model)?;
    study.validate(&compiled)?;
    let model_hash = model_content_hash(model)?;
    let Answer {
        probability,
        detail,
    } = engine.answer(&compiled, study)?;
    Ok(Quantification {
        format: QUANTIFICATION_FORMAT.to_owned(),
        version: QUANTIFICATION_VERSION,
        method: engine.method(),
        provenance: QuantificationProvenance {
            engine_version: env!("CARGO_PKG_VERSION").to_owned(),
            model: model.name.clone(),
            model_hash,
            target: study.target.clone(),
            horizon: study.horizon,
            instants: engine.uses_instants().then(|| study.reporting_instants()),
            seed: engine.uses_seed().then_some(study.seed),
        },
        probability,
        detail,
    })
}
