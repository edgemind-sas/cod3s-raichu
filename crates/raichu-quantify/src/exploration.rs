//! Adapters: exact and discretised exploration (`raichu-explore`).

use raichu_core::CompiledModel;
use raichu_explore::{explore_discretised, explore_exact, ExplorationResult};

use crate::{
    Answer, Detail, DiscretisedExplorationSettings, ExactExplorationSettings, Method,
    QuantificationEngine, QuantifyError, Study, TargetProbability,
};

/// The bounds an exploration states, with its error estimate when it made
/// one.
fn bounds(result: ExplorationResult) -> Answer {
    Answer {
        probability: TargetProbability::Bounds {
            lower: result.lower,
            upper: result.upper,
            inconclusive: result.inconclusive,
            error_estimate: result.error_estimate(),
        },
        detail: Detail::Exploration(Box::new(result)),
    }
}

impl QuantificationEngine for ExactExplorationSettings {
    fn method(&self) -> Method {
        Method::Exact(self.clone())
    }

    fn uses_seed(&self) -> bool {
        false
    }

    fn uses_instants(&self) -> bool {
        false
    }

    fn answer(&self, model: &CompiledModel, study: &Study) -> Result<Answer, QuantifyError> {
        Ok(bounds(explore_exact(model, &self.to_engine(study))?))
    }
}

impl QuantificationEngine for DiscretisedExplorationSettings {
    fn method(&self) -> Method {
        Method::Discretised(self.clone())
    }

    fn uses_seed(&self) -> bool {
        false
    }

    fn uses_instants(&self) -> bool {
        false
    }

    fn answer(&self, model: &CompiledModel, study: &Study) -> Result<Answer, QuantifyError> {
        Ok(bounds(explore_discretised(model, &self.to_engine(study))?))
    }
}
