//! Adapter: biased Monte-Carlo fitted by cross-entropy (`raichu-montecarlo`).

use raichu_core::CompiledModel;
use raichu_montecarlo::{run_cross_entropy, CrossEntropyError};

use crate::{
    Answer, CrossEntropySamplingSettings, Detail, Method, QuantificationEngine, QuantifyError,
    Study, TargetProbability,
};

impl QuantificationEngine for CrossEntropySamplingSettings {
    fn method(&self) -> Method {
        Method::CrossEntropy(self.clone())
    }

    fn uses_seed(&self) -> bool {
        true
    }

    /// The campaign answers one probability, at the horizon: the reporting
    /// instants take no part in it.
    fn uses_instants(&self) -> bool {
        false
    }

    /// Fit the factors, run the final weighted campaign, and state its
    /// estimate with the diagnostics beside it.
    fn answer(&self, model: &CompiledModel, study: &Study) -> Result<Answer, QuantifyError> {
        let mut result = run_cross_entropy(model, &self.to_engine(study)).map_err(|e| match e {
            CrossEntropyError::Engine(e) => QuantifyError::Engine(e),
            e @ CrossEntropyError::NoHit { .. } => QuantifyError::NoHit(e),
            other => QuantifyError::InvalidSettings {
                method: "cross_entropy".to_owned(),
                detail: other.to_string(),
            },
        })?;
        // One entry per replica would dominate the document; the reached
        // count and the estimate say what the envelope needs.
        result.ends.clear();
        let e = &result.estimate;
        Ok(Answer {
            probability: TargetProbability::WeightedEstimate {
                estimate: e.estimate,
                standard_error: e.standard_error,
                reached: result.reached,
                replicas: result.replicas,
                level: e.level,
                method: e.method,
                low: e.low,
                high: e.high,
                effective_sample_size: e.effective_sample_size,
                relative_error: e.relative_error,
                inconclusive: result.estimate_inconclusive,
            },
            detail: Detail::CrossEntropy(Box::new(result)),
        })
    }
}
