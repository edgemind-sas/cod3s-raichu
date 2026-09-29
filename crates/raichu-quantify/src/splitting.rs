//! Adapter for adaptive multilevel splitting with independent batches.

use crate::{
    Answer, Detail, Method, QuantificationEngine, QuantifyError, SplittingSamplingSettings, Study,
    TargetProbability,
};
use raichu_core::CompiledModel;
use raichu_montecarlo::run_splitting;

impl QuantificationEngine for SplittingSamplingSettings {
    fn method(&self) -> Method {
        Method::Splitting(self.clone())
    }

    fn uses_seed(&self) -> bool {
        true
    }

    fn uses_instants(&self) -> bool {
        false
    }

    fn answer(&self, model: &CompiledModel, study: &Study) -> Result<Answer, QuantifyError> {
        let result = run_splitting(model, &self.to_engine(study))?;
        let e = &result.interval;
        Ok(Answer {
            probability: TargetProbability::SplittingEstimate {
                estimate: e.estimate,
                standard_error: e.standard_error,
                level: e.level,
                method: e.method,
                low: e.low,
                high: e.high,
                batches: result.batches.len() as u64,
                extinct_batches: result.extinct_batches,
                inconclusive: result.estimate_inconclusive,
            },
            detail: Detail::Splitting(Box::new(result)),
        })
    }
}
