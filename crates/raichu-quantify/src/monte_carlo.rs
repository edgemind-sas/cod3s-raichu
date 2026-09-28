//! Adapter: Monte-Carlo simulation (`raichu-montecarlo`).

use raichu_core::{CompiledModel, FlowConfig, SolverParams};
use raichu_montecarlo::{
    run_to_targets, run_to_targets_with_fmu, wilson_bounds, IntervalMethod, McConfig,
    TargetCampaign,
};
use std::path::Path;

use crate::{
    Answer, Detail, Method, MonteCarloSettings, QuantificationEngine, QuantifyError, Study,
    TargetProbability,
};

impl MonteCarloSettings {
    /// The campaign configuration these settings stand for, on `study`:
    /// the one [`raichu_montecarlo::run`] reproduces byte for byte with
    /// the same seed.
    #[must_use]
    pub fn to_engine(&self, study: &Study) -> McConfig {
        McConfig {
            nb_runs: self.nb_runs,
            seed: study.seed,
            t_max: study.horizon,
            samples: study.reporting_instants(),
            threads: study.threads,
            quantiles: self.quantiles.clone(),
            confidence: self.confidence,
            ode: SolverParams::default(),
            stop_at_targets: true,
            flow: FlowConfig::default(),
        }
    }

    pub(crate) fn answer_with_fmu(
        &self,
        model: &CompiledModel,
        study: &Study,
        base_dir: &Path,
        require_parallel: bool,
    ) -> Result<Answer, QuantifyError> {
        self.check_nb_runs()?;
        let campaign = run_to_targets_with_fmu(
            model,
            &self.to_engine(study),
            base_dir,
            true,
            require_parallel,
        )?;
        Ok(self.answer_from_campaign(campaign, study))
    }

    fn check_nb_runs(&self) -> Result<(), QuantifyError> {
        if self.nb_runs == 0 {
            return Err(QuantifyError::InvalidSettings {
                method: "monte_carlo".to_owned(),
                detail: "a campaign runs at least 1 replica, got nb_runs = 0".to_owned(),
            });
        }
        Ok(())
    }

    fn answer_from_campaign(&self, campaign: TargetCampaign, study: &Study) -> Answer {
        let reached = campaign.count_reached(&study.target);
        let replicas = self.nb_runs;
        let estimate = reached as f64 / replicas as f64;
        let (low, high) = wilson_bounds(estimate, replicas, self.confidence);
        Answer {
            probability: TargetProbability::ConfidenceInterval {
                estimate,
                reached,
                replicas,
                level: self.confidence,
                method: IntervalMethod::Wilson,
                low,
                high,
            },
            detail: Detail::MonteCarlo(Box::new(campaign.estimates)),
        }
    }
}

impl QuantificationEngine for MonteCarloSettings {
    fn method(&self) -> Method {
        Method::MonteCarlo(self.clone())
    }

    fn uses_seed(&self) -> bool {
        true
    }

    fn uses_instants(&self) -> bool {
        true
    }

    /// One stop-at-targets campaign: the replicas whose end cause is the
    /// study's target by the horizon are counted, and the proportion gets
    /// a Wilson score interval at the declared level.
    fn answer(&self, model: &CompiledModel, study: &Study) -> Result<Answer, QuantifyError> {
        self.check_nb_runs()?;
        let config = self.to_engine(study);
        let campaign = run_to_targets(model, &config)?;
        Ok(self.answer_from_campaign(campaign, study))
    }
}
