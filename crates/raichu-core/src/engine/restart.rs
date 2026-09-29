//! Independent, age-conditioned continuations for adaptive splitting.

use super::*;
use raichu_numeric::laws::Law;

/// The age of a drawn clock and whether a clone must redraw it on rearming.
#[derive(Debug, Clone, Copy)]
pub(super) struct DrawnClock {
    age: DeferredAge,
    pub(super) redraw: bool,
}

impl<'m> Engine<'m> {
    /// Restart a checkpoint on an independent random stream.
    ///
    /// The seed comes from `config`; `rng_stream` selects its substream.
    /// Pending stochastic durations are redrawn conditional on their elapsed
    /// active age, with survival `S(a + r) / S(a)`. A paused `resume` clock
    /// keeps its age and is redrawn when it rearms. Deterministic dates and
    /// watched transitions are preserved. Recorded history is discarded.
    /// Unlike [`Engine::from_snapshot`], this does not replay the old future.
    ///
    /// Call at a completed instant, after all simultaneous events have fired.
    /// The model and configuration must describe the captured trajectory.
    ///
    /// # Errors
    /// Refuses FMU models, deferred dates, biased rates (in either the input
    /// configuration or checkpoint), and invalid conditional clock evolution.
    pub fn restart_from_snapshot(
        model: &'m CompiledModel,
        mut config: EngineConfig,
        snapshot: &Snapshot,
        rng_stream: u64,
    ) -> Result<Self, EngineError> {
        if let Some(unit) = model.fmu_units.first() {
            return Err(EngineError::FmuSnapshotApi {
                unit: unit.name.clone(),
                operation: "independent snapshot restart",
                alternative: "a native-only model",
            });
        }
        if config.stochastic_dates != StochasticDates::Drawn
            || snapshot.stochastic_dates != StochasticDates::Drawn
        {
            return Err(EngineError::InvalidStudyParameter {
                parameter: "stochastic_dates".to_owned(),
                detail: "independent snapshot restart requires drawn dates".to_owned(),
            });
        }
        if !config.rate_factors.is_empty() || snapshot.biased_rates {
            return Err(EngineError::InvalidStudyParameter {
                parameter: "rate_factors".to_owned(),
                detail: "independent snapshot restart requires unbiased rates".to_owned(),
            });
        }
        config.rng_stream = rng_stream;
        let seed = config.seed;
        let mut engine = Self::from_snapshot(model, config, snapshot)?;
        engine.rng = raichu_rng::replica_rng(seed, rng_stream);
        engine.forget_history();
        for idx in 0..model.transitions.len() {
            if let Some(clock) = engine.clocks[idx].as_mut() {
                if clock.age.running {
                    engine.redraw_clock(idx, true)?;
                } else {
                    clock.redraw = true;
                }
            }
        }
        Ok(engine)
    }

    /// Mirror `updateIT` without drawing or changing the legacy countdown.
    pub(super) fn refresh_drawn_clock(&mut self, idx: TransIdx, source: bool, guard: bool) {
        if !schedule::is_stochastic(&self.model.transitions[idx].distrib) {
            return;
        }
        let policy = self.model.transitions[idx].on_interruption;
        if !source || (!guard && policy == raichu_model::InterruptionPolicy::Reset) {
            self.clocks[idx] = None;
            return;
        }
        match self.clocks[idx].as_mut() {
            None if guard => {
                self.clocks[idx] = Some(DrawnClock {
                    age: DeferredAge {
                        armed_at: self.time,
                        banked: 0.0,
                        since: self.time,
                        running: true,
                    },
                    redraw: false,
                });
            }
            Some(clock) if !guard && policy == raichu_model::InterruptionPolicy::Resume => {
                if clock.age.running {
                    clock.age.banked = clock.age.age(self.time);
                    clock.age.since = self.time;
                    clock.age.running = false;
                }
            }
            Some(clock) if guard && !clock.age.running => {
                clock.age.since = self.time;
                clock.age.running = true;
            }
            _ => {}
        }
    }

    /// Redraw an active clock. A rearming paused hazard is already banked;
    /// an active piecewise-constant hazard needs its last stretch accrued.
    pub(super) fn redraw_clock(&mut self, idx: TransIdx, accrue: bool) -> Result<(), EngineError> {
        let transition = &self.model.transitions[idx];
        let invalid = |detail: String| EngineError::TypeError {
            time: self.time,
            detail: format!("independent restart of `{}`: {detail}", transition.name),
        };
        let Some(clock) = self.clocks[idx] else {
            return Err(invalid("missing stochastic clock age".to_owned()));
        };
        let date = if let CLaw::ExpVar { rate, continuous } = &transition.distrib {
            let lambda = self.eval_rate(idx, rate)?;
            let Some(hazard) = self.hazards[idx].as_mut() else {
                return Err(invalid("missing cumulative hazard".to_owned()));
            };
            if accrue && !continuous {
                hazard.accumulated += hazard.rate * (self.time - hazard.since);
            }
            let residual: f64 = rand_distr::Exp1.sample(&mut self.rng);
            hazard.threshold = hazard.accumulated + residual;
            hazard.since = self.time;
            hazard.rate = lambda;
            if *continuous || lambda == 0.0 {
                f64::INFINITY
            } else {
                self.time + residual / lambda
            }
        } else {
            let residual =
                conditional_delay(&transition.distrib, clock.age.age(self.time), &mut self.rng)
                    .map_err(invalid)?;
            self.time + residual
        };
        self.pending[idx] = Some(date);
        self.frozen[idx] = None;
        if let Some(clock) = self.clocks[idx].as_mut() {
            clock.redraw = false;
        }
        Ok(())
    }
}

/// Sample conditional fixed-law survival through the existing numerical
/// law library. Bisection on cumulative hazard also covers tails whose
/// survival underflows, where an inverse survival would lose the draw.
fn conditional_delay(law: &CLaw, age: f64, rng: &mut ChaCha8Rng) -> Result<f64, String> {
    let law = match law {
        CLaw::Exp(rate) => Law::exponential(*rate),
        CLaw::Weibull(shape, scale) => Law::weibull(*shape, *scale),
        CLaw::Lognormal(mu, sigma) => Law::lognormal(*mu, *sigma),
        CLaw::Gamma(shape, scale) => Law::gamma(*shape, *scale),
        CLaw::Uniform(low, high) => Law::uniform(*low, *high),
        CLaw::Empirical(points) => Law::empirical(points.clone()),
        _ => return Err("expected a fixed stochastic law".to_owned()),
    }
    .map_err(|error| error.to_string())?;
    let u: f64 = rand::distr::Open01.sample(rng);
    let target = -(-u).ln_1p();
    let residual = law
        .conditional_quantile(age, u)
        .map_err(|error| error.to_string())?;
    if residual.is_finite() {
        // An empirical law can have atoms: its quantile need not attain
        // the requested hazard exactly, and the generalized inverse is final.
        if law.name() == "empirical" {
            return Ok(residual);
        }
        let got = law
            .conditional_cumulative_hazard(age, residual)
            .map_err(|error| error.to_string())?;
        if (got - target).abs() <= 1e-10 * target.max(1.0) {
            return Ok(residual);
        }
    }
    let hazard = |r| {
        law.conditional_cumulative_hazard(age, r)
            .map_err(|error| error.to_string())
    };
    let mut low = 0.0;
    let mut high = if residual.is_finite() && residual > 0.0 {
        residual
    } else {
        1.0
    };
    while hazard(high)? < target {
        low = high;
        high *= 2.0;
        if !high.is_finite() {
            return Err("conditional firing time is not representable".to_owned());
        }
    }
    for _ in 0..128 {
        let mid = low + (high - low) * 0.5;
        if mid == low || mid == high {
            return Ok(high);
        }
        if hazard(mid)? < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    Err("conditional firing-time inversion did not converge".to_owned())
}
