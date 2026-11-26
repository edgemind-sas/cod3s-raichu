//! Indicators of an exploration: the mean value of a state indicator at
//! declared instants and its sojourn time, with a lower and an upper
//! bound, plus the mean time to the target truncated at the horizon.
//!
//! # What is measured
//!
//! The explored process is the model's process **stopped at the target**:
//! once the target is reached its state stays as it is (the convention of
//! the sequence mode, [`raichu_core::EngineConfig::stop_at_targets`]). For
//! a state indicator `X` (1 while an automaton is in a state, else 0) and
//! an instant `t`:
//!
//! - the mean value is `E[X(t)]`, the probability of being in the state
//!   at `t`; at an instant where a transition fires, the state after it
//!   is read;
//! - the sojourn time is `E[∫_0^t X(s) ds]`, the expected time spent in
//!   the state by `t`;
//! - the mean time to the target is `E[min(T, horizon)]`, `T` the date
//!   the target is reached (infinite when it is not), which is the
//!   sojourn time of "the target is not reached yet" at the horizon.
//!
//! # How a node contributes
//!
//! Between two firings the discrete state does not change, so a state
//! indicator is constant on a node. A node adds its value, weighted by
//! the mass in it, to the mean at every instant and to the sojourn time,
//! from the instant its mass enters it until the instant each child's
//! mass enters the child. Mass that never leaves (a leaf, the target, the
//! survival mass a discretised window drops at the horizon) stays until
//! the end.
//!
//! An entry instant is described by its [`Profile`] over the requested
//! instants and the horizon: the probability it has happened by `t`, and
//! the integral of that probability over `[0, t]`. The discretised driver
//! knows every instant (a child fires at its cell's representative
//! instant, or time runs to the window's end), so its profiles are steps.
//! The exact driver enters a node after the sum of the exponential
//! sojourns along its path, so its profiles are phase-type distribution
//! functions and their integrals
//! ([`raichu_numeric::PhaseTypeAccumulator::absorption_profile`]), each
//! with a guaranteed error bound by which the reported bounds are
//! widened.
//!
//! # Bounds
//!
//! A cut-off makes a child's mass unknown from the instant it would have
//! entered the child. The lower bound counts that mass as out of every
//! state (0), the upper bound as in it (1), from that instant on: the
//! same construction as the bounds on the target probability, so at the
//! horizon the gap of a mean value is at most the gap of the target
//! probability. On the discretised driver both bounds are bounds on the
//! discretised model, and the discretisation error is estimated by
//! refinement, value by value.

use raichu_core::compile::CIndicatorTarget;
use raichu_core::{CompiledModel, Engine, EngineError};
use raichu_expr::Value;
use raichu_numeric::ProfilePoint;
use serde::{Deserialize, Serialize};

use crate::result::relative_gap;
use crate::walk::invalid;

/// The indicators a caller asks an exploration for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndicatorRequest {
    /// Names of model indicators, each a state indicator, each once. May
    /// be empty: the mean time to the target is computed anyway.
    pub names: Vec<String>,
    /// Instants of the mean values and sojourn times, strictly
    /// increasing, in `[0, horizon]`, at least one.
    pub instants: Vec<f64>,
}

/// A value with its bounds on the explored model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundedValue {
    /// Lower bound: every mass a cut-off discarded counted as out of the
    /// state.
    pub lower: f64,
    /// Upper bound: every mass a cut-off discarded counted as in the
    /// state.
    pub upper: f64,
    /// Discretisation error estimate by refinement, `max(|lower_2K -
    /// lower_K|, |upper_2K - upper_K|)`; `None` when no refinement was
    /// made. An estimate, not a bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_estimate: Option<f64>,
}

/// The values of one indicator, one entry per requested instant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndicatorValues {
    /// The indicator's name.
    pub name: String,
    /// Mean value (a probability) at each instant.
    pub mean: Vec<BoundedValue>,
    /// Sojourn time by each instant, in the model's time unit.
    pub sojourn: Vec<BoundedValue>,
    /// `true` when, at some instant, the relative gap `(upper - lower) /
    /// upper` of the mean value or of the sojourn time exceeds the
    /// result's gap tolerance.
    pub inconclusive: bool,
}

/// What an exploration with an [`IndicatorRequest`] reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExploredIndicators {
    /// The requested instants.
    pub instants: Vec<f64>,
    /// One entry per requested indicator, in request order.
    pub values: Vec<IndicatorValues>,
    /// `E[min(T, horizon)]`, `T` the date the target is reached, in the
    /// model's time unit.
    pub time_to_target: BoundedValue,
}

/// A request resolved on a compiled model.
#[derive(Debug, Clone)]
pub(crate) struct IndicatorPlan {
    names: Vec<String>,
    /// Compiled indicator index of each requested indicator.
    indices: Vec<usize>,
    instants: Vec<f64>,
    horizon: f64,
}

impl IndicatorPlan {
    /// Resolve `request` on `model` at `horizon`, refusing an unknown or
    /// repeated name, an indicator that is not a state indicator, and
    /// instants that are not strictly increasing in `[0, horizon]`.
    pub(crate) fn new(
        model: &CompiledModel,
        request: &IndicatorRequest,
        horizon: f64,
    ) -> Result<Self, EngineError> {
        let mut indices = Vec::with_capacity(request.names.len());
        for (k, name) in request.names.iter().enumerate() {
            if request.names[..k].contains(name) {
                return Err(invalid(
                    "indicators",
                    format!("indicator `{name}` is requested twice"),
                ));
            }
            let Some(idx) = model.indicators.iter().position(|i| &i.name == name) else {
                let known: Vec<&str> = model.indicators.iter().map(|i| i.name.as_str()).collect();
                return Err(invalid(
                    "indicators",
                    format!(
                        "no indicator named `{name}` in the model (declared: [{}])",
                        known.join(", ")
                    ),
                ));
            };
            if !matches!(model.indicators[idx].target, CIndicatorTarget::State(..)) {
                return Err(invalid(
                    "indicators",
                    format!(
                        "indicator `{name}` does not observe an automaton state: an \
                         exploration reports state indicators only, whose value is 0 or 1 \
                         and constant between two firings (an attribute or a threshold on \
                         one may move between them, and its bounds would not hold)"
                    ),
                ));
            }
            indices.push(idx);
        }
        let instants = &request.instants;
        if instants.is_empty() {
            return Err(invalid(
                "instants",
                "at least one instant is needed".to_owned(),
            ));
        }
        for (k, &t) in instants.iter().enumerate() {
            if t.is_nan() || !(0.0..=horizon).contains(&t) {
                return Err(invalid(
                    "instants",
                    format!("an instant lies in [0, {horizon}] (the horizon), got {t}"),
                ));
            }
            if k > 0 && t <= instants[k - 1] {
                return Err(invalid(
                    "instants",
                    format!(
                        "instants are strictly increasing, got {} then {t}",
                        instants[k - 1]
                    ),
                ));
            }
        }
        Ok(IndicatorPlan {
            names: request.names.clone(),
            indices,
            instants: instants.clone(),
            horizon,
        })
    }

    /// The value of every requested indicator at the engine's current
    /// state, written into `out`.
    pub(crate) fn read(&self, engine: &Engine<'_>, out: &mut Vec<bool>) {
        out.clear();
        out.extend(
            self.indices.iter().map(
                |&idx| matches!(engine.indicator_value(idx), Some(Value::Float(v)) if v > 0.5),
            ),
        );
    }

    /// The points every profile covers: the requested instants, then the
    /// horizon when it is not the last of them (it carries the mean time
    /// to the target).
    fn points(&self) -> impl Iterator<Item = f64> + '_ {
        let extra = (self.instants.last() != Some(&self.horizon)).then_some(self.horizon);
        self.instants.iter().copied().chain(extra)
    }

    /// The instants and the horizon, as a profile is computed over them.
    pub(crate) fn point_list(&self) -> Vec<f64> {
        self.points().collect()
    }

    /// The profile of a known entry instant `at`.
    pub(crate) fn deterministic(&self, at: f64) -> Profile {
        Profile(
            self.points()
                .map(|t| ProfilePoint {
                    cdf: if at <= t { 1.0 } else { 0.0 },
                    cdf_error: 0.0,
                    integral: (t - at).max(0.0),
                    integral_error: 0.0,
                })
                .collect(),
        )
    }

    /// Empty sums for this plan.
    pub(crate) fn sums(&self) -> IndicatorSums {
        let (n, k) = (self.indices.len(), self.instants.len());
        IndicatorSums {
            mean: vec![0.0; n * k],
            sojourn: vec![0.0; n * k],
            unknown_mean: vec![0.0; k],
            unknown_sojourn: vec![0.0; k],
            error_mean: vec![0.0; k],
            error_sojourn: vec![0.0; k],
            ..IndicatorSums::default()
        }
    }

    /// Mass `weight` in a state where the requested indicators read
    /// `values` and the target is `reached` or not, from its `entry` until
    /// its `exit` (`None` for mass that never leaves).
    pub(crate) fn occupy(
        &self,
        sums: &mut IndicatorSums,
        values: &[bool],
        reached: bool,
        weight: f64,
        entry: &Profile,
        exit: Option<&Profile>,
    ) {
        if weight <= 0.0 {
            return;
        }
        let k = self.instants.len();
        let none = ProfilePoint {
            cdf: 0.0,
            cdf_error: 0.0,
            integral: 0.0,
            integral_error: 0.0,
        };
        let at = |j: usize| {
            let e = entry.0[j];
            let x = exit.map_or(none, |exit| exit.0[j]);
            (
                e.cdf - x.cdf,
                e.cdf_error + x.cdf_error,
                e.integral - x.integral,
                e.integral_error + x.integral_error,
            )
        };
        for j in 0..k {
            let (present, present_error, covered, covered_error) = at(j);
            sums.error_mean[j] += weight * present_error;
            sums.error_sojourn[j] += weight * covered_error;
            for (i, _) in values.iter().enumerate().filter(|(_, &v)| v) {
                sums.mean[i * k + j] += weight * present;
                sums.sojourn[i * k + j] += weight * covered;
            }
        }
        if !reached {
            let (_, _, covered, covered_error) = at(entry.0.len() - 1);
            sums.time_to_target += weight * covered;
            sums.error_time_to_target += weight * covered_error;
        }
    }

    /// Mass `weight` whose state is unknown from its entry `from` on (a
    /// cut-off discarded it).
    pub(crate) fn unknown(&self, sums: &mut IndicatorSums, weight: f64, from: &Profile) {
        if weight <= 0.0 {
            return;
        }
        for (j, point) in from.0.iter().take(self.instants.len()).enumerate() {
            sums.unknown_mean[j] += weight * point.cdf;
            sums.unknown_sojourn[j] += weight * point.integral;
            sums.error_mean[j] += weight * point.cdf_error;
            sums.error_sojourn[j] += weight * point.integral_error;
        }
        if let Some(point) = from.0.last() {
            sums.unknown_time_to_target += weight * point.integral;
            sums.error_time_to_target += weight * point.integral_error;
        }
    }

    /// The reported values of `sums`, flagged inconclusive against
    /// `gap_tolerance`, without error estimates. Each bound is widened by
    /// the numerical error bound of what it sums (zero for a discretised
    /// exploration, whose instants are known).
    pub(crate) fn finish(&self, sums: &IndicatorSums, gap_tolerance: f64) -> ExploredIndicators {
        let k = self.instants.len();
        let bounded = |known: f64, unknown: f64, error: f64| BoundedValue {
            lower: (known - error).max(0.0),
            upper: known + unknown + error,
            error_estimate: None,
        };
        let values = self
            .names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let mean: Vec<BoundedValue> = (0..k)
                    .map(|j| {
                        bounded(
                            sums.mean[i * k + j],
                            sums.unknown_mean[j],
                            sums.error_mean[j],
                        )
                    })
                    .collect();
                let sojourn: Vec<BoundedValue> = (0..k)
                    .map(|j| {
                        bounded(
                            sums.sojourn[i * k + j],
                            sums.unknown_sojourn[j],
                            sums.error_sojourn[j],
                        )
                    })
                    .collect();
                let inconclusive = mean
                    .iter()
                    .chain(&sojourn)
                    .any(|v| relative_gap(v.lower, v.upper) > gap_tolerance);
                IndicatorValues {
                    name: name.clone(),
                    mean,
                    sojourn,
                    inconclusive,
                }
            })
            .collect();
        ExploredIndicators {
            instants: self.instants.clone(),
            values,
            time_to_target: bounded(
                sums.time_to_target,
                sums.unknown_time_to_target,
                sums.error_time_to_target,
            ),
        }
    }
}

/// When a mass enters a node, as a profile over the plan's points
/// ([`IndicatorPlan::point_list`]): at each point `t`, the probability it
/// has entered by `t` and the integral of that probability over `[0, t]`,
/// with their error bounds.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Profile(pub(crate) Vec<ProfilePoint>);

/// Running sums of one worker, merged in exploration order.
#[derive(Debug, Clone, Default)]
pub(crate) struct IndicatorSums {
    /// Known mean value, indicator-major (`i * instants + j`).
    mean: Vec<f64>,
    /// Known sojourn time, same layout.
    sojourn: Vec<f64>,
    /// Unknown mass at each instant (the same for every indicator).
    unknown_mean: Vec<f64>,
    /// Integral of the unknown mass up to each instant.
    unknown_sojourn: Vec<f64>,
    /// Bound on the numerical error of the sums at each instant (the same
    /// for every indicator, so a bound for each).
    error_mean: Vec<f64>,
    error_sojourn: Vec<f64>,
    time_to_target: f64,
    unknown_time_to_target: f64,
    error_time_to_target: f64,
}

impl IndicatorSums {
    pub(crate) fn merge(&mut self, other: &IndicatorSums) {
        let add = |a: &mut Vec<f64>, b: &[f64]| {
            if a.is_empty() {
                a.extend_from_slice(b);
            } else {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
            }
        };
        add(&mut self.mean, &other.mean);
        add(&mut self.sojourn, &other.sojourn);
        add(&mut self.unknown_mean, &other.unknown_mean);
        add(&mut self.unknown_sojourn, &other.unknown_sojourn);
        add(&mut self.error_mean, &other.error_mean);
        add(&mut self.error_sojourn, &other.error_sojourn);
        self.time_to_target += other.time_to_target;
        self.unknown_time_to_target += other.unknown_time_to_target;
        self.error_time_to_target += other.error_time_to_target;
    }
}

/// Attach to `fine` the refinement estimate against `base`, value by value.
pub(crate) fn attach_estimates(fine: &mut ExploredIndicators, base: &ExploredIndicators) {
    let estimate = |f: &mut BoundedValue, b: &BoundedValue| {
        f.error_estimate = Some((f.lower - b.lower).abs().max((f.upper - b.upper).abs()));
    };
    for (f, b) in fine.values.iter_mut().zip(&base.values) {
        for (fv, bv) in f.mean.iter_mut().zip(&b.mean) {
            estimate(fv, bv);
        }
        for (fv, bv) in f.sojourn.iter_mut().zip(&b.sojourn) {
            estimate(fv, bv);
        }
    }
    estimate(&mut fine.time_to_target, &base.time_to_target);
}
