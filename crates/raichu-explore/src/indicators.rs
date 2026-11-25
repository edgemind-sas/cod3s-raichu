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
//! A node of the tree is entered at a known instant with a known path
//! probability, and each of its children leaves it at a known instant
//! (the discretised driver fires a child at its cell's representative
//! instant, or lets time run to the window's end). Between those
//! instants the discrete state does not change, so a state indicator is
//! constant on the node: the node adds its value, weighted by the mass
//! still in it, to the mean at every instant it covers, and to the
//! sojourn time over the time it covers. Mass that never leaves (a leaf,
//! the target, the survival mass dropped at the horizon) stays until the
//! end.
//!
//! # Bounds
//!
//! A cut-off makes a child's mass unknown from the instant it would have
//! entered the child. The lower bound counts that mass as out of every
//! state (0), the upper bound as in it (1), from that instant on: the
//! same construction as the bounds on the target probability, so at the
//! horizon the gap of a mean value is at most the gap of the target
//! probability. Both bounds are bounds on the explored (discretised)
//! model; the discretisation error is estimated by refinement, value by
//! value.

use raichu_core::compile::CIndicatorTarget;
use raichu_core::{CompiledModel, Engine, EngineError};
use raichu_expr::Value;
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

    /// Empty sums for this plan.
    pub(crate) fn sums(&self) -> IndicatorSums {
        let (n, k) = (self.indices.len(), self.instants.len());
        IndicatorSums {
            mean: vec![0.0; n * k],
            sojourn: vec![0.0; n * k],
            unknown_mean: vec![0.0; k],
            unknown_sojourn: vec![0.0; k],
            time_to_target: 0.0,
            unknown_time_to_target: 0.0,
        }
    }

    /// Mass `weight` in a state where the requested indicators read
    /// `values` and the target is `reached` or not, over `[from, to)`
    /// (`to` infinite for mass that never leaves).
    pub(crate) fn occupy(
        &self,
        sums: &mut IndicatorSums,
        values: &[bool],
        reached: bool,
        weight: f64,
        from: f64,
        to: f64,
    ) {
        if weight <= 0.0 || to <= from {
            return;
        }
        let k = self.instants.len();
        for (j, &t) in self.instants.iter().enumerate() {
            if t < from {
                continue;
            }
            let present = t < to;
            let covered = to.min(t) - from;
            for (i, _) in values.iter().enumerate().filter(|(_, &v)| v) {
                if present {
                    sums.mean[i * k + j] += weight;
                }
                sums.sojourn[i * k + j] += weight * covered;
            }
        }
        if !reached && from < self.horizon {
            sums.time_to_target += weight * (to.min(self.horizon) - from);
        }
    }

    /// Mass `weight` whose state is unknown from `from` on (a cut-off
    /// discarded it).
    pub(crate) fn unknown(&self, sums: &mut IndicatorSums, weight: f64, from: f64) {
        if weight <= 0.0 {
            return;
        }
        for (j, &t) in self.instants.iter().enumerate() {
            if t >= from {
                sums.unknown_mean[j] += weight;
                sums.unknown_sojourn[j] += weight * (t - from);
            }
        }
        if from < self.horizon {
            sums.unknown_time_to_target += weight * (self.horizon - from);
        }
    }

    /// The reported values of `sums`, flagged inconclusive against
    /// `gap_tolerance`, without error estimates.
    pub(crate) fn finish(&self, sums: &IndicatorSums, gap_tolerance: f64) -> ExploredIndicators {
        let k = self.instants.len();
        let bounded = |lower: f64, unknown: f64| BoundedValue {
            lower,
            upper: lower + unknown,
            error_estimate: None,
        };
        let values = self
            .names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let mean: Vec<BoundedValue> = (0..k)
                    .map(|j| bounded(sums.mean[i * k + j], sums.unknown_mean[j]))
                    .collect();
                let sojourn: Vec<BoundedValue> = (0..k)
                    .map(|j| bounded(sums.sojourn[i * k + j], sums.unknown_sojourn[j]))
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
            time_to_target: bounded(sums.time_to_target, sums.unknown_time_to_target),
        }
    }
}

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
    time_to_target: f64,
    unknown_time_to_target: f64,
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
        self.time_to_target += other.time_to_target;
        self.unknown_time_to_target += other.unknown_time_to_target;
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
