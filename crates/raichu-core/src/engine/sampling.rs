//! Indicator sampling: the one reading of an indicator target, the
//! change-point series recorded at discrete epochs, and the dense samples
//! flushed at the schedule instants.

use super::*;

/// What an indicator records at the current instant.
///
/// The one reading of a [`CIndicatorTarget`], shared by the three sites
/// that record one: the change-point series, the dense-sample flush, and
/// the solver's own sample callback inside a continuous segment. Three
/// copies of the rule is three chances for a target to be honoured on one
/// recording path and dropped on another, which is the class of silence
/// [`CIndicatorTarget::Predicate`] exists to end.
///
/// Infallible on purpose: none of the three callers can report an error,
/// and none has to. Kind compatibility is refused at model build
/// (`ModelError::IndicatorPredicateKind`), so the only comparison left
/// without an answer is one against a NaN, settled below the way IEEE
/// settles it.
pub(super) fn indicator_value(
    target: &CIndicatorTarget,
    vars: &[Value],
    states: &[StateIdx],
) -> Value {
    match *target {
        CIndicatorTarget::Var(idx) => vars[idx],
        CIndicatorTarget::State(aut, state) => {
            Value::Float(if states[aut] == state { 1.0 } else { 0.0 })
        }
        CIndicatorTarget::Predicate(idx, cmp, bound) => {
            Value::Bool(predicate_holds(vars[idx], cmp, bound))
        }
    }
}

impl<'m> Engine<'m> {
    pub(super) fn record_indicators(&mut self) {
        for (indicator, series) in self
            .model
            .indicators
            .iter()
            .zip(self.indicator_series.iter_mut())
        {
            let value = indicator_value(&indicator.target, &self.vars, &self.states);
            let changed = series.points.last().is_none_or(|(_, last)| *last != value);
            if changed {
                series.points.push((self.time, value));
            }
        }
    }

    /// Record pending sample instants strictly before `t` with the
    /// *current* (pre-jump) state: piecewise-constant hold for the
    /// discrete-only case.
    pub(super) fn flush_samples_before(&mut self, t: f64) {
        self.flush_samples(t, false);
    }

    /// Record pending sample instants up to and including `t` (end of
    /// run).
    pub(super) fn flush_samples_through(&mut self, t: f64) {
        self.flush_samples(t, true);
    }

    fn flush_samples(&mut self, t: f64, inclusive: bool) {
        while self.sample_cursor < self.config.samples.len() {
            let s = self.config.samples[self.sample_cursor];
            let due = if inclusive { s <= t } else { s < t };
            if !due {
                break;
            }
            for (indicator, series) in self.model.indicators.iter().zip(self.sampled.iter_mut()) {
                let value = indicator_value(&indicator.target, &self.vars, &self.states);
                series.points.push((s, value));
            }
            self.sample_cursor += 1;
        }
    }
}
