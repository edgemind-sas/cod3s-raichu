//! Deferred-draw mode ([`StochasticDates::Deferred`]): age bookkeeping
//! of the stochastic transitions armed without a drawn date, the hazard
//! probe, the firing of a deferred transition at a caller-chosen date, and
//! the schedule refresh that keeps the deferred set current.

use super::*;

/// Active age of a stochastic clock, reused by deferred draws and
/// independent restarts of drawn dates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DeferredAge {
    /// Instant the transition was armed (kept across `resume` pauses).
    pub(super) armed_at: f64,
    /// Age banked by the running stretches that ended before `since`.
    pub(super) banked: f64,
    /// Start of the current running stretch (or of the pause).
    pub(super) since: f64,
    /// `false` while paused by a `resume` interruption.
    pub(super) running: bool,
}

impl DeferredAge {
    /// Age at `now`: time spent armed and running.
    pub(super) fn age(&self, now: f64) -> f64 {
        if self.running {
            self.banked + (now - self.since)
        } else {
            self.banked
        }
    }
}

impl<'m> Engine<'m> {
    /// **Deferred draws**: every stochastic transition armed without a
    /// date ([`StochasticDates::Deferred`]), ascending by index, with its
    /// law, arming instant, age and, for a state-dependent exponential,
    /// cumulative hazard at the current instant. Paused ones (`resume`
    /// interruption, guard false) are listed with `paused` set. Always
    /// empty in drawn mode.
    #[must_use]
    pub fn deferred(&self) -> Vec<DeferredTransition> {
        self.deferred
            .iter()
            .enumerate()
            .filter_map(|(idx, age)| {
                let age = (*age)?;
                let transition = &self.model.transitions[idx];
                Some(DeferredTransition {
                    index: idx,
                    transition: transition.name.clone(),
                    law: transition.distrib.clone(),
                    armed_at: age.armed_at,
                    age: age.age(self.time),
                    paused: !age.running,
                    hazard: self.deferred_hazard(idx),
                })
            })
            .collect()
    }

    /// **Deferred draws**: advance deterministically, firing nothing, up
    /// to the first of the next date-scheduled transition, a watched
    /// boundary crossing, the horizon `t_max`, and `limit`, and record
    /// along the way the cumulative hazard of every running deferred
    /// transition whose rate varies continuously.
    ///
    /// The continuous state is integrated exactly as a run would
    /// integrate it, so the recorded hazards `H(t) = ∫ λ(x(u)) du` are
    /// those of the trajectory in which no stochastic event occurs: the
    /// quantity whose inverse gives the date of the first one. The
    /// samples are dense (every accepted solver step end, plus the
    /// solver's interior scan points), start at the current instant and
    /// end at the stop instant. Piecewise-constant and fixed laws need no
    /// samples: read them with [`Engine::deferred`] and the law.
    ///
    /// The engine is left at the stop instant, the event that stopped it
    /// unfired: the explorer restores a [`Snapshot`] afterwards, or fires
    /// a deferred transition from there. Errors with
    /// [`EngineError::NotDeferredMode`] in drawn mode, and with
    /// [`EngineError::DateInPast`] for a `limit` before the current time
    /// or not a number (the engine is unchanged in both cases).
    pub fn probe_deferred(&mut self, limit: f64) -> Result<DeferredProbe, EngineError> {
        if self.config.stochastic_dates != StochasticDates::Deferred {
            return Err(EngineError::NotDeferredMode {
                operation: "probe_deferred".to_owned(),
            });
        }
        if limit.is_nan() || limit < self.time {
            return Err(EngineError::DateInPast {
                transition: "<probe>".to_owned(),
                date: limit,
                time: self.time,
            });
        }
        let transitions = self.deferred_continuous();
        if !transitions.is_empty() {
            self.hazard_trace = Some(vec![self.hazard_sample(&transitions)]);
        }
        let outcome = self.probe_inner(limit);
        let trace = self.hazard_trace.take();
        let (stop, reason) = outcome?;
        let mut samples = trace.unwrap_or_default();
        if !transitions.is_empty() && samples.last().is_none_or(|last| last.time < stop) {
            samples.push(self.hazard_sample(&transitions));
        }
        Ok(DeferredProbe {
            stop,
            reason,
            transitions,
            samples,
        })
    }

    /// The deterministic advance of [`Engine::probe_deferred`].
    pub(super) fn probe_inner(&mut self, limit: f64) -> Result<(f64, ProbeStop), EngineError> {
        if let Some(watched) = self.immediate_watched()? {
            return Ok((self.time, ProbeStop::Watched { index: watched }));
        }
        let t_max = self.config.t_max;
        let (target, reason) = match self.next_pending() {
            Some((idx, date)) if date <= limit && date <= t_max => {
                (date.max(self.time), ProbeStop::Deterministic { index: idx })
            }
            _ if t_max < limit => (t_max.max(self.time), ProbeStop::Horizon),
            _ => (limit, ProbeStop::Limit),
        };
        if self.needs_integration() && target > self.time && target.is_finite() {
            if let Some(watched) = self.advance_continuous(target)? {
                return Ok((self.time, ProbeStop::Watched { index: watched }));
            }
        }
        if target > self.time {
            self.flush_samples_before(target);
            self.time = target;
            self.note_time_change();
        }
        Ok((target, reason))
    }

    /// The current cumulative hazards of `transitions` as one sample.
    fn hazard_sample(&self, transitions: &[TransIdx]) -> HazardSample {
        HazardSample {
            time: self.time,
            hazards: transitions
                .iter()
                .map(|&idx| self.deferred_hazard(idx).unwrap_or(0.0))
                .collect(),
        }
    }

    /// **Deferred draws**: fire the deferred transition `trans_idx` at the
    /// instant `t`, after advancing deterministically to it (continuous
    /// evolution included; no other transition fires on the way). The
    /// other deferred transitions keep their bookkeeping and age up to
    /// `t`. `branch` forces the destination by position in the compiled
    /// target list, as in [`Engine::fire_now`] (a stochastic law has one).
    ///
    /// `t` must not come after an event the engine would have to process
    /// first: the next date-scheduled transition (firing exactly at its
    /// date is allowed, and happens before it), a watched boundary
    /// crossing located on the way (or already holding), or the horizon.
    /// Such a request is refused with
    /// [`EngineError::DeferredFiringTooLate`], naming that event and its
    /// instant, and the engine is left unchanged.
    ///
    /// Also errors with [`EngineError::UnknownTransition`] for an index
    /// out of range, [`EngineError::NotFireable`] if the transition is not
    /// armed and running in deferred mode (not armed, paused by a
    /// `resume` interruption, drawn mode, or not stochastic),
    /// [`EngineError::ForcedBranchOutOfRange`] for an invalid `branch`,
    /// and [`EngineError::DateInPast`] for `t` before the current time or
    /// not finite; the engine is unchanged in all these cases.
    pub fn fire_deferred_at(
        &mut self,
        trans_idx: TransIdx,
        t: f64,
        branch: Option<usize>,
    ) -> Result<Event, EngineError> {
        self.operator_transaction(|engine| engine.fire_deferred_inner(trans_idx, t, branch))
    }

    fn fire_deferred_inner(
        &mut self,
        trans_idx: TransIdx,
        t: f64,
        branch: Option<usize>,
    ) -> Result<Event, EngineError> {
        let Some(transition) = self.model.transitions.get(trans_idx) else {
            return Err(EngineError::UnknownTransition {
                transition: format!("<index {trans_idx}>"),
            });
        };
        let name = transition.name.clone();
        if !self.deferred_running(trans_idx) {
            return Err(EngineError::NotFireable {
                transition: name,
                time: self.time,
            });
        }
        let forced = match branch {
            Some(branch) => Some(self.resolve_branch(trans_idx, branch)?),
            None => None,
        };
        if !t.is_finite() || t < self.time {
            return Err(EngineError::DateInPast {
                transition: name,
                date: t,
                time: self.time,
            });
        }
        let too_late = |limit: f64, cause: String| EngineError::DeferredFiringTooLate {
            transition: name.clone(),
            date: t,
            limit,
            cause,
        };
        if let Some((idx, date)) = self.next_pending() {
            if t > date {
                let cause = format!(
                    "the next deterministic event `{}`",
                    self.model.transitions[idx].name
                );
                return Err(too_late(date, cause));
            }
        }
        if t > self.config.t_max {
            return Err(too_late(self.config.t_max, "the horizon".to_owned()));
        }
        if t > self.time {
            if let Some(watched) = self.immediate_watched()? {
                let cause = format!(
                    "the watched transition `{}`",
                    self.model.transitions[watched].name
                );
                return Err(too_late(self.time, cause));
            }
            if self.needs_integration() {
                // The crossing is known only once integrated: rewind on
                // refusal so the engine is left unchanged.
                let before = self.try_snapshot()?;
                match self.advance_continuous(t) {
                    Ok(None) => {}
                    Ok(Some(watched)) => {
                        let at = self.time;
                        self.try_restore(&before)?;
                        let cause = format!(
                            "the watched transition `{}`",
                            self.model.transitions[watched].name
                        );
                        return Err(too_late(at, cause));
                    }
                    Err(error) => {
                        self.try_restore(&before)?;
                        return Err(error);
                    }
                }
            }
            self.flush_samples_before(t);
            self.time = t;
            self.note_time_change();
            self.watched_streak = (t, 0);
        }
        self.fire(trans_idx, forced)
    }

    /// Deferred-mode counterpart of the stochastic arms of
    /// [`Engine::refresh_schedule`] for one stochastic transition: arm it
    /// without a date and without a draw, and keep its age (and, for a
    /// state-dependent rate, its cumulative hazard) under the
    /// interruption policy. See [`StochasticDates::Deferred`].
    pub(super) fn refresh_deferred(
        &mut self,
        trans_idx: TransIdx,
        in_source: bool,
        guard_ok: bool,
    ) -> Result<(), EngineError> {
        let model = self.model;
        let transition = &model.transitions[trans_idx];
        let now = self.time;
        let (rate, continuous) = match &transition.distrib {
            CLaw::ExpVar { rate, continuous } => (Some(rate), *continuous),
            _ => (None, false),
        };
        let mut dropped = None;
        match self.deferred[trans_idx] {
            Some(_) if !in_source => {
                self.deferred[trans_idx] = None;
                self.hazards[trans_idx] = None;
                dropped = Some(DropReason::SourceLeft);
            }
            None if in_source && guard_ok => {
                // `schedule_stochastic`, deferred: no draw, no date.
                self.deferred[trans_idx] = Some(DeferredAge {
                    armed_at: now,
                    banked: 0.0,
                    since: now,
                    running: true,
                });
                if let Some(rate) = rate {
                    let lambda = self.eval_rate(trans_idx, rate)?;
                    self.hazards[trans_idx] = Some(Hazard {
                        threshold: f64::INFINITY,
                        accumulated: 0.0,
                        rate: lambda,
                        since: now,
                    });
                }
                if self.config.journal {
                    self.journal.push(JournalRecord::TransitionScheduled {
                        time: now,
                        transition: transition.name.clone(),
                        firing_at: f64::INFINITY,
                    });
                }
            }
            None => {}
            Some(age)
                if !guard_ok
                    && transition.on_interruption != raichu_model::InterruptionPolicy::Continue =>
            {
                // `drop_disabled`, deferred: reset forgets the age,
                // resume freezes it (a paused one stays paused).
                if !age.running {
                    return Ok(());
                }
                if transition.on_interruption == raichu_model::InterruptionPolicy::Resume {
                    self.deferred[trans_idx] = Some(DeferredAge {
                        banked: age.age(now),
                        since: now,
                        running: false,
                        ..age
                    });
                    if let Some(hazard) = self.hazards[trans_idx].as_mut() {
                        if !continuous {
                            hazard.accumulated += hazard.rate * (now - hazard.since);
                        }
                        hazard.since = now;
                    }
                    dropped = Some(DropReason::GuardPaused);
                } else {
                    self.deferred[trans_idx] = None;
                    self.hazards[trans_idx] = None;
                    dropped = Some(DropReason::GuardFalse);
                }
            }
            #[allow(clippy::float_cmp)] // λ is re-evaluated exactly
            Some(age) => {
                // Guard true, or `continue` riding through a false guard.
                if !age.running {
                    // A `resume` pause ends: the age runs again.
                    self.deferred[trans_idx] = Some(DeferredAge {
                        since: now,
                        running: true,
                        ..age
                    });
                    if let Some(rate) = rate {
                        let lambda = self.eval_rate(trans_idx, rate)?;
                        if let Some(hazard) = self.hazards[trans_idx].as_mut() {
                            hazard.rate = lambda;
                            hazard.since = now;
                        }
                    }
                } else if let (Some(rate), false) = (rate, continuous) {
                    // `reschedule_modifiable`, deferred: bank the hazard
                    // accrued at the previous rate, then carry on at λ.
                    let lambda = self.eval_rate(trans_idx, rate)?;
                    if let Some(hazard) = self.hazards[trans_idx].as_mut() {
                        if lambda != hazard.rate {
                            hazard.accumulated += hazard.rate * (now - hazard.since);
                            hazard.since = now;
                            hazard.rate = lambda;
                        }
                    }
                }
            }
        }
        if let (Some(reason), true) = (dropped, self.config.journal) {
            self.journal.push(JournalRecord::TransitionDropped {
                time: now,
                transition: transition.name.clone(),
                reason,
            });
        }
        Ok(())
    }

    /// Whether `trans_idx` is armed in deferred mode with its age
    /// running (not paused by a `resume` interruption).
    pub(super) fn deferred_running(&self, trans_idx: TransIdx) -> bool {
        self.deferred[trans_idx].is_some_and(|age| age.running)
    }

    /// Whether `trans_idx` is armed: date-scheduled (`pending`) or
    /// running in deferred mode.
    pub(super) fn is_armed(&self, trans_idx: TransIdx) -> bool {
        self.pending[trans_idx].is_some() || self.deferred_running(trans_idx)
    }

    /// Cumulative hazard of a deferred state-dependent rate at the
    /// current instant (`None` for any other transition).
    fn deferred_hazard(&self, trans_idx: TransIdx) -> Option<f64> {
        let age = self.deferred[trans_idx]?;
        let hazard = self.hazards[trans_idx]?;
        let continuous = matches!(
            self.model.transitions[trans_idx].distrib,
            CLaw::ExpVar {
                continuous: true,
                ..
            }
        );
        if continuous || !age.running {
            // A continuous hazard is committed by the integrator at every
            // segment end, which is where the clock stands.
            Some(hazard.accumulated)
        } else {
            Some(hazard.accumulated + hazard.rate * (self.time - hazard.since))
        }
    }

    /// The running deferred transitions whose rate varies continuously,
    /// ascending: the hazard slots `integrate_to` monitors in deferred
    /// mode, in its order.
    fn deferred_continuous(&self) -> Vec<TransIdx> {
        self.continuous_rates
            .iter()
            .copied()
            .filter(|&idx| self.deferred_running(idx) && self.hazards[idx].is_some())
            .collect()
    }
}

#[cfg(test)]
mod deferred_rng_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{Engine, EngineConfig, StochasticDates};
    use crate::CompiledModel;
    use raichu_model::Model;

    /// Two repairable components with stochastic failure and repair
    /// laws, one of them a state-dependent rate.
    fn model() -> Model {
        let comp = |name: &str, occ: &str, rep: &str| {
            format!(
                r#"{{"name": "{name}", "ports": [], "attributes": [], "equations": [],
                "automata": [{{"name": "fail", "states": ["ok", "nok"], "init": "ok",
                  "transitions": [
                    {{"name": "occ", "source": "ok", "targets": ["nok"], {occ}}},
                    {{"name": "rep", "source": "nok", "targets": ["ok"], {rep}}}]}}]}}"#
            )
        };
        let rate = r#""distrib": "exp", "rate_expr": {"op": "if",
            "cond": {"op": "state_active", "state": {"component": "A", "automaton": "fail", "state": "nok"}},
            "then": {"op": "const", "value": {"kind": "float", "value": 2.0}},
            "otherwise": {"op": "const", "value": {"kind": "float", "value": 0.5}}}"#;
        let json = format!(
            r#"{{"name": "rng", "components": [{}, {}]}}"#,
            comp(
                "A",
                r#""distrib": "weibull", "shape": 2.0, "scale": 3.0"#,
                r#""distrib": "gamma", "shape": 2.0, "scale": 1.0"#
            ),
            comp(
                "B",
                rate,
                r#""distrib": "uniform", "low": 1.0, "high": 2.0"#
            ),
        );
        Model::from_json(&json).unwrap()
    }

    #[test]
    fn operator_commands_leave_rng_untouched_and_restore_hazards() {
        let compiled = CompiledModel::compile(&model()).unwrap();
        let config = EngineConfig {
            seed: 42,
            stochastic_dates: StochasticDates::Operator,
            ..EngineConfig::default()
        };
        let fresh = raichu_rng::replica_rng(config.seed, config.rng_stream);
        let mut engine = Engine::new(&compiled, config).unwrap();
        assert_eq!(engine.rng, fresh);
        engine.set_date("A.fail.occ", 8.0).unwrap();
        let before = engine.snapshot().unwrap();
        engine.advance_operator_to(4.0, 100).unwrap();
        let hazard = engine
            .deferred()
            .into_iter()
            .find(|t| t.transition == "B.fail.occ")
            .unwrap()
            .hazard;
        assert_eq!(hazard, Some(2.0));
        assert_eq!(engine.rng, fresh);
        engine.advance_operator_to(10.0, 100).unwrap();
        assert_eq!(engine.rng, fresh);
        engine.restore(&before).unwrap();
        assert_eq!(
            engine
                .deferred()
                .into_iter()
                .find(|t| t.transition == "B.fail.occ")
                .unwrap()
                .hazard,
            Some(0.0)
        );
        engine.advance_operator_to(4.0, 100).unwrap();
        assert_eq!(
            engine
                .deferred()
                .into_iter()
                .find(|t| t.transition == "B.fail.occ")
                .unwrap()
                .hazard,
            hazard
        );
        assert_eq!(engine.rng, fresh);
    }

    #[test]
    fn operator_choice_consumes_no_draw_before_or_after_explicit_selection() {
        let model = Model::from_json(r#"{"name":"choice_rng","components":[{"name":"C","automata":[{"name":"A","states":["pending","ok","failed"],"init":"pending","transitions":[{"name":"resolve","source":"pending","targets":["ok","failed"],"distrib":"inst","probs":[0.7]}]}]}]}"#).unwrap();
        let compiled = CompiledModel::compile(&model).unwrap();
        let config = EngineConfig {
            seed: 42,
            stochastic_dates: StochasticDates::Operator,
            ..EngineConfig::default()
        };
        let fresh = raichu_rng::replica_rng(config.seed, config.rng_stream);
        let mut engine = Engine::new(&compiled, config).unwrap();
        assert_eq!(
            engine.advance_operator_to(5.0, 100).unwrap().stop,
            super::OperatorStop::Choice
        );
        assert_eq!(engine.rng, fresh);
        engine.fire_named_to("C.A.resolve", "failed").unwrap();
        engine.advance_operator_to(5.0, 100).unwrap();
        assert_eq!(engine.rng, fresh);
    }

    #[test]
    fn operator_state_dependent_rate_change_preserves_explicit_deadline() {
        let compiled = CompiledModel::compile(&model()).unwrap();
        let mut engine = Engine::new(
            &compiled,
            EngineConfig {
                stochastic_dates: StochasticDates::Operator,
                ..EngineConfig::default()
            },
        )
        .unwrap();
        engine.set_date("A.fail.occ", 8.0).unwrap();
        engine.set_date("B.fail.occ", 12.0).unwrap();
        assert_eq!(
            engine.advance_operator_to(14.0, 100).unwrap().reached_time,
            8.0
        );
        assert_eq!(
            engine
                .fireable()
                .into_iter()
                .find(|f| f.transition == "B.fail.occ")
                .unwrap()
                .date,
            Some(12.0)
        );
        assert_eq!(
            engine
                .deferred()
                .into_iter()
                .find(|t| t.transition == "B.fail.occ")
                .unwrap()
                .hazard,
            Some(4.0)
        );
        assert_eq!(
            engine.advance_operator_to(14.0, 100).unwrap().reached_time,
            12.0
        );
    }

    #[test]
    fn deferred_arming_consumes_no_random_number() {
        let compiled = CompiledModel::compile(&model()).unwrap();
        let config = EngineConfig {
            seed: 42,
            stochastic_dates: StochasticDates::Deferred,
            ..EngineConfig::default()
        };
        let fresh = raichu_rng::replica_rng(config.seed, config.rng_stream);
        let mut engine = Engine::new(&compiled, config).unwrap();
        assert_eq!(engine.deferred().len(), 2);
        assert_eq!(engine.rng, fresh);
        // Fire both failures: the rate of B changes, both repairs arm.
        let a = compiled
            .transitions
            .iter()
            .position(|t| t.name == "A.fail.occ");
        let b = compiled
            .transitions
            .iter()
            .position(|t| t.name == "B.fail.occ");
        engine.fire_deferred_at(a.unwrap(), 1.0, None).unwrap();
        engine.fire_deferred_at(b.unwrap(), 1.5, None).unwrap();
        assert_eq!(engine.deferred().len(), 2);
        engine.probe_deferred(4.0).unwrap();
        assert_eq!(engine.rng, fresh);

        // The same arming in drawn mode does draw: the check has teeth.
        let drawn = Engine::new(
            &compiled,
            EngineConfig {
                seed: 42,
                ..EngineConfig::default()
            },
        )
        .unwrap();
        assert_ne!(drawn.rng, fresh);
    }
}
