//! Interactive surface: reading the state, listing and firing transitions
//! by hand, forcing dates, snapshots, restore and reset.

use super::*;

impl<'m> Engine<'m> {
    /// Current simulation time.
    #[must_use]
    pub fn current_time(&self) -> f64 {
        self.time
    }

    /// Fire at most one event, without advancing beyond `date`.
    ///
    /// Returns the next event at or before the bound. If none remains,
    /// evolves the continuous state and clock to `date` and returns `None`.
    /// The configured study horizon is preserved, including on errors.
    /// Call [`Engine::advance_to`] at the returned event's date to finish
    /// its simultaneous reactions before observing a completed instant.
    ///
    /// # Errors
    /// Refuses a non-finite bound, a bound before the current time or beyond
    /// the study horizon, and propagates ordinary stepping errors.
    pub fn step_until(&mut self, date: f64) -> Result<Option<Event>, EngineError> {
        if !date.is_finite() || date < self.time || date > self.config.t_max {
            return Err(EngineError::AdvanceTimeInvalid {
                date,
                time: self.time,
                horizon: self.config.t_max,
            });
        }
        let horizon = self.config.t_max;
        self.config.t_max = date;
        let result = (|| {
            let event = self.step()?;
            if event.is_none() {
                // `step` already integrated to its temporary horizon. Pure
                // discrete trajectories still need their clock moved there.
                if self.time < date {
                    self.flush_samples_before(date);
                    self.time = date;
                    self.note_time_change();
                    self.watched_streak = (date, 0);
                }
                self.flush_samples_through(date);
                self.record_indicators();
            }
            Ok(event)
        })();
        self.config.t_max = horizon;
        result
    }

    /// Advance an interactive trajectory through every event dated at or
    /// before `date`, then leave its clock at `date`. The caller may split
    /// continuous evolution at arbitrary communication points.
    pub fn advance_to(&mut self, date: f64) -> Result<(), EngineError> {
        if self.config.stochastic_dates == StochasticDates::Operator {
            return Err(EngineError::InteractivePolicy {
                operation: "advance_to",
                policy: "advance_operator_to in operator mode",
            });
        }
        if !date.is_finite() || date < self.time || date > self.config.t_max {
            return Err(EngineError::AdvanceTimeInvalid {
                date,
                time: self.time,
                horizon: self.config.t_max,
            });
        }
        let horizon = self.config.t_max;
        self.config.t_max = date;
        let result = (|| {
            while self.step()?.is_some() {}
            if self.time < date {
                if self.needs_integration() {
                    if let Some(trans_idx) = self.advance_continuous(date)? {
                        self.note_watched_firing()?;
                        self.fire(trans_idx, None)?;
                        while self.step()?.is_some() {}
                    }
                }
                self.flush_samples_before(date);
                self.time = date;
                self.note_time_change();
                self.watched_streak = (date, 0);
            }
            self.flush_samples_through(date);
            self.record_indicators();
            Ok(())
        })();
        self.config.t_max = horizon;
        result
    }

    /// Write an explicitly declared input at the current communication
    /// point and complete its discrete reaction before returning. The
    /// allowlist comes from the export manifest, not model naming rules.
    pub fn set_input(
        &mut self,
        qualified: &str,
        value: Value,
        allowed_inputs: &[String],
    ) -> Result<(), EngineError> {
        self.operator_transaction(|engine| engine.set_input_inner(qualified, value, allowed_inputs))
    }

    fn set_input_inner(
        &mut self,
        qualified: &str,
        value: Value,
        allowed_inputs: &[String],
    ) -> Result<(), EngineError> {
        if !allowed_inputs.iter().any(|name| name == qualified) {
            return Err(EngineError::InputNotAllowed {
                attribute: qualified.to_owned(),
            });
        }
        let Some(&idx) = self.model.var_index.get(qualified) else {
            return Err(EngineError::InputNotAllowed {
                attribute: qualified.to_owned(),
            });
        };
        if std::mem::discriminant(&self.model.var_init[idx]) != std::mem::discriminant(&value)
            || matches!(value, Value::Float(number) if !number.is_finite())
        {
            return Err(EngineError::InputType {
                attribute: qualified.to_owned(),
            });
        }
        if self.vars[idx] == value {
            return Ok(());
        }
        let old = self.vars[idx];
        self.vars[idx] = value;
        self.note_var_change(idx);
        if self.config.journal {
            self.journal.push(JournalRecord::AttributeChanged {
                time: self.time,
                attribute: qualified.to_owned(),
                old,
                new: value,
                cause: "external.input".to_owned(),
            });
        }
        self.worklist
            .extend(self.model.var_triggers[idx].iter().copied());
        self.run_fixpoint()?;
        self.resolve_flows()?;
        self.refresh_schedule()?;
        self.record_indicators();
        self.check_targets();
        if self.config.stochastic_dates == StochasticDates::Operator {
            self.advance_operator_checked(self.time, usize::MAX)
                .map(|_| ())
        } else {
            self.advance_to(self.time)
        }
    }

    /// Next FMU communication point, or `None` for a native-only model.
    #[must_use]
    pub fn next_fmu_point(&self) -> Option<f64> {
        self.host.as_ref().and_then(|host| host.next_point())
    }

    /// Counted work done so far (see [`WorkCounters`]): the
    /// machine-independent performance units of this run.
    ///
    /// Cumulative over the engine's life. [`Engine::restore`] rewinds the
    /// trajectory, not the work already done, so a rewound-and-replayed
    /// engine reports *more* work than a straight run of the same
    /// trajectory: compare counters between fresh runs.
    #[must_use]
    pub fn work(&self) -> WorkCounters {
        let solver = self.solver.stats();
        WorkCounters {
            solver_steps_accepted: solver.accepted,
            solver_steps_rejected: solver.rejected,
            ..self.work
        }
    }

    /// Value of an attribute by qualified name (`component.attribute`).
    #[must_use]
    pub fn attribute(&self, qualified: &str) -> Option<Value> {
        attribute_of(self.model, &self.vars, qualified)
    }

    /// Current state name of an automaton by qualified name
    /// (`component.automaton`).
    #[must_use]
    pub fn state(&self, qualified: &str) -> Option<&str> {
        state_of(self.model, &self.states, qualified)
    }

    /// **Interactive control**: every currently-armed transition.
    ///
    /// Lists the date-scheduled transitions (delay / inst / stochastic)
    /// with their firing date, the stochastic transitions armed without a
    /// date in deferred mode ([`StochasticDates::Deferred`], date `None`,
    /// paused ones left out), plus the watched transitions armed in
    /// their source state (date = the current instant when their guard
    /// already holds, else `None`: the boundary being located only
    /// during continuous evolution).
    ///
    /// Sorted by date (unlocated watched last), then transition index,
    /// so the first entry is exactly what [`Engine::step`] would fire
    /// next.
    #[must_use]
    pub fn fireable(&self) -> Vec<Fireable> {
        let mut out: Vec<Fireable> = Vec::new();
        for idx in 0..self.pending.len() {
            if let Some(date) = self.scheduled_date(idx) {
                out.push(Fireable {
                    index: idx,
                    transition: self.model.transitions[idx].name.clone(),
                    kind: fireable_kind(&self.model.transitions[idx].distrib),
                    date: Some(date),
                });
            }
        }
        for (idx, age) in self.deferred.iter().enumerate() {
            if age.is_some_and(|age| age.running)
                && self.operator_dates.get(idx).is_none_or(Option::is_none)
            {
                out.push(Fireable {
                    index: idx,
                    transition: self.model.transitions[idx].name.clone(),
                    kind: FireableKind::Stochastic,
                    date: None,
                });
            }
        }
        for &idx in &self.model.watched {
            let transition = &self.model.transitions[idx];
            if self.states[transition.automaton] != transition.source {
                continue;
            }
            // Guard already true ⇒ fireable at the current instant; else
            // its boundary has not been located yet (date unknown). A
            // guard type error is treated as "not fireable now" here; it
            // resurfaces when the transition is actually stepped/fired.
            let date = self
                .is_immediate_watched(idx)
                .unwrap_or(false)
                .then_some(self.time);
            out.push(Fireable {
                index: idx,
                transition: transition.name.clone(),
                kind: FireableKind::Watched,
                date,
            });
        }
        out.sort_by(|a, b| match (a.date, b.date) {
            (Some(x), Some(y)) => x
                .partial_cmp(&y)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.index.cmp(&b.index)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.index.cmp(&b.index),
        });
        out
    }

    /// **Interactive control**: fire the armed transition carrying this
    /// qualified name (see [`Engine::fire_idx`] for the semantics).
    ///
    /// Errors with [`EngineError::UnknownTransition`] if no transition
    /// bears the name, or [`EngineError::NotFireable`] if it is not armed.
    pub fn fire_named(&mut self, name: &str) -> Result<Event, EngineError> {
        let idx = self.transition_index(name)?;
        self.fire_idx_inner(idx, None)
    }

    /// **Interactive control**: fire the armed transition `name`,
    /// **forcing** its destination branch to the state named `to`
    /// (bypassing the RNG / deterministic-branch resolution). This is
    /// what makes a non-deterministic instantaneous branching (or any
    /// stochastic branch) reproducibly testable: the outcome is chosen,
    /// not drawn.
    ///
    /// Errors with [`EngineError::ForcedTargetInvalid`] if `to` is not
    /// one of the transition's declared target states.
    pub fn fire_named_to(&mut self, name: &str, to: &str) -> Result<Event, EngineError> {
        let idx = self.transition_index(name)?;
        let forced = self.resolve_forced(idx, to)?;
        self.fire_idx_inner(idx, Some(forced))
    }

    /// **Interactive control**: fire a *chosen* armed transition by its
    /// index (the stable handle from [`Engine::fireable`]), resolving the
    /// destination the normal way. See [`Engine::fire_idx_to`] to force
    /// the branch.
    pub fn fire_idx(&mut self, trans_idx: TransIdx) -> Result<Event, EngineError> {
        self.fire_idx_inner(trans_idx, None)
    }

    /// **Interactive control**: fire a chosen armed transition by index,
    /// **forcing** its destination to the state named `to`.
    pub fn fire_idx_to(&mut self, trans_idx: TransIdx, to: &str) -> Result<Event, EngineError> {
        let forced = self.resolve_forced(trans_idx, to)?;
        self.fire_idx_inner(trans_idx, Some(forced))
    }

    /// **Interactive control**: fire a chosen armed transition by index at
    /// its scheduled date (as [`Engine::fire_idx`]), **forcing** its
    /// destination to the target at position `branch` of the compiled
    /// target list ([`crate::compile::CTransition::targets`], declared
    /// order kept).
    ///
    /// The index form avoids the state-name lookup of
    /// [`Engine::fire_idx_to`] and names a branch unambiguously when two
    /// branches share a destination state. Errors with
    /// [`EngineError::ForcedBranchOutOfRange`] when `branch` is not a
    /// valid position; nothing is changed in that case.
    pub fn fire_idx_to_branch(
        &mut self,
        trans_idx: TransIdx,
        branch: usize,
    ) -> Result<Event, EngineError> {
        let forced = self.resolve_branch(trans_idx, branch)?;
        self.fire_idx_inner(trans_idx, Some(forced))
    }

    /// **Exploration**: fire an armed transition **at the current
    /// instant**, whatever its scheduled date, leaving the clock
    /// unmoved. `branch` forces the destination by position in the
    /// compiled target list (see [`Engine::fire_idx_to_branch`]); `None`
    /// resolves it the normal way (drawn for a probabilistic
    /// instantaneous branching).
    ///
    /// This is the move of a sequence-tree explorer over the embedded
    /// jump chain: the *order* of the jumps is chosen, their dates are
    /// not simulated. No continuous evolution takes place, since time
    /// does not advance.
    ///
    /// "Armed" means date-scheduled (any date, including the `+∞` of a
    /// zero-rate exponential, which this method does fire if asked),
    /// armed without a date in deferred mode and not paused, or a watched
    /// transition whose guard already holds.
    ///
    /// The firing itself draws nothing when `branch` is given, but the
    /// discrete step that follows re-arms the schedule, which draws the
    /// dates (and `Exp(1)` hazard thresholds) of transitions it newly
    /// arms, exactly as any firing does: the RNG advances then. An
    /// explorer ignores those dates, and a [`Snapshot`] carries the RNG,
    /// so exploration stays deterministic. The draw order is the
    /// engine's own, so Monte-Carlo runs are unaffected.
    ///
    /// Errors with [`EngineError::UnknownTransition`] for an index out of
    /// range, [`EngineError::NotFireable`] if the transition is not
    /// armed, and [`EngineError::ForcedBranchOutOfRange`] for an invalid
    /// `branch`; the engine is unchanged in all three cases.
    pub fn fire_now(
        &mut self,
        trans_idx: TransIdx,
        branch: Option<usize>,
    ) -> Result<Event, EngineError> {
        self.operator_transaction(|engine| engine.fire_now_inner(trans_idx, branch))
    }

    fn fire_now_inner(
        &mut self,
        trans_idx: TransIdx,
        branch: Option<usize>,
    ) -> Result<Event, EngineError> {
        let Some(transition) = self.model.transitions.get(trans_idx) else {
            return Err(EngineError::UnknownTransition {
                transition: format!("<index {trans_idx}>"),
            });
        };
        let forced = match branch {
            Some(branch) => Some(self.resolve_branch(trans_idx, branch)?),
            None => None,
        };
        if self.is_armed(trans_idx) {
            self.fire(trans_idx, forced)
        } else if self.is_immediate_watched(trans_idx)? {
            self.note_watched_firing()?;
            self.fire(trans_idx, forced)
        } else {
            Err(EngineError::NotFireable {
                transition: transition.name.clone(),
                time: self.time,
            })
        }
    }

    /// **Exploration**: the current occurrence rate λ (events per time
    /// unit) of an armed exponential transition, read without sampling.
    ///
    /// - A fixed-rate exponential returns its rate.
    /// - A state-dependent rate (`rate_expr`) returns λ(x) evaluated on
    ///   the current state. When λ is piecewise constant (it reads only
    ///   discretely updated state) this value holds until the next jump,
    ///   which is what an exact exploration of the embedded jump chain
    ///   needs. When λ varies continuously (it reads an integrated
    ///   attribute or time), the value is the *instantaneous* hazard
    ///   rate at the current instant only: the two cases are told apart
    ///   by the compiled law, `CLaw::ExpVar { continuous, .. }` in
    ///   [`crate::compile::CTransition::distrib`].
    ///
    /// A rate of zero (a dormant spare) is returned as `Some(0.0)`: the
    /// transition is armed, it simply cannot fire from this state.
    ///
    /// Returns `Ok(None)` for a transition that is not armed (neither
    /// date-scheduled nor running in deferred mode, including a countdown
    /// paused by a `resume` interruption) and for every other law (delay, instantaneous,
    /// watched, Weibull, lognormal, gamma, uniform, empirical). The law
    /// family and, for an instantaneous branching, its branch
    /// probabilities are read from the public compiled model
    /// ([`crate::compile::CLaw`]), not through the engine.
    ///
    /// Errors with [`EngineError::UnknownTransition`] for an index out of
    /// range, and with [`EngineError::TypeError`] when a state-dependent
    /// rate evaluates to a non-finite or negative value.
    pub fn armed_rate(&self, trans_idx: TransIdx) -> Result<Option<f64>, EngineError> {
        let Some(transition) = self.model.transitions.get(trans_idx) else {
            return Err(EngineError::UnknownTransition {
                transition: format!("<index {trans_idx}>"),
            });
        };
        if !self.is_armed(trans_idx) {
            return Ok(None);
        }
        match &transition.distrib {
            CLaw::Exp(rate) => Ok(Some(*rate)),
            CLaw::ExpVar { rate, .. } => self.eval_rate(trans_idx, rate).map(Some),
            _ => Ok(None),
        }
    }

    /// **Exploration**: the target (feared event) latched by this
    /// trajectory, with the instant it was reached, or `None` while no
    /// target has been reached.
    ///
    /// The latch is set by the first discrete step (or the
    /// initialization) that makes a declared target state active, and
    /// then holds: it names the *first* target reached, not the ones
    /// active now. It is recorded only when
    /// [`EngineConfig::stop_at_targets`] is set; otherwise this is always
    /// `None`. [`Engine::restore`] rewinds it with the rest of the
    /// trajectory state.
    #[must_use]
    pub fn reached_target(&self) -> Option<(&str, f64)> {
        self.seq_end
            .as_ref()
            .map(|(name, time)| (name.as_str(), *time))
    }

    /// **Interactive control**: override the scheduled firing date of an
    /// armed transition (manual date-setting). The transition must be
    /// date-scheduled (`pending`, i.e. not a watched boundary), and the
    /// new date must not lie in the past (`>=` the current time).
    ///
    /// The override sticks for delay / inst / fixed-rate transitions
    /// until they fire or leave their source state; a *state-dependent
    /// rate* transition may have its date recomputed at the next
    /// discrete step (`reschedule_modifiable`).
    ///
    /// A stochastic transition armed in deferred mode has no date: it is
    /// refused with [`EngineError::DeferredDate`] (fire it with
    /// [`Engine::fire_deferred_at`]).
    pub fn set_date(&mut self, name: &str, date: f64) -> Result<(), EngineError> {
        let idx = self.transition_index(name)?;
        self.set_date_idx(idx, date)
    }

    /// **Interactive control**: override an armed transition's firing
    /// date by index (see [`Engine::set_date`]).
    pub fn set_date_idx(&mut self, trans_idx: TransIdx, date: f64) -> Result<(), EngineError> {
        let Some(transition) = self.model.transitions.get(trans_idx) else {
            return Err(EngineError::UnknownTransition {
                transition: format!("<index {trans_idx}>"),
            });
        };
        let name = transition.name.clone();
        if self.config.stochastic_dates == StochasticDates::Operator {
            if !date.is_finite() || date < self.time || date > self.config.t_max {
                return Err(EngineError::AdvanceTimeInvalid {
                    date,
                    time: self.time,
                    horizon: self.config.t_max,
                });
            }
            let scheduled = self.pending[trans_idx].is_some() || self.deferred_running(trans_idx);
            if !scheduled {
                return Err(EngineError::NotFireable {
                    transition: name,
                    time: self.time,
                });
            }
            self.operator_dates[trans_idx] = Some(OperatorDate::Active(date));
            if self.config.journal {
                self.journal.push(JournalRecord::TransitionRescheduled {
                    time: self.time,
                    transition: name,
                    firing_at: date,
                });
            }
            return Ok(());
        }
        if self.deferred[trans_idx].is_some() {
            return Err(EngineError::DeferredDate { transition: name });
        }
        if !date.is_finite() || date < self.time {
            return Err(EngineError::DateInPast {
                transition: name,
                date,
                time: self.time,
            });
        }
        match self.pending.get_mut(trans_idx) {
            Some(slot) if slot.is_some() => *slot = Some(date),
            _ => {
                return Err(EngineError::NotFireable {
                    transition: name,
                    time: self.time,
                })
            }
        }
        if self.config.journal {
            self.journal.push(JournalRecord::TransitionRescheduled {
                time: self.time,
                transition: name,
                firing_at: date,
            });
        }
        Ok(())
    }

    /// Per-transition firing count and nominal exposure accrued up to the
    /// current time ([`TransitionExposure`]), indexed like the compiled
    /// transitions; `None` unless [`EngineConfig::rate_factors`] is
    /// non-empty. Stretches still running are counted up to now, without
    /// closing them: the next step continues accruing from where the
    /// engine's own tally stands.
    #[must_use]
    pub fn rate_statistics(&self) -> Option<Vec<TransitionExposure>> {
        self.exposure.as_ref().map(|tally| {
            let mut tally = tally.clone();
            tally.accrue(&self.pending, self.time);
            tally.stats
        })
    }

    /// Capture a native-only trajectory as an opaque [`Snapshot`].
    /// Imported units require [`Engine::try_snapshot`] so their native state
    /// is captured together with the Rust state.
    ///
    /// # Errors
    /// Refuses an FMU model before returning an incomplete checkpoint.
    pub fn snapshot(&self) -> Result<Snapshot, EngineError> {
        if let Some(unit) = self.model.fmu_units.first() {
            return Err(EngineError::FmuSnapshotApi {
                unit: unit.name.clone(),
                operation: "snapshot capture",
                alternative: "Engine::try_snapshot",
            });
        }
        Ok(self.capture_rust_snapshot())
    }

    fn capture_rust_snapshot(&self) -> Snapshot {
        Snapshot {
            model_identity: self.model.cache_id,
            fmu_states: None,
            fmu_positions: self
                .host
                .as_ref()
                .map_or_else(Vec::new, |host| host.positions()),
            time: self.time,
            vars: self.vars.clone(),
            states: self.states.clone(),
            pending: self.pending.clone(),
            armed_wave: self.armed_wave.clone(),
            wave: self.wave,
            clocks: self.clocks.clone(),
            stochastic_dates: self.config.stochastic_dates,
            biased_rates: !self.config.rate_factors.is_empty(),
            frozen: self.frozen.clone(),
            hazards: self.hazards.clone(),
            deferred: self.deferred.clone(),
            operator_dates: self.operator_dates.clone(),
            events: self.events.clone(),
            journal: self.journal.clone(),
            seq_events: self.seq_events.clone(),
            seq_end: self.seq_end.clone(),
            indicator_series: self.indicator_series.clone(),
            sampled: self.sampled.clone(),
            sample_cursor: self.sample_cursor,
            watched_streak: self.watched_streak,
            firings: self.firings.clone(),
            first_firing: self.first_firing.clone(),
            flow_restarts: self.flow_restarts,
            first_flow_restart: self.first_flow_restart,
            rng: self.rng.clone(),
            worklist: self.worklist.clone(),
            exposure: self.exposure.clone(),
        }
    }

    /// Capture the engine and serializable native FMU state together.
    ///
    /// # Errors
    /// Refuses a unit without state get/set and serialization capability.
    pub fn try_snapshot(&mut self) -> Result<Snapshot, EngineError> {
        let mut snapshot = self.capture_rust_snapshot();
        if let Some(host) = self.host.as_deref_mut() {
            snapshot.fmu_states = Some(host.save_states()?);
        }
        Ok(snapshot)
    }

    /// Restore the engine and native FMU state from one checkpoint.
    ///
    /// # Errors
    /// Refuses a checkpoint without serialized state for imported units.
    pub fn try_restore(&mut self, snapshot: &Snapshot) -> Result<(), EngineError> {
        self.check_operator_snapshot(snapshot)?;
        if let Some(host) = self.host.as_deref_mut() {
            let states =
                snapshot
                    .fmu_states
                    .as_ref()
                    .ok_or_else(|| EngineError::FmuCapability {
                        unit: self
                            .model
                            .fmu_units
                            .first()
                            .map_or_else(String::new, |unit| unit.name.clone()),
                        capability: "serialized FMU state in snapshot",
                    })?;
            host.restore_states(states)?;
            host.restore_positions(&snapshot.fmu_positions);
        }
        self.restore_rust_snapshot(snapshot);
        Ok(())
    }

    /// **Interactive control**: reinstate a previously captured
    /// [`Snapshot`] (undo) for native-only models. Imported units require
    /// [`Engine::try_restore`] to restore their native state too.
    ///
    /// # Errors
    /// Refuses an FMU model before changing either state.
    pub fn restore(&mut self, snap: &Snapshot) -> Result<(), EngineError> {
        self.check_operator_snapshot(snap)?;
        if let Some(unit) = self.model.fmu_units.first() {
            return Err(EngineError::FmuSnapshotApi {
                unit: unit.name.clone(),
                operation: "snapshot restore",
                alternative: "Engine::try_restore",
            });
        }
        self.restore_rust_snapshot(snap);
        Ok(())
    }

    fn restore_rust_snapshot(&mut self, snap: &Snapshot) {
        self.time = snap.time;
        self.vars = snap.vars.clone();
        self.states = snap.states.clone();
        self.pending = snap.pending.clone();
        self.armed_wave = snap.armed_wave.clone();
        self.wave = snap.wave;
        self.clocks = snap.clocks.clone();
        self.frozen = snap.frozen.clone();
        self.hazards = snap.hazards.clone();
        self.deferred = snap.deferred.clone();
        self.operator_dates = snap.operator_dates.clone();
        self.events = snap.events.clone();
        self.journal = snap.journal.clone();
        self.seq_events = snap.seq_events.clone();
        self.seq_end = snap.seq_end.clone();
        self.indicator_series = snap.indicator_series.clone();
        self.sampled = snap.sampled.clone();
        self.sample_cursor = snap.sample_cursor;
        self.watched_streak = snap.watched_streak;
        self.firings = snap.firings.clone();
        self.first_firing = snap.first_firing.clone();
        self.flow_restarts = snap.flow_restarts;
        self.first_flow_restart = snap.first_flow_restart;
        self.rng = snap.rng.clone();
        self.worklist = snap.worklist.clone();
        self.exposure = snap.exposure.clone();
        // The indexed watched set is *derived*, never carried: rewinding
        // the state rewinds the arming and discards every cached verdict,
        // which is what keeps a replay from a restored snapshot exact.
        self.rebuild_watched_index();
    }

    /// **Interactive control**: the events fired so far, in
    /// chronological order (the same data a finished [`SimulationResult`]
    /// reports in its `events`).
    #[must_use]
    pub fn history(&self) -> &[Event] {
        &self.events
    }

    /// Discard the recorded history (fired events, journal, sequence
    /// events, indicator and sample points) while keeping the trajectory
    /// state, the schedule and the RNG untouched.
    ///
    /// The history is a record only: no rule of the semantics reads it
    /// back, so discarding it changes no later firing, date or value.
    /// A caller that drives the engine itself and keeps its own record,
    /// such as the sequence-tree explorer, calls this after each firing
    /// so that a [`Snapshot`] stays proportional to the model rather
    /// than to the path length.
    pub fn forget_history(&mut self) {
        self.events.clear();
        self.journal.clear();
        self.seq_events.clear();
        for series in &mut self.indicator_series {
            series.points.clear();
        }
        for series in &mut self.sampled {
            series.points.clear();
        }
    }

    /// **Interactive control**: reset the engine to its initial state
    /// (`t = 0`), as freshly built: clears the trajectory and recorded
    /// history, re-seeds the RNG to `(seed, stream)`, and re-runs the
    /// initialization axiom. A run restarted from here is identical to a
    /// fresh [`Engine::new`].
    pub fn reset(&mut self) -> Result<(), EngineError> {
        self.operator_transaction(Self::reset_inner)
    }

    fn reset_inner(&mut self) -> Result<(), EngineError> {
        if let Some(host) = self.host.as_deref_mut() {
            host.start(&self.model.var_init)?;
        }
        let n = self.model.transitions.len();
        self.time = 0.0;
        self.vars = self.model.var_init.clone();
        self.states = self.model.automata.iter().map(|a| a.init).collect();
        self.pending = vec![None; n];
        self.armed_wave = vec![0; n];
        self.wave = 0;
        self.clocks = vec![None; n];
        self.frozen = vec![None; n];
        self.hazards = vec![None; n];
        self.deferred = vec![None; n];
        self.operator_dates = if self.config.stochastic_dates == StochasticDates::Operator {
            vec![None; n]
        } else {
            Vec::new()
        };
        if self.config.stochastic_dates == StochasticDates::Operator {
            self.firings.fill(0);
            self.first_firing.fill(f64::NAN);
            self.flow_restarts = 0;
            self.first_flow_restart = f64::NAN;
        }
        self.events.clear();
        self.journal.clear();
        self.seq_events.clear();
        self.seq_end = None;
        for series in &mut self.indicator_series {
            series.points.clear();
        }
        for series in &mut self.sampled {
            series.points.clear();
        }
        self.sample_cursor = 0;
        self.watched_streak = (0.0, 0);
        self.rng = raichu_rng::replica_rng(self.config.seed, self.config.rng_stream);
        self.worklist.clear();
        if let Some(tally) = self.exposure.as_mut() {
            *tally = ExposureTally::new(self.model);
        }
        self.initialize()
    }

    /// Fire a *chosen* armed transition (rather than the earliest one, as
    /// [`Engine::step`] does), advancing time to its scheduled date and
    /// running the discrete fixpoint. `forced` overrides the destination
    /// branch when set.
    ///
    /// - A date-scheduled transition (delay / inst / stochastic) fires at
    ///   its `pending` date; with continuous evolution the state is
    ///   integrated up to that date first, and a **watched boundary**
    ///   crossed en route fires *instead* (a forced jump cannot be
    ///   skipped: the returned event is that boundary transition, whose
    ///   branch is never forced).
    /// - A watched transition may be fired only while its guard already
    ///   holds (at the current instant).
    ///
    /// Choosing a non-earliest transition deliberately overrides the
    /// schedule: the interactive counterpart of a manually driven run.
    fn fire_idx_inner(
        &mut self,
        trans_idx: TransIdx,
        forced: Option<StateIdx>,
    ) -> Result<Event, EngineError> {
        self.operator_transaction(|engine| engine.fire_idx_unchecked(trans_idx, forced))
    }

    fn fire_idx_unchecked(
        &mut self,
        trans_idx: TransIdx,
        forced: Option<StateIdx>,
    ) -> Result<Event, EngineError> {
        let Some(transition) = self.model.transitions.get(trans_idx) else {
            return Err(EngineError::UnknownTransition {
                transition: format!("<index {trans_idx}>"),
            });
        };
        let name = transition.name.clone();
        let operator = match self.operator_dates.get(trans_idx).copied().flatten() {
            Some(OperatorDate::Active(date)) => Some(date),
            _ => None,
        };
        let explicit_now = (self.config.stochastic_dates == StochasticDates::Operator
            && self.deferred_running(trans_idx))
        .then_some(self.time);
        if let Some(date) = operator
            .or(self.pending.get(trans_idx).copied().flatten())
            .or(explicit_now)
        {
            if !date.is_finite() {
                return Err(EngineError::NotFireable {
                    transition: name,
                    time: self.time,
                });
            }
            // Advance to the scheduled date, but never move the clock
            // backwards: an *overdue* transition (date already passed
            // because an earlier `fire_idx` skipped ahead) fires at the
            // current instant.
            let t_new = date.max(self.time);
            if self.needs_integration() && t_new > self.time {
                if let Some(watched_idx) = self.advance_continuous(t_new)? {
                    self.note_watched_firing()?;
                    return self.fire(watched_idx, None);
                }
            }
            if t_new > self.time {
                self.flush_samples_before(t_new);
            }
            self.time = t_new;
            self.note_time_change();
            self.watched_streak = (t_new, 0);
            self.fire(trans_idx, forced)
        } else if self.is_immediate_watched(trans_idx)? {
            self.note_watched_firing()?;
            self.fire(trans_idx, forced)
        } else {
            Err(EngineError::NotFireable {
                transition: name,
                time: self.time,
            })
        }
    }

    /// Resolve a qualified transition name to its index, or
    /// [`EngineError::UnknownTransition`].
    fn transition_index(&self, name: &str) -> Result<TransIdx, EngineError> {
        self.model
            .transitions
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| EngineError::UnknownTransition {
                transition: name.to_owned(),
            })
    }

    /// Resolve a forced destination *state name* to a valid branch of
    /// `trans_idx`, or [`EngineError::ForcedTargetInvalid`] if the name
    /// is unknown or not one of the transition's declared target states.
    fn resolve_forced(&self, trans_idx: TransIdx, to: &str) -> Result<StateIdx, EngineError> {
        let Some(transition) = self.model.transitions.get(trans_idx) else {
            return Err(EngineError::UnknownTransition {
                transition: format!("<index {trans_idx}>"),
            });
        };
        let automaton = &self.model.automata[transition.automaton];
        match automaton.states.iter().position(|s| s == to) {
            Some(state) if transition.targets.contains(&state) => Ok(state),
            _ => Err(EngineError::ForcedTargetInvalid {
                transition: transition.name.clone(),
                state: to.to_owned(),
            }),
        }
    }

    /// Resolve a forced destination *branch index* to the state it
    /// designates in `trans_idx`'s compiled target list, or
    /// [`EngineError::ForcedBranchOutOfRange`].
    pub(super) fn resolve_branch(
        &self,
        trans_idx: TransIdx,
        branch: usize,
    ) -> Result<StateIdx, EngineError> {
        let Some(transition) = self.model.transitions.get(trans_idx) else {
            return Err(EngineError::UnknownTransition {
                transition: format!("<index {trans_idx}>"),
            });
        };
        transition
            .targets
            .get(branch)
            .copied()
            .ok_or_else(|| EngineError::ForcedBranchOutOfRange {
                transition: transition.name.clone(),
                branch,
                branches: transition.targets.len(),
            })
    }

    /// Whether `trans_idx` is a watched transition sitting in its source
    /// state with its guard already true: i.e. fireable at the current
    /// instant.
    fn is_immediate_watched(&self, trans_idx: TransIdx) -> Result<bool, EngineError> {
        let transition = &self.model.transitions[trans_idx];
        if !matches!(transition.distrib, CLaw::Watched { .. }) {
            return Ok(false);
        }
        if self.states[transition.automaton] != transition.source {
            return Ok(false);
        }
        let Some(guard) = &transition.guard else {
            return Ok(false);
        };
        eval_bool(self.model, &self.vars, &self.states, self.time, guard)
    }
}

impl<'m> Engine<'m> {
    pub(super) fn operator_choice(&self, idx: TransIdx) -> bool {
        matches!(&self.model.transitions[idx].distrib, CLaw::Inst(probs)
            if probs.iter().filter(|prob| **prob > 0.0).count() > 1)
    }

    fn check_operator_snapshot(&self, snapshot: &Snapshot) -> Result<(), EngineError> {
        if (self.config.stochastic_dates == StochasticDates::Operator)
            != (snapshot.stochastic_dates == StochasticDates::Operator)
        {
            return Err(EngineError::OperatorSnapshotPolicy);
        }
        if self.config.stochastic_dates == StochasticDates::Operator
            && snapshot.model_identity != self.model.cache_id
        {
            return Err(EngineError::OperatorSnapshotModel);
        }
        Ok(())
    }

    pub(super) fn operator_transaction<T>(
        &mut self,
        command: impl FnOnce(&mut Self) -> Result<T, EngineError>,
    ) -> Result<T, EngineError> {
        if self.config.stochastic_dates != StochasticDates::Operator {
            return command(self);
        }
        let before = self.try_snapshot()?;
        match command(self) {
            Ok(value) => Ok(value),
            Err(error) => {
                self.try_restore(&before)?;
                Err(error)
            }
        }
    }

    pub(super) fn refresh_operator_date(&mut self, idx: TransIdx, in_source: bool, guard: bool) {
        let Some(date) = self.operator_dates.get(idx).copied().flatten() else {
            return;
        };
        use raichu_model::InterruptionPolicy;
        self.operator_dates[idx] = match (
            in_source,
            guard,
            self.model.transitions[idx].on_interruption,
            date,
        ) {
            (false, _, _, _) | (_, false, InterruptionPolicy::Reset, _) => None,
            (_, false, InterruptionPolicy::Resume, OperatorDate::Active(at)) => {
                Some(OperatorDate::Paused((at - self.time).max(0.0)))
            }
            (_, true, _, OperatorDate::Paused(remaining)) => {
                Some(OperatorDate::Active(self.time + remaining))
            }
            _ => Some(date),
        };
    }

    /// Advance an operator-controlled trajectory to its first relevant instant.
    ///
    /// Stochastic clocks without explicit dates stay armed without any RNG
    /// draw. Deterministic simultaneous reactions complete before returning,
    /// unless a probabilistic branch requires an explicit destination or
    /// `max_events` firings have been committed. The latter is successful
    /// incomplete progress, so the caller may continue from the returned state.
    /// Dates and continuous quantities use the model's time units. No dense
    /// exploration hazard trace is allocated.
    ///
    /// # Errors
    /// Requires [`StochasticDates::Operator`], a finite forward date no later
    /// than the horizon, and propagates numerical/fixpoint failures. Any error
    /// restores the complete trajectory checkpoint, including RNG and dates.
    pub fn advance_operator_to(
        &mut self,
        date: f64,
        max_events: usize,
    ) -> Result<OperatorAdvance, EngineError> {
        // Validate before checkpointing, as invalid commands mutate no state.
        self.validate_operator_advance(date)?;
        self.operator_transaction(|engine| engine.advance_operator_inner(date, max_events))
    }

    fn advance_operator_checked(
        &mut self,
        date: f64,
        max_events: usize,
    ) -> Result<OperatorAdvance, EngineError> {
        self.validate_operator_advance(date)?;
        self.advance_operator_inner(date, max_events)
    }

    fn validate_operator_advance(&self, date: f64) -> Result<(), EngineError> {
        if self.config.stochastic_dates != StochasticDates::Operator {
            return Err(EngineError::InteractivePolicy {
                operation: "advance_operator_to",
                policy: "operator stochastic dates",
            });
        }
        if !date.is_finite() || date < self.time || date > self.config.t_max {
            return Err(EngineError::AdvanceTimeInvalid {
                date,
                time: self.time,
                horizon: self.config.t_max,
            });
        }
        Ok(())
    }

    fn advance_operator_inner(
        &mut self,
        date: f64,
        max_events: usize,
    ) -> Result<OperatorAdvance, EngineError> {
        let mut events = Vec::new();
        let mut event_instant = None;
        let stop;
        let mut choice = None;
        loop {
            if events.len() >= max_events {
                stop = OperatorStop::Incomplete;
                break;
            }
            // Close the current native instant before committing the FMU's
            // zero-order-held inputs. A choice or remaining same-date firing
            // must be resolved first, including after a bounded continuation.
            if self.host.is_some()
                && self.immediate_watched()?.is_none()
                && !self
                    .next_pending()
                    .is_some_and(|(_, pending)| pending <= self.time)
            {
                self.sample_fmu_inputs_at_current_point();
            }
            // A communication point is relevant too; never integrate through
            // external outputs. The host's existing serialized state joins
            // the transaction checkpoint.
            let fmu_point = self.next_fmu_point();
            let bound = event_instant
                .unwrap_or(date)
                .min(fmu_point.unwrap_or(f64::INFINITY));
            let (_, reason) = self.probe_inner(bound)?;
            let idx = match reason {
                ProbeStop::Watched { index } | ProbeStop::Deterministic { index } => Some(index),
                _ => None,
            };
            if let Some(idx) = idx {
                if self.operator_choice(idx) {
                    choice = Some(self.model.transitions[idx].name.clone());
                    stop = OperatorStop::Choice;
                    break;
                }
                if matches!(reason, ProbeStop::Watched { .. }) {
                    self.note_watched_firing()?;
                }
                events.push(self.fire(idx, None)?);
                event_instant = Some(self.time);
            } else if fmu_point.is_some_and(|point| point <= bound) {
                self.process_fmu_point()?;
                events.push(Event {
                    time: self.time,
                    transition: "fmi.communication".to_owned(),
                    from: String::new(),
                    to: String::new(),
                });
                event_instant = Some(self.time);
            } else {
                stop = if event_instant.is_some() {
                    OperatorStop::Event
                } else {
                    OperatorStop::Target
                };
                break;
            }
        }
        self.flush_samples_through(self.time);
        self.record_indicators();
        Ok(OperatorAdvance {
            requested_time: date,
            reached_time: self.time,
            stop,
            events,
            choice,
        })
    }
}
