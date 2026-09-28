//! The run loop and discrete evolution: stepping and running to the
//! horizon, firing the earliest transition (`fire_transition`), and the
//! sensitive-function propagation to fixpoint (`propagate_effects`) with
//! its optional confluence probe.

use super::*;

impl<'m> Engine<'m> {
    /// Sequence analysis: label the trajectory with the first target
    /// (feared event) whose state is active: sets `seq_end` once.
    pub(super) fn check_targets(&mut self) {
        if !self.config.stop_at_targets || self.seq_end.is_some() {
            return;
        }
        for target in &self.model.targets {
            if self.states[target.automaton] == target.state {
                self.seq_end = Some((target.name.clone(), self.time));
                break;
            }
        }
    }

    /// Fire the next transition: discrete (`fire_transition`) or watched at a
    /// located boundary crossing (`schedule_boundary`): if one occurs within the
    /// horizon.
    ///
    /// Returns the fired event, or `None` when nothing remains before
    /// `t_max`. Tie-break: earliest date first, then lowest transition
    /// index (documented deterministic order; the converged state does
    /// not depend on it for confluent models).
    pub fn step(&mut self) -> Result<Option<Event>, EngineError> {
        // Watched transition already past its boundary (initial
        // conditions or post-jump state): fires immediately.
        if let Some(trans_idx) = self.immediate_watched()? {
            self.note_watched_firing()?;
            return self.fire(trans_idx, None).map(Some);
        }

        let next_discrete = self.next_pending();
        let t_target =
            next_discrete.map_or(self.config.t_max, |(_, date)| date.min(self.config.t_max));

        if self.needs_integration() && t_target > self.time && t_target.is_finite() {
            if let Some(trans_idx) = self.advance_continuous(t_target)? {
                self.note_watched_firing()?;
                return self.fire(trans_idx, None).map(Some);
            }
        }

        match next_discrete {
            Some((trans_idx, date)) if date <= self.config.t_max => {
                // The clock never runs backwards: in a step-driven run
                // every scheduled date is ≥ the current time, so `max`
                // is a no-op; it only guards an *overdue* transition left
                // behind after an interactive `fire_idx` skipped ahead.
                let t_new = date.max(self.time);
                self.flush_samples_before(t_new);
                self.time = t_new;
                self.note_time_change();
                self.watched_streak = (t_new, 0);
                self.fire(trans_idx, None).map(Some)
            }
            _ => Ok(None),
        }
    }

    /// Run until the schedule drains or the horizon is reached, then
    /// return the full result with provenance. With
    /// [`EngineConfig::stop_at_targets`] on, a run **early-stops** at the
    /// first target (feared event) reached.
    pub fn run(mut self) -> Result<SimulationResult, EngineError> {
        loop {
            if let Some((_, t_hit)) = &self.seq_end {
                // The target is reached: FINISH the hit instant first,
                // fire every transition still due at it, so the latched
                // state is the completed instant, not a half-propagated
                // one (PyCATSHOO completes the step before stopping).
                let t_hit = *t_hit;
                let still_due = self.pending.iter().flatten().any(|d| *d <= t_hit);
                if !still_due {
                    break;
                }
            }
            if self.step()?.is_none() {
                break;
            }
        }
        // Advance the clock (and the continuous state) to the horizon:
        // unless a target early-stopped the trajectory.
        let final_time = if let Some((_, t)) = &self.seq_end {
            *t
        } else if self.config.t_max.is_finite() {
            if self.needs_integration() && self.config.t_max > self.time {
                self.advance_continuous(self.config.t_max)?;
            }
            self.config.t_max
        } else {
            self.time
        };
        self.time = final_time;
        self.note_time_change();
        // Censor the exposure of every still-running transition at the
        // trajectory's final time (the target instant on an early stop).
        if let Some(tally) = self.exposure.as_mut() {
            tally.accrue(&self.pending, final_time);
        }
        // A target-stopped trajectory holds its frozen state through the
        // remaining sample instants (the latch semantics of a
        // target-stopped study: the feared-event state stays active from
        // the hit to the horizon in every sampled measure). With an
        // infinite horizon the latch extends through the last *requested*
        // sample instant, so every replica's series covers the schedule.
        let flush_to = if self.seq_end.is_some() {
            if self.config.t_max.is_finite() {
                self.config.t_max
            } else {
                self.config
                    .samples
                    .last()
                    .copied()
                    .unwrap_or(final_time)
                    .max(final_time)
            }
        } else {
            final_time
        };
        self.flush_samples_through(flush_to);
        let sequence = self.config.sequences.then(|| {
            let (end_cause, end_time) = match self.seq_end.take() {
                Some((cause, t)) => (Some(cause), t),
                None => (None, final_time),
            };
            Sequence {
                events: std::mem::take(&mut self.seq_events),
                end_cause,
                end_time,
                weight: 1.0,
            }
        });
        let work = self.work();
        Ok(SimulationResult {
            events: self.events,
            indicators: self.indicator_series,
            samples: self.sampled,
            journal: self.journal,
            sequence,
            provenance: Provenance {
                engine_version: env!("CARGO_PKG_VERSION").to_owned(),
                model: self.model.name.clone(),
                t_max: self.config.t_max,
                seed: self.stochastic.then_some(self.config.seed),
                ode_rtol: self.config.ode.rtol,
                ode_tol_event: self.config.ode.tol_event,
            },
            work,
            final_time,
            rate_statistics: self.exposure.map(|tally| tally.stats),
        })
    }

    // ---- internals ----------------------------------------------------

    /// Fire `trans_idx` at the current time: state change, journal,
    /// discrete evolution to fixpoint, schedule update, indicators.
    ///
    /// `forced` overrides the destination branch (interactive control,
    /// bypassing the RNG / deterministic-branch resolution); `None`
    /// resolves the destination the normal way ([`Engine::resolve_target`]).
    pub(super) fn fire(
        &mut self,
        trans_idx: TransIdx,
        forced: Option<StateIdx>,
    ) -> Result<Event, EngineError> {
        // Close the exposure stretch up to this instant while `pending`
        // still describes what was running, then count the firing.
        if let Some(tally) = self.exposure.as_mut() {
            tally.accrue(&self.pending, self.time);
            tally.stats[trans_idx].firings += 1;
        }
        self.pending[trans_idx] = None;
        self.frozen[trans_idx] = None;
        self.hazards[trans_idx] = None;
        self.deferred[trans_idx] = None;
        let target = match forced {
            Some(state) => state,
            None => self.resolve_target(trans_idx)?,
        };
        let transition = &self.model.transitions[trans_idx];
        let automaton = &self.model.automata[transition.automaton];
        let event = Event {
            time: self.time,
            transition: transition.name.clone(),
            from: automaton.states[transition.source].clone(),
            to: automaton.states[target].clone(),
        };
        let owner = transition.automaton;
        self.states[owner] = target;
        // Indexed watched set: re-arm what this automaton owns and
        // invalidate the guards that read its state.
        self.note_state_change(owner);
        let transition = &self.model.transitions[trans_idx];
        if self.config.journal {
            self.journal.push(JournalRecord::TransitionFired {
                time: self.time,
                transition: event.transition.clone(),
                from: event.from.clone(),
                to: event.to.clone(),
            });
        }
        self.events.push(event.clone());
        self.note_firing(trans_idx)?;
        // Sequence analysis: record the entry into a monitored state.
        if self.config.sequences && transition.monitored {
            self.seq_events.push(SeqEvent {
                obj: transition.component.clone(),
                attr: event.to.clone(),
                time: self.time,
                cycle_group: transition.cycle_group.clone(),
            });
        }

        // The firing edge's own writes, once, after the state change and
        // before anything propagates: what they write is read by the
        // fixpoint below like any change.
        self.apply_transition_effects(trans_idx)?;

        // Discrete evolution: functions sensitive to this automaton.
        self.worklist.extend(
            self.model.state_triggers[self.model.transitions[trans_idx].automaton]
                .iter()
                .copied(),
        );
        self.run_fixpoint()?;
        self.resolve_flows()?;
        self.refresh_schedule()?;
        self.record_indicators();
        // Sequence analysis: the first target (feared event) whose state is
        // now active labels the trajectory's end cause (and ends it: see
        // `run`). States change only through transitions (or the declared
        // init, checked in `initialize`), so this catches every activation.
        self.check_targets();
        Ok(event)
    }

    pub(super) fn next_pending(&self) -> Option<(TransIdx, f64)> {
        let mut best: Option<(TransIdx, f64)> = None;
        for (idx, pending) in self.pending.iter().enumerate() {
            if let Some(date) = pending {
                let better = match best {
                    None => true,
                    Some((_, best_date)) => *date < best_date,
                };
                if better {
                    best = Some((idx, *date));
                }
            }
        }
        best
    }

    fn resolve_target(&mut self, trans_idx: TransIdx) -> Result<StateIdx, EngineError> {
        // Copy the `'m` model reference out so the transition borrow is
        // independent of `&mut self` (frees `self.rng` for the draw).
        let model = self.model;
        let transition = &model.transitions[trans_idx];
        match &transition.distrib {
            CLaw::Delay(_)
            | CLaw::Watched { .. }
            | CLaw::Exp(_)
            | CLaw::ExpVar { .. }
            | CLaw::Weibull(..)
            | CLaw::Lognormal(..)
            | CLaw::Gamma(..)
            | CLaw::Uniform(..)
            | CLaw::Empirical(_) => Ok(transition.targets[0]),
            CLaw::Inst(probs) => {
                // Deterministic fast path: exactly one branch with
                // probability 1: resolved without touching the RNG, so a
                // deterministic model stays RNG-free and bit-identical on
                // replay.
                if let Some(branch) = probs
                    .iter()
                    .position(|p| (*p - 1.0).abs() <= f64::EPSILON)
                    .filter(|_| probs.iter().filter(|p| **p > 0.0).count() == 1)
                {
                    return Ok(transition.targets[branch]);
                }
                // Stochastic instantaneous branching (`schedule_stochastic`
                // realised on demand): draw the destination from the
                // categorical distribution over `probs` (Σ = 1, validated at
                // model build) by inverse-CDF on one uniform. The draw
                // happens at fire time in the deterministic firing order, so
                // replay stays bit-identical for a fixed (seed, stream).
                let u: f64 = rand::Rng::random(&mut self.rng);
                let mut cumulative = 0.0;
                for (branch, p) in probs.iter().enumerate() {
                    cumulative += *p;
                    if u < cumulative {
                        return Ok(transition.targets[branch]);
                    }
                }
                // `u` within rounding of 1.0: the last branch.
                Ok(transition.targets[probs.len() - 1])
            }
        }
    }

    /// `propagate_effects`: propagate sensitive functions to a fixpoint in the
    /// documented deterministic order (ascending function index).
    pub(super) fn run_fixpoint(&mut self) -> Result<(), EngineError> {
        if self.config.confluence_check {
            self.confluence_probe()
        } else {
            self.converge(false)
        }
    }

    /// Apply one function's effects; when `trigger` is set, attribute
    /// changes enqueue their dependent functions.
    /// A transition's edge effects (`evolT`'s own writes): evaluated once,
    /// in declaration order, on the state the firing left, and never
    /// re-applied. Each change triggers what reads it, exactly as a
    /// sensitive function's does.
    fn apply_transition_effects(&mut self, trans_idx: TransIdx) -> Result<(), EngineError> {
        let transition = &self.model.transitions[trans_idx];
        for (target, value_expr) in &transition.effects {
            let new = eval_expr(self.model, &self.vars, &self.states, self.time, value_expr)?;
            let old = self.vars[*target];
            if old != new {
                self.vars[*target] = new;
                self.note_var_change(*target);
                if self.config.journal {
                    self.journal.push(JournalRecord::AttributeChanged {
                        time: self.time,
                        attribute: self.model.var_names[*target].clone(),
                        old,
                        new,
                        cause: transition.name.clone(),
                    });
                }
                self.worklist
                    .extend(self.model.var_triggers[*target].iter().copied());
            }
        }
        Ok(())
    }

    fn apply_function(&mut self, fn_idx: FnIdx, trigger: bool) -> Result<(), EngineError> {
        let function = &self.model.functions[fn_idx];
        for (target, value_expr) in &function.effects {
            let new = eval_expr(self.model, &self.vars, &self.states, self.time, value_expr)?;
            let old = self.vars[*target];
            if old != new {
                self.vars[*target] = new;
                // Indexed watched set: invalidate the guards reading it.
                // The comparison above is the sensitive functions' own,
                // older than the explicit pass's: this site was already a
                // change *detector*, it only lacked a consumer.
                self.note_var_change(*target);
                if self.config.journal {
                    self.journal.push(JournalRecord::AttributeChanged {
                        time: self.time,
                        attribute: self.model.var_names[*target].clone(),
                        old,
                        new,
                        cause: function.name.clone(),
                    });
                }
                if trigger {
                    self.worklist
                        .extend(self.model.var_triggers[*target].iter().copied());
                }
            }
        }
        Ok(())
    }

    /// Non-confluence diagnostic: converge a *copy* of the state with
    /// the worklist processed in reverse order and compare. Divergence
    /// means the model's result depends on evaluation order: reported
    /// as a typed error (rather than silently picking an arbitrary order).
    fn confluence_probe(&mut self) -> Result<(), EngineError> {
        let saved_vars = self.vars.clone();
        let saved_worklist = self.worklist.clone();

        // Canonical forward pass (journaled normally).
        self.converge(false)?;
        let forward_vars = std::mem::replace(&mut self.vars, saved_vars);
        // The attribute vector has just been rewound wholesale, and it is
        // rewound again below whichever way the probe ends: discard every
        // cached watched verdict here, once, rather than reason about the
        // union of two passes on every exit path (the diverging one
        // included, which an interactive caller can catch and continue
        // from).
        for stale in &mut self.watched_stale {
            *stale = true;
        }

        // Silent reverse pass on the saved state.
        self.worklist = saved_worklist;
        let journal_flag = std::mem::replace(&mut self.config.journal, false);
        let reverse = self.converge(true);
        self.config.journal = journal_flag;
        reverse?;

        if forward_vars != self.vars {
            // Name the first diverging attribute's writers for the
            // diagnostic message.
            let diverging = forward_vars
                .iter()
                .zip(&self.vars)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            // The distribution operators are named alongside the
            // sensitive functions, though they cannot themselves diverge:
            // they run in the explicit sweep, after the fixpoint, from
            // whatever state it converged to, so both replays recompute
            // them identically. A diverging allocated quantity therefore
            // means a *second* writer, which the model layer refuses
            // (`AllocationTargetWritten`), or a defect in this engine: in
            // either case naming the operator is what makes the report
            // actionable, and leaving it out is the "divergence with no
            // writer name" this probe must never produce.
            let operators = self.model.explicit.iter().filter_map(|step| match step {
                CStep::Allocate(allocation) if allocation.allocated.contains(&diverging) => {
                    Some(allocation.name.clone())
                }
                _ => None,
            });
            let mut writers = self
                .model
                .functions
                .iter()
                .filter(|f| f.effects.iter().any(|(target, _)| *target == diverging))
                .map(|f| f.name.clone())
                .chain(operators);
            let first = writers.next().unwrap_or_else(|| "<unknown>".to_owned());
            let second = writers.next().unwrap_or_else(|| first.clone());
            return Err(EngineError::NonConfluent {
                time: self.time,
                first,
                second,
            });
        }
        // Both orders agree: keep the forward result as canonical (the
        // cached verdicts were already discarded above).
        self.vars = forward_vars;
        Ok(())
    }

    /// Converge the current worklist, ascending or descending order.
    fn converge(&mut self, reverse: bool) -> Result<(), EngineError> {
        let mut iterations = 0usize;
        loop {
            let next = if reverse {
                self.worklist.pop_last()
            } else {
                self.worklist.pop_first()
            };
            let Some(fn_idx) = next else { break };
            iterations += 1;
            if iterations > self.config.max_fixpoint_iterations {
                self.worklist.clear();
                return Err(EngineError::InstantaneousLoop {
                    time: self.time,
                    iterations: self.config.max_fixpoint_iterations,
                });
            }
            if self.config.journal {
                self.journal.push(JournalRecord::FunctionTriggered {
                    time: self.time,
                    function: self.model.functions[fn_idx].name.clone(),
                });
            }
            self.apply_function(fn_idx, true)?;
        }
        Ok(())
    }
}
