//! Continuous evolution up to the next scheduled date
//! (`integrate_continuous`): integration of the ODE system, event location
//! of watched boundary crossings (`schedule_boundary`) and of
//! continuously-varying hazards (`reschedule_modifiable`), and the segment
//! outcome handed back to the run loop.

use super::*;

/// How one integration segment ended.
///
/// Besides reaching the requested date and
/// locating a watched transition, a segment can end because the **active
/// set** it froze stopped holding. That third outcome fires no
/// transition; it re-resolves the network at the crossing instant and
/// integration continues from there. A discarded trial can also request
/// reintegration of its prefix without committing state.
#[derive(Debug)]
enum Segment {
    /// Discard a trial that probed beyond an earlier event and integrate
    /// its prefix again before committing any continuous state.
    Retry { failed_at: f64, error: EngineError },
    /// The requested date was reached.
    Reached,
    /// A watched boundary (or a continuously-varying hazard) was
    /// located: this transition must fire.
    Watched(TransIdx),
    /// An active-set boundary was located and the network was resolved
    /// again from the crossing state. Carries the global edge index that
    /// crossed, so a chattering boundary can be *named* rather than
    /// merely counted.
    Resolved(usize),
}

impl<'m> Engine<'m> {
    /// Advance continuous evolution to `t_target`, restarting a segment
    /// at every located active-set crossing until either the date is
    /// reached or a watched transition must fire.
    ///
    /// The loop is guarded the way [`Engine::note_watched_firing`] guards
    /// watched transitions, and for the same reason: a crossing that
    /// fires no transition is invisible to that guard, so a boundary the
    /// network re-crosses on the spot would spin here forever. Restarts
    /// are counted **per instant** rather than per advance: an instant
    /// admits at most one class change per compiled edge
    /// ([`active_set_budget`]) before the boundary is chattering
    /// rather than moving, whereas a long horizon legitimately crosses
    /// any number of boundaries. Progress is measured against the
    /// event-location tolerance, so a crossing located a hair after the
    /// previous one does not pass for progress.
    pub(super) fn advance_continuous(
        &mut self,
        t_target: f64,
    ) -> Result<Option<TransIdx>, EngineError> {
        let mut stuck_at = f64::NEG_INFINITY;
        let mut stuck: Vec<usize> = Vec::new();
        let mut segment_target = t_target;
        let mut pending_error: Option<(f64, EngineError)> = None;
        loop {
            let mut retrying = false;
            match self.integrate_to(segment_target)? {
                Segment::Retry { failed_at, error } => {
                    retrying = true;
                    // Keep the earliest failed probe as a barrier. A new
                    // clean prefix does not prove that this singularity
                    // disappeared: it may simply not probe the same date.
                    if pending_error
                        .as_ref()
                        .is_none_or(|(time, _)| failed_at < *time)
                    {
                        pending_error = Some((failed_at, error));
                    }
                }
                Segment::Reached if pending_error.is_none() => return Ok(None),
                Segment::Reached => {}
                Segment::Watched(trans_idx) => return Ok(Some(trans_idx)),
                Segment::Resolved(edge) => {
                    pending_error = None;
                    segment_target = t_target;
                    if self.time > stuck_at + self.config.ode.tol_event {
                        stuck_at = self.time;
                        stuck.clear();
                    }
                    stuck.push(edge);
                    if stuck.len() > self.active_set_budget {
                        return Err(EngineError::FlowChattering {
                            time: self.time,
                            restarts: stuck.len(),
                            edges: self.edge_names(&stuck),
                        });
                    }
                    self.note_flow_restart(&stuck)?;
                    if self.time >= t_target {
                        return Ok(None);
                    }
                    continue;
                }
            }
            if let Some((failed_at, error)) = pending_error.take() {
                // Do not use an event estimate from the failed trial:
                // its guard may have read stale explicit values, giving
                // either a false crossing or a false absence of one.
                // Approach the barrier through fresh, clean prefixes.
                // Only a validated event can end this frozen segment;
                // without one, the original typed error remains fatal.
                let ceiling = if retrying { segment_target } else { t_target };
                let until = self.time + (failed_at.min(ceiling) - self.time) * 0.5;
                if until <= self.time + self.config.ode.tol_event
                    || (retrying && until >= segment_target)
                {
                    return Err(error);
                }
                segment_target = until;
                pending_error = Some((failed_at, error));
            }
        }
    }

    /// Whether continuous evolution must run: the model has ODE
    /// attributes, an armed hazard varies continuously
    /// (`reschedule_modifiable` under `integrate_continuous`: possibly
    /// with no ODE at all, e.g. a time-dependent rate), or a step of the
    /// explicit sweep reads the clock.
    ///
    /// The third case has no state behind it at all. A declared time
    /// profile is an explicit equation over `time` and nothing else, so
    /// without this the engine would find nothing to advance, evaluate
    /// the sweep once at the initial instant, and report that value at
    /// every sample instant and to every watched guard for the rest of
    /// the run: a curve reported as a constant, with nothing to signal
    /// it.
    pub(super) fn needs_integration(&self) -> bool {
        !self.model.ode.is_empty()
            || self.model.explicit_reads_time
            || self.continuous_rates.iter().any(|&idx| self.is_armed(idx))
    }

    /// `integrate_continuous`: integrate the continuous state to `t_target`, monitoring
    /// active watched boundaries. Returns the watched transition to fire
    /// if a crossing was located first (time/state already advanced).
    fn integrate_to(&mut self, t_target: f64) -> Result<Segment, EngineError> {
        // **Indexed.** The armed set is maintained on state changes, not
        // re-derived here: a segment restarted at a located active-set
        // crossing changes no automaton state, so the set it monitors is
        // the one the previous segment monitored, and rebuilding it would
        // cost the model's whole watched population per restart.
        let margins: Vec<TransIdx> = self
            .watched_active
            .iter()
            .map(|&position| self.model.watched[position])
            .collect();

        // Armed continuously-varying hazards ride along as auxiliary
        // state (`dH/dt = λ`), their firing located as an event.
        let hazard_monitors: Vec<(TransIdx, f64)> = self
            .continuous_rates
            .iter()
            .copied()
            .filter(|&idx| self.is_armed(idx))
            .filter_map(|idx| {
                self.hazards[idx].map(|hazard| (idx, hazard.threshold - hazard.accumulated))
            })
            .collect();

        let mut y: Vec<f64> = Vec::with_capacity(self.model.ode.len() + hazard_monitors.len());
        for (var, _) in &self.model.ode {
            match self.vars[*var] {
                Value::Float(f) => y.push(f),
                other => {
                    return Err(EngineError::TypeError {
                        time: self.time,
                        detail: format!(
                            "ODE attribute `{}` holds non-float value {other:?}",
                            self.model.var_names[*var]
                        ),
                    });
                }
            }
        }

        y.resize(self.model.ode.len() + hazard_monitors.len(), 0.0);

        // Freeze the active set for this segment. It is *derived* from
        // the resolved state, here, rather than carried from the
        // resolution that produced it: nothing about the search survives
        // outside the attribute vector, so a restored snapshot replays
        // identically.
        let classes = self.frozen_classes()?;
        let mut taken = 0usize;
        let mut flows: Vec<FrozenFlow<'_>> = Vec::with_capacity(self.model.flow_margins.len());
        for margins in &self.model.flow_margins {
            let CStep::Allocate(allocation) = &self.model.explicit[margins.step] else {
                continue;
            };
            let width = allocation.allocated.len();
            let slice = classes
                .get(taken..taken + width)
                .map(<[EdgeClass]>::to_vec)
                .unwrap_or_default();
            taken += width;
            if slice.len() == width {
                flows.push(FrozenFlow {
                    allocation,
                    margins,
                    classes: slice,
                });
            }
        }

        // Threshold indicators located rather than sampled: an ordering on
        // a number. An equality cannot be crossed on a continuum, and keeps
        // its flips at the samples and events.
        let observed: Vec<usize> = self
            .model
            .indicators
            .iter()
            .enumerate()
            .filter(|(_, indicator)| match indicator.target {
                CIndicatorTarget::Predicate(var, cmp, _) => {
                    !matches!(cmp, CmpOp::Eq | CmpOp::Ne)
                        && matches!(self.vars[var], Value::Float(_))
                }
                _ => false,
            })
            .map(|(index, _)| index)
            .collect();

        let trace = self.hazard_trace.as_ref().map(|_| {
            let bases = hazard_monitors
                .iter()
                .map(|(idx, _)| self.hazards[*idx].map_or(0.0, |h| h.accumulated))
                .collect();
            (bases, Vec::new())
        });
        let mut system = ContinuousSystem {
            model: self.model,
            vars: self.vars.clone(),
            states: self.states.clone(),
            margins,
            hazards: hazard_monitors,
            flows,
            observed,
            flips: Vec::new(),
            trace,
            error: None,
            work: WorkCounters::default(),
            scratch: FlowScratch::default(),
            flow_demands: Vec::new(),
            flow_tolerance: self.config.flow.tolerance,
            sweep_budget: self.config.flow.sweep_budget,
            settle_scratch: Vec::new(),
        };
        self.work.segments += 1;

        // Dense sampling: indicator values recorded from the interpolant.
        let segment_samples: Vec<f64> = self.config.samples[self.sample_cursor..]
            .iter()
            .copied()
            .take_while(|s| *s <= t_target)
            .collect();
        let mut recorded: Vec<(f64, Vec<Value>)> = Vec::new();
        // The dense-sample callback runs its own explicit pass, and the
        // solver holds `system` mutably meanwhile: it counts into its own
        // tally, merged with the system's when the segment returns.
        let mut sample_work = WorkCounters::default();
        let mut sample_scratch = FlowScratch::default();
        // Second half of the error stash. The solver's sample callback is
        // as infallible as its system, and it runs while the solver holds
        // `system` mutably, so it cannot reach `system.error`: without a
        // stash of its own an explicit pass failing at a dense-sample
        // instant would be dropped on the floor, and a sample instant is
        // not necessarily an instant any other callback visits.
        let mut sample_error: Option<EngineError> = None;
        // Recorded from a copy of the attribute vector: nothing to
        // invalidate, so the pass compares and reports nothing.
        let mut sample_changed = ChangeLog::discarding(self.config.flow.tolerance);
        {
            let model = self.model;
            let base_vars = self.vars.clone();
            let states = self.states.clone();
            let mut on_sample = |t: f64, y_at: &[f64]| {
                let mut vars = base_vars.clone();
                for (slot, (var, _)) in model.ode.iter().enumerate() {
                    vars[*var] = Value::Float(y_at[slot]);
                }
                let mut ctx = PassContext {
                    work: &mut sample_work,
                    scratch: &mut sample_scratch,
                    changed: &mut sample_changed,
                };
                if let Err(error) = recompute_explicit(model, &mut vars, &states, t, &mut ctx) {
                    // Stashed, then re-raised below: the callback has no
                    // way to report it and this instant's values are not
                    // recordable.
                    sample_error.get_or_insert(error);
                    return;
                }
                let values = model
                    .indicators
                    .iter()
                    .map(|indicator| indicator_value(&indicator.target, &vars, &states))
                    .collect();
                recorded.push((t, values));
            };

            let outcome = self.solver.integrate(
                &mut system,
                self.time,
                &mut y,
                t_target,
                &segment_samples,
                &mut on_sample,
            );

            self.work.explicit_evaluations +=
                system.work.explicit_evaluations + sample_work.explicit_evaluations;
            self.work.allocation_capping_passes +=
                system.work.allocation_capping_passes + sample_work.allocation_capping_passes;
            self.work.margin_evaluations += system.work.margin_evaluations;

            // Re-raise whatever the callbacks stashed. The time it
            // carries is the *evaluation point* at which the failure was
            // detected: inside a bisection that is a probe of the
            // interval, not a located instant, and it is reported as it
            // stands rather than being rewritten to the segment's
            // committed time, which would name a state that never failed.
            let outcome = match (outcome, system.error.take().or(sample_error)) {
                // A solver stage may evaluate the frozen mode beyond a
                // protective event. Never trust this trial's interpolant
                // or event date: later callbacks may read stale values.
                // Retain the singularity as a barrier while reintegrating
                // fresh prefixes, even when the trial missed the event.
                (_, Some(error @ EngineError::AlgebraicSingular { time, .. })) => {
                    return Ok(Segment::Retry {
                        failed_at: time,
                        error,
                    });
                }
                // Preserve the historical solver-error priority for all
                // other callback failures, including models without blocks.
                (Err(error), _) => return Err(error.into()),
                (Ok(_), Some(error)) => return Err(error),
                (Ok(outcome), None) => outcome,
            };

            if let (Some(recorded), Some((_, samples))) =
                (self.hazard_trace.as_mut(), system.trace.take())
            {
                for sample in samples {
                    if recorded.last().is_none_or(|last| sample.time > last.time) {
                        recorded.push(sample);
                    }
                }
            }

            // Commit the reached continuous state. Three families of
            // event share one index space: watched transitions, then
            // continuously-varying hazards, then the active-set margins
            // of the frozen flows. The flows come last on purpose, so a
            // transition crossing in the same sub-interval wins the tie.
            let watched_and_hazards = system.margins.len() + system.hazards.len();
            let (t_reached, fired, crossed) = match outcome {
                Outcome::Reached { t } => (t, None, None),
                Outcome::Event { index, t } if index < watched_and_hazards => {
                    let fired = if index < system.margins.len() {
                        system.margins[index]
                    } else {
                        system.hazards[index - system.margins.len()].0
                    };
                    (t, Some(fired), None)
                }
                Outcome::Event { index, t } => (t, None, Some(index - watched_and_hazards)),
            };
            self.time = t_reached;
            self.note_time_change();
            // Committing the reached continuous state. Compared
            // **exactly** here, unlike the explicit pass: an integrated
            // attribute is the very thing a watched boundary is a
            // predicate on, and a tolerance band on it would let a
            // trajectory cross a boundary while the guard reading it kept
            // a cached verdict.
            let model = self.model;
            for (slot, (var, _)) in model.ode.iter().enumerate() {
                let value = Value::Float(y[slot]);
                if self.vars[*var] != value {
                    self.vars[*var] = value;
                    self.note_var_change(*var);
                }
            }
            // Bank the hazard accrued over this segment (`reschedule_modifiable`
            // bookkeeping; the fired transition's slot, if any, is
            // cleared by `fire`).
            let ode_len = self.model.ode.len();
            for (slot, (trans_idx, _)) in system.hazards.iter().enumerate() {
                if let Some(hazard) = self.hazards[*trans_idx].as_mut() {
                    hazard.accumulated += y[ode_len + slot].max(0.0);
                    hazard.since = t_reached;
                }
            }
            self.resolve_flows()?;

            // Commit the located flips and the dense samples in time
            // order (samples strictly before the reached time: a sample at
            // exactly an event date is recorded post-event by the flush in
            // the next advance). A flip is a change point at its located
            // date; a sample after it then reads the same verdict and adds
            // none. Interleaved, because a sample read before a LATER flip
            // must not be pushed after it.
            let mut flips = std::mem::take(&mut system.flips).into_iter().peekable();
            let mut commit_flips_through = |until: f64, series: &mut Vec<IndicatorSeries>| {
                while let Some(&(t, indicator, now)) = flips.peek() {
                    if t > until {
                        break;
                    }
                    flips.next();
                    let points = &mut series[indicator].points;
                    let value = Value::Bool(now);
                    if points.last().is_none_or(|(_, last)| *last != value) {
                        points.push((t, value));
                    }
                }
            };
            for (t, values) in recorded {
                commit_flips_through(t, &mut self.indicator_series);
                if t < t_reached || (fired.is_none() && crossed.is_none()) {
                    // A threshold that flipped inside the segment is a
                    // change point of the observation, and nothing else
                    // in the segment records one: no transition fired, no
                    // active set moved. Without this the change-point
                    // series of a threshold on an integrated attribute
                    // keeps its value from the last discrete event
                    // forever, and its `sojourn-time` -- the time the
                    // condition held -- comes back as the whole elapsed
                    // time.
                    //
                    // **Only a threshold, and the asymmetry is the
                    // point.** A threshold is piecewise constant BY
                    // NATURE: between two flips it genuinely does not
                    // move, so a change point is an exact
                    // representation of it and refining the schedule
                    // makes the sojourn converge on the true duration.
                    // A free-valued attribute is not piecewise constant,
                    // so pushing its samples in would turn its sojourn
                    // into a coarse Riemann sum -- closer to the
                    // integral than what it computes today, but a
                    // different quantity from the one every recorded
                    // result was produced with, silently.
                    //
                    // An ordering threshold is located by the solver as an
                    // observation (`ContinuousSystem::observations`), so its
                    // change point is already there at the date it was
                    // crossed, and the sample below adds none. What a
                    // sample still records is a flip nothing located: an
                    // equality threshold.
                    for (((indicator, series), sampled), value) in model
                        .indicators
                        .iter()
                        .zip(self.indicator_series.iter_mut())
                        .zip(self.sampled.iter_mut())
                        .zip(values)
                    {
                        sampled.points.push((t, value));
                        if matches!(indicator.target, CIndicatorTarget::Predicate(..))
                            && series.points.last().is_none_or(|(_, last)| *last != value)
                        {
                            series.points.push((t, value));
                        }
                    }
                    self.sample_cursor += 1;
                }
            }
            commit_flips_through(t_reached, &mut self.indicator_series);
            match crossed {
                None => Ok(fired.map_or(Segment::Reached, Segment::Watched)),
                Some(index) => {
                    // A located active-set crossing is a change point of
                    // the network like a fired transition is one of the
                    // discrete state: the resolved quantities take a new
                    // shape there. `record_indicators` appends only on a
                    // change, so a model with no operator (and therefore
                    // no crossing) never reaches this and keeps its
                    // series exactly as before.
                    self.record_indicators();
                    self.journal_active_set_crossing(&system.flows, index)?;
                    Ok(Segment::Resolved(index))
                }
            }
        }
    }
}
