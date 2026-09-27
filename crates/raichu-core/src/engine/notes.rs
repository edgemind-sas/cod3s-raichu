//! Change notes: the change log of the explicit sweep, the indexed watched
//! set kept current by the `note_*` methods, and the causal-journal notes
//! of flow restarts and active-set crossings.

use super::*;

/// **Change detection of the explicit sweep**: every write goes through
/// here, and the ones that actually move a value are reported.
///
/// The sweep rewrites each of its targets at every evaluation point,
/// whether or not the new value differs from the one already there. That
/// is fine as an *assignment*, and useless as a *signal*: an index that
/// tells the engine which margins a change reaches is worth nothing if
/// every attribute is announced as changed on every pass. Comparing
/// before writing is what turns the rewrite into a signal.
///
/// **The comparison is unconditional**, paid by every model, including
/// the discrete-only ones that carry no watched transition and receive
/// nothing in exchange. Only the *recording* is optional: the passes
/// whose result is thrown away (the solver callbacks, the dense-sample
/// callback, the active-set probe) run against a copied attribute vector
/// and have no state to invalidate.
///
/// Floats are compared at the **flow tolerance**, the same number the
/// flow resolution settles to, so a quantity still creeping inside the
/// band the resolution has already declared settled is not announced as a
/// move. Everything else compares exactly.
pub(super) struct ChangeLog {
    /// The band a float write must leave to count as a move
    /// ([`FlowConfig::tolerance`]).
    pub(super) tolerance: f64,
    /// Targets that moved, in write order, or `None` on a pass whose
    /// result is discarded.
    pub(super) moved: Option<Vec<VarIdx>>,
}

impl ChangeLog {
    /// A log that compares but records nothing: the passes run on a copy
    /// of the attribute vector.
    pub(super) fn discarding(tolerance: f64) -> Self {
        ChangeLog {
            tolerance,
            moved: None,
        }
    }

    /// Write one target, reporting it when the value moved.
    ///
    /// Two tests, cheapest first. **Identity** settles the common case:
    /// a sweep of a settled network recomputes most of its targets to the
    /// bits they already held, and that is one comparison. Only a genuine
    /// difference is worth the banded test, which costs a handful of
    /// operations and is where the tolerance enters.
    #[inline]
    pub(super) fn write(&mut self, vars: &mut [Value], target: VarIdx, value: Value) {
        let old = vars[target];
        vars[target] = value;
        if old == value {
            return;
        }
        if !value_settled(&old, &value, self.tolerance) {
            if let Some(sink) = self.moved.as_mut() {
                sink.push(target);
            }
        }
    }
}

impl<'m> Engine<'m> {
    /// Limit-cycle guard on the flow side: the active set must not
    /// restart segments past its budget over the whole run.
    ///
    /// [`EngineError::FlowChattering`] resets as soon as the clock moves
    /// by one event-location tolerance, so it sees only a cycle frozen
    /// in time. This one counts across the run, which is what catches a
    /// network restarting a segment on every accepted solver step: no
    /// single instant is stuck, and the run grinds instead of failing.
    pub(super) fn note_flow_restart(&mut self, stuck: &[usize]) -> Result<(), EngineError> {
        let budget = self.config.max_flow_restarts;
        if budget == 0 {
            return Ok(());
        }
        if self.flow_restarts == 0 {
            self.first_flow_restart = self.time;
        }
        self.flow_restarts += 1;
        if self.flow_restarts < budget {
            return Ok(());
        }
        let since = self.first_flow_restart;
        Err(EngineError::FlowLimitCycle {
            restarts: self.flow_restarts,
            since,
            time: self.time,
            step: if self.flow_restarts > 1 {
                (self.time - since) / (self.flow_restarts - 1) as f64
            } else {
                0.0
            },
            edges: self.edge_names(stuck),
        })
    }

    /// Limit-cycle guard: one transition must not fire past its budget
    /// in a single trajectory.
    ///
    /// The two guards beside this one catch a loop that does not advance
    /// time at all. This one catches the case where time advances by a
    /// little every turn, which is what a volume oscillating across its
    /// bound at the scale of its hysteresis does: the run never fails,
    /// it simply never ends, and the trajectory looks right at every
    /// sample instant while it happens. The mean step reported is what
    /// makes the diagnosis land, being a numerical scale and not a
    /// physical one.
    pub(super) fn note_firing(&mut self, trans_idx: TransIdx) -> Result<(), EngineError> {
        let budget = self.config.max_transition_firings;
        if budget == 0 {
            return Ok(());
        }
        if self.firings[trans_idx] == 0 {
            self.first_firing[trans_idx] = self.time;
        }
        self.firings[trans_idx] += 1;
        if self.firings[trans_idx] < budget {
            return Ok(());
        }
        let since = self.first_firing[trans_idx];
        let span = self.time - since;
        Err(EngineError::TransitionChattering {
            transition: self.model.transitions[trans_idx].name.clone(),
            firings: self.firings[trans_idx],
            since,
            time: self.time,
            step: if self.firings[trans_idx] > 1 {
                span / (self.firings[trans_idx] - 1) as f64
            } else {
                0.0
            },
        })
    }

    /// Zeno guard: watched transitions must not keep firing without
    /// time advancing.
    pub(super) fn note_watched_firing(&mut self) -> Result<(), EngineError> {
        if self.watched_streak.0 == self.time {
            self.watched_streak.1 += 1;
            if self.watched_streak.1 > 1_000 {
                return Err(EngineError::WatchedLoop { time: self.time });
            }
        } else {
            self.watched_streak = (self.time, 1);
        }
        Ok(())
    }

    // ---- the indexed watched set -------------------------------------
    //
    // Three pieces of bookkeeping keep the two scan sites proportional to
    // what moved: which positions are *armed* (their automaton sits in
    // the source state), what each guard last *evaluated to*, and which
    // of those verdicts a change has *invalidated*. Every mutation of the
    // attribute vector, of an automaton state or of the clock passes
    // through one of the `note_*` methods below; forgetting one would
    // leave a verdict cached against a state that no longer holds, which
    // is why the full rebuild is the reinstatement path of every
    // wholesale state change (build, reset, restore, confluence probe).

    /// Rebuild the whole index from the current state: arming derived
    /// from the automaton states, every cached verdict discarded.
    ///
    /// The reinstatement path of a wholesale state change. It is the only
    /// place the watched population is walked in full, and it is
    /// deliberately not on the simulation cycle.
    pub(super) fn rebuild_watched_index(&mut self) {
        let model = self.model;
        self.watched_active.clear();
        for (position, &trans_idx) in model.watched.iter().enumerate() {
            let transition = &model.transitions[trans_idx];
            let armed = self.states[transition.automaton] == transition.source;
            self.watched_armed[position] = armed;
            self.watched_stale[position] = true;
            if armed {
                self.watched_active.push(position);
            }
        }
    }

    /// An attribute moved: invalidate the guards that read it.
    #[inline]
    pub(super) fn note_var_change(&mut self, var: VarIdx) {
        let model = self.model;
        for &position in &model.margin_index.watched_by_var[var] {
            self.watched_stale[position] = true;
        }
    }

    /// The clock moved: invalidate the guards that read it. Normally a
    /// no-op, a boundary being a predicate on the continuous state.
    #[inline]
    pub(super) fn note_time_change(&mut self) {
        let model = self.model;
        for &position in &model.margin_index.watched_by_time {
            self.watched_stale[position] = true;
        }
    }

    /// An automaton changed state: invalidate the guards that read that
    /// state, and re-arm the transitions the automaton owns.
    ///
    /// The arming update keeps `watched_active` **ascending**, inserting
    /// and removing by binary search rather than re-deriving the list, so
    /// the per-segment margin set and the guard scan visit the watched
    /// transitions in the order the full scan visited them.
    pub(super) fn note_state_change(&mut self, automaton: AutIdx) {
        let model = self.model;
        for &position in &model.margin_index.watched_by_state[automaton] {
            self.watched_stale[position] = true;
        }
        let state = self.states[automaton];
        for &position in &model.margin_index.watched_by_owner[automaton] {
            let armed = model.transitions[model.watched[position]].source == state;
            if armed == self.watched_armed[position] {
                continue;
            }
            self.watched_armed[position] = armed;
            match self.watched_active.binary_search(&position) {
                Ok(at) if !armed => {
                    self.watched_active.remove(at);
                }
                Err(at) if armed => {
                    self.watched_active.insert(at, position);
                }
                _ => {}
            }
        }
    }

    /// Drain what the engine's own explicit passes reported (see
    /// [`ChangeLog`]) into the stale marks, keeping the buffer for reuse.
    pub(super) fn drain_changes(&mut self) {
        let Some(mut moved) = self.changed.moved.take() else {
            return;
        };
        let model = self.model;
        for &var in &moved {
            for &position in &model.margin_index.watched_by_var[var] {
                self.watched_stale[position] = true;
            }
        }
        moved.clear();
        self.changed.moved = Some(moved);
    }

    /// A watched transition whose *guard* already holds while its
    /// automaton sits in the source state fires immediately.
    ///
    /// The guard is evaluated exactly (boolean), not through the
    /// margin: after a located crossing, a sibling transition sharing
    /// the boundary may sit within round-off of it: its strict guard
    /// is already true while its ε-tightened margin is still negative.
    /// Conversely a trajectory *resting* exactly on a strict boundary
    /// keeps a false guard and does not fire (no Zeno).
    ///
    /// **Indexed.** The scan walks the armed positions in ascending
    /// order, exactly as the full scan did, but re-evaluates only the
    /// guards a change has invalidated; the rest answer from their cached
    /// verdict. The short-circuit is preserved with the order: the scan
    /// still stops at the first armed guard that holds, so a guard past
    /// it is neither evaluated nor cleared, and a guard that would raise
    /// on a state the model never reaches stays unevaluated exactly as
    /// before.
    pub(super) fn immediate_watched(&mut self) -> Result<Option<TransIdx>, EngineError> {
        let model = self.model;
        for index in 0..self.watched_active.len() {
            let position = self.watched_active[index];
            let trans_idx = model.watched[position];
            let Some(guard) = &model.transitions[trans_idx].guard else {
                continue;
            };
            if self.watched_stale[position] {
                // Counted work: this scan is one of the two sites the
                // index narrows, so its cost is measured here.
                self.work.immediate_guard_scans += 1;
                self.watched_guard[position] =
                    eval_bool(model, &self.vars, &self.states, self.time, guard)?;
                self.watched_stale[position] = false;
            }
            if self.watched_guard[position] {
                return Ok(Some(trans_idx));
            }
        }
        Ok(None)
    }

    /// Record a located active-set crossing in the causal journal, naming
    /// the operator, the edge, and the two saturation classes it moved
    /// between. The network has already been resolved again, so the
    /// destination class is read from the fresh resolution.
    ///
    /// Zero cost when the journal is off, which is the default.
    pub(super) fn journal_active_set_crossing(
        &mut self,
        flows: &[FrozenFlow<'_>],
        index: usize,
    ) -> Result<(), EngineError> {
        if !self.config.journal {
            return Ok(());
        }
        let mut offset = 0usize;
        let mut located: Option<(&FrozenFlow<'_>, usize, usize)> = None;
        for flow in flows {
            if index < offset + flow.classes.len() {
                located = Some((flow, index - offset, offset));
                break;
            }
            offset += flow.classes.len();
        }
        let Some((flow, edge, base)) = located else {
            return Ok(());
        };
        let after = self.frozen_classes()?;
        let record = JournalRecord::ActiveSetCrossed {
            time: self.time,
            operator: flow.margins.name.clone(),
            consumer: flow
                .margins
                .consumers
                .get(edge)
                .cloned()
                .unwrap_or_else(|| format!("edge #{edge}")),
            from: flow.classes[edge],
            to: after
                .get(base + edge)
                .copied()
                .unwrap_or(flow.classes[edge]),
        };
        self.journal.push(record);
        Ok(())
    }
}
