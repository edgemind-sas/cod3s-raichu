//! The run loop and discrete evolution: stepping and running to the
//! horizon, firing the earliest transition (`fire_transition`), and the
//! sensitive-function propagation to fixpoint (`propagate_effects`) with
//! its optional confluence probe.

use super::*;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};

type ProgramCacheKey = (u64, usize, Vec<u64>);
const PROGRAM_CACHE_LIMIT: usize = 1024;

#[derive(Default)]
struct ProgramCache {
    outcomes: HashMap<ProgramCacheKey, raichu_milp::Outcome>,
    order: VecDeque<ProgramCacheKey>,
}

thread_local! {
    static PROGRAM_CACHE: RefCell<ProgramCache> = RefCell::new(ProgramCache::default());
}

fn program_key(program: &raichu_milp::Program, tie_break: &raichu_milp::TieBreak) -> Vec<u64> {
    let mut bits = vec![
        program.columns.len() as u64,
        program.rows.len() as u64,
        u64::from(program.sense == raichu_milp::Sense::Maximize),
        program.node_limit.unwrap_or(100_000),
    ];
    for column in &program.columns {
        bits.extend([
            column.cost.to_bits(),
            column.lower.to_bits(),
            column.upper.to_bits(),
            match column.kind {
                raichu_milp::Kind::Continuous => 0,
                raichu_milp::Kind::Integer => 1,
                raichu_milp::Kind::Binary => 2,
            },
        ]);
    }
    for row in &program.rows {
        bits.extend([
            row.lower.to_bits(),
            row.upper.to_bits(),
            row.terms.len() as u64,
        ]);
        for (column, coefficient) in &row.terms {
            bits.extend([*column as u64, coefficient.to_bits()]);
        }
    }
    match tie_break {
        raichu_milp::TieBreak::Default => bits.push(0),
        raichu_milp::TieBreak::None => bits.push(1),
        raichu_milp::TieBreak::Objectives(objectives) => {
            bits.extend([2, objectives.len() as u64]);
            for objective in objectives {
                bits.push(u64::from(objective.sense == raichu_milp::Sense::Maximize));
                bits.extend(objective.coefficients.iter().map(|value| value.to_bits()));
            }
        }
    }
    bits
}

fn solve_cached(
    model_id: u64,
    index: usize,
    program: &raichu_milp::Program,
    tie_break: &raichu_milp::TieBreak,
) -> (raichu_milp::Outcome, bool) {
    let key = (model_id, index, program_key(program, tie_break));
    if let Some(outcome) = PROGRAM_CACHE.with(|cache| cache.borrow().outcomes.get(&key).cloned()) {
        return (outcome, true);
    }
    let outcome = raichu_milp::solve(program, tie_break);
    if matches!(
        outcome,
        raichu_milp::Outcome::Optimal(_) | raichu_milp::Outcome::Infeasible
    ) {
        PROGRAM_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.order.len() == PROGRAM_CACHE_LIMIT {
                if let Some(oldest) = cache.order.pop_front() {
                    cache.outcomes.remove(&oldest);
                }
            }
            cache.outcomes.insert(key.clone(), outcome.clone());
            cache.order.push_back(key);
        });
    }
    (outcome, false)
}

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
                non_unique_programs: self
                    .model
                    .programs
                    .iter()
                    .filter(|program| matches!(program.tie_break, CProgramTieBreak::None))
                    .map(|program| program.name.clone())
                    .collect(),
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

    fn program_affine(
        &self,
        expr: &CProgramAffine,
        columns: usize,
    ) -> Result<(f64, Vec<f64>), EngineError> {
        let constant = eval_f64(
            self.model,
            &self.vars,
            &self.states,
            self.time,
            &expr.constant,
        )?;
        let mut coefficients = vec![0.0; columns];
        for (column, coefficient) in &expr.terms {
            coefficients[*column] +=
                eval_f64(self.model, &self.vars, &self.states, self.time, coefficient)?;
        }
        Ok((constant, coefficients))
    }

    fn program_number(&self, expr: Option<&CExpr>, default: f64) -> Result<f64, EngineError> {
        expr.map(|expr| eval_f64(self.model, &self.vars, &self.states, self.time, expr))
            .unwrap_or(Ok(default))
    }

    fn program_write(&mut self, target: VarIdx, value: Value, name: &str, trigger: bool) {
        let old = self.vars[target];
        if old == value {
            return;
        }
        self.vars[target] = value;
        self.note_var_change(target);
        if self.config.journal {
            self.journal.push(JournalRecord::AttributeChanged {
                time: self.time,
                attribute: self.model.var_names[target].clone(),
                old,
                new: value,
                cause: name.to_owned(),
            });
        }
        if trigger {
            self.worklist
                .extend(self.model.var_triggers[target].iter().copied());
        }
    }

    fn apply_program(&mut self, program_idx: usize, trigger: bool) -> Result<(), EngineError> {
        let program = &self.model.programs[program_idx];
        let columns = program.variables.len();
        let (objective_offset, costs) = self.program_affine(&program.objective, columns)?;
        if !objective_offset.is_finite() {
            return Err(EngineError::ProgramFailed {
                program: program.name.clone(),
                time: self.time,
                reason: "non-finite objective offset".to_owned(),
            });
        }
        let mut numeric_columns = Vec::with_capacity(columns);
        for (variable, cost) in program.variables.iter().zip(costs) {
            let kind = match variable.kind {
                raichu_model::AttrKind::Bool => raichu_milp::Kind::Binary,
                raichu_model::AttrKind::Int => raichu_milp::Kind::Integer,
                raichu_model::AttrKind::Float => raichu_milp::Kind::Continuous,
            };
            numeric_columns.push(raichu_milp::Column {
                cost,
                lower: self.program_number(
                    variable.lower.as_ref(),
                    if kind == raichu_milp::Kind::Binary {
                        0.0
                    } else {
                        f64::NEG_INFINITY
                    },
                )?,
                upper: self.program_number(
                    variable.upper.as_ref(),
                    if kind == raichu_milp::Kind::Binary {
                        1.0
                    } else {
                        f64::INFINITY
                    },
                )?,
                kind,
            });
        }
        let mut rows = Vec::with_capacity(program.constraints.len());
        for constraint in &program.constraints {
            let (offset, coefficients) = self.program_affine(&constraint.expr, columns)?;
            rows.push(raichu_milp::Row {
                terms: coefficients
                    .into_iter()
                    .enumerate()
                    .filter(|(_, value)| *value != 0.0)
                    .collect(),
                lower: self.program_number(constraint.lower.as_ref(), f64::NEG_INFINITY)? - offset,
                upper: self.program_number(constraint.upper.as_ref(), f64::INFINITY)? - offset,
            });
        }
        let tie_break = match &program.tie_break {
            CProgramTieBreak::Default => raichu_milp::TieBreak::Default,
            CProgramTieBreak::None => raichu_milp::TieBreak::None,
            CProgramTieBreak::Objectives(objectives) => {
                let mut numeric = Vec::with_capacity(objectives.len());
                for (sense, expr) in objectives {
                    let (offset, coefficients) = self.program_affine(expr, columns)?;
                    if !offset.is_finite() {
                        return Err(EngineError::ProgramFailed {
                            program: program.name.clone(),
                            time: self.time,
                            reason: "non-finite secondary objective offset".to_owned(),
                        });
                    }
                    numeric.push(raichu_milp::Objective {
                        sense: *sense,
                        coefficients,
                    });
                }
                raichu_milp::TieBreak::Objectives(numeric)
            }
        };
        let numeric = raichu_milp::Program {
            columns: numeric_columns,
            rows,
            sense: program.sense,
            node_limit: program.node_limit,
        };
        let (outcome, cached) =
            solve_cached(self.model.cache_id, program_idx, &numeric, &tie_break);
        match outcome {
            raichu_milp::Outcome::Optimal(solution) => {
                let objective = objective_offset + solution.objective;
                if !objective.is_finite() {
                    return Err(EngineError::ProgramFailed {
                        program: program.name.clone(),
                        time: self.time,
                        reason: "non-finite objective value".to_owned(),
                    });
                }
                for (variable, value) in program.variables.iter().zip(&solution.values) {
                    if variable.kind == raichu_model::AttrKind::Int
                        && (!value.is_finite() || value.abs() > 9_007_199_254_740_992.0)
                    {
                        return Err(EngineError::ProgramFailed {
                            program: program.name.clone(),
                            time: self.time,
                            reason: "integer decision exceeds exact f64 range".to_owned(),
                        });
                    }
                }
                if self.config.journal {
                    self.journal.push(JournalRecord::ProgramSolved {
                        time: self.time,
                        program: program.name.clone(),
                        status: "optimal",
                        objective: Some(objective),
                        cached,
                    });
                }
                for (variable, value) in program.variables.iter().zip(solution.values) {
                    let typed = match variable.kind {
                        raichu_model::AttrKind::Bool => Value::Bool(value >= 0.5),
                        raichu_model::AttrKind::Int => Value::Int(value as i64),
                        raichu_model::AttrKind::Float => Value::Float(value),
                    };
                    self.program_write(variable.target, typed, &program.name, trigger);
                }
                self.program_write(program.feasible, Value::Bool(true), &program.name, trigger);
                if let Some(target) = program.objective_value {
                    self.program_write(target, Value::Float(objective), &program.name, trigger);
                }
            }
            raichu_milp::Outcome::Infeasible => {
                if self.config.journal {
                    self.journal.push(JournalRecord::ProgramSolved {
                        time: self.time,
                        program: program.name.clone(),
                        status: "infeasible",
                        objective: None,
                        cached,
                    });
                }
                for variable in &program.variables {
                    self.program_write(variable.target, variable.fallback, &program.name, trigger);
                }
                self.program_write(program.feasible, Value::Bool(false), &program.name, trigger);
                if let Some(target) = program.objective_value {
                    self.program_write(target, Value::Float(0.0), &program.name, trigger);
                }
            }
            other => {
                return Err(EngineError::ProgramFailed {
                    program: program.name.clone(),
                    time: self.time,
                    reason: format!("{other:?}"),
                })
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
                .chain(operators)
                .chain(
                    self.model
                        .programs
                        .iter()
                        .filter(|program| {
                            program
                                .variables
                                .iter()
                                .any(|variable| variable.target == diverging)
                                || program.feasible == diverging
                                || program.objective_value == Some(diverging)
                        })
                        .map(|program| program.name.clone()),
                );
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
            if fn_idx < self.model.functions.len() {
                if self.config.journal {
                    self.journal.push(JournalRecord::FunctionTriggered {
                        time: self.time,
                        function: self.model.functions[fn_idx].name.clone(),
                    });
                }
                self.apply_function(fn_idx, true)?;
            } else {
                self.apply_program(fn_idx - self.model.functions.len(), true)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod program_cache_tests {
    use super::*;

    #[test]
    fn final_bit_of_a_coefficient_changes_the_key() {
        let mut program = raichu_milp::Program {
            columns: vec![raichu_milp::Column {
                cost: 1.0,
                lower: 0.0,
                upper: 2.0,
                kind: raichu_milp::Kind::Continuous,
            }],
            rows: vec![],
            sense: raichu_milp::Sense::Minimize,
            node_limit: None,
        };
        let original = program_key(&program, &raichu_milp::TieBreak::Default);
        program.columns[0].cost = f64::from_bits(1.0f64.to_bits() + 1);
        assert_ne!(
            original,
            program_key(&program, &raichu_milp::TieBreak::Default)
        );
    }

    #[test]
    fn cached_and_uncached_solves_match_across_threads() {
        let program = raichu_milp::Program {
            columns: vec![raichu_milp::Column {
                cost: 1.0,
                lower: 0.0,
                upper: 2.0,
                kind: raichu_milp::Kind::Continuous,
            }],
            rows: vec![],
            sense: raichu_milp::Sense::Minimize,
            node_limit: None,
        };
        for threads in [1, 4] {
            std::thread::scope(|scope| {
                let workers: Vec<_> = (0..threads)
                    .map(|worker| {
                        let program = &program;
                        scope.spawn(move || {
                            for replica in (worker..1000).step_by(threads) {
                                let (cached, _) = solve_cached(
                                    10_000 + threads as u64,
                                    replica % 2,
                                    program,
                                    &raichu_milp::TieBreak::Default,
                                );
                                let uncached =
                                    raichu_milp::solve(program, &raichu_milp::TieBreak::Default);
                                assert_eq!(cached, uncached);
                            }
                        })
                    })
                    .collect();
                for worker in workers {
                    assert!(worker.join().is_ok());
                }
            });
        }
    }
}
