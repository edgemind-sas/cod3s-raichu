//! Scheduling: `schedule_deterministic`, `schedule_stochastic`,
//! `reschedule_modifiable` (cumulative hazards of state-dependent rates)
//! and `drop_disabled`, all carried out by [`Engine::refresh_transition`]
//! (run after a discrete step by [`Engine::refresh_affected`] on what the
//! step reached, over the whole model by [`Engine::refresh_schedule`]),
//! plus the law classification and sampling helpers they rely on.

use super::*;

/// Cumulative-hazard state of an armed state-dependent-rate transition
/// (`CLaw::ExpVar`): the transition fires when `accumulated` reaches
/// `threshold`, realising the PDMP survival `P(T > t) = exp(−∫λ dt)`
/// exactly. The threshold is drawn `Exp(1)` at arming (`schedule_stochastic`).
#[derive(Debug, Clone, Copy)]
pub(super) struct Hazard {
    /// `Exp(1)` firing threshold `E`.
    pub(super) threshold: f64,
    /// Hazard accumulated so far, `H = ∫ λ dt ≤ E`.
    pub(super) accumulated: f64,
    /// Rate λ at `since`: supports the lazy piecewise-constant
    /// accumulation of non-continuous rates between discrete steps.
    pub(super) rate: f64,
    /// Time of the last accumulation point.
    pub(super) since: f64,
}

/// Running tally of the per-transition sufficient statistics
/// ([`TransitionExposure`]) of a run with [`EngineConfig::rate_factors`].
///
/// Exposure is accrued lazily: in drawn mode a constant-rate exponential
/// transition is armed and running exactly while it holds a date in
/// `pending` (a `reset` drop, a `resume` pause, a source exit and a
/// firing all clear it; a `continue` transition keeps it through a false
/// guard), and `pending` changes only at discrete events. Accruing
/// `λ · (t − since)` over every dated `CLaw::Exp` transition just before
/// each firing, and once at the final time, therefore integrates the
/// running time exactly.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ExposureTally {
    /// Statistics per transition, indexed like the compiled transitions.
    pub(super) stats: Vec<TransitionExposure>,
    /// Instant up to which `stats` holds the accrued exposure.
    pub(super) since: f64,
    /// `(transition index, nominal rate)` of every constant-rate
    /// exponential transition, the only ones that accrue exposure.
    exponential: Vec<(usize, f64)>,
}

impl ExposureTally {
    /// A fresh tally at `t = 0` for the transitions of `model`.
    pub(super) fn new(model: &CompiledModel) -> Self {
        ExposureTally {
            stats: vec![TransitionExposure::default(); model.transitions.len()],
            since: 0.0,
            exponential: model
                .transitions
                .iter()
                .enumerate()
                .filter_map(|(idx, t)| match t.distrib {
                    CLaw::Exp(rate) => Some((idx, rate)),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Accrue the nominal exposure of every dated constant-rate
    /// exponential transition from `since` to `until`.
    pub(super) fn accrue(&mut self, pending: &[Option<f64>], until: f64) {
        let span = until - self.since;
        if span <= 0.0 {
            return;
        }
        for &(idx, rate) in &self.exponential {
            if pending[idx].is_some() {
                self.stats[idx].exposure += rate * span;
            }
        }
        self.since = until;
    }
}

/// Check [`EngineConfig::ode`] against the domain of each field
/// ([`SolverParams::validate`]), before a run can cost anything: a NaN
/// step cap or a negative tolerance would otherwise be dropped silently
/// or surface as a solver failure that reads as model stiffness.
pub(super) fn validate_solver_params(config: &EngineConfig) -> Result<(), EngineError> {
    config
        .ode
        .validate()
        .map_err(|error| EngineError::InvalidStudyParameter {
            parameter: error.parameter.to_owned(),
            detail: error.detail,
        })
}

/// Check [`EngineConfig::rate_factors`] against the compiled model: empty,
/// or one finite positive factor per transition, other than 1 only on a
/// constant-rate exponential law, and drawn stochastic dates.
pub(super) fn validate_rate_factors(
    model: &CompiledModel,
    config: &EngineConfig,
) -> Result<(), EngineError> {
    let factors = &config.rate_factors;
    if factors.is_empty() {
        return Ok(());
    }
    if config.stochastic_dates != StochasticDates::Drawn {
        return Err(EngineError::InvalidStudyParameter {
            parameter: "rate_factors".to_owned(),
            detail: "rate factors bias drawn dates; they are refused with deferred \
                     stochastic dates"
                .to_owned(),
        });
    }
    if factors.len() != model.transitions.len() {
        return Err(EngineError::RateFactorCount {
            expected: model.transitions.len(),
            found: factors.len(),
        });
    }
    for (transition, &factor) in model.transitions.iter().zip(factors) {
        let reason = if !factor.is_finite() || factor <= 0.0 {
            Some("a rate factor must be finite and positive")
        } else if factor != 1.0 && !matches!(transition.distrib, CLaw::Exp(_)) {
            Some("only a constant-rate exponential law can be biased; any other law takes 1")
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(EngineError::InvalidRateFactor {
                transition: transition.name.clone(),
                factor,
                reason: reason.to_owned(),
            });
        }
    }
    Ok(())
}

/// Whether a compiled law is stochastic in the sense of
/// [`StochasticDates`]: its firing date is drawn in drawn mode, deferred
/// in deferred mode.
pub(super) fn is_stochastic(distrib: &CLaw) -> bool {
    matches!(fireable_kind(distrib), FireableKind::Stochastic)
}

/// Classify a compiled occurrence law for interactive inspection
/// ([`Engine::fireable`]).
pub(super) fn fireable_kind(distrib: &CLaw) -> FireableKind {
    match distrib {
        CLaw::Delay(_) => FireableKind::Delay,
        CLaw::Inst(_) => FireableKind::Inst,
        CLaw::Watched { .. } => FireableKind::Watched,
        CLaw::Exp(_)
        | CLaw::ExpVar { .. }
        | CLaw::Weibull(..)
        | CLaw::Lognormal(..)
        | CLaw::Gamma(..)
        | CLaw::Uniform(..)
        | CLaw::Empirical(_) => FireableKind::Stochastic,
    }
}

/// Inverse-CDF sampling from a validated empirical table: `u` below the
/// first cumulative probability maps to the first time (probability
/// mass); between points the CDF is linearly interpolated.
fn sample_empirical(points: &[(f64, f64)], u: f64) -> f64 {
    let (first_t, first_c) = points[0];
    if u <= first_c {
        return first_t;
    }
    for window in points.windows(2) {
        let (t0, c0) = window[0];
        let (t1, c1) = window[1];
        if u <= c1 {
            if c1 == c0 {
                return t1;
            }
            return t0 + (t1 - t0) * (u - c0) / (c1 - c0);
        }
    }
    // u ≤ 1 and the validated table ends at cumulative 1.
    points[points.len() - 1].0
}

impl<'m> Engine<'m> {
    /// `schedule_deterministic` + `drop_disabled`: (re)schedule fireable transitions and drop
    /// stale ones. Deterministic full scan (fine at fixture model
    /// sizes; the scan is index-ordered so the schedule is
    /// reproducible). Watched transitions are never date-scheduled.
    /// Evaluate a state-dependent rate λ(x) on the current state and
    /// reject non-finite or negative values with a typed error.
    pub(super) fn eval_rate(&self, trans_idx: TransIdx, rate: &CExpr) -> Result<f64, EngineError> {
        let lambda = eval_f64(self.model, &self.vars, &self.states, self.time, rate)?;
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(EngineError::TypeError {
                time: self.time,
                detail: format!(
                    "state-dependent rate of `{}` evaluated to {lambda} \
                     (must be finite and >= 0)",
                    self.model.transitions[trans_idx].name
                ),
            });
        }
        Ok(lambda)
    }

    /// The full walk: refresh every transition. Kept for the wholesale
    /// changes (the initialization axiom, an interactive attribute write),
    /// where nothing narrower is known; it leaves no refresh queued.
    pub(super) fn refresh_schedule(&mut self) -> Result<(), EngineError> {
        for trans_idx in 0..self.model.transitions.len() {
            self.refresh_transition(trans_idx)?;
        }
        self.clear_refresh_queue();
        Ok(())
    }

    /// The refresh after a discrete step: only the transitions the step's
    /// changes reached ([`crate::compile::RefreshIndex`]), plus those refreshed
    /// after every step, in ascending index order.
    ///
    /// A transition left out is at the fixed point of its refresh, so
    /// refreshing it would change nothing; ascending order keeps the
    /// random draws in the order of the full walk. On an error, the
    /// transition that failed and those after it stay queued, so a caller
    /// that survives the error loses no pending refresh.
    pub(super) fn refresh_affected(&mut self) -> Result<(), EngineError> {
        let always = &self.model.refresh_index.always;
        let mut queue = std::mem::take(&mut self.refresh_queue);
        queue.sort_unstable();
        // Merge the two ascending lists; an index in both is refreshed
        // once.
        let (mut next_queued, mut next_always) = (0, 0);
        loop {
            let queued = queue.get(next_queued).copied();
            let every_step = always.get(next_always).copied();
            let trans_idx = match (queued, every_step) {
                (None, None) => break,
                (Some(q), Some(a)) if q == a => {
                    next_queued += 1;
                    next_always += 1;
                    q
                }
                (Some(q), Some(a)) if q < a => {
                    next_queued += 1;
                    q
                }
                (Some(q), None) => {
                    next_queued += 1;
                    q
                }
                (_, Some(a)) => {
                    next_always += 1;
                    a
                }
            };
            if let Err(error) = self.refresh_transition(trans_idx) {
                // Keep what is left queued: the failed transition, the
                // rest of the queue, and the `always` ones not reached
                // (re-merged at the next refresh anyway).
                self.refresh_dirty[trans_idx] = true;
                self.refresh_queue.push(trans_idx);
                self.refresh_queue.extend_from_slice(&queue[next_queued..]);
                self.refresh_queue.sort_unstable();
                self.refresh_queue.dedup();
                return Err(error);
            }
            self.refresh_dirty[trans_idx] = false;
        }
        // A refresh writes no attribute and no state, so nothing was
        // marked while the queue was out.
        debug_assert!(self.refresh_queue.is_empty());
        queue.clear();
        self.refresh_queue = queue;
        Ok(())
    }

    /// `schedule_deterministic` + `drop_disabled` for one transition:
    /// arm it when it sits in its source with its guard holding, drop or
    /// pause it when either stops holding, reschedule a modifiable rate.
    /// [`Engine::refresh_affected`] runs it after a discrete step over the
    /// transitions the step's changes reached; a located crossing runs it
    /// over the located observers, so no other guard is read at that
    /// instant.
    pub(super) fn refresh_transition(&mut self, trans_idx: TransIdx) -> Result<(), EngineError> {
        let transition = &self.model.transitions[trans_idx];
        if matches!(transition.distrib, CLaw::Watched { .. }) {
            return Ok(());
        }
        self.schedule_refreshes += 1;
        let in_source = self.states[transition.automaton] == transition.source;
        if !in_source {
            // Leaving the source cancels any paused countdown and
            // discards the banked hazard.
            self.frozen[trans_idx] = None;
            self.hazards[trans_idx] = None;
        }
        let guard_ok = match &transition.guard {
            None => true,
            Some(guard) => eval_bool(self.model, &self.vars, &self.states, self.time, guard)?,
        };
        if self.config.stochastic_dates != StochasticDates::Drawn
            && is_stochastic(&transition.distrib)
        {
            self.refresh_deferred(trans_idx, in_source, guard_ok)?;
            self.refresh_operator_date(trans_idx, in_source, guard_ok);
            return Ok(());
        }
        self.refresh_operator_date(trans_idx, in_source, guard_ok);
        self.refresh_drawn_clock(trans_idx, in_source, guard_ok);
        match self.pending[trans_idx] {
            Some(_) if !in_source => {
                self.pending[trans_idx] = None;
                if self.config.journal {
                    self.journal.push(JournalRecord::TransitionDropped {
                        time: self.time,
                        transition: transition.name.clone(),
                        reason: DropReason::SourceLeft,
                    });
                }
            }
            Some(date)
                if !guard_ok
                    && transition.on_interruption != raichu_model::InterruptionPolicy::Continue =>
            {
                // `drop_disabled`: reset cancels the occurrence duration
                // (interruptible transition); resume
                // pauses it (RAICHU extension); continue never
                // reaches this arm.
                let reason = match transition.on_interruption {
                    raichu_model::InterruptionPolicy::Resume => {
                        if let Some(hazard) = self.hazards[trans_idx].as_mut() {
                            // Pause the hazard clock: bank what has
                            // accrued (continuous hazards are already
                            // committed by the integrator); the
                            // re-arm recomputes λ at resumption.
                            if !matches!(
                                transition.distrib,
                                CLaw::ExpVar {
                                    continuous: true,
                                    ..
                                }
                            ) {
                                hazard.accumulated += hazard.rate * (self.time - hazard.since);
                            }
                            hazard.since = self.time;
                        } else {
                            self.frozen[trans_idx] = Some(date - self.time);
                        }
                        DropReason::GuardPaused
                    }
                    _ => {
                        self.hazards[trans_idx] = None;
                        DropReason::GuardFalse
                    }
                };
                self.pending[trans_idx] = None;
                if self.config.journal {
                    self.journal.push(JournalRecord::TransitionDropped {
                        time: self.time,
                        transition: transition.name.clone(),
                        reason,
                    });
                }
            }
            #[allow(clippy::float_cmp)] // λ is re-evaluated exactly
            Some(previous) if in_source => {
                // `reschedule_modifiable`: a pending state-dependent rate whose
                // inputs changed at this discrete step is
                // rescheduled against the same `Exp(1)` threshold
                // (reached with the guard still true, or with the
                // `continue` policy riding through a false guard).
                // Continuously-varying rates need no rescheduling:
                // their hazard is integrated by `integrate_continuous` directly.
                let CLaw::ExpVar {
                    rate,
                    continuous: false,
                } = &self.model.transitions[trans_idx].distrib
                else {
                    return Ok(());
                };
                let lambda = self.eval_rate(trans_idx, rate)?;
                let Some(hazard) = self.hazards[trans_idx].as_mut() else {
                    return Ok(());
                };
                if lambda != hazard.rate {
                    hazard.accumulated += hazard.rate * (self.time - hazard.since);
                    hazard.since = self.time;
                    hazard.rate = lambda;
                    let firing_at = if lambda > 0.0 {
                        self.time + (hazard.threshold - hazard.accumulated) / lambda
                    } else {
                        f64::INFINITY
                    };
                    self.arm(trans_idx, firing_at);
                    if self.config.journal && firing_at != previous {
                        self.journal.push(JournalRecord::TransitionRescheduled {
                            time: self.time,
                            transition: self.model.transitions[trans_idx].name.clone(),
                            firing_at,
                        });
                    }
                }
            }
            None if in_source && guard_ok => {
                if self.clocks[trans_idx].is_some_and(|clock| clock.redraw) {
                    self.redraw_clock(trans_idx, false)?;
                    return Ok(());
                }
                // `schedule_stochastic` for a state-dependent rate: draw the
                // `Exp(1)` threshold (fresh arming) or keep the
                // banked hazard (resume re-arm), then schedule
                // against the current λ: `+∞` while λ = 0 or while
                // the hazard is integrated continuously (`integrate_continuous`
                // locates the firing like a boundary crossing).
                if let CLaw::ExpVar { rate, continuous } =
                    &self.model.transitions[trans_idx].distrib
                {
                    let lambda = self.eval_rate(trans_idx, rate)?;
                    let mut hazard = match self.hazards[trans_idx] {
                        Some(banked) => banked,
                        None => Hazard {
                            threshold: rand_distr::Exp1.sample(&mut self.rng),
                            accumulated: 0.0,
                            rate: lambda,
                            since: self.time,
                        },
                    };
                    hazard.rate = lambda;
                    hazard.since = self.time;
                    let firing_at = if *continuous || lambda <= 0.0 {
                        f64::INFINITY
                    } else {
                        self.time + (hazard.threshold - hazard.accumulated) / lambda
                    };
                    self.hazards[trans_idx] = Some(hazard);
                    self.arm(trans_idx, firing_at);
                    if self.config.journal {
                        self.journal.push(JournalRecord::TransitionScheduled {
                            time: self.time,
                            transition: self.model.transitions[trans_idx].name.clone(),
                            firing_at,
                        });
                    }
                    return Ok(());
                }
                // A paused countdown resumes where it stopped.
                if let Some(remaining) = self.frozen[trans_idx].take() {
                    let firing_at = self.time + remaining;
                    self.arm(trans_idx, firing_at);
                    if self.config.journal {
                        self.journal.push(JournalRecord::TransitionScheduled {
                            time: self.time,
                            transition: self.model.transitions[trans_idx].name.clone(),
                            firing_at,
                        });
                    }
                    return Ok(());
                }
                // `schedule_stochastic`: stochastic firing dates are sampled at
                // source-state entry. Draws happen here, in
                // transition-index order: replay is bit-identical
                // for a fixed (seed, stream).
                let time_now = self.time;
                let bad_law = move |detail: String| EngineError::TypeError {
                    time: time_now,
                    detail,
                };
                let firing_at = match &self.model.transitions[trans_idx].distrib {
                    CLaw::Delay(delay) => self.time + delay,
                    CLaw::Inst(_) => self.time,
                    CLaw::Watched { .. } => return Ok(()),
                    // Armed by the dedicated block above.
                    CLaw::ExpVar { .. } => return Ok(()),
                    CLaw::Exp(rate) => {
                        // The biased rate `f·λ` (`EngineConfig::rate_factors`):
                        // `Exp` samples `Exp1 / rate`, so the draw consumes
                        // the nominal random numbers, and `f = 1` is exact.
                        let rate = rate
                            * self
                                .config
                                .rate_factors
                                .get(trans_idx)
                                .copied()
                                .unwrap_or(1.0);
                        let distribution = rand_distr::Exp::new(rate)
                            .map_err(|e| bad_law(format!("exp({rate}): {e}")))?;
                        self.time + distribution.sample(&mut self.rng)
                    }
                    CLaw::Weibull(shape, scale) => {
                        let distribution = rand_distr::Weibull::new(*scale, *shape)
                            .map_err(|e| bad_law(format!("weibull: {e}")))?;
                        self.time + distribution.sample(&mut self.rng)
                    }
                    CLaw::Lognormal(mu, sigma) => {
                        let distribution = rand_distr::LogNormal::new(*mu, *sigma)
                            .map_err(|e| bad_law(format!("lognormal: {e}")))?;
                        self.time + distribution.sample(&mut self.rng)
                    }
                    CLaw::Gamma(shape, scale) => {
                        let distribution = rand_distr::Gamma::new(*shape, *scale)
                            .map_err(|e| bad_law(format!("gamma: {e}")))?;
                        self.time + distribution.sample(&mut self.rng)
                    }
                    CLaw::Uniform(low, high) => {
                        let distribution = rand_distr::Uniform::new(*low, *high)
                            .map_err(|e| bad_law(format!("uniform: {e}")))?;
                        self.time + distribution.sample(&mut self.rng)
                    }
                    CLaw::Empirical(points) => {
                        let u: f64 = rand::Rng::random(&mut self.rng);
                        self.time + sample_empirical(points, u)
                    }
                };
                self.arm(trans_idx, firing_at);
                if self.config.journal {
                    self.journal.push(JournalRecord::TransitionScheduled {
                        time: self.time,
                        transition: self.model.transitions[trans_idx].name.clone(),
                        firing_at,
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod refresh_cost_tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use crate::{CompiledModel, Engine, EngineConfig};
    use raichu_expr::{Assignment, AttrRef, BoolOp, Expr, StateRef, Value};
    use raichu_model::{
        AttrKind, Attribute, Automaton, Component, Distrib, Model, SensitiveFunction, Transition,
    };

    fn exp(name: &str, source: &str, target: &str, rate: f64) -> Transition {
        Transition {
            name: name.into(),
            source: source.into(),
            guard: None,
            targets: vec![target.into()],
            on_interruption: Default::default(),
            monitored: false,
            monitored_states: None,
            cycle_group: None,
            cycle_cancellation: Default::default(),
            kind: None,
            effects: vec![],
            distrib: Distrib::Exp {
                rate: Some(rate),
                rate_expr: None,
            },
        }
    }

    /// `n` independent repairable pumps: nothing any transition reads is
    /// written by another pump.
    fn pumps(n: usize) -> Model {
        let components = (0..n)
            .map(|index| {
                let name = format!("P{index}");
                Component {
                    name: name.clone(),
                    attributes: vec![Attribute {
                        name: "ok".into(),
                        kind: AttrKind::Bool,
                        init: Value::Bool(true),
                    }],
                    ports: vec![],
                    interfaces: vec![],
                    automata: vec![Automaton {
                        name: "fm".into(),
                        states: vec!["ok".into(), "ko".into()],
                        init: "ok".into(),
                        transitions: vec![
                            exp("fail", "ok", "ko", 0.01 + 0.001 * index as f64),
                            exp("repair", "ko", "ok", 0.1),
                        ],
                    }],
                    allocations: vec![],
                    equations: vec![],
                    sensitive_functions: vec![SensitiveFunction {
                        name: "apply".into(),
                        effects: vec![Assignment {
                            target: AttrRef {
                                component: name.clone(),
                                attribute: "ok".into(),
                            },
                            value: Expr::Bool {
                                bool_op: BoolOp::Not,
                                args: vec![Expr::StateActive {
                                    state: StateRef {
                                        component: name.clone(),
                                        automaton: "fm".into(),
                                        state: "ko".into(),
                                    },
                                }],
                            },
                        }],
                    }],
                }
            })
            .collect();
        Model {
            programs: vec![],
            name: "refresh_cost".into(),
            components,
            connections: vec![],
            indicators: vec![],
            targets: vec![],
            evaluation_order: None,
            unbounded_rate: None,
            fmu_units: vec![],
            interface_connections: vec![],
        }
    }

    /// **The cost claim of the refresh index.** The initialization axiom
    /// refreshes every transition once; each firing then refreshes the
    /// two transitions of the automaton that fired, whatever the model's
    /// size. The full walk refreshed all `2n` after every firing.
    #[test]
    fn a_firing_refreshes_the_transitions_that_read_what_it_changed() {
        for n in [10, 100] {
            let model = pumps(n);
            let compiled = CompiledModel::compile(&model).expect("the model compiles");
            let config = EngineConfig {
                t_max: 200.0,
                seed: 7,
                ..EngineConfig::default()
            };
            let mut engine = Engine::new(&compiled, config).expect("the engine builds");
            let mut firings = 0u64;
            while engine.step().expect("the step completes").is_some() {
                firings += 1;
            }
            assert!(firings > n as u64, "the trajectory exercises the pumps");
            let transitions = 2 * n as u64;
            assert_eq!(
                engine.schedule_refreshes,
                transitions + 2 * firings,
                "{n} pumps, {firings} firings"
            );
        }
    }
}
