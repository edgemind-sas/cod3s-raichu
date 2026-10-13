//! The continuous system handed to the ODE solver: right-hand sides,
//! the watched-boundary event functions, and the frozen flow picture a
//! segment integrates against.

use super::*;

/// One operator's active set, frozen for the duration of a segment.
///
/// The combinatorial search that produced `classes` ran once, at the
/// segment boundary. Inside the segment only the *quantities* move, which
/// is what keeps the solver's right-hand side a function of the state and
/// keeps an accepted and a rejected trial of the same step from doing
/// different amounts of work.
pub(super) struct FrozenFlow<'m> {
    /// The operator, borrowed from the compiled sweep.
    pub(super) allocation: &'m CAllocation,
    /// Where its margins are registered (naming, dependencies).
    pub(super) margins: &'m CFlowMargins,
    /// Saturation class per edge, in connection declaration order.
    pub(super) classes: Vec<EdgeClass>,
}

/// Adapter exposing the compiled continuous section to `raichu-numeric`.
/// Errors raised inside the solver callbacks are stashed and re-raised
/// after integration (the trait is infallible by design).
pub(super) struct ContinuousSystem<'m> {
    pub(super) model: &'m CompiledModel,
    pub(super) vars: Vec<Value>,
    pub(super) states: Vec<StateIdx>,
    /// Active watched transitions monitored this segment.
    pub(super) margins: Vec<TransIdx>,
    /// Active continuously-varying hazards monitored this segment:
    /// `(transition, remaining threshold E − H)`. Each occupies one
    /// auxiliary state slot after the ODE attributes, integrating
    /// `dH/dt = λ(x)`; the firing is the event `H − (E − H₀) = 0`,
    /// located exactly like a watched boundary crossing (`reschedule_modifiable`
    /// under continuous evolution).
    pub(super) hazards: Vec<(TransIdx, f64)>,
    /// The **frozen active set** of this segment: one entry per
    /// distribution operator, each carrying the saturation class of every
    /// edge as it was settled at the segment boundary. Their margins ride
    /// alongside the watched ones, so the instant the frozen pattern
    /// stops holding is *located*, not noticed at the next discrete date.
    pub(super) flows: Vec<FrozenFlow<'m>>,
    /// The threshold indicators whose flips this segment locates, as
    /// observations of the solver: an ordering on a number, which a
    /// continuum can cross. Indices into the model's indicators.
    pub(super) observed: Vec<usize>,
    /// The flips the solver located, `(date, indicator, verdict after)`,
    /// in time order.
    pub(super) flips: Vec<(f64, usize, bool)>,
    /// Dense hazard recording of [`Engine::probe_deferred`]: the
    /// cumulative hazard banked at segment start per slot of `hazards`,
    /// and the samples taken. Rides on one extra observed verdict that
    /// always reads `false` (so it never flips and never moves the
    /// solver), because an observation is evaluated at every accepted
    /// step end and at the interior scan points. `None` outside a probe.
    pub(super) trace: Option<(Vec<f64>, Vec<HazardSample>)>,
    pub(super) error: Option<EngineError>,
    /// Work done inside the solver callbacks, merged back into the
    /// engine's counters when the segment returns.
    pub(super) work: WorkCounters,
    /// Scratch of the distribution operators run by the explicit sweep
    /// inside those callbacks.
    pub(super) scratch: FlowScratch,
    /// Scratch holding one operator's demands while its active-set
    /// margins are evaluated (reused: the event callback runs on every
    /// interior scan point and every bisection step).
    pub(super) flow_demands: Vec<f64>,
    /// The run's flow tolerance ([`FlowConfig::tolerance`]): the dead
    /// band of every active-set margin evaluated here. Copied from the
    /// config at segment start, so the band this segment applies and the
    /// tolerance the resolution that opened it settled to are the same
    /// number.
    pub(super) flow_tolerance: f64,
    /// The run's sweep budget ([`FlowConfig::sweep_budget`]): how many
    /// passes a torn sweep may repeat inside one right-hand side.
    pub(super) sweep_budget: usize,
    /// The attribute vector before the last pass of a torn sweep, reused
    /// across evaluations.
    pub(super) settle_scratch: Vec<Value>,
}

impl ContinuousSystem<'_> {
    fn load(&mut self, t: f64, y: &[f64]) {
        for (slot, (var, _)) in self.model.ode.iter().enumerate() {
            self.vars[*var] = Value::Float(y[slot]);
        }
        if self.error.is_none() {
            // Nothing outside this callback survives the segment, so the
            // pass compares but records nothing.
            let mut changed = ChangeLog::discarding(self.flow_tolerance);
            let mut ctx = PassContext {
                work: &mut self.work,
                scratch: &mut self.scratch,
                changed: &mut changed,
            };
            // A torn ring settles inside the right-hand side, so that it is
            // a function of `(t, y)` and not of the evaluation history,
            // rejected trial steps included (`settle_explicit`).
            if let Err(error) = settle_explicit(
                self.model,
                &mut self.vars,
                &self.states,
                t,
                &mut ctx,
                self.sweep_budget,
                self.flow_tolerance,
                &mut self.settle_scratch,
            ) {
                self.error = Some(error);
            }
        }
    }
}

impl ContinuousSystem<'_> {
    /// The [`EngineError::UnboundedRate`] refusal a solver failure at
    /// `(t, y)` really is, if any.
    ///
    /// `evolC` refuses a rate that *reaches* the magnitude the model
    /// reserves for "unbounded". The same draw can arrive diluted: a
    /// volume holding a mixture serves each constituent at its share of
    /// the stand-in, so an unbounded draw on a 2 % constituent integrates
    /// at 2e28, under the magnitude and far beyond what any step can
    /// follow. The step controller then collapses, and its report
    /// ("too stiff") names a symptom rather than the model's statement.
    ///
    /// The test is the one the reservation implies: a physical rate does
    /// not depend on the value chosen for "unbounded". The stand-in is
    /// doubled wherever the model writes it, in an attribute or as a
    /// literal of the continuous section, the explicit sweep is settled
    /// again, and a rate that moves with it is set by the stand-in. One
    /// that does not (a stand-in a `min` takes away) is untouched, so a
    /// genuinely stiff model keeps the solver's error. Called only on a
    /// failed integration: a trajectory the solver could follow costs
    /// nothing and reads nothing new.
    pub(super) fn unbounded_dependence(&self, t: f64, y: &[f64]) -> Option<EngineError> {
        let unbounded = self.model.unbounded_rate?;
        let mut doubled_model = self.model.clone();
        let mut moved = false;
        for (_, expr) in &mut doubled_model.ode {
            moved |= double_stand_in(expr, unbounded);
        }
        for step in &mut doubled_model.explicit {
            match step {
                CStep::Equation { expr, .. } => moved |= double_stand_in(expr, unbounded),
                CStep::Allocate(allocation) => {
                    moved |= double_stand_in(&mut allocation.available, unbounded);
                }
                CStep::Block(block) => {
                    for constant in &mut block.constants {
                        moved |= double_stand_in(constant, unbounded);
                    }
                    for (_, coefficient) in block.coefficients.iter_mut().flatten() {
                        moved |= double_stand_in(coefficient, unbounded);
                    }
                }
            }
        }
        let mut doubled_vars = self.vars.clone();
        for value in &mut doubled_vars {
            if let Value::Float(v) = value {
                if v.abs() >= unbounded {
                    *v *= 2.0;
                    moved = true;
                }
            }
        }
        if !moved {
            return None;
        }
        let base = self.rates_at(self.model, self.vars.clone(), t, y)?;
        let doubled = self.rates_at(&doubled_model, doubled_vars, t, y)?;
        // What is refused is a stock drained without bound, so a draining
        // rate is named first: the stock it fills downstream moves by the
        // same magnitude, and naming it would point at the wrong volume.
        let (slot, rate) = base
            .iter()
            .zip(&doubled)
            .enumerate()
            .filter(|(_, (rate, again))| {
                rate.is_finite()
                    && (*again - *rate).abs() > 1e-6 * rate.abs().max(again.abs()).max(1.0)
            })
            .max_by(|(_, (a, _)), (_, (b, _))| {
                (**a < 0.0)
                    .cmp(&(**b < 0.0))
                    .then(a.abs().total_cmp(&b.abs()))
            })
            .map(|(slot, (rate, _))| (slot, *rate))?;
        Some(EngineError::UnboundedRate {
            time: t,
            variable: self.model.var_names[self.model.ode[slot].0].clone(),
            rate,
            unbounded,
        })
    }

    /// The integrated rates of `model` at `(t, y)`, the explicit sweep
    /// settled from `vars` exactly as the right-hand side settles it.
    /// `None` when an evaluation fails: the diagnosis then has nothing to
    /// say and the solver's error stands.
    fn rates_at(
        &self,
        model: &CompiledModel,
        mut vars: Vec<Value>,
        t: f64,
        y: &[f64],
    ) -> Option<Vec<f64>> {
        for (slot, (var, _)) in model.ode.iter().enumerate() {
            vars[*var] = Value::Float(y[slot]);
        }
        let (mut work, mut scratch) = (WorkCounters::default(), FlowScratch::default());
        let mut changed = ChangeLog::discarding(self.flow_tolerance);
        let mut ctx = PassContext {
            work: &mut work,
            scratch: &mut scratch,
            changed: &mut changed,
        };
        settle_explicit(
            model,
            &mut vars,
            &self.states,
            t,
            &mut ctx,
            self.sweep_budget,
            self.flow_tolerance,
            &mut Vec::new(),
        )
        .ok()?;
        model
            .ode
            .iter()
            .map(|(_, expr)| eval_f64(model, &vars, &self.states, t, expr).ok())
            .collect()
    }
}

/// Double, in place, every float literal of `expr` at or beyond the
/// declared `unbounded` magnitude; whether any was found.
fn double_stand_in(expr: &mut CExpr, unbounded: f64) -> bool {
    match expr {
        CExpr::Const(Value::Float(v)) if v.abs() >= unbounded => {
            *v *= 2.0;
            true
        }
        CExpr::Const(_)
        | CExpr::Var(_)
        | CExpr::StateActive { .. }
        | CExpr::PortAgg { .. }
        | CExpr::Time => false,
        CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
            let left = double_stand_in(lhs, unbounded);
            double_stand_in(rhs, unbounded) | left
        }
        CExpr::Bool { args, .. }
        | CExpr::Add { args }
        | CExpr::Mul { args }
        | CExpr::Min { args }
        | CExpr::Max { args } => args
            .iter_mut()
            .fold(false, |found, arg| double_stand_in(arg, unbounded) | found),
        CExpr::If {
            cond,
            then,
            otherwise,
        } => {
            let found = double_stand_in(cond, unbounded);
            let found = double_stand_in(then, unbounded) | found;
            double_stand_in(otherwise, unbounded) | found
        }
        CExpr::Sin(arg) | CExpr::Exp(arg) => double_stand_in(arg, unbounded),
    }
}

impl OdeSystem for ContinuousSystem<'_> {
    fn dim(&self) -> usize {
        self.model.ode.len() + self.hazards.len()
    }

    fn rhs(&mut self, t: f64, y: &[f64], dydt: &mut [f64]) {
        self.load(t, y);
        for (slot, (var, expr)) in self.model.ode.iter().enumerate() {
            match eval_f64(self.model, &self.vars, &self.states, t, expr) {
                Ok(value) => {
                    // `evolC` refuses a rate the model reserved for
                    // "unbounded", recorded like any callback error and
                    // integrated as zero until the segment returns it.
                    match self.model.unbounded_rate {
                        Some(unbounded) if value.abs() >= unbounded => {
                            self.error.get_or_insert(EngineError::UnboundedRate {
                                time: t,
                                variable: self.model.var_names[*var].clone(),
                                rate: value,
                                unbounded,
                            });
                            dydt[slot] = 0.0;
                        }
                        _ => dydt[slot] = value,
                    }
                }
                Err(error) => {
                    self.error.get_or_insert(error);
                    dydt[slot] = 0.0;
                }
            }
        }
        let ode_len = self.model.ode.len();
        for (slot, (trans_idx, _)) in self.hazards.iter().enumerate() {
            let CLaw::ExpVar { rate, .. } = &self.model.transitions[*trans_idx].distrib else {
                dydt[ode_len + slot] = 0.0;
                continue;
            };
            match eval_f64(self.model, &self.vars, &self.states, t, rate) {
                Ok(lambda) if lambda.is_finite() && lambda >= 0.0 => {
                    dydt[ode_len + slot] = lambda;
                }
                Ok(lambda) => {
                    self.error.get_or_insert(EngineError::TypeError {
                        time: t,
                        detail: format!(
                            "state-dependent rate of `{}` evaluated to {lambda} \
                             (must be finite and >= 0)",
                            self.model.transitions[*trans_idx].name
                        ),
                    });
                    dydt[ode_len + slot] = 0.0;
                }
                Err(error) => {
                    self.error.get_or_insert(error);
                    dydt[ode_len + slot] = 0.0;
                }
            }
        }
    }

    fn n_events(&self) -> usize {
        self.margins.len()
            + self.hazards.len()
            + self.flows.iter().map(|f| f.classes.len()).sum::<usize>()
    }

    fn n_observations(&self) -> usize {
        self.observed.len() + usize::from(self.trace.is_some())
    }

    fn observations(&mut self, t: f64, y: &[f64], out: &mut [bool]) {
        self.load(t, y);
        for (slot, &indicator) in self.observed.iter().enumerate() {
            out[slot] = matches!(
                indicator_value(
                    &self.model.indicators[indicator].target,
                    &self.vars,
                    &self.states
                ),
                Value::Bool(true)
            );
        }
        if let Some((bases, samples)) = self.trace.as_mut() {
            out[self.observed.len()] = false;
            // Interior bisection points of another verdict's flip come
            // after the scan point that bracketed them: keep the samples
            // strictly increasing in time.
            if samples.last().is_none_or(|last| t > last.time) {
                let ode_len = self.model.ode.len();
                samples.push(HazardSample {
                    time: t,
                    hazards: bases
                        .iter()
                        .enumerate()
                        .map(|(slot, base)| base + y[ode_len + slot].max(0.0))
                        .collect(),
                });
            }
        }
    }

    fn observed(&mut self, index: usize, t: f64, now: bool) {
        if let Some(&indicator) = self.observed.get(index) {
            self.flips.push((t, indicator, now));
        }
    }

    fn events(&mut self, t: f64, y: &[f64], out: &mut [f64]) {
        self.load(t, y);
        self.work.margin_evaluations += self.margins.len() as u64;
        for (slot, trans_idx) in self.margins.iter().enumerate() {
            let CLaw::Watched { margin } = &self.model.transitions[*trans_idx].distrib else {
                out[slot] = -1.0;
                continue;
            };
            match eval_f64(self.model, &self.vars, &self.states, t, margin) {
                Ok(value) => out[slot] = value,
                Err(error) => {
                    self.error.get_or_insert(error);
                    out[slot] = -1.0;
                }
            }
        }
        let (n_margins, ode_len) = (self.margins.len(), self.model.ode.len());
        for (slot, (_, remaining)) in self.hazards.iter().enumerate() {
            out[n_margins + slot] = y[ode_len + slot] - remaining;
        }
        // Active-set margins. They are watched guards like the ones
        // above: the segment ends where the frozen saturation pattern
        // stops holding, and the crossing instant is bisected to the same
        // event tolerance.
        let mut slot = n_margins + self.hazards.len();
        for flow in &self.flows {
            self.work.margin_evaluations += flow.classes.len() as u64;
            let inputs = read_flow_inputs(
                self.model,
                &self.vars,
                &self.states,
                t,
                flow.allocation,
                &mut self.flow_demands,
            );
            let available = match inputs {
                Ok(available) => available,
                Err(error) => {
                    self.error.get_or_insert(error);
                    for edge in 0..flow.classes.len() {
                        out[slot + edge] = -1.0;
                    }
                    slot += flow.classes.len();
                    continue;
                }
            };
            let band = flow_band(
                self.flow_demands
                    .iter()
                    .fold(available.abs(), |scale, d| scale.max(d.abs())),
                self.flow_tolerance,
            );
            for edge in 0..flow.classes.len() {
                out[slot + edge] = edge_margin(
                    &flow.allocation.policy,
                    available,
                    &self.flow_demands,
                    &flow.classes,
                    edge,
                    band,
                );
            }
            slot += flow.classes.len();
        }
    }
}
