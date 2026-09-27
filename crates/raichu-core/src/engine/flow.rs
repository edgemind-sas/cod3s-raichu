//! Flow resolution: the explicit sweep (equations and distribution
//! operators), its active-set termination test, and the resolution loop
//! that settles the continuous flows before and during integration.

use super::*;

/// Reusable scratch of the explicit sweep: the demand and allocation
/// vectors of the conservative distribution operators, and their capping
/// marks.
///
/// The sweep runs on every solver stage, so the buffers are owned by the
/// caller and reused: after the first pass the operator allocates nothing.
/// This is scratch, not trajectory state, so it stays out of the snapshot.
#[derive(Debug, Clone, Default)]
pub(super) struct FlowScratch {
    demands: Vec<f64>,
    allocated: Vec<f64>,
    capped: Vec<bool>,
    /// When present, every distribution operator run by the sweep appends
    /// its edge classes here, in sweep order: this is how the active set
    /// is read off **the operator's own capping outcome** instead of being
    /// recomputed by a second, parallel rule that could disagree with it.
    ///
    /// Set only by the boundary resolution. The per-stage path leaves it
    /// `None` and pays nothing.
    classes: Option<Vec<EdgeClass>>,
}

/// Read an attribute as a number, refusing a boolean where a quantity is
/// expected (a distribution operator moves quantities, not flags).
fn quantity(
    model: &CompiledModel,
    vars: &[Value],
    time: f64,
    var: VarIdx,
) -> Result<f64, EngineError> {
    match vars[var] {
        Value::Float(value) => Ok(value),
        Value::Int(value) => Ok(value as f64),
        Value::Bool(_) => Err(EngineError::TypeError {
            time,
            detail: format!(
                "`{}` carries a boolean where a distributed quantity is expected",
                model.var_names[var]
            ),
        }),
    }
}

/// Guard one input of a distribution operator: a non-finite quantity is a
/// loud failure, a negative one is not a quantity and distributes nothing.
///
/// The clamp is deliberate and narrow. A level or a rate can pass through
/// zero during an integration segment and land a few ulps below it; that
/// is rounding, not a negative demand, and aborting a run on it would make
/// every conservative flow fragile at exactly the operating point where
/// it matters. A NaN or an infinity, on the other hand, means the model
/// itself produced no number, and is reported.
fn flow_input(value: f64, time: f64, operator: &str, role: &str) -> Result<f64, EngineError> {
    if !value.is_finite() {
        return Err(EngineError::TypeError {
            time,
            detail: format!("the {role} of distribution operator `{operator}` is {value}"),
        });
    }
    Ok(value.max(0.0))
}

/// The mutable side-channels of one explicit sweep, carried together
/// because every caller holds all three and none of them belongs to the
/// state the sweep computes: what the pass **counts**
/// ([`WorkCounters`]), what it **scribbles on** ([`FlowScratch`]), and
/// what it **reports as moved** ([`ChangeLog`]).
pub(super) struct PassContext<'a> {
    /// Counted work of the pass.
    pub(super) work: &'a mut WorkCounters,
    /// Reused buffers of the distribution operators.
    pub(super) scratch: &'a mut FlowScratch,
    /// Change detection of the pass's writes.
    pub(super) changed: &'a mut ChangeLog,
}

/// Run one conservative distribution operator: read the available
/// quantity and the per-connection demands, split them under the compiled
/// policy, and write one quantity per outgoing connection.
fn run_allocation(
    model: &CompiledModel,
    vars: &mut [Value],
    states: &[StateIdx],
    time: f64,
    allocation: &CAllocation,
    ctx: &mut PassContext<'_>,
) -> Result<(), EngineError> {
    let PassContext {
        work,
        scratch,
        changed,
    } = ctx;
    let raw = eval_f64(model, vars, states, time, &allocation.available)?;
    let available = flow_input(raw, time, &allocation.name, "available quantity")?;
    scratch.demands.clear();
    for &var in &allocation.demands {
        let demand = quantity(model, vars, time, var)?;
        scratch
            .demands
            .push(flow_input(demand, time, &allocation.name, "demand")?);
    }
    scratch.allocated.clear();
    scratch.allocated.resize(scratch.demands.len(), 0.0);
    work.allocation_capping_passes += allocate(
        &allocation.policy,
        available,
        &scratch.demands,
        &mut scratch.allocated,
        &mut scratch.capped,
    );
    if scratch.classes.is_some() {
        // Split the borrow: `classify` reads the three buffers the
        // operator just wrote and appends to the fourth.
        let FlowScratch {
            demands,
            allocated,
            capped,
            classes,
        } = scratch;
        if let Some(sink) = classes.as_mut() {
            classify(&allocation.policy, demands, allocated, capped, sink);
        }
    }
    for (&target, &value) in allocation.allocated.iter().zip(&scratch.allocated) {
        changed.write(vars, target, Value::Float(value));
    }
    Ok(())
}

/// Read one operator's inputs on the current state: the available
/// quantity (returned) and one demand per edge (into `demands`), both
/// through the same guards the operator itself applies, so a margin never
/// sees a quantity the operator would have refused or clamped.
pub(super) fn read_flow_inputs(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    allocation: &CAllocation,
    demands: &mut Vec<f64>,
) -> Result<f64, EngineError> {
    let raw = eval_f64(model, vars, states, time, &allocation.available)?;
    let available = flow_input(raw, time, &allocation.name, "available quantity")?;
    demands.clear();
    for &var in &allocation.demands {
        let demand = quantity(model, vars, time, var)?;
        demands.push(flow_input(demand, time, &allocation.name, "demand")?);
    }
    Ok(available)
}

/// Run the explicit sweep (equations and distribution operators, in table
/// order) into `vars`, counting the pass into `work` (see
/// [`WorkCounters`]) and reporting the targets that moved into `changed`
/// (see [`ChangeLog`]).
pub(super) fn recompute_explicit(
    model: &CompiledModel,
    vars: &mut [Value],
    states: &[StateIdx],
    time: f64,
    ctx: &mut PassContext<'_>,
) -> Result<(), EngineError> {
    ctx.work.explicit_evaluations += 1;
    for step in &model.explicit {
        match step {
            CStep::Equation { target, expr } => {
                let value = eval_f64(model, vars, states, time, expr)?;
                ctx.changed.write(vars, *target, Value::Float(value));
            }
            CStep::Allocate(allocation) => {
                run_allocation(model, vars, states, time, allocation, ctx)?;
            }
        }
    }
    Ok(())
}

/// Record which branch of every comparison the sweep resolves, into a
/// vector whose *equality* between two sweeps is half the active-set
/// termination test (the other half being the operators' capping
/// outcome).
///
/// Minima and maxima contribute the index of the argument they select,
/// conditionals the branch they take. Both are finite, categorical
/// answers: settling them to exact equality turns most of a resolution
/// from an asymptotic question into a combinatorial one, which is why
/// they are tested before the flows are tested to a tolerance.
///
/// Only the **taken** branch of a conditional is walked. The other one is
/// not part of the answer, and evaluating it could fail on a state the
/// model never intended it to see.
fn record_branches(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
    into: &mut Vec<u32>,
) -> Result<(), EngineError> {
    match expr {
        CExpr::Min { args } | CExpr::Max { args } => {
            let wants_min = matches!(expr, CExpr::Min { .. });
            let mut best: Option<(usize, f64)> = None;
            for (index, arg) in args.iter().enumerate() {
                let value = eval_f64(model, vars, states, time, arg)?;
                let better = match best {
                    None => true,
                    Some((_, incumbent)) => {
                        if wants_min {
                            value < incumbent
                        } else {
                            value > incumbent
                        }
                    }
                };
                if better {
                    best = Some((index, value));
                }
            }
            // An empty min/max selects nothing; `u32::MAX` is that
            // answer, and it is as stable as any other.
            into.push(best.map_or(u32::MAX, |(index, _)| index as u32));
            for arg in args {
                record_branches(model, vars, states, time, arg, into)?;
            }
        }
        CExpr::If {
            cond,
            then,
            otherwise,
        } => {
            let taken = eval_bool(model, vars, states, time, cond)?;
            into.push(u32::from(taken));
            record_branches(model, vars, states, time, cond, into)?;
            record_branches(
                model,
                vars,
                states,
                time,
                if taken { then } else { otherwise },
                into,
            )?;
        }
        CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
            record_branches(model, vars, states, time, lhs, into)?;
            record_branches(model, vars, states, time, rhs, into)?;
        }
        CExpr::Bool { args, .. } | CExpr::Add { args } | CExpr::Mul { args } => {
            for arg in args {
                record_branches(model, vars, states, time, arg, into)?;
            }
        }
        CExpr::Sin(arg) | CExpr::Exp(arg) => {
            record_branches(model, vars, states, time, arg, into)?;
        }
        CExpr::Const(_)
        | CExpr::Var(_)
        | CExpr::StateActive { .. }
        | CExpr::PortAgg { .. }
        | CExpr::Time => {}
    }
    Ok(())
}

/// The **active set** of a resolved network: which edges are saturated
/// (read off each operator's capping outcome) and which branch of each
/// comparison the sweep took.
///
/// Two sweeps that produce equal signatures have settled the
/// combinatorial half of the resolution; what remains is numeric and is
/// settled against [`FLOW_TOLERANCE`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ActiveSet {
    edges: Vec<EdgeClass>,
    branches: Vec<u32>,
}

/// Sweeps the **numeric** level of one resolution may spend once its
/// active set has settled: the constant half of the two-level budget of
/// `Engine::resolve_flows`.
///
/// It is a constant because nothing in the compiled network sizes it: the
/// active set is finite and its budget counts edges, whereas the
/// quantities converge at a rate that is a property of the model's
/// arithmetic, not of its size.
///
/// **Why 64.** The undamped iteration reaches the per-edge tolerance in a
/// handful of sweeps on every settling network measured here (three on
/// the contested supply). The budget is not sized for those: it is sized
/// for the damped iteration that [`FLOW_RELAXATION`] substitutes after a
/// two-cycle, whose residual then falls by a factor `q` per sweep. From
/// an initial residual of order one, reaching [`FLOW_TOLERANCE`] costs
/// `ln(1e-9) / ln(q)` sweeps: 30 at `q = 1/2`, 58 at `q = 0.7`. Sixty-four
/// therefore covers every damped contraction up to about `q = 0.72` and
/// refuses the rest with a diagnostic rather than with silence.
///
/// This is the default of [`FlowConfig::sweep_budget`], which is where a
/// caller overrides it.
pub const FLOW_SWEEP_BUDGET: usize = 64;

/// Under-relaxation factor of the flow resolution: the weight given to
/// the sweep's raw answer when blending it with the iterate it started
/// from (`x ← (1 − w)·x + w·F(x)`).
///
/// **Why one half, and not the 0.9 of general practice.** The two figures
/// answer different failure modes. A factor near 0.9 damps a *diverging
/// monotone* iteration, whose linearised multiplier `μ` is above 1: there
/// the aim is to shave the overshoot while keeping most of the step, and
/// taking `w` far below 1 would only make a convergent case crawl. Here
/// the relaxation is engaged **only** in response to an observed
/// two-cycle, whose multiplier sits near −1. The damped multiplier is
/// `|1 − w(1 − μ)|`, which at `μ = −1` is `|1 − 2w|`: minimal, and zero,
/// at `w = 1/2`. The same point under `w = 0.9` leaves `0.8` per sweep,
/// about 95 sweeps to [`FLOW_TOLERANCE`], which no constant budget of the
/// size above could hold.
///
/// The relaxation is **not** applied from the first sweep. The cold-start
/// sequence of `Engine::resolve_flows` descends, and damping a
/// descending sequence buys nothing while costing every well-behaved
/// network a factor on its sweep count. It is latched on the first
/// two-cycle detection and never released within a resolution; it is a
/// local of the resolution, so nothing carries it across a segment.
///
/// This is the default of [`FlowConfig::relaxation`], which is where a
/// caller overrides it.
pub const FLOW_RELAXATION: f64 = 0.5;

/// Sweeps the **combinatorial** level of one resolution may spend, and
/// segment restarts one instant may absorb: the derived half of the
/// two-level budget.
///
/// Derived, not chosen. The descending cold-start sequence saturates at
/// least one more edge per round, so a search that is still changing the
/// saturation pattern after one round per compiled edge is no longer
/// searching. **Two** rounds per edge rather than one, because an edge
/// has three classes under a priority order
/// ([`EdgeClass::Unserved`] → [`EdgeClass::Partial`] → [`EdgeClass::Full`])
/// and therefore two class changes to spend along that descent. Branch
/// decisions (a limiting minimum, a conditional) enter the same pattern
/// and are counted alongside the edges, since a resolution can equally be
/// held up by a minimum that keeps swapping its limiting argument. Two
/// more rounds cover the sweep that produces the first candidate and the
/// sweep that confirms it.
///
/// The same figure bounds the segment restarts of
/// `Engine::advance_continuous` at one instant, for the same reason: at
/// most that many distinct class changes can be located there before the
/// boundary is chattering rather than moving.
///
/// This derivation is the source of the combinatorial budget unless
/// [`FlowConfig::active_set_budget`] overrides it.
#[must_use]
pub fn active_set_budget(model: &CompiledModel) -> usize {
    let mut budget = 2usize;
    for step in &model.explicit {
        match step {
            CStep::Equation { expr, .. } => budget += decision_sites(expr),
            CStep::Allocate(allocation) => {
                budget += 2 * allocation.allocated.len() + decision_sites(&allocation.available);
            }
        }
    }
    budget
}

/// Number of **branch decisions** an expression can contribute to the
/// active set: one per minimum, maximum and conditional, counted over
/// every branch (the sweep walks only the taken one, so this is an upper
/// bound, which is what a budget wants).
fn decision_sites(expr: &CExpr) -> usize {
    match expr {
        CExpr::Min { args } | CExpr::Max { args } => {
            1 + args.iter().map(decision_sites).sum::<usize>()
        }
        CExpr::If {
            cond,
            then,
            otherwise,
        } => 1 + decision_sites(cond) + decision_sites(then) + decision_sites(otherwise),
        CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
            decision_sites(lhs) + decision_sites(rhs)
        }
        CExpr::Bool { args, .. } | CExpr::Add { args } | CExpr::Mul { args } => {
            args.iter().map(decision_sites).sum()
        }
        CExpr::Sin(arg) | CExpr::Exp(arg) => decision_sites(arg),
        CExpr::Const(_)
        | CExpr::Var(_)
        | CExpr::StateActive { .. }
        | CExpr::PortAgg { .. }
        | CExpr::Time => 0,
    }
}

/// Whether one sweep moved every quantity by less than the per-edge flow
/// tolerance: the numeric half of the resolution's stopping test, applied
/// only once the combinatorial half has settled.
///
/// Mixed relative-and-absolute, so a network carrying large quantities is
/// held to the same meaning as one carrying small ones. A non-float
/// attribute compares exactly: nothing in the sweep writes one, so an
/// inequality there is a change, not a residual.
///
/// `tolerance` is [`FlowConfig::tolerance`], passed rather than read from
/// the constant so the stopping test and the margin dead band of the same
/// run are always the same number.
pub(super) fn flows_settled(before: &[Value], after: &[Value], tolerance: f64) -> bool {
    before
        .iter()
        .zip(after)
        .all(|(old, new)| value_settled(old, new, tolerance))
}

/// One attribute's half of [`flows_settled`]: also the test that decides
/// whether an edge "moved" in the sweep the diagnostic reports on, so
/// that what stops the resolution and what the diagnostic names are the
/// same measure.
pub(super) fn value_settled(old: &Value, new: &Value, tolerance: f64) -> bool {
    match (old, new) {
        (Value::Float(old), Value::Float(new)) => {
            (old - new).abs() <= tolerance * old.abs().max(new.abs()).max(1.0)
        }
        _ => old == new,
    }
}

impl<'m> Engine<'m> {
    /// Settle the conservative flow network at a boundary: **the active
    /// set first, to exact equality, then the flows, to the per-edge
    /// tolerance**.
    ///
    /// Which consumers are saturated and which branch of each comparison
    /// is taken is a finite, combinatorial question; how much each edge
    /// carries is not. Settling the finite half first turns most of the
    /// problem from asymptotic into combinatorial, and it is what lets
    /// the rest of the segment run with the answer frozen.
    ///
    /// The sequence descends. Its first term is the ordered sweep run
    /// from a **cold start**, in which every allocated quantity is zero
    /// and every consumer therefore sizes itself as though it held
    /// nothing: that pass over-estimates every delivery, which is what
    /// makes it a post-fixpoint and the sequence that follows
    /// non-increasing. Iterating up from zero would carry no such
    /// argument.
    ///
    /// The cold start is recomputed here rather than carried from the
    /// previous resolution. A warm start would be state living outside
    /// the attribute vector, and it would have to enter the snapshot for
    /// a replay to reproduce it.
    ///
    /// A model with no distribution operator has no active set to settle
    /// and takes none of this: it runs the single ordered pass it ran
    /// before the resolution existed, at the same cost, which is why its
    /// counted-work profile is unchanged.
    ///
    /// # Convergence policy
    ///
    /// The two levels carry **two budgets, both counted in sweeps**, and
    /// the whole policy is [`EngineConfig::flow`]: the figures below are
    /// its defaults, not the only values it can take. A sweep that
    /// changes the saturation pattern is charged to the combinatorial
    /// budget ([`FlowConfig::active_set_budget`], by default derived from
    /// the compiled edge and decision count by [`active_set_budget`]); a
    /// sweep that leaves the pattern alone and only moves quantities is
    /// charged to the numeric one ([`FlowConfig::sweep_budget`], by
    /// default [`FLOW_SWEEP_BUDGET`]). Every sweep is charged to exactly
    /// one of them, so the whole resolution is bounded by their sum, plus
    /// the single regrant described below. Neither is the engine's
    /// [`EngineConfig::max_fixpoint_iterations`], which counts worklist
    /// pops of the *discrete* fixpoint and would bound a different thing.
    ///
    /// **Under-relaxation** ([`FlowConfig::relaxation`], by default
    /// [`FLOW_RELAXATION`]) is latched the first time, and only the first
    /// time, the iterate returns to where it stood two sweeps earlier, in
    /// its saturation pattern or in its quantities. That observation says
    /// the descending argument behind the combinatorial budget has
    /// failed, so the combinatorial budget is granted **once more** at
    /// that point: what follows is a damped numeric iteration, not a
    /// descent, and charging its first class flips against a spent budget
    /// would refuse networks that damping is about to settle.
    ///
    /// Exhausting either budget raises [`EngineError::FlowNotConverged`],
    /// naming every edge that moved in the final sweep. The cause
    /// distinguishes a two-cycle that survived damping from a pattern
    /// that never settled and from quantities that never stopped
    /// creeping, but all three carry the same payload: a long cycle and a
    /// slow monotone sequence exhaust a budget without ever matching the
    /// two-cycle test, and they are the commonest stalls.
    ///
    /// Whatever the resolution moves is drained into the stale marks of
    /// the indexed watched set on every exit path, the failing ones
    /// included: a resolution that raised still left the attribute vector
    /// where its last sweep put it.
    pub(super) fn resolve_flows(&mut self) -> Result<(), EngineError> {
        let outcome = self.resolve_flows_inner();
        self.drain_changes();
        outcome
    }

    /// The resolution proper (see [`Engine::resolve_flows`], which wraps
    /// it to drain the change log).
    fn resolve_flows_inner(&mut self) -> Result<(), EngineError> {
        if self.model.flow_margins.is_empty() {
            let mut ctx = PassContext {
                work: &mut self.work,
                scratch: &mut self.flow_scratch,
                changed: &mut self.changed,
            };
            return recompute_explicit(
                self.model,
                &mut self.vars,
                &self.states,
                self.time,
                &mut ctx,
            );
        }
        let model = self.model;
        for margins in &model.flow_margins {
            if let CStep::Allocate(allocation) = &model.explicit[margins.step] {
                for &target in &allocation.allocated {
                    // The cold start is a *write* like any other: an edge
                    // whose settled quantity is zero would otherwise be
                    // reported as unchanged by the sweep that follows,
                    // and its move from the previous value would go
                    // unannounced.
                    self.changed
                        .write(&mut self.vars, target, Value::Float(0.0));
                }
            }
        }
        // History of the iteration, two sweeps deep: the two-cycle test
        // compares the fresh iterate with the one two steps back, which
        // is what an alternation between two allocations looks like.
        let mut previous: Option<ActiveSet> = None;
        let mut two_back: Option<ActiveSet> = None;
        let mut two_back_values: Option<Vec<Value>> = None;
        let mut before: Vec<Value> = Vec::with_capacity(self.vars.len());
        // Budgets, and the relaxation state. All three are locals: a
        // relaxation that survived a resolution would make the answer
        // depend on what the engine resolved before, and Monte-Carlo
        // results would stop being invariant in the thread count.
        let mut set_spent = 0usize;
        let mut flow_spent = 0usize;
        let mut relaxation = 1.0f64;
        let mut cycled = false;
        let mut sweeps = 0usize;
        // Read once: the policy is fixed for the engine's life, and
        // taking a copy keeps the loop from re-borrowing `self.config`
        // while it mutates `self.vars`.
        let policy = self.config.flow.clone();
        loop {
            before.clear();
            before.extend_from_slice(&self.vars);
            let current = self.sweep_once()?;
            if relaxation < 1.0 {
                self.relax_allocations(&before, relaxation);
            }
            sweeps += 1;
            let pattern_held = previous.as_ref() == Some(&current);
            if pattern_held && flows_settled(&before, &self.vars, policy.tolerance) {
                return Ok(());
            }
            // Two-cycle: the iterate is back where it stood two sweeps
            // ago, in its pattern or in its quantities, while differing
            // from the sweep just before it.
            let cycles = (two_back.as_ref() == Some(&current) && !pattern_held)
                || two_back_values
                    .as_ref()
                    .is_some_and(|old| flows_settled(old, &self.vars, policy.tolerance));
            // Latched on the *first* cycle only, and latched on `cycled`
            // rather than on the weight: a configuration that asks for no
            // damping at all (a weight of one) would otherwise re-grant
            // the combinatorial budget at every detection and never
            // exhaust it, turning a stall into a spin.
            if cycles && !cycled {
                cycled = true;
                relaxation = policy.relaxation;
                set_spent = 0;
            }
            if pattern_held {
                flow_spent += 1;
            } else {
                set_spent += 1;
            }
            if set_spent > self.active_set_budget || flow_spent > policy.sweep_budget {
                let cause = if cycled {
                    FlowStall::TwoCycle
                } else if pattern_held {
                    FlowStall::Quantities
                } else {
                    FlowStall::ActiveSet
                };
                return Err(self.flow_stalled(cause, sweeps, &before, previous.as_ref(), &current));
            }
            two_back_values = Some(before.clone());
            two_back = previous;
            previous = Some(current);
        }
    }

    /// Blend the allocated quantities of every distribution operator
    /// toward the answer the sweep just wrote, leaving the state at
    /// `(1 − w)·before + w·raw` (see [`FLOW_RELAXATION`]).
    ///
    /// Only the **allocated** quantities are blended, because they are
    /// the iteration's variables: every demand and every downstream
    /// equation is a function of them and is recomputed from the blended
    /// values by the next sweep. The state written by this sweep is
    /// therefore momentarily inconsistent with the blend, by exactly the
    /// amount the blend moved; the resolution only returns once that
    /// amount is below [`FLOW_TOLERANCE`], which is the level it promises
    /// anyway.
    fn relax_allocations(&mut self, before: &[Value], weight: f64) {
        let model = self.model;
        for margins in &model.flow_margins {
            let CStep::Allocate(allocation) = &model.explicit[margins.step] else {
                continue;
            };
            for &target in &allocation.allocated {
                let (Value::Float(old), Value::Float(raw)) = (before[target], self.vars[target])
                else {
                    continue;
                };
                // Through the change log like every other write of the
                // resolution: the blend moves the same targets the sweep
                // wrote, and the margins reading them must hear about it.
                self.changed.write(
                    &mut self.vars,
                    target,
                    Value::Float((1.0 - weight) * old + weight * raw),
                );
            }
        }
    }

    /// Build the non-convergence diagnostic: name the component and flow
    /// of every edge that moved in the final sweep, in the shape the
    /// instantaneous-loop and non-confluence diagnostics use.
    ///
    /// An edge "moved" if its saturation class changed or its allocated
    /// quantity moved by more than the flow tolerance: the same measure
    /// the stopping test applies, so the diagnostic can never name an
    /// empty set while the resolution claims something is still moving.
    /// It names one anyway when the movement was a branch decision (a
    /// conditional, a limiting minimum) rather than an edge, since a
    /// silent empty list would read as a defect.
    fn flow_stalled(
        &self,
        cause: FlowStall,
        sweeps: usize,
        before: &[Value],
        previous: Option<&ActiveSet>,
        current: &ActiveSet,
    ) -> EngineError {
        let mut moving: Vec<String> = Vec::new();
        let mut base = 0usize;
        for margins in &self.model.flow_margins {
            let CStep::Allocate(allocation) = &self.model.explicit[margins.step] else {
                continue;
            };
            for (edge, &target) in allocation.allocated.iter().enumerate() {
                let quantity_moved = !value_settled(
                    &before[target],
                    &self.vars[target],
                    self.config.flow.tolerance,
                );
                let class_moved = previous.is_some_and(|old| {
                    old.edges.get(base + edge) != current.edges.get(base + edge)
                });
                if quantity_moved || class_moved {
                    moving.push(self.edge_name(margins, edge));
                }
            }
            base += allocation.allocated.len();
        }
        let moving = if moving.is_empty() {
            "no edge (a conditional or a limiting minimum kept flipping)".to_owned()
        } else {
            moving.join(", ")
        };
        EngineError::FlowNotConverged {
            time: self.time,
            sweeps,
            cause,
            moving,
        }
    }

    /// Qualified `operator[consumer]` name of one edge: the operator is
    /// `component.allocation`, the consumer the allocated attribute, so
    /// the pair names the component and the flow.
    fn edge_name(&self, margins: &CFlowMargins, edge: usize) -> String {
        let consumer = margins
            .consumers
            .get(edge)
            .cloned()
            .unwrap_or_else(|| format!("edge #{edge}"));
        format!("{}[{}]", margins.name, consumer)
    }

    /// Name a set of **global** edge indices (the index space the frozen
    /// active set and [`Segment::Resolved`] use), for the chattering
    /// diagnostic.
    pub(super) fn edge_names(&self, indices: &[usize]) -> String {
        let mut names: Vec<String> = Vec::new();
        let mut base = 0usize;
        for margins in &self.model.flow_margins {
            let CStep::Allocate(allocation) = &self.model.explicit[margins.step] else {
                continue;
            };
            for edge in 0..allocation.allocated.len() {
                if indices.contains(&(base + edge)) {
                    names.push(self.edge_name(margins, edge));
                }
            }
            base += allocation.allocated.len();
        }
        if names.is_empty() {
            "no edge".to_owned()
        } else {
            names.join(", ")
        }
    }

    /// One sweep of the resolution: the ordered explicit pass, plus the
    /// active set it produced.
    fn sweep_once(&mut self) -> Result<ActiveSet, EngineError> {
        self.flow_scratch.classes = Some(Vec::new());
        let mut ctx = PassContext {
            work: &mut self.work,
            scratch: &mut self.flow_scratch,
            changed: &mut self.changed,
        };
        let outcome = recompute_explicit(
            self.model,
            &mut self.vars,
            &self.states,
            self.time,
            &mut ctx,
        );
        let edges = self.flow_scratch.classes.take().unwrap_or_default();
        outcome?;
        self.work.flow_sweeps += 1;
        let mut branches = Vec::new();
        for step in &self.model.explicit {
            match step {
                CStep::Equation { expr, .. } => record_branches(
                    self.model,
                    &self.vars,
                    &self.states,
                    self.time,
                    expr,
                    &mut branches,
                )?,
                CStep::Allocate(allocation) => record_branches(
                    self.model,
                    &self.vars,
                    &self.states,
                    self.time,
                    &allocation.available,
                    &mut branches,
                )?,
            }
        }
        Ok(ActiveSet { edges, branches })
    }

    /// The active set of the **current** state, read off a sweep run on a
    /// copy of the attribute vector so the committed state does not move.
    ///
    /// Derived, never carried: this is what keeps the frozen active set
    /// out of the snapshot and a replay exact.
    pub(super) fn frozen_classes(&mut self) -> Result<Vec<EdgeClass>, EngineError> {
        if self.model.flow_margins.is_empty() {
            return Ok(Vec::new());
        }
        let mut vars = self.vars.clone();
        let mut scratch = FlowScratch {
            classes: Some(Vec::new()),
            ..FlowScratch::default()
        };
        // The copy is thrown away with the pass: nothing to invalidate.
        let mut changed = ChangeLog::discarding(self.config.flow.tolerance);
        let mut ctx = PassContext {
            work: &mut self.work,
            scratch: &mut scratch,
            changed: &mut changed,
        };
        recompute_explicit(self.model, &mut vars, &self.states, self.time, &mut ctx)?;
        Ok(scratch.classes.unwrap_or_default())
    }
}
