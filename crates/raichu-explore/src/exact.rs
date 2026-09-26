//! Exact exploration of the sequence tree of a Markov model.
//!
//! # What is computed
//!
//! Between two jumps, a model of the exact domain sits in a discrete state
//! whose armed exponential transitions compete with constant rates
//! `λ_1..λ_n`; the next jump is transition `i` with probability
//! `λ_i / q`, after a sojourn `T ~ Exp(q)`, `q = Σ λ_i` (the embedded jump
//! chain). An instantaneous transition fires with zero sojourn, choosing
//! its destination with its declared branch probabilities. A sequence (an
//! ordered path from the initial state to the target) therefore has
//! probability
//!
//! ```text
//! P(path, target by t) = Π (branch probabilities) × P(T_1 + ... + T_k <= t)
//! ```
//!
//! where `T_i ~ Exp(q_i)` are the sojourns of the timed nodes along the
//! path, repeated rates included. The time factor is the absorption
//! probability of an acyclic phase chain, computed in nonnegative
//! arithmetic with an explicit error bound
//! ([`raichu_numeric::PhaseTypeAccumulator`]); a probability whose bound
//! exceeds the declared precision is flagged, never reported silently.
//!
//! # How the tree is explored
//!
//! Depth first, over the engine itself (the single source of semantics):
//! one [`Engine`] per worker, a [`raichu_core::Snapshot`] at each node,
//! restored before each child, and each child produced by firing the chosen
//! transition **at the current instant**, forced to the chosen destination
//! ([`Engine::fire_now`]). The clock never moves; the dates the engine
//! draws for newly armed transitions are ignored, and the snapshots carry
//! the RNG, so the exploration is deterministic.
//!
//! At a node, the armed set decides:
//!
//! 1. an armed **instantaneous** transition (instantaneous law or zero
//!    delay) fires first, lowest index first (the engine's own order),
//!    branching over its destinations of nonzero probability, with no
//!    sojourn;
//! 2. otherwise, if the target is reached, the node is a **retained
//!    sequence**. If another target was reached first, it is a leaf that
//!    contributes nothing: a Monte-Carlo trajectory stops there too;
//! 3. otherwise each armed exponential transition of **positive** rate is
//!    a branch (a zero-rate transition, a dormant spare, is not), and a
//!    node whose total exit rate is 0 is an **absorbing leaf**,
//!    contributing nothing;
//! 4. any other armed law stops the exploration with
//!    [`EngineError::LawOutsideExactDomain`], naming the transition and
//!    the sequence that armed it. A law that is never armed does not
//!    block.
//!
//! The walk is iterative, over an explicit heap-allocated frame stack, so
//! the depth of a sequence is bounded by memory, never by the thread's
//! stack. More than [`EngineConfig::max_fixpoint_iterations`] instantaneous
//! firings in a row along one sequence (a cycle of instantaneous or
//! zero-delay transitions, whose mass no probability cut-off ever
//! decreases) stop the exploration with [`EngineError::InstantaneousCycle`].
//!
//! Children are visited in transition index order, then destination
//! order. The minimal-probability, length and failure cut-offs are applied
//! to a child before it is fired, on the probability that its prefix is
//! completed by the horizon, which bounds every sequence extending it; the
//! branch cap is applied when a node is about to be expanded. Every pruned
//! mass is added to the upper bound and to its cut-off's tally.
//!
//! # Determinism and parallelism
//!
//! The root's children are split across workers, each subtree with its
//! own engine and its own share of the branch cap (equal shares, the
//! remainder to the first children in order), and the partial results are
//! reduced in child order. The explored set, the sequences and every
//! number are therefore bit-identical whatever the thread count.

use std::collections::{BTreeMap, VecDeque};

use raichu_core::compile::{CExpr, CLaw, CStep};
use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError, Snapshot};
use raichu_model::TransitionKind;
use raichu_numeric::{PhaseTypeAccumulator, PhaseTypeError};
use serde::{Deserialize, Serialize};

use crate::result::{
    relative_gap, Algorithm, CutoffTallies, Cutoffs, ExplorationResult, ExploredEvent,
    ExploredSequence, ExploredStep, Precision, DEFAULT_GAP_TOLERANCE, EXPLORATION_FORMAT,
    EXPLORATION_VERSION,
};

/// Settings of an exact exploration.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactSettings {
    /// Name of the model [`raichu_model::Target`] (feared event) explored
    /// to.
    pub target: String,
    /// Horizon `t`, in the model's time unit: a sequence counts when the
    /// target is reached by `t`. Finite and nonnegative.
    pub horizon: f64,
    /// The declared cut-offs, each optional.
    pub cutoffs: Cutoffs,
    /// Relative gap `(upper - lower) / upper` above which the result is
    /// flagged inconclusive, in `[0, 1]`. Default
    /// [`DEFAULT_GAP_TOLERANCE`].
    pub gap_tolerance: f64,
    /// Numerical precision of the sequence probabilities.
    pub precision: Precision,
    /// Worker threads (`None` = rayon default). The result does not
    /// depend on it.
    pub threads: Option<usize>,
}

impl ExactSettings {
    /// Settings for `target` at `horizon`, with no cut-off, the default
    /// gap tolerance and precision, and the default thread count.
    #[must_use]
    pub fn new(target: impl Into<String>, horizon: f64) -> Self {
        ExactSettings {
            target: target.into(),
            horizon,
            cutoffs: Cutoffs::default(),
            gap_tolerance: DEFAULT_GAP_TOLERANCE,
            precision: Precision::default(),
            threads: None,
        }
    }
}

/// A reason a model is outside the exact exploration domain, found
/// without exploring it ([`exact_domain_report`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum DomainViolation {
    /// The model integrates an ODE: its state evolves continuously
    /// between jumps.
    Ode {
        /// The integrated attribute, `component.attribute`.
        attribute: String,
    },
    /// A watched (boundary) transition sits in a state its automaton can
    /// reach from its initial state: its firing date is a boundary
    /// crossing, not an exponential draw.
    Watched {
        /// Qualified transition name.
        transition: String,
    },
    /// An expression reads the simulation time, which the exact
    /// exploration never advances.
    ReadsTime {
        /// Where: a transition guard, rate or effect, a sensitive
        /// function, or an explicit equation, with its qualified name.
        site: String,
    },
}

impl std::fmt::Display for DomainViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DomainViolation::Ode { attribute } => {
                write!(f, "`{attribute}` is integrated by an ODE")
            }
            DomainViolation::Watched { transition } => write!(
                f,
                "watched transition `{transition}` can be armed (a boundary crossing)"
            ),
            DomainViolation::ReadsTime { site } => {
                write!(f, "{site} reads the simulation time")
            }
        }
    }
}

/// The static half of the exact domain check: every reason, found
/// without exploring, that `model` is outside the exact domain (empty
/// when none is found).
///
/// Reported: an ODE, a watched transition whose source state is reachable
/// from its automaton's initial state (ignoring guards), and any
/// expression reading the simulation time. A model with an empty report
/// can still stop an exploration when a law outside the domain (delay,
/// Weibull, ...) becomes armed: that half is checked during the run, on
/// the sequences actually explored.
#[must_use]
pub fn exact_domain_report(model: &CompiledModel) -> Vec<DomainViolation> {
    let mut report = Vec::new();
    for (var, _) in &model.ode {
        report.push(DomainViolation::Ode {
            attribute: model.var_names[*var].clone(),
        });
    }
    for &idx in &model.watched {
        let transition = &model.transitions[idx];
        if reachable_states(model, transition.automaton)[transition.source] {
            report.push(DomainViolation::Watched {
                transition: transition.name.clone(),
            });
        }
    }
    for transition in &model.transitions {
        let mut reads = transition.guard.as_ref().is_some_and(reads_time)
            || transition.effects.iter().any(|(_, e)| reads_time(e));
        if let CLaw::ExpVar { rate, .. } = &transition.distrib {
            reads |= reads_time(rate);
        }
        if reads {
            report.push(DomainViolation::ReadsTime {
                site: format!("transition `{}`", transition.name),
            });
        }
    }
    for function in &model.functions {
        if function.effects.iter().any(|(_, e)| reads_time(e)) {
            report.push(DomainViolation::ReadsTime {
                site: format!("sensitive function `{}`", function.name),
            });
        }
    }
    for step in &model.explicit {
        match step {
            CStep::Equation { target, expr } if reads_time(expr) => {
                report.push(DomainViolation::ReadsTime {
                    site: format!("equation of `{}`", model.var_names[*target]),
                });
            }
            CStep::Allocate(allocation) if reads_time(&allocation.available) => {
                report.push(DomainViolation::ReadsTime {
                    site: format!("distribution operator `{}`", allocation.name),
                });
            }
            _ => {}
        }
    }
    report
}

/// States of automaton `aut` reachable from its initial state through its
/// transitions' declared targets, guards ignored.
fn reachable_states(model: &CompiledModel, aut: usize) -> Vec<bool> {
    let automaton = &model.automata[aut];
    let mut seen = vec![false; automaton.states.len()];
    let mut queue = VecDeque::from([automaton.init]);
    seen[automaton.init] = true;
    while let Some(state) = queue.pop_front() {
        for &idx in &automaton.transitions {
            let transition = &model.transitions[idx];
            if transition.source != state {
                continue;
            }
            for &target in &transition.targets {
                if !seen[target] {
                    seen[target] = true;
                    queue.push_back(target);
                }
            }
        }
    }
    seen
}

/// Whether a compiled expression reads the simulation time.
fn reads_time(expr: &CExpr) -> bool {
    match expr {
        CExpr::Time => true,
        CExpr::Const(_) | CExpr::Var(_) | CExpr::StateActive { .. } | CExpr::PortAgg { .. } => {
            false
        }
        CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
            reads_time(lhs) || reads_time(rhs)
        }
        CExpr::Bool { args, .. }
        | CExpr::Add { args }
        | CExpr::Mul { args }
        | CExpr::Min { args }
        | CExpr::Max { args } => args.iter().any(reads_time),
        CExpr::If {
            cond,
            then,
            otherwise,
        } => reads_time(cond) || reads_time(then) || reads_time(otherwise),
        CExpr::Sin(arg) | CExpr::Exp(arg) => reads_time(arg),
    }
}

/// Explore the sequence tree of `model` to the target `settings.target`
/// with the exact algorithm (see the module documentation).
///
/// # Errors
///
/// - [`EngineError::InvalidStudyParameter`], before anything runs, for a
///   setting outside its domain: an unknown target, a negative or
///   non-finite horizon, a minimal probability outside `(0, 1]`, a branch
///   cap of 0, a gap tolerance outside `[0, 1]`, a precision outside its
///   domain, 0 threads, or a failure cut-off on a model where no
///   transition is declared a `failure` (it would silently count
///   nothing).
/// - [`EngineError::OutsideExactDomain`], before anything runs, when
///   [`exact_domain_report`] is not empty.
/// - [`EngineError::LawOutsideExactDomain`] when a law outside the domain
///   becomes armed along an explored sequence (the first such sequence in
///   exploration order).
/// - [`EngineError::InstantaneousCycle`] when more than
///   [`EngineConfig::max_fixpoint_iterations`] instantaneous transitions
///   would fire in a row along an explored sequence.
/// - any engine error raised while firing a transition.
pub fn explore_exact(
    model: &CompiledModel,
    settings: &ExactSettings,
) -> Result<ExplorationResult, EngineError> {
    use rayon::prelude::*;

    validate(model, settings)?;
    let violations = exact_domain_report(model);
    if !violations.is_empty() {
        return Err(EngineError::OutsideExactDomain {
            reasons: violations.iter().map(ToString::to_string).collect(),
        });
    }
    let root_acc = PhaseTypeAccumulator::new(settings.horizon, (&settings.precision).into())
        .map_err(|e| invalid("precision", e.to_string()))?;
    let config = engine_config();

    let mut root = Walker::new(
        model,
        settings,
        Engine::new(model, config.clone())?,
        settings.cutoffs.max_branches,
    );
    let (children, child_acc, child_streak) = match root.examine(1.0, &root_acc, 0)? {
        Node::Leaf => return Ok(assemble(model, settings, vec![root.out])),
        Node::Expand {
            children,
            child_acc,
            child_streak,
        } => (children, child_acc, child_streak),
    };
    let root_snapshot = root.engine.snapshot();
    let shares = split_budget(
        settings
            .cutoffs
            .max_branches
            .map(|cap| cap.saturating_sub(root.out.expanded)),
        children.len(),
    );

    let explore_child = |k: usize| -> Result<Partial, EngineError> {
        let mut walker = Walker::new(
            model,
            settings,
            Engine::new(model, config.clone())?,
            shares[k],
        );
        walker.engine.restore(&root_snapshot);
        walker.explore_below(1.0, 0, child_streak, &child_acc, children[k])?;
        Ok(walker.out)
    };
    let compute = || -> Vec<Result<Partial, EngineError>> {
        (0..children.len())
            .into_par_iter()
            .map(explore_child)
            .collect()
    };
    let outcomes = match settings.threads {
        None => compute(),
        Some(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| invalid("threads", format!("thread-pool construction failed: {e}")))?
            .install(compute),
    };

    // Reduction in child order: the first error in exploration order wins,
    // and every sum is taken in that order.
    let mut partials = vec![root.out];
    for outcome in outcomes {
        partials.push(outcome?);
    }
    Ok(assemble(model, settings, partials))
}

/// The engine configuration of an exploration: no journal, no samples, no
/// trace (the explorer builds its own), the target latch on.
fn engine_config() -> EngineConfig {
    EngineConfig {
        journal: false,
        sequences: false,
        stop_at_targets: true,
        samples: Vec::new(),
        ..EngineConfig::default()
    }
}

fn invalid(parameter: &str, detail: String) -> EngineError {
    EngineError::InvalidStudyParameter {
        parameter: parameter.to_owned(),
        detail,
    }
}

/// Up-front validation of the settings against the model.
fn validate(model: &CompiledModel, settings: &ExactSettings) -> Result<(), EngineError> {
    if !model.targets.iter().any(|t| t.name == settings.target) {
        let known: Vec<&str> = model.targets.iter().map(|t| t.name.as_str()).collect();
        return Err(invalid(
            "target",
            format!(
                "no target named `{}` in the model (declared: [{}])",
                settings.target,
                known.join(", ")
            ),
        ));
    }
    if !settings.horizon.is_finite() || settings.horizon < 0.0 {
        return Err(invalid(
            "horizon",
            format!(
                "a horizon is a finite nonnegative time, got {}",
                settings.horizon
            ),
        ));
    }
    let cutoffs = &settings.cutoffs;
    if let Some(p) = cutoffs.min_probability {
        if p.is_nan() || p <= 0.0 || p > 1.0 {
            return Err(invalid(
                "min_probability",
                format!("a minimal probability lies in (0, 1], got {p}"),
            ));
        }
    }
    if cutoffs.max_branches == Some(0) {
        return Err(invalid(
            "max_branches",
            "the branch cap counts expanded nodes and must be at least 1, got 0".to_owned(),
        ));
    }
    if cutoffs.max_failures.is_some()
        && !model
            .transitions
            .iter()
            .any(|t| t.kind == Some(TransitionKind::Failure))
    {
        return Err(invalid(
            "max_failures",
            "no transition of the model is declared a `failure` (kind), so a failure \
             count would silently count nothing"
                .to_owned(),
        ));
    }
    if settings.gap_tolerance.is_nan() || !(0.0..=1.0).contains(&settings.gap_tolerance) {
        return Err(invalid(
            "gap_tolerance",
            format!(
                "a relative gap tolerance lies in [0, 1], got {}",
                settings.gap_tolerance
            ),
        ));
    }
    if settings.threads == Some(0) {
        return Err(invalid(
            "threads",
            "at least one thread is needed, got 0".to_owned(),
        ));
    }
    PhaseTypeAccumulator::new(0.0, (&settings.precision).into()).map_err(|e| match e {
        PhaseTypeError::InvalidSettings { reason } => invalid("precision", reason),
        other => invalid("precision", other.to_string()),
    })?;
    Ok(())
}

/// Branch-cap shares of the root's children: `remaining` split equally,
/// the remainder to the first children in order. `None` when uncapped.
fn split_budget(remaining: Option<u64>, children: usize) -> Vec<Option<u64>> {
    let Some(remaining) = remaining else {
        return vec![None; children];
    };
    let n = children as u64;
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|k| Some(remaining / n + u64::from(k < remaining % n)))
        .collect()
}

/// Final assembly: partial results summed in the given (exploration)
/// order, the step table built from the distinct steps of the retained
/// paths, then sequences ranked.
fn assemble(
    model: &CompiledModel,
    settings: &ExactSettings,
    partials: Vec<Partial>,
) -> ExplorationResult {
    let mut raw = Vec::new();
    let mut lower = 0.0;
    let mut tallies = CutoffTallies::default();
    let mut expanded = 0;
    for partial in partials {
        lower += partial.lower;
        tallies.merge(&partial.tallies);
        expanded += partial.expanded;
        raw.extend(partial.sequences);
    }
    let upper = lower + tallies.total_mass();

    // The table: every distinct (transition, branch), in that order, so it
    // depends on the retained set only, never on the thread count.
    let mut ids: BTreeMap<(u32, u32), u32> = raw
        .iter()
        .flat_map(|sequence| sequence.path.iter().map(|&pair| (pair, 0)))
        .collect();
    let mut steps = Vec::with_capacity(ids.len());
    for (k, (&(idx, branch), id)) in ids.iter_mut().enumerate() {
        // Fewer distinct steps than (transition, branch) pairs: fits a u32.
        *id = k as u32;
        steps.push(resolve_step(model, idx as usize, branch as usize));
    }
    let mut sequences: Vec<ExploredSequence> = raw
        .into_iter()
        .map(|sequence| ExploredSequence {
            steps: sequence.path.iter().map(|pair| ids[pair]).collect(),
            end_cause: settings.target.clone(),
            probability: sequence.probability,
            error_bound: sequence.error_bound,
            imprecise: sequence.imprecise,
        })
        .collect();
    let key = |id: &u32| {
        let step = &steps[*id as usize];
        (&step.transition, &step.from, &step.to)
    };
    sequences.sort_by(|a, b| {
        b.probability
            .total_cmp(&a.probability)
            .then_with(|| a.steps.iter().map(key).cmp(b.steps.iter().map(key)))
    });
    let imprecise_sequences = sequences.iter().filter(|s| s.imprecise).count();
    ExplorationResult {
        format: EXPLORATION_FORMAT.to_owned(),
        version: EXPLORATION_VERSION,
        engine_version: env!("CARGO_PKG_VERSION").to_owned(),
        model: model.name.clone(),
        algorithm: Algorithm::Exact,
        target: settings.target.clone(),
        horizon: settings.horizon,
        cutoffs: settings.cutoffs.clone(),
        gap_tolerance: settings.gap_tolerance,
        precision: settings.precision.clone(),
        inconclusive: relative_gap(lower, upper) > settings.gap_tolerance,
        steps,
        sequences,
        lower,
        upper,
        cutoff_tallies: tallies,
        expanded_nodes: expanded,
        imprecise_sequences,
    }
}

/// The resolved form of step `(idx, branch)`: names, and the monitored
/// event when the transition is monitored.
fn resolve_step(model: &CompiledModel, idx: usize, branch: usize) -> ExploredStep {
    let transition = &model.transitions[idx];
    let automaton = &model.automata[transition.automaton];
    let to = automaton.states[transition.targets[branch]].clone();
    ExploredStep {
        transition: transition.name.clone(),
        from: automaton.states[transition.source].clone(),
        event: transition.monitored.then(|| ExploredEvent {
            obj: transition.component.clone(),
            attr: to.clone(),
            cycle_group: transition.cycle_group.clone(),
        }),
        to,
    }
}

/// A retained sequence as the walker records it: its path as compact
/// `(transition, branch)` pairs, resolved into the step table only at
/// assembly.
#[derive(Debug)]
struct RawSequence {
    path: Vec<(u32, u32)>,
    probability: f64,
    error_bound: f64,
    imprecise: bool,
}

/// What one worker found: its retained sequences in exploration order,
/// their probability summed in that order, and its cut-off tallies.
#[derive(Debug, Default)]
struct Partial {
    sequences: Vec<RawSequence>,
    lower: f64,
    tallies: CutoffTallies,
    expanded: u64,
}

/// One branch out of a node: fire `transition` into its target at
/// position `branch`, with conditional probability `factor`.
#[derive(Debug, Clone, Copy)]
struct Child {
    transition: usize,
    branch: usize,
    factor: f64,
}

/// A node, examined.
enum Node {
    /// Nothing to explore below (retained sequence, other target,
    /// absorbing state, or branch cap).
    Leaf,
    /// Children in visiting order, and the accumulator they share (the
    /// node's own, extended by the node's sojourn when it is timed).
    Expand {
        children: Vec<Child>,
        child_acc: PhaseTypeAccumulator,
        /// Consecutive instantaneous firings leading to each child: the
        /// node's own count plus one when the node fires an instantaneous
        /// transition, 0 when it takes a timed step.
        child_streak: usize,
    },
}

/// A node of the current path whose children are being visited: the
/// heap-allocated replacement of a native recursion frame.
struct Frame {
    /// Embedded-path probability of the node.
    pi: f64,
    /// Failures fired from the initial state to the node.
    failures: u64,
    /// Consecutive instantaneous firings leading to each child.
    child_streak: usize,
    /// Children in visiting order.
    children: Vec<Child>,
    /// The accumulator the children share.
    child_acc: PhaseTypeAccumulator,
    /// The node's snapshot, taken only when it has more than one child
    /// (a lone child never needs the node restored).
    snapshot: Option<Snapshot>,
    /// Index of the next child to visit.
    next: usize,
    /// Whether the engine has moved since the snapshot was taken.
    dirty: bool,
}

/// A depth-first walker over one engine.
struct Walker<'a, 'm> {
    model: &'m CompiledModel,
    settings: &'a ExactSettings,
    engine: Engine<'m>,
    /// `(transition, branch)` fired from the initial state to the current
    /// node, compact: a transition index and a branch index both fit a
    /// `u32` (a compiled model's transitions are counted in thousands).
    path: Vec<(u32, u32)>,
    /// Remaining expansions allowed (`None` = uncapped).
    budget: Option<u64>,
    /// Cap on consecutive instantaneous firings along the path
    /// ([`EngineConfig::max_fixpoint_iterations`]).
    max_instantaneous: usize,
    out: Partial,
}

impl<'a, 'm> Walker<'a, 'm> {
    fn new(
        model: &'m CompiledModel,
        settings: &'a ExactSettings,
        engine: Engine<'m>,
        budget: Option<u64>,
    ) -> Self {
        Walker {
            model,
            settings,
            engine,
            path: Vec::new(),
            budget,
            max_instantaneous: engine_config().max_fixpoint_iterations,
            out: Partial::default(),
        }
    }

    /// Examine the current node, of embedded-path probability `pi`,
    /// sojourn accumulator `acc`, reached after `streak` consecutive
    /// instantaneous firings: record it when it is a retained sequence,
    /// tally it when the branch cap stops it, or list its children.
    fn examine(
        &mut self,
        pi: f64,
        acc: &PhaseTypeAccumulator,
        streak: usize,
    ) -> Result<Node, EngineError> {
        let armed = self.engine.fireable();

        // 1. Instantaneous first, lowest index first.
        let instantaneous = armed
            .iter()
            .map(|f| f.index)
            .filter(|&idx| is_instantaneous(&self.model.transitions[idx].distrib))
            .min();
        let (children, child_acc, child_streak) = if let Some(idx) = instantaneous {
            if streak >= self.max_instantaneous {
                let prefix = self.path.len().saturating_sub(streak);
                return Err(EngineError::InstantaneousCycle {
                    transition: self.model.transitions[idx].name.clone(),
                    firings: self.max_instantaneous,
                    sequence: self.path_names()[..prefix].to_vec(),
                });
            }
            let children = match &self.model.transitions[idx].distrib {
                CLaw::Inst(probs) => probs
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| **p > 0.0)
                    .map(|(branch, p)| Child {
                        transition: idx,
                        branch,
                        factor: *p,
                    })
                    .collect(),
                _ => vec![Child {
                    transition: idx,
                    branch: 0,
                    factor: 1.0,
                }],
            };
            let mut child_acc = acc.clone();
            child_acc.push_instantaneous();
            (children, child_acc, streak + 1)
        } else {
            // 2. Target reached: a retained sequence (or another target:
            //    a leaf that contributes nothing).
            if let Some((name, _)) = self.engine.reached_target() {
                if name == self.settings.target {
                    self.record(pi, acc);
                }
                return Ok(Node::Leaf);
            }
            // 3. Exponential competition, in transition index order; any
            //    other armed law is outside the domain.
            let mut indices: Vec<usize> = armed.iter().map(|f| f.index).collect();
            indices.sort_unstable();
            let mut rates = Vec::with_capacity(indices.len());
            for idx in indices {
                let law = &self.model.transitions[idx].distrib;
                match law {
                    CLaw::Exp(_)
                    | CLaw::ExpVar {
                        continuous: false, ..
                    } => {
                        let rate = self.engine.armed_rate(idx)?.unwrap_or(0.0);
                        if rate > 0.0 {
                            rates.push((idx, rate));
                        }
                    }
                    _ => {
                        return Err(EngineError::LawOutsideExactDomain {
                            transition: self.model.transitions[idx].name.clone(),
                            law: law_label(law),
                            sequence: self.path_names(),
                        })
                    }
                }
            }
            let total: f64 = rates.iter().map(|(_, rate)| rate).sum();
            if total <= 0.0 {
                // Absorbing leaf.
                return Ok(Node::Leaf);
            }
            let children = rates
                .iter()
                .map(|&(idx, rate)| Child {
                    transition: idx,
                    branch: 0,
                    factor: rate / total,
                })
                .collect();
            let mut child_acc = acc.clone();
            child_acc
                .push(total)
                .map_err(|e: PhaseTypeError| EngineError::TypeError {
                    time: 0.0,
                    detail: format!("sojourn rate after [{}]: {e}", self.path_names().join(", ")),
                })?;
            (children, child_acc, 0)
        };

        // Branch cap: counts expanded nodes.
        if let Some(budget) = self.budget.as_mut() {
            if *budget == 0 {
                self.out
                    .tallies
                    .max_branches
                    .add(pi * acc.absorption().value);
                return Ok(Node::Leaf);
            }
            *budget -= 1;
        }
        self.out.expanded += 1;
        Ok(Node::Expand {
            children,
            child_acc,
            child_streak,
        })
    }

    /// Explore the subtree below `child` of the current node (embedded
    /// probability `pi`, `failures` so far, children reached after
    /// `streak` consecutive instantaneous firings, sharing `child_acc`),
    /// the engine sitting at that node.
    ///
    /// Depth first over an explicit frame stack, in exactly the order a
    /// recursion takes: a node's children in order, each child's subtree
    /// before the next child. The bottom frame holds `child` alone and is
    /// not a node of the path; every frame above it is.
    fn explore_below(
        &mut self,
        pi: f64,
        failures: u64,
        streak: usize,
        child_acc: &PhaseTypeAccumulator,
        child: Child,
    ) -> Result<(), EngineError> {
        let mut stack = vec![Frame {
            pi,
            failures,
            child_streak: streak,
            children: vec![child],
            child_acc: child_acc.clone(),
            snapshot: None,
            next: 0,
            dirty: false,
        }];
        while let Some(top) = stack.len().checked_sub(1) {
            let frame = &mut stack[top];
            let Some(&child) = frame.children.get(frame.next) else {
                stack.pop();
                if !stack.is_empty() {
                    self.path.pop();
                }
                continue;
            };
            frame.next += 1;
            let Some((child_pi, child_failures)) =
                self.admit(frame.pi, frame.failures, &frame.child_acc, &child)
            else {
                continue;
            };
            if frame.dirty {
                if let Some(snapshot) = &frame.snapshot {
                    self.engine.restore(snapshot);
                }
            }
            frame.dirty = true;
            let child_streak = frame.child_streak;
            self.engine.fire_now(child.transition, Some(child.branch))?;
            // The explorer keeps its own path: the engine's history would
            // make every snapshot grow with the depth (quadratic memory).
            self.engine.forget_history();
            self.path
                .push((child.transition as u32, child.branch as u32));
            match self.examine(child_pi, &stack[top].child_acc, child_streak)? {
                Node::Leaf => {
                    self.path.pop();
                }
                Node::Expand {
                    children,
                    child_acc,
                    child_streak,
                } => {
                    let snapshot = (children.len() > 1).then(|| self.engine.snapshot());
                    stack.push(Frame {
                        pi: child_pi,
                        failures: child_failures,
                        child_streak,
                        children,
                        child_acc,
                        snapshot,
                        next: 0,
                        dirty: false,
                    });
                }
            }
        }
        Ok(())
    }

    /// Apply the cut-offs to `child` of a node (probability `pi`,
    /// `failures` so far, children sharing `child_acc`): `None` when it
    /// carries no mass or is pruned (its mass then tallied), otherwise its
    /// embedded probability and failure count.
    fn admit(
        &mut self,
        pi: f64,
        failures: u64,
        child_acc: &PhaseTypeAccumulator,
        child: &Child,
    ) -> Option<(f64, u64)> {
        let child_pi = pi * child.factor;
        let mass = child_pi * child_acc.absorption().value;
        if mass <= 0.0 {
            return None;
        }
        let transition = &self.model.transitions[child.transition];
        let is_failure = transition.kind == Some(TransitionKind::Failure) && child.branch == 0;
        let child_failures = failures + u64::from(is_failure);
        let length = self.path.len() + 1;
        let cutoffs = &self.settings.cutoffs;
        let tallies = &mut self.out.tallies;
        if cutoffs.min_probability.is_some_and(|p| mass < p) {
            tallies.min_probability.add(mass);
            return None;
        }
        if cutoffs.max_length.is_some_and(|n| length > n) {
            tallies.max_length.add(mass);
            return None;
        }
        if cutoffs.max_failures.is_some_and(|n| child_failures > n) {
            tallies.max_failures.add(mass);
            return None;
        }
        Some((child_pi, child_failures))
    }

    /// Record the current node as a retained sequence.
    fn record(&mut self, pi: f64, acc: &PhaseTypeAccumulator) {
        let absorption = acc.absorption();
        let probability = pi * absorption.value;
        if probability <= 0.0 {
            return;
        }
        self.out.lower += probability;
        self.out.sequences.push(RawSequence {
            path: self.path.clone(),
            probability,
            error_bound: pi * absorption.error_bound,
            imprecise: absorption.flagged,
        });
    }

    fn path_names(&self) -> Vec<String> {
        self.path
            .iter()
            .map(|&(idx, _)| self.model.transitions[idx as usize].name.clone())
            .collect()
    }
}

/// Whether a law fires with zero sojourn: an instantaneous branching or a
/// zero delay.
fn is_instantaneous(law: &CLaw) -> bool {
    match law {
        CLaw::Inst(_) => true,
        CLaw::Delay(delay) => *delay == 0.0,
        _ => false,
    }
}

/// A short readable label of a law, for the out-of-domain error.
fn law_label(law: &CLaw) -> String {
    match law {
        CLaw::Delay(delay) => format!("delay {delay}"),
        CLaw::Inst(_) => "instantaneous".to_owned(),
        CLaw::Watched { .. } => "watched boundary".to_owned(),
        CLaw::Exp(rate) => format!("exponential {rate}"),
        CLaw::ExpVar { continuous, .. } => {
            if *continuous {
                "exponential with a continuously varying rate".to_owned()
            } else {
                "state-dependent exponential".to_owned()
            }
        }
        CLaw::Weibull(shape, scale) => format!("weibull shape {shape} scale {scale}"),
        CLaw::Lognormal(mu, sigma) => format!("lognormal mu {mu} sigma {sigma}"),
        CLaw::Gamma(shape, scale) => format!("gamma shape {shape} scale {scale}"),
        CLaw::Uniform(low, high) => format!("uniform [{low}, {high})"),
        CLaw::Empirical(_) => "empirical".to_owned(),
    }
}
