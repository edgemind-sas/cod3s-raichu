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

use std::collections::VecDeque;

use raichu_core::compile::{CExpr, CLaw, CStep};
#[cfg(doc)]
use raichu_core::EngineConfig;
use raichu_core::{CompiledModel, Engine, EngineError};
use raichu_numeric::{PhaseTypeAccumulator, PhaseTypeError};
use serde::{Deserialize, Serialize};

use crate::result::{Algorithm, Cutoffs, ExplorationResult, Precision, DEFAULT_GAP_TOLERANCE};
use crate::walk::{
    assemble, drive, engine_config, instantaneous_branches, invalid, path_names, validate_common,
    Common, Expansion, NodeMass, Strategy, Timed,
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
    let common = settings.common();
    validate(model, settings)?;
    let violations = exact_domain_report(model);
    if !violations.is_empty() {
        return Err(EngineError::OutsideExactDomain {
            reasons: violations.iter().map(ToString::to_string).collect(),
        });
    }
    let root_acc = PhaseTypeAccumulator::new(settings.horizon, (&settings.precision).into())
        .map_err(|e| invalid("precision", e.to_string()))?;
    let partials = drive(
        model,
        &common,
        &ExactStrategy { model },
        &engine_config(),
        &root_acc,
        false,
    )?;
    Ok(assemble(
        model,
        &common,
        Algorithm::Exact,
        &settings.precision,
        partials,
        false,
    ))
}

impl ExactSettings {
    fn common(&self) -> Common<'_> {
        Common {
            target: &self.target,
            horizon: self.horizon,
            cutoffs: &self.cutoffs,
            gap_tolerance: self.gap_tolerance,
            threads: self.threads,
        }
    }
}

/// Up-front validation of the settings against the model.
fn validate(model: &CompiledModel, settings: &ExactSettings) -> Result<(), EngineError> {
    validate_common(model, &settings.common())?;
    PhaseTypeAccumulator::new(0.0, (&settings.precision).into()).map_err(|e| match e {
        PhaseTypeError::InvalidSettings { reason } => invalid("precision", reason),
        other => invalid("precision", other.to_string()),
    })?;
    Ok(())
}

/// One branch out of a node: fire `transition` into its target at
/// position `branch`, with conditional probability `factor`.
#[derive(Debug, Clone, Copy)]
struct Child {
    transition: usize,
    branch: usize,
    factor: f64,
}

/// The exact node expansion: branches over the embedded jump chain, the
/// children of a node sharing the phase accumulator of the sojourns
/// leading to them.
struct ExactStrategy<'m> {
    model: &'m CompiledModel,
}

impl Strategy for ExactStrategy<'_> {
    type Child = Child;
    type Shared = PhaseTypeAccumulator;

    fn immediate(&self, engine: &Engine<'_>) -> Option<usize> {
        let model = self.model;
        engine
            .fireable()
            .iter()
            .map(|f| f.index)
            .filter(|&idx| is_instantaneous(&model.transitions[idx].distrib))
            .min()
    }

    fn instantaneous_children(
        &self,
        model: &CompiledModel,
        idx: usize,
        parent: &PhaseTypeAccumulator,
        _via: Option<&Child>,
    ) -> (Vec<Child>, PhaseTypeAccumulator) {
        let children = instantaneous_branches(model, idx)
            .into_iter()
            .map(|(branch, factor)| Child {
                transition: idx,
                branch,
                factor,
            })
            .collect();
        let mut child_acc = parent.clone();
        child_acc.push_instantaneous();
        (children, child_acc)
    }

    fn timed(
        &self,
        engine: &mut Engine<'_>,
        parent: &PhaseTypeAccumulator,
        _via: Option<&Child>,
        path: &[(u32, u32)],
    ) -> Timed<Child, PhaseTypeAccumulator> {
        let model = self.model;
        // Exponential competition, in transition index order; any other
        // armed law is outside the domain.
        let mut indices: Vec<usize> = engine.fireable().iter().map(|f| f.index).collect();
        indices.sort_unstable();
        let mut rates = Vec::with_capacity(indices.len());
        for idx in indices {
            let law = &model.transitions[idx].distrib;
            match law {
                CLaw::Exp(_)
                | CLaw::ExpVar {
                    continuous: false, ..
                } => {
                    let rate = engine.armed_rate(idx)?.unwrap_or(0.0);
                    if rate > 0.0 {
                        rates.push((idx, rate));
                    }
                }
                _ => {
                    return Err(EngineError::LawOutsideExactDomain {
                        transition: model.transitions[idx].name.clone(),
                        law: law_label(law),
                        sequence: path_names(model, path),
                    })
                }
            }
        }
        let total: f64 = rates.iter().map(|(_, rate)| rate).sum();
        if total <= 0.0 {
            // Absorbing leaf.
            return Ok(None);
        }
        let children = rates
            .iter()
            .map(|&(idx, rate)| Child {
                transition: idx,
                branch: 0,
                factor: rate / total,
            })
            .collect();
        let mut child_acc = parent.clone();
        child_acc
            .push(total)
            .map_err(|e: PhaseTypeError| EngineError::TypeError {
                time: 0.0,
                detail: format!(
                    "sojourn rate after [{}]: {e}",
                    path_names(model, path).join(", ")
                ),
            })?;
        Ok(Some(Expansion {
            children,
            shared: child_acc,
            snapshot: None,
        }))
    }

    fn node_mass(&self, pi: f64, parent: &PhaseTypeAccumulator, _via: Option<&Child>) -> NodeMass {
        let absorption = parent.absorption();
        NodeMass {
            value: pi * absorption.value,
            error_bound: pi * absorption.error_bound,
            imprecise: absorption.flagged,
        }
    }

    fn child_mass(&self, pi: f64, shared: &PhaseTypeAccumulator, child: &Child) -> (f64, f64, f64) {
        let child_pi = pi * child.factor;
        let mass = child_pi * shared.absorption().value;
        (child_pi, mass, mass)
    }

    fn fire(&self, engine: &mut Engine<'_>, child: &Child) -> Result<(), EngineError> {
        engine
            .fire_now(child.transition, Some(child.branch))
            .map(|_| ())
    }

    fn step(&self, child: &Child) -> Option<(u32, u32)> {
        Some((child.transition as u32, child.branch as u32))
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
