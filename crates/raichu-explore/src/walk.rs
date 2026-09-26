//! The depth-first walk shared by the exploration drivers.
//!
//! Both drivers explore the same tree shape over the same engine: a node
//! is an engine state, its children are firings, and the walk handles
//! everything that does not depend on how a node's children are built:
//!
//! - the iterative depth-first order over an explicit, heap-allocated
//!   frame stack, a node's snapshot taken only when it has more than one
//!   child, and the engine's history forgotten after every firing;
//! - an armed instantaneous transition firing first, its consecutive
//!   firings capped ([`EngineError::InstantaneousCycle`]);
//! - the target check (a retained sequence, or another target: a leaf);
//! - the cut-offs (minimal probability, maximal length, maximal failures,
//!   branch cap) and their tallies, which make the upper bound;
//! - the split of the root's children across workers, each with its own
//!   engine and its own share of the branch cap, reduced in child order;
//! - the assembly of the result (step table, ranking, bounds).
//!
//! A driver supplies a [`Strategy`]: which transition fires at the
//! current instant, how a timed node branches, how a node's probability
//! is measured, and how a child is fired.

use std::collections::{BTreeMap, HashMap};

use raichu_core::compile::CLaw;
use raichu_core::{CompiledModel, Engine, EngineConfig, EngineError, Snapshot};
use raichu_model::TransitionKind;

use crate::result::{
    relative_gap, Algorithm, CutoffTallies, Cutoffs, ExplorationResult, ExploredEvent,
    ExploredSequence, ExploredStep, Precision, EXPLORATION_FORMAT,
};

/// The settings every driver shares, borrowed from its own settings type.
pub(crate) struct Common<'a> {
    /// Target (feared event) name.
    pub target: &'a str,
    /// Horizon, in the model's time unit.
    pub horizon: f64,
    /// Declared cut-offs.
    pub cutoffs: &'a Cutoffs,
    /// Relative gap tolerance.
    pub gap_tolerance: f64,
    /// Worker threads (`None` = rayon default).
    pub threads: Option<usize>,
}

/// A timed expansion of a node, built by [`Strategy::timed`].
pub(crate) struct Expansion<C, Sh> {
    /// Children in visiting order.
    pub children: Vec<C>,
    /// What they share.
    pub shared: Sh,
    /// A snapshot of the node's engine state the strategy already took
    /// (the walker reuses it instead of taking another).
    pub snapshot: Option<Snapshot>,
}

/// What [`Strategy::timed`] returns: the node's children, or `None` for a
/// leaf that contributes nothing.
pub(crate) type Timed<C, Sh> = Result<Option<Expansion<C, Sh>>, EngineError>;

/// How a driver builds and measures the children of a node.
///
/// A node is described by its path probability `pi`, the `parent` state
/// its siblings share, and the child `via` which it was reached (`None`
/// at the root).
pub(crate) trait Strategy: Sync {
    /// One branch out of a node.
    type Child: Copy + Send + Sync;
    /// What the children of one node share.
    type Shared: Clone + Send + Sync;

    /// The transition that fires at the current instant before anything
    /// else, lowest index first, if any.
    fn immediate(&self, engine: &Engine<'_>) -> Option<usize>;

    /// The children of the node when `idx` fires at the current instant,
    /// and what they share.
    fn instantaneous_children(
        &self,
        model: &CompiledModel,
        idx: usize,
        parent: &Self::Shared,
        via: Option<&Self::Child>,
    ) -> (Vec<Self::Child>, Self::Shared);

    /// The children of a node where nothing fires at the current instant
    /// and no target is reached; `None` for a leaf that contributes
    /// nothing. The engine must be left at the node's state.
    fn timed(
        &self,
        engine: &mut Engine<'_>,
        parent: &Self::Shared,
        via: Option<&Self::Child>,
        path: &[(u32, u32)],
    ) -> Timed<Self::Child, Self::Shared>;

    /// The node's probability measure: what a retained sequence
    /// contributes and what a pruned node adds to the upper bound, with
    /// the bound on its absolute error and its imprecision flag.
    fn node_mass(&self, pi: f64, parent: &Self::Shared, via: Option<&Self::Child>) -> NodeMass;

    /// A child's path probability, its mass (what pruning it adds to the
    /// upper bound) and the score the minimal-probability cut-off
    /// compares.
    fn child_mass(&self, pi: f64, shared: &Self::Shared, child: &Self::Child) -> (f64, f64, f64);

    /// Fire `child` on the engine sitting at its parent node.
    fn fire(&self, engine: &mut Engine<'_>, child: &Self::Child) -> Result<(), EngineError>;

    /// The `(transition, branch)` a child fires, `None` when it fires
    /// nothing (it only lets time pass).
    fn step(&self, child: &Self::Child) -> Option<(u32, u32)>;
}

/// A node's probability measure (see [`Strategy::node_mass`]).
pub(crate) struct NodeMass {
    pub value: f64,
    pub error_bound: f64,
    pub imprecise: bool,
}

/// A retained sequence as the walker records it: its path as compact
/// `(transition, branch)` pairs, resolved into the step table only at
/// assembly.
#[derive(Debug)]
pub(crate) struct RawSequence {
    pub path: Vec<(u32, u32)>,
    pub probability: f64,
    pub error_bound: f64,
    pub imprecise: bool,
}

/// What one worker found: its retained sequences in exploration order,
/// their probability summed in that order, and its cut-off tallies.
#[derive(Debug, Default)]
pub(crate) struct Partial {
    pub sequences: Vec<RawSequence>,
    pub lower: f64,
    pub tallies: CutoffTallies,
    pub expanded: u64,
    /// Position of each path in `sequences`, when merging.
    index: HashMap<Vec<(u32, u32)>, usize>,
}

/// A node, examined.
enum Node<C, Sh> {
    /// Nothing to explore below.
    Leaf,
    /// Children in visiting order, what they share, the number of
    /// consecutive instantaneous firings leading to each, and an optional
    /// ready snapshot of the node.
    Expand {
        children: Vec<C>,
        shared: Sh,
        child_streak: usize,
        snapshot: Option<Box<Snapshot>>,
    },
}

/// A node of the current path whose children are being visited: the
/// heap-allocated replacement of a native recursion frame.
struct Frame<C, Sh> {
    /// Path probability of the node.
    pi: f64,
    /// Failures fired from the initial state to the node.
    failures: u64,
    /// Consecutive instantaneous firings leading to each child.
    child_streak: usize,
    /// Children in visiting order.
    children: Vec<C>,
    /// What the children share.
    shared: Sh,
    /// The node's snapshot, taken only when it has more than one child
    /// (a lone child never needs the node restored).
    snapshot: Option<Snapshot>,
    /// Index of the next child to visit.
    next: usize,
    /// Whether the engine has moved since the snapshot was taken.
    dirty: bool,
    /// Whether reaching this node pushed a step on the path.
    pushed: bool,
}

/// A depth-first walker over one engine.
pub(crate) struct Walker<'a, 'm, S: Strategy> {
    model: &'m CompiledModel,
    common: &'a Common<'a>,
    strategy: &'a S,
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
    /// Whether retained sequences sharing a path are merged.
    merge: bool,
    out: Partial,
}

impl<'a, 'm, S: Strategy> Walker<'a, 'm, S> {
    fn new(
        model: &'m CompiledModel,
        common: &'a Common<'a>,
        strategy: &'a S,
        engine: Engine<'m>,
        budget: Option<u64>,
        max_instantaneous: usize,
        merge: bool,
    ) -> Self {
        Walker {
            model,
            common,
            strategy,
            engine,
            path: Vec::new(),
            budget,
            max_instantaneous,
            merge,
            out: Partial::default(),
        }
    }

    /// Examine the current node (path probability `pi`, siblings sharing
    /// `parent`, reached via `via`, after `streak` consecutive
    /// instantaneous firings): record it when it is a retained sequence,
    /// tally it when the branch cap stops it, or list its children.
    fn examine(
        &mut self,
        pi: f64,
        parent: &S::Shared,
        via: Option<&S::Child>,
        streak: usize,
    ) -> Result<Node<S::Child, S::Shared>, EngineError> {
        // 1. Instantaneous first, lowest index first.
        let (children, shared, child_streak, snapshot) =
            if let Some(idx) = self.strategy.immediate(&self.engine) {
                if streak >= self.max_instantaneous {
                    let prefix = self.path.len().saturating_sub(streak);
                    return Err(EngineError::InstantaneousCycle {
                        transition: self.model.transitions[idx].name.clone(),
                        firings: self.max_instantaneous,
                        sequence: path_names(self.model, &self.path)[..prefix].to_vec(),
                    });
                }
                let (children, shared) = self
                    .strategy
                    .instantaneous_children(self.model, idx, parent, via);
                (children, shared, streak + 1, None)
            } else {
                // 2. Target reached: a retained sequence (or another
                //    target: a leaf that contributes nothing).
                if let Some((name, _)) = self.engine.reached_target() {
                    if name == self.common.target {
                        self.record(pi, parent, via);
                    }
                    return Ok(Node::Leaf);
                }
                // 3. Timed branching, per the driver.
                match self
                    .strategy
                    .timed(&mut self.engine, parent, via, &self.path)?
                {
                    None => return Ok(Node::Leaf),
                    Some(expansion) => (
                        expansion.children,
                        expansion.shared,
                        0,
                        expansion.snapshot.map(Box::new),
                    ),
                }
            };

        // Branch cap: counts expanded nodes.
        if let Some(budget) = self.budget.as_mut() {
            if *budget == 0 {
                let mass = self.strategy.node_mass(pi, parent, via);
                self.out.tallies.max_branches.add(mass.value);
                return Ok(Node::Leaf);
            }
            *budget -= 1;
        }
        self.out.expanded += 1;
        Ok(Node::Expand {
            children,
            shared,
            child_streak,
            snapshot,
        })
    }

    /// Explore the subtree below `child` of the current node (path
    /// probability `pi`, `failures` so far, children reached after
    /// `streak` consecutive instantaneous firings, sharing `shared`), the
    /// engine sitting at that node.
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
        shared: &S::Shared,
        child: S::Child,
    ) -> Result<(), EngineError> {
        let mut stack = vec![Frame {
            pi,
            failures,
            child_streak: streak,
            children: vec![child],
            shared: shared.clone(),
            snapshot: None,
            next: 0,
            dirty: false,
            pushed: false,
        }];
        while let Some(top) = stack.len().checked_sub(1) {
            let frame = &mut stack[top];
            let Some(&child) = frame.children.get(frame.next) else {
                if let Some(frame) = stack.pop() {
                    if frame.pushed {
                        self.path.pop();
                    }
                }
                continue;
            };
            frame.next += 1;
            let Some((child_pi, child_failures)) =
                self.admit(frame.pi, frame.failures, &frame.shared, &child)
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
            self.strategy.fire(&mut self.engine, &child)?;
            // The explorer keeps its own path: the engine's history would
            // make every snapshot grow with the depth (quadratic memory).
            self.engine.forget_history();
            let step = self.strategy.step(&child);
            if let Some(step) = step {
                self.path.push(step);
            }
            match self.examine(child_pi, &stack[top].shared, Some(&child), child_streak)? {
                Node::Leaf => {
                    if step.is_some() {
                        self.path.pop();
                    }
                }
                Node::Expand {
                    children,
                    shared,
                    child_streak,
                    snapshot,
                } => {
                    let snapshot = if children.len() > 1 {
                        snapshot
                            .map(|s| *s)
                            .or_else(|| Some(self.engine.snapshot()))
                    } else {
                        None
                    };
                    stack.push(Frame {
                        pi: child_pi,
                        failures: child_failures,
                        child_streak,
                        children,
                        shared,
                        snapshot,
                        next: 0,
                        dirty: false,
                        pushed: step.is_some(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Apply the cut-offs to `child` of a node (path probability `pi`,
    /// `failures` so far, children sharing `shared`): `None` when it
    /// carries no mass or is pruned (its mass then tallied), otherwise its
    /// path probability and failure count.
    fn admit(
        &mut self,
        pi: f64,
        failures: u64,
        shared: &S::Shared,
        child: &S::Child,
    ) -> Option<(f64, u64)> {
        let (child_pi, mass, score) = self.strategy.child_mass(pi, shared, child);
        if mass <= 0.0 {
            return None;
        }
        let step = self.strategy.step(child);
        let is_failure = step.is_some_and(|(idx, branch)| {
            self.model.transitions[idx as usize].kind == Some(TransitionKind::Failure)
                && branch == 0
        });
        let child_failures = failures + u64::from(is_failure);
        let length = self.path.len() + usize::from(step.is_some());
        let cutoffs = self.common.cutoffs;
        let tallies = &mut self.out.tallies;
        if cutoffs.min_probability.is_some_and(|p| score < p) {
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

    /// Record the current node as a retained sequence, merged with an
    /// earlier one of the same path when merging.
    fn record(&mut self, pi: f64, parent: &S::Shared, via: Option<&S::Child>) {
        let mass = self.strategy.node_mass(pi, parent, via);
        let probability = mass.value;
        if probability <= 0.0 {
            return;
        }
        self.out.lower += probability;
        let sequence = RawSequence {
            path: self.path.clone(),
            probability,
            error_bound: mass.error_bound,
            imprecise: mass.imprecise,
        };
        if self.merge {
            merge_by_path(&mut self.out.sequences, &mut self.out.index, sequence);
        } else {
            self.out.sequences.push(sequence);
        }
    }
}

/// Qualified names of the transitions along `path`.
pub(crate) fn path_names(model: &CompiledModel, path: &[(u32, u32)]) -> Vec<String> {
    path.iter()
        .map(|&(idx, _)| model.transitions[idx as usize].name.clone())
        .collect()
}

/// Explore the whole tree: the root examined on one engine, then its
/// children split across workers (each with its own engine and its own
/// share of the branch cap), the partial results returned in exploration
/// order (the root's own first).
pub(crate) fn drive<S: Strategy>(
    model: &CompiledModel,
    common: &Common<'_>,
    strategy: &S,
    config: &EngineConfig,
    root_shared: &S::Shared,
    merge: bool,
) -> Result<Vec<Partial>, EngineError> {
    use rayon::prelude::*;

    let mut root = Walker::new(
        model,
        common,
        strategy,
        Engine::new(model, config.clone())?,
        common.cutoffs.max_branches,
        config.max_fixpoint_iterations,
        merge,
    );
    let (children, shared, child_streak) = match root.examine(1.0, root_shared, None, 0)? {
        Node::Leaf => return Ok(vec![root.out]),
        Node::Expand {
            children,
            shared,
            child_streak,
            ..
        } => (children, shared, child_streak),
    };
    let root_snapshot = root.engine.snapshot();
    let shares = split_budget(
        common
            .cutoffs
            .max_branches
            .map(|cap| cap.saturating_sub(root.out.expanded)),
        children.len(),
    );

    let explore_child = |k: usize| -> Result<Partial, EngineError> {
        let mut walker = Walker::new(
            model,
            common,
            strategy,
            Engine::new(model, config.clone())?,
            shares[k],
            config.max_fixpoint_iterations,
            merge,
        );
        walker.engine.restore(&root_snapshot);
        walker.explore_below(1.0, 0, child_streak, &shared, children[k])?;
        Ok(walker.out)
    };
    let compute = || -> Vec<Result<Partial, EngineError>> {
        (0..children.len())
            .into_par_iter()
            .map(explore_child)
            .collect()
    };
    let outcomes = match common.threads {
        None => compute(),
        Some(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| invalid("threads", format!("thread-pool construction failed: {e}")))?
            .install(compute),
    };

    // Reduction in child order: the first error in exploration order wins.
    let mut partials = vec![root.out];
    for outcome in outcomes {
        partials.push(outcome?);
    }
    Ok(partials)
}

/// The engine configuration of an exploration: no journal, no samples, no
/// trace (the explorer builds its own), the target latch on.
pub(crate) fn engine_config() -> EngineConfig {
    EngineConfig {
        journal: false,
        sequences: false,
        stop_at_targets: true,
        samples: Vec::new(),
        ..EngineConfig::default()
    }
}

pub(crate) fn invalid(parameter: &str, detail: String) -> EngineError {
    EngineError::InvalidStudyParameter {
        parameter: parameter.to_owned(),
        detail,
    }
}

/// Up-front validation of the settings every driver shares.
pub(crate) fn validate_common(
    model: &CompiledModel,
    common: &Common<'_>,
) -> Result<(), EngineError> {
    if !model.targets.iter().any(|t| t.name == common.target) {
        let known: Vec<&str> = model.targets.iter().map(|t| t.name.as_str()).collect();
        return Err(invalid(
            "target",
            format!(
                "no target named `{}` in the model (declared: [{}])",
                common.target,
                known.join(", ")
            ),
        ));
    }
    if !common.horizon.is_finite() || common.horizon < 0.0 {
        return Err(invalid(
            "horizon",
            format!(
                "a horizon is a finite nonnegative time, got {}",
                common.horizon
            ),
        ));
    }
    let cutoffs = common.cutoffs;
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
    if common.gap_tolerance.is_nan() || !(0.0..=1.0).contains(&common.gap_tolerance) {
        return Err(invalid(
            "gap_tolerance",
            format!(
                "a relative gap tolerance lies in [0, 1], got {}",
                common.gap_tolerance
            ),
        ));
    }
    if common.threads == Some(0) {
        return Err(invalid(
            "threads",
            "at least one thread is needed, got 0".to_owned(),
        ));
    }
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

/// Add `sequence` to `raw`, merged into the earlier sequence of the same
/// path when there is one (probability and error bound summed, the
/// imprecision flags or-ed), appended otherwise; `index` maps each path of
/// `raw` to its position.
fn merge_by_path(
    raw: &mut Vec<RawSequence>,
    index: &mut HashMap<Vec<(u32, u32)>, usize>,
    sequence: RawSequence,
) {
    if let Some(&k) = index.get(&sequence.path) {
        let kept = &mut raw[k];
        kept.probability += sequence.probability;
        kept.error_bound += sequence.error_bound;
        kept.imprecise |= sequence.imprecise;
    } else {
        index.insert(sequence.path.clone(), raw.len());
        raw.push(sequence);
    }
}

/// Final assembly: partial results summed in the given (exploration)
/// order, sequences sharing a path merged when `merge` (probabilities
/// summed in exploration order), the step table built from the distinct
/// steps of the retained paths, then sequences ranked.
pub(crate) fn assemble(
    model: &CompiledModel,
    common: &Common<'_>,
    algorithm: Algorithm,
    precision: &Precision,
    partials: Vec<Partial>,
    merge: bool,
) -> ExplorationResult {
    let mut raw: Vec<RawSequence> = Vec::new();
    let mut index: HashMap<Vec<(u32, u32)>, usize> = HashMap::new();
    let mut lower = 0.0;
    let mut tallies = CutoffTallies::default();
    let mut expanded = 0;
    for partial in partials {
        lower += partial.lower;
        tallies.merge(&partial.tallies);
        expanded += partial.expanded;
        if merge {
            for sequence in partial.sequences {
                merge_by_path(&mut raw, &mut index, sequence);
            }
        } else {
            raw.extend(partial.sequences);
        }
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
            end_cause: common.target.to_owned(),
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
        version: algorithm.format_version(),
        engine_version: env!("CARGO_PKG_VERSION").to_owned(),
        model: model.name.clone(),
        algorithm,
        target: common.target.to_owned(),
        horizon: common.horizon,
        cutoffs: common.cutoffs.clone(),
        gap_tolerance: common.gap_tolerance,
        precision: precision.clone(),
        inconclusive: relative_gap(lower, upper) > common.gap_tolerance,
        steps,
        sequences,
        lower,
        upper,
        cutoff_tallies: tallies,
        expanded_nodes: expanded,
        imprecise_sequences,
        discretisation: None,
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

/// The branches of an instantaneous firing of `idx`: one per destination
/// of nonzero probability for an instantaneous branching, the single
/// destination otherwise, as `(branch, probability)`.
pub(crate) fn instantaneous_branches(model: &CompiledModel, idx: usize) -> Vec<(usize, f64)> {
    match &model.transitions[idx].distrib {
        CLaw::Inst(probs) => probs
            .iter()
            .enumerate()
            .filter(|(_, p)| **p > 0.0)
            .map(|(branch, p)| (branch, *p))
            .collect(),
        _ => vec![(0, 1.0)],
    }
}
