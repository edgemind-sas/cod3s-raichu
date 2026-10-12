//! Static decision, per cycle group, on whether the minimal-sequence
//! reduction may cancel it.
//!
//! The reduction drops a failure repaired before the feared event, on the
//! hypothesis that a repaired component no longer contributes. The
//! hypothesis fails when the failure's effect is remembered: a storage
//! that drains while a producer is down stays low after the repair, and
//! the feared event can come hours later with every failure repaired.
//!
//! This module decides it from the compiled tables, before any run. A
//! group is **not cancellable** for a target when the effect of the group
//! (the state of its automata, or a value its transitions write) reaches
//! the guards or laws of the target's automaton through a **memory node**:
//! a quantity whose value outlives the change that set it.
//!
//! - an ODE target, which integrates what it reads;
//! - an FMU output, whose state the engine cannot see;
//! - an attribute a transition effect writes, unless the group both sets
//!   it (failure) and restores it (repair, to its initial value), so that
//!   its value is a function of the group's state;
//! - an automaton with a timer that survives an interruption
//!   (`on_interruption: resume | continue`);
//! - any other automaton the search goes through, because a band
//!   controller, a protection trip or a latched standby holds the
//!   failure's effect after the repair. Four shapes are exempt: the
//!   group's own automata; an automaton of another cycle group with
//!   monitored transitions, whose persistence the sequence records as its
//!   own events; an automaton echoed by recorded followers, whose state
//!   reaches the rest of the model only through recorded automata that
//!   copy it at once (two states, left instantly and without effect on
//!   complementary guards), so that its state is at every instant the
//!   followers' recorded state (the occ/rep automaton of a muscadet
//!   `external` failure mode and the mirror it drives on each target);
//!   and a memoryless automaton, two states left and entered instantly on
//!   complementary guards, whose state is a function of its current
//!   inputs.
//!
//! The search never goes through a target automaton that has a single
//! non-target state: until the target is reached, that automaton sits in
//! its one other state, constant, and so carries nothing.
//!
//! What it does not see: a latch or counter written by a sensitive
//! function (level semantics), which `cycle_cancellation: never` covers.
//! The echo exemption narrows that blind spot rather than widening it: a
//! variable between an echoed automaton and its follower may not read the
//! follower, the one shape in which such a function would latch it.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use raichu_expr::{BoolOp, CmpOp, Value};
use raichu_model::{CycleCancellation, InterruptionPolicy, TransitionKind};
use serde::{Deserialize, Serialize};

use crate::compile::{CExpr, CLaw, CTransition, CompiledModel};
use crate::dependency::{self, EdgeSet, Node};

/// Who decided whether a cycle group may be cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionOrigin {
    /// The static analysis of the compiled model.
    Analysis,
    /// The model's [`raichu_model::Transition::cycle_cancellation`].
    Declared,
}

/// The decision for one cycle group and one target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetDecision {
    /// The target (feared event) name.
    pub target: String,
    /// Whether the reduction may cancel the group in a trajectory that
    /// ends on this target.
    pub cancellable: bool,
    /// Who decided.
    pub origin: DecisionOrigin,
    /// When the analysis found the group's effect reaching the target
    /// through a memory node: the path from the group to the target's
    /// automaton, by qualified name. Empty otherwise.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub witness: Vec<String>,
    /// The first memory node on [`TargetDecision::witness`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<String>,
}

/// The decision for one cycle group: one per target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CycleGroupDecision {
    /// The component, as sequence events name it (`obj`).
    pub component: String,
    /// The cycle group, as sequence events carry it.
    pub cycle_group: String,
    /// The automata whose transitions belong to the group, by qualified
    /// name.
    pub automata: Vec<String>,
    /// One decision per target, in the model's target order.
    pub targets: Vec<TargetDecision>,
}

impl CycleGroupDecision {
    /// Whether the group may be cancelled in a trajectory that ended on
    /// `end_cause`.
    ///
    /// A trajectory that reached no target (`None`) cancels every group:
    /// there is no feared event whose cause its history could explain, and
    /// keeping that history would make nearly every such trajectory a
    /// distinct path, so the reduction's memory would grow with the
    /// campaign (measured: 7 against 4 436 cleaned paths on the heated
    /// room at 10 000 replicas). A target this decision does not name is
    /// cancelled only when the group is cancellable for every target.
    #[must_use]
    pub fn cancellable_for(&self, end_cause: Option<&str>) -> bool {
        let Some(cause) = end_cause else {
            return true;
        };
        match self.targets.iter().find(|d| d.target == cause) {
            Some(decision) => decision.cancellable,
            None => self.targets.iter().all(|d| d.cancellable),
        }
    }

    /// One line per target the group is kept for, for a log or a report.
    #[must_use]
    pub fn describe(&self) -> Vec<String> {
        self.targets
            .iter()
            .filter(|d| !d.cancellable)
            .map(|d| match d.origin {
                DecisionOrigin::Declared => format!(
                    "cycle group `{}` of `{}` is not cancelled for `{}`: declared \
                     `cycle_cancellation: never`",
                    self.cycle_group, self.component, d.target
                ),
                DecisionOrigin::Analysis => format!(
                    "cycle group `{}` of `{}` is not cancelled for `{}`: its effect \
                     reaches the feared event through `{}`, which remembers it ({})",
                    self.cycle_group,
                    self.component,
                    d.target,
                    d.memory.as_deref().unwrap_or("?"),
                    d.witness.join(" -> ")
                ),
            })
            .collect()
    }
}

/// The decision for every cycle group of `model`, ordered by component
/// then group name. A model without cycle groups, or without targets,
/// gives an empty list.
#[must_use]
pub fn cycle_groups(model: &CompiledModel) -> Vec<CycleGroupDecision> {
    let groups = collect_groups(model);
    if groups.is_empty() || model.targets.is_empty() {
        return Vec::new();
    }
    let graph = dependency::build(model, EdgeSet::Influences);
    // Edges reversed: who reads each node, sorted for a deterministic walk.
    let mut readers: HashMap<Node, Vec<Node>> = HashMap::new();
    for (&reader, reads) in &graph {
        for &read in reads {
            readers.entry(read).or_default().push(reader);
        }
    }
    for list in readers.values_mut() {
        list.sort_unstable();
        list.dedup();
    }
    let facts = Facts::new(model, &readers);

    groups
        .into_iter()
        .map(|((component, cycle_group), group)| {
            // Several targets on one automaton share one search.
            let mut searched: HashMap<usize, Option<(Vec<String>, String)>> = HashMap::new();
            let targets = model
                .targets
                .iter()
                .map(|target| {
                    let found = searched
                        .entry(target.automaton)
                        .or_insert_with(|| facts.search(model, &readers, &group, target.automaton))
                        .clone();
                    let (cancellable, origin) = match group.declared {
                        CycleCancellation::Never => (false, DecisionOrigin::Declared),
                        CycleCancellation::Always => (true, DecisionOrigin::Declared),
                        CycleCancellation::Auto => (found.is_none(), DecisionOrigin::Analysis),
                    };
                    let (witness, memory) = found.unwrap_or_default();
                    TargetDecision {
                        target: target.name.clone(),
                        cancellable,
                        origin,
                        witness,
                        memory: (!memory.is_empty()).then_some(memory),
                    }
                })
                .collect();
            CycleGroupDecision {
                automata: group
                    .automata
                    .iter()
                    .map(|&a| model.automata[a].name.clone())
                    .collect(),
                component,
                cycle_group,
                targets,
            }
        })
        .collect()
}

/// A search state: a node, and whether a memory node lies behind it.
type State = (Node, bool);

/// What the search starts from for one `(component, cycle_group)`.
struct Group {
    automata: BTreeSet<usize>,
    written: BTreeSet<usize>,
    declared: CycleCancellation,
}

fn collect_groups(model: &CompiledModel) -> BTreeMap<(String, String), Group> {
    let mut groups: BTreeMap<(String, String), Group> = BTreeMap::new();
    for transition in &model.transitions {
        let Some(name) = &transition.cycle_group else {
            continue;
        };
        let group = groups
            .entry((transition.component.clone(), name.clone()))
            .or_insert_with(|| Group {
                automata: BTreeSet::new(),
                written: BTreeSet::new(),
                declared: transition.cycle_cancellation,
            });
        group.automata.insert(transition.automaton);
        group
            .written
            .extend(transition.effects.iter().map(|(target, _)| *target));
    }
    groups
}

/// What the search needs to know about every node, computed once.
struct Facts {
    /// Variables whose value outlives the change that set it.
    memory_vars: HashSet<usize>,
    /// Automata with a timer that survives an interruption.
    timed: HashSet<usize>,
    /// Automata of a cycle group with monitored transitions, and the
    /// automata recorded followers echo (see [`echoed`]).
    recorded: HashSet<usize>,
    /// Automata whose state is a function of their current inputs.
    memoryless: HashSet<usize>,
    /// Target automata with a single non-target state, never traversed.
    constant_targets: HashSet<usize>,
}

impl Facts {
    fn new(model: &CompiledModel, readers: &HashMap<Node, Vec<Node>>) -> Self {
        let mut memory_vars: HashSet<usize> = model.ode.iter().map(|(target, _)| *target).collect();
        for unit in &model.fmu_units {
            memory_vars.extend(
                unit.outputs
                    .iter()
                    .filter_map(|b| dependency::fmu_var(model, b)),
            );
        }
        // Writers of every attribute a transition effect writes.
        let mut writers: BTreeMap<usize, Vec<(usize, &CExpr)>> = BTreeMap::new();
        for (index, transition) in model.transitions.iter().enumerate() {
            for (target, expr) in &transition.effects {
                writers.entry(*target).or_default().push((index, expr));
            }
        }
        for (var, list) in writers {
            if !restored_latch(model, var, &list) {
                memory_vars.insert(var);
            }
        }

        let timed = model
            .transitions
            .iter()
            .filter(|t| {
                matches!(
                    t.on_interruption,
                    InterruptionPolicy::Resume | InterruptionPolicy::Continue
                )
            })
            .map(|t| t.automaton)
            .collect();
        let mut by_automaton: Vec<Vec<&CTransition>> = vec![Vec::new(); model.automata.len()];
        for transition in &model.transitions {
            by_automaton[transition.automaton].push(transition);
        }
        // Exempt only when every transition is a monitored member of a
        // cycle group: one unrecorded trip is enough to hold an effect the
        // sequence never shows. A draw that records its won branch only
        // (`monitored_states`) parks on a loss until the solicitation
        // falls, which follows the solicitation and remembers nothing.
        let mut recorded: HashSet<usize> = (0..model.automata.len())
            .filter(|&a| {
                !by_automaton[a].is_empty()
                    && by_automaton[a]
                        .iter()
                        .all(|t| t.monitored && t.cycle_group.is_some())
            })
            .collect();
        // A recorded automaton that copies its input at once: what an
        // echoed automaton may be read by.
        let followers: HashSet<usize> = recorded
            .iter()
            .copied()
            .filter(|&a| {
                two_state_switch(model, a, &by_automaton[a])
                    .is_some_and(|(g1, g2)| complementary(g1, g2))
            })
            .collect();
        let echoed: Vec<usize> = (0..model.automata.len())
            .filter(|&a| !recorded.contains(&a) && echoed(a, readers, &followers, &memory_vars))
            .collect();
        recorded.extend(echoed);
        let memoryless = (0..model.automata.len())
            .filter(|&a| is_memoryless(model, a, &by_automaton[a]))
            .collect();

        let mut target_states: HashMap<usize, HashSet<usize>> = HashMap::new();
        for target in &model.targets {
            target_states
                .entry(target.automaton)
                .or_default()
                .insert(target.state);
        }
        let constant_targets = target_states
            .into_iter()
            .filter(|(automaton, states)| {
                model.automata[*automaton].states.len() - states.len() == 1
            })
            .map(|(automaton, _)| automaton)
            .collect();

        Facts {
            memory_vars,
            timed,
            recorded,
            memoryless,
            constant_targets,
        }
    }

    /// Whether entering `node` passes through a memory node, for a search
    /// started from the group whose automata are `own`.
    fn remembers(&self, node: Node, own: &BTreeSet<usize>) -> bool {
        match node {
            Node::Var(v) => self.memory_vars.contains(&v),
            Node::Aut(a) => {
                if own.contains(&a) {
                    false
                } else if self.timed.contains(&a) {
                    true
                } else {
                    !(self.recorded.contains(&a) || self.memoryless.contains(&a))
                }
            }
        }
    }

    /// The shortest path from `group` to the guards or laws of `target`
    /// that passes through a memory node, as qualified names, with the
    /// first memory node's name; `None` when there is none.
    fn search(
        &self,
        model: &CompiledModel,
        readers: &HashMap<Node, Vec<Node>>,
        group: &Group,
        target: usize,
    ) -> Option<(Vec<String>, String)> {
        // Breadth-first over sorted reader lists, so the path found is the
        // shortest and, among those, the first in index order:
        // deterministic.
        let mut parent: HashMap<State, Option<State>> = HashMap::new();
        let mut queue: VecDeque<State> = VecDeque::new();
        let starts = group.automata.iter().map(|&a| (Node::Aut(a), false)).chain(
            group
                .written
                .iter()
                .map(|&v| (Node::Var(v), self.memory_vars.contains(&v))),
        );
        for state in starts {
            if parent.insert(state, None).is_none() {
                queue.push_back(state);
            }
        }
        // Reaching the target's own automaton is reaching memory unless
        // that automaton is constant before the target or memoryless: an
        // intermediate state (degraded on A's failure, lost on B's) holds
        // A's effect after A's repair, with no other automaton in between.
        let target_remembers = self.timed.contains(&target)
            || !(self.constant_targets.contains(&target) || self.memoryless.contains(&target));
        while let Some(state @ (node, behind)) = queue.pop_front() {
            for &next in readers.get(&node).map(Vec::as_slice).unwrap_or(&[]) {
                if next == Node::Aut(target) && (behind || target_remembers) {
                    return Some(self.witness(model, &parent, state, next, &group.automata));
                }
                // Reached without memory, the target's own automaton is
                // gone through like any other, unless it is constant.
                if let Node::Aut(a) = next {
                    if self.constant_targets.contains(&a) {
                        continue;
                    }
                }
                let after = (next, behind || self.remembers(next, &group.automata));
                if let Entry::Vacant(slot) = parent.entry(after) {
                    slot.insert(Some(state));
                    queue.push_back(after);
                }
            }
        }
        None
    }

    fn witness(
        &self,
        model: &CompiledModel,
        parent: &HashMap<State, Option<State>>,
        last: State,
        target: Node,
        own: &BTreeSet<usize>,
    ) -> (Vec<String>, String) {
        let mut chain = vec![target];
        let mut cursor = Some(last);
        while let Some(state) = cursor {
            chain.push(state.0);
            cursor = parent.get(&state).copied().flatten();
        }
        chain.reverse();
        let name = |node: Node| match node {
            Node::Var(v) => model.var_names[v].clone(),
            Node::Aut(a) => model.automata[a].name.clone(),
        };
        let memory = chain
            .iter()
            .copied()
            .find(|&node| self.remembers(node, own) || node == target)
            .map(name)
            .unwrap_or_default();
        (chain.into_iter().map(name).collect(), memory)
    }
}

/// Whether automaton `a` reaches the rest of the model only through
/// recorded followers: every reader of its state, followed through
/// variables that remember nothing, is a recorded automaton that copies
/// what it reads at once (see `followers` in [`Facts::new`]), and there is
/// at least one. The state of `a` is then, at every instant, what the
/// followers' recorded events show: the occ/rep automaton of a muscadet
/// `external` failure mode and the mirrors it drives through a control
/// attribute.
///
/// A follower is required, not any recorded reader: a reader whose
/// transition waits (a rate, a delay) or reads `a` in a law fires later,
/// after the repair that a cancelled cycle hides, and nothing would show
/// that `a` held the effect meanwhile. A variable on the way may not read
/// the follower it leads to: through it the follower would latch itself.
///
/// A follower is not walked past: what lies behind it is reached from it,
/// and it is its own events that the sequence keeps.
fn echoed(
    a: usize,
    readers: &HashMap<Node, Vec<Node>>,
    followers: &HashSet<usize>,
    memory_vars: &HashSet<usize>,
) -> bool {
    let mut seen: HashSet<Node> = HashSet::from([Node::Aut(a)]);
    let mut queue: VecDeque<Node> = VecDeque::from([Node::Aut(a)]);
    let mut vars: HashSet<Node> = HashSet::new();
    let mut reached: BTreeSet<usize> = BTreeSet::new();
    while let Some(node) = queue.pop_front() {
        for &next in readers.get(&node).map(Vec::as_slice).unwrap_or(&[]) {
            match next {
                Node::Aut(b) if b == a => {}
                Node::Aut(b) if followers.contains(&b) => {
                    reached.insert(b);
                }
                Node::Aut(_) => return false,
                Node::Var(v) if memory_vars.contains(&v) => return false,
                Node::Var(_) => {
                    if seen.insert(next) {
                        vars.insert(next);
                        queue.push_back(next);
                    }
                }
            }
        }
    }
    // No variable on the way reads a follower it leads to.
    !reached.is_empty()
        && reached.iter().all(|&b| {
            readers
                .get(&Node::Aut(b))
                .is_none_or(|list| list.iter().all(|reader| !vars.contains(reader)))
        })
}

/// Whether every write of `var` by a transition effect comes from one
/// cycle group of a single automaton that sets it on failure and restores
/// its initial value on **every** repair, constants all: its value is then
/// a function of the group's state, and remembers nothing the group's own
/// events do not.
fn restored_latch(model: &CompiledModel, var: usize, writers: &[(usize, &CExpr)]) -> bool {
    let mut owner: Option<(&str, &str, usize)> = None;
    for &(index, expr) in writers {
        let transition = &model.transitions[index];
        let Some(name) = transition.cycle_group.as_deref() else {
            return false;
        };
        let key = (transition.component.as_str(), name, transition.automaton);
        if *owner.get_or_insert(key) != key {
            return false;
        }
        let CExpr::Const(value) = expr else {
            return false;
        };
        match transition.kind {
            Some(TransitionKind::Failure) => {}
            Some(TransitionKind::Repair) if same_value(value, &model.var_init[var]) => {}
            _ => return false,
        }
    }
    let Some((component, group, automaton)) = owner else {
        return false;
    };
    // Every repair of the group writes the initial value back: a repair
    // that leaves the flag set would keep the failure's effect.
    let mut repairs = model.transitions.iter().filter(|t| {
        t.component == component
            && t.cycle_group.as_deref() == Some(group)
            && t.kind == Some(TransitionKind::Repair)
    });
    let mut any = false;
    let all_restore = repairs.all(|t| {
        any = true;
        t.automaton == automaton && t.effects.iter().any(|(target, expr)| {
            *target == var
                && matches!(expr, CExpr::Const(value) if same_value(value, &model.var_init[var]))
        })
    });
    // A single automaton: two of them in one group pair their events
    // across automata, which the per-group view cannot follow.
    let single = model.transitions.iter().all(|t| {
        !(t.component == component && t.cycle_group.as_deref() == Some(group))
            || t.automaton == automaton
    });
    any && all_restore && single
}

fn same_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

/// The two guards of automaton `a` when it is a **switch**: two states,
/// each left by one instantaneous transition without effects towards the
/// other, on guards that do not read the automaton's own state (a guard
/// reading it can latch it: once on, `G or on` never turns false).
fn two_state_switch<'a>(
    model: &CompiledModel,
    a: usize,
    transitions: &[&'a CTransition],
) -> Option<(&'a CExpr, &'a CExpr)> {
    if model.automata[a].states.len() != 2 {
        return None;
    }
    let [first, second] = transitions else {
        return None;
    };
    let instant = |t: &&CTransition| {
        t.effects.is_empty()
            && t.targets.len() == 1
            && match &t.distrib {
                CLaw::Delay(delay) => *delay == 0.0,
                CLaw::Inst(probs) => probs.len() == 1,
                CLaw::Watched { .. } => true,
                _ => false,
            }
    };
    if first.source == second.source || !instant(first) || !instant(second) {
        return None;
    }
    let (Some(g1), Some(g2)) = (&first.guard, &second.guard) else {
        return None;
    };
    if dependency::sensitivity(g1).1.contains(&a) || dependency::sensitivity(g2).1.contains(&a) {
        return None;
    }
    Some((g1, g2))
}

/// Whether automaton `a` is a switch on a condition and its syntactic
/// negation: its state is then a function of what the guards read, and
/// remembers nothing.
fn is_memoryless(model: &CompiledModel, a: usize, transitions: &[&CTransition]) -> bool {
    two_state_switch(model, a, transitions)
        .is_some_and(|(g1, g2)| negates(g1, g2) || negates(g2, g1))
}

fn negates(a: &CExpr, b: &CExpr) -> bool {
    matches!(b, CExpr::Bool { op: BoolOp::Not, args } if args.len() == 1 && args[0] == *a)
}

/// Whether two guards are each other's negation: syntactically (`G`,
/// `not G`), or as the equality of one operand with two opposite boolean
/// constants (`x == true`, `x == false`), the guards of the mirror a
/// muscadet `external` failure mode grafts on its targets.
fn complementary(g1: &CExpr, g2: &CExpr) -> bool {
    if negates(g1, g2) || negates(g2, g1) {
        return true;
    }
    let bool_eq = |g: &CExpr| match g {
        CExpr::Cmp {
            op: CmpOp::Eq,
            lhs,
            rhs,
        } => match (lhs.as_ref(), rhs.as_ref()) {
            (operand, CExpr::Const(Value::Bool(b))) | (CExpr::Const(Value::Bool(b)), operand) => {
                Some((operand.clone(), *b))
            }
            _ => None,
        },
        _ => None,
    };
    matches!((bool_eq(g1), bool_eq(g2)), (Some((x, p)), Some((y, q))) if x == y && p != q)
}
