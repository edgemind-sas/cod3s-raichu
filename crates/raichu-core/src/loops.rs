//! Static detection of a **switching loop**: an automaton whose guard
//! reads a quantity that its own decision moves.
//!
//! The shape is a cycle in the dependency graph, automaton to variable
//! to automaton, and the reason it matters is that each of the two
//! states then produces the condition that justifies the other. The mode
//! has no fixpoint, and what sets its period is not the physics but the
//! width of the narrowest threshold on the cycle. Where that width is
//! zero, the period collapses to the numerical hysteresis of a boundary
//! crossing: the run does not fail, it grinds, and the trajectory reads
//! correctly at every sample instant while it does.
//!
//! `max_transition_firings` and `max_flow_restarts` catch that at run
//! time, after the fact and after the wait. This module catches the same
//! thing **before any simulation**, from the compiled tables alone.
//!
//! It **warns and never refuses**, because the loop is not the fault. A
//! thermostat is a switching loop, and so is every controlled tank; what
//! makes one pathological is a switch with no band. So a loop is reported
//! only when at least one automaton on it switches on a single threshold,
//! and a loop whose every switch has a band is silent.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::compile::{CAllocation, CExpr, CStep, CompiledModel};
use raichu_expr::Value;

/// One switching loop found in a compiled model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchingLoop {
    /// The automata on the cycle, by qualified name, in index order.
    pub automata: Vec<String>,
    /// Those of them that switch on a **single threshold**, which is
    /// what turns the loop from a legitimate oscillation into one whose
    /// period is numerical. Never empty: a loop where every switch has a
    /// band is not reported.
    pub bandless: Vec<String>,
    /// The variables the cycle passes through, by qualified name, in
    /// index order. What makes the report actionable: it names the path,
    /// not just its ends.
    pub through: Vec<String>,
}

impl SwitchingLoop {
    /// The one-line diagnostic, phrased for someone who has just watched
    /// a run take too long.
    ///
    /// It leads with what to change rather than with the whole cycle: a
    /// loop through a flow network passes through dozens of attributes,
    /// and a message that listed them all would bury the one automaton
    /// the reader can act on.
    pub fn describe(&self) -> String {
        format!(
            "switching loop: {} {} on a single threshold a value that its \
             own decision moves, {} attributes and {} automata around; the \
             period of that loop is set by numerical hysteresis and not by \
             the model. Give the threshold a band, or read a quantity the \
             rule does not move.",
            self.bandless.join(", "),
            if self.bandless.len() > 1 {
                "switch"
            } else {
                "switches"
            },
            self.through.len(),
            self.automata.len(),
        )
    }
}

/// A node of the dependency graph: the graph is bipartite in kind, and
/// mixing both in one index space is what gives the walk its
/// transitivity for nothing.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Node {
    Var(usize),
    Aut(usize),
}

/// Every switching loop of `model`, in a deterministic order.
///
/// Linear in the size of the tables: one pass to build the graph, one
/// Tarjan pass over it. Called once at compile time.
pub fn switching_loops(model: &CompiledModel) -> Vec<SwitchingLoop> {
    let graph = build_graph(model);
    let mut found = Vec::new();
    for component in strongly_connected(&graph, model) {
        let mut automata = BTreeSet::new();
        let mut through = BTreeSet::new();
        for node in &component {
            match node {
                Node::Aut(index) => {
                    automata.insert(*index);
                }
                Node::Var(index) => {
                    through.insert(*index);
                }
            }
        }
        if automata.is_empty() {
            // A cycle among variables alone is an algebraic loop, which
            // the flow resolution owns and diagnoses on its own terms.
            continue;
        }
        let bandless: Vec<usize> = automata
            .iter()
            .copied()
            .filter(|&index| switches_without_a_band(model, index))
            .collect();
        if bandless.is_empty() {
            // Every switch on the cycle has a band, so crossing it costs
            // physical time and the loop is a legitimate oscillation.
            continue;
        }
        found.push(SwitchingLoop {
            automata: automata
                .iter()
                .map(|&i| model.automata[i].name.clone())
                .collect(),
            bandless: bandless
                .iter()
                .map(|&i| model.automata[i].name.clone())
                .collect(),
            through: through
                .iter()
                .map(|&i| model.var_names[i].clone())
                .collect(),
        });
    }
    found
}

/// Successors of every node: an automaton leads to what its guards read,
/// a variable to what its defining expression reads.
fn build_graph(model: &CompiledModel) -> HashMap<Node, Vec<Node>> {
    let mut graph: HashMap<Node, Vec<Node>> = HashMap::new();

    for transition in &model.transitions {
        let Some(guard) = &transition.guard else {
            continue;
        };
        let (vars, auts) = sensitivity(guard);
        let entry = graph.entry(Node::Aut(transition.automaton)).or_default();
        entry.extend(vars.into_iter().map(Node::Var));
        // A guard reading another automaton's STATE is an edge of its
        // own: the value it decides on need not pass through a variable.
        entry.extend(auts.into_iter().map(Node::Aut));
    }

    let written = |target: usize, expr: &CExpr, graph: &mut HashMap<Node, Vec<Node>>| {
        let (vars, auts) = sensitivity(expr);
        let entry = graph.entry(Node::Var(target)).or_default();
        entry.extend(vars.into_iter().map(Node::Var));
        entry.extend(auts.into_iter().map(Node::Aut));
    };

    for step in &model.explicit {
        match step {
            CStep::Equation { target, expr } => written(*target, expr, &mut graph),
            CStep::Allocate(allocation) => allocate_edges(allocation, &mut graph),
        }
    }
    for (target, expr) in &model.ode {
        written(*target, expr, &mut graph);
    }
    for function in &model.functions {
        for (target, expr) in &function.effects {
            written(*target, expr, &mut graph);
        }
    }

    for successors in graph.values_mut() {
        successors.sort_by_key(|node| match node {
            Node::Var(index) => (0, *index),
            Node::Aut(index) => (1, *index),
        });
        successors.dedup_by_key(|node| match node {
            Node::Var(index) => (0, *index),
            Node::Aut(index) => (1, *index),
        });
    }
    graph
}

/// What a distribution operator writes depends on what it has to give
/// and on every demand competing for it, so each allocated attribute
/// leads to both.
fn allocate_edges(allocation: &CAllocation, graph: &mut HashMap<Node, Vec<Node>>) {
    let (vars, auts) = sensitivity(&allocation.available);
    for &target in &allocation.allocated {
        let entry = graph.entry(Node::Var(target)).or_default();
        entry.extend(vars.iter().copied().map(Node::Var));
        entry.extend(auts.iter().copied().map(Node::Aut));
        entry.extend(allocation.demands.iter().copied().map(Node::Var));
    }
}

/// The variables and automata one expression reads.
fn sensitivity(expr: &CExpr) -> (Vec<usize>, Vec<usize>) {
    let mut vars = Vec::new();
    let mut auts = Vec::new();
    expr.collect_sensitivity(&mut vars, &mut auts);
    vars.sort_unstable();
    vars.dedup();
    auts.sort_unstable();
    auts.dedup();
    (vars, auts)
}

/// Whether the automaton is entered and left at the **same** threshold on
/// some variable, which is a switch with no band.
///
/// Read off the guards rather than from any declaration: a band is
/// visible in the compiled tables as two comparisons on one variable at
/// two different constants, whatever authoring layer produced them.
fn switches_without_a_band(model: &CompiledModel, automaton: usize) -> bool {
    // Thresholds compared against each variable, per source state.
    let mut per_state: HashMap<usize, HashMap<usize, BTreeSet<u64>>> = HashMap::new();
    for transition in &model.transitions {
        if transition.automaton != automaton {
            continue;
        }
        let Some(guard) = &transition.guard else {
            continue;
        };
        let mut found = HashMap::new();
        thresholds(guard, &mut found);
        let entry = per_state.entry(transition.source).or_default();
        for (var, values) in found {
            entry.entry(var).or_default().extend(values);
        }
    }
    if per_state.len() < 2 {
        // One source state can carry no round trip.
        return false;
    }
    // A variable compared to exactly one constant from every state that
    // compares it at all is a switch with no band: whatever the mode, it
    // flips at that same point.
    let mut compared: HashMap<usize, HashSet<u64>> = HashMap::new();
    let mut states_comparing: HashMap<usize, usize> = HashMap::new();
    for values in per_state.values() {
        for (var, thresholds) in values {
            compared.entry(*var).or_default().extend(thresholds.iter());
            *states_comparing.entry(*var).or_default() += 1;
        }
    }
    compared.iter().any(|(var, values)| {
        values.len() == 1 && states_comparing.get(var).copied().unwrap_or(0) >= 2
    })
}

/// Every `variable <op> constant` comparison of a guard, as the bit
/// pattern of the constant so it can be compared exactly.
fn thresholds(expr: &CExpr, found: &mut HashMap<usize, BTreeSet<u64>>) {
    match expr {
        CExpr::Cmp { lhs, rhs, .. } => {
            match (lhs.as_ref(), rhs.as_ref()) {
                (CExpr::Var(index), CExpr::Const(value))
                | (CExpr::Const(value), CExpr::Var(index)) => {
                    if let Some(bits) = numeric_bits(value) {
                        found.entry(*index).or_default().insert(bits);
                    }
                }
                _ => {}
            }
            thresholds(lhs, found);
            thresholds(rhs, found);
        }
        CExpr::Bool { args, .. } | CExpr::Add { args } => {
            for arg in args {
                thresholds(arg, found);
            }
        }
        CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
            thresholds(lhs, found);
            thresholds(rhs, found);
        }
        CExpr::Mul { args } | CExpr::Min { args } | CExpr::Max { args } => {
            for arg in args {
                thresholds(arg, found);
            }
        }
        CExpr::If {
            cond,
            then,
            otherwise,
        } => {
            thresholds(cond, found);
            thresholds(then, found);
            thresholds(otherwise, found);
        }
        _ => {}
    }
}

/// The bit pattern of a numeric constant, or `None` for one that is not
/// a threshold.
fn numeric_bits(value: &Value) -> Option<u64> {
    match value {
        Value::Float(x) => Some(x.to_bits()),
        Value::Int(n) => Some((*n as f64).to_bits()),
        _ => None,
    }
}

/// Tarjan's strongly connected components, iterative so a deep model
/// cannot blow the stack.
fn strongly_connected(graph: &HashMap<Node, Vec<Node>>, model: &CompiledModel) -> Vec<Vec<Node>> {
    let mut nodes: Vec<Node> = (0..model.var_names.len()).map(Node::Var).collect();
    nodes.extend((0..model.automata.len()).map(Node::Aut));

    let mut index_of: HashMap<Node, usize> = HashMap::new();
    let mut low: HashMap<Node, usize> = HashMap::new();
    let mut on_stack: HashSet<Node> = HashSet::new();
    let mut stack: Vec<Node> = Vec::new();
    let mut next_index = 0usize;
    let mut components = Vec::new();

    for &root in &nodes {
        if index_of.contains_key(&root) {
            continue;
        }
        // (node, position in its successor list)
        let mut call: Vec<(Node, usize)> = vec![(root, 0)];
        index_of.insert(root, next_index);
        low.insert(root, next_index);
        next_index += 1;
        stack.push(root);
        on_stack.insert(root);

        while let Some((node, position)) = call.pop() {
            let successors = graph.get(&node).map(Vec::as_slice).unwrap_or(&[]);
            if position < successors.len() {
                let next = successors[position];
                call.push((node, position + 1));
                if let std::collections::hash_map::Entry::Vacant(slot) = index_of.entry(next) {
                    slot.insert(next_index);
                    low.insert(next, next_index);
                    next_index += 1;
                    stack.push(next);
                    on_stack.insert(next);
                    call.push((next, 0));
                } else if on_stack.contains(&next) {
                    let candidate = index_of[&next];
                    let current = low[&node];
                    low.insert(node, current.min(candidate));
                }
                continue;
            }
            // Every successor explored: close the node.
            if low[&node] == index_of[&node] {
                let mut component = Vec::new();
                while let Some(member) = stack.pop() {
                    on_stack.remove(&member);
                    component.push(member);
                    if member == node {
                        break;
                    }
                }
                let cyclic = component.len() > 1
                    || graph.get(&node).map(|s| s.contains(&node)).unwrap_or(false);
                if cyclic {
                    components.push(component);
                }
            }
            if let Some((parent, _)) = call.last().copied() {
                let child = low[&node];
                let current = low[&parent];
                low.insert(parent, current.min(child));
            }
        }
    }
    components
}
