//! The dependency graph of a compiled model: which quantity reads which.
//!
//! Two static analyses walk it. [`crate::loops`] looks for an automaton
//! whose guard reads what its own decision moves; [`crate::cycle_groups`]
//! looks for a failure whose effect reaches a feared event through a
//! quantity that remembers it. They need different edges, so the builder
//! takes an [`EdgeSet`] rather than each analysis keeping its own copy of
//! the tables it reads.

use std::collections::HashMap;

use crate::compile::{
    CAllocation, CExpr, CLaw, CProgram, CProgramAffine, CProgramTieBreak, CStep, CompiledModel,
};

/// A node of the dependency graph: the graph is bipartite in kind, and
/// mixing both in one index space is what gives a walk its transitivity
/// for nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Node {
    Var(usize),
    Aut(usize),
}

/// Which edges the graph carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EdgeSet {
    /// What decides a mode and what defines a value: an automaton leads
    /// to what its guards read, a variable to what its defining
    /// expression reads (explicit equations, distribution operators,
    /// linear blocks, ODE right-hand sides, sensitive functions). The
    /// set [`crate::loops::switching_loops`] has always used.
    Decisions,
    /// [`EdgeSet::Decisions`], plus everything else through which one
    /// quantity can change another: the laws of an automaton's
    /// transitions (a rate or a boundary reading a value), the writes of
    /// a transition's effects (the written attribute depends on the
    /// automaton that fires it and on what the effect reads), programs
    /// (each output depends on everything the program reads) and FMU
    /// units (each output depends on every input).
    Influences,
}

/// Successors of every node: what it reads. Successor lists are sorted
/// and deduplicated, so a walk over them is deterministic.
pub(crate) fn build(model: &CompiledModel, set: EdgeSet) -> HashMap<Node, Vec<Node>> {
    let mut graph: HashMap<Node, Vec<Node>> = HashMap::new();

    for transition in &model.transitions {
        let Some(guard) = &transition.guard else {
            continue;
        };
        reads(Node::Aut(transition.automaton), guard, &mut graph);
    }

    for step in &model.explicit {
        match step {
            CStep::Equation { target, expr } => reads(Node::Var(*target), expr, &mut graph),
            CStep::Allocate(allocation) => allocate_edges(allocation, &mut graph),
            // Every member of a block is a function of the block's
            // coefficient inputs: each member leads to everything the
            // rows read (its own rows never read the other members, so
            // a solved cycle no longer connects variables to each other
            // here: the algebraic loop it came from is now a block).
            CStep::Block(block) => {
                for &target in &block.targets {
                    for expr in &block.constants {
                        reads(Node::Var(target), expr, &mut graph);
                    }
                    for row in &block.coefficients {
                        for (_, coefficient) in row {
                            reads(Node::Var(target), coefficient, &mut graph);
                        }
                    }
                }
            }
        }
    }
    for (target, expr) in &model.ode {
        reads(Node::Var(*target), expr, &mut graph);
    }
    for function in &model.functions {
        for (target, expr) in &function.effects {
            reads(Node::Var(*target), expr, &mut graph);
        }
    }

    if set == EdgeSet::Influences {
        for transition in &model.transitions {
            let automaton = Node::Aut(transition.automaton);
            match &transition.distrib {
                CLaw::ExpVar { rate, .. } => reads(automaton, rate, &mut graph),
                CLaw::Watched { margin } => reads(automaton, margin, &mut graph),
                _ => {}
            }
            for (target, expr) in &transition.effects {
                reads(Node::Var(*target), expr, &mut graph);
                graph.entry(Node::Var(*target)).or_default().push(automaton);
            }
        }
        for program in &model.programs {
            program_edges(program, &mut graph);
        }
        for unit in &model.fmu_units {
            let inputs: Vec<Node> = unit
                .inputs
                .iter()
                .filter_map(|binding| fmu_var(model, binding))
                .map(Node::Var)
                .collect();
            for output in unit.outputs.iter().filter_map(|b| fmu_var(model, b)) {
                graph
                    .entry(Node::Var(output))
                    .or_default()
                    .extend(inputs.iter().copied());
            }
        }
    }

    for successors in graph.values_mut() {
        successors.sort_unstable();
        successors.dedup();
    }
    graph
}

/// The attribute an FMU binding reads or writes, by index.
pub(crate) fn fmu_var(model: &CompiledModel, binding: &raichu_model::FmuBinding) -> Option<usize> {
    let name = format!(
        "{}.{}",
        binding.attribute.component, binding.attribute.attribute
    );
    model.var_index.get(&name).copied()
}

/// `node` reads what `expr` reads.
fn reads(node: Node, expr: &CExpr, graph: &mut HashMap<Node, Vec<Node>>) {
    let (vars, auts) = sensitivity(expr);
    let entry = graph.entry(node).or_default();
    entry.extend(vars.into_iter().map(Node::Var));
    // Reading another automaton's STATE is an edge of its own: the value
    // decided on need not pass through a variable.
    entry.extend(auts.into_iter().map(Node::Aut));
}

/// What a distribution operator writes depends on what it has to give
/// and on every demand competing for it, so each allocated attribute
/// leads to both.
fn allocate_edges(allocation: &CAllocation, graph: &mut HashMap<Node, Vec<Node>>) {
    for &target in &allocation.allocated {
        reads(Node::Var(target), &allocation.available, graph);
        graph
            .entry(Node::Var(target))
            .or_default()
            .extend(allocation.demands.iter().copied().map(Node::Var));
    }
}

/// Every output of a program depends on every quantity the program reads:
/// its objective, its constraints and its bounds.
fn program_edges(program: &CProgram, graph: &mut HashMap<Node, Vec<Node>>) {
    fn affine(affine: &CProgramAffine) -> impl Iterator<Item = &CExpr> {
        std::iter::once(&affine.constant).chain(affine.terms.iter().map(|(_, expr)| expr))
    }
    let mut exprs: Vec<&CExpr> = affine(&program.objective).collect();
    if let CProgramTieBreak::Objectives(objectives) = &program.tie_break {
        for (_, objective) in objectives {
            exprs.extend(affine(objective));
        }
    }
    for constraint in &program.constraints {
        exprs.extend(affine(&constraint.expr));
        exprs.extend(constraint.lower.iter());
        exprs.extend(constraint.upper.iter());
    }
    for variable in &program.variables {
        exprs.extend(variable.lower.iter());
        exprs.extend(variable.upper.iter());
    }
    let outputs = program
        .variables
        .iter()
        .map(|v| v.target)
        .chain(std::iter::once(program.feasible))
        .chain(program.objective_value);
    for output in outputs {
        for expr in &exprs {
            reads(Node::Var(output), expr, graph);
        }
    }
}

/// The variables and automata one expression reads, sorted and
/// deduplicated.
pub(crate) fn sensitivity(expr: &CExpr) -> (Vec<usize>, Vec<usize>) {
    let mut vars = Vec::new();
    let mut auts = Vec::new();
    expr.collect_sensitivity(&mut vars, &mut auts);
    vars.sort_unstable();
    vars.dedup();
    auts.sort_unstable();
    auts.dedup();
    (vars, auts)
}
