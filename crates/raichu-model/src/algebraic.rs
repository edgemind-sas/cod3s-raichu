//! The instantaneous read relation and the classification of its algebraic cycles.

use crate::graph::has_perfect_matching;
use crate::{
    channel_attribute_name, strongly_connected_components, Allocation, AllocationPolicy, Component,
    Connection, Equation, EquationKind, Model, ModelError, Scope,
};
use raichu_expr::{AggOp, AttrRef, Expr, PortRef, Value};
use std::collections::{HashMap, HashSet};

/// One node of the instantaneous dependency graph: the qualified name
/// `(component, attribute)` of an attribute an explicit equation
/// computes.
pub(crate) type EquationNode = (String, String);

/// Which level of the singularity analysis refused an algebraic cycle
/// (see [`ModelError::AlgebraicSingular`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingularReason {
    /// No assignment of equations to unknowns exists on the block's
    /// sparsity pattern: some attribute nothing determines (or two
    /// equations determine the same one). Refused by the model layer,
    /// which sees the pattern.
    Structural,
    /// The sparsity pattern admits an assignment, but with every
    /// coefficient constant at build time the factorisation of the
    /// system is rank-deficient (`x = y` with `y = x`, or `x = x`).
    /// Refused by the compiler, which runs the factorisation.
    RankDeficient,
}

impl std::fmt::Display for SingularReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SingularReason::Structural => write!(
                f,
                "no assignment of equations to unknowns exists on the \
                 sparsity pattern"
            ),
            SingularReason::RankDeficient => write!(
                f,
                "every coefficient is constant and the constant system is \
                 rank-deficient"
            ),
        }
    }
}

/// Whether `expr` is the literal zero: the one coefficient the sparsity
/// pattern counts as absent. Anything else : a computed expression, an
/// `if`, an attribute read : counts as non-zero, because whether it can
/// vanish is a question about the state, and structure is decided
/// without one.
fn is_literal_zero(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Const {
            value: Value::Float(0.0) | Value::Int(0)
        }
    )
}

/// Render `component.attribute` names, comma-joined, for a diagnostic
/// that names a whole block.
fn render_variables(variables: &[AttrRef]) -> String {
    variables
        .iter()
        .map(|attr| format!("{}.{}", attr.component, attr.attribute))
        .collect::<Vec<_>>()
        .join(", ")
}

/// One equation of a linear algebraic block: its right-hand side
/// separated into an unknown-free part and one coefficient per block
/// unknown it reads.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockRow {
    /// The unknown-free part of the right-hand side.
    pub constant: Expr,
    /// One `(unknown, coefficient)` pair per block unknown this row
    /// reads, in first-occurrence order, with repeated coefficients merged.
    /// Coefficients retain their expression form, including literal zeros.
    pub terms: Vec<(AttrRef, Expr)>,
}

/// A cycle of explicit equations that is **linear in its unknowns**: the
/// unit of simultaneous evaluation the engine solves in place.
///
/// The `n` equations write the `n` attributes of [`AlgebraicBlock::unknowns`]
/// (same order), so the block is the system `x = c + A·x`: row `i` of
/// [`AlgebraicBlock::rows`] gives the unknown-free part `c_i` and the
/// non-zero coefficients of `A`'s row `i`. The engine solves
/// `(I − A)·x = c` wherever the sweep runs.
#[derive(Debug, Clone, PartialEq)]
pub struct AlgebraicBlock {
    /// The block's unknowns: the target attribute of each member
    /// equation, in the member equations' declaration order.
    pub unknowns: Vec<AttrRef>,
    /// One row per member equation, in the same order as `unknowns`.
    pub rows: Vec<BlockRow>,
}

impl Model {
    /// Classify every cycle of the explicit equations (see
    /// [`Model::algebraic_blocks`]).
    ///
    /// The graph has **one node per explicit-equation target** and an
    /// edge from a target to every explicit-equation target its
    /// right-hand side reads, in-port aggregations resolved through the
    /// connections feeding them. An ODE target is deliberately *not* a
    /// node: the integrator carries its value, so it breaks any cycle it
    /// sits on, which is exactly the capacity carve-out. An **allocated
    /// quantity** ([`Allocation`]) is not a node either, and that
    /// carve-out is equally deliberate: a conservative flow network is
    /// cyclic by nature (what a consumer asks for depends on what it was
    /// given), and refusing it here would refuse the very shape the
    /// operator exists for.
    ///
    /// Strongly connected components of this graph are the cycles.
    /// Tarjan's algorithm partitions them in one depth-first pass, in a
    /// deterministic order for a fixed declaration order. Each component
    /// is then classified: nonlinear in its unknowns (refused, naming
    /// the term), structurally singular (refused, naming the
    /// variables), or a linear block handed back to the compiler.
    pub(crate) fn classify_algebraic(
        &self,
        scopes: &HashMap<&str, Scope<'_>>,
    ) -> Result<Vec<AlgebraicBlock>, ModelError> {
        let mut nodes: Vec<EquationNode> = Vec::new();
        let mut index: HashMap<EquationNode, usize> = HashMap::new();
        let mut equations: HashMap<EquationNode, &Equation> = HashMap::new();
        for component in &self.components {
            for equation in &component.equations {
                if equation.kind == EquationKind::Explicit {
                    let node = (component.name.clone(), equation.target.clone());
                    index.insert(node.clone(), nodes.len());
                    equations.insert(node.clone(), equation);
                    nodes.push(node);
                }
            }
        }
        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Connections indexed by destination in port, once: a resolved
        // flow network has as many connections as equations, and walking
        // the whole list per aggregation would make validation quadratic.
        let mut feeds: HashMap<(&str, &str), Vec<&Connection>> = HashMap::new();
        for connection in &self.connections {
            feeds
                .entry((
                    connection.to.component.as_str(),
                    connection.to.port.as_str(),
                ))
                .or_default()
                .push(connection);
        }
        // The resolution every read of an in port goes through, shared
        // by this classification and `collect_reads`.
        let port_sources = |port: &PortRef, channel: Option<&str>| {
            Self::resolve_port_sources(scopes, &feeds, port, channel)
        };

        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
        for component in &self.components {
            for equation in &component.equations {
                if equation.kind != EquationKind::Explicit {
                    continue;
                }
                let key = (component.name.clone(), equation.target.clone());
                let Some(&node) = index.get(&key) else {
                    continue;
                };
                let mut reads = Vec::new();
                Self::collect_reads(scopes, &feeds, &equation.expr, &mut reads);
                for read in reads {
                    if let Some(&dependency) = index.get(&read) {
                        if !edges[node].contains(&dependency) {
                            edges[node].push(dependency);
                        }
                    }
                }
            }
        }

        // Declaration order in, declaration order out: Tarjan emits the
        // components in a fixed order for a fixed declaration order, and
        // the first refused one is always the same.
        let mut blocks = Vec::new();
        for members in strongly_connected_components(&edges) {
            let self_loop = members.len() == 1 && edges[members[0]].contains(&members[0]);
            if members.len() == 1 && !self_loop {
                continue;
            }
            // Members in declaration order: the block's unknown order.
            let mut members = members;
            members.sort_unstable();
            let unknowns: Vec<AttrRef> = members
                .iter()
                .map(|&node| AttrRef {
                    component: nodes[node].0.clone(),
                    attribute: nodes[node].1.clone(),
                })
                .collect();
            let selected: HashSet<AttrRef> = unknowns.iter().cloned().collect();

            // Affinity of every member, coefficients merged per unknown:
            // one entry per (row, unknown), which is the sparsity
            // pattern and, past this point, the compiled block.
            let mut rows: Vec<BlockRow> = Vec::with_capacity(members.len());
            for &node in &members {
                let equation = equations[&nodes[node]];
                let form = equation
                    .expr
                    .affine_in_resolved(&selected, &port_sources)
                    .map_err(|term| ModelError::AlgebraicNonlinear {
                        component: nodes[node].0.clone(),
                        attribute: nodes[node].1.clone(),
                        term: term.expression.render(),
                    })?;
                let mut merged: Vec<(AttrRef, Expr)> = Vec::with_capacity(form.terms.len());
                for (unknown, coefficient) in form.terms {
                    if let Some(seen) = merged.iter_mut().find(|(seen, _)| *seen == unknown) {
                        let previous = std::mem::replace(
                            &mut seen.1,
                            Expr::Const {
                                value: Value::Float(0.0),
                            },
                        );
                        seen.1 = Expr::Add {
                            args: vec![previous, coefficient],
                        };
                    } else {
                        merged.push((unknown, coefficient));
                    }
                }
                rows.push(BlockRow {
                    constant: form.constant,
                    terms: merged,
                });
            }

            // Structural singularity: no perfect matching on the
            // pattern, where any coefficient that is not the literal 0
            // counts as non-zero. A column is an unknown's position in
            // the block, not a term's position in its row.
            let columns: HashMap<&AttrRef, usize> = unknowns
                .iter()
                .enumerate()
                .map(|(column, unknown)| (unknown, column))
                .collect();
            // The solved matrix is I-A. Keep the identity diagonal unless
            // a literal coefficient 1 cancels it. State-dependent or
            // computed coefficients stay conservative; the LU rank test
            // handles their cancellations on the actual matrix.
            let pattern: Vec<Vec<usize>> = rows
                .iter()
                .enumerate()
                .map(|(row_index, row)| {
                    let diagonal_cancels = row.terms.iter().any(|(unknown, coefficient)| {
                        columns[unknown] == row_index
                            && matches!(
                                coefficient,
                                Expr::Const {
                                    value: Value::Float(1.0) | Value::Int(1)
                                }
                            )
                    });
                    let mut entries = if diagonal_cancels {
                        Vec::new()
                    } else {
                        vec![row_index]
                    };
                    entries.extend(
                        row.terms
                            .iter()
                            .filter(|(unknown, coefficient)| {
                                columns[unknown] != row_index && !is_literal_zero(coefficient)
                            })
                            .map(|(unknown, _)| columns[unknown]),
                    );
                    entries.sort_unstable();
                    entries.dedup();
                    entries
                })
                .collect();
            if !has_perfect_matching(&pattern, unknowns.len()) {
                return Err(ModelError::AlgebraicSingular {
                    variables: render_variables(&unknowns),
                    reason: SingularReason::Structural,
                });
            }
            blocks.push(AlgebraicBlock { unknowns, rows });
        }
        Ok(blocks)
    }

    /// The linear blocks of this model: every cycle of the explicit
    /// equations, classified and solved-in-place ready.
    ///
    /// A cycle **linear in its unknowns** comes back as one
    /// [`AlgebraicBlock`]; the engine evaluates its coefficients and
    /// right-hand sides and solves the system `(I − A)·x = c` wherever
    /// the sweep runs. A cycle that is not linear (a `min`, a
    /// threshold, a `median`, an `if` condition reading an unknown) is
    /// refused with the term that makes it so, and a cycle whose system
    /// is structurally singular is refused with the variables it
    /// involves ([`ModelError::AlgebraicSingular`] at the
    /// [`SingularReason::Structural`] level; the rank-deficient level
    /// needs a factorisation and is raised by the compiler).
    ///
    /// An ODE target and an allocated quantity are deliberately absent
    /// from the graph: the integrator and the distribution operator
    /// carry them, and a cycle they sit on is well posed.
    ///
    /// # Errors
    /// [`ModelError::AlgebraicNonlinear`] and
    /// [`ModelError::AlgebraicSingular`], first refusal in a
    /// deterministic order.
    pub fn algebraic_blocks(&self) -> Result<Vec<AlgebraicBlock>, ModelError> {
        let scopes = self.check_components()?;
        self.classify_algebraic(&scopes)
    }

    /// Refuse a **priority split combined with surplus return** (see
    /// [`ModelError::AllocationPrioritySurplusReturn`]).
    ///
    /// *Surplus return* is a structural property, not a declared one: a
    /// consumer returns surplus when the demand it publishes on an edge
    /// depends, through the instantaneous reads of the explicit sweep, on
    /// a quantity that same operator allocated. That is precisely the
    /// cycle [`Model::algebraic_blocks`] deliberately allows, which
    /// is why it has to be recognised here instead.
    ///
    /// The search runs over the same read relation the loop detection
    /// uses, extended with one edge family: an allocated quantity reads
    /// the available quantity and every demand of its operator. It starts
    /// at each demand of a priority operator and looks for any allocated
    /// quantity of that same operator. A demand reaching *another*
    /// operator's output is ordinary network coupling and is left alone.
    pub(crate) fn check_priority_surplus_return(
        &self,
        scopes: &HashMap<&str, Scope<'_>>,
    ) -> Result<(), ModelError> {
        let priority_operators: Vec<(&Component, &Allocation)> = self
            .components
            .iter()
            .flat_map(|component| {
                component
                    .allocations
                    .iter()
                    .map(move |allocation| (component, allocation))
            })
            .filter(|(_, allocation)| {
                matches!(allocation.policy, AllocationPolicy::Priority { .. })
            })
            .collect();
        if priority_operators.is_empty() {
            return Ok(());
        }

        let mut feeds: HashMap<(&str, &str), Vec<&Connection>> = HashMap::new();
        for connection in &self.connections {
            feeds
                .entry((
                    connection.to.component.as_str(),
                    connection.to.port.as_str(),
                ))
                .or_default()
                .push(connection);
        }

        // The read relation: which attributes each computed attribute
        // reads at the instant it is evaluated.
        let mut reads: HashMap<EquationNode, Vec<EquationNode>> = HashMap::new();
        for component in &self.components {
            for equation in &component.equations {
                if equation.kind != EquationKind::Explicit {
                    continue;
                }
                let mut into = Vec::new();
                Self::collect_reads(scopes, &feeds, &equation.expr, &mut into);
                reads.insert((component.name.clone(), equation.target.clone()), into);
            }
            for allocation in &component.allocations {
                let mut into = Vec::new();
                Self::collect_reads(scopes, &feeds, &allocation.available, &mut into);
                for connection in self.allocation_edges(component, allocation) {
                    into.push((
                        component.name.clone(),
                        channel_attribute_name(connection, &allocation.demand),
                    ));
                }
                for connection in self.allocation_edges(component, allocation) {
                    let allocated = (
                        component.name.clone(),
                        channel_attribute_name(connection, &allocation.allocated),
                    );
                    reads.insert(allocated, into.clone());
                }
            }
        }

        for (component, allocation) in priority_operators {
            let edges = self.allocation_edges(component, allocation);
            let allocated: HashSet<EquationNode> = edges
                .iter()
                .map(|connection| {
                    (
                        component.name.clone(),
                        channel_attribute_name(connection, &allocation.allocated),
                    )
                })
                .collect();
            for connection in &edges {
                let start = (
                    component.name.clone(),
                    channel_attribute_name(connection, &allocation.demand),
                );
                if let Some(reached) = Self::reaches(&reads, &start, &allocated) {
                    return Err(ModelError::AllocationPrioritySurplusReturn {
                        component: component.name.clone(),
                        allocation: allocation.name.clone(),
                        demand: start.1,
                        allocated: reached.1,
                    });
                }
            }
        }
        Ok(())
    }

    /// The connections one distribution operator's out port carries, in
    /// declaration order (the order every per-edge vector shares).
    pub(crate) fn allocation_edges(
        &self,
        component: &Component,
        allocation: &Allocation,
    ) -> Vec<&Connection> {
        self.connections
            .iter()
            .filter(|connection| {
                connection.from.component == component.name
                    && connection.from.port == allocation.port
            })
            .collect()
    }

    /// A member of `wanted` reachable from `start` through the read
    /// relation, or `None`.
    ///
    /// A depth-first walk whose candidates are sorted before they are
    /// pushed, so the attribute that ends up in the diagnostic is a
    /// property of the model rather than of a hash order.
    fn reaches(
        reads: &HashMap<EquationNode, Vec<EquationNode>>,
        start: &EquationNode,
        wanted: &HashSet<EquationNode>,
    ) -> Option<EquationNode> {
        let mut seen: HashSet<EquationNode> = HashSet::new();
        let mut frontier = vec![start.clone()];
        while let Some(node) = frontier.pop() {
            if !seen.insert(node.clone()) {
                continue;
            }
            let Some(next) = reads.get(&node) else {
                continue;
            };
            let mut sorted: Vec<EquationNode> = next.clone();
            sorted.sort();
            sorted.dedup();
            for candidate in sorted.into_iter().rev() {
                if wanted.contains(&candidate) {
                    return Some(candidate);
                }
                frontier.push(candidate);
            }
        }
        None
    }

    /// Attributes an expression reads at the instant it is evaluated,
    /// with in-port aggregations resolved through the connections that
    /// feed them (a named channel reads the per-connection attributes
    /// materialised for it).
    ///
    /// `count` reads the *topology* and never a value, so it creates no
    /// data dependency: counting the connections of an in port cannot
    /// close an algebraic loop.
    pub(crate) fn collect_reads(
        scopes: &HashMap<&str, Scope<'_>>,
        feeds: &HashMap<(&str, &str), Vec<&Connection>>,
        expr: &Expr,
        into: &mut Vec<EquationNode>,
    ) {
        match expr {
            Expr::Const { .. } | Expr::StateActive { .. } | Expr::Time => {}
            Expr::Attr { attr } => into.push((attr.component.clone(), attr.attribute.clone())),
            Expr::PortAgg { port, agg, channel } => {
                if *agg == AggOp::Count {
                    return;
                }
                for AttrRef {
                    component,
                    attribute,
                } in Self::resolve_port_sources(scopes, feeds, port, channel.as_deref())
                {
                    into.push((component, attribute));
                }
            }
            Expr::Cmp { lhs, rhs, .. } | Expr::Sub { lhs, rhs } | Expr::Div { lhs, rhs } => {
                Self::collect_reads(scopes, feeds, lhs, into);
                Self::collect_reads(scopes, feeds, rhs, into);
            }
            Expr::Bool { args, .. }
            | Expr::Add { args }
            | Expr::Mul { args }
            | Expr::Min { args }
            | Expr::Max { args } => {
                for arg in args {
                    Self::collect_reads(scopes, feeds, arg, into);
                }
            }
            Expr::If {
                cond,
                then,
                otherwise,
            } => {
                Self::collect_reads(scopes, feeds, cond, into);
                Self::collect_reads(scopes, feeds, then, into);
                Self::collect_reads(scopes, feeds, otherwise, into);
            }
            Expr::Sin { arg }
            | Expr::Exp { arg }
            | Expr::PiecewiseLinear { arg, .. }
            | Expr::SmoothNoise { arg, .. } => {
                Self::collect_reads(scopes, feeds, arg, into);
            }
        }
    }

    /// The attributes one in-port aggregation reads: one per incoming
    /// connection, in connection declaration order : the exported
    /// attribute of the producer, or the per-connection attribute
    /// materialised for the named channel.
    fn resolve_port_sources(
        scopes: &HashMap<&str, Scope<'_>>,
        feeds: &HashMap<(&str, &str), Vec<&Connection>>,
        port: &PortRef,
        channel: Option<&str>,
    ) -> Vec<AttrRef> {
        let key = (port.component.as_str(), port.port.as_str());
        feeds
            .get(&key)
            .into_iter()
            .flatten()
            .filter_map(|connection| match channel {
                Some(channel) => Some(AttrRef {
                    component: connection.from.component.clone(),
                    attribute: channel_attribute_name(connection, channel),
                }),
                None => scopes
                    .get(connection.from.component.as_str())
                    .and_then(|scope| scope.ports.get(connection.from.port.as_str()))
                    .and_then(|source| source.attr.as_ref())
                    .map(|attribute| AttrRef {
                        component: connection.from.component.clone(),
                        attribute: attribute.clone(),
                    }),
            })
            .collect()
    }
}
