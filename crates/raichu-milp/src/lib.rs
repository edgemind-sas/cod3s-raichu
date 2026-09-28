//! Deterministic mixed-integer linear programming for RAICHU.

mod highs;
mod lexicographic;

use highs::HighsBackend;

/// A decision variable's domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A real-valued decision.
    Continuous,
    /// An integral decision.
    Integer,
    /// A decision restricted to zero or one.
    Binary,
}

/// Objective direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sense {
    /// Minimise the objective.
    Minimize,
    /// Maximise the objective.
    Maximize,
}

/// A numeric decision variable.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    /// Coefficient in the primary objective.
    pub cost: f64,
    /// Inclusive lower bound.
    pub lower: f64,
    /// Inclusive upper bound.
    pub upper: f64,
    /// Variable domain.
    pub kind: Kind,
}

/// A ranged linear constraint.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// Sparse terms `(column_index, coefficient)`.
    pub terms: Vec<(usize, f64)>,
    /// Inclusive lower bound.
    pub lower: f64,
    /// Inclusive upper bound.
    pub upper: f64,
}

/// A numeric program ready for a solver.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// Decision variables in declaration order.
    pub columns: Vec<Column>,
    /// Ranged constraints.
    pub rows: Vec<Row>,
    /// Primary objective direction.
    pub sense: Sense,
    /// Deterministic branch-and-bound node cap, or the built-in default.
    pub node_limit: Option<u64>,
}

/// A declared secondary objective.
#[derive(Debug, Clone, PartialEq)]
pub struct Objective {
    /// Objective direction.
    pub sense: Sense,
    /// One coefficient per decision variable.
    pub coefficients: Vec<f64>,
}

/// How equal-cost optima are separated.
#[derive(Debug, Clone, PartialEq)]
pub enum TieBreak {
    /// Minimise the decision variables lexicographically in declaration order.
    Default,
    /// Accept whichever optimum the solver returns.
    None,
    /// Optimise these objectives in sequence, then use the default rule.
    Objectives(Vec<Objective>),
}

/// A proven optimal assignment.
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// Decision values in declaration order.
    pub values: Vec<f64>,
    /// Primary objective value.
    pub objective: f64,
    /// Number of backend solves, including lexicographic re-solves.
    pub solve_count: usize,
}

/// Solver outcome. Only [`Outcome::Optimal`] may publish a decision.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Proven optimal assignment.
    Optimal(Solution),
    /// No feasible assignment exists.
    Infeasible,
    /// The objective can improve without a bound.
    Unbounded,
    /// The deterministic node limit fired before an optimum was proven.
    LimitReached,
    /// Invalid numeric input or a solver failure.
    SolverFault(String),
}

/// A numeric backend used by the lexicographic driver.
trait Backend {
    fn solve(&self, program: &Program) -> Outcome;
}

/// Solve a program and apply the requested tie-break.
#[must_use]
pub fn solve(program: &Program, tie_break: &TieBreak) -> Outcome {
    lexicographic::solve(&HighsBackend, program, tie_break)
}
