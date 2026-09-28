//! Small bounded programs compared with an independent pure-Rust solver.

use microlp::{ComparisonOp, OptimizationDirection, Problem, SolutionStatus, SolveOutcome};
use proptest::prelude::*;
use raichu_milp::{solve, Column, Kind, Outcome, Program, Row, Sense, TieBreak};

fn next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed >> 32
}

fn program(mut seed: u64, columns: usize, rows: usize) -> Program {
    let variables = (0..columns)
        .map(|index| Column {
            cost: (next(&mut seed) % 11) as f64 - 5.0,
            lower: 0.0,
            upper: if index % 3 == 0 { 1.0 } else { 3.0 },
            kind: if index % 3 == 0 {
                Kind::Binary
            } else if index % 3 == 1 {
                Kind::Integer
            } else {
                Kind::Continuous
            },
        })
        .collect();
    let mut constraints = (0..rows)
        .map(|_| Row {
            terms: (0..columns)
                .map(|index| (index, (next(&mut seed) % 7) as f64 - 3.0))
                .filter(|(_, coefficient)| *coefficient != 0.0)
                .collect(),
            lower: f64::NEG_INFINITY,
            upper: (next(&mut seed) % 9) as f64,
        })
        .collect::<Vec<_>>();
    if seed % 5 == 0 {
        constraints.push(Row {
            terms: vec![(0, 1.0)],
            lower: 2.0,
            upper: f64::INFINITY,
        });
    }
    Program {
        columns: variables,
        rows: constraints,
        sense: if seed % 2 == 0 {
            Sense::Minimize
        } else {
            Sense::Maximize
        },
        node_limit: None,
    }
}

fn microlp_outcome(program: &Program) -> Outcome {
    let direction = match program.sense {
        Sense::Minimize => OptimizationDirection::Minimize,
        Sense::Maximize => OptimizationDirection::Maximize,
    };
    let mut problem = Problem::new(direction);
    let vars: Vec<_> = program
        .columns
        .iter()
        .map(|column| match column.kind {
            Kind::Continuous => problem.add_var(column.cost, (column.lower, column.upper)),
            Kind::Integer => {
                problem.add_integer_var(column.cost, (column.lower as i32, column.upper as i32))
            }
            Kind::Binary => problem.add_binary_var(column.cost),
        })
        .collect();
    for row in &program.rows {
        let terms: Vec<_> = row
            .terms
            .iter()
            .map(|&(index, coefficient)| (vars[index], coefficient))
            .collect();
        if row.lower.is_finite() {
            problem.add_constraint(terms.as_slice(), ComparisonOp::Ge, row.lower);
        }
        if row.upper.is_finite() {
            problem.add_constraint(terms.as_slice(), ComparisonOp::Le, row.upper);
        }
    }
    match problem.solve() {
        Ok(SolveOutcome::Solution(solution)) if solution.status() == SolutionStatus::Optimal => {
            Outcome::Optimal(raichu_milp::Solution {
                values: vars.iter().map(|&var| solution.var_value(var)).collect(),
                objective: solution.objective(),
                solve_count: 1,
            })
        }
        Err(microlp::Error::Infeasible) => Outcome::Infeasible,
        Err(microlp::Error::Unbounded) => Outcome::Unbounded,
        other => Outcome::SolverFault(format!("unexpected microlp outcome: {other:?}")),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]
    #[test]
    fn bounded_random_programs_agree(seed in any::<u64>(), columns in 1usize..=8, rows in 0usize..=6) {
        let program = program(seed, columns, rows);
        let highs = solve(&program, &TieBreak::None);
        let microlp = microlp_outcome(&program);
        match (highs, microlp) {
            (Outcome::Optimal(a), Outcome::Optimal(b)) => {
                prop_assert!((a.objective - b.objective).abs() <= 1e-7, "HiGHS={a:?}, microlp={b:?}, program={program:?}");
            }
            (Outcome::Infeasible, Outcome::Infeasible)
            | (Outcome::Unbounded, Outcome::Unbounded) => {}
            (left, right) => prop_assert!(false, "HiGHS={left:?}, microlp={right:?}"),
        }
    }
}

#[test]
fn mixed_integer_feasibility_is_tight_enough_for_objective_parity() -> Result<(), &'static str> {
    let program = program(3967135651774528876, 8, 6);
    let Outcome::Optimal(highs) = solve(&program, &TieBreak::None) else {
        return Err("expected HiGHS optimum");
    };
    let Outcome::Optimal(microlp) = microlp_outcome(&program) else {
        return Err("expected microlp optimum");
    };
    assert!((highs.objective - microlp.objective).abs() <= 1e-7);
    Ok(())
}
