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

/// Issue #212: HiGHS reported `[1, 3, 3, 0, 2, 0, 0, 2]` (objective -25) as
/// optimal. Enumerating every integer assignment, with the two continuous
/// columns at an exact LP vertex, gives the optimum -26 at
/// `[0, 3, 3, 0, 2, 0, 0, 1]`, the point microlp returns.
#[test]
fn the_issue_212_program_reaches_its_enumerated_optimum() -> Result<(), &'static str> {
    let program = program(17161775123737999326, 8, 6);
    let Outcome::Optimal(highs) = solve(&program, &TieBreak::None) else {
        return Err("expected HiGHS optimum");
    };
    assert!((highs.objective - -26.0).abs() <= 1e-7, "HiGHS={highs:?}");
    Ok(())
}

/// Programs on which HiGHS 1.15.0 ends its first configuration in a solve
/// error; the retry ladder reaches the optimum, enumerated exactly.
#[test]
fn a_solve_error_is_retried_until_the_enumerated_optimum() -> Result<(), &'static str> {
    for (seed, columns, rows, optimum) in [
        (7572538445104058331, 8, 6, 7.0),
        (11864884968066695602, 7, 3, 8.0),
        (6378720912266257836, 7, 6, 0.0),
    ] {
        let Outcome::Optimal(highs) = solve(&program(seed, columns, rows), &TieBreak::None) else {
            return Err("expected HiGHS optimum");
        };
        assert!(
            (highs.objective - optimum).abs() <= 1e-7,
            "seed {seed}: HiGHS={highs:?}"
        );
    }
    Ok(())
}

/// A fixed sweep, so a disagreement reproduces on every run instead of on the
/// rare proptest draw that hits it (#212 surfaced that way).
#[test]
fn a_fixed_sweep_of_programs_agrees_with_the_reference_solver() {
    let mut state: u64 = 0x5eed_0212;
    let mut disagreements = Vec::new();
    for case in 0..std::env::var("SWEEP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000usize)
    {
        let seed = next(&mut state) << 32 | next(&mut state);
        let columns = 1 + case % 8;
        let rows = case % 7;
        let program = program(seed, columns, rows);
        let agree = match (solve(&program, &TieBreak::None), microlp_outcome(&program)) {
            (Outcome::Optimal(a), Outcome::Optimal(b)) => (a.objective - b.objective).abs() <= 1e-7,
            (Outcome::Infeasible, Outcome::Infeasible)
            | (Outcome::Unbounded, Outcome::Unbounded) => true,
            _ => false,
        };
        if !agree {
            disagreements.push((seed, columns, rows));
        }
    }
    assert!(
        disagreements.is_empty(),
        "{} disagreements, first {:?}",
        disagreements.len(),
        &disagreements[..disagreements.len().min(5)]
    );
}
