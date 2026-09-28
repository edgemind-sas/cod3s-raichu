use crate::{Backend, Objective, Outcome, Program, Row, Sense, TieBreak};

// Match the solver's feasibility scale while avoiding an exact equality on a
// floating-point objective. Each stage may relax the previous optimum by at
// most this absolute or relative amount.
const OBJECTIVE_FIX_TOLERANCE: f64 = 1e-9;

fn fixed_objective(terms: Vec<(usize, f64)>, sense: Sense, value: f64, relaxed: bool) -> Row {
    if !relaxed {
        return Row {
            terms,
            lower: value,
            upper: value,
        };
    }
    let tolerance = OBJECTIVE_FIX_TOLERANCE * value.abs().max(1.0);
    match sense {
        Sense::Minimize => Row {
            terms,
            lower: f64::NEG_INFINITY,
            upper: value + tolerance,
        },
        Sense::Maximize => Row {
            terms,
            lower: value - tolerance,
            upper: f64::INFINITY,
        },
    }
}

pub(super) fn solve(backend: &impl Backend, program: &Program, tie_break: &TieBreak) -> Outcome {
    let mut solution = match backend.solve(program) {
        Outcome::Optimal(solution) => solution,
        other => return other,
    };
    if *tie_break == TieBreak::None {
        return Outcome::Optimal(solution);
    }
    let primary_costs: Vec<_> = program.columns.iter().map(|column| column.cost).collect();
    let primary_value = solution.objective;
    let mut current = program.clone();
    let mut fixed = vec![(current.rows.len(), program.sense, primary_value)];
    current.rows.push(fixed_objective(
        primary_costs.iter().copied().enumerate().collect(),
        program.sense,
        primary_value,
        false,
    ));
    let declared: &[Objective] = match tie_break {
        TieBreak::Objectives(objectives) => objectives,
        _ => &[],
    };
    let mut objectives = declared.to_vec();
    objectives.extend((0..program.columns.len()).map(|index| {
        let mut coefficients = vec![0.0; program.columns.len()];
        coefficients[index] = 1.0;
        Objective {
            sense: Sense::Minimize,
            coefficients,
        }
    }));
    for objective in objectives {
        if objective.coefficients.len() != program.columns.len()
            || objective
                .coefficients
                .iter()
                .any(|value| !value.is_finite())
        {
            return Outcome::SolverFault("invalid tie-break objective".to_owned());
        }
        current.sense = objective.sense;
        for (column, coefficient) in current.columns.iter_mut().zip(&objective.coefficients) {
            column.cost = *coefficient;
        }
        let mut outcome = backend.solve(&current);
        if matches!(outcome, Outcome::Infeasible) {
            for &(row, sense, value) in &fixed {
                current.rows[row] =
                    fixed_objective(current.rows[row].terms.clone(), sense, value, true);
            }
            outcome = backend.solve(&current);
        }
        let Outcome::Optimal(next) = outcome else {
            return match outcome {
                Outcome::Infeasible => Outcome::SolverFault(
                    "lexicographic fixing made a proven optimum infeasible".to_owned(),
                ),
                other => other,
            };
        };
        solution.solve_count += next.solve_count;
        solution.values = next.values;
        let fixed_value: f64 = objective
            .coefficients
            .iter()
            .zip(&solution.values)
            .map(|(coefficient, value)| coefficient * value)
            .sum();
        fixed.push((current.rows.len(), objective.sense, fixed_value));
        current.rows.push(fixed_objective(
            objective.coefficients.iter().copied().enumerate().collect(),
            objective.sense,
            fixed_value,
            false,
        ));
    }
    solution.objective = primary_costs
        .iter()
        .zip(&solution.values)
        .map(|(cost, value)| cost * value)
        .sum();
    Outcome::Optimal(solution)
}
