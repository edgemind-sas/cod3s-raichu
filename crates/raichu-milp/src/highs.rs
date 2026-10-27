use crate::{Backend, Kind, Outcome, Program, Sense, Solution};
use std::ffi::c_void;

pub(super) struct HighsBackend;

struct Instance(*mut c_void);

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe { highs_sys::Highs_destroy(self.0) };
    }
}

fn numeric_check(program: &Program) -> Result<(), String> {
    if program.columns.is_empty() {
        return Err("a program needs at least one decision variable".to_owned());
    }
    if program.columns.len() > i32::MAX as usize || program.rows.len() > i32::MAX as usize {
        return Err("program dimension exceeds HiGHS index range".to_owned());
    }
    for (index, column) in program.columns.iter().enumerate() {
        if !column.cost.is_finite()
            || column.lower.is_nan()
            || column.upper.is_nan()
            || column.lower > column.upper
            || column.lower == f64::INFINITY
            || column.upper == f64::NEG_INFINITY
        {
            return Err(format!("invalid numeric data in column {index}"));
        }
        if column.kind == Kind::Binary && (column.lower < 0.0 || column.upper > 1.0) {
            return Err(format!("binary column {index} has bounds outside [0, 1]"));
        }
    }
    for (row_index, row) in program.rows.iter().enumerate() {
        if row.lower.is_nan() || row.upper.is_nan() || row.lower > row.upper {
            return Err(format!("invalid bounds in row {row_index}"));
        }
        for &(column_index, coefficient) in &row.terms {
            if column_index >= program.columns.len() || !coefficient.is_finite() {
                return Err(format!("invalid term in row {row_index}"));
            }
        }
    }
    Ok(())
}

fn option_status(status: i32, name: &str) -> Result<(), String> {
    if status == highs_sys::STATUS_ERROR {
        Err(format!("HiGHS refused option {name}"))
    } else {
        Ok(())
    }
}

fn snap_integral(value: f64, index: usize, lower: f64, upper: f64) -> Result<f64, String> {
    if !value.is_finite() {
        return Err(format!("column {index} is not finite"));
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-6 || rounded < lower - 1e-6 || rounded > upper + 1e-6 {
        return Err(format!("column {index} is not integral within its bounds"));
    }
    Ok(rounded)
}

/// One HiGHS configuration of the retry ladder.
#[derive(Clone, Copy)]
struct Attempt {
    presolve: bool,
    restart: bool,
}

/// The configurations tried in turn, stopping at the first answer HiGHS
/// stands behind. HiGHS 1.15.0 has two failure classes on small programs:
/// a run that ends in a solve error, and (issue #212) a run that reports as
/// optimal a pre-restart incumbent worse than its own dual bound. Searching
/// without restart avoids the second on the measured sweep; the other
/// configurations recover most of the first.
const LADDER: [Attempt; 4] = [
    Attempt {
        presolve: true,
        restart: false,
    },
    Attempt {
        presolve: false,
        restart: false,
    },
    Attempt {
        presolve: true,
        restart: true,
    },
    Attempt {
        presolve: false,
        restart: true,
    },
];

/// What one run of the ladder concluded.
enum Run {
    /// An outcome HiGHS stands behind: returned as is.
    Final(Outcome),
    /// The node limit stopped this configuration; another may finish within it.
    Limit,
    /// A configuration-dependent failure: the next configuration is tried.
    Retry(String),
}

fn solve_once(program: &Program, attempt: Attempt) -> Run {
    if let Err(detail) = numeric_check(program) {
        return Run::Final(Outcome::SolverFault(detail));
    }
    let mut starts = Vec::with_capacity(program.rows.len());
    let mut indices = Vec::new();
    let mut coefficients = Vec::new();
    let mut row_lower = Vec::with_capacity(program.rows.len());
    let mut row_upper = Vec::with_capacity(program.rows.len());
    for row in &program.rows {
        if row.terms.iter().all(|(_, coefficient)| *coefficient == 0.0) {
            if row.lower > 0.0 || row.upper < 0.0 {
                return Run::Final(Outcome::Infeasible);
            }
            continue;
        }
        let Ok(start) = i32::try_from(indices.len()) else {
            return Run::Final(Outcome::SolverFault(
                "matrix exceeds HiGHS index range".to_owned(),
            ));
        };
        starts.push(start);
        row_lower.push(row.lower);
        row_upper.push(row.upper);
        for &(index, value) in &row.terms {
            if value == 0.0 {
                continue;
            }
            let Ok(index) = i32::try_from(index) else {
                return Run::Final(Outcome::SolverFault(
                    "matrix exceeds HiGHS index range".to_owned(),
                ));
            };
            indices.push(index);
            coefficients.push(value);
        }
    }
    let Ok(nonzeros) = i32::try_from(indices.len()) else {
        return Run::Final(Outcome::SolverFault(
            "matrix exceeds HiGHS index range".to_owned(),
        ));
    };
    let Ok(node_limit) = i32::try_from(program.node_limit.unwrap_or(100_000)) else {
        return Run::Final(Outcome::SolverFault(
            "node limit exceeds HiGHS index range".to_owned(),
        ));
    };
    let costs: Vec<_> = program.columns.iter().map(|column| column.cost).collect();
    let col_lower: Vec<_> = program.columns.iter().map(|column| column.lower).collect();
    let col_upper: Vec<_> = program.columns.iter().map(|column| column.upper).collect();
    let integrality: Vec<_> = program
        .columns
        .iter()
        .map(|column| match column.kind {
            Kind::Continuous => highs_sys::VAR_TYPE_CONTINUOUS,
            Kind::Integer | Kind::Binary => highs_sys::VAR_TYPE_INTEGER,
        })
        .collect();
    let has_integers = program
        .columns
        .iter()
        .any(|column| column.kind != Kind::Continuous);

    let raw = unsafe { highs_sys::Highs_create() };
    if raw.is_null() {
        return Run::Final(Outcome::SolverFault(
            "HiGHS instance allocation failed".to_owned(),
        ));
    }
    let instance = Instance(raw);
    let output = unsafe { highs_sys::Highs_setBoolOptionValue(raw, c"output_flag".as_ptr(), 0) };
    let threads = unsafe { highs_sys::Highs_setIntOptionValue(raw, c"threads".as_ptr(), 1) };
    let seed = unsafe { highs_sys::Highs_setIntOptionValue(raw, c"random_seed".as_ptr(), 0) };
    let nodes =
        unsafe { highs_sys::Highs_setIntOptionValue(raw, c"mip_max_nodes".as_ptr(), node_limit) };
    let gap = unsafe { highs_sys::Highs_setDoubleOptionValue(raw, c"mip_rel_gap".as_ptr(), 0.0) };
    let absolute_gap =
        unsafe { highs_sys::Highs_setDoubleOptionValue(raw, c"mip_abs_gap".as_ptr(), 0.0) };
    let primal_tolerance = unsafe {
        highs_sys::Highs_setDoubleOptionValue(raw, c"primal_feasibility_tolerance".as_ptr(), 1e-9)
    };
    let mip_tolerance = unsafe {
        highs_sys::Highs_setDoubleOptionValue(raw, c"mip_feasibility_tolerance".as_ptr(), 1e-9)
    };
    let restart = unsafe {
        highs_sys::Highs_setBoolOptionValue(
            raw,
            c"mip_allow_restart".as_ptr(),
            i32::from(attempt.restart),
        )
    };
    let presolve_value = if attempt.presolve { c"on" } else { c"off" };
    let presolve_status = unsafe {
        highs_sys::Highs_setStringOptionValue(raw, c"presolve".as_ptr(), presolve_value.as_ptr())
    };
    for (name, status) in [
        ("output_flag", output),
        ("threads", threads),
        ("random_seed", seed),
        ("mip_max_nodes", nodes),
        ("mip_rel_gap", gap),
        ("mip_abs_gap", absolute_gap),
        ("primal_feasibility_tolerance", primal_tolerance),
        ("mip_feasibility_tolerance", mip_tolerance),
        ("mip_allow_restart", restart),
        ("presolve", presolve_status),
    ] {
        if let Err(detail) = option_status(status, name) {
            return Run::Final(Outcome::SolverFault(detail));
        }
    }
    let sense = match program.sense {
        Sense::Minimize => highs_sys::OBJECTIVE_SENSE_MINIMIZE,
        Sense::Maximize => highs_sys::OBJECTIVE_SENSE_MAXIMIZE,
    };
    let pass_status = unsafe {
        if has_integers {
            highs_sys::Highs_passMip(
                raw,
                costs.len() as i32,
                row_lower.len() as i32,
                nonzeros,
                highs_sys::MATRIX_FORMAT_ROW_WISE,
                sense,
                0.0,
                costs.as_ptr(),
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                row_lower.as_ptr(),
                row_upper.as_ptr(),
                starts.as_ptr(),
                indices.as_ptr(),
                coefficients.as_ptr(),
                integrality.as_ptr(),
            )
        } else {
            highs_sys::Highs_passLp(
                raw,
                costs.len() as i32,
                row_lower.len() as i32,
                nonzeros,
                highs_sys::MATRIX_FORMAT_ROW_WISE,
                sense,
                0.0,
                costs.as_ptr(),
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                row_lower.as_ptr(),
                row_upper.as_ptr(),
                starts.as_ptr(),
                indices.as_ptr(),
                coefficients.as_ptr(),
            )
        }
    };
    if pass_status == highs_sys::STATUS_ERROR {
        return Run::Final(Outcome::SolverFault(
            "HiGHS refused the numeric program".to_owned(),
        ));
    }
    let run_status = unsafe { highs_sys::Highs_run(raw) };
    let model_status = unsafe { highs_sys::Highs_getModelStatus(raw) };
    if run_status == highs_sys::STATUS_ERROR {
        return Run::Retry(format!("HiGHS run failed with status {model_status}"));
    }
    let outcome = match model_status {
        highs_sys::MODEL_STATUS_OPTIMAL => {
            let mut values = vec![0.0; program.columns.len()];
            let mut col_duals = vec![0.0; program.columns.len()];
            let mut row_values = vec![0.0; row_lower.len()];
            let mut row_duals = vec![0.0; row_lower.len()];
            let status = unsafe {
                highs_sys::Highs_getSolution(
                    raw,
                    values.as_mut_ptr(),
                    col_duals.as_mut_ptr(),
                    row_values.as_mut_ptr(),
                    row_duals.as_mut_ptr(),
                )
            };
            if status == highs_sys::STATUS_ERROR {
                return Run::Retry("HiGHS did not return a solution".to_owned());
            }
            for (index, (value, column)) in values.iter_mut().zip(&program.columns).enumerate() {
                if !value.is_finite() {
                    return Run::Retry(format!("column {index} is not finite"));
                }
                if column.kind != Kind::Continuous {
                    *value = match snap_integral(*value, index, column.lower, column.upper) {
                        Ok(snapped) => snapped,
                        Err(detail) => return Run::Retry(detail),
                    };
                }
            }
            let objective = costs
                .iter()
                .zip(&values)
                .map(|(cost, value)| cost * value)
                .sum::<f64>();
            if !objective.is_finite() {
                return Run::Retry("non-finite objective value".to_owned());
            }
            if has_integers {
                // Both gaps are requested at 0, so an optimum HiGHS stands
                // behind meets its own dual bound (issue #212).
                let mut dual_bound = f64::NAN;
                unsafe {
                    highs_sys::Highs_getDoubleInfoValue(
                        raw,
                        c"mip_dual_bound".as_ptr(),
                        &mut dual_bound,
                    )
                };
                if !consistent_with_bound(objective, dual_bound) {
                    return Run::Retry(format!(
                        "HiGHS reported objective {objective} as optimal against dual bound {dual_bound}"
                    ));
                }
            }
            Outcome::Optimal(Solution {
                values,
                objective,
                solve_count: 1,
            })
        }
        highs_sys::MODEL_STATUS_INFEASIBLE => Outcome::Infeasible,
        highs_sys::MODEL_STATUS_UNBOUNDED => Outcome::Unbounded,
        highs_sys::MODEL_STATUS_REACHED_ITERATION_LIMIT
        | highs_sys::MODEL_STATUS_REACHED_TIME_LIMIT
        | highs_sys::MODEL_STATUS_REACHED_SOLUTION_LIMIT
        | highs_sys::MODEL_STATUS_REACHED_MEMORY_LIMIT
        | highs_sys::MODEL_STATUS_REACHED_INTERRUPT => return Run::Limit,
        // Unbounded-or-infeasible, solve and postsolve errors depend on the
        // configuration: the next rung decides.
        _ => return Run::Retry(format!("HiGHS model status {model_status}")),
    };
    drop(instance);
    Run::Final(outcome)
}

/// Whether an objective meets the dual bound of a run asked for a zero gap.
fn consistent_with_bound(objective: f64, dual_bound: f64) -> bool {
    dual_bound.is_finite() && (objective - dual_bound).abs() <= 1e-6 * objective.abs().max(1.0)
}

impl Backend for HighsBackend {
    fn solve(&self, program: &Program) -> Outcome {
        let mut failures = Vec::new();
        let mut limited = false;
        for attempt in LADDER {
            match solve_once(program, attempt) {
                Run::Final(outcome) => return outcome,
                Run::Limit => limited = true,
                Run::Retry(detail) => failures.push(format!(
                    "presolve {}, restart {}: {detail}",
                    if attempt.presolve { "on" } else { "off" },
                    if attempt.restart { "on" } else { "off" },
                )),
            }
        }
        // A limit is the program's own budget: it outranks a configuration fault.
        if limited {
            Outcome::LimitReached
        } else {
            Outcome::SolverFault(failures.join("; "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{consistent_with_bound, snap_integral};

    #[test]
    fn an_optimum_must_meet_the_dual_bound() {
        assert!(consistent_with_bound(-26.0, -26.0));
        assert!(consistent_with_bound(1e9, 1e9 + 1.0));
        assert!(!consistent_with_bound(-25.0, -26.0));
        assert!(!consistent_with_bound(0.0, f64::NAN));
    }

    #[test]
    fn integral_values_are_checked_before_snapping() {
        assert_eq!(snap_integral(0.9999999999, 0, 0.0, 1.0), Ok(1.0));
        assert!(snap_integral(0.99, 0, 0.0, 1.0).is_err());
        assert!(snap_integral(1.0, 0, 0.0, 0.0).is_err());
    }
}
