//! Closed-form linear and mixed-integer programs.

use raichu_milp::{solve, Column, Kind, Objective, Outcome, Program, Row, Sense, TieBreak};

fn dispatch(cheap_capacity: f64, expensive_capacity: f64) -> Program {
    Program {
        columns: vec![
            Column {
                cost: 1.0,
                lower: 0.0,
                upper: cheap_capacity,
                kind: Kind::Continuous,
            },
            Column {
                cost: 2.0,
                lower: 0.0,
                upper: expensive_capacity,
                kind: Kind::Continuous,
            },
        ],
        rows: vec![Row {
            terms: vec![(0, 1.0), (1, 1.0)],
            lower: 80.0,
            upper: 80.0,
        }],
        sense: Sense::Minimize,
        node_limit: None,
    }
}

#[test]
fn two_source_dispatch_and_infeasibility() -> Result<(), &'static str> {
    let Outcome::Optimal(solution) = solve(&dispatch(60.0, 50.0), &TieBreak::Default) else {
        return Err("expected an optimum");
    };
    assert_eq!(solution.values, vec![60.0, 20.0]);
    assert_eq!(solution.objective, 100.0);
    assert_eq!(solution.solve_count, 3);
    let mut maximisation = dispatch(60.0, 50.0);
    maximisation.sense = Sense::Maximize;
    for column in &mut maximisation.columns {
        column.cost = -column.cost;
    }
    let Outcome::Optimal(maximum) = solve(&maximisation, &TieBreak::Default) else {
        return Err("expected a maximising optimum");
    };
    assert_eq!(maximum.values, vec![60.0, 20.0]);
    assert_eq!(maximum.objective, -100.0);
    assert_eq!(
        solve(&dispatch(0.0, 50.0), &TieBreak::Default),
        Outcome::Infeasible
    );
    Ok(())
}

#[test]
fn default_tie_break_selects_one_point() -> Result<(), &'static str> {
    let program = Program {
        columns: vec![
            Column {
                cost: 1.0,
                lower: 0.0,
                upper: 30.0,
                kind: Kind::Continuous,
            },
            Column {
                cost: 1.0,
                lower: 0.0,
                upper: 30.0,
                kind: Kind::Continuous,
            },
        ],
        rows: vec![Row {
            terms: vec![(0, 1.0), (1, 1.0)],
            lower: 30.0,
            upper: 30.0,
        }],
        sense: Sense::Minimize,
        node_limit: None,
    };
    let Outcome::Optimal(solution) = solve(&program, &TieBreak::Default) else {
        return Err("expected an optimum");
    };
    assert!(solution.values[0].abs() <= 1e-8);
    assert!((solution.values[1] - 30.0).abs() <= 1e-8);
    assert!((solution.objective - 30.0).abs() <= 1e-8);
    let Outcome::Optimal(overridden) = solve(
        &program,
        &TieBreak::Objectives(vec![Objective {
            sense: Sense::Maximize,
            coefficients: vec![1.0, 0.0],
        }]),
    ) else {
        return Err("expected a secondary optimum");
    };
    assert!((overridden.values[0] - 30.0).abs() <= 1e-7);
    assert!(overridden.values[1].abs() <= 1e-7);
    let Outcome::Optimal(unseparated) = solve(&program, &TieBreak::None) else {
        return Err("expected an unseparated optimum");
    };
    assert_eq!(unseparated.solve_count, 1);
    Ok(())
}

#[test]
fn unbounded_program_is_explicit() {
    let program = Program {
        columns: vec![Column {
            cost: -1.0,
            lower: 0.0,
            upper: f64::INFINITY,
            kind: Kind::Continuous,
        }],
        rows: vec![],
        sense: Sense::Minimize,
        node_limit: None,
    };
    assert_eq!(solve(&program, &TieBreak::Default), Outcome::Unbounded);
}

#[test]
fn zero_coefficient_rows_are_checked_before_highs() {
    let mut program = dispatch(60.0, 50.0);
    program.rows.push(Row {
        terms: vec![(0, 0.0)],
        lower: f64::NEG_INFINITY,
        upper: 0.0,
    });
    assert!(matches!(
        solve(&program, &TieBreak::None),
        Outcome::Optimal(_)
    ));
    program.rows[1].lower = 1.0;
    program.rows[1].upper = f64::INFINITY;
    assert_eq!(solve(&program, &TieBreak::None), Outcome::Infeasible);
}

#[test]
fn commitment_program_has_enumerated_optimum() -> Result<(), &'static str> {
    let program = Program {
        columns: vec![
            Column {
                cost: 1.0,
                lower: 0.0,
                upper: 60.0,
                kind: Kind::Continuous,
            },
            Column {
                cost: 2.0,
                lower: 0.0,
                upper: 50.0,
                kind: Kind::Continuous,
            },
            Column {
                cost: 10.0,
                lower: 0.0,
                upper: 1.0,
                kind: Kind::Binary,
            },
            Column {
                cost: 15.0,
                lower: 0.0,
                upper: 1.0,
                kind: Kind::Binary,
            },
        ],
        rows: vec![
            Row {
                terms: vec![(0, 1.0), (1, 1.0)],
                lower: 80.0,
                upper: 80.0,
            },
            Row {
                terms: vec![(0, 1.0), (2, -60.0)],
                lower: f64::NEG_INFINITY,
                upper: 0.0,
            },
            Row {
                terms: vec![(1, 1.0), (3, -50.0)],
                lower: f64::NEG_INFINITY,
                upper: 0.0,
            },
            Row {
                terms: vec![(0, 1.0), (2, -20.0)],
                lower: 0.0,
                upper: f64::INFINITY,
            },
            Row {
                terms: vec![(1, 1.0), (3, -20.0)],
                lower: 0.0,
                upper: f64::INFINITY,
            },
        ],
        sense: Sense::Minimize,
        node_limit: None,
    };
    let Outcome::Optimal(solution) = solve(&program, &TieBreak::Default) else {
        return Err("expected an optimum");
    };
    assert_eq!(solution.values, vec![60.0, 20.0, 1.0, 1.0]);
    assert_eq!(solution.objective, 125.0);
    assert_eq!(solution.solve_count, 5);
    assert_eq!(
        solve(&program, &TieBreak::Default),
        Outcome::Optimal(solution)
    );
    Ok(())
}

#[test]
fn node_limit_never_publishes_an_incumbent() {
    let mut state = 1u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        state >> 32
    };
    let mut columns = Vec::new();
    let mut terms = Vec::new();
    let mut total_weight = 0.0;
    for index in 0..20 {
        let weight = (next() % 50 + 1) as f64;
        let profit = (next() % 50 + 1) as f64;
        total_weight += weight;
        columns.push(Column {
            cost: profit,
            lower: 0.0,
            upper: 1.0,
            kind: Kind::Binary,
        });
        terms.push((index, weight));
    }
    let program = Program {
        columns,
        rows: vec![Row {
            terms,
            lower: f64::NEG_INFINITY,
            upper: total_weight * 0.47,
        }],
        sense: Sense::Maximize,
        node_limit: Some(0),
    };
    assert_eq!(solve(&program, &TieBreak::None), Outcome::LimitReached);
}
