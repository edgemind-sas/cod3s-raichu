//! Expression evaluation: guards, effects, rates and right-hand sides
//! evaluated over the attribute vector, the automaton states and the clock.

use super::*;

/// Evaluate a compiled expression with every attribute frozen at `vars`
/// and every automaton in `states`, at time zero: what the fault-tree
/// generator reads a guard, a rate or a boundary as.
pub(crate) fn eval_frozen(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    expr: &CExpr,
) -> Result<Value, EngineError> {
    eval_expr(model, vars, states, 0.0, expr)
}

/// Evaluate a compiled expression against an explicit state (usable
/// both by the engine and by the continuous-system adapter).
pub(super) fn eval_expr(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
) -> Result<Value, EngineError> {
    match expr {
        CExpr::Const(value) => Ok(*value),
        CExpr::Var(idx) => Ok(vars[*idx]),
        CExpr::StateActive { automaton, state } => Ok(Value::Bool(states[*automaton] == *state)),
        CExpr::PortAgg { sources, agg } => eval_agg(model, vars, time, sources, *agg),
        CExpr::Cmp { op, lhs, rhs } => {
            let lhs = eval_expr(model, vars, states, time, lhs)?;
            let rhs = eval_expr(model, vars, states, time, rhs)?;
            eval_cmp(time, *op, lhs, rhs)
        }
        CExpr::Bool { op, args } => match op {
            BoolOp::And => {
                for arg in args {
                    if !eval_bool(model, vars, states, time, arg)? {
                        return Ok(Value::Bool(false));
                    }
                }
                Ok(Value::Bool(true))
            }
            BoolOp::Or => {
                for arg in args {
                    if eval_bool(model, vars, states, time, arg)? {
                        return Ok(Value::Bool(true));
                    }
                }
                Ok(Value::Bool(false))
            }
            // Arity validated at model build (exactly one).
            BoolOp::Not => Ok(Value::Bool(!eval_bool(
                model, vars, states, time, &args[0],
            )?)),
        },
        CExpr::Add { args } | CExpr::Mul { args } => {
            let product = matches!(expr, CExpr::Mul { .. });
            let mut acc_i: i64 = if product { 1 } else { 0 };
            let mut acc_f: f64 = if product { 1.0 } else { 0.0 };
            let mut any_float = false;
            for arg in args {
                match eval_num(model, vars, states, time, arg)? {
                    Num::Int(i) => {
                        if product {
                            acc_i *= i;
                            acc_f *= i as f64;
                        } else {
                            acc_i += i;
                            acc_f += i as f64;
                        }
                    }
                    Num::Float(f) => {
                        any_float = true;
                        if product {
                            acc_f *= f;
                        } else {
                            acc_f += f;
                        }
                    }
                }
            }
            if any_float {
                Ok(Value::Float(acc_f))
            } else {
                Ok(Value::Int(acc_i))
            }
        }
        CExpr::Sub { lhs, rhs } => {
            let lhs = eval_num(model, vars, states, time, lhs)?;
            let rhs = eval_num(model, vars, states, time, rhs)?;
            Ok(match (lhs, rhs) {
                (Num::Int(a), Num::Int(b)) => Value::Int(a - b),
                (a, b) => Value::Float(a.as_f64() - b.as_f64()),
            })
        }
        CExpr::Div { lhs, rhs } => {
            let lhs = eval_num(model, vars, states, time, lhs)?.as_f64();
            let rhs = eval_num(model, vars, states, time, rhs)?.as_f64();
            // IEEE semantics (±inf on zero divisor); NaN is caught by
            // comparisons and the integrator's finiteness checks.
            Ok(Value::Float(lhs / rhs))
        }
        CExpr::Min { args } | CExpr::Max { args } => {
            let take_min = matches!(expr, CExpr::Min { .. });
            let mut best: Option<Num> = None;
            for arg in args {
                let value = eval_num(model, vars, states, time, arg)?;
                best = Some(match best {
                    None => value,
                    Some(current) => {
                        let replace = if take_min {
                            value.as_f64() < current.as_f64()
                        } else {
                            value.as_f64() > current.as_f64()
                        };
                        if replace {
                            value
                        } else {
                            current
                        }
                    }
                });
            }
            // Arity ≥ 1 validated at model build.
            Ok(best.map_or(Value::Int(0), Num::into_value))
        }
        CExpr::If {
            cond,
            then,
            otherwise,
        } => {
            if eval_bool(model, vars, states, time, cond)? {
                eval_expr(model, vars, states, time, then)
            } else {
                eval_expr(model, vars, states, time, otherwise)
            }
        }
        CExpr::Sin(arg) => Ok(Value::Float(
            eval_num(model, vars, states, time, arg)?.as_f64().sin(),
        )),
        CExpr::Exp(arg) => Ok(Value::Float(
            eval_num(model, vars, states, time, arg)?.as_f64().exp(),
        )),
        CExpr::Time => Ok(Value::Float(time)),
    }
}

/// Numeric intermediate for arithmetic evaluation.
#[derive(Debug, Clone, Copy)]
enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    fn as_f64(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }
    fn into_value(self) -> Value {
        match self {
            Num::Int(i) => Value::Int(i),
            Num::Float(f) => Value::Float(f),
        }
    }
}

fn eval_num(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
) -> Result<Num, EngineError> {
    match eval_expr(model, vars, states, time, expr)? {
        Value::Int(i) => Ok(Num::Int(i)),
        Value::Float(f) => Ok(Num::Float(f)),
        Value::Bool(_) => Err(EngineError::TypeError {
            time,
            detail: "arithmetic on a boolean value".to_owned(),
        }),
    }
}

pub(super) fn eval_bool(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
) -> Result<bool, EngineError> {
    match eval_expr(model, vars, states, time, expr)? {
        Value::Bool(b) => Ok(b),
        other => Err(EngineError::TypeError {
            time,
            detail: format!("expected a boolean, got {other:?}"),
        }),
    }
}

pub(super) fn eval_f64(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
) -> Result<f64, EngineError> {
    Ok(eval_num(model, vars, states, time, expr)?.as_f64())
}

fn eval_agg(
    model: &CompiledModel,
    vars: &[Value],
    time: f64,
    sources: &[VarIdx],
    agg: AggOp,
) -> Result<Value, EngineError> {
    match agg {
        AggOp::Count => Ok(Value::Int(sources.len() as i64)),
        AggOp::Sum => {
            let mut int_sum = 0i64;
            let mut float_sum = 0.0f64;
            let mut any_float = false;
            for &idx in sources {
                match vars[idx] {
                    Value::Int(i) => int_sum += i,
                    Value::Float(f) => {
                        any_float = true;
                        float_sum += f;
                    }
                    Value::Bool(b) => int_sum += i64::from(b),
                }
            }
            if any_float {
                Ok(Value::Float(float_sum + int_sum as f64))
            } else {
                Ok(Value::Int(int_sum))
            }
        }
        AggOp::All | AggOp::Any => {
            let mut all = true;
            let mut any = false;
            for &idx in sources {
                match vars[idx] {
                    Value::Bool(b) => {
                        all &= b;
                        any |= b;
                    }
                    other => {
                        return Err(EngineError::TypeError {
                            time,
                            detail: format!(
                                "boolean aggregation over non-boolean value {other:?} \
                                 (attribute `{}`)",
                                model.var_names[idx]
                            ),
                        });
                    }
                }
            }
            Ok(Value::Bool(if agg == AggOp::All { all } else { any }))
        }
        AggOp::Mean | AggOp::Median => {
            let mut values = Vec::with_capacity(sources.len());
            for &idx in sources {
                values.push(match vars[idx] {
                    Value::Int(i) => i as f64,
                    Value::Float(f) => f,
                    Value::Bool(b) => f64::from(u8::from(b)),
                });
            }
            if values.is_empty() {
                return Ok(Value::Float(0.0));
            }
            if agg == AggOp::Mean {
                let n = values.len() as f64;
                Ok(Value::Float(values.iter().sum::<f64>() / n))
            } else {
                values.sort_by(f64::total_cmp);
                let mid = values.len() / 2;
                Ok(Value::Float(if values.len() % 2 == 1 {
                    values[mid]
                } else {
                    0.5 * (values[mid - 1] + values[mid])
                }))
            }
        }
    }
}

/// Whether `value cmp bound` holds, as a total function.
///
/// The same semantics as [`eval_cmp`], minus the two cases model
/// validation has already refused, plus a decision that function leaves
/// to its caller: a comparison involving a NaN has no ordering, and here
/// it answers IEEE's way -- `ne` holds, everything else does not. An
/// indicator is an observation, so the run continues and the threshold
/// simply does not hold while the observed attribute is not a number.
pub(super) fn predicate_holds(value: Value, cmp: CmpOp, bound: Value) -> bool {
    let ordering = match (value, bound) {
        (Value::Bool(a), Value::Bool(b)) => {
            return match cmp {
                CmpOp::Eq => a == b,
                CmpOp::Ne => a != b,
                // Ordering a boolean is refused at model build
                // (`ModelError::IndicatorPredicateKind`), so this is
                // unreachable through a validated model. Spelled out
                // rather than folded into `eq`: a threshold answering
                // the wrong comparison is the defect this whole variant
                // exists to end, and `false` is at least a condition
                // that never holds rather than one that quietly holds
                // half the time.
                CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => false,
            };
        }
        (Value::Int(a), Value::Int(b)) => a.partial_cmp(&b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(&b),
        (Value::Int(a), Value::Float(b)) => (a as f64).partial_cmp(&b),
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(b as f64)),
        // Refused at model build: a boolean against a number.
        _ => None,
    };
    let Some(ordering) = ordering else {
        return matches!(cmp, CmpOp::Ne);
    };
    match cmp {
        CmpOp::Eq => ordering.is_eq(),
        CmpOp::Ne => !ordering.is_eq(),
        CmpOp::Lt => ordering.is_lt(),
        CmpOp::Le => ordering.is_le(),
        CmpOp::Gt => ordering.is_gt(),
        CmpOp::Ge => ordering.is_ge(),
    }
}

fn eval_cmp(time: f64, op: CmpOp, lhs: Value, rhs: Value) -> Result<Value, EngineError> {
    let ordering = match (lhs, rhs) {
        (Value::Bool(a), Value::Bool(b)) => {
            return match op {
                CmpOp::Eq => Ok(Value::Bool(a == b)),
                CmpOp::Ne => Ok(Value::Bool(a != b)),
                _ => Err(EngineError::TypeError {
                    time,
                    detail: format!("ordering comparison {op:?} on booleans"),
                }),
            };
        }
        (Value::Int(a), Value::Int(b)) => a.partial_cmp(&b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(&b),
        (Value::Int(a), Value::Float(b)) => (a as f64).partial_cmp(&b),
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(b as f64)),
        (a, b) => {
            return Err(EngineError::TypeError {
                time,
                detail: format!("comparison between incompatible kinds {a:?} and {b:?}"),
            });
        }
    };
    let Some(ordering) = ordering else {
        return Err(EngineError::TypeError {
            time,
            detail: "comparison involving NaN".to_owned(),
        });
    };
    let result = match op {
        CmpOp::Eq => ordering.is_eq(),
        CmpOp::Ne => !ordering.is_eq(),
        CmpOp::Lt => ordering.is_lt(),
        CmpOp::Le => ordering.is_le(),
        CmpOp::Gt => ordering.is_gt(),
        CmpOp::Ge => ordering.is_ge(),
    };
    Ok(Value::Bool(result))
}
