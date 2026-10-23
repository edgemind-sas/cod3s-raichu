//! Combination of the effects several active modes hold on one attribute.
//!
//! For each targeted attribute the expander writes one sensitive function
//! holding the combined value while at least one targeting mode is
//! occurred, and the attribute's declared initial value otherwise. The
//! nominal value takes no part in `min` or `max`: only the active modes'
//! values compete.

use crate::instantiate::CombineRule;
use crate::modes::occurred;
use raichu_expr::{BoolOp, Expr, Value};

/// One mode's held value on the attribute.
pub(crate) struct Holder {
    pub mode: String,
    pub value: Value,
}

fn constant(value: Value) -> Expr {
    Expr::Const { value }
}

fn any_active(instance: &str, holders: &[Holder]) -> Expr {
    let mut args: Vec<Expr> = holders
        .iter()
        .map(|h| occurred(instance, &h.mode))
        .collect();
    if args.len() == 1 {
        args.remove(0)
    } else {
        Expr::Bool {
            bool_op: BoolOp::Or,
            args,
        }
    }
}

fn as_f64(value: Value) -> f64 {
    match value {
        Value::Float(x) => x,
        #[allow(clippy::cast_precision_loss)]
        Value::Int(x) => x as f64,
        Value::Bool(b) => f64::from(u8::from(b)),
    }
}

/// The combined value of `holders` on an attribute of `instance` whose
/// initial value is `init`, under `rule`.
pub(crate) fn combined(instance: &str, rule: CombineRule, holders: &[Holder], init: Value) -> Expr {
    let held = match rule {
        CombineRule::AnyCuts => constant(Value::Bool(false)),
        CombineRule::AnyActivates => constant(Value::Bool(true)),
        CombineRule::Min | CombineRule::Max => {
            if holders.len() == 1 {
                constant(holders[0].value)
            } else {
                // A mode that is not occurred contributes the neutral value:
                // the largest held value for `min`, the smallest for `max`,
                // which never wins against an active mode's value.
                let values: Vec<Value> = holders.iter().map(|h| h.value).collect();
                let pick = |a: &Value, b: &Value| as_f64(*a).total_cmp(&as_f64(*b));
                let neutral = match rule {
                    CombineRule::Min => values.iter().copied().max_by(pick),
                    _ => values.iter().copied().min_by(pick),
                }
                .unwrap_or(init);
                let args = holders
                    .iter()
                    .map(|h| Expr::If {
                        cond: Box::new(occurred(instance, &h.mode)),
                        then: Box::new(constant(h.value)),
                        otherwise: Box::new(constant(neutral)),
                    })
                    .collect();
                if rule == CombineRule::Min {
                    Expr::Min { args }
                } else {
                    Expr::Max { args }
                }
            }
        }
    };
    Expr::If {
        cond: Box::new(any_active(instance, holders)),
        then: Box::new(held),
        otherwise: Box::new(constant(init)),
    }
}
