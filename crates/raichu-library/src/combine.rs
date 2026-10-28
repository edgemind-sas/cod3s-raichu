//! Combination of the effects several active modes hold on one attribute.
//!
//! For each targeted attribute the expander writes one sensitive function
//! holding the combined value while at least one targeting mode is
//! occurred, and the attribute's declared initial value otherwise. The
//! nominal value takes no part in `min` or `max`: only the active modes'
//! values compete.

use crate::document::Effect;
use crate::instantiate::CombineRule;
use crate::modes::occurred;
use crate::provenance::Origin;
use raichu_expr::{BoolOp, Expr, Value};

/// Effects a mode (or a common-cause combination) holds on an instance
/// while `component.automaton` is in `occ`.
pub(crate) struct EffectSource {
    pub instance: usize,
    /// Name for messages and provenance.
    pub mode: String,
    pub effects: Vec<Effect>,
    pub component: String,
    pub automaton: String,
    /// Where the effects were declared.
    pub effects_origin: Origin,
}

/// One mode's held value on the attribute, held while the automaton
/// `automaton` of the instance (the mode's own, or its effect step) is in
/// `occ`.
pub(crate) struct Holder {
    pub mode: String,
    /// The component holding `automaton`: the instance itself, or a
    /// common-cause group.
    pub component: String,
    pub automaton: String,
    pub value: Value,
}

fn constant(value: Value) -> Expr {
    Expr::Const { value }
}

fn any_active(holders: &[Holder]) -> Expr {
    let mut args: Vec<Expr> = holders
        .iter()
        .map(|h| occurred(&h.component, &h.automaton))
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

/// The combined value of `holders` on an attribute whose initial value is
/// `init`, under `rule`.
pub(crate) fn combined(rule: CombineRule, holders: &[Holder], init: Value) -> Expr {
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
                        cond: Box::new(occurred(&h.component, &h.automaton)),
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
        cond: Box::new(any_active(holders)),
        then: Box::new(held),
        otherwise: Box::new(constant(init)),
    }
}
