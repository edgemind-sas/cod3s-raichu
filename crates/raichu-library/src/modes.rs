//! Mode automata and their activation conditions.
//!
//! An activated mode becomes an automaton named after it inside its
//! target instance, with states `rep` (initial) and `occ` and transitions
//! named after the state they enter, so sequence events read
//! `Instance.mode.occ` and every analysis that keys on `monitored`,
//! `kind` and `cycle_group` keeps working.

use crate::error::{DocumentKind, ExpandError};
use crate::names::split_public;
use crate::strict::typed;
use raichu_expr::{BoolOp, Expr, StateRef};
use raichu_model::{
    Automaton, CycleCancellation, Distrib, InterruptionPolicy, Transition, TransitionKind,
};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

/// State of a mode that has not occurred (initial).
pub const REP: &str = "rep";
/// State of an occurred mode (and of an occurred feared event).
pub const OCC: &str = "occ";
/// Automaton of a feared event.
pub const EVENT_AUTOMATON: &str = "ev";
/// State of a feared event that has not occurred (initial).
pub const NOT_OCC: &str = "not_occ";

/// Rewrites every `{"op": "mode_active", "mode": "Instance.mode"}` leaf of
/// `condition` into the core `state_active` on the mode's `occ` state,
/// refusing a mode the study does not activate.
pub(crate) fn rewrite_mode_leaves(
    condition: &Value,
    active: &BTreeSet<(String, String)>,
    document: DocumentKind,
    pointer: &str,
    context: &str,
) -> Result<Value, ExpandError> {
    match condition {
        Value::Object(map) if map.get("op").and_then(Value::as_str) == Some("mode_active") => {
            if let Some(extra) = map.keys().find(|k| !matches!(k.as_str(), "op" | "mode")) {
                return Err(ExpandError::new(
                    document,
                    format!("{pointer}/{extra}"),
                    format!("{context}: unknown field `{extra}` in `mode_active` (fields: `op`, `mode`)"),
                ));
            }
            let written = map.get("mode").and_then(Value::as_str).ok_or_else(|| {
                ExpandError::new(
                    document,
                    pointer,
                    format!("{context}: `mode_active` names a mode as `\"Instance.mode\"`"),
                )
            })?;
            let (instance, mode) = split_public(written).ok_or_else(|| {
                ExpandError::new(
                    document,
                    pointer,
                    format!(
                        "{context}: `{written}` is not a public mode name; write `Instance.mode`"
                    ),
                )
            })?;
            if !active.contains(&(instance.to_owned(), mode.to_owned())) {
                return Err(ExpandError::new(
                    document,
                    pointer,
                    format!("{context}: the condition names mode `{written}`, which the study does not activate"),
                ));
            }
            Ok(
                json!({"op": "state_active", "state": {"component": instance, "automaton": mode, "state": OCC}}),
            )
        }
        Value::Object(map) => {
            let mut out = Map::with_capacity(map.len());
            for (key, child) in map {
                out.insert(
                    key.clone(),
                    rewrite_mode_leaves(
                        child,
                        active,
                        document,
                        &format!("{pointer}/{key}"),
                        context,
                    )?,
                );
            }
            Ok(Value::Object(out))
        }
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for (index, child) in items.iter().enumerate() {
                out.push(rewrite_mode_leaves(
                    child,
                    active,
                    document,
                    &format!("{pointer}/{index}"),
                    context,
                )?);
            }
            Ok(Value::Array(out))
        }
        other => Ok(other.clone()),
    }
}

/// Types a condition (already rewritten) into a core expression.
pub(crate) fn condition_expr(
    condition: &Value,
    document: DocumentKind,
    pointer: &str,
    context: &str,
) -> Result<Expr, ExpandError> {
    typed::<Expr>(condition, document, pointer, context)
}

/// Types a law object, refusing the shapes a mode cannot carry.
pub(crate) fn law(
    value: &Value,
    document: DocumentKind,
    pointer: &str,
    context: &str,
) -> Result<Distrib, ExpandError> {
    let distrib: Distrib = typed(value, document, pointer, context)?;
    match distrib {
        Distrib::Inst { .. } | Distrib::Watched => Err(ExpandError::new(
            document,
            pointer,
            format!(
                "{context}: a mode law is timed (exp, delay, weibull, lognormal, gamma, uniform, empirical); `{}` is not supported for modes in this version",
                value.get("distrib").and_then(Value::as_str).unwrap_or("?")
            ),
        )),
        other => Ok(other),
    }
}

/// The two-state automaton of an activated mode.
pub(crate) fn mode_automaton(
    name: &str,
    guard: Option<Expr>,
    occurrence: Distrib,
    repair: Option<Distrib>,
    on_interruption: InterruptionPolicy,
) -> Automaton {
    let edge = |edge_name: &str, source: &str, kind, distrib, guard| Transition {
        name: edge_name.to_owned(),
        source: source.to_owned(),
        guard,
        targets: vec![edge_name.to_owned()],
        on_interruption,
        monitored: true,
        monitored_states: None,
        cycle_group: Some(name.to_owned()),
        cycle_cancellation: CycleCancellation::Auto,
        kind: Some(kind),
        effects: Vec::new(),
        distrib,
    };
    let mut transitions = vec![edge(OCC, REP, TransitionKind::Failure, occurrence, guard)];
    if let Some(repair) = repair {
        transitions.push(edge(REP, OCC, TransitionKind::Repair, repair, None));
    }
    Automaton {
        name: name.to_owned(),
        states: vec![REP.to_owned(), OCC.to_owned()],
        init: REP.to_owned(),
        transitions,
    }
}

/// True while `instance.mode` is occurred.
pub(crate) fn occurred(instance: &str, mode: &str) -> Expr {
    Expr::StateActive {
        state: StateRef {
            component: instance.to_owned(),
            automaton: mode.to_owned(),
            state: OCC.to_owned(),
        },
    }
}

/// The conjunction of the present conditions, `None` when there is none.
pub(crate) fn all_of(conditions: Vec<Expr>) -> Option<Expr> {
    match conditions.len() {
        0 => None,
        1 => conditions.into_iter().next(),
        _ => Some(Expr::Bool {
            bool_op: BoolOp::And,
            args: conditions,
        }),
    }
}
