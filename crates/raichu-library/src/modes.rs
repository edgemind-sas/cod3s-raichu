//! Mode automata and their activation conditions.
//!
//! An activated mode becomes an automaton named after it inside its
//! target instance, with states `rep` (initial) and `occ` and transitions
//! named after the state they enter, so sequence events read
//! `Instance.mode.occ` and every analysis that keys on `monitored`,
//! `kind` and `cycle_group` keeps working.

use crate::document::EffectStep;
use crate::error::{DocumentKind, ExpandError};
use crate::names::split_public;
use crate::strict::typed;
use raichu_expr::{BoolOp, Expr, StateRef};
use raichu_model::{
    Automaton, CycleCancellation, Distrib, InterruptionPolicy, Transition, TransitionKind,
};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

/// State of a mode that has not occurred (initial).
pub const REP: &str = "rep";
/// State of an occurred mode (and of an occurred feared event).
pub const OCC: &str = "occ";
/// Automaton of a feared event.
pub const EVENT_AUTOMATON: &str = "ev";
/// State of a feared event that has not occurred (initial).
pub const NOT_OCC: &str = "not_occ";
/// State an on-demand mode parks in after a draw it lost, until its
/// solicitation falls.
pub const PARKED: &str = "parked";

/// The modes a study activates, `(instance, mode)`, with the occurred
/// states a `mode_active` leaf on each reads: the mode's own automaton, or
/// the combinations of the common-cause groups covering the instance.
pub(crate) type ActiveModes = BTreeMap<(String, String), Vec<(String, String)>>;

/// Rewrites every `{"op": "mode_active", "mode": "Instance.mode"}` leaf of
/// `condition` into the core `state_active` on the mode's `occ` state (an
/// OR over the covering combinations for a group member), refusing a mode
/// the study does not activate.
pub(crate) fn rewrite_mode_leaves(
    condition: &Value,
    active: &ActiveModes,
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
            let Some(states) = active.get(&(instance.to_owned(), mode.to_owned())) else {
                return Err(ExpandError::new(
                    document,
                    pointer,
                    format!("{context}: the condition names mode `{written}`, which the study does not activate"),
                ));
            };
            let leaves: Vec<Value> = states
                .iter()
                .map(|(component, automaton)| {
                    json!({"op": "state_active", "state": {"component": component, "automaton": automaton, "state": OCC}})
                })
                .collect();
            Ok(match leaves.len() {
                1 => leaves.into_iter().next().unwrap_or(Value::Null),
                _ => json!({"op": "bool", "bool_op": "or", "args": leaves}),
            })
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

/// Types a law object, refusing the shapes a mode cannot carry. An
/// occurrence may be on demand (`inst` with one probability: the chance
/// the mode occurs on each rising edge of its condition); a repair is
/// timed.
pub(crate) fn law(
    value: &Value,
    document: DocumentKind,
    pointer: &str,
    context: &str,
    occurrence: bool,
) -> Result<Distrib, ExpandError> {
    let distrib: Distrib = typed(value, document, pointer, context)?;
    match distrib {
        Distrib::Inst { ref probs }
            if occurrence && probs.len() == 1 && (0.0..=1.0).contains(&probs[0]) =>
        {
            Ok(distrib)
        }
        Distrib::Inst { .. } if occurrence => Err(ExpandError::new(
            document,
            pointer,
            format!(
                "{context}: an on-demand occurrence is `{{\"distrib\": \"inst\", \"probs\": [p]}}` with one probability p in [0, 1]: the chance the mode occurs when its condition rises"
            ),
        )),
        Distrib::Inst { .. } | Distrib::Watched => Err(ExpandError::new(
            document,
            pointer,
            format!(
                "{context}: a mode law is timed (exp, delay, weibull, lognormal, gamma, uniform, empirical), and an occurrence may also be on demand (inst); `{}` is not supported for a {} in this version",
                value.get("distrib").and_then(Value::as_str).unwrap_or("?"),
                if occurrence { "occurrence" } else { "repair" }
            ),
        )),
        other => Ok(other),
    }
}

/// The automaton of an activated mode: `rep` and `occ`, plus `parked` for
/// an on-demand occurrence.
///
/// An on-demand mode draws once per rising edge of its condition: the
/// draw `occ` leaves `rep` towards `occ` (the law's probability) or
/// `parked`, and only the entry into `occ` is a sequence event; the
/// re-arm `parked` returns to `rep` once the condition falls, and is no
/// event. Parking a lost draw is what stops it from being drawn again
/// within the same solicitation.
///
/// `repair_guard` (a repair condition) guards the repair edge only: the
/// re-arm of an on-demand mode keeps the negated occurrence guard.
pub(crate) fn mode_automaton(
    name: &str,
    guard: Option<Expr>,
    repair_guard: Option<Expr>,
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
    let on_demand = matches!(occurrence, Distrib::Inst { .. });
    let mut states = vec![REP.to_owned(), OCC.to_owned()];
    let mut transitions = Vec::new();
    if on_demand {
        let rearm_guard = Expr::Bool {
            bool_op: BoolOp::Not,
            args: vec![guard.clone().unwrap_or(Expr::Const {
                value: raichu_expr::Value::Bool(true),
            })],
        };
        // A draw is resolved within its instant: only `reset` makes sense
        // (a solicitation that falls first cancels it), whatever the
        // activation asks of a timed law.
        let mut draw = edge(OCC, REP, TransitionKind::Failure, occurrence, guard);
        draw.on_interruption = InterruptionPolicy::Reset;
        draw.targets = vec![OCC.to_owned(), PARKED.to_owned()];
        draw.monitored_states = Some(vec![OCC.to_owned()]);
        transitions.push(draw);
        transitions.push(Transition {
            name: PARKED.to_owned(),
            source: PARKED.to_owned(),
            guard: Some(rearm_guard),
            targets: vec![REP.to_owned()],
            on_interruption: InterruptionPolicy::Reset,
            monitored: false,
            monitored_states: None,
            cycle_group: None,
            cycle_cancellation: CycleCancellation::Auto,
            kind: None,
            effects: Vec::new(),
            distrib: Distrib::Inst { probs: Vec::new() },
        });
        states.push(PARKED.to_owned());
    } else {
        transitions.push(edge(OCC, REP, TransitionKind::Failure, occurrence, guard));
    }
    if let Some(repair) = repair {
        transitions.push(edge(REP, OCC, TransitionKind::Repair, repair, repair_guard));
    }
    Automaton {
        name: name.to_owned(),
        states,
        init: REP.to_owned(),
        transitions,
    }
}

/// The automaton carrying the effects of `mode` one step behind it.
pub(crate) fn effect_automaton_name(mode: &str) -> String {
    format!("{mode}__effect")
}

/// The effect step of `mode` on `instance`: states `rep` and `occ`, an
/// `occ` transition taken once the mode is occurred and a `rep` one once it
/// is not, each a zero delay or an instantaneous law of probability one.
/// Like a PyCATSHOO mirror, it carries the mode's sequence events and its
/// failure and repair kinds, in the mode's cycle group (see
/// [`hand_events_to_step`]).
pub(crate) fn effect_step_automaton(instance: &str, mode: &str, step: EffectStep) -> Automaton {
    step_automaton(
        effect_automaton_name(mode),
        occurred(instance, mode),
        step,
        Some(mode),
    )
}

/// Strips the sequence events, kinds and cycle group off a mode automaton
/// whose effects an effect step carries (the cycle group is the step's
/// alone, as it is the mirror's alone on the declaration route): the step records them, once per instance it
/// holds, as the declaration route's mirrors do.
pub(crate) fn hand_events_to_step(automaton: &mut Automaton) {
    for transition in &mut automaton.transitions {
        transition.monitored = false;
        transition.monitored_states = None;
        transition.kind = None;
        transition.cycle_group = None;
    }
}

/// An effect step named `name`: states `rep` and `occ`, taken to `occ`
/// once `follows` holds and back to `rep` once it does not. With
/// `events`, the step carries the sequence events and the failure and
/// repair kinds, in that cycle group, instead of the automata it follows.
pub(crate) fn step_automaton(
    name: String,
    follows: Expr,
    step: EffectStep,
    events: Option<&str>,
) -> Automaton {
    let distrib = || match step {
        EffectStep::ZeroDelay => Distrib::Delay { time: 0.0 },
        EffectStep::Instant => Distrib::Inst { probs: Vec::new() },
    };
    let edge = |edge_name: &str, source: &str, guard: Expr| Transition {
        name: edge_name.to_owned(),
        source: source.to_owned(),
        guard: Some(guard),
        targets: vec![edge_name.to_owned()],
        on_interruption: InterruptionPolicy::Reset,
        monitored: events.is_some(),
        monitored_states: None,
        cycle_group: events.map(str::to_owned),
        cycle_cancellation: CycleCancellation::Auto,
        kind: events.map(|_| {
            if edge_name == OCC {
                TransitionKind::Failure
            } else {
                TransitionKind::Repair
            }
        }),
        effects: Vec::new(),
        distrib: distrib(),
    };
    Automaton {
        name,
        states: vec![REP.to_owned(), OCC.to_owned()],
        init: REP.to_owned(),
        transitions: vec![
            edge(OCC, REP, follows.clone()),
            edge(
                REP,
                OCC,
                Expr::Bool {
                    bool_op: BoolOp::Not,
                    args: vec![follows],
                },
            ),
        ],
    }
}

/// True while the effect step of `instance.mode` is in `state`.
pub(crate) fn effect_state(instance: &str, mode: &str, state: &str) -> Expr {
    Expr::StateActive {
        state: StateRef {
            component: instance.to_owned(),
            automaton: effect_automaton_name(mode),
            state: state.to_owned(),
        },
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
