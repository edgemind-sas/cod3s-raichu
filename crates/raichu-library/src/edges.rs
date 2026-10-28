//! One-shot writes of active modes on the edges of their lifecycle.
//!
//! An `edge_effects` entry of a catalogued mode is written once, as
//! `Transition.effects`, on the edge that carries the mode's effects: the
//! mode's own automaton, its effect step when it has one, or for a
//! common-cause group each combination covering the member (each member's
//! step when the group has one). The attribute then keeps the value until
//! something else writes it, which is how a gate that memorises (a
//! detection, an alarm) is modelled; no level function writes it.

use crate::combine::EffectSource;
use crate::document::Edge;
use crate::error::ExpandError;
use crate::instantiate::ExpandedInstance;
use crate::modes::{OCC, REP};
use crate::provenance::Origin;
use crate::study::writers;
use raichu_expr::{Assignment, AttrRef, Expr, StateRef, Value};
use raichu_model::{AttrKind, Automaton, Component};
use serde_json::Value as Json;
use std::collections::{BTreeMap, BTreeSet};

fn written_value(kind: AttrKind, value: &Json) -> Option<Value> {
    match kind {
        AttrKind::Bool => value.as_bool().map(Value::Bool),
        AttrKind::Int => value.as_i64().map(Value::Int),
        AttrKind::Float => value.as_f64().map(Value::Float),
    }
}

/// One edge write, checked, waiting to be placed.
struct Write {
    /// Where the edge lives: the member itself, or a group component.
    component: String,
    automaton: String,
    edge: Edge,
    member: String,
    attribute: String,
    value: Value,
    mode: String,
    origin: Origin,
}

fn error(origin: &Origin, message: String) -> ExpandError {
    ExpandError::new(origin.document, origin.pointer.clone(), message)
}

/// Writes the `edge_effects` of every source on its edges. Runs after the
/// held effects are combined, so an attribute a mode holds is seen as such.
pub(crate) fn apply_edge_effects(
    sources: &[EffectSource],
    instances: &mut [ExpandedInstance],
    groups: &mut [Component],
) -> Result<(), ExpandError> {
    let mut writes = Vec::new();
    for (index, instance) in instances.iter().enumerate() {
        let mine = || sources.iter().filter(move |s| s.instance == index);
        if mine().all(|s| s.edge_effects.is_empty()) {
            continue;
        }
        let member = &instance.component.name;
        // Before any edge write lands: the type's own writers, and the
        // level functions of the held effects.
        let written = writers(&instance.component);
        // Per attribute, the group combinations writing it and the edges
        // they write it on.
        let mut through_groups: BTreeMap<&str, (BTreeSet<String>, bool, bool)> = BTreeMap::new();
        for source in mine() {
            for (e, effect) in source.edge_effects.iter().enumerate() {
                let here = source.effects_origin.at(format!(
                    "{}/edge_effects/{e}",
                    source.effects_origin.pointer
                ));
                let fail = |message: String| {
                    error(
                        &here,
                        format!(
                            "mode `{}` of type `{}` (instance `{member}`): {message}",
                            source.mode, instance.type_name
                        ),
                    )
                };
                if !instance
                    .targetable
                    .iter()
                    .any(|(t, _)| t.attribute == effect.attribute)
                {
                    return Err(fail(format!(
                        "attribute `{}` is not targetable, so no mode may write it; declare it under `targetable`",
                        effect.attribute
                    )));
                }
                let Some(attribute) = instance
                    .component
                    .attributes
                    .iter()
                    .find(|a| a.name == effect.attribute)
                else {
                    return Err(fail(format!(
                        "the instance has no attribute `{}`",
                        effect.attribute
                    )));
                };
                let value = written_value(attribute.kind, &effect.value).ok_or_else(|| {
                    fail(format!(
                        "attribute `{}` cannot take the written value `{}`",
                        effect.attribute, effect.value
                    ))
                })?;
                let held = format!("modes__{}", effect.attribute);
                if let Some((_, writer)) = written.iter().find(|(a, _)| a == &effect.attribute) {
                    let by_a_mode = instance
                        .component
                        .sensitive_functions
                        .iter()
                        .any(|f| f.name == held);
                    return Err(fail(if by_a_mode {
                        format!(
                            "attribute `{}` is held by a mode's effect; an attribute written on an edge is written by edges only",
                            effect.attribute
                        )
                    } else {
                        format!(
                            "attribute `{}` is also written by the {writer} of the type; an attribute written on an edge is written by edges only",
                            effect.attribute
                        )
                    }));
                }
                if source.component != *member {
                    // Over combinations overlapping on this member, one
                    // combination's repair write would undo what another,
                    // still occurred, wrote: refused, as the declaration
                    // route refuses it.
                    let (automata, occurrence, repair) =
                        through_groups.entry(effect.attribute.as_str()).or_default();
                    automata.insert(format!("{}.{}", source.component, source.automaton));
                    match effect.on {
                        Edge::Occurrence => *occurrence = true,
                        Edge::Repair => *repair = true,
                    }
                    if *occurrence && *repair && automata.len() > 1 {
                        return Err(fail(format!(
                            "attribute `{}` is written on both edges by overlapping combinations ({}): one combination's repair would undo what another, still occurred, wrote",
                            effect.attribute,
                            automata.iter().map(String::as_str).collect::<Vec<_>>().join(", ")
                        )));
                    }
                }
                writes.push(Write {
                    component: source.component.clone(),
                    automaton: source.automaton.clone(),
                    edge: effect.on,
                    member: member.clone(),
                    attribute: effect.attribute.clone(),
                    value,
                    mode: source.mode.clone(),
                    origin: here,
                });
            }
        }
    }
    for write in writes {
        let automaton = match instances
            .iter_mut()
            .find(|i| i.component.name == write.component)
        {
            Some(instance) => find(&mut instance.component.automata, &write.automaton),
            None => groups
                .iter_mut()
                .find(|g| g.name == write.component)
                .and_then(|g| find(&mut g.automata, &write.automaton)),
        };
        let edge_name = match write.edge {
            Edge::Occurrence => OCC,
            Edge::Repair => REP,
        };
        let Some(transition) =
            automaton.and_then(|a| a.transitions.iter_mut().find(|t| t.name == edge_name))
        else {
            return Err(error(
                &write.origin,
                format!(
                    "mode `{}` (instance `{}`): it writes `{}` on its `{edge_name}` edge, which it does not have (a repair write needs a repair law)",
                    write.mode, write.member, write.attribute
                ),
            ));
        };
        let written = Expr::Const { value: write.value };
        // A draw that may park writes only on the branch entering `occ`:
        // effects apply after the state change, so the guard reads it.
        let value = if transition.targets.len() > 1 {
            Expr::If {
                cond: Box::new(Expr::StateActive {
                    state: StateRef {
                        component: write.component.clone(),
                        automaton: write.automaton.clone(),
                        state: OCC.to_owned(),
                    },
                }),
                then: Box::new(written),
                otherwise: Box::new(Expr::Attr {
                    attr: AttrRef {
                        component: write.member.clone(),
                        attribute: write.attribute.clone(),
                    },
                }),
            }
        } else {
            written
        };
        transition.effects.push(Assignment {
            target: AttrRef {
                component: write.member,
                attribute: write.attribute,
            },
            value,
        });
    }
    Ok(())
}

fn find<'a>(automata: &'a mut [Automaton], name: &str) -> Option<&'a mut Automaton> {
    automata.iter_mut().find(|a| a.name == name)
}
