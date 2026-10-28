//! Application of a study to an expanded system: mode activations, their
//! combination on targeted attributes, feared events, indicators, targets
//! and run settings.

use crate::combine::{combined, EffectSource, Holder};
use crate::document::EffectStep;
use crate::document::{Effect, StudyDocument, SystemModel};
use crate::error::{pointer_index, pointer_key, DocumentKind, ExpandError};
use crate::expand::Expansion;
use crate::groups::Group;
use crate::instantiate::{CombineRule, ExpandedInstance};
use crate::modes::{
    all_of, condition_expr, effect_automaton_name, effect_state, effect_step_automaton,
    hand_events_to_step, law, mode_automaton, rewrite_mode_leaves, ActiveModes, EVENT_AUTOMATON,
    NOT_OCC, OCC, REP,
};
use crate::names::check_name;
use crate::provenance::Origin;
use crate::strict::typed;
use raichu_expr::{Assignment, AttrRef, BoolOp, Expr, Value};
use raichu_model::{
    AttrKind, Automaton, Component, CycleCancellation, Distrib, Indicator, IndicatorTarget,
    InterruptionPolicy, SensitiveFunction, Target, Transition, TransitionKind,
};
use serde_json::Value as Json;

/// How specifically an activation names its instance: an activation on
/// named instances (or a one-off mode) wins over one on a whole type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ActivationLevel {
    Type,
    Instance,
}

/// A mode the study switches on, on one instance.
struct Active {
    instance: usize,
    name: String,
    effects: Vec<Effect>,
    /// (condition, document, pointer) of the catalogue and study conditions.
    conditions: Vec<(Json, DocumentKind, String)>,
    occurrence: (Json, String),
    repair: Option<(Json, String)>,
    /// The repair condition and its pointer.
    repair_condition: Option<(Json, String)>,
    /// The step the effects take after the mode's state, if any.
    effect_step: Option<EffectStep>,
    on_interruption: InterruptionPolicy,
    level: ActivationLevel,
    origin: Origin,
    /// Where the effects were declared, for messages.
    effects_origin: Origin,
}

fn err(pointer: impl Into<String>, message: impl Into<String>) -> ExpandError {
    ExpandError::new(DocumentKind::Study, pointer, message)
}

/// Applies `study` to the expanded `instances` of `system`, and returns the
/// components of its common-cause groups, to place after the instances.
pub(crate) fn apply(
    study: &StudyDocument,
    system: &SystemModel,
    instances: &mut [ExpandedInstance],
    expansion: &mut Expansion,
) -> Result<Vec<Component>, ExpandError> {
    if study.system != system.name {
        return Err(err(
            "/study/system",
            format!(
                "study `{}` applies to system `{}`, not to system `{}`",
                study.name, study.system, system.name
            ),
        ));
    }
    expansion.model.name = format!("{}/{}", system.name, study.name);

    let actives = resolve_activations(study, instances, expansion)?;
    let mut active_names: ActiveModes = actives
        .iter()
        .map(|a| {
            let instance = instances[a.instance].component.name.clone();
            (
                (instance.clone(), a.name.clone()),
                vec![(instance, a.name.clone())],
            )
        })
        .collect();
    let groups = crate::groups::resolve(study, instances, &active_names)?;
    // Two groups may hold the same member's mode: a condition reads every
    // combination covering it, whichever group owns it.
    for group in &groups {
        for (key, states) in group.mode_states(instances) {
            active_names.entry(key).or_default().extend(states);
        }
    }

    for active in &actives {
        emit_mode(active, &active_names, instances)?;
    }
    let mut sources: Vec<EffectSource> = actives
        .iter()
        .map(|active| EffectSource {
            instance: active.instance,
            mode: active.name.clone(),
            effects: active.effects.clone(),
            component: instances[active.instance].component.name.clone(),
            automaton: if active.effect_step.is_some() {
                effect_automaton_name(&active.name)
            } else {
                active.name.clone()
            },
            effects_origin: active.effects_origin.clone(),
        })
        .collect();
    let mut components = Vec::with_capacity(groups.len());
    for group in &groups {
        let (component, group_sources) = group.emit(instances)?;
        sources.extend(group_sources);
        for (key, origin) in group.provenance() {
            expansion.provenance.insert(key, origin);
        }
        components.push(component);
    }
    combine_effects(&sources, instances)?;
    emit_feared_events(study, instances, &active_names, expansion)?;

    for (i, raw) in study.indicators.iter().enumerate() {
        let pointer = pointer_index("/study/indicators", i);
        let indicator: Indicator = typed(raw, DocumentKind::Study, &pointer, "")?;
        if expansion
            .model
            .indicators
            .iter()
            .any(|x| x.name == indicator.name)
        {
            return Err(err(
                pointer,
                format!(
                    "indicator `{}` is already declared by the system",
                    indicator.name
                ),
            ));
        }
        expansion.provenance.insert(
            format!("indicator:{}", indicator.name),
            Origin::study(pointer),
        );
        expansion.model.indicators.push(indicator);
    }
    refuse_group_held_references(study, &groups, instances, &expansion.model.indicators)?;
    for (i, target) in study.targets.iter().enumerate() {
        let pointer = pointer_index("/study/targets", i);
        if expansion
            .model
            .targets
            .iter()
            .any(|x| x.name == target.name)
        {
            return Err(err(
                pointer,
                format!("target `{}` is already declared", target.name),
            ));
        }
        expansion
            .provenance
            .insert(format!("target:{}", target.name), Origin::study(pointer));
        expansion.model.targets.push(target.clone());
    }
    expansion.simulation.clone_from(&study.simulation);
    Ok(components)
}

fn resolve_activations(
    study: &StudyDocument,
    instances: &[ExpandedInstance],
    expansion: &mut Expansion,
) -> Result<Vec<Active>, ExpandError> {
    let mut actives: Vec<Active> = Vec::new();
    for (k, activation) in study.modes.iter().enumerate() {
        let pointer = pointer_index("/study/modes", k);
        let chosen: Vec<usize> = match (&activation.instances, &activation.type_name) {
            (Some(names), None) => {
                let mut out = Vec::new();
                for (j, name) in names.iter().enumerate() {
                    let index = instances
                        .iter()
                        .position(|i| &i.component.name == name)
                        .ok_or_else(|| {
                            err(
                                pointer_index(&pointer_key(&pointer, "instances"), j),
                                format!("no instance `{name}`"),
                            )
                        })?;
                    out.push(index);
                }
                out
            }
            (None, Some(type_name)) => {
                let out: Vec<usize> = instances
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| &i.type_name == type_name)
                    .map(|(index, _)| index)
                    .collect();
                if out.is_empty() {
                    expansion.warnings.push(format!(
                        "study activation {k} of mode `{}` on type `{type_name}` applies to no instance",
                        activation.mode
                    ));
                }
                out
            }
            _ => {
                return Err(err(
                    pointer,
                    "an activation names either `instances` or `type`, exactly one of them",
                ))
            }
        };
        let level = if activation.instances.is_some() {
            ActivationLevel::Instance
        } else {
            ActivationLevel::Type
        };
        for index in chosen {
            let instance = &instances[index];
            let Some((mode, mode_origin)) = instance
                .catalogue
                .iter()
                .find(|(m, _)| m.name == activation.mode)
            else {
                let known: Vec<String> = instance
                    .catalogue
                    .iter()
                    .map(|(m, _)| format!("`{}`", m.name))
                    .collect();
                return Err(err(
                    pointer_key(&pointer, "mode"),
                    format!(
                        "type `{}` (instance `{}`) catalogues no mode `{}` (catalogue: {})",
                        instance.type_name,
                        instance.component.name,
                        activation.mode,
                        if known.is_empty() {
                            "empty".to_owned()
                        } else {
                            known.join(", ")
                        }
                    ),
                ));
            };
            let mut conditions = Vec::new();
            if let Some(c) = &mode.condition {
                conditions.push((
                    c.clone(),
                    DocumentKind::Library,
                    format!("{}/condition", mode_origin.pointer),
                ));
            }
            if let Some(c) = &activation.condition {
                conditions.push((
                    c.clone(),
                    DocumentKind::Study,
                    pointer_key(&pointer, "condition"),
                ));
            }
            let candidate = Active {
                instance: index,
                name: mode.name.clone(),
                effects: mode.effects.clone(),
                conditions,
                occurrence: (
                    activation.occurrence.clone(),
                    pointer_key(&pointer, "occurrence"),
                ),
                repair: activation
                    .repair
                    .clone()
                    .map(|r| (r, pointer_key(&pointer, "repair"))),
                repair_condition: activation
                    .repair_condition
                    .clone()
                    .map(|c| (c, pointer_key(&pointer, "repair_condition"))),
                effect_step: activation.effect_step,
                on_interruption: activation.on_interruption.unwrap_or_default(),
                level,
                origin: Origin::study(pointer.clone())
                    .for_instance(&instance.component.name, &instance.type_name)
                    .with_mode(&mode.name)
                    .with_nature(mode.nature),
                effects_origin: mode_origin.clone(),
            };
            place(&mut actives, candidate, instances, &pointer)?;
        }
    }
    for (k, one_off) in study.one_off_modes.iter().enumerate() {
        let pointer = pointer_index("/study/one_off_modes", k);
        let index = instances
            .iter()
            .position(|i| i.component.name == one_off.instance)
            .ok_or_else(|| {
                err(
                    pointer_key(&pointer, "instance"),
                    format!("no instance `{}`", one_off.instance),
                )
            })?;
        let instance = &instances[index];
        if instance
            .catalogue
            .iter()
            .any(|(m, _)| m.name == one_off.name)
        {
            return Err(err(
                pointer_key(&pointer, "name"),
                format!(
                    "one-off mode `{}` on `{}` collides with the catalogued mode of the same name of type `{}`",
                    one_off.name, one_off.instance, instance.type_name
                ),
            ));
        }
        let origin = Origin::study(pointer.clone())
            .for_instance(&instance.component.name, &instance.type_name)
            .with_mode(&one_off.name)
            .with_nature(one_off.nature);
        let candidate = Active {
            instance: index,
            name: one_off.name.clone(),
            effects: one_off.effects.clone(),
            conditions: one_off
                .condition
                .iter()
                .map(|c| {
                    (
                        c.clone(),
                        DocumentKind::Study,
                        pointer_key(&pointer, "condition"),
                    )
                })
                .collect(),
            occurrence: (
                one_off.occurrence.clone(),
                pointer_key(&pointer, "occurrence"),
            ),
            repair: one_off
                .repair
                .clone()
                .map(|r| (r, pointer_key(&pointer, "repair"))),
            repair_condition: one_off
                .repair_condition
                .clone()
                .map(|c| (c, pointer_key(&pointer, "repair_condition"))),
            effect_step: one_off.effect_step,
            on_interruption: one_off.on_interruption.unwrap_or_default(),
            level: ActivationLevel::Instance,
            effects_origin: origin.at(pointer_key(&pointer, "effects")),
            origin,
        };
        place(&mut actives, candidate, instances, &pointer)?;
    }
    Ok(actives)
}

/// Adds `candidate`, letting an instance-level activation replace a
/// type-level one of the same mode on the same instance, and refusing two
/// activations at the same level.
fn place(
    actives: &mut Vec<Active>,
    candidate: Active,
    instances: &[ExpandedInstance],
    pointer: &str,
) -> Result<(), ExpandError> {
    if let Some(existing) = actives
        .iter_mut()
        .find(|a| a.instance == candidate.instance && a.name == candidate.name)
    {
        if existing.level == candidate.level {
            return Err(err(
                pointer,
                format!(
                    "mode `{}.{}` is activated twice at the same level (also at `{}`)",
                    instances[candidate.instance].component.name,
                    candidate.name,
                    existing.origin.pointer
                ),
            ));
        }
        if candidate.level > existing.level {
            *existing = candidate;
        }
        return Ok(());
    }
    actives.push(candidate);
    Ok(())
}

fn emit_mode(
    active: &Active,
    active_names: &ActiveModes,
    instances: &mut [ExpandedInstance],
) -> Result<(), ExpandError> {
    let instance = &instances[active.instance];
    let instance_name = instance.component.name.clone();
    let public = format!("{instance_name}.{}", active.name);
    let context = format!("mode `{public}`");
    check_name("mode", &active.name)
        .map_err(|m| ExpandError::new(active.origin.document, active.origin.pointer.clone(), m))?;
    if instance
        .component
        .automata
        .iter()
        .any(|a| a.name == active.name)
    {
        return Err(ExpandError::new(
            active.effects_origin.document,
            active.effects_origin.pointer.clone(),
            format!(
                "{context}: type `{}` already has an automaton `{}`; a mode's automaton takes the mode's name",
                instance.type_name, active.name
            ),
        ));
    }
    let mut guards = Vec::new();
    for (condition, document, pointer) in &active.conditions {
        let rewritten = rewrite_mode_leaves(condition, active_names, *document, pointer, &context)?;
        guards.push(condition_expr(&rewritten, *document, pointer, &context)?);
    }
    let occurrence = law(
        &active.occurrence.0,
        DocumentKind::Study,
        &active.occurrence.1,
        &context,
        true,
    )?;
    let repair = match &active.repair {
        Some((value, pointer)) => Some(law(value, DocumentKind::Study, pointer, &context, false)?),
        None => None,
    };
    if active.effect_step.is_some() {
        // The step's mutual lock, as a PyCATSHOO mirror's: the mode occurs
        // again only once its effects have returned to rest, and is repaired
        // only once they have taken hold.
        guards.push(effect_state(&instance_name, &active.name, REP));
    }
    let repair_guard = match &active.repair_condition {
        Some((_, pointer)) if repair.is_none() => {
            return Err(err(
                pointer.clone(),
                format!("{context}: a repair condition needs a repair law"),
            ));
        }
        Some((condition, pointer)) => {
            let rewritten = rewrite_mode_leaves(
                condition,
                active_names,
                DocumentKind::Study,
                pointer,
                &context,
            )?;
            Some(condition_expr(
                &rewritten,
                DocumentKind::Study,
                pointer,
                &context,
            )?)
        }
        None => None,
    };
    let repair_guard = if active.effect_step.is_some() {
        all_of(
            repair_guard
                .into_iter()
                .chain(std::iter::once(effect_state(
                    &instance_name,
                    &active.name,
                    OCC,
                )))
                .collect(),
        )
    } else {
        repair_guard
    };
    let mut automaton = mode_automaton(
        &active.name,
        all_of(guards),
        repair_guard,
        occurrence,
        repair,
        active.on_interruption,
    );
    if active.effect_step.is_some() {
        hand_events_to_step(&mut automaton);
    }
    let instance = &mut instances[active.instance];
    let key = format!("automaton:{instance_name}.{}", active.name);
    instance.provenance.push((key, active.origin.clone()));
    for transition in &automaton.transitions {
        instance.provenance.push((
            format!(
                "transition:{instance_name}.{}.{}",
                active.name, transition.name
            ),
            active.origin.clone(),
        ));
    }
    instance.component.automata.push(automaton);
    if let Some(step) = active.effect_step {
        let name = effect_automaton_name(&active.name);
        if instance.component.automata.iter().any(|a| a.name == name) {
            return Err(ExpandError::new(
                active.origin.document,
                active.origin.pointer.clone(),
                format!(
                    "{context}: type `{}` already has an automaton `{name}`, the name the effect step of the mode takes",
                    instance.type_name
                ),
            ));
        }
        instance.provenance.push((
            format!("automaton:{instance_name}.{name}"),
            active.origin.clone(),
        ));
        // Before the mode's own automaton: at a shared date the step is
        // taken ahead of the mode's zero-delay consequences (a repair at
        // once), as a PyCATSHOO mirror on the target is taken ahead of the
        // mode component's.
        let position = instance.component.automata.len() - 1;
        instance.component.automata.insert(
            position,
            effect_step_automaton(&instance_name, &active.name, step),
        );
    }
    Ok(())
}

/// Every attribute some element of `component` writes, with the writer.
fn writers(component: &Component) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for equation in &component.equations {
        out.push((
            equation.target.clone(),
            format!("equation of `{}`", equation.target),
        ));
    }
    for function in &component.sensitive_functions {
        for effect in &function.effects {
            out.push((
                effect.target.attribute.clone(),
                format!("sensitive function `{}`", function.name),
            ));
        }
    }
    for automaton in &component.automata {
        for transition in &automaton.transitions {
            for effect in &transition.effects {
                out.push((
                    effect.target.attribute.clone(),
                    format!("transition `{}.{}`", automaton.name, transition.name),
                ));
            }
        }
    }
    out
}

fn effect_value(kind: AttrKind, rule: CombineRule, value: &Json) -> Result<Value, String> {
    match (rule, kind) {
        (CombineRule::AnyCuts, AttrKind::Bool) => match value.as_bool() {
            Some(false) => Ok(Value::Bool(false)),
            _ => Err(format!("rule `any_cuts` holds `false` while a mode is occurred; the effect writes `{value}`")),
        },
        (CombineRule::AnyActivates, AttrKind::Bool) => match value.as_bool() {
            Some(true) => Ok(Value::Bool(true)),
            _ => Err(format!("rule `any_activates` holds `true` while a mode is occurred; the effect writes `{value}`")),
        },
        (CombineRule::Min | CombineRule::Max, AttrKind::Float) => value
            .as_f64()
            .map(Value::Float)
            .ok_or_else(|| format!("the attribute is a float; the effect writes `{value}`")),
        (CombineRule::Min | CombineRule::Max, AttrKind::Int) => value
            .as_i64()
            .map(Value::Int)
            .ok_or_else(|| format!("the attribute is an integer; the effect writes `{value}`")),
        (rule, kind) => Err(format!(
            "rule `{}` does not apply to a {} attribute (`any_cuts` and `any_activates` take booleans, `min` and `max` numbers)",
            rule_name(rule),
            kind_name(kind)
        )),
    }
}

fn rule_name(rule: CombineRule) -> &'static str {
    match rule {
        CombineRule::AnyCuts => "any_cuts",
        CombineRule::AnyActivates => "any_activates",
        CombineRule::Min => "min",
        CombineRule::Max => "max",
    }
}

fn kind_name(kind: AttrKind) -> &'static str {
    match kind {
        AttrKind::Bool => "boolean",
        AttrKind::Int => "integer",
        AttrKind::Float => "float",
    }
}

/// The effects active modes hold on one attribute of one instance.
struct AttributeGroup {
    attribute: String,
    rule: CombineRule,
    holders: Vec<Holder>,
    /// Where the first holding effect was declared, for messages.
    first_origin: Origin,
}

fn combine_effects(
    sources: &[EffectSource],
    instances: &mut [ExpandedInstance],
) -> Result<(), ExpandError> {
    for (index, instance) in instances.iter_mut().enumerate() {
        let mut groups: Vec<AttributeGroup> = Vec::new();
        let written = writers(&instance.component);
        for active in sources.iter().filter(|a| a.instance == index) {
            for (e, effect) in active.effects.iter().enumerate() {
                let here = active
                    .effects_origin
                    .at(format!("{}/effects/{e}", active.effects_origin.pointer));
                let fail = |message: String| {
                    ExpandError::new(
                        here.document,
                        here.pointer.clone(),
                        format!(
                            "mode `{}` of type `{}` (instance `{}`): {message}",
                            active.mode, instance.type_name, instance.component.name
                        ),
                    )
                };
                let Some((target, _)) = instance
                    .targetable
                    .iter()
                    .find(|(t, _)| t.attribute == effect.attribute)
                else {
                    return Err(fail(format!(
                        "attribute `{}` declares no combination rule, so no mode may target it; declare it under `targetable` with a rule",
                        effect.attribute
                    )));
                };
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
                let value = effect_value(attribute.kind, target.rule, &effect.value)
                    .map_err(|m| fail(format!("attribute `{}`: {m}", effect.attribute)))?;
                if let Some((_, writer)) = written.iter().find(|(a, _)| a == &effect.attribute) {
                    return Err(fail(format!(
                        "attribute `{}` is also written by the {writer} of the type; a targeted attribute is written by its modes only",
                        effect.attribute
                    )));
                }
                let holder = Holder {
                    mode: active.mode.clone(),
                    component: active.component.clone(),
                    automaton: active.automaton.clone(),
                    value,
                };
                match groups.iter_mut().find(|g| g.attribute == effect.attribute) {
                    Some(group) => group.holders.push(holder),
                    None => groups.push(AttributeGroup {
                        attribute: effect.attribute.clone(),
                        rule: target.rule,
                        holders: vec![holder],
                        first_origin: here,
                    }),
                }
            }
        }
        let name = instance.component.name.clone();
        for AttributeGroup {
            attribute,
            rule,
            holders,
            first_origin: origin,
        } in groups
        {
            let init = instance
                .component
                .attributes
                .iter()
                .find(|a| a.name == attribute)
                .map_or(Value::Bool(false), |a| a.init);
            let function_name = format!("modes__{attribute}");
            if instance
                .component
                .sensitive_functions
                .iter()
                .any(|f| f.name == function_name)
            {
                return Err(ExpandError::new(
                    origin.document,
                    origin.pointer.clone(),
                    format!("instance `{name}`: the type already has a sensitive function `{function_name}`, the name the mode combination of `{attribute}` takes"),
                ));
            }
            let value = combined(rule, &holders, init);
            instance
                .component
                .sensitive_functions
                .push(SensitiveFunction {
                    name: function_name.clone(),
                    effects: vec![Assignment {
                        target: AttrRef {
                            component: name.clone(),
                            attribute: attribute.clone(),
                        },
                        value,
                    }],
                });
            let mut combination_origin = origin.clone();
            combination_origin.mode = Some(
                holders
                    .iter()
                    .map(|h| h.mode.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            instance.provenance.push((
                format!("function:{name}.{function_name}"),
                combination_origin,
            ));
        }
    }
    Ok(())
}

fn emit_feared_events(
    study: &StudyDocument,
    instances: &[ExpandedInstance],
    active_names: &ActiveModes,
    expansion: &mut Expansion,
) -> Result<(), ExpandError> {
    for (i, event) in study.feared_events.iter().enumerate() {
        let pointer = pointer_index("/study/feared_events", i);
        if instances.iter().any(|x| x.component.name == event.name)
            || study.feared_events[..i]
                .iter()
                .any(|e| e.name == event.name)
        {
            return Err(err(
                pointer_key(&pointer, "name"),
                format!(
                    "feared event `{}` takes a name an instance or another event already uses",
                    event.name
                ),
            ));
        }
        let context = format!("feared event `{}`", event.name);
        let condition_pointer = pointer_key(&pointer, "condition");
        let rewritten = rewrite_mode_leaves(
            &event.condition,
            active_names,
            DocumentKind::Study,
            &condition_pointer,
            &context,
        )?;
        let condition = condition_expr(
            &rewritten,
            DocumentKind::Study,
            &condition_pointer,
            &context,
        )?;
        let edge = |name: &str, source: &str, guard: Expr, time: f64| Transition {
            name: name.to_owned(),
            source: source.to_owned(),
            guard: Some(guard),
            targets: vec![name.to_owned()],
            on_interruption: InterruptionPolicy::Reset,
            monitored: true,
            monitored_states: None,
            cycle_group: Some(event.name.clone()),
            cycle_cancellation: CycleCancellation::Auto,
            kind: Some(TransitionKind::Observation),
            effects: Vec::new(),
            distrib: Distrib::Delay { time },
        };
        let not_condition = Expr::Bool {
            bool_op: BoolOp::Not,
            args: vec![condition.clone()],
        };
        let component = Component {
            name: event.name.clone(),
            attributes: Vec::new(),
            ports: Vec::new(),
            interfaces: Vec::new(),
            automata: vec![Automaton {
                name: EVENT_AUTOMATON.to_owned(),
                states: vec![NOT_OCC.to_owned(), OCC.to_owned()],
                init: NOT_OCC.to_owned(),
                transitions: vec![
                    edge(OCC, NOT_OCC, condition, event.tempo_occ),
                    edge(NOT_OCC, OCC, not_condition, event.tempo_not_occ),
                ],
            }],
            sensitive_functions: Vec::new(),
            equations: Vec::new(),
            allocations: Vec::new(),
        };
        let origin = Origin::study(pointer.clone());
        expansion
            .provenance
            .insert(format!("component:{}", event.name), origin.clone());
        expansion.provenance.insert(
            format!("automaton:{}.{EVENT_AUTOMATON}", event.name),
            origin.clone(),
        );
        for t in [OCC, NOT_OCC] {
            expansion.provenance.insert(
                format!("transition:{}.{EVENT_AUTOMATON}.{t}", event.name),
                origin.clone(),
            );
        }
        if event.target {
            if expansion.model.targets.iter().any(|t| t.name == event.name) {
                return Err(err(
                    pointer,
                    format!("target `{}` is already declared", event.name),
                ));
            }
            expansion.model.targets.push(Target {
                name: event.name.clone(),
                component: event.name.clone(),
                automaton: EVENT_AUTOMATON.to_owned(),
                state: OCC.to_owned(),
            });
            expansion
                .provenance
                .insert(format!("target:{}", event.name), origin);
        }
        expansion.model.components.push(component);
    }
    Ok(())
}

/// Refuses a study target naming a group-held mode as `Member.mode.occ`:
/// the member carries no automaton of that name, its combinations live in
/// the group's component.
fn refuse_group_held_references(
    study: &StudyDocument,
    groups: &[Group],
    instances: &[ExpandedInstance],
    indicators: &[Indicator],
) -> Result<(), ExpandError> {
    let held = |component: &str, automaton: &str| {
        groups.iter().any(|g| {
            g.mode_states(instances)
                .iter()
                .any(|((member, mode), _)| member == component && mode == automaton)
        })
    };
    for (i, target) in study.targets.iter().enumerate() {
        if held(&target.component, &target.automaton) {
            return Err(err(
                pointer_index("/study/targets", i),
                format!(
                    "target `{}` names `{}.{}`, a mode a common-cause group holds: its combinations live in the group's component",
                    target.name, target.component, target.automaton
                ),
            ));
        }
    }
    for indicator in indicators {
        if let IndicatorTarget::State {
            component,
            automaton,
            ..
        } = &indicator.target
        {
            if held(component, automaton) {
                let pointer = study
                    .indicators
                    .iter()
                    .position(|raw| {
                        raw.get("name").and_then(|n| n.as_str()) == Some(&indicator.name)
                    })
                    .map_or_else(
                        || "/system/indicators".to_owned(),
                        |i| pointer_index("/study/indicators", i),
                    );
                return Err(err(
                    pointer,
                    format!(
                        "indicator `{}` observes `{component}.{automaton}`, a mode a common-cause group holds: its combinations live in the group's component",
                        indicator.name
                    ),
                ));
            }
        }
    }
    Ok(())
}
