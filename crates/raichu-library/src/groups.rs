//! Common-cause groups: a catalogued mode failing several instances
//! together, one automaton per combination of each active order, in a
//! component named after the group.

use crate::combine::EffectSource;
use crate::document::{CommonCauseGroup, StudyDocument};
use crate::error::{pointer_index, pointer_key, DocumentKind, ExpandError};
use crate::instantiate::ExpandedInstance;
use crate::modes::{
    all_of, hand_events_to_step, law, mode_automaton, occurred, step_automaton, OCC, REP,
};
use crate::names::check_name;
use crate::provenance::Origin;
use raichu_expr::{BoolOp, Expr, StateRef};
use raichu_model::{Component, InterruptionPolicy};
use std::collections::BTreeMap;

/// `(component, automaton)`, or `(instance, mode)`.
type ModeKey = (String, String);

/// A group checked against the instances, its combinations listed.
pub(crate) struct Group<'s> {
    spec: &'s CommonCauseGroup,
    pointer: String,
    /// Instance index of each member, in declaration order.
    members: Vec<usize>,
    /// (automaton name, member positions, index of its order in `orders`).
    combinations: Vec<(String, Vec<usize>, usize)>,
}

fn err(pointer: impl Into<String>, message: impl Into<String>) -> ExpandError {
    ExpandError::new(DocumentKind::Study, pointer, message)
}

/// The combinations of `k` positions among `n`, in lexicographic order.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(k);
    fn walk(start: usize, n: usize, k: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            walk(i + 1, n, k, current, out);
            current.pop();
        }
    }
    walk(0, n, k, &mut current, &mut out);
    out
}

/// The automaton name of the combination of member `positions` (0-based).
fn combination_name(positions: &[usize]) -> String {
    let indices: Vec<String> = positions.iter().map(|p| (p + 1).to_string()).collect();
    format!("cc_{}", indices.join("_"))
}

/// The name of the effect step a group grafts on each member.
pub(crate) fn member_step_name(group: &str) -> String {
    format!("{group}__effect")
}

/// Checks every group of `study` and lists its combinations. `individual`
/// holds the `(instance, mode)` pairs the study activates one by one.
pub(crate) fn resolve<'s>(
    study: &'s StudyDocument,
    instances: &[ExpandedInstance],
    individual: &BTreeMap<(String, String), Vec<(String, String)>>,
) -> Result<Vec<Group<'s>>, ExpandError> {
    let mut groups = Vec::new();
    for (g, spec) in study.common_cause_groups.iter().enumerate() {
        let pointer = pointer_index("/study/common_cause_groups", g);
        let name_pointer = pointer_key(&pointer, "name");
        check_name("common-cause group", &spec.name).map_err(|m| err(&name_pointer, m))?;
        if instances.iter().any(|i| i.component.name == spec.name)
            || study.feared_events.iter().any(|e| e.name == spec.name)
            || study.common_cause_groups[..g]
                .iter()
                .any(|o| o.name == spec.name)
        {
            return Err(err(
                name_pointer,
                format!(
                    "common-cause group `{}` takes a name an instance, a feared event or another group already uses",
                    spec.name
                ),
            ));
        }
        let mut members = Vec::with_capacity(spec.instances.len());
        for (m, member) in spec.instances.iter().enumerate() {
            let at = pointer_index(&pointer_key(&pointer, "instances"), m);
            if spec.instances[..m].contains(member) {
                return Err(err(
                    at,
                    format!("group `{}` lists `{member}` twice", spec.name),
                ));
            }
            let index = instances
                .iter()
                .position(|i| &i.component.name == member)
                .ok_or_else(|| {
                    err(
                        &at,
                        format!("group `{}`: no instance `{member}`", spec.name),
                    )
                })?;
            let instance = &instances[index];
            let Some((mode, _)) = instance.catalogue.iter().find(|(c, _)| c.name == spec.mode)
            else {
                return Err(err(
                    at,
                    format!(
                        "group `{}`: mode `{}` is not in the catalogue of type `{}` (instance `{member}`)",
                        spec.name, spec.mode, instance.type_name
                    ),
                ));
            };
            if mode.condition.is_some() {
                return Err(err(
                    pointer_key(&pointer, "mode"),
                    format!(
                        "group `{}`: mode `{}` of type `{}` carries a condition; a common-cause group takes none",
                        spec.name, spec.mode, instance.type_name
                    ),
                ));
            }
            if individual.contains_key(&(member.clone(), spec.mode.clone())) {
                return Err(err(
                    at,
                    format!(
                        "group `{}`: `{member}.{}` also has an individual activation; activate the mode once, in the group or alone",
                        spec.name, spec.mode
                    ),
                ));
            }
            members.push(index);
        }
        if spec.orders.is_empty() {
            return Err(err(
                pointer_key(&pointer, "orders"),
                format!("group `{}` declares no order", spec.name),
            ));
        }
        let mut combinations_of = Vec::new();
        for (o, order) in spec.orders.iter().enumerate() {
            let at = pointer_key(&pointer_index(&pointer_key(&pointer, "orders"), o), "order");
            if order.order == 0 || order.order > members.len() {
                return Err(err(
                    at,
                    format!(
                        "group `{}`: order {} out of 1..={}",
                        spec.name,
                        order.order,
                        members.len()
                    ),
                ));
            }
            if spec.orders[..o].iter().any(|x| x.order == order.order) {
                return Err(err(
                    at,
                    format!("group `{}` declares order {} twice", spec.name, order.order),
                ));
            }
            combinations_of.push((order.order, o));
        }
        combinations_of.sort_unstable();
        let combinations = combinations_of
            .into_iter()
            .flat_map(|(k, o)| {
                combinations(members.len(), k)
                    .into_iter()
                    .map(move |positions| (combination_name(&positions), positions, o))
            })
            .collect();
        groups.push(Group {
            spec,
            pointer,
            members,
            combinations,
        });
    }
    Ok(groups)
}

fn state(component: &str, automaton: &str, name: &str) -> Expr {
    Expr::StateActive {
        state: StateRef {
            component: component.to_owned(),
            automaton: automaton.to_owned(),
            state: name.to_owned(),
        },
    }
}

fn any_of(mut args: Vec<Expr>) -> Expr {
    if args.len() == 1 {
        args.remove(0)
    } else {
        Expr::Bool {
            bool_op: BoolOp::Or,
            args,
        }
    }
}

impl Group<'_> {
    /// `(component, automaton)` of the combinations covering member `m`.
    fn covering(&self, m: usize) -> Vec<(String, String)> {
        self.combinations
            .iter()
            .filter(|(_, positions, _)| positions.contains(&m))
            .map(|(name, _, _)| (self.spec.name.clone(), name.clone()))
            .collect()
    }

    /// The `(member, mode)` pairs the group holds, with the occurred states
    /// a `mode_active` leaf on them reads: the covering combinations.
    pub(crate) fn mode_states(
        &self,
        instances: &[ExpandedInstance],
    ) -> Vec<(ModeKey, Vec<ModeKey>)> {
        self.members
            .iter()
            .enumerate()
            .map(|(m, &index)| {
                (
                    (
                        instances[index].component.name.clone(),
                        self.spec.mode.clone(),
                    ),
                    self.covering(m),
                )
            })
            .collect()
    }

    /// Emits the group: its component of combinations, and on each member
    /// its effect step when the group has one. Returns the component and
    /// the effect sources the members' combination functions read.
    pub(crate) fn emit(
        &self,
        instances: &mut [ExpandedInstance],
    ) -> Result<(Component, Vec<EffectSource>), ExpandError> {
        let spec = self.spec;
        let context = format!("common-cause group `{}`", spec.name);
        let step_name = member_step_name(&spec.name);
        let member_names: Vec<String> = self
            .members
            .iter()
            .map(|&i| instances[i].component.name.clone())
            .collect();
        let origin = Origin::study(self.pointer.clone()).with_mode(&spec.mode);
        let mut automata = Vec::with_capacity(self.combinations.len());
        for (name, positions, o) in &self.combinations {
            let order_pointer = pointer_index(&pointer_key(&self.pointer, "orders"), *o);
            let order = &spec.orders[*o];
            let occurrence = law(
                &order.occurrence,
                DocumentKind::Study,
                &pointer_key(&order_pointer, "occurrence"),
                &context,
                true,
                false, // a group carries no condition
            )?;
            let repair = match &order.repair {
                Some(value) => Some(law(
                    value,
                    DocumentKind::Study,
                    &pointer_key(&order_pointer, "repair"),
                    &context,
                    false,
                    false,
                )?),
                None => None,
            };
            // The mutual lock of PyCATSHOO's mirrors: a combination occurs
            // only with its members' steps at rest, and is repaired only
            // once they have all taken hold.
            let (guard, repair_guard) = if spec.effect_step.is_some() {
                let at = |s: &str| {
                    all_of(
                        positions
                            .iter()
                            .map(|&p| state(&member_names[p], &step_name, s))
                            .collect(),
                    )
                };
                (at(REP), at(OCC))
            } else {
                (None, None)
            };
            let mut automaton = mode_automaton(
                name,
                guard,
                repair_guard,
                occurrence,
                repair,
                InterruptionPolicy::Reset,
            );
            if spec.effect_step.is_some() {
                // The members' steps carry the events, one per member.
                hand_events_to_step(&mut automaton);
            }
            automata.push(automaton);
        }
        let mut sources = Vec::new();
        for (m, &index) in self.members.iter().enumerate() {
            let covering = self.covering(m);
            let instance = &mut instances[index];
            let member = instance.component.name.clone();
            let (effects, edge_effects, effects_origin) = instance
                .catalogue
                .iter()
                .find(|(c, _)| c.name == spec.mode)
                .map(|(c, o)| (c.effects.clone(), c.edge_effects.clone(), o.clone()))
                .unwrap_or_else(|| (Vec::new(), Vec::new(), origin.clone()));
            let readers: Vec<(String, String)> = match spec.effect_step {
                Some(step) => {
                    if instance
                        .component
                        .automata
                        .iter()
                        .any(|a| a.name == step_name)
                    {
                        return Err(err(
                            pointer_key(&self.pointer, "name"),
                            format!(
                                "{context}: instance `{member}` already has an automaton `{step_name}`, the name of the group's effect step"
                            ),
                        ));
                    }
                    let follows = any_of(
                        covering
                            .iter()
                            .map(|(component, automaton)| occurred(component, automaton))
                            .collect(),
                    );
                    instance.component.automata.push(step_automaton(
                        step_name.clone(),
                        follows,
                        step,
                        Some(&spec.name),
                    ));
                    instance.provenance.push((
                        format!("automaton:{member}.{step_name}"),
                        origin.for_instance(&member, &instance.type_name),
                    ));
                    vec![(member.clone(), step_name.clone())]
                }
                None => covering,
            };
            for (component, automaton) in readers {
                sources.push(EffectSource {
                    instance: index,
                    mode: format!("{}.{}", spec.name, automaton),
                    effects: effects.clone(),
                    edge_effects: edge_effects.clone(),
                    component,
                    automaton,
                    effects_origin: effects_origin.clone(),
                });
            }
        }
        let component = Component {
            name: spec.name.clone(),
            attributes: Vec::new(),
            ports: Vec::new(),
            interfaces: Vec::new(),
            automata,
            sensitive_functions: Vec::new(),
            equations: Vec::new(),
            allocations: Vec::new(),
        };
        Ok((component, sources))
    }

    /// Provenance of the group's component and combinations.
    pub(crate) fn provenance(&self) -> Vec<(String, Origin)> {
        let origin = Origin::study(self.pointer.clone()).with_mode(&self.spec.mode);
        let mut out = vec![(format!("component:{}", self.spec.name), origin.clone())];
        for (name, positions, _) in &self.combinations {
            let members: Vec<String> = positions
                .iter()
                .map(|&p| self.spec.instances[p].clone())
                .collect();
            let mut at = origin.clone();
            at.instance = Some(members.join(", "));
            out.push((format!("automaton:{}.{name}", self.spec.name), at));
        }
        out
    }
}
