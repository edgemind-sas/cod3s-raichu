//! Native **importance measures**: which component contributes most to a
//! feared event, and where reliability investment pays.
//!
//! After the probability of the feared event, the next question of a
//! safety study is always the same one, and simulation engines of this
//! family do not answer it: the measures are rebuilt by hand, downstream,
//! once per study, which makes them costly, irreproducible and
//! unverifiable. This module computes them natively, from **one**
//! Monte-Carlo campaign and nothing else.
//!
//! # What it rests on
//!
//! The support is RAICHU's native [minimal sequences](crate::sequence):
//! run a sequence-recording campaign, truncate every trajectory at the
//! first occurrence of the feared event, and the canonical pipeline
//! (`group → filter cycles → minimal`) returns the irreducible ordered
//! chains that reach it. Dropped to sets and de-duplicated, those chains
//! are the **minimal cut sets** of the system, and the cut sets are the
//! structure function:
//!
//! ```text
//! Φ(D) = 1  ⟺  some minimal cut K satisfies K ⊆ D
//! ```
//!
//! where `D` is the set of basic events (monitored failure states) active
//! at an instant. The same campaign also carries `D` itself: a recorded
//! sequence is the entry dates of every monitored state, so the state of
//! every failure mode at any instant is a replay of its own trace, with
//! no second campaign and no re-simulation.
//!
//! Having both `Φ` and the empirical joint distribution of `D`, the
//! measures are computed **by their definitions**, not by the rare-event
//! sum every fault-tree tool falls back on:
//!
//! - **Birnbaum** `I^B_i(t) = E[Φ(D ∪ E_i) − Φ(D \ E_i)]`, the probability
//!   that the system is *critical* for component `i`: it fails if `i`
//!   fails and holds if `i` holds. This is the pivotal definition, so it
//!   needs no independence assumption between components, which a
//!   model with common-cause failures violates by construction.
//! - **Fussell-Vesely** `FV_i(t) = P(some realized cut contains an event
//!   of i | system down)`, the share of the risk that passes through `i`.
//! - **Criticality** `I^C_i = I^B_i · q_i / Q`, Birnbaum re-weighted by how
//!   likely the component is to be the one that is down.
//! - The two **pivotal risk levels** `Q⁺_i = E[Φ(D ∪ E_i)]` and
//!   `Q⁻_i = E[Φ(D \ E_i)]`: the risk with the component certainly failed
//!   and with it made perfect. Risk-achievement worth is `Q⁺_i / Q` and
//!   risk-reduction worth is `Q / Q⁻_i`; they are left to the caller as
//!   one division rather than reported, because both denominators
//!   legitimately reach zero and a ratio would have to encode infinity in
//!   the wire format.
//!
//! # What it costs
//!
//! One campaign, plus a reduction in
//! `O(replicas × instants × cuts × cut size)` word operations over a
//! bitset. The campaign dominates: the measures are a post-processing
//! pass, never a second run, which is what keeps them usable at the load
//! levels a study is planned for.
//!
//! # What it assumes
//!
//! - The **cut corpus is empirical**. A path no replica walked is not in
//!   it, and a component that only fails through such a path reads as
//!   unimportant. [`ImportanceAnalysis::q_target`] against
//!   [`ImportanceAnalysis::q_cuts`] measures exactly that gap: the first
//!   is the feared event as the trajectories recorded it, the second is
//!   the same instants judged by the reconstructed structure. They agree
//!   when the corpus is complete.
//! - A component is **failed** when any of the basic events the cut
//!   structure attributes to it is active, and `Φ(D ∪ E_i)` sets all of
//!   them at once. For a component with one failure mode, the usual case,
//!   the two readings coincide; for one with several, component-level
//!   Birnbaum answers "what if this component fails, however it fails".
//! - A cut with **no** basic event says the feared event was reached with
//!   nothing recorded as failed, so the structure reads the system as
//!   permanently down and every importance falls to zero. That is a model
//!   or annotation problem rather than a result, and the gap between
//!   [`ImportanceAnalysis::q_cuts`] and [`ImportanceAnalysis::q_target`]
//!   is where it shows.
//! - A monitored state is left when another state of the same
//!   `cycle_group` is entered. A monitored transition with no cycle group
//!   is a state that is entered and never left (which is what a
//!   non-repairable failure is).

use std::collections::HashMap;

use serde::Serialize;

use crate::compile::CompiledModel;
use crate::engine::Sequence;
use crate::sequence::analyse;

/// One **basic event** of the cut structure: a monitored failure state,
/// named by its owning component and the state entered.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct BasicEvent {
    /// Owning component (the `obj` of a recorded [`crate::SeqEvent`]).
    pub obj: String,
    /// The monitored state entered (the `attr` of a recorded event).
    pub attr: String,
}

impl BasicEvent {
    /// Build a basic event from its two names.
    pub fn new(obj: impl Into<String>, attr: impl Into<String>) -> Self {
        BasicEvent {
            obj: obj.into(),
            attr: attr.into(),
        }
    }

    /// The `obj.attr` qualified name.
    #[must_use]
    pub fn name(&self) -> String {
        format!("{}.{}", self.obj, self.attr)
    }
}

/// One **minimal cut set**: the basic events whose simultaneous activity
/// is enough to reach the feared event, and irreducibly so.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Cut {
    /// The events of the cut, in ascending `(obj, attr)` order.
    pub events: Vec<BasicEvent>,
    /// Trajectory weight the minimal-sequence corpus attributed to it:
    /// how many replicas reached the feared event through this cut. The
    /// orders of one cut (`A then B`, `B then A`) are distinct minimal
    /// *sequences* and one cut *set*, so their weights are summed here.
    pub weight: f64,
}

/// The importance measures of one component over the schedule. Every
/// series is indexed like [`ImportanceAnalysis::instants`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ComponentImportance {
    /// Component name.
    pub component: String,
    /// The basic events the cut structure attributes to it, as `attr`
    /// state names, sorted.
    pub events: Vec<String>,
    /// `q_i(t)`: probability that at least one of those events is active.
    pub unavailability: Vec<f64>,
    /// Birnbaum importance: probability that the system is critical for
    /// this component (it fails if the component fails, holds if it
    /// holds). The sensitivity of the risk to this component alone.
    pub birnbaum: Vec<f64>,
    /// Fussell-Vesely importance: the share of the feared-event
    /// probability that passes through a cut containing this component.
    pub fussell_vesely: Vec<f64>,
    /// Criticality importance `I^B_i · q_i / Q`: the probability that the
    /// component is down *and* critical, given the system is down.
    pub criticality: Vec<f64>,
    /// `Q⁺_i(t)`: the feared-event probability with this component
    /// certainly failed. Divided by `q_cuts`, the risk-achievement worth.
    pub q_system_failed: Vec<f64>,
    /// `Q⁻_i(t)`: the feared-event probability with this component made
    /// perfect. `q_cuts` divided by it is the risk-reduction worth.
    pub q_system_intact: Vec<f64>,
}

/// A full importance analysis of one feared event.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ImportanceAnalysis {
    /// The feared event the measures are relative to.
    pub target: BasicEvent,
    /// The instants every series is sampled at, ascending.
    pub instants: Vec<f64>,
    /// The feared-event probability as the **recorded trajectories** have
    /// it: the share of replicas whose target state is active at each
    /// instant.
    pub q_target: Vec<f64>,
    /// The feared-event probability as the **reconstructed cut
    /// structure** has it, judged on the same recorded states. A gap with
    /// [`Self::q_target`] is the corpus-completeness diagnostic: a cut no
    /// replica walked cannot be in the structure, and the structure then
    /// reads the system as up where the trajectory recorded it down.
    pub q_cuts: Vec<f64>,
    /// The minimal cut sets, heaviest first.
    pub cuts: Vec<Cut>,
    /// Per-component measures, most important first (descending
    /// Fussell-Vesely at the last instant, ties by name).
    pub components: Vec<ComponentImportance>,
    /// Total trajectory weight the analysis rests on (one per replica in
    /// a raw corpus).
    pub nb_runs: f64,
    /// Engine version that produced the analysis.
    pub engine_version: String,
}

/// The declared feared events of a compiled model, as
/// `(target name, basic event)`, in declaration order.
///
/// A target names an automaton state; a recorded sequence names the
/// component and the state entered. This is the bridge: the component is
/// read off a transition of the target's automaton, so the pair is
/// exactly what a [`crate::SeqEvent`] would carry when the feared event
/// occurs. A target on an automaton with no transition is unreachable
/// and is reported under its qualified automaton name.
#[must_use]
pub fn target_events(model: &CompiledModel) -> Vec<(String, BasicEvent)> {
    model
        .targets
        .iter()
        .map(|target| {
            let automaton = &model.automata[target.automaton];
            let obj = automaton
                .transitions
                .first()
                .map_or_else(
                    || {
                        automaton
                            .name
                            .rsplit_once('.')
                            .map_or(automaton.name.as_str(), |(c, _)| c)
                    },
                    |&t| model.transitions[t].component.as_str(),
                )
                .to_owned();
            let attr = automaton.states[target.state].clone();
            (target.name.clone(), BasicEvent::new(obj, attr))
        })
        .collect()
}

/// A trajectory truncated at the first occurrence of the feared event, or
/// `None` when it never reached it. Truncation is what turns a
/// free-running campaign into the target-stopped corpus the
/// minimal-sequence pipeline is defined on, without running it twice.
fn truncate(seq: &Sequence, target: &BasicEvent) -> Option<Sequence> {
    let pos = seq
        .events
        .iter()
        .position(|e| e.obj == target.obj && e.attr == target.attr)?;
    Some(Sequence {
        events: seq.events[..=pos].to_vec(),
        end_cause: Some(target.name()),
        end_time: seq.events[pos].time,
        weight: seq.weight,
    })
}

/// Minimal cut **sets** from the minimal cut **sequences**: drop the
/// order, drop the target's own event, merge the orders of one cut, and
/// remove any set that contains another (two incomparable sequences can
/// still nest as sets).
fn cut_sets(minimal: &[Sequence], target: &BasicEvent) -> Vec<Cut> {
    let mut sets: Vec<(Vec<BasicEvent>, f64)> = Vec::new();
    for seq in minimal {
        let mut events: Vec<BasicEvent> = seq
            .events
            .iter()
            .filter(|e| !(e.obj == target.obj && e.attr == target.attr))
            .map(|e| BasicEvent::new(e.obj.clone(), e.attr.clone()))
            .collect();
        events.sort();
        events.dedup();
        match sets.iter_mut().find(|(s, _)| *s == events) {
            Some((_, w)) => *w += seq.weight,
            None => sets.push((events, seq.weight)),
        }
    }
    // Shortest first (a superset can only be dropped once the subset it
    // contains has been kept), ties by descending weight then by content,
    // so the reduction is a pure function of the corpus.
    sets.sort_by(|a, b| {
        a.0.len()
            .cmp(&b.0.len())
            .then(b.1.total_cmp(&a.1))
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut kept: Vec<(Vec<BasicEvent>, f64)> = Vec::new();
    for (events, weight) in sets {
        match kept
            .iter_mut()
            .find(|(k, _)| k.iter().all(|e| events.contains(e)))
        {
            Some((_, w)) => *w += weight,
            None => kept.push((events, weight)),
        }
    }
    kept.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    kept.into_iter()
        .map(|(events, weight)| Cut { events, weight })
        .collect()
}

/// `needle ⊆ haystack` over two equally sized bitsets.
fn covered(needle: &[u64], haystack: &[u64]) -> bool {
    needle.iter().zip(haystack).all(|(n, h)| *n & !*h == 0)
}

/// The two bitsets share at least one bit.
fn meets(left: &[u64], right: &[u64]) -> bool {
    left.iter().zip(right).any(|(l, r)| *l & *r != 0)
}

/// Set bit `bit` of a bitset.
fn set_bit(mask: &mut [u64], bit: usize) {
    mask[bit / 64] |= 1u64 << (bit % 64);
}

/// Compute the importance measures of one feared event over a raw
/// Monte-Carlo sequence corpus.
///
/// `raw` must come from a **free-running** sequence campaign (recording
/// on, target early-stop off): the trajectories have to keep evolving
/// past the feared event for a measure at an instant beyond it to mean
/// anything. `instants` is sorted internally; the returned series follow
/// the sorted order.
///
/// A campaign in which the feared event never occurred returns zeros
/// throughout, with `q_target` and `q_cuts` at zero saying why.
#[must_use]
pub fn importance(raw: &[Sequence], target: &BasicEvent, instants: &[f64]) -> ImportanceAnalysis {
    let mut instants: Vec<f64> = instants.to_vec();
    instants.sort_by(f64::total_cmp);
    let n_instants = instants.len();

    // --- The cut structure, from the target-stopped view of the corpus.
    let truncated: Vec<Sequence> = raw.iter().filter_map(|s| truncate(s, target)).collect();
    let cuts = cut_sets(&analyse(truncated), target);

    // --- Dense indices: one bit per basic event, one mask per cut, one
    // mask per component, in first-seen (hence deterministic) order. The
    // index is keyed by borrowed names so that resolving a recorded event
    // costs a lookup and not two string allocations: the corpus is the
    // one thing here whose size is the campaign's.
    let mut event_order: Vec<BasicEvent> = Vec::new();
    for cut in &cuts {
        for event in &cut.events {
            if !event_order.contains(event) {
                event_order.push(event.clone());
            }
        }
    }
    let index: HashMap<(&str, &str), usize> = event_order
        .iter()
        .enumerate()
        .map(|(bit, event)| ((event.obj.as_str(), event.attr.as_str()), bit))
        .collect();
    let n_events = event_order.len();
    let n_words = n_events.div_ceil(64).max(1);

    let cut_masks: Vec<Vec<u64>> = cuts
        .iter()
        .map(|cut| {
            let mut mask = vec![0u64; n_words];
            for event in &cut.events {
                if let Some(&bit) = index.get(&(event.obj.as_str(), event.attr.as_str())) {
                    set_bit(&mut mask, bit);
                }
            }
            mask
        })
        .collect();

    // Components, sorted by name; each owns the events the cut structure
    // attributes to it.
    let mut names: Vec<String> = event_order.iter().map(|e| e.obj.clone()).collect();
    names.sort();
    names.dedup();
    let comp_masks: Vec<Vec<u64>> = names
        .iter()
        .map(|name| {
            let mut mask = vec![0u64; n_words];
            for (bit, event) in event_order.iter().enumerate() {
                if event.obj == *name {
                    set_bit(&mut mask, bit);
                }
            }
            mask
        })
        .collect();
    let n_comps = names.len();
    // Which components each cut touches, and, per component, the cuts it
    // touches with that component's events removed: `Φ(D ∪ E_i)` is
    // `Φ(D)` unless one of *those* residuals is covered, so the pivotal
    // scan stays proportional to the incidences, not to components × cuts.
    let cut_comps: Vec<Vec<usize>> = cut_masks
        .iter()
        .map(|cut| {
            (0..n_comps)
                .filter(|&c| meets(cut, &comp_masks[c]))
                .collect()
        })
        .collect();
    let residuals: Vec<Vec<Vec<u64>>> = (0..n_comps)
        .map(|c| {
            cut_masks
                .iter()
                .filter(|cut| meets(cut, &comp_masks[c]))
                .map(|cut| {
                    cut.iter()
                        .zip(&comp_masks[c])
                        .map(|(k, m)| *k & !*m)
                        .collect()
                })
                .collect()
        })
        .collect();

    // --- Accumulators, flat and instant-major so the inner loop over
    // components is contiguous. Every fold runs over the replicas in
    // corpus order, so the reduction is byte-reproducible.
    let mut nb_runs = 0.0;
    let mut w_target = vec![0.0; n_instants];
    let mut w_cuts = vec![0.0; n_instants];
    let mut w_plus = vec![0.0; n_instants * n_comps];
    let mut w_minus = vec![0.0; n_instants * n_comps];
    let mut w_down = vec![0.0; n_instants * n_comps];
    let mut w_fv = vec![0.0; n_instants * n_comps];

    // Scratch reused across replicas: the active basic events as a
    // bitset, maintained **incrementally** (an automaton leaves the state
    // it was in when it enters another), and the verdicts of one instant,
    // recomputed only when the trajectory actually moved between two.
    let mut active = vec![0u64; n_words];
    let mut bit_count = vec![0u32; n_events];
    // Per-replica event resolution. The automata of one trajectory are
    // few, so both lookups are linear scans over short `&str` slices
    // rather than hash lookups: hashing a string twice per recorded event
    // was the single heaviest line of the reduction, and a trajectory
    // rarely carries more than a few dozen automata to compare against.
    let mut automata: Vec<(&str, &str)> = Vec::new();
    let mut seen: Vec<Vec<(&str, Option<usize>, bool)>> = Vec::new();
    let mut key_state: Vec<(Option<usize>, bool)> = Vec::new();
    let mut resolved: Vec<(usize, Option<usize>, bool)> = Vec::new();
    let mut touching = vec![0usize; n_comps];
    let mut is_plus = vec![false; n_comps];
    let mut is_minus = vec![false; n_comps];
    let mut is_down = vec![false; n_comps];
    let mut is_fv = vec![false; n_comps];

    for seq in raw {
        nb_runs += seq.weight;
        // Resolve each recorded event once: the automaton it belongs to
        // (its cycle group, or itself when it has none, which is what a
        // state entered and never left looks like), the bit it sets, and
        // whether it is the feared event.
        automata.clear();
        seen.clear();
        resolved.clear();
        for event in &seq.events {
            let key = (
                event.obj.as_str(),
                event.cycle_group.as_deref().unwrap_or(event.attr.as_str()),
            );
            let automaton = match automata.iter().position(|k| *k == key) {
                Some(found) => found,
                None => {
                    automata.push(key);
                    seen.push(Vec::new());
                    automata.len() - 1
                }
            };
            let attr = event.attr.as_str();
            let state = match seen[automaton].iter().position(|(a, _, _)| *a == attr) {
                Some(found) => found,
                None => {
                    let bit = index.get(&(event.obj.as_str(), attr)).copied();
                    let is_target = event.obj == target.obj && attr == target.attr;
                    seen[automaton].push((attr, bit, is_target));
                    seen[automaton].len() - 1
                }
            };
            let (_, bit, is_target) = seen[automaton][state];
            resolved.push((automaton, bit, is_target));
        }
        active.iter_mut().for_each(|w| *w = 0);
        bit_count.iter_mut().for_each(|c| *c = 0);
        key_state.clear();
        key_state.resize(automata.len(), (None, false));
        let mut n_target = 0usize;
        let mut next_event = 0;
        let mut moved = true;
        let mut target_active = false;
        let mut system_down = false;
        for (k, &instant) in instants.iter().enumerate() {
            while let Some(event) = seq.events.get(next_event) {
                if event.time > instant {
                    break;
                }
                let (automaton, bit, is_target) = resolved[next_event];
                let (previous, was_target) = key_state[automaton];
                // A basic event can be reached by more than one automaton
                // of a component, so what a bit tracks is how many of them
                // currently sit in that state, not a single owner.
                if let Some(previous) = previous {
                    bit_count[previous] -= 1;
                    if bit_count[previous] == 0 {
                        active[previous / 64] &= !(1u64 << (previous % 64));
                    }
                }
                n_target -= usize::from(was_target);
                if let Some(bit) = bit {
                    bit_count[bit] += 1;
                    if bit_count[bit] == 1 {
                        active[bit / 64] |= 1u64 << (bit % 64);
                    }
                }
                n_target += usize::from(is_target);
                key_state[automaton] = (bit, is_target);
                next_event += 1;
                moved = true;
            }
            if moved {
                touching.iter_mut().for_each(|c| *c = 0);
                let mut realized = 0usize;
                for (cut, comps) in cut_masks.iter().zip(&cut_comps) {
                    if covered(cut, &active) {
                        realized += 1;
                        for &c in comps {
                            touching[c] += 1;
                        }
                    }
                }
                system_down = realized > 0;
                target_active = n_target > 0;
                for c in 0..n_comps {
                    // Φ(D ∪ E_c): already true when the system is down,
                    // otherwise a cut of `c` that only misses `c`'s events.
                    is_plus[c] = system_down
                        || residuals[c]
                            .iter()
                            .any(|residual| covered(residual, &active));
                    // Φ(D \ E_c): a realized cut that does not pass
                    // through `c` survives the removal, and only those do.
                    is_minus[c] = realized > touching[c];
                    is_down[c] = meets(&active, &comp_masks[c]);
                    is_fv[c] = touching[c] > 0;
                }
                moved = false;
            }
            if target_active {
                w_target[k] += seq.weight;
            }
            if system_down {
                w_cuts[k] += seq.weight;
            }
            let base = k * n_comps;
            for c in 0..n_comps {
                w_plus[base + c] += seq.weight * f64::from(u8::from(is_plus[c]));
                w_minus[base + c] += seq.weight * f64::from(u8::from(is_minus[c]));
                w_down[base + c] += seq.weight * f64::from(u8::from(is_down[c]));
                w_fv[base + c] += seq.weight * f64::from(u8::from(is_fv[c]));
            }
        }
    }

    // --- Derive the measures.
    let ratio = |x: f64| if nb_runs > 0.0 { x / nb_runs } else { 0.0 };
    let q_target: Vec<f64> = w_target.iter().copied().map(ratio).collect();
    let q_cuts: Vec<f64> = w_cuts.iter().copied().map(ratio).collect();
    let column = |flat: &[f64], c: usize| -> Vec<f64> {
        (0..n_instants)
            .map(|k| ratio(flat[k * n_comps + c]))
            .collect()
    };
    let mut components: Vec<ComponentImportance> = (0..n_comps)
        .map(|c| {
            let q_system_failed = column(&w_plus, c);
            let q_system_intact = column(&w_minus, c);
            let unavailability = column(&w_down, c);
            let birnbaum: Vec<f64> = q_system_failed
                .iter()
                .zip(&q_system_intact)
                .map(|(plus, minus)| plus - minus)
                .collect();
            let fussell_vesely: Vec<f64> = (0..n_instants)
                .map(|k| {
                    if w_cuts[k] > 0.0 {
                        w_fv[k * n_comps + c] / w_cuts[k]
                    } else {
                        0.0
                    }
                })
                .collect();
            let criticality: Vec<f64> = (0..n_instants)
                .map(|k| {
                    if q_cuts[k] > 0.0 {
                        birnbaum[k] * unavailability[k] / q_cuts[k]
                    } else {
                        0.0
                    }
                })
                .collect();
            let mut events: Vec<String> = event_order
                .iter()
                .filter(|e| e.obj == names[c])
                .map(|e| e.attr.clone())
                .collect();
            events.sort();
            ComponentImportance {
                component: names[c].clone(),
                events,
                unavailability,
                birnbaum,
                fussell_vesely,
                criticality,
                q_system_failed,
                q_system_intact,
            }
        })
        .collect();
    // Most important first: the ranking the question "where do I invest"
    // is actually asking for.
    components.sort_by(|a, b| {
        let last = |c: &ComponentImportance| c.fussell_vesely.last().copied().unwrap_or(0.0);
        last(b)
            .total_cmp(&last(a))
            .then_with(|| a.component.cmp(&b.component))
    });

    ImportanceAnalysis {
        target: target.clone(),
        instants,
        q_target,
        q_cuts,
        cuts,
        components,
        nb_runs,
        engine_version: env!("CARGO_PKG_VERSION").to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::SeqEvent;

    /// Build a raw trajectory: `events` are `(obj, attr, time)`, each in a
    /// cycle group named after its `obj`, which is the shape the muscadet
    /// plugin emits for a single-mode component.
    fn traj(events: &[(&str, &str, f64)]) -> Sequence {
        Sequence {
            events: events
                .iter()
                .map(|(obj, attr, time)| SeqEvent {
                    obj: (*obj).into(),
                    attr: (*attr).into(),
                    time: *time,
                    cycle_group: Some((*obj).into()),
                })
                .collect(),
            end_cause: None,
            end_time: 10.0,
            weight: 1.0,
        }
    }

    fn target() -> BasicEvent {
        BasicEvent::new("T", "occ")
    }

    fn named(analysis: &ImportanceAnalysis, component: &str) -> ComponentImportance {
        match analysis
            .components
            .iter()
            .find(|c| c.component == component)
        {
            Some(found) => found.clone(),
            None => unreachable!("component `{component}` is absent from the analysis"),
        }
    }

    /// `Φ = A ∨ (B ∧ C)`: one series block and one parallel pair, the
    /// smallest block diagram with two cuts of different sizes.
    fn series_parallel() -> Vec<Sequence> {
        vec![
            traj(&[("A", "occ", 0.5), ("T", "occ", 0.5)]),
            traj(&[("B", "occ", 0.2), ("C", "occ", 0.4), ("T", "occ", 0.4)]),
            traj(&[("B", "occ", 0.2)]),
            traj(&[]),
        ]
    }

    #[test]
    fn cuts_drop_the_order_and_merge_the_permutations() {
        // `[B,C]` and `[C,B]` are two minimal sequences and one cut set;
        // `[B,A]` carries the shorter `[A]` and is absorbed by it.
        let raw = vec![
            traj(&[("A", "occ", 0.5), ("T", "occ", 0.5)]),
            traj(&[("A", "occ", 0.5), ("T", "occ", 0.5)]),
            traj(&[("B", "occ", 0.1), ("A", "occ", 0.5), ("T", "occ", 0.5)]),
            traj(&[("B", "occ", 0.2), ("C", "occ", 0.4), ("T", "occ", 0.4)]),
            traj(&[("C", "occ", 0.2), ("B", "occ", 0.4), ("T", "occ", 0.4)]),
        ];
        let analysis = importance(&raw, &target(), &[1.0]);
        assert_eq!(analysis.cuts.len(), 2);
        assert_eq!(analysis.cuts[0].events, vec![BasicEvent::new("A", "occ")]);
        assert_eq!(analysis.cuts[0].weight, 3.0);
        assert_eq!(
            analysis.cuts[1].events,
            vec![BasicEvent::new("B", "occ"), BasicEvent::new("C", "occ")]
        );
        assert_eq!(analysis.cuts[1].weight, 2.0);
    }

    #[test]
    fn the_pivotal_measures_match_the_hand_computation() {
        // Four replicas, `Φ = A ∨ (B ∧ C)`, read at t = 1 once every
        // failure has happened. The empirical joint is NOT a product of
        // its marginals, which is exactly why the pivotal definition is
        // the one computed: `E[Φ(D ∪ E_i) − Φ(D \ E_i)]` needs no
        // independence between components.
        let analysis = importance(&series_parallel(), &target(), &[1.0]);
        assert_eq!(analysis.q_target, vec![0.5]);
        assert_eq!(analysis.q_cuts, vec![0.5]);

        let a = named(&analysis, "A");
        assert_eq!(a.unavailability, vec![0.25]);
        // 1 − P(B ∧ C) = 1 − 0.25.
        assert_eq!(a.birnbaum, vec![0.75]);
        assert_eq!(a.fussell_vesely, vec![0.5]);
        assert_eq!(a.criticality, vec![0.375]);
        assert_eq!(a.q_system_failed, vec![1.0]);
        assert_eq!(a.q_system_intact, vec![0.25]);

        let b = named(&analysis, "B");
        assert_eq!(b.unavailability, vec![0.5]);
        // P(¬A ∧ C).
        assert_eq!(b.birnbaum, vec![0.25]);
        assert_eq!(b.fussell_vesely, vec![0.5]);

        let c = named(&analysis, "C");
        assert_eq!(c.unavailability, vec![0.25]);
        // P(¬A ∧ B): C is the more critical of the pair here, because B
        // is down more often than C is.
        assert_eq!(c.birnbaum, vec![0.5]);
        assert_eq!(c.fussell_vesely, vec![0.5]);
    }

    #[test]
    fn a_repaired_failure_is_not_active_after_its_repair() {
        // A fails at 1 and is repaired at 2, so the same trajectory reads
        // down at t = 1.5 and up at t = 3: the measures are a function of
        // the instant, not of the trajectory as a whole.
        let raw = vec![
            traj(&[("A", "occ", 1.0), ("T", "occ", 1.0), ("A", "rep", 2.0)]),
            traj(&[]),
        ];
        let analysis = importance(&raw, &target(), &[1.5, 3.0]);
        assert_eq!(analysis.cuts.len(), 1);
        assert_eq!(analysis.cuts[0].events, vec![BasicEvent::new("A", "occ")]);
        let a = named(&analysis, "A");
        assert_eq!(a.unavailability, vec![0.5, 0.0]);
        assert_eq!(analysis.q_cuts, vec![0.5, 0.0]);
        // `{A}` is the only cut, so the system is critical for A in
        // every state: it fails if A fails and holds if A holds,
        // whatever else is going on. Birnbaum is 1 throughout, and that
        // is not a restatement of the unavailability above.
        assert_eq!(a.birnbaum, vec![1.0, 1.0]);
        // Fussell-Vesely is conditioned on a system failure, and there is
        // none at t = 3.
        assert_eq!(a.fussell_vesely, vec![1.0, 0.0]);
    }

    #[test]
    fn a_transient_failure_repaired_before_the_event_leaves_no_cut() {
        // B fails and is repaired before A brings the system down: the
        // cycle filter cancels the pair, so the cut is `{A}` alone and B
        // carries no importance at all.
        let raw = vec![traj(&[
            ("B", "occ", 0.1),
            ("B", "rep", 0.2),
            ("A", "occ", 0.5),
            ("T", "occ", 0.5),
        ])];
        let analysis = importance(&raw, &target(), &[1.0]);
        assert_eq!(analysis.cuts.len(), 1);
        assert_eq!(analysis.cuts[0].events, vec![BasicEvent::new("A", "occ")]);
        assert_eq!(analysis.components.len(), 1);
        assert_eq!(analysis.components[0].component, "A");
    }

    #[test]
    fn components_are_ranked_by_the_share_of_risk_they_carry() {
        // Three replicas down through A, one through the pair: A carries
        // three quarters of the risk and heads the ranking.
        let raw = vec![
            traj(&[("A", "occ", 0.5), ("T", "occ", 0.5)]),
            traj(&[("A", "occ", 0.5), ("T", "occ", 0.5)]),
            traj(&[("A", "occ", 0.5), ("T", "occ", 0.5)]),
            traj(&[("B", "occ", 0.2), ("C", "occ", 0.4), ("T", "occ", 0.4)]),
        ];
        let analysis = importance(&raw, &target(), &[1.0]);
        assert_eq!(analysis.components[0].component, "A");
        assert_eq!(analysis.components[0].fussell_vesely, vec![0.75]);
        assert_eq!(analysis.components[1].fussell_vesely, vec![0.25]);
    }

    #[test]
    fn a_campaign_that_never_reached_the_event_reports_zeros() {
        let raw = vec![traj(&[("A", "occ", 0.5)]), traj(&[])];
        let analysis = importance(&raw, &target(), &[1.0]);
        assert!(analysis.cuts.is_empty());
        assert!(analysis.components.is_empty());
        assert_eq!(analysis.q_target, vec![0.0]);
        assert_eq!(analysis.q_cuts, vec![0.0]);
        assert_eq!(analysis.nb_runs, 2.0);
    }

    #[test]
    fn the_reduction_is_a_pure_function_of_the_corpus() {
        let raw = series_parallel();
        let first = importance(&raw, &target(), &[0.3, 1.0]);
        let second = importance(&raw, &target(), &[1.0, 0.3]);
        // Instants are sorted, so the two calls describe the same
        // schedule and must produce identical bytes.
        assert_eq!(first, second);
    }
}
