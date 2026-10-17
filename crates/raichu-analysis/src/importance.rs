//! Native **importance measures**: which failure mode, which component
//! and which group of them contributes most to a feared event, and where
//! reliability investment pays.
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
//! A **basic event** is a failure state of one automaton, named by its
//! component, its automaton and the state ([`BasicEvent`]). The automaton
//! is part of the name because one component routinely carries several
//! failure modes that reach a state of the same name: two modes grafted on
//! one pump are two automata, and two basic events.
//!
//! Run a sequence-recording campaign and replay each trajectory up to the
//! first occurrence of the feared event: the basic events active at that
//! instant are a **cut**, a set of failures that was enough to reach it.
//! Merged over the campaign and stripped of every cut that contains
//! another, they are the **minimal cut sets** of the system, and the cut
//! sets are the structure function:
//!
//! ```text
//! Φ(D) = 1  ⟺  some minimal cut K satisfies K ⊆ D
//! ```
//!
//! where `D` is the set of basic events active at an instant. The same
//! campaign also carries `D` itself: a recorded sequence is the entry
//! dates of every monitored state, so the state of every failure mode at
//! any instant is a replay of its own trace, with no second campaign and
//! no re-simulation.
//!
//! Having both `Φ` and the empirical joint distribution of `D`, the
//! measures are computed **by their definitions**, not by the rare-event
//! sum every fault-tree tool falls back on. Each is defined for a **unit**
//! `i`, a set `E_i` of basic events, and reported at three levels: every
//! basic event on its own, every component (the basic events of its
//! automata), and every declared group ([`ImportanceGroup`], typically a
//! physical component whose modes live on other objects, or the targets
//! of a common cause):
//!
//! - **Birnbaum** `I^B_i(t) = E[Φ(D ∪ E_i) − Φ(D \ E_i)]`, the probability
//!   that the system is *critical* for unit `i`: it fails if `i` fails and
//!   holds if `i` holds. This is the pivotal definition, so it needs no
//!   independence assumption between units, which a model with
//!   common-cause failures violates by construction. It does not add up
//!   over the modes of a component, which is why it is computed for the
//!   component itself rather than left to a sum.
//! - **Fussell-Vesely** `FV_i(t) = P(some realized cut meets E_i | system
//!   down)`, the share of the risk that passes through `i`.
//! - **Criticality** `I^C_i = I^B_i · q_i / Q`, Birnbaum re-weighted by how
//!   likely the unit is to be the one that is down.
//! - The two **pivotal risk levels** `Q⁺_i = E[Φ(D ∪ E_i)]` and
//!   `Q⁻_i = E[Φ(D \ E_i)]`: the risk with the unit certainly failed and
//!   with it made perfect. Risk-achievement worth is `Q⁺_i / Q` and
//!   risk-reduction worth is `Q / Q⁻_i`; they are left to the caller as
//!   one division rather than reported, because both denominators
//!   legitimately reach zero and a ratio would have to encode infinity in
//!   the wire format.
//!
//! # What it costs
//!
//! One campaign, plus a reduction that replays each recorded state change
//! once, touching only the units and cuts that hold the changed basic
//! event, then reads a few counters per unit at each instant. A unit that
//! meets no cut (a declared group of components outside every cut) costs
//! one counter per instant; one that meets cuts costs its incidences with
//! them. Nothing scales with the number of basic events of the model, so
//! one group per component stays a small share of the reduction at ten
//! thousand components. The campaign dominates: the measures are a
//! post-processing pass, never a second run, which is what keeps them
//! usable at the load levels a study is planned for. What the reduction holds is one compact
//! record per state change of an automaton that carries a basic event
//! ([`ImportanceReducer`]), never the trajectories themselves.
//!
//! # What it assumes
//!
//! - The **basic events are declared**, never guessed: which states count
//!   as failures is the caller's list ([`ImportanceReducer::new`]), which a
//!   model derives from its declared failure transitions
//!   ([`basic_events`]). An observer, a feared event in the middle of the
//!   model or a parked draw is a recorded state and not a failure, and
//!   counting it as one would put it in the cuts.
//! - The **cut corpus is empirical**. A path no replica walked is not in
//!   it, and a unit that only fails through such a path reads as
//!   unimportant. [`ImportanceAnalysis::q_target`] against
//!   [`ImportanceAnalysis::q_cuts`] measures exactly that gap: the first
//!   is the feared event as the trajectories recorded it, the second is
//!   the same instants judged by the reconstructed structure. They agree
//!   when the corpus is complete.
//! - A unit is **failed** when any of its basic events is active, and
//!   `Φ(D ∪ E_i)` sets all of them at once. For a component with one
//!   failure mode the two readings coincide; for one with several,
//!   component-level Birnbaum answers "what if this component fails,
//!   however it fails".
//! - A cut with **no** basic event says the feared event was reached with
//!   nothing recorded as failed, so the structure reads the system as
//!   permanently down and every importance falls to zero. That is a model
//!   or annotation problem rather than a result, and the gap between
//!   [`ImportanceAnalysis::q_cuts`] and [`ImportanceAnalysis::q_target`]
//!   is where it shows.
//! - The state of an automaton is the last state its recorded events
//!   entered. A failure state must therefore be left through a recorded
//!   transition: one left silently would read as active until the
//!   automaton records its next entry.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use raichu_core::compile::CompiledModel;
use raichu_core::Sequence;
use raichu_model::TransitionKind;

/// One **basic event**: a failure state of one automaton, named by its
/// owning component, the automaton and the state entered.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BasicEvent {
    /// Owning component (the `obj` of a recorded [`raichu_core::SeqEvent`]).
    pub obj: String,
    /// The automaton of `obj`, by its name inside the component (the
    /// `automaton` of a recorded event).
    pub automaton: String,
    /// The state entered (the `attr` of a recorded event).
    pub attr: String,
}

impl BasicEvent {
    /// Build a basic event from its three names.
    pub fn new(
        obj: impl Into<String>,
        automaton: impl Into<String>,
        attr: impl Into<String>,
    ) -> Self {
        BasicEvent {
            obj: obj.into(),
            automaton: automaton.into(),
            attr: attr.into(),
        }
    }

    /// The `obj.automaton.attr` qualified name.
    #[must_use]
    pub fn name(&self) -> String {
        format!("{}.{}.{}", self.obj, self.automaton, self.attr)
    }
}

/// A declared **group** of basic events, measured as one unit: a physical
/// component whose failure modes are carried by other objects, the
/// targets of a common cause, a subsystem. A basic event may belong to
/// several groups; a common cause belongs to each of its targets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportanceGroup {
    /// Group name, unique among the groups of one analysis.
    pub name: String,
    /// Its basic events.
    pub events: Vec<BasicEvent>,
}

/// Which recorded states of a model are basic events ([`basic_events`]).
#[derive(Debug, Clone, PartialEq, Default)]
pub enum BasicEventSelection {
    /// The states the model's **failure** transitions enter (declared
    /// `kind: failure`, recorded): failure modes, never observers, feared
    /// events or the park of a lost draw. The default, and the reading
    /// that keeps an intermediate event out of the cuts.
    #[default]
    Failures,
    /// Every recorded state that is not its automaton's initial state, for
    /// a model that declares no reliability role: everything the
    /// trajectories record, observers included.
    Monitored,
    /// An explicit list. Each must be a state the model records.
    Listed(Vec<BasicEvent>),
}

/// Why an importance analysis cannot be set up.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ImportanceError {
    /// The selection found no basic event at all.
    #[error(
        "the model declares no recorded failure transition (`kind: \"failure\"` on a \
         `monitored` transition), so no state can be a basic event: declare the \
         failure roles, select every recorded state, or list the basic events"
    )]
    NoFailureTransition,
    /// A listed basic event is not a state the model records.
    #[error(
        "`{event}` is not a state the model records: a basic event is entered by a \
         `monitored` transition"
    )]
    NotRecorded {
        /// `obj.automaton.attr`.
        event: String,
    },
    /// A group names a basic event outside the selection.
    #[error("group `{group}` names `{event}`, which is not a basic event of the analysis")]
    UnknownGroupEvent {
        /// The group.
        group: String,
        /// `obj.automaton.attr`.
        event: String,
    },
    /// Two groups share a name.
    #[error("two groups are named `{group}`")]
    DuplicateGroup {
        /// The repeated name.
        group: String,
    },
    /// A group lists no basic event.
    #[error("group `{group}` lists no basic event")]
    EmptyGroup {
        /// The group.
        group: String,
    },
}

/// One **minimal cut set**: the basic events whose simultaneous activity
/// is enough to reach the feared event, and irreducibly so.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Cut {
    /// The events of the cut, in ascending `(obj, automaton, attr)` order.
    pub events: Vec<BasicEvent>,
    /// Trajectory weight attributed to it: how many replicas reached the
    /// feared event with this cut realized (a replica whose realized set
    /// strictly contains a minimal cut counts for that cut).
    pub weight: f64,
}

/// The measure series of one unit. Every series is indexed like
/// [`ImportanceAnalysis::instants`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measures {
    /// `q_i(t)`: probability that at least one basic event of the unit is
    /// active.
    pub unavailability: Vec<f64>,
    /// Birnbaum importance: probability that the system is critical for
    /// this unit (it fails if the unit fails, holds if it holds). The
    /// sensitivity of the risk to this unit alone.
    pub birnbaum: Vec<f64>,
    /// Fussell-Vesely importance: the share of the feared-event
    /// probability that passes through a cut meeting this unit.
    pub fussell_vesely: Vec<f64>,
    /// Criticality importance `I^B_i · q_i / Q`: the probability that the
    /// unit is down *and* critical, given the system is down.
    pub criticality: Vec<f64>,
    /// `Q⁺_i(t)`: the feared-event probability with this unit certainly
    /// failed. Divided by `q_cuts`, the risk-achievement worth.
    pub q_system_failed: Vec<f64>,
    /// `Q⁻_i(t)`: the feared-event probability with this unit made
    /// perfect. `q_cuts` divided by it is the risk-reduction worth.
    pub q_system_intact: Vec<f64>,
}

impl Measures {
    fn last_fussell_vesely(&self) -> f64 {
        self.fussell_vesely.last().copied().unwrap_or(0.0)
    }
}

/// The importance measures of one basic event: one failure mode.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EventImportance {
    /// The basic event.
    pub event: BasicEvent,
    /// Its measures.
    #[serde(flatten)]
    pub measures: Measures,
}

/// The importance measures of one component: every basic event its
/// automata carry, as one unit.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ComponentImportance {
    /// Component name.
    pub component: String,
    /// The basic events the cut structure attributes to it, as
    /// `automaton.attr` names, sorted.
    pub events: Vec<String>,
    /// Its measures.
    #[serde(flatten)]
    pub measures: Measures,
}

/// The importance measures of one declared [`ImportanceGroup`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GroupImportance {
    /// Group name.
    pub group: String,
    /// Its basic events, sorted.
    pub events: Vec<BasicEvent>,
    /// Its measures.
    #[serde(flatten)]
    pub measures: Measures,
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
    /// Per-basic-event measures, for the basic events of the cuts, most
    /// important first (descending Fussell-Vesely at the last instant,
    /// ties by name).
    pub basic_events: Vec<EventImportance>,
    /// Per-component measures, for the components of the cuts, ranked the
    /// same way.
    pub components: Vec<ComponentImportance>,
    /// Per-group measures, one per declared group, ranked the same way.
    pub groups: Vec<GroupImportance>,
    /// Total trajectory weight the analysis rests on (one per replica in
    /// a raw corpus).
    pub nb_runs: f64,
    /// Engine version that produced the analysis.
    pub engine_version: String,
}

/// The name of an automaton inside its component.
fn local_name<'a>(qualified: &'a str, component: &str) -> &'a str {
    qualified
        .strip_prefix(component)
        .and_then(|rest| rest.strip_prefix('.'))
        .unwrap_or(qualified)
}

/// The declared feared events of a compiled model, as
/// `(target name, basic event)`, in declaration order.
///
/// A target names an automaton state; a recorded sequence names the
/// component, the automaton and the state entered. This is the bridge:
/// the component is read off a transition of the target's automaton, so
/// the triple is exactly what a [`raichu_core::SeqEvent`] would carry when
/// the feared event occurs. A target on an automaton with no transition is
/// unreachable and is reported under its qualified automaton name.
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
            let local = local_name(&automaton.name, &obj).to_owned();
            let attr = automaton.states[target.state].clone();
            (target.name.clone(), BasicEvent::new(obj, local, attr))
        })
        .collect()
}

/// The basic events of a compiled model under `selection`, sorted, the
/// feared event `target` excluded.
///
/// # Errors
/// [`ImportanceError::NoFailureTransition`] when the selection finds
/// nothing; [`ImportanceError::NotRecorded`] for a listed event the model
/// never records.
pub fn basic_events(
    model: &CompiledModel,
    selection: &BasicEventSelection,
    target: &BasicEvent,
) -> Result<Vec<BasicEvent>, ImportanceError> {
    // Every recorded state, with whether a failure transition enters it
    // and whether it is its automaton's initial (rest) state.
    let mut recorded: HashMap<BasicEvent, (bool, bool)> = HashMap::new();
    for transition in &model.transitions {
        let automaton = &model.automata[transition.automaton];
        let local = local_name(&automaton.name, &transition.component);
        for (branch, &state) in transition.targets.iter().enumerate() {
            if !transition.records(state) {
                continue;
            }
            let event = BasicEvent::new(
                transition.component.clone(),
                local,
                automaton.states[state].clone(),
            );
            // A declared role applies to the first target only: the other
            // branches of a draw are not failures.
            let failure = branch == 0 && transition.kind == Some(TransitionKind::Failure);
            let rest = state == automaton.init;
            let entry = recorded.entry(event).or_insert((false, rest));
            entry.0 |= failure;
        }
    }
    let mut events: Vec<BasicEvent> = match selection {
        BasicEventSelection::Failures => {
            let found: Vec<BasicEvent> = recorded
                .iter()
                .filter(|(_, &(failure, _))| failure)
                .map(|(event, _)| event.clone())
                .collect();
            if found.is_empty() {
                return Err(ImportanceError::NoFailureTransition);
            }
            found
        }
        BasicEventSelection::Monitored => recorded
            .iter()
            .filter(|(_, &(_, rest))| !rest)
            .map(|(event, _)| event.clone())
            .collect(),
        BasicEventSelection::Listed(listed) => {
            for event in listed {
                if !recorded.contains_key(event) {
                    return Err(ImportanceError::NotRecorded {
                        event: event.name(),
                    });
                }
            }
            listed.clone()
        }
    };
    events.retain(|event| event != target);
    events.sort();
    events.dedup();
    Ok(events)
}

/// "No basic event" in a compact record.
const NO_BIT: u32 = u32::MAX;
/// "The feared event" in a compact record.
const TARGET_BIT: u32 = u32::MAX - 1;

/// One recorded state change, as the reduction keeps it: the date, the
/// automaton (an index into [`ImportanceReducer`]'s table) and what its
/// new state is (a basic-event bit, [`TARGET_BIT`] or [`NO_BIT`]).
#[derive(Debug, Clone, Copy)]
struct Change {
    time: f64,
    automaton: u32,
    bit: u32,
}

/// One trajectory, as the reduction keeps it.
#[derive(Debug, Clone)]
struct Trajectory {
    weight: f64,
    changes: Vec<Change>,
}

/// One automaton the reduction follows: its index, and what each of its
/// states means (a basic-event bit or [`TARGET_BIT`]; any other state is
/// [`NO_BIT`]).
#[derive(Debug, Default)]
struct Slot {
    index: u32,
    states: HashMap<String, u32>,
}

/// The importance reduction, built one trajectory at a time.
///
/// Each pushed trajectory is cut down to the state changes of the automata
/// that carry a basic event or the feared event, and its realized cut (the
/// basic events active at its first feared event) is tallied at once. What
/// it holds is therefore a few words per relevant state change, never the
/// trajectory's names, so a campaign of any size reduces in memory
/// proportional to what the measures actually read. Pushing the
/// trajectories in replica order makes the result a pure function of the
/// corpus.
#[derive(Debug)]
pub struct ImportanceReducer {
    target: BasicEvent,
    instants: Vec<f64>,
    events: Vec<BasicEvent>,
    groups: Vec<ImportanceGroup>,
    /// The automata that matter, `obj` → automaton name → slot. Nested so
    /// a recorded event resolves on its borrowed names, with no allocation.
    automata: HashMap<String, HashMap<String, Slot>>,
    n_automata: usize,
    trajectories: Vec<Trajectory>,
    /// Realized cuts, as sorted bit lists, in first-seen order.
    realized: Vec<(Vec<u32>, f64)>,
    realized_index: HashMap<Vec<u32>, usize>,
    nb_runs: f64,
}

impl ImportanceReducer {
    /// An empty reduction of the feared event `target` over the basic
    /// events `events`, reporting the declared `groups` too, at `instants`
    /// (sorted internally).
    ///
    /// # Errors
    /// [`ImportanceError`] for a group that is empty, named twice, or
    /// names an event outside `events`.
    pub fn new(
        target: BasicEvent,
        events: Vec<BasicEvent>,
        groups: Vec<ImportanceGroup>,
        instants: &[f64],
    ) -> Result<Self, ImportanceError> {
        let mut events = events;
        events.retain(|event| *event != target);
        events.sort();
        events.dedup();
        let bits: HashMap<&BasicEvent, u32> = events
            .iter()
            .enumerate()
            .map(|(bit, e)| (e, u32::try_from(bit).unwrap_or(NO_BIT)))
            .collect();
        let mut names = BTreeSet::new();
        let mut checked = Vec::with_capacity(groups.len());
        for mut group in groups {
            if !names.insert(group.name.clone()) {
                return Err(ImportanceError::DuplicateGroup { group: group.name });
            }
            if group.events.is_empty() {
                return Err(ImportanceError::EmptyGroup { group: group.name });
            }
            if let Some(unknown) = group.events.iter().find(|e| !bits.contains_key(e)) {
                return Err(ImportanceError::UnknownGroupEvent {
                    group: group.name.clone(),
                    event: unknown.name(),
                });
            }
            group.events.sort();
            group.events.dedup();
            checked.push(group);
        }
        let mut automata: HashMap<String, HashMap<String, Slot>> = HashMap::new();
        let mut n_automata = 0usize;
        let entries = bits
            .iter()
            .map(|(e, &bit)| (*e, bit))
            .chain(std::iter::once((&target, TARGET_BIT)));
        for (e, bit) in entries {
            let slot = automata
                .entry(e.obj.clone())
                .or_default()
                .entry(e.automaton.clone())
                .or_insert_with(|| {
                    n_automata += 1;
                    Slot {
                        index: u32::try_from(n_automata - 1).unwrap_or(NO_BIT),
                        states: HashMap::new(),
                    }
                });
            slot.states.insert(e.attr.clone(), bit);
        }
        drop(bits);
        let mut instants = instants.to_vec();
        instants.sort_by(f64::total_cmp);
        Ok(ImportanceReducer {
            target,
            instants,
            events,
            groups: checked,
            automata,
            n_automata,
            trajectories: Vec::new(),
            realized: Vec::new(),
            realized_index: HashMap::new(),
            nb_runs: 0.0,
        })
    }

    /// Add one **free-running** trajectory (recorded to the horizon, not
    /// stopped at the feared event).
    pub fn push(&mut self, sequence: &Sequence) {
        self.nb_runs += sequence.weight;
        let mut changes = Vec::new();
        for event in &sequence.events {
            let Some(slot) = self
                .automata
                .get(event.obj.as_str())
                .and_then(|automata| automata.get(event.automaton.as_str()))
            else {
                continue;
            };
            changes.push(Change {
                time: event.time,
                automaton: slot.index,
                bit: slot
                    .states
                    .get(event.attr.as_str())
                    .copied()
                    .unwrap_or(NO_BIT),
            });
        }
        // The realized cut: the basic events active at the first feared
        // event, its own record included and nothing recorded after it.
        if let Some(first) = changes.iter().position(|c| c.bit == TARGET_BIT) {
            let mut state: HashMap<u32, u32> = HashMap::new();
            for change in &changes[..first] {
                state.insert(change.automaton, change.bit);
            }
            state.remove(&changes[first].automaton);
            let mut cut: Vec<u32> = state
                .into_values()
                .filter(|&bit| bit != NO_BIT && bit != TARGET_BIT)
                .collect();
            cut.sort_unstable();
            match self.realized_index.get(&cut) {
                Some(&slot) => self.realized[slot].1 += sequence.weight,
                None => {
                    self.realized_index.insert(cut.clone(), self.realized.len());
                    self.realized.push((cut, sequence.weight));
                }
            }
        }
        self.trajectories.push(Trajectory {
            weight: sequence.weight,
            changes,
        });
    }

    /// The minimal cut sets: the realized cuts, merged, every superset of
    /// a kept cut absorbed into it.
    fn cut_sets(&self) -> Vec<(Vec<u32>, f64)> {
        let named = |bits: &[u32]| -> Vec<&BasicEvent> {
            bits.iter().map(|&b| &self.events[b as usize]).collect()
        };
        let mut sets: Vec<(Vec<u32>, f64)> = self.realized.clone();
        // Shortest first (a superset can only be dropped once the subset it
        // contains has been kept), ties by descending weight then by
        // content, so the reduction is a pure function of the corpus.
        sets.sort_by(|a, b| {
            a.0.len()
                .cmp(&b.0.len())
                .then(b.1.total_cmp(&a.1))
                .then_with(|| named(&a.0).cmp(&named(&b.0)))
        });
        let mut kept: Vec<(Vec<u32>, f64)> = Vec::new();
        for (bits, weight) in sets {
            match kept
                .iter_mut()
                .find(|(k, _)| k.iter().all(|b| bits.binary_search(b).is_ok()))
            {
                Some((_, w)) => *w += weight,
                None => kept.push((bits, weight)),
            }
        }
        kept.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| named(&a.0).cmp(&named(&b.0)))
        });
        kept
    }

    /// The analysis.
    #[must_use]
    pub fn finish(self) -> ImportanceAnalysis {
        let n_instants = self.instants.len();
        let cut_bits = self.cut_sets();
        let n_events = self.events.len();
        let cuts: Vec<Cut> = cut_bits
            .iter()
            .map(|(bits, weight)| Cut {
                events: bits
                    .iter()
                    .map(|&b| self.events[b as usize].clone())
                    .collect(),
                weight: *weight,
            })
            .collect();

        // --- The units: every basic event of the cuts, every component of
        // the cuts, every declared group. Each is its sorted basic-event
        // bits; `events` is sorted, so a bit is its event's position.
        let in_cuts: BTreeSet<usize> = cut_bits
            .iter()
            .flat_map(|(bits, _)| bits.iter().map(|&b| b as usize))
            .collect();
        // Component name → its bits, ascending, in one pass.
        let mut components: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for &bit in &in_cuts {
            components
                .entry(self.events[bit].obj.as_str())
                .or_default()
                .push(bit);
        }
        let mut unit_bits: Vec<Vec<usize>> = Vec::new();
        for &bit in &in_cuts {
            unit_bits.push(vec![bit]);
        }
        unit_bits.extend(components.values().cloned());
        for group in &self.groups {
            // Every group event is a basic event (checked at construction),
            // and the group's events are sorted, so its bits are too.
            unit_bits.push(
                group
                    .events
                    .iter()
                    .filter_map(|e| self.events.binary_search(e).ok())
                    .collect(),
            );
        }
        let n_units = unit_bits.len();
        let index = UnitIndex::new(n_events, &cut_bits, &unit_bits);
        // A unit that meets no cut is never pivotal through a cut of its
        // own: `Φ(D ∪ E_u)` and `Φ(D \ E_u)` are both `Φ(D)`, and no
        // realized cut meets it. Only its unavailability is followed per
        // replica, so declaring one group per component outside the cuts
        // costs one counter each, not a pass over the cuts.
        let bound: Vec<usize> = (0..n_units).filter(|&u| index.meets_a_cut(u)).collect();

        // --- Accumulators, flat and instant-major so the inner loop over
        // units is contiguous. Every fold runs over the replicas in corpus
        // order, so the reduction is byte-reproducible.
        let mut w_target = vec![0.0; n_instants];
        let mut w_cuts = vec![0.0; n_instants];
        let mut w_plus = vec![0.0; n_instants * n_units];
        let mut w_minus = vec![0.0; n_instants * n_units];
        let mut w_down = vec![0.0; n_instants * n_units];
        let mut w_fv = vec![0.0; n_instants * n_units];

        // Scratch reused across replicas: the state of each relevant
        // automaton, the replay counters, and the verdicts of one instant,
        // recomputed only when the trajectory moved between two.
        let mut state = vec![NO_BIT; self.n_automata];
        let mut replay = Replay::new(&index);
        let mut is_plus = vec![false; n_units];
        let mut is_minus = vec![false; n_units];
        let mut is_fv = vec![false; n_units];

        for trajectory in &self.trajectories {
            state.iter_mut().for_each(|s| *s = NO_BIT);
            replay.reset();
            let mut n_target = 0usize;
            let mut next = 0;
            let mut moved = true;
            let mut target_active = false;
            let mut system_down = false;
            for (k, &instant) in self.instants.iter().enumerate() {
                while let Some(change) = trajectory.changes.get(next) {
                    if change.time > instant {
                        break;
                    }
                    let slot = change.automaton as usize;
                    let previous = state[slot];
                    // Each basic event belongs to exactly one automaton, so
                    // leaving a state deactivates its event outright.
                    if previous == TARGET_BIT {
                        n_target -= 1;
                    } else if previous != NO_BIT {
                        replay.toggle(previous as usize, false);
                    }
                    if change.bit == TARGET_BIT {
                        n_target += 1;
                    } else if change.bit != NO_BIT {
                        replay.toggle(change.bit as usize, true);
                    }
                    state[slot] = change.bit;
                    next += 1;
                    moved = true;
                }
                if moved {
                    system_down = replay.realized > 0;
                    target_active = n_target > 0;
                    for &u in &bound {
                        is_plus[u] = system_down || replay.pivotal_with(u);
                        // Φ(D \ E_u): a realized cut that does not meet `u`
                        // survives the removal, and only those do.
                        is_minus[u] = replay.realized > replay.touching[u] as usize;
                        is_fv[u] = replay.touching[u] > 0;
                    }
                    moved = false;
                }
                let weight = trajectory.weight;
                if target_active {
                    w_target[k] += weight;
                }
                if system_down {
                    w_cuts[k] += weight;
                }
                // Only a true verdict adds: `x + 0.0` is `x` for these
                // non-negative sums, so skipping the false ones leaves every
                // sum bit-identical.
                let base = k * n_units;
                for &u in &bound {
                    if is_plus[u] {
                        w_plus[base + u] += weight;
                    }
                    if is_minus[u] {
                        w_minus[base + u] += weight;
                    }
                    if is_fv[u] {
                        w_fv[base + u] += weight;
                    }
                }
                for (u, &active) in replay.down.iter().enumerate() {
                    if active > 0 {
                        w_down[base + u] += weight;
                    }
                }
            }
        }
        // The units that meet no cut: both pivotal terms are `Φ(D)`, summed
        // in the same replica order as `w_cuts`, hence equal to it.
        for u in (0..n_units).filter(|&u| !index.meets_a_cut(u)) {
            for (k, &cuts) in w_cuts.iter().enumerate() {
                w_plus[k * n_units + u] = cuts;
                w_minus[k * n_units + u] = cuts;
            }
        }

        // --- Derive the measures.
        let nb_runs = self.nb_runs;
        let ratio = |x: f64| if nb_runs > 0.0 { x / nb_runs } else { 0.0 };
        let q_target: Vec<f64> = w_target.iter().copied().map(ratio).collect();
        let q_cuts: Vec<f64> = w_cuts.iter().copied().map(ratio).collect();
        let column = |flat: &[f64], u: usize| -> Vec<f64> {
            (0..n_instants)
                .map(|k| ratio(flat[k * n_units + u]))
                .collect()
        };
        let measures = |u: usize| -> Measures {
            let q_system_failed = column(&w_plus, u);
            let q_system_intact = column(&w_minus, u);
            let unavailability = column(&w_down, u);
            let birnbaum: Vec<f64> = q_system_failed
                .iter()
                .zip(&q_system_intact)
                .map(|(plus, minus)| plus - minus)
                .collect();
            let fussell_vesely: Vec<f64> = (0..n_instants)
                .map(|k| {
                    if w_cuts[k] > 0.0 {
                        w_fv[k * n_units + u] / w_cuts[k]
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
            Measures {
                unavailability,
                birnbaum,
                fussell_vesely,
                criticality,
                q_system_failed,
                q_system_intact,
            }
        };

        let mut unit = 0;
        let mut basic_events: Vec<EventImportance> = in_cuts
            .iter()
            .map(|&bit| {
                let found = EventImportance {
                    event: self.events[bit].clone(),
                    measures: measures(unit),
                };
                unit += 1;
                found
            })
            .collect();
        let mut component_units: Vec<ComponentImportance> = components
            .iter()
            .map(|(&name, bits)| {
                let events: Vec<String> = bits
                    .iter()
                    .map(|&b| &self.events[b])
                    .map(|e| format!("{}.{}", e.automaton, e.attr))
                    .collect();
                let found = ComponentImportance {
                    component: name.to_owned(),
                    events,
                    measures: measures(unit),
                };
                unit += 1;
                found
            })
            .collect();
        let mut groups: Vec<GroupImportance> = self
            .groups
            .iter()
            .map(|group| {
                let found = GroupImportance {
                    group: group.name.clone(),
                    events: group.events.clone(),
                    measures: measures(unit),
                };
                unit += 1;
                found
            })
            .collect();
        // Most important first: the ranking the question "where do I
        // invest" is actually asking for.
        basic_events.sort_by(|a, b| {
            b.measures
                .last_fussell_vesely()
                .total_cmp(&a.measures.last_fussell_vesely())
                .then_with(|| a.event.cmp(&b.event))
        });
        component_units.sort_by(|a, b| {
            b.measures
                .last_fussell_vesely()
                .total_cmp(&a.measures.last_fussell_vesely())
                .then_with(|| a.component.cmp(&b.component))
        });
        groups.sort_by(|a, b| {
            b.measures
                .last_fussell_vesely()
                .total_cmp(&a.measures.last_fussell_vesely())
                .then_with(|| a.group.cmp(&b.group))
        });

        ImportanceAnalysis {
            target: self.target,
            instants: self.instants,
            q_target,
            q_cuts,
            cuts,
            basic_events,
            components: component_units,
            groups,
            nb_runs,
            engine_version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }
}

/// Which units and cuts each basic event belongs to, so that replaying a
/// trajectory costs, per state change, the units and cuts that contain the
/// changed event, never a pass over all of them.
///
/// An *incidence* is one (unit, cut) pair whose sets meet; the pivotal
/// test `Φ(D ∪ E_u)` only ever reads the incidences of `u`.
#[derive(Debug)]
struct UnitIndex {
    /// Per basic event, the units holding it.
    event_units: Vec<Vec<usize>>,
    /// Per basic event, the cuts holding it.
    event_cuts: Vec<Vec<usize>>,
    /// Per basic event, the incidences whose shared part holds it.
    event_incidences: Vec<Vec<usize>>,
    /// Per cut, the units it meets.
    cut_units: Vec<Vec<usize>>,
    /// Per cut, its size.
    cut_sizes: Vec<u32>,
    /// The incidences of unit `u` are `incidence_start[u]..incidence_start[u + 1]`.
    incidence_start: Vec<usize>,
    /// Per incidence, its cut.
    incidence_cut: Vec<usize>,
    /// Per incidence, how many events its unit and cut share.
    incidence_shared: Vec<u32>,
}

impl UnitIndex {
    /// The index of `cuts` and `units`, both as sorted basic-event bits
    /// below `n_events`.
    fn new(n_events: usize, cuts: &[(Vec<u32>, f64)], units: &[Vec<usize>]) -> Self {
        let mut event_units = vec![Vec::new(); n_events];
        for (u, bits) in units.iter().enumerate() {
            for &bit in bits {
                event_units[bit].push(u);
            }
        }
        let mut event_cuts = vec![Vec::new(); n_events];
        for (c, (bits, _)) in cuts.iter().enumerate() {
            for &bit in bits {
                event_cuts[bit as usize].push(c);
            }
        }
        let cut_units: Vec<Vec<usize>> = cuts
            .iter()
            .map(|(bits, _)| {
                let mut met: Vec<usize> = bits
                    .iter()
                    .flat_map(|&bit| event_units[bit as usize].iter().copied())
                    .collect();
                met.sort_unstable();
                met.dedup();
                met
            })
            .collect();
        let mut unit_cuts = vec![Vec::new(); units.len()];
        for (c, met) in cut_units.iter().enumerate() {
            for &u in met {
                unit_cuts[u].push(c);
            }
        }
        let mut event_incidences = vec![Vec::new(); n_events];
        let mut incidence_start = Vec::with_capacity(units.len() + 1);
        let mut incidence_cut = Vec::new();
        let mut incidence_shared = Vec::new();
        for (bits, met) in units.iter().zip(&unit_cuts) {
            incidence_start.push(incidence_cut.len());
            for &c in met {
                let incidence = incidence_cut.len();
                let mut shared = 0u32;
                for &bit in bits {
                    if cuts[c].0.binary_search(&(bit as u32)).is_ok() {
                        event_incidences[bit].push(incidence);
                        shared += 1;
                    }
                }
                incidence_cut.push(c);
                incidence_shared.push(shared);
            }
        }
        incidence_start.push(incidence_cut.len());
        UnitIndex {
            event_units,
            event_cuts,
            event_incidences,
            cut_units,
            cut_sizes: cuts
                .iter()
                .map(|(bits, _)| u32::try_from(bits.len()).unwrap_or(u32::MAX))
                .collect(),
            incidence_start,
            incidence_cut,
            incidence_shared,
        }
    }

    /// How many units the index covers.
    fn n_units(&self) -> usize {
        self.incidence_start.len() - 1
    }

    /// Unit `u` meets at least one cut.
    fn meets_a_cut(&self, u: usize) -> bool {
        self.incidence_start[u + 1] > self.incidence_start[u]
    }
}

/// The state of one replica as the measures read it, kept up to date one
/// basic event at a time ([`Replay::toggle`]).
#[derive(Debug)]
struct Replay<'a> {
    index: &'a UnitIndex,
    /// Per cut, its events not active: the cut is realized at zero.
    missing: Vec<u32>,
    /// How many cuts are realized.
    realized: usize,
    /// Per unit, the realized cuts it meets.
    touching: Vec<u32>,
    /// Per unit, its active events.
    down: Vec<u32>,
    /// Per incidence, the shared events not active.
    open: Vec<u32>,
}

impl<'a> Replay<'a> {
    fn new(index: &'a UnitIndex) -> Self {
        let mut replay = Replay {
            index,
            missing: Vec::new(),
            realized: 0,
            touching: vec![0; index.n_units()],
            down: vec![0; index.n_units()],
            open: Vec::new(),
        };
        replay.reset();
        replay
    }

    /// Back to no active basic event.
    fn reset(&mut self) {
        self.missing.clone_from(&self.index.cut_sizes);
        // An empty cut is realized whatever is active.
        self.realized = self.missing.iter().filter(|&&m| m == 0).count();
        self.touching.iter_mut().for_each(|t| *t = 0);
        self.down.iter_mut().for_each(|d| *d = 0);
        self.open.clone_from(&self.index.incidence_shared);
    }

    /// Basic event `bit` becomes active (`on`) or inactive. The caller
    /// toggles an event only into the other value.
    fn toggle(&mut self, bit: usize, on: bool) {
        let index = self.index;
        for &u in &index.event_units[bit] {
            if on {
                self.down[u] += 1;
            } else {
                self.down[u] -= 1;
            }
        }
        for &incidence in &index.event_incidences[bit] {
            if on {
                self.open[incidence] -= 1;
            } else {
                self.open[incidence] += 1;
            }
        }
        for &c in &index.event_cuts[bit] {
            if on {
                self.missing[c] -= 1;
            }
            if self.missing[c] == 0 {
                // Realized from now on (`on`), or no longer.
                if on {
                    self.realized += 1;
                } else {
                    self.realized -= 1;
                }
                for &u in &index.cut_units[c] {
                    if on {
                        self.touching[u] += 1;
                    } else {
                        self.touching[u] -= 1;
                    }
                }
            }
            if !on {
                self.missing[c] += 1;
            }
        }
    }

    /// `Φ(D ∪ E_u)` on a system that is up: some cut of `u` misses only
    /// events of `u`, i.e. all its missing events are shared with `u`.
    fn pivotal_with(&self, u: usize) -> bool {
        let index = self.index;
        (index.incidence_start[u]..index.incidence_start[u + 1])
            .any(|i| self.missing[index.incidence_cut[i]] == self.open[i])
    }
}

/// Compute the importance measures of one feared event over a raw
/// Monte-Carlo sequence corpus, for the basic events `events` and the
/// declared `groups`: [`ImportanceReducer`] over `raw` in corpus order.
///
/// `raw` must come from a **free-running** sequence campaign (recording
/// on, target early-stop off): the trajectories have to keep evolving
/// past the feared event for a measure at an instant beyond it to mean
/// anything. `instants` is sorted internally; the returned series follow
/// the sorted order.
///
/// A campaign in which the feared event never occurred returns zeros
/// throughout, with `q_target` and `q_cuts` at zero saying why.
///
/// # Errors
/// What [`ImportanceReducer::new`] refuses.
pub fn importance(
    raw: &[Sequence],
    target: &BasicEvent,
    events: Vec<BasicEvent>,
    groups: Vec<ImportanceGroup>,
    instants: &[f64],
) -> Result<ImportanceAnalysis, ImportanceError> {
    let mut reducer = ImportanceReducer::new(target.clone(), events, groups, instants)?;
    for sequence in raw {
        reducer.push(sequence);
    }
    Ok(reducer.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use raichu_core::SeqEvent;

    /// Build a raw trajectory: `events` are `(obj, automaton, attr, time)`.
    fn traj(events: &[(&str, &str, &str, f64)]) -> Sequence {
        Sequence {
            events: events
                .iter()
                .map(|(obj, automaton, attr, time)| SeqEvent {
                    obj: (*obj).into(),
                    automaton: (*automaton).into(),
                    attr: (*attr).into(),
                    time: *time,
                    cycle_group: Some((*automaton).into()),
                    kind: None,
                })
                .collect(),
            end_cause: None,
            end_time: 10.0,
            weight: 1.0,
        }
    }

    /// A single-mode failure of `obj`, as the muscadet plugin records it.
    fn fail(obj: &str, time: f64) -> (&str, &'static str, &'static str, f64) {
        (obj, "fm", "occ", time)
    }

    fn repair(obj: &str, time: f64) -> (&str, &'static str, &'static str, f64) {
        (obj, "fm", "rep", time)
    }

    fn feared(time: f64) -> (&'static str, &'static str, &'static str, f64) {
        ("T", "ev", "occ", time)
    }

    fn target() -> BasicEvent {
        BasicEvent::new("T", "ev", "occ")
    }

    fn mode(obj: &str) -> BasicEvent {
        BasicEvent::new(obj, "fm", "occ")
    }

    fn modes(objs: &[&str]) -> Vec<BasicEvent> {
        objs.iter().map(|o| mode(o)).collect()
    }

    fn analyse(raw: &[Sequence], events: Vec<BasicEvent>, instants: &[f64]) -> ImportanceAnalysis {
        match importance(raw, &target(), events, Vec::new(), instants) {
            Ok(found) => found,
            Err(error) => unreachable!("the analysis is refused: {error}"),
        }
    }

    fn component(analysis: &ImportanceAnalysis, name: &str) -> Measures {
        match analysis.components.iter().find(|c| c.component == name) {
            Some(found) => found.measures.clone(),
            None => unreachable!("component `{name}` is absent from the analysis"),
        }
    }

    fn event(analysis: &ImportanceAnalysis, wanted: &BasicEvent) -> Measures {
        match analysis.basic_events.iter().find(|e| e.event == *wanted) {
            Some(found) => found.measures.clone(),
            None => unreachable!("`{}` is absent from the analysis", wanted.name()),
        }
    }

    fn group(analysis: &ImportanceAnalysis, name: &str) -> Measures {
        match analysis.groups.iter().find(|g| g.group == name) {
            Some(found) => found.measures.clone(),
            None => unreachable!("group `{name}` is absent from the analysis"),
        }
    }

    /// `Φ = A ∨ (B ∧ C)`: one series block and one parallel pair, the
    /// smallest block diagram with two cuts of different sizes.
    fn series_parallel() -> Vec<Sequence> {
        vec![
            traj(&[fail("A", 0.5), feared(0.5)]),
            traj(&[fail("B", 0.2), fail("C", 0.4), feared(0.4)]),
            traj(&[fail("B", 0.2)]),
            traj(&[]),
        ]
    }

    #[test]
    fn cuts_drop_the_order_and_absorb_their_supersets() {
        // `[B,C]` and `[C,B]` are one cut set; `{A,B}` contains `{A}` and
        // is absorbed by it.
        let raw = vec![
            traj(&[fail("A", 0.5), feared(0.5)]),
            traj(&[fail("A", 0.5), feared(0.5)]),
            traj(&[fail("B", 0.1), fail("A", 0.5), feared(0.5)]),
            traj(&[fail("B", 0.2), fail("C", 0.4), feared(0.4)]),
            traj(&[fail("C", 0.2), fail("B", 0.4), feared(0.4)]),
        ];
        let analysis = analyse(&raw, modes(&["A", "B", "C"]), &[1.0]);
        assert_eq!(analysis.cuts.len(), 2);
        assert_eq!(analysis.cuts[0].events, vec![mode("A")]);
        assert_eq!(analysis.cuts[0].weight, 3.0);
        assert_eq!(analysis.cuts[1].events, modes(&["B", "C"]));
        assert_eq!(analysis.cuts[1].weight, 2.0);
    }

    #[test]
    fn the_pivotal_measures_match_the_hand_computation() {
        // Four replicas, `Φ = A ∨ (B ∧ C)`, read at t = 1 once every
        // failure has happened. The empirical joint is NOT a product of
        // its marginals, which is exactly why the pivotal definition is
        // the one computed: `E[Φ(D ∪ E_i) − Φ(D \ E_i)]` needs no
        // independence between components.
        let analysis = analyse(&series_parallel(), modes(&["A", "B", "C"]), &[1.0]);
        assert_eq!(analysis.q_target, vec![0.5]);
        assert_eq!(analysis.q_cuts, vec![0.5]);

        let a = component(&analysis, "A");
        assert_eq!(a.unavailability, vec![0.25]);
        // 1 − P(B ∧ C) = 1 − 0.25.
        assert_eq!(a.birnbaum, vec![0.75]);
        assert_eq!(a.fussell_vesely, vec![0.5]);
        assert_eq!(a.criticality, vec![0.375]);
        assert_eq!(a.q_system_failed, vec![1.0]);
        assert_eq!(a.q_system_intact, vec![0.25]);

        let b = component(&analysis, "B");
        assert_eq!(b.unavailability, vec![0.5]);
        // P(¬A ∧ C).
        assert_eq!(b.birnbaum, vec![0.25]);
        assert_eq!(b.fussell_vesely, vec![0.5]);

        let c = component(&analysis, "C");
        assert_eq!(c.unavailability, vec![0.25]);
        // P(¬A ∧ B): C is the more critical of the pair here, because B
        // is down more often than C is.
        assert_eq!(c.birnbaum, vec![0.5]);
        assert_eq!(c.fussell_vesely, vec![0.5]);

        // One mode per component: the mode and its component are one unit.
        assert_eq!(event(&analysis, &mode("C")), c);
    }

    #[test]
    fn two_modes_of_one_component_stay_two_basic_events() {
        // `Φ = A1 ∨ A2 ∨ (B ∧ C)`, where the two modes of A are two
        // automata of A that both reach a state named `occ`, as two
        // external modes grafted on one pump do.
        let a1 = BasicEvent::new("A", "fm1", "occ");
        let a2 = BasicEvent::new("A", "fm2", "occ");
        let raw = vec![
            traj(&[("A", "fm1", "occ", 0.5), feared(0.5)]),
            traj(&[("A", "fm2", "occ", 0.5), feared(0.5)]),
            traj(&[fail("B", 0.2), fail("C", 0.4), feared(0.4)]),
            traj(&[fail("B", 0.2)]),
            traj(&[]),
        ];
        let events = vec![a1.clone(), a2.clone(), mode("B"), mode("C")];
        let analysis = analyse(&raw, events, &[1.0]);
        let found: Vec<Vec<BasicEvent>> = analysis.cuts.iter().map(|c| c.events.clone()).collect();
        assert!(found.contains(&vec![a1.clone()]));
        assert!(found.contains(&vec![a2.clone()]));
        assert_eq!(found.len(), 3);

        // D = {A1}, {A2}, {B,C}, {B}, {} over five replicas. A1 is
        // critical where neither A2 nor the pair is down: replicas 1, 4
        // and 5.
        assert_eq!(event(&analysis, &a1).birnbaum, vec![0.6]);
        assert_eq!(event(&analysis, &a2).birnbaum, vec![0.6]);
        // The component is critical wherever the pair holds: 4 of 5. Not
        // the sum of its modes, which is why it is computed, not added.
        let a = component(&analysis, "A");
        assert_eq!(a.birnbaum, vec![0.8]);
        assert_eq!(a.unavailability, vec![0.4]);
        assert_eq!(a.fussell_vesely, vec![2.0 / 3.0]);
        let named = analysis
            .components
            .iter()
            .find(|c| c.component == "A")
            .map(|c| c.events.clone());
        assert_eq!(
            named,
            Some(vec!["fm1.occ".to_owned(), "fm2.occ".to_owned()])
        );
    }

    #[test]
    fn a_group_is_measured_as_one_unit() {
        // The pair as a declared group: critical where A holds, so in the
        // three replicas that are not down through A.
        let a1 = BasicEvent::new("A", "fm1", "occ");
        let raw = vec![
            traj(&[("A", "fm1", "occ", 0.5), feared(0.5)]),
            traj(&[fail("B", 0.2), fail("C", 0.4), feared(0.4)]),
            traj(&[fail("B", 0.2)]),
            traj(&[]),
        ];
        let groups = vec![
            ImportanceGroup {
                name: "pair".into(),
                events: modes(&["B", "C"]),
            },
            // A common cause is attributed to each of its targets: the
            // same basic event may sit in two groups.
            ImportanceGroup {
                name: "with_b".into(),
                events: vec![mode("B")],
            },
        ];
        let events = vec![a1, mode("B"), mode("C")];
        let analysis = match importance(&raw, &target(), events, groups, &[1.0]) {
            Ok(found) => found,
            Err(error) => unreachable!("{error}"),
        };
        let pair = group(&analysis, "pair");
        assert_eq!(pair.birnbaum, vec![0.75]);
        assert_eq!(pair.unavailability, vec![0.5]);
        assert_eq!(pair.fussell_vesely, vec![0.5]);
        assert_eq!(pair.q_system_failed, vec![1.0]);
        assert_eq!(pair.q_system_intact, vec![0.25]);
        // A one-event group is that event.
        assert_eq!(group(&analysis, "with_b"), event(&analysis, &mode("B")));
    }

    #[test]
    fn only_the_declared_basic_events_enter_the_cuts() {
        // An observer `E` records its entry before the feared event, as an
        // intermediate ObjEvent does, and a lost draw of `D` parks it in
        // `not_occ`: neither is a failure, and neither may sit in a cut.
        let raw = vec![
            traj(&[
                ("E", "ev", "occ", 0.1),
                ("D", "fm", "not_occ", 0.2),
                fail("A", 0.5),
                feared(0.5),
            ]),
            traj(&[("E", "ev", "occ", 0.1), feared(0.5)]),
        ];
        let analysis = analyse(&raw, modes(&["A", "D"]), &[1.0]);
        let found: Vec<Vec<BasicEvent>> = analysis.cuts.iter().map(|c| c.events.clone()).collect();
        // The second replica reaches the event with no failure: the empty
        // cut, which the completeness gap reports.
        assert_eq!(found, vec![vec![]]);
        assert!(analysis.components.is_empty());
    }

    #[test]
    fn a_lost_draw_before_a_won_one_does_not_cancel_the_failure() {
        // Two solicitations: the first draw is lost (parked, then
        // re-armed), the second is won. Whatever the draws recorded, the
        // state at the feared event is `occ`.
        let raw = vec![traj(&[
            ("D", "fm", "not_occ", 0.1),
            ("D", "fm", "rep", 0.2),
            ("D", "fm", "occ", 0.5),
            feared(0.5),
        ])];
        let analysis = analyse(&raw, modes(&["D"]), &[1.0]);
        assert_eq!(analysis.cuts.len(), 1);
        assert_eq!(analysis.cuts[0].events, modes(&["D"]));
        assert_eq!(analysis.q_cuts, analysis.q_target);
    }

    #[test]
    fn a_repaired_failure_is_not_active_after_its_repair() {
        // A fails at 1 and is repaired at 2, so the same trajectory reads
        // down at t = 1.5 and up at t = 3: the measures are a function of
        // the instant, not of the trajectory as a whole.
        let raw = vec![
            traj(&[fail("A", 1.0), feared(1.0), repair("A", 2.0)]),
            traj(&[]),
        ];
        let analysis = analyse(&raw, modes(&["A"]), &[1.5, 3.0]);
        assert_eq!(analysis.cuts.len(), 1);
        assert_eq!(analysis.cuts[0].events, modes(&["A"]));
        let a = component(&analysis, "A");
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
        // B fails and is repaired before A brings the system down: B is
        // up at the feared event, so the cut is `{A}` alone and B carries
        // no importance at all.
        let raw = vec![traj(&[
            fail("B", 0.1),
            repair("B", 0.2),
            fail("A", 0.5),
            feared(0.5),
        ])];
        let analysis = analyse(&raw, modes(&["A", "B"]), &[1.0]);
        assert_eq!(analysis.cuts.len(), 1);
        assert_eq!(analysis.cuts[0].events, modes(&["A"]));
        assert_eq!(analysis.components.len(), 1);
        assert_eq!(analysis.components[0].component, "A");
    }

    #[test]
    fn components_are_ranked_by_the_share_of_risk_they_carry() {
        // Three replicas down through A, one through the pair: A carries
        // three quarters of the risk and heads the ranking.
        let raw = vec![
            traj(&[fail("A", 0.5), feared(0.5)]),
            traj(&[fail("A", 0.5), feared(0.5)]),
            traj(&[fail("A", 0.5), feared(0.5)]),
            traj(&[fail("B", 0.2), fail("C", 0.4), feared(0.4)]),
        ];
        let analysis = analyse(&raw, modes(&["A", "B", "C"]), &[1.0]);
        assert_eq!(analysis.components[0].component, "A");
        assert_eq!(analysis.components[0].measures.fussell_vesely, vec![0.75]);
        assert_eq!(analysis.components[1].measures.fussell_vesely, vec![0.25]);
        assert_eq!(analysis.basic_events[0].event, mode("A"));
    }

    #[test]
    fn a_campaign_that_never_reached_the_event_reports_zeros() {
        let raw = vec![traj(&[fail("A", 0.5)]), traj(&[])];
        let analysis = analyse(&raw, modes(&["A"]), &[1.0]);
        assert!(analysis.cuts.is_empty());
        assert!(analysis.components.is_empty());
        assert!(analysis.basic_events.is_empty());
        assert_eq!(analysis.q_target, vec![0.0]);
        assert_eq!(analysis.q_cuts, vec![0.0]);
        assert_eq!(analysis.nb_runs, 2.0);
    }

    #[test]
    fn the_reduction_is_a_pure_function_of_the_corpus() {
        let raw = series_parallel();
        let first = analyse(&raw, modes(&["A", "B", "C"]), &[0.3, 1.0]);
        let second = analyse(&raw, modes(&["C", "B", "A"]), &[1.0, 0.3]);
        // Instants and basic events are sorted, so the two calls describe
        // the same analysis and must produce identical bytes.
        assert_eq!(first, second);
    }

    #[test]
    fn a_malformed_group_is_refused_by_name() {
        let refused = |groups: Vec<ImportanceGroup>| {
            importance(&[], &target(), modes(&["A"]), groups, &[1.0]).err()
        };
        let named = |name: &str, events: Vec<BasicEvent>| ImportanceGroup {
            name: name.into(),
            events,
        };
        assert_eq!(
            refused(vec![named("g", vec![mode("Z")])]),
            Some(ImportanceError::UnknownGroupEvent {
                group: "g".into(),
                event: "Z.fm.occ".into()
            })
        );
        assert_eq!(
            refused(vec![named("g", modes(&["A"])), named("g", modes(&["A"]))]),
            Some(ImportanceError::DuplicateGroup { group: "g".into() })
        );
        assert_eq!(
            refused(vec![named("g", Vec::new())]),
            Some(ImportanceError::EmptyGroup { group: "g".into() })
        );
    }

    /// The measures straight from their definitions, one replica and one
    /// instant at a time, over the cut sets the analysis reports: the
    /// reference the incremental reduction is checked against.
    fn by_definition(
        raw: &[Sequence],
        analysis: &ImportanceAnalysis,
        unit: &[BasicEvent],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = analysis.instants.len();
        let (mut plus, mut minus, mut down, mut fv) =
            (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut w_cuts = vec![0.0; n];
        let phi = |set: &BTreeSet<BasicEvent>| {
            analysis
                .cuts
                .iter()
                .any(|cut| cut.events.iter().all(|e| set.contains(e)))
        };
        for sequence in raw {
            for (k, &instant) in analysis.instants.iter().enumerate() {
                // The last recorded state of each automaton at `instant`.
                let mut state: HashMap<(&str, &str), &str> = HashMap::new();
                for e in sequence.events.iter().filter(|e| e.time <= instant) {
                    state.insert((e.obj.as_str(), e.automaton.as_str()), e.attr.as_str());
                }
                let active: BTreeSet<BasicEvent> = state
                    .iter()
                    .map(|((obj, automaton), attr)| BasicEvent::new(*obj, *automaton, *attr))
                    .filter(|e| {
                        analysis.basic_events.iter().any(|b| b.event == *e) || unit.contains(e)
                    })
                    .collect();
                let with: BTreeSet<BasicEvent> = active.iter().chain(unit).cloned().collect();
                let without: BTreeSet<BasicEvent> = active
                    .iter()
                    .filter(|e| !unit.contains(e))
                    .cloned()
                    .collect();
                let w = sequence.weight;
                if phi(&with) {
                    plus[k] += w;
                }
                if phi(&without) {
                    minus[k] += w;
                }
                if unit.iter().any(|e| active.contains(e)) {
                    down[k] += w;
                }
                let realized = analysis
                    .cuts
                    .iter()
                    .filter(|cut| cut.events.iter().all(|e| active.contains(e)));
                if realized.clone().next().is_some() {
                    w_cuts[k] += w;
                }
                if realized
                    .into_iter()
                    .any(|cut| cut.events.iter().any(|e| unit.contains(e)))
                {
                    fv[k] += w;
                }
            }
        }
        let total: f64 = raw.iter().map(|s| s.weight).sum();
        let share = |v: Vec<f64>| v.into_iter().map(|x| x / total).collect::<Vec<_>>();
        let fv = fv
            .iter()
            .zip(&w_cuts)
            .map(|(f, c)| if *c > 0.0 { f / c } else { 0.0 })
            .collect();
        (share(plus), share(minus), share(down), fv)
    }

    #[test]
    fn every_unit_matches_its_definition_on_a_random_corpus() {
        // Six components with two modes each, `Φ = A1 ∨ (B1 ∧ C1) ∨ (D1 ∧
        // D2)`, failures and repairs drawn at random, uneven weights. E and
        // F never enter a cut: their groups meet no cut, which is the case
        // one group per component makes the common one.
        let mut seed = 7u64;
        let mut draw = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let objs = ["A", "B", "C", "D", "E", "F"];
        let raw: Vec<Sequence> = (0..300)
            .map(|replica| {
                let mut events: Vec<(String, String, String, f64)> = Vec::new();
                let mut downs: HashMap<(usize, usize), Vec<(f64, f64)>> = HashMap::new();
                for (o, obj) in objs.iter().enumerate() {
                    for m in 1..=2 {
                        let mut t = 0.0;
                        loop {
                            t += 10.0 * draw();
                            if t > 10.0 {
                                break;
                            }
                            let r = t + 2.0 * draw();
                            events.push(((*obj).into(), format!("fm{m}"), "occ".into(), t));
                            downs.entry((o, m)).or_default().push((t, r));
                            if r > 10.0 {
                                break;
                            }
                            events.push(((*obj).into(), format!("fm{m}"), "rep".into(), r));
                            t = r;
                        }
                    }
                }
                let is_down = |o: usize, m: usize, t: f64| {
                    downs
                        .get(&(o, m))
                        .is_some_and(|v| v.iter().any(|(a, b)| *a <= t && t < *b))
                };
                let mut dates: Vec<f64> = events.iter().map(|e| e.3).collect();
                dates.sort_by(f64::total_cmp);
                let mut was = false;
                for t in dates {
                    let now = is_down(0, 1, t)
                        || (is_down(1, 1, t) && is_down(2, 1, t))
                        || (is_down(3, 1, t) && is_down(3, 2, t));
                    if now != was {
                        let attr = if now { "occ" } else { "ok" };
                        events.push(("T".into(), "ev".into(), attr.into(), t));
                        was = now;
                    }
                }
                events.sort_by(|a, b| a.3.total_cmp(&b.3));
                let refs: Vec<(&str, &str, &str, f64)> = events
                    .iter()
                    .map(|(o, a, s, t)| (o.as_str(), a.as_str(), s.as_str(), *t))
                    .collect();
                let mut sequence = traj(&refs);
                sequence.weight = [0.5, 1.0, 1.75][replica % 3];
                sequence
            })
            .collect();
        let named = |o: &str, m: u8| BasicEvent::new(o, format!("fm{m}"), "occ");
        let events: Vec<BasicEvent> = objs
            .iter()
            .flat_map(|o| [named(o, 1), named(o, 2)])
            .collect();
        let declared = |name: &str, events: Vec<BasicEvent>| ImportanceGroup {
            name: name.into(),
            events,
        };
        let groups = vec![
            declared("pair", vec![named("B", 1), named("C", 1)]),
            declared("cross", vec![named("A", 1), named("D", 2), named("E", 1)]),
            declared("free", vec![named("E", 1), named("F", 2)]),
            declared("lone", vec![named("F", 1)]),
            declared("d", vec![named("D", 1), named("D", 2)]),
        ];
        let instants: Vec<f64> = (0..=10).map(f64::from).collect();
        let analysis = match importance(&raw, &target(), events, groups.clone(), &instants) {
            Ok(found) => found,
            Err(error) => unreachable!("{error}"),
        };
        assert!(analysis.q_cuts.iter().any(|q| *q > 0.0));
        let check = |label: &str, measures: &Measures, unit: &[BasicEvent]| {
            let (plus, minus, down, fv) = by_definition(&raw, &analysis, unit);
            assert_eq!(measures.q_system_failed, plus, "{label}: Φ(D ∪ E)");
            assert_eq!(measures.q_system_intact, minus, "{label}: Φ(D \\ E)");
            assert_eq!(measures.unavailability, down, "{label}: unavailability");
            assert_eq!(measures.fussell_vesely, fv, "{label}: Fussell-Vesely");
        };
        for found in &analysis.basic_events {
            check(
                &found.event.name(),
                &found.measures,
                std::slice::from_ref(&found.event),
            );
        }
        for found in &analysis.components {
            let unit: Vec<BasicEvent> = found
                .events
                .iter()
                .map(|e| {
                    let (automaton, attr) = e.split_once('.').unwrap_or((e, ""));
                    BasicEvent::new(&found.component, automaton, attr)
                })
                .collect();
            check(&found.component, &found.measures, &unit);
        }
        assert_eq!(analysis.groups.len(), groups.len());
        for found in &analysis.groups {
            check(&found.group, &found.measures, &found.events);
        }
    }

    #[test]
    fn an_empty_cut_holds_the_system_down_for_every_group() {
        // The feared event reached with no basic event active realizes the
        // empty cut, which absorbs every other cut: the system is down
        // throughout, no unit meets a cut, and only the groups remain.
        let raw = vec![
            traj(&[feared(0.1)]),
            traj(&[
                fail("A", 0.2),
                fail("B", 0.4),
                feared(0.4),
                repair("A", 0.6),
            ]),
            traj(&[fail("B", 0.3)]),
        ];
        let groups = vec![
            ImportanceGroup {
                name: "pair".into(),
                events: modes(&["A", "B"]),
            },
            ImportanceGroup {
                name: "b".into(),
                events: vec![mode("B")],
            },
        ];
        let instants = [0.0, 0.25, 0.5, 1.0];
        let analysis = match importance(&raw, &target(), modes(&["A", "B"]), groups, &instants) {
            Ok(found) => found,
            Err(error) => unreachable!("{error}"),
        };
        assert_eq!(analysis.cuts.len(), 1);
        assert!(analysis.cuts[0].events.is_empty());
        assert!(analysis.basic_events.is_empty());
        assert_eq!(analysis.q_cuts, vec![1.0; 4]);
        for found in &analysis.groups {
            let (plus, minus, down, fv) = by_definition(&raw, &analysis, &found.events);
            assert_eq!(found.measures.q_system_failed, plus, "{}", found.group);
            assert_eq!(found.measures.q_system_intact, minus, "{}", found.group);
            assert_eq!(found.measures.unavailability, down, "{}", found.group);
            assert_eq!(found.measures.fussell_vesely, fv, "{}", found.group);
        }
        assert_eq!(analysis.groups.len(), 2);
    }
}
