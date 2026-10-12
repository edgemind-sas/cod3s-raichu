//! Sequence-analysis pipeline (F3a-A2): the native RAMS post-processing of
//! a Monte-Carlo corpus of recorded trajectory [`Sequence`]s, porting the
//! cod3s `SequenceAnalyser` canonical pipeline, chained by [`analyse`]
//! (`filter_cycles → group_sequences → minimal_sequences`).
//!
//! - **filter cycles** drops transient failure→repair cycles that net out
//!   before the feared event (a failure mode's monitored occ/rep events
//!   strictly alternate, so an even per-group count cancels entirely and an
//!   odd one leaves only the last, persistent failure);
//!   a trajectory that cancellation would leave without a cause is kept
//!   whole and reported ([`ReductionReport`]): the feared event cannot
//!   happen without one, so the hypothesis failed for it;
//! - **group** buckets by end cause (target) and merges sequences with an
//!   identical ordered `(obj, attr)` signature, summing weights;
//! - **minimal** greedily keeps the shortest sequences and absorbs every
//!   longer super-sequence that includes one (order-dependent, matching
//!   cod3s' `compute_minimal_sequences`).
//!
//! The cleaned level is built one trajectory at a time by a
//! [`SequenceReducer`]: each trajectory's cycles are filtered as it
//! arrives and it is merged at once, so a campaign holds one entry per
//! distinct *cleaned* path, never its raw trajectories. Filtering before
//! grouping is what bounds that memory: on a repairable system nearly
//! every raw trajectory is distinct (its transient cycles differ), while
//! the paths left once they are cancelled are few.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use raichu_core::{CycleGroupDecision, SeqEvent, Sequence};
use raichu_model::TransitionKind;
use serde::Serialize;

/// Ordered `(obj, attr)` signature of a sequence: the identity used for
/// grouping and subsequence inclusion (times and cycle groups excluded).
fn signature(seq: &Sequence) -> Vec<(&str, &str)> {
    seq.events
        .iter()
        .map(|e| (e.obj.as_str(), e.attr.as_str()))
        .collect()
}

/// Hash of what [`group_sequences`] merges on: the end cause and the
/// ordered `(obj, attr)` signature. A bucket key only, never an identity:
/// what decides a merge is the exact comparison the caller then runs.
fn signature_hash(seq: &Sequence) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seq.end_cause.hash(&mut hasher);
    for event in &seq.events {
        event.obj.hash(&mut hasher);
        event.attr.hash(&mut hasher);
    }
    hasher.finish()
}

/// Group raw trajectory sequences: bucket by end cause, merge sequences with
/// an identical ordered `(obj, attr)` signature (weights summed, event and
/// end times averaged), and sort each bucket by descending weight (ties keep
/// first-seen order). Deterministic.
#[must_use]
pub fn group_sequences(raw: Vec<Sequence>) -> Vec<Sequence> {
    let mut grouping = Grouping::default();
    for seq in raw {
        grouping.merge(seq);
    }
    grouping.into_sorted()
}

/// The merge table [`group_sequences`] and [`SequenceReducer`] share: one
/// slot per distinct `(end_cause, signature)`, in first-seen order.
#[derive(Debug, Default)]
struct Grouping {
    // (end_cause, signature) → index into `merged`, preserving first-seen
    // order. The accumulated weight *is* `merged[i].weight`, so no
    // parallel count vector is kept: one number, one owner, nothing to
    // keep in sync.
    //
    // The lookup goes through a hash of the signature rather than a scan
    // of `merged`, because a scan is quadratic in the number of DISTINCT
    // sequences, which on a repairable system is very nearly the number of
    // replicas: 20 000 of them took 1.3 s of pure comparison, and doubling
    // the campaign quadrupled it. A hash bucket holds the candidate slots,
    // and the exact comparison below still decides, so a collision costs a
    // comparison and never a wrong merge. Order is untouched: slots are
    // created in first-seen order and merged into in corpus order, which
    // is what keeps the float reduction byte-deterministic.
    merged: Vec<Sequence>,
    buckets: HashMap<u64, Vec<usize>>,
}

impl Grouping {
    /// Merge one sequence into its slot, or open a slot for it.
    fn merge(&mut self, seq: Sequence) {
        let merged = &mut self.merged;
        // Signatures are compared in place: materialising them would
        // allocate on both sides of every candidate.
        let slots = self.buckets.entry(signature_hash(&seq)).or_default();
        let pos = slots.iter().copied().find(|&i| {
            let m: &Sequence = &merged[i];
            m.end_cause == seq.end_cause
                && m.events.len() == seq.events.len()
                && m.events
                    .iter()
                    .zip(&seq.events)
                    .all(|(a, b)| a.obj == b.obj && a.attr == b.attr)
        });
        match pos {
            Some(i) => {
                let acc_weight = merged[i].weight;
                let n = acc_weight + seq.weight;
                // Weight-averaged event and end times.
                for (acc, ev) in merged[i].events.iter_mut().zip(&seq.events) {
                    acc.time = (acc.time * acc_weight + ev.time * seq.weight) / n;
                }
                merged[i].end_time =
                    (merged[i].end_time * acc_weight + seq.end_time * seq.weight) / n;
                merged[i].weight = n;
            }
            None => {
                slots.push(merged.len());
                merged.push(seq);
            }
        }
    }

    /// The slots, by descending weight (ties keep first-seen order).
    fn into_sorted(self) -> Vec<Sequence> {
        let mut merged = self.merged;
        merged.sort_by(|a, b| b.weight.total_cmp(&a.weight));
        merged
    }
}

/// The CLEANED level of a corpus, built one trajectory at a time: each
/// pushed trajectory has its transient cycles filtered, then merges with
/// the distinct paths already held. What it holds grows with the number of
/// distinct cleaned paths, not with the number of trajectories pushed, so
/// a campaign of any size reduces in bounded memory, with one exception: a
/// contradicted trajectory (see [`ReductionReport`]) is held with its
/// cycles, so on a model where many trajectories are contradicted with
/// different cycle histories, memory grows with those histories.
///
/// Pushing the trajectories of a corpus in replica order and taking
/// [`SequenceReducer::cleaned`] is [`clean`]; the result is deterministic
/// for a given push order.
#[derive(Debug, Default)]
pub struct SequenceReducer {
    grouping: Grouping,
    report: ReductionReport,
    pushed: u64,
    /// The cycle groups not to cancel, per end cause, computed from
    /// [`ReductionReport::cycle_groups`] the first time a cause is met.
    /// `None` for a cause whose decisions keep nothing, so the hot path
    /// pays no lookup there.
    kept: HashMap<Option<String>, Option<KeptGroups>>,
}

/// How many contradicted trajectories a [`ReductionReport`] identifies by
/// index. The count and weight stay exact past it; the list stops growing,
/// so the report holds bounded memory on a campaign of any size.
pub const CONTRADICTED_INDICES_CAP: usize = 100;

/// What the reduction did beyond its two steps, for the analyst to audit.
///
/// A **contradicted** trajectory reached a feared event, yet cancelling its
/// failure/repair cycles would leave it without a recorded cause: nothing
/// at all, or only repairs and observations while a declared failure was
/// cancelled. An event cannot happen without a cause, so such a trajectory
/// has one the cancellation would hide: either a repaired failure kept an
/// effect (the hypothesis that a repaired failure no longer contributes
/// failed, typically because a continuous state integrated its effect), or
/// the feared event has a cause that no monitored transition records. It
/// is reduced with no cycle cancelled.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ReductionReport {
    /// Number of contradicted trajectories.
    pub contradicted: u64,
    /// Their total weight: a trajectory count for a Monte-Carlo campaign,
    /// a probability mass for a sequence-tree exploration.
    pub contradicted_weight: f64,
    /// The first [`CONTRADICTED_INDICES_CAP`] of them, in push order: the
    /// replica index for a campaign or a raw corpus, the sequence index for
    /// an exploration.
    pub contradicted_indices: Vec<u64>,
    /// The per-group decisions the reduction applied (see
    /// [`raichu_core::cycle_groups`]): a group not cancellable for a
    /// trajectory's end cause keeps every event. `None` when the reduction
    /// did not know them, as for a corpus or an exploration written before
    /// they were recorded: every group was then cancelled, the way earlier
    /// engines did.
    pub cycle_groups: Option<Vec<CycleGroupDecision>>,
    /// Whether the decisions were unknown while some trajectory carried a
    /// cycle group: the reduction then cancelled groups it could not vouch
    /// for. Only ever true when [`ReductionReport::cycle_groups`] is `None`.
    pub cycle_groups_unknown: bool,
}

impl ReductionReport {
    fn contradicted(&mut self, index: u64, weight: f64) {
        self.contradicted += 1;
        self.contradicted_weight += weight;
        if self.contradicted_indices.len() < CONTRADICTED_INDICES_CAP {
            self.contradicted_indices.push(index);
        }
    }
}

impl SequenceReducer {
    /// An empty reduction that knows no per-group decision: every cycle
    /// group is cancelled, the reduction of engines before 0.86.0.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty reduction applying `decisions` when they are known, and
    /// cancelling every group as [`SequenceReducer::new`] does when they
    /// are not: the reduction of a corpus or an exploration that may
    /// carry them.
    #[must_use]
    pub fn from_decisions(decisions: Option<&[CycleGroupDecision]>) -> Self {
        match decisions {
            Some(decisions) => Self::with_cycle_groups(decisions.to_vec()),
            None => Self::new(),
        }
    }

    /// An empty reduction applying the per-group decisions of the model
    /// the trajectories came from ([`raichu_core::CompiledModel::cycle_groups`]).
    #[must_use]
    pub fn with_cycle_groups(decisions: Vec<CycleGroupDecision>) -> Self {
        SequenceReducer {
            report: ReductionReport {
                cycle_groups: Some(decisions),
                ..ReductionReport::default()
            },
            ..Self::default()
        }
    }

    /// Add one trajectory, identified in the report by its push order
    /// (0-based): the replica index when every replica is pushed in order.
    pub fn push(&mut self, sequence: Sequence) {
        self.push_indexed(self.pushed, sequence);
    }

    /// Add one trajectory, identified in the report by `index`: for a
    /// reduction that skips some replicas (a condition on observations),
    /// the replica index the push order would not give. One reducer is fed
    /// through [`SequenceReducer::push`] or through this, not both: the
    /// push count would not match the indices given here.
    pub fn push_indexed(&mut self, index: u64, sequence: Sequence) {
        self.pushed += 1;
        let weight = sequence.weight;
        if self.report.cycle_groups.is_none()
            && !self.report.cycle_groups_unknown
            && sequence.events.iter().any(|e| e.cycle_group.is_some())
        {
            self.report.cycle_groups_unknown = true;
        }
        let kept = match &self.report.cycle_groups {
            Some(decisions) => {
                // The cause is cloned once per distinct cause, not per push.
                if !self.kept.contains_key(&sequence.end_cause) {
                    let groups = kept_groups(decisions, sequence.end_cause.as_deref());
                    self.kept.insert(
                        sequence.end_cause.clone(),
                        (!groups.is_empty()).then_some(groups),
                    );
                }
                self.kept.get(&sequence.end_cause).and_then(Option::as_ref)
            }
            None => None,
        };
        let (sequence, contradicted) = cancel_cycles(sequence, kept);
        if contradicted {
            self.report.contradicted(index, weight);
        }
        self.grouping.merge(sequence);
    }

    /// What the reduction did so far beyond its two steps.
    #[must_use]
    pub fn report(&self) -> &ReductionReport {
        &self.report
    }

    /// The number of distinct cleaned paths held so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.grouping.merged.len()
    }

    /// Whether nothing was pushed yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grouping.merged.is_empty()
    }

    /// The cleaned level: every distinct path to each end cause, transient
    /// cycles removed, by descending weight (ties keep first-seen order).
    #[must_use]
    pub fn cleaned(self) -> Vec<Sequence> {
        self.grouping.into_sorted()
    }
}

/// Drop transient failure→repair cycles. Within each `cycle_group` the
/// monitored events strictly alternate (occ, rep, occ, …) starting with the
/// failure, so an even per-group count nets out entirely (repaired before
/// the feared event) and an odd one leaves only the last, persistent
/// failure. Events with no cycle group are always kept, order preserved.
///
/// A trajectory the cancellation would leave without a cause is kept
/// whole (see [`ReductionReport`]); [`SequenceReducer`] counts them.
#[must_use]
pub fn filter_cycles(sequences: Vec<Sequence>) -> Vec<Sequence> {
    sequences
        .into_iter()
        .map(|sequence| cancel_cycles(sequence, None).0)
        .collect()
}

/// Whether an event left after cancellation may be a cause: a declared
/// failure, or an event that declares no kind and so cannot be ruled out.
fn may_cause(event: &SeqEvent) -> bool {
    !matches!(
        event.kind,
        Some(TransitionKind::Repair | TransitionKind::Observation)
    )
}

/// [`filter_cycles`] on one sequence, and whether the cancellation was
/// refused because it contradicts the feared event: the trajectory reached
/// a target, cancellation removed something, and what it would leave is
/// empty or holds no possible cause while a declared failure was removed.
/// The test runs on the kept set before anything is drained, so a refused
/// cancellation costs no copy.
///
/// `kept` names the `(component, group)` pairs not to cancel at all (the
/// per-group decisions for this trajectory's end cause); their events all
/// survive, the full history of the group.
fn cancel_cycles(mut seq: Sequence, kept: Option<&KeptGroups>) -> (Sequence, bool) {
    let is_kept = |ev: &SeqEvent, group: &str| {
        kept.and_then(|kept| kept.get(ev.obj.as_str()))
            .is_some_and(|groups| groups.contains(group))
    };
    // For each cancellable (component, group) pair, the index of its last
    // event (kept iff the pair has an odd count); every other grouped
    // event is dropped. The component is part of the key so two DIFFERENT
    // failure modes sharing an automaton name (e.g. every ObjFM's
    // `fm__cc_1`) can never cancel each other.
    let mut last: HashMap<(&str, &str), (usize, usize)> = HashMap::new();
    // Whether each event survives, decided once: the scan below and the
    // drain both read it.
    let mut survives: Vec<bool> = Vec::with_capacity(seq.events.len());
    for (i, ev) in seq.events.iter().enumerate() {
        match ev.cycle_group.as_deref() {
            Some(g) if !is_kept(ev, g) => {
                let entry = last.entry((ev.obj.as_str(), g)).or_insert((0, i));
                entry.0 += 1;
                entry.1 = i;
                survives.push(false);
            }
            _ => survives.push(true),
        }
    }
    for (count, idx) in last.values() {
        if count % 2 == 1 {
            survives[*idx] = true;
        }
    }
    if survives.iter().all(|&s| s) {
        // Nothing cancelled: nothing to test, nothing to rebuild.
        return (seq, false);
    }
    if seq.end_cause.is_some() {
        let mut removed_failure = false;
        let mut left_any = false;
        let mut left_cause = false;
        for (ev, &s) in seq.events.iter().zip(&survives) {
            if s {
                left_any = true;
                left_cause |= may_cause(ev);
            } else {
                removed_failure |= ev.kind == Some(TransitionKind::Failure);
            }
        }
        if !left_any || (removed_failure && !left_cause) {
            return (seq, true);
        }
    }
    let mut flags = survives.into_iter();
    seq.events.retain(|_| flags.next().unwrap_or(true));
    (seq, false)
}

/// The cycle groups not to cancel, by component, then group: looked up by
/// `&str` on the hot path, with no key to build.
type KeptGroups = HashMap<String, HashSet<String>>;

/// The groups not to cancel in a trajectory ending on `end_cause`.
fn kept_groups(decisions: &[CycleGroupDecision], end_cause: Option<&str>) -> KeptGroups {
    let mut kept = KeptGroups::new();
    for decision in decisions.iter().filter(|d| !d.cancellable_for(end_cause)) {
        kept.entry(decision.component.clone())
            .or_default()
            .insert(decision.cycle_group.clone());
    }
    kept
}

/// `short` is an ordered (non-contiguous) subsequence of `long`.
fn is_included(short: &[(&str, &str)], long: &[(&str, &str)]) -> bool {
    let mut it = long.iter();
    short.iter().all(|s| it.any(|l| l == s))
}

/// Minimal sequences (greedy, as cod3s' `compute_minimal_sequences`): per end
/// cause, sort by ascending length, then descending weight, then ascending
/// `(obj, attr)` signature; keep each sequence unless an already-kept one is
/// included in it, in which case absorb it (add its weight to that minimal).
/// Then sort by descending weight, ties by end cause and signature.
///
/// The signature settles every tie the greedy absorption would otherwise
/// leave to the order of `sequences`: a sequence included in two kept ones
/// of the same length and weight goes to the one with the smaller
/// signature. The result therefore depends only on the sequences and their
/// weights, not on the order they come in, so a corpus reduced in another
/// order (read back, filtered, or grouped differently) gives the same
/// minimal level.
#[must_use]
pub fn minimal_sequences(sequences: Vec<Sequence>) -> Vec<Sequence> {
    // Partition by end cause, preserving first-seen order of causes.
    let mut causes: Vec<Option<String>> = Vec::new();
    let mut buckets: Vec<Vec<Sequence>> = Vec::new();
    for seq in sequences {
        match causes.iter().position(|c| *c == seq.end_cause) {
            Some(i) => buckets[i].push(seq),
            None => {
                causes.push(seq.end_cause.clone());
                buckets.push(vec![seq]);
            }
        }
    }
    let mut out = Vec::new();
    for mut bucket in buckets {
        bucket.sort_by(|a, b| {
            a.events
                .len()
                .cmp(&b.events.len())
                .then(b.weight.total_cmp(&a.weight))
                .then_with(|| signature(a).cmp(&signature(b)))
        });
        let mut minimal: Vec<Sequence> = Vec::new();
        for seq in bucket {
            let sig = signature(&seq);
            match minimal
                .iter_mut()
                .find(|m| is_included(&signature(m), &sig))
            {
                Some(m) => m.weight += seq.weight,
                None => minimal.push(seq),
            }
        }
        out.extend(minimal);
    }
    out.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.end_cause.cmp(&b.end_cause))
            .then_with(|| signature(a).cmp(&signature(b)))
    });
    out
}

/// The CLEANED corpus: filter cycles, then group (weights summed, times
/// averaged). Every distinct path to each feared event, transient
/// failure/repair cycles removed, before any absorption into a shorter
/// one: cod3s's post-filter snapshot (`sequences_all.json`).
///
/// This is a [`SequenceReducer`] fed `raw` in order. Grouping before the
/// filter as well would give the same groups and weights wherever a given
/// `(obj, attr)` always carries the same cycle group, which is how the
/// engine records them; the dates are averaged per cleaned path, so they
/// may differ from such a reduction in their last bits, and two paths of
/// equal weight keep the order in which their first trajectory arrived.
/// That order does not reach the minimal level, which
/// [`minimal_sequences`] decides from the paths and weights alone.
#[must_use]
pub fn clean(raw: Vec<Sequence>) -> Vec<Sequence> {
    let mut reducer = SequenceReducer::new();
    for sequence in raw {
        reducer.push(sequence);
    }
    reducer.cleaned()
}

/// The full pipeline on a raw Monte-Carlo corpus: [`clean`], then minimal.
#[must_use]
pub fn analyse(raw: Vec<Sequence>) -> Vec<Sequence> {
    minimal_sequences(clean(raw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use raichu_core::{DecisionOrigin, TargetDecision};
    use raichu_model::TransitionKind;

    /// Build a sequence: `events` are `(obj, attr, group)`, all at time 0.
    fn seq(cause: &str, weight: f64, events: &[(&str, &str, Option<&str>)]) -> Sequence {
        Sequence {
            events: events
                .iter()
                .map(|(o, a, g)| SeqEvent {
                    obj: (*o).into(),
                    automaton: String::new(),
                    attr: (*a).into(),
                    time: 0.0,
                    cycle_group: g.map(Into::into),
                    kind: None,
                })
                .collect(),
            end_cause: Some(cause.into()),
            end_time: 0.0,
            weight,
        }
    }

    fn sig_of(s: &Sequence) -> Vec<(String, String)> {
        s.events
            .iter()
            .map(|e| (e.obj.clone(), e.attr.clone()))
            .collect()
    }

    #[test]
    fn group_merges_identical_signatures_and_sorts_by_weight() {
        let raw = vec![
            seq("F", 1.0, &[("A", "occ", None)]),
            seq("F", 1.0, &[("B", "occ", None), ("A", "occ", None)]),
            seq("F", 1.0, &[("A", "occ", None)]), // same signature as #1
        ];
        let grouped = group_sequences(raw);
        assert_eq!(grouped.len(), 2);
        // Heaviest first: the merged [A] with weight 2.
        assert_eq!(grouped[0].weight, 2.0);
        assert_eq!(sig_of(&grouped[0]), vec![("A".into(), "occ".into())]);
        assert_eq!(grouped[1].weight, 1.0);
    }

    #[test]
    fn filter_drops_netted_cycles_and_keeps_the_persistent_failure() {
        // fm1: occ,rep (even → all dropped). fm2: occ,rep,occ (odd → keep
        // the last occ). An ungrouped feared-event occ is always kept.
        let s = seq(
            "F",
            1.0,
            &[
                ("fm1", "occ", Some("fm1")),
                ("fm2", "occ", Some("fm2")),
                ("fm1", "rep", Some("fm1")),
                ("fm2", "rep", Some("fm2")),
                ("fm2", "occ", Some("fm2")),
                ("ER", "occ", None),
            ],
        );
        let filtered = filter_cycles(vec![s]);
        assert_eq!(
            sig_of(&filtered[0]),
            vec![("fm2".into(), "occ".into()), ("ER".into(), "occ".into())]
        );
    }

    #[test]
    fn filter_never_pairs_events_across_components() {
        // Review finding: two DIFFERENT failure modes share the bare
        // automaton name (every ObjFM's `fm__cc_1`): their persistent
        // failures must NOT cancel each other by group-name parity.
        let s = seq(
            "F",
            1.0,
            &[
                ("fm_A", "occ", Some("fm__cc_1")),
                ("fm_B", "occ", Some("fm__cc_1")),
                ("ER", "occ", None),
            ],
        );
        let filtered = filter_cycles(vec![s]);
        assert_eq!(
            sig_of(&filtered[0]),
            vec![
                ("fm_A".into(), "occ".into()),
                ("fm_B".into(), "occ".into()),
                ("ER".into(), "occ".into())
            ]
        );
    }

    #[test]
    fn minimal_absorbs_super_sequences() {
        // [A] is included in [A,B] → the longer absorbs into [A]; the weight
        // accumulates and only the minimal [A] survives.
        let seqs = vec![
            seq("F", 3.0, &[("A", "occ", None)]),
            seq("F", 2.0, &[("A", "occ", None), ("B", "occ", None)]),
        ];
        let minimal = minimal_sequences(seqs);
        assert_eq!(minimal.len(), 1);
        assert_eq!(sig_of(&minimal[0]), vec![("A".into(), "occ".into())]);
        assert_eq!(minimal[0].weight, 5.0);
    }

    #[test]
    fn minimal_keeps_distinct_incomparable_sequences() {
        // [A] and [B] are incomparable → both kept.
        let seqs = vec![
            seq("F", 1.0, &[("A", "occ", None)]),
            seq("F", 1.0, &[("B", "occ", None)]),
        ];
        assert_eq!(minimal_sequences(seqs).len(), 2);
    }

    #[test]
    fn analyse_end_to_end() {
        // Two raw trajectories reaching F: one where fm1 fails-then-repairs
        // (transient) before the ER occ, one where fm1 stays failed. After
        // cycle filtering both collapse to the single-cause minimal [ER.occ]
        // (transient) vs [fm1.occ, ER.occ] (persistent); the minimal set is
        // {[ER.occ], [fm1.occ, ER.occ]}? No: [ER.occ] ⊆ [fm1.occ, ER.occ],
        // so the latter absorbs into [ER.occ].
        let raw = vec![
            // transient: fm1 occ then rep, then ER occ
            seq(
                "F",
                1.0,
                &[
                    ("fm1", "occ", Some("fm1")),
                    ("fm1", "rep", Some("fm1")),
                    ("ER", "occ", None),
                ],
            ),
            // persistent: fm1 occ (no rep), then ER occ
            seq(
                "F",
                1.0,
                &[("fm1", "occ", Some("fm1")), ("ER", "occ", None)],
            ),
        ];
        let minimal = analyse(raw);
        // Transient → [ER.occ] (weight 1); persistent → [fm1.occ, ER.occ]
        // (weight 1) absorbs into [ER.occ] → single minimal, weight 2.
        assert_eq!(minimal.len(), 1);
        assert_eq!(sig_of(&minimal[0]), vec![("ER".into(), "occ".into())]);
        assert_eq!(minimal[0].weight, 2.0);
    }

    /// Build a sequence with declared kinds: `events` are `(obj, attr,
    /// group, kind)`, at increasing dates.
    fn kseq(
        cause: Option<&str>,
        events: &[(&str, &str, Option<&str>, Option<TransitionKind>)],
    ) -> Sequence {
        Sequence {
            events: events
                .iter()
                .enumerate()
                .map(|(i, (o, a, g, k))| SeqEvent {
                    obj: (*o).into(),
                    automaton: String::new(),
                    attr: (*a).into(),
                    time: i as f64,
                    cycle_group: g.map(Into::into),
                    kind: *k,
                })
                .collect(),
            end_cause: cause.map(Into::into),
            end_time: events.len() as f64,
            weight: 1.0,
        }
    }

    const F: Option<TransitionKind> = Some(TransitionKind::Failure);
    const R: Option<TransitionKind> = Some(TransitionKind::Repair);
    const O: Option<TransitionKind> = Some(TransitionKind::Observation);

    /// The muscadet shape: every failure repaired, the feared event's own
    /// observation surviving the filter as the only event left.
    fn repaired_then_observed() -> Sequence {
        kseq(
            Some("F"),
            &[
                ("fm1", "occ", Some("fm1"), F),
                ("fm1", "rep", Some("fm1"), R),
                ("ER", "occ", Some("ER"), O),
            ],
        )
    }

    fn reduce(raw: Vec<Sequence>) -> (Vec<Sequence>, ReductionReport) {
        let mut reducer = SequenceReducer::new();
        for sequence in raw {
            reducer.push(sequence);
        }
        let report = reducer.report().clone();
        (minimal_sequences(reducer.cleaned()), report)
    }

    #[test]
    fn a_trajectory_left_with_only_observations_keeps_its_failures() {
        // Cancelled, the first trajectory would be [ER.occ], included in
        // every other sequence: it would absorb the persistent one and say
        // that no failure is needed.
        let persistent = kseq(
            Some("F"),
            &[("fm1", "occ", Some("fm1"), F), ("ER", "occ", Some("ER"), O)],
        );
        let (minimal, report) = reduce(vec![repaired_then_observed(), persistent]);
        assert_eq!(report.contradicted, 1);
        assert_eq!(report.contradicted_weight, 1.0);
        assert_eq!(report.contradicted_indices, vec![0]);
        // Kept whole, the contradicted trajectory contains [fm1.occ,
        // ER.occ] and is absorbed into it: no cause-free minimal sequence.
        assert_eq!(minimal.len(), 1);
        assert_eq!(
            sig_of(&minimal[0]),
            vec![("fm1".into(), "occ".into()), ("ER".into(), "occ".into())]
        );
        assert_eq!(minimal[0].weight, 2.0);
    }

    #[test]
    fn a_trajectory_emptied_by_cancellation_is_contradicted_even_without_kinds() {
        let emptied = kseq(
            Some("F"),
            &[("A", "occ", Some("A"), None), ("A", "rep", Some("A"), None)],
        );
        let (minimal, report) = reduce(vec![emptied]);
        assert_eq!(report.contradicted, 1);
        assert_eq!(minimal[0].events.len(), 2);
    }

    #[test]
    fn an_undeclared_event_left_over_counts_as_a_possible_cause() {
        let undeclared = kseq(
            Some("F"),
            &[
                ("A", "occ", Some("A"), None),
                ("A", "rep", Some("A"), None),
                ("ER", "occ", None, None),
            ],
        );
        let (minimal, report) = reduce(vec![undeclared]);
        assert_eq!(report.contradicted, 0);
        assert_eq!(sig_of(&minimal[0]), vec![("ER".into(), "occ".into())]);
    }

    #[test]
    fn only_a_reaching_trajectory_whose_cancellation_removed_something_is_contradicted() {
        let mut horizon = repaired_then_observed();
        horizon.end_cause = None;
        let nothing_recorded = kseq(Some("F"), &[]);
        let one_failure_left = kseq(
            Some("F"),
            &[
                ("fm1", "occ", Some("fm1"), F),
                ("fm1", "rep", Some("fm1"), R),
                ("fm2", "occ", Some("fm2"), F),
                ("ER", "occ", Some("ER"), O),
            ],
        );
        let (_, report) = reduce(vec![horizon, nothing_recorded, one_failure_left]);
        assert_eq!(report.contradicted, 0);
        assert!(report.contradicted_indices.is_empty());
    }

    #[test]
    fn the_report_lists_the_first_contradicted_indices_only() {
        let mut reducer = SequenceReducer::new();
        for _ in 0..150 {
            reducer.push(kseq(Some("F"), &[("X", "occ", None, O)]));
            reducer.push(repaired_then_observed());
        }
        let report = reducer.report();
        assert_eq!(report.contradicted, 150);
        assert_eq!(report.contradicted_weight, 150.0);
        assert_eq!(report.contradicted_indices.len(), CONTRADICTED_INDICES_CAP);
        assert_eq!(report.contradicted_indices[..3], [1, 3, 5]);
    }

    fn decision(
        component: &str,
        group: &str,
        kept_for: &[&str],
        targets: &[&str],
    ) -> CycleGroupDecision {
        CycleGroupDecision {
            component: component.into(),
            cycle_group: group.into(),
            automata: vec![format!("{component}.life")],
            targets: targets
                .iter()
                .map(|t| TargetDecision {
                    target: (*t).into(),
                    cancellable: !kept_for.contains(t),
                    origin: DecisionOrigin::Analysis,
                    witness: Vec::new(),
                    memory: None,
                })
                .collect(),
        }
    }

    /// `A` (kept for `T1` only) and `B` (always cancellable) each fail and
    /// are repaired, then `C` fails for good.
    fn two_cycles_then(cause: Option<&str>) -> Sequence {
        kseq(
            cause,
            &[
                ("A", "occ", Some("A"), F),
                ("A", "rep", Some("A"), R),
                ("B", "occ", Some("B"), F),
                ("B", "rep", Some("B"), R),
                ("C", "occ", Some("C"), F),
            ],
        )
    }

    fn reduced_with_decisions(sequence: Sequence) -> (Vec<(String, String)>, ReductionReport) {
        let targets = ["T1", "T2"];
        let mut reducer = SequenceReducer::with_cycle_groups(vec![
            decision("A", "A", &["T1"], &targets),
            decision("B", "B", &[], &targets),
        ]);
        reducer.push(sequence);
        let report = reducer.report().clone();
        (sig_of(&reducer.cleaned()[0]), report)
    }

    #[test]
    fn a_group_kept_for_the_end_cause_keeps_its_whole_history() {
        let (on_t1, report) = reduced_with_decisions(two_cycles_then(Some("T1")));
        assert_eq!(
            on_t1,
            vec![
                ("A".into(), "occ".into()),
                ("A".into(), "rep".into()),
                ("C".into(), "occ".into())
            ]
        );
        assert_eq!(report.contradicted, 0);
        assert_eq!(report.cycle_groups.as_ref().map(Vec::len), Some(2));
        assert!(!report.cycle_groups_unknown);
        // On T2, A is cancellable like B.
        let (on_t2, _) = reduced_with_decisions(two_cycles_then(Some("T2")));
        assert_eq!(on_t2, vec![("C".into(), "occ".into())]);
    }

    #[test]
    fn a_trajectory_that_reached_no_target_cancels_every_group() {
        let (none, _) = reduced_with_decisions(two_cycles_then(None));
        assert_eq!(none, vec![("C".into(), "occ".into())]);
    }

    #[test]
    fn without_decisions_every_group_is_cancelled_and_the_report_says_so() {
        let mut reducer = SequenceReducer::new();
        reducer.push(two_cycles_then(Some("T1")));
        assert!(reducer.report().cycle_groups.is_none());
        assert!(reducer.report().cycle_groups_unknown);
        assert_eq!(
            sig_of(&reducer.cleaned()[0]),
            vec![("C".into(), "occ".into())]
        );
        // A corpus without any cycle group needs no decision.
        let mut plain = SequenceReducer::new();
        plain.push(kseq(Some("T1"), &[("C", "occ", None, F)]));
        assert!(!plain.report().cycle_groups_unknown);
    }

    #[test]
    fn the_safety_net_still_runs_on_the_groups_left_cancellable() {
        // B is cancellable for T2 and is the only failure: cancelled, the
        // trajectory would be left with the target's observation alone.
        let mut reducer =
            SequenceReducer::with_cycle_groups(vec![decision("B", "B", &[], &["T2"])]);
        reducer.push(kseq(
            Some("T2"),
            &[
                ("B", "occ", Some("B"), F),
                ("B", "rep", Some("B"), R),
                ("ER", "occ", Some("ER"), O),
            ],
        ));
        assert_eq!(reducer.report().contradicted, 1);
    }

    /// The reduction up to 0.81.0: group, filter, group again.
    fn grouped_twice(raw: Vec<Sequence>) -> Vec<Sequence> {
        group_sequences(filter_cycles(group_sequences(raw)))
    }

    /// A trajectory of `cycles` transient failure/repair cycles of `fm`,
    /// then the persistent failures `persistent`, then the feared event,
    /// dated so that no two trajectories share a date.
    fn cycling(fm: &str, cycles: usize, persistent: &[&str], date: f64) -> Sequence {
        let mut events = Vec::new();
        for _ in 0..cycles {
            for attr in ["occ", "rep"] {
                events.push(SeqEvent {
                    obj: fm.into(),
                    automaton: String::new(),
                    attr: attr.into(),
                    time: date,
                    cycle_group: Some("life".into()),
                    kind: None,
                });
            }
        }
        for p in persistent {
            events.push(SeqEvent {
                obj: (*p).into(),
                automaton: String::new(),
                attr: "occ".into(),
                time: date + 1.0,
                cycle_group: Some("life".into()),
                kind: None,
            });
        }
        events.push(SeqEvent {
            obj: "ER".into(),
            automaton: String::new(),
            attr: "occ".into(),
            time: date + 2.0,
            cycle_group: None,
            kind: None,
        });
        Sequence {
            events,
            end_cause: Some("F".into()),
            end_time: date + 2.0,
            weight: 1.0,
        }
    }

    #[test]
    fn the_reducer_holds_one_entry_per_cleaned_path_not_per_trajectory() {
        // 3 000 trajectories, each cycling its own way (which mode, how
        // many times): 3 000 distinct raw signatures, two cleaned paths.
        let mut reducer = SequenceReducer::new();
        for i in 0..3_000_usize {
            let persistent: &[&str] = if i % 3 == 0 { &["A", "B"] } else { &["B", "A"] };
            let mode = format!("C{}", i / 30);
            reducer.push(cycling(&mode, 1 + i % 30, persistent, i as f64));
            assert!(reducer.len() <= 2, "held {} entries", reducer.len());
        }
        let cleaned = reducer.cleaned();
        assert_eq!(cleaned.len(), 2);
        assert_eq!(cleaned[0].weight, 2_000.0);
        assert_eq!(cleaned[1].weight, 1_000.0);
    }

    #[test]
    fn cleaning_one_trajectory_at_a_time_keeps_the_groups_and_weights_of_grouping_twice() {
        let mut raw = Vec::new();
        for i in 0..400_usize {
            let persistent: &[&str] = match i % 4 {
                0 => &["A"],
                1 => &["A", "B"],
                2 => &["B", "A"],
                _ => &[],
            };
            raw.push(cycling(
                ["C", "D"][i % 2],
                i % 5,
                persistent,
                (i % 7) as f64 * 0.37,
            ));
        }
        // Weights here are trajectory counts, hence whole: an exact,
        // totally ordered key.
        let key = |s: &Sequence| (s.end_cause.clone(), sig_of(s), s.weight as u64);
        let mut streamed: Vec<_> = clean(raw.clone()).iter().map(key).collect();
        let mut reference: Vec<_> = grouped_twice(raw.clone()).iter().map(key).collect();
        streamed.sort();
        reference.sort();
        assert_eq!(streamed, reference);
        // Averaged dates agree to rounding.
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-12 * x.abs().max(1.0);
        let reference = grouped_twice(raw.clone());
        for s in clean(raw.clone()) {
            assert!(
                reference.iter().any(|r| {
                    r.end_cause == s.end_cause
                        && sig_of(r) == sig_of(&s)
                        && close(r.end_time, s.end_time)
                        && r.events
                            .iter()
                            .zip(&s.events)
                            .all(|(a, b)| close(a.time, b.time))
                }),
                "no path of the reference reduction matches {:?}",
                sig_of(&s)
            );
        }
        // And so does the minimal level.
        let minimal = |levels: Vec<Sequence>| {
            let mut m: Vec<_> = minimal_sequences(levels).iter().map(key).collect();
            m.sort();
            m
        };
        assert_eq!(minimal(clean(raw.clone())), minimal(grouped_twice(raw)));
    }

    #[test]
    fn the_minimal_level_does_not_depend_on_the_order_of_equal_weight_paths() {
        // r0 = [B, ER], r1 = [D cycle, B, ER], r2 = r3 = [C cycle, A, ER],
        // r4 = [A, B, ER]: [A, ER] and [B, ER] both weigh 2, and [A, B, ER]
        // contains both. Grouping before the filter puts [A, ER] first,
        // filtering first puts [B, ER] first; the absorbed trajectory must
        // go to the same minimal path either way.
        let cycle = |fm: &'static str| vec![(fm, "occ", Some("life")), (fm, "rep", Some("life"))];
        let path = |prefix: Vec<(&'static str, &'static str, Option<&'static str>)>,
                    tail: &[&'static str]| {
            let mut events = prefix;
            events.extend(tail.iter().map(|o| (*o, "occ", None)));
            seq("F", 1.0, &events)
        };
        let raw = vec![
            path(vec![], &["B", "ER"]),
            path(cycle("D"), &["B", "ER"]),
            path(cycle("C"), &["A", "ER"]),
            path(cycle("C"), &["A", "ER"]),
            path(vec![], &["A", "B", "ER"]),
        ];
        let key =
            |m: &[Sequence]| -> Vec<_> { m.iter().map(|s| (sig_of(s), s.weight as u64)).collect() };
        let streamed = minimal_sequences(clean(raw.clone()));
        let mut reversed = clean(raw.clone());
        reversed.reverse();
        assert_eq!(key(&streamed), key(&minimal_sequences(grouped_twice(raw))));
        assert_eq!(key(&streamed), key(&minimal_sequences(reversed)));
        // Ties go to the smaller signature: [A, ER] takes the absorbed one.
        assert_eq!(
            key(&streamed),
            vec![
                (
                    vec![("A".into(), "occ".into()), ("ER".into(), "occ".into())],
                    3
                ),
                (
                    vec![("B".into(), "occ".into()), ("ER".into(), "occ".into())],
                    2
                ),
            ]
        );
    }
}
