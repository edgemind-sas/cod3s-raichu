//! Sequence-analysis pipeline (F3a-A2): the native RAMS post-processing of
//! a Monte-Carlo corpus of recorded trajectory [`Sequence`]s, porting the
//! cod3s `SequenceAnalyser` canonical pipeline, chained by [`analyse`]
//! (`filter_cycles → group_sequences → minimal_sequences`).
//!
//! - **filter cycles** drops transient failure→repair cycles that net out
//!   before the feared event (a failure mode's monitored occ/rep events
//!   strictly alternate, so an even per-group count cancels entirely and an
//!   odd one leaves only the last, persistent failure);
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

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use raichu_core::{SeqEvent, Sequence};

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
/// a campaign of any size reduces in bounded memory.
///
/// Pushing the trajectories of a corpus in replica order and taking
/// [`SequenceReducer::cleaned`] is [`clean`]; the result is deterministic
/// for a given push order.
#[derive(Debug, Default)]
pub struct SequenceReducer {
    grouping: Grouping,
}

impl SequenceReducer {
    /// An empty reduction.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one trajectory.
    pub fn push(&mut self, sequence: Sequence) {
        self.grouping.merge(filter_sequence_cycles(sequence));
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
#[must_use]
pub fn filter_cycles(sequences: Vec<Sequence>) -> Vec<Sequence> {
    sequences.into_iter().map(filter_sequence_cycles).collect()
}

/// [`filter_cycles`] on one sequence.
fn filter_sequence_cycles(mut seq: Sequence) -> Sequence {
    // For each (component, group) pair, the index of its last event (kept
    // iff the pair has an odd count); every other grouped event is
    // dropped. The component is part of the key so two DIFFERENT failure
    // modes sharing an automaton name (e.g. every ObjFM's `fm__cc_1`) can
    // never cancel each other.
    let mut last: HashMap<(&str, &str), (usize, usize)> = HashMap::new();
    for (i, ev) in seq.events.iter().enumerate() {
        if let Some(g) = ev.cycle_group.as_deref() {
            let entry = last.entry((ev.obj.as_str(), g)).or_insert((0, i));
            entry.0 += 1;
            entry.1 = i;
        }
    }
    let keep_idx: std::collections::HashSet<usize> = last
        .values()
        .filter(|(count, _)| count % 2 == 1)
        .map(|(_, idx)| *idx)
        .collect();
    let kept: Vec<SeqEvent> = seq
        .events
        .drain(..)
        .enumerate()
        .filter(|(i, ev)| ev.cycle_group.is_none() || keep_idx.contains(i))
        .map(|(_, ev)| ev)
        .collect();
    seq.events = kept;
    seq
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
            });
        }
        events.push(SeqEvent {
            obj: "ER".into(),
            automaton: String::new(),
            attr: "occ".into(),
            time: date + 2.0,
            cycle_group: None,
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
