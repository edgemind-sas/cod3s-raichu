//! Sequence-analysis pipeline (F3a-A2): the native RAMS post-processing of
//! a Monte-Carlo corpus of recorded trajectory [`Sequence`]s, porting the
//! cod3s `SequenceAnalyser` canonical pipeline
//! (`group_sequences → filter_cycles → minimal_sequences`).
//!
//! - **group** buckets by end cause (target) and merges sequences with an
//!   identical ordered `(obj, attr)` signature, summing weights;
//! - **filter cycles** drops transient failure→repair cycles that net out
//!   before the feared event (a failure mode's monitored occ/rep events
//!   strictly alternate, so an even per-group count cancels entirely and an
//!   odd one leaves only the last, persistent failure);
//! - **minimal** greedily keeps the shortest sequences and absorbs every
//!   longer super-sequence that includes one (order-dependent, matching
//!   cod3s' `compute_minimal_sequences`).

use crate::engine::{SeqEvent, Sequence};

/// Ordered `(obj, attr)` signature of a sequence — the identity used for
/// grouping and subsequence inclusion (times and cycle groups excluded).
fn signature(seq: &Sequence) -> Vec<(&str, &str)> {
    seq.events
        .iter()
        .map(|e| (e.obj.as_str(), e.attr.as_str()))
        .collect()
}

/// Group raw trajectory sequences: bucket by end cause, merge sequences with
/// an identical ordered `(obj, attr)` signature (weights summed, event and
/// end times averaged), and sort each bucket by descending weight (ties keep
/// first-seen order). Deterministic.
#[must_use]
pub fn group_sequences(raw: Vec<Sequence>) -> Vec<Sequence> {
    // (end_cause, signature) → index into `merged`, preserving first-seen
    // order (a plain Vec scan keeps the reduction byte-deterministic).
    let mut merged: Vec<Sequence> = Vec::new();
    let mut counts: Vec<f64> = Vec::new();
    for seq in raw {
        let key = (seq.end_cause.clone(), signature(&seq).to_owned());
        let pos = merged.iter().position(|m| {
            m.end_cause == key.0
                && signature(m)
                    .iter()
                    .map(|(o, a)| (o.to_string(), a.to_string()))
                    .eq(key.1.iter().map(|(o, a)| (o.to_string(), a.to_string())))
        });
        match pos {
            Some(i) => {
                let n = counts[i] + seq.weight;
                // Weight-averaged event and end times.
                for (acc, ev) in merged[i].events.iter_mut().zip(&seq.events) {
                    acc.time = (acc.time * counts[i] + ev.time * seq.weight) / n;
                }
                merged[i].end_time =
                    (merged[i].end_time * counts[i] + seq.end_time * seq.weight) / n;
                merged[i].weight += seq.weight;
                counts[i] = n;
            }
            None => {
                counts.push(seq.weight);
                merged.push(seq);
            }
        }
    }
    // Stable sort by descending weight (ties keep first-seen order).
    merged.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    merged
}

/// Drop transient failure→repair cycles. Within each `cycle_group` the
/// monitored events strictly alternate (occ, rep, occ, …) starting with the
/// failure, so an even per-group count nets out entirely (repaired before
/// the feared event) and an odd one leaves only the last, persistent
/// failure. Events with no cycle group are always kept, order preserved.
#[must_use]
pub fn filter_cycles(sequences: Vec<Sequence>) -> Vec<Sequence> {
    sequences
        .into_iter()
        .map(|mut seq| {
            // For each (component, group) pair, the index of its last event
            // (kept iff the pair has an odd count); every other grouped
            // event is dropped. The component is part of the key so two
            // DIFFERENT failure modes sharing an automaton name (e.g. every
            // ObjFM's `fm__cc_1`) can never cancel each other.
            use std::collections::HashMap;
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
        })
        .collect()
}

/// `short` is an ordered (non-contiguous) subsequence of `long`.
fn is_included(short: &[(&str, &str)], long: &[(&str, &str)]) -> bool {
    let mut it = long.iter();
    short.iter().all(|s| it.any(|l| l == s))
}

/// Minimal sequences (greedy, order-dependent — cod3s' `compute_minimal_sequences`):
/// per end cause, sort by ascending length (ties by descending weight); keep
/// each sequence unless a shorter already-kept one is included in it, in
/// which case absorb it (add its weight to that minimal). Then sort by
/// descending weight.
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
    out.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    out
}

/// The full pipeline on a raw Monte-Carlo corpus: group → filter cycles →
/// group again (cycle filtering can make distinct raw sequences coincide) →
/// minimal.
#[must_use]
pub fn analyse(raw: Vec<Sequence>) -> Vec<Sequence> {
    minimal_sequences(group_sequences(filter_cycles(group_sequences(raw))))
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
        // automaton name (every ObjFM's `fm__cc_1`) — their persistent
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
        // {[ER.occ], [fm1.occ, ER.occ]}? No — [ER.occ] ⊆ [fm1.occ, ER.occ],
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
}
