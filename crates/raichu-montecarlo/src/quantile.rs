//! Quantile summary of a replica column, at a memory logarithmic in its length.
//!
//! A Monte-Carlo quantile is a nearest-rank order statistic of the
//! per-replica values at one schedule instant. Keeping the whole column
//! makes the memory grow with the number of replicas; this summary grows
//! only with its logarithm, at a **rank error** fixed in advance.
//!
//! The summary is the deterministic one of Greenwald and Khanna,
//! *Space-efficient online computation of quantile summaries*, SIGMOD
//! 2001, doi:10.1145/375663.375670. It holds sorted tuples `(v, g, Δ)`:
//! the smallest possible rank of `v` is the sum of the `g` up to it, the
//! largest that sum plus `Δ`. With rank error `ε`, every tuple keeps
//! `g + Δ ≤ 2εn`, so for any order `q` some tuple's rank is within `εn` of
//! the requested rank `⌈qn⌉`: the answer is a sample value whose rank in
//! the sorted column is off by at most `⌊εn⌋`.
//!
//! Three properties the Monte-Carlo contract needs:
//!
//! - **Exact on small campaigns.** Two tuples merge only when
//!   `g + g' + Δ' ≤ 2εn`, which needs `2εn ≥ 2`: below `1/ε` replicas no
//!   tuple ever merges and the summary is the sorted column itself, so the
//!   answer is the exact nearest-rank value.
//! - **Exact at `ε = 0`.** The column is then kept whole and sorted when
//!   asked: the unbounded exact path, by explicit choice.
//! - **Deterministic.** Values arrive in replica order, merges happen at
//!   fixed counts and sort under [`f64::total_cmp`], so the result is the
//!   same bytes whatever the thread count.
//!
//! Values are inserted in sorted batches rather than one at a time: a
//! new value lands before the first tuple `s` it is smaller than, with
//! `Δ = g_s + Δ_s − 1`. Its true rank then lies between the rank bound of
//! its predecessor plus its place in the batch, and the largest rank of
//! `s` minus one plus that same place, which is what the tuple records.
//! A value below every tuple, or not below any, has an exact rank (`Δ = 0`).
//! The minimum and the maximum stay exact.

use std::cmp::Ordering;

/// Rank error used when a campaign asks for quantiles without stating
/// one: a quantile is the exact nearest-rank value below 1 000 replicas,
/// and beyond it a sample value whose rank is within 0.1 % of the
/// campaign size of the requested one.
pub const DEFAULT_QUANTILE_RANK_ERROR: f64 = 1e-3;

/// The fewest pending values merged at once, so a large `ε` does not pay a
/// merge pass per replica (GK itself merges once per `1/(2ε)` insertions).
const MIN_MERGE_BATCH: usize = 64;

/// One summary tuple: a sample value, the number of sample values it
/// stands for (`g`), and the spread of its rank (`delta`).
#[derive(Debug, Clone, Copy)]
struct Tuple {
    value: f64,
    g: u64,
    delta: u64,
}

/// A quantile summary of one column of replica values.
#[derive(Debug, Clone)]
pub(crate) struct QuantileSketch {
    rank_error: f64,
    count: u64,
    tuples: Vec<Tuple>,
    /// Values not yet merged; the whole column while `rank_error == 0`.
    pending: Vec<f64>,
    batch: usize,
}

impl QuantileSketch {
    /// An empty summary with rank error `rank_error`, in `[0, 0.5]`
    /// (checked by the campaign before any replica).
    pub(crate) fn new(rank_error: f64) -> Self {
        let batch = if rank_error > 0.0 {
            ((0.5 / rank_error).ceil() as usize).max(MIN_MERGE_BATCH)
        } else {
            usize::MAX
        };
        Self {
            rank_error,
            count: 0,
            tuples: Vec::new(),
            pending: Vec::new(),
            batch,
        }
    }

    /// Add the next replica value.
    pub(crate) fn push(&mut self, value: f64) {
        self.pending.push(value);
        if self.pending.len() >= self.batch {
            self.merge_pending();
        }
    }

    /// The nearest-rank `q`-quantile of the pushed values: exact while
    /// nothing has merged, else a pushed value whose rank is within
    /// `⌊εn⌋` of `max(1, ⌈qn⌉)`. `NaN` on an empty summary.
    pub(crate) fn quantile(&mut self, q: f64) -> f64 {
        if self.rank_error == 0.0 {
            self.pending.sort_unstable_by(f64::total_cmp);
            return nearest_rank(&self.pending, q);
        }
        self.merge_pending();
        let n = self.count;
        if n == 0 {
            return f64::NAN;
        }
        let rank = target_rank(q, n);
        if self.tuples.len() as u64 == n {
            // Nothing merged: the tuples are the sorted column.
            return self.tuples[(rank - 1) as usize].value;
        }
        // The first and the last tuples hold the minimum and the maximum
        // exactly; the band scan below could stop on an earlier tuple.
        if rank == 1 {
            return self.tuples[0].value;
        }
        if rank == n {
            return self.tuples[self.tuples.len() - 1].value;
        }
        // Half the widest rank band: at most εn by the merge rule, and the
        // GK lemma guarantees a tuple within it on both sides.
        let half_band = self.tuples.iter().map(|t| t.g + t.delta).max().unwrap_or(0) as f64 / 2.0;
        let mut rank_min = 0u64;
        for t in &self.tuples {
            rank_min += t.g;
            let rank_max = rank_min + t.delta;
            if rank as f64 - rank_min as f64 <= half_band
                && rank_max as f64 - rank as f64 <= half_band
            {
                return t.value;
            }
        }
        // Unreachable under the invariant; the maximum is the safe answer.
        self.tuples.last().map_or(f64::NAN, |t| t.value)
    }

    /// Merge the pending values into the summary, then compress it.
    fn merge_pending(&mut self) {
        if self.pending.is_empty() || self.rank_error == 0.0 {
            return;
        }
        self.pending.sort_unstable_by(f64::total_cmp);
        let old = std::mem::take(&mut self.tuples);
        let mut merged = Vec::with_capacity(old.len() + self.pending.len());
        let mut old_iter = old.into_iter().peekable();
        for &value in &self.pending {
            while let Some(t) = old_iter.next_if(|t| t.value.total_cmp(&value) != Ordering::Greater)
            {
                merged.push(t);
            }
            let delta = match (merged.is_empty(), old_iter.peek()) {
                // Below every value seen, or above every one: exact rank.
                (true, _) | (_, None) => 0,
                (false, Some(successor)) => successor.g + successor.delta - 1,
            };
            merged.push(Tuple { value, g: 1, delta });
        }
        merged.extend(old_iter);
        self.count += self.pending.len() as u64;
        self.pending.clear();
        self.tuples = merged;
        self.compress();
    }

    /// Merge each tuple into its successor while the GK band
    /// `g + g' + Δ' ≤ 2εn` allows it, scanning from the top. The first
    /// tuple (the minimum) is never absorbed; the last keeps the maximum.
    fn compress(&mut self) {
        let len = self.tuples.len();
        if len < 3 {
            return;
        }
        let band = (2.0 * self.rank_error * self.count as f64).floor() as u64;
        if band < 2 {
            return;
        }
        let mut kept = Vec::with_capacity(len);
        let mut head = self.tuples[len - 1];
        for &t in self.tuples[1..len - 1].iter().rev() {
            if t.g + head.g + head.delta <= band {
                head.g += t.g;
            } else {
                kept.push(head);
                head = t;
            }
        }
        kept.push(head);
        kept.push(self.tuples[0]);
        kept.reverse();
        self.tuples = kept;
    }

    /// Values held in memory: summary tuples plus pending values.
    #[cfg(test)]
    fn footprint(&self) -> usize {
        self.tuples.len() + self.pending.len()
    }
}

/// The 1-based nearest rank of order `q` among `n` values.
fn target_rank(q: f64, n: u64) -> u64 {
    ((q * n as f64).ceil() as u64).clamp(1, n)
}

/// The nearest-rank `q`-quantile of a sorted column (`NaN` when empty).
pub(crate) fn nearest_rank(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[(target_rank(q, sorted.len() as u64) - 1) as usize]
}

/// The nearest-rank `q`-quantile of a 0/1 column holding `ones` ones among
/// `n` values, exact from the count alone: the sorted column is `n − ones`
/// zeros then the ones.
pub(crate) fn bernoulli_quantile(q: f64, n: u64, ones: u64) -> f64 {
    if n == 0 {
        return f64::NAN;
    }
    if target_rank(q, n) > n - ones {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORDERS: [f64; 9] = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0];

    /// A deterministic, scrambled sample (no RNG dependency): a linear
    /// congruential stream mapped to [0, 1), cubed to skew it.
    fn sample(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                u * u * u * 100.0
            })
            .collect()
    }

    fn sketch_of(values: &[f64], rank_error: f64) -> QuantileSketch {
        let mut sketch = QuantileSketch::new(rank_error);
        for &v in values {
            sketch.push(v);
        }
        sketch
    }

    /// The ranks a value occupies in the sorted column (1-based, ties
    /// spanning a range).
    fn rank_range(sorted: &[f64], value: f64) -> (u64, u64) {
        let low = sorted.partition_point(|x| x.total_cmp(&value) == Ordering::Less) as u64 + 1;
        let high = sorted.partition_point(|x| x.total_cmp(&value) != Ordering::Greater) as u64;
        (low, high)
    }

    fn assert_rank_error_within(values: &[f64], rank_error: f64) {
        let mut sorted = values.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        let n = values.len() as u64;
        let tolerance = (rank_error * n as f64).floor() as u64;
        let mut sketch = sketch_of(values, rank_error);
        for q in ORDERS {
            let answer = sketch.quantile(q);
            let (low, high) = rank_range(&sorted, answer);
            assert!(low <= high, "q={q}: the answer {answer} is a pushed value");
            let wanted = target_rank(q, n);
            let off = if wanted < low {
                low - wanted
            } else {
                wanted.saturating_sub(high)
            };
            assert!(
                off <= tolerance,
                "q={q}, n={n}, eps={rank_error}: rank off by {off} > {tolerance}"
            );
        }
    }

    #[test]
    fn below_one_over_epsilon_the_summary_is_exact() {
        let values = sample(999, 7);
        let mut sorted = values.clone();
        sorted.sort_unstable_by(f64::total_cmp);
        let mut sketch = sketch_of(&values, 1e-3);
        for q in ORDERS {
            assert_eq!(
                sketch.quantile(q).to_bits(),
                nearest_rank(&sorted, q).to_bits(),
                "q={q}"
            );
        }
        assert_eq!(sketch.tuples.len(), 999, "no tuple merged");
    }

    #[test]
    fn zero_rank_error_is_the_exact_nearest_rank() {
        let values = sample(20_000, 3);
        let mut sorted = values.clone();
        sorted.sort_unstable_by(f64::total_cmp);
        let mut sketch = sketch_of(&values, 0.0);
        for q in ORDERS {
            assert_eq!(
                sketch.quantile(q).to_bits(),
                nearest_rank(&sorted, q).to_bits()
            );
        }
    }

    #[test]
    fn rank_error_holds_beyond_one_over_epsilon() {
        for (n, eps, seed) in [
            (50_000, 1e-3, 1),
            (200_000, 1e-3, 2),
            (30_000, 1e-2, 3),
            (5_000, 0.05, 4),
        ] {
            assert_rank_error_within(&sample(n, seed), eps);
        }
    }

    #[test]
    fn rank_error_holds_on_sorted_and_reversed_streams() {
        let mut ascending = sample(40_000, 5);
        ascending.sort_unstable_by(f64::total_cmp);
        assert_rank_error_within(&ascending, 1e-3);
        let descending: Vec<f64> = ascending.iter().rev().copied().collect();
        assert_rank_error_within(&descending, 1e-3);
    }

    #[test]
    fn rank_error_holds_on_a_few_distinct_values() {
        // An occurrence count: small integers, heavily tied.
        let values: Vec<f64> = sample(100_000, 6)
            .iter()
            .map(|v| (v / 20.0).floor())
            .collect();
        assert_rank_error_within(&values, 1e-3);
    }

    #[test]
    fn min_and_max_stay_exact() {
        let values = sample(100_000, 8);
        let mut sketch = sketch_of(&values, 1e-2);
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(sketch.quantile(0.0), min);
        assert_eq!(sketch.quantile(1.0), max);
    }

    #[test]
    fn min_and_max_stay_exact_on_sorted_streams() {
        for eps in [1e-3, 1e-2, 0.25] {
            let ascending: Vec<f64> = (0..100_000).map(f64::from).collect();
            let descending: Vec<f64> = ascending.iter().rev().copied().collect();
            for stream in [&ascending, &descending] {
                let mut sketch = sketch_of(stream, eps);
                assert_eq!(sketch.quantile(0.0), 0.0, "eps={eps}");
                assert_eq!(sketch.quantile(1.0), 99_999.0, "eps={eps}");
            }
        }
    }

    #[test]
    fn memory_stays_bounded_on_a_count_column() {
        // A count column (few distinct values, heavily tied) is the shape
        // whose summary grows most with the replica count.
        let mut sketch = QuantileSketch::new(1e-3);
        for v in sample(1_000_000, 11) {
            sketch.push((v / 10.0).floor());
        }
        let footprint = sketch.footprint();
        assert!(footprint < 10_000, "footprint {footprint} at n = 1e6");
    }

    #[test]
    fn memory_stays_bounded_in_the_number_of_replicas() {
        let eps = 1e-3;
        let mut sketch = QuantileSketch::new(eps);
        let mut peak_small = 0;
        for (i, v) in sample(1_000_000, 9).into_iter().enumerate() {
            sketch.push(v);
            if i < 100_000 {
                peak_small = peak_small.max(sketch.footprint());
            }
        }
        let at_million = sketch.footprint();
        assert!(at_million < 10_000, "footprint {at_million} at n = 1e6");
        assert!(
            at_million < 2 * peak_small,
            "footprint {at_million} at 1e6 vs {peak_small} up to 1e5"
        );
    }

    #[test]
    fn the_same_stream_gives_the_same_bytes() {
        let values = sample(60_000, 10);
        let a = sketch_of(&values, 1e-3).quantile(0.95);
        let b = sketch_of(&values, 1e-3).quantile(0.95);
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn bernoulli_quantile_reads_the_sorted_zero_one_column() {
        for n in 1..40u64 {
            for ones in 0..=n {
                let column: Vec<f64> = (0..n)
                    .map(|i| if i < n - ones { 0.0 } else { 1.0 })
                    .collect();
                for q in ORDERS {
                    assert_eq!(bernoulli_quantile(q, n, ones), nearest_rank(&column, q));
                }
            }
        }
    }

    #[test]
    fn an_empty_summary_answers_nan() {
        assert!(QuantileSketch::new(1e-3).quantile(0.5).is_nan());
        assert!(QuantileSketch::new(0.0).quantile(0.5).is_nan());
    }
}
