//! Confidence intervals on the Monte-Carlo estimators.
//!
//! A campaign reports `P(feared event) = 0.0483`. Nothing in that number
//! says whether it came from 500 replicas or from 100 000, so nothing
//! says at what precision the figure is known. The dispersion statistics
//! already reported (standard deviation, quantiles) describe the
//! **population of replicas**, not the **estimator**: they answer "how
//! spread out are the trajectories", never "is this figure converged".
//! An interval answers the second question, and it is the second one a
//! safety deliverable asks.
//!
//! # What is computed
//!
//! Every estimator of this crate is a mean over `n` independent replicas,
//! so each one carries a two-sided interval at a **declared** level, kept
//! next to the bounds so an artefact can never state a figure without
//! stating its precision. Several constructions are used, and which one
//! applies is decided from the **declared type of the indicator**, never
//! from a scan of the sampled values:
//!
//! - [`IntervalMethod::Wilson`] for an estimator that is a **probability**
//!   by construction: the sampled value of a state indicator, or of a
//!   boolean attribute. The model says the per-replica draw lies in
//!   `{0, 1}`, so the estimator is a binomial proportion and the Wilson
//!   score interval applies. It is preferred to the textbook
//!   `p ± z·√(p(1−p)/n)` for one decisive reason: on a rare feared event
//!   that no replica reached, the textbook interval collapses to
//!   `[0, 0]`, which claims the event is impossible on the strength of a
//!   finite campaign. Wilson returns `[0, z²/(n + z²)]`, which is what
//!   the campaign actually establishes. Its bounds also stay inside
//!   `[0, 1]` at every sample size.
//! - [`IntervalMethod::Normal`] for every other estimator (cumulated
//!   sojourn, occurrence count, a non-boolean attribute): the
//!   central-limit interval `mean ± z·s/√n`.
//! - [`IntervalMethod::WeightedNormal`] for the estimate of a **biased
//!   (likelihood-weighted) campaign**: the mean of `Yᵢ = Wᵢ·1{hitᵢ}`,
//!   with `Wᵢ` the likelihood ratio of replica `i`. The bounds are the
//!   same central-limit construction, but under their own name: a
//!   likelihood weight is unbounded, so the estimate is not a
//!   proportion, Wilson does not apply, and an artefact must never
//!   claim a plain proportion's interval for it. See
//!   [`weighted_interval`], which also reports the weight diagnostics
//!   (effective sample size, relative error) that tell a sound interval
//!   from one a few huge weights made look tight.
//!
//! # When every replica said the same thing
//!
//! A sample standard deviation of zero makes the central-limit
//! half-width zero, and the interval collapses onto the point estimate:
//! `1.0000 ± 0.0000`, reported as `[1.000000, 1.000000]`, which claims
//! the mean is *exactly* one on the strength of a finite campaign. That
//! is the defect Wilson was chosen to avoid at `p = 0`, reappearing on
//! the two normal estimators as soon as every replica returns the same
//! value. A guard on the replica count does not catch it: there can be
//! a thousand replicas, all identical.
//!
//! A constant sample establishes exactly one thing, and it is not a
//! spread: **no replica departed from the value in `n` draws**, so the
//! frequency of a departure is a proportion observed at zero, bounded
//! above by [`unobserved_frequency_bound`], `ε = z²/(n + z²)`. That is
//! the rule of three in exact form (`3.84/n` at 95 %), and it is the
//! very expression Wilson already returns at `p = 0`.
//!
//! Turning a frequency into bounds on a mean needs the **size** of one
//! departure, which is no more observable than the spread was. It is
//! read off the declaration of the quantity, exactly as the
//! construction is: see [`Departure`]. An occurrence count moves by one
//! occurrence at least and never below zero; a cumulated sojourn at `t`
//! lives in `[0, t]`, so one departing replica moves it by at most `t`.
//! The interval becomes `value ± ε·size`, one-sided wherever the
//! constant value sits on the edge of what the quantity can take, which
//! is what the rule of three has always been.
//!
//! The construction itself does **not** change. A dispersion of zero is
//! a draw, not a declaration of type, so it may not turn a sojourn into
//! a proportion: [`ConfidenceInterval::method`] still names the
//! construction the model asked for, and
//! [`ConfidenceInterval::constant_sample`] marks the instants where the
//! frequency bound answered in its place. Everywhere the sample moved,
//! the bounds are the ones they were before, to the last bit.
//!
//! # What is deliberately not done
//!
//! **No Student `t` for ordinary replica means.** Independent splitting
//! batches use [`IntervalMethod::BatchStudent`] separately. Substituting `t_{n−1}` for `z` would correct the
//! one term that is *not* the dominant error: `t` is exact only when the
//! per-replica quantity is itself Gaussian, which a sojourn time or an
//! occurrence count never is. At the sizes a campaign runs the two
//! coincide (`t = 1.9647` against `z = 1.9600` at `n = 500`), and at the
//! sizes where they differ the normality assumption behind `t` has
//! already failed. Reporting `t` there would trade a visible
//! approximation for an invisible one. The binary case, the one where a
//! genuinely small-sample construction exists, gets Wilson instead.
//!
//! **No clamping of the normal bounds to the support.** A negative lower
//! bound on a sojourn time is the interval reporting that the normal
//! approximation is out of its range at this sample size. Clamping it to
//! zero would erase that signal and leave a bound that looks sound.
//!
//! **No interval where none exists.** Below two replicas nothing about
//! dispersion is observable, and the normal construction would answer
//! `[mean, mean]`: perfect precision from one draw. That case is reported
//! as [`IntervalMethod::Undefined`] instead, bounds equal to the point
//! estimate and no coverage claimed.
//!
//! **No pretence of coverage on an unbounded quantity.** An occurrence
//! count has no upper end, so on a constant sample no finite bound on
//! its mean holds in the worst case: a replica that departed could have
//! counted a thousand. Charging it one occurrence is the *narrowest*
//! statement the observation admits, not the widest, and it is exact
//! when the count is 0 or 1, which a target-stopped campaign
//! guarantees. A cumulated sojourn, bounded by the elapsed time, does
//! get a conservative bound. Either way the alternative is a width of
//! zero, which claims a precision no campaign can buy.

use serde::{Deserialize, Serialize};

/// Confidence level applied when a study states none: the conventional
/// two-sided 95 %.
///
/// It is a **default**, not a constant of the method: every entry point
/// takes the level as a study parameter, and the level actually applied
/// travels with the bounds (see [`ConfidenceInterval::level`]) so a
/// reader never has to assume it.
pub const DEFAULT_CONFIDENCE: f64 = 0.95;

/// How the bounds of a [`ConfidenceInterval`] were obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntervalMethod {
    /// Wilson score interval on a binomial proportion: the estimator is
    /// a probability by declaration (a state indicator, or a boolean
    /// attribute). Bounds always lie in `[0, 1]`, and a campaign that
    /// observed no occurrence still yields a non-degenerate upper bound.
    Wilson,
    /// Central-limit interval `mean ± z·s/√n` on a general mean
    /// (cumulated sojourn, occurrence count, non-boolean attribute).
    /// Asymptotic: its coverage reaches the nominal one as `n` grows.
    Normal,
    /// Central-limit interval `mean ± z·s/√n` on a **weighted
    /// indicator** `Yᵢ = Wᵢ·1{hitᵢ}`, `Wᵢ` the likelihood ratio of a
    /// biased replica. The construction is [`Self::Normal`]'s; the
    /// distinct name records that the estimate is a weighted mean, not
    /// a proportion. Not clamped, per the crate policy.
    WeightedNormal,
    /// Student interval on independent splitting batch estimates.
    BatchStudent,
    /// No interval could be formed: fewer than two replicas, so no
    /// dispersion is observable (or, on a weighted indicator, no replica
    /// hit at all). The bounds repeat the point estimate
    /// and carry **no** coverage guarantee.
    Undefined,
}

/// A two-sided confidence interval on one estimator, over the schedule
/// instants.
///
/// The level and the construction travel with the bounds: an artefact
/// that carries this carries everything needed to read it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Confidence level these bounds were computed at, in `(0, 1)`: the
    /// study parameter, recorded rather than assumed.
    pub level: f64,
    /// Which construction produced the bounds.
    pub method: IntervalMethod,
    /// Lower bound at each schedule instant.
    pub low: Vec<f64>,
    /// Upper bound at each schedule instant.
    pub high: Vec<f64>,
    /// Whether the sample was **constant** across the replicas at each
    /// instant: no dispersion at all was observed there, so the bound
    /// reported is the one the *frequency* of a departure establishes
    /// (`z²/(n + z²)`, see [`unobserved_frequency_bound`]) and not one
    /// built on an observed spread.
    ///
    /// It marks where a bound came from, never which construction
    /// applied: [`Self::method`] is read off the model and a constant
    /// draw does not change it. `false` everywhere the sample moved,
    /// and everywhere no interval was formed at all.
    pub constant_sample: Vec<bool>,
}

impl ConfidenceInterval {
    /// Wilson score intervals on a sequence of **proportion** estimates
    /// (one per schedule instant), each measured on the same `n`
    /// replicas.
    #[must_use]
    pub fn on_proportion(level: f64, n: u64, means: &[f64]) -> Self {
        if n == 0 {
            return Self::undefined(level, means);
        }
        // The deviate depends on the level alone: computed once for the
        // whole series rather than once per instant.
        let z = z_of(level);
        let (low, high) = means
            .iter()
            .map(|&p| wilson_at(p, n, z))
            .collect::<(Vec<f64>, Vec<f64>)>();
        Self {
            level,
            method: IntervalMethod::Wilson,
            low,
            high,
            // Wilson needs no fallback: at `p = 0` and at `p = 1`, the
            // only two constant samples a proportion admits, its own
            // closed form already *is* the frequency bound. The flag
            // marks them all the same, so "the campaign saw no
            // dispersion here" reads identically on the three
            // estimators.
            constant_sample: means.iter().map(|&p| p <= 0.0 || p >= 1.0).collect(),
        }
    }

    /// Central-limit intervals on a sequence of **mean** estimates (one
    /// per schedule instant), each with its sample standard deviation
    /// (ddof = 1) over the same `n` replicas.
    ///
    /// `stds` describes the same instants as `means` and is expected to
    /// be as long; a shorter one yields a correspondingly shorter
    /// series rather than an error.
    ///
    /// `departure` is only ever consulted where the observed dispersion
    /// is exactly zero, which is where `mean ± z·s/√n` would answer
    /// with a point. There it says how far one replica that had not
    /// behaved like the others would have moved the estimate, and the
    /// bound becomes `mean ± ε·size` (see [`Departure`] and
    /// [`constant_sample_bounds`]). Everywhere else the central-limit
    /// interval answers unchanged.
    #[must_use]
    pub fn on_mean(
        level: f64,
        n: u64,
        means: &[f64],
        stds: &[f64],
        departure: Departure<'_>,
    ) -> Self {
        if n < 2 {
            return Self::undefined(level, means);
        }
        let z = z_of(level);
        let epsilon = frequency_at(n, z);
        let n_instants = means.len().min(stds.len());
        let mut low = Vec::with_capacity(n_instants);
        let mut high = Vec::with_capacity(n_instants);
        let mut constant_sample = Vec::with_capacity(n_instants);
        for (k, (&mean, &std)) in means.iter().zip(stds).enumerate() {
            // `std` is a square root of a quantity already raised to
            // zero, so it is never negative and never NaN: the test is
            // an equality to zero, not a tolerance. A dispersion merely
            // *small* is a dispersion, and the central-limit interval
            // describes it.
            let constant = std <= 0.0;
            let (bottom, top) = if constant {
                constant_at(mean, departure.size_at(k), departure.floor, epsilon)
            } else {
                normal_at(mean, std, n, z)
            };
            low.push(bottom);
            high.push(top);
            constant_sample.push(constant);
        }
        Self {
            level,
            method: IntervalMethod::Normal,
            low,
            high,
            constant_sample,
        }
    }

    /// The point estimate repeated, with no coverage claimed.
    fn undefined(level: f64, means: &[f64]) -> Self {
        Self {
            level,
            method: IntervalMethod::Undefined,
            low: means.to_vec(),
            high: means.to_vec(),
            // Nothing was established, so nothing is marked: below two
            // replicas there is no sample for a departure to be absent
            // from.
            constant_sample: vec![false; means.len()],
        }
    }
}

/// How far one replica that had **not** behaved like the others would
/// have moved an estimate, where the campaign observed no dispersion at
/// all.
///
/// A constant sample bounds the *frequency* of a departure and says
/// nothing about its size, so the size has to come from elsewhere: from
/// the declaration of the quantity, exactly like the construction, and
/// never from the sample. Each constructor names the quantity it
/// describes and the fact of the model it reads.
#[derive(Debug, Clone, Copy)]
pub struct Departure<'a> {
    /// Departure size at each schedule instant, or `None` when it is
    /// one unit of the quantity at every instant.
    per_instant: Option<&'a [f64]>,
    /// Value the quantity cannot be pushed below, when it declares one.
    /// The interval is then one-sided at that value, which is what the
    /// rule of three has always been.
    floor: Option<f64>,
}

impl<'a> Departure<'a> {
    /// An **occurrence count**: a counting measure, so a departing
    /// replica differs by one occurrence at least, and no departure
    /// takes the mean below zero.
    ///
    /// One occurrence is the *smallest* departure a count admits, not
    /// the largest, because a count has no upper end; see the crate
    /// docs on what that does and does not claim.
    #[must_use]
    pub fn count() -> Self {
        Self {
            per_instant: None,
            floor: Some(0.0),
        }
    }

    /// The sampled value of a **numeric attribute**: one unit of it, on
    /// either side. An attribute that declares neither a support nor a
    /// scale offers nothing else to charge a departure with.
    #[must_use]
    pub fn attribute() -> Self {
        Self {
            per_instant: None,
            floor: None,
        }
    }

    /// The **cumulated sojourn of a 0/1 indicator**, over the schedule
    /// `instants`: it lives in `[0, t]` by construction, so a departing
    /// replica could have spent the whole elapsed time in the state and
    /// none can take the mean below zero.
    ///
    /// The elapsed time, and not "one unit of time", for a reason that
    /// is not cosmetic: a unit of time is the model's own choice, and a
    /// bound charged in units would answer differently on a model
    /// written in hours and on the same model written in seconds.
    #[must_use]
    pub fn sojourn(instants: &'a [f64]) -> Self {
        Self {
            per_instant: Some(instants),
            floor: Some(0.0),
        }
    }

    /// The **time-integral of a numeric indicator**, over the schedule
    /// `instants`: the elapsed time is still its scale, but an
    /// indicator free to take any value gives its integral neither a
    /// sign nor a ceiling.
    #[must_use]
    pub fn integral(instants: &'a [f64]) -> Self {
        Self {
            per_instant: Some(instants),
            floor: None,
        }
    }

    /// Departure size at the `k`-th schedule instant.
    fn size_at(&self, k: usize) -> f64 {
        match self.per_instant {
            None => 1.0,
            // A schedule instant is a duration since the origin: a
            // negative one describes nothing, and a negative size would
            // invert the bounds rather than widen them.
            Some(instants) => instants.get(k).copied().unwrap_or(0.0).max(0.0),
        }
    }
}

/// The estimate of a weighted indicator, its interval and its weight
/// diagnostics: what [`weighted_interval`] returns.
///
/// A biased campaign can produce a handful of huge likelihood weights
/// and an interval that *looks* tight (Glasserman and Wang, 1997): the
/// two diagnostics travel with the bounds so that case is visible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightedInterval {
    /// Confidence level the bounds were computed at.
    pub level: f64,
    /// [`IntervalMethod::WeightedNormal`], or
    /// [`IntervalMethod::Undefined`] below two replicas or when no
    /// replica hit.
    pub method: IntervalMethod,
    /// Replicas in the sample, hits or not.
    pub replicas: u64,
    /// Point estimate: the mean of the weighted indicator values.
    pub estimate: f64,
    /// Standard error of the estimate, `s/√n` with `s` the sample
    /// standard deviation (ddof = 1); 0 when the interval is undefined.
    pub standard_error: f64,
    /// Lower bound (the estimate itself when undefined).
    pub low: f64,
    /// Upper bound (the estimate itself when undefined).
    pub high: f64,
    /// Effective sample size of the hits, `(Σ wᵢ)² / Σ wᵢ²` over the
    /// hit weights: the number of equally weighted hits that would carry
    /// the same information. Equal to the hit count when every hit
    /// weighs the same, close to 1 when one weight dominates, 0 when no
    /// replica hit.
    pub effective_sample_size: f64,
    /// Relative error, standard error over estimate; `None` when the
    /// interval is undefined (no hit, or fewer than two replicas).
    pub relative_error: Option<f64>,
}

/// Normal interval and weight diagnostics of a **weighted indicator**
/// sample, at confidence `level`.
///
/// `values` holds one entry per replica, in replica order: the
/// likelihood weight of the replica if it hit the target, 0 otherwise
/// (`Yᵢ = Wᵢ·1{hitᵢ}`). Values are expected finite and non-negative;
/// the caller (the biased driver) owns that invariant. The non-zero
/// entries are the hit weights the effective sample size is taken over.
///
/// The bounds are [`normal_bounds`] on the mean and ddof-1 standard
/// deviation of `values`, not clamped (crate policy), under
/// [`IntervalMethod::WeightedNormal`]. Two cases get
/// [`IntervalMethod::Undefined`], bounds equal to the estimate and no
/// relative error:
///
/// - fewer than two replicas, as for every estimator of the crate;
/// - **no hit at all**: the sample is all zero, and no bound on a mean
///   of unbounded weights follows from it (the constant-sample rule
///   needs a bounded departure size, which a likelihood weight is not).
///   A quantification refuses this case upstream rather than report it.
#[must_use]
pub fn weighted_interval(values: &[f64], level: f64) -> WeightedInterval {
    let replicas = values.len() as u64;
    let n = values.len() as f64;
    // Serial, replica-ordered sums: the result is bit-reproducible.
    let sum: f64 = values.iter().sum();
    let estimate = if values.is_empty() { 0.0 } else { sum / n };
    let (hit_sum, hit_sum_sq) = values
        .iter()
        .filter(|&&v| v != 0.0)
        .fold((0.0_f64, 0.0_f64), |(s, q), &v| (s + v, q + v * v));
    let effective_sample_size = if hit_sum_sq > 0.0 {
        hit_sum * hit_sum / hit_sum_sq
    } else {
        0.0
    };
    let undefined = WeightedInterval {
        level,
        method: IntervalMethod::Undefined,
        replicas,
        estimate,
        standard_error: 0.0,
        low: estimate,
        high: estimate,
        effective_sample_size,
        relative_error: None,
    };
    if replicas < 2 || hit_sum_sq <= 0.0 {
        return undefined;
    }
    let squared_deviations: f64 = values.iter().map(|v| (v - estimate).powi(2)).sum();
    let std = (squared_deviations / (n - 1.0)).max(0.0).sqrt();
    let (low, high) = normal_bounds(estimate, std, replicas, level);
    let standard_error = std / n.sqrt();
    WeightedInterval {
        method: IntervalMethod::WeightedNormal,
        standard_error,
        low,
        high,
        relative_error: (estimate != 0.0).then(|| standard_error / estimate),
        ..undefined
    }
}

/// Whether a confidence level is usable: a probability strictly between
/// certainty and nothing.
#[must_use]
pub fn is_valid_level(level: f64) -> bool {
    level > 0.0 && level < 1.0
}

/// Two-sided normal deviate of a confidence level: `z = Φ⁻¹((1 + level)/2)`.
///
/// `level = 0.95` gives `1.959963984540054`. Outside `(0, 1)` the result
/// is not meaningful; callers validate the level with [`is_valid_level`]
/// before reaching here.
#[must_use]
pub fn z_of(level: f64) -> f64 {
    normal_quantile(0.5 * (1.0 + level))
}

/// Wilson score bounds of a binomial proportion `p` observed on `n`
/// draws, at confidence `level`.
///
/// Closed form, with `z = z_of(level)` and `p̃ = (p + z²/2n) / (1 + z²/n)`:
///
/// ```text
/// p̃ ± z/(1 + z²/n) · √( p(1−p)/n + z²/(4n²) )
/// ```
///
/// Both bounds lie in `[0, 1]` for every `p ∈ [0, 1]` and every `n ≥ 1`.
/// At `p = 0` the interval is `[0, z²/(n + z²)]`, at `p = 1` it is
/// `[n/(n + z²), 1]`; those two ends are returned from their closed form
/// directly, because there the general expression subtracts two equal
/// quantities and rounding would leave a bound like `4e-19` where the
/// mathematics says zero. They are also the two ends a study reads most
/// closely, an event no replica reached and one every replica did.
#[must_use]
pub fn wilson_bounds(p: f64, n: u64, level: f64) -> (f64, f64) {
    wilson_at(p, n, z_of(level))
}

/// [`wilson_bounds`] with the deviate already computed.
fn wilson_at(p: f64, n: u64, z: f64) -> (f64, f64) {
    let n = n as f64;
    let z2 = z * z;
    if p <= 0.0 {
        return (0.0, z2 / (n + z2));
    }
    if p >= 1.0 {
        return (n / (n + z2), 1.0);
    }
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let spread = (p * (1.0 - p) / n + z2 / (4.0 * n * n)).max(0.0).sqrt();
    let half = z / denom * spread;
    ((center - half).max(0.0), (center + half).min(1.0))
}

/// Central-limit bounds of a mean: `mean ± z·std/√n`, with `std` the
/// sample standard deviation (ddof = 1) over the `n` replicas.
///
/// Not clamped to any support: a bound outside it is the interval saying
/// the normal approximation does not hold at this sample size.
#[must_use]
pub fn normal_bounds(mean: f64, std: f64, n: u64, level: f64) -> (f64, f64) {
    normal_at(mean, std, n, z_of(level))
}

/// [`normal_bounds`] with the deviate already computed.
fn normal_at(mean: f64, std: f64, n: u64, z: f64) -> (f64, f64) {
    let half = z * std / (n as f64).sqrt();
    (mean - half, mean + half)
}

/// Upper bound, at confidence `level`, on the frequency of an outcome
/// that **none** of `n` draws showed: `z²/(n + z²)`.
///
/// The rule of three in exact form. `z² = 3.8415` at 95 %, so the bound
/// is the `3/n` of the textbooks: 0.00762 for 500 draws, 0.00383 for
/// 1000. It is the same expression [`wilson_bounds`] returns at
/// `p = 0`, and it is the whole of what a campaign that observed no
/// departure establishes.
#[must_use]
pub fn unobserved_frequency_bound(n: u64, level: f64) -> f64 {
    frequency_at(n, z_of(level))
}

/// [`unobserved_frequency_bound`] with the deviate already computed.
fn frequency_at(n: u64, z: f64) -> f64 {
    let z2 = z * z;
    z2 / (n as f64 + z2)
}

/// Bounds of a mean whose sample was **constant** over the `n`
/// replicas: `value ± ε·size`, with `ε` the frequency bound
/// [`unobserved_frequency_bound`] returns and `size` how far one
/// departing replica would have moved the estimate.
///
/// `floor` is the value the quantity cannot be pushed below where its
/// declaration gives one, and it is what makes the interval one-sided
/// on an event no replica reached: `[0, ε·size]` rather than a band
/// straddling a count or a duration that can only be positive. The
/// bounds are a closed form, not an asymptotic one, so a bound outside
/// the support here would signal nothing and mislead: this is the
/// opposite case to the unclamped normal bounds.
#[must_use]
pub fn constant_sample_bounds(
    value: f64,
    size: f64,
    floor: Option<f64>,
    n: u64,
    level: f64,
) -> (f64, f64) {
    constant_at(value, size, floor, unobserved_frequency_bound(n, level))
}

/// [`constant_sample_bounds`] with the frequency bound already computed.
fn constant_at(value: f64, size: f64, floor: Option<f64>, epsilon: f64) -> (f64, f64) {
    let half = epsilon * size;
    match floor {
        Some(floor) => ((value - half).max(floor), (value + half).max(floor)),
        None => (value - half, value + half),
    }
}

/// Horner evaluation of a polynomial in `r`, coefficients highest degree
/// first.
fn poly(r: f64, coefficients: &[f64]) -> f64 {
    coefficients.iter().fold(0.0, |acc, &c| acc * r + c)
}

/// Standard-normal quantile `Φ⁻¹(p)` for `p ∈ (0, 1)`.
///
/// Wichura's AS 241 (`PPND16`) rational approximation, accurate to about
/// 16 significant digits over the whole open interval: the quantile is
/// exact for every practical purpose, so the only approximation left in
/// an interval is the statistical one the interval declares.
///
/// Returns `±∞` at the closed ends and `NaN` outside `[0, 1]`.
// The coefficients are transcribed digit for digit from AS 241, several
// of them past what an `f64` distinguishes. Truncating them to the
// nearest representable value would round-trip identically and make the
// transcription impossible to check against the published table, which
// is the only way this block is ever reviewed.
#[allow(clippy::excessive_precision)]
#[must_use]
pub fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return if p == 0.0 {
            f64::NEG_INFINITY
        } else if p == 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        // Central region: rational in r = 0.180625 − q².
        let r = 0.180625 - q * q;
        let num = poly(
            r,
            &[
                2509.0809287301226727,
                33430.575583588128105,
                67265.770927008700853,
                45921.953931549871457,
                13731.693765509461125,
                1971.5909503065514427,
                133.14166789178437745,
                3.387132872796366608,
            ],
        );
        let den = poly(
            r,
            &[
                5226.495278852854561,
                28729.085735721942674,
                39307.89580009271061,
                21213.794301586595867,
                5394.1960214247511077,
                687.1870074920579083,
                42.313330701600911252,
                1.0,
            ],
        );
        return q * num / den;
    }
    // Tails: rational in r = √(−ln(min(p, 1−p))), two ranges.
    let tail = if q < 0.0 { p } else { 1.0 - p };
    let r = (-tail.ln()).sqrt();
    let value = if r <= 5.0 {
        let r = r - 1.6;
        let num = poly(
            r,
            &[
                7.7454501427834140764e-4,
                0.0227238449892691845833,
                0.24178072517745061177,
                1.27045825245236838258,
                3.64784832476320460504,
                5.7694972214606914055,
                4.6303378461565452959,
                1.42343711074968357734,
            ],
        );
        let den = poly(
            r,
            &[
                1.05075007164441684324e-9,
                5.475938084995344946e-4,
                0.0151986665636164571966,
                0.14810397642748007459,
                0.68976733498510000455,
                1.6763848301838038494,
                2.05319162663775882187,
                1.0,
            ],
        );
        num / den
    } else {
        let r = r - 5.0;
        let num = poly(
            r,
            &[
                2.01033439929228813265e-7,
                2.71155556874348757815e-5,
                0.0012426609473880784386,
                0.026532189526576123093,
                0.29656057182850489123,
                1.7848265399172913358,
                5.4637849111641143699,
                6.6579046435011037772,
            ],
        );
        let den = poly(
            r,
            &[
                2.04426310338993978564e-15,
                1.4215117583164458887e-7,
                1.8463183175100546818e-5,
                7.868691311456132591e-4,
                0.0148753612908506148525,
                0.13692988092273580531,
                0.59983220655588793769,
                1.0,
            ],
        );
        num / den
    };
    if q < 0.0 {
        -value
    } else {
        value
    }
}

/// Student interval over independent, equally weighted splitting batches.
/// Its coverage is asymptotic in the particle budget, not exact for skewed batches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchInterval {
    /// Declared two-sided confidence level.
    pub level: f64,
    /// Construction, always `BatchStudent` for at least two batches.
    pub method: IntervalMethod,
    /// Arithmetic mean of batch estimates.
    pub estimate: f64,
    /// Standard error from the ddof-1 sample variance.
    pub standard_error: f64,
    /// Unclamped lower bound.
    pub low: f64,
    /// Unclamped upper bound.
    pub high: f64,
}

/// Form an unclamped Student interval from independent batch estimates.
/// The caller validates finite values and a confidence level in `(0, 1)`.
#[must_use]
pub fn batch_interval(values: &[f64], level: f64) -> BatchInterval {
    let n = values.len() as f64;
    let estimate = if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / n
    };
    let standard_error = if values.len() < 2 {
        0.0
    } else {
        (values.iter().map(|x| (x - estimate).powi(2)).sum::<f64>() / (n * (n - 1.0))).sqrt()
    };
    let half = if values.len() < 2 {
        0.0
    } else {
        student_critical(level, values.len() - 1) * standard_error
    };
    BatchInterval {
        level,
        method: if values.len() < 2 {
            IntervalMethod::Undefined
        } else {
            IntervalMethod::BatchStudent
        },
        estimate,
        standard_error,
        low: estimate - half,
        high: estimate + half,
    }
}

// Lanczos log-gamma, positive arguments. Used only once per quantile setup.
fn log_gamma(z: f64) -> f64 {
    const C: [f64; 8] = [
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    let z = z - 1.0;
    let mut x = 0.9999999999998099;
    for (i, c) in C.iter().enumerate() {
        x += c / (z + i as f64 + 1.0);
    }
    let t = z + 7.5;
    0.9189385332046727 + (z + 0.5) * t.ln() - t + x.ln()
}

// Modified Lentz evaluation of the incomplete-beta continued fraction
// (NIST DLMF 8.17.22), reflecting x when needed for convergence.
fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=10_000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        for (term, aa) in [
            m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0)),
        ]
        .into_iter()
        .enumerate()
        {
            d = 1.0 + aa * d;
            if d.abs() < tiny {
                d = tiny;
            }
            c = 1.0 + aa / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if term == 1 && (delta - 1.0).abs() < 4.0 * f64::EPSILON {
                return h;
            }
        }
    }
    f64::NAN
}
fn beta_regularized(x: f64, a: f64, b: f64, log_norm: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let factor = (a * x.ln() + b * (-x).ln_1p() - log_norm).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        factor * beta_fraction(a, b, x) / a
    } else {
        1.0 - factor * beta_fraction(b, a, 1.0 - x) / b
    }
}
/// Positive Student critical value for a two-sided `level` and integer degrees
/// of freedom. Inverts `I_(nu/(nu+t²))(nu/2, 1/2) = 1-level` by bracketing.
/// Reference: NIST DLMF 8.17 and NIST Engineering Statistics Handbook 1.3.6.7.2.
#[must_use]
pub fn student_critical(level: f64, degrees: usize) -> f64 {
    if !is_valid_level(level) || degrees == 0 {
        return f64::NAN;
    }
    let nu = degrees as f64;
    let a = nu / 2.0;
    let log_norm = log_gamma(a) + log_gamma(0.5) - log_gamma(a + 0.5);
    let tail = |t: f64| beta_regularized(nu / (nu + t * t), a, 0.5, log_norm);
    let mut lo = 0.0;
    let mut hi = 1.0;
    loop {
        let probability = tail(hi);
        if !probability.is_finite() {
            return f64::NAN;
        }
        if probability <= 1.0 - level {
            break;
        }
        hi *= 2.0;
        if !hi.is_finite() {
            return f64::NAN;
        }
    }
    for _ in 0..100 {
        let mid = lo + (hi - lo) / 2.0;
        let probability = tail(mid);
        if !probability.is_finite() {
            return f64::NAN;
        }
        if probability > 1.0 - level {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo + (hi - lo) / 2.0
}
