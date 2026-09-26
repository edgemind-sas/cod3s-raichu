//! Absorption probability of an acyclic chain of exponential phases.
//!
//! Given sojourn rates `q1..qk` (per unit of time) and a horizon `t`, the
//! accumulator computes `P(S_k <= t)` where `S_k = T_1 + ... + T_k` and the
//! `T_i ~ Exp(q_i)` are independent: the probability that a path of the
//! embedded jump chain, visiting phases of total exit rates `q1..qk`, is
//! completed by the horizon. It is the time factor of a sequence
//! probability in the exact sequence-tree exploration.
//!
//! # Method
//!
//! **Uniformization, in nonnegative arithmetic.** With `L = max q_i` and
//! `x = L t`, the chain is observed at the jumps of a Poisson process of
//! rate `L`: at each jump the current phase `i` is left with probability
//! `p_i = q_i / L` and kept otherwise. Then
//!
//! ```text
//! P(S_k <= t) = sum_{n >= k} w_n a_n,   w_n = e^{-x} x^n / n!,
//! ```
//!
//! where `a_n` is the probability that the discrete chain has crossed all
//! `k` phases within `n` jumps. Every term is nonnegative, so the value is
//! the absorption probability itself, never one minus a survival: a
//! probability of 1e-25 keeps its relative precision. The Poisson weights
//! are evaluated in log space through a cancellation-free Stirling form, so
//! `x` in the tens of thousands does not underflow `e^{-x}`.
//!
//! **Truncation bound.** The sum is cut at `n = N`. Two rigorous bounds
//! on the discarded mass are computed and the smaller is kept:
//! `sum_{n>N} w_n` (because `a_n <= 1`), and
//! `prod(p_i) x^k / k! * e^{-x} sum_{m > N-k} x^m / m!` (because
//! `a_n <= C(n, k) prod(p_i)`, a union bound over the jump positions that
//! leave a phase). The second is what keeps the bound relative for tiny
//! probabilities. Poisson tails are bounded by the ratio test
//! `sum_{n>m} w_n <= w_{m+1} / (1 - x / (m + 2))` when `m + 2 > x`. `N` is
//! doubled until the truncation bound is below a quarter of the requested
//! relative precision.
//!
//! **Fallback: scaling and squaring.** When `N + 1` would exceed
//! [`PhaseTypeSettings::max_terms`] (a stiff path, `max(q) t` large), the
//! transient matrix `exp(Q t)` of the Metzler generator `Q` is computed as
//! `exp(Q h)^{2^s}` with `h = t / 2^s` and `L h <= 2`, where
//! `exp(Q h) = e^{-L h} sum_n (L h)^n P^n / n!` with the stochastic
//! matrix `P = I + Q / L`. Every operation is on nonnegative numbers, and
//! the value read is the absorption entry `[0, k]` itself.
//!
//! **Error bound.** Each value carries an error bound: truncation plus a
//! first-order rounding bound valid for sums of products of nonnegative
//! numbers (each such term carries at most one relative rounding per
//! operation along its computation path). The rounding bound treats the
//! stated rates and horizon as exact inputs, the rounding of `L t` aside,
//! which it includes. A value whose bound exceeds
//! [`PhaseTypeSettings::rel_precision`] times the value is **flagged**,
//! never silently returned as precise. Squaring doubles the relative error
//! at each of its `s` steps, so the fallback's bound grows like
//! `2^s ~ L t`: on a very stiff path at a tight precision the result is
//! flagged.
//!
//! # Incremental use and complexity
//!
//! The accumulator is built for a depth-first exploration that keeps one
//! accumulator per depth: clone the parent's, [`PhaseTypeAccumulator::push`]
//! the child's rate, read [`PhaseTypeAccumulator::absorption`]. The
//! horizon is fixed at construction, so the Poisson weights are computed
//! once per uniformization rate and shared between clones (`Arc`).
//!
//! - clone: `O(N)` (one vector of `N + 1` floats);
//! - push, rate not above the current `L`: `O(N)` (the new phase's inflow
//!   is computed from the previous one);
//! - push raising `L`, or needing more terms: `O(N k)` rebuild from the
//!   stored rates;
//! - push on the fallback: `O(k^3 log2(L t))`;
//! - read: `O(1)` (the value is maintained eagerly by every push).
//!
//! `N` is about `k + L t + 8 sqrt(L t) + 32` on well-conditioned paths.
//!
//! Instantaneous phases (zero sojourn) are no-ops, and a phase of rate 0 is
//! never left, so any path containing one has probability exactly 0.

use std::sync::Arc;

use thiserror::Error;

/// Unit roundoff of `f64`.
const U: f64 = f64::EPSILON / 2.0;

/// Safety factor applied to every computed bound (covers second-order
/// rounding terms the first-order analysis drops).
const MARGIN: f64 = 1.01;

/// Poisson terms beyond `L t` in the initial term-count guess.
const INITIAL_EXTRA_TERMS: f64 = 32.0;

/// Series terms beyond the chain length in the fallback's `exp(Q h)`.
const SQUARING_SERIES_EXTRA: usize = 30;

/// Largest `L h` the fallback's series is evaluated at.
const SQUARING_MAX_STEP: f64 = 2.0;

/// Numerical settings of the absorption computation.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseTypeSettings {
    /// Requested relative precision of every returned probability
    /// (dimensionless, in `(0, 1)`). A value whose error bound exceeds
    /// `rel_precision * value` is flagged. Default `1e-9`.
    pub rel_precision: f64,
    /// Cap on the number of uniformization terms `N + 1`. Beyond it the
    /// computation falls back to scaling and squaring. Default `100_000`.
    pub max_terms: usize,
}

impl Default for PhaseTypeSettings {
    fn default() -> Self {
        PhaseTypeSettings {
            rel_precision: 1e-9,
            max_terms: 100_000,
        }
    }
}

impl PhaseTypeSettings {
    fn validate(&self) -> Result<(), PhaseTypeError> {
        if self.rel_precision.is_nan() || self.rel_precision <= 0.0 || self.rel_precision >= 1.0 {
            return Err(PhaseTypeError::InvalidSettings {
                reason: format!(
                    "rel_precision must lie in (0, 1), got {}",
                    self.rel_precision
                ),
            });
        }
        if self.max_terms == 0 {
            return Err(PhaseTypeError::InvalidSettings {
                reason: "max_terms must be at least 1".to_string(),
            });
        }
        Ok(())
    }
}

/// Typed errors of the absorption computation.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PhaseTypeError {
    /// A phase rate is negative, infinite or NaN. An instantaneous phase
    /// is declared with [`PhaseTypeAccumulator::push_instantaneous`], not
    /// with an infinite rate.
    #[error("phase rate must be finite and nonnegative, got {rate}")]
    InvalidRate {
        /// The refused rate.
        rate: f64,
    },
    /// The horizon is negative, infinite or NaN.
    #[error("horizon must be finite and nonnegative, got {horizon}")]
    InvalidHorizon {
        /// The refused horizon.
        horizon: f64,
    },
    /// A setting is out of its domain.
    #[error("invalid phase-type settings: {reason}")]
    InvalidSettings {
        /// What is wrong.
        reason: String,
    },
}

/// How an [`Absorption`] value was obtained.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AbsorptionMethod {
    /// Known exactly: empty path (1), zero horizon (0), or a phase of
    /// rate 0 on the path (0).
    Exact,
    /// Uniformization truncated after `terms` Poisson terms.
    Uniformization {
        /// Number of Poisson terms summed (`N + 1`).
        terms: usize,
    },
    /// Scaling and squaring on the Metzler generator, after the term
    /// count exceeded its cap.
    ScalingAndSquaring {
        /// Number of squarings `s` (the horizon was split into `2^s`).
        squarings: u32,
    },
}

/// Probability that the path is completed by the horizon, with its
/// guaranteed error bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Absorption {
    /// `P(S_k <= t)`, dimensionless, in `[0, 1]`.
    pub value: f64,
    /// Bound on `|value - exact|` (absolute, same unit as `value`):
    /// truncation plus first-order rounding.
    pub error_bound: f64,
    /// `true` when `error_bound > rel_precision * value`: the value is
    /// not guaranteed to the requested precision.
    pub flagged: bool,
    /// The method that produced the value.
    pub method: AbsorptionMethod,
}

impl Absorption {
    fn exact(value: f64) -> Self {
        Absorption {
            value,
            error_bound: 0.0,
            flagged: false,
            method: AbsorptionMethod::Exact,
        }
    }

    fn bounded(value: f64, error_bound: f64, rel_precision: f64, method: AbsorptionMethod) -> Self {
        let flagged = error_bound.is_nan() || error_bound > rel_precision * value;
        Absorption {
            value,
            error_bound,
            flagged,
            method,
        }
    }
}

/// Poisson weights `w_n = e^{-x} x^n / n!` for `n = 0..=N`, with a
/// relative error bound for each.
#[derive(Debug)]
struct PoissonWeights {
    weight: Vec<f64>,
    rel_err: Vec<f64>,
}

impl PoissonWeights {
    fn new(x: f64, last: usize) -> Self {
        let mut weight = Vec::with_capacity(last + 1);
        let mut rel_err = Vec::with_capacity(last + 1);
        for n in 0..=last {
            let (lw, abs_err) = ln_poisson_pmf(n, x);
            weight.push(lw.exp());
            rel_err.push(abs_err + 2.0 * U);
        }
        PoissonWeights { weight, rel_err }
    }

    fn last(&self) -> usize {
        self.weight.len().saturating_sub(1)
    }
}

/// Uniformization state for the current path.
#[derive(Debug, Clone)]
struct UniformState {
    /// Uniformization rate `L` (per unit of time).
    lambda: f64,
    /// `L t`.
    x: f64,
    /// Shared weights for `n = 0..=N`.
    weights: Arc<PoissonWeights>,
    /// `d_n`: probability that the last phase's exit (the absorption)
    /// happens exactly at jump `n`, for `n = 0..=N`.
    inflow: Vec<f64>,
    /// `sum ln(q_i / L)` over the timed phases.
    ln_prod_p: f64,
}

/// Incremental absorption probability of an acyclic phase chain at a
/// fixed horizon (see the module documentation for the method, the error
/// bound and the complexity).
///
/// Cheap to clone, so an exploration can keep one per depth.
#[derive(Debug, Clone)]
pub struct PhaseTypeAccumulator {
    horizon: f64,
    settings: PhaseTypeSettings,
    rates: Vec<f64>,
    blocked: bool,
    uniform: Option<UniformState>,
    result: Absorption,
}

impl PhaseTypeAccumulator {
    /// Empty path (probability 1) at `horizon` (time units of the rates).
    ///
    /// # Errors
    /// [`PhaseTypeError::InvalidHorizon`] for a negative, infinite or NaN
    /// horizon; [`PhaseTypeError::InvalidSettings`] for settings out of
    /// their domain.
    pub fn new(horizon: f64, settings: PhaseTypeSettings) -> Result<Self, PhaseTypeError> {
        if !horizon.is_finite() || horizon < 0.0 {
            return Err(PhaseTypeError::InvalidHorizon { horizon });
        }
        settings.validate()?;
        Ok(PhaseTypeAccumulator {
            horizon,
            settings,
            rates: Vec::new(),
            blocked: false,
            uniform: None,
            result: Absorption::exact(1.0),
        })
    }

    /// Path of the given rates, computed from scratch in one pass.
    ///
    /// # Errors
    /// As [`PhaseTypeAccumulator::new`], plus
    /// [`PhaseTypeError::InvalidRate`] for a negative, infinite or NaN
    /// rate.
    pub fn from_rates(
        horizon: f64,
        rates: &[f64],
        settings: PhaseTypeSettings,
    ) -> Result<Self, PhaseTypeError> {
        let mut acc = Self::new(horizon, settings)?;
        for &rate in rates {
            validate_rate(rate)?;
        }
        acc.rates = rates.to_vec();
        acc.blocked = rates.contains(&0.0);
        acc.recompute(0);
        Ok(acc)
    }

    /// Append a timed phase of exit rate `rate` (per unit of time) at the
    /// end of the path. A rate of 0 makes the path's probability exactly 0.
    ///
    /// # Errors
    /// [`PhaseTypeError::InvalidRate`] for a negative, infinite or NaN
    /// rate; the accumulator is then left unchanged.
    pub fn push(&mut self, rate: f64) -> Result<(), PhaseTypeError> {
        validate_rate(rate)?;
        self.rates.push(rate);
        if rate == 0.0 {
            self.blocked = true;
        }
        if self.blocked || self.horizon == 0.0 {
            self.uniform = None;
            self.result = Absorption::exact(0.0);
            return Ok(());
        }
        let fast = match self.uniform.as_mut() {
            Some(state) if rate <= state.lambda => {
                state.inflow = next_inflow(&state.inflow, rate, state.lambda);
                state.ln_prod_p += (rate / state.lambda).ln();
                true
            }
            _ => false,
        };
        if fast {
            if let Some(state) = self.uniform.as_ref() {
                let (value, trunc, round) = evaluate(state, self.rates.len());
                if trunc <= 0.25 * self.settings.rel_precision * value || value == 0.0 {
                    self.result = Absorption::bounded(
                        value,
                        trunc + round,
                        self.settings.rel_precision,
                        AbsorptionMethod::Uniformization {
                            terms: state.weights.last() + 1,
                        },
                    );
                    return Ok(());
                }
                let doubled = 2 * state.weights.last();
                self.recompute(doubled);
                return Ok(());
            }
        }
        self.recompute(0);
        Ok(())
    }

    /// Append a phase of zero sojourn (an instantaneous branch): a no-op
    /// on the probability.
    pub fn push_instantaneous(&mut self) {}

    /// Probability that the path is completed by the horizon, with its
    /// error bound. `O(1)`: maintained by every push.
    #[must_use]
    pub fn absorption(&self) -> Absorption {
        self.result
    }

    /// The horizon the accumulator was built for.
    #[must_use]
    pub fn horizon(&self) -> f64 {
        self.horizon
    }

    /// Rates of the timed phases pushed so far, in path order.
    #[must_use]
    pub fn rates(&self) -> &[f64] {
        &self.rates
    }

    /// Number of timed phases (instantaneous phases are not counted).
    #[must_use]
    pub fn phase_count(&self) -> usize {
        self.rates.len()
    }

    /// The settings the accumulator was built with.
    #[must_use]
    pub fn settings(&self) -> &PhaseTypeSettings {
        &self.settings
    }

    /// Recompute from the stored rates, with at least `min_last + 1`
    /// uniformization terms.
    fn recompute(&mut self, min_last: usize) {
        let k = self.rates.len();
        if k == 0 {
            self.uniform = None;
            self.result = Absorption::exact(1.0);
            return;
        }
        if self.blocked || self.horizon == 0.0 {
            self.uniform = None;
            self.result = Absorption::exact(0.0);
            return;
        }
        let lambda = self.rates.iter().copied().fold(0.0, f64::max);
        let x = lambda * self.horizon;
        let guess = k as f64 + x + 8.0 * x.sqrt() + INITIAL_EXTRA_TERMS;
        let mut last = if guess >= self.settings.max_terms as f64 {
            self.settings.max_terms
        } else {
            (guess.ceil() as usize).max(min_last)
        };
        let ln_prod_p: f64 = self.rates.iter().map(|&q| (q / lambda).ln()).sum();
        while last < self.settings.max_terms {
            let weights = Arc::new(PoissonWeights::new(x, last));
            let mut inflow = vec![0.0; last + 1];
            inflow[0] = 1.0;
            for &q in &self.rates {
                inflow = next_inflow(&inflow, q, lambda);
            }
            let state = UniformState {
                lambda,
                x,
                weights,
                inflow,
                ln_prod_p,
            };
            let (value, trunc, round) = evaluate(&state, k);
            if trunc <= 0.25 * self.settings.rel_precision * value || value == 0.0 {
                self.result = Absorption::bounded(
                    value,
                    trunc + round,
                    self.settings.rel_precision,
                    AbsorptionMethod::Uniformization { terms: last + 1 },
                );
                self.uniform = Some(state);
                return;
            }
            last = last.saturating_mul(2);
        }
        self.uniform = None;
        let (value, error_bound, squarings) = scaling_and_squaring(&self.rates, self.horizon);
        self.result = Absorption::bounded(
            value,
            error_bound,
            self.settings.rel_precision,
            AbsorptionMethod::ScalingAndSquaring { squarings },
        );
    }
}

fn validate_rate(rate: f64) -> Result<(), PhaseTypeError> {
    if rate.is_finite() && rate >= 0.0 {
        Ok(())
    } else {
        Err(PhaseTypeError::InvalidRate { rate })
    }
}

/// Inflow into the absorbing state after appending a phase of rate `q`:
/// the previous absorption inflow `d` now enters the new phase, which is
/// left with probability `p = q / L` at each jump.
///
/// `s_n = s_{n-1} (1 - p) + d_n`, `d'_n = p s_{n-1}`, `d'_0 = 0`. All
/// operations are on nonnegative numbers.
fn next_inflow(inflow: &[f64], q: f64, lambda: f64) -> Vec<f64> {
    let p = q / lambda;
    let stay = (lambda - q) / lambda;
    let mut out = vec![0.0; inflow.len()];
    let mut sojourn = 0.0;
    for (n, &d) in inflow.iter().enumerate() {
        if n > 0 {
            out[n] = p * sojourn;
        }
        sojourn = sojourn * stay + d;
    }
    out
}

/// Value, truncation bound and rounding bound of the uniformized sum for
/// a path of `k` timed phases.
fn evaluate(state: &UniformState, k: usize) -> (f64, f64, f64) {
    let last = state.weights.last();
    let mut absorbed = 0.0;
    let mut value = 0.0;
    let mut weight_err = 0.0;
    for n in 0..=last {
        absorbed += state.inflow[n];
        let term = state.weights.weight[n] * absorbed;
        value += term;
        weight_err += term * state.weights.rel_err[n];
    }
    let x = state.x;
    // Bound 1: a_n <= 1.
    let tail_all = ln_poisson_tail(x, last).exp();
    // Bound 2: a_n <= C(n, k) prod(p).
    let tail_path = if last >= k {
        (state.ln_prod_p + k as f64 * x.ln() - ln_factorial(k) + ln_poisson_tail(x, last - k)).exp()
    } else {
        f64::INFINITY
    };
    let trunc = MARGIN * tail_all.min(tail_path);
    let round = MARGIN * (value * (6.0 * (last + k) as f64 + 10.0) * U + weight_err);
    (value, trunc, round)
}

/// Bound on `ln sum_{n > m} w_n` (Poisson upper tail), by the ratio test
/// when `m + 2 > x`, else the trivial bound `ln 1`.
fn ln_poisson_tail(x: f64, m: usize) -> f64 {
    let next = (m + 2) as f64;
    if next > x {
        let (lw, abs_err) = ln_poisson_pmf(m + 1, x);
        lw - (1.0 - x / next).ln() + abs_err + 4.0 * U
    } else {
        0.0
    }
}

/// `ln n!`, exact product up to 20 (representable exactly), Stirling
/// series beyond.
fn ln_factorial(n: usize) -> f64 {
    if n <= 20 {
        let mut prod = 1.0;
        for i in 2..=n {
            prod *= i as f64;
        }
        prod.ln()
    } else {
        let nf = n as f64;
        nf * nf.ln() - nf + stirling_correction(nf)
    }
}

/// `ln n! - (n ln n - n)` for `n > 20`, Stirling series to `n^{-9}`
/// (truncation below 1e-17 there).
fn stirling_correction(n: f64) -> f64 {
    let inv = 1.0 / n;
    let inv2 = inv * inv;
    0.5 * (2.0 * std::f64::consts::PI * n).ln()
        + inv
            * (1.0 / 12.0
                - inv2
                    * (1.0 / 360.0 - inv2 * (1.0 / 1260.0 - inv2 * (1.0 / 1680.0 - inv2 / 1188.0))))
}

/// `ln w_n = -x + n ln x - ln n!` for `x > 0`, with an absolute error
/// bound on the logarithm (hence a relative bound on `w_n`).
///
/// Written as `n ln(x/n) + (n - x) - c(n)` with `c(n) = ln n! - n ln n + n`:
/// the three terms are small near the mode, where the textbook form
/// cancels terms of size `n ln n`. The bound includes the effect of the
/// rounding of `x` itself (`|n - x| u`).
fn ln_poisson_pmf(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (-x, 2.0 * U * x);
    }
    let nf = n as f64;
    let ratio = x / nf;
    let (l1, l1_err) = if (0.5..=2.0).contains(&ratio) {
        let r = (x - nf) / nf;
        let l1 = nf * r.ln_1p();
        (l1, 4.0 * U * (l1.abs() + (x - nf).abs()))
    } else {
        let lx = x.ln();
        let ln = nf.ln();
        let l1 = nf * (lx - ln);
        (l1, 4.0 * U * (l1.abs() + nf * (lx.abs() + ln.abs())))
    };
    let (corr, corr_err) = if n <= 20 {
        let lf = ln_factorial(n);
        let c = lf - nf * nf.ln() + nf;
        (c, 4.0 * U * (lf.abs() + nf * nf.ln().abs() + nf))
    } else {
        let c = stirling_correction(nf);
        (c, 4.0 * U * (c.abs() + 1.0))
    };
    let diff = nf - x;
    let lw = l1 + diff - corr;
    let err = l1_err + corr_err + 3.0 * U * diff.abs() + 2.0 * U * lw.abs();
    (lw, err)
}

/// Scaling and squaring on the Metzler generator of the chain: returns
/// the absorption entry of `exp(Q t)`, its error bound and the number of
/// squarings.
fn scaling_and_squaring(rates: &[f64], horizon: f64) -> (f64, f64, u32) {
    let k = rates.len();
    let dim = k + 1;
    let lambda = rates.iter().copied().fold(0.0, f64::max);
    let x = lambda * horizon;
    let mut squarings: u32 = 0;
    let mut y = x;
    while y > SQUARING_MAX_STEP {
        y *= 0.5;
        squarings += 1;
    }
    // Stochastic matrix P = I + Q / L: upper bidiagonal.
    let mut stay = vec![1.0; dim];
    let mut advance = vec![0.0; dim];
    for (i, &q) in rates.iter().enumerate() {
        stay[i] = (lambda - q) / lambda;
        advance[i] = q / lambda;
    }
    // Horner: B = I + (y/1) P (I + (y/2) P (... (I + (y/M) P))).
    let terms = k + SQUARING_SERIES_EXTRA;
    let idx = |i: usize, j: usize| i * dim + j;
    let mut b = vec![0.0; dim * dim];
    for i in 0..dim {
        b[idx(i, i)] = 1.0;
    }
    for level in (1..=terms).rev() {
        let c = y / level as f64;
        let mut next = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in i..dim {
                let mut pb = stay[i] * b[idx(i, j)];
                if i + 1 < dim {
                    pb += advance[i] * b[idx(i + 1, j)];
                }
                next[idx(i, j)] = c * pb + if i == j { 1.0 } else { 0.0 };
            }
        }
        b = next;
    }
    let scale = (-y).exp();
    for v in &mut b {
        *v *= scale;
    }
    // Relative error of every entry of exp(Q h): series truncation
    // (sum_{m > 30} y^m / m!, entrywise relative by the union bound) plus
    // rounding along the Horner levels.
    let extra = SQUARING_SERIES_EXTRA as i32 + 1;
    let (ln_head, _) = ln_poisson_pmf(extra as usize, y);
    let trunc_rel = (ln_head + y).exp() / (1.0 - y / (f64::from(extra) + 1.0));
    let mut rel = trunc_rel + (8.0 * (terms + 1) as f64 + 4.0) * U;
    for _ in 0..squarings {
        let mut next = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in i..dim {
                let mut acc = 0.0;
                for l in i..=j {
                    acc += b[idx(i, l)] * b[idx(l, j)];
                }
                next[idx(i, j)] = acc;
            }
        }
        b = next;
        rel = 2.0 * rel + (dim as f64 + 1.0) * U;
    }
    let value = b[idx(0, k)];
    (value, MARGIN * rel * value, squarings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_poisson_pmf_matches_the_direct_form_where_it_is_accurate() {
        for &(n, x) in &[(0usize, 0.3), (3, 2.5), (10, 10.0), (25, 30.0), (60, 1.0)] {
            let direct = -x + n as f64 * f64::ln(x) - ln_factorial(n);
            let (lw, err) = ln_poisson_pmf(n, x);
            assert!((lw - direct).abs() <= err + 1e-13, "n={n} x={x}");
        }
    }

    #[test]
    fn ln_factorial_is_continuous_across_the_stirling_switch() {
        let exact21 = ln_factorial(20) + 21f64.ln();
        assert!((ln_factorial(21) - exact21).abs() < 1e-13);
    }
}
