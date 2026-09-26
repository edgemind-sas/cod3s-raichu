//! Survival, cumulative hazard and quantile functions of the delay laws
//! carried by the engine.
//!
//! A stochastic transition armed at date `t0` fires at `t0 + T`, where the
//! random delay `T >= 0` follows one of the laws below. For a law with
//! cumulative distribution function `F(t) = P(T <= t)` this module
//! provides:
//!
//! - the survival `S(t) = P(T > t) = 1 - F(t)`;
//! - the cumulative hazard `H(t) = -ln S(t)` (dimensionless), computed
//!   without the `ln(1 - F)` cancellation for small `t`: closed forms
//!   where they exist (`H = rate t`, `H = (t / scale)^shape`), otherwise
//!   `-ln1p(-F)` while `F < 1/2` and the logarithm of a directly computed
//!   upper tail beyond, so that `H` keeps its relative precision both for
//!   `H` far below 1e-12 and for `S` far below the smallest double;
//! - the quantile `F^-1(u) = inf { t : F(t) >= u }` for `u` in `(0, 1]`,
//!   the lower end of the support at `u = 0`; the inverse survival
//!   `S^-1(p) = F^-1(1 - p)`, computed from `p` without forming `1 - p`;
//!   and the inverse cumulative hazard `H^-1(h) = F^-1(1 - e^-h)`;
//! - the same quantities conditional on an age `a` already accrued
//!   without firing: `S(a + s) / S(a)`, `H(a + s) - H(a)` and the
//!   conditional quantile.
//!
//! These are what the discretised sequence-tree exploration needs to cut
//! the random next event into cells of known probability mass.
//!
//! # Conventions
//!
//! - Times are in simulation time units, `t < 0` behaves as `t = 0`
//!   before the support (`S = 1`, `H = 0`), and a NaN time returns NaN.
//! - **Delay** `d`: `T = d` surely. `H(t) = 0` for `t < d` and
//!   `H(t) = +inf` for `t >= d`, so `S` is right-continuous and the
//!   transition fires with certainty at its date.
//! - **Empirical**: the table of `(time, cumulative probability)` points
//!   read by the engine's inverse-CDF sampler (`sample_empirical` in
//!   `raichu-core`). [`Law::quantile`] reproduces that sampler exactly,
//!   and [`Law::cdf`] is its right-continuous inverse: `0` before the
//!   first time, an atom of mass `c0` at the first time, linear
//!   interpolation between points, and a jump wherever two points share a
//!   time.
//!
//! # Precision
//!
//! Exponential, Weibull, uniform, delay and empirical laws are evaluated
//! in closed form to a few units in the last place. Lognormal and gamma
//! rest on the in-crate special functions below:
//!
//! - [`regularized_gamma_p`] / [`regularized_gamma_q`]: series for
//!   `x < a + 1` and a modified-Lentz continued fraction otherwise
//!   (Numerical Recipes, `gser` / `gcf`), with the prefactor
//!   `x^a e^-x / Gamma(a)` evaluated through the Stirling correction and
//!   `ln(1 + mu) - mu` so that it does not cancel when `x` is close to a
//!   large `a`. The directly computed tail has a relative error of about
//!   `1e-14 (1 + x eps / 1e-14)`, the other one is `1` minus it.
//! - [`erf`] / [`erfc`] are `P(1/2, x^2)` / `Q(1/2, x^2)`; the rounding of
//!   `x^2` adds a relative error of about `x^2 eps` in the far tail
//!   (`1e-13` at `x = 26`, where `erfc` is 5.7e-296).
//! - [`standard_normal_quantile`]: Acklam's rational approximation
//!   (relative error 1.15e-9) refined by Newton steps on `ln Phi`, which is
//!   concave, so the refinement converges from either side; the result is
//!   accurate to about 1e-15 relative down to `p = 1e-300`.
//! - The gamma quantile is solved by Newton iteration on `ln P` (or
//!   `ln Q`) against `ln x`: the log-gamma density is log-concave, so both
//!   log tails are concave in `ln x` and the iteration converges
//!   monotonically after at most one overshoot.
//!
//! The tests in `tests/laws.rs` check every function against values
//! computed with mpmath at 50 digits, to 1e-12 relative or tighter.
//! Conditional quantities formed as a difference `H(a + s) - H(a)` (gamma,
//! lognormal, uniform, empirical) carry an absolute error of about
//! `eps (1 + H(a))`; the exponential and Weibull ones are closed forms
//! that stay relative for `s << a`.

use thiserror::Error;

/// `ln(sqrt(2 pi))`.
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

/// Iteration cap of the series and continued fraction of the incomplete
/// gamma function. Both converge in `O(sqrt(a))` iterations near
/// `x = a` and much faster elsewhere; the cap is only reached for shapes
/// beyond 1e10, where a NaN is returned rather than a wrong number.
const MAX_GAMMA_ITERATIONS: usize = 10_000_000;

/// Error raised by a law constructor or by an out-of-domain argument.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LawError {
    /// A law parameter is outside its domain.
    #[error("{law} law: parameter `{parameter}` = {value} is invalid ({requirement})")]
    InvalidParameter {
        /// The law being built.
        law: &'static str,
        /// The offending parameter.
        parameter: &'static str,
        /// Its value.
        value: f64,
        /// The domain it must lie in.
        requirement: &'static str,
    },
    /// An empirical table is malformed.
    #[error("empirical law: {detail}")]
    InvalidEmpiricalTable {
        /// What is wrong with the table.
        detail: String,
    },
    /// A probability argument is outside `[0, 1]` or NaN.
    #[error("probability {value} is outside [0, 1]")]
    ProbabilityOutOfRange {
        /// The offending probability.
        value: f64,
    },
    /// A cumulative hazard argument is negative or NaN.
    #[error("cumulative hazard {value} is not a number >= 0")]
    InvalidHazard {
        /// The offending hazard.
        value: f64,
    },
    /// An age is negative, infinite or NaN.
    #[error("age {age} is not a finite number >= 0")]
    InvalidAge {
        /// The offending age.
        age: f64,
    },
    /// A duration after the age is negative or NaN.
    #[error("duration {duration} is not a number >= 0")]
    InvalidDuration {
        /// The offending duration.
        duration: f64,
    },
    /// The survival at the age is zero: the transition has surely fired,
    /// so no conditional law exists.
    #[error("age {age} is beyond the support of the law (survival is 0 there)")]
    AgeBeyondSupport {
        /// The offending age.
        age: f64,
    },
}

/// A validated delay law: the distribution of the time `T >= 0` between
/// the arming of a stochastic transition and its firing.
///
/// Built through the constructors, which check the same domains as the
/// model schema (except the exponential rate, which may be 0 here so that
/// a state-dependent rate at rest maps onto a law that never fires).
#[derive(Debug, Clone, PartialEq)]
pub struct Law {
    kind: Kind,
}

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    Delay(f64),
    Exponential(f64),
    Weibull { shape: f64, scale: f64 },
    Lognormal { mu: f64, sigma: f64 },
    Gamma { shape: f64, scale: f64 },
    Uniform { low: f64, high: f64 },
    Empirical(Vec<(f64, f64)>),
}

fn invalid(
    law: &'static str,
    parameter: &'static str,
    value: f64,
    requirement: &'static str,
) -> LawError {
    LawError::InvalidParameter {
        law,
        parameter,
        value,
        requirement,
    }
}

fn check_positive(law: &'static str, parameter: &'static str, value: f64) -> Result<(), LawError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid(law, parameter, value, "finite and > 0"))
    }
}

fn check_probability(value: f64) -> Result<(), LawError> {
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(LawError::ProbabilityOutOfRange { value })
    }
}

impl Law {
    /// Deterministic delay `d` (time units, finite, `>= 0`): `T = d`.
    pub fn delay(d: f64) -> Result<Self, LawError> {
        if !d.is_finite() || d < 0.0 {
            return Err(invalid("delay", "time", d, "finite and >= 0"));
        }
        Ok(Self {
            kind: Kind::Delay(d),
        })
    }

    /// Exponential law of rate `rate` (events per time unit, finite,
    /// `>= 0`): `H(t) = rate t`. A zero rate never fires.
    pub fn exponential(rate: f64) -> Result<Self, LawError> {
        if !rate.is_finite() || rate < 0.0 {
            return Err(invalid("exponential", "rate", rate, "finite and >= 0"));
        }
        Ok(Self {
            kind: Kind::Exponential(rate),
        })
    }

    /// Weibull law of shape `k > 0` and scale `lambda > 0` (time units):
    /// `H(t) = (t / lambda)^k`.
    pub fn weibull(shape: f64, scale: f64) -> Result<Self, LawError> {
        check_positive("weibull", "shape", shape)?;
        check_positive("weibull", "scale", scale)?;
        Ok(Self {
            kind: Kind::Weibull { shape, scale },
        })
    }

    /// Lognormal law: `ln T ~ N(mu, sigma^2)`, `mu` finite, `sigma > 0`.
    pub fn lognormal(mu: f64, sigma: f64) -> Result<Self, LawError> {
        if !mu.is_finite() {
            return Err(invalid("lognormal", "mu", mu, "finite"));
        }
        check_positive("lognormal", "sigma", sigma)?;
        Ok(Self {
            kind: Kind::Lognormal { mu, sigma },
        })
    }

    /// Gamma law of shape `k > 0` and scale `theta > 0` (time units):
    /// `F(t) = P(k, t / theta)`, mean `k theta`.
    pub fn gamma(shape: f64, scale: f64) -> Result<Self, LawError> {
        check_positive("gamma", "shape", shape)?;
        check_positive("gamma", "scale", scale)?;
        Ok(Self {
            kind: Kind::Gamma { shape, scale },
        })
    }

    /// Uniform law on `[low, high)`, `0 <= low < high`, both finite.
    pub fn uniform(low: f64, high: f64) -> Result<Self, LawError> {
        if !low.is_finite() || low < 0.0 {
            return Err(invalid("uniform", "low", low, "finite and >= 0"));
        }
        if !high.is_finite() || high <= low {
            return Err(invalid("uniform", "high", high, "finite and > low"));
        }
        Ok(Self {
            kind: Kind::Uniform { low, high },
        })
    }

    /// Empirical law given by a table of `(time, cumulative probability)`
    /// points, validated like the model schema: non-empty, times finite
    /// and `>= 0`, probabilities in `[0, 1]`, both non-decreasing, and the
    /// last probability equal to 1. See the module documentation for how
    /// the table is read.
    pub fn empirical(points: Vec<(f64, f64)>) -> Result<Self, LawError> {
        let table_error = |detail: String| LawError::InvalidEmpiricalTable { detail };
        if points.is_empty() {
            return Err(table_error("empty table".to_owned()));
        }
        let mut prev: Option<(f64, f64)> = None;
        for &(t, c) in &points {
            if !t.is_finite() || t < 0.0 || !(0.0..=1.0).contains(&c) {
                return Err(table_error(format!("invalid point ({t}, {c})")));
            }
            if let Some((pt, pc)) = prev {
                if t < pt || c < pc {
                    return Err(table_error(format!(
                        "non-monotone point ({t}, {c}) after ({pt}, {pc})"
                    )));
                }
            }
            prev = Some((t, c));
        }
        if let Some((_, last)) = prev {
            if last != 1.0 {
                return Err(table_error(format!(
                    "last cumulative probability is {last}, expected 1"
                )));
            }
        }
        Ok(Self {
            kind: Kind::Empirical(points),
        })
    }

    /// Name of the law family (`"delay"`, `"exponential"`, `"weibull"`,
    /// `"lognormal"`, `"gamma"`, `"uniform"` or `"empirical"`).
    pub fn name(&self) -> &'static str {
        match self.kind {
            Kind::Delay(_) => "delay",
            Kind::Exponential(_) => "exponential",
            Kind::Weibull { .. } => "weibull",
            Kind::Lognormal { .. } => "lognormal",
            Kind::Gamma { .. } => "gamma",
            Kind::Uniform { .. } => "uniform",
            Kind::Empirical(_) => "empirical",
        }
    }

    /// `F`, `S` and `H` at `t`, each computed on its precise side.
    fn evaluate(&self, t: f64) -> Tail {
        if t.is_nan() {
            return Tail {
                cdf: f64::NAN,
                sf: f64::NAN,
                hazard: f64::NAN,
            };
        }
        match &self.kind {
            Kind::Delay(d) => {
                if t < *d {
                    Tail::BEFORE
                } else {
                    Tail::AFTER
                }
            }
            Kind::Exponential(rate) => {
                if t <= 0.0 || *rate == 0.0 {
                    Tail::BEFORE
                } else {
                    Tail::from_hazard(rate * t)
                }
            }
            Kind::Weibull { shape, scale } => {
                if t <= 0.0 {
                    Tail::BEFORE
                } else {
                    Tail::from_hazard((t / scale).powf(*shape))
                }
            }
            Kind::Uniform { low, high } => {
                if t <= *low {
                    Tail::BEFORE
                } else if t >= *high {
                    Tail::AFTER
                } else {
                    let width = high - low;
                    Tail::from_cdf_sf((t - low) / width, (high - t) / width)
                }
            }
            Kind::Lognormal { mu, sigma } => {
                if t <= 0.0 {
                    Tail::BEFORE
                } else {
                    let z = (t.ln() - mu) / sigma;
                    let cdf = standard_normal_cdf(z);
                    let sf = standard_normal_cdf(-z);
                    let hazard = if z < 0.0 {
                        -(-cdf).ln_1p()
                    } else {
                        -ln_standard_normal_cdf(-z)
                    };
                    Tail { cdf, sf, hazard }
                }
            }
            Kind::Gamma { shape, scale } => {
                if t <= 0.0 {
                    Tail::BEFORE
                } else {
                    let g = incomplete_gamma(*shape, t / scale);
                    Tail {
                        cdf: g.p,
                        sf: g.q,
                        hazard: -g.ln_q,
                    }
                }
            }
            Kind::Empirical(points) => {
                let cdf = empirical_cdf(points, t);
                Tail::from_cdf_sf(cdf, 1.0 - cdf)
            }
        }
    }

    /// Cumulative distribution function `F(t) = P(T <= t)`.
    pub fn cdf(&self, t: f64) -> f64 {
        self.evaluate(t).cdf
    }

    /// Survival function `S(t) = P(T > t)`.
    pub fn survival(&self, t: f64) -> f64 {
        self.evaluate(t).sf
    }

    /// Cumulative hazard `H(t) = -ln S(t)` (dimensionless, `+inf` once the
    /// transition has surely fired).
    pub fn cumulative_hazard(&self, t: f64) -> f64 {
        self.evaluate(t).hazard
    }

    /// Lower end of the support, `inf { t : F(t) > 0 }`.
    fn lower_support(&self) -> f64 {
        match &self.kind {
            Kind::Delay(d) => *d,
            Kind::Uniform { low, .. } => *low,
            Kind::Empirical(points) => sample_empirical(points, 0.0),
            _ => 0.0,
        }
    }

    /// Upper end of the support, `inf { t : F(t) >= 1 }`.
    fn upper_support(&self) -> f64 {
        match &self.kind {
            Kind::Delay(d) => *d,
            Kind::Uniform { high, .. } => *high,
            Kind::Empirical(points) => sample_empirical(points, 1.0),
            _ => f64::INFINITY,
        }
    }

    /// Quantile `F^-1(u) = inf { t : F(t) >= u }` for `u` in `(0, 1]`, and
    /// the lower end of the support at `u = 0`. For an empirical law this
    /// is exactly the engine's inverse-CDF sampler.
    pub fn quantile(&self, u: f64) -> Result<f64, LawError> {
        check_probability(u)?;
        if let Kind::Empirical(points) = &self.kind {
            return Ok(sample_empirical(points, u));
        }
        if u == 0.0 {
            return Ok(self.lower_support());
        }
        if u == 1.0 {
            return Ok(self.upper_support());
        }
        Ok(match &self.kind {
            Kind::Delay(d) => *d,
            Kind::Exponential(rate) => -(-u).ln_1p() / rate,
            Kind::Weibull { shape, scale } => scale * (-(-u).ln_1p()).powf(1.0 / shape),
            Kind::Uniform { low, high } => low + u * (high - low),
            Kind::Lognormal { mu, sigma } => (mu + sigma * standard_normal_quantile(u)).exp(),
            Kind::Gamma { shape, scale } => {
                if u <= 0.5 {
                    scale * gamma_inverse(*shape, u, Side::Lower)
                } else {
                    scale * gamma_inverse(*shape, 1.0 - u, Side::Upper)
                }
            }
            Kind::Empirical(points) => sample_empirical(points, u),
        })
    }

    /// Inverse survival `S^-1(p) = F^-1(1 - p)`, computed from `p`
    /// directly so that upper-tail quantiles keep their precision for `p`
    /// far below machine epsilon.
    pub fn inverse_survival(&self, p: f64) -> Result<f64, LawError> {
        check_probability(p)?;
        if p == 1.0 {
            return Ok(self.lower_support());
        }
        if p == 0.0 {
            return Ok(self.upper_support());
        }
        Ok(match &self.kind {
            Kind::Delay(d) => *d,
            Kind::Exponential(rate) => -p.ln() / rate,
            Kind::Weibull { shape, scale } => scale * (-p.ln()).powf(1.0 / shape),
            Kind::Uniform { low, high } => high - p * (high - low),
            Kind::Lognormal { mu, sigma } => (mu - sigma * standard_normal_quantile(p)).exp(),
            Kind::Gamma { shape, scale } => {
                if p <= 0.5 {
                    scale * gamma_inverse(*shape, p, Side::Upper)
                } else {
                    scale * gamma_inverse(*shape, 1.0 - p, Side::Lower)
                }
            }
            Kind::Empirical(points) => sample_empirical(points, 1.0 - p),
        })
    }

    /// Inverse cumulative hazard `H^-1(h) = F^-1(1 - e^-h)` for `h >= 0`
    /// (`+inf` allowed): the date at which the cumulative hazard reaches
    /// `h`. Closed form for the exponential and Weibull laws.
    pub fn inverse_cumulative_hazard(&self, h: f64) -> Result<f64, LawError> {
        if h.is_nan() || h < 0.0 {
            return Err(LawError::InvalidHazard { value: h });
        }
        if h == 0.0 {
            return Ok(self.lower_support());
        }
        match &self.kind {
            Kind::Exponential(rate) => Ok(h / rate),
            Kind::Weibull { shape, scale } => Ok(scale * h.powf(1.0 / shape)),
            _ => {
                if h < std::f64::consts::LN_2 {
                    self.quantile(-(-h).exp_m1())
                } else {
                    self.inverse_survival((-h).exp())
                }
            }
        }
    }

    fn check_age(&self, age: f64) -> Result<f64, LawError> {
        if !age.is_finite() || age < 0.0 {
            return Err(LawError::InvalidAge { age });
        }
        let hazard = self.cumulative_hazard(age);
        if hazard == f64::INFINITY {
            return Err(LawError::AgeBeyondSupport { age });
        }
        Ok(hazard)
    }

    /// Conditional cumulative hazard `H(a + s) - H(a)` over the window of
    /// length `s >= 0` following an age `a` accrued without firing.
    ///
    /// Errors when `a` is invalid or beyond the support (`S(a) = 0`).
    pub fn conditional_cumulative_hazard(&self, age: f64, s: f64) -> Result<f64, LawError> {
        let hazard_at_age = self.check_age(age)?;
        if s.is_nan() || s < 0.0 {
            return Err(LawError::InvalidDuration { duration: s });
        }
        Ok(match &self.kind {
            Kind::Exponential(rate) => {
                if *rate == 0.0 {
                    0.0
                } else {
                    rate * s
                }
            }
            Kind::Weibull { shape, scale } if age > 0.0 => {
                // (a/l)^k [(1 + s/a)^k - 1], without cancellation for s << a.
                hazard_at_age * (shape * (s / age).ln_1p()).exp_m1()
            }
            _ => {
                let later = self.cumulative_hazard(age + s);
                if later == f64::INFINITY {
                    f64::INFINITY
                } else {
                    (later - hazard_at_age).max(0.0)
                }
            }
        })
    }

    /// Conditional survival `S(a + s) / S(a)`: the probability of not
    /// firing within `s` more time units given an age `a` without firing.
    pub fn conditional_survival(&self, age: f64, s: f64) -> Result<f64, LawError> {
        Ok((-self.conditional_cumulative_hazard(age, s)?).exp())
    }

    /// Conditional distribution function `1 - S(a + s) / S(a)`, computed
    /// with `expm1` so that it keeps its relative precision when small.
    pub fn conditional_cdf(&self, age: f64, s: f64) -> Result<f64, LawError> {
        Ok(-(-self.conditional_cumulative_hazard(age, s)?).exp_m1())
    }

    /// Conditional quantile: the remaining time `s >= 0` such that the
    /// conditional distribution function reaches `u` given age `a`, that
    /// is `inf { s >= 0 : F(a + s) >= F(a) + u S(a) }` for `u` in `(0, 1]`,
    /// and the start of the conditional support at `u = 0`.
    pub fn conditional_quantile(&self, age: f64, u: f64) -> Result<f64, LawError> {
        let hazard_at_age = self.check_age(age)?;
        check_probability(u)?;
        if u == 0.0 {
            return Ok(self.conditional_support_start(age));
        }
        if let Kind::Exponential(rate) = &self.kind {
            return Ok(-(-u).ln_1p() / rate);
        }
        let target = hazard_at_age - (-u).ln_1p();
        let t = self.inverse_cumulative_hazard(target)?;
        Ok((t - age).max(0.0))
    }

    /// Start of the conditional support given age `a` (with `S(a) > 0`):
    /// `inf { t >= a : F(t) > F(a) } - a`.
    fn conditional_support_start(&self, age: f64) -> f64 {
        match &self.kind {
            Kind::Delay(d) => d - age,
            Kind::Uniform { low, .. } => (low - age).max(0.0),
            Kind::Empirical(points) => {
                let at_age = empirical_cdf(points, age);
                match points.iter().position(|&(_, c)| c > at_age) {
                    Some(0) => points[0].0 - age,
                    Some(j) => points[j - 1].0.max(age) - age,
                    None => 0.0,
                }
            }
            _ => 0.0,
        }
    }
}

/// `F`, `S` and `H` at one date.
#[derive(Debug, Clone, Copy)]
struct Tail {
    cdf: f64,
    sf: f64,
    hazard: f64,
}

impl Tail {
    const BEFORE: Tail = Tail {
        cdf: 0.0,
        sf: 1.0,
        hazard: 0.0,
    };
    const AFTER: Tail = Tail {
        cdf: 1.0,
        sf: 0.0,
        hazard: f64::INFINITY,
    };

    fn from_hazard(hazard: f64) -> Self {
        Self {
            cdf: -(-hazard).exp_m1(),
            sf: (-hazard).exp(),
            hazard,
        }
    }

    /// Both tails given, each precise; `H` from the smaller one.
    fn from_cdf_sf(cdf: f64, sf: f64) -> Self {
        let hazard = if sf == 0.0 {
            f64::INFINITY
        } else if cdf < 0.5 {
            -(-cdf).ln_1p()
        } else {
            -sf.ln()
        };
        Self { cdf, sf, hazard }
    }
}

// --- empirical table -------------------------------------------------------

/// The engine's inverse-CDF sampler (`sample_empirical` in
/// `raichu-core`), reproduced exactly: the generalised inverse of
/// [`empirical_cdf`].
fn sample_empirical(points: &[(f64, f64)], u: f64) -> f64 {
    let Some(&(first_t, first_c)) = points.first() else {
        return f64::NAN;
    };
    if u <= first_c {
        return first_t;
    }
    for window in points.windows(2) {
        let (t0, c0) = window[0];
        let (t1, c1) = window[1];
        if u <= c1 {
            if c1 == c0 {
                return t1;
            }
            return t0 + (t1 - t0) * (u - c0) / (c1 - c0);
        }
    }
    points.last().map_or(f64::NAN, |&(t, _)| t)
}

/// Right-continuous CDF of the empirical table: 0 before the first time,
/// the cumulative probability of the LAST point sharing a time at that
/// time, linear interpolation strictly between distinct times.
fn empirical_cdf(points: &[(f64, f64)], t: f64) -> f64 {
    // Number of points whose time is <= t.
    let count = points.partition_point(|&(time, _)| time <= t);
    if count == 0 {
        return 0.0;
    }
    let j = count - 1;
    let (t0, c0) = points[j];
    match points.get(j + 1) {
        None => c0,
        // t1 > t >= t0 by construction of `count`.
        Some(&(t1, c1)) => c0 + (c1 - c0) * (t - t0) / (t1 - t0),
    }
}

// --- special functions -----------------------------------------------------

/// Both regularised incomplete gamma tails and their logarithms.
#[derive(Debug, Clone, Copy)]
struct IncGamma {
    p: f64,
    q: f64,
    ln_p: f64,
    ln_q: f64,
}

/// `ln(1 + mu) - mu`, without cancellation for small `mu`.
fn ln_1p_minus(mu: f64) -> f64 {
    if mu.abs() < 0.5 {
        // ln(1 + mu) = 2 atanh(r), r = mu / (2 + mu); 2r - mu = -mu r.
        let r = mu / (2.0 + mu);
        let r2 = r * r;
        let mut term = r2;
        let mut sum = 0.0_f64;
        let mut k = 3.0;
        while term > f64::EPSILON * 1e-3 * sum.max(f64::MIN_POSITIVE) {
            sum += term / k;
            term *= r2;
            k += 2.0;
        }
        -mu * r + 2.0 * r * sum
    } else {
        mu.ln_1p() - mu
    }
}

/// Stirling series remainder `ln Gamma(z) - [(z - 1/2) ln z - z + ln sqrt(2 pi)]`
/// for `z >= 15`, truncated after the `z^-9` term (next term below
/// 2e-16 at `z = 15`).
fn stirling_remainder_large(z: f64) -> f64 {
    let z2 = z * z;
    (1.0 / 12.0
        - (1.0 / 360.0 - (1.0 / 1260.0 - (1.0 / 1680.0 - 1.0 / (1188.0 * z2)) / z2) / z2) / z2)
        / z
}

/// Stirling correction `delta(a) = ln Gamma(a + 1) - (a + 1/2) ln a + a - ln sqrt(2 pi)`
/// for any `a > 0`, by upward recurrence to the asymptotic series.
fn stirling_correction(a: f64) -> f64 {
    if a >= 15.0 {
        return stirling_remainder_large(a);
    }
    // ln Gamma(a + 1) = ln Gamma(z) - ln((a + 1) ... (z - 1)), z >= 15.
    let mut z = a + 1.0;
    let mut product = 1.0;
    while z < 15.0 {
        product *= z;
        z += 1.0;
    }
    let ln_gamma_z = (z - 0.5) * z.ln() - z + LN_SQRT_2PI + stirling_remainder_large(z);
    let ln_gamma_a1 = ln_gamma_z - product.ln();
    ln_gamma_a1 - (a + 0.5) * a.ln() + a - LN_SQRT_2PI
}

/// `ln(x^a e^-x / Gamma(a))`, evaluated as
/// `ln sqrt(a / 2 pi) - a phi(x / a) - delta(a)` with
/// `phi(l) = l - 1 - ln l`, so that it does not cancel near `x = a`.
fn ln_gamma_prefactor(a: f64, x: f64) -> f64 {
    let mu = (x - a) / a;
    let a_phi = if mu.abs() < 0.5 {
        -a * ln_1p_minus(mu)
    } else {
        x - a - a * (x.ln() - a.ln())
    };
    0.5 * (a / (2.0 * std::f64::consts::PI)).ln() - a_phi - stirling_correction(a)
}

fn incomplete_gamma(a: f64, x: f64) -> IncGamma {
    if x.is_nan() || a.is_nan() {
        return IncGamma {
            p: f64::NAN,
            q: f64::NAN,
            ln_p: f64::NAN,
            ln_q: f64::NAN,
        };
    }
    if x <= 0.0 {
        return IncGamma {
            p: 0.0,
            q: 1.0,
            ln_p: f64::NEG_INFINITY,
            ln_q: 0.0,
        };
    }
    if x == f64::INFINITY {
        return IncGamma {
            p: 1.0,
            q: 0.0,
            ln_p: 0.0,
            ln_q: f64::NEG_INFINITY,
        };
    }
    let ln_d = ln_gamma_prefactor(a, x);
    if x < a + 1.0 {
        // Series: P = D (1/a) (1 + x/(a+1) + x^2/((a+1)(a+2)) + ...).
        let mut ap = a;
        let mut term = 1.0 / a;
        let mut sum = term;
        let mut converged = false;
        for _ in 0..MAX_GAMMA_ITERATIONS {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * f64::EPSILON {
                converged = true;
                break;
            }
        }
        if !converged {
            return incomplete_gamma(f64::NAN, x);
        }
        let ln_p = sum.ln() + ln_d;
        let p = ln_p.exp();
        let q = -ln_p.exp_m1();
        IncGamma {
            p,
            q,
            ln_p,
            ln_q: (-p).ln_1p(),
        }
    } else {
        // Continued fraction for Q, modified Lentz.
        const TINY: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        let mut converged = false;
        for i in 1..MAX_GAMMA_ITERATIONS {
            let i = i as f64;
            let an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() <= f64::EPSILON {
                converged = true;
                break;
            }
        }
        if !converged {
            return incomplete_gamma(f64::NAN, x);
        }
        let ln_q = h.ln() + ln_d;
        let q = ln_q.exp();
        let p = -ln_q.exp_m1();
        IncGamma {
            p,
            q,
            ln_p: (-q).ln_1p(),
            ln_q,
        }
    }
}

/// Regularised lower incomplete gamma function
/// `P(a, x) = (1 / Gamma(a)) int_0^x s^(a-1) e^-s ds` for `a > 0`,
/// `x >= 0` (`x < 0` returns 0). Precision: see the module documentation.
pub fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    incomplete_gamma(a, x).p
}

/// Regularised upper incomplete gamma function `Q(a, x) = 1 - P(a, x)`,
/// computed directly (not as `1 - P`) for `x >= a + 1`, so that it keeps
/// its relative precision down to the smallest normal double.
pub fn regularized_gamma_q(a: f64, x: f64) -> f64 {
    incomplete_gamma(a, x).q
}

/// Error function `erf(x) = (2 / sqrt(pi)) int_0^x e^(-s^2) ds`,
/// computed as `sign(x) P(1/2, x^2)`.
pub fn erf(x: f64) -> f64 {
    if x < 0.0 {
        -incomplete_gamma(0.5, x * x).p
    } else {
        incomplete_gamma(0.5, x * x).p
    }
}

/// Complementary error function `erfc(x) = 1 - erf(x)`, computed as
/// `Q(1/2, x^2)` for `x >= 0` (relative precision kept in the tail) and
/// `1 + P(1/2, x^2)` for `x < 0`.
pub fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        1.0 + incomplete_gamma(0.5, x * x).p
    } else {
        incomplete_gamma(0.5, x * x).q
    }
}

/// `ln erfc(x)`, finite far beyond the underflow of `erfc`.
fn ln_erfc(x: f64) -> f64 {
    if x < 0.0 {
        incomplete_gamma(0.5, x * x).p.ln_1p()
    } else {
        incomplete_gamma(0.5, x * x).ln_q
    }
}

/// Standard normal distribution function `Phi(z) = erfc(-z / sqrt 2) / 2`.
pub fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// `ln Phi(z)`, finite far into the lower tail.
fn ln_standard_normal_cdf(z: f64) -> f64 {
    -std::f64::consts::LN_2 + ln_erfc(-z / std::f64::consts::SQRT_2)
}

/// Acklam's rational approximation of `Phi^-1(p)` for `p` in `(0, 1/2]`
/// (relative error below 1.15e-9).
fn acklam_lower(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    if p < 0.02425 {
        let r = (-2.0 * p.ln()).sqrt();
        (((((C[0] * r + C[1]) * r + C[2]) * r + C[3]) * r + C[4]) * r + C[5])
            / ((((D[0] * r + D[1]) * r + D[2]) * r + D[3]) * r + 1.0)
    } else {
        let s = p - 0.5;
        let r = s * s;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * s
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

/// `Phi^-1(p)` for `p` in `(0, 1/2]`: Acklam's approximation refined by
/// Newton steps on the concave function `ln Phi(z) - ln p`.
fn normal_quantile_lower(p: f64) -> f64 {
    let ln_p = p.ln();
    let mut z = acklam_lower(p);
    for _ in 0..50 {
        let ln_cdf = ln_standard_normal_cdf(z);
        let ln_density = -0.5 * z * z - LN_SQRT_2PI;
        let step = (ln_cdf - ln_p) * (ln_cdf - ln_density).exp();
        if !step.is_finite() {
            break;
        }
        z -= step;
        if step.abs() <= 2.0 * f64::EPSILON * z.abs() {
            break;
        }
    }
    z
}

/// Standard normal quantile `Phi^-1(p)`: `-inf` at 0, `+inf` at 1, NaN
/// outside `[0, 1]`. The upper half uses the exact symmetry
/// `Phi^-1(p) = -Phi^-1(1 - p)` (`1 - p` is exact for `p >= 1/2`).
pub fn standard_normal_quantile(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p <= 0.5 {
        normal_quantile_lower(p)
    } else {
        -normal_quantile_lower(1.0 - p)
    }
}

/// Which gamma tail a target probability refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    /// Solve `P(a, x) = target`.
    Lower,
    /// Solve `Q(a, x) = target`.
    Upper,
}

/// Solve `P(a, x) = target` or `Q(a, x) = target` for `x`, with `target`
/// in `(0, 1)`, by Newton iteration on `y = ln x`. The log tails of the
/// log-gamma variable are concave in `y`, so the iteration converges
/// monotonically after at most one overshoot.
fn gamma_inverse(a: f64, target: f64, side: Side) -> f64 {
    let ln_target = target.ln();
    // Wilson-Hilferty starting point, or the mean when it is not positive.
    let z = match side {
        Side::Lower => standard_normal_quantile(target),
        Side::Upper => -standard_normal_quantile(target),
    };
    let c = 1.0 / (9.0 * a);
    let wh = a * (1.0 - c + z * c.sqrt()).powi(3);
    let mut y = if wh.is_finite() && wh > 0.0 {
        wh.ln()
    } else {
        a.ln()
    };
    for _ in 0..500 {
        let x = y.exp();
        let g = incomplete_gamma(a, x);
        let ln_d = ln_gamma_prefactor(a, x);
        let (value, slope) = match side {
            Side::Lower => (g.ln_p - ln_target, (ln_d - g.ln_p).exp()),
            Side::Upper => (g.ln_q - ln_target, -(ln_d - g.ln_q).exp()),
        };
        let step = value / slope;
        if !step.is_finite() {
            break;
        }
        // A clamped step keeps the monotone convergence (concavity) while
        // preventing a jump to an underflowing x from a flat region.
        let step = step.clamp(-10.0, 10.0);
        y -= step;
        if step.abs() <= 2.0 * f64::EPSILON * y.abs().max(1.0) {
            break;
        }
    }
    y.exp()
}
