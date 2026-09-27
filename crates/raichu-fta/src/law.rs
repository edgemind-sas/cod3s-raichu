//! The probability that a basic event has occurred by the mission time.
//!
//! A fault tree is quantified at one instant: each basic event stands for
//! a transition that has fired by then, with no repair, so its probability
//! is the cumulative distribution function of its law at the mission time.
//! The laws and their parameters are the engine's own
//! ([`raichu_core::BasicLaw`]), so a tree generated from a model is
//! quantified with the distributions the simulator draws from.
//!
//! Every elementary function comes from `libm`, not from the platform's
//! C library, for the same reason the engine keeps `std_math` off: the
//! same tree gives the same bits on every platform.

use serde::Serialize;

use crate::FtaError;

/// How a basic event occurs.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "law", rename_all = "snake_case")]
pub enum Law {
    /// A probability given directly (an on-demand event, or a value read
    /// from an OpenPSA `<float>`). Independent of the mission time.
    Constant {
        /// The probability.
        probability: f64,
    },
    /// Exponential time to occurrence: `1 - exp(-rate t)`.
    Exponential {
        /// Rate.
        rate: f64,
    },
    /// Weibull time to occurrence: `1 - exp(-(t / scale)^shape)`.
    Weibull {
        /// Shape k.
        shape: f64,
        /// Scale.
        scale: f64,
    },
    /// Log-normal time to occurrence: `Φ((ln t - mu) / sigma)`.
    Lognormal {
        /// Mean of the underlying normal.
        mu: f64,
        /// Deviation of the underlying normal.
        sigma: f64,
    },
    /// Gamma time to occurrence: the regularised lower incomplete gamma
    /// function `P(shape, t / scale)`.
    Gamma {
        /// Shape.
        shape: f64,
        /// Scale.
        scale: f64,
    },
    /// Uniform time to occurrence on `[low, high)`.
    Uniform {
        /// Lower bound.
        low: f64,
        /// Upper bound.
        high: f64,
    },
    /// Occurs exactly at `time`: 0 before, 1 from then on.
    Delay {
        /// The delay.
        time: f64,
    },
    /// An empirical table of `(time, cumulative probability)` pairs,
    /// interpolated linearly as the engine samples it: the first point
    /// carries its cumulative probability as a mass.
    Empirical {
        /// The table, times increasing, ending at cumulative 1.
        points: Vec<(f64, f64)>,
    },
}

impl Law {
    /// The law's name, as it appears in a result or an error.
    pub fn name(&self) -> &'static str {
        match self {
            Law::Constant { .. } => "constant",
            Law::Exponential { .. } => "exponential",
            Law::Weibull { .. } => "weibull",
            Law::Lognormal { .. } => "lognormal",
            Law::Gamma { .. } => "gamma",
            Law::Uniform { .. } => "uniform",
            Law::Delay { .. } => "delay",
            Law::Empirical { .. } => "empirical",
        }
    }

    /// Whether the probability depends on the mission time.
    pub fn is_timed(&self) -> bool {
        !matches!(self, Law::Constant { .. })
    }

    /// The probability of occurrence by `mission_time`, for the event
    /// named `event` (named in the error when it cannot be computed).
    pub fn probability(&self, event: &str, mission_time: Option<f64>) -> Result<f64, FtaError> {
        let bad = |detail: String| FtaError::BadLaw {
            event: event.to_owned(),
            detail,
        };
        let t = match (self, mission_time) {
            (Law::Constant { probability }, _) => {
                if !(0.0..=1.0).contains(probability) {
                    return Err(bad(format!("probability {probability} is outside [0, 1]")));
                }
                return Ok(*probability);
            }
            (_, None) => {
                return Err(FtaError::MissionTimeRequired {
                    event: event.to_owned(),
                    law: self.name(),
                })
            }
            (_, Some(t)) => t,
        };
        if !t.is_finite() || t < 0.0 {
            return Err(FtaError::BadMissionTime(t));
        }
        let p = match self {
            Law::Constant { .. } => unreachable_constant(),
            Law::Exponential { rate } => {
                if !(rate.is_finite() && *rate >= 0.0) {
                    return Err(bad(format!(
                        "rate {rate} is not a finite non-negative number"
                    )));
                }
                -libm::expm1(-rate * t)
            }
            Law::Weibull { shape, scale } => {
                if !(shape.is_finite() && *shape > 0.0 && scale.is_finite() && *scale > 0.0) {
                    return Err(bad(format!(
                        "shape {shape} and scale {scale} must be finite and positive"
                    )));
                }
                if t == 0.0 {
                    0.0
                } else {
                    -libm::expm1(-libm::pow(t / scale, *shape))
                }
            }
            Law::Lognormal { mu, sigma } => {
                if !(mu.is_finite() && sigma.is_finite() && *sigma > 0.0) {
                    return Err(bad(format!(
                        "mu {mu} must be finite and sigma {sigma} finite and positive"
                    )));
                }
                if t == 0.0 {
                    0.0
                } else {
                    0.5 * libm::erfc(-(libm::log(t) - mu) / (sigma * core::f64::consts::SQRT_2))
                }
            }
            Law::Gamma { shape, scale } => {
                if !(shape.is_finite() && *shape > 0.0 && scale.is_finite() && *scale > 0.0) {
                    return Err(bad(format!(
                        "shape {shape} and scale {scale} must be finite and positive"
                    )));
                }
                regularized_lower_gamma(*shape, t / scale)
            }
            Law::Uniform { low, high } => {
                if !(low.is_finite() && high.is_finite() && low < high) {
                    return Err(bad(format!("bounds [{low}, {high}) are not an interval")));
                }
                ((t - low) / (high - low)).clamp(0.0, 1.0)
            }
            Law::Delay { time } => {
                if !(time.is_finite() && *time >= 0.0) {
                    return Err(bad(format!(
                        "delay {time} is not a finite non-negative number"
                    )));
                }
                if t >= *time {
                    1.0
                } else {
                    0.0
                }
            }
            Law::Empirical { points } => empirical_cdf(points, t).map_err(bad)?,
        };
        Ok(p.clamp(0.0, 1.0))
    }
}

// The constant law returned early above; this arm only keeps the match
// exhaustive without a panic on the library path.
fn unreachable_constant() -> f64 {
    0.0
}

/// The cumulative distribution of an empirical table at `t`, the inverse
/// of the engine's sampler: 0 before the first time, the first point's
/// mass at it, linear between points, 1 from the last.
fn empirical_cdf(points: &[(f64, f64)], t: f64) -> Result<f64, String> {
    // The engine's own checks on the table (raichu-model), repeated here
    // because a table read from OpenPSA attributes never went through them.
    let Some(&(first_t, _)) = points.first() else {
        return Err("the empirical table is empty".to_owned());
    };
    for &(pt, pc) in points {
        if !pt.is_finite() || pt < 0.0 || !pc.is_finite() || !(0.0..=1.0).contains(&pc) {
            return Err(format!(
                "the empirical table has an invalid point ({pt}, {pc})"
            ));
        }
    }
    for window in points.windows(2) {
        if window[1].0 < window[0].0 || window[1].1 < window[0].1 {
            return Err("the empirical table is not increasing".to_owned());
        }
    }
    if let Some(&(_, last)) = points.last() {
        if last != 1.0 {
            return Err(format!(
                "the empirical table ends at cumulative probability {last}, not 1"
            ));
        }
    }
    if t < first_t {
        return Ok(0.0);
    }
    for window in points.windows(2) {
        let (t0, c0) = window[0];
        let (t1, c1) = window[1];
        if t < t1 {
            if t1 == t0 {
                return Ok(c1);
            }
            return Ok(c0 + (c1 - c0) * (t - t0) / (t1 - t0));
        }
    }
    Ok(1.0)
}

/// The regularised lower incomplete gamma function `P(a, x)`: its series
/// below `a + 1`, its continued fraction (modified Lentz) above, both to
/// machine precision. The classical split: each converges fast on its
/// side.
fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let log_prefix = a * libm::log(x) - x - libm::lgamma(a);
    if x < a + 1.0 {
        let mut term = 1.0 / a;
        let mut sum = term;
        let mut n = a;
        for _ in 0..10_000 {
            n += 1.0;
            term *= x / n;
            sum += term;
            if term.abs() < sum.abs() * f64::EPSILON {
                break;
            }
        }
        (sum * libm::exp(log_prefix)).min(1.0)
    } else {
        const TINY: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
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
            if (delta - 1.0).abs() < f64::EPSILON {
                break;
            }
        }
        (1.0 - libm::exp(log_prefix) * h).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn at(law: Law, t: f64) -> f64 {
        law.probability("e", Some(t)).unwrap()
    }

    #[test]
    fn exponential_is_its_closed_form() {
        let p = at(Law::Exponential { rate: 1e-3 }, 100.0);
        assert!((p - (1.0 - (-0.1f64).exp())).abs() < 1e-15);
    }

    #[test]
    fn gamma_of_shape_one_is_the_exponential() {
        for x in [0.1, 1.0, 1.9, 2.5, 10.0] {
            let p = at(
                Law::Gamma {
                    shape: 1.0,
                    scale: 2.0,
                },
                x,
            );
            assert!((p - (1.0 - (-x / 2.0f64).exp())).abs() < 1e-14, "{x}: {p}");
        }
    }

    #[test]
    fn gamma_of_integer_shape_is_the_erlang_sum() {
        // P(3, x) = 1 - e^-x (1 + x + x^2/2)
        for x in [0.5, 3.0, 4.5, 20.0] {
            let p = at(
                Law::Gamma {
                    shape: 3.0,
                    scale: 1.0,
                },
                x,
            );
            let exact = 1.0 - (-x).exp() * (1.0 + x + x * x / 2.0);
            assert!((p - exact).abs() < 1e-14, "{x}: {p} vs {exact}");
        }
    }

    #[test]
    fn lognormal_median_is_one_half() {
        let p = at(
            Law::Lognormal {
                mu: 1.5,
                sigma: 0.7,
            },
            1.5f64.exp(),
        );
        assert!((p - 0.5).abs() < 1e-15);
    }

    #[test]
    fn weibull_at_its_scale_is_one_minus_one_over_e() {
        let p = at(
            Law::Weibull {
                shape: 2.5,
                scale: 40.0,
            },
            40.0,
        );
        assert!((p - (1.0 - (-1.0f64).exp())).abs() < 1e-15);
    }

    #[test]
    fn empirical_inverts_the_sampler() {
        let law = Law::Empirical {
            points: vec![(1.0, 0.2), (3.0, 0.6), (5.0, 1.0)],
        };
        assert_eq!(at(law.clone(), 0.5), 0.0);
        assert_eq!(at(law.clone(), 1.0), 0.2);
        assert!((at(law.clone(), 2.0) - 0.4).abs() < 1e-15);
        assert_eq!(at(law, 7.0), 1.0);
    }

    #[test]
    fn uniform_and_delay() {
        assert_eq!(
            at(
                Law::Uniform {
                    low: 2.0,
                    high: 6.0
                },
                3.0
            ),
            0.25
        );
        assert_eq!(at(Law::Delay { time: 5.0 }, 4.999), 0.0);
        assert_eq!(at(Law::Delay { time: 5.0 }, 5.0), 1.0);
    }

    #[test]
    fn a_timed_law_without_a_mission_time_is_refused_by_name() {
        let err = Law::Exponential { rate: 1.0 }
            .probability("pump.fail", None)
            .unwrap_err();
        assert!(err.to_string().contains("pump.fail"));
    }
}
