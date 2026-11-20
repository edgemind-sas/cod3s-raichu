//! Time signals evaluated natively by the expression language: a periodic
//! piecewise-linear shape and a smoothed Gaussian noise.
//!
//! Both are pure functions of their abscissa (the simulation time, in a
//! time profile), evaluated without scheduling any event: a drawn shape
//! simulated over a year costs one binary search per evaluation, not one
//! event per breakpoint and per period.
//!
//! Both are **continuous**. The piecewise-linear shape closes its period
//! with a segment back to its first value, and the noise interpolates its
//! nodes with a smoothstep, so neither ever jumps.

/// Value of a periodic piecewise-linear shape at `x`.
///
/// `points` are `[instant, value]` pairs over one period, the first at
/// instant 0, instants strictly increasing and below `period` (see
/// [`check_periodic_piecewise_linear`]). The shape is evaluated at
/// `u = (x - phase_shift) mod period`, in `[0, period)`, by binary search
/// and linear interpolation; the last segment joins the last point to
/// `(period, first value)`, which closes the period continuously.
#[must_use]
pub fn periodic_piecewise_linear(
    points: &[[f64; 2]],
    period: f64,
    phase_shift: f64,
    x: f64,
) -> f64 {
    let Some(first) = points.first() else {
        return 0.0;
    };
    let u = (x - phase_shift).rem_euclid(period);
    // `rem_euclid` can round up to `period` itself for a tiny negative
    // operand: that is the start of the next period.
    let u = if u >= period { 0.0 } else { u };
    // Index of the last point at or before `u` (the first point is at 0).
    let index = points.partition_point(|point| point[0] <= u).max(1) - 1;
    let [x0, y0] = points[index];
    let [x1, y1] = points.get(index + 1).copied().unwrap_or([period, first[1]]);
    let span = x1 - x0;
    if span <= 0.0 {
        return y0;
    }
    y0 + (y1 - y0) * (u - x0) / span
}

/// Refuse a periodic piecewise-linear shape that is not one: the reason is
/// returned as a sentence naming the offending value.
///
/// # Errors
/// A non-finite or non-positive `period`, a non-finite `phase_shift`, no
/// point, a non-finite instant or value, a first instant other than 0, or
/// instants not strictly increasing or not below `period`.
pub fn check_periodic_piecewise_linear(
    points: &[[f64; 2]],
    period: f64,
    phase_shift: f64,
) -> Result<(), String> {
    if !(period.is_finite() && period > 0.0) {
        return Err(format!("period must be finite and positive, got {period}"));
    }
    if !phase_shift.is_finite() {
        return Err(format!("phase_shift must be finite, got {phase_shift}"));
    }
    let Some(first) = points.first() else {
        return Err("points must hold at least one [instant, value] pair".to_owned());
    };
    if first[0] != 0.0 {
        return Err(format!("the first instant must be 0, got {}", first[0]));
    }
    for (index, [instant, value]) in points.iter().enumerate() {
        if !instant.is_finite() || !value.is_finite() {
            return Err(format!(
                "point {index} must be finite, got [{instant}, {value}]"
            ));
        }
        if *instant >= period {
            return Err(format!(
                "point {index} is at instant {instant}, not below the period {period}"
            ));
        }
        if index > 0 && *instant <= points[index - 1][0] {
            return Err(format!(
                "instants must be strictly increasing: point {index} is at {instant}, \
                 point {} at {}",
                index - 1,
                points[index - 1][0]
            ));
        }
    }
    Ok(())
}

/// Value of a smoothed Gaussian noise at `x`.
///
/// Nodes sit at `t_k = k * time_scale` and carry `amplitude * z_k`, with
/// `z_k` the standard normal [`smooth_noise_node`] of `(seed, key, k)`;
/// between two nodes the value follows the smoothstep
/// `s(u) = 3u² - 2u³`, `u` the position inside the interval:
/// `amplitude * (z_k + (z_{k+1} - z_k) * s(u))`. The noise is therefore
/// continuous with a continuous derivative, null at the nodes. The first
/// node is fixed at `z_0 = 0`, so the noise is 0 at `x = 0` whatever the
/// seed; a negative `x` reads node 0.
///
/// Evaluation is O(1) and draws nothing from any generator: within one
/// `seed` the noise is a pure function of `x`.
#[must_use]
pub fn smooth_noise(seed: u64, key: u64, amplitude: f64, time_scale: f64, x: f64) -> f64 {
    let position = (x / time_scale).max(0.0);
    let floor = position.floor();
    // A saturating cast: past 2⁶⁴ intervals the node index stops moving,
    // far beyond any simulated horizon.
    let k = floor as u64;
    let u = position - floor;
    let z0 = smooth_noise_node(seed, key, k);
    let z1 = smooth_noise_node(seed, key, k.saturating_add(1));
    let s = u * u * (3.0 - 2.0 * u);
    amplitude * (z0 + (z1 - z0) * s)
}

/// The standard normal `z_k` carried by node `k` of the noise of flow
/// `key` under `seed`; `z_0 = 0`.
///
/// A counter-based generator: `(seed, key, k)` is hashed by the
/// splitmix64 finalizer into two uniform numbers, turned into one normal
/// by the Box-Muller transform. Nothing is drawn from a stream, so node
/// `k` is the same number however many times, and in whatever order, it
/// is read.
#[must_use]
pub fn smooth_noise_node(seed: u64, key: u64, k: u64) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let base = mix64(mix64(mix64(seed ^ GOLDEN) ^ key) ^ k);
    // Two independent 53-bit uniforms: `u1` in (0, 1] (its logarithm is
    // finite), `u2` in [0, 1).
    let u1 = ((mix64(base ^ LANE_RADIUS) >> 11) + 1) as f64 * UNIT;
    let u2 = (mix64(base ^ LANE_ANGLE) >> 11) as f64 * UNIT;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Refuse a smoothed noise that is not one.
///
/// # Errors
/// A non-finite or negative `amplitude`, or a non-finite or non-positive
/// `time_scale`.
pub fn check_smooth_noise(amplitude: f64, time_scale: f64) -> Result<(), String> {
    if !(amplitude.is_finite() && amplitude >= 0.0) {
        return Err(format!(
            "amplitude must be finite and non-negative, got {amplitude}"
        ));
    }
    if !(time_scale.is_finite() && time_scale > 0.0) {
        return Err(format!(
            "time_scale must be finite and positive, got {time_scale}"
        ));
    }
    Ok(())
}

/// `2⁻⁵³`: a 53-bit integer times this is a float in `[0, 1)`.
const UNIT: f64 = 1.0 / (1u64 << 53) as f64;
/// Odd constant separating the seed from the other hashed words.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
/// Lane of the Box-Muller radius uniform.
const LANE_RADIUS: u64 = 0x3C6E_F372_FE94_F82B;
/// Lane of the Box-Muller angle uniform.
const LANE_ANGLE: u64 = 0xA54F_F53A_5F1D_36F1;

/// The splitmix64 finalizer: a bijective avalanche of 64 bits.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const RAMP: f64 = 5.0 / 60.0;

    /// A square wave with ramps: low 2, high 8 for 8 time units, ramps of
    /// 5 minutes, a 24-hour period shifted by 8.
    fn square() -> (Vec<[f64; 2]>, f64, f64) {
        let (low, high, high_duration) = (2.0, 8.0, 8.0);
        let points = vec![
            [0.0, low],
            [RAMP, high],
            [RAMP + high_duration, high],
            [2.0 * RAMP + high_duration, low],
        ];
        (points, 24.0, 8.0)
    }

    #[test]
    fn a_square_wave_ramps_holds_and_falls_back() {
        let (points, period, phase) = square();
        assert!(check_periodic_piecewise_linear(&points, period, phase).is_ok());
        let at = |x: f64| periodic_piecewise_linear(&points, period, phase, x);
        assert!((at(8.0 + 2.5 / 60.0) - 5.0).abs() < 1e-12);
        assert!((at(8.0 + RAMP + 4.0) - 8.0).abs() < 1e-12);
        assert!((at(2.0) - 2.0).abs() < 1e-12);
        assert!((at(20.0) - 2.0).abs() < 1e-12);
        assert!((at(8.0) - 2.0).abs() < 1e-12);
        // One period later, and before the start: the same values.
        assert!((at(32.0 + 2.5 / 60.0) - 5.0).abs() < 1e-12);
        assert!((at(-16.0 + 2.5 / 60.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn a_drawn_shape_is_continuous_across_its_period_and_periodic() {
        let points = [[0.0, 1.0], [6.0, 3.0], [15.0, 0.5]];
        let (period, phase) = (24.0, 0.0);
        let at = |x: f64| periodic_piecewise_linear(&points, period, phase, x);
        assert_eq!(at(0.0), 1.0);
        assert!((at(3.0) - 2.0).abs() < 1e-12);
        assert!((at(6.0) - 3.0).abs() < 1e-12);
        // The closing segment runs from (15, 0.5) back to (24, 1).
        assert!((at(19.5) - 0.75).abs() < 1e-12);
        assert!((at(24.0 - 1e-9) - 1.0).abs() < 1e-9);
        assert!((at(24.0 + 1e-9) - 1.0).abs() < 1e-9);
        for x in [0.3, 5.9, 11.0, 17.25, 23.9] {
            assert!((at(x) - at(x + 24.0)).abs() < 1e-12);
            assert!((at(x) - at(x + 240.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn a_single_point_is_a_constant() {
        let points = [[0.0, 4.0]];
        for x in [0.0, 3.0, 23.999, 1e6] {
            assert_eq!(periodic_piecewise_linear(&points, 24.0, 0.0, x), 4.0);
        }
    }

    #[test]
    fn an_ill_formed_shape_is_refused_with_a_reason() {
        let ok = [[0.0, 1.0], [1.0, 2.0]];
        let message = |points: &[[f64; 2]], period: f64| {
            check_periodic_piecewise_linear(points, period, 0.0).unwrap_err()
        };
        assert!(message(&ok, 0.0).contains("period"));
        assert!(message(&ok, f64::INFINITY).contains("period"));
        assert!(message(&[], 2.0).contains("at least one"));
        assert!(message(&[[0.5, 1.0]], 2.0).contains("first instant must be 0"));
        assert!(message(&[[0.0, 1.0], [1.0, 2.0], [1.0, 3.0]], 2.0).contains("strictly increasing"));
        assert!(message(&[[0.0, 1.0], [2.0, 2.0]], 2.0).contains("not below the period"));
        assert!(message(&[[0.0, f64::NAN]], 2.0).contains("finite"));
        assert!(check_periodic_piecewise_linear(&ok, 2.0, f64::NAN)
            .unwrap_err()
            .contains("phase_shift"));
    }

    #[test]
    fn the_noise_starts_at_zero_and_is_reproducible() {
        assert_eq!(smooth_noise_node(1, 2, 0), 0.0);
        for seed in [0, 1, u64::MAX] {
            assert_eq!(smooth_noise(seed, 7, 3.0, 0.5, 0.0), 0.0);
            assert_eq!(smooth_noise(seed, 7, 3.0, 0.5, -4.0), 0.0);
        }
        for x in [0.1, 1.7, 1234.5] {
            assert_eq!(
                smooth_noise(42, 7, 3.0, 0.5, x),
                smooth_noise(42, 7, 3.0, 0.5, x)
            );
        }
    }

    #[test]
    fn the_noise_differs_between_seeds_and_between_keys() {
        let at = |seed, key| smooth_noise(seed, key, 1.0, 1.0, 10.5);
        assert_ne!(at(1, 7), at(2, 7));
        assert_ne!(at(1, 7), at(1, 8));
    }

    #[test]
    fn the_noise_is_continuous_and_passes_through_its_nodes() {
        let (amplitude, scale) = (2.0, 0.25);
        for k in 1..50u64 {
            let node = k as f64 * scale;
            let value = smooth_noise(9, 3, amplitude, scale, node);
            assert!((value - amplitude * smooth_noise_node(9, 3, k)).abs() < 1e-12);
            for x in [node, node + 0.3 * scale] {
                let step = smooth_noise(9, 3, amplitude, scale, x + 1e-9)
                    - smooth_noise(9, 3, amplitude, scale, x);
                assert!(step.abs() < 1e-7, "jump {step} at {x}");
            }
        }
    }

    #[test]
    fn the_nodes_are_standard_normal() {
        let n = 1000u64;
        let nodes: Vec<f64> = (1..=n)
            .map(|k| smooth_noise(2024, 11, 1.5, 1.0, k as f64))
            .collect();
        let mean = nodes.iter().sum::<f64>() / n as f64;
        let variance = nodes.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std = variance.sqrt();
        // 4 sigma: the mean's sigma is 1.5/sqrt(n), the standard
        // deviation's about 1.5/sqrt(2n).
        assert!(mean.abs() < 4.0 * 1.5 / (n as f64).sqrt(), "mean {mean}");
        assert!(
            (std - 1.5).abs() < 4.0 * 1.5 / (2.0 * n as f64).sqrt(),
            "std {std}"
        );
    }

    #[test]
    fn an_ill_formed_noise_is_refused_with_a_reason() {
        assert!(check_smooth_noise(1.0, 0.5).is_ok());
        assert!(check_smooth_noise(0.0, 0.5).is_ok());
        assert!(check_smooth_noise(-1.0, 0.5)
            .unwrap_err()
            .contains("amplitude"));
        assert!(check_smooth_noise(1.0, 0.0)
            .unwrap_err()
            .contains("time_scale"));
        assert!(check_smooth_noise(1.0, f64::NAN)
            .unwrap_err()
            .contains("time_scale"));
    }
}
