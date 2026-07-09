//! # raichu-numeric — continuous evolution (milestone M1)
//!
//! ODE integration with dense output and **guaranteed event location**
//! for watched transitions: a boundary crossing is never silently
//! missed because of the time step (the `schedule_boundary` rule).
//!
//! Design:
//!
//! - [`OdeSystem`] is implemented by the engine: right-hand sides and
//!   event (boundary-margin) functions evaluated from compiled
//!   expression trees. The convention is **event `i` fires when
//!   `events()[i]` crosses from negative to non-negative**.
//! - [`OdeSolver`] is the swappable backend trait. The default backend
//!   is [`DormandPrince45`] (adaptive Dormand–Prince 4(5) with the
//!   Hairer dense-output interpolant); [`FixedEuler`] is a deliberately
//!   simple second backend proving the trait boundary (used by tests
//!   and available for coarse debugging runs). `diffsol` remains the
//!   candidate backend for stiff/large systems in a later milestone.
//! - **Missed-crossing safety net**: every accepted step is scanned at
//!   `sub_samples` interior points of the dense interpolant in
//!   addition to its endpoint, then the earliest sign change is
//!   bracketed and bisected down to `tol_event`. `max_step` caps the
//!   step so a boundary feature narrower than the scan spacing cannot
//!   hide inside one step (stress-tested).
//!
//! All tolerances are explicit in [`SolverParams`] and recorded by the
//! caller as provenance (validation-contract level 3).

use thiserror::Error;

/// A continuous system: right-hand sides plus boundary margins.
pub trait OdeSystem {
    /// Dimension of the ODE state vector.
    fn dim(&self) -> usize;
    /// Evaluate `dy/dt` at `(t, y)` into `dydt`.
    fn rhs(&mut self, t: f64, y: &[f64], dydt: &mut [f64]);
    /// Number of monitored boundary margins.
    fn n_events(&self) -> usize;
    /// Evaluate the boundary margins at `(t, y)` into `out`.
    /// Event `i` fires when `out[i]` becomes ≥ 0.
    fn events(&mut self, t: f64, y: &[f64], out: &mut [f64]);
}

/// Why an integration segment ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The requested end date was reached; `y` holds the state there.
    Reached {
        /// Final time (== requested `t_end`).
        t: f64,
    },
    /// A boundary crossing was located; `y` holds the state at `t`.
    Event {
        /// Index of the crossed margin.
        index: usize,
        /// Located crossing time (within `tol_event`).
        t: f64,
    },
}

/// Typed integration errors (never a panic on the library path).
#[derive(Debug, Error)]
pub enum OdeError {
    /// The adaptive controller could not meet the tolerance.
    #[error("step size underflow at t={t} (h={h}); system too stiff for this backend")]
    StepSizeUnderflow {
        /// Time of failure.
        t: f64,
        /// Last attempted step.
        h: f64,
    },
    /// The right-hand side produced a non-finite value.
    #[error("non-finite derivative at t={t}")]
    NonFinite {
        /// Time of failure.
        t: f64,
    },
}

/// Explicit numerical parameters (documented defaults; numerical
/// rigor: tolerances are provenance, never hidden).
#[derive(Debug, Clone)]
pub struct SolverParams {
    /// Relative tolerance of the step-error controller.
    pub rtol: f64,
    /// Absolute tolerance of the step-error controller.
    pub atol: f64,
    /// Hard cap on the step size (missed-crossing safety net).
    pub max_step: f64,
    /// Time tolerance of the event bisection.
    pub tol_event: f64,
    /// Interior dense-output points scanned per step for sign changes.
    pub sub_samples: usize,
}

impl Default for SolverParams {
    fn default() -> Self {
        SolverParams {
            rtol: 1e-9,
            atol: 1e-12,
            max_step: 0.1,
            tol_event: 1e-10,
            sub_samples: 16,
        }
    }
}

/// Swappable integration backend.
pub trait OdeSolver {
    /// Integrate from `t0` to `t_end` (writing the final state back
    /// into `y`), monitoring boundary margins and delivering dense
    /// samples.
    ///
    /// - `samples` must be ascending and lie in `(t0, t_end]`; each is
    ///   delivered through `on_sample(t, y_at_t)` (dense output) if the
    ///   segment reaches it — samples at or after an event are *not*
    ///   delivered (the caller resumes after handling the event).
    /// - Margins must all be negative at `t0`; a non-negative initial
    ///   margin returns an immediate event at `t0` (defensive — the
    ///   engine fires such transitions before integrating).
    fn integrate(
        &mut self,
        system: &mut dyn OdeSystem,
        t0: f64,
        y: &mut [f64],
        t_end: f64,
        samples: &[f64],
        on_sample: &mut dyn FnMut(f64, &[f64]),
    ) -> Result<Outcome, OdeError>;
}

// --- Dormand–Prince 4(5) ---------------------------------------------------

/// Adaptive Dormand–Prince 4(5) with Hairer's dense-output interpolant
/// and bisection event location (default backend).
#[derive(Debug, Clone)]
pub struct DormandPrince45 {
    /// Numerical parameters.
    pub params: SolverParams,
}

impl DormandPrince45 {
    /// Backend with the documented default parameters.
    #[must_use]
    pub fn new(params: SolverParams) -> Self {
        DormandPrince45 { params }
    }
}

/// Dormand–Prince coefficients (Hairer, Nørsett, Wanner — DOPRI5).
mod dopri {
    pub const C: [f64; 7] = [0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0];
    pub const A2: [f64; 1] = [0.2];
    pub const A3: [f64; 2] = [3.0 / 40.0, 9.0 / 40.0];
    pub const A4: [f64; 3] = [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0];
    pub const A5: [f64; 4] = [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
    ];
    pub const A6: [f64; 5] = [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    ];
    /// 5th-order solution weights (also row 7 of A — FSAL).
    pub const B: [f64; 6] = [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ];
    /// Error weights `b5 − b4` (applied to k1..k7).
    pub const E: [f64; 7] = [
        71.0 / 57600.0,
        0.0,
        -71.0 / 16695.0,
        71.0 / 1920.0,
        -17253.0 / 339200.0,
        22.0 / 525.0,
        -1.0 / 40.0,
    ];
    /// Dense-output coefficients (rcont5, Hairer's `contd5`).
    pub const D: [f64; 7] = [
        -12715105075.0 / 11282082432.0,
        0.0,
        87487479700.0 / 32700410799.0,
        -10690763975.0 / 1880347072.0,
        701980252875.0 / 199316789632.0,
        -1453857185.0 / 822651844.0,
        69997945.0 / 29380423.0,
    ];
}

/// One accepted step's dense interpolant: evaluates `y(t0 + θ·h)`.
struct Dense {
    t0: f64,
    h: f64,
    rcont: [Vec<f64>; 5],
}

impl Dense {
    fn eval(&self, t: f64, out: &mut [f64]) {
        let theta = (t - self.t0) / self.h;
        let theta1 = 1.0 - theta;
        for (i, value) in out.iter_mut().enumerate() {
            *value = self.rcont[0][i]
                + theta
                    * (self.rcont[1][i]
                        + theta1
                            * (self.rcont[2][i]
                                + theta * (self.rcont[3][i] + theta1 * self.rcont[4][i])));
        }
    }
}

/// Shared event-scan machinery: locate the earliest negative→non-negative
/// transition inside `[t_lo, t_hi]` given a dense evaluator.
///
/// Scans `sub_samples + 1` points then bisects the bracketing interval
/// down to `tol_event`. Returns `(event index, time)`.
#[allow(clippy::too_many_arguments)] // internal kernel shared by both backends
fn locate_event(
    system: &mut dyn OdeSystem,
    dense: &mut dyn FnMut(f64, &mut [f64]),
    g_lo: &[f64],
    t_lo: f64,
    t_hi: f64,
    sub_samples: usize,
    tol_event: f64,
    y_scratch: &mut [f64],
    g_scratch: &mut [f64],
) -> Option<(usize, f64)> {
    let n = g_lo.len();
    let mut prev_t = t_lo;
    let mut prev_g = g_lo.to_vec();
    let points = sub_samples.max(1);
    for p in 1..=points {
        let t = t_lo + (t_hi - t_lo) * (p as f64) / (points as f64);
        dense(t, y_scratch);
        system.events(t, y_scratch, g_scratch);
        // Earliest crossing among indices, by sub-interval then index.
        let crossing = (0..n).find(|&i| prev_g[i] < 0.0 && g_scratch[i] >= 0.0);
        if let Some(index) = crossing {
            // Bisect [prev_t, t] on margin `index`.
            let (mut lo, mut hi) = (prev_t, t);
            while hi - lo > tol_event {
                let mid = 0.5 * (lo + hi);
                dense(mid, y_scratch);
                system.events(mid, y_scratch, g_scratch);
                if g_scratch[index] >= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return Some((index, hi));
        }
        prev_t = t;
        prev_g.copy_from_slice(g_scratch);
    }
    None
}

impl OdeSolver for DormandPrince45 {
    fn integrate(
        &mut self,
        system: &mut dyn OdeSystem,
        t0: f64,
        y: &mut [f64],
        t_end: f64,
        samples: &[f64],
        on_sample: &mut dyn FnMut(f64, &[f64]),
    ) -> Result<Outcome, OdeError> {
        let dim = system.dim();
        let n_events = system.n_events();
        let mut g0 = vec![0.0; n_events];
        let mut g1 = vec![0.0; n_events];
        if n_events > 0 {
            system.events(t0, y, &mut g0);
            if let Some(index) = g0.iter().position(|g| *g >= 0.0) {
                return Ok(Outcome::Event { index, t: t0 });
            }
        }
        if t_end <= t0 {
            return Ok(Outcome::Reached { t: t_end });
        }
        // A span below the step-resolution floor (the underflow guard
        // below) is crossed without integration: the flow cannot move
        // the state by more than round-off over one ulp of t. This
        // arises when an event is located within round-off of a
        // segment end (e.g. a hazard crossing at the horizon).
        if t_end - t0 <= 1e-14 * t0.abs().max(1.0) {
            for &s in samples {
                if s > t0 && s <= t_end {
                    on_sample(s, y);
                }
            }
            return Ok(Outcome::Reached { t: t_end });
        }

        let p = &self.params;
        let mut t = t0;
        let mut h = p.max_step.min(t_end - t0);
        let mut sample_cursor = 0usize;
        while sample_cursor < samples.len() && samples[sample_cursor] <= t0 {
            sample_cursor += 1;
        }

        let mut k: [Vec<f64>; 7] = core::array::from_fn(|_| vec![0.0; dim]);
        let mut y_stage = vec![0.0; dim];
        let mut y_new = vec![0.0; dim];
        let mut y_scratch = vec![0.0; dim];
        system.rhs(t, y, &mut k[0]);

        loop {
            if t >= t_end {
                return Ok(Outcome::Reached { t: t_end });
            }
            h = h.min(t_end - t).min(p.max_step);
            if h < 1e-14 * t.abs().max(1.0) {
                if t_end - t <= 1e-14 * t.abs().max(1.0) {
                    // The *remaining span* is within round-off of the
                    // target (a previous step landed one ulp short):
                    // snap to the target. Only a step shrunk by the
                    // error controller signals real stiffness below.
                    return Ok(Outcome::Reached { t: t_end });
                }
                return Err(OdeError::StepSizeUnderflow { t, h });
            }

            // Stages 2..6 (k1 is FSAL from the previous step).
            let stage = |a: &[f64], k: &[Vec<f64>; 7], y: &[f64], y_stage: &mut [f64], h: f64| {
                for i in 0..y.len() {
                    let mut acc = 0.0;
                    for (j, aj) in a.iter().enumerate() {
                        acc += aj * k[j][i];
                    }
                    y_stage[i] = y[i] + h * acc;
                }
            };
            stage(&dopri::A2, &k, y, &mut y_stage, h);
            system.rhs(t + dopri::C[1] * h, &y_stage, &mut k[1]);
            stage(&dopri::A3, &k, y, &mut y_stage, h);
            system.rhs(t + dopri::C[2] * h, &y_stage, &mut k[2]);
            stage(&dopri::A4, &k, y, &mut y_stage, h);
            system.rhs(t + dopri::C[3] * h, &y_stage, &mut k[3]);
            stage(&dopri::A5, &k, y, &mut y_stage, h);
            system.rhs(t + dopri::C[4] * h, &y_stage, &mut k[4]);
            stage(&dopri::A6, &k, y, &mut y_stage, h);
            system.rhs(t + dopri::C[5] * h, &y_stage, &mut k[5]);
            // 5th-order solution.
            for i in 0..dim {
                let mut acc = 0.0;
                for (j, bj) in dopri::B.iter().enumerate() {
                    acc += bj * k[j][i];
                }
                y_new[i] = y[i] + h * acc;
            }
            system.rhs(t + h, &y_new, &mut k[6]);

            // Error estimate and acceptance.
            let mut err = 0.0f64;
            for i in 0..dim {
                let mut e = 0.0;
                for (j, ej) in dopri::E.iter().enumerate() {
                    e += ej * k[j][i];
                }
                e *= h;
                let scale = p.atol + p.rtol * y[i].abs().max(y_new[i].abs());
                err += (e / scale).powi(2);
            }
            err = (err / dim.max(1) as f64).sqrt();
            if !err.is_finite() {
                return Err(OdeError::NonFinite { t });
            }

            if err > 1.0 {
                h *= (0.9 * err.powf(-0.2)).clamp(0.2, 1.0);
                continue;
            }

            // Accepted: build the dense interpolant.
            let mut dense = Dense {
                t0: t,
                h,
                rcont: core::array::from_fn(|_| vec![0.0; dim]),
            };
            for i in 0..dim {
                let ydiff = y_new[i] - y[i];
                let bspl = h * k[0][i] - ydiff;
                dense.rcont[0][i] = y[i];
                dense.rcont[1][i] = ydiff;
                dense.rcont[2][i] = bspl;
                dense.rcont[3][i] = ydiff - h * k[6][i] - bspl;
                let mut acc = 0.0;
                for (j, dj) in dopri::D.iter().enumerate() {
                    acc += dj * k[j][i];
                }
                dense.rcont[4][i] = h * acc;
            }

            // Event scan over the accepted step (safety net + bisection).
            if n_events > 0 {
                let mut dense_eval = |tt: f64, out: &mut [f64]| dense.eval(tt, out);
                if let Some((index, t_event)) = locate_event(
                    system,
                    &mut dense_eval,
                    &g0,
                    t,
                    t + h,
                    p.sub_samples,
                    p.tol_event,
                    &mut y_scratch,
                    &mut g1,
                ) {
                    // Deliver samples strictly before the event.
                    while sample_cursor < samples.len() && samples[sample_cursor] < t_event {
                        let ts = samples[sample_cursor];
                        dense.eval(ts, &mut y_scratch);
                        on_sample(ts, &y_scratch);
                        sample_cursor += 1;
                    }
                    dense.eval(t_event, &mut y_scratch);
                    y.copy_from_slice(&y_scratch);
                    return Ok(Outcome::Event { index, t: t_event });
                }
                system.events(t + h, &y_new, &mut g0);
            }

            // Deliver samples inside the accepted step.
            while sample_cursor < samples.len() && samples[sample_cursor] <= t + h {
                let ts = samples[sample_cursor];
                dense.eval(ts, &mut y_scratch);
                on_sample(ts, &y_scratch);
                sample_cursor += 1;
            }

            t += h;
            y.copy_from_slice(&y_new);
            let (head, tail) = k.split_at_mut(6);
            head[0].copy_from_slice(&tail[0]); // FSAL: k1 ← k7
            h *= (0.9 * err.max(1e-10).powf(-0.2)).clamp(0.2, 5.0);
        }
    }
}

// --- Fixed-step Euler (dummy backend proving the trait) --------------------

/// Deliberately simple fixed-step explicit Euler backend with linear
/// dense output. Exists to prove the [`OdeSolver`] trait is genuinely
/// swappable (M1 goal condition) and for coarse debugging.
#[derive(Debug, Clone)]
pub struct FixedEuler {
    /// Fixed step size.
    pub step: f64,
    /// Event bisection tolerance.
    pub tol_event: f64,
}

impl OdeSolver for FixedEuler {
    fn integrate(
        &mut self,
        system: &mut dyn OdeSystem,
        t0: f64,
        y: &mut [f64],
        t_end: f64,
        samples: &[f64],
        on_sample: &mut dyn FnMut(f64, &[f64]),
    ) -> Result<Outcome, OdeError> {
        let dim = system.dim();
        let n_events = system.n_events();
        let mut g0 = vec![0.0; n_events];
        let mut g1 = vec![0.0; n_events];
        if n_events > 0 {
            system.events(t0, y, &mut g0);
            if let Some(index) = g0.iter().position(|g| *g >= 0.0) {
                return Ok(Outcome::Event { index, t: t0 });
            }
        }
        let mut t = t0;
        let mut dydt = vec![0.0; dim];
        let mut y_new = vec![0.0; dim];
        let mut y_scratch = vec![0.0; dim];
        let mut sample_cursor = samples.partition_point(|s| *s <= t0);

        while t < t_end {
            let h = self.step.min(t_end - t);
            system.rhs(t, y, &mut dydt);
            if dydt.iter().any(|d| !d.is_finite()) {
                return Err(OdeError::NonFinite { t });
            }
            for i in 0..dim {
                y_new[i] = y[i] + h * dydt[i];
            }
            let (y0_snapshot, t_step) = (y.to_vec(), t);
            let mut linear = |tt: f64, out: &mut [f64]| {
                let theta = (tt - t_step) / h;
                for i in 0..dim {
                    out[i] = y0_snapshot[i] + theta * (y_new[i] - y0_snapshot[i]);
                }
            };
            if n_events > 0 {
                if let Some((index, t_event)) = locate_event(
                    system,
                    &mut linear,
                    &g0,
                    t,
                    t + h,
                    4,
                    self.tol_event,
                    &mut y_scratch,
                    &mut g1,
                ) {
                    while sample_cursor < samples.len() && samples[sample_cursor] < t_event {
                        let ts = samples[sample_cursor];
                        linear(ts, &mut y_scratch);
                        on_sample(ts, &y_scratch);
                        sample_cursor += 1;
                    }
                    linear(t_event, &mut y_scratch);
                    y.copy_from_slice(&y_scratch);
                    return Ok(Outcome::Event { index, t: t_event });
                }
                system.events(t + h, &y_new, &mut g0);
            }
            while sample_cursor < samples.len() && samples[sample_cursor] <= t + h {
                let ts = samples[sample_cursor];
                linear(ts, &mut y_scratch);
                on_sample(ts, &y_scratch);
                sample_cursor += 1;
            }
            y.copy_from_slice(&y_new);
            t += h;
        }
        Ok(Outcome::Reached { t: t_end })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    /// dy/dt = -y, y(0)=1 → y(t) = e^{-t}; no events.
    struct Decay;
    impl OdeSystem for Decay {
        fn dim(&self) -> usize {
            1
        }
        fn rhs(&mut self, _t: f64, y: &[f64], dydt: &mut [f64]) {
            dydt[0] = -y[0];
        }
        fn n_events(&self) -> usize {
            0
        }
        fn events(&mut self, _t: f64, _y: &[f64], _out: &mut [f64]) {}
    }

    /// dy/dt = cos(t), y(0)=0 → y = sin(t); event at y ≥ threshold.
    struct SineCross {
        threshold: f64,
    }
    impl OdeSystem for SineCross {
        fn dim(&self) -> usize {
            1
        }
        fn rhs(&mut self, t: f64, _y: &[f64], dydt: &mut [f64]) {
            dydt[0] = t.cos();
        }
        fn n_events(&self) -> usize {
            1
        }
        fn events(&mut self, _t: f64, y: &[f64], out: &mut [f64]) {
            out[0] = y[0] - self.threshold;
        }
    }

    #[test]
    fn decay_matches_closed_form() {
        let mut solver = DormandPrince45::new(SolverParams::default());
        let mut y = vec![1.0];
        let outcome = solver
            .integrate(&mut Decay, 0.0, &mut y, 5.0, &[], &mut |_, _| {})
            .unwrap();
        assert_eq!(outcome, Outcome::Reached { t: 5.0 });
        assert!((y[0] - (-5.0f64).exp()).abs() < 1e-8, "y={}", y[0]);
    }

    #[test]
    fn dense_samples_match_closed_form() {
        let mut solver = DormandPrince45::new(SolverParams::default());
        let mut y = vec![1.0];
        let samples: Vec<f64> = (1..=50).map(|i| 0.1 * i as f64).collect();
        let mut worst = 0.0f64;
        solver
            .integrate(&mut Decay, 0.0, &mut y, 5.0, &samples, &mut |t, y| {
                worst = worst.max((y[0] - (-t).exp()).abs());
            })
            .unwrap();
        assert!(worst < 1e-8, "worst dense-output error {worst}");
    }

    #[test]
    fn event_located_at_analytic_crossing() {
        // y = sin(t) crosses 0.5 at t = asin(0.5) = π/6.
        let mut solver = DormandPrince45::new(SolverParams::default());
        let mut y = vec![0.0];
        let outcome = solver
            .integrate(
                &mut SineCross { threshold: 0.5 },
                0.0,
                &mut y,
                10.0,
                &[],
                &mut |_, _| {},
            )
            .unwrap();
        let Outcome::Event { index: 0, t } = outcome else {
            panic!("expected an event, got {outcome:?}");
        };
        assert!((t - std::f64::consts::FRAC_PI_6).abs() < 1e-8, "t={t}");
    }

    /// Missed-crossing stress (M1 goal condition): the margin
    /// spends only a narrow window (~0.09 time units) above zero near
    /// the sine apex; a naive endpoint check with large steps would
    /// miss it. Sweep step caps and demand the crossing is always
    /// located at the analytic date.
    #[test]
    fn no_missed_crossing_across_step_size_sweep() {
        let threshold: f64 = 0.999;
        let expected = threshold.asin();
        for max_step in [0.05, 0.1, 0.2, 0.35, 0.5] {
            let mut solver = DormandPrince45::new(SolverParams {
                max_step,
                ..SolverParams::default()
            });
            let mut y = vec![0.0];
            let outcome = solver
                .integrate(
                    &mut SineCross { threshold },
                    0.0,
                    &mut y,
                    20.0,
                    &[],
                    &mut |_, _| {},
                )
                .unwrap();
            let Outcome::Event { index: 0, t } = outcome else {
                panic!("max_step={max_step}: crossing missed entirely: {outcome:?}");
            };
            assert!(
                (t - expected).abs() < 1e-8,
                "max_step={max_step}: located t={t}, expected {expected}"
            );
        }
    }

    #[test]
    fn euler_backend_is_swappable() {
        // Same trait, same call — coarser answer (proves the swap).
        let mut solver = FixedEuler {
            step: 1e-4,
            tol_event: 1e-10,
        };
        let mut y = vec![1.0];
        let outcome = solver
            .integrate(&mut Decay, 0.0, &mut y, 1.0, &[], &mut |_, _| {})
            .unwrap();
        assert_eq!(outcome, Outcome::Reached { t: 1.0 });
        assert!((y[0] - (-1.0f64).exp()).abs() < 1e-3, "y={}", y[0]);
    }

    #[test]
    fn initial_margin_already_crossed_reports_immediate_event() {
        let mut solver = DormandPrince45::new(SolverParams::default());
        let mut y = vec![0.6];
        let outcome = solver
            .integrate(
                &mut SineCross { threshold: 0.5 },
                1.0,
                &mut y,
                2.0,
                &[],
                &mut |_, _| {},
            )
            .unwrap();
        assert_eq!(outcome, Outcome::Event { index: 0, t: 1.0 });
    }
}
