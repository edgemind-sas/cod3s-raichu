//! Aitken extrapolation of a flow resolution converging geometrically:
//! the measure of the rate on the attributes a torn sweep reads at its
//! tear, and the jump to the limit it points to. Where a jump may land is
//! the flow resolution's decision (`flow::jump_within_active_set`).

use super::*;

/// Smallest contraction ratio a sequence of sweeps must show before it is
/// extrapolated (see [`Extrapolation`]).
///
/// **Why one half.** Below it the plain iteration is already fast: a
/// ratio of one half reaches [`FLOW_TOLERANCE`] from a residual of order
/// one in 30 sweeps, inside the numeric budget, so extrapolating would buy
/// little and would move the settled answer of networks that settle today
/// (within the tolerance, but move it). The damped iteration that follows
/// a two-cycle contracts at one quarter on the canonical alternation and
/// is left alone for the same reason.
const EXTRAPOLATION_MIN_RATIO: f64 = 0.5;

/// Relative agreement the last three steps of a sequence must show before
/// it is extrapolated (see [`Extrapolation`]): between the two measured
/// ratios, as a fraction of the distance `1 − q` that sets the size of the
/// jump, and between the last step and the ratio times the step before it,
/// as a fraction of the last step.
///
/// **Why one percent.** The jump is `q/(1 − q)` steps long, so an error
/// `δq` on the ratio moves the extrapolated point by `δq/(1 − q)²` steps,
/// and a residual `r` off the line by `r/(1 − q)` steps. Bounding both by
/// one percent of `1 − q` and of the step keeps the extrapolated point
/// within about one percent of the distance that remained on an affine
/// map, which the sweeps that follow then take back geometrically. A sequence that is not
/// geometric (a sublinear creep, two modes still competing) fails the test
/// and keeps iterating plainly.
const EXTRAPOLATION_AGREEMENT: f64 = 0.01;

/// **Aitken extrapolation of a sweep converging geometrically.**
///
/// A torn sweep is a fixed-point map on the attributes it reads at the
/// tear ([`CompiledModel::torn_reads`]). Once the active set holds (every
/// minimum keeps its argument, every conditional its branch, every
/// operator its capping), that map is affine on the flow networks the
/// muscadet layer writes, and smooth on any network whose expressions
/// multiply quantities, so near its fixed point `x*` the sequence of
/// passes approaches it as `x_k − x* ≈ q^k` along one direction. Its rate `q` can sit arbitrarily close to one: a
/// full volume that admits only what leaves it, holding a share `s` of
/// one constituent, feeds back `1 − s` of its intake every pass, so a room
/// holding 1 % of hydrogen contracts at 0.99 and needs about 2 000 passes
/// to reach [`FLOW_TOLERANCE`]. No constant budget holds that, and no
/// derived one either: the rate is a property of the state, not of the
/// network's size.
///
/// Three consecutive steps `d₁, d₂, d₃` along the same direction measure
/// the rate twice, and when the two measures agree the limit is one jump
/// away: `x* = x₃ + q/(1 − q)·d₃` (Aitken's Δ² on the dominant mode).
/// The jump only **proposes** an iterate, and it is taken only as far as
/// it keeps the active set it was measured on
/// (`flow::jump_within_active_set`): on a map that is smooth rather than
/// affine the jump can overshoot, which the plain sweeps that follow take
/// back inside one active set, but which past a breakpoint could settle
/// the resolution on another branch's equilibrium. What the resolution
/// accepts is unchanged, a pass that moves nothing by more than the
/// tolerance under a held active set.
///
/// Engaged only on a contraction (`EXTRAPOLATION_MIN_RATIO ≤ q < 1`) whose
/// steps agree to [`EXTRAPOLATION_AGREEMENT`]; an alternation (`q < 0`)
/// is the under-relaxation's business, a drift (`q ≥ 1`) has no limit to
/// jump to, and both keep iterating as before.
#[derive(Debug, Clone, Default)]
pub(super) struct Extrapolation {
    /// The torn attributes at the last observed iterate.
    last: Vec<f64>,
    /// The step that produced it: `last` minus the iterate before.
    step: Vec<f64>,
    /// The fresh step, reused buffer.
    fresh: Vec<f64>,
    /// The rate measured on the previous pair of steps, if there was one.
    ratio: Option<f64>,
    /// Iterates observed since the last reset.
    seen: usize,
}

/// A torn attribute as a number: what the extrapolation measures steps
/// on. A discrete attribute counts as its numeric reading, so a gate that
/// flips between two passes reads as a step off any line and blocks the
/// jump; it is never written (see [`Extrapolation::observe`]).
fn reading(value: Value) -> f64 {
    match value {
        Value::Float(value) => value,
        Value::Int(value) => value as f64,
        Value::Bool(value) => f64::from(u8::from(value)),
    }
}

impl Extrapolation {
    /// Forget the sequence: the next observed iterate starts a new one.
    pub(super) fn reset(&mut self) {
        self.seen = 0;
        self.ratio = None;
    }

    /// Record the iterate a pass just produced and, when the last three
    /// steps were geometric, return the jump they point to: the multiple
    /// `q/(1 − q)` of the last step that reaches the limit. Nothing is
    /// written here; `flow::jump_within_active_set` decides how much of the
    /// jump to take, and the caller resets the sequence once it has.
    pub(super) fn observe(&mut self, torn: &[VarIdx], vars: &[Value]) -> Option<f64> {
        if torn.is_empty() {
            return None;
        }
        if self.seen == 0 {
            self.last.clear();
            self.last.extend(torn.iter().map(|&var| reading(vars[var])));
            self.seen = 1;
            return None;
        }
        self.fresh.clear();
        self.fresh.extend(
            torn.iter()
                .zip(&self.last)
                .map(|(&var, &last)| reading(vars[var]) - last),
        );
        if self.seen >= 2 {
            if let Some(reach) = self.measure() {
                return Some(reach);
            }
        }
        std::mem::swap(&mut self.step, &mut self.fresh);
        for (last, &var) in self.last.iter_mut().zip(torn) {
            *last = reading(vars[var]);
        }
        self.seen += 1;
        None
    }

    /// Measure the rate of the fresh step against the previous one and
    /// return the jump when both measures agree (see [`Extrapolation`]).
    /// Updates the stored rate either way.
    fn measure(&mut self) -> Option<f64> {
        let (mut along, mut previous) = (0.0f64, 0.0f64);
        for (&fresh, &step) in self.fresh.iter().zip(&self.step) {
            along += fresh * step;
            previous += step * step;
        }
        if !(previous > 0.0 && previous.is_finite() && along.is_finite()) {
            self.ratio = None;
            return None;
        }
        let ratio = along / previous;
        let measured_before = self.ratio.replace(ratio);
        // Neither a contraction nor slow enough to be worth a jump: the
        // common case, decided before the scan below.
        if !(EXTRAPOLATION_MIN_RATIO..1.0).contains(&ratio) {
            return None;
        }
        let (mut off_line, mut largest) = (0.0f64, 0.0f64);
        for (&fresh, &step) in self.fresh.iter().zip(&self.step) {
            off_line = off_line.max((fresh - ratio * step).abs());
            largest = largest.max(fresh.abs());
        }
        let on_line = off_line <= EXTRAPOLATION_AGREEMENT * largest;
        let steady = measured_before.is_some_and(|before| {
            (ratio - before).abs() <= EXTRAPOLATION_AGREEMENT * (1.0 - ratio)
        });
        (on_line && steady && largest > 0.0).then(|| ratio / (1.0 - ratio))
    }

    /// Write `reach` times the last step onto the torn attributes of
    /// `vars`. Only float attributes are written: an integer or a flag
    /// that the map moves is a discrete decision, not a quantity on a
    /// line.
    pub(super) fn write_jump(
        &self,
        torn: &[VarIdx],
        vars: &mut [Value],
        reach: f64,
        changed: &mut ChangeLog,
    ) {
        for (&var, &fresh) in torn.iter().zip(&self.fresh) {
            if let Value::Float(value) = vars[var] {
                changed.write(vars, var, Value::Float(value + reach * fresh));
            }
        }
    }
}
