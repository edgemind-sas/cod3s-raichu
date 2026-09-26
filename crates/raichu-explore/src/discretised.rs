//! Discretised exploration of the sequence tree, for every law the engine
//! carries and for continuous evolution.
//!
//! # What is discretised
//!
//! Not each transition's own firing date, but the **random next event**:
//! which armed stochastic transition fires first, and when. At a node
//! where nothing fires at the current instant `t0`, the engine runs in
//! deferred-draw mode ([`StochasticDates::Deferred`]): every armed
//! stochastic transition `j` is a competitor with a conditional
//! cumulative hazard `H_j(t)` over the window `[t0, t_end]`, where
//! `t_end` is the first of the next deterministic event, the next located
//! watched crossing and the horizon:
//!
//! - a fixed law (exponential, Weibull, lognormal, gamma, uniform,
//!   empirical) contributes `H(a + t - t0) - H(a)`, conditional on the age
//!   `a` it has accrued under its interruption policy;
//! - a state-dependent exponential whose rate is piecewise constant
//!   contributes `λ (t - t0)`;
//! - one whose rate varies continuously contributes its integrated hazard,
//!   read from the engine's dense samples along the deterministic flow
//!   ([`Engine::probe_deferred`]) and interpolated linearly.
//!
//! With `H = Σ H_j`, the next event occurs in the window with probability
//! `M = 1 - exp(-H(t_end))`. That mass is cut into `K` cells of equal
//! probability `M / K` (the boundaries invert `H`, which is monotone).
//! Within a cell `[a, b]`, the mass is split among the competitors in
//! proportion to their hazard increments `H_j(b) - H_j(a)`, and each
//! (competitor, cell) pair of nonzero mass is a child, fired at the cell's
//! **mass-median** instant (where `1 - exp(-H)` reaches the middle of the
//! cell's mass) with [`Engine::fire_deferred_at`]. One more child carries
//! the survival mass `exp(-H(t_end))`: it lets time run to `t_end`, where
//! the deterministic event or the watched crossing fires next. At the
//! horizon it is dropped: a sequence that has not reached the target by
//! the horizon contributes nothing. Children are visited competitor by
//! competitor (transition index order), cell by cell, then the survival
//! child. Between branch points the engine runs deterministically, so the
//! explorer never re-implements a guard, an effect or a policy.
//!
//! An instantaneous transition, a deterministic delay that is due, or a
//! watched transition whose guard holds fires first, exactly as in the
//! exact driver (branching over its destinations).
//!
//! A window no wider than the event-location tolerance
//! ([`raichu_numeric::SolverParams::tol_event`], or `1e-14` relative to
//! `t0` when that is larger), such as two deterministic dates apart by
//! round-off, is treated as empty: its mass, of order `rate x tol_event`,
//! goes to the survival branch, which then carries probability 1: a
//! deterministic end event fires at its date in the same child (the node
//! branches as if the dates coincided exactly), a watched one is reached
//! by letting time run, and at the horizon the node is a leaf. Firing at a cell's median can locate
//! a watched crossing earlier than the probe did, when the median lies
//! inside the crossing's location bracket: the firing is then retried
//! `2 tol_event` before each newly located crossing, a bounded number of
//! times. A competitor's window hazard that is NaN or negative is a typed
//! error, not a competitor silently left out.
//!
//! # Merging, cut-offs and bounds
//!
//! Leaves that reach the target along the same ordered list of fired
//! transitions (through different cells) are one sequence: their
//! probabilities are summed before ranking. The bounds are bounds **on the
//! discretised model**: the lower bound sums the retained mass, the upper
//! bound adds every pruned node's mass. The minimal-probability cut-off
//! compares a node's mass multiplied by `K` to the power of the cell
//! branchings on its path (an estimate of the mass of the whole event
//! sequence, rather than of one cell's fragment of it); the pruned node's
//! own mass is what goes to the upper bound. The maximal length counts
//! fired transitions (letting time run is not one), the failure count and
//! the branch cap work as in the exact driver.
//!
//! # Discretisation error and cost
//!
//! The run is repeated at level `2K` ([`DiscretisedSettings::refine`]);
//! the reported result is the `2K` one, with the estimate
//! `max(|lower_2K - lower_K|, |upper_2K - upper_K|)` ([`Refinement`]),
//! flagged truncation-dominated when either run's gap exceeds it. It is an
//! estimate, not a bound. A node has up to `K x competitors + 1` children,
//! so the tree grows as about `(K x competitors)^depth` nodes, and the
//! refined pass costs about `2^depth` times the base one: the branch cap
//! ([`DEFAULT_MAX_BRANCHES`] by default) bounds each pass, and Monte-Carlo
//! is the cheaper tool for long sequences.
//!
//! # Determinism
//!
//! As in the exact driver: fixed child order, root children split across
//! workers with their own share of the branch cap, reduction in child
//! order. The result is bit-identical whatever the thread count.

use raichu_core::compile::CLaw;
use raichu_core::{
    CompiledModel, DeferredTransition, Engine, EngineConfig, EngineError, ProbeStop,
    StochasticDates,
};
use raichu_numeric::{Law, LawError};

use crate::result::{
    Algorithm, Cutoffs, Discretisation, ExplorationResult, Precision, Refinement,
    DEFAULT_GAP_TOLERANCE,
};
use crate::walk::{
    assemble, drive, engine_config, instantaneous_branches, invalid, validate_common, Common,
    Expansion, NodeMass, Strategy, Timed,
};

/// Default discretisation level `K`: cells per timed node.
pub const DEFAULT_LEVEL: u32 = 8;

/// Default branch cap of a discretised exploration (expanded nodes per
/// pass), set by [`DiscretisedSettings::new`]: the tree grows as
/// `(K x competitors)^depth`, so an uncapped run is a declared choice.
pub const DEFAULT_MAX_BRANCHES: u64 = 1_000_000;

/// Settings of a discretised exploration.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscretisedSettings {
    /// Name of the model [`raichu_model::Target`] (feared event) explored
    /// to.
    pub target: String,
    /// Horizon `t`, in the model's time unit: a sequence counts when the
    /// target is reached by `t`. Finite and nonnegative.
    pub horizon: f64,
    /// The declared cut-offs, each optional. [`DiscretisedSettings::new`]
    /// sets the branch cap to [`DEFAULT_MAX_BRANCHES`]; it applies to each
    /// pass of a refinement separately.
    pub cutoffs: Cutoffs,
    /// Relative gap `(upper - lower) / upper` above which the result is
    /// flagged inconclusive, in `[0, 1]`.
    pub gap_tolerance: f64,
    /// Discretisation level `K >= 1`: the number of equal-mass cells the
    /// next-event distribution is cut into at every timed node. Default
    /// [`DEFAULT_LEVEL`].
    pub level: u32,
    /// Whether to estimate the discretisation error by refinement (a
    /// second run at `2K`, which is the one reported). Default `true`;
    /// when `false`, the result states that no estimate was made.
    pub refine: bool,
    /// Worker threads (`None` = rayon default). The result does not
    /// depend on it.
    pub threads: Option<usize>,
}

impl DiscretisedSettings {
    /// Settings for `target` at `horizon`: level [`DEFAULT_LEVEL`] with
    /// refinement, the branch cap [`DEFAULT_MAX_BRANCHES`] and no other
    /// cut-off, the default gap tolerance and thread count.
    #[must_use]
    pub fn new(target: impl Into<String>, horizon: f64) -> Self {
        DiscretisedSettings {
            target: target.into(),
            horizon,
            cutoffs: Cutoffs {
                max_branches: Some(DEFAULT_MAX_BRANCHES),
                ..Cutoffs::default()
            },
            gap_tolerance: DEFAULT_GAP_TOLERANCE,
            level: DEFAULT_LEVEL,
            refine: true,
            threads: None,
        }
    }

    fn common(&self) -> Common<'_> {
        Common {
            target: &self.target,
            horizon: self.horizon,
            cutoffs: &self.cutoffs,
            gap_tolerance: self.gap_tolerance,
            threads: self.threads,
        }
    }
}

/// Explore the sequence tree of `model` to the target `settings.target`
/// with the discretised algorithm (see the module documentation).
///
/// Every sequence probability is a probability of the discretised model;
/// the per-sequence `error_bound` is 0 and the discretisation error is
/// stated globally, in [`ExplorationResult::discretisation`]. The
/// `precision` field of the result is not used by this algorithm and
/// holds its default.
///
/// # Errors
///
/// - [`EngineError::InvalidStudyParameter`], before anything runs, for a
///   setting outside its domain: an unknown target, a negative or
///   non-finite horizon, a minimal probability outside `(0, 1]`, a branch
///   cap of 0, a gap tolerance outside `[0, 1]`, 0 threads, a failure
///   cut-off on a model declaring no `failure`, or a level of 0 (or one
///   whose double overflows, with refinement).
/// - [`EngineError::InstantaneousCycle`] when more than
///   [`EngineConfig::max_fixpoint_iterations`] instantaneous firings
///   follow one another along an explored sequence.
/// - [`EngineError::TypeError`] when a law's hazard cannot be evaluated.
/// - any engine error raised while advancing or firing.
pub fn explore_discretised(
    model: &CompiledModel,
    settings: &DiscretisedSettings,
) -> Result<ExplorationResult, EngineError> {
    validate_common(model, &settings.common())?;
    let max_level = if settings.refine {
        u32::MAX / 2
    } else {
        u32::MAX
    };
    if settings.level == 0 || settings.level > max_level {
        return Err(invalid(
            "level",
            format!(
                "a discretisation level lies in [1, {max_level}], got {}",
                settings.level
            ),
        ));
    }
    let mut base = run(model, settings, settings.level)?;
    if !settings.refine {
        base.discretisation = Some(Discretisation {
            level: settings.level,
            refinement: None,
        });
        return Ok(base);
    }
    let fine_level = 2 * settings.level;
    let mut fine = run(model, settings, fine_level)?;
    let error_estimate = (fine.lower - base.lower)
        .abs()
        .max((fine.upper - base.upper).abs());
    // The two lower bounds differ by the discretisation error plus what
    // each run truncated, and what a run truncated is at most its gap.
    // When the gaps add up to half the estimate or more, truncation may
    // account for as much of the estimate as discretisation does. This
    // covers the case of a gap above the estimate, and also the case where
    // the pruned nodes were bound to reach the target (their mass then
    // leaves the lower bound one for one, and the estimate absorbs it).
    let gaps = (base.upper - base.lower) + (fine.upper - fine.lower);
    let truncation_dominated = gaps > 0.0 && 2.0 * gaps >= error_estimate;
    fine.discretisation = Some(Discretisation {
        level: fine_level,
        refinement: Some(Refinement {
            base_level: settings.level,
            base_lower: base.lower,
            base_upper: base.upper,
            error_estimate,
            truncation_dominated,
        }),
    });
    Ok(fine)
}

/// One pass at `level`.
fn run(
    model: &CompiledModel,
    settings: &DiscretisedSettings,
    level: u32,
) -> Result<ExplorationResult, EngineError> {
    let common = settings.common();
    let config = EngineConfig {
        t_max: settings.horizon,
        stochastic_dates: StochasticDates::Deferred,
        ..engine_config()
    };
    let strategy = DiscretisedStrategy {
        model,
        level,
        horizon: settings.horizon,
        tol_event: config.ode.tol_event,
    };
    let partials = drive(model, &common, &strategy, &config, &(), true)?;
    Ok(assemble(
        model,
        &common,
        Algorithm::Discretised,
        &Precision::default(),
        partials,
        true,
    ))
}

/// What a child does from its parent node.
#[derive(Debug, Clone, Copy)]
enum Action {
    /// Fire `transition` now, into its destination at `branch`.
    Now { transition: usize, branch: usize },
    /// Fire the deferred `transition` at instant `time`.
    At { transition: usize, time: f64 },
    /// Let time run to `time`, firing nothing.
    Advance { time: f64 },
    /// Let time run to `time`, the date of the deterministic `transition`,
    /// then fire it into its destination at `branch` (a window below the
    /// event-location tolerance, treated as empty).
    AdvanceAndFire {
        time: f64,
        transition: usize,
        branch: usize,
    },
}

/// One branch out of a node, with its conditional probability `factor`
/// and the number of cell branchings on its path.
#[derive(Debug, Clone, Copy)]
struct Child {
    action: Action,
    factor: f64,
    cells: u32,
}

/// How many times a firing refused for coming after a located watched
/// crossing is retried before the crossing ([`DiscretisedStrategy::fire`]).
const MAX_CROSSING_RETRIES: u32 = 8;

/// The discretised node expansion.
struct DiscretisedStrategy<'m> {
    model: &'m CompiledModel,
    level: u32,
    horizon: f64,
    /// The engine's event-location tolerance
    /// ([`raichu_numeric::SolverParams::tol_event`]), in time units: the
    /// width below which a window is treated as empty, and the step back
    /// of a firing retried before a watched crossing.
    tol_event: f64,
}

/// The conditional cumulative hazard of one competitor over the window.
enum Hazard {
    /// Constant rate λ: `λ (t - t0)`.
    Rate(f64),
    /// A fixed law at age `age`: `H(age + t - t0) - H(age)`.
    Law { law: Law, age: f64 },
    /// Dense samples `(t, H(t) - H(t0))`, strictly increasing in `t`.
    Sampled(Vec<(f64, f64)>),
}

impl Hazard {
    fn at(&self, t0: f64, t: f64) -> Result<f64, LawError> {
        let s = (t - t0).max(0.0);
        match self {
            Hazard::Rate(rate) => Ok(rate * s),
            Hazard::Law { law, age } => law.conditional_cumulative_hazard(*age, s),
            Hazard::Sampled(samples) => Ok(interpolate(samples, t)),
        }
    }
}

/// Linear interpolation of `samples` at `t`, held constant outside.
fn interpolate(samples: &[(f64, f64)], t: f64) -> f64 {
    let k = samples.partition_point(|&(time, _)| time <= t);
    match (
        k.checked_sub(1).and_then(|i| samples.get(i)),
        samples.get(k),
    ) {
        (Some(&(t1, h1)), Some(&(t2, h2))) => h1 + (h2 - h1) * (t - t1) / (t2 - t1),
        (Some(&(_, h)), None) | (None, Some(&(_, h))) => h,
        (None, None) => 0.0,
    }
}

/// The largest float strictly below `t` (for `t > 0`), `t` otherwise.
fn just_before(t: f64) -> f64 {
    if t > 0.0 && t.is_finite() {
        f64::from_bits(t.to_bits() - 1)
    } else {
        t
    }
}

/// A competitor's cumulative hazard over the window, `h` at `time`,
/// checked: it is nonnegative, and `+inf` is legitimate (a law whose
/// support ends inside the window fires in it surely). A NaN or a negative
/// value is refused with [`EngineError::TypeError`] naming the transition
/// `name`, rather than silently leaving the competitor out of the window.
fn checked_hazard(name: &str, time: f64, h: f64) -> Result<f64, EngineError> {
    if h >= 0.0 {
        Ok(h)
    } else {
        Err(EngineError::TypeError {
            time,
            detail: format!(
                "the cumulative hazard of deferred transition `{name}` over the window \
                 is {h}, not a nonnegative number"
            ),
        })
    }
}

/// A fixed stochastic law as a [`Law`]; `None` for the others.
fn fixed_law(law: &CLaw) -> Option<Result<Law, LawError>> {
    Some(match law {
        CLaw::Weibull(shape, scale) => Law::weibull(*shape, *scale),
        CLaw::Lognormal(mu, sigma) => Law::lognormal(*mu, *sigma),
        CLaw::Gamma(shape, scale) => Law::gamma(*shape, *scale),
        CLaw::Uniform(low, high) => Law::uniform(*low, *high),
        CLaw::Empirical(points) => Law::empirical(points.clone()),
        _ => return None,
    })
}

impl DiscretisedStrategy<'_> {
    /// Fire the deferred `transition` before the event located at `limit`
    /// by a firing that came too late.
    ///
    /// The flow integrated to a cell's median can locate a watched
    /// crossing earlier than the probe did (the probe's crossing is the
    /// upper end of a bracket `tol_event` wide, and a cell median can lie
    /// inside that bracket past the true crossing). The located `limit` is
    /// itself the upper end of such a bracket, so the firing is retried at
    /// `limit - 2 tol_event` (never before the current instant), each
    /// refusal moving the limit earlier, at most
    /// [`MAX_CROSSING_RETRIES`] times; the last refusal is returned.
    fn fire_before_crossing(
        &self,
        engine: &mut Engine<'_>,
        transition: usize,
        mut limit: f64,
    ) -> Result<(), EngineError> {
        let mut attempt = 1;
        loop {
            let time = (limit - 2.0 * self.tol_event).max(engine.current_time());
            match engine.fire_deferred_at(transition, time, None) {
                Ok(_) => return Ok(()),
                Err(EngineError::DeferredFiringTooLate { limit: earlier, .. })
                    if attempt < MAX_CROSSING_RETRIES =>
                {
                    limit = earlier;
                    attempt += 1;
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// The window hazard `h` of transition `idx` at `time`, checked.
    fn checked(&self, idx: usize, time: f64, h: f64) -> Result<f64, EngineError> {
        checked_hazard(&self.model.transitions[idx].name, time, h)
    }

    fn law_error(&self, idx: usize, time: f64, error: &LawError) -> EngineError {
        EngineError::TypeError {
            time,
            detail: format!(
                "law of deferred transition `{}`: {error}",
                self.model.transitions[idx].name
            ),
        }
    }

    /// The window hazard of a running deferred transition.
    fn hazard(
        &self,
        engine: &Engine<'_>,
        deferred: &DeferredTransition,
        continuous: &[usize],
        samples: &[raichu_core::HazardSample],
    ) -> Result<Hazard, EngineError> {
        let idx = deferred.index;
        let t0 = engine.current_time();
        Ok(match &deferred.law {
            CLaw::Exp(rate) => Hazard::Rate(*rate),
            CLaw::ExpVar {
                continuous: true, ..
            } if continuous.contains(&idx) => {
                let column = continuous.iter().position(|&i| i == idx).unwrap_or(0);
                let start = samples
                    .first()
                    .and_then(|sample| sample.hazards.get(column))
                    .copied()
                    .unwrap_or(0.0);
                Hazard::Sampled(
                    samples
                        .iter()
                        .map(|sample| {
                            let h = sample.hazards.get(column).copied().unwrap_or(start);
                            (sample.time, (h - start).max(0.0))
                        })
                        .collect(),
                )
            }
            CLaw::ExpVar { .. } => Hazard::Rate(engine.armed_rate(idx)?.unwrap_or(0.0)),
            law => match fixed_law(law) {
                Some(Ok(law)) => Hazard::Law {
                    law,
                    age: deferred.age,
                },
                Some(Err(error)) => return Err(self.law_error(idx, t0, &error)),
                None => Hazard::Rate(0.0),
            },
        })
    }
}

/// The competitors of a window: transition index and hazard.
struct Window {
    t0: f64,
    t_end: f64,
    competitors: Vec<(usize, Hazard)>,
}

impl Window {
    fn hazard_of(&self, j: usize, t: f64) -> Result<f64, (usize, LawError)> {
        let (idx, hazard) = &self.competitors[j];
        hazard.at(self.t0, t).map_err(|e| (*idx, e))
    }

    fn total(&self, t: f64) -> Result<f64, (usize, LawError)> {
        let mut sum = 0.0;
        for j in 0..self.competitors.len() {
            sum += self.hazard_of(j, t)?;
        }
        Ok(sum)
    }

    /// The first instant of the window where the total hazard reaches
    /// `h` (`h` below the total at the window's end).
    fn invert(&self, h: f64) -> Result<f64, (usize, LawError)> {
        // Closed forms: constant rates only, or one fixed law.
        if self
            .competitors
            .iter()
            .all(|(_, hazard)| matches!(hazard, Hazard::Rate(_)))
        {
            let rate: f64 = self
                .competitors
                .iter()
                .map(|(_, hazard)| match hazard {
                    Hazard::Rate(rate) => *rate,
                    _ => 0.0,
                })
                .sum();
            return Ok((self.t0 + h / rate).clamp(self.t0, self.t_end));
        }
        if let [(idx, Hazard::Law { law, age })] = self.competitors.as_slice() {
            let u = -(-h).exp_m1();
            let s = law.conditional_quantile(*age, u).map_err(|e| (*idx, e))?;
            return Ok((self.t0 + s).clamp(self.t0, self.t_end));
        }
        // Bisection on the monotone total hazard, to full precision.
        let (mut lo, mut hi) = (self.t0, self.t_end);
        for _ in 0..200 {
            let mid = lo + 0.5 * (hi - lo);
            if mid <= lo || mid >= hi {
                break;
            }
            if self.total(mid)? >= h {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Ok(hi)
    }
}

impl Strategy for DiscretisedStrategy<'_> {
    type Child = Child;
    type Shared = ();

    fn immediate(&self, engine: &Engine<'_>) -> Option<usize> {
        let now = engine.current_time();
        engine
            .fireable()
            .first()
            .filter(|f| f.date.is_some_and(|date| date <= now))
            .map(|f| f.index)
    }

    fn instantaneous_children(
        &self,
        model: &CompiledModel,
        idx: usize,
        _parent: &(),
        via: Option<&Child>,
    ) -> (Vec<Child>, ()) {
        let cells = via.map_or(0, |child| child.cells);
        let children = instantaneous_branches(model, idx)
            .into_iter()
            .map(|(branch, factor)| Child {
                action: Action::Now {
                    transition: idx,
                    branch,
                },
                factor,
                cells,
            })
            .collect();
        (children, ())
    }

    fn timed(
        &self,
        engine: &mut Engine<'_>,
        _parent: &(),
        via: Option<&Child>,
        _path: &[(u32, u32)],
    ) -> Timed<Child, ()> {
        let t0 = engine.current_time();
        if t0 >= self.horizon {
            return Ok(None);
        }
        let cells = via.map_or(0, |child| child.cells);

        // The window: probe the deterministic flow, then come back.
        let snapshot = engine.snapshot();
        let probe = engine.probe_deferred(self.horizon);
        engine.restore(&snapshot);
        let probe = probe?;
        let t_end = probe.stop;
        let end_event = match probe.reason {
            ProbeStop::Deterministic { index } | ProbeStop::Watched { index } => Some(index),
            ProbeStop::Limit | ProbeStop::Horizon => None,
        };
        if t_end <= t0 {
            // An event due at this very instant that the fireable list did
            // not show (a watched guard holding at the stop): fire it.
            let Some(index) = end_event else {
                return Ok(None);
            };
            let (children, ()) = self.instantaneous_children(self.model, index, &(), via);
            return Ok(Some(Expansion {
                children,
                shared: (),
                snapshot: Some(snapshot),
            }));
        }
        if t_end - t0 <= self.tol_event.max(1e-14 * t0.abs().max(1.0)) {
            // A window below the event-location tolerance (two
            // deterministic dates apart by round-off, say) carries a mass
            // of order `rate x tol_event`: it is treated as empty, and the
            // end event happens with probability 1. A deterministic end
            // event fires at its date in the same child, so the node
            // branches as it would had the two dates coincided exactly; a
            // watched one is reached by letting time run. With no end
            // event (the horizon), the node is a leaf, as the survival
            // child is at the horizon.
            let children = match probe.reason {
                ProbeStop::Deterministic { index } => instantaneous_branches(self.model, index)
                    .into_iter()
                    .map(|(branch, factor)| Child {
                        action: Action::AdvanceAndFire {
                            time: t_end,
                            transition: index,
                            branch,
                        },
                        factor,
                        cells,
                    })
                    .collect(),
                ProbeStop::Watched { .. } => vec![Child {
                    action: Action::Advance { time: t_end },
                    factor: 1.0,
                    cells,
                }],
                ProbeStop::Limit | ProbeStop::Horizon => return Ok(None),
            };
            return Ok(Some(Expansion {
                children,
                shared: (),
                snapshot: Some(snapshot),
            }));
        }

        let mut competitors = Vec::new();
        for deferred in engine.deferred().iter().filter(|d| !d.paused) {
            let hazard = self.hazard(engine, deferred, &probe.transitions, &probe.samples)?;
            competitors.push((deferred.index, hazard));
        }
        let window = Window {
            t0,
            t_end,
            competitors,
        };
        let law_error = |(idx, error): (usize, LawError)| self.law_error(idx, t0, &error);

        // Competitors that can fire in the window, and their hazard at its
        // end.
        let mut ends = Vec::with_capacity(window.competitors.len());
        for j in 0..window.competitors.len() {
            let h = window.hazard_of(j, t_end).map_err(law_error)?;
            ends.push(self.checked(window.competitors[j].0, t_end, h)?);
        }
        let keep: Vec<bool> = ends.iter().map(|&h| h > 0.0).collect();
        let window = Window {
            competitors: window
                .competitors
                .into_iter()
                .zip(&keep)
                .filter_map(|(c, &k)| k.then_some(c))
                .collect(),
            ..window
        };
        let total_end: f64 = ends.iter().filter(|&&h| h > 0.0).sum();

        let mut children = Vec::new();
        if total_end > 0.0 {
            let k = f64::from(self.level);
            let mass = -(-total_end).exp_m1();
            let cell_mass = mass / k;
            let n = window.competitors.len();
            // Per cell: the representative instant and each competitor's
            // share of the cell mass.
            let mut cells_out: Vec<(f64, Vec<f64>)> = Vec::with_capacity(self.level as usize);
            let mut a = t0;
            let mut h_a = vec![0.0; n];
            for c in 1..=self.level {
                let cf = f64::from(c);
                let b = if c == self.level {
                    t_end
                } else {
                    window
                        .invert(-(-cf * mass / k).ln_1p())
                        .map_err(law_error)?
                };
                let mut median = window
                    .invert(-(-(cf - 0.5) * mass / k).ln_1p())
                    .map_err(law_error)?;
                if median >= t_end {
                    median = just_before(t_end).max(t0);
                }
                let mut h_b = Vec::with_capacity(n);
                for j in 0..n {
                    let h = window.hazard_of(j, b).map_err(law_error)?;
                    h_b.push(self.checked(window.competitors[j].0, b, h)?);
                }
                let shares = shares(&window, &h_a, &h_b, a, median).map_err(law_error)?;
                cells_out.push((median, shares));
                a = b;
                h_a = h_b;
            }
            for (j, (idx, _)) in window.competitors.iter().enumerate() {
                for (median, shares) in &cells_out {
                    let factor = cell_mass * shares[j];
                    if factor > 0.0 {
                        children.push(Child {
                            action: Action::At {
                                transition: *idx,
                                time: *median,
                            },
                            factor,
                            cells: cells + 1,
                        });
                    }
                }
            }
        }
        if end_event.is_some() {
            let survival = (-total_end).exp();
            if survival > 0.0 {
                children.push(Child {
                    action: Action::Advance { time: t_end },
                    factor: survival,
                    cells,
                });
            }
        }
        if children.is_empty() {
            return Ok(None);
        }
        Ok(Some(Expansion {
            children,
            shared: (),
            snapshot: Some(snapshot),
        }))
    }

    fn node_mass(&self, pi: f64, _parent: &(), _via: Option<&Child>) -> NodeMass {
        NodeMass {
            value: pi,
            error_bound: 0.0,
            imprecise: false,
        }
    }

    fn child_mass(&self, pi: f64, _shared: &(), child: &Child) -> (f64, f64, f64) {
        let child_pi = pi * child.factor;
        let score = if child.cells == 0 || child_pi <= 0.0 {
            child_pi
        } else {
            (child_pi.ln() + f64::from(child.cells) * f64::from(self.level).ln()).exp()
        };
        (child_pi, child_pi, score)
    }

    fn fire(&self, engine: &mut Engine<'_>, child: &Child) -> Result<(), EngineError> {
        match child.action {
            Action::Now { transition, branch } => {
                engine.fire_now(transition, Some(branch)).map(|_| ())
            }
            Action::At { transition, time } => {
                match engine.fire_deferred_at(transition, time, None) {
                    Ok(_) => Ok(()),
                    Err(EngineError::DeferredFiringTooLate { limit, .. }) => {
                        self.fire_before_crossing(engine, transition, limit)
                    }
                    Err(error) => Err(error),
                }
            }
            Action::Advance { time } => engine.probe_deferred(time).map(|_| ()),
            Action::AdvanceAndFire {
                time,
                transition,
                branch,
            } => {
                engine.probe_deferred(time)?;
                engine.fire_now(transition, Some(branch)).map(|_| ())
            }
        }
    }

    fn step(&self, child: &Child) -> Option<(u32, u32)> {
        match child.action {
            Action::Now { transition, branch } => Some((transition as u32, branch as u32)),
            Action::At { transition, .. } => Some((transition as u32, 0)),
            Action::AdvanceAndFire {
                transition, branch, ..
            } => Some((transition as u32, branch as u32)),
            Action::Advance { .. } => None,
        }
    }
}

/// Each competitor's share of the mass of the cell `[a, b]` (hazards
/// `h_a`, `h_b` at its ends, `median` its representative instant), in
/// proportion to the hazard increments. When an increment is infinite (a
/// law whose support ends in the cell), the increments up to the median
/// are used instead, and failing that the competitors with an infinite
/// increment share equally.
fn shares(
    window: &Window,
    h_a: &[f64],
    h_b: &[f64],
    a: f64,
    median: f64,
) -> Result<Vec<f64>, (usize, LawError)> {
    let n = h_a.len();
    let normalise = |w: Vec<f64>| -> Option<Vec<f64>> {
        let sum: f64 = w.iter().sum();
        (sum.is_finite() && sum > 0.0).then(|| w.iter().map(|x| x / sum).collect())
    };
    let increments: Vec<f64> = (0..n).map(|j| (h_b[j] - h_a[j]).max(0.0)).collect();
    if let Some(shares) = normalise(increments.clone()) {
        return Ok(shares);
    }
    if median > a {
        let mut to_median = Vec::with_capacity(n);
        for (j, h) in h_a.iter().enumerate() {
            to_median.push((window.hazard_of(j, median)? - h).max(0.0));
        }
        if let Some(shares) = normalise(to_median) {
            return Ok(shares);
        }
    }
    let infinite: Vec<f64> = increments
        .iter()
        .map(|x| if x.is_infinite() { 1.0 } else { 0.0 })
        .collect();
    Ok(normalise(infinite).unwrap_or_else(|| vec![1.0 / n as f64; n]))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use super::*;

    #[test]
    fn a_nonnegative_or_infinite_window_hazard_passes_the_check() {
        for h in [0.0, 1e-300, 2.5, f64::INFINITY] {
            assert_eq!(checked_hazard("A.fail.occ", 1.0, h).unwrap(), h);
        }
    }

    #[test]
    fn a_nan_or_negative_window_hazard_is_a_typed_error_naming_the_transition() {
        for h in [f64::NAN, -1e-300, -1.0, f64::NEG_INFINITY] {
            match checked_hazard("A.fail.occ", 1.5, h) {
                Err(EngineError::TypeError { time, detail }) => {
                    assert_eq!(time, 1.5);
                    assert!(detail.contains("`A.fail.occ`"), "{detail}");
                }
                other => panic!("{h}: expected a typed error, got {other:?}"),
            }
        }
    }
}
