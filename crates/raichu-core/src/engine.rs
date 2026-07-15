//! The deterministic simulation engine (M0 discrete subset + M1
//! continuous evolution).
//!
//! Implements the cycle `init → schedule → continuous → discrete →
//! update` of Desgeorges et al. 2021. Rule mapping:
//!
//! - scheduling of deterministic transitions — `schedule_deterministic`
//!   ([`Engine::refresh_schedule`]);
//! - continuous evolution up to the next scheduled date — `integrate_continuous`
//!   ([`Engine::integrate_to`]);
//! - watched transitions fired at located boundary crossings — `schedule_boundary`
//!   (margin monitoring inside [`Engine::integrate_to`]);
//! - firing of the earliest transition — `fire_transition` ([`Engine::step`]);
//! - sensitive-function propagation to fixpoint — `propagate_effects`
//!   ([`Engine::run_fixpoint`]);
//! - dropping interruptible transitions whose guard turned false —
//!   `drop_disabled` ([`Engine::refresh_schedule`]).
//!
//! `schedule_stochastic` is implemented (M2, exponential distribution). `reschedule_modifiable` is
//! implemented through the cumulative-hazard realisation of
//! state-dependent rates (`CLaw::ExpVar`): a piecewise-constant rate
//! is rescheduled at each discrete change ([`Engine::refresh_schedule`]),
//! a continuously-varying rate is integrated alongside the ODE state and
//! its firing located like a boundary crossing ([`Engine::integrate_to`]).
//!
//! **Continuous/discrete coupling semantics:** sensitive functions
//! react to *discrete* changes (transition firings and effect
//! cascades); the continuous flow influences the discrete side only
//! through **watched transitions** (the paper's mechanism) and through
//! guards re-evaluated at discrete epochs. Explicit equations are
//! recomputed at every continuous evaluation point, in declaration
//! order, before ODE right-hand sides.
//!
//! **Equality semantics** (validation contract): the engine
//! guarantees a *deterministic* fixpoint order (global function
//! declaration order) but cross-validation only compares the *converged*
//! state and the event dates. The optional confluence check re-runs each
//! fixpoint in reverse order and reports divergence as a diagnostic
//! (rather than silently returning an order-dependent result).

use crate::compile::{
    CExpr, CIndicatorTarget, CLaw, CompiledModel, FnIdx, StateIdx, TransIdx, VarIdx,
};
use raichu_expr::{AggOp, BoolOp, CmpOp, Value};
use raichu_numeric::{DormandPrince45, OdeSolver, OdeSystem, Outcome, SolverParams};
use rand_chacha::ChaCha8Rng;
use rand_distr::Distribution;
use serde::Serialize;
use std::collections::BTreeSet;
use thiserror::Error;

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Simulation horizon (events strictly after it are not fired).
    pub t_max: f64,
    /// Record the structured causal journal (zero cost when `false`).
    pub journal: bool,
    /// Record the per-trajectory sequence trace — the ordered `SeqEvent`s of
    /// fired *monitored* transitions plus the end cause when a target
    /// (feared event) is reached (zero cost when `false`). When on and a
    /// target is reached, the trajectory **early-stops** at that instant
    /// (mirroring cod3s sequence runs).
    pub sequences: bool,
    /// Re-run every fixpoint in reverse order and fail on divergence
    /// (non-confluence diagnostic; ~2× fixpoint cost when enabled).
    pub confluence_check: bool,
    /// Safety cap on fixpoint iterations: beyond it the model is
    /// declared to have an instantaneous loop (typed error, not a hang).
    pub max_fixpoint_iterations: usize,
    /// Numerical parameters of the default ODE backend (explicit,
    /// recorded as provenance — validation-contract level 3).
    pub ode: SolverParams,
    /// Ascending instants at which every indicator is sampled (dense
    /// output for continuous attributes, piecewise-constant hold for
    /// discrete ones). Empty = no sampling.
    pub samples: Vec<f64>,
    /// Master seed of the RNG policy (M2). Only consumed by stochastic
    /// distributions; deterministic models ignore it.
    pub seed: u64,
    /// Substream index (`ChaCha8Rng::set_stream`) — the Monte-Carlo
    /// driver assigns one stream per replica.
    pub rng_stream: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            t_max: f64::INFINITY,
            journal: false,
            sequences: false,
            confluence_check: false,
            max_fixpoint_iterations: 10_000,
            ode: SolverParams::default(),
            samples: Vec::new(),
            seed: 0,
            rng_stream: 0,
        }
    }
}

/// Typed runtime errors. The engine never panics on a library path.
#[derive(Debug, Error)]
pub enum EngineError {
    /// An expression combined values of incompatible kinds.
    #[error("type error at t={time}: {detail}")]
    TypeError {
        /// Simulation time of the failure.
        time: f64,
        /// Human-readable detail.
        detail: String,
    },
    /// The sensitive-function propagation did not reach a fixpoint.
    #[error(
        "no fixpoint after {iterations} iterations at t={time}: \
         probable instantaneous loop (functions keep rewriting state)"
    )]
    InstantaneousLoop {
        /// Simulation time of the failure.
        time: f64,
        /// Iteration cap that was hit.
        iterations: usize,
    },
    /// The converged state depends on the function evaluation order.
    #[error(
        "non-confluent model at t={time}: sensitive functions `{first}` and \
         `{second}` write conflicting values (converged state depends on \
         evaluation order)"
    )]
    NonConfluent {
        /// Simulation time of the diagnostic.
        time: f64,
        /// A function involved in the conflict (declaration order).
        first: String,
        /// The other function involved.
        second: String,
    },
    /// Interactive control: no transition carries this qualified name.
    #[error("unknown transition `{transition}`")]
    UnknownTransition {
        /// The requested (unresolved) transition name.
        transition: String,
    },
    /// Interactive control: the requested transition is not currently
    /// armed — it is neither date-scheduled (`pending`) nor a watched
    /// transition whose guard already holds — so there is nothing to fire.
    #[error("transition `{transition}` is not fireable at t={time} (not armed)")]
    NotFireable {
        /// The transition that could not be fired.
        transition: String,
        /// Simulation time of the attempt.
        time: f64,
    },
    /// Interactive control: a forced destination (`fire_*_to`) named a
    /// state that is not one of the transition's declared target
    /// branches.
    #[error("`{state}` is not a target branch of transition `{transition}`")]
    ForcedTargetInvalid {
        /// The transition whose branch was forced.
        transition: String,
        /// The invalid (unknown or non-target) state name.
        state: String,
    },
    /// Interactive control: a manual firing date (`set_date`) was in the
    /// past (before the current time) or non-finite.
    #[error(
        "cannot schedule transition `{transition}` at t={date} \
         (before the current time t={time})"
    )]
    DateInPast {
        /// The transition being re-dated.
        transition: String,
        /// The rejected date.
        date: f64,
        /// The current simulation time.
        time: f64,
    },
    /// The ODE backend failed (stiffness, non-finite derivatives, …).
    #[error("continuous evolution failed: {0}")]
    Ode(#[from] raichu_numeric::OdeError),
    /// Watched transitions kept firing at the same instant (Zeno-like
    /// loop on a boundary).
    #[error(
        "watched transitions keep firing at t={time} without time \
         advancing (boundary loop)"
    )]
    WatchedLoop {
        /// The stuck instant.
        time: f64,
    },
}

/// One record of the structured causal journal.
///
/// Covers two structured trace levels: attribute
/// modifications during the fixpoint phase, and transition firings.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "record", rename_all = "snake_case")]
pub enum JournalRecord {
    /// A transition fired (`fire_transition` / `schedule_boundary`).
    TransitionFired {
        /// Simulation time.
        time: f64,
        /// Qualified transition name.
        transition: String,
        /// Source state name.
        from: String,
        /// Target state name.
        to: String,
    },
    /// A sensitive function was triggered (`propagate_effects`).
    FunctionTriggered {
        /// Simulation time.
        time: f64,
        /// Qualified function name.
        function: String,
    },
    /// An attribute changed value (cause = the enclosing record above).
    AttributeChanged {
        /// Simulation time.
        time: f64,
        /// Qualified attribute name.
        attribute: String,
        /// Previous value.
        old: Value,
        /// New value.
        new: Value,
        /// Qualified name of the function that wrote it.
        cause: String,
    },
    /// A transition was scheduled (`schedule_deterministic`).
    TransitionScheduled {
        /// Simulation time.
        time: f64,
        /// Qualified transition name.
        transition: String,
        /// Planned firing date.
        firing_at: f64,
    },
    /// A pending stochastic transition was rescheduled because its
    /// state-dependent rate changed at a discrete step (`reschedule_modifiable`).
    TransitionRescheduled {
        /// Simulation time.
        time: f64,
        /// Qualified transition name.
        transition: String,
        /// New planned firing date (`+∞` serialises as `null`: the
        /// rate dropped to zero and the countdown is on hold).
        firing_at: f64,
    },
    /// A pending transition was dropped (`drop_disabled` or source left).
    TransitionDropped {
        /// Simulation time.
        time: f64,
        /// Qualified transition name.
        transition: String,
        /// Why it was dropped.
        reason: DropReason,
    },
}

/// Why a pending transition was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DropReason {
    /// Its guard turned false (`drop_disabled`) under the `reset` policy —
    /// a fresh duration is redrawn when the guard returns
    /// (interruptible transition).
    GuardFalse,
    /// Its guard turned false under the `resume` policy (RAICHU
    /// extension): the countdown is *paused* and the remaining time
    /// resumes when the guard returns.
    GuardPaused,
    /// Its automaton left the source state.
    SourceLeft,
}

/// Cumulative-hazard state of an armed state-dependent-rate transition
/// (`CLaw::ExpVar`): the transition fires when `accumulated` reaches
/// `threshold`, realising the PDMP survival `P(T > t) = exp(−∫λ dt)`
/// exactly. The threshold is drawn `Exp(1)` at arming (`schedule_stochastic`).
#[derive(Debug, Clone, Copy)]
struct Hazard {
    /// `Exp(1)` firing threshold `E`.
    threshold: f64,
    /// Hazard accumulated so far, `H = ∫ λ dt ≤ E`.
    accumulated: f64,
    /// Rate λ at `since` — supports the lazy piecewise-constant
    /// accumulation of non-continuous rates between discrete steps.
    rate: f64,
    /// Time of the last accumulation point.
    since: f64,
}

/// A fired event (the discrete structure compared at validation level 1).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Event {
    /// Firing date.
    pub time: f64,
    /// Qualified transition name.
    pub transition: String,
    /// Source state name.
    pub from: String,
    /// Target state name.
    pub to: String,
}

/// One recorded event of a trajectory's **sequence** (mirrors cod3s
/// `SeqEvent`): the entry into a monitored state. `name()` is `obj.attr`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SeqEvent {
    /// Owning component (cod3s `elt.parent().name()`).
    pub obj: String,
    /// The monitored state entered (cod3s `elt.basename()`, e.g. `occ__cc_12`).
    pub attr: String,
    /// Firing date.
    pub time: f64,
    /// Cycle-pair group id of the firing transition (internal to the
    /// cycle-filtering step; not part of the compared/serialized sequence).
    #[serde(skip)]
    pub cycle_group: Option<String>,
}

/// One trajectory's recorded sequence: the ordered monitored-transition
/// firings plus the end cause/time (the reached target, or `None` if the
/// trajectory ran to `t_max` without reaching one). Weight 1 per raw
/// trajectory; the Monte-Carlo pipeline groups and re-weights them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sequence {
    /// Ordered monitored-state entries.
    pub events: Vec<SeqEvent>,
    /// Reached target's name (`end_cause`), or `None` if none was reached.
    pub end_cause: Option<String>,
    /// Time the target was reached, or the horizon when none was.
    pub end_time: f64,
    /// Statistical weight (1 for a raw trajectory).
    pub weight: f64,
}

/// Kind of an armed transition, for interactive inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FireableKind {
    /// Deterministic delay.
    Delay,
    /// A sampled stochastic law (exponential, Weibull, lognormal, …).
    Stochastic,
    /// Instantaneous branching (fires at the current instant).
    Inst,
    /// Watched boundary transition (fires when its margin is crossed
    /// during continuous evolution).
    Watched,
}

/// One armed transition offered to interactive control
/// ([`Engine::fireable`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Fireable {
    /// Transition index — the stable handle for [`Engine::fire_idx`].
    pub index: usize,
    /// Qualified transition name (`component.automaton.transition`).
    pub transition: String,
    /// Kind of occurrence law.
    pub kind: FireableKind,
    /// Scheduled firing date; `None` for a watched transition whose
    /// boundary has not been located yet (its guard is not yet true —
    /// the crossing is found only during continuous evolution).
    pub date: Option<f64>,
}

/// An opaque checkpoint of the engine's full mutable trajectory state
/// (time, discrete + continuous attributes, schedule, RNG, recorded
/// history), produced by [`Engine::snapshot`] and reinstated by
/// [`Engine::restore`]. Cloning the RNG makes any continuation after a
/// restore bit-for-bit reproducible.
#[derive(Debug, Clone)]
pub struct Snapshot {
    time: f64,
    vars: Vec<Value>,
    states: Vec<StateIdx>,
    pending: Vec<Option<f64>>,
    frozen: Vec<Option<f64>>,
    hazards: Vec<Option<Hazard>>,
    events: Vec<Event>,
    journal: Vec<JournalRecord>,
    seq_events: Vec<SeqEvent>,
    seq_end: Option<(String, f64)>,
    indicator_series: Vec<IndicatorSeries>,
    sampled: Vec<IndicatorSeries>,
    sample_cursor: usize,
    watched_streak: (f64, usize),
    rng: ChaCha8Rng,
    worklist: BTreeSet<FnIdx>,
}

/// An indicator's recorded change-points `(time, value)`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IndicatorSeries {
    /// Indicator name.
    pub name: String,
    /// Change-points: value observed from each time onward (first entry
    /// is the initial value at t = 0).
    pub points: Vec<(f64, Value)>,
}

/// Provenance metadata attached to every result (reproducibility by
/// construction).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Provenance {
    /// Engine version (workspace version).
    pub engine_version: String,
    /// Model name.
    pub model: String,
    /// Simulation horizon.
    pub t_max: f64,
    /// RNG seed (`None` in the deterministic engine; the field exists
    /// so M2 introduces no schema change).
    pub seed: Option<u64>,
    /// Relative tolerance of the ODE controller (level-3 provenance).
    pub ode_rtol: f64,
    /// Event-location time tolerance (level-3 provenance).
    pub ode_tol_event: f64,
}

/// Full result of a simulation run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SimulationResult {
    /// Fired events in order (validation level 1).
    pub events: Vec<Event>,
    /// Indicator change-point series.
    pub indicators: Vec<IndicatorSeries>,
    /// Indicator values at the requested sample instants (level-3
    /// trajectory comparison; empty when no schedule was given).
    pub samples: Vec<IndicatorSeries>,
    /// Causal journal (empty when disabled).
    pub journal: Vec<JournalRecord>,
    /// Recorded sequence (`None` when sequence recording is disabled).
    pub sequence: Option<Sequence>,
    /// Provenance metadata.
    pub provenance: Provenance,
    /// Final simulation time.
    pub final_time: f64,
}

/// Evaluate a compiled expression against an explicit state (usable
/// both by the engine and by the continuous-system adapter).
fn eval_expr(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
) -> Result<Value, EngineError> {
    match expr {
        CExpr::Const(value) => Ok(*value),
        CExpr::Var(idx) => Ok(vars[*idx]),
        CExpr::StateActive { automaton, state } => Ok(Value::Bool(states[*automaton] == *state)),
        CExpr::PortAgg { sources, agg } => eval_agg(model, vars, time, sources, *agg),
        CExpr::Cmp { op, lhs, rhs } => {
            let lhs = eval_expr(model, vars, states, time, lhs)?;
            let rhs = eval_expr(model, vars, states, time, rhs)?;
            eval_cmp(time, *op, lhs, rhs)
        }
        CExpr::Bool { op, args } => match op {
            BoolOp::And => {
                for arg in args {
                    if !eval_bool(model, vars, states, time, arg)? {
                        return Ok(Value::Bool(false));
                    }
                }
                Ok(Value::Bool(true))
            }
            BoolOp::Or => {
                for arg in args {
                    if eval_bool(model, vars, states, time, arg)? {
                        return Ok(Value::Bool(true));
                    }
                }
                Ok(Value::Bool(false))
            }
            // Arity validated at model build (exactly one).
            BoolOp::Not => Ok(Value::Bool(!eval_bool(
                model, vars, states, time, &args[0],
            )?)),
        },
        CExpr::Add { args } | CExpr::Mul { args } => {
            let product = matches!(expr, CExpr::Mul { .. });
            let mut acc_i: i64 = if product { 1 } else { 0 };
            let mut acc_f: f64 = if product { 1.0 } else { 0.0 };
            let mut any_float = false;
            for arg in args {
                match eval_num(model, vars, states, time, arg)? {
                    Num::Int(i) => {
                        if product {
                            acc_i *= i;
                            acc_f *= i as f64;
                        } else {
                            acc_i += i;
                            acc_f += i as f64;
                        }
                    }
                    Num::Float(f) => {
                        any_float = true;
                        if product {
                            acc_f *= f;
                        } else {
                            acc_f += f;
                        }
                    }
                }
            }
            if any_float {
                Ok(Value::Float(acc_f))
            } else {
                Ok(Value::Int(acc_i))
            }
        }
        CExpr::Sub { lhs, rhs } => {
            let lhs = eval_num(model, vars, states, time, lhs)?;
            let rhs = eval_num(model, vars, states, time, rhs)?;
            Ok(match (lhs, rhs) {
                (Num::Int(a), Num::Int(b)) => Value::Int(a - b),
                (a, b) => Value::Float(a.as_f64() - b.as_f64()),
            })
        }
        CExpr::Div { lhs, rhs } => {
            let lhs = eval_num(model, vars, states, time, lhs)?.as_f64();
            let rhs = eval_num(model, vars, states, time, rhs)?.as_f64();
            // IEEE semantics (±inf on zero divisor); NaN is caught by
            // comparisons and the integrator's finiteness checks.
            Ok(Value::Float(lhs / rhs))
        }
        CExpr::Min { args } | CExpr::Max { args } => {
            let take_min = matches!(expr, CExpr::Min { .. });
            let mut best: Option<Num> = None;
            for arg in args {
                let value = eval_num(model, vars, states, time, arg)?;
                best = Some(match best {
                    None => value,
                    Some(current) => {
                        let replace = if take_min {
                            value.as_f64() < current.as_f64()
                        } else {
                            value.as_f64() > current.as_f64()
                        };
                        if replace {
                            value
                        } else {
                            current
                        }
                    }
                });
            }
            // Arity ≥ 1 validated at model build.
            Ok(best.map_or(Value::Int(0), Num::into_value))
        }
        CExpr::If {
            cond,
            then,
            otherwise,
        } => {
            if eval_bool(model, vars, states, time, cond)? {
                eval_expr(model, vars, states, time, then)
            } else {
                eval_expr(model, vars, states, time, otherwise)
            }
        }
        CExpr::Sin(arg) => Ok(Value::Float(
            eval_num(model, vars, states, time, arg)?.as_f64().sin(),
        )),
        CExpr::Exp(arg) => Ok(Value::Float(
            eval_num(model, vars, states, time, arg)?.as_f64().exp(),
        )),
        CExpr::Time => Ok(Value::Float(time)),
    }
}

/// Numeric intermediate for arithmetic evaluation.
#[derive(Debug, Clone, Copy)]
enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    fn as_f64(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }
    fn into_value(self) -> Value {
        match self {
            Num::Int(i) => Value::Int(i),
            Num::Float(f) => Value::Float(f),
        }
    }
}

fn eval_num(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
) -> Result<Num, EngineError> {
    match eval_expr(model, vars, states, time, expr)? {
        Value::Int(i) => Ok(Num::Int(i)),
        Value::Float(f) => Ok(Num::Float(f)),
        Value::Bool(_) => Err(EngineError::TypeError {
            time,
            detail: "arithmetic on a boolean value".to_owned(),
        }),
    }
}

fn eval_bool(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
) -> Result<bool, EngineError> {
    match eval_expr(model, vars, states, time, expr)? {
        Value::Bool(b) => Ok(b),
        other => Err(EngineError::TypeError {
            time,
            detail: format!("expected a boolean, got {other:?}"),
        }),
    }
}

fn eval_f64(
    model: &CompiledModel,
    vars: &[Value],
    states: &[StateIdx],
    time: f64,
    expr: &CExpr,
) -> Result<f64, EngineError> {
    Ok(eval_num(model, vars, states, time, expr)?.as_f64())
}

fn eval_agg(
    model: &CompiledModel,
    vars: &[Value],
    time: f64,
    sources: &[VarIdx],
    agg: AggOp,
) -> Result<Value, EngineError> {
    match agg {
        AggOp::Count => Ok(Value::Int(sources.len() as i64)),
        AggOp::Sum => {
            let mut int_sum = 0i64;
            let mut float_sum = 0.0f64;
            let mut any_float = false;
            for &idx in sources {
                match vars[idx] {
                    Value::Int(i) => int_sum += i,
                    Value::Float(f) => {
                        any_float = true;
                        float_sum += f;
                    }
                    Value::Bool(b) => int_sum += i64::from(b),
                }
            }
            if any_float {
                Ok(Value::Float(float_sum + int_sum as f64))
            } else {
                Ok(Value::Int(int_sum))
            }
        }
        AggOp::All | AggOp::Any => {
            let mut all = true;
            let mut any = false;
            for &idx in sources {
                match vars[idx] {
                    Value::Bool(b) => {
                        all &= b;
                        any |= b;
                    }
                    other => {
                        return Err(EngineError::TypeError {
                            time,
                            detail: format!(
                                "boolean aggregation over non-boolean value {other:?} \
                                 (attribute `{}`)",
                                model.var_names[idx]
                            ),
                        });
                    }
                }
            }
            Ok(Value::Bool(if agg == AggOp::All { all } else { any }))
        }
        AggOp::Mean | AggOp::Median => {
            let mut values = Vec::with_capacity(sources.len());
            for &idx in sources {
                values.push(match vars[idx] {
                    Value::Int(i) => i as f64,
                    Value::Float(f) => f,
                    Value::Bool(b) => f64::from(u8::from(b)),
                });
            }
            if values.is_empty() {
                return Ok(Value::Float(0.0));
            }
            if agg == AggOp::Mean {
                let n = values.len() as f64;
                Ok(Value::Float(values.iter().sum::<f64>() / n))
            } else {
                values.sort_by(f64::total_cmp);
                let mid = values.len() / 2;
                Ok(Value::Float(if values.len() % 2 == 1 {
                    values[mid]
                } else {
                    0.5 * (values[mid - 1] + values[mid])
                }))
            }
        }
    }
}

fn eval_cmp(time: f64, op: CmpOp, lhs: Value, rhs: Value) -> Result<Value, EngineError> {
    let ordering = match (lhs, rhs) {
        (Value::Bool(a), Value::Bool(b)) => {
            return match op {
                CmpOp::Eq => Ok(Value::Bool(a == b)),
                CmpOp::Ne => Ok(Value::Bool(a != b)),
                _ => Err(EngineError::TypeError {
                    time,
                    detail: format!("ordering comparison {op:?} on booleans"),
                }),
            };
        }
        (Value::Int(a), Value::Int(b)) => a.partial_cmp(&b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(&b),
        (Value::Int(a), Value::Float(b)) => (a as f64).partial_cmp(&b),
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(b as f64)),
        (a, b) => {
            return Err(EngineError::TypeError {
                time,
                detail: format!("comparison between incompatible kinds {a:?} and {b:?}"),
            });
        }
    };
    let Some(ordering) = ordering else {
        return Err(EngineError::TypeError {
            time,
            detail: "comparison involving NaN".to_owned(),
        });
    };
    let result = match op {
        CmpOp::Eq => ordering.is_eq(),
        CmpOp::Ne => !ordering.is_eq(),
        CmpOp::Lt => ordering.is_lt(),
        CmpOp::Le => ordering.is_le(),
        CmpOp::Gt => ordering.is_gt(),
        CmpOp::Ge => ordering.is_ge(),
    };
    Ok(Value::Bool(result))
}

/// Recompute explicit equations (declaration order) into `vars`.
fn recompute_explicit(
    model: &CompiledModel,
    vars: &mut [Value],
    states: &[StateIdx],
    time: f64,
) -> Result<(), EngineError> {
    for (target, expr) in &model.explicit {
        let value = eval_f64(model, vars, states, time, expr)?;
        vars[*target] = Value::Float(value);
    }
    Ok(())
}

/// Adapter exposing the compiled continuous section to `raichu-numeric`.
/// Errors raised inside the solver callbacks are stashed and re-raised
/// after integration (the trait is infallible by design).
struct ContinuousSystem<'m> {
    model: &'m CompiledModel,
    vars: Vec<Value>,
    states: Vec<StateIdx>,
    /// Active watched transitions monitored this segment.
    margins: Vec<TransIdx>,
    /// Active continuously-varying hazards monitored this segment:
    /// `(transition, remaining threshold E − H)`. Each occupies one
    /// auxiliary state slot after the ODE attributes, integrating
    /// `dH/dt = λ(x)`; the firing is the event `H − (E − H₀) = 0`,
    /// located exactly like a watched boundary crossing (`reschedule_modifiable`
    /// under continuous evolution).
    hazards: Vec<(TransIdx, f64)>,
    error: Option<EngineError>,
}

impl ContinuousSystem<'_> {
    fn load(&mut self, t: f64, y: &[f64]) {
        for (slot, (var, _)) in self.model.ode.iter().enumerate() {
            self.vars[*var] = Value::Float(y[slot]);
        }
        if self.error.is_none() {
            if let Err(error) = recompute_explicit(self.model, &mut self.vars, &self.states, t) {
                self.error = Some(error);
            }
        }
    }
}

impl OdeSystem for ContinuousSystem<'_> {
    fn dim(&self) -> usize {
        self.model.ode.len() + self.hazards.len()
    }

    fn rhs(&mut self, t: f64, y: &[f64], dydt: &mut [f64]) {
        self.load(t, y);
        for (slot, (_, expr)) in self.model.ode.iter().enumerate() {
            match eval_f64(self.model, &self.vars, &self.states, t, expr) {
                Ok(value) => dydt[slot] = value,
                Err(error) => {
                    self.error.get_or_insert(error);
                    dydt[slot] = 0.0;
                }
            }
        }
        let ode_len = self.model.ode.len();
        for (slot, (trans_idx, _)) in self.hazards.iter().enumerate() {
            let CLaw::ExpVar { rate, .. } = &self.model.transitions[*trans_idx].distrib else {
                dydt[ode_len + slot] = 0.0;
                continue;
            };
            match eval_f64(self.model, &self.vars, &self.states, t, rate) {
                Ok(lambda) if lambda.is_finite() && lambda >= 0.0 => {
                    dydt[ode_len + slot] = lambda;
                }
                Ok(lambda) => {
                    self.error.get_or_insert(EngineError::TypeError {
                        time: t,
                        detail: format!(
                            "state-dependent rate of `{}` evaluated to {lambda} \
                             (must be finite and >= 0)",
                            self.model.transitions[*trans_idx].name
                        ),
                    });
                    dydt[ode_len + slot] = 0.0;
                }
                Err(error) => {
                    self.error.get_or_insert(error);
                    dydt[ode_len + slot] = 0.0;
                }
            }
        }
    }

    fn n_events(&self) -> usize {
        self.margins.len() + self.hazards.len()
    }

    fn events(&mut self, t: f64, y: &[f64], out: &mut [f64]) {
        self.load(t, y);
        for (slot, trans_idx) in self.margins.iter().enumerate() {
            let CLaw::Watched { margin } = &self.model.transitions[*trans_idx].distrib else {
                out[slot] = -1.0;
                continue;
            };
            match eval_f64(self.model, &self.vars, &self.states, t, margin) {
                Ok(value) => out[slot] = value,
                Err(error) => {
                    self.error.get_or_insert(error);
                    out[slot] = -1.0;
                }
            }
        }
        let (n_margins, ode_len) = (self.margins.len(), self.model.ode.len());
        for (slot, (_, remaining)) in self.hazards.iter().enumerate() {
            out[n_margins + slot] = y[ode_len + slot] - remaining;
        }
    }
}

/// A simulation engine over a compiled model.
///
/// **Not a singleton**: any number
/// of engines can coexist in one process; every piece of state lives in
/// this struct.
pub struct Engine<'m> {
    model: &'m CompiledModel,
    config: EngineConfig,
    solver: Box<dyn OdeSolver>,
    time: f64,
    vars: Vec<Value>,
    states: Vec<StateIdx>,
    /// Pending firing date per transition (`None` = not scheduled;
    /// watched transitions are monitored, never date-scheduled).
    pending: Vec<Option<f64>>,
    /// Remaining countdown of paused transitions
    /// (`on_interruption: resume` only).
    frozen: Vec<Option<f64>>,
    /// Cumulative-hazard state per transition (`CLaw::ExpVar` only;
    /// survives a `resume` pause, cleared by `reset`/firing/exit).
    hazards: Vec<Option<Hazard>>,
    /// Transitions whose state-dependent rate varies continuously
    /// (monitored during `integrate_to`, like watched boundaries).
    continuous_rates: Vec<TransIdx>,
    events: Vec<Event>,
    journal: Vec<JournalRecord>,
    /// Ordered monitored-state entries recorded this trajectory (sequence
    /// analysis; empty unless `config.sequences`).
    seq_events: Vec<SeqEvent>,
    /// The reached target `(end_cause, end_time)` once one activates — set
    /// once, triggers the trajectory early-stop.
    seq_end: Option<(String, f64)>,
    indicator_series: Vec<IndicatorSeries>,
    sampled: Vec<IndicatorSeries>,
    sample_cursor: usize,
    /// Consecutive watched firings without time advancing (Zeno guard).
    watched_streak: (f64, usize),
    /// Replica generator (master seed + substream; `schedule_stochastic` draws).
    rng: ChaCha8Rng,
    /// Whether the model carries any stochastic distribution (provenance).
    stochastic: bool,
    /// Scratch worklist for the fixpoint (reused across steps — no
    /// allocation in the hot loop once warmed up).
    worklist: BTreeSet<FnIdx>,
}

impl<'m> Engine<'m> {
    /// Build and initialise an engine with the default ODE backend
    /// (Dormand–Prince 4(5), parameters from the config).
    pub fn new(model: &'m CompiledModel, config: EngineConfig) -> Result<Self, EngineError> {
        let solver = Box::new(DormandPrince45::new(config.ode.clone()));
        Self::with_solver(model, config, solver)
    }

    /// Build an engine with an explicit ODE backend (the trait is the
    /// swap point — see `raichu-numeric`).
    pub fn with_solver(
        model: &'m CompiledModel,
        config: EngineConfig,
        solver: Box<dyn OdeSolver>,
    ) -> Result<Self, EngineError> {
        let mut engine = Self::bare(model, config, solver);
        engine.initialize()?;
        Ok(engine)
    }

    /// Rebuild an engine positioned at a previously captured
    /// [`Snapshot`], **skipping** the initialization axiom (the snapshot
    /// already carries a valid, possibly-advanced state).
    ///
    /// This is the seam a stateful facade uses when it cannot hold the
    /// borrowing [`Engine`] across calls (e.g. the Python `interactive`
    /// object): it keeps the owned model + a `Snapshot`, and rebuilds a
    /// throwaway engine on each call. Restores are exact, so a run
    /// driven this way is identical to one driven on a persistent engine.
    pub fn from_snapshot(
        model: &'m CompiledModel,
        config: EngineConfig,
        snapshot: &Snapshot,
    ) -> Self {
        let solver = Box::new(DormandPrince45::new(config.ode.clone()));
        let mut engine = Self::bare(model, config, solver);
        engine.restore(snapshot);
        engine
    }

    /// Construct the engine struct with its pristine pre-initialization
    /// field values (no fixpoint, no schedule yet). Shared by
    /// [`Engine::with_solver`] and [`Engine::from_snapshot`].
    fn bare(model: &'m CompiledModel, config: EngineConfig, solver: Box<dyn OdeSolver>) -> Self {
        let stochastic = model.transitions.iter().any(|t| {
            matches!(
                t.distrib,
                CLaw::Exp(_)
                    | CLaw::ExpVar { .. }
                    | CLaw::Weibull(..)
                    | CLaw::Lognormal(..)
                    | CLaw::Gamma(..)
                    | CLaw::Uniform(..)
                    | CLaw::Empirical(_)
            )
            // A genuinely-branching instantaneous transition (≥ 2 positive
            // branches) draws its destination from the RNG.
            || matches!(&t.distrib, CLaw::Inst(probs)
                if probs.iter().filter(|p| **p > 0.0).count() >= 2)
        });
        let continuous_rates: Vec<TransIdx> = model
            .transitions
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                matches!(
                    t.distrib,
                    CLaw::ExpVar {
                        continuous: true,
                        ..
                    }
                )
            })
            .map(|(i, _)| i)
            .collect();
        let rng = raichu_rng::replica_rng(config.seed, config.rng_stream);
        Engine {
            time: 0.0,
            vars: model.var_init.clone(),
            states: model.automata.iter().map(|a| a.init).collect(),
            pending: vec![None; model.transitions.len()],
            frozen: vec![None; model.transitions.len()],
            hazards: vec![None; model.transitions.len()],
            continuous_rates,
            events: Vec::new(),
            journal: Vec::new(),
            seq_events: Vec::new(),
            seq_end: None,
            indicator_series: model
                .indicators
                .iter()
                .map(|i| IndicatorSeries {
                    name: i.name.clone(),
                    points: Vec::new(),
                })
                .collect(),
            sampled: model
                .indicators
                .iter()
                .map(|i| IndicatorSeries {
                    name: i.name.clone(),
                    points: Vec::new(),
                })
                .collect(),
            sample_cursor: 0,
            watched_streak: (0.0, 0),
            rng,
            stochastic,
            worklist: BTreeSet::new(),
            solver,
            model,
            config,
        }
    }

    /// Initialization axiom (Desgeorges et al. 2021): run every
    /// sensitive function once in declaration order to a fixpoint, solve
    /// the explicit equations, build the initial schedule, and record
    /// the t = 0 indicator/sample values. Shared by [`Engine::new`] and
    /// [`Engine::reset`] so a reset state is identical to a fresh build.
    fn initialize(&mut self) -> Result<(), EngineError> {
        self.worklist.extend(0..self.model.functions.len());
        self.run_fixpoint()?;
        recompute_explicit(self.model, &mut self.vars, &self.states, self.time)?;
        self.refresh_schedule()?;
        self.record_indicators();
        // Sample instants at or before t = 0 use the initial state.
        self.flush_samples_through(0.0);
        // Sequence analysis: a target already active at initialization
        // (declared init state) ends the trajectory at t = 0.
        self.check_targets();
        Ok(())
    }

    /// Sequence analysis: label the trajectory with the first target
    /// (feared event) whose state is active — sets `seq_end` once.
    fn check_targets(&mut self) {
        if !self.config.sequences || self.seq_end.is_some() {
            return;
        }
        for target in &self.model.targets {
            if self.states[target.automaton] == target.state {
                self.seq_end = Some((target.name.clone(), self.time));
                break;
            }
        }
    }

    /// Current simulation time.
    #[must_use]
    pub fn current_time(&self) -> f64 {
        self.time
    }

    /// Value of an attribute by qualified name (`component.attribute`).
    #[must_use]
    pub fn attribute(&self, qualified: &str) -> Option<Value> {
        self.model
            .var_index
            .get(qualified)
            .map(|&idx| self.vars[idx])
    }

    /// Current state name of an automaton by qualified name
    /// (`component.automaton`).
    #[must_use]
    pub fn state(&self, qualified: &str) -> Option<&str> {
        self.model.automaton_index.get(qualified).map(|&idx| {
            let automaton = &self.model.automata[idx];
            automaton.states[self.states[idx]].as_str()
        })
    }

    /// **Interactive control** — every currently-armed transition.
    ///
    /// Lists the date-scheduled transitions (delay / inst / stochastic)
    /// with their firing date, plus the watched transitions armed in
    /// their source state (date = the current instant when their guard
    /// already holds, else `None` — the boundary being located only
    /// during continuous evolution).
    ///
    /// Sorted by date (unlocated watched last), then transition index,
    /// so the first entry is exactly what [`Engine::step`] would fire
    /// next.
    #[must_use]
    pub fn fireable(&self) -> Vec<Fireable> {
        let mut out: Vec<Fireable> = Vec::new();
        for (idx, pending) in self.pending.iter().enumerate() {
            if let Some(date) = *pending {
                out.push(Fireable {
                    index: idx,
                    transition: self.model.transitions[idx].name.clone(),
                    kind: fireable_kind(&self.model.transitions[idx].distrib),
                    date: Some(date),
                });
            }
        }
        for &idx in &self.model.watched {
            let transition = &self.model.transitions[idx];
            if self.states[transition.automaton] != transition.source {
                continue;
            }
            // Guard already true ⇒ fireable at the current instant; else
            // its boundary has not been located yet (date unknown). A
            // guard type error is treated as "not fireable now" here; it
            // resurfaces when the transition is actually stepped/fired.
            let date = self
                .is_immediate_watched(idx)
                .unwrap_or(false)
                .then_some(self.time);
            out.push(Fireable {
                index: idx,
                transition: transition.name.clone(),
                kind: FireableKind::Watched,
                date,
            });
        }
        out.sort_by(|a, b| match (a.date, b.date) {
            (Some(x), Some(y)) => x
                .partial_cmp(&y)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.index.cmp(&b.index)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.index.cmp(&b.index),
        });
        out
    }

    /// **Interactive control** — fire the armed transition carrying this
    /// qualified name (see [`Engine::fire_idx`] for the semantics).
    ///
    /// Errors with [`EngineError::UnknownTransition`] if no transition
    /// bears the name, or [`EngineError::NotFireable`] if it is not armed.
    pub fn fire_named(&mut self, name: &str) -> Result<Event, EngineError> {
        let idx = self.transition_index(name)?;
        self.fire_idx_inner(idx, None)
    }

    /// **Interactive control** — fire the armed transition `name`,
    /// **forcing** its destination branch to the state named `to`
    /// (bypassing the RNG / deterministic-branch resolution). This is
    /// what makes a non-deterministic instantaneous branching (or any
    /// stochastic branch) reproducibly testable: the outcome is chosen,
    /// not drawn.
    ///
    /// Errors with [`EngineError::ForcedTargetInvalid`] if `to` is not
    /// one of the transition's declared target states.
    pub fn fire_named_to(&mut self, name: &str, to: &str) -> Result<Event, EngineError> {
        let idx = self.transition_index(name)?;
        let forced = self.resolve_forced(idx, to)?;
        self.fire_idx_inner(idx, Some(forced))
    }

    /// **Interactive control** — fire a *chosen* armed transition by its
    /// index (the stable handle from [`Engine::fireable`]), resolving the
    /// destination the normal way. See [`Engine::fire_idx_to`] to force
    /// the branch.
    pub fn fire_idx(&mut self, trans_idx: TransIdx) -> Result<Event, EngineError> {
        self.fire_idx_inner(trans_idx, None)
    }

    /// **Interactive control** — fire a chosen armed transition by index,
    /// **forcing** its destination to the state named `to`.
    pub fn fire_idx_to(&mut self, trans_idx: TransIdx, to: &str) -> Result<Event, EngineError> {
        let forced = self.resolve_forced(trans_idx, to)?;
        self.fire_idx_inner(trans_idx, Some(forced))
    }

    /// **Interactive control** — override the scheduled firing date of an
    /// armed transition (manual date-setting). The transition must be
    /// date-scheduled (`pending`, i.e. not a watched boundary), and the
    /// new date must not lie in the past (`>=` the current time).
    ///
    /// The override sticks for delay / inst / fixed-rate transitions
    /// until they fire or leave their source state; a *state-dependent
    /// rate* transition may have its date recomputed at the next
    /// discrete step (`reschedule_modifiable`).
    pub fn set_date(&mut self, name: &str, date: f64) -> Result<(), EngineError> {
        let idx = self.transition_index(name)?;
        self.set_date_idx(idx, date)
    }

    /// **Interactive control** — override an armed transition's firing
    /// date by index (see [`Engine::set_date`]).
    pub fn set_date_idx(&mut self, trans_idx: TransIdx, date: f64) -> Result<(), EngineError> {
        let Some(transition) = self.model.transitions.get(trans_idx) else {
            return Err(EngineError::UnknownTransition {
                transition: format!("<index {trans_idx}>"),
            });
        };
        let name = transition.name.clone();
        if !date.is_finite() || date < self.time {
            return Err(EngineError::DateInPast {
                transition: name,
                date,
                time: self.time,
            });
        }
        match self.pending.get_mut(trans_idx) {
            Some(slot) if slot.is_some() => *slot = Some(date),
            _ => {
                return Err(EngineError::NotFireable {
                    transition: name,
                    time: self.time,
                })
            }
        }
        if self.config.journal {
            self.journal.push(JournalRecord::TransitionRescheduled {
                time: self.time,
                transition: name,
                firing_at: date,
            });
        }
        Ok(())
    }

    /// **Interactive control** — capture the full mutable trajectory
    /// state as an opaque [`Snapshot`] (checkpoint / undo point). Costs
    /// one clone of the state vectors; the immutable model is untouched.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            time: self.time,
            vars: self.vars.clone(),
            states: self.states.clone(),
            pending: self.pending.clone(),
            frozen: self.frozen.clone(),
            hazards: self.hazards.clone(),
            events: self.events.clone(),
            journal: self.journal.clone(),
            seq_events: self.seq_events.clone(),
            seq_end: self.seq_end.clone(),
            indicator_series: self.indicator_series.clone(),
            sampled: self.sampled.clone(),
            sample_cursor: self.sample_cursor,
            watched_streak: self.watched_streak,
            rng: self.rng.clone(),
            worklist: self.worklist.clone(),
        }
    }

    /// **Interactive control** — reinstate a previously captured
    /// [`Snapshot`] (undo). The RNG is restored too, so any continuation
    /// is bit-for-bit identical to continuing from the original point.
    pub fn restore(&mut self, snap: &Snapshot) {
        self.time = snap.time;
        self.vars = snap.vars.clone();
        self.states = snap.states.clone();
        self.pending = snap.pending.clone();
        self.frozen = snap.frozen.clone();
        self.hazards = snap.hazards.clone();
        self.events = snap.events.clone();
        self.journal = snap.journal.clone();
        self.seq_events = snap.seq_events.clone();
        self.seq_end = snap.seq_end.clone();
        self.indicator_series = snap.indicator_series.clone();
        self.sampled = snap.sampled.clone();
        self.sample_cursor = snap.sample_cursor;
        self.watched_streak = snap.watched_streak;
        self.rng = snap.rng.clone();
        self.worklist = snap.worklist.clone();
    }

    /// **Interactive control** — the events fired so far, in
    /// chronological order (the same data a finished [`SimulationResult`]
    /// reports in its `events`).
    #[must_use]
    pub fn history(&self) -> &[Event] {
        &self.events
    }

    /// **Interactive control** — reset the engine to its initial state
    /// (`t = 0`), as freshly built: clears the trajectory and recorded
    /// history, re-seeds the RNG to `(seed, stream)`, and re-runs the
    /// initialization axiom. A run restarted from here is identical to a
    /// fresh [`Engine::new`].
    pub fn reset(&mut self) -> Result<(), EngineError> {
        let n = self.model.transitions.len();
        self.time = 0.0;
        self.vars = self.model.var_init.clone();
        self.states = self.model.automata.iter().map(|a| a.init).collect();
        self.pending = vec![None; n];
        self.frozen = vec![None; n];
        self.hazards = vec![None; n];
        self.events.clear();
        self.journal.clear();
        for series in &mut self.indicator_series {
            series.points.clear();
        }
        for series in &mut self.sampled {
            series.points.clear();
        }
        self.sample_cursor = 0;
        self.watched_streak = (0.0, 0);
        self.rng = raichu_rng::replica_rng(self.config.seed, self.config.rng_stream);
        self.worklist.clear();
        self.initialize()
    }

    /// Fire a *chosen* armed transition (rather than the earliest one, as
    /// [`Engine::step`] does), advancing time to its scheduled date and
    /// running the discrete fixpoint. `forced` overrides the destination
    /// branch when set.
    ///
    /// - A date-scheduled transition (delay / inst / stochastic) fires at
    ///   its `pending` date; with continuous evolution the state is
    ///   integrated up to that date first, and a **watched boundary**
    ///   crossed en route fires *instead* (a forced jump cannot be
    ///   skipped — the returned event is that boundary transition, whose
    ///   branch is never forced).
    /// - A watched transition may be fired only while its guard already
    ///   holds (at the current instant).
    ///
    /// Choosing a non-earliest transition deliberately overrides the
    /// schedule — the interactive counterpart of a manually driven run.
    fn fire_idx_inner(
        &mut self,
        trans_idx: TransIdx,
        forced: Option<StateIdx>,
    ) -> Result<Event, EngineError> {
        let Some(transition) = self.model.transitions.get(trans_idx) else {
            return Err(EngineError::UnknownTransition {
                transition: format!("<index {trans_idx}>"),
            });
        };
        let name = transition.name.clone();
        if let Some(date) = self.pending.get(trans_idx).copied().flatten() {
            if !date.is_finite() {
                return Err(EngineError::NotFireable {
                    transition: name,
                    time: self.time,
                });
            }
            // Advance to the scheduled date, but never move the clock
            // backwards: an *overdue* transition (date already passed
            // because an earlier `fire_idx` skipped ahead) fires at the
            // current instant.
            let t_new = date.max(self.time);
            if self.needs_integration() && t_new > self.time {
                if let Some(watched_idx) = self.integrate_to(t_new)? {
                    self.note_watched_firing()?;
                    return self.fire(watched_idx, None);
                }
            }
            if t_new > self.time {
                self.flush_samples_before(t_new);
            }
            self.time = t_new;
            self.watched_streak = (t_new, 0);
            self.fire(trans_idx, forced)
        } else if self.is_immediate_watched(trans_idx)? {
            self.note_watched_firing()?;
            self.fire(trans_idx, forced)
        } else {
            Err(EngineError::NotFireable {
                transition: name,
                time: self.time,
            })
        }
    }

    /// Resolve a qualified transition name to its index, or
    /// [`EngineError::UnknownTransition`].
    fn transition_index(&self, name: &str) -> Result<TransIdx, EngineError> {
        self.model
            .transitions
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| EngineError::UnknownTransition {
                transition: name.to_owned(),
            })
    }

    /// Resolve a forced destination *state name* to a valid branch of
    /// `trans_idx`, or [`EngineError::ForcedTargetInvalid`] if the name
    /// is unknown or not one of the transition's declared target states.
    fn resolve_forced(&self, trans_idx: TransIdx, to: &str) -> Result<StateIdx, EngineError> {
        let Some(transition) = self.model.transitions.get(trans_idx) else {
            return Err(EngineError::UnknownTransition {
                transition: format!("<index {trans_idx}>"),
            });
        };
        let automaton = &self.model.automata[transition.automaton];
        match automaton.states.iter().position(|s| s == to) {
            Some(state) if transition.targets.contains(&state) => Ok(state),
            _ => Err(EngineError::ForcedTargetInvalid {
                transition: transition.name.clone(),
                state: to.to_owned(),
            }),
        }
    }

    /// Whether `trans_idx` is a watched transition sitting in its source
    /// state with its guard already true — i.e. fireable at the current
    /// instant.
    fn is_immediate_watched(&self, trans_idx: TransIdx) -> Result<bool, EngineError> {
        let transition = &self.model.transitions[trans_idx];
        if !matches!(transition.distrib, CLaw::Watched { .. }) {
            return Ok(false);
        }
        if self.states[transition.automaton] != transition.source {
            return Ok(false);
        }
        let Some(guard) = &transition.guard else {
            return Ok(false);
        };
        eval_bool(self.model, &self.vars, &self.states, self.time, guard)
    }

    /// Fire the next transition — discrete (`fire_transition`) or watched at a
    /// located boundary crossing (`schedule_boundary`) — if one occurs within the
    /// horizon.
    ///
    /// Returns the fired event, or `None` when nothing remains before
    /// `t_max`. Tie-break: earliest date first, then lowest transition
    /// index (documented deterministic order; the converged state does
    /// not depend on it for confluent models).
    pub fn step(&mut self) -> Result<Option<Event>, EngineError> {
        // Watched transition already past its boundary (initial
        // conditions or post-jump state): fires immediately.
        if let Some(trans_idx) = self.immediate_watched()? {
            self.note_watched_firing()?;
            return self.fire(trans_idx, None).map(Some);
        }

        let next_discrete = self.next_pending();
        let t_target =
            next_discrete.map_or(self.config.t_max, |(_, date)| date.min(self.config.t_max));

        if self.needs_integration() && t_target > self.time && t_target.is_finite() {
            if let Some(trans_idx) = self.integrate_to(t_target)? {
                self.note_watched_firing()?;
                return self.fire(trans_idx, None).map(Some);
            }
        }

        match next_discrete {
            Some((trans_idx, date)) if date <= self.config.t_max => {
                // The clock never runs backwards: in a step-driven run
                // every scheduled date is ≥ the current time, so `max`
                // is a no-op; it only guards an *overdue* transition left
                // behind after an interactive `fire_idx` skipped ahead.
                let t_new = date.max(self.time);
                self.flush_samples_before(t_new);
                self.time = t_new;
                self.watched_streak = (t_new, 0);
                self.fire(trans_idx, None).map(Some)
            }
            _ => Ok(None),
        }
    }

    /// Run until the schedule drains or the horizon is reached, then
    /// return the full result with provenance. When sequence recording is
    /// on, a run **early-stops** at the first target (feared event) reached.
    pub fn run(mut self) -> Result<SimulationResult, EngineError> {
        loop {
            if let Some((_, t_hit)) = &self.seq_end {
                // The target is reached: FINISH the hit instant first —
                // fire every transition still due at it, so the latched
                // state is the completed instant, not a half-propagated
                // one (PyCATSHOO completes the step before stopping).
                let t_hit = *t_hit;
                let still_due = self.pending.iter().flatten().any(|d| *d <= t_hit);
                if !still_due {
                    break;
                }
            }
            if self.step()?.is_none() {
                break;
            }
        }
        // Advance the clock (and the continuous state) to the horizon —
        // unless a target early-stopped the trajectory.
        let final_time = if let Some((_, t)) = &self.seq_end {
            *t
        } else if self.config.t_max.is_finite() {
            if self.needs_integration() && self.config.t_max > self.time {
                self.integrate_to(self.config.t_max)?;
            }
            self.config.t_max
        } else {
            self.time
        };
        self.time = final_time;
        // A target-stopped trajectory holds its frozen state through the
        // remaining sample instants (the latch semantics of a
        // target-stopped study: the feared-event state stays active from
        // the hit to the horizon in every sampled measure). With an
        // infinite horizon the latch extends through the last *requested*
        // sample instant, so every replica's series covers the schedule.
        let flush_to = if self.seq_end.is_some() {
            if self.config.t_max.is_finite() {
                self.config.t_max
            } else {
                self.config
                    .samples
                    .last()
                    .copied()
                    .unwrap_or(final_time)
                    .max(final_time)
            }
        } else {
            final_time
        };
        self.flush_samples_through(flush_to);
        let sequence = self.config.sequences.then(|| {
            let (end_cause, end_time) = match self.seq_end.take() {
                Some((cause, t)) => (Some(cause), t),
                None => (None, final_time),
            };
            Sequence {
                events: std::mem::take(&mut self.seq_events),
                end_cause,
                end_time,
                weight: 1.0,
            }
        });
        Ok(SimulationResult {
            events: self.events,
            indicators: self.indicator_series,
            samples: self.sampled,
            journal: self.journal,
            sequence,
            provenance: Provenance {
                engine_version: env!("CARGO_PKG_VERSION").to_owned(),
                model: self.model.name.clone(),
                t_max: self.config.t_max,
                seed: self.stochastic.then_some(self.config.seed),
                ode_rtol: self.config.ode.rtol,
                ode_tol_event: self.config.ode.tol_event,
            },
            final_time,
        })
    }

    // ---- internals ----------------------------------------------------

    /// Fire `trans_idx` at the current time: state change, journal,
    /// discrete evolution to fixpoint, schedule update, indicators.
    ///
    /// `forced` overrides the destination branch (interactive control,
    /// bypassing the RNG / deterministic-branch resolution); `None`
    /// resolves the destination the normal way ([`Engine::resolve_target`]).
    fn fire(
        &mut self,
        trans_idx: TransIdx,
        forced: Option<StateIdx>,
    ) -> Result<Event, EngineError> {
        self.pending[trans_idx] = None;
        self.frozen[trans_idx] = None;
        self.hazards[trans_idx] = None;
        let target = match forced {
            Some(state) => state,
            None => self.resolve_target(trans_idx)?,
        };
        let transition = &self.model.transitions[trans_idx];
        let automaton = &self.model.automata[transition.automaton];
        let event = Event {
            time: self.time,
            transition: transition.name.clone(),
            from: automaton.states[transition.source].clone(),
            to: automaton.states[target].clone(),
        };
        self.states[transition.automaton] = target;
        if self.config.journal {
            self.journal.push(JournalRecord::TransitionFired {
                time: self.time,
                transition: event.transition.clone(),
                from: event.from.clone(),
                to: event.to.clone(),
            });
        }
        self.events.push(event.clone());
        // Sequence analysis: record the entry into a monitored state.
        if self.config.sequences && transition.monitored {
            self.seq_events.push(SeqEvent {
                obj: transition.component.clone(),
                attr: event.to.clone(),
                time: self.time,
                cycle_group: transition.cycle_group.clone(),
            });
        }

        // Discrete evolution: functions sensitive to this automaton.
        self.worklist.extend(
            self.model.state_triggers[transition.automaton]
                .iter()
                .copied(),
        );
        self.run_fixpoint()?;
        recompute_explicit(self.model, &mut self.vars, &self.states, self.time)?;
        self.refresh_schedule()?;
        self.record_indicators();
        // Sequence analysis: the first target (feared event) whose state is
        // now active labels the trajectory's end cause (and ends it — see
        // `run`). States change only through transitions (or the declared
        // init, checked in `initialize`), so this catches every activation.
        self.check_targets();
        Ok(event)
    }

    /// Whether continuous evolution must run: the model has ODE
    /// attributes, or an armed hazard varies continuously (`reschedule_modifiable`
    /// under `integrate_continuous` — possibly with no ODE at all, e.g. a
    /// time-dependent rate).
    fn needs_integration(&self) -> bool {
        !self.model.ode.is_empty()
            || self
                .continuous_rates
                .iter()
                .any(|&idx| self.pending[idx].is_some())
    }

    /// Zeno guard: watched transitions must not keep firing without
    /// time advancing.
    fn note_watched_firing(&mut self) -> Result<(), EngineError> {
        if self.watched_streak.0 == self.time {
            self.watched_streak.1 += 1;
            if self.watched_streak.1 > 1_000 {
                return Err(EngineError::WatchedLoop { time: self.time });
            }
        } else {
            self.watched_streak = (self.time, 1);
        }
        Ok(())
    }

    /// A watched transition whose *guard* already holds while its
    /// automaton sits in the source state fires immediately.
    ///
    /// The guard is evaluated exactly (boolean), not through the
    /// margin: after a located crossing, a sibling transition sharing
    /// the boundary may sit within round-off of it — its strict guard
    /// is already true while its ε-tightened margin is still negative.
    /// Conversely a trajectory *resting* exactly on a strict boundary
    /// keeps a false guard and does not fire (no Zeno).
    fn immediate_watched(&self) -> Result<Option<TransIdx>, EngineError> {
        for &trans_idx in &self.model.watched {
            let transition = &self.model.transitions[trans_idx];
            if self.states[transition.automaton] != transition.source {
                continue;
            }
            let Some(guard) = &transition.guard else {
                continue;
            };
            if eval_bool(self.model, &self.vars, &self.states, self.time, guard)? {
                return Ok(Some(trans_idx));
            }
        }
        Ok(None)
    }

    /// `integrate_continuous`: integrate the continuous state to `t_target`, monitoring
    /// active watched boundaries. Returns the watched transition to fire
    /// if a crossing was located first (time/state already advanced).
    fn integrate_to(&mut self, t_target: f64) -> Result<Option<TransIdx>, EngineError> {
        let margins: Vec<TransIdx> = self
            .model
            .watched
            .iter()
            .copied()
            .filter(|&idx| {
                let transition = &self.model.transitions[idx];
                self.states[transition.automaton] == transition.source
            })
            .collect();

        // Armed continuously-varying hazards ride along as auxiliary
        // state (`dH/dt = λ`), their firing located as an event.
        let hazard_monitors: Vec<(TransIdx, f64)> = self
            .continuous_rates
            .iter()
            .copied()
            .filter(|&idx| self.pending[idx].is_some())
            .filter_map(|idx| {
                self.hazards[idx].map(|hazard| (idx, hazard.threshold - hazard.accumulated))
            })
            .collect();

        let mut y: Vec<f64> = Vec::with_capacity(self.model.ode.len() + hazard_monitors.len());
        for (var, _) in &self.model.ode {
            match self.vars[*var] {
                Value::Float(f) => y.push(f),
                other => {
                    return Err(EngineError::TypeError {
                        time: self.time,
                        detail: format!(
                            "ODE attribute `{}` holds non-float value {other:?}",
                            self.model.var_names[*var]
                        ),
                    });
                }
            }
        }

        y.resize(self.model.ode.len() + hazard_monitors.len(), 0.0);

        let mut system = ContinuousSystem {
            model: self.model,
            vars: self.vars.clone(),
            states: self.states.clone(),
            margins,
            hazards: hazard_monitors,
            error: None,
        };

        // Dense sampling: indicator values recorded from the interpolant.
        let segment_samples: Vec<f64> = self.config.samples[self.sample_cursor..]
            .iter()
            .copied()
            .take_while(|s| *s <= t_target)
            .collect();
        let mut recorded: Vec<(f64, Vec<Value>)> = Vec::new();
        {
            let model = self.model;
            let base_vars = self.vars.clone();
            let states = self.states.clone();
            let mut on_sample = |t: f64, y_at: &[f64]| {
                let mut vars = base_vars.clone();
                for (slot, (var, _)) in model.ode.iter().enumerate() {
                    vars[*var] = Value::Float(y_at[slot]);
                }
                if recompute_explicit(model, &mut vars, &states, t).is_err() {
                    return; // error surfaces through system.error below
                }
                let values = model
                    .indicators
                    .iter()
                    .map(|indicator| match indicator.target {
                        CIndicatorTarget::Var(idx) => vars[idx],
                        CIndicatorTarget::State(aut, state) => {
                            Value::Float(if states[aut] == state { 1.0 } else { 0.0 })
                        }
                    })
                    .collect();
                recorded.push((t, values));
            };

            let outcome = self.solver.integrate(
                &mut system,
                self.time,
                &mut y,
                t_target,
                &segment_samples,
                &mut on_sample,
            )?;

            if let Some(error) = system.error {
                return Err(error);
            }

            // Commit the reached continuous state.
            let (t_reached, fired) = match outcome {
                Outcome::Reached { t } => (t, None),
                Outcome::Event { index, t } => {
                    let fired = if index < system.margins.len() {
                        system.margins[index]
                    } else {
                        system.hazards[index - system.margins.len()].0
                    };
                    (t, Some(fired))
                }
            };
            self.time = t_reached;
            for (slot, (var, _)) in self.model.ode.iter().enumerate() {
                self.vars[*var] = Value::Float(y[slot]);
            }
            // Bank the hazard accrued over this segment (`reschedule_modifiable`
            // bookkeeping; the fired transition's slot, if any, is
            // cleared by `fire`).
            let ode_len = self.model.ode.len();
            for (slot, (trans_idx, _)) in system.hazards.iter().enumerate() {
                if let Some(hazard) = self.hazards[*trans_idx].as_mut() {
                    hazard.accumulated += y[ode_len + slot].max(0.0);
                    hazard.since = t_reached;
                }
            }
            recompute_explicit(self.model, &mut self.vars, &self.states, self.time)?;

            // Commit dense samples (strictly before the reached time —
            // a sample at exactly an event date is recorded post-event
            // by the flush in the next advance).
            for (t, values) in recorded {
                if t < t_reached || fired.is_none() {
                    for (series, value) in self.sampled.iter_mut().zip(values) {
                        series.points.push((t, value));
                    }
                    self.sample_cursor += 1;
                }
            }
            Ok(fired)
        }
    }

    fn next_pending(&self) -> Option<(TransIdx, f64)> {
        let mut best: Option<(TransIdx, f64)> = None;
        for (idx, pending) in self.pending.iter().enumerate() {
            if let Some(date) = pending {
                let better = match best {
                    None => true,
                    Some((_, best_date)) => *date < best_date,
                };
                if better {
                    best = Some((idx, *date));
                }
            }
        }
        best
    }

    fn resolve_target(&mut self, trans_idx: TransIdx) -> Result<StateIdx, EngineError> {
        // Copy the `'m` model reference out so the transition borrow is
        // independent of `&mut self` (frees `self.rng` for the draw).
        let model = self.model;
        let transition = &model.transitions[trans_idx];
        match &transition.distrib {
            CLaw::Delay(_)
            | CLaw::Watched { .. }
            | CLaw::Exp(_)
            | CLaw::ExpVar { .. }
            | CLaw::Weibull(..)
            | CLaw::Lognormal(..)
            | CLaw::Gamma(..)
            | CLaw::Uniform(..)
            | CLaw::Empirical(_) => Ok(transition.targets[0]),
            CLaw::Inst(probs) => {
                // Deterministic fast path: exactly one branch with
                // probability 1 — resolved without touching the RNG, so a
                // deterministic model stays RNG-free and bit-identical on
                // replay.
                if let Some(branch) = probs
                    .iter()
                    .position(|p| (*p - 1.0).abs() <= f64::EPSILON)
                    .filter(|_| probs.iter().filter(|p| **p > 0.0).count() == 1)
                {
                    return Ok(transition.targets[branch]);
                }
                // Stochastic instantaneous branching (`schedule_stochastic`
                // realised on demand): draw the destination from the
                // categorical distribution over `probs` (Σ = 1, validated at
                // model build) by inverse-CDF on one uniform. The draw
                // happens at fire time in the deterministic firing order, so
                // replay stays bit-identical for a fixed (seed, stream).
                let u: f64 = rand::Rng::random(&mut self.rng);
                let mut cumulative = 0.0;
                for (branch, p) in probs.iter().enumerate() {
                    cumulative += *p;
                    if u < cumulative {
                        return Ok(transition.targets[branch]);
                    }
                }
                // `u` within rounding of 1.0: the last branch.
                Ok(transition.targets[probs.len() - 1])
            }
        }
    }

    /// `propagate_effects`: propagate sensitive functions to a fixpoint in the
    /// documented deterministic order (ascending function index).
    fn run_fixpoint(&mut self) -> Result<(), EngineError> {
        if self.config.confluence_check {
            self.confluence_probe()
        } else {
            self.converge(false)
        }
    }

    /// Apply one function's effects; when `trigger` is set, attribute
    /// changes enqueue their dependent functions.
    fn apply_function(&mut self, fn_idx: FnIdx, trigger: bool) -> Result<(), EngineError> {
        let function = &self.model.functions[fn_idx];
        for (target, value_expr) in &function.effects {
            let new = eval_expr(self.model, &self.vars, &self.states, self.time, value_expr)?;
            let old = self.vars[*target];
            if old != new {
                self.vars[*target] = new;
                if self.config.journal {
                    self.journal.push(JournalRecord::AttributeChanged {
                        time: self.time,
                        attribute: self.model.var_names[*target].clone(),
                        old,
                        new,
                        cause: function.name.clone(),
                    });
                }
                if trigger {
                    self.worklist
                        .extend(self.model.var_triggers[*target].iter().copied());
                }
            }
        }
        Ok(())
    }

    /// Non-confluence diagnostic: converge a *copy* of the state with
    /// the worklist processed in reverse order and compare. Divergence
    /// means the model's result depends on evaluation order — reported
    /// as a typed error (rather than silently picking an arbitrary order).
    fn confluence_probe(&mut self) -> Result<(), EngineError> {
        let saved_vars = self.vars.clone();
        let saved_worklist = self.worklist.clone();

        // Canonical forward pass (journaled normally).
        self.converge(false)?;
        let forward_vars = std::mem::replace(&mut self.vars, saved_vars);

        // Silent reverse pass on the saved state.
        self.worklist = saved_worklist;
        let journal_flag = std::mem::replace(&mut self.config.journal, false);
        let reverse = self.converge(true);
        self.config.journal = journal_flag;
        reverse?;

        if forward_vars != self.vars {
            // Name the first diverging attribute's writers for the
            // diagnostic message.
            let diverging = forward_vars
                .iter()
                .zip(&self.vars)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            let mut writers = self
                .model
                .functions
                .iter()
                .filter(|f| f.effects.iter().any(|(target, _)| *target == diverging))
                .map(|f| f.name.clone());
            let first = writers.next().unwrap_or_else(|| "<unknown>".to_owned());
            let second = writers.next().unwrap_or_else(|| first.clone());
            return Err(EngineError::NonConfluent {
                time: self.time,
                first,
                second,
            });
        }
        // Both orders agree: keep the forward result as canonical.
        self.vars = forward_vars;
        Ok(())
    }

    /// Converge the current worklist, ascending or descending order.
    fn converge(&mut self, reverse: bool) -> Result<(), EngineError> {
        let mut iterations = 0usize;
        loop {
            let next = if reverse {
                self.worklist.pop_last()
            } else {
                self.worklist.pop_first()
            };
            let Some(fn_idx) = next else { break };
            iterations += 1;
            if iterations > self.config.max_fixpoint_iterations {
                self.worklist.clear();
                return Err(EngineError::InstantaneousLoop {
                    time: self.time,
                    iterations: self.config.max_fixpoint_iterations,
                });
            }
            if self.config.journal {
                self.journal.push(JournalRecord::FunctionTriggered {
                    time: self.time,
                    function: self.model.functions[fn_idx].name.clone(),
                });
            }
            self.apply_function(fn_idx, true)?;
        }
        Ok(())
    }

    /// `schedule_deterministic` + `drop_disabled`: (re)schedule fireable transitions and drop
    /// stale ones. Deterministic full scan (fine at fixture model
    /// sizes; the scan is index-ordered so the schedule is
    /// reproducible). Watched transitions are never date-scheduled.
    /// Evaluate a state-dependent rate λ(x) on the current state and
    /// reject non-finite or negative values with a typed error.
    fn eval_rate(&self, trans_idx: TransIdx, rate: &CExpr) -> Result<f64, EngineError> {
        let lambda = eval_f64(self.model, &self.vars, &self.states, self.time, rate)?;
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(EngineError::TypeError {
                time: self.time,
                detail: format!(
                    "state-dependent rate of `{}` evaluated to {lambda} \
                     (must be finite and >= 0)",
                    self.model.transitions[trans_idx].name
                ),
            });
        }
        Ok(lambda)
    }

    fn refresh_schedule(&mut self) -> Result<(), EngineError> {
        for trans_idx in 0..self.model.transitions.len() {
            let transition = &self.model.transitions[trans_idx];
            if matches!(transition.distrib, CLaw::Watched { .. }) {
                continue;
            }
            let in_source = self.states[transition.automaton] == transition.source;
            if !in_source {
                // Leaving the source cancels any paused countdown and
                // discards the banked hazard.
                self.frozen[trans_idx] = None;
                self.hazards[trans_idx] = None;
            }
            let guard_ok = match &transition.guard {
                None => true,
                Some(guard) => eval_bool(self.model, &self.vars, &self.states, self.time, guard)?,
            };
            match self.pending[trans_idx] {
                Some(_) if !in_source => {
                    self.pending[trans_idx] = None;
                    if self.config.journal {
                        self.journal.push(JournalRecord::TransitionDropped {
                            time: self.time,
                            transition: transition.name.clone(),
                            reason: DropReason::SourceLeft,
                        });
                    }
                }
                Some(date)
                    if !guard_ok
                        && transition.on_interruption
                            != raichu_model::InterruptionPolicy::Continue =>
                {
                    // `drop_disabled`: reset cancels the occurrence duration
                    // (interruptible transition); resume
                    // pauses it (RAICHU extension); continue never
                    // reaches this arm.
                    let reason = match transition.on_interruption {
                        raichu_model::InterruptionPolicy::Resume => {
                            if let Some(hazard) = self.hazards[trans_idx].as_mut() {
                                // Pause the hazard clock: bank what has
                                // accrued (continuous hazards are already
                                // committed by the integrator); the
                                // re-arm recomputes λ at resumption.
                                if !matches!(
                                    transition.distrib,
                                    CLaw::ExpVar {
                                        continuous: true,
                                        ..
                                    }
                                ) {
                                    hazard.accumulated += hazard.rate * (self.time - hazard.since);
                                }
                                hazard.since = self.time;
                            } else {
                                self.frozen[trans_idx] = Some(date - self.time);
                            }
                            DropReason::GuardPaused
                        }
                        _ => {
                            self.hazards[trans_idx] = None;
                            DropReason::GuardFalse
                        }
                    };
                    self.pending[trans_idx] = None;
                    if self.config.journal {
                        self.journal.push(JournalRecord::TransitionDropped {
                            time: self.time,
                            transition: transition.name.clone(),
                            reason,
                        });
                    }
                }
                #[allow(clippy::float_cmp)] // λ is re-evaluated exactly
                Some(previous) if in_source => {
                    // `reschedule_modifiable`: a pending state-dependent rate whose
                    // inputs changed at this discrete step is
                    // rescheduled against the same `Exp(1)` threshold
                    // (reached with the guard still true, or with the
                    // `continue` policy riding through a false guard).
                    // Continuously-varying rates need no rescheduling:
                    // their hazard is integrated by `integrate_continuous` directly.
                    let CLaw::ExpVar {
                        rate,
                        continuous: false,
                    } = &self.model.transitions[trans_idx].distrib
                    else {
                        continue;
                    };
                    let lambda = self.eval_rate(trans_idx, rate)?;
                    let Some(hazard) = self.hazards[trans_idx].as_mut() else {
                        continue;
                    };
                    if lambda != hazard.rate {
                        hazard.accumulated += hazard.rate * (self.time - hazard.since);
                        hazard.since = self.time;
                        hazard.rate = lambda;
                        let firing_at = if lambda > 0.0 {
                            self.time + (hazard.threshold - hazard.accumulated) / lambda
                        } else {
                            f64::INFINITY
                        };
                        self.pending[trans_idx] = Some(firing_at);
                        if self.config.journal && firing_at != previous {
                            self.journal.push(JournalRecord::TransitionRescheduled {
                                time: self.time,
                                transition: self.model.transitions[trans_idx].name.clone(),
                                firing_at,
                            });
                        }
                    }
                }
                None if in_source && guard_ok => {
                    // `schedule_stochastic` for a state-dependent rate: draw the
                    // `Exp(1)` threshold (fresh arming) or keep the
                    // banked hazard (resume re-arm), then schedule
                    // against the current λ — `+∞` while λ = 0 or while
                    // the hazard is integrated continuously (`integrate_continuous`
                    // locates the firing like a boundary crossing).
                    if let CLaw::ExpVar { rate, continuous } =
                        &self.model.transitions[trans_idx].distrib
                    {
                        let lambda = self.eval_rate(trans_idx, rate)?;
                        let mut hazard = match self.hazards[trans_idx] {
                            Some(banked) => banked,
                            None => Hazard {
                                threshold: rand_distr::Exp1.sample(&mut self.rng),
                                accumulated: 0.0,
                                rate: lambda,
                                since: self.time,
                            },
                        };
                        hazard.rate = lambda;
                        hazard.since = self.time;
                        let firing_at = if *continuous || lambda <= 0.0 {
                            f64::INFINITY
                        } else {
                            self.time + (hazard.threshold - hazard.accumulated) / lambda
                        };
                        self.hazards[trans_idx] = Some(hazard);
                        self.pending[trans_idx] = Some(firing_at);
                        if self.config.journal {
                            self.journal.push(JournalRecord::TransitionScheduled {
                                time: self.time,
                                transition: self.model.transitions[trans_idx].name.clone(),
                                firing_at,
                            });
                        }
                        continue;
                    }
                    // A paused countdown resumes where it stopped.
                    if let Some(remaining) = self.frozen[trans_idx].take() {
                        let firing_at = self.time + remaining;
                        self.pending[trans_idx] = Some(firing_at);
                        if self.config.journal {
                            self.journal.push(JournalRecord::TransitionScheduled {
                                time: self.time,
                                transition: self.model.transitions[trans_idx].name.clone(),
                                firing_at,
                            });
                        }
                        continue;
                    }
                    // `schedule_stochastic`: stochastic firing dates are sampled at
                    // source-state entry. Draws happen here, in
                    // transition-index order — replay is bit-identical
                    // for a fixed (seed, stream).
                    let time_now = self.time;
                    let bad_law = move |detail: String| EngineError::TypeError {
                        time: time_now,
                        detail,
                    };
                    let firing_at = match &self.model.transitions[trans_idx].distrib {
                        CLaw::Delay(delay) => self.time + delay,
                        CLaw::Inst(_) => self.time,
                        CLaw::Watched { .. } => continue,
                        // Armed by the dedicated block above.
                        CLaw::ExpVar { .. } => continue,
                        CLaw::Exp(rate) => {
                            let distribution = rand_distr::Exp::new(*rate)
                                .map_err(|e| bad_law(format!("exp({rate}): {e}")))?;
                            self.time + distribution.sample(&mut self.rng)
                        }
                        CLaw::Weibull(shape, scale) => {
                            let distribution = rand_distr::Weibull::new(*scale, *shape)
                                .map_err(|e| bad_law(format!("weibull: {e}")))?;
                            self.time + distribution.sample(&mut self.rng)
                        }
                        CLaw::Lognormal(mu, sigma) => {
                            let distribution = rand_distr::LogNormal::new(*mu, *sigma)
                                .map_err(|e| bad_law(format!("lognormal: {e}")))?;
                            self.time + distribution.sample(&mut self.rng)
                        }
                        CLaw::Gamma(shape, scale) => {
                            let distribution = rand_distr::Gamma::new(*shape, *scale)
                                .map_err(|e| bad_law(format!("gamma: {e}")))?;
                            self.time + distribution.sample(&mut self.rng)
                        }
                        CLaw::Uniform(low, high) => {
                            let distribution = rand_distr::Uniform::new(*low, *high)
                                .map_err(|e| bad_law(format!("uniform: {e}")))?;
                            self.time + distribution.sample(&mut self.rng)
                        }
                        CLaw::Empirical(points) => {
                            let u: f64 = rand::Rng::random(&mut self.rng);
                            self.time + sample_empirical(points, u)
                        }
                    };
                    self.pending[trans_idx] = Some(firing_at);
                    if self.config.journal {
                        self.journal.push(JournalRecord::TransitionScheduled {
                            time: self.time,
                            transition: self.model.transitions[trans_idx].name.clone(),
                            firing_at,
                        });
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn record_indicators(&mut self) {
        for (indicator, series) in self
            .model
            .indicators
            .iter()
            .zip(self.indicator_series.iter_mut())
        {
            let value = match indicator.target {
                CIndicatorTarget::Var(idx) => self.vars[idx],
                CIndicatorTarget::State(aut, state) => {
                    Value::Float(if self.states[aut] == state { 1.0 } else { 0.0 })
                }
            };
            let changed = series.points.last().is_none_or(|(_, last)| *last != value);
            if changed {
                series.points.push((self.time, value));
            }
        }
    }

    /// Record pending sample instants strictly before `t` with the
    /// *current* (pre-jump) state — piecewise-constant hold for the
    /// discrete-only case.
    fn flush_samples_before(&mut self, t: f64) {
        self.flush_samples(t, false);
    }

    /// Record pending sample instants up to and including `t` (end of
    /// run).
    fn flush_samples_through(&mut self, t: f64) {
        self.flush_samples(t, true);
    }

    fn flush_samples(&mut self, t: f64, inclusive: bool) {
        while self.sample_cursor < self.config.samples.len() {
            let s = self.config.samples[self.sample_cursor];
            let due = if inclusive { s <= t } else { s < t };
            if !due {
                break;
            }
            for (indicator, series) in self.model.indicators.iter().zip(self.sampled.iter_mut()) {
                let value = match indicator.target {
                    CIndicatorTarget::Var(idx) => self.vars[idx],
                    CIndicatorTarget::State(aut, state) => {
                        Value::Float(if self.states[aut] == state { 1.0 } else { 0.0 })
                    }
                };
                series.points.push((s, value));
            }
            self.sample_cursor += 1;
        }
    }
}

/// Classify a compiled occurrence law for interactive inspection
/// ([`Engine::fireable`]).
fn fireable_kind(distrib: &CLaw) -> FireableKind {
    match distrib {
        CLaw::Delay(_) => FireableKind::Delay,
        CLaw::Inst(_) => FireableKind::Inst,
        CLaw::Watched { .. } => FireableKind::Watched,
        CLaw::Exp(_)
        | CLaw::ExpVar { .. }
        | CLaw::Weibull(..)
        | CLaw::Lognormal(..)
        | CLaw::Gamma(..)
        | CLaw::Uniform(..)
        | CLaw::Empirical(_) => FireableKind::Stochastic,
    }
}

/// Inverse-CDF sampling from a validated empirical table: `u` below the
/// first cumulative probability maps to the first time (probability
/// mass); between points the CDF is linearly interpolated.
fn sample_empirical(points: &[(f64, f64)], u: f64) -> f64 {
    let (first_t, first_c) = points[0];
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
    // u ≤ 1 and the validated table ends at cumulative 1.
    points[points.len() - 1].0
}
