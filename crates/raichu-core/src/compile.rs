//! Name→index resolution: turns a validated [`Model`] into dense tables
//! the engine consumes.
//!
//! Performance contract: all names are resolved here, once, at
//! build time — the simulation hot path only touches vector indices,
//! never string lookups, and never allocates.

use raichu_expr::{AggOp, BoolOp, CmpOp, Expr, Value};
use raichu_model::{
    Distrib, EquationKind, IndicatorTarget, InterruptionPolicy, Model, ModelError, PortDir,
};
use std::collections::{BTreeSet, HashMap};
use thiserror::Error;

/// Margin tightening applied to *strict* watched comparisons (`<`,
/// `>`): the engine fires at margin ≥ 0, so a strict boundary is
/// shifted inward by this amount — a trajectory resting exactly on it
/// does not fire, and a genuine crossing date shifts by ε/slope. Must
/// sit *below* the event-location tolerance (`tol_event` = 1e-10):
/// when two watched guards share one crossing, the located state
/// overshoots by ~tol_event and the sibling must still read a
/// non-negative margin to fire immediately.
pub const STRICT_MARGIN_EPS: f64 = 1e-12;

/// Dense index of an attribute in the engine state vector.
pub type VarIdx = usize;
/// Dense index of an automaton.
pub type AutIdx = usize;
/// Index of a state *within its automaton*.
pub type StateIdx = usize;
/// Dense index of a transition (global).
pub type TransIdx = usize;
/// Dense index of a sensitive function (global).
pub type FnIdx = usize;

/// Errors raised while compiling a model.
#[derive(Debug, Error)]
pub enum CompileError {
    /// The model failed its structural validation.
    #[error(transparent)]
    Invalid(#[from] ModelError),
    /// Internal resolution failure — indicates a validator/compiler
    /// mismatch, reported as a typed error rather than a panic.
    #[error("internal resolution failure: {what} `{name}` not found")]
    Unresolved {
        /// Kind of entity that failed to resolve.
        what: &'static str,
        /// Qualified name.
        name: String,
    },
}

/// A compiled expression: same shape as [`Expr`] but with every
/// reference resolved to dense indices.
#[derive(Debug, Clone)]
pub enum CExpr {
    /// Literal constant.
    Const(Value),
    /// Read the attribute at this index.
    Var(VarIdx),
    /// True while `automaton` is in `state`.
    StateActive {
        /// Automaton index.
        automaton: AutIdx,
        /// State index within the automaton.
        state: StateIdx,
    },
    /// Aggregate the out-attributes connected to an in-port.
    PortAgg {
        /// Indices of the connected out-port attributes (connection
        /// declaration order — deterministic).
        sources: Vec<VarIdx>,
        /// Aggregation operator.
        agg: AggOp,
    },
    /// Comparison.
    Cmp {
        /// Operator.
        op: CmpOp,
        /// Left operand.
        lhs: Box<CExpr>,
        /// Right operand.
        rhs: Box<CExpr>,
    },
    /// Boolean connective.
    Bool {
        /// Operator.
        op: BoolOp,
        /// Operands.
        args: Vec<CExpr>,
    },
    /// N-ary sum.
    Add {
        /// Operands.
        args: Vec<CExpr>,
    },
    /// Binary subtraction.
    Sub {
        /// Minuend.
        lhs: Box<CExpr>,
        /// Subtrahend.
        rhs: Box<CExpr>,
    },
    /// N-ary product.
    Mul {
        /// Operands.
        args: Vec<CExpr>,
    },
    /// Binary division (float semantics).
    Div {
        /// Dividend.
        lhs: Box<CExpr>,
        /// Divisor.
        rhs: Box<CExpr>,
    },
    /// N-ary minimum.
    Min {
        /// Operands.
        args: Vec<CExpr>,
    },
    /// N-ary maximum.
    Max {
        /// Operands.
        args: Vec<CExpr>,
    },
    /// Conditional.
    If {
        /// Boolean condition.
        cond: Box<CExpr>,
        /// Value when true.
        then: Box<CExpr>,
        /// Value when false.
        otherwise: Box<CExpr>,
    },
    /// Sine.
    Sin(Box<CExpr>),
    /// Natural exponential.
    Exp(Box<CExpr>),
    /// Current simulation time.
    Time,
}

/// Compiled occurrence distribution.
#[derive(Debug, Clone)]
pub enum CLaw {
    /// Deterministic delay.
    Delay(f64),
    /// Instantaneous branching; probabilities include the reconstructed
    /// complement (length == number of targets).
    Inst(Vec<f64>),
    /// Watched transition (M1): fires when `margin` crosses from
    /// negative to non-negative during continuous evolution (`schedule_boundary`).
    /// The margin is the signed boundary distance derived from the
    /// guard comparison — guard true ⇔ margin ≥ 0.
    Watched {
        /// Signed boundary margin.
        margin: CExpr,
    },
    /// Exponential distribution (M2, `schedule_stochastic`): firing date sampled at
    /// source-state entry.
    Exp(f64),
    /// Exponential distribution with a state-dependent rate λ(x) (`reschedule_modifiable`):
    /// realised by a cumulative hazard against an `Exp(1)` threshold
    /// (`P(T > t) = exp(−∫λ)` — the PDMP survival function, exactly).
    ExpVar {
        /// Rate expression λ(x) ≥ 0.
        rate: CExpr,
        /// Whether λ varies during continuous evolution (depends —
        /// transitively through explicit equations — on an
        /// ODE-integrated attribute or on time). If so the hazard is
        /// integrated alongside the continuous state and the firing
        /// time located like a boundary crossing; otherwise λ is
        /// piecewise-constant and the firing date is rescheduled at
        /// each discrete change (`reschedule_modifiable` proper).
        continuous: bool,
    },
    /// Weibull distribution (M4): shape k, scale λ.
    Weibull(f64, f64),
    /// Log-normal distribution (M4): μ, σ of the underlying normal.
    Lognormal(f64, f64),
    /// Gamma distribution (M4): shape k, scale θ.
    Gamma(f64, f64),
    /// Uniform distribution (M4): [low, high).
    Uniform(f64, f64),
    /// Empirical inverse-CDF table (M4): (time, cumulative prob).
    Empirical(Vec<(f64, f64)>),
}

/// A compiled transition.
#[derive(Debug, Clone)]
pub struct CTransition {
    /// Qualified name `component.automaton.transition` (journal only).
    pub name: String,
    /// Owning automaton.
    pub automaton: AutIdx,
    /// Source state.
    pub source: StateIdx,
    /// Guard (absent = always true).
    pub guard: Option<CExpr>,
    /// Target states (one per branch).
    pub targets: Vec<StateIdx>,
    /// What happens to a pending countdown when the guard turns false
    /// (paper rule `drop_disabled`).
    pub on_interruption: InterruptionPolicy,
    /// Occurrence distribution.
    pub distrib: CLaw,
}

/// A compiled automaton.
#[derive(Debug, Clone)]
pub struct CAutomaton {
    /// Qualified name `component.automaton`.
    pub name: String,
    /// State names, indexed by [`StateIdx`].
    pub states: Vec<String>,
    /// Initial state.
    pub init: StateIdx,
    /// Indices of the transitions owned by this automaton.
    pub transitions: Vec<TransIdx>,
}

/// A compiled sensitive function.
#[derive(Debug, Clone)]
pub struct CFunction {
    /// Qualified name `component.function`.
    pub name: String,
    /// Ordered effects `target := value`.
    pub effects: Vec<(VarIdx, CExpr)>,
}

/// A compiled indicator.
#[derive(Debug, Clone)]
pub struct CIndicator {
    /// Indicator name.
    pub name: String,
    /// What it observes.
    pub target: CIndicatorTarget,
}

/// Compiled indicator target.
#[derive(Debug, Clone)]
pub enum CIndicatorTarget {
    /// A attribute's value.
    Var(VarIdx),
    /// 1.0 while the automaton is in the state, else 0.0.
    State(AutIdx, StateIdx),
}

/// A validated model resolved to dense tables.
#[derive(Debug, Clone)]
pub struct CompiledModel {
    /// Model name (provenance).
    pub name: String,
    /// Qualified attribute names `component.attribute` (journal, results).
    pub var_names: Vec<String>,
    /// Initial attribute values.
    pub var_init: Vec<Value>,
    /// Automata.
    pub automata: Vec<CAutomaton>,
    /// All transitions (global order = declaration order).
    pub transitions: Vec<CTransition>,
    /// All sensitive functions (global order = declaration order — this
    /// *is* the documented deterministic fixpoint order).
    pub functions: Vec<CFunction>,
    /// var index → functions to re-evaluate when it changes.
    pub var_triggers: Vec<Vec<FnIdx>>,
    /// automaton index → functions to re-evaluate when its state changes.
    pub state_triggers: Vec<Vec<FnIdx>>,
    /// Indicators.
    pub indicators: Vec<CIndicator>,
    /// ODE attributes and right-hand sides, declaration order (CEvol).
    pub ode: Vec<(VarIdx, CExpr)>,
    /// Explicit equations, declaration order (solved before ODE
    /// right-hand sides at every evaluation point).
    pub explicit: Vec<(VarIdx, CExpr)>,
    /// Indices of watched transitions (monitored during continuous
    /// evolution, never date-scheduled).
    pub watched: Vec<TransIdx>,
    /// Lookup: qualified attribute name → index (API convenience).
    pub var_index: HashMap<String, VarIdx>,
    /// Lookup: qualified automaton name → index (API convenience).
    pub automaton_index: HashMap<String, AutIdx>,
}

struct Resolver {
    vars: HashMap<(String, String), VarIdx>,
    states: HashMap<(String, String, String), (AutIdx, StateIdx)>,
    automata: HashMap<(String, String), AutIdx>,
    /// (component, in-port) → connected source attribute indices.
    port_sources: HashMap<(String, String), Vec<VarIdx>>,
}

impl Resolver {
    fn var(&self, component: &str, attribute: &str) -> Result<VarIdx, CompileError> {
        self.vars
            .get(&(component.to_owned(), attribute.to_owned()))
            .copied()
            .ok_or_else(|| CompileError::Unresolved {
                what: "attribute",
                name: format!("{component}.{attribute}"),
            })
    }

    fn state(
        &self,
        component: &str,
        automaton: &str,
        state: &str,
    ) -> Result<(AutIdx, StateIdx), CompileError> {
        self.states
            .get(&(component.to_owned(), automaton.to_owned(), state.to_owned()))
            .copied()
            .ok_or_else(|| CompileError::Unresolved {
                what: "state",
                name: format!("{component}.{automaton}.{state}"),
            })
    }

    fn port(&self, component: &str, port: &str) -> Vec<VarIdx> {
        // An in-port with no connection aggregates over the empty set —
        // legal (muscadet relies on no-connection defaults).
        self.port_sources
            .get(&(component.to_owned(), port.to_owned()))
            .cloned()
            .unwrap_or_default()
    }

    fn compile_expr(&self, expr: &Expr) -> Result<CExpr, CompileError> {
        Ok(match expr {
            Expr::Const { value } => CExpr::Const(*value),
            Expr::Attr { attr } => CExpr::Var(self.var(&attr.component, &attr.attribute)?),
            Expr::StateActive { state } => {
                let (automaton, state) =
                    self.state(&state.component, &state.automaton, &state.state)?;
                CExpr::StateActive { automaton, state }
            }
            Expr::PortAgg { port, agg } => CExpr::PortAgg {
                sources: self.port(&port.component, &port.port),
                agg: *agg,
            },
            Expr::Cmp { cmp, lhs, rhs } => CExpr::Cmp {
                op: *cmp,
                lhs: Box::new(self.compile_expr(lhs)?),
                rhs: Box::new(self.compile_expr(rhs)?),
            },
            Expr::Bool { bool_op, args } => CExpr::Bool {
                op: *bool_op,
                args: self.compile_args(args)?,
            },
            Expr::Add { args } => CExpr::Add {
                args: self.compile_args(args)?,
            },
            Expr::Mul { args } => CExpr::Mul {
                args: self.compile_args(args)?,
            },
            Expr::Min { args } => CExpr::Min {
                args: self.compile_args(args)?,
            },
            Expr::Max { args } => CExpr::Max {
                args: self.compile_args(args)?,
            },
            Expr::Sub { lhs, rhs } => CExpr::Sub {
                lhs: Box::new(self.compile_expr(lhs)?),
                rhs: Box::new(self.compile_expr(rhs)?),
            },
            Expr::Div { lhs, rhs } => CExpr::Div {
                lhs: Box::new(self.compile_expr(lhs)?),
                rhs: Box::new(self.compile_expr(rhs)?),
            },
            Expr::If {
                cond,
                then,
                otherwise,
            } => CExpr::If {
                cond: Box::new(self.compile_expr(cond)?),
                then: Box::new(self.compile_expr(then)?),
                otherwise: Box::new(self.compile_expr(otherwise)?),
            },
            Expr::Sin { arg } => CExpr::Sin(Box::new(self.compile_expr(arg)?)),
            Expr::Exp { arg } => CExpr::Exp(Box::new(self.compile_expr(arg)?)),
            Expr::Time => CExpr::Time,
        })
    }

    fn compile_args(&self, args: &[Expr]) -> Result<Vec<CExpr>, CompileError> {
        args.iter().map(|a| self.compile_expr(a)).collect()
    }

    /// Signed boundary margin of a watched guard (guard true ⇔ margin
    /// ≥ 0). Validation guarantees one of three shapes:
    ///
    /// - a single ordering comparison → `lhs − rhs` (or reversed);
    /// - `and(gates…, cmp)` → `if(and(gates), margin(cmp), −1)`;
    /// - `or(gates…, cmp)`  → `if(or(gates), +1, margin(cmp))`.
    ///
    /// The gate expressions are *discrete* (they only change at
    /// discrete events), so the margin stays continuous within every
    /// integration segment; a gate flip is caught by the
    /// immediate-watched check right after the discrete fixpoint.
    fn compile_watched_margin(&self, guard: &Expr) -> Result<CExpr, CompileError> {
        match guard {
            Expr::Cmp {
                cmp: cmp @ (CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge),
                lhs,
                rhs,
            } => {
                let left = self.compile_expr(lhs)?;
                let right = self.compile_expr(rhs)?;
                let raw = match cmp {
                    CmpOp::Ge | CmpOp::Gt => CExpr::Sub {
                        lhs: Box::new(left),
                        rhs: Box::new(right),
                    },
                    _ => CExpr::Sub {
                        lhs: Box::new(right),
                        rhs: Box::new(left),
                    },
                };
                // The engine fires at margin ≥ 0. For *strict*
                // comparisons a trajectory resting exactly on the
                // boundary (e.g. a ternary signal at 0 against a
                // `< 0` guard) must NOT fire: tighten the margin by
                // STRICT_MARGIN_EPS. The induced crossing-date shift
                // (ε / slope) sits below the documented
                // event tolerances (~1e-9).
                Ok(match cmp {
                    CmpOp::Gt | CmpOp::Lt => CExpr::Sub {
                        lhs: Box::new(raw),
                        rhs: Box::new(CExpr::Const(Value::Float(STRICT_MARGIN_EPS))),
                    },
                    _ => raw,
                })
            }
            // AND: every boundary must hold — the binding one is the
            // *minimum* margin. OR: any suffices — the maximum.
            Expr::Bool {
                bool_op: bool_op @ (BoolOp::And | BoolOp::Or),
                args,
            } => {
                let margins = args
                    .iter()
                    .map(|arg| self.compile_watched_margin(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(match bool_op {
                    BoolOp::And => CExpr::Min { args: margins },
                    _ => CExpr::Max { args: margins },
                })
            }
            // NOT: the guard flips exactly where the margin changes
            // sign — negate it.
            Expr::Bool {
                bool_op: BoolOp::Not,
                args,
            } if args.len() == 1 => {
                let inner = self.compile_watched_margin(&args[0])?;
                Ok(CExpr::Sub {
                    lhs: Box::new(CExpr::Const(Value::Float(0.0))),
                    rhs: Box::new(inner),
                })
            }
            // Any other boolean operand is a *discrete gate*: constant
            // between discrete events, mapped to ±1 so it composes
            // through min/max without hiding the continuous boundary.
            other => {
                let gate = self.compile_expr(other)?;
                Ok(CExpr::If {
                    cond: Box::new(gate),
                    then: Box::new(CExpr::Const(Value::Float(1.0))),
                    otherwise: Box::new(CExpr::Const(Value::Float(-1.0))),
                })
            }
        }
    }
}

impl CExpr {
    /// Collect the attribute and automaton sensitivity sets of this
    /// expression (which changes must re-trigger a function reading it).
    fn collect_sensitivity(&self, vars: &mut Vec<VarIdx>, auts: &mut Vec<AutIdx>) {
        match self {
            CExpr::Const(_) => {}
            CExpr::Var(idx) => vars.push(*idx),
            CExpr::StateActive { automaton, .. } => auts.push(*automaton),
            CExpr::PortAgg { sources, .. } => vars.extend_from_slice(sources),
            CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
                lhs.collect_sensitivity(vars, auts);
                rhs.collect_sensitivity(vars, auts);
            }
            CExpr::Bool { args, .. }
            | CExpr::Add { args }
            | CExpr::Mul { args }
            | CExpr::Min { args }
            | CExpr::Max { args } => {
                for a in args {
                    a.collect_sensitivity(vars, auts);
                }
            }
            CExpr::If {
                cond,
                then,
                otherwise,
            } => {
                cond.collect_sensitivity(vars, auts);
                then.collect_sensitivity(vars, auts);
                otherwise.collect_sensitivity(vars, auts);
            }
            CExpr::Sin(arg) | CExpr::Exp(arg) => arg.collect_sensitivity(vars, auts),
            CExpr::Time => {}
        }
    }

    /// Whether this expression reads the simulation time (which makes
    /// it continuously varying even without ODE attributes).
    fn reads_time(&self) -> bool {
        match self {
            CExpr::Time => true,
            CExpr::Const(_) | CExpr::Var(_) | CExpr::StateActive { .. } | CExpr::PortAgg { .. } => {
                false
            }
            CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
                lhs.reads_time() || rhs.reads_time()
            }
            CExpr::Bool { args, .. }
            | CExpr::Add { args }
            | CExpr::Mul { args }
            | CExpr::Min { args }
            | CExpr::Max { args } => args.iter().any(CExpr::reads_time),
            CExpr::If {
                cond,
                then,
                otherwise,
            } => cond.reads_time() || then.reads_time() || otherwise.reads_time(),
            CExpr::Sin(arg) | CExpr::Exp(arg) => arg.reads_time(),
        }
    }
}

/// Whether `expr` varies during continuous evolution: it reads the
/// simulation time or one of `continuous_vars` (ODE-integrated
/// attributes and their explicit-equation closure).
fn expr_is_continuous(expr: &CExpr, continuous_vars: &BTreeSet<VarIdx>) -> bool {
    if expr.reads_time() {
        return true;
    }
    let mut vars = Vec::new();
    let mut auts = Vec::new();
    expr.collect_sensitivity(&mut vars, &mut auts);
    vars.iter().any(|var| continuous_vars.contains(var))
}

impl CompiledModel {
    /// Validate `model` then resolve every name to dense indices.
    pub fn compile(model: &Model) -> Result<Self, CompileError> {
        model.validate()?;

        // Pass 1: index attributes, automata, states.
        let mut resolver = Resolver {
            vars: HashMap::new(),
            states: HashMap::new(),
            automata: HashMap::new(),
            port_sources: HashMap::new(),
        };
        let mut var_names = Vec::new();
        let mut var_init = Vec::new();
        let mut automata = Vec::new();
        for component in &model.components {
            for attribute in &component.attributes {
                resolver.vars.insert(
                    (component.name.clone(), attribute.name.clone()),
                    var_names.len(),
                );
                var_names.push(format!("{}.{}", component.name, attribute.name));
                var_init.push(attribute.init);
            }
            for automaton in &component.automata {
                let aut_idx = automata.len();
                resolver
                    .automata
                    .insert((component.name.clone(), automaton.name.clone()), aut_idx);
                let mut init = 0;
                for (state_idx, state) in automaton.states.iter().enumerate() {
                    resolver.states.insert(
                        (
                            component.name.clone(),
                            automaton.name.clone(),
                            state.clone(),
                        ),
                        (aut_idx, state_idx),
                    );
                    if state == &automaton.init {
                        init = state_idx;
                    }
                }
                automata.push(CAutomaton {
                    name: format!("{}.{}", component.name, automaton.name),
                    states: automaton.states.clone(),
                    init,
                    transitions: Vec::new(),
                });
            }
        }

        // Pass 2: connections → in-port source lists (declaration order).
        for connection in &model.connections {
            let source_var = model
                .components
                .iter()
                .find(|c| c.name == connection.from.component)
                .and_then(|c| c.ports.iter().find(|p| p.name == connection.from.port))
                .filter(|p| p.dir == PortDir::Out)
                .and_then(|p| p.attr.as_ref())
                .ok_or_else(|| CompileError::Unresolved {
                    what: "out-port",
                    name: format!("{}.{}", connection.from.component, connection.from.port),
                })?;
            let var_idx = resolver.var(&connection.from.component, source_var)?;
            resolver
                .port_sources
                .entry((connection.to.component.clone(), connection.to.port.clone()))
                .or_default()
                .push(var_idx);
        }

        // Pass 3: transitions, functions, indicators.
        let mut transitions = Vec::new();
        let mut explicit: Vec<(VarIdx, CExpr)> = Vec::new();
        let mut ode: Vec<(VarIdx, CExpr)> = Vec::new();
        let mut functions = Vec::new();
        for component in &model.components {
            for automaton in &component.automata {
                let aut_idx = resolver.automata[&(component.name.clone(), automaton.name.clone())];
                for transition in &automaton.transitions {
                    let (_, source) =
                        resolver.state(&component.name, &automaton.name, &transition.source)?;
                    let targets = transition
                        .targets
                        .iter()
                        .map(|t| {
                            resolver
                                .state(&component.name, &automaton.name, t)
                                .map(|(_, s)| s)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let guard = transition
                        .guard
                        .as_ref()
                        .map(|g| resolver.compile_expr(g))
                        .transpose()?;
                    let distribution = match &transition.distrib {
                        Distrib::Delay { time } => CLaw::Delay(*time),
                        Distrib::Exp {
                            rate: Some(rate),
                            rate_expr: None,
                        } => CLaw::Exp(*rate),
                        Distrib::Exp {
                            rate: None,
                            rate_expr: Some(expr),
                        } => CLaw::ExpVar {
                            rate: resolver.compile_expr(expr)?,
                            // Continuity is resolved in pass 3-bis,
                            // once every equation has been collected.
                            continuous: false,
                        },
                        Distrib::Exp { .. } => {
                            // Unreachable after `Model::validate`
                            // (ExpRateSpec), kept as a typed error.
                            return Err(CompileError::Unresolved {
                                what: "exp rate (exactly one of rate/rate_expr)",
                                name: format!(
                                    "{}.{}.{}",
                                    component.name, automaton.name, transition.name
                                ),
                            });
                        }
                        Distrib::Weibull { shape, scale } => CLaw::Weibull(*shape, *scale),
                        Distrib::Lognormal { mu, sigma } => CLaw::Lognormal(*mu, *sigma),
                        Distrib::Gamma { shape, scale } => CLaw::Gamma(*shape, *scale),
                        Distrib::Uniform { low, high } => CLaw::Uniform(*low, *high),
                        Distrib::Empirical { points } => CLaw::Empirical(points.clone()),
                        Distrib::Inst { probs } => {
                            let complement = 1.0 - probs.iter().sum::<f64>();
                            let mut full = probs.clone();
                            full.push(complement);
                            CLaw::Inst(full)
                        }
                        Distrib::Watched => {
                            let Some(guard) = &transition.guard else {
                                return Err(CompileError::Unresolved {
                                    what: "watched guard",
                                    name: format!(
                                        "{}.{}.{}",
                                        component.name, automaton.name, transition.name
                                    ),
                                });
                            };
                            let margin = resolver.compile_watched_margin(guard)?;
                            CLaw::Watched { margin }
                        }
                    };
                    let trans_idx = transitions.len();
                    automata[aut_idx].transitions.push(trans_idx);
                    transitions.push(CTransition {
                        name: format!("{}.{}.{}", component.name, automaton.name, transition.name),
                        automaton: aut_idx,
                        source,
                        guard,
                        targets,
                        on_interruption: transition.on_interruption,
                        distrib: distribution,
                    });
                }
            }
            for function in &component.sensitive_functions {
                let effects = function
                    .effects
                    .iter()
                    .map(|assignment| {
                        let target = resolver
                            .var(&assignment.target.component, &assignment.target.attribute)?;
                        let value = resolver.compile_expr(&assignment.value)?;
                        Ok((target, value))
                    })
                    .collect::<Result<Vec<_>, CompileError>>()?;
                functions.push(CFunction {
                    name: format!("{}.{}", component.name, function.name),
                    effects,
                });
            }
            for equation in &component.equations {
                let target = resolver.var(&component.name, &equation.target)?;
                let expr = resolver.compile_expr(&equation.expr)?;
                match equation.kind {
                    EquationKind::Explicit => explicit.push((target, expr)),
                    EquationKind::Ode => ode.push((target, expr)),
                }
            }
        }
        let watched: Vec<TransIdx> = transitions
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t.distrib, CLaw::Watched { .. }))
            .map(|(i, _)| i)
            .collect();

        // Pass 3-bis: continuity of state-dependent rates (`reschedule_modifiable`
        // routing). A attribute is *continuous* if the ODE integrates it
        // or an explicit equation ties it (transitively) to one.
        let mut continuous_vars: BTreeSet<VarIdx> = ode.iter().map(|(var, _)| *var).collect();
        loop {
            let mut changed = false;
            for (target, expr) in &explicit {
                if !continuous_vars.contains(target) && expr_is_continuous(expr, &continuous_vars) {
                    continuous_vars.insert(*target);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for transition in &mut transitions {
            let is_continuous = match &transition.distrib {
                CLaw::ExpVar { rate, .. } => Some(expr_is_continuous(rate, &continuous_vars)),
                _ => None,
            };
            if let (Some(flag), CLaw::ExpVar { continuous, .. }) =
                (is_continuous, &mut transition.distrib)
            {
                *continuous = flag;
            }
        }

        // Pass 4: sensitivity sets → trigger tables.
        let mut var_triggers = vec![Vec::new(); var_names.len()];
        let mut state_triggers = vec![Vec::new(); automata.len()];
        for (fn_idx, function) in functions.iter().enumerate() {
            let mut vars = Vec::new();
            let mut auts = Vec::new();
            for (_, value) in &function.effects {
                value.collect_sensitivity(&mut vars, &mut auts);
            }
            vars.sort_unstable();
            vars.dedup();
            auts.sort_unstable();
            auts.dedup();
            for var in vars {
                var_triggers[var].push(fn_idx);
            }
            for aut in auts {
                state_triggers[aut].push(fn_idx);
            }
        }

        // Pass 5: indicators.
        let indicators = model
            .indicators
            .iter()
            .map(|indicator| {
                let target = match &indicator.target {
                    IndicatorTarget::Attribute { attr } => {
                        CIndicatorTarget::Var(resolver.var(&attr.component, &attr.attribute)?)
                    }
                    IndicatorTarget::State {
                        component,
                        automaton,
                        state,
                    } => {
                        let (aut, st) = resolver.state(component, automaton, state)?;
                        CIndicatorTarget::State(aut, st)
                    }
                };
                Ok(CIndicator {
                    name: indicator.name.clone(),
                    target,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;

        let var_index = var_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let automaton_index = automata
            .iter()
            .enumerate()
            .map(|(i, a)| (a.name.clone(), i))
            .collect();

        Ok(CompiledModel {
            name: model.name.clone(),
            var_names,
            var_init,
            automata,
            transitions,
            functions,
            var_triggers,
            state_triggers,
            indicators,
            ode,
            explicit,
            watched,
            var_index,
            automaton_index,
        })
    }
}
