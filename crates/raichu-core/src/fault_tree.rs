//! Fault-tree generation by backward chaining (the reference engine's
//! fault-tree generator, user manual 5.5.6 and developer manual 4.1.12;
//! cited, not copied).
//!
//! # What a fault tree explains here
//!
//! A **top expression**, a boolean over automaton states, and **why it can
//! become true**. The explanation rests on the reference method's
//! hypothesis, carried rather than approximated:
//!
//! - every attribute keeps its **initial value**, or the value a declared
//!   **profile** gives it: attributes are constants, so a guard reading only
//!   attributes is a constant too;
//! - only **states** move, and a state is entered by firing a transition
//!   into it. Firing a transition is explained as its **basic event** (the
//!   draw of its law) AND its **source state** being active AND its
//!   **guard** holding.
//!
//! A state is explained as: it is the automaton's initial state (constant
//! true), OR one of the transitions into it fires. The recursion runs back
//! through the source states; a state already being explained further up
//! the same branch contributes nothing to it (a path that loops back adds
//! no new way in), which is how a repair loop is handled rather than
//! truncated: `nok` is reached from `ok`, `ok` is initial, and the repair
//! from `nok` back to `ok` is never needed to explain `nok`.
//!
//! # What this is not
//!
//! Not the minimal sequences of a simulation (`raichu_analysis::sequence`), which let
//! attributes move and order the events. A tree is a static structure; the
//! two answer different questions and are not derived from each other.
//!
//! # What is refused, by name
//!
//! The method explains a value becoming TRUE through states being entered,
//! so it needs the top expression and every guard it crosses to be
//! **coherent** (monotone) in the states: a negated state, or a state read
//! inside arithmetic this module does not recognise (anything but a vote,
//! `sum(if(state, 1, 0)) >= k`), is refused rather than answered. So is a
//! guard reading a continuous quantity's crossing that the frozen
//! attributes cannot settle.

use std::collections::{BTreeSet, HashMap};

use raichu_expr::{BoolOp, CmpOp, Expr, Value};

use crate::compile::{AutIdx, CExpr, CLaw, CompiledModel, StateIdx};

/// Why a tree could not be generated.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FaultTreeError {
    /// An imported FMU is opaque to structural backward chaining.
    #[error("fault tree: FMU unit `{unit}` is opaque to structural analysis")]
    OpaqueFmu {
        /// Name of the imported unit.
        unit: String,
    },
    /// A name in the top expression or the profile designates nothing.
    #[error("fault tree: `{0}` designates no attribute, automaton or state of the model")]
    Unresolved(String),
    /// A state is read where the method cannot explain it becoming true.
    #[error(
        "fault tree: {0}. Backward chaining explains a value becoming TRUE \
         through states being entered, so it needs every expression it \
         crosses to be monotone in the states"
    )]
    NonCoherent(String),
    /// An expression form this module does not carry.
    #[error("fault tree: {0}")]
    Unsupported(String),
    /// Evaluating a frozen expression failed.
    #[error("fault tree: evaluating a frozen expression failed: {0}")]
    Evaluation(String),
    /// The tree or its cut sets outgrew the declared budget.
    #[error("fault tree: {0}")]
    TooLarge(String),
}

/// How a basic event occurs: the law of the transition it stands for.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "law", rename_all = "snake_case")]
pub enum BasicLaw {
    /// Exponential time to occurrence, rate λ.
    Exponential {
        /// Rate.
        rate: f64,
    },
    /// Weibull time to occurrence.
    Weibull {
        /// Shape k.
        shape: f64,
        /// Scale λ.
        scale: f64,
    },
    /// Log-normal time to occurrence (μ, σ of the underlying normal).
    Lognormal {
        /// Mean of the underlying normal.
        mu: f64,
        /// Deviation of the underlying normal.
        sigma: f64,
    },
    /// Gamma time to occurrence.
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
    /// Deterministic delay.
    Delay {
        /// The delay.
        time: f64,
    },
    /// An empirical time-to-occurrence table.
    Empirical {
        /// `(time, cumulative probability)` pairs.
        points: Vec<(f64, f64)>,
    },
    /// An on-demand branch taken with this probability.
    Probability {
        /// The branch probability.
        probability: f64,
    },
}

/// One basic event: a transition's draw (into one target, for a branching
/// transition).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct TreeEvent {
    /// A name OpenPSA accepts, unique in the tree:
    /// `{component}.{automaton}.{transition}`, plus `.{target}` on a
    /// branching transition.
    pub name: String,
    /// The component the transition belongs to.
    pub component: String,
    /// The automaton, qualified (`component.automaton`).
    pub automaton: String,
    /// The transition's own name.
    pub transition: String,
    /// The state it leads to.
    pub target: String,
    /// Its law.
    #[serde(flatten)]
    pub law: BasicLaw,
}

/// The connective of a gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "gate", rename_all = "snake_case")]
pub enum GateOp {
    /// Every child holds.
    And,
    /// At least one child holds.
    Or,
    /// At least `k` children hold.
    AtLeast {
        /// How many.
        k: usize,
    },
}

/// A node of the tree.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "node", rename_all = "snake_case")]
pub enum FtNode {
    /// A constant (only as the whole tree, once simplified).
    Constant {
        /// Its value.
        value: bool,
    },
    /// A basic event, by index into [`FaultTree::basic_events`].
    Basic {
        /// The index.
        event: usize,
    },
    /// A gate over children.
    Gate {
        /// Its connective.
        #[serde(flatten)]
        op: GateOp,
        /// Its children.
        children: Vec<FtNode>,
    },
}

/// A generated tree.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct FaultTree {
    /// The top gate, simplified: constant gates propagated, single-child
    /// gates collapsed, nested gates of one connective merged.
    pub top: FtNode,
    /// Every basic event the tree reaches, in first-reached order.
    pub basic_events: Vec<TreeEvent>,
}

/// What a generation may be given beside the top expression.
#[derive(Debug, Clone, Default)]
pub struct FaultTreeSettings {
    /// Attributes held at a value other than their initial one, by
    /// qualified name (`component.attribute`): the reference method's
    /// profiles.
    pub profile: Vec<(String, Value)>,
    /// The most nodes the unsimplified tree may hold; `None` for the
    /// default of one million.
    pub max_nodes: Option<usize>,
}

const DEFAULT_MAX_NODES: usize = 1_000_000;

/// Generate the fault tree explaining `top` on `model`.
pub fn fault_tree(
    model: &CompiledModel,
    top: &Expr,
    settings: &FaultTreeSettings,
) -> Result<FaultTree, FaultTreeError> {
    if let Some(unit) = model.fmu_units.first() {
        return Err(FaultTreeError::OpaqueFmu {
            unit: unit.name.clone(),
        });
    }
    let mut vars = model.var_init.clone();
    for (name, value) in &settings.profile {
        let index = *model
            .var_index
            .get(name)
            .ok_or_else(|| FaultTreeError::Unresolved(name.clone()))?;
        vars[index] = *value;
    }
    let top = compile_top(model, top)?;
    let mut builder = Builder {
        model,
        vars,
        init_states: model.automata.iter().map(|a| a.init).collect(),
        events: Vec::new(),
        event_index: HashMap::new(),
        stack: Vec::new(),
        nodes: 0,
        max_nodes: settings.max_nodes.unwrap_or(DEFAULT_MAX_NODES),
    };
    let raw = builder.expand(&top)?;
    Ok(FaultTree {
        top: simplify(raw),
        basic_events: builder.events,
    })
}

struct Builder<'a> {
    model: &'a CompiledModel,
    vars: Vec<Value>,
    init_states: Vec<StateIdx>,
    events: Vec<TreeEvent>,
    event_index: HashMap<(usize, StateIdx), usize>,
    stack: Vec<(AutIdx, StateIdx)>,
    nodes: usize,
    max_nodes: usize,
}

impl Builder<'_> {
    fn count(&mut self) -> Result<(), FaultTreeError> {
        self.nodes += 1;
        if self.nodes > self.max_nodes {
            return Err(FaultTreeError::TooLarge(format!(
                "the tree outgrew {} nodes; raise `max_nodes` or explain a \
                 narrower top expression",
                self.max_nodes
            )));
        }
        Ok(())
    }

    fn frozen(&self, expr: &CExpr) -> Result<Value, FaultTreeError> {
        crate::engine::eval_frozen(self.model, &self.vars, &self.init_states, expr)
            .map_err(|e| FaultTreeError::Evaluation(e.to_string()))
    }

    fn frozen_bool(&self, expr: &CExpr) -> Result<bool, FaultTreeError> {
        match self.frozen(expr)? {
            Value::Bool(b) => Ok(b),
            other => Err(FaultTreeError::Evaluation(format!(
                "expected a boolean, got {other:?}"
            ))),
        }
    }

    fn state_name(&self, automaton: AutIdx, state: StateIdx) -> String {
        let aut = &self.model.automata[automaton];
        format!("{}.{}", aut.name, aut.states[state])
    }

    fn expand(&mut self, expr: &CExpr) -> Result<FtNode, FaultTreeError> {
        self.count()?;
        if !reads_state(expr) {
            return Ok(FtNode::Constant {
                value: self.frozen_bool(expr)?,
            });
        }
        match expr {
            CExpr::StateActive { automaton, state } => self.expand_state(*automaton, *state),
            CExpr::Bool { op, args } => match op {
                BoolOp::And | BoolOp::Or => {
                    let children = args
                        .iter()
                        .map(|arg| self.expand(arg))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(FtNode::Gate {
                        op: if *op == BoolOp::And {
                            GateOp::And
                        } else {
                            GateOp::Or
                        },
                        children,
                    })
                }
                BoolOp::Not => Err(FaultTreeError::NonCoherent(format!(
                    "a state is read under a negation ({})",
                    self.describe(expr)
                ))),
            },
            CExpr::Cmp { op, lhs, rhs } => self.expand_vote(*op, lhs, rhs, expr),
            _ => Err(FaultTreeError::Unsupported(format!(
                "a state is read inside an expression this module does not \
                 explain ({}); a vote is written `sum(if(state, 1, 0)) >= k`",
                self.describe(expr)
            ))),
        }
    }

    /// `sum_i if(c_i, 1, 0) >= k` (or `> k - 1`): at least k of the c_i.
    fn expand_vote(
        &mut self,
        op: CmpOp,
        lhs: &CExpr,
        rhs: &CExpr,
        whole: &CExpr,
    ) -> Result<FtNode, FaultTreeError> {
        let refuse = |this: &Self| {
            FaultTreeError::Unsupported(format!(
                "a state is read inside a comparison that is not a vote ({}); a \
                 vote is written `sum(if(state, 1, 0)) >= k`",
                this.describe(whole)
            ))
        };
        if reads_state(rhs) {
            return Err(refuse(self));
        }
        let threshold = match self.frozen(rhs)? {
            Value::Int(i) => i as f64,
            Value::Float(f) => f,
            Value::Bool(_) => return Err(refuse(self)),
        };
        let needed = match op {
            CmpOp::Ge => threshold.ceil(),
            CmpOp::Gt => threshold.floor() + 1.0,
            _ => return Err(refuse(self)),
        };
        let terms: Vec<&CExpr> = match lhs {
            CExpr::Add { args } => args.iter().collect(),
            other => vec![other],
        };
        let mut children = Vec::new();
        let mut constant_votes = 0.0;
        for term in terms {
            match term {
                CExpr::If {
                    cond,
                    then,
                    otherwise,
                } if !reads_state(then) && !reads_state(otherwise) => {
                    let one = number(self.frozen(then)?);
                    let zero = number(self.frozen(otherwise)?);
                    if one != 1.0 || zero != 0.0 {
                        return Err(refuse(self));
                    }
                    if reads_state(cond) {
                        children.push(self.expand(cond)?);
                    } else if self.frozen_bool(cond)? {
                        constant_votes += 1.0;
                    }
                }
                other if !reads_state(other) => constant_votes += number(self.frozen(other)?),
                _ => return Err(refuse(self)),
            }
        }
        let k = needed - constant_votes;
        if k <= 0.0 {
            return Ok(FtNode::Constant { value: true });
        }
        if k > children.len() as f64 {
            return Ok(FtNode::Constant { value: false });
        }
        Ok(FtNode::Gate {
            op: GateOp::AtLeast { k: k as usize },
            children,
        })
    }

    fn expand_state(
        &mut self,
        automaton: AutIdx,
        state: StateIdx,
    ) -> Result<FtNode, FaultTreeError> {
        self.count()?;
        if self.stack.contains(&(automaton, state)) {
            // Already being explained further up this branch: a path that
            // loops back to it adds no new way in.
            return Ok(FtNode::Constant { value: false });
        }
        let mut ways = Vec::new();
        if self.model.automata[automaton].init == state {
            ways.push(FtNode::Constant { value: true });
        }
        self.stack.push((automaton, state));
        let transitions = self.model.automata[automaton].transitions.clone();
        for t_idx in transitions {
            let transition = &self.model.transitions[t_idx];
            let Some(position) = transition.targets.iter().position(|&s| s == state) else {
                continue;
            };
            if transition.source == state {
                continue;
            }
            let source = transition.source;
            let guard = transition.guard.clone();
            // The source and the guard first: a way in that cannot hold
            // registers no basic event, so the tree lists only the events
            // some explanation uses.
            let from = self.expand_state(automaton, source)?;
            if from == (FtNode::Constant { value: false }) {
                continue;
            }
            let mut conjuncts = vec![from];
            if let Some(guard) = guard {
                let condition = self.expand(&guard).map_err(|e| match e {
                    FaultTreeError::NonCoherent(detail) | FaultTreeError::Unsupported(detail) => {
                        FaultTreeError::Unsupported(format!(
                            "the guard of transition `{}` into `{}`: {detail}",
                            self.model.transitions[t_idx].name,
                            self.state_name(automaton, state)
                        ))
                    }
                    other => other,
                })?;
                if condition == (FtNode::Constant { value: false }) {
                    continue;
                }
                conjuncts.push(condition);
            }
            let event = self.event(t_idx, position)?;
            if event == (FtNode::Constant { value: false }) {
                continue;
            }
            conjuncts.insert(0, event);
            ways.push(FtNode::Gate {
                op: GateOp::And,
                children: conjuncts,
            });
        }
        self.stack.pop();
        Ok(FtNode::Gate {
            op: GateOp::Or,
            children: ways,
        })
    }

    /// The basic event of transition `t_idx` into its `position`-th target,
    /// or a constant for a transition no draw governs.
    fn event(&mut self, t_idx: usize, position: usize) -> Result<FtNode, FaultTreeError> {
        let transition = &self.model.transitions[t_idx];
        let law = match &transition.distrib {
            CLaw::Exp(rate) => BasicLaw::Exponential { rate: *rate },
            CLaw::ExpVar { rate, .. } => {
                if reads_state(rate) {
                    return Err(FaultTreeError::Unsupported(format!(
                        "the rate of transition `{}` reads a state, so no single \
                         law stands for its draw",
                        transition.name
                    )));
                }
                BasicLaw::Exponential {
                    rate: number(self.frozen(rate)?),
                }
            }
            CLaw::Weibull(shape, scale) => BasicLaw::Weibull {
                shape: *shape,
                scale: *scale,
            },
            CLaw::Lognormal(mu, sigma) => BasicLaw::Lognormal {
                mu: *mu,
                sigma: *sigma,
            },
            CLaw::Gamma(shape, scale) => BasicLaw::Gamma {
                shape: *shape,
                scale: *scale,
            },
            CLaw::Uniform(low, high) => BasicLaw::Uniform {
                low: *low,
                high: *high,
            },
            CLaw::Delay(time) => BasicLaw::Delay { time: *time },
            CLaw::Empirical(points) => BasicLaw::Empirical {
                points: points.clone(),
            },
            CLaw::Inst(probabilities) => {
                let probability = probabilities.get(position).copied().unwrap_or(0.0);
                if probability <= 0.0 {
                    return Ok(FtNode::Constant { value: false });
                }
                if probability >= 1.0 {
                    return Ok(FtNode::Constant { value: true });
                }
                BasicLaw::Probability { probability }
            }
            CLaw::Watched { margin } => {
                // A crossing no draw governs: with the attributes frozen,
                // it happens or it never does.
                if reads_state(margin) {
                    return Err(FaultTreeError::Unsupported(format!(
                        "the boundary of watched transition `{}` reads a state",
                        transition.name
                    )));
                }
                let crosses = number(self.frozen(margin)?) >= 0.0;
                return Ok(FtNode::Constant { value: crosses });
            }
        };
        let key = (t_idx, position);
        if let Some(&index) = self.event_index.get(&key) {
            return Ok(FtNode::Basic { event: index });
        }
        let automaton = &self.model.automata[transition.automaton];
        let target_state = transition.targets[position];
        let target = automaton.states[target_state].clone();
        // The compiled name is already qualified: `component.automaton.name`.
        let mut name = transition.name.clone();
        if transition.targets.len() > 1 {
            name = format!("{name}.{target}");
        }
        let index = self.events.len();
        self.events.push(TreeEvent {
            name,
            component: transition.component.clone(),
            automaton: automaton.name.clone(),
            transition: transition
                .name
                .rsplit('.')
                .next()
                .unwrap_or(&transition.name)
                .to_owned(),
            target,
            law,
        });
        self.event_index.insert(key, index);
        Ok(FtNode::Basic { event: index })
    }

    fn describe(&self, expr: &CExpr) -> String {
        describe(self.model, expr)
    }
}

/// A value read as a number: a boolean counts one or zero.
fn number(value: Value) -> f64 {
    match value {
        Value::Bool(b) => f64::from(u8::from(b)),
        Value::Int(i) => i as f64,
        Value::Float(f) => f,
    }
}

/// Whether `expr` reads an automaton state anywhere.
fn reads_state(expr: &CExpr) -> bool {
    match expr {
        CExpr::StateActive { .. } => true,
        CExpr::Const(_) | CExpr::Var(_) | CExpr::PortAgg { .. } | CExpr::Time => false,
        CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
            reads_state(lhs) || reads_state(rhs)
        }
        CExpr::Bool { args, .. }
        | CExpr::Add { args }
        | CExpr::Mul { args }
        | CExpr::Min { args }
        | CExpr::Max { args } => args.iter().any(reads_state),
        CExpr::If {
            cond,
            then,
            otherwise,
        } => reads_state(cond) || reads_state(then) || reads_state(otherwise),
        CExpr::Sin(arg) | CExpr::Exp(arg) => reads_state(arg),
    }
}

fn describe(model: &CompiledModel, expr: &CExpr) -> String {
    match expr {
        CExpr::StateActive { automaton, state } => {
            let aut = &model.automata[*automaton];
            format!("{}.{}", aut.name, aut.states[*state])
        }
        CExpr::Var(index) => model.var_names[*index].clone(),
        CExpr::Const(value) => format!("{value:?}"),
        CExpr::Bool { op, args } => format!(
            "{op:?}({})",
            args.iter()
                .map(|a| describe(model, a))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        CExpr::Cmp { op, lhs, rhs } => {
            format!("{op:?}({}, {})", describe(model, lhs), describe(model, rhs))
        }
        other => format!("{other:?}").chars().take(120).collect(),
    }
}

/// Compile a top expression against the model: attributes, states and the
/// operators a tree can explain. Port aggregations and time are refused, a
/// top expression being a statement about the model's states.
fn compile_top(model: &CompiledModel, expr: &Expr) -> Result<CExpr, FaultTreeError> {
    let args = |list: &[Expr]| -> Result<Vec<CExpr>, FaultTreeError> {
        list.iter().map(|e| compile_top(model, e)).collect()
    };
    Ok(match expr {
        Expr::Const { value } => CExpr::Const(*value),
        Expr::Attr { attr } => {
            let name = format!("{}.{}", attr.component, attr.attribute);
            CExpr::Var(
                *model
                    .var_index
                    .get(&name)
                    .ok_or(FaultTreeError::Unresolved(name))?,
            )
        }
        Expr::StateActive { state } => {
            let qualified = format!("{}.{}", state.component, state.automaton);
            let automaton = *model
                .automaton_index
                .get(&qualified)
                .ok_or_else(|| FaultTreeError::Unresolved(qualified.clone()))?;
            let index = model.automata[automaton]
                .states
                .iter()
                .position(|s| *s == state.state)
                .ok_or_else(|| {
                    FaultTreeError::Unresolved(format!("{qualified}.{}", state.state))
                })?;
            CExpr::StateActive {
                automaton,
                state: index,
            }
        }
        Expr::Cmp { cmp, lhs, rhs } => CExpr::Cmp {
            op: *cmp,
            lhs: Box::new(compile_top(model, lhs)?),
            rhs: Box::new(compile_top(model, rhs)?),
        },
        Expr::Bool {
            bool_op,
            args: list,
        } => CExpr::Bool {
            op: *bool_op,
            args: args(list)?,
        },
        Expr::Add { args: list } => CExpr::Add { args: args(list)? },
        Expr::If {
            cond,
            then,
            otherwise,
        } => CExpr::If {
            cond: Box::new(compile_top(model, cond)?),
            then: Box::new(compile_top(model, then)?),
            otherwise: Box::new(compile_top(model, otherwise)?),
        },
        _ => {
            return Err(FaultTreeError::Unsupported(
                "a top expression is written over attributes and states with \
                 `const`, `attr`, `state_active`, `cmp`, `bool`, `add` and `if`"
                    .to_owned(),
            ))
        }
    })
}

/// Propagate constants, collapse single-child gates and merge nested gates
/// of one connective (the reference options `reduce_constant_gt`,
/// `reduce_one_son_gt`, `reduce_same_op_gt`, always applied).
fn simplify(node: FtNode) -> FtNode {
    let FtNode::Gate { op, children } = node else {
        return node;
    };
    let mut kept = Vec::new();
    let mut trues = 0usize;
    let mut falses = 0usize;
    for child in children.into_iter().map(simplify) {
        match child {
            FtNode::Constant { value: true } => trues += 1,
            FtNode::Constant { value: false } => falses += 1,
            FtNode::Gate {
                op: inner,
                children: grandchildren,
            } if inner == op && matches!(op, GateOp::And | GateOp::Or) => {
                kept.extend(grandchildren)
            }
            other => kept.push(other),
        }
    }
    let gate = |op: GateOp, mut kept: Vec<FtNode>| -> FtNode {
        if kept.len() == 1 {
            kept.remove(0)
        } else {
            FtNode::Gate { op, children: kept }
        }
    };
    match op {
        GateOp::And if falses > 0 => FtNode::Constant { value: false },
        GateOp::And if kept.is_empty() => FtNode::Constant { value: true },
        GateOp::And => gate(GateOp::And, kept),
        GateOp::Or if trues > 0 => FtNode::Constant { value: true },
        GateOp::Or if kept.is_empty() => FtNode::Constant { value: false },
        GateOp::Or => gate(GateOp::Or, kept),
        GateOp::AtLeast { k } => {
            if trues >= k {
                return FtNode::Constant { value: true };
            }
            let k = k - trues;
            if k > kept.len() {
                FtNode::Constant { value: false }
            } else if k == kept.len() {
                gate(GateOp::And, kept)
            } else if k == 1 {
                gate(GateOp::Or, kept)
            } else {
                FtNode::Gate {
                    op: GateOp::AtLeast { k },
                    children: kept,
                }
            }
        }
    }
}

impl FaultTree {
    /// The minimal cut sets, each a sorted list of basic-event indices,
    /// sorted by size then lexicographically. `limit` bounds how many
    /// intermediate sets an expansion may hold.
    pub fn minimal_cut_sets(&self, limit: usize) -> Result<Vec<Vec<usize>>, FaultTreeError> {
        let mut sets: Vec<Vec<usize>> = cut_sets(&self.top, limit)?
            .into_iter()
            .map(|set| set.into_iter().collect())
            .collect();
        sets.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
        Ok(sets)
    }

    /// The tree in the OpenPSA model-exchange format (the reference
    /// engine's fault-tree file format, user manual 7.6), as a document
    /// named `name`.
    pub fn to_open_psa(&self, name: &str) -> String {
        let mut gates = Vec::new();
        let top = open_psa_formula(&self.top, &self.basic_events, &mut gates);
        let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<opsa-mef>\n");
        out.push_str(&format!("  <define-fault-tree name=\"{}\">\n", xml(name)));
        out.push_str(&format!(
            "    <define-gate name=\"top\">\n      {top}\n    </define-gate>\n"
        ));
        for (index, formula) in gates.iter().enumerate() {
            out.push_str(&format!(
                "    <define-gate name=\"G{}\">\n      {formula}\n    </define-gate>\n",
                index + 1
            ));
        }
        out.push_str("  </define-fault-tree>\n  <model-data>\n");
        for event in &self.basic_events {
            out.push_str(&format!(
                "    <define-basic-event name=\"{}\">\n{}    </define-basic-event>\n",
                xml(&event.name),
                open_psa_law(&event.law)
            ));
        }
        out.push_str("  </model-data>\n</opsa-mef>\n");
        out
    }
}

type CutSets = Vec<BTreeSet<usize>>;

fn minimise(mut sets: CutSets) -> CutSets {
    sets.sort_by_key(|s| s.len());
    let mut kept: CutSets = Vec::new();
    for set in sets {
        if !kept.iter().any(|k| k.is_subset(&set)) {
            kept.push(set);
        }
    }
    kept
}

fn cut_sets(node: &FtNode, limit: usize) -> Result<CutSets, FaultTreeError> {
    let too_many = || {
        FaultTreeError::TooLarge(format!(
            "the cut sets outgrew {limit} while being expanded; raise the limit"
        ))
    };
    Ok(match node {
        FtNode::Constant { value: true } => vec![BTreeSet::new()],
        FtNode::Constant { value: false } => Vec::new(),
        FtNode::Basic { event } => vec![BTreeSet::from([*event])],
        FtNode::Gate { op, children } => {
            let parts = children
                .iter()
                .map(|c| cut_sets(c, limit))
                .collect::<Result<Vec<_>, _>>()?;
            match op {
                GateOp::Or => {
                    let all: CutSets = parts.into_iter().flatten().collect();
                    if all.len() > limit {
                        return Err(too_many());
                    }
                    minimise(all)
                }
                GateOp::And => {
                    let mut acc: CutSets = vec![BTreeSet::new()];
                    for part in parts {
                        let mut next = Vec::new();
                        for a in &acc {
                            for b in &part {
                                next.push(a.union(b).copied().collect());
                                if next.len() > limit {
                                    return Err(too_many());
                                }
                            }
                        }
                        acc = minimise(next);
                    }
                    acc
                }
                GateOp::AtLeast { k } => {
                    let mut all = Vec::new();
                    for combination in combinations(parts.len(), *k) {
                        let mut acc: CutSets = vec![BTreeSet::new()];
                        for index in combination {
                            let mut next = Vec::new();
                            for a in &acc {
                                for b in &parts[index] {
                                    next.push(a.union(b).copied().collect());
                                    if next.len() > limit {
                                        return Err(too_many());
                                    }
                                }
                            }
                            acc = minimise(next);
                        }
                        all.extend(acc);
                        if all.len() > limit {
                            return Err(too_many());
                        }
                    }
                    minimise(all)
                }
            }
        }
    })
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn walk(start: usize, n: usize, k: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            walk(i + 1, n, k, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    walk(0, n, k, &mut Vec::new(), &mut out);
    out
}

fn xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn open_psa_formula(node: &FtNode, events: &[TreeEvent], gates: &mut Vec<String>) -> String {
    match node {
        FtNode::Constant { value } => format!("<constant value=\"{value}\"/>"),
        FtNode::Basic { event } => {
            format!("<basic-event name=\"{}\"/>", xml(&events[*event].name))
        }
        FtNode::Gate { op, children } => {
            let args: Vec<String> = children
                .iter()
                .map(|child| match child {
                    FtNode::Gate { .. } => {
                        let formula = open_psa_formula(child, events, gates);
                        gates.push(formula);
                        format!("<gate name=\"G{}\"/>", gates.len())
                    }
                    other => open_psa_formula(other, events, gates),
                })
                .collect();
            let (open, close) = match op {
                GateOp::And => ("<and>".to_owned(), "</and>"),
                GateOp::Or => ("<or>".to_owned(), "</or>"),
                GateOp::AtLeast { k } => (format!("<atleast min=\"{k}\">"), "</atleast>"),
            };
            format!("{open}{}{close}", args.join(""))
        }
    }
}

fn open_psa_law(law: &BasicLaw) -> String {
    let float = |v: f64| format!("<float value=\"{v}\"/>");
    match law {
        BasicLaw::Exponential { rate } => format!(
            "      <exponential>{}<system-mission-time/></exponential>\n",
            float(*rate)
        ),
        BasicLaw::Weibull { shape, scale } => format!(
            "      <Weibull>{}{}{}<system-mission-time/></Weibull>\n",
            float(*scale),
            float(*shape),
            float(0.0)
        ),
        BasicLaw::Probability { probability } => format!("      {}\n", float(*probability)),
        other => {
            // No OpenPSA time-to-failure expression for this law: the law is
            // carried as attributes, so a reader loses nothing it could not
            // compute from the law itself.
            let json = serde_json::to_value(other).unwrap_or_default();
            let mut attributes = String::from("      <attributes>\n");
            if let Some(map) = json.as_object() {
                for (key, value) in map {
                    attributes.push_str(&format!(
                        "        <attribute name=\"{}\" value=\"{}\"/>\n",
                        xml(key),
                        xml(&value.to_string().trim_matches('"').replace('"', "'"))
                    ));
                }
            }
            attributes.push_str("      </attributes>\n");
            attributes
        }
    }
}
