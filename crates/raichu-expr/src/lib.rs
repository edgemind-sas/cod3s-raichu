//! # raichu-expr — serializable expression trees
//!
//! Guards, sensitive-function effects and (from milestone M1) ODE
//! right-hand sides are **pure data**: expression trees serialized inside
//! the model file and evaluated/compiled on the Rust side (SymPy/CasADi
//! style). This is what keeps the engine free of Python callbacks on the
//! hot path (Performance contract) while keeping models fully
//! serializable.
//!
//! M0 subset (frozen — see the M0 plan): constant/attribute leaves,
//! comparison and boolean operators, port aggregation (sum/count/all/any),
//! direct assignment. Arithmetic and math functions arrive in M1; the
//! [`Expr`] enum is `#[non_exhaustive]`-in-spirit (tagged serde repr) so
//! extension does not break serialized models.
//!
//! An optional Rust trait escape hatch for compiled custom behaviour is
//! part of the design (reserved API), not exercised in M0.

use serde::{Deserialize, Serialize};

/// A runtime value carried by attributes and expressions.
///
/// M0 kinds only: booleans, integers, floats. `String` discrete state is
/// reserved (a deliberate departure) and will extend this enum
/// without breaking serialized models (tagged representation).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Value {
    /// Boolean value.
    Bool(bool),
    /// 64-bit signed integer value.
    Int(i64),
    /// 64-bit floating-point value.
    Float(f64),
}

/// Reference to an attribute by hierarchical name — the *authoring /
/// serialized* form. Build-time validation resolves it to dense indices
/// (typed errors on dangling references; no stringly-typed lookups at
/// simulation time).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AttrRef {
    /// Name of the component owning the attribute.
    pub component: String,
    /// Name of the attribute inside the component.
    pub attribute: String,
}

/// Reference to a port by hierarchical name (authoring form).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PortRef {
    /// Name of the component owning the port.
    pub component: String,
    /// Name of the port inside the component.
    pub port: String,
}

/// Reference to an automaton state by hierarchical name (authoring form).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateRef {
    /// Name of the component owning the automaton.
    pub component: String,
    /// Name of the automaton.
    pub automaton: String,
    /// Name of the state.
    pub state: String,
}

/// Comparison operators (guards on discrete state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CmpOp {
    /// Equality.
    Eq,
    /// Inequality.
    Ne,
    /// Strictly less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Strictly greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

/// Boolean connectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoolOp {
    /// Conjunction of all arguments.
    And,
    /// Disjunction of all arguments.
    Or,
    /// Negation (exactly one argument; enforced by model validation).
    Not,
}

/// Aggregation over the values connected to an *in* port
/// (sum, count, all, any, mean, median).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggOp {
    /// Sum of connected numeric values.
    Sum,
    /// Number of connections.
    Count,
    /// True iff all connected boolean values are true.
    All,
    /// True iff at least one connected boolean value is true.
    Any,
    /// Arithmetic mean of connected numeric values (M3: sensor
    /// averaging — cod3s `compute_reference_mean`). 0.0 with no
    /// connection.
    Mean,
    /// Median of connected numeric values (M3: redundant-sensor
    /// median/vote). Even count averages the
    /// two central values; 0.0 with no connection.
    Median,
}

/// A serializable expression tree (M0 subset).
///
/// The serde representation is tag-based (`op` field), so adding variants
/// in later milestones (arithmetic, math functions, ODE right-hand sides)
/// keeps old model files loadable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Expr {
    /// Literal constant.
    Const {
        /// The constant value.
        value: Value,
    },
    /// Read an attribute.
    Attr {
        /// The referenced attribute.
        attr: AttrRef,
    },
    /// Aggregate the values connected to an in-port.
    PortAgg {
        /// The referenced in-port.
        port: PortRef,
        /// The aggregation operator.
        agg: AggOp,
    },
    /// Compare two sub-expressions.
    Cmp {
        /// The comparison operator.
        cmp: CmpOp,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// Combine boolean sub-expressions.
    Bool {
        /// The boolean connective.
        bool_op: BoolOp,
        /// Operands (`Not` takes exactly one; validated at model build).
        args: Vec<Expr>,
    },
    /// True while the referenced automaton is in the given state:
    /// muscadet links state changes to attribute changes through it.
    StateActive {
        /// The referenced state.
        state: StateRef,
    },
    /// N-ary sum (M1).
    Add {
        /// Operands (≥ 1, validated at model build).
        args: Vec<Expr>,
    },
    /// Binary subtraction.
    Sub {
        /// Minuend.
        lhs: Box<Expr>,
        /// Subtrahend.
        rhs: Box<Expr>,
    },
    /// N-ary product.
    Mul {
        /// Operands (≥ 1, validated at model build).
        args: Vec<Expr>,
    },
    /// Binary division (always evaluates as float).
    Div {
        /// Dividend.
        lhs: Box<Expr>,
        /// Divisor.
        rhs: Box<Expr>,
    },
    /// N-ary minimum (clamping — `min(max(lo, v), hi)` patterns).
    Min {
        /// Operands (≥ 1, validated at model build).
        args: Vec<Expr>,
    },
    /// N-ary maximum.
    Max {
        /// Operands (≥ 1, validated at model build).
        args: Vec<Expr>,
    },
    /// Conditional expression — with [`Expr::StateActive`] as condition
    /// it covers the piecewise-by-automaton-state right-hand sides of
    /// the corpus (empty/full tank freezing `dv/dt`).
    If {
        /// Boolean condition.
        cond: Box<Expr>,
        /// Value when the condition holds.
        then: Box<Expr>,
        /// Value otherwise.
        otherwise: Box<Expr>,
    },
    /// Sine (M4: sinusoidal sources, RLC generator).
    Sin {
        /// Operand (radians).
        arg: Box<Expr>,
    },
    /// Natural exponential `e^arg` (state-dependent failure rates —
    /// the heated-tank λ(T) of the Aldemir benchmark).
    Exp {
        /// Operand.
        arg: Box<Expr>,
    },
    /// Current simulation time (M4). Meaningful in continuously
    /// evaluated expressions (equations, watched margins, guards);
    /// sensitive functions are *not* re-triggered by the passage of
    /// time alone.
    Time,
}

impl Expr {
    /// Convenience constructor: boolean constant.
    #[must_use]
    pub fn bool(value: bool) -> Self {
        Expr::Const {
            value: Value::Bool(value),
        }
    }

    /// Convenience constructor: read an attribute.
    #[must_use]
    pub fn attr(component: impl Into<String>, attribute: impl Into<String>) -> Self {
        Expr::Attr {
            attr: AttrRef {
                component: component.into(),
                attribute: attribute.into(),
            },
        }
    }

    /// Visit the direct children of this node (traversal backbone of
    /// the reference visitors below — new variants only need a case
    /// here).
    pub fn for_each_child(&self, f: &mut impl FnMut(&Expr)) {
        match self {
            Expr::Const { .. }
            | Expr::Attr { .. }
            | Expr::PortAgg { .. }
            | Expr::StateActive { .. }
            | Expr::Time => {}
            Expr::Sin { arg } | Expr::Exp { arg } => f(arg),
            Expr::Cmp { lhs, rhs, .. } | Expr::Sub { lhs, rhs } | Expr::Div { lhs, rhs } => {
                f(lhs);
                f(rhs);
            }
            Expr::Bool { args, .. }
            | Expr::Add { args }
            | Expr::Mul { args }
            | Expr::Min { args }
            | Expr::Max { args } => {
                for a in args {
                    f(a);
                }
            }
            Expr::If {
                cond,
                then,
                otherwise,
            } => {
                f(cond);
                f(then);
                f(otherwise);
            }
        }
    }

    /// Visit every attribute reference in the tree (used by model
    /// validation to check that all references resolve, and by the engine
    /// to derive sensitivity sets — which attribute changes must re-trigger
    /// which functions).
    pub fn for_each_attr_ref(&self, f: &mut impl FnMut(&AttrRef)) {
        if let Expr::Attr { attr } = self {
            f(attr);
        }
        self.for_each_child(&mut |child| child.for_each_attr_ref(f));
    }

    /// Visit every port reference in the tree (model validation).
    pub fn for_each_port_ref(&self, f: &mut impl FnMut(&PortRef)) {
        if let Expr::PortAgg { port, .. } = self {
            f(port);
        }
        self.for_each_child(&mut |child| child.for_each_port_ref(f));
    }

    /// Visit every automaton-state reference in the tree (model
    /// validation; state-sensitivity derivation in the engine).
    pub fn for_each_state_ref(&self, f: &mut impl FnMut(&StateRef)) {
        if let Expr::StateActive { state } = self {
            f(state);
        }
        self.for_each_child(&mut |child| child.for_each_state_ref(f));
    }
}

/// A direct assignment `target := value` — the M0 form of a
/// sensitive-function *effect* (the mutation is declarative data).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assignment {
    /// The attribute being assigned.
    pub target: AttrRef,
    /// The value expression.
    pub value: Expr,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn sample_expr() -> Expr {
        // (comp_a.on == true) && (sum(comp_b.p_in) >= 2)
        Expr::Bool {
            bool_op: BoolOp::And,
            args: vec![
                Expr::Cmp {
                    cmp: CmpOp::Eq,
                    lhs: Box::new(Expr::attr("comp_a", "on")),
                    rhs: Box::new(Expr::bool(true)),
                },
                Expr::Cmp {
                    cmp: CmpOp::Ge,
                    lhs: Box::new(Expr::PortAgg {
                        port: PortRef {
                            component: "comp_b".into(),
                            port: "p_in".into(),
                        },
                        agg: AggOp::Sum,
                    }),
                    rhs: Box::new(Expr::Const {
                        value: Value::Int(2),
                    }),
                },
            ],
        }
    }

    #[test]
    fn serde_round_trip_preserves_tree() {
        let expr = sample_expr();
        let json = serde_json::to_string_pretty(&expr).unwrap();
        let back: Expr = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, back);
    }

    #[test]
    fn collects_var_and_port_refs() {
        let expr = sample_expr();
        let mut vars = Vec::new();
        expr.for_each_attr_ref(&mut |v| vars.push(v.clone()));
        assert_eq!(
            vars,
            vec![AttrRef {
                component: "comp_a".into(),
                attribute: "on".into()
            }]
        );
        let mut ports = Vec::new();
        expr.for_each_port_ref(&mut |p| ports.push(p.clone()));
        assert_eq!(ports.len(), 1);
        assert_eq!(ports[0].port, "p_in");
    }

    #[test]
    fn tagged_representation_is_stable() {
        // The wire format is part of the model-file contract: `op` tags.
        let json = serde_json::to_value(Expr::bool(true)).unwrap();
        assert_eq!(json["op"], "const");
        assert_eq!(json["value"]["kind"], "bool");
    }
}
