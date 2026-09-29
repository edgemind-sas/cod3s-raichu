//! # raichu-expr: serializable expression trees
//!
//! Guards, sensitive-function effects and (from milestone M1) ODE
//! right-hand sides are **pure data**: expression trees serialized inside
//! the model file and evaluated/compiled on the Rust side (SymPy/CasADi
//! style). This is what keeps the engine free of Python callbacks on the
//! hot path (Performance contract) while keeping models fully
//! serializable.
//!
//! M0 subset (frozen: see the M0 plan): constant/attribute leaves,
//! comparison and boolean operators, port aggregation (sum/count/all/any),
//! direct assignment. Arithmetic and math functions arrive in M1; the
//! [`Expr`] enum is `#[non_exhaustive]`-in-spirit (tagged serde repr) so
//! extension does not break serialized models.
//!
//! An optional Rust trait escape hatch for compiled custom behaviour is
//! part of the design (reserved API), not exercised in M0.

use serde::{Deserialize, Serialize};
mod affine;

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

/// Reference to an attribute by hierarchical name: the *authoring /
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

/// An expression separated into a decision-free constant and one coefficient
/// expression for each occurrence of a decision variable.
#[derive(Debug, Clone, PartialEq)]
pub struct AffineExpr {
    /// The part independent of all selected variables.
    pub constant: Expr,
    /// Coefficients and their corresponding selected variables. Repeated
    /// variables may occur and their contributions must be summed by callers.
    pub terms: Vec<(AttrRef, Expr)>,
}

/// A term that cannot be affine in the selected variables.
#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearTerm {
    /// The offending expression, useful for a model diagnostic.
    pub expression: Expr,
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
    /// averaging: cod3s `compute_reference_mean`). 0.0 with no
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
        /// Optional **channel selector**. Absent (the default) reads the
        /// attribute each connected out-port exports: its single, shared
        /// value. Present, it reads instead the per-connection quantity
        /// the compiler materialises for that channel, so two consumers
        /// on the same out-port receive different numbers (the
        /// conservative-flow shape). Every connected out-port must
        /// declare the named channel; validation refuses otherwise.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        channel: Option<String>,
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
    /// N-ary minimum (clamping: `min(max(lo, v), hi)` patterns).
    Min {
        /// Operands (≥ 1, validated at model build).
        args: Vec<Expr>,
    },
    /// N-ary maximum.
    Max {
        /// Operands (≥ 1, validated at model build).
        args: Vec<Expr>,
    },
    /// Conditional expression: with [`Expr::StateActive`] as condition
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
    /// Natural exponential `e^arg` (state-dependent failure rates:
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
    /// the reference visitors below: new variants only need a case
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
    /// to derive sensitivity sets: which attribute changes must re-trigger
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

    /// Visit each port aggregate with its optional channel selector.
    pub fn for_each_port_agg(&self, f: &mut impl FnMut(&PortRef, Option<&str>)) {
        if let Expr::PortAgg { port, channel, .. } = self {
            f(port, channel.as_deref());
        }
        self.for_each_child(&mut |child| child.for_each_port_agg(f));
    }

    /// Visit every automaton-state reference in the tree (model
    /// validation; state-sensitivity derivation in the engine).
    pub fn for_each_state_ref(&self, f: &mut impl FnMut(&StateRef)) {
        if let Expr::StateActive { state } = self {
            f(state);
        }
        self.for_each_child(&mut |child| child.for_each_state_ref(f));
    }

    /// Render this tree as compact mathematical notation, for
    /// diagnostics that must name a term the way the modeller wrote it.
    ///
    /// This is a rendering, not a parseable form: parenthesisation is
    /// minimal and the serialised JSON remains the wire format.
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            Expr::Const { value } => match value {
                Value::Bool(b) => b.to_string(),
                Value::Int(i) => i.to_string(),
                Value::Float(f) => {
                    let rendered = format!("{f}");
                    if rendered.contains(['.', 'e', 'E', 'n', 'i']) {
                        rendered
                    } else {
                        // A float that formats integrally: keep the kind
                        // readable rather than `5` for `5.0`.
                        format!("{rendered}.0")
                    }
                }
            },
            Expr::Attr { attr } => format!("{}.{}", attr.component, attr.attribute),
            Expr::PortAgg { port, agg, channel } => {
                let operator = match agg {
                    AggOp::Sum => "sum",
                    AggOp::Count => "count",
                    AggOp::All => "all",
                    AggOp::Any => "any",
                    AggOp::Mean => "mean",
                    AggOp::Median => "median",
                };
                match channel {
                    Some(channel) => {
                        format!("{operator}({}.{}[{channel}])", port.component, port.port)
                    }
                    None => format!("{operator}({}.{})", port.component, port.port),
                }
            }
            Expr::Cmp { cmp, lhs, rhs } => {
                let operator = match cmp {
                    CmpOp::Eq => "==",
                    CmpOp::Ne => "!=",
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                    CmpOp::Gt => ">",
                    CmpOp::Ge => ">=",
                };
                format!("({} {} {})", lhs.render(), operator, rhs.render())
            }
            Expr::Bool { bool_op, args } => {
                let operator = match bool_op {
                    BoolOp::And => " and ",
                    BoolOp::Or => " or ",
                    BoolOp::Not => {
                        return match args.first() {
                            Some(operand) => format!("not {}", operand.render()),
                            None => "not <missing operand>".to_owned(),
                        };
                    }
                };
                let rendered: Vec<String> = args.iter().map(Expr::render).collect();
                format!("({})", rendered.join(operator))
            }
            Expr::StateActive { state } => {
                format!("{}.{}.{}", state.component, state.automaton, state.state)
            }
            Expr::Add { args } => {
                let rendered: Vec<String> = args.iter().map(Expr::render).collect();
                format!("({})", rendered.join(" + "))
            }
            Expr::Sub { lhs, rhs } => format!("({} - {})", lhs.render(), rhs.render()),
            Expr::Mul { args } => {
                let rendered: Vec<String> = args.iter().map(Expr::render).collect();
                format!("({})", rendered.join(" * "))
            }
            Expr::Div { lhs, rhs } => format!("({} / {})", lhs.render(), rhs.render()),
            Expr::Min { args } | Expr::Max { args } => {
                let operator = if matches!(self, Expr::Min { .. }) {
                    "min"
                } else {
                    "max"
                };
                let rendered: Vec<String> = args.iter().map(Expr::render).collect();
                format!("{operator}({})", rendered.join(", "))
            }
            Expr::If {
                cond,
                then,
                otherwise,
            } => format!(
                "if {} then {} else {}",
                cond.render(),
                then.render(),
                otherwise.render()
            ),
            Expr::Sin { arg } => format!("sin({})", arg.render()),
            Expr::Exp { arg } => format!("exp({})", arg.render()),
            Expr::Time => "time".to_owned(),
        }
    }
}

/// A direct assignment `target := value`: the M0 form of a
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
    use std::collections::HashSet;

    #[test]
    fn affine_decomposition_distinguishes_coefficients_from_decisions() {
        let x = AttrRef {
            component: "c".into(),
            attribute: "x".into(),
        };
        let selected = HashSet::from([x.clone()]);
        let coefficient = Expr::attr("c", "cost");
        let linear = Expr::Add {
            args: vec![
                Expr::Mul {
                    args: vec![coefficient.clone(), Expr::Attr { attr: x.clone() }],
                },
                Expr::Const {
                    value: Value::Float(3.0),
                },
            ],
        };
        let affine = linear.affine_in(&selected).unwrap();
        assert_eq!(affine.terms.len(), 1);
        assert_eq!(affine.terms[0].0, x);
        assert!(affine.terms[0].1.contains_any(&HashSet::from([AttrRef {
            component: "c".into(),
            attribute: "cost".into(),
        }])));
        assert!(!affine.terms[0].1.contains_any(&selected));

        let quadratic = Expr::Mul {
            args: vec![Expr::attr("c", "x"), Expr::attr("c", "x")],
        };
        assert!(quadratic.affine_in(&selected).is_err());
        let reciprocal = Expr::Div {
            lhs: Box::new(Expr::attr("c", "cost")),
            rhs: Box::new(Expr::attr("c", "x")),
        };
        assert!(reciprocal.affine_in(&selected).is_err());
    }

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
                        channel: None,
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

    // ---- Conditionals and resolved port aggregations ------------------

    fn selected(name: &str) -> HashSet<AttrRef> {
        HashSet::from([AttrRef {
            component: "c".into(),
            attribute: name.into(),
        }])
    }

    fn port_agg(port: &str, agg: AggOp) -> Expr {
        Expr::PortAgg {
            port: PortRef {
                component: "c".into(),
                port: port.into(),
            },
            agg,
            channel: None,
        }
    }

    /// Resolution used by the tests below: `p_in` reads one selected
    /// unknown (`c.x`) and one free attribute (`c.e`).
    fn resolve(p: &PortRef, _channel: Option<&str>) -> Vec<AttrRef> {
        assert_eq!(p.port, "p_in");
        vec![
            AttrRef {
                component: "c".into(),
                attribute: "x".into(),
            },
            AttrRef {
                component: "c".into(),
                attribute: "e".into(),
            },
        ]
    }

    #[test]
    fn conditional_over_non_unknowns_is_affine_in_both_branches() {
        // if(c.on) then 2*x else x + x: affine either way, the branch
        // difference riding on the condition.
        let condition = Expr::attr("c", "on");
        let expr = Expr::If {
            cond: Box::new(condition.clone()),
            then: Box::new(Expr::Mul {
                args: vec![
                    Expr::Const {
                        value: Value::Float(2.0),
                    },
                    Expr::attr("c", "x"),
                ],
            }),
            otherwise: Box::new(Expr::Add {
                args: vec![Expr::attr("c", "x"), Expr::attr("c", "x")],
            }),
        };
        let form = expr.affine_in_resolved(&selected("x"), &resolve).unwrap();
        // One term per occurrence: the else branch contributes two (its
        // own `x + x`), callers sum them.
        assert_eq!(form.terms.len(), 3);
        assert!(form.terms.iter().all(|(attr, _)| attr.attribute == "x"));
        // The constant carries the condition; both coefficients are
        // gated by it.
        let rendered = form.constant.render();
        assert!(rendered.starts_with("if "), "constant: {rendered}");
        for (_, coefficient) in &form.terms {
            assert!(coefficient.render().starts_with("if "));
        }

        // The same conditional with the unknown in the condition stays
        // refused, naming the conditional itself.
        let refused = Expr::If {
            cond: Box::new(Expr::Cmp {
                cmp: CmpOp::Lt,
                lhs: Box::new(Expr::attr("c", "x")),
                rhs: Box::new(Expr::Const {
                    value: Value::Float(0.0),
                }),
            }),
            then: Box::new(Expr::attr("c", "x")),
            otherwise: Box::new(Expr::Const {
                value: Value::Float(0.0),
            }),
        };
        let term = refused
            .affine_in_resolved(&selected("x"), &resolve)
            .unwrap_err();
        assert_eq!(term.expression, refused);
    }

    #[test]
    fn port_aggregations_classify_per_operator() {
        let unknowns = selected("x");
        // sum: affine, one term per unknown connection, free ones in the
        // constant.
        let sum = port_agg("p_in", AggOp::Sum);
        let form = sum.affine_in_resolved(&unknowns, &resolve).unwrap();
        assert_eq!(
            form.terms,
            vec![(
                AttrRef {
                    component: "c".into(),
                    attribute: "x".into()
                },
                Expr::Const {
                    value: Value::Float(1.0)
                },
            )]
        );
        assert_eq!(form.constant, Expr::attr("c", "e"));

        // mean: the same terms at 1/n, the constant divided by n.
        let mean = port_agg("p_in", AggOp::Mean);
        let form = mean.affine_in_resolved(&unknowns, &resolve).unwrap();
        assert_eq!(
            form.terms[0].1,
            Expr::Const {
                value: Value::Float(0.5)
            }
        );
        assert!(matches!(form.constant, Expr::Div { .. }));

        // count reads the topology alone: constant even over unknowns.
        let count = port_agg("p_in", AggOp::Count);
        let form = count.affine_in_resolved(&unknowns, &resolve).unwrap();
        assert!(form.terms.is_empty());
        assert_eq!(form.constant, count);

        // median is nonlinear in its unknown sources.
        let median = port_agg("p_in", AggOp::Median);
        assert!(median.affine_in_resolved(&unknowns, &resolve).is_err());
        // ... and constant once no source is selected.
        assert!(median
            .affine_in_resolved(&selected("y"), &resolve)
            .unwrap()
            .terms
            .is_empty());
    }

    #[test]
    fn render_handles_a_missing_not_operand() {
        let missing = Expr::Bool {
            bool_op: BoolOp::Not,
            args: Vec::new(),
        };
        assert_eq!(missing.render(), "not <missing operand>");
        let valid = Expr::Bool {
            bool_op: BoolOp::Not,
            args: vec![Expr::bool(true)],
        };
        assert_eq!(valid.render(), "not true");
    }

    #[test]
    fn render_reads_as_written() {
        let expr = Expr::Div {
            lhs: Box::new(Expr::Sub {
                lhs: Box::new(Expr::attr("r", "v1")),
                rhs: Box::new(Expr::attr("r", "v2")),
            }),
            rhs: Box::new(Expr::Const {
                value: Value::Float(2.0),
            }),
        };
        assert_eq!(expr.render(), "((r.v1 - r.v2) / 2.0)");
        let min = Expr::Min {
            args: vec![
                Expr::attr("c", "x"),
                Expr::Const {
                    value: Value::Float(1.0),
                },
            ],
        };
        assert_eq!(min.render(), "min(c.x, 1.0)");
    }
}
