//! Affine decomposition of expression trees, with optional port resolution.

use crate::{AffineExpr, AggOp, AttrRef, Expr, NonlinearTerm, PortRef, Value};
use std::collections::HashSet;

impl Expr {
    /// Decompose this expression into an affine form in `variables`.
    /// Coefficients remain expressions over quantities outside that set.
    ///
    /// In-port aggregations are treated as **unknown-free leaves** here:
    /// no connection is resolved, so an aggregation never contributes a
    /// term whatever it aggregates. This is the form the decision
    /// programs use, whose inputs are settled at build time to be
    /// sweep-free. [`Expr::affine_in_resolved`] is the port-aware form
    /// the algebraic blocks need.
    pub fn affine_in(&self, variables: &HashSet<AttrRef>) -> Result<AffineExpr, NonlinearTerm> {
        let no_sources = |_: &PortRef, _: Option<&str>| Vec::new();
        self.affine_in_resolved(variables, &no_sources)
    }

    /// Decompose this expression into an affine form in `variables`,
    /// resolving every in-port aggregation to the attributes its
    /// connections carry: `sources` returns them **per connection, in
    /// connection declaration order**, `None` for the channel meaning
    /// the producer's single exported attribute.
    ///
    /// Two extensions over plain arithmetic decide affinity:
    ///
    /// - a conditional whose condition reads **no** selected variable is
    ///   affine, its branches becoming `if`-valued constants and
    ///   coefficients (a condition that reads one stays nonlinear);
    /// - an aggregation is resolved per operator: `sum` and `mean` are
    ///   affine in their unknown sources, `count` reads the topology and
    ///   never a value, and `median`, `all` and `any` are nonlinear.
    pub fn affine_in_resolved(
        &self,
        variables: &HashSet<AttrRef>,
        sources: &dyn Fn(&PortRef, Option<&str>) -> Vec<AttrRef>,
    ) -> Result<AffineExpr, NonlinearTerm> {
        if !self.reads_any_of(variables, sources) {
            return Ok(AffineExpr {
                constant: self.clone(),
                terms: Vec::new(),
            });
        }
        let zero = || Expr::Const {
            value: Value::Float(0.0),
        };
        let one = || Expr::Const {
            value: Value::Float(1.0),
        };
        match self {
            Expr::Attr { attr } => Ok(AffineExpr {
                constant: zero(),
                terms: vec![(attr.clone(), one())],
            }),
            Expr::Add { args } => {
                let mut constants = Vec::with_capacity(args.len());
                let mut terms = Vec::new();
                for arg in args {
                    let part = arg.affine_in_resolved(variables, sources)?;
                    constants.push(part.constant);
                    terms.extend(part.terms);
                }
                Ok(AffineExpr {
                    constant: Expr::Add { args: constants },
                    terms,
                })
            }
            Expr::Sub { lhs, rhs } => {
                let left = lhs.affine_in_resolved(variables, sources)?;
                let right = rhs.affine_in_resolved(variables, sources)?;
                let mut terms = left.terms;
                terms.extend(right.terms.into_iter().map(|(attr, coefficient)| {
                    (
                        attr,
                        Expr::Sub {
                            lhs: Box::new(zero()),
                            rhs: Box::new(coefficient),
                        },
                    )
                }));
                Ok(AffineExpr {
                    constant: Expr::Sub {
                        lhs: Box::new(left.constant),
                        rhs: Box::new(right.constant),
                    },
                    terms,
                })
            }
            Expr::Mul { args } => {
                let dependent: Vec<_> = args
                    .iter()
                    .filter(|arg| arg.reads_any_of(variables, sources))
                    .collect();
                if dependent.len() != 1 {
                    return Err(NonlinearTerm {
                        expression: self.clone(),
                    });
                }
                let part = dependent[0].affine_in_resolved(variables, sources)?;
                let free: Vec<_> = args
                    .iter()
                    .filter(|arg| !arg.reads_any_of(variables, sources))
                    .cloned()
                    .collect();
                let scaled = |value: Expr| {
                    let mut product = free.clone();
                    product.push(value);
                    Expr::Mul { args: product }
                };
                Ok(AffineExpr {
                    constant: scaled(part.constant),
                    terms: part
                        .terms
                        .into_iter()
                        .map(|(attr, coefficient)| (attr, scaled(coefficient)))
                        .collect(),
                })
            }
            Expr::Div { lhs, rhs } if !rhs.reads_any_of(variables, sources) => {
                let part = lhs.affine_in_resolved(variables, sources)?;
                let divide = |value| Expr::Div {
                    lhs: Box::new(value),
                    rhs: rhs.clone(),
                };
                Ok(AffineExpr {
                    constant: divide(part.constant),
                    terms: part
                        .terms
                        .into_iter()
                        .map(|(attr, coefficient)| (attr, divide(coefficient)))
                        .collect(),
                })
            }
            // A conditional whose condition never reads a selected
            // variable is a piecewise-affine expression: the condition
            // belongs to the coefficient, not to the nonlinearity. The
            // two branches may select different variables, so the terms
            // are the union, each branch's contribution gated by the
            // condition (the untaken side reading zero).
            Expr::If {
                cond,
                then,
                otherwise,
            } if !cond.reads_any_of(variables, sources) => {
                let positive = then.affine_in_resolved(variables, sources)?;
                let negative = otherwise.affine_in_resolved(variables, sources)?;
                let gated = |condition: &Expr, value| Expr::If {
                    cond: Box::new(condition.clone()),
                    then: Box::new(value),
                    otherwise: Box::new(zero()),
                };
                let mut terms: Vec<(AttrRef, Expr)> = positive
                    .terms
                    .into_iter()
                    .map(|(attr, coefficient)| (attr, gated(cond, coefficient)))
                    .collect();
                terms.extend(negative.terms.into_iter().map(|(attr, coefficient)| {
                    let value = Expr::If {
                        cond: Box::new(cond.as_ref().clone()),
                        then: Box::new(zero()),
                        otherwise: Box::new(coefficient),
                    };
                    (attr, value)
                }));
                Ok(AffineExpr {
                    constant: Expr::If {
                        cond: Box::new(cond.as_ref().clone()),
                        then: Box::new(positive.constant),
                        otherwise: Box::new(negative.constant),
                    },
                    terms,
                })
            }
            Expr::PortAgg { port, agg, channel } => {
                // `count` reads the topology, never a value: no resolved
                // source can make it depend on a selected variable (and
                // the early exit above already returned for it).
                let connected = sources(port, channel.as_deref());
                let unknowns: Vec<&AttrRef> = connected
                    .iter()
                    .filter(|attr| variables.contains(*attr))
                    .collect();
                match (*agg, unknowns.as_slice()) {
                    (_, []) => Ok(AffineExpr {
                        constant: self.clone(),
                        terms: Vec::new(),
                    }),
                    (AggOp::Sum | AggOp::Mean, unknowns) => {
                        // One term per unknown connection, the free
                        // connections folding into the constant as an
                        // ordinary sum. `mean` divides the whole
                        // read by the connection count, so each
                        // coefficient carries 1/n.
                        let n = connected.len() as f64;
                        let scale = if *agg == AggOp::Mean { 1.0 / n } else { 1.0 };
                        let terms = unknowns
                            .iter()
                            .map(|attr| {
                                (
                                    (*attr).clone(),
                                    Expr::Const {
                                        value: Value::Float(scale),
                                    },
                                )
                            })
                            .collect();
                        let free: Vec<Expr> = connected
                            .iter()
                            .filter(|attr| !variables.contains(*attr))
                            .map(|attr| Expr::Attr {
                                attr: (*attr).clone(),
                            })
                            .collect();
                        let free = match free.len() {
                            0 => zero(),
                            1 => free.into_iter().next().unwrap_or_else(zero),
                            _ => Expr::Add { args: free },
                        };
                        let constant = if *agg == AggOp::Mean {
                            Expr::Div {
                                lhs: Box::new(free),
                                rhs: Box::new(Expr::Const {
                                    value: Value::Float(n),
                                }),
                            }
                        } else {
                            free
                        };
                        Ok(AffineExpr { constant, terms })
                    }
                    _ => Err(NonlinearTerm {
                        expression: self.clone(),
                    }),
                }
            }
            _ => Err(NonlinearTerm {
                expression: self.clone(),
            }),
        }
    }

    /// Whether this tree reads one of the selected attributes.
    #[must_use]
    pub fn contains_any(&self, variables: &HashSet<AttrRef>) -> bool {
        let mut found = false;
        self.for_each_attr_ref(&mut |attr| found |= variables.contains(attr));
        found
    }

    /// [`Expr::contains_any`] with in-port aggregations resolved: the
    /// aggregation reads what its connections carry, so a selected
    /// variable behind a port counts. `count` reads the topology alone
    /// and never selects.
    fn reads_any_of(
        &self,
        variables: &HashSet<AttrRef>,
        sources: &dyn Fn(&PortRef, Option<&str>) -> Vec<AttrRef>,
    ) -> bool {
        match self {
            Expr::PortAgg { port, agg, channel } => {
                *agg != AggOp::Count
                    && sources(port, channel.as_deref())
                        .iter()
                        .any(|attr| variables.contains(attr))
            }
            _ => {
                let mut found = self.contains_any(variables);
                if !found {
                    // A direct attribute read is covered above; what may
                    // hide a selected variable here is an aggregation.
                    self.for_each_child(&mut |child| {
                        found |= child.reads_any_of(variables, sources);
                    });
                }
                found
            }
        }
    }
}
