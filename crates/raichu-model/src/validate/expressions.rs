//! Validation of connections, expressions and indicators.

use crate::validate::Scope;
use crate::{AttrKind, Connection, Distrib, IndicatorTarget, Model, ModelError, Port, PortDir};
use raichu_expr::{AttrRef, CmpOp, Expr, PortRef, Value};
use std::collections::{HashMap, HashSet};

/// Flow family of a value kind: a boolean carries a **discrete** flow
/// (present or absent), an integer or a float a **continuous** quantity.
/// The two never share one aggregation (see
/// [`ModelError::ConnectionFamilyMismatch`]).
fn is_discrete(kind: AttrKind) -> bool {
    matches!(kind, AttrKind::Bool)
}

impl Model {
    pub(crate) fn check_connections(
        &self,
        scopes: &HashMap<&str, Scope<'_>>,
    ) -> Result<(), ModelError> {
        for connection in &self.connections {
            let from = Self::resolve_port(scopes, &connection.from, "from")?;
            let to = Self::resolve_port(scopes, &connection.to, "to")?;
            if from.dir != PortDir::Out || to.dir != PortDir::In {
                return Err(ModelError::ConnectionDirectionMismatch {
                    from_component: connection.from.component.clone(),
                    from_port: connection.from.port.clone(),
                    to_component: connection.to.component.clone(),
                    to_port: connection.to.port.clone(),
                });
            }
        }
        self.check_flow_families(scopes)
    }

    /// Refuse an in port fed by both a **discrete** flow and a
    /// **continuous** one (see [`ModelError::ConnectionFamilyMismatch`]).
    ///
    /// The family is read off the kind of the attribute each producer
    /// exports, which is the only family the model layer carries: `Bool`
    /// is discrete, `Int` and `Float` are quantities. A single producer
    /// is never refused, whatever its kind: what has no defensible
    /// answer is the *mixture*, because one aggregation then has to give
    /// one meaning to two families.
    ///
    /// Reports the first offending connection in declaration order,
    /// against the first producer of the same in port.
    fn check_flow_families(&self, scopes: &HashMap<&str, Scope<'_>>) -> Result<(), ModelError> {
        // in port → the first connection feeding it and the kind it carries.
        let mut first: HashMap<(&str, &str), (&Connection, AttrKind)> = HashMap::new();
        for connection in &self.connections {
            let Some(kind) = Self::exported_kind(scopes, &connection.from) else {
                continue;
            };
            let key = (
                connection.to.component.as_str(),
                connection.to.port.as_str(),
            );
            if let Some(&(other, other_kind)) = first.get(&key) {
                if is_discrete(other_kind) != is_discrete(kind) {
                    return Err(ModelError::ConnectionFamilyMismatch {
                        port: format!("{}.{}", connection.to.component, connection.to.port),
                        other: format!("{}.{}", other.from.component, other.from.port),
                        other_kind,
                        producer: format!("{}.{}", connection.from.component, connection.from.port),
                        kind,
                    });
                }
            } else {
                first.insert(key, (connection, kind));
            }
        }
        Ok(())
    }

    /// Kind of the attribute an out port exports, `None` when the port
    /// or its attribute does not resolve (reported by the other passes).
    fn exported_kind(scopes: &HashMap<&str, Scope<'_>>, port: &PortRef) -> Option<AttrKind> {
        let scope = scopes.get(port.component.as_str())?;
        let attribute = scope.ports.get(port.port.as_str())?.attr.as_ref()?;
        scope.attribute_kind(attribute)
    }

    fn resolve_port<'m>(
        scopes: &'m HashMap<&str, Scope<'m>>,
        port_ref: &PortRef,
        side: &'static str,
    ) -> Result<&'m Port, ModelError> {
        scopes
            .get(port_ref.component.as_str())
            .and_then(|scope| scope.ports.get(port_ref.port.as_str()))
            .copied()
            .ok_or_else(|| ModelError::ConnectionUnknownPort {
                component: port_ref.component.clone(),
                port: port_ref.port.clone(),
                side,
            })
    }

    pub(crate) fn check_expressions(
        &self,
        scopes: &HashMap<&str, Scope<'_>>,
    ) -> Result<(), ModelError> {
        let sources = self.in_port_sources();
        for component in &self.components {
            for automaton in &component.automata {
                for transition in &automaton.transitions {
                    if let Some(guard) = &transition.guard {
                        let context = format!(
                            "guard of transition `{}` in `{}.{}`",
                            transition.name, component.name, automaton.name
                        );
                        Self::check_expr(scopes, &sources, guard, &context)?;
                    }
                    if let Distrib::Exp {
                        rate_expr: Some(rate),
                        ..
                    } = &transition.distrib
                    {
                        let context = format!(
                            "rate of transition `{}` in `{}.{}`",
                            transition.name, component.name, automaton.name
                        );
                        Self::check_expr(scopes, &sources, rate, &context)?;
                    }
                    for (index, assignment) in transition.effects.iter().enumerate() {
                        let context = format!(
                            "effect #{index} of transition `{}` in `{}.{}`",
                            transition.name, component.name, automaton.name
                        );
                        Self::check_attr_ref(scopes, &assignment.target, &context)?;
                        Self::check_expr(scopes, &sources, &assignment.value, &context)?;
                    }
                }
            }
            for function in &component.propagations {
                for (index, assignment) in function.effects.iter().enumerate() {
                    let context = format!(
                        "effect #{index} of propagation `{}.{}`",
                        component.name, function.name
                    );
                    Self::check_attr_ref(scopes, &assignment.target, &context)?;
                    Self::check_expr(scopes, &sources, &assignment.value, &context)?;
                }
            }
            for equation in &component.equations {
                let context = format!("equation of `{}.{}`", component.name, equation.target);
                Self::check_expr(scopes, &sources, &equation.expr, &context)?;
            }
            for allocation in &component.allocations {
                let context = format!(
                    "available quantity of distribution operator `{}.{}`",
                    component.name, allocation.name
                );
                Self::check_expr(scopes, &sources, &allocation.available, &context)?;
            }
        }
        Ok(())
    }

    /// (in-port) → out-ports feeding it, in connection declaration order.
    /// Used to check a `port_agg` channel selector against what the
    /// producers on the other end of the edges actually declare.
    pub(crate) fn in_port_sources(&self) -> HashMap<(String, String), Vec<(String, String)>> {
        let mut sources: HashMap<(String, String), Vec<(String, String)>> = HashMap::new();
        for connection in &self.connections {
            sources
                .entry((connection.to.component.clone(), connection.to.port.clone()))
                .or_default()
                .push((
                    connection.from.component.clone(),
                    connection.from.port.clone(),
                ));
        }
        sources
    }

    pub(crate) fn check_expr(
        scopes: &HashMap<&str, Scope<'_>>,
        sources: &HashMap<(String, String), Vec<(String, String)>>,
        expr: &Expr,
        context: &str,
    ) -> Result<(), ModelError> {
        if let Expr::Bool { bool_op, args } = expr {
            if matches!(bool_op, raichu_expr::BoolOp::Not) && args.len() != 1 {
                return Err(ModelError::NotArity {
                    context: context.to_owned(),
                    args: args.len(),
                });
            }
        }
        match expr {
            Expr::Const { .. } => Ok(()),
            Expr::Attr { attr } => Self::check_attr_ref(scopes, attr, context),
            Expr::PortAgg { port, channel, .. } => {
                let resolved = scopes
                    .get(port.component.as_str())
                    .and_then(|scope| scope.ports.get(port.port.as_str()));
                match resolved {
                    Some(p) if p.dir == PortDir::In => {}
                    _ => {
                        return Err(ModelError::ExprBadPortAgg {
                            context: context.to_owned(),
                            component: port.component.clone(),
                            port: port.port.clone(),
                        });
                    }
                }
                let Some(channel) = channel else {
                    return Ok(());
                };
                // A channel selector reads the per-connection attributes
                // materialised for it, so every producer on the other end
                // of an edge must declare it. An in port with *no*
                // connection aggregates the empty set and stays legal:
                // no-connection defaults are relied on throughout.
                let edges = sources.get(&(port.component.clone(), port.port.clone()));
                for (from_component, from_port) in edges.into_iter().flatten() {
                    let declares = scopes
                        .get(from_component.as_str())
                        .and_then(|scope| scope.ports.get(from_port.as_str()))
                        .is_some_and(|source| source.channels.iter().any(|c| &c.name == channel));
                    if !declares {
                        return Err(ModelError::ExprUnknownPortChannel {
                            context: context.to_owned(),
                            port: format!("{}.{}", port.component, port.port),
                            channel: channel.clone(),
                            producer: format!("{from_component}.{from_port}"),
                        });
                    }
                }
                Ok(())
            }
            Expr::StateActive { state } => {
                let known = scopes
                    .get(state.component.as_str())
                    .and_then(|scope| scope.automata.get(state.automaton.as_str()))
                    .is_some_and(|states| states.contains(state.state.as_str()));
                if known {
                    Ok(())
                } else {
                    Err(ModelError::ExprUnknownState {
                        context: context.to_owned(),
                        component: state.component.clone(),
                        automaton: state.automaton.clone(),
                        state: state.state.clone(),
                    })
                }
            }
            Expr::Cmp { lhs, rhs, .. } | Expr::Sub { lhs, rhs } | Expr::Div { lhs, rhs } => {
                Self::check_expr(scopes, sources, lhs, context)?;
                Self::check_expr(scopes, sources, rhs, context)
            }
            Expr::Bool { args, .. } => {
                for arg in args {
                    Self::check_expr(scopes, sources, arg, context)?;
                }
                Ok(())
            }
            Expr::Add { args } | Expr::Mul { args } | Expr::Min { args } | Expr::Max { args } => {
                if args.is_empty() {
                    return Err(ModelError::EmptyExprArgs {
                        context: context.to_owned(),
                        op: match expr {
                            Expr::Add { .. } => "add",
                            Expr::Mul { .. } => "mul",
                            Expr::Min { .. } => "min",
                            _ => "max",
                        },
                    });
                }
                for arg in args {
                    Self::check_expr(scopes, sources, arg, context)?;
                }
                Ok(())
            }
            Expr::If {
                cond,
                then,
                otherwise,
            } => {
                Self::check_expr(scopes, sources, cond, context)?;
                Self::check_expr(scopes, sources, then, context)?;
                Self::check_expr(scopes, sources, otherwise, context)
            }
            Expr::Sin { arg } | Expr::Exp { arg } => {
                Self::check_expr(scopes, sources, arg, context)
            }
            Expr::PiecewiseLinear {
                arg,
                points,
                period,
                phase_shift,
            } => {
                raichu_expr::signal::check_periodic_piecewise_linear(points, *period, *phase_shift)
                    .map_err(|detail| ModelError::InvalidTimeSignal {
                        context: context.to_owned(),
                        op: "piecewise_linear",
                        detail,
                    })?;
                Self::check_expr(scopes, sources, arg, context)
            }
            Expr::SmoothNoise {
                arg,
                amplitude,
                time_scale,
                ..
            } => {
                raichu_expr::signal::check_smooth_noise(*amplitude, *time_scale).map_err(
                    |detail| ModelError::InvalidTimeSignal {
                        context: context.to_owned(),
                        op: "smooth_noise",
                        detail,
                    },
                )?;
                Self::check_expr(scopes, sources, arg, context)
            }
            Expr::Time => Ok(()),
        }
    }

    fn check_attr_ref(
        scopes: &HashMap<&str, Scope<'_>>,
        var_ref: &AttrRef,
        context: &str,
    ) -> Result<(), ModelError> {
        let known = scopes
            .get(var_ref.component.as_str())
            .is_some_and(|scope| scope.attribute_kind(&var_ref.attribute).is_some());
        if known {
            Ok(())
        } else {
            Err(ModelError::ExprUnknownVariable {
                context: context.to_owned(),
                component: var_ref.component.clone(),
                attribute: var_ref.attribute.clone(),
            })
        }
    }

    pub(crate) fn check_indicators(
        &self,
        scopes: &HashMap<&str, Scope<'_>>,
    ) -> Result<(), ModelError> {
        let mut names = HashSet::new();
        for indicator in &self.indicators {
            if !names.insert(indicator.name.as_str()) {
                return Err(ModelError::DuplicateIndicator {
                    name: indicator.name.clone(),
                });
            }
            match &indicator.target {
                IndicatorTarget::Attribute { attr } => {
                    let context = format!("target of indicator `{}`", indicator.name);
                    Self::check_attr_ref(scopes, attr, &context).map_err(|_| {
                        ModelError::IndicatorUnresolved {
                            indicator: indicator.name.clone(),
                            detail: format!(
                                "unknown attribute `{}.{}`",
                                attr.component, attr.attribute
                            ),
                        }
                    })?;
                }
                IndicatorTarget::State {
                    component,
                    automaton,
                    state,
                } => {
                    let found = self
                        .components
                        .iter()
                        .find(|c| &c.name == component)
                        .and_then(|c| c.automata.iter().find(|a| &a.name == automaton))
                        .is_some_and(|a| a.states.contains(state));
                    if !found {
                        return Err(ModelError::IndicatorUnresolved {
                            indicator: indicator.name.clone(),
                            detail: format!("unknown state `{component}.{automaton}.{state}`"),
                        });
                    }
                }
                IndicatorTarget::Predicate { attr, cmp, value } => {
                    let kind = scopes
                        .get(attr.component.as_str())
                        .and_then(|scope| scope.attribute_kind(&attr.attribute))
                        .ok_or_else(|| ModelError::IndicatorUnresolved {
                            indicator: indicator.name.clone(),
                            detail: format!(
                                "unknown attribute `{}.{}`",
                                attr.component, attr.attribute
                            ),
                        })?;
                    Self::check_predicate_kinds(&indicator.name, attr, kind, *cmp, value)?;
                }
            }
        }
        // scopes is only used through the helpers above; keep the
        // signature symmetric with the other passes.
        let _ = scopes;
        Ok(())
    }

    /// Refuse an indicator threshold whose two sides do not compare.
    ///
    /// The engine's own [`raichu_expr::CmpOp`] semantics are the rule
    /// being enforced ahead of time: booleans compare for equality only,
    /// numbers compare across `int`/`float`, and the two families do not
    /// mix. Doing it here rather than at record time is what lets the
    /// three recording sites stay infallible, and it is also the more
    /// useful moment: a threshold that can never be evaluated is a
    /// modelling mistake, not a run-time event.
    fn check_predicate_kinds(
        indicator: &str,
        attr: &AttrRef,
        kind: AttrKind,
        cmp: CmpOp,
        value: &Value,
    ) -> Result<(), ModelError> {
        let numeric_attr = matches!(kind, AttrKind::Int | AttrKind::Float);
        let numeric_value = matches!(value, Value::Int(_) | Value::Float(_));
        if numeric_attr != numeric_value {
            return Err(ModelError::IndicatorPredicateKind {
                indicator: indicator.to_owned(),
                detail: format!(
                    "the threshold compares `{}.{}`, declared {kind:?}, with {value:?}: a \
                     boolean and a number have no comparison",
                    attr.component, attr.attribute
                ),
            });
        }
        if !numeric_attr && !matches!(cmp, CmpOp::Eq | CmpOp::Ne) {
            return Err(ModelError::IndicatorPredicateKind {
                indicator: indicator.to_owned(),
                detail: format!(
                    "the threshold orders `{}.{}` with {cmp:?}, and it is declared {kind:?}: a \
                     boolean is tested for equality (`eq`, `ne`) and never ordered",
                    attr.component, attr.attribute
                ),
            });
        }
        Ok(())
    }
}
