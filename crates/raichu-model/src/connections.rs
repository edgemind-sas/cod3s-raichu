//! Connection resolution: interface connections expanded, port constructs lowered.

#[cfg(doc)]
use crate::PortEquation;
use crate::{
    channel_attribute_name, Connection, Distrib, Equation, EquationKind, InterfaceConnection,
    InterfaceRef, Model, ModelError, PortDir, EDGE, PORT,
};
use raichu_expr::{AggOp, Assignment, AttrRef, Expr, PortRef, Value};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

impl Model {
    /// Every port connection the model runs on: the explicit
    /// [`Model::connections`] in declaration order, followed by the
    /// expansion of each [`InterfaceConnection`] in declaration order.
    ///
    /// Borrowed, at no cost, when the model declares no interface
    /// connection. An interface connection is expanded port by port in
    /// the order the `from` interface lists its ports, then the `to`
    /// interface's ports that point back: an out port of `from` becomes
    /// `from → to`, an in port of `from` becomes `to → from`.
    ///
    /// # Errors
    ///
    /// The interface does not exist ([`ModelError::InterfaceConnectionUnknown`]),
    /// or the two interfaces do not pair exactly by port name
    /// ([`ModelError::InterfacePortUnmatched`],
    /// [`ModelError::InterfacePortDirection`]). A port an interface lists
    /// but its component lacks is reported by [`Model::validate`] as
    /// [`ModelError::InterfaceUnknownPort`].
    pub fn resolved_connections(&self) -> Result<Cow<'_, [Connection]>, ModelError> {
        if self.interface_connections.is_empty() {
            return Ok(Cow::Borrowed(&self.connections));
        }
        let mut out = self.connections.clone();
        for link in &self.interface_connections {
            out.extend(self.expand_interface_connection(link)?);
        }
        Ok(Cow::Owned(out))
    }

    /// This model with its authoring forms resolved: interface connections
    /// expanded into [`Model::connections`] and the list emptied, then port
    /// equations and out-port aggregations lowered to explicit equations
    /// ([`PortEquation`]). The model the engine actually runs. Borrowed
    /// when there is nothing to resolve.
    ///
    /// # Errors
    ///
    /// As [`Model::resolved_connections`], and the port-equation refusals
    /// ([`ModelError::PortEquationUnknownChannel`],
    /// [`ModelError::PortEquationUnknownEdgeChannel`],
    /// [`ModelError::OutPortAggregation`]).
    pub fn with_resolved_connections(&self) -> Result<Cow<'_, Model>, ModelError> {
        if !self.needs_resolution() {
            return Ok(Cow::Borrowed(self));
        }
        let connections = self.resolved_connections()?.into_owned();
        let mut model = Model {
            connections,
            interface_connections: Vec::new(),
            ..self.clone()
        };
        if model.has_port_constructs() {
            model.lower_port_constructs()?;
        }
        if model.has_port_effects() {
            model.lower_port_effects()?;
        }
        Ok(Cow::Owned(model))
    }

    /// Whether [`Model::with_resolved_connections`] has anything to do.
    pub(crate) fn needs_resolution(&self) -> bool {
        !self.interface_connections.is_empty()
            || self.has_port_constructs()
            || self.has_port_effects()
    }

    /// Whether an edge effect or a sensitive function writes through a
    /// port ([`PORT`]).
    pub(crate) fn has_port_effects(&self) -> bool {
        let through = |effects: &[Assignment]| effects.iter().any(|a| a.target.component == PORT);
        self.components.iter().any(|c| {
            c.automata
                .iter()
                .any(|a| a.transitions.iter().any(|t| through(&t.effects)))
                || c.sensitive_functions.iter().any(|f| through(&f.effects))
        })
    }

    /// Lower every edge effect written through a port to one ordinary edge
    /// effect per connection feeding that port, on the attribute the
    /// connected out port exports, in connection declaration order. The
    /// lowered effects take the place of the one they come from, so the
    /// order of a transition's effects is kept.
    fn lower_port_effects(&mut self) -> Result<(), ModelError> {
        let exported: HashMap<(&str, &str), &str> = self
            .components
            .iter()
            .flat_map(|c| {
                c.ports
                    .iter()
                    .filter(|p| p.dir == PortDir::Out)
                    .filter_map(move |p| {
                        p.attr
                            .as_deref()
                            .map(|attr| ((c.name.as_str(), p.name.as_str()), attr))
                    })
            })
            .collect();
        let mut sources: HashMap<(String, String), Vec<AttrRef>> = HashMap::new();
        for connection in &self.connections {
            let from = (
                connection.from.component.as_str(),
                connection.from.port.as_str(),
            );
            if let Some(attribute) = exported.get(&from) {
                sources
                    .entry((connection.to.component.clone(), connection.to.port.clone()))
                    .or_default()
                    .push(AttrRef {
                        component: connection.from.component.clone(),
                        attribute: (*attribute).to_owned(),
                    });
            }
        }
        for component in &mut self.components {
            if let Some(function) = component
                .sensitive_functions
                .iter()
                .find(|f| f.effects.iter().any(|a| a.target.component == PORT))
            {
                return Err(ModelError::PortEffectInSensitiveFunction {
                    function: format!("{}.{}", component.name, function.name),
                });
            }
            let in_ports: HashSet<&str> = component
                .ports
                .iter()
                .filter(|p| p.dir == PortDir::In)
                .map(|p| p.name.as_str())
                .collect();
            for automaton in &mut component.automata {
                for transition in &mut automaton.transitions {
                    if !transition
                        .effects
                        .iter()
                        .any(|a| a.target.component == PORT)
                    {
                        continue;
                    }
                    let mut lowered = Vec::with_capacity(transition.effects.len());
                    for assignment in transition.effects.drain(..) {
                        if assignment.target.component != PORT {
                            lowered.push(assignment);
                            continue;
                        }
                        let port = assignment.target.attribute;
                        if !in_ports.contains(port.as_str()) {
                            return Err(ModelError::PortEffectNotAnInPort {
                                transition: format!(
                                    "transition `{}` in `{}.{}`",
                                    transition.name, component.name, automaton.name
                                ),
                                port,
                            });
                        }
                        let key = (component.name.clone(), port);
                        for source in sources.get(&key).map(Vec::as_slice).unwrap_or_default() {
                            lowered.push(Assignment {
                                target: source.clone(),
                                value: assignment.value.clone(),
                            });
                        }
                    }
                    transition.effects = lowered;
                }
            }
        }
        Ok(())
    }

    /// Whether the model uses a port equation or aggregates an out port.
    pub(crate) fn has_port_constructs(&self) -> bool {
        if self
            .components
            .iter()
            .any(|c| c.ports.iter().any(|p| !p.equations.is_empty()))
        {
            return true;
        }
        let out_ports = self.out_ports();
        let mut found = false;
        self.for_each_expr(&mut |_, expr| {
            // Only a channel-named aggregation is the construct; one without
            // a channel stays the malformed read the validation refuses.
            expr.for_each_port_agg(&mut |port, channel| {
                found |= channel.is_some()
                    && out_ports.contains(&(port.component.as_str(), port.port.as_str()));
            });
        });
        found
    }

    fn out_ports(&self) -> HashSet<(&str, &str)> {
        self.components
            .iter()
            .flat_map(|c| {
                c.ports
                    .iter()
                    .filter(|p| p.dir == PortDir::Out)
                    .map(move |p| (c.name.as_str(), p.name.as_str()))
            })
            .collect()
    }

    /// Visit every expression an out-port aggregation may appear in, with
    /// a description of where it is.
    fn for_each_expr(&self, f: &mut impl FnMut(&dyn Fn() -> String, &Expr)) {
        for component in &self.components {
            for automaton in &component.automata {
                for transition in &automaton.transitions {
                    let here = || {
                        format!(
                            "transition `{}` in `{}.{}`",
                            transition.name, component.name, automaton.name
                        )
                    };
                    if let Some(guard) = &transition.guard {
                        f(&here, guard);
                    }
                    if let Distrib::Exp {
                        rate_expr: Some(rate),
                        ..
                    } = &transition.distrib
                    {
                        f(&here, rate);
                    }
                    for assignment in &transition.effects {
                        f(&here, &assignment.value);
                    }
                }
            }
            for function in &component.sensitive_functions {
                let here = || format!("sensitive function `{}.{}`", component.name, function.name);
                for assignment in &function.effects {
                    f(&here, &assignment.value);
                }
            }
            for equation in &component.equations {
                let here = || format!("equation of `{}.{}`", component.name, equation.target);
                f(&here, &equation.expr);
            }
            for allocation in &component.allocations {
                let here = || {
                    format!(
                        "available quantity of distribution operator `{}.{}`",
                        component.name, allocation.name
                    )
                };
                f(&here, &allocation.available);
            }
            for port in &component.ports {
                for equation in &port.equations {
                    let here = || {
                        format!(
                            "port equation `{}` of `{}.{}`",
                            equation.channel, component.name, port.name
                        )
                    };
                    f(&here, &equation.expr);
                }
            }
        }
    }

    /// Visit mutably the same expressions as [`Model::for_each_expr`]: one
    /// slot list, kept beside it so the two cannot drift apart.
    fn for_each_expr_mut(&mut self, f: &mut impl FnMut(&mut Expr)) {
        for component in &mut self.components {
            for automaton in &mut component.automata {
                for transition in &mut automaton.transitions {
                    if let Some(guard) = &mut transition.guard {
                        f(guard);
                    }
                    if let Distrib::Exp {
                        rate_expr: Some(rate),
                        ..
                    } = &mut transition.distrib
                    {
                        f(rate);
                    }
                    for assignment in &mut transition.effects {
                        f(&mut assignment.value);
                    }
                }
            }
            for function in &mut component.sensitive_functions {
                for assignment in &mut function.effects {
                    f(&mut assignment.value);
                }
            }
            for equation in &mut component.equations {
                f(&mut equation.expr);
            }
            for allocation in &mut component.allocations {
                f(&mut allocation.available);
            }
            for port in &mut component.ports {
                for equation in &mut port.equations {
                    f(&mut equation.expr);
                }
            }
        }
    }

    /// The connections through each port, by `(component, port)` of either
    /// end, in declaration order.
    fn connections_by_port(&self) -> HashMap<(&str, &str), Vec<&Connection>> {
        let mut through: HashMap<(&str, &str), Vec<&Connection>> = HashMap::new();
        for connection in &self.connections {
            through
                .entry((
                    connection.from.component.as_str(),
                    connection.from.port.as_str(),
                ))
                .or_default()
                .push(connection);
            through
                .entry((
                    connection.to.component.as_str(),
                    connection.to.port.as_str(),
                ))
                .or_default()
                .push(connection);
        }
        through
    }

    /// Lower every port equation to one explicit equation per connection,
    /// then every out-port aggregation to an explicit sum or count, in
    /// place. Port equations lower in component, port and equation
    /// declaration order, connections in declaration order; each lands on
    /// the producer, after its own equations.
    fn lower_port_constructs(&mut self) -> Result<(), ModelError> {
        let declared: HashMap<(&str, &str), Vec<&str>> = self
            .components
            .iter()
            .flat_map(|c| {
                c.ports
                    .iter()
                    .filter(|p| p.dir == PortDir::Out)
                    .map(move |p| {
                        (
                            (c.name.as_str(), p.name.as_str()),
                            p.channels.iter().map(|ch| ch.name.as_str()).collect(),
                        )
                    })
            })
            .collect();
        let through = self.connections_by_port();
        let index: HashMap<&str, usize> = self
            .components
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.as_str(), i))
            .collect();
        let mut lowered: Vec<(usize, Equation)> = Vec::new();
        for component in &self.components {
            for port in &component.ports {
                let here = format!("{}.{}", component.name, port.name);
                let connections = through
                    .get(&(component.name.as_str(), port.name.as_str()))
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .filter(|k| match port.dir {
                        PortDir::Out => {
                            k.from.component == component.name && k.from.port == port.name
                        }
                        PortDir::In => k.to.component == component.name && k.to.port == port.name,
                    });
                let connections: Vec<&Connection> = connections.copied().collect();
                for equation in &port.equations {
                    // An out port checks its own channels even when nothing
                    // is connected to it yet.
                    if port.dir == PortDir::Out
                        && !declared[&(component.name.as_str(), port.name.as_str())]
                            .contains(&equation.channel.as_str())
                    {
                        return Err(ModelError::PortEquationUnknownChannel {
                            port: here.clone(),
                            channel: equation.channel.clone(),
                            producer_port: here,
                        });
                    }
                    for connection in &connections {
                        let producer = (
                            connection.from.component.as_str(),
                            connection.from.port.as_str(),
                        );
                        let channels: &[&str] = declared.get(&producer).map_or(&[], Vec::as_slice);
                        let unknown = |channel: &str| !channels.contains(&channel);
                        let producer_port = || format!("{}.{}", producer.0, producer.1);
                        if unknown(&equation.channel) {
                            return Err(ModelError::PortEquationUnknownChannel {
                                port: here,
                                channel: equation.channel.clone(),
                                producer_port: producer_port(),
                            });
                        }
                        let mut expr = equation.expr.clone();
                        let mut missing: Option<String> = None;
                        expr.rewrite(&mut |node| {
                            if let Expr::Attr { attr } = node {
                                if attr.component == EDGE {
                                    if unknown(&attr.attribute) {
                                        missing.get_or_insert_with(|| attr.attribute.clone());
                                    }
                                    *attr = AttrRef {
                                        component: connection.from.component.clone(),
                                        attribute: channel_attribute_name(
                                            connection,
                                            &attr.attribute,
                                        ),
                                    };
                                }
                            }
                        });
                        if let Some(channel) = missing {
                            return Err(ModelError::PortEquationUnknownEdgeChannel {
                                port: here,
                                channel,
                                producer_port: producer_port(),
                            });
                        }
                        if let Some(&target) = index.get(producer.0) {
                            lowered.push((
                                target,
                                Equation {
                                    target: channel_attribute_name(connection, &equation.channel),
                                    kind: EquationKind::Explicit,
                                    expr,
                                },
                            ));
                        }
                    }
                }
            }
        }
        for component in &mut self.components {
            for port in &mut component.ports {
                port.equations.clear();
            }
        }
        for (target, equation) in lowered {
            self.components[target].equations.push(equation);
        }
        self.lower_out_port_aggregations()
    }

    /// Rewrite every `port_agg` over an out port as the explicit sum (or
    /// count) of the named channel over the port's connections.
    fn lower_out_port_aggregations(&mut self) -> Result<(), ModelError> {
        // Each out port with the connections leaving it, in declaration
        // order; owned, so the rewrite below can borrow it while it mutates
        // the components.
        let mut leaving: HashMap<(String, String), Vec<Connection>> = self
            .out_ports()
            .into_iter()
            .map(|(c, p)| ((c.to_owned(), p.to_owned()), Vec::new()))
            .collect();
        if leaving.is_empty() {
            return Ok(());
        }
        for connection in &self.connections {
            let key = (
                connection.from.component.clone(),
                connection.from.port.clone(),
            );
            if let Some(edges) = leaving.get_mut(&key) {
                edges.push(connection.clone());
            }
        }
        let is_out =
            |port: &PortRef| leaving.contains_key(&(port.component.clone(), port.port.clone()));
        // Refuse first, over the unmodified model, so the error names
        // where the aggregation was written.
        let mut refusal: Option<ModelError> = None;
        self.for_each_expr(&mut |here, expr| {
            Self::find_out_aggregation(expr, &is_out, &mut |port, agg| {
                if refusal.is_none() && !matches!(agg, AggOp::Sum | AggOp::Count) {
                    refusal = Some(ModelError::OutPortAggregation {
                        context: here(),
                        component: port.component.clone(),
                        port: port.port.clone(),
                        agg: agg.name().to_owned(),
                    });
                }
            });
        });
        if let Some(error) = refusal {
            return Err(error);
        }
        self.for_each_expr_mut(&mut |expr| {
            expr.rewrite(&mut |node| {
                let Expr::PortAgg {
                    port,
                    agg,
                    channel: Some(channel),
                } = node
                else {
                    return;
                };
                let Some(edges) = leaving.get(&(port.component.clone(), port.port.clone())) else {
                    return;
                };
                *node = match agg {
                    AggOp::Count => Expr::Const {
                        value: Value::Int(i64::try_from(edges.len()).unwrap_or(i64::MAX)),
                    },
                    _ if edges.is_empty() => Expr::Const {
                        value: Value::Int(0),
                    },
                    _ => Expr::Add {
                        args: edges
                            .iter()
                            .map(|k| Expr::Attr {
                                attr: AttrRef {
                                    component: k.from.component.clone(),
                                    attribute: channel_attribute_name(k, channel),
                                },
                            })
                            .collect(),
                    },
                };
            });
        });
        Ok(())
    }

    /// Calls `f` on every channel-named aggregation of an out port in `expr`.
    fn find_out_aggregation(
        expr: &Expr,
        is_out: &impl Fn(&PortRef) -> bool,
        f: &mut impl FnMut(&PortRef, AggOp),
    ) {
        if let Expr::PortAgg {
            port,
            agg,
            channel: Some(_),
        } = expr
        {
            if is_out(port) {
                f(port, *agg);
            }
        }
        expr.for_each_child(&mut |child| Self::find_out_aggregation(child, is_out, f));
    }

    fn expand_interface_connection(
        &self,
        link: &InterfaceConnection,
    ) -> Result<Vec<Connection>, ModelError> {
        let side = |reference: &InterfaceRef, side: &'static str| {
            let unknown = || ModelError::InterfaceConnectionUnknown {
                component: reference.component.clone(),
                interface: reference.interface.clone(),
                side,
            };
            let component = self
                .components
                .iter()
                .find(|c| c.name == reference.component)
                .ok_or_else(unknown)?;
            let interface = component
                .interfaces
                .iter()
                .find(|i| i.name == reference.interface)
                .ok_or_else(unknown)?;
            let ports: Vec<(&str, Option<PortDir>)> = interface
                .ports
                .iter()
                .map(|name| {
                    let dir = component
                        .ports
                        .iter()
                        .find(|p| &p.name == name)
                        .map(|p| p.dir);
                    (name.as_str(), dir)
                })
                .collect();
            Ok::<_, ModelError>(ports)
        };
        let from = side(&link.from, "from")?;
        let to = side(&link.to, "to")?;
        let unmatched = |component: &str, port: &str| ModelError::InterfacePortUnmatched {
            from: Box::new(link.from.clone()),
            to: Box::new(link.to.clone()),
            component: component.to_owned(),
            port: port.to_owned(),
        };
        for (port, _) in &to {
            if !from.iter().any(|(name, _)| name == port) {
                return Err(unmatched(&link.to.component, port));
            }
        }
        let mut out = Vec::with_capacity(from.len());
        for (port, dir) in &from {
            let Some((_, other)) = to.iter().find(|(name, _)| name == port) else {
                return Err(unmatched(&link.from.component, port));
            };
            let (source, target) = match (dir, other) {
                (Some(PortDir::Out), Some(PortDir::In)) => (&link.from, &link.to),
                (Some(PortDir::In), Some(PortDir::Out)) => (&link.to, &link.from),
                (Some(same), Some(_)) => {
                    return Err(ModelError::InterfacePortDirection {
                        from: Box::new(link.from.clone()),
                        to: Box::new(link.to.clone()),
                        port: (*port).to_owned(),
                        dir: match same {
                            PortDir::In => "in",
                            PortDir::Out => "out",
                        },
                    });
                }
                // A listed port the component lacks: validation reports
                // it as `InterfaceUnknownPort`, so nothing is expanded.
                _ => continue,
            };
            out.push(Connection {
                name: link.name.clone(),
                from: PortRef {
                    component: source.component.clone(),
                    port: (*port).to_owned(),
                },
                to: PortRef {
                    component: target.component.clone(),
                    port: (*port).to_owned(),
                },
            });
        }
        Ok(out)
    }
}
