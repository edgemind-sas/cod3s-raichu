//! Validation of a component: attributes, ports, automata and their transitions.

use crate::validate::Scope;
use crate::{
    AttrKind, Automaton, ChannelAttribute, Component, CycleCancellation, Distrib, EquationKind,
    InterruptionPolicy, Model, ModelError, PortDir, Transition,
};
use raichu_expr::Expr;
use std::collections::{HashMap, HashSet};

impl Model {
    /// An edge effect's target must have no other writer that would erase
    /// the one-shot write at the next evaluation: an explicit equation or a
    /// sensitive function. The target of an ODE equation is not such a
    /// writer: integration starts from the attribute's current value, so an
    /// edge effect on it is a reset map, the jump of a piecewise-
    /// deterministic Markov process.
    pub(crate) fn check_transition_effects(&self) -> Result<(), ModelError> {
        let writers = self.level_writers();
        for component in &self.components {
            for automaton in &component.automata {
                for transition in &automaton.transitions {
                    for assignment in &transition.effects {
                        let key = (
                            assignment.target.component.as_str(),
                            assignment.target.attribute.as_str(),
                        );
                        if let Some(writer) = writers.get(&key) {
                            return Err(ModelError::TransitionEffectOverwritten {
                                transition: format!(
                                    "{}.{}.{}",
                                    component.name, automaton.name, transition.name
                                ),
                                attribute: format!("{}.{}", key.0, key.1),
                                writer: writer.clone(),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The writers that set an attribute as a *level*, re-evaluated
    /// whenever what they read changes: explicit equations and sensitive
    /// functions, first writer in declaration order. ODE equations are left
    /// out: they move their target from its current value instead of
    /// recomputing it.
    fn level_writers(&self) -> HashMap<(&str, &str), String> {
        let mut writers: HashMap<(&str, &str), String> = HashMap::new();
        for component in &self.components {
            for equation in &component.equations {
                if matches!(equation.kind, EquationKind::Ode) {
                    continue;
                }
                writers
                    .entry((component.name.as_str(), equation.target.as_str()))
                    .or_insert_with(|| {
                        format!(
                            "the explicit equation on `{}.{}`",
                            component.name, equation.target
                        )
                    });
            }
            for function in &component.sensitive_functions {
                for assignment in &function.effects {
                    writers
                        .entry((
                            assignment.target.component.as_str(),
                            assignment.target.attribute.as_str(),
                        ))
                        .or_insert_with(|| {
                            format!("sensitive function `{}.{}`", component.name, function.name)
                        });
                }
            }
        }
        writers
    }

    pub(crate) fn check_component<'m>(
        component: &'m Component,
        materialised: &[&ChannelAttribute],
    ) -> Result<Scope<'m>, ModelError> {
        let mut attributes = HashMap::new();
        for attribute in &component.attributes {
            if attributes
                .insert(attribute.name.as_str(), attribute)
                .is_some()
            {
                return Err(ModelError::DuplicateInComponent {
                    kind: "attribute",
                    name: attribute.name.clone(),
                    component: component.name.clone(),
                });
            }
            if !attribute.kind.matches(&attribute.init) {
                return Err(ModelError::InitKindMismatch {
                    component: component.name.clone(),
                    attribute: attribute.name.clone(),
                    kind: attribute.kind,
                });
            }
        }

        let mut ports = HashMap::new();
        for port in &component.ports {
            if ports.insert(port.name.as_str(), port).is_some() {
                return Err(ModelError::DuplicateInComponent {
                    kind: "port",
                    name: port.name.clone(),
                    component: component.name.clone(),
                });
            }
            match (port.dir, &port.attr) {
                (PortDir::Out, None) => {
                    return Err(ModelError::OutPortWithoutVariable {
                        component: component.name.clone(),
                        port: port.name.clone(),
                    });
                }
                (PortDir::Out, Some(var)) if !attributes.contains_key(var.as_str()) => {
                    return Err(ModelError::PortUnknownVariable {
                        component: component.name.clone(),
                        port: port.name.clone(),
                        attribute: var.clone(),
                    });
                }
                (PortDir::In, Some(_)) => {
                    return Err(ModelError::InPortWithVariable {
                        component: component.name.clone(),
                        port: port.name.clone(),
                    });
                }
                _ => {}
            }
            if port.dir == PortDir::In && !port.channels.is_empty() {
                return Err(ModelError::InPortWithChannels {
                    component: component.name.clone(),
                    port: port.name.clone(),
                });
            }
            let mut channel_names = HashSet::new();
            for channel in &port.channels {
                if !channel_names.insert(channel.name.as_str()) {
                    return Err(ModelError::DuplicatePortChannel {
                        component: component.name.clone(),
                        port: port.name.clone(),
                        channel: channel.name.clone(),
                    });
                }
            }
        }

        // The compiler materialises one float attribute per (connection,
        // channel): refuse a name already taken by a declared attribute,
        // or claimed twice by two unnamed parallel edges. Both are silent
        // wrong answers otherwise, the second attribute quietly winning.
        let mut channel_attributes: HashSet<String> = HashSet::new();
        let mut origin: HashMap<&str, usize> = HashMap::new();
        for entry in materialised {
            if attributes.contains_key(entry.attribute.as_str()) {
                return Err(ModelError::ChannelAttributeCollision {
                    component: component.name.clone(),
                    port: entry.port.clone(),
                    channel: entry.channel.clone(),
                    attribute: entry.attribute.clone(),
                });
            }
            if let Some(first) = origin.insert(entry.attribute.as_str(), entry.connection) {
                return Err(ModelError::DuplicateChannelAttribute {
                    component: component.name.clone(),
                    attribute: entry.attribute.clone(),
                    channel: entry.channel.clone(),
                    first,
                    second: entry.connection,
                });
            }
            channel_attributes.insert(entry.attribute.clone());
        }

        let mut interface_names = HashSet::new();
        for interface in &component.interfaces {
            if !interface_names.insert(interface.name.as_str()) {
                return Err(ModelError::DuplicateInComponent {
                    kind: "interface",
                    name: interface.name.clone(),
                    component: component.name.clone(),
                });
            }
            for port in &interface.ports {
                if !ports.contains_key(port.as_str()) {
                    return Err(ModelError::InterfaceUnknownPort {
                        component: component.name.clone(),
                        interface: interface.name.clone(),
                        port: port.clone(),
                    });
                }
            }
        }

        let mut automata: HashMap<&str, HashSet<&str>> = HashMap::new();
        for automaton in &component.automata {
            if automata.contains_key(automaton.name.as_str()) {
                return Err(ModelError::DuplicateInComponent {
                    kind: "automaton",
                    name: automaton.name.clone(),
                    component: component.name.clone(),
                });
            }
            Self::check_automaton(component, automaton)?;
            automata.insert(
                automaton.name.as_str(),
                automaton.states.iter().map(String::as_str).collect(),
            );
        }
        Self::check_cycle_cancellation(component)?;

        let mut function_names = HashSet::new();
        for function in &component.sensitive_functions {
            if !function_names.insert(function.name.as_str()) {
                return Err(ModelError::DuplicateInComponent {
                    kind: "sensitive function",
                    name: function.name.clone(),
                    component: component.name.clone(),
                });
            }
        }

        let mut equation_targets = HashSet::new();
        let mut explicit_targets = HashSet::new();
        for equation in &component.equations {
            let kind = attributes
                .get(equation.target.as_str())
                .map(|attribute| attribute.kind)
                .or_else(|| {
                    channel_attributes
                        .contains(equation.target.as_str())
                        .then_some(AttrKind::Float)
                });
            let Some(kind) = kind else {
                return Err(ModelError::EquationUnknownTarget {
                    component: component.name.clone(),
                    target: equation.target.clone(),
                });
            };
            if kind != AttrKind::Float {
                return Err(ModelError::EquationTargetNotFloat {
                    component: component.name.clone(),
                    target: equation.target.clone(),
                });
            }
            if !equation_targets.insert(equation.target.as_str()) {
                return Err(ModelError::DuplicateEquation {
                    component: component.name.clone(),
                    target: equation.target.clone(),
                });
            }
            if equation.kind == EquationKind::Explicit {
                explicit_targets.insert(equation.target.as_str());
            }
        }

        // Conservative distribution operators: everything checkable
        // inside the component. Their consumer-keyed policy parameters
        // need the connection list and are checked by
        // [`Model::check_allocations`].
        let mut allocation_names = HashSet::new();
        for allocation in &component.allocations {
            if !allocation_names.insert(allocation.name.as_str()) {
                return Err(ModelError::DuplicateInComponent {
                    kind: "distribution operator",
                    name: allocation.name.clone(),
                    component: component.name.clone(),
                });
            }
            // An operator and an explicit equation are both steps of the
            // sweep, designated in the evaluation order by their name: two
            // steps of one component may not share one.
            if explicit_targets.contains(allocation.name.as_str()) {
                return Err(ModelError::EvaluationStepAmbiguous {
                    component: component.name.clone(),
                    name: allocation.name.clone(),
                });
            }
            let port = ports
                .get(allocation.port.as_str())
                .filter(|port| port.dir == PortDir::Out)
                .ok_or_else(|| ModelError::AllocationPortInvalid {
                    component: component.name.clone(),
                    allocation: allocation.name.clone(),
                    port: allocation.port.clone(),
                })?;
            for (role, channel) in [
                ("demand", &allocation.demand),
                ("allocated", &allocation.allocated),
            ] {
                if !port.channels.iter().any(|c| &c.name == channel) {
                    return Err(ModelError::AllocationUnknownChannel {
                        component: component.name.clone(),
                        allocation: allocation.name.clone(),
                        channel: channel.clone(),
                        role,
                    });
                }
            }
            if allocation.demand == allocation.allocated {
                return Err(ModelError::AllocationChannelReused {
                    component: component.name.clone(),
                    allocation: allocation.name.clone(),
                    channel: allocation.demand.clone(),
                });
            }
        }

        Ok(Scope {
            attributes,
            channel_attributes,
            ports,
            automata,
        })
    }

    /// A [`Transition::cycle_cancellation`] other than `auto` sits on a
    /// transition of a cycle group, and every transition of one
    /// `(component, cycle_group)` declares the same value: the decision is
    /// the group's, so two halves of it cannot disagree.
    fn check_cycle_cancellation(component: &Component) -> Result<(), ModelError> {
        let mut declared: HashMap<&str, (CycleCancellation, String)> = HashMap::new();
        for automaton in &component.automata {
            for transition in &automaton.transitions {
                let qualified =
                    || format!("{}.{}.{}", component.name, automaton.name, transition.name);
                let Some(group) = transition.cycle_group.as_deref() else {
                    if !transition.cycle_cancellation.is_auto() {
                        return Err(ModelError::CycleCancellationInvalid {
                            transition: qualified(),
                            reason: "the transition has no `cycle_group`, so there is no \
                                     cancellation to decide"
                                .to_owned(),
                        });
                    }
                    continue;
                };
                match declared.get(group) {
                    Some((value, first)) if *value != transition.cycle_cancellation => {
                        return Err(ModelError::CycleCancellationInvalid {
                            transition: qualified(),
                            reason: format!(
                                "`{first}` of the same cycle group `{group}` declares \
                                 another value; every transition of a group declares \
                                 the same one"
                            ),
                        });
                    }
                    Some(_) => {}
                    None => {
                        declared.insert(group, (transition.cycle_cancellation, qualified()));
                    }
                }
            }
        }
        Ok(())
    }

    fn check_automaton(component: &Component, automaton: &Automaton) -> Result<(), ModelError> {
        let mut states = HashSet::new();
        for state in &automaton.states {
            if !states.insert(state.as_str()) {
                return Err(ModelError::DuplicateInComponent {
                    kind: "state",
                    name: state.clone(),
                    component: component.name.clone(),
                });
            }
        }
        if !states.contains(automaton.init.as_str()) {
            return Err(ModelError::UnknownInitState {
                component: component.name.clone(),
                automaton: automaton.name.clone(),
                state: automaton.init.clone(),
            });
        }

        let mut transition_names = HashSet::new();
        for transition in &automaton.transitions {
            if !transition_names.insert(transition.name.as_str()) {
                return Err(ModelError::DuplicateInComponent {
                    kind: "transition",
                    name: transition.name.clone(),
                    component: component.name.clone(),
                });
            }
            if transition.guard.is_none() && transition.on_interruption != InterruptionPolicy::Reset
            {
                return Err(ModelError::InterruptionPolicyWithoutGuard {
                    component: component.name.clone(),
                    automaton: automaton.name.clone(),
                    transition: transition.name.clone(),
                    policy: match transition.on_interruption {
                        InterruptionPolicy::Resume => "resume",
                        _ => "continue",
                    },
                });
            }
            for state in std::iter::once(&transition.source).chain(&transition.targets) {
                if !states.contains(state.as_str()) {
                    return Err(ModelError::UnknownTransitionState {
                        component: component.name.clone(),
                        automaton: automaton.name.clone(),
                        transition: transition.name.clone(),
                        state: state.clone(),
                    });
                }
            }
            if let Some(recorded) = &transition.monitored_states {
                let reason = if !transition.monitored {
                    Some("the transition is not `monitored`, so nothing is recorded".to_owned())
                } else if recorded.is_empty() {
                    Some(
                        "the list is empty: a transition that records no target is                          declared `\"monitored\": false`"
                            .to_owned(),
                    )
                } else {
                    recorded
                        .iter()
                        .find(|state| !transition.targets.contains(state))
                        .map(|state| format!("`{state}` is not a target of the transition"))
                };
                if let Some(reason) = reason {
                    return Err(ModelError::MonitoredStatesInvalid {
                        transition: format!(
                            "{}.{}.{}",
                            component.name, automaton.name, transition.name
                        ),
                        reason,
                    });
                }
            }
            match &transition.distrib {
                Distrib::Delay { time } => {
                    if transition.targets.len() != 1 {
                        return Err(ModelError::DelayTargetCount {
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            targets: transition.targets.len(),
                        });
                    }
                    if !time.is_finite() || *time < 0.0 {
                        return Err(ModelError::InvalidDelay {
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            time: *time,
                        });
                    }
                }
                Distrib::Inst { probs } => {
                    if probs.len() + 1 != transition.targets.len() {
                        return Err(ModelError::InstArityMismatch {
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            targets: transition.targets.len(),
                            probs: probs.len(),
                        });
                    }
                    let sum: f64 = probs.iter().sum();
                    let each_valid = probs
                        .iter()
                        .all(|p| p.is_finite() && (0.0..=1.0).contains(p));
                    if !each_valid || sum > 1.0 {
                        return Err(ModelError::InvalidInstProbs {
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            sum,
                        });
                    }
                }
                Distrib::Exp { rate, rate_expr } => {
                    Self::check_single_target("exp", component, automaton, transition)?;
                    match (rate, rate_expr) {
                        (Some(rate), None) => {
                            // A rate of 0 is a transition that cannot
                            // fire, not an error.
                            if !rate.is_finite() || *rate < 0.0 {
                                return Err(ModelError::InvalidExpRate {
                                    component: component.name.clone(),
                                    automaton: automaton.name.clone(),
                                    transition: transition.name.clone(),
                                    rate: *rate,
                                });
                            }
                        }
                        (None, Some(_)) => {}
                        _ => {
                            return Err(ModelError::ExpRateSpec {
                                component: component.name.clone(),
                                automaton: automaton.name.clone(),
                                transition: transition.name.clone(),
                            });
                        }
                    }
                }
                Distrib::Weibull { shape, scale } => {
                    Self::check_single_target("weibull", component, automaton, transition)?;
                    Self::check_positive(
                        "weibull", "shape", *shape, component, automaton, transition,
                    )?;
                    Self::check_positive(
                        "weibull", "scale", *scale, component, automaton, transition,
                    )?;
                }
                Distrib::Lognormal { mu, sigma } => {
                    Self::check_single_target("lognormal", component, automaton, transition)?;
                    if !mu.is_finite() {
                        return Err(ModelError::InvalidLawParameter {
                            distribution: "lognormal",
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            parameter: "mu",
                            value: *mu,
                        });
                    }
                    Self::check_positive(
                        "lognormal",
                        "sigma",
                        *sigma,
                        component,
                        automaton,
                        transition,
                    )?;
                }
                Distrib::Normal {
                    mu,
                    sigma,
                    low,
                    high,
                } => {
                    Self::check_single_target("normal", component, automaton, transition)?;
                    let invalid =
                        |parameter: &'static str, value: f64| ModelError::InvalidLawParameter {
                            distribution: "normal",
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            parameter,
                            value,
                        };
                    if !mu.is_finite() {
                        return Err(invalid("mu", *mu));
                    }
                    Self::check_positive(
                        "normal", "sigma", *sigma, component, automaton, transition,
                    )?;
                    if !low.is_finite() || *low < 0.0 {
                        return Err(invalid("low", *low));
                    }
                    if let Some(high) = high {
                        if !high.is_finite() || *high <= *low {
                            return Err(invalid("high", *high));
                        }
                    }
                }
                Distrib::Gamma { shape, scale } => {
                    Self::check_single_target("gamma", component, automaton, transition)?;
                    Self::check_positive(
                        "gamma", "shape", *shape, component, automaton, transition,
                    )?;
                    Self::check_positive(
                        "gamma", "scale", *scale, component, automaton, transition,
                    )?;
                }
                Distrib::Uniform { low, high } => {
                    Self::check_single_target("uniform", component, automaton, transition)?;
                    if !low.is_finite() || *low < 0.0 {
                        return Err(ModelError::InvalidLawParameter {
                            distribution: "uniform",
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            parameter: "low",
                            value: *low,
                        });
                    }
                    if !high.is_finite() || *high <= *low {
                        return Err(ModelError::InvalidLawParameter {
                            distribution: "uniform",
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            parameter: "high",
                            value: *high,
                        });
                    }
                }
                Distrib::Empirical { points } => {
                    Self::check_single_target("empirical", component, automaton, transition)?;
                    let table_error = |detail: String| ModelError::InvalidEmpiricalTable {
                        component: component.name.clone(),
                        automaton: automaton.name.clone(),
                        transition: transition.name.clone(),
                        detail,
                    };
                    if points.is_empty() {
                        return Err(table_error("empty table".to_owned()));
                    }
                    let mut prev: Option<(f64, f64)> = None;
                    for &(t, c) in points {
                        if !t.is_finite() || t < 0.0 || !c.is_finite() || !(0.0..=1.0).contains(&c)
                        {
                            return Err(table_error(format!("invalid point ({t}, {c})")));
                        }
                        if let Some((pt, pc)) = prev {
                            if t < pt || c < pc {
                                return Err(table_error(format!(
                                    "non-monotone point ({t}, {c}) after ({pt}, {pc})"
                                )));
                            }
                        }
                        prev = Some((t, c));
                    }
                    if let Some((_, last)) = prev {
                        if last != 1.0 {
                            return Err(table_error(format!(
                                "last cumulative probability is {last}, expected 1"
                            )));
                        }
                    }
                }
                Distrib::Watched => {
                    if transition.targets.len() != 1 {
                        return Err(ModelError::WatchedTargetCount {
                            component: component.name.clone(),
                            automaton: automaton.name.clone(),
                            transition: transition.name.clone(),
                            targets: transition.targets.len(),
                        });
                    }
                    match &transition.guard {
                        None => {
                            return Err(ModelError::WatchedGuardMissing {
                                component: component.name.clone(),
                                automaton: automaton.name.clone(),
                                transition: transition.name.clone(),
                            });
                        }
                        Some(guard) if Self::is_watched_guard(guard) => {}
                        Some(_) => {
                            return Err(ModelError::WatchedGuardNotComparison {
                                component: component.name.clone(),
                                automaton: automaton.name.clone(),
                                transition: transition.name.clone(),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// A watched guard must expose exactly one *ordering comparison*
    /// (the located boundary), optionally conjoined/disjoined with
    /// discrete gate expressions (which only change at discrete events):
    /// `cmp`, `and(gates…, cmp)` or `or(gates…, cmp)`.
    /// A watched guard must contain at least one ordering comparison
    /// (the continuous boundary) anywhere under `and`/`or`/`not`
    /// connectives (the other operands act as discrete gates). The
    /// margin compiler maps `and` to the `min` of the operand margins,
    /// `or` to the `max`, `not` to the negation, and any other boolean
    /// operand to ±1.
    fn is_watched_guard(guard: &Expr) -> bool {
        match guard {
            Expr::Cmp {
                cmp:
                    raichu_expr::CmpOp::Lt
                    | raichu_expr::CmpOp::Le
                    | raichu_expr::CmpOp::Gt
                    | raichu_expr::CmpOp::Ge,
                ..
            } => true,
            Expr::Bool {
                bool_op: raichu_expr::BoolOp::And | raichu_expr::BoolOp::Or,
                args,
            } => args.iter().any(Self::is_watched_guard),
            Expr::Bool {
                bool_op: raichu_expr::BoolOp::Not,
                args,
            } => args.iter().any(Self::is_watched_guard),
            _ => false,
        }
    }

    fn check_single_target(
        distribution: &'static str,
        component: &Component,
        automaton: &Automaton,
        transition: &Transition,
    ) -> Result<(), ModelError> {
        if transition.targets.len() != 1 {
            return Err(ModelError::StochasticTargetCount {
                distribution,
                component: component.name.clone(),
                automaton: automaton.name.clone(),
                transition: transition.name.clone(),
                targets: transition.targets.len(),
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)] // flat validation helper
    fn check_positive(
        distribution: &'static str,
        parameter: &'static str,
        value: f64,
        component: &Component,
        automaton: &Automaton,
        transition: &Transition,
    ) -> Result<(), ModelError> {
        if !value.is_finite() || value <= 0.0 {
            return Err(ModelError::InvalidLawParameter {
                distribution,
                component: component.name.clone(),
                automaton: automaton.name.clone(),
                transition: transition.name.clone(),
                parameter,
                value,
            });
        }
        Ok(())
    }
}
