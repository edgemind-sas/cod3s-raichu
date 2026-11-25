//! Validation of the optimisation programs.

use crate::{channel_attribute_name, AttrKind, Model, ModelError, Program, ProgramTieBreak, Scope};
use raichu_expr::{AggOp, AttrRef, Expr, Value};
use std::collections::{HashMap, HashSet};

impl Model {
    /// Validate the discrete-only, affine and single-writer contract of all
    /// model-level programs before compilation can consume them.
    pub(crate) fn check_programs(
        &self,
        scopes: &HashMap<&str, Scope<'_>>,
    ) -> Result<(), ModelError> {
        let mut names = HashSet::new();
        let mut outputs: HashMap<AttrRef, usize> = HashMap::new();
        let invalid =
            |program: &Program, term: String, reason: String| ModelError::ProgramInvalid {
                program: program.name.clone(),
                term,
                reason,
            };
        for (index, program) in self.programs.iter().enumerate() {
            if !names.insert(program.name.as_str()) {
                return Err(invalid(
                    program,
                    "name".into(),
                    "duplicate program name".into(),
                ));
            }
            if program.variables.is_empty() {
                return Err(invalid(
                    program,
                    "variables".into(),
                    "at least one decision variable is required".into(),
                ));
            }
            if program
                .node_limit
                .is_some_and(|limit| limit <= 0 || limit > i32::MAX as i64)
            {
                return Err(invalid(
                    program,
                    "node_limit".into(),
                    "must be in 1..=2147483647".into(),
                ));
            }
            let mut local = HashSet::new();
            for variable in &program.variables {
                let attr = &variable.attribute;
                let term = format!("{}.{}", attr.component, attr.attribute);
                let kind = scopes
                    .get(attr.component.as_str())
                    .and_then(|scope| scope.attribute_kind(&attr.attribute))
                    .ok_or_else(|| {
                        invalid(program, term.clone(), "unknown decision attribute".into())
                    })?;
                if !kind.matches(&variable.on_infeasible) {
                    return Err(invalid(
                        program,
                        term,
                        "fallback kind does not match the decision attribute".into(),
                    ));
                }
                if !local.insert(attr.clone()) {
                    return Err(invalid(
                        program,
                        term,
                        "decision attribute is repeated".into(),
                    ));
                }
                if let Some(other) = outputs.insert(attr.clone(), index) {
                    return Err(invalid(
                        program,
                        term,
                        format!("also written by program `{}`", self.programs[other].name),
                    ));
                }
            }
            for (attr, expected) in [(&program.feasible, AttrKind::Bool)].into_iter().chain(
                program
                    .objective_value
                    .iter()
                    .map(|attr| (attr, AttrKind::Float)),
            ) {
                let term = format!("{}.{}", attr.component, attr.attribute);
                let kind = scopes
                    .get(attr.component.as_str())
                    .and_then(|scope| scope.attribute_kind(&attr.attribute))
                    .ok_or_else(|| {
                        invalid(program, term.clone(), "unknown output attribute".into())
                    })?;
                if kind != expected {
                    return Err(invalid(
                        program,
                        term,
                        format!("output must have kind {expected:?}"),
                    ));
                }
                if let Some(other) = outputs.insert(attr.clone(), index) {
                    return Err(invalid(
                        program,
                        term,
                        format!("also written by program `{}`", self.programs[other].name),
                    ));
                }
            }
        }

        let mut allocated = HashSet::new();
        for component in &self.components {
            for allocation in &component.allocations {
                for edge in self.connections.iter().filter(|edge| {
                    edge.from.component == component.name && edge.from.port == allocation.port
                }) {
                    allocated.insert(AttrRef {
                        component: component.name.clone(),
                        attribute: channel_attribute_name(edge, &allocation.allocated),
                    });
                }
            }
        }
        let equation_targets: HashSet<_> = self
            .components
            .iter()
            .flat_map(|component| {
                component.equations.iter().map(|equation| AttrRef {
                    component: component.name.clone(),
                    attribute: equation.target.clone(),
                })
            })
            .collect();
        let regular_writers = self.attribute_writers();
        let sources = self.in_port_sources();
        let mut dependencies = vec![HashSet::new(); self.programs.len()];
        for (index, program) in self.programs.iter().enumerate() {
            let decisions: HashSet<_> = program
                .variables
                .iter()
                .map(|variable| variable.attribute.clone())
                .collect();
            for output in decisions
                .iter()
                .chain(std::iter::once(&program.feasible))
                .chain(program.objective_value.iter())
            {
                let term = format!("{}.{}", output.component, output.attribute);
                if let Some(writer) =
                    regular_writers.get(&(output.component.as_str(), output.attribute.as_str()))
                {
                    return Err(invalid(program, term, format!("also written by {writer}")));
                }
                if allocated.contains(output) {
                    return Err(invalid(
                        program,
                        term,
                        "also written by a distribution operator".into(),
                    ));
                }
                if self
                    .components
                    .iter()
                    .flat_map(|component| &component.automata)
                    .flat_map(|automaton| &automaton.transitions)
                    .flat_map(|transition| &transition.effects)
                    .any(|effect| &effect.target == output)
                {
                    return Err(invalid(
                        program,
                        term,
                        "also written by a transition effect".into(),
                    ));
                }
            }
            let mut expressions: Vec<(&str, &Expr, bool)> =
                vec![("objective", &program.objective, true)];
            for variable in &program.variables {
                if let Some(expr) = &variable.lower {
                    expressions.push(("variable lower", expr, false));
                }
                if let Some(expr) = &variable.upper {
                    expressions.push(("variable upper", expr, false));
                }
            }
            let mut constraint_names = HashSet::new();
            for constraint in &program.constraints {
                if !constraint_names.insert(constraint.name.as_str()) {
                    return Err(invalid(
                        program,
                        constraint.name.clone(),
                        "duplicate constraint name".into(),
                    ));
                }
                if constraint.lower.is_none() && constraint.upper.is_none() {
                    return Err(invalid(
                        program,
                        constraint.name.clone(),
                        "constraint requires at least one bound".into(),
                    ));
                }
                expressions.push((&constraint.name, &constraint.expr, true));
                if let Some(expr) = &constraint.lower {
                    expressions.push((&constraint.name, expr, false));
                }
                if let Some(expr) = &constraint.upper {
                    expressions.push((&constraint.name, expr, false));
                }
            }
            if let Some(ProgramTieBreak::Objectives(objectives)) = &program.tie_break {
                for objective in objectives {
                    expressions.push(("tie_break", &objective.expr, true));
                }
            }
            for (term, expr, affine) in expressions {
                Self::check_expr(scopes, &sources, expr, term)
                    .map_err(|error| invalid(program, term.into(), error.to_string()))?;
                if !Self::program_expr_numeric(expr, scopes, &decisions) {
                    return Err(invalid(
                        program,
                        term.into(),
                        "expected a numeric expression".into(),
                    ));
                }
                let form = expr.affine_in(&decisions).map_err(|error| {
                    invalid(
                        program,
                        term.into(),
                        format!("nonlinear decision term: {:?}", error.expression),
                    )
                })?;
                if !affine && !form.terms.is_empty() {
                    return Err(invalid(
                        program,
                        term.into(),
                        "bound reads this program's decision variable".into(),
                    ));
                }
                let free_parts = std::iter::once(&form.constant)
                    .chain(form.terms.iter().map(|(_, coefficient)| coefficient));
                for free in free_parts {
                    if Self::expr_contains_time(free) {
                        return Err(invalid(
                            program,
                            term.into(),
                            "time input is continuously varying".into(),
                        ));
                    }
                    let mut refs = Vec::new();
                    free.for_each_attr_ref(&mut |attr| refs.push(attr.clone()));
                    for attr in refs {
                        let name = format!("{}.{}", attr.component, attr.attribute);
                        if equation_targets.contains(&attr) || allocated.contains(&attr) {
                            return Err(invalid(
                                program,
                                name,
                                "input is computed by the explicit sweep or integration".into(),
                            ));
                        }
                        if let Some(&producer) = outputs.get(&attr) {
                            if producer == index {
                                return Err(invalid(
                                    program,
                                    name,
                                    "program reads its own output".into(),
                                ));
                            }
                            dependencies[index].insert(producer);
                        }
                    }
                    let mut ports = Vec::new();
                    free.for_each_port_agg(&mut |port, channel| {
                        ports.push((port.clone(), channel.map(str::to_owned)))
                    });
                    for (port, channel) in ports {
                        for edge in self.connections.iter().filter(|edge| edge.to == port) {
                            let attribute = channel
                                .as_deref()
                                .map(|name| channel_attribute_name(edge, name))
                                .or_else(|| {
                                    scopes
                                        .get(edge.from.component.as_str())
                                        .and_then(|scope| scope.ports.get(edge.from.port.as_str()))
                                        .and_then(|source| source.attr.clone())
                                });
                            if let Some(attribute) = attribute {
                                let source = AttrRef {
                                    component: edge.from.component.clone(),
                                    attribute,
                                };
                                if equation_targets.contains(&source) || allocated.contains(&source)
                                {
                                    return Err(invalid(
                                        program,
                                        format!("{}.{}", port.component, port.port),
                                        "port aggregate reads a sweep or integrated quantity"
                                            .into(),
                                    ));
                                }
                                if let Some(&producer) = outputs.get(&source) {
                                    if producer == index {
                                        return Err(invalid(
                                            program,
                                            format!("{}.{}", port.component, port.port),
                                            "program reads its own output through a port".into(),
                                        ));
                                    }
                                    dependencies[index].insert(producer);
                                }
                            }
                        }
                    }
                }
            }
        }
        fn reaches(
            index: usize,
            next: usize,
            graph: &[HashSet<usize>],
            seen: &mut HashSet<usize>,
        ) -> bool {
            if next == index {
                return true;
            }
            if !seen.insert(next) {
                return false;
            }
            graph[next]
                .iter()
                .any(|&node| reaches(index, node, graph, seen))
        }
        for (index, edges) in dependencies.iter().enumerate() {
            if edges
                .iter()
                .any(|&next| reaches(index, next, &dependencies, &mut HashSet::new()))
            {
                return Err(invalid(
                    &self.programs[index],
                    "inputs".into(),
                    "cycle between programs".into(),
                ));
            }
        }
        Ok(())
    }

    fn expr_contains_time(expr: &Expr) -> bool {
        if matches!(expr, Expr::Time) {
            return true;
        }
        let mut found = false;
        expr.for_each_child(&mut |child| found |= Self::expr_contains_time(child));
        found
    }

    fn program_expr_numeric(
        expr: &Expr,
        scopes: &HashMap<&str, Scope<'_>>,
        decisions: &HashSet<AttrRef>,
    ) -> bool {
        match expr {
            Expr::Const { value } => !matches!(value, Value::Bool(_)),
            Expr::Attr { attr } => {
                decisions.contains(attr)
                    || scopes
                        .get(attr.component.as_str())
                        .and_then(|scope| scope.attribute_kind(&attr.attribute))
                        .is_some_and(|kind| kind != AttrKind::Bool)
            }
            Expr::PortAgg { agg, .. } => !matches!(agg, AggOp::All | AggOp::Any),
            Expr::Add { args } | Expr::Mul { args } | Expr::Min { args } | Expr::Max { args } => {
                args.iter()
                    .all(|arg| Self::program_expr_numeric(arg, scopes, decisions))
            }
            Expr::Sub { lhs, rhs } | Expr::Div { lhs, rhs } => {
                Self::program_expr_numeric(lhs, scopes, decisions)
                    && Self::program_expr_numeric(rhs, scopes, decisions)
            }
            Expr::Sin { arg }
            | Expr::Exp { arg }
            | Expr::PiecewiseLinear { arg, .. }
            | Expr::SmoothNoise { arg, .. } => Self::program_expr_numeric(arg, scopes, decisions),
            Expr::If {
                cond,
                then,
                otherwise,
            } => {
                Self::program_expr_boolean(cond, scopes, decisions)
                    && Self::program_expr_numeric(then, scopes, decisions)
                    && Self::program_expr_numeric(otherwise, scopes, decisions)
            }
            Expr::Time => true,
            Expr::StateActive { .. } | Expr::Cmp { .. } | Expr::Bool { .. } => false,
        }
    }

    fn program_expr_boolean(
        expr: &Expr,
        scopes: &HashMap<&str, Scope<'_>>,
        decisions: &HashSet<AttrRef>,
    ) -> bool {
        match expr {
            Expr::Const { value } => matches!(value, Value::Bool(_)),
            Expr::Attr { attr } => {
                !decisions.contains(attr)
                    && scopes
                        .get(attr.component.as_str())
                        .and_then(|scope| scope.attribute_kind(&attr.attribute))
                        .is_some_and(|kind| kind == AttrKind::Bool)
            }
            Expr::StateActive { .. } => true,
            Expr::PortAgg { agg, .. } => matches!(agg, AggOp::All | AggOp::Any),
            Expr::Cmp { lhs, rhs, .. } => {
                (Self::program_expr_numeric(lhs, scopes, decisions)
                    && Self::program_expr_numeric(rhs, scopes, decisions))
                    || (Self::program_expr_boolean(lhs, scopes, decisions)
                        && Self::program_expr_boolean(rhs, scopes, decisions))
            }
            Expr::Bool { args, .. } => args
                .iter()
                .all(|arg| Self::program_expr_boolean(arg, scopes, decisions)),
            Expr::If {
                cond,
                then,
                otherwise,
            } => {
                Self::program_expr_boolean(cond, scopes, decisions)
                    && Self::program_expr_boolean(then, scopes, decisions)
                    && Self::program_expr_boolean(otherwise, scopes, decisions)
            }
            _ => false,
        }
    }
}
