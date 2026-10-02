//! Static detection of an **unfed trigger**: an in port nothing is
//! connected to, whose emptiness is what decides a mode.
//!
//! An in port aggregates what reaches it, and an aggregation over
//! nothing still answers: `any` and `sum` answer false and zero, `count`
//! answers zero, `all` answers the vacuous truth. So a guard reading an
//! in port that nothing feeds is not undefined, it is **pinned**: it
//! holds the same value from the initial instant to the end of the run,
//! whatever else the model does. The automaton it drives settles into
//! one state and has no transition out of it.
//!
//! That is the whole fault, and its cost is that it does not look like
//! one. A cold standby delivers while its trigger input is *absent*; an
//! input nothing feeds is absent, so the standby delivers from `t = 0`
//! and for ever. The run terminates normally, the journal says nothing,
//! and the campaign returns an availability that is too good. In a
//! safety study that is the most expensive shape an error can take: the
//! one that reassures.
//!
//! [`crate::loops`] catches a different fault on the same schedule, and
//! the two stay apart on purpose: a switching loop is a mode with no
//! fixpoint, this is a mode with no choice.
//!
//! It **warns and never refuses**. A trigger nothing feeds is a *valid*
//! model: the semantics is settled, the engines agree on it, and a run
//! on it is meaningful. It is only, almost always, an oversight, and the
//! way to declare the same behaviour on purpose is to state the mode
//! unconditionally, which is silent here for nothing: no trigger port at
//! all on a standby, no condition on a rule.
//!
//! # The guard rarely reads the port itself
//!
//! An authoring layer puts a **derived attribute** between the two: a
//! muscadet rule set thresholds `P.E_capability_in`, not `P.E_in`. So
//! the fold crosses a definition to reach the port, on one condition
//! that is the whole of its caution: the attribute must have **exactly
//! one writer**, and that writer must be an explicit equation. An
//! attribute an ODE integrates, an allocation distributes, a sensitive
//! function assigns, or two writers share, moves, and the fold abstains
//! on it as it abstains on a clock. An attribute **nothing** writes is
//! the opposite case and is a constant: it holds its declared initial
//! value for the whole run.
//!
//! # And the seal has to flatter the result
//!
//! Crossing the definition reaches modes the port alone did not, and not
//! all of them are worth a line. A rule that *consumes* the flow nothing
//! feeds is sealed just the same, and it is harmless: it cannot draw, so
//! it produces nothing, and a component that produces nothing is what a
//! reader sees in the results. The expensive shape is the other one, the
//! rule that **produces without consuming**, because it delivers for
//! ever and reads as availability.
//!
//! The two are told apart by what the seal does rather than by what it
//! is: every quantity that reads the sealed mode is folded twice, once
//! with the automaton assumed elsewhere and once with it pinned there.
//! A mode is reported when some quantity is **nothing** while the mode
//! is elsewhere and **not nothing** once it is pinned, which is the
//! definition of flattering the result. A seal that raises nothing is
//! silent, and that is where the rule with a `cons` lands.
//!
//! What is reported is therefore narrow by construction. A guard that
//! also reads a clock, a state or a moving attribute does not fold, so
//! the emptiness is not what decides it and nothing is said. An in port
//! nothing feeds whose value reaches only equations no mode reads is
//! silent too: it starves the component that reads it, which shows up in
//! the results as a component that produces nothing, the pessimistic way
//! round.
//!
//! How far that reaches is decided **above**, by whoever writes the
//! guards and the definitions between them and the ports, and it is
//! worth stating rather than discovering. Measured on the muscadet
//! authoring layer: `add_flow_out_on_trigger` puts a port aggregate
//! inside a guard, and `add_rule_set` reaches one through the explicit
//! equation of the input it thresholds. What stays out of reach is the
//! **boolean** half of the same layer, which derives `{flow}_fed_in`
//! through a *sensitive function*: two writers of the same attribute are
//! ordinary there, so the fold does not cross it. The measurement lives
//! with the authoring layer it is about, in
//! `python/tests/unit/test_unfed_triggers.py::test_what_a_guard_reaches_through_its_definitions`.
//!
//! That half is out of reach and no longer worth reaching, which are two
//! different statements and only the second is new. An authoring layer
//! answers for its own ports: muscadet gives a boolean in-flow a declared
//! out-of-connection value (`var_in_default`, false unless stated) and
//! writes it into the aggregating expression, so an in-flow nothing feeds
//! reads unfed rather than the vacuous truth of `all`. A mode guarded on
//! it is therefore sealed SHUT for ever, not armed for ever: the
//! component starves, produces nothing, and a reader of the results sees
//! it. That is the pessimistic direction, the one this diagnostic is
//! deliberately silent on, so widening the fold to cross a sensitive
//! function would buy nothing here. It stays a frontier of reach, and
//! what lies beyond it has stopped flattering the campaign.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use raichu_expr::{AggOp, BoolOp, CmpOp, Expr, StateRef, Value};
use raichu_model::{EquationKind, Model, PortDir};

/// One mode sealed by an in port nothing feeds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnfedTrigger {
    /// The in ports nothing is connected to that the sealing guards
    /// read, directly or through the definitions between, by qualified
    /// name `component.port`, in declaration order. Never empty: a mode
    /// sealed by something other than an empty port is not this
    /// diagnostic's business.
    pub ports: Vec<String>,
    /// The automaton their emptiness decides, qualified
    /// `component.automaton`.
    pub automaton: String,
    /// The state it settles in and has no enabled transition out of.
    pub state: String,
}

impl UnfedTrigger {
    /// The one-line diagnostic, phrased for someone who has just read an
    /// availability they would like to believe.
    ///
    /// It leads with the wire that is missing, because that is the one
    /// thing the reader can act on, and it names the mode so the finding
    /// can be checked rather than trusted. Between the two it says what
    /// the fault costs, in one clause and not one sentence: this warning
    /// exists because the fault does **not** look like one, so a reader
    /// told only that a mode is sealed has no reason to act, and a
    /// reader told it at length stops reading.
    ///
    /// That is the whole of the length budget, and it is deliberate:
    /// measured against `SwitchingLoop::describe` next door, this is the
    /// same register once the two qualified names it carries are set
    /// aside. A warning nobody finishes reading is a warning that does
    /// not exist, and `the_diagnostic_stays_in_the_register_of_its_sibling`
    /// keeps it there.
    pub fn describe(&self) -> String {
        format!(
            "unfed trigger: nothing is connected to {}, so the aggregate never \
             changes and {} settles in `{}` at the initial instant with no \
             transition out of it; armed that way it reports an availability \
             the model does not have. Connect that input, or drop the guard \
             and state the mode unconditionally.",
            self.ports.join(", "),
            self.automaton,
            self.state,
        )
    }
}

/// Every unfed trigger of `model`, in a deterministic order.
///
/// Read off the authored model rather than off the compiled tables,
/// which is where the two diagnostics differ: a switching loop is a
/// property of the dependency graph and survives the resolution to
/// indices, while a port that nothing feeds is a property of the
/// **topology**, and the compiled form has already resolved every port
/// to the list of attributes behind it. Naming the missing wire needs
/// the layer that still has the wire's name.
///
/// One pass over the connections, one over the definitions, then one
/// fold and one walk per automaton, each fold memoised over the
/// attributes it crosses. Called once at compile time.
#[must_use]
pub fn unfed_triggers(model: &Model) -> Vec<UnfedTrigger> {
    // An interface connection feeds ports too. A model whose interfaces
    // do not pair is refused by validation; here it is read as written.
    let resolved = model.with_resolved_connections();
    let model: &Model = match &resolved {
        Ok(resolved) => resolved,
        Err(_) => model,
    };
    let unfed = unfed_in_ports(model);
    if unfed.is_empty() {
        return Vec::new();
    }
    let definitions = definitions(model);

    let mut found = Vec::new();
    for component in &model.components {
        for automaton in &component.automata {
            // Pin what can be pinned, once per transition: the walk
            // below reads the same verdict twice, and folding a guard
            // is the only part of this that is not a lookup.
            let mut fold = Fold::new(&unfed, &definitions);
            let pinned: Vec<Option<bool>> = automaton
                .transitions
                .iter()
                .map(|transition| match &transition.guard {
                    // No guard is "always true", which no emptiness can
                    // change: such a transition is never what seals a
                    // state, and it always keeps the walk going.
                    None => Some(true),
                    Some(guard) => match fold.fold(guard) {
                        Some(Value::Bool(held)) => Some(held),
                        _ => None,
                    },
                })
                .collect();

            for state in reachable(automaton, &pinned) {
                let outgoing: Vec<usize> = automaton
                    .transitions
                    .iter()
                    .enumerate()
                    .filter(|(_, transition)| transition.source == state)
                    .map(|(index, _)| index)
                    .collect();
                // A state with no transition out of it is a terminal
                // state by declaration, and says nothing about a port.
                if outgoing.is_empty() {
                    continue;
                }
                if !outgoing.iter().all(|&index| pinned[index] == Some(false)) {
                    continue;
                }
                let mut ports = BTreeSet::new();
                for &index in &outgoing {
                    if let Some(guard) = &automaton.transitions[index].guard {
                        collect_unfed(guard, &unfed, &definitions, &mut ports);
                    }
                }
                // Sealed by something else than an empty port: a guard
                // comparing two constants seals just as well, and is a
                // declaration rather than an oversight.
                if ports.is_empty() {
                    continue;
                }
                // Sealed the pessimistic way round: the mode is stuck,
                // and stuck on a quantity the reader already sees at
                // zero in the results. Only the seal that flatters is
                // worth a warning.
                if !seal_flatters(
                    model,
                    &unfed,
                    &definitions,
                    &component.name,
                    &automaton.name,
                    &state,
                ) {
                    continue;
                }
                found.push(UnfedTrigger {
                    ports: ports.into_iter().collect(),
                    automaton: format!("{}.{}", component.name, automaton.name),
                    state,
                });
            }
        }
    }
    found
}

/// Qualified names of the in ports no connection reaches.
fn unfed_in_ports(model: &Model) -> HashSet<(String, String)> {
    let fed: HashSet<(&str, &str)> = model
        .connections
        .iter()
        .map(|connection| {
            (
                connection.to.component.as_str(),
                connection.to.port.as_str(),
            )
        })
        .collect();
    let mut unfed = HashSet::new();
    for component in &model.components {
        for port in &component.ports {
            if port.dir == PortDir::In
                && !fed.contains(&(component.name.as_str(), port.name.as_str()))
            {
                unfed.insert((component.name.clone(), port.name.clone()));
            }
        }
    }
    unfed
}

/// The states the automaton can be in, over-approximated: the initial
/// state, plus whatever a transition that is not pinned false can reach.
///
/// Over-approximating is the safe side here. A state this walk reaches
/// and that turns out to be unreachable costs one warning; a state it
/// missed would cost the diagnostic itself. Returned in declaration
/// order of the states, so the report does not depend on the walk.
fn reachable(automaton: &raichu_model::Automaton, pinned: &[Option<bool>]) -> Vec<String> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut queue = VecDeque::new();
    seen.insert(automaton.init.as_str());
    queue.push_back(automaton.init.as_str());
    while let Some(state) = queue.pop_front() {
        for (index, transition) in automaton.transitions.iter().enumerate() {
            if transition.source != state || pinned[index] == Some(false) {
                continue;
            }
            for target in &transition.targets {
                if seen.insert(target.as_str()) {
                    queue.push_back(target.as_str());
                }
            }
        }
    }
    automaton
        .states
        .iter()
        .filter(|state| seen.contains(state.as_str()))
        .cloned()
        .collect()
}

/// The value an aggregation answers over a port with no connection.
///
/// This is the engine's own answer, not a convention of this module:
/// the same empty source list reaches `eval_agg`, which sums nothing,
/// counts nothing and takes the conjunction over nothing.
fn empty_port(agg: AggOp) -> Value {
    match agg {
        AggOp::Sum | AggOp::Count => Value::Int(0),
        AggOp::All => Value::Bool(true),
        AggOp::Any => Value::Bool(false),
        AggOp::Mean | AggOp::Median => Value::Float(0.0),
    }
}

/// What writes an attribute, which is what decides whether a fold may
/// cross it to reach the port behind.
#[derive(Debug, Clone, Copy)]
enum Definition<'m> {
    /// Nothing writes it: it holds its declared initial value for the
    /// whole run, so reading it is reading a constant.
    Held(Value),
    /// One explicit equation writes it and nothing else: the attribute
    /// *is* that expression, re-solved at every evaluation point.
    Solved(&'m Expr),
    /// An ODE integrates it, a sensitive function assigns it, or two
    /// writers share it: it moves, and the fold has nothing to say.
    Moving,
}

/// What writes each **declared** attribute of the model.
///
/// Only declared attributes are keyed here. The per-connection channel
/// quantities the compiler materialises are not declared anywhere, so
/// they are absent and the fold abstains on them, which is the answer it
/// would reach anyway: a channel exists only where a connection does.
fn definitions(model: &Model) -> HashMap<(&str, &str), Definition<'_>> {
    let mut definitions: HashMap<(&str, &str), Definition<'_>> = HashMap::new();
    for component in &model.components {
        for attribute in &component.attributes {
            definitions.insert(
                (component.name.as_str(), attribute.name.as_str()),
                Definition::Held(attribute.init),
            );
        }
    }
    for component in &model.components {
        for equation in &component.equations {
            claim(
                &mut definitions,
                (component.name.as_str(), equation.target.as_str()),
                match equation.kind {
                    EquationKind::Explicit => Definition::Solved(&equation.expr),
                    EquationKind::Ode => Definition::Moving,
                },
            );
        }
        for function in &component.sensitive_functions {
            for effect in &function.effects {
                claim(
                    &mut definitions,
                    (
                        effect.target.component.as_str(),
                        effect.target.attribute.as_str(),
                    ),
                    Definition::Moving,
                );
            }
        }
    }
    definitions
}

/// Record a writer, degrading to [`Definition::Moving`] as soon as there
/// is a second one: "exactly one definition" is the condition under
/// which reading the attribute is reading its definition.
fn claim<'m>(
    definitions: &mut HashMap<(&'m str, &'m str), Definition<'m>>,
    key: (&'m str, &'m str),
    written: Definition<'m>,
) {
    if let Some(slot) = definitions.get_mut(&key) {
        *slot = match *slot {
            Definition::Held(_) => written,
            _ => Definition::Moving,
        };
    }
}

/// The mode a fold is allowed to assume: component, automaton, state,
/// and whether the automaton is assumed to **be** in that state.
type Pin<'m> = (&'m str, &'m str, &'m str, bool);

/// The constant value of an expression once every unfed port is replaced
/// by what it answers, or `None` when it has one that depends on the run.
///
/// Deliberately narrow: it settles the constructs a guard and a flow
/// equation are made of and abstains on everything else. An expression
/// it cannot settle leaves its guard unpinned, and an unpinned guard is
/// silence, so the cost of abstaining is a diagnostic that does not
/// fire. A folder that guessed would cost a diagnostic that fires on a
/// sound model, which is worse: it is how a warning stops being read.
struct Fold<'m, 'x> {
    unfed: &'x HashSet<(String, String)>,
    definitions: &'x HashMap<(&'m str, &'m str), Definition<'m>>,
    pin: Option<Pin<'m>>,
    /// The attributes on the current path, so a definition that reads
    /// its own target abstains instead of recursing for ever.
    open: HashSet<(&'m str, &'m str)>,
    memo: HashMap<(&'m str, &'m str), Option<Value>>,
}

impl<'m, 'x> Fold<'m, 'x> {
    fn new(
        unfed: &'x HashSet<(String, String)>,
        definitions: &'x HashMap<(&'m str, &'m str), Definition<'m>>,
    ) -> Self {
        Fold {
            unfed,
            definitions,
            pin: None,
            open: HashSet::new(),
            memo: HashMap::new(),
        }
    }

    /// The same fold, with one automaton assumed in (or out of) one
    /// state: what tells a seal that flatters from one that starves.
    fn assuming(
        unfed: &'x HashSet<(String, String)>,
        definitions: &'x HashMap<(&'m str, &'m str), Definition<'m>>,
        pin: Pin<'m>,
    ) -> Self {
        Fold {
            pin: Some(pin),
            ..Fold::new(unfed, definitions)
        }
    }

    fn fold(&mut self, expr: &'m Expr) -> Option<Value> {
        match expr {
            Expr::Const { value } => Some(*value),
            Expr::Attr { attr } => self.attribute(attr.component.as_str(), attr.attribute.as_str()),
            Expr::PortAgg { port, agg, .. } => {
                // A channel selector reads the per-connection attributes
                // of the same connections: no connection, no attribute,
                // same empty aggregation.
                self.unfed
                    .contains(&(port.component.clone(), port.port.clone()))
                    .then(|| empty_port(*agg))
            }
            Expr::StateActive { state } => self.state(state),
            Expr::Bool { bool_op, args } => match bool_op {
                // Short-circuit rather than require every operand: one
                // operand pinned false settles a conjunction whatever
                // the others read, and that is exactly the shape of a
                // trigger condition that also mentions something moving.
                BoolOp::And => self.connective(args, false),
                BoolOp::Or => self.connective(args, true),
                // Arity validated at model build (exactly one).
                BoolOp::Not => match self.fold(args.first()?) {
                    Some(Value::Bool(held)) => Some(Value::Bool(!held)),
                    _ => None,
                },
            },
            Expr::Cmp { cmp, lhs, rhs } => {
                let lhs = self.fold(lhs)?;
                let rhs = self.fold(rhs)?;
                compare(*cmp, lhs, rhs).map(Value::Bool)
            }
            Expr::Add { args } => self.accumulate(args, false),
            Expr::Mul { args } => self.accumulate(args, true),
            Expr::Sub { lhs, rhs } => {
                let lhs = self.fold(lhs)?;
                let rhs = self.fold(rhs)?;
                Some(match (lhs, rhs) {
                    (Value::Int(a), Value::Int(b)) => Value::Int(a.checked_sub(b)?),
                    (a, b) => Value::Float(numeric(a)? - numeric(b)?),
                })
            }
            Expr::Div { lhs, rhs } => {
                let lhs = numeric(self.fold(lhs)?)?;
                let rhs = numeric(self.fold(rhs)?)?;
                // IEEE semantics, as the engine's own division has them.
                Some(Value::Float(lhs / rhs))
            }
            Expr::Min { args } => self.extremum(args, true),
            Expr::Max { args } => self.extremum(args, false),
            Expr::If {
                cond,
                then,
                otherwise,
            } => match self.fold(cond) {
                Some(Value::Bool(true)) => self.fold(then),
                Some(Value::Bool(false)) => self.fold(otherwise),
                _ => None,
            },
            // A clock and the transcendental functions over it: nothing
            // an emptiness decides.
            Expr::Time | Expr::Sin { .. } | Expr::Exp { .. } => None,
        }
    }

    /// Reading an attribute is reading its definition, when it has
    /// exactly one and that one is an explicit equation.
    fn attribute(&mut self, component: &'m str, attribute: &'m str) -> Option<Value> {
        let key = (component, attribute);
        if let Some(settled) = self.memo.get(&key) {
            return *settled;
        }
        let definition = *self.definitions.get(&key)?;
        let value = match definition {
            Definition::Held(init) => Some(init),
            Definition::Moving => None,
            Definition::Solved(expr) => {
                if !self.open.insert(key) {
                    // Its own definition reads it back: the solver's
                    // business, not this fold's.
                    return None;
                }
                let value = self.fold(expr);
                self.open.remove(&key);
                value
            }
        };
        self.memo.insert(key, value);
        value
    }

    /// An automaton is in exactly one state at a time, so assuming it
    /// settled in one settles every other state of the same automaton.
    /// Assuming it *out* of a state settles that state alone.
    fn state(&self, state: &StateRef) -> Option<Value> {
        let (component, automaton, pinned, settled) = self.pin?;
        if state.component != component || state.automaton != automaton {
            return None;
        }
        if settled {
            Some(Value::Bool(state.state == pinned))
        } else if state.state == pinned {
            Some(Value::Bool(false))
        } else {
            None
        }
    }

    /// `and` and `or` over operands that need not all fold: `absorbing`
    /// is the operand value that settles the connective on its own.
    fn connective(&mut self, args: &'m [Expr], absorbing: bool) -> Option<Value> {
        let mut all_folded = true;
        for arg in args {
            match self.fold(arg) {
                Some(Value::Bool(held)) if held == absorbing => {
                    return Some(Value::Bool(absorbing));
                }
                Some(Value::Bool(_)) => {}
                // A non-boolean operand is a model the build would have
                // refused; abstaining is still the right answer here.
                _ => all_folded = false,
            }
        }
        all_folded.then_some(Value::Bool(!absorbing))
    }

    /// `add` and `mul`, accumulated the way the engine accumulates them:
    /// integer while every operand is one, float from the first that is
    /// not. An accumulation that would overflow abstains.
    fn accumulate(&mut self, args: &'m [Expr], product: bool) -> Option<Value> {
        let mut whole: i64 = i64::from(product);
        let mut real: f64 = if product { 1.0 } else { 0.0 };
        let mut any_float = false;
        for arg in args {
            match self.fold(arg)? {
                Value::Int(operand) => {
                    whole = if product {
                        whole.checked_mul(operand)?
                    } else {
                        whole.checked_add(operand)?
                    };
                    #[allow(clippy::cast_precision_loss)]
                    let operand = operand as f64;
                    if product {
                        real *= operand;
                    } else {
                        real += operand;
                    }
                }
                Value::Float(operand) => {
                    any_float = true;
                    if product {
                        real *= operand;
                    } else {
                        real += operand;
                    }
                }
                Value::Bool(_) => return None,
            }
        }
        Some(if any_float {
            Value::Float(real)
        } else {
            Value::Int(whole)
        })
    }

    /// `min` and `max`, which compare as floats and keep the winning
    /// operand's own kind, as the engine does.
    fn extremum(&mut self, args: &'m [Expr], take_min: bool) -> Option<Value> {
        let mut best: Option<Value> = None;
        for arg in args {
            let value = self.fold(arg)?;
            let candidate = numeric(value)?;
            best = Some(match best {
                None => value,
                Some(held) => {
                    let replaces = if take_min {
                        candidate < numeric(held)?
                    } else {
                        candidate > numeric(held)?
                    };
                    if replaces {
                        value
                    } else {
                        held
                    }
                }
            });
        }
        // Arity >= 1 validated at model build.
        best
    }
}

/// A numeric value as the engine reads it, or `None` for a boolean,
/// which no arithmetic accepts.
fn numeric(value: Value) -> Option<f64> {
    match value {
        #[allow(clippy::cast_precision_loss)]
        Value::Int(whole) => Some(whole as f64),
        Value::Float(real) => Some(real),
        Value::Bool(_) => None,
    }
}

/// Compare two constants the way the engine does, or `None` where the
/// engine would raise (ordering booleans, NaN).
fn compare(op: CmpOp, lhs: Value, rhs: Value) -> Option<bool> {
    let ordering = match (lhs, rhs) {
        (Value::Bool(a), Value::Bool(b)) => {
            return match op {
                CmpOp::Eq => Some(a == b),
                CmpOp::Ne => Some(a != b),
                _ => None,
            };
        }
        (Value::Int(a), Value::Int(b)) => a.partial_cmp(&b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(&b),
        #[allow(clippy::cast_precision_loss)]
        (Value::Int(a), Value::Float(b)) => (a as f64).partial_cmp(&b),
        #[allow(clippy::cast_precision_loss)]
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(b as f64)),
        _ => None,
    }?;
    Some(match op {
        CmpOp::Eq => ordering.is_eq(),
        CmpOp::Ne => !ordering.is_eq(),
        CmpOp::Lt => ordering.is_lt(),
        CmpOp::Le => ordering.is_le(),
        CmpOp::Gt => ordering.is_gt(),
        CmpOp::Ge => ordering.is_ge(),
    })
}

/// The unfed in ports an expression reads, qualified, whether or not the
/// fold above needed them, and **through the definitions it crosses**:
/// what the report names is the wire to add, and a reader shown one
/// operand of a condition has to go and read the other anyway.
///
/// It crosses exactly what the fold crosses, so the ports it names are
/// the ports that could have decided the guard. A definition it did not
/// cross named no port, which is what says the port played no part.
fn collect_unfed(
    expr: &Expr,
    unfed: &HashSet<(String, String)>,
    definitions: &HashMap<(&str, &str), Definition<'_>>,
    found: &mut BTreeSet<String>,
) {
    let mut open = HashSet::new();
    walk_unfed(expr, unfed, definitions, &mut open, found);
}

fn walk_unfed(
    expr: &Expr,
    unfed: &HashSet<(String, String)>,
    definitions: &HashMap<(&str, &str), Definition<'_>>,
    open: &mut HashSet<(String, String)>,
    found: &mut BTreeSet<String>,
) {
    match expr {
        Expr::PortAgg { port, .. } => {
            let key = (port.component.clone(), port.port.clone());
            if unfed.contains(&key) {
                found.insert(format!("{}.{}", key.0, key.1));
            }
        }
        Expr::Attr { attr } => {
            if let Some(Definition::Solved(definition)) = definitions
                .get(&(attr.component.as_str(), attr.attribute.as_str()))
                .copied()
            {
                let key = (attr.component.clone(), attr.attribute.clone());
                if open.insert(key.clone()) {
                    walk_unfed(definition, unfed, definitions, open, found);
                    open.remove(&key);
                }
            }
        }
        _ => {}
    }
    expr.for_each_child(&mut |child| walk_unfed(child, unfed, definitions, open, found));
}

/// Whether settling in `state` is the direction that **flatters the
/// result**: some quantity that reads the mode is nothing while the
/// automaton is elsewhere, and not nothing once it is pinned there.
///
/// This is what separates the rule that produces without consuming from
/// the rule that consumes what nothing feeds. Both are sealed by the
/// same missing wire and only one is worth a line: the second cannot
/// draw, so it produces nothing, and a component producing nothing is
/// already in front of the reader in the results.
///
/// A quantity that does not fold once the mode is pinned counts as
/// *not nothing*: the seal has removed a reason for it to be zero, which
/// is the optimism this looks for. A quantity that does not fold with
/// the mode elsewhere settles nothing and is skipped.
fn seal_flatters(
    model: &Model,
    unfed: &HashSet<(String, String)>,
    definitions: &HashMap<(&str, &str), Definition<'_>>,
    component: &str,
    automaton: &str,
    state: &str,
) -> bool {
    for expr in quantities(model) {
        if !reads_state(expr, component, automaton, state) {
            continue;
        }
        let elsewhere = Fold::assuming(unfed, definitions, (component, automaton, state, false))
            .fold(expr)
            .filter(|value| is_nothing(*value));
        if elsewhere.is_none() {
            continue;
        }
        let settled =
            Fold::assuming(unfed, definitions, (component, automaton, state, true)).fold(expr);
        if !matches!(settled, Some(value) if is_nothing(value)) {
            return true;
        }
    }
    false
}

/// Every expression the model uses to give a quantity its value: the
/// equations, the sensitive-function effects and what an allocation has
/// to distribute.
///
/// Wider than what the fold crosses, and on purpose. Crossing an
/// attribute asks "can this decide a guard", where two writers make the
/// answer no; asking what a mode is worth only needs the places a mode
/// is read, and a sensitive function is one of them: on the boolean half
/// of muscadet it is *the* one.
fn quantities(model: &Model) -> impl Iterator<Item = &Expr> {
    model.components.iter().flat_map(|component| {
        component
            .equations
            .iter()
            .map(|equation| &equation.expr)
            .chain(
                component
                    .sensitive_functions
                    .iter()
                    .flat_map(|function| function.effects.iter().map(|effect| &effect.value)),
            )
            .chain(
                component
                    .allocations
                    .iter()
                    .map(|allocation| &allocation.available),
            )
    })
}

/// Whether an expression reads one named state of one named automaton.
fn reads_state(expr: &Expr, component: &str, automaton: &str, state: &str) -> bool {
    let mut reads = false;
    expr.for_each_state_ref(&mut |reference| {
        reads |= reference.component == component
            && reference.automaton == automaton
            && reference.state == state;
    });
    reads
}

/// Whether a value is the "nothing" of its type: a quantity at zero, a
/// condition that does not hold. What a component that delivers nothing
/// publishes, and what the seal has to move for the mode to flatter.
fn is_nothing(value: Value) -> bool {
    match value {
        Value::Bool(held) => !held,
        Value::Int(whole) => whole == 0,
        Value::Float(real) => real == 0.0,
    }
}
