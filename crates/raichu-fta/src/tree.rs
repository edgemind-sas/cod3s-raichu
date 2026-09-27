//! A fault tree as data: basic events with their laws, named gates, and a
//! top formula over both.

use serde::Serialize;

use crate::{FtaError, Law};

/// A Boolean formula over basic events and gates.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Formula {
    /// A constant (an OpenPSA house event or `<constant>`).
    Constant {
        /// Its value.
        value: bool,
    },
    /// A basic event, by index into [`Tree::events`].
    Event {
        /// The index.
        event: usize,
    },
    /// A named gate, by index into [`Tree::gates`].
    Gate {
        /// The index.
        gate: usize,
    },
    /// Every argument holds.
    And {
        /// The arguments.
        args: Vec<Formula>,
    },
    /// At least one argument holds.
    Or {
        /// The arguments.
        args: Vec<Formula>,
    },
    /// At least `k` arguments hold.
    AtLeast {
        /// How many.
        k: usize,
        /// The arguments.
        args: Vec<Formula>,
    },
    /// The argument does not hold. A tree carrying one may be
    /// non-coherent: its probability stays exact, its minimal cut sets are
    /// not computed.
    Not {
        /// The argument.
        arg: Box<Formula>,
    },
}

impl Formula {
    /// A basic event.
    pub fn event(event: usize) -> Self {
        Formula::Event { event }
    }

    /// A conjunction.
    pub fn and(args: Vec<Formula>) -> Self {
        Formula::And { args }
    }

    /// A disjunction.
    pub fn or(args: Vec<Formula>) -> Self {
        Formula::Or { args }
    }

    /// A vote.
    pub fn at_least(k: usize, args: Vec<Formula>) -> Self {
        Formula::AtLeast { k, args }
    }

    /// A negation.
    #[allow(clippy::should_implement_trait)]
    pub fn not(arg: Formula) -> Self {
        Formula::Not { arg: Box::new(arg) }
    }
}

/// A basic event.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BasicEvent {
    /// Its name, unique in the tree.
    pub name: String,
    /// How it occurs.
    #[serde(flatten)]
    pub law: Law,
}

/// A named intermediate gate.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GateDef {
    /// Its name, unique among the gates.
    pub name: String,
    /// What it computes.
    pub formula: Formula,
}

/// A fault tree.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Tree {
    /// The tree's name.
    pub name: String,
    /// The basic events.
    pub events: Vec<BasicEvent>,
    /// The named gates the top formula may reference.
    pub gates: Vec<GateDef>,
    /// The top event.
    pub top: Formula,
}

impl Tree {
    /// Check every index resolves, every name is unique and no gate
    /// reaches itself.
    pub fn validate(&self) -> Result<(), FtaError> {
        let mut names = std::collections::HashSet::new();
        for event in &self.events {
            if !names.insert(event.name.as_str()) {
                return Err(FtaError::Invalid(format!(
                    "basic event `{}` is defined twice",
                    event.name
                )));
            }
        }
        let mut gate_names = std::collections::HashSet::new();
        for gate in &self.gates {
            if !gate_names.insert(gate.name.as_str()) {
                return Err(FtaError::Invalid(format!(
                    "gate `{}` is defined twice",
                    gate.name
                )));
            }
        }
        for gate in &self.gates {
            self.check_indices(&gate.formula)?;
        }
        self.check_indices(&self.top)?;
        // Colours: 0 unseen, 1 on the current path, 2 done.
        let mut colour = vec![0u8; self.gates.len()];
        for g in 0..self.gates.len() {
            self.check_cycle(g, &mut colour)?;
        }
        Ok(())
    }

    fn check_indices(&self, formula: &Formula) -> Result<(), FtaError> {
        match formula {
            Formula::Constant { .. } => Ok(()),
            Formula::Event { event } if *event < self.events.len() => Ok(()),
            Formula::Event { event } => Err(FtaError::Invalid(format!(
                "basic event index {event} is out of range"
            ))),
            Formula::Gate { gate } if *gate < self.gates.len() => Ok(()),
            Formula::Gate { gate } => Err(FtaError::Invalid(format!(
                "gate index {gate} is out of range"
            ))),
            Formula::And { args } | Formula::Or { args } | Formula::AtLeast { args, .. } => {
                args.iter().try_for_each(|a| self.check_indices(a))
            }
            Formula::Not { arg } => self.check_indices(arg),
        }
    }

    fn check_cycle(&self, gate: usize, colour: &mut [u8]) -> Result<(), FtaError> {
        match colour[gate] {
            2 => return Ok(()),
            1 => {
                return Err(FtaError::Invalid(format!(
                    "gate `{}` reaches itself",
                    self.gates[gate].name
                )))
            }
            _ => {}
        }
        colour[gate] = 1;
        let mut children = Vec::new();
        gates_of(&self.gates[gate].formula, &mut children);
        for child in children {
            self.check_cycle(child, colour)?;
        }
        colour[gate] = 2;
        Ok(())
    }
}

fn gates_of(formula: &Formula, out: &mut Vec<usize>) {
    match formula {
        Formula::Gate { gate } => out.push(*gate),
        Formula::And { args } | Formula::Or { args } | Formula::AtLeast { args, .. } => {
            args.iter().for_each(|a| gates_of(a, out))
        }
        Formula::Not { arg } => gates_of(arg, out),
        Formula::Constant { .. } | Formula::Event { .. } => {}
    }
}
