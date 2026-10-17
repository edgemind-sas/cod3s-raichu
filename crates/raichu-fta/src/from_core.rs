//! A tree generated from a model by [`raichu_core::fault_tree`], taken as
//! it is: its basic events keep the engine's laws, its gates their
//! connectives.

use std::collections::HashSet;

use raichu_core::{BasicLaw, FaultTree, FtNode, GateOp, TreeRepair};

use crate::{BasicEvent, Formula, GateDef, Law, Repair, Tree};

impl From<&BasicLaw> for Law {
    fn from(law: &BasicLaw) -> Self {
        match law {
            BasicLaw::Exponential { rate } => Law::Exponential { rate: *rate },
            BasicLaw::Weibull { shape, scale } => Law::Weibull {
                shape: *shape,
                scale: *scale,
            },
            BasicLaw::Lognormal { mu, sigma } => Law::Lognormal {
                mu: *mu,
                sigma: *sigma,
            },
            BasicLaw::Gamma { shape, scale } => Law::Gamma {
                shape: *shape,
                scale: *scale,
            },
            BasicLaw::Uniform { low, high } => Law::Uniform {
                low: *low,
                high: *high,
            },
            BasicLaw::Delay { time } => Law::Delay { time: *time },
            BasicLaw::Empirical { points } => Law::Empirical {
                points: points.clone(),
            },
            BasicLaw::Probability { probability } => Law::Constant {
                probability: *probability,
            },
        }
    }
}

impl From<&TreeRepair> for Repair {
    fn from(repair: &TreeRepair) -> Self {
        Repair {
            transition: repair.transition.clone(),
            law: repair.law.as_ref().map(Law::from),
            guarded: repair.guarded,
        }
    }
}

/// The formula of `node`. Every gate below the top becomes a [`GateDef`]
/// named by its occurrence identifier, so the quantifier knows each one by
/// the name the structure and the OpenPSA document give it.
fn formula(node: &FtNode, names: &mut GateNames, gates: &mut Vec<GateDef>) -> Formula {
    match node {
        FtNode::Constant { value } => Formula::Constant { value: *value },
        FtNode::Basic { event } => Formula::Event { event: *event },
        FtNode::Gate { op, children, .. } => {
            let args = children
                .iter()
                .map(|child| match child {
                    FtNode::Gate { id, .. } => {
                        let name = names.name(id.as_ref());
                        let formula = formula(child, names, gates);
                        gates.push(GateDef { name, formula });
                        Formula::Gate {
                            gate: gates.len() - 1,
                        }
                    }
                    other => formula(other, names, gates),
                })
                .collect();
            match op {
                GateOp::And => Formula::And { args },
                GateOp::Or => Formula::Or { args },
                GateOp::AtLeast { k } => Formula::AtLeast { k: *k, args },
            }
        }
    }
}

/// The name of each gate: its identifier, or a fresh `G` number no
/// identifier uses, for a tree written before gates carried them.
struct GateNames {
    taken: HashSet<String>,
    next: usize,
}

impl GateNames {
    fn of(top: &FtNode) -> Self {
        // `top` is the top's own name when the tree is written.
        let mut taken = HashSet::from(["top".to_owned()]);
        let mut pending = vec![top];
        while let Some(node) = pending.pop() {
            if let FtNode::Gate { id, children, .. } = node {
                taken.extend(id.clone());
                pending.extend(children);
            }
        }
        GateNames { taken, next: 0 }
    }

    fn name(&mut self, id: Option<&String>) -> String {
        if let Some(id) = id {
            return id.clone();
        }
        loop {
            self.next += 1;
            let name = format!("G{}", self.next);
            if self.taken.insert(name.clone()) {
                return name;
            }
        }
    }
}

impl Tree {
    /// The tree `generated` names `name`, ready to quantify: its top as
    /// the formula, every gate below it as a [`GateDef`] named by its
    /// occurrence identifier.
    pub fn from_generated(generated: &FaultTree, name: &str) -> Self {
        let mut names = GateNames::of(&generated.top);
        let mut gates = Vec::new();
        let top = formula(&generated.top, &mut names, &mut gates);
        Tree {
            name: name.to_owned(),
            events: generated
                .basic_events
                .iter()
                .map(|e| BasicEvent {
                    name: e.name.clone(),
                    law: Law::from(&e.law),
                    repairs: e.repairs.iter().map(Repair::from).collect(),
                })
                .collect(),
            gates,
            top,
        }
    }
}
