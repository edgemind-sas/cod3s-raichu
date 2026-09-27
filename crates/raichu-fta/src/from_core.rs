//! A tree generated from a model by [`raichu_core::fault_tree`], taken as
//! it is: its basic events keep the engine's laws, its gates their
//! connectives.

use raichu_core::{BasicLaw, FaultTree, FtNode, GateOp};

use crate::{BasicEvent, Formula, Law, Tree};

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

fn formula(node: &FtNode) -> Formula {
    match node {
        FtNode::Constant { value } => Formula::Constant { value: *value },
        FtNode::Basic { event } => Formula::Event { event: *event },
        FtNode::Gate { op, children } => {
            let args = children.iter().map(formula).collect();
            match op {
                GateOp::And => Formula::And { args },
                GateOp::Or => Formula::Or { args },
                GateOp::AtLeast { k } => Formula::AtLeast { k: *k, args },
            }
        }
    }
}

impl Tree {
    /// The tree `generated` names `name`, ready to quantify.
    pub fn from_generated(generated: &FaultTree, name: &str) -> Self {
        Tree {
            name: name.to_owned(),
            events: generated
                .basic_events
                .iter()
                .map(|e| BasicEvent {
                    name: e.name.clone(),
                    law: Law::from(&e.law),
                })
                .collect(),
            gates: Vec::new(),
            top: formula(&generated.top),
        }
    }
}
