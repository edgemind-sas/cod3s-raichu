//! # raichu-model: the native RAICHU formalism
//!
//! Data model of the Distributed Stochastic Hybrid Automata (DSHA)
//! formalism realising Piecewise-Deterministic Markov Processes (PDMP),
//! as formalised in Desgeorges et al. 2021 (RESS).
//!
//! Design decisions:
//!
//! - **In/out ports** are the fundamental connection notion; *interfaces*
//!   group ports so an [`InterfaceConnection`] joins two groups at once.
//! - The model layer is pure data (serde), side-effect-free, validated at
//!   build time with **typed errors: never a crash on bad input**.
//! - Behaviour (guards, propagation effects) is expressed as
//!   serializable expression trees from `raichu-expr`. Trigger sets
//!   (which attribute change re-triggers which function) are *derived*
//!   from the expressions, not declared by hand: one modeller error
//!   class removed.
//!
//! Validation ([`Model::validate`]) is the single gate: a model that
//! passes is structurally sound (all references resolve, distributions are
//! well-formed, initial states exist). The engine only consumes validated
//! models.

mod algebraic;
mod connections;
mod error;
mod evaluation;
mod format;
mod graph;
mod schema;
mod validate;

pub use algebraic::{AlgebraicBlock, BlockRow, SingularReason};
pub use error::ModelError;
pub use format::{Feature, FormatHeader, LoadError, ENVELOPE_KEY, FORMAT_REVISION};
pub use graph::strongly_connected_components;
pub use schema::{
    channel_attribute_name, Allocation, AllocationPolicy, AttrKind, Attribute, Automaton, Channel,
    ChannelAttribute, Component, Connection, ConsumerParam, CycleCancellation, Distrib, Equation,
    EquationKind, EvaluationStepReads, FmuBinding, FmuParameter, FmuUnit, Indicator,
    IndicatorTarget, Interface, InterfaceConnection, InterfaceRef, InterruptionPolicy, Model,
    ModelSource, Port, PortDir, PortEquation, Program, ProgramConstraint, ProgramSense,
    ProgramTieBreak, ProgramTieObjective, ProgramVariable, Propagation, Target, Transition,
    TransitionKind, EDGE, PORT,
};

#[cfg(test)]
mod tests;
