//! Fault-tree generation by backward chaining (the reference engine's
//! fault-tree generator, user manual 5.5.6 and developer manual 4.1.12;
//! cited, not copied), extended to the attributes a model computes.
//!
//! # What a fault tree explains here
//!
//! A **top**, a boolean over the model, and **why it can become true**:
//! either an expression, or the model's declared **targets** (the feared
//! events of a study, [`fault_tree_for_targets`]). The explanation rests on
//! one hypothesis: only **states** move, and a state is entered by firing a
//! transition into it. Firing a transition is explained as its **basic
//! event** (the draw of its law) AND its **source state** being reached AND
//! its **guard** holding.
//!
//! **Attributes are unrolled, not frozen.** An attribute a sensitive
//! function or an explicit equation computes is replaced by the expression
//! that computes it, recursively, down to the states and to the attributes
//! nothing writes, which keep their initial value (or a declared
//! **profile**'s). A flow variable of a model built through the muscadet
//! layer is such an attribute: frozen at its initial value, it would turn
//! every condition reading it into a constant, which is exactly how a tree
//! of probability one used to come out of a two-block parallel system. An
//! attribute something else writes (an ODE, a transition's effect, a
//! distribution operator, a linear block, a program) cannot be unrolled
//! and is refused by name; so is an attribute two places write, and a ring
//! of attributes computing each other.
//!
//! Negations are pushed down to the states once unrolled: "not in `ok`" is
//! "in one of the automaton's other states", which is how the loss of a
//! flow becomes the failure states that cause it.
//!
//! A state is explained as: it is the automaton's initial state (constant
//! true), OR one of the transitions into it fires. The recursion runs back
//! through the source states; a state already being explained further up
//! the same branch contributes nothing to it (a path that loops back adds
//! no new way in), which is how a repair loop is handled rather than
//! truncated: `nok` is reached from `ok`, `ok` is initial, and the repair
//! from `nok` back to `ok` is never needed to explain `nok`.
//!
//! **A transition no draw governs is crossed, not counted.** A delay of
//! zero (an observer's transition, an instantaneous reconfiguration) fires
//! as soon as its source and its guard hold, so it is explained by those
//! two and contributes no basic event: the feared event of a study is
//! explained by its condition, never taken for a failure of its own.
//!
//! # What a gate says
//!
//! Every gate carries an **occurrence identifier** (`top`, then `G1`,
//! `G2`, ... in preorder), unique in the tree and the same from one
//! generation of a model to the next: it names the gate in OpenPSA and in
//! the quantifier. A gate may also carry its **origin** ([`GateOrigin`]):
//! the attribute it unrolls, the state it explains being entered, or the
//! transition it explains firing. An attribute keeps its origin through the
//! negations pushed across it, marked `negated` (the loss of a flow is
//! `not flow`), and a gate with an origin is never merged into its parent.
//! The tree being strict, an attribute read in two places is two gates of
//! one origin, each with its own identifier.
//!
//! # What the number means
//!
//! A basic event's probability is its law's distribution at the mission
//! time, with no repair: the tree gives the **probability without repair**
//! of the top at that time. It equals the model's own probability when
//! every basic event is drawn from the start of the mission, independently
//! of the others: a transition out of an initial state, guarded by nothing
//! that reads a state, competing with no other transition, never undone by
//! a repair. Where one of these does not hold, the tree **over-estimates**
//! (on a coherent top, a draw counted from the start of the mission instead
//! of from when it became possible, or a repair ignored, can only add
//! failures), and [`FaultTree::warnings`] says which transition is
//! concerned and why, as a [`GenerationWarning`]: a code with its
//! parameters, whose `Display` is the English sentence.
//!
//! A repair is a fact of structure, not a warning: each basic event
//! carries the transitions that bring the state it enters back to the
//! state it leaves ([`TreeEvent::repairs`]), with their laws. What they
//! mean is decided when the tree is quantified, by its
//! [`FaultTreeMeasure`]: ignored under the probability without repair,
//! counted under the unavailability.
//!
//! # What this is not
//!
//! Not the minimal sequences of a simulation (`raichu_analysis::sequence`),
//! which order the events. A tree is a static structure; the two answer
//! different questions and are not derived from each other.
//!
//! # What is refused, by name
//!
//! - a **degenerate** tree: a top that reduces to a constant, true from the
//!   initial state or impossible, is refused rather than quantified as
//!   one or zero ([`FaultTreeError::Degenerate`]);
//! - a top, or a feared event's condition, that requires a state the model
//!   can leave to **persist** (an initial state read positively once the
//!   negations are pushed down): a static tree explains states being
//!   entered, not kept;
//! - a negation or a state read inside arithmetic this module does not
//!   recognise (anything but a vote, a count of true conditions compared
//!   with a threshold);
//! - an observer whose transition waits before firing: its state lags its
//!   condition, which a tree at one mission time cannot carry;
//! - an attribute that cannot be unrolled (see above), and a guard reading
//!   a continuous quantity's crossing.

use std::collections::{BTreeSet, HashMap, HashSet};

use raichu_expr::{AggOp, BoolOp, CmpOp, Expr, Value};
use raichu_model::TransitionKind;

use crate::compile::{AutIdx, CExpr, CLaw, CStep, CompiledModel, StateIdx, VarIdx};

/// Why a tree could not be generated.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FaultTreeError {
    /// An imported FMU is opaque to structural backward chaining.
    #[error("fault tree: FMU unit `{unit}` is opaque to structural analysis")]
    OpaqueFmu {
        /// Name of the imported unit.
        unit: String,
    },
    /// A name in the top expression, the targets or the profile designates
    /// nothing.
    #[error("fault tree: `{0}` designates no attribute, automaton or state of the model")]
    Unresolved(String),
    /// A state is read where the method cannot explain it becoming true.
    #[error(
        "fault tree: {0}. Backward chaining explains a value becoming TRUE \
         through states being entered, so it needs every expression it \
         crosses to be monotone in the states"
    )]
    NonCoherent(String),
    /// An expression form this module does not carry.
    #[error("fault tree: {0}")]
    Unsupported(String),
    /// Evaluating a frozen expression failed.
    #[error("fault tree: evaluating a frozen expression failed: {0}")]
    Evaluation(String),
    /// The tree or its cut sets outgrew the declared budget.
    #[error("fault tree: {0}")]
    TooLarge(String),
    /// A target named for the top is not one the model declares.
    #[error("fault tree: `{name}` is not a target of the model, which declares {declared:?}")]
    UnknownTarget {
        /// The name given.
        name: String,
        /// The targets the model declares.
        declared: Vec<String>,
    },
    /// The top reduces to a constant: no failure is needed to reach it, or
    /// none can. Quantified, it would read as a probability of one or
    /// zero; it is refused instead, with its cause.
    #[error("fault tree: degenerate tree, the top reduces to the constant {value}: {cause}")]
    Degenerate {
        /// The constant.
        value: bool,
        /// Why, in the model's terms.
        cause: String,
    },
}

/// How a basic event occurs: the law of the transition it stands for.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "law", rename_all = "snake_case")]
pub enum BasicLaw {
    /// Exponential time to occurrence, rate λ.
    Exponential {
        /// Rate.
        rate: f64,
    },
    /// Weibull time to occurrence.
    Weibull {
        /// Shape k.
        shape: f64,
        /// Scale λ.
        scale: f64,
    },
    /// Log-normal time to occurrence (μ, σ of the underlying normal).
    Lognormal {
        /// Mean of the underlying normal.
        mu: f64,
        /// Deviation of the underlying normal.
        sigma: f64,
    },
    /// Gamma time to occurrence.
    Gamma {
        /// Shape.
        shape: f64,
        /// Scale.
        scale: f64,
    },
    /// Uniform time to occurrence on `[low, high)`.
    Uniform {
        /// Lower bound.
        low: f64,
        /// Upper bound.
        high: f64,
    },
    /// Deterministic delay.
    Delay {
        /// The delay.
        time: f64,
    },
    /// An empirical time-to-occurrence table.
    Empirical {
        /// `(time, cumulative probability)` pairs.
        points: Vec<(f64, f64)>,
    },
    /// An on-demand branch taken with this probability.
    Probability {
        /// The branch probability.
        probability: f64,
    },
}

/// One basic event: a transition's draw (into one target, for a branching
/// transition).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TreeEvent {
    /// A name OpenPSA accepts, unique in the tree:
    /// `{component}.{automaton}.{transition}`, plus `.{target}` on a
    /// branching transition.
    pub name: String,
    /// The component the transition belongs to.
    pub component: String,
    /// The automaton, qualified (`component.automaton`).
    pub automaton: String,
    /// The transition's own name.
    pub transition: String,
    /// The state it leads to.
    pub target: String,
    /// Its law.
    #[serde(flatten)]
    pub law: BasicLaw,
    /// The transitions that bring the state it enters back to the state
    /// it leaves, the event's **repairs**, in the automaton's order. A
    /// fact of structure: whether the tree counts them is decided by the
    /// measure it is quantified in ([`FaultTreeMeasure`]). Empty when
    /// nothing brings it back; absent from a document written before
    /// repairs were carried, which reads as none.
    #[serde(default)]
    pub repairs: Vec<TreeRepair>,
}

/// One repair of a basic event: a transition from the state the event
/// enters back to the state it leaves.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TreeRepair {
    /// The transition, qualified (`component.automaton.transition`).
    pub transition: String,
    /// Its law, read as the event's own is: an on-demand branch is its
    /// branch probability, a transition no draw governs (a delay of zero,
    /// a boundary already crossed) a delay of zero. `None` when no single
    /// law stands for it (a rate that reads a state).
    pub law: Option<BasicLaw>,
    /// Whether a guard reading the model conditions it (an ObjMode2S
    /// `not_occ_cond`): a measure that counts it assumes the guard holds.
    pub guarded: bool,
}

/// What the probabilities of a quantified tree measure.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum FaultTreeMeasure {
    /// The probability that the top holds at the mission time when no
    /// repair is made: each basic event is its law's distribution there.
    /// On a coherent tree, an upper bound of the probability that the top
    /// has occurred once on the model with its repairs.
    #[default]
    ProbabilityWithoutRepair,
    /// The unavailability at the mission time: each repairable basic
    /// event is the probability of being in the state it enters, counting
    /// its repairs (in closed form for an exponential failure with
    /// exponential repairs, asymptotic `MTTR / (MTTF + MTTR)` otherwise);
    /// an event without repair keeps its distribution.
    Unavailability,
}

impl FaultTreeMeasure {
    /// The measure's name in a result document.
    pub fn as_str(self) -> &'static str {
        match self {
            FaultTreeMeasure::ProbabilityWithoutRepair => "probability_without_repair",
            FaultTreeMeasure::Unavailability => "unavailability",
        }
    }

    /// The measure a result document names, `None` for an unknown name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "probability_without_repair" => Some(FaultTreeMeasure::ProbabilityWithoutRepair),
            "unavailability" => Some(FaultTreeMeasure::Unavailability),
            _ => None,
        }
    }
}

/// The connective of a gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "gate", rename_all = "snake_case")]
pub enum GateOp {
    /// Every child holds.
    And,
    /// At least one child holds.
    Or,
    /// At least `k` children hold.
    AtLeast {
        /// How many.
        k: usize,
    },
}

/// What part of the model a gate stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OriginKind {
    /// An attribute a sensitive function or an explicit equation computes,
    /// unrolled into the states that govern it (a muscadet flow is one).
    Attribute,
    /// A state being entered: one of the ways into it.
    State,
    /// A transition firing: its draw, its source and its guard.
    Transition,
}

/// Where a gate comes from in the model, the key a reader labels it by.
///
/// An origin may repeat in one tree: the tree is strict, so an attribute
/// read in two places is unrolled twice, and each occurrence is a gate of
/// its own, with its own identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GateOrigin {
    /// What it stands for.
    pub kind: OriginKind,
    /// The component it belongs to.
    pub component: String,
    /// Its name within the component: the attribute (`out_elec`), the
    /// state (`automaton.state`) or the transition
    /// (`automaton.transition`, plus `.target` on a branching one).
    pub name: String,
    /// Whether the gate holds when the attribute is **false**: a negation
    /// pushed down across it. The feared event of a flow model is most
    /// often the loss of a flow, which reads `not flow`. Always `false` for
    /// a state or a transition, which are explained being entered or
    /// fired.
    pub negated: bool,
}

/// A node of the tree.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "node", rename_all = "snake_case")]
pub enum FtNode {
    /// A constant (only as the whole tree, once simplified).
    Constant {
        /// Its value.
        value: bool,
    },
    /// A basic event, by index into [`FaultTree::basic_events`].
    Basic {
        /// The index.
        event: usize,
    },
    /// A gate over children.
    Gate {
        /// Its connective.
        #[serde(flatten)]
        op: GateOp,
        /// Its occurrence identifier, unique in the tree and the same from
        /// one generation of a model to the next: `top` for the top gate,
        /// `G1`, `G2`, ... for the others in preorder. It names the gate in
        /// OpenPSA and in the quantifier. Every generated gate has one;
        /// absent from a tree written before gates carried them.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// What it stands for in the model; `None` for a gate the
        /// explanation builds with no single counterpart (a conjunction of
        /// a top expression, a vote).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        origin: Option<GateOrigin>,
        /// Its children.
        children: Vec<FtNode>,
    },
}

/// A gate with no identifier, labelled by `origin` if any.
fn gate(op: GateOp, origin: Option<GateOrigin>, children: Vec<FtNode>) -> FtNode {
    FtNode::Gate {
        op,
        id: None,
        origin,
        children,
    }
}

/// A generated tree.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FaultTree {
    /// The top gate, simplified: constant gates propagated, single-child
    /// gates collapsed, nested gates of one connective merged unless the
    /// nested one carries an origin. Never a constant: a degenerate top is
    /// refused.
    pub top: FtNode,
    /// Every basic event the tree reaches, in first-reached order.
    pub basic_events: Vec<TreeEvent>,
    /// Why the tree's probability may exceed the model's own probability
    /// without repair, one coded warning per transition and reason
    /// concerned: a draw timed from a state entered during the mission, a
    /// guard reading a state, a competing transition, a state entered and
    /// left for another than the one it came from. Equal warnings are
    /// listed once. Empty when the tree is the model's exact structure.
    /// Repairs are not warned here: they are carried by the events
    /// ([`TreeEvent::repairs`]), and the warnings that depend on the
    /// measure are the quantification's. Serialised with each warning's
    /// `message` ([`serialize_coded`]).
    /// A document of version 1 carried plain sentences: each reads as
    /// [`GenerationWarning::Legacy`] ([`deserialize_coded`]).
    #[serde(
        serialize_with = "serialize_coded",
        deserialize_with = "deserialize_coded"
    )]
    pub warnings: Vec<GenerationWarning>,
}

/// Why a tree's number may differ from the model's own: a code (`kind`)
/// with its parameters, the English sentence being its `Display` (its
/// [`GenerationWarning::message`] under the probability without repair).
/// Consumers key on `kind`, never on the text.
///
/// Generation emits the structural codes (`guard_reads_state`,
/// `entered_during_mission`, `competing_transition`, `target_left`). The
/// others depend on the measure the tree is quantified in, and are
/// emitted by the quantification's envelope, once per basic event:
/// `repair_ignored`, `asymptotic_unavailability`,
/// `common_cause_repair_approximation`, `guarded_repair_assumed_available`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GenerationWarning {
    /// The transition is guarded by a condition on states: its draw is
    /// timed from the start of the mission, not from when the condition
    /// holds.
    GuardReadsState {
        /// Qualified transition name.
        transition: String,
    },
    /// The transition leaves a state other than its automaton's initial
    /// one: its draw is timed from the start of the mission, not from when
    /// that state is entered.
    EnteredDuringMission {
        /// Qualified transition name.
        transition: String,
        /// The state it leaves.
        source: String,
    },
    /// Other transitions leave the same source state: the draw is counted
    /// as if none of them could fire first.
    CompetingTransition {
        /// Qualified transition name.
        transition: String,
        /// The state they all leave.
        source: String,
        /// The competing transitions, qualified, in the automaton's order.
        competitors: Vec<String>,
    },
    /// The state a basic event enters is left for a state other than the
    /// one the event leaves: not a repair of the event, and the tree
    /// ignores it.
    TargetLeft {
        /// The automaton, qualified.
        automaton: String,
        /// The state left.
        state: String,
        /// The transitions that leave it for another state, qualified, in
        /// the automaton's order.
        transitions: Vec<String>,
    },
    /// The measure does not count the event's repairs: the tree gives
    /// the event's probability without repair.
    RepairIgnored {
        /// The basic event.
        event: String,
        /// The automaton, qualified.
        automaton: String,
        /// The state its repairs leave.
        state: String,
        /// Its repairs, qualified, in the automaton's order.
        transitions: Vec<String>,
    },
    /// Under the unavailability, a repairable event whose failure or
    /// repair law is not exponential is valued by its asymptotic
    /// unavailability `MTTR / (MTTF + MTTR)`, an approximation of unknown
    /// direction at a finite mission time.
    AsymptoticUnavailability {
        /// The basic event.
        event: String,
    },
    /// Under the unavailability, a common-cause combination (`__cc_`) is
    /// valued with its own repair, as if independent of the other
    /// combinations of its mode.
    CommonCauseRepairApproximation {
        /// The basic event.
        event: String,
    },
    /// Under the unavailability, a guarded repair is counted as if its
    /// guard always held.
    GuardedRepairAssumedAvailable {
        /// The basic event.
        event: String,
        /// The repair, qualified.
        transition: String,
    },
    /// A warning read from a document of version 1, which carried a plain
    /// English sentence and no code. Never produced: a reader meets it in
    /// an old document only.
    Legacy {
        /// The sentence, as the document carried it.
        message: String,
    },
}

impl std::fmt::Display for GenerationWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message(FaultTreeMeasure::ProbabilityWithoutRepair))
    }
}

impl GenerationWarning {
    /// The English sentence of the warning under `measure`. A structural
    /// code says the tree over-estimates under the probability without
    /// repair; under the unavailability the direction is not known, and
    /// its sentence only says what the tree does.
    pub fn message(&self, measure: FaultTreeMeasure) -> String {
        let quoted = |names: &[String]| {
            names
                .iter()
                .map(|n| format!("`{n}`"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let consequence = match measure {
            FaultTreeMeasure::ProbabilityWithoutRepair => ", so it over-estimates",
            FaultTreeMeasure::Unavailability => "",
        };
        match self {
            GenerationWarning::GuardReadsState { transition } => format!(
                "transition `{transition}` is guarded by a condition on states; the tree \
                 times its draw from the start of the mission, not from when the \
                 condition holds{consequence}"
            ),
            GenerationWarning::EnteredDuringMission { transition, source } => format!(
                "transition `{transition}` leaves `{source}`, a state entered during the \
                 mission; the tree times its draw from the start of the \
                 mission{consequence}"
            ),
            GenerationWarning::CompetingTransition {
                transition,
                source,
                competitors,
            } => format!(
                "transition `{transition}` competes with {} from `{source}`; the tree \
                 counts its draw as if nothing could fire first{consequence}",
                quoted(competitors)
            ),
            GenerationWarning::TargetLeft {
                automaton,
                state,
                transitions,
            } => format!(
                "`{automaton}.{state}` is left for another state by {}; the tree \
                 ignores it{consequence}",
                quoted(transitions)
            ),
            GenerationWarning::RepairIgnored {
                event,
                automaton,
                state,
                transitions,
            } => format!(
                "basic event `{event}`: `{automaton}.{state}` is left again by {}; \
                 the tree ignores it and gives the probability without repair",
                quoted(transitions)
            ),
            GenerationWarning::AsymptoticUnavailability { event } => format!(
                "basic event `{event}`: a failure or repair law is not exponential, \
                 so its unavailability is the asymptotic MTTR / (MTTF + MTTR), \
                 which may be above or below the unavailability at the mission time"
            ),
            GenerationWarning::CommonCauseRepairApproximation { event } => format!(
                "basic event `{event}`: a common-cause combination is valued with \
                 its own repair, as if independent of the mode's other combinations"
            ),
            GenerationWarning::GuardedRepairAssumedAvailable { event, transition } => format!(
                "basic event `{event}`: repair `{transition}` is guarded; its \
                 unavailability counts it as if the guard always held"
            ),
            GenerationWarning::Legacy { message } => message.clone(),
        }
    }
}

impl LegacyWarning for GenerationWarning {
    fn legacy(message: String) -> Self {
        GenerationWarning::Legacy { message }
    }

    fn is_legacy(&self) -> bool {
        matches!(self, GenerationWarning::Legacy { .. })
    }
}

/// A coded warning that also reads the plain sentence a document of
/// version 1 carried in its place.
pub trait LegacyWarning: Sized {
    /// The variant `{kind: "legacy", message}` standing for `message`.
    fn legacy(message: String) -> Self;
    /// Whether this is that variant.
    fn is_legacy(&self) -> bool;
}

/// Read coded warnings: each an object tagged by `kind` (its `message`
/// ignored, being derived), or a plain sentence, as a document of version
/// 1 carried, which reads as the legacy variant
/// ([`LegacyWarning::legacy`]). A reader of a later version refuses a
/// legacy warning once the document is read.
///
/// # Errors
/// The deserializer's, or a warning that is neither a sentence nor one of
/// the codes.
pub fn deserialize_coded<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: serde::de::DeserializeOwned + LegacyWarning,
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let entries = <Vec<serde_json::Value> as serde::Deserialize>::deserialize(deserializer)?;
    entries
        .into_iter()
        .map(|entry| match entry {
            serde_json::Value::String(message) => Ok(T::legacy(message)),
            other => serde_json::from_value(other).map_err(D::Error::custom),
        })
        .collect()
}

/// Serialise coded warnings as their tagged form (`kind` and its
/// parameters) plus a readable `message`, their `Display`: the form every
/// result document carries. Reading ignores `message`, which is derived.
///
/// # Errors
/// The serializer's, or a warning that does not serialise to an object.
pub fn serialize_coded<T, S>(warnings: &[T], serializer: S) -> Result<S::Ok, S::Error>
where
    T: serde::Serialize + std::fmt::Display,
    S: serde::Serializer,
{
    use serde::ser::{Error, SerializeSeq};
    let mut seq = serializer.serialize_seq(Some(warnings.len()))?;
    for warning in warnings {
        let mut entry = serde_json::to_value(warning).map_err(S::Error::custom)?;
        let Some(fields) = entry.as_object_mut() else {
            return Err(S::Error::custom("a coded warning serialises to an object"));
        };
        fields.insert(
            "message".to_owned(),
            serde_json::Value::String(warning.to_string()),
        );
        seq.serialize_element(&entry)?;
    }
    seq.end()
}

/// What a generation may be given beside the top.
#[derive(Debug, Clone, Default)]
pub struct FaultTreeSettings {
    /// Attributes held at a value other than the one the model gives them,
    /// by qualified name (`component.attribute`): the reference method's
    /// profiles. A profiled attribute is a constant, whatever computes it.
    pub profile: Vec<(String, Value)>,
    /// The most nodes the unsimplified tree may hold, and the most nodes
    /// the unrolled expressions may hold; `None` for the default of one
    /// million.
    pub max_nodes: Option<usize>,
}

/// Default node budget for generation and unrolling.
pub const DEFAULT_MAX_NODES: usize = 1_000_000;

/// Generate the fault tree explaining `top`, an expression over the
/// model's states and attributes, on `model`.
pub fn fault_tree(
    model: &CompiledModel,
    top: &Expr,
    settings: &FaultTreeSettings,
) -> Result<FaultTree, FaultTreeError> {
    explain(model, top, settings)?.into_tree()
}

/// What backward chaining makes of a top: a tree, or a constant.
///
/// [`fault_tree`] and [`fault_tree_for_targets`] refuse the constant as a
/// degenerate tree, because a constant quantified is a probability of one
/// or zero that reads like a result. A consumer that treats a constant as
/// an answer of its own (a target reached from the start, a target no
/// failure can reach) reads it here instead.
#[derive(Debug, Clone, PartialEq)]
pub enum Explanation {
    /// A tree whose top reads at least one basic event.
    Tree(FaultTree),
    /// The top reduces to this constant, for this reason.
    Constant {
        /// The constant.
        value: bool,
        /// Why, in the model's terms.
        cause: String,
    },
}

impl Explanation {
    /// The tree, or [`FaultTreeError::Degenerate`] for a constant.
    pub fn into_tree(self) -> Result<FaultTree, FaultTreeError> {
        match self {
            Explanation::Tree(tree) => Ok(tree),
            Explanation::Constant { value, cause } => {
                Err(FaultTreeError::Degenerate { value, cause })
            }
        }
    }
}

/// Explain `top` on `model` as [`fault_tree`] does, handing a constant
/// back as an [`Explanation::Constant`] rather than refusing it.
pub fn explain(
    model: &CompiledModel,
    top: &Expr,
    settings: &FaultTreeSettings,
) -> Result<Explanation, FaultTreeError> {
    refuse_fmu(model)?;
    let top = compile_top(model, top)?;
    generate(model, &top, settings)
}

/// Generate the fault tree explaining why one of the model's declared
/// **targets** (feared events, `targets` of the model document) is
/// reached: the top is the disjunction of the targets' states, and each
/// target's state is explained through the transition that enters it, an
/// observer's being crossed as its condition.
pub fn fault_tree_for_targets(
    model: &CompiledModel,
    targets: &[String],
    settings: &FaultTreeSettings,
) -> Result<FaultTree, FaultTreeError> {
    refuse_fmu(model)?;
    if targets.is_empty() {
        return Err(FaultTreeError::Unsupported(
            "a tree over targets needs at least one target".to_owned(),
        ));
    }
    let mut args = Vec::new();
    for name in targets {
        let target = model
            .targets
            .iter()
            .find(|t| &t.name == name)
            .ok_or_else(|| FaultTreeError::UnknownTarget {
                name: name.clone(),
                declared: model.targets.iter().map(|t| t.name.clone()).collect(),
            })?;
        args.push(CExpr::StateActive {
            automaton: target.automaton,
            state: target.state,
        });
    }
    let top = if args.len() == 1 {
        args.remove(0)
    } else {
        CExpr::Bool {
            op: BoolOp::Or,
            args,
        }
    };
    generate(model, &top, settings)?.into_tree()
}

fn refuse_fmu(model: &CompiledModel) -> Result<(), FaultTreeError> {
    match model.fmu_units.first() {
        Some(unit) => Err(FaultTreeError::OpaqueFmu {
            unit: unit.name.clone(),
        }),
        None => Ok(()),
    }
}

fn generate(
    model: &CompiledModel,
    top: &CExpr,
    settings: &FaultTreeSettings,
) -> Result<Explanation, FaultTreeError> {
    let mut vars = model.var_init.clone();
    let mut profiled = vec![false; vars.len()];
    for (name, value) in &settings.profile {
        let index = *model
            .var_index
            .get(name)
            .ok_or_else(|| FaultTreeError::Unresolved(name.clone()))?;
        vars[index] = *value;
        profiled[index] = true;
    }
    let max_nodes = settings.max_nodes.unwrap_or(DEFAULT_MAX_NODES);
    let mut builder = Builder {
        model,
        vars,
        profiled,
        writers: writers(model),
        resolution: vec![Resolution::Pending; model.var_init.len()],
        var_stack: Vec::new(),
        unrolled_nodes: 0,
        init_states: model.automata.iter().map(|a| a.init).collect(),
        events: Vec::new(),
        event_index: HashMap::new(),
        stack: Vec::new(),
        nodes: 0,
        max_nodes,
        warnings: Vec::new(),
        warned: HashSet::new(),
        constant_conditions: Vec::new(),
    };
    let unrolled = builder.unroll(top)?;
    let normal = builder.nnf(&unrolled, false)?;
    builder.refuse_persistence(&normal, "the top")?;
    let raw = builder.expand(&normal)?;
    let mut simplified = simplify(raw);
    if let FtNode::Constant { value } = simplified {
        return Ok(Explanation::Constant {
            value,
            cause: builder.degenerate_cause(top, &unrolled, value),
        });
    }
    identify(&mut simplified);
    Ok(Explanation::Tree(FaultTree {
        top: simplified,
        basic_events: builder.events,
        warnings: builder.warnings,
    }))
}

/// Who computes an attribute.
#[derive(Debug, Clone)]
enum Writer {
    /// A sensitive function's effect, or an explicit equation: the
    /// attribute is this expression of the model, once settled.
    Unrollable { by: String, expr: CExpr },
    /// Something whose result is not an expression of the current state
    /// (an ODE, a transition's effect, a distribution operator, a linear
    /// block, a program), named for the refusal.
    Opaque(String),
}

/// Every writer of every attribute, in model order.
fn writers(model: &CompiledModel) -> Vec<Vec<Writer>> {
    let mut out: Vec<Vec<Writer>> = vec![Vec::new(); model.var_init.len()];
    for function in &model.functions {
        for (target, expr) in &function.effects {
            out[*target].push(Writer::Unrollable {
                by: format!("the sensitive function `{}`", function.name),
                expr: expr.clone(),
            });
        }
    }
    for step in &model.explicit {
        match step {
            CStep::Equation { target, expr } => out[*target].push(Writer::Unrollable {
                by: "an explicit equation".to_owned(),
                expr: expr.clone(),
            }),
            CStep::Allocate(allocation) => {
                for target in &allocation.allocated {
                    out[*target].push(Writer::Opaque(format!(
                        "the distribution operator `{}`",
                        allocation.name
                    )));
                }
            }
            CStep::Block(block) => {
                for target in &block.targets {
                    out[*target].push(Writer::Opaque(format!("the linear block `{}`", block.name)));
                }
            }
        }
    }
    for (target, _) in &model.ode {
        out[*target].push(Writer::Opaque("an ODE".to_owned()));
    }
    for transition in &model.transitions {
        for (target, _) in &transition.effects {
            out[*target].push(Writer::Opaque(format!(
                "an effect of transition `{}`",
                transition.name
            )));
        }
    }
    for program in &model.programs {
        let written = program
            .variables
            .iter()
            .map(|v| v.target)
            .chain(std::iter::once(program.feasible))
            .chain(program.objective_value);
        for target in written {
            out[target].push(Writer::Opaque(format!("the program `{}`", program.name)));
        }
    }
    out
}

/// What a transition's law stands for in a tree.
enum Drawn {
    /// A law that draws.
    Law(BasicLaw),
    /// A transition no draw governs: crossed (`true`) or never fired.
    Constant(bool),
}

/// Where the unrolling of one attribute stands.
#[derive(Debug, Clone)]
enum Resolution {
    Pending,
    Visiting,
    /// Reads no state once unrolled: this value, written into the
    /// builder's attribute table.
    Settled,
    /// Reads states: this expression, and its size in nodes.
    Dynamic(TExpr, usize),
}

/// An expression of the generator: the engine's [`CExpr`] as leaves, the
/// connectives the explanation reads as nodes of its own, and the
/// attributes the unrolling crossed kept as labels, so a gate can say
/// which attribute it stands for. Local to generation: the engine's
/// expression is not touched.
///
/// The unrolling builds `Bool`, `Cmp`, `Add` and `If` as nodes of their
/// own; a `Plain` that reads a state is then a state, or an arithmetic
/// form the explanation refuses.
#[derive(Debug, Clone, PartialEq)]
enum TExpr {
    /// An expression no label sits in.
    Plain(CExpr),
    /// What computes an attribute, labelled with it.
    Labelled {
        origin: GateOrigin,
        inner: Box<TExpr>,
    },
    Bool {
        op: BoolOp,
        args: Vec<TExpr>,
    },
    Cmp {
        op: CmpOp,
        lhs: Box<TExpr>,
        rhs: Box<TExpr>,
    },
    Add {
        args: Vec<TExpr>,
    },
    If {
        cond: Box<TExpr>,
        then: Box<TExpr>,
        otherwise: Box<TExpr>,
    },
}

impl TExpr {
    fn constant(value: bool) -> Self {
        TExpr::Plain(CExpr::Const(Value::Bool(value)))
    }

    fn int(value: i64) -> Self {
        TExpr::Plain(CExpr::Const(Value::Int(value)))
    }

    fn is_constant(&self, value: bool) -> bool {
        matches!(self, TExpr::Plain(CExpr::Const(Value::Bool(b))) if *b == value)
    }

    /// The engine's expression, labels dropped: what the value is.
    fn plain(&self) -> CExpr {
        match self {
            TExpr::Plain(expr) => expr.clone(),
            TExpr::Labelled { inner, .. } => inner.plain(),
            TExpr::Bool { op, args } => CExpr::Bool {
                op: *op,
                args: args.iter().map(TExpr::plain).collect(),
            },
            TExpr::Cmp { op, lhs, rhs } => CExpr::Cmp {
                op: *op,
                lhs: Box::new(lhs.plain()),
                rhs: Box::new(rhs.plain()),
            },
            TExpr::Add { args } => CExpr::Add {
                args: args.iter().map(TExpr::plain).collect(),
            },
            TExpr::If {
                cond,
                then,
                otherwise,
            } => CExpr::If {
                cond: Box::new(cond.plain()),
                then: Box::new(then.plain()),
                otherwise: Box::new(otherwise.plain()),
            },
        }
    }

    /// The expression under its outer labels.
    fn peeled(&self) -> &TExpr {
        match self {
            TExpr::Labelled { inner, .. } => inner.peeled(),
            other => other,
        }
    }

    fn reads_state(&self) -> bool {
        match self {
            TExpr::Plain(expr) => reads_state(expr),
            TExpr::Labelled { inner, .. } => inner.reads_state(),
            TExpr::Bool { args, .. } | TExpr::Add { args } => args.iter().any(TExpr::reads_state),
            TExpr::Cmp { lhs, rhs, .. } => lhs.reads_state() || rhs.reads_state(),
            TExpr::If {
                cond,
                then,
                otherwise,
            } => cond.reads_state() || then.reads_state() || otherwise.reads_state(),
        }
    }

    /// Nodes of the expression, labels not counted: the unrolling budget
    /// is the engine expression's.
    fn size(&self) -> usize {
        match self {
            TExpr::Plain(expr) => size_of(expr),
            TExpr::Labelled { inner, .. } => inner.size(),
            TExpr::Bool { args, .. } | TExpr::Add { args } => {
                1 + args.iter().map(TExpr::size).sum::<usize>()
            }
            TExpr::Cmp { lhs, rhs, .. } => 1 + lhs.size() + rhs.size(),
            TExpr::If {
                cond,
                then,
                otherwise,
            } => 1 + cond.size() + then.size() + otherwise.size(),
        }
    }

    fn visit_states(&self, visit: &mut impl FnMut(AutIdx, StateIdx)) {
        match self {
            TExpr::Plain(expr) => visit_states(expr, visit),
            TExpr::Labelled { inner, .. } => inner.visit_states(visit),
            TExpr::Bool { args, .. } | TExpr::Add { args } => {
                args.iter().for_each(|a| a.visit_states(visit))
            }
            TExpr::Cmp { lhs, rhs, .. } => {
                lhs.visit_states(visit);
                rhs.visit_states(visit);
            }
            TExpr::If {
                cond,
                then,
                otherwise,
            } => {
                cond.visit_states(visit);
                then.visit_states(visit);
                otherwise.visit_states(visit);
            }
        }
    }
}

struct Builder<'a> {
    model: &'a CompiledModel,
    /// Attribute values: the initial ones, the profile's, and the settled
    /// value of every attribute unrolled to a constant.
    vars: Vec<Value>,
    profiled: Vec<bool>,
    writers: Vec<Vec<Writer>>,
    resolution: Vec<Resolution>,
    var_stack: Vec<VarIdx>,
    unrolled_nodes: usize,
    init_states: Vec<StateIdx>,
    events: Vec<TreeEvent>,
    event_index: HashMap<(usize, StateIdx), usize>,
    stack: Vec<(AutIdx, StateIdx)>,
    nodes: usize,
    max_nodes: usize,
    warnings: Vec<GenerationWarning>,
    warned: HashSet<GenerationWarning>,
    /// Conditions crossed that read no state once unrolled, for the cause
    /// of a degenerate tree.
    constant_conditions: Vec<String>,
}

impl Builder<'_> {
    fn count(&mut self) -> Result<(), FaultTreeError> {
        self.nodes += 1;
        if self.nodes > self.max_nodes {
            return Err(FaultTreeError::TooLarge(format!(
                "the tree outgrew {} nodes; raise `max_nodes` or explain a \
                 narrower top expression",
                self.max_nodes
            )));
        }
        Ok(())
    }

    fn grow_unrolled(&mut self, size: usize) -> Result<(), FaultTreeError> {
        self.unrolled_nodes += size;
        if self.unrolled_nodes > self.max_nodes {
            return Err(FaultTreeError::TooLarge(format!(
                "the attributes unrolled into the states outgrew {} expression \
                 nodes; raise `max_nodes` or explain a narrower top",
                self.max_nodes
            )));
        }
        Ok(())
    }

    /// Record `warning` once: an equal warning met again (the same
    /// transition explained on another branch) is not repeated.
    fn warn(&mut self, warning: GenerationWarning) {
        if self.warned.insert(warning.clone()) {
            self.warnings.push(warning);
        }
    }

    fn frozen(&self, expr: &CExpr) -> Result<Value, FaultTreeError> {
        crate::engine::eval_frozen(self.model, &self.vars, &self.init_states, expr)
            .map_err(|e| FaultTreeError::Evaluation(e.to_string()))
    }

    fn frozen_bool(&self, expr: &CExpr) -> Result<bool, FaultTreeError> {
        match self.frozen(expr)? {
            Value::Bool(b) => Ok(b),
            other => Err(FaultTreeError::Evaluation(format!(
                "expected a boolean, got {other:?}"
            ))),
        }
    }

    fn state_name(&self, automaton: AutIdx, state: StateIdx) -> String {
        let aut = &self.model.automata[automaton];
        format!("{}.{}", aut.name, aut.states[state])
    }

    fn frozen_t(&self, expr: &TExpr) -> Result<Value, FaultTreeError> {
        self.frozen(&expr.plain())
    }

    fn frozen_bool_t(&self, expr: &TExpr) -> Result<bool, FaultTreeError> {
        self.frozen_bool(&expr.plain())
    }

    /// The origin of a gate standing for attribute `index`.
    fn attribute_origin(&self, index: VarIdx, negated: bool) -> GateOrigin {
        let component = &self.model.var_components[index];
        let qualified = &self.model.var_names[index];
        GateOrigin {
            kind: OriginKind::Attribute,
            component: component.clone(),
            name: local(qualified, component).to_owned(),
            negated,
        }
    }

    /// The origin of the gate explaining `state` being entered.
    fn state_origin(&self, automaton: AutIdx, state: StateIdx) -> GateOrigin {
        let component = &self.model.automaton_components[automaton];
        let aut = &self.model.automata[automaton];
        GateOrigin {
            kind: OriginKind::State,
            component: component.clone(),
            name: format!("{}.{}", local(&aut.name, component), aut.states[state]),
            negated: false,
        }
    }

    /// The origin of the gate explaining transition `t_idx` firing into
    /// its `position`-th target.
    fn transition_origin(&self, t_idx: usize, position: usize) -> GateOrigin {
        let transition = &self.model.transitions[t_idx];
        let mut name = local(&transition.name, &transition.component).to_owned();
        if transition.targets.len() > 1 {
            let automaton = &self.model.automata[transition.automaton];
            name = format!("{name}.{}", automaton.states[transition.targets[position]]);
        }
        GateOrigin {
            kind: OriginKind::Transition,
            component: transition.component.clone(),
            name,
            negated: false,
        }
    }

    // --- unrolling ------------------------------------------------------

    /// `expr` with every attribute replaced by what computes it, down to
    /// the states and the constants, each boolean attribute that reads a
    /// state labelled with its name.
    fn unroll(&mut self, expr: &CExpr) -> Result<TExpr, FaultTreeError> {
        self.grow_unrolled(1)?;
        let all = |this: &mut Self, args: &[CExpr]| -> Result<Vec<TExpr>, FaultTreeError> {
            args.iter().map(|a| this.unroll(a)).collect()
        };
        let plain = |this: &mut Self, args: &[CExpr]| -> Result<Vec<CExpr>, FaultTreeError> {
            args.iter().map(|a| Ok(this.unroll(a)?.plain())).collect()
        };
        let boxed = |this: &mut Self, arg: &CExpr| -> Result<Box<CExpr>, FaultTreeError> {
            Ok(Box::new(this.unroll(arg)?.plain()))
        };
        Ok(match expr {
            CExpr::Const(_) | CExpr::StateActive { .. } => TExpr::Plain(expr.clone()),
            CExpr::Var(index) => self.resolve(*index)?,
            CExpr::PortAgg { sources, agg } => self.unroll_aggregate(sources, *agg)?,
            CExpr::Time => {
                return Err(FaultTreeError::Unsupported(
                    "an expression the tree crosses reads the clock, which a \
                     static structure has no value for"
                        .to_owned(),
                ))
            }
            CExpr::Cmp { op, lhs, rhs } => TExpr::Cmp {
                op: *op,
                lhs: Box::new(self.unroll(lhs)?),
                rhs: Box::new(self.unroll(rhs)?),
            },
            CExpr::Bool { op, args } => TExpr::Bool {
                op: *op,
                args: all(self, args)?,
            },
            CExpr::Add { args } => TExpr::Add {
                args: all(self, args)?,
            },
            CExpr::If {
                cond,
                then,
                otherwise,
            } => TExpr::If {
                cond: Box::new(self.unroll(cond)?),
                then: Box::new(self.unroll(then)?),
                otherwise: Box::new(self.unroll(otherwise)?),
            },
            // Arithmetic the explanation never reads a state through: its
            // value, labels dropped.
            CExpr::Mul { args } => TExpr::Plain(CExpr::Mul {
                args: plain(self, args)?,
            }),
            CExpr::Min { args } => TExpr::Plain(CExpr::Min {
                args: plain(self, args)?,
            }),
            CExpr::Max { args } => TExpr::Plain(CExpr::Max {
                args: plain(self, args)?,
            }),
            CExpr::Sub { lhs, rhs } => TExpr::Plain(CExpr::Sub {
                lhs: boxed(self, lhs)?,
                rhs: boxed(self, rhs)?,
            }),
            CExpr::Div { lhs, rhs } => TExpr::Plain(CExpr::Div {
                lhs: boxed(self, lhs)?,
                rhs: boxed(self, rhs)?,
            }),
            CExpr::Sin(arg) => TExpr::Plain(CExpr::Sin(boxed(self, arg)?)),
            CExpr::Exp(arg) => TExpr::Plain(CExpr::Exp(boxed(self, arg)?)),
        })
    }

    /// The attribute `index`, unrolled: a constant, or an expression over
    /// states, labelled with the attribute when it is a boolean.
    fn resolve(&mut self, index: VarIdx) -> Result<TExpr, FaultTreeError> {
        if self.profiled[index] {
            return Ok(TExpr::Plain(CExpr::Const(self.vars[index])));
        }
        match &self.resolution[index] {
            Resolution::Settled => return Ok(TExpr::Plain(CExpr::Const(self.vars[index]))),
            Resolution::Dynamic(expr, size) => {
                let (expr, size) = (expr.clone(), *size);
                self.grow_unrolled(size)?;
                return Ok(expr);
            }
            Resolution::Visiting => {
                let from = self.var_stack.iter().position(|&v| v == index).unwrap_or(0);
                let ring: Vec<&str> = self.var_stack[from..]
                    .iter()
                    .chain(std::iter::once(&index))
                    .map(|&v| self.model.var_names[v].as_str())
                    .collect();
                return Err(FaultTreeError::Unsupported(format!(
                    "the attributes {} compute each other, a ring the tree cannot \
                     unroll into the states that govern it",
                    ring.join(" -> ")
                )));
            }
            Resolution::Pending => {}
        }
        let name = &self.model.var_names[index];
        let expr = match self.writers[index].as_slice() {
            // Nothing writes it: a constant of the model.
            [] => {
                self.resolution[index] = Resolution::Settled;
                return Ok(TExpr::Plain(CExpr::Const(self.vars[index])));
            }
            [Writer::Unrollable { expr, .. }] => expr.clone(),
            [Writer::Opaque(by)] => {
                return Err(FaultTreeError::Unsupported(format!(
                    "attribute `{name}` is computed by {by}, which the tree cannot \
                     unroll into the states that govern it (only a sensitive \
                     function's effect or an explicit equation can be)"
                )))
            }
            several => {
                let by: Vec<String> = several
                    .iter()
                    .map(|w| match w {
                        Writer::Unrollable { by, .. } => by.clone(),
                        Writer::Opaque(by) => by.clone(),
                    })
                    .collect();
                return Err(FaultTreeError::Unsupported(format!(
                    "attribute `{name}` is written in {} places ({}), so no single \
                     expression of the states says what it is",
                    several.len(),
                    by.join(", ")
                )));
            }
        };
        self.resolution[index] = Resolution::Visiting;
        self.var_stack.push(index);
        let unrolled = self.unroll(&expr);
        self.var_stack.pop();
        let unrolled = unrolled?;
        if unrolled.reads_state() {
            let size = unrolled.size();
            // A boolean is a condition a gate can stand for; a number is
            // read by a comparison, which is the gate.
            let unrolled = if matches!(self.model.var_init[index], Value::Bool(_)) {
                TExpr::Labelled {
                    origin: self.attribute_origin(index, false),
                    inner: Box::new(unrolled),
                }
            } else {
                unrolled
            };
            self.resolution[index] = Resolution::Dynamic(unrolled.clone(), size);
            Ok(unrolled)
        } else {
            let value = self.frozen_t(&unrolled)?;
            self.vars[index] = value;
            self.resolution[index] = Resolution::Settled;
            Ok(TExpr::Plain(CExpr::Const(value)))
        }
    }

    /// A port aggregation over unrolled sources: a constant when none
    /// reads a state, else the connective it stands for.
    fn unroll_aggregate(
        &mut self,
        sources: &[VarIdx],
        agg: AggOp,
    ) -> Result<TExpr, FaultTreeError> {
        let parts = sources
            .iter()
            .map(|&s| self.resolve(s))
            .collect::<Result<Vec<_>, _>>()?;
        if !parts.iter().any(TExpr::reads_state) {
            // Every source settled: the aggregation reads their values.
            return Ok(TExpr::Plain(CExpr::Const(self.frozen(
                &CExpr::PortAgg {
                    sources: sources.to_vec(),
                    agg,
                },
            )?)));
        }
        Ok(match agg {
            AggOp::Any => TExpr::Bool {
                op: BoolOp::Or,
                args: parts,
            },
            AggOp::All => TExpr::Bool {
                op: BoolOp::And,
                args: parts,
            },
            AggOp::Count => TExpr::int(sources.len() as i64),
            AggOp::Sum => TExpr::Add {
                args: parts
                    .into_iter()
                    .zip(sources)
                    .map(|(part, &source)| match self.model.var_init[source] {
                        // A boolean counts one when true: written as the
                        // vote term the tree recognises.
                        Value::Bool(_) => TExpr::If {
                            cond: Box::new(part),
                            then: Box::new(TExpr::int(1)),
                            otherwise: Box::new(TExpr::int(0)),
                        },
                        _ => part,
                    })
                    .collect(),
            },
            AggOp::Mean | AggOp::Median => {
                return Err(FaultTreeError::Unsupported(format!(
                    "a {agg:?} over a port reads states ({}); the tree carries \
                     `any`, `all`, `sum` and `count` aggregations",
                    sources
                        .iter()
                        .map(|&s| self.model.var_names[s].as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )))
            }
        })
    }

    // --- negation normal form -------------------------------------------

    /// `expr` (unrolled, in a boolean position), negated when `negate`,
    /// with every negation pushed down to the states and every vote
    /// written `sum(if(c, 1, 0)) >= k` over positive conditions. A label
    /// stays on what it labels, `negated` flipped when the negation
    /// crosses it.
    fn nnf(&mut self, expr: &TExpr, negate: bool) -> Result<TExpr, FaultTreeError> {
        if !expr.reads_state() {
            return Ok(TExpr::constant(self.frozen_bool_t(expr)? != negate));
        }
        match expr {
            TExpr::Labelled { origin, inner } => {
                let inner = self.nnf(inner, negate)?;
                if let TExpr::Plain(CExpr::Const(_)) = inner {
                    return Ok(inner);
                }
                Ok(TExpr::Labelled {
                    origin: GateOrigin {
                        negated: origin.negated != negate,
                        ..origin.clone()
                    },
                    inner: Box::new(inner),
                })
            }
            TExpr::Plain(CExpr::StateActive { automaton, state }) => {
                if !negate {
                    return Ok(expr.clone());
                }
                // Not in `state`: in one of the automaton's other states.
                let others: Vec<TExpr> = (0..self.model.automata[*automaton].states.len())
                    .filter(|s| s != state)
                    .map(|s| {
                        TExpr::Plain(CExpr::StateActive {
                            automaton: *automaton,
                            state: s,
                        })
                    })
                    .collect();
                Ok(TExpr::Bool {
                    op: BoolOp::Or,
                    args: others,
                })
            }
            TExpr::Bool { op, args } => match op {
                BoolOp::Not => match args.as_slice() {
                    [arg] => self.nnf(arg, !negate),
                    _ => Err(FaultTreeError::Unsupported(format!(
                        "a negation of {} arguments",
                        args.len()
                    ))),
                },
                BoolOp::And | BoolOp::Or => {
                    let is_and = (*op == BoolOp::And) != negate;
                    let args = args
                        .iter()
                        .map(|a| self.nnf(a, negate))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(fold_connective(is_and, args))
                }
            },
            TExpr::If {
                cond,
                then,
                otherwise,
            } => {
                // In a boolean position: (c and a) or (not c and b).
                let rewritten = TExpr::Bool {
                    op: BoolOp::Or,
                    args: vec![
                        TExpr::Bool {
                            op: BoolOp::And,
                            args: vec![(**cond).clone(), (**then).clone()],
                        },
                        TExpr::Bool {
                            op: BoolOp::And,
                            args: vec![
                                TExpr::Bool {
                                    op: BoolOp::Not,
                                    args: vec![(**cond).clone()],
                                },
                                (**otherwise).clone(),
                            ],
                        },
                    ],
                };
                self.nnf(&rewritten, negate)
            }
            TExpr::Cmp { op, lhs, rhs } => self.nnf_cmp(*op, lhs, rhs, negate, expr),
            _ => Err(FaultTreeError::Unsupported(format!(
                "a state is read inside an expression this module does not \
                 explain ({}); a vote is written `sum(if(state, 1, 0)) >= k`",
                describe(self.model, &expr.plain())
            ))),
        }
    }

    fn nnf_cmp(
        &mut self,
        op: CmpOp,
        lhs: &TExpr,
        rhs: &TExpr,
        negate: bool,
        whole: &TExpr,
    ) -> Result<TExpr, FaultTreeError> {
        // A boolean compared with a boolean constant is that boolean or
        // its negation.
        for (dynamic, constant) in [(lhs, rhs), (rhs, lhs)] {
            if dynamic.reads_state() && !constant.reads_state() {
                if let Value::Bool(b) = self.frozen_t(constant)? {
                    let positive = match op {
                        CmpOp::Eq => b,
                        CmpOp::Ne => !b,
                        _ => {
                            return Err(FaultTreeError::Unsupported(format!(
                                "a boolean is ordered against a constant ({})",
                                describe(self.model, &whole.plain())
                            )))
                        }
                    };
                    return self.nnf(dynamic, negate == positive);
                }
            }
        }
        if lhs.reads_state() && rhs.reads_state() {
            return Err(FaultTreeError::NonCoherent(format!(
                "two quantities that both read states are compared ({})",
                describe(self.model, &whole.plain())
            )));
        }
        // A count of conditions against a threshold: a vote.
        let (op, counted, threshold) = if lhs.reads_state() {
            (op, lhs, rhs)
        } else {
            (mirror(op), rhs, lhs)
        };
        self.nnf_vote(op, counted, threshold, negate, whole)
    }

    /// `sum_i if(c_i, 1, 0) + constant  op  threshold`, negated when
    /// `negate`, as `sum_j if(d_j, 1, 0) >= k` over positive conditions.
    fn nnf_vote(
        &mut self,
        op: CmpOp,
        counted: &TExpr,
        threshold: &TExpr,
        negate: bool,
        whole: &TExpr,
    ) -> Result<TExpr, FaultTreeError> {
        let refuse = |model: &CompiledModel| {
            FaultTreeError::Unsupported(format!(
                "a state is read inside a comparison that is not a vote ({}); a \
                 vote is written `sum(if(state, 1, 0)) >= k`",
                describe(model, &whole.plain())
            ))
        };
        let threshold = match self.frozen_t(threshold)? {
            Value::Int(i) => i as f64,
            Value::Float(f) => f,
            Value::Bool(_) => return Err(refuse(self.model)),
        };
        let mut terms = Vec::new();
        flatten_add(counted, &mut terms);
        let mut conditions: Vec<(&TExpr, bool)> = Vec::new();
        let mut constant = 0.0;
        for term in terms {
            // A term is a number: an attribute's label says nothing there.
            match term.peeled() {
                other if !other.reads_state() => constant += number(self.frozen_t(other)?),
                TExpr::If {
                    cond,
                    then,
                    otherwise,
                } if !then.reads_state() && !otherwise.reads_state() => {
                    match (
                        number(self.frozen_t(then)?),
                        number(self.frozen_t(otherwise)?),
                    ) {
                        (one, zero) if one == 1.0 && zero == 0.0 => conditions.push((cond, false)),
                        (zero, one) if one == 1.0 && zero == 0.0 => conditions.push((cond, true)),
                        _ => return Err(refuse(self.model)),
                    }
                }
                _ => return Err(refuse(self.model)),
            }
        }
        let op = if negate { negated(op) } else { op };
        let n = conditions.len() as f64;
        let room = threshold - constant;
        // At least `k` of the conditions, or at least `k` of their
        // complements.
        let (k, complement) = match op {
            CmpOp::Ge => (room.ceil(), false),
            CmpOp::Gt => (room.floor() + 1.0, false),
            CmpOp::Le => (n - room.floor(), true),
            CmpOp::Lt => (n - room.ceil() + 1.0, true),
            CmpOp::Eq | CmpOp::Ne => {
                return Err(FaultTreeError::NonCoherent(format!(
                    "a count of states is tested for {} ({})",
                    if op == CmpOp::Eq {
                        "equality"
                    } else {
                        "inequality"
                    },
                    describe(self.model, &whole.plain())
                )))
            }
        };
        if k <= 0.0 {
            return Ok(TExpr::constant(true));
        }
        if k > n {
            return Ok(TExpr::constant(false));
        }
        let args = conditions
            .into_iter()
            .map(|(cond, negated_term)| {
                Ok(TExpr::If {
                    cond: Box::new(self.nnf(cond, negated_term != complement)?),
                    then: Box::new(TExpr::int(1)),
                    otherwise: Box::new(TExpr::int(0)),
                })
            })
            .collect::<Result<Vec<_>, FaultTreeError>>()?;
        Ok(TExpr::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(TExpr::Add { args }),
            rhs: Box::new(TExpr::int(k as i64)),
        })
    }

    /// Refuse an expression in negation normal form that needs a state
    /// the model can leave to persist: an initial state read positively.
    /// The chaining would count it as reached from the start, which says
    /// nothing about it still holding at the mission time.
    fn refuse_persistence(&self, expr: &TExpr, what: &str) -> Result<(), FaultTreeError> {
        let mut found = None;
        expr.visit_states(&mut |automaton, state| {
            if found.is_none() && self.leavable_initial(automaton, state) {
                found = Some(self.state_name(automaton, state));
            }
        });
        match found {
            Some(name) => Err(FaultTreeError::NonCoherent(format!(
                "{what} needs `{name}` to hold, the initial state of an automaton \
                 that can leave it; a static tree explains states being entered, \
                 so it would count that one as holding throughout (a standby, a \
                 trigger or a reconfiguration depends on the order of events, \
                 which only the sequences of a simulation carry)"
            ))),
            None => Ok(()),
        }
    }

    fn leavable_initial(&self, automaton: AutIdx, state: StateIdx) -> bool {
        let aut = &self.model.automata[automaton];
        aut.init == state
            && aut.transitions.iter().any(|&t| {
                let transition = &self.model.transitions[t];
                transition.source == state && transition.targets.iter().any(|&s| s != state)
            })
    }

    /// Unroll and normalise a condition the tree crosses.
    fn condition(&mut self, expr: &CExpr) -> Result<(TExpr, bool), FaultTreeError> {
        let unrolled = self.unroll(expr)?;
        let reads = unrolled.reads_state();
        Ok((self.nnf(&unrolled, false)?, reads))
    }

    // --- backward chaining ----------------------------------------------

    /// Expand an expression in negation normal form. A label becomes a
    /// gate carrying its origin over what it labels; simplification then
    /// decides whether that gate stays.
    fn expand(&mut self, expr: &TExpr) -> Result<FtNode, FaultTreeError> {
        if let TExpr::Labelled { origin, inner } = expr {
            let child = self.expand(inner)?;
            return Ok(gate(GateOp::Or, Some(origin.clone()), vec![child]));
        }
        self.count()?;
        if !expr.reads_state() {
            return Ok(FtNode::Constant {
                value: self.frozen_bool_t(expr)?,
            });
        }
        match expr {
            TExpr::Plain(CExpr::StateActive { automaton, state }) => {
                self.expand_state(*automaton, *state)
            }
            TExpr::Bool { op, args } => match op {
                BoolOp::And | BoolOp::Or => {
                    let children = args
                        .iter()
                        .map(|arg| self.expand(arg))
                        .collect::<Result<Vec<_>, _>>()?;
                    let op = if *op == BoolOp::And {
                        GateOp::And
                    } else {
                        GateOp::Or
                    };
                    Ok(gate(op, None, children))
                }
                BoolOp::Not => Err(FaultTreeError::NonCoherent(format!(
                    "a state is read under a negation ({})",
                    self.describe(&expr.plain())
                ))),
            },
            TExpr::Cmp { op, lhs, rhs } => self.expand_vote(*op, lhs, rhs, expr),
            _ => Err(FaultTreeError::Unsupported(format!(
                "a state is read inside an expression this module does not \
                 explain ({}); a vote is written `sum(if(state, 1, 0)) >= k`",
                self.describe(&expr.plain())
            ))),
        }
    }

    /// `sum_i if(c_i, 1, 0) >= k` (or `> k - 1`): at least k of the c_i.
    fn expand_vote(
        &mut self,
        op: CmpOp,
        lhs: &TExpr,
        rhs: &TExpr,
        whole: &TExpr,
    ) -> Result<FtNode, FaultTreeError> {
        let refuse = |this: &Self| {
            FaultTreeError::Unsupported(format!(
                "a state is read inside a comparison that is not a vote ({}); a \
                 vote is written `sum(if(state, 1, 0)) >= k`",
                this.describe(&whole.plain())
            ))
        };
        if rhs.reads_state() {
            return Err(refuse(self));
        }
        let threshold = match self.frozen_t(rhs)? {
            Value::Int(i) => i as f64,
            Value::Float(f) => f,
            Value::Bool(_) => return Err(refuse(self)),
        };
        let needed = match op {
            CmpOp::Ge => threshold.ceil(),
            CmpOp::Gt => threshold.floor() + 1.0,
            _ => return Err(refuse(self)),
        };
        let terms: Vec<&TExpr> = match lhs {
            TExpr::Add { args } => args.iter().collect(),
            other => vec![other],
        };
        let mut children = Vec::new();
        let mut constant_votes = 0.0;
        for term in terms {
            match term {
                TExpr::If {
                    cond,
                    then,
                    otherwise,
                } if !then.reads_state() && !otherwise.reads_state() => {
                    let one = number(self.frozen_t(then)?);
                    let zero = number(self.frozen_t(otherwise)?);
                    if one != 1.0 || zero != 0.0 {
                        return Err(refuse(self));
                    }
                    if cond.reads_state() {
                        children.push(self.expand(cond)?);
                    } else if self.frozen_bool_t(cond)? {
                        constant_votes += 1.0;
                    }
                }
                other if !other.reads_state() => constant_votes += number(self.frozen_t(other)?),
                _ => return Err(refuse(self)),
            }
        }
        let k = needed - constant_votes;
        if k <= 0.0 {
            return Ok(FtNode::Constant { value: true });
        }
        if k > children.len() as f64 {
            return Ok(FtNode::Constant { value: false });
        }
        Ok(gate(GateOp::AtLeast { k: k as usize }, None, children))
    }

    fn expand_state(
        &mut self,
        automaton: AutIdx,
        state: StateIdx,
    ) -> Result<FtNode, FaultTreeError> {
        self.count()?;
        if self.stack.contains(&(automaton, state)) {
            // Already being explained further up this branch: a path that
            // loops back to it adds no new way in.
            return Ok(FtNode::Constant { value: false });
        }
        let mut ways = Vec::new();
        if self.model.automata[automaton].init == state {
            ways.push(FtNode::Constant { value: true });
        }
        self.stack.push((automaton, state));
        let transitions = self.model.automata[automaton].transitions.clone();
        for t_idx in transitions {
            let transition = &self.model.transitions[t_idx];
            let Some(position) = transition.targets.iter().position(|&s| s == state) else {
                continue;
            };
            if transition.source == state {
                continue;
            }
            let source = transition.source;
            let guard = transition.guard.clone();
            let observer = transition.kind == Some(TransitionKind::Observation);
            if observer && !matches!(transition.distrib, CLaw::Delay(d) if d == 0.0) {
                return Err(FaultTreeError::Unsupported(format!(
                    "the observer transition `{}` waits before it fires, so the \
                     state it enters lags its condition, which a tree at one \
                     mission time cannot carry",
                    transition.name
                )));
            }
            // The source and the guard first: a way in that cannot hold
            // registers no basic event, so the tree lists only the events
            // some explanation uses.
            let from = self.expand_state(automaton, source)?;
            if from == (FtNode::Constant { value: false }) {
                continue;
            }
            let mut conjuncts = vec![from];
            let mut guard_reads_state = false;
            if let Some(guard) = guard {
                let condition =
                    self.explain_guard(t_idx, &guard, observer)
                        .map_err(|e| match e {
                            FaultTreeError::NonCoherent(detail)
                            | FaultTreeError::Unsupported(detail) => {
                                FaultTreeError::Unsupported(format!(
                                    "the guard of transition `{}` into `{}`: {detail}",
                                    self.model.transitions[t_idx].name,
                                    self.state_name(automaton, state)
                                ))
                            }
                            other => other,
                        })?;
                guard_reads_state = condition.1;
                let condition = condition.0;
                if condition == (FtNode::Constant { value: false }) {
                    continue;
                }
                conjuncts.push(condition);
            }
            let event = self.event(t_idx, position)?;
            if event == (FtNode::Constant { value: false }) {
                continue;
            }
            if !observer {
                self.note_approximations(t_idx, position, guard_reads_state);
            }
            conjuncts.insert(0, event);
            ways.push(gate(
                GateOp::And,
                Some(self.transition_origin(t_idx, position)),
                conjuncts,
            ));
        }
        self.stack.pop();
        Ok(gate(
            GateOp::Or,
            Some(self.state_origin(automaton, state)),
            ways,
        ))
    }

    /// The guard of transition `t_idx`, unrolled and expanded, and whether
    /// it reads a state. An observer's guard is the feared event's own
    /// condition, held to the standard of a top.
    fn explain_guard(
        &mut self,
        t_idx: usize,
        guard: &CExpr,
        observer: bool,
    ) -> Result<(FtNode, bool), FaultTreeError> {
        let (normal, reads) = self.condition(guard)?;
        let value = normal.is_constant(true);
        if !reads && (observer || !value) {
            self.constant_conditions.push(format!(
                "the guard of transition `{}` reads no state once its attributes \
                 are unrolled, it is the constant {value}",
                self.model.transitions[t_idx].name
            ));
        }
        if observer {
            self.refuse_persistence(
                &normal,
                &format!(
                    "the condition of the observer `{}`",
                    self.model.transitions[t_idx].name
                ),
            )?;
        }
        Ok((self.expand(&normal)?, reads))
    }

    /// Whether transition `t_idx` can ever fire: a law that draws, and a
    /// guard that is not constantly false once unrolled. A guard the tree
    /// cannot unroll counts as one that can hold.
    fn can_fire(&mut self, t_idx: usize) -> bool {
        let transition = &self.model.transitions[t_idx];
        let draws = match &transition.distrib {
            CLaw::Exp(rate) => *rate > 0.0,
            CLaw::Inst(probabilities) => probabilities.iter().any(|&p| p > 0.0),
            _ => true,
        };
        if !draws {
            return false;
        }
        match transition.guard.clone() {
            None => true,
            Some(guard) => !matches!(
                self.condition(&guard),
                Ok((normal, _)) if normal.is_constant(false)
            ),
        }
    }

    /// Record why the draw of transition `t_idx` into its `position`-th
    /// target may make the tree exceed the model's probability.
    fn note_approximations(&mut self, t_idx: usize, position: usize, guard_reads_state: bool) {
        let transition = &self.model.transitions[t_idx];
        let automaton = &self.model.automata[transition.automaton];
        let name = transition.name.clone();
        let source = automaton.states[transition.source].clone();
        let target_state = transition.targets[position];
        let target = automaton.states[target_state].clone();
        let mut notes = Vec::new();
        if guard_reads_state {
            notes.push(GenerationWarning::GuardReadsState {
                transition: name.clone(),
            });
        }
        if automaton.init != transition.source {
            notes.push(GenerationWarning::EnteredDuringMission {
                transition: name.clone(),
                source: source.clone(),
            });
        }
        let others: Vec<usize> = automaton
            .transitions
            .iter()
            .copied()
            .filter(|&t| t != t_idx)
            .collect();
        let source_state = transition.source;
        let automaton_name = automaton.name.clone();
        let mut competitors = Vec::new();
        let mut leaving = Vec::new();
        for t in others {
            let other = &self.model.transitions[t];
            let moves =
                |from: StateIdx| other.source == from && other.targets.iter().any(|&s| s != from);
            let competes = moves(source_state);
            // A way back to the event's source is a repair, carried by the
            // event; only a way elsewhere is warned here.
            let leaves = other.source == target_state
                && other
                    .targets
                    .iter()
                    .any(|&s| s != target_state && s != source_state);
            if (competes || leaves) && self.can_fire(t) {
                let other = self.model.transitions[t].name.clone();
                if competes {
                    competitors.push(other.clone());
                }
                if leaves {
                    leaving.push(other);
                }
            }
        }
        if !competitors.is_empty() {
            notes.push(GenerationWarning::CompetingTransition {
                transition: name,
                source,
                competitors,
            });
        }
        if !leaving.is_empty() {
            notes.push(GenerationWarning::TargetLeft {
                automaton: automaton_name,
                state: target,
                transitions: leaving,
            });
        }
        for note in notes {
            self.warn(note);
        }
    }

    /// The law of transition `t_idx` into its `position`-th target, the one
    /// correspondence between the engine's laws and a tree's: a law that
    /// draws, or the constant a transition no draw governs stands for.
    fn drawn_law(&mut self, t_idx: usize, position: usize) -> Result<Drawn, FaultTreeError> {
        let transition = &self.model.transitions[t_idx];
        let law = match &transition.distrib {
            CLaw::Exp(rate) => BasicLaw::Exponential { rate: *rate },
            CLaw::ExpVar { rate, .. } => {
                let rate = self.unroll(rate)?.plain();
                let transition = &self.model.transitions[t_idx];
                if reads_state(&rate) {
                    return Err(FaultTreeError::Unsupported(format!(
                        "the rate of transition `{}` reads a state, so no single \
                         law stands for its draw",
                        transition.name
                    )));
                }
                BasicLaw::Exponential {
                    rate: number(self.frozen(&rate)?),
                }
            }
            CLaw::Weibull(shape, scale) => BasicLaw::Weibull {
                shape: *shape,
                scale: *scale,
            },
            CLaw::Lognormal(mu, sigma) => BasicLaw::Lognormal {
                mu: *mu,
                sigma: *sigma,
            },
            CLaw::Gamma(shape, scale) => BasicLaw::Gamma {
                shape: *shape,
                scale: *scale,
            },
            CLaw::Uniform(low, high) => BasicLaw::Uniform {
                low: *low,
                high: *high,
            },
            // No wait: the transition fires as soon as its source and its
            // guard hold, so it is crossed rather than counted.
            CLaw::Delay(time) if *time == 0.0 => return Ok(Drawn::Constant(true)),
            CLaw::Delay(time) => BasicLaw::Delay { time: *time },
            CLaw::Empirical(points) => BasicLaw::Empirical {
                points: points.clone(),
            },
            CLaw::Inst(probabilities) => {
                let probability = probabilities.get(position).copied().unwrap_or(0.0);
                if probability <= 0.0 {
                    return Ok(Drawn::Constant(false));
                }
                if probability >= 1.0 {
                    return Ok(Drawn::Constant(true));
                }
                BasicLaw::Probability { probability }
            }
            CLaw::Watched { margin } => {
                // A crossing no draw governs: with the states it reads at
                // their initial values, it happens or it never does.
                let margin = self.unroll(margin)?.plain();
                let transition = &self.model.transitions[t_idx];
                if reads_state(&margin) {
                    return Err(FaultTreeError::Unsupported(format!(
                        "the boundary of watched transition `{}` reads a state",
                        transition.name
                    )));
                }
                let crosses = number(self.frozen(&margin)?) >= 0.0;
                return Ok(Drawn::Constant(crosses));
            }
        };
        Ok(Drawn::Law(law))
    }

    /// The basic event of transition `t_idx` into its `position`-th target,
    /// or a constant for a transition no draw governs.
    fn event(&mut self, t_idx: usize, position: usize) -> Result<FtNode, FaultTreeError> {
        let law = match self.drawn_law(t_idx, position)? {
            Drawn::Law(law) => law,
            Drawn::Constant(value) => return Ok(FtNode::Constant { value }),
        };
        let key = (t_idx, position);
        if let Some(&index) = self.event_index.get(&key) {
            return Ok(FtNode::Basic { event: index });
        }
        let transition = &self.model.transitions[t_idx];
        let automaton = &self.model.automata[transition.automaton];
        let target_state = transition.targets[position];
        let target = automaton.states[target_state].clone();
        // The compiled name is already qualified: `component.automaton.name`.
        let mut name = transition.name.clone();
        if transition.targets.len() > 1 {
            name = format!("{name}.{target}");
        }
        let repairs = self.repairs(t_idx, position);
        let transition = &self.model.transitions[t_idx];
        let automaton = &self.model.automata[transition.automaton];
        let index = self.events.len();
        self.events.push(TreeEvent {
            name,
            component: transition.component.clone(),
            automaton: automaton.name.clone(),
            transition: transition
                .name
                .rsplit('.')
                .next()
                .unwrap_or(&transition.name)
                .to_owned(),
            target,
            law,
            repairs,
        });
        self.event_index.insert(key, index);
        Ok(FtNode::Basic { event: index })
    }

    /// The repairs of transition `t_idx`'s draw into its `position`-th
    /// target: the transitions that can fire from that target back to the
    /// transition's source, each with its law read as [`Self::drawn_law`]
    /// reads an event's, and whether a guard conditions it.
    fn repairs(&mut self, t_idx: usize, position: usize) -> Vec<TreeRepair> {
        let transition = &self.model.transitions[t_idx];
        let (source, target) = (transition.source, transition.targets[position]);
        let candidates: Vec<(usize, usize)> = self.model.automata[transition.automaton]
            .transitions
            .iter()
            .copied()
            .filter(|&t| t != t_idx && self.model.transitions[t].source == target)
            .filter_map(|t| {
                self.model.transitions[t]
                    .targets
                    .iter()
                    .position(|&s| s == source)
                    .map(|back| (t, back))
            })
            .collect();
        let mut repairs = Vec::new();
        for (t, back) in candidates {
            if !self.can_fire(t) {
                continue;
            }
            let law = match self.drawn_law(t, back) {
                Ok(Drawn::Law(law)) => Some(law),
                // Fires as soon as it can: an immediate repair.
                Ok(Drawn::Constant(true)) => Some(BasicLaw::Delay { time: 0.0 }),
                Ok(Drawn::Constant(false)) => continue,
                // No single law stands for it: still a repair, whose law
                // the measure cannot read.
                Err(_) => None,
            };
            let guarded = match self.model.transitions[t].guard.clone() {
                None => false,
                Some(guard) => !matches!(
                    self.condition(&guard),
                    Ok((normal, _)) if normal.is_constant(true)
                ),
            };
            repairs.push(TreeRepair {
                transition: self.model.transitions[t].name.clone(),
                law,
                guarded,
            });
        }
        repairs
    }

    /// Why the top came out constant, in the model's terms.
    fn degenerate_cause(&self, top: &CExpr, unrolled_top: &TExpr, value: bool) -> String {
        if !unrolled_top.reads_state() {
            let mut read = BTreeSet::new();
            collect_attributes(top, &mut read);
            let names: Vec<&str> = read
                .iter()
                .map(|&v| self.model.var_names[v].as_str())
                .collect();
            return if names.is_empty() {
                "it reads no state".to_owned()
            } else {
                format!(
                    "it reads no state once the attributes it reads ({}) are \
                     unrolled: nothing in the model computes them from a state, \
                     so they keep the value they start with",
                    names.join(", ")
                )
            };
        }
        let mut cause = if value {
            "it holds from the initial state, so no failure is needed to reach it".to_owned()
        } else {
            "no transition the model can fire leads to it".to_owned()
        };
        if !self.constant_conditions.is_empty() {
            let shown: Vec<&str> = self
                .constant_conditions
                .iter()
                .take(5)
                .map(String::as_str)
                .collect();
            cause.push_str("; ");
            cause.push_str(&shown.join("; "));
            if self.constant_conditions.len() > shown.len() {
                cause.push_str(&format!(
                    "; and {} more",
                    self.constant_conditions.len() - shown.len()
                ));
            }
            cause.push_str(
                " (an attribute nothing in the model computes from a state keeps \
                 the value it starts with)",
            );
        }
        cause
    }

    fn describe(&self, expr: &CExpr) -> String {
        describe(self.model, expr)
    }
}

/// A conjunction (`is_and`) or a disjunction of `args`, its constant
/// arguments folded: a state read beside an absorbing constant is not
/// read at all, which matters to the persistence check (`if(c, false,
/// true)` negated must not leave `c`'s complement under a `true`).
///
/// Nested connectives are not merged here, labelled or not: that is
/// [`simplify`]'s, on the tree, which keeps a labelled gate whole.
fn fold_connective(is_and: bool, args: Vec<TExpr>) -> TExpr {
    let mut kept = Vec::with_capacity(args.len());
    for arg in args {
        match arg {
            TExpr::Plain(CExpr::Const(Value::Bool(b))) if b == is_and => {}
            TExpr::Plain(CExpr::Const(Value::Bool(b))) => return TExpr::constant(b),
            other => kept.push(other),
        }
    }
    match kept.len() {
        0 => TExpr::constant(is_and),
        1 => kept.remove(0),
        _ => TExpr::Bool {
            op: if is_and { BoolOp::And } else { BoolOp::Or },
            args: kept,
        },
    }
}

/// Swap the operands of a comparison.
fn mirror(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
        other => other,
    }
}

/// The comparison that holds exactly when `op` does not.
fn negated(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Lt => CmpOp::Ge,
        CmpOp::Le => CmpOp::Gt,
        CmpOp::Gt => CmpOp::Le,
        CmpOp::Ge => CmpOp::Lt,
        CmpOp::Eq => CmpOp::Ne,
        CmpOp::Ne => CmpOp::Eq,
    }
}

fn flatten_add<'e>(expr: &'e TExpr, out: &mut Vec<&'e TExpr>) {
    match expr {
        TExpr::Add { args } => args.iter().for_each(|a| flatten_add(a, out)),
        other => out.push(other),
    }
}

/// `qualified` (`component.rest`) without its component prefix.
fn local<'n>(qualified: &'n str, component: &str) -> &'n str {
    qualified
        .strip_prefix(component)
        .and_then(|rest| rest.strip_prefix('.'))
        .unwrap_or(qualified)
}

/// Every state an expression reads.
fn visit_states(expr: &CExpr, visit: &mut impl FnMut(AutIdx, StateIdx)) {
    match expr {
        CExpr::StateActive { automaton, state } => visit(*automaton, *state),
        CExpr::Const(_) | CExpr::Var(_) | CExpr::PortAgg { .. } | CExpr::Time => {}
        CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
            visit_states(lhs, visit);
            visit_states(rhs, visit);
        }
        CExpr::Bool { args, .. }
        | CExpr::Add { args }
        | CExpr::Mul { args }
        | CExpr::Min { args }
        | CExpr::Max { args } => args.iter().for_each(|a| visit_states(a, visit)),
        CExpr::If {
            cond,
            then,
            otherwise,
        } => {
            visit_states(cond, visit);
            visit_states(then, visit);
            visit_states(otherwise, visit);
        }
        CExpr::Sin(arg) | CExpr::Exp(arg) => visit_states(arg, visit),
    }
}

/// Every attribute an expression reads, directly or through a port.
fn collect_attributes(expr: &CExpr, out: &mut BTreeSet<VarIdx>) {
    match expr {
        CExpr::Var(index) => {
            out.insert(*index);
        }
        CExpr::PortAgg { sources, .. } => out.extend(sources.iter().copied()),
        CExpr::Const(_) | CExpr::StateActive { .. } | CExpr::Time => {}
        CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
            collect_attributes(lhs, out);
            collect_attributes(rhs, out);
        }
        CExpr::Bool { args, .. }
        | CExpr::Add { args }
        | CExpr::Mul { args }
        | CExpr::Min { args }
        | CExpr::Max { args } => args.iter().for_each(|a| collect_attributes(a, out)),
        CExpr::If {
            cond,
            then,
            otherwise,
        } => {
            collect_attributes(cond, out);
            collect_attributes(then, out);
            collect_attributes(otherwise, out);
        }
        CExpr::Sin(arg) | CExpr::Exp(arg) => collect_attributes(arg, out),
    }
}

/// Nodes of an expression.
fn size_of(expr: &CExpr) -> usize {
    1 + match expr {
        CExpr::Const(_)
        | CExpr::Var(_)
        | CExpr::PortAgg { .. }
        | CExpr::Time
        | CExpr::StateActive { .. } => 0,
        CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
            size_of(lhs) + size_of(rhs)
        }
        CExpr::Bool { args, .. }
        | CExpr::Add { args }
        | CExpr::Mul { args }
        | CExpr::Min { args }
        | CExpr::Max { args } => args.iter().map(size_of).sum(),
        CExpr::If {
            cond,
            then,
            otherwise,
        } => size_of(cond) + size_of(then) + size_of(otherwise),
        CExpr::Sin(arg) | CExpr::Exp(arg) => size_of(arg),
    }
}

/// A value read as a number: a boolean counts one or zero.
fn number(value: Value) -> f64 {
    match value {
        Value::Bool(b) => f64::from(u8::from(b)),
        Value::Int(i) => i as f64,
        Value::Float(f) => f,
    }
}

/// Whether `expr` reads an automaton state anywhere.
fn reads_state(expr: &CExpr) -> bool {
    match expr {
        CExpr::StateActive { .. } => true,
        CExpr::Const(_) | CExpr::Var(_) | CExpr::PortAgg { .. } | CExpr::Time => false,
        CExpr::Cmp { lhs, rhs, .. } | CExpr::Sub { lhs, rhs } | CExpr::Div { lhs, rhs } => {
            reads_state(lhs) || reads_state(rhs)
        }
        CExpr::Bool { args, .. }
        | CExpr::Add { args }
        | CExpr::Mul { args }
        | CExpr::Min { args }
        | CExpr::Max { args } => args.iter().any(reads_state),
        CExpr::If {
            cond,
            then,
            otherwise,
        } => reads_state(cond) || reads_state(then) || reads_state(otherwise),
        CExpr::Sin(arg) | CExpr::Exp(arg) => reads_state(arg),
    }
}

fn describe(model: &CompiledModel, expr: &CExpr) -> String {
    match expr {
        CExpr::StateActive { automaton, state } => {
            let aut = &model.automata[*automaton];
            format!("{}.{}", aut.name, aut.states[*state])
        }
        CExpr::Var(index) => model.var_names[*index].clone(),
        CExpr::Const(value) => format!("{value:?}"),
        CExpr::Bool { op, args } => format!(
            "{op:?}({})",
            args.iter()
                .map(|a| describe(model, a))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        CExpr::Cmp { op, lhs, rhs } => {
            format!("{op:?}({}, {})", describe(model, lhs), describe(model, rhs))
        }
        other => format!("{other:?}").chars().take(120).collect(),
    }
}

/// Compile a top expression against the model: attributes, states and the
/// operators a tree can explain. Port aggregations and time are refused, a
/// top expression being a statement about the model's states.
fn compile_top(model: &CompiledModel, expr: &Expr) -> Result<CExpr, FaultTreeError> {
    let args = |list: &[Expr]| -> Result<Vec<CExpr>, FaultTreeError> {
        list.iter().map(|e| compile_top(model, e)).collect()
    };
    Ok(match expr {
        Expr::Const { value } => CExpr::Const(*value),
        Expr::Attr { attr } => {
            let name = format!("{}.{}", attr.component, attr.attribute);
            CExpr::Var(
                *model
                    .var_index
                    .get(&name)
                    .ok_or(FaultTreeError::Unresolved(name))?,
            )
        }
        Expr::StateActive { state } => {
            let qualified = format!("{}.{}", state.component, state.automaton);
            let automaton = *model
                .automaton_index
                .get(&qualified)
                .ok_or_else(|| FaultTreeError::Unresolved(qualified.clone()))?;
            let index = model.automata[automaton]
                .states
                .iter()
                .position(|s| *s == state.state)
                .ok_or_else(|| {
                    FaultTreeError::Unresolved(format!("{qualified}.{}", state.state))
                })?;
            CExpr::StateActive {
                automaton,
                state: index,
            }
        }
        Expr::Cmp { cmp, lhs, rhs } => CExpr::Cmp {
            op: *cmp,
            lhs: Box::new(compile_top(model, lhs)?),
            rhs: Box::new(compile_top(model, rhs)?),
        },
        Expr::Bool {
            bool_op,
            args: list,
        } => CExpr::Bool {
            op: *bool_op,
            args: args(list)?,
        },
        Expr::Add { args: list } => CExpr::Add { args: args(list)? },
        Expr::If {
            cond,
            then,
            otherwise,
        } => CExpr::If {
            cond: Box::new(compile_top(model, cond)?),
            then: Box::new(compile_top(model, then)?),
            otherwise: Box::new(compile_top(model, otherwise)?),
        },
        _ => {
            return Err(FaultTreeError::Unsupported(
                "a top expression is written over attributes and states with \
                 `const`, `attr`, `state_active`, `cmp`, `bool`, `add` and `if`"
                    .to_owned(),
            ))
        }
    })
}

/// Propagate constants, collapse single-child gates and merge nested gates
/// of one connective (the reference options `reduce_constant_gt`,
/// `reduce_one_son_gt`, `reduce_same_op_gt`, always applied).
///
/// A gate with an origin is never merged into its parent: the Boolean
/// function is the same, and the gate says what it stands for. A gate with
/// one child still reduces to it: a child gate takes the origin when it has
/// none of its own (the nearer origin wins), a basic event drops it, its
/// name already saying which transition it is. Without this, every event
/// of a two-state mode would sit under a state gate and a transition gate
/// of one child each.
fn simplify(node: FtNode) -> FtNode {
    let FtNode::Gate {
        op,
        id,
        origin,
        children,
    } = node
    else {
        return node;
    };
    let mut kept = Vec::new();
    let mut trues = 0usize;
    let mut falses = 0usize;
    for child in children.into_iter().map(simplify) {
        match child {
            FtNode::Constant { value: true } => trues += 1,
            FtNode::Constant { value: false } => falses += 1,
            FtNode::Gate {
                op: inner,
                origin: None,
                children: grandchildren,
                ..
            } if inner == op && matches!(op, GateOp::And | GateOp::Or) => {
                kept.extend(grandchildren)
            }
            other => kept.push(other),
        }
    }
    let gate = |op: GateOp, mut kept: Vec<FtNode>| -> FtNode {
        if kept.len() == 1 {
            match kept.remove(0) {
                FtNode::Gate {
                    op,
                    id: child_id,
                    origin: child_origin,
                    children,
                } => FtNode::Gate {
                    op,
                    id: child_id.or(id),
                    origin: child_origin.or(origin),
                    children,
                },
                other => other,
            }
        } else {
            FtNode::Gate {
                op,
                id,
                origin,
                children: kept,
            }
        }
    };
    match op {
        GateOp::And if falses > 0 => FtNode::Constant { value: false },
        GateOp::And if kept.is_empty() => FtNode::Constant { value: true },
        GateOp::And => gate(GateOp::And, kept),
        GateOp::Or if trues > 0 => FtNode::Constant { value: true },
        GateOp::Or if kept.is_empty() => FtNode::Constant { value: false },
        GateOp::Or => gate(GateOp::Or, kept),
        GateOp::AtLeast { k } => {
            if trues >= k {
                return FtNode::Constant { value: true };
            }
            let k = k - trues;
            if k > kept.len() {
                FtNode::Constant { value: false }
            } else if k == kept.len() {
                gate(GateOp::And, kept)
            } else if k == 1 {
                gate(GateOp::Or, kept)
            } else {
                gate(GateOp::AtLeast { k }, kept)
            }
        }
    }
}

/// The occurrence identifier of the top gate.
const TOP_GATE_ID: &str = "top";

/// Give every gate of a simplified tree its occurrence identifier: `top`
/// for the root, `G1`, `G2`, ... for the others in preorder. Generation is
/// deterministic, so one model gives one numbering, generation after
/// generation.
fn identify(top: &mut FtNode) {
    let mut next = 0usize;
    let mut pending = vec![top];
    while let Some(node) = pending.pop() {
        if let FtNode::Gate { id, children, .. } = node {
            *id = Some(if next == 0 {
                TOP_GATE_ID.to_owned()
            } else {
                format!("G{next}")
            });
            next += 1;
            pending.extend(children.iter_mut().rev());
        }
    }
}

impl FaultTree {
    /// The minimal cut sets, each a sorted list of basic-event indices,
    /// sorted by size then lexicographically. `limit` bounds how many
    /// intermediate sets an expansion may hold.
    pub fn minimal_cut_sets(&self, limit: usize) -> Result<Vec<Vec<usize>>, FaultTreeError> {
        let mut sets: Vec<Vec<usize>> = cut_sets(&self.top, limit)?
            .into_iter()
            .map(|set| set.into_iter().collect())
            .collect();
        sets.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
        Ok(sets)
    }

    /// The tree in the OpenPSA model-exchange format (the reference
    /// engine's fault-tree file format, user manual 7.6), as a document
    /// named `name`, to be quantified in `measure`. The measure is a
    /// `measure` attribute of the `define-fault-tree`, and each event's
    /// repairs a `repairs` attribute of its definition, so a reader
    /// quantifies the document as the tree it came from.
    ///
    /// The top is the gate named `top`, and every other gate is named by
    /// its occurrence identifier; a gate without one (a tree written before
    /// gates carried them) gets `G` and a number no identifier uses.
    pub fn to_open_psa(&self, name: &str, measure: FaultTreeMeasure) -> String {
        let mut names = GateNames::of(&self.top);
        let mut gates = Vec::new();
        let top = open_psa_formula(&self.top, &self.basic_events, &mut names, &mut gates);
        let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<opsa-mef>\n");
        out.push_str(&format!("  <define-fault-tree name=\"{}\">\n", xml(name)));
        out.push_str(&format!(
            "    <attributes>\n      <attribute name=\"measure\" value=\"{}\"/>\n    </attributes>\n",
            measure.as_str()
        ));
        out.push_str(&format!(
            "    <define-gate name=\"top\">\n      {top}\n    </define-gate>\n"
        ));
        for (gate, formula) in &gates {
            out.push_str(&format!(
                "    <define-gate name=\"{}\">\n      {formula}\n    </define-gate>\n",
                xml(gate)
            ));
        }
        out.push_str("  </define-fault-tree>\n  <model-data>\n");
        for event in &self.basic_events {
            out.push_str(&format!(
                "    <define-basic-event name=\"{}\">\n{}    </define-basic-event>\n",
                xml(&event.name),
                open_psa_event(&event.law, &event.repairs)
            ));
        }
        out.push_str("  </model-data>\n</opsa-mef>\n");
        out
    }
}

type CutSets = Vec<BTreeSet<usize>>;

fn minimise(mut sets: CutSets) -> CutSets {
    sets.sort_by_key(|s| s.len());
    let mut kept: CutSets = Vec::new();
    for set in sets {
        if !kept.iter().any(|k| k.is_subset(&set)) {
            kept.push(set);
        }
    }
    kept
}

fn cut_sets(node: &FtNode, limit: usize) -> Result<CutSets, FaultTreeError> {
    let too_many = || {
        FaultTreeError::TooLarge(format!(
            "the cut sets outgrew {limit} while being expanded; raise the limit"
        ))
    };
    Ok(match node {
        FtNode::Constant { value: true } => vec![BTreeSet::new()],
        FtNode::Constant { value: false } => Vec::new(),
        FtNode::Basic { event } => vec![BTreeSet::from([*event])],
        FtNode::Gate { op, children, .. } => {
            let parts = children
                .iter()
                .map(|c| cut_sets(c, limit))
                .collect::<Result<Vec<_>, _>>()?;
            match op {
                GateOp::Or => {
                    let all: CutSets = parts.into_iter().flatten().collect();
                    if all.len() > limit {
                        return Err(too_many());
                    }
                    minimise(all)
                }
                GateOp::And => {
                    let mut acc: CutSets = vec![BTreeSet::new()];
                    for part in parts {
                        let mut next = Vec::new();
                        for a in &acc {
                            for b in &part {
                                next.push(a.union(b).copied().collect());
                                if next.len() > limit {
                                    return Err(too_many());
                                }
                            }
                        }
                        acc = minimise(next);
                    }
                    acc
                }
                GateOp::AtLeast { k } => {
                    let mut all = Vec::new();
                    for combination in combinations(parts.len(), *k) {
                        let mut acc: CutSets = vec![BTreeSet::new()];
                        for index in combination {
                            let mut next = Vec::new();
                            for a in &acc {
                                for b in &parts[index] {
                                    next.push(a.union(b).copied().collect());
                                    if next.len() > limit {
                                        return Err(too_many());
                                    }
                                }
                            }
                            acc = minimise(next);
                        }
                        all.extend(acc);
                        if all.len() > limit {
                            return Err(too_many());
                        }
                    }
                    minimise(all)
                }
            }
        }
    })
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn walk(start: usize, n: usize, k: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            walk(i + 1, n, k, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    walk(0, n, k, &mut Vec::new(), &mut out);
    out
}

fn xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// The OpenPSA name of each gate: its identifier, or a fresh `G` number.
struct GateNames {
    taken: HashSet<String>,
    next: usize,
}

impl GateNames {
    fn of(top: &FtNode) -> Self {
        let mut taken = HashSet::from([TOP_GATE_ID.to_owned()]);
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

fn open_psa_formula(
    node: &FtNode,
    events: &[TreeEvent],
    names: &mut GateNames,
    gates: &mut Vec<(String, String)>,
) -> String {
    match node {
        FtNode::Constant { value } => format!("<constant value=\"{value}\"/>"),
        FtNode::Basic { event } => {
            format!("<basic-event name=\"{}\"/>", xml(&events[*event].name))
        }
        FtNode::Gate { op, children, .. } => {
            let args: Vec<String> = children
                .iter()
                .map(|child| match child {
                    FtNode::Gate { id, .. } => {
                        let name = names.name(id.as_ref());
                        let formula = open_psa_formula(child, events, names, gates);
                        let reference = format!("<gate name=\"{}\"/>", xml(&name));
                        gates.push((name, formula));
                        reference
                    }
                    other => open_psa_formula(other, events, names, gates),
                })
                .collect();
            let (open, close) = match op {
                GateOp::And => ("<and>".to_owned(), "</and>"),
                GateOp::Or => ("<or>".to_owned(), "</or>"),
                GateOp::AtLeast { k } => (format!("<atleast min=\"{k}\">"), "</atleast>"),
            };
            format!("{open}{}{close}", args.join(""))
        }
    }
}

/// A basic event's body in OpenPSA: its law as an expression when the
/// format has one, as `law` attributes otherwise, and its repairs as a
/// `repairs` attribute holding their JSON list.
fn open_psa_event(law: &BasicLaw, repairs: &[TreeRepair]) -> String {
    let float = |v: f64| format!("<float value=\"{v}\"/>");
    let mut attributes = Vec::new();
    let mut out = match law {
        BasicLaw::Exponential { rate } => format!(
            "      <exponential>{}<system-mission-time/></exponential>\n",
            float(*rate)
        ),
        BasicLaw::Weibull { shape, scale } => format!(
            "      <Weibull>{}{}{}<system-mission-time/></Weibull>\n",
            float(*scale),
            float(*shape),
            float(0.0)
        ),
        BasicLaw::Probability { probability } => format!("      {}\n", float(*probability)),
        other => {
            // No OpenPSA time-to-failure expression for this law: the law is
            // carried as attributes, so a reader loses nothing it could not
            // compute from the law itself.
            let json = serde_json::to_value(other).unwrap_or_default();
            if let Some(map) = json.as_object() {
                for (key, value) in map {
                    attributes.push((
                        key.clone(),
                        value.to_string().trim_matches('"').replace('"', "'"),
                    ));
                }
            }
            String::new()
        }
    };
    if !repairs.is_empty() {
        let json = serde_json::to_string(repairs).unwrap_or_default();
        attributes.push(("repairs".to_owned(), json));
    }
    if !attributes.is_empty() {
        out.push_str("      <attributes>\n");
        for (key, value) in attributes {
            out.push_str(&format!(
                "        <attribute name=\"{}\" value=\"{}\"/>\n",
                xml(&key),
                xml(&value)
            ));
        }
        out.push_str("      </attributes>\n");
    }
    out
}
