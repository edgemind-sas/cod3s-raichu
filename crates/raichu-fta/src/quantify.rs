//! Exact quantification: normalisation, module detection, one decision
//! diagram per module, and the measures read off the diagrams.
//!
//! 1. **Normalisation** pushes every negation down to the basic events
//!    (a negated vote `not atleast(k, n)` becomes `atleast(n - k + 1)` of
//!    the negated arguments), propagates constants, drops duplicate
//!    arguments of a conjunction or a disjunction, recognises a literal
//!    beside its own negation, turns a vote of `1` or of `n` into a
//!    disjunction or a conjunction, coalesces a gate into a parent of the
//!    same connective when nothing else references it, and shares
//!    identical gates.
//! 2. **Modules** are found in linear time (Dutuit and Rauzy 1996): a gate
//!    is a module when every basic event and gate below it is reached only
//!    through it, which the dates of a depth-first traversal decide. A
//!    module is quantified once, on its own diagram, and stands as a
//!    single variable in its parent's.
//! 3. **Each module** gets a reduced ordered BDD, its variables ordered by
//!    first encounter in a depth-first left-most traversal. The heuristic
//!    and the order it produced are recorded, because the size of the
//!    diagram depends on them.
//! 4. **Measures.** The probability is the root's, computed bottom-up over
//!    the modules. The Birnbaum importance of every event comes from one
//!    backward pass per module, chained through the modules
//!    (`∂top/∂e = ∂top/∂M · ∂M/∂e`). The other factors follow from the
//!    multilinearity of the top probability in each event:
//!    `P(top | e) = P + (1 - p) B` and `P(top | not e) = P - p B`.
//! 5. **Minimal cut sets**, only when every module is monotone: Rauzy's
//!    minimal solutions of each module's diagram, their exact number
//!    counted on the diagrams, and the sets themselves enumerated when
//!    that number fits the declared limit.
//! 6. **A module too large for its diagram** (with [`Engine::Auto`], the
//!    default) is quantified from its cut sets under cutoffs instead
//!    ([`crate::approx`]): its value is the pivotal upper bound of the
//!    retained sets (Rauzy 2020), and a guaranteed upper bound adds the
//!    mass the cutoffs neglected. The other modules stay exact; the result
//!    says which were not, and carries the bound.

use std::collections::HashMap;

use serde::Serialize;

use crate::dd::{Bdd, ZERO};
use crate::{FaultTreeMeasure, Formula, FtaError, Tree};

/// What a quantification may be given beside the tree.
#[derive(Debug, Clone)]
pub struct QuantifySettings {
    /// The instant the basic events' laws are read at. Required when any
    /// basic event the top reaches has a timed law.
    pub mission_time: Option<f64>,
    /// The most nodes all decision diagrams together may hold.
    pub max_bdd_nodes: usize,
    /// The most minimal cut sets enumerated. Their number is always
    /// computed when they are; past this limit the sets themselves are
    /// omitted and the result says why.
    pub cut_set_limit: usize,
    /// Whether to extract the minimal cut sets at all. Their extraction is
    /// usually the costliest step by far, and neither the probability nor
    /// the importance measures need it.
    pub cut_sets: bool,
    /// Which engine quantifies each module.
    pub engine: Engine,
    /// Cut-set engine: the largest cut-set order kept (`None`: no limit).
    pub max_order: Option<usize>,
    /// Cut-set engine: the smallest cut-set probability kept.
    pub min_cut_probability: f64,
    /// Cut-set engine: the most cut sets kept per module; past it the
    /// least probable are dropped and the probability cutoff raised.
    pub max_cut_sets: usize,
    /// Cut-set engine: the most partial cut sets expanded per module;
    /// past it, what remains open is neglected (and counted in the bound).
    pub max_expansions: u64,
    /// What a basic event's probability measures: its distribution
    /// without repair (the default), or its unavailability counting its
    /// [`crate::BasicEvent::repairs`].
    pub measure: FaultTreeMeasure,
    /// Whether to value every gate as well as the top
    /// ([`Quantification::gates`]). Off by default: it reads each gate off
    /// the literal the normalisation made of it, which costs the nodes of
    /// the gates the coalescing dissolved into their parents. Those nodes
    /// are bounded by their module's diagram budget but are not counted
    /// against [`QuantifySettings::max_bdd_nodes`] for the next modules,
    /// so asking for the gates changes no other number.
    pub gate_probabilities: bool,
}

/// Which engine quantifies a module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Engine {
    /// The exact diagram, and the cut-set engine for a module whose
    /// diagram outgrows `max_bdd_nodes`.
    Auto,
    /// The exact diagram only: a module that outgrows the budget is an
    /// error.
    Exact,
    /// The cut-set engine for every module.
    CutSets,
}

impl Default for QuantifySettings {
    fn default() -> Self {
        Self {
            mission_time: None,
            max_bdd_nodes: 10_000_000,
            cut_set_limit: 100_000,
            cut_sets: true,
            engine: Engine::Auto,
            max_order: None,
            min_cut_probability: 0.0,
            max_cut_sets: 1_000_000,
            max_expansions: 100_000_000,
            measure: FaultTreeMeasure::ProbabilityWithoutRepair,
            gate_probabilities: false,
        }
    }
}

/// The probability of one gate at the mission time.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GateProbability {
    /// The gate, by index into [`Tree::gates`]; `None` for the top
    /// formula itself.
    pub gate: Option<usize>,
    /// Its exact probability, in the quantification's measure; `None`
    /// when it has none, for the [`GateProbability::reason`] given.
    pub probability: Option<f64>,
    /// Why there is no exact probability; `None` when there is one.
    pub reason: Option<GateReason>,
}

/// Why a gate has no exact probability. The top is still quantified: a
/// gate without a value is a partial result announced, not a refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateReason {
    /// The gate lies in a module quantified from its cut sets under
    /// cutoffs: there is no diagram to read it off, and the module's own
    /// value is an estimate.
    CutSets,
    /// The gate reads a module quantified from its cut sets: its value
    /// would be an estimate.
    ReadsCutSets,
    /// The diagram of the gate outgrew its module's node budget.
    OutgrewBdd,
    /// The top, once simplified, does not depend on the gate.
    NotReached,
}

/// The importance measures of one basic event.
///
/// With `P` the top probability, `p` the event's, `P1` and `P0` the top
/// probability with the event certain and impossible:
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EventImportance {
    /// The event, by index into the tree's events.
    pub event: usize,
    /// Its probability at the mission time; `None` for an event the top
    /// does not reach whose law could not be read there.
    pub probability: Option<f64>,
    /// Birnbaum's marginal importance, `P1 - P0`.
    pub birnbaum: f64,
    /// The critical importance factor, `p (P1 - P0) / P`.
    pub criticality: Option<f64>,
    /// Fussell-Vesely, in the PSA-code convention `(P - P0) / P`: the
    /// fraction of the risk removed by making the event impossible. Equal
    /// to the critical importance factor in exact arithmetic on a
    /// multilinear top probability; computed from its own definition so a
    /// cut-set-based approximation can later differ from it visibly.
    pub fussell_vesely: Option<f64>,
    /// The diagnostic importance factor, `p P1 / P`: the probability the
    /// event has occurred given that the top has.
    pub diagnostic: Option<f64>,
    /// Risk achievement worth, `P1 / P`.
    pub risk_achievement_worth: Option<f64>,
    /// Risk reduction worth, `P / P0`; `None` when `P0 = 0` (making the
    /// event impossible removes the whole risk, an infinite worth) or when
    /// `P = 0`.
    pub risk_reduction_worth: Option<f64>,
}

/// One module, recorded with the order it was quantified in.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModuleRecord {
    /// `M0` is the top; a submodule appears in its parent's variables
    /// under this name.
    pub name: String,
    /// `"bdd"` (its exact diagram) or `"cut_sets"` (its cut sets under
    /// cutoffs).
    pub method: &'static str,
    /// Whether its probability is exact: its own method is, and so is
    /// every submodule it reads.
    pub exact: bool,
    /// The variables, in diagram order: basic-event names and submodule
    /// names.
    pub variables: Vec<String>,
    /// Nodes of its BDD, terminals included: the size of the function's
    /// diagram.
    pub bdd_nodes: usize,
    /// Nodes its construction created, intermediate results included.
    pub nodes_created: usize,
    /// Its probability: exact, or the pivotal upper bound of its retained
    /// cut sets.
    pub probability: f64,
    /// A guaranteed upper bound on its probability; `None` when none can
    /// be certified (a non-monotone module reading an approximated one).
    pub upper_bound: Option<f64>,
    /// Whether its function is monotone in its variables.
    pub monotone: bool,
    /// Cut-set engine: how many minimal cut sets it retained.
    pub retained_cut_sets: Option<u128>,
    /// Cut-set engine: an upper bound on the probability of what the
    /// cutoffs neglected, capped at 1.
    pub neglected: Option<f64>,
    /// Cut-set engine: the rare-event approximation of the retained sets.
    pub rare_event: Option<f64>,
    /// Cut-set engine: the min-cut upper bound of the retained sets.
    pub mincut_upper_bound: Option<f64>,
    /// Cut-set engine: the pivotal upper bound of the retained sets.
    pub pivotal_upper_bound: Option<f64>,
    /// Cut-set engine: partial cut sets expanded.
    pub expansions: Option<u64>,
}

/// The cutoffs a quantification ran under.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CutoffRecord {
    /// The largest cut-set order kept.
    pub max_order: Option<usize>,
    /// The smallest cut-set probability kept.
    pub min_probability: f64,
    /// The most cut sets kept per module.
    pub max_cut_sets: usize,
    /// The most partial cut sets expanded per module.
    pub max_expansions: u64,
}

/// How the numbers were produced.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Provenance {
    /// The variable-ordering heuristic.
    pub variable_order: &'static str,
    /// The engine asked for.
    pub engine: Engine,
    /// The cut-set engine's cutoffs (used only by modules it quantified).
    pub cutoffs: CutoffRecord,
    /// Every module, top first.
    pub modules: Vec<ModuleRecord>,
    /// Nodes of all module BDDs together.
    pub bdd_nodes: usize,
    /// Nodes of all minimal-cut-set diagrams together (0 when none was
    /// built).
    pub zbdd_nodes: usize,
    /// Nodes created by every construction, intermediate results
    /// included: what `max_bdd_nodes` bounds.
    pub nodes_created: usize,
}

/// A quantified tree.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Quantification {
    /// The top-event probability at the mission time: exact, or, when a
    /// module fell back to its cut sets, computed with that module's
    /// pivotal upper bound.
    pub probability: f64,
    /// `"bdd"`, `"cut_sets"`, or `"bdd+cut_sets"` when the modules used
    /// both; [`Provenance::modules`] says which used which.
    pub method: &'static str,
    /// Whether the probability is exact (no cutoff, no bound).
    pub exact: bool,
    /// A guaranteed upper bound on the top-event probability: equal to
    /// the probability when exact; `None` when none can be certified.
    pub upper_bound: Option<f64>,
    /// The instant the laws were read at.
    pub mission_time: Option<f64>,
    /// Whether the top is monotone in every basic event.
    pub coherent: bool,
    /// Every basic event's importance, in the tree's event order.
    pub importance: Vec<EventImportance>,
    /// The number of minimal cut sets found (saturating at `u128::MAX`),
    /// exact when [`Quantification::cut_sets_complete`]; `None` for a
    /// non-coherent tree, or when they were not requested or not
    /// extracted.
    pub cut_set_count: Option<u128>,
    /// Whether no cutoff removed any cut set: the probability may still be
    /// a bound (a module quantified from its cut sets), but the sets are
    /// all of them.
    pub cut_sets_complete: bool,
    /// The minimal cut sets, each a sorted list of event indices, by size
    /// then lexicographically; `None` when omitted, with the reason in
    /// [`Quantification::cut_sets_omitted`].
    pub minimal_cut_sets: Option<Vec<Vec<usize>>>,
    /// Why the minimal cut sets are not listed.
    pub cut_sets_omitted: Option<String>,
    /// What deserves attention before trusting the number: a module that
    /// fell back to its cut sets, a cutoff that bit, a neglected mass
    /// larger than the estimate, each coded with its module. Empty for an
    /// exact result. Serialised with each warning's `message`
    /// ([`raichu_core::serialize_coded`]).
    #[serde(serialize_with = "raichu_core::serialize_coded")]
    pub warnings: Vec<QuantifyWarning>,
    /// How the numbers were produced.
    pub provenance: Provenance,
    /// When [`QuantifySettings::gate_probabilities`] asks for them: the
    /// top first (`gate: None`), then every gate in [`Tree::gates`]
    /// order. Empty otherwise.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub gates: Vec<GateProbability>,
}

/// What deserves attention in a quantification, module by module: a code
/// (`kind`) with its parameters, the English sentence being its
/// `Display`. `module` is the module's index in
/// [`Provenance::modules`], named `M{module}` there. Consumers key on
/// `kind`, never on the text.
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum QuantifyWarning {
    /// The module's exact diagram outgrew the node budget: it was
    /// quantified from its cut sets under cutoffs.
    ModuleOutgrewBdd {
        /// The module's index in the provenance.
        module: usize,
        /// The budget it outgrew ([`QuantifySettings::max_bdd_nodes`]).
        max_bdd_nodes: usize,
    },
    /// The count cutoff was reached: the least probable cut sets were
    /// dropped and the probability cutoff raised.
    MaxCutSetsReached {
        /// The module's index in the provenance.
        module: usize,
        /// The cutoff ([`QuantifySettings::max_cut_sets`]).
        max_cut_sets: usize,
    },
    /// The expansion budget ran out: what remained open is only bounded.
    MaxExpansionsExhausted {
        /// The module's index in the provenance.
        module: usize,
        /// The budget ([`QuantifySettings::max_expansions`]).
        max_expansions: u64,
    },
    /// The probability mass the cutoffs neglected exceeds the retained
    /// estimate: the estimate may be far too optimistic.
    NeglectedMassExceedsEstimate {
        /// The module's index in the provenance.
        module: usize,
        /// The neglected mass.
        neglected: f64,
        /// The retained estimate (the pivotal upper bound).
        estimate: f64,
    },
    /// A warning read from a document of version 1, which carried a plain
    /// English sentence and no code. Never produced.
    Legacy {
        /// The sentence, as the document carried it.
        message: String,
    },
}

impl raichu_core::LegacyWarning for QuantifyWarning {
    fn legacy(message: String) -> Self {
        QuantifyWarning::Legacy { message }
    }

    fn is_legacy(&self) -> bool {
        matches!(self, QuantifyWarning::Legacy { .. })
    }
}

impl std::fmt::Display for QuantifyWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantifyWarning::ModuleOutgrewBdd {
                module,
                max_bdd_nodes,
            } => write!(
                f,
                "module M{module}: its exact diagram outgrew max_bdd_nodes \
                 ({max_bdd_nodes}): quantified from its cut sets under cutoffs"
            ),
            QuantifyWarning::MaxCutSetsReached {
                module,
                max_cut_sets,
            } => write!(
                f,
                "module M{module}: max_cut_sets ({max_cut_sets}) was reached: the \
                 least probable cut sets were dropped and the probability cutoff \
                 raised"
            ),
            QuantifyWarning::MaxExpansionsExhausted {
                module,
                max_expansions,
            } => write!(
                f,
                "module M{module}: max_expansions ({max_expansions}) ran out: what \
                 remained open is only bounded"
            ),
            QuantifyWarning::NeglectedMassExceedsEstimate {
                module,
                neglected,
                estimate,
            } => write!(
                f,
                "module M{module}: the neglected mass ({neglected:.3e}) exceeds the \
                 retained estimate ({estimate:.3e}): the estimate may be far too \
                 optimistic; lower the cutoffs or raise max_cut_sets"
            ),
            QuantifyWarning::Legacy { message } => f.write_str(message),
        }
    }
}

/// A warning of one module, before the module's index is known.
#[derive(Debug, Clone, Copy)]
enum ModuleNote {
    OutgrewBdd,
    CutSetsReached,
    ExpansionsExhausted,
    NeglectedExceeds { neglected: f64, estimate: f64 },
}

impl ModuleNote {
    fn at(self, module: usize, settings: &QuantifySettings) -> QuantifyWarning {
        match self {
            ModuleNote::OutgrewBdd => QuantifyWarning::ModuleOutgrewBdd {
                module,
                max_bdd_nodes: settings.max_bdd_nodes,
            },
            ModuleNote::CutSetsReached => QuantifyWarning::MaxCutSetsReached {
                module,
                max_cut_sets: settings.max_cut_sets,
            },
            ModuleNote::ExpansionsExhausted => QuantifyWarning::MaxExpansionsExhausted {
                module,
                max_expansions: settings.max_expansions,
            },
            ModuleNote::NeglectedExceeds {
                neglected,
                estimate,
            } => QuantifyWarning::NeglectedMassExceedsEstimate {
                module,
                neglected,
                estimate,
            },
        }
    }
}

/// Quantify `tree`: exactly, or, for a module too large for its diagram,
/// from its cut sets under cutoffs with a guaranteed upper bound (see
/// [`QuantifySettings::engine`]).
///
/// Runs on a thread of its own with a large stack (1 GiB of address
/// space, committed as used): the diagram operations
/// recurse once per variable level and the normalisation once per gate
/// level, so a flat gate of a hundred thousand events or a chain of as
/// many gates would overflow a default stack.
pub fn quantify(tree: &Tree, settings: &QuantifySettings) -> Result<Quantification, FtaError> {
    crate::with_large_stack(|| quantify_here(tree, settings))
}

fn quantify_here(tree: &Tree, settings: &QuantifySettings) -> Result<Quantification, FtaError> {
    tree.validate()?;
    if let Some(t) = settings.mission_time {
        if !t.is_finite() || t < 0.0 {
            return Err(FtaError::BadMissionTime(t));
        }
    }
    let mut graph = Graph::default();
    let mut memo = HashMap::new();
    let top = graph.build(tree, &tree.top, true, &mut memo);
    let track = settings.gate_probabilities;
    // Each gate's literal before coalescing, and whether the literal is
    // its negation: a gate reached under a negation only is built
    // negated.
    let gate_lits: Vec<Option<(Lit, bool)>> = if track {
        (0..tree.gates.len())
            .map(|g| {
                memo.get(&(g, true))
                    .map(|&lit| (lit, false))
                    .or_else(|| memo.get(&(g, false)).map(|&lit| (lit, true)))
            })
            .collect()
    } else {
        Vec::new()
    };
    let (graph, root, remap) = graph.coalesce(top, track);
    let targets: Vec<Target> = if track {
        let after = |(lit, negated): (Lit, bool)| match lit {
            Lit::Node(n) => remap.get(&n).map(|&lit| (lit, negated)),
            other => Some((other, negated)),
        };
        let top = match root {
            Root::Node(n) => Lit::Node(n),
            Root::Constant(v) => Lit::Const(v),
        };
        std::iter::once(Target {
            gate: None,
            lit: Some((top, false)),
        })
        .chain(gate_lits.into_iter().enumerate().map(|(g, lit)| Target {
            gate: Some(g),
            lit: lit.and_then(after),
        }))
        .collect()
    } else {
        Vec::new()
    };

    let n_events = tree.events.len();
    // Only the events the top reaches must have a readable law.
    let mut reached = vec![false; n_events];
    if let Root::Node(n) = root {
        graph.mark_events(Lit::Node(n), &mut reached);
    }
    let mut probability = Vec::with_capacity(n_events);
    for (index, event) in tree.events.iter().enumerate() {
        match crate::repair::event_probability(event, settings.measure, settings.mission_time) {
            Ok(p) => probability.push(Some(p)),
            Err(e) if reached[index] => return Err(e),
            Err(_) => probability.push(None),
        }
    }

    match root {
        Root::Node(n) => Quantifier::new(&graph, tree, settings, probability).run(n, &targets),
        Root::Constant(value) => {
            let mut result = constant_result(settings, value, &probability);
            result.gates = targets
                .iter()
                .map(|t| {
                    t.lit
                        .and_then(|(lit, negated)| {
                            direct(lit, &probability).map(|p| t.with(p, negated))
                        })
                        .unwrap_or_else(|| t.without(GateReason::NotReached))
                })
                .collect();
            Ok(result)
        }
    }
}

/// A gate to value: its index (`None` for the top) and its literal in the
/// coalesced graph, negated or not; `None` when the simplified top does
/// not reach it.
struct Target {
    gate: Option<usize>,
    lit: Option<(Lit, bool)>,
}

impl Target {
    fn with(&self, probability: f64, negated: bool) -> GateProbability {
        GateProbability {
            gate: self.gate,
            probability: Some(if negated {
                1.0 - probability
            } else {
                probability
            }),
            reason: None,
        }
    }

    fn without(&self, reason: GateReason) -> GateProbability {
        GateProbability {
            gate: self.gate,
            probability: None,
            reason: Some(reason),
        }
    }
}

/// The probability of a literal that needs no diagram: a constant, or an
/// event whose probability could be read; `None` for a gate node or an
/// event without a probability.
fn direct(lit: Lit, probability: &[Option<f64>]) -> Option<f64> {
    match lit {
        Lit::Const(v) => Some(f64::from(u8::from(v))),
        Lit::Var(e, positive) => {
            let p = probability.get(e as usize).copied().flatten()?;
            Some(if positive { p } else { 1.0 - p })
        }
        Lit::Node(_) => None,
    }
}

fn constant_result(
    settings: &QuantifySettings,
    value: bool,
    probability: &[Option<f64>],
) -> Quantification {
    let p = if value { 1.0 } else { 0.0 };
    let importance = probability
        .iter()
        .enumerate()
        .map(|(event, q)| measures(event, *q, p, 0.0))
        .collect();
    Quantification {
        probability: p,
        method: "bdd",
        exact: true,
        upper_bound: Some(p),
        cut_sets_complete: true,
        mission_time: settings.mission_time,
        coherent: true,
        importance,
        cut_set_count: settings.cut_sets.then_some(u128::from(value)),
        minimal_cut_sets: settings
            .cut_sets
            .then(|| if value { vec![Vec::new()] } else { Vec::new() }),
        cut_sets_omitted: (!settings.cut_sets).then(|| "not requested".to_owned()),
        warnings: Vec::new(),
        provenance: Provenance {
            variable_order: VARIABLE_ORDER,
            engine: settings.engine,
            cutoffs: CutoffRecord {
                max_order: settings.max_order,
                min_probability: settings.min_cut_probability,
                max_cut_sets: settings.max_cut_sets,
                max_expansions: settings.max_expansions,
            },
            modules: Vec::new(),
            bdd_nodes: 0,
            zbdd_nodes: 0,
            nodes_created: 0,
        },
        gates: Vec::new(),
    }
}

const VARIABLE_ORDER: &str = "depth_first_left_most";

fn measures(event: usize, q: Option<f64>, top: f64, birnbaum: f64) -> EventImportance {
    let p = q.unwrap_or(0.0);
    let p1 = top + (1.0 - p) * birnbaum;
    // `P - p B` cancels when the event carries the whole risk: a residue
    // of a few ulps of `P`, of either sign, is a zero, not a finite worth.
    let p0 = top - p * birnbaum;
    let p0 = if p0 <= 16.0 * f64::EPSILON * top {
        0.0
    } else {
        p0
    };
    let ratio = |num: f64, den: f64| if den > 0.0 { Some(num / den) } else { None };
    EventImportance {
        event,
        probability: q,
        birnbaum,
        criticality: ratio(p * birnbaum, top),
        fussell_vesely: ratio(top - p0, top),
        diagnostic: ratio(p * p1, top),
        risk_achievement_worth: ratio(p1, top),
        risk_reduction_worth: if top > 0.0 { ratio(top, p0) } else { None },
    }
}

// ---------------------------------------------------------------------
// The normalised graph.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Lit {
    Const(bool),
    Var(u32, bool),
    Node(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GOp {
    And,
    Or,
    AtLeast(usize),
}

/// The normalised top: a gate, or a constant once simplified.
enum Root {
    Node(u32),
    Constant(bool),
}

/// A conjunction or a disjunction of `args`, simplified: constants
/// propagated, duplicates dropped, a literal beside its own negation
/// recognised. A single literal or a constant when it reduces to one,
/// otherwise the arguments to keep, sorted.
fn simplify(op: GOp, args: Vec<Lit>) -> Result<Lit, Vec<Lit>> {
    let (absorbing, neutral) = match op {
        GOp::And => (false, true),
        GOp::Or => (true, false),
        GOp::AtLeast(_) => return Err(args),
    };
    let mut kept = Vec::with_capacity(args.len());
    for arg in args {
        match arg {
            Lit::Const(v) if v == neutral => {}
            Lit::Const(_) => return Ok(Lit::Const(absorbing)),
            other => kept.push(other),
        }
    }
    kept.sort();
    kept.dedup();
    for pair in kept.windows(2) {
        if let (Lit::Var(a, pa), Lit::Var(b, pb)) = (pair[0], pair[1]) {
            if a == b && pa != pb {
                return Ok(Lit::Const(absorbing));
            }
        }
    }
    match kept.len() {
        0 => Ok(Lit::Const(neutral)),
        1 => Ok(kept[0]),
        _ => Err(kept),
    }
}

#[derive(Default)]
struct Graph {
    nodes: Vec<(GOp, Vec<Lit>)>,
    shared: HashMap<(GOp, Vec<Lit>), u32>,
}

impl Graph {
    fn build(
        &mut self,
        tree: &Tree,
        formula: &Formula,
        positive: bool,
        memo: &mut HashMap<(usize, bool), Lit>,
    ) -> Lit {
        match formula {
            Formula::Constant { value } => Lit::Const(*value == positive),
            Formula::Event { event } => Lit::Var(*event as u32, positive),
            Formula::Gate { gate } => {
                if let Some(&lit) = memo.get(&(*gate, positive)) {
                    return lit;
                }
                let lit = self.build(tree, &tree.gates[*gate].formula, positive, memo);
                memo.insert((*gate, positive), lit);
                lit
            }
            Formula::Not { arg } => self.build(tree, arg, !positive, memo),
            Formula::And { args } | Formula::Or { args } => {
                let conjunction = matches!(formula, Formula::And { .. }) == positive;
                let args = args
                    .iter()
                    .map(|a| self.build(tree, a, positive, memo))
                    .collect();
                self.make(if conjunction { GOp::And } else { GOp::Or }, args)
            }
            Formula::AtLeast { k, args } => {
                let args: Vec<Lit> = args
                    .iter()
                    .map(|a| self.build(tree, a, positive, memo))
                    .collect();
                let k = if positive {
                    *k as isize
                } else {
                    // not atleast(k, n) = atleast(n - k + 1) of the negations.
                    args.len() as isize - *k as isize + 1
                };
                self.make_vote(k, args)
            }
        }
    }

    fn make_vote(&mut self, mut k: isize, args: Vec<Lit>) -> Lit {
        let mut kept = Vec::with_capacity(args.len());
        for arg in args {
            match arg {
                Lit::Const(true) => k -= 1,
                Lit::Const(false) => {}
                other => kept.push(other),
            }
        }
        if k <= 0 {
            return Lit::Const(true);
        }
        let k = k as usize;
        if k > kept.len() {
            return Lit::Const(false);
        }
        if k == 1 {
            return self.make(GOp::Or, kept);
        }
        if k == kept.len() {
            return self.make(GOp::And, kept);
        }
        kept.sort();
        self.intern(GOp::AtLeast(k), kept)
    }

    fn make(&mut self, op: GOp, args: Vec<Lit>) -> Lit {
        if let GOp::AtLeast(k) = op {
            return self.make_vote(k as isize, args);
        }
        match simplify(op, args) {
            Ok(lit) => lit,
            Err(kept) => self.intern(op, kept),
        }
    }

    /// A conjunction or a disjunction simplified as [`Graph::make`] does,
    /// as a node no later [`Graph::make`] shares: the literal of a gate
    /// the coalescing dissolved into its parent, kept for its probability
    /// alone. Detached nodes are never reached from the root, so the graph
    /// the top is quantified on is the one it would be without them.
    fn make_detached(&mut self, op: GOp, args: Vec<Lit>) -> Lit {
        match simplify(op, args) {
            Ok(lit) => lit,
            Err(kept) => match self.shared.get(&(op, kept.clone())) {
                Some(&id) => Lit::Node(id),
                None => {
                    self.nodes.push((op, kept));
                    Lit::Node(self.nodes.len() as u32 - 1)
                }
            },
        }
    }

    fn intern(&mut self, op: GOp, args: Vec<Lit>) -> Lit {
        if let Some(&id) = self.shared.get(&(op, args.clone())) {
            return Lit::Node(id);
        }
        let id = self.nodes.len() as u32;
        self.nodes.push((op, args.clone()));
        self.shared.insert((op, args), id);
        Lit::Node(id)
    }

    /// Rebuild the graph reachable from `root`, merging a gate into a
    /// parent of the same connective when that parent is its only
    /// referrer. A top that simplifies to a single literal is wrapped in
    /// a one-argument gate, so the root is a node unless constant.
    ///
    /// Answers also the literal each node of this graph became. With
    /// `track`, a merged node becomes a detached one
    /// ([`Graph::make_detached`]) computing what it computed; without, it
    /// has no literal.
    fn coalesce(self, root: Lit, track: bool) -> (Graph, Root, HashMap<u32, Lit>) {
        let mut refs = vec![0usize; self.nodes.len()];
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = Vec::new();
        if let Lit::Node(n) = root {
            stack.push(n);
        }
        while let Some(n) = stack.pop() {
            if seen[n as usize] {
                continue;
            }
            seen[n as usize] = true;
            for arg in &self.nodes[n as usize].1 {
                if let Lit::Node(c) = arg {
                    refs[*c as usize] += 1;
                    stack.push(*c);
                }
            }
        }
        let mut out = Graph::default();
        let mut memo: HashMap<u32, Lit> = HashMap::new();
        let copied = match root {
            Lit::Node(n) => self.copy(n, &refs, &mut out, &mut memo, track),
            other => other,
        };
        let root = match copied {
            Lit::Node(n) => Root::Node(n),
            Lit::Const(v) => Root::Constant(v),
            var @ Lit::Var(..) => {
                let id = out.nodes.len() as u32;
                out.nodes.push((GOp::Or, vec![var]));
                Root::Node(id)
            }
        };
        (out, root, memo)
    }

    fn copy(
        &self,
        n: u32,
        refs: &[usize],
        out: &mut Graph,
        memo: &mut HashMap<u32, Lit>,
        track: bool,
    ) -> Lit {
        if let Some(&lit) = memo.get(&n) {
            return lit;
        }
        let (op, args) = &self.nodes[n as usize];
        let mut flat = Vec::with_capacity(args.len());
        self.flatten(*op, args, refs, out, memo, &mut flat, track);
        let lit = out.make(*op, flat);
        memo.insert(n, lit);
        lit
    }

    /// Whether `child`, an argument of a gate of connective `op`, is merged
    /// into it.
    fn dissolves(&self, op: GOp, child: u32, refs: &[usize]) -> bool {
        refs[child as usize] == 1
            && matches!(op, GOp::And | GOp::Or)
            && self.nodes[child as usize].0 == op
    }

    #[allow(clippy::too_many_arguments)]
    fn flatten(
        &self,
        op: GOp,
        args: &[Lit],
        refs: &[usize],
        out: &mut Graph,
        memo: &mut HashMap<u32, Lit>,
        flat: &mut Vec<Lit>,
        track: bool,
    ) {
        for arg in args {
            match arg {
                Lit::Node(c) if self.dissolves(op, *c, refs) => {
                    let (_, inner) = &self.nodes[*c as usize];
                    self.flatten(op, inner, refs, out, memo, flat, track);
                    if track {
                        self.detach(*c, refs, out, memo);
                    }
                }
                Lit::Node(c) => flat.push(self.copy(*c, refs, out, memo, track)),
                other => flat.push(*other),
            }
        }
    }

    /// Give the merged node `n` the literal of a detached node over its
    /// own arguments: their literals, a merged argument's being its
    /// detached node, so nested merged gates share their diagrams.
    /// Called once its arguments are copied or detached.
    fn detach(&self, n: u32, refs: &[usize], out: &mut Graph, memo: &mut HashMap<u32, Lit>) {
        if memo.contains_key(&n) {
            return;
        }
        let (op, args) = &self.nodes[n as usize];
        let mut mapped = Vec::with_capacity(args.len());
        for arg in args {
            mapped.push(match arg {
                Lit::Node(c) => match memo.get(c) {
                    Some(&lit) => lit,
                    None => self.copy(*c, refs, out, memo, true),
                },
                other => *other,
            });
        }
        let lit = out.make_detached(*op, mapped);
        memo.insert(n, lit);
    }

    fn mark_events(&self, root: Lit, reached: &mut [bool]) {
        let mut stack = vec![root];
        let mut seen = vec![false; self.nodes.len()];
        while let Some(lit) = stack.pop() {
            match lit {
                Lit::Var(e, _) => reached[e as usize] = true,
                Lit::Node(n) if !seen[n as usize] => {
                    seen[n as usize] = true;
                    stack.extend(self.nodes[n as usize].1.iter().copied());
                }
                _ => {}
            }
        }
    }

    /// The gates that are modules (Dutuit and Rauzy 1996), the root
    /// included.
    fn modules(&self, root: u32, n_events: usize) -> Vec<bool> {
        let n = self.nodes.len();
        let mut first = vec![0u64; n];
        let mut second = vec![0u64; n];
        let mut last = vec![0u64; n];
        let mut first_v = vec![0u64; n_events];
        let mut last_v = vec![0u64; n_events];
        let mut date = 0u64;
        // Iterative depth-first traversal: (node, next argument index).
        enum Step {
            Enter(Lit),
            Leave(u32),
        }
        let mut stack = vec![Step::Enter(Lit::Node(root))];
        while let Some(step) = stack.pop() {
            match step {
                Step::Enter(Lit::Var(e, _)) => {
                    date += 1;
                    if first_v[e as usize] == 0 {
                        first_v[e as usize] = date;
                    }
                    last_v[e as usize] = date;
                }
                Step::Enter(Lit::Node(g)) => {
                    date += 1;
                    if first[g as usize] == 0 {
                        first[g as usize] = date;
                        stack.push(Step::Leave(g));
                        for arg in self.nodes[g as usize].1.iter().rev() {
                            stack.push(Step::Enter(*arg));
                        }
                    }
                    last[g as usize] = date;
                }
                Step::Enter(Lit::Const(_)) => {}
                Step::Leave(g) => {
                    date += 1;
                    second[g as usize] = date;
                }
            }
        }
        // Children are created before parents: ascending ids are bottom-up.
        let mut min_d = vec![u64::MAX; n];
        let mut max_d = vec![0u64; n];
        let mut module = vec![false; n];
        for g in 0..n {
            if first[g] == 0 {
                continue;
            }
            let mut lo = u64::MAX;
            let mut hi = 0u64;
            for arg in &self.nodes[g].1 {
                match *arg {
                    Lit::Var(e, _) => {
                        lo = lo.min(first_v[e as usize]);
                        hi = hi.max(last_v[e as usize]);
                    }
                    Lit::Node(c) => {
                        let c = c as usize;
                        lo = lo.min(first[c]).min(min_d[c]);
                        hi = hi.max(last[c]).max(max_d[c]);
                    }
                    Lit::Const(_) => {}
                }
            }
            min_d[g] = lo;
            max_d[g] = hi;
            module[g] = first[g] < lo && hi < second[g];
        }
        module[root as usize] = true;
        module
    }
}

// ---------------------------------------------------------------------
// Where each gate is valued.
// ---------------------------------------------------------------------

/// The module whose diagram holds each node: a module holds itself and
/// every node below it short of a submodule; an event and a submodule are
/// variables of exactly one module. A detached node (a gate the
/// coalescing dissolved, [`Graph::make_detached`]) is held by the module
/// of its arguments.
#[derive(Default)]
struct Homes {
    node: HashMap<u32, u32>,
    event: HashMap<u32, u32>,
    submodule: HashMap<u32, u32>,
}

impl Homes {
    fn of(graph: &Graph, modules: &[u32], is_module: &[bool]) -> Self {
        let mut homes = Homes::default();
        for &m in modules {
            homes.node.insert(m, m);
            let mut stack = vec![m];
            while let Some(n) = stack.pop() {
                for arg in &graph.nodes[n as usize].1 {
                    match *arg {
                        Lit::Var(e, _) => {
                            homes.event.insert(e, m);
                        }
                        Lit::Node(c) if is_module[c as usize] => {
                            homes.submodule.insert(c, m);
                        }
                        Lit::Node(c) => {
                            if homes.node.insert(c, m).is_none() {
                                stack.push(c);
                            }
                        }
                        Lit::Const(_) => {}
                    }
                }
            }
        }
        homes
    }

    /// The module holding `n`; `None` for a detached node whose
    /// arguments the top no longer reaches.
    fn of_node(&self, graph: &Graph, n: u32) -> Option<u32> {
        if let Some(&m) = self.node.get(&n) {
            return Some(m);
        }
        graph.nodes[n as usize].1.iter().find_map(|arg| match *arg {
            Lit::Var(e, _) => self.event.get(&e).copied(),
            Lit::Node(c) => self
                .submodule
                .get(&c)
                .copied()
                .or_else(|| self.of_node(graph, c)),
            Lit::Const(_) => None,
        })
    }
}

// ---------------------------------------------------------------------
// Per-module results.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Event(u32),
    Module(u32),
}

/// One module quantified, by its exact diagram or by its cut sets under
/// cutoffs.
struct ModuleResult {
    node: u32,
    keys: Vec<Key>,
    method: &'static str,
    /// Its probability: exact, or the pivotal upper bound of its retained
    /// cut sets.
    value: f64,
    /// A guaranteed upper bound on its probability, when one can be
    /// certified.
    upper: Option<f64>,
    /// Whether `value` is exact: its own method is, and so is every
    /// submodule it reads.
    exact: bool,
    monotone: bool,
    /// `∂value / ∂p(level)`.
    gradient: Vec<f64>,
    /// Which levels its function depends on.
    support: Vec<bool>,
    diagram: Option<(Bdd, u32)>,
    retained: Option<crate::dd::Zbdd>,
    truncated: bool,
    neglected: Option<f64>,
    estimators: Option<(f64, f64, f64)>,
    expansions: Option<u64>,
    /// Why this module's number deserves attention, if it does.
    warnings: Vec<ModuleNote>,
    bdd_nodes: usize,
    nodes_created: usize,
    /// The probability of each gate node asked of this module, read off
    /// its diagram; `None` for one whose diagram outgrew the budget, and
    /// for every one in a module quantified from its cut sets.
    gates: HashMap<u32, Option<f64>>,
}

struct Quantifier<'a> {
    graph: &'a Graph,
    tree: &'a Tree,
    settings: &'a QuantifySettings,
    probability: Vec<Option<f64>>,
}

impl<'a> Quantifier<'a> {
    fn new(
        graph: &'a Graph,
        tree: &'a Tree,
        settings: &'a QuantifySettings,
        probability: Vec<Option<f64>>,
    ) -> Self {
        Self {
            graph,
            tree,
            settings,
            probability,
        }
    }

    fn run(self, root: u32, targets: &[Target]) -> Result<Quantification, FtaError> {
        let n_events = self.tree.events.len();
        let is_module = self.graph.modules(root, n_events);
        // Bottom-up: ascending node ids put every submodule before its
        // parent. An unreachable node is never a module: its dates stay
        // at zero.
        let module_nodes: Vec<u32> = (0..=root).filter(|&g| is_module[g as usize]).collect();
        let homes = if targets.is_empty() {
            Homes::default()
        } else {
            Homes::of(self.graph, &module_nodes, &is_module)
        };
        // The gate nodes each module values, by the module whose diagram
        // holds them.
        let mut wanted: HashMap<u32, Vec<u32>> = HashMap::new();
        for target in targets {
            if let Some((Lit::Node(n), _)) = target.lit {
                if let Some(m) = homes.of_node(self.graph, n) {
                    wanted.entry(m).or_default().push(n);
                }
            }
        }
        // A module that reduced to a terminal is that constant in its
        // parent, never a variable: a variable standing for a tautology
        // would make its parent's cut sets carry the empty set beside real
        // ones.
        let mut constant: HashMap<u32, bool> = HashMap::new();
        let mut done: HashMap<u32, ModuleResult> = HashMap::new();
        let mut order: Vec<u32> = Vec::new();
        let mut used = 0usize;
        for &m in &module_nodes {
            let asked = wanted.get(&m).map_or(&[][..], Vec::as_slice);
            let result = self.module(m, &is_module, &constant, &done, &mut used, asked)?;
            // A function of no variable is a constant, whatever its inputs'
            // methods; a truncated family is not (what was cut may hold).
            if !result.truncated
                && result.support.iter().all(|s| !s)
                && (result.value == 1.0 || result.value == 0.0)
            {
                constant.insert(m, result.value == 1.0);
            }
            done.insert(m, result);
            order.push(m);
        }
        let gates = self.gates(targets, &homes, &is_module, &constant, &done);

        // The live modules: the root, and every module its live parent's
        // function still depends on. A module its parent absorbed takes
        // part in nothing: not in the cut sets, not in the coherence of the
        // top.
        let mut live: std::collections::HashSet<u32> = std::collections::HashSet::from([root]);
        for m in order.iter().rev() {
            if !live.contains(m) {
                continue;
            }
            let result = &done[m];
            for (level, key) in result.keys.iter().enumerate() {
                if let Key::Module(n) = key {
                    // A cut-set module reads a submodule through its bounds
                    // even when no retained set holds it: pruned partials
                    // did, and their mass used its upper bound.
                    if result.support[level] || result.method == "cut_sets" {
                        live.insert(*n);
                    }
                }
            }
        }
        let mut results: Vec<ModuleResult> = order
            .iter()
            .filter(|m| live.contains(*m))
            .filter_map(|m| done.remove(m))
            .collect();
        let Some(top_result) = results.iter().find(|r| r.node == root) else {
            return Err(FtaError::Invalid(
                "internal: the top module is missing".to_owned(),
            ));
        };
        let top = top_result.value;
        let upper_bound = top_result.upper;

        // Top-down derivatives through the modules.
        let mut derivative: HashMap<u32, f64> = HashMap::from([(root, 1.0)]);
        let mut birnbaum = vec![0.0; n_events];
        for result in results.iter().rev() {
            let d = derivative.get(&result.node).copied().unwrap_or(0.0);
            for (level, key) in result.keys.iter().enumerate() {
                match *key {
                    Key::Event(e) => birnbaum[e as usize] += d * result.gradient[level],
                    Key::Module(n) => {
                        *derivative.entry(n).or_insert(0.0) += d * result.gradient[level];
                    }
                }
            }
        }
        let importance = self
            .probability
            .iter()
            .enumerate()
            .map(|(event, q)| measures(event, *q, top, birnbaum[event]))
            .collect();

        let exact = results.iter().all(|r| r.exact);
        let truncated = results.iter().any(|r| r.truncated);
        let retained_counts: HashMap<u32, u128> = results
            .iter()
            .filter_map(|r| {
                r.retained
                    .as_ref()
                    .map(|z| (r.node, z.weighted_count(&vec![1; r.keys.len()])))
            })
            .collect();
        let coherent = results.iter().all(|r| r.monotone);
        let method = if results.iter().all(|r| r.method == "bdd") {
            "bdd"
        } else if results.iter().all(|r| r.method == "cut_sets") {
            "cut_sets"
        } else {
            "bdd+cut_sets"
        };
        let indices: HashMap<u32, usize> = results
            .iter()
            .rev()
            .enumerate()
            .map(|(i, r)| (r.node, i))
            .collect();
        let names: HashMap<u32, String> = indices
            .iter()
            .map(|(&node, i)| (node, format!("M{i}")))
            .collect();
        let warnings: Vec<QuantifyWarning> = results
            .iter()
            .rev()
            .enumerate()
            .flat_map(|(module, r)| {
                r.warnings
                    .iter()
                    .map(move |note| note.at(module, self.settings))
            })
            .collect();
        let mut zbdd_nodes = 0usize;
        let mut zbdd_created = 0usize;
        let (cut_set_count, minimal_cut_sets, cut_sets_omitted) = if !self.settings.cut_sets {
            (None, None, Some("not requested".to_owned()))
        } else if coherent {
            let (count, sets, omitted, (size, created)) =
                self.cut_sets(&mut results, root, used)?;
            zbdd_nodes = size;
            zbdd_created = created;
            (count, sets, omitted)
        } else {
            (
                None,
                None,
                Some(
                    "the top is not monotone in its basic events (a negation \
                     survives simplification): its minimal cut sets are prime \
                     implicants, which this method does not compute"
                        .to_owned(),
                ),
            )
        };
        let modules = results
            .iter()
            .rev()
            .map(|r| ModuleRecord {
                name: names.get(&r.node).cloned().unwrap_or_default(),
                method: r.method,
                exact: r.exact,
                variables: r
                    .keys
                    .iter()
                    .map(|k| match k {
                        Key::Event(e) => self.tree.events[*e as usize].name.clone(),
                        // A submodule the parent's function no longer
                        // depends on: listed in the order, never quantified.
                        Key::Module(n) => names
                            .get(n)
                            .cloned()
                            .unwrap_or_else(|| "absorbed".to_owned()),
                    })
                    .collect(),
                bdd_nodes: r.bdd_nodes,
                nodes_created: r.nodes_created,
                probability: r.value,
                upper_bound: r.upper,
                monotone: r.monotone,
                retained_cut_sets: retained_counts.get(&r.node).copied(),
                neglected: r.neglected.map(|n| n.min(1.0)),
                rare_event: r.estimators.map(|e| e.0),
                mincut_upper_bound: r.estimators.map(|e| e.1),
                pivotal_upper_bound: r.estimators.map(|e| e.2),
                expansions: r.expansions,
            })
            .collect();
        Ok(Quantification {
            probability: top,
            method,
            exact,
            upper_bound,
            mission_time: self.settings.mission_time,
            coherent,
            importance,
            cut_set_count,
            cut_sets_complete: !truncated,
            warnings,
            minimal_cut_sets,
            cut_sets_omitted,
            provenance: Provenance {
                variable_order: VARIABLE_ORDER,
                engine: self.settings.engine,
                cutoffs: CutoffRecord {
                    max_order: self.settings.max_order,
                    min_probability: self.settings.min_cut_probability,
                    max_cut_sets: self.settings.max_cut_sets,
                    max_expansions: self.settings.max_expansions,
                },
                bdd_nodes: results.iter().map(|r| r.bdd_nodes).sum(),
                modules,
                zbdd_nodes,
                nodes_created: used + zbdd_created,
            },
            gates,
        })
    }

    /// Value every target off the module that holds it: exact when the
    /// module is exact for it, `None` with the reason otherwise.
    fn gates(
        &self,
        targets: &[Target],
        homes: &Homes,
        is_module: &[bool],
        constant: &HashMap<u32, bool>,
        done: &HashMap<u32, ModuleResult>,
    ) -> Vec<GateProbability> {
        // Whether a node, valued in module `m`, reads a submodule that is
        // not exact (memoised per node; a node is valued in one module).
        let mut inexact: HashMap<u32, bool> = HashMap::new();
        targets
            .iter()
            .map(|target| {
                let Some((lit, negated)) = target.lit else {
                    return target.without(GateReason::NotReached);
                };
                let Lit::Node(n) = lit else {
                    return direct(lit, &self.probability).map_or_else(
                        || target.without(GateReason::NotReached),
                        |p| target.with(p, negated),
                    );
                };
                let Some(m) = homes.of_node(self.graph, n) else {
                    return target.without(GateReason::NotReached);
                };
                let Some(module) = done.get(&m) else {
                    return target.without(GateReason::NotReached);
                };
                if module.method == "cut_sets" {
                    return target.without(GateReason::CutSets);
                }
                let Some(&value) = module.gates.get(&n) else {
                    return target.without(GateReason::NotReached);
                };
                let Some(value) = value else {
                    return target.without(GateReason::OutgrewBdd);
                };
                let reads_estimate = !module.exact
                    && (n == m
                        || self.reads_inexact(n, m, is_module, constant, done, &mut inexact));
                if reads_estimate {
                    return target.without(GateReason::ReadsCutSets);
                }
                target.with(value, negated)
            })
            .collect()
    }

    /// Whether node `n`, in module `m`'s diagram, depends on a submodule
    /// whose value is not exact.
    fn reads_inexact(
        &self,
        n: u32,
        m: u32,
        is_module: &[bool],
        constant: &HashMap<u32, bool>,
        done: &HashMap<u32, ModuleResult>,
        memo: &mut HashMap<u32, bool>,
    ) -> bool {
        if let Some(&known) = memo.get(&n) {
            return known;
        }
        let mut reads = false;
        for arg in &self.graph.nodes[n as usize].1 {
            reads = match *arg {
                Lit::Node(c) if c != m && is_module[c as usize] => {
                    !constant.contains_key(&c) && !done.get(&c).is_some_and(|r| r.exact)
                }
                Lit::Node(c) => self.reads_inexact(c, m, is_module, constant, done, memo),
                _ => false,
            };
            if reads {
                break;
            }
        }
        memo.insert(n, reads);
        reads
    }

    /// Quantify module `m`, its submodules already done.
    fn module(
        &self,
        m: u32,
        is_module: &[bool],
        constant: &HashMap<u32, bool>,
        done: &HashMap<u32, ModuleResult>,
        used: &mut usize,
        asked: &[u32],
    ) -> Result<ModuleResult, FtaError> {
        // Variable order: depth-first left-most, first encounter.
        let mut keys: Vec<Key> = Vec::new();
        let mut level_of: HashMap<Key, u32> = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        self.order(
            m,
            m,
            is_module,
            constant,
            &mut keys,
            &mut level_of,
            &mut seen,
        );
        let sub = |n: &u32| done.get(n);
        let level_p: Vec<f64> = keys
            .iter()
            .map(|k| match k {
                Key::Event(e) => self.probability[*e as usize].unwrap_or(0.0),
                Key::Module(n) => sub(n).map_or(0.0, |r| r.value),
            })
            .collect();
        // Upper bounds of the variables: exact for events, certified for
        // submodules when they could be.
        let level_u: Option<Vec<f64>> = keys
            .iter()
            .map(|k| match k {
                Key::Event(e) => Some(self.probability[*e as usize].unwrap_or(0.0)),
                Key::Module(n) => sub(n).and_then(|r| r.upper),
            })
            .collect();
        let inputs_exact = keys.iter().all(|k| match k {
            Key::Event(_) => true,
            Key::Module(n) => sub(n).is_some_and(|r| r.exact),
        });

        let exact_attempt = match self.settings.engine {
            Engine::CutSets => None,
            Engine::Exact | Engine::Auto => {
                let budget = self.settings.max_bdd_nodes.saturating_sub(*used).max(2);
                let mut bdd = Bdd::new(budget);
                let mut memo: HashMap<u32, u32> = HashMap::new();
                match self.build(m, m, is_module, constant, &level_of, &mut bdd, &mut memo) {
                    Ok(root) => Some((bdd, root, memo)),
                    Err(FtaError::TooLarge(_)) if self.settings.engine == Engine::Auto => None,
                    Err(e) => return Err(e),
                }
            }
        };
        if let Some((mut bdd, root, mut memo)) = exact_attempt {
            // The top's own work: what the budget of the next modules and
            // the provenance count, whether gates are asked for or not.
            let built = bdd.len();
            // A gate the coalescing dissolved is built over the nodes
            // already there; the others are in the diagram already.
            let mut gate_ids = Vec::with_capacity(asked.len());
            for &node in asked {
                let id = match memo.get(&node) {
                    Some(&id) => Some(id),
                    None => match self
                        .build(node, m, is_module, constant, &level_of, &mut bdd, &mut memo)
                    {
                        Ok(id) => Some(id),
                        Err(FtaError::TooLarge(_)) => None,
                        Err(e) => return Err(e),
                    },
                };
                gate_ids.push((node, id));
            }
            *used += built;
            let prob = bdd.probabilities(&level_p);
            let gates = gate_ids
                .into_iter()
                .map(|(node, id)| (node, id.map(|id| prob[id as usize])))
                .collect();
            let value = prob[root as usize];
            let gradient = bdd.birnbaum(root, &level_p, &prob);
            let monotone = bdd.is_monotone(root);
            let upper = if inputs_exact {
                Some(value)
            } else if monotone {
                // Monotone: evaluating on upper bounds of the inputs bounds
                // the output.
                level_u.map(|u| bdd.probabilities(&u)[root as usize])
            } else {
                None
            };
            return Ok(ModuleResult {
                node: m,
                support: bdd.support(root, keys.len()),
                keys,
                method: "bdd",
                value,
                upper,
                exact: inputs_exact,
                monotone,
                gradient,
                bdd_nodes: bdd.size(root),
                nodes_created: built,
                gates,
                diagram: Some((bdd, root)),
                retained: None,
                truncated: false,
                neglected: None,
                estimators: None,
                expansions: None,
                warnings: Vec::new(),
            });
        }

        // The cut-set engine: coherent modules only.
        let formula = self.local_formula(m, is_module, constant, &level_of)?;
        let certified = level_u.is_some();
        let p_extract = level_u.unwrap_or_else(|| level_p.clone());
        let cutoffs = crate::approx::Cutoffs {
            max_order: self.settings.max_order,
            min_probability: self.settings.min_cut_probability,
            max_cut_sets: self.settings.max_cut_sets,
            max_expansions: self.settings.max_expansions,
        };
        let extracted = crate::approx::extract(&formula, &p_extract, &cutoffs);
        // The retained family is already bounded by `max_cut_sets`; the
        // node budget that sent the module here does not apply to it.
        let mut zbdd = crate::dd::Zbdd::new(usize::MAX);
        let family = zbdd.family_of(&extracted.sets)?;
        let family = zbdd.minimal(family)?;
        zbdd.root = family;
        let (pivotal, gradient) = zbdd.pivotal(family, &level_p);
        let rare_event = zbdd.rare_event(family, &level_p);
        let mcub = 1.0
            - zbdd
                .sets()
                .iter()
                .map(|set| 1.0 - set.iter().map(|l| level_p[*l as usize]).product::<f64>())
                .product::<f64>();
        let upper = certified.then(|| {
            let (pivotal_u, _) = zbdd.pivotal(family, &p_extract);
            let mcub_u = 1.0
                - zbdd
                    .sets()
                    .iter()
                    .map(|set| 1.0 - set.iter().map(|l| p_extract[*l as usize]).product::<f64>())
                    .product::<f64>();
            (pivotal_u.min(mcub_u) + extracted.neglected).min(1.0)
        });
        let mut warnings = Vec::new();
        if self.settings.engine == Engine::Auto {
            warnings.push(ModuleNote::OutgrewBdd);
        }
        if extracted.count_reached {
            warnings.push(ModuleNote::CutSetsReached);
        }
        if extracted.expansions_exhausted {
            warnings.push(ModuleNote::ExpansionsExhausted);
        }
        if extracted.neglected > pivotal {
            warnings.push(ModuleNote::NeglectedExceeds {
                neglected: extracted.neglected,
                estimate: pivotal,
            });
        }
        Ok(ModuleResult {
            warnings,
            node: m,
            support: zbdd.support(family, keys.len()),
            keys,
            method: "cut_sets",
            value: pivotal,
            upper,
            exact: false,
            monotone: true,
            gradient,
            bdd_nodes: 0,
            nodes_created: zbdd.len(),
            diagram: None,
            truncated: extracted.truncated,
            neglected: Some(extracted.neglected),
            estimators: Some((rare_event, mcub, pivotal)),
            expansions: Some(extracted.expansions),
            retained: Some(zbdd),
            gates: asked.iter().map(|&node| (node, None)).collect(),
        })
    }

    /// Module `m` as a formula over its own variables, for the cut-set
    /// engine; a negated event is refused, since the engine's bounds hold
    /// for coherent formulas only.
    fn local_formula(
        &self,
        m: u32,
        is_module: &[bool],
        constant: &HashMap<u32, bool>,
        level_of: &HashMap<Key, u32>,
    ) -> Result<crate::approx::LocalFormula, FtaError> {
        use crate::approx::{Arg, Kind};
        let mut index: HashMap<u32, u32> = HashMap::new();
        let mut gates: Vec<(Kind, Vec<Arg>)> = Vec::new();
        // Post-order, iteratively: a gate is emitted once its children are.
        let mut stack: Vec<(u32, bool)> = vec![(m, false)];
        while let Some((g, expanded)) = stack.pop() {
            if index.contains_key(&g) {
                continue;
            }
            let (op, args) = &self.graph.nodes[g as usize];
            if !expanded {
                stack.push((g, true));
                for arg in args {
                    if let Lit::Node(c) = *arg {
                        let submodule = c != m && is_module[c as usize];
                        if !submodule && !index.contains_key(&c) {
                            stack.push((c, false));
                        }
                    }
                }
                continue;
            }
            let mut local = Vec::with_capacity(args.len());
            for arg in args {
                local.push(match *arg {
                    Lit::Var(e, true) => Arg::Level(level_of[&Key::Event(e)]),
                    Lit::Var(e, false) => {
                        let name = &self.tree.events[e as usize].name;
                        return Err(if self.settings.engine == Engine::Auto {
                            FtaError::TooLarge(format!(
                                "a module holding the negated event `{name}` outgrew \
                                 max_bdd_nodes, and the cut-set engine it would fall \
                                 back to only bounds coherent formulas; raise \
                                 max_bdd_nodes"
                            ))
                        } else {
                            FtaError::NonCoherent(format!(
                                "the cut-set engine only bounds coherent formulas, \
                                 and a module holds the negated event `{name}`; use \
                                 the exact engine"
                            ))
                        });
                    }
                    Lit::Node(c) if c != m && is_module[c as usize] => match constant.get(&c) {
                        Some(v) => Arg::Const(*v),
                        None => Arg::Level(level_of[&Key::Module(c)]),
                    },
                    Lit::Node(c) => Arg::Gate(index[&c]),
                    Lit::Const(v) => Arg::Const(v),
                });
            }
            let kind = match op {
                GOp::And => Kind::And,
                GOp::Or => Kind::Or,
                GOp::AtLeast(k) => Kind::AtLeast(*k),
            };
            index.insert(g, gates.len() as u32);
            gates.push((kind, local));
        }
        Ok(crate::approx::LocalFormula {
            root: index[&m],
            gates,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn order(
        &self,
        g: u32,
        m: u32,
        is_module: &[bool],
        constant: &HashMap<u32, bool>,
        keys: &mut Vec<Key>,
        level_of: &mut HashMap<Key, u32>,
        seen: &mut std::collections::HashSet<u32>,
    ) {
        if !seen.insert(g) {
            return;
        }
        for arg in &self.graph.nodes[g as usize].1 {
            let key = match *arg {
                Lit::Var(e, _) => Key::Event(e),
                Lit::Node(c) if c != m && is_module[c as usize] => {
                    if constant.contains_key(&c) {
                        continue;
                    }
                    Key::Module(c)
                }
                Lit::Node(c) => {
                    self.order(c, m, is_module, constant, keys, level_of, seen);
                    continue;
                }
                Lit::Const(_) => continue,
            };
            if let std::collections::hash_map::Entry::Vacant(slot) = level_of.entry(key) {
                slot.insert(keys.len() as u32);
                keys.push(key);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        g: u32,
        m: u32,
        is_module: &[bool],
        constant: &HashMap<u32, bool>,
        level_of: &HashMap<Key, u32>,
        bdd: &mut Bdd,
        memo: &mut HashMap<u32, u32>,
    ) -> Result<u32, FtaError> {
        if let Some(&id) = memo.get(&g) {
            return Ok(id);
        }
        let (op, args) = &self.graph.nodes[g as usize];
        let level = |key: Key| {
            level_of.get(&key).copied().ok_or_else(|| {
                FtaError::Invalid(format!("internal: {key:?} is not a variable of its module"))
            })
        };
        let mut ids = Vec::with_capacity(args.len());
        for arg in args {
            let id = match *arg {
                Lit::Var(e, positive) => bdd.literal(level(Key::Event(e))?, positive)?,
                Lit::Node(c) if c != m && is_module[c as usize] => match constant.get(&c) {
                    Some(true) => crate::dd::ONE,
                    Some(false) => ZERO,
                    None => bdd.literal(level(Key::Module(c))?, true)?,
                },
                Lit::Node(c) => self.build(c, m, is_module, constant, level_of, bdd, memo)?,
                Lit::Const(true) => crate::dd::ONE,
                Lit::Const(false) => ZERO,
            };
            ids.push(id);
        }
        let id = match op {
            GOp::And => crate::dd::fold_balanced(bdd, true, ids)?,
            GOp::Or => crate::dd::fold_balanced(bdd, false, ids)?,
            GOp::AtLeast(k) => bdd.at_least(*k, &ids)?,
        };
        memo.insert(g, id);
        Ok(id)
    }

    /// The minimal cut sets across the modules: each module's family (its
    /// diagram's minimal solutions, or its retained cut sets), counted
    /// through the modules and expanded when the count fits the limit.
    #[allow(clippy::type_complexity)]
    fn cut_sets(
        &self,
        results: &mut [ModuleResult],
        root: u32,
        used: usize,
    ) -> Result<
        (
            Option<u128>,
            Option<Vec<Vec<usize>>>,
            Option<String>,
            (usize, usize),
        ),
        FtaError,
    > {
        let mut families: HashMap<u32, (Vec<Key>, crate::dd::Zbdd)> = HashMap::new();
        let mut counts: HashMap<u32, u128> = HashMap::new();
        let mut zbdd_nodes = 0usize;
        let mut zbdd_created = 0usize;
        for result in results.iter_mut() {
            let family = match (&result.diagram, result.retained.take()) {
                (Some((bdd, diagram_root)), _) => {
                    let budget = self
                        .settings
                        .max_bdd_nodes
                        .saturating_sub(used + zbdd_created)
                        .max(2);
                    let zbdd = match bdd.minimal_solutions(*diagram_root, budget) {
                        Ok(zbdd) => zbdd,
                        // The probability is already in hand: under `auto`
                        // a cut-set family over the budget is omitted, not
                        // a reason to lose the whole result.
                        Err(FtaError::TooLarge(why)) if self.settings.engine == Engine::Auto => {
                            return Ok((
                                None,
                                None,
                                Some(format!("the minimal cut sets were not extracted: {why}")),
                                (zbdd_nodes, zbdd_created),
                            ));
                        }
                        Err(e) => return Err(e),
                    };
                    zbdd_created += zbdd.len();
                    zbdd
                }
                (None, Some(retained)) => retained,
                (None, None) => continue,
            };
            zbdd_nodes += family.size();
            let weight: Vec<u128> = result
                .keys
                .iter()
                .map(|k| match k {
                    Key::Event(_) => 1,
                    Key::Module(n) => counts.get(n).copied().unwrap_or(0),
                })
                .collect();
            counts.insert(result.node, family.weighted_count(&weight));
            families.insert(result.node, (result.keys.clone(), family));
        }
        let count = counts.get(&root).copied().unwrap_or(0);
        let limit = self.settings.cut_set_limit;
        if count > limit as u128 {
            return Ok((
                Some(count),
                None,
                Some(format!(
                    "{count} minimal cut sets, more than the limit of {limit}; \
                     raise cut_set_limit to list them"
                )),
                (zbdd_nodes, zbdd_created),
            ));
        }
        let mut expanded: HashMap<u32, Vec<Vec<usize>>> = HashMap::new();
        for result in results.iter() {
            let Some((keys, zbdd)) = families.get(&result.node) else {
                continue;
            };
            let mut family: Vec<Vec<usize>> = Vec::new();
            for set in zbdd.sets() {
                let mut partial: Vec<Vec<usize>> = vec![Vec::new()];
                for level in set {
                    match keys[level as usize] {
                        Key::Event(e) => partial.iter_mut().for_each(|s| s.push(e as usize)),
                        Key::Module(n) => {
                            let Some(sub) = expanded.get(&n) else {
                                return Err(FtaError::Invalid(format!(
                                    "internal: submodule {n} expanded before its parent"
                                )));
                            };
                            let mut next = Vec::with_capacity(partial.len() * sub.len());
                            for s in &partial {
                                for t in sub {
                                    let mut u = s.clone();
                                    u.extend_from_slice(t);
                                    next.push(u);
                                }
                            }
                            partial = next;
                        }
                    }
                }
                family.extend(partial);
            }
            for set in &mut family {
                set.sort_unstable();
            }
            expanded.insert(result.node, family);
        }
        let mut sets = expanded.remove(&root).unwrap_or_default();
        sets.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        Ok((Some(count), Some(sets), None, (zbdd_nodes, zbdd_created)))
    }
}
