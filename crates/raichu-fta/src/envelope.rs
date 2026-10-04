//! The result envelope of a generated fault tree, `raichu.fault_tree`
//! version 1: what was explained, how the tree was generated, its
//! probability at each mission time asked for, and at the last of them
//! (the horizon) its minimal cut sets and the importance of each basic
//! event.
//!
//! The format is RAICHU's own and is versioned here, by its producer, so a
//! reader can refuse a document it does not know rather than show an empty
//! field. Its schema is documented in `docs/reference/fault-tree-format.md`.

use raichu_core::FaultTree;
use serde::{Deserialize, Serialize};

use crate::{quantify, FtaError, Quantification, QuantifySettings, Tree};

/// The `format` of a fault-tree envelope.
pub const FAULT_TREE_FORMAT: &str = "raichu.fault_tree";

/// The format version this crate writes and the highest it reads.
pub const FAULT_TREE_VERSION: u32 = 1;

/// The most mission times one envelope quantifies.
pub const MAX_MISSION_TIMES: usize = 20;

/// What the probabilities of an envelope measure.
pub const MEASURE_WITHOUT_REPAIR: &str = "probability_without_repair";

/// A generated fault tree, quantified: `raichu.fault_tree` version 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultTreeEnvelope {
    /// Always [`FAULT_TREE_FORMAT`].
    pub format: String,
    /// The format version, [`FAULT_TREE_VERSION`] when written here.
    pub version: u32,
    /// Where the result comes from.
    pub provenance: EnvelopeProvenance,
    /// What the tree explains.
    pub top: EnvelopeTop,
    /// What every probability of the envelope measures: always
    /// [`MEASURE_WITHOUT_REPAIR`], the probability that the top holds at
    /// the mission time when no repair is made, which bounds from above
    /// the probability that it has occurred once on the model with its
    /// repairs (on a coherent tree).
    pub measure: String,
    /// How the tree was generated.
    pub generation: EnvelopeGeneration,
    /// The quantification settings applied.
    pub settings: EnvelopeSettings,
    /// One entry per mission time asked for, in increasing time; the last
    /// one is the horizon.
    pub instants: Vec<InstantResult>,
    /// The quantification at the horizon, with its cut sets and the
    /// importance measures.
    pub horizon: HorizonResult,
}

impl FaultTreeEnvelope {
    /// The envelope as compact JSON.
    ///
    /// # Errors
    /// Serialization errors from `serde_json`.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// The provenance of a [`FaultTreeEnvelope`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeProvenance {
    /// The engine version that generated and quantified the tree.
    pub engine_version: String,
    /// The tree's name (the `define-fault-tree` of its OpenPSA form).
    pub tree: String,
}

/// What a tree explains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EnvelopeTop {
    /// The model's declared targets: the top is their disjunction.
    Targets {
        /// The target names.
        targets: Vec<String>,
    },
    /// An expression over the model, as given.
    Expression {
        /// The expression, in the model's expression vocabulary.
        expression: serde_json::Value,
    },
}

/// How a tree was generated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeGeneration {
    /// Whether the tree is the model's exact structure: no generation
    /// warning, so its probability without repair is the model's own.
    pub exact: bool,
    /// Why the tree may over-estimate, one sentence per transition
    /// concerned ([`raichu_core::FaultTree::warnings`]).
    pub warnings: Vec<String>,
    /// The basic events, each a transition's draw, in the tree's order.
    pub basic_events: Vec<EnvelopeEvent>,
}

/// One basic event of the tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeEvent {
    /// Its name, unique in the tree.
    pub name: String,
    /// The component the transition belongs to.
    pub component: String,
    /// The automaton, qualified.
    pub automaton: String,
    /// The transition's own name.
    pub transition: String,
    /// The state it leads to.
    pub target: String,
    /// Its law: `{"law": <name>, <parameters>}`.
    pub law: serde_json::Value,
}

/// The quantification settings an envelope was produced with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeSettings {
    /// The most nodes all decision diagrams together may hold.
    pub max_bdd_nodes: usize,
    /// The most minimal cut sets listed.
    pub cut_set_limit: usize,
    /// Whether the minimal cut sets were asked for.
    pub cut_sets: bool,
    /// `"auto"`, `"exact"` or `"cut_sets"`.
    pub engine: String,
    /// Cut-set engine: the largest order kept.
    pub max_order: Option<usize>,
    /// Cut-set engine: the smallest cut-set probability kept.
    pub min_cut_probability: f64,
    /// Cut-set engine: the most cut sets kept per module.
    pub max_cut_sets: usize,
    /// Cut-set engine: the most partial cut sets expanded per module.
    pub max_expansions: u64,
}

/// The top's probability at one mission time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstantResult {
    /// The mission time.
    pub mission_time: f64,
    /// The top-event probability without repair at that time.
    pub probability: f64,
    /// `"bdd"`, `"cut_sets"` or `"bdd+cut_sets"`.
    pub method: String,
    /// Whether the quantification is exact (no cutoff, no bound).
    pub exact: bool,
    /// A guaranteed upper bound, equal to the probability when exact;
    /// `None` when none can be certified.
    pub upper_bound: Option<f64>,
    /// The quantifier's warnings at that time.
    pub warnings: Vec<String>,
}

/// The quantification at the horizon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HorizonResult {
    /// The horizon: the last mission time.
    pub mission_time: f64,
    /// The top-event probability without repair at the horizon.
    pub probability: f64,
    /// `"bdd"`, `"cut_sets"` or `"bdd+cut_sets"`.
    pub method: String,
    /// Whether the quantification is exact.
    pub exact: bool,
    /// A guaranteed upper bound; `None` when none can be certified.
    pub upper_bound: Option<f64>,
    /// Whether the top is monotone in every basic event.
    pub coherent: bool,
    /// The number of minimal cut sets; `None` when not extracted.
    pub cut_set_count: Option<u128>,
    /// Whether no cutoff removed any cut set.
    pub cut_sets_complete: bool,
    /// Why the cut sets are not listed, when they are not.
    pub cut_sets_omitted: Option<String>,
    /// The minimal cut sets, by order then name; `None` when omitted.
    pub minimal_cut_sets: Option<Vec<EnvelopeCutSet>>,
    /// Every basic event's importance, in the tree's event order.
    pub importance: Vec<EnvelopeImportance>,
    /// The quantifier's warnings at the horizon.
    pub warnings: Vec<String>,
    /// How the numbers were produced: the quantifier's own provenance
    /// (variable order, modules, diagram sizes), carried as it is.
    pub provenance: serde_json::Value,
}

/// One minimal cut set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeCutSet {
    /// The basic events, sorted by name.
    pub events: Vec<String>,
    /// Its order: how many events.
    pub order: usize,
    /// The product of its events' probabilities at the horizon (they are
    /// independent); `None` when one of them has none.
    pub probability: Option<f64>,
}

/// The importance measures of one basic event at the horizon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeImportance {
    /// The basic event.
    pub event: String,
    /// Its probability at the horizon.
    pub probability: Option<f64>,
    /// Birnbaum's marginal importance, `P1 - P0`.
    pub birnbaum: f64,
    /// The critical importance factor, `p (P1 - P0) / P`.
    pub criticality: Option<f64>,
    /// Fussell-Vesely, `(P - P0) / P`.
    pub fussell_vesely: Option<f64>,
    /// The diagnostic importance factor, `p P1 / P`.
    pub diagnostic: Option<f64>,
    /// Risk achievement worth, `P1 / P`.
    pub risk_achievement_worth: Option<f64>,
    /// Risk reduction worth, `P / P0`.
    pub risk_reduction_worth: Option<f64>,
}

/// Quantify a generated tree at each of `mission_times` and wrap the
/// result in a [`FaultTreeEnvelope`].
///
/// `mission_times` holds 1 to [`MAX_MISSION_TIMES`] finite, non-negative,
/// strictly increasing instants; the last is the horizon, the only one
/// whose minimal cut sets and importance measures are computed (each
/// other instant is quantified without its cut sets). `settings` applies
/// to every instant, its own `mission_time` ignored.
///
/// # Errors
/// [`FtaError::Invalid`] for a bad list of mission times,
/// [`FtaError::BadMissionTime`] for a non-finite or negative one, and
/// every error [`quantify`] raises.
pub fn fault_tree_envelope(
    generated: &FaultTree,
    name: &str,
    top: EnvelopeTop,
    mission_times: &[f64],
    settings: &QuantifySettings,
) -> Result<FaultTreeEnvelope, FtaError> {
    check_mission_times(mission_times)?;
    let tree = Tree::from_generated(generated, name);
    let mut instants = Vec::with_capacity(mission_times.len());
    let (&horizon, before) = mission_times
        .split_last()
        .ok_or_else(|| FtaError::Invalid("no mission time".to_owned()))?;
    for &time in before {
        let result = quantify(
            &tree,
            &QuantifySettings {
                mission_time: Some(time),
                cut_sets: false,
                ..settings.clone()
            },
        )?;
        instants.push(instant(time, &result));
    }
    let result = quantify(
        &tree,
        &QuantifySettings {
            mission_time: Some(horizon),
            ..settings.clone()
        },
    )?;
    instants.push(instant(horizon, &result));
    let names: Vec<&str> = tree.events.iter().map(|e| e.name.as_str()).collect();
    let probability_of = |event: usize| result.importance.get(event).and_then(|i| i.probability);
    let minimal_cut_sets = result.minimal_cut_sets.as_ref().map(|sets| {
        let mut sets: Vec<EnvelopeCutSet> = sets
            .iter()
            .map(|set| {
                let mut events: Vec<String> = set.iter().map(|&e| names[e].to_owned()).collect();
                events.sort();
                let probability = set
                    .iter()
                    .map(|&e| probability_of(e))
                    .try_fold(1.0, |acc, p| p.map(|p| acc * p));
                EnvelopeCutSet {
                    order: events.len(),
                    events,
                    probability,
                }
            })
            .collect();
        sets.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.events.cmp(&b.events)));
        sets
    });
    let importance = result
        .importance
        .iter()
        .map(|i| EnvelopeImportance {
            event: names[i.event].to_owned(),
            probability: i.probability,
            birnbaum: i.birnbaum,
            criticality: i.criticality,
            fussell_vesely: i.fussell_vesely,
            diagnostic: i.diagnostic,
            risk_achievement_worth: i.risk_achievement_worth,
            risk_reduction_worth: i.risk_reduction_worth,
        })
        .collect();
    let provenance = serde_json::to_value(&result.provenance)
        .map_err(|e| FtaError::Invalid(format!("the quantifier's provenance: {e}")))?;
    let basic_events = generated
        .basic_events
        .iter()
        .map(|e| {
            Ok(EnvelopeEvent {
                name: e.name.clone(),
                component: e.component.clone(),
                automaton: e.automaton.clone(),
                transition: e.transition.clone(),
                target: e.target.clone(),
                law: serde_json::to_value(&e.law)
                    .map_err(|err| FtaError::Invalid(format!("the law of `{}`: {err}", e.name)))?,
            })
        })
        .collect::<Result<Vec<_>, FtaError>>()?;
    Ok(FaultTreeEnvelope {
        format: FAULT_TREE_FORMAT.to_owned(),
        version: FAULT_TREE_VERSION,
        provenance: EnvelopeProvenance {
            engine_version: env!("CARGO_PKG_VERSION").to_owned(),
            tree: name.to_owned(),
        },
        top,
        measure: MEASURE_WITHOUT_REPAIR.to_owned(),
        generation: EnvelopeGeneration {
            exact: generated.warnings.is_empty(),
            warnings: generated.warnings.clone(),
            basic_events,
        },
        settings: EnvelopeSettings {
            max_bdd_nodes: settings.max_bdd_nodes,
            cut_set_limit: settings.cut_set_limit,
            cut_sets: settings.cut_sets,
            engine: engine_name(settings.engine).to_owned(),
            max_order: settings.max_order,
            min_cut_probability: settings.min_cut_probability,
            max_cut_sets: settings.max_cut_sets,
            max_expansions: settings.max_expansions,
        },
        instants,
        horizon: HorizonResult {
            mission_time: horizon,
            probability: result.probability,
            method: result.method.to_owned(),
            exact: result.exact,
            upper_bound: result.upper_bound,
            coherent: result.coherent,
            cut_set_count: result.cut_set_count,
            cut_sets_complete: result.cut_sets_complete,
            cut_sets_omitted: result.cut_sets_omitted.clone(),
            minimal_cut_sets,
            importance,
            warnings: result.warnings.clone(),
            provenance,
        },
    })
}

fn engine_name(engine: crate::Engine) -> &'static str {
    match engine {
        crate::Engine::Auto => "auto",
        crate::Engine::Exact => "exact",
        crate::Engine::CutSets => "cut_sets",
    }
}

fn instant(time: f64, result: &Quantification) -> InstantResult {
    InstantResult {
        mission_time: time,
        probability: result.probability,
        method: result.method.to_owned(),
        exact: result.exact,
        upper_bound: result.upper_bound,
        warnings: result.warnings.clone(),
    }
}

fn check_mission_times(times: &[f64]) -> Result<(), FtaError> {
    if times.is_empty() {
        return Err(FtaError::Invalid(
            "an envelope needs at least one mission time".to_owned(),
        ));
    }
    if times.len() > MAX_MISSION_TIMES {
        return Err(FtaError::Invalid(format!(
            "{} mission times asked for, at most {MAX_MISSION_TIMES} are quantified",
            times.len()
        )));
    }
    for &time in times {
        if !time.is_finite() || time < 0.0 {
            return Err(FtaError::BadMissionTime(time));
        }
    }
    if times.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(FtaError::Invalid(format!(
            "the mission times {times:?} are not strictly increasing"
        )));
    }
    Ok(())
}

/// Why a document is not a fault-tree envelope this crate reads.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ReadFaultTreeError {
    /// The document is not valid JSON, or lacks a member of the format.
    #[error("invalid fault-tree envelope JSON: {0}")]
    Json(String),
    /// The document declares another format (or none).
    #[error("not a fault-tree envelope: format {0:?}, expected `raichu.fault_tree`")]
    Format(Option<String>),
    /// The document declares a version this engine does not read.
    #[error(
        "fault-tree envelope version {0:?} is not readable by this engine (reads up to version {max})",
        max = FAULT_TREE_VERSION
    )]
    Version(Option<u32>),
}

/// Read a fault-tree envelope from its JSON document, refusing another
/// `format` and a `version` above [`FAULT_TREE_VERSION`]. The format and
/// version are read first, so a document of another format is refused by
/// name rather than by the first member it happens to lack.
///
/// This is the shape check of the format. Its numbers go through
/// `serde_json`'s default float parser, which may land one unit in the
/// last place away from the written value; a reader that needs the bits
/// (pyraichu's, for one) parses the text with a correctly rounding parser
/// once this check has passed.
///
/// # Errors
/// [`ReadFaultTreeError`].
pub fn read_fault_tree_envelope(json: &str) -> Result<FaultTreeEnvelope, ReadFaultTreeError> {
    #[derive(Deserialize)]
    struct Head {
        format: Option<serde_json::Value>,
        version: Option<serde_json::Value>,
    }
    let head: Head =
        serde_json::from_str(json).map_err(|e| ReadFaultTreeError::Json(e.to_string()))?;
    let format = head
        .format
        .as_ref()
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    if format.as_deref() != Some(FAULT_TREE_FORMAT) {
        return Err(ReadFaultTreeError::Format(format));
    }
    let version = head
        .version
        .as_ref()
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok());
    match version {
        Some(version) if (1..=FAULT_TREE_VERSION).contains(&version) => {}
        other => return Err(ReadFaultTreeError::Version(other)),
    }
    // Parsed from the text, not from a `serde_json::Value`, which cannot
    // hold a cut-set count past `u64`.
    serde_json::from_str(json).map_err(|e| ReadFaultTreeError::Json(e.to_string()))
}
