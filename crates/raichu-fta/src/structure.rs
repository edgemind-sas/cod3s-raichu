//! Versioned structural fault-tree output, without numerical quantification.

use raichu_core::{FaultTree, FtNode};
use serde::{Deserialize, Serialize};

use crate::{EnvelopeTop, FtaError};
use sha2::{Digest, Sha256};

/// The format of a generated structural result.
pub const FAULT_TREE_STRUCTURE_FORMAT: &str = "raichu.fault_tree.structure";
/// The structural format version written and read by this crate.
pub const FAULT_TREE_STRUCTURE_VERSION: u32 = 1;

/// A generated tree and optionally its structural minimal cut sets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultTreeStructure {
    /// Always [`FAULT_TREE_STRUCTURE_FORMAT`].
    pub format: String,
    /// Always [`FAULT_TREE_STRUCTURE_VERSION`] when written here.
    pub version: u32,
    /// Engine version and tree name.
    pub provenance: StructureProvenance,
    /// The expression or declared targets explained.
    pub top: EnvelopeTop,
    /// Gates, basic events and generation warnings.
    pub tree: FaultTree,
    /// Extraction settings applied.
    pub settings: StructureSettings,
    /// Minimal sets, by size then name; `None` when not requested.
    pub minimal_cut_sets: Option<Vec<Vec<String>>>,
    /// `"not requested"` when omitted, otherwise `None`.
    pub cut_sets_omitted: Option<String>,
}

/// Applicable provenance for deterministic structural analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructureProvenance {
    /// The engine version producing this result.
    pub engine_version: String,
    /// The tree name.
    pub tree: String,
    /// What source and generation configuration are known.
    pub source: StructureSource,
    /// Result production time in milliseconds since the Unix epoch (UTC).
    pub generated_at_unix_ms: u64,
    /// SHA-256 of the exact Cargo lockfile embedded at compilation.
    pub dependency_lock_hash: String,
}

/// Explicit source identity: never infer a source model from a supplied tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StructureSource {
    /// A model with known identity and the actual generation configuration.
    Model {
        /// Canonical sealed-model identity, as returned by `model_content_hash`.
        model_hash: String,
        /// Applied structural generation configuration.
        generation: StructureGeneration,
    },
    /// The caller supplied a tree; source model and generation configuration
    /// are unknown. The producer still records packaging time and dependencies.
    SuppliedTree,
}

/// Configuration used to generate a structure from a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructureGeneration {
    /// Effective node budget, including the resolved default.
    pub max_nodes: usize,
    /// Attribute profiles as qualified-name and typed-value pairs.
    pub profile: Vec<(String, serde_json::Value)>,
}

/// Settings for structural cut-set extraction, without probabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructureSettings {
    /// Whether extraction was requested.
    pub cut_sets: bool,
    /// The declared expansion budget, unused when extraction is omitted.
    pub cut_set_limit: usize,
}

impl FaultTreeStructure {
    /// Serialize this result as compact JSON.
    ///
    /// # Errors
    /// Serialization errors from `serde_json`.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Produce a structural result, extracting cuts only when requested.
///
/// # Errors
/// [`FtaError::Invalid`] if extraction exceeds its declared budget.
pub fn fault_tree_structure(
    generated: &FaultTree,
    name: &str,
    top: EnvelopeTop,
    settings: &StructureSettings,
    source: StructureSource,
) -> Result<FaultTreeStructure, FtaError> {
    validate_tree(generated).map_err(|e| FtaError::Invalid(e.to_owned()))?;
    let minimal_cut_sets = if settings.cut_sets {
        let cuts = generated
            .minimal_cut_sets(settings.cut_set_limit)
            .map_err(|e| FtaError::Invalid(e.to_string()))?;
        let mut named: Vec<Vec<String>> = cuts
            .iter()
            .map(|cut| {
                let mut names: Vec<String> = cut
                    .iter()
                    .map(|&event| generated.basic_events[event].name.clone())
                    .collect();
                names.sort();
                names
            })
            .collect();
        named.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        Some(named)
    } else {
        None
    };
    Ok(FaultTreeStructure {
        format: FAULT_TREE_STRUCTURE_FORMAT.to_owned(),
        version: FAULT_TREE_STRUCTURE_VERSION,
        provenance: StructureProvenance {
            engine_version: env!("CARGO_PKG_VERSION").to_owned(),
            tree: name.to_owned(),
            source,
            generated_at_unix_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| FtaError::Invalid(format!("structural production timestamp: {e}")))?
                .as_millis()
                .try_into()
                .map_err(|_| {
                    FtaError::Invalid("structural production timestamp overflow".to_owned())
                })?,
            dependency_lock_hash: format!(
                "sha256:{:x}",
                Sha256::digest(include_bytes!("../../../Cargo.lock"))
            ),
        },
        top,
        tree: generated.clone(),
        settings: settings.clone(),
        minimal_cut_sets,
        cut_sets_omitted: (!settings.cut_sets).then(|| "not requested".to_owned()),
    })
}

/// Why a structural document cannot be read.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ReadFaultTreeStructureError {
    /// Invalid JSON, missing fields or inconsistent structure.
    #[error("invalid fault-tree structure JSON: {0}")]
    Json(String),
    /// An unknown format.
    #[error("fault-tree structure format {0:?}, expected `raichu.fault_tree.structure`")]
    Format(Option<String>),
    /// An unsupported version.
    #[error("fault-tree structure version {0:?} is not readable (reads version 1)")]
    Version(Option<u32>),
}

/// Read the producer's structural output, checking format, version and references.
///
/// # Errors
/// [`ReadFaultTreeStructureError`] for unknown or malformed documents.
pub fn read_fault_tree_structure(
    json: &str,
) -> Result<FaultTreeStructure, ReadFaultTreeStructureError> {
    crate::with_large_stack(|| Ok(read_structure(json)))
        .map_err(|e| ReadFaultTreeStructureError::Json(e.to_string()))?
}

fn read_structure(json: &str) -> Result<FaultTreeStructure, ReadFaultTreeStructureError> {
    #[derive(Deserialize)]
    struct Header {
        format: Option<serde_json::Value>,
        version: Option<serde_json::Value>,
    }
    let mut decoder = serde_json::Deserializer::from_str(json);
    decoder.disable_recursion_limit();
    let header = Header::deserialize(&mut decoder)
        .map_err(|e| ReadFaultTreeStructureError::Json(e.to_string()))?;
    decoder
        .end()
        .map_err(|e| ReadFaultTreeStructureError::Json(e.to_string()))?;
    let format = header
        .format
        .as_ref()
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    if format.as_deref() != Some(FAULT_TREE_STRUCTURE_FORMAT) {
        return Err(ReadFaultTreeStructureError::Format(format));
    }
    let version = header
        .version
        .as_ref()
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok());
    if version != Some(FAULT_TREE_STRUCTURE_VERSION) {
        return Err(ReadFaultTreeStructureError::Version(version));
    }
    let mut decoder = serde_json::Deserializer::from_str(json);
    decoder.disable_recursion_limit();
    let result = FaultTreeStructure::deserialize(&mut decoder)
        .map_err(|e| ReadFaultTreeStructureError::Json(e.to_string()))?;
    decoder
        .end()
        .map_err(|e| ReadFaultTreeStructureError::Json(e.to_string()))?;
    let invalid = |message: &str| ReadFaultTreeStructureError::Json(message.to_owned());
    validate_tree(&result.tree).map_err(invalid)?;
    let names: std::collections::HashSet<&str> = result
        .tree
        .basic_events
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    if result.settings.cut_sets != result.minimal_cut_sets.is_some()
        || result.cut_sets_omitted.as_deref()
            != if result.settings.cut_sets {
                None
            } else {
                Some("not requested")
            }
    {
        return Err(invalid("inconsistent cut-set omission"));
    }
    if let Some(cuts) = &result.minimal_cut_sets {
        for cut in cuts {
            if cut.iter().any(|name| !names.contains(name.as_str())) {
                return Err(invalid("unknown basic-event name in a cut set"));
            }
        }
    }
    Ok(result)
}

fn validate_tree(tree: &FaultTree) -> Result<(), &'static str> {
    let names: std::collections::HashSet<&str> =
        tree.basic_events.iter().map(|e| e.name.as_str()).collect();
    if names.len() != tree.basic_events.len() {
        return Err("duplicate basic-event names");
    }
    let mut pending = vec![&tree.top];
    while let Some(node) = pending.pop() {
        match node {
            FtNode::Basic { event } if *event >= tree.basic_events.len() => {
                return Err("unknown basic-event index")
            }
            FtNode::Constant { .. } => return Err("degenerate constant tree"),
            FtNode::Gate { children, .. } => pending.extend(children),
            _ => {}
        }
    }
    Ok(())
}
