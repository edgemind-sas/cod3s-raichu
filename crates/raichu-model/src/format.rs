//! The format envelope: revision, feature gate, loading and sealing a model document.

#[cfg(doc)]
use crate::Component;
#[cfg(doc)]
use crate::ModelError;
#[cfg(doc)]
use crate::Port;
#[cfg(doc)]
use crate::Transition;
#[cfg(doc)]
use crate::PORT;
use crate::{Model, TransitionKind};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

/// Top-level key of the **format envelope** of a serialized model
/// document (see [`FormatHeader`]).
pub const ENVELOPE_KEY: &str = "raichu_model";

/// Revision of the envelope itself, not of the engine: it changes only
/// if the envelope's own shape changes, which is why it is a small
/// integer and not the crate version.
pub const FORMAT_REVISION: u32 = 1;

/// A named construct of the model format that an engine either
/// implements or does not.
///
/// The registry opens with this enum: everything the format could
/// express *before* it is the **baseline** and carries no name, so
/// every model written until now needs no feature at all. From here on,
/// a construct an older engine would silently ignore gets a variant,
/// and a model using it must say so in its [`FormatHeader`].
///
/// The list a document carries is never taken on trust: [`Model::from_json`]
/// derives the truth from the parsed content ([`Model::required_features`])
/// and refuses a document whose declaration does not cover it. A
/// hand-written model therefore cannot use a construct and stay silent
/// about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Feature {
    /// Model-level [`Model::evaluation_order`]: the explicit equations
    /// are swept in a declared order instead of the positional one. An
    /// engine that ignored the field would sweep a different order and
    /// return a different number without a word.
    EvaluationOrder,
    /// Component-level [`Component::allocations`]: the conservative
    /// distribution operator. An engine that ignored the field would
    /// leave every allocated quantity at its channel default, and report
    /// a network that delivers nothing without a word.
    Allocation,
    /// Model-level [`Model::unbounded_rate`]: the magnitude an authoring
    /// layer writes for "no ceiling", which no integrated rate may reach.
    /// An engine that ignored the field would integrate that magnitude as
    /// a rate and report a stock of -1e19 without a word.
    UnboundedRate,
    /// Transition-level [`Transition::effects`]: assignments made once, on
    /// the firing edge. An engine that ignored the field would never make
    /// the write, and a latched detection would read its initial value for
    /// the whole trajectory without a word.
    TransitionEffects,
    /// Model-level [`Model::fmu_units`]: imported native FMI co-simulation
    /// units. An older engine must refuse instead of ignoring their physics.
    Fmi,
    /// Model-level mixed-integer programs solved at discrete fixpoints.
    MixedIntegerProgram,
    /// Transition-level [`TransitionKind::Observation`]: observers fire first
    /// among transitions due at the same date. An engine that ignored it
    /// would fire them in positional order and miss every state the model
    /// reaches and leaves within one instant, without a word.
    ObserverPriority,
    /// Model-level [`Model::interface_connections`]. An engine that
    /// ignored the field would leave the ports they connect unfed, and
    /// run a model whose components never hear from each other without
    /// a word.
    InterfaceConnections,
    /// Transition-level [`Transition::monitored_states`]: a monitored
    /// transition records the entry into some of its targets only. An
    /// engine that ignored the field would record a lost Bernoulli draw as
    /// an event of the mission, and the sequence and importance analyses
    /// would cancel or invent failures without a word.
    MonitoredStates,
    /// Transition-level [`Transition::cycle_cancellation`] other than
    /// `auto`: the analyst forces or forbids the cancellation of a cycle
    /// group. An engine that ignored it would cancel a group declared
    /// `never`, and the minimal sequences would lose the failures the
    /// declaration was there to keep, without a word.
    CycleCancellation,
    /// Port-level [`Port::equations`] and `port_agg` over an out port:
    /// per-connection channel equations. An engine that ignored them would
    /// leave every channel they write at its default, and a flow network
    /// would deliver nothing without a word.
    PortEquations,
    /// Edge effects written through a port ([`PORT`]). An engine that
    /// ignored them would refuse the reserved target or write nothing, and
    /// a unit whose repair refills its buffer would leave it empty without
    /// a word.
    PortEffects,
}

impl Feature {
    /// Every feature this engine implements, in declaration order.
    pub const ALL: &'static [Feature] = &[
        Feature::EvaluationOrder,
        Feature::Allocation,
        Feature::UnboundedRate,
        Feature::TransitionEffects,
        Feature::Fmi,
        Feature::MixedIntegerProgram,
        Feature::ObserverPriority,
        Feature::InterfaceConnections,
        Feature::MonitoredStates,
        Feature::CycleCancellation,
        Feature::PortEquations,
        Feature::PortEffects,
    ];

    /// Serialized name of the feature.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Feature::EvaluationOrder => "evaluation_order",
            Feature::Allocation => "allocation",
            Feature::UnboundedRate => "unbounded_rate",
            Feature::TransitionEffects => "transition_effects",
            Feature::Fmi => "fmi",
            Feature::MixedIntegerProgram => "mixed_integer_program",
            Feature::ObserverPriority => "observer_priority",
            Feature::InterfaceConnections => "interface_connections",
            Feature::MonitoredStates => "monitored_states",
            Feature::CycleCancellation => "cycle_cancellation",
            Feature::PortEquations => "port_equations",
            Feature::PortEffects => "port_effects",
        }
    }

    /// The feature this name designates, or `None` when this engine
    /// does not implement it (a newer construct, or a typo: both are
    /// refused, and refusing is the point).
    #[must_use]
    pub fn parse(name: &str) -> Option<Feature> {
        Feature::ALL.iter().copied().find(|f| f.name() == name)
    }

    /// Comma-separated list of the implemented feature names, for
    /// diagnostics.
    #[must_use]
    pub fn known() -> String {
        Feature::ALL
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Header of the **format envelope**: the mandatory, reader-visible
/// preamble of a serialized model document.
///
/// # Why a wrapper and not a field
///
/// The model schema accepts unknown fields (no `deny_unknown_fields`
/// anywhere) and carried no version. An optional `requires` field
/// alongside `name` and `components` would therefore be *invisible* to
/// an engine that predates it: that engine would parse the document,
/// ignore the field **and the new construct it announces**, and return a
/// different number instead of a refusal. Silence is exactly the
/// failure to avoid.
///
/// The envelope instead **displaces** the fields a legacy reader
/// requires: the model body moves under `model`, so a reader expecting
/// `name` and `components` at the top level fails on a missing field
/// rather than succeeding on a misread. The refusal is structural, so
/// it works even against engines released before the envelope existed.
/// Readers from this version on know the shape, so *their* refusal is
/// the precise, named [`LoadError::UnsupportedFeature`].
///
/// A serialized document looks like:
///
/// ```json
/// {
///   "raichu_model": {"format": 1, "requires": ["evaluation_order"]},
///   "model": {"name": "…", "components": []}
/// }
/// ```
///
/// The bare (envelope-less) body stays readable, so the whole existing
/// corpus loads unchanged; it declares no feature, so it may only use
/// baseline constructs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormatHeader {
    /// Envelope revision ([`FORMAT_REVISION`]).
    pub format: u32,
    /// Names of the features the document requires. Derived from the
    /// body by the writer, verified against the body by the reader.
    #[serde(default)]
    pub requires: Vec<String>,
}

/// A serialized model document: envelope header plus body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ModelDocument {
    #[serde(rename = "raichu_model")]
    header: FormatHeader,
    model: Model,
}

/// Borrowed counterpart of [`ModelDocument`] for writing (serializing a
/// model must not clone it).
#[derive(Debug, Serialize)]
struct ModelDocumentRef<'m> {
    #[serde(rename = "raichu_model")]
    header: FormatHeader,
    model: &'m Model,
}

/// Failure to load a serialized model **document**: a malformed
/// document, or one this engine must refuse rather than misread.
///
/// Distinct from [`ModelError`], which reports an unsound *model*: this
/// one reports a document the engine cannot honour as written.
#[derive(Debug, Error)]
pub enum LoadError {
    /// The document is not valid JSON, or does not match the schema.
    #[error("malformed model document: {0}")]
    Json(#[from] serde_json::Error),
    /// The document declares a feature this engine does not implement:
    /// it was written by a newer engine (or names a construct that does
    /// not exist). Refused, never loaded with the construct ignored.
    #[error(
        "model requires the `{feature}` feature, which this engine \
         (RAICHU {version}) does not implement; implemented: {known}"
    )]
    UnsupportedFeature {
        /// The feature name the document declares.
        feature: String,
        /// Version of the engine that refuses it.
        version: &'static str,
        /// The implemented feature names.
        known: String,
    },
    /// The document *uses* a non-baseline construct without declaring
    /// it. Refused so the declaration can never lag the content: an
    /// undeclared construct would be ignored in silence by any older
    /// engine reading the same file.
    #[error(
        "model uses the `{feature}` construct but does not declare it: \
         wrap the model body in the `{envelope}` envelope and list \
         `{feature}` under `requires`, so an engine that predates the \
         construct refuses the model instead of ignoring it"
    )]
    FeatureNotDeclared {
        /// The construct's feature name.
        feature: &'static str,
        /// The envelope key to add ([`ENVELOPE_KEY`]).
        envelope: &'static str,
    },
}

impl Model {
    /// Load a model from its JSON representation, in either accepted
    /// shape: the **format envelope** (see [`FormatHeader`]) or the
    /// bare, envelope-less body kept readable for the existing corpus.
    ///
    /// The feature gate runs here: the document's declared features
    /// must all be implemented by this engine, and must cover every
    /// non-baseline construct the body actually contains
    /// ([`Model::required_features`]). A bare body declares nothing, so
    /// it may only use baseline constructs.
    ///
    /// This checks the *document*; call [`Model::validate`] afterwards
    /// for the model's structural soundness.
    pub fn from_json(json: &str) -> Result<Self, LoadError> {
        let (declared, model) = Self::parse_document(json)?;
        model.check_features(declared.as_deref())?;
        Ok(model)
    }

    /// Serialize the model as a complete document: the mandatory format
    /// envelope plus the body. The envelope's `requires` list is
    /// **derived** from the model ([`Model::required_features`]), never
    /// hand-written, so it cannot lag what the body contains.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&ModelDocumentRef {
            header: self.format_header(),
            model: self,
        })
    }

    /// Rewrite an authored document (bare or already enveloped) as a
    /// sealed one, deriving the feature list from the body.
    ///
    /// This is the **writer** side of the gate, for an authoring layer
    /// that produces a body and must not compose the envelope itself.
    /// It tolerates a body that uses a construct without declaring it:
    /// that is precisely what this call exists to fix. It does *not*
    /// tolerate a declaration naming a feature this engine does not
    /// implement, which sealing cannot fix and re-deriving would
    /// silently drop: refused as [`Model::from_json`] refuses it.
    pub fn seal_json(document_json: &str) -> Result<String, LoadError> {
        let (declared, model) = Self::parse_document(document_json)?;
        Self::declared_features(declared.as_deref())?;
        Ok(model.to_json()?)
    }

    /// The envelope header this model would be written with.
    #[must_use]
    pub fn format_header(&self) -> FormatHeader {
        FormatHeader {
            format: FORMAT_REVISION,
            requires: self
                .required_features()
                .into_iter()
                .map(|feature| feature.name().to_owned())
                .collect(),
        }
    }

    /// The features this model **actually** requires, derived from the
    /// constructs it contains.
    ///
    /// This is the authority the declared list is checked against: a
    /// declaration is a mirror the reader verifies, never a claim it
    /// trusts, so a hand-written model cannot use a construct and omit
    /// it from the envelope.
    #[must_use]
    pub fn required_features(&self) -> BTreeSet<Feature> {
        let mut features = BTreeSet::new();
        if !self.fmu_units.is_empty() {
            features.insert(Feature::Fmi);
        }
        if !self.programs.is_empty() {
            features.insert(Feature::MixedIntegerProgram);
        }
        if !self.interface_connections.is_empty() {
            features.insert(Feature::InterfaceConnections);
        }
        if self.has_port_constructs() {
            features.insert(Feature::PortEquations);
        }
        if self.has_port_effects() {
            features.insert(Feature::PortEffects);
        }
        if self.evaluation_order.is_some() {
            features.insert(Feature::EvaluationOrder);
        }
        if self.unbounded_rate.is_some() {
            features.insert(Feature::UnboundedRate);
        }
        if self.components.iter().any(|component| {
            component
                .automata
                .iter()
                .any(|automaton| automaton.transitions.iter().any(|t| !t.effects.is_empty()))
        }) {
            features.insert(Feature::TransitionEffects);
        }
        if self.components.iter().any(|component| {
            component.automata.iter().any(|automaton| {
                automaton
                    .transitions
                    .iter()
                    .any(|t| t.kind == Some(TransitionKind::Observation))
            })
        }) {
            features.insert(Feature::ObserverPriority);
        }
        if self.components.iter().any(|component| {
            component.automata.iter().any(|automaton| {
                automaton
                    .transitions
                    .iter()
                    .any(|t| t.monitored_states.is_some())
            })
        }) {
            features.insert(Feature::MonitoredStates);
        }
        if self.components.iter().any(|component| {
            component.automata.iter().any(|automaton| {
                automaton
                    .transitions
                    .iter()
                    .any(|t| !t.cycle_cancellation.is_auto())
            })
        }) {
            features.insert(Feature::CycleCancellation);
        }
        if self
            .components
            .iter()
            .any(|component| !component.allocations.is_empty())
        {
            features.insert(Feature::Allocation);
        }
        features
    }

    /// Split a document into its declared feature list (`None` for a
    /// bare, envelope-less body) and its model body. No feature gate:
    /// the callers apply the part of it they need.
    fn parse_document(json: &str) -> Result<(Option<Vec<String>>, Self), LoadError> {
        let value: serde_json::Value = serde_json::from_str(json)?;
        if value.get(ENVELOPE_KEY).is_some() {
            let document: ModelDocument = serde_json::from_value(value)?;
            Ok((Some(document.header.requires), document.model))
        } else {
            Ok((None, serde_json::from_value(value)?))
        }
    }

    /// Resolve a declared feature list, refusing any name this engine
    /// does not implement. `None` is a bare body: it declares nothing.
    fn declared_features(declared: Option<&[String]>) -> Result<BTreeSet<Feature>, LoadError> {
        let mut allowed: BTreeSet<Feature> = BTreeSet::new();
        for name in declared.unwrap_or(&[]) {
            let Some(feature) = Feature::parse(name) else {
                return Err(LoadError::UnsupportedFeature {
                    feature: name.clone(),
                    version: env!("CARGO_PKG_VERSION"),
                    known: Feature::known(),
                });
            };
            allowed.insert(feature);
        }
        Ok(allowed)
    }

    /// Feature gate: every declared name must be implemented, and the
    /// declaration must cover every non-baseline construct the body
    /// contains. `declared` is `None` for a bare body, which declares
    /// nothing and may therefore only use baseline constructs.
    fn check_features(&self, declared: Option<&[String]>) -> Result<(), LoadError> {
        let allowed = Self::declared_features(declared)?;
        for feature in self.required_features() {
            if !allowed.contains(&feature) {
                return Err(LoadError::FeatureNotDeclared {
                    feature: feature.name(),
                    envelope: ENVELOPE_KEY,
                });
            }
        }
        Ok(())
    }
}
