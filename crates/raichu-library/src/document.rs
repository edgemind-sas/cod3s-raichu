//! The three documents: library, system and study.
//!
//! Each document is an envelope (`raichu_library`, `raichu_system` or
//! `raichu_study`, each `{ "format": 1 }`) beside its body. Every struct
//! here refuses unknown fields and accepts a free `metadata` value the
//! expander never reads (an editor's layout, an author's comments).
//!
//! Parts written in core constructs or in the templating grammar (type
//! bodies, mode catalogues, laws, conditions, indicators) are held as raw
//! JSON here and typed strictly once their templates are resolved.

use crate::error::{DocumentKind, ExpandError};
use crate::names::check_name;
use raichu_model::{InterruptionPolicy, Target};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Envelope key of a library document.
pub const LIBRARY_KEY: &str = "raichu_library";
/// Envelope key of a system document.
pub const SYSTEM_KEY: &str = "raichu_system";
/// Envelope key of a study document.
pub const STUDY_KEY: &str = "raichu_study";
/// Revision of the three envelopes; a reader refuses any other.
pub const FORMAT_REVISION: u32 = 1;

/// The envelope header shared by the three documents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    /// Envelope revision ([`FORMAT_REVISION`]).
    pub format: u32,
}

/// The kind of a declared parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ParamKind {
    /// A float (an integer literal is accepted).
    Number,
    /// An integer.
    Integer,
    /// A boolean.
    Bool,
    /// A string (names, interpolated into identifiers).
    String,
    /// A probability law, written as a core law object
    /// (`{"distrib": "exp", "rate": 0.1}`).
    Law,
    /// A list whose items repeat a part of the type (`$for_each`).
    List,
    /// One value among `values`, selecting a variant (`$choose`).
    Enum,
}

/// The kind of the items of a list parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ItemKind {
    /// Float items.
    Number,
    /// Integer items.
    Integer,
    /// Boolean items.
    Bool,
    /// String items.
    String,
    /// Items that are themselves lists of strings (a production condition:
    /// a disjunction of conjunctions, each a list of attribute names).
    StringList,
    /// Items that are JSON objects, inserted as written where the list is
    /// substituted with `$param` (a split share keyed by its consumer:
    /// `{"to": {"component", "port"}, "value"}`); the core reader then
    /// checks them strictly at the place they land.
    Object,
}

/// A parameter of a type or a fragment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Parameter {
    /// Parameter name, unique in its type or fragment.
    pub name: String,
    /// Parameter kind.
    pub kind: ParamKind,
    /// Default value; a parameter without one is required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    /// The allowed values of an `enum` parameter.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<String>,
    /// The item kind of a `list` parameter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<ItemKind>,
    /// Free text for the reader of the library.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Free editor data, never read by the expander.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// The application of a fragment inside a type or another fragment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Use {
    /// Fragment name: local (`gate`) or from an imported library
    /// (`std.bool_flow_out`).
    pub fragment: String,
    /// Parameter values of the fragment; a value may itself be a template
    /// of the enclosing type (`{"$param": "name"}`, `"${name}_in"`).
    #[serde(default, rename = "with", skip_serializing_if = "Map::is_empty")]
    pub with: Map<String, Value>,
}

/// A named, parameterised piece of behaviour composed into types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Fragment {
    /// Fragment name, unique in its library.
    pub name: String,
    /// Parameters.
    #[serde(default)]
    pub parameters: Vec<Parameter>,
    /// Fragments this fragment applies.
    #[serde(default)]
    pub uses: Vec<Use>,
    /// Body in core constructs, templated (attributes, ports, interfaces,
    /// automata, sensitive_functions, equations, allocations).
    #[serde(default = "empty_object")]
    #[cfg_attr(feature = "schema", schemars(with = "crate::instantiate::BodyParts"))]
    pub body: Value,
    /// Attributes a mode may target, with their combination rule.
    #[serde(default = "empty_array")]
    #[cfg_attr(
        feature = "schema",
        schemars(with = "Vec<crate::instantiate::Targetable>")
    )]
    pub targetable: Value,
    /// Free text for the reader of the library.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Free editor data, never read by the expander.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// A component type: parameters, behaviour and mode catalogue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TypeDef {
    /// Type name, unique in its library.
    pub name: String,
    /// Parameters.
    #[serde(default)]
    pub parameters: Vec<Parameter>,
    /// Fragments this type applies.
    #[serde(default)]
    pub uses: Vec<Use>,
    /// Own body in core constructs, templated.
    #[serde(default = "empty_object")]
    #[cfg_attr(feature = "schema", schemars(with = "crate::instantiate::BodyParts"))]
    pub body: Value,
    /// Attributes a mode may target, with their combination rule.
    #[serde(default = "empty_array")]
    #[cfg_attr(
        feature = "schema",
        schemars(with = "Vec<crate::instantiate::Targetable>")
    )]
    pub targetable: Value,
    /// Mode catalogue, templated.
    #[serde(default = "empty_array")]
    #[cfg_attr(
        feature = "schema",
        schemars(with = "Vec<crate::instantiate::CatalogueMode>")
    )]
    pub modes: Value,
    /// Free text for the reader of the library.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Free editor data, never read by the expander.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// A library of types and fragments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Library {
    /// Library name (no `.`: it prefixes imported fragment names).
    pub name: String,
    /// Library version, matched exactly by the systems that use it.
    pub version: String,
    /// Imported libraries, by name (`std`).
    #[serde(default)]
    pub imports: Vec<String>,
    /// Fragments.
    #[serde(default)]
    pub fragments: Vec<Fragment>,
    /// Types.
    #[serde(default)]
    pub types: Vec<TypeDef>,
    /// Free text for the reader of the library.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Free editor data, never read by the expander.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// The library a system instantiates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct LibraryRef {
    /// Library name.
    pub name: String,
    /// Exact library version.
    pub version: String,
}

/// An instance of a type in a system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Instance {
    /// Instance name, unique in the system; it becomes the component name.
    pub name: String,
    /// Type name.
    #[serde(rename = "type")]
    pub type_name: String,
    /// Parameter values; omitted parameters take their type default.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub parameters: Map<String, Value>,
    /// Free editor data, never read by the expander.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// A connection between two instance ports, written `Instance.port`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SystemConnection {
    /// Out port, `Instance.port`.
    pub from: String,
    /// In port, `Instance.port`.
    pub to: String,
    /// Optional edge name (names the per-connection channel attributes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// A system: instances of library types, connected by ports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SystemModel {
    /// System name.
    pub name: String,
    /// The library this system instantiates.
    pub library: LibraryRef,
    /// Instances.
    #[serde(default)]
    pub instances: Vec<Instance>,
    /// Connections.
    #[serde(default)]
    pub connections: Vec<SystemConnection>,
    /// Indicators, in the core form, referencing instances by name.
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(with = "Vec<raichu_model::Indicator>"))]
    pub indicators: Vec<Value>,
    /// Sequence-analysis targets, in the core form.
    #[serde(default)]
    pub targets: Vec<Target>,
    /// The model's `unbounded_rate`: the magnitude no integrated rate may
    /// reach. A system applying the standard continuous flow fragments
    /// gets the standard one without stating it, and may state only that
    /// one; any other system declares none unless it states it here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unbounded_rate: Option<f64>,
    /// Free text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Free editor data, never read by the expander.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// The activation of a catalogued mode by a study.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Activation {
    /// Mode name in the type's catalogue.
    pub mode: String,
    /// The instances it applies to; exclusive with `type`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instances: Option<Vec<String>>,
    /// Every instance of this type; exclusive with `instances`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_name: Option<String>,
    /// Occurrence law (core law object), required.
    #[cfg_attr(feature = "schema", schemars(with = "raichu_model::Distrib"))]
    pub occurrence: Value,
    /// Repair law; absent means the mode, once occurred, stays occurred.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(with = "Option<raichu_model::Distrib>"))]
    pub repair: Option<Value>,
    /// Interruption policy of both transitions (default `reset`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_interruption: Option<InterruptionPolicy>,
    /// An extra activation condition, ANDed with the catalogue condition;
    /// a core expression that may also use `{"op": "mode_active", "mode":
    /// "Instance.mode"}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(with = "Option<crate::schema::Condition>")
    )]
    pub condition: Option<Value>,
    /// A repair condition: the repair cannot complete while it is false
    /// (a guard on the repair edge only, never on the occurrence or an
    /// on-demand re-arm). Same grammar as `condition`; requires `repair`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(with = "Option<crate::schema::Condition>")
    )]
    pub repair_condition: Option<Value>,
    /// How the mode's effects follow its state: absent, at once; with
    /// `zero_delay` or `instant`, one step later, through an automaton
    /// `<mode>__effect` whose transition follows the mode's occurred state
    /// with a zero delay or an instantaneous law of probability one. The
    /// step is what a PyCATSHOO external mode does: its effects ride on a
    /// mirror automaton one zero-delay step behind the mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect_step: Option<EffectStep>,
    /// Free editor data, never read by the expander.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// A mode the study declares on one instance only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct OneOffMode {
    /// The instance carrying the mode.
    pub instance: String,
    /// Mode name, unique on its instance.
    pub name: String,
    /// `failure` or `compromise`.
    pub nature: Nature,
    /// Effects on targetable attributes of the instance.
    #[serde(default)]
    pub effects: Vec<Effect>,
    /// Activation condition (core expression, `mode_active` allowed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(with = "Option<crate::schema::Condition>")
    )]
    pub condition: Option<Value>,
    /// Occurrence law, required.
    #[cfg_attr(feature = "schema", schemars(with = "raichu_model::Distrib"))]
    pub occurrence: Value,
    /// Repair law; absent means absorbing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(with = "Option<raichu_model::Distrib>"))]
    pub repair: Option<Value>,
    /// Interruption policy (default `reset`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_interruption: Option<InterruptionPolicy>,
    /// A repair condition: the repair cannot complete while it is false
    /// (a guard on the repair edge only, never on the occurrence or an
    /// on-demand re-arm). Same grammar as `condition`; requires `repair`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(with = "Option<crate::schema::Condition>")
    )]
    pub repair_condition: Option<Value>,
    /// How the mode's effects follow its state: absent, at once; with
    /// `zero_delay` or `instant`, one step later, through an automaton
    /// `<mode>__effect` whose transition follows the mode's occurred state
    /// with a zero delay or an instantaneous law of probability one. The
    /// step is what a PyCATSHOO external mode does: its effects ride on a
    /// mirror automaton one zero-delay step behind the mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect_step: Option<EffectStep>,
    /// Free editor data, never read by the expander.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// A common-cause group: a mode catalogued on every member's type, failing
/// the members together by combination. Each active order `k` gives one
/// automaton per combination of `k` members, named after their 1-based
/// positions in `instances` (`cc_1_2`), in a component named after the
/// group; a combination holds every member it covers with the catalogued
/// effects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct CommonCauseGroup {
    /// Group name, also the name of its generated component.
    pub name: String,
    /// The catalogued mode it fails its members with.
    pub mode: String,
    /// The members, in order: their positions name the combinations.
    pub instances: Vec<String>,
    /// One entry per active order; an absent order builds no combination.
    pub orders: Vec<GroupOrder>,
    /// With a step, each member's effects follow the combinations covering
    /// it one step behind, through an automaton `<group>__effect` on the
    /// member, and a combination occurs only with its members' steps at
    /// rest and is repaired only once they have taken hold: a PyCATSHOO
    /// external common-cause mode. Without, effects are immediate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect_step: Option<EffectStep>,
    /// Free editor data, never read by the expander.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// The laws of one order of a common-cause group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct GroupOrder {
    /// How many members a combination of this order fails together.
    pub order: usize,
    /// Occurrence law of each combination.
    #[cfg_attr(feature = "schema", schemars(with = "raichu_model::Distrib"))]
    pub occurrence: Value,
    /// Repair law; absent means a combination, once occurred, stays so.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(with = "Option<raichu_model::Distrib>"))]
    pub repair: Option<Value>,
}

/// The step a mode's effects take after its state (`effect_step`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum EffectStep {
    /// A deterministic transition with a zero delay.
    ZeroDelay,
    /// An instantaneous transition of probability one.
    Instant,
}

/// The nature of a mode. It serves selection and reporting and changes
/// no semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Nature {
    /// A failure mode.
    Failure,
    /// A compromise (cyber) mode.
    Compromise,
}

/// One effect of a mode: the value it holds on an attribute while the
/// mode is occurred.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Effect {
    /// Targeted attribute of the instance.
    pub attribute: String,
    /// Held value: a plain JSON boolean or number.
    #[cfg_attr(feature = "schema", schemars(with = "crate::schema::EffectValue"))]
    pub value: Value,
}

/// A one-shot write a catalogued mode makes on one edge of its lifecycle:
/// written once when the edge fires, then kept until something else writes
/// the attribute (a gate that memorises). On an on-demand draw the write
/// lands only on the branch that enters `occ`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct EdgeEffect {
    /// Targetable attribute of the instance, written by no held effect.
    pub attribute: String,
    /// Written value: a plain JSON boolean or number.
    #[cfg_attr(feature = "schema", schemars(with = "crate::schema::EffectValue"))]
    pub value: Value,
    /// The edge that writes it.
    pub on: Edge,
}

/// An edge of a mode's lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Edge {
    /// The mode occurs (`occ`).
    Occurrence,
    /// The mode is repaired (`rep`).
    Repair,
}

/// A feared event: a generated component watching a condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct FearedEvent {
    /// Event name; it becomes the component name.
    pub name: String,
    /// Condition (core expression over instance attributes).
    #[cfg_attr(feature = "schema", schemars(with = "crate::schema::Condition"))]
    pub condition: Value,
    /// Delay the condition must hold before the event occurs.
    #[serde(default)]
    pub tempo_occ: f64,
    /// Delay the condition must fail before the event clears.
    #[serde(default)]
    pub tempo_not_occ: f64,
    /// Whether reaching the event ends and labels a sequence.
    #[serde(default)]
    pub target: bool,
    /// Free editor data, never read by the expander.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// How the study is run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Method {
    /// Indicator statistics over replicas.
    MonteCarlo,
    /// A sequence campaign.
    Sequences,
}

/// Run settings of a study, plain data mapped onto the engine's runs by
/// the caller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Simulation {
    /// Run method.
    pub method: Method,
    /// Number of replicas.
    pub nb_runs: u64,
    /// Horizon.
    pub horizon: f64,
    /// Sampling instants of the indicators.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instants: Option<Vec<f64>>,
    /// Master seed.
    #[serde(default)]
    pub seed: u64,
    /// Stop each replica at the first target reached.
    #[serde(default)]
    pub stop_at_targets: bool,
}

/// A study: which modes run, with which laws, and what is observed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct StudyDocument {
    /// Study name.
    pub name: String,
    /// The system the study applies to, by name.
    pub system: String,
    /// Activations of catalogued modes.
    #[serde(default)]
    pub modes: Vec<Activation>,
    /// Modes declared for one instance only.
    #[serde(default)]
    pub one_off_modes: Vec<OneOffMode>,
    /// Feared events.
    #[serde(default)]
    pub feared_events: Vec<FearedEvent>,
    /// Common-cause groups: a catalogued mode failing several instances
    /// together, one law per order.
    #[serde(default)]
    pub common_cause_groups: Vec<CommonCauseGroup>,
    /// Indicators, in the core form.
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(with = "Vec<raichu_model::Indicator>"))]
    pub indicators: Vec<Value>,
    /// Sequence-analysis targets, in the core form.
    #[serde(default)]
    pub targets: Vec<Target>,
    /// Run settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub simulation: Option<Simulation>,
    /// Free text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Free editor data, never read by the expander.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

fn empty_object() -> Value {
    Value::Object(Map::new())
}

fn empty_array() -> Value {
    Value::Array(Vec::new())
}

const ENVELOPES: [(&str, &str, DocumentKind); 3] = [
    (LIBRARY_KEY, "library", DocumentKind::Library),
    (SYSTEM_KEY, "system", DocumentKind::System),
    (STUDY_KEY, "study", DocumentKind::Study),
];

/// Reads the envelope of a document of the expected `kind` and returns
/// the body value, refusing a document of another kind (the flat core
/// model included) and any other revision.
fn body_of(text: &str, kind: DocumentKind) -> Result<Value, ExpandError> {
    let (key, body_key, _) = ENVELOPES
        .iter()
        .copied()
        .find(|(_, _, k)| *k == kind)
        .unwrap_or(ENVELOPES[0]);
    let root: Value = serde_json::from_str(text)
        .map_err(|e| ExpandError::new(kind, "", format!("not valid JSON: {e}")))?;
    let Value::Object(mut map) = root else {
        return Err(ExpandError::new(kind, "", "a document is a JSON object"));
    };
    if !map.contains_key(key) {
        let found = ENVELOPES
            .iter()
            .map(|(k, _, _)| *k)
            .chain(std::iter::once("raichu_model"))
            .find(|k| map.contains_key(*k));
        let message = match found {
            Some(other) => format!(
                "this is a `{other}` document; a {kind} document carries the envelope `{key}`"
            ),
            None => format!("missing envelope `{key}` (expected `\"{key}\": {{\"format\": {FORMAT_REVISION}}}`)"),
        };
        return Err(ExpandError::new(kind, "", message));
    }
    let header_value = map.remove(key).unwrap_or(Value::Null);
    let header: Header = serde_path_to_error::deserialize(header_value)
        .map_err(|e| ExpandError::new(kind, format!("/{key}"), e.into_inner().to_string()))?;
    if header.format != FORMAT_REVISION {
        return Err(ExpandError::new(
            kind,
            format!("/{key}/format"),
            format!(
                "envelope revision {} is not supported; this engine reads revision {FORMAT_REVISION}",
                header.format
            ),
        ));
    }
    let body = map.remove(body_key).ok_or_else(|| {
        ExpandError::new(
            kind,
            "",
            format!("missing `{body_key}` beside the envelope"),
        )
    })?;
    if let Some(extra) = map.keys().next() {
        return Err(ExpandError::new(
            kind,
            format!("/{extra}"),
            format!(
                "unknown field `{extra}`: a {kind} document holds `{key}` and `{body_key}` only"
            ),
        ));
    }
    Ok(body)
}

fn deserialize_body<T: serde::de::DeserializeOwned>(
    body: Value,
    kind: DocumentKind,
    body_key: &str,
) -> Result<T, ExpandError> {
    crate::strict::located(&body, kind, &format!("/{body_key}"), "")
}

fn named<'a>(
    kind: DocumentKind,
    what: &str,
    pointer: String,
    name: &'a str,
) -> Result<&'a str, ExpandError> {
    check_name(what, name).map_err(|m| ExpandError::new(kind, pointer, m))?;
    Ok(name)
}

/// Parses a library document.
pub fn parse_library(text: &str) -> Result<Library, ExpandError> {
    let kind = DocumentKind::Library;
    let library: Library = deserialize_body(body_of(text, kind)?, kind, "library")?;
    named(kind, "library", "/library/name".into(), &library.name)?;
    for (i, fragment) in library.fragments.iter().enumerate() {
        named(
            kind,
            "fragment",
            format!("/library/fragments/{i}/name"),
            &fragment.name,
        )?;
        if library.fragments[..i]
            .iter()
            .any(|f| f.name == fragment.name)
        {
            return Err(ExpandError::new(
                kind,
                format!("/library/fragments/{i}/name"),
                format!("fragment `{}` is declared twice", fragment.name),
            ));
        }
    }
    for (i, ty) in library.types.iter().enumerate() {
        named(kind, "type", format!("/library/types/{i}/name"), &ty.name)?;
    }
    Ok(library)
}

/// Parses a system document.
pub fn parse_system(text: &str) -> Result<SystemModel, ExpandError> {
    let kind = DocumentKind::System;
    let system: SystemModel = deserialize_body(body_of(text, kind)?, kind, "system")?;
    for (i, instance) in system.instances.iter().enumerate() {
        named(
            kind,
            "instance",
            format!("/system/instances/{i}/name"),
            &instance.name,
        )?;
    }
    Ok(system)
}

/// Parses a study document.
pub fn parse_study(text: &str) -> Result<StudyDocument, ExpandError> {
    let kind = DocumentKind::Study;
    let study: StudyDocument = deserialize_body(body_of(text, kind)?, kind, "study")?;
    for (i, event) in study.feared_events.iter().enumerate() {
        named(
            kind,
            "feared event",
            format!("/study/feared_events/{i}/name"),
            &event.name,
        )?;
    }
    for (i, mode) in study.one_off_modes.iter().enumerate() {
        named(
            kind,
            "mode",
            format!("/study/one_off_modes/{i}/name"),
            &mode.name,
        )?;
    }
    Ok(study)
}

/// Writes a library back as a document.
pub fn library_document(library: &Library) -> Value {
    envelope(LIBRARY_KEY, "library", serde_json::to_value(library))
}

/// Writes a system back as a document.
pub fn system_document(system: &SystemModel) -> Value {
    envelope(SYSTEM_KEY, "system", serde_json::to_value(system))
}

/// Writes a study back as a document.
pub fn study_document(study: &StudyDocument) -> Value {
    envelope(STUDY_KEY, "study", serde_json::to_value(study))
}

fn envelope(key: &str, body_key: &str, body: Result<Value, serde_json::Error>) -> Value {
    let mut map = Map::new();
    map.insert(
        key.to_owned(),
        serde_json::json!({ "format": FORMAT_REVISION }),
    );
    map.insert(body_key.to_owned(), body.unwrap_or(Value::Null));
    Value::Object(map)
}
