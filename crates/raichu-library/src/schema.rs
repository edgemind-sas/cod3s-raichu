//! JSON Schemas (draft 2020-12) of the three documents, derived from the
//! reader's own types so the schema and the reader cannot disagree.
//!
//! The document structs, the core types of `raichu-model` and
//! `raichu-expr`, and the private shapes the expander types a body into
//! all derive [`JsonSchema`] behind the `schema` feature. This module
//! assembles the derived definitions into one schema per document and
//! applies two transforms the derive cannot express:
//!
//! - **closing**: the reader refuses unknown keys everywhere (its own
//!   structs deny them, core constructs go through a strict probe), so
//!   every object definition is closed. A plain object gets
//!   `additionalProperties: false`; an object flattening a tagged enum
//!   (a transition and its law, an allocation and its policy, an
//!   indicator and its target) gets `unevaluatedProperties: false`, which
//!   sees the properties of the matching variant;
//! - **templating** (library document only): type and fragment bodies,
//!   targetable lists and mode catalogues are read after template
//!   substitution, so every value there also accepts `{"$param": name}`
//!   and a `{"$choose": enum, "cases": {...}}` object, and every array
//!   also accepts `{"$for_each": list, "as": var, "items": [...]}` and
//!   `{"$when": param, "is": values, "items": [...]}` items.
//!   Tag values (`"distrib": "exp"`, `"op": "cmp"`) stay literal: a
//!   variant is selected with `$choose`, never with `$param`.
//!
//! System and study documents are not templated. Their conditions accept
//! the `{"op": "mode_active", "mode": "Instance.mode"}` leaf at any depth,
//! their laws are core distributions and their indicators core
//! indicators.
//!
//! The schemas describe shape only: name syntax, cross references and
//! value ranges stay with the reader, which reports them with a JSON
//! Pointer.

use crate::document::{Library, StudyDocument, SystemModel, LIBRARY_KEY, STUDY_KEY, SYSTEM_KEY};
use raichu_expr::Expr;
use schemars::generate::SchemaSettings;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde_json::{json, Map, Value};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

/// The `$schema` URI of the three schemas.
pub const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";

const DEFS: &str = "#/$defs/";
const PARAM_DEF: &str = "TemplateParam";

/// Definitions written by the library document itself, outside any
/// templated part: they are never templated.
const LIBRARY_OWN: [&str; 7] = [
    "Library",
    "Fragment",
    "TypeDef",
    "Parameter",
    "ParamKind",
    "ItemKind",
    "Use",
];

/// Schema-only stand-in for a condition: a core expression in which the
/// leaf `{"op": "mode_active", "mode": "Instance.mode"}` may stand
/// wherever a sub-expression goes.
pub struct Condition;

impl JsonSchema for Condition {
    fn schema_name() -> Cow<'static, str> {
        "Condition".into()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        let mut value = Expr::json_schema(generator).to_value();
        rewrite_refs(
            &mut value,
            &format!("{DEFS}Expr"),
            &format!("{DEFS}Condition"),
        );
        if let Value::Object(map) = &mut value {
            map.insert(
                "description".to_owned(),
                Value::String(
                    "A condition: a core expression in which `mode_active` names an \
                     activated mode as `Instance.mode`."
                        .to_owned(),
                ),
            );
            if let Some(Value::Array(branches)) = map.get_mut("oneOf") {
                branches.push(json!({
                    "description": "True while the named mode is occurred.",
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "const": "mode_active"},
                        "mode": {"type": "string"}
                    },
                    "required": ["op", "mode"]
                }));
            }
        }
        Schema::try_from(value).unwrap_or_default()
    }
}

/// Schema-only stand-in for the value a mode effect holds: a plain JSON
/// boolean or number.
pub struct EffectValue;

impl JsonSchema for EffectValue {
    fn schema_name() -> Cow<'static, str> {
        "EffectValue".into()
    }

    fn inline_schema() -> bool {
        true
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        schemars::json_schema!({"type": ["boolean", "number"]})
    }
}

/// The schemas of the three documents.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentSchemas {
    /// Schema of a library document (`raichu_library`).
    pub library: Value,
    /// Schema of a system document (`raichu_system`).
    pub system: Value,
    /// Schema of a study document (`raichu_study`).
    pub study: Value,
}

/// Derives the schemas of the three documents.
#[must_use]
pub fn document_schemas() -> DocumentSchemas {
    DocumentSchemas {
        library: document_schema::<Library>(
            LIBRARY_KEY,
            "library",
            "RAICHU library document",
            true,
        ),
        system: document_schema::<SystemModel>(
            SYSTEM_KEY,
            "system",
            "RAICHU system document",
            false,
        ),
        study: document_schema::<StudyDocument>(STUDY_KEY, "study", "RAICHU study document", false),
    }
}

fn document_schema<T: JsonSchema>(
    envelope: &str,
    body_key: &str,
    title: &str,
    templated: bool,
) -> Value {
    let mut generator = SchemaSettings::draft2020_12().into_generator();
    let body = generator.subschema_for::<T>().to_value();
    let mut defs: BTreeMap<String, Value> = generator.take_definitions(true).into_iter().collect();
    for schema in defs.values_mut() {
        close(schema);
    }
    if templated {
        let mut templater = Templater::default();
        let names: Vec<String> = defs
            .keys()
            .filter(|name| !LIBRARY_OWN.contains(&name.as_str()))
            .cloned()
            .collect();
        for name in names {
            if let Some(schema) = defs.remove(&name) {
                let templated = templater.def(&name, schema);
                defs.insert(name, templated);
            }
        }
        // The targetable list and the mode catalogue of a type or a fragment
        // are read after substitution like the body: their items may repeat
        // with `$for_each`, although the type and fragment objects themselves
        // are the library's own, untemplated shape.
        for owner in ["Fragment", "TypeDef"] {
            let Some(Value::Object(properties)) = defs
                .get_mut(owner)
                .and_then(|def| def.get_mut("properties"))
            else {
                continue;
            };
            for key in ["targetable", "modes"] {
                if let Some(list) = properties.remove(key) {
                    let templated = templater.node(list, &format!("{owner}_{key}"));
                    properties.insert(key.to_owned(), templated);
                }
            }
        }
        spread_law_branch(&mut defs);
        defs.insert(
            PARAM_DEF.to_owned(),
            json!({
                "description": "A template parameter: stands for the parameter's value.",
                "type": "object",
                "properties": {"$param": {"type": "string"}},
                "required": ["$param"],
                "additionalProperties": false
            }),
        );
        defs.append(&mut templater.added);
    }
    let mut root = json!({
        "$schema": DIALECT,
        "title": title,
        "type": "object",
        "properties": {
            envelope: {
                "description": "Envelope header.",
                "type": "object",
                "properties": {"format": {"const": crate::document::FORMAT_REVISION}},
                "required": ["format"],
                "additionalProperties": false
            },
            body_key: body
        },
        "required": [envelope, body_key],
        "additionalProperties": false
    });
    let reachable = reachable_defs(&root, &defs);
    defs.retain(|name, _| reachable.contains(name));
    if let Value::Object(map) = &mut root {
        map.insert(
            "$defs".to_owned(),
            Value::Object(defs.into_iter().collect()),
        );
    }
    canonical(root)
}

/// Closes every object schema under `schema` (see the module docs).
fn close(schema: &mut Value) {
    close_node(schema, false);
}

/// `partial` marks a branch of a flattened enum: it describes only part of
/// its object, so it is not closed itself.
fn close_node(node: &mut Value, partial: bool) {
    let Value::Object(map) = node else {
        if let Value::Array(items) = node {
            for item in items {
                close_node(item, false);
            }
        }
        return;
    };
    let has_properties = map.contains_key("properties");
    let flattened = has_properties
        && ["oneOf", "anyOf", "allOf"]
            .iter()
            .any(|key| map.contains_key(*key));
    if has_properties && !partial {
        if flattened {
            map.insert("unevaluatedProperties".to_owned(), Value::Bool(false));
        } else if !map.contains_key("additionalProperties") {
            map.insert("additionalProperties".to_owned(), Value::Bool(false));
        }
    }
    for (key, child) in map.iter_mut() {
        match key.as_str() {
            "oneOf" | "anyOf" | "allOf" => {
                if let Value::Array(branches) = child {
                    for branch in branches {
                        close_node(branch, flattened);
                    }
                }
            }
            "properties" | "$defs" | "patternProperties" => {
                if let Value::Object(entries) = child {
                    for entry in entries.values_mut() {
                        close_node(entry, false);
                    }
                }
            }
            "items" | "additionalProperties" | "prefixItems" | "not" => {
                close_node(child, false);
            }
            _ => {}
        }
    }
}

/// A transition may take its law from a `law` parameter with
/// `{"$law": "param", ...}` (the reader spreads the law's keys onto it), so
/// its templated definition gains a branch: the transition's own keys
/// beside `$law`, and none of the law's keys, which the parameter writes.
fn spread_law_branch(defs: &mut BTreeMap<String, Value>) {
    let Some(Value::Object(transition)) = defs.get_mut("Transition") else {
        return;
    };
    let Some(Value::Array(branches)) = transition.get_mut("anyOf") else {
        return;
    };
    let Some(Value::Object(own)) = branches.first().cloned() else {
        return;
    };
    let mut properties = own
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    properties.insert(
        "$law".to_owned(),
        json!({"type": "string", "description": "A `law` parameter whose keys the reader writes onto the transition."}),
    );
    let mut required: Vec<Value> = own
        .get("required")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    required.push(Value::String("$law".to_owned()));
    branches.push(json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    }));
}

/// Builds the templated form of the library definitions.
#[derive(Default)]
struct Templater {
    /// `$for_each` definitions created on the way, by name.
    added: BTreeMap<String, Value>,
}

impl Templater {
    /// The templated form of definition `name`: its own shape with every
    /// slot templated, or a `$param`, or a `$choose` among such values.
    fn def(&mut self, name: &str, schema: Value) -> Value {
        let inner = self.node(schema, name);
        let mut description = None;
        let inner = match inner {
            Value::Object(mut map) => {
                description = map.remove("description");
                Value::Object(map)
            }
            other => other,
        };
        let mut wrapped = json!({
            "anyOf": [
                inner,
                param_ref(),
                choose(json!({"$ref": format!("{DEFS}{name}")}))
            ]
        });
        if let (Some(text), Value::Object(map)) = (description, &mut wrapped) {
            map.insert("description".to_owned(), text);
        }
        wrapped
    }

    /// Templates the slots inside `node` (its properties, its array items
    /// and the branches of its combinators), not `node` itself.
    fn node(&mut self, node: Value, owner: &str) -> Value {
        let Value::Object(mut map) = node else {
            return node;
        };
        if map.contains_key("const") || map.contains_key("$ref") {
            return Value::Object(map);
        }
        if let Some(Value::Object(properties)) = map.remove("properties") {
            let properties = properties
                .into_iter()
                .map(|(key, value)| {
                    let slot = self.slot(value, &format!("{owner}_{key}"));
                    (key, slot)
                })
                .collect();
            map.insert("properties".to_owned(), Value::Object(properties));
        }
        for key in ["oneOf", "anyOf", "allOf"] {
            if let Some(Value::Array(branches)) = map.remove(key) {
                let branches = branches
                    .into_iter()
                    .map(|branch| self.node(branch, owner))
                    .collect();
                map.insert(key.to_owned(), Value::Array(branches));
            }
        }
        if let Some(Value::Array(items)) = map.remove("prefixItems") {
            let items = items
                .into_iter()
                .map(|item| self.slot(item, owner))
                .collect();
            map.insert("prefixItems".to_owned(), Value::Array(items));
        }
        if let Some(items) = map.remove("items") {
            let item_slot = self.slot(items.clone(), owner);
            let suffix = repeat_suffix(&items, owner);
            let name = format!("ForEach_{suffix}");
            let when_name = format!("When_{suffix}");
            let item_or_repeat = json!({
                "anyOf": [
                    item_slot,
                    {"$ref": format!("{DEFS}{name}")},
                    {"$ref": format!("{DEFS}{when_name}")}
                ]
            });
            if !self.added.contains_key(&name) {
                self.added
                    .insert(name.clone(), for_each(item_or_repeat.clone()));
                self.added.insert(when_name, when(item_or_repeat.clone()));
            }
            map.insert("items".to_owned(), item_or_repeat);
        }
        Value::Object(map)
    }

    /// A value slot: a reference (its definition is templated), a tag
    /// literal (kept), or an inline schema wrapped so it also accepts a
    /// `$param` and a `$choose`.
    fn slot(&mut self, value: Value, owner: &str) -> Value {
        let keep = matches!(&value, Value::Object(map) if map.contains_key("const") || map.contains_key("$ref"))
            || !value.is_object();
        if keep {
            return value;
        }
        let inner = self.node(value, owner);
        let bare = without_annotations(&inner);
        let case = json!({"anyOf": [bare.clone(), param_ref()]});
        let templated = json!({"anyOf": [bare.clone(), param_ref(), choose(case)]});
        // A scalar slot recurs on almost every object: share its templated
        // form as one definition, keeping the slot's own annotations.
        if let Some(name) = scalar_name(&bare) {
            let shared = match self.added.get(&name) {
                None => {
                    self.added.insert(name.clone(), templated.clone());
                    true
                }
                Some(existing) => *existing == templated,
            };
            if shared {
                let mut reference = Map::new();
                reference.insert("$ref".to_owned(), Value::String(format!("{DEFS}{name}")));
                if let Value::Object(map) = &inner {
                    for key in ["description", "default"] {
                        if let Some(annotation) = map.get(key) {
                            reference.insert(key.to_owned(), annotation.clone());
                        }
                    }
                }
                return Value::Object(reference);
            }
        }
        let mut templated = templated;
        if let (Value::Object(map), Value::Object(target)) = (&inner, &mut templated) {
            for key in ["description", "default"] {
                if let Some(annotation) = map.get(key) {
                    target.insert(key.to_owned(), annotation.clone());
                }
            }
        }
        templated
    }
}

/// Name of the shared templated definition of a bare scalar schema
/// (`type`, optionally `format` and `minimum`), or `None` for any other
/// shape.
fn scalar_name(bare: &Value) -> Option<String> {
    let Value::Object(map) = bare else {
        return None;
    };
    if map
        .keys()
        .any(|key| !matches!(key.as_str(), "type" | "format" | "minimum"))
    {
        return None;
    }
    let kinds: Vec<&str> = match map.get("type")? {
        Value::String(kind) => vec![kind.as_str()],
        Value::Array(kinds) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => return None,
    };
    if kinds.iter().any(|kind| matches!(*kind, "array" | "object")) {
        return None;
    }
    let mut name = format!("Template_{}", kinds.join("_or_"));
    if let Some(Value::String(format)) = map.get("format") {
        name.push('_');
        name.push_str(format);
    }
    Some(name)
}

fn param_ref() -> Value {
    json!({"$ref": format!("{DEFS}{PARAM_DEF}")})
}

fn choose(case: Value) -> Value {
    json!({
        "type": "object",
        "properties": {
            "$choose": {"type": "string"},
            "cases": {"type": "object", "additionalProperties": case}
        },
        "required": ["$choose", "cases"],
        "additionalProperties": false
    })
}

fn for_each(item: Value) -> Value {
    json!({
        "description": "Repeats `items` once per element of a list parameter, bound to `as`.",
        "type": "object",
        "properties": {
            "$for_each": {"type": "string"},
            "as": {"type": "string"},
            "items": {"type": "array", "items": item}
        },
        "required": ["$for_each", "as", "items"],
        "additionalProperties": false
    })
}

fn when(item: Value) -> Value {
    let scalar = json!({"type": ["string", "boolean", "number"]});
    json!({
        "description": "Keeps `items` when the parameter's value is one of `is` (without `is`, when the boolean parameter is true), and drops them otherwise.",
        "type": "object",
        "properties": {
            "$when": {"type": "string"},
            "is": {"anyOf": [scalar.clone(), {"type": "array", "items": scalar}]},
            "items": {"type": "array", "items": item}
        },
        "required": ["$when", "items"],
        "additionalProperties": false
    })
}

/// Suffix of the `$for_each` and `$when` definitions over `items`: the
/// referenced definition, the scalar type, or the owning slot.
fn repeat_suffix(items: &Value, owner: &str) -> String {
    match items {
        Value::Object(map) => match (map.get("$ref"), map.get("type"), map.len()) {
            (Some(Value::String(reference)), _, _) => {
                reference.strip_prefix(DEFS).unwrap_or(reference).to_owned()
            }
            (None, Some(Value::String(kind)), _) if kind != "array" && kind != "object" => {
                kind.clone()
            }
            _ => owner.to_owned(),
        },
        _ => owner.to_owned(),
    }
}

fn without_annotations(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(key, _)| !matches!(key.as_str(), "description" | "default"))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn rewrite_refs(value: &mut Value, from: &str, to: &str) {
    match value {
        Value::Object(map) => {
            if map.get("$ref").and_then(Value::as_str) == Some(from) {
                map.insert("$ref".to_owned(), Value::String(to.to_owned()));
            }
            for child in map.values_mut() {
                rewrite_refs(child, from, to);
            }
        }
        Value::Array(items) => {
            for child in items {
                rewrite_refs(child, from, to);
            }
        }
        _ => {}
    }
}

fn collect_refs(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(reference)) = map.get("$ref") {
                if let Some(name) = reference.strip_prefix(DEFS) {
                    out.push(name.to_owned());
                }
            }
            for child in map.values() {
                collect_refs(child, out);
            }
        }
        Value::Array(items) => {
            for child in items {
                collect_refs(child, out);
            }
        }
        _ => {}
    }
}

/// Names of the definitions reachable from `root`.
fn reachable_defs(root: &Value, defs: &BTreeMap<String, Value>) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut pending = Vec::new();
    collect_refs(root, &mut pending);
    while let Some(name) = pending.pop() {
        if seen.insert(name.clone()) {
            if let Some(schema) = defs.get(&name) {
                collect_refs(schema, &mut pending);
            }
        }
    }
    seen
}

/// Sorts every object's keys, so the written file does not depend on the
/// map implementation `serde_json` was built with.
fn canonical(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<String, Value> = map
                .into_iter()
                .map(|(key, value)| (key, canonical(value)))
                .collect();
            Value::Object(sorted.into_iter().collect::<Map<String, Value>>())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonical).collect()),
        other => other,
    }
}
