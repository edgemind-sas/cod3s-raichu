//! The templating grammar of type and fragment bodies.
//!
//! A body is ordinary core JSON in which four forms are resolved at
//! expansion, against the parameter values of the instance:
//!
//! - `{"$param": "name"}` stands for the parameter's value, anywhere a
//!   value goes (a law parameter included);
//! - `"${name}"` inside a string interpolates a string, number or boolean
//!   parameter (or a `$for_each` variable): `"${flow}_fed_out"`;
//! - `{"$for_each": "list", "as": "var", "items": [...]}`, as an item of
//!   an array, repeats `items` once per element of a list parameter, with
//!   `var` bound to the element;
//! - `{"$law": "law_param", ...}` writes the keys of a law parameter
//!   (`distrib`, `rate`, ...) onto the object beside its other keys, the
//!   way a transition carries its law;
//! - `{"$choose": "enum", "cases": {"a": ..., "b": ...}}` selects the case
//!   named by an enum (or string, or boolean) parameter;
//! - `{"$when": "param", "is": [...], "items": [...]}`, as an item of an
//!   array, keeps `items` in place when the parameter's value is one of
//!   `is` (a value or a list of values), and drops them otherwise; without
//!   `is` the parameter is a boolean and `items` are kept when it is true.
//!   An `is` value an `enum` or `bool` parameter never takes is refused.
//!
//! The string `"$self"`, and the prefix `"$self."`, name the instance
//! being expanded. Parameters reduce to literals here: nothing computes at
//! expansion beyond selection and repetition, arithmetic stays in runtime
//! expressions.

use crate::document::{ItemKind, ParamKind, Parameter};
use crate::error::{pointer_index, pointer_key, DocumentKind, ExpandError};
use serde_json::{Map, Number, Value};
use std::collections::BTreeMap;

/// Bound parameter values, by name.
#[derive(Debug, Clone, Default)]
pub(crate) struct Env {
    values: BTreeMap<String, Value>,
    /// The values an `enum` or `bool` parameter can take, as selection
    /// keys: what a `$when` may list in `is`.
    selectable: BTreeMap<String, Vec<String>>,
}

impl Env {
    fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }

    fn with(&self, name: &str, value: Value) -> Env {
        let mut next = self.clone();
        next.values.insert(name.to_owned(), value);
        // A loop variable takes whatever its list holds.
        next.selectable.remove(name);
        next
    }
}

/// Where a template is being resolved, for located errors.
#[derive(Debug, Clone)]
pub(crate) struct Ctx<'a> {
    /// The document holding the template.
    pub document: DocumentKind,
    /// Instance being expanded (the `$self` value).
    pub instance: &'a str,
    /// Human description of what is being expanded, prefixed to messages.
    pub context: String,
}

impl Ctx<'_> {
    fn err(&self, pointer: &str, message: impl AsRef<str>) -> ExpandError {
        ExpandError::new(
            self.document,
            pointer,
            format!("{}: {}", self.context, message.as_ref()),
        )
    }
}

fn kind_name(kind: ParamKind) -> &'static str {
    match kind {
        ParamKind::Number => "number",
        ParamKind::Integer => "integer",
        ParamKind::Bool => "bool",
        ParamKind::String => "string",
        ParamKind::Law => "law",
        ParamKind::List => "list",
        ParamKind::Enum => "enum",
    }
}

/// Checks one value against a parameter declaration, normalising an
/// integer given for a number to a float.
fn check_value(param: &Parameter, value: &Value) -> Result<Value, String> {
    let wrong = || {
        format!(
            "parameter `{}` is a {} but the value given is `{value}`",
            param.name,
            kind_name(param.kind)
        )
    };
    match param.kind {
        ParamKind::Number => {
            let x = value.as_f64().ok_or_else(wrong)?;
            Number::from_f64(x).map(Value::Number).ok_or_else(wrong)
        }
        ParamKind::Integer => value.as_i64().map(Value::from).ok_or_else(wrong),
        ParamKind::Bool => value.as_bool().map(Value::Bool).ok_or_else(wrong),
        ParamKind::String => value
            .as_str()
            .map(|s| Value::String(s.to_owned()))
            .ok_or_else(wrong),
        ParamKind::Law => {
            if value.get("distrib").and_then(Value::as_str).is_some() {
                Ok(value.clone())
            } else {
                Err(format!(
                    "parameter `{}` is a law, written as a core law object such as {{\"distrib\": \"exp\", \"rate\": 0.1}}; got `{value}`",
                    param.name
                ))
            }
        }
        ParamKind::Enum => {
            let s = value.as_str().ok_or_else(wrong)?;
            if param.values.iter().any(|v| v == s) {
                Ok(Value::String(s.to_owned()))
            } else {
                Err(format!(
                    "parameter `{}` takes one of {}; `{s}` is not one of them",
                    param.name,
                    param
                        .values
                        .iter()
                        .map(|v| format!("`{v}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            }
        }
        ParamKind::List => {
            let items = value.as_array().ok_or_else(wrong)?;
            let item_kind = param.items.unwrap_or(ItemKind::String);
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let ok = match item_kind {
                    ItemKind::Number => item.as_f64().and_then(Number::from_f64).map(Value::Number),
                    ItemKind::Integer => item.as_i64().map(Value::from),
                    ItemKind::Bool => item.as_bool().map(Value::Bool),
                    ItemKind::String => item.as_str().map(|s| Value::String(s.to_owned())),
                    ItemKind::StringList => item
                        .as_array()
                        .filter(|xs| xs.iter().all(Value::is_string))
                        .map(|xs| Value::Array(xs.clone())),
                    ItemKind::Object => item.as_object().map(|o| Value::Object(o.clone())),
                };
                out.push(ok.ok_or_else(|| {
                    let kind = match item_kind {
                        ItemKind::Number => "number",
                        ItemKind::Integer => "integer",
                        ItemKind::Bool => "bool",
                        ItemKind::String => "string",
                        ItemKind::StringList => "string_list",
                        ItemKind::Object => "object",
                    };
                    format!(
                        "parameter `{}` is a list of {kind} items; `{item}` is not one",
                        param.name
                    )
                })?);
            }
            Ok(Value::Array(out))
        }
    }
}

/// Binds the parameter values `given` (found at `given_pointer`) against
/// the declarations `params` (found at `params_pointer`).
///
/// A value the declarations do not name is refused; an omitted parameter
/// takes its default, and an omitted parameter without default is refused
/// naming the parameter.
pub(crate) fn bind(
    params: &[Parameter],
    given: &Map<String, Value>,
    given_doc: DocumentKind,
    given_pointer: &str,
    params_pointer: &str,
    ctx: &Ctx<'_>,
) -> Result<Env, ExpandError> {
    for (index, param) in params.iter().enumerate() {
        if params[..index].iter().any(|p| p.name == param.name) {
            return Err(ExpandError::new(
                DocumentKind::Library,
                pointer_index(params_pointer, index),
                format!(
                    "{}: parameter `{}` is declared twice",
                    ctx.context, param.name
                ),
            ));
        }
    }
    for key in given.keys() {
        if !params.iter().any(|p| &p.name == key) {
            let known: Vec<String> = params.iter().map(|p| format!("`{}`", p.name)).collect();
            return Err(ExpandError::new(
                given_doc,
                pointer_key(given_pointer, key),
                format!(
                    "{}: no parameter `{key}` is declared (declared: {})",
                    ctx.context,
                    if known.is_empty() {
                        "none".to_owned()
                    } else {
                        known.join(", ")
                    }
                ),
            ));
        }
    }
    let mut env = Env::default();
    for (index, param) in params.iter().enumerate() {
        let (value, doc, pointer) = match given.get(&param.name) {
            Some(v) => (v, given_doc, pointer_key(given_pointer, &param.name)),
            None => match &param.default {
                Some(d) => (
                    d,
                    DocumentKind::Library,
                    format!("{}/default", pointer_index(params_pointer, index)),
                ),
                None => {
                    return Err(ExpandError::new(
                        given_doc,
                        given_pointer,
                        format!(
                            "{}: required parameter `{}` is missing",
                            ctx.context, param.name
                        ),
                    ))
                }
            },
        };
        let checked = check_value(param, value)
            .map_err(|m| ExpandError::new(doc, pointer, format!("{}: {m}", ctx.context)))?;
        env.values.insert(param.name.clone(), checked);
        let keys = match param.kind {
            ParamKind::Enum => Some(param.values.clone()),
            ParamKind::Bool => Some(vec!["true".to_owned(), "false".to_owned()]),
            _ => None,
        };
        if let Some(keys) = keys {
            env.selectable.insert(param.name.clone(), keys);
        }
    }
    Ok(env)
}

/// Resolves every template form inside `value`, found at `pointer`.
pub(crate) fn substitute(
    value: &Value,
    env: &Env,
    ctx: &Ctx<'_>,
    pointer: &str,
) -> Result<Value, ExpandError> {
    match value {
        Value::String(s) => Ok(Value::String(interpolate(s, env, ctx, pointer)?)),
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for (index, item) in items.iter().enumerate() {
                let here = pointer_index(pointer, index);
                if let Some(list_name) = item.get("$for_each") {
                    out.extend(for_each(item, list_name, env, ctx, &here)?);
                } else if let Some(map) = item.as_object().filter(|m| m.contains_key("$when")) {
                    out.extend(when(map, env, ctx, &here)?);
                } else {
                    out.push(substitute(item, env, ctx, &here)?);
                }
            }
            Ok(Value::Array(out))
        }
        Value::Object(map) => {
            if let Some(name) = map.get("$param") {
                if map.len() != 1 {
                    return Err(ctx.err(pointer, "`$param` stands alone in its object"));
                }
                let name = name
                    .as_str()
                    .ok_or_else(|| ctx.err(pointer, "`$param` names a parameter by a string"))?;
                return env
                    .get(name)
                    .cloned()
                    .ok_or_else(|| ctx.err(pointer, format!("unknown parameter `{name}`")));
            }
            if let Some(selector) = map.get("$choose") {
                return choose(map, selector, env, ctx, pointer);
            }
            if let Some(name) = map.get("$law") {
                return spread_law(map, name, env, ctx, pointer);
            }
            if map.contains_key("$for_each") {
                return Err(ctx.err(pointer, "`$for_each` repeats items of an array; it cannot stand where a single value goes"));
            }
            if map.contains_key("$when") {
                return Err(ctx.err(pointer, "`$when` keeps or drops items of an array; it cannot stand where a single value goes"));
            }
            let mut out = Map::with_capacity(map.len());
            for (key, child) in map {
                let key_out = interpolate(key, env, ctx, pointer)?;
                out.insert(
                    key_out,
                    substitute(child, env, ctx, &pointer_key(pointer, key))?,
                );
            }
            Ok(Value::Object(out))
        }
        other => Ok(other.clone()),
    }
}

/// `{"$law": "param", ...}`: the keys of a law parameter written onto the
/// object that names it, beside its other keys, as a transition carries
/// its law flattened (`"distrib": "exp", "rate": ...`). A key both the law
/// and the object write is refused rather than silently overwritten.
fn spread_law(
    map: &Map<String, Value>,
    name: &Value,
    env: &Env,
    ctx: &Ctx<'_>,
    pointer: &str,
) -> Result<Value, ExpandError> {
    let name = name
        .as_str()
        .ok_or_else(|| ctx.err(pointer, "`$law` names a law parameter by a string"))?;
    let law = env
        .get(name)
        .ok_or_else(|| ctx.err(pointer, format!("unknown parameter `{name}`")))?;
    let Some(keys) = law.as_object().filter(|m| m.contains_key("distrib")) else {
        return Err(ctx.err(
            pointer,
            format!("`$law` spreads a law parameter; `{name}` is not a law object"),
        ));
    };
    let mut out = Map::with_capacity(map.len() + keys.len());
    for (key, child) in map {
        if key == "$law" {
            continue;
        }
        let key_out = interpolate(key, env, ctx, pointer)?;
        out.insert(
            key_out,
            substitute(child, env, ctx, &pointer_key(pointer, key))?,
        );
    }
    for (key, value) in keys {
        if out.contains_key(key) {
            return Err(ctx.err(
                pointer,
                format!("`$law` spreads `{key}` from `{name}`, which the object already writes"),
            ));
        }
        out.insert(key.clone(), value.clone());
    }
    Ok(Value::Object(out))
}

fn for_each(
    item: &Value,
    list_name: &Value,
    env: &Env,
    ctx: &Ctx<'_>,
    pointer: &str,
) -> Result<Vec<Value>, ExpandError> {
    let Some(map) = item.as_object() else {
        return Ok(Vec::new());
    };
    if let Some(extra) = map
        .keys()
        .find(|k| !matches!(k.as_str(), "$for_each" | "as" | "items"))
    {
        return Err(ctx.err(
            &pointer_key(pointer, extra),
            format!("unknown field `{extra}` in `$for_each` (fields: `$for_each`, `as`, `items`)"),
        ));
    }
    let list_name = list_name
        .as_str()
        .ok_or_else(|| ctx.err(pointer, "`$for_each` names a list parameter by a string"))?;
    let var = map.get("as").and_then(Value::as_str).ok_or_else(|| {
        ctx.err(
            pointer,
            "`$for_each` needs `as`, the name bound to each element",
        )
    })?;
    let items = map.get("items").and_then(Value::as_array).ok_or_else(|| {
        ctx.err(
            pointer,
            "`$for_each` needs `items`, the array repeated per element",
        )
    })?;
    let list = env
        .get(list_name)
        .ok_or_else(|| ctx.err(pointer, format!("unknown parameter `{list_name}`")))?;
    let elements = list.as_array().ok_or_else(|| {
        ctx.err(
            pointer,
            format!("`$for_each` repeats over a list; `{list_name}` is not one"),
        )
    })?;
    let mut out = Vec::new();
    for element in elements {
        let inner = env.with(var, element.clone());
        let items_pointer = pointer_key(pointer, "items");
        let expanded = substitute(&Value::Array(items.clone()), &inner, ctx, &items_pointer)?;
        if let Value::Array(values) = expanded {
            out.extend(values);
        }
    }
    Ok(out)
}

/// The items of a `$when` array item: `items`, substituted, when the
/// parameter's value is one of `is` (or, without `is`, when the boolean
/// parameter is true); nothing otherwise.
fn when(
    map: &Map<String, Value>,
    env: &Env,
    ctx: &Ctx<'_>,
    pointer: &str,
) -> Result<Vec<Value>, ExpandError> {
    if let Some(extra) = map
        .keys()
        .find(|k| !matches!(k.as_str(), "$when" | "is" | "items"))
    {
        return Err(ctx.err(
            &pointer_key(pointer, extra),
            format!("unknown field `{extra}` in `$when` (fields: `$when`, `is`, `items`)"),
        ));
    }
    let name = map
        .get("$when")
        .and_then(Value::as_str)
        .ok_or_else(|| ctx.err(pointer, "`$when` names a parameter by a string"))?;
    let items = map.get("items").and_then(Value::as_array).ok_or_else(|| {
        ctx.err(
            pointer,
            "`$when` needs `items`, the array kept when the condition holds",
        )
    })?;
    let value = env
        .get(name)
        .ok_or_else(|| ctx.err(pointer, format!("unknown parameter `{name}`")))?;
    let keep = match map.get("is") {
        None => value.as_bool().ok_or_else(|| {
            ctx.err(
                pointer,
                format!("`$when` without `is` reads a boolean parameter; `{name}` is not one, so name the values that keep the items with `is`"),
            )
        })?,
        Some(listed) => {
            let key = scalar_key(value, name, "$when", ctx, pointer)?;
            let listed: Vec<&Value> = match listed {
                Value::Array(values) => values.iter().collect(),
                single => vec![single],
            };
            let is_pointer = pointer_key(pointer, "is");
            let declared = env.selectable.get(name);
            let mut keys = Vec::with_capacity(listed.len());
            for (index, one) in listed.iter().enumerate() {
                let here = if map.get("is").is_some_and(Value::is_array) {
                    pointer_index(&is_pointer, index)
                } else {
                    is_pointer.clone()
                };
                let listed_key = scalar_key(one, name, "$when", ctx, &here)?;
                // A value the parameter never takes would drop the items
                // for every instance without a word.
                if let Some(declared) = declared {
                    if !declared.contains(&listed_key) {
                        return Err(ctx.err(
                            &here,
                            format!(
                                "`$when` on `{name}` lists `{listed_key}`, which `{name}` never takes (values: {})",
                                declared
                                    .iter()
                                    .map(|v| format!("`{v}`"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                        ));
                    }
                }
                keys.push(listed_key);
            }
            keys.contains(&key)
        }
    };
    // Substituted as one array, so a `$when` or `$for_each` among the
    // items splices in turn; substituted even when dropped, so an error in
    // them shows on every instance, not only on those that keep them.
    let substituted = substitute(
        &Value::Array(items.clone()),
        env,
        ctx,
        &pointer_key(pointer, "items"),
    )?;
    if !keep {
        return Ok(Vec::new());
    }
    match substituted {
        Value::Array(values) => Ok(values),
        other => Ok(vec![other]),
    }
}

/// A scalar as it reads for a selection (`$choose` case key, `$when`
/// value): a string as written, a boolean as `true` or `false`, a number
/// as [`number_text`] writes it.
fn scalar_key(
    value: &Value,
    name: &str,
    form: &str,
    ctx: &Ctx<'_>,
    pointer: &str,
) -> Result<String, ExpandError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Bool(b) => Ok(b.to_string()),
        Value::Number(n) => Ok(number_text(n)),
        _ => Err(ctx.err(
            pointer,
            format!("`{form}` selects on a scalar; `{name}` is not one"),
        )),
    }
}

fn choose(
    map: &Map<String, Value>,
    selector: &Value,
    env: &Env,
    ctx: &Ctx<'_>,
    pointer: &str,
) -> Result<Value, ExpandError> {
    if let Some(extra) = map
        .keys()
        .find(|k| !matches!(k.as_str(), "$choose" | "cases"))
    {
        return Err(ctx.err(
            &pointer_key(pointer, extra),
            format!("unknown field `{extra}` in `$choose` (fields: `$choose`, `cases`)"),
        ));
    }
    let name = selector
        .as_str()
        .ok_or_else(|| ctx.err(pointer, "`$choose` names a parameter by a string"))?;
    let cases = map.get("cases").and_then(Value::as_object).ok_or_else(|| {
        ctx.err(
            pointer,
            "`$choose` needs `cases`, an object keyed by the parameter's values",
        )
    })?;
    let chosen = env
        .get(name)
        .ok_or_else(|| ctx.err(pointer, format!("unknown parameter `{name}`")))?;
    let key = scalar_key(chosen, name, "$choose", ctx, pointer)?;
    let case = cases.get(&key).ok_or_else(|| {
        ctx.err(
            pointer,
            format!(
                "`$choose` on `{name}` has no case `{key}` (cases: {})",
                cases
                    .keys()
                    .map(|k| format!("`{k}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )
    })?;
    substitute(
        case,
        env,
        ctx,
        &pointer_key(&pointer_key(pointer, "cases"), &key),
    )
}

/// A number as it reads inside a name or a `$choose` key: an integral
/// value without a fraction (`2`, not `2.0`), any other value as JSON
/// writes it.
fn number_text(n: &Number) -> String {
    match n.as_f64() {
        #[allow(clippy::cast_possible_truncation)]
        Some(x) if x.is_finite() && x.fract() == 0.0 && x.abs() < 9.0e15 => format!("{}", x as i64),
        _ => n.to_string(),
    }
}

/// Resolves `$self` and `${name}` inside one string.
fn interpolate(s: &str, env: &Env, ctx: &Ctx<'_>, pointer: &str) -> Result<String, ExpandError> {
    if s == "$self" {
        return Ok(ctx.instance.to_owned());
    }
    if let Some(rest) = s.strip_prefix("$self.") {
        return Ok(format!(
            "{}.{}",
            ctx.instance,
            interpolate(rest, env, ctx, pointer)?
        ));
    }
    if !s.contains("${") {
        return Ok(s.to_owned());
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let end = after
            .find('}')
            .ok_or_else(|| ctx.err(pointer, format!("unclosed `${{` in `{s}`")))?;
        let name = &after[..end];
        let value = env
            .get(name)
            .ok_or_else(|| ctx.err(pointer, format!("unknown parameter `{name}` in `{s}`")))?;
        match value {
            Value::String(v) => out.push_str(v),
            Value::Number(v) => out.push_str(&number_text(v)),
            Value::Bool(v) => out.push_str(&v.to_string()),
            _ => {
                return Err(ctx.err(
                    pointer,
                    format!("`${{{name}}}` interpolates a scalar; `{name}` is not one"),
                ))
            }
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn params(spec: Value) -> Vec<Parameter> {
        serde_json::from_value(spec).unwrap_or_default()
    }

    fn ctx() -> Ctx<'static> {
        Ctx {
            document: DocumentKind::Library,
            instance: "B1",
            context: "instance `B1`".into(),
        }
    }

    fn env_of(spec: Value, given: Value) -> Result<Env, ExpandError> {
        let given = given.as_object().cloned().unwrap_or_default();
        bind(
            &params(spec),
            &given,
            DocumentKind::System,
            "/system/instances/0/parameters",
            "/p",
            &ctx(),
        )
    }

    #[test]
    fn for_each_repeats_items_and_interpolates_names() {
        let env = env_of(
            json!([{"name": "flow_types", "kind": "list"}]),
            json!({"flow_types": ["H2", "O2"]}),
        )
        .unwrap_or_default();
        let body =
            json!([{"$for_each": "flow_types", "as": "f", "items": [{"name": "${f}_content"}]}]);
        let out = substitute(&body, &env, &ctx(), "").ok();
        assert_eq!(
            out,
            Some(json!([{"name": "H2_content"}, {"name": "O2_content"}]))
        );
    }

    #[test]
    fn an_empty_list_repeats_nothing() {
        let env =
            env_of(json!([{"name": "xs", "kind": "list"}]), json!({"xs": []})).unwrap_or_default();
        let body = json!([1, {"$for_each": "xs", "as": "x", "items": ["${x}"]}, 2]);
        assert_eq!(
            substitute(&body, &env, &ctx(), "").ok(),
            Some(json!([1, 2]))
        );
    }

    #[test]
    fn when_keeps_its_items_on_a_true_boolean_and_drops_them_on_false() {
        let spec = json!([{"name": "ccf", "kind": "bool"}]);
        let body = json!(["a", {"$when": "ccf", "items": ["ccf_1", "ccf_2"]}, "b"]);
        let on = env_of(spec.clone(), json!({"ccf": true})).unwrap_or_default();
        assert_eq!(
            substitute(&body, &on, &ctx(), "").ok(),
            Some(json!(["a", "ccf_1", "ccf_2", "b"]))
        );
        let off = env_of(spec, json!({"ccf": false})).unwrap_or_default();
        assert_eq!(
            substitute(&body, &off, &ctx(), "").ok(),
            Some(json!(["a", "b"]))
        );
    }

    #[test]
    fn when_on_an_enum_keeps_its_items_for_the_listed_values_only() {
        let spec =
            json!([{"name": "rule", "kind": "enum", "values": ["printed", "tables", "perfect"]}]);
        let body = json!([{"$when": "rule", "is": ["printed", "tables"], "items": ["start"]}]);
        for (rule, expected) in [
            ("printed", json!(["start"])),
            ("tables", json!(["start"])),
            ("perfect", json!([])),
        ] {
            let env = env_of(spec.clone(), json!({"rule": rule})).unwrap_or_default();
            assert_eq!(
                substitute(&body, &env, &ctx(), "").ok(),
                Some(expected),
                "{rule}"
            );
        }
        let one = json!([{"$when": "rule", "is": "perfect", "items": ["start"]}]);
        let env = env_of(spec, json!({"rule": "perfect"})).unwrap_or_default();
        assert_eq!(
            substitute(&one, &env, &ctx(), "").ok(),
            Some(json!(["start"]))
        );
    }

    #[test]
    fn when_naming_an_unknown_parameter_is_refused_with_its_pointer() {
        let env = env_of(json!([]), json!({})).unwrap_or_default();
        let body = json!(["a", {"$when": "ccf", "items": ["x"]}]);
        let err = substitute(&body, &env, &ctx(), "/body").err();
        let (pointer, message) = err.map(|e| (e.pointer, e.message)).unwrap_or_default();
        assert_eq!(pointer, "/body/1");
        assert!(message.contains("unknown parameter `ccf`"), "{message}");
    }

    #[test]
    fn when_without_is_needs_a_boolean_parameter() {
        let spec = json!([{"name": "rule", "kind": "enum", "values": ["printed", "perfect"]}]);
        let env = env_of(spec, json!({"rule": "printed"})).unwrap_or_default();
        let body = json!([{"$when": "rule", "items": ["x"]}]);
        let message = substitute(&body, &env, &ctx(), "")
            .err()
            .map(|e| e.message)
            .unwrap_or_default();
        assert!(message.contains("`is`"), "{message}");
    }

    #[test]
    fn when_refuses_an_is_value_the_parameter_never_takes() {
        // A misspelled value would otherwise drop the items for every
        // instance without a word.
        let spec = json!([{"name": "rule", "kind": "enum", "values": ["printed", "tables"]}]);
        let env = env_of(spec, json!({"rule": "printed"})).unwrap_or_default();
        let body = json!([{"$when": "rule", "is": ["printed", "tabels"], "items": ["x"]}]);
        let err = substitute(&body, &env, &ctx(), "/body").err();
        let (pointer, message) = err.map(|e| (e.pointer, e.message)).unwrap_or_default();
        assert_eq!(pointer, "/body/0/is/1");
        assert!(message.contains("`tabels`"), "{message}");
        assert!(message.contains("`printed`, `tables`"), "{message}");
        let flag = env_of(
            json!([{"name": "ccf", "kind": "bool"}]),
            json!({"ccf": true}),
        )
        .unwrap_or_default();
        let body = json!([{"$when": "ccf", "is": "yes", "items": ["x"]}]);
        assert!(substitute(&body, &flag, &ctx(), "").is_err());
    }

    #[test]
    fn when_reports_an_error_in_the_items_it_drops() {
        let env = env_of(
            json!([{"name": "ccf", "kind": "bool"}]),
            json!({"ccf": false}),
        )
        .unwrap_or_default();
        let body = json!([{"$when": "ccf", "items": [{"$param": "nowhere"}]}]);
        let message = substitute(&body, &env, &ctx(), "")
            .err()
            .map(|e| e.message)
            .unwrap_or_default();
        assert!(message.contains("unknown parameter `nowhere`"), "{message}");
    }

    #[test]
    fn when_on_a_loop_variable_takes_any_value() {
        let env = env_of(
            json!([{"name": "flows", "kind": "list"}]),
            json!({"flows": ["H2", "O2"]}),
        )
        .unwrap_or_default();
        let body = json!([{"$for_each": "flows", "as": "f", "items": [
            {"$when": "f", "is": "H2", "items": ["${f}_kept"]}]}]);
        assert_eq!(
            substitute(&body, &env, &ctx(), "").ok(),
            Some(json!(["H2_kept"]))
        );
    }

    #[test]
    fn when_outside_an_array_is_refused() {
        let spec = json!([{"name": "ccf", "kind": "bool"}]);
        let env = env_of(spec, json!({"ccf": true})).unwrap_or_default();
        let body = json!({"effects": {"$when": "ccf", "items": ["x"]}});
        let message = substitute(&body, &env, &ctx(), "")
            .err()
            .map(|e| e.message)
            .unwrap_or_default();
        assert!(message.contains("`$when`"), "{message}");
    }

    #[test]
    fn choose_selects_and_unknown_enum_values_are_refused() {
        let spec =
            json!([{"name": "method", "kind": "enum", "values": ["mean", "median", "and", "or"]}]);
        let env = env_of(spec.clone(), json!({"method": "median"})).unwrap_or_default();
        let body =
            json!({"$choose": "method", "cases": {"mean": 1, "median": 2, "and": 3, "or": 4}});
        assert_eq!(substitute(&body, &env, &ctx(), "").ok(), Some(json!(2)));
        let err = env_of(spec, json!({"method": "average"}))
            .err()
            .map(|e| e.message)
            .unwrap_or_default();
        assert!(err.contains("`mean`, `median`, `and`, `or`"), "{err}");
    }

    #[test]
    fn param_values_and_self_are_substituted() {
        let env = env_of(
            json!([{"name": "lambda", "kind": "number"}, {"name": "law", "kind": "law", "default": {"distrib": "delay", "time": 5}}]),
            json!({"lambda": 2}),
        )
        .unwrap_or_default();
        let body = json!({"rate": {"$param": "lambda"}, "law": {"$param": "law"}, "who": "$self", "mode": "$self.mdc_a"});
        assert_eq!(
            substitute(&body, &env, &ctx(), "").ok(),
            Some(
                json!({"rate": 2.0, "law": {"distrib": "delay", "time": 5}, "who": "B1", "mode": "B1.mdc_a"})
            )
        );
    }

    #[test]
    fn binding_refuses_missing_unknown_and_mistyped_values() {
        let spec = json!([{"name": "capacity", "kind": "number"}, {"name": "content_ini", "kind": "number", "default": 0.0}]);
        let missing = env_of(spec.clone(), json!({}))
            .err()
            .map(|e| e.message)
            .unwrap_or_default();
        assert!(
            missing.contains("required parameter `capacity` is missing"),
            "{missing}"
        );
        let unknown = env_of(spec.clone(), json!({"capacity": 1, "colour": "red"})).err();
        assert_eq!(
            unknown.map(|e| e.pointer),
            Some("/system/instances/0/parameters/colour".into())
        );
        let mistyped = env_of(spec, json!({"capacity": "big"})).err();
        assert_eq!(
            mistyped.map(|e| e.pointer),
            Some("/system/instances/0/parameters/capacity".into())
        );
    }

    #[test]
    fn unknown_parameters_in_templates_are_refused() {
        let env = Env::default();
        assert!(substitute(&json!({"$param": "nope"}), &env, &ctx(), "").is_err());
        assert!(substitute(&json!("${nope}_x"), &env, &ctx(), "").is_err());
    }
}
