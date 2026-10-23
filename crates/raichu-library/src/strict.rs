//! Strict, located deserialization.
//!
//! The library documents refuse unknown fields, but their bodies are
//! written in core constructs (`Transition`, `Indicator`, `Allocation`)
//! whose serde shapes flatten tagged enums, which rules out
//! `deny_unknown_fields` on them; the core reader silently drops a
//! misspelled key. [`typed`] closes that gap without touching the core
//! types: after a successful deserialization, it probes every object key
//! by replacing its value with two values no field type accepts at once
//! (a string and an object). A key the type knows rejects at least one of
//! them; a key the type ignores accepts both, and is refused with its JSON
//! Pointer. The probe needs no list of field names, so it cannot drift
//! from the core types.

use crate::error::{pointer_index, pointer_key, DocumentKind, ExpandError};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

/// JSON Pointer of the node `serde_path_to_error` reports, appended to
/// `base`.
fn pointer_of(base: &str, path: &serde_path_to_error::Path) -> String {
    let mut pointer = base.to_owned();
    for segment in path.iter() {
        match segment {
            serde_path_to_error::Segment::Seq { index } => {
                pointer = pointer_index(&pointer, *index);
            }
            serde_path_to_error::Segment::Map { key } => {
                pointer = pointer_key(&pointer, key);
            }
            serde_path_to_error::Segment::Enum { .. } | serde_path_to_error::Segment::Unknown => {}
        }
    }
    pointer
}

/// Deserializes `value` (found at `base` in a `document`) into `T` with a
/// located error, then refuses any key `T` silently ignores.
///
/// For core types only: a field holding free JSON (`serde_json::Value`)
/// accepts any probe and would read as unknown; this crate's own structs
/// use [`located`] and `deny_unknown_fields` instead.
///
/// `context` prefixes every message (for example "instance `Srv` of type
/// `Serveur`"), so an error raised while expanding a type body names the
/// instance being expanded as well as the library node.
pub(crate) fn typed<T: DeserializeOwned>(
    value: &Value,
    document: DocumentKind,
    base: &str,
    context: &str,
) -> Result<T, ExpandError> {
    let parsed: T = match serde_path_to_error::deserialize(value) {
        Ok(parsed) => parsed,
        Err(err) => {
            let pointer = pointer_of(base, err.path());
            let message = err.into_inner().to_string();
            return Err(ExpandError::new(
                document,
                pointer,
                prefixed(context, &message),
            ));
        }
    };
    if let Some((pointer, key)) = first_ignored_key::<T>(value) {
        return Err(ExpandError::new(
            document,
            format!("{base}{pointer}"),
            prefixed(
                context,
                &format!("unknown field `{key}`: no construct at this place reads it"),
            ),
        ));
    }
    Ok(parsed)
}

/// Deserializes `value` into `T` with a located error, without the
/// probe: for this crate's own structs, which refuse unknown fields
/// themselves and may hold free JSON values the probe would misread.
pub(crate) fn located<T: DeserializeOwned>(
    value: &Value,
    document: DocumentKind,
    base: &str,
    context: &str,
) -> Result<T, ExpandError> {
    serde_path_to_error::deserialize(value).map_err(|err| {
        let pointer = pointer_of(base, err.path());
        let message = err.into_inner().to_string();
        ExpandError::new(document, pointer, prefixed(context, &message))
    })
}

fn prefixed(context: &str, message: &str) -> String {
    if context.is_empty() {
        message.to_owned()
    } else {
        format!("{context}: {message}")
    }
}

/// The relative pointer and name of the first key `T` ignores, in
/// document order, or `None`. Keys under an ignored key are not probed.
fn first_ignored_key<T: DeserializeOwned>(root: &Value) -> Option<(String, String)> {
    let mut paths = Vec::new();
    collect_key_paths(root, &mut Vec::new(), &mut paths);
    let probes = [
        Value::String("\u{1}raichu-probe".to_owned()),
        Value::Object(Map::from_iter([("\u{1}".to_owned(), Value::Bool(true))])),
    ];
    let mut work = root.clone();
    for path in paths {
        let ignored = probes.iter().all(|probe| {
            let original = swap_at(&mut work, &path, probe.clone());
            let accepted = T::deserialize(&work).is_ok();
            if let Some(original) = original {
                swap_at(&mut work, &path, original);
            }
            accepted
        });
        if ignored {
            let mut pointer = String::new();
            let mut key = String::new();
            for step in &path {
                match step {
                    Step::Key(k) => {
                        pointer = pointer_key(&pointer, k);
                        key.clone_from(k);
                    }
                    Step::Index(i) => pointer = pointer_index(&pointer, *i),
                }
            }
            return Some((pointer, key));
        }
    }
    None
}

#[derive(Clone)]
enum Step {
    Key(String),
    Index(usize),
}

/// Every path ending on an object key, parents before children.
fn collect_key_paths(value: &Value, prefix: &mut Vec<Step>, out: &mut Vec<Vec<Step>>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                prefix.push(Step::Key(key.clone()));
                out.push(prefix.clone());
                collect_key_paths(child, prefix, out);
                prefix.pop();
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                prefix.push(Step::Index(index));
                collect_key_paths(child, prefix, out);
                prefix.pop();
            }
        }
        _ => {}
    }
}

/// Puts `replacement` at `path` and returns the value it replaced.
fn swap_at(root: &mut Value, path: &[Step], replacement: Value) -> Option<Value> {
    let (last, parents) = path.split_last()?;
    let mut node = root;
    for step in parents {
        node = match (step, node) {
            (Step::Key(key), Value::Object(map)) => map.get_mut(key)?,
            (Step::Index(index), Value::Array(items)) => items.get_mut(*index)?,
            _ => return None,
        };
    }
    let slot = match (last, node) {
        (Step::Key(key), Value::Object(map)) => map.get_mut(key)?,
        (Step::Index(index), Value::Array(items)) => items.get_mut(*index)?,
        _ => return None,
    };
    Some(std::mem::replace(slot, replacement))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use raichu_model::{Indicator, Transition};
    use serde_json::json;

    #[test]
    fn a_misspelled_guard_is_refused_with_its_pointer() {
        let transition = json!({
            "name": "occ", "source": "rep", "targets": ["occ"],
            "gaurd": {"op": "const", "value": {"kind": "bool", "value": true}},
            "distrib": "delay", "time": 1.0
        });
        let err = typed::<Transition>(&transition, DocumentKind::Library, "/x", "")
            .expect_err("misspelled key");
        assert_eq!(err.pointer, "/x/gaurd");
        assert!(
            err.message.contains("unknown field `gaurd`"),
            "{}",
            err.message
        );
    }

    #[test]
    fn known_default_keys_and_flattened_fields_pass() {
        let transition = json!({
            "name": "occ", "source": "rep", "targets": ["occ"],
            "monitored": false, "on_interruption": "reset",
            "distrib": "exp", "rate": 0.1
        });
        assert!(typed::<Transition>(&transition, DocumentKind::Library, "", "").is_ok());
        let indicator = json!({
            "name": "x", "target": "attribute",
            "attr": {"component": "a", "attribute": "b"}
        });
        assert!(typed::<Indicator>(&indicator, DocumentKind::System, "", "").is_ok());
    }

    #[test]
    fn a_misspelled_key_inside_an_expression_is_refused() {
        let transition = json!({
            "name": "occ", "source": "rep", "targets": ["occ"],
            "guard": {"op": "attr", "attr": {"component": "a", "attribute": "b", "atribute": "c"}},
            "distrib": "delay", "time": 1.0
        });
        let err = typed::<Transition>(&transition, DocumentKind::Library, "", "")
            .expect_err("misspelled key");
        assert_eq!(err.pointer, "/guard/attr/atribute");
    }

    #[test]
    fn a_structural_error_carries_the_path() {
        let transition = json!({
            "name": "occ", "source": "rep", "targets": ["occ"],
            "distrib": "delay", "time": "soon"
        });
        let err = typed::<Transition>(&transition, DocumentKind::Library, "/t", "instance `A`")
            .expect_err("bad time");
        assert!(err.pointer.starts_with("/t"), "{}", err.pointer);
        assert!(err.message.starts_with("instance `A`: "), "{}", err.message);
    }
}
