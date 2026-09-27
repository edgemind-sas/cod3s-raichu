//! The content hash of a model document.

use raichu_model::Model;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::QuantifyError;

/// The content hash of `model`: `sha256:` followed by the lowercase hex
/// SHA-256 of its **sealed** document ([`Model::to_json`], format envelope
/// included) written as canonical JSON (object keys sorted, no
/// whitespace).
///
/// Two models that differ in any declared value have different hashes;
/// the same model hashes the same whatever the key order or layout of
/// the document it was read from.
///
/// The hash identifies the model **as sealed by the engine version that
/// computed it**: the sealed document spells out every defaulted field
/// and the format revision, so an engine release that adds a field with
/// a default changes the hash of an unchanged model. Compare hashes only
/// between envelopes carrying the same `engine_version`.
///
/// # Errors
/// [`QuantifyError::ModelSerialization`] when the model cannot be
/// serialized (a non-finite number where the document needs one).
pub fn model_content_hash(model: &Model) -> Result<String, QuantifyError> {
    let sealed = model
        .to_json()
        .map_err(|e| QuantifyError::ModelSerialization(e.to_string()))?;
    // Parsed correctly rounded: two models one ulp apart in a parameter
    // must not hash the same.
    let value: Value =
        crate::exact_json::parse(&sealed).map_err(QuantifyError::ModelSerialization)?;
    let mut canonical = String::new();
    write_canonical(&value, &mut canonical)
        .map_err(|e| QuantifyError::ModelSerialization(e.to_string()))?;
    let digest = Sha256::digest(canonical.as_bytes());
    let mut hex = String::with_capacity(7 + 2 * digest.len());
    hex.push_str("sha256:");
    for byte in digest {
        hex.push(nibble(byte >> 4));
        hex.push(nibble(byte & 0x0f));
    }
    Ok(hex)
}

fn nibble(n: u8) -> char {
    char::from(if n < 10 { b'0' + n } else { b'a' + n - 10 })
}

/// Write `value` as canonical JSON: objects with their keys sorted by
/// byte order, no whitespace, scalars as `serde_json` writes them. The
/// key order is imposed here rather than inherited from the map type, so
/// it does not depend on a `serde_json` feature another crate enables.
fn write_canonical(value: &Value, out: &mut String) -> Result<(), serde_json::Error> {
    match value {
        Value::Array(items) => {
            out.push('[');
            for (k, item) in items.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                write_canonical(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            out.push('{');
            for (k, (key, item)) in entries.into_iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key)?);
                out.push(':');
                write_canonical(item, out)?;
            }
            out.push('}');
        }
        scalar => out.push_str(&serde_json::to_string(scalar)?),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::write_canonical;

    #[test]
    fn canonical_json_sorts_keys_at_every_depth_and_drops_whitespace() {
        let value = serde_json::json!({"b": [1, {"z": true, "a": null}], "a": "x"});
        let mut out = String::new();
        write_canonical(&value, &mut out).unwrap();
        assert_eq!(out, r#"{"a":"x","b":[1,{"a":null,"z":true}]}"#);
    }
}
