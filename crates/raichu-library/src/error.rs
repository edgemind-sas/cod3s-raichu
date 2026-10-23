//! Located expansion errors.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Which of the three documents an error was found in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentKind {
    /// A library document (`raichu_library`).
    Library,
    /// A system document (`raichu_system`).
    System,
    /// A study document (`raichu_study`).
    Study,
}

impl fmt::Display for DocumentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DocumentKind::Library => "library",
            DocumentKind::System => "system",
            DocumentKind::Study => "study",
        })
    }
}

/// An error refusing a library, system or study document, or their
/// expansion.
///
/// Every error names the document it was found in and the JSON Pointer
/// (RFC 6901) of the offending node, so an author, an editor or an LLM
/// can go straight to the fault. The pointer is empty when the fault is
/// the document as a whole.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error("{document} document, at `{pointer}`: {message}")]
pub struct ExpandError {
    /// The document the fault was found in.
    pub document: DocumentKind,
    /// JSON Pointer of the offending node inside that document.
    pub pointer: String,
    /// What is wrong, naming the elements involved.
    pub message: String,
}

impl ExpandError {
    /// Builds an error.
    pub fn new(
        document: DocumentKind,
        pointer: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            document,
            pointer: pointer.into(),
            message: message.into(),
        }
    }
}

/// Escapes one reference token of a JSON Pointer (RFC 6901).
#[must_use]
pub fn escape_token(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

/// Appends a key to a JSON Pointer.
#[must_use]
pub fn pointer_key(base: &str, key: &str) -> String {
    format!("{base}/{}", escape_token(key))
}

/// Appends an array index to a JSON Pointer.
#[must_use]
pub fn pointer_index(base: &str, index: usize) -> String {
    format!("{base}/{index}")
}
