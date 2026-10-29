//! Where every expanded element came from.
//!
//! The expander records, for each element of the flat model, the document
//! node that produced it and the instance, type, fragment and mode it
//! belongs to. The map is keyed by an element path:
//!
//! | Key | Element |
//! |---|---|
//! | `component:C` | component `C` (an instance or a feared event) |
//! | `attribute:C.a` | attribute `a` of `C` |
//! | `port:C.p`, `interface:C.i` | port, interface |
//! | `automaton:C.a`, `transition:C.a.t` | automaton, transition |
//! | `function:C.f` | sensitive function |
//! | `equation:C.a` | equation targeting `a` |
//! | `allocation:C.n` | allocation |
//! | `connection:E` | connection named `E` (or `from->to`) |
//! | `indicator:N`, `target:N` | indicator, sequence target |

use crate::document::Nature;
use crate::error::DocumentKind;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The origin of one expanded element.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Origin {
    /// The document holding the node that produced the element.
    pub document: DocumentKind,
    /// The library, when the node is in a library document.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library: Option<String>,
    /// JSON Pointer of the node in that document.
    pub pointer: String,
    /// The instance the element belongs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
    /// The instance's type.
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
    pub type_name: Option<String>,
    /// The fragment that declared the element, qualified (`std.bool_flow_in`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fragment: Option<String>,
    /// The mode the element implements, by its name on the instance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// The nature of that mode (`failure` or `compromise`), for reporting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nature: Option<String>,
}

impl Origin {
    /// The body of type `type_name` in `library`, for `instance`.
    pub(crate) fn type_body(library: &str, instance: &str, type_name: &str, pointer: &str) -> Self {
        Self {
            document: DocumentKind::Library,
            library: Some(library.to_owned()),
            pointer: pointer.to_owned(),
            instance: Some(instance.to_owned()),
            type_name: Some(type_name.to_owned()),
            fragment: None,
            mode: None,
            nature: None,
        }
    }

    /// A node of the study document.
    pub(crate) fn study(pointer: impl Into<String>) -> Self {
        Self {
            document: DocumentKind::Study,
            library: None,
            pointer: pointer.into(),
            instance: None,
            type_name: None,
            fragment: None,
            mode: None,
            nature: None,
        }
    }

    /// A node of the system document.
    pub(crate) fn system(pointer: impl Into<String>) -> Self {
        Self {
            document: DocumentKind::System,
            ..Self::study(pointer)
        }
    }

    /// The same origin, for `instance` of type `type_name`.
    pub(crate) fn for_instance(&self, instance: &str, type_name: &str) -> Self {
        Self {
            instance: Some(instance.to_owned()),
            type_name: Some(type_name.to_owned()),
            ..self.clone()
        }
    }

    /// The same origin at another node.
    pub(crate) fn at(&self, pointer: impl Into<String>) -> Self {
        Self {
            pointer: pointer.into(),
            ..self.clone()
        }
    }

    /// The same instance, through a fragment of `library`.
    pub(crate) fn through_fragment(&self, qualified: &str, library: &str) -> Self {
        Self {
            fragment: Some(qualified.to_owned()),
            library: Some(library.to_owned()),
            ..self.clone()
        }
    }

    /// The same origin, implementing mode `mode`.
    pub(crate) fn with_mode(&self, mode: &str) -> Self {
        Self {
            mode: Some(mode.to_owned()),
            ..self.clone()
        }
    }

    /// The same origin, recording the nature of its mode.
    pub(crate) fn with_nature(&self, nature: Nature) -> Self {
        let text = match nature {
            Nature::Failure => "failure",
            Nature::Compromise => "compromise",
        };
        Self {
            nature: Some(text.to_owned()),
            ..self.clone()
        }
    }

    /// The same instance, located in the system document.
    pub(crate) fn in_system(&self) -> Self {
        Self {
            document: DocumentKind::System,
            library: None,
            fragment: None,
            ..self.clone()
        }
    }

    /// A short description for messages: "fragment `std.x` (library
    /// `std`, at `/library/fragments/0/body`)".
    #[must_use]
    pub fn describe(&self) -> String {
        let what = if let Some(fragment) = &self.fragment {
            format!("fragment `{fragment}`")
        } else if let Some(mode) = &self.mode {
            format!("mode `{mode}`")
        } else if let Some(type_name) = &self.type_name {
            format!("type `{type_name}`")
        } else {
            format!("the {} document", self.document)
        };
        match &self.library {
            Some(library) => format!("{what} (library `{library}`, at `{}`)", self.pointer),
            None => format!(
                "{what} (at `{}` in the {} document)",
                self.pointer, self.document
            ),
        }
    }
}

/// The origin of every element of an expanded model, by element path.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Element path to origin (see the module documentation).
    pub elements: BTreeMap<String, Origin>,
}

/// Element kinds a core message names in words before the element's name,
/// with the provenance key prefix each one is recorded under.
const NAMED_KINDS: [(&str, &str); 2] = [("FMU unit ", "fmu_unit"), ("program ", "program")];

impl Provenance {
    /// The origin of the component `name`, when the expansion produced it.
    #[must_use]
    pub fn component(&self, name: &str) -> Option<&Origin> {
        self.elements.get(&format!("component:{name}"))
    }

    /// The origin of the most specific element a message names between
    /// backticks (`C.x` before `C`), with its key.
    ///
    /// Core validation errors name the elements they refuse this way
    /// (`automaton \`C.a\` has unknown initial state ...`), which is how an
    /// error on the flat model is traced back to the document node.
    ///
    /// When the message names the kind of the element just before its name
    /// (`FMU unit \`plant\``, `program \`pick\``), that kind is looked up
    /// first: a program or an FMU unit may carry the name of an instance,
    /// and is then traced to the system node that declares it, not to that
    /// instance's type.
    #[must_use]
    pub fn locate(&self, message: &str) -> Option<(&str, &Origin)> {
        let parts: Vec<&str> = message.split('`').collect();
        for (index, token) in parts.iter().enumerate().skip(1).step_by(2) {
            let before = parts[index - 1];
            for (phrase, kind) in NAMED_KINDS {
                if before.ends_with(phrase) {
                    if let Some((key, origin)) =
                        self.elements.get_key_value(&format!("{kind}:{token}"))
                    {
                        return Some((key.as_str(), origin));
                    }
                }
            }
        }
        let quoted: Vec<&str> = message.split('`').skip(1).step_by(2).collect();
        let mut best: Option<(&str, &Origin)> = None;
        for token in quoted {
            for (key, origin) in &self.elements {
                let Some((_, name)) = key.split_once(':') else {
                    continue;
                };
                if name == token && best.is_none_or(|(k, _)| k.len() < key.len()) {
                    best = Some((key.as_str(), origin));
                }
            }
            if best.is_some() {
                return best;
            }
        }
        best
    }

    pub(crate) fn insert(&mut self, key: String, origin: Origin) {
        self.elements.entry(key).or_insert(origin);
    }
}
