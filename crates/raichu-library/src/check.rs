//! Checking a library without a system.

use crate::document::{Instance, Library, LibraryRef, ParamKind, SystemModel};
use crate::error::ExpandError;
use crate::expand::expand;
use serde_json::{Map, Value};

/// The outcome of checking one type of a library.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeCheck {
    /// Every variant of the type expanded into a valid component.
    Valid,
    /// A variant failed; the errors name the variant's parameter values.
    Invalid(Vec<ExpandError>),
    /// The type has a required parameter without default, so no instance
    /// can be built without a system; the parameters are named.
    Unchecked(Vec<String>),
}

/// Checks every type of `library` without a system: each type is expanded
/// once with its defaults, and once more per value of each of its enum
/// parameters, against the other libraries in `libraries` (the standard
/// library is added when imported).
pub fn check_library(library: &Library, libraries: &[Library]) -> Vec<(String, TypeCheck)> {
    let mut available: Vec<Library> = libraries
        .iter()
        .filter(|l| l.name != library.name)
        .cloned()
        .collect();
    available.push(library.clone());
    library
        .types
        .iter()
        .map(|ty| {
            let required: Vec<String> = ty
                .parameters
                .iter()
                .filter(|p| p.default.is_none())
                .map(|p| p.name.clone())
                .collect();
            if !required.is_empty() {
                return (ty.name.clone(), TypeCheck::Unchecked(required));
            }
            let mut variants: Vec<Map<String, Value>> = vec![Map::new()];
            for param in ty.parameters.iter().filter(|p| p.kind == ParamKind::Enum) {
                for value in &param.values {
                    let mut given = Map::new();
                    given.insert(param.name.clone(), Value::String(value.clone()));
                    variants.push(given);
                }
            }
            let errors: Vec<ExpandError> = variants
                .into_iter()
                .filter_map(|parameters| {
                    let system = SystemModel {
                        name: format!("check {}", ty.name),
                        library: LibraryRef {
                            name: library.name.clone(),
                            version: library.version.clone(),
                        },
                        instances: vec![Instance {
                            name: "x".to_owned(),
                            type_name: ty.name.clone(),
                            parameters,
                            metadata: None,
                        }],
                        connections: Vec::new(),
                        indicators: Vec::new(),
                        targets: Vec::new(),
                        description: None,
                        metadata: None,
                    };
                    expand(&available, &system, None).err()
                })
                .collect();
            let outcome = if errors.is_empty() {
                TypeCheck::Valid
            } else {
                TypeCheck::Invalid(errors)
            };
            (ty.name.clone(), outcome)
        })
        .collect()
}
