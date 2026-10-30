//! The libraries shipped with the engine: the registry, and the standard
//! library `std` it holds.

use crate::document::{parse_library, Library};
use crate::error::{DocumentKind, ExpandError};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

/// Name of the standard library, as libraries import it.
pub const STD: &str = "std";

/// Version of the standard library: the engine's.
pub const STD_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The magnitude the standard continuous fragments write for "unbounded"
/// (a volume's ceiling nobody set), and the one an expanded continuous
/// network declares as its `unbounded_rate`: the muscadet layer's
/// `UNBOUNDED_SERVICE`, so a library-built network refuses the same
/// integrated rates the plugin-built one does.
pub const STD_UNBOUNDED_RATE: f64 = 1.0e30;

/// The standard fragments whose application makes the instance part of a
/// continuous network: the two that declare a continuous flow (the
/// plugin's test: a system declaring a continuous flow carries the guard),
/// and the volume, which writes the stand-in itself.
pub(crate) const CONTINUOUS_FLOW_FRAGMENTS: [&str; 3] = [
    "std.continuous_flow_out",
    "std.continuous_flow_in",
    "std.volume_flow",
];

const STD_DOCUMENT: &str = include_str!("../std/std.json");

/// The standard library: the fragments shipped with RAICHU, written in
/// the library format itself. Its version is the engine's, because a
/// standard fragment changes with the engine.
pub fn standard_library() -> Result<Library, ExpandError> {
    static PARSED: OnceLock<Result<Library, ExpandError>> = OnceLock::new();
    PARSED
        .get_or_init(|| {
            let mut library = parse_library(STD_DOCUMENT)?;
            STD_VERSION.clone_into(&mut library.version);
            Ok(library)
        })
        .clone()
}

/// Names of the libraries shipped with the engine, in registry order.
pub const EMBEDDED: [&str; 1] = [STD];

/// A library shipped with the engine, by name; `None` for a name the
/// registry does not hold. Each is versioned with the engine.
pub fn embedded_library(name: &str) -> Option<Result<Library, ExpandError>> {
    match name {
        STD => Some(standard_library()),
        _ => None,
    }
}

/// The libraries an expansion draws on: `given`, plus every embedded
/// library some library imports, directly or through another library
/// (a fixpoint over imports, so a library importing only one that imports
/// `std` still gets `std`).
///
/// A given library named after an embedded one is refused unless it is
/// that very library, so a user library cannot silently stand in for an
/// embedded one; an import cycle is refused, naming its libraries.
pub(crate) fn with_embedded(given: &[Library]) -> Result<Vec<Library>, ExpandError> {
    let mut all: Vec<Library> = Vec::with_capacity(given.len() + EMBEDDED.len());
    for library in given {
        if let Some(embedded) = embedded_library(&library.name) {
            let embedded = embedded?;
            if &embedded != library {
                let detail = if embedded.version == library.version {
                    format!(
                        "library `{}` is named after a library shipped with the engine and differs from it; rename it",
                        library.name
                    )
                } else {
                    format!(
                        "library `{}` is supplied at version `{}`, but the engine ships `{}`; drop it from the supplied libraries (an embedded library is added on import), or rename a library of your own",
                        library.name, library.version, embedded.version
                    )
                };
                return Err(ExpandError::new(
                    DocumentKind::Library,
                    "/library/name",
                    detail,
                ));
            }
        }
        if !all.iter().any(|l| l.name == library.name && l == library) {
            all.push(library.clone());
        }
    }
    loop {
        let missing: Vec<&str> = EMBEDDED
            .iter()
            .copied()
            .filter(|name| !all.iter().any(|l| l.name == *name))
            .filter(|name| all.iter().any(|l| l.imports.iter().any(|i| i == name)))
            .collect();
        if missing.is_empty() {
            break;
        }
        for name in missing {
            if let Some(library) = embedded_library(name) {
                all.push(library?);
            }
        }
    }
    refuse_import_cycles(&all)?;
    Ok(all)
}

/// The embedded libraries `bound` reaches through its imports, with their
/// versions: what the expanded model records as its source.
pub(crate) fn embedded_versions(bound: &Library, all: &[Library]) -> BTreeMap<String, String> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut stack: Vec<&str> = bound.imports.iter().map(String::as_str).collect();
    while let Some(name) = stack.pop() {
        if !seen.insert(name) {
            continue;
        }
        if let Some(library) = all.iter().find(|l| l.name == name) {
            stack.extend(library.imports.iter().map(String::as_str));
        }
    }
    seen.into_iter()
        .filter(|name| EMBEDDED.contains(name))
        .filter_map(|name| all.iter().find(|l| l.name == name))
        .map(|l| (l.name.clone(), l.version.clone()))
        .collect()
}

fn refuse_import_cycles(all: &[Library]) -> Result<(), ExpandError> {
    fn visit<'a>(
        name: &'a str,
        all: &'a [Library],
        path: &mut Vec<&'a str>,
        done: &mut BTreeSet<&'a str>,
    ) -> Result<(), ExpandError> {
        if let Some(start) = path.iter().position(|n| *n == name) {
            let mut cycle: Vec<&str> = path[start..].to_vec();
            cycle.push(name);
            return Err(ExpandError::new(
                DocumentKind::Library,
                "/library/imports",
                format!(
                    "libraries import each other in a cycle: {}",
                    cycle.join(" -> ")
                ),
            ));
        }
        if done.contains(name) {
            return Ok(());
        }
        path.push(name);
        if let Some(library) = all.iter().find(|l| l.name == name) {
            for import in &library.imports {
                visit(import, all, path, done)?;
            }
        }
        path.pop();
        done.insert(name);
        Ok(())
    }
    let mut done = BTreeSet::new();
    for library in all {
        visit(&library.name, all, &mut Vec::new(), &mut done)?;
    }
    Ok(())
}
