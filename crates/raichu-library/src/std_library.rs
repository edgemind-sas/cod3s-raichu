//! The standard library `std`, shipped with the engine.

use crate::document::{parse_library, Library};
use crate::error::ExpandError;
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

/// The libraries an expansion draws on: `given`, plus the standard library
/// when one of them imports it and none of them is it.
pub(crate) fn with_std(given: &[Library]) -> Result<Vec<Library>, ExpandError> {
    let mut all = given.to_vec();
    let imported = given.iter().any(|l| l.imports.iter().any(|i| i == STD));
    if imported && !given.iter().any(|l| l.name == STD) {
        all.push(standard_library()?);
    }
    Ok(all)
}
