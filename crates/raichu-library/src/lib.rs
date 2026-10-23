//! # raichu-library: component type libraries
//!
//! Three data documents let a model be written without code:
//!
//! - a **library** (`raichu_library`) declares parameterised component
//!   types, built from core constructs and named fragments, each with the
//!   catalogue of its failure and compromise modes;
//! - a **system** (`raichu_system`) instantiates those types by name and
//!   connects their ports;
//! - a **study** (`raichu_study`) activates catalogued modes with their
//!   laws, adds conditions across instances, and declares feared events,
//!   indicators and run settings.
//!
//! The expander turns the three into the flat core [`raichu_model::Model`]
//! the engine already runs, recording where every element came from.
//! Nothing here changes how a flat model is validated, compiled or
//! simulated.
//!
//! With the `schema` feature (off by default), the `schema` module
//! derives the JSON Schemas of the three documents from these types; the
//! published copies live under `docs/reference/schema/`.

mod check;
mod combine;
pub mod document;
pub mod error;
mod expand;
mod instantiate;
mod modes;
pub mod names;
pub mod provenance;
#[cfg(feature = "schema")]
pub mod schema;
mod std_library;
mod strict;
mod study;
mod template;

pub use check::{check_library, TypeCheck};
pub use document::{
    parse_library, parse_study, parse_system, Activation, Effect, FearedEvent, Fragment, Header,
    Instance, ItemKind, Library, LibraryRef, Method, Nature, OneOffMode, ParamKind, Parameter,
    Simulation, StudyDocument, SystemConnection, SystemModel, TypeDef, Use, FORMAT_REVISION,
    LIBRARY_KEY, STUDY_KEY, SYSTEM_KEY,
};
pub use error::{DocumentKind, ExpandError};
pub use expand::{expand, Expansion};
pub use instantiate::CombineRule;
pub use provenance::{Origin, Provenance};
pub use std_library::{standard_library, STD};
