//! Generic FMI 3.0 co-simulation runtime and model-description generator.
//!
//! The C ABI follows the vendored FMI 3.0.2 headers in `include/`.

mod description;
mod entry;
mod instance;

pub use description::{model_description, prepare_document, ExportError, ExportManifest, Variable};

/// Fixed binary name used by every exported RAICHU model.
pub const MODEL_IDENTIFIER: &str = "raichu_fmu";
/// Resource filename inside an exported FMU.
pub const MODEL_RESOURCE: &str = "raichu-model.json";
