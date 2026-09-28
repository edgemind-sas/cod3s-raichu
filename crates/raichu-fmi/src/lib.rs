//! FMI 2.0 and 3.0 co-simulation archive reading and native instance hosting.
//!
//! Loading an FMU executes untrusted native code in the current process. Callers
//! must obtain explicit run permission before opening an archive or instance.

mod archive;
mod description;
mod instance;

pub use archive::Archive;
pub use description::{Description, FmiVersion, Variable, VariableKind};
pub use instance::{FmuState, Instance, Value};

/// A refusal while reading or executing a co-simulation unit.
#[derive(Debug, thiserror::Error)]
pub enum FmiError {
    /// The archive could not be read.
    #[error("FMU archive I/O: {0}")]
    Io(#[from] std::io::Error),
    /// The ZIP container is malformed.
    #[error("FMU ZIP archive: {0}")]
    Zip(#[from] zip::result::ZipError),
    /// An entry could escape its private extraction directory or has an unsafe type.
    #[error("unsafe FMU archive entry `{entry}`: {reason}")]
    UnsafeEntry {
        /// Archive entry name.
        entry: String,
        /// Rule violated by this entry.
        reason: String,
    },
    /// A configured extraction limit was reached.
    #[error("FMU archive exceeds {limit} limit ({actual} > {maximum})")]
    Limit {
        /// Limit kind.
        limit: &'static str,
        /// Observed size or count.
        actual: u64,
        /// Maximum size or count.
        maximum: u64,
    },
    /// The model description is invalid or outside this reader's scope.
    #[error("FMU model description: {0}")]
    Description(String),
    /// No native library is available for this platform.
    #[error("unit `{unit}` has no FMU binary for `{platform}`")]
    MissingBinary {
        /// Declared unit name.
        unit: String,
        /// FMI binaries folder sought.
        platform: String,
    },
    /// Loading a native library or one of its required symbols failed.
    #[error("unit `{unit}` native library: {detail}")]
    Library {
        /// Declared unit name.
        unit: String,
        /// Loader detail.
        detail: String,
    },
    /// The FMU refused instantiation, often because the token differs.
    #[error("unit `{unit}` refused instantiation (check instantiation token): {log}")]
    Instantiate {
        /// Declared unit name.
        unit: String,
        /// Last diagnostic from the FMU, if any.
        log: String,
    },
    /// An FMI function returned discard, error, or fatal.
    #[error("unit `{unit}` {operation} returned status {status} at t={time}: {log}")]
    Status {
        /// Declared unit name.
        unit: String,
        /// FMI operation.
        operation: &'static str,
        /// FMI status code.
        status: i32,
        /// Current simulation time.
        time: f64,
        /// Last FMU diagnostic.
        log: String,
    },
    /// A requested FMI capability or variable type is absent.
    #[error("unit `{unit}`: {detail}")]
    Unsupported {
        /// Declared unit name.
        unit: String,
        /// Missing capability or type.
        detail: String,
    },
}
