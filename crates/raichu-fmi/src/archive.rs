use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::{Description, FmiError, Instance};

const MAX_ENTRIES: u64 = 10_000;
const MAX_UNCOMPRESSED: u64 = 512 * 1024 * 1024;

/// An FMU unpacked into a private directory, with its native library still unloaded.
#[derive(Clone)]
pub struct Archive {
    directory: Arc<tempfile::TempDir>,
    description: Arc<Description>,
    sha256: Arc<str>,
}

impl Archive {
    /// Read and securely unpack an `.fmu`, then parse `modelDescription.xml`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FmiError> {
        let mut source = File::open(path)?;
        let mut staged = tempfile::tempfile()?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 8192];
        loop {
            let read = source.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            staged.write_all(&buffer[..read])?;
            hasher.update(&buffer[..read]);
        }
        let sha256 = format!("sha256:{:x}", hasher.finalize());
        staged.seek(SeekFrom::Start(0))?;
        let directory = tempfile::Builder::new().prefix("raichu-fmu-").tempdir()?;
        let mut zip = zip::ZipArchive::new(staged)?;
        if zip.len() as u64 > MAX_ENTRIES {
            return Err(FmiError::Limit {
                limit: "entry count",
                actual: zip.len() as u64,
                maximum: MAX_ENTRIES,
            });
        }
        let mut total = 0_u64;
        for index in 0..zip.len() {
            let entry = zip.by_index(index)?;
            let name = entry.name().to_owned();
            let relative = safe_path(&name, entry.unix_mode())?;
            if relative.as_os_str().is_empty() {
                continue;
            }
            let size = entry.size();
            total = total.checked_add(size).ok_or(FmiError::Limit {
                limit: "uncompressed bytes",
                actual: u64::MAX,
                maximum: MAX_UNCOMPRESSED,
            })?;
            if total > MAX_UNCOMPRESSED {
                return Err(FmiError::Limit {
                    limit: "uncompressed bytes",
                    actual: total,
                    maximum: MAX_UNCOMPRESSED,
                });
            }
            let target = directory.path().join(relative);
            if entry.is_dir() {
                fs::create_dir_all(target)?;
            } else {
                if let Some(parent) = target.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut destination = File::create(target)?;
                let written = std::io::copy(
                    &mut entry.take(MAX_UNCOMPRESSED - (total - size) + 1),
                    &mut destination,
                )?;
                if written != size {
                    return Err(FmiError::Description(format!(
                        "entry `{name}` has inconsistent uncompressed size"
                    )));
                }
                destination.flush()?;
            }
        }
        let xml = fs::read_to_string(directory.path().join("modelDescription.xml"))?;
        let description = Description::parse(&xml)?;
        Ok(Self {
            directory: Arc::new(directory),
            description: Arc::new(description),
            sha256: Arc::from(sha256),
        })
    }

    /// Parsed model description, available without loading native code.
    pub fn description(&self) -> &Description {
        &self.description
    }

    /// SHA-256 digest of the original FMU bytes, prefixed by `sha256:`.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Private unpacked root, retained for the lifetime of this archive.
    pub fn path(&self) -> &Path {
        self.directory.path()
    }

    pub(crate) fn keep_directory(&self) -> Arc<tempfile::TempDir> {
        Arc::clone(&self.directory)
    }

    /// Load the platform library and initialise one co-simulation instance.
    pub fn instantiate(&self, unit_name: &str, start_time: f64) -> Result<Instance, FmiError> {
        let mut instance = self.instantiate_uninitialized(unit_name)?;
        instance.initialize(start_time)?;
        Ok(instance)
    }

    /// Load and instantiate without initialization, so fixed parameters can be set first.
    pub fn instantiate_uninitialized(&self, unit_name: &str) -> Result<Instance, FmiError> {
        Instance::new(self, unit_name)
    }
}

fn safe_path(name: &str, mode: Option<u32>) -> Result<PathBuf, FmiError> {
    let invalid = |reason: &str| FmiError::UnsafeEntry {
        entry: name.to_owned(),
        reason: reason.to_owned(),
    };
    if name.contains('\\') || name.contains('\0') {
        return Err(invalid("backslash or NUL"));
    }
    let path = Path::new(name);
    if path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(invalid("absolute or non-normal path"));
    }
    if let Some(mode) = mode {
        let kind = mode & 0o170000;
        if kind != 0 && kind != 0o100000 && kind != 0o040000 {
            return Err(invalid("symlink or non-regular entry"));
        }
    }
    Ok(path.to_path_buf())
}
