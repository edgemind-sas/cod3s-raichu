use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;
use std::sync::Arc;

use libloading::Library;

use crate::{Archive, FmiError, FmiVersion, Variable, VariableKind};

type Component = *mut c_void;
type Status = c_int;
type Set2<T> = unsafe extern "C" fn(Component, *const u32, usize, *const T) -> Status;
type Get2<T> = unsafe extern "C" fn(Component, *const u32, usize, *mut T) -> Status;
type Set3<T> = unsafe extern "C" fn(Component, *const u32, usize, *const T, usize) -> Status;
type Get3<T> = unsafe extern "C" fn(Component, *const u32, usize, *mut T, usize) -> Status;
type Step2 = unsafe extern "C" fn(Component, f64, f64, c_int) -> Status;
type Step3 =
    unsafe extern "C" fn(Component, f64, f64, u8, *mut u8, *mut u8, *mut u8, *mut f64) -> Status;

#[derive(Default)]
struct Functions {
    set2_real: Option<Set2<f64>>,
    set2_integer: Option<Set2<i32>>,
    set2_boolean: Option<Set2<i32>>,
    get2_real: Option<Get2<f64>>,
    get2_integer: Option<Get2<i32>>,
    get2_boolean: Option<Get2<i32>>,
    step2: Option<Step2>,
    set3_float64: Option<Set3<f64>>,
    set3_int32: Option<Set3<i32>>,
    set3_int64: Option<Set3<i64>>,
    set3_boolean: Option<Set3<u8>>,
    get3_float64: Option<Get3<f64>>,
    get3_int32: Option<Get3<i32>>,
    get3_int64: Option<Get3<i64>>,
    get3_boolean: Option<Get3<u8>>,
    step3: Option<Step3>,
}

impl Functions {
    unsafe fn load(library: &Library, version: FmiVersion) -> Self {
        let mut functions = Self::default();
        if version == FmiVersion::V2 {
            functions.set2_real = optional_symbol(library, b"fmi2SetReal\0");
            functions.set2_integer = optional_symbol(library, b"fmi2SetInteger\0");
            functions.set2_boolean = optional_symbol(library, b"fmi2SetBoolean\0");
            functions.get2_real = optional_symbol(library, b"fmi2GetReal\0");
            functions.get2_integer = optional_symbol(library, b"fmi2GetInteger\0");
            functions.get2_boolean = optional_symbol(library, b"fmi2GetBoolean\0");
            functions.step2 = optional_symbol(library, b"fmi2DoStep\0");
        } else {
            functions.set3_float64 = optional_symbol(library, b"fmi3SetFloat64\0");
            functions.set3_int32 = optional_symbol(library, b"fmi3SetInt32\0");
            functions.set3_int64 = optional_symbol(library, b"fmi3SetInt64\0");
            functions.set3_boolean = optional_symbol(library, b"fmi3SetBoolean\0");
            functions.get3_float64 = optional_symbol(library, b"fmi3GetFloat64\0");
            functions.get3_int32 = optional_symbol(library, b"fmi3GetInt32\0");
            functions.get3_int64 = optional_symbol(library, b"fmi3GetInt64\0");
            functions.get3_boolean = optional_symbol(library, b"fmi3GetBoolean\0");
            functions.step3 = optional_symbol(library, b"fmi3DoStep\0");
        }
        functions
    }
}

/// A scalar value exchanged with an FMU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    /// 64-bit floating-point value.
    Float(f64),
    /// Integer value.
    Int(i64),
    /// Boolean value.
    Bool(bool),
}

/// Portable serialized FMU state, suitable for a snapshot.
pub type FmuState = Vec<u8>;

#[repr(C)]
struct Fmi2Callbacks {
    logger: Option<
        unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *const c_char, *const c_char, ...),
    >,
    allocate: Option<unsafe extern "C" fn(usize, usize) -> *mut c_void>,
    free: Option<unsafe extern "C" fn(*mut c_void)>,
    step_finished: Option<unsafe extern "C" fn(*mut c_void, c_int)>,
    environment: *mut c_void,
}

unsafe extern "C" {
    fn raichu_fmi2_logger(
        env: *mut c_void,
        instance: *const c_char,
        status: c_int,
        category: *const c_char,
        format: *const c_char,
        ...
    );
    fn calloc(count: usize, size: usize) -> *mut c_void;
    fn free(pointer: *mut c_void);
}

#[derive(Default)]
struct LogBuffer {
    last: String,
}

#[unsafe(no_mangle)]
extern "C" fn raichu_fmi_record_log(environment: *mut c_void, message: *const c_char) {
    if environment.is_null() || message.is_null() {
        return;
    }
    // SAFETY: The environment is the boxed log buffer kept alive by Instance.
    let log = unsafe { &mut *environment.cast::<LogBuffer>() };
    // SAFETY: FMI callback strings are NUL terminated.
    log.last = unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned();
}

unsafe extern "C" fn fmi3_logger(
    environment: *mut c_void,
    _status: c_int,
    _category: *const c_char,
    message: *const c_char,
) {
    raichu_fmi_record_log(environment, message);
}

/// One native co-simulation instance. It is intentionally not `Send` or `Sync`.
pub struct Instance {
    component: Component,
    library: Library,
    functions: Functions,
    version: FmiVersion,
    unit: String,
    time: f64,
    log: Box<LogBuffer>,
    _fmi2_callbacks: Option<Box<Fmi2Callbacks>>,
    _directory: Arc<tempfile::TempDir>,
}

impl Instance {
    pub(crate) fn new(archive: &Archive, unit: &str) -> Result<Self, FmiError> {
        let desc = archive.description();
        let version = desc.fmi_version;
        let platform = platform_folder(version)?;
        let filename = format!("{}{}", desc.model_identifier, std::env::consts::DLL_SUFFIX);
        let mut candidates = vec![archive
            .path()
            .join("binaries")
            .join(platform)
            .join(&filename)];
        let prefixed = format!("{}{}", std::env::consts::DLL_PREFIX, filename);
        candidates.push(
            archive
                .path()
                .join("binaries")
                .join(platform)
                .join(&prefixed),
        );
        if version == FmiVersion::V2 {
            candidates.push(
                archive
                    .path()
                    .join("binaries")
                    .join(platform_folder(FmiVersion::V3)?)
                    .join(&filename),
            );
            candidates.push(
                archive
                    .path()
                    .join("binaries")
                    .join(platform_folder(FmiVersion::V3)?)
                    .join(&prefixed),
            );
        }
        let mut library = None;
        for path in candidates {
            // SAFETY: The caller explicitly authorized opening this FMU.
            match unsafe { Library::new(&path) } {
                Ok(opened) => {
                    library = Some(opened);
                    break;
                }
                Err(error) if path.is_file() => {
                    return Err(FmiError::Library {
                        unit: unit.into(),
                        detail: error.to_string(),
                    });
                }
                Err(_) => {}
            }
        }
        let library = library.ok_or_else(|| FmiError::MissingBinary {
            unit: unit.into(),
            platform: platform.into(),
        })?;
        // SAFETY: Function pointers remain valid while Instance owns the library.
        let functions = unsafe { Functions::load(&library, version) };
        let name = CString::new(unit).map_err(|_| FmiError::Unsupported {
            unit: unit.into(),
            detail: "unit name contains NUL".into(),
        })?;
        let token = CString::new(desc.token.as_str())
            .map_err(|_| FmiError::Description("instantiation token contains NUL".into()))?;
        let mut log = Box::<LogBuffer>::default();
        let environment = (&mut *log as *mut LogBuffer).cast::<c_void>();
        let resource = resource_path(archive, version)?;
        let callbacks = if version == FmiVersion::V2 {
            Some(Box::new(Fmi2Callbacks {
                logger: Some(raichu_fmi2_logger),
                allocate: Some(calloc),
                free: Some(free),
                step_finished: None,
                environment,
            }))
        } else {
            None
        };
        let component = unsafe {
            if version == FmiVersion::V2 {
                let instantiate: unsafe extern "C" fn(
                    *const c_char,
                    c_int,
                    *const c_char,
                    *const c_char,
                    *const Fmi2Callbacks,
                    c_int,
                    c_int,
                ) -> Component = symbol(&library, unit, b"fmi2Instantiate\0")?;
                instantiate(
                    name.as_ptr(),
                    1,
                    token.as_ptr(),
                    resource.as_ptr(),
                    callbacks
                        .as_deref()
                        .map_or(ptr::null(), |v| v as *const Fmi2Callbacks),
                    0,
                    0,
                )
            } else {
                let instantiate: unsafe extern "C" fn(
                    *const c_char,
                    *const c_char,
                    *const c_char,
                    u8,
                    u8,
                    u8,
                    u8,
                    *const u32,
                    usize,
                    *mut c_void,
                    Option<unsafe extern "C" fn(*mut c_void, c_int, *const c_char, *const c_char)>,
                    *const c_void,
                ) -> Component = symbol(&library, unit, b"fmi3InstantiateCoSimulation\0")?;
                instantiate(
                    name.as_ptr(),
                    token.as_ptr(),
                    resource.as_ptr(),
                    0,
                    0,
                    0,
                    0,
                    ptr::null(),
                    0,
                    environment,
                    Some(fmi3_logger),
                    ptr::null(),
                )
            }
        };
        if component.is_null() {
            return Err(FmiError::Instantiate {
                unit: unit.into(),
                log: log.last.clone(),
            });
        }
        Ok(Self {
            component,
            library,
            functions,
            version,
            unit: unit.into(),
            time: 0.0,
            log,
            _fmi2_callbacks: callbacks,
            _directory: archive.keep_directory(),
        })
    }

    /// Enter and exit initialization mode after any fixed parameter values are set.
    pub fn initialize(&mut self, start_time: f64) -> Result<(), FmiError> {
        if !start_time.is_finite() {
            return Err(self.unsupported("start time must be finite".into()));
        }
        unsafe {
            if self.version == FmiVersion::V2 {
                let setup: unsafe extern "C" fn(Component, c_int, f64, f64, c_int, f64) -> Status =
                    self.symbol(b"fmi2SetupExperiment\0")?;
                self.check(
                    "setup experiment",
                    setup(self.component, 0, 0.0, start_time, 0, 0.0),
                )?;
                let enter: unsafe extern "C" fn(Component) -> Status =
                    self.symbol(b"fmi2EnterInitializationMode\0")?;
                self.check("enter initialization", enter(self.component))?;
                let exit: unsafe extern "C" fn(Component) -> Status =
                    self.symbol(b"fmi2ExitInitializationMode\0")?;
                self.check("exit initialization", exit(self.component))?;
            } else {
                let enter: unsafe extern "C" fn(Component, u8, f64, f64, u8, f64) -> Status =
                    self.symbol(b"fmi3EnterInitializationMode\0")?;
                self.check(
                    "enter initialization",
                    enter(self.component, 0, 0.0, start_time, 0, 0.0),
                )?;
                let exit: unsafe extern "C" fn(Component) -> Status =
                    self.symbol(b"fmi3ExitInitializationMode\0")?;
                self.check("exit initialization", exit(self.component))?;
            }
        }
        self.time = start_time;
        Ok(())
    }

    /// Set one scalar by its parsed variable definition.
    pub fn set_value(&mut self, variable: &Variable, value: Value) -> Result<(), FmiError> {
        let vr = variable.value_reference;
        unsafe {
            match (&variable.kind, value, self.version) {
                (VariableKind::Float64, Value::Float(v), FmiVersion::V2) => {
                    let f = self.cached_symbol(self.functions.set2_real, b"fmi2SetReal\0")?;
                    self.check("set real", f(self.component, &vr, 1, &v))
                }
                (VariableKind::Float64, Value::Float(v), FmiVersion::V3) => {
                    let f = self.cached_symbol(self.functions.set3_float64, b"fmi3SetFloat64\0")?;
                    self.check("set Float64", f(self.component, &vr, 1, &v, 1))
                }
                (VariableKind::Int64, Value::Int(v), FmiVersion::V2) => {
                    let narrowed = i32::try_from(v).map_err(|_| {
                        self.unsupported(format!("integer {v} exceeds FMI 2 range"))
                    })?;
                    let f = self.cached_symbol(self.functions.set2_integer, b"fmi2SetInteger\0")?;
                    self.check("set integer", f(self.component, &vr, 1, &narrowed))
                }
                (VariableKind::Int64, Value::Int(v), FmiVersion::V3)
                    if variable.integer_width == Some(32) =>
                {
                    let narrowed = i32::try_from(v).map_err(|_| {
                        self.unsupported(format!("integer {v} exceeds FMI 3 Int32 range"))
                    })?;
                    let f = self.cached_symbol(self.functions.set3_int32, b"fmi3SetInt32\0")?;
                    self.check("set Int32", f(self.component, &vr, 1, &narrowed, 1))
                }
                (VariableKind::Int64, Value::Int(v), FmiVersion::V3) => {
                    let f = self.cached_symbol(self.functions.set3_int64, b"fmi3SetInt64\0")?;
                    self.check("set Int64", f(self.component, &vr, 1, &v, 1))
                }
                (VariableKind::Boolean, Value::Bool(v), FmiVersion::V2) => {
                    let value = i32::from(v);
                    let f = self.cached_symbol(self.functions.set2_boolean, b"fmi2SetBoolean\0")?;
                    self.check("set Boolean", f(self.component, &vr, 1, &value))
                }
                (VariableKind::Boolean, Value::Bool(v), FmiVersion::V3) => {
                    let value = u8::from(v);
                    let f = self.cached_symbol(self.functions.set3_boolean, b"fmi3SetBoolean\0")?;
                    self.check("set Boolean", f(self.component, &vr, 1, &value, 1))
                }
                _ => Err(self.unsupported(format!(
                    "variable `{}` cannot receive {value:?}",
                    variable.name
                ))),
            }
        }
    }

    /// Read one scalar by its parsed variable definition.
    pub fn get_value(&self, variable: &Variable) -> Result<Value, FmiError> {
        let vr = variable.value_reference;
        unsafe {
            match (&variable.kind, self.version) {
                (VariableKind::Float64, FmiVersion::V2) => {
                    let mut v = 0.0;
                    let f = self.cached_symbol(self.functions.get2_real, b"fmi2GetReal\0")?;
                    self.check("get real", f(self.component, &vr, 1, &mut v))?;
                    Ok(Value::Float(v))
                }
                (VariableKind::Float64, FmiVersion::V3) => {
                    let mut v = 0.0;
                    let f = self.cached_symbol(self.functions.get3_float64, b"fmi3GetFloat64\0")?;
                    self.check("get Float64", f(self.component, &vr, 1, &mut v, 1))?;
                    Ok(Value::Float(v))
                }
                (VariableKind::Int64, FmiVersion::V2) => {
                    let mut v = 0_i32;
                    let f = self.cached_symbol(self.functions.get2_integer, b"fmi2GetInteger\0")?;
                    self.check("get integer", f(self.component, &vr, 1, &mut v))?;
                    Ok(Value::Int(i64::from(v)))
                }
                (VariableKind::Int64, FmiVersion::V3) if variable.integer_width == Some(32) => {
                    let mut v = 0_i32;
                    let f = self.cached_symbol(self.functions.get3_int32, b"fmi3GetInt32\0")?;
                    self.check("get Int32", f(self.component, &vr, 1, &mut v, 1))?;
                    Ok(Value::Int(i64::from(v)))
                }
                (VariableKind::Int64, FmiVersion::V3) => {
                    let mut v = 0_i64;
                    let f = self.cached_symbol(self.functions.get3_int64, b"fmi3GetInt64\0")?;
                    self.check("get Int64", f(self.component, &vr, 1, &mut v, 1))?;
                    Ok(Value::Int(v))
                }
                (VariableKind::Boolean, FmiVersion::V2) => {
                    let mut v = 0_i32;
                    let f = self.cached_symbol(self.functions.get2_boolean, b"fmi2GetBoolean\0")?;
                    self.check("get Boolean", f(self.component, &vr, 1, &mut v))?;
                    Ok(Value::Bool(v != 0))
                }
                (VariableKind::Boolean, FmiVersion::V3) => {
                    let mut v = 0_u8;
                    let f = self.cached_symbol(self.functions.get3_boolean, b"fmi3GetBoolean\0")?;
                    self.check("get Boolean", f(self.component, &vr, 1, &mut v, 1))?;
                    Ok(Value::Bool(v != 0))
                }
                _ => {
                    Err(self
                        .unsupported(format!("variable `{}` has unsupported type", variable.name)))
                }
            }
        }
    }

    /// Advance by one strictly positive communication interval.
    pub fn do_step(&mut self, current_time: f64, step_size: f64) -> Result<(), FmiError> {
        if !current_time.is_finite() || !step_size.is_finite() || step_size <= 0.0 {
            return Err(self.unsupported("communication step must be finite and positive".into()));
        }
        unsafe {
            if self.version == FmiVersion::V2 {
                let f = self.cached_symbol(self.functions.step2, b"fmi2DoStep\0")?;
                self.check("do step", f(self.component, current_time, step_size, 0))?;
            } else {
                let f = self.cached_symbol(self.functions.step3, b"fmi3DoStep\0")?;
                let (mut event, mut terminate, mut early, mut last) =
                    (0_u8, 0_u8, 0_u8, current_time);
                self.check(
                    "do step",
                    f(
                        self.component,
                        current_time,
                        step_size,
                        0,
                        &mut event,
                        &mut terminate,
                        &mut early,
                        &mut last,
                    ),
                )?;
                if event != 0 || terminate != 0 || early != 0 {
                    return Err(self.unsupported(format!(
                        "FMI 3 step requested event mode, termination, or early return at t={last}"
                    )));
                }
            }
        }
        self.time = current_time + step_size;
        Ok(())
    }

    /// Reset and reinitialise this instance for another trajectory.
    pub fn reset(&mut self, start_time: f64) -> Result<(), FmiError> {
        self.reset_uninitialized()?;
        self.initialize(start_time)
    }

    /// Reset without initialization, so fixed parameter values can be reapplied.
    pub fn reset_uninitialized(&mut self) -> Result<(), FmiError> {
        unsafe {
            let name = if self.version == FmiVersion::V2 {
                b"fmi2Reset\0".as_slice()
            } else {
                b"fmi3Reset\0".as_slice()
            };
            let f: unsafe extern "C" fn(Component) -> Status = self.symbol(name)?;
            self.check("reset", f(self.component))?;
        }
        Ok(())
    }

    /// Terminate a completed trajectory.
    pub fn terminate(&mut self) -> Result<(), FmiError> {
        unsafe {
            let name = if self.version == FmiVersion::V2 {
                b"fmi2Terminate\0".as_slice()
            } else {
                b"fmi3Terminate\0".as_slice()
            };
            let f: unsafe extern "C" fn(Component) -> Status = self.symbol(name)?;
            self.check("terminate", f(self.component))
        }
    }

    /// Serialize FMU state for a checkpoint.
    pub fn get_state(&self) -> Result<FmuState, FmiError> {
        unsafe {
            let prefix = if self.version == FmiVersion::V2 {
                "fmi2"
            } else {
                "fmi3"
            };
            let state = self.native_state(
                format!(
                    "{prefix}GetFMU{}",
                    if self.version == FmiVersion::V2 {
                        "state"
                    } else {
                        "State"
                    }
                )
                .as_bytes(),
            )?;
            let result = (|| {
                let size_name = format!(
                    "{prefix}SerializedFMU{}Size\0",
                    if self.version == FmiVersion::V2 {
                        "state"
                    } else {
                        "State"
                    }
                );
                let size_f: unsafe extern "C" fn(Component, Component, *mut usize) -> Status =
                    self.symbol(size_name.as_bytes())?;
                let mut size = 0_usize;
                self.check(
                    "serialized state size",
                    size_f(self.component, state, &mut size),
                )?;
                if size > 512 * 1024 * 1024 {
                    return Err(self.unsupported("serialized state exceeds 512 MiB".into()));
                }
                let mut bytes = vec![0_u8; size];
                let serialize_name = format!(
                    "{prefix}SerializeFMU{}\0",
                    if self.version == FmiVersion::V2 {
                        "state"
                    } else {
                        "State"
                    }
                );
                let serialize: unsafe extern "C" fn(
                    Component,
                    Component,
                    *mut u8,
                    usize,
                ) -> Status = self.symbol(serialize_name.as_bytes())?;
                self.check(
                    "serialize state",
                    serialize(self.component, state, bytes.as_mut_ptr(), size),
                )?;
                Ok(bytes)
            })();
            self.free_native_state(state)?;
            result
        }
    }

    /// Restore an owned serialized state into this instance.
    pub fn set_state(&mut self, bytes: &[u8]) -> Result<(), FmiError> {
        unsafe {
            let mut state: Component = ptr::null_mut();
            let name = if self.version == FmiVersion::V2 {
                b"fmi2DeSerializeFMUstate\0".as_slice()
            } else {
                b"fmi3DeserializeFMUState\0".as_slice()
            };
            let deserialize: unsafe extern "C" fn(
                Component,
                *const u8,
                usize,
                *mut Component,
            ) -> Status = self.symbol(name)?;
            self.check(
                "deserialize state",
                deserialize(self.component, bytes.as_ptr(), bytes.len(), &mut state),
            )?;
            let result = {
                let name = if self.version == FmiVersion::V2 {
                    b"fmi2SetFMUstate\0".as_slice()
                } else {
                    b"fmi3SetFMUState\0".as_slice()
                };
                let set: unsafe extern "C" fn(Component, Component) -> Status =
                    self.symbol(name)?;
                self.check("set state", set(self.component, state))
            };
            self.free_native_state(state)?;
            result
        }
    }

    unsafe fn native_state(&self, raw_name: &[u8]) -> Result<Component, FmiError> {
        let mut name = raw_name.to_vec();
        name.push(0);
        let get: unsafe extern "C" fn(Component, *mut Component) -> Status = self.symbol(&name)?;
        let mut state = ptr::null_mut();
        self.check("get state", get(self.component, &mut state))?;
        Ok(state)
    }

    unsafe fn free_native_state(&self, mut state: Component) -> Result<(), FmiError> {
        let name = if self.version == FmiVersion::V2 {
            b"fmi2FreeFMUstate\0".as_slice()
        } else {
            b"fmi3FreeFMUState\0".as_slice()
        };
        let free: unsafe extern "C" fn(Component, *mut Component) -> Status = self.symbol(name)?;
        self.check("free state", free(self.component, &mut state))
    }

    unsafe fn symbol<T: Copy>(&self, name: &[u8]) -> Result<T, FmiError> {
        symbol(&self.library, &self.unit, name)
    }

    fn cached_symbol<T: Copy>(&self, cached: Option<T>, name: &[u8]) -> Result<T, FmiError> {
        cached.map_or_else(|| unsafe { self.symbol(name) }, Ok)
    }

    fn check(&self, operation: &'static str, status: Status) -> Result<(), FmiError> {
        if status == 0 || status == 1 {
            Ok(())
        } else {
            Err(FmiError::Status {
                unit: self.unit.clone(),
                operation,
                status,
                time: self.time,
                log: self.log.last.clone(),
            })
        }
    }
    fn unsupported(&self, detail: String) -> FmiError {
        FmiError::Unsupported {
            unit: self.unit.clone(),
            detail,
        }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        if self.component.is_null() {
            return;
        }
        let name = if self.version == FmiVersion::V2 {
            b"fmi2FreeInstance\0".as_slice()
        } else {
            b"fmi3FreeInstance\0".as_slice()
        };
        // SAFETY: The library remains loaded until after this destructor returns.
        if let Ok(free) = unsafe { self.symbol::<unsafe extern "C" fn(Component)>(name) } {
            unsafe { free(self.component) };
        }
        self.component = ptr::null_mut();
    }
}

unsafe fn symbol<T: Copy>(library: &Library, unit: &str, name: &[u8]) -> Result<T, FmiError> {
    library
        .get::<T>(name)
        .map(|sym| *sym)
        .map_err(|e| FmiError::Library {
            unit: unit.into(),
            detail: e.to_string(),
        })
}

unsafe fn optional_symbol<T: Copy>(library: &Library, name: &[u8]) -> Option<T> {
    library.get::<T>(name).ok().map(|symbol| *symbol)
}

fn resource_path(archive: &Archive, version: FmiVersion) -> Result<CString, FmiError> {
    let path = archive.path().join("resources");
    let value = if version == FmiVersion::V2 {
        format!("file://{}", path.to_string_lossy())
    } else {
        path.to_string_lossy().into_owned()
    };
    CString::new(value).map_err(|_| FmiError::Description("resource path contains NUL".into()))
}

fn platform_folder(version: FmiVersion) -> Result<&'static str, FmiError> {
    let folder = match (version, std::env::consts::OS, std::env::consts::ARCH) {
        (FmiVersion::V2, "linux", "x86_64") => "linux64",
        (FmiVersion::V2, "windows", "x86_64") => "win64",
        (FmiVersion::V2, "macos", "aarch64") => "darwin64",
        (FmiVersion::V2, "macos", "x86_64") => "darwin64",
        (FmiVersion::V3, "linux", "x86_64") => "x86_64-linux",
        (FmiVersion::V3, "linux", "aarch64") => "aarch64-linux",
        (FmiVersion::V3, "windows", "x86_64") => "x86_64-windows",
        (FmiVersion::V3, "macos", "aarch64") => "aarch64-darwin",
        (FmiVersion::V3, "macos", "x86_64") => "x86_64-darwin",
        _ => {
            return Err(FmiError::Description(format!(
                "platform {}-{} has no FMI binary mapping",
                std::env::consts::ARCH,
                std::env::consts::OS
            )));
        }
    };
    Ok(folder)
}
