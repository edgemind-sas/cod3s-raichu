#![allow(non_snake_case)]

use std::ffi::{c_char, c_int, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr;

use raichu_expr::Value;

use crate::instance::{Instance, Logger, Phase};
use crate::MODEL_RESOURCE;

const OK: c_int = 0;
const ERROR: c_int = 3;
type Handle = *mut c_void;

fn status(
    handle: Handle,
    operation: &str,
    f: impl FnOnce(&mut Instance) -> Result<(), String>,
) -> c_int {
    if handle.is_null() {
        return ERROR;
    }
    // SAFETY: A non-null FMI handle must be an active instance returned by this library.
    let instance = unsafe { &mut *handle.cast::<Instance>() };
    match catch_unwind(AssertUnwindSafe(|| f(instance))) {
        Ok(Ok(())) => OK,
        Ok(Err(error)) => {
            instance.log(&format!("{operation}: {error}"));
            ERROR
        }
        Err(_) => {
            instance.phase = Phase::Error;
            instance.log(&format!("{operation}: internal panic"));
            ERROR
        }
    }
}

fn required_string(pointer: *const c_char, name: &str) -> Result<String, String> {
    if pointer.is_null() {
        return Err(format!("null {name}"));
    }
    // SAFETY: FMI requires NUL-terminated strings at non-null string arguments.
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map(str::to_owned)
        .map_err(|_| format!("invalid UTF-8 {name}"))
}

/// FMI 3 version string.
#[no_mangle]
pub extern "C" fn fmi3GetVersion() -> *const c_char {
    c"3.0".as_ptr()
}

/// Model exchange is outside this runtime's declared interface.
#[no_mangle]
pub unsafe extern "C" fn fmi3InstantiateModelExchange(
    _instanceName: *const c_char,
    _instantiationToken: *const c_char,
    _resourcePath: *const c_char,
    _visible: bool,
    _loggingOn: bool,
    _instanceEnvironment: *mut c_void,
    _logMessage: Option<Logger>,
) -> Handle {
    ptr::null_mut()
}

/// Scheduled execution is outside this runtime's declared interface.
#[no_mangle]
pub unsafe extern "C" fn fmi3InstantiateScheduledExecution(
    _instanceName: *const c_char,
    _instantiationToken: *const c_char,
    _resourcePath: *const c_char,
    _visible: bool,
    _loggingOn: bool,
    _instanceEnvironment: *mut c_void,
    _logMessage: Option<Logger>,
    _clockUpdate: Option<unsafe extern "C" fn(*mut c_void)>,
    _lockPreemption: Option<unsafe extern "C" fn()>,
    _unlockPreemption: Option<unsafe extern "C" fn()>,
) -> Handle {
    ptr::null_mut()
}

/// Instantiate one independent native co-simulation engine.
#[no_mangle]
pub unsafe extern "C" fn fmi3InstantiateCoSimulation(
    instanceName: *const c_char,
    instantiationToken: *const c_char,
    resourcePath: *const c_char,
    _visible: bool,
    loggingOn: bool,
    eventModeUsed: bool,
    earlyReturnAllowed: bool,
    _requiredIntermediateVariables: *const u32,
    nRequiredIntermediateVariables: usize,
    instanceEnvironment: *mut c_void,
    logMessage: Option<Logger>,
    _intermediateUpdate: Option<
        unsafe extern "C" fn(*mut c_void, f64, bool, bool, bool, bool, *mut bool, *mut f64),
    >,
) -> Handle {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<Handle, String> {
        let name = required_string(instanceName, "instance name")?;
        if name.is_empty() {
            return Err("empty instance name".into());
        }
        let supplied_token = required_string(instantiationToken, "instantiation token")?;
        let resource = required_string(resourcePath, "resource path")?;
        if resource.is_empty() || !Path::new(&resource).is_dir() {
            return Err("resource path must be a native directory".into());
        }
        if eventModeUsed || earlyReturnAllowed || nRequiredIntermediateVariables != 0 {
            return Err(
                "event mode, early return and intermediate variables are unavailable".into(),
            );
        }
        let document = std::fs::read_to_string(Path::new(&resource).join(MODEL_RESOURCE))
            .map_err(|e| e.to_string())?;
        let instance = Instance::new(&document, logMessage, instanceEnvironment, loggingOn)?;
        if instance.token != supplied_token {
            return Err("instantiation token does not match model resource".into());
        }
        Ok(Box::into_raw(Box::new(instance)).cast())
    }));
    match result {
        Ok(Ok(handle)) => handle,
        Ok(Err(message)) => {
            if let Some(logger) = logMessage {
                if let Ok(message) = std::ffi::CString::new(message.replace('\0', " ")) {
                    // SAFETY: callback is supplied by importer for this invocation.
                    unsafe {
                        logger(
                            instanceEnvironment,
                            ERROR,
                            c"error".as_ptr(),
                            message.as_ptr(),
                        )
                    }
                }
            }
            ptr::null_mut()
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Release an FMU instance. Null is accepted as a no-op.
#[no_mangle]
pub unsafe extern "C" fn fmi3FreeInstance(instance: Handle) {
    if !instance.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: FMI transfers its valid instance handle back exactly once.
            unsafe { drop(Box::from_raw(instance.cast::<Instance>())) }
        }));
    }
}

/// Configure diagnostic logging.
#[no_mangle]
pub unsafe extern "C" fn fmi3SetDebugLogging(
    instance: Handle,
    loggingOn: bool,
    nCategories: usize,
    _categories: *const *const c_char,
) -> c_int {
    status(instance, "set debug logging", |i| {
        if nCategories != 0 {
            return Err("log category selection is unavailable".into());
        }
        i.logging_on = loggingOn;
        Ok(())
    })
}

/// Begin FMI initialization.
#[no_mangle]
pub unsafe extern "C" fn fmi3EnterInitializationMode(
    instance: Handle,
    _toleranceDefined: bool,
    _tolerance: f64,
    startTime: f64,
    stopTimeDefined: bool,
    stopTime: f64,
) -> c_int {
    status(instance, "enter initialization", |i| {
        i.start(startTime, stopTimeDefined.then_some(stopTime))
    })
}

/// Enter step mode after initialization.
#[no_mangle]
pub unsafe extern "C" fn fmi3ExitInitializationMode(instance: Handle) -> c_int {
    status(instance, "exit initialization", |i| {
        if i.phase != Phase::Initialization {
            return Err("invalid lifecycle state".into());
        }
        i.phase = Phase::Step;
        Ok(())
    })
}

/// Complete a co-simulation communication step.
#[no_mangle]
pub unsafe extern "C" fn fmi3DoStep(
    instance: Handle,
    currentCommunicationPoint: f64,
    communicationStepSize: f64,
    _noSetFMUStatePriorToCurrentPoint: bool,
    eventHandlingNeeded: *mut bool,
    terminateSimulation: *mut bool,
    earlyReturn: *mut bool,
    lastSuccessfulTime: *mut f64,
) -> c_int {
    status(instance, "do step", |i| {
        if eventHandlingNeeded.is_null()
            || terminateSimulation.is_null()
            || earlyReturn.is_null()
            || lastSuccessfulTime.is_null()
        {
            return Err("null step output pointer".into());
        }
        let time = i.step(currentCommunicationPoint, communicationStepSize)?;
        // SAFETY: pointers were checked and are required writable output locations.
        unsafe {
            *eventHandlingNeeded = false;
            *terminateSimulation = false;
            *earlyReturn = false;
            *lastSuccessfulTime = time;
        }
        Ok(())
    })
}

/// Terminate a running instance.
#[no_mangle]
pub unsafe extern "C" fn fmi3Terminate(instance: Handle) -> c_int {
    status(instance, "terminate", |i| {
        if i.phase != Phase::Step {
            return Err("invalid lifecycle state".into());
        }
        i.phase = Phase::Terminated;
        Ok(())
    })
}

/// Reset an instance to its pre-initialized state.
#[no_mangle]
pub unsafe extern "C" fn fmi3Reset(instance: Handle) -> c_int {
    status(instance, "reset", |i| i.reset())
}

/// Capture an opaque native trajectory checkpoint.
#[no_mangle]
pub unsafe extern "C" fn fmi3GetFMUState(instance: Handle, FMUState: *mut Handle) -> c_int {
    status(instance, "get FMU state", |i| {
        if FMUState.is_null() {
            return Err("null state output pointer".into());
        }
        // SAFETY: the pointer was checked above.
        let previous = unsafe { *FMUState };
        if !previous.is_null() && !i.owns_state(previous) {
            return Err("unknown FMU state pointer".into());
        }
        let state = i.snapshot()?;
        if !previous.is_null() {
            i.free_state(previous);
        }
        // SAFETY: pointer checked above.
        unsafe {
            *FMUState = state;
        }
        Ok(())
    })
}

/// Rewind to an opaque checkpoint belonging to this instance.
#[no_mangle]
pub unsafe extern "C" fn fmi3SetFMUState(instance: Handle, FMUState: Handle) -> c_int {
    status(instance, "set FMU state", |i| i.restore(FMUState))
}

/// Free a checkpoint belonging to this instance.
#[no_mangle]
pub unsafe extern "C" fn fmi3FreeFMUState(instance: Handle, FMUState: *mut Handle) -> c_int {
    status(instance, "free FMU state", |i| {
        if FMUState.is_null() {
            return Ok(());
        }
        // SAFETY: output pointer checked above.
        let pointer = unsafe { *FMUState };
        if pointer.is_null() {
            return Ok(());
        }
        if !i.free_state(pointer) {
            return Err("unknown FMU state pointer".into());
        }
        unsafe {
            *FMUState = ptr::null_mut();
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)] // Mirrors FMI's scalar array ABI plus conversion hooks.
fn array_access<T: Copy>(
    instance: Handle,
    operation: &str,
    kind: &str,
    references: *const u32,
    nReferences: usize,
    values: *mut T,
    nValues: usize,
    set: bool,
    from: impl Fn(T) -> Value,
    into: impl Fn(Value) -> Option<T>,
) -> c_int {
    status(instance, operation, |i| {
        if references.is_null() || values.is_null() || nReferences == 0 || nValues != nReferences {
            return Err("null array, zero count or mismatched counts".into());
        }
        // SAFETY: pointers and counts are validated according to the FMI C contract.
        let refs = unsafe { std::slice::from_raw_parts(references, nReferences) };
        for &vr in refs {
            i.variable(vr, kind)?;
        }
        if set {
            let input = unsafe { std::slice::from_raw_parts(values, nValues) };
            for (&vr, &value) in refs.iter().zip(input) {
                i.set(vr, kind, from(value))?;
            }
        } else {
            let output = unsafe { std::slice::from_raw_parts_mut(values, nValues) };
            for (&vr, slot) in refs.iter().zip(output) {
                *slot = into(i.get(vr, kind)?).ok_or("internal value type mismatch")?;
            }
        }
        Ok(())
    })
}

macro_rules! scalar_access {
    ($get:ident, $set:ident, $ty:ty, $kind:literal, $from:expr, $into:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $get(
            instance: Handle,
            valueReferences: *const u32,
            nValueReferences: usize,
            values: *mut $ty,
            nValues: usize,
        ) -> c_int {
            array_access(
                instance,
                stringify!($get),
                $kind,
                valueReferences,
                nValueReferences,
                values,
                nValues,
                false,
                $from,
                $into,
            )
        }
        #[no_mangle]
        pub unsafe extern "C" fn $set(
            instance: Handle,
            valueReferences: *const u32,
            nValueReferences: usize,
            values: *const $ty,
            nValues: usize,
        ) -> c_int {
            array_access(
                instance,
                stringify!($set),
                $kind,
                valueReferences,
                nValueReferences,
                values.cast_mut(),
                nValues,
                true,
                $from,
                $into,
            )
        }
    };
}

scalar_access!(
    fmi3GetFloat64,
    fmi3SetFloat64,
    f64,
    "Float64",
    Value::Float,
    |v| if let Value::Float(x) = v {
        Some(x)
    } else {
        None
    }
);
scalar_access!(
    fmi3GetInt64,
    fmi3SetInt64,
    i64,
    "Int64",
    Value::Int,
    |v| if let Value::Int(x) = v { Some(x) } else { None }
);
scalar_access!(
    fmi3GetBoolean,
    fmi3SetBoolean,
    bool,
    "Boolean",
    Value::Bool,
    |v| if let Value::Bool(x) = v {
        Some(x)
    } else {
        None
    }
);

include!("unsupported.rs");

/// Read seed and RNG substream parameters.
#[no_mangle]
pub unsafe extern "C" fn fmi3GetUInt64(
    instance: Handle,
    valueReferences: *const u32,
    nValueReferences: usize,
    values: *mut u64,
    nValues: usize,
) -> c_int {
    status(instance, "get UInt64", |i| {
        if matches!(i.phase, Phase::Error | Phase::Terminated) {
            return Err("get in invalid lifecycle state".into());
        }
        if valueReferences.is_null()
            || values.is_null()
            || nValueReferences == 0
            || nValues != nValueReferences
        {
            return Err("null array, zero count or mismatched counts".into());
        }
        let refs = unsafe { std::slice::from_raw_parts(valueReferences, nValueReferences) };
        let output = unsafe { std::slice::from_raw_parts_mut(values, nValues) };
        for (&vr, slot) in refs.iter().zip(output) {
            i.variable(vr, "UInt64")?;
            *slot = if vr == 1 { i.seed } else { i.rng_stream };
        }
        Ok(())
    })
}

/// Set seed and RNG substream before initialization completes.
#[no_mangle]
pub unsafe extern "C" fn fmi3SetUInt64(
    instance: Handle,
    valueReferences: *const u32,
    nValueReferences: usize,
    values: *const u64,
    nValues: usize,
) -> c_int {
    status(instance, "set UInt64", |i| {
        if valueReferences.is_null()
            || values.is_null()
            || nValueReferences == 0
            || nValues != nValueReferences
        {
            return Err("null array, zero count or mismatched counts".into());
        }
        if !matches!(i.phase, Phase::Instantiated | Phase::Initialization) {
            return Err("seed and rng_stream cannot change after initialization".into());
        }
        let refs = unsafe { std::slice::from_raw_parts(valueReferences, nValueReferences) };
        let input = unsafe { std::slice::from_raw_parts(values, nValues) };
        for &vr in refs {
            i.variable(vr, "UInt64")?;
        }
        for (&vr, &value) in refs.iter().zip(input) {
            i.set_seed_parameter(vr, value)?;
        }
        Ok(())
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::ffi::CString;

    use crate::{model_description, prepare_document, ExportManifest};

    fn document(distribution: &str) -> String {
        let model = format!(
            r#"{{"name":"Delay","components":[{{"name":"C","attributes":[{{"name":"out","kind":"bool","init":{{"kind":"bool","value":false}}}}],"automata":[{{"name":"A","states":["off","on"],"init":"off","transitions":[{{"name":"go","source":"off","targets":["on"],{distribution}}}]}}],"sensitive_functions":[{{"name":"reflect","effects":[{{"target":{{"component":"C","attribute":"out"}},"value":{{"op":"state_active","state":{{"component":"C","automaton":"A","state":"on"}}}}}}]}}]}}]}}"#
        );
        prepare_document(
            &model,
            &ExportManifest {
                inputs: vec![],
                outputs: vec!["C.out".into()],
                parameters: vec![],
            },
        )
        .expect("valid model")
    }

    unsafe fn new_instance(doc: &str) -> (tempfile::TempDir, Handle) {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(MODEL_RESOURCE), doc).expect("resource");
        let xml = model_description(doc).expect("description");
        assert!(xml.contains("canGetAndSetFMUState=\"true\""));
        let name = CString::new("probe").unwrap();
        let token = CString::new(crate::description::token(doc.as_bytes())).unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let instance = unsafe {
            fmi3InstantiateCoSimulation(
                name.as_ptr(),
                token.as_ptr(),
                path.as_ptr(),
                false,
                false,
                false,
                false,
                ptr::null(),
                0,
                ptr::null_mut(),
                None,
                None,
            )
        };
        assert!(!instance.is_null());
        (dir, instance)
    }

    unsafe fn initialize(instance: Handle, stop: f64) {
        assert_eq!(
            unsafe { fmi3EnterInitializationMode(instance, false, 0.0, 0.0, true, stop) },
            OK
        );
        assert_eq!(unsafe { fmi3ExitInitializationMode(instance) }, OK);
    }

    unsafe fn step(instance: Handle, at: f64, size: f64) -> c_int {
        let (mut event, mut terminate, mut early, mut last) = (false, false, false, 0.0);
        unsafe {
            fmi3DoStep(
                instance,
                at,
                size,
                false,
                &mut event,
                &mut terminate,
                &mut early,
                &mut last,
            )
        }
    }

    unsafe fn output(instance: Handle) -> bool {
        let mut value = false;
        assert_eq!(
            unsafe { fmi3GetBoolean(instance, &4, 1, &mut value, 1) },
            OK
        );
        value
    }

    #[test]
    fn delay_probe_and_argument_errors() {
        let doc = document("\"distrib\":\"delay\",\"time\":1.0");
        if let Some(path) = std::env::var_os("RAICHU_FMU_PROBE_DIR") {
            let path = std::path::PathBuf::from(path);
            std::fs::create_dir_all(&path).expect("probe directory");
            std::fs::write(path.join("raichu-model.json"), &doc).expect("probe document");
            std::fs::write(
                path.join("modelDescription.xml"),
                model_description(&doc).expect("probe XML"),
            )
            .expect("probe XML file");
        }
        let (_dir, instance) = unsafe { new_instance(&doc) };
        unsafe {
            initialize(instance, 3.0);
            assert!(!output(instance));
            assert_eq!(step(instance, 0.0, -1.0), ERROR);
            assert_eq!(fmi3GetBoolean(instance, &999, 1, &mut false, 1), ERROR);
            assert_eq!(
                fmi3GetBoolean(instance, ptr::null(), 1, &mut false, 1),
                ERROR
            );
            assert_eq!(fmi3GetBoolean(instance, &4, 0, &mut false, 0), ERROR);
            assert_eq!(fmi3SetBoolean(instance, &4, 1, &true, 1), ERROR);
            assert_eq!(step(instance, 0.0, 0.5), OK);
            assert!(!output(instance));
            assert_eq!(step(instance, 0.5, 0.5), OK);
            assert!(output(instance));
            assert_eq!(fmi3Terminate(instance), OK);
            fmi3FreeInstance(instance);
        }
    }

    #[test]
    fn interleaved_seeded_instances_and_snapshot_replay() {
        let doc = document("\"distrib\":\"exp\",\"rate\":0.7");
        if let Some(path) = std::env::var_os("RAICHU_FMU_STOCHASTIC_PROBE_DIR") {
            let path = std::path::PathBuf::from(path);
            std::fs::create_dir_all(&path).expect("probe directory");
            std::fs::write(path.join("raichu-model.json"), &doc).expect("probe document");
            std::fs::write(
                path.join("modelDescription.xml"),
                model_description(&doc).expect("probe XML"),
            )
            .expect("probe XML file");
        }
        let (_left_dir, left) = unsafe { new_instance(&doc) };
        let (_right_dir, right) = unsafe { new_instance(&doc) };
        unsafe {
            assert_eq!(fmi3SetUInt64(left, &1, 1, &42, 1), OK);
            assert_eq!(fmi3SetUInt64(right, &1, 1, &43, 1), OK);
            initialize(left, 10.0);
            initialize(right, 10.0);
            let mut state = ptr::null_mut();
            assert_eq!(fmi3GetFMUState(left, &mut state), OK);
            let mut first = Vec::new();
            for n in 0..10 {
                let t = f64::from(n);
                assert_eq!(step(left, t, 1.0), OK);
                assert_eq!(step(right, t, 1.0), OK);
                first.push(output(left));
            }
            assert_eq!(fmi3SetFMUState(left, state), OK);
            let mut replay = Vec::new();
            for n in 0..10 {
                assert_eq!(step(left, f64::from(n), 1.0), OK);
                replay.push(output(left));
            }
            assert_eq!(first, replay);
            assert_eq!(fmi3FreeFMUState(left, &mut state), OK);
            assert!(state.is_null());
            fmi3FreeInstance(left);
            fmi3FreeInstance(right);
        }
    }

    #[test]
    fn reusing_and_freeing_state_handles_follows_fmi_contract() {
        let doc = document("\"distrib\":\"delay\",\"time\":1.0");
        let (_dir, instance) = unsafe { new_instance(&doc) };
        unsafe {
            initialize(instance, 3.0);
            let mut state = ptr::null_mut();
            assert_eq!(fmi3FreeFMUState(instance, ptr::null_mut()), OK);
            assert_eq!(fmi3FreeFMUState(instance, &mut state), OK);
            for _ in 0..3 {
                assert_eq!(fmi3GetFMUState(instance, &mut state), OK);
                assert_eq!((*instance.cast::<Instance>()).checkpoint_count(), 1);
            }
            let mut unknown = 1_usize as Handle;
            assert_eq!(fmi3GetFMUState(instance, &mut unknown), ERROR);
            assert_eq!((*instance.cast::<Instance>()).checkpoint_count(), 1);
            assert_eq!(fmi3FreeFMUState(instance, &mut state), OK);
            assert!(state.is_null());
            assert_eq!((*instance.cast::<Instance>()).checkpoint_count(), 0);
            fmi3FreeInstance(instance);
        }
    }

    #[test]
    fn engine_failure_poisoned_instance() {
        let doc = document("\"distrib\":\"delay\",\"time\":1.0");
        let (_dir, instance) = unsafe { new_instance(&doc) };
        unsafe {
            initialize(instance, 1.0);
            let mut state = ptr::null_mut();
            assert_eq!(fmi3GetFMUState(instance, &mut state), OK);
            assert_eq!(step(instance, 0.0, 2.0), ERROR);
            assert_eq!(step(instance, 0.0, 0.5), ERROR);
            assert_eq!(fmi3SetFMUState(instance, state), OK);
            assert_eq!(step(instance, 0.0, 0.5), OK);
            assert_eq!(fmi3FreeFMUState(instance, &mut state), OK);
            assert_eq!(step(instance, 0.5, 1.0), ERROR);
            assert_eq!(fmi3Reset(instance), OK);
            initialize(instance, 2.0);
            assert_eq!(step(instance, 0.0, 1.0), OK);
            assert!(output(instance));
            fmi3FreeInstance(instance);
        }
    }

    #[test]
    fn initialization_checkpoint_restores_seed_parameters_and_phase() {
        let model = r#"{"name":"Checkpoint","components":[{"name":"C","attributes":[{"name":"input","kind":"int","init":{"kind":"int","value":0}},{"name":"gain","kind":"int","init":{"kind":"int","value":1}}]}]}"#;
        let doc = prepare_document(
            model,
            &ExportManifest {
                inputs: vec!["C.input".into()],
                outputs: vec![],
                parameters: vec!["C.gain".into()],
            },
        )
        .expect("manifest");
        let (_dir, instance) = unsafe { new_instance(&doc) };
        unsafe {
            assert_eq!(
                fmi3EnterInitializationMode(instance, false, 0.0, 0.0, true, 2.0),
                OK
            );
            assert_eq!(fmi3SetUInt64(instance, &1, 1, &42, 1), OK);
            assert_eq!(fmi3SetUInt64(instance, &2, 1, &7, 1), OK);
            assert_eq!(fmi3SetInt64(instance, &4, 1, &2, 1), OK);
            assert_eq!(fmi3SetInt64(instance, &5, 1, &3, 1), OK);
            let mut state = ptr::null_mut();
            assert_eq!(fmi3GetFMUState(instance, &mut state), OK);
            assert_eq!(fmi3SetUInt64(instance, &1, 1, &99, 1), OK);
            assert_eq!(fmi3SetUInt64(instance, &2, 1, &8, 1), OK);
            assert_eq!(fmi3SetInt64(instance, &4, 1, &8, 1), OK);
            assert_eq!(fmi3SetInt64(instance, &5, 1, &9, 1), OK);
            assert_eq!(fmi3SetFMUState(instance, state), OK);
            let mut seed = 0;
            let mut stream = 0;
            let mut gain = 0;
            let mut input = 0;
            assert_eq!(fmi3GetUInt64(instance, &1, 1, &mut seed, 1), OK);
            assert_eq!(fmi3GetUInt64(instance, &2, 1, &mut stream, 1), OK);
            assert_eq!(fmi3GetInt64(instance, &5, 1, &mut gain, 1), OK);
            assert_eq!(fmi3GetInt64(instance, &4, 1, &mut input, 1), OK);
            assert_eq!((seed, stream, gain), (42, 7, 3));
            assert_eq!(input, 2);
            assert_eq!(fmi3SetInt64(instance, &5, 1, &6, 1), OK);
            assert_eq!(fmi3GetInt64(instance, &4, 1, &mut input, 1), OK);
            assert_eq!(input, 2);
            assert_eq!(fmi3ExitInitializationMode(instance), OK);
            assert_eq!(fmi3FreeFMUState(instance, &mut state), OK);
            fmi3FreeInstance(instance);
        }
    }

    #[test]
    fn input_allowlist_and_parameter_in_initialization() {
        let model = r#"{"name":"Input","components":[{"name":"C","attributes":[{"name":"input","kind":"int","init":{"kind":"int","value":0}},{"name":"output","kind":"int","init":{"kind":"int","value":7}},{"name":"gain","kind":"int","init":{"kind":"int","value":1}}]}]}"#;
        let doc = prepare_document(
            model,
            &ExportManifest {
                inputs: vec!["C.input".into()],
                outputs: vec!["C.output".into()],
                parameters: vec!["C.gain".into()],
            },
        )
        .expect("manifest");
        let (_dir, instance) = unsafe { new_instance(&doc) };
        unsafe {
            assert_eq!(
                fmi3EnterInitializationMode(instance, false, 0.0, 0.0, true, 2.0),
                OK
            );
            assert_eq!(fmi3SetUInt64(instance, &1, 1, &u64::MAX, 1), OK);
            assert_eq!(fmi3SetInt64(instance, &6, 1, &9, 1), OK);
            assert_eq!(fmi3ExitInitializationMode(instance), OK);
            assert_eq!(fmi3SetInt64(instance, &4, 1, &3, 1), OK);
            assert_eq!(fmi3SetInt64(instance, &5, 1, &4, 1), ERROR);
            assert_eq!(fmi3SetInt64(instance, &6, 1, &4, 1), ERROR);
            let mut value = 0_i64;
            assert_eq!(fmi3GetInt64(instance, &4, 1, &mut value, 1), OK);
            assert_eq!(value, 3);
            assert_eq!(fmi3GetInt64(instance, &6, 1, &mut value, 1), OK);
            assert_eq!(value, 9);
            fmi3FreeInstance(instance);
        }
    }
}
