// Signatures match the vendored FMI 3.0.2 headers. These capabilities are not advertised.
#[no_mangle]
pub unsafe extern "C" fn fmi3EnterEventMode(instance: Handle) -> c_int {
    status(instance, "fmi3EnterEventMode", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetFloat32(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut f32,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3GetFloat32", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetInt8(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut i8,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3GetInt8", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetUInt8(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut u8,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3GetUInt8", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetInt16(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut i16,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3GetInt16", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetUInt16(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut u16,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3GetUInt16", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetInt32(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut i32,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3GetInt32", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetUInt32(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut u32,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3GetUInt32", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetString(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut *const c_char,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3GetString", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetBinary(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut usize,
    _arg4: *mut *const u8,
    _arg5: usize,
) -> c_int {
    status(instance, "fmi3GetBinary", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetClock(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut bool,
) -> c_int {
    status(instance, "fmi3GetClock", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetFloat32(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const f32,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3SetFloat32", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetInt8(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const i8,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3SetInt8", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetUInt8(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const u8,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3SetUInt8", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetInt16(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const i16,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3SetInt16", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetUInt16(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const u16,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3SetUInt16", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetInt32(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const i32,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3SetInt32", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetUInt32(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const u32,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3SetUInt32", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetString(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const *const c_char,
    _arg4: usize,
) -> c_int {
    status(instance, "fmi3SetString", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetBinary(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const usize,
    _arg4: *const *const u8,
    _arg5: usize,
) -> c_int {
    status(instance, "fmi3SetBinary", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetClock(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const bool,
) -> c_int {
    status(instance, "fmi3SetClock", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetNumberOfVariableDependencies(
    instance: Handle,
    _arg1: u32,
    _arg2: *mut usize,
) -> c_int {
    status(instance, "fmi3GetNumberOfVariableDependencies", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetVariableDependencies(
    instance: Handle,
    _arg1: u32,
    _arg2: *mut usize,
    _arg3: *mut u32,
    _arg4: *mut usize,
    _arg5: *mut c_int,
    _arg6: usize,
) -> c_int {
    status(instance, "fmi3GetVariableDependencies", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SerializedFMUStateSize(
    instance: Handle,
    _arg1: Handle,
    _arg2: *mut usize,
) -> c_int {
    status(instance, "fmi3SerializedFMUStateSize", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SerializeFMUState(
    instance: Handle,
    _arg1: Handle,
    _arg2: *mut u8,
    _arg3: usize,
) -> c_int {
    status(instance, "fmi3SerializeFMUState", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3DeserializeFMUState(
    instance: Handle,
    _arg1: *const u8,
    _arg2: usize,
    _arg3: *mut Handle,
) -> c_int {
    status(instance, "fmi3DeserializeFMUState", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetDirectionalDerivative(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const u32,
    _arg4: usize,
    _arg5: *const f64,
    _arg6: usize,
    _arg7: *mut f64,
    _arg8: usize,
) -> c_int {
    status(instance, "fmi3GetDirectionalDerivative", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetAdjointDerivative(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const u32,
    _arg4: usize,
    _arg5: *const f64,
    _arg6: usize,
    _arg7: *mut f64,
    _arg8: usize,
) -> c_int {
    status(instance, "fmi3GetAdjointDerivative", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3EnterConfigurationMode(instance: Handle) -> c_int {
    status(instance, "fmi3EnterConfigurationMode", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3ExitConfigurationMode(instance: Handle) -> c_int {
    status(instance, "fmi3ExitConfigurationMode", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetIntervalDecimal(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut f64,
    _arg4: *mut c_int,
) -> c_int {
    status(instance, "fmi3GetIntervalDecimal", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetIntervalFraction(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut u64,
    _arg4: *mut u64,
    _arg5: *mut c_int,
) -> c_int {
    status(instance, "fmi3GetIntervalFraction", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetShiftDecimal(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut f64,
) -> c_int {
    status(instance, "fmi3GetShiftDecimal", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetShiftFraction(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *mut u64,
    _arg4: *mut u64,
) -> c_int {
    status(instance, "fmi3GetShiftFraction", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetIntervalDecimal(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const f64,
) -> c_int {
    status(instance, "fmi3SetIntervalDecimal", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetIntervalFraction(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const u64,
    _arg4: *const u64,
) -> c_int {
    status(instance, "fmi3SetIntervalFraction", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetShiftDecimal(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const f64,
) -> c_int {
    status(instance, "fmi3SetShiftDecimal", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetShiftFraction(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const u64,
    _arg4: *const u64,
) -> c_int {
    status(instance, "fmi3SetShiftFraction", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3EvaluateDiscreteStates(instance: Handle) -> c_int {
    status(instance, "fmi3EvaluateDiscreteStates", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3UpdateDiscreteStates(
    instance: Handle,
    _arg1: *mut bool,
    _arg2: *mut bool,
    _arg3: *mut bool,
    _arg4: *mut bool,
    _arg5: *mut bool,
    _arg6: *mut f64,
) -> c_int {
    status(instance, "fmi3UpdateDiscreteStates", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3EnterContinuousTimeMode(instance: Handle) -> c_int {
    status(instance, "fmi3EnterContinuousTimeMode", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3CompletedIntegratorStep(
    instance: Handle,
    _arg1: bool,
    _arg2: *mut bool,
    _arg3: *mut bool,
) -> c_int {
    status(instance, "fmi3CompletedIntegratorStep", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetTime(instance: Handle, _arg1: f64) -> c_int {
    status(instance, "fmi3SetTime", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3SetContinuousStates(
    instance: Handle,
    _arg1: *const f64,
    _arg2: usize,
) -> c_int {
    status(instance, "fmi3SetContinuousStates", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetContinuousStateDerivatives(
    instance: Handle,
    _arg1: *mut f64,
    _arg2: usize,
) -> c_int {
    status(instance, "fmi3GetContinuousStateDerivatives", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetEventIndicators(
    instance: Handle,
    _arg1: *mut f64,
    _arg2: usize,
) -> c_int {
    status(instance, "fmi3GetEventIndicators", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetContinuousStates(
    instance: Handle,
    _arg1: *mut f64,
    _arg2: usize,
) -> c_int {
    status(instance, "fmi3GetContinuousStates", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetNominalsOfContinuousStates(
    instance: Handle,
    _arg1: *mut f64,
    _arg2: usize,
) -> c_int {
    status(instance, "fmi3GetNominalsOfContinuousStates", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetNumberOfEventIndicators(
    instance: Handle,
    _arg1: *mut usize,
) -> c_int {
    status(instance, "fmi3GetNumberOfEventIndicators", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetNumberOfContinuousStates(
    instance: Handle,
    _arg1: *mut usize,
) -> c_int {
    status(instance, "fmi3GetNumberOfContinuousStates", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3EnterStepMode(instance: Handle) -> c_int {
    status(instance, "fmi3EnterStepMode", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3GetOutputDerivatives(
    instance: Handle,
    _arg1: *const u32,
    _arg2: usize,
    _arg3: *const i32,
    _arg4: *mut f64,
    _arg5: usize,
) -> c_int {
    status(instance, "fmi3GetOutputDerivatives", |_i| {
        Err("FMI capability unavailable".into())
    })
}

#[no_mangle]
pub unsafe extern "C" fn fmi3ActivateModelPartition(
    instance: Handle,
    _arg1: u32,
    _arg2: f64,
) -> c_int {
    status(instance, "fmi3ActivateModelPartition", |_i| {
        Err("FMI capability unavailable".into())
    })
}
