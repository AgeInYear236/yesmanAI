//! Dynamic bindings for the CUDA Driver API (`libcuda.so.1`).
//!
//! Loads `libcuda.so.1` at runtime via `dlopen` without compile-time toolkit dependencies.

#![allow(non_snake_case)]
#![allow(clippy::missing_transmute_annotations)]

use std::{
    ffi::{CStr, CString, c_char, c_void},
    sync::OnceLock,
};

use anyhow::{Result, bail};

pub type CUresult = i32;
pub const CUDA_SUCCESS: CUresult = 0;

pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUdeviceptr = u64;

#[derive(Clone, Copy)]
pub struct CudaDriver {
    pub cuInit: unsafe extern "C" fn(flags: u32) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult,
    pub cuDeviceGetName:
        unsafe extern "C" fn(name: *mut c_char, len: i32, dev: CUdevice) -> CUresult,
    pub cuDeviceTotalMem: unsafe extern "C" fn(bytes: *mut usize, dev: CUdevice) -> CUresult,
    pub cuCtxCreate:
        unsafe extern "C" fn(pctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult,
    pub cuCtxDestroy: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    pub cuCtxSetCurrent: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    pub cuModuleLoadData:
        unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult,
    pub cuModuleUnload: unsafe extern "C" fn(module: CUmodule) -> CUresult,
    pub cuModuleGetFunction: unsafe extern "C" fn(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult,
    pub cuMemAlloc: unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    pub cuMemFree: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
    pub cuMemcpyHtoD: unsafe extern "C" fn(
        dst_device: CUdeviceptr,
        src_host: *const c_void,
        byte_count: usize,
    ) -> CUresult,
    pub cuMemcpyDtoH: unsafe extern "C" fn(
        dst_host: *mut c_void,
        src_device: CUdeviceptr,
        byte_count: usize,
    ) -> CUresult,
    pub cuMemcpyDtoD: unsafe extern "C" fn(
        dst_device: CUdeviceptr,
        src_device: CUdeviceptr,
        byte_count: usize,
    ) -> CUresult,
    pub cuLaunchKernel: unsafe extern "C" fn(
        f: CUfunction,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        h_stream: *mut c_void,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult,
    pub cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
    pub cuGetErrorString:
        unsafe extern "C" fn(error: CUresult, p_str: *mut *const c_char) -> CUresult,
}

static DRIVER: OnceLock<Result<CudaDriver, String>> = OnceLock::new();

pub fn get_driver() -> Result<CudaDriver> {
    let res = DRIVER.get_or_init(|| unsafe { load_driver() });
    match res {
        Ok(d) => Ok(*d),
        Err(e) => bail!("{e}"),
    }
}

unsafe fn load_driver() -> Result<CudaDriver, String> {
    let lib_names = ["libcuda.so.1", "libcuda.so"];
    let mut handle: *mut c_void = std::ptr::null_mut();
    for name in lib_names {
        let c_name = CString::new(name).unwrap();
        handle = unsafe { libc::dlopen(c_name.as_ptr(), libc::RTLD_LAZY) };
        if !handle.is_null() {
            break;
        }
    }
    if handle.is_null() {
        return Err(
            "could not load libcuda.so.1 (NVIDIA driver not installed or not in library path)"
                .into(),
        );
    }

    macro_rules! sym {
        ($sym_str:expr) => {{
            let c_sym = CString::new($sym_str).unwrap();
            let ptr = unsafe { libc::dlsym(handle, c_sym.as_ptr()) };
            if ptr.is_null() {
                return Err(format!("symbol {} not found in libcuda.so.1", $sym_str));
            }
            unsafe { std::mem::transmute(ptr) }
        }};
    }

    macro_rules! sym_v2 {
        ($base:expr) => {{
            let v2_str = format!("{}_v2", $base);
            let c_v2 = CString::new(v2_str.as_str()).unwrap();
            let mut ptr = unsafe { libc::dlsym(handle, c_v2.as_ptr()) };
            if ptr.is_null() {
                let c_base = CString::new($base).unwrap();
                ptr = unsafe { libc::dlsym(handle, c_base.as_ptr()) };
            }
            if ptr.is_null() {
                return Err(format!("symbol {} not found in libcuda.so.1", $base));
            }
            unsafe { std::mem::transmute(ptr) }
        }};
    }

    Ok(CudaDriver {
        cuInit: sym!("cuInit"),
        cuDeviceGet: sym!("cuDeviceGet"),
        cuDeviceGetName: sym!("cuDeviceGetName"),
        cuDeviceTotalMem: sym_v2!("cuDeviceTotalMem"),
        cuCtxCreate: sym_v2!("cuCtxCreate"),
        cuCtxDestroy: sym_v2!("cuCtxDestroy"),
        cuCtxSetCurrent: sym!("cuCtxSetCurrent"),
        cuModuleLoadData: sym!("cuModuleLoadData"),
        cuModuleUnload: sym!("cuModuleUnload"),
        cuModuleGetFunction: sym!("cuModuleGetFunction"),
        cuMemAlloc: sym_v2!("cuMemAlloc"),
        cuMemFree: sym_v2!("cuMemFree"),
        cuMemcpyHtoD: sym_v2!("cuMemcpyHtoD"),
        cuMemcpyDtoH: sym_v2!("cuMemcpyDtoH"),
        cuMemcpyDtoD: sym_v2!("cuMemcpyDtoD"),
        cuLaunchKernel: sym!("cuLaunchKernel"),
        cuCtxSynchronize: sym!("cuCtxSynchronize"),
        cuGetErrorString: sym!("cuGetErrorString"),
    })
}

pub fn check_cuda(driver: &CudaDriver, res: CUresult, op: &str) -> Result<()> {
    if res == CUDA_SUCCESS {
        Ok(())
    } else {
        let mut err_str: *const c_char = std::ptr::null();
        unsafe { (driver.cuGetErrorString)(res, &mut err_str) };
        let msg = if !err_str.is_null() {
            unsafe { CStr::from_ptr(err_str).to_string_lossy().into_owned() }
        } else {
            format!("error code {res}")
        };
        bail!("CUDA error during {op}: {msg} ({res})")
    }
}
