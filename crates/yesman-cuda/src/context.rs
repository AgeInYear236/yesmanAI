use std::{
    ffi::{CString, c_void},
    marker::PhantomData,
    sync::Arc,
};

use anyhow::Result;

use crate::driver::{
    CUcontext, CUdevice, CUdeviceptr, CUfunction, CUmodule, CudaDriver, check_cuda, get_driver,
};

pub struct CudaDevice {
    pub id: CUdevice,
    pub name: String,
    pub total_memory: usize,
    pub driver: CudaDriver,
}

impl CudaDevice {
    pub fn new(ordinal: i32) -> Result<Self> {
        let driver = get_driver()?;
        check_cuda(&driver, unsafe { (driver.cuInit)(0) }, "cuInit")?;
        let mut dev: CUdevice = 0;
        check_cuda(
            &driver,
            unsafe { (driver.cuDeviceGet)(&mut dev, ordinal) },
            "cuDeviceGet",
        )?;
        let mut name_buf = [0i8; 256];
        check_cuda(
            &driver,
            unsafe { (driver.cuDeviceGetName)(name_buf.as_mut_ptr(), 255, dev) },
            "cuDeviceGetName",
        )?;
        let name = unsafe { std::ffi::CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let mut total_memory = 0usize;
        check_cuda(
            &driver,
            unsafe { (driver.cuDeviceTotalMem)(&mut total_memory, dev) },
            "cuDeviceTotalMem",
        )?;
        Ok(Self {
            id: dev,
            name,
            total_memory,
            driver,
        })
    }

    pub fn create_context(&self) -> Result<Arc<CudaContext>> {
        let mut ctx: CUcontext = std::ptr::null_mut();
        check_cuda(
            &self.driver,
            unsafe { (self.driver.cuCtxCreate)(&mut ctx, 0, self.id) },
            "cuCtxCreate",
        )?;
        Ok(Arc::new(CudaContext {
            ctx,
            driver: self.driver,
            device_name: self.name.clone(),
            total_memory: self.total_memory,
        }))
    }
}

pub struct CudaContext {
    pub ctx: CUcontext,
    pub driver: CudaDriver,
    pub device_name: String,
    pub total_memory: usize,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { (self.driver.cuCtxDestroy)(self.ctx) };
        }
    }
}

impl CudaContext {
    pub fn set_current(&self) -> Result<()> {
        check_cuda(
            &self.driver,
            unsafe { (self.driver.cuCtxSetCurrent)(self.ctx) },
            "cuCtxSetCurrent",
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        check_cuda(
            &self.driver,
            unsafe { (self.driver.cuCtxSynchronize)() },
            "cuCtxSynchronize",
        )
    }

    pub fn load_module(&self, ptx: &str) -> Result<CudaModule> {
        let mut module: CUmodule = std::ptr::null_mut();
        let c_ptx = CString::new(ptx)?;
        check_cuda(
            &self.driver,
            unsafe { (self.driver.cuModuleLoadData)(&mut module, c_ptx.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        Ok(CudaModule {
            module,
            driver: self.driver,
        })
    }

    pub fn alloc<T>(&self, len: usize) -> Result<CudaBuffer<T>> {
        let bytes = len * std::mem::size_of::<T>();
        let mut dptr: CUdeviceptr = 0;
        check_cuda(
            &self.driver,
            unsafe { (self.driver.cuMemAlloc)(&mut dptr, bytes.max(4)) },
            "cuMemAlloc",
        )?;
        Ok(CudaBuffer {
            ptr: dptr,
            len,
            driver: self.driver,
            _marker: PhantomData,
        })
    }

    pub fn from_slice<T>(&self, data: &[T]) -> Result<CudaBuffer<T>> {
        let mut buf = self.alloc::<T>(data.len())?;
        buf.copy_from_host(data)?;
        Ok(buf)
    }
}

pub struct CudaModule {
    pub module: CUmodule,
    driver: CudaDriver,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe { (self.driver.cuModuleUnload)(self.module) };
        }
    }
}

impl CudaModule {
    pub fn get_kernel(&self, name: &str) -> Result<CUfunction> {
        let mut func: CUfunction = std::ptr::null_mut();
        let c_name = CString::new(name)?;
        check_cuda(
            &self.driver,
            unsafe { (self.driver.cuModuleGetFunction)(&mut func, self.module, c_name.as_ptr()) },
            &format!("cuModuleGetFunction({name})"),
        )?;
        Ok(func)
    }
}

pub struct CudaBuffer<T> {
    pub ptr: CUdeviceptr,
    pub len: usize,
    driver: CudaDriver,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for CudaBuffer<T> {}
unsafe impl<T: Sync> Sync for CudaBuffer<T> {}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe { (self.driver.cuMemFree)(self.ptr) };
        }
    }
}

impl<T> CudaBuffer<T> {
    pub fn bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        assert!(src.len() <= self.len);
        let byte_count = std::mem::size_of_val(src);
        check_cuda(
            &self.driver,
            unsafe {
                (self.driver.cuMemcpyHtoD)(self.ptr, src.as_ptr() as *const c_void, byte_count)
            },
            "cuMemcpyHtoD",
        )
    }

    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        assert!(dst.len() >= self.len);
        let byte_count = self.len * std::mem::size_of::<T>();
        check_cuda(
            &self.driver,
            unsafe {
                (self.driver.cuMemcpyDtoH)(dst.as_mut_ptr() as *mut c_void, self.ptr, byte_count)
            },
            "cuMemcpyDtoH",
        )
    }

    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Default + Clone,
    {
        let mut out = vec![T::default(); self.len];
        self.copy_to_host(&mut out)?;
        Ok(out)
    }
}
