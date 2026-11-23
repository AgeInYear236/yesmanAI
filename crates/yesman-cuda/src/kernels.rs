use std::sync::Arc;

use anyhow::Result;

use crate::{
    context::{CudaBuffer, CudaContext, CudaModule},
    driver::CUfunction,
};

pub const KERNELS_PTX: &str = include_str!("kernels.ptx");

pub struct CudaKernels {
    pub module: CudaModule,
    pub rmsnorm: CUfunction,
    pub swiglu: CUfunction,
    pub residual_add: CUfunction,
    pub rope: CUfunction,
    pub gemv_f32: CUfunction,
    pub gemv_q8: CUfunction,
    pub gemv_q4k: CUfunction,
    pub decode_attention: CUfunction,
    ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaKernels {}
unsafe impl Sync for CudaKernels {}

impl CudaKernels {
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self> {
        let module = ctx.load_module(KERNELS_PTX)?;
        let rmsnorm = module.get_kernel("rmsnorm_kernel")?;
        let swiglu = module.get_kernel("swiglu_kernel")?;
        let residual_add = module.get_kernel("residual_add_kernel")?;
        let rope = module.get_kernel("rope_kernel")?;
        let gemv_f32 = module.get_kernel("gemv_f32_kernel")?;
        let gemv_q8 = module.get_kernel("gemv_q8_kernel")?;
        let gemv_q4k = module.get_kernel("gemv_q4k_kernel")?;
        let decode_attention = module.get_kernel("decode_attention_kernel")?;

        Ok(Self {
            module,
            rmsnorm,
            swiglu,
            residual_add,
            rope,
            gemv_f32,
            gemv_q8,
            gemv_q4k,
            decode_attention,
            ctx,
        })
    }

    pub fn rmsnorm(
        &self,
        out: &CudaBuffer<f32>,
        x: &CudaBuffer<f32>,
        weight: &CudaBuffer<f32>,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        let d_i32 = d as i32;
        let mut args: [*mut std::ffi::c_void; 5] = [
            &out.ptr as *const _ as *mut _,
            &x.ptr as *const _ as *mut _,
            &weight.ptr as *const _ as *mut _,
            &d_i32 as *const _ as *mut _,
            &eps as *const _ as *mut _,
        ];
        unsafe {
            (self.ctx.driver.cuLaunchKernel)(
                self.rmsnorm,
                rows as u32,
                1,
                1,
                32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        Ok(())
    }

    pub fn swiglu(
        &self,
        act: &CudaBuffer<f32>,
        gate: &CudaBuffer<f32>,
        up: &CudaBuffer<f32>,
        n: usize,
    ) -> Result<()> {
        let n_i32 = n as i32;
        let mut args: [*mut std::ffi::c_void; 4] = [
            &act.ptr as *const _ as *mut _,
            &gate.ptr as *const _ as *mut _,
            &up.ptr as *const _ as *mut _,
            &n_i32 as *const _ as *mut _,
        ];
        let block_dim = 256u32;
        let grid_dim = (n as u32).div_ceil(block_dim);
        unsafe {
            (self.ctx.driver.cuLaunchKernel)(
                self.swiglu,
                grid_dim,
                1,
                1,
                block_dim,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        Ok(())
    }

    pub fn residual_add(&self, a: &CudaBuffer<f32>, b: &CudaBuffer<f32>, n: usize) -> Result<()> {
        let n_i32 = n as i32;
        let mut args: [*mut std::ffi::c_void; 3] = [
            &a.ptr as *const _ as *mut _,
            &b.ptr as *const _ as *mut _,
            &n_i32 as *const _ as *mut _,
        ];
        let block_dim = 256u32;
        let grid_dim = (n as u32).div_ceil(block_dim);
        unsafe {
            (self.ctx.driver.cuLaunchKernel)(
                self.residual_add,
                grid_dim,
                1,
                1,
                block_dim,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        Ok(())
    }

    pub fn rope(
        &self,
        x: &CudaBuffer<f32>,
        cos_tab: &CudaBuffer<f32>,
        sin_tab: &CudaBuffer<f32>,
        pos: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> Result<()> {
        let pos_i32 = pos as i32;
        let hd_i32 = head_dim as i32;
        let nh_i32 = num_heads as i32;
        let mut args: [*mut std::ffi::c_void; 6] = [
            &x.ptr as *const _ as *mut _,
            &cos_tab.ptr as *const _ as *mut _,
            &sin_tab.ptr as *const _ as *mut _,
            &pos_i32 as *const _ as *mut _,
            &hd_i32 as *const _ as *mut _,
            &nh_i32 as *const _ as *mut _,
        ];
        unsafe {
            (self.ctx.driver.cuLaunchKernel)(
                self.rope,
                num_heads as u32,
                1,
                1,
                32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        Ok(())
    }

    pub fn gemv_f32(
        &self,
        out: &CudaBuffer<f32>,
        x: &CudaBuffer<f32>,
        w: &CudaBuffer<f32>,
        bias: Option<&CudaBuffer<f32>>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let k_i32 = k as i32;
        let n_i32 = n as i32;
        let bias_ptr = bias.map(|b| b.ptr).unwrap_or(0);
        let mut args: [*mut std::ffi::c_void; 6] = [
            &out.ptr as *const _ as *mut _,
            &x.ptr as *const _ as *mut _,
            &w.ptr as *const _ as *mut _,
            &bias_ptr as *const _ as *mut _,
            &k_i32 as *const _ as *mut _,
            &n_i32 as *const _ as *mut _,
        ];
        unsafe {
            (self.ctx.driver.cuLaunchKernel)(
                self.gemv_f32,
                n as u32,
                1,
                1,
                32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        Ok(())
    }

    pub fn gemv_q8(
        &self,
        out: &CudaBuffer<f32>,
        x: &CudaBuffer<f32>,
        w_blocks: &CudaBuffer<u8>,
        bias: Option<&CudaBuffer<f32>>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let k_i32 = k as i32;
        let n_i32 = n as i32;
        let bias_ptr = bias.map(|b| b.ptr).unwrap_or(0);
        let mut args: [*mut std::ffi::c_void; 6] = [
            &out.ptr as *const _ as *mut _,
            &x.ptr as *const _ as *mut _,
            &w_blocks.ptr as *const _ as *mut _,
            &bias_ptr as *const _ as *mut _,
            &k_i32 as *const _ as *mut _,
            &n_i32 as *const _ as *mut _,
        ];
        unsafe {
            (self.ctx.driver.cuLaunchKernel)(
                self.gemv_q8,
                n as u32,
                1,
                1,
                32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        Ok(())
    }

    pub fn gemv_q4k(
        &self,
        out: &CudaBuffer<f32>,
        x: &CudaBuffer<f32>,
        w_blocks: &CudaBuffer<u8>,
        bias: Option<&CudaBuffer<f32>>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let k_i32 = k as i32;
        let n_i32 = n as i32;
        let bias_ptr = bias.map(|b| b.ptr).unwrap_or(0);
        let mut args: [*mut std::ffi::c_void; 6] = [
            &out.ptr as *const _ as *mut _,
            &x.ptr as *const _ as *mut _,
            &w_blocks.ptr as *const _ as *mut _,
            &bias_ptr as *const _ as *mut _,
            &k_i32 as *const _ as *mut _,
            &n_i32 as *const _ as *mut _,
        ];
        unsafe {
            (self.ctx.driver.cuLaunchKernel)(
                self.gemv_q4k,
                n as u32,
                1,
                1,
                32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_attention(
        &self,
        out: &CudaBuffer<f32>,
        q: &CudaBuffer<f32>,
        kv_k: &CudaBuffer<f32>,
        kv_v: &CudaBuffer<f32>,
        head_dim: usize,
        num_heads: usize,
        group: usize,
        kv_dim: usize,
        t_end: usize,
        scale: f32,
    ) -> Result<()> {
        let hd_i32 = head_dim as i32;
        let nh_i32 = num_heads as i32;
        let grp_i32 = group as i32;
        let kvd_i32 = kv_dim as i32;
        let tend_i32 = t_end as i32;
        let mut args: [*mut std::ffi::c_void; 10] = [
            &out.ptr as *const _ as *mut _,
            &q.ptr as *const _ as *mut _,
            &kv_k.ptr as *const _ as *mut _,
            &kv_v.ptr as *const _ as *mut _,
            &hd_i32 as *const _ as *mut _,
            &nh_i32 as *const _ as *mut _,
            &grp_i32 as *const _ as *mut _,
            &kvd_i32 as *const _ as *mut _,
            &tend_i32 as *const _ as *mut _,
            &scale as *const _ as *mut _,
        ];
        unsafe {
            (self.ctx.driver.cuLaunchKernel)(
                self.decode_attention,
                num_heads as u32,
                1,
                1,
                32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        Ok(())
    }
}
