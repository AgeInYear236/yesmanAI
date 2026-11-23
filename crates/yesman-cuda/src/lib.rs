//! CUDA backend for yesmanAI using the dynamic CUDA Driver API (`libcuda.so.1`).
//!
//! Provides zero-dependency GPU inference on modern NVIDIA GPUs (Ampere/Ada/Blackwell).

pub mod context;
pub mod driver;
pub mod kernels;
pub mod model;

pub use context::{CudaBuffer, CudaContext, CudaDevice, CudaModule};
pub use driver::{CudaDriver, get_driver};
pub use kernels::CudaKernels;
pub use model::{CudaQwen2, CudaQwen2Weights};

pub fn is_cuda_available() -> bool {
    driver::get_driver()
        .and_then(|d| driver::check_cuda(&d, unsafe { (d.cuInit)(0) }, "cuInit"))
        .is_ok()
}
