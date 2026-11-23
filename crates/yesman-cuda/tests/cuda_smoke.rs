use yesman_cuda::{CudaDevice, is_cuda_available};

const TEST_PTX: &str = r#"
.version 7.1
.target sm_86
.address_size 64

.visible .entry vec_add(
	.param .b64 p_a,
	.param .b64 p_b,
	.param .b64 p_c,
	.param .b32 n
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<8>;

	mov.u32 %r1, %tid.x;
	mov.u32 %r2, %ctaid.x;
	ld.param.b32 	%r4, [n];
	mov.u32 %r3, %ntid.x;
	mad.lo.s32 	%r5, %r3, %r2, %r1;
	setp.ge.s32 	%p1, %r5, %r4;
	@%p1 bra 	DONE;

	ld.param.b64 	%rd4, [p_a];
	ld.param.b64 	%rd5, [p_b];
	ld.param.b64 	%rd6, [p_c];
	mul.wide.s32 	%rd7, %r5, 4;
	add.s64 	%rd1, %rd6, %rd7;
	add.s64 	%rd2, %rd5, %rd7;
	add.s64 	%rd3, %rd4, %rd7;
	ld.b32 	%r6, [%rd3];
	ld.b32 	%r7, [%rd2];
	add.rn.f32 	%r8, %r6, %r7;
	st.b32 	[%rd1], %r8;
DONE:
	ret;
}
"#;

#[test]
fn test_cuda_device_and_kernel_execution() {
    if !is_cuda_available() {
        println!("CUDA is not available, skipping GPU smoke test");
        return;
    }

    let dev = CudaDevice::new(0).expect("failed to get CUDA device 0");
    println!(
        "CUDA Device 0: {} (total memory: {:.2} GB)",
        dev.name,
        dev.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    assert!(
        dev.total_memory > 1_000_000_000,
        "expected at least 1 GB VRAM"
    );

    let ctx = dev.create_context().expect("failed to create context");
    let module = ctx
        .load_module(TEST_PTX)
        .expect("failed to load PTX module");
    let kernel = module
        .get_kernel("vec_add")
        .expect("failed to get vec_add kernel");

    let n = 2048usize;
    let h_a: Vec<f32> = (0..n).map(|i| i as f32 * 1.5).collect();
    let h_b: Vec<f32> = (0..n).map(|i| 500.0 - i as f32).collect();

    let d_a = ctx.from_slice(&h_a).expect("alloc d_a");
    let d_b = ctx.from_slice(&h_b).expect("alloc d_b");
    let d_c = ctx.alloc::<f32>(n).expect("alloc d_c");

    let n_u32 = n as u32;
    let mut args: [*mut std::ffi::c_void; 4] = [
        &d_a.ptr as *const _ as *mut _,
        &d_b.ptr as *const _ as *mut _,
        &d_c.ptr as *const _ as *mut _,
        &n_u32 as *const _ as *mut _,
    ];

    let block_dim = 256u32;
    let grid_dim = (n as u32).div_ceil(block_dim);

    unsafe {
        (ctx.driver.cuLaunchKernel)(
            kernel,
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

    ctx.synchronize().expect("cuCtxSynchronize failed");
    let h_c = d_c.to_vec().expect("copy back d_c");

    for i in 0..n {
        let expected = h_a[i] + h_b[i];
        assert_eq!(
            h_c[i], expected,
            "mismatch at index {i}: got {}, expected {}",
            h_c[i], expected
        );
    }
    println!("GPU vector add passed for {n} elements on {}!", dev.name);
}
