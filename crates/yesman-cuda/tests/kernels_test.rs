use std::sync::Arc;
use yesman_core::{ops, q4k::Q4KTensor, quant::Q8Tensor};
use yesman_cuda::{CudaDevice, CudaKernels, is_cuda_available};

#[test]
fn test_cuda_kernels_match_cpu() {
    if !is_cuda_available() {
        println!("CUDA is not available, skipping GPU kernels test");
        return;
    }

    let dev = CudaDevice::new(0).expect("failed to get CUDA device");
    let ctx = dev.create_context().expect("create context");
    let kernels = CudaKernels::new(Arc::clone(&ctx)).expect("load kernels");

    // 1. RMSNorm test
    {
        let d = 1536usize;
        let eps = 1e-6f32;
        let h_x: Vec<f32> = (0..d).map(|i| (i as f32 % 13.0 - 6.0) * 0.2).collect();
        let h_w: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32 % 7.0) * 0.05).collect();

        let mut cpu_out = vec![0.0f32; d];
        ops::rmsnorm(&mut cpu_out, &h_x, &h_w, eps);

        let d_x = ctx.from_slice(&h_x).unwrap();
        let d_w = ctx.from_slice(&h_w).unwrap();
        let d_out = ctx.alloc::<f32>(d).unwrap();

        kernels.rmsnorm(&d_out, &d_x, &d_w, 1, d, eps).unwrap();
        ctx.synchronize().unwrap();
        let gpu_out = d_out.to_vec().unwrap();

        let max_diff = cpu_out
            .iter()
            .zip(&gpu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("RMSNorm max diff: {max_diff:.3e}");
        assert!(
            max_diff < 1e-5,
            "RMSNorm GPU vs CPU diff too large: {max_diff}"
        );
    }

    // 2. SwiGLU test
    {
        let n = 8960usize;
        let h_gate: Vec<f32> = (0..n).map(|i| (i as f32 % 19.0 - 9.0) * 0.1).collect();
        let h_up: Vec<f32> = (0..n).map(|i| (i as f32 % 11.0 - 5.0) * 0.1).collect();

        let mut cpu_out = vec![0.0f32; n];
        ops::swiglu(&mut cpu_out, &h_gate, &h_up);

        let d_gate = ctx.from_slice(&h_gate).unwrap();
        let d_up = ctx.from_slice(&h_up).unwrap();
        let d_act = ctx.alloc::<f32>(n).unwrap();

        kernels.swiglu(&d_act, &d_gate, &d_up, n).unwrap();
        ctx.synchronize().unwrap();
        let gpu_out = d_act.to_vec().unwrap();

        let max_diff = cpu_out
            .iter()
            .zip(&gpu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("SwiGLU max diff: {max_diff:.3e}");
        assert!(
            max_diff < 1e-4,
            "SwiGLU GPU vs CPU diff too large: {max_diff}"
        );
    }

    // 3. Residual Add test
    {
        let n = 1536usize;
        let h_a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let h_b: Vec<f32> = (0..n).map(|i| 100.0 - i as f32 * 0.25).collect();

        let mut cpu_out = h_a.clone();
        ops::add_into(&mut cpu_out, &h_b);

        let d_a = ctx.from_slice(&h_a).unwrap();
        let d_b = ctx.from_slice(&h_b).unwrap();

        kernels.residual_add(&d_a, &d_b, n).unwrap();
        ctx.synchronize().unwrap();
        let gpu_out = d_a.to_vec().unwrap();

        let max_diff = cpu_out
            .iter()
            .zip(&gpu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Residual Add max diff: {max_diff:.3e}");
        assert!(max_diff < 1e-6, "Residual Add diff too large: {max_diff}");
    }

    // 4. GEMV FP32 test
    {
        let k = 1536usize;
        let n = 256usize;
        let h_x: Vec<f32> = (0..k).map(|i| (i as f32 % 17.0 - 8.0) * 0.05).collect();
        let h_w: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 % 23.0 - 11.0) * 0.01)
            .collect();
        let h_bias: Vec<f32> = (0..n).map(|i| (i as f32 % 5.0) * 0.1).collect();

        let mut cpu_out = vec![0.0f32; n];
        ops::matmul(&mut cpu_out, &h_x, &h_w, 1, k, n);
        for i in 0..n {
            cpu_out[i] += h_bias[i];
        }

        let d_x = ctx.from_slice(&h_x).unwrap();
        let d_w = ctx.from_slice(&h_w).unwrap();
        let d_bias = ctx.from_slice(&h_bias).unwrap();
        let d_out = ctx.alloc::<f32>(n).unwrap();

        kernels
            .gemv_f32(&d_out, &d_x, &d_w, Some(&d_bias), k, n)
            .unwrap();
        ctx.synchronize().unwrap();
        let gpu_out = d_out.to_vec().unwrap();

        let max_diff = cpu_out
            .iter()
            .zip(&gpu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("GEMV FP32 max diff: {max_diff:.3e}");
        assert!(max_diff < 1e-4, "GEMV FP32 diff too large: {max_diff}");
    }

    // 5. GEMV Q8_0 test
    {
        let k = 1536usize;
        let n = 256usize;
        let h_x: Vec<f32> = (0..k).map(|i| (i as f32 % 17.0 - 8.0) * 0.05).collect();
        let h_w: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 % 23.0 - 11.0) * 0.02)
            .collect();
        let q8 = Q8Tensor::from_f32(&h_w, n, k);

        let mut cpu_out = vec![0.0f32; n];
        let xq = yesman_core::quant::quantize_rows(&h_x, k);
        yesman_core::quant::matmul_q8(&mut cpu_out, &xq, &q8, 1, k, n);

        let d_x = ctx.from_slice(&h_x).unwrap();
        let d_q8 = ctx.from_slice(q8.as_bytes()).unwrap();
        let d_out = ctx.alloc::<f32>(n).unwrap();

        kernels.gemv_q8(&d_out, &d_x, &d_q8, None, k, n).unwrap();
        ctx.synchronize().unwrap();
        let gpu_out = d_out.to_vec().unwrap();

        let max_diff = cpu_out
            .iter()
            .zip(&gpu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("GEMV Q8 max diff: {max_diff:.3e}");
        assert!(max_diff < 0.05, "GEMV Q8 diff too large: {max_diff}");
    }

    // 6. GEMV Q4_K test
    {
        let k = 1536usize;
        let n = 256usize;
        let h_x: Vec<f32> = (0..k).map(|i| (i as f32 % 17.0 - 8.0) * 0.05).collect();
        let h_w: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 % 23.0 - 11.0) * 0.02)
            .collect();
        let q4k = Q4KTensor::from_f32(&h_w, n, k);

        let mut cpu_out = vec![0.0f32; n];
        let xq = yesman_core::quant::quantize_rows(&h_x, k);
        yesman_core::q4k::matmul_q8q4k(&mut cpu_out, &xq, &q4k, 1, k, n);

        let d_x = ctx.from_slice(&h_x).unwrap();
        let d_q4k = ctx.from_slice(q4k.as_bytes()).unwrap();
        let d_out = ctx.alloc::<f32>(n).unwrap();

        kernels.gemv_q4k(&d_out, &d_x, &d_q4k, None, k, n).unwrap();
        ctx.synchronize().unwrap();
        let gpu_out = d_out.to_vec().unwrap();

        let max_diff = cpu_out
            .iter()
            .zip(&gpu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("GEMV Q4_K max diff: {max_diff:.3e}");
        assert!(max_diff < 0.05, "GEMV Q4_K diff too large: {max_diff}");
    }

    println!("All CUDA kernels tested and verified against CPU reference!");
}
