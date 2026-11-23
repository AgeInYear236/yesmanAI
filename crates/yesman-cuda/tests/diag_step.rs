use std::{path::Path, sync::Arc};
use yesman_cuda::{CudaDevice, CudaQwen2, is_cuda_available};
use yesman_model::{DType, Qwen2};

#[test]
fn test_cuda_step_by_step_diag() {
    if !is_cuda_available() {
        return;
    }

    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("yesman-model/tests/data/tiny256");

    let cpu = Qwen2::from_dir(&dir, 64, DType::F32).unwrap();
    let dev = CudaDevice::new(0).unwrap();
    let ctx = dev.create_context().unwrap();
    let mut gpu = CudaQwen2::from_cpu(&cpu, Arc::clone(&ctx), 64).unwrap();

    let t = 5u32;
    let pos = 0usize;

    let h = gpu.cfg().hidden_size;
    let eps = gpu.cfg().rms_norm_eps;

    // 1. Embed
    let cpu_embed = cpu.gather_embed(t as usize);
    gpu.x.copy_from_host(&cpu_embed).unwrap();
    let gpu_x = gpu.x.to_vec().unwrap();
    let diff: f32 = cpu_embed
        .iter()
        .zip(&gpu_x)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("Step 1 (Embed) diff: {diff:.3e}");

    // 2. Pre-norm
    let mut cpu_hb = vec![0.0f32; h];
    yesman_core::ops::rmsnorm(&mut cpu_hb, &cpu_embed, &cpu.weights.layers[0].in_norm, eps);
    gpu.weights
        .kernels
        .rmsnorm(&gpu.hb, &gpu.x, &gpu.weights.layers[0].in_norm, 1, h, eps)
        .unwrap();
    ctx.synchronize().unwrap();
    let gpu_hb = gpu.hb.to_vec().unwrap();
    let diff: f32 = cpu_hb
        .iter()
        .zip(&gpu_hb)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("Step 2 (In-Norm) diff: {diff:.3e}");

    // 3. Q projection
    let mut cpu_q = vec![0.0f32; h];
    cpu.weights.layers[0]
        .wq
        .matmul(&mut cpu_q, &cpu_hb, 1, h, h);
    for (v, b) in cpu_q.iter_mut().zip(&cpu.weights.layers[0].bq) {
        *v += b;
    }
    gpu.weights.layers[0]
        .wq
        .gemv(
            &gpu.weights.kernels,
            &gpu.q,
            &gpu.hb,
            Some(&gpu.weights.layers[0].bq),
        )
        .unwrap();
    ctx.synchronize().unwrap();
    let gpu_q = gpu.q.to_vec().unwrap();
    let diff: f32 = cpu_q
        .iter()
        .zip(&gpu_q)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("Step 3 (Q-proj) diff: {diff:.3e}");

    // 4. RoPE
    let mut cpu_q_rope = cpu_q.clone();
    yesman_core::ops::rope(
        &mut cpu_q_rope,
        &cpu.weights.cos,
        &cpu.weights.sin,
        pos,
        gpu.cfg().head_dim(),
    );
    gpu.weights
        .kernels
        .rope(
            &gpu.q,
            &gpu.weights.cos_tab,
            &gpu.weights.sin_tab,
            pos,
            gpu.cfg().head_dim(),
            gpu.cfg().num_attention_heads,
        )
        .unwrap();
    ctx.synchronize().unwrap();
    let gpu_q_rope = gpu.q.to_vec().unwrap();
    let diff: f32 = cpu_q_rope
        .iter()
        .zip(&gpu_q_rope)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("Step 4 (Q-RoPE) diff: {diff:.3e}");

    // 5. Attention
    let hd = gpu.cfg().head_dim();
    let n_heads = gpu.cfg().num_attention_heads;
    let group = gpu.cfg().gqa_group();
    let kv_dim = gpu.cfg().kv_dim();
    let scale = 1.0 / (hd as f32).sqrt();

    // Compute K, V
    let mut cpu_k = vec![0.0f32; kv_dim];
    let mut cpu_v = vec![0.0f32; kv_dim];
    cpu.weights.layers[0]
        .wk
        .matmul(&mut cpu_k, &cpu_hb, 1, h, kv_dim);
    cpu.weights.layers[0]
        .wv
        .matmul(&mut cpu_v, &cpu_hb, 1, h, kv_dim);
    for (v, b) in cpu_k.iter_mut().zip(&cpu.weights.layers[0].bk) {
        *v += b;
    }
    for (v, b) in cpu_v.iter_mut().zip(&cpu.weights.layers[0].bv) {
        *v += b;
    }
    yesman_core::ops::rope(&mut cpu_k, &cpu.weights.cos, &cpu.weights.sin, pos, hd);

    gpu.weights.layers[0]
        .wk
        .gemv(
            &gpu.weights.kernels,
            &gpu.k,
            &gpu.hb,
            Some(&gpu.weights.layers[0].bk),
        )
        .unwrap();
    gpu.weights.layers[0]
        .wv
        .gemv(
            &gpu.weights.kernels,
            &gpu.v,
            &gpu.hb,
            Some(&gpu.weights.layers[0].bv),
        )
        .unwrap();
    gpu.weights
        .kernels
        .rope(
            &gpu.k,
            &gpu.weights.cos_tab,
            &gpu.weights.sin_tab,
            pos,
            hd,
            gpu.cfg().num_key_value_heads,
        )
        .unwrap();

    let byte_count = kv_dim * std::mem::size_of::<f32>();
    unsafe {
        (gpu.weights.ctx.driver.cuMemcpyDtoD)(gpu.kv_k[0].ptr, gpu.k.ptr, byte_count);
        (gpu.weights.ctx.driver.cuMemcpyDtoD)(gpu.kv_v[0].ptr, gpu.v.ptr, byte_count);
    }

    // CPU attention for pos=0 (single token attends to itself -> attention weight is 1.0, output is V!)
    let mut cpu_attn = vec![0.0f32; h];
    for head in 0..n_heads {
        let kv_head = head / group;
        let v_slice = &cpu_v[kv_head * hd..(kv_head + 1) * hd];
        cpu_attn[head * hd..(head + 1) * hd].copy_from_slice(v_slice);
    }

    gpu.weights
        .kernels
        .decode_attention(
            &gpu.attn_out,
            &gpu.q,
            &gpu.kv_k[0],
            &gpu.kv_v[0],
            hd,
            n_heads,
            group,
            kv_dim,
            1,
            scale,
        )
        .unwrap();
    ctx.synchronize().unwrap();
    let gpu_attn = gpu.attn_out.to_vec().unwrap();
    let diff: f32 = cpu_attn
        .iter()
        .zip(&gpu_attn)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("Step 5 (Attention) diff: {diff:.3e}");
    if diff > 1e-4 {
        println!("CPU attn[0..8]: {:?}", &cpu_attn[0..8]);
        println!("GPU attn[0..8]: {:?}", &gpu_attn[0..8]);
    }
}
