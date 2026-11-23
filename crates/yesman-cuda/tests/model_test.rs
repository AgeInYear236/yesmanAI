use std::{path::Path, sync::Arc};
use yesman_cuda::{CudaDevice, CudaQwen2, is_cuda_available};
use yesman_model::{DType, Qwen2};

#[test]
fn test_cuda_model_forward_matches_cpu() {
    if !is_cuda_available() {
        println!("CUDA is not available, skipping GPU model test");
        return;
    }

    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("yesman-model/tests/data/tiny256");

    let raw = std::fs::read_to_string(dir.join("expected.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let ids: Vec<u32> = v["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x: &serde_json::Value| x.as_u64().unwrap() as u32)
        .collect();

    let mut cpu_model = Qwen2::from_dir(&dir, 64, DType::F32).unwrap();
    let dev = CudaDevice::new(0).expect("failed to get CUDA device");
    let ctx = dev.create_context().expect("create context");

    let mut gpu_model =
        CudaQwen2::from_cpu(&cpu_model, Arc::clone(&ctx), 64).expect("upload to GPU");

    // Run incremental decode on CPU
    let mut cpu_last_logits = Vec::new();
    for (i, &t) in ids.iter().enumerate() {
        cpu_last_logits = cpu_model.forward(&[t], i);
    }

    // Run incremental decode on GPU
    let mut gpu_last_logits = Vec::new();
    for (i, &t) in ids.iter().enumerate() {
        gpu_last_logits = gpu_model.decode_step(t, i, &cpu_model).unwrap();
    }

    assert_eq!(cpu_last_logits.len(), gpu_last_logits.len());
    let mut max_diff = 0.0f32;
    for (i, (c, g)) in cpu_last_logits.iter().zip(&gpu_last_logits).enumerate() {
        let diff = (c - g).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        println!("logit {i}: cpu={c:.4}, gpu={g:.4}, diff={diff:.3e}");
    }
    println!("Max logit diff between CPU and GPU: {max_diff:.3e}");
    assert!(max_diff < 1e-4, "GPU vs CPU logits diverged: {max_diff}");
    println!("GPU Qwen2 forward pass verified on {}!", dev.name);
}
