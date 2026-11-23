use std::sync::Arc;
use yesman_cuda::{CudaDevice, CudaQwen2Weights, is_cuda_available};
use yesman_model::Qwen2;

#[test]
fn test_real_model_sessions_vram() {
    if !is_cuda_available() {
        return;
    }

    let cache_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("models/Qwen2.5-1.5B-Instruct/model.q4km.ymq");
    if !cache_path.exists() {
        println!("Cache path {} not found", cache_path.display());
        return;
    }

    let dev = CudaDevice::new(0).unwrap();
    let ctx = dev.create_context().unwrap();

    let model_dir = cache_path.parent().unwrap();
    let cpu = Qwen2::from_cache(model_dir, 4096, &cache_path).unwrap();
    println!("Loaded CPU model from cache!");

    let weights = CudaQwen2Weights::from_cpu(&cpu, Arc::clone(&ctx)).unwrap();
    println!("Uploaded weights to GPU!");

    let mut sessions = Vec::new();
    for i in 0..4 {
        let s = weights
            .create_session(4096)
            .unwrap_or_else(|_| panic!("failed to create session {i}"));
        sessions.push(s);
        println!("Successfully created session {i}!");
    }
    assert_eq!(sessions.len(), 4);
}
