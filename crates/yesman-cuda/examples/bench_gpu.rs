use std::{path::Path, sync::Arc, time::Instant};

use anyhow::Result;
use yesman_cuda::{CudaDevice, CudaQwen2, is_cuda_available};
use yesman_model::{Qwen2, Sampler, Tokenizer};

fn main() -> Result<()> {
    if !is_cuda_available() {
        eprintln!("CUDA is not available on this machine.");
        return Ok(());
    }

    let model_dir = Path::new("models/Qwen2.5-1.5B-Instruct");
    let cache_path = model_dir.join("model.q4km.ymq");
    if !cache_path.exists() {
        eprintln!("Quant cache {} not found", cache_path.display());
        return Ok(());
    }

    println!("Initializing CUDA Device 0...");
    let dev = CudaDevice::new(0)?;
    println!(
        "GPU: {} (total VRAM: {:.2} GB)",
        dev.name,
        dev.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let ctx = dev.create_context()?;

    let t0 = Instant::now();
    println!("Loading CPU model from cache...");
    let cpu_model = Qwen2::from_cache(model_dir, 4096, &cache_path)?;
    println!("Loaded CPU model in {:.2}s", t0.elapsed().as_secs_f32());

    let t_up = Instant::now();
    println!("Uploading weights to RTX 3060 VRAM...");
    let mut gpu_model = CudaQwen2::from_cpu(&cpu_model, Arc::clone(&ctx), 4096)?;
    println!(
        "Uploaded weights to GPU in {:.2}s!",
        t_up.elapsed().as_secs_f32()
    );

    let tok = Tokenizer::from_dir(model_dir)?;
    let prompt = "Explain in two sentences why Rust is fast.";
    let prompt_ids = tok.encode(prompt)?;
    println!("\nPrompt: \"{prompt}\" ({} tokens)", prompt_ids.len());

    // Prefill tokens through GPU in a single batched pass
    let t_prefill = Instant::now();
    let mut logits = gpu_model.prefill_chunk(&prompt_ids, 0, &cpu_model)?;
    let prefill_time = t_prefill.elapsed().as_secs_f32();
    println!(
        "GPU Batched Prefill done in {:.4}s ({:.1} tok/s)",
        prefill_time,
        prompt_ids.len() as f32 / prefill_time
    );

    let mut sampler = Sampler::new(42);
    sampler.temperature = 0.0; // greedy

    let max_gen = 48usize;
    let mut pos = prompt_ids.len();
    let mut history = prompt_ids.clone();

    print!("Generated: ");
    let t_dec = Instant::now();
    for _ in 0..max_gen {
        let next = sampler.sample(&mut logits, &history);
        if next == 151645 || next == 151643 {
            break;
        }
        let piece = tok.decode(&[next])?;
        print!("{piece}");
        std::io::Write::flush(&mut std::io::stdout())?;
        history.push(next);
        logits = gpu_model.decode_step(next, pos, &cpu_model)?;
        pos += 1;
    }
    let gen_time = t_dec.elapsed().as_secs_f32();
    let gen_tokens = pos - prompt_ids.len();
    println!(
        "\n\n[GPU Generation: {} tokens in {:.2}s = {:.1} tok/s]",
        gen_tokens,
        gen_time,
        gen_tokens as f32 / gen_time.max(1e-4)
    );

    Ok(())
}
