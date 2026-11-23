//! Microbenchmark for the three dot-product kernels at lm_head size.
//!
//! `cargo run --release -p yesman-core --example bench_matmul`

use std::time::Instant;

use yesman_core::{Bf16Tensor, Q8Tensor, bf16, ops, quant};

fn main() {
    let k = 1536;
    let n = 151936;
    let x: Vec<f32> = (0..k).map(|i| ((i % 31) as f32 - 15.0) / 15.0).collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| (((i * 7) % 251) as f32 - 125.0) / 125.0)
        .collect();

    let bw = Bf16Tensor::from_f32(&w, n, k);
    let qw = Q8Tensor::from_f32(&w, n, k);
    let xq = quant::quantize_rows(&x, k);
    let mut out = vec![0.0f32; n];
    let reps = 5;

    // warm up all three paths
    ops::matmul(&mut out, &x, &w, 1, k, n);
    bf16::matmul(&mut out, &x, &bw, 1, k, n);
    quant::matmul_q8(&mut out, &xq, &qw, 1, k, n);

    let t = Instant::now();
    for _ in 0..reps {
        ops::matmul(&mut out, &x, &w, 1, k, n);
    }
    report(
        "f32  ",
        (n * k * 4) as f64 / 1e9,
        t.elapsed().as_secs_f64() / reps as f64,
    );

    let t = Instant::now();
    for _ in 0..reps {
        bf16::matmul(&mut out, &x, &bw, 1, k, n);
    }
    report(
        "bf16 ",
        (n * k * 2) as f64 / 1e9,
        t.elapsed().as_secs_f64() / reps as f64,
    );

    let t = Instant::now();
    for _ in 0..reps {
        quant::matmul_q8(&mut out, &xq, &qw, 1, k, n);
    }
    report(
        "q8_0 ",
        (n * k) as f64 / 1e9,
        t.elapsed().as_secs_f64() / reps as f64,
    );
    println!("(checksum {} values, last {})", out.len(), out[n - 1]);
}

fn report(name: &str, gb: f64, secs: f64) {
    println!("{name}: {:.4}s/tok, {:.2} GB/s", secs, gb / secs);
}
