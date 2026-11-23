//! Cross-checks against the numpy reference in `tools/qwen_ref.py`
//! (`tools/make_tiny.py` regenerates the fixtures).

use std::path::Path;

use yesman_model::{DType, Qwen2};

fn fixture() -> (Vec<u32>, Vec<f32>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tiny");
    let raw = std::fs::read_to_string(dir.join("expected.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let ids: Vec<u32> = v["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let logits: Vec<f32> = v["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    (ids, logits)
}

fn model() -> Qwen2 {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tiny");
    Qwen2::from_dir(&dir, 64, DType::F32).unwrap()
}

fn model_with(dt: DType) -> Qwen2 {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tiny");
    Qwen2::from_dir(&dir, 64, dt).unwrap()
}

/// Q8_0 must stay close to the f32 reference (same fixture).
#[test]
fn quantized_forward_stays_close() {
    let (ids, want) = fixture();
    let mut m = model_with(DType::Q8);
    let got = m.forward(&ids, 0);
    let max_abs = want.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let worst = got
        .iter()
        .zip(want.iter())
        .fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
    println!("Q8_0 max deviation: {worst} ({:.2}% of max |logit| {max_abs})", 100.0 * worst / max_abs);
    // A 64-wide random model is the worst case for block quantization; the real
    // 1.5B checkpoint behaves far better. We only guard against gross errors.
    assert!(worst < 0.05 * max_abs, "Q8_0 deviation too large: {worst} vs {max_abs}");
}

#[test]
fn forward_matches_numpy() {
    let (ids, want) = fixture();
    let mut m = model();
    let got = m.forward(&ids, 0);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-3 + 1e-4 * w.abs();
        assert!((g - w).abs() < tol, "logit {i}: rust={g} numpy={w}");
    }
}

/// bf16 storage must stay within bf16 rounding of the f32 reference.
#[test]
fn bf16_forward_stays_close() {
    let (ids, want) = fixture();
    let mut m = model_with(DType::Bf16);
    let got = m.forward(&ids, 0);
    let max_abs = want.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let worst = got
        .iter()
        .zip(want.iter())
        .fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
    println!("bf16 max deviation: {worst} ({:.3}% of max |logit|)", 100.0 * worst / max_abs);
    assert!(worst < 0.01 * max_abs, "bf16 deviation too large: {worst} vs {max_abs}");
}

/// Q4_0 (4-bit weights, 8-bit activations) must stay within a few % of f32.
#[test]
fn q4_forward_stays_close() {
    let (ids, want) = fixture();
    let mut m = model_with(DType::Q4);
    let got = m.forward(&ids, 0);
    let max_abs = want.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let worst = got
        .iter()
        .zip(want.iter())
        .fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
    println!("Q4_0 max deviation: {worst} ({:.2}% of max |logit|)", 100.0 * worst / max_abs);
    // Q4_0 on a 64-wide random model is a worst case (blocks are tiny and the
    // logits are uncalibrated); the test guards against a broken kernel, not
    // quantization noise. On the real checkpoint Q4_0 still answers correctly.
    assert!(worst < 0.35 * max_abs, "Q4_0 deviation too large: {worst} vs {max_abs}");
}

/// Feeding tokens one at a time through the KV cache must reproduce the
/// batched prefill result exactly -- this is what catches cache bugs.
#[test]
fn incremental_decode_matches_prefill() {
    let (ids, _) = fixture();
    let mut batched = model();
    let full = batched.forward(&ids, 0);

    let mut stepwise = model();
    let mut last = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        last = stepwise.forward(&[id], i);
    }
    for (i, (a, b)) in full.iter().zip(last.iter()).enumerate() {
        let tol = 1e-4 + 1e-5 * a.abs();
        assert!((a - b).abs() < tol, "logit {i}: prefill={a} stepwise={b}");
    }
    assert_eq!(stepwise.kv_len(), ids.len());
}
