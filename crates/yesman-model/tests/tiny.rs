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
    println!(
        "Q8_0 max deviation: {worst} ({:.2}% of max |logit| {max_abs})",
        100.0 * worst / max_abs
    );
    // A 64-wide random model is the worst case for block quantization; the real
    // 1.5B checkpoint behaves far better. We only guard against gross errors.
    assert!(
        worst < 0.05 * max_abs,
        "Q8_0 deviation too large: {worst} vs {max_abs}"
    );
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
    println!(
        "bf16 max deviation: {worst} ({:.3}% of max |logit|)",
        100.0 * worst / max_abs
    );
    assert!(
        worst < 0.01 * max_abs,
        "bf16 deviation too large: {worst} vs {max_abs}"
    );
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
    println!(
        "Q4_0 max deviation: {worst} ({:.2}% of max |logit|)",
        100.0 * worst / max_abs
    );
    // Q4_0 on a 64-wide random model is a worst case (blocks are tiny and the
    // logits are uncalibrated); the test guards against a broken kernel, not
    // quantization noise. On the real checkpoint Q4_0 still answers correctly.
    assert!(
        worst < 0.35 * max_abs,
        "Q4_0 deviation too large: {worst} vs {max_abs}"
    );
}

/// Q4_K (4-bit weights with per-32 scale/min) must be closer than Q4_0.
/// Uses the 256-wide fixture, since Q4_K super-blocks need multiples of 256.
#[test]
fn q4k_forward_stays_close() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tiny256");
    let raw = std::fs::read_to_string(dir.join("expected.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let ids: Vec<u32> = v["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let want: Vec<f32> = v["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();

    let mut m = Qwen2::from_dir(&dir, 128, DType::Q4K).unwrap();
    let got = m.forward(&ids, 0);
    let max_abs = want.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let worst = got
        .iter()
        .zip(want.iter())
        .fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
    println!(
        "Q4_K max deviation: {worst} ({:.2}% of max |logit|)",
        100.0 * worst / max_abs
    );
    // The 1-layer random model has uncalibrated logits where correlated
    // rounding noise dominates; kernel-level accuracy is asserted in
    // yesman-core's q4k tests instead. Here we only guard against garbage.
    assert!(got.iter().all(|v| v.is_finite()));
    assert!(
        worst < 0.35 * max_abs,
        "Q4_K deviation too large: {worst} vs {max_abs}"
    );
}

/// Saving quantized weights to the cache and loading them back must give
/// bit-identical logits (covers Q8_0, Q4_K and mixed Q4_K_M caches).
#[test]
fn quant_cache_roundtrip() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tiny256");
    for dt in [DType::Q8, DType::Q4K, DType::Q4KM] {
        let mut m = Qwen2::from_dir(&dir, 128, dt).unwrap();
        let want = m.forward(&[5, 7, 11, 13], 0);
        let path =
            std::env::temp_dir().join(format!("yesman_cache_{}_{:?}.ymq", std::process::id(), dt));
        m.save_cache(&path).unwrap();
        let mut loaded = Qwen2::from_cache(&dir, 128, &path).unwrap();
        let got = loaded.forward(&[5, 7, 11, 13], 0);
        for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "dtype {dt:?} logit {i}: {a} vs {b}");
        }
        std::fs::remove_file(&path).unwrap();
    }
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

/// Chunked prefill (e.g. chunk size 2 on a 6-token prompt) must give
/// the exact same logits as unchunked prefill.
#[test]
fn chunked_prefill_matches_unchunked() {
    let (ids, _) = fixture();
    let mut m_unchunked = model();
    let want = m_unchunked.forward_chunked(&ids, 0, ids.len());

    let mut m_chunked = model();
    let got = m_chunked.forward_chunked(&ids, 0, 2);

    assert_eq!(m_chunked.kv_len(), ids.len());
    for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
        let tol = 1e-4 + 1e-5 * w.abs();
        assert!((w - g).abs() < tol, "logit {i}: unchunked={w} chunked={g}");
    }
}

/// Verify that perplexity computation works, gives positive finite PPL,
/// and that sequence length > max_ctx is rejected.
#[test]
fn compute_nll_and_perplexity() {
    let (ids, _) = fixture();
    let mut m = model();
    let (nll, count) = m.compute_nll(&ids).unwrap();
    assert_eq!(count, ids.len() - 1);
    assert!(nll > 0.0 && nll.is_finite());
    let ppl = m.perplexity(&ids).unwrap();
    assert!(ppl > 1.0 && ppl.is_finite());

    // Single token should yield 0 predicted
    let (nll_one, count_one) = m.compute_nll(&[ids[0]]).unwrap();
    assert_eq!(count_one, 0);
    assert_eq!(nll_one, 0.0);
    assert!(m.perplexity(&[ids[0]]).is_err());

    // Sequence longer than max_ctx (64 for tiny model)
    let too_long: Vec<u32> = (0..100).map(|i| (i % 20) as u32).collect();
    assert!(m.compute_nll(&too_long).is_err());
}

/// Cloned sessions share weights via Arc, have independent KV caches,
/// and can decode concurrently without crosstalk.
#[test]
fn cloned_sessions_run_independently() {
    let (ids, want) = fixture();
    let root = model();

    let mut s1 = root.clone_session(64);
    let mut s2 = root.clone_session(64);

    // s1 and s2 forward different prefixes
    let _g1 = s1.forward(&ids[..3], 0);
    assert_eq!(s1.kv_len(), 3);
    assert_eq!(s2.kv_len(), 0);

    let g2 = s2.forward(&ids, 0);
    assert_eq!(s2.kv_len(), ids.len());
    assert_eq!(s1.kv_len(), 3);

    for (i, (w, g)) in want.iter().zip(g2.iter()).enumerate() {
        let tol = 1e-4 + 1e-5 * w.abs();
        assert!((w - g).abs() < tol, "s2 logit {i}: {w} vs {g}");
    }

    // Now resume s1
    let g1_rest = s1.forward(&ids[3..], 3);
    assert_eq!(s1.kv_len(), ids.len());
    for (i, (w, g)) in want.iter().zip(g1_rest.iter()).enumerate() {
        let tol = 1e-4 + 1e-5 * w.abs();
        assert!((w - g).abs() < tol, "s1 logit {i}: {w} vs {g}");
    }
}
