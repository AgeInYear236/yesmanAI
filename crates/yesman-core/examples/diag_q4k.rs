//! Diagnostic: dequantization error of Q4_K vs Q4_0 on a real (tiny) fixture.
//! `cargo run --release -p yesman-core --example diag_q4k`

use std::path::Path;

use yesman_core::{Q4KTensor, SafeFile, q4::Q4Tensor};

fn main() -> anyhow::Result<()> {
    let p = Path::new("crates/yesman-model/tests/data/tiny256/model.safetensors");
    let f = SafeFile::open(p)?;
    for name in f.names() {
        if !name.ends_with("weight") || name.contains("layernorm") || name.contains("norm") {
            continue;
        }
        let t = f.get(name)?;
        if t.shape.len() != 2 || t.cols() % 256 != 0 {
            continue;
        }
        let q4k = Q4KTensor::from_f32(&t.data, t.rows(), t.cols());
        let q4 = Q4Tensor::from_f32(&t.data, t.rows(), t.cols());
        let mut ek = 0.0f64;
        let mut e4 = 0.0f64;
        let mut norm = 0.0f64;
        for r in 0..t.rows() {
            let src = t.row(r);
            let a = q4k.dequantize_row(r);
            let b = q4.dequantize_row(r);
            for i in 0..t.cols() {
                ek += (a[i] as f64 - src[i] as f64).powi(2);
                e4 += (b[i] as f64 - src[i] as f64).powi(2);
                norm += (src[i] as f64).powi(2);
            }
        }
        println!(
            "{:60} Q4_K {:6.2}%  Q4_0 {:6.2}%",
            name,
            100.0 * (ek / norm).sqrt(),
            100.0 * (e4 / norm).sqrt()
        );
    }
    Ok(())
}
