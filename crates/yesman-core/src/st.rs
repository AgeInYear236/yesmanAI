//! Minimal safetensors reader: mmap the file, convert tensors to f32.
//!
//! The `safetensors` crate is a pure-Rust (serde-based) container format
//! reader, not an ML framework -- it does no math and pulls in no BLAS.

use std::{collections::HashMap, fs::File, path::Path};

use anyhow::{Context, Result, anyhow};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::{Bf16Tensor, Tensor};

pub struct SafeFile {
    mmap: Mmap,
    names: Vec<String>,
}

impl SafeFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let names: Vec<String> = Self::parse(&mmap)?
            .names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        Ok(Self { mmap, names })
    }

    fn parse<'a>(bytes: &'a [u8]) -> Result<SafeTensors<'a>> {
        Ok(SafeTensors::deserialize(bytes)?)
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Tensor converted to f32, shape preserved (row-major, same as PyTorch).
    pub fn get(&self, name: &str) -> Result<Tensor> {
        let st = Self::parse(&self.mmap)?;
        let view = st
            .tensor(name)
            .map_err(|e| anyhow!("tensor `{name}` not found: {e}"))?;
        let shape = view.shape().to_vec();
        let data = to_f32(view.dtype(), view.data())?;
        Tensor::new(data, shape)
    }

    /// Shape only -- cheap, no data copy.
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        let st = Self::parse(&self.mmap)?;
        Ok(st.tensor(name)?.shape().to_vec())
    }

    /// Tensor kept in bf16 (the dtype Qwen2.5 ships), no f32 detour.
    pub fn get_bf16(&self, name: &str) -> Result<Bf16Tensor> {
        let st = Self::parse(&self.mmap)?;
        let view = st.tensor(name)?;
        let shape = view.shape().to_vec();
        let (rows, cols) = match shape.as_slice() {
            [r, c] => (*r, *c),
            other => return Err(anyhow!("{name}: expected 2-D tensor, got {other:?}")),
        };
        use safetensors::Dtype::*;
        let data = match view.dtype() {
            BF16 => view
                .data()
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]))
                .collect(),
            F32 => view
                .data()
                .chunks_exact(4)
                .map(|c| bf16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect(),
            F16 => view
                .data()
                .chunks_exact(2)
                .map(|c| bf16::from_f32(f16::from_le_bytes([c[0], c[1]]).to_f32()))
                .collect(),
            other => return Err(anyhow!("{name}: unsupported dtype {other:?}")),
        };
        Ok(Bf16Tensor::from_raw(data, rows, cols))
    }

    /// Load every tensor. Convenient, but it materializes the whole file in RAM.
    pub fn load_all(&self) -> Result<HashMap<String, Tensor>> {
        let st = Self::parse(&self.mmap)?;
        self.names
            .iter()
            .map(|name| {
                let view = st.tensor(name)?;
                let t = Tensor::new(to_f32(view.dtype(), view.data())?, view.shape().to_vec())?;
                Ok((name.clone(), t))
            })
            .collect()
    }
}

fn to_f32(dtype: safetensors::Dtype, data: &[u8]) -> Result<Vec<f32>> {
    use safetensors::Dtype::*;
    Ok(match dtype {
        F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        F16 => data
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        BF16 => data
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => return Err(anyhow!("unsupported dtype {other:?}")),
    })
}
