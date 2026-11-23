//! Persistent cache for quantized weights.
//!
//! Quantizing the 3 GB checkpoint takes ~6 s on every startup; the cache
//! stores the already quantized blocks plus the small f32 tensors (norms and
//! biases), so loading becomes a single mmap + memcpy.
//!
//! Layout: 64-byte fixed header (magic, version, payload offset), a JSON
//! table of contents and then the raw block arrays / f32 arrays, 64-aligned.

use std::{collections::HashMap, fs::File, io::Write, path::Path};

use anyhow::{Result, anyhow, bail};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::{Q4Tensor, Q8Tensor};

const MAGIC: &[u8; 8] = b"YESMANQ1";
const HEADER_LEN: usize = 64;
const ALIGN: usize = 64;

/// 0 = f32 array, 1 = Q8_0 blocks, 2 = Q4_0 blocks.
pub const KIND_F32: u8 = 0;
pub const KIND_Q8: u8 = 1;
pub const KIND_Q4: u8 = 2;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Entry {
    pub kind: u8,
    pub rows: usize,
    pub cols: usize,
    pub offset: u64,
    pub len: u64,
}

#[derive(Serialize, Deserialize)]
struct Header {
    entries: HashMap<String, Entry>,
}

pub struct CacheEntry<'a> {
    pub name: &'a str,
    pub kind: u8,
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [u8],
}

pub fn write_cache(path: &Path, entries: &[CacheEntry<'_>]) -> Result<()> {
    let mut toc = Header { entries: HashMap::new() };
    let mut offset = 0u64;
    for e in entries {
        toc.entries.insert(
            e.name.to_string(),
            Entry {
                kind: e.kind,
                rows: e.rows,
                cols: e.cols,
                offset,
                len: e.data.len() as u64,
            },
        );
        offset += e.data.len() as u64;
        offset += ((ALIGN - (offset as usize % ALIGN)) % ALIGN) as u64;
    }

    let toc_bytes = serde_json::to_vec(&toc)?;
    let data_start = (HEADER_LEN + toc_bytes.len() + ALIGN - 1) / ALIGN * ALIGN;

    // Stream straight from the entries: no giant intermediate Vec.
    let mut f = std::io::BufWriter::with_capacity(1 << 20, File::create(path)?);
    let mut head = [0u8; HEADER_LEN];
    head[..8].copy_from_slice(MAGIC);
    head[8] = 1; // version
    head[16..24].copy_from_slice(&(toc_bytes.len() as u64).to_le_bytes());
    head[24..32].copy_from_slice(&(data_start as u64).to_le_bytes());
    f.write_all(&head)?;
    f.write_all(&toc_bytes)?;
    f.write_all(&vec![0u8; data_start - HEADER_LEN - toc_bytes.len()])?;
    for e in entries {
        f.write_all(e.data)?;
        let pad = (ALIGN - (e.data.len() % ALIGN)) % ALIGN;
        f.write_all(&vec![0u8; pad])?;
    }
    f.flush()?;
    Ok(())
}

pub struct QuantCache {
    mmap: Mmap,
    toc: HashMap<String, Entry>,
    /// Byte offset in the file where the payload starts.
    base: usize,
}

impl QuantCache {
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        if mmap.len() < HEADER_LEN || &mmap[..8] != MAGIC {
            bail!("not a yesman quant cache: {}", path.display());
        }
        if mmap[8] != 1 {
            bail!("unsupported cache version {}", mmap[8]);
        }
        let toc_len = u64::from_le_bytes(mmap[16..24].try_into()?) as usize;
        let base = u64::from_le_bytes(mmap[24..32].try_into()?) as usize;
        let toc: Header = serde_json::from_slice(&mmap[HEADER_LEN..HEADER_LEN + toc_len])?;
        Ok(Self { mmap, toc: toc.entries, base })
    }

    pub fn has(&self, name: &str) -> bool {
        self.toc.contains_key(name)
    }

    pub fn entry(&self, name: &str) -> Result<&Entry> {
        self.toc
            .get(name)
            .ok_or_else(|| anyhow!("tensor `{name}` missing from cache"))
    }

    fn raw(&self, name: &str) -> Result<&[u8]> {
        let e = self.entry(name)?;
        let s = self.base + e.offset as usize;
        let t = s + e.len as usize;
        if t > self.mmap.len() {
            bail!("cache truncated at tensor `{name}`");
        }
        Ok(&self.mmap[s..t])
    }

    pub fn f32_vec(&self, name: &str) -> Result<Vec<f32>> {
        let e = self.entry(name)?;
        if e.kind != KIND_F32 {
            bail!("tensor `{name}` is not f32");
        }
        let raw = self.raw(name)?;
        assert_eq!(raw.len() % 4, 0);
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn q8(&self, name: &str) -> Result<Q8Tensor> {
        let e = self.entry(name)?;
        if e.kind != KIND_Q8 {
            bail!("tensor `{name}` is not Q8_0");
        }
        Ok(Q8Tensor::from_bytes(e.rows, e.cols, self.raw(name)?))
    }

    pub fn q4(&self, name: &str) -> Result<Q4Tensor> {
        let e = self.entry(name)?;
        if e.kind != KIND_Q4 {
            bail!("tensor `{name}` is not Q4_0");
        }
        Ok(Q4Tensor::from_bytes(e.rows, e.cols, self.raw(name)?))
    }
}
