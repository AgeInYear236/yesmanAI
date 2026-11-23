//! Clean-room GGUF (v2 / v3) binary parser.
//!
//! Parses GGUF container headers, metadata key-values, and tensor headers with
//! zero-copy mmap tensor data slicing. Supports common GGML tensor types:
//! F32, F16, BF16, Q8_0, Q4_0, Q4_K.

use std::{collections::HashMap, fs::File, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

pub const GGUF_TYPE_UINT8: u32 = 0;
pub const GGUF_TYPE_INT8: u32 = 1;
pub const GGUF_TYPE_UINT16: u32 = 2;
pub const GGUF_TYPE_INT16: u32 = 3;
pub const GGUF_TYPE_UINT32: u32 = 4;
pub const GGUF_TYPE_INT32: u32 = 5;
pub const GGUF_TYPE_FLOAT32: u32 = 6;
pub const GGUF_TYPE_BOOL: u32 = 7;
pub const GGUF_TYPE_STRING: u32 = 8;
pub const GGUF_TYPE_ARRAY: u32 = 9;
pub const GGUF_TYPE_UINT64: u32 = 10;
pub const GGUF_TYPE_INT64: u32 = 11;
pub const GGUF_TYPE_FLOAT64: u32 = 12;

// GGML Tensor Type IDs
pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q4_1: u32 = 3;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_BF16: u32 = 28;

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: u32,
    pub offset: u64,
}

pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    pub data_offset: usize,
    pub alignment: usize,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&f)? };
        Self::from_mmap(mmap)
    }

    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        let mut cur = Cursor::new(&mmap);
        let magic = cur.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            bail!(
                "invalid GGUF magic: expected {:?}, got {:?}",
                GGUF_MAGIC,
                magic
            );
        }

        let version = cur.read_u32()?;
        if version != 2 && version != 3 {
            bail!("unsupported GGUF version {version} (expected 2 or 3)");
        }

        let tensor_count = cur.read_u64()? as usize;
        let metadata_kv_count = cur.read_u64()? as usize;

        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = cur.read_string()?;
            let val_type = cur.read_u32()?;
            let val = cur.read_value(val_type)?;
            metadata.insert(key, val);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::Uint32(a)) => *a as usize,
            Some(GgufValue::Uint64(a)) => *a as usize,
            _ => 32,
        };

        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cur.read_u64()? as usize);
            }
            // GGUF stores dimensions in column-major order; reverse for row-major
            dims.reverse();

            let ggml_type = cur.read_u32()?;
            let offset = cur.read_u64()?;
            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    shape: dims,
                    ggml_type,
                    offset,
                },
            );
        }

        // Align data offset to `alignment` bytes
        let current_pos = cur.pos;
        let data_offset = current_pos.div_ceil(alignment) * alignment;
        if data_offset > mmap.len() {
            bail!("GGUF header points beyond end of file");
        }

        Ok(Self {
            mmap,
            version,
            metadata,
            tensors,
            data_offset,
            alignment,
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|k| k.as_str())
    }

    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("tensor `{name}` not found in GGUF"))?;
        let start = self.data_offset + info.offset as usize;
        if start > self.mmap.len() {
            bail!("tensor `{name}` offset is out of bounds");
        }
        Ok(&self.mmap[start..])
    }

    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(GgufValue::Uint32(v)) => Some(*v),
            Some(GgufValue::Uint64(v)) => Some(*v as u32),
            Some(GgufValue::Int32(v)) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn metadata_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key) {
            Some(GgufValue::Uint64(v)) => Some(*v),
            Some(GgufValue::Uint32(v)) => Some(*v as u64),
            Some(GgufValue::Int64(v)) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(GgufValue::Float32(v)) => Some(*v),
            Some(GgufValue::Float64(v)) => Some(*v as f32),
            _ => None,
        }
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            bail!(
                "unexpected EOF parsing GGUF: need {n} bytes at {}",
                self.pos
            );
        }
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        Ok(String::from_utf8(bytes.to_vec())?)
    }

    fn read_value(&mut self, val_type: u32) -> Result<GgufValue> {
        match val_type {
            GGUF_TYPE_UINT8 => Ok(GgufValue::Uint8(self.read_u8()?)),
            GGUF_TYPE_INT8 => Ok(GgufValue::Int8(self.read_i8()?)),
            GGUF_TYPE_UINT16 => Ok(GgufValue::Uint16(self.read_u16()?)),
            GGUF_TYPE_INT16 => Ok(GgufValue::Int16(self.read_i16()?)),
            GGUF_TYPE_UINT32 => Ok(GgufValue::Uint32(self.read_u32()?)),
            GGUF_TYPE_INT32 => Ok(GgufValue::Int32(self.read_i32()?)),
            GGUF_TYPE_FLOAT32 => Ok(GgufValue::Float32(self.read_f32()?)),
            GGUF_TYPE_BOOL => Ok(GgufValue::Bool(self.read_bool()?)),
            GGUF_TYPE_STRING => Ok(GgufValue::String(self.read_string()?)),
            GGUF_TYPE_ARRAY => {
                let elem_type = self.read_u32()?;
                let len = self.read_u64()? as usize;
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array(items))
            }
            GGUF_TYPE_UINT64 => Ok(GgufValue::Uint64(self.read_u64()?)),
            GGUF_TYPE_INT64 => Ok(GgufValue::Int64(self.read_i64()?)),
            GGUF_TYPE_FLOAT64 => Ok(GgufValue::Float64(self.read_f64()?)),
            other => bail!("unsupported GGUF value type {other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_binary_builder_and_parse_roundtrip() {
        let mut buf = Vec::new();

        // Magic + version
        buf.extend_from_slice(GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes()); // v3
        buf.extend_from_slice(&2u64.to_le_bytes()); // 2 tensors
        buf.extend_from_slice(&3u64.to_le_bytes()); // 3 metadata entries

        // Metadata 1: general.architecture = "qwen2"
        let k1 = "general.architecture";
        buf.extend_from_slice(&(k1.len() as u64).to_le_bytes());
        buf.extend_from_slice(k1.as_bytes());
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        let v1 = "qwen2";
        buf.extend_from_slice(&(v1.len() as u64).to_le_bytes());
        buf.extend_from_slice(v1.as_bytes());

        // Metadata 2: qwen2.block_count = 28u32
        let k2 = "qwen2.block_count";
        buf.extend_from_slice(&(k2.len() as u64).to_le_bytes());
        buf.extend_from_slice(k2.as_bytes());
        buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        buf.extend_from_slice(&28u32.to_le_bytes());

        // Metadata 3: general.alignment = 32u32
        let k3 = "general.alignment";
        buf.extend_from_slice(&(k3.len() as u64).to_le_bytes());
        buf.extend_from_slice(k3.as_bytes());
        buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        buf.extend_from_slice(&32u32.to_le_bytes());

        // Tensor 1: "token_embd.weight", shape [256, 128], type F32, offset 0
        let t1_name = "token_embd.weight";
        buf.extend_from_slice(&(t1_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(t1_name.as_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // 2 dims
        buf.extend_from_slice(&128u64.to_le_bytes()); // GGUF column-major: dim0 = cols = 128
        buf.extend_from_slice(&256u64.to_le_bytes()); // dim1 = rows = 256
        buf.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset 0

        // Tensor 2: "output_norm.weight", shape [128], type F32, offset 256*128*4 = 131072
        let t2_name = "output_norm.weight";
        buf.extend_from_slice(&(t2_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(t2_name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 dim
        buf.extend_from_slice(&128u64.to_le_bytes()); // dim0 = 128
        buf.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
        buf.extend_from_slice(&131072u64.to_le_bytes()); // offset 131072

        // Pad to alignment 32
        let cur_len = buf.len();
        let aligned = cur_len.div_ceil(32) * 32;
        buf.resize(aligned, 0u8);

        // Append tensor payload: 131072 bytes + 512 bytes = 131584 bytes
        buf.resize(aligned + 131072 + 512, 42u8);

        let temp_path = std::env::temp_dir().join(format!("test_gguf_{}.gguf", std::process::id()));
        std::fs::write(&temp_path, &buf).unwrap();

        let gguf = GgufFile::open(&temp_path).unwrap();
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.metadata_str("general.architecture"), Some("qwen2"));
        assert_eq!(gguf.metadata_u32("qwen2.block_count"), Some(28));
        assert_eq!(gguf.alignment, 32);

        let t1 = gguf.tensor_info("token_embd.weight").unwrap();
        assert_eq!(t1.shape, vec![256, 128]); // reversed to row-major [rows, cols]
        assert_eq!(t1.ggml_type, GGML_TYPE_F32);
        assert_eq!(t1.offset, 0);

        let t2 = gguf.tensor_info("output_norm.weight").unwrap();
        assert_eq!(t2.shape, vec![128]);
        assert_eq!(t2.ggml_type, GGML_TYPE_F32);
        assert_eq!(t2.offset, 131072);

        let t1_data = gguf.tensor_data("token_embd.weight").unwrap();
        assert_eq!(t1_data[0], 42);

        std::fs::remove_file(&temp_path).unwrap();
    }
}
