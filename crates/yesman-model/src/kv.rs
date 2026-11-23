//! Per-layer key/value cache. Rows are written at absolute positions, so a
//! decode step only pays for the new token while prefill writes a whole block.

pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    kv_dim: usize,
    len: usize,
    max: usize,
}

impl KvCache {
    pub fn new(n_layers: usize, max_ctx: usize, kv_dim: usize) -> Self {
        Self {
            k: vec![vec![0.0; max_ctx * kv_dim]; n_layers],
            v: vec![vec![0.0; max_ctx * kv_dim]; n_layers],
            kv_dim,
            len: 0,
            max: max_ctx,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn write(&mut self, layer: usize, pos: usize, krow: &[f32], vrow: &[f32]) {
        assert!(pos < self.max, "kv cache overflow: pos {pos} >= {}", self.max);
        let off = pos * self.kv_dim;
        self.k[layer][off..off + self.kv_dim].copy_from_slice(krow);
        self.v[layer][off..off + self.kv_dim].copy_from_slice(vrow);
    }

    pub fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    pub fn k(&self, layer: usize, t: usize) -> &[f32] {
        let off = t * self.kv_dim;
        &self.k[layer][off..off + self.kv_dim]
    }

    pub fn v(&self, layer: usize, t: usize) -> &[f32] {
        let off = t * self.kv_dim;
        &self.v[layer][off..off + self.kv_dim]
    }
}
