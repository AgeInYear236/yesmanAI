use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

impl Config {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(dir.join("config.json"))?;
        Ok(serde_json::from_str(&s)?)
    }

    /// Qwen2 derives head_dim from hidden_size / num_heads (there is no
    /// `head_dim` field in the released config).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    /// Number of query heads sharing one key/value head (grouped-query attention).
    pub fn gqa_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn eos_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .filter_map(|v| v.as_u64().map(|v| v as u32))
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// Defaults from `generation_config.json` (temperature, top-p, top-k, ...).
#[derive(Debug, Clone, Deserialize, Default)]
pub struct GenerationConfig {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub do_sample: Option<bool>,
}

impl GenerationConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("generation_config.json");
        if !p.exists() {
            return Ok(Self::default());
        }
        Ok(serde_json::from_str(&std::fs::read_to_string(p)?)?)
    }
}
