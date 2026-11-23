use std::path::Path;

use anyhow::Result;
use tokenizers::Tokenizer as HfTokenizer;

/// Thin wrapper over the official `tokenizers` crate (pure Rust BPE).
pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("tokenizer.json");
        let inner =
            HfTokenizer::from_file(&p).map_err(|e| anyhow::anyhow!("load {}: {e}", p.display()))?;
        Ok(Self { inner })
    }

    /// `add_special_tokens = false`: the chat template already contains the
    /// `<|im_start|>` markers, adding BOS on top would double them.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("detokenize: {e}"))
    }

    pub fn id_of(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
