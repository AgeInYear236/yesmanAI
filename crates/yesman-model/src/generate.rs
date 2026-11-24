use anyhow::{Result, anyhow};

use crate::{chat, config::Config, qwen2::Qwen2, sampler::Sampler, tokenizer::Tokenizer};

#[derive(Clone)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub stop_ids: Vec<u32>,
    pub max_ctx: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            stop_ids: Vec::new(),
            max_ctx: 4096,
        }
    }
}

/// Prefill the prompt, then decode token by token, invoking `on_token` with the
/// incremental text so the caller can stream to stdout.
pub fn generate(
    model: &mut Qwen2,
    tok: &Tokenizer,
    prompt_ids: &[u32],
    opts: &GenerateOptions,
    sampler: &mut Sampler,
    mut on_token: impl FnMut(&str) -> Result<()>,
) -> Result<(Vec<u32>, String)> {
    if prompt_ids.is_empty() {
        return Err(anyhow!("empty prompt"));
    }
    let prompt_slice = if prompt_ids.len() >= opts.max_ctx {
        let budget = opts.max_ctx.saturating_sub(16).max(1);
        let start = prompt_ids.len().saturating_sub(budget);
        &prompt_ids[start..]
    } else {
        prompt_ids
    };

    model.reset();
    let mut history = prompt_slice.to_vec();
    let mut logits = model.forward(prompt_slice, 0);
    let mut out_ids: Vec<u32> = Vec::new();
    let mut text = String::new();
    let mut pos = prompt_slice.len();

    for _ in 0..opts.max_new_tokens {
        let next = sampler.sample(&mut logits, &history);
        if opts.stop_ids.contains(&next) {
            break;
        }
        let piece = tok.decode(&[next])?;
        out_ids.push(next);
        text.push_str(&piece);
        on_token(&piece)?;
        history.push(next);
        pos += 1;
        if pos >= opts.max_ctx {
            break;
        }
        logits = model.forward(&[next], pos - 1);
    }

    Ok((out_ids, text))
}

/// Convenience: render ChatML, tokenize, generate, return the reply text.
pub fn chat_once(
    model: &mut Qwen2,
    tok: &Tokenizer,
    cfg: &Config,
    messages: &[chat::Message],
    opts: &GenerateOptions,
    sampler: &mut Sampler,
    on_token: impl FnMut(&str) -> Result<()>,
) -> Result<String> {
    let prompt = chat::render(messages);
    let ids = tok.encode(&prompt)?;
    let mut stop = cfg.eos_ids();
    if let Some(id) = tok.id_of(chat::IM_END)
        && !stop.contains(&id)
    {
        stop.push(id);
    }
    if let Some(id) = tok.id_of("<|endoftext|>")
        && !stop.contains(&id)
    {
        stop.push(id);
    }
    let opts = GenerateOptions {
        stop_ids: stop,
        ..opts.clone()
    };
    let (_, text) = generate(model, tok, &ids, &opts, sampler, on_token)?;
    Ok(text)
}
