use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
    time::Instant,
};

use anyhow::Result;
use clap::Parser;
use yesman_model::{
    Config, DType, GenerateOptions, GenerationConfig, Qwen2, Sampler, Tokenizer, chat, chat_once,
};

#[derive(Parser, Debug)]
#[command(
    name = "yesman",
    about = "Qwen2.5 inference, pure Rust, no ML frameworks"
)]
struct Args {
    /// Directory with config.json / tokenizer.json / model.safetensors
    #[arg(short, long, default_value = "models/Qwen2.5-1.5B-Instruct")]
    model: PathBuf,

    /// Context window for the KV cache
    #[arg(long, default_value_t = 4096)]
    ctx: usize,

    /// One-shot prompt; omit for an interactive chat
    #[arg(short, long)]
    prompt: Option<String>,

    #[arg(long, default_value_t = 512)]
    max_tokens: usize,

    #[arg(long)]
    temperature: Option<f32>,

    #[arg(long)]
    top_p: Option<f32>,

    #[arg(long)]
    top_k: Option<usize>,

    #[arg(long)]
    repetition_penalty: Option<f32>,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long)]
    system: Option<String>,

    /// Treat --prompt as raw text (skip the ChatML template)
    #[arg(long)]
    raw: bool,

    /// Print the top-N logits for the prompt and exit (verification hook)
    #[arg(long)]
    top_logits: Option<usize>,

    /// Print the token ids for the prompt and exit (verification hook)
    #[arg(long)]
    print_ids: bool,

    /// Weights: f32 | bf16 (default) | q8 | q4 | q4k (all-4bit) | q4km (FFN@Q8)
    #[arg(long, default_value = "bf16")]
    dtype: String,

    /// Load pre-quantized weights from this cache instead of safetensors
    #[arg(long)]
    cache: Option<PathBuf>,

    /// Quantize, save to this cache and exit (build once, start fast after)
    #[arg(long)]
    save_cache: Option<PathBuf>,

    /// Calculate perplexity on the provided text file and exit
    #[arg(long)]
    perplexity: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = Config::from_dir(&args.model)?;
    let gen_cfg = GenerationConfig::from_dir(&args.model).unwrap_or_default();

    let mut sampler = Sampler::new(args.seed);
    sampler.temperature = args.temperature.or(gen_cfg.temperature).unwrap_or(0.7);
    sampler.top_p = args.top_p.or(gen_cfg.top_p).unwrap_or(0.8);
    sampler.top_k = args.top_k.or(gen_cfg.top_k).unwrap_or(20);
    sampler.repetition_penalty = args
        .repetition_penalty
        .or(gen_cfg.repetition_penalty)
        .unwrap_or(1.1);

    let tok = Tokenizer::from_dir(&args.model)?;

    if args.print_ids {
        let ids = tok.encode(&prompt_text(&args))?;
        println!(
            "{}",
            ids.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        return Ok(());
    }

    let t0 = Instant::now();
    let dtype = DType::parse(&args.dtype)?;
    let mut model = match &args.cache {
        Some(p) => Qwen2::from_cache(&args.model, args.ctx, p)?,
        None => Qwen2::from_dir(&args.model, args.ctx, dtype)?,
    };
    eprintln!(
        "loaded {} layers, hidden {}, {} heads / {} kv-heads, {} weights, in {:.2}s",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        if args.cache.is_some() {
            "cached"
        } else {
            dtype.name()
        },
        t0.elapsed().as_secs_f32()
    );

    if let Some(p) = &args.save_cache {
        model.save_cache(p)?;
        eprintln!(
            "saved quant cache to {} ({:.2}s total)",
            p.display(),
            t0.elapsed().as_secs_f32()
        );
        return Ok(());
    }

    if let Some(p) = &args.perplexity {
        let text = std::fs::read_to_string(p)?;
        let ids = tok.encode(&text)?;
        if ids.len() < 2 {
            anyhow::bail!("corpus has fewer than 2 tokens; cannot compute perplexity");
        }
        let max_ctx = model.max_ctx();
        let eval_ids = if ids.len() > max_ctx {
            eprintln!(
                "warning: corpus length ({} tokens) exceeds model max_ctx ({}); evaluating first {} tokens",
                ids.len(),
                max_ctx,
                max_ctx
            );
            &ids[..max_ctx]
        } else {
            &ids[..]
        };
        let t_eval = Instant::now();
        let (nll, count) = model.compute_nll(eval_ids)?;
        let elapsed = t_eval.elapsed().as_secs_f64();
        let ppl = (nll / count as f64).exp();
        let bpc = nll / (count as f64 * std::f64::consts::LN_2);
        eprintln!(
            "evaluated {count} tokens in {elapsed:.2}s ({:.1} tok/s): nll={nll:.4}, bpc={bpc:.3}",
            count as f64 / elapsed.max(1e-6)
        );
        println!("perplexity: {ppl:.4}");
        return Ok(());
    }

    if let Some(n) = args.top_logits {
        let text = prompt_text(&args);
        let ids = tok.encode(&text)?;
        let logits = model.forward(&ids, 0);
        top_n(&logits, n);
        return Ok(());
    }

    let opts = GenerateOptions {
        max_new_tokens: args.max_tokens,
        max_ctx: args.ctx,
        ..Default::default()
    };

    if let Some(p) = &args.prompt {
        let messages = if args.raw {
            vec![chat::user(p.clone())]
        } else {
            vec![system_message(&args), chat::user(p.clone())]
        };
        print_reply(&mut model, &tok, &cfg, &messages, &opts, &mut sampler)?;
        return Ok(());
    }

    // --- interactive REPL -----------------------------------------------------
    let mut history: Vec<chat::Message> = vec![system_message(&args)];
    let stdin = io::stdin();
    println!(
        "yesman — Qwen2.5 в чистом Rust. Пустая строка или /exit — выход, /reset — сброс истории.\n"
    );
    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() || line == "/exit" || line == "/quit" {
            break;
        }
        if line == "/reset" || line == "/clear" {
            history = vec![system_message(&args)];
            model.reset();
            println!("История диалога и KV-кэш сброшены.\n");
            continue;
        }
        history.push(chat::user(line.to_string()));
        let reply = print_reply(&mut model, &tok, &cfg, &history, &opts, &mut sampler)?;
        history.push(chat::assistant(reply));
        println!();
    }
    Ok(())
}

fn system_message(args: &Args) -> chat::Message {
    chat::system(
        args.system
            .clone()
            .unwrap_or_else(|| chat::DEFAULT_SYSTEM.to_string()),
    )
}

fn prompt_text(args: &Args) -> String {
    let p = args.prompt.clone().unwrap_or_default();
    if args.raw {
        p
    } else {
        chat::render(&[system_message(args), chat::user(p)])
    }
}

fn print_reply(
    model: &mut Qwen2,
    tok: &Tokenizer,
    cfg: &Config,
    messages: &[chat::Message],
    opts: &GenerateOptions,
    sampler: &mut Sampler,
) -> Result<String> {
    let t0 = Instant::now();
    let mut tokens = 0usize;
    let mut first_token_at = t0;
    let text = chat_once(model, tok, cfg, messages, opts, sampler, |piece| {
        if tokens == 0 {
            first_token_at = Instant::now();
        }
        print!("{piece}");
        io::stdout().flush()?;
        tokens += 1;
        Ok(())
    })?;
    let prefill = first_token_at.duration_since(t0).as_secs_f32();
    let decode = first_token_at.elapsed().as_secs_f32();
    eprintln!(
        "\n[prefill {prefill:.2}s, {tokens} tokens decode {decode:.2}s = {:.1} tok/s]",
        tokens as f32 / decode.max(1e-6)
    );
    Ok(text)
}

fn top_n(logits: &[f32], n: usize) {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|a, b| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for id in order.into_iter().take(n) {
        println!("{}\t{:.6}", id, logits[id]);
    }
}
