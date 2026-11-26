use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

use anyhow::Result;
use clap::Parser;
use yesman_cuda::{CudaDevice, CudaQwen2, is_cuda_available};
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

    /// Execution backend: auto | cpu | cuda
    #[arg(long, default_value = "auto")]
    device: String,

    /// Context window for the KV cache
    #[arg(long, default_value_t = 8192)]
    ctx: usize,

    /// One-shot prompt; omit for an interactive chat
    #[arg(short, long)]
    prompt: Option<String>,

    /// Max tokens per assistant turn (a detailed bpy script + plan is ~2.5k).
    #[arg(long, default_value_t = 3072)]
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

    /// Connect to an MCP server via stdio command (e.g. "uvx mcp-for-blender")
    #[arg(long)]
    mcp: Option<String>,

    /// Auto-connect to local Blender MCP server (mcp-for-blender)
    #[arg(long)]
    blender: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let is_gguf = args.model.is_file()
        && args
            .model
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);

    let model_dir = if is_gguf {
        args.model.parent().unwrap_or(std::path::Path::new("."))
    } else {
        args.model.as_path()
    };

    let (cfg, gen_cfg) = if is_gguf {
        let temp_m = Qwen2::from_gguf(&args.model, args.ctx)?;
        let c = temp_m.cfg.clone();
        let g = GenerationConfig::from_dir(model_dir).unwrap_or_default();
        (c, g)
    } else {
        let c = Config::from_dir(model_dir)?;
        let g = GenerationConfig::from_dir(model_dir).unwrap_or_default();
        (c, g)
    };

    let mut sampler = Sampler::new(args.seed);
    sampler.temperature = args.temperature.or(gen_cfg.temperature).unwrap_or(0.7);
    sampler.top_p = args.top_p.or(gen_cfg.top_p).unwrap_or(0.8);
    sampler.top_k = args.top_k.or(gen_cfg.top_k).unwrap_or(20);
    sampler.repetition_penalty = args
        .repetition_penalty
        .or(gen_cfg.repetition_penalty)
        .unwrap_or(1.1);

    let tok = if model_dir.join("tokenizer.json").exists() {
        Tokenizer::from_dir(model_dir)?
    } else {
        Tokenizer::from_dir(std::path::Path::new("models/Qwen2.5-1.5B-Instruct"))?
    };

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
    let mut model = if is_gguf {
        Qwen2::from_gguf(&args.model, args.ctx)?
    } else {
        match &args.cache {
            Some(p) => Qwen2::from_cache(&args.model, args.ctx, p)?,
            None => Qwen2::from_dir(&args.model, args.ctx, dtype)?,
        }
    };
    eprintln!(
        "loaded {} layers, hidden {}, {} heads / {} kv-heads, {} weights, in {:.2}s",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        if is_gguf {
            "GGUF"
        } else if args.cache.is_some() {
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

    let use_cuda = match args.device.to_lowercase().as_str() {
        "cuda" | "gpu" => {
            if !is_cuda_available() {
                anyhow::bail!("CUDA is not available on this machine (driver not found)");
            }
            true
        }
        "cpu" => false,
        "auto" => is_cuda_available(),
        other => anyhow::bail!("unknown device `{other}` (auto|cpu|cuda)"),
    };

    let mut engine = if use_cuda {
        let dev = CudaDevice::new(0)?;
        let ctx = dev.create_context()?;
        eprintln!(
            "using GPU backend: {} ({:.2} GB VRAM)",
            dev.name,
            dev.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        let gpu_model = CudaQwen2::from_cpu(&model, Arc::clone(&ctx), args.ctx)?;
        Engine::Gpu(Box::new(gpu_model), model)
    } else {
        eprintln!("using CPU backend (AVX2/FMA)");
        Engine::Cpu(model)
    };

    let mut mcp_client = if args.blender {
        let default_cmd = "uvx mcp-for-blender";
        eprintln!("connecting to Blender MCP server ({default_cmd})...");
        Some(yesman_model::McpClient::spawn(default_cmd)?)
    } else if let Some(cmd) = &args.mcp {
        eprintln!("connecting to MCP server (`{cmd}`)...");
        Some(yesman_model::McpClient::spawn(cmd)?)
    } else {
        None
    };

    let system_prompt = if let Some(client) = &mut mcp_client {
        let all_tools = client.list_tools()?;
        eprintln!(
            "loaded {} tools from MCP server: {}",
            all_tools.len(),
            all_tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        // Show the model only the compact schemas it was fine-tuned on. All
        // live tools remain callable by name; this keeps the prompt short and
        // identical to training (the full schemas cost ~5k tokens of prefill).
        let tools: Vec<yesman_model::McpTool> = if args.blender {
            eprintln!("using 4 compact Blender tool schemas for the system prompt");
            yesman_model::blender_training_tools()
        } else {
            all_tools
        };
        // The workflow rules live in one place (training/system_prompt.txt),
        // the same text the model was fine-tuned on. Tool schemas come from
        // the live MCP server, so training and inference stay in sync.
        let default_rules = std::fs::read_to_string("training/system_prompt.txt")
            .unwrap_or_else(|_| {
                "Ты — профессиональный 3D-моделлер Blender через mcp-for-blender.".to_string()
            })
            .trim_end()
            .to_string();
        let base_sys = args.system.clone().unwrap_or(default_rules);
        yesman_model::format_tools_system_prompt(&base_sys, &tools)
    } else {
        args.system
            .clone()
            .unwrap_or_else(|| chat::DEFAULT_SYSTEM.to_string())
    };

    if let Some(p) = &args.prompt {
        let mut messages = if args.raw {
            vec![chat::user(p.clone())]
        } else {
            vec![chat::system(&system_prompt), chat::user(p.clone())]
        };
        handle_turn(
            &mut engine,
            &tok,
            &cfg,
            &mut messages,
            &opts,
            &mut sampler,
            &mut mcp_client,
        )?;
        return Ok(());
    }

    // --- interactive REPL -----------------------------------------------------
    let mut history: Vec<chat::Message> = vec![chat::system(&system_prompt)];
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
            history = vec![chat::system(&system_prompt)];
            engine.reset();
            println!("История диалога и KV-кэш сброшены.\n");
            continue;
        }
        history.push(chat::user(line.to_string()));
        handle_turn(
            &mut engine,
            &tok,
            &cfg,
            &mut history,
            &opts,
            &mut sampler,
            &mut mcp_client,
        )?;
        println!();
    }
    Ok(())
}

fn handle_turn(
    engine: &mut Engine,
    tok: &Tokenizer,
    cfg: &Config,
    history: &mut Vec<chat::Message>,
    opts: &GenerateOptions,
    sampler: &mut Sampler,
    mcp: &mut Option<yesman_model::McpClient>,
) -> Result<String> {
    // plan -> build -> verify (look) -> summary, with headroom for one retry.
    const MAX_STEPS: usize = 6;
    let mut step = 0;
    let mut consecutive_failures = 0usize;
    #[allow(unused_assignments)]
    let mut last_reply = String::new();

    loop {
        let reply = print_reply(engine, tok, cfg, history, opts, sampler)?;
        history.push(chat::assistant(&reply));
        last_reply = reply.clone();

        if let Some(client) = mcp {
            let (_, calls) = yesman_model::parse_tool_calls(&reply);
            if !calls.is_empty() && step < MAX_STEPS {
                step += 1;

                for call in calls {
                    // Skip invalid empty tool calls
                    if call.name == "execute_blender_code"
                        && call
                            .arguments
                            .get("code")
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim().is_empty())
                            .unwrap_or(true)
                    {
                        continue;
                    }

                    println!("\n🛠️  [Вызов инструмента MCP: {}]", call.name);
                    println!(
                        "    Аргументы: {}",
                        serde_json::to_string(&call.arguments).unwrap_or_default()
                    );
                    let result = match client.call_tool(&call.name, call.arguments) {
                        Ok(res) => res,
                        Err(e) => format!("Error executing tool {}: {e}", call.name),
                    };
                    println!("📦 [Результат]: {}\n", result.trim());

                    if result.contains("Error") || result.contains("validation error") {
                        consecutive_failures += 1;
                    } else {
                        consecutive_failures = 0;
                    }

                    let tool_resp = yesman_model::format_tool_response(&result);
                    history.push(chat::user(tool_resp));
                }

                // Two failed calls in a row mean the model is stuck in a retry
                // loop (e.g. bad argument shapes). Stop instead of burning
                // minutes of prefill on the same mistake.
                if consecutive_failures >= 2 {
                    println!("⚠️  [Инструмент дважды вернул ошибку — останавливаю цикл]");
                    break;
                }

                // Do not stop after a successful build: the model is trained to
                // verify with `look` and then write a final summary. The loop
                // ends when a reply contains no tool calls.
                print!("💬 ");
                continue;
            }
        }
        break;
    }

    Ok(last_reply)
}

enum Engine {
    Cpu(Qwen2),
    Gpu(Box<CudaQwen2>, Qwen2),
}

impl Engine {
    fn reset(&mut self) {
        match self {
            Self::Cpu(m) => m.reset(),
            Self::Gpu(g, _) => g.reset(),
        }
    }

    fn chat_once(
        &mut self,
        tok: &Tokenizer,
        cfg: &Config,
        messages: &[chat::Message],
        opts: &GenerateOptions,
        sampler: &mut Sampler,
        on_token: impl FnMut(&str) -> Result<()>,
    ) -> Result<String> {
        match self {
            Self::Cpu(m) => chat_once(m, tok, cfg, messages, opts, sampler, on_token),
            Self::Gpu(g, cpu) => g.chat_once(tok, cpu, cfg, messages, opts, sampler, on_token),
        }
    }
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
    engine: &mut Engine,
    tok: &Tokenizer,
    cfg: &Config,
    messages: &[chat::Message],
    opts: &GenerateOptions,
    sampler: &mut Sampler,
) -> Result<String> {
    let t0 = Instant::now();
    let mut tokens = 0usize;
    let mut first_token_at = t0;
    let text = engine.chat_once(tok, cfg, messages, opts, sampler, |piece| {
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
