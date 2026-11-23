//! OpenAI-compatible HTTP server for the pure-Rust Qwen2 inference engine.
//!
//!   POST /v1/chat/completions   (streaming via SSE supported)
//!   GET  /v1/models
//!   GET  /health

use std::{
    convert::Infallible,
    path::PathBuf,
    sync::{Arc, Mutex, atomic::{AtomicU64, Ordering}},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Result, anyhow};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Sse, sse::Event},
    routing::{get, post},
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tower_http::cors::CorsLayer;
use yesman_model::{
    Config, DType, GenerateOptions, GenerationConfig, Qwen2, Sampler, Tokenizer, chat, generate,
};

#[derive(Parser, Debug)]
#[command(name = "yesman-server", about = "OpenAI-compatible API for yesman")]
struct Args {
    #[arg(short, long, default_value = "models/Qwen2.5-1.5B-Instruct")]
    model: PathBuf,
    /// f32 | bf16 | q8 | q4 | q4k (all 4-bit) | q4km (FFN in Q8, best quality)
    #[arg(long, default_value = "bf16")]
    dtype: String,
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Model id reported by /v1/models and in responses
    #[arg(long, default_value = "qwen2.5-1.5b-instruct")]
    model_id: String,
    /// Load pre-quantized weights from this cache (startup drops to ~0.2 s)
    #[arg(long)]
    cache: Option<PathBuf>,
}

struct Inner {
    model: Mutex<Qwen2>,
    tok: Tokenizer,
    cfg: Config,
    generation: GenerationConfig,
    model_id: String,
    ctx: usize,
    seq: AtomicU64,
}

#[derive(Clone)]
struct AppState(Arc<Inner>);

impl AppState {
    fn id(&self) -> String {
        let n = self.0.seq.fetch_add(1, Ordering::Relaxed);
        format!("chatcmpl-{n:x}")
    }
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<Message>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Message {
    role: String,
    content: Value,
}

impl Message {
    fn text(&self) -> String {
        match &self.content {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            other => other.to_string(),
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

fn build_messages(msgs: &[Message]) -> Vec<chat::Message> {
    let mut out: Vec<chat::Message> = msgs
        .iter()
        .map(|m| chat::Message { role: m.role.clone(), content: m.text() })
        .collect();
    if !out.iter().any(|m| m.role == "system") {
        out.insert(0, chat::system(chat::DEFAULT_SYSTEM));
    }
    out
}

struct RequestPlan {
    prompt_ids: Vec<u32>,
    opts: GenerateOptions,
    sampler: Sampler,
    id: String,
    model_id: String,
}

impl AppState {
    fn plan(&self, req: &ChatRequest) -> Result<RequestPlan> {
        if req.messages.is_empty() {
            return Err(anyhow!("messages must not be empty"));
        }
        let messages = build_messages(&req.messages);
        let prompt = chat::render(&messages);
        let prompt_ids = self.0.tok.encode(&prompt)?;
        let mut stop = self.0.cfg.eos_ids();
        if let Some(id) = self.0.tok.id_of(chat::IM_END) {
            stop.push(id);
        }
        let max_ctx = self.0.ctx;
        let opts = GenerateOptions {
            max_new_tokens: req.max_tokens.unwrap_or(512).min(max_ctx.saturating_sub(prompt_ids.len())),
            stop_ids: stop,
            max_ctx,
        };

        let mut s = Sampler::new(req.seed.unwrap_or_else(rand_u64));
        s.temperature = req.temperature.or(self.0.generation.temperature).unwrap_or(0.7);
        s.top_p = req.top_p.or(self.0.generation.top_p).unwrap_or(0.8);
        s.top_k = req.top_k.or(self.0.generation.top_k).unwrap_or(20);
        s.repetition_penalty = req
            .repetition_penalty
            .or(self.0.generation.repetition_penalty)
            .unwrap_or(1.1);

        Ok(RequestPlan {
            prompt_ids,
            opts,
            sampler: s,
            id: self.id(),
            // Echo the model the client asked for; OpenAI clients expect that.
            model_id: req.model.clone().unwrap_or_else(|| self.0.model_id.clone()),
        })
    }
}

fn rand_u64() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = Config::from_dir(&args.model)?;
    let generation = GenerationConfig::from_dir(&args.model).unwrap_or_default();
    let dtype = DType::parse(&args.dtype)?;
    let t0 = std::time::Instant::now();
    let model = match &args.cache {
        Some(p) => Qwen2::from_cache(&args.model, args.ctx, p)?,
        None => Qwen2::from_dir(&args.model, args.ctx, dtype)?,
    };
    let tok = Tokenizer::from_dir(&args.model)?;
    eprintln!(
        "loaded {} layers, {} weights in {:.2}s",
        cfg.num_hidden_layers,
        if args.cache.is_some() { "cached" } else { dtype.name() },
        t0.elapsed().as_secs_f32()
    );

    let state = AppState(Arc::new(Inner {
        model: Mutex::new(model),
        tok,
        cfg,
        generation,
        model_id: args.model_id,
        ctx: args.ctx,
        seq: AtomicU64::new(0),
    }));

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("yesman-server listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> &'static str {
    "ok\n"
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.0.model_id,
            "object": "model",
            "created": now_secs(),
            "owned_by": "yesman",
        }]
    }))
}

fn error_response(e: anyhow::Error) -> impl IntoResponse {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": {"message": e.to_string(), "type": "invalid_request_error"}})),
    )
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let plan = match state.plan(&req) {
        Ok(p) => p,
        Err(e) => return Err(error_response(e)),
    };
    if !req.stream {
        Ok(non_stream_response(state, plan).await.into_response())
    } else {
        Ok(stream_response(state, plan).await.into_response())
    }
}

async fn non_stream_response(state: AppState, plan: RequestPlan) -> impl IntoResponse {
    let RequestPlan { prompt_ids, opts, sampler, id, model_id } = plan;
    let prompt_len = prompt_ids.len();
    let result = tokio::task::spawn_blocking(move || {
        let mut model = state.0.model.lock().unwrap();
        let mut sampler = sampler;
        let res: Result<(Vec<u32>, String)> =
            generate(&mut *model, &state.0.tok, &prompt_ids, &opts, &mut sampler, |_| Ok(()));
        res.map(|(ids, text)| (ids.len(), text))
    })
    .await;

    match result {
        Ok(Ok((completion_tokens, text))) => Json(json!({
            "id": id,
            "object": "chat.completion",
            "created": now_secs(),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_len + completion_tokens,
            }
        }))
        .into_response(),
        Ok(Err(e)) => error_response(e).into_response(),
        Err(e) => error_response(anyhow!("worker panicked: {e}")).into_response(),
    }
}

async fn stream_response(state: AppState, plan: RequestPlan) -> impl IntoResponse {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
    let id = plan.id.clone();
    let model_id = plan.model_id.clone();
    let created = now_secs();

    let worker_state = state.clone();
    tokio::task::spawn_blocking(move || {
        let mut model = worker_state.0.model.lock().unwrap();
        let mut sampler = plan.sampler;
        let res: Result<(Vec<u32>, String)> = generate(
            &mut *model,
            &worker_state.0.tok,
            &plan.prompt_ids,
            &plan.opts,
            &mut sampler,
            |piece| {
                let _ = tx.blocking_send(piece.to_string());
                Ok(())
            },
        );
        if let Err(e) = res {
            let _ = tx.blocking_send(format!("\u{0}ERROR:{e}"));
        }
    });

    let stream = async_stream::stream! {
        yield Ok::<Event, Infallible>(
            Event::default().data(
                json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": Value::Null}],
                })
                .to_string()
            )
        );
        while let Some(piece) = rx.recv().await {
            if let Some(msg) = piece.strip_prefix("\u{0}ERROR:") {
                yield Ok::<Event, Infallible>(Event::default().data(
                    json!({"error": {"message": msg}}).to_string()
                ));
                break;
            }
            yield Ok::<Event, Infallible>(
                Event::default().data(
                    json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": Value::Null}],
                    })
                    .to_string()
                )
            );
        }
        yield Ok::<Event, Infallible>(
            Event::default().data(
                json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                })
                .to_string()
            )
        );
        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
    };

    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default())
}
