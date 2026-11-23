//! OpenAI-compatible HTTP server for the pure-Rust Qwen2 inference engine.
//!
//!   POST /v1/chat/completions   (JSON & streaming SSE)
//!   POST /v1/completions        (legacy text completions, JSON & streaming SSE)
//!   GET  /v1/models             (model listing)
//!   GET  /health                (health check)
//!   GET  /metrics               (Prometheus metrics)
//!   GET  /stats                 (JSON runtime stats)

use std::{
    convert::Infallible,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
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
use yesman_cuda::{CudaDevice, CudaQwen2, CudaQwen2Weights, is_cuda_available};
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
    /// Execution backend: auto | cpu | cuda
    #[arg(long, default_value = "auto")]
    device: String,
    /// Context window size per session
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Maximum concurrent model sessions running in parallel (default: 4)
    #[arg(long, default_value_t = 4)]
    max_concurrency: usize,
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

enum ServerSession {
    Cpu(Qwen2),
    Gpu(Box<CudaQwen2>, Qwen2),
}

impl ServerSession {
    fn generate(
        &mut self,
        tok: &Tokenizer,
        prompt_ids: &[u32],
        opts: &GenerateOptions,
        sampler: &mut Sampler,
        on_token: impl FnMut(&str) -> Result<()>,
    ) -> Result<(Vec<u32>, String)> {
        match self {
            Self::Cpu(m) => generate(m, tok, prompt_ids, opts, sampler, on_token),
            Self::Gpu(g, cpu) => g.generate(tok, cpu, prompt_ids, opts, sampler, on_token),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Cpu(m) => m.reset(),
            Self::Gpu(g, _) => g.reset(),
        }
    }
}

enum PoolBackend {
    Cpu,
    Gpu(Arc<CudaQwen2Weights>),
}

/// Pool of reusable model sessions sharing immutable model weights via `Arc`.
struct SessionPool {
    root: Qwen2,
    ctx: usize,
    backend: PoolBackend,
    idle: Mutex<Vec<ServerSession>>,
    sem: Arc<tokio::sync::Semaphore>,
    max_sessions: usize,
}

impl SessionPool {
    fn new(root: Qwen2, ctx: usize, max_sessions: usize, backend: PoolBackend) -> Self {
        let max_sessions = max_sessions.max(1);
        Self {
            root,
            ctx,
            backend,
            idle: Mutex::new(Vec::new()),
            sem: Arc::new(tokio::sync::Semaphore::new(max_sessions)),
            max_sessions,
        }
    }

    async fn acquire(&self) -> Result<(ServerSession, tokio::sync::OwnedSemaphorePermit)> {
        let permit = Arc::clone(&self.sem)
            .acquire_owned()
            .await
            .map_err(|e| anyhow!("semaphore closed: {e}"))?;
        let session = {
            let mut idle = self.idle.lock().unwrap();
            if let Some(s) = idle.pop() {
                s
            } else {
                match &self.backend {
                    PoolBackend::Cpu => ServerSession::Cpu(self.root.clone_session(self.ctx)),
                    PoolBackend::Gpu(weights) => {
                        let gpu = weights.create_session(self.ctx)?;
                        ServerSession::Gpu(Box::new(gpu), self.root.clone_session(self.ctx))
                    }
                }
            }
        };
        Ok((session, permit))
    }

    fn release(&self, mut session: ServerSession, permit: tokio::sync::OwnedSemaphorePermit) {
        session.reset();
        {
            let mut idle = self.idle.lock().unwrap();
            idle.push(session);
        }
        drop(permit);
    }
}

/// Runtime metrics and telemetry counters.
struct ServerMetrics {
    start_time: Instant,
    chat_requests: AtomicU64,
    text_requests: AtomicU64,
    active_requests: AtomicU64,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    total_decode_time_ms: AtomicU64,
}

impl ServerMetrics {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            chat_requests: AtomicU64::new(0),
            text_requests: AtomicU64::new(0),
            active_requests: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
            total_decode_time_ms: AtomicU64::new(0),
        }
    }

    fn record_completion(
        &self,
        prompt_len: usize,
        comp_len: usize,
        duration_ms: u64,
        is_chat: bool,
    ) {
        if is_chat {
            self.chat_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.text_requests.fetch_add(1, Ordering::Relaxed);
        }
        self.prompt_tokens
            .fetch_add(prompt_len as u64, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(comp_len as u64, Ordering::Relaxed);
        self.total_decode_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }
}

struct Inner {
    pool: SessionPool,
    tok: Tokenizer,
    cfg: Config,
    generation: GenerationConfig,
    model_id: String,
    dtype_name: String,
    backend_name: String,
    ctx: usize,
    seq: AtomicU64,
    metrics: ServerMetrics,
}

#[derive(Clone)]
struct AppState(Arc<Inner>);

impl AppState {
    fn chat_id(&self) -> String {
        let n = self.0.seq.fetch_add(1, Ordering::Relaxed);
        format!("chatcmpl-{n:x}")
    }

    fn completion_id(&self) -> String {
        let n = self.0.seq.fetch_add(1, Ordering::Relaxed);
        format!("cmpl-{n:x}")
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

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: Value,
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
    #[serde(default)]
    stop: Option<Value>,
}

impl CompletionRequest {
    fn prompt_str(&self) -> String {
        match &self.prompt {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(""),
            other => other.to_string(),
        }
    }
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
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn build_messages(msgs: &[Message]) -> Vec<chat::Message> {
    let mut out: Vec<chat::Message> = msgs
        .iter()
        .map(|m| chat::Message {
            role: m.role.clone(),
            content: m.text(),
        })
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
    is_chat: bool,
}

impl AppState {
    fn plan_chat(&self, req: &ChatRequest) -> Result<RequestPlan> {
        if req.messages.is_empty() {
            return Err(anyhow!("messages must not be empty"));
        }
        let messages = build_messages(&req.messages);
        let prompt = chat::render(&messages);
        let prompt_ids = self.0.tok.encode(&prompt)?;
        let mut stop = self.0.cfg.eos_ids();
        if let Some(id) = self.0.tok.id_of(chat::IM_END)
            && !stop.contains(&id)
        {
            stop.push(id);
        }
        if let Some(id) = self.0.tok.id_of("<|endoftext|>")
            && !stop.contains(&id)
        {
            stop.push(id);
        }
        let max_ctx = self.0.ctx;
        let opts = GenerateOptions {
            max_new_tokens: req
                .max_tokens
                .unwrap_or(512)
                .min(max_ctx.saturating_sub(prompt_ids.len())),
            stop_ids: stop,
            max_ctx,
        };

        let mut s = Sampler::new(req.seed.unwrap_or_else(rand_u64));
        s.temperature = req
            .temperature
            .or(self.0.generation.temperature)
            .unwrap_or(0.7);
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
            id: self.chat_id(),
            model_id: req.model.clone().unwrap_or_else(|| self.0.model_id.clone()),
            is_chat: true,
        })
    }

    fn plan_completion(&self, req: &CompletionRequest) -> Result<RequestPlan> {
        let prompt = req.prompt_str();
        if prompt.is_empty() {
            return Err(anyhow!("prompt must not be empty"));
        }
        let prompt_ids = self.0.tok.encode(&prompt)?;
        let mut stop = self.0.cfg.eos_ids();
        if let Some(id) = self.0.tok.id_of(chat::IM_END)
            && !stop.contains(&id)
        {
            stop.push(id);
        }
        if let Some(id) = self.0.tok.id_of("<|endoftext|>")
            && !stop.contains(&id)
        {
            stop.push(id);
        }

        if let Some(stop_val) = &req.stop {
            match stop_val {
                Value::String(s) => {
                    if let Some(id) = self.0.tok.id_of(s)
                        && !stop.contains(&id)
                    {
                        stop.push(id);
                    }
                }
                Value::Array(arr) => {
                    for v in arr {
                        if let Some(s) = v.as_str()
                            && let Some(id) = self.0.tok.id_of(s)
                            && !stop.contains(&id)
                        {
                            stop.push(id);
                        }
                    }
                }
                _ => {}
            }
        }

        let max_ctx = self.0.ctx;
        let opts = GenerateOptions {
            max_new_tokens: req
                .max_tokens
                .unwrap_or(512)
                .min(max_ctx.saturating_sub(prompt_ids.len())),
            stop_ids: stop,
            max_ctx,
        };

        let mut s = Sampler::new(req.seed.unwrap_or_else(rand_u64));
        s.temperature = req
            .temperature
            .or(self.0.generation.temperature)
            .unwrap_or(0.7);
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
            id: self.completion_id(),
            model_id: req.model.clone().unwrap_or_else(|| self.0.model_id.clone()),
            is_chat: false,
        })
    }
}

fn rand_u64() -> u64 {
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
    let t0 = Instant::now();
    let root_model = match &args.cache {
        Some(p) => Qwen2::from_cache(&args.model, args.ctx, p)?,
        None => Qwen2::from_dir(&args.model, args.ctx, dtype)?,
    };
    let dtype_name = if args.cache.is_some() {
        "cached".to_string()
    } else {
        dtype.name().to_string()
    };
    let tok = Tokenizer::from_dir(&args.model)?;
    eprintln!(
        "loaded {} layers, {} weights in {:.2}s",
        cfg.num_hidden_layers,
        dtype_name,
        t0.elapsed().as_secs_f32()
    );

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

    let (backend, backend_name) = if use_cuda {
        let dev = CudaDevice::new(0)?;
        let ctx = dev.create_context()?;
        eprintln!(
            "using GPU backend: {} ({:.2} GB VRAM)",
            dev.name,
            dev.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        let weights = CudaQwen2Weights::from_cpu(&root_model, ctx)?;
        (PoolBackend::Gpu(weights), format!("cuda ({})", dev.name))
    } else {
        eprintln!("using CPU backend (AVX2/FMA)");
        (PoolBackend::Cpu, "cpu".to_string())
    };

    let pool = SessionPool::new(root_model, args.ctx, args.max_concurrency, backend);
    eprintln!(
        "initialized session pool: max_concurrency={}, ctx={}",
        args.max_concurrency, args.ctx
    );

    let state = AppState(Arc::new(Inner {
        pool,
        tok,
        cfg,
        generation,
        model_id: args.model_id,
        dtype_name,
        backend_name,
        ctx: args.ctx,
        seq: AtomicU64::new(0),
        metrics: ServerMetrics::new(),
    }));

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(text_completions))
        .route("/metrics", get(metrics_handler))
        .route("/stats", get(stats_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("yesman-server listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    println!("yesman-server stopped cleanly.");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    eprintln!("\nreceived shutdown signal, draining active requests...");
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

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let m = &state.0.metrics;
    let chat_req = m.chat_requests.load(Ordering::Relaxed);
    let text_req = m.text_requests.load(Ordering::Relaxed);
    let active = m.active_requests.load(Ordering::Relaxed);
    let prompt_tok = m.prompt_tokens.load(Ordering::Relaxed);
    let comp_tok = m.completion_tokens.load(Ordering::Relaxed);
    let decode_ms = m.total_decode_time_ms.load(Ordering::Relaxed);
    let tok_per_sec = if decode_ms > 0 {
        (comp_tok as f64) / (decode_ms as f64 / 1000.0)
    } else {
        0.0
    };
    let uptime = m.start_time.elapsed().as_secs();

    let out = format!(
        "# HELP yesman_uptime_seconds Server uptime in seconds.\n\
         # TYPE yesman_uptime_seconds gauge\n\
         yesman_uptime_seconds {uptime}\n\
         # HELP yesman_requests_total Total number of completion requests.\n\
         # TYPE yesman_requests_total counter\n\
         yesman_requests_total{{type=\"chat\"}} {chat_req}\n\
         yesman_requests_total{{type=\"text\"}} {text_req}\n\
         # HELP yesman_active_requests Currently executing requests.\n\
         # TYPE yesman_active_requests gauge\n\
         yesman_active_requests {active}\n\
         # HELP yesman_tokens_prompt_total Total prompt tokens processed.\n\
         # TYPE yesman_tokens_prompt_total counter\n\
         yesman_tokens_prompt_total {prompt_tok}\n\
         # HELP yesman_tokens_generated_total Total tokens generated.\n\
         # TYPE yesman_tokens_generated_total counter\n\
         yesman_tokens_generated_total {comp_tok}\n\
         # HELP yesman_decode_tokens_per_second Mean generation speed across all requests.\n\
         # TYPE yesman_decode_tokens_per_second gauge\n\
         yesman_decode_tokens_per_second {tok_per_sec:.2}\n"
    );
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        out,
    )
}

async fn stats_handler(State(state): State<AppState>) -> Json<Value> {
    let m = &state.0.metrics;
    let chat_req = m.chat_requests.load(Ordering::Relaxed);
    let text_req = m.text_requests.load(Ordering::Relaxed);
    let active = m.active_requests.load(Ordering::Relaxed);
    let prompt_tok = m.prompt_tokens.load(Ordering::Relaxed);
    let comp_tok = m.completion_tokens.load(Ordering::Relaxed);
    let decode_ms = m.total_decode_time_ms.load(Ordering::Relaxed);
    let tok_per_sec = if decode_ms > 0 {
        (comp_tok as f64) / (decode_ms as f64 / 1000.0)
    } else {
        0.0
    };
    Json(json!({
        "uptime_secs": m.start_time.elapsed().as_secs(),
        "model_id": state.0.model_id,
        "dtype": state.0.dtype_name,
        "backend": state.0.backend_name,
        "ctx": state.0.ctx,
        "max_concurrency": state.0.pool.max_sessions,
        "active_requests": active,
        "total_requests": chat_req + text_req,
        "chat_requests": chat_req,
        "text_requests": text_req,
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "avg_decode_tok_per_sec": (tok_per_sec * 100.0).round() / 100.0,
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
    let plan = match state.plan_chat(&req) {
        Ok(p) => p,
        Err(e) => return Err(error_response(e)),
    };
    if !req.stream {
        Ok(non_stream_response(state, plan).await.into_response())
    } else {
        Ok(stream_response(state, plan).await.into_response())
    }
}

async fn text_completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let plan = match state.plan_completion(&req) {
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
    let RequestPlan {
        prompt_ids,
        opts,
        sampler,
        id,
        model_id,
        is_chat,
    } = plan;
    let prompt_len = prompt_ids.len();

    let (session, permit) = match state.0.pool.acquire().await {
        Ok(pair) => pair,
        Err(e) => return error_response(e).into_response(),
    };
    state
        .0
        .metrics
        .active_requests
        .fetch_add(1, Ordering::Relaxed);
    let worker_state = state.clone();

    let result = tokio::task::spawn_blocking(move || {
        let t_start = Instant::now();
        let mut session = session;
        let mut sampler = sampler;
        let res: Result<(Vec<u32>, String)> = session.generate(
            &worker_state.0.tok,
            &prompt_ids,
            &opts,
            &mut sampler,
            |_| Ok(()),
        );
        let duration_ms = t_start.elapsed().as_millis() as u64;
        worker_state.0.pool.release(session, permit);
        res.map(|(ids, text)| (ids.len(), text, duration_ms))
    })
    .await;

    state
        .0
        .metrics
        .active_requests
        .fetch_sub(1, Ordering::Relaxed);

    match result {
        Ok(Ok((completion_tokens, text, duration_ms))) => {
            state
                .0
                .metrics
                .record_completion(prompt_len, completion_tokens, duration_ms, is_chat);
            if is_chat {
                Json(json!({
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
                .into_response()
            } else {
                Json(json!({
                    "id": id,
                    "object": "text_completion",
                    "created": now_secs(),
                    "model": model_id,
                    "choices": [{
                        "text": text,
                        "index": 0,
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_len + completion_tokens,
                    }
                }))
                .into_response()
            }
        }
        Ok(Err(e)) => error_response(e).into_response(),
        Err(e) => error_response(anyhow!("worker panicked: {e}")).into_response(),
    }
}

async fn stream_response(state: AppState, plan: RequestPlan) -> impl IntoResponse {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
    let id = plan.id.clone();
    let model_id = plan.model_id.clone();
    let is_chat = plan.is_chat;
    let prompt_len = plan.prompt_ids.len();
    let created = now_secs();

    let (session, permit) = match state.0.pool.acquire().await {
        Ok(pair) => pair,
        Err(e) => return error_response(e).into_response(),
    };
    state
        .0
        .metrics
        .active_requests
        .fetch_add(1, Ordering::Relaxed);
    let worker_state = state.clone();

    tokio::task::spawn_blocking(move || {
        let t_start = Instant::now();
        let mut session = session;
        let mut sampler = plan.sampler;
        let mut count = 0usize;
        let res: Result<(Vec<u32>, String)> = session.generate(
            &worker_state.0.tok,
            &plan.prompt_ids,
            &plan.opts,
            &mut sampler,
            |piece| {
                count += 1;
                let _ = tx.blocking_send(piece.to_string());
                Ok(())
            },
        );
        let duration_ms = t_start.elapsed().as_millis() as u64;
        worker_state
            .0
            .metrics
            .record_completion(prompt_len, count, duration_ms, is_chat);
        worker_state
            .0
            .metrics
            .active_requests
            .fetch_sub(1, Ordering::Relaxed);
        worker_state.0.pool.release(session, permit);

        if let Err(e) = res {
            let _ = tx.blocking_send(format!("\u{0}ERROR:{e}"));
        }
    });

    let stream = async_stream::stream! {
        let mut token_count = 0usize;
        if is_chat {
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
        }
        while let Some(piece) = rx.recv().await {
            if let Some(msg) = piece.strip_prefix("\u{0}ERROR:") {
                yield Ok::<Event, Infallible>(Event::default().data(
                    json!({"error": {"message": msg}}).to_string()
                ));
                break;
            }
            token_count += 1;
            if is_chat {
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
            } else {
                yield Ok::<Event, Infallible>(
                    Event::default().data(
                        json!({
                            "id": id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_id,
                            "choices": [{"text": piece, "index": 0, "finish_reason": Value::Null}],
                        })
                        .to_string()
                    )
                );
            }
        }

        if is_chat {
            yield Ok::<Event, Infallible>(
                Event::default().data(
                    json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": prompt_len,
                            "completion_tokens": token_count,
                            "total_tokens": prompt_len + token_count,
                        }
                    })
                    .to_string()
                )
            );
        } else {
            yield Ok::<Event, Infallible>(
                Event::default().data(
                    json!({
                        "id": id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_id,
                        "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": prompt_len,
                            "completion_tokens": token_count,
                            "total_tokens": prompt_len + token_count,
                        }
                    })
                    .to_string()
                )
            );
        }
        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_text_parsing() {
        let msg_str = Message {
            role: "user".into(),
            content: Value::String("hello world".into()),
        };
        assert_eq!(msg_str.text(), "hello world");

        let msg_arr = Message {
            role: "user".into(),
            content: json!([
                {"type": "text", "text": "part 1"},
                {"type": "text", "text": "part 2"}
            ]),
        };
        assert_eq!(msg_arr.text(), "part 1\npart 2");
    }

    #[test]
    fn build_messages_injects_system() {
        let msgs = vec![Message {
            role: "user".into(),
            content: Value::String("hi".into()),
        }];
        let built = build_messages(&msgs);
        assert_eq!(built.len(), 2);
        assert_eq!(built[0].role, "system");
        assert_eq!(built[1].role, "user");
        assert_eq!(built[1].content, "hi");

        // When system is already present, do not duplicate
        let msgs_with_sys = vec![
            Message {
                role: "system".into(),
                content: Value::String("custom system".into()),
            },
            Message {
                role: "user".into(),
                content: Value::String("hi".into()),
            },
        ];
        let built2 = build_messages(&msgs_with_sys);
        assert_eq!(built2.len(), 2);
        assert_eq!(built2[0].content, "custom system");
    }

    #[test]
    fn completion_request_prompt_string() {
        let req_str = CompletionRequest {
            model: None,
            prompt: Value::String("once upon a time".into()),
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            repetition_penalty: None,
            seed: None,
            stream: false,
            stop: None,
        };
        assert_eq!(req_str.prompt_str(), "once upon a time");

        let req_arr = CompletionRequest {
            model: None,
            prompt: json!(["once ", "upon ", "a time"]),
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            repetition_penalty: None,
            seed: None,
            stream: false,
            stop: None,
        };
        assert_eq!(req_arr.prompt_str(), "once upon a time");
    }
}
