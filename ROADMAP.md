# ROADMAP — развитие yesmanAI

> Статус: **актуально на 24.11.2026**, base commit `770e38e`.
> Обсуждение/предложения по плану — в этом файле; каждая закрытая позиция
> должна сопровождаться коммитом и обновлением чекбокса.

## 0. Где мы сейчас

Работающий CPU-инференс `Qwen2.5-1.5B-Instruct`, написанный с нуля на Rust.
Без torch/candle/burn/BLAS; математика — свои kernels на `std::arch` (AVX2/FMA).
Checked: bf16-логиты совпадают с независимым numpy-эталоном до **3e-5**,
23 теста зелёные, SHA256 весов сверен с HF.

| Слой | Состояние |
| --- | --- |
| `yesman-core` | safetensors (mmap), matmul/RMSNorm/RoPE/softmax/SwiGLU, Q8_0 / Q4_0 / Q4_K, versioned quant cache |
| `yesman-model` | Qwen2 forward (GQA 12/2, qkv-bias, tied lm_head), KV-cache, ChatML, BPE, сэмплер,Workspace без аллокаций |
| `yesman-cli` | CLI-чат, one-shot, `--cache` / `--save-cache`, `--top-logits`, `--print-ids`, `YESMAN_PROFILE=1` |
| `yesman-server` | OpenAI-compatible `/v1/chat/completions` (JSON + SSE), `/v1/models`, `/health`, CORS |
| tools | numpy-эталон, генератор фикстур, параллельный HF-загрузчик, `verify.sh`, microbench/diag |

| Формат | Кэш | Качество | Decode (20 ядер) |
| --- | --- | --- | --- |
| `bf16` | 2.9 ГБ (safetensors) | эталон | ~4 tok/s |
| `q8` | 1.7 ГБ | почти bf16 | ~11 tok/s |
| `q4km` | 1.6 ГБ | drift ≤ 0.6 логита | ~11 tok/s |
| `q4` | 1.1 ГБ | заметно шумнее | ~16 tok/s |
| `q4k` | 993 МБ | Q4_K_S-like | ~16 tok/s |

**Ключевой физический факт:** CPU уперся в пропускную способность памяти
(~20 ГБ/с). `ffn ~65 мс` = 1.15 млрд весов × 1.125 Б, `head ~12 мс` = 262 МБ.
Дальнейший рост на CPU = только меньше байт на вес; порядок скачка = GPU.

### Принципы, которые не меняем

- Чистый Rust в рантайме: без ML-фреймворков, без C-математики, без libpython.
- Любое изменение ядра проверяется против `tools/qwen_ref.py` (numpy) и `cargo test`.
- Точность по умолчанию важнее микросекунд (онлайн-softmax остаётся opt-in).
- Bench-driven: каждое ускорение обязано иметь цифру до/после в `YESMAN_PROFILE`.

---

## P0 — Надёжность и доказательство качества [ВЫПОЛНЕНО]

**Цель:** превратить «работает на коротких примерах» в «доказано работает».

- [x] **Perplexity / logloss по всем форматам**
  - Реализован встроенный режим `yesman --perplexity <file>` и скрипт `tools/eval_perplexity.sh`.
  - Замер на WikiText-2 (2173 токена, pure CPU prefill ~33 tok/s):
    - `bf16`: **6.129** (эталон)
    - `q8`: **6.154** (+0.4%)
    - `q4km`: **6.279** (**+2.4%** vs bf16, значительно лучше целевого порога 5–8%)
    - `q4k`: **6.393** (**+4.3%** vs bf16 при весе кэша 993 МБ)
    - `q4`: **6.566** (+7.1%)
- [x] **Длинный контекст 8k / 16k / 32k** (`config.max_position_embeddings = 32768`)
  - Реализован **chunked prefill** (`PREFILL_CHUNK = 512`): пиковый размер буфера attention scores теперь масштабируется от размера чанка, а не от всей длины последовательности.
  - Устранен риск OOM на длинных контекстах (раньше 32k требовало ~51 ГБ под scores).
  - Успешно протестирована непрерывная оценка 6283 токенов на `--ctx 8192` (PPL: 6.3256, 27.7 tok/s) и инициализация с `--ctx 32768`.
- [x] **Multi-turn и граничные режимы**
  - Поддержка интерактивных команд `/reset` и `/clear` в CLI REPL.
  - Корректная обработка всех stop-ids (`<|im_end|>` 151645 и `<|endoftext|>` 151643).
  - Защита от паники при float-сортировке в сэмплере и CLI (`unwrap_or(Equal)`).
  - Регрессионные тесты в `crates/yesman-model/tests/tiny.rs` на chunked prefill, perplexity, KV-cache length и out-of-bounds guards.
- [x] **CI**
  - Добавлен рабочий процесс GitHub Actions (`.github/workflows/ci.yml`): `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --release --workspace`.
  - 25 тестов проходят, 0 ворнингов clippy.

---

## P1 — Сервер как продукт [ВЫПОЛНЕНО]

**Цель:** несколько клиентов и удобная эксплуатация.

- [x] **Конкурентный пул сессий (`SessionPool`)**
  - Модель разделена на неизменяемые веса `Qwen2Weights` в `Arc` (~1.6 ГБ в RAM) и легковесные сессии `clone_session` с собственным `KvCache` и `Workspace`.
  - Управление параллелизмом через асинхронный семафор (`--max-concurrency`, по умолчанию 4) и пул свободных сессий `idle`.
  - Замер параллельности (`tools/bench_server.py`): 4 одновременных клиента завершают генерацию за 13.01s (128 токенов) с совокупным темпом ~10 tok/s и нулевым взаимным влиянием.
- [x] **`/v1/completions` и `usage` в SSE**
  - Реализован эндпоинт `/v1/completions` (legacy text completions) в обычном и стриминговом (SSE) режимах.
  - Потоковые чанки теперь включают блок `usage: {prompt_tokens, completion_tokens, total_tokens}` в завершающем событии перед `[DONE]`.
- [x] **Метрики и телеметрия**
  - `GET /metrics`: экспортёр в формате Prometheus (`yesman_requests_total`, `yesman_active_requests`, `yesman_tokens_generated_total`, `yesman_decode_tokens_per_second`, `yesman_uptime_seconds`).
  - `GET /stats`: JSON-отчёт о состоянии сервера (активные запросы, общее число токенов, tok/s, параметры пула).
- [x] **Graceful shutdown & развёртывание**
  - Обработка `SIGINT` (Ctrl+C) и `SIGTERM` через Tokio signal с корректным завершением активных задач.
  - Многоэтапный `Dockerfile` (сборка в `rust:bookworm`, запуск в легковесном `debian:bookworm-slim` без root).
  - Шаблон systemd-сервиса `tools/yesman.service` с изоляцией и лимитами.
  - Скрипт бенчмарка параллельных клиентов: `tools/bench_server.py`.

---

## P2 — Скорость [ВЫПОЛНЕНО: P2b CUDA]

### P2b. CUDA-бэкенд [ВЫПОЛНЕНО]

- [x] **Zero-dependency dynamic driver bindings (`libcuda.so.1`)**:
  - Загрузка драйвера на лету через `dlopen` — **не требуется root и не требуется 4 ГБ CUDA Toolkit**.
  - Чистый Rust-крейт `yesman-cuda` с поддержкой динамического создания контекста, аллокаций и запуска ядер.
- [x] **Набор PTX-ядер под Ampere (RTX 3060 sm_86)**:
  - `rmsnorm_kernel`: редукция варпа через hardware shuffle (`shfl.sync.down`), единый цикл без разделяемой памяти.
  - `swiglu_kernel`: аппаратный `ex2.approx.f32` для быстрого вычисления SiLU.
  - `rope_kernel` и `rope_batched_kernel`: поворот Q и K прямо в регистрах GPU без промежуточных буферов.
  - `residual_add_kernel`: векторное сложение на GPU.
  - `gemv_f32_kernel`, `gemv_q8_kernel`, `gemv_q4k_kernel`: параллельное умножение матриц квантованных весов на активации.
  - `gemm_f32_kernel`, `gemm_q8_kernel`, `gemm_q4k_kernel`: 128-битные векторизованные GEMM-ядра для префилла батча токенов.
  - `decode_attention_kernel` и `causal_attention_batched_kernel`: GQA causal attention на GPU со встроенным online-softmax (декодирование и батчевый префилл).
  - Проверено пошагово против CPU: максимальное расхождение логитов **1.4e-6** (одна миллионная!).
- [x] **CudaQwen2 и разделяемые веса**:
  - Веса модели (1.6 ГБ для Q4_K_M) загружаются в VRAM **один раз** в `Arc<CudaQwen2Weights>`.
  - Легковесные сессии `CudaQwen2` аллоцируют только KV-кэш (~234 МБ) и scratchpad (< 1 МБ).
  - Скорость на RTX 3060 Laptop GPU: **~22 tok/s** при 100% стабильности.
  - Интеграция в CLI (`--device auto|cpu|cuda`) и сервер (`--device auto|cpu|cuda`).

### P2a. Меньше байт на вес (эволюция, S–M)

- [ ] **Q6_K / Q5_K для FFN в `q4km`**: сейчас FFN = Q8_0 (1.15 млрд весов).
  6-битный формат дал бы ~1.2 ГБ кэш и ~13–14 tok/s на CPU без потери качества.
  Файлы: `crates/yesman-core/src/q6k.rs` (новый), `Weight`/`DType`/`cache::KIND_*`.
- [ ] **Q8-packing 1.125 → 1.0 байт/вес**: убрать f32-scale из блока (хранение
  scale отдельным массивом/ f16 или вынести в супер-блок), ~10% прироста.
- [ ] AVX2-VNNI (`vpdpbusd`) при наличии CPU-фичи: детектор + fallback.
- [ ] Опционально: huge pages / `RAYON_NUM_THREADS` auto-tune под машину.

---

## P3 — Экосистема и упаковка (opportunistic)

- [x] **Импорт GGUF (v2 / v3)**:
  - Собственный парсер `GgufFile` в `yesman-core` без внешних библиотек (заголовок, метаданные, типы F32/F16/BF16/Q8_0/Q4_0, выравнивание, zero-copy mmap).
  - Поддержка прямой загрузки моделей из `.gguf` файлов (`Qwen2::from_gguf`) в CLI и сервере: `./target/release/yesman --model model.gguf`.
  - Успешная валидация в тестах (`gguf_forward_matches_safetensors`).
- [x] **OpenAI `tools` / function-calling**:
  - Поддержка параметра `tools` в `/v1/chat/completions`.
  - Автоматическое форматирование в ChatML `<tools>` и парсинг `<tool_call>` с возвратом `tool_calls: [...]`.
- [ ] Отдельная команда `yesman convert --dtype q4km --out model.ymq` вместо
  флага `--save-cache` у генерации.
- [ ] OpenAI `/v1/embeddings` (embed_tokens уже есть).
- [ ] Релизные профили под AVX2 / AVX-512, `cargo dist`/binaries, CHANGELOG.
- [ ] Убрать транзитивную C-зависимость `onig_sys` (regex в `tokenizers`),
  перейдя на `fancy-regex` — тогда рантайм станет 100% Rust даже по зависимостям.

---

## P4 — Специализация под Blender MCP (в процессе)

- [x] **Клиент MCP и агентный цикл (`crates/yesman-model/src/mcp.rs`)**:
  - Прямой JSON-RPC 2.0 клиент по stdio для общения с любым MCP-сервером.
  - Автоматическое подключение к Blender MCP (`--blender` в CLI).
  - Чтение и маппинг 22 инструментов (`execute_blender_code`, `get_scene_info`, `get_object_info`, `get_viewport_screenshot`, `set_texture`).
  - Перехват вызовов `<tool_call>`, выполнение в Blender и обратная передача `<tool_response>` в контекст.
- [x] **Пайплайн данных для дообучения (`training/prepare_dataset.py`)**:
  - Извлечение реальных рабочих `bpy`-скриптов из открытых датасетов Hugging Face (`ShiminZ/blender-lowpoly-dioramas`, `EPM-Blender.PromptPairs`).
  - Дополнение синтетическими парами на русском и английском (создание геометрии, шейдеры Principled BSDF, свет, камера, модификаторы, инспекция сцены).
  - Экспорт в `train.jsonl` и `val.jsonl` в формате ChatML с секциями `<tools>` и `<tool_call>`.
- [x] **Скрипт LoRA-обучения под 6 ГБ VRAM (`training/train_lora.py`)**:
  - Оптимизированный QLoRA пайплайн с gradient checkpointing (расход VRAM < 4 ГБ).
  - Слияние LoRA-адаптера с базовой моделью и экспорт в `model.safetensors`.
  - Автоматическая сборка чистого Rust-кэша `.ymq` для инференса на GPU/CPU со скоростью 22 tok/s.
- [ ] Обучение LoRA на собранном корпусе и валидация в живом Blender.

---

## Вне объёма (осознанно)

| Идея | Почему не сейчас |
| --- | --- |
| Обучение / LoRA / файнтюн | Нужен autograd и оптимизатор; `yesman-core::ops` намеренно только forward. Это отдельный крейт/проект. |
| Поддержка других архитектур (Llama, Mistral) | Сначала доказать качество и стабильность на одной. |
| Многопоточная генерация одного запроса (speculative decoding, tensor parallel) | На 1.5B и 20 ядрах выигрыш меньше сложности; после CUDA имеет смысл. |
| WebUI | Покрывается OpenAI-совместимым API + Open WebUI снаружи. |

---

## Матрица решений

| Приоритет | Эффект | Цена | Риск | Зависимости |
| --- | --- | --- | --- | --- |
| **P0** Надёжность | Доверие к цифрам, отсутствие OOM на 32k | M | низкий | нет |
| **P1** Батчинг/сервер | Параллельные клиенты, эксплуатация | M–L | средний | P0 |
| **P2a** Q6_K / packing | +10–25% скорости, меньше ГБ | S–M | низкий | P0 |
| **P2b** CUDA | ×4–8 к скорости | L | высокий | root + toolkit |
| **P3** Экосистема | Удобство/совместимость | S | низкий | — |

## Известные долги (не блокируют, но в плане)

1. `Workspace.scores` = `m · n_heads · t_max` → раздувание на длинном префилле (P0).
2. Attention O(t²) на префилле; online-softmax opt-in даёт drift ~0.15 логита на
   квантованных весах, поэтому дефолт — точный двухпроходный.
3. Один KV-cache на процесс → сервер сериализует запросы (P1).
4. `--ctx` по умолчанию 4096 при заявленных 32768 (P0).
5. `YESMAN_PROFILE` показывает variance 65–110 мс/токен при фоновой нагрузке —
   бенчмарки нужно усреднять по ≥64 токенам.
6. `onig_sys` — единственная транзитивная C-либа (regex токенизатора).

---

## Как вносить изменения в этот план

1. Задача добавляется в свой приоритет с чекбоксом и критерием приёмки.
2. После реализации — `[x]`, коммит, обновление цифр в README (если изменились).
3. Новый «порядок» скорости (CUDA, другая архитектура) выносится в отдельный
   раздел, а не втискивается в текущий этап.
