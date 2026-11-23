# Multi-stage Dockerfile for yesmanAI
# Pure-Rust inference engine for Qwen2.5

FROM rust:1.85-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN cargo build --release --workspace

# Runtime image
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -U yesman

COPY --from=builder /app/target/release/yesman /usr/local/bin/yesman
COPY --from=builder /app/target/release/yesman-server /usr/local/bin/yesman-server

USER yesman
WORKDIR /home/yesman

EXPOSE 8000

ENV RUST_LOG=info
ENTRYPOINT ["yesman-server", "--host", "0.0.0.0", "--port", "8000"]
