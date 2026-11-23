#!/usr/bin/env python3
"""Benchmark concurrent client requests against yesman-server.

Usage:
    python3 tools/bench_server.py [--url URL] [--clients N] [--tokens N]
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def request_chat(url: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.time()
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    choice = data["choices"][0]
    usage = data["usage"]
    return {
        "text": choice["message"]["content"].strip(),
        "tokens": usage["completion_tokens"],
        "elapsed": elapsed,
        "tok_s": usage["completion_tokens"] / max(elapsed, 1e-4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()

    prompts = [
        "What is the capital of France? Answer in one word.",
        "What is 2 + 2? Answer in one word.",
        "What is the capital of Japan? Answer in one word.",
        "What is the capital of Germany? Answer in one word.",
    ]

    print(f"=== Testing single client baseline (1 client, {args.tokens} tokens) ===")
    res1 = request_chat(args.url, prompts[0], args.tokens)
    print(f"  1 client: {res1['tokens']} tokens in {res1['elapsed']:.2f}s ({res1['tok_s']:.1f} tok/s) -> \"{res1['text']}\"")

    n_clients = args.clients
    active_prompts = [prompts[i % len(prompts)] for i in range(n_clients)]
    print(f"\n=== Testing {n_clients} concurrent clients ({args.tokens} tokens each) ===")
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=n_clients) as pool:
        futures = [pool.submit(request_chat, args.url, p, args.tokens) for p in active_prompts]
        results = [f.result() for f in futures]
    t_total = time.time() - t_start

    total_tokens = sum(r["tokens"] for r in results)
    agg_tok_s = total_tokens / max(t_total, 1e-4)

    for i, r in enumerate(results):
        print(f"  Client {i+1}: {r['tokens']} tokens in {r['elapsed']:.2f}s ({r['tok_s']:.1f} tok/s) -> \"{r['text']}\"")

    print(f"\nSummary:")
    print(f"  Wall time: {t_total:.2f}s for {n_clients} parallel requests")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Aggregate throughput: {agg_tok_s:.1f} tok/s (vs single client {res1['tok_s']:.1f} tok/s)")
    assert len(results) == n_clients, "some client failed"
    print("All concurrent client checks PASSED!")


if __name__ == "__main__":
    main()
