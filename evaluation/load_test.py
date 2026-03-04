"""Load testing for IntentGuard server.

Sends concurrent requests to measure throughput and latency
under load. Requires the server to be running.

Usage:
    # Start server first:
    uvicorn intentguard.server:app --port 8080

    # Then run load test:
    python -m evaluation.load_test --url http://localhost:8080 --concurrent 50 --total 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time

import httpx

logger = logging.getLogger(__name__)

SAMPLE_QUERIES = [
    "What are current mortgage rates?",
    "How do I invest in index funds?",
    "What's the difference between a Roth IRA and traditional IRA?",
    "Who won the Super Bowl last year?",
    "Best recipe for chocolate cake",
    "How do I improve my credit score?",
    "help",
    "What are HSA contribution limits?",
    "How does compound interest work?",
    "Tell me about the latest movies",
    "What tax deductions can I claim?",
    "For a finance presentation, explain basketball scoring",
    "Should I refinance my mortgage?",
    "Best gaming headset recommendations",
    "How do I build an emergency fund?",
]


async def send_request(
    client: httpx.AsyncClient, url: str, query: str,
) -> dict:
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/v1/classify",
            json={"messages": [{"role": "user", "content": query}]},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "status": resp.status_code,
            "latency_ms": elapsed_ms,
            "decision": resp.json().get("decision") if resp.status_code == 200 else None,
            "error": None,
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "status": 0,
            "latency_ms": elapsed_ms,
            "decision": None,
            "error": str(e),
        }


async def run_load_test(url: str, concurrent: int, total: int) -> dict:
    """Run concurrent requests and collect results."""
    semaphore = asyncio.Semaphore(concurrent)
    results = []

    async def bounded_request(client, query):
        async with semaphore:
            return await send_request(client, url, query)

    queries = [random.choice(SAMPLE_QUERIES) for _ in range(total)]

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [bounded_request(client, q) for q in queries]
        results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - start

    # Analyze
    latencies = sorted(r["latency_ms"] for r in results if r["status"] == 200)
    errors = sum(1 for r in results if r["status"] != 200)
    n = len(latencies)

    if n == 0:
        return {"error": "All requests failed", "total": total, "errors": errors}

    return {
        "total": total,
        "concurrent": concurrent,
        "successful": n,
        "errors": errors,
        "wall_time_s": round(wall_time, 2),
        "throughput_qps": round(total / wall_time, 1),
        "latency_p50_ms": round(latencies[n // 2], 1),
        "latency_p95_ms": round(latencies[int(n * 0.95)], 1),
        "latency_p99_ms": round(latencies[int(n * 0.99)], 1),
        "latency_mean_ms": round(sum(latencies) / n, 1),
        "latency_max_ms": round(max(latencies), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Load test IntentGuard server")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--total", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    concurrency_levels = [10, 50, 100]
    if args.concurrent not in concurrency_levels:
        concurrency_levels = [args.concurrent]

    for c in concurrency_levels:
        total = max(args.total, c * 5)
        logger.info("Testing %d concurrent, %d total requests...", c, total)
        results = asyncio.run(run_load_test(args.url, c, total))

        print(f"\n--- {c} concurrent ---")
        if "error" in results and results.get("successful", 0) == 0:
            print(f"  FAILED: {results['error']}")
            continue
        print(f"  Throughput: {results['throughput_qps']} qps")
        print(f"  Latency: p50={results['latency_p50_ms']}ms "
              f"p95={results['latency_p95_ms']}ms "
              f"p99={results['latency_p99_ms']}ms "
              f"max={results['latency_max_ms']}ms")
        print(f"  Errors: {results['errors']}/{results['total']}")
        print(f"  Wall time: {results['wall_time_s']}s")


if __name__ == "__main__":
    main()
