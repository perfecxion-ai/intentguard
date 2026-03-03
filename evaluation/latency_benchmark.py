"""Latency benchmark for candidate models.

Measures end-to-end inference time (tokenization + model forward pass)
for each candidate model at various sequence lengths. Runs on CPU to
match production deployment.

Usage:
    python -m evaluation.latency_benchmark [--models all] [--rounds 100]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Candidates to benchmark
CANDIDATES = {
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-xsmall": "microsoft/deberta-v3-xsmall",
    "minilm-l6": "microsoft/MiniLM-L6-H384-uncased",
}

# Test inputs at different lengths
TEST_INPUTS = {
    "short": "What are mortgage rates?",
    "medium": (
        "I'm looking to refinance my home mortgage and wondering what "
        "the current interest rates look like for a 30-year fixed loan "
        "in the state of California."
    ),
    "long": (
        "I have a complex financial situation where I need to balance "
        "my 401k contributions with my Roth IRA, while also managing "
        "student loan payments and saving for a down payment on a house. "
        "My employer offers a 50% match up to 6% of my salary. I also "
        "have some stock options that are about to vest. What would be "
        "the most tax-efficient approach to handle all of these "
        "financial instruments simultaneously? Should I consider "
        "consulting with a certified financial planner?"
    ),
}


@dataclass
class BenchmarkResult:
    model_name: str
    model_id: str
    input_label: str
    token_count: int
    rounds: int
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies_ms)

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95))

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99))

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms)

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms)

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "input": self.input_label,
            "tokens": self.token_count,
            "rounds": self.rounds,
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
        }


def benchmark_transformers(model_id: str, model_name: str, rounds: int) -> list[BenchmarkResult]:
    """Benchmark a model using HuggingFace transformers (PyTorch, CPU)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    print(f"\n  Loading {model_name} ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=3
    )
    model.eval()

    results = []
    for label, text in TEST_INPUTS.items():
        # Warm up
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            model(**inputs)

        # Benchmark
        latencies = []
        for _ in range(rounds):
            start = time.perf_counter()
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                model(**inputs)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        token_count = inputs["input_ids"].shape[1]
        result = BenchmarkResult(
            model_name=model_name,
            model_id=model_id,
            input_label=label,
            token_count=token_count,
            rounds=rounds,
            latencies_ms=latencies,
        )
        results.append(result)
        s = result.summary()
        print(f"    {label} ({s['tokens']} tokens): "
              f"mean={s['mean_ms']}ms p95={s['p95_ms']}ms p99={s['p99_ms']}ms")

    return results


def benchmark_onnx(model_id: str, model_name: str, rounds: int) -> list[BenchmarkResult]:
    """Benchmark a model exported to ONNX (CPU)."""
    import onnxruntime as ort
    from transformers import AutoTokenizer
    from optimum.onnxruntime import ORTModelForSequenceClassification

    print(f"\n  Loading {model_name} ONNX ({model_id})...")

    try:
        # Use optimum to auto-export to ONNX
        model = ORTModelForSequenceClassification.from_pretrained(
            model_id, export=True, provider="CPUExecutionProvider"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"    ONNX export failed for {model_name}: {e}")
        return []

    # Get the ONNX session for direct benchmarking
    session = model.model

    # Configure threads
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1

    results = []
    for label, text in TEST_INPUTS.items():
        # Tokenize once to get input names
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=256)
        # DeBERTa-v3 doesn't use token_type_ids
        inputs.pop("token_type_ids", None)

        # Warm up
        feed = {k: v for k, v in inputs.items() if k in [i.name for i in session.get_inputs()]}
        session.run(None, feed)

        # Benchmark full pipeline: tokenize + inference
        latencies = []
        for _ in range(rounds):
            start = time.perf_counter()
            tok = tokenizer(text, return_tensors="np", truncation=True, max_length=256)
            tok.pop("token_type_ids", None)
            feed = {k: v for k, v in tok.items() if k in [i.name for i in session.get_inputs()]}
            session.run(None, feed)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        token_count = inputs["input_ids"].shape[1]
        result = BenchmarkResult(
            model_name=f"{model_name}-onnx",
            model_id=model_id,
            input_label=label,
            token_count=token_count,
            rounds=rounds,
            latencies_ms=latencies,
        )
        results.append(result)
        s = result.summary()
        print(f"    {label} ({s['tokens']} tokens): "
              f"mean={s['mean_ms']}ms p95={s['p95_ms']}ms p99={s['p99_ms']}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark candidate models for IntentGuard")
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    parser.add_argument("--rounds", type=int, default=100, help="Number of inference rounds")
    parser.add_argument("--onnx", action="store_true", help="Also benchmark ONNX versions")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if args.models == "all":
        models = CANDIDATES
    else:
        models = {k: v for k, v in CANDIDATES.items() if k in args.models.split(",")}

    if not models:
        print("No valid models selected. Available:", list(CANDIDATES.keys()))
        sys.exit(1)

    print(f"Benchmarking {len(models)} model(s), {args.rounds} rounds each")
    print(f"Test inputs: {', '.join(f'{k} ({len(v)} chars)' for k, v in TEST_INPUTS.items())}")

    all_results = []

    for name, model_id in models.items():
        # PyTorch baseline
        try:
            results = benchmark_transformers(model_id, name, args.rounds)
            all_results.extend(results)
        except Exception as e:
            print(f"  PyTorch benchmark failed for {name}: {e}")

        # ONNX if requested
        if args.onnx:
            try:
                results = benchmark_onnx(model_id, name, args.rounds)
                all_results.extend(results)
            except Exception as e:
                print(f"  ONNX benchmark failed for {name}: {e}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Model':<25} {'Input':<10} {'Tokens':<8} {'Mean':<10} {'P95':<10} {'P99':<10}")
    print("-" * 80)
    for r in all_results:
        s = r.summary()
        print(f"{s['model']:<25} {s['input']:<10} {s['tokens']:<8} "
              f"{s['mean_ms']:<10} {s['p95_ms']:<10} {s['p99_ms']:<10}")
    print("=" * 80)

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(
            [r.summary() for r in all_results], indent=2
        ))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
