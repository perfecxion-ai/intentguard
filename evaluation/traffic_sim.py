"""Production traffic simulation.

Replays a realistic query distribution against the classifier
and measures accuracy, latency, and throughput.

Usage:
    python -m evaluation.traffic_sim \
        --model dist/finance/model.onnx \
        --tokenizer dist/finance/tokenizer \
        --calibration dist/finance/calibration_params.json \
        --policy policies/finance.json \
        --data evaluation/test_sets/finance_adversarial.reviewed.jsonl \
        --rounds 3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path

from intentguard.classifier import ONNXClassifier
from intentguard.policy import Policy

logger = logging.getLogger(__name__)


def build_traffic(examples: list[dict], total: int = 1000) -> list[dict]:
    """Build a realistic traffic mix from test data.

    Distribution: 70% clean on-topic, 15% clean off-topic,
    10% borderline/ambiguous, 5% adversarial.
    """
    by_cat = {}
    for ex in examples:
        cat = ex.get("category", "unknown")
        by_cat.setdefault(cat, []).append(ex)

    traffic = []
    clean_on = by_cat.get("clean_on_topic", []) + [
        e for e in examples if e["label"] == "allow" and e["category"] != "clean_on_topic"
    ]
    clean_off = by_cat.get("clean_off_topic", []) + [
        e for e in examples if e["label"] == "deny" and "adversarial" not in e.get("category", "")
    ]
    ambiguous = [e for e in examples if e["label"] == "abstain"]
    adversarial = [
        e for e in examples
        if "adversarial" in e.get("category", "")
        or "wrapping" in e.get("category", "")
        or "multi_intent" in e.get("category", "")
    ]

    counts = {
        "on_topic": int(total * 0.70),
        "off_topic": int(total * 0.15),
        "ambiguous": int(total * 0.10),
        "adversarial": int(total * 0.05),
    }

    for _ in range(counts["on_topic"]):
        traffic.append(random.choice(clean_on) if clean_on else random.choice(examples))
    for _ in range(counts["off_topic"]):
        traffic.append(random.choice(clean_off) if clean_off else random.choice(examples))
    for _ in range(counts["ambiguous"]):
        traffic.append(random.choice(ambiguous) if ambiguous else random.choice(examples))
    for _ in range(counts["adversarial"]):
        traffic.append(random.choice(adversarial) if adversarial else random.choice(examples))

    random.shuffle(traffic)
    return traffic


def run_simulation(
    classifier: ONNXClassifier, traffic: list[dict],
) -> dict:
    """Run all queries and collect results."""
    results = []
    latencies = []

    for ex in traffic:
        start = time.perf_counter()
        result = classifier.classify(ex["text"])
        elapsed_ms = (time.perf_counter() - start) * 1000

        correct = result.decision.value == ex["label"]
        results.append({
            "correct": correct,
            "predicted": result.decision.value,
            "expected": ex["label"],
        })
        latencies.append(elapsed_ms)

    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results)

    latencies.sort()
    n = len(latencies)

    return {
        "total": len(results),
        "accuracy": round(accuracy, 4),
        "correct": correct_count,
        "predictions": dict(Counter(r["predicted"] for r in results)),
        "expected": dict(Counter(r["expected"] for r in results)),
        "latency_p50_ms": round(latencies[n // 2], 2),
        "latency_p95_ms": round(latencies[int(n * 0.95)], 2),
        "latency_p99_ms": round(latencies[int(n * 0.99)], 2),
        "latency_mean_ms": round(sum(latencies) / n, 2),
        "latency_max_ms": round(max(latencies), 2),
        "throughput_qps": round(n / (sum(latencies) / 1000), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Simulate production traffic")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    policy = Policy.from_file(args.policy)
    cal = Path(args.calibration) if args.calibration else None
    classifier = ONNXClassifier(
        policy, Path(args.model), Path(args.tokenizer), cal,
    )

    examples = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    logger.info("Loaded %d test examples", len(examples))

    for i in range(args.rounds):
        traffic = build_traffic(examples, args.total)
        logger.info("Round %d: %d queries...", i + 1, len(traffic))
        results = run_simulation(classifier, traffic)

        print(f"\n--- Round {i + 1} ---")
        print(f"  Accuracy: {results['accuracy']:.1%}")
        print(f"  Throughput: {results['throughput_qps']} qps")
        print(f"  Latency: p50={results['latency_p50_ms']}ms "
              f"p95={results['latency_p95_ms']}ms "
              f"p99={results['latency_p99_ms']}ms")
        print(f"  Predictions: {results['predictions']}")


if __name__ == "__main__":
    main()
