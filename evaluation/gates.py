"""Shipping gates: compute LBR, OPR, AOC and pass/fail against thresholds.

Usage:
    python -m evaluation.gates \
        --model dist/finance/model.onnx \
        --tokenizer dist/finance/tokenizer \
        --calibration dist/finance/calibration_params.json \
        --policy policies/finance.json \
        --data evaluation/test_sets/finance_adversarial.jsonl \
        --output evaluation/gating_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from intentguard.classifier import ONNXClassifier
from intentguard.policy import Policy

logger = logging.getLogger(__name__)

# Default shipping gates
DEFAULT_GATES = {
    "lbr_max": 0.005,   # legitimate-block rate < 0.5%
    "opr_max": 0.02,    # off-topic-pass rate < 2%
    "aoc_max": 0.10,    # abstain-on-clean < 10%
    "ece_max": 0.03,    # expected calibration error < 0.03
}


def evaluate(classifier: ONNXClassifier, examples: list[dict]) -> dict:
    """Run classifier on all examples and compute metrics."""
    results = []
    for ex in examples:
        result = classifier.classify(ex["text"])
        results.append({
            "text": ex["text"],
            "gold_label": ex["label"],
            "predicted": result.decision.value,
            "confidence": result.confidence,
            "category": ex.get("category", "unknown"),
        })

    # Compute core metrics
    gold_allow = [r for r in results if r["gold_label"] == "allow"]
    gold_deny = [r for r in results if r["gold_label"] == "deny"]

    # LBR: P(model=DENY | gold=ALLOW)
    lbr = 0.0
    if gold_allow:
        blocked = sum(1 for r in gold_allow if r["predicted"] == "deny")
        lbr = blocked / len(gold_allow)

    # OPR: P(model=ALLOW | gold=DENY)
    opr = 0.0
    if gold_deny:
        passed = sum(1 for r in gold_deny if r["predicted"] == "allow")
        opr = passed / len(gold_deny)

    # AOC: P(model=ABSTAIN | gold=ALLOW, clean traffic)
    clean_allow = [r for r in gold_allow if r["category"] in ("positive", "clean")]
    aoc = 0.0
    if clean_allow:
        abstained = sum(1 for r in clean_allow if r["predicted"] == "abstain")
        aoc = abstained / len(clean_allow)

    # Coverage: fraction of non-abstain decisions
    total = len(results)
    abstain_count = sum(1 for r in results if r["predicted"] == "abstain")
    coverage = 1 - (abstain_count / total) if total > 0 else 0

    # Per-category breakdown
    empty_cat = {"total": 0, "correct": 0, "allow": 0, "deny": 0, "abstain": 0}
    category_metrics = defaultdict(lambda: dict(empty_cat))
    for r in results:
        cat = r["category"]
        category_metrics[cat]["total"] += 1
        category_metrics[cat][r["predicted"]] += 1
        if r["predicted"] == r["gold_label"]:
            category_metrics[cat]["correct"] += 1

    # Overall accuracy
    correct = sum(1 for r in results if r["predicted"] == r["gold_label"])
    accuracy = correct / total if total else 0

    return {
        "total_examples": total,
        "accuracy": round(accuracy, 4),
        "lbr": round(lbr, 4),
        "opr": round(opr, 4),
        "aoc": round(aoc, 4),
        "coverage": round(coverage, 4),
        "label_distribution": dict(Counter(r["gold_label"] for r in results)),
        "prediction_distribution": dict(Counter(r["predicted"] for r in results)),
        "per_category": dict(category_metrics),
        "details": results,
    }


def check_gates(metrics: dict, gates: dict = DEFAULT_GATES) -> dict:
    """Check metrics against shipping gates."""
    checks = {}

    checks["lbr"] = {
        "value": metrics["lbr"],
        "threshold": gates["lbr_max"],
        "passed": metrics["lbr"] <= gates["lbr_max"],
        "description": f"Legitimate-block rate: {metrics['lbr']:.2%} (max {gates['lbr_max']:.2%})",
    }

    checks["opr"] = {
        "value": metrics["opr"],
        "threshold": gates["opr_max"],
        "passed": metrics["opr"] <= gates["opr_max"],
        "description": f"Off-topic-pass rate: {metrics['opr']:.2%} (max {gates['opr_max']:.2%})",
    }

    checks["aoc"] = {
        "value": metrics["aoc"],
        "threshold": gates["aoc_max"],
        "passed": metrics["aoc"] <= gates["aoc_max"],
        "description": f"Abstain-on-clean: {metrics['aoc']:.2%} (max {gates['aoc_max']:.2%})",
    }

    all_passed = all(c["passed"] for c in checks.values())

    return {
        "ship_decision": "SHIP" if all_passed else "NO-SHIP",
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description="Run shipping gate evaluation")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer directory")
    parser.add_argument("--calibration", default=None, help="Path to calibration_params.json")
    parser.add_argument("--policy", required=True, help="Path to policy.json")
    parser.add_argument("--data", required=True, help="Path to evaluation JSONL")
    parser.add_argument("--output", default=None, help="Save report to JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load policy and classifier
    policy = Policy.from_file(args.policy)
    cal_path = Path(args.calibration) if args.calibration else None

    classifier = ONNXClassifier(
        policy=policy,
        model_path=Path(args.model),
        tokenizer_path=Path(args.tokenizer),
        calibration_path=cal_path,
    )

    # Load evaluation data
    examples = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    logger.info("Evaluating on %d examples...", len(examples))

    # Run evaluation
    metrics = evaluate(classifier, examples)
    gate_report = check_gates(metrics)

    # Print results
    print("\n" + "=" * 60)
    print(f"  GATING REPORT: {gate_report['ship_decision']}")
    print("=" * 60)

    for name, check in gate_report["checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['description']}")

    print(f"\n  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Coverage: {metrics['coverage']:.2%}")
    print(f"  Total examples: {metrics['total_examples']}")
    print(f"  Labels: {metrics['label_distribution']}")
    print(f"  Predictions: {metrics['prediction_distribution']}")
    print("=" * 60)

    # Save report
    if args.output:
        report = {**metrics, **gate_report}
        # Remove details from saved report (too large)
        report.pop("details", None)
        Path(args.output).write_text(json.dumps(report, indent=2))
        logger.info("Report saved to %s", args.output)


if __name__ == "__main__":
    main()
