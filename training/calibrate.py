"""Calibration script for temperature scaling.

Fits a temperature parameter on held-out data so that softmax
probabilities match actual accuracy. Outputs calibration_params.json
and optionally a reliability diagram.

Usage:
    python -m training.calibrate \
        --model models/finance/best \
        --data data/finance/calibration.jsonl \
        --output models/finance/calibration_params.json \
        --plot models/finance/reliability_diagram.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

LABEL_MAP = {"allow": 0, "deny": 1, "abstain": 2}


def load_calibration_data(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                if ex.get("label") in LABEL_MAP:
                    examples.append(ex)
    return examples


def collect_logits(
    model,
    tokenizer,
    examples: list[dict],
    vertical_context: str,
    max_length: int = 256,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on all examples and collect raw logits."""
    model.eval()
    device = next(model.parameters()).device

    all_logits = []
    all_labels = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        texts = [ex["text"] for ex in batch]
        contexts = [vertical_context] * len(texts)
        labels = [LABEL_MAP[ex["label"]] for ex in batch]

        encoded = tokenizer(
            texts, contexts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.extend(labels)

    return np.concatenate(all_logits, axis=0), np.array(all_labels)


class TemperatureScaler(nn.Module):
    """Learns a single temperature parameter for calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> float:
    """Fit temperature parameter by minimizing NLL on calibration data."""
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits_tensor)
        loss = criterion(scaled, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = scaler.temperature.item()
    logger.info("Fitted temperature: %.4f", temperature)
    return temperature


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_accuracy = accuracies[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        bin_weight = in_bin.sum() / len(labels)
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str = "Reliability Diagram",
    n_bins: int = 15,
):
    """Plot reliability diagram: predicted confidence vs actual accuracy."""
    import matplotlib.pyplot as plt

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
        bin_accuracies.append(accuracies[in_bin].mean())
        bin_counts.append(in_bin.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(bin_centers, bin_accuracies, width=1.0 / n_bins, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Mean predicted confidence")
    ax1.set_ylabel("Fraction of positives (accuracy)")
    ax1.set_title(title)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()

    # Histogram of predictions
    ax2.bar(bin_centers, bin_counts, width=1.0 / n_bins, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Mean predicted confidence")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Reliability diagram saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Calibrate model with temperature scaling")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--data", required=True, help="Path to calibration JSONL data")
    parser.add_argument("--output", required=True, help="Output path for calibration_params.json")
    parser.add_argument("--plot", default=None, help="Output path for reliability diagram")
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_dir = Path(args.model)
    data_path = Path(args.data)

    # Load model and tokenizer
    logger.info("Loading model from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    # Load training metadata for vertical context
    metadata_path = model_dir.parent / "training_metadata.json"
    vertical_context = ""
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        vertical_context = metadata.get("vertical_context", "")
    logger.info("Vertical context: %s", vertical_context[:80])

    # Load calibration data
    examples = load_calibration_data(data_path)
    logger.info("Loaded %d calibration examples", len(examples))

    if len(examples) < 50:
        logger.warning("Calibration set is small (%d). Recommend at least 200 examples.", len(examples))

    # Collect logits
    logits, labels = collect_logits(
        model, tokenizer, examples, vertical_context, args.max_length
    )
    logger.info("Collected logits: shape %s", logits.shape)

    # Compute pre-calibration ECE
    pre_probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pre_ece = compute_ece(pre_probs, labels)
    logger.info("Pre-calibration ECE: %.4f", pre_ece)

    # Fit temperature
    temperature = fit_temperature(logits, labels)

    # Compute post-calibration ECE
    scaled_logits = logits / temperature
    post_probs = torch.softmax(torch.tensor(scaled_logits), dim=1).numpy()
    post_ece = compute_ece(post_probs, labels)
    logger.info("Post-calibration ECE: %.4f", post_ece)

    # Save calibration params
    params = {
        "temperature": temperature,
        "pre_calibration_ece": round(pre_ece, 4),
        "post_calibration_ece": round(post_ece, 4),
        "calibration_set_size": len(examples),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(params, indent=2))
    logger.info("Calibration params saved to %s", output_path)

    # Plot if requested
    if args.plot:
        plot_reliability_diagram(
            post_probs, labels,
            Path(args.plot),
            title=f"Reliability Diagram (T={temperature:.3f}, ECE={post_ece:.4f})",
        )

    # Summary
    if post_ece < 0.03:
        logger.info("PASS: ECE %.4f < 0.03 threshold", post_ece)
    else:
        logger.warning("FAIL: ECE %.4f >= 0.03 threshold — consider more calibration data", post_ece)


if __name__ == "__main__":
    main()
