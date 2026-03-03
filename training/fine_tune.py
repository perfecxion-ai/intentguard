"""Fine-tuning script for vertical intent classifiers.

Fine-tunes a DeBERTa-v3 model for 3-way classification
(ALLOW / DENY / ABSTAIN) using policy-driven training data.

Usage:
    python -m training.fine_tune \
        --data data/finance/ \
        --config training/train_config.yaml \
        --output models/finance/
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

LABEL_MAP = {"allow": 0, "deny": 1, "abstain": 2}
LABEL_NAMES = ["allow", "deny", "abstain"]


def load_data(data_dir: Path) -> list[dict]:
    """Load all JSONL files from a directory."""
    examples = []
    for path in sorted(data_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    if ex.get("label") in LABEL_MAP:
                        examples.append(ex)
    logger.info("Loaded %d examples from %s", len(examples), data_dir)
    return examples


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Compute inverse-frequency weights for class imbalance."""
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(len(LABEL_MAP)):
        count = counts.get(i, 1)
        weights.append(total / (len(LABEL_MAP) * count))
    return torch.tensor(weights, dtype=torch.float32)


def build_vertical_context(data_dir: Path) -> str:
    """Build vertical context from the policy file if available."""
    # Look for a policy reference in the data
    policy_path = data_dir.parent / "policies" / f"{data_dir.name}.json"
    if not policy_path.exists():
        # Try relative to project root
        for candidate in [
            Path("policies") / f"{data_dir.name}.json",
            data_dir / "policy.json",
        ]:
            if candidate.exists():
                policy_path = candidate
                break

    if policy_path.exists():
        from intentguard.policy import Policy
        policy = Policy.from_file(policy_path)
        return policy.vertical_context()

    return ""


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def freeze_embeddings(model):
    """Freeze the word embedding layer to prevent overfitting on small data."""
    for param in model.deberta.embeddings.word_embeddings.parameters():
        param.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total = sum(1 for p in model.parameters())
    logger.info("Frozen %d/%d parameter groups (embeddings)", frozen, total)


def get_layer_wise_lr_groups(model, base_lr: float, decay: float = 0.9):
    """Create parameter groups with decaying LR from top to bottom layers."""
    groups = []

    # Classifier head: full learning rate
    groups.append({
        "params": [p for n, p in model.named_parameters()
                   if "classifier" in n or "pooler" in n],
        "lr": base_lr * 5,  # higher LR for fresh head
    })

    # Get encoder layers
    if hasattr(model, "deberta"):
        encoder = model.deberta.encoder
        num_layers = len(encoder.layer)
        for i in reversed(range(num_layers)):
            layer_lr = base_lr * (decay ** (num_layers - 1 - i))
            groups.append({
                "params": list(encoder.layer[i].parameters()),
                "lr": layer_lr,
            })

    return groups


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a vertical intent classifier")
    parser.add_argument("--data", required=True, help="Directory with training JSONL files")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument("--vertical-context", default=None, help="Override vertical context string")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(Path(args.config))
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    examples = load_data(data_dir)
    if not examples:
        logger.error("No training examples found in %s", data_dir)
        return

    # Build vertical context
    vertical_context = args.vertical_context or build_vertical_context(data_dir)
    logger.info("Vertical context: %s", vertical_context[:100] + "..." if len(vertical_context) > 100 else vertical_context)

    # Split
    eval_split = config.get("eval_split", 0.15)
    train_examples, eval_examples = train_test_split(
        examples,
        test_size=eval_split,
        stratify=[ex["label"] for ex in examples],
        random_state=42,
    )
    logger.info("Train: %d, Eval: %d", len(train_examples), len(eval_examples))

    # Log class distribution
    train_labels = [LABEL_MAP[ex["label"]] for ex in train_examples]
    eval_labels = [LABEL_MAP[ex["label"]] for ex in eval_examples]
    logger.info("Train distribution: %s", Counter(train_labels))
    logger.info("Eval distribution: %s", Counter(eval_labels))

    # Load tokenizer and model
    base_model = config.get("base_model", "microsoft/deberta-v3-base")
    max_length = config.get("max_length", 256)

    logger.info("Loading %s...", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=3,
        problem_type="single_label_classification",
    )

    # Set label names for the model config
    model.config.label2id = LABEL_MAP
    model.config.id2label = {v: k for k, v in LABEL_MAP.items()}

    # Freeze embeddings
    if config.get("freeze_embeddings", True):
        freeze_embeddings(model)

    # Tokenize
    def tokenize(examples_list: list[dict]) -> dict:
        texts = [ex["text"] for ex in examples_list]
        contexts = [vertical_context] * len(texts)
        encoded = tokenizer(
            texts, contexts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        encoded["labels"] = [LABEL_MAP[ex["label"]] for ex in examples_list]
        return encoded

    train_encoded = tokenize(train_examples)
    eval_encoded = tokenize(eval_examples)

    train_dataset = Dataset.from_dict(train_encoded)
    eval_dataset = Dataset.from_dict(eval_encoded)

    # Class weights
    class_weights = compute_class_weights(train_labels)
    logger.info("Class weights: %s", class_weights.tolist())

    # Training arguments
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 5)
    lr = config.get("learning_rate", 2e-5)
    label_smoothing = config.get("label_smoothing", 0.05)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        label_smoothing_factor=label_smoothing,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
    )

    # Metrics
    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).mean()

        # Per-class accuracy
        metrics = {"accuracy": accuracy}
        for label_name, label_id in LABEL_MAP.items():
            mask = labels == label_id
            if mask.sum() > 0:
                metrics[f"accuracy_{label_name}"] = (preds[mask] == labels[mask]).mean()
        return metrics

    # Train
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    trainer.train()

    # Save best model
    best_model_dir = output_dir / "best"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    logger.info("Best model saved to %s", best_model_dir)

    # Save training metadata
    metadata = {
        "base_model": base_model,
        "vertical_context": vertical_context,
        "train_count": len(train_examples),
        "eval_count": len(eval_examples),
        "label_map": LABEL_MAP,
        "max_length": max_length,
        "class_weights": class_weights.tolist(),
        "config": config,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Training metadata saved")

    # Final eval
    eval_results = trainer.evaluate()
    logger.info("Final eval: %s", eval_results)


if __name__ == "__main__":
    main()
