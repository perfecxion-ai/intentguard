"""Train vertical router model.

Loads data from all verticals, labels with source vertical name,
and trains an N-way classifier for vertical routing.

Usage:
    python -m training.train_router \
        --verticals finance healthcare legal \
        --output models/router \
        --base-model microsoft/deberta-v3-xsmall
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def load_vertical_data(vertical: str, data_dir: Path, max_per_vertical: int = 5000) -> list[dict]:
    """Load and label data from a vertical's data directory."""
    examples = []
    vert_dir = data_dir / vertical

    if not vert_dir.exists():
        logger.warning("Data directory not found: %s", vert_dir)
        return examples

    for jsonl_file in sorted(vert_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                text = ex.get("text", ex.get("query", ""))
                if text:
                    examples.append({"text": text, "vertical": vertical})
                    if len(examples) >= max_per_vertical:
                        return examples

    return examples


def main():
    parser = argparse.ArgumentParser(description="Train vertical router classifier")
    parser.add_argument("--verticals", nargs="+", required=True,
                        help="List of vertical names")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--output", required=True, help="Output directory for router model")
    parser.add_argument("--base-model", default="microsoft/deberta-v3-xsmall")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-per-vertical", type=int, default=5000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data from all verticals
    all_data = []
    label2id = {}
    for i, vertical in enumerate(args.verticals):
        label2id[vertical] = i
        examples = load_vertical_data(vertical, data_dir, args.max_per_vertical)
        logger.info("Loaded %d examples for vertical '%s'", len(examples), vertical)
        all_data.extend(examples)

    id2label = {v: k for k, v in label2id.items()}
    logger.info("Total examples: %d across %d verticals", len(all_data), len(label2id))

    # Split
    texts = [ex["text"] for ex in all_data]
    labels = [label2id[ex["vertical"]] for ex in all_data]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, stratify=labels, random_state=42,
    )

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # Training
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()

    # Save best model
    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Export to ONNX
    logger.info("Exporting router to ONNX...")
    from training.export_onnx import export_to_onnx

    onnx_path = output_dir / "model.onnx"
    export_to_onnx(model, tokenizer, onnx_path, max_length=args.max_length)
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    # Save metadata
    metadata = {
        "verticals": args.verticals,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "base_model": args.base_model,
        "max_length": args.max_length,
        "train_examples": len(train_texts),
        "val_examples": len(val_texts),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info("Router training complete: %s", output_dir)


if __name__ == "__main__":
    main()
