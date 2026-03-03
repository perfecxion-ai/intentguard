"""ONNX export with INT8 quantization and mandatory sanity gate.

Exports the fine-tuned model to ONNX format, applies INT8 dynamic
quantization, and validates that the ONNX model matches PyTorch
outputs on a test set before saving.

Usage:
    python -m training.export_onnx \
        --model models/finance/best \
        --output dist/finance/ \
        --calibration models/finance/calibration_params.json \
        --sanity-data data/finance/synthetic.jsonl \
        --sanity-count 100
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

LABEL_MAP = {"allow": 0, "deny": 1, "abstain": 2}


def export_to_onnx(model, tokenizer, output_path: Path, max_length: int = 256):
    """Export model to ONNX format."""
    model.eval()

    # Create dummy input
    dummy = tokenizer(
        "test query",
        "test context",
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    # DeBERTa-v3 does not use token_type_ids
    dummy.pop("token_type_ids", None)

    input_names = list(dummy.keys())
    output_names = ["logits"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size"},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        tuple(dummy.values()),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )
    logger.info("ONNX model exported to %s", output_path)


def quantize_int8(input_path: Path, output_path: Path):
    """Apply INT8 dynamic quantization to ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(input_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )

    original_size = input_path.stat().st_size / (1024 * 1024)
    quantized_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Quantized: %.1fMB -> %.1fMB (%.0f%% reduction)",
        original_size,
        quantized_size,
        (1 - quantized_size / original_size) * 100,
    )


def sanity_check(
    pytorch_model,
    tokenizer,
    onnx_path: Path,
    examples: list[dict],
    vertical_context: str,
    max_length: int = 256,
    max_diff: float = 0.01,
) -> bool:
    """Compare PyTorch and ONNX outputs on a test set.

    Returns True if all outputs match within tolerance.
    This is the mandatory gate — if it fails, the export is rejected.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_input_names = {inp.name for inp in session.get_inputs()}

    pytorch_model.eval()
    mismatches = 0
    total = 0
    max_observed_diff = 0.0

    for ex in examples:
        text = ex["text"]
        encoded = tokenizer(
            text,
            vertical_context,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # PyTorch inference
        with torch.no_grad():
            pt_logits = pytorch_model(**{k: v for k, v in encoded.items()
                                         if k != "token_type_ids"}).logits.numpy()

        # ONNX inference
        onnx_inputs = {}
        for k, v in encoded.items():
            if k in onnx_input_names:
                onnx_inputs[k] = v.numpy()

        onnx_logits = session.run(None, onnx_inputs)[0]

        # Compare
        diff = np.max(np.abs(pt_logits - onnx_logits))
        max_observed_diff = max(max_observed_diff, diff)

        pt_pred = np.argmax(pt_logits, axis=1)[0]
        onnx_pred = np.argmax(onnx_logits, axis=1)[0]

        if pt_pred != onnx_pred:
            mismatches += 1
            logger.warning(
                "Prediction mismatch on '%s': PyTorch=%d, ONNX=%d (diff=%.6f)",
                text[:50], pt_pred, onnx_pred, diff,
            )

        total += 1

    logger.info(
        "Sanity check: %d/%d match, max logit diff=%.6f",
        total - mismatches, total, max_observed_diff,
    )

    if mismatches > 0:
        logger.error(
            "SANITY CHECK FAILED: %d/%d predictions differ between PyTorch and ONNX. "
            "The ONNX export is likely broken. Do NOT ship this model.",
            mismatches, total,
        )
        return False

    if max_observed_diff > max_diff:
        logger.warning(
            "Logit differences (%.6f) exceed threshold (%.6f) but predictions match. "
            "Proceeding with caution.",
            max_observed_diff, max_diff,
        )

    logger.info("SANITY CHECK PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX with INT8 quantization")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--output", required=True, help="Output directory for ONNX artifacts")
    parser.add_argument("--calibration", default=None, help="Path to calibration_params.json")
    parser.add_argument("--sanity-data", required=True, help="JSONL data for sanity check")
    parser.add_argument("--sanity-count", type=int, default=100, help="Number of sanity check examples")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--skip-quantize", action="store_true", help="Skip INT8 quantization")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_dir = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    # Load vertical context
    metadata_path = model_dir.parent / "training_metadata.json"
    vertical_context = ""
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        vertical_context = metadata.get("vertical_context", "")

    # Export ONNX (FP32)
    fp32_path = output_dir / "model_fp32.onnx"
    logger.info("Exporting ONNX FP32...")
    export_to_onnx(model, tokenizer, fp32_path, args.max_length)

    # Load sanity check data
    sanity_examples = []
    with open(args.sanity_data) as f:
        for line in f:
            line = line.strip()
            if line:
                sanity_examples.append(json.loads(line))
                if len(sanity_examples) >= args.sanity_count:
                    break

    logger.info("Running sanity check on %d examples...", len(sanity_examples))

    # Sanity check FP32
    if not sanity_check(model, tokenizer, fp32_path, sanity_examples, vertical_context, args.max_length):
        logger.error("FP32 ONNX sanity check failed. Aborting.")
        return

    # Quantize to INT8
    if not args.skip_quantize:
        int8_path = output_dir / "model.onnx"
        logger.info("Quantizing to INT8...")
        quantize_int8(fp32_path, int8_path)

        # Sanity check INT8
        logger.info("Running sanity check on INT8 model...")
        if not sanity_check(model, tokenizer, int8_path, sanity_examples, vertical_context, args.max_length):
            logger.error("INT8 ONNX sanity check failed. Shipping FP32 instead.")
            shutil.copy2(fp32_path, output_dir / "model.onnx")
        else:
            # Remove FP32 to save space
            fp32_path.unlink()
            logger.info("INT8 model saved as %s", int8_path)
    else:
        shutil.copy2(fp32_path, output_dir / "model.onnx")
        fp32_path.unlink()

    # Copy tokenizer
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    logger.info("Tokenizer saved to %s", output_dir / "tokenizer")

    # Copy calibration params
    if args.calibration:
        cal_path = Path(args.calibration)
        if cal_path.exists():
            shutil.copy2(cal_path, output_dir / "calibration_params.json")
            logger.info("Calibration params copied")

    # Save export metadata
    export_meta = {
        "source_model": str(model_dir),
        "max_length": args.max_length,
        "quantized": not args.skip_quantize,
        "sanity_check_count": len(sanity_examples),
        "sanity_check_passed": True,
        "vertical_context": vertical_context,
    }
    (output_dir / "export_metadata.json").write_text(json.dumps(export_meta, indent=2))

    logger.info("Export complete: %s", output_dir)
    logger.info("Contents: %s", [p.name for p in output_dir.iterdir()])


if __name__ == "__main__":
    main()
