"""Intent classifier interface and implementations."""

from __future__ import annotations

import json
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from intentguard.normalize import has_encoding_tricks, normalize
from intentguard.policy import Policy
from intentguard.schema import ClassifyResponse, Decision

logger = logging.getLogger(__name__)

LABEL_NAMES = ["allow", "deny", "abstain"]


class BaseClassifier(ABC):
    """Interface for intent classifiers."""

    def __init__(self, policy: Policy):
        self.policy = policy

    @abstractmethod
    def predict(self, text: str) -> tuple[dict[str, float], bool]:
        """Run inference and return class probabilities.

        Returns:
            (probabilities dict, model_loaded flag)
            probabilities keys: "allow", "deny", "abstain"
        """
        ...

    def classify(self, text: str) -> ClassifyResponse:
        """Classify a user message. Full pipeline: normalize, predict, decide."""
        normalized = normalize(text, max_chars=2000)

        if not normalized.strip():
            return ClassifyResponse(
                decision=Decision.ABSTAIN,
                confidence=1.0,
                vertical=self.policy.vertical,
                message=self.policy.abstain_response(),
            )

        # Encoding tricks bias toward abstain
        tricks_detected = has_encoding_tricks(text)

        probs, _ = self.predict(normalized)
        decision, confidence = self._apply_thresholds(probs, tricks_detected)

        message = ""
        if decision == Decision.DENY:
            message = self.policy.deny_response()
        elif decision == Decision.ABSTAIN:
            message = self.policy.abstain_response()

        return ClassifyResponse(
            decision=decision,
            confidence=confidence,
            vertical=self.policy.vertical,
            message=message,
            probabilities=probs,
        )

    def _apply_thresholds(
        self,
        probs: dict[str, float],
        tricks_detected: bool = False,
    ) -> tuple[Decision, float]:
        """Apply margin-based decision rule.

        ALLOW if: p_allow >= tau_allow AND p_allow - max(p_deny, p_abstain) >= margin_allow
        DENY if:  p_deny >= tau_deny AND p_deny - max(p_allow, p_abstain) >= margin_deny
        Otherwise: ABSTAIN
        """
        t = self.policy.thresholds
        p_allow = probs.get("allow", 0.0)
        p_deny = probs.get("deny", 0.0)
        p_abstain = probs.get("abstain", 0.0)

        # If encoding tricks detected, don't allow high-confidence ALLOW
        if tricks_detected:
            return Decision.ABSTAIN, p_abstain

        # Check ALLOW first (bias toward allowing legitimate queries)
        if p_allow >= t.tau_allow and (p_allow - max(p_deny, p_abstain)) >= t.margin_allow:
            return Decision.ALLOW, p_allow

        # Check DENY
        if p_deny >= t.tau_deny and (p_deny - max(p_allow, p_abstain)) >= t.margin_deny:
            return Decision.DENY, p_deny

        # Default to ABSTAIN
        return Decision.ABSTAIN, max(p_abstain, 1.0 - p_allow - p_deny)


class ONNXClassifier(BaseClassifier):
    """Production classifier using ONNX Runtime."""

    def __init__(
        self,
        policy: Policy,
        model_path: Path,
        tokenizer_path: Path,
        calibration_path: Path | None = None,
        max_length: int = 256,
        intra_op_threads: int = 4,
        inter_op_threads: int = 1,
    ):
        super().__init__(policy)
        import onnxruntime as ort

        # Load ONNX model
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_op_threads
        sess_options.inter_op_num_threads = inter_op_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}
        logger.info("ONNX model loaded: %s", model_path)

        # Load tokenizer (HuggingFace format)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self.max_length = max_length

        # Load calibration
        self.temperature = 1.0
        if calibration_path and calibration_path.exists():
            params = json.loads(calibration_path.read_text())
            self.temperature = params.get("temperature", 1.0)
            logger.info("Calibration loaded: temperature=%.4f", self.temperature)

        self._vertical_context = policy.vertical_context()

        # Warm up: run one inference to trigger JIT compilation
        self._warmup()

    def _warmup(self):
        """Run a dummy inference to warm up ONNX Runtime."""
        logger.info("Warming up ONNX model...")
        self.predict("warmup query")
        logger.info("Warmup complete")

    def predict(self, text: str) -> tuple[dict[str, float], bool]:
        encoded = self.tokenizer(
            text,
            self._vertical_context,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="np",
        )

        # DeBERTa-v3 does not use token_type_ids
        feed = {k: v for k, v in encoded.items() if k in self.input_names}

        logits = self.session.run(None, feed)[0]

        # Apply temperature scaling
        scaled = logits / self.temperature

        # Softmax
        exp_scores = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs_arr = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        probs_arr = probs_arr[0]  # batch size 1

        probs = {name: float(probs_arr[i]) for i, name in enumerate(LABEL_NAMES)}
        return probs, True


class StubClassifier(BaseClassifier):
    """Returns random decisions for testing. No model required."""

    def predict(self, text: str) -> tuple[dict[str, float], bool]:
        raw = [random.random() for _ in range(3)]
        total = sum(raw)
        probs = {
            "allow": raw[0] / total,
            "deny": raw[1] / total,
            "abstain": raw[2] / total,
        }
        return probs, False
