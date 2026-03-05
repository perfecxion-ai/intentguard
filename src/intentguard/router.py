"""Multi-vertical embeddings router.

Routes queries to the appropriate vertical classifier using a lightweight
N-way classifier, then delegates to the per-vertical ONNXClassifier.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from intentguard.classifier import BaseClassifier, ONNXClassifier
from intentguard.schema import ClassifyResponse

logger = logging.getLogger(__name__)


class VerticalRouter:
    """Routes queries to per-vertical classifiers via a trained router model."""

    def __init__(
        self,
        router_model_path: Path,
        router_tokenizer_path: Path,
        vertical_labels: list[str],
        classifiers: dict[str, BaseClassifier],
        max_length: int = 128,
    ):
        import onnxruntime as ort

        self.vertical_labels = vertical_labels
        self.classifiers = classifiers
        self.max_length = max_length

        # Load router ONNX model
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(router_model_path),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(router_tokenizer_path))

        logger.info(
            "Router loaded: %d verticals (%s)",
            len(vertical_labels), ", ".join(vertical_labels),
        )

    def route(self, text: str) -> str:
        """Return the best vertical name for a query."""
        scores = self.route_scores(text)
        return max(scores, key=scores.get)

    def route_scores(self, text: str) -> dict[str, float]:
        """Return confidence scores per vertical."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="np",
        )

        feed = {k: v for k, v in encoded.items() if k in self.input_names}
        logits = self.session.run(None, feed)[0]

        # Softmax
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        probs = probs[0]

        return {label: float(probs[i]) for i, label in enumerate(self.vertical_labels)}

    def classify(self, text: str) -> tuple[ClassifyResponse, str, dict[str, float]]:
        """Route to the best vertical and classify.

        Returns:
            (ClassifyResponse, routed_vertical, router_scores)
        """
        scores = self.route_scores(text)
        vertical = max(scores, key=scores.get)

        if vertical not in self.classifiers:
            logger.warning("Routed to unknown vertical '%s', falling back to first", vertical)
            vertical = next(iter(self.classifiers))

        result = self.classifiers[vertical].classify(text)
        return result, vertical, scores

    @classmethod
    def from_config(cls, config_path: Path) -> VerticalRouter:
        """Load router from a JSON config file.

        Config format:
        {
            "router_model": "models/router/model.onnx",
            "router_tokenizer": "models/router/tokenizer",
            "verticals": {
                "finance": {
                    "model": "dist/finance/model.onnx",
                    "tokenizer": "dist/finance/tokenizer",
                    "policy": "policies/finance.json",
                    "calibration": "dist/finance/calibration_params.json"
                },
                ...
            }
        }
        """
        from intentguard.policy import Policy

        config = json.loads(config_path.read_text())

        vertical_labels = list(config["verticals"].keys())
        classifiers: dict[str, BaseClassifier] = {}

        for name, vert_config in config["verticals"].items():
            policy = Policy.from_file(vert_config["policy"])
            cal_path = Path(vert_config.get("calibration", ""))

            classifier = ONNXClassifier(
                policy=policy,
                model_path=Path(vert_config["model"]),
                tokenizer_path=Path(vert_config["tokenizer"]),
                calibration_path=cal_path if cal_path.exists() else None,
            )
            classifiers[name] = classifier
            logger.info("Loaded vertical classifier: %s", name)

        router = cls(
            router_model_path=Path(config["router_model"]),
            router_tokenizer_path=Path(config["router_tokenizer"]),
            vertical_labels=vertical_labels,
            classifiers=classifiers,
        )
        return router
