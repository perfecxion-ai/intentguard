"""Intent classifier interface and implementations."""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod

from intentguard.normalize import has_encoding_tricks, normalize
from intentguard.policy import Policy
from intentguard.schema import ClassifyResponse, Decision

logger = logging.getLogger(__name__)


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


class StubClassifier(BaseClassifier):
    """Returns random decisions for testing. No model required."""

    def predict(self, text: str) -> tuple[dict[str, float], bool]:
        # Generate random probabilities that sum to 1
        raw = [random.random() for _ in range(3)]
        total = sum(raw)
        probs = {
            "allow": raw[0] / total,
            "deny": raw[1] / total,
            "abstain": raw[2] / total,
        }
        return probs, False
