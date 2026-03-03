"""Tests for classifier logic."""

from intentguard.classifier import StubClassifier
from intentguard.policy import Policy
from intentguard.schema import Decision


def _make_policy(**overrides):
    base = {
        "vertical": "test",
        "version": "1.0",
        "display_name": "Test",
        "scope": {"core_topics": ["testing"]},
        "responses": {
            "deny_message": "Off topic.",
            "abstain_message": "Can you clarify?",
        },
    }
    base.update(overrides)
    return Policy.from_dict(base)


class TestThresholdLogic:
    def test_allow_when_confident(self):
        policy = _make_policy()
        classifier = StubClassifier(policy)
        probs = {"allow": 0.92, "deny": 0.04, "abstain": 0.04}
        decision, conf = classifier._apply_thresholds(probs)
        assert decision == Decision.ALLOW
        assert conf == 0.92

    def test_deny_when_confident(self):
        policy = _make_policy()
        classifier = StubClassifier(policy)
        probs = {"allow": 0.03, "deny": 0.94, "abstain": 0.03}
        decision, conf = classifier._apply_thresholds(probs)
        assert decision == Decision.DENY
        assert conf == 0.94

    def test_abstain_on_borderline(self):
        policy = _make_policy()
        classifier = StubClassifier(policy)
        # High allow but not enough margin
        probs = {"allow": 0.50, "deny": 0.45, "abstain": 0.05}
        decision, _ = classifier._apply_thresholds(probs)
        assert decision == Decision.ABSTAIN

    def test_abstain_when_no_class_dominant(self):
        policy = _make_policy()
        classifier = StubClassifier(policy)
        probs = {"allow": 0.35, "deny": 0.33, "abstain": 0.32}
        decision, _ = classifier._apply_thresholds(probs)
        assert decision == Decision.ABSTAIN

    def test_encoding_tricks_force_abstain(self):
        policy = _make_policy()
        classifier = StubClassifier(policy)
        probs = {"allow": 0.95, "deny": 0.02, "abstain": 0.03}
        decision, _ = classifier._apply_thresholds(probs, tricks_detected=True)
        assert decision == Decision.ABSTAIN

    def test_allow_needs_margin(self):
        policy = _make_policy()
        classifier = StubClassifier(policy)
        # p_allow is above tau_allow but margin is too small
        probs = {"allow": 0.82, "deny": 0.78, "abstain": 0.0}
        decision, _ = classifier._apply_thresholds(probs)
        assert decision == Decision.ABSTAIN

    def test_custom_thresholds(self):
        policy = _make_policy(decision={
            "tau_allow": 0.60,
            "tau_deny": 0.60,
            "margin_allow": 0.05,
            "margin_deny": 0.05,
        })
        classifier = StubClassifier(policy)
        probs = {"allow": 0.65, "deny": 0.20, "abstain": 0.15}
        decision, _ = classifier._apply_thresholds(probs)
        assert decision == Decision.ALLOW


class TestClassifyPipeline:
    def test_empty_input_abstains(self):
        policy = _make_policy()
        classifier = StubClassifier(policy)
        result = classifier.classify("")
        assert result.decision == Decision.ABSTAIN

    def test_whitespace_only_abstains(self):
        policy = _make_policy()
        classifier = StubClassifier(policy)
        result = classifier.classify("   \n\t  ")
        assert result.decision == Decision.ABSTAIN

    def test_returns_message_on_deny(self):
        policy = _make_policy()
        classifier = StubClassifier(policy)
        # Run multiple times — stub is random, but deny should always have a message
        for _ in range(50):
            result = classifier.classify("test query")
            if result.decision == Decision.DENY:
                assert result.message != ""
                assert "Off topic" in result.message
                break

    def test_returns_vertical(self):
        policy = _make_policy()
        classifier = StubClassifier(policy)
        result = classifier.classify("some query")
        assert result.vertical == "test"
