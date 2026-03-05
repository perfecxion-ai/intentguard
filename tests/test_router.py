"""Tests for the multi-vertical router."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from intentguard.classifier import BaseClassifier
from intentguard.policy import Policy
from intentguard.router import VerticalRouter
from intentguard.schema import Decision


def _make_policy(vertical: str) -> Policy:
    return Policy.from_dict({
        "vertical": vertical,
        "version": "1.0",
        "display_name": f"{vertical.title()} Services",
        "scope": {"core_topics": [vertical]},
        "responses": {
            "deny_message": f"Not a {vertical} question.",
            "abstain_message": f"Is this about {vertical}?",
        },
    })


class MockClassifier(BaseClassifier):
    """Returns a fixed decision for testing."""

    def __init__(self, vertical: str, decision: Decision = Decision.ALLOW):
        super().__init__(_make_policy(vertical))
        self._decision = decision

    def predict(self, text: str) -> tuple[dict[str, float], bool]:
        if self._decision == Decision.ALLOW:
            return {"allow": 0.95, "deny": 0.03, "abstain": 0.02}, True
        elif self._decision == Decision.DENY:
            return {"allow": 0.02, "deny": 0.95, "abstain": 0.03}, True
        else:
            return {"allow": 0.3, "deny": 0.3, "abstain": 0.4}, True


class TestVerticalRouter:
    @pytest.fixture
    def mock_router(self):
        """Create a router with mock classifiers and a mock ONNX session."""
        classifiers = {
            "finance": MockClassifier("finance", Decision.ALLOW),
            "healthcare": MockClassifier("healthcare", Decision.DENY),
            "legal": MockClassifier("legal", Decision.ABSTAIN),
        }

        router = VerticalRouter.__new__(VerticalRouter)
        router.vertical_labels = ["finance", "healthcare", "legal"]
        router.classifiers = classifiers
        router.max_length = 128

        # Mock ONNX session that returns finance as highest score
        mock_session = MagicMock()
        # Finance=[0.8, 0.1, 0.1]
        mock_session.run.return_value = [np.array([[2.0, -1.0, -1.0]])]
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        router.session = mock_session
        router.input_names = {"input_ids", "attention_mask"}

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.ones((1, 128), dtype=np.int64),
            "attention_mask": np.ones((1, 128), dtype=np.int64),
        }
        router.tokenizer = mock_tokenizer

        return router

    def test_route_returns_best_vertical(self, mock_router):
        result = mock_router.route("What is a stock?")
        assert result == "finance"

    def test_route_scores_returns_all_verticals(self, mock_router):
        scores = mock_router.route_scores("test query")
        assert set(scores.keys()) == {"finance", "healthcare", "legal"}
        assert all(0 <= v <= 1 for v in scores.values())
        assert abs(sum(scores.values()) - 1.0) < 1e-5

    def test_classify_routes_to_correct_classifier(self, mock_router):
        result, vertical, scores = mock_router.classify("What is a stock?")
        assert vertical == "finance"
        assert result.decision == Decision.ALLOW
        assert result.vertical == "finance"

    def test_classify_returns_router_scores(self, mock_router):
        _, _, scores = mock_router.classify("test")
        assert "finance" in scores
        assert scores["finance"] > scores["healthcare"]

    def test_classify_healthcare_route(self, mock_router):
        # Override session to route to healthcare
        mock_router.session.run.return_value = [np.array([[-1.0, 2.0, -1.0]])]
        result, vertical, _ = mock_router.classify("What are flu symptoms?")
        assert vertical == "healthcare"
        assert result.decision == Decision.DENY  # mock healthcare classifier returns DENY

    def test_classify_legal_route(self, mock_router):
        mock_router.session.run.return_value = [np.array([[-1.0, -1.0, 2.0]])]
        result, vertical, _ = mock_router.classify("What is a contract?")
        assert vertical == "legal"
        assert result.decision == Decision.ABSTAIN

    def test_fallback_on_unknown_vertical(self, mock_router):
        """If router returns a vertical not in classifiers, fall back to first."""
        mock_router.vertical_labels = ["unknown", "finance", "healthcare"]
        mock_router.session.run.return_value = [np.array([[3.0, -1.0, -1.0]])]
        result, vertical, _ = mock_router.classify("test")
        # Falls back to first classifier in dict
        assert vertical == "finance"


class TestSingleVerticalFallback:
    def test_server_works_without_router(self, client):
        """When ROUTER_ENABLED is unset, server works in single-vertical mode."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["verticals"] is None
        assert data["vertical"] == "finance"

    def test_classify_without_router(self, client):
        resp = client.post("/v1/classify", json={
            "messages": [{"role": "user", "content": "What are mortgage rates?"}]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["routed_vertical"] is None
        assert data["router_scores"] is None


class TestPolicyPackLookup:
    def test_policy_pack_allow(self, finance_policy):
        pack = finance_policy.get_policy_pack("allow")
        assert pack is not None
        assert "calculator" in pack.allowed_tools
        assert "no_trade_execution" in pack.guardrails

    def test_policy_pack_deny(self, finance_policy):
        pack = finance_policy.get_policy_pack("deny")
        assert pack is not None
        assert pack.allowed_tools == []
        assert "block_response" in pack.guardrails

    def test_policy_pack_abstain(self, finance_policy):
        pack = finance_policy.get_policy_pack("abstain")
        assert pack is not None
        assert "require_clarification" in pack.guardrails

    def test_policy_pack_missing(self):
        policy = Policy.from_dict({
            "vertical": "test",
            "version": "1.0",
            "display_name": "Test",
            "scope": {"core_topics": ["test"]},
            "responses": {"deny_message": "no", "abstain_message": "maybe"},
        })
        assert policy.get_policy_pack("allow") is None
