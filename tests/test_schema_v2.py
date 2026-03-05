"""Tests for v2 schema additions — backward compatibility and new fields."""

from intentguard.schema import (
    ClassifyResponse,
    Decision,
    HealthResponse,
    PolicyPackResponse,
)


class TestClassifyResponseV2:
    def test_backward_compat_no_new_fields(self):
        """Existing responses without new fields still work."""
        resp = ClassifyResponse(
            decision=Decision.ALLOW,
            confidence=0.95,
            vertical="finance",
        )
        assert resp.routed_vertical is None
        assert resp.router_scores is None
        assert resp.policy_pack is None

    def test_routed_vertical_field(self):
        resp = ClassifyResponse(
            decision=Decision.ALLOW,
            confidence=0.95,
            vertical="finance",
            routed_vertical="finance",
        )
        assert resp.routed_vertical == "finance"

    def test_router_scores_field(self):
        scores = {"finance": 0.8, "healthcare": 0.15, "legal": 0.05}
        resp = ClassifyResponse(
            decision=Decision.ALLOW,
            confidence=0.95,
            vertical="finance",
            router_scores=scores,
        )
        assert resp.router_scores == scores

    def test_policy_pack_field(self):
        pack = PolicyPackResponse(
            vertical="finance",
            decision="allow",
            allowed_tools=["calculator"],
            guardrails=["disclaimer_required"],
            metadata={"risk_level": "standard"},
        )
        resp = ClassifyResponse(
            decision=Decision.ALLOW,
            confidence=0.95,
            vertical="finance",
            policy_pack=pack,
        )
        assert resp.policy_pack.allowed_tools == ["calculator"]
        assert resp.policy_pack.guardrails == ["disclaimer_required"]

    def test_serialization_excludes_none(self):
        resp = ClassifyResponse(
            decision=Decision.ALLOW,
            confidence=0.95,
            vertical="finance",
        )
        data = resp.model_dump(exclude_none=True)
        assert "routed_vertical" not in data
        assert "router_scores" not in data
        assert "policy_pack" not in data

    def test_full_response_serialization(self):
        resp = ClassifyResponse(
            decision=Decision.DENY,
            confidence=0.92,
            vertical="finance",
            message="Not finance",
            routed_vertical="finance",
            router_scores={"finance": 0.7, "healthcare": 0.3},
            policy_pack=PolicyPackResponse(
                vertical="finance",
                decision="deny",
                allowed_tools=[],
                guardrails=["block_response"],
            ),
        )
        data = resp.model_dump()
        assert data["routed_vertical"] == "finance"
        assert data["policy_pack"]["guardrails"] == ["block_response"]


class TestHealthResponseV2:
    def test_backward_compat_no_verticals(self):
        resp = HealthResponse(
            status="ok",
            model_loaded=True,
            policy_loaded=True,
            vertical="finance",
            version="1.0",
        )
        assert resp.verticals is None

    def test_verticals_field(self):
        resp = HealthResponse(
            status="ok",
            model_loaded=True,
            policy_loaded=True,
            vertical="finance",
            version="1.0",
            verticals=["finance", "healthcare", "legal"],
        )
        assert resp.verticals == ["finance", "healthcare", "legal"]


class TestPolicyPackResponse:
    def test_empty_policy_pack(self):
        pack = PolicyPackResponse(vertical="test", decision="allow")
        assert pack.allowed_tools == []
        assert pack.guardrails == []
        assert pack.metadata == {}

    def test_full_policy_pack(self):
        pack = PolicyPackResponse(
            vertical="finance",
            decision="allow",
            allowed_tools=["calc", "search"],
            guardrails=["disclaimer"],
            metadata={"risk_level": "low"},
        )
        assert len(pack.allowed_tools) == 2
        assert pack.metadata["risk_level"] == "low"
