"""Tests for policy loading and validation."""

import json
import tempfile
from pathlib import Path

import pytest

from intentguard.policy import Policy


class TestPolicyLoading:
    def test_load_finance_policy(self, finance_policy):
        assert finance_policy.vertical == "finance"
        assert finance_policy.version == "1.0"
        assert finance_policy.display_name == "Financial Services"

    def test_core_topics_populated(self, finance_policy):
        ctx = finance_policy.vertical_context()
        assert "banking" in ctx
        assert "investing" in ctx

    def test_thresholds_loaded(self, finance_policy):
        t = finance_policy.thresholds
        assert t.tau_allow == 0.80
        assert t.tau_deny == 0.90
        assert t.margin_allow == 0.10
        assert t.margin_deny == 0.10

    def test_deny_response_has_content(self, finance_policy):
        msg = finance_policy.deny_response()
        assert "financial" in msg.lower()
        assert len(msg) > 20

    def test_abstain_response_has_content(self, finance_policy):
        msg = finance_policy.abstain_response()
        assert "clarify" in msg.lower() or "help" in msg.lower()

    def test_vertical_context_format(self, finance_policy):
        ctx = finance_policy.vertical_context()
        assert "VERTICAL=finance" in ctx
        assert "CORE_TOPICS=" in ctx
        assert "HARD_EXCLUSIONS=" in ctx

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            Policy.from_file("/nonexistent/policy.json")

    def test_invalid_json_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            f.flush()
            with pytest.raises(ValueError, match="Invalid JSON"):
                Policy.from_file(f.name)
        Path(f.name).unlink()

    def test_missing_required_fields_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"vertical": "test"}, f)
            f.flush()
            with pytest.raises(Exception):  # pydantic ValidationError
                Policy.from_file(f.name)
        Path(f.name).unlink()

    def test_invalid_threshold_raises(self):
        with pytest.raises(Exception):
            Policy.from_dict({
                "vertical": "test",
                "version": "1.0",
                "display_name": "Test",
                "scope": {"core_topics": ["test"]},
                "responses": {
                    "deny_message": "no",
                    "abstain_message": "maybe",
                },
                "decision": {"tau_allow": 1.5},  # out of range
            })

    def test_from_dict(self):
        policy = Policy.from_dict({
            "vertical": "test",
            "version": "1.0",
            "display_name": "Test Vertical",
            "scope": {
                "core_topics": ["testing", "qa"],
            },
            "responses": {
                "deny_message": "Not a test question.",
                "abstain_message": "Is this about testing?",
            },
        })
        assert policy.vertical == "test"
        assert "testing" in policy.vertical_context()
