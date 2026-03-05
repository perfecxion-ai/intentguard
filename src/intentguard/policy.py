"""Policy configuration loader and response builder."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ConditionalAllow(BaseModel):
    topic: str
    condition: str
    examples_allow: list[str] = Field(default_factory=list)
    examples_deny: list[str] = Field(default_factory=list)
    disambiguation_questions: list[str] = Field(default_factory=list)


class Scope(BaseModel):
    core_topics: list[str]
    conditional_allow: list[ConditionalAllow] = Field(default_factory=list)
    hard_exclusions: list[str] = Field(default_factory=list)


class Responses(BaseModel):
    deny_message: str
    deny_examples: list[str] = Field(default_factory=list)
    abstain_message: str
    abstain_followup_template: str = ""


class DecisionConfig(BaseModel):
    tau_allow: float = 0.80
    tau_deny: float = 0.90
    margin_allow: float = 0.10
    margin_deny: float = 0.10
    context_window: int = 1

    @field_validator("tau_allow", "tau_deny", "margin_allow", "margin_deny")
    @classmethod
    def validate_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Must be between 0 and 1, got {v}")
        return v


class LabelingRules(BaseModel):
    allow_definition: str = ""
    abstain_definition: str = ""
    deny_definition: str = ""


class PrivacyConfig(BaseModel):
    log_query_text_default: bool = False
    pii_redaction_default: bool = True
    log_sampling_rate: float = 0.1


class PolicyPack(BaseModel):
    """Tools and guardrails associated with a classification decision."""

    allowed_tools: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class PolicySpec(BaseModel):
    vertical: str
    version: str
    display_name: str
    labeling_rules: LabelingRules = Field(default_factory=LabelingRules)
    scope: Scope
    responses: Responses
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    context_template_version: str = "ctv1"
    policy_packs: dict[str, PolicyPack] = Field(default_factory=dict)


class Policy:
    """Loads and provides access to a vertical policy configuration."""

    def __init__(self, spec: PolicySpec):
        self.spec = spec
        self._vertical_context = self._build_vertical_context()

    @classmethod
    def from_file(cls, path: str | Path) -> Policy:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")

        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in policy file {path}: {e}") from e

        spec = PolicySpec(**raw)
        return cls(spec)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        spec = PolicySpec(**data)
        return cls(spec)

    @property
    def vertical(self) -> str:
        return self.spec.vertical

    @property
    def version(self) -> str:
        return self.spec.version

    @property
    def display_name(self) -> str:
        return self.spec.display_name

    @property
    def thresholds(self) -> DecisionConfig:
        return self.spec.decision

    def vertical_context(self) -> str:
        """Return the context string used as model input alongside the query."""
        return self._vertical_context

    def deny_response(self) -> str:
        """Build the deny message shown to users."""
        resp = self.spec.responses
        parts = [resp.deny_message]
        if resp.deny_examples:
            examples = random.sample(resp.deny_examples, min(3, len(resp.deny_examples)))
            parts.append("Try asking about: " + ", ".join(examples))
        return " ".join(parts)

    def abstain_response(self) -> str:
        """Build the abstain/clarification message shown to users."""
        resp = self.spec.responses
        msg = resp.abstain_message
        if resp.abstain_followup_template:
            sample = ", ".join(self.spec.scope.core_topics[:4])
            followup = resp.abstain_followup_template.replace("{core_topics_sample}", sample)
            msg = f"{msg} {followup}"
        return msg

    def get_policy_pack(self, decision: str) -> PolicyPack | None:
        """Look up the policy pack for a given decision."""
        return self.spec.policy_packs.get(decision)

    def _build_vertical_context(self) -> str:
        """Build the context segment appended to model input.

        Format: compact description of the vertical's scope that the model
        uses to determine if a query is on-topic.
        """
        scope = self.spec.scope
        parts = [
            f"VERTICAL={self.spec.vertical}",
            f"CONTEXT_VERSION={self.spec.context_template_version}",
            f"CORE_TOPICS=[{', '.join(scope.core_topics)}]",
        ]

        if scope.conditional_allow:
            conditions = [
                f"{ca.topic}: {ca.condition}" for ca in scope.conditional_allow
            ]
            parts.append(f"CONDITIONAL_ALLOW=[{'; '.join(conditions)}]")

        if scope.hard_exclusions:
            parts.append(f"HARD_EXCLUSIONS=[{', '.join(scope.hard_exclusions)}]")

        return "; ".join(parts)
