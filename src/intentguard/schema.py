"""Request and response models for the IntentGuard API."""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# -- Classification types --


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"


class ClassifyRequest(BaseModel):
    """Request body for /v1/classify."""

    messages: list[Message]
    vertical: str | None = None  # override policy vertical if needed

    def last_user_message(self) -> str | None:
        """Extract the last user message from the conversation."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None


class PolicyPackResponse(BaseModel):
    """Policy pack attached to a classification response."""

    vertical: str
    decision: str
    allowed_tools: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class ClassifyResponse(BaseModel):
    """Response body for /v1/classify."""

    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    vertical: str
    message: str = ""  # refusal or clarification text (empty on allow)
    probabilities: dict[str, float] | None = None  # raw class probabilities (debug)
    routed_vertical: str | None = None  # set when router is active
    router_scores: dict[str, float] | None = None  # debug: per-vertical confidence
    policy_pack: PolicyPackResponse | None = None  # optional policy pack


# -- OpenAI-compatible types --


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible request for /v1/chat/completions."""

    model: str = "intentguard"
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False

    def last_user_message(self) -> str | None:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible response for /v1/chat/completions."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "intentguard"
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)
    classification: ClassifyResponse | None = None  # extension field


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "perfecxion"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class FeedbackRequest(BaseModel):
    """Request body for /v1/feedback."""

    query: str
    expected_decision: Decision
    actual_decision: Decision
    notes: str = ""


class FeedbackResponse(BaseModel):
    status: str = "recorded"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    policy_loaded: bool
    vertical: str
    version: str
    verticals: list[str] | None = None  # multi-vertical mode
