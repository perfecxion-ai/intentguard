"""FastAPI server with OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from enum import StrEnum

import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from intentguard import metrics
from intentguard.classifier import BaseClassifier, ONNXClassifier, StubClassifier
from intentguard.config import Settings, load_settings
from intentguard.policy import Policy
from intentguard.schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ClassifyRequest,
    ClassifyResponse,
    Decision,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    Message,
    ModelInfo,
    ModelList,
)

logger = logging.getLogger(__name__)

_classifier: BaseClassifier | None = None
_policy: Policy | None = None
_settings: Settings | None = None


class ClassifyMode(StrEnum):
    ENFORCE = "enforce"
    SHADOW = "shadow"


def _load_classifier(settings: Settings, policy: Policy) -> BaseClassifier:
    if settings.model_path.exists() and settings.tokenizer_path.exists():
        logger.info("Loading ONNX model from %s", settings.model_path)
        return ONNXClassifier(
            policy=policy,
            model_path=settings.model_path,
            tokenizer_path=settings.tokenizer_path,
            calibration_path=settings.calibration_path,
            intra_op_threads=settings.intra_op_threads,
            inter_op_threads=settings.inter_op_threads,
        )
    logger.info("No model files found, using stub classifier")
    return StubClassifier(policy)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier, _policy, _settings

    _settings = load_settings()
    logger.info("Loading policy from %s", _settings.policy_path)

    try:
        _policy = Policy.from_file(_settings.policy_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load policy: %s", e)
        raise SystemExit(1) from e

    logger.info(
        "Policy loaded: %s v%s (%s)",
        _policy.vertical, _policy.version, _policy.display_name,
    )

    _classifier = _load_classifier(_settings, _policy)
    model_loaded = not isinstance(_classifier, StubClassifier)
    metrics.set_model_loaded(model_loaded)
    logger.info("IntentGuard ready on port %s", _settings.port)

    yield
    logger.info("Shutting down")


app = FastAPI(title="IntentGuard", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/classify", response_model=ClassifyResponse)
async def classify(
    req: ClassifyRequest,
    response: Response,
    mode: ClassifyMode = Query(default=ClassifyMode.ENFORCE),
):
    """Classify a message. Use mode=shadow to classify without enforcing."""
    text = req.last_user_message()
    if text is None:
        raise HTTPException(status_code=400, detail="No user message found in request")

    start = time.perf_counter()
    result = _classifier.classify(text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if not _settings.debug:
        result.probabilities = None

    # Record metrics
    metrics.record_classification(result.decision.value, result.vertical, elapsed_ms / 1000)

    response.headers["X-Classification-Decision"] = result.decision.value
    response.headers["X-Classification-Latency-Ms"] = f"{elapsed_ms:.1f}"

    # Shadow mode: return real classification data but override decision to allow
    if mode == ClassifyMode.SHADOW:
        response.headers["X-Classification-Shadow"] = result.decision.value
        result = ClassifyResponse(
            decision=Decision.ALLOW,
            confidence=result.confidence,
            vertical=result.vertical,
            message="",
            probabilities=result.probabilities,
        )
        response.headers["X-Classification-Decision"] = "allow"

    _log_classification(text, result, elapsed_ms, shadow=(mode == ClassifyMode.SHADOW))
    return result


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest, response: Response):
    """OpenAI-compatible endpoint. Proxy mode if DOWNSTREAM_URL is set."""
    text = req.last_user_message()
    if text is None:
        raise HTTPException(status_code=400, detail="No user message found in request")

    start = time.perf_counter()
    result = _classifier.classify(text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    metrics.record_classification(result.decision.value, result.vertical, elapsed_ms / 1000)
    response.headers["X-Classification-Decision"] = result.decision.value
    response.headers["X-Classification-Latency-Ms"] = f"{elapsed_ms:.1f}"
    _log_classification(text, result, elapsed_ms)

    if result.decision == Decision.ALLOW and _settings.downstream_url:
        return await _proxy_downstream(req, result)

    content = result.message if result.decision != Decision.ALLOW else ""
    if result.decision == Decision.ALLOW and not _settings.downstream_url:
        content = (
            "Request allowed. No downstream LLM configured "
            "— set DOWNSTREAM_URL to enable proxy mode."
        )

    classification = result if _settings.debug else None

    return ChatCompletionResponse(
        model=req.model,
        choices=[Choice(message=Message(role="assistant", content=content))],
        classification=classification,
    )


@app.post("/v1/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    """Record feedback on a classification decision."""
    metrics.record_feedback(req.expected_decision.value, req.actual_decision.value)

    entry = {
        "type": "feedback",
        "query": req.query[:200],
        "expected": req.expected_decision.value,
        "actual": req.actual_decision.value,
        "notes": req.notes[:500] if req.notes else "",
        "vertical": _policy.vertical if _policy else "",
    }
    logger.info(json.dumps(entry))

    # Append to feedback file if configured
    if _settings and _settings.log_query_text:
        _write_feedback(entry)

    return FeedbackResponse()


@app.get("/health", response_model=HealthResponse)
async def health():
    model_loaded = _classifier is not None and not isinstance(_classifier, StubClassifier)
    return HealthResponse(
        status="ok" if _classifier else "error",
        model_loaded=model_loaded,
        policy_loaded=_policy is not None,
        vertical=_policy.vertical if _policy else "",
        version=_policy.version if _policy else "",
    )


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    vertical = _policy.vertical if _policy else "unknown"
    return ModelList(
        data=[ModelInfo(id=f"intentguard-{vertical}", owned_by="perfecxion")]
    )


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    if not metrics.is_enabled():
        return PlainTextResponse(
            "# prometheus_client not installed\n",
            media_type="text/plain",
        )
    return PlainTextResponse(metrics.get_metrics(), media_type="text/plain; charset=utf-8")


async def _proxy_downstream(
    req: ChatCompletionRequest,
    classification: ClassifyResponse,
) -> ChatCompletionResponse:
    headers = {"Content-Type": "application/json"}
    if _settings.downstream_api_key:
        headers["Authorization"] = f"Bearer {_settings.downstream_api_key}"

    body = req.model_dump(exclude_none=True)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(_settings.downstream_url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        logger.error("Downstream request failed: %s", e)
        raise HTTPException(status_code=502, detail="Downstream LLM request failed") from e

    downstream = ChatCompletionResponse(**data)
    if _settings.debug:
        downstream.classification = classification
    return downstream


def _log_classification(
    text: str, result: ClassifyResponse, elapsed_ms: float, shadow: bool = False,
):
    entry = {
        "decision": result.decision.value,
        "confidence": round(result.confidence, 4),
        "vertical": result.vertical,
        "latency_ms": round(elapsed_ms, 1),
    }
    if shadow:
        entry["mode"] = "shadow"
    if _settings and _settings.log_query_text:
        entry["query_text"] = text[:200]
    logger.info(json.dumps(entry))


def _write_feedback(entry: dict):
    """Append feedback to a JSONL file for batch export."""
    try:
        with open("feedback.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        logger.warning("Failed to write feedback: %s", e)
