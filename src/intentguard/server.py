"""FastAPI server with OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from intentguard.classifier import BaseClassifier, StubClassifier
from intentguard.config import Settings, load_settings
from intentguard.policy import Policy
from intentguard.schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ClassifyRequest,
    ClassifyResponse,
    Decision,
    HealthResponse,
    Message,
    ModelInfo,
    ModelList,
)

logger = logging.getLogger(__name__)

# Module-level state, set during lifespan
_classifier: BaseClassifier | None = None
_policy: Policy | None = None
_settings: Settings | None = None


def _load_classifier(settings: Settings, policy: Policy) -> BaseClassifier:
    """Load the appropriate classifier based on available model files."""
    if settings.model_path.exists():
        # ONNX model available — will be implemented when model is trained
        logger.info("ONNX model found at %s (not yet implemented, using stub)", settings.model_path)

    logger.info("Using stub classifier (no model loaded)")
    return StubClassifier(policy)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load policy and model on startup."""
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
        _policy.vertical,
        _policy.version,
        _policy.display_name,
    )

    _classifier = _load_classifier(_settings, _policy)
    logger.info("IntentGuard ready on port %s", _settings.port)

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="IntentGuard",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest, response: Response):
    """Classify a message and return the decision. Recommended for sidecar mode."""
    text = req.last_user_message()
    if text is None:
        raise HTTPException(status_code=400, detail="No user message found in request")

    start = time.perf_counter()
    result = _classifier.classify(text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Strip debug info unless debug mode is on
    if not _settings.debug:
        result.probabilities = None

    response.headers["X-Classification-Decision"] = result.decision.value
    response.headers["X-Classification-Latency-Ms"] = f"{elapsed_ms:.1f}"

    _log_classification(text, result, elapsed_ms)
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

    response.headers["X-Classification-Decision"] = result.decision.value
    response.headers["X-Classification-Latency-Ms"] = f"{elapsed_ms:.1f}"
    _log_classification(text, result, elapsed_ms)

    if result.decision == Decision.ALLOW and _settings.downstream_url:
        return await _proxy_downstream(req, result)

    # DENY, ABSTAIN, or no downstream — return the classifier's response
    content = result.message if result.decision != Decision.ALLOW else ""
    if result.decision == Decision.ALLOW and not _settings.downstream_url:
        content = "Request allowed. No downstream LLM configured — set DOWNSTREAM_URL to enable proxy mode."

    classification = result if _settings.debug else None
    if classification:
        classification.probabilities = None if not _settings.debug else classification.probabilities

    return ChatCompletionResponse(
        model=req.model,
        choices=[
            Choice(message=Message(role="assistant", content=content))
        ],
        classification=classification,
    )


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
        data=[
            ModelInfo(
                id=f"intentguard-{vertical}",
                owned_by="perfecxion",
            )
        ]
    )


async def _proxy_downstream(
    req: ChatCompletionRequest,
    classification: ClassifyResponse,
) -> ChatCompletionResponse:
    """Forward an ALLOW'd request to the downstream LLM."""
    headers = {"Content-Type": "application/json"}
    if _settings.downstream_api_key:
        headers["Authorization"] = f"Bearer {_settings.downstream_api_key}"

    body = req.model_dump(exclude_none=True)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                _settings.downstream_url,
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        logger.error("Downstream request failed: %s", e)
        raise HTTPException(status_code=502, detail="Downstream LLM request failed") from e

    # Parse downstream response and attach classification metadata
    downstream = ChatCompletionResponse(**data)
    if _settings.debug:
        downstream.classification = classification
    return downstream


def _log_classification(text: str, result: ClassifyResponse, elapsed_ms: float):
    """Log classification decision as structured JSON."""
    entry = {
        "decision": result.decision.value,
        "confidence": round(result.confidence, 4),
        "vertical": result.vertical,
        "latency_ms": round(elapsed_ms, 1),
    }
    if _settings and _settings.log_query_text:
        entry["query_text"] = text[:200]  # truncate for safety

    logger.info(json.dumps(entry))
