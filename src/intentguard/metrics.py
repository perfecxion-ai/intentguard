"""Prometheus metrics for IntentGuard.

Optional — only active if prometheus_client is installed.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest

    REQUESTS = Counter(
        "intentguard_requests_total",
        "Total classification requests",
        ["decision", "vertical"],
    )
    LATENCY = Histogram(
        "intentguard_latency_seconds",
        "Classification latency",
        ["vertical"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    MODEL_LOADED = Gauge(
        "intentguard_model_loaded",
        "Whether a real model is loaded (1) or stub (0)",
    )
    FEEDBACK_COUNT = Counter(
        "intentguard_feedback_total",
        "Total feedback submissions",
        ["expected_decision", "actual_decision"],
    )
    _ENABLED = True
except ImportError:
    _ENABLED = False
    logger.info("prometheus_client not installed, metrics disabled")


def is_enabled() -> bool:
    return _ENABLED


def record_classification(decision: str, vertical: str, latency_s: float):
    if _ENABLED:
        REQUESTS.labels(decision=decision, vertical=vertical).inc()
        LATENCY.labels(vertical=vertical).observe(latency_s)


def record_feedback(expected: str, actual: str):
    if _ENABLED:
        FEEDBACK_COUNT.labels(expected_decision=expected, actual_decision=actual).inc()


def set_model_loaded(loaded: bool):
    if _ENABLED:
        MODEL_LOADED.set(1 if loaded else 0)


def get_metrics() -> bytes:
    if _ENABLED:
        return generate_latest()
    return b""


@contextmanager
def track_latency(vertical: str):
    """Context manager that records latency to both metrics and returns elapsed ms."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if _ENABLED:
        LATENCY.labels(vertical=vertical).observe(elapsed)
