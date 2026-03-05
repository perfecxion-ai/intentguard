"""Server configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    # Model and policy paths
    model_path: Path = field(default_factory=lambda: Path(os.getenv(
        "MODEL_PATH", "models/model.onnx"
    )))
    tokenizer_path: Path = field(default_factory=lambda: Path(os.getenv(
        "TOKENIZER_PATH", "models/tokenizer"
    )))
    calibration_path: Path = field(default_factory=lambda: Path(os.getenv(
        "CALIBRATION_PATH", "models/calibration_params.json"
    )))
    policy_path: Path = field(default_factory=lambda: Path(os.getenv(
        "POLICY_PATH", "policies/finance.json"
    )))

    # Server
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8080")))

    # Proxy mode
    downstream_url: str | None = field(
        default_factory=lambda: os.getenv("DOWNSTREAM_URL")
    )
    downstream_api_key: str | None = field(
        default_factory=lambda: os.getenv("DOWNSTREAM_API_KEY")
    )

    # ONNX Runtime threading
    intra_op_threads: int = field(
        default_factory=lambda: int(os.getenv("ORT_INTRA_OP_THREADS", "4"))
    )
    inter_op_threads: int = field(
        default_factory=lambda: int(os.getenv("ORT_INTER_OP_THREADS", "1"))
    )

    # Input limits
    max_input_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_INPUT_CHARS", "2000"))
    )

    # Logging
    log_query_text: bool = field(
        default_factory=lambda: os.getenv("LOG_QUERY_TEXT", "false").lower() == "true"
    )

    # Debug mode (exposes probabilities in responses)
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )

    # Multi-vertical router
    router_enabled: bool = field(
        default_factory=lambda: os.getenv("ROUTER_ENABLED", "false").lower() == "true"
    )
    router_config_path: Path = field(
        default_factory=lambda: Path(os.getenv("ROUTER_CONFIG_PATH", "router_config.json"))
    )


def load_settings() -> Settings:
    return Settings()
