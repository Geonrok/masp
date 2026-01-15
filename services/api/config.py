"""API configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
import logging
from typing import List

logger = logging.getLogger(__name__)

def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_origins(value: str | None) -> List[str]:
    if not value:
        origins = [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ]
        logger.info("[Config] CORS default origins: %s", origins)
        return origins
    if value.strip() == "*":
        logger.warning("⚠️  [Config] CORS set to '*' - INSECURE FOR PRODUCTION!")
        return ["*"]
    origins = [part.strip() for part in value.split(",") if part.strip()]
    logger.info("[Config] CORS custom origins: %s", origins)
    return origins


@dataclass(frozen=True)
class APIConfig:
    host: str
    port: int
    debug: bool
    cors_origins: List[str]


def load_config() -> APIConfig:
    host = os.getenv("API_HOST", "0.0.0.0")
    port_raw = os.getenv("API_PORT", "8000")
    try:
        port = int(port_raw)
    except ValueError:
        port = 8000

    debug = _parse_bool(os.getenv("API_DEBUG"))
    cors_origins = _parse_origins(os.getenv("API_CORS_ORIGINS"))

    return APIConfig(
        host=host,
        port=port,
        debug=debug,
        cors_origins=cors_origins,
    )


api_config = load_config()
