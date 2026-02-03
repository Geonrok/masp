"""API configuration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
        logger.warning("[Config] CORS set to '*' - INSECURE FOR PRODUCTION!")
        return ["*"]
    origins = [part.strip() for part in value.split(",") if part.strip()]
    logger.info("[Config] CORS custom origins: %s", origins)
    return origins


def _validate_ssl_file(path: str | None) -> Optional[str]:
    """Validate SSL certificate/key file path."""
    if not path:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(f"[Config] SSL file not found: {path}")
        return None
    if not path_obj.is_file():
        logger.warning(f"[Config] SSL path is not a file: {path}")
        return None
    return str(path_obj.resolve())


@dataclass(frozen=True)
class APIConfig:
    """API server configuration."""

    host: str
    port: int
    debug: bool
    cors_origins: List[str]

    # SSL/TLS configuration
    ssl_enabled: bool = False
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_keyfile_password: Optional[str] = None

    @property
    def is_https(self) -> bool:
        """Check if HTTPS is properly configured."""
        return (
            self.ssl_enabled
            and self.ssl_certfile is not None
            and self.ssl_keyfile is not None
        )

    @property
    def protocol(self) -> str:
        """Get protocol string (http or https)."""
        return "https" if self.is_https else "http"

    @property
    def base_url(self) -> str:
        """Get base URL for the API server."""
        port_str = f":{self.port}" if self.port not in (80, 443) else ""
        return f"{self.protocol}://{self.host}{port_str}"


def load_config() -> APIConfig:
    """Load API configuration from environment variables.

    Environment Variables:
        API_HOST: Server host (default: 0.0.0.0)
        API_PORT: Server port (default: 8000)
        API_DEBUG: Debug mode (default: false)
        API_CORS_ORIGINS: Comma-separated CORS origins

        SSL/TLS Configuration:
        API_SSL_ENABLED: Enable HTTPS (default: false)
        API_SSL_CERTFILE: Path to SSL certificate file (.pem)
        API_SSL_KEYFILE: Path to SSL private key file (.pem)
        API_SSL_KEYFILE_PASSWORD: Password for encrypted private key (optional)
    """
    host = os.getenv("API_HOST", "0.0.0.0")
    port_raw = os.getenv("API_PORT", "8000")
    try:
        port = int(port_raw)
    except ValueError:
        port = 8000

    debug = _parse_bool(os.getenv("API_DEBUG"))
    cors_origins = _parse_origins(os.getenv("API_CORS_ORIGINS"))

    # SSL/TLS configuration
    ssl_enabled = _parse_bool(os.getenv("API_SSL_ENABLED"))
    ssl_certfile = _validate_ssl_file(os.getenv("API_SSL_CERTFILE"))
    ssl_keyfile = _validate_ssl_file(os.getenv("API_SSL_KEYFILE"))
    ssl_keyfile_password = os.getenv("API_SSL_KEYFILE_PASSWORD")

    # Validate SSL configuration
    if ssl_enabled:
        if not ssl_certfile or not ssl_keyfile:
            logger.warning(
                "[Config] SSL enabled but certificate/key files not found. "
                "Set API_SSL_CERTFILE and API_SSL_KEYFILE environment variables."
            )
            ssl_enabled = False
        else:
            logger.info("[Config] HTTPS enabled with certificate: %s", ssl_certfile)

    return APIConfig(
        host=host,
        port=port,
        debug=debug,
        cors_origins=cors_origins,
        ssl_enabled=ssl_enabled,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_keyfile_password=ssl_keyfile_password,
    )


api_config = load_config()
