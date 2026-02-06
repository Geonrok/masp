"""Dashboard constants and shared values."""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

# =============================================================================
# Version
# =============================================================================

VERSION = "5.2.0"


def get_version() -> str:
    """Get application version, trying pyproject.toml first."""
    try:
        from importlib.metadata import version

        return version("multi-asset-strategy-platform")
    except Exception:
        return VERSION


# =============================================================================
# Supported Exchanges
# =============================================================================

SUPPORTED_EXCHANGES: List[str] = [
    "upbit",
    "bithumb",
    "binance",
    "binance_futures",
    "kiwoom",
]
SIGNAL_PREVIEW_EXCHANGES: List[str] = [
    "upbit",
    "bithumb",
    "binance",
    "binance_futures",
    "kiwoom",
]


# =============================================================================
# Default Values
# =============================================================================

DEFAULT_POSITION_SIZE_KRW = 10000
DEFAULT_AUTO_REFRESH_INTERVAL = 10.0
DEFAULT_TIMEZONE = "Asia/Seoul"


# =============================================================================
# Fee Rates (per exchange)
# =============================================================================

FEE_RATES = {
    "upbit": 0.0005,  # 0.05%
    "bithumb": 0.0004,  # 0.04%
    "binance": 0.001,  # 0.1% (maker)
    "binance_futures": 0.0004,  # 0.04% (maker)
}


def get_fee_rate(exchange: str) -> float:
    """Get fee rate for exchange."""
    return FEE_RATES.get(exchange.lower(), 0.0005)


# =============================================================================
# Signal Preview Limits
# =============================================================================

MAX_SYMBOLS_LIVE = 20
MAX_SYMBOLS_DEMO = 50
