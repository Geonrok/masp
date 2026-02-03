"""
Upbit Public API client for symbol discovery (no auth).
MASP Phase 3B - KRW market symbol loader.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import List

import requests

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_all_krw_symbols(timeout_sec: int = 10) -> List[str]:
    """
    Fetch all KRW market symbols.

    Returns:
        List[str]: ["BTC/KRW", "ETH/KRW", ...] (MASP format)

    Notes:
        - Upbit API format: "KRW-BTC" -> "BTC/KRW"
        - Retries 3 times with fallback
        - Cached per process
    """
    url = "https://api.upbit.com/v1/market/all"

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=timeout_sec)
            response.raise_for_status()

            markets = response.json()
            symbols: List[str] = []

            for market in markets:
                code = market.get("market", "")
                if not isinstance(code, str) or not code.startswith("KRW-"):
                    continue
                quote, base = code.split("-", 1)
                symbols.append(f"{base}/{quote}")

            logger.info("[UpbitPublic] Loaded %d KRW symbols", len(symbols))
            return sorted(symbols)

        except requests.exceptions.RequestException as exc:
            logger.warning("[UpbitPublic] Attempt %d/3 failed: %s", attempt + 1, exc)
            if attempt < 2:
                time.sleep(1)
            else:
                logger.error("[UpbitPublic] Failed to fetch symbols, using fallback")
                return ["BTC/KRW", "ETH/KRW", "XRP/KRW"]

    return ["BTC/KRW"]


def clear_symbol_cache() -> None:
    """Clear cached symbols (tests)."""
    get_all_krw_symbols.cache_clear()
