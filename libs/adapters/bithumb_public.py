"""Bithumb public market data - KRW symbol loader with fallback."""

from __future__ import annotations

import logging
import time
from typing import List

import requests

try:
    import pybithumb  # type: ignore
except Exception:
    pybithumb = None

logger = logging.getLogger(__name__)


class BithumbPublic:
    """
    Load all KRW trading pairs from Bithumb.
    Fallback: pybithumb -> HTTP API -> ["BTC/KRW"]
    Cache TTL: 1 hour
    """

    def __init__(self, cache_ttl: int = 3600) -> None:
        self.symbols: List[str] = []
        self._cache_ttl = cache_ttl
        self._last_update: float = 0
        self._load_symbols()

    def _load_symbols(self) -> None:
        """Fetch all KRW markets with 3-stage fallback."""
        if time.time() - self._last_update < self._cache_ttl and self.symbols:
            return

        symbols: List[str] = []

        if pybithumb is not None:
            try:
                tickers = pybithumb.get_tickers(payment_currency="KRW")
                if tickers:
                    symbols = [f"{t}/KRW" for t in tickers if t]
                    symbols = list(dict.fromkeys(symbols))
                    logger.info("[BithumbPublic] pybithumb: %d symbols", len(symbols))
            except Exception as exc:
                logger.warning("[BithumbPublic] pybithumb failed: %s", exc)

        if not symbols:
            try:
                resp = requests.get(
                    "https://api.bithumb.com/public/ticker/ALL_KRW",
                    timeout=10,
                )
                data = resp.json()
                if data.get("status") == "0000":
                    symbols = [
                        f"{k}/KRW"
                        for k in data.get("data", {}).keys()
                        if k not in ("date", "timestamp")
                    ]
                    symbols = list(dict.fromkeys(symbols))
                    logger.info("[BithumbPublic] HTTP API: %d symbols", len(symbols))
            except Exception as exc:
                logger.warning("[BithumbPublic] HTTP API failed: %s", exc)

        if not symbols:
            symbols = ["BTC/KRW"]
            logger.warning("[BithumbPublic] Using default: BTC/KRW")

        self.symbols = symbols
        self._last_update = time.time()

    def get_all_krw_symbols(self) -> List[str]:
        """Return all KRW trading pairs (refresh if cache expired)."""
        self._load_symbols()
        return self.symbols.copy()
