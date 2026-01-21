"""Price refresh utilities with TTL cache."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import streamlit as st


@dataclass
class PriceCache:
    """Cached price data with timestamp."""
    prices: Dict[str, float]
    updated_at: float


_CACHE_KEY = "masp_price_cache"
_DEFAULT_TTL = 5  # seconds (match auto-refresh interval)


def get_cached_prices() -> Optional[PriceCache]:
    """Return cached prices if not expired."""
    cache = st.session_state.get(_CACHE_KEY)
    if not isinstance(cache, PriceCache):
        return None
    return cache


def is_cache_stale(ttl: int = _DEFAULT_TTL) -> bool:
    """Return True if cache is missing or expired."""
    cache = get_cached_prices()
    if cache is None:
        return True
    return (time.time() - cache.updated_at) > ttl


def update_price_cache(prices: Dict[str, float]) -> None:
    """Update price cache with new data."""
    st.session_state[_CACHE_KEY] = PriceCache(
        prices=prices,
        updated_at=time.time(),
    )


def clear_price_cache() -> None:
    """Clear price cache."""
    st.session_state.pop(_CACHE_KEY, None)
