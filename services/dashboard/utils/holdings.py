"""Holdings management for MASP Dashboard (READ-ONLY)."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import streamlit as st

from services.dashboard.utils.symbols import upbit_to_dashboard

logger = logging.getLogger(__name__)


def is_private_api_enabled() -> bool:
    """Check if private API (balance query) is enabled."""
    return os.getenv("MASP_ENABLE_LIVE_TRADING", "").strip() == "1"


@st.cache_data(ttl=60, show_spinner=False)
def get_holdings_upbit() -> List[Dict[str, Any]]:
    """Get Upbit holdings (cached 60s).

    Requires: MASP_ENABLE_LIVE_TRADING=1
    """
    if not is_private_api_enabled():
        logger.debug("Private API not enabled")
        return []

    try:
        from libs.adapters.upbit_spot import UpbitSpotExecution

        adapter = UpbitSpotExecution()
        return adapter.get_all_balances()
    except Exception as exc:
        logger.warning("Failed to get holdings: %s", type(exc).__name__)
        return []


def get_holding_symbols() -> List[str]:
    """Get list of held symbols in Dashboard format (BTC/KRW)."""
    holdings = get_holdings_upbit()
    symbols: List[str] = []

    for entry in holdings:
        currency = entry.get("currency", "")
        try:
            balance = float(entry.get("balance", 0))
        except (ValueError, TypeError):
            balance = 0.0

        if currency and currency != "KRW" and balance > 0:
            symbols.append(upbit_to_dashboard(currency))

    return symbols


def clear_holdings_cache() -> None:
    """Clear holdings cache for refresh."""
    get_holdings_upbit.clear()
