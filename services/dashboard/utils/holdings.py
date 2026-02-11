"""Holdings management for MASP Dashboard (READ-ONLY)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import streamlit as st

logger = logging.getLogger(__name__)


def is_private_api_enabled() -> bool:
    """Check if private API (balance query) is enabled."""
    return os.getenv("MASP_ENABLE_LIVE_TRADING", "").strip() == "1"


@st.cache_data(ttl=5, show_spinner=False)
def get_holdings_upbit() -> List[Dict[str, Any]]:
    """Get Upbit holdings (cached 5s for real-time updates).

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


@st.cache_data(ttl=5, show_spinner=False)
def get_holdings_bithumb() -> List[Dict[str, Any]]:
    """Get Bithumb holdings (cached 5s for real-time updates).

    Requires: MASP_ENABLE_LIVE_TRADING=1

    Returns:
        List of dicts with currency, balance (avg_buy_price not available from API)
    """
    if not is_private_api_enabled():
        logger.debug("Private API not enabled")
        return []

    try:
        from libs.adapters.real_bithumb_execution import BithumbExecution

        adapter = BithumbExecution()
        balances = adapter.get_all_balances()  # Dict[str, float]

        # Convert to same format as Upbit for consistency
        holdings = []
        for currency, balance in balances.items():
            if balance > 0:
                holdings.append(
                    {
                        "currency": currency,
                        "balance": balance,
                        "avg_buy_price": None,  # Bithumb API doesn't provide this
                    }
                )
        return holdings
    except Exception as exc:
        logger.warning("Failed to get Bithumb holdings: %s", type(exc).__name__)
        return []


@st.cache_data(ttl=5, show_spinner=False)
def get_holdings_binance() -> List[Dict[str, Any]]:
    """Get Binance Spot holdings (cached 5s for real-time updates).

    Requires: MASP_ENABLE_LIVE_TRADING=1

    Returns:
        List of dicts with currency, balance (avg_buy_price not available from API)
    """
    if not is_private_api_enabled():
        logger.debug("Private API not enabled")
        return []

    try:
        from libs.adapters.real_binance_spot import BinanceSpotExecution

        adapter = BinanceSpotExecution()
        balances = adapter.get_all_balances()  # Dict[str, float]

        holdings = []
        for currency, balance in balances.items():
            if balance > 0:
                holdings.append(
                    {
                        "currency": currency,
                        "balance": balance,
                        "avg_buy_price": None,
                    }
                )
        return holdings
    except Exception as exc:
        logger.warning("Failed to get Binance holdings: %s", type(exc).__name__)
        return []


def get_holding_symbols() -> List[str]:
    """Get list of held symbols in Dashboard format (e.g. BTC/KRW, ETH/USDT)."""
    symbols: List[str] = []

    for getter, quote in [
        (get_holdings_upbit, "KRW"),
        (get_holdings_bithumb, "KRW"),
        (get_holdings_binance, "USDT"),
    ]:
        for entry in getter():
            currency = entry.get("currency", "")
            try:
                balance = float(entry.get("balance", 0) or 0)
            except (ValueError, TypeError):
                balance = 0.0

            if currency and currency != quote and balance > 0:
                symbols.append(f"{currency}/{quote}")

    return symbols


def clear_holdings_cache() -> None:
    """Clear holdings cache for refresh."""
    get_holdings_upbit.clear()
    get_holdings_bithumb.clear()
    get_holdings_binance.clear()
