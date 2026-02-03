"""Upbit symbol list utilities with caching."""

from __future__ import annotations

import logging
from typing import List

import streamlit as st

from services.dashboard.utils.symbols import convert_symbols_to_dashboard

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300, show_spinner=False)
def get_all_upbit_symbols() -> List[str]:
    """Get all Upbit KRW market symbols (cached 5min).

    Returns:
        List of symbols in Dashboard format (BTC/KRW)
    """
    try:
        from libs.adapters.upbit_public import get_all_krw_symbols

        raw_symbols = get_all_krw_symbols()
        return convert_symbols_to_dashboard(raw_symbols)
    except Exception as exc:
        logger.warning("Failed to get Upbit symbols: %s", type(exc).__name__)
        return []


def get_symbol_count() -> int:
    """Get total symbol count."""
    return len(get_all_upbit_symbols())
