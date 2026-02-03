"""Positions provider - connects holdings to positions_panel component."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)


def is_live_trading_enabled() -> bool:
    """Check if live trading is enabled."""
    return os.getenv("MASP_ENABLE_LIVE_TRADING", "0") == "1"


def _get_holdings_upbit() -> List[Dict]:
    """Get holdings from Upbit.

    Returns:
        List of holding dicts
    """
    try:
        from services.dashboard.utils.holdings import get_holdings_upbit

        return get_holdings_upbit() or []
    except ImportError:
        return []
    except Exception as e:
        logger.debug("Failed to get holdings: %s", e)
        return []


def _get_current_prices(symbols: List[str]) -> Dict[str, float]:
    """Get current prices for symbols using batch API.

    Args:
        symbols: List of symbols (e.g., ["BTC", "ETH"])

    Returns:
        Dict mapping symbol to current price
    """
    if not symbols:
        return {}

    prices: Dict[str, float] = {}

    try:
        from libs.adapters.upbit_spot import UpbitSpotMarketData

        market_data = UpbitSpotMarketData()

        # Convert to full symbols and use batch API
        full_symbols = [f"{symbol}/KRW" for symbol in symbols]
        quotes = market_data.get_quotes(full_symbols)

        for full_symbol, quote in quotes.items():
            if quote and "price" in quote:
                # Convert back to currency code (BTC/KRW -> BTC)
                currency = full_symbol.split("/")[0]
                prices[currency] = float(quote["price"])

    except ImportError:
        logger.debug("UpbitSpotMarketData not available")
    except Exception as e:
        logger.debug("Failed to fetch prices: %s", e)

    return prices


@st.cache_data(ttl=5, show_spinner=False)
def get_positions_data() -> Optional[Tuple[List[Dict], Dict[str, float]]]:
    """Get positions and current prices for positions_panel.

    Cached for 5 seconds to reduce API calls.

    Returns:
        Tuple of (positions_list, current_prices_dict) or None for demo mode
    """
    if not is_live_trading_enabled():
        return None

    holdings = _get_holdings_upbit()
    if not holdings:
        return None

    # Convert holdings to positions format
    positions: List[Dict] = []
    symbols: List[str] = []

    for entry in holdings:
        currency = entry.get("currency", "")

        # Skip KRW
        if not currency or currency == "KRW":
            continue

        try:
            balance = float(entry.get("balance", 0))
            avg_buy_price_raw = entry.get("avg_buy_price")
            avg_price = (
                float(avg_buy_price_raw) if avg_buy_price_raw is not None else 0.0
            )
        except (ValueError, TypeError):
            continue

        # Skip zero balances
        if balance <= 0:
            continue

        positions.append(
            {
                "symbol": currency,
                "quantity": balance,
                "avg_price": avg_price,
            }
        )
        symbols.append(currency)

    if not positions:
        return None

    # Get current prices
    current_prices = _get_current_prices(symbols)

    # Fill missing prices with avg_price
    for pos in positions:
        symbol = pos["symbol"]
        if symbol not in current_prices:
            current_prices[symbol] = pos["avg_price"]

    return positions, current_prices
