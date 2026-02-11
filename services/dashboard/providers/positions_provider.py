"""Positions provider - connects holdings to positions_panel component."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import streamlit as st

from services.dashboard.utils.holdings import (
    get_holdings_binance,
    get_holdings_bithumb,
    get_holdings_upbit,
    is_private_api_enabled,
)

logger = logging.getLogger(__name__)


def _get_current_prices_upbit(symbols: List[str]) -> Dict[str, float]:
    """Get current prices from Upbit.

    Args:
        symbols: List of currency codes (e.g., ["BTC", "ETH"])

    Returns:
        Dict mapping currency code to current price
    """
    if not symbols:
        return {}

    prices: Dict[str, float] = {}

    try:
        from libs.adapters.upbit_spot import UpbitSpotMarketData

        market_data = UpbitSpotMarketData()
        full_symbols = [f"{s}/KRW" for s in symbols]
        quotes = market_data.get_quotes(full_symbols)

        for full_symbol, quote in quotes.items():
            if quote and "price" in quote:
                currency = full_symbol.split("/")[0]
                prices[currency] = float(quote["price"])

    except ImportError:
        logger.debug("UpbitSpotMarketData not available")
    except Exception as e:
        logger.debug("Failed to fetch Upbit prices: %s", e)

    return prices


def _get_current_prices_bithumb(symbols: List[str]) -> Dict[str, float]:
    """Get current prices from Bithumb.

    Args:
        symbols: List of currency codes (e.g., ["BTC", "ETH"])

    Returns:
        Dict mapping currency code to current price
    """
    if not symbols:
        return {}

    prices: Dict[str, float] = {}

    try:
        from libs.adapters.real_bithumb_execution import BithumbExecution

        adapter = BithumbExecution()
        for symbol in symbols:
            try:
                price = adapter.get_current_price(f"{symbol}/KRW")
                if price is not None:
                    prices[symbol] = float(price)
            except Exception as e:
                logger.debug("Failed to get Bithumb price for %s: %s", symbol, e)

    except ImportError:
        logger.debug("BithumbExecution not available")
    except Exception as e:
        logger.debug("Failed to fetch Bithumb prices: %s", e)

    return prices


def _get_current_prices_binance(symbols: List[str]) -> Dict[str, float]:
    """Get current prices from Binance Spot.

    Args:
        symbols: List of currency codes (e.g., ["BTC", "ETH"])

    Returns:
        Dict mapping currency code to current price
    """
    if not symbols:
        return {}

    prices: Dict[str, float] = {}

    try:
        from libs.adapters.real_binance_spot import BinanceSpotMarketData

        market_data = BinanceSpotMarketData()
        full_symbols = [f"{s}/USDT" for s in symbols]
        quotes = market_data.get_quotes(full_symbols)

        for full_symbol, quote in quotes.items():
            if quote and quote.last is not None:
                currency = full_symbol.split("/")[0]
                prices[currency] = float(quote.last)

    except ImportError:
        logger.debug("BinanceSpotMarketData not available")
    except Exception as e:
        logger.debug("Failed to fetch Binance prices: %s", e)

    return prices


def _convert_holdings_to_positions(
    holdings: List[Dict],
    exchange: str,
    quote_currency: str,
) -> Tuple[List[Dict], List[str]]:
    """Convert raw holdings to position dicts.

    Args:
        holdings: Raw holdings from exchange API
        exchange: Exchange name
        quote_currency: Quote currency (KRW or USDT)

    Returns:
        Tuple of (positions, currency_codes)
    """
    positions: List[Dict] = []
    symbols: List[str] = []

    for entry in holdings:
        currency = entry.get("currency", "")

        if not currency or currency in ("KRW", "USDT"):
            continue

        try:
            balance = float(entry.get("balance", 0))
            avg_buy_price_raw = entry.get("avg_buy_price")
            avg_price = (
                float(avg_buy_price_raw) if avg_buy_price_raw is not None else 0.0
            )
        except (ValueError, TypeError):
            continue

        if balance <= 0:
            continue

        positions.append(
            {
                "symbol": currency,
                "quantity": balance,
                "avg_price": avg_price,
                "exchange": exchange,
                "quote_currency": quote_currency,
            }
        )
        symbols.append(currency)

    return positions, symbols


@st.cache_data(ttl=5, show_spinner=False)
def get_positions_data() -> Optional[Tuple[List[Dict], Dict[str, float]]]:
    """Get positions and current prices for positions_panel.

    Aggregates holdings from Upbit, Bithumb, and Binance Spot.
    Cached for 5 seconds to reduce API calls.

    Returns:
        Tuple of (positions_list, current_prices_dict) or None for demo mode
    """
    if not is_private_api_enabled():
        return None

    all_positions: List[Dict] = []
    current_prices: Dict[str, float] = {}

    # Upbit
    upbit_holdings = get_holdings_upbit()
    if upbit_holdings:
        positions, symbols = _convert_holdings_to_positions(
            upbit_holdings, "upbit", "KRW"
        )
        if symbols:
            prices = _get_current_prices_upbit(symbols)
            current_prices.update(prices)
        all_positions.extend(positions)

    # Bithumb
    bithumb_holdings = get_holdings_bithumb()
    if bithumb_holdings:
        positions, symbols = _convert_holdings_to_positions(
            bithumb_holdings, "bithumb", "KRW"
        )
        if symbols:
            prices = _get_current_prices_bithumb(symbols)
            current_prices.update(prices)
        all_positions.extend(positions)

    # Binance Spot
    binance_holdings = get_holdings_binance()
    if binance_holdings:
        positions, symbols = _convert_holdings_to_positions(
            binance_holdings, "binance", "USDT"
        )
        if symbols:
            prices = _get_current_prices_binance(symbols)
            current_prices.update(prices)
        all_positions.extend(positions)

    if not all_positions:
        return None

    # Fill missing prices with avg_price
    for pos in all_positions:
        symbol = pos["symbol"]
        if symbol not in current_prices:
            current_prices[symbol] = pos["avg_price"]

    return all_positions, current_prices
