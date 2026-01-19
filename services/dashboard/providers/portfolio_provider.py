"""Portfolio data provider - connects holdings to portfolio_summary component."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import streamlit as st

from services.dashboard.components.portfolio_summary import (
    PortfolioPosition,
    PortfolioSummary,
)
from services.dashboard.utils.holdings import get_holdings_upbit, is_private_api_enabled
from services.dashboard.utils.price_refresh import (
    get_cached_prices,
    is_cache_stale,
    update_price_cache,
)

logger = logging.getLogger(__name__)


def _get_current_prices(symbols: List[str]) -> Dict[str, float]:
    """Get current prices for symbols, with caching.

    Args:
        symbols: List of symbols in dashboard format (e.g., "BTC/KRW")

    Returns:
        Dict mapping symbol to current price
    """
    # Check cache first
    if not is_cache_stale():
        cache = get_cached_prices()
        if cache and cache.prices:
            return cache.prices

    # Fetch fresh prices
    prices: Dict[str, float] = {}

    try:
        from libs.adapters.upbit_spot import UpbitSpotMarketData

        market_data = UpbitSpotMarketData()

        for symbol in symbols:
            try:
                quote = market_data.get_quote(symbol)
                if quote and "price" in quote:
                    prices[symbol] = float(quote["price"])
            except Exception as e:
                logger.debug("Failed to get price for %s: %s", symbol, e)

        # Update cache if we got any prices
        if prices:
            update_price_cache(prices)

    except ImportError:
        logger.warning("UpbitSpotMarketData not available")
    except Exception as e:
        logger.warning("Failed to fetch prices: %s", type(e).__name__)

    return prices


def _parse_holdings_to_positions(
    holdings: List[Dict],
    prices: Dict[str, float],
) -> List[PortfolioPosition]:
    """Convert raw holdings to PortfolioPosition objects.

    Args:
        holdings: Raw holdings from Upbit API
        prices: Current prices by symbol

    Returns:
        List of PortfolioPosition objects
    """
    positions: List[PortfolioPosition] = []

    for entry in holdings:
        currency = entry.get("currency", "")

        # Skip KRW (handled as cash)
        if not currency or currency == "KRW":
            continue

        try:
            balance = float(entry.get("balance", 0))
            avg_buy_price = float(entry.get("avg_buy_price", 0))
        except (ValueError, TypeError):
            continue

        # Skip zero balances
        if balance <= 0:
            continue

        # Convert to dashboard symbol format
        symbol = f"{currency}/KRW"
        current_price = prices.get(symbol, avg_buy_price)

        positions.append(
            PortfolioPosition(
                symbol=symbol,
                exchange="upbit",
                quantity=balance,
                avg_price=avg_buy_price,
                current_price=current_price,
            )
        )

    return positions


def _get_cash_balance(holdings: List[Dict]) -> float:
    """Extract KRW cash balance from holdings.

    Args:
        holdings: Raw holdings from Upbit API

    Returns:
        KRW cash balance
    """
    for entry in holdings:
        if entry.get("currency") == "KRW":
            try:
                return float(entry.get("balance", 0))
            except (ValueError, TypeError):
                return 0.0
    return 0.0


def get_portfolio_summary() -> Optional[PortfolioSummary]:
    """Get portfolio summary from real holdings data.

    Returns:
        PortfolioSummary if data available, None otherwise (triggers demo mode)
    """
    # Check if private API is enabled
    if not is_private_api_enabled():
        logger.debug("Private API not enabled, returning None for demo mode")
        return None

    # Get holdings
    holdings = get_holdings_upbit()
    if not holdings:
        logger.debug("No holdings data, returning None for demo mode")
        return None

    # Extract cash balance
    cash_balance = _get_cash_balance(holdings)

    # Get symbols for price lookup
    symbols = [
        f"{entry.get('currency')}/KRW"
        for entry in holdings
        if entry.get("currency") and entry.get("currency") != "KRW"
    ]

    # Get current prices
    prices = _get_current_prices(symbols) if symbols else {}

    # Convert to positions
    positions = _parse_holdings_to_positions(holdings, prices)

    # Calculate totals
    total_cost = sum(p.cost for p in positions)
    total_value = sum(p.value for p in positions)

    return PortfolioSummary(
        total_cost=total_cost,
        total_value=total_value,
        cash_balance=cash_balance,
        positions=positions,
    )
