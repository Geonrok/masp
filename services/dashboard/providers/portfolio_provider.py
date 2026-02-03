"""Portfolio data provider - connects holdings to portfolio_summary component."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import streamlit as st

from services.dashboard.components.portfolio_summary import (
    PortfolioPosition,
    PortfolioSummary,
)
from services.dashboard.utils.holdings import (
    get_holdings_bithumb,
    get_holdings_upbit,
    is_private_api_enabled,
)

# Price caching removed for real-time updates
# Holdings cache (5s TTL) is sufficient

logger = logging.getLogger(__name__)


def _get_current_prices_upbit(symbols: List[str]) -> Dict[str, float]:
    """Get current prices from Upbit for symbols using batch API.

    Args:
        symbols: List of symbols in dashboard format (e.g., "BTC/KRW")

    Returns:
        Dict mapping symbol to current price
    """
    if not symbols:
        return {}

    prices: Dict[str, float] = {}

    try:
        from libs.adapters.upbit_spot import UpbitSpotMarketData

        market_data = UpbitSpotMarketData()

        # Use batch API to get all quotes at once (single API call)
        quotes = market_data.get_quotes(symbols)
        for symbol, quote in quotes.items():
            if quote and "price" in quote:
                prices[symbol] = float(quote["price"])

    except ImportError:
        logger.warning("UpbitSpotMarketData not available")
    except Exception as e:
        logger.warning("Failed to fetch Upbit prices: %s", type(e).__name__)

    return prices


def _get_current_prices_bithumb(symbols: List[str]) -> Dict[str, float]:
    """Get current prices from Bithumb for symbols.

    Args:
        symbols: List of symbols in dashboard format (e.g., "BTC/KRW")

    Returns:
        Dict mapping symbol to current price
    """
    prices: Dict[str, float] = {}

    try:
        from libs.adapters.real_bithumb_execution import BithumbExecution

        adapter = BithumbExecution()

        for symbol in symbols:
            try:
                price = adapter.get_current_price(symbol)
                if price is not None:
                    prices[symbol] = float(price)
            except Exception as e:
                logger.debug("Failed to get Bithumb price for %s: %s", symbol, e)

    except ImportError:
        logger.warning("BithumbExecution not available")
    except Exception as e:
        logger.warning("Failed to fetch Bithumb prices: %s", type(e).__name__)

    return prices


def _get_current_prices(
    symbols: List[str], exchange: str = "upbit"
) -> Dict[str, float]:
    """Get current prices for symbols.

    Args:
        symbols: List of symbols in dashboard format (e.g., "BTC/KRW")
        exchange: Exchange name ("upbit" or "bithumb")

    Returns:
        Dict mapping symbol to current price
    """
    # Always fetch fresh prices for real-time updates
    # Cache is handled at holdings level with 5s TTL
    if exchange.lower() == "bithumb":
        prices = _get_current_prices_bithumb(symbols)
    else:
        prices = _get_current_prices_upbit(symbols)

    return prices


def _parse_holdings_to_positions(
    holdings: List[Dict],
    prices: Dict[str, float],
    exchange: str = "upbit",
) -> List[PortfolioPosition]:
    """Convert raw holdings to PortfolioPosition objects.

    Args:
        holdings: Raw holdings from exchange API
        prices: Current prices by symbol
        exchange: Exchange name ("upbit" or "bithumb")

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
            # avg_buy_price might be None for Bithumb
            avg_buy_price_raw = entry.get("avg_buy_price")
        except (ValueError, TypeError):
            continue

        # Skip zero balances
        if balance <= 0:
            continue

        # Convert to dashboard symbol format
        symbol = f"{currency}/KRW"
        current_price = prices.get(symbol, 0)

        # Handle avg_buy_price: use current_price if not available (Bithumb)
        if avg_buy_price_raw is not None:
            try:
                avg_buy_price = float(avg_buy_price_raw)
            except (ValueError, TypeError):
                avg_buy_price = current_price
        else:
            # Bithumb doesn't provide avg_buy_price, use current_price (0% PnL)
            avg_buy_price = current_price

        # If no current price and no avg_price, skip
        if current_price <= 0 and avg_buy_price <= 0:
            continue

        # Fallback: if current_price is 0 but avg_price exists, use avg_price
        if current_price <= 0:
            current_price = avg_buy_price

        positions.append(
            PortfolioPosition(
                symbol=symbol,
                exchange=exchange,
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


@st.cache_data(ttl=5, show_spinner=False)
def _fetch_portfolio_summary_cached() -> Optional[dict]:
    """Internal cached function to fetch portfolio data.

    Returns raw dict to avoid Streamlit serialization issues with dataclasses.
    """
    # Check if private API is enabled
    if not is_private_api_enabled():
        logger.debug("Private API not enabled, returning None for demo mode")
        return None

    all_positions_data: List[dict] = []
    total_cash_balance = 0.0

    # ===== Upbit Holdings =====
    upbit_holdings = get_holdings_upbit()
    if upbit_holdings:
        total_cash_balance += _get_cash_balance(upbit_holdings)
        upbit_symbols = [
            f"{entry.get('currency')}/KRW"
            for entry in upbit_holdings
            if entry.get("currency") and entry.get("currency") != "KRW"
        ]
        upbit_prices = (
            _get_current_prices(upbit_symbols, "upbit") if upbit_symbols else {}
        )
        upbit_positions = _parse_holdings_to_positions(
            upbit_holdings, upbit_prices, "upbit"
        )
        for p in upbit_positions:
            all_positions_data.append(
                {
                    "symbol": p.symbol,
                    "exchange": p.exchange,
                    "quantity": p.quantity,
                    "avg_price": p.avg_price,
                    "current_price": p.current_price,
                }
            )

    # ===== Bithumb Holdings =====
    bithumb_holdings = get_holdings_bithumb()
    if bithumb_holdings:
        total_cash_balance += _get_cash_balance(bithumb_holdings)
        bithumb_symbols = [
            f"{entry.get('currency')}/KRW"
            for entry in bithumb_holdings
            if entry.get("currency") and entry.get("currency") != "KRW"
        ]
        bithumb_prices = (
            _get_current_prices(bithumb_symbols, "bithumb") if bithumb_symbols else {}
        )
        bithumb_positions = _parse_holdings_to_positions(
            bithumb_holdings, bithumb_prices, "bithumb"
        )
        for p in bithumb_positions:
            all_positions_data.append(
                {
                    "symbol": p.symbol,
                    "exchange": p.exchange,
                    "quantity": p.quantity,
                    "avg_price": p.avg_price,
                    "current_price": p.current_price,
                }
            )

    if not all_positions_data and total_cash_balance <= 0:
        return None

    total_cost = sum(p["quantity"] * p["avg_price"] for p in all_positions_data)
    total_value = sum(p["quantity"] * p["current_price"] for p in all_positions_data)

    return {
        "total_cost": total_cost,
        "total_value": total_value,
        "cash_balance": total_cash_balance,
        "positions": all_positions_data,
    }


def get_portfolio_summary() -> Optional[PortfolioSummary]:
    """Get portfolio summary from real holdings data (Upbit + Bithumb).

    Uses cached data (5s TTL) to improve tab switching performance.

    Returns:
        PortfolioSummary if data available, None otherwise (triggers demo mode)
    """
    cached = _fetch_portfolio_summary_cached()
    if cached is None:
        return None

    # Convert cached dict back to dataclasses
    positions = [
        PortfolioPosition(
            symbol=p["symbol"],
            exchange=p["exchange"],
            quantity=p["quantity"],
            avg_price=p["avg_price"],
            current_price=p["current_price"],
        )
        for p in cached["positions"]
    ]

    return PortfolioSummary(
        total_cost=cached["total_cost"],
        total_value=cached["total_value"],
        cash_balance=cached["cash_balance"],
        positions=positions,
    )


def _get_portfolio_summary_uncached() -> Optional[PortfolioSummary]:
    """Get portfolio summary without caching (for manual refresh).

    Returns:
        PortfolioSummary if data available, None otherwise (triggers demo mode)
    """
    # Check if private API is enabled
    if not is_private_api_enabled():
        logger.debug("Private API not enabled, returning None for demo mode")
        return None

    all_positions: List[PortfolioPosition] = []
    total_cash_balance = 0.0

    # ===== Upbit Holdings =====
    upbit_holdings = get_holdings_upbit()
    if upbit_holdings:
        # Extract Upbit cash balance
        total_cash_balance += _get_cash_balance(upbit_holdings)

        # Get symbols for Upbit price lookup
        upbit_symbols = [
            f"{entry.get('currency')}/KRW"
            for entry in upbit_holdings
            if entry.get("currency") and entry.get("currency") != "KRW"
        ]

        # Get Upbit prices
        upbit_prices = (
            _get_current_prices(upbit_symbols, "upbit") if upbit_symbols else {}
        )

        # Convert to positions
        upbit_positions = _parse_holdings_to_positions(
            upbit_holdings, upbit_prices, "upbit"
        )
        all_positions.extend(upbit_positions)

    # ===== Bithumb Holdings =====
    bithumb_holdings = get_holdings_bithumb()
    if bithumb_holdings:
        # Extract Bithumb cash balance
        total_cash_balance += _get_cash_balance(bithumb_holdings)

        # Get symbols for Bithumb price lookup
        bithumb_symbols = [
            f"{entry.get('currency')}/KRW"
            for entry in bithumb_holdings
            if entry.get("currency") and entry.get("currency") != "KRW"
        ]

        # Get Bithumb prices
        bithumb_prices = (
            _get_current_prices(bithumb_symbols, "bithumb") if bithumb_symbols else {}
        )

        # Convert to positions
        bithumb_positions = _parse_holdings_to_positions(
            bithumb_holdings, bithumb_prices, "bithumb"
        )
        all_positions.extend(bithumb_positions)

    # If no holdings from any exchange, return None for demo mode
    if not all_positions and total_cash_balance <= 0:
        logger.debug("No holdings from any exchange, returning None for demo mode")
        return None

    # Calculate totals
    total_cost = sum(p.cost for p in all_positions)
    total_value = sum(p.value for p in all_positions)

    return PortfolioSummary(
        total_cost=total_cost,
        total_value=total_value,
        cash_balance=total_cash_balance,
        positions=all_positions,
    )
