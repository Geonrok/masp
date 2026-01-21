"""Multi-exchange provider - connects exchange registry to dashboard."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Singleton coordinator
_coordinator = None


def _get_coordinator():
    """Get or create MultiExchangeCoordinator instance."""
    global _coordinator

    if _coordinator is not None:
        return _coordinator

    try:
        from libs.adapters.multi_exchange import MultiExchangeCoordinator

        _coordinator = MultiExchangeCoordinator()
        return _coordinator
    except Exception as e:
        logger.debug("MultiExchangeCoordinator init failed: %s", e)
        return None


def _get_registry():
    """Get ExchangeRegistry instance."""
    try:
        from libs.adapters.exchange_registry import get_registry

        return get_registry()
    except Exception as e:
        logger.debug("ExchangeRegistry import failed: %s", e)
        return None


@st.cache_data(ttl=60, show_spinner=False)
def get_exchange_list() -> List[Dict[str, Any]]:
    """Get list of all registered exchanges.

    Cached for 60 seconds since exchange list rarely changes.

    Returns:
        List of exchange info dicts
    """
    registry = _get_registry()

    if registry is None:
        # Return demo data
        return [
            {
                "name": "upbit_spot",
                "display_name": "Upbit",
                "exchange_type": "spot",
                "region": "kr",
                "status": "online",
                "latency_ms": 50,
            },
            {
                "name": "bithumb_spot",
                "display_name": "Bithumb",
                "exchange_type": "spot",
                "region": "kr",
                "status": "online",
                "latency_ms": 60,
            },
            {
                "name": "binance_futures",
                "display_name": "Binance Futures",
                "exchange_type": "futures",
                "region": "global",
                "status": "unknown",
                "latency_ms": 0,
            },
        ]

    try:
        exchanges = registry.get_all()
        return [info.to_dict() for info in exchanges.values()]
    except Exception as e:
        logger.error("Failed to get exchanges: %s", e)
        return []


def get_exchange_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all exchanges.

    Returns:
        Dict of exchange name to status info
    """
    coordinator = _get_coordinator()

    if coordinator is None:
        return {}

    try:
        return coordinator.get_exchange_status()
    except Exception as e:
        logger.error("Failed to get exchange status: %s", e)
        return {}


def get_price_comparison(
    symbol: str, exchanges: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """Get price comparison across exchanges.

    Args:
        symbol: Trading symbol
        exchanges: List of exchanges (None = all)

    Returns:
        Price comparison dict or None
    """
    coordinator = _get_coordinator()

    if coordinator is None:
        return None

    try:
        return coordinator.get_price_comparison(symbol, exchanges)
    except Exception as e:
        logger.error("Failed to get price comparison: %s", e)
        return None


def find_arbitrage_opportunities(
    symbols: List[str], min_profit_pct: float = 0.1
) -> List[Dict[str, Any]]:
    """Find arbitrage opportunities.

    Args:
        symbols: List of symbols to check
        min_profit_pct: Minimum profit threshold

    Returns:
        List of arbitrage opportunities
    """
    coordinator = _get_coordinator()

    if coordinator is None:
        return []

    try:
        return coordinator.find_arbitrage_opportunities(symbols, min_profit_pct)
    except Exception as e:
        logger.error("Failed to find arbitrage: %s", e)
        return []


def get_best_exchange(symbol: str, side: str = "buy") -> Optional[Dict[str, Any]]:
    """Get best exchange for buying or selling.

    Args:
        symbol: Trading symbol
        side: "buy" or "sell"

    Returns:
        Dict with exchange and price, or None
    """
    coordinator = _get_coordinator()

    if coordinator is None:
        return None

    try:
        if side.lower() == "buy":
            result = coordinator.get_best_exchange_for_buy(symbol)
        else:
            result = coordinator.get_best_exchange_for_sell(symbol)

        if result:
            return {"exchange": result[0], "price": result[1], "symbol": symbol, "side": side}
        return None
    except Exception as e:
        logger.error("Failed to get best exchange: %s", e)
        return None


def perform_health_check() -> Dict[str, str]:
    """Perform health check on all exchanges.

    Returns:
        Dict of exchange name to status
    """
    coordinator = _get_coordinator()

    if coordinator is None:
        return {}

    try:
        results = coordinator.health_check()
        return {name: status.value for name, status in results.items()}
    except Exception as e:
        logger.error("Failed to perform health check: %s", e)
        return {}


def get_registry_summary() -> Dict[str, Any]:
    """Get exchange registry summary.

    Returns:
        Summary dict
    """
    registry = _get_registry()

    if registry is None:
        return {
            "total": 3,
            "online": 2,
            "offline": 0,
            "by_type": {"spot": 2, "futures": 1},
            "by_region": {"korea": 2, "global": 1},
        }

    try:
        return registry.get_summary()
    except Exception as e:
        logger.error("Failed to get summary: %s", e)
        return {}
