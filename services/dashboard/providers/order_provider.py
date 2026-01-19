"""Order execution provider - connects UpbitExecution to order_panel component."""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def is_live_trading_enabled() -> bool:
    """Check if live trading is enabled.

    Returns:
        True if MASP_ENABLE_LIVE_TRADING is set to "1"
    """
    return os.getenv("MASP_ENABLE_LIVE_TRADING", "0") == "1"


def _get_upbit_execution():
    """Get UpbitExecutionAdapter instance.

    Returns:
        UpbitExecutionAdapter instance or None if unavailable
    """
    if not is_live_trading_enabled():
        return None

    try:
        from libs.adapters.real_upbit_execution import UpbitExecutionAdapter
        from libs.core.config import Config

        config = Config(asset_class="spot", strategy_name="dashboard")
        return UpbitExecutionAdapter(config)
    except ImportError as e:
        logger.warning("UpbitExecutionAdapter import failed: %s", e)
        return None
    except ValueError as e:
        logger.warning("UpbitExecutionAdapter config error: %s", e)
        return None
    except Exception as e:
        logger.warning("UpbitExecutionAdapter initialization failed: %s", e)
        return None


class OrderExecutionWrapper:
    """Wrapper that adapts UpbitExecutionAdapter to order_panel interface.

    This wrapper translates the order_panel's expected interface to
    the actual UpbitExecutionAdapter interface.
    """

    def __init__(self, adapter):
        """Initialize with UpbitExecutionAdapter.

        Args:
            adapter: UpbitExecutionAdapter instance
        """
        self._adapter = adapter

    def place_order(
        self,
        symbol: str,
        side: str,
        units: Optional[float] = None,
        amount_krw: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place order via UpbitExecutionAdapter.

        This method adapts the order_panel interface to UpbitExecutionAdapter.

        Args:
            symbol: Currency code (e.g., "BTC")
            side: Order side ("buy" or "sell")
            units: Quantity to trade
            amount_krw: Amount in KRW (for BUY orders)

        Returns:
            Dict with order result in order_panel expected format
        """
        # Convert symbol format: BTC -> BTC/KRW
        full_symbol = f"{symbol}/KRW"

        # Determine quantity
        if side.lower() == "buy" and amount_krw and not units:
            # For market buy with amount, get current price to calculate quantity
            current_price = self._adapter.get_current_price(full_symbol)
            if current_price and current_price > 0:
                quantity = amount_krw / current_price
            else:
                return {
                    "success": False,
                    "order_id": "",
                    "executed_qty": 0.0,
                    "executed_price": 0.0,
                    "total_value": 0.0,
                    "fee": 0.0,
                    "message": "Price unavailable for amount-based order",
                }
        else:
            quantity = units or 0.0

        if quantity <= 0:
            return {
                "success": False,
                "order_id": "",
                "executed_qty": 0.0,
                "executed_price": 0.0,
                "total_value": 0.0,
                "fee": 0.0,
                "message": "Invalid quantity",
            }

        # Execute order
        result = self._adapter.place_order(
            symbol=full_symbol,
            side=side.upper(),
            quantity=quantity,
            order_type="MARKET",
        )

        # Convert result to order_panel format
        success = result.status not in ("REJECTED",)
        total_value = result.filled_quantity * result.filled_price

        return {
            "success": success,
            "order_id": result.order_id,
            "executed_qty": result.filled_quantity,
            "executed_price": result.filled_price,
            "total_value": total_value,
            "fee": result.fee,
            "message": result.message,
        }


def get_execution_adapter() -> Optional[OrderExecutionWrapper]:
    """Get wrapped execution adapter for order_panel.

    Returns:
        OrderExecutionWrapper if live trading enabled and adapter available,
        None otherwise (triggers demo mode in order_panel)
    """
    adapter = _get_upbit_execution()
    if adapter is None:
        return None
    return OrderExecutionWrapper(adapter)


def get_price_provider() -> Optional[Callable[[str], float]]:
    """Get price provider function for order_panel.

    Returns:
        Function that returns current price for a symbol,
        or None if unavailable
    """
    adapter = _get_upbit_execution()
    if adapter is None:
        return None

    def price_provider(symbol: str) -> float:
        """Get current price for symbol.

        Args:
            symbol: Currency code (e.g., "BTC")

        Returns:
            Current price in KRW, or 0.0 if unavailable
        """
        full_symbol = f"{symbol}/KRW"
        price = adapter.get_current_price(full_symbol)
        return float(price) if price else 0.0

    return price_provider


def get_balance_provider() -> Optional[Callable[[], Dict[str, Dict[str, float]]]]:
    """Get balance provider function for order_panel.

    Returns:
        Function that returns balances dict,
        or None if unavailable
    """
    adapter = _get_upbit_execution()
    if adapter is None:
        return None

    def balance_provider() -> Dict[str, Dict[str, float]]:
        """Get all balances.

        Returns:
            Dict mapping currency to {"available": float, "locked": float}
        """
        balances_raw = adapter.get_all_balances()
        balances: Dict[str, Dict[str, float]] = {}

        for entry in balances_raw:
            currency = entry.get("currency", "")
            if currency:
                available = float(entry.get("balance", 0))
                locked = float(entry.get("locked", 0))
                balances[currency] = {"available": available, "locked": locked}

        return balances

    return balance_provider
