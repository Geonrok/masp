"""
Upbit Spot Real Adapter (Phase 1)

Phase 0: Skeleton only - raises RuntimeError
Phase 1: Real implementation with Upbit API integration

Design principles:
- No actual API calls in Phase 0
- Clear error messages
- API key validation in Phase 1
"""

import logging
from typing import Optional
from libs.adapters.base import (
    MarketDataAdapter,
    ExecutionAdapter,
    MarketQuote,
    OrderResult,
)

logger = logging.getLogger(__name__)


class UpbitSpotMarketData(MarketDataAdapter):
    """
    Upbit Spot Market Data Adapter

    Phase 0: Not implemented (RuntimeError)
    Phase 1: Real Upbit API integration planned
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        logger.warning(
            "[UpbitSpotMarketData] Initialized. "
            "Phase 0: NOT USED. "
            "Phase 1: Real trading implementation pending."
        )

    def get_quote(self, symbol: str) -> Optional[MarketQuote]:
        """
        Get current quote for a symbol.
        Phase 1: Will implement Upbit ticker API
        """
        raise RuntimeError(
            f"[UpbitSpotMarketData] get_quote({symbol}): "
            "Not implemented. Phase 1 implementation pending. "
            "Phase 0: Use MockMarketDataAdapter."
        )

    def get_quotes(self, symbols: list[str]) -> dict[str, MarketQuote]:
        """
        Get quotes for multiple symbols.
        Phase 1: Will implement Upbit ticker API
        """
        raise RuntimeError(
            f"[UpbitSpotMarketData] get_quotes({len(symbols)} symbols): "
            "Not implemented. Phase 1 implementation pending."
        )

    def is_market_open(self) -> bool:
        """
        Check if Upbit market is open.
        Phase 1: Upbit operates 24/7, always returns True
        """
        raise RuntimeError(
            "[UpbitSpotMarketData] is_market_open(): "
            "Not implemented. Phase 1 implementation pending."
        )


class UpbitSpotExecution(ExecutionAdapter):
    """
    Upbit Spot Execution Adapter

    Phase 0: Not implemented (RuntimeError)
    Phase 1: Real Upbit API integration planned (API keys required)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        logger.warning(
            "[UpbitSpotExecution] Initialized. "
            "Phase 0: NOT USED. "
            "Phase 1: Real trading implementation pending (API keys required)."
        )

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> OrderResult:
        """
        Place an order on Upbit.
        Phase 1: Will implement Upbit order API
        """
        raise RuntimeError(
            f"[UpbitSpotExecution] place_order({symbol}, {side}, {quantity}): "
            "Not implemented. Phase 1 implementation pending. "
            "DANGER: Real money execution risk. Use Mock mode."
        )

    def get_order_status(self, order_id: str) -> Optional[dict]:
        """
        Get order status from Upbit.
        Phase 1: Will implement Upbit order query API
        """
        raise RuntimeError(
            f"[UpbitSpotExecution] get_order_status({order_id}): "
            "Not implemented. Phase 1 implementation pending."
        )

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order on Upbit.
        Phase 1: Will implement Upbit order cancellation API
        """
        raise RuntimeError(
            f"[UpbitSpotExecution] cancel_order({order_id}): "
            "Not implemented. Phase 1 implementation pending."
        )

    def get_balance(self, asset: str) -> Optional[float]:
        """
        Get balance from Upbit.
        Phase 1: Will implement Upbit balance query API
        """
        raise RuntimeError(
            f"[UpbitSpotExecution] get_balance({asset}): "
            "Not implemented. Phase 1 implementation pending."
        )
