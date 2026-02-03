"""
Strategy Engine - Async/Sync Bridge Layer.
Phase 4B-v5: asyncio.to_thread integration.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from libs.exchanges.binance_futures import BinanceFuturesAdapter
from libs.strategies.atlas_futures import ATLASFuturesStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class StrategyEngine:
    """Strategy execution engine (async/sync bridge)."""

    def __init__(
        self,
        strategy: ATLASFuturesStrategy,
        adapter: BinanceFuturesAdapter,
    ):
        self.strategy = strategy
        self.adapter = adapter
        self._running = False

    async def start(self) -> None:
        """Start engine."""
        await self.adapter.initialize()
        self._running = True
        logger.info("[ENGINE] Started")

    async def stop(self) -> None:
        """Stop engine."""
        self._running = False
        await self.adapter.close()
        logger.info("[ENGINE] Stopped")

    async def execute_trading_cycle(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Run a single trading cycle."""
        if not self._running:
            return None

        try:
            ohlcv_raw = await self.adapter.fetch_ohlcv(
                symbol=symbol,
                timeframe=self.strategy.config.timeframe,
                limit=500,
            )

            df = pd.DataFrame(ohlcv_raw).copy()

            signal = await asyncio.to_thread(
                self.strategy.generate_signal,
                symbol,
                df,
            )

            result = await self._execute_signal(symbol, signal)

            return {
                "symbol": symbol,
                "signal": signal.signal_type.value,
                "reason": signal.reason,
                "executed": result is not None,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as exc:
            logger.error("[ENGINE] Trading cycle error: %s", exc)
            return None

    async def _execute_signal(self, symbol: str, signal: Signal) -> Optional[Any]:
        """Execute orders for a signal."""
        if signal.signal_type == SignalType.HOLD:
            return None

        await self.adapter.set_leverage(symbol, self.strategy.config.leverage)

        if signal.signal_type == SignalType.LONG:
            balance = await self.adapter.get_balance()
            amount = self._calculate_order_amount(balance["free"], signal.price)
            return await self.adapter.create_market_order(symbol, "buy", amount)

        if signal.signal_type == SignalType.SHORT:
            balance = await self.adapter.get_balance()
            amount = self._calculate_order_amount(balance["free"], signal.price)
            return await self.adapter.create_market_order(symbol, "sell", amount)

        if signal.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT):
            return await self.adapter.close_position(symbol)

        return None

    def _calculate_order_amount(self, balance: float, price: float) -> float:
        """Calculate order size."""
        position_value = balance * (self.strategy.config.position_size_pct / 100)
        return position_value / price
