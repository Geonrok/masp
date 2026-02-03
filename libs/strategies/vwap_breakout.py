"""
KAMA+EMA Hybrid VWAP Breakout Strategy v2.0

Long-only trend-following strategy for Binance USDT-M Futures (1h timeframe).

Entry conditions (ALL must be true):
    1. Close > Donchian Channel Upper(48) [shifted 1 bar]
    2. Close > VWAP(48) * 1.02
    3. EMA(50) > EMA(200)
    4. KAMA(20) slope > 0 (rising over 10 bars)
    5. Long only

Exit conditions (ANY triggers):
    1. Unrealized loss > ATR(14) * 3.0 (stop-loss)
    2. Unrealized profit > ATR(14) * 8.0 (take-profit)
    3. Bars held >= 72 (time-based exit)

Portfolio management:
    - Max 10 concurrent positions
    - Select lowest-volatility symbols
    - Vol-targeting position sizing
    - 720-bar (30-day) rebalance cycle

Validated: Phase 18-19 TRUE OOS (2yr, 18 windows)
Performance: Sharpe 1.41, +59.6% (5x), MDD -4.9%
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from libs.strategies.base import BaseStrategy, Signal, TradeSignal
from libs.strategies.indicators import (
    ATR,
    ATR_series,
    EMA,
    EMA_series,
    KAMA_series,
)

logger = logging.getLogger(__name__)

# Cost assumptions
COMMISSION = 0.0004  # 0.04% per trade (taker)
FUNDING_PER_8H = 0.0001  # 0.01% per 8h funding rate


class VwapBreakoutStrategy(BaseStrategy):
    """
    KAMA+EMA Hybrid VWAP Breakout (Long-Only).

    Designed for Binance USDT-M Futures on 1h timeframe.
    Generates per-symbol BUY/SELL/HOLD signals based on breakout conditions
    with ATR-based stop-loss and take-profit.

    Parameters:
        donchian_period: Donchian channel lookback (default 48).
        vwap_period: VWAP lookback (default 48).
        vwap_mult: VWAP multiplier threshold (default 1.02).
        ema_fast: Fast EMA period (default 50).
        ema_slow: Slow EMA period (default 200).
        kama_period: KAMA ER window (default 20).
        kama_slope_bars: Bars to measure KAMA slope (default 10).
        atr_stop: ATR multiplier for stop-loss (default 3.0).
        atr_target: ATR multiplier for take-profit (default 8.0).
        max_hold_bars: Maximum bars to hold (default 72).
        max_positions: Maximum concurrent positions (default 10).
    """

    strategy_id: str = "vwap_breakout"
    name: str = "KAMA+EMA Hybrid VWAP Breakout"
    version: str = "2.0.0"
    description: str = (
        "Long-only trend-following breakout strategy with KAMA adaptive filter. "
        "Validated via TRUE OOS (Sharpe 1.41, MDD -4.9%)."
    )

    # Default parameters (from Phase 18-19 validation)
    DEFAULT_DONCHIAN_PERIOD = 48
    DEFAULT_VWAP_PERIOD = 48
    DEFAULT_VWAP_MULT = 1.02
    DEFAULT_EMA_FAST = 50
    DEFAULT_EMA_SLOW = 200
    DEFAULT_KAMA_PERIOD = 20
    DEFAULT_KAMA_SLOPE_BARS = 10
    DEFAULT_ATR_STOP = 3.0
    DEFAULT_ATR_TARGET = 8.0
    DEFAULT_MAX_HOLD_BARS = 72
    DEFAULT_MAX_POSITIONS = 10

    def __init__(
        self,
        donchian_period: int = None,
        vwap_period: int = None,
        vwap_mult: float = None,
        ema_fast: int = None,
        ema_slow: int = None,
        kama_period: int = None,
        kama_slope_bars: int = None,
        atr_stop: float = None,
        atr_target: float = None,
        max_hold_bars: int = None,
        max_positions: int = None,
        market_data_adapter=None,
    ):
        super().__init__(name="VWAP-Breakout")

        self.donchian_period = donchian_period or self.DEFAULT_DONCHIAN_PERIOD
        self.vwap_period = vwap_period or self.DEFAULT_VWAP_PERIOD
        self.vwap_mult = vwap_mult or self.DEFAULT_VWAP_MULT
        self.ema_fast = ema_fast or self.DEFAULT_EMA_FAST
        self.ema_slow = ema_slow or self.DEFAULT_EMA_SLOW
        self.kama_period = kama_period or self.DEFAULT_KAMA_PERIOD
        self.kama_slope_bars = kama_slope_bars or self.DEFAULT_KAMA_SLOPE_BARS
        self.atr_stop = atr_stop or self.DEFAULT_ATR_STOP
        self.atr_target = atr_target or self.DEFAULT_ATR_TARGET
        self.max_hold_bars = max_hold_bars or self.DEFAULT_MAX_HOLD_BARS
        self.max_positions = max_positions or self.DEFAULT_MAX_POSITIONS

        self._market_data = market_data_adapter
        self._ohlcv_cache: Dict[str, dict] = {}  # symbol -> {close, high, low, volume}

        # Position tracking
        self._entry_prices: Dict[str, float] = {}
        self._entry_bars: Dict[str, int] = {}
        self._bar_counter: int = 0

        # Minimum bars needed for indicators
        self._min_bars = max(self.ema_slow, self.donchian_period, self.kama_period) + 50

        logger.info("[VwapBreakout] Initialized v%s", self.version)
        logger.info(
            "  Donchian=%d, VWAP=%d*%.2f, EMA=%d/%d, KAMA=%d, "
            "Stop=%.1fATR, Target=%.1fATR, MaxHold=%d",
            self.donchian_period,
            self.vwap_period,
            self.vwap_mult,
            self.ema_fast,
            self.ema_slow,
            self.kama_period,
            self.atr_stop,
            self.atr_target,
            self.max_hold_bars,
        )

    def set_market_data(self, adapter) -> None:
        """Set market data adapter."""
        self._market_data = adapter

    def _fetch_ohlcv(self, symbol: str) -> Optional[dict]:
        """Fetch OHLCV data for a symbol via market data adapter."""
        if not self._market_data:
            logger.warning("[VwapBreakout] No market data adapter")
            return None

        try:
            limit = self._min_bars + 50
            ohlcv_list = self._market_data.get_ohlcv(symbol, interval="1h", limit=limit)
            if not ohlcv_list or len(ohlcv_list) < self._min_bars:
                logger.debug(
                    "[VwapBreakout] Insufficient data for %s: %d bars",
                    symbol,
                    len(ohlcv_list) if ohlcv_list else 0,
                )
                return None

            close = np.array([c.close for c in ohlcv_list], dtype=float)
            high = np.array([c.high for c in ohlcv_list], dtype=float)
            low = np.array([c.low for c in ohlcv_list], dtype=float)
            volume = np.array([c.volume for c in ohlcv_list], dtype=float)

            return {"close": close, "high": high, "low": low, "volume": volume}

        except Exception as exc:
            logger.error("[VwapBreakout] OHLCV fetch failed for %s: %s", symbol, exc)
            return None

    def update_ohlcv(
        self, symbol: str, close: list, high: list, low: list, volume: list
    ) -> None:
        """Manually update OHLCV cache (for testing or external data feeds)."""
        self._ohlcv_cache[symbol] = {
            "close": np.array(close, dtype=float),
            "high": np.array(high, dtype=float),
            "low": np.array(low, dtype=float),
            "volume": np.array(volume, dtype=float),
        }

    def _check_entry(self, data: dict) -> bool:
        """
        Check all 5 entry conditions.

        Args:
            data: Dict with close, high, low, volume arrays.

        Returns:
            True if all conditions met.
        """
        close = data["close"]
        high = data["high"]
        volume = data["volume"]
        n = len(close)

        if n < self._min_bars:
            return False

        current_close = close[-1]

        # 1. Donchian breakout: close > highest high of last N bars (shifted 1)
        donchian_upper = np.max(high[-(self.donchian_period + 1) : -1])
        if current_close <= donchian_upper:
            return False

        # 2. VWAP filter: close > VWAP(N) * multiplier
        recent_close = close[-self.vwap_period :]
        recent_vol = volume[-self.vwap_period :]
        vol_sum = np.sum(recent_vol)
        if vol_sum <= 0:
            return False
        vwap = np.sum(recent_close * recent_vol) / vol_sum
        if current_close <= vwap * self.vwap_mult:
            return False

        # 3. EMA trend: EMA(fast) > EMA(slow)
        ema_f = EMA(close.tolist(), self.ema_fast)
        ema_s = EMA(close.tolist(), self.ema_slow)
        if ema_f <= ema_s:
            return False

        # 4. KAMA slope: KAMA(20) rising over last 10 bars
        kama_arr = KAMA_series(close, period=self.kama_period, fast_sc=2, slow_sc=30)
        if len(kama_arr) < self.kama_slope_bars + 1:
            return False
        kama_slope = kama_arr[-1] - kama_arr[-self.kama_slope_bars]
        if kama_slope <= 0:
            return False

        return True

    def _check_exit(self, symbol: str, data: dict) -> bool:
        """
        Check exit conditions for an open position.

        Args:
            symbol: Trading symbol.
            data: OHLCV data dict.

        Returns:
            True if position should be closed.
        """
        if symbol not in self._entry_prices:
            return False

        close = data["close"]
        high = data["high"]
        low = data["low"]
        current_close = close[-1]
        entry_price = self._entry_prices[symbol]

        # Current ATR
        current_atr = ATR(high.tolist(), low.tolist(), close.tolist(), period=14)
        if current_atr <= 0:
            current_atr = current_close * 0.01

        # Unrealized P&L in ATR units
        unrealized_atr = (current_close - entry_price) / current_atr

        # 1. Stop-loss
        if unrealized_atr < -self.atr_stop:
            logger.info(
                "[VwapBreakout] %s STOP-LOSS: %.2f ATR (limit: -%.1f)",
                symbol,
                unrealized_atr,
                self.atr_stop,
            )
            return True

        # 2. Take-profit
        if unrealized_atr > self.atr_target:
            logger.info(
                "[VwapBreakout] %s TAKE-PROFIT: %.2f ATR (target: %.1f)",
                symbol,
                unrealized_atr,
                self.atr_target,
            )
            return True

        # 3. Time-based exit
        bars_held = self._bar_counter - self._entry_bars.get(symbol, 0)
        if bars_held >= self.max_hold_bars:
            logger.info(
                "[VwapBreakout] %s TIME-EXIT: %d bars (max: %d)",
                symbol,
                bars_held,
                self.max_hold_bars,
            )
            return True

        return False

    def generate_signal(
        self,
        symbol: str,
        gate_pass: Optional[bool] = None,
    ) -> TradeSignal:
        """
        Generate signal for a single symbol.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT:PERP").
            gate_pass: Not used (no gate in this strategy).
        """
        # Get data
        data = self._ohlcv_cache.get(symbol)
        if data is None:
            data = self._fetch_ohlcv(symbol)
            if data is not None:
                self._ohlcv_cache[symbol] = data

        if data is None:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                price=0,
                timestamp=datetime.now(),
                reason="Data unavailable",
            )

        current_price = float(data["close"][-1])
        has_pos = self.has_position(symbol)

        # Check exit first (if we have a position)
        if has_pos:
            if self._check_exit(symbol, data):
                # Clean up tracking
                self._entry_prices.pop(symbol, None)
                self._entry_bars.pop(symbol, None)
                return TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    price=current_price,
                    timestamp=datetime.now(),
                    reason="Exit triggered",
                )
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                price=current_price,
                timestamp=datetime.now(),
                reason="Position held",
            )

        # Check entry (if no position)
        if self._check_entry(data):
            # Track entry
            self._entry_prices[symbol] = current_price
            self._entry_bars[symbol] = self._bar_counter
            return TradeSignal(
                symbol=symbol,
                signal=Signal.BUY,
                price=current_price,
                timestamp=datetime.now(),
                reason="VWAP Breakout entry",
                strength=1.0,
            )

        return TradeSignal(
            symbol=symbol,
            signal=Signal.HOLD,
            price=current_price,
            timestamp=datetime.now(),
            reason="No signal",
        )

    def generate_signals(self, symbols: List[str]) -> List[TradeSignal]:
        """Generate signals for multiple symbols."""
        self._bar_counter += 1
        signals: List[TradeSignal] = []

        for symbol in symbols:
            try:
                signal = self.generate_signal(symbol)
                signals.append(signal)

                if signal.signal != Signal.HOLD:
                    logger.info(
                        "[VwapBreakout] %s: %s - %s @ %.4f",
                        symbol,
                        signal.signal.value,
                        signal.reason,
                        signal.price,
                    )
            except Exception as exc:
                logger.error("[VwapBreakout] Error for %s: %s", symbol, exc)

        return signals

    def get_parameters(self) -> dict:
        """Return strategy parameters."""
        return {
            "donchian_period": self.donchian_period,
            "vwap_period": self.vwap_period,
            "vwap_mult": self.vwap_mult,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "kama_period": self.kama_period,
            "kama_slope_bars": self.kama_slope_bars,
            "atr_stop": self.atr_stop,
            "atr_target": self.atr_target,
            "max_hold_bars": self.max_hold_bars,
            "max_positions": self.max_positions,
        }
