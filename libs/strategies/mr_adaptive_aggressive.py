"""
MR_ADAPTIVE_AGGRESSIVE Strategy

Mean Reversion with Adaptive Trend Filter for Binance USDT-M Futures (1D timeframe).

Entry conditions (ALL must be true):
    1. Close < Bollinger Band Lower (20, 2)
    2. RSI(14) < 35 (oversold)

Exit conditions (ANY triggers):
    1. RSI > 55 (exit threshold)
    2. Close > Bollinger Band Middle

Position scaling:
    - 100% position when price > MA(50) (uptrend)
    - 30% position when price < MA(50) (downtrend)

Performance (Backtest):
    - Sharpe: 0.312 (+31.2% vs baseline)
    - MDD: -19.9%
    - Coverage: 99%

Direction: Long-only
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from libs.strategies.base import BaseStrategy, Signal, TradeSignal
from libs.strategies.indicators import (
    MA,
    RSI,
    RSI_series,
    BollingerBands,
    BollingerBands_series,
)

logger = logging.getLogger(__name__)


@dataclass
class MRAdaptiveConfig:
    """Configuration for MR_ADAPTIVE_AGGRESSIVE strategy."""

    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_low: int = 35
    rsi_exit: int = 55
    trend_ma: int = 50
    trend_scale: float = 0.3
    max_positions: int = 30


class MRAdaptiveAggressiveStrategy(BaseStrategy):
    """
    Mean Reversion with Adaptive Trend Filter (Aggressive variant).

    Designed for Binance USDT-M Futures on 1D timeframe.
    Generates per-symbol BUY/SELL/HOLD signals based on oversold conditions
    with trend-based position scaling.

    Parameters:
        bb_period: Bollinger Band period (default 20).
        bb_std: Standard deviation multiplier (default 2.0).
        rsi_period: RSI period (default 14).
        rsi_low: RSI entry threshold (default 35).
        rsi_exit: RSI exit threshold (default 55).
        trend_ma: Trend MA period (default 50).
        trend_scale: Position scale in downtrend (default 0.3).
        max_positions: Maximum concurrent positions (default 30).
    """

    strategy_id: str = "mr_adaptive_aggressive"
    name: str = "MR Adaptive Aggressive"
    version: str = "1.0.0"
    description: str = (
        "Mean reversion strategy with adaptive trend filter. "
        "Long-only, buys oversold conditions, scales position by trend. "
        "Validated: Sharpe 0.312, MDD -19.9%."
    )

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_low: int = 35,
        rsi_exit: int = 55,
        trend_ma: int = 50,
        trend_scale: float = 0.3,
        max_positions: int = 30,
        market_data_adapter=None,
        name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        super().__init__(name=name, config=config)

        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_low = rsi_low
        self.rsi_exit = rsi_exit
        self.trend_ma = trend_ma
        self.trend_scale = trend_scale
        self.max_positions = max_positions

        self._market_data = market_data_adapter
        self._ohlcv_cache: Dict[str, dict] = {}

        # Position tracking
        self._entry_prices: Dict[str, float] = {}
        self._position_scales: Dict[str, float] = {}

        # Minimum bars needed for indicators
        self._min_bars = max(self.trend_ma, self.bb_period) + 20

        logger.info("[MRAdaptiveAggressive] Initialized v%s", self.version)
        logger.info(
            "  BB=%d/%.1f, RSI=%d (<%d entry, >%d exit), TrendMA=%d, Scale=%.1f",
            self.bb_period,
            self.bb_std,
            self.rsi_period,
            self.rsi_low,
            self.rsi_exit,
            self.trend_ma,
            self.trend_scale,
        )

    def set_market_data(self, adapter) -> None:
        """Set market data adapter."""
        self._market_data = adapter

    def _fetch_ohlcv(self, symbol: str) -> Optional[dict]:
        """Fetch OHLCV data for a symbol via market data adapter."""
        if not self._market_data:
            logger.warning("[MRAdaptiveAggressive] No market data adapter")
            return None

        try:
            limit = self._min_bars + 50
            ohlcv_list = self._market_data.get_ohlcv(symbol, interval="1d", limit=limit)
            if not ohlcv_list or len(ohlcv_list) < self._min_bars:
                logger.debug(
                    "[MRAdaptiveAggressive] Insufficient data for %s: %d bars",
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
            logger.error(
                "[MRAdaptiveAggressive] OHLCV fetch failed for %s: %s", symbol, exc
            )
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

    def _calculate_indicators(self, data: dict) -> Optional[dict]:
        """
        Calculate all required indicators.

        Args:
            data: Dict with close, high, low, volume arrays.

        Returns:
            Dict with indicator values or None if insufficient data.
        """
        close = data["close"]
        n = len(close)

        if n < self._min_bars:
            return None

        current_close = close[-1]

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = BollingerBands(
            close, period=self.bb_period, std_dev=self.bb_std
        )

        # RSI
        rsi = RSI(close, period=self.rsi_period)

        # Trend MA
        trend_ma = MA(close, period=self.trend_ma)

        # Trend direction and position scale
        in_uptrend = current_close > trend_ma
        position_scale = 1.0 if in_uptrend else self.trend_scale

        return {
            "close": current_close,
            "bb_upper": bb_upper,
            "bb_mid": bb_mid,
            "bb_lower": bb_lower,
            "rsi": rsi,
            "trend_ma": trend_ma,
            "in_uptrend": in_uptrend,
            "position_scale": position_scale,
        }

    def _check_entry(self, indicators: dict) -> bool:
        """
        Check entry conditions.

        Entry: Close < BB_Lower AND RSI < rsi_low
        """
        return (
            indicators["close"] < indicators["bb_lower"]
            and indicators["rsi"] < self.rsi_low
        )

    def _check_exit(self, indicators: dict) -> bool:
        """
        Check exit conditions.

        Exit: RSI > rsi_exit OR Close > BB_Mid
        """
        return (
            indicators["rsi"] > self.rsi_exit
            or indicators["close"] > indicators["bb_mid"]
        )

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

        # Calculate indicators
        indicators = self._calculate_indicators(data)
        if indicators is None:
            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                price=0,
                timestamp=datetime.now(),
                reason="Insufficient data",
            )

        current_price = indicators["close"]
        has_pos = self.has_position(symbol)

        # Check exit first (if we have a position)
        if has_pos:
            if self._check_exit(indicators):
                # Clean up tracking
                self._entry_prices.pop(symbol, None)
                self._position_scales.pop(symbol, None)

                trend = "UP" if indicators["in_uptrend"] else "DOWN"
                reason = (
                    f"Exit: RSI={indicators['rsi']:.1f}, "
                    f"BB_Mid={indicators['bb_mid']:.4f}, Trend={trend}"
                )

                return TradeSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    price=current_price,
                    timestamp=datetime.now(),
                    reason=reason,
                )

            return TradeSignal(
                symbol=symbol,
                signal=Signal.HOLD,
                price=current_price,
                timestamp=datetime.now(),
                reason="Position held",
            )

        # Check entry (if no position)
        if self._check_entry(indicators):
            # Track entry
            self._entry_prices[symbol] = current_price
            self._position_scales[symbol] = indicators["position_scale"]

            trend = "UP" if indicators["in_uptrend"] else "DOWN"
            scale_pct = int(indicators["position_scale"] * 100)
            reason = (
                f"Entry: RSI={indicators['rsi']:.1f}, "
                f"BB_Lower={indicators['bb_lower']:.4f}, "
                f"Trend={trend}, Scale={scale_pct}%"
            )

            return TradeSignal(
                symbol=symbol,
                signal=Signal.BUY,
                price=current_price,
                timestamp=datetime.now(),
                reason=reason,
                strength=indicators["position_scale"],
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
        signals: List[TradeSignal] = []

        for symbol in symbols:
            try:
                signal = self.generate_signal(symbol)
                signals.append(signal)

                if signal.signal != Signal.HOLD:
                    logger.info(
                        "[MRAdaptiveAggressive] %s: %s - %s @ %.4f",
                        symbol,
                        signal.signal.value,
                        signal.reason,
                        signal.price,
                    )
            except Exception as exc:
                logger.error("[MRAdaptiveAggressive] Error for %s: %s", symbol, exc)

        return signals

    def get_position_scale(self, symbol: str) -> float:
        """Get the position scale for a symbol (for execution adapter)."""
        return self._position_scales.get(symbol, 1.0)

    def get_parameters(self) -> dict:
        """Return strategy parameters."""
        return {
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "rsi_period": self.rsi_period,
            "rsi_low": self.rsi_low,
            "rsi_exit": self.rsi_exit,
            "trend_ma": self.trend_ma,
            "trend_scale": self.trend_scale,
            "max_positions": self.max_positions,
        }
