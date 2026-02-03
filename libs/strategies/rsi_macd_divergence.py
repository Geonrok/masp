"""
RSI-MACD Divergence Strategy

Detects divergences between price action and RSI/MACD indicators
to identify potential trend reversals.

Bullish Divergence: Price makes lower low, RSI/MACD makes higher low
Bearish Divergence: Price makes higher high, RSI/MACD makes lower high
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from libs.strategies.base import (
    Action,
    BaseStrategy,
    Decision,
    Signal,
    StrategyContext,
    TradeSignal,
)
from libs.strategies.indicators import (
    MACD,
    RSI,
    MACD_series,
    RSI_series,
)

logger = logging.getLogger(__name__)


class DivergenceType(Enum):
    """Types of divergence patterns."""

    BULLISH_REGULAR = "bullish_regular"  # Price lower low, RSI higher low
    BEARISH_REGULAR = "bearish_regular"  # Price higher high, RSI lower high
    BULLISH_HIDDEN = "bullish_hidden"  # Price higher low, RSI lower low (continuation)
    BEARISH_HIDDEN = (
        "bearish_hidden"  # Price lower high, RSI higher high (continuation)
    )
    NONE = "none"


@dataclass
class DivergenceSignal:
    """Detected divergence signal."""

    type: DivergenceType
    price_pivot1: float
    price_pivot2: float
    indicator_pivot1: float
    indicator_pivot2: float
    strength: float  # 0-1, higher = stronger divergence
    indicator_name: str  # "RSI" or "MACD"


class RSIMACDDivergenceStrategy(BaseStrategy):
    """
    RSI-MACD Divergence Strategy.

    Combines RSI and MACD divergence detection for higher probability signals.
    Enters long on bullish divergence, short on bearish divergence.

    Parameters:
        rsi_period: RSI calculation period (default 14)
        rsi_oversold: RSI oversold threshold (default 30)
        rsi_overbought: RSI overbought threshold (default 70)
        macd_fast: MACD fast period (default 12)
        macd_slow: MACD slow period (default 26)
        macd_signal: MACD signal period (default 9)
        lookback_pivots: Number of bars to look back for pivots (default 20)
        min_divergence_bars: Minimum bars between pivots (default 5)
        require_both: Require both RSI and MACD divergence (default False)
    """

    strategy_id: str = "rsi_macd_divergence"
    name: str = "RSI-MACD Divergence"
    version: str = "1.0.0"
    description: str = "Divergence-based reversal strategy using RSI and MACD"

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        lookback_pivots: int = 20,
        min_divergence_bars: int = 5,
        require_both: bool = False,
    ):
        """Initialize RSI-MACD Divergence Strategy."""
        super().__init__(name=name, config=config)

        # RSI parameters
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # MACD parameters
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

        # Divergence detection parameters
        self.lookback_pivots = lookback_pivots
        self.min_divergence_bars = min_divergence_bars
        self.require_both = require_both

        # OHLCV data cache
        self._price_data: Dict[str, np.ndarray] = {}

        logger.info(
            f"[{self.name}] Initialized: RSI({rsi_period}), "
            f"MACD({macd_fast},{macd_slow},{macd_signal}), "
            f"require_both={require_both}"
        )

    def set_price_data(self, symbol: str, closes: List[float]) -> None:
        """Set price data for a symbol."""
        self._price_data[symbol] = np.array(closes, dtype=float)

    def _find_pivots(
        self,
        data: np.ndarray,
        window: int = 5,
    ) -> tuple[List[int], List[int]]:
        """
        Find pivot highs and lows in data.

        Args:
            data: Array of values.
            window: Window size for pivot detection.

        Returns:
            Tuple of (pivot_high_indices, pivot_low_indices)
        """
        n = len(data)
        pivot_highs = []
        pivot_lows = []

        for i in range(window, n - window):
            # Check for pivot high
            is_high = True
            for j in range(i - window, i + window + 1):
                if j != i and data[j] >= data[i]:
                    is_high = False
                    break
            if is_high:
                pivot_highs.append(i)

            # Check for pivot low
            is_low = True
            for j in range(i - window, i + window + 1):
                if j != i and data[j] <= data[i]:
                    is_low = False
                    break
            if is_low:
                pivot_lows.append(i)

        return pivot_highs, pivot_lows

    def _detect_rsi_divergence(
        self,
        prices: np.ndarray,
        rsi_values: np.ndarray,
    ) -> Optional[DivergenceSignal]:
        """
        Detect RSI divergence.

        Args:
            prices: Close price array.
            rsi_values: RSI value array.

        Returns:
            DivergenceSignal if divergence detected, None otherwise.
        """
        if len(prices) < self.lookback_pivots:
            return None

        # Get recent data
        recent_prices = prices[-self.lookback_pivots :]
        recent_rsi = rsi_values[-self.lookback_pivots :]

        # Find pivots
        price_highs, price_lows = self._find_pivots(recent_prices, window=3)
        rsi_highs, rsi_lows = self._find_pivots(recent_rsi, window=3)

        # Check for bullish divergence (in oversold territory)
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            # Get last two price lows
            p1_idx, p2_idx = price_lows[-2], price_lows[-1]
            if p2_idx - p1_idx >= self.min_divergence_bars:
                p1_price, p2_price = recent_prices[p1_idx], recent_prices[p2_idx]
                p1_rsi, p2_rsi = recent_rsi[p1_idx], recent_rsi[p2_idx]

                # Bullish: price lower low, RSI higher low
                if p2_price < p1_price and p2_rsi > p1_rsi:
                    if p2_rsi < self.rsi_oversold + 20:  # Near oversold
                        strength = min(1.0, (p1_rsi - p2_rsi) / 20 + 0.3)
                        return DivergenceSignal(
                            type=DivergenceType.BULLISH_REGULAR,
                            price_pivot1=p1_price,
                            price_pivot2=p2_price,
                            indicator_pivot1=p1_rsi,
                            indicator_pivot2=p2_rsi,
                            strength=strength,
                            indicator_name="RSI",
                        )

        # Check for bearish divergence (in overbought territory)
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            p1_idx, p2_idx = price_highs[-2], price_highs[-1]
            if p2_idx - p1_idx >= self.min_divergence_bars:
                p1_price, p2_price = recent_prices[p1_idx], recent_prices[p2_idx]
                p1_rsi, p2_rsi = recent_rsi[p1_idx], recent_rsi[p2_idx]

                # Bearish: price higher high, RSI lower high
                if p2_price > p1_price and p2_rsi < p1_rsi:
                    if p2_rsi > self.rsi_overbought - 20:  # Near overbought
                        strength = min(1.0, (p1_rsi - p2_rsi) / 20 + 0.3)
                        return DivergenceSignal(
                            type=DivergenceType.BEARISH_REGULAR,
                            price_pivot1=p1_price,
                            price_pivot2=p2_price,
                            indicator_pivot1=p1_rsi,
                            indicator_pivot2=p2_rsi,
                            strength=strength,
                            indicator_name="RSI",
                        )

        return None

    def _detect_macd_divergence(
        self,
        prices: np.ndarray,
        macd_hist: np.ndarray,
    ) -> Optional[DivergenceSignal]:
        """
        Detect MACD histogram divergence.

        Args:
            prices: Close price array.
            macd_hist: MACD histogram array.

        Returns:
            DivergenceSignal if divergence detected, None otherwise.
        """
        if len(prices) < self.lookback_pivots:
            return None

        recent_prices = prices[-self.lookback_pivots :]
        recent_macd = macd_hist[-self.lookback_pivots :]

        price_highs, price_lows = self._find_pivots(recent_prices, window=3)
        macd_highs, macd_lows = self._find_pivots(recent_macd, window=3)

        # Bullish MACD divergence
        if len(price_lows) >= 2 and len(macd_lows) >= 2:
            p1_idx, p2_idx = price_lows[-2], price_lows[-1]
            if p2_idx - p1_idx >= self.min_divergence_bars:
                p1_price, p2_price = recent_prices[p1_idx], recent_prices[p2_idx]
                p1_macd, p2_macd = recent_macd[p1_idx], recent_macd[p2_idx]

                if p2_price < p1_price and p2_macd > p1_macd:
                    if p2_macd < 0:  # MACD below zero line
                        strength = min(
                            1.0,
                            (
                                abs(p2_macd - p1_macd) / abs(p1_macd)
                                if p1_macd != 0
                                else 0.5
                            ),
                        )
                        return DivergenceSignal(
                            type=DivergenceType.BULLISH_REGULAR,
                            price_pivot1=p1_price,
                            price_pivot2=p2_price,
                            indicator_pivot1=p1_macd,
                            indicator_pivot2=p2_macd,
                            strength=strength,
                            indicator_name="MACD",
                        )

        # Bearish MACD divergence
        if len(price_highs) >= 2 and len(macd_highs) >= 2:
            p1_idx, p2_idx = price_highs[-2], price_highs[-1]
            if p2_idx - p1_idx >= self.min_divergence_bars:
                p1_price, p2_price = recent_prices[p1_idx], recent_prices[p2_idx]
                p1_macd, p2_macd = recent_macd[p1_idx], recent_macd[p2_idx]

                if p2_price > p1_price and p2_macd < p1_macd:
                    if p2_macd > 0:  # MACD above zero line
                        strength = min(
                            1.0,
                            (
                                abs(p1_macd - p2_macd) / abs(p1_macd)
                                if p1_macd != 0
                                else 0.5
                            ),
                        )
                        return DivergenceSignal(
                            type=DivergenceType.BEARISH_REGULAR,
                            price_pivot1=p1_price,
                            price_pivot2=p2_price,
                            indicator_pivot1=p1_macd,
                            indicator_pivot2=p2_macd,
                            strength=strength,
                            indicator_name="MACD",
                        )

        return None

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze a symbol for divergence signals.

        Args:
            symbol: Trading symbol.

        Returns:
            Dictionary with analysis results.
        """
        if symbol not in self._price_data:
            return {"error": f"No price data for {symbol}"}

        prices = self._price_data[symbol]
        if len(prices) < max(self.rsi_period, self.macd_slow) + self.lookback_pivots:
            return {"error": "Insufficient price data"}

        # Calculate indicators
        rsi_values = RSI_series(prices, self.rsi_period)
        macd_line, signal_line, macd_hist = MACD_series(
            prices, self.macd_fast, self.macd_slow, self.macd_signal
        )

        # Detect divergences
        rsi_div = self._detect_rsi_divergence(prices, rsi_values)
        macd_div = self._detect_macd_divergence(prices, macd_hist)

        current_rsi = float(rsi_values[-1])
        current_macd = float(macd_line[-1])
        current_signal = float(signal_line[-1])
        current_hist = float(macd_hist[-1])

        return {
            "symbol": symbol,
            "current_price": float(prices[-1]),
            "rsi": current_rsi,
            "macd_line": current_macd,
            "macd_signal": current_signal,
            "macd_histogram": current_hist,
            "rsi_divergence": rsi_div,
            "macd_divergence": macd_div,
            "rsi_oversold": current_rsi < self.rsi_oversold,
            "rsi_overbought": current_rsi > self.rsi_overbought,
        }

    def generate_signals(self, symbols: List[str]) -> List[TradeSignal]:
        """Generate trade signals based on divergence analysis."""
        signals = []
        now = datetime.now()

        for symbol in symbols:
            analysis = self.analyze(symbol)

            if "error" in analysis:
                logger.warning(f"[{self.name}] {symbol}: {analysis['error']}")
                continue

            rsi_div = analysis.get("rsi_divergence")
            macd_div = analysis.get("macd_divergence")
            current_price = analysis["current_price"]

            # Determine signal based on divergence detection
            signal = Signal.HOLD
            reason = ""
            strength = 0.0

            if self.require_both:
                # Require both RSI and MACD divergence
                if rsi_div and macd_div:
                    if (
                        rsi_div.type == DivergenceType.BULLISH_REGULAR
                        and macd_div.type == DivergenceType.BULLISH_REGULAR
                    ):
                        signal = Signal.BUY
                        strength = (rsi_div.strength + macd_div.strength) / 2
                        reason = (
                            f"Bullish divergence (RSI + MACD), strength={strength:.2f}"
                        )
                    elif (
                        rsi_div.type == DivergenceType.BEARISH_REGULAR
                        and macd_div.type == DivergenceType.BEARISH_REGULAR
                    ):
                        signal = Signal.SELL
                        strength = (rsi_div.strength + macd_div.strength) / 2
                        reason = (
                            f"Bearish divergence (RSI + MACD), strength={strength:.2f}"
                        )
            else:
                # Either RSI or MACD divergence is enough
                if rsi_div:
                    if rsi_div.type == DivergenceType.BULLISH_REGULAR:
                        signal = Signal.BUY
                        strength = rsi_div.strength
                        reason = f"RSI bullish divergence, strength={strength:.2f}"
                    elif rsi_div.type == DivergenceType.BEARISH_REGULAR:
                        signal = Signal.SELL
                        strength = rsi_div.strength
                        reason = f"RSI bearish divergence, strength={strength:.2f}"
                elif macd_div:
                    if macd_div.type == DivergenceType.BULLISH_REGULAR:
                        signal = Signal.BUY
                        strength = macd_div.strength
                        reason = f"MACD bullish divergence, strength={strength:.2f}"
                    elif macd_div.type == DivergenceType.BEARISH_REGULAR:
                        signal = Signal.SELL
                        strength = macd_div.strength
                        reason = f"MACD bearish divergence, strength={strength:.2f}"

            if signal == Signal.HOLD:
                reason = f"No divergence (RSI={analysis['rsi']:.1f})"

            signals.append(
                TradeSignal(
                    symbol=symbol,
                    signal=signal,
                    price=current_price,
                    timestamp=now,
                    reason=reason,
                    strength=strength,
                )
            )

            if signal != Signal.HOLD:
                logger.info(f"[{self.name}] {symbol}: {signal.value} - {reason}")

        return signals

    def execute(self, ctx: StrategyContext) -> List[Decision]:
        """Execute strategy and return decisions."""
        # Load price data from context
        for symbol in ctx.symbols:
            if symbol in ctx.market_data:
                closes = ctx.market_data[symbol].get("closes", [])
                if closes:
                    self.set_price_data(symbol, closes)

        return super().execute(ctx)
