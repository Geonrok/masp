"""
MA Crossover Strategy - Simple Moving Average Crossover Implementation

Phase 1: Mock implementation with simulated MA values
Phase 2+: Real implementation with actual price data

Strategy Logic:
- Calculate short-term MA (e.g., 5-period) and long-term MA (e.g., 20-period)
- BUY when short MA crosses above long MA (golden cross)
- SELL when short MA crosses below long MA (death cross)
- HOLD when no crossover occurs
"""

import hashlib
from datetime import datetime
from typing import Any

import pytz

from libs.strategies.base import Action, BaseStrategy, Decision, StrategyContext


class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy

    Phase 1: Mock implementation with simulated MA values
    Uses deterministic hash-based simulation to generate reproducible signals

    Parameters:
    - short_period: Short MA period (default: 5)
    - long_period: Long MA period (default: 20)
    """

    strategy_id = "ma_crossover_v1"
    name = "MA Crossover Strategy"
    version = "1.0.0"
    description = "Simple Moving Average crossover strategy (golden/death cross)"

    def __init__(self, short_period: int = 5, long_period: int = 20):
        """
        Initialize MA Crossover strategy.

        Args:
            short_period: Short-term MA period
            long_period: Long-term MA period
        """
        super().__init__()
        self.short_period = short_period
        self.long_period = long_period
        self._kst = pytz.timezone("Asia/Seoul")

        if short_period >= long_period:
            raise ValueError(
                f"Short period ({short_period}) must be less than long period ({long_period})"
            )

    def _simulate_ma_values(
        self, symbol: str, date_str: str
    ) -> tuple[float, float, float, float]:
        """
        Simulate MA values for Phase 1 testing.

        In Phase 2+, this will be replaced with actual price data calculation.

        Args:
            symbol: Trading symbol
            date_str: Date string

        Returns:
            Tuple of (current_price, short_ma, long_ma, prev_short_ma, prev_long_ma)
        """
        # Use hash for reproducible simulation
        hash_input = f"{symbol}:{date_str}"
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()

        # Simulate base price (scaled by symbol hash for variety)
        symbol_hash = hashlib.sha256(symbol.encode()).digest()
        base_price = 100 + (symbol_hash[0] % 100)  # 100-200 range

        # Simulate price movement (-5% to +5%)
        price_delta = (hash_bytes[0] / 255 - 0.5) * 10
        current_price = base_price * (1 + price_delta / 100)

        # Simulate MAs with some lag
        short_ma = base_price * (1 + price_delta / 100 * 0.7)  # Less lag
        long_ma = base_price * (1 + price_delta / 100 * 0.3)  # More lag

        # Simulate previous MAs (for crossover detection)
        prev_short_ma = base_price * (1 + (hash_bytes[1] / 255 - 0.5) * 8)
        prev_long_ma = base_price * (1 + (hash_bytes[2] / 255 - 0.5) * 6)

        return current_price, short_ma, long_ma, prev_short_ma, prev_long_ma

    def _detect_crossover(
        self, short_ma: float, long_ma: float, prev_short_ma: float, prev_long_ma: float
    ) -> tuple[Action, str]:
        """
        Detect MA crossover and generate trading signal.

        Args:
            short_ma: Current short-term MA
            long_ma: Current long-term MA
            prev_short_ma: Previous short-term MA
            prev_long_ma: Previous long-term MA

        Returns:
            Tuple of (Action, explanation)
        """
        # Check for golden cross (bullish signal)
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            return (
                Action.BUY,
                f"Golden Cross: MA{self.short_period} crossed above MA{self.long_period}",
            )

        # Check for death cross (bearish signal)
        if prev_short_ma >= prev_long_ma and short_ma < long_ma:
            return (
                Action.SELL,
                f"Death Cross: MA{self.short_period} crossed below MA{self.long_period}",
            )

        # No crossover - hold current position
        if short_ma > long_ma:
            return (
                Action.HOLD,
                f"Uptrend: MA{self.short_period} above MA{self.long_period}, holding",
            )
        else:
            return (
                Action.HOLD,
                f"Downtrend: MA{self.short_period} below MA{self.long_period}, holding",
            )

    def _generate_metrics(
        self, current_price: float, short_ma: float, long_ma: float
    ) -> dict[str, Any]:
        """Generate decision metrics."""
        return {
            f"ma{self.short_period}": round(short_ma, 2),
            f"ma{self.long_period}": round(long_ma, 2),
            "current_price": round(current_price, 2),
            "ma_spread_pct": round((short_ma - long_ma) / long_ma * 100, 2),
            "price_vs_short_ma_pct": round(
                (current_price - short_ma) / short_ma * 100, 2
            ),
        }

    def execute(self, ctx: StrategyContext) -> list[Decision]:
        """
        Execute MA Crossover strategy.

        Args:
            ctx: Strategy context

        Returns:
            List of trading decisions
        """
        decisions = []

        # Get current date in KST
        now_kst = datetime.now(self._kst)
        date_str = now_kst.strftime("%Y-%m-%d")

        for symbol in ctx.symbols:
            # Phase 1: Simulate MA values
            # Phase 2+: Calculate from real price data
            current_price, short_ma, long_ma, prev_short_ma, prev_long_ma = (
                self._simulate_ma_values(symbol, date_str)
            )

            # Detect crossover and generate signal
            action, notes = self._detect_crossover(
                short_ma, long_ma, prev_short_ma, prev_long_ma
            )

            # Generate metrics
            metrics = self._generate_metrics(current_price, short_ma, long_ma)

            decision = Decision(
                symbol=symbol,
                action=action,
                notes=notes,
                metrics=metrics,
            )
            decisions.append(decision)

        return decisions
