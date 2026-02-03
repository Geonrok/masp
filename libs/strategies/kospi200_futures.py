"""
KOSPI200 Futures Strategy Module

Multi-sub-strategy composite approach for KOSPI200 futures trading.
Uses T-1 data shift to prevent look-ahead bias.

Sub-strategies:
- VIX Below SMA20: Long when VIX(T-1) < SMA20(T-1)
- VIX Declining: Long when VIX(T-1) < VIX(T-2)
- Semicon+Foreign: Long when semicon > SMA20 AND foreign flow > 0
- Hourly MA: SMA 15/30 crossover on hourly data
"""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from libs.strategies.base import BaseStrategy, Signal, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class KOSPI200FuturesConfig:
    """Configuration for KOSPI200 Futures Strategy."""

    enabled_strategies: List[str] = field(
        default_factory=lambda: [
            "vix_below_sma20",
            "vix_declining",
            "semicon_foreign",
        ]
    )
    strategy_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "vix_below_sma20": 0.4,
            "vix_declining": 0.3,
            "semicon_foreign": 0.3,
        }
    )
    vix_sma_period: int = 20
    semicon_sma_period: int = 20
    foreign_sum_period: int = 20
    hourly_sma_short: int = 15
    hourly_sma_long: int = 30
    composite_threshold: float = 0.5
    round_trip_cost: float = 0.0009  # 0.09% round-trip


class KOSPI200SubStrategy(ABC):
    """Base class for KOSPI200 sub-strategies."""

    pass


class KOSPI200FuturesStrategy(BaseStrategy):
    """
    KOSPI200 Futures Composite Strategy.

    Combines multiple sub-strategies with configurable weights
    to generate trading signals for KOSPI200 futures.
    """

    def __init__(self, config: Optional[KOSPI200FuturesConfig] = None):
        """
        Initialize KOSPI200 Futures Strategy.

        Args:
            config: Strategy configuration (optional)
        """
        self.config = config or KOSPI200FuturesConfig()

        # Strategy metadata
        self.strategy_id = "kospi200_futures_v1"
        self.name = "KOSPI200 Futures Strategy"
        self.version = "1.0.0"

        # Data storage
        self._vix_data: Optional[pd.Series] = None
        self._semicon_data: Optional[pd.Series] = None
        self._foreign_data: Optional[pd.Series] = None
        self._hourly_data: Optional[pd.DataFrame] = None
        self._daily_data: Optional[pd.DataFrame] = None

        # State
        self._last_signals: Dict[str, int] = {}
        self._last_update: Optional[datetime] = None

        logger.info(
            f"[KOSPI200Futures] Initialized with {len(self.config.enabled_strategies)} "
            f"sub-strategies"
        )

    def load_data(self) -> bool:
        """
        Load required data for signal generation.

        In production, this would fetch from data providers.
        Returns True if data is available, False otherwise.
        """
        # Check if data is already loaded
        if (
            self._vix_data is not None
            and len(self._vix_data) >= self.config.vix_sma_period
        ):
            return True
        return False

    def _calculate_vix_below_sma20_signal(self) -> int:
        """
        Calculate VIX Below SMA20 signal.

        Uses T-1 shift: compares VIX(T-1) vs SMA(T-1 basis).
        Returns 1 (LONG) if VIX(T-1) < SMA, 0 (CASH) otherwise.
        """
        if (
            self._vix_data is None
            or len(self._vix_data) < self.config.vix_sma_period + 1
        ):
            return 0

        # T-1 shift: use data up to T-1 (exclude today)
        vix_t1 = self._vix_data.iloc[-2]  # T-1 VIX
        sma_t1 = self._vix_data.iloc[:-1].tail(self.config.vix_sma_period).mean()

        if vix_t1 < sma_t1:
            return 1  # LONG
        return 0  # CASH

    def _calculate_vix_declining_signal(self) -> int:
        """
        Calculate VIX Declining signal.

        Uses T-1 shift: compares VIX(T-1) vs VIX(T-2).
        Returns 1 (LONG) if VIX(T-1) < VIX(T-2), 0 (CASH) otherwise.
        """
        if self._vix_data is None or len(self._vix_data) < 3:
            return 0

        vix_t1 = self._vix_data.iloc[-2]  # T-1
        vix_t2 = self._vix_data.iloc[-3]  # T-2

        if vix_t1 < vix_t2:
            return 1  # LONG (VIX declining)
        return 0  # CASH

    def _calculate_semicon_foreign_signal(self) -> int:
        """
        Calculate Semicon + Foreign signal.

        Long when:
        - Semicon index > SMA(20)
        - Foreign 20-day cumulative flow > 0

        Returns 1 (LONG) if both conditions met, 0 (CASH) otherwise.
        """
        if (
            self._semicon_data is None
            or len(self._semicon_data) < self.config.semicon_sma_period
        ):
            return 0
        if (
            self._foreign_data is None
            or len(self._foreign_data) < self.config.foreign_sum_period
        ):
            return 0

        # Semicon condition
        semicon_current = self._semicon_data.iloc[-1]
        semicon_sma = self._semicon_data.tail(self.config.semicon_sma_period).mean()
        semicon_above = semicon_current > semicon_sma

        # Foreign flow condition
        foreign_sum = self._foreign_data.tail(self.config.foreign_sum_period).sum()
        foreign_positive = foreign_sum > 0

        if semicon_above and foreign_positive:
            return 1  # LONG
        return 0  # CASH

    def _calculate_sma_15_30_hourly_signal(self) -> int:
        """
        Calculate SMA 15/30 hourly crossover signal.

        Returns 1 (LONG) if SMA15 > SMA30, 0 (CASH) otherwise.
        """
        if (
            self._hourly_data is None
            or len(self._hourly_data) < self.config.hourly_sma_long
        ):
            return 0

        close = self._hourly_data["close"]
        sma_short = close.tail(self.config.hourly_sma_short).mean()
        sma_long = close.tail(self.config.hourly_sma_long).mean()

        if sma_short > sma_long:
            return 1  # LONG (uptrend)
        return 0  # CASH

    def _calculate_ema_15_20_hourly_signal(self) -> int:
        """Calculate EMA 15/20 hourly crossover signal."""
        if self._hourly_data is None or len(self._hourly_data) < 20:
            return 0

        close = self._hourly_data["close"]
        ema_15 = close.ewm(span=15, adjust=False).mean().iloc[-1]
        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]

        if ema_15 > ema_20:
            return 1
        return 0

    def _calculate_sma_20_30_hourly_signal(self) -> int:
        """Calculate SMA 20/30 hourly crossover signal."""
        if self._hourly_data is None or len(self._hourly_data) < 30:
            return 0

        close = self._hourly_data["close"]
        sma_20 = close.tail(20).mean()
        sma_30 = close.tail(30).mean()

        if sma_20 > sma_30:
            return 1
        return 0

    def calculate_composite_signal(self) -> Tuple[float, Dict[str, int]]:
        """
        Calculate composite signal from all enabled sub-strategies.

        Returns:
            Tuple of (composite_score, individual_signals)
            composite_score: Weighted average of signals (0-1)
            individual_signals: Dict of strategy_name -> signal (0 or 1)
        """
        signals = {}
        composite = 0.0

        signal_funcs = {
            "vix_below_sma20": self._calculate_vix_below_sma20_signal,
            "vix_declining": self._calculate_vix_declining_signal,
            "semicon_foreign": self._calculate_semicon_foreign_signal,
            "sma_15_30_hourly": self._calculate_sma_15_30_hourly_signal,
            "ema_15_20_hourly": self._calculate_ema_15_20_hourly_signal,
            "sma_20_30_hourly": self._calculate_sma_20_30_hourly_signal,
        }

        for strategy_name in self.config.enabled_strategies:
            if strategy_name in signal_funcs:
                signal = signal_funcs[strategy_name]()
                signals[strategy_name] = signal
                weight = self.config.strategy_weights.get(strategy_name, 0)
                composite += signal * weight

        self._last_signals = signals
        self._last_update = datetime.now()

        return composite, signals

    def generate_signals(self, symbols: List[str]) -> List[TradeSignal]:
        """
        Generate trading signals for given symbols.

        Args:
            symbols: List of symbols (typically ["KOSPI200"])

        Returns:
            List of TradeSignal objects
        """
        results = []

        for symbol in symbols:
            # Check if data is available
            if not self.load_data():
                results.append(
                    TradeSignal(
                        symbol=symbol,
                        signal=Signal.HOLD,
                        price=0.0,
                        timestamp=datetime.now(),
                        reason="Data load failed",
                        strength=0.0,
                    )
                )
                continue

            # Calculate composite signal
            composite, individual_signals = self.calculate_composite_signal()

            # Get current price from daily data
            price = 0.0
            if self._daily_data is not None and len(self._daily_data) > 0:
                price = float(self._daily_data["close"].iloc[-1])

            # Determine signal based on composite score
            if composite >= self.config.composite_threshold:
                signal = Signal.BUY
                reason = (
                    f"Composite {composite:.2f} >= {self.config.composite_threshold}"
                )
            else:
                signal = Signal.HOLD
                reason = (
                    f"Composite {composite:.2f} < {self.config.composite_threshold}"
                )

            results.append(
                TradeSignal(
                    symbol=symbol,
                    signal=signal,
                    price=price,
                    timestamp=datetime.now(),
                    reason=reason,
                    strength=composite,
                )
            )

        return results

    def get_current_status(self) -> Dict:
        """Get current strategy status."""
        vix_value = None
        if self._vix_data is not None and len(self._vix_data) > 0:
            vix_value = float(self._vix_data.iloc[-1])

        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "version": self.version,
            "enabled_strategies": self.config.enabled_strategies,
            "last_signals": self._last_signals,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "indicators": {
                "vix": vix_value,
            },
        }


# Convenience sub-strategy classes


class VIXBelowSMA20Strategy(KOSPI200FuturesStrategy):
    """Single-strategy variant: VIX Below SMA20 only."""

    def __init__(self):
        config = KOSPI200FuturesConfig(
            enabled_strategies=["vix_below_sma20"],
            strategy_weights={"vix_below_sma20": 1.0},
        )
        super().__init__(config)
        self.strategy_id = "kospi200_vix_below_sma20"
        self.name = "KOSPI200 VIX Below SMA20"


class VIXDecliningStrategy(KOSPI200FuturesStrategy):
    """Single-strategy variant: VIX Declining only."""

    def __init__(self):
        config = KOSPI200FuturesConfig(
            enabled_strategies=["vix_declining"],
            strategy_weights={"vix_declining": 1.0},
        )
        super().__init__(config)
        self.strategy_id = "kospi200_vix_declining"
        self.name = "KOSPI200 VIX Declining"


class SemiconForeignStrategy(KOSPI200FuturesStrategy):
    """Single-strategy variant: Semicon + Foreign only."""

    def __init__(self):
        config = KOSPI200FuturesConfig(
            enabled_strategies=["semicon_foreign"],
            strategy_weights={"semicon_foreign": 1.0},
        )
        super().__init__(config)
        self.strategy_id = "kospi200_semicon_foreign"
        self.name = "KOSPI200 Semicon+Foreign"


class KOSPI200HourlyStrategy(KOSPI200FuturesStrategy):
    """Hourly MA crossover strategies."""

    def __init__(self):
        config = KOSPI200FuturesConfig(
            enabled_strategies=[
                "sma_15_30_hourly",
                "ema_15_20_hourly",
                "sma_20_30_hourly",
            ],
            strategy_weights={
                "sma_15_30_hourly": 0.4,
                "ema_15_20_hourly": 0.3,
                "sma_20_30_hourly": 0.3,
            },
        )
        super().__init__(config)
        self.strategy_id = "kospi200_hourly_ma"
        self.name = "KOSPI200 Hourly MA"


class KOSPI200StablePortfolioStrategy(KOSPI200FuturesStrategy):
    """Stable portfolio: conservative weights."""

    def __init__(self):
        config = KOSPI200FuturesConfig(
            enabled_strategies=[
                "vix_below_sma20",
                "vix_declining",
                "semicon_foreign",
            ],
            strategy_weights={
                "vix_below_sma20": 0.5,
                "vix_declining": 0.3,
                "semicon_foreign": 0.2,
            },
        )
        super().__init__(config)
        self.strategy_id = "kospi200_stable_portfolio"
        self.name = "KOSPI200 Stable Portfolio"


class KOSPI200AggressivePortfolioStrategy(KOSPI200FuturesStrategy):
    """Aggressive portfolio: includes hourly strategies."""

    def __init__(self):
        config = KOSPI200FuturesConfig(
            enabled_strategies=[
                "vix_below_sma20",
                "ema_15_20_hourly",
                "sma_20_30_hourly",
            ],
            strategy_weights={
                "vix_below_sma20": 0.4,
                "ema_15_20_hourly": 0.35,
                "sma_20_30_hourly": 0.25,
            },
        )
        super().__init__(config)
        self.strategy_id = "kospi200_aggressive_portfolio"
        self.name = "KOSPI200 Aggressive Portfolio"
