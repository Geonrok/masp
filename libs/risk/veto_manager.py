"""
Veto Manager - 4-Layer Risk Control System

Hierarchical veto system to prevent trades during unfavorable conditions.

Layers:
1. Kill Switch - Manual emergency stop (highest priority)
2. Market Structure - ADX < 20 (no trend), CI > 61.8 (choppy)
3. On-Chain - Exchange inflow Z-score > 2.0 (sell pressure)
4. Derivatives - Funding rate > 0.1% or < -0.1% (extreme sentiment)

Reference: Multiple AI consensus from ChatGPT, Gemini, DeepSeek, Perplexity, Grok
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VetoLevel(Enum):
    """Veto layer levels."""

    KILL_SWITCH = auto()  # Level 1: Manual emergency stop
    MARKET_STRUCTURE = auto()  # Level 2: Trend/volatility conditions
    ON_CHAIN = auto()  # Level 3: Exchange flow analysis
    DERIVATIVES = auto()  # Level 4: Funding rate analysis


@dataclass
class VetoResult:
    """Result of veto check."""

    can_trade: bool
    veto_level: Optional[VetoLevel] = None
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "can_trade": self.can_trade,
            "veto_level": self.veto_level.name if self.veto_level else None,
            "reason": self.reason,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VetoConfig:
    """Configuration for veto thresholds."""

    # Kill Switch
    kill_switch_enabled: bool = False

    # Market Structure (Level 2)
    adx_threshold: float = 20.0  # ADX < this = no trend
    adx_period: int = 14
    ci_threshold: float = 61.8  # Choppiness Index > this = choppy
    ci_period: int = 14

    # On-Chain (Level 3)
    inflow_zscore_threshold: float = 2.0  # Z-score > this = sell pressure
    inflow_lookback: int = 30  # Days for Z-score calculation

    # Derivatives (Level 4)
    funding_rate_threshold: float = 0.001  # 0.1% absolute threshold
    funding_rate_dynamic: bool = True  # Use dynamic threshold (mean - 2*std)
    funding_rate_lookback: int = 30


class VetoManager:
    """
    4-Layer Veto Manager

    Checks conditions in order of priority:
    1. Kill Switch (if enabled, blocks all trades)
    2. Market Structure (ADX, CI)
    3. On-Chain (exchange inflow)
    4. Derivatives (funding rate)

    If any layer vetoes, trade is blocked.
    """

    def __init__(self, config: Optional[VetoConfig] = None):
        """
        Initialize Veto Manager.

        Args:
            config: Veto configuration (uses defaults if None)
        """
        self.config = config or VetoConfig()
        self._kill_switch = self.config.kill_switch_enabled
        self._veto_history: List[VetoResult] = []

    def can_trade(
        self,
        symbol: str,
        side: str,
        context: Dict[str, Any],
    ) -> VetoResult:
        """
        Check if trade is allowed through all veto layers.

        Args:
            symbol: Trading symbol
            side: Trade direction ("long" or "short")
            context: Market context data with keys:
                - ohlcv: DataFrame with OHLCV data
                - funding_rate: Current funding rate (optional)
                - funding_history: Historical funding rates (optional)
                - inflow_data: Exchange inflow data (optional)

        Returns:
            VetoResult indicating if trade is allowed
        """
        # Layer 1: Kill Switch
        if self._kill_switch:
            result = VetoResult(
                can_trade=False,
                veto_level=VetoLevel.KILL_SWITCH,
                reason="Kill switch is enabled",
            )
            self._log_veto(result)
            return result

        # Layer 2: Market Structure
        ohlcv = context.get("ohlcv")
        if ohlcv is not None and len(ohlcv) >= self.config.adx_period:
            market_result = self._check_market_structure(ohlcv, side)
            if not market_result.can_trade:
                self._log_veto(market_result)
                return market_result

        # Layer 3: On-Chain
        inflow_data = context.get("inflow_data")
        if inflow_data is not None:
            onchain_result = self._check_on_chain(inflow_data, side)
            if not onchain_result.can_trade:
                self._log_veto(onchain_result)
                return onchain_result

        # Layer 4: Derivatives
        funding_rate = context.get("funding_rate")
        funding_history = context.get("funding_history")
        if funding_rate is not None:
            derivatives_result = self._check_derivatives(
                funding_rate, funding_history, side
            )
            if not derivatives_result.can_trade:
                self._log_veto(derivatives_result)
                return derivatives_result

        # All checks passed
        result = VetoResult(
            can_trade=True,
            reason="All veto checks passed",
        )
        return result

    def _check_market_structure(self, ohlcv: pd.DataFrame, side: str) -> VetoResult:
        """Check market structure conditions (ADX, CI)."""
        high = ohlcv["high"].values
        low = ohlcv["low"].values
        close = ohlcv["close"].values

        # Calculate ADX
        adx = calculate_adx(high, low, close, self.config.adx_period)
        current_adx = adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else 0

        # Check ADX threshold (trend following only works in trending markets)
        if current_adx < self.config.adx_threshold:
            return VetoResult(
                can_trade=False,
                veto_level=VetoLevel.MARKET_STRUCTURE,
                reason=f"ADX={current_adx:.1f} < {self.config.adx_threshold} (no trend)",
                details={"adx": current_adx, "threshold": self.config.adx_threshold},
            )

        # Calculate Choppiness Index
        ci = calculate_choppiness_index(high, low, close, self.config.ci_period)
        current_ci = ci[-1] if len(ci) > 0 and not np.isnan(ci[-1]) else 0

        # Check CI threshold (high CI = choppy market)
        if current_ci > self.config.ci_threshold:
            return VetoResult(
                can_trade=False,
                veto_level=VetoLevel.MARKET_STRUCTURE,
                reason=f"CI={current_ci:.1f} > {self.config.ci_threshold} (choppy market)",
                details={"ci": current_ci, "threshold": self.config.ci_threshold},
            )

        return VetoResult(
            can_trade=True,
            details={"adx": current_adx, "ci": current_ci},
        )

    def _check_on_chain(self, inflow_data: pd.Series, side: str) -> VetoResult:
        """Check on-chain conditions (exchange inflow)."""
        if len(inflow_data) < self.config.inflow_lookback:
            return VetoResult(can_trade=True, reason="Insufficient inflow data")

        # Calculate Z-score
        recent = inflow_data.tail(self.config.inflow_lookback)
        mean_inflow = recent.mean()
        std_inflow = recent.std()

        if std_inflow == 0:
            return VetoResult(can_trade=True, reason="No variance in inflow data")

        current_inflow = inflow_data.iloc[-1]
        zscore = (current_inflow - mean_inflow) / std_inflow

        # High inflow Z-score = selling pressure (bad for longs)
        if side == "long" and zscore > self.config.inflow_zscore_threshold:
            return VetoResult(
                can_trade=False,
                veto_level=VetoLevel.ON_CHAIN,
                reason=f"Inflow Z={zscore:.2f} > {self.config.inflow_zscore_threshold} (sell pressure)",
                details={
                    "zscore": zscore,
                    "threshold": self.config.inflow_zscore_threshold,
                },
            )

        # Low inflow Z-score = accumulation (bad for shorts)
        if side == "short" and zscore < -self.config.inflow_zscore_threshold:
            return VetoResult(
                can_trade=False,
                veto_level=VetoLevel.ON_CHAIN,
                reason=f"Inflow Z={zscore:.2f} < -{self.config.inflow_zscore_threshold} (accumulation)",
                details={
                    "zscore": zscore,
                    "threshold": -self.config.inflow_zscore_threshold,
                },
            )

        return VetoResult(
            can_trade=True,
            details={"inflow_zscore": zscore},
        )

    def _check_derivatives(
        self,
        funding_rate: float,
        funding_history: Optional[pd.Series],
        side: str,
    ) -> VetoResult:
        """Check derivatives conditions (funding rate)."""
        threshold = self.config.funding_rate_threshold

        # Dynamic threshold if history available
        if self.config.funding_rate_dynamic and funding_history is not None:
            if len(funding_history) >= self.config.funding_rate_lookback:
                recent = funding_history.tail(self.config.funding_rate_lookback)
                mean_fr = recent.mean()
                std_fr = recent.std()
                # Dynamic threshold: mean - 2*std (for long entry)
                dynamic_threshold = abs(mean_fr) + 2 * std_fr
                threshold = max(threshold, dynamic_threshold)

        # High positive funding = crowded long (bad for longs)
        if side == "long" and funding_rate > threshold:
            return VetoResult(
                can_trade=False,
                veto_level=VetoLevel.DERIVATIVES,
                reason=f"Funding={funding_rate*100:.3f}% > {threshold*100:.3f}% (crowded long)",
                details={"funding_rate": funding_rate, "threshold": threshold},
            )

        # High negative funding = crowded short (bad for shorts)
        if side == "short" and funding_rate < -threshold:
            return VetoResult(
                can_trade=False,
                veto_level=VetoLevel.DERIVATIVES,
                reason=f"Funding={funding_rate*100:.3f}% < -{threshold*100:.3f}% (crowded short)",
                details={"funding_rate": funding_rate, "threshold": -threshold},
            )

        return VetoResult(
            can_trade=True,
            details={"funding_rate": funding_rate},
        )

    def enable_kill_switch(self) -> None:
        """Enable kill switch (blocks all trades)."""
        self._kill_switch = True
        logger.warning("[VetoManager] Kill switch ENABLED")

    def disable_kill_switch(self) -> None:
        """Disable kill switch."""
        self._kill_switch = False
        logger.info("[VetoManager] Kill switch disabled")

    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self._kill_switch

    def _log_veto(self, result: VetoResult) -> None:
        """Log veto event."""
        self._veto_history.append(result)
        logger.info(
            "[VetoManager] VETO: %s - %s",
            result.veto_level.name if result.veto_level else "NONE",
            result.reason,
        )

    def get_veto_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent veto history."""
        return [v.to_dict() for v in self._veto_history[-limit:]]


def calculate_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Calculate Average Directional Index (ADX).

    ADX measures trend strength:
    - ADX < 20: Weak or no trend (range-bound)
    - ADX 20-40: Trending
    - ADX > 40: Strong trend

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Calculation period

    Returns:
        ADX values
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)

    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Smoothed averages (Wilder's smoothing)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)

    atr[period] = np.mean(tr[1 : period + 1])
    plus_di[period] = np.mean(plus_dm[1 : period + 1])
    minus_di[period] = np.mean(minus_dm[1 : period + 1])

    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        plus_di[i] = (plus_di[i - 1] * (period - 1) + plus_dm[i]) / period
        minus_di[i] = (minus_di[i - 1] * (period - 1) + minus_dm[i]) / period

    # DI values
    plus_di_pct = np.zeros(n)
    minus_di_pct = np.zeros(n)
    for i in range(period, n):
        if atr[i] > 0:
            plus_di_pct[i] = 100 * plus_di[i] / atr[i]
            minus_di_pct[i] = 100 * minus_di[i] / atr[i]

    # DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di_pct[i] + minus_di_pct[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di_pct[i] - minus_di_pct[i]) / di_sum

    # ADX (smoothed DX)
    adx = np.full(n, np.nan)
    adx[2 * period - 1] = np.mean(dx[period : 2 * period])

    for i in range(2 * period, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


def calculate_choppiness_index(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Calculate Choppiness Index (CI).

    CI measures market choppiness:
    - CI < 38.2: Trending market
    - CI 38.2-61.8: Transitional
    - CI > 61.8: Choppy/sideways market

    Formula: 100 * LOG10(SUM(TR, n) / (MAX(H, n) - MIN(L, n))) / LOG10(n)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Calculation period

    Returns:
        CI values
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)

    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Choppiness Index
    ci = np.full(n, np.nan)
    log_period = np.log10(period)

    for i in range(period - 1, n):
        tr_sum = np.sum(tr[i - period + 1 : i + 1])
        high_max = np.max(high[i - period + 1 : i + 1])
        low_min = np.min(low[i - period + 1 : i + 1])

        range_val = high_max - low_min
        if range_val > 0 and tr_sum > 0:
            ci[i] = 100 * np.log10(tr_sum / range_val) / log_period

    return ci


def calculate_funding_rate_signal(
    funding_rate: float,
    funding_history: Optional[pd.Series] = None,
    threshold: float = 0.001,
) -> Dict[str, Any]:
    """
    Analyze funding rate for trading signal.

    Args:
        funding_rate: Current funding rate
        funding_history: Historical funding rates
        threshold: Threshold for extreme readings

    Returns:
        Signal analysis dict
    """
    signal = {
        "funding_rate": funding_rate,
        "signal": "neutral",
        "zscore": None,
        "percentile": None,
    }

    # Basic threshold check
    if funding_rate > threshold:
        signal["signal"] = "bearish"  # Crowded longs
    elif funding_rate < -threshold:
        signal["signal"] = "bullish"  # Crowded shorts

    # Z-score if history available
    if funding_history is not None and len(funding_history) >= 30:
        mean_fr = funding_history.mean()
        std_fr = funding_history.std()
        if std_fr > 0:
            signal["zscore"] = (funding_rate - mean_fr) / std_fr
            signal["percentile"] = (funding_history < funding_rate).mean() * 100

    return signal
