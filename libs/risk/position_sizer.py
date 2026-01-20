"""
Position Sizing Algorithms for MASP

Implements various position sizing strategies:
- Fixed Fractional: Risk a fixed percentage of capital per trade
- Kelly Criterion: Optimal sizing based on win rate and payoff ratio
- Volatility-Based: Size positions based on asset volatility (ATR)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Result of position size calculation."""

    quantity: float
    risk_amount: float
    position_value: float
    risk_percent: float
    method: str


class PositionSizer(ABC):
    """Abstract base class for position sizing algorithms."""

    @abstractmethod
    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        **kwargs,
    ) -> PositionSize:
        """
        Calculate position size.

        Args:
            capital: Available trading capital
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price (optional)
            **kwargs: Additional parameters

        Returns:
            PositionSize with calculated values
        """
        raise NotImplementedError


class FixedFractionalSizer(PositionSizer):
    """
    Fixed Fractional Position Sizing.

    Risks a fixed percentage of capital on each trade.
    Position size = (Capital * Risk%) / (Entry - StopLoss)
    """

    def __init__(self, risk_percent: float = 0.02, max_position_percent: float = 0.25):
        """
        Initialize Fixed Fractional Sizer.

        Args:
            risk_percent: Fraction of capital to risk per trade (default 2%)
            max_position_percent: Maximum position size as fraction of capital (default 25%)
        """
        if not 0 < risk_percent <= 0.1:
            raise ValueError("risk_percent must be between 0 and 0.1 (10%)")
        if not 0 < max_position_percent <= 1.0:
            raise ValueError("max_position_percent must be between 0 and 1.0")

        self.risk_percent = risk_percent
        self.max_position_percent = max_position_percent
        logger.info(
            f"[FixedFractional] Initialized with risk={risk_percent:.1%}, "
            f"max_position={max_position_percent:.1%}"
        )

    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        **kwargs,
    ) -> PositionSize:
        """Calculate position size using fixed fractional method."""
        if capital <= 0:
            raise ValueError("Capital must be positive")
        if entry_price <= 0:
            raise ValueError("Entry price must be positive")

        risk_amount = capital * self.risk_percent
        max_position_value = capital * self.max_position_percent

        if stop_loss_price is not None and stop_loss_price > 0:
            # Calculate based on stop loss distance
            risk_per_unit = abs(entry_price - stop_loss_price)
            if risk_per_unit > 0:
                quantity = risk_amount / risk_per_unit
            else:
                quantity = risk_amount / entry_price
        else:
            # No stop loss - use risk_percent of capital directly
            quantity = risk_amount / entry_price

        # Apply maximum position limit
        position_value = quantity * entry_price
        if position_value > max_position_value:
            quantity = max_position_value / entry_price
            position_value = max_position_value
            logger.warning(
                f"[FixedFractional] Position capped to max {self.max_position_percent:.1%}"
            )

        return PositionSize(
            quantity=quantity,
            risk_amount=risk_amount,
            position_value=position_value,
            risk_percent=self.risk_percent,
            method="fixed_fractional",
        )


class KellyCriterionSizer(PositionSizer):
    """
    Kelly Criterion Position Sizing.

    Calculates optimal position size based on historical win rate and payoff ratio.
    Kelly% = W - [(1-W) / R]
    Where: W = Win rate, R = Win/Loss ratio

    Uses fractional Kelly (default 25%) to reduce volatility.
    """

    def __init__(
        self,
        win_rate: float = 0.5,
        payoff_ratio: float = 1.5,
        kelly_fraction: float = 0.25,
        max_position_percent: float = 0.25,
    ):
        """
        Initialize Kelly Criterion Sizer.

        Args:
            win_rate: Historical win rate (0-1)
            payoff_ratio: Average win / Average loss ratio
            kelly_fraction: Fraction of full Kelly to use (default 25%)
            max_position_percent: Maximum position size as fraction of capital
        """
        if not 0 < win_rate < 1:
            raise ValueError("win_rate must be between 0 and 1")
        if payoff_ratio <= 0:
            raise ValueError("payoff_ratio must be positive")
        if not 0 < kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be between 0 and 1")

        self.win_rate = win_rate
        self.payoff_ratio = payoff_ratio
        self.kelly_fraction = kelly_fraction
        self.max_position_percent = max_position_percent

        # Calculate full Kelly percentage
        self.full_kelly = win_rate - ((1 - win_rate) / payoff_ratio)
        self.adjusted_kelly = max(0, self.full_kelly * kelly_fraction)

        logger.info(
            f"[Kelly] W={win_rate:.1%}, R={payoff_ratio:.2f}, "
            f"Full Kelly={self.full_kelly:.1%}, Adjusted={self.adjusted_kelly:.1%}"
        )

    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        **kwargs,
    ) -> PositionSize:
        """Calculate position size using Kelly Criterion."""
        if capital <= 0:
            raise ValueError("Capital must be positive")
        if entry_price <= 0:
            raise ValueError("Entry price must be positive")

        # Kelly suggests no position if expected value is negative
        if self.adjusted_kelly <= 0:
            return PositionSize(
                quantity=0,
                risk_amount=0,
                position_value=0,
                risk_percent=0,
                method="kelly_criterion",
            )

        position_percent = min(self.adjusted_kelly, self.max_position_percent)
        position_value = capital * position_percent
        quantity = position_value / entry_price

        # Calculate actual risk based on stop loss
        if stop_loss_price is not None and stop_loss_price > 0:
            risk_per_unit = abs(entry_price - stop_loss_price)
            risk_amount = quantity * risk_per_unit
        else:
            risk_amount = position_value * 0.1  # Assume 10% risk without stop

        return PositionSize(
            quantity=quantity,
            risk_amount=risk_amount,
            position_value=position_value,
            risk_percent=position_percent,
            method="kelly_criterion",
        )

    def update_stats(self, win_rate: float, payoff_ratio: float) -> None:
        """Update win rate and payoff ratio from recent trades."""
        self.win_rate = win_rate
        self.payoff_ratio = payoff_ratio
        self.full_kelly = win_rate - ((1 - win_rate) / payoff_ratio)
        self.adjusted_kelly = max(0, self.full_kelly * self.kelly_fraction)
        logger.info(
            f"[Kelly] Updated: W={win_rate:.1%}, R={payoff_ratio:.2f}, "
            f"Adjusted Kelly={self.adjusted_kelly:.1%}"
        )


class VolatilityBasedSizer(PositionSizer):
    """
    Volatility-Based Position Sizing.

    Sizes positions inversely to asset volatility using ATR.
    Higher volatility = smaller position size.
    Position size = (Capital * Risk%) / (ATR * ATR_Multiplier)
    """

    def __init__(
        self,
        risk_percent: float = 0.02,
        atr_multiplier: float = 2.0,
        max_position_percent: float = 0.25,
    ):
        """
        Initialize Volatility-Based Sizer.

        Args:
            risk_percent: Fraction of capital to risk per trade
            atr_multiplier: Multiplier for ATR to set stop distance
            max_position_percent: Maximum position size as fraction of capital
        """
        if not 0 < risk_percent <= 0.1:
            raise ValueError("risk_percent must be between 0 and 0.1")
        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be positive")

        self.risk_percent = risk_percent
        self.atr_multiplier = atr_multiplier
        self.max_position_percent = max_position_percent
        logger.info(
            f"[VolatilityBased] risk={risk_percent:.1%}, "
            f"ATR_mult={atr_multiplier}, max={max_position_percent:.1%}"
        )

    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        atr: Optional[float] = None,
        **kwargs,
    ) -> PositionSize:
        """
        Calculate position size based on volatility.

        Args:
            capital: Available trading capital
            entry_price: Entry price
            stop_loss_price: Not used (ATR-based stop)
            atr: Average True Range value (required)
        """
        if capital <= 0:
            raise ValueError("Capital must be positive")
        if entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if atr is None or atr <= 0:
            raise ValueError("ATR must be provided and positive")

        risk_amount = capital * self.risk_percent
        stop_distance = atr * self.atr_multiplier

        # Position size based on ATR stop
        quantity = risk_amount / stop_distance
        position_value = quantity * entry_price

        # Apply maximum position limit
        max_position_value = capital * self.max_position_percent
        if position_value > max_position_value:
            quantity = max_position_value / entry_price
            position_value = max_position_value

        return PositionSize(
            quantity=quantity,
            risk_amount=risk_amount,
            position_value=position_value,
            risk_percent=self.risk_percent,
            method="volatility_based",
        )
