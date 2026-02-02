"""
Stop Loss / Take Profit Manager for MASP

Implements position-level exit logic:
- Fixed percentage stop loss / take profit
- Trailing stop
- ATR-based dynamic stop
- Time-based stop (maximum holding period)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Reason for exit signal."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"
    MANUAL = "manual"
    STRATEGY = "strategy"


@dataclass
class ExitSignal:
    """Exit signal from stop loss manager."""

    should_exit: bool
    reason: Optional[ExitReason] = None
    exit_price: Optional[float] = None
    pnl_percent: Optional[float] = None
    message: str = ""


@dataclass
class Position:
    """Position tracking for stop loss management."""

    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    quantity: float
    entry_time: datetime = field(default_factory=datetime.now)
    highest_price: float = 0.0  # For trailing stop (long)
    lowest_price: float = float("inf")  # For trailing stop (short)

    def update_extremes(self, current_price: float) -> None:
        """Update highest/lowest prices for trailing stop."""
        if self.side == "long":
            self.highest_price = max(self.highest_price, current_price)
        else:
            self.lowest_price = min(self.lowest_price, current_price)

    def get_pnl_percent(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.side == "long":
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price


class StopLossStrategy(ABC):
    """Abstract base class for stop loss strategies."""

    @abstractmethod
    def check_exit(self, position: Position, current_price: float) -> ExitSignal:
        """
        Check if position should be exited.

        Args:
            position: Current position
            current_price: Current market price

        Returns:
            ExitSignal indicating whether to exit
        """
        raise NotImplementedError


class FixedPercentageStop(StopLossStrategy):
    """
    Fixed Percentage Stop Loss / Take Profit.

    Exits position when price moves specified percentage from entry.
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.05,
        take_profit_pct: Optional[float] = 0.10,
    ):
        """
        Initialize Fixed Percentage Stop.

        Args:
            stop_loss_pct: Stop loss percentage (default 5%)
            take_profit_pct: Take profit percentage (optional, default 10%)
        """
        if not 0 < stop_loss_pct <= 0.5:
            raise ValueError("stop_loss_pct must be between 0 and 0.5")
        if take_profit_pct is not None and take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive if specified")

        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        logger.info(
            f"[FixedPercentageStop] SL={stop_loss_pct:.1%}, "
            f"TP={take_profit_pct:.1%}" if take_profit_pct else ""
        )

    def check_exit(self, position: Position, current_price: float) -> ExitSignal:
        """Check if position hits stop loss or take profit."""
        pnl_pct = position.get_pnl_percent(current_price)

        # Check stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                exit_price=current_price,
                pnl_percent=pnl_pct,
                message=f"Stop loss triggered at {pnl_pct:.1%}",
            )

        # Check take profit
        if self.take_profit_pct and pnl_pct >= self.take_profit_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT,
                exit_price=current_price,
                pnl_percent=pnl_pct,
                message=f"Take profit triggered at {pnl_pct:.1%}",
            )

        return ExitSignal(
            should_exit=False,
            pnl_percent=pnl_pct,
            message=f"Position active, P&L: {pnl_pct:.1%}",
        )


class TrailingStop(StopLossStrategy):
    """
    Trailing Stop Loss.

    Stop loss moves up (for long) or down (for short) as price moves favorably.
    Only activates after position reaches activation threshold.
    """

    def __init__(
        self,
        trail_pct: float = 0.03,
        activation_pct: Optional[float] = 0.02,
        initial_stop_pct: float = 0.05,
    ):
        """
        Initialize Trailing Stop.

        Args:
            trail_pct: Trail distance as percentage (default 3%)
            activation_pct: Minimum profit to activate trailing (default 2%)
            initial_stop_pct: Initial fixed stop before activation (default 5%)
        """
        if not 0 < trail_pct <= 0.3:
            raise ValueError("trail_pct must be between 0 and 0.3")
        if not 0 < initial_stop_pct <= 0.5:
            raise ValueError("initial_stop_pct must be between 0 and 0.5")

        self.trail_pct = trail_pct
        self.activation_pct = activation_pct
        self.initial_stop_pct = initial_stop_pct
        logger.info(
            f"[TrailingStop] trail={trail_pct:.1%}, "
            f"activation={activation_pct:.1%}, initial_stop={initial_stop_pct:.1%}"
        )

    def check_exit(self, position: Position, current_price: float) -> ExitSignal:
        """Check if position hits trailing stop."""
        # Update extremes
        position.update_extremes(current_price)

        pnl_pct = position.get_pnl_percent(current_price)

        # Check if trailing stop is activated
        is_activated = False
        if self.activation_pct is None:
            is_activated = True
        elif position.side == "long":
            max_pnl = (position.highest_price - position.entry_price) / position.entry_price
            is_activated = max_pnl >= self.activation_pct
        else:
            max_pnl = (position.entry_price - position.lowest_price) / position.entry_price
            is_activated = max_pnl >= self.activation_pct

        if is_activated:
            # Calculate trailing stop price
            if position.side == "long":
                stop_price = position.highest_price * (1 - self.trail_pct)
                if current_price <= stop_price:
                    return ExitSignal(
                        should_exit=True,
                        reason=ExitReason.TRAILING_STOP,
                        exit_price=current_price,
                        pnl_percent=pnl_pct,
                        message=f"Trailing stop triggered. High: {position.highest_price:.2f}, "
                        f"Stop: {stop_price:.2f}, Current: {current_price:.2f}",
                    )
            else:
                stop_price = position.lowest_price * (1 + self.trail_pct)
                if current_price >= stop_price:
                    return ExitSignal(
                        should_exit=True,
                        reason=ExitReason.TRAILING_STOP,
                        exit_price=current_price,
                        pnl_percent=pnl_pct,
                        message=f"Trailing stop triggered. Low: {position.lowest_price:.2f}, "
                        f"Stop: {stop_price:.2f}, Current: {current_price:.2f}",
                    )
        else:
            # Use initial fixed stop before activation
            if pnl_pct <= -self.initial_stop_pct:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.STOP_LOSS,
                    exit_price=current_price,
                    pnl_percent=pnl_pct,
                    message=f"Initial stop loss triggered at {pnl_pct:.1%}",
                )

        return ExitSignal(
            should_exit=False,
            pnl_percent=pnl_pct,
            message=f"Position active (trailing {'activated' if is_activated else 'pending'}), P&L: {pnl_pct:.1%}",
        )


class ATRBasedStop(StopLossStrategy):
    """
    ATR-Based Dynamic Stop Loss.

    Stop distance is based on Average True Range for volatility adjustment.
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        take_profit_multiplier: Optional[float] = 3.0,
    ):
        """
        Initialize ATR-Based Stop.

        Args:
            atr_multiplier: Multiplier for ATR to set stop distance
            take_profit_multiplier: Multiplier for ATR to set take profit (optional)
        """
        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be positive")

        self.atr_multiplier = atr_multiplier
        self.take_profit_multiplier = take_profit_multiplier
        self._current_atr: Optional[float] = None
        logger.info(
            f"[ATRBasedStop] atr_mult={atr_multiplier}, tp_mult={take_profit_multiplier}"
        )

    def set_atr(self, atr: float) -> None:
        """Update current ATR value."""
        if atr <= 0:
            raise ValueError("ATR must be positive")
        self._current_atr = atr

    def check_exit(self, position: Position, current_price: float) -> ExitSignal:
        """Check if position hits ATR-based stop."""
        if self._current_atr is None:
            logger.warning("[ATRBasedStop] ATR not set, using 2% of entry price")
            atr = position.entry_price * 0.02
        else:
            atr = self._current_atr

        pnl_pct = position.get_pnl_percent(current_price)
        stop_distance = atr * self.atr_multiplier

        # Calculate stop and take profit prices
        if position.side == "long":
            stop_price = position.entry_price - stop_distance
            tp_price = (
                position.entry_price + (atr * self.take_profit_multiplier)
                if self.take_profit_multiplier
                else None
            )

            if current_price <= stop_price:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.STOP_LOSS,
                    exit_price=current_price,
                    pnl_percent=pnl_pct,
                    message=f"ATR stop triggered. Entry: {position.entry_price:.2f}, "
                    f"Stop: {stop_price:.2f} (ATR={atr:.2f})",
                )

            if tp_price and current_price >= tp_price:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.TAKE_PROFIT,
                    exit_price=current_price,
                    pnl_percent=pnl_pct,
                    message=f"ATR take profit triggered at {tp_price:.2f}",
                )
        else:
            stop_price = position.entry_price + stop_distance
            tp_price = (
                position.entry_price - (atr * self.take_profit_multiplier)
                if self.take_profit_multiplier
                else None
            )

            if current_price >= stop_price:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.STOP_LOSS,
                    exit_price=current_price,
                    pnl_percent=pnl_pct,
                    message=f"ATR stop triggered. Entry: {position.entry_price:.2f}, "
                    f"Stop: {stop_price:.2f} (ATR={atr:.2f})",
                )

            if tp_price and current_price <= tp_price:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.TAKE_PROFIT,
                    exit_price=current_price,
                    pnl_percent=pnl_pct,
                    message=f"ATR take profit triggered at {tp_price:.2f}",
                )

        return ExitSignal(
            should_exit=False,
            pnl_percent=pnl_pct,
            message=f"Position active, P&L: {pnl_pct:.1%}, ATR stop at {stop_price:.2f}",
        )


class TimeBasedStop(StopLossStrategy):
    """
    Time-Based Stop (Maximum Holding Period).

    Forces exit after specified holding period regardless of P&L.
    """

    def __init__(
        self,
        max_holding_hours: float = 24.0,
        fallback_stop_pct: float = 0.05,
    ):
        """
        Initialize Time-Based Stop.

        Args:
            max_holding_hours: Maximum holding period in hours
            fallback_stop_pct: Fallback stop loss percentage
        """
        if max_holding_hours <= 0:
            raise ValueError("max_holding_hours must be positive")

        self.max_holding_hours = max_holding_hours
        self.max_holding_delta = timedelta(hours=max_holding_hours)
        self.fallback_stop_pct = fallback_stop_pct
        logger.info(f"[TimeBasedStop] max_holding={max_holding_hours}h")

    def check_exit(self, position: Position, current_price: float) -> ExitSignal:
        """Check if position exceeds maximum holding period."""
        pnl_pct = position.get_pnl_percent(current_price)
        holding_time = datetime.now() - position.entry_time

        # Check time stop
        if holding_time >= self.max_holding_delta:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TIME_STOP,
                exit_price=current_price,
                pnl_percent=pnl_pct,
                message=f"Time stop triggered after {holding_time.total_seconds()/3600:.1f}h",
            )

        # Check fallback stop loss
        if pnl_pct <= -self.fallback_stop_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                exit_price=current_price,
                pnl_percent=pnl_pct,
                message=f"Fallback stop loss triggered at {pnl_pct:.1%}",
            )

        hours_remaining = (self.max_holding_delta - holding_time).total_seconds() / 3600
        return ExitSignal(
            should_exit=False,
            pnl_percent=pnl_pct,
            message=f"Position active, P&L: {pnl_pct:.1%}, {hours_remaining:.1f}h remaining",
        )


class CompositeStopManager:
    """
    Composite Stop Loss Manager.

    Combines multiple stop strategies and triggers exit on first signal.
    """

    def __init__(self, strategies: Optional[list[StopLossStrategy]] = None):
        """
        Initialize Composite Stop Manager.

        Args:
            strategies: List of stop loss strategies to apply
        """
        self.strategies = strategies or []
        self.positions: dict[str, Position] = {}
        logger.info(f"[CompositeStopManager] Initialized with {len(self.strategies)} strategies")

    def add_strategy(self, strategy: StopLossStrategy) -> None:
        """Add a stop loss strategy."""
        self.strategies.append(strategy)
        logger.info(f"[CompositeStopManager] Added {strategy.__class__.__name__}")

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
    ) -> Position:
        """
        Register a new position for monitoring.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            entry_price: Entry price
            quantity: Position quantity

        Returns:
            Position object
        """
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            highest_price=entry_price,
            lowest_price=entry_price,
        )
        self.positions[symbol] = position
        logger.info(
            f"[CompositeStopManager] Opened {side} position: {symbol} @ {entry_price:.2f}"
        )
        return position

    def close_position(self, symbol: str) -> Optional[Position]:
        """Remove position from monitoring."""
        position = self.positions.pop(symbol, None)
        if position:
            logger.info(f"[CompositeStopManager] Closed position: {symbol}")
        return position

    def check_position(self, symbol: str, current_price: float) -> ExitSignal:
        """
        Check if position should be exited.

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            ExitSignal from first triggered strategy
        """
        position = self.positions.get(symbol)
        if not position:
            return ExitSignal(
                should_exit=False,
                message=f"No position found for {symbol}",
            )

        # Update extremes for trailing stops
        position.update_extremes(current_price)

        # Check each strategy
        for strategy in self.strategies:
            signal = strategy.check_exit(position, current_price)
            if signal.should_exit:
                logger.warning(
                    f"[CompositeStopManager] Exit signal for {symbol}: "
                    f"{signal.reason.value if signal.reason else 'unknown'} - {signal.message}"
                )
                return signal

        # No exit signal
        pnl_pct = position.get_pnl_percent(current_price)
        return ExitSignal(
            should_exit=False,
            pnl_percent=pnl_pct,
            message=f"Position {symbol} active, P&L: {pnl_pct:.1%}",
        )

    def check_all_positions(self, prices: dict[str, float]) -> dict[str, ExitSignal]:
        """
        Check all positions against current prices.

        Args:
            prices: Dictionary of symbol -> current price

        Returns:
            Dictionary of symbol -> ExitSignal
        """
        results = {}
        for symbol, position in self.positions.items():
            if symbol in prices:
                results[symbol] = self.check_position(symbol, prices[symbol])
            else:
                logger.warning(f"[CompositeStopManager] No price for {symbol}")
        return results

    def get_positions(self) -> dict[str, dict]:
        """Get all positions as dictionary."""
        return {
            symbol: {
                "symbol": p.symbol,
                "side": p.side,
                "entry_price": p.entry_price,
                "quantity": p.quantity,
                "entry_time": p.entry_time.isoformat(),
                "highest_price": p.highest_price,
                "lowest_price": p.lowest_price,
            }
            for symbol, p in self.positions.items()
        }


def create_default_stop_manager() -> CompositeStopManager:
    """
    Create a stop manager with recommended default settings.

    Returns:
        CompositeStopManager with balanced risk settings
    """
    manager = CompositeStopManager()

    # Primary: Trailing stop with 2% activation, 3% trail
    manager.add_strategy(
        TrailingStop(
            trail_pct=0.03,
            activation_pct=0.02,
            initial_stop_pct=0.05,
        )
    )

    # Secondary: Fixed take profit at 15%
    manager.add_strategy(
        FixedPercentageStop(
            stop_loss_pct=0.10,  # Wider than trailing, acts as emergency stop
            take_profit_pct=0.15,
        )
    )

    # Time limit: 5 days maximum
    manager.add_strategy(
        TimeBasedStop(
            max_holding_hours=120.0,
            fallback_stop_pct=0.05,
        )
    )

    logger.info("[create_default_stop_manager] Created with trailing + fixed + time stops")
    return manager
