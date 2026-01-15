"""
Strategy base interfaces and models.

Supports decision-based and signal-based strategies.
"""
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from libs.core.config import Config
from libs.core.event_logger import EventLogger


class Action(str, Enum):
    """Trading action types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SKIP = "SKIP"


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Decision:
    """
    Strategy decision for a symbol.

    Attributes:
        symbol: The trading symbol
        action: BUY/SELL/HOLD/SKIP
        notes: Human-readable explanation
        metrics: Optional numerical metrics that led to decision
    """
    symbol: str
    action: Action
    notes: Optional[str] = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert decision to dictionary."""
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "notes": self.notes,
            "metrics": self.metrics,
        }


@dataclass
class StrategyContext:
    """
    Context passed to strategy execution.
    Contains all information needed for strategy to make decisions.
    """
    config: Config
    run_id: str
    event_logger: EventLogger
    symbols: list[str]

    # Market data (populated by adapters)
    market_data: dict[str, Any] = field(default_factory=dict)

    # Additional context
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeSignal:
    """Signal output for signal-based strategies."""
    symbol: str
    signal: Signal
    price: float
    timestamp: datetime
    reason: str = ""
    strength: float = 1.0


@dataclass
class StrategyState:
    """Strategy runtime state snapshot."""
    name: str
    is_running: bool
    gate_status: bool
    positions: dict[str, float]
    last_update: datetime


class BaseStrategy(ABC):
    """
    Base class for all strategies.

    Decision-based strategies should override execute().
    Signal-based strategies should override generate_signals().
    """

    # Unique identifier for this strategy
    strategy_id: str = "base_strategy"

    # Human-readable name
    name: str = "Base Strategy"

    # Strategy version
    version: str = "1.0.0"

    # Description
    description: str = "Base strategy class"

    def __init__(self, name: Optional[str] = None, config: Optional[dict] = None):
        """Initialize strategy."""
        if name:
            self.name = name
        self.config = config or {}
        self._is_running = False
        self._positions: dict[str, float] = {}

    def execute(self, ctx: StrategyContext) -> list[Decision]:
        """
        Execute strategy and return decisions.

        Args:
            ctx: Strategy context with config, market data, etc.

        Returns:
            List of Decision objects, one per symbol
        """
        signals = self.generate_signals(ctx.symbols)
        decisions: list[Decision] = []
        for signal in signals:
            if signal.signal == Signal.BUY:
                action = Action.BUY
            elif signal.signal == Signal.SELL:
                action = Action.SELL
            else:
                action = Action.HOLD
            decisions.append(
                Decision(
                    symbol=signal.symbol,
                    action=action,
                    notes=signal.reason,
                    metrics={"price": signal.price, "strength": signal.strength},
                )
            )
        return decisions

    def generate_signals(self, symbols: list[str]) -> list[TradeSignal]:
        """Generate trade signals for a list of symbols."""
        raise NotImplementedError("generate_signals not implemented")

    def check_gate(self) -> bool:
        """Check gate conditions for signal-based strategies."""
        return True

    def get_state(self) -> StrategyState:
        """Return current strategy state."""
        return StrategyState(
            name=self.name,
            is_running=self._is_running,
            gate_status=self.check_gate(),
            positions=self._positions.copy(),
            last_update=datetime.now(),
        )

    def update_position(self, symbol: str, quantity: float) -> None:
        """Update position quantity for a symbol."""
        if quantity <= 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = quantity

    def get_position(self, symbol: str) -> float:
        """Get position quantity for a symbol."""
        return self._positions.get(symbol, 0.0)

    def has_position(self, symbol: str) -> bool:
        """Check if a position exists for a symbol."""
        return self.get_position(symbol) > 0

    def start(self) -> None:
        """Mark strategy as running."""
        self._is_running = True

    def stop(self) -> None:
        """Mark strategy as stopped."""
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.strategy_id} v{self.version}>"
