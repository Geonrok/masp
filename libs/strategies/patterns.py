"""
Strategy Patterns and Utilities

Common patterns for building and composing trading strategies.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from libs.strategies.base import (
    Action,
    BaseStrategy,
    Decision,
    Signal,
    StrategyContext,
    TradeSignal,
)

logger = logging.getLogger(__name__)


# ==================== Strategy Filters ====================


class StrategyFilter(ABC):
    """
    Base class for strategy filters.

    Filters can pre-filter symbols or post-filter decisions.
    """

    @abstractmethod
    def filter_symbols(
        self,
        symbols: List[str],
        context: StrategyContext,
    ) -> List[str]:
        """Filter symbols before strategy execution."""
        pass

    @abstractmethod
    def filter_decisions(
        self,
        decisions: List[Decision],
        context: StrategyContext,
    ) -> List[Decision]:
        """Filter decisions after strategy execution."""
        pass


class VolumeFilter(StrategyFilter):
    """Filter symbols by trading volume."""

    def __init__(
        self,
        min_volume: float = 0,
        volume_key: str = "volume_24h",
    ):
        self.min_volume = min_volume
        self.volume_key = volume_key

    def filter_symbols(
        self,
        symbols: List[str],
        context: StrategyContext,
    ) -> List[str]:
        """Filter symbols with volume below threshold."""
        filtered = []
        for symbol in symbols:
            data = context.market_data.get(symbol, {})
            volume = data.get(self.volume_key, 0)
            if volume >= self.min_volume:
                filtered.append(symbol)
            else:
                logger.debug(f"[VolumeFilter] Filtered out {symbol}: volume={volume}")
        return filtered

    def filter_decisions(
        self,
        decisions: List[Decision],
        context: StrategyContext,
    ) -> List[Decision]:
        """Pass through decisions (volume filtering done at symbol level)."""
        return decisions


class PositionFilter(StrategyFilter):
    """Filter based on existing positions."""

    def __init__(
        self,
        max_positions: int = 10,
        get_positions: Optional[Callable[[], Dict[str, float]]] = None,
    ):
        self.max_positions = max_positions
        self._get_positions = get_positions or (lambda: {})

    def filter_symbols(
        self,
        symbols: List[str],
        context: StrategyContext,
    ) -> List[str]:
        """Return all symbols (filtering done at decision level)."""
        return symbols

    def filter_decisions(
        self,
        decisions: List[Decision],
        context: StrategyContext,
    ) -> List[Decision]:
        """Limit new BUY decisions based on position count."""
        positions = self._get_positions()
        position_count = sum(1 for v in positions.values() if v > 0)

        filtered = []
        for decision in decisions:
            if decision.action == Action.BUY:
                if position_count >= self.max_positions:
                    logger.info(
                        f"[PositionFilter] Blocked BUY for {decision.symbol}: "
                        f"max positions ({self.max_positions}) reached"
                    )
                    continue
                position_count += 1
            filtered.append(decision)

        return filtered


class TimeFilter(StrategyFilter):
    """Filter based on trading hours."""

    def __init__(
        self,
        trading_hours: Optional[List[tuple]] = None,
        blocked_days: Optional[List[int]] = None,
    ):
        """
        Args:
            trading_hours: List of (start_hour, end_hour) tuples
            blocked_days: List of weekday numbers (0=Monday, 6=Sunday)
        """
        self.trading_hours = trading_hours or [(0, 24)]
        self.blocked_days = set(blocked_days or [])

    def _is_trading_time(self) -> bool:
        """Check if current time is within trading hours."""
        now = datetime.now()

        if now.weekday() in self.blocked_days:
            return False

        current_hour = now.hour
        for start, end in self.trading_hours:
            if start <= current_hour < end:
                return True

        return False

    def filter_symbols(
        self,
        symbols: List[str],
        context: StrategyContext,
    ) -> List[str]:
        """Return empty list if outside trading hours."""
        if not self._is_trading_time():
            logger.info("[TimeFilter] Outside trading hours, skipping all symbols")
            return []
        return symbols

    def filter_decisions(
        self,
        decisions: List[Decision],
        context: StrategyContext,
    ) -> List[Decision]:
        """Block trade decisions outside trading hours."""
        if not self._is_trading_time():
            return [
                Decision(d.symbol, Action.HOLD, notes="Outside trading hours")
                for d in decisions
            ]
        return decisions


# ==================== Strategy Composition ====================


class CompositeStrategy(BaseStrategy):
    """
    Combines multiple strategies with configurable voting.

    Example:
        strategy = CompositeStrategy(
            strategies=[kama_strategy, macd_strategy],
            voting_mode="majority",
        )
    """

    class VotingMode(Enum):
        MAJORITY = "majority"  # Action with most votes
        UNANIMOUS = "unanimous"  # All must agree
        ANY_BUY = "any_buy"  # BUY if any strategy says BUY
        WEIGHTED = "weighted"  # Weight by strategy.weight attribute

    def __init__(
        self,
        strategies: List[BaseStrategy],
        voting_mode: str = "majority",
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            strategies: List of strategies to combine
            voting_mode: How to combine decisions
            weights: Optional weights for each strategy (by name)
        """
        self.strategies = strategies
        self.voting_mode = self.VotingMode(voting_mode)
        self.weights = weights or {}

    def execute(self, context: StrategyContext) -> List[Decision]:
        """Execute all strategies and combine decisions."""
        # Collect decisions from all strategies
        all_decisions: Dict[str, List[tuple]] = {}

        for strategy in self.strategies:
            try:
                decisions = strategy.execute(context)
                weight = self.weights.get(strategy.name, 1.0)

                for decision in decisions:
                    if decision.symbol not in all_decisions:
                        all_decisions[decision.symbol] = []
                    all_decisions[decision.symbol].append((decision, weight))

            except Exception as e:
                logger.error(f"[CompositeStrategy] {strategy.name} failed: {e}")

        # Combine decisions per symbol
        combined = []
        for symbol, decision_list in all_decisions.items():
            combined_decision = self._combine_decisions(symbol, decision_list)
            combined.append(combined_decision)

        return combined

    def _combine_decisions(
        self,
        symbol: str,
        decisions: List[tuple],
    ) -> Decision:
        """Combine decisions for a single symbol."""
        if self.voting_mode == self.VotingMode.UNANIMOUS:
            return self._unanimous_vote(symbol, decisions)
        elif self.voting_mode == self.VotingMode.ANY_BUY:
            return self._any_buy_vote(symbol, decisions)
        elif self.voting_mode == self.VotingMode.WEIGHTED:
            return self._weighted_vote(symbol, decisions)
        else:  # MAJORITY
            return self._majority_vote(symbol, decisions)

    def _majority_vote(
        self,
        symbol: str,
        decisions: List[tuple],
    ) -> Decision:
        """Majority voting."""
        action_counts = {Action.BUY: 0, Action.SELL: 0, Action.HOLD: 0}

        for decision, _ in decisions:
            if decision.action in action_counts:
                action_counts[decision.action] += 1

        winner = max(action_counts.items(), key=lambda x: x[1])
        return Decision(
            symbol=symbol,
            action=winner[0],
            notes=f"Majority: {action_counts}",
            metrics={"votes": dict(action_counts)},
        )

    def _unanimous_vote(
        self,
        symbol: str,
        decisions: List[tuple],
    ) -> Decision:
        """Unanimous voting (all must agree for non-HOLD)."""
        actions = [d.action for d, _ in decisions]
        unique_actions = set(actions)

        if len(unique_actions) == 1 and Action.HOLD not in unique_actions:
            return Decision(
                symbol=symbol,
                action=actions[0],
                notes="Unanimous agreement",
            )

        return Decision(
            symbol=symbol,
            action=Action.HOLD,
            notes=f"No unanimous agreement: {actions}",
        )

    def _any_buy_vote(
        self,
        symbol: str,
        decisions: List[tuple],
    ) -> Decision:
        """Any BUY signal triggers buy."""
        for decision, _ in decisions:
            if decision.action == Action.BUY:
                return Decision(
                    symbol=symbol,
                    action=Action.BUY,
                    notes="At least one strategy says BUY",
                )

        # Check for SELL
        sell_count = sum(1 for d, _ in decisions if d.action == Action.SELL)
        if sell_count > len(decisions) // 2:
            return Decision(symbol=symbol, action=Action.SELL)

        return Decision(symbol=symbol, action=Action.HOLD)

    def _weighted_vote(
        self,
        symbol: str,
        decisions: List[tuple],
    ) -> Decision:
        """Weighted voting."""
        action_weights = {Action.BUY: 0.0, Action.SELL: 0.0, Action.HOLD: 0.0}

        for decision, weight in decisions:
            if decision.action in action_weights:
                action_weights[decision.action] += weight

        winner = max(action_weights.items(), key=lambda x: x[1])
        return Decision(
            symbol=symbol,
            action=winner[0],
            notes=f"Weighted: {action_weights}",
            metrics={"weights": dict(action_weights)},
        )


# ==================== Strategy Decorators ====================


class StrategyWrapper(BaseStrategy):
    """Base wrapper for strategy decorators."""

    def __init__(self, strategy: BaseStrategy):
        self._strategy = strategy

    @property
    def name(self) -> str:
        return self._strategy.name

    def execute(self, context: StrategyContext) -> List[Decision]:
        return self._strategy.execute(context)


class FilteredStrategy(StrategyWrapper):
    """Apply filters to a strategy."""

    def __init__(
        self,
        strategy: BaseStrategy,
        filters: Sequence[StrategyFilter],
    ):
        super().__init__(strategy)
        self.filters = list(filters)

    def execute(self, context: StrategyContext) -> List[Decision]:
        """Execute with filtering."""
        # Pre-filter symbols
        symbols = context.symbols
        for f in self.filters:
            symbols = f.filter_symbols(symbols, context)

        if not symbols:
            return [
                Decision(s, Action.SKIP, notes="Filtered out")
                for s in context.symbols
            ]

        # Execute with filtered symbols
        filtered_context = StrategyContext(
            config=context.config,
            run_id=context.run_id,
            event_logger=context.event_logger,
            symbols=symbols,
            market_data=context.market_data,
            extra=context.extra,
        )

        decisions = self._strategy.execute(filtered_context)

        # Post-filter decisions
        for f in self.filters:
            decisions = f.filter_decisions(decisions, context)

        return decisions


class CooldownStrategy(StrategyWrapper):
    """Add cooldown period between trades for same symbol."""

    def __init__(
        self,
        strategy: BaseStrategy,
        cooldown_minutes: int = 60,
    ):
        super().__init__(strategy)
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self._last_trade: Dict[str, datetime] = {}

    def execute(self, context: StrategyContext) -> List[Decision]:
        """Execute with cooldown filtering."""
        decisions = self._strategy.execute(context)
        now = datetime.now()

        filtered = []
        for decision in decisions:
            if decision.action in (Action.BUY, Action.SELL):
                last_trade = self._last_trade.get(decision.symbol)
                if last_trade and (now - last_trade) < self.cooldown:
                    remaining = self.cooldown - (now - last_trade)
                    filtered.append(Decision(
                        symbol=decision.symbol,
                        action=Action.HOLD,
                        notes=f"Cooldown: {remaining.seconds}s remaining",
                    ))
                    continue

                # Record trade time
                self._last_trade[decision.symbol] = now

            filtered.append(decision)

        return filtered


class LoggingStrategy(StrategyWrapper):
    """Add detailed logging to strategy execution."""

    def execute(self, context: StrategyContext) -> List[Decision]:
        """Execute with logging."""
        logger.info(
            f"[{self.name}] Executing for {len(context.symbols)} symbols"
        )

        start = datetime.now()
        decisions = self._strategy.execute(context)
        duration = (datetime.now() - start).total_seconds()

        # Log summary
        buy_count = sum(1 for d in decisions if d.action == Action.BUY)
        sell_count = sum(1 for d in decisions if d.action == Action.SELL)
        hold_count = sum(1 for d in decisions if d.action == Action.HOLD)

        logger.info(
            f"[{self.name}] Complete in {duration:.2f}s: "
            f"BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}"
        )

        return decisions


# ==================== Strategy State Manager ====================


@dataclass
class StrategyStateSnapshot:
    """Snapshot of strategy state."""

    name: str
    timestamp: datetime
    is_active: bool
    position_count: int
    total_value: float
    metrics: Dict[str, Any] = field(default_factory=dict)


class StrategyStateManager:
    """
    Manages strategy state persistence and recovery.

    Example:
        manager = StrategyStateManager()

        # Save state
        manager.save_state("kama_strategy", {
            "positions": {"BTC": 0.5},
            "metrics": {"sharpe": 1.2},
        })

        # Load state
        state = manager.load_state("kama_strategy")
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: Path for state persistence (optional)
        """
        self.storage_path = storage_path
        self._states: Dict[str, Dict[str, Any]] = {}

    def save_state(
        self,
        strategy_name: str,
        state: Dict[str, Any],
    ) -> None:
        """Save strategy state."""
        state["_saved_at"] = datetime.now().isoformat()
        self._states[strategy_name] = state

        if self.storage_path:
            self._persist_to_file(strategy_name, state)

        logger.debug(f"[StateManager] Saved state for {strategy_name}")

    def load_state(
        self,
        strategy_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Load strategy state."""
        if strategy_name in self._states:
            return self._states[strategy_name]

        if self.storage_path:
            state = self._load_from_file(strategy_name)
            if state:
                self._states[strategy_name] = state
                return state

        return None

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all strategy states."""
        return dict(self._states)

    def clear_state(self, strategy_name: str) -> None:
        """Clear strategy state."""
        self._states.pop(strategy_name, None)

    def _persist_to_file(
        self,
        strategy_name: str,
        state: Dict[str, Any],
    ) -> None:
        """Persist state to file."""
        import json
        import os

        os.makedirs(self.storage_path, exist_ok=True)
        filepath = os.path.join(self.storage_path, f"{strategy_name}.json")

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_from_file(
        self,
        strategy_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Load state from file."""
        import json
        import os

        if not self.storage_path:
            return None

        filepath = os.path.join(self.storage_path, f"{strategy_name}.json")

        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)

        return None


# ==================== Convenience Functions ====================


def with_filters(
    strategy: BaseStrategy,
    filters: Sequence[StrategyFilter],
) -> FilteredStrategy:
    """Wrap strategy with filters."""
    return FilteredStrategy(strategy, filters)


def with_cooldown(
    strategy: BaseStrategy,
    cooldown_minutes: int = 60,
) -> CooldownStrategy:
    """Wrap strategy with cooldown."""
    return CooldownStrategy(strategy, cooldown_minutes)


def with_logging(strategy: BaseStrategy) -> LoggingStrategy:
    """Wrap strategy with logging."""
    return LoggingStrategy(strategy)
