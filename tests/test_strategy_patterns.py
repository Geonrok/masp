"""
Tests for strategy patterns module.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from libs.strategies.base import Action, Decision, StrategyContext, BaseStrategy
from libs.strategies.patterns import (
    VolumeFilter,
    PositionFilter,
    TimeFilter,
    CompositeStrategy,
    FilteredStrategy,
    CooldownStrategy,
    LoggingStrategy,
    StrategyStateManager,
    with_filters,
    with_cooldown,
    with_logging,
)


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    def __init__(self, name: str, decisions: list):
        self._name = name
        self._decisions = decisions

    @property
    def name(self) -> str:
        return self._name

    def execute(self, context) -> list:
        return self._decisions


@pytest.fixture
def mock_context():
    """Create mock strategy context."""
    config = MagicMock()
    event_logger = MagicMock()

    return StrategyContext(
        config=config,
        run_id="test_run",
        event_logger=event_logger,
        symbols=["BTC", "ETH", "XRP"],
        market_data={
            "BTC": {"volume_24h": 1000000},
            "ETH": {"volume_24h": 500000},
            "XRP": {"volume_24h": 100},
        },
    )


class TestVolumeFilter:
    """Tests for VolumeFilter."""

    def test_filter_by_volume(self, mock_context):
        """Test filtering symbols by volume."""
        filter_ = VolumeFilter(min_volume=100000)
        filtered = filter_.filter_symbols(mock_context.symbols, mock_context)

        assert "BTC" in filtered
        assert "ETH" in filtered
        assert "XRP" not in filtered

    def test_filter_all_passed(self, mock_context):
        """Test when all symbols pass filter."""
        filter_ = VolumeFilter(min_volume=0)
        filtered = filter_.filter_symbols(mock_context.symbols, mock_context)

        assert len(filtered) == 3


class TestPositionFilter:
    """Tests for PositionFilter."""

    def test_limit_buy_decisions(self, mock_context):
        """Test limiting BUY decisions based on position count."""
        positions = {"BTC": 1.0, "ETH": 0.5}
        filter_ = PositionFilter(
            max_positions=2,
            get_positions=lambda: positions,
        )

        decisions = [
            Decision("XRP", Action.BUY),
            Decision("SOL", Action.BUY),
        ]

        filtered = filter_.filter_decisions(decisions, mock_context)

        # Should only allow 0 new BUY (already at max)
        buy_count = sum(1 for d in filtered if d.action == Action.BUY)
        assert buy_count == 0

    def test_allow_sell_decisions(self, mock_context):
        """Test SELL decisions are not blocked."""
        filter_ = PositionFilter(max_positions=1, get_positions=lambda: {"BTC": 1.0})

        decisions = [
            Decision("BTC", Action.SELL),
            Decision("ETH", Action.BUY),
        ]

        filtered = filter_.filter_decisions(decisions, mock_context)

        assert any(d.symbol == "BTC" and d.action == Action.SELL for d in filtered)


class TestTimeFilter:
    """Tests for TimeFilter."""

    def test_trading_hours(self, mock_context):
        """Test trading hours filtering."""
        # Always pass for this test (9 AM - 5 PM)
        filter_ = TimeFilter(trading_hours=[(0, 24)])
        filtered = filter_.filter_symbols(mock_context.symbols, mock_context)

        assert len(filtered) > 0


class TestCompositeStrategy:
    """Tests for CompositeStrategy."""

    def test_majority_voting(self, mock_context):
        """Test majority voting mode."""
        s1 = MockStrategy("s1", [Decision("BTC", Action.BUY)])
        s2 = MockStrategy("s2", [Decision("BTC", Action.BUY)])
        s3 = MockStrategy("s3", [Decision("BTC", Action.SELL)])

        composite = CompositeStrategy([s1, s2, s3], voting_mode="majority")
        decisions = composite.execute(mock_context)

        assert len(decisions) == 1
        assert decisions[0].action == Action.BUY

    def test_unanimous_voting(self, mock_context):
        """Test unanimous voting mode."""
        s1 = MockStrategy("s1", [Decision("BTC", Action.BUY)])
        s2 = MockStrategy("s2", [Decision("BTC", Action.SELL)])

        composite = CompositeStrategy([s1, s2], voting_mode="unanimous")
        decisions = composite.execute(mock_context)

        # No unanimous agreement, should HOLD
        assert decisions[0].action == Action.HOLD

    def test_any_buy_voting(self, mock_context):
        """Test any_buy voting mode."""
        s1 = MockStrategy("s1", [Decision("BTC", Action.HOLD)])
        s2 = MockStrategy("s2", [Decision("BTC", Action.BUY)])
        s3 = MockStrategy("s3", [Decision("BTC", Action.HOLD)])

        composite = CompositeStrategy([s1, s2, s3], voting_mode="any_buy")
        decisions = composite.execute(mock_context)

        assert decisions[0].action == Action.BUY

    def test_weighted_voting(self, mock_context):
        """Test weighted voting mode."""
        s1 = MockStrategy("s1", [Decision("BTC", Action.BUY)])
        s2 = MockStrategy("s2", [Decision("BTC", Action.SELL)])

        composite = CompositeStrategy(
            [s1, s2],
            voting_mode="weighted",
            weights={"s1": 2.0, "s2": 1.0},
        )
        decisions = composite.execute(mock_context)

        # s1 has higher weight, should win
        assert decisions[0].action == Action.BUY


class ContextAwareMockStrategy(BaseStrategy):
    """Mock strategy that uses context.symbols."""

    def __init__(self, name: str, action: Action = Action.BUY):
        self._name = name
        self._action = action

    @property
    def name(self) -> str:
        return self._name

    def execute(self, context) -> list:
        # Only return decisions for symbols in context
        return [Decision(s, self._action) for s in context.symbols]


class TestFilteredStrategy:
    """Tests for FilteredStrategy."""

    def test_with_volume_filter(self, mock_context):
        """Test strategy with volume filter."""
        # Use a context-aware strategy that respects filtered symbols
        strategy = ContextAwareMockStrategy("test", Action.BUY)

        volume_filter = VolumeFilter(min_volume=100000)
        filtered_strategy = FilteredStrategy(strategy, [volume_filter])

        # Execute (XRP should be filtered out due to low volume < 100000)
        mock_context.symbols = ["BTC", "ETH", "XRP"]
        decisions = filtered_strategy.execute(mock_context)

        symbols = [d.symbol for d in decisions]
        # XRP filtered at symbol level, so not in decisions
        assert "XRP" not in symbols
        assert "BTC" in symbols
        assert "ETH" in symbols


class TestCooldownStrategy:
    """Tests for CooldownStrategy."""

    def test_cooldown_blocks_rapid_trades(self, mock_context):
        """Test cooldown prevents rapid repeated trades."""
        strategy = MockStrategy("test", [Decision("BTC", Action.BUY)])
        cooldown = CooldownStrategy(strategy, cooldown_minutes=60)

        # First execution
        decisions1 = cooldown.execute(mock_context)
        assert decisions1[0].action == Action.BUY

        # Second execution should be blocked by cooldown
        decisions2 = cooldown.execute(mock_context)
        assert decisions2[0].action == Action.HOLD
        assert "Cooldown" in decisions2[0].notes


class TestLoggingStrategy:
    """Tests for LoggingStrategy."""

    def test_logging_wrapper(self, mock_context, caplog):
        """Test logging wrapper adds logs."""
        strategy = MockStrategy("test", [Decision("BTC", Action.BUY)])
        logged = LoggingStrategy(strategy)

        import logging

        with caplog.at_level(logging.INFO):
            decisions = logged.execute(mock_context)

        assert len(decisions) == 1
        assert "Executing" in caplog.text or "Complete" in caplog.text


class TestStrategyStateManager:
    """Tests for StrategyStateManager."""

    def test_save_and_load_state(self):
        """Test saving and loading state."""
        manager = StrategyStateManager()

        state = {"positions": {"BTC": 1.0}, "pnl": 100.0}
        manager.save_state("test_strategy", state)

        loaded = manager.load_state("test_strategy")
        assert loaded["positions"] == {"BTC": 1.0}
        assert loaded["pnl"] == 100.0

    def test_clear_state(self):
        """Test clearing state."""
        manager = StrategyStateManager()

        manager.save_state("test", {"data": 1})
        manager.clear_state("test")

        assert manager.load_state("test") is None

    def test_get_all_states(self):
        """Test getting all states."""
        manager = StrategyStateManager()

        manager.save_state("s1", {"data": 1})
        manager.save_state("s2", {"data": 2})

        states = manager.get_all_states()
        assert "s1" in states
        assert "s2" in states


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_with_filters(self, mock_context):
        """Test with_filters function."""
        strategy = MockStrategy("test", [Decision("BTC", Action.BUY)])
        filtered = with_filters(strategy, [VolumeFilter(min_volume=0)])

        assert isinstance(filtered, FilteredStrategy)
        decisions = filtered.execute(mock_context)
        assert len(decisions) > 0

    def test_with_cooldown(self, mock_context):
        """Test with_cooldown function."""
        strategy = MockStrategy("test", [Decision("BTC", Action.BUY)])
        with_cd = with_cooldown(strategy, cooldown_minutes=30)

        assert isinstance(with_cd, CooldownStrategy)

    def test_with_logging(self, mock_context):
        """Test with_logging function."""
        strategy = MockStrategy("test", [Decision("BTC", Action.BUY)])
        logged = with_logging(strategy)

        assert isinstance(logged, LoggingStrategy)
