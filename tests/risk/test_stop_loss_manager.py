"""
Tests for Stop Loss Manager

Validates position-level exit logic:
- Fixed percentage stop loss / take profit
- Trailing stop
- ATR-based dynamic stop
- Time-based stop
- Composite stop manager
"""

import pytest
from datetime import datetime, timedelta

from libs.risk.stop_loss_manager import (
    Position,
    ExitReason,
    FixedPercentageStop,
    TrailingStop,
    ATRBasedStop,
    TimeBasedStop,
    CompositeStopManager,
    create_default_stop_manager,
)


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test basic position creation."""
        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
        )
        assert pos.symbol == "BTC/KRW"
        assert pos.side == "long"
        assert pos.entry_price == 50000000
        assert pos.quantity == 0.1

    def test_pnl_percent_long(self):
        """Test P&L calculation for long position."""
        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
        )
        # 10% gain
        assert abs(pos.get_pnl_percent(55000000) - 0.10) < 0.001
        # 5% loss
        assert abs(pos.get_pnl_percent(47500000) - (-0.05)) < 0.001

    def test_pnl_percent_short(self):
        """Test P&L calculation for short position."""
        pos = Position(
            symbol="BTC/KRW",
            side="short",
            entry_price=50000000,
            quantity=0.1,
        )
        # 10% gain (price dropped)
        assert abs(pos.get_pnl_percent(45000000) - 0.10) < 0.001
        # 5% loss (price rose)
        assert abs(pos.get_pnl_percent(52500000) - (-0.05)) < 0.001

    def test_update_extremes_long(self):
        """Test extreme price tracking for long position."""
        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
            highest_price=50000000,
        )
        pos.update_extremes(55000000)
        assert pos.highest_price == 55000000

        pos.update_extremes(53000000)
        assert pos.highest_price == 55000000  # Should not decrease

    def test_update_extremes_short(self):
        """Test extreme price tracking for short position."""
        pos = Position(
            symbol="BTC/KRW",
            side="short",
            entry_price=50000000,
            quantity=0.1,
            lowest_price=50000000,
        )
        pos.update_extremes(45000000)
        assert pos.lowest_price == 45000000

        pos.update_extremes(47000000)
        assert pos.lowest_price == 45000000  # Should not increase


class TestFixedPercentageStop:
    """Tests for FixedPercentageStop strategy."""

    def test_stop_loss_triggered_long(self):
        """Test stop loss triggers for long position."""
        stop = FixedPercentageStop(stop_loss_pct=0.05, take_profit_pct=0.10)
        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
        )

        # Price dropped 6% - should trigger stop loss
        signal = stop.check_exit(pos, 47000000)
        assert signal.should_exit is True
        assert signal.reason == ExitReason.STOP_LOSS

    def test_take_profit_triggered_long(self):
        """Test take profit triggers for long position."""
        stop = FixedPercentageStop(stop_loss_pct=0.05, take_profit_pct=0.10)
        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
        )

        # Price rose 11% - should trigger take profit
        signal = stop.check_exit(pos, 55500000)
        assert signal.should_exit is True
        assert signal.reason == ExitReason.TAKE_PROFIT

    def test_no_exit_within_range(self):
        """Test no exit when price within range."""
        stop = FixedPercentageStop(stop_loss_pct=0.05, take_profit_pct=0.10)
        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
        )

        # Price changed 3% - should not trigger
        signal = stop.check_exit(pos, 51500000)
        assert signal.should_exit is False

    def test_stop_loss_short_position(self):
        """Test stop loss for short position."""
        stop = FixedPercentageStop(stop_loss_pct=0.05, take_profit_pct=0.10)
        pos = Position(
            symbol="BTC/KRW",
            side="short",
            entry_price=50000000,
            quantity=0.1,
        )

        # Price rose 6% - should trigger stop loss for short
        signal = stop.check_exit(pos, 53000000)
        assert signal.should_exit is True
        assert signal.reason == ExitReason.STOP_LOSS

    def test_invalid_stop_loss_pct(self):
        """Test validation of stop loss percentage."""
        with pytest.raises(ValueError):
            FixedPercentageStop(stop_loss_pct=0.6)  # Too high

        with pytest.raises(ValueError):
            FixedPercentageStop(stop_loss_pct=0)  # Zero


class TestTrailingStop:
    """Tests for TrailingStop strategy."""

    def test_trailing_stop_activated_and_triggered(self):
        """Test trailing stop activation and triggering."""
        stop = TrailingStop(
            trail_pct=0.03,
            activation_pct=0.02,
            initial_stop_pct=0.05,
        )
        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
            highest_price=50000000,
        )

        # Price rises 5% - activates trailing
        pos.update_extremes(52500000)
        signal = stop.check_exit(pos, 52500000)
        assert signal.should_exit is False
        assert "activated" in signal.message

        # Price drops to 50925000 (3% below high of 52500000)
        signal = stop.check_exit(pos, 50925000)
        assert signal.should_exit is True
        assert signal.reason == ExitReason.TRAILING_STOP

    def test_initial_stop_before_activation(self):
        """Test initial stop triggers before trailing activation."""
        stop = TrailingStop(
            trail_pct=0.03,
            activation_pct=0.05,  # Need 5% profit to activate
            initial_stop_pct=0.03,  # 3% initial stop
        )
        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
            highest_price=50000000,
        )

        # Price dropped 4% - should trigger initial stop
        signal = stop.check_exit(pos, 48000000)
        assert signal.should_exit is True
        assert signal.reason == ExitReason.STOP_LOSS

    def test_trailing_stop_short_position(self):
        """Test trailing stop for short position."""
        stop = TrailingStop(
            trail_pct=0.03,
            activation_pct=0.02,
            initial_stop_pct=0.05,
        )
        pos = Position(
            symbol="BTC/KRW",
            side="short",
            entry_price=50000000,
            quantity=0.1,
            lowest_price=50000000,
        )

        # Price drops 5% - activates trailing
        pos.update_extremes(47500000)
        signal = stop.check_exit(pos, 47500000)
        assert signal.should_exit is False

        # Price rises to 48925000 (3% above low of 47500000)
        signal = stop.check_exit(pos, 48925000)
        assert signal.should_exit is True
        assert signal.reason == ExitReason.TRAILING_STOP


class TestATRBasedStop:
    """Tests for ATRBasedStop strategy."""

    def test_atr_stop_triggered(self):
        """Test ATR-based stop loss."""
        stop = ATRBasedStop(atr_multiplier=2.0, take_profit_multiplier=3.0)
        stop.set_atr(1000000)  # 1M KRW ATR

        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
        )

        # Price dropped below entry - 2*ATR (48M) - should trigger
        signal = stop.check_exit(pos, 47900000)
        assert signal.should_exit is True
        assert signal.reason == ExitReason.STOP_LOSS

    def test_atr_take_profit_triggered(self):
        """Test ATR-based take profit."""
        stop = ATRBasedStop(atr_multiplier=2.0, take_profit_multiplier=3.0)
        stop.set_atr(1000000)  # 1M KRW ATR

        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
        )

        # Price rose above entry + 3*ATR (53M) - should trigger TP
        signal = stop.check_exit(pos, 53100000)
        assert signal.should_exit is True
        assert signal.reason == ExitReason.TAKE_PROFIT

    def test_atr_not_set_fallback(self):
        """Test fallback when ATR not set."""
        stop = ATRBasedStop(atr_multiplier=2.0)
        # Don't call set_atr - should use 2% fallback

        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
        )

        # Should still work with fallback ATR
        signal = stop.check_exit(pos, 50000000)
        assert signal.should_exit is False


class TestTimeBasedStop:
    """Tests for TimeBasedStop strategy."""

    def test_time_stop_triggered(self):
        """Test time-based stop triggers after holding period."""
        stop = TimeBasedStop(max_holding_hours=24.0, fallback_stop_pct=0.05)

        # Create position 25 hours ago
        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
            entry_time=datetime.now() - timedelta(hours=25),
        )

        signal = stop.check_exit(pos, 50000000)
        assert signal.should_exit is True
        assert signal.reason == ExitReason.TIME_STOP

    def test_time_stop_not_triggered_within_period(self):
        """Test time stop doesn't trigger within holding period."""
        stop = TimeBasedStop(max_holding_hours=24.0, fallback_stop_pct=0.05)

        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
            entry_time=datetime.now() - timedelta(hours=12),
        )

        signal = stop.check_exit(pos, 50000000)
        assert signal.should_exit is False
        assert "remaining" in signal.message

    def test_fallback_stop_during_holding(self):
        """Test fallback stop loss during holding period."""
        stop = TimeBasedStop(max_holding_hours=24.0, fallback_stop_pct=0.05)

        pos = Position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
            entry_time=datetime.now() - timedelta(hours=1),
        )

        # Price dropped 6% - should trigger fallback stop
        signal = stop.check_exit(pos, 47000000)
        assert signal.should_exit is True
        assert signal.reason == ExitReason.STOP_LOSS


class TestCompositeStopManager:
    """Tests for CompositeStopManager."""

    def test_open_and_close_position(self):
        """Test position management."""
        manager = CompositeStopManager()

        pos = manager.open_position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
        )
        assert "BTC/KRW" in manager.positions
        assert pos.symbol == "BTC/KRW"

        closed = manager.close_position("BTC/KRW")
        assert closed is not None
        assert "BTC/KRW" not in manager.positions

    def test_composite_first_signal_wins(self):
        """Test that first triggered strategy wins."""
        manager = CompositeStopManager()

        # Add tight stop first
        manager.add_strategy(
            FixedPercentageStop(stop_loss_pct=0.02, take_profit_pct=0.05)
        )
        # Add wider stop second
        manager.add_strategy(
            FixedPercentageStop(stop_loss_pct=0.05, take_profit_pct=0.10)
        )

        manager.open_position(
            symbol="BTC/KRW",
            side="long",
            entry_price=50000000,
            quantity=0.1,
        )

        # 3% drop - first strategy triggers, second doesn't
        signal = manager.check_position("BTC/KRW", 48500000)
        assert signal.should_exit is True
        assert signal.reason == ExitReason.STOP_LOSS

    def test_check_all_positions(self):
        """Test checking all positions at once."""
        manager = CompositeStopManager()
        manager.add_strategy(
            FixedPercentageStop(stop_loss_pct=0.05, take_profit_pct=0.10)
        )

        manager.open_position("BTC/KRW", "long", 50000000, 0.1)
        manager.open_position("ETH/KRW", "long", 3000000, 1.0)

        prices = {
            "BTC/KRW": 47000000,  # -6% - should exit
            "ETH/KRW": 3100000,  # +3% - should hold
        }

        results = manager.check_all_positions(prices)

        assert results["BTC/KRW"].should_exit is True
        assert results["ETH/KRW"].should_exit is False

    def test_get_positions(self):
        """Test getting positions as dictionary."""
        manager = CompositeStopManager()
        manager.open_position("BTC/KRW", "long", 50000000, 0.1)

        positions = manager.get_positions()
        assert "BTC/KRW" in positions
        assert positions["BTC/KRW"]["side"] == "long"
        assert positions["BTC/KRW"]["entry_price"] == 50000000

    def test_nonexistent_position(self):
        """Test checking non-existent position."""
        manager = CompositeStopManager()
        signal = manager.check_position("NONEXISTENT", 100)
        assert signal.should_exit is False
        assert "No position found" in signal.message


class TestCreateDefaultStopManager:
    """Tests for create_default_stop_manager factory."""

    def test_default_manager_created(self):
        """Test default manager has expected strategies."""
        manager = create_default_stop_manager()
        assert len(manager.strategies) == 3  # Trailing + Fixed + Time

    def test_default_manager_works(self):
        """Test default manager functions correctly."""
        manager = create_default_stop_manager()
        manager.open_position("BTC/KRW", "long", 50000000, 0.1)

        # Normal price - no exit
        signal = manager.check_position("BTC/KRW", 50500000)
        assert signal.should_exit is False

        # Large loss - should exit
        signal = manager.check_position("BTC/KRW", 42000000)
        assert signal.should_exit is True
