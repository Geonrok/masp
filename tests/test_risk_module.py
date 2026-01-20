"""
Tests for the Risk Management Module.
"""

import pytest
from datetime import datetime, timedelta

from libs.risk.position_sizer import (
    FixedFractionalSizer,
    KellyCriterionSizer,
    VolatilityBasedSizer,
)
from libs.risk.drawdown_guard import DrawdownGuard, RiskStatus


class TestFixedFractionalSizer:
    """Tests for Fixed Fractional Position Sizing."""

    def test_basic_calculation(self):
        """Test basic position size calculation."""
        # Use max_position_percent=0.6 to avoid cap (position_value=50000, cap=60000)
        sizer = FixedFractionalSizer(risk_percent=0.02, max_position_percent=0.6)
        result = sizer.calculate(
            capital=100000,
            entry_price=50000,
            stop_loss_price=48000,
        )

        # Risk = 100000 * 0.02 = 2000
        # Risk per unit = 50000 - 48000 = 2000
        # Quantity = 2000 / 2000 = 1
        assert result.quantity == pytest.approx(1.0, rel=0.01)
        assert result.risk_amount == pytest.approx(2000, rel=0.01)
        assert result.method == "fixed_fractional"

    def test_max_position_cap(self):
        """Test that position is capped at max_position_percent."""
        sizer = FixedFractionalSizer(risk_percent=0.02, max_position_percent=0.10)
        result = sizer.calculate(
            capital=100000,
            entry_price=100,  # Very low price would create large position
            stop_loss_price=99,
        )

        # Max position = 100000 * 0.10 = 10000
        assert result.position_value <= 10000

    def test_no_stop_loss(self):
        """Test calculation without stop loss."""
        sizer = FixedFractionalSizer(risk_percent=0.02)
        result = sizer.calculate(
            capital=100000,
            entry_price=50000,
        )

        # Without stop loss: quantity = (capital * risk%) / entry_price
        # = (100000 * 0.02) / 50000 = 0.04
        assert result.quantity == pytest.approx(0.04, rel=0.01)

    def test_invalid_risk_percent(self):
        """Test that invalid risk_percent raises error."""
        with pytest.raises(ValueError):
            FixedFractionalSizer(risk_percent=0.15)  # > 10%

        with pytest.raises(ValueError):
            FixedFractionalSizer(risk_percent=0)  # = 0

    def test_invalid_capital(self):
        """Test that invalid capital raises error."""
        sizer = FixedFractionalSizer()
        with pytest.raises(ValueError):
            sizer.calculate(capital=-1000, entry_price=100)


class TestKellyCriterionSizer:
    """Tests for Kelly Criterion Position Sizing."""

    def test_positive_edge(self):
        """Test Kelly with positive expected value."""
        # Win rate 55%, payoff ratio 1.5 = positive edge
        sizer = KellyCriterionSizer(
            win_rate=0.55,
            payoff_ratio=1.5,
            kelly_fraction=0.25,
        )

        # Full Kelly = 0.55 - (0.45/1.5) = 0.55 - 0.30 = 0.25
        # Adjusted = 0.25 * 0.25 = 0.0625
        assert sizer.full_kelly == pytest.approx(0.25, rel=0.01)
        assert sizer.adjusted_kelly == pytest.approx(0.0625, rel=0.01)

        result = sizer.calculate(capital=100000, entry_price=1000)
        assert result.quantity > 0
        assert result.method == "kelly_criterion"

    def test_negative_edge(self):
        """Test Kelly with negative expected value (no position)."""
        # Win rate 40%, payoff ratio 1.0 = negative edge
        sizer = KellyCriterionSizer(
            win_rate=0.40,
            payoff_ratio=1.0,
            kelly_fraction=0.25,
        )

        # Full Kelly = 0.40 - (0.60/1.0) = -0.20
        # Adjusted = 0 (capped)
        assert sizer.full_kelly < 0
        assert sizer.adjusted_kelly == 0

        result = sizer.calculate(capital=100000, entry_price=1000)
        assert result.quantity == 0  # No position when edge is negative

    def test_update_stats(self):
        """Test updating win rate and payoff ratio."""
        sizer = KellyCriterionSizer(win_rate=0.5, payoff_ratio=1.0)
        original_kelly = sizer.adjusted_kelly

        sizer.update_stats(win_rate=0.6, payoff_ratio=2.0)
        assert sizer.adjusted_kelly > original_kelly


class TestVolatilityBasedSizer:
    """Tests for Volatility-Based Position Sizing."""

    def test_basic_calculation(self):
        """Test basic ATR-based sizing."""
        # Use max_position_percent=0.6 to avoid cap (position_value=50000, cap=60000)
        sizer = VolatilityBasedSizer(
            risk_percent=0.02,
            atr_multiplier=2.0,
            max_position_percent=0.6,
        )

        result = sizer.calculate(
            capital=100000,
            entry_price=50000,
            atr=1000,  # ATR = 1000
        )

        # Risk amount = 100000 * 0.02 = 2000
        # Stop distance = 1000 * 2 = 2000
        # Quantity = 2000 / 2000 = 1
        assert result.quantity == pytest.approx(1.0, rel=0.01)
        assert result.method == "volatility_based"

    def test_high_volatility_reduces_size(self):
        """Test that higher ATR results in smaller position."""
        # Use higher max_position_percent to avoid cap affecting the comparison
        sizer = VolatilityBasedSizer(
            risk_percent=0.02, atr_multiplier=2.0, max_position_percent=0.6
        )

        result_low_vol = sizer.calculate(
            capital=100000,
            entry_price=50000,
            atr=500,  # Low volatility -> quantity = 2000/1000 = 2
        )

        result_high_vol = sizer.calculate(
            capital=100000,
            entry_price=50000,
            atr=2000,  # High volatility -> quantity = 2000/4000 = 0.5
        )

        # Higher ATR should result in smaller position
        assert result_high_vol.quantity < result_low_vol.quantity

    def test_missing_atr_raises_error(self):
        """Test that missing ATR raises error."""
        sizer = VolatilityBasedSizer()
        with pytest.raises(ValueError):
            sizer.calculate(capital=100000, entry_price=1000)  # No ATR


class TestDrawdownGuard:
    """Tests for Drawdown Guard."""

    def test_initialization(self):
        """Test guard initialization."""
        guard = DrawdownGuard(
            daily_loss_limit=0.03,
            weekly_loss_limit=0.07,
            max_drawdown_limit=0.15,
        )
        guard.initialize(starting_capital=100000)

        assert guard.peak_capital == 100000
        assert guard.current_capital == 100000
        assert not guard.is_halted

    def test_normal_state(self):
        """Test normal risk state with no losses."""
        guard = DrawdownGuard()
        guard.initialize(100000)

        state = guard.check_risk()
        assert state.status == RiskStatus.NORMAL
        assert state.can_trade is True

    def test_record_trade_profit(self):
        """Test recording profitable trade."""
        guard = DrawdownGuard()
        guard.initialize(100000)

        guard.record_trade("BTC", pnl=5000, side="buy")

        assert guard.current_capital == 105000
        assert guard.peak_capital == 105000  # New peak

    def test_record_trade_loss(self):
        """Test recording losing trade."""
        guard = DrawdownGuard()
        guard.initialize(100000)

        guard.record_trade("BTC", pnl=-2000, side="buy")

        assert guard.current_capital == 98000
        assert guard.peak_capital == 100000  # Peak unchanged

    def test_daily_loss_limit_breach(self):
        """Test that daily loss limit triggers halt."""
        guard = DrawdownGuard(daily_loss_limit=0.03)  # 3% daily limit
        guard.initialize(100000)

        # Lose 3% (3000) in one day
        guard.record_trade("BTC", pnl=-3000, side="buy")

        state = guard.check_risk()
        assert state.status == RiskStatus.HALTED
        assert state.can_trade is False
        assert guard.is_halted is True

    def test_max_drawdown_breach(self):
        """Test that max drawdown triggers halt."""
        # Set daily_loss_limit high (20%) so max_drawdown_limit (15%) triggers first
        guard = DrawdownGuard(
            max_drawdown_limit=0.15,
            daily_loss_limit=0.20,
            weekly_loss_limit=0.30,
        )
        guard.initialize(100000)

        # Multiple losses totaling 16%
        guard.record_trade("BTC", pnl=-10000, side="buy")
        guard.record_trade("ETH", pnl=-6000, side="buy")

        state = guard.check_risk()
        # Drawdown = 16000 / 100000 = 16% > 15%
        assert state.status == RiskStatus.HALTED
        assert "drawdown" in state.message.lower()

    def test_warning_threshold(self):
        """Test warning state before limit breach."""
        guard = DrawdownGuard(
            daily_loss_limit=0.03,
            warning_threshold=0.7,
        )
        guard.initialize(100000)

        # Lose 2.1% (70% of 3% limit)
        guard.record_trade("BTC", pnl=-2100, side="buy")

        state = guard.check_risk()
        assert state.status == RiskStatus.WARNING
        assert state.can_trade is True  # Still allowed to trade

    def test_manual_halt_reset(self):
        """Test manual reset of halt."""
        guard = DrawdownGuard(daily_loss_limit=0.01)
        guard.initialize(100000)

        # Record a trade that exceeds daily limit (1.5% > 1%)
        guard.record_trade("BTC", pnl=-1500, side="buy")
        # check_risk() triggers the halt check
        state = guard.check_risk()
        assert state.status == RiskStatus.HALTED
        assert guard.is_halted is True

        # Reset halt - this clears is_halted flag
        guard.reset_halt()
        assert guard.is_halted is False

        # Note: can_trade() calls check_risk() which will re-halt if limits
        # are still breached. This is correct risk management behavior.
        # To truly resume trading, either clear history or add profit.
        # Add profit to bring daily P&L within limits
        guard.record_trade("BTC", pnl=1000, side="sell")  # Now -500 (0.5% < 1%)
        assert guard.can_trade() is True

    def test_get_metrics(self):
        """Test metrics dictionary output."""
        guard = DrawdownGuard()
        guard.initialize(100000)
        guard.record_trade("BTC", pnl=1000, side="buy")

        metrics = guard.get_metrics()

        assert "status" in metrics
        assert "can_trade" in metrics
        assert "current_drawdown" in metrics
        assert "daily_pnl" in metrics
        assert metrics["current_capital"] == 101000
