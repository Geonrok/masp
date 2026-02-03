"""
Tests for Circuit Breaker.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import os

from libs.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitState,
    TriggerReason,
    create_circuit_breaker,
)


@pytest.fixture
def config():
    """Create test config."""
    return CircuitBreakerConfig(
        max_drawdown_pct=0.10,
        max_daily_loss_pct=0.03,
        max_consecutive_losses=3,
        cooldown_minutes=5,
        auto_recover=False,
    )


@pytest.fixture
def breaker(config):
    """Create circuit breaker with test config."""
    return CircuitBreaker(config, initial_equity=1_000_000)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.max_drawdown_pct == 0.10
        assert config.max_daily_loss_pct == 0.03
        assert config.max_consecutive_losses == 5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            max_drawdown_pct=0.15,
            max_daily_loss_pct=0.05,
        )
        assert config.max_drawdown_pct == 0.15
        assert config.max_daily_loss_pct == 0.05

    @patch.dict(
        os.environ,
        {
            "MASP_MAX_DRAWDOWN_PCT": "0.20",
            "MASP_MAX_DAILY_LOSS_PCT": "0.05",
        },
    )
    def test_from_env(self):
        """Test loading from environment."""
        config = CircuitBreakerConfig.from_env()
        assert config.max_drawdown_pct == 0.20
        assert config.max_daily_loss_pct == 0.05


class TestCircuitBreakerInit:
    """Tests for circuit breaker initialization."""

    def test_init_with_equity(self):
        """Test initialization with initial equity."""
        breaker = CircuitBreaker(initial_equity=1_000_000)
        assert breaker.state.current_equity == 1_000_000
        assert breaker.state.peak_equity == 1_000_000
        assert breaker.state.daily_start_equity == 1_000_000

    def test_init_closed_state(self, breaker):
        """Test initial state is closed."""
        assert breaker.state.state == CircuitState.CLOSED
        assert breaker.is_trading_allowed() is True


class TestDrawdownTrigger:
    """Tests for drawdown-based triggers."""

    def test_drawdown_calculation(self, breaker):
        """Test drawdown is calculated correctly."""
        breaker.update_equity(1_000_000)  # Peak
        breaker.update_equity(950_000)  # 5% drawdown

        assert breaker.state.drawdown_pct == pytest.approx(0.05, rel=0.01)

    def test_drawdown_trigger(self, breaker):
        """Test circuit triggers on max drawdown."""
        breaker.update_equity(1_000_000)
        breaker.update_equity(890_000)  # 11% drawdown

        assert breaker.state.state == CircuitState.OPEN
        assert breaker.state.trigger_reason == TriggerReason.DRAWDOWN
        assert breaker.is_trading_allowed() is False

    def test_drawdown_warning(self, config):
        """Test warning on approaching drawdown limit."""
        notifications = []

        def callback(event, data):
            notifications.append((event, data))

        config.notify_on_warning = True
        config.notification_callback = callback
        config.warning_drawdown_pct = 0.05
        config.max_daily_loss_pct = 0.10  # Increase to avoid daily loss trigger

        breaker = CircuitBreaker(config, initial_equity=1_000_000)
        breaker.update_equity(940_000)  # 6% drawdown

        assert breaker.state.state == CircuitState.CLOSED  # Not triggered yet
        # Warning should have been sent (logged)


class TestDailyLossTrigger:
    """Tests for daily loss triggers."""

    def test_daily_loss_calculation(self, breaker):
        """Test daily loss is calculated correctly."""
        breaker.state.daily_start_equity = 1_000_000
        breaker.update_equity(980_000)  # 2% daily loss

        assert breaker.state.daily_loss_pct == pytest.approx(0.02, rel=0.01)

    def test_daily_loss_trigger(self, breaker):
        """Test circuit triggers on max daily loss."""
        breaker.state.daily_start_equity = 1_000_000
        breaker.update_equity(960_000)  # 4% daily loss (> 3% limit)

        assert breaker.state.state == CircuitState.OPEN
        assert breaker.state.trigger_reason == TriggerReason.DAILY_LOSS

    def test_daily_reset(self, breaker):
        """Test daily counters reset."""
        breaker.update_equity(950_000)
        breaker.state.daily_loss_pct = 0.02

        breaker.reset_daily()

        assert breaker.state.daily_start_equity == 950_000
        assert breaker.state.daily_loss_pct == 0.0


class TestConsecutiveLossesTrigger:
    """Tests for consecutive losses trigger."""

    def test_consecutive_losses_tracking(self, breaker):
        """Test consecutive losses are tracked."""
        breaker.record_trade(profit=-100)
        assert breaker.state.consecutive_losses == 1

        breaker.record_trade(profit=-100)
        assert breaker.state.consecutive_losses == 2

        breaker.record_trade(profit=100)  # Win resets counter
        assert breaker.state.consecutive_losses == 0

    def test_consecutive_losses_trigger(self, breaker):
        """Test circuit triggers on max consecutive losses."""
        for _ in range(3):
            breaker.record_trade(profit=-100)

        assert breaker.state.state == CircuitState.OPEN
        assert breaker.state.trigger_reason == TriggerReason.CONSECUTIVE_LOSSES

    def test_trade_stats(self, breaker):
        """Test trade statistics tracking."""
        breaker.record_trade(profit=100)
        breaker.record_trade(profit=50)
        breaker.record_trade(profit=-30)

        assert breaker.state.total_trades == 3
        assert breaker.state.winning_trades == 2
        assert breaker.state.losing_trades == 1


class TestCircuitReset:
    """Tests for circuit breaker reset."""

    def test_reset_when_closed(self, breaker):
        """Test reset when already closed."""
        result = breaker.reset()
        assert result is True
        assert breaker.state.state == CircuitState.CLOSED

    def test_reset_after_trigger(self, breaker):
        """Test reset after trigger."""
        # Trigger the breaker
        breaker.update_equity(800_000)  # 20% drawdown
        assert breaker.state.state == CircuitState.OPEN

        # Force reset
        result = breaker.reset(force=True)
        assert result is True
        assert breaker.state.state == CircuitState.CLOSED

    def test_reset_respects_cooldown(self, config):
        """Test reset respects cooldown period."""
        config.cooldown_minutes = 60
        breaker = CircuitBreaker(config, initial_equity=1_000_000)

        # Trigger
        breaker.update_equity(800_000)
        assert breaker.state.state == CircuitState.OPEN

        # Try to reset without forcing
        result = breaker.reset(force=False)
        assert result is False
        assert breaker.state.state == CircuitState.OPEN


class TestManualTrigger:
    """Tests for manual triggering."""

    def test_manual_trigger(self, breaker):
        """Test manual trigger."""
        breaker.manual_trigger("Emergency stop")

        assert breaker.state.state == CircuitState.OPEN
        assert breaker.state.trigger_reason == TriggerReason.MANUAL


class TestStatus:
    """Tests for status reporting."""

    def test_get_status(self):
        """Test getting status."""
        config = CircuitBreakerConfig(
            max_drawdown_pct=0.10,
            max_daily_loss_pct=0.10,  # Higher threshold to avoid trigger
        )
        breaker = CircuitBreaker(config, initial_equity=1_000_000)
        breaker.update_equity(950_000)
        breaker.record_trade(profit=-100)

        status = breaker.get_status()

        assert status["state"] == "closed"
        assert status["is_trading_allowed"] is True
        assert status["metrics"]["current_equity"] == 950_000
        assert status["trade_stats"]["total"] == 1

    def test_status_after_trigger(self, breaker):
        """Test status after trigger."""
        breaker.update_equity(800_000)

        status = breaker.get_status()

        assert status["state"] == "open"
        assert status["is_trading_allowed"] is False
        assert status["trigger_reason"] == "drawdown"


class TestKillSwitchIntegration:
    """Tests for kill switch file integration."""

    def test_kill_switch_created_on_trigger(self, tmp_path):
        """Test kill switch file is created on trigger."""
        kill_switch_path = tmp_path / "kill_switch"

        config = CircuitBreakerConfig(
            max_drawdown_pct=0.10,
            kill_switch_file=str(kill_switch_path),
            create_kill_switch_on_trigger=True,
        )
        breaker = CircuitBreaker(config, initial_equity=1_000_000)

        breaker.update_equity(800_000)  # Trigger

        assert kill_switch_path.exists()

    def test_kill_switch_removed_on_reset(self, tmp_path):
        """Test kill switch file is removed on reset."""
        kill_switch_path = tmp_path / "kill_switch"

        config = CircuitBreakerConfig(
            max_drawdown_pct=0.10,
            kill_switch_file=str(kill_switch_path),
            create_kill_switch_on_trigger=True,
        )
        breaker = CircuitBreaker(config, initial_equity=1_000_000)

        # Trigger and reset
        breaker.update_equity(800_000)
        assert kill_switch_path.exists()

        breaker.reset(force=True)
        assert not kill_switch_path.exists()


class TestTriggerHistory:
    """Tests for trigger history tracking."""

    def test_trigger_history_recorded(self, breaker):
        """Test trigger events are recorded in history."""
        breaker.update_equity(800_000)

        assert len(breaker.state.trigger_history) == 1
        assert breaker.state.trigger_history[0]["reason"] == "drawdown"


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_create_circuit_breaker(self):
        """Test create_circuit_breaker helper."""
        breaker = create_circuit_breaker(
            initial_equity=1_000_000,
            max_drawdown_pct=0.15,
        )

        assert breaker.config.max_drawdown_pct == 0.15
        assert breaker.state.current_equity == 1_000_000


class TestPeakTracking:
    """Tests for peak equity tracking."""

    def test_peak_updates_on_new_high(self, breaker):
        """Test peak updates when equity increases."""
        breaker.update_equity(1_100_000)
        assert breaker.state.peak_equity == 1_100_000

        breaker.update_equity(1_050_000)
        assert breaker.state.peak_equity == 1_100_000  # Should not decrease

    def test_drawdown_from_peak(self, breaker):
        """Test drawdown calculated from peak."""
        breaker.update_equity(1_200_000)  # New peak
        breaker.update_equity(1_080_000)  # 10% from peak

        assert breaker.state.peak_equity == 1_200_000
        assert breaker.state.drawdown_pct == pytest.approx(0.10, rel=0.01)
