"""
Tests for BinanceFuturesV6Strategy.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from libs.strategies.base import BaseStrategy
from libs.strategies.base import Signal as BaseSignal
from libs.strategies.binance_futures_v6 import (
    BinanceFuturesV6Config,
    BinanceFuturesV6Strategy,
    MarketRegime,
    SignalType,
)


@pytest.fixture
def strategy():
    """Default strategy instance."""
    return BinanceFuturesV6Strategy()


@pytest.fixture
def sample_4h():
    """Sample 4H OHLCV DataFrame (300 bars)."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range(start="2024-01-01", periods=n, freq="4h")
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 20
    volume = np.abs(np.random.randn(n) * 1000000) + 500000
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def sample_1d():
    """Sample 1D OHLCV DataFrame (400 bars)."""
    np.random.seed(99)
    n = 400
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    close = 40000 + np.cumsum(np.random.randn(n) * 200)
    high = close + np.abs(np.random.randn(n) * 80)
    low = close - np.abs(np.random.randn(n) * 80)
    open_ = close + np.random.randn(n) * 30
    volume = np.abs(np.random.randn(n) * 2000000) + 1000000
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# --- P0: BaseStrategy Inheritance ---


def test_inherits_base_strategy(strategy):
    """P0-7: Should inherit from BaseStrategy."""
    assert isinstance(strategy, BaseStrategy)
    assert strategy.strategy_id == "binance_futures_v6"


def test_strategy_id_lowercase(strategy):
    """NEW-3: strategy_id should be lowercase, uppercase is alias."""
    assert strategy.strategy_id == "binance_futures_v6"
    assert strategy.STRATEGY_ID == "binance_futures_v6"


# --- BaseStrategy Interface ---


def test_generate_signals_interface(strategy):
    """BaseStrategy generate_signals() should return TradeSignals."""
    results = strategy.generate_signals(["BTCUSDT", "ETHUSDT"])
    assert len(results) == 2
    assert all(r.signal == BaseSignal.HOLD for r in results)


def test_check_gate(strategy):
    """Gate always True (BTC gate is per-signal)."""
    assert strategy.check_gate() is True


def test_update_position_cleans_internal(strategy):
    """BaseStrategy update_position should sync internal positions."""
    from libs.strategies.binance_futures_v6 import Position

    strategy.positions["BTCUSDT"] = Position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        entry_time=datetime.now(),
        size=0.1,
        leverage=5,
        stop_loss=49000.0,
        regime_at_entry=MarketRegime.BULL,
    )
    strategy.update_position("BTCUSDT", 0.0)
    assert "BTCUSDT" not in strategy.positions


# --- Auto-Reset ---


def test_auto_reset_daily(strategy):
    """P1-13: daily_pnl should auto-reset on new day."""
    strategy.daily_pnl = -2.5
    strategy._last_reset_date = (datetime.now() - timedelta(days=1)).date()
    strategy._auto_reset_risk_counters()
    assert strategy.daily_pnl == 0.0


def test_auto_reset_weekly(strategy):
    """P1-13: weekly_pnl should auto-reset on new week."""
    strategy.weekly_pnl = -5.0
    strategy._last_reset_week = (
        datetime.now() - timedelta(weeks=1)
    ).isocalendar()[1]
    strategy._auto_reset_risk_counters()
    assert strategy.weekly_pnl == 0.0


# --- DataFrame Validation ---


def test_generate_signal_none_df(strategy):
    """NEW-4: None DataFrames should return HOLD."""
    sig = strategy.generate_signal("BTCUSDT", None, None)
    assert sig.signal_type == SignalType.HOLD
    assert "unavailable" in sig.reason.lower()


def test_generate_signal_empty_df(strategy):
    """NEW-4: Empty DataFrames should return HOLD."""
    empty_df = pd.DataFrame()
    sig = strategy.generate_signal("BTCUSDT", empty_df, empty_df)
    assert sig.signal_type == SignalType.HOLD


# --- Drawdown Guard ---


def test_drawdown_guard_updated_on_close(strategy):
    """NEW-2: peak_equity/current_drawdown should update on close."""
    from libs.strategies.binance_futures_v6 import Position, Signal

    strategy.positions["ETHUSDT"] = Position(
        symbol="ETHUSDT",
        side="LONG",
        entry_price=3000.0,
        entry_time=datetime.now(),
        size=1.0,
        leverage=5,
        stop_loss=2900.0,
        regime_at_entry=MarketRegime.BULL,
    )

    exit_signal = Signal(
        signal_type=SignalType.EXIT_LONG,
        symbol="ETHUSDT",
        price=2900.0,
        reason="Stop",
        timestamp=datetime.now(),
        regime=MarketRegime.BULL,
    )

    pnl = strategy.close_position("ETHUSDT", exit_signal)
    assert pnl < 0  # Loss
    assert strategy.current_drawdown < 0


# --- Risk Limits ---


def test_risk_limits_ok(strategy):
    """Risk limits should pass with default state."""
    ok, reason = strategy.check_risk_limits()
    assert ok is True
    assert reason == "OK"


def test_risk_limits_daily_loss(strategy):
    """Daily loss limit should block trading."""
    strategy.daily_pnl = -4.0
    ok, reason = strategy.check_risk_limits()
    assert ok is False
    assert "Daily" in reason


def test_risk_limits_weekly_loss(strategy):
    """Weekly loss limit should block trading."""
    strategy.weekly_pnl = -8.0
    ok, reason = strategy.check_risk_limits()
    assert ok is False
    assert "Weekly" in reason


# --- Market Regime ---


def test_regime_detection(strategy):
    """Regime detection with BTC data."""
    # Bull: above EMA, near 52w high
    strategy.update_btc_data(50000, 45000, 52000, 1.5)
    assert strategy.current_regime == MarketRegime.BULL

    # Bear: below EMA, -30%+ from high
    strategy.update_btc_data(30000, 45000, 50000, -2.0)
    assert strategy.current_regime == MarketRegime.BEAR


# --- BTC Gate ---


def test_btc_gate_blocks_long(strategy):
    """BTC gate should block longs when BTC drops > 5%."""
    strategy._btc_change_24h = -6.0
    assert strategy.check_btc_gate("LONG") is False
    assert strategy.check_btc_gate("SHORT") is True


# --- Signal Generation ---


def test_generate_signal_with_data(strategy, sample_4h, sample_1d):
    """Should produce a valid signal with proper data."""
    strategy.update_btc_data(50000, 45000, 52000, 1.0)
    sig = strategy.generate_signal("BTCUSDT", sample_4h, sample_1d)
    assert sig.signal_type in (
        SignalType.LONG,
        SignalType.SHORT,
        SignalType.HOLD,
    )
    assert sig.regime == strategy.current_regime


# --- State & Reset ---


def test_get_state(strategy):
    """get_state should return complete state dict."""
    state = strategy.get_state()
    assert state["strategy_id"] == "binance_futures_v6"
    assert state["version"] == "6.0.0"
    assert "positions" in state
    assert "risk" in state
    assert "config" in state


def test_reset(strategy):
    """Reset should clear all state."""
    strategy.daily_pnl = -2.0
    strategy.weekly_pnl = -5.0
    strategy.consecutive_losses = 3
    strategy.reset()
    assert strategy.daily_pnl == 0.0
    assert strategy.weekly_pnl == 0.0
    assert strategy.consecutive_losses == 0


def test_get_state_json_serializable(strategy):
    """get_state() should produce JSON-serializable output."""
    import json

    state = strategy.get_state()
    json_str = json.dumps(state)
    assert json_str is not None
