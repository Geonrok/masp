"""
ATLAS-Futures Strategy Tests.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from libs.strategies.atlas_futures import (  # noqa: E402
    ATLASFuturesConfig,
    ATLASFuturesStrategy,
    SignalType,
)


@pytest.fixture
def strategy():
    """Strategy instance for tests."""
    return ATLASFuturesStrategy()


@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV data."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range(start="2024-01-01", periods=n, freq="4h")

    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 20
    volume = np.abs(np.random.randn(n) * 1000000) + 500000

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_strategy_initialization(strategy):
    assert strategy.STRATEGY_ID == "atlas_futures_p04"
    assert strategy.VERSION == "v2.6.2-r1"
    assert strategy.config.leverage == 3
    assert strategy.config.max_positions == 3


def test_config_defaults():
    config = ATLASFuturesConfig()
    assert config.bbwp_threshold == 10.0
    assert config.squeeze_min_bars == 6
    assert config.adx_threshold == 12.0
    assert config.chandelier_multiplier == 3.0
    assert "BTCUSDT" in config.symbols


def test_calculate_indicators(strategy, sample_ohlcv):
    df = strategy.calculate_indicators(sample_ohlcv)
    assert "bbwp" in df.columns
    assert "adx" in df.columns
    assert "ema_200" in df.columns
    assert "atr" in df.columns
    assert "rvol" in df.columns
    assert "bb_upper" in df.columns
    assert "bb_lower" in df.columns


def test_risk_limits_ok(strategy):
    ok, reason = strategy.check_risk_limits()
    assert ok is True
    assert reason == "OK"


def test_risk_limits_daily_loss(strategy):
    strategy.daily_pnl = -6.0
    ok, reason = strategy.check_risk_limits()
    assert ok is False
    assert "Daily loss" in reason


def test_risk_limits_consecutive_losses(strategy):
    strategy.consecutive_losses = 10
    ok, reason = strategy.check_risk_limits()
    assert ok is False
    assert "Consecutive" in reason


def test_link_blocking(strategy, sample_ohlcv):
    strategy.link_blocked = True
    signal = strategy.generate_signal("LINKUSDT", sample_ohlcv)
    assert signal.signal_type == SignalType.HOLD
    assert "blocked" in signal.reason.lower()


def test_track_switch_b_to_c(strategy):
    strategy.track_b_pnl = -1.0
    strategy.track_c_pnl = 1.0
    result = strategy.check_track_switch()
    assert result == "SWITCH_TO_C"
    assert strategy.link_blocked is True


def test_track_switch_c_to_b(strategy):
    strategy.track_b_pnl = 10.0
    strategy.track_c_pnl = 2.0
    strategy.link_blocked = True
    result = strategy.check_track_switch()
    assert result == "REVERT_TO_B"
    assert strategy.link_blocked is False


def test_get_state(strategy):
    state = strategy.get_state()
    assert state["strategy_id"] == "atlas_futures_p04"
    assert state["version"] == "v2.6.2-r1"
    assert "positions" in state
    assert "risk" in state
    assert "track" in state
    assert "config" in state


def test_reset(strategy):
    strategy.consecutive_losses = 5
    strategy.daily_pnl = -3.0
    strategy.reset()
    assert strategy.consecutive_losses == 0
    assert strategy.daily_pnl == 0.0
    assert len(strategy.positions) == 0


def test_position_to_dict_serialization(strategy):
    """Position.to_dict() JSON serialization."""
    import json
    from datetime import datetime

    from libs.strategies.atlas_futures import Position

    pos = Position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        entry_time=datetime.now(),
        size=0.1,
        leverage=3,
    )

    pos_dict = pos.to_dict()
    json_str = json.dumps(pos_dict)

    assert json_str is not None
    assert isinstance(pos_dict["entry_time"], str)
    assert "T" in pos_dict["entry_time"]


def test_get_state_json_serializable(strategy):
    """get_state() JSON serialization."""
    import json

    state = strategy.get_state()
    json_str = json.dumps(state)

    assert json_str is not None


def test_link_concentration_blocking(strategy):
    """LINK concentration > threshold should block."""
    from datetime import datetime

    from libs.strategies.atlas_futures import Position

    strategy.positions["LINKUSDT"] = Position(
        symbol="LINKUSDT",
        side="LONG",
        entry_price=20.0,
        entry_time=datetime.now(),
        size=100.0,
        leverage=3,
    )

    strategy.check_link_concentration()

    assert strategy.link_blocked is True


# --- P0/P1 Fix Verification Tests ---


def test_inherits_base_strategy(strategy):
    """P0-6: Should inherit from BaseStrategy."""
    from libs.strategies.base import BaseStrategy

    assert isinstance(strategy, BaseStrategy)
    assert strategy.strategy_id == "atlas_futures_p04"


def test_generate_signals_interface(strategy):
    """P0-6: BaseStrategy generate_signals() should work."""
    # Without market data adapter, should return HOLD for all
    results = strategy.generate_signals(["BTCUSDT", "ETHUSDT"])
    assert len(results) == 2
    from libs.strategies.base import Signal as BaseSignal

    assert all(r.signal == BaseSignal.HOLD for r in results)


def test_risk_auto_reset_daily(strategy):
    """P0-8: daily_pnl should auto-reset on new day."""
    from datetime import date, timedelta

    strategy.daily_pnl = -3.0
    strategy._last_reset_date = date.today() - timedelta(days=1)
    strategy._auto_reset_risk_counters()
    assert strategy.daily_pnl == 0.0


def test_chandelier_uses_current_bar(strategy, sample_ohlcv):
    """P1-3: _update_position should run before exit check."""
    from libs.strategies.atlas_futures import Position

    pos = Position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        entry_time=datetime.now(),
        size=0.1,
        leverage=3,
        highest_price=50000.0,
    )
    strategy.positions["BTCUSDT"] = pos

    # After generate_signal, highest_price should include current bar's high
    signal = strategy.generate_signal("BTCUSDT", sample_ohlcv)
    if "BTCUSDT" in strategy.positions:
        p = strategy.positions["BTCUSDT"]
        assert p.highest_price >= sample_ohlcv.iloc[-1]["high"] or p.bars_held >= 1


def test_track_pnl_updated_on_close(strategy):
    """P1-10: track_b/c_pnl should update on close_position."""
    from libs.strategies.atlas_futures import Position, Signal, SignalType

    strategy.positions["BTCUSDT"] = Position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        entry_time=datetime.now(),
        size=0.1,
        leverage=3,
    )

    exit_signal = Signal(
        signal_type=SignalType.EXIT_LONG,
        symbol="BTCUSDT",
        price=51000.0,
        reason="Test exit",
        timestamp=datetime.now(),
    )

    strategy.close_position("BTCUSDT", exit_signal)
    # BTCUSDT is in major_symbols -> track_b_pnl should be updated
    assert strategy.track_b_pnl != 0.0


def test_drawdown_guard_updated(strategy):
    """NEW-3: peak_equity and current_drawdown should update on close."""
    from libs.strategies.atlas_futures import Position, Signal, SignalType

    strategy.positions["ETHUSDT"] = Position(
        symbol="ETHUSDT",
        side="LONG",
        entry_price=3000.0,
        entry_time=datetime.now(),
        size=1.0,
        leverage=3,
    )

    # Close at a loss
    exit_signal = Signal(
        signal_type=SignalType.EXIT_LONG,
        symbol="ETHUSDT",
        price=2900.0,
        reason="Stop",
        timestamp=datetime.now(),
    )

    strategy.close_position("ETHUSDT", exit_signal)
    assert strategy.current_drawdown < 0  # Should be negative after loss


def test_update_position_cleans_internal(strategy):
    """BaseStrategy update_position should sync internal positions."""
    from libs.strategies.atlas_futures import Position

    strategy.positions["BTCUSDT"] = Position(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        entry_time=datetime.now(),
        size=0.1,
        leverage=3,
    )
    strategy.update_position("BTCUSDT", 0.0)
    assert "BTCUSDT" not in strategy.positions
