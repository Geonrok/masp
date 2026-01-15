"""
ATLAS-Futures Strategy Tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from libs.strategies.atlas_futures import (  # noqa: E402
    ATLASFuturesStrategy,
    ATLASFuturesConfig,
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
