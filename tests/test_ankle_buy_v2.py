"""Tests for Ankle Buy v2.0 strategy."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from libs.strategies.ankle_buy_v2 import (
    ATR_PERIOD,
    ATR_STOP_MULT,
    BTC_GATE_SMA,
    MIN_BARS,
    SMA_LONG,
    SMA_SHORT,
    TP_INCREMENT,
    TP_SELL_FRACTION,
    AnkleBuyV2Strategy,
)
from libs.strategies.base import Signal, TradeSignal


def _make_ohlcv(n=80, base_price=100.0, trend=0.001):
    """Generate synthetic OHLCV data."""
    prices = base_price * np.cumprod(1 + np.random.randn(n) * 0.02 + trend)
    return {
        "open": prices * (1 - np.random.rand(n) * 0.01),
        "high": prices * (1 + np.random.rand(n) * 0.02),
        "low": prices * (1 - np.random.rand(n) * 0.02),
        "close": prices,
        "volume": np.random.rand(n) * 1000 + 100,
    }


def _make_breakout_data(n=80, base_price=100.0):
    """Generate data with a clear SMA breakout on t-1 bar."""
    data = _make_ohlcv(n, base_price, trend=-0.001)
    opens = data["open"].copy()
    closes = data["close"].copy()

    # Compute SMA(25) on opens at t-1
    sma25_prev = float(np.mean(opens[-(SMA_SHORT + 1):-1]))

    # Force t-1 candle: open below SMA, close above SMA, green candle
    opens[-2] = sma25_prev * 0.98
    closes[-2] = sma25_prev * 1.02  # close > open and close > SMA
    data["open"] = opens
    data["close"] = closes
    data["high"][-2] = closes[-2] * 1.01
    data["low"][-2] = opens[-2] * 0.99
    return data


class TestAnkleBuyV2Init:
    """Test strategy initialization."""

    def test_default_init(self):
        s = AnkleBuyV2Strategy()
        assert s.strategy_id == "ankle_buy_v2"
        assert s.version == "2.0.0"
        assert s._btc_symbol is None
        assert s._pos_info == {}

    def test_custom_btc_symbol(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/KRW")
        assert s._btc_symbol == "BTC/KRW"


class TestBTCGate:
    """Test BTC gate logic."""

    def test_gate_on_when_btc_above_sma(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        # BTC trending up → close[t-1] > SMA50[t-1]
        btc = _make_ohlcv(n=80, base_price=50000, trend=0.005)
        s._btc_cache = btc
        assert s.check_gate() is True

    def test_gate_off_when_btc_below_sma(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        # BTC trending down → close[t-1] < SMA50[t-1]
        btc = _make_ohlcv(n=80, base_price=50000, trend=-0.005)
        s._btc_cache = btc
        assert s.check_gate() is False

    def test_gate_default_true_insufficient_data(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        btc = _make_ohlcv(n=10, base_price=50000)
        s._btc_cache = btc
        assert s.check_gate() is True

    def test_gate_default_true_no_data(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        s._btc_cache = None
        # No market_data adapter → no BTC data
        assert s.check_gate() is True


class TestBTCSymbolDetection:
    """Test BTC symbol auto-detection."""

    def test_detect_krw(self):
        s = AnkleBuyV2Strategy()
        sym = s._detect_btc_symbol("ETH/KRW")
        assert sym == "BTC/KRW"

    def test_detect_usdt(self):
        s = AnkleBuyV2Strategy()
        sym = s._detect_btc_symbol("ETH/USDT")
        assert sym == "BTC/USDT"

    def test_explicit_overrides_detect(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/EUR")
        sym = s._detect_btc_symbol("ETH/USDT")
        assert sym == "BTC/EUR"


class TestEntrySignal:
    """Test entry signal conditions."""

    def test_breakout_detected(self):
        s = AnkleBuyV2Strategy()
        data = _make_breakout_data()
        assert s._check_entry_signal(data) is True

    def test_no_signal_red_candle(self):
        s = AnkleBuyV2Strategy()
        data = _make_ohlcv(n=80)
        # Force red candle at t-1
        data["close"][-2] = data["open"][-2] * 0.98
        assert s._check_entry_signal(data) is False

    def test_no_signal_insufficient_data(self):
        s = AnkleBuyV2Strategy()
        data = _make_ohlcv(n=20)
        assert s._check_entry_signal(data) is False


class TestExitConditions:
    """Test exit conditions."""

    def test_sma_exit_below_open_and_sma(self):
        s = AnkleBuyV2Strategy()
        data = _make_ohlcv(n=80, base_price=100)
        # Force: today's close < today's open AND close <= upper SMA
        upper_sma = s._compute_upper_sma(data)
        data["close"][-1] = min(data["open"][-1] * 0.95, upper_sma * 0.99)
        assert s._check_sma_exit(data) is True

    def test_no_sma_exit_above_open(self):
        s = AnkleBuyV2Strategy()
        data = _make_ohlcv(n=80, base_price=100)
        # Force: close > open → no SMA exit
        data["close"][-1] = data["open"][-1] * 1.05
        assert s._check_sma_exit(data) is False

    def test_stop_loss_computation(self):
        s = AnkleBuyV2Strategy()
        data = _make_ohlcv(n=80, base_price=100)
        stop = s._compute_stop_loss(data, 100.0)
        assert stop < 100.0
        assert stop > 0


class TestTPLevels:
    """Test take-profit multi-level logic."""

    def test_single_tp_level(self):
        s = AnkleBuyV2Strategy()
        s._positions = {"ETH/USDT": 10.0}
        s._pos_info = {
            "ETH/USDT": {
                "entry_price": 100.0,
                "original_qty": 10.0,
                "stop_loss": 80.0,
                "tp_sold": set(),
                "upper_sma": 95.0,
            }
        }
        # Price at +12% → level 1 hit
        frac, reason = s._check_tp_levels("ETH/USDT", 112.0)
        assert frac > 0
        assert "+10%" in reason
        assert 1 in s._pos_info["ETH/USDT"]["tp_sold"]

    def test_multi_level_tp(self):
        s = AnkleBuyV2Strategy()
        s._positions = {"ETH/USDT": 10.0}
        s._pos_info = {
            "ETH/USDT": {
                "entry_price": 100.0,
                "original_qty": 10.0,
                "stop_loss": 80.0,
                "tp_sold": set(),
                "upper_sma": 95.0,
            }
        }
        # Price at +25% → levels 1,2 hit
        frac, reason = s._check_tp_levels("ETH/USDT", 125.0)
        assert frac == pytest.approx(0.2, abs=0.01)  # 2 levels * 10%
        assert {1, 2} == s._pos_info["ETH/USDT"]["tp_sold"]

    def test_no_tp_below_threshold(self):
        s = AnkleBuyV2Strategy()
        s._positions = {"ETH/USDT": 10.0}
        s._pos_info = {
            "ETH/USDT": {
                "entry_price": 100.0,
                "original_qty": 10.0,
                "stop_loss": 80.0,
                "tp_sold": set(),
                "upper_sma": 95.0,
            }
        }
        frac, _ = s._check_tp_levels("ETH/USDT", 105.0)
        assert frac == 0.0

    def test_tp_level_not_repeated(self):
        s = AnkleBuyV2Strategy()
        s._positions = {"ETH/USDT": 9.0}
        s._pos_info = {
            "ETH/USDT": {
                "entry_price": 100.0,
                "original_qty": 10.0,
                "stop_loss": 80.0,
                "tp_sold": {1},  # Level 1 already sold
                "upper_sma": 95.0,
            }
        }
        # Price still at +12% → level 1 already sold, no new TP
        frac, _ = s._check_tp_levels("ETH/USDT", 112.0)
        assert frac == 0.0


class TestGenerateSignal:
    """Test generate_signal integration."""

    def test_hold_no_data(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        sig = s.generate_signal("ETH/USDT", gate_pass=True)
        assert sig.signal == Signal.HOLD
        assert "unavailable" in sig.reason.lower()

    def test_skip_btc_symbol(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        sig = s.generate_signal("BTC/USDT", gate_pass=True)
        assert sig.signal == Signal.HOLD
        assert "Gate asset" in sig.reason

    def test_buy_on_breakout(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        data = _make_breakout_data(n=80, base_price=100)
        s._ohlcv_cache["ETH/USDT"] = data
        sig = s.generate_signal("ETH/USDT", gate_pass=True)
        assert sig.signal == Signal.BUY
        assert "breakout" in sig.reason.lower()
        assert "ETH/USDT" in s._pos_info

    def test_hold_on_breakout_gate_off(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        data = _make_breakout_data(n=80)
        s._ohlcv_cache["ETH/USDT"] = data
        sig = s.generate_signal("ETH/USDT", gate_pass=False)
        assert sig.signal == Signal.HOLD
        assert "Gate OFF" in sig.reason

    def test_sell_gate_off_with_position(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        s._positions["ETH/USDT"] = 10.0
        s._pos_info["ETH/USDT"] = {
            "entry_price": 100.0,
            "original_qty": 10.0,
            "stop_loss": 80.0,
            "tp_sold": set(),
            "upper_sma": 95.0,
        }
        data = _make_ohlcv(n=80, base_price=100)
        s._ohlcv_cache["ETH/USDT"] = data
        sig = s.generate_signal("ETH/USDT", gate_pass=False)
        assert sig.signal == Signal.SELL
        assert sig.strength == 1.0
        assert "Gate OFF" in sig.reason

    def test_sell_stop_loss(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        s._positions["ETH/USDT"] = 10.0
        s._pos_info["ETH/USDT"] = {
            "entry_price": 100.0,
            "original_qty": 10.0,
            "stop_loss": 90.0,
            "tp_sold": set(),
            "upper_sma": 95.0,
        }
        data = _make_ohlcv(n=80, base_price=100)
        data["close"][-1] = 85.0  # Below stop
        s._ohlcv_cache["ETH/USDT"] = data
        sig = s.generate_signal("ETH/USDT", gate_pass=True)
        assert sig.signal == Signal.SELL
        assert sig.strength == 1.0
        assert "Stop loss" in sig.reason

    def test_partial_sell_tp(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        s._positions["ETH/USDT"] = 10.0
        s._pos_info["ETH/USDT"] = {
            "entry_price": 100.0,
            "original_qty": 10.0,
            "stop_loss": 80.0,
            "tp_sold": set(),
            "upper_sma": 95.0,
        }
        data = _make_ohlcv(n=80, base_price=100)
        data["close"][-1] = 112.0  # +12% → TP level 1
        data["open"][-1] = 111.0  # Ensure no SMA exit
        s._ohlcv_cache["ETH/USDT"] = data
        sig = s.generate_signal("ETH/USDT", gate_pass=True)
        assert sig.signal == Signal.SELL
        assert sig.strength < 1.0  # Partial sell
        assert "+10%" in sig.reason


class TestPositionSync:
    """Test position update/sync."""

    def test_original_qty_recorded(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        s._pos_info["ETH/USDT"] = {
            "entry_price": 100.0,
            "original_qty": 0,
            "stop_loss": 80.0,
            "tp_sold": set(),
            "upper_sma": 95.0,
        }
        s.update_position("ETH/USDT", 5.0)
        assert s._pos_info["ETH/USDT"]["original_qty"] == 5.0

    def test_cleanup_on_zero_qty(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        s._pos_info["ETH/USDT"] = {
            "entry_price": 100.0,
            "original_qty": 5.0,
            "stop_loss": 80.0,
            "tp_sold": set(),
            "upper_sma": 95.0,
        }
        s.update_position("ETH/USDT", 0)
        assert "ETH/USDT" not in s._pos_info


class TestStatePersistence:
    """Test state save/load."""

    def test_save_and_load(self, tmp_path):
        state_file = tmp_path / "state.json"
        s1 = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        s1._state_path = state_file
        s1._pos_info = {
            "ETH/USDT": {
                "entry_price": 100.0,
                "original_qty": 10.0,
                "stop_loss": 80.0,
                "tp_sold": {1, 2},
                "upper_sma": 95.0,
                "entry_date": "2026-02-09T10:00:00",
            }
        }
        s1._save_state()

        s2 = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        s2._state_path = state_file
        s2._load_state()
        assert "ETH/USDT" in s2._pos_info
        assert s2._pos_info["ETH/USDT"]["entry_price"] == 100.0
        assert s2._pos_info["ETH/USDT"]["tp_sold"] == {1, 2}


class TestExchangeIsolation:
    """Test per-exchange state isolation."""

    def test_set_exchange_name_changes_path(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        assert "default" in str(s._state_path)
        s.set_exchange_name("upbit_ankle")
        assert "upbit_ankle" in str(s._state_path)

    def test_different_exchanges_different_state(self, tmp_path):
        s1 = AnkleBuyV2Strategy(btc_symbol="BTC/KRW")
        s1._state_path = tmp_path / "ankle_buy_v2_upbit.json"
        s1._pos_info = {"ETH/KRW": {"entry_price": 1000, "original_qty": 1, "stop_loss": 900, "tp_sold": set(), "upper_sma": 950}}
        s1._save_state()

        s2 = AnkleBuyV2Strategy(btc_symbol="BTC/KRW")
        s2._state_path = tmp_path / "ankle_buy_v2_bithumb.json"
        s2._pos_info = {"ETH/KRW": {"entry_price": 2000, "original_qty": 2, "stop_loss": 1800, "tp_sold": set(), "upper_sma": 1900}}
        s2._save_state()

        # Reload s1 - should have its own data
        s1._load_state()
        assert s1._pos_info["ETH/KRW"]["entry_price"] == 1000

        # Reload s2 - should have its own data
        s2._load_state()
        assert s2._pos_info["ETH/KRW"]["entry_price"] == 2000


class TestCacheClear:
    """Test cache clearing."""

    def test_clear_cache(self):
        s = AnkleBuyV2Strategy(btc_symbol="BTC/USDT")
        s._ohlcv_cache["ETH/USDT"] = _make_ohlcv()
        s._btc_cache = _make_ohlcv()
        s._gate_status = True
        s.clear_cache()
        assert s._ohlcv_cache == {}
        assert s._btc_cache is None
        assert s._gate_status is None
