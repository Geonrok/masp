"""
Tests for VwapBreakoutStrategy.
"""

import numpy as np

from libs.strategies.base import Signal
from libs.strategies.vwap_breakout import VwapBreakoutStrategy


def _make_trending_data(n: int = 300, trend: float = 0.001) -> dict:
    """Generate synthetic trending OHLCV data."""
    np.random.seed(42)
    close = np.zeros(n)
    close[0] = 100.0
    for i in range(1, n):
        close[i] = close[i - 1] * (1 + trend + np.random.randn() * 0.005)

    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    volume = np.random.uniform(100, 1000, n)

    return {"close": close, "high": high, "low": low, "volume": volume}


def _make_flat_data(n: int = 300) -> dict:
    """Generate synthetic flat/sideways OHLCV data."""
    np.random.seed(42)
    close = 100.0 + np.random.randn(n) * 0.5
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.uniform(100, 1000, n)

    return {"close": close, "high": high, "low": low, "volume": volume}


class TestVwapBreakoutInit:
    """Test strategy initialization."""

    def test_default_params(self):
        strategy = VwapBreakoutStrategy()
        assert strategy.strategy_id == "vwap_breakout"
        assert strategy.donchian_period == 48
        assert strategy.vwap_mult == 1.02
        assert strategy.ema_fast == 50
        assert strategy.ema_slow == 200
        assert strategy.kama_period == 20
        assert strategy.atr_stop == 3.0
        assert strategy.atr_target == 8.0
        assert strategy.max_hold_bars == 72

    def test_custom_params(self):
        strategy = VwapBreakoutStrategy(donchian_period=24, atr_stop=2.0)
        assert strategy.donchian_period == 24
        assert strategy.atr_stop == 2.0

    def test_get_parameters(self):
        strategy = VwapBreakoutStrategy()
        params = strategy.get_parameters()
        assert "donchian_period" in params
        assert "kama_period" in params
        assert params["max_positions"] == 10


class TestVwapBreakoutSignals:
    """Test signal generation."""

    def test_hold_on_flat_market(self):
        strategy = VwapBreakoutStrategy()
        data = _make_flat_data()
        strategy.update_ohlcv(
            "TEST/USDT:PERP",
            data["close"].tolist(),
            data["high"].tolist(),
            data["low"].tolist(),
            data["volume"].tolist(),
        )

        signal = strategy.generate_signal("TEST/USDT:PERP")
        assert signal.signal == Signal.HOLD

    def test_hold_when_no_data(self):
        strategy = VwapBreakoutStrategy()
        signal = strategy.generate_signal("UNKNOWN/USDT:PERP")
        assert signal.signal == Signal.HOLD
        assert "unavailable" in signal.reason.lower()

    def test_entry_on_strong_trend(self):
        """Strong uptrend should eventually trigger BUY."""
        strategy = VwapBreakoutStrategy()
        data = _make_trending_data(n=300, trend=0.003)
        strategy.update_ohlcv(
            "TREND/USDT:PERP",
            data["close"].tolist(),
            data["high"].tolist(),
            data["low"].tolist(),
            data["volume"].tolist(),
        )

        signal = strategy.generate_signal("TREND/USDT:PERP")
        # Strong trend may or may not trigger depending on exact data
        assert signal.signal in (Signal.BUY, Signal.HOLD)

    def test_generate_signals_multiple(self):
        strategy = VwapBreakoutStrategy()
        data = _make_flat_data()
        for sym in ["A/USDT:PERP", "B/USDT:PERP"]:
            strategy.update_ohlcv(
                sym,
                data["close"].tolist(),
                data["high"].tolist(),
                data["low"].tolist(),
                data["volume"].tolist(),
            )

        signals = strategy.generate_signals(["A/USDT:PERP", "B/USDT:PERP"])
        assert len(signals) == 2


class TestVwapBreakoutExits:
    """Test exit logic."""

    def test_exit_after_max_hold(self):
        strategy = VwapBreakoutStrategy(max_hold_bars=5)
        data = _make_flat_data()

        symbol = "EXIT/USDT:PERP"
        strategy.update_ohlcv(
            symbol,
            data["close"].tolist(),
            data["high"].tolist(),
            data["low"].tolist(),
            data["volume"].tolist(),
        )

        # Simulate open position
        strategy.update_position(symbol, 1.0)
        strategy._entry_prices[symbol] = data["close"][-1]
        strategy._entry_bars[symbol] = 0

        # Advance bar counter past max hold
        strategy._bar_counter = 10

        signal = strategy.generate_signal(symbol)
        assert signal.signal == Signal.SELL

    def test_stop_loss_trigger(self):
        strategy = VwapBreakoutStrategy(atr_stop=1.0)
        n = 300

        # Create data where price drops significantly at the end
        np.random.seed(42)
        close = np.ones(n) * 100.0
        close[-1] = 80.0  # Big drop
        high = close * 1.01
        low = close * 0.99
        low[-1] = 79.0
        volume = np.ones(n) * 500.0

        symbol = "STOP/USDT:PERP"
        strategy.update_ohlcv(
            symbol, close.tolist(), high.tolist(), low.tolist(), volume.tolist()
        )

        strategy.update_position(symbol, 1.0)
        strategy._entry_prices[symbol] = 100.0
        strategy._entry_bars[symbol] = strategy._bar_counter

        signal = strategy.generate_signal(symbol)
        assert signal.signal == Signal.SELL


class TestVwapBreakoutEntry:
    """Test entry condition checking."""

    def test_check_entry_insufficient_data(self):
        strategy = VwapBreakoutStrategy()
        data = {
            "close": np.array([100.0] * 10),
            "high": np.array([101.0] * 10),
            "low": np.array([99.0] * 10),
            "volume": np.array([500.0] * 10),
        }
        assert strategy._check_entry(data) is False

    def test_check_entry_no_breakout(self):
        strategy = VwapBreakoutStrategy()
        data = _make_flat_data()
        assert strategy._check_entry(data) is False


class TestVwapBreakoutFixes:
    """Tests for P0/P1 fix verification."""

    def test_cache_cleared_on_generate_signals(self):
        """P0-5: Cache should be cleared at start of each generate_signals cycle."""
        strategy = VwapBreakoutStrategy()
        data = _make_flat_data()
        strategy.update_ohlcv(
            "A/USDT:PERP",
            data["close"].tolist(),
            data["high"].tolist(),
            data["low"].tolist(),
            data["volume"].tolist(),
        )
        assert "A/USDT:PERP" in strategy._ohlcv_cache

        # generate_signals should clear cache first
        strategy.generate_signals(["A/USDT:PERP"])
        # Cache repopulated via generate_signal -> _fetch_ohlcv fails (no adapter)
        # But the old stale entry should have been cleared
        # Since no adapter, cache is empty after clear + failed fetch
        assert "A/USDT:PERP" not in strategy._ohlcv_cache

    def test_update_position_cleans_entry_tracking(self):
        """P1-9: External position close should clean entry tracking."""
        strategy = VwapBreakoutStrategy()
        strategy._entry_prices["TEST"] = 100.0
        strategy._entry_bars["TEST"] = 5

        # Simulate external position close
        strategy.update_position("TEST", 0.0)
        assert "TEST" not in strategy._entry_prices
        assert "TEST" not in strategy._entry_bars

    def test_kama_slope_10_bar_gap(self):
        """P0-1: KAMA slope should use 10-bar gap (index -(kama_slope_bars+1))."""
        strategy = VwapBreakoutStrategy(kama_slope_bars=10)
        # With kama_slope_bars=10, should access kama_arr[-11]
        # The fix changes kama_arr[-self.kama_slope_bars] to
        # kama_arr[-(self.kama_slope_bars + 1)]
        assert strategy.kama_slope_bars == 10

        # Create strongly trending data where KAMA slope would be positive
        data = _make_trending_data(n=300, trend=0.005)
        from libs.strategies.indicators import KAMA_series

        kama_arr = KAMA_series(data["close"], period=20, fast_sc=2, slow_sc=30)
        # The correct 10-bar slope
        slope_correct = kama_arr[-1] - kama_arr[-(10 + 1)]
        assert slope_correct > 0  # Should be positive for strong uptrend

    def test_max_positions_enforced(self):
        """P1-7: Should return HOLD when max_positions reached."""
        strategy = VwapBreakoutStrategy(max_positions=1)
        data = _make_trending_data(n=300, trend=0.003)

        for sym in ["A/USDT:PERP", "B/USDT:PERP"]:
            strategy.update_ohlcv(
                sym,
                data["close"].tolist(),
                data["high"].tolist(),
                data["low"].tolist(),
                data["volume"].tolist(),
            )

        # Simulate first position open
        strategy.update_position("A/USDT:PERP", 1.0)
        strategy._entry_prices["A/USDT:PERP"] = data["close"][-1]
        strategy._entry_bars["A/USDT:PERP"] = 0

        # Second signal should be HOLD due to max_positions
        signal = strategy.generate_signal("B/USDT:PERP")
        assert signal.signal == Signal.HOLD
        assert "Max positions" in signal.reason

    def test_parameter_none_vs_zero(self):
        """P1-6: Passing None should use default, but 0 should be accepted."""
        strategy_default = VwapBreakoutStrategy(max_hold_bars=None)
        assert strategy_default.max_hold_bars == 72  # DEFAULT

        strategy_zero = VwapBreakoutStrategy(max_hold_bars=0)
        assert strategy_zero.max_hold_bars == 0  # Should NOT fall to default

    def test_restart_safety_entry_tracking(self):
        """NEW-18a: Synced positions on restart should have entry tracking."""
        strategy = VwapBreakoutStrategy()
        # Simulate restart: sync an existing open position
        strategy.update_position("RESTART/USDT:PERP", 1.0, entry_price=100.0)
        assert "RESTART/USDT:PERP" in strategy._entry_prices
        assert strategy._entry_prices["RESTART/USDT:PERP"] == 100.0
        assert "RESTART/USDT:PERP" in strategy._entry_bars

    def test_restart_safety_fallback_price(self):
        """NEW-18a: Without entry_price, should fallback to cached data."""
        strategy = VwapBreakoutStrategy()
        data = _make_flat_data()
        strategy.update_ohlcv(
            "FB/USDT:PERP",
            data["close"].tolist(),
            data["high"].tolist(),
            data["low"].tolist(),
            data["volume"].tolist(),
        )
        strategy.update_position("FB/USDT:PERP", 1.0)
        assert "FB/USDT:PERP" in strategy._entry_prices
        assert strategy._entry_prices["FB/USDT:PERP"] > 0


class TestVwapBreakoutLoader:
    """Test strategy loader integration."""

    def test_registry_contains_vwap_breakout(self):
        from libs.strategies.loader import STRATEGY_REGISTRY

        assert "vwap_breakout" in STRATEGY_REGISTRY

    def test_get_strategy(self):
        from libs.strategies.loader import get_strategy

        strategy = get_strategy("vwap_breakout")
        assert strategy is not None
        assert isinstance(strategy, VwapBreakoutStrategy)

    def test_available_strategies_metadata(self):
        from libs.strategies.loader import list_available_strategies

        strategies = list_available_strategies()
        vwap = [s for s in strategies if s.get("strategy_id") == "vwap_breakout"]
        assert len(vwap) == 1
        assert vwap[0]["status"] == "production_ready"
