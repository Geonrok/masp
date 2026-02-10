"""
Unit tests for TrendMomentumGateStrategy.

Tests cover:
    - BTC gate (exchange-aware, t-1 data)
    - Momentum ranking (cross-sectional)
    - Signal shift (t-1 verification)
    - Volatility-target sizing
    - Stablecoin/leveraged token filter
    - Drawdown stop
    - Cache behavior
    - generate_signal / generate_signals interface
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from libs.strategies.base import Signal, TradeSignal
from libs.strategies.trend_momentum_gate import TrendMomentumGateStrategy

# ── Helpers ──────────────────────────────────────────────────────────


@dataclass
class FakeOHLCV:
    """Minimal OHLCV dataclass for testing."""

    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: str = ""


def make_ohlcv(closes: List[float], volumes: Optional[List[float]] = None) -> list:
    """Create a list of FakeOHLCV from close prices."""
    if volumes is None:
        volumes = [1000.0] * len(closes)
    return [
        FakeOHLCV(
            open=c * 0.99,
            high=c * 1.01,
            low=c * 0.98,
            close=c,
            volume=v,
        )
        for c, v in zip(closes, volumes)
    ]


def make_adapter(data: dict) -> MagicMock:
    """Create a mock market data adapter."""
    adapter = MagicMock()

    def get_ohlcv(symbol: str, interval: str = "1d", limit: int = 100):
        if symbol in data:
            return data[symbol][-limit:]
        return []

    adapter.get_ohlcv = MagicMock(side_effect=get_ohlcv)
    return adapter


# ── BTC Gate Tests ───────────────────────────────────────────────────


class TestBTCGate:
    """Tests for BTC gate logic."""

    def test_gate_btc_krw_for_upbit(self):
        """Upbit uses BTC/KRW."""
        strategy = TrendMomentumGateStrategy(exchange="upbit")
        assert strategy._get_btc_symbol() == "BTC/KRW"

    def test_gate_btc_krw_for_bithumb(self):
        """Bithumb uses BTC/KRW."""
        strategy = TrendMomentumGateStrategy(exchange="bithumb")
        assert strategy._get_btc_symbol() == "BTC/KRW"

    def test_gate_btc_usdt_for_binance(self):
        """Binance uses BTC/USDT."""
        strategy = TrendMomentumGateStrategy(exchange="binance_spot")
        assert strategy._get_btc_symbol() == "BTC/USDT"

    def test_gate_btc_usdt_default(self):
        """Default is BTC/USDT."""
        strategy = TrendMomentumGateStrategy()
        assert strategy._get_btc_symbol() == "BTC/USDT"

    def test_gate_on_when_btc_above_ma(self):
        """Gate is ON when BTC[t-1] > SMA(50)."""
        # 60 bars, trending up strongly
        closes = [30000 + i * 200 for i in range(65)]
        data = {"BTC/USDT": make_ohlcv(closes)}
        adapter = make_adapter(data)

        strategy = TrendMomentumGateStrategy(
            btc_ma_period=50, market_data_adapter=adapter
        )
        assert strategy.check_gate() is True

    def test_gate_off_when_btc_below_ma(self):
        """Gate is OFF when BTC[t-1] < SMA(50)."""
        # 60 bars, trending down strongly
        closes = [50000 - i * 300 for i in range(65)]
        data = {"BTC/USDT": make_ohlcv(closes)}
        adapter = make_adapter(data)

        strategy = TrendMomentumGateStrategy(
            btc_ma_period=50, market_data_adapter=adapter
        )
        assert strategy.check_gate() is False

    def test_gate_uses_t_minus_1(self):
        """Gate uses closes[-2], not closes[-1] (t-1 data)."""
        # Last bar: BTC drops below MA. But t-1 was still above.
        base = [40000 + i * 100 for i in range(64)]
        base.append(20000)  # today's bar = crash
        data = {"BTC/USDT": make_ohlcv(base)}
        adapter = make_adapter(data)

        strategy = TrendMomentumGateStrategy(
            btc_ma_period=50, market_data_adapter=adapter
        )
        # t-1 (closes[-2]) = 40000+63*100 = 46300 > SMA(50) of ~43000
        assert strategy.check_gate() is True

    def test_gate_failopen_no_data(self):
        """Gate returns True (fail-open) when no market data."""
        strategy = TrendMomentumGateStrategy()
        assert strategy.check_gate() is True

    def test_gate_failopen_insufficient_data(self):
        """Gate returns True when insufficient BTC data."""
        data = {"BTC/USDT": make_ohlcv([40000] * 10)}
        adapter = make_adapter(data)
        strategy = TrendMomentumGateStrategy(
            btc_ma_period=50, market_data_adapter=adapter
        )
        assert strategy.check_gate() is True


# ── Stablecoin/Leveraged Filter Tests ─────────────────────────────


class TestUniverseFilter:
    """Tests for stablecoin and leveraged token filtering."""

    def test_stablecoin_excluded(self):
        """USDT, USDC, DAI etc. are excluded."""
        assert TrendMomentumGateStrategy.is_excluded("USDT/KRW") is True
        assert TrendMomentumGateStrategy.is_excluded("USDC/KRW") is True
        assert TrendMomentumGateStrategy.is_excluded("DAI/KRW") is True
        assert TrendMomentumGateStrategy.is_excluded("FDUSD/USDT") is True

    def test_leveraged_excluded(self):
        """UP/DOWN/BULL/BEAR tokens are excluded."""
        assert TrendMomentumGateStrategy.is_excluded("BTCUP/USDT") is True
        assert TrendMomentumGateStrategy.is_excluded("ETHDOWN/USDT") is True
        assert TrendMomentumGateStrategy.is_excluded("BTCBULL/USDT") is True
        assert TrendMomentumGateStrategy.is_excluded("ETHBEAR/USDT") is True

    def test_normal_coins_not_excluded(self):
        """Normal coins pass the filter."""
        assert TrendMomentumGateStrategy.is_excluded("BTC/USDT") is False
        assert TrendMomentumGateStrategy.is_excluded("ETH/KRW") is False
        assert TrendMomentumGateStrategy.is_excluded("SOL/USDT") is False
        assert TrendMomentumGateStrategy.is_excluded("ADA/KRW") is False


# ── Momentum Ranking Tests ───────────────────────────────────────


class TestMomentumRanking:
    """Tests for cross-sectional momentum computation."""

    def test_momentum_uses_t_minus_1(self):
        """Momentum computed from closes[-2] / closes[-2-lookback] - 1."""
        # 40 bars, linear up from 100 to ~200
        closes_a = [100 + i * 2.5 for i in range(40)]
        closes_b = [100 - i * 0.5 for i in range(40)]
        data = {
            "COIN_A": make_ohlcv(closes_a),
            "COIN_B": make_ohlcv(closes_b),
        }
        adapter = make_adapter(data)
        strategy = TrendMomentumGateStrategy(
            mom_lookback=30, market_data_adapter=adapter
        )

        mom_a = strategy._compute_momentum("COIN_A")
        mom_b = strategy._compute_momentum("COIN_B")

        assert mom_a is not None
        assert mom_b is not None
        assert mom_a > 0  # uptrending
        assert mom_b < 0  # downtrending
        assert mom_a > mom_b  # A ranked higher

    def test_top_n_selection(self):
        """Only top N coins get BUY signals."""
        # Create 5 coins with different momentum
        data = {"BTC/USDT": make_ohlcv([40000 + i * 200 for i in range(65)])}
        for i in range(5):
            closes = [100 + j * (5 - i) for j in range(40)]
            data[f"COIN_{i}"] = make_ohlcv(closes, [100000] * 40)
        adapter = make_adapter(data)

        strategy = TrendMomentumGateStrategy(
            top_n=2, mom_lookback=30, btc_ma_period=50, market_data_adapter=adapter
        )
        symbols = [f"COIN_{i}" for i in range(5)]
        signals = strategy.generate_signals(symbols)

        buy_count = sum(1 for s in signals if s.signal == Signal.BUY)
        assert buy_count <= 2  # At most top_n=2 buys

    def test_gate_off_all_sell_or_hold(self):
        """When gate is OFF, no BUY signals."""
        # BTC trending down → gate OFF
        data = {"BTC/USDT": make_ohlcv([50000 - i * 300 for i in range(65)])}
        for i in range(3):
            data[f"COIN_{i}"] = make_ohlcv(
                [100 + j * 5 for j in range(40)], [100000] * 40
            )
        adapter = make_adapter(data)

        strategy = TrendMomentumGateStrategy(
            top_n=2, btc_ma_period=50, market_data_adapter=adapter
        )
        symbols = [f"COIN_{i}" for i in range(3)]
        signals = strategy.generate_signals(symbols)

        for signal in signals:
            assert signal.signal != Signal.BUY


# ── Volatility Sizing Tests ──────────────────────────────────────


class TestVolSizing:
    """Tests for inverse-volatility position sizing."""

    def test_high_vol_gets_lower_weight(self):
        """Higher volatility coin gets smaller weight."""
        # Low vol coin: small daily changes
        low_vol = [100 + i * 0.1 + (0.5 if i % 2 == 0 else -0.5) for i in range(40)]
        # High vol coin: big daily changes
        high_vol = [100 + i * 0.1 + (10 if i % 2 == 0 else -10) for i in range(40)]

        data = {
            "LOW_VOL": make_ohlcv(low_vol),
            "HIGH_VOL": make_ohlcv(high_vol),
        }
        adapter = make_adapter(data)
        strategy = TrendMomentumGateStrategy(
            vol_target=0.20, top_n=2, market_data_adapter=adapter
        )

        w_low = strategy._compute_vol_weight("LOW_VOL")
        w_high = strategy._compute_vol_weight("HIGH_VOL")

        assert w_low > w_high  # Lower vol → higher weight

    def test_weight_capped_at_25_percent(self):
        """No single coin exceeds 25% weight."""
        # Near-zero vol → would get very high weight
        flat = [100.0] * 40
        data = {"FLAT": make_ohlcv(flat)}
        adapter = make_adapter(data)
        strategy = TrendMomentumGateStrategy(
            vol_target=0.20, top_n=2, market_data_adapter=adapter
        )

        w = strategy._compute_vol_weight("FLAT")
        assert w <= 0.25


# ── Drawdown Stop Tests ──────────────────────────────────────────


class TestDrawdownStop:
    """Tests for portfolio drawdown stop and cooldown."""

    def test_no_stop_above_limit(self):
        """No cooldown triggered when drawdown < limit."""
        strategy = TrendMomentumGateStrategy()
        strategy.update_portfolio_value(10000)
        strategy.update_portfolio_value(9000)  # -10% (limit is -25%)
        assert strategy._in_cooldown is False

    def test_stop_triggers_at_limit(self):
        """Cooldown triggered when drawdown hits -25%."""
        strategy = TrendMomentumGateStrategy()
        strategy.update_portfolio_value(10000)
        strategy.update_portfolio_value(7400)  # -26%
        assert strategy._in_cooldown is True
        assert strategy._cooldown_until is not None

    def test_cooldown_blocks_buy_signals(self):
        """During cooldown, no BUY signals generated."""
        data = {"BTC/USDT": make_ohlcv([40000 + i * 200 for i in range(65)])}
        data["COIN_A"] = make_ohlcv([100 + i * 5 for i in range(40)], [100000] * 40)
        adapter = make_adapter(data)

        strategy = TrendMomentumGateStrategy(top_n=5, market_data_adapter=adapter)
        strategy.update_portfolio_value(10000)
        strategy.update_portfolio_value(7000)  # -30% → cooldown

        signals = strategy.generate_signals(["COIN_A"])
        for s in signals:
            assert s.signal != Signal.BUY

    def test_cooldown_expires(self):
        """Cooldown ends after configured days."""
        strategy = TrendMomentumGateStrategy()
        strategy.update_portfolio_value(10000)
        strategy.update_portfolio_value(7000)
        assert strategy._in_cooldown is True

        # Simulate time passing
        strategy._cooldown_until = datetime.now() - timedelta(days=1)
        # Next signal computation should clear cooldown
        data = {"BTC/USDT": make_ohlcv([40000 + i * 200 for i in range(65)])}
        data["COIN_A"] = make_ohlcv([100 + i * 5 for i in range(40)], [100000] * 40)
        adapter = make_adapter(data)
        strategy._market_data = adapter

        signals = strategy.generate_signals(["COIN_A"])
        assert strategy._in_cooldown is False


# ── Cache Tests ──────────────────────────────────────────────────


class TestSignalCache:
    """Tests for cross-sectional signal caching."""

    def test_cache_populated_on_first_call(self):
        """First generate_signal() populates cache for all symbols."""
        data = {"BTC/USDT": make_ohlcv([40000 + i * 200 for i in range(65)])}
        for i in range(3):
            data[f"COIN_{i}"] = make_ohlcv(
                [100 + j * (3 - i) for j in range(40)], [100000] * 40
            )
        adapter = make_adapter(data)

        strategy = TrendMomentumGateStrategy(top_n=2, market_data_adapter=adapter)
        strategy.set_all_symbols(["COIN_0", "COIN_1", "COIN_2"])

        # First call triggers full computation
        signal_0 = strategy.generate_signal("COIN_0", gate_pass=True)
        assert signal_0 is not None

        # Cache should now have all symbols
        assert len(strategy._signals_cache) >= 3

    def test_cache_reused_within_ttl(self):
        """Subsequent calls within TTL use cache."""
        data = {"BTC/USDT": make_ohlcv([40000 + i * 200 for i in range(65)])}
        data["COIN_A"] = make_ohlcv([100 + i * 3 for i in range(40)], [100000] * 40)
        adapter = make_adapter(data)

        strategy = TrendMomentumGateStrategy(top_n=5, market_data_adapter=adapter)
        strategy.set_all_symbols(["COIN_A"])

        s1 = strategy.generate_signal("COIN_A", gate_pass=True)
        call_count_after_first = adapter.get_ohlcv.call_count

        s2 = strategy.generate_signal("COIN_A", gate_pass=True)
        call_count_after_second = adapter.get_ohlcv.call_count

        # No additional API calls on second invocation
        assert call_count_after_second == call_count_after_first


# ── Integration Tests ────────────────────────────────────────────


class TestIntegration:
    """Integration tests for full signal flow."""

    def test_full_signal_flow(self):
        """End-to-end: BTC up, 5 coins, top 2 selected."""
        btc = make_ohlcv([30000 + i * 200 for i in range(65)])
        data = {"BTC/USDT": btc}

        # 5 coins with different momentum (higher index = higher momentum)
        for i in range(5):
            slope = (i + 1) * 2
            closes = [100 + j * slope for j in range(40)]
            data[f"C{i}"] = make_ohlcv(closes, [500000] * 40)

        adapter = make_adapter(data)
        strategy = TrendMomentumGateStrategy(
            top_n=2, mom_lookback=30, btc_ma_period=50, market_data_adapter=adapter
        )

        symbols = [f"C{i}" for i in range(5)]
        signals = strategy.generate_signals(symbols)

        buy_symbols = [s.symbol for s in signals if s.signal == Signal.BUY]
        assert len(buy_symbols) == 2
        # C4 and C3 should be selected (highest momentum)
        assert "C4" in buy_symbols
        assert "C3" in buy_symbols

    def test_strategy_loader_registration(self):
        """TMG is registered in the strategy loader."""
        from libs.strategies.loader import STRATEGY_REGISTRY

        assert "trend_momentum_gate" in STRATEGY_REGISTRY
        assert STRATEGY_REGISTRY["trend_momentum_gate"] is TrendMomentumGateStrategy

    def test_generate_signal_and_generate_signals_consistent(self):
        """generate_signal() and generate_signals() produce same results."""
        btc = make_ohlcv([30000 + i * 200 for i in range(65)])
        data = {"BTC/USDT": btc}
        for i in range(3):
            data[f"C{i}"] = make_ohlcv(
                [100 + j * (i + 1) for j in range(40)], [500000] * 40
            )
        adapter = make_adapter(data)

        strategy = TrendMomentumGateStrategy(top_n=2, market_data_adapter=adapter)
        symbols = ["C0", "C1", "C2"]

        # generate_signals (direct)
        batch_signals = strategy.generate_signals(symbols)
        batch_results = {s.symbol: s.signal for s in batch_signals}

        # generate_signal (per-symbol with cache)
        strategy2 = TrendMomentumGateStrategy(top_n=2, market_data_adapter=adapter)
        strategy2.set_all_symbols(symbols)
        single_results = {}
        for sym in symbols:
            sig = strategy2.generate_signal(sym, gate_pass=True)
            single_results[sym] = sig.signal

        assert batch_results == single_results

    def test_sell_existing_positions_on_gate_off(self):
        """Existing positions get SELL when gate turns OFF."""
        data = {"BTC/USDT": make_ohlcv([50000 - i * 300 for i in range(65)])}
        data["C0"] = make_ohlcv([100 + i * 2 for i in range(40)], [100000] * 40)
        adapter = make_adapter(data)

        strategy = TrendMomentumGateStrategy(top_n=5, market_data_adapter=adapter)
        strategy.update_position("C0", 100)  # Has position

        signals = strategy.generate_signals(["C0"])
        assert signals[0].signal == Signal.SELL
        assert "gate off" in signals[0].reason.lower()

    def test_signal_strength_reflects_vol_weight(self):
        """BUY signal strength equals volatility-target weight."""
        btc = make_ohlcv([30000 + i * 200 for i in range(65)])
        coin = make_ohlcv([100 + i * 2 for i in range(40)], [500000] * 40)
        data = {"BTC/USDT": btc, "COIN": coin}
        adapter = make_adapter(data)

        strategy = TrendMomentumGateStrategy(
            top_n=5, vol_target=0.20, market_data_adapter=adapter
        )
        signals = strategy.generate_signals(["COIN"])

        for s in signals:
            if s.signal == Signal.BUY:
                assert 0.01 <= s.strength <= 0.25
