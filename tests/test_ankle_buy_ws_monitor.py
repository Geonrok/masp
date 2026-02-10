"""Tests for Ankle Buy v2.0 WebSocket Monitor."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from services.ankle_buy_ws_monitor import GATE_HYSTERESIS_PCT, AnkleBuyWSMonitor


def _make_strategy_mock():
    """Create a mock strategy with pos_info."""
    strategy = MagicMock()
    strategy._pos_info = {}
    strategy.get_position_info.return_value = None
    strategy.get_btc_sma50_today.return_value = 50000.0
    strategy._check_tp_levels.return_value = (0.0, "")
    strategy._compute_upper_sma.return_value = 100.0
    strategy._get_ohlcv.return_value = None
    strategy.check_gate.return_value = True
    strategy.clear_cache.return_value = None
    strategy._cleanup_position.return_value = None
    return strategy


def _make_monitor(exchange_name="upbit", quote_currency="KRW"):
    """Create a monitor with mocked dependencies."""
    strategy = _make_strategy_mock()
    execution = MagicMock()
    market_data = MagicMock()
    symbols = ["BTC/KRW", "ETH/KRW", "XRP/KRW"]

    monitor = AnkleBuyWSMonitor(
        exchange_name=exchange_name,
        strategy=strategy,
        execution=execution,
        market_data=market_data,
        symbols=symbols,
        min_position=50000,
        quote_currency=quote_currency,
    )
    monitor.btc_sma50 = 50000.0
    return monitor


class TestSymbolConversion:
    """Test symbol format conversion."""

    def test_upbit_strategy_to_ws(self):
        m = _make_monitor("upbit")
        assert m._strategy_to_ws("BTC/KRW") == "KRW-BTC"
        assert m._strategy_to_ws("ETH/KRW") == "KRW-ETH"

    def test_upbit_ws_to_strategy(self):
        m = _make_monitor("upbit")
        assert m._ws_to_strategy("KRW-BTC") == "BTC/KRW"
        assert m._ws_to_strategy("KRW-ETH") == "ETH/KRW"

    def test_binance_strategy_to_ws(self):
        m = _make_monitor("binance_spot", "USDT")
        assert m._strategy_to_ws("BTC/USDT") == "btcusdt"
        assert m._strategy_to_ws("ETH/USDT") == "ethusdt"

    def test_binance_ws_to_strategy(self):
        m = _make_monitor("binance_spot", "USDT")
        assert m._ws_to_strategy("btcusdt") == "BTC/USDT"
        assert m._ws_to_strategy("ethusdt") == "ETH/USDT"

    def test_bithumb_strategy_to_ws(self):
        m = _make_monitor("bithumb")
        assert m._strategy_to_ws("BTC/KRW") == "BTC_KRW"

    def test_bithumb_ws_to_strategy(self):
        m = _make_monitor("bithumb")
        assert m._ws_to_strategy("BTC_KRW") == "BTC/KRW"


class TestTickParsing:
    """Test exchange-specific tick parsing."""

    def test_upbit_ticker(self):
        m = _make_monitor("upbit")
        ws_sym, price = m._parse_tick({"code": "KRW-BTC", "trade_price": 50000000})
        assert ws_sym == "KRW-BTC"
        assert price == 50000000.0

    def test_binance_24hr_ticker(self):
        m = _make_monitor("binance_spot", "USDT")
        ws_sym, price = m._parse_tick(
            {"e": "24hrTicker", "s": "BTCUSDT", "c": "50000.00"}
        )
        assert ws_sym == "btcusdt"
        assert price == 50000.0

    def test_binance_kline(self):
        m = _make_monitor("binance_spot", "USDT")
        ws_sym, price = m._parse_tick(
            {"e": "kline", "k": {"s": "BTCUSDT", "c": "51000.00", "x": True}}
        )
        assert ws_sym == "btcusdt"
        assert price == 51000.0

    def test_bithumb_ticker(self):
        m = _make_monitor("bithumb")
        ws_sym, price = m._parse_tick(
            {
                "type": "ticker",
                "content": {"symbol": "BTC_KRW", "closePrice": "50000000"},
            }
        )
        assert ws_sym == "BTC_KRW"
        assert price == 50000000.0

    def test_invalid_message(self):
        m = _make_monitor("upbit")
        ws_sym, price = m._parse_tick({})
        assert ws_sym is None
        assert price == 0


class TestGateRealtime:
    """Test BTC Gate real-time with hysteresis."""

    def test_gate_off_when_price_below_sma50(self):
        m = _make_monitor()
        m.gate_status = True
        m.btc_sma50 = 50000.0

        m._check_gate_realtime(49000.0)  # Below SMA50
        assert m.gate_status is False

    def test_gate_stays_on_above_sma50(self):
        m = _make_monitor()
        m.gate_status = True
        m.btc_sma50 = 50000.0

        m._check_gate_realtime(51000.0)  # Above SMA50
        assert m.gate_status is True

    def test_gate_recovery_requires_hysteresis(self):
        m = _make_monitor()
        m.gate_status = False
        m.btc_sma50 = 50000.0

        # Just above SMA50 - not enough (need 0.5% above)
        m._check_gate_realtime(50100.0)
        assert m.gate_status is False

        # Above hysteresis threshold
        threshold = 50000.0 * (1 + GATE_HYSTERESIS_PCT / 100)
        m._check_gate_realtime(threshold + 1)
        assert m.gate_status is True

    def test_gate_off_triggers_liquidation(self):
        m = _make_monitor()
        m.gate_status = True
        m.btc_sma50 = 50000.0
        m.strategy._pos_info = {"ETH/KRW": {"entry_price": 3000}}

        with patch.object(m, "_schedule_sell") as mock_sell:
            m._check_gate_realtime(49000.0)
            mock_sell.assert_called_once_with("ETH/KRW", 1.0, "BTC Gate OFF (realtime)")

    def test_gate_no_action_without_sma50(self):
        m = _make_monitor()
        m.btc_sma50 = None
        m.gate_status = True

        m._check_gate_realtime(49000.0)
        assert m.gate_status is True  # No change when SMA50 unknown


class TestExitChecks:
    """Test real-time exit condition checks."""

    def test_stop_loss_triggers_sell(self):
        m = _make_monitor()
        info = {"stop_loss": 95.0, "entry_price": 100.0, "tp_sold": set()}

        with patch.object(m, "_schedule_sell") as mock_sell:
            m._check_exit_realtime("ETH/KRW", 94.0, info)
            mock_sell.assert_called_once()
            assert mock_sell.call_args[0][0] == "ETH/KRW"
            assert mock_sell.call_args[0][1] == 1.0  # Full sell

    def test_tp_triggers_partial_sell(self):
        m = _make_monitor()
        info = {"stop_loss": 80.0, "entry_price": 100.0, "tp_sold": set()}
        m.strategy._check_tp_levels.return_value = (0.1, "TP +10%")

        with patch.object(m, "_schedule_sell") as mock_sell:
            m._check_exit_realtime("ETH/KRW", 111.0, info)
            mock_sell.assert_called_once_with("ETH/KRW", 0.1, "TP +10%")

    def test_sma_exit_with_cache(self):
        m = _make_monitor()
        info = {"stop_loss": 80.0, "entry_price": 100.0, "tp_sold": set()}
        m._sma_cache["ETH/KRW"] = {"upper_sma": 98.0, "today_open": 99.0}
        m.strategy._check_tp_levels.return_value = (0.0, "")

        with patch.object(m, "_schedule_sell") as mock_sell:
            m._check_exit_realtime("ETH/KRW", 97.0, info)  # < open and <= SMA
            mock_sell.assert_called_once_with("ETH/KRW", 1.0, "SMA exit (realtime)")

    def test_no_exit_when_all_conditions_pass(self):
        m = _make_monitor()
        info = {"stop_loss": 80.0, "entry_price": 100.0, "tp_sold": set()}
        m.strategy._check_tp_levels.return_value = (0.0, "")

        with patch.object(m, "_schedule_sell") as mock_sell:
            m._check_exit_realtime("ETH/KRW", 105.0, info)
            mock_sell.assert_not_called()


class TestDoubleExecution:
    """Test double execution prevention."""

    def test_duplicate_sell_blocked(self):
        m = _make_monitor()
        m._executing.add("ETH/KRW")

        with patch.object(m, "_execute_sell_sync") as mock_exec:
            m._schedule_sell("ETH/KRW", 1.0, "test")
            mock_exec.assert_not_called()

    def test_is_executing_check(self):
        m = _make_monitor()
        assert m.is_executing("ETH/KRW") is False
        m._executing.add("ETH/KRW")
        assert m.is_executing("ETH/KRW") is True


class TestDailyCloseEntry:
    """Test daily close entry logic."""

    def test_buys_sorted_by_volume(self):
        """_execute_buys_sync processes signals in given order (pre-sorted by caller)."""
        m = _make_monitor()
        m.execution.get_balance.return_value = 1000000

        # Pre-sorted by volume descending (as _on_daily_close would do)
        signals = [
            ("XRP/KRW", MagicMock(), 5000000),  # High volume - first
            ("ETH/KRW", MagicMock(), 500000),  # Low volume - second
        ]

        m._execute_buys_sync(signals)

        calls = m.execution.place_order.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == "XRP/KRW"  # First buy (highest volume)
        assert calls[1][0][0] == "ETH/KRW"  # Second buy

    def test_balance_exhaustion_stops_buying(self):
        m = _make_monitor()
        m.min_position = 50000
        m.execution.get_balance.return_value = 60000  # Only enough for 1

        signals = [
            ("ETH/KRW", MagicMock(), 1000),
            ("XRP/KRW", MagicMock(), 2000),
        ]

        m._execute_buys_sync(signals)

        # Should only buy 1 (balance exhausted after first)
        assert m.execution.place_order.call_count == 1

    def test_no_balance_no_buys(self):
        m = _make_monitor()
        m.execution.get_balance.return_value = 0

        signals = [("ETH/KRW", MagicMock(), 1000)]
        m._execute_buys_sync(signals)

        m.execution.place_order.assert_not_called()


class TestOnTick:
    """Test the main tick handler."""

    def test_btc_tick_triggers_gate_check(self):
        m = _make_monitor("upbit")

        with patch.object(m, "_check_gate_realtime") as mock_gate:
            m._on_tick({"code": "KRW-BTC", "trade_price": 49000.0})
            mock_gate.assert_called_once_with(49000.0)

    def test_position_tick_triggers_exit_check(self):
        m = _make_monitor("upbit")
        info = {"stop_loss": 80.0, "entry_price": 100.0}
        m.strategy.get_position_info.return_value = info

        with patch.object(m, "_check_exit_realtime") as mock_exit:
            m._on_tick({"code": "KRW-ETH", "trade_price": 3000000})
            mock_exit.assert_called_once_with("ETH/KRW", 3000000.0, info)

    def test_non_position_tick_ignored(self):
        m = _make_monitor("upbit")
        m.strategy.get_position_info.return_value = None

        with patch.object(m, "_check_exit_realtime") as mock_exit:
            m._on_tick({"code": "KRW-DOGE", "trade_price": 100})
            mock_exit.assert_not_called()

    def test_invalid_tick_ignored(self):
        m = _make_monitor("upbit")

        with patch.object(m, "_check_gate_realtime") as mock_gate:
            m._on_tick({})  # No code/price
            mock_gate.assert_not_called()
