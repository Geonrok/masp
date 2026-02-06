"""Tests for Kiwoom strategy status provider."""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import patch

import pytest

from services.dashboard.providers.kiwoom_provider import (
    KiwoomAccount,
    KiwoomConfig,
    KiwoomPosition,
    KiwoomStatusData,
    KiwoomTradeStats,
    _compute_account_summary,
    _compute_trade_stats,
    _get_demo_status,
    _get_stock_name,
    _infer_positions,
    _load_strategy_config,
    _norm_side,
    _safe_float,
)


class TestSafeFloat:
    """Tests for _safe_float helper."""

    def test_none(self):
        assert _safe_float(None) == 0.0

    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string(self):
        assert _safe_float("123.45") == 123.45

    def test_invalid_string(self):
        assert _safe_float("abc") == 0.0

    def test_empty_string(self):
        assert _safe_float("") == 0.0

    def test_default(self):
        assert _safe_float(None, 99.9) == 99.9


class TestNormSide:
    """Tests for _norm_side helper."""

    def test_buy_variations(self):
        assert _norm_side("BUY") == "BUY"
        assert _norm_side("buy") == "BUY"
        assert _norm_side("B") == "BUY"
        assert _norm_side("LONG") == "BUY"

    def test_sell_variations(self):
        assert _norm_side("SELL") == "SELL"
        assert _norm_side("sell") == "SELL"
        assert _norm_side("S") == "SELL"
        assert _norm_side("SHORT") == "SELL"

    def test_unknown(self):
        assert _norm_side("UNKNOWN") == "UNKNOWN"
        assert _norm_side(None) == ""


class TestGetStockName:
    """Tests for _get_stock_name helper."""

    def test_known_stocks(self):
        assert _get_stock_name("000660") == "SK하이닉스"
        assert _get_stock_name("005930") == "삼성전자"

    def test_unknown_stock(self):
        assert _get_stock_name("999999") == "999999"


class TestLoadStrategyConfig:
    """Tests for _load_strategy_config."""

    def test_returns_config_object(self):
        config = _load_strategy_config()
        assert isinstance(config, KiwoomConfig)

    def test_mode_defaults_to_paper(self):
        with patch.dict(os.environ, {}, clear=True):
            config = _load_strategy_config()
            assert config.mode == "paper"

    def test_mode_live_when_env_set(self):
        with patch.dict(os.environ, {"MASP_ENABLE_LIVE_TRADING": "1"}):
            config = _load_strategy_config()
            assert config.mode == "live"


class TestComputeAccountSummary:
    """Tests for _compute_account_summary."""

    def test_empty_trades(self):
        account = _compute_account_summary([])
        assert isinstance(account, KiwoomAccount)
        assert account.total_pnl == 0.0
        assert account.total_fees == 0.0
        assert account.estimated_balance == 1_000_000.0

    def test_with_pnl_and_fees(self):
        trades = [
            {"pnl": "100.0", "fee": "10.0", "quantity": "1", "price": "1000"},
            {"pnl": "50.0", "fee": "5.0", "quantity": "1", "price": "1000"},
        ]
        account = _compute_account_summary(trades)
        assert account.total_pnl == 150.0
        assert account.total_fees == 15.0
        assert account.estimated_balance == 1_000_000.0 + 150.0 - 15.0

    def test_pnl_percent(self):
        trades = [{"pnl": "10000", "fee": "0", "quantity": "1", "price": "1000"}]
        account = _compute_account_summary(trades)
        assert account.pnl_percent == 1.0  # 10000 / 1_000_000 * 100


class TestComputeTradeStats:
    """Tests for _compute_trade_stats."""

    def test_empty_trades(self):
        stats = _compute_trade_stats([])
        assert isinstance(stats, KiwoomTradeStats)
        assert stats.trades_total == 0
        assert stats.buy_count == 0
        assert stats.sell_count == 0

    def test_buy_sell_counts(self):
        trades = [
            {"side": "BUY", "symbol": "005930", "timestamp": "2026-01-01T10:00:00"},
            {"side": "BUY", "symbol": "000660", "timestamp": "2026-01-01T10:01:00"},
            {"side": "SELL", "symbol": "005930", "timestamp": "2026-01-02T10:00:00"},
        ]
        stats = _compute_trade_stats(trades)
        assert stats.trades_total == 3
        assert stats.buy_count == 2
        assert stats.sell_count == 1
        assert stats.unique_symbols == 2
        assert stats.rebalance_count == 2  # 2 unique dates

    def test_first_last_trade_date(self):
        trades = [
            {"side": "BUY", "timestamp": "2026-02-05T15:00:00"},
            {"side": "SELL", "timestamp": "2026-01-15T10:00:00"},
        ]
        stats = _compute_trade_stats(trades)
        assert stats.last_trade_date == "2026-02-05"
        assert stats.first_trade_date == "2026-01-15"


class TestInferPositions:
    """Tests for _infer_positions."""

    def test_empty(self):
        positions = _infer_positions([])
        assert positions == []

    def test_single_buy_creates_long(self):
        trades = [
            {"symbol": "005930", "side": "BUY", "quantity": "10", "price": "70000"},
        ]
        positions = _infer_positions(trades)
        assert len(positions) == 1
        assert positions[0].symbol == "005930"
        assert positions[0].side == "LONG"
        assert positions[0].net_quantity == 10.0
        assert positions[0].avg_entry_price == 70000.0

    def test_multiple_buys_weighted_avg(self):
        trades = [
            {"symbol": "005930", "side": "BUY", "quantity": "10", "price": "70000"},
            {"symbol": "005930", "side": "BUY", "quantity": "10", "price": "72000"},
        ]
        positions = _infer_positions(trades)
        assert len(positions) == 1
        assert positions[0].net_quantity == 20.0
        assert positions[0].avg_entry_price == 71000.0  # (700000 + 720000) / 20

    def test_buy_sell_equal_is_flat(self):
        trades = [
            {"symbol": "005930", "side": "BUY", "quantity": "10", "price": "70000"},
            {"symbol": "005930", "side": "SELL", "quantity": "10", "price": "72000"},
        ]
        positions = _infer_positions(trades)
        assert positions == []  # Net quantity is 0

    def test_partial_close(self):
        trades = [
            {"symbol": "005930", "side": "BUY", "quantity": "10", "price": "70000"},
            {"symbol": "005930", "side": "SELL", "quantity": "5", "price": "72000"},
        ]
        positions = _infer_positions(trades)
        assert len(positions) == 1
        assert positions[0].net_quantity == 5.0
        assert positions[0].side == "LONG"

    def test_multiple_symbols(self):
        trades = [
            {"symbol": "005930", "side": "BUY", "quantity": "10", "price": "70000"},
            {"symbol": "000660", "side": "BUY", "quantity": "5", "price": "180000"},
        ]
        positions = _infer_positions(trades)
        assert len(positions) == 2

    def test_sorted_by_notional_desc(self):
        trades = [
            {"symbol": "005930", "side": "BUY", "quantity": "10", "price": "70000"},
            {"symbol": "000660", "side": "BUY", "quantity": "5", "price": "180000"},
        ]
        positions = _infer_positions(trades)
        # 000660: 5 * 180000 = 900000
        # 005930: 10 * 70000 = 700000
        assert positions[0].symbol == "000660"
        assert positions[1].symbol == "005930"


class TestGetDemoStatus:
    """Tests for _get_demo_status."""

    def test_returns_valid_data(self):
        status = _get_demo_status()
        assert isinstance(status, KiwoomStatusData)
        assert status.data_available is False
        assert len(status.positions) > 0
        assert len(status.recent_trades) > 0

    def test_config_loaded(self):
        status = _get_demo_status()
        assert isinstance(status.config, KiwoomConfig)


class TestImports:
    """Tests for module imports."""

    def test_import_component(self):
        from services.dashboard.components.kiwoom_status import render_kiwoom_status

        assert callable(render_kiwoom_status)

    def test_import_provider(self):
        from services.dashboard.providers.kiwoom_provider import get_kiwoom_status

        assert callable(get_kiwoom_status)
