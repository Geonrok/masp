"""Tests for Binance Futures MR Strategy provider."""

from __future__ import annotations

import pytest

from services.dashboard.providers.futures_mr_provider import (
    FuturesMRAccount,
    FuturesMRConfig,
    FuturesMRPosition,
    FuturesMRStatusData,
    FuturesMRTradeStats,
    _compute_account_summary,
    _compute_trade_stats,
    _get_demo_status,
    _infer_positions,
    _load_strategy_config,
    _safe_float,
)


class TestSafeFloat:
    def test_none(self):
        assert _safe_float(None) == 0.0

    def test_int(self):
        assert _safe_float(5) == 5.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string(self):
        assert _safe_float("42.5") == 42.5

    def test_invalid_string(self):
        assert _safe_float("abc") == 0.0

    def test_empty_string(self):
        assert _safe_float("") == 0.0

    def test_default(self):
        assert _safe_float(None, -1.0) == -1.0


class TestLoadStrategyConfig:
    def test_returns_config_object(self):
        config = _load_strategy_config()
        assert isinstance(config, FuturesMRConfig)
        assert config.strategy_name  # Should have a name

    def test_mode_defaults_to_paper(self, monkeypatch):
        monkeypatch.delenv("MASP_ENABLE_LIVE_TRADING", raising=False)
        config = _load_strategy_config()
        assert config.mode == "paper"

    def test_mode_live_when_env_set(self, monkeypatch):
        monkeypatch.setenv("MASP_ENABLE_LIVE_TRADING", "1")
        config = _load_strategy_config()
        assert config.mode == "live"


class TestComputeAccountSummary:
    def test_empty_trades(self):
        acct = _compute_account_summary([])
        assert acct.initial_balance == 10_000.0
        assert acct.estimated_balance == 10_000.0
        assert acct.total_pnl == 0.0
        assert acct.total_fees == 0.0

    def test_with_pnl_and_fees(self):
        trades = [
            {"pnl": "100.50", "fee": "2.0", "quantity": "0.01", "price": "70000"},
            {"pnl": "-50.00", "fee": "1.5", "quantity": "0.02", "price": "71000"},
        ]
        acct = _compute_account_summary(trades)
        assert abs(acct.total_pnl - 50.5) < 0.01
        assert abs(acct.total_fees - 3.5) < 0.01
        assert abs(acct.estimated_balance - (10_000 + 50.5 - 3.5)) < 0.01

    def test_pnl_percent(self):
        trades = [{"pnl": "500.0", "fee": "100.0", "quantity": "1", "price": "100"}]
        acct = _compute_account_summary(trades)
        # (500 - 100) / 10000 * 100 = 4.0%
        assert abs(acct.pnl_percent - 4.0) < 0.01


class TestComputeTradeStats:
    def test_empty_trades(self):
        stats = _compute_trade_stats([])
        assert stats.trades_total == 0
        assert stats.buy_count == 0
        assert stats.sell_count == 0

    def test_buy_sell_counts(self):
        trades = [
            {"side": "BUY", "symbol": "BTC/USDT:PERP", "timestamp": "2026-01-01T00:00:00"},
            {"side": "BUY", "symbol": "ETH/USDT:PERP", "timestamp": "2026-01-01T00:00:00"},
            {"side": "SELL", "symbol": "BTC/USDT:PERP", "timestamp": "2026-01-01T00:00:00"},
        ]
        stats = _compute_trade_stats(trades)
        assert stats.trades_total == 3
        assert stats.buy_count == 2
        assert stats.sell_count == 1
        assert stats.unique_symbols == 2

    def test_first_last_trade_date(self):
        trades = [
            {"side": "BUY", "symbol": "BTC/USDT:PERP", "timestamp": "2026-02-05T10:00:00"},
            {"side": "SELL", "symbol": "BTC/USDT:PERP", "timestamp": "2026-02-01T10:00:00"},
        ]
        stats = _compute_trade_stats(trades)
        assert stats.last_trade_date == "2026-02-05"
        assert stats.first_trade_date == "2026-02-01"


class TestInferPositions:
    def test_empty(self):
        assert _infer_positions([]) == []

    def test_single_buy_creates_long(self):
        trades = [
            {"symbol": "BTC/USDT:PERP", "side": "BUY", "quantity": "0.01", "price": "70000"},
        ]
        positions = _infer_positions(trades)
        assert len(positions) == 1
        assert positions[0].side == "LONG"
        assert abs(positions[0].net_quantity - 0.01) < 1e-8
        assert abs(positions[0].avg_entry_price - 70000) < 0.01

    def test_multiple_buys_weighted_avg(self):
        trades = [
            {"symbol": "BTC/USDT:PERP", "side": "BUY", "quantity": "0.01", "price": "70000"},
            {"symbol": "BTC/USDT:PERP", "side": "BUY", "quantity": "0.02", "price": "71000"},
        ]
        positions = _infer_positions(trades)
        assert len(positions) == 1
        assert abs(positions[0].net_quantity - 0.03) < 1e-8
        # Weighted avg: (0.01*70000 + 0.02*71000) / 0.03 = 70666.67
        expected_avg = (0.01 * 70000 + 0.02 * 71000) / 0.03
        assert abs(positions[0].avg_entry_price - expected_avg) < 0.01

    def test_buy_sell_equal_is_flat(self):
        trades = [
            {"symbol": "BTC/USDT:PERP", "side": "BUY", "quantity": "0.01", "price": "70000"},
            {"symbol": "BTC/USDT:PERP", "side": "SELL", "quantity": "0.01", "price": "71000"},
        ]
        positions = _infer_positions(trades)
        assert len(positions) == 0

    def test_partial_close(self):
        trades = [
            {"symbol": "BTC/USDT:PERP", "side": "BUY", "quantity": "0.03", "price": "70000"},
            {"symbol": "BTC/USDT:PERP", "side": "SELL", "quantity": "0.01", "price": "71000"},
        ]
        positions = _infer_positions(trades)
        assert len(positions) == 1
        assert abs(positions[0].net_quantity - 0.02) < 1e-8

    def test_multiple_symbols(self):
        trades = [
            {"symbol": "BTC/USDT:PERP", "side": "BUY", "quantity": "0.01", "price": "70000"},
            {"symbol": "ETH/USDT:PERP", "side": "BUY", "quantity": "1.0", "price": "2500"},
        ]
        positions = _infer_positions(trades)
        assert len(positions) == 2
        symbols = {p.symbol for p in positions}
        assert symbols == {"BTC/USDT:PERP", "ETH/USDT:PERP"}

    def test_sorted_by_notional_desc(self):
        trades = [
            {"symbol": "ETH/USDT:PERP", "side": "BUY", "quantity": "1.0", "price": "2500"},
            {"symbol": "BTC/USDT:PERP", "side": "BUY", "quantity": "0.01", "price": "70000"},
        ]
        positions = _infer_positions(trades)
        # ETH notional = 2500, BTC notional = 700
        assert positions[0].symbol == "ETH/USDT:PERP"


class TestGetDemoStatus:
    def test_returns_valid_data(self):
        data = _get_demo_status()
        assert isinstance(data, FuturesMRStatusData)
        assert data.data_available is False
        assert data.account.initial_balance == 10_000.0
        assert len(data.recent_trades) > 0
        assert len(data.positions) > 0

    def test_config_loaded(self):
        data = _get_demo_status()
        assert data.config.strategy_name


class TestImports:
    def test_import_component(self):
        from services.dashboard.components.futures_mr_status import (
            render_futures_mr_status,
        )

        assert callable(render_futures_mr_status)

    def test_import_provider(self):
        from services.dashboard.providers.futures_mr_provider import (
            get_futures_mr_status,
        )

        assert callable(get_futures_mr_status)
