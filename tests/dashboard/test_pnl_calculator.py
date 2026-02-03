"""Tests for PnL calculator."""

from __future__ import annotations

from services.dashboard.utils.pnl_calculator import (
    PositionPnL,
    _safe_float,
    calculate_portfolio_pnl,
    calculate_total_pnl,
)


def test_position_pnl_profit():
    pnl = PositionPnL(
        symbol="BTC",
        quantity=1.0,
        avg_price=50000.0,
        current_price=55000.0,
    )
    assert pnl.cost_basis == 50000.0
    assert pnl.current_value == 55000.0
    assert pnl.pnl_amount == 5000.0
    assert pnl.pnl_percent == 10.0


def test_position_pnl_loss():
    pnl = PositionPnL(
        symbol="ETH",
        quantity=2.0,
        avg_price=3000.0,
        current_price=2700.0,
    )
    assert pnl.pnl_amount == -600.0
    assert pnl.pnl_percent == -10.0


def test_calculate_portfolio_pnl():
    positions = [
        {"symbol": "BTC", "quantity": 1.0, "avg_price": 50000.0},
        {"symbol": "ETH", "quantity": 10.0, "avg_price": 3000.0},
    ]
    prices = {"BTC": 55000.0, "ETH": 3300.0}

    result = calculate_portfolio_pnl(positions, prices)
    assert len(result) == 2
    assert result[0].symbol == "BTC"
    assert result[1].pnl_percent == 10.0


def test_calculate_total_pnl():
    pnl_list = [
        PositionPnL("BTC", 1.0, 50000.0, 55000.0),
        PositionPnL("ETH", 10.0, 3000.0, 3300.0),
    ]
    total = calculate_total_pnl(pnl_list)

    assert total["total_cost"] == 80000.0
    assert total["total_value"] == 88000.0
    assert total["total_pnl"] == 8000.0
    assert total["total_pnl_percent"] == 10.0


def test_zero_quantity_excluded():
    positions = [
        {"symbol": "BTC", "quantity": 0, "avg_price": 50000.0},
    ]
    result = calculate_portfolio_pnl(positions, {"BTC": 55000.0})
    assert len(result) == 0


def test_safe_float_none_input():
    assert _safe_float(None) == 0.0
    assert _safe_float(None, 99.0) == 99.0


def test_safe_float_invalid_string():
    assert _safe_float("invalid") == 0.0
    assert _safe_float("") == 0.0


def test_safe_float_nan():
    assert _safe_float(float("nan")) == 0.0


def test_safe_float_inf():
    assert _safe_float(float("inf")) == 0.0
    assert _safe_float(float("-inf")) == 0.0


def test_portfolio_pnl_missing_symbol():
    positions = [
        {"symbol": "", "quantity": 1.0, "avg_price": 100.0},
        {"quantity": 1.0, "avg_price": 100.0},
    ]
    result = calculate_portfolio_pnl(positions, {})
    assert len(result) == 0


def test_portfolio_pnl_invalid_values():
    positions = [
        {"symbol": "BTC", "quantity": "invalid", "avg_price": 100.0},
        {"symbol": "ETH", "quantity": 1.0, "avg_price": None},
    ]
    result = calculate_portfolio_pnl(positions, {"BTC": 100.0, "ETH": 100.0})
    assert len(result) == 0
