"""Tests for backtest viewer component."""

from __future__ import annotations

import math
from datetime import date


def test_import_backtest_viewer():
    """Test module imports correctly."""
    from services.dashboard.components import backtest_viewer

    assert hasattr(backtest_viewer, "render_backtest_viewer")
    assert hasattr(backtest_viewer, "_safe_float")
    assert hasattr(backtest_viewer, "_safe_div")
    assert hasattr(backtest_viewer, "_get_demo_backtest_data")
    assert hasattr(backtest_viewer, "_calculate_equity_curve")
    assert hasattr(backtest_viewer, "_calculate_drawdown")
    assert hasattr(backtest_viewer, "_calculate_metrics")


def test_demo_data_structure():
    """Test demo data has required structure (100 days, all keys)."""
    from services.dashboard.components.backtest_viewer import _get_demo_backtest_data

    data = _get_demo_backtest_data()

    # Required keys
    assert "dates" in data
    assert "daily_returns" in data
    assert "initial_capital" in data
    assert "strategy_name" in data

    # Real BTC data (1260 trading days)
    assert len(data["dates"]) >= 100
    assert len(data["daily_returns"]) == len(data["dates"])

    # Start date from real data
    assert data["dates"][0] == date(2019, 1, 1)

    # Deterministic: calling twice gives same results
    data2 = _get_demo_backtest_data()
    assert data["daily_returns"] == data2["daily_returns"]


def test_equity_curve_calculation():
    """Test equity curve calculation with cumprod."""
    from services.dashboard.components.backtest_viewer import _calculate_equity_curve

    initial = 1_000_000
    returns = [0.01, 0.02, -0.01]  # +1%, +2%, -1%

    equity = _calculate_equity_curve(returns, initial)

    assert len(equity) == 3
    # Day 1: 1,000,000 * 1.01 = 1,010,000
    assert abs(equity[0] - 1_010_000) < 1
    # Day 2: 1,010,000 * 1.02 = 1,030,200
    assert abs(equity[1] - 1_030_200) < 1
    # Day 3: 1,030,200 * 0.99 = 1,019,898
    assert abs(equity[2] - 1_019_898) < 1


def test_equity_curve_empty():
    """Test equity curve with empty returns."""
    from services.dashboard.components.backtest_viewer import _calculate_equity_curve

    equity = _calculate_equity_curve([], 1_000_000)
    assert equity == [1_000_000]


def test_drawdown_calculation():
    """Test drawdown calculation (0 at new peaks)."""
    from services.dashboard.components.backtest_viewer import _calculate_drawdown

    # Equity: 100 -> 110 -> 105 -> 115
    equity = [100, 110, 105, 115]
    dd = _calculate_drawdown(equity)

    assert len(dd) == 4
    # At 100 (initial peak): 0
    assert dd[0] == 0.0
    # At 110 (new peak): 0
    assert dd[1] == 0.0
    # At 105 (5 below peak of 110): -5/110 ≈ -0.0455
    assert abs(dd[2] - (-5 / 110)) < 0.0001
    # At 115 (new peak): 0
    assert dd[3] == 0.0


def test_drawdown_empty():
    """Test drawdown with empty equity curve."""
    from services.dashboard.components.backtest_viewer import _calculate_drawdown

    dd = _calculate_drawdown([])
    assert dd == []


def test_metrics_calculation():
    """Test metrics calculation with known values."""
    from services.dashboard.components.backtest_viewer import _calculate_metrics

    initial = 1_000_000
    # 10 days of varying positive returns (need variance for Sharpe calculation)
    returns = [0.015, 0.008, 0.012, 0.005, 0.018, 0.007, 0.011, 0.009, 0.014, 0.006]
    # Calculate expected equity
    equity = [initial]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    equity = equity[1:]

    metrics = _calculate_metrics(returns, equity, initial)

    # Total return: (final/initial - 1) * 100
    expected_final = initial
    for r in returns:
        expected_final *= 1 + r
    expected_total = (expected_final / initial - 1) * 100
    assert abs(metrics["total_return"] - expected_total) < 0.01

    # Win rate: 100% (all positive)
    assert metrics["win_rate"] == 100.0

    # Sharpe > 0 (positive returns with variance)
    assert metrics["sharpe"] > 0

    # MDD = 0 (always increasing)
    assert metrics["mdd"] == 0.0


def test_metrics_calculation_constant_returns():
    """Test Sharpe=0 when all returns are identical (zero variance)."""
    from services.dashboard.components.backtest_viewer import _calculate_metrics

    initial = 1_000_000
    # All identical returns -> std = 0 -> Sharpe undefined, returns 0
    returns = [0.01] * 10
    equity = [initial]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    equity = equity[1:]

    metrics = _calculate_metrics(returns, equity, initial)

    # Sharpe should be 0 when std = 0 (mathematically undefined)
    assert metrics["sharpe"] == 0.0
    # Win rate still 100%
    assert metrics["win_rate"] == 100.0


def test_zero_returns_handling():
    """Test Sharpe=0 when std=0 (all same returns)."""
    from services.dashboard.components.backtest_viewer import _calculate_metrics

    initial = 1_000_000
    # All zero returns -> std = 0
    returns = [0.0] * 10
    equity = [initial] * 10

    metrics = _calculate_metrics(returns, equity, initial)

    # Sharpe should be 0 when std = 0
    assert metrics["sharpe"] == 0.0
    # Win rate should be 0 (no positive returns)
    assert metrics["win_rate"] == 0.0


def test_safe_float():
    """Test _safe_float handles various inputs."""
    from services.dashboard.components.backtest_viewer import _safe_float

    # Normal values
    assert _safe_float(1.5) == 1.5
    assert _safe_float(0) == 0.0
    assert _safe_float(-10) == -10.0

    # None
    assert _safe_float(None) == 0.0
    assert _safe_float(None, 5.0) == 5.0

    # Non-finite
    assert _safe_float(float("inf")) == 0.0
    assert _safe_float(float("-inf")) == 0.0
    assert _safe_float(float("nan")) == 0.0

    # Invalid types
    assert _safe_float("invalid") == 0.0
    assert _safe_float([1, 2, 3]) == 0.0


def test_safe_div():
    """Test _safe_div handles edge cases."""
    from services.dashboard.components.backtest_viewer import _safe_div

    # Normal division
    assert _safe_div(10, 2) == 5.0
    assert _safe_div(1, 4) == 0.25

    # Zero denominator
    assert _safe_div(10, 0) == 0.0
    assert _safe_div(10, 0, -1.0) == -1.0

    # Inf denominator
    assert _safe_div(10, float("inf")) == 0.0

    # Result is inf
    assert _safe_div(float("inf"), 1) == 0.0


def test_nonfinite_handling():
    """Test metrics handle non-finite values in returns."""
    from services.dashboard.components.backtest_viewer import _calculate_metrics

    initial = 1_000_000
    # Returns with non-finite values (should be treated as 0)
    returns = [0.01, float("nan"), 0.02, float("inf"), -0.01]
    equity = [
        initial * 1.01,
        initial * 1.01,
        initial * 1.0302,
        initial * 1.0302,
        initial * 1.02,
    ]

    metrics = _calculate_metrics(returns, equity, initial)

    # Should not crash and return finite values
    assert math.isfinite(metrics["total_return"])
    assert math.isfinite(metrics["sharpe"])
    assert math.isfinite(metrics["mdd"])
    assert math.isfinite(metrics["win_rate"])


def test_empty_data_handling():
    """Test metrics handle empty data gracefully."""
    from services.dashboard.components.backtest_viewer import _calculate_metrics

    metrics = _calculate_metrics([], [], 1_000_000)

    assert metrics["total_return"] == 0.0
    assert metrics["cagr"] == 0.0
    assert metrics["mdd"] == 0.0
    assert metrics["sharpe"] == 0.0
    assert metrics["win_rate"] == 0.0
