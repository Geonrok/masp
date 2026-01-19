"""Tests for risk metrics component."""
from __future__ import annotations

import math
from datetime import date


def test_import_risk_metrics():
    """Test module imports correctly."""
    from services.dashboard.components import risk_metrics

    assert hasattr(risk_metrics, "render_risk_metrics_panel")
    assert hasattr(risk_metrics, "calculate_risk_metrics")
    assert hasattr(risk_metrics, "calculate_sharpe_ratio")
    assert hasattr(risk_metrics, "calculate_max_drawdown")
    assert hasattr(risk_metrics, "RiskMetrics")


def test_safe_float_valid():
    """Test _safe_float with valid values."""
    from services.dashboard.components.risk_metrics import _safe_float

    assert _safe_float(1.5) == 1.5
    assert _safe_float(0) == 0.0
    assert _safe_float(-2.5) == -2.5


def test_safe_float_invalid():
    """Test _safe_float with invalid values."""
    from services.dashboard.components.risk_metrics import _safe_float

    assert _safe_float(None) == 0.0
    assert _safe_float("invalid") == 0.0
    assert _safe_float(float("inf")) == 0.0
    assert _safe_float(float("-inf")) == 0.0
    assert _safe_float(float("nan")) == 0.0


def test_safe_div_valid():
    """Test _safe_div with valid values."""
    from services.dashboard.components.risk_metrics import _safe_div

    assert _safe_div(10, 2) == 5.0
    assert _safe_div(0, 5) == 0.0
    assert _safe_div(-10, 2) == -5.0


def test_safe_div_zero_denominator():
    """Test _safe_div with zero denominator."""
    from services.dashboard.components.risk_metrics import _safe_div

    assert _safe_div(10, 0) == 0.0
    assert _safe_div(10, 0, default=-1.0) == -1.0


def test_safe_sqrt_valid():
    """Test _safe_sqrt with valid values."""
    from services.dashboard.components.risk_metrics import _safe_sqrt

    assert _safe_sqrt(4) == 2.0
    assert _safe_sqrt(0) == 0.0
    assert abs(_safe_sqrt(2) - 1.4142135623730951) < 0.0001


def test_safe_sqrt_negative():
    """Test _safe_sqrt with negative value."""
    from services.dashboard.components.risk_metrics import _safe_sqrt

    assert _safe_sqrt(-4) == 0.0
    assert _safe_sqrt(-4, default=1.0) == 1.0


def test_calculate_sharpe_ratio_basic():
    """Test Sharpe ratio calculation."""
    from services.dashboard.components.risk_metrics import calculate_sharpe_ratio

    returns = [0.01, 0.02, -0.01, 0.015, 0.005] * 10  # 50 days
    sharpe = calculate_sharpe_ratio(returns)

    assert isinstance(sharpe, float)
    assert math.isfinite(sharpe)


def test_calculate_sharpe_ratio_empty():
    """Test Sharpe ratio with empty returns."""
    from services.dashboard.components.risk_metrics import calculate_sharpe_ratio

    assert calculate_sharpe_ratio([]) == 0.0
    assert calculate_sharpe_ratio([0.01]) == 0.0  # Less than 2 samples


def test_calculate_sharpe_ratio_zero_std():
    """Test Sharpe ratio with zero standard deviation."""
    from services.dashboard.components.risk_metrics import calculate_sharpe_ratio

    returns = [0.01, 0.01, 0.01, 0.01, 0.01]  # No variance
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    # With zero variance, sharpe should be 0
    assert sharpe == 0.0


def test_calculate_sortino_ratio_basic():
    """Test Sortino ratio calculation."""
    from services.dashboard.components.risk_metrics import calculate_sortino_ratio

    returns = [0.01, 0.02, -0.01, 0.015, -0.005] * 10
    sortino = calculate_sortino_ratio(returns)

    assert isinstance(sortino, float)
    assert math.isfinite(sortino)


def test_calculate_sortino_ratio_no_downside():
    """Test Sortino ratio with no downside."""
    from services.dashboard.components.risk_metrics import calculate_sortino_ratio

    returns = [0.01, 0.02, 0.015, 0.005, 0.01]  # All positive
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino == 0.0  # No downside returns


def test_calculate_max_drawdown_basic():
    """Test max drawdown calculation."""
    from services.dashboard.components.risk_metrics import calculate_max_drawdown

    equity_curve = [100, 110, 105, 115, 100, 120]
    max_dd, drawdowns = calculate_max_drawdown(equity_curve)

    assert isinstance(max_dd, float)
    assert max_dd > 0  # Should have some drawdown
    assert len(drawdowns) == len(equity_curve)


def test_calculate_max_drawdown_empty():
    """Test max drawdown with empty equity curve."""
    from services.dashboard.components.risk_metrics import calculate_max_drawdown

    max_dd, drawdowns = calculate_max_drawdown([])
    assert max_dd == 0.0
    assert drawdowns == []


def test_calculate_max_drawdown_single_value():
    """Test max drawdown with single value."""
    from services.dashboard.components.risk_metrics import calculate_max_drawdown

    max_dd, drawdowns = calculate_max_drawdown([100])
    assert max_dd == 0.0


def test_calculate_max_drawdown_no_drawdown():
    """Test max drawdown with monotonically increasing curve."""
    from services.dashboard.components.risk_metrics import calculate_max_drawdown

    equity_curve = [100, 110, 120, 130, 140]
    max_dd, drawdowns = calculate_max_drawdown(equity_curve)

    assert max_dd == 0.0
    assert all(d == 0.0 for d in drawdowns)


def test_calculate_calmar_ratio_basic():
    """Test Calmar ratio calculation."""
    from services.dashboard.components.risk_metrics import calculate_calmar_ratio

    calmar = calculate_calmar_ratio(total_return_pct=20.0, max_drawdown_pct=10.0)
    assert calmar == 2.0


def test_calculate_calmar_ratio_zero_drawdown():
    """Test Calmar ratio with zero drawdown."""
    from services.dashboard.components.risk_metrics import calculate_calmar_ratio

    calmar = calculate_calmar_ratio(total_return_pct=20.0, max_drawdown_pct=0.0)
    assert calmar == 0.0


def test_calculate_volatility_basic():
    """Test volatility calculation."""
    from services.dashboard.components.risk_metrics import calculate_volatility

    returns = [0.01, -0.01, 0.02, -0.02, 0.01] * 10
    vol = calculate_volatility(returns)

    assert isinstance(vol, float)
    assert vol > 0  # Should have some volatility


def test_calculate_volatility_empty():
    """Test volatility with empty returns."""
    from services.dashboard.components.risk_metrics import calculate_volatility

    assert calculate_volatility([]) == 0.0
    assert calculate_volatility([0.01]) == 0.0


def test_calculate_win_rate_basic():
    """Test win rate calculation."""
    from services.dashboard.components.risk_metrics import calculate_win_rate

    returns = [0.01, -0.01, 0.02, -0.02, 0.01]  # 3 wins, 2 losses
    win_rate = calculate_win_rate(returns)

    assert win_rate == 60.0  # 3/5 = 60%


def test_calculate_win_rate_empty():
    """Test win rate with empty returns."""
    from services.dashboard.components.risk_metrics import calculate_win_rate

    assert calculate_win_rate([]) == 0.0


def test_calculate_win_rate_all_wins():
    """Test win rate with all positive returns."""
    from services.dashboard.components.risk_metrics import calculate_win_rate

    returns = [0.01, 0.02, 0.03, 0.04]
    win_rate = calculate_win_rate(returns)

    assert win_rate == 100.0


def test_calculate_profit_factor_basic():
    """Test profit factor calculation."""
    from services.dashboard.components.risk_metrics import calculate_profit_factor

    returns = [0.02, 0.03, -0.01, -0.02]  # Profit: 0.05, Loss: 0.03
    pf = calculate_profit_factor(returns)

    assert abs(pf - (0.05 / 0.03)) < 0.0001


def test_calculate_profit_factor_no_loss():
    """Test profit factor with no losses."""
    from services.dashboard.components.risk_metrics import calculate_profit_factor

    returns = [0.01, 0.02, 0.03]
    pf = calculate_profit_factor(returns)

    assert pf == 0.0  # Avoid inf


def test_calculate_profit_factor_empty():
    """Test profit factor with empty returns."""
    from services.dashboard.components.risk_metrics import calculate_profit_factor

    assert calculate_profit_factor([]) == 0.0


def test_calculate_risk_metrics_comprehensive():
    """Test comprehensive risk metrics calculation."""
    from services.dashboard.components.risk_metrics import calculate_risk_metrics

    returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 10
    equity_curve = [100]
    for r in returns:
        equity_curve.append(equity_curve[-1] * (1 + r))

    metrics = calculate_risk_metrics(returns, equity_curve)

    assert metrics.trading_days == len(returns)
    assert math.isfinite(metrics.sharpe_ratio)
    assert math.isfinite(metrics.sortino_ratio)
    assert math.isfinite(metrics.max_drawdown_pct)
    assert metrics.max_drawdown_pct >= 0
    assert 0 <= metrics.win_rate_pct <= 100


def test_calculate_risk_metrics_empty():
    """Test risk metrics with empty data."""
    from services.dashboard.components.risk_metrics import calculate_risk_metrics

    metrics = calculate_risk_metrics([], [])

    assert metrics.sharpe_ratio == 0.0
    assert metrics.sortino_ratio == 0.0
    assert metrics.max_drawdown_pct == 0.0
    assert metrics.trading_days == 0


def test_get_demo_returns():
    """Test demo returns generation."""
    from services.dashboard.components.risk_metrics import _get_demo_returns

    returns = _get_demo_returns()

    assert len(returns) == 90
    assert all(isinstance(r, float) for r in returns)


def test_get_demo_returns_deterministic():
    """Test demo returns are deterministic."""
    from services.dashboard.components.risk_metrics import _get_demo_returns

    returns1 = _get_demo_returns()
    returns2 = _get_demo_returns()

    assert returns1 == returns2


def test_get_demo_equity_curve():
    """Test demo equity curve generation."""
    from services.dashboard.components.risk_metrics import _get_demo_equity_curve

    equity_curve = _get_demo_equity_curve()

    assert len(equity_curve) == 91  # 90 returns + initial value
    assert equity_curve[0] == 100_000_000
    assert all(isinstance(e, (int, float)) for e in equity_curve)


def test_get_demo_dates():
    """Test demo dates generation."""
    from services.dashboard.components.risk_metrics import _get_demo_dates

    dates = _get_demo_dates()

    assert len(dates) == 91
    assert all(isinstance(d, date) for d in dates)


def test_key_function():
    """Test _key generates namespaced keys."""
    from services.dashboard.components.risk_metrics import _key, _KEY_PREFIX

    result = _key("chart")
    assert result == f"{_KEY_PREFIX}chart"
    assert result == "risk_metrics.chart"


def test_risk_metrics_dataclass():
    """Test RiskMetrics dataclass."""
    from services.dashboard.components.risk_metrics import RiskMetrics

    metrics = RiskMetrics(
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown_pct=10.0,
        calmar_ratio=1.2,
        volatility_pct=15.0,
        win_rate_pct=55.0,
        profit_factor=1.8,
        avg_return_pct=0.05,
        total_return_pct=12.0,
        trading_days=252,
    )

    assert metrics.sharpe_ratio == 1.5
    assert metrics.trading_days == 252


def test_calculate_max_drawdown_known_values():
    """Test max drawdown with known expected values."""
    from services.dashboard.components.risk_metrics import calculate_max_drawdown

    # Peak at 120, drops to 90 = 25% drawdown
    equity_curve = [100, 110, 120, 100, 90, 110]
    max_dd, _ = calculate_max_drawdown(equity_curve)

    assert abs(max_dd - 25.0) < 0.01  # 25% max drawdown


def test_sanitize_returns():
    """Test _sanitize_returns filters non-finite values."""
    from services.dashboard.components.risk_metrics import _sanitize_returns

    returns = [0.01, float("nan"), 0.02, float("inf"), -0.01, float("-inf")]
    sanitized = _sanitize_returns(returns)

    assert len(sanitized) == 6
    assert sanitized[0] == 0.01
    assert sanitized[1] == 0.0  # nan -> 0
    assert sanitized[2] == 0.02
    assert sanitized[3] == 0.0  # inf -> 0
    assert sanitized[4] == -0.01
    assert sanitized[5] == 0.0  # -inf -> 0


def test_generate_equity_from_returns():
    """Test equity curve generation from returns."""
    from services.dashboard.components.risk_metrics import _generate_equity_from_returns

    returns = [0.1, -0.05, 0.02]  # 10%, -5%, 2%
    equity = _generate_equity_from_returns(returns, initial_equity=100)

    assert len(equity) == 4  # initial + 3 returns
    assert equity[0] == 100.0
    assert abs(equity[1] - 110.0) < 0.01  # 100 * 1.1
    assert abs(equity[2] - 104.5) < 0.01  # 110 * 0.95
    assert abs(equity[3] - 106.59) < 0.01  # 104.5 * 1.02


def test_generate_returns_from_equity():
    """Test returns generation from equity curve."""
    from services.dashboard.components.risk_metrics import _generate_returns_from_equity

    equity = [100, 110, 104.5, 106.59]
    returns = _generate_returns_from_equity(equity)

    assert len(returns) == 3
    assert abs(returns[0] - 0.1) < 0.001  # 10%
    assert abs(returns[1] - (-0.05)) < 0.001  # -5%
    assert abs(returns[2] - 0.02) < 0.001  # 2%


def test_generate_returns_from_equity_empty():
    """Test returns generation with insufficient equity curve."""
    from services.dashboard.components.risk_metrics import _generate_returns_from_equity

    assert _generate_returns_from_equity([]) == []
    assert _generate_returns_from_equity([100]) == []


def test_generate_dates_for_length():
    """Test date series generation."""
    from services.dashboard.components.risk_metrics import _generate_dates_for_length

    dates = _generate_dates_for_length(5)

    assert len(dates) == 5
    assert all(isinstance(d, date) for d in dates)
    # Check consecutive days
    for i in range(1, len(dates)):
        diff = (dates[i] - dates[i - 1]).days
        assert diff == 1


def test_sharpe_with_nan_returns():
    """Test Sharpe ratio handles NaN in returns."""
    from services.dashboard.components.risk_metrics import calculate_sharpe_ratio

    returns = [0.01, float("nan"), 0.02, 0.015, -0.01] * 10
    sharpe = calculate_sharpe_ratio(returns)

    assert isinstance(sharpe, float)
    assert math.isfinite(sharpe)


def test_sharpe_ratio_zero_trading_days():
    """Test Sharpe ratio with zero trading days."""
    from services.dashboard.components.risk_metrics import calculate_sharpe_ratio

    returns = [0.01, 0.02, -0.01, 0.015] * 10
    sharpe = calculate_sharpe_ratio(returns, trading_days=0)

    assert sharpe == 0.0


def test_sharpe_ratio_negative_trading_days():
    """Test Sharpe ratio with negative trading days."""
    from services.dashboard.components.risk_metrics import calculate_sharpe_ratio

    returns = [0.01, 0.02, -0.01, 0.015] * 10
    sharpe = calculate_sharpe_ratio(returns, trading_days=-1)

    assert sharpe == 0.0


def test_sortino_ratio_zero_trading_days():
    """Test Sortino ratio with zero trading days."""
    from services.dashboard.components.risk_metrics import calculate_sortino_ratio

    returns = [0.01, 0.02, -0.01, 0.015] * 10
    sortino = calculate_sortino_ratio(returns, trading_days=0)

    assert sortino == 0.0


def test_sharpe_ratio_nan_risk_free_rate():
    """Test Sharpe ratio with NaN risk-free rate."""
    from services.dashboard.components.risk_metrics import calculate_sharpe_ratio

    returns = [0.01, 0.02, -0.01, 0.015] * 10
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=float("nan"))

    assert isinstance(sharpe, float)
    assert math.isfinite(sharpe)
