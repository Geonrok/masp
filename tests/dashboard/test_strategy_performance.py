"""Tests for strategy performance component."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest


def test_import_strategy_performance():
    """Test module imports correctly."""
    from services.dashboard.components import strategy_performance

    assert hasattr(strategy_performance, "render_strategy_performance")
    assert hasattr(strategy_performance, "StrategyPerformance")
    assert hasattr(strategy_performance, "PerformanceMetrics")
    assert hasattr(strategy_performance, "TradeStats")
    assert hasattr(strategy_performance, "TimePeriod")


def test_time_period_enum():
    """Test TimePeriod enum values."""
    from services.dashboard.components.strategy_performance import TimePeriod

    assert TimePeriod.DAY_1.value == "1D"
    assert TimePeriod.WEEK_1.value == "1W"
    assert TimePeriod.MONTH_1.value == "1M"
    assert TimePeriod.MONTH_3.value == "3M"
    assert TimePeriod.YEAR_1.value == "1Y"
    assert TimePeriod.ALL.value == "ALL"


def test_trade_stats_dataclass_defaults():
    """Test TradeStats dataclass defaults."""
    from services.dashboard.components.strategy_performance import TradeStats

    stats = TradeStats()
    assert stats.total_trades == 0
    assert stats.winning_trades == 0
    assert stats.losing_trades == 0
    assert stats.win_rate == 0.0
    assert stats.profit_factor == 0.0


def test_trade_stats_with_values():
    """Test TradeStats dataclass with values."""
    from services.dashboard.components.strategy_performance import TradeStats

    stats = TradeStats(
        total_trades=100,
        winning_trades=60,
        losing_trades=40,
        win_rate=60.0,
        avg_win=1000.0,
        avg_loss=500.0,
        profit_factor=2.0,
        avg_holding_time_hours=12.5,
    )

    assert stats.total_trades == 100
    assert stats.winning_trades == 60
    assert stats.avg_win == 1000.0


def test_performance_metrics_dataclass_defaults():
    """Test PerformanceMetrics dataclass defaults."""
    from services.dashboard.components.strategy_performance import PerformanceMetrics

    metrics = PerformanceMetrics()
    assert metrics.total_return == 0.0
    assert metrics.sharpe_ratio == 0.0
    assert metrics.max_drawdown == 0.0
    assert metrics.volatility == 0.0


def test_performance_metrics_with_values():
    """Test PerformanceMetrics dataclass with values."""
    from services.dashboard.components.strategy_performance import PerformanceMetrics

    metrics = PerformanceMetrics(
        total_return=15.5,
        total_return_krw=1_550_000.0,
        sharpe_ratio=1.8,
        sortino_ratio=2.1,
        max_drawdown=10.0,
        max_drawdown_krw=1_000_000.0,
        volatility=20.0,
        calmar_ratio=1.55,
    )

    assert metrics.total_return == 15.5
    assert metrics.sharpe_ratio == 1.8
    assert metrics.max_drawdown == 10.0


def test_strategy_performance_dataclass():
    """Test StrategyPerformance dataclass."""
    from services.dashboard.components.strategy_performance import (
        StrategyPerformance,
        PerformanceMetrics,
        TradeStats,
    )

    strategy = StrategyPerformance(
        strategy_id="test_001",
        strategy_name="Test Strategy",
        is_active=True,
        start_date=datetime.now(),
    )

    assert strategy.strategy_id == "test_001"
    assert strategy.strategy_name == "Test Strategy"
    assert strategy.is_active is True
    assert isinstance(strategy.metrics, PerformanceMetrics)
    assert isinstance(strategy.trade_stats, TradeStats)


def test_key_function():
    """Test _key generates namespaced keys."""
    from services.dashboard.components.strategy_performance import _key, _KEY_PREFIX

    result = _key("period")
    assert result == f"{_KEY_PREFIX}period"
    assert result == "strategy_perf.period"


def test_safe_float_valid():
    """Test _safe_float with valid values."""
    from services.dashboard.components.strategy_performance import _safe_float

    assert _safe_float(1.5) == 1.5
    assert _safe_float(0) == 0.0
    assert _safe_float("-2.5") == -2.5


def test_safe_float_invalid():
    """Test _safe_float with invalid values."""
    from services.dashboard.components.strategy_performance import _safe_float

    assert _safe_float(None) == 0.0
    assert _safe_float("invalid") == 0.0
    assert _safe_float(float("inf")) == 0.0
    assert _safe_float(float("nan")) == 0.0


def test_safe_divide_valid():
    """Test _safe_divide with valid values."""
    from services.dashboard.components.strategy_performance import _safe_divide

    assert _safe_divide(10.0, 2.0) == 5.0
    assert _safe_divide(100.0, 4.0) == 25.0
    assert _safe_divide(0.0, 5.0) == 0.0


def test_safe_divide_zero_denominator():
    """Test _safe_divide with zero denominator."""
    from services.dashboard.components.strategy_performance import _safe_divide

    assert _safe_divide(10.0, 0.0) == 0.0
    assert _safe_divide(100.0, 0.0, default=-1.0) == -1.0


def test_safe_divide_invalid_values():
    """Test _safe_divide with invalid values."""
    from services.dashboard.components.strategy_performance import _safe_divide

    assert _safe_divide(None, 2.0) == 0.0
    assert _safe_divide(10.0, None) == 0.0
    assert _safe_divide(float("inf"), 2.0) == 0.0


def test_format_percent_positive():
    """Test _format_percent with positive values."""
    from services.dashboard.components.strategy_performance import _format_percent

    result = _format_percent(15.5)
    assert result == "+15.50%"


def test_format_percent_negative():
    """Test _format_percent with negative values."""
    from services.dashboard.components.strategy_performance import _format_percent

    result = _format_percent(-8.25)
    assert result == "-8.25%"


def test_format_percent_zero():
    """Test _format_percent with zero."""
    from services.dashboard.components.strategy_performance import _format_percent

    result = _format_percent(0.0)
    assert result == "0.00%"


def test_format_percent_decimals():
    """Test _format_percent with custom decimals."""
    from services.dashboard.components.strategy_performance import _format_percent

    result = _format_percent(15.5678, decimals=1)
    assert result == "+15.6%"


def test_format_plain_percent_positive():
    """Test _format_plain_percent with positive values (no sign)."""
    from services.dashboard.components.strategy_performance import _format_plain_percent

    result = _format_plain_percent(22.5)
    assert result == "22.50%"
    assert "+" not in result


def test_format_plain_percent_negative():
    """Test _format_plain_percent with negative values (keeps negative sign)."""
    from services.dashboard.components.strategy_performance import _format_plain_percent

    result = _format_plain_percent(-5.5)
    assert result == "-5.50%"


def test_format_plain_percent_zero():
    """Test _format_plain_percent with zero."""
    from services.dashboard.components.strategy_performance import _format_plain_percent

    result = _format_plain_percent(0.0)
    assert result == "0.00%"


def test_format_plain_percent_decimals():
    """Test _format_plain_percent with custom decimals."""
    from services.dashboard.components.strategy_performance import _format_plain_percent

    result = _format_plain_percent(15.567, decimals=1)
    assert result == "15.6%"


def test_format_krw_large():
    """Test _format_krw with large values."""
    from services.dashboard.components.strategy_performance import _format_krw

    result = _format_krw(1_500_000.0)
    assert "1,500,000" in result
    assert "KRW" in result


def test_format_krw_small():
    """Test _format_krw with small values."""
    from services.dashboard.components.strategy_performance import _format_krw

    result = _format_krw(500.0)
    assert "500" in result
    assert "KRW" in result


def test_format_ratio():
    """Test _format_ratio."""
    from services.dashboard.components.strategy_performance import _format_ratio

    assert _format_ratio(1.85) == "1.85"
    assert _format_ratio(-0.35) == "-0.35"
    assert _format_ratio(2.5, decimals=1) == "2.5"


def test_get_period_days():
    """Test _get_period_days returns correct values."""
    from services.dashboard.components.strategy_performance import (
        _get_period_days,
        TimePeriod,
    )

    assert _get_period_days(TimePeriod.DAY_1) == 1
    assert _get_period_days(TimePeriod.WEEK_1) == 7
    assert _get_period_days(TimePeriod.MONTH_1) == 30
    assert _get_period_days(TimePeriod.MONTH_3) == 90
    assert _get_period_days(TimePeriod.YEAR_1) == 365
    assert _get_period_days(TimePeriod.ALL) is None


def test_calculate_win_rate():
    """Test _calculate_win_rate."""
    from services.dashboard.components.strategy_performance import _calculate_win_rate

    assert _calculate_win_rate(60, 100) == 60.0
    assert _calculate_win_rate(75, 100) == 75.0
    assert _calculate_win_rate(0, 100) == 0.0


def test_calculate_win_rate_zero_total():
    """Test _calculate_win_rate with zero total."""
    from services.dashboard.components.strategy_performance import _calculate_win_rate

    assert _calculate_win_rate(10, 0) == 0.0
    assert _calculate_win_rate(0, 0) == 0.0


def test_calculate_profit_factor():
    """Test _calculate_profit_factor."""
    from services.dashboard.components.strategy_performance import _calculate_profit_factor

    # Gross profit: 1000 * 60 = 60000, Gross loss: 500 * 40 = 20000
    result = _calculate_profit_factor(1000.0, 500.0, 60, 40)
    assert result == 3.0

    # Equal profit and loss
    result = _calculate_profit_factor(100.0, 100.0, 50, 50)
    assert result == 1.0


def test_calculate_profit_factor_zero_loss():
    """Test _calculate_profit_factor with zero losses."""
    from services.dashboard.components.strategy_performance import _calculate_profit_factor

    result = _calculate_profit_factor(1000.0, 500.0, 60, 0)
    assert result == 0.0  # Division by zero returns default


def test_get_demo_strategies():
    """Test _get_demo_strategies returns expected data."""
    from services.dashboard.components.strategy_performance import _get_demo_strategies

    strategies = _get_demo_strategies()

    assert len(strategies) >= 2
    assert any(s.strategy_name == "BTC Momentum" for s in strategies)
    assert all(s.strategy_id != "" for s in strategies)


def test_get_demo_strategies_deterministic():
    """Test _get_demo_strategies returns consistent data."""
    from services.dashboard.components.strategy_performance import _get_demo_strategies

    strat1 = _get_demo_strategies()
    strat2 = _get_demo_strategies()

    assert len(strat1) == len(strat2)
    assert strat1[0].strategy_name == strat2[0].strategy_name
    assert strat1[0].metrics.total_return == strat2[0].metrics.total_return


def test_get_performance_summary():
    """Test _get_performance_summary calculates correctly."""
    from services.dashboard.components.strategy_performance import (
        _get_performance_summary,
        _get_demo_strategies,
    )

    strategies = _get_demo_strategies()
    summary = _get_performance_summary(strategies)

    assert "total_return" in summary
    assert "total_return_krw" in summary
    assert "total_trades" in summary
    assert "overall_win_rate" in summary
    assert "avg_sharpe" in summary
    assert "max_mdd" in summary
    assert "active_count" in summary


def test_get_performance_summary_empty():
    """Test _get_performance_summary with empty list."""
    from services.dashboard.components.strategy_performance import _get_performance_summary

    summary = _get_performance_summary([])

    assert summary["total_return"] == 0.0
    assert summary["total_trades"] == 0
    assert summary["active_count"] == 0


def test_get_return_indicator():
    """Test _get_return_indicator."""
    from services.dashboard.components.strategy_performance import _get_return_indicator

    assert _get_return_indicator(15.5) == "[+]"
    assert _get_return_indicator(-5.0) == "[-]"
    assert _get_return_indicator(0.0) == "[=]"


def test_get_quality_indicator():
    """Test _get_quality_indicator based on Sharpe ratio."""
    from services.dashboard.components.strategy_performance import _get_quality_indicator

    assert _get_quality_indicator(2.5) == "[EXCELLENT]"
    assert _get_quality_indicator(1.5) == "[GOOD]"
    assert _get_quality_indicator(0.7) == "[FAIR]"
    assert _get_quality_indicator(0.2) == "[POOR]"
    assert _get_quality_indicator(-0.5) == "[NEGATIVE]"


def test_get_performance_export_data():
    """Test get_performance_export_data returns valid structure."""
    from services.dashboard.components.strategy_performance import get_performance_export_data

    data = get_performance_export_data()

    assert "timestamp" in data
    assert "summary" in data
    assert "strategies" in data
    assert isinstance(data["strategies"], list)


def test_get_performance_export_data_strategy_structure():
    """Test get_performance_export_data strategy structure."""
    from services.dashboard.components.strategy_performance import get_performance_export_data

    data = get_performance_export_data()
    strategies = data["strategies"]

    assert len(strategies) > 0
    for s in strategies:
        assert "id" in s
        assert "name" in s
        assert "is_active" in s
        assert "metrics" in s
        assert "trade_stats" in s


def test_strategy_performance_with_equity_curve():
    """Test StrategyPerformance with equity curve data."""
    from services.dashboard.components.strategy_performance import StrategyPerformance

    equity = [1000000.0, 1050000.0, 1020000.0, 1100000.0]
    timestamps = [datetime.now() - timedelta(days=i) for i in range(4)]

    strategy = StrategyPerformance(
        strategy_id="test",
        strategy_name="Test",
        equity_curve=equity,
        timestamps=timestamps,
    )

    assert strategy.equity_curve == equity
    assert len(strategy.timestamps) == 4


def test_format_percent_with_nan():
    """Test _format_percent handles NaN gracefully."""
    from services.dashboard.components.strategy_performance import _format_percent

    result = _format_percent(float("nan"))
    assert result == "0.00%"


def test_format_krw_negative():
    """Test _format_krw with negative values."""
    from services.dashboard.components.strategy_performance import _format_krw

    result = _format_krw(-500_000.0)
    assert "-500,000" in result
    assert "KRW" in result


def test_safe_divide_inf_result():
    """Test _safe_divide when result would be inf."""
    from services.dashboard.components.strategy_performance import _safe_divide

    # Very large numerator with very small denominator
    result = _safe_divide(1e308, 1e-308, default=0.0)
    assert result == 0.0  # Should return default for inf result


def test_performance_metrics_all_negative():
    """Test PerformanceMetrics with all negative values."""
    from services.dashboard.components.strategy_performance import PerformanceMetrics

    metrics = PerformanceMetrics(
        total_return=-25.0,
        total_return_krw=-2_500_000.0,
        sharpe_ratio=-1.5,
        sortino_ratio=-2.0,
        max_drawdown=30.0,
        volatility=45.0,
        calmar_ratio=-0.83,
    )

    assert metrics.total_return == -25.0
    assert metrics.sharpe_ratio == -1.5


def test_trade_stats_boundary_values():
    """Test TradeStats with boundary values."""
    from services.dashboard.components.strategy_performance import TradeStats

    # All wins
    stats = TradeStats(
        total_trades=100,
        winning_trades=100,
        losing_trades=0,
        win_rate=100.0,
    )
    assert stats.win_rate == 100.0

    # All losses
    stats = TradeStats(
        total_trades=50,
        winning_trades=0,
        losing_trades=50,
        win_rate=0.0,
    )
    assert stats.win_rate == 0.0


def test_filter_strategies_by_period_all():
    """Test _filter_strategies_by_period with ALL period returns all."""
    from services.dashboard.components.strategy_performance import (
        _filter_strategies_by_period,
        _get_demo_strategies,
        TimePeriod,
    )

    strategies = _get_demo_strategies()
    result, used_fallback = _filter_strategies_by_period(strategies, TimePeriod.ALL)

    assert result == strategies
    assert used_fallback is False


def test_filter_strategies_by_period_with_reference_time():
    """Test _filter_strategies_by_period filters by sufficient history."""
    from services.dashboard.components.strategy_performance import (
        _filter_strategies_by_period,
        StrategyPerformance,
        TimePeriod,
    )

    ref_time = datetime(2026, 1, 15, 12, 0, 0)

    strategies = [
        StrategyPerformance(
            strategy_id="s1",
            strategy_name="Old Strategy",
            start_date=datetime(2025, 10, 1),  # 106 days before ref_time
        ),
        StrategyPerformance(
            strategy_id="s2",
            strategy_name="Recent Strategy",
            start_date=datetime(2026, 1, 10),  # Only 5 days before ref_time
        ),
        StrategyPerformance(
            strategy_id="s3",
            strategy_name="Future Strategy",
            start_date=datetime(2026, 2, 1),  # After ref_time
        ),
    ]

    # 1M filter: only strategies with 30+ days of history
    result_1m, fallback_1m = _filter_strategies_by_period(
        strategies, TimePeriod.MONTH_1, reference_time=ref_time
    )
    # Only "Old Strategy" has 30+ days of history
    assert len(result_1m) == 1
    assert result_1m[0].strategy_name == "Old Strategy"
    assert fallback_1m is False

    # 1W filter: only strategies with 7+ days of history
    result_1w, fallback_1w = _filter_strategies_by_period(
        strategies, TimePeriod.WEEK_1, reference_time=ref_time
    )
    # Only "Old Strategy" has 7+ days of history (Recent is only 5 days)
    assert len(result_1w) == 1
    assert result_1w[0].strategy_name == "Old Strategy"
    assert fallback_1w is False

    # 1D filter: only strategies with 1+ days of history
    result_1d, fallback_1d = _filter_strategies_by_period(
        strategies, TimePeriod.DAY_1, reference_time=ref_time
    )
    # Both Old and Recent have 1+ days (Recent has 5 days)
    assert len(result_1d) == 2
    assert any(s.strategy_name == "Old Strategy" for s in result_1d)
    assert any(s.strategy_name == "Recent Strategy" for s in result_1d)
    assert fallback_1d is False


def test_filter_strategies_by_period_no_start_date():
    """Test _filter_strategies_by_period includes strategies without start_date."""
    from services.dashboard.components.strategy_performance import (
        _filter_strategies_by_period,
        StrategyPerformance,
        TimePeriod,
    )

    strategies = [
        StrategyPerformance(
            strategy_id="s1",
            strategy_name="No Date",
            start_date=None,
        ),
        StrategyPerformance(
            strategy_id="s2",
            strategy_name="Has Date",
            start_date=datetime(2025, 12, 1),  # More than 1 week before ref_time
        ),
    ]

    ref_time = datetime(2026, 1, 15)
    result, used_fallback = _filter_strategies_by_period(
        strategies, TimePeriod.WEEK_1, reference_time=ref_time
    )

    # Both should be included (no date = full history, has date = 45 days > 7 days)
    assert len(result) == 2
    assert used_fallback is False


def test_filter_strategies_by_period_empty_fallback():
    """Test _filter_strategies_by_period returns original with fallback flag if filter empties."""
    from services.dashboard.components.strategy_performance import (
        _filter_strategies_by_period,
        StrategyPerformance,
        TimePeriod,
    )

    # All strategies start in the future (no sufficient history)
    future_date = datetime(2030, 1, 1)
    strategies = [
        StrategyPerformance(
            strategy_id="s1",
            strategy_name="Future",
            start_date=future_date,
        ),
    ]

    ref_time = datetime(2026, 1, 15)
    result, used_fallback = _filter_strategies_by_period(
        strategies, TimePeriod.MONTH_1, reference_time=ref_time
    )

    # Should fallback to original list and indicate fallback was used
    assert result == strategies
    assert used_fallback is True


def test_filter_strategies_by_period_allow_fallback_false():
    """Test _filter_strategies_by_period returns empty list when allow_fallback=False."""
    from services.dashboard.components.strategy_performance import (
        _filter_strategies_by_period,
        StrategyPerformance,
        TimePeriod,
    )

    # All strategies start in the future (no sufficient history)
    future_date = datetime(2030, 1, 1)
    strategies = [
        StrategyPerformance(
            strategy_id="s1",
            strategy_name="Future",
            start_date=future_date,
        ),
    ]

    ref_time = datetime(2026, 1, 15)
    result, used_fallback = _filter_strategies_by_period(
        strategies, TimePeriod.MONTH_1, reference_time=ref_time, allow_fallback=False
    )

    # Should return empty list with no fallback
    assert result == []
    assert used_fallback is False


def test_filter_strategies_by_period_allow_fallback_true_explicit():
    """Test _filter_strategies_by_period with explicit allow_fallback=True."""
    from services.dashboard.components.strategy_performance import (
        _filter_strategies_by_period,
        StrategyPerformance,
        TimePeriod,
    )

    # All strategies start in the future (no sufficient history)
    future_date = datetime(2030, 1, 1)
    strategies = [
        StrategyPerformance(
            strategy_id="s1",
            strategy_name="Future",
            start_date=future_date,
        ),
    ]

    ref_time = datetime(2026, 1, 15)
    result, used_fallback = _filter_strategies_by_period(
        strategies, TimePeriod.MONTH_1, reference_time=ref_time, allow_fallback=True
    )

    # Should fallback to original list when allow_fallback=True
    assert result == strategies
    assert used_fallback is True
