"""Strategy performance component for displaying trading strategy metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

# Session state key prefix
_KEY_PREFIX = "strategy_perf."

# Demo mode reference date (fixed for deterministic behavior)
_DEMO_REFERENCE_DATE = datetime(2026, 1, 1, 12, 0, 0)


class TimePeriod(str, Enum):
    """Time period for performance filtering."""

    DAY_1 = "1D"
    WEEK_1 = "1W"
    MONTH_1 = "1M"
    MONTH_3 = "3M"
    YEAR_1 = "1Y"
    ALL = "ALL"


@dataclass
class TradeStats:
    """Trade statistics for a strategy."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_holding_time_hours: float = 0.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for a strategy."""

    total_return: float = 0.0  # Total return percentage
    total_return_krw: float = 0.0  # Total return in KRW
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0  # MDD percentage
    max_drawdown_krw: float = 0.0  # MDD in KRW
    volatility: float = 0.0  # Annualized volatility
    calmar_ratio: float = 0.0


@dataclass
class StrategyPerformance:
    """Complete performance data for a strategy."""

    strategy_id: str
    strategy_name: str
    is_active: bool = True
    start_date: Optional[datetime] = None
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    trade_stats: TradeStats = field(default_factory=TradeStats)
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)


def _key(name: str) -> str:
    """Generate namespaced session state key."""
    return f"{_KEY_PREFIX}{name}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero or invalid."""
    num = _safe_float(numerator, 0.0)
    den = _safe_float(denominator, 0.0)
    if den == 0.0:
        return default
    result = num / den
    if not math.isfinite(result):
        return default
    return result


def _format_percent(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    safe_val = _safe_float(value, 0.0)
    sign = "+" if safe_val > 0 else ""
    return f"{sign}{safe_val:.{decimals}f}%"


def _format_krw(value: float) -> str:
    """Format value as KRW currency."""
    safe_val = _safe_float(value, 0.0)
    if abs(safe_val) >= 1_000_000:
        return f"{safe_val:,.0f} KRW"
    elif abs(safe_val) >= 1000:
        return f"{safe_val:,.0f} KRW"
    else:
        return f"{safe_val:,.2f} KRW"


def _format_ratio(value: float, decimals: int = 2) -> str:
    """Format ratio value."""
    safe_val = _safe_float(value, 0.0)
    return f"{safe_val:.{decimals}f}"


def _get_period_days(period: TimePeriod) -> Optional[int]:
    """Get number of days for a time period."""
    mapping = {
        TimePeriod.DAY_1: 1,
        TimePeriod.WEEK_1: 7,
        TimePeriod.MONTH_1: 30,
        TimePeriod.MONTH_3: 90,
        TimePeriod.YEAR_1: 365,
        TimePeriod.ALL: None,
    }
    return mapping.get(period)


def _calculate_win_rate(winning: int, total: int) -> float:
    """Calculate win rate percentage."""
    if total <= 0:
        return 0.0
    return (winning / total) * 100


def _calculate_profit_factor(avg_win: float, avg_loss: float, win_count: int, loss_count: int) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    gross_profit = _safe_float(avg_win, 0.0) * max(win_count, 0)
    gross_loss = abs(_safe_float(avg_loss, 0.0)) * max(loss_count, 0)
    return _safe_divide(gross_profit, gross_loss, 0.0)


def _get_demo_strategies() -> List[StrategyPerformance]:
    """Get demo strategy performance data (deterministic).

    Uses fixed reference date for consistent demo data.
    """
    base_date = _DEMO_REFERENCE_DATE - timedelta(days=90)

    return [
        StrategyPerformance(
            strategy_id="strat_001",
            strategy_name="BTC Momentum",
            is_active=True,
            start_date=base_date,
            metrics=PerformanceMetrics(
                total_return=15.8,
                total_return_krw=1_580_000.0,
                sharpe_ratio=1.85,
                sortino_ratio=2.12,
                max_drawdown=8.5,
                max_drawdown_krw=850_000.0,
                volatility=22.5,
                calmar_ratio=1.86,
            ),
            trade_stats=TradeStats(
                total_trades=45,
                winning_trades=28,
                losing_trades=17,
                win_rate=62.2,
                avg_win=125_000.0,
                avg_loss=75_000.0,
                profit_factor=1.96,
                avg_holding_time_hours=18.5,
            ),
        ),
        StrategyPerformance(
            strategy_id="strat_002",
            strategy_name="ETH Mean Reversion",
            is_active=True,
            start_date=base_date + timedelta(days=15),
            metrics=PerformanceMetrics(
                total_return=8.2,
                total_return_krw=820_000.0,
                sharpe_ratio=1.25,
                sortino_ratio=1.58,
                max_drawdown=12.3,
                max_drawdown_krw=1_230_000.0,
                volatility=28.5,
                calmar_ratio=0.67,
            ),
            trade_stats=TradeStats(
                total_trades=62,
                winning_trades=35,
                losing_trades=27,
                win_rate=56.5,
                avg_win=85_000.0,
                avg_loss=62_000.0,
                profit_factor=1.42,
                avg_holding_time_hours=6.2,
            ),
        ),
        StrategyPerformance(
            strategy_id="strat_003",
            strategy_name="Multi-Asset Balanced",
            is_active=False,
            start_date=base_date - timedelta(days=30),
            metrics=PerformanceMetrics(
                total_return=-2.5,
                total_return_krw=-250_000.0,
                sharpe_ratio=-0.35,
                sortino_ratio=-0.42,
                max_drawdown=15.8,
                max_drawdown_krw=1_580_000.0,
                volatility=18.2,
                calmar_ratio=-0.16,
            ),
            trade_stats=TradeStats(
                total_trades=28,
                winning_trades=11,
                losing_trades=17,
                win_rate=39.3,
                avg_win=95_000.0,
                avg_loss=85_000.0,
                profit_factor=0.72,
                avg_holding_time_hours=24.5,
            ),
        ),
    ]


def _filter_strategies_by_period(
    strategies: List[StrategyPerformance],
    period: TimePeriod,
    reference_time: Optional[datetime] = None,
) -> tuple[List[StrategyPerformance], bool]:
    """Filter strategies that have sufficient history for the selected time period.

    Only includes strategies that have been running for at least the selected period,
    so their metrics are meaningful for that timeframe.

    Args:
        strategies: List of strategy performances
        period: Time period filter
        reference_time: Reference time for filtering (defaults to now, injectable for tests)

    Returns:
        Tuple of (filtered strategies, used_fallback flag)
    """
    if period == TimePeriod.ALL:
        return strategies, False

    period_days = _get_period_days(period)
    if period_days is None:
        return strategies, False

    now = reference_time if reference_time is not None else datetime.now()
    period_start = now - timedelta(days=period_days)

    filtered = []
    for strategy in strategies:
        # Include if:
        # - No start_date (assume has full history)
        # - Started on or before period_start (has at least period_days of data)
        if strategy.start_date is None:
            filtered.append(strategy)
        elif strategy.start_date <= period_start:
            filtered.append(strategy)

    if filtered:
        return filtered, False
    else:
        # Fallback to all strategies when filter would return empty
        return strategies, True


def _get_performance_summary(strategies: List[StrategyPerformance]) -> Dict[str, Any]:
    """Calculate aggregate performance summary across all strategies."""
    if not strategies:
        return {
            "total_return": 0.0,
            "total_return_krw": 0.0,
            "total_trades": 0,
            "overall_win_rate": 0.0,
            "avg_sharpe": 0.0,
            "max_mdd": 0.0,
            "active_count": 0,
        }

    total_return = sum(_safe_float(s.metrics.total_return, 0.0) for s in strategies)
    total_return_krw = sum(_safe_float(s.metrics.total_return_krw, 0.0) for s in strategies)
    total_trades = sum(s.trade_stats.total_trades for s in strategies)
    total_wins = sum(s.trade_stats.winning_trades for s in strategies)

    sharpe_values = [_safe_float(s.metrics.sharpe_ratio, 0.0) for s in strategies]
    avg_sharpe = sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0

    mdd_values = [_safe_float(s.metrics.max_drawdown, 0.0) for s in strategies]
    max_mdd = max(mdd_values) if mdd_values else 0.0

    active_count = sum(1 for s in strategies if s.is_active)

    return {
        "total_return": total_return,
        "total_return_krw": total_return_krw,
        "total_trades": total_trades,
        "overall_win_rate": _calculate_win_rate(total_wins, total_trades),
        "avg_sharpe": avg_sharpe,
        "max_mdd": max_mdd,
        "active_count": active_count,
    }


def _get_return_indicator(value: float) -> str:
    """Get text indicator based on return value."""
    if value > 0:
        return "[+]"
    elif value < 0:
        return "[-]"
    return "[=]"


def _get_quality_indicator(sharpe: float) -> str:
    """Get quality indicator based on Sharpe ratio."""
    if sharpe >= 2.0:
        return "[EXCELLENT]"
    elif sharpe >= 1.0:
        return "[GOOD]"
    elif sharpe >= 0.5:
        return "[FAIR]"
    elif sharpe >= 0:
        return "[POOR]"
    return "[NEGATIVE]"


def render_strategy_performance(
    performance_provider: Optional[Callable[[], List[StrategyPerformance]]] = None,
    show_summary: bool = True,
    show_details: bool = True,
    show_trade_stats: bool = True,
    compact: bool = False,
) -> None:
    """Render strategy performance panel.

    Args:
        performance_provider: Function to get strategy performance data
        show_summary: Whether to show aggregate summary
        show_details: Whether to show individual strategy details
        show_trade_stats: Whether to show trade statistics
        compact: Whether to use compact layout
    """
    st.subheader("Strategy Performance")

    # Get data
    is_demo = performance_provider is None
    if is_demo:
        st.caption("Demo Mode")

    try:
        strategies = (
            performance_provider() if performance_provider is not None else _get_demo_strategies()
        )
    except Exception:
        strategies = _get_demo_strategies()
        st.warning("Failed to load performance data, showing demo data")

    if not strategies:
        st.info("No strategy performance data available.")
        return

    # Min history filter (filters strategies by minimum running period)
    col_filter, col_spacer = st.columns([2, 4])
    with col_filter:
        selected_period = st.selectbox(
            "Min History",
            options=[p.value for p in TimePeriod],
            index=2,  # Default to 1M
            key=_key("period"),
            help="Filter strategies by minimum running period. Metrics shown are all-time values.",
        )

    # Convert selected period string to enum (with fallback for corrupted session state)
    try:
        period_enum = TimePeriod(selected_period)
    except (ValueError, KeyError):
        period_enum = TimePeriod.MONTH_1  # Safe fallback

    # Use demo reference time for demo mode to ensure deterministic filtering
    reference_time = _DEMO_REFERENCE_DATE if is_demo else None
    filtered_strategies, used_fallback = _filter_strategies_by_period(
        strategies, period_enum, reference_time=reference_time
    )

    # Show filter status
    if used_fallback:
        st.warning(
            f"No strategies with {selected_period}+ history; showing all {len(strategies)} strategies"
        )
    elif len(filtered_strategies) < len(strategies):
        st.caption(
            f"Showing {len(filtered_strategies)} strategies with {selected_period}+ history "
            f"(filtered from {len(strategies)} total)"
        )

    # Note about metrics scope
    if period_enum != TimePeriod.ALL:
        st.caption("Note: Metrics shown are all-time values, not period-specific.")

    # Summary section
    if show_summary:
        summary = _get_performance_summary(filtered_strategies)

        st.markdown("**Portfolio Summary**")
        sum_cols = st.columns(4)

        with sum_cols[0]:
            ret_ind = _get_return_indicator(summary["total_return"])
            st.metric(
                f"Sum of Returns {ret_ind}",
                _format_percent(summary["total_return"]),
                help="Sum of individual strategy returns (not portfolio return)",
            )
            st.caption(f"Sum PnL: {_format_krw(summary['total_return_krw'])}")

        with sum_cols[1]:
            st.metric("Total Trades", f"{summary['total_trades']:,}")

        with sum_cols[2]:
            st.metric("Win Rate", _format_percent(summary["overall_win_rate"]))

        with sum_cols[3]:
            quality_ind = _get_quality_indicator(summary["avg_sharpe"])
            st.metric(f"Avg Sharpe {quality_ind}", _format_ratio(summary["avg_sharpe"]))

        st.caption(
            f"Active Strategies: {summary['active_count']} / {len(filtered_strategies)} | "
            f"Max MDD: {_format_percent(summary['max_mdd'])}"
        )

    # Individual strategies
    if show_details:
        st.divider()
        st.markdown("**Strategy Details**")

        for strategy in filtered_strategies:
            with st.expander(
                f"{'[ACTIVE]' if strategy.is_active else '[PAUSED]'} {strategy.strategy_name}",
                expanded=strategy.is_active and not compact,
            ):
                _render_strategy_card(strategy, show_trade_stats, compact)


def _render_strategy_card(
    strategy: StrategyPerformance,
    show_trade_stats: bool = True,
    compact: bool = False,
) -> None:
    """Render individual strategy performance card."""
    metrics = strategy.metrics
    stats = strategy.trade_stats

    # Performance metrics row
    if compact:
        cols = st.columns(4)
        with cols[0]:
            st.metric("Return", _format_percent(metrics.total_return))
        with cols[1]:
            st.metric("Sharpe", _format_ratio(metrics.sharpe_ratio))
        with cols[2]:
            st.metric("MDD", _format_percent(metrics.max_drawdown))
        with cols[3]:
            st.metric("Win Rate", _format_percent(stats.win_rate))
    else:
        # Full layout
        st.markdown("**Performance Metrics**")
        perf_cols = st.columns(4)

        with perf_cols[0]:
            ret_ind = _get_return_indicator(metrics.total_return)
            st.metric(
                f"Total Return {ret_ind}",
                _format_percent(metrics.total_return),
                delta=_format_krw(metrics.total_return_krw),
            )

        with perf_cols[1]:
            st.metric("Sharpe Ratio", _format_ratio(metrics.sharpe_ratio))
            st.caption(f"Sortino: {_format_ratio(metrics.sortino_ratio)}")

        with perf_cols[2]:
            st.metric("Max Drawdown", _format_percent(metrics.max_drawdown))
            st.caption(_format_krw(metrics.max_drawdown_krw))

        with perf_cols[3]:
            st.metric("Volatility", _format_percent(metrics.volatility))
            st.caption(f"Calmar: {_format_ratio(metrics.calmar_ratio)}")

        # Trade statistics
        if show_trade_stats:
            st.markdown("**Trade Statistics**")
            trade_cols = st.columns(4)

            with trade_cols[0]:
                st.metric("Total Trades", f"{stats.total_trades:,}")
                st.caption(f"W: {stats.winning_trades} / L: {stats.losing_trades}")

            with trade_cols[1]:
                st.metric("Win Rate", _format_percent(stats.win_rate))

            with trade_cols[2]:
                st.metric("Profit Factor", _format_ratio(stats.profit_factor))
                st.caption(
                    f"Avg W: {_format_krw(stats.avg_win)} / Avg L: {_format_krw(stats.avg_loss)}"
                )

            with trade_cols[3]:
                st.metric("Avg Hold Time", f"{stats.avg_holding_time_hours:.1f}h")

    # Start date info
    if strategy.start_date:
        st.caption(f"Started: {strategy.start_date.strftime('%Y-%m-%d')}")


def get_performance_export_data(
    strategies: Optional[List[StrategyPerformance]] = None,
) -> Dict[str, Any]:
    """Get performance data for export/API.

    Args:
        strategies: List of strategy performances (uses demo if None)

    Returns:
        Dict containing exportable performance data
    """
    if strategies is None:
        strategies = _get_demo_strategies()

    summary = _get_performance_summary(strategies)

    return {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "strategies": [
            {
                "id": s.strategy_id,
                "name": s.strategy_name,
                "is_active": s.is_active,
                "start_date": s.start_date.isoformat() if s.start_date else None,
                "metrics": {
                    "total_return": s.metrics.total_return,
                    "total_return_krw": s.metrics.total_return_krw,
                    "sharpe_ratio": s.metrics.sharpe_ratio,
                    "sortino_ratio": s.metrics.sortino_ratio,
                    "max_drawdown": s.metrics.max_drawdown,
                    "volatility": s.metrics.volatility,
                    "calmar_ratio": s.metrics.calmar_ratio,
                },
                "trade_stats": {
                    "total_trades": s.trade_stats.total_trades,
                    "winning_trades": s.trade_stats.winning_trades,
                    "losing_trades": s.trade_stats.losing_trades,
                    "win_rate": s.trade_stats.win_rate,
                    "profit_factor": s.trade_stats.profit_factor,
                    "avg_holding_time_hours": s.trade_stats.avg_holding_time_hours,
                },
            }
            for s in strategies
        ],
    }
