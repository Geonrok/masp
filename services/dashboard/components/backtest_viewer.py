"""Backtest viewer component for equity curve and performance metrics."""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# Safe Math Helpers (GPT 필수보강 #3)
# =============================================================================


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float with finite check."""
    if value is None:
        return default
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Safe division with zero and non-finite guards."""
    if den == 0 or not math.isfinite(den):
        return default
    result = num / den
    return result if math.isfinite(result) else default


# =============================================================================
# Demo Data (GPT 필수보강 #2: 시드 고정)
# =============================================================================


def _get_demo_backtest_data() -> Dict[str, Any]:
    """Generate backtest data based on Ankle Buy v2.0 Bithumb OOS performance.

    OOS Performance:
        - Sharpe: 0.98
        - MDD: -22%
        - CAGR: 25.4%
        - Activation: 60%

    Uses fixed seed (42) for reproducibility.
    """
    rng = random.Random(42)

    # Backtest period: 5 years (2019-01-01 to 2024-01-01)
    start_date = date(2019, 1, 1)
    num_days = 252 * 5  # 5 years of trading days

    # Target metrics (Ankle Buy v2.0 Bithumb OOS):
    # - CAGR: 25.4% => daily_r = (1.254)^(1/252) - 1 ≈ 0.0009
    # - Sharpe: 0.98 => std = mean * sqrt(252) / 0.98
    # - MDD: -22%

    target_daily_mean = 0.0009  # ~25.4% CAGR
    target_daily_std = target_daily_mean * math.sqrt(252) / 0.98  # ~0.0146

    # Generate daily returns with occasional larger drawdowns
    daily_returns = []
    for i in range(num_days):
        # Base return
        ret = rng.gauss(target_daily_mean, target_daily_std)

        # Add occasional larger drawdowns to simulate MDD events
        if rng.random() < 0.02:  # 2% chance of bad day
            ret = rng.gauss(-0.03, 0.02)  # -3% avg on bad days

        # Clip extreme values
        ret = max(-0.15, min(0.15, ret))
        daily_returns.append(ret)

    # Generate dates (trading days only, skip weekends)
    dates = []
    current_date = start_date
    for _ in range(num_days):
        while current_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
            current_date += timedelta(days=1)
        dates.append(current_date)
        current_date += timedelta(days=1)

    return {
        "dates": dates,
        "daily_returns": daily_returns,
        "initial_capital": 10_000_000,  # 1000만원
        "strategy_name": "Ankle Buy v2.0",
    }


# =============================================================================
# Calculation Functions
# =============================================================================


def _calculate_equity_curve(
    daily_returns: List[float], initial_capital: float
) -> List[float]:
    """Calculate equity curve from daily returns using cumulative product."""
    if not daily_returns:
        return [initial_capital]

    equity = [initial_capital]
    for ret in daily_returns:
        safe_ret = _safe_float(ret, 0.0)
        new_value = equity[-1] * (1 + safe_ret)
        equity.append(_safe_float(new_value, equity[-1]))

    return equity[1:]  # Exclude initial capital (return values at end of each day)


def _calculate_drawdown(equity_curve: List[float]) -> List[float]:
    """Calculate drawdown series from equity curve.

    Drawdown = (current - peak) / peak
    Returns negative values (0 at new peaks).
    """
    if not equity_curve:
        return []

    drawdowns = []
    peak = equity_curve[0]

    for value in equity_curve:
        safe_value = _safe_float(value, peak)
        if safe_value > peak:
            peak = safe_value
        dd = _safe_div(safe_value - peak, peak, 0.0)
        drawdowns.append(dd)

    return drawdowns


def _calculate_metrics(
    daily_returns: List[float], equity_curve: List[float], initial_capital: float
) -> Dict[str, float]:
    """Calculate backtest performance metrics.

    Annualization unified to 252 trading days (GPT 필수보강 #1).
    """
    TRADING_DAYS = 252

    # Handle empty/insufficient data (GPT 필수보강 #4)
    if not daily_returns or not equity_curve:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "mdd": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
        }

    n_days = len(daily_returns)
    final_value = _safe_float(equity_curve[-1], initial_capital)
    initial = _safe_float(initial_capital, 1.0)

    # Total Return: (final/initial - 1) * 100
    total_return = _safe_div(final_value, initial, 1.0) - 1.0
    total_return_pct = total_return * 100

    # CAGR: ((final/initial) ** (252/days) - 1) * 100
    if n_days > 0 and initial > 0:
        ratio = _safe_div(final_value, initial, 1.0)
        if ratio > 0:
            exponent = _safe_div(TRADING_DAYS, n_days, 1.0)
            cagr = (ratio**exponent - 1) * 100
            cagr = _safe_float(cagr, 0.0)
        else:
            cagr = 0.0
    else:
        cagr = 0.0

    # MDD: min(drawdown) * 100
    drawdowns = _calculate_drawdown(equity_curve)
    mdd = min(drawdowns) * 100 if drawdowns else 0.0
    mdd = _safe_float(mdd, 0.0)

    # Sharpe: (mean/std) * sqrt(252), std=0 → 0
    safe_returns = [_safe_float(r, 0.0) for r in daily_returns]
    if len(safe_returns) > 1:
        mean_ret = sum(safe_returns) / len(safe_returns)
        variance = sum((r - mean_ret) ** 2 for r in safe_returns) / len(safe_returns)
        std_ret = math.sqrt(variance) if variance > 0 else 0.0

        # Use tolerance to handle floating point precision issues
        if std_ret > 1e-10:
            sharpe = _safe_div(mean_ret, std_ret, 0.0) * math.sqrt(TRADING_DAYS)
            sharpe = _safe_float(sharpe, 0.0)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Win Rate: count(r>0)/N * 100, N=0 → 0
    if safe_returns:
        wins = sum(1 for r in safe_returns if r > 0)
        win_rate = _safe_div(wins, len(safe_returns), 0.0) * 100
    else:
        win_rate = 0.0

    return {
        "total_return": _safe_float(total_return_pct, 0.0),
        "cagr": _safe_float(cagr, 0.0),
        "mdd": _safe_float(mdd, 0.0),
        "sharpe": _safe_float(sharpe, 0.0),
        "win_rate": _safe_float(win_rate, 0.0),
    }


# =============================================================================
# Visualization
# =============================================================================


def _render_equity_chart(dates: List[date], equity_curve: List[float]) -> None:
    """Render equity curve line chart."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity_curve,
            mode="lines",
            name="자산",
            line=dict(color="#00C853", width=2),
        )
    )

    fig.update_layout(
        title="자산 곡선",
        xaxis_title="날짜",
        yaxis_title="포트폴리오 가치 (KRW)",
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, width="stretch")


def _render_drawdown_chart(dates: List[date], drawdowns: List[float]) -> None:
    """Render drawdown area chart (red, fill below 0)."""
    # Convert to percentage
    dd_percent = [d * 100 for d in drawdowns]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=dd_percent,
            mode="lines",
            name="낙폭",
            line=dict(color="#FF5252", width=1),
            fill="tozeroy",
            fillcolor="rgba(255, 82, 82, 0.3)",
        )
    )

    fig.update_layout(
        title="낙폭 (Drawdown)",
        xaxis_title="날짜",
        yaxis_title="낙폭 (%)",
        template="plotly_dark",
        height=250,
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(range=[min(dd_percent) * 1.1 if dd_percent else -10, 5]),
    )

    st.plotly_chart(fig, width="stretch")


# =============================================================================
# Main Render Function
# =============================================================================


def render_backtest_viewer(backtest_data: Dict[str, Any] | None = None) -> None:
    """Render backtest performance viewer with metrics and charts.

    Args:
        backtest_data: Dict with keys: dates, daily_returns, initial_capital, strategy_name
                      If None, uses demo data.
    """
    st.subheader("백테스트 성과")

    # Use demo data if not provided
    if backtest_data is None:
        st.caption("OOS 성과 기반 시뮬레이션 (2019-2024)")
        backtest_data = _get_demo_backtest_data()

    # Extract data
    dates = backtest_data.get("dates", [])
    daily_returns = backtest_data.get("daily_returns", [])
    initial_capital = _safe_float(backtest_data.get("initial_capital", 10_000_000))
    strategy_name = backtest_data.get("strategy_name", "Strategy")

    # Handle empty data (GPT 필수보강 #4)
    if not daily_returns or len(daily_returns) < 1:
        st.info("백테스트 데이터가 없습니다. 백테스트를 실행하세요.")
        return

    # Calculate equity curve and metrics
    equity_curve = _calculate_equity_curve(daily_returns, initial_capital)

    # Ensure dates and equity_curve have same length
    if len(dates) != len(equity_curve):
        # Adjust dates to match equity curve length
        if len(dates) > len(equity_curve):
            dates = dates[: len(equity_curve)]
        else:
            # Generate missing dates
            last_date = dates[-1] if dates else date(2025, 1, 1)
            while len(dates) < len(equity_curve):
                last_date = last_date + timedelta(days=1)
                dates.append(last_date)

    metrics = _calculate_metrics(daily_returns, equity_curve, initial_capital)
    drawdowns = _calculate_drawdown(equity_curve)

    # Display strategy name
    st.caption(f"전략: {strategy_name}")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "총 수익률",
            f"{metrics['total_return']:+.2f}%",
            delta=f"CAGR: {metrics['cagr']:+.2f}%",
        )

    with col2:
        st.metric(
            "최대 낙폭 (MDD)",
            f"{metrics['mdd']:.2f}%",
        )

    with col3:
        st.metric(
            "샤프 비율",
            f"{metrics['sharpe']:.2f}",
        )

    with col4:
        st.metric(
            "승률",
            f"{metrics['win_rate']:.1f}%",
        )

    # Charts
    _render_equity_chart(dates, equity_curve)
    _render_drawdown_chart(dates, drawdowns)
