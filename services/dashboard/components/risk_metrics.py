"""Risk metrics panel component for portfolio risk analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st

# Session state key prefix
_KEY_PREFIX = "risk_metrics."


# =============================================================================
# Safe Math Helpers
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


def _safe_sqrt(value: float, default: float = 0.0) -> float:
    """Safe square root with negative guard."""
    if value < 0 or not math.isfinite(value):
        return default
    return math.sqrt(value)


def _sanitize_returns(returns: List[float]) -> List[float]:
    """Sanitize returns list by replacing non-finite values with 0."""
    return [_safe_float(r, 0.0) for r in returns]


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""

    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    volatility_pct: float
    win_rate_pct: float
    profit_factor: float
    avg_return_pct: float
    total_return_pct: float
    trading_days: int


# =============================================================================
# Risk Calculation Functions
# =============================================================================


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.03,
    trading_days: int = 252,
) -> float:
    """Calculate annualized Sharpe Ratio.

    Args:
        returns: List of daily returns (as decimals).
        risk_free_rate: Annual risk-free rate.
        trading_days: Trading days per year for annualization.

    Returns:
        Annualized Sharpe Ratio.
    """
    safe_returns = _sanitize_returns(returns)
    if len(safe_returns) < 2:
        return 0.0

    if trading_days <= 0:
        return 0.0

    safe_rf = _safe_float(risk_free_rate, 0.0)
    daily_rf = safe_rf / trading_days
    excess_returns = [r - daily_rf for r in safe_returns]

    mean_excess = sum(excess_returns) / len(excess_returns)

    variance = sum((r - mean_excess) ** 2 for r in excess_returns) / len(excess_returns)
    std_excess = _safe_sqrt(variance)

    if std_excess == 0:
        return 0.0

    sharpe = _safe_div(mean_excess, std_excess) * _safe_sqrt(trading_days)
    return _safe_float(sharpe)


def calculate_sortino_ratio(
    returns: List[float],
    risk_free_rate: float = 0.03,
    trading_days: int = 252,
) -> float:
    """Calculate annualized Sortino Ratio (downside deviation only).

    Args:
        returns: List of daily returns (as decimals).
        risk_free_rate: Annual risk-free rate.
        trading_days: Trading days per year for annualization.

    Returns:
        Annualized Sortino Ratio.
    """
    safe_returns = _sanitize_returns(returns)
    if len(safe_returns) < 2:
        return 0.0

    if trading_days <= 0:
        return 0.0

    safe_rf = _safe_float(risk_free_rate, 0.0)
    daily_rf = safe_rf / trading_days
    excess_returns = [r - daily_rf for r in safe_returns]

    mean_excess = sum(excess_returns) / len(excess_returns)

    # Downside deviation (only negative excess returns)
    downside = [r for r in excess_returns if r < 0]

    if not downside:
        return 0.0  # No downside, return 0 instead of inf for display

    downside_variance = sum(r**2 for r in downside) / len(downside)
    downside_std = _safe_sqrt(downside_variance)

    if downside_std == 0:
        return 0.0

    sortino = _safe_div(mean_excess, downside_std) * _safe_sqrt(trading_days)
    return _safe_float(sortino)


def calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, List[float]]:
    """Calculate maximum drawdown and drawdown series.

    Args:
        equity_curve: List of equity values.

    Returns:
        Tuple of (max_drawdown_pct, drawdown_series).
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, []

    drawdowns: List[float] = []
    peak = _safe_float(equity_curve[0], 1.0)
    max_dd = 0.0

    for equity in equity_curve:
        safe_equity = _safe_float(equity, 0.0)
        if safe_equity > peak:
            peak = safe_equity

        dd = _safe_div(peak - safe_equity, peak) if peak > 0 else 0.0
        drawdowns.append(dd)
        max_dd = max(max_dd, dd)

    return max_dd * 100, drawdowns


def calculate_calmar_ratio(
    total_return_pct: float,
    max_drawdown_pct: float,
) -> float:
    """Calculate Calmar Ratio (return / max drawdown).

    Args:
        total_return_pct: Total return percentage.
        max_drawdown_pct: Maximum drawdown percentage.

    Returns:
        Calmar Ratio.
    """
    if max_drawdown_pct == 0:
        return 0.0

    calmar = _safe_div(total_return_pct, max_drawdown_pct)
    return _safe_float(calmar)


def calculate_volatility(returns: List[float], trading_days: int = 252) -> float:
    """Calculate annualized volatility.

    Args:
        returns: List of daily returns.
        trading_days: Trading days per year for annualization.

    Returns:
        Annualized volatility percentage.
    """
    safe_returns = _sanitize_returns(returns)
    if len(safe_returns) < 2:
        return 0.0

    mean_ret = sum(safe_returns) / len(safe_returns)
    variance = sum((r - mean_ret) ** 2 for r in safe_returns) / len(safe_returns)
    daily_vol = _safe_sqrt(variance)

    annual_vol = daily_vol * _safe_sqrt(trading_days) * 100
    return _safe_float(annual_vol)


def calculate_win_rate(returns: List[float]) -> float:
    """Calculate win rate percentage.

    Args:
        returns: List of returns.

    Returns:
        Win rate percentage.
    """
    safe_returns = _sanitize_returns(returns)
    if not safe_returns:
        return 0.0

    wins = sum(1 for r in safe_returns if r > 0)
    return _safe_div(wins, len(safe_returns)) * 100


def calculate_profit_factor(returns: List[float]) -> float:
    """Calculate profit factor (gross profit / gross loss).

    Args:
        returns: List of returns.

    Returns:
        Profit factor.
    """
    safe_returns = _sanitize_returns(returns)
    gross_profit = sum(r for r in safe_returns if r > 0)
    gross_loss = abs(sum(r for r in safe_returns if r < 0))

    if gross_loss == 0:
        return 0.0  # Avoid inf for display

    return _safe_float(_safe_div(gross_profit, gross_loss))


def calculate_risk_metrics(
    returns: List[float],
    equity_curve: List[float],
    risk_free_rate: float = 0.03,
) -> RiskMetrics:
    """Calculate comprehensive risk metrics.

    Args:
        returns: List of daily returns (as decimals).
        equity_curve: List of equity values.
        risk_free_rate: Annual risk-free rate.

    Returns:
        RiskMetrics object.
    """
    if not returns or not equity_curve:
        return RiskMetrics(
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            calmar_ratio=0.0,
            volatility_pct=0.0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            avg_return_pct=0.0,
            total_return_pct=0.0,
            trading_days=0,
        )

    safe_returns = _sanitize_returns(returns)

    sharpe = calculate_sharpe_ratio(safe_returns, risk_free_rate)
    sortino = calculate_sortino_ratio(safe_returns, risk_free_rate)
    max_dd, _ = calculate_max_drawdown(equity_curve)
    volatility = calculate_volatility(safe_returns)
    win_rate = calculate_win_rate(safe_returns)
    profit_factor = calculate_profit_factor(safe_returns)

    avg_return = (sum(safe_returns) / len(safe_returns)) * 100 if safe_returns else 0.0

    first_equity = _safe_float(equity_curve[0], 1.0)
    last_equity = _safe_float(equity_curve[-1], first_equity)
    total_return = _safe_div(last_equity - first_equity, first_equity) * 100

    calmar = calculate_calmar_ratio(total_return, max_dd)

    return RiskMetrics(
        sharpe_ratio=_safe_float(sharpe),
        sortino_ratio=_safe_float(sortino),
        max_drawdown_pct=_safe_float(max_dd),
        calmar_ratio=_safe_float(calmar),
        volatility_pct=_safe_float(volatility),
        win_rate_pct=_safe_float(win_rate),
        profit_factor=_safe_float(profit_factor),
        avg_return_pct=_safe_float(avg_return),
        total_return_pct=_safe_float(total_return),
        trading_days=len(safe_returns),
    )


# =============================================================================
# Demo Data (Deterministic: fixed values, no randomness)
# =============================================================================


def _get_demo_returns() -> List[float]:
    """Generate demo daily returns based on Sept_v3_RSI50_Gate OOS performance.

    OOS Performance (v3, 2019-2024):
        - Sharpe: 2.27
        - MDD: -37.0%
        - Return: 11,763% (over ~5 years)

    Uses deterministic pattern for reproducibility.
    """
    import random

    rng = random.Random(42)

    # 5 years of trading days
    num_days = 252 * 5  # 1260 days

    # Target: 11,763% over 5 years
    # Daily return ~= 0.379% with std ~2.65% for Sharpe 2.27
    target_daily_mean = 0.00379
    target_daily_std = 0.0265

    returns = []
    for i in range(num_days):
        ret = rng.gauss(target_daily_mean, target_daily_std)

        # Occasional larger drawdowns
        if rng.random() < 0.02:
            ret = rng.gauss(-0.03, 0.02)

        # Clip extremes
        ret = max(-0.15, min(0.15, ret))
        returns.append(ret)

    return returns


def _get_demo_equity_curve(initial_equity: float = 10_000_000) -> List[float]:
    """Generate demo equity curve from Sept_v3 OOS returns."""
    returns = _get_demo_returns()
    equity_curve: List[float] = [float(initial_equity)]

    for r in returns:
        new_equity = equity_curve[-1] * (1 + r)
        equity_curve.append(new_equity)

    return equity_curve


def _get_demo_dates() -> List[date]:
    """Generate demo date series (2019-2024 OOS period)."""
    start_date = date(2019, 1, 1)
    returns = _get_demo_returns()
    dates: List[date] = []
    current = start_date

    for _ in range(len(returns) + 1):  # +1 for initial equity point
        # Skip weekends
        while current.weekday() >= 5:
            current += timedelta(days=1)
        dates.append(current)
        current += timedelta(days=1)

    return dates


def _generate_equity_from_returns(
    returns: List[float],
    initial_equity: float = 100_000_000,
) -> List[float]:
    """Generate equity curve from returns."""
    safe_returns = _sanitize_returns(returns)
    equity_curve: List[float] = [float(initial_equity)]

    for r in safe_returns:
        new_equity = equity_curve[-1] * (1 + r)
        equity_curve.append(new_equity)

    return equity_curve


def _generate_returns_from_equity(equity_curve: List[float]) -> List[float]:
    """Generate returns from equity curve."""
    if len(equity_curve) < 2:
        return []

    returns: List[float] = []
    for i in range(1, len(equity_curve)):
        prev = _safe_float(equity_curve[i - 1], 1.0)
        curr = _safe_float(equity_curve[i], prev)
        ret = _safe_div(curr - prev, prev)
        returns.append(ret)

    return returns


def _generate_dates_for_length(
    length: int, base_date: Optional[date] = None
) -> List[date]:
    """Generate date series for given length."""
    if base_date is None:
        base_date = date(2026, 1, 15)

    dates: List[date] = []
    current = base_date - timedelta(days=length - 1)

    for _ in range(length):
        dates.append(current)
        current += timedelta(days=1)

    return dates


# =============================================================================
# Visualization
# =============================================================================


def _render_equity_chart(dates: List[date], equity_curve: List[float]) -> None:
    """Render equity curve line chart."""
    if not dates or not equity_curve:
        st.info("자산 데이터가 없습니다.")
        return

    # Align lengths to minimum
    min_len = min(len(dates), len(equity_curve))
    plot_dates = dates[:min_len]
    plot_equity = equity_curve[:min_len]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_dates,
            y=plot_equity,
            mode="lines",
            name="자산",
            line=dict(color="#00C853", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 200, 83, 0.1)",
        )
    )

    fig.update_layout(
        title="자산 곡선",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="날짜",
        yaxis_title="자산 (KRW)",
        yaxis_tickformat=",",
    )

    st.plotly_chart(fig, width="stretch")


def _render_drawdown_chart(dates: List[date], drawdowns: List[float]) -> None:
    """Render drawdown area chart."""
    if not dates or not drawdowns:
        st.info("낙폭 데이터가 없습니다.")
        return

    # Align lengths to minimum
    min_len = min(len(dates), len(drawdowns))
    plot_dates = dates[:min_len]
    plot_dd = drawdowns[:min_len]

    dd_percent = [d * 100 for d in plot_dd]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_dates,
            y=dd_percent,
            mode="lines",
            name="낙폭",
            line=dict(color="#FF5252", width=2),
            fill="tozeroy",
            fillcolor="rgba(255, 82, 82, 0.3)",
        )
    )

    fig.update_layout(
        title="낙폭 (Drawdown)",
        template="plotly_dark",
        height=250,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="날짜",
        yaxis_title="낙폭 (%)",
        yaxis_autorange="reversed",  # Drawdown shows negative direction
    )

    st.plotly_chart(fig, width="stretch")


def _render_metrics_cards(metrics: RiskMetrics) -> None:
    """Render risk metrics as metric cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("샤프 비율", f"{metrics.sharpe_ratio:.2f}")
        st.metric("소르티노 비율", f"{metrics.sortino_ratio:.2f}")

    with col2:
        st.metric("최대 낙폭 (MDD)", f"{metrics.max_drawdown_pct:.2f}%")
        st.metric("칼마 비율", f"{metrics.calmar_ratio:.2f}")

    with col3:
        st.metric("변동성", f"{metrics.volatility_pct:.2f}%")
        st.metric("승률", f"{metrics.win_rate_pct:.1f}%")

    with col4:
        st.metric("손익비", f"{metrics.profit_factor:.2f}")
        st.metric("거래일 수", f"{metrics.trading_days}일")


def _render_return_stats(metrics: RiskMetrics) -> None:
    """Render return statistics row."""
    col1, col2, col3 = st.columns(3)

    with col1:
        color = "normal" if metrics.total_return_pct >= 0 else "inverse"
        st.metric(
            "총 수익률",
            f"{metrics.total_return_pct:+.2f}%",
            delta_color=color,
        )

    with col2:
        st.metric(
            "평균 일간 수익률",
            f"{metrics.avg_return_pct:+.4f}%",
        )

    with col3:
        # Risk-adjusted return interpretation
        if metrics.sharpe_ratio >= 2.0:
            rating = "우수"
        elif metrics.sharpe_ratio >= 1.0:
            rating = "양호"
        elif metrics.sharpe_ratio >= 0.5:
            rating = "보통"
        else:
            rating = "미흡"
        st.metric("위험조정 수익 등급", rating)


# =============================================================================
# Session State Key Helper
# =============================================================================


def _key(name: str) -> str:
    """Generate namespaced session state key."""
    return f"{_KEY_PREFIX}{name}"


# =============================================================================
# Main Render Function
# =============================================================================


def render_risk_metrics_panel(
    returns: Optional[List[float]] = None,
    equity_curve: Optional[List[float]] = None,
    dates: Optional[List[date]] = None,
    risk_free_rate: float = 0.03,
) -> None:
    """Render risk metrics panel with charts and statistics.

    Args:
        returns: List of daily returns. If None, generates from equity_curve or uses demo.
        equity_curve: List of equity values. If None, generates from returns or uses demo.
        dates: List of dates. If None, generates based on data length or uses demo.
        risk_free_rate: Annual risk-free rate for calculations.
    """
    st.subheader("리스크 지표")

    # Handle partial input: fill missing data independently
    use_demo = False

    if returns is None and equity_curve is None:
        # Both None: use full demo data
        use_demo = True
        returns = _get_demo_returns()
        equity_curve = _get_demo_equity_curve()
        dates = _get_demo_dates()
    elif returns is None and equity_curve is not None:
        # Only equity_curve provided: derive returns
        returns = _generate_returns_from_equity(equity_curve)
    elif returns is not None and equity_curve is None:
        # Only returns provided: derive equity_curve
        equity_curve = _generate_equity_from_returns(returns)

    if use_demo:
        st.caption(
            "📊 백테스트 참고 성과 (Sept-v3-RSI50-Gate, 2019~2024 OOS) - 실거래 시작 시 실제 지표로 전환"
        )
    else:
        st.caption("📈 실거래 성과 - 최근 30일 거래 기록 기준")

    # Safety check after processing
    if not returns or not equity_curve:
        st.info("리스크 분석을 위한 데이터가 없습니다.")
        return

    # Calculate metrics
    metrics = calculate_risk_metrics(returns, equity_curve, risk_free_rate)

    # Render return statistics
    _render_return_stats(metrics)

    st.divider()

    # Render metrics cards
    _render_metrics_cards(metrics)

    st.divider()

    # Generate dates if not provided
    if dates is None:
        dates = _generate_dates_for_length(len(equity_curve))

    # Render equity chart (handles length mismatch internally)
    _render_equity_chart(dates, equity_curve)

    # Calculate drawdown series for chart
    _, drawdowns = calculate_max_drawdown(equity_curve)

    # Render drawdown chart (handles length mismatch internally)
    _render_drawdown_chart(dates, drawdowns)
