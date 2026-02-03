"""
Advanced Performance Metrics

Provides additional risk-adjusted performance metrics including:
- Omega Ratio
- Value at Risk (VaR)
- Conditional VaR (CVaR / Expected Shortfall)
- Information Ratio
- Treynor Ratio
- Tail Ratio
- Gain-to-Pain Ratio
- Underwater Analysis
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum sample sizes for statistical validity
MIN_SAMPLE_VAR = 30
MIN_SAMPLE_OMEGA = 20


@dataclass
class RiskMetrics:
    """Advanced risk metrics."""

    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    cvar_95: float  # Conditional VaR (Expected Shortfall) at 95%
    cvar_99: float  # Conditional VaR (Expected Shortfall) at 99%
    omega_ratio: float  # Omega ratio
    tail_ratio: float  # Ratio of right tail to left tail
    gain_to_pain: float  # Gain-to-Pain ratio


@dataclass
class DrawdownMetrics:
    """Detailed drawdown analysis."""

    max_drawdown: float  # Maximum drawdown (decimal)
    max_drawdown_pct: float  # Maximum drawdown (percentage)
    max_drawdown_duration: int  # Duration in periods
    avg_drawdown: float  # Average drawdown
    avg_drawdown_duration: float  # Average drawdown duration
    num_drawdowns: int  # Number of drawdowns
    current_drawdown: float  # Current drawdown
    recovery_time: Optional[int]  # Time to recover from max drawdown
    underwater_curve: List[float]  # Full underwater curve


@dataclass
class AdvancedPerformanceMetrics:
    """Comprehensive advanced performance metrics."""

    # Basic ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Advanced ratios
    omega_ratio: float
    information_ratio: float
    treynor_ratio: float
    gain_to_pain_ratio: float
    tail_ratio: float

    # Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    # Drawdown metrics
    max_drawdown_pct: float
    max_drawdown_duration: int
    avg_drawdown_pct: float

    # Trade statistics
    win_rate: float
    profit_factor: float
    expectancy: float
    payoff_ratio: float

    # Volatility metrics
    volatility_annual: float
    downside_deviation: float
    upside_potential: float

    # Skewness and kurtosis
    skewness: float
    kurtosis: float


def calculate_omega_ratio(
    returns: List[float],
    threshold: float = 0.0,
) -> float:
    """
    Calculate Omega Ratio.

    Omega = Probability-weighted gains above threshold /
            Probability-weighted losses below threshold

    Higher values indicate better risk-adjusted returns.

    Args:
        returns: List of period returns
        threshold: Return threshold (default 0)

    Returns:
        Omega Ratio (>1 is good)
    """
    if len(returns) < MIN_SAMPLE_OMEGA:
        logger.warning(
            f"[OMEGA] Insufficient samples: {len(returns)} < {MIN_SAMPLE_OMEGA}"
        )

    if len(returns) < 2:
        return 0.0

    returns_arr = np.array(returns)

    gains = np.sum(np.maximum(returns_arr - threshold, 0))
    losses = np.sum(np.maximum(threshold - returns_arr, 0))

    if losses == 0:
        return float("inf") if gains > 0 else 1.0

    return gains / losses


def calculate_var(
    returns: List[float],
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate Value at Risk (VaR).

    VaR represents the maximum expected loss at a given confidence level.

    Args:
        returns: List of period returns
        confidence: Confidence level (e.g., 0.95 for 95%)
        method: "historical" (percentile) or "parametric" (normal distribution)

    Returns:
        VaR as positive number (loss amount)
    """
    if len(returns) < MIN_SAMPLE_VAR:
        logger.warning(f"[VAR] Insufficient samples: {len(returns)} < {MIN_SAMPLE_VAR}")

    if len(returns) < 2:
        return 0.0

    returns_arr = np.array(returns)

    if method == "historical":
        # Historical simulation (empirical percentile)
        var = -np.percentile(returns_arr, (1 - confidence) * 100)

    elif method == "parametric":
        # Parametric (assumes normal distribution)
        mean = np.mean(returns_arr)
        std = np.std(returns_arr)
        z_score = {0.95: 1.645, 0.99: 2.326}.get(confidence, 1.645)
        var = -(mean - z_score * std)

    else:
        var = 0.0

    return max(0, var)


def calculate_cvar(
    returns: List[float],
    confidence: float = 0.95,
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR is the expected loss given that loss exceeds VaR.
    It provides information about the tail of the loss distribution.

    Args:
        returns: List of period returns
        confidence: Confidence level

    Returns:
        CVaR as positive number
    """
    if len(returns) < MIN_SAMPLE_VAR:
        logger.warning(
            f"[CVAR] Insufficient samples: {len(returns)} < {MIN_SAMPLE_VAR}"
        )

    if len(returns) < 2:
        return 0.0

    returns_arr = np.array(returns)
    var = calculate_var(returns, confidence, "historical")

    # CVaR is the mean of returns below the VaR threshold
    tail_returns = returns_arr[returns_arr <= -var]

    if len(tail_returns) == 0:
        return var

    return -np.mean(tail_returns)


def calculate_information_ratio(
    returns: List[float],
    benchmark_returns: List[float],
) -> float:
    """
    Calculate Information Ratio.

    IR = (Portfolio Return - Benchmark Return) / Tracking Error

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Information Ratio (annualized)
    """
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0

    returns_arr = np.array(returns)
    benchmark_arr = np.array(benchmark_returns)

    active_returns = returns_arr - benchmark_arr
    tracking_error = np.std(active_returns)

    if tracking_error == 0:
        return 0.0

    avg_active = np.mean(active_returns)
    ir = (avg_active / tracking_error) * np.sqrt(252)

    return ir


def calculate_treynor_ratio(
    returns: List[float],
    benchmark_returns: List[float],
    risk_free_rate: float = 0.03,
) -> float:
    """
    Calculate Treynor Ratio.

    Treynor = (Portfolio Return - Risk-Free Rate) / Beta

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Treynor Ratio
    """
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0

    returns_arr = np.array(returns)
    benchmark_arr = np.array(benchmark_returns)

    # Calculate beta
    covariance = np.cov(returns_arr, benchmark_arr)[0, 1]
    benchmark_variance = np.var(benchmark_arr)

    if benchmark_variance == 0:
        return 0.0

    beta = covariance / benchmark_variance

    if beta == 0:
        return 0.0

    # Calculate Treynor
    daily_rf = risk_free_rate / 252
    excess_return = np.mean(returns_arr) - daily_rf
    treynor = (excess_return * 252) / beta

    return treynor


def calculate_tail_ratio(
    returns: List[float],
    percentile: float = 5.0,
) -> float:
    """
    Calculate Tail Ratio.

    Ratio of the right tail (gains) to the left tail (losses)
    at a given percentile.

    Values > 1 indicate positive skew in extreme returns.

    Args:
        returns: List of returns
        percentile: Percentile for tails (default 5%)

    Returns:
        Tail Ratio
    """
    if len(returns) < 10:
        return 1.0

    returns_arr = np.array(returns)

    right_tail = np.percentile(returns_arr, 100 - percentile)
    left_tail = np.percentile(returns_arr, percentile)

    if left_tail == 0:
        return float("inf") if right_tail > 0 else 1.0

    return abs(right_tail / left_tail)


def calculate_gain_to_pain(returns: List[float]) -> float:
    """
    Calculate Gain-to-Pain Ratio.

    Sum of all returns / Absolute sum of negative returns.

    Higher values indicate better risk-adjusted performance.

    Args:
        returns: List of returns

    Returns:
        Gain-to-Pain Ratio
    """
    if not returns:
        return 0.0

    returns_arr = np.array(returns)

    total_return = np.sum(returns_arr)
    pain = np.sum(np.abs(returns_arr[returns_arr < 0]))

    if pain == 0:
        return float("inf") if total_return > 0 else 0.0

    return total_return / pain


def calculate_skewness(returns: List[float]) -> float:
    """
    Calculate return distribution skewness.

    Positive skew: More extreme positive returns
    Negative skew: More extreme negative returns

    Args:
        returns: List of returns

    Returns:
        Skewness coefficient
    """
    if len(returns) < 3:
        return 0.0

    returns_arr = np.array(returns)
    n = len(returns_arr)
    mean = np.mean(returns_arr)
    std = np.std(returns_arr)

    if std == 0:
        return 0.0

    skew = (n / ((n - 1) * (n - 2))) * np.sum(((returns_arr - mean) / std) ** 3)
    return skew


def calculate_kurtosis(returns: List[float]) -> float:
    """
    Calculate return distribution kurtosis (excess kurtosis).

    Higher kurtosis = fatter tails = more extreme events.
    Normal distribution has kurtosis of 0 (excess).

    Args:
        returns: List of returns

    Returns:
        Excess kurtosis
    """
    if len(returns) < 4:
        return 0.0

    returns_arr = np.array(returns)
    n = len(returns_arr)
    mean = np.mean(returns_arr)
    std = np.std(returns_arr)

    if std == 0:
        return 0.0

    m4 = np.mean((returns_arr - mean) ** 4)
    kurt = (m4 / (std**4)) - 3  # Excess kurtosis

    return kurt


def analyze_drawdowns(equity_curve: List[float]) -> DrawdownMetrics:
    """
    Comprehensive drawdown analysis.

    Args:
        equity_curve: List of equity values over time

    Returns:
        DrawdownMetrics with full analysis
    """
    if len(equity_curve) < 2:
        return DrawdownMetrics(
            max_drawdown=0,
            max_drawdown_pct=0,
            max_drawdown_duration=0,
            avg_drawdown=0,
            avg_drawdown_duration=0,
            num_drawdowns=0,
            current_drawdown=0,
            recovery_time=None,
            underwater_curve=[],
        )

    equity = np.array(equity_curve)
    n = len(equity)

    # Calculate running maximum
    running_max = np.maximum.accumulate(equity)

    # Calculate underwater curve (drawdown at each point)
    underwater = (running_max - equity) / running_max
    underwater_curve = underwater.tolist()

    # Find max drawdown
    max_dd = np.max(underwater)
    max_dd_idx = np.argmax(underwater)

    # Find max drawdown start (peak before max drawdown)
    peak_idx = np.argmax(equity[: max_dd_idx + 1])

    # Find recovery (when equity returns to peak)
    recovery_time = None
    for i in range(max_dd_idx + 1, n):
        if equity[i] >= running_max[max_dd_idx]:
            recovery_time = i - peak_idx
            break

    # Analyze individual drawdowns
    drawdowns = []
    in_drawdown = False
    dd_start = 0
    current_peak = equity[0]

    for i in range(1, n):
        if equity[i] < current_peak:
            if not in_drawdown:
                in_drawdown = True
                dd_start = i - 1
        else:
            if in_drawdown:
                # Drawdown ended
                dd_depth = (current_peak - np.min(equity[dd_start:i])) / current_peak
                dd_duration = i - dd_start
                drawdowns.append((dd_depth, dd_duration))
                in_drawdown = False
            current_peak = equity[i]

    # Handle ongoing drawdown
    current_dd = underwater[-1]
    if in_drawdown:
        dd_duration = n - dd_start

    # Calculate averages
    if drawdowns:
        avg_dd = np.mean([d[0] for d in drawdowns])
        avg_duration = np.mean([d[1] for d in drawdowns])
    else:
        avg_dd = 0
        avg_duration = 0

    # Max drawdown duration
    max_dd_duration = 0
    current_duration = 0
    for i in range(1, n):
        if underwater[i] > 0:
            current_duration += 1
            max_dd_duration = max(max_dd_duration, current_duration)
        else:
            current_duration = 0

    return DrawdownMetrics(
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd * 100,
        max_drawdown_duration=max_dd_duration,
        avg_drawdown=avg_dd,
        avg_drawdown_duration=avg_duration,
        num_drawdowns=len(drawdowns),
        current_drawdown=current_dd,
        recovery_time=recovery_time,
        underwater_curve=underwater_curve,
    )


def calculate_expectancy(
    wins: List[float],
    losses: List[float],
) -> float:
    """
    Calculate trade expectancy.

    Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

    Args:
        wins: List of winning trade returns
        losses: List of losing trade returns (as positive numbers)

    Returns:
        Expectancy per trade
    """
    total_trades = len(wins) + len(losses)
    if total_trades == 0:
        return 0.0

    win_rate = len(wins) / total_trades
    loss_rate = len(losses) / total_trades

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    return (win_rate * avg_win) - (loss_rate * avg_loss)


def calculate_payoff_ratio(
    wins: List[float],
    losses: List[float],
) -> float:
    """
    Calculate payoff ratio (reward-to-risk ratio).

    Args:
        wins: List of winning trade returns
        losses: List of losing trade returns

    Returns:
        Average win / Average loss
    """
    if not wins or not losses:
        return 0.0

    avg_win = np.mean(wins)
    avg_loss = np.mean(np.abs(losses))

    if avg_loss == 0:
        return float("inf") if avg_win > 0 else 0.0

    return avg_win / avg_loss


def calculate_advanced_metrics(
    returns: List[float],
    equity_curve: List[float],
    benchmark_returns: Optional[List[float]] = None,
    risk_free_rate: float = 0.03,
) -> AdvancedPerformanceMetrics:
    """
    Calculate comprehensive advanced performance metrics.

    Args:
        returns: List of period returns
        equity_curve: List of equity values
        benchmark_returns: Optional benchmark returns for relative metrics
        risk_free_rate: Annual risk-free rate

    Returns:
        AdvancedPerformanceMetrics
    """
    if not returns or not equity_curve:
        return AdvancedPerformanceMetrics(
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            omega_ratio=0,
            information_ratio=0,
            treynor_ratio=0,
            gain_to_pain_ratio=0,
            tail_ratio=0,
            var_95=0,
            var_99=0,
            cvar_95=0,
            cvar_99=0,
            max_drawdown_pct=0,
            max_drawdown_duration=0,
            avg_drawdown_pct=0,
            win_rate=0,
            profit_factor=0,
            expectancy=0,
            payoff_ratio=0,
            volatility_annual=0,
            downside_deviation=0,
            upside_potential=0,
            skewness=0,
            kurtosis=0,
        )

    returns_arr = np.array(returns)

    # Basic metrics
    daily_rf = risk_free_rate / 252
    excess_returns = returns_arr - daily_rf

    avg_excess = np.mean(excess_returns)
    std_returns = np.std(returns_arr)
    sharpe = (avg_excess / std_returns * np.sqrt(252)) if std_returns > 0 else 0

    # Sortino (downside deviation)
    downside = excess_returns[excess_returns < 0]
    downside_dev = np.std(downside) if len(downside) > 0 else 0
    sortino = (avg_excess / downside_dev * np.sqrt(252)) if downside_dev > 0 else 0

    # Drawdown analysis
    dd_metrics = analyze_drawdowns(equity_curve)

    # Calmar
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    calmar = (
        (total_return * 100) / dd_metrics.max_drawdown_pct
        if dd_metrics.max_drawdown_pct > 0
        else 0
    )

    # Advanced ratios
    omega = calculate_omega_ratio(returns)
    tail_ratio = calculate_tail_ratio(returns)
    gain_to_pain = calculate_gain_to_pain(returns)

    # Risk metrics
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    cvar_95 = calculate_cvar(returns, 0.95)
    cvar_99 = calculate_cvar(returns, 0.99)

    # Benchmark-relative metrics
    if benchmark_returns and len(benchmark_returns) == len(returns):
        info_ratio = calculate_information_ratio(returns, benchmark_returns)
        treynor = calculate_treynor_ratio(returns, benchmark_returns, risk_free_rate)
    else:
        info_ratio = 0
        treynor = 0

    # Trade statistics
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]

    win_rate = len(wins) / len(returns) * 100 if returns else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")
    expectancy = calculate_expectancy(wins, [abs(l) for l in losses])
    payoff = calculate_payoff_ratio(wins, losses)

    # Volatility metrics
    volatility_annual = std_returns * np.sqrt(252) * 100
    upside = returns_arr[returns_arr > 0]
    upside_potential = np.mean(upside) if len(upside) > 0 else 0

    # Distribution metrics
    skewness = calculate_skewness(returns)
    kurtosis = calculate_kurtosis(returns)

    return AdvancedPerformanceMetrics(
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        omega_ratio=omega,
        information_ratio=info_ratio,
        treynor_ratio=treynor,
        gain_to_pain_ratio=gain_to_pain,
        tail_ratio=tail_ratio,
        var_95=var_95 * 100,
        var_99=var_99 * 100,
        cvar_95=cvar_95 * 100,
        cvar_99=cvar_99 * 100,
        max_drawdown_pct=dd_metrics.max_drawdown_pct,
        max_drawdown_duration=dd_metrics.max_drawdown_duration,
        avg_drawdown_pct=dd_metrics.avg_drawdown * 100,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy * 100,
        payoff_ratio=payoff,
        volatility_annual=volatility_annual,
        downside_deviation=downside_dev * np.sqrt(252) * 100 if downside_dev > 0 else 0,
        upside_potential=upside_potential * 100,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def generate_performance_report(
    metrics: AdvancedPerformanceMetrics,
    strategy_name: str = "Strategy",
) -> str:
    """
    Generate a formatted performance report.

    Args:
        metrics: Calculated metrics
        strategy_name: Name of the strategy

    Returns:
        Formatted report string
    """
    report = f"""
================================================================================
                    PERFORMANCE REPORT: {strategy_name}
================================================================================

RISK-ADJUSTED RETURNS
---------------------
  Sharpe Ratio:         {metrics.sharpe_ratio:>10.2f}
  Sortino Ratio:        {metrics.sortino_ratio:>10.2f}
  Calmar Ratio:         {metrics.calmar_ratio:>10.2f}
  Omega Ratio:          {metrics.omega_ratio:>10.2f}
  Information Ratio:    {metrics.information_ratio:>10.2f}
  Treynor Ratio:        {metrics.treynor_ratio:>10.2f}
  Gain-to-Pain:         {metrics.gain_to_pain_ratio:>10.2f}

RISK METRICS
------------
  VaR (95%):            {metrics.var_95:>10.2f}%
  VaR (99%):            {metrics.var_99:>10.2f}%
  CVaR (95%):           {metrics.cvar_95:>10.2f}%
  CVaR (99%):           {metrics.cvar_99:>10.2f}%
  Max Drawdown:         {metrics.max_drawdown_pct:>10.2f}%
  Max DD Duration:      {metrics.max_drawdown_duration:>10d} periods
  Avg Drawdown:         {metrics.avg_drawdown_pct:>10.2f}%

TRADE STATISTICS
----------------
  Win Rate:             {metrics.win_rate:>10.2f}%
  Profit Factor:        {metrics.profit_factor:>10.2f}
  Expectancy:           {metrics.expectancy:>10.2f}%
  Payoff Ratio:         {metrics.payoff_ratio:>10.2f}

VOLATILITY METRICS
------------------
  Annual Volatility:    {metrics.volatility_annual:>10.2f}%
  Downside Deviation:   {metrics.downside_deviation:>10.2f}%
  Upside Potential:     {metrics.upside_potential:>10.2f}%
  Tail Ratio:           {metrics.tail_ratio:>10.2f}

DISTRIBUTION
------------
  Skewness:             {metrics.skewness:>10.2f}
  Kurtosis:             {metrics.kurtosis:>10.2f}

================================================================================
"""
    return report
