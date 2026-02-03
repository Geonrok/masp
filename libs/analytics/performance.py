"""
Performance Analytics Module

Provides performance measurement tools for trading strategies.
"""

import logging
import statistics
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

# [CRITICAL PATCH] Minimum sample size for statistical validity
MIN_SAMPLE_SIZE = 30  # Industry standard for Sharpe ratio


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""

    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_return: float
    volatility: float


def calculate_sharpe(returns: List[float], risk_free_rate: float = 0.03) -> float:
    """
    Calculate Sharpe Ratio (annualized).

    Args:
        returns: List of period returns (as decimals, e.g., 0.01 for 1%)
        risk_free_rate: Annual risk-free rate (default: 3%)

    Returns:
        Annualized Sharpe Ratio
    """
    # [CRITICAL PATCH] Minimum sample guard
    if len(returns) < MIN_SAMPLE_SIZE:
        logger.warning(
            f"[SHARPE] Insufficient samples: {len(returns)} < {MIN_SAMPLE_SIZE}. "
            f"Result may be statistically unreliable."
        )

    if len(returns) < 2:
        return 0.0

    # Calculate excess returns
    daily_rf = risk_free_rate / 252
    excess_returns = [r - daily_rf for r in returns]

    avg_excess = statistics.mean(excess_returns)
    std_excess = statistics.stdev(excess_returns)

    if std_excess == 0:
        return 0.0

    # Annualize
    sharpe = (avg_excess / std_excess) * (252**0.5)
    return sharpe


def calculate_sortino(returns: List[float], risk_free_rate: float = 0.03) -> float:
    """
    Calculate Sortino Ratio (annualized).

    Uses downside deviation instead of total volatility.

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Annualized Sortino Ratio
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / 252
    excess_returns = [r - daily_rf for r in returns]

    # Downside deviation (only negative returns)
    downside = [r for r in excess_returns if r < 0]

    if not downside:
        return float("inf")  # No downside = infinite Sortino

    downside_std = statistics.stdev(downside)
    if downside_std == 0:
        return 0.0

    avg_excess = statistics.mean(excess_returns)
    sortino = (avg_excess / downside_std) * (252**0.5)
    return sortino


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate maximum drawdown percentage.

    Args:
        equity_curve: List of equity values over time

    Returns:
        Max drawdown as percentage
    """
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0

    for equity in equity_curve:
        if equity > peak:
            peak = equity

        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return max_dd * 100


def calculate_calmar(
    total_return: float, max_drawdown_pct: float, years: float = 1.0
) -> float:
    """
    Calculate Calmar Ratio.

    Calmar = Annualized Return / Max Drawdown

    Args:
        total_return: Total return as decimal (e.g., 0.15 for 15%)
        max_drawdown_pct: Max drawdown as percentage (e.g., 10 for 10%)
        years: Number of years

    Returns:
        Calmar Ratio
    """
    if max_drawdown_pct == 0:
        return float("inf")

    annualized_return = (
        (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
    )
    calmar = (annualized_return * 100) / max_drawdown_pct
    return calmar


def calculate_performance_metrics(
    returns: List[float], equity_curve: List[float], risk_free_rate: float = 0.03
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: List of period returns
        equity_curve: List of equity values
        risk_free_rate: Annual risk-free rate

    Returns:
        PerformanceMetrics object
    """
    if not returns or not equity_curve:
        return PerformanceMetrics(
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown_pct=0,
            calmar_ratio=0,
            win_rate=0,
            profit_factor=0,
            avg_return=0,
            volatility=0,
        )

    sharpe = calculate_sharpe(returns, risk_free_rate)
    sortino = calculate_sortino(returns, risk_free_rate)
    max_dd = calculate_max_drawdown(equity_curve)

    winning = [r for r in returns if r > 0]
    losing = [r for r in returns if r <= 0]

    win_rate = len(winning) / len(returns) * 100 if returns else 0

    gross_profits = sum(winning)
    gross_losses = abs(sum(losing))
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")

    avg_return = statistics.mean(returns) if returns else 0
    volatility = statistics.stdev(returns) if len(returns) > 1 else 0

    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    calmar = calculate_calmar(total_return, max_dd)

    return PerformanceMetrics(
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=max_dd,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_return=avg_return * 100,  # Convert to percentage
        volatility=volatility * 100,
    )
