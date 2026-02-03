"""Strategy performance provider - connects trade history to strategy_performance component."""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from services.dashboard.components.strategy_performance import (
    PerformanceMetrics,
    StrategyPerformance,
    TradeStats,
)

logger = logging.getLogger(__name__)


def _get_trade_logger():
    """Get TradeLogger instance.

    Returns:
        TradeLogger instance or None if unavailable
    """
    try:
        from libs.adapters.trade_logger import TradeLogger

        return TradeLogger()
    except ImportError as e:
        logger.debug("TradeLogger import failed: %s", e)
        return None
    except Exception as e:
        logger.debug("TradeLogger initialization failed: %s", e)
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return default
        return float(value)
    try:
        result = float(str(value).strip())
        if not math.isfinite(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def _calculate_returns(trades: List[Dict]) -> List[float]:
    """Calculate daily returns from trades.

    Args:
        trades: List of trade dicts

    Returns:
        List of daily return percentages
    """
    if not trades:
        return []

    # Group trades by date
    daily_pnl: Dict[str, float] = {}
    daily_volume: Dict[str, float] = {}

    for trade in trades:
        timestamp = trade.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                continue
        elif not isinstance(timestamp, datetime):
            continue

        date_str = timestamp.strftime("%Y-%m-%d")
        pnl = _safe_float(trade.get("pnl", 0))
        qty = _safe_float(trade.get("quantity", 0))
        price = _safe_float(trade.get("price", 0))
        volume = qty * price

        daily_pnl[date_str] = daily_pnl.get(date_str, 0) + pnl
        daily_volume[date_str] = daily_volume.get(date_str, 0) + volume

    # Calculate returns as percentage
    returns = []
    for date_str in sorted(daily_pnl.keys()):
        volume = daily_volume.get(date_str, 0)
        if volume > 0:
            returns.append((daily_pnl[date_str] / volume) * 100)
        else:
            returns.append(0.0)

    return returns


def _calculate_sharpe_ratio(
    returns: List[float], risk_free_rate: float = 0.02
) -> float:
    """Calculate Sharpe ratio from returns.

    Args:
        returns: List of return percentages
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0

    if std_dev == 0:
        return 0.0

    # Annualize (assuming daily returns)
    mean_return * 252
    annual_std = std_dev * math.sqrt(252)
    daily_rf = risk_free_rate / 252

    excess_return = mean_return - daily_rf
    annual_excess = excess_return * 252

    return annual_excess / annual_std if annual_std > 0 else 0.0


def _calculate_sortino_ratio(
    returns: List[float], risk_free_rate: float = 0.02
) -> float:
    """Calculate Sortino ratio from returns.

    Args:
        returns: List of return percentages
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    negative_returns = [r for r in returns if r < 0]

    if not negative_returns:
        return 0.0  # No downside deviation

    downside_variance = sum(r**2 for r in negative_returns) / len(negative_returns)
    downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0

    if downside_std == 0:
        return 0.0

    # Annualize
    mean_return * 252
    annual_downside_std = downside_std * math.sqrt(252)
    daily_rf = risk_free_rate / 252

    excess_return = mean_return - daily_rf
    annual_excess = excess_return * 252

    return annual_excess / annual_downside_std if annual_downside_std > 0 else 0.0


def _calculate_max_drawdown(returns: List[float]) -> float:
    """Calculate maximum drawdown from returns.

    Args:
        returns: List of return percentages

    Returns:
        Maximum drawdown as percentage
    """
    if not returns:
        return 0.0

    # Calculate cumulative equity curve
    equity = [100.0]  # Start at 100
    for r in returns:
        equity.append(equity[-1] * (1 + r / 100))

    # Find maximum drawdown
    peak = equity[0]
    max_dd = 0.0

    for value in equity[1:]:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd


def _calculate_volatility(returns: List[float]) -> float:
    """Calculate annualized volatility from returns.

    Args:
        returns: List of return percentages

    Returns:
        Annualized volatility as percentage
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0

    # Annualize
    return std_dev * math.sqrt(252)


def _calculate_trade_stats(trades: List[Dict]) -> TradeStats:
    """Calculate trade statistics.

    Args:
        trades: List of trade dicts

    Returns:
        TradeStats instance
    """
    if not trades:
        return TradeStats()

    wins = []
    losses = []

    for trade in trades:
        pnl = _safe_float(trade.get("pnl", 0))
        if pnl > 0:
            wins.append(pnl)
        elif pnl < 0:
            losses.append(abs(pnl))

    total_trades = len(trades)
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    total_wins = sum(wins)
    total_losses = sum(losses)
    profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0

    return TradeStats(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        avg_holding_time_hours=0.0,  # Would need entry/exit timestamps
    )


def _calculate_strategy_performance(
    strategy_id: str,
    strategy_name: str,
    trades: List[Dict],
    start_date: Optional[datetime] = None,
) -> StrategyPerformance:
    """Calculate performance for a strategy.

    Args:
        strategy_id: Strategy identifier
        strategy_name: Strategy display name
        trades: List of trade dicts
        start_date: Strategy start date

    Returns:
        StrategyPerformance instance
    """
    returns = _calculate_returns(trades)
    trade_stats = _calculate_trade_stats(trades)

    total_pnl = sum(_safe_float(t.get("pnl", 0)) for t in trades)
    total_volume = sum(
        _safe_float(t.get("quantity", 0)) * _safe_float(t.get("price", 0))
        for t in trades
    )
    total_return = (total_pnl / total_volume * 100) if total_volume > 0 else 0.0

    max_dd = _calculate_max_drawdown(returns)
    sharpe = _calculate_sharpe_ratio(returns)
    sortino = _calculate_sortino_ratio(returns)
    volatility = _calculate_volatility(returns)

    calmar = (total_return / max_dd) if max_dd > 0 else 0.0

    metrics = PerformanceMetrics(
        total_return=total_return,
        total_return_krw=total_pnl,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_krw=total_volume * max_dd / 100 if total_volume > 0 else 0.0,
        volatility=volatility,
        calmar_ratio=calmar,
    )

    return StrategyPerformance(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        is_active=True,
        start_date=start_date,
        metrics=metrics,
        trade_stats=trade_stats,
    )


@st.cache_data(ttl=30, show_spinner=False)
def get_strategy_performances(days: int = 30) -> List[StrategyPerformance]:
    """Get strategy performances from trade history.

    Cached for 30 seconds to reduce file reads.

    Args:
        days: Number of days to analyze

    Returns:
        List of StrategyPerformance instances
    """
    trade_logger = _get_trade_logger()

    if trade_logger is None:
        return []

    # Collect trades for the period
    all_trades: List[Dict] = []
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    current_date = start_date
    while current_date <= end_date:
        try:
            day_trades = trade_logger.get_trades(current_date)
            all_trades.extend(day_trades)
        except Exception as e:
            logger.debug("Failed to get trades for %s: %s", current_date, e)
        current_date += timedelta(days=1)

    if not all_trades:
        return []

    # Group trades by strategy (using exchange as proxy)
    trades_by_strategy: Dict[str, List[Dict]] = {}
    for trade in all_trades:
        exchange = trade.get("exchange", "unknown")
        strategy_key = f"{exchange}_strategy"
        if strategy_key not in trades_by_strategy:
            trades_by_strategy[strategy_key] = []
        trades_by_strategy[strategy_key].append(trade)

    # Calculate performance for each strategy
    performances = []
    for strategy_key, trades in trades_by_strategy.items():
        exchange = strategy_key.replace("_strategy", "")
        performance = _calculate_strategy_performance(
            strategy_id=strategy_key,
            strategy_name=f"{exchange.capitalize()} Trading Strategy",
            trades=trades,
            start_date=datetime.combine(start_date, datetime.min.time()),
        )
        performances.append(performance)

    return performances


def get_strategy_performance_provider() -> (
    Optional[Callable[[], List[StrategyPerformance]]]
):
    """Get performance provider function for strategy_performance component.

    Returns:
        Function that returns List[StrategyPerformance], or None for demo mode
    """
    # Check if trade logger is available
    trade_logger = _get_trade_logger()

    if trade_logger is None:
        return None

    def performance_provider() -> List[StrategyPerformance]:
        return get_strategy_performances()

    return performance_provider
