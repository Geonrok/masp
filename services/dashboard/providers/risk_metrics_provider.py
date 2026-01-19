"""Risk metrics provider - provides returns data for risk_metrics_panel component."""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _get_trade_logger():
    """Get TradeLogger instance."""
    try:
        from libs.adapters.trade_logger import TradeLogger
        return TradeLogger()
    except ImportError:
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


def _calculate_daily_returns(trades: List[Dict]) -> Tuple[List[float], List[date]]:
    """Calculate daily returns from trades.

    Args:
        trades: List of trade dicts

    Returns:
        Tuple of (returns_list, dates_list)
    """
    if not trades:
        return [], []

    # Group trades by date
    daily_data: Dict[str, Dict[str, float]] = {}  # date -> {pnl, volume}

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

        if date_str not in daily_data:
            daily_data[date_str] = {"pnl": 0.0, "volume": 0.0}

        daily_data[date_str]["pnl"] += pnl
        daily_data[date_str]["volume"] += volume

    # Sort by date and calculate returns
    sorted_dates = sorted(daily_data.keys())
    returns = []
    dates = []

    for date_str in sorted_dates:
        data = daily_data[date_str]
        volume = data["volume"]

        if volume > 0:
            return_pct = (data["pnl"] / volume) * 100
        else:
            return_pct = 0.0

        returns.append(return_pct)
        dates.append(datetime.strptime(date_str, "%Y-%m-%d").date())

    return returns, dates


def _calculate_equity_curve(
    returns: List[float], initial_capital: float = 10_000_000
) -> List[float]:
    """Calculate equity curve from returns.

    Args:
        returns: List of daily return percentages
        initial_capital: Starting capital (default 10M KRW)

    Returns:
        List of equity values
    """
    equity = [initial_capital]

    for ret in returns:
        new_value = equity[-1] * (1 + ret / 100)
        equity.append(new_value)

    return equity


def get_risk_metrics_data(days: int = 30) -> Optional[Tuple[List[float], List[float], List[date]]]:
    """Get data for risk_metrics_panel.

    Args:
        days: Number of days to analyze

    Returns:
        Tuple of (returns, equity_curve, dates) or None for demo mode
    """
    trade_logger = _get_trade_logger()

    if trade_logger is None:
        return None

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
        return None

    # Calculate returns
    returns, dates = _calculate_daily_returns(all_trades)

    if not returns:
        return None

    # Calculate equity curve
    equity_curve = _calculate_equity_curve(returns)

    return returns, equity_curve, dates
