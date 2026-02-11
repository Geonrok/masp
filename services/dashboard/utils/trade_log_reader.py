"""Trade log aggregation across multiple log directories.

The strategy runner logs trades to ``logs/{exchange}_trades/trades/`` while
execution adapters log to ``logs/trades/``.  This module discovers all known
directories and returns de-duplicated trades.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEDULE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "config"
    / "schedule_config.json"
)


def _get_trade_log_dirs() -> List[str]:
    """Discover all trade log directories.

    Returns:
        List of existing trade log directory paths.
    """
    dirs: List[str] = []

    # Default adapter path
    default_dir = "logs/trades"
    if Path(default_dir).exists():
        dirs.append(default_dir)

    # Per-exchange directories from schedule_config.json
    try:
        with open(_SCHEDULE_CONFIG_PATH, encoding="utf-8") as f:
            exchanges = json.load(f).get("exchanges", {})
        for name in exchanges:
            exchange_dir = f"logs/{name}_trades/trades"
            if Path(exchange_dir).exists():
                dirs.append(exchange_dir)
    except Exception:
        pass

    # Fallback: if no dirs found, still include default so TradeLogger can init
    if not dirs:
        dirs.append(default_dir)

    return dirs


def get_aggregated_trades(
    start_date: date,
    end_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """Read trades from all known log directories, de-duplicated by order_id.

    Args:
        start_date: First date to include.
        end_date: Last date to include (default today).

    Returns:
        List of trade dicts from all sources.
    """
    try:
        from libs.adapters.trade_logger import TradeLogger
    except ImportError:
        logger.debug("TradeLogger not available")
        return []

    if end_date is None:
        end_date = date.today()

    seen_ids: set = set()
    all_trades: List[Dict[str, Any]] = []

    for log_dir in _get_trade_log_dirs():
        try:
            trade_logger = TradeLogger(log_dir=log_dir)
            current = start_date
            while current <= end_date:
                try:
                    for trade in trade_logger.get_trades(current):
                        order_id = trade.get("order_id", "")
                        if order_id and order_id in seen_ids:
                            continue
                        if order_id:
                            seen_ids.add(order_id)
                        all_trades.append(trade)
                except Exception:
                    pass
                current += timedelta(days=1)
        except Exception:
            continue

    return all_trades
