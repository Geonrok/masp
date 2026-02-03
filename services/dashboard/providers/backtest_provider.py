"""Backtest provider - connects BacktestStore to backtest_viewer component."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_backtest_store():
    """Get BacktestStore instance.

    Returns:
        BacktestStore instance or None if unavailable
    """
    try:
        from libs.adapters.backtest_store import BacktestStore

        return BacktestStore()
    except ImportError as e:
        logger.debug("BacktestStore import failed: %s", e)
        return None
    except Exception as e:
        logger.debug("BacktestStore initialization failed: %s", e)
        return None


def get_backtest_data(
    strategy_name: Optional[str] = None,
    backtest_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Get backtest data for backtest_viewer component.

    Args:
        strategy_name: Optional strategy name filter
        backtest_id: Optional specific backtest ID

    Returns:
        Dict in backtest_viewer format or None for demo mode
    """
    store = _get_backtest_store()

    if store is None:
        return None

    # Get specific backtest or latest
    if backtest_id and strategy_name:
        result = store.load(strategy_name, backtest_id)
    else:
        result = store.get_latest(strategy_name)

    if result is None:
        return None

    return result.to_viewer_format()


def get_backtest_list() -> List[Dict[str, Any]]:
    """Get list of all available backtests.

    Returns:
        List of backtest metadata dicts
    """
    store = _get_backtest_store()

    if store is None:
        return []

    return store.list_backtests()


def get_strategy_names() -> List[str]:
    """Get list of strategy names with backtests.

    Returns:
        List of strategy names
    """
    store = _get_backtest_store()

    if store is None:
        return []

    return store.get_strategy_names()


def get_backtest_provider(
    strategy_name: Optional[str] = None,
) -> Optional[Callable[[], Dict[str, Any]]]:
    """Get backtest data provider function.

    Args:
        strategy_name: Optional strategy name filter

    Returns:
        Provider function or None if no backtests available
    """
    store = _get_backtest_store()

    if store is None:
        return None

    # Check if any backtests exist
    backtests = store.list_backtests()
    if not backtests:
        return None

    def provider() -> Dict[str, Any]:
        data = get_backtest_data(strategy_name=strategy_name)
        return data if data else {}

    return provider
