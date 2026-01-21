"""Trade history synchronization utility.

Fetches closed orders from Upbit and Bithumb APIs and imports them into TradeLogger.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _is_live_trading_enabled() -> bool:
    """Check if live trading is enabled."""
    return os.getenv("MASP_ENABLE_LIVE_TRADING") == "1"


def _get_trade_logger():
    """Get TradeLogger instance."""
    try:
        from libs.adapters.trade_logger import TradeLogger
        return TradeLogger()
    except Exception as e:
        logger.error(f"Failed to initialize TradeLogger: {e}")
        return None


def _convert_upbit_order_to_trade(order: Dict) -> Optional[Dict]:
    """Convert Upbit order to TradeLogger format.

    Args:
        order: Upbit order dict

    Returns:
        Trade dict for TradeLogger, or None if conversion fails
    """
    try:
        # Extract symbol from market (e.g., "KRW-BTC" -> "BTC/KRW")
        market = order.get("market", "")
        if market.startswith("KRW-"):
            symbol = f"{market[4:]}/KRW"
        else:
            symbol = market

        # Map side: "bid" -> "BUY", "ask" -> "SELL"
        side_raw = order.get("side", "").lower()
        side = "BUY" if side_raw == "bid" else "SELL" if side_raw == "ask" else side_raw.upper()

        # Extract values
        executed_volume = float(order.get("executed_volume", 0) or 0)
        avg_price = float(order.get("avg_price", 0) or 0)
        paid_fee = float(order.get("paid_fee", 0) or 0)

        # Skip if no execution
        if executed_volume <= 0:
            return None

        return {
            "timestamp": order.get("created_at", datetime.now().isoformat()),
            "exchange": "upbit",
            "order_id": order.get("uuid", ""),
            "symbol": symbol,
            "side": side,
            "quantity": executed_volume,
            "price": avg_price,
            "fee": paid_fee,
            "pnl": 0.0,  # PnL calculated separately
            "status": "FILLED",
            "message": f"Synced from Upbit API (trades_count: {order.get('trades_count', 0)})",
        }
    except Exception as e:
        logger.warning(f"Failed to convert Upbit order: {e}")
        return None


def _convert_bithumb_order_to_trade(order: Dict) -> Optional[Dict]:
    """Convert Bithumb order to TradeLogger format.

    Args:
        order: Bithumb order dict

    Returns:
        Trade dict for TradeLogger, or None if conversion fails
    """
    try:
        # Extract symbol from market (e.g., "KRW-BTC" -> "BTC/KRW")
        market = order.get("market", "")
        if market.startswith("KRW-"):
            symbol = f"{market[4:]}/KRW"
        else:
            symbol = market

        # Map side: "bid" -> "BUY", "ask" -> "SELL"
        side_raw = order.get("side", "").lower()
        side = "BUY" if side_raw == "bid" else "SELL" if side_raw == "ask" else side_raw.upper()

        # Extract values
        executed_volume = float(order.get("executed_volume", 0) or 0)
        avg_price = float(order.get("avg_price", 0) or 0)
        paid_fee = float(order.get("paid_fee", 0) or 0)

        # Skip if no execution
        if executed_volume <= 0:
            return None

        return {
            "timestamp": order.get("created_at", datetime.now().isoformat()),
            "exchange": "bithumb",
            "order_id": order.get("uuid", ""),
            "symbol": symbol,
            "side": side,
            "quantity": executed_volume,
            "price": avg_price,
            "fee": paid_fee,
            "pnl": 0.0,  # PnL calculated separately
            "status": "FILLED",
            "message": f"Synced from Bithumb API (trades_count: {order.get('trades_count', 0)})",
        }
    except Exception as e:
        logger.warning(f"Failed to convert Bithumb order: {e}")
        return None


def sync_upbit_trades(limit: int = 100) -> Tuple[int, int, str]:
    """Sync closed orders from Upbit to TradeLogger.

    Args:
        limit: Maximum number of orders to fetch

    Returns:
        Tuple of (synced_count, skipped_count, message)
    """
    if not _is_live_trading_enabled():
        return 0, 0, "Live trading is not enabled (MASP_ENABLE_LIVE_TRADING=1)"

    trade_logger = _get_trade_logger()
    if not trade_logger:
        return 0, 0, "Failed to initialize TradeLogger"

    try:
        from libs.adapters.real_upbit_spot import UpbitSpotExecution

        adapter = UpbitSpotExecution(
            access_key=os.getenv("UPBIT_ACCESS_KEY"),
            secret_key=os.getenv("UPBIT_SECRET_KEY"),
        )

        orders = adapter.get_closed_orders(limit=limit)
        if not orders:
            return 0, 0, "No closed orders found on Upbit"

        synced = 0
        skipped = 0

        for order in orders:
            trade = _convert_upbit_order_to_trade(order)
            if trade:
                if trade_logger.log_trade(trade):
                    synced += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        return synced, skipped, f"Upbit: synced {synced} trades, skipped {skipped}"

    except Exception as e:
        logger.error(f"Failed to sync Upbit trades: {e}")
        return 0, 0, f"Error: {str(e)}"


def sync_bithumb_trades(limit: int = 100) -> Tuple[int, int, str]:
    """Sync closed orders from Bithumb to TradeLogger.

    Args:
        limit: Maximum number of orders to fetch

    Returns:
        Tuple of (synced_count, skipped_count, message)
    """
    if not _is_live_trading_enabled():
        return 0, 0, "Live trading is not enabled (MASP_ENABLE_LIVE_TRADING=1)"

    trade_logger = _get_trade_logger()
    if not trade_logger:
        return 0, 0, "Failed to initialize TradeLogger"

    try:
        from libs.adapters.real_bithumb_execution import BithumbExecution

        adapter = BithumbExecution()

        orders = adapter.get_closed_orders(limit=limit)
        if not orders:
            return 0, 0, "No closed orders found on Bithumb"

        synced = 0
        skipped = 0

        for order in orders:
            trade = _convert_bithumb_order_to_trade(order)
            if trade:
                if trade_logger.log_trade(trade):
                    synced += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        return synced, skipped, f"Bithumb: synced {synced} trades, skipped {skipped}"

    except Exception as e:
        logger.error(f"Failed to sync Bithumb trades: {e}")
        return 0, 0, f"Error: {str(e)}"


def sync_all_trades(limit_per_exchange: int = 100) -> Dict[str, Any]:
    """Sync closed orders from all exchanges to TradeLogger.

    Args:
        limit_per_exchange: Maximum number of orders to fetch per exchange

    Returns:
        Dict with sync results:
        {
            "upbit": {"synced": int, "skipped": int, "message": str},
            "bithumb": {"synced": int, "skipped": int, "message": str},
            "total_synced": int,
            "total_skipped": int,
        }
    """
    results = {
        "upbit": {"synced": 0, "skipped": 0, "message": ""},
        "bithumb": {"synced": 0, "skipped": 0, "message": ""},
        "total_synced": 0,
        "total_skipped": 0,
    }

    # Sync Upbit
    upbit_synced, upbit_skipped, upbit_msg = sync_upbit_trades(limit_per_exchange)
    results["upbit"] = {
        "synced": upbit_synced,
        "skipped": upbit_skipped,
        "message": upbit_msg,
    }

    # Sync Bithumb
    bithumb_synced, bithumb_skipped, bithumb_msg = sync_bithumb_trades(limit_per_exchange)
    results["bithumb"] = {
        "synced": bithumb_synced,
        "skipped": bithumb_skipped,
        "message": bithumb_msg,
    }

    results["total_synced"] = upbit_synced + bithumb_synced
    results["total_skipped"] = upbit_skipped + bithumb_skipped

    logger.info(
        f"[TradeSync] Complete: {results['total_synced']} synced, "
        f"{results['total_skipped']} skipped"
    )

    return results
