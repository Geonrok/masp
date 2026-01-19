"""Trade history provider - connects TradeLogger to trade_history component."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

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
        logger.warning("TradeLogger import failed: %s", e)
        return None
    except Exception as e:
        logger.warning("TradeLogger initialization failed: %s", e)
        return None


class TradeHistoryApiClient:
    """API client wrapper for trade history component.

    This class provides the interface expected by trade_history_panel.
    """

    def __init__(self, trade_logger=None):
        """Initialize with optional TradeLogger.

        Args:
            trade_logger: TradeLogger instance (auto-creates if None)
        """
        self._logger = trade_logger or _get_trade_logger()

    def get_trade_history(
        self,
        days: int = 7,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Get trade history from TradeLogger.

        Args:
            days: Number of days to look back (default 7)
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            List of trade dicts in trade_history component format
        """
        if self._logger is None:
            return []

        trades: List[Dict[str, Any]] = []

        # Determine date range
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=days)

        # Collect trades from each day
        current_date = start_date
        while current_date <= end_date:
            try:
                day_trades = self._logger.get_trades(current_date)
                for trade in day_trades:
                    converted = self._convert_trade(trade)
                    if converted:
                        trades.append(converted)
            except Exception as e:
                logger.debug("Failed to get trades for %s: %s", current_date, e)
            current_date += timedelta(days=1)

        # Sort by timestamp descending (newest first)
        trades.sort(key=lambda t: t.get("timestamp", datetime.min), reverse=True)

        return trades

    def _convert_trade(self, raw_trade: Dict) -> Optional[Dict[str, Any]]:
        """Convert TradeLogger format to trade_history component format.

        Args:
            raw_trade: Trade dict from TradeLogger

        Returns:
            Converted trade dict or None if invalid
        """
        try:
            # Parse timestamp
            timestamp_str = raw_trade.get("timestamp", "")
            if isinstance(timestamp_str, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except ValueError:
                    timestamp = datetime.now()
            else:
                timestamp = timestamp_str or datetime.now()

            # Parse numeric values
            quantity = self._safe_float(raw_trade.get("quantity", 0))
            price = self._safe_float(raw_trade.get("price", 0))
            total = quantity * price
            fee = self._safe_float(raw_trade.get("fee", 0))

            return {
                "id": raw_trade.get("order_id", ""),
                "timestamp": timestamp,
                "exchange": raw_trade.get("exchange", "unknown"),
                "symbol": raw_trade.get("symbol", ""),
                "side": raw_trade.get("side", "").upper(),
                "quantity": quantity,
                "price": price,
                "total": total,
                "fee": fee,
                "status": raw_trade.get("status", "UNKNOWN"),
                "pnl": self._safe_float(raw_trade.get("pnl", 0)),
                "message": raw_trade.get("message", ""),
            }
        except Exception as e:
            logger.debug("Failed to convert trade: %s", e)
            return None

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert value to float."""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except (ValueError, TypeError):
            return default

    def get_daily_summary(self, trade_date: Optional[date] = None) -> Dict[str, Any]:
        """Get daily trade summary.

        Args:
            trade_date: Date to get summary for (default today)

        Returns:
            Summary dict with trade statistics
        """
        if self._logger is None:
            return {
                "date": (trade_date or date.today()).isoformat(),
                "total_trades": 0,
                "buy_count": 0,
                "sell_count": 0,
                "total_volume": 0.0,
                "total_fee": 0.0,
                "total_pnl": 0.0,
            }

        return self._logger.get_daily_summary(trade_date)


def get_trade_history_client() -> Optional[TradeHistoryApiClient]:
    """Get trade history API client.

    Returns:
        TradeHistoryApiClient if TradeLogger available, None otherwise
    """
    trade_logger = _get_trade_logger()
    if trade_logger is None:
        return None
    return TradeHistoryApiClient(trade_logger)
