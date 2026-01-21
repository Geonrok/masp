"""
Trading Metrics Aggregator for MASP.

Tracks and aggregates trading performance metrics including:
- Order success/failure rates
- Execution latency
- Slippage analysis
- Exchange-level metrics
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order execution status."""

    SUCCESS = "success"
    PARTIAL = "partial"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class OrderMetrics:
    """Metrics for a single order execution."""

    order_id: str
    exchange: str
    symbol: str
    side: str  # BUY or SELL
    status: OrderStatus
    timestamp: datetime = field(default_factory=datetime.now)

    # Timing
    submit_latency_ms: float = 0.0  # Time to submit order
    fill_latency_ms: float = 0.0  # Time to fill
    total_latency_ms: float = 0.0  # Total round-trip

    # Execution quality
    requested_price: Optional[float] = None
    executed_price: Optional[float] = None
    slippage_pct: float = 0.0

    requested_quantity: float = 0.0
    filled_quantity: float = 0.0
    fill_rate: float = 1.0

    # Fees
    fees: float = 0.0
    fee_currency: str = "KRW"


@dataclass
class LatencyMetrics:
    """Aggregated latency statistics."""

    exchange: str
    period_minutes: int
    sample_count: int

    avg_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


class TradingMetricsAggregator:
    """
    Aggregates trading metrics across exchanges.

    Tracks order success rates, latency percentiles, slippage,
    and provides exchange-level breakdown.

    Example:
        aggregator = TradingMetricsAggregator()

        # Record an order
        aggregator.record_order(OrderMetrics(
            order_id="123",
            exchange="upbit",
            symbol="BTC/KRW",
            side="BUY",
            status=OrderStatus.SUCCESS,
            submit_latency_ms=50,
            fill_latency_ms=100,
            total_latency_ms=150,
            slippage_pct=0.001,
        ))

        # Get statistics
        stats = aggregator.get_exchange_stats("upbit")
        print(f"Success rate: {stats['success_rate']*100}%")
    """

    _instance: Optional["TradingMetricsAggregator"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        history_hours: int = 24,
        max_records_per_exchange: int = 10000,
    ):
        """
        Initialize trading metrics aggregator.

        Args:
            history_hours: Hours of history to keep
            max_records_per_exchange: Max records per exchange
        """
        self._history_hours = history_hours
        self._max_records = max_records_per_exchange

        # Storage by exchange
        self._orders: Dict[str, List[OrderMetrics]] = defaultdict(list)
        self._lock = threading.Lock()

        # Cached aggregations
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl_seconds = 60
        self._cache_timestamps: Dict[str, float] = {}

    @classmethod
    def get_instance(cls) -> "TradingMetricsAggregator":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def record_order(self, metrics: OrderMetrics) -> None:
        """
        Record order execution metrics.

        Args:
            metrics: Order metrics to record
        """
        with self._lock:
            self._orders[metrics.exchange].append(metrics)

            # Trim old records
            if len(self._orders[metrics.exchange]) > self._max_records:
                self._orders[metrics.exchange] = self._orders[metrics.exchange][
                    -self._max_records :
                ]

            # Invalidate cache
            cache_key = f"stats_{metrics.exchange}"
            if cache_key in self._cache:
                del self._cache[cache_key]

        logger.debug(
            "[TradingMetrics] Order recorded: %s %s %s %s",
            metrics.exchange,
            metrics.symbol,
            metrics.side,
            metrics.status.value,
        )

    def record_order_simple(
        self,
        order_id: str,
        exchange: str,
        symbol: str,
        side: str,
        success: bool,
        latency_ms: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> None:
        """
        Simple method to record order metrics.

        Args:
            order_id: Order identifier
            exchange: Exchange name
            symbol: Trading pair
            side: BUY or SELL
            success: Whether order was successful
            latency_ms: Total execution time in ms
            slippage_pct: Slippage percentage
        """
        self.record_order(
            OrderMetrics(
                order_id=order_id,
                exchange=exchange,
                symbol=symbol,
                side=side,
                status=OrderStatus.SUCCESS if success else OrderStatus.ERROR,
                total_latency_ms=latency_ms,
                slippage_pct=slippage_pct,
            )
        )

    def get_exchange_stats(
        self,
        exchange: str,
        minutes: int = 60,
    ) -> Dict[str, Any]:
        """
        Get aggregated statistics for an exchange.

        Args:
            exchange: Exchange name
            minutes: Lookback period in minutes

        Returns:
            Dictionary with success rate, latency stats, etc.
        """
        cache_key = f"stats_{exchange}_{minutes}"

        # Check cache
        if cache_key in self._cache:
            if time.time() - self._cache_timestamps.get(cache_key, 0) < self._cache_ttl_seconds:
                return self._cache[cache_key]

        with self._lock:
            orders = self._get_recent_orders(exchange, minutes)

        if not orders:
            return {
                "exchange": exchange,
                "period_minutes": minutes,
                "total_orders": 0,
                "success_rate": 0.0,
                "latency": None,
                "slippage_avg_pct": 0.0,
            }

        # Calculate statistics
        total = len(orders)
        successful = sum(
            1 for o in orders if o.status in (OrderStatus.SUCCESS, OrderStatus.PARTIAL)
        )
        success_rate = successful / total if total > 0 else 0.0

        # Latency stats
        latencies = [o.total_latency_ms for o in orders if o.total_latency_ms > 0]
        latency_stats = self._calculate_latency_stats(latencies, exchange, minutes)

        # Slippage
        slippages = [o.slippage_pct for o in orders if o.slippage_pct != 0]
        avg_slippage = statistics.mean(slippages) if slippages else 0.0

        # By status breakdown
        status_counts = defaultdict(int)
        for o in orders:
            status_counts[o.status.value] += 1

        # By symbol breakdown
        symbol_counts = defaultdict(int)
        for o in orders:
            symbol_counts[o.symbol] += 1

        stats = {
            "exchange": exchange,
            "period_minutes": minutes,
            "total_orders": total,
            "success_rate": round(success_rate, 4),
            "successful_orders": successful,
            "failed_orders": total - successful,
            "latency": latency_stats.__dict__ if latency_stats else None,
            "slippage_avg_pct": round(avg_slippage, 6),
            "by_status": dict(status_counts),
            "by_symbol": dict(symbol_counts),
            "last_order_time": orders[-1].timestamp.isoformat() if orders else None,
        }

        # Update cache
        self._cache[cache_key] = stats
        self._cache_timestamps[cache_key] = time.time()

        return stats

    def get_all_exchanges_stats(
        self,
        minutes: int = 60,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all exchanges.

        Args:
            minutes: Lookback period

        Returns:
            Dictionary mapping exchange names to their stats
        """
        with self._lock:
            exchanges = list(self._orders.keys())

        return {exchange: self.get_exchange_stats(exchange, minutes) for exchange in exchanges}

    def get_latency_percentiles(
        self,
        exchange: str,
        minutes: int = 60,
    ) -> Optional[LatencyMetrics]:
        """
        Get latency percentiles for an exchange.

        Args:
            exchange: Exchange name
            minutes: Lookback period

        Returns:
            LatencyMetrics with percentile data
        """
        with self._lock:
            orders = self._get_recent_orders(exchange, minutes)

        latencies = [o.total_latency_ms for o in orders if o.total_latency_ms > 0]
        return self._calculate_latency_stats(latencies, exchange, minutes)

    def _get_recent_orders(
        self,
        exchange: str,
        minutes: int,
    ) -> List[OrderMetrics]:
        """Get orders within the time window."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [o for o in self._orders.get(exchange, []) if o.timestamp >= cutoff]

    def _calculate_latency_stats(
        self,
        latencies: List[float],
        exchange: str,
        period_minutes: int,
    ) -> Optional[LatencyMetrics]:
        """Calculate latency statistics from values."""
        if not latencies:
            return None

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return LatencyMetrics(
            exchange=exchange,
            period_minutes=period_minutes,
            sample_count=n,
            avg_ms=round(statistics.mean(latencies), 2),
            min_ms=round(sorted_latencies[0], 2),
            max_ms=round(sorted_latencies[-1], 2),
            p50_ms=round(sorted_latencies[n // 2], 2),
            p95_ms=round(sorted_latencies[int(n * 0.95)] if n >= 20 else sorted_latencies[-1], 2),
            p99_ms=round(sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1], 2),
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get overall trading metrics summary.

        Returns:
            Summary dictionary with totals and per-exchange breakdown
        """
        all_stats = self.get_all_exchanges_stats()

        total_orders = sum(s["total_orders"] for s in all_stats.values())
        total_successful = sum(s["successful_orders"] for s in all_stats.values())

        return {
            "total_orders": total_orders,
            "total_successful": total_successful,
            "overall_success_rate": round(total_successful / total_orders, 4) if total_orders > 0 else 0.0,
            "exchanges": list(all_stats.keys()),
            "by_exchange": all_stats,
            "timestamp": datetime.now().isoformat(),
        }

    def cleanup_old_records(self) -> int:
        """
        Remove records older than history_hours.

        Returns:
            Number of records removed
        """
        cutoff = datetime.now() - timedelta(hours=self._history_hours)
        removed = 0

        with self._lock:
            for exchange in self._orders:
                before = len(self._orders[exchange])
                self._orders[exchange] = [
                    o for o in self._orders[exchange] if o.timestamp >= cutoff
                ]
                removed += before - len(self._orders[exchange])

        if removed > 0:
            logger.info("[TradingMetrics] Cleaned up %d old records", removed)

        return removed

    def clear(self) -> None:
        """Clear all records (for testing)."""
        with self._lock:
            self._orders.clear()
            self._cache.clear()
            self._cache_timestamps.clear()


# ============================================================================
# Convenience functions
# ============================================================================


def get_trading_metrics() -> TradingMetricsAggregator:
    """Get global trading metrics aggregator."""
    return TradingMetricsAggregator.get_instance()


def record_order(
    order_id: str,
    exchange: str,
    symbol: str,
    side: str,
    success: bool,
    latency_ms: float = 0.0,
) -> None:
    """Convenience function to record an order."""
    get_trading_metrics().record_order_simple(
        order_id=order_id,
        exchange=exchange,
        symbol=symbol,
        side=side,
        success=success,
        latency_ms=latency_ms,
    )
