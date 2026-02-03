"""
Tests for Trading Metrics Aggregator.
"""

from datetime import datetime, timedelta

import pytest

from libs.monitoring.trading_metrics import (
    OrderMetrics,
    OrderStatus,
    TradingMetricsAggregator,
    get_trading_metrics,
    record_order,
)


@pytest.fixture
def aggregator():
    """Create test aggregator."""
    TradingMetricsAggregator.reset()
    return TradingMetricsAggregator(
        history_hours=24,
        max_records_per_exchange=100,
    )


class TestOrderStatus:
    """Tests for OrderStatus enum."""

    def test_values(self):
        """Test enum values."""
        assert OrderStatus.SUCCESS.value == "success"
        assert OrderStatus.PARTIAL.value == "partial"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.ERROR.value == "error"


class TestOrderMetrics:
    """Tests for OrderMetrics dataclass."""

    def test_creation(self):
        """Test creating order metrics."""
        metrics = OrderMetrics(
            order_id="123",
            exchange="upbit",
            symbol="BTC/KRW",
            side="BUY",
            status=OrderStatus.SUCCESS,
            total_latency_ms=150.0,
        )
        assert metrics.order_id == "123"
        assert metrics.exchange == "upbit"
        assert metrics.symbol == "BTC/KRW"
        assert metrics.side == "BUY"
        assert metrics.status == OrderStatus.SUCCESS
        assert metrics.total_latency_ms == 150.0


class TestTradingMetricsAggregator:
    """Tests for TradingMetricsAggregator class."""

    def test_init(self, aggregator):
        """Test initialization."""
        assert aggregator._history_hours == 24
        assert aggregator._max_records == 100

    def test_singleton(self):
        """Test singleton pattern."""
        TradingMetricsAggregator.reset()
        a1 = TradingMetricsAggregator.get_instance()
        a2 = TradingMetricsAggregator.get_instance()
        assert a1 is a2
        TradingMetricsAggregator.reset()

    def test_record_order(self, aggregator):
        """Test recording an order."""
        metrics = OrderMetrics(
            order_id="1",
            exchange="upbit",
            symbol="BTC/KRW",
            side="BUY",
            status=OrderStatus.SUCCESS,
            total_latency_ms=100.0,
        )
        aggregator.record_order(metrics)

        stats = aggregator.get_exchange_stats("upbit", minutes=60)
        assert stats["total_orders"] == 1
        assert stats["success_rate"] == 1.0

    def test_record_order_simple(self, aggregator):
        """Test simple order recording."""
        aggregator.record_order_simple(
            order_id="1",
            exchange="upbit",
            symbol="BTC/KRW",
            side="BUY",
            success=True,
            latency_ms=100.0,
        )

        stats = aggregator.get_exchange_stats("upbit")
        assert stats["total_orders"] == 1

    def test_success_rate_calculation(self, aggregator):
        """Test success rate calculation."""
        # 3 successful, 1 failed
        for i in range(3):
            aggregator.record_order(
                OrderMetrics(
                    order_id=str(i),
                    exchange="binance",
                    symbol="BTC/USDT",
                    side="BUY",
                    status=OrderStatus.SUCCESS,
                )
            )

        aggregator.record_order(
            OrderMetrics(
                order_id="fail",
                exchange="binance",
                symbol="BTC/USDT",
                side="BUY",
                status=OrderStatus.ERROR,
            )
        )

        stats = aggregator.get_exchange_stats("binance")
        assert stats["total_orders"] == 4
        assert stats["success_rate"] == 0.75

    def test_latency_percentiles(self, aggregator):
        """Test latency percentile calculation."""
        # Add orders with varying latency (start from 10 to avoid 0 being filtered)
        for i in range(50):
            aggregator.record_order(
                OrderMetrics(
                    order_id=str(i),
                    exchange="upbit",
                    symbol="BTC/KRW",
                    side="BUY",
                    status=OrderStatus.SUCCESS,
                    total_latency_ms=float((i + 1) * 10),  # 10, 20, 30, ..., 500
                )
            )

        latency = aggregator.get_latency_percentiles("upbit")
        assert latency is not None
        assert latency.sample_count == 50
        assert latency.min_ms == 10
        assert latency.max_ms == 500

    def test_exchange_stats_by_status(self, aggregator):
        """Test status breakdown in stats."""
        aggregator.record_order(
            OrderMetrics(
                order_id="1",
                exchange="upbit",
                symbol="BTC/KRW",
                side="BUY",
                status=OrderStatus.SUCCESS,
            )
        )
        aggregator.record_order(
            OrderMetrics(
                order_id="2",
                exchange="upbit",
                symbol="BTC/KRW",
                side="BUY",
                status=OrderStatus.REJECTED,
            )
        )

        stats = aggregator.get_exchange_stats("upbit")
        assert stats["by_status"]["success"] == 1
        assert stats["by_status"]["rejected"] == 1

    def test_exchange_stats_by_symbol(self, aggregator):
        """Test symbol breakdown in stats."""
        aggregator.record_order(
            OrderMetrics(
                order_id="1",
                exchange="upbit",
                symbol="BTC/KRW",
                side="BUY",
                status=OrderStatus.SUCCESS,
            )
        )
        aggregator.record_order(
            OrderMetrics(
                order_id="2",
                exchange="upbit",
                symbol="ETH/KRW",
                side="BUY",
                status=OrderStatus.SUCCESS,
            )
        )

        stats = aggregator.get_exchange_stats("upbit")
        assert stats["by_symbol"]["BTC/KRW"] == 1
        assert stats["by_symbol"]["ETH/KRW"] == 1

    def test_all_exchanges_stats(self, aggregator):
        """Test getting stats for all exchanges."""
        aggregator.record_order_simple("1", "upbit", "BTC/KRW", "BUY", True)
        aggregator.record_order_simple("2", "binance", "BTC/USDT", "BUY", True)

        all_stats = aggregator.get_all_exchanges_stats()
        assert "upbit" in all_stats
        assert "binance" in all_stats

    def test_summary(self, aggregator):
        """Test getting overall summary."""
        aggregator.record_order_simple("1", "upbit", "BTC/KRW", "BUY", True)
        aggregator.record_order_simple("2", "upbit", "BTC/KRW", "SELL", False)
        aggregator.record_order_simple("3", "binance", "BTC/USDT", "BUY", True)

        summary = aggregator.get_summary()
        assert summary["total_orders"] == 3
        assert summary["total_successful"] == 2
        assert summary["overall_success_rate"] == pytest.approx(0.6667, rel=0.01)

    def test_empty_exchange_stats(self, aggregator):
        """Test stats for exchange with no orders."""
        stats = aggregator.get_exchange_stats("nonexistent")
        assert stats["total_orders"] == 0
        assert stats["success_rate"] == 0.0

    def test_cleanup(self, aggregator):
        """Test cleanup of old records."""
        # Add an old record
        old_metrics = OrderMetrics(
            order_id="old",
            exchange="upbit",
            symbol="BTC/KRW",
            side="BUY",
            status=OrderStatus.SUCCESS,
        )
        old_metrics.timestamp = datetime.now() - timedelta(hours=48)
        aggregator.record_order(old_metrics)

        # Add a new record
        aggregator.record_order_simple("new", "upbit", "BTC/KRW", "BUY", True)

        removed = aggregator.cleanup_old_records()
        assert removed == 1

    def test_clear(self, aggregator):
        """Test clearing all records."""
        aggregator.record_order_simple("1", "upbit", "BTC/KRW", "BUY", True)
        aggregator.clear()

        stats = aggregator.get_exchange_stats("upbit")
        assert stats["total_orders"] == 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_trading_metrics(self):
        """Test get_trading_metrics function."""
        TradingMetricsAggregator.reset()
        metrics = get_trading_metrics()
        assert isinstance(metrics, TradingMetricsAggregator)
        TradingMetricsAggregator.reset()

    def test_record_order_function(self):
        """Test record_order convenience function."""
        TradingMetricsAggregator.reset()
        record_order(
            order_id="1",
            exchange="upbit",
            symbol="BTC/KRW",
            side="BUY",
            success=True,
            latency_ms=100.0,
        )

        metrics = get_trading_metrics()
        stats = metrics.get_exchange_stats("upbit")
        assert stats["total_orders"] == 1
        TradingMetricsAggregator.reset()
