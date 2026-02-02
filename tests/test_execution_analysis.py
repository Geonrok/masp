"""
Tests for Execution Analysis Module

Validates trade execution quality analysis:
- Slippage statistics
- Fill rate analysis
- Execution timing
- Cost analysis
- Report generation
"""

import pytest
from datetime import datetime, timedelta

from libs.analytics.execution_analysis import (
    ExecutionRecord,
    ExecutionAnalyzer,
    SlippageStats,
    FillRateStats,
    ExecutionTimeStats,
    CostAnalysis,
    format_execution_report,
)


class TestExecutionRecord:
    """Tests for ExecutionRecord dataclass."""

    def test_slippage_calculation_buy(self):
        """Test slippage for buy orders."""
        record = ExecutionRecord(
            order_id="1",
            symbol="BTC/KRW",
            side="buy",
            order_type="market",
            intended_price=50000000,
            executed_price=50100000,  # 0.2% slippage
            quantity=0.1,
            filled_quantity=0.1,
            commission=5000,
            timestamp_submitted=datetime.now(),
        )
        assert record.slippage == 100000
        assert abs(record.slippage_percent - 0.002) < 0.0001

    def test_slippage_calculation_sell(self):
        """Test slippage for sell orders."""
        record = ExecutionRecord(
            order_id="1",
            symbol="BTC/KRW",
            side="sell",
            order_type="market",
            intended_price=50000000,
            executed_price=49900000,  # 0.2% slippage (negative for seller)
            quantity=0.1,
            filled_quantity=0.1,
            commission=5000,
            timestamp_submitted=datetime.now(),
        )
        assert record.slippage == 100000
        assert abs(record.slippage_percent - 0.002) < 0.0001

    def test_fill_rate(self):
        """Test fill rate calculation."""
        # Full fill
        full = ExecutionRecord(
            order_id="1",
            symbol="BTC/KRW",
            side="buy",
            order_type="market",
            intended_price=50000000,
            executed_price=50000000,
            quantity=1.0,
            filled_quantity=1.0,
            commission=5000,
            timestamp_submitted=datetime.now(),
        )
        assert full.fill_rate == 1.0

        # Partial fill
        partial = ExecutionRecord(
            order_id="2",
            symbol="BTC/KRW",
            side="buy",
            order_type="limit",
            intended_price=50000000,
            executed_price=50000000,
            quantity=1.0,
            filled_quantity=0.7,
            commission=3500,
            timestamp_submitted=datetime.now(),
        )
        assert partial.fill_rate == 0.7

    def test_execution_time(self):
        """Test execution time calculation."""
        submitted = datetime(2024, 1, 1, 12, 0, 0)
        filled = datetime(2024, 1, 1, 12, 0, 2, 500000)  # 2.5 seconds later

        record = ExecutionRecord(
            order_id="1",
            symbol="BTC/KRW",
            side="buy",
            order_type="market",
            intended_price=50000000,
            executed_price=50000000,
            quantity=1.0,
            filled_quantity=1.0,
            commission=5000,
            timestamp_submitted=submitted,
            timestamp_filled=filled,
        )
        assert record.execution_time_ms == 2500

    def test_total_cost(self):
        """Test total cost calculation."""
        record = ExecutionRecord(
            order_id="1",
            symbol="BTC/KRW",
            side="buy",
            order_type="market",
            intended_price=50000000,
            executed_price=50100000,  # 100000 slippage per unit
            quantity=0.1,
            filled_quantity=0.1,
            commission=5000,
            timestamp_submitted=datetime.now(),
        )
        # Slippage cost = 100000 * 0.1 = 10000
        # Total = 10000 + 5000 = 15000
        assert record.total_cost == 15000


class TestExecutionAnalyzer:
    """Tests for ExecutionAnalyzer."""

    @pytest.fixture
    def analyzer_with_data(self):
        """Create analyzer with sample data."""
        analyzer = ExecutionAnalyzer()
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Record various executions
        analyzer.record_execution(
            order_id="1",
            symbol="BTC/KRW",
            side="buy",
            order_type="market",
            intended_price=50000000,
            executed_price=50100000,  # 0.2% slippage
            quantity=0.1,
            filled_quantity=0.1,
            commission=5000,
            timestamp_submitted=base_time,
            timestamp_filled=base_time + timedelta(seconds=1),
            exchange="upbit",
        )

        analyzer.record_execution(
            order_id="2",
            symbol="ETH/KRW",
            side="sell",
            order_type="limit",
            intended_price=3000000,
            executed_price=2997000,  # 0.1% slippage
            quantity=1.0,
            filled_quantity=0.8,  # Partial fill
            commission=3000,
            timestamp_submitted=base_time + timedelta(minutes=5),
            timestamp_filled=base_time + timedelta(minutes=5, seconds=2),
            exchange="upbit",
        )

        analyzer.record_execution(
            order_id="3",
            symbol="BTC/KRW",
            side="sell",
            order_type="market",
            intended_price=51000000,
            executed_price=50900000,  # 0.2% slippage
            quantity=0.1,
            filled_quantity=0.1,
            commission=5100,
            timestamp_submitted=base_time + timedelta(hours=1),
            timestamp_filled=base_time + timedelta(hours=1, milliseconds=500),
            exchange="bithumb",
        )

        return analyzer

    def test_record_execution(self):
        """Test recording executions."""
        analyzer = ExecutionAnalyzer()
        record = analyzer.record_execution(
            order_id="1",
            symbol="BTC/KRW",
            side="buy",
            order_type="market",
            intended_price=50000000,
            executed_price=50100000,
            quantity=0.1,
            filled_quantity=0.1,
            commission=5000,
            timestamp_submitted=datetime.now(),
        )
        assert len(analyzer.executions) == 1
        assert record.symbol == "BTC/KRW"

    def test_slippage_stats(self, analyzer_with_data):
        """Test slippage statistics calculation."""
        stats = analyzer_with_data.calculate_slippage_stats()
        assert stats.count == 3
        assert stats.mean > 0
        assert stats.min <= stats.mean <= stats.max

    def test_fill_rate_stats(self, analyzer_with_data):
        """Test fill rate statistics."""
        stats = analyzer_with_data.calculate_fill_rate_stats()
        assert stats.count == 3
        assert 0 < stats.mean_fill_rate <= 1
        # 2 full fills, 1 partial
        assert stats.full_fill_rate > 0.5

    def test_execution_time_stats(self, analyzer_with_data):
        """Test execution time statistics."""
        stats = analyzer_with_data.calculate_execution_time_stats()
        assert stats.count == 3
        assert stats.mean_ms > 0
        assert stats.min_ms <= stats.mean_ms <= stats.max_ms

    def test_cost_analysis(self, analyzer_with_data):
        """Test cost analysis."""
        cost = analyzer_with_data.calculate_cost_analysis()
        assert cost.count == 3
        assert cost.total_commission > 0
        assert cost.total_slippage_cost > 0
        assert cost.total_cost == cost.total_commission + cost.total_slippage_cost

    def test_generate_report(self, analyzer_with_data):
        """Test report generation."""
        report = analyzer_with_data.generate_report()

        assert report.slippage.count == 3
        assert report.fill_rate.count == 3
        assert "BTC/KRW" in report.by_symbol
        assert "upbit" in report.by_exchange
        assert "buy" in report.by_side
        assert len(report.recommendations) > 0

    def test_filter_by_symbol(self, analyzer_with_data):
        """Test filtering executions by symbol."""
        filtered = analyzer_with_data._filter_executions(symbol="BTC/KRW")
        assert len(filtered) == 2

    def test_filter_by_exchange(self, analyzer_with_data):
        """Test filtering executions by exchange."""
        filtered = analyzer_with_data._filter_executions(exchange="upbit")
        assert len(filtered) == 2

    def test_get_summary(self, analyzer_with_data):
        """Test summary generation."""
        summary = analyzer_with_data.get_summary()
        assert summary["total_executions"] == 3
        assert "avg_slippage_pct" in summary
        assert "total_cost" in summary

    def test_clear(self, analyzer_with_data):
        """Test clearing records."""
        analyzer_with_data.clear()
        assert len(analyzer_with_data.executions) == 0

    def test_empty_analyzer_stats(self):
        """Test statistics with no data."""
        analyzer = ExecutionAnalyzer()

        slippage = analyzer.calculate_slippage_stats()
        assert slippage.count == 0
        assert slippage.mean == 0

        fill_rate = analyzer.calculate_fill_rate_stats()
        assert fill_rate.count == 0

        cost = analyzer.calculate_cost_analysis()
        assert cost.count == 0


class TestReportGeneration:
    """Tests for report formatting."""

    def test_format_execution_report(self):
        """Test report formatting."""
        analyzer = ExecutionAnalyzer()
        analyzer.record_execution(
            order_id="1",
            symbol="BTC/KRW",
            side="buy",
            order_type="market",
            intended_price=50000000,
            executed_price=50100000,
            quantity=0.1,
            filled_quantity=0.1,
            commission=5000,
            timestamp_submitted=datetime.now(),
            timestamp_filled=datetime.now() + timedelta(seconds=1),
            exchange="upbit",
        )

        report = analyzer.generate_report()
        formatted = format_execution_report(report)

        assert "TRADE EXECUTION ANALYSIS REPORT" in formatted
        assert "SLIPPAGE ANALYSIS" in formatted
        assert "FILL RATE ANALYSIS" in formatted
        assert "EXECUTION TIMING" in formatted
        assert "COST ANALYSIS" in formatted
        assert "RECOMMENDATIONS" in formatted
        assert "BTC/KRW" in formatted


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_high_slippage_recommendation(self):
        """Test recommendation for high slippage."""
        analyzer = ExecutionAnalyzer()

        # Record high slippage trades
        for i in range(10):
            analyzer.record_execution(
                order_id=str(i),
                symbol="BTC/KRW",
                side="buy",
                order_type="market",
                intended_price=50000000,
                executed_price=50300000,  # 0.6% slippage
                quantity=0.1,
                filled_quantity=0.1,
                commission=5000,
                timestamp_submitted=datetime.now(),
            )

        report = analyzer.generate_report()
        recommendations = " ".join(report.recommendations)
        assert "slippage" in recommendations.lower()

    def test_low_fill_rate_recommendation(self):
        """Test recommendation for low fill rate."""
        analyzer = ExecutionAnalyzer()

        # Record low fill rate trades
        for i in range(10):
            analyzer.record_execution(
                order_id=str(i),
                symbol="BTC/KRW",
                side="buy",
                order_type="limit",
                intended_price=50000000,
                executed_price=50000000,
                quantity=1.0,
                filled_quantity=0.5,  # 50% fill
                commission=2500,
                timestamp_submitted=datetime.now(),
            )

        report = analyzer.generate_report()
        recommendations = " ".join(report.recommendations)
        assert "fill" in recommendations.lower()
