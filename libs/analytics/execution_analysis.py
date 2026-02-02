"""
Trade Execution Analysis Module for MASP

Analyzes trade execution quality:
- Slippage statistics
- Fill rate analysis
- Execution timing
- Cost analysis (commissions, spread costs)
- Best execution metrics
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Record of a single order execution."""

    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "market", "limit"
    intended_price: float  # Expected/requested price
    executed_price: float  # Actual fill price
    quantity: float
    filled_quantity: float
    commission: float
    timestamp_submitted: datetime
    timestamp_filled: Optional[datetime] = None
    exchange: str = ""
    notes: str = ""

    @property
    def slippage(self) -> float:
        """Calculate slippage in price units."""
        if self.side == "buy":
            return self.executed_price - self.intended_price
        else:
            return self.intended_price - self.executed_price

    @property
    def slippage_percent(self) -> float:
        """Calculate slippage as percentage."""
        if self.intended_price == 0:
            return 0.0
        return self.slippage / self.intended_price

    @property
    def fill_rate(self) -> float:
        """Calculate fill rate (0-1)."""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity

    @property
    def execution_time_ms(self) -> Optional[float]:
        """Calculate execution time in milliseconds."""
        if self.timestamp_filled is None:
            return None
        delta = self.timestamp_filled - self.timestamp_submitted
        return delta.total_seconds() * 1000

    @property
    def total_cost(self) -> float:
        """Calculate total cost including slippage and commission."""
        slippage_cost = abs(self.slippage) * self.filled_quantity
        return slippage_cost + self.commission


@dataclass
class SlippageStats:
    """Slippage statistics."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    percentile_95: float
    total_slippage_cost: float
    count: int


@dataclass
class FillRateStats:
    """Fill rate statistics."""

    mean_fill_rate: float
    full_fill_rate: float  # Percentage of orders fully filled
    partial_fill_rate: float  # Percentage of orders partially filled
    zero_fill_rate: float  # Percentage of orders not filled
    count: int


@dataclass
class ExecutionTimeStats:
    """Execution timing statistics."""

    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    percentile_95_ms: float
    count: int


@dataclass
class CostAnalysis:
    """Cost breakdown analysis."""

    total_commission: float
    total_slippage_cost: float
    total_cost: float
    avg_commission_per_trade: float
    avg_slippage_per_trade: float
    avg_cost_per_trade: float
    commission_as_pct_of_volume: float
    slippage_as_pct_of_volume: float
    total_volume: float
    count: int


@dataclass
class ExecutionReport:
    """Comprehensive execution analysis report."""

    period_start: datetime
    period_end: datetime
    slippage: SlippageStats
    fill_rate: FillRateStats
    execution_time: ExecutionTimeStats
    cost: CostAnalysis
    by_symbol: dict[str, dict]
    by_exchange: dict[str, dict]
    by_side: dict[str, dict]
    recommendations: list[str] = field(default_factory=list)


class ExecutionAnalyzer:
    """
    Trade Execution Quality Analyzer.

    Tracks and analyzes execution quality metrics to identify
    areas for improvement and verify best execution.
    """

    def __init__(self):
        """Initialize Execution Analyzer."""
        self.executions: list[ExecutionRecord] = []
        logger.info("[ExecutionAnalyzer] Initialized")

    def record_execution(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        intended_price: float,
        executed_price: float,
        quantity: float,
        filled_quantity: float,
        commission: float,
        timestamp_submitted: datetime,
        timestamp_filled: Optional[datetime] = None,
        exchange: str = "",
        notes: str = "",
    ) -> ExecutionRecord:
        """
        Record a trade execution for analysis.

        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            side: "buy" or "sell"
            order_type: "market" or "limit"
            intended_price: Expected price
            executed_price: Actual fill price
            quantity: Requested quantity
            filled_quantity: Actually filled quantity
            commission: Commission paid
            timestamp_submitted: Order submission time
            timestamp_filled: Order fill time (optional)
            exchange: Exchange name
            notes: Additional notes

        Returns:
            ExecutionRecord object
        """
        record = ExecutionRecord(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            intended_price=intended_price,
            executed_price=executed_price,
            quantity=quantity,
            filled_quantity=filled_quantity,
            commission=commission,
            timestamp_submitted=timestamp_submitted,
            timestamp_filled=timestamp_filled,
            exchange=exchange,
            notes=notes,
        )
        self.executions.append(record)
        logger.debug(
            f"[ExecutionAnalyzer] Recorded: {symbol} {side} "
            f"slippage={record.slippage_percent:.4%}"
        )
        return record

    def _filter_executions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        side: Optional[str] = None,
    ) -> list[ExecutionRecord]:
        """Filter executions based on criteria."""
        filtered = self.executions

        if start_time:
            filtered = [e for e in filtered if e.timestamp_submitted >= start_time]
        if end_time:
            filtered = [e for e in filtered if e.timestamp_submitted <= end_time]
        if symbol:
            filtered = [e for e in filtered if e.symbol == symbol]
        if exchange:
            filtered = [e for e in filtered if e.exchange == exchange]
        if side:
            filtered = [e for e in filtered if e.side == side]

        return filtered

    def calculate_slippage_stats(
        self,
        executions: Optional[list[ExecutionRecord]] = None,
    ) -> SlippageStats:
        """Calculate slippage statistics."""
        execs = executions or self.executions
        if not execs:
            return SlippageStats(
                mean=0, median=0, std=0, min=0, max=0,
                percentile_95=0, total_slippage_cost=0, count=0,
            )

        slippages = [e.slippage_percent for e in execs]
        slippage_costs = [abs(e.slippage) * e.filled_quantity for e in execs]

        return SlippageStats(
            mean=statistics.mean(slippages),
            median=statistics.median(slippages),
            std=statistics.stdev(slippages) if len(slippages) > 1 else 0,
            min=min(slippages),
            max=max(slippages),
            percentile_95=float(np.percentile(slippages, 95)),
            total_slippage_cost=sum(slippage_costs),
            count=len(execs),
        )

    def calculate_fill_rate_stats(
        self,
        executions: Optional[list[ExecutionRecord]] = None,
    ) -> FillRateStats:
        """Calculate fill rate statistics."""
        execs = executions or self.executions
        if not execs:
            return FillRateStats(
                mean_fill_rate=0, full_fill_rate=0,
                partial_fill_rate=0, zero_fill_rate=0, count=0,
            )

        fill_rates = [e.fill_rate for e in execs]
        full_fills = sum(1 for r in fill_rates if r >= 0.999)
        partial_fills = sum(1 for r in fill_rates if 0 < r < 0.999)
        zero_fills = sum(1 for r in fill_rates if r == 0)

        return FillRateStats(
            mean_fill_rate=statistics.mean(fill_rates),
            full_fill_rate=full_fills / len(execs),
            partial_fill_rate=partial_fills / len(execs),
            zero_fill_rate=zero_fills / len(execs),
            count=len(execs),
        )

    def calculate_execution_time_stats(
        self,
        executions: Optional[list[ExecutionRecord]] = None,
    ) -> ExecutionTimeStats:
        """Calculate execution timing statistics."""
        execs = executions or self.executions
        times = [e.execution_time_ms for e in execs if e.execution_time_ms is not None]

        if not times:
            return ExecutionTimeStats(
                mean_ms=0, median_ms=0, std_ms=0,
                min_ms=0, max_ms=0, percentile_95_ms=0, count=0,
            )

        return ExecutionTimeStats(
            mean_ms=statistics.mean(times),
            median_ms=statistics.median(times),
            std_ms=statistics.stdev(times) if len(times) > 1 else 0,
            min_ms=min(times),
            max_ms=max(times),
            percentile_95_ms=float(np.percentile(times, 95)),
            count=len(times),
        )

    def calculate_cost_analysis(
        self,
        executions: Optional[list[ExecutionRecord]] = None,
    ) -> CostAnalysis:
        """Calculate comprehensive cost analysis."""
        execs = executions or self.executions
        if not execs:
            return CostAnalysis(
                total_commission=0, total_slippage_cost=0, total_cost=0,
                avg_commission_per_trade=0, avg_slippage_per_trade=0,
                avg_cost_per_trade=0, commission_as_pct_of_volume=0,
                slippage_as_pct_of_volume=0, total_volume=0, count=0,
            )

        total_commission = sum(e.commission for e in execs)
        total_slippage_cost = sum(abs(e.slippage) * e.filled_quantity for e in execs)
        total_cost = total_commission + total_slippage_cost
        total_volume = sum(e.executed_price * e.filled_quantity for e in execs)
        count = len(execs)

        return CostAnalysis(
            total_commission=total_commission,
            total_slippage_cost=total_slippage_cost,
            total_cost=total_cost,
            avg_commission_per_trade=total_commission / count,
            avg_slippage_per_trade=total_slippage_cost / count,
            avg_cost_per_trade=total_cost / count,
            commission_as_pct_of_volume=total_commission / total_volume if total_volume > 0 else 0,
            slippage_as_pct_of_volume=total_slippage_cost / total_volume if total_volume > 0 else 0,
            total_volume=total_volume,
            count=count,
        )

    def _analyze_by_dimension(
        self,
        executions: list[ExecutionRecord],
        key_func,
    ) -> dict[str, dict]:
        """Analyze executions grouped by a dimension."""
        groups: dict[str, list[ExecutionRecord]] = {}
        for e in executions:
            key = key_func(e)
            if key not in groups:
                groups[key] = []
            groups[key].append(e)

        results = {}
        for key, group in groups.items():
            slippage = self.calculate_slippage_stats(group)
            fill_rate = self.calculate_fill_rate_stats(group)
            results[key] = {
                "count": len(group),
                "avg_slippage_pct": slippage.mean,
                "avg_fill_rate": fill_rate.mean_fill_rate,
                "total_slippage_cost": slippage.total_slippage_cost,
            }
        return results

    def _generate_recommendations(
        self,
        slippage: SlippageStats,
        fill_rate: FillRateStats,
        execution_time: ExecutionTimeStats,
        cost: CostAnalysis,
    ) -> list[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Slippage recommendations
        if slippage.mean > 0.005:  # > 0.5%
            recommendations.append(
                f"High average slippage ({slippage.mean:.2%}): Consider using limit orders "
                "or reducing order size for better execution."
            )
        if slippage.percentile_95 > 0.01:  # > 1%
            recommendations.append(
                f"Slippage outliers detected (95th percentile: {slippage.percentile_95:.2%}): "
                "Review large orders or volatile periods."
            )

        # Fill rate recommendations
        if fill_rate.mean_fill_rate < 0.95:
            recommendations.append(
                f"Low fill rate ({fill_rate.mean_fill_rate:.1%}): "
                "Consider adjusting limit prices or switching to market orders."
            )
        if fill_rate.zero_fill_rate > 0.05:
            recommendations.append(
                f"High unfilled order rate ({fill_rate.zero_fill_rate:.1%}): "
                "Review limit price settings."
            )

        # Execution time recommendations
        if execution_time.mean_ms > 5000:  # > 5 seconds
            recommendations.append(
                f"Slow execution times (avg {execution_time.mean_ms:.0f}ms): "
                "Check network latency or consider co-location."
            )

        # Cost recommendations
        if cost.slippage_as_pct_of_volume > cost.commission_as_pct_of_volume:
            recommendations.append(
                "Slippage costs exceed commissions: Focus on execution quality "
                "over commission reduction."
            )

        if not recommendations:
            recommendations.append("Execution quality is within acceptable parameters.")

        return recommendations

    def generate_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> ExecutionReport:
        """
        Generate comprehensive execution analysis report.

        Args:
            start_time: Report period start (optional)
            end_time: Report period end (optional)

        Returns:
            ExecutionReport with all metrics and recommendations
        """
        if start_time is None:
            start_time = min(
                (e.timestamp_submitted for e in self.executions),
                default=datetime.now(),
            )
        if end_time is None:
            end_time = max(
                (e.timestamp_submitted for e in self.executions),
                default=datetime.now(),
            )

        filtered = self._filter_executions(start_time, end_time)

        slippage = self.calculate_slippage_stats(filtered)
        fill_rate = self.calculate_fill_rate_stats(filtered)
        execution_time = self.calculate_execution_time_stats(filtered)
        cost = self.calculate_cost_analysis(filtered)

        by_symbol = self._analyze_by_dimension(filtered, lambda e: e.symbol)
        by_exchange = self._analyze_by_dimension(filtered, lambda e: e.exchange)
        by_side = self._analyze_by_dimension(filtered, lambda e: e.side)

        recommendations = self._generate_recommendations(
            slippage, fill_rate, execution_time, cost
        )

        report = ExecutionReport(
            period_start=start_time,
            period_end=end_time,
            slippage=slippage,
            fill_rate=fill_rate,
            execution_time=execution_time,
            cost=cost,
            by_symbol=by_symbol,
            by_exchange=by_exchange,
            by_side=by_side,
            recommendations=recommendations,
        )

        logger.info(
            f"[ExecutionAnalyzer] Report generated: {len(filtered)} executions, "
            f"avg slippage={slippage.mean:.4%}, avg fill={fill_rate.mean_fill_rate:.1%}"
        )

        return report

    def get_summary(self) -> dict:
        """Get summary statistics as dictionary."""
        slippage = self.calculate_slippage_stats()
        fill_rate = self.calculate_fill_rate_stats()
        cost = self.calculate_cost_analysis()

        return {
            "total_executions": len(self.executions),
            "avg_slippage_pct": slippage.mean,
            "avg_fill_rate": fill_rate.mean_fill_rate,
            "total_cost": cost.total_cost,
            "total_volume": cost.total_volume,
        }

    def clear(self) -> None:
        """Clear all execution records."""
        self.executions.clear()
        logger.info("[ExecutionAnalyzer] Records cleared")


def format_execution_report(report: ExecutionReport) -> str:
    """
    Format execution report as human-readable string.

    Args:
        report: ExecutionReport to format

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "TRADE EXECUTION ANALYSIS REPORT",
        "=" * 60,
        f"Period: {report.period_start:%Y-%m-%d %H:%M} to {report.period_end:%Y-%m-%d %H:%M}",
        "",
        "--- SLIPPAGE ANALYSIS ---",
        f"  Mean Slippage:     {report.slippage.mean:.4%}",
        f"  Median Slippage:   {report.slippage.median:.4%}",
        f"  Std Dev:           {report.slippage.std:.4%}",
        f"  95th Percentile:   {report.slippage.percentile_95:.4%}",
        f"  Total Cost:        {report.slippage.total_slippage_cost:,.0f}",
        "",
        "--- FILL RATE ANALYSIS ---",
        f"  Mean Fill Rate:    {report.fill_rate.mean_fill_rate:.1%}",
        f"  Full Fills:        {report.fill_rate.full_fill_rate:.1%}",
        f"  Partial Fills:     {report.fill_rate.partial_fill_rate:.1%}",
        f"  Zero Fills:        {report.fill_rate.zero_fill_rate:.1%}",
        "",
        "--- EXECUTION TIMING ---",
        f"  Mean Time:         {report.execution_time.mean_ms:.0f} ms",
        f"  Median Time:       {report.execution_time.median_ms:.0f} ms",
        f"  95th Percentile:   {report.execution_time.percentile_95_ms:.0f} ms",
        "",
        "--- COST ANALYSIS ---",
        f"  Total Commission:  {report.cost.total_commission:,.0f}",
        f"  Total Slippage:    {report.cost.total_slippage_cost:,.0f}",
        f"  Total Cost:        {report.cost.total_cost:,.0f}",
        f"  Commission/Volume: {report.cost.commission_as_pct_of_volume:.4%}",
        f"  Slippage/Volume:   {report.cost.slippage_as_pct_of_volume:.4%}",
        "",
        "--- BY SYMBOL ---",
    ]

    for symbol, stats in report.by_symbol.items():
        lines.append(
            f"  {symbol}: {stats['count']} trades, "
            f"slip={stats['avg_slippage_pct']:.4%}, "
            f"fill={stats['avg_fill_rate']:.1%}"
        )

    lines.extend([
        "",
        "--- RECOMMENDATIONS ---",
    ])
    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"  {i}. {rec}")

    lines.append("=" * 60)
    return "\n".join(lines)
