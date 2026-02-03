"""
Automated Backtest Pipeline for MASP.

Provides automated backtesting capabilities:
- Scheduled strategy validation
- Parameter optimization
- Performance comparison
- Report generation
- Alerting on performance degradation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from libs.backtest.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


class BacktestStatus(Enum):
    """Backtest execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestJob:
    """Backtest job configuration."""

    job_id: str
    strategy_name: str
    exchange: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10_000_000

    # Execution state
    status: BacktestStatus = BacktestStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Results
    result: Optional[BacktestResult] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "strategy_name": self.strategy_name,
            "exchange": self.exchange,
            "symbols": self.symbols,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


@dataclass
class BacktestSchedule:
    """Scheduled backtest configuration."""

    schedule_id: str
    strategy_name: str
    exchange: str
    symbols: List[str]

    # Schedule configuration
    frequency: str = "daily"  # daily, weekly, monthly
    time_of_day: str = "00:00"  # HH:MM
    timezone: str = "Asia/Seoul"

    # Backtest parameters
    lookback_days: int = 90
    initial_capital: float = 10_000_000

    # Alerting
    min_sharpe_ratio: float = 0.5
    max_drawdown_pct: float = 15.0
    alert_on_degradation: bool = True

    enabled: bool = True


@dataclass
class BacktestComparison:
    """Comparison between two backtest results."""

    baseline_job_id: str
    comparison_job_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Metric changes (as percentages)
    sharpe_change_pct: float = 0.0
    drawdown_change_pct: float = 0.0
    win_rate_change_pct: float = 0.0
    total_pnl_change_pct: float = 0.0

    # Assessment
    is_improvement: bool = False
    significant_degradation: bool = False
    recommendations: List[str] = field(default_factory=list)


class BacktestResultStore:
    """
    Stores backtest results for historical analysis.
    """

    def __init__(self, storage_dir: str = "storage/backtests"):
        """
        Initialize result store.

        Args:
            storage_dir: Directory for storing results
        """
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._results_file = self._storage_dir / "results.jsonl"
        self._lock = threading.Lock()

    def save_result(self, job: BacktestJob) -> bool:
        """
        Save backtest result.

        Args:
            job: Completed backtest job

        Returns:
            True if save was successful
        """
        with self._lock:
            try:
                data = job.to_dict()
                if job.result:
                    data["result"] = asdict(job.result)

                line = json.dumps(data, default=str) + "\n"

                with open(self._results_file, "a", encoding="utf-8") as f:
                    f.write(line)
                return True
            except Exception as e:
                logger.error("[BacktestStore] Save failed: %s", e)
                return False

    def get_results(
        self,
        strategy_name: Optional[str] = None,
        exchange: Optional[str] = None,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get historical results.

        Args:
            strategy_name: Filter by strategy
            exchange: Filter by exchange
            days: Number of days of history

        Returns:
            List of result dictionaries
        """
        if not self._results_file.exists():
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        results = []

        with self._lock:
            try:
                with open(self._results_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)

                            # Date filter
                            created_str = data.get("created_at", "")
                            if created_str:
                                created = datetime.fromisoformat(
                                    created_str.replace("Z", "+00:00")
                                )
                                if created < cutoff:
                                    continue

                            # Strategy filter
                            if (
                                strategy_name
                                and data.get("strategy_name") != strategy_name
                            ):
                                continue

                            # Exchange filter
                            if exchange and data.get("exchange") != exchange:
                                continue

                            results.append(data)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error("[BacktestStore] Read failed: %s", e)

        return sorted(results, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_latest_result(
        self,
        strategy_name: str,
        exchange: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest result for a strategy/exchange combination.

        Args:
            strategy_name: Strategy name
            exchange: Exchange name

        Returns:
            Latest result or None
        """
        results = self.get_results(
            strategy_name=strategy_name,
            exchange=exchange,
            days=90,
        )
        return results[0] if results else None


class BacktestPipeline:
    """
    Automated backtest pipeline.

    Manages scheduled backtests, stores results, and generates alerts.

    Example:
        pipeline = BacktestPipeline()

        # Add a scheduled backtest
        pipeline.add_schedule(BacktestSchedule(
            schedule_id="upbit_kama",
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW", "ETH/KRW"],
            frequency="daily",
            time_of_day="06:00",
        ))

        # Run a one-time backtest
        job = pipeline.run_backtest(
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
            lookback_days=90,
        )

        # Get results
        results = pipeline.get_results(strategy_name="KAMA-TSMOM")
    """

    def __init__(
        self,
        storage_dir: str = "storage/backtests",
        alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize backtest pipeline.

        Args:
            storage_dir: Directory for storing results
            alert_callback: Function to call for alerts
        """
        self._store = BacktestResultStore(storage_dir=storage_dir)
        self._schedules: Dict[str, BacktestSchedule] = {}
        self._jobs: Dict[str, BacktestJob] = {}
        self._alert_callback = alert_callback
        self._lock = threading.Lock()

        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None

    def add_schedule(self, schedule: BacktestSchedule) -> None:
        """
        Add a scheduled backtest.

        Args:
            schedule: Backtest schedule configuration
        """
        with self._lock:
            self._schedules[schedule.schedule_id] = schedule
        logger.info(
            "[Pipeline] Schedule added: %s (%s)",
            schedule.schedule_id,
            schedule.frequency,
        )

    def remove_schedule(self, schedule_id: str) -> bool:
        """
        Remove a scheduled backtest.

        Args:
            schedule_id: Schedule ID to remove

        Returns:
            True if schedule was found and removed
        """
        with self._lock:
            if schedule_id in self._schedules:
                del self._schedules[schedule_id]
                return True
            return False

    def get_schedules(self) -> List[BacktestSchedule]:
        """Get all schedules."""
        with self._lock:
            return list(self._schedules.values())

    def run_backtest(
        self,
        strategy_name: str,
        exchange: str,
        symbols: List[str],
        lookback_days: int = 90,
        initial_capital: float = 10_000_000,
        data_provider: Optional[Callable] = None,
    ) -> BacktestJob:
        """
        Run a backtest job.

        Args:
            strategy_name: Strategy to test
            exchange: Exchange for data
            symbols: Symbols to test
            lookback_days: Days of historical data
            initial_capital: Starting capital
            data_provider: Optional function to provide price data

        Returns:
            BacktestJob with results
        """
        job_id = f"bt_{int(datetime.now().timestamp())}_{strategy_name}"

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)

        job = BacktestJob(
            job_id=job_id,
            strategy_name=strategy_name,
            exchange=exchange,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
        )

        with self._lock:
            self._jobs[job_id] = job

        # Execute backtest
        self._execute_job(job, data_provider)

        return job

    def _execute_job(
        self,
        job: BacktestJob,
        data_provider: Optional[Callable] = None,
    ) -> None:
        """Execute a backtest job."""
        job.status = BacktestStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)

        try:
            engine = BacktestEngine(initial_capital=job.initial_capital)

            # Generate simulated data if no provider
            if data_provider is None:
                signals, prices = self._generate_mock_data(job)
            else:
                signals, prices = data_provider(
                    job.symbols,
                    job.start_date,
                    job.end_date,
                )

            # Run backtest
            result = engine.run_simple(signals, prices)

            job.result = result
            job.status = BacktestStatus.COMPLETED
            job.metrics = {
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown_pct,
                "win_rate": result.win_rate,
                "total_pnl_pct": result.total_pnl_pct,
                "profit_factor": result.profit_factor,
                "total_trades": result.total_trades,
            }

            logger.info(
                "[Pipeline] Backtest completed: %s, Sharpe=%.2f, MDD=%.1f%%",
                job.job_id,
                result.sharpe_ratio,
                result.max_drawdown_pct,
            )

        except Exception as e:
            job.status = BacktestStatus.FAILED
            job.error_message = str(e)
            logger.error("[Pipeline] Backtest failed: %s - %s", job.job_id, e)

        job.completed_at = datetime.now(timezone.utc)

        # Save result
        self._store.save_result(job)

        # Check for alerts
        self._check_alerts(job)

    def _generate_mock_data(
        self,
        job: BacktestJob,
    ) -> Tuple[List[str], List[float]]:
        """
        Generate mock data for testing.

        In production, this should be replaced with actual data fetching.
        """
        import random

        days = (job.end_date - job.start_date).days
        signals = []
        prices = []

        price = 50_000_000  # Starting price

        for _ in range(days):
            # Random walk with trend
            change = random.gauss(0.001, 0.02)  # 0.1% drift, 2% volatility
            price *= 1 + change
            prices.append(price)

            # Simple signal based on momentum
            if len(prices) > 5:
                momentum = (prices[-1] - prices[-5]) / prices[-5]
                if momentum > 0.03:
                    signals.append("BUY")
                elif momentum < -0.03:
                    signals.append("SELL")
                else:
                    signals.append("HOLD")
            else:
                signals.append("HOLD")

        return signals, prices

    def _check_alerts(self, job: BacktestJob) -> None:
        """Check if alerts should be triggered."""
        if not job.result or not self._alert_callback:
            return

        # Get schedule for this job
        schedule = None
        for s in self._schedules.values():
            if s.strategy_name == job.strategy_name and s.exchange == job.exchange:
                schedule = s
                break

        if not schedule or not schedule.alert_on_degradation:
            return

        alerts = []

        # Check Sharpe ratio
        if job.result.sharpe_ratio < schedule.min_sharpe_ratio:
            alerts.append(
                f"Sharpe ratio ({job.result.sharpe_ratio:.2f}) below threshold ({schedule.min_sharpe_ratio})"
            )

        # Check drawdown
        if job.result.max_drawdown_pct > schedule.max_drawdown_pct:
            alerts.append(
                f"Max drawdown ({job.result.max_drawdown_pct:.1f}%) exceeds threshold ({schedule.max_drawdown_pct}%)"
            )

        if alerts:
            alert_data = {
                "job_id": job.job_id,
                "strategy_name": job.strategy_name,
                "exchange": job.exchange,
                "alerts": alerts,
                "metrics": job.metrics,
            }
            try:
                self._alert_callback("backtest_degradation", alert_data)
            except Exception as e:
                logger.error("[Pipeline] Alert callback failed: %s", e)

    def compare_results(
        self,
        baseline_job_id: str,
        comparison_job_id: str,
    ) -> Optional[BacktestComparison]:
        """
        Compare two backtest results.

        Args:
            baseline_job_id: Baseline job ID
            comparison_job_id: Comparison job ID

        Returns:
            BacktestComparison or None if jobs not found
        """
        with self._lock:
            baseline = self._jobs.get(baseline_job_id)
            comparison = self._jobs.get(comparison_job_id)

        if not baseline or not comparison:
            return None

        if not baseline.result or not comparison.result:
            return None

        base_r = baseline.result
        comp_r = comparison.result

        # Calculate changes
        sharpe_change = 0.0
        if base_r.sharpe_ratio != 0:
            sharpe_change = (
                (comp_r.sharpe_ratio - base_r.sharpe_ratio)
                / abs(base_r.sharpe_ratio)
                * 100
            )

        dd_change = 0.0
        if base_r.max_drawdown_pct != 0:
            dd_change = (
                (comp_r.max_drawdown_pct - base_r.max_drawdown_pct)
                / base_r.max_drawdown_pct
                * 100
            )

        win_rate_change = comp_r.win_rate - base_r.win_rate

        pnl_change = 0.0
        if base_r.total_pnl_pct != 0:
            pnl_change = (
                (comp_r.total_pnl_pct - base_r.total_pnl_pct)
                / abs(base_r.total_pnl_pct)
                * 100
            )

        # Assess changes
        is_improvement = (
            comp_r.sharpe_ratio >= base_r.sharpe_ratio
            and comp_r.max_drawdown_pct <= base_r.max_drawdown_pct
        )

        significant_degradation = sharpe_change < -20 or dd_change > 30

        recommendations = []
        if sharpe_change < -10:
            recommendations.append("Consider reviewing strategy parameters")
        if dd_change > 20:
            recommendations.append("Evaluate position sizing rules")
        if win_rate_change < -5:
            recommendations.append("Review entry criteria")

        return BacktestComparison(
            baseline_job_id=baseline_job_id,
            comparison_job_id=comparison_job_id,
            sharpe_change_pct=sharpe_change,
            drawdown_change_pct=dd_change,
            win_rate_change_pct=win_rate_change,
            total_pnl_change_pct=pnl_change,
            is_improvement=is_improvement,
            significant_degradation=significant_degradation,
            recommendations=recommendations,
        )

    def get_job(self, job_id: str) -> Optional[BacktestJob]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_results(
        self,
        strategy_name: Optional[str] = None,
        exchange: Optional[str] = None,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get historical results from store."""
        return self._store.get_results(
            strategy_name=strategy_name,
            exchange=exchange,
            days=days,
        )

    def get_performance_trend(
        self,
        strategy_name: str,
        exchange: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get performance trend over time.

        Args:
            strategy_name: Strategy name
            exchange: Exchange name
            days: Number of days

        Returns:
            Trend analysis dictionary
        """
        results = self._store.get_results(
            strategy_name=strategy_name,
            exchange=exchange,
            days=days,
        )

        if not results:
            return {"trend": "unknown", "data_points": 0}

        sharpe_values = [
            r.get("metrics", {}).get("sharpe_ratio", 0)
            for r in results
            if r.get("metrics")
        ]

        if len(sharpe_values) < 2:
            return {"trend": "insufficient_data", "data_points": len(sharpe_values)}

        # Simple trend analysis
        first_half = sharpe_values[len(sharpe_values) // 2 :]  # Older
        second_half = sharpe_values[: len(sharpe_values) // 2]  # Newer

        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0

        if second_avg > first_avg * 1.1:
            trend = "improving"
        elif second_avg < first_avg * 0.9:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "data_points": len(sharpe_values),
            "current_sharpe": sharpe_values[0] if sharpe_values else 0,
            "average_sharpe": sum(sharpe_values) / len(sharpe_values),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline summary."""
        with self._lock:
            jobs_by_status = {}
            for job in self._jobs.values():
                status = job.status.value
                jobs_by_status[status] = jobs_by_status.get(status, 0) + 1

        return {
            "schedules_count": len(self._schedules),
            "jobs_count": len(self._jobs),
            "jobs_by_status": jobs_by_status,
            "is_running": self._running,
        }


# ============================================================================
# Convenience functions
# ============================================================================

_pipeline_instance: Optional[BacktestPipeline] = None


def get_backtest_pipeline() -> BacktestPipeline:
    """Get global backtest pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = BacktestPipeline()
    return _pipeline_instance


def run_backtest(
    strategy_name: str,
    exchange: str,
    symbols: List[str],
    lookback_days: int = 90,
) -> BacktestJob:
    """Convenience function to run a backtest."""
    return get_backtest_pipeline().run_backtest(
        strategy_name=strategy_name,
        exchange=exchange,
        symbols=symbols,
        lookback_days=lookback_days,
    )
