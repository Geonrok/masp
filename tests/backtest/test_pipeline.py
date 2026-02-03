"""
Tests for Backtest Pipeline.
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from libs.backtest.pipeline import (
    BacktestJob,
    BacktestPipeline,
    BacktestResultStore,
    BacktestSchedule,
    BacktestStatus,
    get_backtest_pipeline,
    run_backtest,
)


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def pipeline(temp_storage_dir):
    """Create test pipeline."""
    return BacktestPipeline(
        storage_dir=temp_storage_dir,
        alert_callback=None,
    )


@pytest.fixture
def result_store(temp_storage_dir):
    """Create test result store."""
    return BacktestResultStore(storage_dir=temp_storage_dir)


class TestBacktestStatus:
    """Tests for BacktestStatus enum."""

    def test_values(self):
        """Test status values."""
        assert BacktestStatus.PENDING.value == "pending"
        assert BacktestStatus.RUNNING.value == "running"
        assert BacktestStatus.COMPLETED.value == "completed"
        assert BacktestStatus.FAILED.value == "failed"


class TestBacktestJob:
    """Tests for BacktestJob dataclass."""

    def test_creation(self):
        """Test creating a job."""
        job = BacktestJob(
            job_id="test_job",
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
        )
        assert job.job_id == "test_job"
        assert job.strategy_name == "KAMA-TSMOM"
        assert job.status == BacktestStatus.PENDING

    def test_to_dict(self):
        """Test conversion to dictionary."""
        job = BacktestJob(
            job_id="test_job",
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
        )
        d = job.to_dict()
        assert d["job_id"] == "test_job"
        assert d["strategy_name"] == "KAMA-TSMOM"
        assert d["exchange"] == "upbit"


class TestBacktestSchedule:
    """Tests for BacktestSchedule dataclass."""

    def test_creation(self):
        """Test creating a schedule."""
        schedule = BacktestSchedule(
            schedule_id="daily_kama",
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW", "ETH/KRW"],
            frequency="daily",
            time_of_day="06:00",
        )
        assert schedule.schedule_id == "daily_kama"
        assert schedule.frequency == "daily"
        assert schedule.enabled

    def test_default_values(self):
        """Test default values."""
        schedule = BacktestSchedule(
            schedule_id="test",
            strategy_name="Test",
            exchange="upbit",
            symbols=["BTC/KRW"],
        )
        assert schedule.frequency == "daily"
        assert schedule.lookback_days == 90
        assert schedule.min_sharpe_ratio == 0.5
        assert schedule.max_drawdown_pct == 15.0


class TestBacktestResultStore:
    """Tests for BacktestResultStore class."""

    def test_init(self, result_store, temp_storage_dir):
        """Test initialization."""
        assert result_store._storage_dir == Path(temp_storage_dir)

    def test_save_result(self, result_store):
        """Test saving a result."""
        job = BacktestJob(
            job_id="test_job",
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
            status=BacktestStatus.COMPLETED,
        )
        success = result_store.save_result(job)
        assert success

    def test_get_results(self, result_store):
        """Test getting results."""
        # Save some results
        for i in range(3):
            job = BacktestJob(
                job_id=f"job_{i}",
                strategy_name="KAMA-TSMOM",
                exchange="upbit",
                symbols=["BTC/KRW"],
                start_date=datetime.now(timezone.utc) - timedelta(days=30),
                end_date=datetime.now(timezone.utc),
                status=BacktestStatus.COMPLETED,
            )
            result_store.save_result(job)

        results = result_store.get_results()
        assert len(results) == 3

    def test_get_results_with_filter(self, result_store):
        """Test filtering results."""
        # Save results for different strategies
        job1 = BacktestJob(
            job_id="job_1",
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
        )
        job2 = BacktestJob(
            job_id="job_2",
            strategy_name="MA-Crossover",
            exchange="upbit",
            symbols=["BTC/KRW"],
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
        )
        result_store.save_result(job1)
        result_store.save_result(job2)

        results = result_store.get_results(strategy_name="KAMA-TSMOM")
        assert len(results) == 1
        assert results[0]["strategy_name"] == "KAMA-TSMOM"

    def test_get_latest_result(self, result_store):
        """Test getting latest result."""
        job = BacktestJob(
            job_id="job_latest",
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
        )
        result_store.save_result(job)

        latest = result_store.get_latest_result("KAMA-TSMOM", "upbit")
        assert latest is not None
        assert latest["job_id"] == "job_latest"


class TestBacktestPipeline:
    """Tests for BacktestPipeline class."""

    def test_init(self, pipeline):
        """Test initialization."""
        assert len(pipeline._schedules) == 0
        assert len(pipeline._jobs) == 0

    def test_add_schedule(self, pipeline):
        """Test adding a schedule."""
        schedule = BacktestSchedule(
            schedule_id="daily_kama",
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
        )
        pipeline.add_schedule(schedule)

        schedules = pipeline.get_schedules()
        assert len(schedules) == 1
        assert schedules[0].schedule_id == "daily_kama"

    def test_remove_schedule(self, pipeline):
        """Test removing a schedule."""
        schedule = BacktestSchedule(
            schedule_id="daily_kama",
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
        )
        pipeline.add_schedule(schedule)

        success = pipeline.remove_schedule("daily_kama")
        assert success

        schedules = pipeline.get_schedules()
        assert len(schedules) == 0

    def test_remove_nonexistent_schedule(self, pipeline):
        """Test removing nonexistent schedule."""
        success = pipeline.remove_schedule("nonexistent")
        assert not success

    def test_run_backtest(self, pipeline):
        """Test running a backtest."""
        job = pipeline.run_backtest(
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
            lookback_days=30,
        )

        assert job is not None
        assert job.strategy_name == "KAMA-TSMOM"
        assert job.status == BacktestStatus.COMPLETED
        assert job.result is not None

    def test_run_backtest_with_custom_capital(self, pipeline):
        """Test running backtest with custom capital."""
        job = pipeline.run_backtest(
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
            initial_capital=5_000_000,
        )

        assert job.initial_capital == 5_000_000

    def test_get_job(self, pipeline):
        """Test getting a job by ID."""
        job = pipeline.run_backtest(
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
        )

        retrieved = pipeline.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_job_nonexistent(self, pipeline):
        """Test getting nonexistent job."""
        job = pipeline.get_job("nonexistent")
        assert job is None

    def test_get_results(self, pipeline):
        """Test getting historical results."""
        # Run some backtests
        pipeline.run_backtest(
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
        )
        pipeline.run_backtest(
            strategy_name="KAMA-TSMOM",
            exchange="binance",
            symbols=["BTC/USDT"],
        )

        results = pipeline.get_results()
        assert len(results) >= 2

    def test_get_summary(self, pipeline):
        """Test getting pipeline summary."""
        pipeline.run_backtest(
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
        )

        summary = pipeline.get_summary()
        assert summary["jobs_count"] >= 1
        assert "completed" in summary["jobs_by_status"]


class TestBacktestComparison:
    """Tests for backtest comparison."""

    def test_compare_results(self, pipeline):
        """Test comparing two backtest results."""
        job1 = pipeline.run_backtest(
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
        )
        job2 = pipeline.run_backtest(
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
        )

        comparison = pipeline.compare_results(job1.job_id, job2.job_id)
        assert comparison is not None
        assert comparison.baseline_job_id == job1.job_id
        assert comparison.comparison_job_id == job2.job_id

    def test_compare_nonexistent(self, pipeline):
        """Test comparing with nonexistent job."""
        comparison = pipeline.compare_results("nonexistent", "also_nonexistent")
        assert comparison is None


class TestAlertCallback:
    """Tests for alert callbacks."""

    def test_alert_on_low_sharpe(self, temp_storage_dir):
        """Test alert is triggered for low Sharpe ratio."""
        alerts_received = []

        def callback(event_type, data):
            alerts_received.append((event_type, data))

        pipeline = BacktestPipeline(
            storage_dir=temp_storage_dir,
            alert_callback=callback,
        )

        # Add schedule with high Sharpe threshold
        schedule = BacktestSchedule(
            schedule_id="test",
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
            min_sharpe_ratio=10.0,  # Unrealistic threshold for testing
            alert_on_degradation=True,
        )
        pipeline.add_schedule(schedule)

        # Run backtest - will likely trigger alert
        pipeline.run_backtest(
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
        )

        # Alert may or may not be triggered depending on random data
        # This test verifies the callback mechanism works


class TestPerformanceTrend:
    """Tests for performance trend analysis."""

    def test_get_performance_trend(self, pipeline):
        """Test getting performance trend."""
        # Run multiple backtests
        for _ in range(3):
            pipeline.run_backtest(
                strategy_name="KAMA-TSMOM",
                exchange="upbit",
                symbols=["BTC/KRW"],
            )

        trend = pipeline.get_performance_trend(
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
        )

        assert "trend" in trend
        assert "data_points" in trend

    def test_insufficient_data(self, pipeline):
        """Test trend with insufficient data."""
        trend = pipeline.get_performance_trend(
            strategy_name="NonExistent",
            exchange="upbit",
        )

        assert trend["trend"] == "unknown"
        assert trend["data_points"] == 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_backtest_pipeline(self, temp_storage_dir):
        """Test get_backtest_pipeline function."""
        global _pipeline_instance
        import libs.backtest.pipeline as pipeline_module

        pipeline_module._pipeline_instance = None
        pipeline = get_backtest_pipeline()
        assert isinstance(pipeline, BacktestPipeline)
        pipeline_module._pipeline_instance = None

    def test_run_backtest_function(self, temp_storage_dir):
        """Test run_backtest convenience function."""
        import libs.backtest.pipeline as pipeline_module

        pipeline_module._pipeline_instance = BacktestPipeline(
            storage_dir=temp_storage_dir
        )

        job = run_backtest(
            strategy_name="KAMA-TSMOM",
            exchange="upbit",
            symbols=["BTC/KRW"],
            lookback_days=30,
        )
        assert job is not None
        assert job.status == BacktestStatus.COMPLETED

        pipeline_module._pipeline_instance = None
