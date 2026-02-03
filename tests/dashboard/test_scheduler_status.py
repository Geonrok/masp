"""Tests for scheduler status component."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def test_import_scheduler_status():
    """Test module imports correctly."""
    from services.dashboard.components import scheduler_status

    assert hasattr(scheduler_status, "render_scheduler_status")
    assert hasattr(scheduler_status, "ScheduledJob")
    assert hasattr(scheduler_status, "JobStatus")
    assert hasattr(scheduler_status, "JobType")
    assert hasattr(scheduler_status, "JobSchedule")
    assert hasattr(scheduler_status, "JobExecution")


def test_job_status_enum():
    """Test JobStatus enum values."""
    from services.dashboard.components.scheduler_status import JobStatus

    assert JobStatus.RUNNING.value == "RUNNING"
    assert JobStatus.SCHEDULED.value == "SCHEDULED"
    assert JobStatus.COMPLETED.value == "COMPLETED"
    assert JobStatus.FAILED.value == "FAILED"
    assert JobStatus.PAUSED.value == "PAUSED"


def test_job_type_enum():
    """Test JobType enum values."""
    from services.dashboard.components.scheduler_status import JobType

    assert JobType.STRATEGY.value == "STRATEGY"
    assert JobType.DATA_FETCH.value == "DATA_FETCH"
    assert JobType.CLEANUP.value == "CLEANUP"
    assert JobType.REPORT.value == "REPORT"
    assert JobType.BACKUP.value == "BACKUP"


def test_job_schedule_dataclass():
    """Test JobSchedule dataclass."""
    from services.dashboard.components.scheduler_status import JobSchedule

    schedule = JobSchedule(
        cron_expression="*/5 * * * *",
        interval_seconds=300,
        next_run=datetime(2026, 1, 1, 12, 5, 0),
    )

    assert schedule.cron_expression == "*/5 * * * *"
    assert schedule.interval_seconds == 300
    assert schedule.next_run is not None


def test_job_schedule_defaults():
    """Test JobSchedule dataclass defaults."""
    from services.dashboard.components.scheduler_status import JobSchedule

    schedule = JobSchedule()

    assert schedule.cron_expression is None
    assert schedule.interval_seconds is None
    assert schedule.next_run is None


def test_job_execution_dataclass():
    """Test JobExecution dataclass."""
    from services.dashboard.components.scheduler_status import JobExecution, JobStatus

    execution = JobExecution(
        started_at=datetime(2026, 1, 1, 12, 0, 0),
        finished_at=datetime(2026, 1, 1, 12, 0, 5),
        status=JobStatus.COMPLETED,
        duration_ms=5000,
    )

    assert execution.started_at is not None
    assert execution.finished_at is not None
    assert execution.status == JobStatus.COMPLETED
    assert execution.duration_ms == 5000


def test_job_execution_with_error():
    """Test JobExecution dataclass with error."""
    from services.dashboard.components.scheduler_status import JobExecution, JobStatus

    execution = JobExecution(
        started_at=datetime(2026, 1, 1, 12, 0, 0),
        finished_at=datetime(2026, 1, 1, 12, 0, 1),
        status=JobStatus.FAILED,
        duration_ms=1000,
        error_message="Connection timeout",
    )

    assert execution.status == JobStatus.FAILED
    assert execution.error_message == "Connection timeout"


def test_scheduled_job_dataclass():
    """Test ScheduledJob dataclass."""
    from services.dashboard.components.scheduler_status import (
        ScheduledJob,
        JobType,
        JobStatus,
        JobSchedule,
    )

    job = ScheduledJob(
        job_id="test_001",
        name="Test Job",
        job_type=JobType.STRATEGY,
        status=JobStatus.SCHEDULED,
    )

    assert job.job_id == "test_001"
    assert job.name == "Test Job"
    assert job.job_type == JobType.STRATEGY
    assert job.status == JobStatus.SCHEDULED
    assert job.is_enabled is True
    assert job.execution_count == 0


def test_scheduled_job_with_all_fields():
    """Test ScheduledJob dataclass with all fields."""
    from services.dashboard.components.scheduler_status import (
        ScheduledJob,
        JobType,
        JobStatus,
        JobSchedule,
        JobExecution,
    )

    job = ScheduledJob(
        job_id="test_001",
        name="Full Job",
        job_type=JobType.DATA_FETCH,
        status=JobStatus.RUNNING,
        schedule=JobSchedule(interval_seconds=60),
        last_execution=JobExecution(
            started_at=datetime.now(),
            status=JobStatus.RUNNING,
        ),
        execution_count=100,
        failure_count=5,
        is_enabled=True,
        description="Test description",
    )

    assert job.execution_count == 100
    assert job.failure_count == 5
    assert job.description == "Test description"


def test_key_function():
    """Test _key generates namespaced keys."""
    from services.dashboard.components.scheduler_status import _key, _KEY_PREFIX

    result = _key("job_type")
    assert result == f"{_KEY_PREFIX}job_type"
    assert result == "scheduler_status.job_type"


def test_get_status_indicator():
    """Test _get_status_indicator returns correct indicators."""
    from services.dashboard.components.scheduler_status import (
        _get_status_indicator,
        JobStatus,
    )

    assert _get_status_indicator(JobStatus.RUNNING) == "[RUN]"
    assert _get_status_indicator(JobStatus.SCHEDULED) == "[SCH]"
    assert _get_status_indicator(JobStatus.COMPLETED) == "[OK]"
    assert _get_status_indicator(JobStatus.FAILED) == "[ERR]"
    assert _get_status_indicator(JobStatus.PAUSED) == "[PSE]"


def test_get_status_color():
    """Test _get_status_color returns correct colors."""
    from services.dashboard.components.scheduler_status import (
        _get_status_color,
        JobStatus,
    )

    assert _get_status_color(JobStatus.RUNNING) == "blue"
    assert _get_status_color(JobStatus.SCHEDULED) == "gray"
    assert _get_status_color(JobStatus.COMPLETED) == "green"
    assert _get_status_color(JobStatus.FAILED) == "red"
    assert _get_status_color(JobStatus.PAUSED) == "orange"


def test_get_job_type_indicator():
    """Test _get_job_type_indicator returns correct indicators."""
    from services.dashboard.components.scheduler_status import (
        _get_job_type_indicator,
        JobType,
    )

    assert _get_job_type_indicator(JobType.STRATEGY) == "[STR]"
    assert _get_job_type_indicator(JobType.DATA_FETCH) == "[DAT]"
    assert _get_job_type_indicator(JobType.CLEANUP) == "[CLN]"
    assert _get_job_type_indicator(JobType.REPORT) == "[RPT]"
    assert _get_job_type_indicator(JobType.BACKUP) == "[BAK]"


def test_format_datetime():
    """Test _format_datetime formats correctly."""
    from services.dashboard.components.scheduler_status import _format_datetime

    dt = datetime(2026, 1, 15, 14, 30, 45)
    result = _format_datetime(dt)
    assert result == "2026-01-15 14:30:45"


def test_format_datetime_none():
    """Test _format_datetime with None."""
    from services.dashboard.components.scheduler_status import _format_datetime

    result = _format_datetime(None)
    assert result == "-"


def test_format_relative_time_past():
    """Test _format_relative_time for past times."""
    from services.dashboard.components.scheduler_status import _format_relative_time

    ref = datetime(2026, 1, 15, 12, 0, 0)

    assert (
        _format_relative_time(ref - timedelta(seconds=30), reference=ref) == "30s ago"
    )
    assert _format_relative_time(ref - timedelta(minutes=5), reference=ref) == "5m ago"
    assert _format_relative_time(ref - timedelta(hours=3), reference=ref) == "3h ago"
    assert _format_relative_time(ref - timedelta(days=2), reference=ref) == "2d ago"


def test_format_relative_time_future():
    """Test _format_relative_time for future times."""
    from services.dashboard.components.scheduler_status import _format_relative_time

    ref = datetime(2026, 1, 15, 12, 0, 0)

    assert _format_relative_time(ref + timedelta(seconds=30), reference=ref) == "in 30s"
    assert _format_relative_time(ref + timedelta(minutes=5), reference=ref) == "in 5m"
    assert _format_relative_time(ref + timedelta(hours=3), reference=ref) == "in 3h"
    assert _format_relative_time(ref + timedelta(days=2), reference=ref) == "in 2d"


def test_format_relative_time_none():
    """Test _format_relative_time with None."""
    from services.dashboard.components.scheduler_status import _format_relative_time

    assert _format_relative_time(None) == "-"


def test_format_relative_time_timezone_aware():
    """Test _format_relative_time handles timezone-aware datetimes."""
    from services.dashboard.components.scheduler_status import _format_relative_time

    dt_aware = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    ref_aware = datetime(2026, 1, 15, 12, 5, 0, tzinfo=timezone.utc)

    result = _format_relative_time(dt_aware, reference=ref_aware)
    assert result == "5m ago"


def test_format_relative_time_mixed_timezone():
    """Test _format_relative_time handles mixed tz-awareness."""
    from services.dashboard.components.scheduler_status import _format_relative_time

    dt_aware = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    ref_naive = datetime(2026, 1, 15, 12, 5, 0)

    # Should not raise TypeError
    result = _format_relative_time(dt_aware, reference=ref_naive)
    assert "ago" in result or "in" in result


def test_format_duration():
    """Test _format_duration formats correctly."""
    from services.dashboard.components.scheduler_status import _format_duration

    assert _format_duration(500) == "500ms"
    assert _format_duration(1500) == "1.5s"
    assert _format_duration(65000) == "1m 5s"
    assert _format_duration(None) == "-"


def test_format_interval():
    """Test _format_interval formats correctly."""
    from services.dashboard.components.scheduler_status import _format_interval

    assert _format_interval(30) == "Every 30s"
    assert _format_interval(300) == "Every 5m"
    assert _format_interval(7200) == "Every 2h"
    assert _format_interval(172800) == "Every 2d"
    assert _format_interval(None) == "-"


def test_calculate_success_rate():
    """Test _calculate_success_rate calculates correctly."""
    from services.dashboard.components.scheduler_status import _calculate_success_rate

    assert _calculate_success_rate(100, 10) == 90.0
    assert _calculate_success_rate(100, 0) == 100.0
    assert _calculate_success_rate(100, 100) == 0.0
    assert _calculate_success_rate(0, 0) == 0.0


def test_get_demo_jobs():
    """Test _get_demo_jobs returns expected data."""
    from services.dashboard.components.scheduler_status import _get_demo_jobs

    jobs = _get_demo_jobs()

    assert len(jobs) >= 3
    assert all(job.job_id != "" for job in jobs)
    assert all(job.name != "" for job in jobs)


def test_get_demo_jobs_deterministic():
    """Test _get_demo_jobs returns consistent data."""
    from services.dashboard.components.scheduler_status import _get_demo_jobs

    jobs1 = _get_demo_jobs()
    jobs2 = _get_demo_jobs()

    assert len(jobs1) == len(jobs2)
    assert jobs1[0].job_id == jobs2[0].job_id
    assert jobs1[0].name == jobs2[0].name


def test_filter_jobs_by_type():
    """Test _filter_jobs_by_type filters correctly."""
    from services.dashboard.components.scheduler_status import (
        _filter_jobs_by_type,
        ScheduledJob,
        JobType,
    )

    jobs = [
        ScheduledJob(job_id="1", name="Strategy 1", job_type=JobType.STRATEGY),
        ScheduledJob(job_id="2", name="Data Fetch", job_type=JobType.DATA_FETCH),
        ScheduledJob(job_id="3", name="Strategy 2", job_type=JobType.STRATEGY),
    ]

    result = _filter_jobs_by_type(jobs, JobType.STRATEGY)
    assert len(result) == 2
    assert all(job.job_type == JobType.STRATEGY for job in result)


def test_filter_jobs_by_type_none():
    """Test _filter_jobs_by_type with None returns all."""
    from services.dashboard.components.scheduler_status import (
        _filter_jobs_by_type,
        ScheduledJob,
        JobType,
    )

    jobs = [
        ScheduledJob(job_id="1", name="Job 1", job_type=JobType.STRATEGY),
        ScheduledJob(job_id="2", name="Job 2", job_type=JobType.DATA_FETCH),
    ]

    result = _filter_jobs_by_type(jobs, None)
    assert len(result) == 2


def test_filter_jobs_by_status():
    """Test _filter_jobs_by_status filters correctly."""
    from services.dashboard.components.scheduler_status import (
        _filter_jobs_by_status,
        ScheduledJob,
        JobType,
        JobStatus,
    )

    jobs = [
        ScheduledJob(
            job_id="1",
            name="Running",
            job_type=JobType.STRATEGY,
            status=JobStatus.RUNNING,
        ),
        ScheduledJob(
            job_id="2",
            name="Failed",
            job_type=JobType.STRATEGY,
            status=JobStatus.FAILED,
        ),
        ScheduledJob(
            job_id="3",
            name="Scheduled",
            job_type=JobType.STRATEGY,
            status=JobStatus.SCHEDULED,
        ),
    ]

    result = _filter_jobs_by_status(jobs, [JobStatus.RUNNING, JobStatus.FAILED])
    assert len(result) == 2


def test_filter_jobs_by_status_empty():
    """Test _filter_jobs_by_status with empty list returns none."""
    from services.dashboard.components.scheduler_status import (
        _filter_jobs_by_status,
        ScheduledJob,
        JobType,
        JobStatus,
    )

    jobs = [
        ScheduledJob(
            job_id="1", name="Job", job_type=JobType.STRATEGY, status=JobStatus.RUNNING
        ),
    ]

    result = _filter_jobs_by_status(jobs, [])
    assert len(result) == 0


def test_get_scheduler_summary():
    """Test _get_scheduler_summary calculates correctly."""
    from services.dashboard.components.scheduler_status import (
        _get_scheduler_summary,
        ScheduledJob,
        JobType,
        JobStatus,
    )

    jobs = [
        ScheduledJob(
            job_id="1",
            name="Job 1",
            job_type=JobType.STRATEGY,
            status=JobStatus.RUNNING,
            execution_count=100,
            failure_count=5,
        ),
        ScheduledJob(
            job_id="2",
            name="Job 2",
            job_type=JobType.DATA_FETCH,
            status=JobStatus.FAILED,
            execution_count=50,
            failure_count=10,
        ),
    ]

    summary = _get_scheduler_summary(jobs)

    assert summary["total_jobs"] == 2
    assert summary["running"] == 1
    assert summary["failed"] == 1
    assert summary["total_executions"] == 150
    assert summary["total_failures"] == 15
    assert summary["overall_success_rate"] == 90.0


def test_get_scheduler_summary_empty():
    """Test _get_scheduler_summary with empty list."""
    from services.dashboard.components.scheduler_status import _get_scheduler_summary

    summary = _get_scheduler_summary([])

    assert summary["total_jobs"] == 0
    assert summary["running"] == 0
    assert summary["overall_success_rate"] == 0.0


def test_get_scheduler_export_data():
    """Test get_scheduler_export_data returns valid structure."""
    from services.dashboard.components.scheduler_status import get_scheduler_export_data

    data = get_scheduler_export_data()

    assert "timestamp" in data
    assert "summary" in data
    assert "jobs" in data
    assert isinstance(data["jobs"], list)


def test_get_scheduler_export_data_job_structure():
    """Test get_scheduler_export_data job structure."""
    from services.dashboard.components.scheduler_status import get_scheduler_export_data

    data = get_scheduler_export_data()
    jobs = data["jobs"]

    assert len(jobs) > 0
    for job in jobs:
        assert "id" in job
        assert "name" in job
        assert "type" in job
        assert "status" in job
        assert "schedule" in job
        assert "stats" in job


def test_scheduled_job_disabled():
    """Test ScheduledJob with is_enabled=False."""
    from services.dashboard.components.scheduler_status import (
        ScheduledJob,
        JobType,
        JobStatus,
    )

    job = ScheduledJob(
        job_id="disabled_001",
        name="Disabled Job",
        job_type=JobType.CLEANUP,
        status=JobStatus.PAUSED,
        is_enabled=False,
    )

    assert job.is_enabled is False
    assert job.status == JobStatus.PAUSED


def test_job_execution_running():
    """Test JobExecution for running job (no finish time)."""
    from services.dashboard.components.scheduler_status import JobExecution, JobStatus

    execution = JobExecution(
        started_at=datetime.now(),
        status=JobStatus.RUNNING,
    )

    assert execution.finished_at is None
    assert execution.duration_ms is None
    assert execution.status == JobStatus.RUNNING
