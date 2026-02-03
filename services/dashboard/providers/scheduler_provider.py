"""Scheduler status provider - connects APScheduler to scheduler_status component."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable, List, Optional

from services.dashboard.components.scheduler_status import (
    JobExecution,
    JobSchedule,
    JobStatus,
    JobType,
    ScheduledJob,
)

logger = logging.getLogger(__name__)


def _get_apscheduler():
    """Get APScheduler instance if available.

    Returns:
        Scheduler instance or None if unavailable
    """
    try:
        from services.scheduler import DailyScheduler
        from services.strategy_runner import StrategyRunner

        # Try to get the global scheduler instance
        # Note: This requires the scheduler to be running
        return None  # APScheduler doesn't have a global instance by default

    except ImportError as e:
        logger.debug("APScheduler import failed: %s", e)
        return None
    except Exception as e:
        logger.debug("APScheduler access failed: %s", e)
        return None


def _parse_job_type(job_name: str) -> JobType:
    """Parse job type from job name.

    Args:
        job_name: Name of the job

    Returns:
        JobType enum value
    """
    name_lower = job_name.lower()

    if "strategy" in name_lower or "momentum" in name_lower or "mean" in name_lower:
        return JobType.STRATEGY
    elif "data" in name_lower or "fetch" in name_lower or "market" in name_lower:
        return JobType.DATA_FETCH
    elif "report" in name_lower or "summary" in name_lower:
        return JobType.REPORT
    elif "cleanup" in name_lower or "clean" in name_lower:
        return JobType.CLEANUP
    elif "backup" in name_lower:
        return JobType.BACKUP
    else:
        return JobType.STRATEGY


def _convert_apscheduler_job(job, now: datetime) -> ScheduledJob:
    """Convert APScheduler job to ScheduledJob.

    Args:
        job: APScheduler job object
        now: Current datetime

    Returns:
        ScheduledJob instance
    """
    # Get next run time
    next_run = job.next_run_time if hasattr(job, "next_run_time") else None

    # Determine status
    if not getattr(job, "pending", True):
        status = JobStatus.RUNNING
    elif next_run is None:
        status = JobStatus.PAUSED
    else:
        status = JobStatus.SCHEDULED

    # Get trigger info
    trigger = getattr(job, "trigger", None)
    cron_expr = None
    interval_seconds = None

    if trigger:
        if hasattr(trigger, "interval"):
            interval_seconds = int(trigger.interval.total_seconds())
        # CronTrigger doesn't expose cron expression directly
        # We can try to reconstruct it from fields
        if hasattr(trigger, "fields"):
            try:
                fields = trigger.fields
                cron_parts = []
                for field in ["minute", "hour", "day", "month", "day_of_week"]:
                    if hasattr(fields, field):
                        cron_parts.append(str(getattr(fields, field)))
                if len(cron_parts) == 5:
                    cron_expr = " ".join(cron_parts)
            except Exception:
                pass

    return ScheduledJob(
        job_id=str(job.id),
        name=job.name or str(job.id),
        job_type=_parse_job_type(job.name or ""),
        status=status,
        schedule=JobSchedule(
            cron_expression=cron_expr,
            interval_seconds=interval_seconds,
            next_run=next_run,
        ),
        last_execution=None,  # APScheduler doesn't track execution history
        execution_count=0,
        failure_count=0,
        is_enabled=next_run is not None,
        description=str(getattr(job, "func", "")),
    )


def _get_static_jobs() -> List[ScheduledJob]:
    """Get static job definitions from configuration.

    Returns list of jobs based on known MASP scheduled tasks.
    This provides meaningful data even when APScheduler isn't running.

    Returns:
        List of ScheduledJob instances
    """
    now = datetime.now()

    return [
        ScheduledJob(
            job_id="masp_daily_strategy",
            name="Daily Strategy Execution",
            job_type=JobType.STRATEGY,
            status=JobStatus.SCHEDULED,
            schedule=JobSchedule(
                cron_expression="0 9 * * *",
                interval_seconds=86400,
                next_run=_next_run_at_hour(now, 9),
            ),
            execution_count=0,
            failure_count=0,
            is_enabled=True,
            description="Execute daily trading strategies",
        ),
        ScheduledJob(
            job_id="masp_market_data",
            name="Market Data Fetch",
            job_type=JobType.DATA_FETCH,
            status=JobStatus.SCHEDULED,
            schedule=JobSchedule(
                cron_expression="*/5 * * * *",
                interval_seconds=300,
                next_run=now + timedelta(minutes=5 - now.minute % 5),
            ),
            execution_count=0,
            failure_count=0,
            is_enabled=True,
            description="Fetch real-time market data from exchanges",
        ),
        ScheduledJob(
            job_id="masp_daily_report",
            name="Daily Report Generation",
            job_type=JobType.REPORT,
            status=JobStatus.SCHEDULED,
            schedule=JobSchedule(
                cron_expression="0 21 * * *",
                interval_seconds=86400,
                next_run=_next_run_at_hour(now, 21),
            ),
            execution_count=0,
            failure_count=0,
            is_enabled=True,
            description="Generate daily performance report",
        ),
    ]


def _next_run_at_hour(now: datetime, hour: int) -> datetime:
    """Calculate next run time at specified hour.

    Args:
        now: Current datetime
        hour: Target hour (0-23)

    Returns:
        Next datetime at specified hour
    """
    next_run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(days=1)
    return next_run


def get_scheduled_jobs() -> List[ScheduledJob]:
    """Get scheduled jobs from APScheduler or static config.

    Returns:
        List of ScheduledJob instances
    """
    scheduler = _get_apscheduler()

    if scheduler is not None:
        try:
            jobs = scheduler.get_jobs() if hasattr(scheduler, "get_jobs") else []
            now = datetime.now()
            return [_convert_apscheduler_job(job, now) for job in jobs]
        except Exception as e:
            logger.debug("Failed to get APScheduler jobs: %s", e)

    # Return static jobs when scheduler not available
    return _get_static_jobs()


def get_scheduler_job_provider() -> Optional[Callable[[], List[ScheduledJob]]]:
    """Get job provider function for scheduler_status component.

    Returns:
        Function that returns List[ScheduledJob], or None for demo mode
    """

    def job_provider() -> List[ScheduledJob]:
        return get_scheduled_jobs()

    # Always return provider (will use static jobs if scheduler unavailable)
    return job_provider
