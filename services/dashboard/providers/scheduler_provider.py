"""Scheduler status provider - reads schedule_config.json for real job data."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional

from services.dashboard.components.scheduler_status import (
    JobSchedule,
    JobStatus,
    JobType,
    ScheduledJob,
)

logger = logging.getLogger(__name__)

_SCHEDULE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "config"
    / "schedule_config.json"
)


def _next_run_at_hour(now: datetime, hour: int, minute: int = 0) -> datetime:
    """Calculate next run time at specified hour and minute.

    Args:
        now: Current datetime
        hour: Target hour (0-23)
        minute: Target minute (0-59)

    Returns:
        Next datetime at specified hour/minute
    """
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(days=1)
    return next_run


def _get_config_jobs() -> List[ScheduledJob]:
    """Read jobs from schedule_config.json.

    Returns:
        List of ScheduledJob instances based on actual config
    """
    try:
        with open(_SCHEDULE_CONFIG_PATH, encoding="utf-8") as f:
            exchanges = json.load(f).get("exchanges", {})
    except Exception as e:
        logger.debug("Failed to read schedule_config.json: %s", e)
        return []

    jobs: List[ScheduledJob] = []
    now = datetime.now()

    for name, cfg in exchanges.items():
        enabled = cfg.get("enabled", False)
        strategy = cfg.get("strategy", "unknown")
        schedule = cfg.get("schedule", {})

        # Build cron expression from schedule
        minute = schedule.get("minute", 0)
        tz = schedule.get("timezone", "UTC")

        hours = schedule.get("hours")
        hour = schedule.get("hour")

        if hours:
            hours_str = ",".join(str(h) for h in hours)
            cron_expr = f"{minute} {hours_str} * * *"
            # Next run: find the closest upcoming hour
            next_run = None
            for h in sorted(hours):
                candidate = now.replace(hour=h, minute=minute, second=0, microsecond=0)
                if candidate > now:
                    next_run = candidate
                    break
            if next_run is None:
                # Wrap to tomorrow's first hour
                next_run = (now + timedelta(days=1)).replace(
                    hour=sorted(hours)[0],
                    minute=minute,
                    second=0,
                    microsecond=0,
                )
        elif hour is not None:
            cron_expr = f"{minute} {hour} * * *"
            next_run = _next_run_at_hour(now, hour, minute)
        else:
            # Special schedules (e.g., monthly)
            day = schedule.get("day", "")
            cron_expr = f"{minute} {hour or 0} {day} * *"
            next_run = _next_run_at_hour(now, hour or 0, minute)

        comment = cfg.get("_comment", "")

        jobs.append(
            ScheduledJob(
                job_id=name,
                name=f"{name} ({strategy})",
                job_type=JobType.STRATEGY,
                status=JobStatus.SCHEDULED if enabled else JobStatus.PAUSED,
                schedule=JobSchedule(
                    cron_expression=cron_expr,
                    next_run=next_run if enabled else None,
                ),
                is_enabled=enabled,
                description=f"Strategy: {strategy}, TZ: {tz}. {comment}",
            )
        )

    return jobs


def get_scheduled_jobs() -> List[ScheduledJob]:
    """Get scheduled jobs from schedule_config.json.

    Returns:
        List of ScheduledJob instances
    """
    return _get_config_jobs()


def get_scheduler_job_provider() -> Optional[Callable[[], List[ScheduledJob]]]:
    """Get job provider function for scheduler_status component.

    Returns:
        Function that returns List[ScheduledJob]
    """

    def job_provider() -> List[ScheduledJob]:
        return get_scheduled_jobs()

    return job_provider
