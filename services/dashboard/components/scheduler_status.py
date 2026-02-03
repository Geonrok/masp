"""Scheduler status component for displaying scheduled job information."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

# Session state key prefix
_KEY_PREFIX = "scheduler_status."

# Demo mode reference date (fixed for deterministic behavior)
_DEMO_REFERENCE_DATE = datetime(2026, 1, 1, 12, 0, 0)


class JobStatus(str, Enum):
    """Job execution status."""

    RUNNING = "RUNNING"
    SCHEDULED = "SCHEDULED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"


class JobType(str, Enum):
    """Type of scheduled job."""

    STRATEGY = "STRATEGY"
    DATA_FETCH = "DATA_FETCH"
    CLEANUP = "CLEANUP"
    REPORT = "REPORT"
    BACKUP = "BACKUP"


@dataclass
class JobSchedule:
    """Job schedule configuration."""

    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    next_run: Optional[datetime] = None


@dataclass
class JobExecution:
    """Job execution record."""

    started_at: datetime
    finished_at: Optional[datetime] = None
    status: JobStatus = JobStatus.RUNNING
    duration_ms: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class ScheduledJob:
    """Scheduled job information."""

    job_id: str
    name: str
    job_type: JobType
    status: JobStatus = JobStatus.SCHEDULED
    schedule: JobSchedule = field(default_factory=JobSchedule)
    last_execution: Optional[JobExecution] = None
    execution_count: int = 0
    failure_count: int = 0
    is_enabled: bool = True
    description: str = ""


def _key(name: str) -> str:
    """Generate namespaced session state key."""
    return f"{_KEY_PREFIX}{name}"


def _get_status_indicator(status: JobStatus) -> str:
    """Get text indicator for job status."""
    indicators = {
        JobStatus.RUNNING: "[RUN]",
        JobStatus.SCHEDULED: "[SCH]",
        JobStatus.COMPLETED: "[OK]",
        JobStatus.FAILED: "[ERR]",
        JobStatus.PAUSED: "[PSE]",
    }
    return indicators.get(status, "[???]")


def _get_status_color(status: JobStatus) -> str:
    """Get color for job status (for UI styling reference)."""
    colors = {
        JobStatus.RUNNING: "blue",
        JobStatus.SCHEDULED: "gray",
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
        JobStatus.PAUSED: "orange",
    }
    return colors.get(status, "gray")


def _get_job_type_indicator(job_type: JobType) -> str:
    """Get text indicator for job type."""
    indicators = {
        JobType.STRATEGY: "[STR]",
        JobType.DATA_FETCH: "[DAT]",
        JobType.CLEANUP: "[CLN]",
        JobType.REPORT: "[RPT]",
        JobType.BACKUP: "[BAK]",
    }
    return indicators.get(job_type, "[???]")


def _format_datetime(dt: Optional[datetime]) -> str:
    """Format datetime for display."""
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _format_relative_time(
    dt: Optional[datetime], reference: Optional[datetime] = None
) -> str:
    """Format timestamp as relative time."""
    if dt is None:
        return "-"

    if reference is not None:
        now = reference
    else:
        # Match timezone-awareness of the input datetime
        if dt.tzinfo is not None:
            now = datetime.now(dt.tzinfo)
        else:
            now = datetime.now()

    # Handle mixed tz-awareness safely
    try:
        diff = now - dt
    except TypeError:
        now_naive = now.replace(tzinfo=None) if now.tzinfo else now
        dt_naive = dt.replace(tzinfo=None) if dt.tzinfo else dt
        diff = now_naive - dt_naive

    total_seconds = diff.total_seconds()

    if total_seconds < 0:
        # Future time
        abs_seconds = abs(int(total_seconds))
        if abs_seconds < 60:
            return f"in {abs_seconds}s"
        elif abs_seconds < 3600:
            return f"in {abs_seconds // 60}m"
        elif abs_seconds < 86400:
            return f"in {abs_seconds // 3600}h"
        else:
            return f"in {abs_seconds // 86400}d"
    else:
        # Past time
        seconds = int(total_seconds)
        if seconds < 60:
            return f"{seconds}s ago"
        elif seconds < 3600:
            return f"{seconds // 60}m ago"
        elif seconds < 86400:
            return f"{seconds // 3600}h ago"
        else:
            return f"{seconds // 86400}d ago"


def _format_duration(duration_ms: Optional[int]) -> str:
    """Format duration in milliseconds to human readable string."""
    if duration_ms is None:
        return "-"

    if duration_ms < 1000:
        return f"{duration_ms}ms"
    elif duration_ms < 60000:
        return f"{duration_ms / 1000:.1f}s"
    else:
        minutes = duration_ms // 60000
        seconds = (duration_ms % 60000) / 1000
        return f"{minutes}m {seconds:.0f}s"


def _format_interval(seconds: Optional[int]) -> str:
    """Format interval in seconds to human readable string."""
    if seconds is None:
        return "-"

    if seconds < 60:
        return f"Every {seconds}s"
    elif seconds < 3600:
        return f"Every {seconds // 60}m"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"Every {hours}h"
    else:
        days = seconds // 86400
        return f"Every {days}d"


def _calculate_success_rate(execution_count: int, failure_count: int) -> float:
    """Calculate success rate as percentage."""
    if execution_count <= 0:
        return 0.0
    success_count = execution_count - failure_count
    return (success_count / execution_count) * 100


def _get_demo_jobs() -> List[ScheduledJob]:
    """Get demo scheduled job data (deterministic).

    Uses fixed reference date for consistent demo data.
    """
    base_time = _DEMO_REFERENCE_DATE

    return [
        ScheduledJob(
            job_id="job_001",
            name="BTC Momentum Strategy",
            job_type=JobType.STRATEGY,
            status=JobStatus.SCHEDULED,
            schedule=JobSchedule(
                cron_expression="*/5 * * * *",
                interval_seconds=300,
                next_run=base_time + timedelta(minutes=3),
            ),
            last_execution=JobExecution(
                started_at=base_time - timedelta(minutes=2),
                finished_at=base_time
                - timedelta(minutes=2)
                + timedelta(seconds=1, milliseconds=245),
                status=JobStatus.COMPLETED,
                duration_ms=1245,
            ),
            execution_count=288,
            failure_count=2,
            is_enabled=True,
            description="Execute BTC momentum strategy signals",
        ),
        ScheduledJob(
            job_id="job_002",
            name="Market Data Fetch",
            job_type=JobType.DATA_FETCH,
            status=JobStatus.RUNNING,
            schedule=JobSchedule(
                cron_expression="* * * * *",
                interval_seconds=60,
                next_run=base_time + timedelta(seconds=45),
            ),
            last_execution=JobExecution(
                started_at=base_time - timedelta(seconds=15),
                status=JobStatus.RUNNING,
                duration_ms=None,
            ),
            execution_count=1440,
            failure_count=5,
            is_enabled=True,
            description="Fetch real-time market data from exchanges",
        ),
        ScheduledJob(
            job_id="job_003",
            name="Daily Report Generation",
            job_type=JobType.REPORT,
            status=JobStatus.SCHEDULED,
            schedule=JobSchedule(
                cron_expression="0 9 * * *",
                interval_seconds=86400,
                next_run=base_time + timedelta(hours=21),
            ),
            last_execution=JobExecution(
                started_at=base_time - timedelta(hours=3),
                finished_at=base_time - timedelta(hours=3) + timedelta(seconds=45),
                status=JobStatus.COMPLETED,
                duration_ms=45000,
            ),
            execution_count=30,
            failure_count=0,
            is_enabled=True,
            description="Generate daily performance report",
        ),
        ScheduledJob(
            job_id="job_004",
            name="Log Cleanup",
            job_type=JobType.CLEANUP,
            status=JobStatus.PAUSED,
            schedule=JobSchedule(
                cron_expression="0 0 * * 0",
                interval_seconds=604800,
                next_run=None,
            ),
            last_execution=JobExecution(
                started_at=base_time - timedelta(days=7),
                finished_at=base_time - timedelta(days=7) + timedelta(minutes=2),
                status=JobStatus.COMPLETED,
                duration_ms=120000,
            ),
            execution_count=4,
            failure_count=0,
            is_enabled=False,
            description="Clean up old log files",
        ),
        ScheduledJob(
            job_id="job_005",
            name="ETH Strategy",
            job_type=JobType.STRATEGY,
            status=JobStatus.FAILED,
            schedule=JobSchedule(
                cron_expression="*/10 * * * *",
                interval_seconds=600,
                next_run=base_time + timedelta(minutes=8),
            ),
            last_execution=JobExecution(
                started_at=base_time - timedelta(minutes=2),
                finished_at=base_time
                - timedelta(minutes=2)
                + timedelta(seconds=0, milliseconds=500),
                status=JobStatus.FAILED,
                duration_ms=500,
                error_message="API rate limit exceeded",
            ),
            execution_count=144,
            failure_count=12,
            is_enabled=True,
            description="Execute ETH mean reversion strategy",
        ),
    ]


def _filter_jobs_by_type(
    jobs: List[ScheduledJob], job_type: Optional[JobType]
) -> List[ScheduledJob]:
    """Filter jobs by type."""
    if job_type is None:
        return jobs
    return [job for job in jobs if job.job_type == job_type]


def _filter_jobs_by_status(
    jobs: List[ScheduledJob], statuses: List[JobStatus]
) -> List[ScheduledJob]:
    """Filter jobs by status."""
    if not statuses:
        return []  # Empty status selection = show nothing
    return [job for job in jobs if job.status in statuses]


def _get_scheduler_summary(jobs: List[ScheduledJob]) -> Dict[str, Any]:
    """Calculate scheduler summary statistics."""
    if not jobs:
        return {
            "total_jobs": 0,
            "running": 0,
            "scheduled": 0,
            "failed": 0,
            "paused": 0,
            "total_executions": 0,
            "total_failures": 0,
            "overall_success_rate": 0.0,
        }

    status_counts = {status: 0 for status in JobStatus}
    total_executions = 0
    total_failures = 0

    for job in jobs:
        status_counts[job.status] = status_counts.get(job.status, 0) + 1
        total_executions += job.execution_count
        total_failures += job.failure_count

    return {
        "total_jobs": len(jobs),
        "running": status_counts.get(JobStatus.RUNNING, 0),
        "scheduled": status_counts.get(JobStatus.SCHEDULED, 0),
        "failed": status_counts.get(JobStatus.FAILED, 0),
        "paused": status_counts.get(JobStatus.PAUSED, 0),
        "total_executions": total_executions,
        "total_failures": total_failures,
        "overall_success_rate": _calculate_success_rate(
            total_executions, total_failures
        ),
    }


def render_scheduler_status(
    job_provider: Optional[Callable[[], List[ScheduledJob]]] = None,
    show_summary: bool = True,
    show_filters: bool = True,
    show_details: bool = True,
    compact: bool = False,
) -> None:
    """Render scheduler status panel.

    Args:
        job_provider: Function to get scheduled job data
        show_summary: Whether to show summary statistics
        show_filters: Whether to show filter controls
        show_details: Whether to show detailed job information
        compact: Whether to use compact layout
    """
    st.subheader("Scheduler Status")

    # Get data
    is_demo = job_provider is None
    if is_demo:
        st.caption("Demo Mode")

    try:
        jobs = job_provider() if job_provider is not None else _get_demo_jobs()
    except Exception:
        jobs = _get_demo_jobs()
        st.warning("Failed to load scheduler data, showing demo data")

    if not jobs:
        st.info("No scheduled jobs available.")
        return

    # Use demo reference time for consistent display
    reference_time = _DEMO_REFERENCE_DATE if is_demo else None

    # Filters
    filtered_jobs = jobs
    if show_filters:
        filter_cols = st.columns([2, 3])

        with filter_cols[0]:
            # Job type filter
            type_options = ["All Types"] + [jt.value for jt in JobType]
            selected_type = st.selectbox(
                "Job Type",
                options=type_options,
                index=0,
                key=_key("job_type"),
            )
            if selected_type != "All Types":
                filtered_jobs = _filter_jobs_by_type(
                    filtered_jobs, JobType(selected_type)
                )

        with filter_cols[1]:
            # Status filter
            status_options = [s.value for s in JobStatus]
            selected_statuses = st.multiselect(
                "Status",
                options=status_options,
                default=status_options,  # All selected by default
                key=_key("statuses"),
            )
            filtered_jobs = _filter_jobs_by_status(
                filtered_jobs, [JobStatus(s) for s in selected_statuses]
            )

    # Summary section
    if show_summary:
        summary = _get_scheduler_summary(filtered_jobs)

        summary_cols = st.columns(5)
        with summary_cols[0]:
            st.metric("Total Jobs", summary["total_jobs"])
        with summary_cols[1]:
            st.metric(f"{_get_status_indicator(JobStatus.RUNNING)}", summary["running"])
        with summary_cols[2]:
            st.metric(
                f"{_get_status_indicator(JobStatus.SCHEDULED)}", summary["scheduled"]
            )
        with summary_cols[3]:
            st.metric(f"{_get_status_indicator(JobStatus.FAILED)}", summary["failed"])
        with summary_cols[4]:
            st.metric("Success Rate", f"{summary['overall_success_rate']:.1f}%")

        st.caption(
            f"Total Executions: {summary['total_executions']:,} | "
            f"Failures: {summary['total_failures']:,}"
        )

    st.divider()

    # Job list
    if not filtered_jobs:
        st.info("No jobs match the current filter.")
        return

    for job in filtered_jobs:
        _render_job_card(job, show_details, compact, reference_time)


def _render_job_card(
    job: ScheduledJob,
    show_details: bool = True,
    compact: bool = False,
    reference_time: Optional[datetime] = None,
) -> None:
    """Render a single job card."""
    status_ind = _get_status_indicator(job.status)
    type_ind = _get_job_type_indicator(job.job_type)
    enabled_text = "" if job.is_enabled else " [DISABLED]"

    if compact:
        # Compact layout: single line
        cols = st.columns([3, 1, 2, 2])
        with cols[0]:
            st.markdown(f"{status_ind} **{job.name}**{enabled_text}")
        with cols[1]:
            st.caption(type_ind)
        with cols[2]:
            if job.schedule.next_run:
                st.caption(
                    f"Next: {_format_relative_time(job.schedule.next_run, reference_time)}"
                )
            else:
                st.caption("Next: -")
        with cols[3]:
            success_rate = _calculate_success_rate(
                job.execution_count, job.failure_count
            )
            st.caption(f"Success: {success_rate:.1f}%")
    else:
        # Full layout with expander
        with st.expander(
            f"{status_ind} {type_ind} {job.name}{enabled_text}",
            expanded=job.status in [JobStatus.RUNNING, JobStatus.FAILED],
        ):
            # Job info row
            info_cols = st.columns(4)
            with info_cols[0]:
                st.markdown("**Status**")
                st.markdown(f"`{job.status.value}`")
            with info_cols[1]:
                st.markdown("**Type**")
                st.markdown(f"`{job.job_type.value}`")
            with info_cols[2]:
                st.markdown("**Schedule**")
                if job.schedule.cron_expression:
                    st.caption(f"Cron: `{job.schedule.cron_expression}`")
                if job.schedule.interval_seconds:
                    st.caption(_format_interval(job.schedule.interval_seconds))
            with info_cols[3]:
                st.markdown("**Next Run**")
                if job.schedule.next_run:
                    st.caption(_format_datetime(job.schedule.next_run))
                    st.caption(
                        f"({_format_relative_time(job.schedule.next_run, reference_time)})"
                    )
                else:
                    st.caption("-")

            # Execution stats
            if show_details:
                st.markdown("---")
                stats_cols = st.columns(4)
                with stats_cols[0]:
                    st.metric("Executions", f"{job.execution_count:,}")
                with stats_cols[1]:
                    st.metric("Failures", f"{job.failure_count:,}")
                with stats_cols[2]:
                    success_rate = _calculate_success_rate(
                        job.execution_count, job.failure_count
                    )
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                with stats_cols[3]:
                    if job.last_execution:
                        st.metric(
                            "Last Duration",
                            _format_duration(job.last_execution.duration_ms),
                        )
                    else:
                        st.metric("Last Duration", "-")

                # Last execution details
                if job.last_execution:
                    st.markdown("**Last Execution**")
                    exec_cols = st.columns(3)
                    with exec_cols[0]:
                        st.caption(
                            f"Started: {_format_datetime(job.last_execution.started_at)}"
                        )
                    with exec_cols[1]:
                        st.caption(
                            f"Status: {_get_status_indicator(job.last_execution.status)}"
                        )
                    with exec_cols[2]:
                        if job.last_execution.finished_at:
                            st.caption(
                                f"Finished: {_format_datetime(job.last_execution.finished_at)}"
                            )

                    # Error message if failed
                    if (
                        job.last_execution.status == JobStatus.FAILED
                        and job.last_execution.error_message
                    ):
                        st.error(f"Error: {job.last_execution.error_message}")

            # Description
            if job.description:
                st.caption(f"_{job.description}_")


def get_scheduler_export_data(
    jobs: Optional[List[ScheduledJob]] = None,
) -> Dict[str, Any]:
    """Get scheduler data for export/API.

    Args:
        jobs: List of scheduled jobs (uses demo if None)

    Returns:
        Dict containing exportable scheduler data
    """
    if jobs is None:
        jobs = _get_demo_jobs()

    summary = _get_scheduler_summary(jobs)

    return {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "jobs": [
            {
                "id": job.job_id,
                "name": job.name,
                "type": job.job_type.value,
                "status": job.status.value,
                "is_enabled": job.is_enabled,
                "schedule": {
                    "cron": job.schedule.cron_expression,
                    "interval_seconds": job.schedule.interval_seconds,
                    "next_run": (
                        job.schedule.next_run.isoformat()
                        if job.schedule.next_run
                        else None
                    ),
                },
                "stats": {
                    "execution_count": job.execution_count,
                    "failure_count": job.failure_count,
                    "success_rate": _calculate_success_rate(
                        job.execution_count, job.failure_count
                    ),
                },
                "last_execution": (
                    {
                        "started_at": job.last_execution.started_at.isoformat(),
                        "finished_at": (
                            job.last_execution.finished_at.isoformat()
                            if job.last_execution.finished_at
                            else None
                        ),
                        "status": job.last_execution.status.value,
                        "duration_ms": job.last_execution.duration_ms,
                        "error": job.last_execution.error_message,
                    }
                    if job.last_execution
                    else None
                ),
                "description": job.description,
            }
            for job in jobs
        ],
    }
