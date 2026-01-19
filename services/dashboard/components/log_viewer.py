"""Log viewer component for displaying application logs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

# Session state key prefix
_KEY_PREFIX = "log_viewer."

# Demo mode reference date (fixed for deterministic behavior)
_DEMO_REFERENCE_DATE = datetime(2026, 1, 1, 12, 0, 0)

# Maximum log entries to display
_MAX_DISPLAY_ENTRIES = 500


class LogLevel(str, Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Single log entry."""

    timestamp: datetime
    level: LogLevel
    source: str  # Module/component name
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class LogFilter:
    """Log filter settings."""

    levels: List[LogLevel] = field(default_factory=lambda: list(LogLevel))
    source: Optional[str] = None
    keyword: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


def _key(name: str) -> str:
    """Generate namespaced session state key."""
    return f"{_KEY_PREFIX}{name}"


def _get_level_indicator(level: LogLevel) -> str:
    """Get text indicator for log level."""
    indicators = {
        LogLevel.DEBUG: "[DBG]",
        LogLevel.INFO: "[INF]",
        LogLevel.WARNING: "[WRN]",
        LogLevel.ERROR: "[ERR]",
        LogLevel.CRITICAL: "[CRT]",
    }
    return indicators.get(level, "[???]")


def _get_level_color(level: LogLevel) -> str:
    """Get color for log level (for UI styling reference)."""
    colors = {
        LogLevel.DEBUG: "gray",
        LogLevel.INFO: "blue",
        LogLevel.WARNING: "orange",
        LogLevel.ERROR: "red",
        LogLevel.CRITICAL: "darkred",
    }
    return colors.get(level, "gray")


def _get_level_priority(level: LogLevel) -> int:
    """Get numeric priority for log level (higher = more severe)."""
    priorities = {
        LogLevel.DEBUG: 0,
        LogLevel.INFO: 1,
        LogLevel.WARNING: 2,
        LogLevel.ERROR: 3,
        LogLevel.CRITICAL: 4,
    }
    return priorities.get(level, 0)


def _get_demo_logs() -> List[LogEntry]:
    """Get demo log entries (deterministic).

    Uses fixed reference date for consistent demo data.
    """
    base_time = _DEMO_REFERENCE_DATE

    return [
        LogEntry(
            timestamp=base_time - timedelta(minutes=1),
            level=LogLevel.INFO,
            source="scheduler",
            message="Strategy execution completed successfully",
            details={"strategy": "BTC Momentum", "duration_ms": 245},
        ),
        LogEntry(
            timestamp=base_time - timedelta(minutes=2),
            level=LogLevel.DEBUG,
            source="api.upbit",
            message="Market data fetched",
            details={"pairs": ["BTC/KRW", "ETH/KRW"], "latency_ms": 45},
        ),
        LogEntry(
            timestamp=base_time - timedelta(minutes=3),
            level=LogLevel.WARNING,
            source="risk_manager",
            message="Position size approaching limit",
            details={"current_pct": 85.5, "limit_pct": 90.0},
        ),
        LogEntry(
            timestamp=base_time - timedelta(minutes=5),
            level=LogLevel.INFO,
            source="order_executor",
            message="Order placed successfully",
            details={"order_id": "ORD-001", "side": "BUY", "amount": 0.01},
        ),
        LogEntry(
            timestamp=base_time - timedelta(minutes=8),
            level=LogLevel.ERROR,
            source="api.upbit",
            message="API rate limit exceeded, retrying in 60s",
            details={"retry_count": 1, "max_retries": 3},
        ),
        LogEntry(
            timestamp=base_time - timedelta(minutes=10),
            level=LogLevel.INFO,
            source="system",
            message="Database connection pool initialized",
            details={"pool_size": 5, "timeout_ms": 30000},
        ),
        LogEntry(
            timestamp=base_time - timedelta(minutes=15),
            level=LogLevel.DEBUG,
            source="strategy.momentum",
            message="Signal calculation completed",
            details={"signal": 0.75, "threshold": 0.5},
        ),
        LogEntry(
            timestamp=base_time - timedelta(minutes=20),
            level=LogLevel.WARNING,
            source="scheduler",
            message="Task execution delayed",
            details={"task": "price_update", "delay_ms": 1500},
        ),
        LogEntry(
            timestamp=base_time - timedelta(minutes=30),
            level=LogLevel.INFO,
            source="system",
            message="Application started",
            details={"version": "0.1.0", "environment": "development"},
        ),
        LogEntry(
            timestamp=base_time - timedelta(hours=1),
            level=LogLevel.CRITICAL,
            source="order_executor",
            message="Order execution failed - insufficient balance",
            details={"order_id": "ORD-000", "required": 1000000, "available": 500000},
        ),
    ]


def _strip_tzinfo(dt: datetime) -> datetime:
    """Strip timezone info from datetime for fallback comparison.

    Used only when comparing mixed naive/aware datetimes, which is a
    best-effort scenario with no perfect solution.

    Args:
        dt: Datetime to strip

    Returns:
        Naive datetime with tzinfo removed
    """
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _safe_datetime_compare(dt1: datetime, dt2: datetime) -> int:
    """Safely compare two datetimes, handling mixed timezone-awareness.

    Comparison strategy:
    - Both aware: Python compares by UTC absolute time (correct)
    - Both naive: direct comparison (correct)
    - Mixed (aware + naive): fallback to stripping tzinfo (best effort)

    The mixed case has no perfect solution since naive datetime's timezone
    is ambiguous. Stripping tzinfo treats both as local times.

    Args:
        dt1: First datetime
        dt2: Second datetime

    Returns:
        -1 if dt1 < dt2, 0 if equal, 1 if dt1 > dt2
    """
    try:
        # Python handles aware vs aware correctly (compares by UTC)
        if dt1 < dt2:
            return -1
        elif dt1 > dt2:
            return 1
        return 0
    except TypeError:
        # Mixed aware/naive: fallback to stripping tzinfo (best effort)
        dt1_naive = _strip_tzinfo(dt1)
        dt2_naive = _strip_tzinfo(dt2)
        if dt1_naive < dt2_naive:
            return -1
        elif dt1_naive > dt2_naive:
            return 1
        return 0


def _filter_logs(logs: List[LogEntry], log_filter: LogFilter) -> List[LogEntry]:
    """Apply filter to log entries.

    Args:
        logs: List of log entries
        log_filter: Filter settings

    Returns:
        Filtered list of log entries
    """
    filtered = []

    for entry in logs:
        # Filter by level (empty levels list means no entries pass the filter)
        if log_filter.levels is not None:
            if len(log_filter.levels) == 0:
                # User explicitly deselected all levels - show nothing
                continue
            if entry.level not in log_filter.levels:
                continue

        # Filter by source
        if log_filter.source:
            if log_filter.source.lower() not in entry.source.lower():
                continue

        # Filter by keyword
        if log_filter.keyword:
            keyword_lower = log_filter.keyword.lower()
            if (
                keyword_lower not in entry.message.lower()
                and keyword_lower not in entry.source.lower()
            ):
                continue

        # Filter by time range (with safe timezone handling)
        if log_filter.start_time:
            if _safe_datetime_compare(entry.timestamp, log_filter.start_time) < 0:
                continue
        if log_filter.end_time:
            if _safe_datetime_compare(entry.timestamp, log_filter.end_time) > 0:
                continue

        filtered.append(entry)

    return filtered


def _get_unique_sources(logs: List[LogEntry]) -> List[str]:
    """Get unique source names from logs."""
    sources = set()
    for entry in logs:
        sources.add(entry.source)
    return sorted(sources)


def _format_timestamp(dt: datetime) -> str:
    """Format timestamp for display."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _format_relative_time(dt: datetime, reference: Optional[datetime] = None) -> str:
    """Format timestamp as relative time (e.g., '5 minutes ago')."""
    if reference is not None:
        now = reference
    else:
        # Match timezone-awareness of the input datetime
        if dt.tzinfo is not None:
            now = datetime.now(dt.tzinfo)
        else:
            now = datetime.now()

    # Calculate time difference (handle mixed tz-awareness)
    try:
        diff = now - dt
    except TypeError:
        # Fallback: strip timezone info for comparison
        now_naive = _strip_tzinfo(now)
        dt_naive = _strip_tzinfo(dt)
        diff = now_naive - dt_naive

    if diff.total_seconds() < 0:
        return "in the future"

    seconds = int(diff.total_seconds())

    if seconds < 60:
        return f"{seconds}s ago"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m ago"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours}h ago"
    else:
        days = seconds // 86400
        return f"{days}d ago"


def _count_by_level(logs: List[LogEntry]) -> Dict[LogLevel, int]:
    """Count log entries by level."""
    counts: Dict[LogLevel, int] = {level: 0 for level in LogLevel}
    for entry in logs:
        counts[entry.level] = counts.get(entry.level, 0) + 1
    return counts


def render_log_viewer(
    log_provider: Optional[Callable[[], List[LogEntry]]] = None,
    show_filters: bool = True,
    show_summary: bool = True,
    show_details: bool = True,
    compact: bool = False,
    max_entries: int = _MAX_DISPLAY_ENTRIES,
) -> None:
    """Render log viewer panel.

    Args:
        log_provider: Function to get log entries
        show_filters: Whether to show filter controls
        show_summary: Whether to show log summary/statistics
        show_details: Whether to show detail expansion for entries
        compact: Whether to use compact layout
        max_entries: Maximum number of entries to display
    """
    st.subheader("Log Viewer")

    # Get data
    is_demo = log_provider is None
    if is_demo:
        st.caption("Demo Mode")

    try:
        logs = log_provider() if log_provider is not None else _get_demo_logs()
    except Exception:
        logs = _get_demo_logs()
        st.warning("Failed to load logs, showing demo data")

    if not logs:
        st.info("No log entries available.")
        return

    # Build filter from UI
    log_filter = LogFilter()

    if show_filters:
        # Filter controls row
        filter_cols = st.columns([2, 2, 2])

        with filter_cols[0]:
            # Level filter
            level_options = [level.value for level in LogLevel]
            selected_levels = st.multiselect(
                "Log Level",
                options=level_options,
                default=["INFO", "WARNING", "ERROR", "CRITICAL"],
                key=_key("levels"),
                help="Select log levels to display",
            )
            log_filter.levels = [LogLevel(lv) for lv in selected_levels]

        with filter_cols[1]:
            # Source filter
            sources = _get_unique_sources(logs)
            source_options = ["All Sources"] + sources
            selected_source = st.selectbox(
                "Source",
                options=source_options,
                index=0,
                key=_key("source"),
            )
            if selected_source != "All Sources":
                log_filter.source = selected_source

        with filter_cols[2]:
            # Keyword search
            keyword = st.text_input(
                "Search",
                value="",
                key=_key("keyword"),
                placeholder="Filter by keyword...",
            )
            if keyword:
                log_filter.keyword = keyword

    # Apply filter
    filtered_logs = _filter_logs(logs, log_filter)

    # Sort by timestamp (newest first) with safe timezone handling
    def _safe_sort_key(entry: LogEntry) -> datetime:
        """Get sort key, stripping tzinfo for mixed-timezone ordering."""
        return _strip_tzinfo(entry.timestamp)

    filtered_logs.sort(key=_safe_sort_key, reverse=True)

    # Limit entries
    display_logs = filtered_logs[:max_entries]
    truncated = len(filtered_logs) > max_entries

    # Summary section
    if show_summary:
        counts = _count_by_level(filtered_logs)

        summary_cols = st.columns(6)
        with summary_cols[0]:
            st.metric("Total", len(filtered_logs))
        with summary_cols[1]:
            st.metric(f"{_get_level_indicator(LogLevel.DEBUG)}", counts[LogLevel.DEBUG])
        with summary_cols[2]:
            st.metric(f"{_get_level_indicator(LogLevel.INFO)}", counts[LogLevel.INFO])
        with summary_cols[3]:
            st.metric(
                f"{_get_level_indicator(LogLevel.WARNING)}", counts[LogLevel.WARNING]
            )
        with summary_cols[4]:
            st.metric(f"{_get_level_indicator(LogLevel.ERROR)}", counts[LogLevel.ERROR])
        with summary_cols[5]:
            st.metric(
                f"{_get_level_indicator(LogLevel.CRITICAL)}", counts[LogLevel.CRITICAL]
            )

    if truncated:
        st.caption(
            f"Showing {max_entries} of {len(filtered_logs)} entries (oldest truncated)"
        )

    st.divider()

    # Log entries
    if not display_logs:
        st.info("No log entries match the current filter.")
        return

    # Use demo reference time for consistent relative time display
    reference_time = _DEMO_REFERENCE_DATE if is_demo else None

    for entry in display_logs:
        _render_log_entry(entry, show_details, compact, reference_time)


def _render_log_entry(
    entry: LogEntry,
    show_details: bool = True,
    compact: bool = False,
    reference_time: Optional[datetime] = None,
) -> None:
    """Render a single log entry."""
    level_ind = _get_level_indicator(entry.level)
    relative = _format_relative_time(entry.timestamp, reference_time)

    if compact:
        # Compact: single line
        st.markdown(
            f"`{_format_timestamp(entry.timestamp)}` {level_ind} **{entry.source}**: {entry.message}"
        )
    else:
        # Full layout with columns
        cols = st.columns([2, 1, 2, 5])

        with cols[0]:
            st.caption(_format_timestamp(entry.timestamp))
            st.caption(f"({relative})")

        with cols[1]:
            st.markdown(f"**{level_ind}**")

        with cols[2]:
            st.markdown(f"`{entry.source}`")

        with cols[3]:
            st.markdown(entry.message)

            # Show details if available and enabled
            if show_details and entry.details:
                with st.expander("Details", expanded=False):
                    for key, value in entry.details.items():
                        st.text(f"{key}: {value}")


def get_log_export_data(
    logs: Optional[List[LogEntry]] = None,
    log_filter: Optional[LogFilter] = None,
) -> Dict[str, Any]:
    """Get log data for export/API.

    Args:
        logs: List of log entries (uses demo if None)
        log_filter: Optional filter to apply

    Returns:
        Dict containing exportable log data
    """
    if logs is None:
        logs = _get_demo_logs()

    if log_filter is not None:
        logs = _filter_logs(logs, log_filter)

    return {
        "timestamp": datetime.now().isoformat(),
        "total_count": len(logs),
        "counts_by_level": {
            level.value: count for level, count in _count_by_level(logs).items()
        },
        "entries": [
            {
                "timestamp": entry.timestamp.isoformat(),
                "level": entry.level.value,
                "source": entry.source,
                "message": entry.message,
                "details": entry.details,
            }
            for entry in logs
        ],
    }
