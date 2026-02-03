"""Tests for log viewer component."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest


def test_import_log_viewer():
    """Test module imports correctly."""
    from services.dashboard.components import log_viewer

    assert hasattr(log_viewer, "render_log_viewer")
    assert hasattr(log_viewer, "LogEntry")
    assert hasattr(log_viewer, "LogLevel")
    assert hasattr(log_viewer, "LogFilter")


def test_log_level_enum():
    """Test LogLevel enum values."""
    from services.dashboard.components.log_viewer import LogLevel

    assert LogLevel.DEBUG.value == "DEBUG"
    assert LogLevel.INFO.value == "INFO"
    assert LogLevel.WARNING.value == "WARNING"
    assert LogLevel.ERROR.value == "ERROR"
    assert LogLevel.CRITICAL.value == "CRITICAL"


def test_log_entry_dataclass():
    """Test LogEntry dataclass."""
    from services.dashboard.components.log_viewer import LogEntry, LogLevel

    entry = LogEntry(
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        level=LogLevel.INFO,
        source="test_module",
        message="Test message",
    )

    assert entry.level == LogLevel.INFO
    assert entry.source == "test_module"
    assert entry.message == "Test message"
    assert entry.details is None


def test_log_entry_with_details():
    """Test LogEntry dataclass with details."""
    from services.dashboard.components.log_viewer import LogEntry, LogLevel

    entry = LogEntry(
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        level=LogLevel.ERROR,
        source="api",
        message="Request failed",
        details={"status_code": 500, "retry": True},
    )

    assert entry.details == {"status_code": 500, "retry": True}


def test_log_filter_defaults():
    """Test LogFilter dataclass defaults."""
    from services.dashboard.components.log_viewer import LogFilter, LogLevel

    log_filter = LogFilter()

    assert len(log_filter.levels) == 5  # All levels
    assert log_filter.source is None
    assert log_filter.keyword is None
    assert log_filter.start_time is None
    assert log_filter.end_time is None


def test_log_filter_with_values():
    """Test LogFilter dataclass with values."""
    from services.dashboard.components.log_viewer import LogFilter, LogLevel

    log_filter = LogFilter(
        levels=[LogLevel.ERROR, LogLevel.CRITICAL],
        source="api",
        keyword="failed",
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 2),
    )

    assert len(log_filter.levels) == 2
    assert log_filter.source == "api"
    assert log_filter.keyword == "failed"


def test_key_function():
    """Test _key generates namespaced keys."""
    from services.dashboard.components.log_viewer import _key, _KEY_PREFIX

    result = _key("levels")
    assert result == f"{_KEY_PREFIX}levels"
    assert result == "log_viewer.levels"


def test_get_level_indicator():
    """Test _get_level_indicator returns correct indicators."""
    from services.dashboard.components.log_viewer import _get_level_indicator, LogLevel

    assert _get_level_indicator(LogLevel.DEBUG) == "[DBG]"
    assert _get_level_indicator(LogLevel.INFO) == "[INF]"
    assert _get_level_indicator(LogLevel.WARNING) == "[WRN]"
    assert _get_level_indicator(LogLevel.ERROR) == "[ERR]"
    assert _get_level_indicator(LogLevel.CRITICAL) == "[CRT]"


def test_get_level_color():
    """Test _get_level_color returns correct colors."""
    from services.dashboard.components.log_viewer import _get_level_color, LogLevel

    assert _get_level_color(LogLevel.DEBUG) == "gray"
    assert _get_level_color(LogLevel.INFO) == "blue"
    assert _get_level_color(LogLevel.WARNING) == "orange"
    assert _get_level_color(LogLevel.ERROR) == "red"
    assert _get_level_color(LogLevel.CRITICAL) == "darkred"


def test_get_level_priority():
    """Test _get_level_priority returns correct priorities."""
    from services.dashboard.components.log_viewer import _get_level_priority, LogLevel

    assert _get_level_priority(LogLevel.DEBUG) == 0
    assert _get_level_priority(LogLevel.INFO) == 1
    assert _get_level_priority(LogLevel.WARNING) == 2
    assert _get_level_priority(LogLevel.ERROR) == 3
    assert _get_level_priority(LogLevel.CRITICAL) == 4


def test_get_demo_logs():
    """Test _get_demo_logs returns expected data."""
    from services.dashboard.components.log_viewer import _get_demo_logs

    logs = _get_demo_logs()

    assert len(logs) >= 5
    assert all(hasattr(entry, "timestamp") for entry in logs)
    assert all(hasattr(entry, "level") for entry in logs)
    assert all(hasattr(entry, "source") for entry in logs)
    assert all(hasattr(entry, "message") for entry in logs)


def test_get_demo_logs_deterministic():
    """Test _get_demo_logs returns consistent data."""
    from services.dashboard.components.log_viewer import _get_demo_logs

    logs1 = _get_demo_logs()
    logs2 = _get_demo_logs()

    assert len(logs1) == len(logs2)
    assert logs1[0].timestamp == logs2[0].timestamp
    assert logs1[0].message == logs2[0].message


def test_filter_logs_by_level():
    """Test _filter_logs filters by level correctly."""
    from services.dashboard.components.log_viewer import (
        _filter_logs,
        LogEntry,
        LogFilter,
        LogLevel,
    )

    logs = [
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="test",
            message="Info message",
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            source="test",
            message="Error message",
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.DEBUG,
            source="test",
            message="Debug message",
        ),
    ]

    log_filter = LogFilter(levels=[LogLevel.ERROR])
    result = _filter_logs(logs, log_filter)

    assert len(result) == 1
    assert result[0].level == LogLevel.ERROR


def test_filter_logs_by_source():
    """Test _filter_logs filters by source correctly."""
    from services.dashboard.components.log_viewer import (
        _filter_logs,
        LogEntry,
        LogFilter,
        LogLevel,
    )

    logs = [
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="api.upbit",
            message="API message",
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="scheduler",
            message="Scheduler message",
        ),
    ]

    log_filter = LogFilter(source="api")
    result = _filter_logs(logs, log_filter)

    assert len(result) == 1
    assert "api" in result[0].source.lower()


def test_filter_logs_by_keyword():
    """Test _filter_logs filters by keyword correctly."""
    from services.dashboard.components.log_viewer import (
        _filter_logs,
        LogEntry,
        LogFilter,
        LogLevel,
    )

    logs = [
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="test",
            message="Order placed successfully",
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            source="test",
            message="Connection failed",
        ),
    ]

    log_filter = LogFilter(keyword="order")
    result = _filter_logs(logs, log_filter)

    assert len(result) == 1
    assert "order" in result[0].message.lower()


def test_filter_logs_by_time_range():
    """Test _filter_logs filters by time range correctly."""
    from services.dashboard.components.log_viewer import (
        _filter_logs,
        LogEntry,
        LogFilter,
        LogLevel,
    )

    base_time = datetime(2026, 1, 15, 12, 0, 0)

    logs = [
        LogEntry(
            timestamp=base_time - timedelta(hours=1),
            level=LogLevel.INFO,
            source="test",
            message="Old message",
        ),
        LogEntry(
            timestamp=base_time,
            level=LogLevel.INFO,
            source="test",
            message="Current message",
        ),
        LogEntry(
            timestamp=base_time + timedelta(hours=1),
            level=LogLevel.INFO,
            source="test",
            message="Future message",
        ),
    ]

    log_filter = LogFilter(
        start_time=base_time - timedelta(minutes=30),
        end_time=base_time + timedelta(minutes=30),
    )
    result = _filter_logs(logs, log_filter)

    assert len(result) == 1
    assert result[0].message == "Current message"


def test_filter_logs_default_filter_returns_all():
    """Test _filter_logs with default filter (all levels) returns all."""
    from services.dashboard.components.log_viewer import (
        _filter_logs,
        LogEntry,
        LogFilter,
        LogLevel,
    )

    logs = [
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="test",
            message="Message 1",
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            source="test",
            message="Message 2",
        ),
    ]

    log_filter = LogFilter()  # All levels, no other filters
    result = _filter_logs(logs, log_filter)

    assert len(result) == 2


def test_filter_logs_empty_levels_returns_none():
    """Test _filter_logs with empty levels list returns no entries."""
    from services.dashboard.components.log_viewer import (
        _filter_logs,
        LogEntry,
        LogFilter,
        LogLevel,
    )

    logs = [
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="test",
            message="Message 1",
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            source="test",
            message="Message 2",
        ),
    ]

    # User deselected all levels - should show nothing
    log_filter = LogFilter(levels=[])
    result = _filter_logs(logs, log_filter)

    assert len(result) == 0


def test_get_unique_sources():
    """Test _get_unique_sources returns unique sorted sources."""
    from services.dashboard.components.log_viewer import (
        _get_unique_sources,
        LogEntry,
        LogLevel,
    )

    logs = [
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="scheduler",
            message="Msg",
        ),
        LogEntry(
            timestamp=datetime.now(), level=LogLevel.INFO, source="api", message="Msg"
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="scheduler",
            message="Msg",
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="database",
            message="Msg",
        ),
    ]

    sources = _get_unique_sources(logs)

    assert len(sources) == 3
    assert sources == ["api", "database", "scheduler"]  # Sorted


def test_format_timestamp():
    """Test _format_timestamp formats correctly."""
    from services.dashboard.components.log_viewer import _format_timestamp

    dt = datetime(2026, 1, 15, 14, 30, 45)
    result = _format_timestamp(dt)

    assert result == "2026-01-15 14:30:45"


def test_format_relative_time_seconds():
    """Test _format_relative_time for seconds."""
    from services.dashboard.components.log_viewer import _format_relative_time

    ref = datetime(2026, 1, 15, 12, 0, 0)
    dt = ref - timedelta(seconds=30)
    result = _format_relative_time(dt, reference=ref)

    assert result == "30s ago"


def test_format_relative_time_minutes():
    """Test _format_relative_time for minutes."""
    from services.dashboard.components.log_viewer import _format_relative_time

    ref = datetime(2026, 1, 15, 12, 0, 0)
    dt = ref - timedelta(minutes=5)
    result = _format_relative_time(dt, reference=ref)

    assert result == "5m ago"


def test_format_relative_time_hours():
    """Test _format_relative_time for hours."""
    from services.dashboard.components.log_viewer import _format_relative_time

    ref = datetime(2026, 1, 15, 12, 0, 0)
    dt = ref - timedelta(hours=3)
    result = _format_relative_time(dt, reference=ref)

    assert result == "3h ago"


def test_format_relative_time_days():
    """Test _format_relative_time for days."""
    from services.dashboard.components.log_viewer import _format_relative_time

    ref = datetime(2026, 1, 15, 12, 0, 0)
    dt = ref - timedelta(days=2)
    result = _format_relative_time(dt, reference=ref)

    assert result == "2d ago"


def test_format_relative_time_future():
    """Test _format_relative_time for future timestamps."""
    from services.dashboard.components.log_viewer import _format_relative_time

    ref = datetime(2026, 1, 15, 12, 0, 0)
    dt = ref + timedelta(hours=1)
    result = _format_relative_time(dt, reference=ref)

    assert result == "in the future"


def test_count_by_level():
    """Test _count_by_level counts correctly."""
    from services.dashboard.components.log_viewer import (
        _count_by_level,
        LogEntry,
        LogLevel,
    )

    logs = [
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="test",
            message="Msg",
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="test",
            message="Msg",
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            source="test",
            message="Msg",
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            source="test",
            message="Msg",
        ),
    ]

    counts = _count_by_level(logs)

    assert counts[LogLevel.INFO] == 2
    assert counts[LogLevel.ERROR] == 1
    assert counts[LogLevel.WARNING] == 1
    assert counts[LogLevel.DEBUG] == 0
    assert counts[LogLevel.CRITICAL] == 0


def test_get_log_export_data():
    """Test get_log_export_data returns valid structure."""
    from services.dashboard.components.log_viewer import get_log_export_data

    data = get_log_export_data()

    assert "timestamp" in data
    assert "total_count" in data
    assert "counts_by_level" in data
    assert "entries" in data
    assert isinstance(data["entries"], list)


def test_get_log_export_data_entry_structure():
    """Test get_log_export_data entry structure."""
    from services.dashboard.components.log_viewer import get_log_export_data

    data = get_log_export_data()
    entries = data["entries"]

    assert len(entries) > 0
    for entry in entries:
        assert "timestamp" in entry
        assert "level" in entry
        assert "source" in entry
        assert "message" in entry


def test_get_log_export_data_with_filter():
    """Test get_log_export_data with filter applied."""
    from services.dashboard.components.log_viewer import (
        get_log_export_data,
        LogFilter,
        LogLevel,
    )

    log_filter = LogFilter(levels=[LogLevel.ERROR, LogLevel.CRITICAL])
    data = get_log_export_data(log_filter=log_filter)

    # All entries should be ERROR or CRITICAL
    for entry in data["entries"]:
        assert entry["level"] in ["ERROR", "CRITICAL"]


def test_filter_logs_keyword_in_source():
    """Test _filter_logs finds keyword in source as well as message."""
    from services.dashboard.components.log_viewer import (
        _filter_logs,
        LogEntry,
        LogFilter,
        LogLevel,
    )

    logs = [
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="api.upbit",
            message="Connected",
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="scheduler",
            message="API call completed",
        ),
    ]

    # Search for "api" - should find both (one in source, one in message)
    log_filter = LogFilter(keyword="api")
    result = _filter_logs(logs, log_filter)

    assert len(result) == 2


def test_filter_logs_case_insensitive():
    """Test _filter_logs is case insensitive."""
    from services.dashboard.components.log_viewer import (
        _filter_logs,
        LogEntry,
        LogFilter,
        LogLevel,
    )

    logs = [
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source="API",
            message="ERROR occurred",
        ),
    ]

    # Search with different case
    log_filter = LogFilter(source="api", keyword="error")
    result = _filter_logs(logs, log_filter)

    assert len(result) == 1


def test_count_by_level_empty():
    """Test _count_by_level with empty list."""
    from services.dashboard.components.log_viewer import _count_by_level, LogLevel

    counts = _count_by_level([])

    assert all(count == 0 for count in counts.values())
    assert len(counts) == 5  # All levels present with 0 count


def test_format_relative_time_timezone_aware():
    """Test _format_relative_time handles timezone-aware datetimes."""
    from datetime import timezone
    from services.dashboard.components.log_viewer import _format_relative_time

    # Create timezone-aware datetime
    dt_aware = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    ref_aware = datetime(2026, 1, 15, 12, 5, 0, tzinfo=timezone.utc)

    result = _format_relative_time(dt_aware, reference=ref_aware)
    assert result == "5m ago"


def test_format_relative_time_mixed_timezone():
    """Test _format_relative_time handles mixed tz-awareness gracefully."""
    from datetime import timezone
    from services.dashboard.components.log_viewer import _format_relative_time

    # Naive reference, timezone-aware input - should not crash
    dt_aware = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    ref_naive = datetime(2026, 1, 15, 12, 5, 0)

    # Should not raise TypeError
    result = _format_relative_time(dt_aware, reference=ref_naive)
    assert "ago" in result or result == "in the future"


def test_safe_datetime_compare():
    """Test _safe_datetime_compare handles various cases."""
    from datetime import timezone
    from services.dashboard.components.log_viewer import _safe_datetime_compare

    dt1 = datetime(2026, 1, 15, 12, 0, 0)
    dt2 = datetime(2026, 1, 15, 13, 0, 0)

    assert _safe_datetime_compare(dt1, dt2) == -1
    assert _safe_datetime_compare(dt2, dt1) == 1
    assert _safe_datetime_compare(dt1, dt1) == 0


def test_safe_datetime_compare_mixed_tz():
    """Test _safe_datetime_compare with mixed timezone-awareness."""
    from datetime import timezone
    from services.dashboard.components.log_viewer import _safe_datetime_compare

    dt_naive = datetime(2026, 1, 15, 12, 0, 0)
    dt_aware = datetime(2026, 1, 15, 13, 0, 0, tzinfo=timezone.utc)

    # Should not crash, should return comparison result
    result = _safe_datetime_compare(dt_naive, dt_aware)
    assert result in [-1, 0, 1]


def test_filter_logs_timezone_aware_time_range():
    """Test _filter_logs with timezone-aware time range."""
    from datetime import timezone
    from services.dashboard.components.log_viewer import (
        _filter_logs,
        LogEntry,
        LogFilter,
        LogLevel,
    )

    base_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    logs = [
        LogEntry(
            timestamp=base_time - timedelta(hours=1),
            level=LogLevel.INFO,
            source="test",
            message="Old message",
        ),
        LogEntry(
            timestamp=base_time,
            level=LogLevel.INFO,
            source="test",
            message="Current message",
        ),
    ]

    # Filter with timezone-aware boundaries
    log_filter = LogFilter(
        start_time=base_time - timedelta(minutes=30),
        end_time=base_time + timedelta(minutes=30),
    )

    result = _filter_logs(logs, log_filter)
    assert len(result) == 1
    assert result[0].message == "Current message"


def test_sorting_mixed_timezone_logs():
    """Test that sorting handles mixed naive and tz-aware logs."""
    from datetime import timezone
    from services.dashboard.components.log_viewer import (
        _filter_logs,
        LogEntry,
        LogFilter,
        LogLevel,
    )

    # Mix of naive and tz-aware timestamps
    logs = [
        LogEntry(
            timestamp=datetime(2026, 1, 15, 10, 0, 0),  # naive
            level=LogLevel.INFO,
            source="test",
            message="Naive early",
        ),
        LogEntry(
            timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),  # aware
            level=LogLevel.INFO,
            source="test",
            message="Aware middle",
        ),
        LogEntry(
            timestamp=datetime(2026, 1, 15, 14, 0, 0),  # naive
            level=LogLevel.INFO,
            source="test",
            message="Naive late",
        ),
    ]

    # The _safe_sort_key in render_log_viewer handles this
    # Here we test that the list can be sorted without TypeError
    def _safe_sort_key(entry):
        ts = entry.timestamp
        return ts.replace(tzinfo=None) if ts.tzinfo else ts

    # Should not raise TypeError
    sorted_logs = sorted(logs, key=_safe_sort_key, reverse=True)

    assert len(sorted_logs) == 3
    assert sorted_logs[0].message == "Naive late"
    assert sorted_logs[1].message == "Aware middle"
    assert sorted_logs[2].message == "Naive early"


def test_strip_tzinfo_naive():
    """Test _strip_tzinfo with naive datetime returns unchanged."""
    from services.dashboard.components.log_viewer import _strip_tzinfo

    naive = datetime(2026, 1, 15, 12, 0, 0)
    result = _strip_tzinfo(naive)

    assert result == naive
    assert result.tzinfo is None


def test_strip_tzinfo_aware():
    """Test _strip_tzinfo with aware datetime strips tzinfo."""
    from services.dashboard.components.log_viewer import _strip_tzinfo
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Asia/Seoul")
    aware = datetime(2026, 1, 15, 12, 0, 0, tzinfo=tz)
    result = _strip_tzinfo(aware)

    # Same time values, tzinfo stripped
    assert result.hour == 12  # Hour preserved (not converted to UTC)
    assert result.tzinfo is None


def test_safe_datetime_compare_aware_aware_correct():
    """Test that two aware datetimes are compared by absolute UTC time."""
    from services.dashboard.components.log_viewer import _safe_datetime_compare
    from zoneinfo import ZoneInfo

    utc = ZoneInfo("UTC")
    kst = ZoneInfo("Asia/Seoul")  # UTC+9

    # Same absolute time: 03:00 UTC = 12:00 KST
    utc_time = datetime(2026, 1, 15, 3, 0, 0, tzinfo=utc)
    kst_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=kst)

    # Python compares aware datetimes by absolute UTC time
    result = _safe_datetime_compare(utc_time, kst_time)
    assert result == 0  # Equal


def test_safe_datetime_compare_aware_different_times():
    """Test aware datetimes with different absolute times."""
    from services.dashboard.components.log_viewer import _safe_datetime_compare
    from zoneinfo import ZoneInfo

    utc = ZoneInfo("UTC")
    kst = ZoneInfo("Asia/Seoul")  # UTC+9

    # 04:00 UTC > 03:00 UTC (12:00 KST)
    utc_later = datetime(2026, 1, 15, 4, 0, 0, tzinfo=utc)
    kst_earlier = datetime(2026, 1, 15, 12, 0, 0, tzinfo=kst)  # = 03:00 UTC

    result = _safe_datetime_compare(utc_later, kst_earlier)
    assert result == 1  # utc_later > kst_earlier
