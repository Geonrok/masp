"""Tests for system status component."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest


def test_import_system_status():
    """Test module imports correctly."""
    from services.dashboard.components import system_status

    assert hasattr(system_status, "render_system_status")
    assert hasattr(system_status, "ServiceStatus")
    assert hasattr(system_status, "ResourceUsage")
    assert hasattr(system_status, "ServiceHealth")
    assert hasattr(system_status, "SystemInfo")


def test_service_status_enum():
    """Test ServiceStatus enum values."""
    from services.dashboard.components.system_status import ServiceStatus

    assert ServiceStatus.HEALTHY.value == "HEALTHY"
    assert ServiceStatus.DEGRADED.value == "DEGRADED"
    assert ServiceStatus.UNHEALTHY.value == "UNHEALTHY"
    assert ServiceStatus.UNKNOWN.value == "UNKNOWN"


def test_resource_usage_dataclass():
    """Test ResourceUsage dataclass defaults."""
    from services.dashboard.components.system_status import ResourceUsage

    usage = ResourceUsage()
    assert usage.cpu_percent == 0.0
    assert usage.memory_percent == 0.0
    assert usage.disk_percent == 0.0


def test_resource_usage_with_values():
    """Test ResourceUsage dataclass with values."""
    from services.dashboard.components.system_status import ResourceUsage

    usage = ResourceUsage(
        cpu_percent=50.0,
        memory_percent=75.0,
        memory_used_mb=8192.0,
        memory_total_mb=16384.0,
        disk_percent=60.0,
        disk_used_gb=500.0,
        disk_total_gb=1000.0,
    )

    assert usage.cpu_percent == 50.0
    assert usage.memory_percent == 75.0
    assert usage.memory_used_mb == 8192.0


def test_service_health_dataclass():
    """Test ServiceHealth dataclass."""
    from services.dashboard.components.system_status import (
        ServiceHealth,
        ServiceStatus,
    )

    health = ServiceHealth(
        name="Test Service",
        status=ServiceStatus.HEALTHY,
        latency_ms=25.5,
        message="OK",
        last_check=datetime.now(),
    )

    assert health.name == "Test Service"
    assert health.status == ServiceStatus.HEALTHY
    assert health.latency_ms == 25.5


def test_service_health_defaults():
    """Test ServiceHealth dataclass defaults."""
    from services.dashboard.components.system_status import (
        ServiceHealth,
        ServiceStatus,
    )

    health = ServiceHealth(
        name="Minimal",
        status=ServiceStatus.UNKNOWN,
    )

    assert health.latency_ms is None
    assert health.message == ""
    assert health.last_check is None


def test_system_info_dataclass():
    """Test SystemInfo dataclass."""
    from services.dashboard.components.system_status import SystemInfo

    info = SystemInfo(
        app_version="1.0.0",
        python_version="3.11.0",
        platform="Windows",
        hostname="test-host",
        uptime=timedelta(hours=5),
        start_time=datetime.now(),
    )

    assert info.app_version == "1.0.0"
    assert info.python_version == "3.11.0"
    assert info.platform == "Windows"


def test_key_function():
    """Test _key generates namespaced keys."""
    from services.dashboard.components.system_status import _key, _KEY_PREFIX

    result = _key("resources")
    assert result == f"{_KEY_PREFIX}resources"
    assert result == "system_status.resources"


def test_safe_float_valid():
    """Test _safe_float with valid values."""
    from services.dashboard.components.system_status import _safe_float

    assert _safe_float(1.5) == 1.5
    assert _safe_float(0) == 0.0
    assert _safe_float("3.14") == 3.14


def test_safe_float_invalid():
    """Test _safe_float with invalid values."""
    from services.dashboard.components.system_status import _safe_float

    assert _safe_float(None) == 0.0
    assert _safe_float("invalid") == 0.0
    assert _safe_float(float("inf")) == 0.0
    assert _safe_float(float("nan")) == 0.0


def test_clamp():
    """Test _clamp function."""
    from services.dashboard.components.system_status import _clamp

    assert _clamp(50.0, 0.0, 100.0) == 50.0
    assert _clamp(-10.0, 0.0, 100.0) == 0.0
    assert _clamp(150.0, 0.0, 100.0) == 100.0
    assert _clamp(0.0, 0.0, 100.0) == 0.0
    assert _clamp(100.0, 0.0, 100.0) == 100.0


def test_get_demo_resource_usage():
    """Test _get_demo_resource_usage returns deterministic data."""
    from services.dashboard.components.system_status import _get_demo_resource_usage

    usage1 = _get_demo_resource_usage()
    usage2 = _get_demo_resource_usage()

    assert usage1.cpu_percent == usage2.cpu_percent
    assert usage1.memory_percent == usage2.memory_percent
    assert usage1.disk_percent == usage2.disk_percent
    assert usage1.cpu_percent == 35.5


def test_get_demo_services():
    """Test _get_demo_services returns expected services."""
    from services.dashboard.components.system_status import (
        _get_demo_services,
        ServiceStatus,
    )

    services = _get_demo_services()

    assert len(services) >= 3
    assert any(s.name == "Upbit API" for s in services)
    assert any(s.name == "Database" for s in services)
    assert all(s.status == ServiceStatus.HEALTHY for s in services)


def test_get_system_info():
    """Test _get_system_info returns valid info."""
    from services.dashboard.components.system_status import _get_system_info

    info = _get_system_info()

    assert info.app_version == "0.1.0"
    assert info.python_version != ""
    assert info.platform != ""
    assert info.uptime is not None
    assert info.start_time is not None


def test_format_uptime_seconds():
    """Test _format_uptime with seconds only."""
    from services.dashboard.components.system_status import _format_uptime

    result = _format_uptime(timedelta(seconds=45))
    assert "45s" in result


def test_format_uptime_minutes():
    """Test _format_uptime with minutes."""
    from services.dashboard.components.system_status import _format_uptime

    result = _format_uptime(timedelta(minutes=5, seconds=30))
    assert "5m" in result
    assert "30s" in result


def test_format_uptime_hours():
    """Test _format_uptime with hours."""
    from services.dashboard.components.system_status import _format_uptime

    result = _format_uptime(timedelta(hours=2, minutes=15))
    assert "2h" in result
    assert "15m" in result


def test_format_uptime_days():
    """Test _format_uptime with days."""
    from services.dashboard.components.system_status import _format_uptime

    result = _format_uptime(timedelta(days=3, hours=12))
    assert "3d" in result
    assert "12h" in result


def test_format_uptime_none():
    """Test _format_uptime with None."""
    from services.dashboard.components.system_status import _format_uptime

    result = _format_uptime(None)
    assert result == "Unknown"


def test_format_uptime_negative():
    """Test _format_uptime with negative timedelta."""
    from services.dashboard.components.system_status import _format_uptime

    result = _format_uptime(timedelta(seconds=-10))
    assert result == "Unknown"


def test_format_bytes_mb():
    """Test _format_bytes in MB."""
    from services.dashboard.components.system_status import _format_bytes

    result = _format_bytes(4096.0)
    assert "4096 MB" in result


def test_format_bytes_gb():
    """Test _format_bytes in GB."""
    from services.dashboard.components.system_status import _format_bytes

    result = _format_bytes(512.5, use_gb=True)
    assert "512.5 GB" in result


def test_get_status_color():
    """Test _get_status_color returns correct colors."""
    from services.dashboard.components.system_status import (
        _get_status_color,
        ServiceStatus,
    )

    assert _get_status_color(ServiceStatus.HEALTHY) == "green"
    assert _get_status_color(ServiceStatus.DEGRADED) == "orange"
    assert _get_status_color(ServiceStatus.UNHEALTHY) == "red"
    assert _get_status_color(ServiceStatus.UNKNOWN) == "gray"


def test_get_status_indicator():
    """Test _get_status_indicator returns correct text indicators."""
    from services.dashboard.components.system_status import (
        _get_status_indicator,
        ServiceStatus,
    )

    assert _get_status_indicator(ServiceStatus.HEALTHY) == "[OK]"
    assert _get_status_indicator(ServiceStatus.DEGRADED) == "[WARN]"
    assert _get_status_indicator(ServiceStatus.UNHEALTHY) == "[ERR]"
    assert _get_status_indicator(ServiceStatus.UNKNOWN) == "[?]"


def test_calculate_overall_status_all_healthy():
    """Test _calculate_overall_status with all healthy services."""
    from services.dashboard.components.system_status import (
        _calculate_overall_status,
        ServiceHealth,
        ServiceStatus,
    )

    services = [
        ServiceHealth(name="A", status=ServiceStatus.HEALTHY),
        ServiceHealth(name="B", status=ServiceStatus.HEALTHY),
        ServiceHealth(name="C", status=ServiceStatus.HEALTHY),
    ]

    result = _calculate_overall_status(services)
    assert result == ServiceStatus.HEALTHY


def test_calculate_overall_status_one_unhealthy():
    """Test _calculate_overall_status with one unhealthy service."""
    from services.dashboard.components.system_status import (
        _calculate_overall_status,
        ServiceHealth,
        ServiceStatus,
    )

    services = [
        ServiceHealth(name="A", status=ServiceStatus.HEALTHY),
        ServiceHealth(name="B", status=ServiceStatus.UNHEALTHY),
        ServiceHealth(name="C", status=ServiceStatus.HEALTHY),
    ]

    result = _calculate_overall_status(services)
    assert result == ServiceStatus.UNHEALTHY


def test_calculate_overall_status_one_degraded():
    """Test _calculate_overall_status with one degraded service."""
    from services.dashboard.components.system_status import (
        _calculate_overall_status,
        ServiceHealth,
        ServiceStatus,
    )

    services = [
        ServiceHealth(name="A", status=ServiceStatus.HEALTHY),
        ServiceHealth(name="B", status=ServiceStatus.DEGRADED),
        ServiceHealth(name="C", status=ServiceStatus.HEALTHY),
    ]

    result = _calculate_overall_status(services)
    assert result == ServiceStatus.DEGRADED


def test_calculate_overall_status_empty():
    """Test _calculate_overall_status with empty list."""
    from services.dashboard.components.system_status import (
        _calculate_overall_status,
        ServiceStatus,
    )

    result = _calculate_overall_status([])
    assert result == ServiceStatus.UNKNOWN


def test_calculate_overall_status_all_unknown():
    """Test _calculate_overall_status with all unknown services."""
    from services.dashboard.components.system_status import (
        _calculate_overall_status,
        ServiceHealth,
        ServiceStatus,
    )

    services = [
        ServiceHealth(name="A", status=ServiceStatus.UNKNOWN),
        ServiceHealth(name="B", status=ServiceStatus.UNKNOWN),
    ]

    result = _calculate_overall_status(services)
    assert result == ServiceStatus.UNKNOWN


def test_get_system_health_summary():
    """Test get_system_health_summary returns valid structure."""
    from services.dashboard.components.system_status import get_system_health_summary

    summary = get_system_health_summary()

    assert "status" in summary
    assert "timestamp" in summary
    assert "system_info" in summary
    assert "resources" in summary
    assert "services" in summary

    # Verify nested structure
    assert "app_version" in summary["system_info"]
    assert "cpu_percent" in summary["resources"]
    assert isinstance(summary["services"], list)


def test_get_system_health_summary_services():
    """Test get_system_health_summary services have correct structure."""
    from services.dashboard.components.system_status import get_system_health_summary

    summary = get_system_health_summary()
    services = summary["services"]

    assert len(services) > 0
    for service in services:
        assert "name" in service
        assert "status" in service
        assert "latency_ms" in service
        assert "message" in service


def test_get_app_start_time_consistent():
    """Test _get_app_start_time returns consistent value."""
    from services.dashboard.components.system_status import _get_app_start_time

    time1 = _get_app_start_time()
    time2 = _get_app_start_time()

    assert time1 == time2


def test_service_health_with_all_fields():
    """Test ServiceHealth with all fields populated."""
    from services.dashboard.components.system_status import (
        ServiceHealth,
        ServiceStatus,
    )

    now = datetime.now()
    health = ServiceHealth(
        name="Full Service",
        status=ServiceStatus.DEGRADED,
        latency_ms=100.5,
        message="High latency detected",
        last_check=now,
    )

    assert health.name == "Full Service"
    assert health.status == ServiceStatus.DEGRADED
    assert health.latency_ms == 100.5
    assert health.message == "High latency detected"
    assert health.last_check == now


def test_resource_usage_boundary_values():
    """Test ResourceUsage with boundary values."""
    from services.dashboard.components.system_status import ResourceUsage

    # Test 100% values
    usage = ResourceUsage(
        cpu_percent=100.0,
        memory_percent=100.0,
        disk_percent=100.0,
    )

    assert usage.cpu_percent == 100.0
    assert usage.memory_percent == 100.0
    assert usage.disk_percent == 100.0


def test_clamp_edge_cases():
    """Test _clamp with edge cases."""
    from services.dashboard.components.system_status import _clamp

    # Exact boundaries
    assert _clamp(0.0, 0.0, 100.0) == 0.0
    assert _clamp(100.0, 0.0, 100.0) == 100.0

    # Very small range
    assert _clamp(5.0, 4.9, 5.1) == 5.0
    assert _clamp(4.8, 4.9, 5.1) == 4.9
    assert _clamp(5.2, 4.9, 5.1) == 5.1


def test_get_level_indicator():
    """Test _get_level_indicator returns correct indicators."""
    from services.dashboard.components.system_status import _get_level_indicator

    # Below warning threshold
    assert _get_level_indicator(50.0) == "[LOW]"
    assert _get_level_indicator(0.0) == "[LOW]"
    assert _get_level_indicator(69.9) == "[LOW]"

    # Warning level
    assert _get_level_indicator(70.0) == "[MED]"
    assert _get_level_indicator(85.0) == "[MED]"
    assert _get_level_indicator(89.9) == "[MED]"

    # Critical level
    assert _get_level_indicator(90.0) == "[HIGH]"
    assert _get_level_indicator(100.0) == "[HIGH]"


def test_get_level_indicator_custom_thresholds():
    """Test _get_level_indicator with custom thresholds."""
    from services.dashboard.components.system_status import _get_level_indicator

    # Custom thresholds: warning=50, critical=80
    assert _get_level_indicator(40.0, warning=50.0, critical=80.0) == "[LOW]"
    assert _get_level_indicator(60.0, warning=50.0, critical=80.0) == "[MED]"
    assert _get_level_indicator(85.0, warning=50.0, critical=80.0) == "[HIGH]"
