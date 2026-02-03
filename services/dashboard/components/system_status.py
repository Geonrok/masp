"""System status component for monitoring system health."""

from __future__ import annotations

import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

# Session state key prefix
_KEY_PREFIX = "system_status."

# Application start time for uptime calculation
_APP_START_TIME: Optional[datetime] = None


class ServiceStatus(str, Enum):
    """Service status enumeration."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


@dataclass
class ResourceUsage:
    """System resource usage data."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0


@dataclass
class ServiceHealth:
    """Service health status."""

    name: str
    status: ServiceStatus
    latency_ms: Optional[float] = None
    message: str = ""
    last_check: Optional[datetime] = None


@dataclass
class SystemInfo:
    """System information."""

    app_version: str = "0.1.0"
    python_version: str = ""
    platform: str = ""
    hostname: str = ""
    uptime: Optional[timedelta] = None
    start_time: Optional[datetime] = None


def _key(name: str) -> str:
    """Generate namespaced session state key."""
    return f"{_KEY_PREFIX}{name}"


def _get_app_start_time() -> datetime:
    """Get or initialize application start time."""
    global _APP_START_TIME
    if _APP_START_TIME is None:
        _APP_START_TIME = datetime.now()
    return _APP_START_TIME


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def _get_demo_resource_usage() -> ResourceUsage:
    """Get demo resource usage data (deterministic)."""
    return ResourceUsage(
        cpu_percent=35.5,
        memory_percent=62.3,
        memory_used_mb=4096.0,
        memory_total_mb=6576.0,
        disk_percent=45.8,
        disk_used_gb=234.5,
        disk_total_gb=512.0,
    )


def _get_demo_services() -> List[ServiceHealth]:
    """Get demo service health data (deterministic)."""
    now = datetime.now()
    return [
        ServiceHealth(
            name="Upbit API",
            status=ServiceStatus.HEALTHY,
            latency_ms=45.2,
            message="Connected",
            last_check=now,
        ),
        ServiceHealth(
            name="Database",
            status=ServiceStatus.HEALTHY,
            latency_ms=12.5,
            message="SQLite OK",
            last_check=now,
        ),
        ServiceHealth(
            name="Scheduler",
            status=ServiceStatus.HEALTHY,
            latency_ms=None,
            message="Running (3 jobs)",
            last_check=now,
        ),
        ServiceHealth(
            name="Paper Trading",
            status=ServiceStatus.HEALTHY,
            latency_ms=None,
            message="Active",
            last_check=now,
        ),
    ]


def _get_system_info() -> SystemInfo:
    """Get system information."""
    start_time = _get_app_start_time()
    uptime = datetime.now() - start_time

    return SystemInfo(
        app_version="0.1.0",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=platform.system(),
        hostname=platform.node() or "unknown",
        uptime=uptime,
        start_time=start_time,
    )


def _format_uptime(td: Optional[timedelta]) -> str:
    """Format timedelta as human-readable uptime string."""
    if td is None:
        return "Unknown"

    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        return "Unknown"

    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")

    return " ".join(parts)


def _format_bytes(mb: float, use_gb: bool = False) -> str:
    """Format bytes as human-readable string."""
    if use_gb:
        return f"{mb:.1f} GB"
    return f"{mb:.0f} MB"


def _get_status_color(status: ServiceStatus) -> str:
    """Get color for service status."""
    colors = {
        ServiceStatus.HEALTHY: "green",
        ServiceStatus.DEGRADED: "orange",
        ServiceStatus.UNHEALTHY: "red",
        ServiceStatus.UNKNOWN: "gray",
    }
    return colors.get(status, "gray")


def _get_status_indicator(status: ServiceStatus) -> str:
    """Get text indicator for service status."""
    indicators = {
        ServiceStatus.HEALTHY: "[OK]",
        ServiceStatus.DEGRADED: "[WARN]",
        ServiceStatus.UNHEALTHY: "[ERR]",
        ServiceStatus.UNKNOWN: "[?]",
    }
    return indicators.get(status, "[?]")


def _calculate_overall_status(services: List[ServiceHealth]) -> ServiceStatus:
    """Calculate overall system status from service statuses."""
    if not services:
        return ServiceStatus.UNKNOWN

    statuses = [s.status for s in services]

    if all(s == ServiceStatus.HEALTHY for s in statuses):
        return ServiceStatus.HEALTHY
    if any(s == ServiceStatus.UNHEALTHY for s in statuses):
        return ServiceStatus.UNHEALTHY
    if any(s == ServiceStatus.DEGRADED for s in statuses):
        return ServiceStatus.DEGRADED
    return ServiceStatus.UNKNOWN


def _get_level_indicator(
    value: float, warning: float = 70.0, critical: float = 90.0
) -> str:
    """Get level indicator based on threshold."""
    if value >= critical:
        return "[HIGH]"
    elif value >= warning:
        return "[MED]"
    return "[LOW]"


def _render_resource_gauge(
    label: str,
    value: float,
    max_value: float = 100.0,
    warning_threshold: float = 70.0,
    critical_threshold: float = 90.0,
    suffix: str = "%",
) -> None:
    """Render a resource usage gauge using Streamlit progress bar."""
    # Normalize and clamp value to 0-100 range
    clamped_value = _clamp(value, 0.0, max_value)
    normalized = clamped_value / max_value * 100 if max_value > 0 else 0.0

    # Determine indicator based on thresholds
    indicator = _get_level_indicator(normalized, warning_threshold, critical_threshold)

    st.markdown(f"**{label}** {indicator}")
    st.progress(normalized / 100.0)
    st.caption(f"{clamped_value:.1f}{suffix}")


def render_system_status(
    resource_provider: Optional[Callable[[], ResourceUsage]] = None,
    service_provider: Optional[Callable[[], List[ServiceHealth]]] = None,
    show_resources: bool = True,
    show_services: bool = True,
    show_system_info: bool = True,
    compact: bool = False,
) -> None:
    """Render system status panel.

    Args:
        resource_provider: Function to get resource usage data
        service_provider: Function to get service health data
        show_resources: Whether to show resource usage section
        show_services: Whether to show service status section
        show_system_info: Whether to show system info section
        compact: Whether to use compact layout
    """
    st.subheader("System Status")

    # Get data
    is_demo = resource_provider is None and service_provider is None
    if is_demo:
        st.caption("Demo Mode")

    resources = (
        resource_provider()
        if resource_provider is not None
        else _get_demo_resource_usage()
    )
    services = (
        service_provider() if service_provider is not None else _get_demo_services()
    )
    system_info = _get_system_info()

    # Overall status
    overall_status = _calculate_overall_status(services)
    status_emoji = _get_status_indicator(overall_status)
    st.markdown(f"### {status_emoji} System {overall_status.value}")

    if show_system_info:
        # System info row
        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Version", system_info.app_version)
        with info_cols[1]:
            st.metric("Python", system_info.python_version)
        with info_cols[2]:
            st.metric("Platform", system_info.platform)
        with info_cols[3]:
            st.metric("Uptime", _format_uptime(system_info.uptime))

    if show_resources:
        st.divider()
        st.markdown("**Resource Usage**")

        if compact:
            # Compact: single row with text indicators
            res_cols = st.columns(3)
            with res_cols[0]:
                cpu_ind = _get_level_indicator(resources.cpu_percent)
                st.metric(
                    f"CPU {cpu_ind}", f"{_clamp(resources.cpu_percent, 0, 100):.1f}%"
                )
            with res_cols[1]:
                mem_ind = _get_level_indicator(resources.memory_percent)
                st.metric(
                    f"Memory {mem_ind}",
                    f"{_clamp(resources.memory_percent, 0, 100):.1f}%",
                )
            with res_cols[2]:
                disk_ind = _get_level_indicator(resources.disk_percent)
                st.metric(
                    f"Disk {disk_ind}", f"{_clamp(resources.disk_percent, 0, 100):.1f}%"
                )
        else:
            # Full: with progress bars
            res_cols = st.columns(3)
            with res_cols[0]:
                _render_resource_gauge("CPU", resources.cpu_percent)
            with res_cols[1]:
                _render_resource_gauge("Memory", resources.memory_percent)
                st.caption(
                    f"{_format_bytes(resources.memory_used_mb)} / {_format_bytes(resources.memory_total_mb)}"
                )
            with res_cols[2]:
                _render_resource_gauge("Disk", resources.disk_percent)
                st.caption(
                    f"{_format_bytes(resources.disk_used_gb, use_gb=True)} / {_format_bytes(resources.disk_total_gb, use_gb=True)}"
                )

    if show_services:
        st.divider()
        st.markdown("**Service Status**")

        if compact:
            # Compact: inline badges
            service_text = " | ".join(
                f"{_get_status_indicator(s.status)} {s.name}" for s in services
            )
            st.markdown(service_text)
        else:
            # Full: table-like layout
            for service in services:
                with st.container():
                    cols = st.columns([3, 2, 3, 2])
                    with cols[0]:
                        st.markdown(
                            f"{_get_status_indicator(service.status)} **{service.name}**"
                        )
                    with cols[1]:
                        st.markdown(f"`{service.status.value}`")
                    with cols[2]:
                        st.caption(service.message)
                    with cols[3]:
                        if service.latency_ms is not None:
                            st.caption(f"{service.latency_ms:.1f}ms")
                        else:
                            st.caption("-")


def get_system_health_summary() -> Dict[str, Any]:
    """Get system health summary as dictionary (for API/export).

    Returns:
        Dict containing system health data
    """
    resources = _get_demo_resource_usage()
    services = _get_demo_services()
    system_info = _get_system_info()
    overall_status = _calculate_overall_status(services)

    return {
        "status": overall_status.value,
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "app_version": system_info.app_version,
            "python_version": system_info.python_version,
            "platform": system_info.platform,
            "hostname": system_info.hostname,
            "uptime_seconds": (
                int(system_info.uptime.total_seconds()) if system_info.uptime else None
            ),
        },
        "resources": {
            "cpu_percent": resources.cpu_percent,
            "memory_percent": resources.memory_percent,
            "memory_used_mb": resources.memory_used_mb,
            "memory_total_mb": resources.memory_total_mb,
            "disk_percent": resources.disk_percent,
            "disk_used_gb": resources.disk_used_gb,
            "disk_total_gb": resources.disk_total_gb,
        },
        "services": [
            {
                "name": s.name,
                "status": s.status.value,
                "latency_ms": s.latency_ms,
                "message": s.message,
            }
            for s in services
        ],
    }
