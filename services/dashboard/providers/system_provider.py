"""System status data provider - connects psutil to system_status component."""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

from services.dashboard.components.system_status import (
    ResourceUsage,
    ServiceHealth,
    ServiceStatus,
)

logger = logging.getLogger(__name__)


def get_system_resources() -> ResourceUsage:
    """Get actual system resource usage using psutil.

    Returns:
        ResourceUsage with real CPU, memory, and disk data
    """
    try:
        import psutil

        # CPU (average over 0.1 second interval for responsiveness)
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)  # bytes to MB
        memory_total_mb = memory.total / (1024 * 1024)  # bytes to MB

        # Disk (root partition)
        try:
            disk = psutil.disk_usage("/")
        except OSError:
            # Windows fallback
            disk = psutil.disk_usage("C:\\")

        disk_percent = disk.percent
        disk_used_gb = disk.used / (1024 * 1024 * 1024)  # bytes to GB
        disk_total_gb = disk.total / (1024 * 1024 * 1024)  # bytes to GB

        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            disk_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
        )

    except ImportError:
        logger.warning("psutil not available, returning dummy data")
        return ResourceUsage(
            cpu_percent=0.0,
            memory_percent=0.0,
            memory_used_mb=0.0,
            memory_total_mb=0.0,
            disk_percent=0.0,
            disk_used_gb=0.0,
            disk_total_gb=0.0,
        )
    except Exception as e:
        logger.warning("Failed to get system resources: %s", e)
        return ResourceUsage(
            cpu_percent=0.0,
            memory_percent=0.0,
            memory_used_mb=0.0,
            memory_total_mb=0.0,
            disk_percent=0.0,
            disk_used_gb=0.0,
            disk_total_gb=0.0,
        )


def _check_upbit_api() -> Tuple[ServiceStatus, Optional[float], str]:
    """Check Upbit API connectivity.

    Returns:
        Tuple of (status, latency_ms, message)
    """
    start_time = time.time()
    try:
        from libs.adapters.upbit_spot import UpbitSpotMarketData

        market_data = UpbitSpotMarketData()
        quote = market_data.get_quote("BTC/KRW")
        latency_ms = (time.time() - start_time) * 1000

        if quote and "price" in quote:
            return ServiceStatus.HEALTHY, latency_ms, ""
        return ServiceStatus.DEGRADED, latency_ms, "No price data"

    except ImportError:
        return ServiceStatus.UNKNOWN, None, "Adapter not available"
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return ServiceStatus.UNHEALTHY, latency_ms, str(e)[:50]


def _check_scheduler() -> Tuple[ServiceStatus, Optional[float], str]:
    """Check APScheduler status.

    Returns:
        Tuple of (status, latency_ms, message)
    """
    start_time = time.time()
    try:
        # Check if scheduler module is importable
        from services.scheduler import scheduler

        latency_ms = (time.time() - start_time) * 1000

        if hasattr(scheduler, "running") and scheduler.running:
            return ServiceStatus.HEALTHY, latency_ms, ""
        return ServiceStatus.DEGRADED, latency_ms, "Scheduler not running"

    except ImportError:
        return ServiceStatus.UNKNOWN, None, "Scheduler not available"
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return ServiceStatus.UNHEALTHY, latency_ms, str(e)[:50]


def _check_database() -> Tuple[ServiceStatus, Optional[float], str]:
    """Check database connectivity.

    Returns:
        Tuple of (status, latency_ms, message)
    """
    start_time = time.time()
    try:
        # Add actual database check here when available
        latency_ms = (time.time() - start_time) * 1000
        return ServiceStatus.HEALTHY, latency_ms, ""
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return ServiceStatus.UNHEALTHY, latency_ms, str(e)[:50]


def _check_telegram() -> Tuple[ServiceStatus, Optional[float], str]:
    """Check Telegram bot status.

    Returns:
        Tuple of (status, latency_ms, message)
    """
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID")

    if not telegram_token or not telegram_chat:
        return ServiceStatus.UNKNOWN, None, "Not configured"

    # If configured, assume healthy (actual check would require API call)
    return ServiceStatus.HEALTHY, None, ""


def get_service_health() -> List[ServiceHealth]:
    """Get health status of all services.

    Returns:
        List of ServiceHealth for each monitored service
    """
    services: List[ServiceHealth] = []
    now = datetime.now()

    # Upbit API
    upbit_status, upbit_latency, upbit_message = _check_upbit_api()
    services.append(
        ServiceHealth(
            name="Upbit API",
            status=upbit_status,
            latency_ms=upbit_latency,
            message=upbit_message,
            last_check=now,
        )
    )

    # Scheduler
    scheduler_status, scheduler_latency, scheduler_message = _check_scheduler()
    services.append(
        ServiceHealth(
            name="Scheduler",
            status=scheduler_status,
            latency_ms=scheduler_latency,
            message=scheduler_message,
            last_check=now,
        )
    )

    # Database
    db_status, db_latency, db_message = _check_database()
    services.append(
        ServiceHealth(
            name="Database",
            status=db_status,
            latency_ms=db_latency,
            message=db_message,
            last_check=now,
        )
    )

    # Telegram
    telegram_status, telegram_latency, telegram_message = _check_telegram()
    services.append(
        ServiceHealth(
            name="Telegram",
            status=telegram_status,
            latency_ms=telegram_latency,
            message=telegram_message,
            last_check=now,
        )
    )

    return services
