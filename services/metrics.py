"""
MASP Prometheus Metrics (Phase 4 Final)
Test Isolation: cleanup_for_tests() + conftest fixture
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict

logger = logging.getLogger(__name__)

_initialized = False
_prometheus_available = False
_registry = None
_metrics: Dict[str, object] = {}


def _is_test_environment() -> bool:
    return "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST") is not None


def init_metrics() -> bool:
    global _initialized, _prometheus_available, _registry, _metrics

    if _initialized:
        return _prometheus_available

    _initialized = True

    if os.getenv("MASP_ENABLE_METRICS", "0") != "1":
        logger.info("[Metrics] Disabled (MASP_ENABLE_METRICS != 1)")
        return False

    try:
        from prometheus_client import (
            REGISTRY,
            CollectorRegistry,
            Counter,
            Gauge,
            Histogram,
        )

        if _is_test_environment():
            _registry = CollectorRegistry()
            logger.debug("[Metrics] Using isolated CollectorRegistry for tests")
        else:
            _registry = REGISTRY

        _metrics["heartbeat_total"] = Counter(
            "masp_scheduler_heartbeat_total",
            "Total scheduler heartbeats",
            registry=_registry,
        )
        _metrics["running"] = Gauge(
            "masp_scheduler_running",
            "Scheduler running status",
            registry=_registry,
        )
        _metrics["uptime"] = Gauge(
            "masp_scheduler_uptime_seconds",
            "Scheduler uptime",
            registry=_registry,
        )
        _metrics["active_exchanges"] = Gauge(
            "masp_scheduler_active_exchanges",
            "Active exchange count",
            registry=_registry,
        )

        _prometheus_available = True
        logger.info("[Metrics] Prometheus initialized")
        return True
    except ImportError:
        logger.warning("[Metrics] prometheus_client not installed")
        return False


def is_metrics_enabled() -> bool:
    return _initialized and _prometheus_available


def inc_heartbeat() -> None:
    if is_metrics_enabled() and "heartbeat_total" in _metrics:
        _metrics["heartbeat_total"].inc()


def set_scheduler_running(running: bool) -> None:
    if is_metrics_enabled() and "running" in _metrics:
        _metrics["running"].set(1 if running else 0)


def set_uptime(seconds: float) -> None:
    if is_metrics_enabled() and "uptime" in _metrics:
        _metrics["uptime"].set(seconds)


def set_active_exchanges(count: int) -> None:
    if is_metrics_enabled() and "active_exchanges" in _metrics:
        _metrics["active_exchanges"].set(count)


def get_metrics_output() -> bytes:
    if not is_metrics_enabled():
        return b"# Metrics disabled\n"
    from prometheus_client import generate_latest

    return generate_latest(_registry)


def cleanup_for_tests() -> None:
    """Test teardown: reset all state."""
    global _initialized, _prometheus_available, _registry, _metrics

    if _registry and hasattr(_registry, "_collector_to_names"):
        for collector in list(_registry._collector_to_names.keys()):
            try:
                _registry.unregister(collector)
            except Exception:
                pass

    _metrics.clear()
    _initialized = False
    _prometheus_available = False
    _registry = None
