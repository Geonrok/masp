"""
System Resource Monitor for MASP.

Monitors CPU, Memory, Disk, Network usage and system health.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Current system resource metrics."""

    timestamp: datetime = field(default_factory=datetime.now)

    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 1
    cpu_freq_mhz: Optional[float] = None

    # Memory
    memory_total_mb: float = 0.0
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0

    # Disk
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_percent: float = 0.0

    # Process
    process_memory_mb: float = 0.0
    process_threads: int = 0
    process_open_files: int = 0

    # System
    uptime_seconds: float = 0.0
    platform: str = ""
    python_version: str = ""


@dataclass
class ResourceThresholds:
    """Thresholds for resource alerts."""

    cpu_warning_pct: float = 70.0
    cpu_critical_pct: float = 90.0

    memory_warning_pct: float = 70.0
    memory_critical_pct: float = 90.0

    disk_warning_pct: float = 80.0
    disk_critical_pct: float = 95.0

    @classmethod
    def from_env(cls) -> "ResourceThresholds":
        """Load thresholds from environment variables."""
        return cls(
            cpu_warning_pct=float(os.getenv("MASP_CPU_WARNING_PCT", "70")),
            cpu_critical_pct=float(os.getenv("MASP_CPU_CRITICAL_PCT", "90")),
            memory_warning_pct=float(os.getenv("MASP_MEM_WARNING_PCT", "70")),
            memory_critical_pct=float(os.getenv("MASP_MEM_CRITICAL_PCT", "90")),
            disk_warning_pct=float(os.getenv("MASP_DISK_WARNING_PCT", "80")),
            disk_critical_pct=float(os.getenv("MASP_DISK_CRITICAL_PCT", "95")),
        )


class SystemMonitor:
    """
    System resource monitor with alerting.

    Monitors CPU, memory, disk usage and raises alerts when
    thresholds are exceeded.

    Example:
        monitor = SystemMonitor()
        monitor.start()

        metrics = monitor.get_metrics()
        print(f"CPU: {metrics.cpu_percent}%")

        alerts = monitor.check_alerts()
        if alerts:
            for alert in alerts:
                print(f"Alert: {alert}")

        monitor.stop()
    """

    _instance: Optional["SystemMonitor"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        thresholds: Optional[ResourceThresholds] = None,
        check_interval_seconds: float = 30.0,
        history_size: int = 100,
    ):
        """
        Initialize system monitor.

        Args:
            thresholds: Resource alert thresholds
            check_interval_seconds: How often to collect metrics
            history_size: Number of metric snapshots to keep
        """
        self.thresholds = thresholds or ResourceThresholds.from_env()
        self._check_interval = check_interval_seconds
        self._history_size = history_size

        self._metrics_history: List[SystemMetrics] = []
        self._current_metrics: Optional[SystemMetrics] = None
        self._start_time = time.time()

        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        self._psutil_available = False
        self._init_psutil()

    def _init_psutil(self) -> None:
        """Try to import psutil for detailed metrics."""
        try:
            import psutil

            self._psutil_available = True
            self._psutil = psutil
            logger.info("[SystemMonitor] psutil available for detailed metrics")
        except ImportError:
            logger.warning("[SystemMonitor] psutil not available, basic metrics only")

    @classmethod
    def get_instance(cls) -> "SystemMonitor":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        if cls._instance:
            cls._instance.stop()
        cls._instance = None

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="SystemMonitor",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("[SystemMonitor] Started with %ds interval", self._check_interval)

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("[SystemMonitor] Stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self._collect_metrics()
                self._check_and_alert()
            except Exception as e:
                logger.error("[SystemMonitor] Error collecting metrics: %s", e)

            time.sleep(self._check_interval)

    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            uptime_seconds=time.time() - self._start_time,
            platform=platform.platform(),
            python_version=platform.python_version(),
        )

        if self._psutil_available:
            self._collect_psutil_metrics(metrics)
        else:
            self._collect_basic_metrics(metrics)

        self._current_metrics = metrics

        # Keep history
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._history_size:
            self._metrics_history = self._metrics_history[-self._history_size :]

        return metrics

    def _collect_psutil_metrics(self, metrics: SystemMetrics) -> None:
        """Collect metrics using psutil."""
        psutil = self._psutil

        # CPU
        metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.cpu_count = psutil.cpu_count() or 1
        try:
            freq = psutil.cpu_freq()
            if freq:
                metrics.cpu_freq_mhz = freq.current
        except Exception:
            pass

        # Memory
        mem = psutil.virtual_memory()
        metrics.memory_total_mb = mem.total / (1024 * 1024)
        metrics.memory_used_mb = mem.used / (1024 * 1024)
        metrics.memory_percent = mem.percent

        # Disk
        try:
            disk = psutil.disk_usage("/")
            metrics.disk_total_gb = disk.total / (1024 * 1024 * 1024)
            metrics.disk_used_gb = disk.used / (1024 * 1024 * 1024)
            metrics.disk_percent = disk.percent
        except Exception:
            pass

        # Process
        try:
            process = psutil.Process()
            metrics.process_memory_mb = process.memory_info().rss / (1024 * 1024)
            metrics.process_threads = process.num_threads()
            try:
                metrics.process_open_files = len(process.open_files())
            except Exception:
                pass
        except Exception:
            pass

    def _collect_basic_metrics(self, metrics: SystemMetrics) -> None:
        """Collect basic metrics without psutil."""
        metrics.cpu_count = os.cpu_count() or 1

    def _check_and_alert(self) -> None:
        """Check metrics and trigger alerts if thresholds exceeded."""
        if not self._current_metrics:
            return

        alerts = self.check_alerts()
        for alert_type, alert_data in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(alert_type, alert_data)
                except Exception as e:
                    logger.error("[SystemMonitor] Alert callback error: %s", e)

    def check_alerts(self) -> List[tuple]:
        """
        Check current metrics against thresholds.

        Returns:
            List of (alert_type, alert_data) tuples
        """
        if not self._current_metrics:
            return []

        alerts = []
        m = self._current_metrics
        t = self.thresholds

        # CPU alerts
        if m.cpu_percent >= t.cpu_critical_pct:
            alerts.append(
                (
                    "cpu_critical",
                    {"value": m.cpu_percent, "threshold": t.cpu_critical_pct},
                )
            )
        elif m.cpu_percent >= t.cpu_warning_pct:
            alerts.append(
                (
                    "cpu_warning",
                    {"value": m.cpu_percent, "threshold": t.cpu_warning_pct},
                )
            )

        # Memory alerts
        if m.memory_percent >= t.memory_critical_pct:
            alerts.append(
                (
                    "memory_critical",
                    {"value": m.memory_percent, "threshold": t.memory_critical_pct},
                )
            )
        elif m.memory_percent >= t.memory_warning_pct:
            alerts.append(
                (
                    "memory_warning",
                    {"value": m.memory_percent, "threshold": t.memory_warning_pct},
                )
            )

        # Disk alerts
        if m.disk_percent >= t.disk_critical_pct:
            alerts.append(
                (
                    "disk_critical",
                    {"value": m.disk_percent, "threshold": t.disk_critical_pct},
                )
            )
        elif m.disk_percent >= t.disk_warning_pct:
            alerts.append(
                (
                    "disk_warning",
                    {"value": m.disk_percent, "threshold": t.disk_warning_pct},
                )
            )

        return alerts

    def register_alert_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], None],
    ) -> None:
        """Register callback for alerts."""
        self._alert_callbacks.append(callback)

    def get_metrics(self) -> Optional[SystemMetrics]:
        """Get current metrics snapshot."""
        if not self._current_metrics:
            return self._collect_metrics()
        return self._current_metrics

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get current metrics as dictionary."""
        metrics = self.get_metrics()
        if not metrics:
            return {}

        return {
            "timestamp": metrics.timestamp.isoformat(),
            "cpu": {
                "percent": metrics.cpu_percent,
                "count": metrics.cpu_count,
                "freq_mhz": metrics.cpu_freq_mhz,
            },
            "memory": {
                "total_mb": round(metrics.memory_total_mb, 2),
                "used_mb": round(metrics.memory_used_mb, 2),
                "percent": metrics.memory_percent,
            },
            "disk": {
                "total_gb": round(metrics.disk_total_gb, 2),
                "used_gb": round(metrics.disk_used_gb, 2),
                "percent": metrics.disk_percent,
            },
            "process": {
                "memory_mb": round(metrics.process_memory_mb, 2),
                "threads": metrics.process_threads,
                "open_files": metrics.process_open_files,
            },
            "system": {
                "uptime_seconds": round(metrics.uptime_seconds, 2),
                "platform": metrics.platform,
                "python_version": metrics.python_version,
            },
        }

    def get_history(
        self,
        minutes: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get metrics history for the last N minutes.

        Args:
            minutes: Number of minutes of history to return

        Returns:
            List of metric dictionaries
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        history = []

        for m in self._metrics_history:
            if m.timestamp >= cutoff:
                history.append(
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "cpu_percent": m.cpu_percent,
                        "memory_percent": m.memory_percent,
                        "disk_percent": m.disk_percent,
                    }
                )

        return history

    def get_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        metrics = self.get_metrics()
        if not metrics:
            return {"status": "unknown"}

        alerts = self.check_alerts()
        has_critical = any("critical" in a[0] for a in alerts)
        has_warning = any("warning" in a[0] for a in alerts)

        if has_critical:
            status = "critical"
        elif has_warning:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "alerts": [{"type": a[0], "data": a[1]} for a in alerts],
            "metrics": self.get_metrics_dict(),
        }


# ============================================================================
# Convenience functions
# ============================================================================


def get_system_monitor() -> SystemMonitor:
    """Get global system monitor instance."""
    return SystemMonitor.get_instance()


def get_system_status() -> Dict[str, Any]:
    """Get current system status."""
    return get_system_monitor().get_status()
