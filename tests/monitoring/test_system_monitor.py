"""
Tests for System Monitor.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from libs.monitoring.system_monitor import (
    ResourceThresholds,
    SystemMetrics,
    SystemMonitor,
    get_system_monitor,
    get_system_status,
)


@pytest.fixture
def thresholds():
    """Create test thresholds."""
    return ResourceThresholds(
        cpu_warning_pct=70.0,
        cpu_critical_pct=90.0,
        memory_warning_pct=70.0,
        memory_critical_pct=90.0,
        disk_warning_pct=80.0,
        disk_critical_pct=95.0,
    )


@pytest.fixture
def monitor(thresholds):
    """Create test monitor."""
    SystemMonitor.reset()
    return SystemMonitor(
        thresholds=thresholds,
        check_interval_seconds=1.0,
        history_size=10,
    )


class TestResourceThresholds:
    """Tests for ResourceThresholds."""

    def test_default_values(self):
        """Test default threshold values."""
        t = ResourceThresholds()
        assert t.cpu_warning_pct == 70.0
        assert t.cpu_critical_pct == 90.0
        assert t.memory_warning_pct == 70.0
        assert t.memory_critical_pct == 90.0
        assert t.disk_warning_pct == 80.0
        assert t.disk_critical_pct == 95.0

    @patch.dict(
        "os.environ",
        {
            "MASP_CPU_WARNING_PCT": "50",
            "MASP_CPU_CRITICAL_PCT": "80",
        },
    )
    def test_from_env(self):
        """Test loading from environment."""
        t = ResourceThresholds.from_env()
        assert t.cpu_warning_pct == 50.0
        assert t.cpu_critical_pct == 80.0


class TestSystemMetrics:
    """Tests for SystemMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        m = SystemMetrics()
        assert m.cpu_percent == 0.0
        assert m.cpu_count == 1
        assert m.memory_percent == 0.0
        assert m.disk_percent == 0.0


class TestSystemMonitor:
    """Tests for SystemMonitor class."""

    def test_init(self, monitor):
        """Test initialization."""
        assert monitor.thresholds.cpu_warning_pct == 70.0
        assert not monitor._running

    def test_singleton(self):
        """Test singleton pattern."""
        SystemMonitor.reset()
        m1 = SystemMonitor.get_instance()
        m2 = SystemMonitor.get_instance()
        assert m1 is m2
        SystemMonitor.reset()

    def test_get_metrics_basic(self, monitor):
        """Test getting metrics."""
        metrics = monitor.get_metrics()
        assert metrics is not None
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_count >= 1

    def test_get_metrics_dict(self, monitor):
        """Test getting metrics as dictionary."""
        metrics_dict = monitor.get_metrics_dict()
        assert "timestamp" in metrics_dict
        assert "cpu" in metrics_dict
        assert "memory" in metrics_dict
        assert "disk" in metrics_dict
        assert "process" in metrics_dict
        assert "system" in metrics_dict

    def test_check_alerts_no_alerts(self, monitor, thresholds):
        """Test no alerts when metrics are normal."""
        # Manually set low metrics
        monitor._current_metrics = SystemMetrics(
            cpu_percent=30.0,
            memory_percent=40.0,
            disk_percent=50.0,
        )
        alerts = monitor.check_alerts()
        assert len(alerts) == 0

    def test_check_alerts_cpu_warning(self, monitor, thresholds):
        """Test CPU warning alert."""
        monitor._current_metrics = SystemMetrics(
            cpu_percent=75.0,  # Above 70% warning
            memory_percent=40.0,
            disk_percent=50.0,
        )
        alerts = monitor.check_alerts()
        assert len(alerts) == 1
        assert alerts[0][0] == "cpu_warning"

    def test_check_alerts_cpu_critical(self, monitor, thresholds):
        """Test CPU critical alert."""
        monitor._current_metrics = SystemMetrics(
            cpu_percent=95.0,  # Above 90% critical
            memory_percent=40.0,
            disk_percent=50.0,
        )
        alerts = monitor.check_alerts()
        assert len(alerts) == 1
        assert alerts[0][0] == "cpu_critical"

    def test_check_alerts_multiple(self, monitor, thresholds):
        """Test multiple alerts."""
        monitor._current_metrics = SystemMetrics(
            cpu_percent=95.0,  # Critical
            memory_percent=92.0,  # Critical
            disk_percent=96.0,  # Critical
        )
        alerts = monitor.check_alerts()
        assert len(alerts) == 3

    def test_get_status_healthy(self, monitor):
        """Test healthy status."""
        monitor._current_metrics = SystemMetrics(
            cpu_percent=30.0,
            memory_percent=40.0,
            disk_percent=50.0,
        )
        status = monitor.get_status()
        assert status["status"] == "healthy"
        assert len(status["alerts"]) == 0

    def test_get_status_warning(self, monitor):
        """Test warning status."""
        monitor._current_metrics = SystemMetrics(
            cpu_percent=75.0,  # Warning
            memory_percent=40.0,
            disk_percent=50.0,
        )
        status = monitor.get_status()
        assert status["status"] == "warning"

    def test_get_status_critical(self, monitor):
        """Test critical status."""
        monitor._current_metrics = SystemMetrics(
            cpu_percent=95.0,  # Critical
            memory_percent=40.0,
            disk_percent=50.0,
        )
        status = monitor.get_status()
        assert status["status"] == "critical"

    def test_alert_callback(self, monitor):
        """Test alert callback registration."""
        alerts_received = []

        def callback(alert_type, data):
            alerts_received.append((alert_type, data))

        monitor.register_alert_callback(callback)
        monitor._current_metrics = SystemMetrics(
            cpu_percent=95.0,
            memory_percent=40.0,
            disk_percent=50.0,
        )
        monitor._check_and_alert()

        assert len(alerts_received) == 1
        assert alerts_received[0][0] == "cpu_critical"

    def test_history(self, monitor):
        """Test metrics history."""
        # Collect some metrics
        for _ in range(3):
            monitor._collect_metrics()

        history = monitor.get_history(minutes=30)
        assert len(history) >= 3

    def test_start_stop(self, monitor):
        """Test start and stop."""
        monitor.start()
        assert monitor._running
        time.sleep(0.1)

        monitor.stop()
        assert not monitor._running


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_system_monitor(self):
        """Test get_system_monitor function."""
        SystemMonitor.reset()
        monitor = get_system_monitor()
        assert isinstance(monitor, SystemMonitor)
        SystemMonitor.reset()

    def test_get_system_status(self):
        """Test get_system_status function."""
        SystemMonitor.reset()
        status = get_system_status()
        assert "status" in status
        SystemMonitor.reset()
