"""
MASP Test Configuration (Phase 4)
Prometheus Registry Isolation
"""

import pytest


@pytest.fixture(autouse=True)
def reset_prometheus_registry():
    """Reset Prometheus state before/after each test."""
    from services import metrics

    metrics._initialized = False
    metrics._prometheus_available = False
    metrics._metrics.clear()

    yield

    if hasattr(metrics, "cleanup_for_tests"):
        metrics.cleanup_for_tests()
