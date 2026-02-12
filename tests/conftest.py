"""
MASP Test Configuration (Phase 4)
- Prometheus Registry Isolation
- Global State Reset
"""

import pytest


@pytest.fixture(autouse=True)
def disable_telegram_in_tests(monkeypatch):
    """Prevent tests from sending real Telegram notifications.

    strategy_runner.py has module-level load_dotenv() that loads production
    TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from .env at import time.
    Without this fixture, tests creating StrategyRunner send real messages.
    """
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


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


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset all registered global state after each test."""
    yield
    # Reset after test completes
    try:
        from libs.core.state_manager import reset_all_state

        reset_all_state()
    except ImportError:
        pass  # Module not available yet


@pytest.fixture
def clean_state():
    """
    Fixture for tests that need completely clean state.

    Usage:
        def test_something(clean_state):
            # State is reset before this test runs
            pass
    """
    from libs.core.state_manager import reset_all_state

    reset_all_state()
    yield
    reset_all_state()
