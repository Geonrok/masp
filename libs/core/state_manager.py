"""
Global State Management for MASP.

Provides centralized management of global state including:
- Singleton instances
- Caches
- Registry entries

This module helps with testing by providing methods to reset
all global state to a clean state.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Registry of reset functions for global state
_state_reset_functions: Dict[str, Callable[[], None]] = {}


def register_state_reset(name: str, reset_fn: Callable[[], None]) -> None:
    """
    Register a function to reset global state.

    Args:
        name: Unique name for this state component
        reset_fn: Function that resets the state when called
    """
    _state_reset_functions[name] = reset_fn
    logger.debug("[StateManager] Registered reset function: %s", name)


def unregister_state_reset(name: str) -> bool:
    """
    Unregister a state reset function.

    Args:
        name: Name of the state component to unregister

    Returns:
        True if the function was found and removed
    """
    if name in _state_reset_functions:
        del _state_reset_functions[name]
        return True
    return False


def reset_all_state() -> List[str]:
    """
    Reset all registered global state.

    This is primarily intended for use in tests to ensure
    a clean state between test cases.

    Returns:
        List of state component names that were reset
    """
    reset_names = []

    for name, reset_fn in _state_reset_functions.items():
        try:
            reset_fn()
            reset_names.append(name)
            logger.debug("[StateManager] Reset: %s", name)
        except Exception as e:
            logger.error("[StateManager] Failed to reset %s: %s", name, e)

    return reset_names


def get_registered_state() -> List[str]:
    """
    Get list of registered state component names.

    Returns:
        List of registered state names
    """
    return list(_state_reset_functions.keys())


# ============================================================================
# Auto-register known global state components
# ============================================================================


def _register_known_state() -> None:
    """Register known global state components."""

    # AdapterFactory cache
    try:
        from libs.adapters.factory import clear_adapter_cache

        register_state_reset("adapter_factory_cache", clear_adapter_cache)
    except ImportError:
        pass

    # MetricsRegistry singleton
    try:
        from libs.core.metrics import MetricsRegistry

        register_state_reset("metrics_registry", MetricsRegistry.reset)
    except ImportError:
        pass

    # PerformanceMonitor singleton
    try:
        from libs.utils.performance import get_performance_monitor

        def reset_perf_monitor():
            monitor = get_performance_monitor()
            if hasattr(monitor, "_stats"):
                monitor._stats.clear()
            if hasattr(monitor, "_timers"):
                monitor._timers.clear()
            if hasattr(monitor, "_records"):
                monitor._records.clear()

        register_state_reset("performance_monitor", reset_perf_monitor)
    except (ImportError, AttributeError):
        pass


# Register known state on module load
_register_known_state()


# ============================================================================
# Pytest fixture helper
# ============================================================================


def create_state_reset_fixture():
    """
    Create a pytest fixture that resets global state.

    Usage in conftest.py:
        from libs.core.state_manager import create_state_reset_fixture
        reset_state = create_state_reset_fixture()

    Then use @pytest.fixture decorated with autouse=True
    """

    def fixture_fn():
        """Pytest fixture to reset global state before each test."""
        yield
        reset_all_state()

    return fixture_fn


# ============================================================================
# Context manager for temporary state changes
# ============================================================================


class StateContext:
    """
    Context manager for temporary state changes.

    Example:
        with StateContext():
            # State is isolated within this block
            adapter = AdapterFactory.create_market_data("upbit_spot")
            # ...
        # State is reset when exiting
    """

    def __init__(self, reset_on_enter: bool = True, reset_on_exit: bool = True):
        """
        Initialize state context.

        Args:
            reset_on_enter: Reset state when entering context
            reset_on_exit: Reset state when exiting context
        """
        self._reset_on_enter = reset_on_enter
        self._reset_on_exit = reset_on_exit

    def __enter__(self) -> "StateContext":
        if self._reset_on_enter:
            reset_all_state()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._reset_on_exit:
            reset_all_state()
