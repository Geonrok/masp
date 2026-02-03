"""
Tests for state manager.
"""


from libs.core.state_manager import (
    StateContext,
    get_registered_state,
    register_state_reset,
    reset_all_state,
    unregister_state_reset,
)


class TestStateManagerRegistration:
    """Tests for state registration."""

    def test_register_state_reset(self):
        """Test registering a reset function."""
        call_count = {"value": 0}

        def reset_fn():
            call_count["value"] += 1

        register_state_reset("test_component", reset_fn)
        assert "test_component" in get_registered_state()

        # Clean up
        unregister_state_reset("test_component")

    def test_unregister_state_reset(self):
        """Test unregistering a reset function."""
        register_state_reset("test_unregister", lambda: None)
        assert "test_unregister" in get_registered_state()

        result = unregister_state_reset("test_unregister")
        assert result is True
        assert "test_unregister" not in get_registered_state()

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent component."""
        result = unregister_state_reset("nonexistent_component")
        assert result is False


class TestStateManagerReset:
    """Tests for state reset functionality."""

    def test_reset_all_state(self):
        """Test resetting all state."""
        call_log = []

        def reset_a():
            call_log.append("a")

        def reset_b():
            call_log.append("b")

        register_state_reset("test_a", reset_a)
        register_state_reset("test_b", reset_b)

        reset_names = reset_all_state()

        assert "test_a" in reset_names
        assert "test_b" in reset_names
        assert "a" in call_log
        assert "b" in call_log

        # Clean up
        unregister_state_reset("test_a")
        unregister_state_reset("test_b")

    def test_reset_handles_errors(self):
        """Test reset handles errors gracefully."""

        def failing_reset():
            raise RuntimeError("Reset failed")

        register_state_reset("failing_test", failing_reset)

        # Should not raise
        reset_names = reset_all_state()

        # Failed reset should not be in successful list
        assert "failing_test" not in reset_names

        # Clean up
        unregister_state_reset("failing_test")


class TestStateContext:
    """Tests for StateContext context manager."""

    def test_state_context_resets_on_exit(self):
        """Test state is reset when exiting context."""
        call_count = {"value": 0}

        def counting_reset():
            call_count["value"] += 1

        register_state_reset("counting_test", counting_reset)
        initial_count = call_count["value"]

        with StateContext(reset_on_enter=False, reset_on_exit=True):
            pass

        assert call_count["value"] == initial_count + 1

        # Clean up
        unregister_state_reset("counting_test")

    def test_state_context_resets_on_enter(self):
        """Test state is reset when entering context."""
        call_count = {"value": 0}

        def counting_reset():
            call_count["value"] += 1

        register_state_reset("counting_enter_test", counting_reset)
        initial_count = call_count["value"]

        with StateContext(reset_on_enter=True, reset_on_exit=False):
            assert call_count["value"] == initial_count + 1

        # Clean up
        unregister_state_reset("counting_enter_test")


class TestKnownStateComponents:
    """Tests for known state components."""

    def test_adapter_factory_cache_registered(self):
        """Test adapter factory cache is registered."""
        registered = get_registered_state()
        assert "adapter_factory_cache" in registered

    def test_adapter_factory_cache_clears(self):
        """Test adapter factory cache can be cleared."""
        from libs.adapters.factory import (
            AdapterFactory,
            clear_adapter_cache,
            get_cached_adapters,
        )

        # Create an adapter to populate cache
        AdapterFactory._get_market_data_class("mock")
        cached = get_cached_adapters()
        assert "mock" in cached["market_data"]

        # Clear cache
        clear_adapter_cache()
        cached = get_cached_adapters()
        assert "mock" not in cached["market_data"]


class TestMetricsRegistryReset:
    """Tests for metrics registry reset."""

    def test_metrics_registry_registered(self):
        """Test metrics registry is registered."""
        registered = get_registered_state()
        assert "metrics_registry" in registered

    def test_metrics_registry_reset(self):
        """Test metrics registry can be reset."""
        from libs.core.metrics import MetricsRegistry

        # Get instance and add a counter
        registry = MetricsRegistry.get_instance()
        registry.increment("test_counter")

        # Reset
        MetricsRegistry.reset()

        # Get new instance - should be fresh
        new_registry = MetricsRegistry.get_instance()
        value = new_registry.get_counter("test_counter")
        assert value == 0


class TestGetRegisteredState:
    """Tests for get_registered_state function."""

    def test_returns_list(self):
        """Test returns list of strings."""
        registered = get_registered_state()
        assert isinstance(registered, list)

    def test_contains_expected_components(self):
        """Test contains expected components."""
        registered = get_registered_state()
        # At minimum these should be registered
        assert len(registered) >= 2  # adapter_factory_cache, metrics_registry
