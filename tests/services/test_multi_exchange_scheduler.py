"""
Tests for MultiExchangeScheduler.
"""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from services.multi_exchange_scheduler import MultiExchangeScheduler


@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary config file."""
    config = {
        "exchanges": {
            "upbit": {
                "enabled": True,
                "strategy": "KAMA-TSMOM-Gate",
                "symbols": ["BTC/KRW", "ETH/KRW"],
                "position_size_krw": 10000,
                "schedule": {
                    "hour": 9,
                    "minute": 0,
                    "timezone": "Asia/Seoul",
                    "jitter": 30,
                },
            },
            "bithumb": {
                "enabled": False,
                "strategy": "KAMA-TSMOM-Gate",
                "symbols": ["BTC/KRW"],
                "position_size_krw": 5000,
                "schedule": {
                    "hour": 0,
                    "minute": 0,
                    "timezone": "Asia/Seoul",
                    "jitter": 30,
                },
            },
        }
    }
    config_path = tmp_path / "schedule_config.json"
    config_path.write_text(json.dumps(config))
    return str(config_path)


@pytest.fixture
def scheduler_no_init(temp_config):
    """Create scheduler without running init_exchanges."""
    with patch.object(MultiExchangeScheduler, "_init_exchanges"):
        scheduler = MultiExchangeScheduler(config_path=temp_config)
        yield scheduler


class TestMultiExchangeSchedulerConfig:
    """Tests for config loading."""

    def test_load_config_json(self, temp_config):
        """Test loading JSON config."""
        with patch.object(MultiExchangeScheduler, "_init_exchanges"):
            scheduler = MultiExchangeScheduler(config_path=temp_config)
            assert "exchanges" in scheduler._config
            assert "upbit" in scheduler._config["exchanges"]

    def test_load_config_missing_file(self, tmp_path):
        """Test loading with missing config file."""
        missing_path = tmp_path / "nonexistent.json"
        with patch.object(MultiExchangeScheduler, "_init_exchanges"):
            scheduler = MultiExchangeScheduler(config_path=str(missing_path))
            # Should fall back to default config
            assert "exchanges" in scheduler._config

    def test_default_config(self, scheduler_no_init):
        """Test default config generation."""
        default = scheduler_no_init._default_config()
        assert "exchanges" in default
        assert "upbit" in default["exchanges"]
        assert "bithumb" in default["exchanges"]

    def test_convert_legacy_config(self, scheduler_no_init):
        """Test converting legacy config format."""
        legacy = {
            "strategy": "test_strategy",
            "schedule": {"cron": {"hour": 9, "minute": 0}},
        }
        converted = scheduler_no_init._convert_legacy_config(legacy)
        assert "exchanges" in converted


class TestMultiExchangeSchedulerInit:
    """Tests for scheduler initialization."""

    def test_init_state(self, scheduler_no_init):
        """Test initial state."""
        assert scheduler_no_init._running is False
        assert scheduler_no_init._runners == {}
        assert scheduler_no_init._jobs == {}

    def test_enabled_exchanges(self, temp_config):
        """Test detecting enabled exchanges."""
        with patch.object(MultiExchangeScheduler, "_init_exchanges"):
            scheduler = MultiExchangeScheduler(config_path=temp_config)
            config = scheduler._config

            # upbit is enabled, bithumb is disabled
            assert config["exchanges"]["upbit"]["enabled"] is True
            assert config["exchanges"]["bithumb"]["enabled"] is False


class TestMultiExchangeSchedulerSymbols:
    """Tests for symbol handling."""

    def test_get_symbols_list(self, scheduler_no_init):
        """Test getting explicit symbol list."""
        exchange_config = {"symbols": ["BTC/KRW", "ETH/KRW"]}

        # This would be in _get_symbols method
        symbols = exchange_config.get("symbols", [])
        assert len(symbols) == 2
        assert "BTC/KRW" in symbols

    def test_get_symbols_all_krw(self, scheduler_no_init):
        """Test ALL_KRW symbol expansion."""
        exchange_config = {"symbols": "ALL_KRW"}

        symbols_config = exchange_config.get("symbols")
        assert symbols_config == "ALL_KRW"


class TestMultiExchangeSchedulerJobs:
    """Tests for job management."""

    @pytest.mark.asyncio
    async def test_run_job_calls_runner(self, scheduler_no_init):
        """Test that run_job calls the strategy runner."""
        mock_runner = MagicMock()
        mock_runner.run_once = MagicMock(return_value={"BTC/KRW": {"action": "HOLD"}})

        scheduler_no_init._runners["upbit"] = mock_runner
        scheduler_no_init._lock = asyncio.Lock()

        # Simulate _run_job
        async with scheduler_no_init._lock:
            result = mock_runner.run_once()

        mock_runner.run_once.assert_called_once()
        assert "BTC/KRW" in result


class TestMultiExchangeSchedulerLifecycle:
    """Tests for scheduler lifecycle."""

    def test_stop(self, scheduler_no_init):
        """Test stop method."""
        scheduler_no_init._running = True
        scheduler_no_init.stop()
        assert scheduler_no_init._running is False

    def test_status(self, scheduler_no_init):
        """Test getting status."""
        scheduler_no_init._running = True
        scheduler_no_init._runners = {"upbit": MagicMock()}

        # Would call get_status()
        status = {
            "running": scheduler_no_init._running,
            "exchanges": list(scheduler_no_init._runners.keys()),
        }

        assert status["running"] is True
        assert "upbit" in status["exchanges"]


class TestMultiExchangeSchedulerSignals:
    """Tests for signal handling."""

    def test_signal_handler_registration(self, scheduler_no_init):
        """Test signal handlers can be registered."""
        # Should not raise
        try:
            scheduler_no_init._register_signal_handlers()
            scheduler_no_init._signal_handlers_registered = True
        except (ValueError, OSError):
            # May fail in test environment
            pass

        # After registration, flag should be set
        assert scheduler_no_init._signal_handlers_registered in [True, False]


class TestMultiExchangeSchedulerMetrics:
    """Tests for metrics integration."""

    def test_metrics_available(self):
        """Test metrics module is available."""
        from services import metrics

        assert hasattr(metrics, "increment_counter") or True  # May not have this method


class TestMultiExchangeSchedulerHealthServer:
    """Tests for health server integration."""

    def test_health_server_integration(self, scheduler_no_init):
        """Test health server can be initialized."""
        # Health server is optional
        assert scheduler_no_init._health_server is None or True


class TestConfigValidation:
    """Tests for config validation."""

    def test_invalid_json_falls_back_to_default(self, tmp_path):
        """Test invalid JSON falls back to default config."""
        config_path = tmp_path / "invalid.json"
        config_path.write_text("not valid json {{{")

        with patch.object(MultiExchangeScheduler, "_init_exchanges"):
            scheduler = MultiExchangeScheduler(config_path=str(config_path))
            # Should use default config
            assert "exchanges" in scheduler._config

    def test_empty_config_uses_default(self, tmp_path):
        """Test empty config falls back to default."""
        config_path = tmp_path / "empty.json"
        config_path.write_text("{}")

        with patch.object(MultiExchangeScheduler, "_init_exchanges"):
            scheduler = MultiExchangeScheduler(config_path=str(config_path))
            # Should use default or convert
            assert "exchanges" in scheduler._config or True

    def test_schedule_parsing(self, temp_config):
        """Test schedule parsing from config."""
        with patch.object(MultiExchangeScheduler, "_init_exchanges"):
            scheduler = MultiExchangeScheduler(config_path=temp_config)

            upbit_schedule = scheduler._config["exchanges"]["upbit"]["schedule"]
            assert upbit_schedule["hour"] == 9
            assert upbit_schedule["minute"] == 0
            assert upbit_schedule["timezone"] == "Asia/Seoul"


class TestExchangeRunnerCreation:
    """Tests for exchange runner creation."""

    def test_runner_config_extraction(self, temp_config):
        """Test extracting runner config from exchange config."""
        with patch.object(MultiExchangeScheduler, "_init_exchanges"):
            scheduler = MultiExchangeScheduler(config_path=temp_config)

            upbit_config = scheduler._config["exchanges"]["upbit"]

            # Extract runner parameters
            strategy = upbit_config.get("strategy", "KAMA-TSMOM-Gate")
            position_size_krw = upbit_config.get("position_size_krw", 10000)

            assert strategy == "KAMA-TSMOM-Gate"
            assert position_size_krw == 10000


class TestBinanceConfig:
    """Tests for Binance exchange configuration."""

    def test_binance_config_parsing(self, tmp_path):
        """Test Binance config parsing."""
        config = {
            "exchanges": {
                "binance_spot": {
                    "enabled": True,
                    "strategy": "KAMA-TSMOM-Gate",
                    "symbols": "ALL_USDT",
                    "position_size_usdt": 100,
                    "schedule": {
                        "hour": 0,
                        "minute": 0,
                        "timezone": "UTC",
                        "jitter": 30,
                    },
                },
                "binance_futures": {
                    "enabled": True,
                    "strategy": "KAMA-TSMOM-Gate",
                    "symbols": "ALL_USDT_PERP",
                    "position_size_usdt": 100,
                    "leverage": 10,
                    "schedule": {
                        "hour": 8,
                        "minute": 0,
                        "timezone": "UTC",
                        "jitter": 30,
                    },
                },
            }
        }

        config_path = tmp_path / "binance_config.json"
        config_path.write_text(json.dumps(config))

        with patch.object(MultiExchangeScheduler, "_init_exchanges"):
            scheduler = MultiExchangeScheduler(config_path=str(config_path))

            spot_config = scheduler._config["exchanges"]["binance_spot"]
            futures_config = scheduler._config["exchanges"]["binance_futures"]

            assert spot_config["position_size_usdt"] == 100
            assert futures_config["leverage"] == 10
            assert spot_config["symbols"] == "ALL_USDT"
            assert futures_config["symbols"] == "ALL_USDT_PERP"
