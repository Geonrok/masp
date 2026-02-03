"""
Live ACK gate tests.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestLiveACKGate:
    """Live ACK gate checks."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.bithumb_api_key.get_secret_value.return_value = (
            "test_api_key_12345678901234567890"
        )
        config.bithumb_secret_key.get_secret_value.return_value = (
            "test_secret_key_1234567890123456"
        )
        config.is_kill_switch_active.return_value = False
        config.max_order_value_krw = 1_000_000
        return config

    def test_live_adapter_requires_ack_env_vars(self):
        with patch.dict(os.environ, {"MASP_ENABLE_LIVE_TRADING": "0"}, clear=False):
            from services.strategy_runner import StrategyRunner

            runner = StrategyRunner(
                strategy_name="kama_tsmom_gate",
                exchange="bithumb",
                symbols=["BTC/KRW"],
                position_size_krw=10000,
            )

            assert "Paper" in runner.execution.__class__.__name__

    def test_live_mode_requires_ack(self):
        with patch.dict(
            os.environ,
            {
                "MASP_ENABLE_LIVE_TRADING": "1",
                "MASP_ACK_BITHUMB_LIVE": "0",
            },
            clear=False,
        ):
            with patch("libs.adapters.factory.logger") as mock_logger:
                from libs.adapters.factory import AdapterFactory

                config = MagicMock()
                config.is_kill_switch_active.return_value = False
                config.bithumb_api_key = MagicMock()
                config.bithumb_api_key.get_secret_value.return_value = (
                    "test_key_1234567890123456"
                )
                config.bithumb_secret_key = MagicMock()
                config.bithumb_secret_key.get_secret_value.return_value = (
                    "test_secret_1234567890"
                )

                with patch(
                    "libs.adapters.real_bithumb_execution.BithumbAPIV2"
                ) as mock_api:
                    mock_api.return_value = MagicMock()
                    adapter = AdapterFactory.create_execution(
                        exchange_name="bithumb_spot",
                        adapter_mode="live",
                        config=config,
                    )

                assert "Bithumb" in adapter.__class__.__name__

                mock_logger.warning.assert_called()
                warning_calls = [
                    str(call) for call in mock_logger.warning.call_args_list
                ]
                assert any(
                    "Real trading" in c or "Kill-Switch" in c for c in warning_calls
                )

    def test_kill_switch_blocks_before_order(self, mock_config):
        mock_config.is_kill_switch_active.return_value = True

        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            result = adapter.place_order("BTC/KRW", "BUY", units=0.001)

            assert result.status == "REJECTED"
            assert "Kill-Switch" in result.message
            mock_instance.post_order.assert_not_called()

    def test_strategy_runner_checks_env_before_execution(self):
        with patch.dict(os.environ, {"STOP_TRADING": "1"}, clear=False):
            from services.strategy_runner import StrategyRunner

            runner = StrategyRunner(
                strategy_name="kama_tsmom_gate",
                exchange="paper",
                symbols=["BTC/KRW"],
                position_size_krw=10000,
            )

            with pytest.raises(RuntimeError, match="Kill-Switch"):
                runner.run_once()


class TestOrderIdTracking:
    """Order ID parsing checks."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.bithumb_api_key.get_secret_value.return_value = (
            "test_api_key_12345678901234567890"
        )
        config.bithumb_secret_key.get_secret_value.return_value = (
            "test_secret_key_1234567890123456"
        )
        config.is_kill_switch_active.return_value = False
        config.max_order_value_krw = 10_000_000
        return config

    def test_order_id_not_fallback_to_symbol(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.post_order.return_value = {"uuid": "order_12345"}
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)
            adapter.get_current_price = MagicMock(return_value=50_000_000)

            result = adapter.place_order("BTC/KRW", "BUY", units=0.001)

            assert result.order_id == "order_12345"
            assert result.order_id != "BTC/KRW"
            assert result.order_id != "BTC"

    def test_order_id_handles_none_response(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.post_order.return_value = None
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)
            adapter.get_current_price = MagicMock(return_value=50_000_000)

            result = adapter.place_order("BTC/KRW", "BUY", units=0.001)

            assert result.status == "REJECTED"
            assert "None response" in result.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
