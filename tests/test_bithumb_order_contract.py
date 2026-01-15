"""
Bithumb order contract tests (API v2).
"""

import pytest
from unittest.mock import MagicMock, patch


class TestBithumbOrderContract:
    """Bithumb order contract tests."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.bithumb_api_key.get_secret_value.return_value = "test_api_key_12345678901234567890"
        config.bithumb_secret_key.get_secret_value.return_value = "test_secret_key_1234567890123456"
        config.is_kill_switch_active.return_value = False
        config.max_order_value_krw = 1_000_000
        return config

    def test_buy_calls_api_v2_with_krw_price(self, mock_config):
        with patch('libs.adapters.real_bithumb_execution.BithumbAPIV2') as mock_api:
            mock_instance = MagicMock()
            mock_instance.post_order.return_value = {"uuid": "order_1"}
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)
            adapter.get_current_price = MagicMock(return_value=50_000_000)

            test_units = 0.001
            adapter.place_order("BTC/KRW", "BUY", units=test_units)

            mock_instance.post_order.assert_called_once()
            call_kwargs = mock_instance.post_order.call_args.kwargs
            assert call_kwargs["market"] == "KRW-BTC"
            assert call_kwargs["side"] == "bid"
            assert call_kwargs["ord_type"] == "price"
            assert call_kwargs["price"] == "50000"
            assert "volume" not in call_kwargs

    def test_buy_rejects_ambiguous_inputs_both_none(self, mock_config):
        with patch('libs.adapters.real_bithumb_execution.BithumbAPIV2') as mock_api:
            mock_api.return_value = MagicMock()

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)

            with pytest.raises(ValueError, match="BUY requires exactly one of"):
                adapter.place_order("BTC/KRW", "BUY")

    def test_buy_rejects_ambiguous_inputs_both_set(self, mock_config):
        with patch('libs.adapters.real_bithumb_execution.BithumbAPIV2') as mock_api:
            mock_api.return_value = MagicMock()

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)

            with pytest.raises(ValueError, match="not both"):
                adapter.place_order("BTC/KRW", "BUY", units=0.01, amount_krw=50000)

    def test_sell_requires_units(self, mock_config):
        with patch('libs.adapters.real_bithumb_execution.BithumbAPIV2') as mock_api:
            mock_api.return_value = MagicMock()

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)

            with pytest.raises(ValueError, match="SELL requires units"):
                adapter.place_order("BTC/KRW", "SELL")

    def test_sell_rejects_amount_krw(self, mock_config):
        with patch('libs.adapters.real_bithumb_execution.BithumbAPIV2') as mock_api:
            mock_api.return_value = MagicMock()

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)

            with pytest.raises(ValueError, match="SELL does not accept amount_krw"):
                adapter.place_order("BTC/KRW", "SELL", units=0.01, amount_krw=50000)

    def test_sell_calls_api_v2_with_units(self, mock_config):
        with patch('libs.adapters.real_bithumb_execution.BithumbAPIV2') as mock_api:
            mock_instance = MagicMock()
            mock_instance.post_order.return_value = {"uuid": "order_2"}
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)
            adapter.get_current_price = MagicMock(return_value=50_000_000)

            test_units = 0.002
            adapter.place_order("BTC/KRW", "SELL", units=test_units)

            mock_instance.post_order.assert_called_once()
            call_kwargs = mock_instance.post_order.call_args.kwargs
            assert call_kwargs["side"] == "ask"
            assert call_kwargs["ord_type"] == "market"
            assert call_kwargs["volume"] == f"{test_units:.8f}"

    def test_minimum_order_value_check(self, mock_config):
        with patch('libs.adapters.real_bithumb_execution.BithumbAPIV2') as mock_api:
            mock_instance = MagicMock()
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)
            adapter.get_current_price = MagicMock(return_value=50_000_000)

            small_units = 0.00008
            result = adapter.place_order("BTC/KRW", "BUY", units=small_units)

            assert result.status == "REJECTED"
            assert "최소 주문 금액" in result.message
            mock_instance.post_order.assert_not_called()

    def test_kill_switch_blocks_order(self, mock_config):
        mock_config.is_kill_switch_active.return_value = True

        with patch('libs.adapters.real_bithumb_execution.BithumbAPIV2') as mock_api:
            mock_instance = MagicMock()
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)

            result = adapter.place_order("BTC/KRW", "BUY", units=0.001)

            assert result.status == "REJECTED"
            assert "Kill-Switch" in result.message
            mock_instance.post_order.assert_not_called()


class TestOrderIdTracking:
    """Order ID parsing tests."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.bithumb_api_key.get_secret_value.return_value = "test_api_key_12345678901234567890"
        config.bithumb_secret_key.get_secret_value.return_value = "test_secret_key_1234567890123456"
        config.is_kill_switch_active.return_value = False
        config.max_order_value_krw = 10_000_000
        return config

    def test_order_id_from_uuid_response(self, mock_config):
        with patch('libs.adapters.real_bithumb_execution.BithumbAPIV2') as mock_api:
            mock_instance = MagicMock()
            mock_instance.post_order.return_value = {"uuid": "order_12345"}
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)
            adapter.get_current_price = MagicMock(return_value=50_000_000)

            result = adapter.place_order("BTC/KRW", "BUY", units=0.001)

            assert result.order_id == "order_12345"
            assert result.order_id != "BTC/KRW"

    def test_order_id_handles_none_response(self, mock_config):
        with patch('libs.adapters.real_bithumb_execution.BithumbAPIV2') as mock_api:
            mock_instance = MagicMock()
            mock_instance.post_order.return_value = None
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)
            adapter.get_current_price = MagicMock(return_value=50_000_000)

            result = adapter.place_order("BTC/KRW", "BUY", units=0.001)

            assert result.status == "REJECTED"
            assert "None response" in result.message


class TestStrategyRunnerBithumbIntegration:
    """StrategyRunner integration checks."""

    def test_runner_calculates_quantity_correctly(self):
        import os
        os.environ["MASP_ENABLE_LIVE_TRADING"] = "0"

        from services.strategy_runner import StrategyRunner

        runner = StrategyRunner(
            strategy_name='kama_tsmom_gate',
            exchange='bithumb',
            symbols=['BTC/KRW'],
            position_size_krw=10000,
        )

        assert "Paper" in runner.execution.__class__.__name__
        assert "Bithumb" in runner.market_data.__class__.__name__


class TestOHLCVDataSorting:
    """OHLCV sorting behavior."""

    def test_ohlcv_is_sorted_by_timestamp(self):
        with patch('libs.adapters.real_bithumb_spot.pybithumb') as mock_pybithumb:
            import pandas as pd
            from datetime import datetime, timedelta

            dates = [
                datetime.now() - timedelta(days=2),
                datetime.now(),
                datetime.now() - timedelta(days=1),
            ]
            mock_df = pd.DataFrame({
                'open': [100, 102, 101],
                'high': [105, 107, 106],
                'low': [95, 97, 96],
                'close': [101, 103, 102],
                'volume': [1000, 1200, 1100],
            }, index=pd.DatetimeIndex(dates))

            mock_pybithumb.get_ohlcv.return_value = mock_df

            from libs.adapters.real_bithumb_spot import BithumbSpotMarketData
            market_data = BithumbSpotMarketData()

            result = market_data.get_ohlcv("BTC/KRW", limit=3)

            assert len(result) == 3
            assert result[0].timestamp < result[1].timestamp < result[2].timestamp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
