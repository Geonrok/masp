"""
Bithumb order contract tests (API v2).
"""

from unittest.mock import MagicMock, patch

import pytest

from libs.adapters.bithumb_api_v2 import BithumbAPIV2


class TestBithumbOrderContract:
    """Bithumb order contract tests."""

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

    def test_buy_calls_api_v2_with_krw_price(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
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
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_api.return_value = MagicMock()

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            with pytest.raises(ValueError, match="BUY requires exactly one of"):
                adapter.place_order("BTC/KRW", "BUY")

    def test_buy_rejects_ambiguous_inputs_both_set(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_api.return_value = MagicMock()

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            with pytest.raises(ValueError, match="not both"):
                adapter.place_order("BTC/KRW", "BUY", units=0.01, amount_krw=50000)

    def test_sell_requires_units(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_api.return_value = MagicMock()

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            with pytest.raises(ValueError, match="SELL requires units"):
                adapter.place_order("BTC/KRW", "SELL")

    def test_sell_rejects_amount_krw(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_api.return_value = MagicMock()

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            with pytest.raises(ValueError, match="SELL does not accept amount_krw"):
                adapter.place_order("BTC/KRW", "SELL", units=0.01, amount_krw=50000)

    def test_sell_calls_api_v2_with_units(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
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
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
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

        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
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
        config.bithumb_api_key.get_secret_value.return_value = (
            "test_api_key_12345678901234567890"
        )
        config.bithumb_secret_key.get_secret_value.return_value = (
            "test_secret_key_1234567890123456"
        )
        config.is_kill_switch_active.return_value = False
        config.max_order_value_krw = 10_000_000
        return config

    def test_order_id_from_uuid_response(self, mock_config):
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


class TestStrategyRunnerBithumbIntegration:
    """StrategyRunner integration checks."""

    def test_runner_calculates_quantity_correctly(self):
        import os

        os.environ["MASP_ENABLE_LIVE_TRADING"] = "0"

        from services.strategy_runner import StrategyRunner

        runner = StrategyRunner(
            strategy_name="kama_tsmom_gate",
            exchange="bithumb",
            symbols=["BTC/KRW"],
            position_size_krw=10000,
        )

        assert "Paper" in runner.execution.__class__.__name__
        assert "Bithumb" in runner.market_data.__class__.__name__


class TestOHLCVDataSorting:
    """OHLCV sorting behavior."""

    def test_ohlcv_is_sorted_by_timestamp(self):
        with patch("libs.adapters.real_bithumb_spot.pybithumb") as mock_pybithumb:
            from datetime import datetime, timedelta

            import pandas as pd

            dates = [
                datetime.now() - timedelta(days=2),
                datetime.now(),
                datetime.now() - timedelta(days=1),
            ]
            mock_df = pd.DataFrame(
                {
                    "open": [100, 102, 101],
                    "high": [105, 107, 106],
                    "low": [95, 97, 96],
                    "close": [101, 103, 102],
                    "volume": [1000, 1200, 1100],
                },
                index=pd.DatetimeIndex(dates),
            )

            mock_pybithumb.get_ohlcv.return_value = mock_df

            from libs.adapters.real_bithumb_spot import BithumbSpotMarketData

            market_data = BithumbSpotMarketData()

            result = market_data.get_ohlcv("BTC/KRW", limit=3)

            assert len(result) == 3
            assert result[0].timestamp < result[1].timestamp < result[2].timestamp


class TestOrderStatusTracking:
    """Phase 8: Order status tracking tests."""

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

    def test_get_order_status_parses_response(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_order.return_value = {
                "uuid": "order_123",
                "market": "KRW-BTC",
                "side": "bid",
                "ord_type": "price",
                "state": "done",
                "volume": "0.001",
                "remaining_volume": "0",
                "executed_volume": "0.001",
                "price": "50000",
                "trades_count": 1,
                "created_at": "2024-01-01T00:00:00",
                "paid_fee": "12.5",
                "locked": "0",
            }
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            status = adapter.get_order_status("order_123")

            assert status is not None
            assert status.order_id == "order_123"
            assert status.state == "done"
            assert status.is_done is True
            assert status.is_pending is False
            assert status.fill_ratio == 1.0

    def test_get_order_status_returns_none_on_error(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_order.side_effect = Exception("Network error")
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            status = adapter.get_order_status("order_123")

            assert status is None

    def test_order_status_properties(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_api.return_value = MagicMock()

            from datetime import datetime

            from libs.adapters.real_bithumb_execution import OrderState, OrderStatus

            # Test pending order
            pending_status = OrderStatus(
                order_id="test_1",
                market="KRW-BTC",
                side="bid",
                ord_type="limit",
                state="wait",
                volume=0.01,
                remaining_volume=0.005,
                executed_volume=0.005,
                price=50000000,
                avg_price=50000000,
                trades_count=1,
                created_at=datetime.now(),
            )

            assert pending_status.is_pending is True
            assert pending_status.is_done is False
            assert pending_status.is_canceled is False
            assert pending_status.fill_ratio == 0.5

            # Test done order
            done_status = OrderStatus(
                order_id="test_2",
                market="KRW-BTC",
                side="ask",
                ord_type="market",
                state="done",
                volume=0.01,
                remaining_volume=0,
                executed_volume=0.01,
                price=None,
                avg_price=50100000,
                trades_count=2,
                created_at=datetime.now(),
            )

            assert done_status.is_done is True
            assert done_status.is_pending is False
            assert done_status.fill_ratio == 1.0

            # Test canceled order
            canceled_status = OrderStatus(
                order_id="test_3",
                market="KRW-BTC",
                side="bid",
                ord_type="limit",
                state="cancel",
                volume=0.01,
                remaining_volume=0.01,
                executed_volume=0,
                price=49000000,
                avg_price=None,
                trades_count=0,
                created_at=datetime.now(),
            )

            assert canceled_status.is_canceled is True
            assert canceled_status.is_done is False
            assert canceled_status.fill_ratio == 0.0

    def test_get_open_orders(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_orders.return_value = [
                {
                    "uuid": "order_1",
                    "market": "KRW-BTC",
                    "side": "bid",
                    "ord_type": "limit",
                    "state": "wait",
                    "volume": "0.001",
                    "remaining_volume": "0.001",
                    "executed_volume": "0",
                    "price": "50000000",
                    "trades_count": 0,
                    "created_at": "2024-01-01T00:00:00",
                },
                {
                    "uuid": "order_2",
                    "market": "KRW-ETH",
                    "side": "ask",
                    "ord_type": "limit",
                    "state": "wait",
                    "volume": "0.1",
                    "remaining_volume": "0.05",
                    "executed_volume": "0.05",
                    "price": "3000000",
                    "trades_count": 1,
                    "created_at": "2024-01-01T00:01:00",
                },
            ]
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            orders = adapter.get_open_orders()

            assert len(orders) == 2
            assert orders[0].order_id == "order_1"
            assert orders[1].fill_ratio == 0.5

    def test_get_recent_orders_with_filter(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_orders.return_value = []
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            adapter.get_recent_orders(symbol="BTC/KRW", limit=10, states=["done"])

            mock_instance.get_orders.assert_called_once_with(
                market="KRW-BTC",
                states=["done"],
                limit=10,
            )


class TestPositionSync:
    """Phase 8-2: Position synchronization tests."""

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

    def test_sync_positions_basic(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_accounts.return_value = [
                {"currency": "KRW", "balance": "1000000", "locked": "0"},
                {
                    "currency": "BTC",
                    "balance": "0.01",
                    "locked": "0.001",
                    "avg_buy_price": "50000000",
                },
                {
                    "currency": "ETH",
                    "balance": "0.5",
                    "locked": "0",
                    "avg_buy_price": "3000000",
                },
                {"currency": "DOGE", "balance": "0.000001", "locked": "0"},  # Too small
            ]
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            positions = adapter.sync_positions()

            assert len(positions) == 2  # BTC and ETH only
            assert "BTC/KRW" in positions
            assert "ETH/KRW" in positions
            assert "DOGE/KRW" not in positions  # Dust excluded
            assert positions["BTC/KRW"]["balance"] == 0.01
            assert positions["BTC/KRW"]["locked"] == 0.001
            assert positions["BTC/KRW"]["total"] == 0.011
            assert positions["BTC/KRW"]["avg_buy_price"] == 50000000

    def test_sync_positions_with_filter(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_accounts.return_value = [
                {
                    "currency": "BTC",
                    "balance": "0.01",
                    "locked": "0",
                    "avg_buy_price": "50000000",
                },
                {
                    "currency": "ETH",
                    "balance": "0.5",
                    "locked": "0",
                    "avg_buy_price": "3000000",
                },
            ]
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            positions = adapter.sync_positions(["BTC/KRW"])

            assert len(positions) == 1
            assert "BTC/KRW" in positions
            assert "ETH/KRW" not in positions

    def test_get_position_single(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_accounts.return_value = [
                {
                    "currency": "BTC",
                    "balance": "0.05",
                    "locked": "0",
                    "avg_buy_price": "45000000",
                },
            ]
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            position = adapter.get_position("BTC/KRW")

            assert position is not None
            assert position["balance"] == 0.05
            assert position["avg_buy_price"] == 45000000

    def test_get_position_value(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_accounts.return_value = [
                {
                    "currency": "BTC",
                    "balance": "0.1",
                    "locked": "0",
                    "avg_buy_price": "50000000",
                },
            ]
            mock_instance.get_ticker.return_value = [{"trade_price": 55000000}]
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            value = adapter.get_position_value("BTC/KRW")

            assert value == 5500000  # 0.1 * 55,000,000

    def test_get_pnl_calculation(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_accounts.return_value = [
                {
                    "currency": "BTC",
                    "balance": "0.1",
                    "locked": "0",
                    "avg_buy_price": "50000000",
                },
            ]
            mock_instance.get_ticker.return_value = [{"trade_price": 55000000}]
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            pnl = adapter.get_pnl("BTC/KRW")

            assert pnl is not None
            assert pnl["quantity"] == 0.1
            assert pnl["avg_buy_price"] == 50000000
            assert pnl["current_price"] == 55000000
            assert pnl["cost_basis"] == 5000000  # 0.1 * 50M
            assert pnl["current_value"] == 5500000  # 0.1 * 55M
            assert pnl["unrealized_pnl"] == 500000  # 5.5M - 5M
            assert abs(pnl["unrealized_pnl_pct"] - 10.0) < 0.01  # +10%

    def test_get_total_portfolio_value(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_accounts.return_value = [
                {"currency": "KRW", "balance": "1000000", "locked": "0"},
                {
                    "currency": "BTC",
                    "balance": "0.1",
                    "locked": "0",
                    "avg_buy_price": "50000000",
                },
            ]
            mock_instance.get_ticker.return_value = [{"trade_price": 55000000}]
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            portfolio = adapter.get_total_portfolio_value()

            assert portfolio["krw_balance"] == 1000000
            assert portfolio["positions_value"] == 5500000  # 0.1 BTC @ 55M
            assert portfolio["total_value"] == 6500000
            assert "BTC/KRW" in portfolio["positions"]


class TestRetryAndRecovery:
    """Phase 8-3: Retry and error recovery tests."""

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

    def test_retry_operation_succeeds_after_failures(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_api.return_value = MagicMock()

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            # Operation that fails twice then succeeds
            call_count = 0

            def flaky_operation():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("Network error")
                return "success"

            result = adapter._retry_operation(
                flaky_operation,
                "test_op",
                max_retries=3,
                base_delay=0.01,
            )

            assert result == "success"
            assert call_count == 3

    def test_retry_operation_raises_after_max_retries(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_api.return_value = MagicMock()

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            def always_fails():
                raise ConnectionError("Persistent error")

            with pytest.raises(ConnectionError):
                adapter._retry_operation(
                    always_fails,
                    "test_op",
                    max_retries=2,
                    base_delay=0.01,
                )

    def test_retry_operation_non_retryable_exception(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_api.return_value = MagicMock()

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            def value_error():
                raise ValueError("Invalid input")

            # ValueError is not in retryable_exceptions, should raise immediately
            with pytest.raises(ValueError):
                adapter._retry_operation(
                    value_error,
                    "test_op",
                    max_retries=3,
                    base_delay=0.01,
                )

    def test_calculate_backoff_with_jitter(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_api.return_value = MagicMock()

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            # Test backoff values
            delay_0 = BithumbExecutionAdapter._calculate_backoff(0, 1.0)
            delay_1 = BithumbExecutionAdapter._calculate_backoff(1, 1.0)
            delay_2 = BithumbExecutionAdapter._calculate_backoff(2, 1.0)

            # Base values: 1.0, 2.0, 4.0 with ±20% jitter
            assert 0.8 <= delay_0 <= 1.2
            assert 1.6 <= delay_1 <= 2.4
            assert 3.2 <= delay_2 <= 4.8

    def test_safe_cancel_order_already_filled(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_order.return_value = {
                "uuid": "order_123",
                "market": "KRW-BTC",
                "side": "bid",
                "ord_type": "price",
                "state": "done",
                "volume": "0.001",
                "remaining_volume": "0",
                "executed_volume": "0.001",
                "price": "50000",
                "trades_count": 1,
                "created_at": "2024-01-01T00:00:00",
            }
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            # Should return False since order is already filled
            result = adapter.safe_cancel_order("order_123")

            assert result is False
            mock_instance.cancel_order.assert_not_called()

    def test_safe_cancel_order_already_canceled(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_order.return_value = {
                "uuid": "order_123",
                "market": "KRW-BTC",
                "side": "bid",
                "ord_type": "limit",
                "state": "cancel",
                "volume": "0.001",
                "remaining_volume": "0.001",
                "executed_volume": "0",
                "price": "50000000",
                "trades_count": 0,
                "created_at": "2024-01-01T00:00:00",
            }
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            result = adapter.safe_cancel_order("order_123")

            assert result is True
            mock_instance.cancel_order.assert_not_called()

    def test_recover_from_connection_error(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_accounts.return_value = [
                {"currency": "KRW", "balance": "1000000"}
            ]
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            error = ConnectionError("Network error")
            context = {"symbol": "BTC/KRW", "side": "BUY"}

            result = adapter.recover_from_error(error, context)

            assert result["recovered"] is True
            assert result["action"] == "connection_restored"

    def test_recover_from_error_with_order_id(self, mock_config):
        with patch("libs.adapters.real_bithumb_execution.BithumbAPIV2") as mock_api:
            mock_instance = MagicMock()
            mock_instance.get_order.return_value = {
                "uuid": "order_123",
                "market": "KRW-BTC",
                "side": "bid",
                "ord_type": "price",
                "state": "done",
                "volume": "0.001",
                "remaining_volume": "0",
                "executed_volume": "0.001",
                "price": "50000",
                "trades_count": 1,
                "created_at": "2024-01-01T00:00:00",
            }
            mock_api.return_value = mock_instance

            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter

            adapter = BithumbExecutionAdapter(mock_config)

            error = ValueError("Some error")
            context = {"symbol": "BTC/KRW", "side": "BUY", "order_id": "order_123"}

            result = adapter.recover_from_error(error, context)

            assert result["recovered"] is True
            assert result["action"] == "order_status_found"
            assert result["details"]["state"] == "done"


class TestBithumbAPIV2Orders:
    """Tests for new BithumbAPIV2 order methods."""

    def test_get_orders_with_states(self):
        session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        session.get.return_value = mock_response

        client = BithumbAPIV2(
            "access_key_12345678", "secret_key_12345678", session=session
        )
        result = client.get_orders(market="KRW-BTC", states=["wait", "watch"])

        assert result == []
        session.get.assert_called_once()

    def test_get_orders_chance(self):
        session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "bid_fee": "0.0025",
            "ask_fee": "0.0025",
            "market": {"id": "KRW-BTC"},
        }
        mock_response.raise_for_status = MagicMock()
        session.get.return_value = mock_response

        client = BithumbAPIV2(
            "access_key_12345678", "secret_key_12345678", session=session
        )
        result = client.get_orders_chance("KRW-BTC")

        assert result["bid_fee"] == "0.0025"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
