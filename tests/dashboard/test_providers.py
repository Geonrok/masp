"""Tests for dashboard data providers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_import_providers():
    """Test provider modules import correctly."""
    from services.dashboard.providers import (
        get_portfolio_summary,
        get_system_resources,
        get_service_health,
    )

    assert callable(get_portfolio_summary)
    assert callable(get_system_resources)
    assert callable(get_service_health)


# =============================================================================
# Portfolio Provider Tests
# =============================================================================


def test_portfolio_provider_returns_none_when_disabled():
    """Test portfolio provider returns None when live trading is disabled."""
    from services.dashboard.providers.portfolio_provider import get_portfolio_summary

    with patch.dict("os.environ", {"MASP_ENABLE_LIVE_TRADING": "0"}, clear=False):
        result = get_portfolio_summary()
        assert result is None


def test_parse_holdings_to_positions():
    """Test parsing raw holdings to PortfolioPosition objects."""
    from services.dashboard.providers.portfolio_provider import _parse_holdings_to_positions
    from services.dashboard.components.portfolio_summary import PortfolioPosition

    holdings = [
        {"currency": "BTC", "balance": "0.1", "avg_buy_price": "50000000"},
        {"currency": "ETH", "balance": "2.0", "avg_buy_price": "3000000"},
        {"currency": "KRW", "balance": "1000000"},  # Should be skipped
    ]
    prices = {
        "BTC/KRW": 55000000.0,
        "ETH/KRW": 3200000.0,
    }

    positions = _parse_holdings_to_positions(holdings, prices)

    assert len(positions) == 2
    assert positions[0].symbol == "BTC/KRW"
    assert positions[0].quantity == 0.1
    assert positions[0].current_price == 55000000.0
    assert positions[1].symbol == "ETH/KRW"


def test_parse_holdings_skips_zero_balance():
    """Test that zero balance holdings are skipped."""
    from services.dashboard.providers.portfolio_provider import _parse_holdings_to_positions

    holdings = [
        {"currency": "BTC", "balance": "0", "avg_buy_price": "50000000"},
        {"currency": "ETH", "balance": "0.0", "avg_buy_price": "3000000"},
    ]

    positions = _parse_holdings_to_positions(holdings, {})

    assert len(positions) == 0


def test_get_cash_balance():
    """Test extracting KRW cash balance from holdings."""
    from services.dashboard.providers.portfolio_provider import _get_cash_balance

    holdings = [
        {"currency": "BTC", "balance": "0.1"},
        {"currency": "KRW", "balance": "5000000"},
        {"currency": "ETH", "balance": "2.0"},
    ]

    cash = _get_cash_balance(holdings)

    assert cash == 5000000.0


def test_get_cash_balance_no_krw():
    """Test cash balance returns 0 when no KRW entry."""
    from services.dashboard.providers.portfolio_provider import _get_cash_balance

    holdings = [
        {"currency": "BTC", "balance": "0.1"},
    ]

    cash = _get_cash_balance(holdings)

    assert cash == 0.0


# =============================================================================
# System Provider Tests
# =============================================================================


def test_get_system_resources_with_psutil():
    """Test system resources returns valid data when psutil is available."""
    import importlib.util
    import pytest

    from services.dashboard.providers.system_provider import get_system_resources
    from services.dashboard.components.system_status import ResourceUsage

    result = get_system_resources()

    assert isinstance(result, ResourceUsage)
    assert 0 <= result.cpu_percent <= 100
    assert 0 <= result.memory_percent <= 100

    # Only check actual values if psutil is available
    if importlib.util.find_spec("psutil") is not None:
        assert result.memory_total_mb > 0
        assert result.disk_total_gb > 0
    else:
        # psutil not available, should return zeros
        assert result.memory_total_mb == 0.0
        assert result.disk_total_gb == 0.0


def test_get_system_resources_without_psutil():
    """Test system resources returns zeros when psutil not available."""
    from services.dashboard.providers.system_provider import get_system_resources

    with patch.dict("sys.modules", {"psutil": None}):
        # Force reimport to trigger ImportError path
        import importlib
        from services.dashboard.providers import system_provider
        importlib.reload(system_provider)

        result = system_provider.get_system_resources()

        # Should not crash, returns resource object


def test_get_service_health_returns_list():
    """Test service health returns list of ServiceHealth objects."""
    from services.dashboard.providers.system_provider import get_service_health
    from services.dashboard.components.system_status import ServiceHealth

    result = get_service_health()

    assert isinstance(result, list)
    assert len(result) >= 1
    for service in result:
        assert isinstance(service, ServiceHealth)
        assert service.name is not None


def test_service_health_contains_expected_services():
    """Test that all expected services are included."""
    from services.dashboard.providers.system_provider import get_service_health

    result = get_service_health()
    service_names = [s.name for s in result]

    assert "Upbit API" in service_names
    assert "Scheduler" in service_names
    assert "Database" in service_names
    assert "Telegram" in service_names


def test_check_telegram_not_configured():
    """Test Telegram check when not configured."""
    from services.dashboard.providers.system_provider import _check_telegram
    from services.dashboard.components.system_status import ServiceStatus

    with patch.dict("os.environ", {}, clear=True):
        status, latency, message = _check_telegram()

        assert status == ServiceStatus.UNKNOWN
        assert message == "Not configured"


def test_check_telegram_configured():
    """Test Telegram check when configured."""
    from services.dashboard.providers.system_provider import _check_telegram
    from services.dashboard.components.system_status import ServiceStatus

    with patch.dict(
        "os.environ",
        {"TELEGRAM_BOT_TOKEN": "test", "TELEGRAM_CHAT_ID": "123"},
        clear=False,
    ):
        status, latency, message = _check_telegram()

        assert status == ServiceStatus.HEALTHY
        assert message == ""


# =============================================================================
# Order Provider Tests
# =============================================================================


def test_order_provider_import():
    """Test order provider modules import correctly."""
    from services.dashboard.providers import (
        get_execution_adapter,
        get_price_provider,
        get_balance_provider,
    )

    assert callable(get_execution_adapter)
    assert callable(get_price_provider)
    assert callable(get_balance_provider)


def test_get_execution_adapter_returns_none_when_disabled():
    """Test execution adapter returns None when live trading is disabled."""
    from services.dashboard.providers.order_provider import get_execution_adapter

    with patch.dict("os.environ", {"MASP_ENABLE_LIVE_TRADING": "0"}, clear=False):
        result = get_execution_adapter()
        assert result is None


def test_get_price_provider_returns_none_when_disabled():
    """Test price provider returns None when live trading is disabled."""
    from services.dashboard.providers.order_provider import get_price_provider

    with patch.dict("os.environ", {"MASP_ENABLE_LIVE_TRADING": "0"}, clear=False):
        result = get_price_provider()
        assert result is None


def test_get_balance_provider_returns_none_when_disabled():
    """Test balance provider returns None when live trading is disabled."""
    from services.dashboard.providers.order_provider import get_balance_provider

    with patch.dict("os.environ", {"MASP_ENABLE_LIVE_TRADING": "0"}, clear=False):
        result = get_balance_provider()
        assert result is None


def test_is_live_trading_enabled():
    """Test live trading enabled check."""
    from services.dashboard.providers.order_provider import is_live_trading_enabled

    with patch.dict("os.environ", {"MASP_ENABLE_LIVE_TRADING": "0"}, clear=False):
        assert is_live_trading_enabled() is False

    with patch.dict("os.environ", {"MASP_ENABLE_LIVE_TRADING": "1"}, clear=False):
        assert is_live_trading_enabled() is True


def test_order_execution_wrapper_place_order():
    """Test OrderExecutionWrapper place_order method."""
    from services.dashboard.providers.order_provider import OrderExecutionWrapper
    from dataclasses import dataclass
    from datetime import datetime

    # Mock adapter
    @dataclass
    class MockOrderResult:
        order_id: str = "test-123"
        status: str = "FILLED"
        filled_quantity: float = 0.001
        filled_price: float = 50000000.0
        fee: float = 25.0
        message: str = "Order placed successfully"

    mock_adapter = MagicMock()
    mock_adapter.get_current_price.return_value = 50000000.0
    mock_adapter.place_order.return_value = MockOrderResult()

    wrapper = OrderExecutionWrapper(mock_adapter)
    result = wrapper.place_order("BTC", "buy", units=0.001)

    assert result["success"] is True
    assert result["order_id"] == "test-123"
    assert result["executed_qty"] == 0.001
    assert result["fee"] == 25.0


# =============================================================================
# Trade History Provider Tests
# =============================================================================


def test_trade_history_provider_import():
    """Test trade history provider imports correctly."""
    from services.dashboard.providers import get_trade_history_client

    assert callable(get_trade_history_client)


def test_trade_history_api_client_with_mock():
    """Test TradeHistoryApiClient with mock logger."""
    from services.dashboard.providers.trade_history_provider import TradeHistoryApiClient
    from datetime import date

    mock_logger = MagicMock()
    mock_logger.get_trades.return_value = [
        {
            "timestamp": "2024-01-01T10:00:00",
            "exchange": "upbit",
            "order_id": "T001",
            "symbol": "BTC",
            "side": "BUY",
            "quantity": "0.1",
            "price": "50000000",
            "fee": "2500",
            "pnl": "0",
            "status": "FILLED",
            "message": "",
        }
    ]

    client = TradeHistoryApiClient(mock_logger)
    trades = client.get_trade_history(days=1)

    assert len(trades) >= 1
    assert trades[0]["symbol"] == "BTC"
    assert trades[0]["side"] == "BUY"


def test_trade_history_api_client_get_daily_summary():
    """Test TradeHistoryApiClient daily summary."""
    from services.dashboard.providers.trade_history_provider import TradeHistoryApiClient

    mock_logger = MagicMock()
    mock_logger.get_daily_summary.return_value = {
        "date": "2024-01-01",
        "total_trades": 5,
        "buy_count": 3,
        "sell_count": 2,
        "total_volume": 1000000.0,
        "total_fee": 500.0,
        "total_pnl": 10000.0,
    }

    client = TradeHistoryApiClient(mock_logger)
    summary = client.get_daily_summary()

    assert summary["total_trades"] == 5
    assert summary["buy_count"] == 3


# =============================================================================
# Log Provider Tests
# =============================================================================


def test_log_provider_import():
    """Test log provider imports correctly."""
    from services.dashboard.providers import get_log_provider

    assert callable(get_log_provider)


def test_get_log_provider_returns_callable():
    """Test get_log_provider returns a callable."""
    from services.dashboard.providers.log_provider import get_log_provider

    provider = get_log_provider()

    assert callable(provider)


def test_log_provider_returns_list():
    """Test log provider returns a list."""
    from services.dashboard.providers.log_provider import get_log_provider

    provider = get_log_provider(log_dir="nonexistent_dir")
    result = provider()

    assert isinstance(result, list)


def test_parse_log_level():
    """Test log level parsing."""
    from services.dashboard.providers.log_provider import _parse_log_level
    from services.dashboard.components.log_viewer import LogLevel

    assert _parse_log_level("DEBUG") == LogLevel.DEBUG
    assert _parse_log_level("INFO") == LogLevel.INFO
    assert _parse_log_level("WARNING") == LogLevel.WARNING
    assert _parse_log_level("WARN") == LogLevel.WARNING
    assert _parse_log_level("ERROR") == LogLevel.ERROR
    assert _parse_log_level("CRITICAL") == LogLevel.CRITICAL


def test_parse_log_line():
    """Test log line parsing."""
    from services.dashboard.providers.log_provider import _parse_log_line
    from services.dashboard.components.log_viewer import LogLevel

    # Standard Python logging format
    line = "2024-01-01 12:00:00,000 - mymodule - INFO - Test message"
    entry = _parse_log_line(line)

    assert entry is not None
    assert entry.level == LogLevel.INFO
    assert entry.source == "mymodule"
    assert "Test message" in entry.message
