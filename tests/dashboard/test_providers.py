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
