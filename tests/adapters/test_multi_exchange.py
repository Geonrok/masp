"""Tests for multi-exchange functionality."""

from __future__ import annotations

from unittest.mock import patch

# =============================================================================
# Exchange Registry Tests
# =============================================================================


def test_exchange_status_enum():
    """Test ExchangeStatus enum values."""
    from libs.adapters.exchange_registry import ExchangeStatus

    assert ExchangeStatus.ONLINE.value == "online"
    assert ExchangeStatus.OFFLINE.value == "offline"
    assert ExchangeStatus.MAINTENANCE.value == "maintenance"


def test_exchange_type_enum():
    """Test ExchangeType enum values."""
    from libs.adapters.exchange_registry import ExchangeType

    assert ExchangeType.SPOT.value == "spot"
    assert ExchangeType.FUTURES.value == "futures"
    assert ExchangeType.MARGIN.value == "margin"


def test_exchange_region_enum():
    """Test ExchangeRegion enum values."""
    from libs.adapters.exchange_registry import ExchangeRegion

    assert ExchangeRegion.KOREA.value == "kr"
    assert ExchangeRegion.GLOBAL.value == "global"


def test_exchange_info_dataclass():
    """Test ExchangeInfo dataclass."""
    from libs.adapters.exchange_registry import (
        ExchangeInfo,
        ExchangeRegion,
        ExchangeType,
    )

    info = ExchangeInfo(
        name="test_exchange",
        display_name="Test Exchange",
        exchange_type=ExchangeType.SPOT,
        region=ExchangeRegion.KOREA,
        base_currency="KRW",
    )

    assert info.name == "test_exchange"
    assert info.display_name == "Test Exchange"
    assert info.exchange_type == ExchangeType.SPOT


def test_exchange_info_to_dict():
    """Test ExchangeInfo to_dict method."""
    from libs.adapters.exchange_registry import (
        ExchangeInfo,
        ExchangeRegion,
        ExchangeType,
    )

    info = ExchangeInfo(
        name="test",
        display_name="Test",
        exchange_type=ExchangeType.SPOT,
        region=ExchangeRegion.KOREA,
    )

    data = info.to_dict()

    assert data["name"] == "test"
    assert data["exchange_type"] == "spot"
    assert data["region"] == "kr"


def test_exchange_config_from_env():
    """Test ExchangeConfig from_env method."""
    from libs.adapters.exchange_registry import ExchangeConfig

    with patch.dict(
        "os.environ",
        {
            "UPBIT_ENABLED": "1",
            "UPBIT_API_KEY": "test_key",
            "UPBIT_API_SECRET": "test_secret",
        },
    ):
        config = ExchangeConfig.from_env("upbit")

        assert config.exchange_name == "upbit"
        assert config.enabled is True
        assert config.api_key == "test_key"


def test_exchange_registry_singleton():
    """Test ExchangeRegistry is a singleton."""
    from libs.adapters.exchange_registry import ExchangeRegistry

    registry1 = ExchangeRegistry()
    registry2 = ExchangeRegistry()

    assert registry1 is registry2


def test_exchange_registry_default_exchanges():
    """Test ExchangeRegistry has default exchanges."""
    from libs.adapters.exchange_registry import get_registry

    registry = get_registry()
    exchanges = registry.get_all()

    assert "upbit_spot" in exchanges
    assert "bithumb_spot" in exchanges
    assert "binance_futures" in exchanges


def test_exchange_registry_get_by_type():
    """Test filtering exchanges by type."""
    from libs.adapters.exchange_registry import ExchangeType, get_registry

    registry = get_registry()

    spot_exchanges = registry.get_by_type(ExchangeType.SPOT)

    assert len(spot_exchanges) >= 2
    for ex in spot_exchanges:
        assert ex.exchange_type == ExchangeType.SPOT


def test_exchange_registry_get_by_region():
    """Test filtering exchanges by region."""
    from libs.adapters.exchange_registry import ExchangeRegion, get_registry

    registry = get_registry()

    kr_exchanges = registry.get_by_region(ExchangeRegion.KOREA)

    assert len(kr_exchanges) >= 2
    for ex in kr_exchanges:
        assert ex.region == ExchangeRegion.KOREA


def test_exchange_registry_update_status():
    """Test updating exchange status."""
    from libs.adapters.exchange_registry import ExchangeStatus, get_registry

    registry = get_registry()

    registry.update_status("upbit_spot", ExchangeStatus.ONLINE, latency_ms=50.0)

    info = registry.get("upbit_spot")
    assert info.status == ExchangeStatus.ONLINE
    assert info.latency_ms == 50.0


def test_exchange_registry_summary():
    """Test registry summary."""
    from libs.adapters.exchange_registry import get_registry

    registry = get_registry()
    summary = registry.get_summary()

    assert "total" in summary
    assert "by_type" in summary
    assert "by_region" in summary
    assert summary["total"] >= 3


# =============================================================================
# Multi-Exchange Coordinator Tests
# =============================================================================


def test_multi_exchange_quote_dataclass():
    """Test MultiExchangeQuote dataclass."""
    from libs.adapters.base import MarketQuote
    from libs.adapters.multi_exchange import MultiExchangeQuote

    multi_quote = MultiExchangeQuote(symbol="BTC")

    quote1 = MarketQuote(symbol="BTC", bid=50000, ask=50100)
    quote2 = MarketQuote(symbol="BTC", bid=50050, ask=50150)

    multi_quote.add_quote("exchange1", quote1)
    multi_quote.add_quote("exchange2", quote2)

    assert len(multi_quote.quotes) == 2
    assert multi_quote.best_bid == ("exchange2", 50050)
    assert multi_quote.best_ask == ("exchange1", 50100)


def test_multi_exchange_quote_arbitrage():
    """Test arbitrage detection."""
    from libs.adapters.base import MarketQuote
    from libs.adapters.multi_exchange import MultiExchangeQuote

    multi_quote = MultiExchangeQuote(symbol="BTC")

    # Create arbitrage opportunity: bid on ex2 > ask on ex1
    quote1 = MarketQuote(symbol="BTC", bid=50000, ask=50100)
    quote2 = MarketQuote(symbol="BTC", bid=50200, ask=50300)

    multi_quote.add_quote("exchange1", quote1)
    multi_quote.add_quote("exchange2", quote2)

    arb = multi_quote.get_arbitrage_opportunity()

    assert arb is not None
    assert arb["buy_exchange"] == "exchange1"
    assert arb["sell_exchange"] == "exchange2"
    assert arb["profit_pct"] > 0


def test_multi_exchange_quote_to_dict():
    """Test MultiExchangeQuote to_dict method."""
    from libs.adapters.base import MarketQuote
    from libs.adapters.multi_exchange import MultiExchangeQuote

    multi_quote = MultiExchangeQuote(symbol="BTC")
    multi_quote.add_quote("ex1", MarketQuote(symbol="BTC", bid=50000, ask=50100))

    data = multi_quote.to_dict()

    assert data["symbol"] == "BTC"
    assert "quotes" in data
    assert "ex1" in data["quotes"]


def test_multi_exchange_coordinator_init():
    """Test MultiExchangeCoordinator initialization."""
    from libs.adapters.multi_exchange import MultiExchangeCoordinator

    # Should not raise
    coordinator = MultiExchangeCoordinator(exchanges=["mock"])

    assert coordinator is not None


def test_create_multi_exchange_coordinator():
    """Test factory function."""
    from libs.adapters.multi_exchange import create_multi_exchange_coordinator

    coordinator = create_multi_exchange_coordinator(exchanges=["mock"])

    assert coordinator is not None


# =============================================================================
# Provider Tests
# =============================================================================


def test_multi_exchange_provider_import():
    """Test multi-exchange provider imports."""
    from services.dashboard.providers.multi_exchange_provider import (
        get_exchange_list,
        get_exchange_status,
        get_price_comparison,
    )

    assert callable(get_exchange_list)
    assert callable(get_exchange_status)
    assert callable(get_price_comparison)


def test_get_exchange_list_returns_list():
    """Test get_exchange_list returns a list."""
    from services.dashboard.providers.multi_exchange_provider import get_exchange_list

    result = get_exchange_list()

    assert isinstance(result, list)
    assert len(result) >= 1


def test_get_registry_summary():
    """Test get_registry_summary returns dict."""
    from services.dashboard.providers.multi_exchange_provider import (
        get_registry_summary,
    )

    result = get_registry_summary()

    assert isinstance(result, dict)
    assert "total" in result


# =============================================================================
# Component Tests
# =============================================================================


def test_multi_exchange_view_import():
    """Test multi-exchange view component imports."""
    from services.dashboard.components.multi_exchange_view import (
        render_exchange_list,
        render_multi_exchange_view,
    )

    assert callable(render_multi_exchange_view)
    assert callable(render_exchange_list)


def test_get_demo_exchanges():
    """Test demo exchanges generation."""
    from services.dashboard.components.multi_exchange_view import _get_demo_exchanges

    exchanges = _get_demo_exchanges()

    assert len(exchanges) >= 2
    assert any(e["name"] == "upbit_spot" for e in exchanges)


def test_get_demo_price_comparison():
    """Test demo price comparison generation."""
    from services.dashboard.components.multi_exchange_view import (
        _get_demo_price_comparison,
    )

    comparison = _get_demo_price_comparison()

    assert "symbol" in comparison
    assert "exchanges" in comparison
    assert "best_bid" in comparison
    assert "best_ask" in comparison


def test_get_status_color():
    """Test status color mapping."""
    from services.dashboard.components.multi_exchange_view import _get_status_color

    assert _get_status_color("online") == "green"
    assert _get_status_color("offline") == "red"
    assert _get_status_color("unknown") == "gray"
