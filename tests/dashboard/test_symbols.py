"""Tests for symbol format utilities."""

from services.dashboard.utils.symbols import (
    convert_symbols_to_dashboard,
    dashboard_to_upbit,
    upbit_to_dashboard,
)


class TestSymbolConversion:
    def test_upbit_to_dashboard(self):
        assert upbit_to_dashboard("KRW-BTC") == "BTC/KRW"
        assert upbit_to_dashboard("KRW-ETH") == "ETH/KRW"

    def test_upbit_to_dashboard_idempotent(self):
        assert upbit_to_dashboard("BTC/KRW") == "BTC/KRW"

    def test_dashboard_to_upbit(self):
        assert dashboard_to_upbit("BTC/KRW") == "KRW-BTC"
        assert dashboard_to_upbit("ETH/KRW") == "KRW-ETH"

    def test_dashboard_to_upbit_idempotent(self):
        assert dashboard_to_upbit("KRW-BTC") == "KRW-BTC"

    def test_convert_list(self):
        result = convert_symbols_to_dashboard(["KRW-BTC", "KRW-ETH"])
        assert result == ["BTC/KRW", "ETH/KRW"]
