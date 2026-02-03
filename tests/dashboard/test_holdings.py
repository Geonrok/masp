"""Tests for holdings utilities."""

import os
from unittest.mock import patch

from services.dashboard.utils.holdings import (
    get_holding_symbols,
    is_private_api_enabled,
)


class TestPrivateApiEnabled:
    def test_enabled(self):
        with patch.dict(os.environ, {"MASP_ENABLE_LIVE_TRADING": "1"}, clear=True):
            assert is_private_api_enabled() is True

    def test_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            assert is_private_api_enabled() is False


class TestGetHoldingSymbols:
    def test_filters_krw(self):
        mock_holdings = [
            {"currency": "KRW", "balance": "100000"},
            {"currency": "BTC", "balance": "0.01"},
        ]
        with patch(
            "services.dashboard.utils.holdings.get_holdings_upbit",
            return_value=mock_holdings,
        ):
            symbols = get_holding_symbols()
            assert "KRW/KRW" not in symbols
            assert "BTC/KRW" in symbols

    def test_filters_zero_balance(self):
        mock_holdings = [
            {"currency": "BTC", "balance": "0.01"},
            {"currency": "ETH", "balance": "0"},
        ]
        with patch(
            "services.dashboard.utils.holdings.get_holdings_upbit",
            return_value=mock_holdings,
        ):
            symbols = get_holding_symbols()
            assert "BTC/KRW" in symbols
            assert "ETH/KRW" not in symbols
