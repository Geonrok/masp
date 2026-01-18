"""Tests for price refresh utilities."""
from __future__ import annotations

from unittest.mock import patch

import streamlit as st

from services.dashboard.utils import price_refresh


def setup_function():
    st.session_state.clear()


def test_update_and_get_cache():
    prices = {"BTC": 50000.0, "ETH": 3000.0}
    price_refresh.update_price_cache(prices)

    cache = price_refresh.get_cached_prices()
    assert cache is not None
    assert cache.prices == prices


def test_cache_stale_when_empty():
    assert price_refresh.is_cache_stale() is True


def test_cache_fresh_after_update():
    price_refresh.update_price_cache({"BTC": 50000.0})
    assert price_refresh.is_cache_stale(ttl=10) is False


def test_cache_stale_after_ttl():
    with patch("services.dashboard.utils.price_refresh.time.time", return_value=1000.0):
        price_refresh.update_price_cache({"BTC": 50000.0})

    with patch("services.dashboard.utils.price_refresh.time.time", return_value=1011.0):
        assert price_refresh.is_cache_stale(ttl=10) is True


def test_clear_cache():
    price_refresh.update_price_cache({"BTC": 50000.0})
    price_refresh.clear_price_cache()
    assert price_refresh.get_cached_prices() is None
