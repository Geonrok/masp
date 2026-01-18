"""Tests for trade history component."""
from __future__ import annotations

from datetime import datetime, timedelta


def test_import_trade_history():
    from services.dashboard.components import trade_history

    assert hasattr(trade_history, "render_trade_history_panel")
    assert hasattr(trade_history, "_filter_trades")
    assert hasattr(trade_history, "_paginate")


def test_filter_trades_by_exchange():
    from services.dashboard.components.trade_history import _filter_trades

    trades = [
        {"exchange": "upbit", "symbol": "BTC"},
        {"exchange": "bithumb", "symbol": "ETH"},
        {"exchange": "upbit", "symbol": "XRP"},
    ]

    result = _filter_trades(trades, exchange="UPBIT")
    assert len(result) == 2
    assert all(t["exchange"] == "upbit" for t in result)


def test_filter_trades_by_symbol():
    from services.dashboard.components.trade_history import _filter_trades

    trades = [
        {"exchange": "upbit", "symbol": "BTC"},
        {"exchange": "upbit", "symbol": "ETH"},
        {"exchange": "upbit", "symbol": "BTC"},
    ]

    result = _filter_trades(trades, symbol="BTC")
    assert len(result) == 2


def test_filter_trades_by_date():
    from services.dashboard.components.trade_history import _filter_trades

    now = datetime.now()
    trades = [
        {"timestamp": now - timedelta(days=5)},
        {"timestamp": now - timedelta(days=2)},
        {"timestamp": now},
    ]

    result = _filter_trades(
        trades,
        start_date=now - timedelta(days=3),
        end_date=now,
    )
    assert len(result) == 2


def test_filter_trades_all_filter():
    from services.dashboard.components.trade_history import _filter_trades

    trades = [{"exchange": "upbit", "symbol": "BTC"}]
    result = _filter_trades(trades, exchange="ALL", symbol="ALL")
    assert len(result) == 1


def test_paginate_basic():
    from services.dashboard.components.trade_history import _paginate

    items = list(range(50))
    page_items, total_pages, clamped_page = _paginate(items, page=1, per_page=20)

    assert len(page_items) == 20
    assert page_items[0] == 0
    assert total_pages == 3
    assert clamped_page == 1


def test_paginate_last_page():
    from services.dashboard.components.trade_history import _paginate

    items = list(range(50))
    page_items, total_pages, clamped_page = _paginate(items, page=3, per_page=20)

    assert len(page_items) == 10
    assert page_items[0] == 40
    assert clamped_page == 3


def test_paginate_empty():
    from services.dashboard.components.trade_history import _paginate

    page_items, total_pages, clamped_page = _paginate([], page=1, per_page=20)

    assert len(page_items) == 0
    assert total_pages == 1
    assert clamped_page == 1


def test_get_demo_trades():
    from services.dashboard.components.trade_history import _get_demo_trades

    trades = _get_demo_trades()
    assert len(trades) >= 2
    assert "timestamp" in trades[0]
    assert "exchange" in trades[0]
    assert "symbol" in trades[0]


def test_paginate_clamp_high_page():
    """Test pagination clamps page when exceeding total."""
    from services.dashboard.components.trade_history import _paginate

    items = list(range(10))
    page_items, total_pages, clamped_page = _paginate(items, page=5, per_page=20)

    assert clamped_page == 1
    assert len(page_items) == 10


def test_paginate_clamp_negative_page():
    """Test pagination clamps negative page to 1."""
    from services.dashboard.components.trade_history import _paginate

    items = list(range(50))
    page_items, total_pages, clamped_page = _paginate(items, page=-1, per_page=20)

    assert clamped_page == 1
    assert page_items[0] == 0


def test_get_filter_hash():
    """Test filter hash generation for change detection."""
    from services.dashboard.components.trade_history import _get_filter_hash

    hash1 = _get_filter_hash("ALL", "BTC", "2024-01-01", "2024-01-31")
    hash2 = _get_filter_hash("ALL", "ETH", "2024-01-01", "2024-01-31")
    hash3 = _get_filter_hash("ALL", "BTC", "2024-01-01", "2024-01-31")

    assert hash1 != hash2
    assert hash1 == hash3
