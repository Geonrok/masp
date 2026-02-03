"""Tests for alert history component."""

from __future__ import annotations

from datetime import date, datetime, timedelta


def test_import_alert_history():
    """Test module imports correctly."""
    from services.dashboard.components import alert_history

    assert hasattr(alert_history, "render_alert_history_panel")
    assert hasattr(alert_history, "_filter_alerts")
    assert hasattr(alert_history, "_paginate")
    assert hasattr(alert_history, "_get_demo_alerts")
    assert hasattr(alert_history, "_normalize_timestamp")
    assert hasattr(alert_history, "_get_default_start_date")


def test_filter_alerts_by_type():
    """Test filtering alerts by alert_type."""
    from services.dashboard.components.alert_history import (
        ALERT_TYPE_TRADE,
        ALERT_TYPE_SIGNAL,
        _filter_alerts,
    )

    alerts = [
        {"alert_type": ALERT_TYPE_TRADE, "exchange": "upbit"},
        {"alert_type": ALERT_TYPE_SIGNAL, "exchange": "upbit"},
        {"alert_type": ALERT_TYPE_TRADE, "exchange": "bithumb"},
    ]

    result = _filter_alerts(alerts, alert_type=ALERT_TYPE_TRADE)
    assert len(result) == 2
    assert all(a["alert_type"] == ALERT_TYPE_TRADE for a in result)


def test_filter_alerts_by_exchange():
    """Test filtering alerts by exchange."""
    from services.dashboard.components.alert_history import _filter_alerts

    alerts = [
        {"alert_type": "TRADE", "exchange": "upbit"},
        {"alert_type": "TRADE", "exchange": "bithumb"},
        {"alert_type": "TRADE", "exchange": "upbit"},
    ]

    result = _filter_alerts(alerts, exchange="UPBIT")
    assert len(result) == 2
    assert all(a["exchange"] == "upbit" for a in result)


def test_filter_alerts_by_status():
    """Test filtering alerts by status."""
    from services.dashboard.components.alert_history import (
        STATUS_SENT,
        STATUS_FAILED,
        _filter_alerts,
    )

    alerts = [
        {"status": STATUS_SENT},
        {"status": STATUS_FAILED},
        {"status": STATUS_SENT},
    ]

    result = _filter_alerts(alerts, status=STATUS_SENT)
    assert len(result) == 2
    assert all(a["status"] == STATUS_SENT for a in result)


def test_filter_alerts_by_date():
    """Test filtering alerts by date range."""
    from services.dashboard.components.alert_history import _filter_alerts

    now = datetime.now()
    alerts = [
        {"timestamp": now - timedelta(days=5)},
        {"timestamp": now - timedelta(days=2)},
        {"timestamp": now},
    ]

    result = _filter_alerts(
        alerts,
        start_date=now - timedelta(days=3),
        end_date=now,
    )
    assert len(result) == 2


def test_filter_alerts_all_filter():
    """Test that ALL filter returns all items."""
    from services.dashboard.components.alert_history import _filter_alerts

    alerts = [{"alert_type": "TRADE", "exchange": "upbit", "status": "SENT"}]
    result = _filter_alerts(alerts, alert_type="ALL", exchange="ALL", status="ALL")
    assert len(result) == 1


def test_paginate_basic():
    """Test basic pagination."""
    from services.dashboard.components.alert_history import _paginate

    items = list(range(50))
    page_items, total_pages, clamped_page = _paginate(items, page=1, per_page=20)

    assert len(page_items) == 20
    assert page_items[0] == 0
    assert total_pages == 3
    assert clamped_page == 1


def test_paginate_last_page():
    """Test pagination on last page."""
    from services.dashboard.components.alert_history import _paginate

    items = list(range(50))
    page_items, total_pages, clamped_page = _paginate(items, page=3, per_page=20)

    assert len(page_items) == 10
    assert page_items[0] == 40
    assert clamped_page == 3


def test_paginate_empty():
    """Test pagination with empty list."""
    from services.dashboard.components.alert_history import _paginate

    page_items, total_pages, clamped_page = _paginate([], page=1, per_page=20)

    assert len(page_items) == 0
    assert total_pages == 1
    assert clamped_page == 1


def test_paginate_clamp_high_page():
    """Test pagination clamps page when exceeding total."""
    from services.dashboard.components.alert_history import _paginate

    items = list(range(10))
    page_items, total_pages, clamped_page = _paginate(items, page=5, per_page=20)

    assert clamped_page == 1
    assert len(page_items) == 10


def test_paginate_clamp_negative_page():
    """Test pagination clamps negative page to 1."""
    from services.dashboard.components.alert_history import _paginate

    items = list(range(50))
    page_items, total_pages, clamped_page = _paginate(items, page=-1, per_page=20)

    assert clamped_page == 1
    assert page_items[0] == 0


def test_get_demo_alerts():
    """Test demo data structure."""
    from services.dashboard.components.alert_history import _get_demo_alerts

    alerts = _get_demo_alerts()
    assert len(alerts) >= 2
    assert "timestamp" in alerts[0]
    assert "alert_type" in alerts[0]
    assert "message" in alerts[0]
    assert "status" in alerts[0]


def test_get_demo_alerts_deterministic():
    """Test demo data is deterministic (no random values)."""
    from services.dashboard.components.alert_history import _get_demo_alerts

    alerts1 = _get_demo_alerts()
    alerts2 = _get_demo_alerts()

    assert alerts1[0]["id"] == alerts2[0]["id"]
    assert alerts1[0]["message"] == alerts2[0]["message"]
    assert alerts1[0]["timestamp"] == alerts2[0]["timestamp"]


def test_get_filter_hash():
    """Test filter hash generation for change detection."""
    from services.dashboard.components.alert_history import _get_filter_hash

    hash1 = _get_filter_hash("TRADE", "UPBIT", "SENT", "2024-01-01", "2024-01-31")
    hash2 = _get_filter_hash("SIGNAL", "UPBIT", "SENT", "2024-01-01", "2024-01-31")
    hash3 = _get_filter_hash("TRADE", "UPBIT", "SENT", "2024-01-01", "2024-01-31")

    assert hash1 != hash2
    assert hash1 == hash3


def test_get_status_icon():
    """Test status icon mapping."""
    from services.dashboard.components.alert_history import (
        STATUS_SENT,
        STATUS_FAILED,
        _get_status_icon,
    )

    assert _get_status_icon(STATUS_SENT) == "SENT"
    assert _get_status_icon(STATUS_FAILED) == "FAIL"
    assert _get_status_icon("UNKNOWN") == "UNKNOWN"


def test_get_type_label():
    """Test alert type label mapping."""
    from services.dashboard.components.alert_history import (
        ALERT_TYPE_TRADE,
        ALERT_TYPE_SIGNAL,
        ALERT_TYPE_ERROR,
        _get_type_label,
    )

    assert _get_type_label(ALERT_TYPE_TRADE) == "Trade"
    assert _get_type_label(ALERT_TYPE_SIGNAL) == "Signal"
    assert _get_type_label(ALERT_TYPE_ERROR) == "Error"
    assert _get_type_label("UNKNOWN") == "UNKNOWN"


def test_get_alert_type_options():
    """Test alert type options include ALL and all types."""
    from services.dashboard.components.alert_history import (
        ALERT_TYPE_TRADE,
        ALERT_TYPE_SIGNAL,
        get_alert_type_options,
    )

    options = get_alert_type_options()
    assert "ALL" in options
    assert ALERT_TYPE_TRADE in options
    assert ALERT_TYPE_SIGNAL in options


def test_get_status_options():
    """Test status options include ALL and all statuses."""
    from services.dashboard.components.alert_history import (
        STATUS_SENT,
        STATUS_FAILED,
        get_status_options,
    )

    options = get_status_options()
    assert "ALL" in options
    assert STATUS_SENT in options
    assert STATUS_FAILED in options


def test_filter_alerts_combined():
    """Test filtering with multiple criteria."""
    from services.dashboard.components.alert_history import (
        ALERT_TYPE_TRADE,
        STATUS_SENT,
        _filter_alerts,
    )

    alerts = [
        {"alert_type": ALERT_TYPE_TRADE, "exchange": "upbit", "status": STATUS_SENT},
        {"alert_type": ALERT_TYPE_TRADE, "exchange": "upbit", "status": "FAILED"},
        {"alert_type": "SIGNAL", "exchange": "upbit", "status": STATUS_SENT},
        {"alert_type": ALERT_TYPE_TRADE, "exchange": "bithumb", "status": STATUS_SENT},
    ]

    result = _filter_alerts(
        alerts,
        alert_type=ALERT_TYPE_TRADE,
        exchange="UPBIT",
        status=STATUS_SENT,
    )
    assert len(result) == 1
    assert result[0]["exchange"] == "upbit"


def test_normalize_timestamp_with_datetime():
    """Test _normalize_timestamp with valid datetime."""
    from services.dashboard.components.alert_history import _normalize_timestamp

    dt = datetime(2026, 1, 15, 10, 30, 0)
    result = _normalize_timestamp(dt)
    assert result == dt


def test_normalize_timestamp_with_date():
    """Test _normalize_timestamp with date object."""
    from services.dashboard.components.alert_history import _normalize_timestamp

    d = date(2026, 1, 15)
    result = _normalize_timestamp(d)
    assert result == datetime(2026, 1, 15, 0, 0, 0)


def test_normalize_timestamp_with_none():
    """Test _normalize_timestamp with None returns datetime.min."""
    from services.dashboard.components.alert_history import _normalize_timestamp

    result = _normalize_timestamp(None)
    assert result == datetime.min


def test_normalize_timestamp_with_invalid_type():
    """Test _normalize_timestamp with invalid type returns datetime.min."""
    from services.dashboard.components.alert_history import _normalize_timestamp

    result = _normalize_timestamp("invalid")
    assert result == datetime.min

    result = _normalize_timestamp(12345)
    assert result == datetime.min


def test_get_default_start_date_with_alerts():
    """Test _get_default_start_date calculates from max timestamp."""
    from services.dashboard.components.alert_history import _get_default_start_date

    alerts = [
        {"timestamp": datetime(2026, 1, 10, 10, 0, 0)},
        {"timestamp": datetime(2026, 1, 15, 10, 0, 0)},
        {"timestamp": datetime(2026, 1, 12, 10, 0, 0)},
    ]

    result = _get_default_start_date(alerts)
    expected = date(2026, 1, 8)  # max (Jan 15) - 7 days = Jan 8
    assert result == expected


def test_get_default_start_date_with_empty_list():
    """Test _get_default_start_date with empty list uses current date."""
    from services.dashboard.components.alert_history import _get_default_start_date

    result = _get_default_start_date([])
    expected = datetime.now().date() - timedelta(days=7)
    assert result == expected


def test_get_default_start_date_with_no_valid_timestamps():
    """Test _get_default_start_date with no valid timestamps."""
    from services.dashboard.components.alert_history import _get_default_start_date

    alerts = [
        {"timestamp": None},
        {"timestamp": None},
    ]

    result = _get_default_start_date(alerts)
    expected = datetime.now().date() - timedelta(days=7)
    assert result == expected


def test_normalize_date_for_hash_with_datetime():
    """Test _normalize_date_for_hash with datetime."""
    from services.dashboard.components.alert_history import _normalize_date_for_hash

    dt = datetime(2026, 1, 15, 10, 30, 0)
    result = _normalize_date_for_hash(dt)
    assert result == "2026-01-15"


def test_normalize_date_for_hash_with_date():
    """Test _normalize_date_for_hash with date."""
    from services.dashboard.components.alert_history import _normalize_date_for_hash

    d = date(2026, 1, 15)
    result = _normalize_date_for_hash(d)
    assert result == "2026-01-15"


def test_normalize_date_for_hash_with_none():
    """Test _normalize_date_for_hash with None."""
    from services.dashboard.components.alert_history import _normalize_date_for_hash

    result = _normalize_date_for_hash(None)
    assert result == "None"


def test_normalize_date_for_hash_with_string():
    """Test _normalize_date_for_hash with string fallback."""
    from services.dashboard.components.alert_history import _normalize_date_for_hash

    result = _normalize_date_for_hash("2026-01-15")
    assert result == "2026-01-15"


def test_key_function():
    """Test _key generates namespaced keys."""
    from services.dashboard.components.alert_history import _key, _KEY_PREFIX

    result = _key("type")
    assert result == f"{_KEY_PREFIX}type"
    assert result == "alert_history.type"


def test_filter_alerts_with_none_timestamp():
    """Test filtering handles None timestamp gracefully."""
    from services.dashboard.components.alert_history import _filter_alerts

    now = datetime.now()
    alerts = [
        {"timestamp": now},
        {"timestamp": None},
        {"timestamp": now - timedelta(days=1)},
    ]

    result = _filter_alerts(alerts, start_date=now - timedelta(days=2))
    assert len(result) == 2


def test_filter_alerts_with_invalid_timestamp_type():
    """Test filtering handles invalid timestamp type gracefully."""
    from services.dashboard.components.alert_history import _filter_alerts

    now = datetime.now()
    alerts = [
        {"timestamp": now},
        {"timestamp": "invalid"},
        {"timestamp": 12345},
    ]

    result = _filter_alerts(alerts, start_date=now - timedelta(days=1))
    assert len(result) == 1
