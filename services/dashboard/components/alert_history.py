"""Alert history panel component for Telegram notifications."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import streamlit as st

# Alert type constants
ALERT_TYPE_TRADE = "TRADE"
ALERT_TYPE_SIGNAL = "SIGNAL"
ALERT_TYPE_ERROR = "ERROR"
ALERT_TYPE_DAILY = "DAILY"
ALERT_TYPE_SYSTEM = "SYSTEM"

# Status constants
STATUS_SENT = "SENT"
STATUS_FAILED = "FAILED"

# Session state key prefix
_KEY_PREFIX = "alert_history."

# Generic type for pagination
T = TypeVar("T")


def _get_demo_alerts() -> List[Dict[str, Any]]:
    """Generate deterministic demo alert data."""
    base_time = datetime(2026, 1, 15, 10, 0, 0)
    return [
        {
            "id": "A001",
            "timestamp": base_time - timedelta(hours=5),
            "alert_type": ALERT_TYPE_TRADE,
            "exchange": "upbit",
            "message": "BUY BTC 0.1 @ 54,000,000 KRW",
            "status": STATUS_SENT,
        },
        {
            "id": "A002",
            "timestamp": base_time - timedelta(hours=4, minutes=30),
            "alert_type": ALERT_TYPE_SIGNAL,
            "exchange": "upbit",
            "message": "KAMA crossover detected for ETH",
            "status": STATUS_SENT,
        },
        {
            "id": "A003",
            "timestamp": base_time - timedelta(hours=4),
            "alert_type": ALERT_TYPE_TRADE,
            "exchange": "bithumb",
            "message": "SELL ETH 2.0 @ 2,900,000 KRW",
            "status": STATUS_SENT,
        },
        {
            "id": "A004",
            "timestamp": base_time - timedelta(hours=3, minutes=45),
            "alert_type": ALERT_TYPE_ERROR,
            "exchange": "bithumb",
            "message": "API rate limit exceeded",
            "status": STATUS_SENT,
        },
        {
            "id": "A005",
            "timestamp": base_time - timedelta(hours=3),
            "alert_type": ALERT_TYPE_TRADE,
            "exchange": "upbit",
            "message": "BUY XRP 1000 @ 820 KRW",
            "status": STATUS_FAILED,
        },
        {
            "id": "A006",
            "timestamp": base_time - timedelta(hours=2, minutes=30),
            "alert_type": ALERT_TYPE_SIGNAL,
            "exchange": "bithumb",
            "message": "Momentum signal triggered for SOL",
            "status": STATUS_SENT,
        },
        {
            "id": "A007",
            "timestamp": base_time - timedelta(hours=2),
            "alert_type": ALERT_TYPE_DAILY,
            "exchange": "upbit",
            "message": "Daily summary: 5 trades, +120,000 KRW",
            "status": STATUS_SENT,
        },
        {
            "id": "A008",
            "timestamp": base_time - timedelta(hours=1, minutes=30),
            "alert_type": ALERT_TYPE_SYSTEM,
            "exchange": "",
            "message": "Strategy engine restarted",
            "status": STATUS_SENT,
        },
        {
            "id": "A009",
            "timestamp": base_time - timedelta(hours=1),
            "alert_type": ALERT_TYPE_TRADE,
            "exchange": "upbit",
            "message": "BUY SOL 5.0 @ 160,000 KRW",
            "status": STATUS_SENT,
        },
        {
            "id": "A010",
            "timestamp": base_time - timedelta(minutes=30),
            "alert_type": ALERT_TYPE_ERROR,
            "exchange": "upbit",
            "message": "Connection timeout to exchange",
            "status": STATUS_FAILED,
        },
    ]


def _normalize_timestamp(value: Any) -> datetime:
    """Normalize timestamp value to datetime, fallback to datetime.min."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return datetime.min


def _get_default_start_date(alerts: List[Dict[str, Any]]) -> date:
    """Calculate default start date based on alert data (max timestamp - 7 days)."""
    if not alerts:
        return datetime.now().date() - timedelta(days=7)

    timestamps = [
        _normalize_timestamp(a.get("timestamp"))
        for a in alerts
        if a.get("timestamp") is not None
    ]

    if not timestamps:
        return datetime.now().date() - timedelta(days=7)

    max_ts = max(timestamps)
    return (max_ts - timedelta(days=7)).date()


def _filter_alerts(
    alerts: List[Dict[str, Any]],
    alert_type: Optional[str] = None,
    exchange: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Filter alerts by criteria."""
    result = alerts

    if alert_type and alert_type != "ALL":
        result = [a for a in result if a.get("alert_type") == alert_type]

    if exchange and exchange != "ALL":
        result = [a for a in result if a.get("exchange") == exchange.lower()]

    if status and status != "ALL":
        result = [a for a in result if a.get("status") == status]

    if start_date:
        result = [
            a for a in result if _normalize_timestamp(a.get("timestamp")) >= start_date
        ]

    if end_date:
        result = [
            a for a in result if _normalize_timestamp(a.get("timestamp")) <= end_date
        ]

    return result


def _paginate(
    items: List[T], page: int, per_page: int = 20
) -> Tuple[List[T], int, int]:
    """Paginate items. Returns (page_items, total_pages, clamped_page)."""
    total = len(items)
    total_pages = max(1, (total + per_page - 1) // per_page)

    clamped_page = max(1, min(page, total_pages))
    start = (clamped_page - 1) * per_page
    end = start + per_page
    return items[start:end], total_pages, clamped_page


def _normalize_date_for_hash(value: Any) -> str:
    """Normalize date/datetime to ISO format string for hash stability."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return "None"
    return str(value)


def _get_filter_hash(
    alert_type: str, exchange: str, status: str, start_date: Any, end_date: Any
) -> str:
    """Generate hash for filter state to detect changes."""
    start_str = _normalize_date_for_hash(start_date)
    end_str = _normalize_date_for_hash(end_date)
    return f"{alert_type}|{exchange}|{status}|{start_str}|{end_str}"


def _get_status_icon(status: str) -> str:
    """Return status indicator icon."""
    if status == STATUS_SENT:
        return "SENT"
    elif status == STATUS_FAILED:
        return "FAIL"
    return status


def _get_type_label(alert_type: str) -> str:
    """Return display label for alert type."""
    labels = {
        ALERT_TYPE_TRADE: "Trade",
        ALERT_TYPE_SIGNAL: "Signal",
        ALERT_TYPE_ERROR: "Error",
        ALERT_TYPE_DAILY: "Daily",
        ALERT_TYPE_SYSTEM: "System",
    }
    return labels.get(alert_type, alert_type)


def get_alert_type_options() -> List[str]:
    """Return available alert type options for filter."""
    return [
        "ALL",
        ALERT_TYPE_TRADE,
        ALERT_TYPE_SIGNAL,
        ALERT_TYPE_ERROR,
        ALERT_TYPE_DAILY,
        ALERT_TYPE_SYSTEM,
    ]


def get_status_options() -> List[str]:
    """Return available status options for filter."""
    return ["ALL", STATUS_SENT, STATUS_FAILED]


def _key(name: str) -> str:
    """Generate namespaced session state key."""
    return f"{_KEY_PREFIX}{name}"


def render_alert_history_panel(
    alert_store: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render alert history table with filters."""
    st.subheader("Alert History")

    if alert_store is not None:
        alerts = alert_store
    else:
        st.caption("Demo Data - Connect to live alert store for real history")
        alerts = _get_demo_alerts()

    if not alerts:
        st.info("No alerts to display.")
        return

    default_start = _get_default_start_date(alerts)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        type_options = get_alert_type_options()
        type_filter = st.selectbox("Type", type_options, key=_key("type"))

    with col2:
        exchange_set = set(
            a.get("exchange", "").upper() for a in alerts if a.get("exchange")
        )
        exchanges = ["ALL"] + sorted(exchange_set)
        exchange_filter = st.selectbox("Exchange", exchanges, key=_key("exchange"))

    with col3:
        status_options = get_status_options()
        status_filter = st.selectbox("Status", status_options, key=_key("status"))

    with col4:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            key=_key("start"),
        )

    current_filter_hash = _get_filter_hash(
        type_filter, exchange_filter, status_filter, start_date, None
    )
    prev_filter_hash = st.session_state.get(_key("filter_hash"), "")

    if current_filter_hash != prev_filter_hash:
        st.session_state[_key("page")] = 1
        st.session_state[_key("filter_hash")] = current_filter_hash

    filtered = _filter_alerts(
        alerts,
        alert_type=type_filter,
        exchange=exchange_filter,
        status=status_filter,
        start_date=(
            datetime.combine(start_date, datetime.min.time()) if start_date else None
        ),
        end_date=None,
    )

    page = st.session_state.get(_key("page"), 1)
    page_items, total_pages, clamped_page = _paginate(filtered, page, per_page=20)

    if clamped_page != page:
        st.session_state[_key("page")] = clamped_page
        page = clamped_page

    if page_items:
        data = [
            {
                "Time": _normalize_timestamp(a.get("timestamp")).strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "Type": _get_type_label(a.get("alert_type", "")),
                "Exchange": (
                    a.get("exchange", "-").upper() if a.get("exchange") else "-"
                ),
                "Message": a.get("message", "")[:50],
                "Status": _get_status_icon(a.get("status", "")),
            }
            for a in page_items
        ]
        st.dataframe(data, width="stretch", hide_index=True)

        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("Prev", disabled=(page <= 1), key=_key("prev")):
                st.session_state[_key("page")] = page - 1
                st.rerun()
        with col_info:
            st.caption(f"Page {page} / {total_pages} ({len(filtered)} alerts)")
        with col_next:
            if st.button("Next", disabled=(page >= total_pages), key=_key("next")):
                st.session_state[_key("page")] = page + 1
                st.rerun()
    else:
        st.info("No alerts found for the selected filters.")
