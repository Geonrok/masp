"""Trade history panel component."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import streamlit as st


def _get_demo_trades() -> List[Dict[str, Any]]:
    """Generate demo trade data for development."""
    return [
        {
            "id": "T001",
            "timestamp": datetime.now() - timedelta(hours=2),
            "exchange": "upbit",
            "symbol": "BTC",
            "side": "BUY",
            "quantity": 0.1,
            "price": 54_000_000,
            "total": 5_400_000,
            "status": "FILLED",
        },
        {
            "id": "T002",
            "timestamp": datetime.now() - timedelta(hours=1),
            "exchange": "upbit",
            "symbol": "ETH",
            "side": "SELL",
            "quantity": 2.0,
            "price": 2_900_000,
            "total": 5_800_000,
            "status": "FILLED",
        },
        {
            "id": "T003",
            "timestamp": datetime.now() - timedelta(minutes=45),
            "exchange": "bithumb",
            "symbol": "XRP",
            "side": "BUY",
            "quantity": 1000.0,
            "price": 820,
            "total": 820_000,
            "status": "FILLED",
        },
        {
            "id": "T004",
            "timestamp": datetime.now() - timedelta(minutes=30),
            "exchange": "upbit",
            "symbol": "SOL",
            "side": "BUY",
            "quantity": 5.0,
            "price": 160_000,
            "total": 800_000,
            "status": "FILLED",
        },
        {
            "id": "T005",
            "timestamp": datetime.now() - timedelta(minutes=25),
            "exchange": "bithumb",
            "symbol": "ADA",
            "side": "SELL",
            "quantity": 500.0,
            "price": 650,
            "total": 325_000,
            "status": "FILLED",
        },
        {
            "id": "T006",
            "timestamp": datetime.now() - timedelta(minutes=20),
            "exchange": "upbit",
            "symbol": "AVAX",
            "side": "BUY",
            "quantity": 3.5,
            "price": 45_000,
            "total": 157_500,
            "status": "FILLED",
        },
        {
            "id": "T007",
            "timestamp": datetime.now() - timedelta(minutes=18),
            "exchange": "bithumb",
            "symbol": "DOT",
            "side": "BUY",
            "quantity": 12.0,
            "price": 3_200,
            "total": 38_400,
            "status": "FILLED",
        },
        {
            "id": "T008",
            "timestamp": datetime.now() - timedelta(minutes=12),
            "exchange": "upbit",
            "symbol": "LINK",
            "side": "SELL",
            "quantity": 2.0,
            "price": 20_000,
            "total": 40_000,
            "status": "FILLED",
        },
        {
            "id": "T009",
            "timestamp": datetime.now() - timedelta(minutes=8),
            "exchange": "bithumb",
            "symbol": "ATOM",
            "side": "BUY",
            "quantity": 4.0,
            "price": 8_000,
            "total": 32_000,
            "status": "FILLED",
        },
        {
            "id": "T010",
            "timestamp": datetime.now() - timedelta(minutes=5),
            "exchange": "upbit",
            "symbol": "DOGE",
            "side": "BUY",
            "quantity": 5000.0,
            "price": 120,
            "total": 600_000,
            "status": "FILLED",
        },
    ]


def _filter_trades(
    trades: List[Dict[str, Any]],
    exchange: Optional[str] = None,
    symbol: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Filter trades by criteria."""
    result = trades

    if exchange and exchange != "ALL":
        result = [t for t in result if t.get("exchange") == exchange.lower()]

    if symbol and symbol != "ALL":
        result = [t for t in result if t.get("symbol") == symbol]

    if start_date:
        result = [t for t in result if t.get("timestamp", datetime.min) >= start_date]

    if end_date:
        result = [t for t in result if t.get("timestamp", datetime.max) <= end_date]

    return result


def _paginate(items: List, page: int, per_page: int = 20) -> tuple[List, int, int]:
    """Paginate items. Returns (page_items, total_pages, clamped_page)."""
    total = len(items)
    total_pages = max(1, (total + per_page - 1) // per_page)

    clamped_page = max(1, min(page, total_pages))
    start = (clamped_page - 1) * per_page
    end = start + per_page
    return items[start:end], total_pages, clamped_page


def _get_filter_hash(exchange: str, symbol: str, start_date, end_date) -> str:
    """Generate hash for filter state to detect changes."""
    return f"{exchange}|{symbol}|{start_date}|{end_date}"


def render_trade_history_panel(api_client=None) -> None:
    """Render trade history table with filters."""
    st.subheader("Trade History")

    if api_client is not None:
        try:
            trades = api_client.get_trade_history()
        except Exception as exc:
            st.warning(f"API error: {exc}. Using demo data.")
            trades = _get_demo_trades()
    else:
        st.caption("Demo Data - Connect to live API for real trades")
        trades = _get_demo_trades()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        exchange_set = set(
            t.get("exchange", "").upper() for t in trades if t.get("exchange")
        )
        exchanges = ["ALL"] + sorted(exchange_set)
        exchange_filter = st.selectbox("Exchange", exchanges, key="th_exchange")

    with col2:
        symbol_set = set(t.get("symbol", "") for t in trades if t.get("symbol"))
        symbols = ["ALL"] + sorted(symbol_set)
        symbol_filter = st.selectbox("Symbol", symbols, key="th_symbol")

    with col3:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=7),
            key="th_start",
        )

    with col4:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            key="th_end",
        )

    current_filter_hash = _get_filter_hash(
        exchange_filter, symbol_filter, start_date, end_date
    )
    prev_filter_hash = st.session_state.get("th_filter_hash", "")

    if current_filter_hash != prev_filter_hash:
        st.session_state["th_page"] = 1
        st.session_state["th_filter_hash"] = current_filter_hash

    filtered = _filter_trades(
        trades,
        exchange=exchange_filter,
        symbol=symbol_filter,
        start_date=datetime.combine(start_date, datetime.min.time())
        if start_date
        else None,
        end_date=datetime.combine(end_date, datetime.max.time()) if end_date else None,
    )

    page = st.session_state.get("th_page", 1)
    page_items, total_pages, clamped_page = _paginate(filtered, page, per_page=20)

    if clamped_page != page:
        st.session_state["th_page"] = clamped_page
        page = clamped_page

    if page_items:
        data = [
            {
                "Time": t["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "Exchange": t.get("exchange", "").upper(),
                "Symbol": t.get("symbol", ""),
                "Side": t.get("side", ""),
                "Qty": f"{t.get('quantity', 0):,.4f}",
                "Price": f"₩{t.get('price', 0):,.0f}",
                "Total": f"₩{t.get('total', 0):,.0f}",
                "Status": t.get("status", ""),
            }
            for t in page_items
        ]
        st.dataframe(data, use_container_width=True, hide_index=True)

        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("◀ Prev", disabled=(page <= 1), key="th_prev"):
                st.session_state["th_page"] = page - 1
                st.rerun()
        with col_info:
            st.caption(f"Page {page} / {total_pages} ({len(filtered)} trades)")
        with col_next:
            if st.button("Next ▶", disabled=(page >= total_pages), key="th_next"):
                st.session_state["th_page"] = page + 1
                st.rerun()
    else:
        st.info("No trades found for the selected filters.")
