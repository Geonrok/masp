"""Positions panel for MASP Dashboard - displays account balances."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import streamlit as st


def _check_live_conditions(exchange: str) -> Tuple[bool, str]:
    """Check LIVE mode conditions."""
    live_switch = os.getenv("MASP_DASHBOARD_LIVE", "").strip() == "1"
    if not live_switch:
        return False, "MASP_DASHBOARD_LIVE not set"

    has_keys = False
    try:
        from libs.core.key_manager import KeyManager

        km = KeyManager()
        raw = km.get_raw_key(exchange)
        has_keys = bool(raw and raw.get("api_key") and raw.get("secret_key"))
    except Exception:
        pass

    if not has_keys:
        api_key = os.getenv(f"{exchange.upper()}_API_KEY")
        secret_key = os.getenv(f"{exchange.upper()}_SECRET_KEY")
        has_keys = bool(api_key and secret_key)

    if not has_keys:
        return False, "API keys not configured"

    return True, "Ready"


def _get_mock_balances() -> List[Dict[str, Any]]:
    """Generate mock balance data for demo mode."""
    return [
        {"currency": "KRW", "balance": 1_000_000.0, "locked": 0.0, "avg_buy_price": 1.0},
        {"currency": "BTC", "balance": 0.05, "locked": 0.0, "avg_buy_price": 95_000_000.0},
        {"currency": "ETH", "balance": 1.5, "locked": 0.0, "avg_buy_price": 4_500_000.0},
    ]


def _map_execution_exchange(exchange: str) -> str:
    """Map dashboard exchange name to execution adapter exchange name."""
    mapping = {
        "upbit": "upbit_spot",
        "bithumb": "bithumb",
    }
    return mapping.get(exchange, exchange)


def _get_real_balances(exchange: str) -> Optional[List[Dict[str, Any]]]:
    """Get real balances from execution adapter."""
    try:
        from libs.adapters.factory import create_execution_adapter

        adapter = create_execution_adapter(_map_execution_exchange(exchange), live_mode=True)
        if adapter:
            return adapter.get_balances()
    except Exception:
        pass
    return None


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_balances(exchange: str, use_real: bool) -> Tuple[List[Dict[str, Any]], bool]:
    """Fetch balances with caching."""
    if use_real:
        balances = _get_real_balances(exchange)
        if balances:
            return balances, False

    return _get_mock_balances(), True


def render_positions_panel() -> None:
    """Render positions/balances panel."""
    st.subheader("Account Positions")

    exchange = st.selectbox(
        "Exchange",
        options=["upbit", "bithumb"],
        key="positions_exchange_select",
    )

    can_live, reason = _check_live_conditions(exchange)

    if can_live:
        st.success(f"LIVE Mode - Connected to {exchange.upper()}")
    else:
        st.info(f"DEMO Mode - {reason}")

    st.divider()

    if st.button("Refresh Balances", key="positions_refresh_btn", type="primary"):
        with st.spinner("Fetching balances..."):
            balances, is_mock = _fetch_balances(exchange, use_real=can_live)
            st.session_state["positions_balances"] = balances
            st.session_state["positions_is_mock"] = is_mock
            st.session_state["positions_timestamp"] = datetime.now().isoformat()

    if "positions_balances" in st.session_state:
        balances = st.session_state["positions_balances"]
        is_mock = st.session_state.get("positions_is_mock", True)
        timestamp = st.session_state.get("positions_timestamp", "")

        if is_mock:
            st.caption("Showing mock data (DEMO mode)")

        st.caption(f"Last updated: {timestamp[:19] if timestamp else 'N/A'}")

        if balances:
            krw_balance = next((b for b in balances if b["currency"] == "KRW"), None)
            if krw_balance:
                st.metric(
                    "KRW Balance",
                    f"KRW {krw_balance['balance']:,.0f}",
                    delta=f"Locked: KRW {krw_balance.get('locked', 0):,.0f}"
                    if krw_balance.get("locked")
                    else None,
                )

            coin_balances = [
                b for b in balances if b["currency"] != "KRW" and b["balance"] > 0
            ]

            if coin_balances:
                st.dataframe(
                    [
                        {
                            "Currency": b["currency"],
                            "Balance": f"{b['balance']:.8f}",
                            "Locked": f"{b.get('locked', 0):.8f}",
                            "Avg Buy Price": f"KRW {b.get('avg_buy_price', 0):,.0f}",
                        }
                        for b in coin_balances
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No coin holdings")
        else:
            st.warning("No balance data available")
    else:
        st.info("Click 'Refresh Balances' to load data")
