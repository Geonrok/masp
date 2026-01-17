"""Positions panel for MASP Dashboard - displays account balances."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from services.dashboard.utils.live_mode import check_live_conditions
from services.dashboard.utils.holdings import (
    clear_holdings_cache,
    get_holdings_upbit,
    is_private_api_enabled,
)

def _render_demo_positions() -> None:
    """Render demo positions for DEMO mode."""
    demo_data = [
        {"Currency": "BTC", "Balance": 0.01, "Locked": 0, "Avg Price": 50_000_000},
        {"Currency": "ETH", "Balance": 0.5, "Locked": 0, "Avg Price": 3_000_000},
    ]
    st.dataframe(demo_data, use_container_width=True, hide_index=True)


def render_positions_panel(
    exchange: Optional[str] = None, is_live: Optional[bool] = None
) -> None:
    """Render positions/balances panel."""
    st.subheader("Positions")

    if exchange is None:
        exchange = st.selectbox(
            "Exchange",
            options=["upbit", "bithumb"],
            key="positions_exchange_select",
        )

    if is_live is None:
        is_live, reason = check_live_conditions(exchange)
    else:
        reason = "LIVE mode provided" if is_live else "DEMO mode provided"

    if not is_live:
        st.info(f"DEMO Mode - {reason}")
        _render_demo_positions()
        return

    if exchange != "upbit":
        st.warning("Only Upbit holdings are supported in READ-ONLY mode.")
        _render_demo_positions()
        return

    if not is_private_api_enabled():
        st.warning("Set MASP_ENABLE_LIVE_TRADING=1 to view real positions")
        _render_demo_positions()
        return

    if st.button("Refresh", key="positions_refresh_btn", type="primary"):
        clear_holdings_cache()
        st.rerun()

    holdings = get_holdings_upbit()

    if not holdings:
        st.warning("Unable to fetch positions. Check API configuration.")
        return

    data: List[Dict[str, Any]] = []
    for entry in holdings:
        try:
            balance = float(entry.get("balance", 0))
            locked = float(entry.get("locked", 0))
            avg_price = float(entry.get("avg_buy_price", 0))
        except (ValueError, TypeError):
            continue

        if balance > 0 or locked > 0:
            data.append(
                {
                    "Currency": entry.get("currency", ""),
                    "Balance": balance,
                    "Locked": locked,
                    "Avg Price": avg_price,
                }
            )

    if data:
        st.dataframe(data, use_container_width=True, hide_index=True)
    else:
        st.info("No positions found")
