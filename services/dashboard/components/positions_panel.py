"""Positions panel component with PnL visualization."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import streamlit as st

from services.dashboard.components.pnl_chart import render_pnl_chart, render_pnl_summary
from services.dashboard.utils.pnl_calculator import (
    calculate_portfolio_pnl,
    calculate_total_pnl,
)


def _get_demo_data() -> Tuple[List[Dict], Dict[str, float]]:
    """Get demo positions and prices."""
    positions = [
        {"symbol": "BTC", "quantity": 0.5, "avg_price": 50_000_000},
        {"symbol": "ETH", "quantity": 5.0, "avg_price": 3_000_000},
        {"symbol": "XRP", "quantity": 1000.0, "avg_price": 800},
        {"symbol": "SOL", "quantity": 10.0, "avg_price": 150_000},
    ]
    current_prices = {
        "BTC": 55_000_000,
        "ETH": 2_800_000,
        "XRP": 850,
        "SOL": 160_000,
    }
    return positions, current_prices


def render_positions_panel(
    positions_data: Optional[Tuple[List[Dict], Dict[str, float]]] = None,
) -> None:
    """Render positions table and PnL charts.

    Args:
        positions_data: Tuple of (positions_list, current_prices_dict)
                       If None, uses demo data.
    """
    st.subheader("Portfolio PnL Dashboard")

    is_demo = positions_data is None
    if is_demo:
        st.caption("Demo Data - Connect to live API for real positions")
        positions, current_prices = _get_demo_data()
    else:
        positions, current_prices = positions_data

    pnl_list = calculate_portfolio_pnl(positions, current_prices)
    total_pnl = calculate_total_pnl(pnl_list)

    render_pnl_summary(total_pnl)

    st.divider()

    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        render_pnl_chart(pnl_list)

    with col_table:
        st.subheader("Position Details")
        if pnl_list:
            data = [
                {
                    "Symbol": p.symbol,
                    "Qty": f"{p.quantity:,.4f}",
                    "Avg": f"₩{p.avg_price:,.0f}",
                    "Cur": f"₩{p.current_price:,.0f}",
                    "PnL": f"₩{p.pnl_amount:+,.0f}",
                    "RoI": f"{p.pnl_percent:+.2f}%",
                }
                for p in pnl_list
            ]
            st.dataframe(data, use_container_width=True, hide_index=True)
        else:
            st.info("No active positions.")
