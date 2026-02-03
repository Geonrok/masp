"""PnL visualization component using Plotly."""

from __future__ import annotations

from typing import Dict, List

import plotly.graph_objects as go
import streamlit as st

from services.dashboard.utils.pnl_calculator import PositionPnL


def render_pnl_chart(pnl_list: List[PositionPnL]) -> None:
    """Render PnL bar chart for positions."""
    if not pnl_list:
        st.info("No positions to display.")
        return

    symbols = [p.symbol for p in pnl_list]
    pnl_amounts = [p.pnl_amount for p in pnl_list]
    pnl_percents = [p.pnl_percent for p in pnl_list]

    colors = ["#00C853" if pnl >= 0 else "#FF5252" for pnl in pnl_amounts]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=symbols,
            y=pnl_amounts,
            marker_color=colors,
            text=[f"{pct:+.2f}%" for pct in pnl_percents],
            textposition="auto",
            name="PnL (KRW)",
        )
    )

    fig.update_layout(
        title="Position PnL",
        xaxis_title="Symbol",
        yaxis_title="PnL (KRW)",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_pnl_summary(total_pnl: Dict[str, float]) -> None:
    """Render PnL summary metrics with safe defaults."""
    total_cost = float(total_pnl.get("total_cost", 0.0))
    total_value = float(total_pnl.get("total_value", 0.0))
    pnl = float(total_pnl.get("total_pnl", 0.0))
    pnl_pct = float(total_pnl.get("total_pnl_percent", 0.0))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cost", f"₩{total_cost:,.0f}")
    with col2:
        st.metric("Current Value", f"₩{total_value:,.0f}")
    with col3:
        st.metric("Total PnL", f"₩{pnl:,.0f}", delta=f"{pnl_pct:+.2f}%")
    with col4:
        st.metric("Status", "Profit" if pnl >= 0 else "Loss")
