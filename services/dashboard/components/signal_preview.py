"""Strategy signal preview - mock data only."""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def get_mock_signals(exchange: str) -> List[Dict[str, Any]]:
    """Return mock signal data for UI demonstration."""
    if exchange == "upbit":
        return [
            {"symbol": "BTC/KRW", "signal": "BUY", "strength": 0.85, "price": 145000000},
            {"symbol": "ETH/KRW", "signal": "BUY", "strength": 0.72, "price": 4850000},
            {"symbol": "XRP/KRW", "signal": "HOLD", "strength": 0.45, "price": 3200},
            {"symbol": "SOL/KRW", "signal": "SELL", "strength": 0.65, "price": 285000},
        ]
    if exchange == "bithumb":
        return [
            {"symbol": "BTC/KRW", "signal": "BUY", "strength": 0.78, "price": 144800000},
            {"symbol": "ETH/KRW", "signal": "HOLD", "strength": 0.52, "price": 4820000},
        ]
    return []


def get_signal_icon(signal: str) -> str:
    """Return ASCII indicator for signal."""
    return {"BUY": "^", "SELL": "v", "HOLD": "-"}.get(signal, "?")


def render_signal_table(signals: List[Dict[str, Any]]) -> None:
    """Render signal table."""
    if not signals:
        st.info("No signals available.")
        return

    table_data = []
    for item in signals:
        table_data.append(
            {
                "Signal": f"{get_signal_icon(item['signal'])} {item['signal']}",
                "Symbol": item["symbol"],
                "Strength": f"{item['strength']:.0%}",
                "Price": f"{item['price']:,.0f} KRW",
            }
        )

    st.table(table_data)


def render_signal_summary(signals: List[Dict[str, Any]]) -> None:
    """Render signal summary."""
    buy = sum(1 for s in signals if s["signal"] == "BUY")
    sell = sum(1 for s in signals if s["signal"] == "SELL")
    hold = sum(1 for s in signals if s["signal"] == "HOLD")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(signals))
    col2.metric("BUY", buy)
    col3.metric("SELL", sell)
    col4.metric("HOLD", hold)


def render_signal_preview() -> None:
    """Render signal preview UI."""
    st.subheader("Strategy Signal Preview")
    st.warning(
        "DEMO MODE: This preview uses mock data for UI demonstration. "
        "Actual signals will be available after StrategyRunner integration."
    )

    exchanges = ["upbit", "bithumb"]
    selected = st.selectbox("Select Exchange", exchanges, key="signal_exchange")

    st.divider()

    if st.button("Generate Preview", key="gen_preview"):
        with st.spinner("Generating mock signals..."):
            signals = get_mock_signals(selected)

        if signals:
            render_signal_summary(signals)
            st.divider()
            st.subheader("Signal Details")
            render_signal_table(signals)

            filter_opts = st.multiselect(
                "Filter by Signal",
                options=["BUY", "SELL", "HOLD"],
                default=["BUY", "SELL", "HOLD"],
                key="signal_filter",
            )

            filtered = [s for s in signals if s["signal"] in filter_opts]
            if len(filtered) != len(signals):
                st.subheader("Filtered Results")
                render_signal_table(filtered)
        else:
            st.info("No signals for this exchange.")
    else:
        st.info("Click 'Generate Preview' to see mock signals.")
