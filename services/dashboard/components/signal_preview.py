"""Signal preview component for MASP Dashboard."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from services.dashboard.utils.live_mode import check_live_conditions
from services.dashboard.utils.signal_generator import (
    get_cached_signals,
    get_cached_symbols,
    get_signal_generator_status,
)
from services.dashboard.utils.holdings import (
    get_holding_symbols,
    is_private_api_enabled,
)
from services.dashboard.utils.symbols import upbit_to_dashboard
from services.dashboard.utils.upbit_symbols import (
    get_all_upbit_symbols,
    get_symbol_count,
)


def render_signal_preview_panel() -> None:
    """Render signal preview panel with LIVE/DEMO mode support."""
    st.subheader("Signal Preview")

    exchange = st.selectbox(
        "Exchange",
        options=["upbit", "bithumb"],
        key="signal_exchange_select",
    )

    can_live, reason = check_live_conditions(exchange)
    status = get_signal_generator_status(exchange, allow_live=can_live)

    col1, col2 = st.columns(2)
    with col1:
        if can_live and not status["is_demo_mode"]:
            st.success("LIVE Mode")
        else:
            st.info("DEMO Mode")

    with col2:
        st.caption(status["mode_description"])
        if not can_live:
            st.caption(f"Reason: {reason}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        show_all_list = st.checkbox(
            "Show all symbols (list only)",
            value=False,
            key="signal_show_all_list",
        )
    with col2:
        if is_private_api_enabled():
            filter_by_holdings = st.checkbox(
                "Filter by holdings",
                value=True,
                key="signal_filter_holdings",
            )
        else:
            filter_by_holdings = False
            st.caption("Holdings filter requires MASP_ENABLE_LIVE_TRADING=1")

    if show_all_list:
        if not can_live or exchange != "upbit":
            st.info("Full symbol list is available for LIVE Upbit mode only.")
        else:
            all_symbols = get_all_upbit_symbols()
            total_count = get_symbol_count()
            st.caption(f"Total available: {total_count} symbols")

            query = st.text_input(
                "Search symbols", value="", key="signal_symbol_search"
            )
            if query:
                filtered = [
                    symbol for symbol in all_symbols if query.upper() in symbol.upper()
                ]
            else:
                filtered = all_symbols

            st.dataframe(
                [{"Symbol": symbol} for symbol in filtered],
                use_container_width=True,
                hide_index=True,
            )

    max_symbols = 20 if can_live else 50
    n_symbols = st.slider(
        "Number of symbols",
        min_value=5,
        max_value=max_symbols,
        value=10,
        step=5,
        key="signal_n_symbols",
    )
    if can_live and max_symbols < 50:
        st.caption("LIVE mode limits to 20 symbols to prevent rate-limiting.")

    if st.button("Generate Signals", key="signal_generate_btn", type="primary"):
        with st.spinner("Generating signals..."):
            try:
                symbols = get_cached_symbols(
                    exchange, limit=n_symbols, allow_live=can_live, show_all=False
                )
                if not symbols:
                    st.warning("No symbols available")
                    return

                signals = get_cached_signals(
                    exchange, tuple(symbols), allow_live=can_live
                )
                st.session_state["signal_results"] = signals
                st.session_state["signal_timestamp"] = datetime.now().isoformat()
            except Exception:
                st.error("Failed to generate signals. Please check configuration.")
                st.session_state["signal_results"] = []

    if "signal_results" in st.session_state and st.session_state["signal_results"]:
        signals = st.session_state["signal_results"]

        holding_symbols = get_holding_symbols() if filter_by_holdings else []
        filtered_signals = []
        for signal in signals:
            symbol_raw = signal.get("symbol", "")
            symbol = (
                upbit_to_dashboard(symbol_raw) if exchange == "upbit" else symbol_raw
            )
            signal_type = signal.get("signal", "")
            is_holding = symbol in holding_symbols

            if filter_by_holdings:
                if signal_type == "BUY" and is_holding:
                    continue
                if signal_type in ("SELL", "HOLD") and not is_holding:
                    continue

            filtered_signals.append(signal)

        st.caption(f"Showing {len(filtered_signals)} of {len(signals)} signals")

        signals = filtered_signals

        buy_count = sum(1 for s in signals if s.get("signal") == "BUY")
        sell_count = sum(1 for s in signals if s.get("signal") == "SELL")
        hold_count = sum(1 for s in signals if s.get("signal") == "HOLD")

        col1, col2, col3 = st.columns(3)
        col1.metric("BUY", buy_count)
        col2.metric("SELL", sell_count)
        col3.metric("HOLD", hold_count)

        st.dataframe(
            [
                {
                    "Symbol": s["symbol"],
                    "Signal": s["signal"],
                    "Strength": f"{s.get('strength', 0):.2%}",
                    "Time": s.get("timestamp", "")[:19],
                }
                for s in signals
            ],
            use_container_width=True,
            hide_index=True,
        )

        if any(s.get("is_mock") for s in signals):
            st.caption("Some signals are generated from mock data (DEMO mode)")
