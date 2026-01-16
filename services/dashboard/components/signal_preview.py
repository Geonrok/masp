"""Signal preview component for MASP Dashboard."""
from __future__ import annotations

import os
from datetime import datetime

import streamlit as st

from services.dashboard.utils.signal_generator import (
    get_cached_signals,
    get_cached_symbols,
    get_signal_generator_status,
)


def _check_live_conditions(exchange: str) -> tuple[bool, str]:
    """
    Check if LIVE mode conditions are met.

    Returns:
        (can_live, reason)
    """
    live_switch = os.getenv("MASP_DASHBOARD_LIVE", "").strip() == "1"
    if not live_switch:
        return False, "MASP_DASHBOARD_LIVE not set to '1'"

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
        return False, f"API keys not configured for {exchange}"

    return True, "All conditions met"


def render_signal_preview_panel() -> None:
    """Render signal preview panel with LIVE/DEMO mode support."""
    st.subheader("Signal Preview")

    exchange = st.selectbox(
        "Exchange",
        options=["upbit", "bithumb"],
        key="signal_exchange_select",
    )

    can_live, reason = _check_live_conditions(exchange)
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

    n_symbols = st.slider(
        "Number of symbols",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        key="signal_n_symbols",
    )

    if st.button("Generate Signals", key="signal_generate_btn", type="primary"):
        with st.spinner("Generating signals..."):
            try:
                symbols = get_cached_symbols(exchange, limit=n_symbols, allow_live=can_live)
                if not symbols:
                    st.warning("No symbols available")
                    return

                signals = get_cached_signals(exchange, tuple(symbols), allow_live=can_live)
                st.session_state["signal_results"] = signals
                st.session_state["signal_timestamp"] = datetime.now().isoformat()
            except Exception:
                st.error("Failed to generate signals. Please check configuration.")
                st.session_state["signal_results"] = []

    if "signal_results" in st.session_state and st.session_state["signal_results"]:
        signals = st.session_state["signal_results"]

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
