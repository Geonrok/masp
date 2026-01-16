"""MASP dashboard Streamlit app."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.dashboard.components.api_key_status import (
    ApiKeyStatusPanel,
    render_api_key_status_panel,
)
from services.dashboard.components.exchange_control import render_exchange_controls
from services.dashboard.components.exchange_status import ExchangeStatusPanel
from services.dashboard.components.signal_preview import render_signal_preview
from services.dashboard.components.strategy_config import StrategyConfigPanel
from services.dashboard.components.strategy_indicators import render_strategy_indicators
from services.dashboard.components.telegram_settings import render_telegram_settings
from services.dashboard.utils.api_client import ConfigApiClient

load_dotenv()

st.set_page_config(page_title="MASP Dashboard", layout="wide")

if not os.getenv("MASP_ADMIN_TOKEN"):
    st.warning("MASP_ADMIN_TOKEN is not set. API calls may fail.")

st.title("MASP Dashboard")
st.caption("Multi-Asset Strategy Platform - Phase 5G")

api = ConfigApiClient()

tabs = st.tabs(
    [
        "Status",
        "Quick Controls",
        "Telegram",
        "API Keys",
        "Strategy",
        "Signals",
        "Config",
    ]
)

with tabs[0]:
    ExchangeStatusPanel(api).render()

with tabs[1]:
    st.header("Quick Controls")
    render_exchange_controls(["upbit", "bithumb"])

with tabs[2]:
    st.header("Telegram Settings")
    render_telegram_settings()

with tabs[3]:
    st.header("API Key Management")
    try:
        render_api_key_status_panel()
    except Exception:
        ApiKeyStatusPanel(api).render()

with tabs[4]:
    st.header("Strategy Indicators")
    render_strategy_indicators()

with tabs[5]:
    st.header("Signal Preview")
    render_signal_preview()

with tabs[6]:
    StrategyConfigPanel(api).render()

with st.sidebar:
    st.header("System Info")
    st.text("Version: 5.0.0")
    st.text("Phase: 5G")
    if st.button("Refresh"):
        st.rerun()
