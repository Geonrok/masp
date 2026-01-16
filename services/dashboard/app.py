"""MASP dashboard Streamlit app."""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from components.api_key_status import ApiKeyStatusPanel
from components.exchange_control import render_exchange_controls
from components.exchange_status import ExchangeStatusPanel
from components.strategy_config import StrategyConfigPanel
from utils.api_client import ConfigApiClient

load_dotenv()

st.set_page_config(page_title="MASP Dashboard", page_icon="", layout="wide")

if not os.getenv("MASP_ADMIN_TOKEN"):
    st.warning("MASP_ADMIN_TOKEN is not set. API calls may fail.")

st.title("MASP Configuration Dashboard")
st.caption("Multi-Asset Strategy Platform - Phase 5C-D")

api = ConfigApiClient()

tab_status, tab_control, tab_keys, tab_config = st.tabs(
    ["Exchange Status", "Quick Controls", "API Keys", "Strategy Config"]
)

with tab_status:
    ExchangeStatusPanel(api).render()

with tab_control:
    st.header("Quick Controls")
    render_exchange_controls(["upbit", "bithumb"])

with tab_keys:
    ApiKeyStatusPanel(api).render()

with tab_config:
    StrategyConfigPanel(api).render()

with st.sidebar:
    st.header("System Info")
    st.text("Version: 5.0.0")
    st.text("Phase: 5C-D")
    if st.button("Refresh"):
        st.rerun()
