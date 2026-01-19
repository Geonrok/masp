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

# Existing components
from services.dashboard.components.api_key_status import render_api_key_status_panel
from services.dashboard.components.exchange_control import render_exchange_controls
from services.dashboard.components.exchange_status import ExchangeStatusPanel
from services.dashboard.components.positions_panel import render_positions_panel
from services.dashboard.components.signal_preview import render_signal_preview_panel
from services.dashboard.components.strategy_config import StrategyConfigPanel
from services.dashboard.components.strategy_indicators import render_strategy_indicators
from services.dashboard.components.telegram_settings import render_telegram_settings

# Phase 7C-4 components
from services.dashboard.components.system_status import render_system_status
from services.dashboard.components.portfolio_summary import render_portfolio_summary
from services.dashboard.components.order_panel import render_order_panel
from services.dashboard.components.trade_history import render_trade_history_panel
from services.dashboard.components.strategy_performance import render_strategy_performance
from services.dashboard.components.backtest_viewer import render_backtest_viewer
from services.dashboard.components.risk_metrics import render_risk_metrics_panel
from services.dashboard.components.log_viewer import render_log_viewer
from services.dashboard.components.alert_history import render_alert_history_panel
from services.dashboard.components.scheduler_status import render_scheduler_status

from services.dashboard.utils import auth
from services.dashboard.utils.auth_middleware import enforce_auth
from services.dashboard.utils.api_client import ConfigApiClient

load_dotenv()

st.set_page_config(page_title="MASP Dashboard", layout="wide")

if not os.getenv("MASP_ADMIN_TOKEN"):
    st.warning("MASP_ADMIN_TOKEN is not set. API calls may fail.")

if not enforce_auth():
    st.stop()

st.title("MASP Dashboard")
st.caption("Multi-Asset Strategy Platform - Phase 7C-5")

api = ConfigApiClient()

# Main navigation tabs
tabs = st.tabs(
    [
        "Overview",
        "Trading",
        "Analytics",
        "Monitoring",
        "Settings",
    ]
)

# =============================================================================
# Tab 1: Overview - System status and portfolio summary
# =============================================================================
with tabs[0]:
    col1, col2 = st.columns([1, 1])

    with col1:
        render_system_status()

    with col2:
        render_portfolio_summary()

    st.divider()

    # Exchange status (existing)
    st.subheader("Exchange Status")
    ExchangeStatusPanel(api).render()

# =============================================================================
# Tab 2: Trading - Order panel, positions, trade history
# =============================================================================
with tabs[1]:
    trading_subtabs = st.tabs(["Order Panel", "Positions", "Trade History"])

    with trading_subtabs[0]:
        render_order_panel()

    with trading_subtabs[1]:
        render_positions_panel()

    with trading_subtabs[2]:
        render_trade_history_panel()

# =============================================================================
# Tab 3: Analytics - Strategy performance, backtest, risk metrics
# =============================================================================
with tabs[2]:
    analytics_subtabs = st.tabs(
        ["Strategy Performance", "Backtest Viewer", "Risk Metrics", "Signals"]
    )

    with analytics_subtabs[0]:
        render_strategy_performance()

    with analytics_subtabs[1]:
        render_backtest_viewer()

    with analytics_subtabs[2]:
        render_risk_metrics_panel()

    with analytics_subtabs[3]:
        col1, col2 = st.columns([1, 1])
        with col1:
            render_strategy_indicators()
        with col2:
            render_signal_preview_panel()

# =============================================================================
# Tab 4: Monitoring - Logs, alerts, scheduler
# =============================================================================
with tabs[3]:
    monitoring_subtabs = st.tabs(["Logs", "Alerts", "Scheduler"])

    with monitoring_subtabs[0]:
        render_log_viewer()

    with monitoring_subtabs[1]:
        render_alert_history_panel()

    with monitoring_subtabs[2]:
        render_scheduler_status()

# =============================================================================
# Tab 5: Settings - Config, API keys, Telegram, Exchange controls
# =============================================================================
with tabs[4]:
    settings_subtabs = st.tabs(
        ["Strategy Config", "API Keys", "Telegram", "Exchange Controls"]
    )

    with settings_subtabs[0]:
        StrategyConfigPanel(api).render()

    with settings_subtabs[1]:
        render_api_key_status_panel()

    with settings_subtabs[2]:
        render_telegram_settings()

    with settings_subtabs[3]:
        render_exchange_controls(["upbit", "bithumb"])

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("MASP")
    st.caption("Multi-Asset Strategy Platform")

    st.divider()

    st.subheader("Quick Info")
    st.text("Version: 5.0.0")
    st.text("Phase: 7C-5")

    st.divider()

    st.subheader("Actions")
    if st.button("Refresh", use_container_width=True):
        st.rerun()

    if st.button("Logout", use_container_width=True, type="secondary"):
        auth.clear_token()
        st.rerun()
