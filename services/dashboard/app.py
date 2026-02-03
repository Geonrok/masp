"""MASP dashboard Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.dashboard.components.alert_history import render_alert_history_panel
from services.dashboard.components.alert_settings import render_alert_settings

# Existing components
from services.dashboard.components.api_key_status import render_api_key_status_panel
from services.dashboard.components.backtest_viewer import render_backtest_viewer
from services.dashboard.components.exchange_control import render_exchange_controls
from services.dashboard.components.exchange_status import ExchangeStatusPanel
from services.dashboard.components.log_viewer import render_log_viewer

# Phase 7E: Market Regime & Signal components
from services.dashboard.components.market_regime_panel import (
    render_market_regime_panel,
    render_signal_status_panel,
)
from services.dashboard.components.multi_exchange_view import render_multi_exchange_view
from services.dashboard.components.order_panel import render_order_panel
from services.dashboard.components.portfolio_summary import render_portfolio_summary
from services.dashboard.components.positions_panel import render_positions_panel
from services.dashboard.components.risk_metrics import render_risk_metrics_panel
from services.dashboard.components.scheduler_status import render_scheduler_status
from services.dashboard.components.signal_preview import render_signal_preview_panel
from services.dashboard.components.strategy_config import StrategyConfigPanel
from services.dashboard.components.strategy_indicators import render_strategy_indicators
from services.dashboard.components.strategy_performance import (
    render_strategy_performance,
)

# Phase 7C-4 components
from services.dashboard.components.system_status import render_system_status
from services.dashboard.components.telegram_settings import render_telegram_settings
from services.dashboard.components.trade_history import render_trade_history_panel
from services.dashboard.providers.alert_manager_provider import (
    get_alert_rules,
    get_alert_settings_callbacks,
    get_anomaly_thresholds,
)
from services.dashboard.providers.alert_provider import get_alert_store
from services.dashboard.providers.backtest_provider import get_backtest_data
from services.dashboard.providers.log_provider import get_log_provider
from services.dashboard.providers.market_regime_provider import (
    get_daily_signal_summary,
    get_market_regime_analysis,
    get_trading_recommendation,
)
from services.dashboard.providers.multi_exchange_provider import (
    find_arbitrage_opportunities,
    get_exchange_list,
    get_price_comparison,
    perform_health_check,
)
from services.dashboard.providers.order_provider import (
    get_balance_provider,
    get_execution_adapter,
    get_price_provider,
)

# Phase 7C-6: Data providers
from services.dashboard.providers.portfolio_provider import get_portfolio_summary
from services.dashboard.providers.positions_provider import get_positions_data
from services.dashboard.providers.risk_metrics_provider import get_risk_metrics_data
from services.dashboard.providers.scheduler_provider import get_scheduler_job_provider
from services.dashboard.providers.strategy_performance_provider import (
    get_strategy_performance_provider,
)
from services.dashboard.providers.system_provider import (
    get_service_health,
    get_system_resources,
)
from services.dashboard.providers.trade_history_provider import get_trade_history_client
from services.dashboard.utils.api_client import ConfigApiClient
from services.dashboard.utils.auth_middleware import enforce_auth, logout

load_dotenv()

st.set_page_config(page_title="MASP 대시보드", layout="wide")

if not enforce_auth():
    st.stop()

st.title("MASP 대시보드")
st.caption("멀티에셋 전략 플랫폼 - Phase 7E")

api = ConfigApiClient()

# Main navigation tabs
tabs = st.tabs(
    [
        "개요",
        "거래",
        "분석",
        "모니터링",
        "설정",
    ]
)

# =============================================================================
# Tab 1: Overview - System status and portfolio summary
# =============================================================================
with tabs[0]:
    col1, col2 = st.columns([1, 1])

    with col1:
        # Real system resources via psutil
        render_system_status(
            resource_provider=get_system_resources,
            service_provider=get_service_health,
        )

    with col2:
        # Real portfolio data (or demo if MASP_ENABLE_LIVE_TRADING!=1)
        portfolio = get_portfolio_summary()
        render_portfolio_summary(portfolio=portfolio)

    st.divider()

    # Exchange status (existing)
    st.subheader("거래소 상태")
    ExchangeStatusPanel(api).render()

# =============================================================================
# Tab 2: Trading - Order panel, positions, trade history, multi-exchange
# =============================================================================
with tabs[1]:
    trading_subtabs = st.tabs(["주문", "포지션", "거래 내역", "멀티 거래소"])

    with trading_subtabs[0]:
        # Real execution adapter (or demo if MASP_ENABLE_LIVE_TRADING!=1)
        render_order_panel(
            execution_adapter=get_execution_adapter(),
            price_provider=get_price_provider(),
            balance_provider=get_balance_provider(),
        )

    with trading_subtabs[1]:
        # Real positions data (or demo if MASP_ENABLE_LIVE_TRADING!=1)
        render_positions_panel(positions_data=get_positions_data())

    with trading_subtabs[2]:
        # Real trade history from TradeLogger (or demo if unavailable)
        render_trade_history_panel(api_client=get_trade_history_client())

    with trading_subtabs[3]:
        # Multi-exchange price comparison and arbitrage (Phase 7D-3)
        render_multi_exchange_view(
            exchanges=get_exchange_list(),
            comparison_provider=get_price_comparison,
            arbitrage_provider=find_arbitrage_opportunities,
            health_check_callback=perform_health_check,
        )

# =============================================================================
# Tab 3: Analytics - Strategy performance, backtest, risk metrics, market regime
# =============================================================================
with tabs[2]:
    analytics_subtabs = st.tabs(
        ["시장 국면", "전략 성과", "백테스트", "리스크 지표", "시그널"]
    )

    with analytics_subtabs[0]:
        # Market Regime Analysis (Phase 7E)
        col1, col2 = st.columns([1, 1])
        with col1:
            render_market_regime_panel(analysis_provider=get_market_regime_analysis)
        with col2:
            render_signal_status_panel(signal_provider=get_daily_signal_summary)

        # Trading Recommendation
        st.divider()
        recommendation = get_trading_recommendation()
        rec_color = {
            "AGGRESSIVE_LONG": "#4CAF50",
            "LONG": "#8BC34A",
            "CAUTIOUS_LONG": "#CDDC39",
            "SELECTIVE": "#FFEB3B",
            "CAUTIOUS": "#FF9800",
            "WAIT": "#9E9E9E",
            "STAY_OUT": "#F44336",
        }.get(recommendation.get("action", "WAIT"), "#9E9E9E")

        st.markdown(
            f"""
            <div style="background-color: {rec_color}; padding: 15px; border-radius: 8px;">
                <h3 style="color: white; margin: 0;">매매 권고: {recommendation.get('action', 'N/A')}</h3>
                <p style="color: white; margin: 5px 0;">포지션 사이즈: {recommendation.get('position_size', 0) * 100:.0f}%</p>
                <p style="color: white; margin: 0; font-size: 14px;">{recommendation.get('message', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with analytics_subtabs[1]:
        # Real strategy performance from trade history (or demo if unavailable)
        render_strategy_performance(
            performance_provider=get_strategy_performance_provider()
        )

    with analytics_subtabs[2]:
        # Real backtest data from BacktestStore (or demo if unavailable)
        render_backtest_viewer(backtest_data=get_backtest_data())

    with analytics_subtabs[3]:
        # Real risk metrics from trade history (or demo if unavailable)
        risk_data = get_risk_metrics_data()
        if risk_data:
            returns, equity_curve, dates = risk_data
            render_risk_metrics_panel(
                returns=returns, equity_curve=equity_curve, dates=dates
            )
        else:
            render_risk_metrics_panel()

    with analytics_subtabs[4]:
        col1, col2 = st.columns([1, 1])
        with col1:
            render_strategy_indicators()
        with col2:
            render_signal_preview_panel()

# =============================================================================
# Tab 4: Monitoring - Logs, alerts, scheduler
# =============================================================================
with tabs[3]:
    monitoring_subtabs = st.tabs(["로그", "알림", "스케줄러"])

    with monitoring_subtabs[0]:
        # Real logs from log files (or demo if unavailable)
        render_log_viewer(log_provider=get_log_provider())

    with monitoring_subtabs[1]:
        # Real alert history from logs (or demo if unavailable)
        render_alert_history_panel(alert_store=get_alert_store())

    with monitoring_subtabs[2]:
        # Real scheduler status from APScheduler (or static jobs if unavailable)
        render_scheduler_status(job_provider=get_scheduler_job_provider())

# =============================================================================
# Tab 5: Settings - Config, API keys, Telegram, Alerts, Exchange controls
# =============================================================================
with tabs[4]:
    settings_subtabs = st.tabs(
        ["전략 설정", "API 키", "텔레그램", "알림 설정", "거래소 제어"]
    )

    with settings_subtabs[0]:
        StrategyConfigPanel(api).render()

    with settings_subtabs[1]:
        render_api_key_status_panel()

    with settings_subtabs[2]:
        render_telegram_settings()

    with settings_subtabs[3]:
        # Alert rules and anomaly threshold settings (Phase 7D-2)
        callbacks = get_alert_settings_callbacks()
        render_alert_settings(
            rules=get_alert_rules(),
            thresholds=get_anomaly_thresholds(),
            on_rule_toggle=callbacks.get("on_rule_toggle"),
            on_rule_delete=callbacks.get("on_rule_delete"),
            on_rule_create=callbacks.get("on_rule_create"),
            on_thresholds_save=callbacks.get("on_thresholds_save"),
        )

    with settings_subtabs[4]:
        render_exchange_controls(["upbit", "bithumb"])

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("MASP")
    st.caption("멀티에셋 전략 플랫폼")

    st.divider()

    st.subheader("정보")
    st.text("버전: 5.2.0")
    st.text("Phase: 7D")

    st.divider()

    st.subheader("작업")
    if st.button("새로고침", use_container_width=True):
        st.rerun()

    if st.button("로그아웃", use_container_width=True, type="secondary"):
        logout()
        st.rerun()
