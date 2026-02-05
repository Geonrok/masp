"""Binance Futures MR Strategy status component."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import streamlit as st

from services.dashboard.providers.futures_mr_provider import FuturesMRStatusData

logger = logging.getLogger(__name__)


def render_futures_mr_status(
    status_provider: Optional[Callable[[], FuturesMRStatusData]] = None,
) -> None:
    """Render the Binance Futures MR Strategy status panel.

    Args:
        status_provider: Callback returning FuturesMRStatusData.
                        If None or returns None, shows demo data.
    """
    st.subheader("Binance Futures MR Strategy")

    # Fetch data
    data: Optional[FuturesMRStatusData] = None
    if status_provider is not None:
        try:
            data = status_provider()
        except Exception as exc:
            logger.error("[FuturesMRStatus] Provider error: %s", exc)
            st.warning(f"데이터 로딩 오류: {exc}")

    if data is None:
        from services.dashboard.providers.futures_mr_provider import (
            _get_demo_status,
        )

        data = _get_demo_status()

    if not data.data_available:
        st.caption("Demo Data - 실제 거래 데이터가 없습니다")

    cfg = data.config
    acct = data.account
    stats = data.stats

    # --- Section 1: Strategy Overview (color box) ---
    status_color = "#4CAF50" if cfg.enabled else "#9E9E9E"
    mode_label = "Paper" if cfg.mode == "paper" else "LIVE"
    mode_bg = "#FF9800" if cfg.mode == "paper" else "#F44336"

    st.markdown(
        f"""
        <div style="background-color: {status_color}; padding: 15px;
                    border-radius: 8px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between;
                        align-items: center;">
                <h4 style="color: white; margin: 0;">
                    {cfg.strategy_name.upper().replace('_', ' ')}
                </h4>
                <span style="background: {mode_bg}; color: white;
                             padding: 3px 10px; border-radius: 4px;
                             font-weight: bold; font-size: 13px;">
                    {mode_label}
                </span>
            </div>
            <p style="color: rgba(255,255,255,0.9); margin: 6px 0 0 0;
                      font-size: 14px;">
                Schedule: {cfg.schedule_hour:02d}:{cfg.schedule_minute:02d}
                {cfg.schedule_timezone}
                (jitter: {cfg.schedule_jitter}s)
                &nbsp;|&nbsp;
                Max Pos: {cfg.max_positions}
                &nbsp;|&nbsp;
                Size: {cfg.position_size_usdt} USDT
                &nbsp;|&nbsp;
                Leverage: {cfg.leverage}x
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Section 2: Account Summary (4 metrics) ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Initial Balance", f"{acct.initial_balance:,.2f} USDT")
    with c2:
        delta_bal = acct.estimated_balance - acct.initial_balance
        st.metric(
            "Estimated Balance",
            f"{acct.estimated_balance:,.2f} USDT",
            delta=f"{delta_bal:+,.2f}",
        )
    with c3:
        st.metric(
            "Realized PnL",
            f"{acct.total_pnl:+,.2f} USDT",
            delta=f"{acct.pnl_percent:+.2f}%",
        )
    with c4:
        st.metric("Total Fees", f"{acct.total_fees:,.4f} USDT")

    # --- Section 3: Trade Statistics (4 metrics) ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Trades Today", stats.trades_today)
    with c2:
        st.metric("Total Trades", stats.trades_total)
    with c3:
        st.metric("BUY / SELL", f"{stats.buy_count} / {stats.sell_count}")
    with c4:
        st.metric("Unique Symbols", stats.unique_symbols)

    # --- Section 4: Active Positions ---
    pos_count = len(data.positions)
    with st.expander(f"Active Positions ({pos_count})", expanded=pos_count <= 10):
        if data.positions:
            pos_data = [
                {
                    "Symbol": p.symbol,
                    "Side": p.side,
                    "Quantity": f"{p.net_quantity:.6f}",
                    "Avg Entry": f"${p.avg_entry_price:,.2f}",
                    "Notional (USDT)": f"${p.notional_value:,.2f}",
                    "Trades": p.trade_count,
                }
                for p in data.positions
            ]
            st.dataframe(pos_data, use_container_width=True, hide_index=True)
        else:
            st.info("No active positions.")

    # --- Section 5: Recent Trades ---
    with st.expander(f"Recent Trades ({len(data.recent_trades)})"):
        if data.recent_trades:
            trade_data = []
            for t in data.recent_trades:
                ts = t.get("timestamp", "")
                if isinstance(ts, str) and len(ts) > 19:
                    ts = ts[:19]
                side = t.get("side", "").upper()
                trade_data.append(
                    {
                        "Time": ts,
                        "Symbol": t.get("symbol", ""),
                        "Side": side,
                        "Quantity": t.get("quantity", ""),
                        "Price": t.get("price", ""),
                        "Fee": t.get("fee", ""),
                        "PnL": t.get("pnl", ""),
                        "Status": t.get("status", ""),
                    }
                )
            st.dataframe(trade_data, use_container_width=True, hide_index=True)
        else:
            st.info("No recent trades.")

    # Footer
    if data.stats.first_trade_date:
        st.caption(
            f"Data range: {data.stats.first_trade_date} ~ "
            f"{data.stats.last_trade_date} | "
            f"Updated: {data.last_updated:%H:%M:%S}"
        )
