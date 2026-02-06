"""Kiwoom Sector Rotation strategy status component.

Displays strategy configuration, account summary, trade statistics,
positions, and recent trades for the Kiwoom monthly rebalancing strategy.
"""

from __future__ import annotations

from typing import Callable, Optional

import streamlit as st


def render_kiwoom_status(status_provider: Optional[Callable] = None) -> None:
    """Render Kiwoom Sector Rotation strategy status panel.

    Args:
        status_provider: Callable that returns KiwoomStatusData.
                        If None, uses default provider.
    """
    if status_provider is None:
        from services.dashboard.providers.kiwoom_provider import get_kiwoom_status

        status_provider = get_kiwoom_status

    status = status_provider()

    # Header with strategy name and mode
    mode_color = "#4CAF50" if status.config.mode == "live" else "#2196F3"
    mode_text = "LIVE" if status.config.mode == "live" else "Paper"

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {mode_color}22, {mode_color}11);
                    border-left: 4px solid {mode_color};
                    padding: 12px 16px;
                    border-radius: 4px;
                    margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 18px; font-weight: bold;">
                        🇰🇷 KIWOOM - {status.config.strategy_name.upper()}
                    </span>
                    <span style="background: {mode_color}; color: white;
                                 padding: 2px 8px; border-radius: 4px;
                                 font-size: 12px; margin-left: 8px;">
                        {mode_text}
                    </span>
                </div>
                <div style="text-align: right; font-size: 12px; color: #666;">
                    {"✅ 실시간 데이터" if status.data_available else "📊 데모 데이터"}
                </div>
            </div>
            <div style="font-size: 13px; color: #555; margin-top: 6px;">
                스케줄: {status.config.schedule_day} {status.config.schedule_hour:02d}:{status.config.schedule_minute:02d} {status.config.schedule_timezone}
                | 종목: {len(status.config.symbols)}개
                | 포지션 크기: ₩{status.config.position_size_krw:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Account summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "초기 잔고",
            f"₩{status.account.initial_balance:,.0f}",
        )

    with col2:
        st.metric(
            "현재 잔고",
            f"₩{status.account.estimated_balance:,.0f}",
            delta=f"{status.account.pnl_percent:+.2f}%",
        )

    with col3:
        pnl_delta = "+" if status.account.total_pnl >= 0 else ""
        st.metric(
            "실현 PnL",
            f"₩{status.account.total_pnl:,.0f}",
            delta=f"{pnl_delta}{status.account.total_pnl:,.0f}",
        )

    with col4:
        st.metric(
            "수수료",
            f"₩{status.account.total_fees:,.0f}",
        )

    # Trade statistics
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("리밸런싱 횟수", status.stats.rebalance_count)

    with col2:
        st.metric("전체 거래", status.stats.trades_total)

    with col3:
        st.metric("매수/매도", f"{status.stats.buy_count}/{status.stats.sell_count}")

    with col4:
        st.metric("보유 종목", status.stats.unique_symbols)

    # Positions expander
    with st.expander(f"📈 활성 포지션 ({len(status.positions)}개)", expanded=False):
        if status.positions:
            position_data = [
                {
                    "종목코드": p.symbol,
                    "종목명": p.symbol_name,
                    "방향": p.side,
                    "수량": f"{p.net_quantity:,.0f}",
                    "평균단가": f"₩{p.avg_entry_price:,.0f}",
                    "평가금액": f"₩{p.notional_value:,.0f}",
                }
                for p in status.positions
            ]
            st.dataframe(position_data, use_container_width=True, hide_index=True)
        else:
            st.info("활성 포지션이 없습니다.")

    # Recent trades expander
    with st.expander(f"📋 최근 거래 ({len(status.recent_trades)}건)", expanded=False):
        if status.recent_trades:
            trade_data = [
                {
                    "시간": t.get("timestamp", "")[:19],
                    "종목": t.get("symbol", ""),
                    "종목명": _get_stock_name(t.get("symbol", "")),
                    "방향": t.get("side", ""),
                    "수량": t.get("quantity", ""),
                    "가격": f"₩{float(t.get('price', 0)):,.0f}",
                    "수수료": f"₩{float(t.get('fee', 0)):,.0f}",
                }
                for t in status.recent_trades
            ]
            st.dataframe(trade_data, use_container_width=True, hide_index=True)
        else:
            st.info("최근 거래 내역이 없습니다.")

    # Footer with last update time
    st.caption(
        f"마지막 업데이트: {status.last_updated.strftime('%Y-%m-%d %H:%M:%S')}"
    )


def _get_stock_name(symbol: str) -> str:
    """Get stock name from symbol code."""
    names = {
        "000660": "SK하이닉스",
        "005930": "삼성전자",
        "003670": "포스코퓨처엠",
        "042700": "한미반도체",
        "006400": "삼성SDI",
    }
    return names.get(symbol, symbol)
