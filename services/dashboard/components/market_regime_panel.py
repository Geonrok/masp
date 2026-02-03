"""
Market Regime Panel Component

시장 국면 분석 결과를 대시보드에 표시합니다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.analysis.market_regime import (
    MarketRegime,
    MomentumState,
    RegimeAnalysis,
    VolatilityRegime,
)


def render_market_regime_panel(
    analysis_provider: Optional[Callable[[], RegimeAnalysis]] = None,
) -> None:
    """
    시장 국면 패널 렌더링

    Args:
        analysis_provider: RegimeAnalysis를 반환하는 함수
    """
    st.subheader("시장 국면 분석")

    if analysis_provider is None:
        st.warning("시장 국면 데이터 프로바이더가 설정되지 않았습니다.")
        return

    try:
        analysis = analysis_provider()
    except Exception as e:
        st.error(f"시장 국면 분석 오류: {e}")
        return

    if analysis.regime == MarketRegime.UNKNOWN:
        st.warning("시장 국면을 분석할 수 없습니다.")
        return

    # === Row 1: Regime Overview ===
    col1, col2, col3 = st.columns(3)

    with col1:
        regime_colors = {
            MarketRegime.BULL: "green",
            MarketRegime.BEAR: "red",
            MarketRegime.SIDEWAYS: "orange",
            MarketRegime.UNKNOWN: "gray",
        }
        regime_icons = {
            MarketRegime.BULL: "UP",
            MarketRegime.BEAR: "DOWN",
            MarketRegime.SIDEWAYS: "--",
            MarketRegime.UNKNOWN: "?",
        }
        color = regime_colors[analysis.regime]
        icon = regime_icons[analysis.regime]

        st.markdown(
            f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0;">{icon}</h2>
                <h3 style="color: white; margin: 5px 0;">{analysis.regime.value.upper()}</h3>
                <p style="color: white; margin: 0; font-size: 12px;">Market Regime</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        vol_colors = {
            VolatilityRegime.LOW: "#4CAF50",
            VolatilityRegime.NORMAL: "#2196F3",
            VolatilityRegime.HIGH: "#FF9800",
            VolatilityRegime.EXTREME: "#F44336",
        }
        vol_color = vol_colors[analysis.volatility]

        st.markdown(
            f"""
            <div style="background-color: {vol_color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0;">{analysis.atr_pct:.1f}%</h2>
                <h3 style="color: white; margin: 5px 0;">{analysis.volatility.value.upper()}</h3>
                <p style="color: white; margin: 0; font-size: 12px;">Volatility (ATR%)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        mom_colors = {
            MomentumState.STRONG_UP: "#4CAF50",
            MomentumState.WEAK_UP: "#8BC34A",
            MomentumState.NEUTRAL: "#9E9E9E",
            MomentumState.WEAK_DOWN: "#FF9800",
            MomentumState.STRONG_DOWN: "#F44336",
        }
        mom_color = mom_colors[analysis.momentum]

        st.markdown(
            f"""
            <div style="background-color: {mom_color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0;">{analysis.tsmom_30d:+.1f}%</h2>
                <h3 style="color: white; margin: 5px 0;">{analysis.momentum.value.replace('_', ' ').upper()}</h3>
                <p style="color: white; margin: 0; font-size: 12px;">30D Momentum</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # === Row 2: Price & MAs ===
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**BTC 가격 & 이동평균**")

        price_data = [
            ("현재가", analysis.price, ""),
            ("MA20", analysis.ma20, _calc_diff_str(analysis.price, analysis.ma20)),
            ("MA50", analysis.ma50, _calc_diff_str(analysis.price, analysis.ma50)),
            ("MA200", analysis.ma200, _calc_diff_str(analysis.price, analysis.ma200)),
        ]

        for label, value, diff in price_data:
            col_a, col_b, col_c = st.columns([2, 3, 2])
            with col_a:
                st.text(label)
            with col_b:
                st.text(f"{value:,.0f}")
            with col_c:
                if diff:
                    color = "green" if diff.startswith("+") else "red"
                    st.markdown(
                        f"<span style='color:{color}'>{diff}</span>",
                        unsafe_allow_html=True,
                    )

    with col2:
        st.markdown("**추세 & 모멘텀 지표**")

        # Trend Strength Bar
        st.markdown(f"추세 강도: {analysis.trend_strength:.0f}/100")
        st.progress(int(analysis.trend_strength) / 100)

        # Momentum
        st.markdown(f"30일 모멘텀: {analysis.tsmom_30d:+.1f}%")
        st.markdown(f"90일 모멘텀: {analysis.tsmom_90d:+.1f}%")

    # === Row 3: Analysis Message ===
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"**분석**: {analysis.message}")

    # Timestamp
    st.caption(f"마지막 업데이트: {analysis.timestamp.strftime('%Y-%m-%d %H:%M')}")


def _calc_diff_str(current: float, ma: float) -> str:
    """가격과 MA 차이 문자열"""
    if ma == 0:
        return ""
    diff_pct = (current - ma) / ma * 100
    return f"{diff_pct:+.1f}%"


def render_signal_status_panel(
    signal_provider: Optional[Callable[[], dict]] = None,
) -> None:
    """
    시그널 상태 패널 렌더링

    Args:
        signal_provider: 시그널 정보를 반환하는 함수
    """
    st.subheader("일간 시그널 현황")

    if signal_provider is None:
        st.warning("시그널 프로바이더가 설정되지 않았습니다.")
        return

    try:
        summary = signal_provider()
    except Exception as e:
        st.error(f"시그널 조회 오류: {e}")
        return

    if summary.get("error"):
        st.warning(summary.get("message", "Unknown error"))
        return

    # === BTC Gate Status ===
    col1, col2, col3 = st.columns(3)

    with col1:
        gate_status = summary.get("btc_gate", False)
        gate_color = "#4CAF50" if gate_status else "#F44336"
        gate_text = "PASS" if gate_status else "FAIL"

        st.markdown(
            f"""
            <div style="background-color: {gate_color}; padding: 15px; border-radius: 8px; text-align: center;">
                <h3 style="color: white; margin: 0;">BTC Gate: {gate_text}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric(
            label="BTC 현재가",
            value=f"{summary.get('btc_price', 0):,.0f}",
        )

    with col3:
        st.metric(
            label="Entry 시그널",
            value=f"{summary.get('entry_signals', 0)}/{summary.get('total_symbols', 0)}",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # === Signal List ===
    signals = summary.get("signals", {})

    if signals:
        # Entry signals
        entry_list = [(s, d) for s, d in signals.items() if d.get("entry_signal")]
        entry_list.sort(key=lambda x: x[1].get("change_7d", 0), reverse=True)

        if entry_list:
            st.markdown("**Entry 시그널**")

            for symbol, data in entry_list[:10]:
                kama = "KAMA" if data.get("kama_signal") else ""
                tsmom = "TSMOM" if data.get("tsmom_signal") else ""
                signal_str = "+".join(filter(None, [kama, tsmom]))

                col_a, col_b, col_c, col_d = st.columns([3, 2, 2, 2])
                with col_a:
                    st.text(symbol)
                with col_b:
                    st.text(f"{data.get('price', 0):,.0f}")
                with col_c:
                    st.text(signal_str)
                with col_d:
                    change = data.get("change_7d", 0)
                    color = "green" if change > 0 else "red"
                    st.markdown(
                        f"<span style='color:{color}'>{change:+.1f}%</span>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("현재 Entry 시그널이 없습니다.")

    st.caption(f"전략: {summary.get('strategy', 'N/A')}")
    st.caption(f"마지막 업데이트: {summary.get('timestamp', 'N/A')}")
