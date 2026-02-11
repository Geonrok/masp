"""Ankle Buy v2.0 strategy status panel for monitoring tab."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

_SCHEDULE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "config"
    / "schedule_config.json"
)
_LOGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"


def _load_ankle_configs() -> Dict[str, Dict[str, Any]]:
    """Load ankle_buy configs from schedule_config.json."""
    try:
        with open(_SCHEDULE_CONFIG_PATH, encoding="utf-8") as f:
            all_configs = json.load(f).get("exchanges", {})
    except Exception:
        return {}

    return {
        name: cfg
        for name, cfg in all_configs.items()
        if cfg.get("strategy") == "ankle_buy_v2"
    }


def _load_recent_trades(
    exchange_key: str, max_rows: int = 10
) -> Optional[pd.DataFrame]:
    """Load recent trades from CSV trade logs."""
    trade_dir = _LOGS_DIR / f"{exchange_key}_trades" / "trades"
    if not trade_dir.exists():
        return None

    csv_files = sorted(trade_dir.glob("*.csv"), reverse=True)
    if not csv_files:
        return None

    try:
        frames: List[pd.DataFrame] = []
        for csv_path in csv_files[:3]:
            df = pd.read_csv(csv_path)
            frames.append(df)
            if sum(len(f) for f in frames) >= max_rows:
                break

        if not frames:
            return None

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.tail(max_rows)

        display_cols = []
        for col in ["timestamp", "symbol", "side", "price", "quantity", "reason"]:
            if col in combined.columns:
                display_cols.append(col)

        if not display_cols:
            return combined.tail(max_rows)

        return combined[display_cols].tail(max_rows)
    except Exception as exc:
        logger.warning("Failed to load trades for %s: %s", exchange_key, exc)
        return None


def render_ankle_strategy_status() -> None:
    """Ankle Buy v2.0 strategy status - 3-exchange unified view."""
    configs = _load_ankle_configs()

    if not configs:
        st.info("Ankle Buy v2.0 설정을 찾을 수 없습니다.")
        return

    st.subheader("Ankle Buy v2.0 전략 상태")

    cols = st.columns(len(configs))
    for col, (name, cfg) in zip(cols, configs.items()):
        with col:
            exchange = cfg.get("exchange", name)
            enabled = cfg.get("enabled", False)

            st.markdown(f"### {exchange.upper()}")

            # Status
            if enabled:
                st.markdown(":green[활성]")
            else:
                st.markdown(":red[비활성]")

            # Config details
            close_hour = cfg.get("daily_close_hour", "-")
            close_tz = cfg.get("daily_close_timezone", "-")
            st.caption(f"일봉마감: {close_hour}:00 {close_tz}")

            ws_on = cfg.get("websocket_monitor", False)
            st.caption(f"WS 모니터: {'ON' if ws_on else 'OFF'}")

            sizing = cfg.get("position_size_mode", "-")
            st.caption(f"사이징: {sizing}")

            # Min position
            min_krw = cfg.get("min_position_krw")
            min_usdt = cfg.get("min_position_usdt")
            if min_krw:
                st.caption(f"최소: {min_krw:,} KRW")
            elif min_usdt:
                st.caption(f"최소: ${min_usdt}")

            # Recent trades
            trades = _load_recent_trades(name)
            if trades is not None and not trades.empty:
                st.divider()
                st.caption("최근 거래")
                st.dataframe(trades, hide_index=True, use_container_width=True)
            else:
                st.info("거래 내역 없음")
