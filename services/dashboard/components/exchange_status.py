"""Exchange enable/disable status panel."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from services.dashboard.constants import DEFAULT_AUTO_REFRESH_INTERVAL


def _get_refresh_interval() -> float:
    """Safely get auto refresh interval from environment."""
    raw = os.getenv("MASP_AUTO_REFRESH_INTERVAL", str(DEFAULT_AUTO_REFRESH_INTERVAL))
    try:
        v = float(raw)
        if not math.isfinite(v) or v <= 0:
            return DEFAULT_AUTO_REFRESH_INTERVAL
        return v
    except ValueError:
        return 10.0


_AUTO_REFRESH_INTERVAL = _get_refresh_interval()

_SCHEDULE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "config"
    / "schedule_config.json"
)

# Fallback demo data when schedule_config.json is unavailable
_DEMO_EXCHANGE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "upbit_ankle": {
        "enabled": True,
        "exchange": "upbit",
        "strategy": "ankle_buy_v2",
        "websocket_monitor": True,
    },
    "bithumb_ankle": {
        "enabled": True,
        "exchange": "bithumb",
        "strategy": "ankle_buy_v2",
        "websocket_monitor": True,
    },
    "binance_spot_ankle": {
        "enabled": True,
        "exchange": "binance_spot",
        "strategy": "ankle_buy_v2",
        "websocket_monitor": True,
    },
}


def _load_schedule_config() -> Dict[str, Dict[str, Any]]:
    """Load exchange configs from schedule_config.json."""
    try:
        with open(_SCHEDULE_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f).get("exchanges", {})
    except Exception:
        return {}


class ExchangeStatusPanel:
    _LAST_RERUN_TS_KEY = "masp_last_auto_refresh_rerun_ts"

    def __init__(self, api_client) -> None:
        self.api = api_client
        self._is_demo_mode = False

    def _get_api_config(self, exchange: str) -> Optional[Dict[str, Any]]:
        """Get exchange config from API."""
        config = self.api.get_exchange_config(exchange)
        if config is None:
            self._is_demo_mode = True
        return config

    def render(self) -> None:
        self._is_demo_mode = False

        auto_refresh = st.checkbox(
            f"자동 새로고침 ({_AUTO_REFRESH_INTERVAL:.0f}초)",
            value=False,
            key="auto_refresh_enabled",
        )
        st.session_state["masp_auto_refresh"] = bool(auto_refresh)

        if auto_refresh:
            now = time.time()
            last = float(st.session_state.get(self._LAST_RERUN_TS_KEY, 0.0))

            if (now - last) >= _AUTO_REFRESH_INTERVAL:
                st.session_state[self._LAST_RERUN_TS_KEY] = now
                st.rerun()

        configs = _load_schedule_config()
        if not configs:
            configs = _DEMO_EXCHANGE_CONFIGS
            self._is_demo_mode = True

        for name, cfg in configs.items():
            enabled = cfg.get("enabled", False)
            strategy = cfg.get("strategy", "unknown")
            exchange = cfg.get("exchange", name)
            has_ws = cfg.get("websocket_monitor", False)

            col1, col2, col3, col4, col5 = st.columns([2, 1.5, 0.8, 0.5, 1])

            with col1:
                st.text(name)

            with col2:
                st.caption(strategy)

            with col3:
                if enabled:
                    st.markdown(":green[활성]")
                else:
                    st.markdown(":gray[비활성]")

            with col4:
                if has_ws:
                    st.caption("WS")

            with col5:
                # Try API toggle for active exchanges
                api_config = self._get_api_config(exchange)
                if api_config is not None and enabled:
                    btn_label = "비활성화"
                    if st.button(btn_label, key=f"status_toggle_{name}"):
                        success = self.api.toggle_exchange(exchange, False)
                        if success:
                            st.rerun()
                        else:
                            st.error(f"{name} 전환 실패.")

        if self._is_demo_mode:
            st.caption("데모 모드 - API 서버 연결 시 실제 상태 표시")
