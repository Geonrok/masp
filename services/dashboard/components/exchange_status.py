"""Exchange enable/disable status panel."""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Optional

import streamlit as st

from services.dashboard.constants import (
    DEFAULT_AUTO_REFRESH_INTERVAL,
    SUPPORTED_EXCHANGES,
)


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

# Demo exchange configurations (used when API is unavailable)
_DEMO_EXCHANGE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "upbit": {"enabled": True, "name": "Upbit", "region": "KR", "type": "spot"},
    "bithumb": {"enabled": True, "name": "Bithumb", "region": "KR", "type": "spot"},
    "binance": {
        "enabled": False,
        "name": "Binance",
        "region": "Global",
        "type": "spot",
    },
    "binance_futures": {
        "enabled": False,
        "name": "Binance Futures",
        "region": "Global",
        "type": "futures",
    },
    "kiwoom": {
        "enabled": False,
        "name": "Kiwoom",
        "region": "KR",
        "type": "stock",
    },
}


class ExchangeStatusPanel:
    EXCHANGES = SUPPORTED_EXCHANGES
    _LAST_RERUN_TS_KEY = "masp_last_auto_refresh_rerun_ts"
    _DEMO_MODE_KEY = "exchange_status_demo_mode"

    def __init__(self, api_client) -> None:
        self.api = api_client
        self._is_demo_mode = False

    def _get_config(self, exchange: str) -> Optional[Dict[str, Any]]:
        """Get exchange config from API, fallback to demo data."""
        config = self.api.get_exchange_config(exchange)
        if config is None:
            self._is_demo_mode = True
            return _DEMO_EXCHANGE_CONFIGS.get(exchange)
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

        for exchange in self.EXCHANGES:
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.text(exchange.upper())

            with col2:
                config = self._get_config(exchange)
                if config is None:
                    status = "오프라인"
                    enabled = False
                else:
                    enabled = config.get("enabled", False)
                    status = "활성" if enabled else "비활성"

                # Color-coded status
                if enabled:
                    st.markdown(f":green[{status}]")
                else:
                    st.markdown(f":gray[{status}]")

            with col3:
                if config is not None:
                    btn_label = "비활성화" if enabled else "활성화"
                    if st.button(btn_label, key=f"status_toggle_{exchange}"):
                        if self._is_demo_mode:
                            # Toggle in demo mode (session state only)
                            demo_key = f"demo_exchange_{exchange}_enabled"
                            current = st.session_state.get(demo_key, enabled)
                            st.session_state[demo_key] = not current
                            _DEMO_EXCHANGE_CONFIGS[exchange]["enabled"] = not current
                            st.rerun()
                        else:
                            success = self.api.toggle_exchange(exchange, not enabled)
                            if success:
                                st.rerun()
                            else:
                                st.error(
                                    f"{exchange} 전환 실패. API 서버를 확인하세요."
                                )

        # Show demo mode indicator
        if self._is_demo_mode:
            st.caption("데모 모드 - API 서버 연결 시 실제 상태 표시")
