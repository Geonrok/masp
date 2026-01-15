"""
Live test utilities (v2.2 Final).
Guards, logging, and shared constants.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


LIVE_GUARD_ENV = {
    "MASP_ENABLE_LIVE_TRADING": "1",
    "MASP_LIVE_TEST_ACK": "I_UNDERSTAND",
}


def live_test_enabled() -> bool:
    """Return True if live test guard envs are satisfied."""
    for key, expected in LIVE_GUARD_ENV.items():
        if os.getenv(key) != expected:
            return False
    return True


def require_live_guard() -> None:
    """Enforce live trading guard (fail fast)."""
    for key, expected in LIVE_GUARD_ENV.items():
        value = os.getenv(key)
        if value != expected:
            raise RuntimeError(f"Live guard failed: {key}={value} (need {expected})")


def get_loss_cap_krw() -> int:
    """Return loss cap in KRW (default: 50,000)."""
    raw = os.getenv("MASP_LIVE_LOSS_CAP_KRW", "50000")
    try:
        return int(raw)
    except ValueError:
        return 50000


def utc_now_iso_z() -> str:
    """UTC now in ISO 8601 Zulu format (Python 3.12+ compatible)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def enforce_budget_cap(order_notional_krw: int, cap_krw: Optional[int] = None) -> None:
    """
    Fail fast if the intended order notional exceeds the configured cap.

    NOTE: This is the maximum order notional allowed per test, not a minimum.
    """
    cap = int(cap_krw if cap_krw is not None else get_loss_cap_krw())
    if order_notional_krw > cap:
        raise RuntimeError(
            f"Order notional exceeds cap: {order_notional_krw:,} > {cap:,} KRW"
        )


def log_event(event: Dict[str, Any], log_path: str | None = None) -> None:
    """Append JSONL event to logs/live_tests/events.jsonl."""
    payload = dict(event)
    payload.setdefault("ts", utc_now_iso_z())
    path = Path(log_path or "logs/live_tests/events.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
