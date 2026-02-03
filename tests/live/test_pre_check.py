"""
Pre-check for live trading tests (v2.2 Final).
"""

from __future__ import annotations

import os
import socket
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, ROOT_DIR)

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from libs.adapters.factory import AdapterFactory
from _live_test_utils import live_test_enabled

if pytest and not live_test_enabled():
    pytest.skip(
        "Live trading tests disabled (env guard not satisfied)", allow_module_level=True
    )


def test_pre_check() -> bool:
    """Validate network, market access, and env guards."""
    print("=" * 60)
    print("Pre-Check: network/market/env")
    print("=" * 60)

    print("\n[1] Network connectivity")
    try:
        socket.create_connection(("api.upbit.com", 443), timeout=5)
        print("✅ api.upbit.com:443 reachable")
    except Exception as exc:
        print(f"❌ Network check failed: {exc}")
        return False

    print("\n[2] Market check (BTC/KRW)")
    try:
        market_data = AdapterFactory.create_market_data("upbit_spot")
        quote = market_data.get_quote("BTC/KRW")
        if not quote or not quote.last:
            print("❌ BTC price unavailable")
            return False
        print(f"✅ BTC/KRW = {quote.last:,.0f} KRW")
        if quote.bid and quote.ask:
            spread = (quote.ask - quote.bid) / quote.last * 100
            if spread > 1.0:
                print(f"⚠️ Spread warning: {spread:.2f}% (>1%)")
            else:
                print(f"✅ Spread: {spread:.4f}%")
    except Exception as exc:
        print(f"❌ Market check failed: {exc}")
        return False

    print("\n[3] Environment guards")
    required = {
        "MASP_ENABLE_LIVE_TRADING": "1",
        "MASP_LIVE_TEST_ACK": "I_UNDERSTAND",
    }
    for key, expected in required.items():
        value = os.getenv(key)
        if value != expected:
            print(f"❌ {key} = {value} (need {expected})")
            return False
        print(f"✅ {key} = {value}")

    print("\n" + "=" * 60)
    print("✅ Pre-Check PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_pre_check()
    sys.exit(0 if success else 1)
