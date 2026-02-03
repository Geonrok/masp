"""
Live balance check (v2.2 Final).
"""

from __future__ import annotations

import os
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
from libs.core.config import Config
from _live_test_utils import live_test_enabled, require_live_guard

MIN_ORDER_KRW = 5000

if pytest and not live_test_enabled():
    pytest.skip(
        "Live trading tests disabled (env guard not satisfied)", allow_module_level=True
    )


def test_live_balance(execution=None) -> bool:
    """Check KRW/BTC balances before live tests."""
    print("=" * 60)
    print("Live Balance Test")
    print("=" * 60)

    require_live_guard()

    if execution is None:
        config = Config(asset_class="crypto_spot", strategy_name="live_test")
        execution = AdapterFactory.create_execution(
            "upbit_spot", adapter_mode="live", config=config
        )

    krw_balance = execution.get_balance("KRW") or 0
    btc_balance = execution.get_balance("BTC") or 0
    print(f"KRW balance: {krw_balance:,.0f}")
    print(f"BTC balance: {btc_balance:.8f}")

    if krw_balance < MIN_ORDER_KRW:
        print(
            f"❌ Insufficient KRW for min order: {krw_balance:,.0f} < {MIN_ORDER_KRW:,}"
        )
        return False

    print("✅ Balance check PASSED")
    return True


if __name__ == "__main__":
    success = test_live_balance()
    sys.exit(0 if success else 1)
