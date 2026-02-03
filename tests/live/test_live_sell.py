"""
Market sell test (v2.2 Final).
"""

from __future__ import annotations

import os
import sys
import time

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
from _helpers import resolve_market, get_fee_rates
from _rate_limiter import RateLimiter
from _live_test_utils import live_test_enabled, require_live_guard, log_event

if pytest and not live_test_enabled():
    pytest.skip(
        "Live trading tests disabled (env guard not satisfied)", allow_module_level=True
    )


MIN_ORDER_KRW = 5000
limiter = RateLimiter()


def test_live_sell(execution=None, fee_rates=None) -> dict:
    """
    Market sell test (volume-based).
    """
    print("=" * 60)
    print("Live Sell Test (Market)")
    print("=" * 60)

    require_live_guard()

    if execution is None:
        config = Config(asset_class="crypto_spot", strategy_name="live_test")
        execution = AdapterFactory.create_execution(
            "upbit_spot", adapter_mode="live", config=config
        )

    market_data = AdapterFactory.create_market_data("upbit_spot")
    quote = market_data.get_quote("BTC/KRW")
    current_price = quote.last if quote else 0
    print(f"\n[1] BTC price: {current_price:,.0f} KRW")

    btc_before = execution.get_balance("BTC") or 0
    krw_before = execution.get_balance("KRW") or 0
    estimated_value = btc_before * current_price
    print(f"\n[2] Before: BTC={btc_before:.8f} (~{estimated_value:,.0f} KRW)")
    print(f"    KRW={krw_before:,.0f}")

    if estimated_value < MIN_ORDER_KRW:
        print(f"\n⚠️ Dust detected: {estimated_value:,.0f} < {MIN_ORDER_KRW:,} KRW")
        print("    Skip sell (expected)")
        return {"status": "skipped", "reason": "Dust holding"}

    print(f"\n[3] Market sell: {btc_before:.8f} BTC")
    try:
        order = limiter.execute(
            execution.place_order,
            symbol="BTC/KRW",
            side="SELL",
            quantity=btc_before,
            order_type="MARKET",
        )
        order_id = (
            getattr(order, "order_id", None) or getattr(order, "uuid", None) or "N/A"
        )
        if hasattr(order, "success") and not order.success:
            return {
                "status": "failed",
                "reason": getattr(order, "message", "Order failed"),
            }
        print(f"    Order ID: {order_id}")
    except Exception as exc:
        print(f"❌ Order failed: {exc}")
        return {"status": "failed", "reason": str(exc)}

    print("\n[4] Waiting for fill...")
    for i in range(10):
        time.sleep(1)
        try:
            status = execution.get_order_status(order_id)
            state = (
                status.status if hasattr(status, "status") else status.get("state", "")
            )
            print(f"    [{i + 1}s] Status: {state}")
            if str(state).lower() in ["done", "filled", "cancel", "cancelled"]:
                break
        except Exception:
            pass

    btc_after = execution.get_balance("BTC") or 0
    krw_after = execution.get_balance("KRW") or 0
    print(f"\n[5] After: BTC={btc_after:.8f}, KRW={krw_after:,.0f}")

    krw_received = krw_after - krw_before
    print(f"\n[6] KRW received: +{krw_received:,.0f}")

    log_event(
        {
            "type": "sell_test",
            "order_id": order_id,
            "btc_sold": float(btc_before - btc_after),
            "krw_received": float(krw_received),
        }
    )

    if btc_after < btc_before and krw_received > 0:
        print("\n" + "=" * 60)
        print("✅ Live Sell Test PASSED")
        print("=" * 60)
        return {"status": "passed", "order_id": order_id, "krw_received": krw_received}

    print("\n❌ Verification failed")
    return {"status": "failed", "reason": "Verification failed"}


if __name__ == "__main__":
    result = test_live_sell()
    sys.exit(0 if result.get("status") in ["passed", "skipped"] else 1)
