"""
Market buy test (v2.2 Final).
"""
from __future__ import annotations

import os
import sys
import time
from decimal import Decimal
from typing import Any

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
from _helpers import resolve_market, get_fee_rates, calc_expected_fee
from _rate_limiter import RateLimiter
from _live_test_utils import (
    live_test_enabled,
    require_live_guard,
    log_event,
    get_loss_cap_krw,
    enforce_budget_cap,
)

if pytest and not live_test_enabled():
    pytest.skip("Live trading tests disabled (env guard not satisfied)", allow_module_level=True)


MIN_ORDER_KRW = 5000
limiter = RateLimiter()


def _extract_field(order_info: Any, key: str, default=None):
    if order_info is None:
        return default
    if isinstance(order_info, dict):
        return order_info.get(key, default)
    return getattr(order_info, key, default)


def _post_verify_upbit_market_buy(order_info: Any, order_amount_krw: int) -> str:
    """
    Upbit market-buy post verification.
    Spec: BUY + MARKET -> ord_type='price', price=KRW, volume absent.
    """
    ord_type = _extract_field(order_info, "ord_type", None)
    price = _extract_field(order_info, "price", None)
    volume = _extract_field(order_info, "volume", None)

    if ord_type is None and price is None and volume is None:
        return "SKIP (order details not available)"

    if str(ord_type) != "price":
        return f"FAIL (ord_type={ord_type}, expected 'price')"

    try:
        price_num = float(price) if price is not None else 0.0
        if price_num <= 0:
            return f"FAIL (price={price}, expected > 0)"
        if abs(price_num - float(order_amount_krw)) / float(order_amount_krw) > 0.01:
            return f"FAIL (price={price_num}, expected ~= {order_amount_krw})"
    except Exception:
        return f"FAIL (price parse error: {price})"

    if volume not in (None, "", "0", 0, 0.0):
        return f"FAIL (volume should be absent, got {volume})"

    return "PASS"


def test_live_buy(execution=None, fee_rates=None) -> dict:
    """
    Market buy test.

    Note: UpbitSpotExecution expects KRW amount via quantity for MARKET BUY.
    """
    print("=" * 60)
    print("Live Buy Test (Market)")
    print("=" * 60)

    require_live_guard()

    if execution is None:
        config = Config(asset_class="crypto_spot", strategy_name="live_test")
        execution = AdapterFactory.create_execution("upbit_spot", adapter_mode="live", config=config)

    market = resolve_market("BTC/KRW")

    if fee_rates is None:
        fee_rates = get_fee_rates(market, execution)

    bid_fee = fee_rates["bid_fee"]
    print(f"\n[1] Fee rate: {bid_fee * 100:.4f}%")

    krw_before = execution.get_balance("KRW") or 0
    btc_before = execution.get_balance("BTC") or 0
    print(f"\n[2] Before: KRW={krw_before:,.0f}, BTC={btc_before:.8f}")

    if krw_before < MIN_ORDER_KRW:
        print(f"❌ Insufficient KRW: {krw_before:,.0f} < {MIN_ORDER_KRW:,}")
        return {"status": "failed", "reason": "Insufficient balance"}

    min_total = fee_rates.get("min_total", Decimal(str(MIN_ORDER_KRW)))
    order_amount = max(MIN_ORDER_KRW, int(min_total))
    loss_cap = get_loss_cap_krw()
    try:
        enforce_budget_cap(order_amount, loss_cap)
    except Exception as exc:
        print(f"❌ Budget cap guard: {exc}")
        return {"status": "failed", "reason": "Budget cap exceeded"}
    expected_fee = calc_expected_fee("buy", order_amount, fee_rates)
    print(f"\n[3] Order amount: {order_amount:,} KRW (fee ~{expected_fee:.2f} KRW)")

    print("\n[4] Placing market buy...")
    try:
        order = limiter.execute(
            execution.place_order,
            symbol="BTC/KRW",
            side="BUY",
            quantity=order_amount,
            order_type="MARKET",
        )
        order_id = getattr(order, "order_id", None) or getattr(order, "uuid", None) or "N/A"
        if hasattr(order, "success") and not order.success:
            return {"status": "failed", "reason": getattr(order, "message", "Order failed")}
        print(f"    Order ID: {order_id}")
    except Exception as exc:
        print(f"❌ Order failed: {exc}")
        return {"status": "failed", "reason": str(exc)}

    print("\n[5] Waiting for fill...")
    final_status = None
    last_info = None
    for i in range(10):
        time.sleep(1)
        try:
            status = execution.get_order_status(order_id)
            last_info = status
            state = status.status if hasattr(status, "status") else status.get("state", "")
            print(f"    [{i + 1}s] Status: {state}")
            if str(state).lower() in ["done", "filled", "cancel", "cancelled"]:
                final_status = state
                break
        except Exception:
            pass

    krw_after = execution.get_balance("KRW") or 0
    btc_after = execution.get_balance("BTC") or 0
    print(f"\n[6] After: KRW={krw_after:,.0f}, BTC={btc_after:.8f}")

    krw_diff = krw_before - krw_after
    btc_diff = btc_after - btc_before
    print(f"\n[7] Diff: KRW=-{krw_diff:,.0f}, BTC=+{btc_diff:.8f}")

    post_verify = _post_verify_upbit_market_buy(last_info, order_amount)
    print(f"\n[8] Post-verify (ord_type/price/volume): {post_verify}")

    log_event(
        {
            "type": "buy_test",
            "order_id": order_id,
            "amount": order_amount,
            "krw_diff": float(krw_diff),
            "btc_diff": float(btc_diff),
            "fee_rate": float(bid_fee),
            "final_status": str(final_status),
        }
    )

    if krw_diff >= order_amount * 0.98 and btc_diff > 0:
        print("\n" + "=" * 60)
        print("✅ Live Buy Test PASSED")
        print("=" * 60)
        return {"status": "passed", "order_id": order_id, "btc_received": btc_diff}

    print("\n❌ Verification failed")
    return {"status": "failed", "reason": "Verification failed"}


if __name__ == "__main__":
    result = test_live_buy()
    sys.exit(0 if result.get("status") == "passed" else 1)
