"""
Open order cleanup for live tests (v2.2 Final).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Dict, Optional


def _to_aware_utc(dt: datetime) -> datetime:
    """Convert naive datetime to aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def list_open_orders(
    execution,
    market: Optional[str] = None,
    since_ts: Optional[datetime] = None,
) -> List[Dict]:
    """Fetch open orders with optional time filter."""
    try:
        if hasattr(execution, "get_open_orders"):
            orders = execution.get_open_orders(market)
        else:
            params = {"state": "wait"}
            if market:
                params["market"] = market
            orders = execution._request("GET", "/orders", params=params, is_order=True)
        if orders is None:
            return []
        if since_ts:
            since_utc = _to_aware_utc(since_ts)
            filtered = []
            for order in orders:
                created_at = order.get("created_at", "")
                if created_at:
                    try:
                        order_time = datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        )
                        order_time_utc = order_time.astimezone(timezone.utc)
                        if order_time_utc >= since_utc:
                            filtered.append(order)
                    except Exception:
                        pass
            orders = filtered
        print(f"Open orders: {len(orders)}")
        return orders
    except Exception as exc:
        print(f"⚠️ Open order lookup failed: {exc}")
        return []


def cancel_all_open_orders(
    execution,
    market: Optional[str] = None,
    since_ts: Optional[datetime] = None,
) -> Dict:
    """Cancel open orders (bulk then single fallback)."""
    result = {"bulk_success": False, "cancelled": 0, "failed": 0}
    open_orders = list_open_orders(execution, market, since_ts)
    if not open_orders:
        print("✅ No open orders to cancel")
        return result
    if market:
        try:
            if hasattr(execution, "cancel_all_orders"):
                bulk_result = execution.cancel_all_orders(market)
            else:
                bulk_result = execution._request(
                    "DELETE", "/orders/open", params={"market": market}, is_order=True
                )
            if bulk_result is not None:
                result["bulk_success"] = True
                result["cancelled"] = len(open_orders)
                print(f"✅ Batch cancel ok: {len(open_orders)}")
                return result
        except Exception as exc:
            print(f"⚠️ Batch cancel failed, fallback to single: {exc}")
    for order in open_orders:
        uuid = order.get("uuid")
        if not uuid:
            continue
        try:
            execution.cancel_order(uuid)
            result["cancelled"] += 1
            print(f"  Cancelled: {uuid[:8]}...")
        except Exception as exc:
            result["failed"] += 1
            print(f"  Failed: {uuid[:8]}... ({exc})")
        time.sleep(0.15)
    return result


def verify_no_open_orders(
    execution,
    market: Optional[str] = None,
    since_ts: Optional[datetime] = None,
    retry: int = 3,
    backoff: float = 0.5,
) -> bool:
    """Verify no open orders remain."""
    last_count = 0
    for attempt in range(retry):
        orders = list_open_orders(execution, market, since_ts)
        last_count = len(orders)
        if last_count == 0:
            print("✅ No open orders remaining")
            return True
        time.sleep(backoff * (2**attempt))
    print(f"⚠️ Open orders remaining: {last_count}")
    return False
