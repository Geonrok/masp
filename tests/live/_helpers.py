"""
Fee retrieval and market normalization (v2.2 Final).
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Dict
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, ROOT_DIR)


def resolve_market(symbol: str) -> str:
    """
    Convert symbol to Upbit market code.
    BTC/KRW -> KRW-BTC
    """
    if "/" in symbol:
        base, quote = symbol.split("/")
        return f"{quote}-{base}"
    if "-" in symbol:
        return symbol
    return f"KRW-{symbol}"


def _fee_rate_from_chance(chance: Dict, side: str) -> Decimal:
    """
    Extract fee rate from /orders/chance response.

    Priority:
    1) maker_{side}_fee
    2) {side}_fee
    """
    keys = [f"maker_{side}_fee", f"{side}_fee"]
    for key in keys:
        value = chance.get(key)
        if value is not None:
            try:
                return Decimal(str(value))
            except (InvalidOperation, ValueError) as exc:
                raise RuntimeError(f"Fee parsing failed: {key}={value}, error={exc}")
    raise RuntimeError(f"Fee field missing: tried {keys}")


def get_fee_rates(market: str, execution) -> Dict:
    """
    Retrieve fee rates from /v1/orders/chance.
    Fail fast on errors.
    """
    try:
        if hasattr(execution, "get_order_chance"):
            data = execution.get_order_chance(market)
        else:
            data = execution._request("GET", "/orders/chance", params={"market": market}, is_order=True)
        if data is None:
            raise RuntimeError("orders/chance returned None")
        market_info = data.get("market", {}).get("bid", {})
        min_total = market_info.get("min_total", "5000")
        return {
            "bid_fee": _fee_rate_from_chance(data, "bid"),
            "ask_fee": _fee_rate_from_chance(data, "ask"),
            "source": "chance",
            "min_total": Decimal(str(min_total)),
            "raw": data,
        }
    except Exception as exc:
        raise RuntimeError(f"Fee retrieval FAIL FAST: {exc}")


def calc_expected_fee(side: str, notional_krw: float, fee_rates: Dict) -> Decimal:
    """Calculate expected fee for notional in KRW."""
    rate = fee_rates["bid_fee"] if side.lower() == "buy" else fee_rates["ask_fee"]
    return Decimal(str(notional_krw)) * rate
