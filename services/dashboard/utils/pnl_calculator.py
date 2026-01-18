"""PnL (Profit and Loss) calculator utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class PositionPnL:
    """PnL data for a single position."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float

    @property
    def cost_basis(self) -> float:
        """Total cost basis."""
        return self.quantity * self.avg_price

    @property
    def current_value(self) -> float:
        """Current market value."""
        return self.quantity * self.current_price

    @property
    def pnl_amount(self) -> float:
        """Unrealized PnL in currency."""
        return self.current_value - self.cost_basis

    @property
    def pnl_percent(self) -> float:
        """Unrealized PnL percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.pnl_amount / self.cost_basis) * 100


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float, returning default on failure."""
    if value is None:
        return default
    try:
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def calculate_portfolio_pnl(
    positions: List[Dict],
    current_prices: Dict[str, float],
) -> List[PositionPnL]:
    """Calculate PnL for all positions with safe parsing."""
    results = []
    for pos in positions:
        symbol = pos.get("symbol", "")
        if not symbol:
            continue

        quantity = _safe_float(pos.get("quantity"), 0.0)
        avg_price = _safe_float(pos.get("avg_price"), 0.0)
        current_price = _safe_float(current_prices.get(symbol), avg_price)

        if quantity > 0 and avg_price > 0:
            results.append(
                PositionPnL(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=avg_price,
                    current_price=current_price,
                )
            )
    return results


def calculate_total_pnl(pnl_list: List[PositionPnL]) -> Dict[str, float]:
    """Calculate total portfolio PnL summary."""
    total_cost = sum(p.cost_basis for p in pnl_list)
    total_value = sum(p.current_value for p in pnl_list)
    total_pnl = total_value - total_cost
    total_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

    return {
        "total_cost": total_cost,
        "total_value": total_value,
        "total_pnl": total_pnl,
        "total_pnl_percent": total_pct,
    }
