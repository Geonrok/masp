"""PnL (Profit and Loss) calculator utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


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


def calculate_portfolio_pnl(
    positions: List[Dict],
    current_prices: Dict[str, float],
) -> List[PositionPnL]:
    """Calculate PnL for all positions."""
    results = []
    for pos in positions:
        symbol = pos.get("symbol", "")
        quantity = float(pos.get("quantity", 0))
        avg_price = float(pos.get("avg_price", 0))
        current_price = current_prices.get(symbol, avg_price)

        if quantity > 0:
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
