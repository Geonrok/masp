"""Test Paper Execution Adapter"""

from libs.adapters.paper_execution import PaperExecutionAdapter
from libs.adapters.factory import AdapterFactory

# Create market data adapter
md = AdapterFactory.create_market_data('upbit_spot')

# Create paper execution with 10M KRW
pe = PaperExecutionAdapter(md, 10_000_000)

print("=== Paper Trading Test ===\n")

# Place BUY order
print("[1] Placing BUY order: 0.01 BTC")
order = pe.place_order('BTC/KRW', 'BUY', 0.01)
print(f"  Order ID: {order.order_id}")
print(f"  Status: {order.status}")
if order.price:
    print(f"  Fill Price: {order.price:,.0f} KRW")

# Check PnL
print(f"\n[2] PnL after order:")
pnl = pe.get_pnl()
print(f"  Equity: {pnl['equity']:,.0f} KRW")
print(f"  PnL: {pnl['total_pnl']:,.0f} KRW ({pnl['total_pnl_pct']:.2f}%)")
print(f"  Fees: {pnl['fees_paid']:,.0f} KRW")

# Check positions
print(f"\n[3] Positions:")
positions = pe.get_positions()
for symbol, pos in positions.items():
    print(f"  {symbol}: {pos.quantity} @ {pos.avg_price:,.0f} KRW avg")

print("\n✅ Paper Trading Test Complete")
