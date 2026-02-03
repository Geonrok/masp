"""Test Backtest Engine and Performance Analytics"""

from libs.analytics.performance import calculate_max_drawdown, calculate_sharpe
from libs.backtest.engine import BacktestEngine

print("=== Backtest Engine Test ===\n")

# Create simple test data
signals = ["BUY", "HOLD", "HOLD", "SELL", "BUY", "HOLD", "SELL", "BUY", "SELL"]
prices = [
    100_000_000,
    105_000_000,
    110_000_000,
    115_000_000,
    110_000_000,
    115_000_000,
    120_000_000,
    115_000_000,
    125_000_000,
]

# Run backtest
engine = BacktestEngine(initial_capital=10_000_000)
result = engine.run_simple(signals, prices)

print(f"[1] Backtest Results:")
print(f"  Total Trades: {result.total_trades}")
print(f"  Win Rate: {result.win_rate:.1f}%")
print(f"  Total PnL: {result.total_pnl:,.0f} KRW ({result.total_pnl_pct:.2f}%)")
print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")
print(f"  Profit Factor: {result.profit_factor:.2f}")

print(f"\n[2] Capital:")
print(f"  Initial: {result.initial_capital:,.0f} KRW")
print(f"  Final: {result.final_capital:,.0f} KRW")

print(f"\n[3] Performance Metrics:")
returns = [0.02, 0.01, -0.01, 0.03, 0.02]  # 2%, 1%, -1%, 3%, 2%
equity = [10_000_000, 10_200_000, 10_302_000, 10_199_000, 10_505_000, 10_715_000]

sharpe = calculate_sharpe(returns)
max_dd = calculate_max_drawdown(equity)

print(f"  Sharpe (test): {sharpe:.2f}")
print(f"  Max DD (test): {max_dd:.2f}%")

print("\n✅ Backtest Test Complete")
