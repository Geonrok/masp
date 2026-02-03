"""
2025 Holdout Test - Fair Comparison v2
KAMA=10, TSMOM=60 vs KAMA=5, TSMOM=90

Fixed: Handle different column names across markets
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 80)
print("2025 HOLDOUT TEST - FAIR COMPARISON v2")
print("=" * 80)

# ============================================================
# Configuration
# ============================================================
DATA_ROOT = Path("E:/data/crypto_ohlcv")
MARKETS = {"binance": "binance_spot_1d", "upbit": "upbit_1d", "bithumb": "bithumb_1d"}

STRATEGIES = [
    {"name": "KAMA=10, TSMOM=60", "kama": 10, "tsmom": 60, "gate": 30},
    {"name": "KAMA=5, TSMOM=90", "kama": 5, "tsmom": 90, "gate": 30},
]

HOLDOUT_START = pd.Timestamp("2025-01-01")
HOLDOUT_END = pd.Timestamp("2025-12-31")
INITIAL_CAPITAL = 10000
MAX_POSITIONS = 20


# ============================================================
# Helper Functions
# ============================================================
def load_market_data(market_folder: Path) -> dict:
    """Load all CSV files from market folder, handling different column names"""
    data = {}
    csv_files = list(market_folder.glob("*.csv"))

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Find date column
            date_col = None
            for col in df.columns:
                if "timestamp" in col.lower() or "date" in col.lower():
                    date_col = col
                    break

            if date_col is None or "close" not in df.columns:
                continue

            # Normalize to 'date' column
            df["date"] = pd.to_datetime(df[date_col])
            df = df.sort_values("date").reset_index(drop=True)

            # Only keep necessary columns
            df = df[["date", "open", "high", "low", "close", "volume"]].copy()

            symbol = csv_file.stem
            if symbol and symbol != "":
                data[symbol] = df

        except Exception as e:
            continue

    return data


def calc_kama(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate KAMA (Kaufman Adaptive Moving Average)"""
    n = len(prices)
    kama = np.full(n, np.nan)

    if n < period + 1:
        return kama

    # Initial value
    kama[period - 1] = np.mean(prices[:period])

    fast = 2 / (2 + 1)
    slow = 2 / (30 + 1)

    for i in range(period, n):
        change = abs(prices[i] - prices[i - period])
        volatility = sum(
            abs(prices[j] - prices[j - 1]) for j in range(i - period + 1, i + 1)
        )
        er = change / volatility if volatility > 0 else 0
        sc = (er * (fast - slow) + slow) ** 2
        kama[i] = kama[i - 1] + sc * (prices[i] - kama[i - 1])

    return kama


def calc_ma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average"""
    result = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1 : i + 1])
    return result


def run_backtest(
    signal_data: dict,
    common_dates: list,
    initial_capital: float = 10000,
    max_positions: int = 20,
) -> dict:
    """Run backtest with given signals"""
    portfolio_values = []
    daily_returns = []
    position_counts = []

    current_value = initial_capital
    prev_positions = {}

    for i, date in enumerate(common_dates):
        # Get today's signals and prices
        signals_today = {}
        prices_today = {}

        for symbol, df in signal_data.items():
            if date in df.index:
                signals_today[symbol] = df.loc[date, "final_signal"]
                prices_today[symbol] = df.loc[date, "close"]

        # Active signals
        active = [s for s, sig in signals_today.items() if sig]
        selected = active[:max_positions]
        position_counts.append(len(selected))

        # Calculate daily P&L
        if i > 0 and len(prev_positions) > 0:
            daily_pnl = 0
            for symbol, (weight, prev_price) in prev_positions.items():
                if symbol in prices_today:
                    curr_price = prices_today[symbol]
                    ret = (curr_price - prev_price) / prev_price
                    daily_pnl += current_value * weight * ret

            current_value += daily_pnl
            if portfolio_values:
                daily_ret = daily_pnl / portfolio_values[-1]
            else:
                daily_ret = 0
            daily_returns.append(daily_ret)
        else:
            daily_returns.append(0)

        portfolio_values.append(current_value)

        # Set positions for next day
        if len(selected) > 0:
            weight = 1.0 / len(selected)
            prev_positions = {
                s: (weight, prices_today[s]) for s in selected if s in prices_today
            }
        else:
            prev_positions = {}

    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    daily_returns = np.array(daily_returns)

    total_return = (portfolio_values[-1] - initial_capital) / initial_capital

    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0

    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)

    n_days = len(portfolio_values)
    cagr = (
        (portfolio_values[-1] / initial_capital) ** (365 / n_days) - 1
        if n_days > 0
        else 0
    )

    return {
        "sharpe": sharpe,
        "mdd": max_drawdown,
        "total_return": total_return,
        "cagr": cagr,
        "avg_positions": np.mean(position_counts),
        "final_value": portfolio_values[-1],
        "n_days": n_days,
    }


def run_strategy_on_market(price_data: dict, strategy: dict, market_name: str) -> dict:
    """Run a strategy on a single market's data"""
    kama_period = strategy["kama"]
    tsmom_period = strategy["tsmom"]
    gate_period = strategy["gate"]

    # Find BTC for gate signal
    btc_key = None
    for key in price_data.keys():
        if key.upper() == "BTC" or key.upper() == "BTCUSDT":
            btc_key = key
            break

    # If not found, try partial match
    if btc_key is None:
        for key in price_data.keys():
            if (
                "BTC" in key.upper()
                and "DOWN" not in key.upper()
                and "UP" not in key.upper()
            ):
                btc_key = key
                break

    if btc_key is None:
        print(f"  [WARN] No BTC found in {market_name}")
        return None

    btc_df = price_data[btc_key].copy()
    btc_prices = btc_df["close"].values
    btc_ma = calc_ma(btc_prices, gate_period)
    btc_gate = btc_prices > btc_ma
    btc_df["gate"] = btc_gate
    btc_df = btc_df.set_index("date")

    # Calculate signals for each symbol
    signal_data = {}

    # Track signal statistics
    entry_count = 0
    gate_filtered = 0

    for symbol, df in price_data.items():
        df = df.copy()
        prices = df["close"].values
        n = len(prices)

        if n < max(kama_period, tsmom_period, gate_period) + 10:
            continue

        # KAMA signal
        kama = calc_kama(prices, kama_period)
        kama_signal = prices > kama

        # TSMOM signal
        tsmom_signal = np.zeros(n, dtype=bool)
        for i in range(tsmom_period, n):
            tsmom_signal[i] = prices[i] > prices[i - tsmom_period]

        # Entry signal (KAMA OR TSMOM)
        entry_signal = kama_signal | tsmom_signal

        df["entry_signal"] = entry_signal
        df = df.set_index("date")

        # Merge with gate signal
        df = df.join(btc_df[["gate"]], how="left")
        df["gate"] = df["gate"].fillna(False)
        df["final_signal"] = df["gate"] & df["entry_signal"]

        # Filter to 2025
        df_2025 = df[(df.index >= HOLDOUT_START) & (df.index <= HOLDOUT_END)]

        if len(df_2025) > 0:
            signal_data[symbol] = df_2025
            entry_count += df_2025["entry_signal"].sum()
            gate_filtered += (df_2025["entry_signal"] & ~df_2025["gate"]).sum()

    if len(signal_data) == 0:
        return None

    # Get common dates
    all_dates = set()
    for symbol, df in signal_data.items():
        all_dates.update(df.index.tolist())
    common_dates = sorted(list(all_dates))

    if len(common_dates) < 10:
        return None

    # Run backtest
    result = run_backtest(signal_data, common_dates, INITIAL_CAPITAL, MAX_POSITIONS)
    result["n_symbols"] = len(signal_data)
    result["period"] = (
        f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}"
    )
    result["entry_signals"] = entry_count
    result["gate_filtered"] = gate_filtered
    result["btc_key"] = btc_key

    return result


# ============================================================
# Main Execution
# ============================================================
print("\n[1] Loading Data")
print("-" * 60)

market_data = {}
for market_name, folder_name in MARKETS.items():
    folder_path = DATA_ROOT / folder_name
    if folder_path.exists():
        data = load_market_data(folder_path)
        market_data[market_name] = data
        print(f"  {market_name}: {len(data)} symbols loaded")
    else:
        print(f"  {market_name}: folder not found")

print("\n[2] Running Backtests")
print("-" * 60)

results = []

for strategy in STRATEGIES:
    print(f"\nStrategy: {strategy['name']}")

    for market_name in MARKETS.keys():
        if market_name not in market_data:
            continue

        data = market_data[market_name]
        result = run_strategy_on_market(data, strategy, market_name)

        if result:
            results.append(
                {
                    "strategy": strategy["name"],
                    "market": market_name,
                    "sharpe": result["sharpe"],
                    "mdd": result["mdd"],
                    "return": result["total_return"],
                    "cagr": result["cagr"],
                    "avg_pos": result["avg_positions"],
                    "n_symbols": result["n_symbols"],
                    "n_days": result["n_days"],
                    "period": result["period"],
                }
            )
            print(
                f"  {market_name}: Sharpe={result['sharpe']:.3f}, MDD={result['mdd']*100:.1f}%, Return={result['total_return']*100:.1f}%, AvgPos={result['avg_positions']:.1f}"
            )
        else:
            print(f"  {market_name}: No valid data")

# ============================================================
# Results Summary
# ============================================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)

if len(results_df) > 0:
    # Full comparison table
    print(
        f"\n{'Strategy':<20} {'Market':<10} {'Sharpe':>8} {'MDD':>8} {'Return':>10} {'AvgPos':>8}"
    )
    print("-" * 70)

    for _, row in results_df.iterrows():
        print(
            f"{row['strategy']:<20} {row['market']:<10} {row['sharpe']:>8.3f} {row['mdd']*100:>7.1f}% {row['return']*100:>9.1f}% {row['avg_pos']:>8.1f}"
        )

    # Cross-market average
    print("\n" + "-" * 70)
    print("CROSS-MARKET AVERAGE")
    print("-" * 70)

    for strategy_name in results_df["strategy"].unique():
        strategy_results = results_df[results_df["strategy"] == strategy_name]
        avg_sharpe = strategy_results["sharpe"].mean()
        avg_mdd = strategy_results["mdd"].mean()
        avg_return = strategy_results["return"].mean()

        print(
            f"{strategy_name}: Sharpe={avg_sharpe:.3f}, MDD={avg_mdd*100:.1f}%, Return={avg_return*100:.1f}%"
        )

    # Save results
    output_path = "E:/data/holdout_2025_comparison_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

# ============================================================
# Compare with Known Results
# ============================================================
print("\n" + "=" * 80)
print("COMPARISON WITH PREVIOUS HOLDOUT RESULTS")
print("=" * 80)

print("""
Known Previous Results (from optimization reports):
---------------------------------------------------
KAMA=10, TSMOM=60:
  - Upbit 2025:   Sharpe 3.162, MDD -17.3%
  - Bithumb 2025: Sharpe 2.581, MDD -18.9%
  - Binance 2025: (not tested)

KAMA=5, TSMOM=90:
  - Upbit 2025:   Sharpe 2.350, MDD -21.8%
  - Bithumb 2025: Sharpe 1.810, MDD -26.2%
  - Binance 2025: Sharpe 2.540, MDD -18.4%
""")

print("\nCurrent Test Results:")
print("-" * 60)
for _, row in results_df.iterrows():
    print(
        f"  {row['strategy']:<20} {row['market']:<10}: Sharpe={row['sharpe']:.3f}, MDD={row['mdd']*100:.1f}%"
    )

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Check if results differ significantly
print("""
If results differ significantly from previous tests:
1. Check if BTC gate is working correctly
2. Check symbol universe differences
3. Check date alignment issues
4. Check if KAMA/TSMOM calculation matches original
""")

print("=" * 80)
