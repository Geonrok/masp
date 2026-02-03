"""
2025 Holdout Test - Original Methodology
KAMA=10, TSMOM=60 vs KAMA=5, TSMOM=90

Original methodology:
- Signal checked at CLOSE of Day N
- If signal is HOLD/BUY: attribute Day N return (from Day N-1 close to Day N close)
- If signal is EXIT: 0% return for Day N (exit at previous day's close)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("2025 HOLDOUT TEST - ORIGINAL METHODOLOGY")
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
    """Load all CSV files, normalize dates"""
    data = {}
    csv_files = list(market_folder.glob("*.csv"))

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            date_col = None
            for col in df.columns:
                if "timestamp" in col.lower() or "date" in col.lower():
                    date_col = col
                    break

            if date_col is None or "close" not in df.columns:
                continue

            df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
            df = df.sort_values("date").reset_index(drop=True)
            df = df.drop_duplicates(subset=["date"], keep="last")
            df = df[["date", "open", "high", "low", "close", "volume"]].copy()

            symbol = csv_file.stem
            if symbol and symbol != "":
                data[symbol] = df

        except Exception as e:
            continue

    return data


def calc_kama(prices: np.ndarray, period: int) -> np.ndarray:
    n = len(prices)
    kama = np.full(n, np.nan)

    if n < period + 1:
        return kama

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
    result = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1 : i + 1])
    return result


def run_backtest_original_method(
    signal_data: dict,
    common_dates: list,
    initial_capital: float = 10000,
    max_positions: int = 20,
) -> dict:
    """
    Original methodology backtest:
    - Check signal at Day N close
    - If signal is HOLD: earn Day N return
    - If signal changes to EXIT: 0% return for Day N
    """
    portfolio_values = []
    daily_returns = []
    position_counts = []
    exposure_days = 0

    current_value = initial_capital
    was_invested = False  # Were we invested at previous close?

    for i, date in enumerate(common_dates):
        # Get today's signals and prices
        signals_today = {}
        prices_today = {}
        prices_yesterday = {}

        for symbol, df in signal_data.items():
            if date in df.index:
                signals_today[symbol] = df.loc[date, "final_signal"]
                prices_today[symbol] = df.loc[date, "close"]

                # Get yesterday's price for return calculation
                idx = df.index.get_loc(date)
                if idx > 0:
                    prev_date = df.index[idx - 1]
                    prices_yesterday[symbol] = df.loc[prev_date, "close"]

        # Active signals TODAY (at today's close)
        active_today = [s for s, sig in signals_today.items() if sig]
        selected_today = active_today[:max_positions]

        # ORIGINAL METHODOLOGY:
        # - Return is earned only if we were invested AND signal is still HOLD
        # - If signal turns to EXIT at today's close, we get 0% return today

        if i == 0:
            # First day - no previous position
            daily_ret = 0
            position_counts.append(len(selected_today))
        else:
            if was_invested and len(selected_today) > 0:
                # We were invested and signal says HOLD -> earn today's return
                daily_pnl = 0
                valid_positions = 0

                for symbol in prev_selected:
                    if symbol in prices_today and symbol in prices_yesterday:
                        prev_price = prices_yesterday[symbol]
                        curr_price = prices_today[symbol]
                        if prev_price > 0:
                            ret = (curr_price - prev_price) / prev_price
                            daily_pnl += ret
                            valid_positions += 1

                if valid_positions > 0:
                    avg_ret = daily_pnl / valid_positions
                    daily_ret = avg_ret
                    exposure_days += 1
                else:
                    daily_ret = 0
            elif was_invested and len(selected_today) == 0:
                # We were invested but signal says EXIT -> 0% return (exit at prev close)
                daily_ret = 0
            else:
                # We were not invested
                daily_ret = 0

            position_counts.append(len(selected_today))

        current_value = current_value * (1 + daily_ret)
        daily_returns.append(daily_ret)
        portfolio_values.append(current_value)

        # Update state for next day
        was_invested = len(selected_today) > 0
        prev_selected = selected_today.copy() if selected_today else []

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

    return {
        "sharpe": sharpe,
        "mdd": max_drawdown,
        "total_return": total_return,
        "avg_positions": np.mean(position_counts),
        "final_value": portfolio_values[-1],
        "n_days": n_days,
        "exposure_days": exposure_days,
    }


def run_strategy_on_market(price_data: dict, strategy: dict, market_name: str) -> dict:
    kama_period = strategy["kama"]
    tsmom_period = strategy["tsmom"]
    gate_period = strategy["gate"]

    # Find BTC
    btc_key = None
    for key in price_data.keys():
        if key.upper() == "BTC" or key.upper() == "BTCUSDT":
            btc_key = key
            break
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
        return None

    btc_df = price_data[btc_key].copy()
    btc_prices = btc_df["close"].values
    btc_ma = calc_ma(btc_prices, gate_period)
    btc_gate = btc_prices > btc_ma
    btc_df["gate"] = btc_gate
    btc_df = btc_df.set_index("date")

    signal_data = {}

    for symbol, df in price_data.items():
        df = df.copy()
        prices = df["close"].values
        n = len(prices)

        if n < max(kama_period, tsmom_period, gate_period) + 10:
            continue

        kama = calc_kama(prices, kama_period)
        kama_signal = prices > kama

        tsmom_signal = np.zeros(n, dtype=bool)
        for i in range(tsmom_period, n):
            tsmom_signal[i] = prices[i] > prices[i - tsmom_period]

        entry_signal = kama_signal | tsmom_signal

        df["entry_signal"] = entry_signal
        df = df.set_index("date")

        df = df.join(btc_df[["gate"]], how="left")
        df["gate"] = df["gate"].fillna(False).astype(bool)
        df["final_signal"] = df["gate"] & df["entry_signal"]

        df_2025 = df[(df.index >= HOLDOUT_START) & (df.index <= HOLDOUT_END)]

        if len(df_2025) > 0:
            signal_data[symbol] = df_2025

    if len(signal_data) == 0:
        return None

    all_dates = set()
    for symbol, df in signal_data.items():
        all_dates.update(df.index.tolist())
    common_dates = sorted(list(all_dates))

    if len(common_dates) < 10:
        return None

    result = run_backtest_original_method(
        signal_data, common_dates, INITIAL_CAPITAL, MAX_POSITIONS
    )
    result["n_symbols"] = len(signal_data)
    result["period"] = (
        f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}"
    )

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

print("\n[2] Running Backtests (Original Methodology)")
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
                    "avg_pos": result["avg_positions"],
                    "n_symbols": result["n_symbols"],
                    "n_days": result["n_days"],
                    "exposure_days": result["exposure_days"],
                }
            )
            print(
                f"  {market_name}: Sharpe={result['sharpe']:.3f}, MDD={result['mdd']*100:.1f}%, Return={result['total_return']*100:.1f}%, Exposure={result['exposure_days']}d"
            )

# ============================================================
# Results Summary
# ============================================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY (Original Methodology)")
print("=" * 80)

results_df = pd.DataFrame(results)

if len(results_df) > 0:
    print(
        f"\n{'Strategy':<20} {'Market':<10} {'Sharpe':>8} {'MDD':>8} {'Return':>10} {'AvgPos':>8}"
    )
    print("-" * 75)

    for _, row in results_df.iterrows():
        print(
            f"{row['strategy']:<20} {row['market']:<10} {row['sharpe']:>8.3f} {row['mdd']*100:>7.1f}% {row['return']*100:>9.1f}% {row['avg_pos']:>8.1f}"
        )

    print("\n" + "-" * 75)
    print("CROSS-MARKET AVERAGE")
    print("-" * 75)

    for strategy_name in results_df["strategy"].unique():
        strategy_results = results_df[results_df["strategy"] == strategy_name]
        avg_sharpe = strategy_results["sharpe"].mean()
        avg_mdd = strategy_results["mdd"].mean()
        avg_return = strategy_results["return"].mean()

        print(
            f"{strategy_name}: Sharpe={avg_sharpe:.3f}, MDD={avg_mdd*100:.1f}%, Return={avg_return*100:.1f}%"
        )

    # Compare with original report
    print("\n" + "=" * 80)
    print("COMPARISON WITH ORIGINAL REPORT")
    print("=" * 80)

    print("""
Original Report (KAMA=5, TSMOM=90):
  Upbit:   Sharpe 2.35, MDD -21.8%, Return +133%
  Bithumb: Sharpe 1.81, MDD -26.2%, Return +109%
  Binance: Sharpe 2.54, MDD -18.4%, Return +176%
""")

    print("Current Test Results:")
    for _, row in results_df[results_df["strategy"] == "KAMA=5, TSMOM=90"].iterrows():
        print(
            f"  {row['market']}: Sharpe {row['sharpe']:.2f}, MDD {row['mdd']*100:.1f}%, Return {row['return']*100:.0f}%"
        )

    output_path = "E:/data/holdout_2025_original_methodology_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

print("\n" + "=" * 80)
