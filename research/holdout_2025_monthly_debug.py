"""
Monthly breakdown debug for 2025 holdout test
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

DATA_ROOT = Path("E:/data/crypto_ohlcv")


def load_market_data(market_folder: Path) -> dict:
    data = {}
    for csv_file in market_folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            date_col = [
                c for c in df.columns if "date" in c.lower() or "time" in c.lower()
            ][0]
            df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
            df = df.sort_values("date").reset_index(drop=True)
            df = df.drop_duplicates(subset=["date"], keep="last")
            df = df[["date", "open", "high", "low", "close", "volume"]].copy()
            symbol = csv_file.stem
            if symbol:
                data[symbol] = df
        except:
            continue
    return data


def calc_kama(prices, period):
    n = len(prices)
    kama = np.full(n, np.nan)
    if n < period + 1:
        return kama
    kama[period - 1] = np.mean(prices[:period])
    fast, slow = 2 / (2 + 1), 2 / (30 + 1)
    for i in range(period, n):
        change = abs(prices[i] - prices[i - period])
        volatility = sum(
            abs(prices[j] - prices[j - 1]) for j in range(i - period + 1, i + 1)
        )
        er = change / volatility if volatility > 0 else 0
        sc = (er * (fast - slow) + slow) ** 2
        kama[i] = kama[i - 1] + sc * (prices[i] - kama[i - 1])
    return kama


def calc_ma(prices, period):
    result = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1 : i + 1])
    return result


print("=" * 70)
print("MONTHLY BREAKDOWN DEBUG - Upbit KAMA=5, TSMOM=90")
print("=" * 70)

# Load Upbit data
upbit_data = load_market_data(DATA_ROOT / "upbit_1d")
print(f"Loaded {len(upbit_data)} symbols")

# BTC Gate
btc_df = upbit_data["BTC"].copy()
btc_prices = btc_df["close"].values
btc_ma30 = calc_ma(btc_prices, 30)
btc_df["gate"] = btc_prices > btc_ma30
btc_df = btc_df.set_index("date")

# Process all symbols
KAMA_PERIOD, TSMOM_PERIOD = 5, 90
signal_data = {}

for symbol, df in upbit_data.items():
    df = df.copy()
    prices = df["close"].values
    n = len(prices)
    if n < 100:
        continue

    kama = calc_kama(prices, KAMA_PERIOD)
    kama_signal = prices > kama
    tsmom_signal = np.zeros(n, dtype=bool)
    for i in range(TSMOM_PERIOD, n):
        tsmom_signal[i] = prices[i] > prices[i - TSMOM_PERIOD]

    df["entry"] = kama_signal | tsmom_signal
    df = df.set_index("date")
    df = df.join(btc_df[["gate"]], how="left")
    df["gate"] = df["gate"].fillna(False).astype(bool)
    df["final"] = df["gate"] & df["entry"]

    df_2025 = df[(df.index >= "2025-01-01") & (df.index <= "2025-12-31")]
    if len(df_2025) > 0:
        signal_data[symbol] = df_2025

print(f"Symbols with 2025 data: {len(signal_data)}")

# Get all dates
all_dates = set()
for df in signal_data.values():
    all_dates.update(df.index.tolist())
common_dates = sorted(list(all_dates))

# Run backtest with monthly tracking
monthly_returns = {}
daily_returns = []
portfolio_value = 10000
prev_positions = {}

for i, date in enumerate(common_dates):
    month = date.strftime("%Y-%m")
    if month not in monthly_returns:
        monthly_returns[month] = []

    signals_today = {}
    prices_today = {}

    for symbol, df in signal_data.items():
        if date in df.index:
            signals_today[symbol] = df.loc[date, "final"]
            prices_today[symbol] = df.loc[date, "close"]

    active = [s for s, sig in signals_today.items() if sig]
    selected = active[:20]

    # Calculate daily P&L
    if i > 0 and len(prev_positions) > 0:
        daily_pnl = 0
        for symbol, (weight, prev_price) in prev_positions.items():
            if symbol in prices_today:
                curr_price = prices_today[symbol]
                ret = (curr_price - prev_price) / prev_price
                daily_pnl += portfolio_value * weight * ret

        daily_ret = daily_pnl / portfolio_value
        portfolio_value += daily_pnl
    else:
        daily_ret = 0

    daily_returns.append(daily_ret)
    monthly_returns[month].append(daily_ret)

    # Set next positions
    if len(selected) > 0:
        weight = 1.0 / len(selected)
        prev_positions = {
            s: (weight, prices_today[s]) for s in selected if s in prices_today
        }
    else:
        prev_positions = {}

# Calculate monthly summaries
print("\n" + "=" * 70)
print("MONTHLY RETURNS COMPARISON")
print("=" * 70)
print(f"{'Month':<10} {'My Test':>12} {'Original':>12} {'Diff':>10}")
print("-" * 50)

original_monthly = {
    "2025-01": 5.3,
    "2025-02": -11.4,
    "2025-03": 8.0,
    "2025-04": 20.1,
    "2025-05": 37.7,
    "2025-06": 10.4,
    "2025-07": 22.9,
    "2025-08": 9.0,
    "2025-09": 2.0,
    "2025-10": -8.1,
    "2025-11": 0.0,
    "2025-12": 1.1,
}

for month in sorted(monthly_returns.keys()):
    rets = monthly_returns[month]
    my_ret = ((1 + np.array(rets)).prod() - 1) * 100
    orig_ret = original_monthly.get(month, 0)
    diff = my_ret - orig_ret
    print(f"{month:<10} {my_ret:>11.1f}% {orig_ret:>11.1f}% {diff:>9.1f}%")

total_return = (portfolio_value - 10000) / 10000 * 100
print("-" * 50)
print(f"{'TOTAL':<10} {total_return:>11.1f}% {133.3:>11.1f}%")

print("\n" + "=" * 70)
