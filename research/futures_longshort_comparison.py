"""
Futures Long/Short Strategy vs Spot Long-Only Comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

DATA_ROOT = Path("E:/data/crypto_ohlcv")
SPOT_FOLDER = DATA_ROOT / "binance_spot_1d"
FUTURES_FOLDER = DATA_ROOT / "binance_futures_1d"


def load_market(folder, start, end):
    data = {}
    for csv_file in folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            date_col = [
                c for c in df.columns if "date" in c.lower() or "time" in c.lower()
            ][0]
            df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
            df = df.sort_values("date").drop_duplicates("date", keep="last")
            df = df[["date", "close", "volume"]].set_index("date")
            if len(df[(df.index >= start) & (df.index <= end)]) >= 30:
                data[csv_file.stem] = df
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


def run_long_only(data, start, end):
    """현물: 롱 온리 전략"""
    btc_key = None
    for k in data.keys():
        if "BTC" in k.upper() and "DOWN" not in k.upper() and "UP" not in k.upper():
            btc_key = k
            break
    if not btc_key:
        return None

    btc_df = data[btc_key]
    btc_prices = btc_df["close"].values
    btc_ma30 = calc_ma(btc_prices, 30)
    btc_gate = pd.Series(btc_prices > btc_ma30, index=btc_df.index)

    signal_data = {}
    for symbol, df in data.items():
        prices = df["close"].values
        n = len(prices)
        if n < 100:
            continue
        kama = calc_kama(prices, 5)
        kama_sig = prices > kama
        tsmom_sig = np.array(
            [prices[i] > prices[i - 90] if i >= 90 else False for i in range(n)]
        )
        df = df.copy()
        df["long_signal"] = kama_sig | tsmom_sig
        df = df.join(pd.DataFrame({"gate": btc_gate}), how="left")
        df["gate"] = df["gate"].fillna(False)
        df["final_signal"] = df["gate"] & df["long_signal"]
        df_period = df[(df.index >= start) & (df.index <= end)]
        if len(df_period) > 0:
            signal_data[symbol] = df_period

    if not signal_data:
        return None
    all_dates = sorted(set().union(*[df.index.tolist() for df in signal_data.values()]))
    dates = [d for d in all_dates if start <= d <= end]

    portfolio = 10000
    daily_rets = []
    prev_positions = {}

    for i, date in enumerate(dates):
        prices_today, signals_today, volumes_today = {}, {}, {}
        for symbol, df in signal_data.items():
            if date in df.index:
                signals_today[symbol] = df.loc[date, "final_signal"]
                prices_today[symbol] = df.loc[date, "close"]
                volumes_today[symbol] = df.loc[date, "volume"] * df.loc[date, "close"]

        if prev_positions:
            pnl = sum(
                portfolio * w * direction * (prices_today[sym] - prev_p) / prev_p
                for sym, (w, prev_p, direction) in prev_positions.items()
                if sym in prices_today
            )
            portfolio += pnl
            daily_rets.append(pnl / (portfolio - pnl) if portfolio != pnl else 0)
        else:
            daily_rets.append(0)

        # Long only
        active = [
            (s, volumes_today.get(s, 0)) for s, sig in signals_today.items() if sig
        ]
        selected = [s for s, v in sorted(active, key=lambda x: x[1], reverse=True)[:20]]

        if selected:
            w = 1.0 / len(selected)
            prev_positions = {
                s: (w, prices_today[s], 1) for s in selected if s in prices_today
            }  # 1 = long
        else:
            prev_positions = {}

    daily_rets = np.array(daily_rets)
    total_ret = (portfolio - 10000) / 10000
    sharpe = (
        np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
        if np.std(daily_rets) > 0
        else 0
    )
    pv = np.cumprod(1 + daily_rets) * 10000
    mdd = np.min((pv - np.maximum.accumulate(pv)) / np.maximum.accumulate(pv))

    return {"return": total_ret, "sharpe": sharpe, "mdd": mdd}


def run_long_short(data, start, end):
    """선물: 롱/숏 양방향 전략"""
    btc_key = None
    for k in data.keys():
        if "BTC" in k.upper() and "DOWN" not in k.upper() and "UP" not in k.upper():
            btc_key = k
            break
    if not btc_key:
        return None

    btc_df = data[btc_key]
    btc_prices = btc_df["close"].values
    btc_ma30 = calc_ma(btc_prices, 30)
    btc_above_ma = pd.Series(btc_prices > btc_ma30, index=btc_df.index)

    signal_data = {}
    for symbol, df in data.items():
        prices = df["close"].values
        n = len(prices)
        if n < 100:
            continue

        kama = calc_kama(prices, 5)
        kama_long = prices > kama
        kama_short = prices < kama
        tsmom_long = np.array(
            [prices[i] > prices[i - 90] if i >= 90 else False for i in range(n)]
        )
        tsmom_short = np.array(
            [prices[i] < prices[i - 90] if i >= 90 else False for i in range(n)]
        )

        df = df.copy()
        df["long_signal"] = kama_long | tsmom_long
        df["short_signal"] = kama_short & tsmom_short  # 둘 다 하락일 때만 숏
        df = df.join(pd.DataFrame({"btc_above": btc_above_ma}), how="left")
        df["btc_above"] = df["btc_above"].fillna(False)

        # BTC > MA30: Long 허용, BTC < MA30: Short 허용
        df["final_long"] = df["btc_above"] & df["long_signal"]
        df["final_short"] = (~df["btc_above"]) & df["short_signal"]

        df_period = df[(df.index >= start) & (df.index <= end)]
        if len(df_period) > 0:
            signal_data[symbol] = df_period

    if not signal_data:
        return None
    all_dates = sorted(set().union(*[df.index.tolist() for df in signal_data.values()]))
    dates = [d for d in all_dates if start <= d <= end]

    portfolio = 10000
    daily_rets = []
    prev_positions = {}
    long_days = 0
    short_days = 0

    for i, date in enumerate(dates):
        prices_today, volumes_today = {}, {}
        longs_today, shorts_today = [], []

        for symbol, df in signal_data.items():
            if date in df.index:
                prices_today[symbol] = df.loc[date, "close"]
                volumes_today[symbol] = df.loc[date, "volume"] * df.loc[date, "close"]
                if df.loc[date, "final_long"]:
                    longs_today.append((symbol, volumes_today[symbol]))
                elif df.loc[date, "final_short"]:
                    shorts_today.append((symbol, volumes_today[symbol]))

        if prev_positions:
            pnl = 0
            for sym, (w, prev_p, direction) in prev_positions.items():
                if sym in prices_today:
                    ret = (prices_today[sym] - prev_p) / prev_p
                    pnl += portfolio * w * direction * ret
            portfolio += pnl
            daily_rets.append(pnl / (portfolio - pnl) if portfolio != pnl else 0)
        else:
            daily_rets.append(0)

        # Select top 20 by volume (longs or shorts)
        if longs_today:
            selected = [
                s for s, v in sorted(longs_today, key=lambda x: x[1], reverse=True)[:20]
            ]
            direction = 1
            long_days += 1
        elif shorts_today:
            selected = [
                s
                for s, v in sorted(shorts_today, key=lambda x: x[1], reverse=True)[:20]
            ]
            direction = -1
            short_days += 1
        else:
            selected = []
            direction = 0

        if selected:
            w = 1.0 / len(selected)
            prev_positions = {
                s: (w, prices_today[s], direction)
                for s in selected
                if s in prices_today
            }
        else:
            prev_positions = {}

    daily_rets = np.array(daily_rets)
    total_ret = (portfolio - 10000) / 10000
    sharpe = (
        np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
        if np.std(daily_rets) > 0
        else 0
    )
    pv = np.cumprod(1 + daily_rets) * 10000
    mdd = np.min((pv - np.maximum.accumulate(pv)) / np.maximum.accumulate(pv))

    return {
        "return": total_ret,
        "sharpe": sharpe,
        "mdd": mdd,
        "long_days": long_days,
        "short_days": short_days,
    }


# Test periods
periods = [
    ("2021 Bull", "2021-01-01", "2021-12-31"),
    ("2022 Bear", "2022-01-01", "2022-12-31"),
    ("2023 Recovery", "2023-01-01", "2023-12-31"),
    ("2024 Bull", "2024-01-01", "2024-12-31"),
    ("2025 Current", "2025-01-01", "2025-12-31"),
]

print("=" * 85)
print("SPOT LONG-ONLY vs FUTURES LONG/SHORT COMPARISON")
print("=" * 85)
print()
print("Strategy: KAMA=5, TSMOM=90, BTC Gate")
print("  - Long-Only: BTC > MA30 → Long signals only")
print("  - Long/Short: BTC > MA30 → Long, BTC < MA30 → Short")
print()
print("=" * 85)
print(
    f"{'Period':<15} {'Spot L-Only':>14} {'Futures L/S':>14} {'Diff':>10} {'L-days':>8} {'S-days':>8}"
)
print("-" * 85)

for name, start, end in periods:
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    spot_data = load_market(SPOT_FOLDER, start_dt, end_dt)
    futures_data = load_market(FUTURES_FOLDER, start_dt, end_dt)

    if not spot_data or not futures_data:
        continue

    r_spot = run_long_only(spot_data, start_dt, end_dt)
    r_futures = run_long_short(futures_data, start_dt, end_dt)

    if r_spot and r_futures:
        spot_ret = r_spot["return"] * 100
        futures_ret = r_futures["return"] * 100
        diff = futures_ret - spot_ret
        l_days = r_futures.get("long_days", 0)
        s_days = r_futures.get("short_days", 0)
        print(
            f"{name:<15} {spot_ret:>+13.1f}% {futures_ret:>+13.1f}% {diff:>+9.1f}%p {l_days:>8} {s_days:>8}"
        )

print()
print("=" * 85)
print("DETAILED 2025 ANALYSIS")
print("=" * 85)

start_dt = pd.Timestamp("2025-01-01")
end_dt = pd.Timestamp("2025-12-31")

futures_data = load_market(FUTURES_FOLDER, start_dt, end_dt)
r = run_long_short(futures_data, start_dt, end_dt)

if r:
    print(f"\n2025 Futures Long/Short:")
    print(f"  Total Return: {r['return']*100:+.1f}%")
    print(f"  Sharpe Ratio: {r['sharpe']:.3f}")
    print(f"  Max Drawdown: {r['mdd']*100:.1f}%")
    print(f"  Long Days:    {r['long_days']}")
    print(f"  Short Days:   {r['short_days']}")
    print(f"\n  Short 포지션이 하락장 손실을 상쇄했는가?")

    spot_data = load_market(SPOT_FOLDER, start_dt, end_dt)
    r_spot = run_long_only(spot_data, start_dt, end_dt)
    if r_spot:
        improvement = (r["return"] - r_spot["return"]) * 100
        print(
            f"  → Spot 대비 {improvement:+.1f}%p {'개선' if improvement > 0 else '악화'}"
        )

print()
print("=" * 85)
