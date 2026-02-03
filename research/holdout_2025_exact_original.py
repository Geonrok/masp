"""
2025 Holdout Test - EXACT Original Methodology

Original methodology discovered:
1. On EXIT days: 0% return (positions sold at previous close conceptually)
2. On ENTRY days: Full day's return counted (entered at previous close)
3. On HOLD days: Normal return

This creates look-ahead bias:
- Avoids losses on exit days
- Captures gains on entry days
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 80)
print("2025 HOLDOUT TEST - EXACT ORIGINAL METHODOLOGY")
print("=" * 80)

ORIGINAL_SYMBOLS = [
    "AAVE",
    "ADA",
    "AERGO",
    "AGLD",
    "AHT",
    "AKT",
    "ALGO",
    "ALT",
    "ANIME",
    "API3",
    "AQT",
    "ARB",
    "ARDR",
    "ARK",
    "ATH",
    "AUCTION",
    "AVAX",
    "AVNT",
    "AWE",
    "BARD",
    "BAT",
    "BCH",
    "BEAM",
    "BERA",
    "BIGTIME",
    "BLAST",
    "BONK",
    "BORA",
    "BOUNTY",
    "BSV",
    "BTC",
    "BTT",
    "CARV",
    "CBK",
    "CELO",
    "CHZ",
    "CKB",
    "COW",
    "CRO",
    "CTC",
    "CVC",
    "CYBER",
    "DEEP",
    "DOGE",
    "DOOD",
    "DRIFT",
    "ENA",
    "ENS",
    "ETC",
    "ETH",
    "FLUID",
    "GAME2",
    "GAS",
    "GLM",
    "GMT",
    "GRS",
    "HBAR",
    "HIVE",
    "HP",
    "HYPER",
    "ICX",
    "IMX",
    "IOST",
    "IP",
    "JST",
    "JTO",
    "JUP",
    "KAITO",
    "KAVA",
    "KNC",
    "LA",
    "LAYER",
    "LINEA",
    "LINK",
    "LPT",
    "MASK",
    "ME",
    "MEW",
    "MIRA",
    "MNT",
    "MOC",
    "MOCA",
    "MOODENG",
    "MOVE",
    "NEAR",
    "NEO",
    "NOM",
    "NXPC",
    "ONDO",
    "OPEN",
    "ORBS",
    "ORCA",
    "PENDLE",
    "PENGU",
    "PEPE",
    "POKT",
    "POL",
    "POLYX",
    "PROVE",
    "PUNDIX",
    "PYTH",
    "QTUM",
    "RAY",
    "RENDER",
    "SAFE",
    "SAHARA",
    "SAND",
    "SEI",
    "SHIB",
    "SIGN",
    "SNT",
    "SOL",
    "SONIC",
    "SOPH",
    "STEEM",
    "STG",
    "STRAX",
    "STX",
    "SUI",
    "TOKAMAK",
    "TRUMP",
    "TRX",
    "UNI",
    "USD1",
    "USDC",
    "USDT",
    "VANA",
    "VET",
    "VIRTUAL",
    "VTHO",
    "W",
    "WAL",
    "WAVES",
    "WCT",
    "WLFI",
    "XLM",
    "XRP",
    "XTZ",
    "ZETA",
    "ZKC",
    "ZORA",
    "ZRO",
]

DATA_ROOT = Path("E:/data/crypto_ohlcv")
HOLDOUT_START = pd.Timestamp("2025-01-01")
HOLDOUT_END = pd.Timestamp("2025-12-31")
MAX_POSITIONS = 20


def load_data(folder: Path, symbols: list) -> dict:
    data = {}
    for csv_file in folder.glob("*.csv"):
        symbol = csv_file.stem
        if symbol not in symbols:
            continue
        try:
            df = pd.read_csv(csv_file)
            date_col = [
                c for c in df.columns if "date" in c.lower() or "time" in c.lower()
            ][0]
            df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
            df = df.sort_values("date").reset_index(drop=True)
            df = df.drop_duplicates(subset=["date"], keep="last")
            data[symbol] = df[["date", "close", "volume"]].set_index("date")
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


def generate_signals(
    data: dict, kama_period: int, tsmom_period: int, gate_period: int
) -> dict:
    """Generate entry signals for each symbol"""
    btc = data.get("BTC")
    if btc is None:
        return {}

    btc_prices = btc["close"].values
    btc_ma = calc_ma(btc_prices, gate_period)
    btc_gate = pd.Series(btc_prices > btc_ma, index=btc.index)

    signals = {}
    for symbol, df in data.items():
        prices = df["close"].values
        n = len(prices)
        if n < max(kama_period, tsmom_period, gate_period) + 10:
            continue

        kama = calc_kama(prices, kama_period)
        kama_sig = prices > kama
        tsmom_sig = np.array(
            [
                prices[i] > prices[i - tsmom_period] if i >= tsmom_period else False
                for i in range(n)
            ]
        )

        entry = kama_sig | tsmom_sig
        signals[symbol] = pd.Series(entry, index=df.index)

    return signals, btc_gate


def run_backtest_exact_original(data: dict, signals: dict, btc_gate: pd.Series) -> dict:
    """
    EXACT Original Methodology:
    - On entry day (was_cash -> invested): count today's return
    - On exit day (was_invested -> cash): 0% return
    - On hold day (was_invested -> invested): count today's return
    """

    all_dates = sorted(set().union(*[df.index.tolist() for df in data.values()]))
    dates_2025 = [d for d in all_dates if HOLDOUT_START <= d <= HOLDOUT_END]

    portfolio_values = [10000]
    daily_returns = []
    was_invested = False
    prev_positions = {}

    for i, date in enumerate(dates_2025):
        # Get current gate and signals
        gate = btc_gate.get(date, False)

        # Get active symbols with signals
        active = []
        for symbol, sig_series in signals.items():
            if date in sig_series.index and sig_series.loc[date] and gate:
                if symbol in data and date in data[symbol].index:
                    vol = (
                        data[symbol].loc[date, "volume"]
                        * data[symbol].loc[date, "close"]
                    )
                    active.append((symbol, vol))

        # Sort by volume and take top N
        active_sorted = sorted(active, key=lambda x: x[1], reverse=True)
        selected = [s for s, v in active_sorted[:MAX_POSITIONS]]
        is_invested = len(selected) > 0

        # Calculate return based on original methodology
        if i == 0:
            daily_ret = 0
        else:
            if was_invested and is_invested:
                # HOLD: count return
                daily_pnl = 0
                for symbol, weight in prev_positions.items():
                    if symbol in data and date in data[symbol].index:
                        curr = data[symbol].loc[date, "close"]
                        prev_date = dates_2025[i - 1]
                        if prev_date in data[symbol].index:
                            prev = data[symbol].loc[prev_date, "close"]
                            daily_pnl += weight * (curr - prev) / prev
                daily_ret = daily_pnl
            elif not was_invested and is_invested:
                # ENTRY: count today's return (from yesterday's close)
                daily_pnl = 0
                weight = 1.0 / len(selected)
                for symbol in selected:
                    if symbol in data and date in data[symbol].index:
                        curr = data[symbol].loc[date, "close"]
                        prev_date = dates_2025[i - 1]
                        if prev_date in data[symbol].index:
                            prev = data[symbol].loc[prev_date, "close"]
                            daily_pnl += weight * (curr - prev) / prev
                daily_ret = daily_pnl
            elif was_invested and not is_invested:
                # EXIT: 0% return (sold at previous close)
                daily_ret = 0
            else:
                # CASH: 0% return
                daily_ret = 0

        portfolio_values.append(portfolio_values[-1] * (1 + daily_ret))
        daily_returns.append(daily_ret)

        # Update state
        was_invested = is_invested
        if is_invested:
            weight = 1.0 / len(selected)
            prev_positions = {s: weight for s in selected}
        else:
            prev_positions = {}

    portfolio_values = np.array(portfolio_values[1:])
    daily_returns = np.array(daily_returns)

    total_return = (portfolio_values[-1] - 10000) / 10000
    sharpe = (
        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        if np.std(daily_returns) > 0
        else 0
    )
    peak = np.maximum.accumulate(portfolio_values)
    mdd = np.min((portfolio_values - peak) / peak)

    return {
        "sharpe": sharpe,
        "mdd": mdd,
        "total_return": total_return,
        "n_days": len(daily_returns),
    }


# Main
print("\n[1] Loading Data")
print("-" * 60)

upbit_data = load_data(DATA_ROOT / "upbit_1d", ORIGINAL_SYMBOLS)
print(f"Loaded {len(upbit_data)} symbols")

print("\n[2] Running Backtests (Exact Original Methodology)")
print("-" * 60)

for name, kama, tsmom in [("KAMA=5, TSMOM=90", 5, 90), ("KAMA=10, TSMOM=60", 10, 60)]:
    signals, gate = generate_signals(upbit_data, kama, tsmom, 30)
    result = run_backtest_exact_original(upbit_data, signals, gate)
    print(f"{name}:")
    print(f"  Sharpe: {result['sharpe']:.3f}")
    print(f"  MDD:    {result['mdd']*100:.1f}%")
    print(f"  Return: {result['total_return']*100:.1f}%")

print("\n" + "=" * 80)
print("COMPARISON WITH ORIGINAL REPORT")
print("=" * 80)
print("""
Original (KAMA=5, TSMOM=90) Upbit:
  Sharpe 2.35, MDD -21.8%, Return +133%

Key difference in methodology:
  - EXIT days: 0% return (avoids the loss)
  - ENTRY days: Full return counted (captures the gain)
""")
print("=" * 80)
