"""
ATLAS Futures Strategy Backtest
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    import ta
except ImportError:
    print("Installing ta library...")
    import subprocess

    subprocess.run(["pip", "install", "ta", "-q"])
    import ta

from scipy.stats import percentileofscore

DATA_PATH = Path("E:/data/crypto_ohlcv/binance_futures_4h")


def load_and_calculate(symbol: str, start_date: str = "2023-01-01"):
    """Load data and calculate all indicators at once."""
    file_path = DATA_PATH / f"{symbol}.csv"
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path, parse_dates=["datetime"])
    df = df[df["datetime"] >= start_date].copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2.0)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    # BBWP (Bollinger Band Width Percentile)
    df["bbwp"] = (
        df["bb_width"]
        .rolling(252)
        .apply(
            lambda x: percentileofscore(x, x.iloc[-1]) if len(x) >= 10 else 50,
            raw=False,
        )
    )

    # ADX
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    # EMA 200
    df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)

    # ATR
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )

    # RVOL
    df["rvol"] = df["volume"] / df["volume"].rolling(20).mean()

    return df


def vectorized_backtest(df, symbol, leverage=3, position_size_pct=1.6):
    """Fast vectorized backtest."""
    df = df.dropna().copy()

    # Squeeze condition
    squeeze = df["bbwp"] < 10
    squeeze_count = squeeze.groupby((~squeeze).cumsum()).cumsum()
    squeeze_ok = squeeze_count >= 6

    # Entry conditions
    adx_ok = df["adx"] >= 12
    rvol_threshold = 1.5 if symbol in ["BTCUSDT", "ETHUSDT"] else 3.0
    rvol_ok = df["rvol"] > rvol_threshold

    long_entry = (
        squeeze_ok
        & adx_ok
        & rvol_ok
        & (df["close"] > df["ema_200"])
        & (df["close"] > df["bb_upper"])
    )

    short_entry = (
        squeeze_ok
        & adx_ok
        & rvol_ok
        & (df["close"] < df["ema_200"])
        & (df["close"] < df["bb_lower"])
    )

    # Simulate trades
    trades = []
    position = None
    capital = 10000.0

    for i in range(len(df)):
        row = df.iloc[i]

        if position is None:
            if long_entry.iloc[i]:
                position = {
                    "side": "LONG",
                    "entry": row["close"],
                    "entry_idx": i,
                    "high": row["close"],
                }
            elif short_entry.iloc[i]:
                position = {
                    "side": "SHORT",
                    "entry": row["close"],
                    "entry_idx": i,
                    "low": row["close"],
                }
        else:
            bars_held = i - position["entry_idx"]

            if position["side"] == "LONG":
                position["high"] = max(position["high"], row["high"])
                chandelier_stop = position["high"] - row["atr"] * 3
                pnl_pct = (
                    (row["close"] - position["entry"]) / position["entry"] * leverage
                )

                # Exit conditions
                if row["close"] < chandelier_stop or (
                    bars_held >= 4 and pnl_pct < -0.0016
                ):
                    trade_return = pnl_pct - 0.002  # costs
                    trade_pnl = capital * (position_size_pct / 100) * trade_return
                    capital += trade_pnl

                    trades.append(
                        {
                            "symbol": symbol,
                            "side": "LONG",
                            "entry": position["entry"],
                            "exit": row["close"],
                            "bars": bars_held,
                            "pnl_pct": trade_return * 100,
                            "capital": capital,
                            "time": row["datetime"],
                        }
                    )
                    position = None
            else:
                position["low"] = min(position["low"], row["low"])
                chandelier_stop = position["low"] + row["atr"] * 3
                pnl_pct = (
                    (position["entry"] - row["close"]) / position["entry"] * leverage
                )

                if row["close"] > chandelier_stop or (
                    bars_held >= 4 and pnl_pct < -0.0016
                ):
                    trade_return = pnl_pct - 0.002
                    trade_pnl = capital * (position_size_pct / 100) * trade_return
                    capital += trade_pnl

                    trades.append(
                        {
                            "symbol": symbol,
                            "side": "SHORT",
                            "entry": position["entry"],
                            "exit": row["close"],
                            "bars": bars_held,
                            "pnl_pct": trade_return * 100,
                            "capital": capital,
                            "time": row["datetime"],
                        }
                    )
                    position = None

    return trades, capital


def main():
    print("=" * 70)
    print("ATLAS Futures Strategy Backtest (2023-2025)")
    print("=" * 70)

    symbols = ["BTCUSDT", "ETHUSDT", "LINKUSDT"]
    all_trades = []
    results = {}

    for symbol in symbols:
        print(f"\n{symbol}:")
        df = load_and_calculate(symbol, "2023-01-01")

        if df is None:
            print("  No data")
            continue

        print(
            f'  Bars: {len(df)} ({df["datetime"].min().date()} ~ {df["datetime"].max().date()})'
        )

        trades, final_capital = vectorized_backtest(df, symbol)

        if trades:
            trades_df = pd.DataFrame(trades)
            total_return = (final_capital / 10000 - 1) * 100
            win_rate = len(trades_df[trades_df["pnl_pct"] > 0]) / len(trades_df) * 100

            # Calculate MDD
            capitals = [10000] + [t["capital"] for t in trades]
            peak = pd.Series(capitals).cummax()
            dd = (pd.Series(capitals) - peak) / peak * 100
            max_dd = dd.min()

            results[symbol] = {
                "trades": len(trades),
                "long": len(trades_df[trades_df["side"] == "LONG"]),
                "short": len(trades_df[trades_df["side"] == "SHORT"]),
                "return": total_return,
                "mdd": max_dd,
                "win_rate": win_rate,
                "avg_pnl": trades_df["pnl_pct"].mean(),
                "final": final_capital,
            }

            print(
                f'  Trades: {len(trades)} (L:{results[symbol]["long"]}, S:{results[symbol]["short"]})'
            )
            print(f"  Return: {total_return:.2f}%")
            print(f"  MDD: {max_dd:.2f}%")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f'  Avg P/L: {trades_df["pnl_pct"].mean():.2f}%')

            all_trades.extend(trades)
        else:
            print("  No trades")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_trades:
        all_df = pd.DataFrame(all_trades)

        print(f"\nTotal Trades: {len(all_df)}")
        print(f'  LONG: {len(all_df[all_df["side"]=="LONG"])}')
        print(f'  SHORT: {len(all_df[all_df["side"]=="SHORT"])}')

        print(
            f'\nOverall Win Rate: {len(all_df[all_df["pnl_pct"]>0])/len(all_df)*100:.1f}%'
        )
        print(f'Average P/L per trade: {all_df["pnl_pct"].mean():.2f}%')

        print("\nBy Symbol:")
        print(f'{"Symbol":<10} {"Return":>10} {"MDD":>10} {"Trades":>8} {"Win%":>8}')
        print("-" * 50)
        for sym, r in results.items():
            print(
                f'{sym:<10} {r["return"]:>9.1f}% {r["mdd"]:>9.1f}% {r["trades"]:>8} {r["win_rate"]:>7.1f}%'
            )

        # Recent trades
        print("\nRecent 10 Trades:")
        recent = all_df.tail(10)
        for _, t in recent.iterrows():
            print(
                f'  {t["time"].date()} {t["symbol"]} {t["side"]:5} {t["pnl_pct"]:+.2f}%'
            )


if __name__ == "__main__":
    main()
