"""
Round 13: Volume-Enhanced Volatility Breakout
=============================================
Adding Volume filters to the Larry Williams Breakout.
Hypothesis: Breakouts with massive volume surge are more reliable (Smart Money footprint).

Strategies:
1. VolBreakout + Volume > 200% of MA(5) Volume
2. VolBreakout + OBV Trending Up
3. VolBreakout + MFI (Money Flow Index) Check
"""

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("E:/투자/data/kr_stock")
KOSPI_DIR = DATA_DIR / "kospi_ohlcv"
KOSDAQ_DIR = DATA_DIR / "kosdaq_ohlcv"
OUTPUT_PATH = Path("E:/투자/data/round13_results")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

TOTAL_COST = 0.0035  # 0.35% (Slippage + Comm + Tax)


def run_volume_strategy(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 500:
            return []

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
        else:
            return []

        df = df.sort_index()

        # Indicators
        df["range"] = df["High"] - df["Low"]
        df["prev_range"] = df["range"].shift(1)

        # Volume Indicators
        df["vol_ma5"] = df["Volume"].rolling(5).mean().shift(1)
        df["vol_ratio"] = df["Volume"] / df["vol_ma5"]

        # OBV (Simplified Trend)
        obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
        df["obv_ma20"] = obv.rolling(20).mean()
        df["obv_trend"] = obv > df["obv_ma20"]

        results = []
        k = 0.5  # Fixed k for comparison

        # Strategy 1: Volume Surge Breakout
        target = df["Open"] + df["prev_range"] * k
        cond_surge = (df["High"] > target) & (df["vol_ratio"] > 2.0)  # 2x Volume

        # Strategy 2: OBV Trend Breakout
        cond_obv = (df["High"] > target) & df["obv_trend"].shift(1)

        for condition, name in [(cond_surge, "VolSurge_200"), (cond_obv, "OBV_Trend")]:
            daily_ret = (df["Close"] - target) / target - TOTAL_COST
            strat_ret = daily_ret.where(condition, 0)

            test_ret = strat_ret["2022-01-01":"2024-12-31"]
            if len(test_ret) == 0:
                continue

            trades = condition["2022-01-01":"2024-12-31"].sum()
            if trades < 10:
                continue

            total_ret = (1 + test_ret).cumprod().iloc[-1] - 1
            win_rate = (test_ret > 0).sum() / trades

            # Sharpe
            if test_ret.std() == 0:
                continue
            sharpe = (test_ret.mean() * 252) / (test_ret.std() * np.sqrt(252))

            # MDD
            cum = (1 + test_ret).cumprod()
            mdd = (cum / cum.cummax() - 1).min()

            results.append(
                {
                    "ticker": file_path.stem,
                    "strategy": name,
                    "sharpe": sharpe,
                    "return": total_ret,
                    "mdd": mdd,
                    "win_rate": win_rate,
                    "trades": trades,
                }
            )

        return results

    except Exception:
        return []


def main():
    print("=" * 80)
    print("Round 13: Volume-Enhanced Strategies")
    print("=" * 80)

    all_files = list(KOSPI_DIR.glob("*.csv")) + list(KOSDAQ_DIR.glob("*.csv"))
    all_results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_volume_strategy, f) for f in all_files]
        for f in tqdm(as_completed(futures), total=len(futures)):
            res = f.result()
            if res:
                all_results.extend(res)

    if not all_results:
        print("No results.")
        return

    df = pd.DataFrame(all_results)

    viable = df[
        (df["sharpe"] > 1.0) & (df["return"] > 0.5) & (df["trades"] >= 20)
    ].sort_values("sharpe", ascending=False)

    print(f"\nTotal Viable Volume Strategies: {len(viable)}")
    print(viable.head(20))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viable.to_csv(OUTPUT_PATH / f"round13_viable_{timestamp}.csv", index=False)


if __name__ == "__main__":
    main()
