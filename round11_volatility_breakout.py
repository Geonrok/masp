"""
Round 11: Larry Williams Volatility Breakout (Proxy Simulation)
===============================================================
A classic strategy for volatile markets (like KOSDAQ).
Buy if Price > Open + (Range * k). Sell at Close.

Simulation Logic (using Daily OHLCV):
1. Target Price = Open + (Prev High - Prev Low) * k
2. If High > Target Price:
   - Assume Entry at Target Price (Stop Limit Order)
   - Exit at Close
   - Profit = (Close - Target Price) - Costs
3. No Overnight Position.

Parameters:
- k: 0.4, 0.5, 0.6, 0.7
- Moving Average Filter: Trade only if Open > MA(3, 5, 10)
"""

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("E:/투자/data/kr_stock")
KOSPI_DIR = DATA_DIR / "kospi_ohlcv"
KOSDAQ_DIR = DATA_DIR / "kosdaq_ohlcv"
OUTPUT_PATH = Path("E:/투자/data/round11_results")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Costs (Intraday, so slippage is critical)
# Commission: 0.015% * 2 (Buy/Sell)
# Tax: 0.20% (Sell)
# Slippage: 0.05% * 2 (Buy Stop / Sell MOC)
TOTAL_COST = 0.00015 * 2 + 0.0020 + 0.0005 * 2
# Total ~ 0.33% per trade (Very high hurdle)


def run_volatility_breakout(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 500:
            return []

        # Date parsing
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
        else:
            return []

        df = df.sort_index()

        # Calculate Indicators
        df["range"] = df["High"] - df["Low"]
        df["prev_range"] = df["range"].shift(1)
        df["ma5"] = df["Close"].rolling(5).mean().shift(1)

        results = []

        # Test Parameters
        k_values = [0.4, 0.5, 0.6]

        for k in k_values:
            # Vectorized Backtest
            target_price = df["Open"] + df["prev_range"] * k

            # Entry Condition: High > Target
            # Filter: Open > MA5 (Trend Filter)
            condition = (df["High"] > target_price) & (df["Open"] > df["ma5"])

            # Calculate Returns
            # If triggered, Buy at Target, Sell at Close
            daily_profit = (df["Close"] - target_price) / target_price
            daily_profit -= TOTAL_COST

            # Apply signals (0 if not triggered)
            strategy_ret = daily_profit.where(condition, 0)

            # Metrics (2022-2024 Test Period)
            test_ret = strategy_ret["2022-01-01":"2024-12-31"]

            if len(test_ret) == 0:
                continue

            cum_ret = (1 + test_ret).cumprod()
            total_return = cum_ret.iloc[-1] - 1

            # Count trades
            trades = condition["2022-01-01":"2024-12-31"].sum()
            if trades < 20:
                continue  # Too few trades

            # MDD
            roll_max = cum_ret.cummax()
            dd = cum_ret / roll_max - 1
            mdd = dd.min()

            # Sharpe (Daily)
            if test_ret.std() == 0:
                continue
            sharpe = (test_ret.mean() * 252) / (test_ret.std() * np.sqrt(252))

            # Win Rate
            win_rate = (test_ret > 0).sum() / trades

            results.append(
                {
                    "ticker": file_path.stem,
                    "k": k,
                    "sharpe": sharpe,
                    "return": total_return,
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
    print("Round 11: Larry Williams Volatility Breakout")
    print("=" * 80)

    all_files = list(KOSPI_DIR.glob("*.csv")) + list(KOSDAQ_DIR.glob("*.csv"))
    print(f"Scanning {len(all_files)} files...")

    all_results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_volatility_breakout, f) for f in all_files]

        for f in tqdm(as_completed(futures), total=len(futures)):
            res = f.result()
            if res:
                all_results.extend(res)

    if not all_results:
        print("No results.")
        return

    df = pd.DataFrame(all_results)

    # Filter Viable
    viable = df[
        (df["sharpe"] > 1.0) & (df["return"] > 0.3) & (df["mdd"] > -0.3)
    ].sort_values("sharpe", ascending=False)

    print(f"\nTotal Viable Strategies: {len(viable)}")
    print("\nTOP 20 Volatility Breakout Candidates:")
    print(viable.head(20))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viable.to_csv(OUTPUT_PATH / f"round11_viable_{timestamp}.csv", index=False)
    df.to_csv(OUTPUT_PATH / f"round11_all_{timestamp}.csv", index=False)


if __name__ == "__main__":
    main()
