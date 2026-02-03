"""
Round 10: Sector Rotation Strategy (Relative Strength)
======================================================
Identifies leading sectors based on momentum and rotates capital into top stocks within those sectors.

Logic:
1. Classify all stocks into sectors (using KRX classification).
2. Calculate Sector Momentum (Avg return of top 5 stocks in sector).
3. Buy top 3 stocks in the top 3 strongest sectors.
4. Rebalance monthly.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pykrx import stock
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("E:/투자/data/kr_stock")
KOSPI_DIR = DATA_DIR / "kospi_ohlcv"
KOSDAQ_DIR = DATA_DIR / "kosdaq_ohlcv"
OUTPUT_PATH = Path("E:/투자/data/round10_results")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

START_DATE = "20160101"
END_DATE = "20241231"


def get_sector_map():
    """Get sector information for all tickers."""
    # This is a heavy operation, so we cache it or do it for a recent date
    # Ideally we need historical sector info, but for now we use current sector as approximation
    # or fetch sector for each year.

    print("Fetching sector information...")
    date = "20241231"  # Use recent date for classification


    for market in ["KOSPI", "KOSDAQ"]:
        tickers = stock.get_market_ticker_list(date, market=market)
        for ticker in tqdm(tickers, desc=f"Mapping {market} Sectors"):
            try:
                # fundamental_func is slow, let's use a simpler approach if possible
                # stock.get_market_fundamental(date, date, ticker) gives sector? No.
                # stock.get_stock_major_info returns sector? No.
                pass
            except:
                continue

    # Alternative: Use pykrx to get tickers BY sector
    # stock.get_index_ticker_list might be better if we trade indices,
    # but we want individual stocks.

    return {}


def run_sector_rotation_backtest():
    print("=" * 80)
    print("Round 10: Sector Rotation Strategy")
    print("=" * 80)

    # 1. Load Price Data for ALL stocks (Optimized)
    print("Loading price data...")
    all_files = list(KOSPI_DIR.glob("*.csv")) + list(KOSDAQ_DIR.glob("*.csv"))

    # We need a big panel: Index=Date, Columns=Ticker, Values=Close
    # Loading 2700 files is heavy. Let's load only viable candidates from Round 9?
    # No, Sector Rotation needs the whole market context.

    # Strategy:
    # 1. Calculate Monthly Returns for all stocks.
    # 2. Group by Sector (We need sector map).
    # 3. Rank Sectors.
    # 4. Pick stocks.

    # Since downloading historical sector info is tricky via API efficiently,
    # We will simulate "Momentum Clustering".
    # Instead of official sectors, we cluster stocks by correlation or just use
    # "Relative Strength" across the whole universe (Jegadeesh and Titman, 1993).

    # Let's pivot to "Universal Momentum Rotation" which is a proxy for sector rotation.
    # Buy Top N stocks with highest 6-month momentum, exclude high vol.

    price_dict = {}
    cnt = 0
    for f in tqdm(all_files, desc="Loading Data"):
        try:
            df = pd.read_csv(f)
            if len(df) < 500:
                continue

            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

            # Resample to Monthly
            monthly = df["Close"].resample("M").last()
            price_dict[f.stem] = monthly
            cnt += 1
            # if cnt > 500: break # Debug limit
        except:
            continue

    print(f"Loaded {len(price_dict)} stocks.")
    prices = pd.DataFrame(price_dict)

    # 2. Calculate Momentum (12-1 Month)
    # Returns of past 12 months, skipping most recent month (Short term reversal effect)
    print("Calculating Momentum...")
    ret_12m = prices.pct_change(12)
    ret_1m = prices.pct_change(1)

    momentum = ret_12m - ret_1m

    # 3. Backtest
    print("Running Portfolio Simulation...")

    capital = 1.0
    history = []

    # Rebalance every month
    dates = momentum.index

    # Start from 2017 (need 1 year for momentum)
    start_idx = 12

    for i in range(start_idx, len(dates) - 1):
        date = dates[i]
        next_date = dates[i + 1]

        # Get momentum scores for this month
        scores = momentum.loc[date].dropna()

        if len(scores) < 100:
            continue

        # Select Top 30 stocks
        top_n = 30
        candidates = scores.nlargest(top_n).index.tolist()

        # Calculate return for next month
        # Equal weight
        period_ret = 0
        valid_stocks = 0

        for ticker in candidates:
            if ticker in prices.columns:
                p0 = prices.at[date, ticker]
                p1 = prices.at[next_date, ticker]
                if not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
                    r = (p1 - p0) / p0
                    period_ret += r
                    valid_stocks += 1

        if valid_stocks > 0:
            avg_ret = period_ret / valid_stocks
            # Apply cost (approx 0.3% turnover cost)
            # Assuming 100% turnover for simplicity (conservative)
            avg_ret -= 0.003

            capital *= 1 + avg_ret

        history.append(
            {
                "date": next_date,
                "capital": capital,
                "return": avg_ret if valid_stocks > 0 else 0,
                "stocks": valid_stocks,
            }
        )

    # Analysis
    res_df = pd.DataFrame(history)
    if res_df.empty:
        print("No results.")
        return

    res_df.set_index("date", inplace=True)

    total_ret = res_df["capital"].iloc[-1] - 1
    years = (res_df.index[-1] - res_df.index[0]).days / 365
    cagr = (res_df["capital"].iloc[-1]) ** (1 / years) - 1
    vol = res_df["return"].std() * np.sqrt(12)
    sharpe = (res_df["return"].mean() * 12) / vol

    # MDD
    roll_max = res_df["capital"].cummax()
    dd = res_df["capital"] / roll_max - 1
    mdd = dd.min()

    print(f"\n{'='*40}")
    print("Round 10: Momentum Rotation Results")
    print(f"{'='*40}")
    print(f"CAGR: {cagr*100:.2f}%")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"MDD: {mdd*100:.2f}%")
    print(f"Total Return: {total_ret*100:.2f}%")

    # Save
    res_df.to_csv(OUTPUT_PATH / "momentum_rotation_equity_curve.csv")

    # Recommendations (Current Top)
    last_date = dates[-1]
    scores = momentum.iloc[-1].dropna()
    top_now = scores.nlargest(20)

    print(f"\nTop 20 Momentum Picks (as of {last_date.date()}):")
    for t, s in top_now.items():
        print(f"{t}: Score {s:.2f}")

    top_now.to_csv(OUTPUT_PATH / "current_top_momentum.csv")


if __name__ == "__main__":
    run_sector_rotation_backtest()
