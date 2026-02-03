#!/usr/bin/env python3
"""
Google Trends Data Collector for Bitcoin

Collects search interest data for Bitcoin-related keywords.
Weekly data available for long history (2015+).

Usage:
    pip install pytrends
    python collect_google_trends.py
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    from pytrends.request import TrendReq
except ImportError:
    print("Please install pytrends: pip install pytrends")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("E:/data/crypto_ohlcv/sentiment")


def collect_google_trends(
    keywords: list = None,
    start_date: str = "2017-01-01",
    end_date: str = None,
) -> pd.DataFrame:
    """
    Collect Google Trends data for keywords.

    Note: For long timeframes, Google returns weekly data.
    For <90 days, daily data is available.
    """
    if keywords is None:
        keywords = ["bitcoin"]

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Collecting Google Trends for: {keywords}")
    logger.info(f"Period: {start_date} to {end_date}")

    pytrends = TrendReq(hl="en-US", tz=360)

    # Build timeframe string
    timeframe = f"{start_date} {end_date}"

    try:
        pytrends.build_payload(
            kw_list=keywords,
            cat=0,  # All categories
            timeframe=timeframe,
            geo="",  # Worldwide
        )

        df = pytrends.interest_over_time()

        if df.empty:
            logger.warning("No data returned from Google Trends")
            return pd.DataFrame()

        # Remove 'isPartial' column if exists
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])

        df = df.reset_index()
        df = df.rename(columns={"date": "datetime"})

        logger.info(
            f"Collected {len(df)} records ({df['datetime'].min()} to {df['datetime'].max()})"
        )

        return df

    except Exception as e:
        logger.error(f"Error collecting Google Trends: {e}")
        return pd.DataFrame()


def collect_multiple_periods(
    keywords: list = None,
    start_year: int = 2017,
) -> pd.DataFrame:
    """
    Collect Google Trends in yearly chunks to avoid rate limits.
    Then combine into single DataFrame.
    """
    if keywords is None:
        keywords = ["bitcoin"]

    all_data = []
    current_year = datetime.now().year

    for year in range(start_year, current_year + 1):
        start = f"{year}-01-01"
        end = (
            f"{year}-12-31"
            if year < current_year
            else datetime.now().strftime("%Y-%m-%d")
        )

        logger.info(f"Collecting {year}...")

        df = collect_google_trends(keywords, start, end)

        if not df.empty:
            all_data.append(df)

        # Rate limit - wait between requests
        time.sleep(2)

    if not all_data:
        return pd.DataFrame()

    # Combine all years
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")

    return combined


def normalize_trends_data(df: pd.DataFrame, keyword: str = "bitcoin") -> pd.DataFrame:
    """
    Normalize Google Trends data to 0-100 scale.
    Google already normalizes to peak=100 for each request,
    but combining multiple requests needs re-normalization.
    """
    if keyword in df.columns:
        max_val = df[keyword].max()
        if max_val > 0:
            df[f"{keyword}_normalized"] = (df[keyword] / max_val) * 100

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keywords to collect
    keywords_list = [
        ["bitcoin"],
        ["bitcoin buy"],
        ["crypto crash"],
        ["bitcoin halving"],
    ]

    all_results = {}

    for keywords in keywords_list:
        keyword_str = keywords[0].replace(" ", "_")
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting: {keywords}")
        logger.info("=" * 60)

        # Collect long-term data (weekly)
        df = collect_google_trends(
            keywords=keywords,
            start_date="2017-01-01",
        )

        if not df.empty:
            # Save raw data
            output_path = OUTPUT_DIR / f"google_trends_{keyword_str}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved: {output_path}")

            all_results[keyword_str] = df

        time.sleep(3)  # Rate limit between different keywords

    # Create combined sentiment index
    if "bitcoin" in all_results:
        bitcoin_df = all_results["bitcoin"].copy()
        bitcoin_df = bitcoin_df.rename(columns={"bitcoin": "search_interest"})

        # Calculate rolling averages
        bitcoin_df["search_interest_ma4"] = (
            bitcoin_df["search_interest"].rolling(4).mean()
        )
        bitcoin_df["search_interest_ma12"] = (
            bitcoin_df["search_interest"].rolling(12).mean()
        )

        # Calculate momentum (rate of change)
        bitcoin_df["search_momentum"] = (
            bitcoin_df["search_interest"].pct_change(4) * 100
        )

        # Save combined data
        output_path = OUTPUT_DIR / "google_trends_bitcoin_enhanced.csv"
        bitcoin_df.to_csv(output_path, index=False)
        logger.info(f"\nSaved enhanced data: {output_path}")

    logger.info("\n" + "=" * 60)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("=" * 60)

    # Print summary
    for keyword, df in all_results.items():
        logger.info(f"  {keyword}: {len(df)} records")


if __name__ == "__main__":
    main()
