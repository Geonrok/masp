"""
Data loader for KOSPI backtesting.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd


class KOSPIDataLoader:
    """
    Load and manage KOSPI stock data for backtesting.
    """

    DATA_DIR = Path("E:/투자/data/kr_stock/kospi_ohlcv")

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or self.DATA_DIR
        self._cache: Dict[str, pd.DataFrame] = {}
        self._ticker_list: Optional[List[str]] = None

    def get_available_tickers(self) -> List[str]:
        """Get list of available ticker files."""
        if self._ticker_list is None:
            files = list(self.data_dir.glob("*.csv"))
            self._ticker_list = [f.stem for f in files if not f.stem.startswith("_")]
        return self._ticker_list

    def load_ticker(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a single ticker.

        Args:
            ticker: Stock ticker code
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = ticker

        if cache_key not in self._cache:
            filepath = self.data_dir / f"{ticker}.csv"
            if not filepath.exists():
                raise FileNotFoundError(f"No data for ticker: {ticker}")

            df = pd.read_csv(filepath, parse_dates=["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            self._cache[cache_key] = df

        df = self._cache[cache_key].copy()

        # Filter by date range
        if start_date:
            df = df[df["Date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["Date"] <= pd.to_datetime(end_date)]

        return df.reset_index(drop=True)

    def load_all_tickers(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_history_days: int = 250,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all available tickers with minimum history requirement.

        Args:
            start_date: Start date
            end_date: End date
            min_history_days: Minimum trading days required

        Returns:
            Dict of ticker -> DataFrame
        """
        data = {}
        tickers = self.get_available_tickers()

        for ticker in tickers:
            try:
                df = self.load_ticker(ticker, start_date, end_date)
                if len(df) >= min_history_days:
                    data[ticker] = df
            except Exception:
                continue

        return data

    def get_universe_panel(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Create a panel DataFrame with all tickers.

        Returns wide DataFrame with Date as index and ticker columns for Close prices.
        """
        dfs = []
        for ticker in tickers:
            try:
                df = self.load_ticker(ticker, start_date, end_date)
                df = df[["Date", "Close"]].copy()
                df = df.rename(columns={"Close": ticker})
                df = df.set_index("Date")
                dfs.append(df)
            except Exception:
                continue

        if not dfs:
            return pd.DataFrame()

        panel = pd.concat(dfs, axis=1)
        return panel.sort_index()

    def get_benchmark_data(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get KOSPI index data as benchmark.

        Uses KODEX 200 (069500) as proxy if available.
        """
        # Try KODEX 200 ETF as benchmark
        try:
            return self.load_ticker("069500", start_date, end_date)
        except FileNotFoundError:
            # Calculate equal-weighted index from top stocks
            return pd.DataFrame()
