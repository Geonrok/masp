"""
Historical Data Loader

Provides loading, caching, and management of historical OHLCV data
for backtesting purposes.
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

import numpy as np
import pandas as pd

from libs.adapters.base import OHLCV

logger = logging.getLogger(__name__)


@dataclass
class OHLCVDataset:
    """Container for historical OHLCV data."""

    symbol: str
    interval: str  # "1m", "5m", "1h", "1d"
    data: pd.DataFrame  # columns: timestamp, open, high, low, close, volume
    start_date: datetime
    end_date: datetime
    source: str = "unknown"

    @property
    def length(self) -> int:
        """Number of candles in dataset."""
        return len(self.data)

    @property
    def closes(self) -> np.ndarray:
        """Close prices as numpy array."""
        return self.data["close"].values

    @property
    def opens(self) -> np.ndarray:
        """Open prices as numpy array."""
        return self.data["open"].values

    @property
    def highs(self) -> np.ndarray:
        """High prices as numpy array."""
        return self.data["high"].values

    @property
    def lows(self) -> np.ndarray:
        """Low prices as numpy array."""
        return self.data["low"].values

    @property
    def volumes(self) -> np.ndarray:
        """Volumes as numpy array."""
        return self.data["volume"].values

    @property
    def timestamps(self) -> List[datetime]:
        """Timestamps as list."""
        return pd.to_datetime(self.data["timestamp"]).tolist()

    def slice(self, start: int, end: int) -> OHLCVDataset:
        """Get a slice of the dataset."""
        sliced_data = self.data.iloc[start:end].reset_index(drop=True)
        return OHLCVDataset(
            symbol=self.symbol,
            interval=self.interval,
            data=sliced_data,
            start_date=pd.to_datetime(sliced_data["timestamp"].iloc[0]),
            end_date=pd.to_datetime(sliced_data["timestamp"].iloc[-1]),
            source=self.source,
        )

    def to_ohlcv_list(self) -> List[OHLCV]:
        """Convert to list of OHLCV objects."""
        return [
            OHLCV(
                timestamp=str(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for _, row in self.data.iterrows()
        ]


class HistoricalDataLoader:
    """
    Loads and caches historical OHLCV data from various sources.

    Supports:
    - CSV files
    - JSON files
    - Exchange adapters (real-time fetch)
    - Local SQLite cache
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize data loader.

        Args:
            cache_dir: Directory for cached data (default: storage/backtest_cache)
            enable_cache: Enable file-based caching
        """
        self.cache_dir = Path(cache_dir or "storage/backtest_cache")
        self.enable_cache = enable_cache

        if enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._memory_cache: Dict[str, OHLCVDataset] = {}

        logger.info(f"[HistoricalDataLoader] Initialized, cache_dir={self.cache_dir}")

    def _get_cache_key(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> str:
        """Generate cache key for data request."""
        key_str = f"{symbol}_{interval}_{start_date.isoformat()}_{end_date.isoformat()}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cached data."""
        return self.cache_dir / f"{cache_key}.parquet"

    def load_from_csv(
        self,
        filepath: str,
        symbol: str,
        interval: str = "1d",
        date_column: str = "timestamp",
        date_format: Optional[str] = None,
    ) -> OHLCVDataset:
        """
        Load historical data from CSV file.

        Expected columns: timestamp/date, open, high, low, close, volume

        Args:
            filepath: Path to CSV file
            symbol: Symbol name
            interval: Data interval
            date_column: Name of date column
            date_format: Date format string (auto-detect if None)

        Returns:
            OHLCVDataset
        """
        logger.info(f"[HistoricalDataLoader] Loading CSV: {filepath}")

        df = pd.read_csv(filepath)

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Map common column name variations
        column_mapping = {
            "date": "timestamp",
            "datetime": "timestamp",
            "time": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vol": "volume",
        }
        df.rename(columns=column_mapping, inplace=True)

        # Parse dates
        if date_format:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format=date_format)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort by timestamp
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Ensure required columns
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Select only required columns
        df = df[required].copy()

        # Convert to float
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop NaN rows
        df.dropna(inplace=True)

        dataset = OHLCVDataset(
            symbol=symbol,
            interval=interval,
            data=df,
            start_date=df["timestamp"].iloc[0],
            end_date=df["timestamp"].iloc[-1],
            source=f"csv:{filepath}",
        )

        logger.info(
            f"[HistoricalDataLoader] Loaded {dataset.length} candles "
            f"from {dataset.start_date} to {dataset.end_date}"
        )

        return dataset

    def load_from_json(
        self,
        filepath: str,
        symbol: str,
        interval: str = "1d",
    ) -> OHLCVDataset:
        """
        Load historical data from JSON file.

        Expected format: List of objects with timestamp, open, high, low, close, volume

        Args:
            filepath: Path to JSON file
            symbol: Symbol name
            interval: Data interval

        Returns:
            OHLCVDataset
        """
        logger.info(f"[HistoricalDataLoader] Loading JSON: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df.columns = df.columns.str.lower()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        dataset = OHLCVDataset(
            symbol=symbol,
            interval=interval,
            data=df[["timestamp", "open", "high", "low", "close", "volume"]],
            start_date=df["timestamp"].iloc[0],
            end_date=df["timestamp"].iloc[-1],
            source=f"json:{filepath}",
        )

        return dataset

    def load_from_adapter(
        self,
        adapter: Any,
        symbol: str,
        interval: str = "1d",
        limit: int = 1000,
        use_cache: bool = True,
    ) -> OHLCVDataset:
        """
        Load historical data from market data adapter.

        Args:
            adapter: MarketDataAdapter instance
            symbol: Symbol to fetch
            interval: Data interval
            limit: Number of candles to fetch
            use_cache: Use cached data if available

        Returns:
            OHLCVDataset
        """
        cache_key = f"{symbol}_{interval}_{limit}"

        # Check memory cache
        if use_cache and cache_key in self._memory_cache:
            logger.debug(f"[HistoricalDataLoader] Memory cache hit: {cache_key}")
            return self._memory_cache[cache_key]

        # Fetch from adapter
        logger.info(f"[HistoricalDataLoader] Fetching from adapter: {symbol}")

        ohlcv_list = adapter.get_ohlcv(symbol, interval=interval, limit=limit)

        if not ohlcv_list:
            raise ValueError(f"No data returned for {symbol}")

        # Convert to DataFrame
        data = []
        for candle in ohlcv_list:
            data.append({
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            })

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        dataset = OHLCVDataset(
            symbol=symbol,
            interval=interval,
            data=df,
            start_date=df["timestamp"].iloc[0],
            end_date=df["timestamp"].iloc[-1],
            source=f"adapter:{type(adapter).__name__}",
        )

        # Cache in memory
        if use_cache:
            self._memory_cache[cache_key] = dataset

        return dataset

    def save_to_cache(self, dataset: OHLCVDataset) -> str:
        """
        Save dataset to file cache.

        Args:
            dataset: Dataset to cache

        Returns:
            Cache key
        """
        if not self.enable_cache:
            return ""

        cache_key = self._get_cache_key(
            dataset.symbol,
            dataset.interval,
            dataset.start_date,
            dataset.end_date,
        )

        cache_path = self._get_cache_path(cache_key)
        dataset.data.to_parquet(cache_path, index=False)

        # Save metadata
        meta_path = cache_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "symbol": dataset.symbol,
                "interval": dataset.interval,
                "start_date": dataset.start_date.isoformat(),
                "end_date": dataset.end_date.isoformat(),
                "source": dataset.source,
                "length": dataset.length,
            }, f)

        logger.info(f"[HistoricalDataLoader] Cached: {cache_key}")
        return cache_key

    def load_from_cache(self, cache_key: str) -> Optional[OHLCVDataset]:
        """
        Load dataset from file cache.

        Args:
            cache_key: Cache key

        Returns:
            OHLCVDataset or None if not found
        """
        cache_path = self._get_cache_path(cache_key)
        meta_path = cache_path.with_suffix(".meta.json")

        if not cache_path.exists() or not meta_path.exists():
            return None

        df = pd.read_parquet(cache_path)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        return OHLCVDataset(
            symbol=meta["symbol"],
            interval=meta["interval"],
            data=df,
            start_date=datetime.fromisoformat(meta["start_date"]),
            end_date=datetime.fromisoformat(meta["end_date"]),
            source=meta["source"],
        )

    def generate_synthetic(
        self,
        symbol: str = "SYNTHETIC",
        days: int = 365,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0001,
        interval: str = "1d",
    ) -> OHLCVDataset:
        """
        Generate synthetic OHLCV data for testing.

        Uses geometric Brownian motion for price simulation.

        Args:
            symbol: Symbol name
            days: Number of days to generate
            initial_price: Starting price
            volatility: Daily volatility (default 2%)
            drift: Daily drift/trend
            interval: Data interval

        Returns:
            OHLCVDataset with synthetic data
        """
        logger.info(f"[HistoricalDataLoader] Generating synthetic data: {days} days")

        np.random.seed(42)  # Reproducible

        # Generate prices using GBM
        dt = 1.0  # Daily
        n_periods = days

        # Random returns
        returns = np.random.normal(drift, volatility, n_periods)

        # Calculate prices
        prices = initial_price * np.exp(np.cumsum(returns))
        prices = np.insert(prices, 0, initial_price)[:n_periods]

        # Generate OHLCV
        data = []
        start_date = datetime.now() - timedelta(days=days)

        for i in range(n_periods):
            close = prices[i]

            # Generate intraday range
            daily_range = close * volatility * np.random.uniform(0.5, 1.5)
            high = close + daily_range * np.random.uniform(0.3, 0.7)
            low = close - daily_range * np.random.uniform(0.3, 0.7)

            # Open somewhere between high and low
            open_price = low + (high - low) * np.random.uniform(0.2, 0.8)

            # Volume with some randomness
            base_volume = 1_000_000
            volume = base_volume * np.random.uniform(0.5, 2.0)

            data.append({
                "timestamp": start_date + timedelta(days=i),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": round(volume, 0),
            })

        df = pd.DataFrame(data)

        return OHLCVDataset(
            symbol=symbol,
            interval=interval,
            data=df,
            start_date=df["timestamp"].iloc[0],
            end_date=df["timestamp"].iloc[-1],
            source="synthetic:GBM",
        )

    def clear_cache(self) -> int:
        """
        Clear all cached data.

        Returns:
            Number of files cleared
        """
        self._memory_cache.clear()

        if not self.enable_cache:
            return 0

        count = 0
        for file in self.cache_dir.glob("*"):
            file.unlink()
            count += 1

        logger.info(f"[HistoricalDataLoader] Cleared {count} cached files")
        return count
