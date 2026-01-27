#!/usr/bin/env python3
"""
Multi-Factor Daily Data Collector
Automatically collects 5 core factors for all Binance Futures symbols

Factors:
1. Price Data (OHLCV) - Binance API
2. Fear & Greed Index - alternative.me API
3. Funding Rate - Binance Futures API
4. Macro (DXY, VIX, SP500) - Yahoo Finance
5. BTC Correlation - Calculated from price data

Run daily via Task Scheduler or cron
"""
from __future__ import annotations

import asyncio
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import time

import aiohttp
import pandas as pd
import numpy as np

# Configuration
DATA_ROOT = Path("E:/data/crypto_ohlcv")
LOG_FILE = DATA_ROOT / "collector_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class BinanceCollector:
    """Collect data from Binance Futures API"""

    BASE_URL = "https://fapi.binance.com"

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def get_all_symbols(self) -> List[str]:
        """Get all USDT perpetual futures symbols"""
        url = f"{self.BASE_URL}/fapi/v1/exchangeInfo"
        async with self.session.get(url) as resp:
            data = await resp.json()
            symbols = [
                s['symbol'] for s in data['symbols']
                if s['symbol'].endswith('USDT')
                and s['contractType'] == 'PERPETUAL'
                and s['status'] == 'TRADING'
            ]
            return sorted(symbols)

    async def get_klines(self, symbol: str, interval: str = "4h", limit: int = 100) -> pd.DataFrame:
        """Get OHLCV klines data"""
        url = f"{self.BASE_URL}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        try:
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    return df
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
        return pd.DataFrame()

    async def get_funding_rate(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get funding rate history"""
        url = f"{self.BASE_URL}/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": limit}

        try:
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    df['datetime'] = pd.to_datetime(df['fundingTime'], unit='ms')
                    df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
                    return df[['datetime', 'funding_rate', 'symbol']]
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
        return pd.DataFrame()

    async def get_mark_price(self, symbol: str) -> Dict:
        """Get current mark price and funding rate"""
        url = f"{self.BASE_URL}/fapi/v1/premiumIndex"
        params = {"symbol": symbol}

        try:
            async with self.session.get(url, params=params) as resp:
                return await resp.json()
        except Exception as e:
            logger.error(f"Error fetching mark price for {symbol}: {e}")
        return {}


class FearGreedCollector:
    """Collect Fear & Greed Index from alternative.me"""

    URL = "https://api.alternative.me/fng/?limit=30"

    async def fetch(self) -> pd.DataFrame:
        """Fetch Fear & Greed Index data"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.URL) as resp:
                    data = await resp.json()
                    if 'data' in data:
                        records = []
                        for item in data['data']:
                            records.append({
                                'datetime': pd.to_datetime(int(item['timestamp']), unit='s'),
                                'fear_greed': int(item['value']),
                                'classification': item['value_classification']
                            })
                        return pd.DataFrame(records)
            except Exception as e:
                logger.error(f"Error fetching Fear & Greed: {e}")
        return pd.DataFrame()


class MacroCollector:
    """Collect macro indicators from Yahoo Finance"""

    SYMBOLS = {
        'DXY': 'DX-Y.NYB',      # US Dollar Index
        'VIX': '^VIX',          # Volatility Index
        'SP500': '^GSPC',       # S&P 500
        'NASDAQ': '^IXIC',      # NASDAQ
        'US10Y': '^TNX',        # 10-Year Treasury Yield
        'GOLD': 'GC=F',         # Gold Futures
    }

    async def fetch(self, days: int = 30) -> Dict[str, pd.DataFrame]:
        """Fetch macro data from Yahoo Finance"""
        results = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        async with aiohttp.ClientSession() as session:
            for name, ticker in self.SYMBOLS.items():
                try:
                    # Yahoo Finance API
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                    params = {
                        'period1': int(start_date.timestamp()),
                        'period2': int(end_date.timestamp()),
                        'interval': '1d'
                    }
                    headers = {'User-Agent': 'Mozilla/5.0'}

                    async with session.get(url, params=params, headers=headers) as resp:
                        data = await resp.json()

                        if 'chart' in data and 'result' in data['chart']:
                            result = data['chart']['result'][0]
                            timestamps = result['timestamp']
                            quotes = result['indicators']['quote'][0]

                            df = pd.DataFrame({
                                'datetime': pd.to_datetime(timestamps, unit='s'),
                                'open': quotes.get('open'),
                                'high': quotes.get('high'),
                                'low': quotes.get('low'),
                                'close': quotes.get('close'),
                                'volume': quotes.get('volume')
                            })
                            results[name] = df
                            logger.info(f"Fetched {name}: {len(df)} rows")

                    await asyncio.sleep(0.5)  # Rate limiting

                except Exception as e:
                    logger.error(f"Error fetching {name}: {e}")

        return results


class DataCollector:
    """Main data collector orchestrator"""

    def __init__(self):
        self.data_root = DATA_ROOT
        self.data_root.mkdir(parents=True, exist_ok=True)

    async def collect_all(self, symbols: Optional[List[str]] = None):
        """Collect all data"""
        logger.info("=" * 60)
        logger.info("STARTING DAILY DATA COLLECTION")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        results = {
            'timestamp': datetime.now().isoformat(),
            'success': [],
            'failed': []
        }

        # 1. Collect Price Data & Funding Rate
        async with BinanceCollector() as binance:
            if symbols is None:
                symbols = await binance.get_all_symbols()
                logger.info(f"Found {len(symbols)} trading symbols")

            # Collect OHLCV
            logger.info("\n[1/5] Collecting Price Data (OHLCV)...")
            ohlcv_dir = self.data_root / "binance_futures_4h"
            ohlcv_dir.mkdir(exist_ok=True)

            for i, symbol in enumerate(symbols):
                try:
                    df = await binance.get_klines(symbol, "4h", 500)
                    if not df.empty:
                        # Append to existing file
                        filepath = ohlcv_dir / f"{symbol}.csv"
                        if filepath.exists():
                            existing = pd.read_csv(filepath)
                            existing['datetime'] = pd.to_datetime(existing['datetime'])
                            df = pd.concat([existing, df]).drop_duplicates(subset='datetime').sort_values('datetime')
                        df.to_csv(filepath, index=False)
                        results['success'].append(f"OHLCV:{symbol}")

                    if (i + 1) % 50 == 0:
                        logger.info(f"  Progress: {i+1}/{len(symbols)}")
                    await asyncio.sleep(0.1)  # Rate limiting
                except Exception as e:
                    results['failed'].append(f"OHLCV:{symbol}:{str(e)}")

            logger.info(f"  Completed: {len([s for s in results['success'] if s.startswith('OHLCV')])} symbols")

            # Collect Funding Rate
            logger.info("\n[2/5] Collecting Funding Rate...")
            funding_dir = self.data_root / "binance_funding_rate"
            funding_dir.mkdir(exist_ok=True)

            for i, symbol in enumerate(symbols[:100]):  # Top 100 symbols
                try:
                    df = await binance.get_funding_rate(symbol, 500)
                    if not df.empty:
                        filepath = funding_dir / f"{symbol}.csv"
                        if filepath.exists():
                            existing = pd.read_csv(filepath)
                            existing['datetime'] = pd.to_datetime(existing['datetime'])
                            df = pd.concat([existing, df]).drop_duplicates(subset='datetime').sort_values('datetime')
                        df.to_csv(filepath, index=False)
                        results['success'].append(f"FUNDING:{symbol}")
                    await asyncio.sleep(0.1)
                except Exception as e:
                    results['failed'].append(f"FUNDING:{symbol}:{str(e)}")

            logger.info(f"  Completed: {len([s for s in results['success'] if s.startswith('FUNDING')])} symbols")

        # 2. Collect Fear & Greed
        logger.info("\n[3/5] Collecting Fear & Greed Index...")
        fg_collector = FearGreedCollector()
        fg_df = await fg_collector.fetch()
        if not fg_df.empty:
            filepath = self.data_root / "FEAR_GREED_INDEX_updated.csv"
            if filepath.exists():
                existing = pd.read_csv(filepath)
                existing['datetime'] = pd.to_datetime(existing['datetime'])
                fg_df = pd.concat([existing, fg_df]).drop_duplicates(subset='datetime').sort_values('datetime')
            fg_df.to_csv(filepath, index=False)
            results['success'].append("FEAR_GREED")
            logger.info(f"  Latest: {fg_df['fear_greed'].iloc[-1]} ({fg_df['classification'].iloc[-1]})")
        else:
            results['failed'].append("FEAR_GREED")

        # 3. Collect Macro Data
        logger.info("\n[4/5] Collecting Macro Data...")
        macro_collector = MacroCollector()
        macro_data = await macro_collector.fetch(days=30)

        macro_dir = self.data_root / "macro"
        macro_dir.mkdir(exist_ok=True)

        for name, df in macro_data.items():
            if not df.empty:
                filepath = macro_dir / f"{name}.csv"
                if filepath.exists():
                    existing = pd.read_csv(filepath)
                    existing['datetime'] = pd.to_datetime(existing['datetime'])
                    df = pd.concat([existing, df]).drop_duplicates(subset='datetime').sort_values('datetime')
                df.to_csv(filepath, index=False)
                results['success'].append(f"MACRO:{name}")

        logger.info(f"  Completed: {len([s for s in results['success'] if s.startswith('MACRO')])} indicators")

        # 4. Calculate BTC Correlation (done during signal generation)
        logger.info("\n[5/5] BTC Correlation - Calculated on-demand from price data")
        results['success'].append("BTC_CORRELATION:calculated")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("COLLECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Success: {len(results['success'])} items")
        logger.info(f"Failed: {len(results['failed'])} items")

        if results['failed']:
            logger.warning("Failed items:")
            for item in results['failed'][:10]:
                logger.warning(f"  - {item}")

        # Save collection log
        log_path = self.data_root / "collection_history.json"
        history = []
        if log_path.exists():
            with open(log_path, 'r') as f:
                history = json.load(f)
        history.append(results)
        history = history[-100:]  # Keep last 100 runs
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)

        return results


class SignalGenerator:
    """Generate Multi-Factor signals from collected data"""

    def __init__(self):
        self.data_root = DATA_ROOT

    def load_price(self, symbol: str) -> pd.DataFrame:
        filepath = self.data_root / "binance_futures_4h" / f"{symbol}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            for col in ['datetime', 'timestamp', 'date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col).sort_index()
                    break
            return df
        return pd.DataFrame()

    def load_fear_greed(self) -> pd.DataFrame:
        filepath = self.data_root / "FEAR_GREED_INDEX_updated.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            for col in ['datetime', 'timestamp', 'date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col).sort_index()
                    break
            # Rename value column to fear_greed
            if 'value' in df.columns:
                df = df.rename(columns={'value': 'fear_greed'})
            elif 'close' in df.columns:
                df = df.rename(columns={'close': 'fear_greed'})
            return df
        return pd.DataFrame()

    def load_macro(self) -> pd.DataFrame:
        macro_dir = self.data_root / "macro"
        dfs = []
        for name in ['DXY', 'VIX', 'SP500']:
            filepath = macro_dir / f"{name}.csv"
            if filepath.exists():
                df = pd.read_csv(filepath)
                for col in ['datetime', 'timestamp', 'date']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col).sort_index()
                        break
                if 'close' in df.columns:
                    dfs.append(df[['close']].rename(columns={'close': name.lower()}))
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    def load_funding(self, symbol: str) -> pd.DataFrame:
        filepath = self.data_root / "binance_funding_rate" / f"{symbol}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            for col in ['datetime', 'timestamp', 'date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col).sort_index()
                    break
            return df
        return pd.DataFrame()

    def generate_signal(self, symbol: str) -> Dict:
        """Generate multi-factor signal for a symbol"""
        df = self.load_price(symbol)
        if df.empty or len(df) < 200:
            return {'symbol': symbol, 'signal': 0, 'error': 'Insufficient data'}

        btc_df = self.load_price("BTCUSDT")
        fg_df = self.load_fear_greed()
        macro_df = self.load_macro()
        funding_df = self.load_funding(symbol)

        scores = {}

        # 1. Technical Score
        close = df['close']
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        ema200 = close.ewm(span=200).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        latest = df.iloc[-1]
        latest_ema20 = ema20.iloc[-1]
        latest_ema50 = ema50.iloc[-1]
        latest_ema200 = ema200.iloc[-1]
        latest_rsi = rsi.iloc[-1]

        tech_score = 0
        if latest['close'] > latest_ema20 > latest_ema50 > latest_ema200:
            tech_score = 2
        elif latest['close'] > latest_ema50:
            tech_score = 1
        elif latest['close'] < latest_ema20 < latest_ema50 < latest_ema200:
            tech_score = -2
        elif latest['close'] < latest_ema50:
            tech_score = -1

        if latest_rsi < 30:
            tech_score += 1.5
        elif latest_rsi < 40:
            tech_score += 0.5
        elif latest_rsi > 70:
            tech_score -= 1.5
        elif latest_rsi > 60:
            tech_score -= 0.5

        scores['technical'] = tech_score * 1.5

        # 2. Fear & Greed Score
        if not fg_df.empty:
            latest_fg = fg_df['fear_greed'].iloc[-1]
            if latest_fg < 20:
                fg_score = 2
            elif latest_fg < 35:
                fg_score = 1
            elif latest_fg > 80:
                fg_score = -2
            elif latest_fg > 65:
                fg_score = -1
            else:
                fg_score = 0
            scores['fear_greed'] = fg_score * 1.2
        else:
            scores['fear_greed'] = 0

        # 3. BTC Correlation Score
        if not btc_df.empty:
            btc_close = btc_df['close']
            btc_ret = btc_close.pct_change(20).iloc[-1]
            btc_ema50 = btc_close.ewm(span=50).mean().iloc[-1]
            btc_ema200 = btc_close.ewm(span=200).mean().iloc[-1]

            if btc_close.iloc[-1] > btc_ema50 > btc_ema200 and btc_ret > 0.1:
                btc_score = 2
            elif btc_close.iloc[-1] > btc_ema50 and btc_ret > 0.03:
                btc_score = 1
            elif btc_close.iloc[-1] < btc_ema50 < btc_ema200 and btc_ret < -0.1:
                btc_score = -2
            elif btc_close.iloc[-1] < btc_ema50 and btc_ret < -0.03:
                btc_score = -1
            else:
                btc_score = 0
            scores['btc_correlation'] = btc_score * 1.0
        else:
            scores['btc_correlation'] = 0

        # 4. Macro Score
        if not macro_df.empty:
            macro_score = 0
            if 'dxy' in macro_df.columns:
                dxy = macro_df['dxy'].dropna()
                if len(dxy) > 50:
                    dxy_ma = dxy.rolling(50).mean().iloc[-1]
                    dxy_latest = dxy.iloc[-1]
                    if dxy_latest < dxy_ma * 0.98:
                        macro_score += 1
                    elif dxy_latest > dxy_ma * 1.02:
                        macro_score -= 1

            if 'vix' in macro_df.columns:
                vix = macro_df['vix'].dropna()
                if len(vix) > 0:
                    vix_latest = vix.iloc[-1]
                    if vix_latest > 30:
                        macro_score += 1
                    elif vix_latest > 25:
                        macro_score += 0.5
                    elif vix_latest < 15:
                        macro_score -= 0.5

            scores['macro'] = macro_score * 1.0
        else:
            scores['macro'] = 0

        # 5. Funding Rate Score
        if not funding_df.empty:
            latest_funding = funding_df['funding_rate'].iloc[-1]
            avg_funding = funding_df['funding_rate'].tail(10).mean()

            if latest_funding < -0.001:  # Negative funding = bullish
                funding_score = 1.5
            elif latest_funding < 0:
                funding_score = 0.5
            elif latest_funding > 0.001:  # High positive = bearish
                funding_score = -1.5
            elif latest_funding > 0.0005:
                funding_score = -0.5
            else:
                funding_score = 0
            scores['funding'] = funding_score * 1.0
        else:
            scores['funding'] = 0

        total_score = sum(scores.values())

        return {
            'symbol': symbol,
            'datetime': df.index[-1].isoformat(),
            'price': latest['close'],
            'signal': round(total_score, 2),
            'scores': scores,
            'rsi': round(latest_rsi, 1) if not pd.isna(latest_rsi) else None,
            'recommendation': 'STRONG_BUY' if total_score >= 4 else
                            'BUY' if total_score >= 2 else
                            'NEUTRAL' if total_score >= -2 else
                            'SELL' if total_score >= -4 else 'STRONG_SELL'
        }

    def generate_all_signals(self, symbols: List[str] = None) -> pd.DataFrame:
        """Generate signals for all symbols"""
        if symbols is None:
            ohlcv_dir = self.data_root / "binance_futures_4h"
            symbols = [f.stem for f in ohlcv_dir.glob("*.csv") if f.stem.endswith('USDT')]

        results = []
        for symbol in symbols:
            signal = self.generate_signal(symbol)
            results.append(signal)

        df = pd.DataFrame(results)
        df = df.sort_values('signal', ascending=False)

        # Save daily signals
        today = datetime.now().strftime('%Y-%m-%d')
        signals_dir = self.data_root / "daily_signals"
        signals_dir.mkdir(exist_ok=True)
        df.to_csv(signals_dir / f"signals_{today}.csv", index=False)

        return df


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Factor Data Collector')
    parser.add_argument('--collect', action='store_true', help='Collect data')
    parser.add_argument('--signals', action='store_true', help='Generate signals')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to process')
    args = parser.parse_args()

    if args.collect or (not args.collect and not args.signals):
        collector = DataCollector()
        await collector.collect_all(args.symbols)

    if args.signals or (not args.collect and not args.signals):
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING SIGNALS")
        logger.info("=" * 60)

        generator = SignalGenerator()
        signals = generator.generate_all_signals(args.symbols)

        logger.info(f"\nTop 10 BUY signals:")
        for _, row in signals.head(10).iterrows():
            logger.info(f"  {row['symbol']:<14} Signal={row['signal']:+.2f} ({row['recommendation']})")

        logger.info(f"\nTop 10 SELL signals:")
        for _, row in signals.tail(10).iterrows():
            logger.info(f"  {row['symbol']:<14} Signal={row['signal']:+.2f} ({row['recommendation']})")

        # Summary
        strong_buy = len(signals[signals['signal'] >= 4])
        buy = len(signals[(signals['signal'] >= 2) & (signals['signal'] < 4)])
        neutral = len(signals[(signals['signal'] > -2) & (signals['signal'] < 2)])
        sell = len(signals[(signals['signal'] <= -2) & (signals['signal'] > -4)])
        strong_sell = len(signals[signals['signal'] <= -4])

        logger.info(f"\nSignal Distribution:")
        logger.info(f"  STRONG_BUY:  {strong_buy}")
        logger.info(f"  BUY:         {buy}")
        logger.info(f"  NEUTRAL:     {neutral}")
        logger.info(f"  SELL:        {sell}")
        logger.info(f"  STRONG_SELL: {strong_sell}")


if __name__ == "__main__":
    asyncio.run(main())
