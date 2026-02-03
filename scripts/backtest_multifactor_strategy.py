#!/usr/bin/env python3
"""
Multi-Factor Strategy - Combining ALL Available Data

Factors:
1. Price Patterns (EMA, RSI, Bollinger) - Technical
2. Fear & Greed Index - Sentiment
3. Funding Rate - Market Positioning
4. BTC Correlation - Market Leader
5. On-chain (Active Addresses, Hash Rate) - Network Health
6. Google Trends - Social/Retail Interest
7. Macro (DXY, SP500, VIX) - Global Risk
8. TVL (DeFiLlama) - DeFi Health
9. Long/Short Ratio - Crowd Positioning
10. Open Interest - Market Leverage

Scoring System:
- Each factor generates a score from -2 to +2
- Total score determines position sizing and direction
- Strong consensus = larger position
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_ROOT = Path("E:/data/crypto_ohlcv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================
def calc_ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def calc_sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p).mean()


def calc_rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


def calc_bollinger(s: pd.Series, p: int = 20, std: float = 2.0):
    sma = calc_sma(s, p)
    st = s.rolling(p).std()
    return sma + std * st, sma, sma - std * st


# ============================================================================
# DATA LOADER
# ============================================================================
class MultiFactorDataLoader:
    def __init__(self):
        self.root = DATA_ROOT
        self._cache = {}

    def load_ohlcv(self, symbol: str, tf: str = "4h") -> pd.DataFrame:
        key = f"{symbol}_{tf}"
        if key in self._cache:
            return self._cache[key].copy()

        folder = {"4h": "binance_futures_4h", "1d": "binance_futures_1d"}.get(tf)
        for fn in [f"{symbol}.csv", f"{symbol.replace('USDT', '')}.csv"]:
            fp = self.root / folder / fn
            if fp.exists():
                df = pd.read_csv(fp)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                if all(
                    c in df.columns for c in ["open", "high", "low", "close", "volume"]
                ):
                    df = df[["open", "high", "low", "close", "volume"]].sort_index()
                    self._cache[key] = df
                    return df.copy()
        return pd.DataFrame()

    def load_fear_greed(self) -> pd.DataFrame:
        for fn in ["FEAR_GREED_INDEX_updated.csv", "FEAR_GREED_INDEX.csv"]:
            fp = self.root / fn
            if fp.exists():
                df = pd.read_csv(fp)
                if "timestamp" in df.columns:
                    df["datetime"] = pd.to_datetime(df["timestamp"])
                elif "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
                if "close" in df.columns:
                    return df[["close"]].rename(columns={"close": "fear_greed"})
        return pd.DataFrame()

    def load_funding_rate(self, symbol: str) -> pd.DataFrame:
        for fn in [f"{symbol}_funding_full.csv", f"{symbol}_funding.csv"]:
            fp = self.root / "binance_funding_rate" / fn
            if fp.exists():
                df = pd.read_csv(fp)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                cols = [
                    c for c in df.columns if "fund" in c.lower() or "rate" in c.lower()
                ]
                if cols:
                    return df[[cols[0]]].rename(columns={cols[0]: "funding"})
        return pd.DataFrame()

    def load_onchain(self) -> pd.DataFrame:
        dfs = []
        for fn, col in [
            ("btc_active_addresses.csv", "active_addr"),
            ("btc_hash_rate.csv", "hash_rate"),
            ("btc_tx_count.csv", "tx_count"),
        ]:
            fp = self.root / "onchain" / fn
            if fp.exists():
                df = pd.read_csv(fp)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                if len(df.columns) > 0:
                    df = df.iloc[:, 0:1]
                    df.columns = [col]
                    dfs.append(df)
        if dfs:
            return pd.concat(dfs, axis=1)
        return pd.DataFrame()

    def load_google_trends(self) -> pd.DataFrame:
        fp = self.root / "sentiment" / "google_trends_bitcoin.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            if "bitcoin" in df.columns:
                return df[["bitcoin"]].rename(columns={"bitcoin": "gtrends"})
        return pd.DataFrame()

    def load_macro(self) -> pd.DataFrame:
        dfs = []
        for fn, col in [
            ("DXY.csv", "dxy"),
            ("SP500.csv", "sp500"),
            ("VIX.csv", "vix"),
            ("US10Y.csv", "us10y"),
        ]:
            fp = self.root / "macro" / fn
            if fp.exists():
                df = pd.read_csv(fp)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                if "close" in df.columns:
                    dfs.append(df[["close"]].rename(columns={"close": col}))
        if dfs:
            return pd.concat(dfs, axis=1)
        return pd.DataFrame()

    def load_tvl(self) -> pd.DataFrame:
        fp = self.root / "defillama" / "total_tvl_history.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            cols = [c for c in df.columns if "tvl" in c.lower() or "value" in c.lower()]
            if cols:
                return df[[cols[0]]].rename(columns={cols[0]: "tvl"})
            if len(df.columns) > 0:
                return df.iloc[:, 0:1].rename(columns={df.columns[0]: "tvl"})
        return pd.DataFrame()

    def load_long_short_ratio(self, symbol: str) -> pd.DataFrame:
        for fn in [f"{symbol}_lsratio_full.csv", f"{symbol}_lsratio.csv"]:
            fp = self.root / "binance_long_short_ratio" / fn
            if fp.exists():
                df = pd.read_csv(fp)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                cols = [
                    c for c in df.columns if "ratio" in c.lower() or "long" in c.lower()
                ]
                if cols:
                    return df[[cols[0]]].rename(columns={cols[0]: "ls_ratio"})
        return pd.DataFrame()

    def load_open_interest(self, symbol: str) -> pd.DataFrame:
        for fn in [f"{symbol}_oi_full.csv", f"{symbol}_oi.csv"]:
            fp = self.root / "binance_open_interest" / fn
            if fp.exists():
                df = pd.read_csv(fp)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                cols = [
                    c
                    for c in df.columns
                    if "oi" in c.lower()
                    or "interest" in c.lower()
                    or "value" in c.lower()
                ]
                if cols:
                    return df[[cols[0]]].rename(columns={cols[0]: "oi"})
        return pd.DataFrame()


# ============================================================================
# MULTI-FACTOR SCORING
# ============================================================================
class MultiFactorScorer:
    """
    Each factor returns a score from -2 to +2:
    -2 = Strong Bearish
    -1 = Bearish
     0 = Neutral
    +1 = Bullish
    +2 = Strong Bullish
    """

    def score_technical(self, df: pd.DataFrame) -> pd.Series:
        """Price patterns: EMA, RSI, Bollinger"""
        scores = pd.Series(0.0, index=df.index)

        # EMA trend
        ema20 = calc_ema(df["close"], 20)
        ema50 = calc_ema(df["close"], 50)
        ema200 = calc_ema(df["close"], 200)

        # EMA alignment score
        scores += np.where(
            (df["close"] > ema20) & (ema20 > ema50) & (ema50 > ema200),
            2,
            np.where(
                (df["close"] > ema50) & (ema50 > ema200),
                1,
                np.where(
                    (df["close"] < ema20) & (ema20 < ema50) & (ema50 < ema200),
                    -2,
                    np.where((df["close"] < ema50) & (ema50 < ema200), -1, 0),
                ),
            ),
        )

        # RSI
        rsi = calc_rsi(df["close"], 14)
        scores += np.where(
            rsi < 30,
            1.5,
            np.where(
                rsi < 40, 0.5, np.where(rsi > 70, -1.5, np.where(rsi > 60, -0.5, 0))
            ),
        )

        # Bollinger position
        bb_upper, bb_mid, bb_lower = calc_bollinger(df["close"])
        bb_pos = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        scores += np.where(bb_pos < 0.2, 1, np.where(bb_pos > 0.8, -1, 0))

        return scores / 3  # Normalize to roughly -2 to +2

    def score_fear_greed(self, fg: pd.Series) -> pd.Series:
        """Fear & Greed: Contrarian - buy fear, sell greed"""
        scores = pd.Series(0.0, index=fg.index)
        scores = np.where(
            fg < 20,
            2,  # Extreme fear = strong buy
            np.where(
                fg < 35,
                1,  # Fear = buy
                np.where(
                    fg > 80, -2, np.where(fg > 65, -1, 0)  # Extreme greed = strong sell
                ),
            ),
        )  # Greed = sell
        return pd.Series(scores, index=fg.index)

    def score_funding(self, funding: pd.Series) -> pd.Series:
        """Funding rate: Negative = crowded shorts = bullish"""
        scores = pd.Series(0.0, index=funding.index)
        funding_ma = funding.rolling(6).mean()  # Smooth
        scores = np.where(
            funding_ma < -0.001,
            2,  # Very negative = strong buy
            np.where(
                funding_ma < -0.0003,
                1,  # Negative = buy
                np.where(
                    funding_ma > 0.001,
                    -2,  # Very positive = strong sell
                    np.where(funding_ma > 0.0003, -1, 0),
                ),
            ),
        )
        return pd.Series(scores, index=funding.index)

    def score_btc_correlation(self, df: pd.DataFrame, btc: pd.DataFrame) -> pd.Series:
        """BTC momentum as market leader"""
        scores = pd.Series(0.0, index=df.index)
        if btc.empty:
            return scores

        btc_close = btc["close"].reindex(df.index, method="ffill")
        btc_ret = btc_close.pct_change(20)  # 20-bar momentum
        btc_ema50 = calc_ema(btc_close, 50)
        btc_ema200 = calc_ema(btc_close, 200)

        # BTC trend
        btc_uptrend = (btc_close > btc_ema50) & (btc_ema50 > btc_ema200)
        btc_downtrend = (btc_close < btc_ema50) & (btc_ema50 < btc_ema200)

        scores = np.where(
            btc_uptrend & (btc_ret > 0.1),
            2,
            np.where(
                btc_uptrend & (btc_ret > 0.03),
                1,
                np.where(
                    btc_downtrend & (btc_ret < -0.1),
                    -2,
                    np.where(btc_downtrend & (btc_ret < -0.03), -1, 0),
                ),
            ),
        )
        return pd.Series(scores, index=df.index)

    def score_onchain(self, onchain: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
        """On-chain: Active addresses, hash rate growing = bullish"""
        scores = pd.Series(0.0, index=idx)
        if onchain.empty:
            return scores

        onchain = onchain.reindex(idx, method="ffill")

        if "active_addr" in onchain.columns:
            addr = onchain["active_addr"]
            addr_ma = addr.rolling(30).mean()
            addr_growth = (addr - addr_ma) / addr_ma
            scores += np.where(
                addr_growth > 0.1, 1, np.where(addr_growth < -0.1, -1, 0)
            )

        if "hash_rate" in onchain.columns:
            hr = onchain["hash_rate"]
            hr_ma = hr.rolling(30).mean()
            hr_growth = (hr - hr_ma) / hr_ma
            scores += np.where(
                hr_growth > 0.05, 0.5, np.where(hr_growth < -0.05, -0.5, 0)
            )

        return scores

    def score_social(self, gtrends: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
        """Google Trends: Contrarian - high search = retail FOMO = sell"""
        scores = pd.Series(0.0, index=idx)
        if gtrends.empty:
            return scores

        gt = gtrends.reindex(idx, method="ffill")
        gt.rolling(4).mean()  # Weekly average
        gt_pct = gt.rolling(52).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )

        scores = np.where(
            gt_pct > 90,
            -1.5,  # Extreme interest = sell
            np.where(
                gt_pct > 75,
                -0.5,
                np.where(
                    gt_pct < 10,
                    1.5,  # Low interest = buy
                    np.where(gt_pct < 25, 0.5, 0),
                ),
            ),
        )
        return pd.Series(scores, index=idx)

    def score_macro(self, macro: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
        """Macro: DXY down, SP500 up, VIX low = risk-on = bullish"""
        scores = pd.Series(0.0, index=idx)
        if macro.empty:
            return scores

        macro = macro.reindex(idx, method="ffill")

        # DXY (inverse - strong dollar = bad for crypto)
        if "dxy" in macro.columns:
            dxy = macro["dxy"]
            dxy_ma = calc_sma(dxy, 50)
            scores += np.where(
                dxy < dxy_ma * 0.98, 1, np.where(dxy > dxy_ma * 1.02, -1, 0)
            )

        # SP500 (risk-on = good for crypto)
        if "sp500" in macro.columns:
            sp = macro["sp500"]
            sp_ma = calc_sma(sp, 50)
            scores += np.where(
                sp > sp_ma * 1.02, 0.5, np.where(sp < sp_ma * 0.98, -0.5, 0)
            )

        # VIX (high = fear = contrarian buy)
        if "vix" in macro.columns:
            vix = macro["vix"]
            scores += np.where(
                vix > 30, 1, np.where(vix > 25, 0.5, np.where(vix < 15, -0.5, 0))
            )

        return scores

    def score_tvl(self, tvl: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
        """TVL: Growing = bullish for crypto/DeFi"""
        scores = pd.Series(0.0, index=idx)
        if tvl.empty:
            return scores

        tvl = tvl.reindex(idx, method="ffill")
        tvl_ma = tvl.rolling(30).mean()
        tvl_growth = (tvl - tvl_ma) / tvl_ma

        scores = np.where(
            tvl_growth > 0.1,
            1,
            np.where(
                tvl_growth > 0.03,
                0.5,
                np.where(tvl_growth < -0.1, -1, np.where(tvl_growth < -0.03, -0.5, 0)),
            ),
        )
        return pd.Series(scores, index=idx)

    def score_long_short(self, ls: pd.Series) -> pd.Series:
        """Long/Short ratio: Contrarian - crowded longs = sell"""
        scores = pd.Series(0.0, index=ls.index)
        ls_ma = ls.rolling(6).mean()

        scores = np.where(
            ls_ma > 2.0,
            -1.5,  # Very crowded longs
            np.where(
                ls_ma > 1.5,
                -0.5,
                np.where(
                    ls_ma < 0.5, 1.5, np.where(ls_ma < 0.7, 0.5, 0)  # Crowded shorts
                ),
            ),
        )
        return pd.Series(scores, index=ls.index)

    def score_open_interest(self, oi: pd.Series, price: pd.Series) -> pd.Series:
        """OI: Rising OI + rising price = trend confirmation"""
        scores = pd.Series(0.0, index=oi.index)

        oi_change = oi.pct_change(6)
        price_change = price.pct_change(6)

        # Rising OI + rising price = bullish confirmation
        scores = np.where(
            (oi_change > 0.05) & (price_change > 0.02),
            1,
            np.where(
                (oi_change > 0.05) & (price_change < -0.02),
                -1,  # Rising OI + falling price = bearish
                np.where(
                    (oi_change < -0.05) & (price_change > 0.02),
                    0.5,  # Falling OI + rising price = short squeeze
                    0,
                ),
            ),
        )
        return pd.Series(scores, index=oi.index)


# ============================================================================
# BACKTEST
# ============================================================================
@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: int
    size_mult: float  # Position size multiplier based on score
    pnl: float = 0.0


@dataclass
class Result:
    symbol: str
    pf: float
    ret: float
    wr: float
    mdd: float
    trades: int
    avg_score: float


class MultiFactorBacktester:
    def __init__(self, threshold: float = 3.0, max_threshold: float = 6.0):
        self.threshold = threshold  # Minimum score to enter
        self.max_threshold = max_threshold  # Score for max position

    def run(self, df: pd.DataFrame, scores: pd.Series, symbol: str) -> Optional[Result]:
        if len(df) < 200 or scores.isna().all():
            return None

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        commission = 0.001
        slippage = 0.0005
        base_leverage = 2
        base_size = 0.2

        for i in range(1, len(df)):
            price = df["close"].iloc[i]
            score = scores.iloc[i] if i < len(scores) else 0

            if pd.isna(score):
                score = 0

            # Exit logic
            if position:
                # Exit when score reverses or crosses zero
                should_exit = False
                if position.direction == 1 and score < 0:
                    should_exit = True
                elif position.direction == -1 and score > 0:
                    should_exit = True

                if should_exit:
                    exit_p = price * (1 - slippage * position.direction)
                    pnl_pct = (exit_p / position.entry_price - 1) * position.direction
                    pnl_pct -= commission * 2
                    leverage = base_leverage * position.size_mult
                    size = base_size * position.size_mult
                    pnl = capital * size * leverage * pnl_pct
                    position.pnl = pnl
                    trades.append(position)
                    capital += pnl
                    position = None

            # Entry logic
            if not position and abs(score) >= self.threshold:
                direction = 1 if score > 0 else -1
                # Size based on score strength
                size_mult = min(abs(score) / self.threshold, 2.0)
                entry_p = price * (1 + slippage * direction)
                position = Trade(df.index[i], entry_p, direction, size_mult)

            equity.append(capital)

        # Close open position
        if position:
            pnl_pct = (
                df["close"].iloc[-1] / position.entry_price - 1
            ) * position.direction
            pnl_pct -= commission * 2
            position.pnl = (
                capital
                * base_size
                * position.size_mult
                * base_leverage
                * position.size_mult
                * pnl_pct
            )
            trades.append(position)
            capital += position.pnl

        if len(trades) < 5:
            return None

        equity = pd.Series(equity)
        peak = equity.expanding().max()
        mdd = ((equity - peak) / peak).min() * 100

        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / len(trades) * 100

        gp = sum(t.pnl for t in trades if t.pnl > 0)
        gl = abs(sum(t.pnl for t in trades if t.pnl < 0))
        pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)

        avg_score = scores.abs().mean()

        return Result(
            symbol,
            min(pf, 999),
            (capital / init - 1) * 100,
            wr,
            mdd,
            len(trades),
            avg_score,
        )


# ============================================================================
# MAIN
# ============================================================================
def main():
    loader = MultiFactorDataLoader()
    scorer = MultiFactorScorer()
    backtester = MultiFactorBacktester(threshold=3.0)

    # Symbols with most data available
    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "DOTUSDT",
        "LTCUSDT",
        "LINKUSDT",
    ]

    # Load common data once
    logger.info("Loading common data...")
    fear_greed = loader.load_fear_greed()
    onchain = loader.load_onchain()
    gtrends = loader.load_google_trends()
    macro = loader.load_macro()
    tvl = loader.load_tvl()
    btc_df = loader.load_ohlcv("BTCUSDT", "4h")

    logger.info(f"Fear&Greed: {len(fear_greed)} rows")
    logger.info(f"On-chain: {len(onchain)} rows")
    logger.info(f"Google Trends: {len(gtrends)} rows")
    logger.info(f"Macro: {len(macro)} rows")
    logger.info(f"TVL: {len(tvl)} rows")

    logger.info("\n" + "=" * 70)
    logger.info("MULTI-FACTOR STRATEGY BACKTEST")
    logger.info("Using ALL available data sources")
    logger.info("=" * 70)

    results = []

    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {symbol}")
        logger.info(f"{'='*50}")

        df = loader.load_ohlcv(symbol, "4h")
        if df.empty or len(df) < 500:
            logger.warning("  Insufficient price data")
            continue

        df = df[df.index >= "2020-01-01"]

        # Load symbol-specific data
        funding = loader.load_funding_rate(symbol)
        ls_ratio = loader.load_long_short_ratio(symbol)
        oi = loader.load_open_interest(symbol)

        # Calculate all factor scores
        logger.info("  Calculating factor scores...")

        # 1. Technical
        tech_score = scorer.score_technical(df)
        logger.info(f"    Technical: mean={tech_score.mean():.2f}")

        # 2. Fear & Greed
        if not fear_greed.empty:
            fg = fear_greed["fear_greed"].reindex(df.index, method="ffill")
            fg_score = scorer.score_fear_greed(fg)
            logger.info(f"    Fear&Greed: mean={fg_score.mean():.2f}")
        else:
            fg_score = pd.Series(0, index=df.index)

        # 3. Funding
        if not funding.empty:
            fund = funding["funding"].reindex(df.index, method="ffill")
            fund_score = scorer.score_funding(fund)
            logger.info(f"    Funding: mean={fund_score.mean():.2f}")
        else:
            fund_score = pd.Series(0, index=df.index)

        # 4. BTC Correlation
        btc_score = scorer.score_btc_correlation(df, btc_df)
        logger.info(f"    BTC Corr: mean={btc_score.mean():.2f}")

        # 5. On-chain
        oc_score = scorer.score_onchain(onchain, df.index)
        logger.info(f"    On-chain: mean={oc_score.mean():.2f}")

        # 6. Google Trends
        if not gtrends.empty:
            gt = gtrends["gtrends"]
            gt_score = scorer.score_social(gt, df.index)
            logger.info(f"    G.Trends: mean={gt_score.mean():.2f}")
        else:
            gt_score = pd.Series(0, index=df.index)

        # 7. Macro
        macro_score = scorer.score_macro(macro, df.index)
        logger.info(f"    Macro: mean={macro_score.mean():.2f}")

        # 8. TVL
        if not tvl.empty:
            tv = tvl["tvl"]
            tvl_score = scorer.score_tvl(tv, df.index)
            logger.info(f"    TVL: mean={tvl_score.mean():.2f}")
        else:
            tvl_score = pd.Series(0, index=df.index)

        # 9. Long/Short Ratio
        if not ls_ratio.empty:
            ls = ls_ratio["ls_ratio"].reindex(df.index, method="ffill")
            ls_score = scorer.score_long_short(ls)
            logger.info(f"    L/S Ratio: mean={ls_score.mean():.2f}")
        else:
            ls_score = pd.Series(0, index=df.index)

        # 10. Open Interest
        if not oi.empty:
            o = oi["oi"].reindex(df.index, method="ffill")
            oi_score = scorer.score_open_interest(o, df["close"])
            logger.info(f"    Open Int: mean={oi_score.mean():.2f}")
        else:
            oi_score = pd.Series(0, index=df.index)

        # Combine all scores with weights
        weights = {
            "technical": 1.5,  # Core signal
            "fear_greed": 1.2,  # Proven contrarian
            "funding": 1.0,
            "btc": 1.0,
            "onchain": 0.8,
            "social": 0.7,
            "macro": 1.0,
            "tvl": 0.6,
            "ls_ratio": 0.8,
            "oi": 0.7,
        }

        total_score = (
            tech_score * weights["technical"]
            + fg_score * weights["fear_greed"]
            + fund_score * weights["funding"]
            + btc_score * weights["btc"]
            + oc_score * weights["onchain"]
            + gt_score * weights["social"]
            + macro_score * weights["macro"]
            + tvl_score * weights["tvl"]
            + ls_score * weights["ls_ratio"]
            + oi_score * weights["oi"]
        )

        logger.info(
            f"\n  TOTAL SCORE: mean={total_score.mean():.2f}, "
            f"std={total_score.std():.2f}, "
            f"min={total_score.min():.2f}, max={total_score.max():.2f}"
        )

        # Run backtest
        result = backtester.run(df, total_score, symbol)

        if result:
            results.append(result)
            status = "OK" if result.pf > 1.0 else "FAIL"
            logger.info(
                f"\n  RESULT: [{status}] PF={result.pf:.2f}, Ret={result.ret:+.1f}%, "
                f"WR={result.wr:.0f}%, MDD={result.mdd:.1f}%, Trades={result.trades}"
            )
        else:
            logger.info("\n  RESULT: Insufficient trades")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("MULTI-FACTOR STRATEGY SUMMARY")
    logger.info("=" * 70)

    if results:
        profitable = sum(1 for r in results if r.pf > 1.0)
        avg_pf = np.mean([r.pf for r in results if r.pf < 999])
        avg_ret = np.mean([r.ret for r in results])

        logger.info(
            f"\nProfitable: {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)"
        )
        logger.info(f"Average PF: {avg_pf:.2f}")
        logger.info(f"Average Return: {avg_ret:+.1f}%")

        logger.info("\nPer-symbol results:")
        for r in sorted(results, key=lambda x: -x.pf):
            status = "OK" if r.pf > 1.0 else "X"
            logger.info(
                f"  [{status}] {r.symbol}: PF={r.pf:.2f}, Ret={r.ret:+.1f}%, "
                f"WR={r.wr:.0f}%, MDD={r.mdd:.1f}%"
            )

    return results


if __name__ == "__main__":
    results = main()
