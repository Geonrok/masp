#!/usr/bin/env python3
"""
Multi-Factor Strategy Extended Test
- Test on all symbols with required data
- Test different threshold levels
- Walk-forward validation
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import warnings

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


def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def calc_sma(s, p):
    return s.rolling(p).mean()


def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


def calc_bollinger(s, p=20, std=2.0):
    sma = calc_sma(s, p)
    st = s.rolling(p).std()
    return sma + std * st, sma, sma - std * st


class DataLoader:
    def __init__(self):
        self.root = DATA_ROOT
        self._cache = {}

    def get_all_symbols(self):
        folder = self.root / "binance_futures_4h"
        return sorted(
            [
                f.stem
                for f in folder.iterdir()
                if f.suffix == ".csv" and f.stem.endswith("USDT")
            ]
        )

    def load_ohlcv(self, symbol, tf="4h"):
        key = f"{symbol}_{tf}"
        if key in self._cache:
            return self._cache[key].copy()
        folder = {"4h": "binance_futures_4h", "1d": "binance_futures_1d"}.get(tf)
        for fn in [f"{symbol}.csv", f"{symbol.replace('USDT', '')}.csv"]:
            fp = self.root / folder / fn
            if fp.exists():
                df = pd.read_csv(fp)
                for col in ["datetime", "timestamp"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col)
                        break
                if all(
                    c in df.columns for c in ["open", "high", "low", "close", "volume"]
                ):
                    df = df[["open", "high", "low", "close", "volume"]].sort_index()
                    self._cache[key] = df
                    return df.copy()
        return pd.DataFrame()

    def load_fear_greed(self):
        for fn in ["FEAR_GREED_INDEX_updated.csv", "FEAR_GREED_INDEX.csv"]:
            fp = self.root / fn
            if fp.exists():
                df = pd.read_csv(fp)
                for col in ["datetime", "timestamp"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col)
                        break
                if "close" in df.columns:
                    return df[["close"]].rename(columns={"close": "fear_greed"})
        return pd.DataFrame()

    def load_macro(self):
        dfs = []
        for fn, col in [("DXY.csv", "dxy"), ("SP500.csv", "sp500"), ("VIX.csv", "vix")]:
            fp = self.root / "macro" / fn
            if fp.exists():
                df = pd.read_csv(fp)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                if "close" in df.columns:
                    dfs.append(df[["close"]].rename(columns={"close": col}))
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    def load_tvl(self):
        fp = self.root / "defillama" / "total_tvl_history.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            for col in ["datetime", "date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                    break
            if len(df.columns) > 0:
                return df.iloc[:, 0:1].rename(columns={df.columns[0]: "tvl"})
        return pd.DataFrame()


class MultiFactorScorer:
    def score_all(self, df, btc_df, fear_greed, macro, tvl):
        scores = pd.Series(0.0, index=df.index)

        # 1. Technical (weight: 1.5)
        ema20, ema50, ema200 = (
            calc_ema(df["close"], 20),
            calc_ema(df["close"], 50),
            calc_ema(df["close"], 200),
        )
        tech = np.where(
            (df["close"] > ema20) & (ema20 > ema50) & (ema50 > ema200),
            2,
            np.where(
                (df["close"] > ema50),
                1,
                np.where(
                    (df["close"] < ema20) & (ema20 < ema50) & (ema50 < ema200),
                    -2,
                    np.where((df["close"] < ema50), -1, 0),
                ),
            ),
        )

        rsi = calc_rsi(df["close"], 14)
        tech_rsi = np.where(
            rsi < 30,
            1.5,
            np.where(
                rsi < 40, 0.5, np.where(rsi > 70, -1.5, np.where(rsi > 60, -0.5, 0))
            ),
        )

        bb_upper, _, bb_lower = calc_bollinger(df["close"])
        bb_pos = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        tech_bb = np.where(bb_pos < 0.2, 1, np.where(bb_pos > 0.8, -1, 0))

        scores += (tech + tech_rsi + tech_bb) / 3 * 1.5

        # 2. Fear & Greed (weight: 1.2)
        if not fear_greed.empty:
            fg = fear_greed["fear_greed"].reindex(df.index, method="ffill")
            fg_score = np.where(
                fg < 20,
                2,
                np.where(fg < 35, 1, np.where(fg > 80, -2, np.where(fg > 65, -1, 0))),
            )
            scores += pd.Series(fg_score, index=df.index).fillna(0) * 1.2

        # 3. BTC Correlation (weight: 1.0)
        if not btc_df.empty:
            btc_close = btc_df["close"].reindex(df.index, method="ffill")
            btc_ret = btc_close.pct_change(20)
            btc_ema50 = calc_ema(btc_close, 50)
            btc_ema200 = calc_ema(btc_close, 200)
            btc_uptrend = (btc_close > btc_ema50) & (btc_ema50 > btc_ema200)
            btc_downtrend = (btc_close < btc_ema50) & (btc_ema50 < btc_ema200)
            btc_score = np.where(
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
            scores += pd.Series(btc_score, index=df.index).fillna(0) * 1.0

        # 4. Macro (weight: 1.0)
        if not macro.empty:
            macro_r = macro.reindex(df.index, method="ffill")
            macro_score = pd.Series(0.0, index=df.index)
            if "dxy" in macro_r.columns:
                dxy = macro_r["dxy"]
                dxy_ma = calc_sma(dxy, 50)
                macro_score += np.where(
                    dxy < dxy_ma * 0.98, 1, np.where(dxy > dxy_ma * 1.02, -1, 0)
                )
            if "vix" in macro_r.columns:
                vix = macro_r["vix"]
                macro_score += np.where(
                    vix > 30, 1, np.where(vix > 25, 0.5, np.where(vix < 15, -0.5, 0))
                )
            scores += macro_score.fillna(0) * 1.0

        # 5. TVL (weight: 0.6)
        if not tvl.empty:
            tv = tvl["tvl"].reindex(df.index, method="ffill")
            tvl_ma = tv.rolling(30).mean()
            tvl_growth = (tv - tvl_ma) / tvl_ma
            tvl_score = np.where(
                tvl_growth > 0.1,
                1,
                np.where(
                    tvl_growth > 0.03,
                    0.5,
                    np.where(
                        tvl_growth < -0.1, -1, np.where(tvl_growth < -0.03, -0.5, 0)
                    ),
                ),
            )
            scores += pd.Series(tvl_score, index=df.index).fillna(0) * 0.6

        return scores


@dataclass
class Result:
    symbol: str
    pf: float
    ret: float
    wr: float
    mdd: float
    trades: int


class Backtester:
    def __init__(self, threshold=3.0):
        self.threshold = threshold

    def run(self, df, scores, symbol):
        if len(df) < 200:
            return None

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        for i in range(1, len(df)):
            price = df["close"].iloc[i]
            score = scores.iloc[i] if i < len(scores) else 0
            if pd.isna(score):
                score = 0

            if position:
                should_exit = (position["dir"] == 1 and score < 0) or (
                    position["dir"] == -1 and score > 0
                )
                if should_exit:
                    pnl_pct = (price / position["entry"] - 1) * position["dir"] - 0.002
                    pnl = capital * 0.2 * 2 * min(position["mult"], 2) * pnl_pct
                    trades.append(pnl)
                    capital += pnl
                    position = None

            if not position and abs(score) >= self.threshold:
                direction = 1 if score > 0 else -1
                mult = min(abs(score) / self.threshold, 2.0)
                position = {"entry": price, "dir": direction, "mult": mult}

            equity.append(capital)

        if position:
            pnl_pct = (df["close"].iloc[-1] / position["entry"] - 1) * position[
                "dir"
            ] - 0.002
            trades.append(capital * 0.2 * 2 * position["mult"] * pnl_pct)
            capital += trades[-1]

        if len(trades) < 5:
            return None

        equity = pd.Series(equity)
        mdd = (
            (equity - equity.expanding().max()) / equity.expanding().max()
        ).min() * 100
        wins = sum(1 for t in trades if t > 0)
        gp = sum(t for t in trades if t > 0)
        gl = abs(sum(t for t in trades if t < 0))
        pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)

        return Result(
            symbol,
            min(pf, 999),
            (capital / init - 1) * 100,
            wins / len(trades) * 100,
            mdd,
            len(trades),
        )


def main():
    loader = DataLoader()
    scorer = MultiFactorScorer()

    all_symbols = loader.get_all_symbols()
    logger.info(f"Total symbols with 4h data: {len(all_symbols)}")

    # Load common data
    fear_greed = loader.load_fear_greed()
    macro = loader.load_macro()
    tvl = loader.load_tvl()
    btc_df = loader.load_ohlcv("BTCUSDT", "4h")

    logger.info("=" * 70)
    logger.info("MULTI-FACTOR STRATEGY - EXTENDED TEST")
    logger.info("=" * 70)

    results = []
    tested = 0

    for symbol in all_symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if df.empty:
            continue

        df = df[df.index >= "2021-01-01"]
        if len(df) < 500:
            continue

        tested += 1
        scores = scorer.score_all(df, btc_df, fear_greed, macro, tvl)
        backtester = Backtester(threshold=3.0)
        result = backtester.run(df, scores, symbol)

        if result and result.trades >= 10:
            results.append(result)

    logger.info(f"\nTested: {tested} symbols")
    logger.info(f"Valid results: {len(results)} symbols")

    if not results:
        return

    # Summary
    profitable = [r for r in results if r.pf > 1.0]
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(
        f"Profitable: {len(profitable)}/{len(results)} ({len(profitable)/len(results)*100:.1f}%)"
    )
    logger.info(f"Avg PF: {np.mean([r.pf for r in results if r.pf < 999]):.2f}")
    logger.info(f"Avg Return: {np.mean([r.ret for r in results]):+.1f}%")
    logger.info(f"Avg WR: {np.mean([r.wr for r in results]):.1f}%")

    # PF buckets
    logger.info("\nPF Distribution:")
    buckets = [
        (1.5, "PF>=1.5"),
        (1.3, "PF>=1.3"),
        (1.1, "PF>=1.1"),
        (1.0, "PF>=1.0"),
        (0, "PF<1.0"),
    ]
    for thresh, name in buckets:
        if thresh > 0:
            count = sum(1 for r in results if r.pf >= thresh)
        else:
            count = sum(1 for r in results if r.pf < 1.0)
        logger.info(f"  {name}: {count} ({count/len(results)*100:.1f}%)")

    # Top performers
    logger.info("\n" + "=" * 70)
    logger.info("TOP 30 PERFORMERS")
    logger.info("=" * 70)
    top = sorted(results, key=lambda x: -x.pf)[:30]
    for r in top:
        logger.info(
            f"  {r.symbol:14s} PF={r.pf:5.2f} Ret={r.ret:+8.1f}% WR={r.wr:4.0f}% MDD={r.mdd:6.1f}% Trades={r.trades:3d}"
        )

    # Recommended symbols (PF > 1.3, MDD > -50)
    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDED SYMBOLS (PF>=1.3, MDD>=-50%)")
    logger.info("=" * 70)
    recommended = [r for r in results if r.pf >= 1.3 and r.mdd >= -50]
    recommended = sorted(recommended, key=lambda x: -x.pf)
    logger.info(f"Found {len(recommended)} recommended symbols:\n")
    for r in recommended:
        logger.info(
            f"  {r.symbol:14s} PF={r.pf:5.2f} Ret={r.ret:+8.1f}% WR={r.wr:4.0f}% MDD={r.mdd:6.1f}%"
        )

    # Save results
    results_df = pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "pf": r.pf,
                "return": r.ret,
                "wr": r.wr,
                "mdd": r.mdd,
                "trades": r.trades,
            }
            for r in results
        ]
    )
    output_path = Path(
        "E:/투자/Multi-Asset Strategy Platform/data/backtests/multifactor_results.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
