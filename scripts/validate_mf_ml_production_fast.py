#!/usr/bin/env python3
"""
MF+ML Production-Ready Validation (Fast Version)
실제 매매와 동일한 환경으로 테스트 - 최적화 버전

테스트 항목:
1. Sequential Walk-Forward (실시간 재학습 시뮬레이션)
2. Monte Carlo Simulation (신뢰구간)
3. Stress Test (위기 기간)
4. Kelly Criterion (포지션 크기)
5. Execution Lag Test (신호 지연)
6. Drawdown Analysis
"""

from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

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


class ProductionCostModel:
    """실제 Binance Futures 비용"""

    def __init__(self):
        self.taker_fee = 0.0004
        self.slippage = 0.001
        self.funding_rate = 0.0001

    def calc_cost(self, bars_held: int) -> float:
        fee = self.taker_fee * 2
        slippage = self.slippage * 2
        funding = self.funding_rate * (bars_held / 2)
        return fee + slippage + funding


class DataLoader:
    def __init__(self):
        self._cache = {}

    def load_ohlcv(self, symbol: str) -> pd.DataFrame:
        if symbol in self._cache:
            return self._cache[symbol].copy()
        fp = DATA_ROOT / "binance_futures_4h" / f"{symbol}.csv"
        if not fp.exists():
            return pd.DataFrame()
        df = pd.read_csv(fp)
        for col in ["datetime", "timestamp", "date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col).sort_index()
                break
        self._cache[symbol] = df
        return df.copy()

    def load_fear_greed(self) -> pd.DataFrame:
        for fn in ["FEAR_GREED_INDEX_updated.csv", "FEAR_GREED_INDEX.csv"]:
            fp = DATA_ROOT / fn
            if fp.exists():
                df = pd.read_csv(fp)
                for col in ["datetime", "timestamp"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col)
                        break
                if "value" in df.columns:
                    return df[["value"]].rename(columns={"value": "fear_greed"})
        return pd.DataFrame()

    def load_macro(self) -> pd.DataFrame:
        dfs = []
        for fn, col in [("DXY.csv", "dxy"), ("VIX.csv", "vix")]:
            fp = DATA_ROOT / "macro" / fn
            if fp.exists():
                df = pd.read_csv(fp)
                for c in ["datetime", "timestamp"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c])
                        df = df.set_index(c)
                        break
                if "close" in df.columns:
                    dfs.append(df[["close"]].rename(columns={"close": col}))
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()


class MultiFactorSignal:
    def generate(
        self,
        df: pd.DataFrame,
        btc_df: pd.DataFrame,
        fear_greed: pd.DataFrame,
        macro: pd.DataFrame,
    ) -> pd.Series:
        scores = pd.Series(0.0, index=df.index)

        # Technical
        ema20, ema50 = calc_ema(df["close"], 20), calc_ema(df["close"], 50)
        tech = np.where(df["close"] > ema50, 1, np.where(df["close"] < ema50, -1, 0))
        rsi = calc_rsi(df["close"], 14)
        tech_rsi = np.where(rsi < 30, 1.5, np.where(rsi > 70, -1.5, 0))
        scores += (
            pd.Series(tech, index=df.index) + pd.Series(tech_rsi, index=df.index)
        ) * 1.5

        # Fear & Greed
        if not fear_greed.empty:
            fg = fear_greed["fear_greed"].reindex(df.index, method="ffill")
            fg_score = np.where(fg < 25, 2, np.where(fg > 75, -2, 0))
            scores += pd.Series(fg_score, index=df.index).fillna(0) * 1.2

        # BTC Correlation
        if not btc_df.empty:
            btc_close = btc_df["close"].reindex(df.index, method="ffill")
            btc_ema50 = calc_ema(btc_close, 50)
            btc_score = np.where(
                btc_close > btc_ema50, 1, np.where(btc_close < btc_ema50, -1, 0)
            )
            scores += pd.Series(btc_score, index=df.index).fillna(0) * 1.0

        # Macro
        if not macro.empty:
            macro_r = macro.reindex(df.index, method="ffill")
            if "vix" in macro_r.columns:
                vix = macro_r["vix"]
                scores += np.where(vix > 30, 1, np.where(vix < 15, -0.5, 0))

        return scores


class MLSignal:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        c = df["close"]
        v = df["volume"]
        for p in [5, 10, 20]:
            f[f"r{p}"] = c.pct_change(p)
            f[f"v{p}"] = v.pct_change(p)
        f["rsi"] = calc_rsi(c, 14)
        f["ema20"] = c / calc_ema(c, 20) - 1
        return f.replace([np.inf, -np.inf], np.nan).fillna(0)

    def train(self, df: pd.DataFrame, end_idx: int):
        feat = self._features(df)
        fut = df["close"].shift(-6) / df["close"] - 1
        tgt = np.where(fut > 0.02, 1, np.where(fut < -0.02, -1, 0))
        tgt = pd.Series(tgt, index=df.index)

        mask = feat.index <= df.index[end_idx]
        X = feat[mask].iloc[50:]
        y = tgt[mask].iloc[50:]

        if len(X) < 200:
            return False

        self.scaler.fit(X)
        self.model = GradientBoostingClassifier(
            n_estimators=30, max_depth=3, random_state=42
        )
        self.model.fit(self.scaler.transform(X), y)
        return True

    def predict_all(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            return pd.Series(0.0, index=df.index)
        feat = self._features(df)
        pred = self.model.predict(self.scaler.transform(feat))
        prob = np.max(self.model.predict_proba(self.scaler.transform(feat)), axis=1)
        return pd.Series(pred * prob * 3, index=df.index)


def backtest_sequential(
    df: pd.DataFrame,
    mf_scores: pd.Series,
    cost_model: ProductionCostModel,
    retrain_bars: int = 180,
) -> Dict:
    """순차적 Walk-Forward 백테스트"""
    if len(df) < 800:
        return None

    ml = MLSignal()
    init_train = 500
    last_train = init_train

    capital = 10000
    init_cap = capital
    position = None
    trades = []
    equity = [capital]

    # 초기 학습
    ml.train(df, init_train - 1)
    ml_scores = ml.predict_all(df)

    for i in range(init_train, len(df)):
        # 주기적 재학습
        if i - last_train >= retrain_bars:
            ml.train(df, i - 1)
            ml_scores = ml.predict_all(df)
            last_train = i

        score = mf_scores.iloc[i] + ml_scores.iloc[i] * 0.8
        price = df["close"].iloc[i]

        if position:
            bars = i - position["idx"]
            exit_cond = (
                (position["dir"] == 1 and score < 0)
                or (position["dir"] == -1 and score > 0)
                or bars >= 36
            )

            if exit_cond:
                gross = (price / position["entry"] - 1) * position["dir"]
                cost = cost_model.calc_cost(bars)
                net = gross - cost
                pnl = position["val"] * net
                trades.append({"pnl": pnl, "pct": net, "bars": bars})
                capital += pnl
                position = None

        if not position and abs(score) >= 4.0 and capital > 0:
            mult = min(abs(score) / 4.0, 2.0)
            position = {
                "entry": price,
                "dir": 1 if score > 0 else -1,
                "idx": i,
                "val": capital * 0.2 * 2 * mult,
            }

        equity.append(max(capital, 0))

    if len(trades) < 5:
        return None

    pnls = [t["pnl"] for t in trades]
    eq = pd.Series(equity)
    mdd = ((eq - eq.expanding().max()) / eq.expanding().max()).min() * 100

    wins = sum(1 for p in pnls if p > 0)
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = gp / gl if gl > 0 else 999

    # 연속 손실
    max_loss_streak = 0
    streak = 0
    for p in pnls:
        if p < 0:
            streak += 1
            max_loss_streak = max(max_loss_streak, streak)
        else:
            streak = 0

    return {
        "pf": min(pf, 999),
        "ret": (capital / init_cap - 1) * 100,
        "wr": wins / len(trades) * 100,
        "mdd": mdd,
        "trades": len(trades),
        "max_loss_streak": max_loss_streak,
        "trade_pcts": [t["pct"] for t in trades],
    }


def monte_carlo(trade_pcts: List[float], n_sim: int = 5000) -> Dict:
    """Monte Carlo 시뮬레이션"""
    finals = []
    mdds = []

    for _ in range(n_sim):
        shuffled = random.sample(trade_pcts, len(trade_pcts))
        cap = 10000
        peak = cap
        mdd = 0
        for pct in shuffled:
            cap *= 1 + pct
            if cap > peak:
                peak = cap
            dd = (peak - cap) / peak
            mdd = max(mdd, dd)
        finals.append(cap)
        mdds.append(mdd)

    finals = np.array(finals)
    return {
        "mean": (np.mean(finals) / 10000 - 1) * 100,
        "median": (np.median(finals) / 10000 - 1) * 100,
        "p5": (np.percentile(finals, 5) / 10000 - 1) * 100,
        "p25": (np.percentile(finals, 25) / 10000 - 1) * 100,
        "p75": (np.percentile(finals, 75) / 10000 - 1) * 100,
        "p95": (np.percentile(finals, 95) / 10000 - 1) * 100,
        "prob_profit": (finals > 10000).mean() * 100,
        "prob_loss_10": (finals < 9000).mean() * 100,
        "mean_mdd": np.mean(mdds) * 100,
        "worst_mdd": np.max(mdds) * 100,
    }


def stress_test(
    df: pd.DataFrame, scores: pd.Series, cost_model: ProductionCostModel
) -> Dict:
    """위기 기간 테스트"""
    periods = [
        ("Luna Crash", "2022-05-01", "2022-05-31"),
        ("FTX Collapse", "2022-11-01", "2022-11-30"),
        ("Bear 2022", "2022-06-01", "2022-08-31"),
        ("Recovery 2023", "2023-01-01", "2023-03-31"),
        ("Bull 2024", "2024-01-01", "2024-03-31"),
    ]

    results = {}
    for name, start, end in periods:
        mask = (df.index >= start) & (df.index <= end)
        if mask.sum() < 30:
            continue

        cap = 10000
        pos = None
        trades = []

        for i in range(mask.sum()):
            idx = df[mask].index[i]
            price = df.loc[idx, "close"]
            score = scores.loc[idx] if idx in scores.index else 0

            if pos:
                bars = i - pos["i"]
                exit_c = (
                    (pos["d"] == 1 and score < 0)
                    or (pos["d"] == -1 and score > 0)
                    or bars >= 36
                )
                if exit_c:
                    gross = (price / pos["e"] - 1) * pos["d"]
                    net = gross - cost_model.calc_cost(bars)
                    trades.append(cap * 0.4 * net)
                    cap += trades[-1]
                    pos = None

            if not pos and abs(score) >= 4.0:
                pos = {"e": price, "d": 1 if score > 0 else -1, "i": i}

        if trades:
            wins = sum(1 for t in trades if t > 0)
            results[name] = {
                "ret": (cap / 10000 - 1) * 100,
                "wr": wins / len(trades) * 100 if trades else 0,
                "trades": len(trades),
            }

    return results


def lag_test(
    df: pd.DataFrame, scores: pd.Series, cost_model: ProductionCostModel, lag: int
) -> Dict:
    """신호 지연 테스트"""
    if len(df) < 300:
        return None

    cap = 10000
    pos = None
    trades = []

    for i in range(50 + lag, len(df)):
        score = scores.iloc[i - lag]
        price = df["close"].iloc[i]

        if pos:
            bars = i - pos["i"]
            exit_c = (
                (pos["d"] == 1 and scores.iloc[i - lag] < 0)
                or (pos["d"] == -1 and scores.iloc[i - lag] > 0)
                or bars >= 36
            )
            if exit_c:
                gross = (price / pos["e"] - 1) * pos["d"]
                trades.append(cap * 0.4 * (gross - cost_model.calc_cost(bars)))
                cap += trades[-1]
                pos = None

        if not pos and abs(score) >= 4.0:
            pos = {"e": price, "d": 1 if score > 0 else -1, "i": i}

    if len(trades) < 3:
        return None

    wins = sum(1 for t in trades if t > 0)
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))

    return {
        "lag_h": lag * 4,
        "pf": gp / gl if gl > 0 else 999,
        "ret": (cap / 10000 - 1) * 100,
        "wr": wins / len(trades) * 100,
    }


def main():
    logger.info("=" * 70)
    logger.info("MF+ML PRODUCTION VALIDATION (Fast)")
    logger.info("=" * 70)

    loader = DataLoader()
    fg = loader.load_fear_greed()
    macro = loader.load_macro()
    btc = loader.load_ohlcv("BTCUSDT")
    cost = ProductionCostModel()
    mf = MultiFactorSignal()

    # 주요 30개 심볼
    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "SOLUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "DOTUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "MATICUSDT",
        "LTCUSDT",
        "ATOMUSDT",
        "UNIUSDT",
        "ETCUSDT",
        "NEARUSDT",
        "APTUSDT",
        "ARBUSDT",
        "OPUSDT",
        "LDOUSDT",
        "FILUSDT",
        "AAVEUSDT",
        "MKRUSDT",
        "SNXUSDT",
        "COMPUSDT",
        "RUNEUSDT",
        "INJUSDT",
        "SUIUSDT",
        "SEIUSDT",
        "TIAUSDT",
    ]

    # =========================================================================
    # TEST 1: Sequential Walk-Forward
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("TEST 1: SEQUENTIAL WALK-FORWARD")
    logger.info("=" * 70)

    results = []
    all_trades = []

    for sym in symbols:
        df = loader.load_ohlcv(sym)
        if df.empty:
            continue
        df = df[df.index >= "2022-01-01"]
        if len(df) < 800:
            continue

        mf_scores = mf.generate(df, btc, fg, macro)
        r = backtest_sequential(df, mf_scores, cost, retrain_bars=180)

        if r:
            r["symbol"] = sym
            results.append(r)
            all_trades.extend(r["trade_pcts"])
            status = "PASS" if r["pf"] > 1.0 else "FAIL"
            logger.info(
                f"  [{status}] {sym:<12} PF={r['pf']:5.2f} Ret={r['ret']:+6.1f}% "
                f"WR={r['wr']:.0f}% MDD={r['mdd']:.1f}%"
            )

    if results:
        df_r = pd.DataFrame(results)
        profitable = (df_r["pf"] > 1.0).sum()
        logger.info(f"\n[Summary]")
        logger.info(
            f"  Profitable: {profitable}/{len(results)} ({profitable/len(results)*100:.1f}%)"
        )
        logger.info(f"  Avg PF: {df_r[df_r['pf'] < 999]['pf'].mean():.2f}")
        logger.info(f"  Avg Return: {df_r['ret'].mean():+.1f}%")
        logger.info(f"  Avg MDD: {df_r['mdd'].mean():.1f}%")

    # =========================================================================
    # TEST 2: Monte Carlo
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("TEST 2: MONTE CARLO SIMULATION (5,000x)")
    logger.info("=" * 70)

    if all_trades:
        mc = monte_carlo(all_trades, 5000)
        logger.info(f"\n[Return Distribution]")
        logger.info(f"  Mean: {mc['mean']:+.1f}%")
        logger.info(f"  Median: {mc['median']:+.1f}%")
        logger.info(f"\n[Confidence Interval]")
        logger.info(f"  5% (Worst): {mc['p5']:+.1f}%")
        logger.info(f"  25%: {mc['p25']:+.1f}%")
        logger.info(f"  75%: {mc['p75']:+.1f}%")
        logger.info(f"  95% (Best): {mc['p95']:+.1f}%")
        logger.info(f"\n[Probability]")
        logger.info(f"  Profit: {mc['prob_profit']:.1f}%")
        logger.info(f"  Loss > 10%: {mc['prob_loss_10']:.1f}%")
        logger.info(f"\n[Drawdown]")
        logger.info(f"  Mean MDD: {mc['mean_mdd']:.1f}%")
        logger.info(f"  Worst MDD: {mc['worst_mdd']:.1f}%")

    # =========================================================================
    # TEST 3: Stress Test
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("TEST 3: STRESS TEST (Crisis Periods)")
    logger.info("=" * 70)

    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        df = loader.load_ohlcv(sym)
        if df.empty:
            continue
        df = df[df.index >= "2022-01-01"]

        mf_scores = mf.generate(df, btc, fg, macro)
        ml = MLSignal()
        ml.train(df, int(len(df) * 0.7))
        ml_scores = ml.predict_all(df)
        scores = mf_scores + ml_scores * 0.8

        stress = stress_test(df, scores, cost)
        if stress:
            logger.info(f"\n[{sym}]")
            for name, r in stress.items():
                status = (
                    "PASS" if r["ret"] > -15 else "WARN" if r["ret"] > -25 else "FAIL"
                )
                logger.info(
                    f"  [{status}] {name:<15} Ret={r['ret']:+.1f}% WR={r['wr']:.0f}%"
                )

    # =========================================================================
    # TEST 4: Execution Lag
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("TEST 4: EXECUTION LAG TEST")
    logger.info("=" * 70)

    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        df = loader.load_ohlcv(sym)
        if df.empty:
            continue
        df = df[df.index >= "2022-01-01"]

        mf_scores = mf.generate(df, btc, fg, macro)
        ml = MLSignal()
        ml.train(df, int(len(df) * 0.7))
        scores = mf_scores + ml.predict_all(df) * 0.8

        logger.info(f"\n[{sym}]")
        for lag in [0, 1, 2, 3]:
            r = lag_test(df, scores, cost, lag)
            if r:
                status = "PASS" if r["ret"] > 0 else "FAIL"
                logger.info(
                    f"  [{status}] Lag {r['lag_h']:2d}h: PF={r['pf']:5.2f} Ret={r['ret']:+6.1f}%"
                )

    # =========================================================================
    # TEST 5: Kelly Criterion
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("TEST 5: KELLY CRITERION")
    logger.info("=" * 70)

    if all_trades:
        wins = [t for t in all_trades if t > 0]
        losses = [t for t in all_trades if t <= 0]
        if wins and losses:
            wr = len(wins) / len(all_trades)
            avg_win = np.mean(wins)
            avg_loss = np.mean([abs(l) for l in losses])
            ratio = avg_win / avg_loss

            kelly = wr - (1 - wr) / ratio
            half_kelly = kelly / 2

            logger.info(f"\n[Stats]")
            logger.info(f"  Win Rate: {wr*100:.1f}%")
            logger.info(f"  Avg Win: {avg_win*100:.2f}%")
            logger.info(f"  Avg Loss: {avg_loss*100:.2f}%")
            logger.info(f"  Win/Loss Ratio: {ratio:.2f}")
            logger.info(f"\n[Kelly]")
            logger.info(f"  Full Kelly: {kelly*100:.1f}%")
            logger.info(f"  Half Kelly (Recommended): {max(0, half_kelly)*100:.1f}%")
            logger.info(f"  Current Setting: 20%")

    # =========================================================================
    # FINAL ASSESSMENT
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("FINAL ASSESSMENT")
    logger.info("=" * 70)

    criteria = []

    # C1: Walk-Forward Profitable >= 40%
    if results:
        df_r = pd.DataFrame(results)
        wf_rate = (df_r["pf"] > 1.0).sum() / len(df_r) * 100
        c1 = wf_rate >= 40
        logger.info(
            f"\n  [{'PASS' if c1 else 'FAIL'}] Walk-Forward Profitable >= 40%: {wf_rate:.1f}%"
        )
        criteria.append(c1)

    # C2: Monte Carlo Profit Prob >= 55%
    if all_trades:
        c2 = mc["prob_profit"] >= 55
        logger.info(
            f"  [{'PASS' if c2 else 'FAIL'}] MC Profit Probability >= 55%: {mc['prob_profit']:.1f}%"
        )
        criteria.append(c2)

    # C3: Worst Case (5%) >= -30%
    if all_trades:
        c3 = mc["p5"] >= -30
        logger.info(
            f"  [{'PASS' if c3 else 'FAIL'}] MC 5% Percentile >= -30%: {mc['p5']:.1f}%"
        )
        criteria.append(c3)

    # C4: Avg Return > 0
    if results:
        avg_ret = df_r["ret"].mean()
        c4 = avg_ret > 0
        logger.info(
            f"  [{'PASS' if c4 else 'FAIL'}] Average Return > 0%: {avg_ret:.1f}%"
        )
        criteria.append(c4)

    # C5: Kelly >= 5%
    if all_trades and wins and losses:
        c5 = half_kelly >= 0.05
        logger.info(
            f"  [{'PASS' if c5 else 'FAIL'}] Half Kelly >= 5%: {max(0,half_kelly)*100:.1f}%"
        )
        criteria.append(c5)

    passed = sum(criteria)
    total = len(criteria)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"RESULT: {passed}/{total} PASSED")
    logger.info("=" * 70)

    if passed == total:
        logger.info("\n>>> ALL PASSED - READY FOR PRODUCTION <<<")
    elif passed >= total * 0.6:
        logger.info("\n>>> PARTIAL PASS - USE WITH CAUTION <<<")
    else:
        logger.info("\n>>> FAILED - NOT READY FOR PRODUCTION <<<")

    # Top performers
    if results:
        logger.info(f"\n[Top 5 Performers]")
        df_r = pd.DataFrame(results)
        top5 = df_r.nlargest(5, "pf")
        for _, r in top5.iterrows():
            logger.info(f"  {r['symbol']:<12} PF={r['pf']:5.2f} Ret={r['ret']:+6.1f}%")

        # Save
        df_r.to_csv(DATA_ROOT / "mf_ml_production_fast.csv", index=False)
        logger.info(f"\nSaved: {DATA_ROOT / 'mf_ml_production_fast.csv'}")

    return {
        "results": results,
        "mc": mc if all_trades else None,
        "passed": passed,
        "total": total,
    }


if __name__ == "__main__":
    main()
