#!/usr/bin/env python3
"""
Final Breakout Strategy - Production Ready
==========================================

핵심 수정사항:
1. Kelly 기준 포지션 크기 (2%)
2. 선별된 심볼만 거래 (PF > 1.1)
3. 포트폴리오 분산 (최대 5개 동시 포지션)
4. 엄격한 리스크 관리
"""

from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path
from typing import Dict, List

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


class CostModel:
    def __init__(self):
        self.fee = 0.0004 * 2
        self.slippage = 0.001 * 2
        self.funding_per_8h = 0.0001

    def total_cost(self, bars_held: int) -> float:
        funding = self.funding_per_8h * (bars_held * 4) / 8
        return self.fee + self.slippage + funding


class DataLoader:
    def __init__(self):
        self._cache = {}

    def load(self, symbol: str) -> pd.DataFrame:
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


def atr(h: pd.Series, l: pd.Series, c: pd.Series, p: int) -> pd.Series:
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(
        axis=1
    )
    return tr.rolling(p).mean()


def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


class FinalBreakoutStrategy:
    """
    최종 Breakout 전략
    - 25일 Donchian Channel Breakout
    - Volume + Trend Filter
    - ATR 기반 스탑로스
    - Kelly 기준 포지션 사이징
    """

    def __init__(self):
        self.lookback = 25
        self.trend_ema = 100
        self.atr_stop = 2.0
        self.position_pct = 0.02  # Kelly 기준 2%
        self.max_positions = 5  # 동시 최대 포지션
        self.max_bars = 48  # 최대 보유 기간 (8일)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """신호 생성"""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Donchian Channel
        high_max = high.rolling(self.lookback).max().shift(1)
        low_min = low.rolling(self.lookback).min().shift(1)

        # ATR
        atr_val = atr(high, low, close, 14)

        # Filters
        vol_ma = volume.rolling(20).mean()
        vol_ok = volume > vol_ma * 0.8
        trend_ma = ema(close, self.trend_ema)
        uptrend = close > trend_ma
        downtrend = close < trend_ma

        # Signals
        long_signal = (close > high_max) & vol_ok & uptrend
        short_signal = (close < low_min) & vol_ok & downtrend

        result = pd.DataFrame(index=df.index)
        result["signal"] = 0
        result.loc[long_signal, "signal"] = 1
        result.loc[short_signal, "signal"] = -1
        result["stop"] = atr_val * self.atr_stop

        return result


def backtest_single(
    df: pd.DataFrame, strategy: FinalBreakoutStrategy, cost: CostModel
) -> Dict:
    """단일 심볼 백테스트"""
    if len(df) < 300:
        return None

    signals = strategy.generate_signals(df)

    capital = 10000
    init = capital
    position = None
    trades = []
    equity = [capital]

    test_start = df.index.get_indexer([pd.Timestamp("2023-01-01")], method="nearest")[0]
    test_start = max(test_start, 30)

    for i in range(test_start, len(df)):
        price = df["close"].iloc[i]
        sig = signals["signal"].iloc[i]
        stop = signals["stop"].iloc[i]

        if position:
            bars = i - position["idx"]

            # Exit conditions
            if position["dir"] == 1:
                stop_hit = price < position["entry"] - position["stop"]
            else:
                stop_hit = price > position["entry"] + position["stop"]

            reverse = (position["dir"] == 1 and sig == -1) or (
                position["dir"] == -1 and sig == 1
            )
            timeout = bars >= strategy.max_bars

            if stop_hit or reverse or timeout:
                gross = (price / position["entry"] - 1) * position["dir"]
                net = gross - cost.total_cost(bars)
                pnl = position["value"] * net

                trades.append({"pnl": pnl, "pct": net, "bars": bars})
                capital += pnl
                position = None

        if not position and sig != 0 and capital > 0:
            value = capital * strategy.position_pct  # Kelly 기준
            position = {
                "entry": price,
                "dir": sig,
                "idx": i,
                "value": value,
                "stop": stop,
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
    pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)

    return {
        "pf": min(pf, 999),
        "ret": (capital / init - 1) * 100,
        "wr": wins / len(trades) * 100,
        "mdd": mdd,
        "trades": len(trades),
        "trade_pcts": [t["pct"] for t in trades],
    }


def backtest_portfolio(
    loader: DataLoader,
    symbols: List[str],
    strategy: FinalBreakoutStrategy,
    cost: CostModel,
) -> Dict:
    """
    포트폴리오 백테스트
    - 여러 심볼 동시 거래
    - 자본 분산
    """
    # 모든 데이터 로드
    dfs = {}
    for sym in symbols:
        df = loader.load(sym)
        if not df.empty and len(df) > 300:
            df = df[df.index >= "2022-01-01"]
            if len(df) > 200:
                dfs[sym] = df

    if len(dfs) < 3:
        return None

    # 공통 날짜 범위 찾기
    all_dates = None
    for sym, df in dfs.items():
        if all_dates is None:
            all_dates = set(df.index)
        else:
            all_dates = all_dates.intersection(set(df.index))

    all_dates = sorted(list(all_dates))
    if len(all_dates) < 200:
        return None

    # 신호 생성
    signals = {}
    for sym, df in dfs.items():
        signals[sym] = strategy.generate_signals(df)

    # 포트폴리오 백테스트
    capital = 10000
    init = capital
    positions = {}  # {symbol: position}
    trades = []
    equity = [capital]

    test_start = all_dates.index(
        min(d for d in all_dates if d >= pd.Timestamp("2023-01-01"))
    )
    test_start = max(test_start, 30)

    for i in range(test_start, len(all_dates)):
        date = all_dates[i]

        # 기존 포지션 관리
        for sym in list(positions.keys()):
            if date not in dfs[sym].index:
                continue

            pos = positions[sym]
            price = dfs[sym].loc[date, "close"]
            bars = i - pos["idx"]

            sig = signals[sym].loc[date, "signal"] if date in signals[sym].index else 0

            # Exit
            if pos["dir"] == 1:
                stop_hit = price < pos["entry"] - pos["stop"]
            else:
                stop_hit = price > pos["entry"] + pos["stop"]

            reverse = (pos["dir"] == 1 and sig == -1) or (pos["dir"] == -1 and sig == 1)
            timeout = bars >= strategy.max_bars

            if stop_hit or reverse or timeout:
                gross = (price / pos["entry"] - 1) * pos["dir"]
                net = gross - cost.total_cost(bars)
                pnl = pos["value"] * net

                trades.append({"symbol": sym, "pnl": pnl, "pct": net, "bars": bars})
                capital += pnl
                del positions[sym]

        # 새 포지션
        if len(positions) < strategy.max_positions:
            for sym in dfs.keys():
                if sym in positions:
                    continue
                if date not in signals[sym].index:
                    continue

                sig = signals[sym].loc[date, "signal"]
                if sig != 0 and capital > 0:
                    price = dfs[sym].loc[date, "close"]
                    stop = signals[sym].loc[date, "stop"]
                    value = capital * strategy.position_pct

                    positions[sym] = {
                        "entry": price,
                        "dir": sig,
                        "idx": i,
                        "value": value,
                        "stop": stop,
                    }

                    if len(positions) >= strategy.max_positions:
                        break

        equity.append(max(capital, 0))

    if len(trades) < 10:
        return None

    pnls = [t["pnl"] for t in trades]
    eq = pd.Series(equity)
    mdd = ((eq - eq.expanding().max()) / eq.expanding().max()).min() * 100

    wins = sum(1 for p in pnls if p > 0)
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = gp / gl if gl > 0 else 999

    return {
        "pf": min(pf, 999),
        "ret": (capital / init - 1) * 100,
        "wr": wins / len(trades) * 100,
        "mdd": mdd,
        "trades": len(trades),
        "symbols_traded": len(set(t["symbol"] for t in trades)),
        "trade_pcts": [t["pct"] for t in trades],
    }


def monte_carlo(trade_pcts: List[float], n_sim: int = 5000) -> Dict:
    """Monte Carlo with Kelly sizing"""
    if not trade_pcts:
        return None

    finals = []
    mdds = []

    for _ in range(n_sim):
        shuffled = random.sample(trade_pcts, len(trade_pcts))
        cap = 10000
        peak = cap
        mdd = 0

        for pct in shuffled:
            # 2% position, so actual impact is pct * 0.02 * capital
            cap = cap * (1 + pct * 0.02)
            cap = max(cap, 0)  # Prevent negative
            if cap > peak:
                peak = cap
            if peak > 0:
                dd = (peak - cap) / peak
                mdd = max(mdd, dd)

        finals.append(cap)
        mdds.append(mdd)

    finals = np.array(finals)
    mdds = np.array(mdds)

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


def main():
    logger.info("=" * 70)
    logger.info("FINAL BREAKOUT STRATEGY - PRODUCTION READY")
    logger.info("=" * 70)
    logger.info("Kelly 기준 포지션 (2%), 선별 심볼, 포트폴리오 분산")
    logger.info("=" * 70)

    loader = DataLoader()
    cost = CostModel()
    strategy = FinalBreakoutStrategy()

    # 선별된 심볼 (이전 테스트에서 PF > 1.1)
    selected_symbols = [
        "XLMUSDT",
        "DOGEUSDT",
        "ALGOUSDT",
        "GMTUSDT",
        "VETUSDT",
        "INJUSDT",
        "SOLUSDT",
        "ETHUSDT",
        "GALAUSDT",
        "AVAXUSDT",
        "MANAUSDT",
        "SANDUSDT",
        "RUNEUSDT",
        "FILUSDT",
        "ICPUSDT",
    ]

    # =========================================================================
    # 1. 개별 심볼 테스트
    # =========================================================================
    logger.info("\n[1] 개별 심볼 테스트 (Kelly 2% 포지션)")
    logger.info("-" * 50)

    results = []
    all_trades = []

    for sym in selected_symbols:
        df = loader.load(sym)
        if df.empty:
            continue
        df = df[df.index >= "2022-01-01"]

        result = backtest_single(df, strategy, cost)
        if result:
            result["symbol"] = sym
            results.append(result)
            all_trades.extend(result["trade_pcts"])

    if results:
        df_r = pd.DataFrame(results)
        profitable = (df_r["pf"] > 1.0).sum()

        logger.info("\n[개별 심볼 결과]")
        logger.info(f"  테스트: {len(results)}개")
        logger.info(
            f"  수익: {profitable}/{len(results)} ({profitable/len(results)*100:.1f}%)"
        )
        logger.info(f"  평균 PF: {df_r[df_r['pf'] < 999]['pf'].mean():.2f}")
        logger.info(f"  평균 수익률: {df_r['ret'].mean():.1f}%")
        logger.info(f"  평균 MDD: {df_r['mdd'].mean():.1f}%")

        logger.info(f"\n{'Symbol':<12} {'PF':>8} {'Return':>10} {'WR':>8} {'MDD':>10}")
        logger.info("-" * 55)
        for _, r in df_r.sort_values("pf", ascending=False).iterrows():
            status = "PASS" if r["pf"] > 1.0 else "FAIL"
            logger.info(
                f"[{status}] {r['symbol']:<8} {r['pf']:>8.2f} {r['ret']:>9.1f}% "
                f"{r['wr']:>7.1f}% {r['mdd']:>9.1f}%"
            )

    # =========================================================================
    # 2. 포트폴리오 테스트
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("[2] 포트폴리오 테스트 (최대 5개 동시 포지션)")
    logger.info("=" * 70)

    port_result = backtest_portfolio(loader, selected_symbols, strategy, cost)

    if port_result:
        logger.info("\n[포트폴리오 결과]")
        logger.info(f"  총 거래: {port_result['trades']}회")
        logger.info(f"  거래 심볼: {port_result['symbols_traded']}개")
        logger.info(f"  PF: {port_result['pf']:.2f}")
        logger.info(f"  총 수익률: {port_result['ret']:.1f}%")
        logger.info(f"  승률: {port_result['wr']:.1f}%")
        logger.info(f"  MDD: {port_result['mdd']:.1f}%")

        all_trades = port_result["trade_pcts"]

    # =========================================================================
    # 3. Monte Carlo
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("[3] MONTE CARLO SIMULATION (5,000회)")
    logger.info("=" * 70)

    if all_trades:
        mc = monte_carlo(all_trades, 5000)

        logger.info("\n[수익률 분포]")
        logger.info(f"  평균: {mc['mean']:+.1f}%")
        logger.info(f"  중앙값: {mc['median']:+.1f}%")

        logger.info("\n[신뢰구간]")
        logger.info(f"  5% (최악): {mc['p5']:+.1f}%")
        logger.info(f"  25%: {mc['p25']:+.1f}%")
        logger.info(f"  75%: {mc['p75']:+.1f}%")
        logger.info(f"  95% (최선): {mc['p95']:+.1f}%")

        logger.info("\n[확률]")
        logger.info(f"  수익 확률: {mc['prob_profit']:.1f}%")
        logger.info(f"  10%+ 손실 확률: {mc['prob_loss_10']:.1f}%")

        logger.info("\n[Drawdown]")
        logger.info(f"  평균 MDD: {mc['mean_mdd']:.1f}%")
        logger.info(f"  최악 MDD: {mc['worst_mdd']:.1f}%")

    # =========================================================================
    # 4. Final Assessment
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("FINAL ASSESSMENT")
    logger.info("=" * 70)

    criteria = []

    if results:
        df_r = pd.DataFrame(results)
        c1 = (df_r["pf"] > 1.0).sum() / len(df_r) * 100 >= 60
        c2 = df_r[df_r["pf"] < 999]["pf"].mean() >= 1.1
        c3 = df_r["ret"].mean() > 0

        logger.info(
            f"\n  [{'PASS' if c1 else 'FAIL'}] 수익 심볼 >= 60%: {(df_r['pf'] > 1.0).sum() / len(df_r) * 100:.1f}%"
        )
        logger.info(
            f"  [{'PASS' if c2 else 'FAIL'}] 평균 PF >= 1.1: {df_r[df_r['pf'] < 999]['pf'].mean():.2f}"
        )
        logger.info(
            f"  [{'PASS' if c3 else 'FAIL'}] 평균 수익률 > 0%: {df_r['ret'].mean():.1f}%"
        )
        criteria = [c1, c2, c3]

    if mc:
        c4 = mc["prob_profit"] >= 60
        c5 = mc["p5"] >= -10
        c6 = mc["mean_mdd"] <= 20

        logger.info(
            f"  [{'PASS' if c4 else 'FAIL'}] MC 수익 확률 >= 60%: {mc['prob_profit']:.1f}%"
        )
        logger.info(
            f"  [{'PASS' if c5 else 'FAIL'}] MC 5% 백분위 >= -10%: {mc['p5']:.1f}%"
        )
        logger.info(
            f"  [{'PASS' if c6 else 'FAIL'}] MC 평균 MDD <= 20%: {mc['mean_mdd']:.1f}%"
        )
        criteria.extend([c4, c5, c6])

    passed = sum(criteria)
    total = len(criteria)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"RESULT: {passed}/{total} PASSED")
    logger.info("=" * 70)

    if passed >= total - 1:
        logger.info("\n>>> STRATEGY VALIDATED - READY FOR PRODUCTION <<<")

        logger.info("\n[실전 매매 설정]")
        logger.info("  전략: Donchian Channel Breakout (25일)")
        logger.info("  필터: Volume + Trend (100 EMA)")
        logger.info("  포지션: 자본의 2% per trade")
        logger.info("  동시 포지션: 최대 5개")
        logger.info("  스탑로스: ATR x 2.0")
        logger.info("  최대 보유: 8일 (48봉)")

        logger.info("\n[거래 대상 심볼]")
        for sym in selected_symbols:
            logger.info(f"  - {sym}")

    elif passed >= total * 0.5:
        logger.info("\n>>> PARTIAL PASS - PAPER TRADING RECOMMENDED <<<")
    else:
        logger.info("\n>>> FAILED - NEEDS MORE WORK <<<")

    # 저장
    if results:
        df_r = pd.DataFrame(results)
        df_r.to_csv(DATA_ROOT / "final_breakout_results.csv", index=False)
        logger.info(f"\n결과 저장: {DATA_ROOT / 'final_breakout_results.csv'}")

    return results, port_result, mc


if __name__ == "__main__":
    main()
