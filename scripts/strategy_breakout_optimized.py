#!/usr/bin/env python3
"""
Breakout Strategy - Optimized Version
=====================================

Donchian Channel Breakout + 필터 조합

최적화 항목:
1. Lookback Period (10, 15, 20, 25, 30)
2. Volume Filter (거래량 확인)
3. Trend Filter (장기 추세 방향)
4. ATR-based Stop Loss
5. Position Sizing
"""

from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

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

    def get_symbols(self) -> List[str]:
        folder = DATA_ROOT / "binance_futures_4h"
        return sorted([f.stem for f in folder.glob("*.csv") if f.stem.endswith("USDT")])


def atr(h: pd.Series, l: pd.Series, c: pd.Series, p: int) -> pd.Series:
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(
        axis=1
    )
    return tr.rolling(p).mean()


def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


class BreakoutStrategy:
    """
    Donchian Channel Breakout Strategy
    """

    def __init__(
        self,
        lookback: int = 20,
        use_volume_filter: bool = True,
        use_trend_filter: bool = True,
        atr_stop_mult: float = 2.0,
        trend_ema: int = 100,
    ):
        self.lookback = lookback
        self.use_volume_filter = use_volume_filter
        self.use_trend_filter = use_trend_filter
        self.atr_stop_mult = atr_stop_mult
        self.trend_ema = trend_ema

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        신호 및 스탑로스 레벨 생성

        Returns:
            signals: 1 (Long), -1 (Short), 0 (No signal)
            stops: ATR 기반 스탑로스 레벨
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Donchian Channel
        high_max = high.rolling(self.lookback).max().shift(1)
        low_min = low.rolling(self.lookback).min().shift(1)

        # ATR for stop loss
        atr_val = atr(high, low, close, 14)

        # Volume filter: 거래량이 평균 이상
        vol_ma = volume.rolling(20).mean()
        vol_ok = volume > vol_ma * 0.8

        # Trend filter: 장기 추세 방향
        trend_ma = ema(close, self.trend_ema)
        uptrend = close > trend_ma
        downtrend = close < trend_ma

        # 기본 신호
        long_signal = close > high_max
        short_signal = close < low_min

        # 필터 적용
        if self.use_volume_filter:
            long_signal = long_signal & vol_ok
            short_signal = short_signal & vol_ok

        if self.use_trend_filter:
            long_signal = long_signal & uptrend
            short_signal = short_signal & downtrend

        # 신호 생성
        signals = pd.Series(0, index=df.index)
        signals[long_signal] = 1
        signals[short_signal] = -1

        # 스탑로스 레벨
        stops = atr_val * self.atr_stop_mult

        return signals, stops

    def backtest(
        self,
        df: pd.DataFrame,
        cost: CostModel,
        position_pct: float = 0.15,
        leverage: float = 2.0,
        max_bars: int = 48,
    ) -> Dict:
        """
        백테스트 실행
        """
        if len(df) < 300:
            return None

        signals, stops = self.generate_signals(df)

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        # 테스트 시작: 2023-01-01
        test_start = df.index.get_indexer(
            [pd.Timestamp("2023-01-01")], method="nearest"
        )[0]
        test_start = max(test_start, self.lookback + 10)

        for i in range(test_start, len(df)):
            price = df["close"].iloc[i]
            sig = signals.iloc[i]
            stop = stops.iloc[i]

            if position:
                bars = i - position["idx"]

                # 청산 조건
                # 1. 스탑로스
                if position["dir"] == 1:  # Long
                    stop_hit = price < position["entry"] - position["stop"]
                else:  # Short
                    stop_hit = price > position["entry"] + position["stop"]

                # 2. 반대 신호
                reverse_signal = (position["dir"] == 1 and sig == -1) or (
                    position["dir"] == -1 and sig == 1
                )

                # 3. 최대 보유 기간
                max_hold = bars >= max_bars

                if stop_hit or reverse_signal or max_hold:
                    gross_pnl = (price / position["entry"] - 1) * position["dir"]
                    net_pnl = gross_pnl - cost.total_cost(bars)
                    pnl = position["value"] * net_pnl

                    trades.append(
                        {
                            "pnl": pnl,
                            "pct": net_pnl,
                            "bars": bars,
                            "dir": position["dir"],
                            "exit_reason": (
                                "stop"
                                if stop_hit
                                else ("reverse" if reverse_signal else "timeout")
                            ),
                        }
                    )
                    capital += pnl
                    position = None

            # 진입
            if not position and sig != 0 and capital > 0:
                value = capital * position_pct * leverage
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

        # 통계
        pnls = [t["pnl"] for t in trades]
        pcts = [t["pct"] for t in trades]
        eq = pd.Series(equity)
        mdd = ((eq - eq.expanding().max()) / eq.expanding().max()).min() * 100

        wins = sum(1 for p in pnls if p > 0)
        gp = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)

        # 연속 손실
        max_streak = 0
        streak = 0
        for p in pnls:
            if p < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        return {
            "pf": min(pf, 999),
            "ret": (capital / init - 1) * 100,
            "wr": wins / len(trades) * 100,
            "mdd": mdd,
            "trades": len(trades),
            "avg_bars": np.mean([t["bars"] for t in trades]),
            "max_loss_streak": max_streak,
            "trade_pcts": pcts,
        }


def parameter_optimization(loader: DataLoader, cost: CostModel) -> Dict:
    """파라미터 최적화"""

    test_symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ARBUSDT",
        "APTUSDT",
        "AVAXUSDT",
        "DOTUSDT",
        "NEARUSDT",
        "INJUSDT",
    ]

    # 파라미터 조합
    lookbacks = [15, 20, 25]
    volume_filters = [True, False]
    trend_filters = [True, False]

    results = []

    for lookback in lookbacks:
        for vol_filter in volume_filters:
            for trend_filter in trend_filters:
                strategy = BreakoutStrategy(
                    lookback=lookback,
                    use_volume_filter=vol_filter,
                    use_trend_filter=trend_filter,
                )

                symbol_results = []
                for symbol in test_symbols:
                    df = loader.load(symbol)
                    if df.empty or len(df) < 500:
                        continue
                    df = df[df.index >= "2022-01-01"]

                    result = strategy.backtest(df, cost)
                    if result:
                        symbol_results.append(result)

                if symbol_results:
                    df_r = pd.DataFrame(symbol_results)
                    profitable = (df_r["pf"] > 1.0).sum()

                    results.append(
                        {
                            "lookback": lookback,
                            "vol_filter": vol_filter,
                            "trend_filter": trend_filter,
                            "n": len(symbol_results),
                            "profitable_pct": profitable / len(symbol_results) * 100,
                            "avg_pf": df_r[df_r["pf"] < 999]["pf"].mean(),
                            "avg_ret": df_r["ret"].mean(),
                            "avg_wr": df_r["wr"].mean(),
                        }
                    )

    return pd.DataFrame(results)


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
        "p95": (np.percentile(finals, 95) / 10000 - 1) * 100,
        "prob_profit": (finals > 10000).mean() * 100,
        "mean_mdd": np.mean(mdds) * 100,
    }


def kelly_criterion(trade_pcts: List[float]) -> float:
    """Kelly Criterion 계산"""
    wins = [p for p in trade_pcts if p > 0]
    losses = [p for p in trade_pcts if p <= 0]

    if not wins or not losses:
        return 0

    wr = len(wins) / len(trade_pcts)
    avg_win = np.mean(wins)
    avg_loss = np.mean([abs(l) for l in losses])
    ratio = avg_win / avg_loss

    kelly = wr - (1 - wr) / ratio
    return max(0, kelly / 2)  # Half Kelly


def main():
    logger.info("=" * 70)
    logger.info("BREAKOUT STRATEGY - OPTIMIZED")
    logger.info("=" * 70)

    loader = DataLoader()
    cost = CostModel()

    # =========================================================================
    # 1. 파라미터 최적화
    # =========================================================================
    logger.info("\n[1] 파라미터 최적화")
    logger.info("-" * 50)

    opt_results = parameter_optimization(loader, cost)
    opt_results = opt_results.sort_values("avg_pf", ascending=False)

    logger.info(
        f"\n{'Lookback':>8} {'Vol':>6} {'Trend':>6} {'Profit%':>10} {'Avg PF':>10} {'Avg Ret':>10}"
    )
    logger.info("-" * 60)

    for _, r in opt_results.head(10).iterrows():
        logger.info(
            f"{r['lookback']:>8} {str(r['vol_filter']):>6} {str(r['trend_filter']):>6} "
            f"{r['profitable_pct']:>9.1f}% {r['avg_pf']:>10.2f} {r['avg_ret']:>9.1f}%"
        )

    # 최적 파라미터
    best = opt_results.iloc[0]
    logger.info(f"\n최적 파라미터:")
    logger.info(f"  Lookback: {best['lookback']}")
    logger.info(f"  Volume Filter: {best['vol_filter']}")
    logger.info(f"  Trend Filter: {best['trend_filter']}")

    # =========================================================================
    # 2. 최적 파라미터로 전체 테스트
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("[2] 최적 파라미터 전체 테스트")
    logger.info("=" * 70)

    strategy = BreakoutStrategy(
        lookback=int(best["lookback"]),
        use_volume_filter=best["vol_filter"],
        use_trend_filter=best["trend_filter"],
    )

    # 유동성 상위 40개 심볼
    all_symbols = loader.get_symbols()
    test_symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "SOLUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "DOTUSDT",
        "MATICUSDT",
        "LTCUSDT",
        "ATOMUSDT",
        "UNIUSDT",
        "NEARUSDT",
        "APTUSDT",
        "ARBUSDT",
        "OPUSDT",
        "INJUSDT",
        "SUIUSDT",
        "SEIUSDT",
        "TIAUSDT",
        "LDOUSDT",
        "RUNEUSDT",
        "AAVEUSDT",
        "FILUSDT",
        "ETCUSDT",
        "XLMUSDT",
        "ALGOUSDT",
        "VETUSDT",
        "ICPUSDT",
        "SANDUSDT",
        "MANAUSDT",
        "AXSUSDT",
        "GALAUSDT",
        "FLOWUSDT",
        "APEUSDT",
        "GMTUSDT",
        "CHZUSDT",
        "LRCUSDT",
    ]

    results = []
    all_trades = []

    for symbol in test_symbols:
        df = loader.load(symbol)
        if df.empty or len(df) < 500:
            continue
        df = df[df.index >= "2022-01-01"]

        result = strategy.backtest(df, cost)
        if result:
            result["symbol"] = symbol
            results.append(result)
            all_trades.extend(result["trade_pcts"])

    if results:
        df_r = pd.DataFrame(results)
        profitable = (df_r["pf"] > 1.0).sum()

        logger.info(f"\n[결과 요약]")
        logger.info(f"  테스트 심볼: {len(results)}")
        logger.info(
            f"  수익 심볼: {profitable}/{len(results)} ({profitable/len(results)*100:.1f}%)"
        )
        logger.info(f"  평균 PF: {df_r[df_r['pf'] < 999]['pf'].mean():.2f}")
        logger.info(f"  평균 수익률: {df_r['ret'].mean():.1f}%")
        logger.info(f"  평균 승률: {df_r['wr'].mean():.1f}%")
        logger.info(f"  평균 MDD: {df_r['mdd'].mean():.1f}%")

        # 상세 결과
        logger.info(f"\n[심볼별 상세]")
        logger.info(
            f"{'Symbol':<12} {'PF':>8} {'Return':>10} {'WR':>8} {'Trades':>8} {'MDD':>10}"
        )
        logger.info("-" * 60)

        for _, r in df_r.sort_values("pf", ascending=False).iterrows():
            status = "PASS" if r["pf"] > 1.0 else "FAIL"
            logger.info(
                f"[{status}] {r['symbol']:<8} {r['pf']:>8.2f} {r['ret']:>9.1f}% "
                f"{r['wr']:>7.1f}% {r['trades']:>8} {r['mdd']:>9.1f}%"
            )

    # =========================================================================
    # 3. Monte Carlo
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("[3] MONTE CARLO SIMULATION")
    logger.info("=" * 70)

    if all_trades:
        mc = monte_carlo(all_trades, 5000)
        logger.info(f"\n  평균 수익률: {mc['mean']:+.1f}%")
        logger.info(f"  중앙값: {mc['median']:+.1f}%")
        logger.info(f"  5% 백분위 (최악): {mc['p5']:+.1f}%")
        logger.info(f"  95% 백분위 (최선): {mc['p95']:+.1f}%")
        logger.info(f"  수익 확률: {mc['prob_profit']:.1f}%")
        logger.info(f"  평균 MDD: {mc['mean_mdd']:.1f}%")

    # =========================================================================
    # 4. Kelly Criterion
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("[4] KELLY CRITERION")
    logger.info("=" * 70)

    if all_trades:
        kelly = kelly_criterion(all_trades)
        wins = [p for p in all_trades if p > 0]
        losses = [p for p in all_trades if p <= 0]

        logger.info(f"\n  승률: {len(wins)/len(all_trades)*100:.1f}%")
        logger.info(f"  평균 이익: {np.mean(wins)*100:.2f}%")
        logger.info(f"  평균 손실: {np.mean([abs(l) for l in losses])*100:.2f}%")
        logger.info(f"  Half Kelly: {kelly*100:.1f}%")
        logger.info(f"  현재 설정: 15% * 2x = 30%")

        if kelly > 0.15:
            logger.info(f"  >>> Kelly 충족: 현재 포지션 적정 <<<")
        else:
            logger.info(
                f"  >>> 경고: 포지션 크기 조정 필요 (권장: {kelly*100:.0f}%) <<<"
            )

    # =========================================================================
    # 5. Final Assessment
    # =========================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("FINAL ASSESSMENT")
    logger.info("=" * 70)

    criteria = []

    if results:
        df_r = pd.DataFrame(results)
        profitable_pct = (df_r["pf"] > 1.0).sum() / len(df_r) * 100
        avg_pf = df_r[df_r["pf"] < 999]["pf"].mean()
        avg_ret = df_r["ret"].mean()

        c1 = profitable_pct >= 50
        c2 = avg_pf >= 1.0
        c3 = avg_ret > 0

        logger.info(
            f"\n  [{'PASS' if c1 else 'FAIL'}] 수익 심볼 >= 50%: {profitable_pct:.1f}%"
        )
        logger.info(f"  [{'PASS' if c2 else 'FAIL'}] 평균 PF >= 1.0: {avg_pf:.2f}")
        logger.info(f"  [{'PASS' if c3 else 'FAIL'}] 평균 수익률 > 0%: {avg_ret:.1f}%")

        criteria = [c1, c2, c3]

    if all_trades:
        c4 = mc["prob_profit"] >= 55
        c5 = kelly > 0.05

        logger.info(
            f"  [{'PASS' if c4 else 'FAIL'}] MC 수익 확률 >= 55%: {mc['prob_profit']:.1f}%"
        )
        logger.info(f"  [{'PASS' if c5 else 'FAIL'}] Kelly >= 5%: {kelly*100:.1f}%")

        criteria.extend([c4, c5])

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
        logger.info("\n>>> FAILED - NEEDS IMPROVEMENT <<<")

    # 권장 심볼
    if results:
        df_r = pd.DataFrame(results)
        recommended = df_r[df_r["pf"] > 1.0].nlargest(10, "pf")

        logger.info(f"\n[권장 거래 심볼 TOP 10]")
        for _, r in recommended.iterrows():
            logger.info(f"  {r['symbol']:<12} PF={r['pf']:.2f} Ret={r['ret']:+.1f}%")

        # 저장
        df_r.to_csv(DATA_ROOT / "breakout_optimized_results.csv", index=False)
        logger.info(f"\n결과 저장: {DATA_ROOT / 'breakout_optimized_results.csv'}")

    return results


if __name__ == "__main__":
    main()
