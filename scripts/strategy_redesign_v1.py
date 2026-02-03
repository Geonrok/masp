#!/usr/bin/env python3
"""
Strategy Redesign v1: 기본 전략 탐색
=====================================

목표: Look-Ahead Bias 없이 실제 매매와 동일한 조건에서 작동하는 전략 찾기

테스트 전략:
1. Trend Following - 이동평균 크로스
2. Momentum - N일 수익률 기반
3. Breakout - 가격 채널 돌파
4. Volatility - ATR 기반 변동성 돌파
5. Mean Reversion - 볼린저밴드 역추세
6. Volume Profile - 거래량 이상 감지

원칙:
- 단순함 (파라미터 최소화)
- 미래 정보 사용 금지
- 실제 비용 반영
- 순차적 테스트
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List
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


class CostModel:
    """실제 거래 비용"""

    def __init__(self):
        self.fee = 0.0004 * 2  # 왕복 수수료
        self.slippage = 0.001 * 2  # 왕복 슬리피지
        self.funding_per_8h = 0.0001  # 평균 펀딩비

    def total_cost(self, bars_held: int, timeframe_h: int = 4) -> float:
        funding_periods = (bars_held * timeframe_h) / 8
        return self.fee + self.slippage + self.funding_per_8h * funding_periods


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


# ============================================================================
# 기본 지표 함수
# ============================================================================


def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p).mean()


def atr(h: pd.Series, l: pd.Series, c: pd.Series, p: int) -> pd.Series:
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(
        axis=1
    )
    return tr.rolling(p).mean()


def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


# ============================================================================
# 전략 1: Trend Following (이동평균 크로스)
# ============================================================================


def strategy_trend_follow(
    df: pd.DataFrame, fast: int = 20, slow: int = 50
) -> pd.Series:
    """
    단순 이동평균 크로스
    - Fast EMA > Slow EMA: Long
    - Fast EMA < Slow EMA: Short
    """
    fast_ma = ema(df["close"], fast)
    slow_ma = ema(df["close"], slow)

    signal = pd.Series(0, index=df.index)
    signal[fast_ma > slow_ma] = 1
    signal[fast_ma < slow_ma] = -1

    return signal


# ============================================================================
# 전략 2: Momentum (수익률 기반)
# ============================================================================


def strategy_momentum(
    df: pd.DataFrame, lookback: int = 20, threshold: float = 0.05
) -> pd.Series:
    """
    N일 수익률 기반 모멘텀
    - 수익률 > threshold: Long
    - 수익률 < -threshold: Short
    """
    ret = df["close"].pct_change(lookback)

    signal = pd.Series(0, index=df.index)
    signal[ret > threshold] = 1
    signal[ret < -threshold] = -1

    return signal


# ============================================================================
# 전략 3: Breakout (채널 돌파)
# ============================================================================


def strategy_breakout(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Donchian Channel 돌파
    - 최고가 돌파: Long
    - 최저가 돌파: Short
    """
    high_max = df["high"].rolling(lookback).max().shift(1)
    low_min = df["low"].rolling(lookback).min().shift(1)

    signal = pd.Series(0, index=df.index)
    signal[df["close"] > high_max] = 1
    signal[df["close"] < low_min] = -1

    return signal


# ============================================================================
# 전략 4: Volatility Breakout (변동성 돌파)
# ============================================================================


def strategy_volatility_breakout(
    df: pd.DataFrame, atr_mult: float = 1.5, atr_period: int = 14
) -> pd.Series:
    """
    ATR 기반 변동성 돌파
    - 종가 > 전일 종가 + ATR * mult: Long
    - 종가 < 전일 종가 - ATR * mult: Short
    """
    atr_val = atr(df["high"], df["low"], df["close"], atr_period)
    prev_close = df["close"].shift(1)

    signal = pd.Series(0, index=df.index)
    signal[df["close"] > prev_close + atr_val * atr_mult] = 1
    signal[df["close"] < prev_close - atr_val * atr_mult] = -1

    return signal


# ============================================================================
# 전략 5: Mean Reversion (볼린저밴드)
# ============================================================================


def strategy_mean_reversion(
    df: pd.DataFrame, period: int = 20, std_mult: float = 2.0
) -> pd.Series:
    """
    볼린저밴드 역추세
    - 하단 밴드 아래: Long (과매도)
    - 상단 밴드 위: Short (과매수)
    """
    mid = sma(df["close"], period)
    std = df["close"].rolling(period).std()
    upper = mid + std * std_mult
    lower = mid - std * std_mult

    signal = pd.Series(0, index=df.index)
    signal[df["close"] < lower] = 1  # 과매도 -> Long
    signal[df["close"] > upper] = -1  # 과매수 -> Short

    return signal


# ============================================================================
# 전략 6: Volume Spike (거래량 급증)
# ============================================================================


def strategy_volume_spike(
    df: pd.DataFrame, vol_mult: float = 2.0, lookback: int = 20
) -> pd.Series:
    """
    거래량 급증 + 가격 방향
    - 거래량 급증 + 상승: Long
    - 거래량 급증 + 하락: Short
    """
    vol_ma = sma(df["volume"], lookback)
    vol_spike = df["volume"] > vol_ma * vol_mult
    price_up = df["close"] > df["close"].shift(1)

    signal = pd.Series(0, index=df.index)
    signal[vol_spike & price_up] = 1
    signal[vol_spike & ~price_up] = -1

    return signal


# ============================================================================
# 전략 7: RSI Extreme (RSI 극단값)
# ============================================================================


def strategy_rsi_extreme(
    df: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70
) -> pd.Series:
    """
    RSI 극단값 역추세
    - RSI < oversold: Long
    - RSI > overbought: Short
    """
    rsi_val = rsi(df["close"], period)

    signal = pd.Series(0, index=df.index)
    signal[rsi_val < oversold] = 1
    signal[rsi_val > overbought] = -1

    return signal


# ============================================================================
# 전략 8: Dual Momentum (상대 + 절대 모멘텀)
# ============================================================================


def strategy_dual_momentum(
    df: pd.DataFrame, btc_df: pd.DataFrame, lookback: int = 30
) -> pd.Series:
    """
    이중 모멘텀
    - 절대 모멘텀: 본인 수익률 > 0
    - 상대 모멘텀: BTC 대비 우수
    """
    if btc_df.empty:
        return pd.Series(0, index=df.index)

    coin_ret = df["close"].pct_change(lookback)
    btc_ret = btc_df["close"].reindex(df.index, method="ffill").pct_change(lookback)

    # 상대 모멘텀: BTC 대비 초과 수익
    relative = coin_ret - btc_ret

    signal = pd.Series(0, index=df.index)
    # 절대 + 상대 모두 양수: Long
    signal[(coin_ret > 0) & (relative > 0)] = 1
    # 절대 + 상대 모두 음수: Short
    signal[(coin_ret < 0) & (relative < 0)] = -1

    return signal


# ============================================================================
# 백테스터
# ============================================================================


def backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    cost: CostModel,
    position_pct: float = 0.2,
    leverage: float = 2.0,
    max_bars: int = 36,
) -> Dict:
    """순차적 백테스트 (미래 정보 사용 안 함)"""

    if len(df) < 200:
        return None

    capital = 10000
    init = capital
    position = None
    trades = []
    equity = [capital]

    # 테스트 기간: 2023-01-01 이후
    test_start = df.index.get_indexer([pd.Timestamp("2023-01-01")], method="nearest")[0]
    test_start = max(test_start, 50)

    for i in range(test_start, len(df)):
        price = df["close"].iloc[i]
        sig = signals.iloc[i]
        prev_sig = signals.iloc[i - 1] if i > 0 else 0

        if position:
            bars = i - position["idx"]
            # 청산 조건: 신호 반전 또는 최대 보유 기간
            exit_cond = (position["dir"] != sig and sig != 0) or bars >= max_bars

            if exit_cond:
                gross_pnl = (price / position["entry"] - 1) * position["dir"]
                net_pnl = gross_pnl - cost.total_cost(bars)
                pnl = position["value"] * net_pnl
                trades.append(
                    {"pnl": pnl, "pct": net_pnl, "bars": bars, "dir": position["dir"]}
                )
                capital += pnl
                position = None

        # 진입 조건: 신호 발생 + 포지션 없음
        if not position and sig != 0 and capital > 0:
            value = capital * position_pct * leverage
            position = {"entry": price, "dir": sig, "idx": i, "value": value}

        equity.append(max(capital, 0))

    if len(trades) < 5:
        return None

    # 통계 계산
    pnls = [t["pnl"] for t in trades]
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

    # 평균 보유 기간
    avg_bars = np.mean([t["bars"] for t in trades])

    return {
        "pf": min(pf, 999),
        "ret": (capital / init - 1) * 100,
        "wr": wins / len(trades) * 100,
        "mdd": mdd,
        "trades": len(trades),
        "avg_bars": avg_bars,
        "max_loss_streak": max_streak,
    }


# ============================================================================
# 메인
# ============================================================================


def main():
    logger.info("=" * 70)
    logger.info("STRATEGY REDESIGN v1: 기본 전략 탐색")
    logger.info("=" * 70)
    logger.info("목표: Look-Ahead Bias 없이 실제 환경에서 작동하는 전략 찾기")
    logger.info("테스트 기간: 2023-01-01 ~ 현재")
    logger.info("=" * 70)

    loader = DataLoader()
    cost = CostModel()
    btc_df = loader.load("BTCUSDT")

    # 테스트 심볼 (유동성 상위)
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
    ]

    # 전략 정의
    strategies = {
        "TrendFollow": lambda df, btc: strategy_trend_follow(df, 20, 50),
        "Momentum": lambda df, btc: strategy_momentum(df, 20, 0.05),
        "Breakout": lambda df, btc: strategy_breakout(df, 20),
        "VolBreakout": lambda df, btc: strategy_volatility_breakout(df, 1.5, 14),
        "MeanRevert": lambda df, btc: strategy_mean_reversion(df, 20, 2.0),
        "VolumeSpike": lambda df, btc: strategy_volume_spike(df, 2.0, 20),
        "RSI_Extreme": lambda df, btc: strategy_rsi_extreme(df, 14, 30, 70),
        "DualMomentum": lambda df, btc: strategy_dual_momentum(df, btc, 30),
    }

    results = {name: [] for name in strategies}

    # 각 심볼에 대해 모든 전략 테스트
    for symbol in test_symbols:
        df = loader.load(symbol)
        if df.empty or len(df) < 500:
            continue

        df = df[df.index >= "2022-01-01"]

        for name, strat_func in strategies.items():
            signals = strat_func(df, btc_df)
            result = backtest(df, signals, cost)

            if result:
                result["symbol"] = symbol
                results[name].append(result)

    # 결과 요약
    logger.info(f"\n{'=' * 70}")
    logger.info("전략별 성과 요약")
    logger.info("=" * 70)

    summary = []

    logger.info(
        f"\n{'Strategy':<15} {'Symbols':>8} {'Profitable':>12} {'Avg PF':>10} {'Avg Ret':>10} {'Avg WR':>8} {'Avg MDD':>10}"
    )
    logger.info("-" * 80)

    for name, res_list in results.items():
        if not res_list:
            continue

        df_r = pd.DataFrame(res_list)
        n = len(res_list)
        profitable = (df_r["pf"] > 1.0).sum()
        avg_pf = df_r[df_r["pf"] < 999]["pf"].mean()
        avg_ret = df_r["ret"].mean()
        avg_wr = df_r["wr"].mean()
        avg_mdd = df_r["mdd"].mean()

        summary.append(
            {
                "strategy": name,
                "n": n,
                "profitable_pct": profitable / n * 100,
                "avg_pf": avg_pf,
                "avg_ret": avg_ret,
                "avg_wr": avg_wr,
                "avg_mdd": avg_mdd,
            }
        )

        logger.info(
            f"{name:<15} {n:>8} {profitable/n*100:>11.1f}% {avg_pf:>10.2f} "
            f"{avg_ret:>9.1f}% {avg_wr:>7.1f}% {avg_mdd:>9.1f}%"
        )

    # 최고 전략 찾기
    if summary:
        df_summary = pd.DataFrame(summary)
        best = df_summary.loc[df_summary["avg_pf"].idxmax()]

        logger.info(f"\n{'=' * 70}")
        logger.info(f"최고 전략: {best['strategy']}")
        logger.info("=" * 70)
        logger.info(f"  수익 심볼: {best['profitable_pct']:.1f}%")
        logger.info(f"  평균 PF: {best['avg_pf']:.2f}")
        logger.info(f"  평균 수익률: {best['avg_ret']:.1f}%")

        # 최고 전략의 심볼별 상세
        best_name = best["strategy"]
        best_results = results[best_name]
        df_best = pd.DataFrame(best_results)

        logger.info(f"\n[{best_name}] 심볼별 상세:")
        logger.info(f"{'Symbol':<14} {'PF':>8} {'Return':>10} {'WR':>8} {'Trades':>8}")
        logger.info("-" * 55)

        for _, r in df_best.sort_values("pf", ascending=False).iterrows():
            status = "PASS" if r["pf"] > 1.0 else "FAIL"
            logger.info(
                f"[{status}] {r['symbol']:<10} {r['pf']:>8.2f} {r['ret']:>9.1f}% "
                f"{r['wr']:>7.1f}% {r['trades']:>8}"
            )

    # 전략 조합 테스트
    logger.info(f"\n{'=' * 70}")
    logger.info("전략 조합 테스트")
    logger.info("=" * 70)

    # 상위 3개 전략 조합
    if len(summary) >= 3:
        df_summary = pd.DataFrame(summary)
        top3 = df_summary.nlargest(3, "avg_pf")["strategy"].tolist()

        logger.info(f"조합 대상: {', '.join(top3)}")

        combo_results = []

        for symbol in test_symbols:
            df = loader.load(symbol)
            if df.empty or len(df) < 500:
                continue

            df = df[df.index >= "2022-01-01"]

            # 각 전략의 신호
            signals_list = []
            for name in top3:
                sig = strategies[name](df, btc_df)
                signals_list.append(sig)

            # 다수결 (2/3 이상 동의)
            combo_signal = pd.Series(0, index=df.index)
            for i in range(len(df)):
                votes = sum(s.iloc[i] for s in signals_list)
                if votes >= 2:
                    combo_signal.iloc[i] = 1
                elif votes <= -2:
                    combo_signal.iloc[i] = -1

            result = backtest(df, combo_signal, cost)
            if result:
                result["symbol"] = symbol
                combo_results.append(result)

        if combo_results:
            df_combo = pd.DataFrame(combo_results)
            profitable = (df_combo["pf"] > 1.0).sum()

            logger.info(f"\n[조합 전략 결과]")
            logger.info(f"  테스트 심볼: {len(combo_results)}")
            logger.info(
                f"  수익 심볼: {profitable}/{len(combo_results)} ({profitable/len(combo_results)*100:.1f}%)"
            )
            logger.info(f"  평균 PF: {df_combo[df_combo['pf'] < 999]['pf'].mean():.2f}")
            logger.info(f"  평균 수익률: {df_combo['ret'].mean():.1f}%")
            logger.info(f"  평균 승률: {df_combo['wr'].mean():.1f}%")

            # 상세
            logger.info(f"\n심볼별 상세:")
            for _, r in df_combo.sort_values("pf", ascending=False).head(10).iterrows():
                status = "PASS" if r["pf"] > 1.0 else "FAIL"
                logger.info(
                    f"  [{status}] {r['symbol']:<12} PF={r['pf']:5.2f} Ret={r['ret']:+6.1f}%"
                )

    # 최종 권장사항
    logger.info(f"\n{'=' * 70}")
    logger.info("분석 결론")
    logger.info("=" * 70)

    if summary:
        df_summary = pd.DataFrame(summary)
        profitable_strategies = df_summary[df_summary["profitable_pct"] >= 50]

        if len(profitable_strategies) > 0:
            logger.info(f"\n50% 이상 수익 전략:")
            for _, s in profitable_strategies.iterrows():
                logger.info(
                    f"  - {s['strategy']}: {s['profitable_pct']:.1f}% 수익, PF {s['avg_pf']:.2f}"
                )
        else:
            logger.info("\n50% 이상 수익률을 보이는 단일 전략 없음")
            logger.info("전략 파라미터 최적화 또는 새로운 접근 필요")

    # 결과 저장
    all_results = []
    for name, res_list in results.items():
        for r in res_list:
            r["strategy"] = name
            all_results.append(r)

    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(DATA_ROOT / "strategy_redesign_v1.csv", index=False)
        logger.info(f"\n결과 저장: {DATA_ROOT / 'strategy_redesign_v1.csv'}")

    return results, summary


if __name__ == "__main__":
    results, summary = main()
