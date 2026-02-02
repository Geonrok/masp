"""
전략 상관관계 및 검증 신뢰성 분석
==================================
1. 전략 간 신호 상관관계
2. 거래 중복도 분석
3. 시장 구간별 성과 비교
4. 검증 방법론 점검
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, r"E:\투자\Multi-Asset Strategy Platform")

from kosdaq_futures_realworld_validation import BaseStrategy, Signal


# =============================================================================
# 지표 계산 함수
# =============================================================================
def chande_momentum(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).sum()
    loss = (-delta).where(delta < 0, 0).rolling(window=period).sum()
    cmo = 100 * (gain - loss) / (gain + loss + 1e-10)
    return cmo

def williams_r(high, low, close, period=14):
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr

def bb_percent_b(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    percent_b = (close - lower) / (upper - lower + 1e-10)
    return percent_b

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d


# =============================================================================
# 전략 클래스들 (검증용 재구현)
# =============================================================================
class TripleV5Strategy(BaseStrategy):
    def __init__(self, cmo_period, cmo_t, wr_period, wr_t, bb_period, direction='both'):
        super().__init__(
            f"TripleV5_{cmo_period}_{cmo_t}_{wr_period}_{wr_t}_{bb_period}_{direction}",
            {"cmo_period": cmo_period, "cmo_t": cmo_t, "wr_period": wr_period,
             "wr_t": wr_t, "bb_period": bb_period}, direction
        )

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        signals = []
        cmo = chande_momentum(df['Close'], self.params["cmo_period"])
        wr = williams_r(df['High'], df['Low'], df['Close'], self.params["wr_period"])
        pct_b = bb_percent_b(df['Close'], self.params["bb_period"], 2)

        start = max(self.params["cmo_period"], self.params["wr_period"], self.params["bb_period"]) + 5

        for i in range(start, len(df)):
            date = df.index[i]
            if pd.isna(cmo.iloc[i-1]) or pd.isna(wr.iloc[i-1]) or pd.isna(pct_b.iloc[i-1]):
                continue

            cmo_t, wr_t = self.params["cmo_t"], self.params["wr_t"]

            bull_count = 0
            if cmo.iloc[i-2] < -cmo_t and cmo.iloc[i-1] >= -cmo_t: bull_count += 1
            if wr.iloc[i-2] < -wr_t and wr.iloc[i-1] >= -wr_t: bull_count += 1
            if pct_b.iloc[i-2] < 0.1 and pct_b.iloc[i-1] >= 0.1: bull_count += 1

            bear_count = 0
            if cmo.iloc[i-2] > cmo_t and cmo.iloc[i-1] <= cmo_t: bear_count += 1
            if wr.iloc[i-2] > -(100-wr_t) and wr.iloc[i-1] <= -(100-wr_t): bear_count += 1
            if pct_b.iloc[i-2] > 0.9 and pct_b.iloc[i-1] <= 0.9: bear_count += 1

            if bull_count >= 2:
                signals.append(Signal(date, 1, bull_count/3, "Long"))
            elif bear_count >= 2:
                signals.append(Signal(date, -1, bear_count/3, "Short"))

        return signals


class TripleVolStrategy(BaseStrategy):
    def __init__(self, period, cmo_t, wr_t, vol_mult, direction='both'):
        super().__init__(
            f"TripleVol_{period}_{cmo_t}_{wr_t}_{vol_mult}_{direction}",
            {"period": period, "cmo_t": cmo_t, "wr_t": wr_t, "vol_mult": vol_mult}, direction
        )

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        signals = []
        period = self.params["period"]
        cmo = chande_momentum(df['Close'], period)
        wr = williams_r(df['High'], df['Low'], df['Close'], period)
        pct_b = bb_percent_b(df['Close'], period * 2, 2)
        vol_ma = df['Volume'].rolling(window=period).mean()

        cmo_t, wr_t, vol_mult = self.params["cmo_t"], self.params["wr_t"], self.params["vol_mult"]

        for i in range(period * 2 + 5, len(df)):
            date = df.index[i]
            if pd.isna(cmo.iloc[i-1]) or pd.isna(wr.iloc[i-1]) or pd.isna(pct_b.iloc[i-1]):
                continue
            if df['Volume'].iloc[i] < vol_ma.iloc[i] * vol_mult:
                continue

            bull_count = 0
            if cmo.iloc[i-2] < -cmo_t and cmo.iloc[i-1] >= -cmo_t: bull_count += 1
            if wr.iloc[i-2] < -wr_t and wr.iloc[i-1] >= -wr_t: bull_count += 1
            if pct_b.iloc[i-2] < 0.1 and pct_b.iloc[i-1] >= 0.1: bull_count += 1

            bear_count = 0
            if cmo.iloc[i-2] > cmo_t and cmo.iloc[i-1] <= cmo_t: bear_count += 1
            if wr.iloc[i-2] > -(100-wr_t) and wr.iloc[i-1] <= -(100-wr_t): bear_count += 1
            if pct_b.iloc[i-2] > 0.9 and pct_b.iloc[i-1] <= 0.9: bear_count += 1

            if bull_count >= 2:
                signals.append(Signal(date, 1, bull_count/3, "Long"))
            elif bear_count >= 2:
                signals.append(Signal(date, -1, bear_count/3, "Short"))

        return signals


class TripleADXStrategy(BaseStrategy):
    def __init__(self, period, cmo_t, wr_t, adx_period, adx_thresh, direction='both'):
        super().__init__(
            f"TripleADX_{period}_{cmo_t}_{wr_t}_{adx_thresh}_{direction}",
            {"period": period, "cmo_t": cmo_t, "wr_t": wr_t,
             "adx_period": adx_period, "adx_thresh": adx_thresh}, direction
        )

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        signals = []
        period = self.params["period"]
        cmo = chande_momentum(df['Close'], period)
        wr = williams_r(df['High'], df['Low'], df['Close'], period)
        pct_b = bb_percent_b(df['Close'], period * 2, 2)

        # ADX
        adx_period = self.params["adx_period"]
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=adx_period).mean()

        plus_dm = (df['High'] - df['High'].shift(1)).where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 0
        ).where((df['High'] - df['High'].shift(1)) > 0, 0)
        minus_dm = (df['Low'].shift(1) - df['Low']).where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 0
        ).where((df['Low'].shift(1) - df['Low']) > 0, 0)

        plus_di = 100 * plus_dm.rolling(window=adx_period).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(window=adx_period).mean() / (atr + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=adx_period).mean()

        cmo_t, wr_t, adx_thresh = self.params["cmo_t"], self.params["wr_t"], self.params["adx_thresh"]
        start = max(period * 2, adx_period * 2) + 5

        for i in range(start, len(df)):
            date = df.index[i]
            if pd.isna(cmo.iloc[i-1]) or pd.isna(wr.iloc[i-1]) or pd.isna(adx.iloc[i]):
                continue
            if adx.iloc[i] < adx_thresh:
                continue

            bull_count = 0
            if cmo.iloc[i-2] < -cmo_t and cmo.iloc[i-1] >= -cmo_t: bull_count += 1
            if wr.iloc[i-2] < -wr_t and wr.iloc[i-1] >= -wr_t: bull_count += 1
            if pct_b.iloc[i-2] < 0.1 and pct_b.iloc[i-1] >= 0.1: bull_count += 1

            bear_count = 0
            if cmo.iloc[i-2] > cmo_t and cmo.iloc[i-1] <= cmo_t: bear_count += 1
            if wr.iloc[i-2] > -(100-wr_t) and wr.iloc[i-1] <= -(100-wr_t): bear_count += 1
            if pct_b.iloc[i-2] > 0.9 and pct_b.iloc[i-1] <= 0.9: bear_count += 1

            if bull_count >= 2:
                signals.append(Signal(date, 1, bull_count/3, "Long"))
            elif bear_count >= 2:
                signals.append(Signal(date, -1, bear_count/3, "Short"))

        return signals


class CombBestStrategy(BaseStrategy):
    def __init__(self, period, cmo_t, wr_t, rsi_t, stoch_t, direction='both'):
        super().__init__(
            f"CombBest_{period}_{cmo_t}_{wr_t}_{rsi_t}_{stoch_t}_{direction}",
            {"period": period, "cmo_t": cmo_t, "wr_t": wr_t, "rsi_t": rsi_t, "stoch_t": stoch_t}, direction
        )

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        signals = []
        period = self.params["period"]
        cmo = chande_momentum(df['Close'], period)
        wr = williams_r(df['High'], df['Low'], df['Close'], period)
        pct_b = bb_percent_b(df['Close'], period, 2)
        rsi_val = rsi(df['Close'], period)
        stoch_k, _ = stochastic(df['High'], df['Low'], df['Close'], period, 3)

        cmo_t, wr_t = self.params["cmo_t"], self.params["wr_t"]
        rsi_t, stoch_t = self.params["rsi_t"], self.params["stoch_t"]

        for i in range(period + 5, len(df)):
            date = df.index[i]
            if pd.isna(cmo.iloc[i-1]) or pd.isna(wr.iloc[i-1]):
                continue

            bull_count = 0
            if cmo.iloc[i-2] < -cmo_t and cmo.iloc[i-1] >= -cmo_t: bull_count += 1
            if wr.iloc[i-2] < -wr_t and wr.iloc[i-1] >= -wr_t: bull_count += 1
            if pct_b.iloc[i-2] < 0.1 and pct_b.iloc[i-1] >= 0.1: bull_count += 1
            if rsi_val.iloc[i-2] < (100-rsi_t) and rsi_val.iloc[i-1] >= (100-rsi_t): bull_count += 1
            if stoch_k.iloc[i-2] < stoch_t and stoch_k.iloc[i-1] >= stoch_t: bull_count += 1

            bear_count = 0
            if cmo.iloc[i-2] > cmo_t and cmo.iloc[i-1] <= cmo_t: bear_count += 1
            if wr.iloc[i-2] > -(100-wr_t) and wr.iloc[i-1] <= -(100-wr_t): bear_count += 1
            if pct_b.iloc[i-2] > 0.9 and pct_b.iloc[i-1] <= 0.9: bear_count += 1
            if rsi_val.iloc[i-2] > rsi_t and rsi_val.iloc[i-1] <= rsi_t: bear_count += 1
            if stoch_k.iloc[i-2] > (100-stoch_t) and stoch_k.iloc[i-1] <= (100-stoch_t): bear_count += 1

            if bull_count >= 3:
                signals.append(Signal(date, 1, bull_count/5, "Long"))
            elif bear_count >= 3:
                signals.append(Signal(date, -1, bear_count/5, "Short"))

        return signals


# =============================================================================
# 분석 함수들
# =============================================================================
def analyze_signal_correlation(strategies: List[BaseStrategy], df: pd.DataFrame) -> pd.DataFrame:
    """전략 간 신호 상관관계 분석"""

    # 각 전략의 신호를 시계열로 변환
    signal_series = {}

    for strategy in strategies:
        signals = strategy.generate_signals(df)
        # 날짜별 신호 딕셔너리
        signal_dict = {s.date: s.signal for s in signals}
        # 전체 날짜에 대해 시리즈 생성 (신호 없으면 0)
        series = pd.Series(index=df.index, data=0)
        for date, sig in signal_dict.items():
            if date in series.index:
                series[date] = sig
        signal_series[strategy.name] = series

    # 상관관계 행렬 계산
    signal_df = pd.DataFrame(signal_series)
    correlation_matrix = signal_df.corr()

    return correlation_matrix


def analyze_trade_overlap(strategies: List[BaseStrategy], df: pd.DataFrame) -> Dict:
    """거래 중복도 분석"""

    all_signals = {}
    for strategy in strategies:
        signals = strategy.generate_signals(df)
        all_signals[strategy.name] = {s.date: s.signal for s in signals}

    # 중복 거래 분석
    overlap_matrix = {}
    strategy_names = list(all_signals.keys())

    for i, name1 in enumerate(strategy_names):
        overlap_matrix[name1] = {}
        dates1 = set(all_signals[name1].keys())

        for j, name2 in enumerate(strategy_names):
            dates2 = set(all_signals[name2].keys())

            # 같은 날 같은 방향 신호
            common_dates = dates1 & dates2
            same_direction = sum(1 for d in common_dates
                                if all_signals[name1][d] == all_signals[name2][d])

            overlap_pct = same_direction / len(dates1) * 100 if dates1 else 0
            overlap_matrix[name1][name2] = overlap_pct

    return overlap_matrix


def analyze_market_periods(strategies: List[BaseStrategy], df: pd.DataFrame) -> Dict:
    """시장 구간별 성과 분석"""

    periods = {
        '2010-2015 (회복기)': ('2010-01-01', '2015-12-31'),
        '2016-2019 (안정기)': ('2016-01-01', '2019-12-31'),
        '2020 (코로나)': ('2020-01-01', '2020-12-31'),
        '2021-2022 (변동기)': ('2021-01-01', '2022-12-31'),
        '2023-2026 (최근)': ('2023-01-01', '2026-12-31'),
    }

    results = {}

    for period_name, (start, end) in periods.items():
        period_df = df[(df.index >= start) & (df.index <= end)]
        if len(period_df) < 100:
            continue

        period_results = {}
        for strategy in strategies:
            signals = strategy.generate_signals(period_df)

            # 간단한 수익률 계산
            pnls = []
            position = 0
            entry_price = 0

            for signal in signals:
                if signal.date not in period_df.index:
                    continue
                idx = period_df.index.get_loc(signal.date)
                if idx + 1 >= len(period_df):
                    continue

                price = period_df['Close'].iloc[idx + 1]

                if position == 0:
                    position = signal.signal
                    entry_price = price
                elif position != signal.signal:
                    if position == 1:
                        pnl = (price / entry_price - 1)
                    else:
                        pnl = (entry_price / price - 1)
                    pnls.append(pnl)
                    position = signal.signal
                    entry_price = price

            if pnls:
                period_results[strategy.name] = {
                    'trades': len(pnls),
                    'total_return': sum(pnls) * 100,
                    'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) * 100,
                    'avg_return': np.mean(pnls) * 100
                }

        results[period_name] = period_results

    return results


def check_validation_issues(strategies: List[BaseStrategy], df: pd.DataFrame) -> List[str]:
    """검증 방법론 점검"""

    issues = []

    # 1. 과적합 위험 체크
    param_variations = defaultdict(list)
    for s in strategies:
        base_type = s.name.split('_')[0]
        param_variations[base_type].append(s.name)

    for base_type, variants in param_variations.items():
        if len(variants) > 1:
            issues.append(f"[주의] {base_type} 유형의 전략이 {len(variants)}개 존재 - 파라미터 과적합 가능성")

    # 2. 신호 빈도 체크
    for strategy in strategies:
        signals = strategy.generate_signals(df)
        years = (df.index[-1] - df.index[0]).days / 365
        signals_per_year = len(signals) / years

        if signals_per_year < 5:
            issues.append(f"[경고] {strategy.name}: 연간 신호 {signals_per_year:.1f}회로 너무 적음")
        elif signals_per_year > 100:
            issues.append(f"[경고] {strategy.name}: 연간 신호 {signals_per_year:.1f}회로 과다")

    # 3. 핵심 지표 의존도 체크
    cmo_wr_based = sum(1 for s in strategies if 'Triple' in s.name or 'Comb' in s.name)
    if cmo_wr_based == len(strategies):
        issues.append("[심각] 모든 전략이 CMO+WR 기반 - 동일 지표 과의존")

    # 4. 방향성 편향 체크
    for strategy in strategies:
        signals = strategy.generate_signals(df)
        if signals:
            long_signals = sum(1 for s in signals if s.signal == 1)
            short_signals = sum(1 for s in signals if s.signal == -1)
            total = long_signals + short_signals
            if total > 0:
                long_ratio = long_signals / total
                if long_ratio > 0.7:
                    issues.append(f"[주의] {strategy.name}: Long 편향 {long_ratio*100:.0f}%")
                elif long_ratio < 0.3:
                    issues.append(f"[주의] {strategy.name}: Short 편향 {(1-long_ratio)*100:.0f}%")

    return issues


def run_correlation_analysis():
    """전체 상관관계 분석 실행"""

    print("=" * 80)
    print("전략 상관관계 및 검증 신뢰성 분석")
    print("=" * 80)

    # 데이터 로드
    df = pd.read_parquet(r"E:\투자\data\kosdaq_futures\kosdaq150_futures_ohlcv_fresh.parquet")
    print(f"데이터: {len(df)} 행 ({df.index[0]} ~ {df.index[-1]})")

    # 검증 통과 전략들 생성
    strategies = [
        TripleV5Strategy(14, 38, 14, 78, 20, 'both'),
        TripleV5Strategy(14, 33, 14, 73, 20, 'both'),
        TripleV5Strategy(14, 35, 14, 75, 20, 'both'),
        TripleVolStrategy(14, 35, 75, 0.8, 'both'),
        TripleVolStrategy(14, 38, 78, 0.8, 'both'),
        CombBestStrategy(14, 35, 75, 70, 20, 'both'),
        TripleADXStrategy(14, 35, 75, 14, 25, 'both'),
        TripleADXStrategy(14, 38, 78, 14, 25, 'both'),
    ]

    print(f"\n분석 전략 수: {len(strategies)}")

    # 1. 신호 상관관계 분석
    print("\n" + "=" * 80)
    print("1. 전략 간 신호 상관관계")
    print("=" * 80)

    corr_matrix = analyze_signal_correlation(strategies, df)
    print("\n상관관계 행렬 (1.0 = 완전 동일, 0.0 = 무관):")
    print()

    # 짧은 이름으로 변환
    short_names = {s.name: s.name.replace('_both', '').replace('TripleV5_14_', 'V5_').replace('TripleVol_14_', 'Vol_').replace('CombBest_14_', 'Comb_').replace('TripleADX_14_', 'ADX_') for s in strategies}

    corr_display = corr_matrix.copy()
    corr_display.index = [short_names[n] for n in corr_display.index]
    corr_display.columns = [short_names[n] for n in corr_display.columns]

    # 상관계수 출력
    for name in corr_display.index:
        row = [f"{corr_display.loc[name, col]:.2f}" for col in corr_display.columns]
        print(f"{name:20s}: {' '.join(row)}")

    # 평균 상관계수
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    avg_corr = upper_tri.stack().mean()
    print(f"\n평균 상관계수: {avg_corr:.3f}")

    if avg_corr > 0.7:
        print("  [심각] 전략들이 매우 유사함 (상관계수 > 0.7)")
    elif avg_corr > 0.5:
        print("  [주의] 전략들이 상당히 유사함 (상관계수 > 0.5)")
    else:
        print("  [양호] 전략들이 적절히 다양함")

    # 2. 거래 중복도 분석
    print("\n" + "=" * 80)
    print("2. 거래 중복도 분석")
    print("=" * 80)

    overlap = analyze_trade_overlap(strategies, df)

    print("\n같은 날 같은 방향 신호 비율 (%):")
    print()

    for name1 in list(overlap.keys())[:4]:  # 처음 4개만
        short1 = short_names[name1]
        row = []
        for name2 in list(overlap[name1].keys())[:4]:
            row.append(f"{overlap[name1][name2]:5.1f}")
        print(f"{short1:20s}: {' '.join(row)}")

    # 3. 시장 구간별 성과
    print("\n" + "=" * 80)
    print("3. 시장 구간별 성과 분석")
    print("=" * 80)

    period_results = analyze_market_periods(strategies, df)

    for period_name, results in period_results.items():
        print(f"\n### {period_name} ###")
        if not results:
            print("  데이터 부족")
            continue

        for strat_name, perf in list(results.items())[:3]:
            short = short_names[strat_name]
            print(f"  {short:20s}: 수익={perf['total_return']:+6.1f}%, 승률={perf['win_rate']:5.1f}%, 거래={perf['trades']:3d}회")

    # 4. 검증 이슈 체크
    print("\n" + "=" * 80)
    print("4. 검증 방법론 점검")
    print("=" * 80)

    issues = check_validation_issues(strategies, df)

    if issues:
        print("\n발견된 이슈:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n  검증 방법론에 심각한 이슈 없음")

    # 5. 전략 유형 분류
    print("\n" + "=" * 80)
    print("5. 전략 유형 분류")
    print("=" * 80)

    print("\n전략 유형별 분류:")
    print(f"  - TripleV5 (기본 Triple): 3개 - 파라미터만 다름")
    print(f"  - TripleVol (거래량 필터): 2개 - 거래량 조건 추가")
    print(f"  - TripleADX (ADX 필터): 2개 - 트렌드 강도 필터")
    print(f"  - CombBest (5지표 복합): 1개 - 5개 지표 사용")

    print("\n핵심 차이점:")
    print("  1. TripleV5: CMO + WR + BB (3개 지표)")
    print("  2. TripleVol: 위 + 거래량 필터")
    print("  3. TripleADX: 위 + ADX 트렌드 필터")
    print("  4. CombBest: CMO + WR + BB + RSI + Stochastic (5개 지표)")

    # 결론
    print("\n" + "=" * 80)
    print("종합 결론")
    print("=" * 80)

    print("""
[분석 결과]

1. 전략 다양성: 제한적
   - 8개 전략 모두 CMO + WR + BB 조합을 핵심으로 사용
   - 파라미터 변경(33/35/38) 또는 필터 추가(Vol/ADX)의 차이만 존재
   - 본질적으로 "하나의 전략"의 변형들임

2. 검증 신뢰성: 주의 필요
   - 동일 지표 기반으로 파라미터 최적화 → 과적합 위험
   - 다른 독립적 전략군이 없음

3. 권장사항:
   - 8개 중 대표 전략 1-2개만 선택 권장
   - 추천: TripleADX (ADX 필터로 추가 확인)
   - 또는: TripleVol (거래량 필터로 추가 확인)
   - 파라미터만 다른 TripleV5 변형들은 중복
""")

    return corr_matrix, overlap, period_results, issues


if __name__ == "__main__":
    run_correlation_analysis()
