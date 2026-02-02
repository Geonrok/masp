# -*- coding: utf-8 -*-
"""
키움 REST API 외국인 수급 데이터 기반 백테스트
- 17개 종목의 외국인 보유/매매 데이터 활용
- 다양한 외국인 수급 전략 테스트
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 경로
DATA_PATH = "E:/투자/data/kiwoom_investor"
RESULT_PATH = "E:/투자/data/kiwoom_backtest_results"
os.makedirs(RESULT_PATH, exist_ok=True)


class ForeignInvestorStrategy:
    """외국인 투자자 수급 기반 전략 베이스 클래스"""

    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}

    def generate_signals(self, df):
        raise NotImplementedError


class ForeignNetBuyMomentum(ForeignInvestorStrategy):
    """외국인 순매수 모멘텀
    - 외국인이 N일간 순매수하면 매수
    - 외국인이 N일간 순매도하면 매도
    """
    def generate_signals(self, df):
        period = self.params.get('period', 5)
        threshold = self.params.get('threshold', 0)

        df = df.copy()
        df['foreign_sum'] = df['chg_qty'].rolling(period).sum()

        df['signal'] = 0
        df.loc[df['foreign_sum'] > threshold, 'signal'] = 1
        df.loc[df['foreign_sum'] < -threshold, 'signal'] = -1

        return df


class ForeignRatioMomentum(ForeignInvestorStrategy):
    """외국인 지분율 변화 모멘텀
    - 외국인 지분율이 상승하면 매수
    - 외국인 지분율이 하락하면 매도
    """
    def generate_signals(self, df):
        period = self.params.get('period', 5)

        df = df.copy()
        df['wght_change'] = df['wght'].diff(period)

        df['signal'] = 0
        df.loc[df['wght_change'] > 0, 'signal'] = 1
        df.loc[df['wght_change'] < 0, 'signal'] = -1

        return df


class ForeignAccelerationStrategy(ForeignInvestorStrategy):
    """외국인 매수 가속화 전략
    - 최근 5일 순매수가 이전 5일보다 크면 매수
    - 반대면 매도
    """
    def generate_signals(self, df):
        period = self.params.get('period', 5)

        df = df.copy()
        df['recent_sum'] = df['chg_qty'].rolling(period).sum()
        df['prev_sum'] = df['chg_qty'].shift(period).rolling(period).sum()

        df['signal'] = 0
        df.loc[df['recent_sum'] > df['prev_sum'], 'signal'] = 1
        df.loc[df['recent_sum'] < df['prev_sum'], 'signal'] = -1

        return df


class ForeignRatioThreshold(ForeignInvestorStrategy):
    """외국인 지분율 임계값 전략
    - 외국인 지분율이 특정 값 이상이면 매수
    - 특정 값 이하면 매도
    """
    def generate_signals(self, df):
        buy_threshold = self.params.get('buy_threshold', 30)
        sell_threshold = self.params.get('sell_threshold', 20)

        df = df.copy()
        df['signal'] = 0
        df.loc[df['wght'] >= buy_threshold, 'signal'] = 1
        df.loc[df['wght'] < sell_threshold, 'signal'] = -1

        return df


class ForeignRatioMA(ForeignInvestorStrategy):
    """외국인 지분율 이동평균 크로스오버
    - 지분율이 이동평균 상향 돌파 시 매수
    - 하향 돌파 시 매도
    """
    def generate_signals(self, df):
        period = self.params.get('period', 20)

        df = df.copy()
        df['wght_ma'] = df['wght'].rolling(period).mean()

        df['signal'] = 0
        df.loc[df['wght'] > df['wght_ma'], 'signal'] = 1
        df.loc[df['wght'] < df['wght_ma'], 'signal'] = -1

        return df


class ForeignPriceDivergence(ForeignInvestorStrategy):
    """외국인 순매수 - 가격 다이버전스
    - 가격 하락 + 외국인 순매수 증가 = 매수 신호
    - 가격 상승 + 외국인 순매도 증가 = 매도 신호
    """
    def generate_signals(self, df):
        period = self.params.get('period', 10)

        df = df.copy()
        df['price_change'] = df['cur_prc'].pct_change(period)
        df['foreign_change'] = df['chg_qty'].rolling(period).sum()

        df['signal'] = 0
        # 가격 하락 + 외국인 순매수 = 매수
        df.loc[(df['price_change'] < 0) & (df['foreign_change'] > 0), 'signal'] = 1
        # 가격 상승 + 외국인 순매도 = 매도
        df.loc[(df['price_change'] > 0) & (df['foreign_change'] < 0), 'signal'] = -1

        return df


class ForeignExtreme(ForeignInvestorStrategy):
    """외국인 극단적 매수/매도 전략
    - 외국인 순매수가 상위 N%면 매수
    - 하위 N%면 매도
    """
    def generate_signals(self, df):
        percentile = self.params.get('percentile', 20)
        lookback = self.params.get('lookback', 60)

        df = df.copy()
        df['foreign_upper'] = df['chg_qty'].rolling(lookback).apply(
            lambda x: np.percentile(x, 100-percentile), raw=True)
        df['foreign_lower'] = df['chg_qty'].rolling(lookback).apply(
            lambda x: np.percentile(x, percentile), raw=True)

        df['signal'] = 0
        df.loc[df['chg_qty'] > df['foreign_upper'], 'signal'] = 1
        df.loc[df['chg_qty'] < df['foreign_lower'], 'signal'] = -1

        return df


class ForeignTrendFollowing(ForeignInvestorStrategy):
    """외국인 추세 추종
    - 5일, 10일, 20일 순매수가 모두 양수면 매수
    - 모두 음수면 매도
    """
    def generate_signals(self, df):
        df = df.copy()

        # 이미 계산된 누적 순매수 사용
        df['signal'] = 0
        df.loc[(df['foreign_net_5d'] > 0) & (df['foreign_net_10d'] > 0) & (df['foreign_net_20d'] > 0), 'signal'] = 1
        df.loc[(df['foreign_net_5d'] < 0) & (df['foreign_net_10d'] < 0) & (df['foreign_net_20d'] < 0), 'signal'] = -1

        return df


class ForeignMeanReversion(ForeignInvestorStrategy):
    """외국인 순매수 평균회귀
    - 순매수가 평균 대비 많이 낮으면 매수 (반등 기대)
    - 순매수가 평균 대비 많이 높으면 매도 (조정 기대)
    """
    def generate_signals(self, df):
        period = self.params.get('period', 20)
        std_mult = self.params.get('std_mult', 2)

        df = df.copy()
        df['foreign_ma'] = df['chg_qty'].rolling(period).mean()
        df['foreign_std'] = df['chg_qty'].rolling(period).std()

        df['signal'] = 0
        # 순매수가 평균 대비 많이 낮음 = 매수 (반등 기대)
        df.loc[df['chg_qty'] < df['foreign_ma'] - std_mult * df['foreign_std'], 'signal'] = 1
        # 순매수가 평균 대비 많이 높음 = 매도 (조정 기대)
        df.loc[df['chg_qty'] > df['foreign_ma'] + std_mult * df['foreign_std'], 'signal'] = -1

        return df


def backtest_strategy(df, strategy, initial_capital=100_000_000, fee_rate=0.001):
    """전략 백테스트 수행"""
    df = strategy.generate_signals(df)
    df = df.dropna(subset=['chg_qty', 'signal'])

    if len(df) < 100:
        return None

    capital = initial_capital
    position = 0
    shares = 0
    entry_price = 0
    trades = []

    for i, row in df.iterrows():
        price = float(row['cur_prc'])
        signal = int(row['signal'])
        date = row['dt']

        if signal == 1 and position == 0:  # 매수
            shares = int(capital * 0.95 / price)  # 95% 투자
            cost = shares * price * (1 + fee_rate)
            if cost <= capital:
                capital -= cost
                position = 1
                entry_price = price
                trades.append({
                    'date': date, 'type': 'BUY', 'price': price,
                    'shares': shares, 'capital': capital
                })

        elif signal == -1 and position == 1:  # 매도
            proceeds = shares * price * (1 - fee_rate)
            capital += proceeds
            pnl = (price - entry_price) / entry_price
            trades.append({
                'date': date, 'type': 'SELL', 'price': price,
                'shares': shares, 'capital': capital, 'pnl': pnl
            })
            position = 0
            shares = 0

    # 최종 청산
    if position == 1:
        final_price = float(df.iloc[-1]['cur_prc'])
        capital += shares * final_price * (1 - fee_rate)

    # 성과 계산
    total_return = (capital - initial_capital) / initial_capital
    num_trades = len([t for t in trades if t['type'] == 'SELL'])

    if num_trades == 0:
        return None

    win_trades = [t for t in trades if t['type'] == 'SELL' and t.get('pnl', 0) > 0]
    win_rate = len(win_trades) / num_trades if num_trades > 0 else 0

    avg_pnl = np.mean([t.get('pnl', 0) for t in trades if t['type'] == 'SELL'])

    # 연환산 수익률
    years = len(df) / 252
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0

    return {
        'strategy': strategy.name,
        'params': str(strategy.params),
        'total_return': total_return,
        'annual_return': annual_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_trade_pnl': avg_pnl,
        'final_capital': capital,
        'data_points': len(df),
    }


def run_all_backtests():
    """모든 종목과 전략에 대해 백테스트 수행"""
    print("=" * 70)
    print("키움 REST API 외국인 수급 전략 백테스트")
    print("=" * 70)

    # 전략 목록
    strategies = [
        # 외국인 순매수 모멘텀 (다양한 기간)
        ForeignNetBuyMomentum("Foreign_NetBuy_5d", {'period': 5, 'threshold': 0}),
        ForeignNetBuyMomentum("Foreign_NetBuy_10d", {'period': 10, 'threshold': 0}),
        ForeignNetBuyMomentum("Foreign_NetBuy_20d", {'period': 20, 'threshold': 0}),

        # 외국인 지분율 변화
        ForeignRatioMomentum("Foreign_Ratio_5d", {'period': 5}),
        ForeignRatioMomentum("Foreign_Ratio_10d", {'period': 10}),
        ForeignRatioMomentum("Foreign_Ratio_20d", {'period': 20}),

        # 외국인 매수 가속화
        ForeignAccelerationStrategy("Foreign_Accel_5d", {'period': 5}),
        ForeignAccelerationStrategy("Foreign_Accel_10d", {'period': 10}),

        # 외국인 지분율 임계값
        ForeignRatioThreshold("Foreign_Threshold_30_20", {'buy_threshold': 30, 'sell_threshold': 20}),
        ForeignRatioThreshold("Foreign_Threshold_40_30", {'buy_threshold': 40, 'sell_threshold': 30}),
        ForeignRatioThreshold("Foreign_Threshold_50_40", {'buy_threshold': 50, 'sell_threshold': 40}),

        # 외국인 지분율 이동평균
        ForeignRatioMA("Foreign_RatioMA_10d", {'period': 10}),
        ForeignRatioMA("Foreign_RatioMA_20d", {'period': 20}),

        # 외국인-가격 다이버전스
        ForeignPriceDivergence("Foreign_Divergence_10d", {'period': 10}),
        ForeignPriceDivergence("Foreign_Divergence_20d", {'period': 20}),

        # 외국인 극단적 매수/매도
        ForeignExtreme("Foreign_Extreme_20pct", {'percentile': 20, 'lookback': 60}),
        ForeignExtreme("Foreign_Extreme_10pct", {'percentile': 10, 'lookback': 60}),

        # 외국인 추세 추종
        ForeignTrendFollowing("Foreign_TrendFollow", {}),

        # 외국인 평균회귀
        ForeignMeanReversion("Foreign_MeanRev_2std", {'period': 20, 'std_mult': 2}),
        ForeignMeanReversion("Foreign_MeanRev_1.5std", {'period': 20, 'std_mult': 1.5}),
    ]

    # 병합 데이터 파일 로드
    merged_files = [f for f in os.listdir(DATA_PATH) if f.endswith('_merged.csv')]
    print(f"백테스트 대상 종목: {len(merged_files)}개")
    print(f"테스트 전략 수: {len(strategies)}개")
    print()

    all_results = []

    for file in sorted(merged_files):
        ticker = file.replace('_merged.csv', '')
        filepath = os.path.join(DATA_PATH, file)

        df = pd.read_csv(filepath, encoding='utf-8-sig')
        df['dt'] = pd.to_datetime(df['dt'])

        # 외국인 데이터가 있는 행만 사용
        df = df[df['chg_qty'].notna()].copy()

        if len(df) < 100:
            print(f"{ticker}: 데이터 부족 ({len(df)}행), 스킵")
            continue

        name = df['name'].iloc[0] if 'name' in df.columns else ticker
        print(f"\n{name} ({ticker}) - {len(df)}일 데이터")

        for strategy in strategies:
            result = backtest_strategy(df, strategy)
            if result:
                result['ticker'] = ticker
                result['name'] = name
                all_results.append(result)

                if result['total_return'] > 0:
                    print(f"  {strategy.name}: {result['total_return']*100:.1f}% "
                          f"(승률: {result['win_rate']*100:.0f}%, 거래: {result['num_trades']}회)")

    # 결과 저장
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('total_return', ascending=False)

        # CSV 저장
        results_df.to_csv(f"{RESULT_PATH}/foreign_strategy_results.csv",
                         index=False, encoding='utf-8-sig')

        # 요약 출력
        print("\n" + "=" * 70)
        print("백테스트 결과 요약")
        print("=" * 70)

        print(f"\n총 백테스트 수: {len(results_df)}")
        print(f"수익 전략: {(results_df['total_return'] > 0).sum()}개")
        print(f"손실 전략: {(results_df['total_return'] < 0).sum()}개")

        # 전략별 평균 성과
        print("\n[전략별 평균 성과 (상위 10)]")
        strategy_avg = results_df.groupby('strategy').agg({
            'total_return': 'mean',
            'annual_return': 'mean',
            'win_rate': 'mean',
            'num_trades': 'mean'
        }).sort_values('total_return', ascending=False)

        for i, (strategy, row) in enumerate(strategy_avg.head(10).iterrows()):
            print(f"  {i+1}. {strategy}: 평균수익 {row['total_return']*100:.1f}%, "
                  f"연환산 {row['annual_return']*100:.1f}%, 승률 {row['win_rate']*100:.0f}%")

        # 종목별 최고 성과
        print("\n[종목별 최고 성과]")
        for ticker in results_df['ticker'].unique():
            ticker_df = results_df[results_df['ticker'] == ticker]
            best = ticker_df.loc[ticker_df['total_return'].idxmax()]
            name = best['name']
            print(f"  {name} ({ticker}): {best['strategy']} = {best['total_return']*100:.1f}%")

        # 전체 최고 성과 Top 20
        print("\n[전체 최고 성과 Top 20]")
        top20 = results_df.head(20)
        for i, row in top20.iterrows():
            print(f"  {row['name']} / {row['strategy']}: "
                  f"{row['total_return']*100:.1f}% (승률: {row['win_rate']*100:.0f}%)")

        print(f"\n결과 저장: {RESULT_PATH}/foreign_strategy_results.csv")

    return results_df if all_results else None


if __name__ == "__main__":
    results = run_all_backtests()
