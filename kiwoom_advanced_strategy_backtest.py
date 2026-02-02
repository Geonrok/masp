# -*- coding: utf-8 -*-
"""
키움 데이터 기반 고급 복합 전략 백테스트
- 기술적 지표 + 외국인 수급 결합
- 추세 필터 + 수급 신호
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "E:/투자/data/kiwoom_investor"
RESULT_PATH = "E:/투자/data/kiwoom_backtest_results"
os.makedirs(RESULT_PATH, exist_ok=True)


class AdvancedStrategy:
    """고급 복합 전략 베이스 클래스"""
    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}

    def generate_signals(self, df):
        raise NotImplementedError


# ============== 추세 필터 + 수급 전략 ==============

class TrendFilterForeign(AdvancedStrategy):
    """추세 필터 + 외국인 순매수
    - 가격이 이동평균 위에 있을 때만 외국인 순매수 신호 따름
    """
    def generate_signals(self, df):
        ma_period = self.params.get('ma_period', 50)
        foreign_period = self.params.get('foreign_period', 5)

        df = df.copy()
        df['ma'] = df['cur_prc'].rolling(ma_period).mean()
        df['foreign_sum'] = df['chg_qty'].rolling(foreign_period).sum()

        df['signal'] = 0
        # 가격 > MA 이고 외국인 순매수 = 매수
        df.loc[(df['cur_prc'] > df['ma']) & (df['foreign_sum'] > 0), 'signal'] = 1
        # 가격 < MA 이고 외국인 순매도 = 매도
        df.loc[(df['cur_prc'] < df['ma']) & (df['foreign_sum'] < 0), 'signal'] = -1

        return df


class BreakoutForeign(AdvancedStrategy):
    """돌파 + 외국인 확인
    - 신고가 돌파 + 외국인 순매수 = 강한 매수
    """
    def generate_signals(self, df):
        lookback = self.params.get('lookback', 20)
        foreign_period = self.params.get('foreign_period', 5)

        df = df.copy()
        df['highest'] = df['high_pric'].rolling(lookback).max()
        df['lowest'] = df['low_pric'].rolling(lookback).min()
        df['foreign_sum'] = df['chg_qty'].rolling(foreign_period).sum()

        df['signal'] = 0
        # 신고가 돌파 + 외국인 순매수 = 매수
        df.loc[(df['cur_prc'] >= df['highest']) & (df['foreign_sum'] > 0), 'signal'] = 1
        # 신저가 돌파 + 외국인 순매도 = 매도
        df.loc[(df['cur_prc'] <= df['lowest']) & (df['foreign_sum'] < 0), 'signal'] = -1

        return df


class RSIForeign(AdvancedStrategy):
    """RSI + 외국인 다이버전스
    - RSI 과매도 + 외국인 순매수 = 반등 기대
    """
    def generate_signals(self, df):
        rsi_period = self.params.get('rsi_period', 14)
        foreign_period = self.params.get('foreign_period', 5)
        oversold = self.params.get('oversold', 30)
        overbought = self.params.get('overbought', 70)

        df = df.copy()

        # RSI 계산
        delta = df['cur_prc'].diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        df['foreign_sum'] = df['chg_qty'].rolling(foreign_period).sum()

        df['signal'] = 0
        # RSI 과매도 + 외국인 순매수 = 매수
        df.loc[(df['rsi'] < oversold) & (df['foreign_sum'] > 0), 'signal'] = 1
        # RSI 과매수 + 외국인 순매도 = 매도
        df.loc[(df['rsi'] > overbought) & (df['foreign_sum'] < 0), 'signal'] = -1

        return df


class MACDForeign(AdvancedStrategy):
    """MACD + 외국인
    - MACD 골든크로스 + 외국인 순매수 = 강한 매수
    """
    def generate_signals(self, df):
        fast = self.params.get('fast', 12)
        slow = self.params.get('slow', 26)
        signal_period = self.params.get('signal', 9)
        foreign_period = self.params.get('foreign_period', 5)

        df = df.copy()

        # MACD 계산
        df['ema_fast'] = df['cur_prc'].ewm(span=fast).mean()
        df['ema_slow'] = df['cur_prc'].ewm(span=slow).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=signal_period).mean()

        df['foreign_sum'] = df['chg_qty'].rolling(foreign_period).sum()

        df['signal'] = 0
        # MACD > Signal + 외국인 순매수 = 매수
        df.loc[(df['macd'] > df['macd_signal']) & (df['foreign_sum'] > 0), 'signal'] = 1
        # MACD < Signal + 외국인 순매도 = 매도
        df.loc[(df['macd'] < df['macd_signal']) & (df['foreign_sum'] < 0), 'signal'] = -1

        return df


class BollingerForeign(AdvancedStrategy):
    """볼린저밴드 + 외국인
    - 하단밴드 + 외국인 순매수 = 반등 기대
    """
    def generate_signals(self, df):
        period = self.params.get('period', 20)
        std_mult = self.params.get('std_mult', 2)
        foreign_period = self.params.get('foreign_period', 5)

        df = df.copy()

        df['ma'] = df['cur_prc'].rolling(period).mean()
        df['std'] = df['cur_prc'].rolling(period).std()
        df['upper'] = df['ma'] + std_mult * df['std']
        df['lower'] = df['ma'] - std_mult * df['std']

        df['foreign_sum'] = df['chg_qty'].rolling(foreign_period).sum()

        df['signal'] = 0
        # 가격 < 하단밴드 + 외국인 순매수 = 매수
        df.loc[(df['cur_prc'] < df['lower']) & (df['foreign_sum'] > 0), 'signal'] = 1
        # 가격 > 상단밴드 + 외국인 순매도 = 매도
        df.loc[(df['cur_prc'] > df['upper']) & (df['foreign_sum'] < 0), 'signal'] = -1

        return df


class VolumeBreakoutForeign(AdvancedStrategy):
    """거래량 돌파 + 외국인
    - 거래량 급증 + 외국인 순매수 = 세력 유입
    """
    def generate_signals(self, df):
        vol_period = self.params.get('vol_period', 20)
        vol_mult = self.params.get('vol_mult', 2)
        foreign_period = self.params.get('foreign_period', 5)

        df = df.copy()

        df['vol_ma'] = df['trde_qty'].rolling(vol_period).mean()
        df['foreign_sum'] = df['chg_qty'].rolling(foreign_period).sum()

        df['signal'] = 0
        # 거래량 > 평균 * vol_mult + 외국인 순매수 = 매수
        df.loc[(df['trde_qty'] > df['vol_ma'] * vol_mult) & (df['foreign_sum'] > 0), 'signal'] = 1
        # 거래량 급증 + 외국인 순매도 = 매도
        df.loc[(df['trde_qty'] > df['vol_ma'] * vol_mult) & (df['foreign_sum'] < 0), 'signal'] = -1

        return df


class MomentumForeign(AdvancedStrategy):
    """모멘텀 + 외국인
    - 가격 모멘텀 양수 + 외국인 순매수 = 추세 확인
    """
    def generate_signals(self, df):
        mom_period = self.params.get('mom_period', 20)
        foreign_period = self.params.get('foreign_period', 5)

        df = df.copy()

        df['momentum'] = df['cur_prc'].pct_change(mom_period)
        df['foreign_sum'] = df['chg_qty'].rolling(foreign_period).sum()

        df['signal'] = 0
        # 모멘텀 양수 + 외국인 순매수 = 매수
        df.loc[(df['momentum'] > 0) & (df['foreign_sum'] > 0), 'signal'] = 1
        # 모멘텀 음수 + 외국인 순매도 = 매도
        df.loc[(df['momentum'] < 0) & (df['foreign_sum'] < 0), 'signal'] = -1

        return df


class DualMAForeign(AdvancedStrategy):
    """이중 이동평균 + 외국인
    - 단기MA > 장기MA + 외국인 순매수 = 추세 매수
    """
    def generate_signals(self, df):
        short_ma = self.params.get('short_ma', 10)
        long_ma = self.params.get('long_ma', 50)
        foreign_period = self.params.get('foreign_period', 5)

        df = df.copy()

        df['ma_short'] = df['cur_prc'].rolling(short_ma).mean()
        df['ma_long'] = df['cur_prc'].rolling(long_ma).mean()
        df['foreign_sum'] = df['chg_qty'].rolling(foreign_period).sum()

        df['signal'] = 0
        # 단기MA > 장기MA + 외국인 순매수 = 매수
        df.loc[(df['ma_short'] > df['ma_long']) & (df['foreign_sum'] > 0), 'signal'] = 1
        # 단기MA < 장기MA + 외국인 순매도 = 매도
        df.loc[(df['ma_short'] < df['ma_long']) & (df['foreign_sum'] < 0), 'signal'] = -1

        return df


class ForeignRatioAcceleration(AdvancedStrategy):
    """외국인 지분율 가속화
    - 지분율 증가 가속화 시 매수
    """
    def generate_signals(self, df):
        period = self.params.get('period', 5)

        df = df.copy()

        df['wght_change'] = df['wght'].diff()
        df['wght_accel'] = df['wght_change'].diff()

        df['signal'] = 0
        # 지분율 변화 가속화 (증가) = 매수
        df.loc[(df['wght_change'] > 0) & (df['wght_accel'] > 0), 'signal'] = 1
        # 지분율 변화 감속화 (감소) = 매도
        df.loc[(df['wght_change'] < 0) & (df['wght_accel'] < 0), 'signal'] = -1

        return df


def prepare_data(filepath):
    """데이터 준비"""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df['dt'] = pd.to_datetime(df['dt'])
    df = df[df['chg_qty'].notna()].copy()

    # 가격 데이터 정리
    for col in ['cur_prc', 'open_pric', 'high_pric', 'low_pric']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('+', '').str.replace('-', '').astype(float)

    df['chg_qty'] = pd.to_numeric(df['chg_qty'], errors='coerce')
    df['wght'] = df['wght'].astype(str).str.replace('+', '').astype(float)

    return df


def backtest_strategy(df, strategy, initial_capital=100_000_000, fee_rate=0.001):
    """전략 백테스트 수행"""
    try:
        df = strategy.generate_signals(df)
    except Exception as e:
        return None

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

        if signal == 1 and position == 0:
            shares = int(capital * 0.95 / price)
            cost = shares * price * (1 + fee_rate)
            if cost <= capital and shares > 0:
                capital -= cost
                position = 1
                entry_price = price
                trades.append({
                    'date': date, 'type': 'BUY', 'price': price,
                    'shares': shares, 'capital': capital
                })

        elif signal == -1 and position == 1:
            proceeds = shares * price * (1 - fee_rate)
            capital += proceeds
            pnl = (price - entry_price) / entry_price
            trades.append({
                'date': date, 'type': 'SELL', 'price': price,
                'shares': shares, 'capital': capital, 'pnl': pnl
            })
            position = 0
            shares = 0

    if position == 1:
        final_price = float(df.iloc[-1]['cur_prc'])
        capital += shares * final_price * (1 - fee_rate)

    total_return = (capital - initial_capital) / initial_capital
    num_trades = len([t for t in trades if t['type'] == 'SELL'])

    if num_trades == 0:
        return None

    win_trades = [t for t in trades if t['type'] == 'SELL' and t.get('pnl', 0) > 0]
    win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
    avg_pnl = np.mean([t.get('pnl', 0) for t in trades if t['type'] == 'SELL'])
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
    """모든 고급 전략 백테스트"""
    print("=" * 70)
    print("키움 데이터 기반 고급 복합 전략 백테스트")
    print("=" * 70)

    # 전략 목록
    strategies = [
        # 추세 필터 + 수급
        TrendFilterForeign("TrendFilter_MA50_F5", {'ma_period': 50, 'foreign_period': 5}),
        TrendFilterForeign("TrendFilter_MA20_F5", {'ma_period': 20, 'foreign_period': 5}),
        TrendFilterForeign("TrendFilter_MA100_F10", {'ma_period': 100, 'foreign_period': 10}),

        # 돌파 + 수급
        BreakoutForeign("Breakout_20_F5", {'lookback': 20, 'foreign_period': 5}),
        BreakoutForeign("Breakout_50_F10", {'lookback': 50, 'foreign_period': 10}),

        # RSI + 수급
        RSIForeign("RSI_30_70_F5", {'rsi_period': 14, 'oversold': 30, 'overbought': 70, 'foreign_period': 5}),
        RSIForeign("RSI_25_75_F5", {'rsi_period': 14, 'oversold': 25, 'overbought': 75, 'foreign_period': 5}),

        # MACD + 수급
        MACDForeign("MACD_12_26_9_F5", {'fast': 12, 'slow': 26, 'signal': 9, 'foreign_period': 5}),
        MACDForeign("MACD_12_26_9_F10", {'fast': 12, 'slow': 26, 'signal': 9, 'foreign_period': 10}),

        # 볼린저 + 수급
        BollingerForeign("Bollinger_20_2_F5", {'period': 20, 'std_mult': 2, 'foreign_period': 5}),
        BollingerForeign("Bollinger_20_2.5_F5", {'period': 20, 'std_mult': 2.5, 'foreign_period': 5}),

        # 거래량 돌파 + 수급
        VolumeBreakoutForeign("VolBreakout_2x_F5", {'vol_period': 20, 'vol_mult': 2, 'foreign_period': 5}),
        VolumeBreakoutForeign("VolBreakout_3x_F5", {'vol_period': 20, 'vol_mult': 3, 'foreign_period': 5}),

        # 모멘텀 + 수급
        MomentumForeign("Momentum_20_F5", {'mom_period': 20, 'foreign_period': 5}),
        MomentumForeign("Momentum_10_F5", {'mom_period': 10, 'foreign_period': 5}),

        # 이중 MA + 수급
        DualMAForeign("DualMA_10_50_F5", {'short_ma': 10, 'long_ma': 50, 'foreign_period': 5}),
        DualMAForeign("DualMA_20_100_F10", {'short_ma': 20, 'long_ma': 100, 'foreign_period': 10}),

        # 외국인 지분율 가속화
        ForeignRatioAcceleration("ForeignRatioAccel", {'period': 5}),
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

        df = prepare_data(filepath)

        if len(df) < 100:
            continue

        name = df['name'].iloc[0] if 'name' in df.columns else ticker
        print(f"\n{name} ({ticker}) - {len(df)}일 데이터")

        best_result = None
        best_return = -999

        for strategy in strategies:
            result = backtest_strategy(df, strategy)
            if result:
                result['ticker'] = ticker
                result['name'] = name
                all_results.append(result)

                if result['total_return'] > best_return:
                    best_return = result['total_return']
                    best_result = result

        if best_result and best_return > 0:
            print(f"  최고: {best_result['strategy']} = {best_result['total_return']*100:.1f}% "
                  f"(승률: {best_result['win_rate']*100:.0f}%)")

    # 결과 저장
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('total_return', ascending=False)

        results_df.to_csv(f"{RESULT_PATH}/advanced_strategy_results.csv",
                         index=False, encoding='utf-8-sig')

        print("\n" + "=" * 70)
        print("고급 전략 백테스트 결과 요약")
        print("=" * 70)

        print(f"\n총 백테스트 수: {len(results_df)}")
        print(f"수익 전략: {(results_df['total_return'] > 0).sum()}개")
        print(f"손실 전략: {(results_df['total_return'] < 0).sum()}개")

        print("\n[전략별 평균 성과 (상위 10)]")
        strategy_avg = results_df.groupby('strategy').agg({
            'total_return': 'mean',
            'annual_return': 'mean',
            'win_rate': 'mean'
        }).sort_values('total_return', ascending=False)

        for i, (strategy, row) in enumerate(strategy_avg.head(10).iterrows()):
            print(f"  {i+1}. {strategy}: 평균수익 {row['total_return']*100:.1f}%, "
                  f"연환산 {row['annual_return']*100:.1f}%, 승률 {row['win_rate']*100:.0f}%")

        print("\n[전체 최고 성과 Top 15]")
        top15 = results_df.head(15)
        for i, row in top15.iterrows():
            print(f"  {row['name']} / {row['strategy']}: "
                  f"{row['total_return']*100:.1f}% (승률: {row['win_rate']*100:.0f}%)")

        print(f"\n결과 저장: {RESULT_PATH}/advanced_strategy_results.csv")

    return results_df if all_results else None


if __name__ == "__main__":
    results = run_all_backtests()
