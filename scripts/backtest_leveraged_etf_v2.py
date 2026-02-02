"""
레버리지 ETF 전략 v2: 보수적 접근
===================================
1차 테스트 결과: 기존 VIX 전략은 MDD가 너무 높음 (40-80%)

v2 접근:
1. 더 엄격한 진입 조건 (다중 필터)
2. 조기 청산 규칙 (손절)
3. VIX 급등 시 즉시 청산
4. 현금 비중 확대
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'E:/투자/data/kosdaq_futures/multi_asset'
OUTPUT_DIR = 'E:/투자/Multi-Asset Strategy Platform/logs'

LEVERAGE = 2.0
TOTAL_COST = 0.0013  # 0.13%

print("=" * 80)
print("레버리지 ETF 전략 v2: 보수적 접근")
print("=" * 80)

# 데이터 로드
kospi200 = pd.read_parquet(f'{DATA_DIR}/kospi200.parquet')
vix = pd.read_parquet(f'{DATA_DIR}/vix.parquet')
semicon = pd.read_parquet(f'{DATA_DIR}/semicon.parquet')
sp500 = pd.read_parquet(f'{DATA_DIR}/sp500.parquet')

data = kospi200[['Close']].copy()
data.columns = ['close']
data['returns'] = data['close'].pct_change()
data['leveraged_returns'] = data['returns'] * LEVERAGE

# T-1 shift 적용
data['vix'] = vix['Close'].shift(1)
data['vix_prev'] = vix['Close'].shift(2)
data['vix_prev2'] = vix['Close'].shift(3)
data['sp500'] = sp500['Close'].shift(1)
data['semicon'] = semicon['Close'].shift(1)

for col in ['vix', 'vix_prev', 'vix_prev2', 'sp500', 'semicon']:
    data[col] = data[col].ffill()

# 지표
data['vix_sma20'] = data['vix'].rolling(20).mean()
data['vix_sma10'] = data['vix'].rolling(10).mean()
data['vix_sma5'] = data['vix'].rolling(5).mean()
data['vix_change'] = data['vix'].pct_change()
data['vix_change_5d'] = data['vix'].pct_change(5)

data['close_sma20'] = data['close'].rolling(20).mean()
data['close_sma50'] = data['close'].rolling(50).mean()
data['close_sma200'] = data['close'].rolling(200).mean()

data['sp500_sma50'] = data['sp500'].rolling(50).mean()
data['semicon_sma20'] = data['semicon'].rolling(20).mean()

# 변동성 지표
data['volatility_20d'] = data['returns'].rolling(20).std() * np.sqrt(252)

# 누적 수익 (최근 20일)
data['cum_ret_20d'] = data['close'].pct_change(20)

data = data.dropna()
print(f"데이터 기간: {data.index.min()} ~ {data.index.max()}")


def backtest_with_stoploss(data, signal, leverage=LEVERAGE, cost=TOTAL_COST,
                           stop_loss=-0.05, trailing_stop=None):
    """
    스톱로스 포함 백테스트.
    """
    base_returns = data['returns']
    leveraged_returns = base_returns * leverage

    position = pd.Series(0.0, index=data.index)
    entry_price = None
    peak_price = None

    for i in range(1, len(data)):
        idx = data.index[i]
        prev_idx = data.index[i-1]

        # 이전 포지션
        prev_pos = position.iloc[i-1]
        current_signal = signal.iloc[i-1]  # 시그널은 전일 기준

        current_price = data['close'].iloc[i]

        if prev_pos == 0 and current_signal == 1:
            # 진입
            position.iloc[i] = 1
            entry_price = current_price
            peak_price = current_price
        elif prev_pos == 1:
            # 보유 중
            if entry_price is not None:
                pnl = (current_price / entry_price - 1) * leverage

                # 스톱로스 체크
                if stop_loss is not None and pnl <= stop_loss:
                    position.iloc[i] = 0
                    entry_price = None
                    peak_price = None
                    continue

                # 트레일링 스톱
                if trailing_stop and peak_price is not None:
                    peak_price = max(peak_price, current_price)
                    from_peak = (current_price / peak_price - 1) * leverage
                    if from_peak <= trailing_stop:
                        position.iloc[i] = 0
                        entry_price = None
                        peak_price = None
                        continue

            # 시그널 청산 체크
            if current_signal == 0:
                position.iloc[i] = 0
                entry_price = None
                peak_price = None
            else:
                position.iloc[i] = 1
                if peak_price is not None:
                    peak_price = max(peak_price, current_price)

    # 수익 계산
    position_change = position.diff().abs()
    costs = position_change * cost
    strategy_returns = position.shift(1) * leveraged_returns - costs
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) < 252:
        return None

    cumulative = (1 + strategy_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    years = len(strategy_returns) / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1

    volatility = strategy_returns.std() * np.sqrt(252)
    sharpe = cagr / volatility if volatility > 0 else 0

    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    mdd = drawdown.min()

    time_in_market = (position == 1).mean()
    trades = (position.diff().abs() > 0).sum()

    return {
        'sharpe': sharpe,
        'cagr': cagr,
        'mdd': mdd,
        'volatility': volatility,
        'time_in_market': time_in_market,
        'trades': int(trades),
        'cumulative': cumulative,
    }


# ===== 보수적 전략 =====

def strategy_ultra_conservative(data):
    """
    초보수적: VIX < 16 AND VIX 하락 AND KOSPI 상승추세
    """
    signal = pd.Series(0, index=data.index)
    cond = (
        (data['vix'] < 16) &
        (data['vix'] < data['vix_prev']) &
        (data['close'] > data['close_sma50'])
    )
    signal[cond] = 1
    return signal


def strategy_vix_extreme_low(data):
    """
    VIX 극저점: VIX < 15 (매우 낮은 공포)
    """
    signal = pd.Series(0, index=data.index)
    signal[data['vix'] < 15] = 1
    return signal


def strategy_vix_below_sma_and_declining_3d(data):
    """
    VIX < SMA20 AND 3일 연속 하락
    """
    signal = pd.Series(0, index=data.index)
    declining_3d = (
        (data['vix'] < data['vix_prev']) &
        (data['vix_prev'] < data['vix_prev2'])
    )
    cond = (data['vix'] < data['vix_sma20']) & declining_3d
    signal[cond] = 1
    return signal


def strategy_multi_filter_strict(data):
    """
    다중 필터 (엄격):
    - VIX < SMA20
    - VIX < 20 (절대값)
    - KOSPI200 > SMA50
    - S&P500 > SMA50
    """
    signal = pd.Series(0, index=data.index)
    cond = (
        (data['vix'] < data['vix_sma20']) &
        (data['vix'] < 20) &
        (data['close'] > data['close_sma50']) &
        (data['sp500'] > data['sp500_sma50'])
    )
    signal[cond] = 1
    return signal


def strategy_vix_regime_filter(data):
    """
    VIX 레짐 필터:
    - 저변동성 레짐: VIX < 18 AND VIX < SMA20
    - 고변동성 레짐: 현금
    """
    signal = pd.Series(0, index=data.index)
    low_vol_regime = (data['vix'] < 18) & (data['vix'] < data['vix_sma20'])
    signal[low_vol_regime] = 1
    return signal


def strategy_crisis_avoidance(data):
    """
    위기 회피: VIX 급등 시 즉시 청산
    - 진입: VIX < SMA20
    - 청산: VIX > 25 OR VIX 5일 변화율 > 20%
    """
    signal = pd.Series(0, index=data.index)

    in_position = False
    for i in range(len(data)):
        vix_val = data['vix'].iloc[i]
        vix_sma = data['vix_sma20'].iloc[i]
        vix_change_5d = data['vix_change_5d'].iloc[i]

        # 위기 신호 (즉시 청산)
        crisis = (vix_val > 25) or (vix_change_5d > 0.20)

        if crisis:
            in_position = False
        elif not in_position and vix_val < vix_sma:
            in_position = True
        elif in_position and vix_val >= vix_sma:
            in_position = False

        signal.iloc[i] = 1 if in_position else 0

    return signal


def strategy_low_volatility_only(data):
    """
    저변동성 시기만 투자:
    - 20일 변동성 < 15% (연율화)
    - VIX < SMA20
    """
    signal = pd.Series(0, index=data.index)
    cond = (data['volatility_20d'] < 0.15) & (data['vix'] < data['vix_sma20'])
    signal[cond] = 1
    return signal


def strategy_momentum_filter(data):
    """
    모멘텀 필터:
    - 최근 20일 수익 > 0%
    - VIX < SMA20
    """
    signal = pd.Series(0, index=data.index)
    cond = (data['cum_ret_20d'] > 0) & (data['vix'] < data['vix_sma20'])
    signal[cond] = 1
    return signal


def strategy_simple_vix_below_sma(data):
    """기존 VIX Below SMA20 (비교용)."""
    signal = pd.Series(0, index=data.index)
    signal[data['vix'] < data['vix_sma20']] = 1
    return signal


def strategy_buy_hold(data):
    """Buy & Hold (비교용)."""
    return pd.Series(1, index=data.index)


# 전략 목록
strategies_v2 = {
    # 기존 (비교용)
    'VIX_Below_SMA20': strategy_simple_vix_below_sma,
    'Buy_Hold': strategy_buy_hold,

    # 보수적 전략
    'Ultra_Conservative': strategy_ultra_conservative,
    'VIX_Extreme_Low': strategy_vix_extreme_low,
    'VIX_Below_SMA_Declining_3d': strategy_vix_below_sma_and_declining_3d,
    'Multi_Filter_Strict': strategy_multi_filter_strict,
    'VIX_Regime_Filter': strategy_vix_regime_filter,
    'Crisis_Avoidance': strategy_crisis_avoidance,
    'Low_Volatility_Only': strategy_low_volatility_only,
    'Momentum_Filter': strategy_momentum_filter,
}

# 스톱로스 옵션
stoploss_options = [
    {'stop_loss': None, 'trailing_stop': None, 'name': 'No_SL'},
    {'stop_loss': -0.05, 'trailing_stop': None, 'name': 'SL_5%'},
    {'stop_loss': -0.10, 'trailing_stop': None, 'name': 'SL_10%'},
    {'stop_loss': -0.05, 'trailing_stop': -0.08, 'name': 'SL_5%_TS_8%'},
]

print("\n" + "=" * 80)
print("보수적 전략 + 스톱로스 테스트")
print("=" * 80)

all_results = []

for strat_name, strat_func in strategies_v2.items():
    signal = strat_func(data)

    for sl_opt in stoploss_options:
        result = backtest_with_stoploss(
            data, signal,
            stop_loss=sl_opt['stop_loss'],
            trailing_stop=sl_opt['trailing_stop']
        )

        if result is None:
            continue

        entry = {
            'strategy': f"{strat_name}_{sl_opt['name']}",
            'base_strategy': strat_name,
            'stoploss': sl_opt['name'],
            'sharpe': result['sharpe'],
            'cagr': result['cagr'],
            'mdd': result['mdd'],
            'time_in_market': result['time_in_market'],
            'trades': result['trades'],
        }
        all_results.append(entry)

# 결과 정리
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('sharpe', ascending=False)

# 유효 전략 필터 (완화된 기준)
# Sharpe > 0.5, MDD > -30%
viable = results_df[
    (results_df['sharpe'] > 0.5) &
    (results_df['mdd'] > -0.30) &
    (~results_df['base_strategy'].isin(['Buy_Hold']))
]

print(f"\n총 테스트: {len(results_df)}")
print(f"유효 전략 (Sharpe>0.5, MDD>-30%): {len(viable)}")

if len(viable) > 0:
    print("\n" + "=" * 80)
    print("유효 전략")
    print("=" * 80)
    for _, row in viable.iterrows():
        print(f"\n{row['strategy']}:")
        print(f"  Sharpe: {row['sharpe']:.3f}, CAGR: {row['cagr']*100:.1f}%, MDD: {row['mdd']*100:.1f}%")
        print(f"  Time in Market: {row['time_in_market']*100:.1f}%, Trades: {row['trades']}")
else:
    print("\n유효한 전략이 없습니다.")

# Buy & Hold 비교
print("\n" + "=" * 80)
print("Buy & Hold 비교")
print("=" * 80)

bh_results = results_df[results_df['base_strategy'] == 'Buy_Hold']
print(bh_results[['strategy', 'sharpe', 'cagr', 'mdd']].to_string())

# 전체 결과
print("\n" + "=" * 80)
print("전체 결과 (Sharpe 순)")
print("=" * 80)
print(results_df[['strategy', 'sharpe', 'cagr', 'mdd', 'time_in_market']].head(20).to_string())

# 결과 저장
output = {
    'generated': datetime.now().isoformat(),
    'type': 'leveraged_etf_backtest_v2',
    'approach': 'conservative_with_stoploss',
    'data_period': f"{data.index.min()} ~ {data.index.max()}",
    'viable_strategies': viable.to_dict('records') if len(viable) > 0 else [],
    'all_results': results_df.to_dict('records'),
}

output_path = f'{OUTPUT_DIR}/leveraged_etf_backtest_v2.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)

print(f"\n결과 저장: {output_path}")

# 최종 판정
print("\n" + "=" * 80)
print("v2 최종 판정")
print("=" * 80)

if len(viable) > 0:
    best = viable.iloc[0]
    print(f"\n최선 전략: {best['strategy']}")
    print(f"  Sharpe: {best['sharpe']:.3f}")
    print(f"  CAGR: {best['cagr']*100:.1f}%")
    print(f"  MDD: {best['mdd']*100:.1f}%")

    # 실전 적합성 판단
    if best['sharpe'] > 1.0 and best['mdd'] > -0.20:
        print("\n판정: A (실전 적합)")
    elif best['sharpe'] > 0.7 and best['mdd'] > -0.25:
        print("\n판정: B (조건부 적합)")
    else:
        print("\n판정: C (주의 필요)")
else:
    print("\n판정: F (적합한 전략 없음)")
    print("\n권장: 레버리지 ETF 대신 일반 ETF(TIGER 200) 사용")
