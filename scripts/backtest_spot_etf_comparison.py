"""
1x ETF vs 2x ETF 비교 백테스트
==============================
동일한 VIX 전략을 1배/2배 ETF에 적용하여 비교.

결론 도출: 어느 ETF가 실전에 적합한가?
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'E:/투자/data/kosdaq_futures/multi_asset'
OUTPUT_DIR = 'E:/투자/Multi-Asset Strategy Platform/logs'

print("=" * 80)
print("1x ETF vs 2x ETF 비교 백테스트")
print("=" * 80)

# 데이터 로드
kospi200 = pd.read_parquet(f'{DATA_DIR}/kospi200.parquet')
vix = pd.read_parquet(f'{DATA_DIR}/vix.parquet')
semicon = pd.read_parquet(f'{DATA_DIR}/semicon.parquet')

data = kospi200[['Close']].copy()
data.columns = ['close']
data['returns'] = data['close'].pct_change()

# T-1 shift
data['vix'] = vix['Close'].shift(1)
data['vix_prev'] = vix['Close'].shift(2)
data['semicon'] = semicon['Close'].shift(1)

for col in ['vix', 'vix_prev', 'semicon']:
    data[col] = data[col].ffill()

data['vix_sma20'] = data['vix'].rolling(20).mean()
data['semicon_sma20'] = data['semicon'].rolling(20).mean()

data = data.dropna()
print(f"데이터: {data.index.min()} ~ {data.index.max()} ({len(data)}일)")


def backtest(data, signal, leverage=1.0, cost=0.001):
    """백테스트."""
    leveraged_returns = data['returns'] * leverage
    position_change = signal.diff().abs()
    costs = position_change * cost
    strategy_returns = signal.shift(1) * leveraged_returns - costs
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

    negative_returns = strategy_returns[strategy_returns < 0]
    downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.001
    sortino = cagr / downside_vol

    calmar = cagr / abs(mdd) if mdd != 0 else 0

    time_in_market = (signal == 1).mean()
    trades = (signal.diff().abs() > 0).sum()

    return {
        'sharpe': sharpe,
        'cagr': cagr,
        'mdd': mdd,
        'sortino': sortino,
        'calmar': calmar,
        'volatility': volatility,
        'time_in_market': time_in_market,
        'trades': int(trades),
    }


def walk_forward(data, signal_func, n_splits=5):
    """Walk-Forward 검증."""
    total_len = len(data)
    split_size = total_len // n_splits
    oos_results = []

    for i in range(n_splits - 1):
        oos_start = (i + 1) * split_size
        oos_end = min((i + 2) * split_size, total_len)
        oos_data = data.iloc[oos_start:oos_end]
        oos_signal = signal_func(oos_data)
        result = backtest(oos_data, oos_signal, leverage=1.0)
        if result:
            oos_results.append(result['sharpe'])

    return np.mean(oos_results) if oos_results else None


# 전략
def strategy_vix_below_sma20(data):
    signal = pd.Series(0, index=data.index)
    signal[data['vix'] < data['vix_sma20']] = 1
    return signal


def strategy_vix_declining(data):
    signal = pd.Series(0, index=data.index)
    signal[data['vix'] < data['vix_prev']] = 1
    return signal


def strategy_composite(data):
    vix_below = (data['vix'] < data['vix_sma20']).astype(int)
    vix_declining = (data['vix'] < data['vix_prev']).astype(int)
    semicon_above = (data['semicon'] > data['semicon_sma20']).astype(int)
    composite = vix_below * 0.5 + vix_declining * 0.3 + semicon_above * 0.2
    signal = pd.Series(0, index=data.index)
    signal[composite >= 0.5] = 1
    return signal


def strategy_buy_hold(data):
    return pd.Series(1, index=data.index)


strategies = {
    'VIX_Below_SMA20': strategy_vix_below_sma20,
    'VIX_Declining': strategy_vix_declining,
    'Composite_50_30_20': strategy_composite,
    'Buy_Hold': strategy_buy_hold,
}

# 테스트
leverage_options = [
    {'leverage': 1.0, 'cost': 0.0005, 'name': '1x_ETF'},
    {'leverage': 2.0, 'cost': 0.0013, 'name': '2x_ETF'},
]

print("\n" + "=" * 80)
print("비교 결과")
print("=" * 80)

comparison = []

for strat_name, strat_func in strategies.items():
    signal = strat_func(data)

    for lev in leverage_options:
        result = backtest(data, signal, leverage=lev['leverage'], cost=lev['cost'])

        if result is None:
            continue

        # WF 검증 (1x만)
        wf_sharpe = walk_forward(data, strat_func) if lev['leverage'] == 1.0 else None

        entry = {
            'strategy': strat_name,
            'etf_type': lev['name'],
            'leverage': lev['leverage'],
            'sharpe': result['sharpe'],
            'cagr': result['cagr'],
            'mdd': result['mdd'],
            'sortino': result['sortino'],
            'calmar': result['calmar'],
            'time_in_market': result['time_in_market'],
            'wf_oos_sharpe': wf_sharpe,
        }
        comparison.append(entry)

        print(f"\n{strat_name} ({lev['name']}):")
        print(f"  Sharpe: {result['sharpe']:.3f}, CAGR: {result['cagr']*100:.1f}%, MDD: {result['mdd']*100:.1f}%")
        print(f"  Sortino: {result['sortino']:.3f}, Calmar: {result['calmar']:.3f}")
        if wf_sharpe:
            print(f"  WF OOS Sharpe: {wf_sharpe:.3f}")

# 결과 정리
results_df = pd.DataFrame(comparison)

# 1x vs 2x 비교 테이블
print("\n" + "=" * 80)
print("1x ETF vs 2x ETF 상세 비교")
print("=" * 80)

for strat_name in strategies.keys():
    strat_data = results_df[results_df['strategy'] == strat_name]
    if len(strat_data) < 2:
        continue

    etf_1x = strat_data[strat_data['etf_type'] == '1x_ETF'].iloc[0]
    etf_2x = strat_data[strat_data['etf_type'] == '2x_ETF'].iloc[0]

    print(f"\n{strat_name}:")
    print(f"  {'지표':<15} {'1x ETF':<15} {'2x ETF':<15} {'비고'}")
    print(f"  {'-'*50}")
    print(f"  {'Sharpe':<15} {etf_1x['sharpe']:.3f}{'':<10} {etf_2x['sharpe']:.3f}{'':<10} {'1x 우위' if etf_1x['sharpe'] > etf_2x['sharpe'] else '2x 우위'}")
    print(f"  {'CAGR':<15} {etf_1x['cagr']*100:.1f}%{'':<9} {etf_2x['cagr']*100:.1f}%{'':<9} {'2x가 2배?' if etf_2x['cagr'] > etf_1x['cagr']*1.8 else '2배 미달'}")
    print(f"  {'MDD':<15} {etf_1x['mdd']*100:.1f}%{'':<9} {etf_2x['mdd']*100:.1f}%{'':<9} {'2x MDD 심각' if etf_2x['mdd'] < -0.30 else 'OK'}")
    print(f"  {'Calmar':<15} {etf_1x['calmar']:.3f}{'':<10} {etf_2x['calmar']:.3f}{'':<10} {'1x 우위' if etf_1x['calmar'] > etf_2x['calmar'] else '2x 우위'}")

# 유효 전략 필터
print("\n" + "=" * 80)
print("유효 전략 판정")
print("=" * 80)

# 1x ETF 기준
viable_1x = results_df[
    (results_df['etf_type'] == '1x_ETF') &
    (results_df['sharpe'] > 1.5) &
    (results_df['mdd'] > -0.15) &
    (results_df['strategy'] != 'Buy_Hold')
]

# 2x ETF 기준 (완화)
viable_2x = results_df[
    (results_df['etf_type'] == '2x_ETF') &
    (results_df['sharpe'] > 0.5) &
    (results_df['mdd'] > -0.25) &
    (results_df['strategy'] != 'Buy_Hold')
]

print(f"\n1x ETF 유효 전략 (Sharpe>1.5, MDD>-15%): {len(viable_1x)}")
if len(viable_1x) > 0:
    for _, row in viable_1x.iterrows():
        print(f"  - {row['strategy']}: Sharpe {row['sharpe']:.3f}, MDD {row['mdd']*100:.1f}%")

print(f"\n2x ETF 유효 전략 (Sharpe>0.5, MDD>-25%): {len(viable_2x)}")
if len(viable_2x) > 0:
    for _, row in viable_2x.iterrows():
        print(f"  - {row['strategy']}: Sharpe {row['sharpe']:.3f}, MDD {row['mdd']*100:.1f}%")

# 최종 결론
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

if len(viable_1x) > 0 and len(viable_2x) == 0:
    print("\n결론: 1x ETF(TIGER 200) 사용 권장")
    print("\n이유:")
    print("  1. 1x ETF에서 유효한 전략 존재 (Sharpe > 1.5)")
    print("  2. 2x ETF는 어떤 전략도 MDD -25% 미만 달성 불가")
    print("  3. 2x ETF의 변동성 손실로 장기 수익 감소")

    best_1x = viable_1x.iloc[0]
    print(f"\n추천 전략: {best_1x['strategy']} (1x ETF)")
    print(f"  Sharpe: {best_1x['sharpe']:.3f}")
    print(f"  CAGR: {best_1x['cagr']*100:.1f}%")
    print(f"  MDD: {best_1x['mdd']*100:.1f}%")

    recommendation = {
        'decision': '1x_ETF',
        'strategy': best_1x['strategy'],
        'sharpe': best_1x['sharpe'],
        'cagr': best_1x['cagr'],
        'mdd': best_1x['mdd'],
        'reason': '2x ETF는 MDD -25% 기준 충족 불가'
    }

elif len(viable_2x) > 0:
    print("\n결론: 2x ETF 조건부 사용 가능")
    best_2x = viable_2x.iloc[0]
    recommendation = {
        'decision': '2x_ETF_conditional',
        'strategy': best_2x['strategy'],
        'sharpe': best_2x['sharpe'],
        'cagr': best_2x['cagr'],
        'mdd': best_2x['mdd'],
    }
else:
    print("\n결론: ETF 투자 재검토 필요")
    print("  - 1x, 2x 모두 유효 전략 없음")
    recommendation = {'decision': 'reconsider', 'reason': 'No viable strategy'}

# 저장
output = {
    'generated': datetime.now().isoformat(),
    'type': 'etf_comparison',
    'data_period': f"{data.index.min()} ~ {data.index.max()}",
    'comparison': comparison,
    'viable_1x': viable_1x.to_dict('records') if len(viable_1x) > 0 else [],
    'viable_2x': viable_2x.to_dict('records') if len(viable_2x) > 0 else [],
    'recommendation': recommendation,
}

output_path = f'{OUTPUT_DIR}/etf_comparison_results.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)

print(f"\n결과 저장: {output_path}")
