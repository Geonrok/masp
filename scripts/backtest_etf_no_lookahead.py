"""
Look-Ahead Bias 제거 후 ETF 백테스트
=====================================
원본 검증 스크립트에서 VIX에 shift(1) 미적용 확인.
이 스크립트는 올바른 shift(1) 적용 후 성과 비교.

핵심:
- 한국 T일 거래 시점에 미국 T일 데이터는 아직 없음
- 미국 데이터는 반드시 T-1 (shift(1)) 적용 필요
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'E:/투자/data/kosdaq_futures/multi_asset'
KOSPI_DIR = 'E:/투자/data/kospi_futures'
INVESTOR_DIR = 'E:/투자/data/kr_stock/investor_trading'
OUTPUT_DIR = 'E:/투자/Multi-Asset Strategy Platform/logs'

# 비용 설정 (원본과 동일)
REALISTIC_FEE = 0.00005
REALISTIC_SLIPPAGE = 0.0004
REALISTIC_COST = (REALISTIC_FEE + REALISTIC_SLIPPAGE) * 2

print("=" * 80)
print("Look-Ahead Bias 제거 후 ETF 백테스트")
print(f"거래 비용: {REALISTIC_COST*100:.3f}% (왕복)")
print("=" * 80)

# 데이터 로드
kospi200 = pd.read_parquet(f'{KOSPI_DIR}/kospi200_daily_yf.parquet')
kospi200.columns = [c.lower() for c in kospi200.columns]
if kospi200.index.tz is not None:
    kospi200.index = kospi200.index.tz_localize(None)

vix = pd.read_parquet(f'{DATA_DIR}/vix.parquet')
if vix.index.tz is not None:
    vix.index = vix.index.tz_localize(None)

semicon = pd.read_parquet(f'{DATA_DIR}/semicon.parquet')
if semicon.index.tz is not None:
    semicon.index = semicon.index.tz_localize(None)

# 외국인 데이터
import os
investor_files = [f for f in os.listdir(INVESTOR_DIR) if f.endswith('_investor.csv')]
all_investor = []
for f in investor_files:
    try:
        df = pd.read_csv(f'{INVESTOR_DIR}/{f}', encoding='utf-8-sig')
        df['날짜'] = pd.to_datetime(df['날짜'])
        df = df.set_index('날짜')
        all_investor.append(df[['외국인합계']])
    except:
        pass

foreign_data = all_investor[0].copy()
for df in all_investor[1:]:
    foreign_data = foreign_data.add(df, fill_value=0)
foreign_data = foreign_data.sort_index()

# 데이터 병합
data = kospi200[['close']].copy()
data['returns'] = data['close'].pct_change()

# ===== 두 가지 버전 비교 =====

# 버전 1: Look-Ahead Bias 있음 (원본 방식)
data_with_bias = data.copy()
data_with_bias['vix'] = vix['Close']  # shift 없음!
data_with_bias['semicon'] = semicon['Close']
data_with_bias['foreign'] = foreign_data['외국인합계']
for col in ['vix', 'semicon', 'foreign']:
    data_with_bias[col] = data_with_bias[col].ffill()

data_with_bias['vix_sma_20'] = data_with_bias['vix'].rolling(20).mean()
data_with_bias['semicon_sma_20'] = data_with_bias['semicon'].rolling(20).mean()
data_with_bias['foreign_20d'] = data_with_bias['foreign'].rolling(20).sum()
data_with_bias = data_with_bias.dropna()

# 버전 2: Look-Ahead Bias 제거 (올바른 방식)
data_no_bias = data.copy()
data_no_bias['vix'] = vix['Close'].shift(1)  # T-1 VIX
data_no_bias['semicon'] = semicon['Close'].shift(1)  # T-1 Semicon
data_no_bias['foreign'] = foreign_data['외국인합계']  # 한국 데이터는 shift 불필요
for col in ['vix', 'semicon', 'foreign']:
    data_no_bias[col] = data_no_bias[col].ffill()

data_no_bias['vix_sma_20'] = data_no_bias['vix'].rolling(20).mean()
data_no_bias['semicon_sma_20'] = data_no_bias['semicon'].rolling(20).mean()
data_no_bias['foreign_20d'] = data_no_bias['foreign'].rolling(20).sum()
data_no_bias = data_no_bias.dropna()

print(f"\nLook-Ahead Bias 버전: {len(data_with_bias)}일")
print(f"Bias 제거 버전: {len(data_no_bias)}일")


def detailed_backtest(data, signal, cost, annual_factor=252):
    """백테스트."""
    returns = data['returns']
    position_change = signal.diff().abs()
    costs = position_change * cost
    strategy_returns = signal.shift(1) * returns - costs
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) < 252:
        return None

    total_return = (1 + strategy_returns).prod() - 1
    if total_return <= -1:
        return None

    annual_return = (1 + total_return) ** (annual_factor / len(strategy_returns)) - 1
    volatility = strategy_returns.std() * np.sqrt(annual_factor)
    sharpe = annual_return / volatility if volatility > 0 else 0

    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    mdd = drawdown.min()

    return {
        'annual_return': annual_return,
        'sharpe': sharpe,
        'mdd': mdd,
        'volatility': volatility,
    }


def walk_forward(data, signal_func, cost, train_ratio=0.7):
    """Walk-Forward 검증."""
    n = len(data)
    train_end = int(n * train_ratio)
    test_data = data.iloc[train_end:].copy()

    if len(test_data) < 50:
        return None

    test_signal = signal_func(test_data)
    test_result = detailed_backtest(test_data, test_signal, cost)

    return test_result


# 전략 정의 (원본과 동일)
def vix_below_sma20(d):
    return (d['vix'] < d['vix_sma_20']).astype(int)


def semicon_foreign(d):
    return ((d['semicon'] > d['semicon_sma_20']) & (d['foreign_20d'] > 0)).astype(int)


strategies = {
    'VIX_Below_SMA20': vix_below_sma20,
    'Semicon_Foreign': semicon_foreign,
}

# 비교 테스트
print("\n" + "=" * 80)
print("Look-Ahead Bias 비교")
print("=" * 80)

comparison_results = []

for strat_name, strat_func in strategies.items():
    print(f"\n{strat_name}:")

    # With Bias
    signal_bias = strat_func(data_with_bias)
    result_bias = detailed_backtest(data_with_bias, signal_bias, REALISTIC_COST)
    wf_bias = walk_forward(data_with_bias, strat_func, REALISTIC_COST)

    # Without Bias
    signal_no_bias = strat_func(data_no_bias)
    result_no_bias = detailed_backtest(data_no_bias, signal_no_bias, REALISTIC_COST)
    wf_no_bias = walk_forward(data_no_bias, strat_func, REALISTIC_COST)

    if result_bias and result_no_bias:
        print(f"\n  {'지표':<15} {'With Bias':<12} {'No Bias':<12} {'차이'}")
        print(f"  {'-'*50}")
        print(f"  {'Sharpe':<15} {result_bias['sharpe']:.3f}{'':<8} {result_no_bias['sharpe']:.3f}{'':<8} {result_bias['sharpe'] - result_no_bias['sharpe']:+.3f}")
        print(f"  {'CAGR':<15} {result_bias['annual_return']*100:.1f}%{'':<7} {result_no_bias['annual_return']*100:.1f}%{'':<7} {(result_bias['annual_return'] - result_no_bias['annual_return'])*100:+.1f}%p")
        print(f"  {'MDD':<15} {result_bias['mdd']*100:.1f}%{'':<7} {result_no_bias['mdd']*100:.1f}%{'':<7} {(result_bias['mdd'] - result_no_bias['mdd'])*100:+.1f}%p")

        if wf_bias and wf_no_bias:
            print(f"  {'WF Sharpe':<15} {wf_bias['sharpe']:.3f}{'':<8} {wf_no_bias['sharpe']:.3f}")

        comparison_results.append({
            'strategy': strat_name,
            'with_bias_sharpe': result_bias['sharpe'],
            'with_bias_cagr': result_bias['annual_return'],
            'with_bias_mdd': result_bias['mdd'],
            'no_bias_sharpe': result_no_bias['sharpe'],
            'no_bias_cagr': result_no_bias['annual_return'],
            'no_bias_mdd': result_no_bias['mdd'],
            'sharpe_diff': result_bias['sharpe'] - result_no_bias['sharpe'],
            'wf_with_bias': wf_bias['sharpe'] if wf_bias else None,
            'wf_no_bias': wf_no_bias['sharpe'] if wf_no_bias else None,
        })

# 레버리지 ETF 적용 (Bias 제거 버전)
print("\n" + "=" * 80)
print("레버리지별 성과 (Look-Ahead Bias 제거 후)")
print("=" * 80)

leverage_results = []

for strat_name, strat_func in strategies.items():
    signal = strat_func(data_no_bias)

    for leverage in [1.0, 2.0]:
        # 레버리지 적용
        leveraged_returns = data_no_bias['returns'] * leverage
        data_lev = data_no_bias.copy()
        data_lev['returns'] = leveraged_returns

        result = detailed_backtest(data_lev, signal, REALISTIC_COST * (1.5 if leverage > 1 else 1))

        if result:
            leverage_results.append({
                'strategy': strat_name,
                'leverage': leverage,
                'sharpe': result['sharpe'],
                'cagr': result['annual_return'],
                'mdd': result['mdd'],
            })

            print(f"\n{strat_name} ({leverage}x):")
            print(f"  Sharpe: {result['sharpe']:.3f}")
            print(f"  CAGR: {result['annual_return']*100:.1f}%")
            print(f"  MDD: {result['mdd']*100:.1f}%")

# 결론
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

print("\n1. Look-Ahead Bias 영향:")
for r in comparison_results:
    print(f"   {r['strategy']}: Sharpe {r['with_bias_sharpe']:.3f} → {r['no_bias_sharpe']:.3f} (차이: {r['sharpe_diff']:+.3f})")

print("\n2. 레버리지 ETF 실전 적합성:")
viable_2x = [r for r in leverage_results if r['leverage'] == 2.0 and r['sharpe'] > 0.5 and r['mdd'] > -0.30]
if viable_2x:
    print("   유효한 2x ETF 전략:")
    for v in viable_2x:
        print(f"   - {v['strategy']}: Sharpe {v['sharpe']:.3f}, MDD {v['mdd']*100:.1f}%")
else:
    print("   2x ETF에 적합한 전략 없음 (MDD > -30% 기준)")

viable_1x = [r for r in leverage_results if r['leverage'] == 1.0 and r['sharpe'] > 0.3 and r['mdd'] > -0.25]
if viable_1x:
    print("\n   유효한 1x ETF 전략:")
    for v in viable_1x:
        print(f"   - {v['strategy']}: Sharpe {v['sharpe']:.3f}, MDD {v['mdd']*100:.1f}%")

# 최종 권장
print("\n3. 최종 권장:")
if viable_2x:
    best = max(viable_2x, key=lambda x: x['sharpe'])
    print(f"   2x ETF 사용 가능: {best['strategy']} (Sharpe {best['sharpe']:.3f})")
elif viable_1x:
    best = max(viable_1x, key=lambda x: x['sharpe'])
    print(f"   1x ETF 권장: {best['strategy']} (Sharpe {best['sharpe']:.3f})")
    print("   2x ETF는 MDD 과다로 부적합")
else:
    print("   VIX 기반 ETF 전략은 실전 부적합")
    print("   원본 검증의 높은 Sharpe는 Look-Ahead Bias 때문")

# 저장
output = {
    'generated': datetime.now().isoformat(),
    'type': 'lookahead_bias_comparison',
    'conclusion': 'Original validation had look-ahead bias in VIX data',
    'comparison': comparison_results,
    'leverage_results': leverage_results,
    'recommendation': {
        'viable_2x': viable_2x,
        'viable_1x': viable_1x,
    }
}

output_path = f'{OUTPUT_DIR}/etf_lookahead_bias_comparison.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)

print(f"\n결과 저장: {output_path}")
