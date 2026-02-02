"""
과적합(Overfitting) 정밀 검증
- In-Sample vs Out-of-Sample 성과 비교
- 파라미터 민감도 분석
- 시간에 따른 성과 안정성
- 거래 횟수 대비 파라미터 수
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

# 데이터 로드
data_path = Path("E:/투자/data/kosdaq_futures")
clean_path = data_path / "kosdaq150_futures_ohlcv_clean.parquet"
cache_path = data_path / "kosdaq150_futures_ohlcv.parquet"

if clean_path.exists():
    data = pd.read_parquet(clean_path)
elif cache_path.exists():
    data = pd.read_parquet(cache_path)
else:
    raise FileNotFoundError("데이터 파일을 찾을 수 없습니다")

print("=" * 80)
print("과적합(Overfitting) 정밀 검증")
print("=" * 80)

# 기간 분할
# In-Sample: 2010-2020 (10년)
# Out-of-Sample: 2021-2026 (5년)
split_date = '2021-01-01'
in_sample = data[data.index < split_date]
out_sample = data[data.index >= split_date]

print(f"\nIn-Sample 기간: {in_sample.index[0].date()} ~ {in_sample.index[-1].date()} ({len(in_sample)}일)")
print(f"Out-of-Sample 기간: {out_sample.index[0].date()} ~ {out_sample.index[-1].date()} ({len(out_sample)}일)")

# 검증된 8개 전략 정의
def generate_triple_signals(df, cmo_period, wr_low, wr_high, bb_period):
    """기본 Triple 전략 신호"""
    close = df['Close']
    high = df['High']
    low = df['Low']

    # CMO
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(cmo_period).sum()
    loss = (-delta.where(delta < 0, 0)).rolling(cmo_period).sum()
    cmo = 100 * (gain - loss) / (gain + loss)

    # Williams %R
    hh = high.rolling(cmo_period).max()
    ll = low.rolling(cmo_period).min()
    wr = -100 * (hh - close) / (hh - ll)

    # Bollinger %B
    sma = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    bb_upper = sma + 2 * std
    bb_lower = sma - 2 * std
    pctb = (close - bb_lower) / (bb_upper - bb_lower)

    signals = pd.Series(0, index=df.index)

    # Long: 과매도
    long_cond = (cmo < -50) & (wr < wr_low) & (pctb < 0.2)
    # Short: 과매수
    short_cond = (cmo > 50) & (wr > wr_high) & (pctb > 0.8)

    signals[long_cond] = 1
    signals[short_cond] = -1

    return signals

def generate_vol_signals(df, cmo_period, wr_low, wr_high, vol_mult):
    """거래량 필터 추가 전략"""
    signals = generate_triple_signals(df, cmo_period, wr_low, wr_high, 20)

    # 거래량 필터
    vol_ma = df['Volume'].rolling(20).mean()
    vol_high = df['Volume'] > vol_ma * vol_mult

    signals = signals.where(vol_high, 0)
    return signals

def generate_adx_signals(df, cmo_period, wr_low, wr_high, adx_thresh):
    """ADX 필터 추가 전략"""
    signals = generate_triple_signals(df, cmo_period, wr_low, wr_high, 20)

    # ADX 계산
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr
    minus_di = 100 * minus_dm.rolling(14).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(14).mean()

    signals = signals.where(adx > adx_thresh, 0)
    return signals

# 전략 목록
strategies = {
    'TripleV5_14_38_14_78_20': lambda df: generate_triple_signals(df, 14, -78, -38, 20),
    'TripleV5_14_33_14_73_20': lambda df: generate_triple_signals(df, 14, -73, -33, 20),
    'TripleV5_14_35_14_75_20': lambda df: generate_triple_signals(df, 14, -75, -35, 20),
    'TripleVol_14_35_75_0.8': lambda df: generate_vol_signals(df, 14, -75, -35, 0.8),
    'TripleVol_14_38_78_0.8': lambda df: generate_vol_signals(df, 14, -78, -38, 0.8),
    'TripleADX_14_35_75_25': lambda df: generate_adx_signals(df, 14, -75, -35, 25),
    'TripleADX_14_38_78_25': lambda df: generate_adx_signals(df, 14, -78, -38, 25),
    'CombBest_14_35_75_70_20': lambda df: generate_triple_signals(df, 14, -75, -35, 20),  # 간소화
}

# 백테스트 실행 함수
def run_backtest_simple(df, signals):
    """간단한 백테스트"""
    returns = df['Close'].pct_change()

    # 신호 다음날 진입
    position = signals.shift(1).fillna(0)

    # 슬리피지/수수료 적용
    trade_cost = 0.001 + 0.0001  # 0.11%
    trades = position.diff().abs()
    costs = trades * trade_cost

    strat_returns = position * returns - costs

    # 성과 지표
    total_return = (1 + strat_returns).prod() - 1

    daily_std = strat_returns.std()
    if daily_std > 0:
        sharpe = strat_returns.mean() / daily_std * np.sqrt(252)
    else:
        sharpe = 0

    cum_returns = (1 + strat_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    mdd = drawdown.min()

    num_trades = (signals != 0).sum()

    if num_trades > 0:
        winning = strat_returns[position != 0]
        win_rate = (winning > 0).sum() / len(winning) if len(winning) > 0 else 0
    else:
        win_rate = 0

    return {
        'return': total_return,
        'sharpe': sharpe,
        'mdd': mdd,
        'trades': num_trades,
        'win_rate': win_rate
    }

print("\n" + "=" * 80)
print("1. In-Sample vs Out-of-Sample 성과 비교")
print("=" * 80)

results = []
for name, signal_func in strategies.items():
    # In-Sample
    is_signals = signal_func(in_sample)
    is_result = run_backtest_simple(in_sample, is_signals)

    # Out-of-Sample
    os_signals = signal_func(out_sample)
    os_result = run_backtest_simple(out_sample, os_signals)

    # 성과 저하율
    if is_result['sharpe'] != 0:
        sharpe_decay = (os_result['sharpe'] - is_result['sharpe']) / abs(is_result['sharpe']) * 100
    else:
        sharpe_decay = 0

    results.append({
        'name': name,
        'is_sharpe': is_result['sharpe'],
        'os_sharpe': os_result['sharpe'],
        'sharpe_decay': sharpe_decay,
        'is_return': is_result['return'],
        'os_return': os_result['return'],
        'is_trades': is_result['trades'],
        'os_trades': os_result['trades'],
        'is_winrate': is_result['win_rate'],
        'os_winrate': os_result['win_rate']
    })

print(f"\n{'전략':<30} {'IS Sharpe':>10} {'OS Sharpe':>10} {'저하율':>10} {'판정':>10}")
print("-" * 75)

overfitting_count = 0
for r in results:
    # 과적합 판정 기준
    # 1. Sharpe 저하율 > 50%
    # 2. Out-of-Sample Sharpe < 0.3
    is_overfit = r['sharpe_decay'] < -50 or r['os_sharpe'] < 0.3
    if is_overfit:
        overfitting_count += 1

    verdict = "[!] 과적합" if is_overfit else "[OK] 양호"

    print(f"{r['name']:<30} {r['is_sharpe']:>10.3f} {r['os_sharpe']:>10.3f} {r['sharpe_decay']:>9.1f}% {verdict:>10}")

print("\n" + "=" * 80)
print("2. 파라미터 민감도 분석")
print("=" * 80)

# WR 파라미터 변화에 따른 성과
print("\nWilliams %R 파라미터 변화 분석 (CMO=14, BB=20 고정):")
print(f"{'WR Low/High':<15} {'Sharpe':>10} {'거래수':>10} {'승률':>10}")
print("-" * 50)

param_results = []
for wr_low in [-70, -73, -75, -78, -80, -83, -85]:
    wr_high = wr_low + 40  # 예: -75 -> -35
    signals = generate_triple_signals(data, 14, wr_low, wr_high, 20)
    result = run_backtest_simple(data, signals)
    param_results.append({
        'wr_low': wr_low,
        'sharpe': result['sharpe'],
        'trades': result['trades'],
        'win_rate': result['win_rate']
    })
    print(f"{wr_low}/{wr_high:<10} {result['sharpe']:>10.3f} {result['trades']:>10} {result['win_rate']:>9.1%}")

# 파라미터 민감도 계산
sharpes = [p['sharpe'] for p in param_results]
sensitivity = np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) > 0 else float('inf')
print(f"\n파라미터 민감도 (CV): {sensitivity:.2f}")
if sensitivity > 0.5:
    print("  [[!] 경고] 높은 파라미터 민감도 - 과적합 가능성 높음")
else:
    print("  [[OK]] 안정적인 파라미터 범위")

print("\n" + "=" * 80)
print("3. 롤링 성과 안정성 (연도별)")
print("=" * 80)

# 연도별 성과
years = range(2010, 2027)
yearly_sharpes = {name: [] for name in strategies.keys()}

print(f"\n{'연도':<6}", end='')
for name in list(strategies.keys())[:4]:  # 처음 4개만 표시
    short_name = name.replace('Triple', '').replace('_both', '')[:12]
    print(f"{short_name:>12}", end='')
print()
print("-" * 60)

for year in years:
    year_data = data[data.index.year == year]
    if len(year_data) < 50:
        continue

    print(f"{year:<6}", end='')
    for i, (name, signal_func) in enumerate(strategies.items()):
        signals = signal_func(year_data)
        result = run_backtest_simple(year_data, signals)
        yearly_sharpes[name].append(result['sharpe'])

        if i < 4:  # 처음 4개만 표시
            color = "+" if result['sharpe'] > 0 else ""
            print(f"{color}{result['sharpe']:>11.2f}", end='')
    print()

print("\n연도별 Sharpe 표준편차 (낮을수록 안정적):")
for name in list(strategies.keys())[:4]:
    sharpes = yearly_sharpes[name]
    if len(sharpes) > 1:
        std = np.std(sharpes)
        mean = np.mean(sharpes)
        short_name = name.replace('Triple', '').replace('_both', '')[:15]
        stability = "안정" if std < 1.0 else "불안정"
        print(f"  {short_name:<20}: 평균={mean:>6.2f}, 표준편차={std:>5.2f} [{stability}]")

print("\n" + "=" * 80)
print("4. 자유도 분석 (거래 수 vs 파라미터 수)")
print("=" * 80)

# 전략별 자유도 계산
strategy_params = {
    'TripleV5': 4,  # CMO기간, WR_low, WR_high, BB기간
    'TripleVol': 5,  # + 거래량 배수
    'TripleADX': 5,  # + ADX 임계값
    'CombBest': 6,   # + RSI, Stochastic
}

print(f"\n{'전략 유형':<15} {'파라미터수':>10} {'거래수':>10} {'거래/파라미터':>15} {'판정':>10}")
print("-" * 65)

for name, signal_func in strategies.items():
    signals = signal_func(data)
    result = run_backtest_simple(data, signals)

    # 전략 유형 추출
    if 'V5' in name:
        strat_type = 'TripleV5'
    elif 'Vol' in name:
        strat_type = 'TripleVol'
    elif 'ADX' in name:
        strat_type = 'TripleADX'
    else:
        strat_type = 'CombBest'

    num_params = strategy_params[strat_type]
    trades_per_param = result['trades'] / num_params if num_params > 0 else 0

    # 일반적으로 파라미터당 20-30개 이상의 거래가 필요
    verdict = "[OK] 충분" if trades_per_param >= 20 else "[!] 부족"

    print(f"{name:<30} {num_params:>5} {result['trades']:>10} {trades_per_param:>15.1f} {verdict:>10}")

print("\n" + "=" * 80)
print("5. 최종 과적합 진단")
print("=" * 80)

# 종합 점수
diagnosis = []
for r in results:
    score = 0
    issues = []

    # 1. IS vs OS Sharpe 비교
    if r['sharpe_decay'] < -50:
        score += 2
        issues.append("Sharpe 50% 이상 저하")
    elif r['sharpe_decay'] < -30:
        score += 1
        issues.append("Sharpe 30-50% 저하")

    # 2. OS Sharpe 절대값
    if r['os_sharpe'] < 0:
        score += 2
        issues.append("OS Sharpe 음수")
    elif r['os_sharpe'] < 0.3:
        score += 1
        issues.append("OS Sharpe 0.3 미만")

    # 3. 승률 저하
    if r['is_winrate'] - r['os_winrate'] > 0.15:
        score += 1
        issues.append("승률 15%p 이상 저하")

    diagnosis.append({
        'name': r['name'],
        'score': score,
        'issues': issues,
        'os_sharpe': r['os_sharpe']
    })

print(f"\n{'전략':<30} {'과적합점수':>10} {'OS Sharpe':>10} {'진단':>15}")
print("-" * 70)

for d in sorted(diagnosis, key=lambda x: x['score'], reverse=True):
    if d['score'] >= 3:
        verdict = "[HIGH] 과적합"
    elif d['score'] >= 2:
        verdict = "[MED] 의심"
    elif d['score'] >= 1:
        verdict = "[LOW] 주의"
    else:
        verdict = "[SAFE] 양호"

    print(f"{d['name']:<30} {d['score']:>10} {d['os_sharpe']:>10.3f} {verdict:>15}")
    if d['issues']:
        for issue in d['issues']:
            print(f"    - {issue}")

# 최종 결론
high_risk = sum(1 for d in diagnosis if d['score'] >= 3)
medium_risk = sum(1 for d in diagnosis if d['score'] == 2)
low_risk = sum(1 for d in diagnosis if d['score'] == 1)
safe = sum(1 for d in diagnosis if d['score'] == 0)

print("\n" + "=" * 80)
print(" 최종 결론")
print("=" * 80)

print(f"""
과적합 진단 결과:
  - [HIGH] 과적합 (고위험): {high_risk}개
  - [MED] 의심 (중위험): {medium_risk}개
  - [LOW] 주의 (저위험): {low_risk}개
  - [SAFE] 양호: {safe}개

종합 판단:
""")

if high_risk >= 4:
    print("  [X] 심각한 과적합 - 전략 전면 재검토 필요")
    print("  - 모든 전략이 동일한 핵심 로직 사용")
    print("  - In-Sample에서 최적화된 파라미터가 Out-of-Sample에서 실패")
    print("  - 실거래 적용 비권장")
elif high_risk >= 2 or medium_risk >= 3:
    print("  [!] 상당한 과적합 우려")
    print("  - 일부 전략만 Out-of-Sample에서 성과 유지")
    print("  - OS Sharpe > 0.5인 전략만 고려")
elif medium_risk >= 1:
    print("  [!] 부분적 과적합 가능성")
    print("  - 대부분 전략이 Out-of-Sample에서도 유효")
    print("  - 단, 성과 저하는 예상해야 함")
else:
    print("  [OK] 과적합 위험 낮음")
    print("  - Out-of-Sample 성과가 In-Sample과 유사")

# 추천 전략
print("\n실거래 고려 가능 전략 (OS Sharpe > 0.5):")
viable = [d for d in diagnosis if d['os_sharpe'] > 0.5 and d['score'] < 3]
if viable:
    for v in sorted(viable, key=lambda x: x['os_sharpe'], reverse=True):
        print(f"  - {v['name']}: OS Sharpe = {v['os_sharpe']:.3f}")
else:
    print("  없음 - 전략 재개발 권장")

# 결과 저장
output = {
    'diagnosis': diagnosis,
    'summary': {
        'high_risk': high_risk,
        'medium_risk': medium_risk,
        'low_risk': low_risk,
        'safe': safe
    },
    'recommendation': [d['name'] for d in viable] if viable else []
}

output_path = Path("E:/투자/data/kosdaq_futures/validated_strategies/overfitting_analysis.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n분석 결과 저장: {output_path}")
