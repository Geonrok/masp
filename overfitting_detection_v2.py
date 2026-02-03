"""
과적합(Overfitting) 정밀 검증 V2
- 원래 전략 클래스 사용
- In-Sample vs Out-of-Sample 비교
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\투자\Multi-Asset Strategy Platform")

from kosdaq_live_trading_validation import (
    CombBestStrategy,
    TripleADXStrategy,
    TripleV5Strategy,
    TripleVolStrategy,
)

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
print("과적합(Overfitting) 정밀 검증 V2 - 원래 전략 사용")
print("=" * 80)

# 기간 분할
split_date = "2021-01-01"
in_sample = data[data.index < split_date]
out_sample = data[data.index >= split_date]

print(
    f"\n전체 데이터: {data.index[0].date()} ~ {data.index[-1].date()} ({len(data)}일)"
)
print(
    f"In-Sample: {in_sample.index[0].date()} ~ {in_sample.index[-1].date()} ({len(in_sample)}일)"
)
print(
    f"Out-of-Sample: {out_sample.index[0].date()} ~ {out_sample.index[-1].date()} ({len(out_sample)}일)"
)

# 검증된 8개 전략 인스턴스
strategies = [
    TripleV5Strategy(14, 38, 14, 78, 20, "both"),
    TripleV5Strategy(14, 33, 14, 73, 20, "both"),
    TripleV5Strategy(14, 35, 14, 75, 20, "both"),
    TripleVolStrategy(14, 35, 75, 0.8, "both"),
    TripleVolStrategy(14, 38, 78, 0.8, "both"),
    TripleADXStrategy(14, 35, 75, 14, 25, "both"),
    TripleADXStrategy(14, 38, 78, 14, 25, "both"),
    CombBestStrategy(14, 35, 75, 70, 20, "both"),
]


# 기본 백테스트 함수
def simple_backtest(strategy, df, commission=0.0001, slippage=0.001):
    """간단한 백테스트 - 신호 → 수익률"""
    signals = strategy.generate_signals(df)

    if len(signals) < 10:
        return {
            "sharpe": 0,
            "return": 0,
            "trades": len(signals),
            "win_rate": 0,
            "mdd": 0,
        }

    # 일별 수익률 계산
    daily_returns = df["Close"].pct_change()

    # 포지션 시리즈 생성
    position = pd.Series(0, index=df.index)
    for sig in signals:
        if sig.date in position.index:
            position.loc[sig.date :] = sig.signal

    # 포지션 shift (신호 다음날 진입)
    position = position.shift(1).fillna(0)

    # 거래 비용 계산
    trades = position.diff().abs()
    costs = trades * (commission + slippage)

    # 전략 수익률
    strat_returns = position * daily_returns - costs

    # 성과 지표
    if strat_returns.std() > 0:
        sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    total_return = (1 + strat_returns).prod() - 1

    # MDD
    cum_returns = (1 + strat_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    mdd = drawdown.min()

    # 승률
    trade_returns = strat_returns[position != 0]
    win_rate = (
        (trade_returns > 0).sum() / len(trade_returns) if len(trade_returns) > 0 else 0
    )

    return {
        "sharpe": sharpe,
        "return": total_return,
        "trades": len(signals),
        "win_rate": win_rate,
        "mdd": mdd,
    }


print("\n" + "=" * 80)
print("1. In-Sample vs Out-of-Sample 성과 비교")
print("=" * 80)

print(f"\n{'전략':<35} {'IS Sharpe':>10} {'OS Sharpe':>10} {'저하율':>10} {'판정':>12}")
print("-" * 82)

results = []
for strategy in strategies:
    # In-Sample
    is_result = simple_backtest(strategy, in_sample)

    # Out-of-Sample
    os_result = simple_backtest(strategy, out_sample)

    # 저하율 계산
    if abs(is_result["sharpe"]) > 0.01:
        decay = (
            (os_result["sharpe"] - is_result["sharpe"]) / abs(is_result["sharpe"]) * 100
        )
    else:
        decay = 0

    # 판정
    if os_result["sharpe"] < 0:
        verdict = "[X] 과적합"
    elif os_result["sharpe"] < 0.3:
        verdict = "[!] 위험"
    elif decay < -50:
        verdict = "[?] 의심"
    else:
        verdict = "[OK] 양호"

    results.append(
        {
            "name": strategy.name,
            "is_sharpe": is_result["sharpe"],
            "os_sharpe": os_result["sharpe"],
            "decay": decay,
            "is_return": is_result["return"],
            "os_return": os_result["return"],
            "is_trades": is_result["trades"],
            "os_trades": os_result["trades"],
            "is_winrate": is_result["win_rate"],
            "os_winrate": os_result["win_rate"],
            "is_mdd": is_result["mdd"],
            "os_mdd": os_result["mdd"],
            "verdict": verdict,
        }
    )

    print(
        f"{strategy.name:<35} {is_result['sharpe']:>10.3f} {os_result['sharpe']:>10.3f} {decay:>9.1f}% {verdict:>12}"
    )

print("\n" + "=" * 80)
print("2. 상세 성과 비교")
print("=" * 80)

print(
    f"\n{'전략':<35} {'IS 수익률':>12} {'OS 수익률':>12} {'IS MDD':>10} {'OS MDD':>10}"
)
print("-" * 85)

for r in results:
    print(
        f"{r['name']:<35} {r['is_return']*100:>11.1f}% {r['os_return']*100:>11.1f}% {r['is_mdd']*100:>9.1f}% {r['os_mdd']*100:>9.1f}%"
    )

print(f"\n{'전략':<35} {'IS 승률':>10} {'OS 승률':>10} {'IS 거래':>10} {'OS 거래':>10}")
print("-" * 80)

for r in results:
    print(
        f"{r['name']:<35} {r['is_winrate']*100:>9.1f}% {r['os_winrate']*100:>9.1f}% {r['is_trades']:>10} {r['os_trades']:>10}"
    )

print("\n" + "=" * 80)
print("3. 연도별 성과 안정성")
print("=" * 80)

# 연도별 Sharpe 계산
years = range(2010, 2027)
yearly_data = {}

for strategy in strategies[:4]:  # 처음 4개만
    yearly_sharpes = []
    for year in years:
        year_data = data[data.index.year == year]
        if len(year_data) < 50:
            continue
        result = simple_backtest(strategy, year_data)
        yearly_sharpes.append(result["sharpe"])
    yearly_data[strategy.name] = yearly_sharpes

print("\n연도별 Sharpe (2010-2026):")
print(f"{'연도':<6}", end="")
for name in list(yearly_data.keys()):
    short = name[:15]
    print(f"{short:>16}", end="")
print()
print("-" * 70)

for i, year in enumerate(range(2010, 2026)):
    print(f"{year:<6}", end="")
    for name in yearly_data:
        if i < len(yearly_data[name]):
            val = yearly_data[name][i]
            sign = "+" if val > 0 else " "
            print(f"{sign}{val:>14.2f}", end="")
        else:
            print(f"{'N/A':>16}", end="")
    print()

# 표준편차 계산
print(f"\n{'전략':<35} {'평균 Sharpe':>12} {'표준편차':>12} {'안정성':>10}")
print("-" * 75)

for name, sharpes in yearly_data.items():
    if len(sharpes) > 1:
        mean = np.mean(sharpes)
        std = np.std(sharpes)
        stability = "안정" if std < 0.8 else "불안정"
        print(f"{name:<35} {mean:>12.3f} {std:>12.3f} {stability:>10}")

print("\n" + "=" * 80)
print("4. Walk-Forward 검증")
print("=" * 80)

# 3년 학습, 1년 테스트로 Walk-Forward
wf_results = []
periods = [
    ("2010-2012", "2013"),
    ("2011-2013", "2014"),
    ("2012-2014", "2015"),
    ("2013-2015", "2016"),
    ("2014-2016", "2017"),
    ("2015-2017", "2018"),
    ("2016-2018", "2019"),
    ("2017-2019", "2020"),
    ("2018-2020", "2021"),
    ("2019-2021", "2022"),
    ("2020-2022", "2023"),
    ("2021-2023", "2024"),
    ("2022-2024", "2025"),
]

print("\nWalk-Forward 기간별 OS Sharpe (상위 3 전략):")
print(f"{'기간':<20}", end="")
for strategy in strategies[:3]:
    short = strategy.name[:12]
    print(f"{short:>14}", end="")
print()
print("-" * 65)

wf_os_sharpes = {s.name: [] for s in strategies}

for train_period, test_year in periods:
    train_start, train_end = train_period.split("-")

    train_data = data[
        (data.index.year >= int(train_start)) & (data.index.year <= int(train_end))
    ]
    test_data = data[data.index.year == int(test_year)]

    if len(train_data) < 200 or len(test_data) < 50:
        continue

    print(f"{train_period} -> {test_year:<6}", end="")

    for i, strategy in enumerate(strategies):
        os_result = simple_backtest(strategy, test_data)
        wf_os_sharpes[strategy.name].append(os_result["sharpe"])

        if i < 3:
            val = os_result["sharpe"]
            sign = "+" if val > 0 else " "
            print(f"{sign}{val:>13.2f}", end="")
    print()

print("\n" + "=" * 80)
print("5. 최종 과적합 진단")
print("=" * 80)

print(
    f"\n{'전략':<35} {'전체 Sharpe':>12} {'OS Sharpe':>12} {'WF 평균':>10} {'최종 판정':>15}"
)
print("-" * 90)

final_diagnosis = []
for r in results:
    wf_mean = np.mean(wf_os_sharpes[r["name"]]) if wf_os_sharpes[r["name"]] else 0

    # 종합 점수
    score = 0
    issues = []

    # 1. OS Sharpe
    if r["os_sharpe"] < 0:
        score += 3
        issues.append("OS Sharpe 음수")
    elif r["os_sharpe"] < 0.3:
        score += 2
        issues.append("OS Sharpe < 0.3")
    elif r["os_sharpe"] < 0.5:
        score += 1
        issues.append("OS Sharpe < 0.5")

    # 2. Walk-Forward 평균
    if wf_mean < 0:
        score += 2
        issues.append("WF 평균 음수")
    elif wf_mean < 0.3:
        score += 1
        issues.append("WF 평균 < 0.3")

    # 3. 저하율
    if r["decay"] < -60:
        score += 2
        issues.append("60% 이상 저하")
    elif r["decay"] < -40:
        score += 1
        issues.append("40-60% 저하")

    if score >= 4:
        verdict = "[X] 과적합"
    elif score >= 3:
        verdict = "[!] 고위험"
    elif score >= 2:
        verdict = "[?] 주의"
    elif score >= 1:
        verdict = "[-] 경계"
    else:
        verdict = "[OK] 양호"

    full_sharpe = simple_backtest(strategies[results.index(r)], data)["sharpe"]

    final_diagnosis.append(
        {
            "name": r["name"],
            "full_sharpe": full_sharpe,
            "os_sharpe": r["os_sharpe"],
            "wf_mean": wf_mean,
            "score": score,
            "issues": issues,
            "verdict": verdict,
        }
    )

    print(
        f"{r['name']:<35} {full_sharpe:>12.3f} {r['os_sharpe']:>12.3f} {wf_mean:>10.3f} {verdict:>15}"
    )

# 결론
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

overfit = sum(1 for d in final_diagnosis if d["score"] >= 4)
high_risk = sum(1 for d in final_diagnosis if d["score"] == 3)
caution = sum(1 for d in final_diagnosis if d["score"] == 2)
safe = sum(1 for d in final_diagnosis if d["score"] < 2)

print(f"""
과적합 진단 결과:
  - [X] 과적합: {overfit}개
  - [!] 고위험: {high_risk}개
  - [?] 주의: {caution}개
  - [OK] 양호: {safe}개
""")

# 실거래 추천 전략
viable = [
    d
    for d in final_diagnosis
    if d["os_sharpe"] > 0.5 and d["wf_mean"] > 0.3 and d["score"] < 3
]
print("실거래 권장 전략 (OS Sharpe > 0.5, WF 평균 > 0.3):")
if viable:
    for v in sorted(viable, key=lambda x: x["os_sharpe"], reverse=True):
        print(
            f"  - {v['name']}: OS Sharpe={v['os_sharpe']:.3f}, WF평균={v['wf_mean']:.3f}"
        )
else:
    cautious = [d for d in final_diagnosis if d["os_sharpe"] > 0.3 and d["score"] < 4]
    if cautious:
        print("  권장 전략 없음. 차선책 (주의 요함):")
        for c in sorted(cautious, key=lambda x: x["os_sharpe"], reverse=True)[:3]:
            print(f"  - {c['name']}: OS Sharpe={c['os_sharpe']:.3f}")
    else:
        print("  실거래 권장 전략 없음 - 전략 재개발 필요")

# 결과 저장
output = {
    "results": results,
    "final_diagnosis": final_diagnosis,
    "summary": {
        "overfit": overfit,
        "high_risk": high_risk,
        "caution": caution,
        "safe": safe,
    },
}

output_path = Path(
    "E:/투자/data/kosdaq_futures/validated_strategies/overfitting_analysis_v2.json"
)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\n결과 저장: {output_path}")
