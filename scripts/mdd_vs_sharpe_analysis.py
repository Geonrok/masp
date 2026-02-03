"""
MDD 고집 vs Sharpe 최적화 - 어떤 것이 더 합리적인가?
================================================================================
비교 기준:
1. 장기 복리 효과 (기하평균 수익률)
2. 파산 확률 (Gambler's Ruin)
3. 심리적 감내 가능성
4. 포트폴리오 관점 (Kelly Criterion)
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def simulate_long_term(ann_return, ann_vol, mdd, years=20, simulations=10000):
    """장기 시뮬레이션"""
    np.random.seed(42)

    # 일별 파라미터
    daily_ret = ann_return / 252
    daily_vol = ann_vol / np.sqrt(252)

    results = []

    for _ in range(simulations):
        equity = 1.0
        peak = 1.0
        max_dd = 0

        for day in range(252 * years):
            # 일별 수익률 (정규분포 가정)
            ret = np.random.normal(daily_ret, daily_vol)
            equity *= 1 + ret

            # Drawdown 체크
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak
            max_dd = min(max_dd, dd)

            # 파산 체크 (80% 손실)
            if equity < 0.2:
                break

        results.append(
            {"final_equity": equity, "max_dd": max_dd, "bankrupt": equity < 0.2}
        )

    df = pd.DataFrame(results)

    return {
        "median_return": (df["final_equity"].median() ** (1 / years) - 1) * 100,
        "mean_return": (df["final_equity"].mean() ** (1 / years) - 1) * 100,
        "worst_10pct": df["final_equity"].quantile(0.1),
        "best_10pct": df["final_equity"].quantile(0.9),
        "bankrupt_rate": df["bankrupt"].mean() * 100,
        "median_mdd": df["max_dd"].median() * 100,
        "worst_mdd": df["max_dd"].min() * 100,
    }


def kelly_criterion(win_rate, win_loss_ratio):
    """켈리 기준 최적 베팅 비율"""
    # Kelly % = W - (1-W)/R
    # W = 승률, R = 평균이익/평균손실
    kelly = win_rate - (1 - win_rate) / win_loss_ratio
    return max(0, kelly)


def geometric_mean_return(arithmetic_mean, volatility):
    """기하평균 수익률 (실제 복리 수익률)"""
    # G ≈ A - V²/2
    return arithmetic_mean - (volatility**2) / 2


def main():
    print("=" * 80)
    print("MDD 고집 vs Sharpe 최적화 - 합리성 분석")
    print("=" * 80)
    print(f"시간: {datetime.now()}")

    # 두 전략 비교
    strategies = {
        "MDD_2.3%_고집": {
            "ann_return": 0.0123,  # 1.23%
            "ann_vol": 0.014,  # 추정
            "sharpe": 0.885,
            "mdd": -0.0226,
        },
        "Sharpe_최적화": {
            "ann_return": 0.0419,  # 4.19%
            "ann_vol": 0.050,  # 추정
            "sharpe": 0.830,
            "mdd": -0.0728,
        },
        "Conservative_중간": {
            "ann_return": 0.0284,  # 2.84%
            "ann_vol": 0.032,  # 추정
            "sharpe": 0.888,
            "mdd": -0.0425,
        },
    }

    print("\n" + "=" * 80)
    print("1. 기본 지표 비교")
    print("=" * 80)

    print(f"\n{'전략':<20} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8}")
    print("-" * 50)
    for name, s in strategies.items():
        print(
            f"{name:<20} {s['ann_return']*100:>7.2f}% {s['mdd']*100:>7.2f}% {s['sharpe']:>8.3f}"
        )

    # 기하평균 수익률 계산
    print("\n" + "=" * 80)
    print("2. 기하평균 수익률 (실제 복리 수익률)")
    print("=" * 80)

    print("""
산술평균 vs 기하평균:
- 산술평균: 단순 평균 (예: +50%, -50% → 평균 0%)
- 기하평균: 실제 복리 (예: +50%, -50% → 실제 -25%)

공식: G = A - V^2/2 (근사)
- 변동성이 높을수록 기하평균이 더 낮아짐
- 이것이 변동성 손실 (Volatility Drag)
""")

    for name, s in strategies.items():
        geo_return = geometric_mean_return(s["ann_return"], s["ann_vol"])
        vol_drag = s["ann_return"] - geo_return
        print(f"\n[{name}]")
        print(f"  산술평균: {s['ann_return']*100:.2f}%")
        print(f"  기하평균: {geo_return*100:.2f}%")
        print(f"  변동성 손실: {vol_drag*100:.3f}%")

    # 장기 시뮬레이션
    print("\n" + "=" * 80)
    print("3. 20년 장기 시뮬레이션 (10,000회)")
    print("=" * 80)

    for name, s in strategies.items():
        sim = simulate_long_term(
            s["ann_return"], s["ann_vol"], s["mdd"], years=20, simulations=10000
        )

        print(f"\n[{name}]")
        print(f"  중간값 연 수익률: {sim['median_return']:.2f}%")
        print(f"  최악 10% 최종 자산: {sim['worst_10pct']:.2f}x")
        print(f"  최고 10% 최종 자산: {sim['best_10pct']:.2f}x")
        print(f"  파산 확률 (80% 손실): {sim['bankrupt_rate']:.2f}%")
        print(f"  중간값 MDD: {sim['median_mdd']:.1f}%")

    # 심리적 관점
    print("\n" + "=" * 80)
    print("4. 심리적 감내 가능성")
    print("=" * 80)

    print("""
연구 결과 (Kahneman & Tversky):
- 손실의 심리적 고통 = 이익의 기쁨 × 2~2.5배
- MDD -7%의 고통 = +14~17.5% 이익의 기쁨과 동등

실제 투자자 행동:
- MDD -10% 이상: 대부분 전략 포기
- MDD -20% 이상: 패닉 셀링 발생
- MDD -5% 이하: 장기 유지 가능

사용자 케이스 (12년간 MDD 2.3% 유지):
- 심리적 안정감 → 전략 일관성 유지 → 장기 복리 효과
- 이것이 실제로 작동한 검증된 접근법
""")

    # 포트폴리오 관점
    print("\n" + "=" * 80)
    print("5. 포트폴리오 관점 (Kelly Criterion)")
    print("=" * 80)

    print("""
Kelly Criterion: 장기 자산 성장 최대화 베팅 비율

문제: Full Kelly는 변동성이 너무 큼
해결: Half Kelly 또는 Quarter Kelly 사용

실무적 접근:
- 전체 자산의 일부만 이 전략에 배분
- 나머지는 안전자산 또는 비상관 전략
""")

    # 각 전략의 최적 배분 비율 계산
    for name, s in strategies.items():
        # 간단한 Kelly 추정: Sharpe² / (수익률 / 변동성)
        kelly_fraction = (s["sharpe"] ** 2) / 2
        half_kelly = kelly_fraction / 2

        print(f"\n[{name}]")
        print(f"  Full Kelly 배분: {kelly_fraction*100:.1f}%")
        print(f"  Half Kelly 배분: {half_kelly*100:.1f}%")
        print(f"  권장 배분: {min(half_kelly, 0.25)*100:.1f}%")

    # 최종 비교
    print("\n" + "=" * 80)
    print("6. 최종 비교표")
    print("=" * 80)

    comparison = """
| 기준 | MDD 2.3% 고집 | Sharpe 최적화 | 승자 |
|------|---------------|---------------|------|
| CAGR | 1.23% | 4.19% | Sharpe |
| MDD | -2.26% | -7.28% | MDD 고집 |
| Sharpe | 0.885 | 0.830 | MDD 고집 |
| 20년 파산확률 | ~0% | ~0% | 동일 |
| 심리적 안정 | 높음 | 낮음 | MDD 고집 |
| 전략 유지 가능성 | 높음 | 중간 | MDD 고집 |
| 인플레이션 대비 | 부족 | 충분 | Sharpe |
| 변동성 손실 | 0.01% | 0.13% | MDD 고집 |
"""
    print(comparison)

    # 결론
    print("\n" + "=" * 80)
    print("7. 결론: 어떤 것이 더 합리적인가?")
    print("=" * 80)

    print("""
정답: 상황에 따라 다름

[MDD 2.3% 고집이 합리적인 경우]
1. 이 전략이 전체 자산의 100%인 경우
2. 심리적으로 손실 감내가 어려운 경우
3. 단기간 내 자금 인출 가능성이 있는 경우
4. 이미 12년간 검증된 본인의 방식을 유지하고 싶은 경우

[Sharpe 최적화가 합리적인 경우]
1. 이 전략이 전체 자산의 일부 (10-30%)인 경우
2. 다른 비상관 전략과 결합하는 경우
3. 10년 이상 장기 투자가 확실한 경우
4. 인플레이션 대비가 중요한 경우

[권장 절충안]
Conservative (MDD -4.25%, CAGR 2.84%, Sharpe 0.888)
- MDD 5% 이하로 심리적 안정
- CAGR 3%로 인플레이션 대비 가능
- Sharpe 0.888로 효율적

또는:
- 포트폴리오의 20%만 Moderate (MDD -7.28%) 전략
- 나머지 80%는 안전자산
- 전체 포트폴리오 MDD = 약 1.5%, CAGR = 약 1.5% + 안전자산 수익
""")

    # 사용자 맞춤 권장
    print("\n" + "=" * 80)
    print("8. 사용자 맞춤 권장")
    print("=" * 80)

    print("""
사용자 프로필:
- 12년간 MDD 2.3% 유지 (검증된 리스크 관리 능력)
- 선물/옵션 전환 계획 (ETF는 중간 단계)
- 포트폴리오 최적화 경험 보유

권장:

1. ETF 단독 운용 시:
   → Conservative (MDD -4.25%, CAGR 2.84%)
   → 기존 2.3%보다 약간 높지만 수용 가능 범위

2. 포트폴리오 일부로 운용 시:
   → Moderate (MDD -7.28%, CAGR 4.19%)
   → 전체 자산의 30% 배분
   → 나머지 70%로 MDD 헷지

3. 선물 전환 후:
   → Long/Short로 하락장 방어
   → MDD 3-5% + CAGR 5-8% 목표 가능

핵심: MDD 2.3%를 고집하는 것이 '틀린' 것은 아님.
      12년간 검증된 방식이며, 심리적 안정이 장기 성과의 핵심.
      다만 CAGR 1%는 인플레이션 대비 부족할 수 있음.
""")

    return strategies


if __name__ == "__main__":
    main()
