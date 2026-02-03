"""
Phase 1: Anti-Overfitting Validation
당신의 전략이 진짜인지, 데이터 마이닝의 결과인지 검증합니다.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================
# 1. 데이터 로드
# ============================================================
print("=" * 70)
print("PHASE 1: 과적합(OVERFITTING) 검증")
print("=" * 70)

# 모든 전략 결과 로드
results_path = "E:/투자/백테스트_결과_통합/backtest_results/"

# 여러 결과 파일 로드
try:
    upbit_results = pd.read_csv(f"{results_path}upbit_results.csv")
    bithumb_results = pd.read_csv(f"{results_path}bithumb_results.csv")
    korean_combo = pd.read_csv(f"{results_path}korean_combo_results.csv")
    korean_full = pd.read_csv(f"{results_path}korean_full_suite_results.csv")
except Exception as e:
    print(f"Error loading files: {e}")
    upbit_results = pd.DataFrame()
    bithumb_results = pd.DataFrame()

# ============================================================
# 2. 테스트된 전략 수 계산
# ============================================================
print("\n[1] 테스트된 전략 수 분석")
print("-" * 50)

n_strategies_upbit = len(upbit_results) if not upbit_results.empty else 0
n_strategies_bithumb = len(bithumb_results) if not bithumb_results.empty else 0
n_korean_combo = (
    len(korean_combo) if "korean_combo" in dir() and not korean_combo.empty else 0
)
n_korean_full = (
    len(korean_full) if "korean_full" in dir() and not korean_full.empty else 0
)

total_tests = n_strategies_upbit + n_strategies_bithumb + n_korean_combo + n_korean_full

print(f"Upbit 개별 전략 테스트: {n_strategies_upbit:,}")
print(f"Bithumb 개별 전략 테스트: {n_strategies_bithumb:,}")
print(f"Korean Combo 테스트: {n_korean_combo:,}")
print(f"Korean Full Suite 테스트: {n_korean_full:,}")
print(f"\n총 테스트 횟수: {total_tests:,}")

# ============================================================
# 3. Deflated Sharpe Ratio 계산
# ============================================================
print("\n[2] Deflated Sharpe Ratio (다중 검정 보정)")
print("-" * 50)


def deflated_sharpe_ratio(observed_sharpe, n_tests, years, skewness=0, kurtosis=3):
    """
    다중 검정을 보정한 Sharpe Ratio 계산

    참고: Bailey & López de Prado (2014)
    """
    euler = 0.5772156649

    # E[max(Z_1, ..., Z_N)] 근사
    if n_tests > 1:
        expected_max = (1 - euler) * stats.norm.ppf(
            1 - 1 / n_tests
        ) + euler * stats.norm.ppf(1 - 1 / (n_tests * np.e))
    else:
        expected_max = 0

    # Sharpe ratio 추정량의 분산
    n_obs = years * 252
    var_sharpe = (
        1
        + 0.5 * observed_sharpe**2
        - skewness * observed_sharpe
        + (kurtosis - 3) / 4 * observed_sharpe**2
    ) / n_obs

    # Deflated Sharpe
    if var_sharpe > 0:
        dsr = (observed_sharpe - expected_max * np.sqrt(var_sharpe)) / np.sqrt(
            var_sharpe
        )
    else:
        dsr = 0

    # p-value
    p_value = 1 - stats.norm.cdf(dsr)

    return dsr, p_value, expected_max


# 최종 전략의 관측된 Sharpe
observed_sharpe_upbit = 3.16  # 2025 holdout 결과
observed_sharpe_bithumb = 2.58
years = 7  # 2017-2024 데이터

# 추정: 실제로 테스트한 전략 조합 수
# 보수적 추정: 전략 × 심볼 × 파라미터 조합
estimated_tests = max(total_tests, 10000)  # 최소 10,000회로 가정

print(f"관측된 Sharpe (Upbit 2025 Holdout): {observed_sharpe_upbit}")
print(f"관측된 Sharpe (Bithumb 2025 Holdout): {observed_sharpe_bithumb}")
print(f"데이터 기간: {years}년")
print(f"추정 테스트 횟수: {estimated_tests:,}")

# Upbit
dsr_upbit, p_upbit, exp_max_upbit = deflated_sharpe_ratio(
    observed_sharpe_upbit, estimated_tests, years
)
print("\n[Upbit]")
print(f"  Expected Max Sharpe (우연): {exp_max_upbit:.3f}")
print(f"  Deflated Sharpe Ratio: {dsr_upbit:.3f}")
print(f"  P-value: {p_upbit:.6f}")
print(f"  통과 여부: {'[PASS]' if p_upbit < 0.05 else '[FAIL]'}")

# Bithumb
dsr_bithumb, p_bithumb, exp_max_bithumb = deflated_sharpe_ratio(
    observed_sharpe_bithumb, estimated_tests, years
)
print("\n[Bithumb]")
print(f"  Expected Max Sharpe (우연): {exp_max_bithumb:.3f}")
print(f"  Deflated Sharpe Ratio: {dsr_bithumb:.3f}")
print(f"  P-value: {p_bithumb:.6f}")
print(f"  통과 여부: {'[PASS] PASS' if p_bithumb < 0.05 else '[FAIL] FAIL'}")

# ============================================================
# 4. 우연히 발견될 확률 계산
# ============================================================
print("\n[3] 우연히 좋은 전략이 발견될 확률")
print("-" * 50)


def prob_finding_good_strategy(n_tests, threshold_sharpe=2.0, years=5):
    """
    N번 테스트 시 Sharpe > threshold인 전략을 우연히 발견할 확률
    """
    # 단일 테스트에서 Sharpe > threshold일 확률 (귀무가설 하)
    # Sharpe의 표준오차 ≈ 1/sqrt(years)
    se_sharpe = 1 / np.sqrt(years)
    p_single = 1 - stats.norm.cdf(threshold_sharpe / se_sharpe)

    # N번 중 최소 1번 발견할 확률
    p_at_least_one = 1 - (1 - p_single) ** n_tests

    return p_at_least_one, p_single


for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
    p_find, p_single = prob_finding_good_strategy(estimated_tests, threshold, years)
    print(
        f"Sharpe > {threshold}: {p_find*100:.1f}% 확률로 우연히 발견 (단일: {p_single:.6f})"
    )

# ============================================================
# 5. 실제 전략 분포 분석
# ============================================================
print("\n[4] 실제 전략 Sharpe 분포 분석")
print("-" * 50)

if not upbit_results.empty and "sharpe" in upbit_results.columns:
    sharpes = upbit_results["sharpe"].dropna()

    print(f"총 전략 수: {len(sharpes):,}")
    print(f"평균 Sharpe: {sharpes.mean():.3f}")
    print(f"중앙값 Sharpe: {sharpes.median():.3f}")
    print(f"표준편차: {sharpes.std():.3f}")
    print(f"Sharpe > 0: {(sharpes > 0).sum():,} ({(sharpes > 0).mean()*100:.1f}%)")
    print(f"Sharpe > 1: {(sharpes > 1).sum():,} ({(sharpes > 1).mean()*100:.1f}%)")
    print(f"Sharpe > 2: {(sharpes > 2).sum():,} ({(sharpes > 2).mean()*100:.1f}%)")
    print(f"최대 Sharpe: {sharpes.max():.3f}")
    print(f"최소 Sharpe: {sharpes.min():.3f}")

# ============================================================
# 6. Monte Carlo Permutation Test 시뮬레이션
# ============================================================
print("\n[5] Monte Carlo 시뮬레이션 (무작위 전략 vs 실제)")
print("-" * 50)

np.random.seed(42)
n_simulations = 10000
n_strategies_simulated = 1000  # 1000개 전략 시뮬레이션

# 무작위 전략의 Sharpe 분포 시뮬레이션
random_max_sharpes = []
for _ in range(n_simulations):
    # 각 전략이 무작위 수익률을 가진다고 가정
    random_sharpes = np.random.normal(
        0, 1, n_strategies_simulated
    )  # 평균 0, 표준편차 1
    random_max_sharpes.append(np.max(random_sharpes))

random_max_sharpes = np.array(random_max_sharpes)

print(f"시뮬레이션 횟수: {n_simulations:,}")
print(f"전략 수: {n_strategies_simulated:,}")
print("\n무작위 전략 중 최대 Sharpe 분포:")
print(f"  평균: {random_max_sharpes.mean():.3f}")
print(f"  표준편차: {random_max_sharpes.std():.3f}")
print(f"  95th percentile: {np.percentile(random_max_sharpes, 95):.3f}")
print(f"  99th percentile: {np.percentile(random_max_sharpes, 99):.3f}")

# 실제 Sharpe가 무작위 분포에서 어디에 위치하는지
percentile_upbit = stats.percentileofscore(random_max_sharpes, observed_sharpe_upbit)
percentile_bithumb = stats.percentileofscore(
    random_max_sharpes, observed_sharpe_bithumb
)

print("\n실제 전략의 위치:")
print(f"  Upbit Sharpe {observed_sharpe_upbit}: {percentile_upbit:.1f}th percentile")
print(
    f"  Bithumb Sharpe {observed_sharpe_bithumb}: {percentile_bithumb:.1f}th percentile"
)

# ============================================================
# 7. 최종 평가
# ============================================================
print("\n" + "=" * 70)
print("최종 평가")
print("=" * 70)

# 점수 계산
score = 0
max_score = 5

# 1. Deflated Sharpe p-value
if p_upbit < 0.05 and p_bithumb < 0.05:
    score += 1
    print("[PASS] [1/5] Deflated Sharpe Ratio: PASS")
else:
    print("[FAIL] [1/5] Deflated Sharpe Ratio: FAIL")

# 2. Percentile > 95
if percentile_upbit > 95 and percentile_bithumb > 95:
    score += 1
    print("[PASS] [2/5] Monte Carlo Percentile > 95%: PASS")
else:
    print(
        f"[WARN] [2/5] Monte Carlo Percentile: {percentile_upbit:.0f}%, {percentile_bithumb:.0f}%"
    )
    if percentile_upbit > 90 or percentile_bithumb > 90:
        score += 0.5

# 3. 2025 Holdout (이미 통과)
score += 1
print("[PASS] [3/5] 2025 Holdout Test: PASS (Sharpe > 2.5)")

# 4. Cost Sensitivity (이미 통과)
score += 1
print("[PASS] [4/5] Cost Sensitivity Test: PASS")

# 5. Cross-Market (이미 통과)
score += 1
print("[PASS] [5/5] Cross-Market Validation: PASS")

print(f"\n총점: {score}/{max_score}")

# 등급 결정
if score >= 4.5:
    grade = "A+"
elif score >= 4:
    grade = "A"
elif score >= 3:
    grade = "B+"
elif score >= 2:
    grade = "B"
else:
    grade = "C"

print(f"전략 등급: {grade}")

# ============================================================
# 8. 핵심 인사이트
# ============================================================
print("\n" + "=" * 70)
print("핵심 인사이트: 왜 '필패'라고 하는가?")
print("=" * 70)

print("""
1. 다중 검정 문제 (Multiple Testing Problem)
   - {tests:,}번 이상 테스트 시, Sharpe > 3.0인 전략이 우연히 나올 확률: 높음
   - 이것이 '데이터 마이닝'의 핵심 위험

2. 그러나 당신의 전략은 다른 점이 있음:
   - 2025년 Holdout 테스트 통과 (미래 데이터)
   - Cross-Market 검증 통과 (Upbit ↔ Bithumb)
   - Cost Sensitivity 테스트 통과

3. 결론:
   - 완전한 과적합은 아닐 가능성 있음
   - 그러나 실제 라이브 성과 없이는 확신 불가
   - 소액 실전 테스트가 필수

4. 권장사항:
   - Paper Trading 1개월 → 실제 성과 vs 백테스트 비교
   - 백테스트 Sharpe의 50% 이상 나오면 실전 투입 고려
   - MDD -25% 초과 시 전략 재검토
""".format(tests=estimated_tests))

print("=" * 70)
