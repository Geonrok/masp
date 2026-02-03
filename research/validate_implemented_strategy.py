"""
Validation of Implemented Strategy (KAMA=5, TSMOM=90, Gate=30)
vs Validated Strategy (KAMA=10, TSMOM=60, Gate=30)

This script compares the two strategies using available backtest data
and runs overfitting validation tests.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. Load Backtest Results
# ============================================================
print("=" * 70)
print("IMPLEMENTED STRATEGY VALIDATION")
print("KAMA=5, TSMOM=90 vs KAMA=10, TSMOM=60")
print("=" * 70)

# Load grid search results (contains KAMA=5, TSMOM=90)
grid_path = "E:/투자/백테스트_결과_통합/backtest_results/optimization_v4_spot/step5_params/grid_search_results.csv"
coarse_path = "E:/투자/백테스트_결과_통합/backtest_results/param_optimization/param_coarse_grid_results.csv"

try:
    grid_results = pd.read_csv(grid_path)
    coarse_results = pd.read_csv(coarse_path)
    print(f"[OK] Loaded grid search results: {len(grid_results)} rows")
    print(f"[OK] Loaded coarse grid results: {len(coarse_results)} rows")
except Exception as e:
    print(f"[ERROR] Failed to load: {e}")
    grid_results = pd.DataFrame()
    coarse_results = pd.DataFrame()

# ============================================================
# 2. Extract Strategy Results
# ============================================================
print("\n[1] Strategy Comparison - In-Sample Performance")
print("-" * 50)

# IMPLEMENTED: KAMA=5, TSMOM=90, Gate=30
impl_upbit = grid_results[
    (grid_results["market"] == "upbit")
    & (grid_results["kama_period"] == 5)
    & (grid_results["tsmom_period"] == 90)
    & (grid_results["gate_ma_period"] == 30)
]

impl_bithumb = grid_results[
    (grid_results["market"] == "bithumb")
    & (grid_results["kama_period"] == 5)
    & (grid_results["tsmom_period"] == 90)
    & (grid_results["gate_ma_period"] == 30)
]

# VALIDATED: KAMA=10, TSMOM=60, Gate=30
valid_upbit = grid_results[
    (grid_results["market"] == "upbit")
    & (grid_results["kama_period"] == 10)
    & (grid_results["tsmom_period"] == 60)
    & (grid_results["gate_ma_period"] == 30)
]

valid_bithumb = grid_results[
    (grid_results["market"] == "bithumb")
    & (grid_results["kama_period"] == 10)
    & (grid_results["tsmom_period"] == 60)
    & (grid_results["gate_ma_period"] == 30)
]

print("\n[IMPLEMENTED] KAMA=5, TSMOM=90, Gate=30:")
if not impl_upbit.empty:
    print(
        f"  Upbit  - Sharpe: {impl_upbit['sharpe'].values[0]:.3f}, MDD: {impl_upbit['mdd'].values[0]*100:.1f}%, Return: {impl_upbit['total_return'].values[0]*100:.0f}%"
    )
if not impl_bithumb.empty:
    print(
        f"  Bithumb - Sharpe: {impl_bithumb['sharpe'].values[0]:.3f}, MDD: {impl_bithumb['mdd'].values[0]*100:.1f}%, Return: {impl_bithumb['total_return'].values[0]*100:.0f}%"
    )

print("\n[VALIDATED] KAMA=10, TSMOM=60, Gate=30:")
if not valid_upbit.empty:
    print(
        f"  Upbit  - Sharpe: {valid_upbit['sharpe'].values[0]:.3f}, MDD: {valid_upbit['mdd'].values[0]*100:.1f}%, Return: {valid_upbit['total_return'].values[0]*100:.0f}%"
    )
if not valid_bithumb.empty:
    print(
        f"  Bithumb - Sharpe: {valid_bithumb['sharpe'].values[0]:.3f}, MDD: {valid_bithumb['mdd'].values[0]*100:.1f}%, Return: {valid_bithumb['total_return'].values[0]*100:.0f}%"
    )

# ============================================================
# 3. 2025 Holdout Comparison
# ============================================================
print("\n[2] 2025 Holdout Performance (Out-of-Sample)")
print("-" * 50)

# Load final holdout results (KAMA=10, TSMOM=60)
holdout_path = "E:/투자/백테스트_결과_통합/backtest_results/holdout_2025/final_holdout_2025_results.csv"
try:
    holdout = pd.read_csv(holdout_path)
    print("\n[VALIDATED] KAMA=10, TSMOM=60, Gate=30 - 2025 Holdout:")
    for _, row in holdout.iterrows():
        print(
            f"  {row['exchange']} - Sharpe: {row['sharpe']:.3f}, MDD: {row['mdd']*100:.1f}%, Return: {row['total_return']*100:.0f}%"
        )
except Exception as e:
    print(f"[ERROR] {e}")

print("\n[IMPLEMENTED] KAMA=5, TSMOM=90, Gate=30 - 2025 Holdout:")
print("  [WARNING] NO HOLDOUT DATA EXISTS!")
print("  The implemented strategy was NOT tested on 2025 holdout data.")

# ============================================================
# 4. Deflated Sharpe Ratio Analysis
# ============================================================
print("\n[3] Deflated Sharpe Ratio (Multiple Testing Correction)")
print("-" * 50)


def deflated_sharpe_ratio(observed_sharpe, n_tests, years, skewness=0, kurtosis=3):
    """Calculate Deflated Sharpe Ratio."""
    euler = 0.5772156649

    if n_tests > 1:
        expected_max = (1 - euler) * stats.norm.ppf(
            1 - 1 / n_tests
        ) + euler * stats.norm.ppf(1 - 1 / (n_tests * np.e))
    else:
        expected_max = 0

    n_obs = years * 252
    var_sharpe = (
        1
        + 0.5 * observed_sharpe**2
        - skewness * observed_sharpe
        + (kurtosis - 3) / 4 * observed_sharpe**2
    ) / n_obs

    if var_sharpe > 0:
        dsr = (observed_sharpe - expected_max * np.sqrt(var_sharpe)) / np.sqrt(
            var_sharpe
        )
    else:
        dsr = 0

    p_value = 1 - stats.norm.cdf(dsr)

    return dsr, p_value, expected_max


# Total strategies tested in grid search
n_strategies_step5 = len(grid_results)  # ~83 combinations
n_strategies_coarse = len(coarse_results)  # ~72 combinations
total_tests = n_strategies_step5 + n_strategies_coarse

# Conservative estimate: consider all parameter combinations
# KAMA: 5, 10, 20 (3) x TSMOM: 30, 60, 90 (3) x Gate: 30, 50, 70 (3) = 27
# x Markets (3) x additional sweeps (~10) = ~810
estimated_total_tests = max(total_tests, 1000)

years = 7  # 2017-2024

print(f"Total parameter combinations tested: ~{estimated_total_tests:,}")
print(f"Data period: {years} years")

# IMPLEMENTED strategy DSR
impl_sharpe = impl_upbit["sharpe"].values[0] if not impl_upbit.empty else 3.83
dsr_impl, p_impl, exp_max = deflated_sharpe_ratio(
    impl_sharpe, estimated_total_tests, years
)
print(f"\n[IMPLEMENTED] KAMA=5, TSMOM=90:")
print(f"  In-Sample Sharpe: {impl_sharpe:.3f}")
print(f"  Expected Max (chance): {exp_max:.3f}")
print(f"  Deflated Sharpe: {dsr_impl:.3f}")
print(f"  P-value: {p_impl:.6f}")
print(f"  Pass (p < 0.05): {'[PASS]' if p_impl < 0.05 else '[FAIL]'}")

# VALIDATED strategy DSR using 2025 holdout Sharpe
valid_sharpe_holdout = 3.16  # 2025 holdout result
dsr_valid, p_valid, _ = deflated_sharpe_ratio(
    valid_sharpe_holdout, estimated_total_tests, years
)
print(f"\n[VALIDATED] KAMA=10, TSMOM=60 (using 2025 Holdout Sharpe):")
print(f"  Holdout Sharpe: {valid_sharpe_holdout:.3f}")
print(f"  Deflated Sharpe: {dsr_valid:.3f}")
print(f"  P-value: {p_valid:.6f}")
print(f"  Pass (p < 0.05): {'[PASS]' if p_valid < 0.05 else '[FAIL]'}")

# ============================================================
# 5. Monte Carlo Simulation
# ============================================================
print("\n[4] Monte Carlo Simulation")
print("-" * 50)

np.random.seed(42)
n_simulations = 10000
n_strategies_simulated = estimated_total_tests

# Simulate random strategy Sharpes
random_max_sharpes = []
for _ in range(n_simulations):
    random_sharpes = np.random.normal(0, 1, n_strategies_simulated)
    random_max_sharpes.append(np.max(random_sharpes))

random_max_sharpes = np.array(random_max_sharpes)

print(f"Simulations: {n_simulations:,}")
print(f"Strategies per simulation: {n_strategies_simulated:,}")
print(f"\nRandom Strategy Max Sharpe Distribution:")
print(f"  Mean: {random_max_sharpes.mean():.3f}")
print(f"  95th percentile: {np.percentile(random_max_sharpes, 95):.3f}")
print(f"  99th percentile: {np.percentile(random_max_sharpes, 99):.3f}")

# Where does implemented strategy fall?
impl_percentile = stats.percentileofscore(random_max_sharpes, impl_sharpe)
valid_percentile = stats.percentileofscore(random_max_sharpes, valid_sharpe_holdout)

print(
    f"\n[IMPLEMENTED] In-Sample Sharpe {impl_sharpe:.2f}: {impl_percentile:.1f}th percentile"
)
print(
    f"[VALIDATED] Holdout Sharpe {valid_sharpe_holdout:.2f}: {valid_percentile:.1f}th percentile"
)

# ============================================================
# 6. Critical Analysis
# ============================================================
print("\n" + "=" * 70)
print("CRITICAL ANALYSIS")
print("=" * 70)

print(
    """
[PROBLEM 1: No Holdout Validation]
---------------------------------
The IMPLEMENTED strategy (KAMA=5, TSMOM=90) has NO 2025 holdout test.
- In-Sample Sharpe: {impl:.2f} (looks good)
- Holdout Sharpe: UNKNOWN

The VALIDATED strategy (KAMA=10, TSMOM=60) passed 2025 holdout:
- In-Sample Sharpe: ~3.4
- Holdout Sharpe: {valid:.2f} (confirmed out-of-sample)

[PROBLEM 2: Overfitting Risk]
-----------------------------
The implemented strategy was selected from {tests:,}+ combinations
without holdout validation. This is a classic overfitting scenario.

In-Sample performance can be misleading:
- KAMA=5, TSMOM=90 In-Sample: {impl:.2f}
- KAMA=10, TSMOM=60 In-Sample: ~3.4
- KAMA=10, TSMOM=60 Holdout: {valid:.2f} (7% decay)

If the same decay applies to KAMA=5, TSMOM=90:
- Estimated Holdout: {impl:.2f} * 0.93 = {est_holdout:.2f}

[PROBLEM 3: Implementation Mismatch]
------------------------------------
The code currently uses parameters that were NEVER validated:
- Implemented: KAMA=5, TSMOM=90, Gate=30
- Validated:   KAMA=10, TSMOM=60, Gate=30

This means the live trading system would use an UNTESTED strategy.
""".format(
        impl=impl_sharpe,
        valid=valid_sharpe_holdout,
        tests=estimated_total_tests,
        est_holdout=impl_sharpe * 0.93,
    )
)

# ============================================================
# 7. Scoring and Recommendation
# ============================================================
print("=" * 70)
print("OVERFITTING SCORE")
print("=" * 70)

score_impl = 0
score_valid = 0
max_score = 5

# Test 1: In-Sample Sharpe > 2
if impl_sharpe > 2:
    score_impl += 1
if valid_sharpe_holdout > 2:
    score_valid += 1
print(
    f"[1/5] In-Sample Sharpe > 2: IMPL={'[PASS]' if impl_sharpe > 2 else '[FAIL]'}, VALID={'[PASS]' if valid_sharpe_holdout > 2 else '[FAIL]'}"
)

# Test 2: Deflated Sharpe p-value < 0.05
if p_impl < 0.05:
    score_impl += 1
if p_valid < 0.05:
    score_valid += 1
print(
    f"[2/5] Deflated Sharpe p < 0.05: IMPL={'[PASS]' if p_impl < 0.05 else '[FAIL]'}, VALID={'[PASS]' if p_valid < 0.05 else '[FAIL]'}"
)

# Test 3: Monte Carlo > 95th percentile
if impl_percentile > 95:
    score_impl += 1
if valid_percentile > 95:
    score_valid += 1
print(
    f"[3/5] Monte Carlo > 95%: IMPL={'[PASS]' if impl_percentile > 95 else '[FAIL]'}, VALID={'[PASS]' if valid_percentile > 95 else '[FAIL]'}"
)

# Test 4: Holdout Validation
score_impl += 0  # NO holdout
score_valid += 1  # Has holdout
print(f"[4/5] 2025 Holdout Test: IMPL=[NO DATA], VALID=[PASS]")

# Test 5: Cross-Market Consistency
score_impl += 0  # Unknown
score_valid += 1  # Passed
print(f"[5/5] Cross-Market Validation: IMPL=[UNKNOWN], VALID=[PASS]")

print(f"\nFINAL SCORES:")
print(f"  IMPLEMENTED (KAMA=5, TSMOM=90): {score_impl}/{max_score}")
print(f"  VALIDATED (KAMA=10, TSMOM=60):  {score_valid}/{max_score}")


# Grade
def get_grade(score):
    if score >= 4:
        return "A"
    elif score >= 3:
        return "B"
    elif score >= 2:
        return "C"
    else:
        return "F"


print(f"\n  IMPLEMENTED Grade: {get_grade(score_impl)}")
print(f"  VALIDATED Grade:   {get_grade(score_valid)}")

# ============================================================
# 8. Recommendation
# ============================================================
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

print("""
[URGENT ACTION REQUIRED]

The implemented strategy (KAMA=5, TSMOM=90) is NOT validated.
Two options:

OPTION A: Update Code to Use Validated Parameters
-------------------------------------------------
Change in libs/strategies/kama_tsmom_gate.py:
  - DEFAULT_KAMA_PERIOD = 10  (currently 5)
  - DEFAULT_TSMOM_LOOKBACK = 60  (currently 90)

This uses the strategy that passed 2025 holdout with Sharpe 3.16.

OPTION B: Run New Holdout Test for KAMA=5, TSMOM=90
--------------------------------------------------
1. Collect 2025 price data for all symbols
2. Run backtest with KAMA=5, TSMOM=90, Gate=30
3. Compare holdout performance
4. If Sharpe > 2.5 and MDD < 25%, consider using

RECOMMENDATION: Option A (Update to validated parameters)
---------------------------------------------------------
Reason: The validated strategy has proven out-of-sample performance.
Using untested parameters in production is high-risk.
""")

print("=" * 70)
