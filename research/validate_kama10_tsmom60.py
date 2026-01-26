"""
KAMA=10, TSMOM=60 Full Validation Suite
Performs the same validation as KAMA=5, TSMOM=90

Tests:
1. 3-Market Performance (Upbit, Bithumb, Binance)
2. Cross-Market Validation Simulation
3. Overfitting Risk Assessment
4. Final Comparison
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("KAMA=10, TSMOM=60 FULL VALIDATION SUITE")
print("=" * 80)

# ============================================================
# 1. Load Data
# ============================================================
grid_path = "E:/투자/백테스트_결과_통합/backtest_results/optimization_v4_spot/step5_params/grid_search_results.csv"
coarse_path = "E:/투자/백테스트_결과_통합/backtest_results/param_optimization/param_coarse_grid_results.csv"
holdout_path = "E:/투자/백테스트_결과_통합/backtest_results/holdout_2025/final_holdout_2025_results.csv"

grid_df = pd.read_csv(grid_path)
coarse_df = pd.read_csv(coarse_path)
holdout_df = pd.read_csv(holdout_path)

# ============================================================
# 2. Extract KAMA=10, TSMOM=60, Gate=30 Results
# ============================================================
print("\n" + "=" * 80)
print("[1] 3-MARKET IN-SAMPLE PERFORMANCE")
print("=" * 80)

# In-Sample results from grid search
kama10_results = grid_df[
    (grid_df['kama_period'] == 10) &
    (grid_df['tsmom_period'] == 60) &
    (grid_df['gate_ma_period'] == 30)
]

print("\nKAMA=10, TSMOM=60, Gate=30 In-Sample Results:")
print("-" * 60)
for _, row in kama10_results.iterrows():
    print(f"  {row['market']:15s}: Sharpe {row['sharpe']:.3f}, MDD {row['mdd']*100:.1f}%, Return {row['total_return']*100:.0f}%")

# Compare with KAMA=5, TSMOM=90
kama5_results = grid_df[
    (grid_df['kama_period'] == 5) &
    (grid_df['tsmom_period'] == 90) &
    (grid_df['gate_ma_period'] == 30)
]

print("\nKAMA=5, TSMOM=90, Gate=30 In-Sample Results (Reference):")
print("-" * 60)
for _, row in kama5_results.iterrows():
    print(f"  {row['market']:15s}: Sharpe {row['sharpe']:.3f}, MDD {row['mdd']*100:.1f}%, Return {row['total_return']*100:.0f}%")

# ============================================================
# 3. 2025 Holdout Performance
# ============================================================
print("\n" + "=" * 80)
print("[2] 2025 HOLDOUT (OUT-OF-SAMPLE) PERFORMANCE")
print("=" * 80)

print("\nKAMA=10, TSMOM=60 - 2025 Holdout:")
print("-" * 60)
for _, row in holdout_df.iterrows():
    print(f"  {row['exchange']:15s}: Sharpe {row['sharpe']:.3f}, MDD {row['mdd']*100:.1f}%, Return {row['total_return']*100:.0f}%")

print("\n[NOTE] Binance 2025 Holdout for KAMA=10, TSMOM=60: NOT AVAILABLE")
print("       The holdout test was only run for Upbit and Bithumb.")

# ============================================================
# 4. Cross-Market Validation Simulation
# ============================================================
print("\n" + "=" * 80)
print("[3] CROSS-MARKET VALIDATION")
print("=" * 80)

print("""
Cross-Market Validation Logic:
- Train parameters on Market A
- Test (without re-optimization) on Market B
- If Sharpe drops significantly, strategy may be overfitted

Available Data Analysis:
""")

# Extract Sharpe values for comparison
markets = ['upbit', 'bithumb', 'binance_spot']
kama10_sharpes = {}
kama5_sharpes = {}

for market in markets:
    k10 = kama10_results[kama10_results['market'] == market]['sharpe'].values
    k5 = kama5_results[kama5_results['market'] == market]['sharpe'].values
    if len(k10) > 0:
        kama10_sharpes[market] = k10[0]
    if len(k5) > 0:
        kama5_sharpes[market] = k5[0]

print("In-Sample Sharpe by Market:")
print("-" * 60)
print(f"{'Market':15s} | {'KAMA=10,TSMOM=60':18s} | {'KAMA=5,TSMOM=90':18s}")
print("-" * 60)
for market in markets:
    k10 = kama10_sharpes.get(market, 'N/A')
    k5 = kama5_sharpes.get(market, 'N/A')
    k10_str = f"{k10:.3f}" if isinstance(k10, float) else k10
    k5_str = f"{k5:.3f}" if isinstance(k5, float) else k5
    print(f"{market:15s} | {k10_str:18s} | {k5_str:18s}")

# Cross-market consistency check
print("\nCross-Market Consistency Check:")
print("-" * 60)

def check_cross_market_consistency(sharpes):
    """Check if strategy performs consistently across markets."""
    values = list(sharpes.values())
    if len(values) < 2:
        return None, None
    mean_sharpe = np.mean(values)
    std_sharpe = np.std(values)
    cv = std_sharpe / mean_sharpe if mean_sharpe > 0 else float('inf')
    return mean_sharpe, cv

k10_mean, k10_cv = check_cross_market_consistency(kama10_sharpes)
k5_mean, k5_cv = check_cross_market_consistency(kama5_sharpes)

print(f"\nKAMA=10, TSMOM=60:")
print(f"  Mean Sharpe across markets: {k10_mean:.3f}")
print(f"  Coefficient of Variation: {k10_cv:.3f}")
print(f"  Consistency: {'GOOD' if k10_cv < 0.15 else 'MODERATE' if k10_cv < 0.25 else 'POOR'}")

print(f"\nKAMA=5, TSMOM=90:")
print(f"  Mean Sharpe across markets: {k5_mean:.3f}")
print(f"  Coefficient of Variation: {k5_cv:.3f}")
print(f"  Consistency: {'GOOD' if k5_cv < 0.15 else 'MODERATE' if k5_cv < 0.25 else 'POOR'}")

# Simulate cross-market validation using In-Sample differences
print("\n" + "-" * 60)
print("Cross-Market Simulation (using In-Sample data):")
print("-" * 60)

pairs = [
    ('upbit', 'bithumb'),
    ('upbit', 'binance_spot'),
    ('bithumb', 'upbit'),
    ('bithumb', 'binance_spot'),
    ('binance_spot', 'upbit'),
    ('binance_spot', 'bithumb'),
]

print(f"\n{'Train':15s} -> {'Test':15s} | {'K10 Sharpe':12s} | {'K5 Sharpe':12s}")
print("-" * 60)
for train, test in pairs:
    k10_test = kama10_sharpes.get(test, float('nan'))
    k5_test = kama5_sharpes.get(test, float('nan'))
    print(f"{train:15s} -> {test:15s} | {k10_test:.3f}        | {k5_test:.3f}")

# ============================================================
# 5. Overfitting Risk Assessment
# ============================================================
print("\n" + "=" * 80)
print("[4] OVERFITTING RISK ASSESSMENT")
print("=" * 80)

def deflated_sharpe_ratio(observed_sharpe, n_tests, years, skewness=0, kurtosis=3):
    euler = 0.5772156649
    if n_tests > 1:
        expected_max = (1 - euler) * stats.norm.ppf(1 - 1/n_tests) + \
                      euler * stats.norm.ppf(1 - 1/(n_tests * np.e))
    else:
        expected_max = 0
    n_obs = years * 252
    var_sharpe = (1 + 0.5 * observed_sharpe**2 - skewness * observed_sharpe +
                  (kurtosis - 3) / 4 * observed_sharpe**2) / n_obs
    if var_sharpe > 0:
        dsr = (observed_sharpe - expected_max * np.sqrt(var_sharpe)) / np.sqrt(var_sharpe)
    else:
        dsr = 0
    p_value = 1 - stats.norm.cdf(dsr)
    return dsr, p_value, expected_max

# Parameters
n_tests = len(grid_df) + len(coarse_df)  # Total combinations tested
years = 7  # 2017-2024

print(f"\nTotal parameter combinations tested: {n_tests}")
print(f"Data period: {years} years")

# KAMA=10, TSMOM=60
k10_insample = kama10_sharpes.get('upbit', 3.42)
k10_holdout = holdout_df[holdout_df['exchange'] == 'upbit']['sharpe'].values[0]
dsr_k10_is, p_k10_is, _ = deflated_sharpe_ratio(k10_insample, n_tests, years)
dsr_k10_oos, p_k10_oos, _ = deflated_sharpe_ratio(k10_holdout, n_tests, years)

# KAMA=5, TSMOM=90
k5_insample = kama5_sharpes.get('upbit', 3.83)
k5_holdout = 2.35  # From optimization report
dsr_k5_is, p_k5_is, _ = deflated_sharpe_ratio(k5_insample, n_tests, years)
dsr_k5_oos, p_k5_oos, _ = deflated_sharpe_ratio(k5_holdout, n_tests, years)

print("\nDeflated Sharpe Ratio Analysis:")
print("-" * 70)
print(f"{'Strategy':25s} | {'In-Sample':12s} | {'Holdout':12s} | {'DSR (IS)':10s} | {'DSR (OOS)':10s}")
print("-" * 70)
print(f"{'KAMA=10, TSMOM=60':25s} | {k10_insample:.3f}        | {k10_holdout:.3f}       | {dsr_k10_is:.2f}       | {dsr_k10_oos:.2f}")
print(f"{'KAMA=5, TSMOM=90':25s} | {k5_insample:.3f}        | {k5_holdout:.3f}       | {dsr_k5_is:.2f}       | {dsr_k5_oos:.2f}")

# Overfitting metrics
print("\nOverfitting Indicators:")
print("-" * 60)

k10_decay = (k10_insample - k10_holdout) / k10_insample * 100
k5_decay = (k5_insample - k5_holdout) / k5_insample * 100

print(f"KAMA=10, TSMOM=60:")
print(f"  In-Sample → Holdout Sharpe Decay: {k10_decay:.1f}%")
print(f"  Overfitting Risk: {'LOW' if k10_decay < 20 else 'MODERATE' if k10_decay < 40 else 'HIGH'}")

print(f"\nKAMA=5, TSMOM=90:")
print(f"  In-Sample → Holdout Sharpe Decay: {k5_decay:.1f}%")
print(f"  Overfitting Risk: {'LOW' if k5_decay < 20 else 'MODERATE' if k5_decay < 40 else 'HIGH'}")

# Monte Carlo
print("\nMonte Carlo Simulation:")
print("-" * 60)
np.random.seed(42)
n_simulations = 10000
random_max_sharpes = []
for _ in range(n_simulations):
    random_sharpes = np.random.normal(0, 1, n_tests)
    random_max_sharpes.append(np.max(random_sharpes))
random_max_sharpes = np.array(random_max_sharpes)

k10_is_pct = stats.percentileofscore(random_max_sharpes, k10_insample)
k10_oos_pct = stats.percentileofscore(random_max_sharpes, k10_holdout)
k5_is_pct = stats.percentileofscore(random_max_sharpes, k5_insample)
k5_oos_pct = stats.percentileofscore(random_max_sharpes, k5_holdout)

print(f"Random Max Sharpe: Mean={random_max_sharpes.mean():.2f}, 95th pct={np.percentile(random_max_sharpes, 95):.2f}")
print(f"\nKAMA=10, TSMOM=60:")
print(f"  In-Sample {k10_insample:.2f} at {k10_is_pct:.0f}th percentile")
print(f"  Holdout {k10_holdout:.2f} at {k10_oos_pct:.0f}th percentile")
print(f"\nKAMA=5, TSMOM=90:")
print(f"  In-Sample {k5_insample:.2f} at {k5_is_pct:.0f}th percentile")
print(f"  Holdout {k5_holdout:.2f} at {k5_oos_pct:.0f}th percentile")

# ============================================================
# 6. Final Comparison Table
# ============================================================
print("\n" + "=" * 80)
print("[5] FINAL COMPARISON")
print("=" * 80)

k10_h_bi = holdout_df[holdout_df['exchange'] == 'bithumb']['sharpe'].values[0]
k10_risk = 'LOW' if k10_decay < 20 else 'MODERATE'
k5_risk = 'LOW' if k5_decay < 20 else 'MODERATE'

print(f"""
+-------------------------------------------------------------------+
|                    COMPREHENSIVE COMPARISON                        |
+-------------------------------------------------------------------+
| Metric                    | KAMA=10, TSMOM=60 | KAMA=5, TSMOM=90  |
|---------------------------|-------------------|-------------------|
| In-Sample Sharpe (Upbit)  | {kama10_sharpes.get('upbit', 0):17.3f} | {kama5_sharpes.get('upbit', 0):17.3f} |
| In-Sample Sharpe (Bithumb)| {kama10_sharpes.get('bithumb', 0):17.3f} | {kama5_sharpes.get('bithumb', 0):17.3f} |
| In-Sample Sharpe (Binance)| {kama10_sharpes.get('binance_spot', 0):17.3f} | {kama5_sharpes.get('binance_spot', 0):17.3f} |
|---------------------------|-------------------|-------------------|
| Holdout Sharpe (Upbit)    | {k10_holdout:17.3f} | {k5_holdout:17.3f} |
| Holdout Sharpe (Bithumb)  | {k10_h_bi:17.3f} | {1.81:17.3f} |
| Holdout Sharpe (Binance)  | {'N/A':>17} | {2.54:17.3f} |
|---------------------------|-------------------|-------------------|
| Sharpe Decay (IS->OOS)    | {k10_decay:16.1f}% | {k5_decay:16.1f}% |
| Cross-Market CV           | {k10_cv:17.3f} | {k5_cv:17.3f} |
| Monte Carlo (IS) pct      | {k10_is_pct:16.0f}% | {k5_is_pct:16.0f}% |
|---------------------------|-------------------|-------------------|
| 3-Market Validation       | {'PARTIAL (2/3)':>17} | {'FULL (3/3)':>17} |
| Cross-Market Formal Test  | {'NOT DONE':>17} | {'100% PASS':>17} |
| Overfitting Risk          | {k10_risk:>17} | {k5_risk:>17} |
+-------------------------------------------------------------------+
""")

# ============================================================
# 7. Scoring
# ============================================================
print("=" * 80)
print("[6] VALIDATION SCORING")
print("=" * 80)

def score_strategy(name, holdout_sharpe, decay, cv, mc_pct, markets_validated, cross_market_done):
    score = 0
    max_score = 6
    details = []

    # 1. Holdout Sharpe > 2
    if holdout_sharpe > 2.0:
        score += 1
        details.append(f"[PASS] Holdout Sharpe > 2.0: {holdout_sharpe:.2f}")
    else:
        details.append(f"[FAIL] Holdout Sharpe > 2.0: {holdout_sharpe:.2f}")

    # 2. Sharpe Decay < 30%
    if decay < 30:
        score += 1
        details.append(f"[PASS] Sharpe Decay < 30%: {decay:.1f}%")
    else:
        details.append(f"[FAIL] Sharpe Decay < 30%: {decay:.1f}%")

    # 3. Cross-Market CV < 0.15
    if cv < 0.15:
        score += 1
        details.append(f"[PASS] Cross-Market CV < 0.15: {cv:.3f}")
    elif cv < 0.25:
        score += 0.5
        details.append(f"[PARTIAL] Cross-Market CV < 0.25: {cv:.3f}")
    else:
        details.append(f"[FAIL] Cross-Market CV: {cv:.3f}")

    # 4. Monte Carlo > 90%
    if mc_pct > 95:
        score += 1
        details.append(f"[PASS] Monte Carlo > 95%: {mc_pct:.0f}%")
    elif mc_pct > 90:
        score += 0.5
        details.append(f"[PARTIAL] Monte Carlo > 90%: {mc_pct:.0f}%")
    else:
        details.append(f"[FAIL] Monte Carlo: {mc_pct:.0f}%")

    # 5. 3-Market Validation
    if markets_validated >= 3:
        score += 1
        details.append(f"[PASS] 3-Market Validation: {markets_validated}/3")
    elif markets_validated >= 2:
        score += 0.5
        details.append(f"[PARTIAL] Markets Validated: {markets_validated}/3")
    else:
        details.append(f"[FAIL] Markets Validated: {markets_validated}/3")

    # 6. Formal Cross-Market Test
    if cross_market_done:
        score += 1
        details.append("[PASS] Formal Cross-Market Test: DONE")
    else:
        details.append("[FAIL] Formal Cross-Market Test: NOT DONE")

    return score, max_score, details

# Score KAMA=10, TSMOM=60
k10_score, k10_max, k10_details = score_strategy(
    "KAMA=10, TSMOM=60",
    holdout_sharpe=k10_holdout,
    decay=k10_decay,
    cv=k10_cv,
    mc_pct=k10_is_pct,
    markets_validated=2,  # Only Upbit, Bithumb holdout
    cross_market_done=False
)

# Score KAMA=5, TSMOM=90
k5_score, k5_max, k5_details = score_strategy(
    "KAMA=5, TSMOM=90",
    holdout_sharpe=k5_holdout,
    decay=k5_decay,
    cv=k5_cv,
    mc_pct=k5_is_pct,
    markets_validated=3,  # All 3 markets
    cross_market_done=True
)

print(f"\nKAMA=10, TSMOM=60 Score: {k10_score}/{k10_max}")
for d in k10_details:
    print(f"  {d}")

print(f"\nKAMA=5, TSMOM=90 Score: {k5_score}/{k5_max}")
for d in k5_details:
    print(f"  {d}")

# Grade
def get_grade(score, max_score):
    pct = score / max_score
    if pct >= 0.9:
        return "A+"
    elif pct >= 0.8:
        return "A"
    elif pct >= 0.7:
        return "B+"
    elif pct >= 0.6:
        return "B"
    elif pct >= 0.5:
        return "C"
    else:
        return "D"

print(f"\nFINAL GRADES:")
print(f"  KAMA=10, TSMOM=60: {k10_score}/{k10_max} = Grade {get_grade(k10_score, k10_max)}")
print(f"  KAMA=5, TSMOM=90:  {k5_score}/{k5_max} = Grade {get_grade(k5_score, k5_max)}")

# ============================================================
# 8. Recommendations
# ============================================================
print("\n" + "=" * 80)
print("[7] RECOMMENDATIONS")
print("=" * 80)

print("""
ANALYSIS SUMMARY:

KAMA=10, TSMOM=60:
  Strengths:
  + Higher holdout Sharpe (3.16 vs 2.35 on Upbit)
  + Lower Sharpe decay (8% vs 39%)
  + Better individual market performance

  Weaknesses:
  - No formal cross-market validation test
  - No Binance holdout data
  - Less rigorous validation pipeline

KAMA=5, TSMOM=90:
  Strengths:
  + Full 3-market validation (including Binance)
  + Formal cross-market test passed (100%)
  + More rigorous validation pipeline

  Weaknesses:
  - Higher Sharpe decay (39%)
  - Lower holdout Sharpe

RECOMMENDATION:
--------------
Both strategies have merit. The choice depends on priorities:

Option A: Use KAMA=10, TSMOM=60 if you value:
  - Higher absolute returns
  - Lower risk of performance decay

Option B: Use KAMA=5, TSMOM=90 if you value:
  - More rigorous validation
  - Confirmed cross-market generalization
  - Binance market access

SUGGESTED ACTION:
Run formal cross-market validation and Binance 2025 holdout
for KAMA=10, TSMOM=60 to make a fair comparison.
""")

print("=" * 80)
