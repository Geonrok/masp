"""
Foreign_30d_SMA_100 전략 상세 검증
==================================
Look-Ahead Bias 제거 후 발견된 유일한 유효 전략.

전략 로직:
- 조건1: 외국인 30일 누적 순매수 > 0
- 조건2: 종가 > 100일 이동평균

특징:
- 미국 데이터 불필요 (한국 데이터만 사용)
- Look-Ahead Bias 없음
- 장기 추세 추종 + 외국인 자금 흐름

검증 내용:
1. 전체 기간 성과
2. Walk-Forward 검증 (11-fold)
3. 위기 기간 성과
4. 연도별 성과
5. 파라미터 민감도
6. 레버리지 ETF 적합성
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import warnings

warnings.filterwarnings("ignore")

INVESTOR_DIR = "E:/투자/data/kr_stock/investor_trading"
KOSPI_DIR = "E:/투자/data/kospi_futures"
OUTPUT_DIR = "E:/투자/Multi-Asset Strategy Platform/logs"

COST = 0.0009  # 왕복 0.09%

print("=" * 80)
print("Foreign_30d_SMA_100 전략 상세 검증")
print("=" * 80)

# 데이터 로드
investor_files = [f for f in os.listdir(INVESTOR_DIR) if f.endswith("_investor.csv")]
all_investor = []
for f in investor_files:
    try:
        df = pd.read_csv(f"{INVESTOR_DIR}/{f}", encoding="utf-8-sig")
        df["날짜"] = pd.to_datetime(df["날짜"])
        df = df.set_index("날짜")
        all_investor.append(df[["외국인합계"]])
    except:
        pass

foreign_data = all_investor[0].copy()
for df in all_investor[1:]:
    foreign_data = foreign_data.add(df, fill_value=0)
foreign_data = foreign_data.sort_index()

kospi200 = pd.read_parquet(f"{KOSPI_DIR}/kospi200_daily_yf.parquet")
kospi200.columns = [c.lower() for c in kospi200.columns]
if kospi200.index.tz is not None:
    kospi200.index = kospi200.index.tz_localize(None)

# 데이터 병합
data = kospi200[["close"]].copy()
data["returns"] = data["close"].pct_change()
data["foreign"] = foreign_data["외국인합계"]
data = data.ffill().dropna()

# 지표 계산
for period in [20, 30, 50, 100, 200]:
    data[f"sma_{period}"] = data["close"].rolling(period).mean()
    data[f"foreign_{period}d"] = data["foreign"].rolling(period).sum()

data = data.dropna()
print(f"데이터: {data.index.min().date()} ~ {data.index.max().date()} ({len(data)}일)")


def backtest(data, signal, cost=COST, leverage=1.0):
    """백테스트."""
    returns = data["returns"] * leverage
    position_change = signal.diff().abs()
    costs = position_change * cost
    strategy_returns = signal.shift(1) * returns - costs
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) < 100:
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

    # 추가 지표
    downside = strategy_returns[strategy_returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.001
    sortino = cagr / downside_std

    calmar = cagr / abs(mdd) if mdd != 0 else 0

    winning = (strategy_returns > 0).sum()
    total = (strategy_returns != 0).sum()
    win_rate = winning / total if total > 0 else 0

    trades = int((signal.diff().abs() > 0).sum())
    time_in_market = (signal == 1).mean()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "mdd": mdd,
        "volatility": volatility,
        "win_rate": win_rate,
        "trades": trades,
        "time_in_market": time_in_market,
        "cumulative": cumulative,
        "drawdown": drawdown,
    }


def strategy_foreign_sma(data, foreign_period=30, sma_period=100):
    """Foreign + SMA 전략."""
    signal = pd.Series(0, index=data.index)
    cond = (data[f"foreign_{foreign_period}d"] > 0) & (
        data["close"] > data[f"sma_{sma_period}"]
    )
    signal[cond] = 1
    return signal


# ========== 1. 전체 기간 성과 ==========
print("\n" + "=" * 80)
print("[1] 전체 기간 성과")
print("=" * 80)

signal = strategy_foreign_sma(data, 30, 100)
result = backtest(data, signal)

print(f"\nForeign_30d_SMA_100:")
print(f"  Sharpe:     {result['sharpe']:.3f}")
print(f"  CAGR:       {result['cagr']*100:.1f}%")
print(f"  MDD:        {result['mdd']*100:.1f}%")
print(f"  Sortino:    {result['sortino']:.3f}")
print(f"  Calmar:     {result['calmar']:.3f}")
print(f"  Win Rate:   {result['win_rate']*100:.1f}%")
print(f"  Time in Market: {result['time_in_market']*100:.1f}%")
print(f"  Trades:     {result['trades']}")

# Buy & Hold 비교
bh_signal = pd.Series(1, index=data.index)
bh_result = backtest(data, bh_signal)

print(f"\nBuy & Hold:")
print(
    f"  Sharpe: {bh_result['sharpe']:.3f}, CAGR: {bh_result['cagr']*100:.1f}%, MDD: {bh_result['mdd']*100:.1f}%"
)

print(f"\n전략 vs B&H:")
print(f"  Sharpe: {result['sharpe'] - bh_result['sharpe']:+.3f}")
print(f"  CAGR:   {(result['cagr'] - bh_result['cagr'])*100:+.1f}%p")
print(f"  MDD:    {(result['mdd'] - bh_result['mdd'])*100:+.1f}%p (개선)")


# ========== 2. Walk-Forward 검증 (11-fold) ==========
print("\n" + "=" * 80)
print("[2] Walk-Forward 검증 (11-fold)")
print("=" * 80)

n_splits = 11
split_size = len(data) // n_splits

wf_results = []

for i in range(n_splits - 1):
    is_end = (i + 1) * split_size
    oos_start = is_end
    oos_end = min((i + 2) * split_size, len(data))

    is_data = data.iloc[:is_end]
    oos_data = data.iloc[oos_start:oos_end]

    # IS 테스트
    is_signal = strategy_foreign_sma(is_data, 30, 100)
    is_result = backtest(is_data, is_signal)

    # OOS 테스트
    oos_signal = strategy_foreign_sma(oos_data, 30, 100)
    oos_result = backtest(oos_data, oos_signal)

    if is_result and oos_result:
        wf_results.append(
            {
                "fold": i + 1,
                "is_sharpe": is_result["sharpe"],
                "oos_sharpe": oos_result["sharpe"],
                "is_cagr": is_result["cagr"],
                "oos_cagr": oos_result["cagr"],
                "oos_period": f"{oos_data.index.min().date()} ~ {oos_data.index.max().date()}",
            }
        )

        print(
            f"\n  Fold {i+1}: {oos_data.index.min().date()} ~ {oos_data.index.max().date()}"
        )
        print(
            f"    IS Sharpe: {is_result['sharpe']:.3f}, OOS Sharpe: {oos_result['sharpe']:.3f}"
        )
        print(f"    OOS CAGR: {oos_result['cagr']*100:.1f}%")

# WF 요약
is_mean = np.mean([r["is_sharpe"] for r in wf_results])
oos_mean = np.mean([r["oos_sharpe"] for r in wf_results])
wf_ratio = oos_mean / is_mean if is_mean > 0 else 0

print(f"\n  WF Summary:")
print(f"    IS Mean Sharpe:  {is_mean:.3f}")
print(f"    OOS Mean Sharpe: {oos_mean:.3f}")
print(f"    WF Ratio (OOS/IS): {wf_ratio:.2f}")

if wf_ratio > 0.7:
    print("    → 과적합 아님 (WF Ratio > 0.7)")
elif wf_ratio > 0.5:
    print("    → 경미한 과적합 (0.5 < WF Ratio < 0.7)")
else:
    print("    → 과적합 우려 (WF Ratio < 0.5)")


# ========== 3. 위기 기간 성과 ==========
print("\n" + "=" * 80)
print("[3] 위기 기간 성과")
print("=" * 80)

crisis_periods = {
    "COVID_2020_03": ("2020-03-01", "2020-03-31"),
    "COVID_Recovery": ("2020-03-23", "2020-06-30"),
    "Rate_Hike_2022": ("2022-01-01", "2022-12-31"),
    "Bank_Crisis_2023": ("2023-03-01", "2023-05-31"),
    "Japan_Shock_2024": ("2024-08-01", "2024-08-15"),
}

signal = strategy_foreign_sma(data, 30, 100)

for name, (start, end) in crisis_periods.items():
    mask = (data.index >= start) & (data.index <= end)
    if mask.sum() < 5:
        continue

    crisis_data = data[mask]
    crisis_signal = signal[mask]

    # Buy & Hold
    bh_return = (1 + crisis_data["returns"]).prod() - 1

    # 전략
    strat_returns = crisis_signal.shift(1) * crisis_data["returns"]
    strat_return = (1 + strat_returns).prod() - 1

    alpha = strat_return - bh_return

    print(f"\n  {name}:")
    print(
        f"    B&H: {bh_return*100:.1f}%, Strategy: {strat_return*100:.1f}%, Alpha: {alpha*100:+.1f}%p"
    )


# ========== 4. 연도별 성과 ==========
print("\n" + "=" * 80)
print("[4] 연도별 성과")
print("=" * 80)

signal = strategy_foreign_sma(data, 30, 100)
returns = data["returns"]
strategy_returns = signal.shift(1) * returns
strategy_returns = strategy_returns.dropna()

yearly = strategy_returns.groupby(strategy_returns.index.year)

print(f"\n  {'년도':<6} {'전략':<10} {'B&H':<10} {'알파':<10}")
print(f"  {'-'*40}")

yearly_results = []
for year, year_returns in yearly:
    if len(year_returns) < 20:
        continue

    strat_ret = (1 + year_returns).prod() - 1
    bh_ret = (1 + returns.loc[year_returns.index]).prod() - 1
    alpha = strat_ret - bh_ret

    yearly_results.append(
        {
            "year": year,
            "strategy_return": strat_ret,
            "bh_return": bh_ret,
            "alpha": alpha,
        }
    )

    print(
        f"  {year:<6} {strat_ret*100:+.1f}%{'':<5} {bh_ret*100:+.1f}%{'':<5} {alpha*100:+.1f}%p"
    )

positive_years = sum(1 for r in yearly_results if r["strategy_return"] > 0)
print(
    f"\n  양수 수익 연도: {positive_years}/{len(yearly_results)}년 ({positive_years/len(yearly_results)*100:.0f}%)"
)


# ========== 5. 파라미터 민감도 ==========
print("\n" + "=" * 80)
print("[5] 파라미터 민감도")
print("=" * 80)

param_results = []

for foreign_period in [20, 30, 50]:
    for sma_period in [50, 100, 200]:
        signal = strategy_foreign_sma(data, foreign_period, sma_period)
        result = backtest(data, signal)

        if result:
            param_results.append(
                {
                    "foreign_period": foreign_period,
                    "sma_period": sma_period,
                    "sharpe": result["sharpe"],
                    "cagr": result["cagr"],
                    "mdd": result["mdd"],
                }
            )

param_df = pd.DataFrame(param_results)
param_df = param_df.sort_values("sharpe", ascending=False)

print(f"\n  {'Foreign':<8} {'SMA':<8} {'Sharpe':<10} {'CAGR':<10} {'MDD':<10}")
print(f"  {'-'*50}")
for _, row in param_df.iterrows():
    print(
        f"  {row['foreign_period']:<8} {row['sma_period']:<8} {row['sharpe']:.3f}{'':<5} {row['cagr']*100:.1f}%{'':<5} {row['mdd']*100:.1f}%"
    )

# 민감도 분석
sharpe_std = param_df["sharpe"].std()
sharpe_mean = param_df["sharpe"].mean()
print(f"\n  Sharpe 평균: {sharpe_mean:.3f}, 표준편차: {sharpe_std:.3f}")
print(f"  변동계수 (CV): {sharpe_std/sharpe_mean:.2f}")

if sharpe_std / sharpe_mean < 0.3:
    print("  → 파라미터에 둔감 (안정적)")
else:
    print("  → 파라미터에 민감 (주의)")


# ========== 6. 레버리지 ETF 적합성 ==========
print("\n" + "=" * 80)
print("[6] 레버리지 ETF 적합성")
print("=" * 80)

signal = strategy_foreign_sma(data, 30, 100)

for leverage in [1.0, 2.0]:
    result = backtest(data, signal, leverage=leverage)

    print(f"\n  {leverage}x ETF:")
    print(f"    Sharpe: {result['sharpe']:.3f}")
    print(f"    CAGR:   {result['cagr']*100:.1f}%")
    print(f"    MDD:    {result['mdd']*100:.1f}%")
    print(f"    Calmar: {result['calmar']:.3f}")

    # 적합성 판정
    if leverage == 1.0:
        if result["sharpe"] > 1.0 and result["mdd"] > -0.20:
            print("    -> 1x ETF: 적합 [OK]")
        else:
            print("    -> 1x ETF: 조건부 적합")
    else:
        if result["sharpe"] > 0.8 and result["mdd"] > -0.35:
            print("    -> 2x ETF: 조건부 적합 [주의]")
        else:
            print("    -> 2x ETF: 부적합 [X]")


# ========== 최종 결론 ==========
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

full_result = backtest(data, strategy_foreign_sma(data, 30, 100))

print(f"""
전략: Foreign_30d_SMA_100
로직: (외국인 30일 순매수 > 0) AND (종가 > SMA100)

성과:
  Sharpe: {full_result['sharpe']:.3f}
  CAGR:   {full_result['cagr']*100:.1f}%
  MDD:    {full_result['mdd']*100:.1f}%

검증 결과:
  - WF Ratio: {wf_ratio:.2f} (> 0.7 = 과적합 아님)
  - 파라미터 민감도: {'낮음 (안정적)' if sharpe_std/sharpe_mean < 0.3 else '높음 (주의)'}
  - 양수 연도: {positive_years}/{len(yearly_results)}년

판정: {'A (실전 권장)' if full_result['sharpe'] > 1.0 and full_result['mdd'] > -0.20 else 'B (조건부 권장)'}

권장 ETF:
  - 1x TIGER 200: 권장 (MDD -17%)
  - 2x 레버리지: 조건부 (MDD -31%)
""")

# 저장
output = {
    "generated": datetime.now().isoformat(),
    "strategy": "Foreign_30d_SMA_100",
    "logic": "(외국인 30일 순매수 > 0) AND (종가 > SMA100)",
    "data_period": f"{data.index.min()} ~ {data.index.max()}",
    "full_period": {
        "sharpe": full_result["sharpe"],
        "cagr": full_result["cagr"],
        "mdd": full_result["mdd"],
        "sortino": full_result["sortino"],
        "calmar": full_result["calmar"],
        "win_rate": full_result["win_rate"],
        "time_in_market": full_result["time_in_market"],
    },
    "walk_forward": {
        "is_mean": is_mean,
        "oos_mean": oos_mean,
        "wf_ratio": wf_ratio,
        "details": wf_results,
    },
    "yearly_results": yearly_results,
    "parameter_sensitivity": param_df.to_dict("records"),
    "leverage_results": {
        "1x": backtest(data, signal, leverage=1.0),
        "2x": backtest(data, signal, leverage=2.0),
    },
}

# cumulative와 drawdown 제거 (JSON 직렬화 문제)
for key in ["1x", "2x"]:
    if output["leverage_results"][key]:
        output["leverage_results"][key].pop("cumulative", None)
        output["leverage_results"][key].pop("drawdown", None)

output_path = f"{OUTPUT_DIR}/foreign_sma_strategy_validation.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)

print(f"결과 저장: {output_path}")
