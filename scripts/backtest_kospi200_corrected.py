"""
KOSPI200 선물 전략 - Look-Ahead Bias 수정 후 재검증
====================================================
원본 검증에서 VIX shift(1) 미적용 확인.
모든 미국 데이터에 shift(1) 적용 후 재검증.

검증 기준:
- Sharpe > 1.0: 양호
- Sharpe > 1.5: 우수
- MDD > -15%: 관리 가능
- WF OOS Sharpe > IS Sharpe * 0.7: 과적합 아님
"""

import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

INVESTOR_DIR = "E:/투자/data/kr_stock/investor_trading"
KOSPI_DIR = "E:/투자/data/kospi_futures"
MULTI_ASSET_DIR = "E:/투자/data/kosdaq_futures/multi_asset"
OUTPUT_DIR = "E:/투자/Multi-Asset Strategy Platform/logs"

# 비용 설정
REALISTIC_FEE = 0.00005
REALISTIC_SLIPPAGE = 0.0004
REALISTIC_COST = (REALISTIC_FEE + REALISTIC_SLIPPAGE) * 2  # 0.09%

print("=" * 80)
print("KOSPI200 전략 - Look-Ahead Bias 수정 후 재검증")
print(f"거래 비용: {REALISTIC_COST*100:.3f}% (왕복)")
print("=" * 80)

# ========== 데이터 로드 ==========
print("\n[1] 데이터 로드...")

# 투자자 데이터
investor_files = [f for f in os.listdir(INVESTOR_DIR) if f.endswith("_investor.csv")]
all_investor = []
for f in investor_files:
    try:
        df = pd.read_csv(f"{INVESTOR_DIR}/{f}", encoding="utf-8-sig")
        df["날짜"] = pd.to_datetime(df["날짜"])
        df = df.set_index("날짜")
        all_investor.append(df[["기관합계", "외국인합계", "개인"]])
    except:
        pass
market_investor = all_investor[0].copy()
for df in all_investor[1:]:
    market_investor = market_investor.add(df, fill_value=0)
market_investor = market_investor.sort_index()

# KOSPI200 일봉
kospi200 = pd.read_parquet(f"{KOSPI_DIR}/kospi200_daily_yf.parquet")
kospi200.columns = [c.lower() for c in kospi200.columns]
if kospi200.index.tz is not None:
    kospi200.index = kospi200.index.tz_localize(None)

# 멀티 자산
multi_assets = {}
for asset in ["vix", "sp500", "nasdaq", "dxy", "usdkrw", "semicon"]:
    try:
        df = pd.read_parquet(f"{MULTI_ASSET_DIR}/{asset}.parquet")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        multi_assets[asset] = df["Close"]
    except Exception as e:
        print(f"  {asset} 로드 실패: {e}")

# 데이터 병합
daily_data = kospi200[["close", "high", "low", "open", "volume"]].copy()
daily_data["returns"] = daily_data["close"].pct_change()

# 한국 데이터 (shift 불필요)
daily_data["foreign"] = market_investor["외국인합계"]
daily_data["institution"] = market_investor["기관합계"]

# ★★★ 핵심 수정: 미국 데이터에 shift(1) 적용 ★★★
for name, series in multi_assets.items():
    daily_data[name] = series.shift(1)  # T-1 데이터 사용!

daily_data = daily_data.ffill().dropna()
print(
    f"  데이터: {len(daily_data)}일 ({daily_data.index.min().date()} ~ {daily_data.index.max().date()})"
)

# 지표 계산
for period in [5, 7, 10, 15, 20, 30, 50, 100, 200]:
    if len(daily_data) > period:
        daily_data[f"sma_{period}"] = daily_data["close"].rolling(period).mean()
        daily_data[f"ema_{period}"] = (
            daily_data["close"].ewm(span=period, adjust=False).mean()
        )

# 외국인 누적
for period in [10, 20, 30]:
    daily_data[f"foreign_{period}d"] = daily_data["foreign"].rolling(period).sum()

# RSI
delta = daily_data["close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
daily_data["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

# CCI
tp = (daily_data["high"] + daily_data["low"] + daily_data["close"]) / 3
sma_tp = tp.rolling(14).mean()
mad = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean())
daily_data["cci_14"] = (tp - sma_tp) / (0.015 * mad + 1e-10)

# VIX 지표
daily_data["vix_sma_10"] = daily_data["vix"].rolling(10).mean()
daily_data["vix_sma_20"] = daily_data["vix"].rolling(20).mean()
daily_data["vix_change_5d"] = daily_data["vix"].pct_change(5)

# Semicon 지표
daily_data["semicon_sma_20"] = daily_data["semicon"].rolling(20).mean()
daily_data["semicon_sma_50"] = daily_data["semicon"].rolling(50).mean()

daily_data = daily_data.dropna()


def detailed_backtest(data, signal, cost, annual_factor=252, min_bars=252):
    """상세 백테스트."""
    returns = data["returns"]
    position_change = signal.diff().abs()
    costs = position_change * cost
    strategy_returns = signal.shift(1) * returns - costs
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) < min_bars:
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

    # MDD 기간
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        dd_groups = (~in_drawdown).cumsum()
        dd_lengths = in_drawdown.groupby(dd_groups).sum()
        max_dd_duration = dd_lengths.max() if len(dd_lengths) > 0 else 0
    else:
        max_dd_duration = 0

    # Sortino
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = (
        downside_returns.std() * np.sqrt(annual_factor)
        if len(downside_returns) > 0
        else 0
    )
    sortino = annual_return / downside_std if downside_std > 0 else 0

    # Calmar
    calmar = annual_return / abs(mdd) if mdd != 0 else 0

    # 승률
    winning = (strategy_returns > 0).sum()
    losing = (strategy_returns < 0).sum()
    total = winning + losing
    win_rate = winning / total if total > 0 else 0

    # 거래 횟수
    trades = int((signal.diff().abs() > 0).sum())
    trades_per_year = trades / (len(strategy_returns) / annual_factor)

    return {
        "annual_return": annual_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "mdd": mdd,
        "max_dd_duration": int(max_dd_duration),
        "win_rate": win_rate,
        "trades_per_year": trades_per_year,
        "total_return": total_return,
    }


def walk_forward_analysis(data, signal_func, cost, train_ratio=0.7, annual_factor=252):
    """Walk-forward 검증."""
    n = len(data)
    train_end = int(n * train_ratio)

    train_data = data.iloc[:train_end].copy()
    test_data = data.iloc[train_end:].copy()

    train_signal = signal_func(train_data)
    train_result = detailed_backtest(train_data, train_signal, cost, annual_factor)

    test_signal = signal_func(test_data)
    test_result = detailed_backtest(
        test_data, test_signal, cost, annual_factor, min_bars=50
    )

    return {
        "train": train_result,
        "test": test_result,
        "train_period": f"{train_data.index.min().date()} ~ {train_data.index.max().date()}",
        "test_period": f"{test_data.index.min().date()} ~ {test_data.index.max().date()}",
    }


def yearly_analysis(data, signal_func, cost):
    """연도별 성과."""
    signal = signal_func(data)
    returns = data["returns"]
    position_change = signal.diff().abs()
    costs = position_change * cost
    strategy_returns = signal.shift(1) * returns - costs
    strategy_returns = strategy_returns.dropna()

    yearly = strategy_returns.groupby(strategy_returns.index.year)

    results = {}
    for year, year_returns in yearly:
        if len(year_returns) > 20:
            total_ret = (1 + year_returns).prod() - 1
            results[year] = {"return": total_ret}

    return results


# ========== 전략 정의 ==========
print("\n[2] 전략 정의...")

strategies = {
    # VIX 기반 (미국 데이터 - 이제 T-1 적용됨)
    "VIX_Below_SMA20": lambda d: (d["vix"] < d["vix_sma_20"]).astype(int),
    "VIX_Declining": lambda d: (
        (d["vix"] < d["vix_sma_10"]) & (d["vix_change_5d"] < -0.05)
    ).astype(int),
    # 반도체 + 외국인 (Semicon은 T-1, Foreign은 한국 데이터)
    "Semicon_Foreign": lambda d: (
        (d["semicon"] > d["semicon_sma_20"]) & (d["foreign_20d"] > 0)
    ).astype(int),
    "Semicon_CCI": lambda d: (
        (d["semicon"] > d["semicon_sma_20"]) & (d["cci_14"] > 0)
    ).astype(int),
    # 외국인 기반 (한국 데이터 - shift 불필요)
    "Foreign_20d_Trend": lambda d: (
        (d["foreign_20d"] > 0) & (d["close"] > d["sma_50"])
    ).astype(int),
    "Foreign_30d_SMA_100": lambda d: (
        (d["foreign_30d"] > 0) & (d["close"] > d["sma_100"])
    ).astype(int),
    # 가격 기반 (한국 데이터)
    "SMA_20_50_Cross": lambda d: (d["sma_20"] > d["sma_50"]).astype(int),
    "SMA_50_200_Cross": lambda d: (d["sma_50"] > d["sma_200"]).astype(int),
    "EMA_20_50_Cross": lambda d: (d["ema_20"] > d["ema_50"]).astype(int),
    # 복합 전략
    "Conservative_Combo": lambda d: (
        (d["vix"] < d["vix_sma_20"])
        & (d["foreign_20d"] > 0)
        & (d["close"] > d["sma_50"])
    ).astype(int),
}

print(f"  전략 수: {len(strategies)}")

# ========== 검증 실행 ==========
print("\n[3] 검증 실행...")
print("=" * 80)

all_results = []

for name, signal_func in strategies.items():
    print(f"\n  {name}:")

    result = {"strategy": name}

    # 전체 기간 백테스트
    signal = signal_func(daily_data)
    full_result = detailed_backtest(daily_data, signal, REALISTIC_COST, 252)

    if full_result:
        result["sharpe"] = full_result["sharpe"]
        result["cagr"] = full_result["annual_return"]
        result["mdd"] = full_result["mdd"]
        result["sortino"] = full_result["sortino"]
        result["calmar"] = full_result["calmar"]
        result["win_rate"] = full_result["win_rate"]
        result["trades_per_year"] = full_result["trades_per_year"]

        print(
            f"    Sharpe: {full_result['sharpe']:.3f}, CAGR: {full_result['annual_return']*100:.1f}%, MDD: {full_result['mdd']*100:.1f}%"
        )

    # Walk-Forward 검증
    wf = walk_forward_analysis(daily_data, signal_func, REALISTIC_COST, 0.7, 252)
    if wf["test"]:
        result["wf_test_sharpe"] = wf["test"]["sharpe"]
        result["wf_test_cagr"] = wf["test"]["annual_return"]
        result["test_period"] = wf["test_period"]
        print(
            f"    WF Test: Sharpe {wf['test']['sharpe']:.3f}, CAGR {wf['test']['annual_return']*100:.1f}%"
        )

    # 연도별 분석
    yearly = yearly_analysis(daily_data, signal_func, REALISTIC_COST)
    positive_years = sum(1 for y in yearly.values() if y["return"] > 0)
    total_years = len(yearly)
    result["positive_years"] = positive_years
    result["total_years"] = total_years
    print(f"    연간 양수: {positive_years}/{total_years}년")

    all_results.append(result)

# 결과 정리
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values("sharpe", ascending=False)

# 유효 전략 필터
viable_strict = results_df[(results_df["sharpe"] > 1.0) & (results_df["mdd"] > -0.15)]

viable_relaxed = results_df[(results_df["sharpe"] > 0.5) & (results_df["mdd"] > -0.20)]

print("\n" + "=" * 80)
print("검증 결과 요약")
print("=" * 80)

print(f"\n총 전략: {len(results_df)}")
print(f"엄격 기준 (Sharpe>1.0, MDD>-15%): {len(viable_strict)}")
print(f"완화 기준 (Sharpe>0.5, MDD>-20%): {len(viable_relaxed)}")

if len(viable_strict) > 0:
    print("\n엄격 기준 통과 전략:")
    for _, row in viable_strict.iterrows():
        print(
            f"  - {row['strategy']}: Sharpe {row['sharpe']:.3f}, MDD {row['mdd']*100:.1f}%"
        )

if len(viable_relaxed) > 0:
    print("\n완화 기준 통과 전략:")
    for _, row in viable_relaxed.iterrows():
        print(
            f"  - {row['strategy']}: Sharpe {row['sharpe']:.3f}, MDD {row['mdd']*100:.1f}%"
        )

# 전체 결과 출력
print("\n전체 결과 (Sharpe 순):")
cols = ["strategy", "sharpe", "cagr", "mdd", "wf_test_sharpe"]
print(results_df[cols].to_string(index=False))

# ========== 레버리지 ETF 테스트 ==========
print("\n" + "=" * 80)
print("레버리지 ETF 적합성 테스트")
print("=" * 80)

if len(viable_relaxed) > 0:
    best_strategy = viable_relaxed.iloc[0]["strategy"]
    best_func = strategies[best_strategy]
    signal = best_func(daily_data)

    for leverage in [1.0, 2.0]:
        lev_data = daily_data.copy()
        lev_data["returns"] = daily_data["returns"] * leverage
        lev_cost = REALISTIC_COST * (1.5 if leverage > 1 else 1)

        result = detailed_backtest(lev_data, signal, lev_cost)

        if result:
            print(f"\n{best_strategy} ({leverage}x):")
            print(f"  Sharpe: {result['sharpe']:.3f}")
            print(f"  CAGR: {result['annual_return']*100:.1f}%")
            print(f"  MDD: {result['mdd']*100:.1f}%")

# 최종 결론
print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

if len(viable_strict) > 0:
    best = viable_strict.iloc[0]
    print(f"\n추천 전략: {best['strategy']}")
    print(f"  Sharpe: {best['sharpe']:.3f}")
    print(f"  CAGR: {best['cagr']*100:.1f}%")
    print(f"  MDD: {best['mdd']*100:.1f}%")
    print("\n판정: A (실거래 적합)")
elif len(viable_relaxed) > 0:
    best = viable_relaxed.iloc[0]
    print(f"\n차선 전략: {best['strategy']}")
    print(f"  Sharpe: {best['sharpe']:.3f}")
    print(f"  CAGR: {best['cagr']*100:.1f}%")
    print(f"  MDD: {best['mdd']*100:.1f}%")
    print("\n판정: B (조건부 적합)")
else:
    print("\n판정: F (실거래 부적합)")
    print("  Look-Ahead Bias 제거 후 유효한 전략이 없습니다.")

# 저장
output = {
    "generated": datetime.now().isoformat(),
    "type": "kospi200_corrected_validation",
    "note": "Look-Ahead Bias removed (US data shifted by 1 day)",
    "cost": REALISTIC_COST,
    "data_period": f"{daily_data.index.min()} ~ {daily_data.index.max()}",
    "results": results_df.to_dict("records"),
    "viable_strict": viable_strict.to_dict("records") if len(viable_strict) > 0 else [],
    "viable_relaxed": (
        viable_relaxed.to_dict("records") if len(viable_relaxed) > 0 else []
    ),
}

output_path = f"{OUTPUT_DIR}/kospi200_corrected_validation.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)

print(f"\n결과 저장: {output_path}")
