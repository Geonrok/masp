"""
검증된 방법론으로 ETF 백테스트
==============================
원본 KOSPI200 검증 결과(Sharpe 2.25)와 일치하는 방법론 적용.

핵심 차이점:
1. 원본은 KOSPI200 선물 수익률 사용
2. 시그널 기반 포지션 유지 방식
3. 비용 처리 방식

이 스크립트는 원본 방법론을 따라
1x ETF와 2x ETF 성과를 비교합니다.
"""

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = "E:/투자/data/kosdaq_futures/multi_asset"
OUTPUT_DIR = "E:/투자/Multi-Asset Strategy Platform/logs"

print("=" * 80)
print("검증된 방법론으로 ETF 백테스트")
print("=" * 80)

# 데이터 로드
kospi200 = pd.read_parquet(f"{DATA_DIR}/kospi200.parquet")
vix = pd.read_parquet(f"{DATA_DIR}/vix.parquet")
semicon = pd.read_parquet(f"{DATA_DIR}/semicon.parquet")

# 데이터 병합
data = kospi200[["Close"]].copy()
data.columns = ["close"]
data["returns"] = data["close"].pct_change()

# T-1 shift 적용 (한국-미국 시차)
data["vix"] = vix["Close"].shift(1)  # T-1 VIX
data["vix_prev"] = vix["Close"].shift(2)  # T-2 VIX
data["semicon"] = semicon["Close"].shift(1)

for col in ["vix", "vix_prev", "semicon"]:
    data[col] = data[col].ffill()

# 지표 계산
data["vix_sma20"] = data["vix"].rolling(20).mean()
data["semicon_sma20"] = data["semicon"].rolling(20).mean()

data = data.dropna()
print(f"데이터: {data.index.min()} ~ {data.index.max()} ({len(data)}일)")

# ===== 원본 방법론 기반 백테스트 =====


def backtest_original_method(
    data, signal_series, leverage=1.0, fee=0.00003, slippage=0.0002
):
    """
    원본 KOSPI200 백테스트 방법론.

    원본 코드:
    position_change = signal.diff().abs()
    costs = position_change * cost
    strategy_returns = signal.shift(1) * returns - costs

    핵심:
    - signal.shift(1): 전일 시그널로 당일 포지션
    - 비용은 포지션 변경 시에만 발생
    """
    returns = data["returns"]
    total_cost = (fee + slippage) * 2  # 왕복

    # 레버리지 적용
    leveraged_returns = returns * leverage

    # 포지션 변경 감지
    position_change = signal_series.diff().abs()

    # 비용 (포지션 변경 시에만)
    costs = position_change * total_cost

    # 전략 수익률
    # signal.shift(1): T일 시그널은 T+1일 수익에 적용
    strategy_returns = signal_series.shift(1) * leveraged_returns - costs
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) < 252:
        return None

    # 누적 수익
    cumulative = (1 + strategy_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    if total_return <= -1:
        return None

    # 연환산
    years = len(strategy_returns) / 252
    annual_return = (1 + total_return) ** (1 / years) - 1

    # 변동성
    volatility = strategy_returns.std() * np.sqrt(252)

    # Sharpe
    sharpe = annual_return / volatility if volatility > 0 else 0

    # MDD
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    mdd = drawdown.min()

    # 추가 지표
    negative_returns = strategy_returns[strategy_returns < 0]
    downside_vol = (
        negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.001
    )
    sortino = annual_return / downside_vol

    calmar = annual_return / abs(mdd) if mdd != 0 else 0

    # 거래 횟수
    trades = (position_change > 0).sum()
    trades_per_year = trades / years

    # 시장 참여율
    time_in_market = (signal_series == 1).mean()

    # 승률
    winning = (strategy_returns > 0).sum()
    total = (strategy_returns != 0).sum()
    win_rate = winning / total if total > 0 else 0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "mdd": mdd,
        "volatility": volatility,
        "trades": int(trades),
        "trades_per_year": trades_per_year,
        "time_in_market": time_in_market,
        "win_rate": win_rate,
        "cumulative": cumulative,
        "drawdown": drawdown,
    }


def walk_forward_validation(data, signal_func, leverage=1.0, n_splits=11):
    """
    Walk-Forward 검증 (원본 방식: 11-fold).
    """
    total_len = len(data)
    test_size = total_len // n_splits

    is_sharpes = []
    oos_sharpes = []

    for i in range(n_splits - 1):
        # IS: 처음 ~ fold i까지
        is_end = (i + 1) * test_size
        is_data = data.iloc[:is_end]

        # OOS: fold i ~ fold i+1
        oos_start = is_end
        oos_end = min((i + 2) * test_size, total_len)
        oos_data = data.iloc[oos_start:oos_end]

        if len(oos_data) < 50:
            continue

        # IS 시그널 및 백테스트
        is_signal = signal_func(is_data)
        is_result = backtest_original_method(is_data, is_signal, leverage=leverage)

        # OOS 시그널 및 백테스트
        oos_signal = signal_func(oos_data)
        oos_result = backtest_original_method(oos_data, oos_signal, leverage=leverage)

        if is_result and oos_result:
            is_sharpes.append(is_result["sharpe"])
            oos_sharpes.append(oos_result["sharpe"])

    return {
        "is_mean": np.mean(is_sharpes) if is_sharpes else None,
        "oos_mean": np.mean(oos_sharpes) if oos_sharpes else None,
        "wf_ratio": (
            np.mean(oos_sharpes) / np.mean(is_sharpes)
            if is_sharpes and np.mean(is_sharpes) > 0
            else None
        ),
    }


def crisis_test(data, signal_series, leverage=1.0):
    """위기 기간 성과 테스트."""
    crisis_periods = {
        "COVID_2020_03": ("2020-03-01", "2020-03-31"),
        "Rate_Hike_2022": ("2022-01-01", "2022-12-31"),
    }

    results = {}

    for name, (start, end) in crisis_periods.items():
        mask = (data.index >= start) & (data.index <= end)
        if mask.sum() < 5:
            continue

        crisis_data = data[mask]
        crisis_signal = signal_series[mask]

        # Buy & Hold
        bh_return = (1 + crisis_data["returns"] * leverage).prod() - 1

        # 전략
        result = backtest_original_method(crisis_data, crisis_signal, leverage=leverage)

        if result:
            results[name] = {
                "strategy_return": result["total_return"],
                "bh_return": bh_return,
                "alpha": result["total_return"] - bh_return,
            }

    return results


# ===== 전략 정의 =====


def strategy_vix_below_sma20(data):
    """VIX Below SMA20 (A+ 등급)."""
    signal = pd.Series(0, index=data.index)
    signal[data["vix"] < data["vix_sma20"]] = 1
    return signal


def strategy_vix_declining(data):
    """VIX Declining (A+ 등급)."""
    signal = pd.Series(0, index=data.index)
    signal[data["vix"] < data["vix_prev"]] = 1
    return signal


def strategy_composite_50_30_20(data):
    """Composite 50/30/20."""
    vix_below = (data["vix"] < data["vix_sma20"]).astype(int)
    vix_declining = (data["vix"] < data["vix_prev"]).astype(int)
    semicon_above = (data["semicon"] > data["semicon_sma20"]).astype(int)

    composite = vix_below * 0.5 + vix_declining * 0.3 + semicon_above * 0.2

    signal = pd.Series(0, index=data.index)
    signal[composite >= 0.5] = 1
    return signal


def strategy_buy_hold(data):
    """Buy & Hold."""
    return pd.Series(1, index=data.index)


strategies = {
    "VIX_Below_SMA20": strategy_vix_below_sma20,
    "VIX_Declining": strategy_vix_declining,
    "Composite_50_30_20": strategy_composite_50_30_20,
    "Buy_Hold": strategy_buy_hold,
}

# ===== 테스트 실행 =====

# ETF 설정
etf_configs = [
    {"name": "1x_ETF_TIGER200", "leverage": 1.0, "fee": 0.00003, "slippage": 0.0001},
    {"name": "2x_ETF_Leveraged", "leverage": 2.0, "fee": 0.00003, "slippage": 0.0003},
]

print("\n" + "=" * 80)
print("전략 성과 비교")
print("=" * 80)

all_results = []

for strat_name, strat_func in strategies.items():
    signal = strat_func(data)

    for etf in etf_configs:
        result = backtest_original_method(
            data,
            signal,
            leverage=etf["leverage"],
            fee=etf["fee"],
            slippage=etf["slippage"],
        )

        if result is None:
            continue

        # Walk-Forward 검증
        wf = walk_forward_validation(data, strat_func, leverage=etf["leverage"])

        # 위기 테스트
        crisis = crisis_test(data, signal, leverage=etf["leverage"])

        entry = {
            "strategy": strat_name,
            "etf": etf["name"],
            "leverage": etf["leverage"],
            "sharpe": result["sharpe"],
            "cagr": result["annual_return"],
            "mdd": result["mdd"],
            "sortino": result["sortino"],
            "calmar": result["calmar"],
            "volatility": result["volatility"],
            "time_in_market": result["time_in_market"],
            "trades_per_year": result["trades_per_year"],
            "win_rate": result["win_rate"],
            "wf_is_sharpe": wf["is_mean"],
            "wf_oos_sharpe": wf["oos_mean"],
            "wf_ratio": wf["wf_ratio"],
        }

        # 위기 성과 추가
        for crisis_name, crisis_result in crisis.items():
            entry[f"{crisis_name}_alpha"] = crisis_result["alpha"]

        all_results.append(entry)

        print(f"\n{strat_name} ({etf['name']}):")
        print(f"  Sharpe: {result['sharpe']:.3f}")
        print(f"  CAGR: {result['annual_return']*100:.1f}%")
        print(f"  MDD: {result['mdd']*100:.1f}%")
        print(f"  Sortino: {result['sortino']:.3f}")
        print(f"  Time in Market: {result['time_in_market']*100:.1f}%")
        if wf["oos_mean"]:
            print(f"  WF OOS Sharpe: {wf['oos_mean']:.3f}")
            if wf["wf_ratio"]:
                print(f"  WF Ratio: {wf['wf_ratio']:.2f}")

# 결과 정리
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(["strategy", "leverage"])

# 1x vs 2x 비교
print("\n" + "=" * 80)
print("1x ETF vs 2x ETF 상세 비교")
print("=" * 80)

for strat_name in ["VIX_Below_SMA20", "VIX_Declining", "Composite_50_30_20"]:
    strat_data = results_df[results_df["strategy"] == strat_name]

    if len(strat_data) < 2:
        continue

    etf_1x = strat_data[strat_data["leverage"] == 1.0].iloc[0]
    etf_2x = strat_data[strat_data["leverage"] == 2.0].iloc[0]

    print(f"\n{strat_name}:")
    print(f"  {'지표':<20} {'1x ETF':<12} {'2x ETF':<12} {'비고'}")
    print(f"  {'-'*60}")
    print(
        f"  {'Sharpe':<20} {etf_1x['sharpe']:.3f}{'':<8} {etf_2x['sharpe']:.3f}{'':<8} {'1x 우위' if etf_1x['sharpe'] > etf_2x['sharpe'] else '2x 우위'}"
    )
    print(
        f"  {'CAGR':<20} {etf_1x['cagr']*100:.1f}%{'':<7} {etf_2x['cagr']*100:.1f}%{'':<7}"
    )
    print(
        f"  {'MDD':<20} {etf_1x['mdd']*100:.1f}%{'':<7} {etf_2x['mdd']*100:.1f}%{'':<7} {'MDD 2배 이상' if etf_2x['mdd'] < etf_1x['mdd'] * 1.8 else 'OK'}"
    )
    print(
        f"  {'Calmar (수익/위험)':<20} {etf_1x['calmar']:.3f}{'':<8} {etf_2x['calmar']:.3f}{'':<8} {'1x 우위' if etf_1x['calmar'] > etf_2x['calmar'] else '2x 우위'}"
    )

    if etf_1x["wf_oos_sharpe"] and etf_2x["wf_oos_sharpe"]:
        print(
            f"  {'WF OOS Sharpe':<20} {etf_1x['wf_oos_sharpe']:.3f}{'':<8} {etf_2x['wf_oos_sharpe']:.3f}"
        )

# 유효 전략 판정
print("\n" + "=" * 80)
print("유효 전략 판정")
print("=" * 80)

# 1x ETF 기준: Sharpe > 1.5, MDD > -15%
viable_1x_strict = results_df[
    (results_df["leverage"] == 1.0)
    & (results_df["sharpe"] > 1.5)
    & (results_df["mdd"] > -0.15)
    & (results_df["strategy"] != "Buy_Hold")
]

# 1x ETF 완화 기준: Sharpe > 0.8, MDD > -20%
viable_1x_relaxed = results_df[
    (results_df["leverage"] == 1.0)
    & (results_df["sharpe"] > 0.8)
    & (results_df["mdd"] > -0.20)
    & (results_df["strategy"] != "Buy_Hold")
]

# 2x ETF 기준: Sharpe > 0.8, MDD > -25%
viable_2x = results_df[
    (results_df["leverage"] == 2.0)
    & (results_df["sharpe"] > 0.8)
    & (results_df["mdd"] > -0.25)
    & (results_df["strategy"] != "Buy_Hold")
]

print(f"\n1x ETF 유효 전략 (엄격: Sharpe>1.5, MDD>-15%): {len(viable_1x_strict)}")
print(f"1x ETF 유효 전략 (완화: Sharpe>0.8, MDD>-20%): {len(viable_1x_relaxed)}")
print(f"2x ETF 유효 전략 (Sharpe>0.8, MDD>-25%): {len(viable_2x)}")

if len(viable_1x_relaxed) > 0:
    print("\n1x ETF 유효 전략:")
    for _, row in viable_1x_relaxed.iterrows():
        print(
            f"  - {row['strategy']}: Sharpe {row['sharpe']:.3f}, MDD {row['mdd']*100:.1f}%"
        )

if len(viable_2x) > 0:
    print("\n2x ETF 유효 전략:")
    for _, row in viable_2x.iterrows():
        print(
            f"  - {row['strategy']}: Sharpe {row['sharpe']:.3f}, MDD {row['mdd']*100:.1f}%"
        )

# ===== 최종 결론 =====
print("\n" + "=" * 80)
print("최종 결론 및 권장사항")
print("=" * 80)

# 권장사항 결정
recommendation = {}

if len(viable_2x) > 0:
    best_2x = viable_2x.sort_values("sharpe", ascending=False).iloc[0]
    recommendation = {
        "decision": "2x_ETF",
        "strategy": best_2x["strategy"],
        "sharpe": best_2x["sharpe"],
        "cagr": best_2x["cagr"],
        "mdd": best_2x["mdd"],
        "note": "2x ETF 사용 가능 (단, MDD 주의)",
    }
    print(f"\n결론: 2x ETF 사용 가능")
    print(f"\n추천 전략: {best_2x['strategy']} (2x ETF)")
    print(f"  Sharpe: {best_2x['sharpe']:.3f}")
    print(f"  CAGR: {best_2x['cagr']*100:.1f}%")
    print(f"  MDD: {best_2x['mdd']*100:.1f}%")

elif len(viable_1x_relaxed) > 0:
    best_1x = viable_1x_relaxed.sort_values("sharpe", ascending=False).iloc[0]
    recommendation = {
        "decision": "1x_ETF",
        "strategy": best_1x["strategy"],
        "sharpe": best_1x["sharpe"],
        "cagr": best_1x["cagr"],
        "mdd": best_1x["mdd"],
        "note": "1x ETF 권장 (2x는 MDD 과다)",
    }
    print(f"\n결론: 1x ETF(TIGER 200) 사용 권장")
    print(f"\n추천 전략: {best_1x['strategy']} (1x ETF)")
    print(f"  Sharpe: {best_1x['sharpe']:.3f}")
    print(f"  CAGR: {best_1x['cagr']*100:.1f}%")
    print(f"  MDD: {best_1x['mdd']*100:.1f}%")
    print(f"\n이유: 2x ETF는 모든 전략에서 MDD -25% 초과")

else:
    # 원본 검증 결과와 비교
    print("\n경고: 백테스트 결과가 원본 검증과 다릅니다.")
    print("\n원본 검증 결과 (kospi200_realistic_validation.json):")
    print("  VIX_Below_SMA20: Sharpe 2.25, MDD -12.0%")
    print("\n현재 백테스트와의 차이점 분석이 필요합니다.")

    # 현재 결과에서 최선
    vix_1x = results_df[
        (results_df["strategy"] == "VIX_Below_SMA20") & (results_df["leverage"] == 1.0)
    ]
    if len(vix_1x) > 0:
        vix_1x = vix_1x.iloc[0]
        recommendation = {
            "decision": "1x_ETF_with_caution",
            "strategy": "VIX_Below_SMA20",
            "sharpe": vix_1x["sharpe"],
            "cagr": vix_1x["cagr"],
            "mdd": vix_1x["mdd"],
            "note": "원본 검증과 차이 있음, 추가 검증 필요",
        }
        print(f"\n현재 백테스트 결과:")
        print(
            f"  VIX_Below_SMA20 (1x): Sharpe {vix_1x['sharpe']:.3f}, MDD {vix_1x['mdd']*100:.1f}%"
        )

# 결과 저장
output = {
    "generated": datetime.now().isoformat(),
    "type": "etf_validated_backtest",
    "data_period": f"{data.index.min()} ~ {data.index.max()}",
    "results": results_df.to_dict("records"),
    "viable_1x_strict": (
        viable_1x_strict.to_dict("records") if len(viable_1x_strict) > 0 else []
    ),
    "viable_1x_relaxed": (
        viable_1x_relaxed.to_dict("records") if len(viable_1x_relaxed) > 0 else []
    ),
    "viable_2x": viable_2x.to_dict("records") if len(viable_2x) > 0 else [],
    "recommendation": recommendation,
}

output_path = f"{OUTPUT_DIR}/etf_validated_backtest.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)

print(f"\n결과 저장: {output_path}")
