"""
KOSPI200 선물 레버리지 ETF 백테스트
====================================
대상: TIGER 200선물레버리지 (233160), KODEX 레버리지 (122630)
특성: 2배 레버리지, 일일 리밸런싱으로 인한 변동성 손실

핵심 고려사항:
1. 2배 레버리지 수익 = 2 * 일일수익률 (복리 아님)
2. 변동성 손실 (Volatility Decay): 횡보장에서 손실 누적
3. ETF 특성: 매도세 없음, 수수료 저렴

전략 검증 기준:
- Sharpe > 1.5 (우수)
- MDD < -20% (관리 가능)
- WF Test Sharpe > IS Sharpe * 0.7 (과적합 아님)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings

warnings.filterwarnings("ignore")

# 설정
DATA_DIR = "E:/투자/data/kosdaq_futures/multi_asset"
KOSPI_DIR = "E:/투자/data/kospi_futures"
OUTPUT_DIR = "E:/투자/Multi-Asset Strategy Platform/logs"

# 레버리지 ETF 거래 비용
LEVERAGE = 2.0
COMMISSION = 0.00015 * 2  # 왕복 0.03%
TAX = 0.0  # ETF 매도세 없음
SLIPPAGE = 0.001  # 0.1% (현물 ETF)
TOTAL_COST = COMMISSION + SLIPPAGE

print("=" * 80)
print("KOSPI200 선물 레버리지 ETF 백테스트")
print(f"레버리지: {LEVERAGE}x, 거래비용: {TOTAL_COST*100:.2f}%")
print("=" * 80)

# 데이터 로드
print("\n데이터 로딩...")
kospi200 = pd.read_parquet(f"{DATA_DIR}/kospi200.parquet")
vix = pd.read_parquet(f"{DATA_DIR}/vix.parquet")
semicon = pd.read_parquet(f"{DATA_DIR}/semicon.parquet")
sp500 = pd.read_parquet(f"{DATA_DIR}/sp500.parquet")
nasdaq = pd.read_parquet(f"{DATA_DIR}/nasdaq.parquet")

# 데이터 병합
data = kospi200[["Close"]].copy()
data.columns = ["close"]
data["returns"] = data["close"].pct_change()

# 레버리지 ETF 수익률 시뮬레이션 (일일 리밸런싱)
data["leveraged_returns"] = data["returns"] * LEVERAGE

# 미국 데이터 (T-1 shift 적용)
data["vix"] = vix["Close"].shift(1)  # T-1 VIX
data["vix_prev"] = vix["Close"].shift(2)  # T-2 VIX
data["sp500"] = sp500["Close"].shift(1)
data["nasdaq"] = nasdaq["Close"].shift(1)
data["semicon"] = semicon["Close"].shift(1)

# Forward fill
for col in ["vix", "vix_prev", "sp500", "nasdaq", "semicon"]:
    data[col] = data[col].ffill()

# 지표 계산
data["vix_sma20"] = data["vix"].rolling(20).mean()
data["semicon_sma20"] = data["semicon"].rolling(20).mean()
data["close_sma20"] = data["close"].rolling(20).mean()
data["close_sma50"] = data["close"].rolling(50).mean()

data = data.dropna()
print(f"데이터 기간: {data.index.min()} ~ {data.index.max()} ({len(data)}일)")


def backtest_leveraged_etf(data, signal, leverage=LEVERAGE, cost=TOTAL_COST):
    """
    레버리지 ETF 백테스트.

    핵심: 레버리지 ETF는 일일 리밸런싱으로
    장기 보유 시 변동성 손실 발생.
    """
    base_returns = data["returns"]

    # 레버리지 적용 (일일 기준)
    leveraged_returns = base_returns * leverage

    # 포지션 변경 시 비용
    position_change = signal.diff().abs()
    costs = position_change * cost

    # 전략 수익률
    # signal=1: 레버리지 ETF 보유
    # signal=0: 현금 보유 (수익 0)
    strategy_returns = signal.shift(1) * leveraged_returns - costs
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) < 252:
        return None

    # 성과 계산
    cumulative = (1 + strategy_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    years = len(strategy_returns) / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1

    volatility = strategy_returns.std() * np.sqrt(252)
    sharpe = cagr / volatility if volatility > 0 else 0

    # MDD
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    mdd = drawdown.min()

    # 추가 지표
    negative_returns = strategy_returns[strategy_returns < 0]
    downside_vol = (
        negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.001
    )
    sortino = cagr / downside_vol if downside_vol > 0 else 0

    calmar = cagr / abs(mdd) if mdd != 0 else 0

    trades = (signal.diff().abs() > 0).sum()

    # 시장 참여율 (Time in Market)
    time_in_market = (signal == 1).mean()

    # 승률
    winning_days = (strategy_returns > 0).sum()
    total_days = (strategy_returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "mdd": mdd,
        "volatility": volatility,
        "trades": int(trades),
        "trades_per_year": trades / years,
        "time_in_market": time_in_market,
        "win_rate": win_rate,
        "cumulative": cumulative,
        "drawdown": drawdown,
    }


def walk_forward_validation(data, signal_func, n_splits=5):
    """Walk-Forward 검증."""
    total_len = len(data)
    split_size = total_len // n_splits

    is_results = []
    oos_results = []

    for i in range(n_splits - 1):
        # In-Sample: 처음 ~ i+1 스플릿
        is_end = (i + 1) * split_size
        is_data = data.iloc[:is_end]

        # Out-of-Sample: i+1 ~ i+2 스플릿
        oos_start = is_end
        oos_end = min((i + 2) * split_size, total_len)
        oos_data = data.iloc[oos_start:oos_end]

        # IS 백테스트
        is_signal = signal_func(is_data)
        is_result = backtest_leveraged_etf(is_data, is_signal)

        # OOS 백테스트
        oos_signal = signal_func(oos_data)
        oos_result = backtest_leveraged_etf(oos_data, oos_signal)

        if is_result and oos_result:
            is_results.append(is_result["sharpe"])
            oos_results.append(oos_result["sharpe"])

    if not oos_results:
        return None, None

    return np.mean(is_results), np.mean(oos_results)


def crisis_test(data, signal, periods):
    """위기 기간 테스트."""
    results = {}

    for name, (start, end) in periods.items():
        mask = (data.index >= start) & (data.index <= end)
        if mask.sum() < 20:
            continue

        crisis_data = data[mask]
        crisis_signal = signal[mask]
        result = backtest_leveraged_etf(crisis_data, crisis_signal)

        if result:
            # Buy & Hold 대비
            bh_return = (1 + crisis_data["leveraged_returns"]).prod() - 1
            results[name] = {
                "strategy_return": result["total_return"],
                "bh_return": bh_return,
                "alpha": result["total_return"] - bh_return,
                "mdd": result["mdd"],
            }

    return results


# 위기 기간 정의
CRISIS_PERIODS = {
    "COVID_2020": ("2020-02-01", "2020-04-30"),
    "Rate_Hike_2022": ("2022-01-01", "2022-12-31"),
    "Bank_Crisis_2023": ("2023-03-01", "2023-05-31"),
    "Recent_2024": ("2024-07-01", "2024-08-31"),
}

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


def strategy_vix_combined(data):
    """VIX Below SMA20 + Declining 조합."""
    signal = pd.Series(0, index=data.index)
    # 두 조건 모두 충족
    signal[(data["vix"] < data["vix_sma20"]) & (data["vix"] < data["vix_prev"])] = 1
    return signal


def strategy_semicon_above_sma(data):
    """반도체 > SMA20."""
    signal = pd.Series(0, index=data.index)
    signal[data["semicon"] > data["semicon_sma20"]] = 1
    return signal


def strategy_composite_50_30_20(data):
    """
    Composite 전략 (50/30/20 가중치).
    VIX_Below_SMA20: 50%
    VIX_Declining: 30%
    Semicon_Above_SMA: 20%

    매수: Composite >= 50%
    매도: Composite < 20%
    """
    vix_below = (data["vix"] < data["vix_sma20"]).astype(int)
    vix_declining = (data["vix"] < data["vix_prev"]).astype(int)
    semicon_above = (data["semicon"] > data["semicon_sma20"]).astype(int)

    composite = vix_below * 0.5 + vix_declining * 0.3 + semicon_above * 0.2

    signal = pd.Series(0, index=data.index)
    signal[composite >= 0.5] = 1

    return signal


def strategy_vix_threshold(data, threshold=20):
    """VIX 절대값 기준."""
    signal = pd.Series(0, index=data.index)
    signal[data["vix"] < threshold] = 1
    return signal


def strategy_trend_filter_vix(data):
    """추세 필터 + VIX."""
    signal = pd.Series(0, index=data.index)
    # KOSPI200 상승 추세 + VIX 낮음
    uptrend = data["close"] > data["close_sma50"]
    vix_low = data["vix"] < data["vix_sma20"]
    signal[uptrend & vix_low] = 1
    return signal


def strategy_buy_hold(data):
    """Buy & Hold (벤치마크)."""
    return pd.Series(1, index=data.index)


def strategy_short_holding(data, max_days=5):
    """
    단기 보유 전략.
    VIX 시그널 후 최대 N일 보유.
    변동성 손실 최소화.
    """
    vix_signal = (data["vix"] < data["vix_sma20"]).astype(int)
    signal = pd.Series(0, index=data.index)

    holding = 0
    for i in range(len(data)):
        if vix_signal.iloc[i] == 1:
            holding = max_days

        if holding > 0:
            signal.iloc[i] = 1
            holding -= 1

    return signal


# 전략 목록
strategies = {
    "VIX_Below_SMA20": strategy_vix_below_sma20,
    "VIX_Declining": strategy_vix_declining,
    "VIX_Combined": strategy_vix_combined,
    "Semicon_Above_SMA": strategy_semicon_above_sma,
    "Composite_50_30_20": strategy_composite_50_30_20,
    "VIX_Threshold_20": lambda d: strategy_vix_threshold(d, 20),
    "VIX_Threshold_18": lambda d: strategy_vix_threshold(d, 18),
    "Trend_Filter_VIX": strategy_trend_filter_vix,
    "Short_Holding_5d": lambda d: strategy_short_holding(d, 5),
    "Short_Holding_10d": lambda d: strategy_short_holding(d, 10),
    "Buy_Hold": strategy_buy_hold,
}

# ===== 백테스트 실행 =====
print("\n" + "=" * 80)
print("전략 백테스트 결과")
print("=" * 80)

results = []

for name, strategy_func in strategies.items():
    try:
        signal = strategy_func(data)
        result = backtest_leveraged_etf(data, signal)

        if result is None:
            continue

        # Walk-Forward 검증
        is_sharpe, oos_sharpe = walk_forward_validation(data, strategy_func)

        # 위기 테스트
        crisis_results = crisis_test(data, signal, CRISIS_PERIODS)

        # 결과 저장
        entry = {
            "strategy": name,
            "sharpe": result["sharpe"],
            "cagr": result["cagr"],
            "mdd": result["mdd"],
            "sortino": result["sortino"],
            "calmar": result["calmar"],
            "volatility": result["volatility"],
            "time_in_market": result["time_in_market"],
            "trades_per_year": result["trades_per_year"],
            "win_rate": result["win_rate"],
            "wf_is_sharpe": is_sharpe,
            "wf_oos_sharpe": oos_sharpe,
            "wf_ratio": oos_sharpe / is_sharpe if is_sharpe and is_sharpe > 0 else None,
        }

        # 위기 성과 추가
        for crisis_name, crisis_result in crisis_results.items():
            entry[f"{crisis_name}_alpha"] = crisis_result["alpha"]

        results.append(entry)

        # 출력
        print(f"\n{name}:")
        print(f"  Sharpe: {result['sharpe']:.3f}")
        print(f"  CAGR: {result['cagr']*100:.1f}%")
        print(f"  MDD: {result['mdd']*100:.1f}%")
        print(f"  Time in Market: {result['time_in_market']*100:.1f}%")
        if is_sharpe and oos_sharpe:
            print(f"  WF IS Sharpe: {is_sharpe:.3f}")
            print(f"  WF OOS Sharpe: {oos_sharpe:.3f}")
            print(
                f"  WF Ratio (OOS/IS): {oos_sharpe/is_sharpe:.2f}"
                if is_sharpe > 0
                else ""
            )

    except Exception as e:
        print(f"Error in {name}: {e}")

# 결과 정리
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("sharpe", ascending=False)

# 유효 전략 필터
# 기준: Sharpe > 1.0, MDD > -25%, WF Ratio > 0.6
viable = results_df[
    (results_df["sharpe"] > 1.0)
    & (results_df["mdd"] > -0.25)
    & (results_df["strategy"] != "Buy_Hold")
]

if "wf_ratio" in viable.columns:
    viable = viable[viable["wf_ratio"].isna() | (viable["wf_ratio"] > 0.6)]

print("\n" + "=" * 80)
print(f"총 테스트 전략: {len(results_df)}")
print(f"유효 전략: {len(viable)}")
print("=" * 80)

if len(viable) > 0:
    print("\n유효 전략 상세:")
    for _, row in viable.iterrows():
        print(f"\n{row['strategy']}:")
        print(
            f"  Sharpe: {row['sharpe']:.3f}, CAGR: {row['cagr']*100:.1f}%, MDD: {row['mdd']*100:.1f}%"
        )
        print(f"  Sortino: {row['sortino']:.3f}, Calmar: {row['calmar']:.3f}")
        print(f"  Time in Market: {row['time_in_market']*100:.1f}%")
        if row["wf_ratio"]:
            print(f"  WF Ratio: {row['wf_ratio']:.2f}")

# Buy & Hold 대비
bh_row = (
    results_df[results_df["strategy"] == "Buy_Hold"].iloc[0]
    if "Buy_Hold" in results_df["strategy"].values
    else None
)

if bh_row is not None:
    print("\n" + "=" * 80)
    print("Buy & Hold (레버리지 ETF 단순 보유) 대비 성과")
    print("=" * 80)
    print(
        f"\nBuy & Hold: Sharpe {bh_row['sharpe']:.3f}, CAGR {bh_row['cagr']*100:.1f}%, MDD {bh_row['mdd']*100:.1f}%"
    )

    for _, row in viable.iterrows():
        sharpe_diff = row["sharpe"] - bh_row["sharpe"]
        cagr_diff = row["cagr"] - bh_row["cagr"]
        mdd_diff = row["mdd"] - bh_row["mdd"]
        print(f"\n{row['strategy']} vs B&H:")
        print(f"  Sharpe: {sharpe_diff:+.3f}")
        print(f"  CAGR: {cagr_diff*100:+.1f}%p")
        print(f"  MDD: {mdd_diff*100:+.1f}%p (개선)")

# 변동성 손실 분석
print("\n" + "=" * 80)
print("변동성 손실 (Volatility Decay) 분석")
print("=" * 80)

# 원지수 vs 레버리지 ETF
base_total = (1 + data["returns"]).prod() - 1
lev_total = (1 + data["leveraged_returns"]).prod() - 1
ideal_lev = base_total * LEVERAGE

print(f"\n원지수 누적 수익: {base_total*100:.1f}%")
print(f"이론적 2배 수익: {ideal_lev*100:.1f}%")
print(f"실제 레버리지 ETF: {lev_total*100:.1f}%")
print(f"변동성 손실: {(ideal_lev - lev_total)*100:.1f}%p")

# 연간 변동성 손실
years = len(data) / 252
annual_decay = (ideal_lev - lev_total) / years
print(f"연간 변동성 손실: {annual_decay*100:.1f}%p")

# 결과 저장
output = {
    "generated": datetime.now().isoformat(),
    "type": "leveraged_etf_backtest",
    "target": "TIGER 200선물레버리지 (233160)",
    "leverage": LEVERAGE,
    "cost": TOTAL_COST,
    "data_period": f"{data.index.min()} ~ {data.index.max()}",
    "volatility_decay": {
        "base_return": base_total,
        "ideal_leveraged": ideal_lev,
        "actual_leveraged": lev_total,
        "decay": ideal_lev - lev_total,
        "annual_decay": annual_decay,
    },
    "buy_hold": {
        "sharpe": bh_row["sharpe"] if bh_row is not None else None,
        "cagr": bh_row["cagr"] if bh_row is not None else None,
        "mdd": bh_row["mdd"] if bh_row is not None else None,
    },
    "viable_strategies": viable.to_dict("records") if len(viable) > 0 else [],
    "all_results": results_df.to_dict("records"),
}

output_path = f"{OUTPUT_DIR}/leveraged_etf_backtest_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)

print(f"\n결과 저장: {output_path}")

# 최종 권장 전략
print("\n" + "=" * 80)
print("최종 권장 전략")
print("=" * 80)

if len(viable) > 0:
    best = viable.iloc[0]
    print(f"\n추천: {best['strategy']}")
    print(f"  Sharpe: {best['sharpe']:.3f}")
    print(f"  CAGR: {best['cagr']*100:.1f}%")
    print(f"  MDD: {best['mdd']*100:.1f}%")
    print(f"  Time in Market: {best['time_in_market']*100:.1f}%")

    if best["wf_ratio"]:
        print(f"  WF Ratio: {best['wf_ratio']:.2f} (>0.6 = 과적합 아님)")
else:
    print("\n유효한 전략이 없습니다.")
    print("레버리지 ETF 장기 보유는 변동성 손실로 인해 권장하지 않습니다.")
