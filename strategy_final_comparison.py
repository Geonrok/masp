"""
전략 최종 비교 분석
1. 전체 테스트 전략 중 최선 확인
2. 시장 구간별 (상승/횡보/하락) 성과 분석
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, r"E:\투자\Multi-Asset Strategy Platform")

from kosdaq_live_trading_validation import (
    TripleV5Strategy,
    TripleVolStrategy,
    TripleADXStrategy,
    CombBestStrategy,
    LiveTradingConfig,
    LiveTradingValidator,
)

# 데이터 로드
data_path = Path("E:/투자/data/kosdaq_futures")
data = pd.read_parquet(data_path / "kosdaq150_futures_ohlcv_clean.parquet")

print("=" * 80)
print("전략 최종 비교 및 시장 구간별 분석")
print("=" * 80)

# =============================================================================
# 1. 시장 구간 분류
# =============================================================================
print("\n" + "=" * 80)
print("1. 시장 구간 분류 (KOSDAQ 150 지수 기준)")
print("=" * 80)

# 6개월 이동평균으로 추세 판단
data["MA120"] = data["Close"].rolling(120).mean()
data["MA60"] = data["Close"].rolling(60).mean()
data["Returns_60d"] = data["Close"].pct_change(60)


# 시장 구간 정의
def classify_market(row):
    if pd.isna(row["MA120"]) or pd.isna(row["Returns_60d"]):
        return "unknown"

    ret = row["Returns_60d"]
    close = row["Close"]
    ma120 = row["MA120"]

    # 상승장: 60일 수익률 > 10% 또는 가격 > MA120 * 1.05
    if ret > 0.10 or close > ma120 * 1.05:
        return "bull"
    # 하락장: 60일 수익률 < -10% 또는 가격 < MA120 * 0.95
    elif ret < -0.10 or close < ma120 * 0.95:
        return "bear"
    # 횡보장
    else:
        return "sideways"


data["Market"] = data.apply(classify_market, axis=1)

# 구간별 일수
market_counts = data["Market"].value_counts()
print(f"\n시장 구간별 일수:")
for market, count in market_counts.items():
    pct = count / len(data) * 100
    print(f"  {market:>10}: {count:>5}일 ({pct:.1f}%)")

# 주요 시장 구간 기간
print(f"\n주요 시장 구간:")
market_periods = {
    "2010-2011": "횡보/하락",
    "2012-2013": "상승장",
    "2014-2015": "상승 후 급락",
    "2016-2017": "상승장",
    "2018": "하락장",
    "2019": "회복",
    "2020 상반기": "코로나 급락",
    "2020 하반기": "급등장",
    "2021": "고점 후 하락",
    "2022": "하락장 (금리인상)",
    "2023": "회복/횡보",
    "2024-2025": "횡보/변동",
}
for period, desc in market_periods.items():
    print(f"  {period}: {desc}")

# =============================================================================
# 2. 전략 목록 (검증 통과 전략들)
# =============================================================================
strategies = [
    # 권장 전략
    TripleV5Strategy(14, 38, 14, 78, 20, "both"),
    TripleV5Strategy(14, 33, 14, 73, 20, "both"),
    TripleV5Strategy(14, 35, 14, 75, 20, "both"),
    # 거래량 필터
    TripleVolStrategy(14, 35, 75, 0.8, "both"),
    TripleVolStrategy(14, 38, 78, 0.8, "both"),
    # ADX 필터
    TripleADXStrategy(14, 35, 75, 14, 25, "both"),
    TripleADXStrategy(14, 38, 78, 14, 25, "both"),
    # 복합 지표
    CombBestStrategy(14, 35, 75, 70, 20, "both"),
]


# =============================================================================
# 3. 백테스트 함수
# =============================================================================
def backtest_strategy(strategy, df, commission=0.0001, slippage=0.001):
    """백테스트 실행"""
    signals = strategy.generate_signals(df)

    if len(signals) < 5:
        return {
            "sharpe": 0,
            "return": 0,
            "trades": len(signals),
            "win_rate": 0,
            "mdd": 0,
            "cagr": 0,
            "avg_trade": 0,
            "profit_factor": 0,
        }

    daily_returns = df["Close"].pct_change()
    position = pd.Series(0, index=df.index)

    for sig in signals:
        if sig.date in position.index:
            position.loc[sig.date :] = sig.signal

    position = position.shift(1).fillna(0)
    trades = position.diff().abs()
    costs = trades * (commission + slippage)
    strat_returns = position * daily_returns - costs

    # 성과 지표
    if strat_returns.std() > 0:
        sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    total_return = (1 + strat_returns).prod() - 1

    # CAGR
    years = len(df) / 252
    if years > 0 and total_return > -1:
        cagr = (1 + total_return) ** (1 / years) - 1
    else:
        cagr = 0

    # MDD
    cum_returns = (1 + strat_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    mdd = drawdown.min()

    # 승률 및 Profit Factor
    trade_returns = strat_returns[position != 0]
    if len(trade_returns) > 0:
        win_rate = (trade_returns > 0).sum() / len(trade_returns)
        wins = trade_returns[trade_returns > 0].sum()
        losses = abs(trade_returns[trade_returns < 0].sum())
        profit_factor = wins / losses if losses > 0 else 999
        avg_trade = trade_returns.mean() * 100
    else:
        win_rate = 0
        profit_factor = 0
        avg_trade = 0

    return {
        "sharpe": sharpe,
        "return": total_return,
        "cagr": cagr,
        "trades": len(signals),
        "win_rate": win_rate,
        "mdd": mdd,
        "avg_trade": avg_trade,
        "profit_factor": profit_factor,
    }


# =============================================================================
# 4. 전체 기간 성과 비교
# =============================================================================
print("\n" + "=" * 80)
print("2. 전체 기간 성과 비교 (2010-2026)")
print("=" * 80)

full_results = []
for strategy in strategies:
    result = backtest_strategy(strategy, data)
    result["name"] = strategy.name
    full_results.append(result)

# 정렬 (Sharpe 기준)
full_results.sort(key=lambda x: x["sharpe"], reverse=True)

print(f"\n{'순위':<4} {'전략':<35} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8} {'승률':>8}")
print("-" * 80)

for i, r in enumerate(full_results, 1):
    print(
        f"{i:<4} {r['name']:<35} {r['sharpe']:>8.3f} {r['cagr']*100:>7.1f}% {r['mdd']*100:>7.1f}% {r['win_rate']*100:>7.1f}%"
    )

# =============================================================================
# 5. 시장 구간별 성과 분석
# =============================================================================
print("\n" + "=" * 80)
print("3. 시장 구간별 성과 분석")
print("=" * 80)

# 시장 구간별 데이터 분리
bull_data = data[data["Market"] == "bull"]
bear_data = data[data["Market"] == "bear"]
sideways_data = data[data["Market"] == "sideways"]

print(f"\n분석 대상:")
print(f"  상승장: {len(bull_data)}일")
print(f"  하락장: {len(bear_data)}일")
print(f"  횡보장: {len(sideways_data)}일")

# 구간별 성과
market_results = {}
for market_name, market_data in [
    ("상승장", bull_data),
    ("하락장", bear_data),
    ("횡보장", sideways_data),
]:
    if len(market_data) < 100:
        continue

    market_results[market_name] = []
    for strategy in strategies:
        result = backtest_strategy(strategy, market_data)
        result["name"] = strategy.name
        market_results[market_name].append(result)

# 결과 출력
for market_name in ["상승장", "하락장", "횡보장"]:
    if market_name not in market_results:
        continue

    print(f"\n### {market_name} 성과 ###")
    print(f"{'전략':<35} {'Sharpe':>8} {'수익률':>10} {'승률':>8} {'거래수':>8}")
    print("-" * 75)

    results = sorted(
        market_results[market_name], key=lambda x: x["sharpe"], reverse=True
    )
    for r in results:
        print(
            f"{r['name']:<35} {r['sharpe']:>8.3f} {r['return']*100:>9.1f}% {r['win_rate']*100:>7.1f}% {r['trades']:>8}"
        )

# =============================================================================
# 6. 추천 전략의 구간별 상세 분석
# =============================================================================
print("\n" + "=" * 80)
print("4. 추천 전략 (TripleV5_14_38) 상세 분석")
print("=" * 80)

best_strategy = TripleV5Strategy(14, 38, 14, 78, 20, "both")

print(f"\n### 시장 구간별 성과 ###")
print(
    f"{'구간':<12} {'Sharpe':>10} {'수익률':>12} {'승률':>10} {'거래수':>10} {'평균손익':>12}"
)
print("-" * 70)

for market_name, market_data in [
    ("상승장", bull_data),
    ("하락장", bear_data),
    ("횡보장", sideways_data),
    ("전체", data),
]:
    if len(market_data) < 50:
        continue
    result = backtest_strategy(best_strategy, market_data)
    print(
        f"{market_name:<12} {result['sharpe']:>10.3f} {result['return']*100:>11.1f}% {result['win_rate']*100:>9.1f}% {result['trades']:>10} {result['avg_trade']:>11.3f}%"
    )

# =============================================================================
# 7. 연도별 상세 분석
# =============================================================================
print("\n" + "=" * 80)
print("5. 추천 전략 연도별 성과")
print("=" * 80)

print(
    f"\n{'연도':<6} {'시장상황':<12} {'Sharpe':>10} {'수익률':>12} {'승률':>10} {'거래수':>8}"
)
print("-" * 65)

year_market = {
    2010: "횡보",
    2011: "하락",
    2012: "상승",
    2013: "상승",
    2014: "상승",
    2015: "급락",
    2016: "상승",
    2017: "상승",
    2018: "하락",
    2019: "회복",
    2020: "급등락",
    2021: "하락",
    2022: "하락",
    2023: "회복",
    2024: "횡보",
    2025: "횡보",
}

yearly_results = []
for year in range(2010, 2027):
    year_data = data[data.index.year == year]
    if len(year_data) < 50:
        continue

    result = backtest_strategy(best_strategy, year_data)
    market = year_market.get(year, "?")

    yearly_results.append(
        {
            "year": year,
            "market": market,
            "sharpe": result["sharpe"],
            "return": result["return"],
            "win_rate": result["win_rate"],
            "trades": result["trades"],
        }
    )

    print(
        f"{year:<6} {market:<12} {result['sharpe']:>10.3f} {result['return']*100:>11.1f}% {result['win_rate']*100:>9.1f}% {result['trades']:>8}"
    )

# 시장 유형별 평균
print("\n### 시장 유형별 평균 성과 ###")
market_types = {}
for yr in yearly_results:
    m = yr["market"]
    if m not in market_types:
        market_types[m] = []
    market_types[m].append(yr)

print(f"{'시장유형':<10} {'평균Sharpe':>12} {'평균수익률':>12} {'연도수':>8}")
print("-" * 45)
for mtype, years in market_types.items():
    avg_sharpe = np.mean([y["sharpe"] for y in years])
    avg_return = np.mean([y["return"] for y in years])
    print(f"{mtype:<10} {avg_sharpe:>12.3f} {avg_return*100:>11.1f}% {len(years):>8}")

# =============================================================================
# 8. 최종 결론
# =============================================================================
print("\n" + "=" * 80)
print("6. 최종 결론")
print("=" * 80)

# 구간별 최고 전략
print("\n### 시장 구간별 최적 전략 ###")
for market_name in ["상승장", "하락장", "횡보장"]:
    if market_name in market_results:
        best = max(market_results[market_name], key=lambda x: x["sharpe"])
        print(f"  {market_name}: {best['name']} (Sharpe: {best['sharpe']:.3f})")

# TripleV5_14_38 강점/약점
print("\n### TripleV5_14_38_14_78_20 분석 ###")
bull_result = backtest_strategy(best_strategy, bull_data)
bear_result = backtest_strategy(best_strategy, bear_data)
sideways_result = backtest_strategy(best_strategy, sideways_data)

print(f"""
강점:
  - 상승장 Sharpe: {bull_result['sharpe']:.3f}
  - 하락장 Sharpe: {bear_result['sharpe']:.3f}
  - 횡보장 Sharpe: {sideways_result['sharpe']:.3f}
""")

# 최고 구간 판단
best_market = max(
    [
        ("상승장", bull_result["sharpe"]),
        ("하락장", bear_result["sharpe"]),
        ("횡보장", sideways_result["sharpe"]),
    ],
    key=lambda x: x[1],
)

worst_market = min(
    [
        ("상승장", bull_result["sharpe"]),
        ("하락장", bear_result["sharpe"]),
        ("횡보장", sideways_result["sharpe"]),
    ],
    key=lambda x: x[1],
)

print(f"결론:")
print(f"  - 가장 강한 구간: {best_market[0]} (Sharpe {best_market[1]:.3f})")
print(f"  - 가장 약한 구간: {worst_market[0]} (Sharpe {worst_market[1]:.3f})")

# 전체 비교에서 순위
rank = next(i for i, r in enumerate(full_results, 1) if "V5_14_38" in r["name"])
print(f"  - 전체 {len(full_results)}개 전략 중 순위: {rank}위")

if rank == 1:
    print(f"\n[결론] TripleV5_14_38_14_78_20이 테스트된 전략 중 최선입니다.")
else:
    best_overall = full_results[0]
    print(
        f"\n[결론] 전체 최고 전략은 {best_overall['name']} (Sharpe {best_overall['sharpe']:.3f})"
    )
    print(
        f"       TripleV5_14_38은 {rank}위이지만 Out-of-Sample 성과 기준으로는 권장됩니다."
    )
