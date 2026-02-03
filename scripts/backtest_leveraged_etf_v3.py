# -*- coding: utf-8 -*-
"""
코스피 선물 ETF 전략 v3 - 개선된 전략
=====================================

v2 실패 원인:
- 레버리지+인버스 스위칭: MDD -50%+ 과도
- VIX 기반: 신호 빈도 낮거나 타이밍 부정확

v3 개선 방향:
- 레버리지 ONLY (인버스 제외) - 하락장 현금
- VIX 레벨 필터 강화
- 외국인 + 추세 조합 (검증된 전략 적용)

Author: Claude Code
Date: 2026-01-31
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "E:/투자/data/leveraged_etf"
KOSPI_DIR = "E:/투자/data/kospi_futures"
INVESTOR_DIR = "E:/투자/data/kr_stock/investor_trading"
OUTPUT_DIR = "E:/투자/Multi-Asset Strategy Platform/logs"

ROUND_TRIP_COST = 0.0009


def load_data():
    """데이터 로드."""
    data = {}

    # 레버리지 ETF
    data["lev2x"] = pd.read_parquet(f"{DATA_DIR}/122630_KODEX_레버리지.parquet")
    data["lev2x"].columns = [c.lower() for c in data["lev2x"].columns]

    # 현물 ETF
    data["spot"] = pd.read_parquet(f"{DATA_DIR}/069500_KODEX_200.parquet")
    data["spot"].columns = [c.lower() for c in data["spot"].columns]

    # VIX
    data["vix"] = pd.read_parquet(f"{DATA_DIR}/VIX.parquet")
    data["vix"].columns = [c.lower() for c in data["vix"].columns]

    # KOSPI200
    data["kospi200"] = pd.read_parquet(f"{KOSPI_DIR}/kospi200_daily_yf.parquet")
    data["kospi200"].columns = [c.lower() for c in data["kospi200"].columns]

    # 외국인 데이터
    try:
        foreign_file = f"{INVESTOR_DIR}/all_stocks_foreign_sum.csv"
        foreign = pd.read_csv(foreign_file, encoding="utf-8-sig")
        foreign["날짜"] = pd.to_datetime(foreign["날짜"])
        foreign = foreign.set_index("날짜")
        data["foreign"] = foreign["외국인합계"]
    except:
        data["foreign"] = None

    # 인덱스 정규화
    for key in data:
        if data[key] is not None and hasattr(data[key], "index"):
            if hasattr(data[key].index, "tz") and data[key].index.tz is not None:
                data[key].index = data[key].index.tz_localize(None)

    return data


def calculate_metrics(equity: pd.Series, position: pd.Series = None):
    """성과 지표 계산."""
    returns = equity.pct_change().dropna()

    if len(returns) < 252:
        return None

    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(returns) / 252
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0

    peak = equity.cummax()
    dd = (equity - peak) / peak
    mdd = dd.min()

    # 거래 통계
    if position is not None:
        turnover = position.diff().abs().fillna(0)
        trades = (turnover > 0).sum() / 2
        trades_per_year = trades / years if years > 0 else 0
        exposure = (position > 0).mean()
    else:
        trades_per_year = 0
        exposure = 1.0

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "trades_per_year": trades_per_year,
        "exposure": exposure,
    }


def strategy_f_foreign_trend_lev(data, foreign_period=30, sma_period=100):
    """
    전략 F: 외국인+추세 레버리지

    검증된 Foreign_30d_SMA_100 전략을 레버리지 ETF에 적용
    - 외국인 30일 순매수 > 0 AND 종가 > SMA100 → 레버리지 매수
    - 조건 미충족 → 현금
    """
    kospi = data["kospi200"]["close"]
    lev2x = data["lev2x"]["close"]
    foreign = data["foreign"]

    if foreign is None:
        return None, None

    # 공통 인덱스
    common_idx = kospi.index.intersection(lev2x.index).intersection(foreign.index)
    kospi = kospi.reindex(common_idx)
    lev2x = lev2x.reindex(common_idx)
    foreign = foreign.reindex(common_idx)

    # 지표
    sma = kospi.rolling(sma_period).mean()
    foreign_roll = foreign.rolling(foreign_period).sum()

    # 신호 (T-1 데이터로 T일 거래)
    signal = ((foreign_roll > 0) & (kospi > sma)).shift(1)

    # 포지션
    position = signal.astype(float).fillna(0)

    # 수익률 계산
    lev2x_ret = lev2x.pct_change()
    strat_ret = position * lev2x_ret

    # 비용
    turnover = position.diff().abs().fillna(0)
    costs = turnover * ROUND_TRIP_COST / 2
    strat_ret = strat_ret - costs

    # Equity
    equity = (1 + strat_ret.fillna(0)).cumprod()

    return equity, position


def strategy_g_vix_level_lev(data, vix_threshold=20):
    """
    전략 G: VIX 레벨 기반 레버리지

    - VIX < 20 AND VIX < SMA20 → 레버리지 매수
    - VIX >= 20 OR VIX > SMA20 → 현금
    """
    vix = data["vix"]["close"]
    lev2x = data["lev2x"]["close"]

    common_idx = vix.index.intersection(lev2x.index)
    vix = vix.reindex(common_idx)
    lev2x = lev2x.reindex(common_idx)

    # 지표
    vix_sma20 = vix.rolling(20).mean()

    # 신호
    signal = ((vix < vix_threshold) & (vix < vix_sma20)).shift(1)
    position = signal.astype(float).fillna(0)

    # 수익률
    lev2x_ret = lev2x.pct_change()
    strat_ret = position * lev2x_ret

    turnover = position.diff().abs().fillna(0)
    costs = turnover * ROUND_TRIP_COST / 2
    strat_ret = strat_ret - costs

    equity = (1 + strat_ret.fillna(0)).cumprod()

    return equity, position


def strategy_h_trend_vix_combo(data):
    """
    전략 H: 추세 + VIX 콤보

    - KOSPI200 > SMA50 AND VIX < SMA20 → 레버리지
    - 조건 미충족 → 현금
    """
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]
    lev2x = data["lev2x"]["close"]

    common_idx = kospi.index.intersection(vix.index).intersection(lev2x.index)
    kospi = kospi.reindex(common_idx)
    vix = vix.reindex(common_idx)
    lev2x = lev2x.reindex(common_idx)

    # 지표
    kospi_sma50 = kospi.rolling(50).mean()
    vix_sma20 = vix.rolling(20).mean()

    # 신호
    signal = ((kospi > kospi_sma50) & (vix < vix_sma20)).shift(1)
    position = signal.astype(float).fillna(0)

    # 수익률
    lev2x_ret = lev2x.pct_change()
    strat_ret = position * lev2x_ret

    turnover = position.diff().abs().fillna(0)
    costs = turnover * ROUND_TRIP_COST / 2
    strat_ret = strat_ret - costs

    equity = (1 + strat_ret.fillna(0)).cumprod()

    return equity, position


def strategy_i_conservative_lev(data):
    """
    전략 I: 보수적 레버리지

    모든 조건 충족 시에만 레버리지:
    - KOSPI200 > SMA100
    - VIX < 25
    - VIX < SMA20
    - 외국인 30일 순매수 > 0
    """
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]
    lev2x = data["lev2x"]["close"]
    foreign = data["foreign"]

    if foreign is None:
        return None, None

    common_idx = (
        kospi.index.intersection(vix.index)
        .intersection(lev2x.index)
        .intersection(foreign.index)
    )
    kospi = kospi.reindex(common_idx)
    vix = vix.reindex(common_idx)
    lev2x = lev2x.reindex(common_idx)
    foreign = foreign.reindex(common_idx)

    # 지표
    kospi_sma100 = kospi.rolling(100).mean()
    vix_sma20 = vix.rolling(20).mean()
    foreign_30d = foreign.rolling(30).sum()

    # 모든 조건
    signal = (
        (kospi > kospi_sma100) & (vix < 25) & (vix < vix_sma20) & (foreign_30d > 0)
    ).shift(1)

    position = signal.astype(float).fillna(0)

    # 수익률
    lev2x_ret = lev2x.pct_change()
    strat_ret = position * lev2x_ret

    turnover = position.diff().abs().fillna(0)
    costs = turnover * ROUND_TRIP_COST / 2
    strat_ret = strat_ret - costs

    equity = (1 + strat_ret.fillna(0)).cumprod()

    return equity, position


def strategy_j_spot_then_lev(data):
    """
    전략 J: 현물 기본 + 강세장 레버리지

    - 기본: 현물 ETF 보유
    - 강세 조건 (KOSPI>SMA50 AND VIX<SMA20): 현물→레버리지 전환
    """
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]
    spot = data["spot"]["close"]
    lev2x = data["lev2x"]["close"]

    common_idx = (
        kospi.index.intersection(vix.index)
        .intersection(spot.index)
        .intersection(lev2x.index)
    )
    kospi = kospi.reindex(common_idx)
    vix = vix.reindex(common_idx)
    spot = spot.reindex(common_idx)
    lev2x = lev2x.reindex(common_idx)

    # 지표
    kospi_sma50 = kospi.rolling(50).mean()
    vix_sma20 = vix.rolling(20).mean()

    # 강세 신호
    bullish = ((kospi > kospi_sma50) & (vix < vix_sma20)).shift(1)

    # 수익률
    spot_ret = spot.pct_change()
    lev2x_ret = lev2x.pct_change()

    # 강세 시 레버리지, 아니면 현물
    strat_ret = pd.Series(0.0, index=common_idx)
    strat_ret[bullish == True] = lev2x_ret[bullish == True]
    strat_ret[bullish == False] = spot_ret[bullish == False]

    # 비용 (스위칭 시에만)
    switch = bullish.astype(int).diff().abs().fillna(0)
    costs = switch * ROUND_TRIP_COST
    strat_ret = strat_ret - costs

    equity = (1 + strat_ret.fillna(0)).cumprod()
    position = bullish.astype(float).fillna(0) + 1  # 1=현물, 2=레버리지

    return equity, position


def walk_forward(equity: pd.Series, train_ratio=0.7):
    """간단한 Walk-Forward 검증."""
    n = len(equity)
    train_end = int(n * train_ratio)

    train_eq = equity.iloc[:train_end]
    test_eq = equity.iloc[train_end:]

    if len(test_eq) < 50:
        return {"train_sharpe": 0, "test_sharpe": 0, "wf_ratio": 0}

    train_ret = train_eq.pct_change().dropna()
    test_ret = test_eq.pct_change().dropna()

    train_sharpe = (
        (train_ret.mean() * 252) / (train_ret.std() * np.sqrt(252))
        if train_ret.std() > 0
        else 0
    )
    test_sharpe = (
        (test_ret.mean() * 252) / (test_ret.std() * np.sqrt(252))
        if test_ret.std() > 0
        else 0
    )

    wf_ratio = test_sharpe / train_sharpe if train_sharpe != 0 else 0

    return {
        "train_sharpe": train_sharpe,
        "test_sharpe": test_sharpe,
        "wf_ratio": wf_ratio,
    }


def main():
    print("=" * 80)
    print("코스피 선물 ETF 전략 v3 - 개선된 레버리지 전략")
    print("=" * 80)
    print()

    data = load_data()
    print(f'레버리지 ETF: {len(data["lev2x"])}일')
    print(f'현물 ETF: {len(data["spot"])}일')
    print(f'외국인 데이터: {"있음" if data["foreign"] is not None else "없음"}')
    print()

    strategies = {
        "F_Foreign_Trend_Lev": strategy_f_foreign_trend_lev,
        "G_VIX_Level_Lev": strategy_g_vix_level_lev,
        "H_Trend_VIX_Combo": strategy_h_trend_vix_combo,
        "I_Conservative_Lev": strategy_i_conservative_lev,
        "J_Spot_Then_Lev": strategy_j_spot_then_lev,
    }

    results = []

    # Buy & Hold 벤치마크
    print("--- Buy & Hold (레버리지) ---")
    lev_ret = data["lev2x"]["close"].pct_change()
    lev_eq = (1 + lev_ret.fillna(0)).cumprod()
    bh_metrics = calculate_metrics(lev_eq)
    print(f'  CAGR: {bh_metrics["cagr"]*100:.1f}%')
    print(f'  Sharpe: {bh_metrics["sharpe"]:.3f}')
    print(f'  MDD: {bh_metrics["mdd"]*100:.1f}%')
    print()

    for name, func in strategies.items():
        print(f"--- {name} ---")

        try:
            equity, position = func(data)

            if equity is None:
                print("  [SKIP] 데이터 부족")
                continue

            metrics = calculate_metrics(equity, position)
            if metrics is None:
                print("  [SKIP] 기간 부족")
                continue

            wf = walk_forward(equity)

            print(f'  CAGR: {metrics["cagr"]*100:.1f}%')
            print(f'  Sharpe: {metrics["sharpe"]:.3f}')
            print(f'  MDD: {metrics["mdd"]*100:.1f}%')
            print(f'  거래/년: {metrics["trades_per_year"]:.1f}')
            print(f'  Exposure: {metrics["exposure"]*100:.1f}%')
            print(f'  WF Test Sharpe: {wf["test_sharpe"]:.3f}')
            print(f'  WF Ratio: {wf["wf_ratio"]:.2f}')

            # 판정
            passed = (
                wf["test_sharpe"] > 0.5
                and metrics["mdd"] > -0.40
                and metrics["sharpe"] > 0.5
            )
            verdict = "PASS" if passed else "FAIL"
            print(f"  결론: {verdict}")

            results.append(
                {
                    "strategy": name,
                    "cagr": metrics["cagr"],
                    "sharpe": metrics["sharpe"],
                    "mdd": metrics["mdd"],
                    "trades_per_year": metrics["trades_per_year"],
                    "exposure": metrics["exposure"],
                    "wf_test_sharpe": wf["test_sharpe"],
                    "wf_ratio": wf["wf_ratio"],
                    "verdict": verdict,
                }
            )

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback

            traceback.print_exc()

        print()

    # 요약
    print("=" * 80)
    print("결과 요약")
    print("=" * 80)

    if results:
        df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
        print()
        print(df.to_string(index=False))

        passed = df[df["verdict"] == "PASS"]
        print()
        if len(passed) > 0:
            print(f"통과 전략: {len(passed)}개")
            for _, row in passed.iterrows():
                print(
                    f'  - {row["strategy"]}: Sharpe {row["sharpe"]:.3f}, MDD {row["mdd"]*100:.1f}%'
                )
        else:
            print("통과 전략 없음 - 기준 완화 필요")

            # 차선책 제안
            best = df.iloc[0]
            print(f'\n차선책: {best["strategy"]}')
            print(f'  Sharpe: {best["sharpe"]:.3f}')
            print(f'  MDD: {best["mdd"]*100:.1f}%')
            print(f'  WF Test Sharpe: {best["wf_test_sharpe"]:.3f}')

        # 저장
        output = {
            "generated": datetime.now().isoformat(),
            "type": "leveraged_etf_v3",
            "benchmark": bh_metrics,
            "results": results,
        }
        output_path = f"{OUTPUT_DIR}/leveraged_etf_v3_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n저장: {output_path}")


if __name__ == "__main__":
    main()
