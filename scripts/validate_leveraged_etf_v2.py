# -*- coding: utf-8 -*-
"""
레버리지 ETF 전략 실거래 검증 v2
================================

v1 문제점: Walk-Forward에서 신호 생성이 test 구간에서 안됨
v2 수정: 전체 데이터로 신호 생성 후 구간별로 성과 측정

검증 기준:
- Full Sample Sharpe > 0.8
- Full Sample MDD > -35%
- 70/30 WF Test Sharpe > 0.5
- 최근 1년 수익 > 0

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

TOTAL_COST = 0.0013  # 0.13% 왕복


def load_data():
    """데이터 로드."""
    data = {}

    # ETF
    data["lev2x"] = pd.read_parquet(f"{DATA_DIR}/122630_KODEX_레버리지.parquet")
    data["lev2x"].columns = [c.lower() for c in data["lev2x"].columns]

    data["inv2x"] = pd.read_parquet(f"{DATA_DIR}/252670_KODEX_200선물인버스2X.parquet")
    data["inv2x"].columns = [c.lower() for c in data["inv2x"].columns]

    # VIX
    data["vix"] = pd.read_parquet(f"{DATA_DIR}/VIX.parquet")
    data["vix"].columns = [c.lower() for c in data["vix"].columns]

    # KOSPI200
    data["kospi200"] = pd.read_parquet(f"{KOSPI_DIR}/kospi200_daily_yf.parquet")
    data["kospi200"].columns = [c.lower() for c in data["kospi200"].columns]

    # 외국인
    try:
        foreign = pd.read_csv(
            f"{INVESTOR_DIR}/all_stocks_foreign_sum.csv", encoding="utf-8-sig"
        )
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


def backtest_with_cost(signal, etf_close, cost=TOTAL_COST):
    """간단한 백테스트 with 비용."""
    position = signal.shift(1).fillna(0)  # T-1 신호로 T일 진입
    ret = etf_close.pct_change()

    strat_ret = position * ret

    # 비용
    turnover = position.diff().abs().fillna(0)
    costs = turnover * cost / 2
    strat_ret = strat_ret - costs

    equity = (1 + strat_ret.fillna(0)).cumprod()

    return equity, strat_ret, position


def calc_metrics(equity, strat_ret, position):
    """성과 지표."""
    if len(equity) < 100:
        return None

    years = len(equity) / 252
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    vol = strat_ret.std() * np.sqrt(252)
    sharpe = (strat_ret.mean() * 252) / vol if vol > 0 else 0

    peak = equity.cummax()
    dd = (equity - peak) / peak
    mdd = dd.min()

    trades = position.diff().abs().fillna(0).sum() / 2
    trades_py = trades / years if years > 0 else 0

    exposure = (position > 0).mean()

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "trades_py": trades_py,
        "exposure": exposure,
    }


def walk_forward_7030(signal, etf_close):
    """70/30 Walk-Forward."""
    n = len(signal.dropna())
    cut = int(n * 0.7)

    idx = signal.dropna().index
    train_idx = idx[:cut]
    test_idx = idx[cut:]

    if len(test_idx) < 50:
        return {"train_sharpe": 0, "test_sharpe": 0}

    # Train
    train_signal = signal.reindex(train_idx)
    train_close = etf_close.reindex(train_idx)
    train_eq, train_ret, _ = backtest_with_cost(train_signal, train_close)
    train_sharpe = (
        (train_ret.mean() * 252) / (train_ret.std() * np.sqrt(252))
        if train_ret.std() > 0
        else 0
    )

    # Test
    test_signal = signal.reindex(test_idx)
    test_close = etf_close.reindex(test_idx)
    test_eq, test_ret, _ = backtest_with_cost(test_signal, test_close)
    test_sharpe = (
        (test_ret.mean() * 252) / (test_ret.std() * np.sqrt(252))
        if test_ret.std() > 0
        else 0
    )

    return {
        "train_sharpe": train_sharpe,
        "test_sharpe": test_sharpe,
        "train_period": f'{train_idx[0].strftime("%Y-%m-%d")} ~ {train_idx[-1].strftime("%Y-%m-%d")}',
        "test_period": f'{test_idx[0].strftime("%Y-%m-%d")} ~ {test_idx[-1].strftime("%Y-%m-%d")}',
    }


def recent_1y_return(equity):
    """최근 1년 수익률."""
    if len(equity) < 252:
        return 0
    recent = equity.iloc[-252:]
    return recent.iloc[-1] / recent.iloc[0] - 1


# ============ 전략들 ============


def strat_1_conservative(data):
    """보수적 레버리지."""
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]
    foreign = data["foreign"]

    if foreign is None:
        return None

    common = kospi.index.intersection(vix.index).intersection(foreign.index)
    kospi = kospi.reindex(common).ffill()
    vix = vix.reindex(common).ffill()
    foreign = foreign.reindex(common).ffill()

    sma100 = kospi.rolling(100, min_periods=100).mean()
    vix_sma20 = vix.rolling(20, min_periods=20).mean()
    foreign_30d = foreign.rolling(30, min_periods=30).sum()

    signal = (
        (kospi > sma100) & (vix < 25) & (vix < vix_sma20) & (foreign_30d > 0)
    ).astype(float)

    return signal


def strat_2_vix_level(data):
    """VIX 레벨."""
    vix = data["vix"]["close"]
    vix_sma20 = vix.rolling(20, min_periods=20).mean()

    signal = ((vix < 20) & (vix < vix_sma20)).astype(float)
    return signal


def strat_3_trend_vix(data):
    """추세 + VIX."""
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]

    common = kospi.index.intersection(vix.index)
    kospi = kospi.reindex(common).ffill()
    vix = vix.reindex(common).ffill()

    sma50 = kospi.rolling(50, min_periods=50).mean()
    vix_sma20 = vix.rolling(20, min_periods=20).mean()

    signal = ((kospi > sma50) & (vix < vix_sma20)).astype(float)
    return signal


def strat_4_foreign_trend(data):
    """외국인 + 추세."""
    kospi = data["kospi200"]["close"]
    foreign = data["foreign"]

    if foreign is None:
        return None

    common = kospi.index.intersection(foreign.index)
    kospi = kospi.reindex(common).ffill()
    foreign = foreign.reindex(common).ffill()

    sma100 = kospi.rolling(100, min_periods=100).mean()
    foreign_30d = foreign.rolling(30, min_periods=30).sum()

    signal = ((kospi > sma100) & (foreign_30d > 0)).astype(float)
    return signal


def strat_5_dual_mom(data):
    """듀얼 모멘텀."""
    kospi = data["kospi200"]["close"]

    mom_12m = kospi.pct_change(252)
    mom_1m = kospi.pct_change(21)

    signal = ((mom_12m > 0) & (mom_1m > 0)).astype(float)
    return signal


def strat_6_triple_filter(data):
    """
    전략 6: 트리플 필터

    - KOSPI200 > SMA(100)
    - VIX < SMA(20)
    - 최근 5일 수익률 > 0
    """
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]

    common = kospi.index.intersection(vix.index)
    kospi = kospi.reindex(common).ffill()
    vix = vix.reindex(common).ffill()

    sma100 = kospi.rolling(100, min_periods=100).mean()
    vix_sma20 = vix.rolling(20, min_periods=20).mean()
    mom5 = kospi.pct_change(5)

    signal = ((kospi > sma100) & (vix < vix_sma20) & (mom5 > 0)).astype(float)
    return signal


def strat_7_vix_crash(data):
    """
    전략 7: VIX 급락 후 레버리지

    - VIX 5일 변화 < -15%
    - KOSPI200 > SMA(50)
    """
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]

    common = kospi.index.intersection(vix.index)
    kospi = kospi.reindex(common).ffill()
    vix = vix.reindex(common).ffill()

    sma50 = kospi.rolling(50, min_periods=50).mean()
    vix_chg5 = vix.pct_change(5)

    signal = ((vix_chg5 < -0.15) & (kospi > sma50)).astype(float)
    return signal


def strat_8_steady_uptrend(data):
    """
    전략 8: 안정적 상승추세

    - KOSPI200 > SMA(20) > SMA(50) > SMA(100)
    - 20일 변동성 < 20%
    """
    kospi = data["kospi200"]["close"]

    sma20 = kospi.rolling(20, min_periods=20).mean()
    sma50 = kospi.rolling(50, min_periods=50).mean()
    sma100 = kospi.rolling(100, min_periods=100).mean()

    ret = kospi.pct_change()
    vol20 = ret.rolling(20).std() * np.sqrt(252)

    signal = (
        (kospi > sma20) & (sma20 > sma50) & (sma50 > sma100) & (vol20 < 0.20)
    ).astype(float)

    return signal


def strat_9_foreign_vix_combo(data):
    """
    전략 9: 외국인 + VIX 콤보

    - 외국인 20일 순매수 > 0
    - VIX < 22
    - KOSPI200 > SMA(50)
    """
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]
    foreign = data["foreign"]

    if foreign is None:
        return None

    common = kospi.index.intersection(vix.index).intersection(foreign.index)
    kospi = kospi.reindex(common).ffill()
    vix = vix.reindex(common).ffill()
    foreign = foreign.reindex(common).ffill()

    sma50 = kospi.rolling(50, min_periods=50).mean()
    foreign_20d = foreign.rolling(20, min_periods=20).sum()

    signal = ((foreign_20d > 0) & (vix < 22) & (kospi > sma50)).astype(float)

    return signal


def strat_10_momentum_filter(data):
    """
    전략 10: 모멘텀 필터

    - 20일 수익률 > 0
    - 60일 수익률 > 0
    - VIX < SMA(20)
    """
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]

    common = kospi.index.intersection(vix.index)
    kospi = kospi.reindex(common).ffill()
    vix = vix.reindex(common).ffill()

    mom20 = kospi.pct_change(20)
    mom60 = kospi.pct_change(60)
    vix_sma20 = vix.rolling(20, min_periods=20).mean()

    signal = ((mom20 > 0) & (mom60 > 0) & (vix < vix_sma20)).astype(float)
    return signal


def main():
    print("=" * 80)
    print("레버리지 ETF 전략 실거래 검증 v2")
    print("=" * 80)
    print()

    data = load_data()
    lev2x = data["lev2x"]["close"]

    print(
        f'레버리지 ETF: {lev2x.index[0].strftime("%Y-%m-%d")} ~ {lev2x.index[-1].strftime("%Y-%m-%d")}'
    )
    print(f"거래비용: {TOTAL_COST*100:.2f}% 왕복")
    print()

    # B&H 벤치마크
    bh_ret = lev2x.pct_change()
    bh_eq = (1 + bh_ret.fillna(0)).cumprod()
    bh_years = len(bh_eq) / 252
    bh_cagr = bh_eq.iloc[-1] ** (1 / bh_years) - 1
    bh_sharpe = (bh_ret.mean() * 252) / (bh_ret.std() * np.sqrt(252))
    bh_mdd = ((bh_eq - bh_eq.cummax()) / bh_eq.cummax()).min()

    print(
        f"B&H: CAGR {bh_cagr*100:.1f}%, Sharpe {bh_sharpe:.3f}, MDD {bh_mdd*100:.1f}%"
    )
    print()

    strategies = {
        "1_Conservative": strat_1_conservative,
        "2_VIX_Level": strat_2_vix_level,
        "3_Trend_VIX": strat_3_trend_vix,
        "4_Foreign_Trend": strat_4_foreign_trend,
        "5_Dual_Momentum": strat_5_dual_mom,
        "6_Triple_Filter": strat_6_triple_filter,
        "7_VIX_Crash": strat_7_vix_crash,
        "8_Steady_Uptrend": strat_8_steady_uptrend,
        "9_Foreign_VIX": strat_9_foreign_vix_combo,
        "10_Momentum_Filter": strat_10_momentum_filter,
    }

    results = []

    for name, func in strategies.items():
        print(f"--- {name} ---")

        try:
            signal = func(data)
            if signal is None:
                print("  [SKIP] 데이터 부족")
                continue

            # 레버리지 ETF 인덱스에 맞춤
            common = signal.index.intersection(lev2x.index)
            signal = signal.reindex(common)
            etf_close = lev2x.reindex(common)

            # Full sample 백테스트
            equity, strat_ret, position = backtest_with_cost(signal, etf_close)
            metrics = calc_metrics(equity, strat_ret, position)

            if metrics is None:
                print("  [SKIP] 기간 부족")
                continue

            # Walk-Forward
            wf = walk_forward_7030(signal, etf_close)

            # 최근 1년
            r1y = recent_1y_return(equity)

            print(f'  CAGR: {metrics["cagr"]*100:.1f}%')
            print(f'  Sharpe: {metrics["sharpe"]:.3f}')
            print(f'  MDD: {metrics["mdd"]*100:.1f}%')
            print(f'  Exposure: {metrics["exposure"]*100:.1f}%')
            print(
                f'  WF Train: {wf["train_sharpe"]:.3f}, Test: {wf["test_sharpe"]:.3f}'
            )
            print(f"  최근 1년: {r1y*100:.1f}%")

            # 판정
            passed = (
                metrics["sharpe"] > 0.8
                and metrics["mdd"] > -0.35
                and wf["test_sharpe"] > 0.5
                and r1y > 0
            )
            verdict = "PASS" if passed else "FAIL"
            print(f"  판정: {verdict}")

            results.append(
                {
                    "strategy": name,
                    "cagr": metrics["cagr"],
                    "sharpe": metrics["sharpe"],
                    "mdd": metrics["mdd"],
                    "exposure": metrics["exposure"],
                    "trades_py": metrics["trades_py"],
                    "wf_train": wf["train_sharpe"],
                    "wf_test": wf["test_sharpe"],
                    "recent_1y": r1y,
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
        print(
            df[
                ["strategy", "cagr", "sharpe", "mdd", "wf_test", "recent_1y", "verdict"]
            ].to_string(index=False)
        )

        passed = df[df["verdict"] == "PASS"]
        print()

        if len(passed) > 0:
            print(f"실거래 가능 전략: {len(passed)}개")
            for _, row in passed.iterrows():
                print(
                    f'  {row["strategy"]}: Sharpe {row["sharpe"]:.3f}, MDD {row["mdd"]*100:.1f}%, WF {row["wf_test"]:.3f}'
                )
        else:
            print("실거래 가능 전략 없음")
            print()

            # 기준 완화 검토
            print("기준 완화 시 후보:")
            for _, row in df.head(3).iterrows():
                issues = []
                if row["sharpe"] <= 0.8:
                    issues.append(f'Sharpe {row["sharpe"]:.2f}')
                if row["mdd"] <= -0.35:
                    issues.append(f'MDD {row["mdd"]*100:.0f}%')
                if row["wf_test"] <= 0.5:
                    issues.append(f'WF {row["wf_test"]:.2f}')
                if row["recent_1y"] <= 0:
                    issues.append(f'1Y {row["recent_1y"]*100:.0f}%')

                print(f'  {row["strategy"]}: {", ".join(issues)}')

        # 저장
        output = {
            "generated": datetime.now().isoformat(),
            "type": "leveraged_etf_v2_validation",
            "benchmark": {"cagr": bh_cagr, "sharpe": bh_sharpe, "mdd": bh_mdd},
            "results": results,
        }

        path = f"{OUTPUT_DIR}/leveraged_etf_v2_validation.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
