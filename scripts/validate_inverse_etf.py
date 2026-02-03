# -*- coding: utf-8 -*-
"""
인버스 ETF (KODEX 200선물인버스2X) 전략 검증
=============================================

대상: KODEX 200선물인버스2X (252670) - 숏 2x

인버스 ETF 특성:
- 시장 하락 시 수익
- 장기 보유 시 변동성 드래그로 손실
- 단기 전략에 적합

Author: Claude Code
Date: 2026-01-31
"""

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = "E:/투자/data/leveraged_etf"
KOSPI_DIR = "E:/투자/data/kospi_futures"
INVESTOR_DIR = "E:/투자/data/kr_stock/investor_trading"
OUTPUT_DIR = "E:/투자/Multi-Asset Strategy Platform/logs"

TOTAL_COST = 0.0013


def load_data():
    """데이터 로드."""
    data = {}

    data["inv2x"] = pd.read_parquet(f"{DATA_DIR}/252670_KODEX_200선물인버스2X.parquet")
    data["inv2x"].columns = [c.lower() for c in data["inv2x"].columns]

    data["vix"] = pd.read_parquet(f"{DATA_DIR}/VIX.parquet")
    data["vix"].columns = [c.lower() for c in data["vix"].columns]

    data["kospi200"] = pd.read_parquet(f"{KOSPI_DIR}/kospi200_daily_yf.parquet")
    data["kospi200"].columns = [c.lower() for c in data["kospi200"].columns]

    try:
        foreign = pd.read_csv(
            f"{INVESTOR_DIR}/all_stocks_foreign_sum.csv", encoding="utf-8-sig"
        )
        foreign["날짜"] = pd.to_datetime(foreign["날짜"])
        foreign = foreign.set_index("날짜")
        data["foreign"] = foreign["외국인합계"]
    except:
        data["foreign"] = None

    for key in data:
        if data[key] is not None and hasattr(data[key], "index"):
            if hasattr(data[key].index, "tz") and data[key].index.tz is not None:
                data[key].index = data[key].index.tz_localize(None)

    return data


def backtest(signal, etf_close, cost=TOTAL_COST):
    """백테스트."""
    position = signal.shift(1).fillna(0)
    ret = etf_close.pct_change()
    strat_ret = position * ret

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

    exposure = (position > 0).mean()

    return {"cagr": cagr, "sharpe": sharpe, "mdd": mdd, "exposure": exposure}


def walk_forward(signal, etf_close):
    """70/30 Walk-Forward."""
    n = len(signal.dropna())
    cut = int(n * 0.7)
    idx = signal.dropna().index

    train_signal = signal.reindex(idx[:cut])
    train_close = etf_close.reindex(idx[:cut])
    _, train_ret, _ = backtest(train_signal, train_close)
    train_sharpe = (
        (train_ret.mean() * 252) / (train_ret.std() * np.sqrt(252))
        if train_ret.std() > 0
        else 0
    )

    test_signal = signal.reindex(idx[cut:])
    test_close = etf_close.reindex(idx[cut:])
    _, test_ret, _ = backtest(test_signal, test_close)
    test_sharpe = (
        (test_ret.mean() * 252) / (test_ret.std() * np.sqrt(252))
        if test_ret.std() > 0
        else 0
    )

    return {"train_sharpe": train_sharpe, "test_sharpe": test_sharpe}


# ============ 인버스 전략 ============


def inv_1_vix_spike(data):
    """
    인버스 전략 1: VIX 스파이크

    - VIX > SMA(20) * 1.2 (20% 이상 높음)
    - VIX 상승 중
    """
    vix = data["vix"]["close"]
    vix_sma20 = vix.rolling(20, min_periods=20).mean()
    vix_rising = vix > vix.shift(1)

    signal = ((vix > vix_sma20 * 1.2) & vix_rising).astype(float)
    return signal


def inv_2_breakdown(data):
    """
    인버스 전략 2: 지지선 이탈

    - KOSPI200 < 20일 최저가
    - VIX > SMA(20)
    """
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]

    common = kospi.index.intersection(vix.index)
    kospi = kospi.reindex(common).ffill()
    vix = vix.reindex(common).ffill()

    low_20 = kospi.rolling(20, min_periods=20).min()
    vix_sma20 = vix.rolling(20, min_periods=20).mean()

    signal = ((kospi < low_20.shift(1)) & (vix > vix_sma20)).astype(float)
    return signal


def inv_3_downtrend(data):
    """
    인버스 전략 3: 하락 추세

    - KOSPI200 < SMA(50) < SMA(100)
    """
    kospi = data["kospi200"]["close"]

    sma50 = kospi.rolling(50, min_periods=50).mean()
    sma100 = kospi.rolling(100, min_periods=100).mean()

    signal = ((kospi < sma50) & (sma50 < sma100)).astype(float)
    return signal


def inv_4_foreign_selling(data):
    """
    인버스 전략 4: 외국인 순매도

    - 외국인 20일 누적 < 0
    - KOSPI200 < SMA(50)
    """
    kospi = data["kospi200"]["close"]
    foreign = data["foreign"]

    if foreign is None:
        return None

    common = kospi.index.intersection(foreign.index)
    kospi = kospi.reindex(common).ffill()
    foreign = foreign.reindex(common).ffill()

    sma50 = kospi.rolling(50, min_periods=50).mean()
    foreign_20d = foreign.rolling(20, min_periods=20).sum()

    signal = ((foreign_20d < 0) & (kospi < sma50)).astype(float)
    return signal


def inv_5_high_vix_momentum(data):
    """
    인버스 전략 5: 고VIX + 하락 모멘텀

    - VIX > 25
    - KOSPI200 5일 수익률 < -2%
    """
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]

    common = kospi.index.intersection(vix.index)
    kospi = kospi.reindex(common).ffill()
    vix = vix.reindex(common).ffill()

    mom5 = kospi.pct_change(5)

    signal = ((vix > 25) & (mom5 < -0.02)).astype(float)
    return signal


def inv_6_crisis_detector(data):
    """
    인버스 전략 6: 위기 감지

    - VIX > 30
    - VIX 5일 변화 > +30%
    """
    vix = data["vix"]["close"]

    vix_chg5 = vix.pct_change(5)

    signal = ((vix > 30) & (vix_chg5 > 0.30)).astype(float)
    return signal


def main():
    print("=" * 80)
    print("인버스 ETF (KODEX 200선물인버스2X) 전략 검증")
    print("=" * 80)
    print()

    data = load_data()
    inv2x = data["inv2x"]["close"]

    print(
        f'인버스 ETF: {inv2x.index[0].strftime("%Y-%m-%d")} ~ {inv2x.index[-1].strftime("%Y-%m-%d")}'
    )
    print()

    # B&H 벤치마크 (인버스 장기 보유는 손실)
    bh_ret = inv2x.pct_change()
    bh_eq = (1 + bh_ret.fillna(0)).cumprod()
    bh_years = len(bh_eq) / 252
    bh_cagr = bh_eq.iloc[-1] ** (1 / bh_years) - 1
    bh_sharpe = (bh_ret.mean() * 252) / (bh_ret.std() * np.sqrt(252))
    bh_mdd = ((bh_eq - bh_eq.cummax()) / bh_eq.cummax()).min()

    print(
        f"B&H 인버스: CAGR {bh_cagr*100:.1f}%, Sharpe {bh_sharpe:.3f}, MDD {bh_mdd*100:.1f}%"
    )
    print("(인버스 장기 보유는 변동성 드래그로 손실)")
    print()

    strategies = {
        "INV_1_VIX_Spike": inv_1_vix_spike,
        "INV_2_Breakdown": inv_2_breakdown,
        "INV_3_Downtrend": inv_3_downtrend,
        "INV_4_Foreign_Sell": inv_4_foreign_selling,
        "INV_5_HighVIX_Mom": inv_5_high_vix_momentum,
        "INV_6_Crisis": inv_6_crisis_detector,
    }

    results = []

    for name, func in strategies.items():
        print(f"--- {name} ---")

        try:
            signal = func(data)
            if signal is None:
                print("  [SKIP] 데이터 부족")
                continue

            common = signal.index.intersection(inv2x.index)
            signal = signal.reindex(common)
            etf_close = inv2x.reindex(common)

            equity, strat_ret, position = backtest(signal, etf_close)
            metrics = calc_metrics(equity, strat_ret, position)

            if metrics is None:
                print("  [SKIP] 기간 부족")
                continue

            wf = walk_forward(signal, etf_close)

            # 최근 1년
            if len(equity) > 252:
                r1y = equity.iloc[-1] / equity.iloc[-252] - 1
            else:
                r1y = 0

            print(f'  CAGR: {metrics["cagr"]*100:.1f}%')
            print(f'  Sharpe: {metrics["sharpe"]:.3f}')
            print(f'  MDD: {metrics["mdd"]*100:.1f}%')
            print(f'  Exposure: {metrics["exposure"]*100:.1f}%')
            print(f'  WF Test: {wf["test_sharpe"]:.3f}')
            print(f"  최근 1년: {r1y*100:.1f}%")

            # 인버스 전략 기준 (더 엄격)
            # Sharpe > 0.5, MDD > -30%, WF > 0.3
            passed = (
                metrics["sharpe"] > 0.5
                and metrics["mdd"] > -0.30
                and wf["test_sharpe"] > 0.3
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
                    "wf_test": wf["test_sharpe"],
                    "recent_1y": r1y,
                    "verdict": verdict,
                }
            )

        except Exception as e:
            print(f"  [ERROR] {e}")

        print()

    # 요약
    print("=" * 80)
    print("결과 요약")
    print("=" * 80)

    if results:
        df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
        print()
        print(
            df[["strategy", "cagr", "sharpe", "mdd", "wf_test", "verdict"]].to_string(
                index=False
            )
        )

        passed = df[df["verdict"] == "PASS"]
        print()

        if len(passed) > 0:
            print(f"실거래 가능 인버스 전략: {len(passed)}개")
            for _, row in passed.iterrows():
                print(
                    f'  {row["strategy"]}: Sharpe {row["sharpe"]:.3f}, MDD {row["mdd"]*100:.1f}%'
                )
        else:
            print("실거래 가능 인버스 전략 없음")
            print()
            print("인버스 ETF는 단기 헷지용으로만 권장")

        # 저장
        output = {
            "generated": datetime.now().isoformat(),
            "type": "inverse_etf_validation",
            "results": results,
        }
        path = f"{OUTPUT_DIR}/inverse_etf_validation.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
