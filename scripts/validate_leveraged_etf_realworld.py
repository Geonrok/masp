# -*- coding: utf-8 -*-
"""
레버리지 ETF 전략 실거래 시뮬레이션 검증
=========================================

검증 항목:
1. 실제 ETF 가격 기반 백테스트 (지수 × 2 아님)
2. 체결 가정: 익일 시가 체결 (갭 리스크 반영)
3. 거래 비용: 0.15% 왕복 (보수적)
4. 슬리피지: 0.1% (레버리지 ETF 특성)
5. Walk-Forward 11-fold 검증
6. 위기 구간 성과 분석
7. 최근 1년 성과 확인 (레짐 변화)

대상 ETF:
- KODEX 레버리지 (122630) - 롱 2x
- KODEX 200선물인버스2X (252670) - 숏 2x

Author: Claude Code
Date: 2026-01-31
"""

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 경로
DATA_DIR = "E:/투자/data/leveraged_etf"
KOSPI_DIR = "E:/투자/data/kospi_futures"
INVESTOR_DIR = "E:/투자/data/kr_stock/investor_trading"
OUTPUT_DIR = "E:/투자/Multi-Asset Strategy Platform/logs"

# 실거래 비용 (보수적)
COMMISSION = 0.00015 * 2  # 매수+매도
SLIPPAGE = 0.001  # 0.1% (레버리지 ETF 갭/스프레드)
TOTAL_COST = COMMISSION + SLIPPAGE  # 약 0.13%


def load_all_data():
    """모든 데이터 로드."""
    data = {}

    # ETF 데이터
    etf_map = {
        "lev2x": "122630_KODEX_레버리지",
        "inv2x": "252670_KODEX_200선물인버스2X",
        "spot": "069500_KODEX_200",
    }

    for key, filename in etf_map.items():
        path = f"{DATA_DIR}/{filename}.parquet"
        try:
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            data[key] = df
        except Exception as e:
            print(f"[WARN] {key}: {e}")

    # VIX
    try:
        vix = pd.read_parquet(f"{DATA_DIR}/VIX.parquet")
        vix.columns = [c.lower() for c in vix.columns]
        if hasattr(vix.index, "tz") and vix.index.tz is not None:
            vix.index = vix.index.tz_localize(None)
        data["vix"] = vix
    except:
        pass

    # KOSPI200
    try:
        kospi = pd.read_parquet(f"{KOSPI_DIR}/kospi200_daily_yf.parquet")
        kospi.columns = [c.lower() for c in kospi.columns]
        if hasattr(kospi.index, "tz") and kospi.index.tz is not None:
            kospi.index = kospi.index.tz_localize(None)
        data["kospi200"] = kospi
    except:
        pass

    # 외국인 데이터
    try:
        foreign = pd.read_csv(
            f"{INVESTOR_DIR}/all_stocks_foreign_sum.csv", encoding="utf-8-sig"
        )
        foreign["날짜"] = pd.to_datetime(foreign["날짜"])
        foreign = foreign.set_index("날짜")
        data["foreign"] = foreign["외국인합계"]
    except:
        data["foreign"] = None

    return data


def realworld_backtest(
    signal: pd.Series,
    etf_prices: pd.DataFrame,
    cost_per_trade: float = TOTAL_COST,
    entry_on: str = "open",  # 'open' or 'close'
) -> dict:
    """
    실거래 시뮬레이션 백테스트.

    - signal: T일 종가 기준 신호 (1=매수, 0=현금)
    - 체결: T+1일 시가 (갭 리스크 반영)
    - 수익률: T+1일 시가 → T+2일 시가 (실제 보유 수익)
    """
    # 데이터 정렬
    prices = etf_prices[["open", "close"]].copy()
    signal = signal.reindex(prices.index).fillna(0)

    # T일 신호 → T+1일 시가 체결
    position = signal.shift(1).fillna(0)  # T+1일 포지션

    if entry_on == "open":
        # 수익률: 시가 to 시가 (익일 시가 체결 가정)
        ret = prices["open"].pct_change().shift(-1)  # T+1 open → T+2 open
    else:
        # 종가 체결 가정
        ret = prices["close"].pct_change()

    # 전략 수익률
    strat_ret = position * ret

    # 거래 비용
    turnover = position.diff().abs().fillna(0)
    costs = turnover * cost_per_trade
    strat_ret = strat_ret - costs

    # Equity curve
    equity = (1 + strat_ret.fillna(0)).cumprod()

    # 지표 계산
    valid_eq = equity.dropna()
    if len(valid_eq) < 252:
        return None

    total_ret = valid_eq.iloc[-1] / valid_eq.iloc[0] - 1
    years = len(valid_eq) / 252
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    daily_ret = strat_ret.dropna()
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / vol if vol > 0 else 0

    peak = valid_eq.cummax()
    dd = (valid_eq - peak) / peak
    mdd = dd.min()

    # 거래 통계
    trades = (turnover > 0).sum() / 2
    trades_per_year = trades / years if years > 0 else 0

    # 승률
    winning_days = (daily_ret[position.shift(1) > 0] > 0).mean()

    # Exposure
    exposure = (position > 0).mean()

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "trades_per_year": trades_per_year,
        "win_rate": winning_days,
        "exposure": exposure,
        "equity": equity,
        "position": position,
        "daily_ret": strat_ret,
    }


def walk_forward_11fold(signal: pd.Series, etf_prices: pd.DataFrame) -> dict:
    """11-fold expanding window Walk-Forward 검증."""
    common_idx = signal.dropna().index.intersection(etf_prices.index)
    n = len(common_idx)

    if n < 500:
        return {"avg_test_sharpe": 0, "consistency": 0, "folds": []}

    fold_size = n // 11
    folds = []

    for i in range(1, 11):  # 10 test folds
        train_end_idx = i * fold_size
        test_end_idx = (i + 1) * fold_size

        train_dates = common_idx[:train_end_idx]
        test_dates = common_idx[train_end_idx:test_end_idx]

        if len(test_dates) < 20:
            continue

        # Test 구간 백테스트
        test_signal = signal.reindex(test_dates)
        test_prices = etf_prices.reindex(test_dates)

        result = realworld_backtest(test_signal, test_prices)
        if result is None:
            continue

        folds.append(
            {
                "fold": i,
                "test_start": test_dates[0].strftime("%Y-%m-%d"),
                "test_end": test_dates[-1].strftime("%Y-%m-%d"),
                "test_sharpe": result["sharpe"],
                "test_return": result["cagr"],
            }
        )

    if not folds:
        return {"avg_test_sharpe": 0, "consistency": 0, "folds": []}

    sharpes = [f["test_sharpe"] for f in folds]
    avg_sharpe = np.mean(sharpes)
    consistency = sum(1 for s in sharpes if s > 0) / len(sharpes)

    return {
        "avg_test_sharpe": avg_sharpe,
        "consistency": consistency,
        "n_folds": len(folds),
        "folds": folds,
    }


def crisis_analysis(equity: pd.Series) -> dict:
    """위기 구간 성과 분석."""
    crises = {
        "covid_2020": ("2020-02-15", "2020-04-30"),
        "rate_hike_2022": ("2022-01-01", "2022-12-31"),
        "recent_1y": (
            (
                equity.index[-252].strftime("%Y-%m-%d")
                if len(equity) > 252
                else equity.index[0].strftime("%Y-%m-%d")
            ),
            equity.index[-1].strftime("%Y-%m-%d"),
        ),
    }

    results = {}
    for name, (start, end) in crises.items():
        try:
            sub = equity[(equity.index >= start) & (equity.index <= end)]
            if len(sub) < 5:
                continue
            ret = sub.iloc[-1] / sub.iloc[0] - 1
            peak = sub.cummax()
            mdd = ((sub - peak) / peak).min()
            results[name] = {"return": ret, "mdd": mdd}
        except:
            pass

    return results


# ============ 전략 정의 ============


def strategy_1_conservative_lev(data) -> pd.Series:
    """
    전략 1: I_Conservative_Lev (보수적 레버리지)

    조건 (모두 충족):
    - KOSPI200 > SMA(100)
    - VIX < 25
    - VIX < SMA(20)
    - 외국인 30일 순매수 > 0
    """
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]
    foreign = data["foreign"]

    if foreign is None:
        return None

    common = kospi.index.intersection(vix.index).intersection(foreign.index)
    kospi = kospi.reindex(common)
    vix = vix.reindex(common)
    foreign = foreign.reindex(common)

    kospi_sma100 = kospi.rolling(100).mean()
    vix_sma20 = vix.rolling(20).mean()
    foreign_30d = foreign.rolling(30).sum()

    signal = (kospi > kospi_sma100) & (vix < 25) & (vix < vix_sma20) & (foreign_30d > 0)

    return signal.astype(float)


def strategy_2_vix_level(data) -> pd.Series:
    """
    전략 2: G_VIX_Level_Lev (VIX 레벨 기반)

    조건:
    - VIX < 20
    - VIX < SMA(20)
    """
    vix = data["vix"]["close"]

    vix_sma20 = vix.rolling(20).mean()
    signal = (vix < 20) & (vix < vix_sma20)

    return signal.astype(float)


def strategy_3_trend_vix(data) -> pd.Series:
    """
    전략 3: H_Trend_VIX_Combo (추세 + VIX)

    조건:
    - KOSPI200 > SMA(50)
    - VIX < SMA(20)
    """
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]

    common = kospi.index.intersection(vix.index)
    kospi = kospi.reindex(common)
    vix = vix.reindex(common)

    kospi_sma50 = kospi.rolling(50).mean()
    vix_sma20 = vix.rolling(20).mean()

    signal = (kospi > kospi_sma50) & (vix < vix_sma20)

    return signal.astype(float)


def strategy_4_foreign_trend(data) -> pd.Series:
    """
    전략 4: F_Foreign_Trend_Lev (외국인 + 추세)

    조건:
    - 외국인 30일 순매수 > 0
    - KOSPI200 > SMA(100)
    """
    kospi = data["kospi200"]["close"]
    foreign = data["foreign"]

    if foreign is None:
        return None

    common = kospi.index.intersection(foreign.index)
    kospi = kospi.reindex(common)
    foreign = foreign.reindex(common)

    kospi_sma100 = kospi.rolling(100).mean()
    foreign_30d = foreign.rolling(30).sum()

    signal = (kospi > kospi_sma100) & (foreign_30d > 0)

    return signal.astype(float)


# ============ 추가 전략 탐색 ============


def strategy_5_dual_momentum(data) -> pd.Series:
    """
    전략 5: 듀얼 모멘텀

    조건:
    - KOSPI200 12개월 수익률 > 0
    - KOSPI200 1개월 수익률 > 0
    """
    kospi = data["kospi200"]["close"]

    mom_12m = kospi.pct_change(252)
    mom_1m = kospi.pct_change(21)

    signal = (mom_12m > 0) & (mom_1m > 0)

    return signal.astype(float)


def strategy_6_vix_mean_reversion(data) -> pd.Series:
    """
    전략 6: VIX 평균회귀

    조건:
    - VIX가 SMA(50) 대비 20% 이상 높았다가 하락 시작
    - VIX가 전일 대비 하락
    """
    vix = data["vix"]["close"]

    vix_sma50 = vix.rolling(50).mean()
    vix_ratio = vix / vix_sma50

    # VIX 스파이크 후 하락 시작
    vix_spike = vix_ratio > 1.2
    vix_declining = vix < vix.shift(1)

    signal = vix_spike.shift(1) & vix_declining & (vix < vix_sma50)

    return signal.astype(float)


def strategy_7_breakout(data) -> pd.Series:
    """
    전략 7: 돌파 전략

    조건:
    - KOSPI200 > 20일 최고가
    - VIX < SMA(20)
    """
    kospi = data["kospi200"]["close"]
    vix = data["vix"]["close"]

    common = kospi.index.intersection(vix.index)
    kospi = kospi.reindex(common)
    vix = vix.reindex(common)

    high_20 = kospi.rolling(20).max()
    vix_sma20 = vix.rolling(20).mean()

    signal = (kospi > high_20.shift(1)) & (vix < vix_sma20)

    return signal.astype(float)


def strategy_8_low_vol(data) -> pd.Series:
    """
    전략 8: 저변동성 레버리지

    조건:
    - KOSPI200 20일 변동성 < 15%
    - KOSPI200 > SMA(50)
    """
    kospi = data["kospi200"]["close"]

    ret = kospi.pct_change()
    vol_20d = ret.rolling(20).std() * np.sqrt(252)
    sma50 = kospi.rolling(50).mean()

    signal = (vol_20d < 0.15) & (kospi > sma50)

    return signal.astype(float)


def main():
    print("=" * 80)
    print("레버리지 ETF 전략 실거래 시뮬레이션 검증")
    print("=" * 80)
    print()
    print(f"거래비용: {TOTAL_COST*100:.2f}% (왕복)")
    print(f"체결가정: 익일 시가")
    print()

    # 데이터 로드
    data = load_all_data()

    if "lev2x" not in data:
        print("[ERROR] 레버리지 ETF 데이터 없음")
        return

    lev2x = data["lev2x"]
    print(
        f'레버리지 ETF 기간: {lev2x.index[0].strftime("%Y-%m-%d")} ~ {lev2x.index[-1].strftime("%Y-%m-%d")}'
    )
    print(f'외국인 데이터: {"있음" if data.get("foreign") is not None else "없음"}')
    print()

    # 전략 정의
    strategies = {
        "1_Conservative_Lev": strategy_1_conservative_lev,
        "2_VIX_Level": strategy_2_vix_level,
        "3_Trend_VIX": strategy_3_trend_vix,
        "4_Foreign_Trend": strategy_4_foreign_trend,
        "5_Dual_Momentum": strategy_5_dual_momentum,
        "6_VIX_MeanRev": strategy_6_vix_mean_reversion,
        "7_Breakout": strategy_7_breakout,
        "8_Low_Vol": strategy_8_low_vol,
    }

    # Buy & Hold 벤치마크
    print("--- Buy & Hold (레버리지) ---")
    bh_ret = lev2x["close"].pct_change()
    bh_eq = (1 + bh_ret.fillna(0)).cumprod()

    bh_years = len(bh_eq) / 252
    bh_cagr = (bh_eq.iloc[-1] ** (1 / bh_years) - 1) if bh_years > 0 else 0
    bh_sharpe = (bh_ret.mean() * 252) / (bh_ret.std() * np.sqrt(252))
    bh_mdd = ((bh_eq - bh_eq.cummax()) / bh_eq.cummax()).min()

    print(f"  CAGR: {bh_cagr*100:.1f}%")
    print(f"  Sharpe: {bh_sharpe:.3f}")
    print(f"  MDD: {bh_mdd*100:.1f}%")
    print()

    results = []

    for name, func in strategies.items():
        print(f"--- {name} ---")

        try:
            signal = func(data)

            if signal is None:
                print("  [SKIP] 데이터 부족")
                continue

            # 레버리지 ETF 인덱스에 맞춤
            signal = signal.reindex(lev2x.index)

            # 실거래 백테스트
            result = realworld_backtest(signal, lev2x)

            if result is None:
                print("  [SKIP] 기간 부족")
                continue

            # Walk-Forward 검증
            wf = walk_forward_11fold(signal, lev2x)

            # 위기 구간 분석
            crisis = crisis_analysis(result["equity"])

            # 출력
            print(f'  CAGR: {result["cagr"]*100:.1f}%')
            print(f'  Sharpe: {result["sharpe"]:.3f}')
            print(f'  MDD: {result["mdd"]*100:.1f}%')
            print(f'  거래/년: {result["trades_per_year"]:.1f}')
            print(f'  승률: {result["win_rate"]*100:.1f}%')
            print(f'  Exposure: {result["exposure"]*100:.1f}%')
            print(f'  WF Test Sharpe: {wf["avg_test_sharpe"]:.3f}')
            print(f'  WF Consistency: {wf["consistency"]*100:.1f}%')

            if "recent_1y" in crisis:
                print(f'  최근 1년: {crisis["recent_1y"]["return"]*100:.1f}%')

            # 실거래 적합성 판정
            # 기준: Sharpe > 0.8, MDD > -35%, WF Sharpe > 0.5, Consistency > 50%
            passed = (
                result["sharpe"] > 0.8
                and result["mdd"] > -0.35
                and wf["avg_test_sharpe"] > 0.5
                and wf["consistency"] > 0.5
            )

            verdict = "PASS" if passed else "FAIL"
            print(f"  실거래 판정: {verdict}")

            results.append(
                {
                    "strategy": name,
                    "cagr": result["cagr"],
                    "sharpe": result["sharpe"],
                    "mdd": result["mdd"],
                    "trades_per_year": result["trades_per_year"],
                    "win_rate": result["win_rate"],
                    "exposure": result["exposure"],
                    "wf_test_sharpe": wf["avg_test_sharpe"],
                    "wf_consistency": wf["consistency"],
                    "crisis": crisis,
                    "verdict": verdict,
                }
            )

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback

            traceback.print_exc()

        print()

    # 결과 요약
    print("=" * 80)
    print("실거래 검증 결과 요약")
    print("=" * 80)
    print()

    if results:
        df = pd.DataFrame(results)
        df_display = df[
            [
                "strategy",
                "cagr",
                "sharpe",
                "mdd",
                "wf_test_sharpe",
                "wf_consistency",
                "verdict",
            ]
        ]
        df_display = df_display.sort_values("sharpe", ascending=False)

        print(df_display.to_string(index=False))
        print()

        passed = df[df["verdict"] == "PASS"]

        if len(passed) > 0:
            print(f"실거래 가능 전략: {len(passed)}개")
            for _, row in passed.iterrows():
                print(f'  - {row["strategy"]}')
                print(f'    Sharpe: {row["sharpe"]:.3f}, MDD: {row["mdd"]*100:.1f}%')
                print(
                    f'    WF Sharpe: {row["wf_test_sharpe"]:.3f}, Consistency: {row["wf_consistency"]*100:.0f}%'
                )
        else:
            print("실거래 가능 전략 없음")
            print()
            # 차선책
            best = df.iloc[df["sharpe"].idxmax()] if len(df) > 0 else None
            if best is not None:
                print(f'차선책: {best["strategy"]}')
                print(f'  Sharpe: {best["sharpe"]:.3f}, MDD: {best["mdd"]*100:.1f}%')
                print(f"  개선 필요: ", end="")
                issues = []
                if best["sharpe"] <= 0.8:
                    issues.append("Sharpe < 0.8")
                if best["mdd"] <= -0.35:
                    issues.append("MDD > -35%")
                if best["wf_test_sharpe"] <= 0.5:
                    issues.append("WF Sharpe < 0.5")
                print(", ".join(issues) if issues else "없음")

        # 저장
        output = {
            "generated": datetime.now().isoformat(),
            "type": "leveraged_etf_realworld_validation",
            "cost": TOTAL_COST,
            "benchmark": {
                "cagr": bh_cagr,
                "sharpe": bh_sharpe,
                "mdd": bh_mdd,
            },
            "results": results,
        }

        output_path = f"{OUTPUT_DIR}/leveraged_etf_realworld_validation.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n저장: {output_path}")

    print()
    print("검증 완료")


if __name__ == "__main__":
    main()
