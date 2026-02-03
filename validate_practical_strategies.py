# -*- coding: utf-8 -*-
"""
실거래 가능 전략 검증
- 슬리피지 고려
- 거래 비용 현실화
- 거래량 제약
- 과적합 검증
- 워크포워드 분석
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

KIWOOM_DATA_PATH = "E:/투자/data/kiwoom_investor"
RESULTS_PATH = "E:/투자/Multi-Asset Strategy Platform"


def load_merged_data():
    """키움 merged 데이터 로드"""
    merged_files = [
        f for f in os.listdir(KIWOOM_DATA_PATH) if f.endswith("_merged.csv")
    ]
    all_data = {}

    for f in merged_files:
        ticker = f.replace("_merged.csv", "")
        try:
            df = pd.read_csv(os.path.join(KIWOOM_DATA_PATH, f))
            df["dt"] = pd.to_datetime(df["dt"])
            df = df.sort_values("dt").reset_index(drop=True)
            df = df[df["dt"] >= "2018-01-01"].copy()
            df = df.dropna(subset=["chg_qty", "wght", "cur_prc"])

            if len(df) > 100:
                all_data[ticker] = df
        except:
            pass

    return all_data


def calculate_realistic_metrics(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100_000_000,
    fee_rate: float = 0.0015,  # 0.15% (세금+수수료)
    slippage: float = 0.001,  # 0.1% 슬리피지
    max_position_pct: float = 0.1,
):  # 거래량의 10% 이하
    """
    현실적인 백테스트 메트릭 계산
    - 거래비용 0.15% (매수: 0.015% 수수료, 매도: 0.015% 수수료 + 0.23% 세금 ≈ 0.25%)
    - 슬리피지 0.1%
    - 거래량 제약 (일 거래량의 10% 이하)
    """
    capital = initial_capital
    position = 0
    shares = 0
    trades = []
    equity_curve = [initial_capital]

    for i in range(1, len(df)):
        current = df.iloc[i]
        price = current["cur_prc"]
        volume = current["trde_qty"]
        signal = signals.iloc[i]

        # 현재 포트폴리오 가치
        if position == 1:
            capital + shares * price
        else:
            pass

        if position == 0 and signal == 1:
            # 매수 - 슬리피지 적용
            buy_price = price * (1 + slippage)

            # 거래량 제약
            max_shares = int(volume * max_position_pct)
            invest_amount = capital * 0.95
            desired_shares = int(invest_amount / buy_price)
            actual_shares = min(desired_shares, max_shares)

            if actual_shares > 0:
                cost = actual_shares * buy_price * (1 + fee_rate)
                capital -= cost
                shares = actual_shares
                position = 1
                trades.append(
                    {
                        "type": "buy",
                        "price": buy_price,
                        "shares": shares,
                        "date": current["dt"],
                        "volume_pct": shares / volume * 100 if volume > 0 else 0,
                    }
                )

        elif position == 1 and signal == 0:
            # 매도 - 슬리피지 적용
            sell_price = price * (1 - slippage)

            # 거래량 제약 (대량 청산 시 분할 매도 필요할 수 있음)
            max_sell = int(volume * max_position_pct)
            if shares > max_sell:
                # 일부만 매도 가능 - 단순화를 위해 전량 매도 가정
                pass

            revenue = shares * sell_price * (1 - fee_rate * 1.5)  # 매도세 포함
            capital += revenue
            trades.append(
                {
                    "type": "sell",
                    "price": sell_price,
                    "shares": shares,
                    "date": current["dt"],
                }
            )
            position = 0
            shares = 0

        if position == 1:
            equity_curve.append(capital + shares * price)
        else:
            equity_curve.append(capital)

    # 마지막 포지션 청산
    if position == 1 and shares > 0:
        final_price = df["cur_prc"].iloc[-1] * (1 - slippage)
        capital += shares * final_price * (1 - fee_rate * 1.5)

    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital * 100

    # 최대 낙폭
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd = drawdown.min() * 100

    # 승률 계산
    buy_trades = [t for t in trades if t["type"] == "buy"]
    sell_trades = [t for t in trades if t["type"] == "sell"]

    win_count = 0
    total_profit = 0
    total_loss = 0

    for i in range(min(len(buy_trades), len(sell_trades))):
        profit = (sell_trades[i]["price"] - buy_trades[i]["price"]) / buy_trades[i][
            "price"
        ]
        if profit > 0:
            win_count += 1
            total_profit += profit
        else:
            total_loss += abs(profit)

    win_rate = win_count / len(buy_trades) * 100 if len(buy_trades) > 0 else 0

    # 손익비
    avg_profit = total_profit / win_count if win_count > 0 else 0
    avg_loss = (
        total_loss / (len(buy_trades) - win_count)
        if (len(buy_trades) - win_count) > 0
        else 1
    )
    profit_factor = avg_profit / avg_loss if avg_loss > 0 else 0

    # 샤프비율
    if len(equity_curve) > 1:
        returns = pd.Series(equity_curve).pct_change().dropna()
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # 연간 수익률 (CAGR)
    days = (df["dt"].iloc[-1] - df["dt"].iloc[0]).days
    years = days / 365.25
    if years > 0 and final_capital > 0:
        cagr = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
    else:
        cagr = 0

    return {
        "total_return_pct": total_return,
        "cagr_pct": cagr,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": len(buy_trades),
        "avg_volume_pct": (
            np.mean([t.get("volume_pct", 0) for t in buy_trades]) if buy_trades else 0
        ),
    }


def walk_forward_validation(df: pd.DataFrame, strategy_func, n_splits: int = 5):
    """
    워크포워드 검증
    - 데이터를 n개 구간으로 나눔
    - 각 구간에서 이전 데이터로 학습, 다음 구간으로 테스트
    """
    results = []
    split_size = len(df) // n_splits

    for i in range(1, n_splits):
        train_end = i * split_size
        test_end = (i + 1) * split_size if i < n_splits - 1 else len(df)

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        if len(test_df) < 50:
            continue

        # 전략 신호 생성
        try:
            signals = strategy_func(test_df)
            metrics = calculate_realistic_metrics(test_df, signals)
            metrics["period"] = i
            metrics["train_size"] = len(train_df)
            metrics["test_size"] = len(test_df)
            results.append(metrics)
        except:
            pass

    return results


# 검증할 전략 함수들


def strategy_trend_filter_foreign(
    df: pd.DataFrame, ma_period: int = 100, foreign_period: int = 10
):
    """추세 필터 + 외국인"""
    df = df.copy()
    df["ma"] = df["cur_prc"].rolling(ma_period).mean()
    df["foreign_sum"] = df["chg_qty"].rolling(foreign_period).sum()

    signals = pd.Series(0, index=df.index)
    signals[(df["cur_prc"] > df["ma"]) & (df["foreign_sum"] > 0)] = 1

    return signals


def strategy_momentum_foreign(
    df: pd.DataFrame, mom_period: int = 20, foreign_period: int = 10
):
    """모멘텀 + 외국인"""
    df = df.copy()
    df["momentum"] = df["cur_prc"].pct_change(mom_period)
    df["foreign_sum"] = df["chg_qty"].rolling(foreign_period).sum()

    signals = pd.Series(0, index=df.index)
    signals[(df["momentum"] > 0) & (df["foreign_sum"] > 0)] = 1

    return signals


def strategy_ma_crossover_foreign(
    df: pd.DataFrame, short: int = 20, long: int = 60, foreign_period: int = 5
):
    """이평선 크로스 + 외국인"""
    df = df.copy()
    df["ma_short"] = df["cur_prc"].rolling(short).mean()
    df["ma_long"] = df["cur_prc"].rolling(long).mean()
    df["foreign_sum"] = df["chg_qty"].rolling(foreign_period).sum()

    signals = pd.Series(0, index=df.index)
    signals[(df["ma_short"] > df["ma_long"]) & (df["foreign_sum"] > 0)] = 1

    return signals


def strategy_breakout_foreign(
    df: pd.DataFrame, breakout_period: int = 20, foreign_period: int = 5
):
    """돌파 + 외국인"""
    df = df.copy()
    df["high_max"] = df["high_pric"].rolling(breakout_period).max()
    df["foreign_sum"] = df["chg_qty"].rolling(foreign_period).sum()

    signals = pd.Series(0, index=df.index)
    signals[(df["cur_prc"] > df["high_max"].shift(1)) & (df["foreign_sum"] > 0)] = 1

    return signals


def strategy_volatility_breakout(
    df: pd.DataFrame, k: float = 0.5, foreign_period: int = 3
):
    """변동성 돌파 + 외국인"""
    df = df.copy()
    df["range"] = df["high_pric"] - df["low_pric"]
    df["target"] = df["open_pric"] + df["range"].shift(1) * k
    df["foreign_sum"] = df["chg_qty"].rolling(foreign_period).sum()

    signals = pd.Series(0, index=df.index)
    signals[(df["cur_prc"] > df["target"]) & (df["foreign_sum"] > 0)] = 1

    return signals


def strategy_foreign_ratio_ma(df: pd.DataFrame, ratio_period: int = 10):
    """외국인 비중 이평선"""
    df = df.copy()
    df["wght_ma"] = df["wght"].rolling(ratio_period).mean()

    signals = pd.Series(0, index=df.index)
    signals[(df["wght"] > df["wght_ma"]) & (df["chg_qty"] > 0)] = 1

    return signals


def strategy_dual_momentum_foreign(
    df: pd.DataFrame, short: int = 10, long: int = 30, foreign_period: int = 5
):
    """듀얼 모멘텀 + 외국인"""
    df = df.copy()
    df["mom_short"] = df["cur_prc"].pct_change(short)
    df["mom_long"] = df["cur_prc"].pct_change(long)
    df["foreign_sum"] = df["chg_qty"].rolling(foreign_period).sum()

    signals = pd.Series(0, index=df.index)
    signals[(df["mom_short"] > 0) & (df["mom_long"] > 0) & (df["foreign_sum"] > 0)] = 1

    return signals


def strategy_rsi_foreign(
    df: pd.DataFrame, rsi_period: int = 14, oversold: int = 30, foreign_period: int = 5
):
    """RSI + 외국인"""
    df = df.copy()

    delta = df["cur_prc"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["foreign_sum"] = df["chg_qty"].rolling(foreign_period).sum()

    signals = pd.Series(0, index=df.index)
    signals[(df["rsi"] < oversold) & (df["foreign_sum"] > 0)] = 1

    return signals


def run_validation():
    """전략 검증 실행"""
    print("=" * 80)
    print("실거래 가능 전략 검증")
    print("=" * 80)

    print("\n검증 기준:")
    print("  - 거래비용: 0.15% (수수료 + 세금)")
    print("  - 슬리피지: 0.1%")
    print("  - 거래량 제약: 일 거래량의 10% 이하")
    print("  - 워크포워드 검증: 5분할")

    print("\n[1] 데이터 로드...")
    all_data = load_merged_data()
    print(f"  종목 수: {len(all_data)}개")

    # 전략 정의
    strategies = [
        (
            "TrendFilter_MA100_F10",
            lambda df: strategy_trend_filter_foreign(df, 100, 10),
        ),
        ("TrendFilter_MA50_F5", lambda df: strategy_trend_filter_foreign(df, 50, 5)),
        ("Momentum_20_F10", lambda df: strategy_momentum_foreign(df, 20, 10)),
        ("Momentum_10_F5", lambda df: strategy_momentum_foreign(df, 10, 5)),
        ("MACross_20_60_F5", lambda df: strategy_ma_crossover_foreign(df, 20, 60, 5)),
        ("MACross_10_30_F5", lambda df: strategy_ma_crossover_foreign(df, 10, 30, 5)),
        ("Breakout_20_F5", lambda df: strategy_breakout_foreign(df, 20, 5)),
        ("Breakout_50_F10", lambda df: strategy_breakout_foreign(df, 50, 10)),
        ("VolBreak_0.5_F3", lambda df: strategy_volatility_breakout(df, 0.5, 3)),
        ("VolBreak_0.6_F5", lambda df: strategy_volatility_breakout(df, 0.6, 5)),
        ("ForeignRatioMA_10", lambda df: strategy_foreign_ratio_ma(df, 10)),
        ("ForeignRatioMA_20", lambda df: strategy_foreign_ratio_ma(df, 20)),
        ("DualMom_10_30_F5", lambda df: strategy_dual_momentum_foreign(df, 10, 30, 5)),
        (
            "DualMom_20_60_F10",
            lambda df: strategy_dual_momentum_foreign(df, 20, 60, 10),
        ),
        ("RSI_14_30_F5", lambda df: strategy_rsi_foreign(df, 14, 30, 5)),
        ("RSI_7_25_F10", lambda df: strategy_rsi_foreign(df, 7, 25, 10)),
    ]

    print(f"  전략 수: {len(strategies)}개")

    all_results = []

    print("\n[2] 전체 기간 현실적 백테스트...")
    for ticker, df in all_data.items():
        name = df["name"].iloc[0] if "name" in df.columns else ticker

        for strategy_name, strategy_func in strategies:
            try:
                signals = strategy_func(df)
                metrics = calculate_realistic_metrics(df, signals)
                metrics["ticker"] = ticker
                metrics["name"] = name
                metrics["strategy"] = strategy_name
                metrics["type"] = "full_period"
                all_results.append(metrics)
            except:
                pass

    print(f"  완료: {len(all_results)}개 테스트")

    # 워크포워드 검증
    print("\n[3] 워크포워드 검증...")
    wf_results = []

    for ticker, df in all_data.items():
        name = df["name"].iloc[0] if "name" in df.columns else ticker

        for strategy_name, strategy_func in strategies:
            try:
                wf = walk_forward_validation(df, strategy_func, n_splits=5)
                if wf:
                    avg_return = np.mean([r["total_return_pct"] for r in wf])
                    avg_sharpe = np.mean([r["sharpe_ratio"] for r in wf])
                    avg_win_rate = np.mean([r["win_rate"] for r in wf])
                    consistency = len(
                        [r for r in wf if r["total_return_pct"] > 0]
                    ) / len(wf)

                    wf_results.append(
                        {
                            "ticker": ticker,
                            "name": name,
                            "strategy": strategy_name,
                            "avg_return_pct": avg_return,
                            "avg_sharpe": avg_sharpe,
                            "avg_win_rate": avg_win_rate,
                            "consistency": consistency,
                            "n_periods": len(wf),
                        }
                    )
            except:
                pass

    print(f"  완료: {len(wf_results)}개 워크포워드 테스트")

    # 결과 분석
    print("\n" + "=" * 80)
    print("검증 결과")
    print("=" * 80)

    if all_results:
        results_df = pd.DataFrame(all_results)

        # 실거래 가능 기준 필터링
        # 1. 수익률 > 0
        # 2. 최대낙폭 < -50%
        # 3. 승률 > 40%
        # 4. 거래 횟수 > 10
        # 5. 샤프비율 > 0.5

        practical = results_df[
            (results_df["total_return_pct"] > 0)
            & (results_df["max_drawdown_pct"] > -50)
            & (results_df["win_rate"] > 40)
            & (results_df["num_trades"] > 10)
            & (results_df["sharpe_ratio"] > 0.5)
        ].copy()

        print("\n[전체 기간 테스트]")
        print(f"  총 테스트: {len(results_df)}개")
        print(
            f"  실거래 가능 기준 통과: {len(practical)}개 ({len(practical)/len(results_df)*100:.1f}%)"
        )

        if len(practical) > 0:
            practical = practical.sort_values("sharpe_ratio", ascending=False)

            print("\n[실거래 가능 전략 - 샤프비율 상위 20개]")
            print("-" * 100)
            print(
                f"{'종목':<12} {'전략':<25} {'수익률':>10} {'CAGR':>8} {'MDD':>8} {'승률':>6} {'샤프':>6} {'거래':>5}"
            )
            print("-" * 100)

            for _, row in practical.head(20).iterrows():
                print(
                    f"{row['name'][:10]:<12} {row['strategy']:<25} "
                    f"{row['total_return_pct']:>9.1f}% {row['cagr_pct']:>7.1f}% "
                    f"{row['max_drawdown_pct']:>7.1f}% {row['win_rate']:>5.1f}% "
                    f"{row['sharpe_ratio']:>6.2f} {row['num_trades']:>5}회"
                )

    # 워크포워드 결과
    if wf_results:
        wf_df = pd.DataFrame(wf_results)

        # 일관성 기준 필터링
        # 1. 평균 수익률 > 0
        # 2. 일관성 > 60% (5구간 중 3구간 이상 수익)
        # 3. 평균 샤프비율 > 0.3

        consistent = wf_df[
            (wf_df["avg_return_pct"] > 0)
            & (wf_df["consistency"] >= 0.6)
            & (wf_df["avg_sharpe"] > 0.3)
        ].copy()

        print("\n[워크포워드 검증]")
        print(f"  총 테스트: {len(wf_df)}개")
        print(
            f"  일관성 기준 통과: {len(consistent)}개 ({len(consistent)/len(wf_df)*100:.1f}%)"
        )

        if len(consistent) > 0:
            consistent = consistent.sort_values("consistency", ascending=False)

            print("\n[일관성 높은 전략 - 상위 20개]")
            print("-" * 90)
            print(
                f"{'종목':<12} {'전략':<25} {'평균수익':>10} {'평균샤프':>8} {'승률':>6} {'일관성':>8}"
            )
            print("-" * 90)

            for _, row in consistent.head(20).iterrows():
                print(
                    f"{row['name'][:10]:<12} {row['strategy']:<25} "
                    f"{row['avg_return_pct']:>9.1f}% {row['avg_sharpe']:>8.2f} "
                    f"{row['avg_win_rate']:>5.1f}% {row['consistency']:>7.0%}"
                )

    # 종합 추천
    print("\n" + "=" * 80)
    print("종합 추천 전략")
    print("=" * 80)

    if len(practical) > 0 and len(consistent) > 0:
        # 두 기준 모두 통과한 전략
        practical_set = set(zip(practical["ticker"], practical["strategy"]))
        consistent_set = set(zip(consistent["ticker"], consistent["strategy"]))
        both_pass = practical_set & consistent_set

        if both_pass:
            print(f"\n두 기준 모두 통과한 전략: {len(both_pass)}개")
            print("-" * 80)

            final_recommendations = []
            for ticker, strategy in both_pass:
                p_row = practical[
                    (practical["ticker"] == ticker)
                    & (practical["strategy"] == strategy)
                ].iloc[0]
                c_row = consistent[
                    (consistent["ticker"] == ticker)
                    & (consistent["strategy"] == strategy)
                ].iloc[0]

                final_recommendations.append(
                    {
                        "ticker": ticker,
                        "name": p_row["name"],
                        "strategy": strategy,
                        "total_return_pct": p_row["total_return_pct"],
                        "cagr_pct": p_row["cagr_pct"],
                        "max_drawdown_pct": p_row["max_drawdown_pct"],
                        "sharpe_ratio": p_row["sharpe_ratio"],
                        "win_rate": p_row["win_rate"],
                        "consistency": c_row["consistency"],
                        "score": p_row["sharpe_ratio"] * c_row["consistency"],
                    }
                )

            final_df = pd.DataFrame(final_recommendations)
            final_df = final_df.sort_values("score", ascending=False)

            print(
                f"\n{'종목':<12} {'전략':<22} {'수익률':>9} {'CAGR':>7} {'MDD':>7} {'샤프':>6} {'일관':>6}"
            )
            print("-" * 80)

            for _, row in final_df.iterrows():
                print(
                    f"{row['name'][:10]:<12} {row['strategy']:<22} "
                    f"{row['total_return_pct']:>8.1f}% {row['cagr_pct']:>6.1f}% "
                    f"{row['max_drawdown_pct']:>6.1f}% {row['sharpe_ratio']:>6.2f} "
                    f"{row['consistency']:>5.0%}"
                )

            # 결과 저장
            final_df.to_csv(
                os.path.join(RESULTS_PATH, "practical_strategies.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            print("\n결과 저장: practical_strategies.csv")

        else:
            print("\n두 기준을 모두 통과한 전략이 없습니다.")
            print("개별 기준 통과 전략을 참고하세요.")

    # 전략별 통계
    if all_results:
        print("\n[전략별 평균 성과]")
        print("-" * 70)
        strategy_stats = (
            results_df.groupby("strategy")
            .agg(
                {
                    "total_return_pct": "mean",
                    "sharpe_ratio": "mean",
                    "win_rate": "mean",
                    "max_drawdown_pct": "mean",
                }
            )
            .sort_values("sharpe_ratio", ascending=False)
        )

        for strategy, row in strategy_stats.iterrows():
            print(
                f"{strategy:<25} | 수익: {row['total_return_pct']:>7.1f}% | "
                f"샤프: {row['sharpe_ratio']:>5.2f} | MDD: {row['max_drawdown_pct']:>6.1f}%"
            )

        strategy_stats.to_csv(
            os.path.join(RESULTS_PATH, "strategy_stats.csv"), encoding="utf-8-sig"
        )

    print("\n" + "=" * 80)
    print("검증 완료")
    print("=" * 80)


if __name__ == "__main__":
    run_validation()
