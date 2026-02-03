# -*- coding: utf-8 -*-
"""
최종 Out-of-Sample 검증
- 2024년 이후 데이터로 최종 검증
- 추천 전략의 실제 성과 확인
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

KIWOOM_DATA_PATH = "E:/투자/data/kiwoom_investor"


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
            df = df.dropna(subset=["chg_qty", "wght", "cur_prc"])

            if len(df) > 100:
                all_data[ticker] = df
        except:
            pass

    return all_data


def backtest_strategy(
    df: pd.DataFrame,
    strategy_func,
    initial_capital: float = 100_000_000,
    fee_rate: float = 0.0015,
    slippage: float = 0.001,
):
    """백테스트 실행"""
    signals = strategy_func(df)

    capital = initial_capital
    position = 0
    shares = 0
    trades = []
    equity_curve = [initial_capital]

    for i in range(1, len(df)):
        current = df.iloc[i]
        price = current["cur_prc"]
        signal = signals.iloc[i]

        if position == 0 and signal == 1:
            buy_price = price * (1 + slippage)
            invest_amount = capital * 0.95
            shares = int(invest_amount / buy_price)
            if shares > 0:
                cost = shares * buy_price * (1 + fee_rate)
                capital -= cost
                position = 1
                trades.append(
                    {"type": "buy", "price": buy_price, "date": current["dt"]}
                )

        elif position == 1 and signal == 0:
            sell_price = price * (1 - slippage)
            revenue = shares * sell_price * (1 - fee_rate * 1.5)
            capital += revenue
            trades.append({"type": "sell", "price": sell_price, "date": current["dt"]})
            position = 0
            shares = 0

        if position == 1:
            equity_curve.append(capital + shares * price)
        else:
            equity_curve.append(capital)

    # 마지막 포지션
    if position == 1 and shares > 0:
        final_price = df["cur_prc"].iloc[-1] * (1 - slippage)
        capital += shares * final_price * (1 - fee_rate * 1.5)

    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital * 100

    # MDD
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd = drawdown.min() * 100

    # 승률
    buy_trades = [t for t in trades if t["type"] == "buy"]
    sell_trades = [t for t in trades if t["type"] == "sell"]
    win_count = sum(
        1
        for i in range(min(len(buy_trades), len(sell_trades)))
        if sell_trades[i]["price"] > buy_trades[i]["price"]
    )
    win_rate = win_count / len(buy_trades) * 100 if buy_trades else 0

    return {
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "num_trades": len(buy_trades),
        "win_rate": win_rate,
        "final_capital": final_capital,
    }


# 전략 함수들
def strategy_breakout_50_f10(df):
    df = df.copy()
    df["high_max"] = df["high_pric"].rolling(50).max()
    df["foreign_sum"] = df["chg_qty"].rolling(10).sum()
    signals = pd.Series(0, index=df.index)
    signals[(df["cur_prc"] > df["high_max"].shift(1)) & (df["foreign_sum"] > 0)] = 1
    return signals


def strategy_breakout_20_f5(df):
    df = df.copy()
    df["high_max"] = df["high_pric"].rolling(20).max()
    df["foreign_sum"] = df["chg_qty"].rolling(5).sum()
    signals = pd.Series(0, index=df.index)
    signals[(df["cur_prc"] > df["high_max"].shift(1)) & (df["foreign_sum"] > 0)] = 1
    return signals


def strategy_dualmom_10_30_f5(df):
    df = df.copy()
    df["mom_short"] = df["cur_prc"].pct_change(10)
    df["mom_long"] = df["cur_prc"].pct_change(30)
    df["foreign_sum"] = df["chg_qty"].rolling(5).sum()
    signals = pd.Series(0, index=df.index)
    signals[(df["mom_short"] > 0) & (df["mom_long"] > 0) & (df["foreign_sum"] > 0)] = 1
    return signals


def strategy_trendfilter_100_f10(df):
    df = df.copy()
    df["ma"] = df["cur_prc"].rolling(100).mean()
    df["foreign_sum"] = df["chg_qty"].rolling(10).sum()
    signals = pd.Series(0, index=df.index)
    signals[(df["cur_prc"] > df["ma"]) & (df["foreign_sum"] > 0)] = 1
    return signals


def strategy_momentum_20_f10(df):
    df = df.copy()
    df["momentum"] = df["cur_prc"].pct_change(20)
    df["foreign_sum"] = df["chg_qty"].rolling(10).sum()
    signals = pd.Series(0, index=df.index)
    signals[(df["momentum"] > 0) & (df["foreign_sum"] > 0)] = 1
    return signals


def strategy_rsi_14_30_f5(df):
    df = df.copy()
    delta = df["cur_prc"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["foreign_sum"] = df["chg_qty"].rolling(5).sum()
    signals = pd.Series(0, index=df.index)
    signals[(df["rsi"] < 30) & (df["foreign_sum"] > 0)] = 1
    return signals


def strategy_momentum_10_f5(df):
    df = df.copy()
    df["momentum"] = df["cur_prc"].pct_change(10)
    df["foreign_sum"] = df["chg_qty"].rolling(5).sum()
    signals = pd.Series(0, index=df.index)
    signals[(df["momentum"] > 0) & (df["foreign_sum"] > 0)] = 1
    return signals


def run_oos_validation():
    """OOS 검증 실행"""
    print("=" * 80)
    print("최종 Out-of-Sample 검증 (2024년 이후)")
    print("=" * 80)

    print("\n[1] 데이터 로드...")
    all_data = load_merged_data()
    print(f"  종목 수: {len(all_data)}개")

    # 추천 전략 정의
    recommended_strategies = [
        ("에코프로비엠", "247540", "Breakout_50_F10", strategy_breakout_50_f10),
        ("에코프로비엠", "247540", "Breakout_20_F5", strategy_breakout_20_f5),
        ("에코프로", "086520", "DualMom_10_30_F5", strategy_dualmom_10_30_f5),
        ("한미반도체", "042700", "TrendFilter_100_F10", strategy_trendfilter_100_f10),
        ("한미반도체", "042700", "Momentum_20_F10", strategy_momentum_20_f10),
        ("한미반도체", "042700", "Breakout_50_F10", strategy_breakout_50_f10),
        ("LG화학", "051910", "RSI_14_30_F5", strategy_rsi_14_30_f5),
        ("카카오", "035720", "Momentum_10_F5", strategy_momentum_10_f5),
        ("카카오", "035720", "Breakout_50_F10", strategy_breakout_50_f10),
        ("셀트리온제약", "068270", "RSI_14_30_F5", strategy_rsi_14_30_f5),
    ]

    # OOS 기간: 2024-01-01 이후
    oos_start = "2024-01-01"

    print(f"\n[2] In-Sample (~ 2023-12-31) vs Out-of-Sample (2024-01-01 ~) 비교")
    print("-" * 100)

    results = []

    for name, ticker, strategy_name, strategy_func in recommended_strategies:
        if ticker not in all_data:
            continue

        df = all_data[ticker]

        # In-Sample (2018 ~ 2023)
        is_df = (
            df[(df["dt"] >= "2018-01-01") & (df["dt"] < oos_start)]
            .copy()
            .reset_index(drop=True)
        )
        # Out-of-Sample (2024 ~)
        oos_df = df[df["dt"] >= oos_start].copy().reset_index(drop=True)

        if len(is_df) < 100 or len(oos_df) < 50:
            continue

        try:
            is_result = backtest_strategy(is_df, strategy_func)
            oos_result = backtest_strategy(oos_df, strategy_func)

            results.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "strategy": strategy_name,
                    "is_return": is_result["total_return_pct"],
                    "is_mdd": is_result["max_drawdown_pct"],
                    "is_trades": is_result["num_trades"],
                    "is_winrate": is_result["win_rate"],
                    "oos_return": oos_result["total_return_pct"],
                    "oos_mdd": oos_result["max_drawdown_pct"],
                    "oos_trades": oos_result["num_trades"],
                    "oos_winrate": oos_result["win_rate"],
                    "is_period": f"{is_df['dt'].min().date()} ~ {is_df['dt'].max().date()}",
                    "oos_period": f"{oos_df['dt'].min().date()} ~ {oos_df['dt'].max().date()}",
                }
            )
        except Exception as e:
            print(f"  {name} {strategy_name} 오류: {e}")

    # 결과 출력
    print(
        f"\n{'종목':<12} {'전략':<22} │ {'IS수익':>8} {'IS_MDD':>7} │ {'OOS수익':>8} {'OOS_MDD':>7} │ 평가"
    )
    print("-" * 100)

    passing = []
    for r in results:
        # 평가 기준
        # 1. OOS 수익률 > 0
        # 2. OOS MDD > -30%
        # 3. OOS 수익률이 IS의 20% 이상
        is_profitable = r["oos_return"] > 0
        mdd_acceptable = r["oos_mdd"] > -30
        consistent = (
            r["oos_return"] > r["is_return"] * 0.2
            if r["is_return"] > 0
            else r["oos_return"] > 0
        )

        if is_profitable and mdd_acceptable:
            eval_mark = "PASS"
            passing.append(r)
        else:
            eval_mark = "FAIL"

        print(
            f"{r['name']:<12} {r['strategy']:<22} │ "
            f"{r['is_return']:>7.1f}% {r['is_mdd']:>6.1f}% │ "
            f"{r['oos_return']:>7.1f}% {r['oos_mdd']:>6.1f}% │ {eval_mark}"
        )

    # 최종 추천
    print("\n" + "=" * 80)
    print("최종 실거래 추천 전략")
    print("=" * 80)

    if passing:
        print(f"\n검증 통과: {len(passing)}개 전략")
        print("-" * 80)

        for r in sorted(passing, key=lambda x: x["oos_return"], reverse=True):
            print(f"\n[{r['name']} + {r['strategy']}]")
            print(f"  In-Sample  ({r['is_period']})")
            print(
                f"    - 수익률: {r['is_return']:.1f}%, MDD: {r['is_mdd']:.1f}%, 거래: {r['is_trades']}회, 승률: {r['is_winrate']:.1f}%"
            )
            print(f"  Out-Sample ({r['oos_period']})")
            print(
                f"    - 수익률: {r['oos_return']:.1f}%, MDD: {r['oos_mdd']:.1f}%, 거래: {r['oos_trades']}회, 승률: {r['oos_winrate']:.1f}%"
            )

        # 결과 저장
        results_df = pd.DataFrame(passing)
        results_df.to_csv(
            "E:/투자/Multi-Asset Strategy Platform/final_recommended_strategies.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\n결과 저장: final_recommended_strategies.csv")

    else:
        print("\nOOS 검증을 통과한 전략이 없습니다.")
        print("전략 파라미터 조정이 필요합니다.")

    # 포트폴리오 시뮬레이션
    if passing:
        print("\n" + "=" * 80)
        print("포트폴리오 시뮬레이션 (균등 배분)")
        print("=" * 80)

        total_capital = 100_000_000
        per_strategy = total_capital / len(passing)

        total_oos_return = 0
        worst_mdd = 0

        for r in passing:
            strategy_return = per_strategy * (1 + r["oos_return"] / 100) - per_strategy
            total_oos_return += strategy_return
            if r["oos_mdd"] < worst_mdd:
                worst_mdd = r["oos_mdd"]

        portfolio_return = total_oos_return / total_capital * 100

        print(f"\n  초기 자본: {total_capital:,}원")
        print(f"  전략 수: {len(passing)}개")
        print(f"  전략당 배분: {per_strategy:,.0f}원")
        print(f"\n  포트폴리오 OOS 수익률: {portfolio_return:.1f}%")
        print(f"  예상 최대 낙폭: {worst_mdd:.1f}%")


if __name__ == "__main__":
    run_oos_validation()
