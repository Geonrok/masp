# -*- coding: utf-8 -*-
"""
범용 전략 검증 - 코스닥 전체 종목 대상
- 특정 종목이 아닌 "전략 자체"의 성과 검증
- 모든 종목에서 일관되게 수익을 내는 전략 찾기
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

KOSDAQ_OHLCV_PATH = "E:/투자/data/kr_stock/kosdaq_ohlcv"


def load_all_kosdaq():
    """코스닥 전체 종목 로드"""
    files = [f for f in os.listdir(KOSDAQ_OHLCV_PATH) if f.endswith(".csv")]
    all_data = {}

    for f in files:
        ticker = f.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(KOSDAQ_OHLCV_PATH, f))
            date_col = "Date" if "Date" in df.columns else "date"
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.rename(
                columns={
                    date_col: "date",
                    "Close": "close",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Volume": "volume",
                }
            )
            df = df.sort_values("date").reset_index(drop=True)

            # 최소 2년 데이터 필요
            if len(df) > 500:
                all_data[ticker] = df
        except:
            pass

    return all_data


def backtest_strategy(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 10_000_000,
    fee_rate: float = 0.0015,
    slippage: float = 0.001,
):
    """백테스트 실행"""
    capital = initial_capital
    position = 0
    shares = 0
    trades = []

    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        signal = signals.iloc[i] if i < len(signals) else 0

        if position == 0 and signal == 1:
            buy_price = price * (1 + slippage)
            invest = capital * 0.95
            shares = int(invest / buy_price)
            if shares > 0:
                capital -= shares * buy_price * (1 + fee_rate)
                position = 1
                trades.append({"type": "buy", "price": buy_price})

        elif position == 1 and signal == 0:
            sell_price = price * (1 - slippage)
            capital += shares * sell_price * (1 - fee_rate * 1.5)
            trades.append({"type": "sell", "price": sell_price})
            position = 0
            shares = 0

    if position == 1 and shares > 0:
        capital += shares * df["close"].iloc[-1] * (1 - slippage) * (1 - fee_rate * 1.5)

    total_return = (capital - initial_capital) / initial_capital * 100

    buy_trades = [t for t in trades if t["type"] == "buy"]
    sell_trades = [t for t in trades if t["type"] == "sell"]
    wins = sum(
        1
        for i in range(min(len(buy_trades), len(sell_trades)))
        if sell_trades[i]["price"] > buy_trades[i]["price"]
    )
    win_rate = wins / len(buy_trades) * 100 if buy_trades else 0

    return {
        "total_return": total_return,
        "num_trades": len(buy_trades),
        "win_rate": win_rate,
    }


# ========== 전략 정의 (외국인 데이터 없이 순수 기술적 분석) ==========


def strategy_golden_cross(df, short=20, long=60):
    """골든크로스 전략"""
    df = df.copy()
    df["ma_short"] = df["close"].rolling(short).mean()
    df["ma_long"] = df["close"].rolling(long).mean()
    signals = pd.Series(0, index=df.index)
    signals[df["ma_short"] > df["ma_long"]] = 1
    return signals


def strategy_trend_following(df, period=50):
    """추세추종 - 이평선 위에서 매수"""
    df = df.copy()
    df["ma"] = df["close"].rolling(period).mean()
    signals = pd.Series(0, index=df.index)
    signals[df["close"] > df["ma"]] = 1
    return signals


def strategy_breakout(df, period=20):
    """채널 돌파"""
    df = df.copy()
    df["high_max"] = df["high"].rolling(period).max()
    signals = pd.Series(0, index=df.index)
    signals[df["close"] > df["high_max"].shift(1)] = 1
    return signals


def strategy_momentum(df, period=20):
    """모멘텀 - 가격 상승 추세"""
    df = df.copy()
    df["momentum"] = df["close"].pct_change(period)
    signals = pd.Series(0, index=df.index)
    signals[df["momentum"] > 0] = 1
    return signals


def strategy_rsi_oversold(df, period=14, threshold=30):
    """RSI 과매도 반전"""
    df = df.copy()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    signals = pd.Series(0, index=df.index)
    signals[df["rsi"] < threshold] = 1
    return signals


def strategy_bollinger_mean_reversion(df, period=20, std_mult=2):
    """볼린저 밴드 평균회귀"""
    df = df.copy()
    df["ma"] = df["close"].rolling(period).mean()
    df["std"] = df["close"].rolling(period).std()
    df["lower"] = df["ma"] - std_mult * df["std"]
    signals = pd.Series(0, index=df.index)
    signals[df["close"] < df["lower"]] = 1
    return signals


def strategy_macd_cross(df, fast=12, slow=26, signal=9):
    """MACD 크로스오버"""
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast).mean()
    df["ema_slow"] = df["close"].ewm(span=slow).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd"].ewm(span=signal).mean()
    signals = pd.Series(0, index=df.index)
    signals[df["macd"] > df["macd_signal"]] = 1
    return signals


def strategy_volatility_breakout(df, k=0.5):
    """변동성 돌파"""
    df = df.copy()
    df["range"] = df["high"] - df["low"]
    df["target"] = df["open"] + df["range"].shift(1) * k
    signals = pd.Series(0, index=df.index)
    signals[df["close"] > df["target"]] = 1
    return signals


def strategy_dual_momentum(df, short=10, long=30):
    """듀얼 모멘텀"""
    df = df.copy()
    df["mom_short"] = df["close"].pct_change(short)
    df["mom_long"] = df["close"].pct_change(long)
    signals = pd.Series(0, index=df.index)
    signals[(df["mom_short"] > 0) & (df["mom_long"] > 0)] = 1
    return signals


def strategy_volume_breakout(df, vol_mult=2, price_period=20):
    """거래량 돌파 + 가격 추세"""
    df = df.copy()
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["price_ma"] = df["close"].rolling(price_period).mean()
    signals = pd.Series(0, index=df.index)
    signals[
        (df["volume"] > df["vol_ma"] * vol_mult) & (df["close"] > df["price_ma"])
    ] = 1
    return signals


def run_universal_validation():
    """범용 전략 검증"""
    print("=" * 80)
    print("코스닥 전체 종목 대상 범용 전략 검증")
    print("=" * 80)
    print("\n목표: 어떤 종목에든 적용할 수 있는 전략 찾기")

    print("\n[1] 데이터 로드...")
    all_data = load_all_kosdaq()
    print(f"  코스닥 종목 수: {len(all_data)}개")

    # 전략 정의
    strategies = [
        ("GoldenCross_20_60", lambda df: strategy_golden_cross(df, 20, 60)),
        ("GoldenCross_10_30", lambda df: strategy_golden_cross(df, 10, 30)),
        ("GoldenCross_5_20", lambda df: strategy_golden_cross(df, 5, 20)),
        ("TrendFollow_20", lambda df: strategy_trend_following(df, 20)),
        ("TrendFollow_50", lambda df: strategy_trend_following(df, 50)),
        ("TrendFollow_100", lambda df: strategy_trend_following(df, 100)),
        ("Breakout_10", lambda df: strategy_breakout(df, 10)),
        ("Breakout_20", lambda df: strategy_breakout(df, 20)),
        ("Breakout_50", lambda df: strategy_breakout(df, 50)),
        ("Momentum_10", lambda df: strategy_momentum(df, 10)),
        ("Momentum_20", lambda df: strategy_momentum(df, 20)),
        ("Momentum_60", lambda df: strategy_momentum(df, 60)),
        ("RSI_14_30", lambda df: strategy_rsi_oversold(df, 14, 30)),
        ("RSI_7_25", lambda df: strategy_rsi_oversold(df, 7, 25)),
        ("RSI_14_40", lambda df: strategy_rsi_oversold(df, 14, 40)),
        ("Bollinger_20_2", lambda df: strategy_bollinger_mean_reversion(df, 20, 2)),
        ("Bollinger_20_2.5", lambda df: strategy_bollinger_mean_reversion(df, 20, 2.5)),
        ("MACD_12_26_9", lambda df: strategy_macd_cross(df, 12, 26, 9)),
        ("MACD_8_21_5", lambda df: strategy_macd_cross(df, 8, 21, 5)),
        ("VolBreak_0.5", lambda df: strategy_volatility_breakout(df, 0.5)),
        ("VolBreak_0.6", lambda df: strategy_volatility_breakout(df, 0.6)),
        ("DualMom_10_30", lambda df: strategy_dual_momentum(df, 10, 30)),
        ("DualMom_20_60", lambda df: strategy_dual_momentum(df, 20, 60)),
        ("VolumeBK_2x", lambda df: strategy_volume_breakout(df, 2, 20)),
        ("VolumeBK_3x", lambda df: strategy_volume_breakout(df, 3, 20)),
    ]

    print(f"  전략 수: {len(strategies)}개")

    # 각 전략을 모든 종목에 적용
    print("\n[2] 전략별 전체 종목 테스트...")

    strategy_results = {}

    for strategy_name, strategy_func in strategies:
        returns = []
        win_rates = []
        trade_counts = []
        profitable_count = 0

        for ticker, df in all_data.items():
            try:
                signals = strategy_func(df)
                result = backtest_strategy(df, signals)
                returns.append(result["total_return"])
                win_rates.append(result["win_rate"])
                trade_counts.append(result["num_trades"])
                if result["total_return"] > 0:
                    profitable_count += 1
            except:
                pass

        if returns:
            strategy_results[strategy_name] = {
                "avg_return": np.mean(returns),
                "median_return": np.median(returns),
                "std_return": np.std(returns),
                "avg_win_rate": np.mean(win_rates),
                "avg_trades": np.mean(trade_counts),
                "profitable_pct": profitable_count / len(returns) * 100,
                "tested_stocks": len(returns),
                "sharpe_approx": (
                    np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                ),
            }

    # 결과 정렬 및 출력
    print("\n" + "=" * 80)
    print("전략별 성과 (전체 종목 평균)")
    print("=" * 80)

    sorted_strategies = sorted(
        strategy_results.items(), key=lambda x: x[1]["profitable_pct"], reverse=True
    )

    print(
        f"\n{'전략':<20} | {'수익종목%':>8} | {'평균수익':>10} | {'중앙수익':>10} | {'승률':>6} | {'샤프':>6}"
    )
    print("-" * 80)

    for strategy_name, stats in sorted_strategies:
        print(
            f"{strategy_name:<20} | "
            f"{stats['profitable_pct']:>7.1f}% | "
            f"{stats['avg_return']:>9.1f}% | "
            f"{stats['median_return']:>9.1f}% | "
            f"{stats['avg_win_rate']:>5.1f}% | "
            f"{stats['sharpe_approx']:>6.2f}"
        )

    # 실거래 가능 기준
    print("\n" + "=" * 80)
    print("실거래 가능 전략 (수익종목 50% 이상)")
    print("=" * 80)

    practical = [
        (name, stats)
        for name, stats in sorted_strategies
        if stats["profitable_pct"] >= 50
    ]

    if practical:
        print(f"\n통과 전략: {len(practical)}개")
        print("-" * 80)

        for strategy_name, stats in practical:
            print(f"\n[{strategy_name}]")
            print(f"  - 수익 종목 비율: {stats['profitable_pct']:.1f}%")
            print(f"  - 평균 수익률: {stats['avg_return']:.1f}%")
            print(f"  - 중앙값 수익률: {stats['median_return']:.1f}%")
            print(f"  - 평균 승률: {stats['avg_win_rate']:.1f}%")
            print(f"  - 테스트 종목 수: {stats['tested_stocks']}개")
    else:
        print("\n50% 이상 종목에서 수익을 내는 전략이 없습니다.")
        print("기준을 40%로 낮춰서 확인:")

        practical_40 = [
            (name, stats)
            for name, stats in sorted_strategies
            if stats["profitable_pct"] >= 40
        ]

        for strategy_name, stats in practical_40[:5]:
            print(
                f"  {strategy_name}: 수익종목 {stats['profitable_pct']:.1f}%, "
                f"평균 {stats['avg_return']:.1f}%"
            )

    # OOS 검증 (2024년 이후)
    print("\n" + "=" * 80)
    print("Out-of-Sample 검증 (2024년 이후)")
    print("=" * 80)

    oos_start = pd.Timestamp("2024-01-01")

    for strategy_name, strategy_func in strategies[:10]:  # 상위 10개만
        is_returns = []
        oos_returns = []

        for ticker, df in all_data.items():
            try:
                is_df = df[df["date"] < oos_start].copy().reset_index(drop=True)
                oos_df = df[df["date"] >= oos_start].copy().reset_index(drop=True)

                if len(is_df) > 200 and len(oos_df) > 50:
                    is_signals = strategy_func(is_df)
                    oos_signals = strategy_func(oos_df)

                    is_result = backtest_strategy(is_df, is_signals)
                    oos_result = backtest_strategy(oos_df, oos_signals)

                    is_returns.append(is_result["total_return"])
                    oos_returns.append(oos_result["total_return"])
            except:
                pass

        if is_returns and oos_returns:
            is_profitable = sum(1 for r in is_returns if r > 0) / len(is_returns) * 100
            oos_profitable = (
                sum(1 for r in oos_returns if r > 0) / len(oos_returns) * 100
            )

            print(
                f"{strategy_name:<20} | IS 수익종목: {is_profitable:>5.1f}% | "
                f"OOS 수익종목: {oos_profitable:>5.1f}% | "
                f"IS평균: {np.mean(is_returns):>7.1f}% | OOS평균: {np.mean(oos_returns):>7.1f}%"
            )

    # 최종 추천
    print("\n" + "=" * 80)
    print("최종 결론")
    print("=" * 80)

    # 결과 저장
    results_df = pd.DataFrame(
        [{"strategy": name, **stats} for name, stats in sorted_strategies]
    )
    results_df.to_csv(
        "E:/투자/Multi-Asset Strategy Platform/universal_strategy_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n결과 저장: universal_strategy_results.csv")


if __name__ == "__main__":
    run_universal_validation()
