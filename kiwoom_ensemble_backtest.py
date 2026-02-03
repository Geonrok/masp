# -*- coding: utf-8 -*-
"""
키움 데이터 기반 앙상블 & 팩터 전략 백테스트
- 다중 신호 앙상블
- 팩터 가중치 최적화
- 동적 포지션 사이징
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore")

KIWOOM_DATA_PATH = "E:/투자/data/kiwoom_investor"


def load_merged_data():
    """merged 데이터 로드"""
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


def calculate_factors(df: pd.DataFrame) -> pd.DataFrame:
    """팩터 계산"""
    df = df.copy()

    # 가격 모멘텀 (20일)
    df["mom_20"] = df["cur_prc"].pct_change(20)

    # 외국인 순매수 모멘텀
    df["foreign_mom_5"] = df["chg_qty"].rolling(5).sum()
    df["foreign_mom_10"] = df["chg_qty"].rolling(10).sum()
    df["foreign_mom_20"] = df["chg_qty"].rolling(20).sum()

    # 외국인 비중 변화
    df["wght_change_5"] = df["wght"].diff(5)
    df["wght_change_10"] = df["wght"].diff(10)

    # 가격 추세
    df["ma_20"] = df["cur_prc"].rolling(20).mean()
    df["ma_50"] = df["cur_prc"].rolling(50).mean()
    df["ma_100"] = df["cur_prc"].rolling(100).mean()
    df["trend_short"] = (df["cur_prc"] > df["ma_20"]).astype(int)
    df["trend_mid"] = (df["cur_prc"] > df["ma_50"]).astype(int)
    df["trend_long"] = (df["cur_prc"] > df["ma_100"]).astype(int)

    # 변동성
    df["volatility"] = df["return"].rolling(20).std() * np.sqrt(252)

    # RSI
    delta = df["cur_prc"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # 거래량 비율
    df["vol_ratio"] = df["trde_qty"] / df["trde_qty"].rolling(20).mean()

    return df


class EnsembleStrategy:
    """앙상블 전략"""

    def __init__(self, name: str, params: dict = None):
        self.name = name
        self.params = params or {}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class MultiFactorEnsemble(EnsembleStrategy):
    """다중 팩터 앙상블"""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        foreign_weight = self.params.get("foreign_weight", 0.4)
        momentum_weight = self.params.get("momentum_weight", 0.3)
        trend_weight = self.params.get("trend_weight", 0.3)
        threshold = self.params.get("threshold", 0.5)

        df = calculate_factors(df)
        df["signal"] = 0

        # 각 팩터 점수 계산 (0 또는 1)
        df["f_foreign"] = (df["foreign_mom_10"] > 0).astype(float)
        df["f_momentum"] = (df["mom_20"] > 0).astype(float)
        df["f_trend"] = ((df["trend_short"] + df["trend_mid"]) >= 1).astype(float)

        # 가중 합계
        df["score"] = (
            df["f_foreign"] * foreign_weight
            + df["f_momentum"] * momentum_weight
            + df["f_trend"] * trend_weight
        )

        df.loc[df["score"] >= threshold, "signal"] = 1

        return df


class VotingEnsemble(EnsembleStrategy):
    """투표 앙상블 - 다수결"""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        min_votes = self.params.get("min_votes", 3)

        df = calculate_factors(df)
        df["signal"] = 0

        # 개별 전략 신호
        df["vote_1"] = (df["foreign_mom_5"] > 0).astype(int)  # 외국인 5일 순매수
        df["vote_2"] = (df["foreign_mom_10"] > 0).astype(int)  # 외국인 10일 순매수
        df["vote_3"] = (df["trend_short"] == 1).astype(int)  # 20일선 위
        df["vote_4"] = (df["rsi"] < 70).astype(int)  # 과매수 아님
        df["vote_5"] = (df["mom_20"] > 0).astype(int)  # 20일 모멘텀 양수

        df["total_votes"] = df[["vote_1", "vote_2", "vote_3", "vote_4", "vote_5"]].sum(
            axis=1
        )

        df.loc[df["total_votes"] >= min_votes, "signal"] = 1

        return df


class AdaptiveEnsemble(EnsembleStrategy):
    """적응형 앙상블 - 변동성에 따라 가중치 조절"""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        vol_threshold = self.params.get("vol_threshold", 0.3)

        df = calculate_factors(df)
        df["signal"] = 0
        df["position_size"] = 1.0

        for i in range(100, len(df)):
            vol = df["volatility"].iloc[i]
            foreign_mom = df["foreign_mom_10"].iloc[i]
            trend = df["trend_short"].iloc[i]

            # 저변동성: 외국인 중심, 고변동성: 추세 중심
            if pd.notna(vol):
                if vol < vol_threshold:
                    # 저변동성 환경
                    if foreign_mom > 0 and trend == 1:
                        df.iloc[i, df.columns.get_loc("signal")] = 1
                        df.iloc[i, df.columns.get_loc("position_size")] = 1.2
                else:
                    # 고변동성 환경 - 추세 강화
                    if (
                        foreign_mom > 0
                        and df["trend_mid"].iloc[i] == 1
                        and df["trend_long"].iloc[i] == 1
                    ):
                        df.iloc[i, df.columns.get_loc("signal")] = 1
                        df.iloc[i, df.columns.get_loc("position_size")] = 0.8

        return df


class MomentumQuality(EnsembleStrategy):
    """모멘텀 + 퀄리티 팩터"""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        mom_lookback = self.params.get("mom_lookback", 20)
        foreign_lookback = self.params.get("foreign_lookback", 10)

        df = calculate_factors(df)
        df["signal"] = 0

        # 모멘텀 스코어
        df["mom_score"] = df["cur_prc"].pct_change(mom_lookback).rank(pct=True)

        # 퀄리티 스코어 (외국인 비중 수준)
        df["quality_score"] = df["wght"].rank(pct=True)

        # 외국인 순매수 스코어
        df["flow_score"] = df[f"foreign_mom_{foreign_lookback}"].rank(pct=True)

        # 종합 스코어
        df["total_score"] = (
            df["mom_score"] + df["quality_score"] + df["flow_score"]
        ) / 3

        # 상위 50%만 매수
        df.loc[df["total_score"] > 0.5, "signal"] = 1

        return df


class TrendBreakout(EnsembleStrategy):
    """추세 돌파 + 외국인 확인"""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        breakout_period = self.params.get("breakout_period", 20)
        foreign_confirm = self.params.get("foreign_confirm", 5)

        df = calculate_factors(df)
        df["signal"] = 0

        df["high_max"] = df["high_pric"].rolling(breakout_period).max()
        df["breakout"] = (df["cur_prc"] > df["high_max"].shift(1)).astype(int)

        df["foreign_confirm"] = (
            df["chg_qty"].rolling(foreign_confirm).sum() > 0
        ).astype(int)

        # 돌파 + 외국인 순매수
        df.loc[(df["breakout"] == 1) & (df["foreign_confirm"] == 1), "signal"] = 1

        return df


class MeanReversionForeign(EnsembleStrategy):
    """평균회귀 + 외국인 반전"""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        lookback = self.params.get("lookback", 20)
        std_mult = self.params.get("std_mult", 2)

        df = calculate_factors(df)
        df["signal"] = 0

        df["price_ma"] = df["cur_prc"].rolling(lookback).mean()
        df["price_std"] = df["cur_prc"].rolling(lookback).std()
        df["lower_band"] = df["price_ma"] - std_mult * df["price_std"]

        # 가격이 하단 밴드 아래 + 외국인 순매수 전환
        df["foreign_turn"] = (df["chg_qty"] > 0) & (df["chg_qty"].shift(1) < 0)

        df.loc[(df["cur_prc"] < df["lower_band"]) & (df["foreign_turn"]), "signal"] = 1

        return df


class VolatilityBreakout(EnsembleStrategy):
    """변동성 돌파 + 외국인"""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        k = self.params.get("k", 0.5)
        foreign_period = self.params.get("foreign_period", 3)

        df = calculate_factors(df)
        df["signal"] = 0

        df["range"] = df["high_pric"] - df["low_pric"]
        df["target"] = df["open_pric"] + df["range"].shift(1) * k

        df["foreign_positive"] = (
            df["chg_qty"].rolling(foreign_period).sum() > 0
        ).astype(int)

        # 변동성 돌파 + 외국인 순매수
        df.loc[
            (df["cur_prc"] > df["target"]) & (df["foreign_positive"] == 1), "signal"
        ] = 1

        return df


def backtest_strategy(
    df: pd.DataFrame,
    strategy: EnsembleStrategy,
    ticker: str,
    initial_capital: float = 100_000_000,
    fee_rate: float = 0.001,
):
    """전략 백테스트"""
    if df is None or len(df) < 150:
        return None

    try:
        df = strategy.generate_signals(df)
    except Exception as e:
        return None

    capital = initial_capital
    position = 0
    shares = 0
    trades = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        price = current["cur_prc"]
        signal = current["signal"]
        position_size = (
            current.get("position_size", 1.0) if "position_size" in df.columns else 1.0
        )

        if position == 0 and signal == 1:
            invest_amount = capital * position_size * 0.95
            shares = int(invest_amount / price)
            if shares > 0:
                cost = shares * price * (1 + fee_rate)
                capital -= cost
                position = 1
                trades.append({"type": "buy", "price": price, "date": current["dt"]})

        elif position == 1 and signal == 0:
            revenue = shares * price * (1 - fee_rate)
            capital += revenue
            position = 0
            trades.append({"type": "sell", "price": price, "date": current["dt"]})
            shares = 0

    # 마지막 포지션 청산
    if position == 1 and shares > 0:
        final_price = df["cur_prc"].iloc[-1]
        capital += shares * final_price * (1 - fee_rate)

    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital * 100

    # 거래 통계
    buy_trades = [t for t in trades if t["type"] == "buy"]
    sell_trades = [t for t in trades if t["type"] == "sell"]

    win_count = 0
    for i in range(min(len(buy_trades), len(sell_trades))):
        if sell_trades[i]["price"] > buy_trades[i]["price"]:
            win_count += 1

    win_rate = win_count / len(buy_trades) * 100 if len(buy_trades) > 0 else 0

    # 최대 낙폭 계산
    equity_curve = []
    temp_capital = initial_capital
    temp_shares = 0
    temp_position = 0

    for i in range(len(df)):
        current = df.iloc[i]
        price = current["cur_prc"]
        signal = current["signal"]

        if temp_position == 1:
            equity = temp_capital + temp_shares * price
        else:
            equity = temp_capital

        equity_curve.append(equity)

        if temp_position == 0 and signal == 1 and i > 0:
            invest = temp_capital * 0.95
            temp_shares = int(invest / price)
            if temp_shares > 0:
                temp_capital -= temp_shares * price * (1 + fee_rate)
                temp_position = 1
        elif temp_position == 1 and signal == 0:
            temp_capital += temp_shares * price * (1 - fee_rate)
            temp_position = 0
            temp_shares = 0

    equity_df = pd.Series(equity_curve)
    running_max = equity_df.cummax()
    drawdown = (equity_df - running_max) / running_max
    max_dd = drawdown.min() * 100

    return {
        "strategy": strategy.name,
        "ticker": ticker,
        "name": df["name"].iloc[0] if "name" in df.columns else ticker,
        "total_return_pct": total_return,
        "final_capital": final_capital,
        "num_trades": len(buy_trades),
        "win_rate": win_rate,
        "max_drawdown": max_dd,
    }


def run_ensemble_backtests():
    """앙상블 전략 백테스트 실행"""
    print("=" * 70)
    print("키움 데이터 기반 앙상블 & 팩터 전략 백테스트")
    print("=" * 70)

    print("\n[1] 데이터 로드...")
    all_data = load_merged_data()
    print(f"  로드된 종목: {len(all_data)}개")

    # 전략 정의
    strategies = [
        # 다중 팩터
        MultiFactorEnsemble(
            "MultiFactor_443_50",
            {
                "foreign_weight": 0.4,
                "momentum_weight": 0.4,
                "trend_weight": 0.3,
                "threshold": 0.5,
            },
        ),
        MultiFactorEnsemble(
            "MultiFactor_532_60",
            {
                "foreign_weight": 0.5,
                "momentum_weight": 0.3,
                "trend_weight": 0.2,
                "threshold": 0.6,
            },
        ),
        MultiFactorEnsemble(
            "MultiFactor_334_40",
            {
                "foreign_weight": 0.3,
                "momentum_weight": 0.3,
                "trend_weight": 0.4,
                "threshold": 0.4,
            },
        ),
        # 투표 앙상블
        VotingEnsemble("Voting_3", {"min_votes": 3}),
        VotingEnsemble("Voting_4", {"min_votes": 4}),
        VotingEnsemble("Voting_5", {"min_votes": 5}),
        # 적응형
        AdaptiveEnsemble("Adaptive_30", {"vol_threshold": 0.3}),
        AdaptiveEnsemble("Adaptive_40", {"vol_threshold": 0.4}),
        # 모멘텀 + 퀄리티
        MomentumQuality(
            "MomQuality_20_10", {"mom_lookback": 20, "foreign_lookback": 10}
        ),
        MomentumQuality("MomQuality_10_5", {"mom_lookback": 10, "foreign_lookback": 5}),
        # 돌파 전략
        TrendBreakout("Breakout_20_5", {"breakout_period": 20, "foreign_confirm": 5}),
        TrendBreakout("Breakout_10_3", {"breakout_period": 10, "foreign_confirm": 3}),
        TrendBreakout("Breakout_50_10", {"breakout_period": 50, "foreign_confirm": 10}),
        # 평균회귀
        MeanReversionForeign("MeanRev_20_2", {"lookback": 20, "std_mult": 2}),
        MeanReversionForeign("MeanRev_20_2.5", {"lookback": 20, "std_mult": 2.5}),
        # 변동성 돌파
        VolatilityBreakout("VolBreak_0.5_3", {"k": 0.5, "foreign_period": 3}),
        VolatilityBreakout("VolBreak_0.6_5", {"k": 0.6, "foreign_period": 5}),
        VolatilityBreakout("VolBreak_0.4_3", {"k": 0.4, "foreign_period": 3}),
    ]

    print(f"  전략 수: {len(strategies)}개")
    print(f"  예상 테스트: {len(strategies) * len(all_data)}개")

    results = []
    test_count = 0

    print("\n[2] 백테스트 실행...")
    for ticker, df in all_data.items():
        for strategy in strategies:
            result = backtest_strategy(df, strategy, ticker)
            if result:
                results.append(result)
                test_count += 1

    print(f"\n총 테스트: {len(results)}개")

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("total_return_pct", ascending=False)

        profitable = results_df[results_df["total_return_pct"] > 0]
        print(
            f"수익 전략: {len(profitable)}개 ({len(profitable)/len(results)*100:.1f}%)"
        )

        print("\n[상위 20개 결과]")
        print("-" * 90)
        for _, row in results_df.head(20).iterrows():
            print(
                f"{row['name'][:10]:<10} | {row['strategy']:<25} | "
                f"수익: {row['total_return_pct']:>8.1f}% | "
                f"MDD: {row['max_drawdown']:>6.1f}% | "
                f"거래: {row['num_trades']:>3}회 | "
                f"승률: {row['win_rate']:>5.1f}%"
            )

        # 전략별 평균 성과
        print("\n[전략별 평균 성과]")
        print("-" * 80)
        strategy_avg = (
            results_df.groupby("strategy")
            .agg(
                {
                    "total_return_pct": "mean",
                    "win_rate": "mean",
                    "max_drawdown": "mean",
                    "num_trades": "mean",
                }
            )
            .sort_values("total_return_pct", ascending=False)
        )

        for strategy, row in strategy_avg.iterrows():
            print(
                f"{strategy:<25} | 평균수익: {row['total_return_pct']:>8.1f}% | "
                f"MDD: {row['max_drawdown']:>6.1f}% | "
                f"승률: {row['win_rate']:>5.1f}% | "
                f"거래: {row['num_trades']:>5.1f}회"
            )

        # 결과 저장
        output_file = (
            "E:/투자/Multi-Asset Strategy Platform/ensemble_backtest_results.csv"
        )
        results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n결과 저장: {output_file}")

        return results_df

    return None


if __name__ == "__main__":
    results = run_ensemble_backtests()
