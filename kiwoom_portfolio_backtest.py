# -*- coding: utf-8 -*-
"""
키움 데이터 기반 포트폴리오 & 앙상블 전략 백테스트
- 다중 종목 포트폴리오 전략
- 모멘텀 스코어 기반 종목 선정
- 리스크 패리티 배분
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings("ignore")

# 데이터 경로
KIWOOM_DATA_PATH = "E:/투자/data/kiwoom_investor"


def load_all_merged_data():
    """모든 merged 데이터 로드"""
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

            # 외국인 데이터가 있는 기간만 사용 (2018년 이후)
            df = df[df["dt"] >= "2018-01-01"].copy()
            df = df.dropna(subset=["chg_qty", "wght"])

            if len(df) > 100:
                all_data[ticker] = df

        except Exception as e:
            print(f"  {ticker} 로드 실패: {e}")

    return all_data


class PortfolioBacktester:
    """포트폴리오 백테스터"""

    def __init__(self, all_data: dict, initial_capital: float = 100_000_000):
        self.all_data = all_data
        self.initial_capital = initial_capital
        self.fee_rate = 0.001

    def get_common_dates(self) -> pd.DatetimeIndex:
        """모든 종목의 공통 날짜 추출"""
        date_sets = [set(df["dt"].values) for df in self.all_data.values()]
        common = set.intersection(*date_sets)
        return pd.DatetimeIndex(sorted(common))

    def calculate_scores(self, date, lookback=20):
        """각 종목의 스코어 계산"""
        scores = {}

        for ticker, df in self.all_data.items():
            df_until = df[df["dt"] <= date].tail(lookback + 1)

            if len(df_until) < lookback:
                continue

            current = df_until.iloc[-1]
            past = df_until.iloc[:-1]

            # 외국인 순매수 합계 (정규화)
            foreign_sum = past["chg_qty"].sum()

            # 가격 모멘텀 (수익률)
            if past["cur_prc"].iloc[0] > 0:
                price_mom = (current["cur_prc"] - past["cur_prc"].iloc[0]) / past[
                    "cur_prc"
                ].iloc[0]
            else:
                price_mom = 0

            # 외국인 비중 변화
            wght_change = (
                current["wght"] - past["wght"].iloc[0]
                if pd.notna(current["wght"]) and pd.notna(past["wght"].iloc[0])
                else 0
            )

            scores[ticker] = {
                "foreign_sum": foreign_sum,
                "price_mom": price_mom,
                "wght_change": wght_change,
                "current_price": current["cur_prc"],
            }

        return scores


def backtest_portfolio_strategy(
    all_data: dict,
    strategy_name: str,
    top_n: int = 5,
    rebalance_freq: int = 20,
    score_type: str = "foreign_sum",
    initial_capital: float = 100_000_000,
):
    """
    포트폴리오 전략 백테스트

    Parameters:
    - top_n: 보유할 종목 수
    - rebalance_freq: 리밸런싱 주기 (일)
    - score_type: 'foreign_sum', 'price_mom', 'combined'
    """

    # 공통 날짜 추출
    date_sets = [set(df["dt"].values) for df in all_data.values()]
    common_dates = sorted(set.intersection(*date_sets))

    if len(common_dates) < 100:
        return None

    dates = pd.DatetimeIndex(common_dates)

    # 초기 설정
    capital = initial_capital
    holdings = {}  # {ticker: shares}
    fee_rate = 0.001

    portfolio_values = []
    last_rebalance = 0

    for i, date in enumerate(dates):
        current_date = pd.Timestamp(date)

        # 현재 보유 종목 가치 계산
        holdings_value = 0
        for ticker, shares in holdings.items():
            df = all_data[ticker]
            current_row = df[df["dt"] == current_date]
            if len(current_row) > 0:
                holdings_value += shares * current_row["cur_prc"].values[0]

        total_value = capital + holdings_value

        # 리밸런싱 체크
        if i - last_rebalance >= rebalance_freq:
            # 스코어 계산
            scores = {}
            for ticker, df in all_data.items():
                df_until = df[df["dt"] <= current_date].tail(21)

                if len(df_until) < 20:
                    continue

                current = df_until.iloc[-1]
                past = df_until.iloc[:-1]

                if score_type == "foreign_sum":
                    score = past["chg_qty"].sum()
                elif score_type == "price_mom":
                    if past["cur_prc"].iloc[0] > 0:
                        score = (current["cur_prc"] - past["cur_prc"].iloc[0]) / past[
                            "cur_prc"
                        ].iloc[0]
                    else:
                        score = 0
                elif score_type == "combined":
                    foreign_score = past["chg_qty"].sum()
                    if past["cur_prc"].iloc[0] > 0:
                        mom_score = (
                            current["cur_prc"] - past["cur_prc"].iloc[0]
                        ) / past["cur_prc"].iloc[0]
                    else:
                        mom_score = 0
                    # 외국인 순매수가 양수이고 모멘텀도 양수일 때만
                    if foreign_score > 0:
                        score = mom_score * 100  # 모멘텀 기준 정렬
                    else:
                        score = -999
                else:
                    score = past["chg_qty"].sum()

                scores[ticker] = {"score": score, "price": current["cur_prc"]}

            # 상위 N개 종목 선정
            sorted_tickers = sorted(
                scores.keys(), key=lambda x: scores[x]["score"], reverse=True
            )
            selected = sorted_tickers[:top_n]

            # 기존 보유 종목 매도
            for ticker, shares in list(holdings.items()):
                if ticker in scores:
                    sell_price = scores[ticker]["price"]
                    sell_value = shares * sell_price * (1 - fee_rate)
                    capital += sell_value
            holdings = {}

            # 새 종목 매수 (균등 배분)
            if len(selected) > 0:
                allocation = capital / len(selected)
                for ticker in selected:
                    if ticker in scores and scores[ticker]["price"] > 0:
                        buy_price = scores[ticker]["price"]
                        shares = int(allocation * (1 - fee_rate) / buy_price)
                        if shares > 0:
                            cost = shares * buy_price * (1 + fee_rate)
                            capital -= cost
                            holdings[ticker] = shares

            last_rebalance = i

        portfolio_values.append(
            {
                "date": current_date,
                "total_value": total_value,
                "capital": capital,
                "holdings": len(holdings),
            }
        )

    if len(portfolio_values) == 0:
        return None

    # 최종 가치 계산
    final_value = capital
    for ticker, shares in holdings.items():
        df = all_data[ticker]
        final_row = df[df["dt"] == dates[-1]]
        if len(final_row) > 0:
            final_value += shares * final_row["cur_prc"].values[0]

    total_return = (final_value - initial_capital) / initial_capital * 100

    # 성과 지표 계산
    pv_df = pd.DataFrame(portfolio_values)
    pv_df["daily_return"] = pv_df["total_value"].pct_change()

    # 최대 낙폭 계산
    pv_df["cummax"] = pv_df["total_value"].cummax()
    pv_df["drawdown"] = (pv_df["total_value"] - pv_df["cummax"]) / pv_df["cummax"]
    max_dd = pv_df["drawdown"].min() * 100

    # 샤프비율 계산
    if pv_df["daily_return"].std() > 0:
        sharpe = (
            pv_df["daily_return"].mean() / pv_df["daily_return"].std() * np.sqrt(252)
        )
    else:
        sharpe = 0

    years = (dates[-1] - dates[0]).days / 365.25
    if years > 0:
        cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100
    else:
        cagr = 0

    return {
        "strategy": strategy_name,
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return_pct": total_return,
        "cagr_pct": cagr,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "trading_days": len(dates),
        "start_date": str(dates[0].date()),
        "end_date": str(dates[-1].date()),
    }


def backtest_momentum_rotation(
    all_data: dict, lookback: int = 20, top_n: int = 3, rebalance_freq: int = 5
):
    """모멘텀 로테이션 전략"""
    strategy_name = f"MomRotation_L{lookback}_Top{top_n}_R{rebalance_freq}"
    return backtest_portfolio_strategy(
        all_data,
        strategy_name,
        top_n=top_n,
        rebalance_freq=rebalance_freq,
        score_type="price_mom",
    )


def backtest_foreign_rotation(
    all_data: dict, lookback: int = 20, top_n: int = 3, rebalance_freq: int = 10
):
    """외국인 순매수 로테이션 전략"""
    strategy_name = f"ForeignRotation_L{lookback}_Top{top_n}_R{rebalance_freq}"
    return backtest_portfolio_strategy(
        all_data,
        strategy_name,
        top_n=top_n,
        rebalance_freq=rebalance_freq,
        score_type="foreign_sum",
    )


def backtest_combined_rotation(
    all_data: dict, top_n: int = 3, rebalance_freq: int = 10
):
    """외국인 + 모멘텀 결합 로테이션"""
    strategy_name = f"CombinedRotation_Top{top_n}_R{rebalance_freq}"
    return backtest_portfolio_strategy(
        all_data,
        strategy_name,
        top_n=top_n,
        rebalance_freq=rebalance_freq,
        score_type="combined",
    )


def backtest_long_short(all_data: dict, top_n: int = 3, rebalance_freq: int = 10):
    """롱숏 전략 (외국인 상위 매수, 하위 공매도 시뮬레이션)"""

    date_sets = [set(df["dt"].values) for df in all_data.values()]
    common_dates = sorted(set.intersection(*date_sets))

    if len(common_dates) < 100:
        return None

    dates = pd.DatetimeIndex(common_dates)

    initial_capital = 100_000_000
    capital = initial_capital
    long_holdings = {}
    short_holdings = {}
    fee_rate = 0.001

    portfolio_values = []
    last_rebalance = 0

    for i, date in enumerate(dates):
        current_date = pd.Timestamp(date)

        # 포지션 가치 계산
        long_value = 0
        short_pnl = 0

        for ticker, (shares, entry_price) in long_holdings.items():
            df = all_data[ticker]
            current_row = df[df["dt"] == current_date]
            if len(current_row) > 0:
                long_value += shares * current_row["cur_prc"].values[0]

        for ticker, (shares, entry_price) in short_holdings.items():
            df = all_data[ticker]
            current_row = df[df["dt"] == current_date]
            if len(current_row) > 0:
                current_price = current_row["cur_prc"].values[0]
                # 숏 포지션 손익 = (진입가 - 현재가) * 수량
                short_pnl += (entry_price - current_price) * shares

        total_value = capital + long_value + short_pnl

        # 리밸런싱
        if i - last_rebalance >= rebalance_freq:
            scores = {}
            for ticker, df in all_data.items():
                df_until = df[df["dt"] <= current_date].tail(21)
                if len(df_until) < 20:
                    continue
                current = df_until.iloc[-1]
                past = df_until.iloc[:-1]
                score = past["chg_qty"].sum()
                scores[ticker] = {"score": score, "price": current["cur_prc"]}

            sorted_tickers = sorted(
                scores.keys(), key=lambda x: scores[x]["score"], reverse=True
            )
            long_selected = sorted_tickers[:top_n]
            short_selected = (
                sorted_tickers[-top_n:] if len(sorted_tickers) >= 2 * top_n else []
            )

            # 기존 포지션 청산
            for ticker, (shares, _) in list(long_holdings.items()):
                if ticker in scores:
                    capital += shares * scores[ticker]["price"] * (1 - fee_rate)
            for ticker, (shares, entry_price) in list(short_holdings.items()):
                if ticker in scores:
                    capital += (
                        entry_price - scores[ticker]["price"]
                    ) * shares - scores[ticker]["price"] * fee_rate

            long_holdings = {}
            short_holdings = {}

            # 새 포지션 구축
            long_alloc = capital * 0.5 / max(len(long_selected), 1)
            short_alloc = capital * 0.5 / max(len(short_selected), 1)

            for ticker in long_selected:
                if ticker in scores and scores[ticker]["price"] > 0:
                    price = scores[ticker]["price"]
                    shares = int(long_alloc * (1 - fee_rate) / price)
                    if shares > 0:
                        capital -= shares * price * (1 + fee_rate)
                        long_holdings[ticker] = (shares, price)

            for ticker in short_selected:
                if ticker in scores and scores[ticker]["price"] > 0:
                    price = scores[ticker]["price"]
                    shares = int(short_alloc / price)
                    if shares > 0:
                        # 공매도 증거금은 별도로 관리 (간소화)
                        short_holdings[ticker] = (shares, price)

            last_rebalance = i

        portfolio_values.append({"date": current_date, "total_value": total_value})

    if len(portfolio_values) == 0:
        return None

    pv_df = pd.DataFrame(portfolio_values)
    final_value = pv_df["total_value"].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100

    pv_df["daily_return"] = pv_df["total_value"].pct_change()
    pv_df["cummax"] = pv_df["total_value"].cummax()
    pv_df["drawdown"] = (pv_df["total_value"] - pv_df["cummax"]) / pv_df["cummax"]
    max_dd = pv_df["drawdown"].min() * 100

    if pv_df["daily_return"].std() > 0:
        sharpe = (
            pv_df["daily_return"].mean() / pv_df["daily_return"].std() * np.sqrt(252)
        )
    else:
        sharpe = 0

    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (
        ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    )

    return {
        "strategy": f"LongShort_Top{top_n}_R{rebalance_freq}",
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return_pct": total_return,
        "cagr_pct": cagr,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "trading_days": len(dates),
        "start_date": str(dates[0].date()),
        "end_date": str(dates[-1].date()),
    }


def run_all_portfolio_backtests():
    """모든 포트폴리오 전략 백테스트 실행"""
    print("=" * 70)
    print("키움 데이터 기반 포트폴리오 & 앙상블 전략 백테스트")
    print("=" * 70)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    all_data = load_all_merged_data()
    print(f"  로드된 종목: {len(all_data)}개")

    for ticker, df in all_data.items():
        name = df["name"].iloc[0] if "name" in df.columns else ticker
        print(
            f"    {ticker}: {name} ({len(df)} rows, {df['dt'].min().date()} ~ {df['dt'].max().date()})"
        )

    results = []

    # 포트폴리오 전략 테스트
    print("\n[2] 포트폴리오 전략 테스트...")

    # 외국인 순매수 로테이션
    print("\n  (A) 외국인 순매수 로테이션 전략:")
    for top_n in [2, 3, 5]:
        for rebalance in [5, 10, 20]:
            result = backtest_foreign_rotation(
                all_data, top_n=top_n, rebalance_freq=rebalance
            )
            if result:
                results.append(result)
                print(
                    f"    {result['strategy']}: 수익률={result['total_return_pct']:.1f}%, "
                    f"CAGR={result['cagr_pct']:.1f}%, MDD={result['max_drawdown_pct']:.1f}%, "
                    f"Sharpe={result['sharpe_ratio']:.2f}"
                )

    # 모멘텀 로테이션
    print("\n  (B) 모멘텀 로테이션 전략:")
    for top_n in [2, 3, 5]:
        for rebalance in [5, 10, 20]:
            result = backtest_momentum_rotation(
                all_data, top_n=top_n, rebalance_freq=rebalance
            )
            if result:
                results.append(result)
                print(
                    f"    {result['strategy']}: 수익률={result['total_return_pct']:.1f}%, "
                    f"CAGR={result['cagr_pct']:.1f}%, MDD={result['max_drawdown_pct']:.1f}%, "
                    f"Sharpe={result['sharpe_ratio']:.2f}"
                )

    # 외국인 + 모멘텀 결합
    print("\n  (C) 외국인 + 모멘텀 결합 전략:")
    for top_n in [2, 3, 5]:
        for rebalance in [5, 10, 20]:
            result = backtest_combined_rotation(
                all_data, top_n=top_n, rebalance_freq=rebalance
            )
            if result:
                results.append(result)
                print(
                    f"    {result['strategy']}: 수익률={result['total_return_pct']:.1f}%, "
                    f"CAGR={result['cagr_pct']:.1f}%, MDD={result['max_drawdown_pct']:.1f}%, "
                    f"Sharpe={result['sharpe_ratio']:.2f}"
                )

    # 롱숏 전략
    print("\n  (D) 롱숏 전략:")
    for top_n in [2, 3]:
        for rebalance in [5, 10, 20]:
            result = backtest_long_short(
                all_data, top_n=top_n, rebalance_freq=rebalance
            )
            if result:
                results.append(result)
                print(
                    f"    {result['strategy']}: 수익률={result['total_return_pct']:.1f}%, "
                    f"CAGR={result['cagr_pct']:.1f}%, MDD={result['max_drawdown_pct']:.1f}%, "
                    f"Sharpe={result['sharpe_ratio']:.2f}"
                )

    # 결과 정리
    print("\n" + "=" * 70)
    print("포트폴리오 전략 백테스트 결과 요약")
    print("=" * 70)

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("total_return_pct", ascending=False)

        print(f"\n전체 테스트: {len(results)}개 포트폴리오 전략")
        profitable = results_df[results_df["total_return_pct"] > 0]
        print(
            f"수익 전략: {len(profitable)}개 ({len(profitable)/len(results)*100:.1f}%)"
        )

        print("\n[상위 15개 전략]")
        print("-" * 70)
        for i, row in results_df.head(15).iterrows():
            print(
                f"{row['strategy'][:40]:<40} | "
                f"수익: {row['total_return_pct']:>8.1f}% | "
                f"CAGR: {row['cagr_pct']:>6.1f}% | "
                f"MDD: {row['max_drawdown_pct']:>6.1f}% | "
                f"Sharpe: {row['sharpe_ratio']:>5.2f}"
            )

        # 위험조정수익률 기준 상위
        print("\n[샤프비율 상위 10개]")
        print("-" * 70)
        sharpe_sorted = results_df.sort_values("sharpe_ratio", ascending=False)
        for i, row in sharpe_sorted.head(10).iterrows():
            print(
                f"{row['strategy'][:40]:<40} | "
                f"Sharpe: {row['sharpe_ratio']:>5.2f} | "
                f"수익: {row['total_return_pct']:>8.1f}% | "
                f"MDD: {row['max_drawdown_pct']:>6.1f}%"
            )

        # 결과 저장
        output_file = (
            "E:/투자/Multi-Asset Strategy Platform/portfolio_backtest_results.csv"
        )
        results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n결과 저장: {output_file}")

        return results_df

    return None


if __name__ == "__main__":
    results = run_all_portfolio_backtests()
