# -*- coding: utf-8 -*-
"""
키움 REST API 다운로드 데이터 분석
"""

import pandas as pd
import os
from datetime import datetime

DATA_PATH = "E:/투자/data/kiwoom_investor"


def analyze_foreign_data():
    """외국인 데이터 분석"""
    print("=" * 60)
    print("외국인 투자 데이터 분석")
    print("=" * 60)

    foreign_files = [f for f in os.listdir(DATA_PATH) if f.endswith("_foreign.csv")]

    results = []
    for file in sorted(foreign_files):
        filepath = os.path.join(DATA_PATH, file)
        df = pd.read_csv(filepath, encoding="utf-8-sig")

        ticker = file.replace("_foreign.csv", "")
        name = df["name"].iloc[0] if "name" in df.columns else ticker

        # 데이터 통계
        df["dt"] = pd.to_datetime(df["dt"], format="%Y%m%d")
        df = df.sort_values("dt")

        # chg_qty가 외국인 순매수량
        df["chg_qty"] = pd.to_numeric(df["chg_qty"], errors="coerce")
        df["wght"] = df["wght"].astype(str).str.replace("+", "").astype(float)

        stats = {
            "ticker": ticker,
            "name": name,
            "data_count": len(df),
            "start_date": df["dt"].min().strftime("%Y-%m-%d"),
            "end_date": df["dt"].max().strftime("%Y-%m-%d"),
            "latest_foreign_ratio": df.iloc[-1]["wght"],
            "avg_daily_net_buy": df["chg_qty"].mean(),
            "total_net_buy_30d": df.head(30)["chg_qty"].sum(),
            "total_net_buy_60d": df.head(60)["chg_qty"].sum(),
        }
        results.append(stats)

        print(f"\n{name} ({ticker})")
        print(
            f"  데이터: {stats['start_date']} ~ {stats['end_date']} ({stats['data_count']}일)"
        )
        print(f"  현재 외국인 지분율: {stats['latest_foreign_ratio']:.2f}%")
        print(f"  30일 누적 순매수: {stats['total_net_buy_30d']:,.0f}주")
        print(f"  60일 누적 순매수: {stats['total_net_buy_60d']:,.0f}주")

    # 요약 데이터프레임 저장
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(
        f"{DATA_PATH}/foreign_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n요약 저장: {DATA_PATH}/foreign_summary.csv")

    return summary_df


def analyze_daily_data():
    """일봉 데이터 분석"""
    print("\n" + "=" * 60)
    print("일봉 차트 데이터 분석")
    print("=" * 60)

    daily_files = [f for f in os.listdir(DATA_PATH) if f.endswith("_daily.csv")]

    results = []
    for file in sorted(daily_files):
        filepath = os.path.join(DATA_PATH, file)
        df = pd.read_csv(filepath, encoding="utf-8-sig")

        ticker = file.replace("_daily.csv", "")
        name = df["name"].iloc[0] if "name" in df.columns else ticker

        # 데이터 통계
        df["dt"] = pd.to_datetime(df["dt"], format="%Y%m%d")
        df = df.sort_values("dt")

        # 가격 데이터 정리 (부호 제거)
        for col in ["cur_prc", "open_pric", "high_pric", "low_pric"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace("+", "")
                    .str.replace("-", "")
                    .astype(float)
                )

        latest = df.iloc[-1]
        stats = {
            "ticker": ticker,
            "name": name,
            "data_count": len(df),
            "start_date": df["dt"].min().strftime("%Y-%m-%d"),
            "end_date": df["dt"].max().strftime("%Y-%m-%d"),
            "latest_close": latest["cur_prc"],
            "avg_volume": df["trde_qty"].mean(),
        }
        results.append(stats)

        print(f"\n{name} ({ticker})")
        print(
            f"  데이터: {stats['start_date']} ~ {stats['end_date']} ({stats['data_count']}일)"
        )
        print(f"  최근 종가: {stats['latest_close']:,.0f}원")
        print(f"  평균 거래량: {stats['avg_volume']:,.0f}주")

    return pd.DataFrame(results) if results else None


def merge_data_for_strategy():
    """전략 분석용 데이터 병합"""
    print("\n" + "=" * 60)
    print("전략 분석용 데이터 병합")
    print("=" * 60)

    # 외국인 데이터와 일봉 데이터가 모두 있는 종목 찾기
    foreign_files = {
        f.replace("_foreign.csv", "")
        for f in os.listdir(DATA_PATH)
        if f.endswith("_foreign.csv")
    }
    daily_files = {
        f.replace("_daily.csv", "")
        for f in os.listdir(DATA_PATH)
        if f.endswith("_daily.csv")
    }

    common_tickers = foreign_files & daily_files
    print(f"외국인 + 일봉 데이터 모두 있는 종목: {len(common_tickers)}개")

    merged_data = {}
    for ticker in common_tickers:
        # 외국인 데이터
        foreign_df = pd.read_csv(
            f"{DATA_PATH}/{ticker}_foreign.csv", encoding="utf-8-sig"
        )
        foreign_df["dt"] = pd.to_datetime(foreign_df["dt"], format="%Y%m%d")
        foreign_df = foreign_df.sort_values("dt")

        # 일봉 데이터
        daily_df = pd.read_csv(f"{DATA_PATH}/{ticker}_daily.csv", encoding="utf-8-sig")
        daily_df["dt"] = pd.to_datetime(daily_df["dt"], format="%Y%m%d")
        daily_df = daily_df.sort_values("dt")

        # 병합 (날짜 기준)
        merged = pd.merge(
            daily_df,
            foreign_df[["dt", "chg_qty", "wght", "poss_stkcnt"]],
            on="dt",
            how="left",
        )
        merged["ticker"] = ticker

        # 가격 데이터 정리
        for col in ["cur_prc", "open_pric", "high_pric", "low_pric"]:
            if col in merged.columns:
                merged[col] = (
                    merged[col]
                    .astype(str)
                    .str.replace("+", "")
                    .str.replace("-", "")
                    .astype(float)
                )

        merged["chg_qty"] = pd.to_numeric(merged["chg_qty"], errors="coerce")
        merged["wght"] = merged["wght"].astype(str).str.replace("+", "").astype(float)

        # 수익률 계산
        merged["return"] = merged["cur_prc"].pct_change()

        # 외국인 누적 순매수 (5일, 10일, 20일)
        merged["foreign_net_5d"] = merged["chg_qty"].rolling(5).sum()
        merged["foreign_net_10d"] = merged["chg_qty"].rolling(10).sum()
        merged["foreign_net_20d"] = merged["chg_qty"].rolling(20).sum()

        merged_data[ticker] = merged

        # 저장
        merged.to_csv(
            f"{DATA_PATH}/{ticker}_merged.csv", index=False, encoding="utf-8-sig"
        )
        name = merged["name"].iloc[0] if "name" in merged.columns else ticker
        print(f"  {name} ({ticker}): {len(merged)}일 데이터 병합 완료")

    return merged_data


def main():
    print("키움 REST API 데이터 분석")
    print("=" * 60)
    print(f"데이터 경로: {DATA_PATH}")
    print(f"총 파일 수: {len(os.listdir(DATA_PATH))}개")

    # 1. 외국인 데이터 분석
    foreign_summary = analyze_foreign_data()

    # 2. 일봉 데이터 분석
    daily_summary = analyze_daily_data()

    # 3. 전략용 데이터 병합
    merged_data = merge_data_for_strategy()

    # 4. 외국인 순매수 상위/하위 종목
    if foreign_summary is not None and len(foreign_summary) > 0:
        print("\n" + "=" * 60)
        print("최근 30일 외국인 순매수 상위 종목")
        print("=" * 60)
        top5 = foreign_summary.nlargest(5, "total_net_buy_30d")
        for _, row in top5.iterrows():
            print(
                f"  {row['name']} ({row['ticker']}): {row['total_net_buy_30d']:,.0f}주"
            )

        print("\n최근 30일 외국인 순매도 상위 종목")
        print("-" * 40)
        bottom5 = foreign_summary.nsmallest(5, "total_net_buy_30d")
        for _, row in bottom5.iterrows():
            print(
                f"  {row['name']} ({row['ticker']}): {row['total_net_buy_30d']:,.0f}주"
            )

    print("\n분석 완료!")


if __name__ == "__main__":
    main()
