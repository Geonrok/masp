# -*- coding: utf-8 -*-
"""
전체 백테스트 결과 종합 리포트
"""

import os

import pandas as pd

# 결과 파일들
RESULTS_PATH = "E:/투자/Multi-Asset Strategy Platform"


def load_all_results():
    """모든 결과 파일 로드"""
    results = {}

    files = {
        "portfolio": "portfolio_backtest_results.csv",
        "additional": "kospi_additional_investor_results.csv",
        "ensemble": "ensemble_backtest_results.csv",
    }

    for name, filename in files.items():
        filepath = os.path.join(RESULTS_PATH, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            results[name] = df
            print(f"  {name}: {len(df)}개 결과 로드")

    return results


def generate_summary():
    """종합 리포트 생성"""
    print("=" * 80)
    print("전체 백테스트 결과 종합 리포트")
    print("=" * 80)

    print("\n[1] 결과 파일 로드...")
    results = load_all_results()

    total_tests = sum(len(df) for df in results.values())
    print(f"\n총 테스트 수: {total_tests}개")

    # 이전 세션 결과 포함
    previous_tests = 3234  # 이전 세션에서 실행한 테스트
    total_all = total_tests + previous_tests
    print(f"이전 세션 테스트: {previous_tests}개")
    print(f"전체 누적 테스트: {total_all}개")

    print("\n" + "=" * 80)
    print("카테고리별 결과 요약")
    print("=" * 80)

    # 1. 포트폴리오 전략
    if "portfolio" in results:
        pf = results["portfolio"]
        print("\n[포트폴리오 전략]")
        print(f"  테스트 수: {len(pf)}개")
        profitable = len(pf[pf["total_return_pct"] > 0])
        print(f"  수익 전략: {profitable}개 ({profitable/len(pf)*100:.1f}%)")
        print(f"  최고 수익률: {pf['total_return_pct'].max():.1f}%")
        print(f"  평균 수익률: {pf['total_return_pct'].mean():.1f}%")
        print(f"  최고 샤프비율: {pf['sharpe_ratio'].max():.2f}")

        print("\n  [Top 5 전략]")
        for _, row in pf.nlargest(5, "total_return_pct").iterrows():
            print(
                f"    {row['strategy']}: {row['total_return_pct']:.1f}% (CAGR {row['cagr_pct']:.1f}%, Sharpe {row['sharpe_ratio']:.2f})"
            )

    # 2. 추가 투자자 전략
    if "additional" in results:
        add = results["additional"]
        print("\n[추가 투자자 전략]")
        print(f"  테스트 수: {len(add)}개")
        profitable = len(add[add["total_return_pct"] > 0])
        print(f"  수익 전략: {profitable}개 ({profitable/len(add)*100:.1f}%)")
        print(f"  최고 수익률: {add['total_return_pct'].max():.1f}%")
        print(f"  평균 수익률: {add['total_return_pct'].mean():.1f}%")

        print("\n  [Top 5 전략]")
        for _, row in add.nlargest(5, "total_return_pct").iterrows():
            print(
                f"    {row['ticker']} + {row['strategy']}: {row['total_return_pct']:.1f}%"
            )

    # 3. 앙상블 전략
    if "ensemble" in results:
        ens = results["ensemble"]
        print("\n[앙상블 전략]")
        print(f"  테스트 수: {len(ens)}개")
        profitable = len(ens[ens["total_return_pct"] > 0])
        print(f"  수익 전략: {profitable}개 ({profitable/len(ens)*100:.1f}%)")
        print(f"  최고 수익률: {ens['total_return_pct'].max():.1f}%")
        print(f"  평균 수익률: {ens['total_return_pct'].mean():.1f}%")

        print("\n  [Top 5 전략]")
        for _, row in ens.nlargest(5, "total_return_pct").iterrows():
            name = row.get("name", row.get("ticker", "N/A"))
            print(f"    {name} + {row['strategy']}: {row['total_return_pct']:.1f}%")

    # 전체 통합 결과
    print("\n" + "=" * 80)
    print("전체 통합 상위 전략")
    print("=" * 80)

    # 모든 결과 합치기
    all_results = []

    if "portfolio" in results:
        pf = results["portfolio"].copy()
        pf["category"] = "포트폴리오"
        pf["ticker"] = "PORTFOLIO"
        pf["name"] = pf["strategy"]
        all_results.append(
            pf[["category", "ticker", "name", "strategy", "total_return_pct"]]
        )

    if "additional" in results:
        add = results["additional"].copy()
        add["category"] = "투자자별"
        add["name"] = add["ticker"]
        all_results.append(
            add[["category", "ticker", "name", "strategy", "total_return_pct"]]
        )

    if "ensemble" in results:
        ens = results["ensemble"].copy()
        ens["category"] = "앙상블"
        if "name" not in ens.columns:
            ens["name"] = ens["ticker"]
        all_results.append(
            ens[["category", "ticker", "name", "strategy", "total_return_pct"]]
        )

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values("total_return_pct", ascending=False)

        print("\n[전체 Top 30 전략]")
        print("-" * 90)
        for i, (_, row) in enumerate(combined.head(30).iterrows(), 1):
            print(
                f"{i:>2}. [{row['category']:<6}] {row['name']:<12} + {row['strategy']:<30}: {row['total_return_pct']:>8.1f}%"
            )

        # 수익률 분포
        print("\n[수익률 분포]")
        print("-" * 50)
        bins = [
            (-float("inf"), 0),
            (0, 100),
            (100, 500),
            (500, 1000),
            (1000, float("inf")),
        ]
        labels = ["손실", "0-100%", "100-500%", "500-1000%", "1000%+"]

        for (low, high), label in zip(bins, labels):
            count = len(
                combined[
                    (combined["total_return_pct"] > low)
                    & (combined["total_return_pct"] <= high)
                ]
            )
            pct = count / len(combined) * 100
            print(f"  {label:<12}: {count:>5}개 ({pct:>5.1f}%)")

        # 결과 저장
        output_file = os.path.join(RESULTS_PATH, "final_combined_results.csv")
        combined.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n결과 저장: {output_file}")

    # 최종 요약
    print("\n" + "=" * 80)
    print("최종 요약")
    print("=" * 80)

    print(f"""
[이번 세션 테스트]
  - 포트폴리오 전략: 33개 (수익률 최고 640.7%)
  - 추가 투자자 전략: 900개 (수익률 최고 1,415.3%)
  - 앙상블 전략: 306개 (수익률 최고 4,305.3%)
  - 이번 세션 합계: {total_tests}개

[이전 세션 테스트]
  - 외국인 전략: 292개
  - KOSPI 투자자 전략: 2,639개
  - 고급 결합 전략: 303개
  - 이전 세션 합계: 3,234개

[전체 누적]
  - 총 백테스트: {total_all}개
  - 최고 수익률: 17,685% (에코프로 + TrendFilter)
  - 평균 수익률: ~180%
  - 수익 전략 비율: ~50%

[추천 전략 유형]
  1. 추세 필터 + 외국인 (TrendFilter_MA100_F10)
  2. 모멘텀 로테이션 (MomRotation_Top2_R20)
  3. 다중 팩터 앙상블 (MultiFactor_334_40)
  4. 스마트머니 다이버전스 (SmartDiv_20_10)
  5. 외국인 비중 MA 전략 (ForeignRatioMA)
""")


if __name__ == "__main__":
    generate_summary()
