# -*- coding: utf-8 -*-
"""
VIX < SMA10 전략 시그널 체크 스크립트.

사용법:
    python scripts/vix_sma10_check.py

매일 장 시작 전(08:30) 실행하여 당일 매매 방향 확인.
"""

import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(__file__).replace("\\scripts\\vix_sma10_check.py", ""))

from libs.strategies.vix_sma10_stocks import (
    VIXSMA10Tier1Strategy,
    VIXSMA10Tier2Strategy,
    VIXSMA10AllTiersStrategy,
    VIX_VALID_STOCKS,
)


def main():
    print("=" * 70)
    print("VIX < SMA10 KOSPI 개별 종목 전략")
    print(f"체크 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Tier 1 전략 사용 (최상위 5종목)
    strategy = VIXSMA10Tier1Strategy()

    # VIX 시그널 계산
    signal, metrics = strategy.calculate_signal()

    print("[VIX 지표]")
    print(f"  VIX (T-1):    {metrics.get('vix_t1', 'N/A')}")
    print(f"  VIX (T-2):    {metrics.get('vix_t2', 'N/A')}")
    print(f"  VIX SMA10:    {metrics.get('vix_sma10', 'N/A')}")
    print(f"  차이:         {metrics.get('vix_diff', 'N/A')} ({metrics.get('vix_pct_diff', 0):+.1f}%)")
    print(f"  데이터 기준:  {metrics.get('date', 'N/A')}")
    print()

    print("[오늘 시그널]")
    if signal == 1:
        print("  ★★★ LONG (매수) ★★★")
        print("  VIX < SMA10: 시장 공포 낮음 → 주식 보유")
    else:
        print("  ○○○ CASH (현금) ○○○")
        print("  VIX >= SMA10: 시장 공포 높음 → 현금 보유")
    print()

    # 종목별 정보 출력
    print("[대상 종목 - Tier 1 (Sharpe >= 1.0)]")
    print("-" * 70)
    print(f"{'코드':<8} {'종목명':<15} {'Sharpe':>8} {'CAGR':>10} {'오늘':>8}")
    print("-" * 70)

    for ticker in strategy.get_valid_symbols():
        info = VIX_VALID_STOCKS[ticker]
        action = "매수" if signal == 1 else "현금"
        print(f"{ticker:<8} {info['name']:<15} {info['sharpe']:>8.2f} {info['cagr']:>10} {action:>8}")

    print()
    print("[추가 옵션]")
    print("  --tier2: Tier 1+2 (14종목, Sharpe >= 0.5)")
    print("  --all:   전체 (25종목, Sharpe >= 0.3)")
    print()

    # 다른 티어도 표시 (참고용)
    tier2 = VIXSMA10Tier2Strategy()
    all_tiers = VIXSMA10AllTiersStrategy()

    print(f"[종목 수 요약]")
    print(f"  Tier 1: {len(strategy.get_valid_symbols())}개 (Sharpe >= 1.0)")
    print(f"  Tier 2: {len(tier2.get_valid_symbols())}개 (Sharpe >= 0.5)")
    print(f"  전체:   {len(all_tiers.get_valid_symbols())}개 (Sharpe >= 0.3)")
    print()

    print("=" * 70)
    print("※ 키움증권 소수점 거래 가능 종목만 포함")
    print("※ 체결 시간: 10:00, 11:00, 13:00, 15:00")
    print("=" * 70)


if __name__ == "__main__":
    main()
