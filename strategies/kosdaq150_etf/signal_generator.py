"""
KOSDAQ 150 ETF 일일 신호 생성기
================================

매일 장 마감 후 실행하여 다음날 거래 신호 확인

사용법:
    python run_kosdaq150_etf.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
import json

from .etf_strategy import (
    KOSDAQ150ETFStrategy,
    PortfolioConfig,
    Position,
    TradeRecommendation,
)


class KOSDAQ150ETFSignalGenerator:
    """
    ETF 일일 신호 생성기

    사용법:
        generator = KOSDAQ150ETFSignalGenerator()
        generator.run()
    """

    def __init__(self, data_path: Optional[Path] = None, capital: float = 1_000_000):
        self.data_path = data_path or Path("E:/투자/data/kosdaq_futures")
        self.output_path = self.data_path / "validated_strategies"

        # 설정
        config = PortfolioConfig(initial_capital=capital)
        self.strategy = KOSDAQ150ETFStrategy(config)
        self.capital = capital

        # 데이터
        self.data = None
        self.last_date = None

    def load_data(self) -> pd.DataFrame:
        """데이터 로드"""
        clean_path = self.data_path / "kosdaq150_futures_ohlcv_clean.parquet"
        cache_path = self.data_path / "kosdaq150_futures_ohlcv.parquet"

        if clean_path.exists():
            self.data = pd.read_parquet(clean_path)
        elif cache_path.exists():
            self.data = pd.read_parquet(cache_path)
        else:
            raise FileNotFoundError(f"데이터 없음: {clean_path}")

        self.last_date = self.data.index[-1]
        return self.data

    def get_recommendation(self) -> TradeRecommendation:
        """오늘 거래 권장 조회"""
        if self.data is None:
            self.load_data()
        return self.strategy.get_recommendation(self.data)

    def calculate_position_size(self, etf_price: float = 7000) -> Dict:
        """
        포지션 크기 계산

        Args:
            etf_price: ETF 현재가 (기본 7,000원)

        Returns:
            포지션 정보
        """
        invest_amount = self.capital * self.strategy.config.position_size_pct
        shares = int(invest_amount / etf_price)
        actual_amount = shares * etf_price

        return {
            "total_capital": self.capital,
            "invest_amount": invest_amount,
            "etf_price": etf_price,
            "shares": shares,
            "actual_amount": actual_amount,
            "reserve": self.capital - actual_amount,
        }

    def print_signal(self):
        """신호 출력"""
        rec = self.get_recommendation()

        print("=" * 65)
        print("  KOSDAQ 150 ETF 일일 거래 신호")
        print("=" * 65)
        print(
            f"  기준일: {self.last_date.strftime('%Y-%m-%d') if self.last_date else 'N/A'}"
        )
        print(f"  자본금: {self.capital:,.0f}원")
        print("=" * 65)

        # 개별 전략 신호
        print("\n[전략별 신호]")
        today_signals = self.strategy.get_today_signals(self.data)
        for name, sig in today_signals.items():
            if sig:
                direction = "▲ 롱" if sig.direction == 1 else "▼ 숏"
                print(
                    f"  {name:<15}: {direction} (강도: {sig.strength:.0%}) - {sig.reason}"
                )
            else:
                print(f"  {name:<15}: - 신호 없음")

        # 종합 권장
        print("\n" + "-" * 65)
        print("[종합 권장]")

        if rec.action == "BUY" and rec.position_type == Position.LONG:
            print(f"\n  ▲▲▲ 롱 진입 (강도: {rec.strength:.0%})")
            print(f"\n  매수 ETF: {rec.etf_name}")
            print(f"  종목코드: {rec.etf_code}")

            # 포지션 크기
            pos = self.calculate_position_size(7000)
            print(f"\n  [주문 가이드]")
            print(f"  - 투자금액: {pos['invest_amount']:,.0f}원")
            print(f"  - 예상 수량: {pos['shares']:,}주")
            print(f"  - 예비 현금: {pos['reserve']:,.0f}원")

        elif rec.action == "BUY" and rec.position_type == Position.SHORT:
            print(f"\n  ▼▼▼ 숏 진입 (강도: {rec.strength:.0%})")
            print(f"\n  매수 ETF: {rec.etf_name}")
            print(f"  종목코드: {rec.etf_code}")

            pos = self.calculate_position_size(3500)
            print(f"\n  [주문 가이드]")
            print(f"  - 투자금액: {pos['invest_amount']:,.0f}원")
            print(f"  - 예상 수량: {pos['shares']:,}주")
            print(f"  - 예비 현금: {pos['reserve']:,.0f}원")

        else:
            print(f"\n  ━━━ HOLD (관망)")
            print(f"\n  기존 포지션이 있다면 유지")
            print(f"  신규 진입 대기")

        # 사유
        print(f"\n  [사유]")
        for reason in rec.reasons:
            print(f"  - {reason}")

        # 행동 가이드
        print("\n" + "-" * 65)
        print("[행동 가이드]")

        if rec.action == "BUY":
            print(f"""
  1. 내일 장 시작 전 호가창 확인
  2. 시초가 또는 09:05 이후 매수
  3. {rec.etf_code} ({rec.etf_name}) 지정가 주문
  4. 체결 후 손절가 설정 (-10%)
            """)
        else:
            print("""
  1. 기존 포지션 유지
  2. 손절/익절 조건 확인
  3. 내일 신호 재확인
            """)

        print("=" * 65)

    def save_signal(self) -> Path:
        """신호 저장"""
        rec = self.get_recommendation()
        today_signals = self.strategy.get_today_signals(self.data)

        output = {
            "timestamp": datetime.now().isoformat(),
            "date": str(self.last_date.date()) if self.last_date else None,
            "capital": self.capital,
            "recommendation": {
                "action": rec.action,
                "position": rec.position_type.value,
                "etf_code": rec.etf_code,
                "etf_name": rec.etf_name,
                "strength": rec.strength,
                "reasons": rec.reasons,
            },
            "individual_signals": {
                name: {
                    "direction": sig.direction if sig else None,
                    "strength": sig.strength if sig else None,
                    "reason": sig.reason if sig else None,
                }
                for name, sig in today_signals.items()
            },
        }

        filename = f"etf_signal_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = self.output_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n신호 저장: {filepath}")
        return filepath

    def run_backtest(self) -> Dict:
        """백테스트 실행"""
        if self.data is None:
            self.load_data()

        result = self.strategy.backtest(self.data)

        print("\n" + "=" * 50)
        print("백테스트 결과")
        print("=" * 50)
        print(f"Sharpe Ratio: {result['sharpe']:.3f}")
        print(f"CAGR: {result['cagr']*100:.1f}%")
        print(f"Total Return: {result['total_return']*100:.1f}%")
        print(f"MDD: {result['mdd']*100:.1f}%")
        print(f"Win Rate: {result['win_rate']*100:.1f}%")
        print("=" * 50)

        return result

    def run(self, save: bool = False):
        """실행"""
        self.load_data()
        self.print_signal()
        if save:
            self.save_signal()


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="KOSDAQ 150 ETF 신호 생성")
    parser.add_argument(
        "--capital", type=int, default=1_000_000, help="투자 자본금 (기본: 100만원)"
    )
    parser.add_argument("--save", action="store_true", help="신호 저장")
    parser.add_argument("--backtest", action="store_true", help="백테스트 실행")
    parser.add_argument("--summary", action="store_true", help="전략 요약")

    args = parser.parse_args()

    generator = KOSDAQ150ETFSignalGenerator(capital=args.capital)

    if args.summary:
        print(generator.strategy.get_summary())
    elif args.backtest:
        generator.load_data()
        generator.run_backtest()
    else:
        generator.run(save=args.save)


if __name__ == "__main__":
    main()
