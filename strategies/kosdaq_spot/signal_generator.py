# -*- coding: utf-8 -*-
"""
코스닥 현물 전략 - 일일 신호 생성기
=================================

매일 장 마감 후 실행하여 다음날 거래 신호 확인

사용법:
    python -m strategies.kosdaq_spot.signal_generator

또는:
    from strategies.kosdaq_spot import KOSDAQSpotSignalGenerator
    generator = KOSDAQSpotSignalGenerator()
    generator.run()
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .core_strategies import (
    MultiTFShortStrategy,
    SignalType,
    StrategyParams,
    TradingSignal,
)
from .universe_manager import KOSDAQUniverseManager


class KOSDAQSpotSignalGenerator:
    """
    코스닥 현물 일일 신호 생성기

    사용법:
        generator = KOSDAQSpotSignalGenerator()
        generator.run()
    """

    def __init__(self, params: Optional[StrategyParams] = None):
        self.params = params or StrategyParams()
        self.strategy = MultiTFShortStrategy(self.params)
        self.universe_manager = KOSDAQUniverseManager()

        self.signals: List[TradingSignal] = []
        self.buy_signals: List[TradingSignal] = []
        self.scan_date: Optional[datetime] = None

    def scan_universe(self) -> List[TradingSignal]:
        """
        유니버스 전체 스캔

        Returns:
            모든 종목의 신호 리스트
        """
        universe = self.universe_manager.get_universe()
        self.signals = []
        self.buy_signals = []
        self.scan_date = datetime.now()

        print(f"\n스캔 시작: {len(universe)}개 종목")
        print("-" * 50)

        for i, ticker in enumerate(universe):
            df = self.universe_manager.get_stock_data(ticker)
            if df is None or len(df) < 60:
                continue

            signal = self.strategy.get_latest_signal(df, ticker)
            self.signals.append(signal)

            if signal.signal_type == SignalType.BUY:
                self.buy_signals.append(signal)

            # 진행 상황 (50개마다)
            if (i + 1) % 50 == 0:
                print(f"  진행: {i+1}/{len(universe)}")

        print(
            f"\n스캔 완료: {len(self.signals)}개 분석, {len(self.buy_signals)}개 매수 신호"
        )
        return self.signals

    def get_buy_signals(self) -> List[TradingSignal]:
        """매수 신호 종목 반환"""
        if not self.signals:
            self.scan_universe()
        return self.buy_signals

    def get_signal_summary(self) -> Dict:
        """신호 요약"""
        if not self.signals:
            self.scan_universe()

        # 강도별 분류
        strong_buys = [s for s in self.buy_signals if s.strength == 1.0]
        weak_buys = [s for s in self.buy_signals if s.strength < 1.0]

        return {
            "scan_date": self.scan_date.isoformat() if self.scan_date else None,
            "total_scanned": len(self.signals),
            "buy_signals": len(self.buy_signals),
            "strong_buys": len(strong_buys),
            "weak_buys": len(weak_buys),
            "buy_rate": (
                len(self.buy_signals) / len(self.signals) * 100 if self.signals else 0
            ),
        }

    def print_signals(self):
        """신호 출력"""
        if not self.signals:
            self.scan_universe()

        summary = self.get_signal_summary()

        print("\n" + "=" * 70)
        print("코스닥 현물 전략 - 일일 신호")
        print(f"기준일: {summary['scan_date'][:10] if summary['scan_date'] else 'N/A'}")
        print("전략: Multi_TF_Short (5,20,10,40,50)")
        print("=" * 70)

        print("\n[스캔 결과]")
        print(f"  총 스캔: {summary['total_scanned']}개")
        print(f"  매수 신호: {summary['buy_signals']}개 ({summary['buy_rate']:.1f}%)")
        print(f"  - 강한 신호 (3/3 조건): {summary['strong_buys']}개")
        print(f"  - 약한 신호 (2/3 조건): {summary['weak_buys']}개")

        if self.buy_signals:
            print("\n[매수 신호 종목]")
            print(
                f"{'순위':<4} {'종목':<10} {'강도':<6} {'가격':<12} {'외국인%':<8} {'사유'}"
            )
            print("-" * 75)

            # 강도순 정렬
            sorted_signals = sorted(
                self.buy_signals, key=lambda x: x.strength, reverse=True
            )

            for i, sig in enumerate(sorted_signals[:20], 1):
                strength_str = f"{sig.strength*100:.0f}%"
                print(
                    f"{i:<4} {sig.ticker:<10} {strength_str:<6} {sig.price:<12,.0f} "
                    f"{sig.foreign_wght:<8.1f} {sig.reason[:30]}"
                )

            if len(sorted_signals) > 20:
                print(f"  ... 외 {len(sorted_signals) - 20}개 종목")

        else:
            print("\n[매수 신호 없음]")

        print("\n" + "=" * 70)
        print("[행동 가이드]")
        if summary["strong_buys"] > 0:
            print(f"  - {summary['strong_buys']}개 종목 강한 매수 신호")
            print("  - 다음날 시초가에 분산 매수 권장")
            print("  - 종목당 최대 10% 비중")
        elif summary["weak_buys"] > 0:
            print(f"  - {summary['weak_buys']}개 종목 약한 매수 신호")
            print("  - 추가 확인 후 선별 매수 권장")
        else:
            print("  - 매수 신호 없음")
            print("  - 기존 보유 종목 청산 검토")
        print("=" * 70)

    def save_signals(self, output_dir: Optional[str] = None) -> str:
        """신호를 JSON으로 저장"""
        if not self.signals:
            self.scan_universe()

        if output_dir is None:
            output_dir = (
                "E:/투자/Multi-Asset Strategy Platform/strategies/kosdaq_spot/signals"
            )

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        filename = f"daily_signal_{datetime.now().strftime('%Y%m%d')}.json"
        output_path = Path(output_dir) / filename

        data = {
            "generated": datetime.now().isoformat(),
            "strategy": "Multi_TF_Short",
            "params": asdict(self.params),
            "summary": self.get_signal_summary(),
            "buy_signals": [
                {
                    "ticker": s.ticker,
                    "strength": s.strength,
                    "price": s.price,
                    "foreign_wght": s.foreign_wght,
                    "reason": s.reason,
                    "conditions": s.conditions,
                }
                for s in sorted(
                    self.buy_signals, key=lambda x: x.strength, reverse=True
                )
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n신호 저장: {output_path}")
        return str(output_path)

    def run(self):
        """전체 실행"""
        self.scan_universe()
        self.print_signals()
        self.save_signals()


def main():
    """메인 실행"""
    generator = KOSDAQSpotSignalGenerator()
    generator.run()


if __name__ == "__main__":
    main()
