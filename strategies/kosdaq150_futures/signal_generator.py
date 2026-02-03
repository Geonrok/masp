"""
KOSDAQ 150 선물 일일 신호 생성기
================================

매일 장 마감 후 실행하여 다음날 거래 신호 확인
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .core_strategies import TradingSignal
from .equal_weight_portfolio import KOSDAQ150EqualWeightPortfolio


class KOSDAQ150SignalGenerator:
    """
    일일 거래 신호 생성기

    사용법:
        generator = KOSDAQ150SignalGenerator()
        signals = generator.generate_daily_signals()
        generator.print_signals()
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path("E:/투자/data/kosdaq_futures")
        self.output_path = self.data_path / "validated_strategies"

        # 포트폴리오 초기화
        self.portfolio = KOSDAQ150EqualWeightPortfolio()

        # 데이터
        self.data = None
        self.last_update = None

    def load_data(self) -> pd.DataFrame:
        """데이터 로드"""
        clean_path = self.data_path / "kosdaq150_futures_ohlcv_clean.parquet"
        cache_path = self.data_path / "kosdaq150_futures_ohlcv.parquet"

        if clean_path.exists():
            self.data = pd.read_parquet(clean_path)
        elif cache_path.exists():
            self.data = pd.read_parquet(cache_path)
        else:
            raise FileNotFoundError(f"데이터 파일 없음: {clean_path}")

        self.last_update = self.data.index[-1]
        return self.data

    def generate_daily_signals(self) -> Dict[str, Optional[TradingSignal]]:
        """
        오늘 기준 신호 생성

        Returns:
            전략별 신호 딕셔너리
        """
        if self.data is None:
            self.load_data()

        return self.portfolio.get_latest_signals(self.data)

    def get_signal_summary(self) -> Dict:
        """신호 요약 정보"""
        signals = self.generate_daily_signals()

        summary = {
            "date": str(self.last_update.date()) if self.last_update else None,
            "signals": {},
            "recommendation": None,
        }

        long_count = 0
        short_count = 0

        for name, signal in signals.items():
            if signal:
                summary["signals"][name] = {
                    "direction": "LONG" if signal.direction == 1 else "SHORT",
                    "strength": signal.strength,
                    "reason": signal.reason,
                }
                if signal.direction == 1:
                    long_count += 1
                elif signal.direction == -1:
                    short_count += 1
            else:
                summary["signals"][name] = None

        # 종합 권장
        if long_count >= 2:
            summary["recommendation"] = "STRONG LONG"
        elif long_count == 1:
            summary["recommendation"] = "WEAK LONG"
        elif short_count >= 2:
            summary["recommendation"] = "STRONG SHORT"
        elif short_count == 1:
            summary["recommendation"] = "WEAK SHORT"
        else:
            summary["recommendation"] = "HOLD"

        return summary

    def print_signals(self):
        """신호 출력"""
        summary = self.get_signal_summary()

        print("=" * 60)
        print("KOSDAQ 150 선물 일일 신호")
        print(f"기준일: {summary['date']}")
        print("=" * 60)

        print("\n전략별 신호:")
        for name, sig in summary["signals"].items():
            if sig:
                direction = sig["direction"]
                strength = sig["strength"]
                icon = "▲" if direction == "LONG" else "▼"
                print(f"  {name:<20}: {icon} {direction} (강도: {strength:.1%})")
                print(f"                        사유: {sig['reason']}")
            else:
                print(f"  {name:<20}: - 신호 없음")

        print(f"\n{'='*60}")
        print(f"종합 권장: {summary['recommendation']}")
        print("=" * 60)

        # 행동 가이드
        rec = summary["recommendation"]
        if rec == "STRONG LONG":
            print("\n[행동] 다음날 시초가에 롱 포지션 진입")
            print("       기존 숏 포지션이 있다면 청산 후 롱 진입")
        elif rec == "STRONG SHORT":
            print("\n[행동] 다음날 시초가에 숏 포지션 진입")
            print("       기존 롱 포지션이 있다면 청산 후 숏 진입")
        elif rec in ["WEAK LONG", "WEAK SHORT"]:
            print("\n[행동] 신호 강도 약함 - 기존 포지션 유지 권장")
        else:
            print("\n[행동] 신호 없음 - 기존 포지션 유지")

    def save_signals(self) -> Path:
        """신호를 JSON 파일로 저장"""
        summary = self.get_signal_summary()

        filename = f"daily_signal_{datetime.now().strftime('%Y%m%d')}.json"
        output_file = self.output_path / filename

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n신호 저장: {output_file}")
        return output_file

    def run_backtest(self) -> Dict:
        """백테스트 실행"""
        if self.data is None:
            self.load_data()

        result = self.portfolio.backtest(self.data)

        print("\n" + "=" * 60)
        print("백테스트 결과")
        print("=" * 60)
        print(f"Sharpe Ratio: {result['sharpe']:.3f}")
        cagr = result["cagr"] if not np.isnan(result["cagr"]) else 0
        total_ret = (
            result["total_return"] if not np.isnan(result["total_return"]) else 0
        )
        print(f"CAGR: {cagr*100:.1f}%")
        print(f"Total Return: {total_ret*100:.1f}%")
        print(f"MDD: {result['mdd']*100:.1f}%")
        print(f"Win Rate: {result['win_rate']*100:.1f}%")
        print(f"Total Trades: {result['total_trades']}")
        print("=" * 60)

        return result


def main():
    """메인 실행 함수"""
    generator = KOSDAQ150SignalGenerator()
    generator.load_data()
    generator.print_signals()
    generator.save_signals()


if __name__ == "__main__":
    main()
