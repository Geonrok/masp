# -*- coding: utf-8 -*-
"""
코스닥 현물 전략 - 일일 실행 스크립트
====================================

매일 장 마감 후 실행하여 다음날 거래 신호 확인

실행 방법:
    python -m strategies.kosdaq_spot.run_daily

또는:
    python strategies/kosdaq_spot/run_daily.py
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.kosdaq_spot import KOSDAQSpotSignalGenerator


def main():
    """일일 신호 생성 실행"""
    print("=" * 70)
    print("코스닥 현물 전략 - 일일 신호 생성기")
    print("전략: Multi_TF_Short (외국인 비중 5/20, 10/40 + 가격 MA50)")
    print("대상: 시총 상위 1/8 (유동성 상위 50% 내)")
    print("=" * 70)

    generator = KOSDAQSpotSignalGenerator()
    generator.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
