"""
Binance Futures Adapter Skeleton (Phase 1 준비)

Phase 0에서는 사용하지 않습니다.
Phase 1에서 실거래 구현 시 이 파일을 채워넣습니다.

설계 원칙:
- 실제 네트워크 호출 금지 (Phase 0)
- 메서드 시그니처와 예외 정의만
- 호출 시 명확한 안내 로그
"""

import logging
from typing import Dict, List, Optional
from libs.adapters.base import MarketDataAdapter, ExecutionAdapter

logger = logging.getLogger(__name__)


class BinanceFuturesMarketDataSkeleton(MarketDataAdapter):
    """
    Binance 선물 시세 어댑터 스켈레톤
    Phase 0: 구현 없음 (RuntimeError)
    Phase 1: 실제 Binance Futures API 연동 예정
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        logger.warning(
            "[BinanceFuturesMarketDataSkeleton] 초기화됨. "
            "Phase 0에서는 사용하지 않습니다. "
            "Phase 1에서 실거래 구현 예정."
        )

    def get_ticker(self, symbol: str) -> Dict:
        """
        현재가 조회
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            f"[BinanceFuturesMarketDataSkeleton] get_ticker({symbol}): "
            "Not implemented. Phase 1에서 실거래 구현 예정. "
            "Phase 0에서는 MockAdapter를 사용하세요."
        )

    def get_orderbook(self, symbol: str, depth: int = 10) -> Dict:
        """
        호가창 조회
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            f"[BinanceFuturesMarketDataSkeleton] get_orderbook({symbol}): "
            "Not implemented. Phase 1에서 실거래 구현 예정."
        )

    def get_funding_rate(self, symbol: str) -> Dict:
        """
        펀딩 비율 조회 (선물 전용)
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            f"[BinanceFuturesMarketDataSkeleton] get_funding_rate({symbol}): "
            "Not implemented. Phase 1에서 실거래 구현 예정."
        )


class BinanceFuturesExecutionSkeleton(ExecutionAdapter):
    """
    Binance 선물 주문 실행 어댑터 스켈레톤
    Phase 0: 구현 없음 (RuntimeError)
    Phase 1: 실제 Binance Futures API 연동 예정 (API 키 필수)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        logger.warning(
            "[BinanceFuturesExecutionSkeleton] 초기화됨. "
            "Phase 0에서는 사용하지 않습니다. "
            "Phase 1에서 실거래 구현 예정 (API 키 필요)."
        )

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        leverage: Optional[int] = None,
    ) -> Dict:
        """
        주문 실행 (레버리지 옵션 포함)
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            f"[BinanceFuturesExecutionSkeleton] place_order({symbol}, {side}): "
            "Not implemented. Phase 1에서 실거래 구현 예정. "
            "레버리지 거래 위험. Mock 모드를 사용하세요."
        )

    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        레버리지 설정 (선물 전용)
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            f"[BinanceFuturesExecutionSkeleton] set_leverage({symbol}, {leverage}): "
            "Not implemented. Phase 1에서 실거래 구현 예정."
        )

    def cancel_order(self, order_id: str) -> Dict:
        """
        주문 취소
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            f"[BinanceFuturesExecutionSkeleton] cancel_order({order_id}): "
            "Not implemented. Phase 1에서 실거래 구현 예정."
        )

    def get_position(self, symbol: str) -> Dict:
        """
        포지션 조회 (선물 전용)
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            f"[BinanceFuturesExecutionSkeleton] get_position({symbol}): "
            "Not implemented. Phase 1에서 실거래 구현 예정."
        )
