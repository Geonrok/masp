"""
Korea Stock Adapter Skeleton (Phase 1 준비)

Phase 0에서는 사용하지 않습니다.
Phase 1에서 실거래 구현 시 이 파일을 채워넣습니다.

설계 원칙:
- 실제 네트워크 호출 금지 (Phase 0)
- 메서드 시그니처와 예외 정의만
- 호출 시 명확한 안내 로그
- 현물/선물 분리 가능 (현재는 통합)
"""

import logging
from typing import Dict, List, Optional

from libs.adapters.base import ExecutionAdapter, MarketDataAdapter

logger = logging.getLogger(__name__)


class KRStockMarketDataSkeleton(MarketDataAdapter):
    """
    한국 주식 시세 어댑터 스켈레톤 (현물/선물 통합)
    Phase 0: 구현 없음 (RuntimeError)
    Phase 1: 실제 증권사 API 연동 예정
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        logger.warning(
            "[KRStockMarketDataSkeleton] 초기화됨. "
            "Phase 0에서는 사용하지 않습니다. "
            "Phase 1에서 실거래 구현 예정."
        )

    def get_ticker(self, symbol: str) -> Dict:
        """
        현재가 조회
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            f"[KRStockMarketDataSkeleton] get_ticker({symbol}): "
            "Not implemented. Phase 1에서 실거래 구현 예정. "
            "Phase 0에서는 MockAdapter를 사용하세요."
        )

    def get_orderbook(self, symbol: str, depth: int = 10) -> Dict:
        """
        호가창 조회
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            f"[KRStockMarketDataSkeleton] get_orderbook({symbol}): "
            "Not implemented. Phase 1에서 실거래 구현 예정."
        )

    def get_market_data(self, symbol: str) -> Dict:
        """
        시장 데이터 조회 (거래량, 시가총액 등)
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            f"[KRStockMarketDataSkeleton] get_market_data({symbol}): "
            "Not implemented. Phase 1에서 실거래 구현 예정."
        )


class KRStockExecutionSkeleton(ExecutionAdapter):
    """
    한국 주식 주문 실행 어댑터 스켈레톤 (현물/선물 통합)
    Phase 0: 구현 없음 (RuntimeError)
    Phase 1: 실제 증권사 API 연동 예정 (API 키/계좌정보 필수)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        logger.warning(
            "[KRStockExecutionSkeleton] 초기화됨. "
            "Phase 0에서는 사용하지 않습니다. "
            "Phase 1에서 실거래 구현 예정 (증권사 API 키/계좌정보 필요)."
        )

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: int,  # 한국 주식은 정수
        price: Optional[int] = None,  # 원화는 정수
    ) -> Dict:
        """
        주문 실행
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            f"[KRStockExecutionSkeleton] place_order({symbol}, {side}): "
            "Not implemented. Phase 1에서 실거래 구현 예정. "
            "실제 자금 집행 위험. Mock 모드를 사용하세요."
        )

    def cancel_order(self, order_id: str) -> Dict:
        """
        주문 취소
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            f"[KRStockExecutionSkeleton] cancel_order({order_id}): "
            "Not implemented. Phase 1에서 실거래 구현 예정."
        )

    def get_order_status(self, order_id: str) -> Dict:
        """
        주문 상태 조회
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            f"[KRStockExecutionSkeleton] get_order_status({order_id}): "
            "Not implemented. Phase 1에서 실거래 구현 예정."
        )

    def get_balance(self) -> Dict:
        """
        잔고 조회 (현금 + 보유 주식)
        Phase 1에서 구현 예정
        """
        raise RuntimeError(
            "[KRStockExecutionSkeleton] get_balance(): "
            "Not implemented. Phase 1에서 실거래 구현 예정."
        )
