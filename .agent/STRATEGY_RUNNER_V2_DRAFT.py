"""
StrategyRunner 주문 실행 부분 수정 초안

핵심 변경:
- quantity (코인 수량) → amount_krw (KRW 금액) 기반으로 통일
- 모든 ExecutionAdapter가 동일한 인터페이스 사용
"""

# ========== 기존 코드 (문제점) ==========


def _execute_trade_signal_OLD(self, symbol: str, signal, quote) -> Dict:
    """문제: quantity가 코인 수량인지 KRW인지 불명확"""
    action, _gate_pass = self._parse_signal(signal, True)
    current_price = self._extract_price(quote)

    if action == "BUY":
        order = self.execution.place_order(
            symbol,
            "BUY",
            self.position_size_krw,  # ❌ 이게 KRW인데, Adapter는 quantity로 받음
            order_type="MARKET",
        )
        # ...


# ========== 수정된 코드 ==========


def _execute_trade_signal_NEW(self, symbol: str, signal, quote) -> Dict:
    """
    수정: amount_krw 기반으로 명확화

    - BUY: position_size_krw (KRW 금액) 전달
    - SELL: 보유 코인의 현재가 * 수량 = KRW 금액으로 변환
    """
    action, _gate_pass = self._parse_signal(signal, True)
    current_price = self._extract_price(quote)

    if action == "HOLD":
        return {"action": "HOLD", "reason": getattr(signal, "reason", "N/A")}

    if action == "BUY":
        # amount_krw: 매수할 금액 (KRW)
        amount_krw = self.position_size_krw

        order = self.execution.place_order(
            symbol=symbol,
            side="BUY",
            amount_krw=amount_krw,  # ✅ 명확: KRW 금액
            order_type="MARKET",
        )
        return {
            "action": "BUY",
            "order_id": order.order_id,
            "amount_krw": amount_krw,
            "status": order.status,
        }

    if action == "SELL":
        # 보유 코인 수량 조회
        base_asset = self._base_asset(symbol)
        balance = self.execution.get_balance(base_asset) or 0

        # KRW 금액으로 변환
        amount_krw = balance * current_price

        if amount_krw < MIN_ORDER_KRW:
            logger.info(
                f"[{symbol}] Dust Skip: {amount_krw:,.0f} KRW < {MIN_ORDER_KRW}"
            )
            return {"action": "SKIP", "reason": f"Dust ({amount_krw:.0f} KRW)"}

        order = self.execution.place_order(
            symbol=symbol,
            side="SELL",
            amount_krw=amount_krw,  # ✅ 명확: KRW 금액
            order_type="MARKET",
        )
        return {
            "action": "SELL",
            "order_id": order.order_id,
            "amount_krw": amount_krw,
            "status": order.status,
        }

    return {"action": "UNKNOWN", "reason": str(action)}


# ========== 호환성 레이어 (마이그레이션용) ==========


class ExecutionAdapterWrapper:
    """
    기존 quantity 기반 어댑터를 amount_krw 기반으로 변환하는 래퍼

    Usage:
        old_adapter = UpbitSpotExecution()
        new_adapter = ExecutionAdapterWrapper(old_adapter, price_fetcher)
    """

    def __init__(self, adapter, price_fetcher):
        self._adapter = adapter
        self._price_fetcher = price_fetcher

    def place_order(
        self,
        symbol: str,
        side: str,
        amount_krw: float,
        order_type: str = "MARKET",
        price: float = None,
    ):
        """amount_krw → quantity 변환 후 기존 어댑터 호출"""
        current_price = price or self._price_fetcher(symbol)

        if side.upper() == "BUY":
            # Upbit의 경우 BUY도 KRW 기반일 수 있음 - 거래소별 확인 필요
            quantity = amount_krw / current_price
        else:
            quantity = amount_krw / current_price

        return self._adapter.place_order(symbol, side, quantity, order_type, price)

    def __getattr__(self, name):
        """기타 메서드는 원본 어댑터로 위임"""
        return getattr(self._adapter, name)
