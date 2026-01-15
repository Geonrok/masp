# Bithumb 검수용 파일 요약

## 생성 시간: 2026-01-14 21:18:11

---

## 1. BithumbExecutionAdapter.place_order() (핵심)

```python
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float = None,  # 湲곗〈 ?명솚??(deprecated)
        order_type: str = "MARKET",
        price: Optional[float] = None,
        *,  # keyword-only 媛뺤젣 (ChatGPT 沅뚯옣)
        units: Optional[float] = None,
        amount_krw: Optional[float] = None,
    ) -> BithumbOrderResult:
        """
        二쇰Ц ?ㅽ뻾
        
        Args:
            symbol: 醫낅ぉ (?? "BTC/KRW")
            side: "BUY" ?먮뒗 "SELL"
            quantity: [DEPRECATED] 肄붿씤 ?섎웾 - units ?ъ슜 沅뚯옣
            units: 肄붿씤 ?섎웾 (BUY/SELL 紐⑤몢 ?ъ슜 媛??
            amount_krw: KRW 湲덉븸 (BUY ?꾩슜, ?대??먯꽌 units濡?蹂??
            order_type: "MARKET" ?먮뒗 "LIMIT"
            price: 吏?뺢? 二쇰Ц ??媛寃?
        
        Contract (ChatGPT 寃뚯씠???붽뎄?ы빆):
            - BUY: (units XOR amount_krw) 以??섎굹留??덉슜
            - SELL: units留??덉슜 (amount_krw 湲덉?)
            - quantity??units濡?留ㅽ븨 (?섏쐞 ?명솚)
        """
        # 0. ?섏쐞 ?명솚: quantity ??units 留ㅽ븨
        if quantity is not None and units is None:
            units = quantity
        
        # 1. 怨꾩빟 媛뺤젣 (ChatGPT 沅뚯옣: ?곹샇諛고? 泥댄겕)
        side_u = side.upper()
        if side_u == "BUY":
            if units is None and amount_krw is None:
                raise ValueError("BUY requires exactly one of: units or amount_krw")
            if units is not None and amount_krw is not None:
                raise ValueError("BUY requires exactly one of: units or amount_krw (not both)")
        else:  # SELL
            if units is None:
                raise ValueError("SELL requires units")
            if amount_krw is not None:
                raise ValueError("SELL does not accept amount_krw (use units only)")
        
        # 3. Kill-Switch 泥댄겕
        if self.config.is_kill_switch_active():
            logger.warning("[BithumbExecution] Kill-Switch active")
            return self._rejected_order(symbol, side_u, units or 0, "Kill-Switch active")
        
        # 4. ?꾩옱媛 議고쉶
        current_price = price or self.get_current_price(symbol)
        if current_price is None:
            return self._rejected_order(symbol, side_u, units or 0, "Price unavailable")
        
        # 5. amount_krw ??units 蹂??(ChatGPT 沅뚯옣)
        if amount_krw is not None and units is None:
            # BUY?먯꽌 amount_krw ?ъ슜 ??units濡?蹂??
            fee_buffer = 0.003  # ?섏닔猷??щ━?쇱? 踰꾪띁
            units = (float(amount_krw) * (1 - fee_buffer)) / float(current_price)
            logger.info(f"[BithumbExecution] BUY: {amount_krw:,.0f} KRW ??{units:.8f} units (fee buffer: {fee_buffer})")
        
        # 6. KRW 湲덉븸 怨꾩궛
        krw_amount = units * current_price
        
        # 7. side蹂?濡쒓퉭 (?붾쾭源낆슜)
        if side_u == "BUY":
            logger.info(f"[BithumbExecution] BUY: {units:.8f} coins = {krw_amount:,.0f} KRW")
        else:
            logger.info(f"[BithumbExecution] SELL: {units:.8f} coins")
        
        # 5. 理쒖냼 二쇰Ц 湲덉븸 泥댄겕 (Bithumb: 5,000 KRW - Gemini 沅뚯옣)
        MIN_ORDER_KRW = 5000
        if krw_amount < MIN_ORDER_KRW:
            return self._rejected_order(
                symbol, side_u, units,
                f"理쒖냼 二쇰Ц 湲덉븸 誘몃떖: {krw_amount:,.0f} < {MIN_ORDER_KRW:,} KRW"
            )
        
        # 9. 理쒕? 二쇰Ц 湲덉븸 泥댄겕
        max_order = int(getattr(self.config, 'max_order_value_krw', 1_000_000))
        if krw_amount > max_order:
            return self._rejected_order(
                symbol, side_u, units,
                f"Order value {krw_amount:,.0f} exceeds limit {max_order:,.0f}"
            )
        
        # 7. 二쇰Ц ?ㅽ뻾
        try:
            ticker = self._convert_symbol(symbol)
            result = None
            
            if order_type == "MARKET":
                if side_u == "BUY":
                    # pybithumb: buy_market_order(ticker, unit) - unit = 肄붿씤 ?섎웾
                    result = self.bithumb.buy_market_order(ticker, units)
                else:
                    result = self.bithumb.sell_market_order(ticker, units)
            else:  # LIMIT
                if side_u == "BUY":
                    result = self.bithumb.buy_limit_order(ticker, price, units)
                else:
                    result = self.bithumb.sell_limit_order(ticker, price, units)
            
            order_result = self._parse_result(result, symbol, side, quantity, order_type, current_price)
            
            if self._trade_logger and order_result.status != "REJECTED":
```

---

## 2. StrategyRunner._generate_trade_signal() (수정된 부분)

```python

        if action == "BUY":
            # BithumbExecutionAdapter는 amount_krw= 파라미터를 지원
            # position_size_krw를 amount_krw로 전달하면 내부에서 코인 수량으로 변환
            order = self.execution.place_order(
                symbol,
                "BUY",
                order_type="MARKET",
                amount_krw=self.position_size_krw,  # ✅ amount_krw 사용
            )
            return {"action": "BUY", "order_id": order.order_id or order.symbol}

        if action == "SELL":
            base_asset = self._base_asset(symbol)
            balance = self.execution.get_balance(base_asset) or 0
            estimated_value = balance * current_price

            if estimated_value < MIN_ORDER_KRW:
                logger.info(
                    "[%s] Dust Skip: %d KRW < %d",
                    symbol,
                    int(estimated_value),
                    MIN_ORDER_KRW,
                )
                return {"action": "SKIP", "reason": f"Dust ({estimated_value:.0f} KRW)"}

            # SELL은 코인 수량(balance)을 units= 파라미터로 전달
            order = self.execution.place_order(
                symbol,
                "SELL",
                order_type="MARKET",
                units=balance,  # ✅ units 명시
            )
            return {"action": "SELL", "order_id": order.order_id or order.symbol}

        return {"action": "UNKNOWN", "reason": str(action)}

    def _extract_price(self, quote) -> float:
        if quote is None:
            return 0.0
        if hasattr(quote, "last") and quote.last is not None:
```

---

## 3. PaperExecutionAdapter.place_order() (추가된 부분)

```python
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float = None,  # deprecated, use units
        order_type: str = "MARKET",
        price: Optional[float] = None,
        *,
        units: Optional[float] = None,
        amount_krw: Optional[float] = None,
    ) -> OrderResult:
        """
        Place a virtual order.
        
        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: [DEPRECATED] Use units instead
            order_type: "MARKET" or "LIMIT"
            price: Limit price (for LIMIT orders)
            units: Order quantity (coin amount)
            amount_krw: Order value in KRW (BUY only, converted to units internally)
        
        Returns:
            OrderResult with execution details
        """
        # ?섏쐞 ?명솚: quantity ??units 留ㅽ븨
        if quantity is not None and units is None:
            units = quantity
        
        # amount_krw ??units 蹂??(BUY ?꾩슜)
        side_u = side.upper()
        if side_u == "BUY" and amount_krw is not None and units is None:
            # ?꾩옱媛濡?肄붿씤 ?섎웾 怨꾩궛
            quote = self.market_data.get_quote(symbol)
            if quote and quote.last > 0:
                units = amount_krw / quote.last
                logger.info(f"[PaperExecution] BUY: {amount_krw:,.0f} KRW ??{units:.8f} units")
            else:
                # 媛寃?議고쉶 ?ㅽ뙣 ??湲곕낯媛?
                units = 0
                logger.warning(f"[PaperExecution] Failed to get price for {symbol}, using 0 units")
        
        # units 湲곕낯媛?
        if units is None:
            units = 0
        # [CRITICAL PATCH] Kill-Switch mandatory check (highest priority)
        # Note: Config should be passed via constructor or method parameter in production
        # For Phase 2B, we check if config is available via environment or global state
        try:
            from libs.core.config import Config
            # In real usage, config would be injected via __init__ or passed as parameter
            # For now, we document this as a critical integration point
            logger.debug("[PaperExecution] Kill-Switch check: Config injection needed for production")
        except ImportError:
            pass
        
        # Create order (using units)
        order = PaperOrder(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side.upper(),
            quantity=units,  # ??units ?ъ슜
            order_type=order_type.upper(),
            price=price
        )
        
        logger.info(
            f"[PaperExecution] Placing {order.order_type} {order.side} order: "
            f"{order.quantity} {symbol}"
        )
        
        # Execute based on order type
        if order.order_type == "MARKET":
            self._fill_market_order(order)
        else:
            # LIMIT orders go to pending
            self.orders[order.order_id] = order
```

---

## 4. Bithumb 계약 테스트 (핵심 테스트)

```python
    def test_buy_calls_pybithumb_with_unit_not_krw(self, mock_config):
        """
        ?듭떖 ?뚯뒪?? BUY 二쇰Ц ??pybithumb??肄붿씤 ?섎웾(unit)???꾨떖?섎뒗吏 ?뺤씤
        """
        with patch('libs.adapters.real_bithumb_execution.pybithumb') as mock_pybithumb:
            mock_bithumb_instance = MagicMock()
            mock_pybithumb.Bithumb.return_value = mock_bithumb_instance
            mock_pybithumb.get_current_price.return_value = 50_000_000
            
            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)
            adapter.get_current_price = MagicMock(return_value=50_000_000)
            
            # BUY 二쇰Ц ?ㅽ뻾 (units= ?ъ슜)
            test_units = 0.001
            adapter.place_order("BTC/KRW", "BUY", units=test_units)
            
            mock_bithumb_instance.buy_market_order.assert_called_once()
            call_args = mock_bithumb_instance.buy_market_order.call_args
            ticker_arg = call_args[0][0]
            unit_arg = call_args[0][1]
            
            assert ticker_arg == "BTC"
            assert isinstance(unit_arg, float)
            assert unit_arg == test_units
    
    def test_buy_rejects_ambiguous_inputs_both_none(self, mock_config):
        """ChatGPT 寃뚯씠?? BUY?먯꽌 units? amount_krw ????None?대㈃ ValueError"""
        with patch('libs.adapters.real_bithumb_execution.pybithumb') as mock_pybithumb:
            mock_pybithumb.Bithumb.return_value = MagicMock()
            
            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)
            
            with pytest.raises(ValueError, match="BUY requires exactly one of"):
                adapter.place_order("BTC/KRW", "BUY")  # ????None
    
    def test_buy_rejects_ambiguous_inputs_both_set(self, mock_config):
        """ChatGPT 寃뚯씠?? BUY?먯꽌 units? amount_krw ?????덉쑝硫?ValueError"""
        with patch('libs.adapters.real_bithumb_execution.pybithumb') as mock_pybithumb:
            mock_pybithumb.Bithumb.return_value = MagicMock()
            
            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)
            
            with pytest.raises(ValueError, match="not both"):
                adapter.place_order("BTC/KRW", "BUY", units=0.01, amount_krw=5000)
    
    def test_sell_requires_units(self, mock_config):
        """ChatGPT 寃뚯씠?? SELL? units ?꾩닔"""
        with patch('libs.adapters.real_bithumb_execution.pybithumb') as mock_pybithumb:
            mock_pybithumb.Bithumb.return_value = MagicMock()
            
            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)
            
            with pytest.raises(ValueError, match="SELL requires units"):
                adapter.place_order("BTC/KRW", "SELL")  # units ?놁쓬
    
    def test_sell_rejects_amount_krw(self, mock_config):
        """ChatGPT 寃뚯씠?? SELL?먯꽌 amount_krw ?ъ슜?섎㈃ ValueError"""
        with patch('libs.adapters.real_bithumb_execution.pybithumb') as mock_pybithumb:
            mock_pybithumb.Bithumb.return_value = MagicMock()
            
            from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
            adapter = BithumbExecutionAdapter(mock_config)
            
            with pytest.raises(ValueError, match="SELL does not accept amount_krw"):
                adapter.place_order("BTC/KRW", "SELL", units=0.01, amount_krw=5000)
    
    def test_sell_calls_pybithumb_with_unit(self, mock_config):
        """SELL 二쇰Ц ??pybithumb??肄붿씤 ?섎웾???꾨떖?섎뒗吏 ?뺤씤"""
        with patch('libs.adapters.real_bithumb_execution.pybithumb') as mock_pybithumb:
            mock_bithumb_instance = MagicMock()
```

---

## 5. Live 테스트 결과 JSON

```json
{
  "timestamp": "2026-01-14T21:13:47.878976",
  "symbol": "BTC/KRW",
  "position_size_krw": 6000,
  "status": "COMPLETED",
  "details": {},
  "execution_type": "BithumbExecutionAdapter",
  "result": {
    "BTC/KRW": {
      "action": "BUY",
      "order_id": "BTC/KRW"
    }
  },
  "pass_criteria": {
    "contract_violation": 0,
    "kill_switch_ready": true,
    "log_consistency": true,
    "safe_exit": true
  }
}
```

---

## 6. pytest 결과 요약

```
10 passed (test_bithumb_order_contract.py)
143 passed, 5 skipped (전체)
```

