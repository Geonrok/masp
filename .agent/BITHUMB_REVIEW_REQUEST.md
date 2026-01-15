# MASP 빗썸 통합 검수 요청

## 📋 검수 대상

**프로젝트**: MASP (Multi-Asset Strategy Platform)  
**작업 내용**: 빗썸(Bithumb) 현물 거래 자동매매 통합  
**완료일**: 2026-01-14  
**작업자**: Antigravity AI  

---

## 🎯 검수 요청 사항

빗썸 현물 거래 기능이 MASP 플랫폼에 올바르게 통합되었는지 검수해 주세요.

### 검수 항목

1. **코드 품질 및 일관성**
   - Upbit 어댑터와의 코드 패턴 일관성
   - 에러 처리 및 로깅 적절성
   - Python 타입 힌트 정확성

2. **안전장치 검증**
   - `MASP_ENABLE_LIVE_TRADING` 환경변수 체크 로직
   - Paper/Live 모드 자동 전환 로직
   - Kill-Switch 연동 여부

3. **잠재적 버그 및 런타임 오류**
   - 누락된 의존성 또는 import
   - None 체크 누락
   - 예외 처리 부족

4. **설계 개선 제안**
   - 코드 중복 제거 가능성
   - 향후 확장성 고려
   - 성능 최적화 가능성

---

## 📁 핵심 파일 목록

### 1. StrategyRunner (수정됨)
**파일**: `services/strategy_runner.py`

```python
"""
Strategy Runner - 전략 신호를 실거래 주문으로 변환
- 전략 신호 수신
- Kill-Switch 체크
- Health Monitor 검증
- 주문 실행
- TradeLogger 기록
"""

import logging
import os
import time
from datetime import datetime, date
from typing import Optional, Dict, List
from pathlib import Path

from libs.core.config import Config
from libs.adapters.factory import AdapterFactory
from libs.adapters.trade_logger import TradeLogger
from libs.analytics.strategy_health import StrategyHealthMonitor
from libs.analytics.daily_report import DailyReportGenerator
from libs.strategies.base import Signal as StrategySignal
from libs.strategies.loader import get_strategy

logger = logging.getLogger(__name__)
MIN_ORDER_KRW = 5000


class StrategyRunner:
    """
    전략 실행기
    
    Usage:
        runner = StrategyRunner(
            strategy_name="ma_crossover",
            exchange="paper",  # "paper" | "upbit" | "bithumb"
            symbols=["BTC/KRW"],
            position_size_krw=10000
        )
        runner.run_once()  # 1회 실행
        runner.run_loop(interval_seconds=60)  # 반복 실행
    """
    
    def __init__(
        self,
        strategy_name: str,
        exchange: str = "paper",
        symbols: List[str] = None,
        position_size_krw: float = 10000,
        config: Config = None,
        strategy=None,
        market_data=None,
        execution=None
    ):
        """
        초기화
        
        Args:
            strategy_name: 전략 이름
            exchange: "paper" | "upbit" | "bithumb"
            symbols: 거래 종목 리스트
            position_size_krw: 1회 주문 금액 (KRW)
            config: Config 객체
        """
        self.strategy_name = strategy_name
        self.exchange = exchange
        self.symbols = symbols or ["BTC/KRW"]
        self.position_size_krw = position_size_krw
        
        # Config 로드
        self.config = config or Config(
            asset_class="crypto_spot",
            strategy_name=strategy_name
        )

        self.strategy = strategy or get_strategy(strategy_name)
        
        # 로그 디렉토리
        log_base = Path(f"logs/{exchange}_trades")
        
        # 컴포넌트 초기화
        self.trade_logger = TradeLogger(log_dir=str(log_base / "trades"))
        self.health_monitor = StrategyHealthMonitor(self.config)
        self.daily_reporter = DailyReportGenerator(
            self.trade_logger,
            self.health_monitor,
            report_dir=str(log_base / "reports")
        )
        
        # 실행 어댑터 - 핵심 수정 부분
        execution_exchange = exchange
        adapter_mode = "paper"
        live_trading_enabled = os.getenv("MASP_ENABLE_LIVE_TRADING") == "1"

        if exchange in {"upbit", "upbit_spot"}:
            execution_exchange = "upbit_spot"
            adapter_mode = "live" if live_trading_enabled else "paper"
        elif exchange in {"bithumb", "bithumb_spot"}:
            execution_exchange = "bithumb"
            adapter_mode = "live" if live_trading_enabled else "paper"
        
        self.execution = execution or AdapterFactory.create_execution(
            execution_exchange,
            adapter_mode=adapter_mode,
            config=self.config if exchange != "paper" else None,
            trade_logger=self.trade_logger,
        )
        
        # 시세 어댑터 - 핵심 수정 부분
        if exchange in ["paper", "upbit", "upbit_spot"]:
            md_exchange = "upbit_spot"
        elif exchange in ["bithumb", "bithumb_spot"]:
            md_exchange = "bithumb_spot"
        else:
            md_exchange = "upbit_spot"
        self.market_data = market_data or AdapterFactory.create_market_data(md_exchange)
        
        # 포지션 상태
        self._positions: Dict[str, float] = {}  # symbol -> quantity
        self._last_signals: Dict[str, str] = {}  # symbol -> signal
        
        logger.info(f"[StrategyRunner] Initialized: {strategy_name} on {exchange}")
```

---

### 2. AdapterFactory (수정됨)
**파일**: `libs/adapters/factory.py`

```python
"""
Adapter Factory for creating market data and execution adapters

Provides a centralized way to create adapters for different exchanges.
Phase 2C - Extended with Upbit/Bithumb execution, Paper trading
"""

import logging
import os
import warnings
from typing import Literal, Optional
from libs.adapters.base import MarketDataAdapter, ExecutionAdapter

logger = logging.getLogger(__name__)


AdapterType = Literal["upbit_spot", "bithumb_spot", "binance_futures", "mock", "paper"]


class AdapterFactory:
    """
    Factory for creating exchange adapters.
    
    Market Data:
        - upbit_spot: Upbit 현물 시세
        - bithumb_spot: Bithumb 현물 시세
        - binance_futures: Binance 선물 시세
        - mock: Mock 시세
        
    Execution:
        - paper: 모의 거래 (Paper Trading)
        - upbit_spot: Upbit 실거래
        - upbit: Upbit 실거래 (deprecated alias)
        - bithumb: Bithumb 실거래
        - binance_futures: Binance 선물 실거래
        - mock: Mock 실행
    """
    
    @staticmethod
    def create_execution(
        exchange_name: str,
        adapter_mode: str = "paper",
        config=None,
        trade_logger=None,
        **kwargs
    ) -> ExecutionAdapter:
        """
        Create an Execution adapter.
        """
        logger.info(f"[FACTORY] Creating Execution adapter: {exchange_name}")

        if config is None:
            config = kwargs.pop("config", None)
        else:
            kwargs.pop("config", None)
        if trade_logger is None:
            trade_logger = kwargs.pop("trade_logger", None)
        else:
            kwargs.pop("trade_logger", None)

        # Upbit 처리
        if exchange_name == "upbit_spot":
            if adapter_mode in {"live", "execution"}:
                if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
                    raise RuntimeError(
                        "[Factory] Live trading disabled. "
                        "Set MASP_ENABLE_LIVE_TRADING=1 or use adapter_mode='paper'"
                    )
                from libs.adapters.real_upbit_spot import UpbitSpotExecution
                return UpbitSpotExecution(
                    access_key=kwargs.get("access_key"),
                    secret_key=kwargs.get("secret_key"),
                )

            from libs.adapters.paper_execution import PaperExecutionAdapter
            market_data = AdapterFactory.create_market_data("upbit_spot")
            return PaperExecutionAdapter(
                market_data_adapter=market_data,
                initial_balance=kwargs.pop("initial_balance", 1_000_000),
                config=config,
                trade_logger=trade_logger,
                **kwargs,
            )

        # 빗썸 처리 - 핵심 추가 부분
        if exchange_name in {"bithumb", "bithumb_spot"}:
            if adapter_mode in {"live", "execution"}:
                if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
                    raise RuntimeError(
                        "[Factory] Bithumb live trading disabled. "
                        "Set MASP_ENABLE_LIVE_TRADING=1 or use adapter_mode='paper'"
                    )
                from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
                from libs.core.config import Config as ConfigClass
                if config is None:
                    config = ConfigClass()
                adapter = BithumbExecutionAdapter(config, **kwargs)
                if trade_logger:
                    adapter.set_trade_logger(trade_logger)
                return adapter
            
            # Paper mode for bithumb
            from libs.adapters.paper_execution import PaperExecutionAdapter
            market_data = AdapterFactory.create_market_data("bithumb_spot")
            return PaperExecutionAdapter(
                market_data_adapter=market_data,
                initial_balance=kwargs.pop("initial_balance", 1_000_000),
                config=config,
                trade_logger=trade_logger,
                **kwargs,
            )

        # ... (나머지 거래소)
```

---

### 3. BithumbExecutionAdapter (기존)
**파일**: `libs/adapters/real_bithumb_execution.py`

```python
"""
Bithumb 실주문 어댑터
- 잔고 조회
- 시장가/지정가 주문
- Kill-Switch 연동
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass

try:
    import pybithumb
except ImportError:
    pybithumb = None

from libs.core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class BithumbOrderResult:
    """Bithumb 주문 결과"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    filled_quantity: float
    filled_price: float
    fee: float
    created_at: datetime
    message: str = ""


class BithumbExecutionAdapter:
    """
    Bithumb 실주문 어댑터
    
    Features:
    - Kill-Switch 연동
    - TradeLogger 자동 기록
    - 잔고/주문 조회
    """
    
    FEE_RATE = 0.0025  # 0.25% (Bithumb 기본)
    
    def __init__(self, config: Config):
        """초기화"""
        if pybithumb is None:
            raise ImportError("pybithumb not installed. Run: pip install pybithumb")
        
        self.config = config
        self._validate_config()
        
        # pybithumb 인스턴스 생성
        api_key = config.bithumb_api_key.get_secret_value()
        secret_key = config.bithumb_secret_key.get_secret_value()
        self.bithumb = pybithumb.Bithumb(api_key, secret_key)
        
        self._trade_logger = None
        logger.info("[BithumbExecution] Adapter initialized")
    
    def _validate_config(self):
        """설정 검증"""
        api_key = self.config.bithumb_api_key.get_secret_value()
        secret_key = self.config.bithumb_secret_key.get_secret_value()
        
        if not api_key or api_key == "your_api_key_here":
            raise ValueError("BITHUMB_API_KEY not set or invalid")
        if not secret_key or secret_key == "your_secret_key_here":
            raise ValueError("BITHUMB_SECRET_KEY not set or invalid")
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None
    ) -> BithumbOrderResult:
        """주문 실행"""
        # 1. Kill-Switch 체크
        if self.config.is_kill_switch_active():
            logger.warning("[BithumbExecution] Kill-Switch active")
            return self._rejected_order(symbol, side, quantity, "Kill-Switch active")
        
        # 2. 현재가 조회
        current_price = price or self.get_current_price(symbol)
        if current_price is None:
            return self._rejected_order(symbol, side, quantity, "Price unavailable")
        
        # 3. 주문 금액 검증
        order_value = quantity * current_price
        max_order = int(getattr(self.config, 'max_order_value_krw', 1_000_000))
        
        if order_value > max_order:
            return self._rejected_order(
                symbol, side, quantity,
                f"Order value {order_value:,.0f} exceeds limit {max_order:,.0f}"
            )
        
        # 4. 주문 실행
        try:
            ticker = self._convert_symbol(symbol)
            result = None
            
            if order_type == "MARKET":
                if side.upper() == "BUY":
                    result = self.bithumb.buy_market_order(ticker, quantity)
                else:
                    result = self.bithumb.sell_market_order(ticker, quantity)
            else:  # LIMIT
                if side.upper() == "BUY":
                    result = self.bithumb.buy_limit_order(ticker, price, quantity)
                else:
                    result = self.bithumb.sell_limit_order(ticker, price, quantity)
            
            order_result = self._parse_result(result, symbol, side, quantity, order_type, current_price)
            
            if self._trade_logger and order_result.status != "REJECTED":
                self._log_trade(order_result)
            
            return order_result
            
        except Exception as e:
            logger.error(f"[BithumbExecution] Order failed: {e}")
            return self._rejected_order(symbol, side, quantity, str(e))
```

---

### 4. BithumbSpotMarketData (기존)
**파일**: `libs/adapters/real_bithumb_spot.py`

```python
"""
Bithumb 시세 조회 어댑터
- 현재가 조회
- 호가창 조회
- OHLCV 조회
"""

import logging
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

try:
    import pybithumb
except ImportError:
    pybithumb = None

logger = logging.getLogger(__name__)


@dataclass
class BithumbQuote:
    """시세 정보"""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime


@dataclass
class BithumbOHLCV:
    """캔들 데이터"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BithumbSpotMarketData:
    """
    Bithumb 시세 조회 어댑터 (Public API - 키 불필요)
    
    Methods:
        get_quote(symbol): 현재가 조회
        get_orderbook(symbol, depth): 호가창 조회
        get_ohlcv(symbol, interval, limit): OHLCV 조회
        get_tickers(): 전체 종목 조회
    """
    
    def __init__(self):
        """초기화"""
        if pybithumb is None:
            raise ImportError("pybithumb not installed. Run: pip install pybithumb")
        logger.info("[BithumbMarketData] Adapter initialized")
    
    def get_quote(self, symbol: str) -> Optional[BithumbQuote]:
        """현재가 조회"""
        try:
            ticker = self._convert_symbol(symbol)
            price = pybithumb.get_current_price(ticker)
            
            if price is None:
                return None
            
            # 호가 조회
            orderbook = pybithumb.get_orderbook(ticker)
            bid = float(orderbook['bids'][0]['price']) if orderbook and orderbook.get('bids') else price * 0.999
            ask = float(orderbook['asks'][0]['price']) if orderbook and orderbook.get('asks') else price * 1.001
            
            return BithumbQuote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=float(price),
                volume=0,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"[BithumbMarketData] Quote failed: {e}")
            return None
    
    def get_ohlcv(self, symbol: str, interval: str = "1d", limit: int = 100) -> List[BithumbOHLCV]:
        """OHLCV 조회"""
        try:
            ticker = self._convert_symbol(symbol)
            
            # interval 변환
            interval_map = {
                "1m": "1m", "5m": "5m", "10m": "10m", "30m": "30m",
                "1h": "1h", "6h": "6h", "12h": "12h", "1d": "24h"
            }
            bithumb_interval = interval_map.get(interval, "24h")
            
            df = pybithumb.get_ohlcv(ticker, interval=bithumb_interval)
            
            if df is None or df.empty:
                return []
            
            result = []
            for idx, row in df.tail(limit).iterrows():
                result.append(BithumbOHLCV(
                    timestamp=idx,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume'])
                ))
            
            return result
        except Exception as e:
            logger.error(f"[BithumbMarketData] OHLCV failed: {e}")
            return []
    
    def _convert_symbol(self, symbol: str) -> str:
        """심볼 변환: BTC/KRW -> BTC"""
        return symbol.split("/")[0]
```

---

## 📊 테스트 결과

### Paper Trading 테스트
```
=== Bithumb Paper Trading Test ===
[Loader] Dynamically loaded: kama_tsmom_gate
[1] StrategyRunner created
    Exchange: bithumb
    Strategy: KamaTsmomGateStrategy
    MarketData: BithumbSpotMarketData
    Execution: PaperExecutionAdapter
[2] Result: {'BTC/KRW': {'action': 'BUY', 'order_id': '79bf2a78'}}
[3] Paper Trading Test: PASS
```

### pytest 회귀 테스트
```
133 passed, 5 skipped in 10.60s
```

---

## ❓ 검수에서 확인해야 할 질문

1. **안전장치 충분성**
   - `MASP_ENABLE_LIVE_TRADING=0`일 때 Paper 모드로 자동 전환하는 로직이 적절한가?
   - Upbit과 Bithumb에 대해 동일한 안전 패턴이 적용되었는가?

2. **코드 일관성**
   - `UpbitSpotExecution`과 `BithumbExecutionAdapter` 간의 인터페이스가 호환되는가?
   - 반환 타입(`OrderResult` vs `BithumbOrderResult`)의 차이가 문제가 될 수 있는가?

3. **에러 처리**
   - `pybithumb` 라이브러리 예외 처리가 충분한가?
   - API 키 없이 실행 시 적절한 오류 메시지가 표시되는가?

4. **잠재적 버그**
   - `quantity` 파라미터가 BUY 주문에서 KRW 금액인지 코인 수량인지 혼동 가능성은?
   - Rate Limit 처리가 필요한가?

5. **향후 확장성**
   - `bithumb_spot`과 `bithumb` 네이밍 통일이 필요한가?
   - Circuit Breaker 패턴 추가가 필요한가?

---

## 📝 각 AI에게 요청 사항

1. **위 코드를 검토**하고 잠재적 버그나 문제점을 지적해 주세요.
2. **안전장치 로직**이 충분한지 평가해 주세요.
3. **설계 개선 제안**이 있다면 제시해 주세요.
4. **각 항목에 대해 PASS/FAIL/WARNING** 판정을 내려 주세요.

```
[검수 결과 템플릿]

## 검수 결과 요약

| 항목 | 판정 | 비고 |
|------|------|------|
| 코드 품질 | PASS/FAIL/WARNING | |
| 안전장치 | PASS/FAIL/WARNING | |
| 에러 처리 | PASS/FAIL/WARNING | |
| 일관성 | PASS/FAIL/WARNING | |
| 확장성 | PASS/FAIL/WARNING | |

## 상세 피드백

### 발견된 문제점
- 

### 개선 제안
- 

### 강한 반박 가능성
- 
```
