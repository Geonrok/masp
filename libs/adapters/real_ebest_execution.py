"""
eBest (LS증권) Spot Execution Adapter

Provides order execution for KOSPI/KOSDAQ stocks via LS증권 Open API.
Uses the `ebest` Python package (v1.0.2+) for REST API access.

API Documentation: https://openapi.ls-sec.co.kr
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional, Dict, Any, List

from libs.adapters.base import ExecutionAdapter, OrderResult

logger = logging.getLogger(__name__)


class EbestSpotExecution(ExecutionAdapter):
    """
    eBest (LS증권) Spot Execution Adapter

    Provides:
        - place_order(symbol, side, quantity, order_type, price): 주문 (CSPAT00600)
        - cancel_order(order_id): 취소 (CSPAT00800)
        - get_order_status(order_id): 주문 상태 조회
        - get_balance(asset): 잔고 조회 (t0424)

    Authentication:
        Uses EBEST_APP_KEY, EBEST_APP_SECRET, and EBEST_ACCOUNT_NO environment variables.

    Safety:
        - Requires MASP_ENABLE_LIVE_TRADING=1 for real orders
        - Implements trade logging via TradeLogger
    """

    def __init__(
        self,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        account_no: Optional[str] = None,
        account_pwd: Optional[str] = None,
    ):
        """
        Initialize eBest execution adapter.

        Args:
            app_key: LS Securities Open API app key (or EBEST_APP_KEY env)
            app_secret: LS Securities Open API app secret (or EBEST_APP_SECRET env)
            account_no: Trading account number (or EBEST_ACCOUNT_NO env)
            account_pwd: Account transaction password (or EBEST_ACCOUNT_PWD env)
        """
        self._app_key = app_key or os.getenv("EBEST_APP_KEY", "")
        self._app_secret = app_secret or os.getenv("EBEST_APP_SECRET", "")
        self._account_no = account_no or os.getenv("EBEST_ACCOUNT_NO", "")
        self._account_pwd = account_pwd or os.getenv("EBEST_ACCOUNT_PWD", "")
        self._api: Optional[Any] = None
        self._logged_in = False
        self._trade_logger = None

        if not self._app_key or not self._app_secret:
            logger.warning(
                "[eBest] API credentials not provided. "
                "Set EBEST_APP_KEY and EBEST_APP_SECRET environment variables."
            )

        if not self._account_no:
            logger.warning(
                "[eBest] Account number not provided. "
                "Set EBEST_ACCOUNT_NO environment variable for order execution."
            )

        logger.info("[eBest] Execution adapter initialized")

    def set_trade_logger(self, trade_logger) -> None:
        """Set trade logger for recording trades."""
        self._trade_logger = trade_logger

    def _run_async(self, coro):
        """Run async coroutine synchronously.

        Handles the case where we may or may not already be in an async context.
        Reuses API connection when possible for efficiency.
        """
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - use thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Reset API state for new thread/loop
                self._api = None
                self._logged_in = False
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=60)
        except RuntimeError:
            # No running loop - safe to use asyncio.run directly
            # Keep existing API state for connection reuse
            return asyncio.run(coro)

    async def _ensure_login(self) -> bool:
        """Ensure API is logged in."""
        if self._logged_in and self._api:
            return True

        if not self._app_key or not self._app_secret:
            logger.error("[eBest] Cannot login: missing credentials")
            return False

        try:
            from ebest import OpenApi
            self._api = OpenApi()
            success = await self._api.login(self._app_key, self._app_secret)
            if success:
                self._logged_in = True
                logger.info("[eBest] Login successful")
                return True
            else:
                logger.error("[eBest] Login failed")
                return False
        except ImportError:
            logger.error("[eBest] 'ebest' package not installed. Run: pip install ebest")
            return False
        except Exception as e:
            logger.error(f"[eBest] Login error: {e}")
            return False

    def _ensure_live_trading(self) -> None:
        """Check if live trading is enabled."""
        if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
            raise RuntimeError(
                "[eBest] Live trading disabled. Set MASP_ENABLE_LIVE_TRADING=1"
            )

    def _ensure_account(self) -> None:
        """Check if account number is configured."""
        if not self._account_no:
            raise ValueError(
                "[eBest] Account number required. Set EBEST_ACCOUNT_NO environment variable."
            )

    async def _place_order_async(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> OrderResult:
        """Place order (async)."""
        if not await self._ensure_login():
            return OrderResult(
                success=False,
                message="Login failed",
                mock=False,
            )

        try:
            # CSPAT00600: 현물 주문
            # BnsTpCode: 1=매도, 2=매수
            # OrdprcPtnCode: 00=지정가, 03=시장가
            bns_tp_code = "2" if side.upper() == "BUY" else "1"
            ordprc_ptn_code = "03" if order_type.upper() == "MARKET" else "00"

            # For market orders, price should be 0
            order_price = 0 if order_type.upper() == "MARKET" else int(price or 0)

            data = {
                "CSPAT00600InBlock1": {
                    "AcntNo": self._account_no,           # Account number
                    "InptPwd": self._account_pwd,         # Transaction password
                    "IsuNo": symbol,                      # Stock code (A + 6 digits, e.g., A005930)
                    "OrdQty": int(quantity),              # 주문수량
                    "OrdPrc": order_price,                # 주문가격
                    "BnsTpCode": bns_tp_code,             # 매매구분
                    "OrdprcPtnCode": ordprc_ptn_code,     # 호가유형
                    "MgntrnCode": "000",                  # 신용거래코드
                    "LoanDt": "",                         # 대출일
                    "OrdCndiTpCode": "0",                 # 주문조건
                }
            }

            # Add "A" prefix if not present (stock code format)
            if not data["CSPAT00600InBlock1"]["IsuNo"].startswith("A"):
                data["CSPAT00600InBlock1"]["IsuNo"] = "A" + symbol

            result = await self._api.request("CSPAT00600", data, path="/stock/order")
            if not result:
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    side=side.upper(),
                    quantity=quantity,
                    message="No response from API",
                    mock=False,
                )

            out_block = result.get("CSPAT00600OutBlock2", {})
            if not out_block:
                error_msg = result.get("rsp_msg", "Unknown error")
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    side=side.upper(),
                    quantity=quantity,
                    message=error_msg,
                    mock=False,
                )

            order_id = str(out_block.get("OrdNo", ""))
            executed_qty = int(out_block.get("OrdQty", quantity))

            return OrderResult(
                success=True,
                order_id=order_id,
                symbol=symbol,
                side=side.upper(),
                quantity=executed_qty,
                price=price,
                status="PENDING",
                mock=False,
            )

        except Exception as e:
            logger.error(f"[eBest] Order error for {symbol}: {e}")
            return OrderResult(
                success=False,
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                message=str(e),
                mock=False,
            )

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> OrderResult:
        """
        Place an order.

        Args:
            symbol: Stock code (6-digit, e.g., "005930")
            side: "BUY" or "SELL"
            quantity: Number of shares
            order_type: "MARKET" or "LIMIT"
            price: Limit price (required for LIMIT orders)

        Returns:
            OrderResult with execution details
        """
        self._ensure_live_trading()
        self._ensure_account()

        if side.upper() not in {"BUY", "SELL"}:
            raise ValueError(f"Invalid side: {side}")

        if order_type.upper() not in {"MARKET", "LIMIT"}:
            raise ValueError(f"Invalid order type: {order_type}")

        if order_type.upper() == "LIMIT" and price is None:
            raise ValueError("Limit price required for LIMIT orders")

        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        result = self._run_async(
            self._place_order_async(symbol, side, quantity, order_type, price)
        )

        # Log trade
        if self._trade_logger and result.success:
            try:
                self._trade_logger.log_trade(
                    exchange="ebest",
                    symbol=symbol,
                    side=side.upper(),
                    quantity=quantity,
                    price=price or 0,
                    order_id=result.order_id,
                )
            except Exception as e:
                logger.warning(f"[eBest] Failed to log trade: {e}")

        return result

    async def _cancel_order_async(self, order_id: str, symbol: str = "") -> bool:
        """Cancel order (async)."""
        if not await self._ensure_login():
            return False

        try:
            # CSPAT00800: 현물 취소 주문
            data = {
                "CSPAT00800InBlock1": {
                    "OrgOrdNo": int(order_id),            # 원주문번호
                    "AcntNo": self._account_no,           # 계좌번호
                    "InptPwd": "",                        # 입력비밀번호
                    "IsuNo": f"A{symbol}" if symbol else "",  # 종목코드
                    "OrdQty": 0,                          # 취소수량 (0=전량)
                }
            }

            result = await self._api.request("CSPAT00800", data, path="/stock/order")
            if not result:
                logger.error(f"[eBest] Cancel order failed: no response")
                return False

            out_block = result.get("CSPAT00800OutBlock2", {})
            if out_block and out_block.get("OrdNo"):
                logger.info(f"[eBest] Order {order_id} cancelled")
                return True

            error_msg = result.get("rsp_msg", "Unknown error")
            logger.error(f"[eBest] Cancel order failed: {error_msg}")
            return False

        except Exception as e:
            logger.error(f"[eBest] Cancel order error: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Order number to cancel

        Returns:
            True if cancelled successfully
        """
        self._ensure_live_trading()
        self._ensure_account()
        return self._run_async(self._cancel_order_async(order_id))

    async def _get_order_status_async(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status (async)."""
        if not await self._ensure_login():
            return None

        try:
            # t0425: 주식 체결/미체결 조회
            data = {
                "t0425InBlock": {
                    "accno": self._account_no,
                    "expcode": "",
                    "chegession": "0",  # 0=전체
                    "medession": "0",   # 0=전체
                    "sortgb": "1",      # 1=주문번호순
                    "cts_ordno": "",
                }
            }

            result = await self._api.request("t0425", data)
            if not result:
                return None

            out_block = result.get("t0425OutBlock1", [])
            for order in out_block:
                if str(order.get("ordno", "")) == order_id:
                    return {
                        "order_id": order_id,
                        "symbol": order.get("expcode", ""),
                        "side": "BUY" if order.get("medosu", "") == "2" else "SELL",
                        "quantity": int(order.get("qty", 0)),
                        "executed_qty": int(order.get("cheqty", 0)),
                        "price": float(order.get("price", 0)),
                        "status": self._parse_order_status(order.get("status", "")),
                    }

            return None

        except Exception as e:
            logger.error(f"[eBest] Get order status error: {e}")
            return None

    def _parse_order_status(self, status_code: str) -> str:
        """Parse eBest order status code to standard status."""
        status_map = {
            "": "UNKNOWN",
            "1": "PENDING",
            "2": "FILLED",
            "3": "PARTIAL",
            "4": "CANCELLED",
        }
        return status_map.get(status_code, "UNKNOWN")

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an existing order.

        Args:
            order_id: Order number

        Returns:
            Order status dict or None if not found
        """
        self._ensure_live_trading()
        self._ensure_account()
        return self._run_async(self._get_order_status_async(order_id))

    async def _get_balance_async(self, asset: str = "KRW") -> Optional[float]:
        """Get balance (async)."""
        if not await self._ensure_login():
            return None

        try:
            # t0424: 주식 잔고 조회
            data = {
                "t0424InBlock": {
                    "accno": self._account_no,
                    "passwd": "",
                    "prcgb": "1",      # 1=평가금액
                    "chegession": "2",  # 2=체결기준
                    "dangession": "0",  # 0=정상
                    "charge": "1",      # 1=제비용포함
                    "cts_expcode": "",
                }
            }

            result = await self._api.request("t0424", data)
            if not result:
                return None

            # If asking for KRW (cash), return available buying power
            if asset.upper() == "KRW":
                out_block = result.get("t0424OutBlock", {})
                # mamt = available buying power (tradable cash)
                # sunamt = total net assets (includes stock valuation)
                available_cash = float(out_block.get("mamt", 0))
                return available_cash

            # If asking for a specific stock, find it in holdings
            out_block1 = result.get("t0424OutBlock1", [])
            for holding in out_block1:
                if holding.get("expcode", "") == asset:
                    return float(holding.get("janqty", 0))  # 잔고수량

            return 0.0

        except Exception as e:
            logger.error(f"[eBest] Get balance error: {e}")
            return None

    def get_balance(self, asset: str = "KRW") -> Optional[float]:
        """
        Get balance for an asset.

        Args:
            asset: "KRW" for cash, or stock code for holdings

        Returns:
            Available balance or None
        """
        self._ensure_live_trading()
        self._ensure_account()
        return self._run_async(self._get_balance_async(asset))

    def get_all_balances(self) -> List[Dict[str, Any]]:
        """
        Get all balances including cash and stock holdings.

        Returns:
            List of balance entries
        """
        self._ensure_live_trading()
        self._ensure_account()
        return self._run_async(self._get_all_balances_async())

    async def _get_all_balances_async(self) -> List[Dict[str, Any]]:
        """Get all balances (async)."""
        if not await self._ensure_login():
            return []

        try:
            data = {
                "t0424InBlock": {
                    "accno": self._account_no,
                    "passwd": "",
                    "prcgb": "1",
                    "chegession": "2",
                    "dangession": "0",
                    "charge": "1",
                    "cts_expcode": "",
                }
            }

            result = await self._api.request("t0424", data)
            if not result:
                return []

            balances = []

            # Cash balance
            out_block = result.get("t0424OutBlock", {})
            balances.append({
                "currency": "KRW",
                "balance": float(out_block.get("sunamt", 0)),
                "available": float(out_block.get("mamt", 0)),  # 매수가능금액
            })

            # Stock holdings
            out_block1 = result.get("t0424OutBlock1", [])
            for holding in out_block1:
                balances.append({
                    "currency": holding.get("expcode", ""),
                    "name": holding.get("hname", ""),
                    "balance": float(holding.get("janqty", 0)),
                    "avg_price": float(holding.get("pamt", 0)),  # 평균단가
                    "current_price": float(holding.get("price", 0)),
                    "eval_amount": float(holding.get("appamt", 0)),  # 평가금액
                    "profit_loss": float(holding.get("dtsuik", 0)),  # 평가손익
                })

            return balances

        except Exception as e:
            logger.error(f"[eBest] Get all balances error: {e}")
            return []
