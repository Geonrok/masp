import functools
import hashlib
import logging
import os
import random
import time
import uuid
import urllib.parse
from typing import Callable, Dict, List, Optional, TypeVar

import jwt
import requests

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_on_error(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    ),
    retryable_status_codes: tuple = (429, 500, 502, 503, 504),
) -> Callable:
    """
    Retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        retryable_exceptions: Exceptions that trigger retry
        retryable_status_codes: HTTP status codes that trigger retry
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = _calculate_delay(attempt, base_delay, max_delay, exponential_base)
                        logger.warning(
                            "[Retry] %s attempt %d/%d failed: %s. Retrying in %.1fs",
                            func.__name__,
                            attempt + 1,
                            max_retries,
                            type(e).__name__,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "[Retry] %s failed after %d attempts: %s",
                            func.__name__,
                            max_retries + 1,
                            e,
                        )

                except requests.exceptions.HTTPError as e:
                    if (
                        hasattr(e, "response")
                        and e.response is not None
                        and e.response.status_code in retryable_status_codes
                    ):
                        last_exception = e
                        if attempt < max_retries:
                            delay = _calculate_delay(attempt, base_delay, max_delay, exponential_base)
                            logger.warning(
                                "[Retry] %s attempt %d/%d got HTTP %d. Retrying in %.1fs",
                                func.__name__,
                                attempt + 1,
                                max_retries,
                                e.response.status_code,
                                delay,
                            )
                            time.sleep(delay)
                        else:
                            logger.error(
                                "[Retry] %s failed after %d attempts: HTTP %d",
                                func.__name__,
                                max_retries + 1,
                                e.response.status_code,
                            )
                    else:
                        raise

            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected retry failure in {func.__name__}")

        return wrapper

    return decorator


def _calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = base_delay * (exponential_base ** attempt)
    delay = min(delay, max_delay)
    # Add jitter (±20%)
    jitter = delay * 0.2 * (random.random() * 2 - 1)
    return max(0.1, delay + jitter)


class BithumbAPIV2:
    """Bithumb Open API 2.0 (JWT) client."""

    BASE_URL = "https://api.bithumb.com"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        session: Optional[requests.Session] = None,
        timeout: int = 10,
    ) -> None:
        if not api_key or api_key == "your_api_key_here":
            raise ValueError("BITHUMB_API_KEY not set or invalid")
        if not secret_key or secret_key == "your_secret_key_here":
            raise ValueError("BITHUMB_SECRET_KEY not set or invalid")

        self.api_key = api_key
        self.secret_key = secret_key
        self.session = session or requests.Session()
        self.timeout = timeout

    @staticmethod
    def _to_query_items(params: Optional[Dict]):
        """
        Convert params into a list of (key, value) pairs, preserving insertion order.
        - Skips None values
        - Encodes list/tuple values as key[]=v1&key[]=v2 (per Bithumb docs)
        
        [ChatGPT 필수 보강 #1] sorted() 제거 - 서버와 동일한 순서 유지
        """
        if not params:
            return []

        items = []
        for key, value in params.items():  # preserves dict insertion order (Python 3.7+)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                for item in value:
                    items.append((f"{key}[]", str(item)))
            else:
                items.append((key, str(value)))

        return items

    @classmethod
    def _encode_query(cls, params: Optional[Dict]) -> str:
        items = cls._to_query_items(params)
        return urllib.parse.urlencode(items)

    @staticmethod
    def _make_query_hash(query: str) -> str:
        return hashlib.sha512(query.encode("utf-8")).hexdigest()

    def _generate_jwt(self, params: Optional[Dict] = None) -> str:
        payload = {
            "access_key": self.api_key,
            "nonce": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),
        }

        force_empty_query_hash = os.getenv("BITHUMB_JWT_INCLUDE_EMPTY_QUERY_HASH") == "1"
        has_params = bool(params)
        if has_params or force_empty_query_hash:
            query = self._encode_query(params or {})
            payload["query_hash"] = self._make_query_hash(query)
            payload["query_hash_alg"] = "SHA512"
        else:
            query = ""

        if logger.isEnabledFor(logging.DEBUG):
            masked_key = f"{self.api_key[:8]}..." if self.api_key else ""
            debug_payload = dict(payload)
            debug_payload["access_key"] = masked_key
            if "query_hash" in debug_payload:
                debug_payload["query_hash"] = f"{debug_payload['query_hash'][:12]}..."
            logger.debug("[BithumbAPIV2] JWT Payload: %s", debug_payload)
            if "query_hash" in payload:
                logger.debug(
                    "[BithumbAPIV2] JWT Query: %s",
                    query if query else "<empty>",
                )

        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        if isinstance(token, bytes):
            return token.decode("utf-8")
        return token

    def _raise_if_error(self, result: Dict) -> None:
        if not isinstance(result, dict) or "error" not in result:
            return

        error = result.get("error")
        if isinstance(error, dict):
            name = error.get("name")
            message = error.get("message") or error.get("msg")
            if name and message:
                raise ValueError(f"Bithumb API Error: {name}: {message}")
            if message:
                raise ValueError(f"Bithumb API Error: {message}")
            raise ValueError(f"Bithumb API Error: {error}")
        raise ValueError(f"Bithumb API Error: {error}")

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None):
        url = f"{self.BASE_URL}{endpoint}"
        jwt_token = self._generate_jwt(params)
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }

        if logger.isEnabledFor(logging.DEBUG):
            token_preview = f"{jwt_token[:32]}..." if jwt_token else ""
            logger.debug("[BithumbAPIV2] Request: %s %s", method, url)
            logger.debug("[BithumbAPIV2] Params: %s", params)
            logger.debug(
                "[BithumbAPIV2] Authorization: Bearer %s (len=%s)",
                token_preview,
                len(jwt_token) if jwt_token else 0,
            )

        try:
            # [ChatGPT 필수 보강 #2] GET/DELETE에서 query_hash와 동일한 인코딩 사용
            if method in ("GET", "DELETE"):
                # IMPORTANT: pass query items to requests so the actual query string encoding
                # matches what we used to compute query_hash (especially for key[] arrays)
                req_params = self._to_query_items(params) if params else None
                if method == "GET":
                    resp = self.session.get(
                        url, headers=headers, params=req_params, timeout=self.timeout
                    )
                else:
                    resp = self.session.delete(
                        url, headers=headers, params=req_params, timeout=self.timeout
                    )
            elif method == "POST":
                resp = self.session.post(
                    url, headers=headers, json=params, timeout=self.timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            resp.raise_for_status()
            result = resp.json()
            self._raise_if_error(result)
            return result
        except requests.exceptions.HTTPError as http_err:
            # [ChatGPT 필수 보강] 401 에러 바디에서 error.name 추출
            detail = self._extract_error_detail(http_err.response)
            raise PermissionError(
                f"Bithumb HTTP {http_err.response.status_code}: {detail}"
            ) from http_err
        except requests.exceptions.RequestException as exc:
            raise ConnectionError(f"Network Error: {exc}") from exc

    @staticmethod
    def _extract_error_detail(resp) -> str:
        """
        401 에러 바디에서 error.name/message 추출
        - NotAllowIP: 허용되지 않은 IP
        - out_of_scope: 권한 부족
        - jwt_verification: 키/시크릿 불일치
        - expired_jwt: 시간 만료
        """
        if resp is None:
            return "No response body"
        try:
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                err = data.get("error")
                if isinstance(err, dict):
                    name = err.get("name")
                    msg = err.get("message") or err.get("msg")
                    if name and msg:
                        return f"{name}: {msg}"
                    if name:
                        return str(name)
                    if msg:
                        return str(msg)
                return str(err)
            return str(data)
        except Exception:
            return (getattr(resp, 'text', '') or "").strip()[:200]

    def _public_get(self, endpoint: str, params: Optional[Dict] = None):
        url = f"{self.BASE_URL}{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            raise ConnectionError(f"Network Error: {exc}") from exc

    # === Public API ===

    def get_ticker(self, markets: List[str]):
        params = {"markets": ",".join(markets)}
        return self._public_get("/v1/ticker", params)

    # === Private API ===

    def get_accounts(self) -> List[Dict]:
        return self._request("GET", "/v1/accounts")

    def post_order(
        self,
        *,
        market: str,
        side: str,
        ord_type: str,
        volume: Optional[str] = None,
        price: Optional[str] = None,
    ) -> Dict:
        params = {
            "market": market,
            "side": side,
            "ord_type": ord_type,
        }
        if volume is not None:
            params["volume"] = volume
        if price is not None:
            params["price"] = price

        return self._request("POST", "/v1/orders", params)

    def get_order(self, uuid_value: str) -> Dict:
        return self._request("GET", "/v1/order", {"uuid": uuid_value})

    def cancel_order(self, uuid_value: str) -> Dict:
        return self._request("DELETE", "/v1/order", {"uuid": uuid_value})

    def get_orders(
        self,
        market: Optional[str] = None,
        state: Optional[str] = None,
        states: Optional[List[str]] = None,
        uuids: Optional[List[str]] = None,
        page: int = 1,
        limit: int = 100,
        order_by: str = "desc",
    ) -> List[Dict]:
        """
        주문 목록 조회.

        Args:
            market: 마켓 (예: "KRW-BTC")
            state: 단일 상태 필터 (wait, watch, done, cancel)
            states: 복수 상태 필터
            uuids: UUID 목록으로 필터
            page: 페이지 번호
            limit: 페이지당 결과 수
            order_by: 정렬 (asc/desc)

        Returns:
            주문 목록
        """
        params: Dict = {
            "page": page,
            "limit": limit,
            "order_by": order_by,
        }
        if market:
            params["market"] = market
        if state:
            params["state"] = state
        if states:
            params["states"] = states
        if uuids:
            params["uuids"] = uuids

        return self._request("GET", "/v1/orders", params)

    def get_orders_chance(self, market: str) -> Dict:
        """
        주문 가능 정보 조회.

        Args:
            market: 마켓 (예: "KRW-BTC")

        Returns:
            주문 가능 정보 (수수료, 제한 등)
        """
        return self._request("GET", "/v1/orders/chance", {"market": market})
