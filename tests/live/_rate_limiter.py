"""
Rate limit controller for live tests (v2.2 Final).
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, Callable, Dict


class RateLimiter:
    """Upbit API rate limit helper."""

    THRESHOLD_SEC = 2

    def __init__(self, base: float = 0.5, cap: float = 30.0, max_attempts: int = 5):
        self.base = base
        self.cap = cap
        self.max_attempts = max_attempts
        self.remaining_sec = 99

    def parse_remaining_req(self, headers: Dict) -> None:
        """Parse Remaining-Req header for sec value."""
        if not headers:
            return
        header_value = headers.get("Remaining-Req", "")
        if not header_value:
            return
        sec_match = re.search(r"sec=(\d+)", header_value)
        if sec_match:
            self.remaining_sec = int(sec_match.group(1))

    def wait_if_needed(self) -> None:
        """Pre-emptive wait when remaining sec is low."""
        if self.remaining_sec <= self.THRESHOLD_SEC:
            wait_time = 1.1
            print(
                f"⏳ Rate limit threshold (sec={self.remaining_sec}): wait {wait_time}s"
            )
            time.sleep(wait_time)
            self.remaining_sec = 8

    def exponential_backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter."""
        wait = min(self.cap, self.base * (2**attempt))
        jitter = random.uniform(0, self.base)
        return wait + jitter

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with rate limit protections."""
        for attempt in range(self.max_attempts):
            try:
                self.wait_if_needed()
                result = func(*args, **kwargs)
                if hasattr(result, "headers"):
                    self.parse_remaining_req(dict(result.headers))
                return result
            except Exception as exc:
                error_str = str(exc).lower()
                if "418" in str(exc):
                    print(" 418 IP Ban detected - abort")
                    raise RuntimeError(
                        "IP Banned - Immediate Shutdown Required"
                    ) from exc
                if "429" in str(exc) or "too many" in error_str:
                    wait = self.exponential_backoff(attempt)
                    print(
                        f"⚠️ 429 Rate Limit ({attempt + 1}/{self.max_attempts}), wait {wait:.2f}s"
                    )
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"Max retries exceeded ({self.max_attempts})")
