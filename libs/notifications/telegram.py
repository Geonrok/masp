"""Telegram notification service - best effort, non-blocking."""
from __future__ import annotations

import html
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send notifications to Telegram bot (best-effort)."""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self._enabled = bool(self.bot_token and self.chat_id)

        if not self._enabled:
            logger.info("[Telegram] Not configured - disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send_message_sync(self, text: str, parse_mode: str = "HTML") -> bool:
        """Sync send with 3s timeout (best-effort, swallow failures)."""
        if not self._enabled:
            return False

        if len(text) > 4000:
            text = text[:3900] + "\n...[truncated]"

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            import httpx

            with httpx.Client(timeout=3.0) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    logger.debug("[Telegram] Sent")
                    return True
                logger.warning("[Telegram] Failed: %s", resp.status_code)
        except Exception as exc:
            logger.warning("[Telegram] Error (swallowed): %s", exc)

        return False


def format_trade_message(
    exchange: str, symbol: str, side: str, quantity: float, price: float, status: str
) -> str:
    """Format trade notification with HTML escape."""
    emoji = ""
    return (
        f"{emoji} <b>{html.escape(exchange.upper())}</b>\n"
        f"Symbol: {html.escape(symbol)}\n"
        f"Side: {side}\n"
        f"Qty: {quantity:.8f}\n"
        f"Price: {price:,.0f} KRW\n"
        f"Status: {status}"
    )


def format_daily_summary(exchange: str, trades: int, pnl: float) -> str:
    """Format daily summary."""
    emoji = ""
    return (
        f"{emoji} <b>Daily - {html.escape(exchange.upper())}</b>\n"
        f"Trades: {trades}\n"
        f"PnL: {pnl:+,.0f} KRW"
    )
