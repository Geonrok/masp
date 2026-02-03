"""
Slack Notification Service

Sends notifications to Slack via incoming webhooks.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SlackAttachment:
    """Slack message attachment."""

    fallback: str
    color: str = "#36a64f"  # Good (green)
    title: Optional[str] = None
    text: Optional[str] = None
    fields: Optional[List[Dict[str, Any]]] = None
    footer: Optional[str] = None
    ts: Optional[int] = None


class SlackNotifier:
    """Send notifications to Slack via webhooks."""

    # Color codes for different alert levels
    COLORS = {
        "CRITICAL": "#FF0000",  # Red
        "HIGH": "#FFA500",  # Orange
        "NORMAL": "#36a64f",  # Green
        "LOW": "#808080",  # Gray
        "ERROR": "#FF0000",
        "SUCCESS": "#36a64f",
        "WARNING": "#FFA500",
        "INFO": "#0000FF",
    }

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        channel: Optional[str] = None,
        username: str = "MASP Bot",
        icon_emoji: str = ":chart_with_upwards_trend:",
    ):
        """
        Initialize Slack notifier.

        Args:
            webhook_url: Slack incoming webhook URL
            channel: Channel to post to (optional, uses webhook default)
            username: Bot username
            icon_emoji: Bot icon emoji
        """
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.channel = channel or os.getenv("SLACK_CHANNEL")
        self.username = username
        self.icon_emoji = icon_emoji
        self._enabled = bool(self.webhook_url)

        if not self._enabled:
            logger.info("[Slack] Not configured - disabled")
        else:
            logger.info("[Slack] Configured and enabled")

    @property
    def enabled(self) -> bool:
        """Check if Slack is enabled."""
        return self._enabled

    def send_message(
        self,
        text: str,
        attachments: Optional[List[SlackAttachment]] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Send a message to Slack.

        Args:
            text: Main message text
            attachments: Optional list of attachments
            blocks: Optional block kit elements

        Returns:
            True if sent successfully
        """
        if not self._enabled:
            return False

        payload: Dict[str, Any] = {
            "text": text,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
        }

        if self.channel:
            payload["channel"] = self.channel

        if attachments:
            payload["attachments"] = [
                {
                    "fallback": a.fallback,
                    "color": a.color,
                    "title": a.title,
                    "text": a.text,
                    "fields": a.fields or [],
                    "footer": a.footer,
                    "ts": a.ts or int(datetime.now().timestamp()),
                }
                for a in attachments
            ]

        if blocks:
            payload["blocks"] = blocks

        try:
            import httpx

            with httpx.Client(timeout=5.0) as client:
                resp = client.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                if resp.status_code == 200:
                    logger.debug("[Slack] Message sent")
                    return True

                logger.warning(f"[Slack] Failed: {resp.status_code} - {resp.text}")

        except Exception as exc:
            logger.warning(f"[Slack] Error (swallowed): {exc}")

        return False

    def send_trade_notification(
        self,
        exchange: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        status: str = "FILLED",
        pnl: Optional[float] = None,
    ) -> bool:
        """
        Send a trade notification.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Trade quantity
            price: Trade price
            status: Order status
            pnl: Profit/loss (optional)

        Returns:
            True if sent
        """
        side_upper = side.upper()
        emoji = (
            ":chart_with_upwards_trend:"
            if side_upper == "BUY"
            else ":chart_with_downwards_trend:"
        )
        color = (
            self.COLORS["SUCCESS"] if side_upper == "BUY" else self.COLORS["WARNING"]
        )

        fields = [
            {"title": "Exchange", "value": exchange.upper(), "short": True},
            {"title": "Symbol", "value": symbol, "short": True},
            {"title": "Side", "value": side_upper, "short": True},
            {"title": "Status", "value": status, "short": True},
            {"title": "Quantity", "value": f"{quantity:.8f}", "short": True},
            {"title": "Price", "value": f"{price:,.0f} KRW", "short": True},
        ]

        if pnl is not None:
            pnl_emoji = ":money_with_wings:" if pnl > 0 else ":money_mouth_face:"
            fields.append(
                {
                    "title": "PnL",
                    "value": f"{pnl_emoji} {pnl:+,.0f} KRW",
                    "short": True,
                }
            )

        attachment = SlackAttachment(
            fallback=f"{side_upper} {quantity} {symbol} @ {price}",
            color=color,
            title=f"{emoji} Trade Executed",
            fields=fields,
            footer="MASP Trading System",
        )

        return self.send_message(
            text=f"Trade: {side_upper} {symbol}",
            attachments=[attachment],
        )

    def send_signal_notification(
        self,
        strategy: str,
        symbol: str,
        signal: str,
        strength: float = 0.0,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Send a signal notification.

        Args:
            strategy: Strategy name
            symbol: Trading symbol
            signal: Signal type (BUY, SELL, HOLD)
            strength: Signal strength (0-1)
            reason: Reason for signal

        Returns:
            True if sent
        """
        signal_upper = signal.upper()

        if signal_upper == "BUY":
            color = self.COLORS["SUCCESS"]
            emoji = ":arrow_up:"
        elif signal_upper == "SELL":
            color = self.COLORS["WARNING"]
            emoji = ":arrow_down:"
        else:
            color = self.COLORS["INFO"]
            emoji = ":pause_button:"

        fields = [
            {"title": "Strategy", "value": strategy, "short": True},
            {"title": "Symbol", "value": symbol, "short": True},
            {"title": "Signal", "value": signal_upper, "short": True},
            {"title": "Strength", "value": f"{strength:.1%}", "short": True},
        ]

        if reason:
            fields.append({"title": "Reason", "value": reason, "short": False})

        attachment = SlackAttachment(
            fallback=f"{strategy}: {signal_upper} {symbol}",
            color=color,
            title=f"{emoji} Signal Generated",
            fields=fields,
            footer="MASP Strategy Engine",
        )

        return self.send_message(
            text=f"Signal: {strategy} -> {signal_upper} {symbol}",
            attachments=[attachment],
        )

    def send_alert(
        self,
        title: str,
        message: str,
        priority: str = "NORMAL",
        alert_type: str = "SYSTEM",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send a general alert.

        Args:
            title: Alert title
            message: Alert message
            priority: Alert priority
            alert_type: Type of alert
            metadata: Additional metadata

        Returns:
            True if sent
        """
        color = self.COLORS.get(priority.upper(), self.COLORS["INFO"])

        priority_emojis = {
            "CRITICAL": ":rotating_light:",
            "HIGH": ":warning:",
            "NORMAL": ":information_source:",
            "LOW": ":speech_balloon:",
        }
        emoji = priority_emojis.get(priority.upper(), ":bell:")

        fields = [
            {"title": "Priority", "value": priority.upper(), "short": True},
            {"title": "Type", "value": alert_type.upper(), "short": True},
        ]

        if metadata:
            for key, value in metadata.items():
                fields.append(
                    {
                        "title": key.replace("_", " ").title(),
                        "value": str(value),
                        "short": True,
                    }
                )

        attachment = SlackAttachment(
            fallback=f"{priority}: {title}",
            color=color,
            title=f"{emoji} {title}",
            text=message,
            fields=fields,
            footer="MASP Alert System",
        )

        return self.send_message(
            text=f"Alert: {title}",
            attachments=[attachment],
        )

    def send_daily_summary(
        self,
        exchange: str,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        win_rate: float,
        best_trade: Optional[str] = None,
        worst_trade: Optional[str] = None,
    ) -> bool:
        """
        Send daily trading summary.

        Args:
            exchange: Exchange name
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            total_pnl: Total profit/loss
            win_rate: Win rate percentage
            best_trade: Best trade description
            worst_trade: Worst trade description

        Returns:
            True if sent
        """
        pnl_emoji = (
            ":chart_with_upwards_trend:"
            if total_pnl >= 0
            else ":chart_with_downwards_trend:"
        )
        color = self.COLORS["SUCCESS"] if total_pnl >= 0 else self.COLORS["ERROR"]

        fields = [
            {"title": "Exchange", "value": exchange.upper(), "short": True},
            {"title": "Total Trades", "value": str(total_trades), "short": True},
            {"title": "Winning", "value": str(winning_trades), "short": True},
            {"title": "Losing", "value": str(losing_trades), "short": True},
            {"title": "Win Rate", "value": f"{win_rate:.1f}%", "short": True},
            {"title": "Total PnL", "value": f"{total_pnl:+,.0f} KRW", "short": True},
        ]

        if best_trade:
            fields.append({"title": "Best Trade", "value": best_trade, "short": False})
        if worst_trade:
            fields.append(
                {"title": "Worst Trade", "value": worst_trade, "short": False}
            )

        attachment = SlackAttachment(
            fallback=f"Daily Summary: {total_pnl:+,.0f} KRW",
            color=color,
            title=f"{pnl_emoji} Daily Trading Summary",
            fields=fields,
            footer=f"MASP - {datetime.now().strftime('%Y-%m-%d')}",
        )

        return self.send_message(
            text=f"Daily Summary for {exchange}",
            attachments=[attachment],
        )
