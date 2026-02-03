"""
Email Notification Service

Sends email notifications via SMTP.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    """Email configuration."""

    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    username: str = ""
    password: str = ""
    from_address: str = ""
    from_name: str = "MASP Trading System"


class EmailNotifier:
    """Send notifications via email."""

    def __init__(
        self,
        config: Optional[EmailConfig] = None,
        recipients: Optional[List[str]] = None,
    ):
        """
        Initialize email notifier.

        Args:
            config: Email configuration (uses env vars if None)
            recipients: Default recipient list
        """
        if config:
            self.config = config
        else:
            self.config = EmailConfig(
                smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
                smtp_port=int(os.getenv("SMTP_PORT", "587")),
                use_tls=os.getenv("SMTP_USE_TLS", "true").lower() == "true",
                username=os.getenv("SMTP_USERNAME", ""),
                password=os.getenv("SMTP_PASSWORD", ""),
                from_address=os.getenv("SMTP_FROM_ADDRESS", ""),
                from_name=os.getenv("SMTP_FROM_NAME", "MASP Trading System"),
            )

        self.recipients = recipients or []
        if not self.recipients:
            env_recipients = os.getenv("EMAIL_RECIPIENTS", "")
            if env_recipients:
                self.recipients = [r.strip() for r in env_recipients.split(",")]

        self._enabled = bool(
            self.config.username
            and self.config.password
            and self.config.from_address
            and self.recipients
        )

        if not self._enabled:
            logger.info("[Email] Not configured - disabled")
        else:
            logger.info(f"[Email] Configured for {len(self.recipients)} recipient(s)")

    @property
    def enabled(self) -> bool:
        """Check if email is enabled."""
        return self._enabled

    def send_email(
        self,
        subject: str,
        body_html: str,
        body_text: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        priority: str = "normal",
    ) -> bool:
        """
        Send an email.

        Args:
            subject: Email subject
            body_html: HTML body content
            body_text: Plain text body (optional, generated from HTML if not provided)
            recipients: Override default recipients
            priority: Email priority (high, normal, low)

        Returns:
            True if sent successfully
        """
        if not self._enabled:
            return False

        to_list = recipients or self.recipients
        if not to_list:
            logger.warning("[Email] No recipients specified")
            return False

        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{self.config.from_name} <{self.config.from_address}>"
        msg["To"] = ", ".join(to_list)

        # Set priority
        if priority.lower() == "high":
            msg["X-Priority"] = "1"
            msg["X-MSMail-Priority"] = "High"
        elif priority.lower() == "low":
            msg["X-Priority"] = "5"
            msg["X-MSMail-Priority"] = "Low"

        # Add plain text part
        if body_text:
            msg.attach(MIMEText(body_text, "plain"))
        else:
            # Generate plain text from HTML (basic)
            import re

            plain = re.sub(r"<[^>]+>", "", body_html)
            plain = plain.replace("&nbsp;", " ").replace("&amp;", "&")
            msg.attach(MIMEText(plain, "plain"))

        # Add HTML part
        msg.attach(MIMEText(body_html, "html"))

        try:
            if self.config.use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(
                    self.config.smtp_host, self.config.smtp_port
                ) as server:
                    server.starttls(context=context)
                    server.login(self.config.username, self.config.password)
                    server.sendmail(self.config.from_address, to_list, msg.as_string())
            else:
                with smtplib.SMTP_SSL(
                    self.config.smtp_host, self.config.smtp_port
                ) as server:
                    server.login(self.config.username, self.config.password)
                    server.sendmail(self.config.from_address, to_list, msg.as_string())

            logger.info(f"[Email] Sent to {len(to_list)} recipient(s)")
            return True

        except Exception as exc:
            logger.error(f"[Email] Failed to send: {exc}")
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
        Send trade notification email.

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
        side_color = "#28a745" if side_upper == "BUY" else "#dc3545"

        pnl_html = ""
        if pnl is not None:
            pnl_color = "#28a745" if pnl >= 0 else "#dc3545"
            pnl_html = f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>PnL</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd; color: {pnl_color};">
                    {pnl:+,.0f} KRW
                </td>
            </tr>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; }}
                .header {{ background: {side_color}; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background: #f9f9f9; }}
                table {{ width: 100%; border-collapse: collapse; }}
                td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                .footer {{ padding: 10px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Trade Executed: {side_upper}</h2>
                </div>
                <div class="content">
                    <table>
                        <tr>
                            <td><strong>Exchange</strong></td>
                            <td>{exchange.upper()}</td>
                        </tr>
                        <tr>
                            <td><strong>Symbol</strong></td>
                            <td>{symbol}</td>
                        </tr>
                        <tr>
                            <td><strong>Side</strong></td>
                            <td style="color: {side_color};"><strong>{side_upper}</strong></td>
                        </tr>
                        <tr>
                            <td><strong>Quantity</strong></td>
                            <td>{quantity:.8f}</td>
                        </tr>
                        <tr>
                            <td><strong>Price</strong></td>
                            <td>{price:,.0f} KRW</td>
                        </tr>
                        <tr>
                            <td><strong>Status</strong></td>
                            <td>{status}</td>
                        </tr>
                        {pnl_html}
                    </table>
                </div>
                <div class="footer">
                    MASP Trading System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """

        return self.send_email(
            subject=f"[MASP] Trade: {side_upper} {symbol} @ {price:,.0f}",
            body_html=html,
            priority="high" if abs(pnl or 0) > 100000 else "normal",
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
        Send alert email.

        Args:
            title: Alert title
            message: Alert message
            priority: Alert priority
            alert_type: Type of alert
            metadata: Additional metadata

        Returns:
            True if sent
        """
        priority_colors = {
            "CRITICAL": "#dc3545",
            "HIGH": "#fd7e14",
            "NORMAL": "#28a745",
            "LOW": "#6c757d",
        }
        color = priority_colors.get(priority.upper(), "#17a2b8")

        metadata_html = ""
        if metadata:
            rows = "".join(
                f"<tr><td><strong>{k.replace('_', ' ').title()}</strong></td><td>{v}</td></tr>"
                for k, v in metadata.items()
            )
            metadata_html = f"""
            <h3>Details</h3>
            <table>{rows}</table>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; }}
                .header {{ background: {color}; color: white; padding: 20px; }}
                .content {{ padding: 20px; background: #f9f9f9; }}
                .message {{ background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                .footer {{ padding: 10px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>{title}</h2>
                    <p>Priority: {priority.upper()} | Type: {alert_type.upper()}</p>
                </div>
                <div class="content">
                    <div class="message">
                        {message}
                    </div>
                    {metadata_html}
                </div>
                <div class="footer">
                    MASP Alert System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """

        email_priority = (
            "high" if priority.upper() in ["CRITICAL", "HIGH"] else "normal"
        )

        return self.send_email(
            subject=f"[MASP] [{priority.upper()}] {title}",
            body_html=html,
            priority=email_priority,
        )

    def send_daily_summary(
        self,
        exchange: str,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        win_rate: float,
        equity_curve: Optional[List[float]] = None,
    ) -> bool:
        """
        Send daily trading summary email.

        Args:
            exchange: Exchange name
            total_trades: Total trades
            winning_trades: Winning trades
            losing_trades: Losing trades
            total_pnl: Total PnL
            win_rate: Win rate percentage
            equity_curve: Optional equity values

        Returns:
            True if sent
        """
        pnl_color = "#28a745" if total_pnl >= 0 else "#dc3545"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; }}
                .header {{ background: #343a40; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background: #f9f9f9; }}
                .pnl {{ font-size: 32px; color: {pnl_color}; text-align: center; margin: 20px 0; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat {{ text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; }}
                .stat-label {{ font-size: 12px; color: #666; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                .footer {{ padding: 10px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Daily Trading Summary</h2>
                    <p>{exchange.upper()} | {datetime.now().strftime('%Y-%m-%d')}</p>
                </div>
                <div class="content">
                    <div class="pnl">
                        {total_pnl:+,.0f} KRW
                    </div>
                    <table>
                        <tr>
                            <td><strong>Total Trades</strong></td>
                            <td>{total_trades}</td>
                        </tr>
                        <tr>
                            <td><strong>Winning Trades</strong></td>
                            <td style="color: #28a745;">{winning_trades}</td>
                        </tr>
                        <tr>
                            <td><strong>Losing Trades</strong></td>
                            <td style="color: #dc3545;">{losing_trades}</td>
                        </tr>
                        <tr>
                            <td><strong>Win Rate</strong></td>
                            <td>{win_rate:.1f}%</td>
                        </tr>
                    </table>
                </div>
                <div class="footer">
                    MASP Trading System
                </div>
            </div>
        </body>
        </html>
        """

        return self.send_email(
            subject=f"[MASP] Daily Summary: {total_pnl:+,.0f} KRW ({exchange})",
            body_html=html,
        )
