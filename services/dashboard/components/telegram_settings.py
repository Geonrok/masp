"""Telegram notification settings - status display and test."""
from __future__ import annotations

import html
import os

import streamlit as st


def get_telegram_status() -> dict:
    """Return current Telegram configuration status."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    return {
        "configured": bool(bot_token and chat_id),
        "bot_token_set": bool(bot_token),
        "chat_id_set": bool(chat_id),
        "chat_id": chat_id if chat_id else "Not set",
        "bot_token": bot_token,
    }


def mask_token(token: str) -> str:
    """Mask token display."""
    if not token or len(token) < 10:
        return "Not configured"
    return f"{token[:6]}...{token[-4:]}"


def render_telegram_settings() -> None:
    """Render Telegram configuration status and test."""
    st.subheader("Telegram Notification Status")

    status = get_telegram_status()

    if status["configured"]:
        st.success("Telegram is configured and ready.")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bot Token", mask_token(status["bot_token"]))
        with col2:
            st.metric("Chat ID", status["chat_id"])
    else:
        st.warning("Telegram is not fully configured.")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bot Token", "Set" if status["bot_token_set"] else "Missing")
        with col2:
            st.metric("Chat ID", "Set" if status["chat_id_set"] else "Missing")

    st.divider()

    with st.expander("How to Configure", expanded=not status["configured"]):
        st.markdown(
            "**Set environment variables before starting the dashboard:**\n"
            "```powershell\n"
            "$env:TELEGRAM_BOT_TOKEN=\"your-bot-token\"\n"
            "$env:TELEGRAM_CHAT_ID=\"your-chat-id\"\n"
            "```\n\n"
            "**How to get these values:**\n"
            "1. Create a bot via @BotFather\n"
            "2. Get your Chat ID via @userinfobot"
        )

    st.divider()
    st.subheader("Test Connection")

    if not status["configured"]:
        st.info("Configure Telegram first to test connection.")
        return

    test_message = st.text_input(
        "Test Message",
        value="MASP Dashboard Test",
        key="telegram_test_msg",
        max_chars=200,
    )

    if st.button("Send Test Message", key="telegram_test_btn"):
        try:
            from libs.notifications.telegram import TelegramNotifier

            with st.spinner("Sending..."):
                notifier = TelegramNotifier()
                safe_message = html.escape(test_message)
                success = notifier.send_message_sync(safe_message, parse_mode=None)

            if success:
                st.success("Test message sent successfully.")
            else:
                st.error("Failed to send. Check logs for details.")
        except ImportError:
            st.error("TelegramNotifier not available.")
        except Exception as exc:
            st.error(f"Error: {str(exc)[:100]}")
        finally:
            st.session_state["telegram_test_msg"] = ""
