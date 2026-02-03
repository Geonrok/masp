from unittest.mock import MagicMock, patch


def test_disabled_without_credentials():
    """Missing credentials disables notifier."""
    with patch.dict("os.environ", {}, clear=True):
        from libs.notifications.telegram import TelegramNotifier

        t = TelegramNotifier(bot_token=None, chat_id=None)
        assert t.enabled is False
        assert t.send_message_sync("test") is False


def test_enabled_with_credentials():
    """Credentials enable notifier."""
    from libs.notifications.telegram import TelegramNotifier

    t = TelegramNotifier(bot_token="123:ABC", chat_id="12345")
    assert t.enabled is True


def test_format_trade_message_buy():
    """BUY message format."""
    from libs.notifications.telegram import format_trade_message

    msg = format_trade_message("upbit", "BTC/KRW", "BUY", 0.001, 50000000, "FILLED")
    assert "UPBIT" in msg
    assert "BUY" in msg


def test_format_trade_message_sell():
    """SELL message format."""
    from libs.notifications.telegram import format_trade_message

    msg = format_trade_message("bithumb", "ETH/KRW", "SELL", 0.1, 3500000, "FILLED")
    assert "SELL" in msg


def test_format_daily_summary_positive():
    """Daily summary with profit."""
    from libs.notifications.telegram import format_daily_summary

    msg = format_daily_summary("upbit", 10, 150000)
    assert "+150,000" in msg


def test_format_daily_summary_negative():
    """Daily summary with loss."""
    from libs.notifications.telegram import format_daily_summary

    msg = format_daily_summary("upbit", 5, -50000)
    assert "-50,000" in msg


def test_html_escape():
    """HTML escape is applied."""
    from libs.notifications.telegram import format_trade_message

    msg = format_trade_message("<script>", "BTC/KRW", "BUY", 0.001, 50000000, "OK")
    assert "<script>" not in msg
    assert "&lt;SCRIPT&gt;" in msg


def test_message_truncation():
    """Long messages are truncated."""
    from libs.notifications.telegram import TelegramNotifier

    t = TelegramNotifier(bot_token="test", chat_id="test")
    long_text = "x" * 5000

    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = (
            mock_response
        )
        assert t.send_message_sync(long_text) is True

        args, kwargs = mock_client.return_value.__enter__.return_value.post.call_args
        payload = kwargs.get("json", {})
        sent_text = payload.get("text", "")
        assert len(sent_text) <= 4000
        assert sent_text.endswith("...[truncated]")
