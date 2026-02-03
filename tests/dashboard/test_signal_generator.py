"""Tests for SignalGenerator."""

from __future__ import annotations

from services.dashboard.utils.signal_generator import (
    SignalGenerator,
    get_signal_generator_status,
)


class TestSignalGenerator:
    """Tests for SignalGenerator class."""

    def test_init_demo_mode(self):
        """Demo mode init."""
        gen = SignalGenerator("upbit", allow_live=False)
        assert gen.is_demo_mode is True
        assert "DEMO" in gen.mode_description

    def test_get_symbols_fallback(self):
        """Symbol list in demo mode."""
        gen = SignalGenerator("upbit", allow_live=False)
        symbols = gen.get_symbols(limit=5)
        assert len(symbols) == 5
        assert "BTC/KRW" in symbols
        assert all(isinstance(symbol, str) for symbol in symbols)

    def test_generate_signal_mock(self):
        """Mock signal generation."""
        gen = SignalGenerator("upbit", allow_live=False)
        signal = gen.generate_signal("BTC/KRW")

        assert signal["symbol"] == "BTC/KRW"
        assert signal["signal"] in ["BUY", "SELL", "HOLD", "ERROR"]
        assert signal["is_mock"] is True
        assert "timestamp" in signal

    def test_status_function(self):
        """Status helper."""
        status = get_signal_generator_status("upbit", allow_live=False)
        assert status["exchange"] == "upbit"
        assert status["is_demo_mode"] is True
        assert "mode_description" in status
