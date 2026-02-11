"""Signal generator for MASP Dashboard with fallback support."""

from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st

logger = logging.getLogger(__name__)


def _map_market_exchange(exchange: str) -> str:
    """Map dashboard exchange name to adapter exchange name."""
    mapping = {
        "upbit": "upbit_spot",
        "bithumb": "bithumb_spot",
        "binance": "binance_spot",
    }
    return mapping.get(exchange, exchange)


def _load_strategy(force_demo: bool):
    """Load strategy class with fallback to mock."""
    if force_demo:
        return _MockStrategy(), True
    try:
        from libs.strategies.ankle_buy_v2 import AnkleBuyV2Strategy

        logger.info("Loaded AnkleBuyV2Strategy")
        return AnkleBuyV2Strategy(), False
    except Exception as exc:
        logger.warning("Strategy load failed, using mock: %s", type(exc).__name__)
        return _MockStrategy(), True


def _load_market_adapter(exchange: str, force_demo: bool):
    """Load market data adapter with fallback."""
    if force_demo:
        return _MockMarketAdapter(exchange), True
    try:
        from libs.adapters.factory import AdapterFactory

        adapter = AdapterFactory.create_market_data(_map_market_exchange(exchange))
        return adapter, False
    except Exception as exc:
        logger.warning("Market adapter load failed, using mock: %s", type(exc).__name__)
        return _MockMarketAdapter(exchange), True


class _MockStrategy:
    """Mock strategy for demo mode."""

    def generate_signal(
        self, symbol: str, ohlcv: List[Dict[str, Any]] | None = None
    ) -> Dict[str, Any]:
        """Generate mock signal for a symbol."""
        data = ohlcv or []
        result = self.calculate_signal(data)
        result["symbol"] = symbol
        return result

    def calculate_signal(self, ohlcv: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate mock signal."""
        if not ohlcv:
            return {"signal": "HOLD", "strength": 0.0}

        recent_close = ohlcv[-1].get("close", 0)
        prev_close = (
            ohlcv[-2].get("close", recent_close) if len(ohlcv) > 1 else recent_close
        )

        if recent_close > prev_close * 1.01:
            return {"signal": "BUY", "strength": random.uniform(0.6, 0.9)}
        if recent_close < prev_close * 0.99:
            return {"signal": "SELL", "strength": random.uniform(0.6, 0.9)}
        return {"signal": "HOLD", "strength": random.uniform(0.3, 0.5)}


class _MockMarketAdapter:
    """Mock market adapter for demo mode."""

    _KRW_SYMBOLS = [
        "BTC/KRW",
        "ETH/KRW",
        "XRP/KRW",
        "SOL/KRW",
        "DOGE/KRW",
        "ADA/KRW",
        "AVAX/KRW",
        "DOT/KRW",
        "MATIC/KRW",
        "LINK/KRW",
    ]

    _USDT_SYMBOLS = [
        "BTC/USDT",
        "ETH/USDT",
        "XRP/USDT",
        "SOL/USDT",
        "DOGE/USDT",
        "ADA/USDT",
        "AVAX/USDT",
        "DOT/USDT",
        "MATIC/USDT",
        "LINK/USDT",
    ]

    def __init__(self, exchange: str):
        self.exchange = exchange
        if exchange == "binance":
            self._symbols = self._USDT_SYMBOLS
        else:
            self._symbols = self._KRW_SYMBOLS

    def get_available_symbols(self) -> List[str]:
        return self._symbols

    def get_tickers(self) -> List[str]:
        return [symbol.split("/")[0] for symbol in self._symbols]

    def get_ohlcv(
        self, symbol: str, interval: str = "1d", limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate mock OHLCV data."""
        if "USDT" in symbol:
            base_price = 70_000 if "BTC" in symbol else 3_000
        else:
            base_price = 50_000_000 if "BTC" in symbol else 1_000_000
        ohlcv = []

        for idx in range(limit):
            noise = random.uniform(-0.02, 0.02)
            price = base_price * (1 + noise)
            ohlcv.append(
                {
                    "timestamp": datetime.now().timestamp() - (limit - idx) * 86400,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "close": price,
                    "volume": random.uniform(100, 1000),
                }
            )

        return ohlcv


class SignalGenerator:
    """Generate trading signals for dashboard display."""

    def __init__(self, exchange: str = "upbit", allow_live: bool = True):
        self.exchange = exchange
        self._allow_live = allow_live
        force_demo = not allow_live
        self.strategy, self._is_mock_strategy = _load_strategy(force_demo)
        self.adapter, self._is_mock_adapter = _load_market_adapter(exchange, force_demo)
        self._is_demo_mode = self._is_mock_strategy or self._is_mock_adapter
        if hasattr(self.strategy, "set_market_data"):
            try:
                self.strategy.set_market_data(self.adapter)
            except Exception:
                pass

    @property
    def is_demo_mode(self) -> bool:
        """Check if running in demo mode (mock components)."""
        return self._is_demo_mode

    @property
    def mode_description(self) -> str:
        """Get description of current mode."""
        if not self._is_demo_mode:
            return "LIVE (Real Strategy + Real Adapter)"

        parts = []
        if self._is_mock_strategy:
            parts.append("Mock Strategy")
        if self._is_mock_adapter:
            parts.append("Mock Adapter")
        return f"DEMO ({' + '.join(parts)})"

    def get_symbols(self, limit: int = 20, show_all: bool = False) -> List[str]:
        """Get available trading symbols.

        Args:
            limit: Maximum symbols (ignored if show_all=True)
            show_all: If True, return all symbols (listing only)
        """
        try:
            if not self._is_demo_mode and self.exchange == "upbit":
                from services.dashboard.utils.upbit_symbols import get_all_upbit_symbols

                symbols = get_all_upbit_symbols()
                if symbols:
                    return symbols if show_all else symbols[:limit]

            if hasattr(self.adapter, "get_available_symbols"):
                symbols = self.adapter.get_available_symbols()
            elif hasattr(self.adapter, "get_symbols"):
                symbols = self.adapter.get_symbols()
            elif hasattr(self.adapter, "get_tickers"):
                from services.dashboard.utils.symbols import upbit_to_dashboard

                tickers = self.adapter.get_tickers()
                symbols = [upbit_to_dashboard(ticker) for ticker in tickers]
            else:
                symbols = []
            return symbols if show_all else (symbols[:limit] if symbols else [])
        except Exception as exc:
            logger.warning("Failed to get symbols: %s", type(exc).__name__)
            return ["BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW", "DOGE/KRW"][:limit]

    def _get_ohlcv_safe(self, symbol: str) -> List[Dict[str, Any]]:
        """Get OHLCV with interval/timeframe/positional fallback."""
        if not hasattr(self.adapter, "get_ohlcv"):
            return []
        try:
            return self.adapter.get_ohlcv(symbol, interval="1d", limit=100)
        except TypeError:
            pass
        try:
            return self.adapter.get_ohlcv(symbol, timeframe="1d", limit=100)
        except TypeError:
            pass
        try:
            return self.adapter.get_ohlcv(symbol, "1d", 100)
        except Exception as exc:
            logger.debug("get_ohlcv all attempts failed: %s", type(exc).__name__)
            return []

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate signal for a single symbol."""
        try:
            result = None
            if hasattr(self.strategy, "generate_signal"):
                try:
                    result = self.strategy.generate_signal(symbol)
                except TypeError:
                    logger.debug(
                        "generate_signal(symbol) failed, trying (symbol, ohlcv)"
                    )
                    ohlcv = self._get_ohlcv_safe(symbol)
                    if ohlcv:
                        try:
                            result = self.strategy.generate_signal(symbol, ohlcv)
                        except TypeError:
                            logger.debug("generate_signal(symbol, ohlcv) also failed")
                            result = None

            if result is None and hasattr(self.strategy, "calculate_signal"):
                ohlcv = self._get_ohlcv_safe(symbol)
                if ohlcv:
                    result = self.strategy.calculate_signal(ohlcv)

            if result is None:
                return {
                    "symbol": symbol,
                    "signal": "ERROR",
                    "strength": 0.0,
                    "error": "Signal generation failed",
                    "timestamp": datetime.now().isoformat(),
                    "is_mock": self._is_demo_mode,
                }

            if isinstance(result, dict):
                signal_value = result.get("signal", "HOLD")
                strength = result.get("strength", 0.0)
                timestamp = result.get("timestamp") or datetime.now().isoformat()
            else:
                signal_value = getattr(result.signal, "value", result.signal)
                strength = getattr(result, "strength", 0.0)
                ts = getattr(result, "timestamp", None)
                timestamp = ts.isoformat() if ts else datetime.now().isoformat()

            return {
                "symbol": symbol,
                "signal": signal_value,
                "strength": strength,
                "timestamp": timestamp,
                "is_mock": self._is_demo_mode,
            }
        except Exception as exc:
            logger.warning(
                "Signal generation failed for %s: %s", symbol, type(exc).__name__
            )
            return {
                "symbol": symbol,
                "signal": "ERROR",
                "strength": 0.0,
                "error": "Signal generation failed",
                "timestamp": datetime.now().isoformat(),
                "is_mock": self._is_demo_mode,
            }


@st.cache_data(ttl=60, show_spinner=False)
def get_cached_symbols(
    exchange: str, limit: int = 20, allow_live: bool = True, show_all: bool = False
) -> List[str]:
    """Get symbols with caching."""
    generator = SignalGenerator(exchange, allow_live=allow_live)
    return generator.get_symbols(limit, show_all)


@st.cache_data(ttl=60, show_spinner=False)
def get_cached_signals(
    exchange: str, symbols: tuple, allow_live: bool = True
) -> List[Dict[str, Any]]:
    """Generate signals with caching. Note: symbols must be tuple for hashing."""
    generator = SignalGenerator(exchange, allow_live=allow_live)
    return [generator.generate_signal(symbol) for symbol in symbols]


def get_signal_generator_status(
    exchange: str, allow_live: bool = True
) -> Dict[str, Any]:
    """Get status of signal generator components."""
    generator = SignalGenerator(exchange, allow_live=allow_live)
    return {
        "exchange": exchange,
        "is_demo_mode": generator.is_demo_mode,
        "mode_description": generator.mode_description,
        "mock_strategy": generator._is_mock_strategy,
        "mock_adapter": generator._is_mock_adapter,
    }
