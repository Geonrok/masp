"""Symbol format utilities for MASP Dashboard."""

from __future__ import annotations

from typing import List


def upbit_to_dashboard(symbol: str) -> str:
    """Convert Upbit format to Dashboard format.

    KRW-BTC -> BTC/KRW
    """
    if symbol.startswith("KRW-"):
        return symbol[4:] + "/KRW"
    if "/" not in symbol:
        return symbol + "/KRW"
    return symbol


def dashboard_to_upbit(symbol: str) -> str:
    """Convert Dashboard format to Upbit format.

    BTC/KRW -> KRW-BTC
    """
    if symbol.startswith("KRW-"):
        return symbol
    if symbol.endswith("/KRW"):
        return "KRW-" + symbol[:-4]
    if "/" not in symbol:
        return "KRW-" + symbol
    return symbol


def binance_to_dashboard(symbol: str) -> str:
    """Convert Binance format to Dashboard format.

    BTCUSDT -> BTC/USDT
    """
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        return symbol[:-4] + "/USDT"
    return symbol


def dashboard_to_binance(symbol: str) -> str:
    """Convert Dashboard format to Binance format.

    BTC/USDT -> BTCUSDT
    """
    return symbol.replace("/", "")


def convert_symbols_to_dashboard(symbols: List[str]) -> List[str]:
    """Convert list of Upbit symbols to Dashboard format."""
    return [upbit_to_dashboard(symbol) for symbol in symbols]
