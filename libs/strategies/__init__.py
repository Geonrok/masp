"""
Strategies module.
"""
from __future__ import annotations

import importlib

from libs.strategies.base import (
    Action,
    BaseStrategy,
    Decision,
    Signal,
    StrategyContext,
    StrategyState,
    TradeSignal,
)
from libs.strategies.loader import load_strategies

__all__ = [
    "Action",
    "BaseStrategy",
    "Decision",
    "Signal",
    "TradeSignal",
    "StrategyContext",
    "StrategyState",
    "MA",
    "KAMA",
    "KAMA_series",
    "TSMOM",
    "TSMOM_signal",
    "KamaTsmomGateStrategy",
    "load_strategies",
]

_LAZY_IMPORTS = {
    "MA": "libs.strategies.indicators",
    "KAMA": "libs.strategies.indicators",
    "KAMA_series": "libs.strategies.indicators",
    "TSMOM": "libs.strategies.indicators",
    "TSMOM_signal": "libs.strategies.indicators",
    "KamaTsmomGateStrategy": "libs.strategies.kama_tsmom_gate",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
