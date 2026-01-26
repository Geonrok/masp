"""
Strategy plugin loader.
Loads strategies by ID from registry or import path.
"""

import importlib
from typing import Optional, Type

from libs.strategies.base import BaseStrategy
from libs.strategies.mock_strategy import MockStrategy, TrendFollowingMockStrategy
from libs.strategies.ma_crossover_strategy import MACrossoverStrategy
from libs.strategies.atlas_futures import ATLASFuturesStrategy


# Strategy registry - maps strategy_id to class
STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "mock_strategy": MockStrategy,
    "trend_following_mock": TrendFollowingMockStrategy,
    "ma_crossover_v1": MACrossoverStrategy,
    "atlas_futures_p04": ATLASFuturesStrategy,
}

# Strategy metadata registry (not necessarily loadable)
AVAILABLE_STRATEGIES: list[dict] = []

# ATLAS-Futures registration (fully integrated)
AVAILABLE_STRATEGIES.append({
    "strategy_id": "atlas_futures_p04",
    "id": "atlas_futures_p04",
    "name": "P0-4 Squeeze-Surge",
    "version": "v2.6.2-r1",
    "description": "ATLAS-Futures volatility squeeze + surge strategy (6x 3AI PASS)",
    "module": "libs.strategies.atlas_futures",
    "class_name": "ATLASFuturesStrategy",
    "config_class": "ATLASFuturesConfig",
    "markets": ["futures"],
    "exchanges": ["binance_futures"],
    "status": "phase_4_ready",
})

# KAMA-TSMOM-Gate registration (metadata only)
AVAILABLE_STRATEGIES.append({
    "strategy_id": "kama_tsmom_gate",
    "id": "kama_tsmom_gate",
    "name": "KAMA-TSMOM-Gate",
    "version": "v1.0",
    "description": "KAMA-TSMOM-Gate strategy (BTC gate + KAMA/TSMOM)",
    "module": "libs.strategies.kama_tsmom_gate",
    "class_name": "KamaTsmomGateStrategy",
    "markets": ["spot"],
    "exchanges": ["upbit_spot", "paper"],
    "status": "phase_3a_ready",
})


def register_strategy(strategy_class: type[BaseStrategy]) -> None:
    """
    Register a strategy class in the registry.
    
    Args:
        strategy_class: Strategy class to register
    """
    STRATEGY_REGISTRY[strategy_class.strategy_id] = strategy_class


def load_strategy_class(strategy_id: str) -> Optional[Type[BaseStrategy]]:
    """Load strategy class dynamically."""
    if strategy_id in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[strategy_id]

    for entry in AVAILABLE_STRATEGIES:
        if entry.get("strategy_id") == strategy_id:
            module_path = entry.get("module")
            class_name = entry.get("class_name")

            if not module_path or not class_name:
                print(f"[Loader] Warning: Missing module/class_name for {strategy_id}")
                return None

            try:
                module = importlib.import_module(module_path)
                strategy_class = getattr(module, class_name)
                STRATEGY_REGISTRY[strategy_id] = strategy_class
                print(f"[Loader] Dynamically loaded: {strategy_id}")
                return strategy_class
            except Exception as exc:
                print(f"[Loader] Failed to load {strategy_id}: {exc}")
                return None

    return None


def get_strategy(strategy_id: str) -> Optional[BaseStrategy]:
    """
    Get a strategy instance by ID.
    
    Args:
        strategy_id: Strategy identifier
    
    Returns:
        Strategy instance or None if not found
    """
    strategy_class = load_strategy_class(strategy_id)
    if strategy_class:
        return strategy_class()
    return None


def load_strategies(strategy_ids: list[str]) -> list[BaseStrategy]:
    """
    Load multiple strategies by their IDs.
    
    Args:
        strategy_ids: List of strategy identifiers
    
    Returns:
        List of strategy instances (skips unknown IDs with warning)
    """
    strategies = []
    
    for strategy_id in strategy_ids:
        strategy = get_strategy(strategy_id)
        if strategy:
            strategies.append(strategy)
            print(f"[Loader] Loaded strategy: {strategy}")
        else:
            print(f"[Loader] Warning: Unknown strategy ID '{strategy_id}', skipping")
    
    return strategies


def _get_attr(cls, *names, default: str = "unknown") -> str:
    """Get attribute from class, trying multiple names (case-insensitive fallback)."""
    for name in names:
        if hasattr(cls, name):
            return getattr(cls, name)
    return default


def list_available_strategies() -> list[dict]:
    """
    List all available strategies.

    Returns:
        List of strategy metadata dicts
    """
    result: dict[str, dict] = {}
    for strategy_id, strategy_class in STRATEGY_REGISTRY.items():
        result[strategy_id] = {
            "strategy_id": strategy_id,
            "id": strategy_id,
            "name": _get_attr(strategy_class, "name", "NAME"),
            "version": _get_attr(strategy_class, "version", "VERSION"),
            "description": _get_attr(strategy_class, "description", "DESCRIPTION"),
        }
    for entry in AVAILABLE_STRATEGIES:
        strategy_id = entry.get("strategy_id")
        if not strategy_id:
            continue
        if strategy_id not in result:
            merged = dict(entry)
            if "id" not in merged:
                merged["id"] = strategy_id
            result[strategy_id] = merged
        else:
            merged = dict(result[strategy_id])
            merged.update(entry)
            if "id" not in merged:
                merged["id"] = strategy_id
            result[strategy_id] = merged
    return list(result.values())
