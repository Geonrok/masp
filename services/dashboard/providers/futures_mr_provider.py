"""Binance Futures MR Strategy status provider.

Reads trade CSV logs and schedule_config.json from shared Docker volumes
to provide strategy status data for the dashboard component.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# --- Constants ---
_CONFIG_PATH = Path("config/schedule_config.json")
_EXCHANGE_KEY = "binance_futures_mr"
_INITIAL_BALANCE = 10_000.0  # USDT


def _mr_trade_log_dir() -> str:
    """Return the trade log directory for binance_futures_mr.

    In Docker, the scheduler writes to /app/logs/binance_futures_mr_trades/trades/.
    The dashboard shares the same ./logs volume.
    """
    base = os.environ.get("TRADE_LOG_DIR", "logs/trades")
    # The strategy runner creates: logs/{exchange}_trades/trades/
    # So the MR log dir is a sibling of the default TRADE_LOG_DIR base.
    parent = Path(base).parent  # logs/ (strip trailing /trades)
    return str(parent / "binance_futures_mr_trades" / "trades")


# --- Data classes ---


@dataclass
class FuturesMRConfig:
    """Strategy configuration from schedule_config.json."""

    enabled: bool = False
    strategy_name: str = "mr_adaptive_aggressive"
    symbols: str = "ALL_USDT_PERP"
    position_size_usdt: float = 50
    leverage: int = 2
    max_positions: int = 30
    schedule_hour: int = 0
    schedule_minute: int = 5
    schedule_timezone: str = "UTC"
    schedule_jitter: int = 60
    mode: str = "paper"
    comment: str = ""


@dataclass
class FuturesMRAccount:
    """Account summary derived from trade logs."""

    initial_balance: float = _INITIAL_BALANCE
    estimated_balance: float = _INITIAL_BALANCE
    total_pnl: float = 0.0
    total_fees: float = 0.0
    total_volume: float = 0.0
    pnl_percent: float = 0.0


@dataclass
class FuturesMRTradeStats:
    """Trade statistics."""

    trades_today: int = 0
    trades_total: int = 0
    buy_count: int = 0
    sell_count: int = 0
    unique_symbols: int = 0
    first_trade_date: str = ""
    last_trade_date: str = ""


@dataclass
class FuturesMRPosition:
    """Inferred active position from net trades."""

    symbol: str = ""
    net_quantity: float = 0.0
    avg_entry_price: float = 0.0
    notional_value: float = 0.0
    side: str = "FLAT"
    trade_count: int = 0


@dataclass
class FuturesMRStatusData:
    """Complete data bundle for the component."""

    config: FuturesMRConfig = field(default_factory=FuturesMRConfig)
    account: FuturesMRAccount = field(default_factory=FuturesMRAccount)
    stats: FuturesMRTradeStats = field(default_factory=FuturesMRTradeStats)
    positions: List[FuturesMRPosition] = field(default_factory=list)
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)
    data_available: bool = False
    last_updated: datetime = field(default_factory=datetime.now)


# --- Helpers ---


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def _norm_side(value: Any) -> str:
    """Normalize side to BUY/SELL."""
    s = ("" if value is None else str(value)).strip().upper()
    if s in ("B", "BUY", "LONG"):
        return "BUY"
    if s in ("S", "SELL", "SHORT"):
        return "SELL"
    return s


# --- Core functions ---


def _load_strategy_config() -> FuturesMRConfig:
    """Read binance_futures_mr config from schedule_config.json."""
    config = FuturesMRConfig()
    try:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                data = json.load(f)
            mr = data.get("exchanges", {}).get(_EXCHANGE_KEY, {})
            if mr:
                config.enabled = mr.get("enabled", False)
                config.strategy_name = mr.get("strategy", config.strategy_name)
                config.symbols = mr.get("symbols", config.symbols)
                config.position_size_usdt = mr.get(
                    "position_size_usdt", config.position_size_usdt
                )
                config.leverage = mr.get("leverage", config.leverage)
                config.max_positions = mr.get("max_positions", config.max_positions)
                sched = mr.get("schedule", {})
                config.schedule_hour = sched.get("hour", config.schedule_hour)
                config.schedule_minute = sched.get("minute", config.schedule_minute)
                config.schedule_timezone = sched.get(
                    "timezone", config.schedule_timezone
                )
                config.schedule_jitter = sched.get("jitter", config.schedule_jitter)
                config.comment = mr.get("_comment", "")
    except Exception as e:
        logger.warning("[FuturesMRProvider] Config load failed: %s", e)

    # Determine mode
    config.mode = "live" if os.getenv("MASP_ENABLE_LIVE_TRADING") == "1" else "paper"
    return config


def _get_trade_logger():
    """Get TradeLogger pointed at the MR strategy trade log directory."""
    try:
        from libs.adapters.trade_logger import TradeLogger

        return TradeLogger(log_dir=_mr_trade_log_dir())
    except Exception as e:
        logger.debug("[FuturesMRProvider] TradeLogger init failed: %s", e)
        return None


def _load_all_trades(lookback_days: int = 90) -> List[Dict[str, Any]]:
    """Load all trades from CSV files within lookback window."""
    trade_logger = _get_trade_logger()
    if trade_logger is None:
        return []

    trades: List[Dict[str, Any]] = []
    end = date.today()
    start = end - timedelta(days=lookback_days)
    current = start

    while current <= end:
        try:
            day_trades = trade_logger.get_trades(current)
            trades.extend(day_trades)
        except Exception:
            pass
        current += timedelta(days=1)

    # Sort by timestamp descending
    trades.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
    return trades


def _compute_account_summary(trades: List[Dict[str, Any]]) -> FuturesMRAccount:
    """Compute estimated account balance from trade history."""
    account = FuturesMRAccount()
    total_pnl = 0.0
    total_fees = 0.0
    total_volume = 0.0

    for t in trades:
        total_pnl += _safe_float(t.get("pnl", 0))
        total_fees += _safe_float(t.get("fee", 0))
        qty = _safe_float(t.get("quantity", 0))
        price = _safe_float(t.get("price", 0))
        total_volume += qty * price

    account.total_pnl = total_pnl
    account.total_fees = total_fees
    account.total_volume = total_volume
    account.estimated_balance = _INITIAL_BALANCE + total_pnl - total_fees
    if _INITIAL_BALANCE > 0:
        account.pnl_percent = ((total_pnl - total_fees) / _INITIAL_BALANCE) * 100

    return account


def _compute_trade_stats(trades: List[Dict[str, Any]]) -> FuturesMRTradeStats:
    """Compute trade statistics from all loaded trades."""
    stats = FuturesMRTradeStats()
    stats.trades_total = len(trades)

    if not trades:
        return stats

    symbols = set()
    today_str = date.today().isoformat()

    for t in trades:
        side = _norm_side(t.get("side", ""))
        if side == "BUY":
            stats.buy_count += 1
        elif side == "SELL":
            stats.sell_count += 1

        sym = t.get("symbol", "")
        if sym:
            symbols.add(sym)

        ts = t.get("timestamp", "")
        if isinstance(ts, str) and ts.startswith(today_str):
            stats.trades_today += 1

    stats.unique_symbols = len(symbols)

    # First/last trade dates (trades are sorted descending)
    if trades:
        stats.last_trade_date = trades[0].get("timestamp", "")[:10]
        stats.first_trade_date = trades[-1].get("timestamp", "")[:10]

    return stats


def _infer_positions(trades: List[Dict[str, Any]]) -> List[FuturesMRPosition]:
    """Infer active positions from net BUY-SELL per symbol."""
    symbol_data: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "buy_qty": 0.0,
            "sell_qty": 0.0,
            "buy_cost": 0.0,
            "count": 0,
        }
    )

    for t in trades:
        symbol = t.get("symbol", "")
        if not symbol:
            continue
        side = _norm_side(t.get("side", ""))
        qty = _safe_float(t.get("quantity", 0))
        price = _safe_float(t.get("price", 0))

        d = symbol_data[symbol]
        d["count"] += 1
        if side == "BUY":
            d["buy_qty"] += qty
            d["buy_cost"] += qty * price
        elif side == "SELL":
            d["sell_qty"] += qty

    positions = []
    for symbol, d in symbol_data.items():
        net_qty = d["buy_qty"] - d["sell_qty"]
        if abs(net_qty) < 1e-8:
            continue

        avg_entry = d["buy_cost"] / d["buy_qty"] if d["buy_qty"] > 0 else 0.0
        side = "LONG" if net_qty > 0 else "SHORT"
        notional = abs(net_qty) * avg_entry

        positions.append(
            FuturesMRPosition(
                symbol=symbol,
                net_quantity=net_qty,
                avg_entry_price=avg_entry,
                notional_value=notional,
                side=side,
                trade_count=int(d["count"]),
            )
        )

    positions.sort(key=lambda p: p.notional_value, reverse=True)
    return positions


def _get_demo_status() -> FuturesMRStatusData:
    """Return deterministic demo data for when no real trades exist."""
    config = _load_strategy_config()

    demo_trades = [
        {
            "timestamp": "2026-02-05T00:05:12",
            "exchange": "paper",
            "symbol": "BTC/USDT:PERP",
            "side": "BUY",
            "quantity": "0.0014",
            "price": "71382.62",
            "fee": "0.05",
            "pnl": "0.0",
            "status": "FILLED",
        },
        {
            "timestamp": "2026-02-05T00:05:13",
            "exchange": "paper",
            "symbol": "ETH/USDT:PERP",
            "side": "BUY",
            "quantity": "0.02",
            "price": "2650.50",
            "fee": "0.03",
            "pnl": "0.0",
            "status": "FILLED",
        },
        {
            "timestamp": "2026-02-05T00:05:14",
            "exchange": "paper",
            "symbol": "SOL/USDT:PERP",
            "side": "BUY",
            "quantity": "0.5",
            "price": "98.40",
            "fee": "0.02",
            "pnl": "0.0",
            "status": "FILLED",
        },
    ]

    return FuturesMRStatusData(
        config=config,
        account=FuturesMRAccount(
            initial_balance=_INITIAL_BALANCE,
            estimated_balance=_INITIAL_BALANCE - 0.10,
            total_pnl=0.0,
            total_fees=0.10,
            total_volume=152.85,
            pnl_percent=-0.001,
        ),
        stats=FuturesMRTradeStats(
            trades_today=3,
            trades_total=3,
            buy_count=3,
            sell_count=0,
            unique_symbols=3,
            first_trade_date="2026-02-05",
            last_trade_date="2026-02-05",
        ),
        positions=[
            FuturesMRPosition(
                symbol="BTC/USDT:PERP",
                net_quantity=0.0014,
                avg_entry_price=71382.62,
                notional_value=99.94,
                side="LONG",
                trade_count=1,
            ),
            FuturesMRPosition(
                symbol="ETH/USDT:PERP",
                net_quantity=0.02,
                avg_entry_price=2650.50,
                notional_value=53.01,
                side="LONG",
                trade_count=1,
            ),
            FuturesMRPosition(
                symbol="SOL/USDT:PERP",
                net_quantity=0.5,
                avg_entry_price=98.40,
                notional_value=49.20,
                side="LONG",
                trade_count=1,
            ),
        ],
        recent_trades=demo_trades,
        data_available=False,
        last_updated=datetime.now(),
    )


_cache: Optional[FuturesMRStatusData] = None
_cache_ts: float = 0.0
_CACHE_TTL = 60.0  # seconds


def get_futures_mr_status() -> FuturesMRStatusData:
    """Main entry point: return complete MR strategy status data.

    Returns real data if trade logs exist, otherwise demo data.
    Results are cached for 60 seconds to avoid repeated CSV I/O.
    """
    import time

    global _cache, _cache_ts  # noqa: PLW0603
    now = time.monotonic()
    if _cache is not None and (now - _cache_ts) < _CACHE_TTL:
        return _cache
    result = _fetch_status()
    _cache = result
    _cache_ts = now
    return result


def _fetch_status() -> FuturesMRStatusData:
    """Fetch and compute status data (uncached)."""
    try:
        config = _load_strategy_config()
        trades = _load_all_trades()

        if not trades:
            demo = _get_demo_status()
            demo.config = config
            return demo

        return FuturesMRStatusData(
            config=config,
            account=_compute_account_summary(trades),
            stats=_compute_trade_stats(trades),
            positions=_infer_positions(trades),
            recent_trades=trades[:20],
            data_available=True,
            last_updated=datetime.now(),
        )
    except Exception as e:
        logger.error("[FuturesMRProvider] Failed: %s", e)
        return _get_demo_status()
