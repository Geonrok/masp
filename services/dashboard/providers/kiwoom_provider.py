"""Kiwoom Sector Rotation strategy status provider.

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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# --- Constants ---
_CONFIG_PATH = Path("config/schedule_config.json")
_EXCHANGE_KEY = "kiwoom_spot"
_INITIAL_BALANCE = 1_000_000.0  # KRW

# Stock code to name mapping
_STOCK_NAMES = {
    "000660": "SK하이닉스",
    "005930": "삼성전자",
    "003670": "포스코퓨처엠",
    "042700": "한미반도체",
    "006400": "삼성SDI",
}


def _kiwoom_trade_log_dir() -> str:
    """Return the trade log directory for kiwoom_spot."""
    base = os.environ.get("TRADE_LOG_DIR", "logs/trades")
    parent = Path(base).parent
    return str(parent / "kiwoom_spot_trades" / "trades")


# --- Data classes ---


@dataclass
class KiwoomConfig:
    """Strategy configuration from schedule_config.json."""

    enabled: bool = False
    strategy_name: str = "sector_rotation_m"
    symbols: List[str] = field(default_factory=list)
    position_size_krw: float = 100000
    schedule_day: str = "last_trading_day"
    schedule_hour: int = 15
    schedule_minute: int = 20
    schedule_timezone: str = "Asia/Seoul"
    mode: str = "paper"
    comment: str = ""


@dataclass
class KiwoomAccount:
    """Account summary derived from trade logs."""

    initial_balance: float = _INITIAL_BALANCE
    estimated_balance: float = _INITIAL_BALANCE
    total_pnl: float = 0.0
    total_fees: float = 0.0
    total_volume: float = 0.0
    pnl_percent: float = 0.0


@dataclass
class KiwoomTradeStats:
    """Trade statistics."""

    trades_today: int = 0
    trades_total: int = 0
    buy_count: int = 0
    sell_count: int = 0
    unique_symbols: int = 0
    first_trade_date: str = ""
    last_trade_date: str = ""
    rebalance_count: int = 0


@dataclass
class KiwoomPosition:
    """Inferred active position from net trades."""

    symbol: str = ""
    symbol_name: str = ""
    net_quantity: float = 0.0
    avg_entry_price: float = 0.0
    notional_value: float = 0.0
    side: str = "FLAT"
    trade_count: int = 0


@dataclass
class KiwoomStatusData:
    """Complete data bundle for the component."""

    config: KiwoomConfig = field(default_factory=KiwoomConfig)
    account: KiwoomAccount = field(default_factory=KiwoomAccount)
    stats: KiwoomTradeStats = field(default_factory=KiwoomTradeStats)
    positions: List[KiwoomPosition] = field(default_factory=list)
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


def _get_stock_name(symbol: str) -> str:
    """Get stock name from symbol code."""
    return _STOCK_NAMES.get(symbol, symbol)


# --- Core functions ---


def _load_strategy_config() -> KiwoomConfig:
    """Read kiwoom_spot config from schedule_config.json."""
    config = KiwoomConfig()
    try:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                data = json.load(f)
            kw = data.get("exchanges", {}).get(_EXCHANGE_KEY, {})
            if kw:
                config.enabled = kw.get("enabled", False)
                config.strategy_name = kw.get("strategy", config.strategy_name)
                config.symbols = kw.get("symbols", config.symbols)
                config.position_size_krw = kw.get(
                    "position_size_krw", config.position_size_krw
                )
                sched = kw.get("schedule", {})
                config.schedule_day = sched.get("day", config.schedule_day)
                config.schedule_hour = sched.get("hour", config.schedule_hour)
                config.schedule_minute = sched.get("minute", config.schedule_minute)
                config.schedule_timezone = sched.get(
                    "timezone", config.schedule_timezone
                )
                config.comment = kw.get("_comment", "")
    except Exception as e:
        logger.warning("[KiwoomProvider] Config load failed: %s", e)

    config.mode = "live" if os.getenv("MASP_ENABLE_LIVE_TRADING") == "1" else "paper"
    return config


def _get_trade_logger():
    """Get TradeLogger pointed at the Kiwoom strategy trade log directory."""
    try:
        from libs.adapters.trade_logger import TradeLogger

        return TradeLogger(log_dir=_kiwoom_trade_log_dir())
    except Exception as e:
        logger.debug("[KiwoomProvider] TradeLogger init failed: %s", e)
        return None


def _load_all_trades(lookback_days: int = 180) -> List[Dict[str, Any]]:
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

    trades.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
    return trades


def _compute_account_summary(trades: List[Dict[str, Any]]) -> KiwoomAccount:
    """Compute estimated account balance from trade history."""
    account = KiwoomAccount()
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


def _compute_trade_stats(trades: List[Dict[str, Any]]) -> KiwoomTradeStats:
    """Compute trade statistics from all loaded trades."""
    stats = KiwoomTradeStats()
    stats.trades_total = len(trades)

    if not trades:
        return stats

    symbols = set()
    today_str = date.today().isoformat()
    rebalance_dates = set()

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
        if isinstance(ts, str):
            if ts.startswith(today_str):
                stats.trades_today += 1
            # Count unique dates as rebalance events
            rebalance_dates.add(ts[:10])

    stats.unique_symbols = len(symbols)
    stats.rebalance_count = len(rebalance_dates)

    if trades:
        stats.last_trade_date = trades[0].get("timestamp", "")[:10]
        stats.first_trade_date = trades[-1].get("timestamp", "")[:10]

    return stats


def _infer_positions(trades: List[Dict[str, Any]]) -> List[KiwoomPosition]:
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
            KiwoomPosition(
                symbol=symbol,
                symbol_name=_get_stock_name(symbol),
                net_quantity=net_qty,
                avg_entry_price=avg_entry,
                notional_value=notional,
                side=side,
                trade_count=int(d["count"]),
            )
        )

    positions.sort(key=lambda p: p.notional_value, reverse=True)
    return positions


def _get_demo_status() -> KiwoomStatusData:
    """Return deterministic demo data for when no real trades exist."""
    config = _load_strategy_config()

    demo_trades = [
        {
            "timestamp": "2026-02-05T15:20:05",
            "exchange": "paper",
            "symbol": "005930",
            "side": "BUY",
            "quantity": "10",
            "price": "71500",
            "fee": "358",
            "pnl": "0.0",
            "status": "FILLED",
        },
        {
            "timestamp": "2026-02-05T15:20:06",
            "exchange": "paper",
            "symbol": "000660",
            "side": "BUY",
            "quantity": "5",
            "price": "185000",
            "fee": "463",
            "pnl": "0.0",
            "status": "FILLED",
        },
    ]

    return KiwoomStatusData(
        config=config,
        account=KiwoomAccount(
            initial_balance=_INITIAL_BALANCE,
            estimated_balance=_INITIAL_BALANCE - 821,
            total_pnl=0.0,
            total_fees=821,
            total_volume=1640000,
            pnl_percent=-0.08,
        ),
        stats=KiwoomTradeStats(
            trades_today=2,
            trades_total=2,
            buy_count=2,
            sell_count=0,
            unique_symbols=2,
            first_trade_date="2026-02-05",
            last_trade_date="2026-02-05",
            rebalance_count=1,
        ),
        positions=[
            KiwoomPosition(
                symbol="000660",
                symbol_name="SK하이닉스",
                net_quantity=5,
                avg_entry_price=185000,
                notional_value=925000,
                side="LONG",
                trade_count=1,
            ),
            KiwoomPosition(
                symbol="005930",
                symbol_name="삼성전자",
                net_quantity=10,
                avg_entry_price=71500,
                notional_value=715000,
                side="LONG",
                trade_count=1,
            ),
        ],
        recent_trades=demo_trades,
        data_available=False,
        last_updated=datetime.now(),
    )


_cache: Optional[KiwoomStatusData] = None
_cache_ts: float = 0.0
_CACHE_TTL = 60.0  # seconds


def get_kiwoom_status() -> KiwoomStatusData:
    """Main entry point: return complete Kiwoom strategy status data.

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


def _fetch_status() -> KiwoomStatusData:
    """Fetch and compute status data (uncached)."""
    try:
        config = _load_strategy_config()
        trades = _load_all_trades()

        if not trades:
            demo = _get_demo_status()
            demo.config = config
            return demo

        return KiwoomStatusData(
            config=config,
            account=_compute_account_summary(trades),
            stats=_compute_trade_stats(trades),
            positions=_infer_positions(trades),
            recent_trades=trades[:20],
            data_available=True,
            last_updated=datetime.now(),
        )
    except Exception as e:
        logger.error("[KiwoomProvider] Failed: %s", e)
        return _get_demo_status()
