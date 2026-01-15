"""Pydantic schemas for API responses and requests."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class ExchangeType(str, Enum):
    upbit = "upbit"
    bithumb = "bithumb"
    binance_spot = "binance_spot"
    binance_futures = "binance_futures"
    paper = "paper"


class APIKeyRequest(BaseModel):
    exchange: ExchangeType
    api_key: str
    secret_key: str


class APIKeyResponse(BaseResponse):
    exchange: ExchangeType
    masked_key: str
    created_at: datetime


class APIKeyListResponse(BaseResponse):
    keys: List[APIKeyResponse] = Field(default_factory=list)


class SystemStatus(BaseResponse):
    version: str
    uptime_seconds: float
    exchanges: List[str]
    active_strategies: int


class KillSwitchRequest(BaseModel):
    confirm: bool


class KillSwitchResponse(BaseResponse):
    positions_closed: int
    strategies_stopped: int


class StrategyInfo(BaseModel):
    strategy_id: str
    name: str
    version: str
    description: str
    active: bool = False


class StrategyListResponse(BaseResponse):
    strategies: List[StrategyInfo] = Field(default_factory=list)


class Position(BaseModel):
    symbol: str
    side: str  # LONG/SHORT
    quantity: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percent: float


class Trade(BaseModel):
    id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    timestamp: datetime
    strategy_id: str


class PositionsResponse(BaseResponse):
    positions: List[Position] = Field(default_factory=list)


class TradesResponse(BaseResponse):
    trades: List[Trade] = Field(default_factory=list)
