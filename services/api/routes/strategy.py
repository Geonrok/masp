"""Strategy management routes."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from libs.strategies.loader import list_available_strategies
from services.api.models.schemas import BaseResponse, StrategyInfo, StrategyListResponse

router = APIRouter()


class StrategyActionRequest(BaseModel):
    strategy_id: str


class StrategyManager:
    """Manages active strategies."""

    def __init__(self) -> None:
        self.active_strategies: set[str] = set()

    def list_strategies(self) -> List[StrategyInfo]:
        available = list_available_strategies()
        result: List[StrategyInfo] = []
        for item in available:
            strategy_id = item.get("strategy_id", "")
            result.append(
                StrategyInfo(
                    strategy_id=strategy_id,
                    name=item.get("name", ""),
                    version=item.get("version", ""),
                    description=item.get("description", ""),
                    active=strategy_id in self.active_strategies,
                )
            )
        return result

    def start(self, strategy_id: str) -> None:
        known_ids = {item.get("strategy_id") for item in list_available_strategies()}
        if strategy_id not in known_ids:
            raise ValueError(f"Unknown strategy_id: {strategy_id}")
        self.active_strategies.add(strategy_id)

    def stop(self, strategy_id: str) -> None:
        self.active_strategies.discard(strategy_id)

    def stop_all(self) -> int:
        count = len(self.active_strategies)
        self.active_strategies.clear()
        return count


# Legacy global instance (for backward compatibility with imports and tests)
strategy_manager = StrategyManager()


# Dependency injection: Get StrategyManager from app state (with fallback)
def get_strategy_manager(request: Request) -> StrategyManager:
    """Get StrategyManager from app state with fallback to module-level instance."""
    if hasattr(request.app.state, "strategy_manager"):
        return request.app.state.strategy_manager
    # Fallback for tests that don't use lifespan
    return strategy_manager


@router.get("/list", response_model=StrategyListResponse)
async def list_strategies(manager: StrategyManager = Depends(get_strategy_manager)):
    strategies = manager.list_strategies()
    return StrategyListResponse(
        success=True,
        message="Strategy list",
        strategies=strategies,
    )


@router.post("/start", response_model=BaseResponse)
async def start_strategy(
    request: StrategyActionRequest,
    manager: StrategyManager = Depends(get_strategy_manager),
):
    try:
        manager.start(request.strategy_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return BaseResponse(
        success=True, message=f"Strategy started: {request.strategy_id}"
    )


@router.post("/stop", response_model=BaseResponse)
async def stop_strategy(
    request: StrategyActionRequest,
    manager: StrategyManager = Depends(get_strategy_manager),
):
    manager.stop(request.strategy_id)
    return BaseResponse(
        success=True, message=f"Strategy stopped: {request.strategy_id}"
    )
