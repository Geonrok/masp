"""Trades routes."""
from fastapi import APIRouter

from services.api.models.schemas import TradesResponse

router = APIRouter()


@router.get("/", response_model=TradesResponse)
async def list_trades():
    return TradesResponse(
        success=True,
        message="Trades list",
        trades=[],
    )
