"""Positions routes."""
from fastapi import APIRouter

from services.api.models.schemas import PositionsResponse

router = APIRouter()


@router.get("/", response_model=PositionsResponse)
async def list_positions():
    return PositionsResponse(
        success=True,
        message="Positions list",
        positions=[],
    )
