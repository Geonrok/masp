"""Health routes."""

from datetime import datetime

from fastapi import APIRouter

from services.api.models.schemas import BaseResponse

router = APIRouter()


@router.get("/", response_model=BaseResponse)
async def health_check():
    return BaseResponse(
        success=True,
        message="healthy",
        timestamp=datetime.now(),
    )
