"""
Market Regime Provider

대시보드를 위한 시장 국면 및 시그널 데이터 제공
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_market_regime_detector():
    """MarketRegimeDetector 인스턴스 (캐싱)"""
    from libs.analysis.market_regime import MarketRegimeDetector

    return MarketRegimeDetector()


@lru_cache(maxsize=1)
def _get_signal_alert_service():
    """DailySignalAlertService 인스턴스 (캐싱)"""
    from services.daily_signal_alert import DailySignalAlertService

    return DailySignalAlertService()


def get_market_regime_analysis():
    """
    시장 국면 분석 결과 반환

    Returns:
        RegimeAnalysis 객체
    """
    try:
        detector = _get_market_regime_detector()
        return detector.analyze()
    except Exception as e:
        logger.error(f"[MarketRegimeProvider] Analysis failed: {e}")
        from datetime import datetime

        from libs.analysis.market_regime import (
            MarketRegime,
            MomentumState,
            RegimeAnalysis,
            VolatilityRegime,
        )

        return RegimeAnalysis(
            regime=MarketRegime.UNKNOWN,
            volatility=VolatilityRegime.NORMAL,
            momentum=MomentumState.NEUTRAL,
            price=0,
            ma20=0,
            ma50=0,
            ma200=0,
            atr_pct=0,
            tsmom_30d=0,
            tsmom_90d=0,
            trend_strength=50,
            message=f"Analysis failed: {e}",
            timestamp=datetime.now(),
        )


def get_daily_signal_summary() -> dict:
    """
    일간 시그널 요약 반환

    Returns:
        시그널 요약 딕셔너리
    """
    try:
        service = _get_signal_alert_service()
        return service.get_signal_summary()
    except Exception as e:
        logger.error(f"[MarketRegimeProvider] Signal summary failed: {e}")
        return {"error": True, "message": f"Signal calculation failed: {e}"}


def get_trading_recommendation() -> dict:
    """
    현재 시장 상황에 따른 매매 권고

    Returns:
        권고 정보 딕셔너리
    """
    try:
        detector = _get_market_regime_detector()
        analysis = detector.analyze()
        return detector.get_trading_recommendation(analysis)
    except Exception as e:
        logger.error(f"[MarketRegimeProvider] Recommendation failed: {e}")
        return {
            "action": "ERROR",
            "position_size": 0,
            "message": f"Failed to generate recommendation: {e}",
        }


def clear_cache():
    """캐시 초기화"""
    _get_market_regime_detector.cache_clear()
    _get_signal_alert_service.cache_clear()
    logger.info("[MarketRegimeProvider] Cache cleared")
