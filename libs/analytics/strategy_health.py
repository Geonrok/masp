"""
Strategy Health Monitor - 전략 성과 모니터링 및 리밸런싱 트리거

업계 표준 기반:
- Sharpe Floor: 0.5 (30일 롤링)
- MDD Ceiling: 15%
- Consecutive Loss: 5회 경고, 8회 중단
- Daily Loss Halt: 3%
"""

import logging
import statistics
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """전략 건강 상태"""

    HEALTHY = "HEALTHY"  # 정상 운영
    WARNING = "WARNING"  # 파라미터 검토 권장
    CRITICAL = "CRITICAL"  # 거래 중단 필요
    HALTED = "HALTED"  # Kill-Switch 활성


@dataclass
class HealthCheckResult:
    """건강 체크 결과"""

    status: HealthStatus
    sharpe_30d: Optional[float]
    mdd_current: float
    consecutive_losses: int
    daily_pnl_pct: float
    triggers: List[str]
    recommendation: str
    checked_at: datetime


class StrategyHealthMonitor:
    """
    전략 건강 모니터링 및 리밸런싱 트리거

    Methods:
        add_trade(trade): 거래 기록 추가
        add_daily_pnl(pnl_pct): 일일 PnL 기록
        check_health(): 건강 상태 종합 평가
        get_summary(): 현재 상태 요약
    """

    # 임계값 설정 (업계 표준)
    SHARPE_WARNING = 0.5
    SHARPE_CRITICAL = 0.0
    MDD_WARNING = 0.10
    MDD_CRITICAL = 0.15
    CONSECUTIVE_LOSS_WARNING = 5
    CONSECUTIVE_LOSS_CRITICAL = 8
    DAILY_LOSS_HALT = 0.03
    SHARPE_LOOKBACK_DAYS = 30
    MIN_TRADES_FOR_SHARPE = 10

    def __init__(self, config=None):
        """
        초기화

        Args:
            config: Config 인스턴스 (Kill-Switch 연동용)
        """
        self.config = config
        self._trade_history: List[Dict[str, Any]] = []
        self._daily_pnl: List[Dict[str, Any]] = []
        self._equity_history: List[float] = [1.0]  # 초기 자본 1.0 (100%)

        logger.info("[HealthMonitor] Strategy Health Monitor initialized")

    def add_trade(self, trade: dict):
        """
        거래 기록 추가

        Args:
            trade: 거래 정보 dict (pnl, pnl_pct 필수)
                   예: {"pnl": 10000, "pnl_pct": 0.01, "timestamp": datetime.now()}
        """
        if "pnl" not in trade or "pnl_pct" not in trade:
            logger.error("[HealthMonitor] Trade must have 'pnl' and 'pnl_pct'")
            return

        if "timestamp" not in trade:
            trade["timestamp"] = datetime.now()

        self._trade_history.append(trade)

        # Equity 업데이트 (복리)
        last_equity = self._equity_history[-1]
        new_equity = last_equity * (1 + trade["pnl_pct"])
        self._equity_history.append(new_equity)

        logger.debug(
            f"[HealthMonitor] Trade added: PnL={trade['pnl']:.0f}, "
            f"PnL%={trade['pnl_pct']*100:.2f}%, Equity={new_equity:.4f}"
        )

    def add_daily_pnl(self, pnl_pct: float, date: datetime = None):
        """
        일일 PnL 기록 + equity curve 업데이트

        Args:
            pnl_pct: 일일 PnL (예: 0.02 = 2%)
            date: 날짜 (기본: 오늘)
        """
        if date is None:
            date = datetime.now().date()

        self._daily_pnl.append({"date": date, "pnl_pct": pnl_pct})

        # [FIX] Equity curve 업데이트 (MDD 계산용)
        last_equity = self._equity_history[-1]
        new_equity = last_equity * (1 + pnl_pct)
        self._equity_history.append(new_equity)

        logger.debug(
            f"[HealthMonitor] Daily PnL added: {pnl_pct*100:.2f}% on {date}, "
            f"Equity: {new_equity:.4f}"
        )

    def check_health(self) -> HealthCheckResult:
        """
        건강 상태 종합 평가

        체크 순서:
        1. Kill-Switch (최우선)
        2. 일일 손실 > 3%
        3. MDD > 15%
        4. 연속 손실 >= 8회
        5. Sharpe < 0.5 (30일)

        Returns:
            HealthCheckResult
        """
        triggers = []
        status = HealthStatus.HEALTHY

        # [1] Kill-Switch 체크 (최우선)
        if self.config and hasattr(self.config, "is_kill_switch_active"):
            if self.config.is_kill_switch_active():
                triggers.append("KILL_SWITCH_ACTIVE")
                status = HealthStatus.HALTED
                logger.error("[HealthMonitor] Kill-Switch is ACTIVE!")

                return HealthCheckResult(
                    status=status,
                    sharpe_30d=None,
                    mdd_current=self._calculate_current_mdd(),
                    consecutive_losses=self._count_consecutive_losses(),
                    daily_pnl_pct=self._get_today_pnl(),
                    triggers=triggers,
                    recommendation=self._generate_recommendation(status, triggers),
                    checked_at=datetime.now(),
                )

        # [2] 일일 손실 > 3%
        today_pnl = self._get_today_pnl()
        if today_pnl < -self.DAILY_LOSS_HALT:
            triggers.append(f"DAILY_LOSS_{abs(today_pnl)*100:.1f}%")
            status = HealthStatus.CRITICAL
            logger.error(f"[HealthMonitor] Daily loss exceeded: {today_pnl*100:.2f}%")

        # [3] MDD > 15%
        mdd = self._calculate_current_mdd()
        if mdd > self.MDD_CRITICAL:
            triggers.append(f"MDD_CRITICAL_{mdd*100:.1f}%")
            if status != HealthStatus.CRITICAL:
                status = HealthStatus.CRITICAL
            logger.error(f"[HealthMonitor] MDD critical: {mdd*100:.2f}%")
        elif mdd > self.MDD_WARNING:
            triggers.append(f"MDD_WARNING_{mdd*100:.1f}%")
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            logger.warning(f"[HealthMonitor] MDD warning: {mdd*100:.2f}%")

        # [4] 연속 손실 >= 8회
        consecutive_losses = self._count_consecutive_losses()
        if consecutive_losses >= self.CONSECUTIVE_LOSS_CRITICAL:
            triggers.append(f"CONSECUTIVE_LOSS_{consecutive_losses}")
            if status != HealthStatus.CRITICAL:
                status = HealthStatus.CRITICAL
            logger.error(
                f"[HealthMonitor] Consecutive losses critical: {consecutive_losses}"
            )
        elif consecutive_losses >= self.CONSECUTIVE_LOSS_WARNING:
            triggers.append(f"CONSECUTIVE_LOSS_WARNING_{consecutive_losses}")
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            logger.warning(
                f"[HealthMonitor] Consecutive losses warning: {consecutive_losses}"
            )

        # [5] Sharpe < 0.5 (30일)
        sharpe_30d = self._calculate_rolling_sharpe()
        if sharpe_30d is not None:
            if sharpe_30d < self.SHARPE_CRITICAL:
                triggers.append(f"SHARPE_CRITICAL_{sharpe_30d:.2f}")
                if status != HealthStatus.CRITICAL:
                    status = HealthStatus.CRITICAL
                logger.error(f"[HealthMonitor] Sharpe critical: {sharpe_30d:.2f}")
            elif sharpe_30d < self.SHARPE_WARNING:
                triggers.append(f"SHARPE_WARNING_{sharpe_30d:.2f}")
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                logger.warning(f"[HealthMonitor] Sharpe warning: {sharpe_30d:.2f}")

        # 결과 생성
        result = HealthCheckResult(
            status=status,
            sharpe_30d=sharpe_30d,
            mdd_current=mdd,
            consecutive_losses=consecutive_losses,
            daily_pnl_pct=today_pnl,
            triggers=triggers,
            recommendation=self._generate_recommendation(status, triggers),
            checked_at=datetime.now(),
        )

        # 로깅
        sharpe_str = f"{sharpe_30d:.2f}" if sharpe_30d is not None else "N/A"
        if status == HealthStatus.HEALTHY:
            logger.info(f"[HealthMonitor] Status: HEALTHY, Sharpe={sharpe_str}")
        elif status == HealthStatus.WARNING:
            logger.warning(f"[HealthMonitor] Status: WARNING, Triggers: {triggers}")
        elif status == HealthStatus.CRITICAL:
            logger.error(f"[HealthMonitor] Status: CRITICAL, Triggers: {triggers}")

        return result

    def get_summary(self) -> dict:
        """
        현재 상태 요약 (JSON 직렬화 가능)

        Returns:
            dict with status, metrics, recommendation
        """
        result = self.check_health()

        return {
            "status": result.status.value,
            "sharpe_30d": result.sharpe_30d,
            "mdd_pct": result.mdd_current * 100,
            "consecutive_losses": result.consecutive_losses,
            "daily_pnl_pct": result.daily_pnl_pct * 100,
            "triggers": result.triggers,
            "recommendation": result.recommendation,
            "total_trades": len(self._trade_history),
            "checked_at": result.checked_at.isoformat(),
        }

    # ==================== Private Methods ====================

    def _get_today_pnl(self) -> float:
        """오늘 PnL 반환 (% 단위)"""
        if not self._daily_pnl:
            return 0.0

        today = datetime.now().date()
        today_records = [d for d in self._daily_pnl if d["date"] == today]

        if today_records:
            return sum(d["pnl_pct"] for d in today_records)

        return 0.0

    def _calculate_current_mdd(self) -> float:
        """현재 MDD 계산"""
        if len(self._equity_history) < 2:
            return 0.0

        peak = self._equity_history[0]
        max_dd = 0.0

        for equity in self._equity_history:
            if equity > peak:
                peak = equity

            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def _count_consecutive_losses(self) -> int:
        """최근 연속 손실 횟수"""
        if not self._trade_history:
            return 0

        consecutive = 0
        for trade in reversed(self._trade_history):
            if trade["pnl"] < 0:
                consecutive += 1
            else:
                break

        return consecutive

    def _calculate_rolling_sharpe(self) -> Optional[float]:
        """
        30일 롤링 Sharpe (연율화)

        Returns:
            Sharpe ratio or None if insufficient data
        """
        if len(self._trade_history) < self.MIN_TRADES_FOR_SHARPE:
            return None

        # 최근 30일 거래 필터링
        cutoff_date = datetime.now() - timedelta(days=self.SHARPE_LOOKBACK_DAYS)
        recent_trades = [
            t
            for t in self._trade_history
            if t.get("timestamp", datetime.now()) >= cutoff_date
        ]

        if len(recent_trades) < self.MIN_TRADES_FOR_SHARPE:
            # 30일 데이터 부족 시 전체 데이터 사용
            recent_trades = self._trade_history

        returns = [t["pnl_pct"] for t in recent_trades]

        if len(returns) < 2:
            return None

        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        if std_return == 0:
            return 0.0

        # 연율화 (거래 빈도는 일별로 가정)
        sharpe = (mean_return / std_return) * (252**0.5)
        return sharpe

    def _generate_recommendation(
        self, status: HealthStatus, triggers: List[str]
    ) -> str:
        """
        상태별 권장 사항 생성

        Args:
            status: 건강 상태
            triggers: 트리거 목록

        Returns:
            권장 사항 문자열
        """
        if status == HealthStatus.HALTED:
            return (
                "⛔ Kill-Switch 활성화됨. "
                "즉시 거래 중단하고 kill_switch.txt 파일을 삭제한 후 시스템을 재검토하세요."
            )

        if status == HealthStatus.CRITICAL:
            recommendations = ["🔴 CRITICAL: 즉시 거래를 중단하고 다음을 확인하세요:"]

            for trigger in triggers:
                if "DAILY_LOSS" in trigger:
                    recommendations.append(
                        "  - 일일 손실 한도 초과: 오늘 거래 중단 권장"
                    )
                elif "MDD_CRITICAL" in trigger:
                    recommendations.append(
                        "  - MDD 15% 초과: 포지션 청산 및 전략 재검토"
                    )
                elif "CONSECUTIVE_LOSS" in trigger:
                    recommendations.append(
                        "  - 연속 손실 8회: 전략 파라미터 재조정 필요"
                    )
                elif "SHARPE_CRITICAL" in trigger:
                    recommendations.append(
                        "  - Sharpe < 0: 전략 효과성 상실, 중단 권장"
                    )

            return "\n".join(recommendations)

        if status == HealthStatus.WARNING:
            recommendations = ["⚠️ WARNING: 다음 항목을 검토하세요:"]

            for trigger in triggers:
                if "MDD_WARNING" in trigger:
                    recommendations.append("  - MDD 10% 초과: 포지션 크기 축소 권장")
                elif "CONSECUTIVE_LOSS_WARNING" in trigger:
                    recommendations.append("  - 연속 손실 5회: 전략 파라미터 점검")
                elif "SHARPE_WARNING" in trigger:
                    recommendations.append(
                        "  - Sharpe < 0.5: 전략 효과성 낮음, 조정 검토"
                    )

            return "\n".join(recommendations)

        return "✅ HEALTHY: 전략이 정상적으로 작동 중입니다."
