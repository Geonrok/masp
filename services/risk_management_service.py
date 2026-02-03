"""
Automated Risk Management Service

자동 리스크 관리 시스템:
- 포트폴리오 리스크 실시간 모니터링
- 리스크 임계치 초과 시 자동 알림
- DrawdownGuard + AlertManager + Telegram 통합
- 일일/주간 손실 한도 관리
- 비상 청산 트리거

사용법:
    python -m services.risk_management_service

환경변수:
    TELEGRAM_BOT_TOKEN: 텔레그램 봇 토큰
    TELEGRAM_CHAT_ID: 알림을 받을 채팅 ID
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.monitoring.alert_manager import (
    Alert,
    AlertCategory,
    AlertManager,
    AlertRule,
    AlertSeverity,
)
from libs.notifications.telegram import TelegramNotifier
from libs.risk.drawdown_guard import DrawdownGuard, RiskState, RiskStatus

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """리스크 수준"""

    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


@dataclass
class RiskAlert:
    """리스크 알림"""

    level: RiskLevel
    title: str
    message: str
    metrics: Dict
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class RiskConfig:
    """리스크 설정"""

    # 손실 한도
    daily_loss_limit: float = 0.03  # 3%
    weekly_loss_limit: float = 0.07  # 7%
    max_drawdown_limit: float = 0.15  # 15%

    # 경고 임계치 (한도의 X%)
    caution_threshold: float = 0.5  # 50%
    warning_threshold: float = 0.7  # 70%
    danger_threshold: float = 0.9  # 90%

    # 변동성 한도
    max_daily_volatility: float = 0.05  # 5%

    # 알림 설정
    telegram_enabled: bool = True
    alert_cooldown_seconds: float = 300  # 5분 중복 알림 방지


class RiskManagementService:
    """
    자동 리스크 관리 서비스

    기능:
    1. 포트폴리오 리스크 모니터링
    2. 리스크 임계치 초과 시 자동 알림 (Telegram)
    3. 비상 청산 트리거
    4. 리스크 메트릭스 로깅
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        initial_capital: float = 10_000_000,
    ):
        self.config = config or RiskConfig()

        # DrawdownGuard 초기화
        self.drawdown_guard = DrawdownGuard(
            daily_loss_limit=self.config.daily_loss_limit,
            weekly_loss_limit=self.config.weekly_loss_limit,
            max_drawdown_limit=self.config.max_drawdown_limit,
            warning_threshold=self.config.warning_threshold,
        )
        self.drawdown_guard.initialize(initial_capital)

        # AlertManager 초기화
        self.alert_manager = AlertManager.get_instance()
        self._setup_alert_rules()

        # Telegram 알림
        self.notifier = TelegramNotifier()

        # 상태
        self.last_alert_time: Dict[str, float] = {}
        self.alert_history: List[RiskAlert] = []
        self.is_emergency_mode: bool = False

        logger.info("[RiskManagement] Service initialized")
        logger.info(
            f"  Daily limit: {self.config.daily_loss_limit:.1%}, "
            f"Weekly: {self.config.weekly_loss_limit:.1%}, "
            f"Max DD: {self.config.max_drawdown_limit:.1%}"
        )

    def _setup_alert_rules(self):
        """알림 규칙 설정"""
        # 텔레그램 알림 규칙 - CRITICAL 등급
        self.alert_manager.register_rule(
            AlertRule(
                name="telegram_critical",
                category=AlertCategory.RISK,
                min_severity=AlertSeverity.CRITICAL,
                callback=self._send_telegram_alert,
                rate_limit_seconds=60,
            )
        )

        # 텔레그램 알림 규칙 - WARNING 등급
        self.alert_manager.register_rule(
            AlertRule(
                name="telegram_warning",
                category=AlertCategory.RISK,
                min_severity=AlertSeverity.WARNING,
                callback=self._send_telegram_alert,
                rate_limit_seconds=300,
            )
        )

    def _send_telegram_alert(self, alert: Alert):
        """텔레그램으로 알림 전송"""
        if not self.config.telegram_enabled or not self.notifier.enabled:
            return

        # Severity에 따른 이모지
        severity_prefix = {
            AlertSeverity.INFO: "[INFO]",
            AlertSeverity.WARNING: "[!] WARNING",
            AlertSeverity.ERROR: "[!!] ERROR",
            AlertSeverity.CRITICAL: "[!!!] CRITICAL",
        }
        prefix = severity_prefix.get(alert.severity, "")

        message = (
            f"<b>{prefix}</b>\n"
            f"<b>{alert.title}</b>\n\n"
            f"{alert.message}\n\n"
            f"Source: {alert.source}\n"
            f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.notifier.send_message_sync(message)

    def update_capital(self, current_capital: float):
        """현재 자본 업데이트"""
        self.drawdown_guard.current_capital = current_capital
        if current_capital > self.drawdown_guard.peak_capital:
            self.drawdown_guard.peak_capital = current_capital

    def record_trade(self, symbol: str, pnl: float, side: str = "buy"):
        """거래 기록"""
        self.drawdown_guard.record_trade(symbol, pnl, side)

    def check_risk(self) -> RiskState:
        """
        리스크 상태 확인

        Returns:
            RiskState 객체
        """
        return self.drawdown_guard.check_risk()

    def assess_risk_level(self) -> RiskLevel:
        """
        현재 리스크 수준 평가

        Returns:
            RiskLevel enum
        """
        state = self.check_risk()

        if state.status == RiskStatus.HALTED:
            return RiskLevel.CRITICAL

        # 각 메트릭스별 사용률 계산
        daily_usage = (
            abs(state.daily_pnl)
            / (self.drawdown_guard.peak_capital * self.config.daily_loss_limit)
            if state.daily_pnl < 0
            else 0
        )
        weekly_usage = (
            abs(state.weekly_pnl)
            / (self.drawdown_guard.peak_capital * self.config.weekly_loss_limit)
            if state.weekly_pnl < 0
            else 0
        )
        dd_usage = state.current_drawdown / self.config.max_drawdown_limit

        max_usage = max(daily_usage, weekly_usage, dd_usage)

        if max_usage >= self.config.danger_threshold:
            return RiskLevel.DANGER
        elif max_usage >= self.config.warning_threshold:
            return RiskLevel.WARNING
        elif max_usage >= self.config.caution_threshold:
            return RiskLevel.CAUTION
        else:
            return RiskLevel.SAFE

    def monitor(self) -> Optional[RiskAlert]:
        """
        리스크 모니터링 및 알림 생성

        Returns:
            RiskAlert if alert triggered, None otherwise
        """
        state = self.check_risk()
        risk_level = self.assess_risk_level()

        # 메트릭스 수집
        metrics = self.drawdown_guard.get_metrics()

        # 알림 필요 여부 확인
        alert = self._evaluate_alert(state, risk_level, metrics)

        if alert:
            self.alert_history.append(alert)
            self._trigger_alert(alert)

        return alert

    def _evaluate_alert(
        self,
        state: RiskState,
        risk_level: RiskLevel,
        metrics: Dict,
    ) -> Optional[RiskAlert]:
        """알림 필요 여부 평가"""
        # 쿨다운 확인
        now = time.time()
        cooldown_key = f"{risk_level.value}"
        last_time = self.last_alert_time.get(cooldown_key, 0)

        if now - last_time < self.config.alert_cooldown_seconds:
            return None

        # SAFE 수준은 알림 없음
        if risk_level == RiskLevel.SAFE:
            return None

        # 알림 생성
        title, message = self._generate_alert_message(state, risk_level, metrics)

        self.last_alert_time[cooldown_key] = now

        return RiskAlert(
            level=risk_level,
            title=title,
            message=message,
            metrics=metrics,
        )

    def _generate_alert_message(
        self,
        state: RiskState,
        risk_level: RiskLevel,
        metrics: Dict,
    ) -> tuple[str, str]:
        """알림 메시지 생성"""
        level_names = {
            RiskLevel.CAUTION: "Caution",
            RiskLevel.WARNING: "Warning",
            RiskLevel.DANGER: "Danger",
            RiskLevel.CRITICAL: "CRITICAL",
        }

        title = f"[Risk {level_names[risk_level]}] Portfolio Risk Alert"

        lines = [
            f"Risk Level: {risk_level.value.upper()}",
            "",
            (
                f"Daily P&L: {metrics['daily_pnl']:+,.0f} ({metrics['daily_pnl']/metrics['peak_capital']*100:+.2f}%)"
                if metrics["peak_capital"] > 0
                else "Daily P&L: N/A"
            ),
            (
                f"Weekly P&L: {metrics['weekly_pnl']:+,.0f} ({metrics['weekly_pnl']/metrics['peak_capital']*100:+.2f}%)"
                if metrics["peak_capital"] > 0
                else "Weekly P&L: N/A"
            ),
            f"Drawdown: {metrics['current_drawdown']*100:.1f}%",
            "",
            f"Daily Limit: {metrics['daily_limit']*100:.1f}%",
            f"Weekly Limit: {metrics['weekly_limit']*100:.1f}%",
            f"Max DD Limit: {metrics['max_drawdown_limit']*100:.1f}%",
            "",
            f"Status: {metrics['status'].upper()}",
            f"Can Trade: {'Yes' if metrics['can_trade'] else 'NO'}",
        ]

        if metrics.get("halt_reason"):
            lines.append(f"\nHalt Reason: {metrics['halt_reason']}")

        message = "\n".join(lines)
        return title, message

    def _trigger_alert(self, alert: RiskAlert):
        """알림 트리거"""
        # AlertManager를 통한 알림
        severity_map = {
            RiskLevel.CAUTION: AlertSeverity.INFO,
            RiskLevel.WARNING: AlertSeverity.WARNING,
            RiskLevel.DANGER: AlertSeverity.ERROR,
            RiskLevel.CRITICAL: AlertSeverity.CRITICAL,
        }

        self.alert_manager.alert(
            category=AlertCategory.RISK,
            severity=severity_map[alert.level],
            title=alert.title,
            message=alert.message,
            source="RiskManagementService",
            details=alert.metrics,
        )

        logger.warning(f"[RiskManagement] {alert.level.value}: {alert.title}")

    def trigger_emergency_mode(self, reason: str):
        """
        비상 모드 활성화

        모든 포지션 청산 신호 발생
        """
        self.is_emergency_mode = True
        logger.critical(f"[RiskManagement] EMERGENCY MODE ACTIVATED: {reason}")

        # CRITICAL 알림 전송
        self.alert_manager.critical(
            title="EMERGENCY MODE ACTIVATED",
            message=f"Reason: {reason}\nAll positions should be closed immediately.",
            category=AlertCategory.RISK,
            source="RiskManagementService",
        )

    def reset_emergency_mode(self):
        """비상 모드 해제"""
        if self.is_emergency_mode:
            self.is_emergency_mode = False
            logger.info("[RiskManagement] Emergency mode deactivated")

            self.alert_manager.info(
                title="Emergency Mode Deactivated",
                message="Risk management returned to normal operation.",
                category=AlertCategory.RISK,
                source="RiskManagementService",
            )

    def can_trade(self) -> bool:
        """거래 가능 여부 확인"""
        if self.is_emergency_mode:
            return False
        return self.drawdown_guard.can_trade()

    def get_position_size_multiplier(self) -> float:
        """
        리스크 수준에 따른 포지션 사이즈 승수 반환

        Returns:
            0.0 ~ 1.0 범위의 승수
        """
        if not self.can_trade():
            return 0.0

        risk_level = self.assess_risk_level()

        multipliers = {
            RiskLevel.SAFE: 1.0,
            RiskLevel.CAUTION: 0.7,
            RiskLevel.WARNING: 0.5,
            RiskLevel.DANGER: 0.2,
            RiskLevel.CRITICAL: 0.0,
        }

        return multipliers.get(risk_level, 0.5)

    def format_telegram_status(self) -> str:
        """텔레그램 상태 메시지 포맷팅"""
        self.check_risk()
        metrics = self.drawdown_guard.get_metrics()
        risk_level = self.assess_risk_level()

        level_emoji = {
            RiskLevel.SAFE: "[OK]",
            RiskLevel.CAUTION: "[!]",
            RiskLevel.WARNING: "[!!]",
            RiskLevel.DANGER: "[!!!]",
            RiskLevel.CRITICAL: "[ALERT]",
        }

        lines = [
            "<b>[MASP] Risk Status Report</b>",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"<b>Risk Level: {risk_level.value.upper()} {level_emoji[risk_level]}</b>",
            f"Can Trade: {'Yes' if self.can_trade() else 'NO'}",
            f"Position Multiplier: {self.get_position_size_multiplier()*100:.0f}%",
            "",
            "<b>Portfolio</b>",
            f"  Current: {metrics['current_capital']:,.0f}",
            f"  Peak: {metrics['peak_capital']:,.0f}",
            f"  Drawdown: {metrics['current_drawdown']*100:.1f}%",
            "",
            "<b>P&L Limits</b>",
            f"  Daily: {metrics['daily_pnl']:+,.0f} / Limit: {metrics['daily_limit']*100:.1f}%",
            f"  Weekly: {metrics['weekly_pnl']:+,.0f} / Limit: {metrics['weekly_limit']*100:.1f}%",
            f"  Max DD: {metrics['current_drawdown']*100:.1f}% / Limit: {metrics['max_drawdown_limit']*100:.1f}%",
        ]

        if self.is_emergency_mode:
            lines.insert(2, "\n<b>*** EMERGENCY MODE ACTIVE ***</b>\n")

        return "\n".join(lines)

    def send_status_report(self) -> bool:
        """상태 리포트 전송"""
        message = self.format_telegram_status()
        return self.notifier.send_message_sync(message)

    def get_summary(self) -> dict:
        """요약 정보 반환 (대시보드용)"""
        state = self.check_risk()
        metrics = self.drawdown_guard.get_metrics()
        risk_level = self.assess_risk_level()

        return {
            "risk_level": risk_level.value,
            "can_trade": self.can_trade(),
            "position_multiplier": self.get_position_size_multiplier(),
            "is_emergency": self.is_emergency_mode,
            "current_capital": metrics["current_capital"],
            "peak_capital": metrics["peak_capital"],
            "current_drawdown": metrics["current_drawdown"],
            "daily_pnl": metrics["daily_pnl"],
            "weekly_pnl": metrics["weekly_pnl"],
            "status": metrics["status"],
            "is_halted": metrics["is_halted"],
            "halt_reason": metrics.get("halt_reason"),
            "message": state.message,
            "alert_count": len(self.alert_history),
        }


def main():
    """CLI 실행"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("MASP Risk Management Service")
    print("=" * 60)

    # 초기화
    config = RiskConfig(
        daily_loss_limit=0.03,
        weekly_loss_limit=0.07,
        max_drawdown_limit=0.15,
    )

    service = RiskManagementService(
        config=config,
        initial_capital=10_000_000,
    )

    # 시뮬레이션: 손실 발생
    print("\nSimulating trades...")
    service.record_trade("BTC", -100000, "sell")
    service.record_trade("ETH", -50000, "sell")

    # 리스크 확인
    summary = service.get_summary()

    print(f"\nRisk Level: {summary['risk_level'].upper()}")
    print(f"Can Trade: {summary['can_trade']}")
    print(f"Position Multiplier: {summary['position_multiplier']*100:.0f}%")
    print(f"\nCurrent Capital: {summary['current_capital']:,.0f}")
    print(f"Peak Capital: {summary['peak_capital']:,.0f}")
    print(f"Drawdown: {summary['current_drawdown']*100:.1f}%")
    print(f"\nDaily P&L: {summary['daily_pnl']:+,.0f}")
    print(f"Weekly P&L: {summary['weekly_pnl']:+,.0f}")
    print(f"\nStatus: {summary['status']}")
    print(f"Message: {summary['message']}")

    # 모니터링 실행
    print("\n" + "-" * 60)
    print("Running risk monitor...")
    alert = service.monitor()
    if alert:
        print("\nAlert Triggered!")
        print(f"Level: {alert.level.value}")
        print(f"Title: {alert.title}")
        print(f"Message:\n{alert.message}")

    # 텔레그램 전송
    if service.notifier.enabled:
        print("\n" + "=" * 60)
        response = input("Send status to Telegram? (y/n): ")
        if response.lower() == "y":
            success = service.send_status_report()
            print(f"Telegram sent: {success}")
    else:
        print("\nTelegram not configured.")


if __name__ == "__main__":
    main()
