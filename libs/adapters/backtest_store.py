"""
백테스트 결과 저장소 - JSON 기반 백테스트 결과 저장/조회
- 전략별 디렉토리 구조
- 메타데이터 및 일별 수익률 저장
- 결과 목록 조회 및 비교
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """백테스트 결과 데이터 구조."""

    # 메타데이터
    backtest_id: str
    strategy_name: str
    created_at: str  # ISO format

    # 설정
    initial_capital: float
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD

    # 결과 데이터
    dates: List[str] = field(default_factory=list)  # YYYY-MM-DD strings
    daily_returns: List[float] = field(default_factory=list)

    # 성과 지표 (선택)
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0

    # 추가 메타데이터
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestResult":
        """Create from dictionary."""
        return cls(
            backtest_id=data.get("backtest_id", ""),
            strategy_name=data.get("strategy_name", ""),
            created_at=data.get("created_at", ""),
            initial_capital=float(data.get("initial_capital", 0)),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            dates=data.get("dates", []),
            daily_returns=data.get("daily_returns", []),
            total_return=float(data.get("total_return", 0)),
            sharpe_ratio=float(data.get("sharpe_ratio", 0)),
            max_drawdown=float(data.get("max_drawdown", 0)),
            win_rate=float(data.get("win_rate", 0)),
            total_trades=int(data.get("total_trades", 0)),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
        )

    def to_viewer_format(self) -> Dict[str, Any]:
        """Convert to backtest_viewer component format."""
        # Convert date strings to date objects
        date_objects = []
        for d in self.dates:
            try:
                date_objects.append(datetime.strptime(d, "%Y-%m-%d").date())
            except ValueError:
                continue

        return {
            "dates": date_objects,
            "daily_returns": self.daily_returns,
            "initial_capital": self.initial_capital,
            "strategy_name": self.strategy_name,
        }


class BacktestStore:
    """
    백테스트 결과 저장소.

    저장 경로: data/backtests/{strategy_name}/{backtest_id}.json

    Methods:
        save(result): 백테스트 결과 저장
        load(backtest_id): 특정 결과 로드
        list_backtests(): 모든 백테스트 목록
        list_by_strategy(strategy_name): 전략별 백테스트 목록
        get_latest(strategy_name): 최신 백테스트 결과
        delete(backtest_id): 백테스트 삭제
    """

    DEFAULT_STORE_DIR = "data/backtests"

    def __init__(self, store_dir: Optional[str] = None):
        """초기화.

        Args:
            store_dir: 저장 디렉토리 경로 (기본: data/backtests)
        """
        self.store_dir = Path(store_dir or self.DEFAULT_STORE_DIR)
        self._ensure_directory()
        logger.info("[BacktestStore] Initialized: %s", self.store_dir)

    def _ensure_directory(self) -> None:
        """디렉토리 생성."""
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _get_strategy_dir(self, strategy_name: str) -> Path:
        """전략 디렉토리 경로."""
        # 파일 시스템 안전한 이름으로 변환
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in strategy_name)
        strategy_dir = self.store_dir / safe_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        return strategy_dir

    def _get_file_path(self, strategy_name: str, backtest_id: str) -> Path:
        """백테스트 파일 경로."""
        strategy_dir = self._get_strategy_dir(strategy_name)
        return strategy_dir / f"{backtest_id}.json"

    def save(self, result: BacktestResult) -> bool:
        """백테스트 결과 저장.

        Args:
            result: BacktestResult 객체

        Returns:
            bool: 성공 여부
        """
        try:
            file_path = self._get_file_path(result.strategy_name, result.backtest_id)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info("[BacktestStore] Saved: %s", file_path)
            return True

        except Exception as e:
            logger.error("[BacktestStore] Save failed: %s", e)
            return False

    def load(self, strategy_name: str, backtest_id: str) -> Optional[BacktestResult]:
        """백테스트 결과 로드.

        Args:
            strategy_name: 전략 이름
            backtest_id: 백테스트 ID

        Returns:
            BacktestResult or None
        """
        file_path = self._get_file_path(strategy_name, backtest_id)

        if not file_path.exists():
            logger.debug("[BacktestStore] Not found: %s", file_path)
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BacktestResult.from_dict(data)

        except Exception as e:
            logger.error("[BacktestStore] Load failed: %s", e)
            return None

    def list_backtests(self) -> List[Dict[str, Any]]:
        """모든 백테스트 목록 조회.

        Returns:
            List of backtest metadata dicts
        """
        backtests = []

        try:
            for strategy_dir in self.store_dir.iterdir():
                if not strategy_dir.is_dir():
                    continue

                for file_path in strategy_dir.glob("*.json"):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        backtests.append({
                            "backtest_id": data.get("backtest_id", file_path.stem),
                            "strategy_name": data.get("strategy_name", strategy_dir.name),
                            "created_at": data.get("created_at", ""),
                            "start_date": data.get("start_date", ""),
                            "end_date": data.get("end_date", ""),
                            "total_return": data.get("total_return", 0),
                            "sharpe_ratio": data.get("sharpe_ratio", 0),
                            "max_drawdown": data.get("max_drawdown", 0),
                        })
                    except Exception as e:
                        logger.debug("[BacktestStore] Skip file %s: %s", file_path, e)
        except Exception as e:
            logger.error("[BacktestStore] List failed: %s", e)

        # Sort by created_at descending
        backtests.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return backtests

    def list_by_strategy(self, strategy_name: str) -> List[Dict[str, Any]]:
        """전략별 백테스트 목록.

        Args:
            strategy_name: 전략 이름

        Returns:
            List of backtest metadata dicts
        """
        all_backtests = self.list_backtests()
        return [b for b in all_backtests if b.get("strategy_name") == strategy_name]

    def get_latest(self, strategy_name: Optional[str] = None) -> Optional[BacktestResult]:
        """최신 백테스트 결과 조회.

        Args:
            strategy_name: 전략 이름 (None이면 전체에서 최신)

        Returns:
            BacktestResult or None
        """
        if strategy_name:
            backtests = self.list_by_strategy(strategy_name)
        else:
            backtests = self.list_backtests()

        if not backtests:
            return None

        latest = backtests[0]
        return self.load(latest["strategy_name"], latest["backtest_id"])

    def delete(self, strategy_name: str, backtest_id: str) -> bool:
        """백테스트 삭제.

        Args:
            strategy_name: 전략 이름
            backtest_id: 백테스트 ID

        Returns:
            bool: 성공 여부
        """
        file_path = self._get_file_path(strategy_name, backtest_id)

        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            logger.info("[BacktestStore] Deleted: %s", file_path)
            return True
        except Exception as e:
            logger.error("[BacktestStore] Delete failed: %s", e)
            return False

    def get_strategy_names(self) -> List[str]:
        """사용 가능한 전략 이름 목록.

        Returns:
            List of strategy names
        """
        strategies = set()

        try:
            for strategy_dir in self.store_dir.iterdir():
                if strategy_dir.is_dir() and list(strategy_dir.glob("*.json")):
                    strategies.add(strategy_dir.name)
        except Exception as e:
            logger.error("[BacktestStore] Get strategies failed: %s", e)

        return sorted(strategies)


def generate_backtest_id() -> str:
    """고유 백테스트 ID 생성.

    Returns:
        Timestamp-based ID (e.g., bt_20250119_153045)
    """
    return f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
