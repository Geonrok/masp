"""
TIGER 200 ETF Paper Trading.

TIGER 200 ETF (102110) 가상 매매 시스템.

Usage:
    python scripts/tiger200_paper_trading.py --once      # 1회 실행
    python scripts/tiger200_paper_trading.py --daemon    # 데몬 모드
    python scripts/tiger200_paper_trading.py --status    # 상태 확인

ETF 정보:
    종목코드: 102110
    운용사: 미래에셋자산운용
    총보수: 0.05%
    현재가: 약 4만원
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from libs.strategies.tiger200_etf import TIGER200Config, TIGER200Strategy

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paper Trading 디렉토리
PAPER_DIR = PROJECT_ROOT / "logs" / "tiger200_paper_trading"
PAPER_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Position:
    """보유 포지션."""

    shares: int  # 보유 주수
    avg_price: float  # 평균 매수가
    entry_date: str  # 매수일
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


@dataclass
class Trade:
    """거래 기록."""

    timestamp: str
    action: str  # BUY, SELL
    shares: int
    price: float
    amount: float  # 거래금액
    commission: float  # 수수료
    reason: str
    pnl: float = 0.0  # 실현 손익 (SELL 시)


@dataclass
class TradingState:
    """거래 상태."""

    # 계좌
    initial_capital: float = 1_000_000  # 초기 자본 100만원
    cash: float = 1_000_000
    position: Optional[dict] = None

    # 성과
    total_trades: int = 0
    winning_trades: int = 0
    realized_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_equity: float = 1_000_000

    # 기록
    trades: list = field(default_factory=list)
    equity_history: list = field(default_factory=list)
    signal_history: list = field(default_factory=list)

    # 마지막 상태
    last_update: str = ""
    last_action: str = "HOLD"
    last_composite: float = 0.0


class TIGER200PaperTrader:
    """TIGER 200 ETF Paper Trader."""

    # ETF 설정
    ETF_CODE = "102110"
    ETF_NAME = "TIGER 200"
    COMMISSION_RATE = 0.00015  # 0.015% (매수/매도 각각)
    MIN_ORDER_AMOUNT = 10_000  # 최소 주문금액 1만원

    def __init__(self, initial_capital: float = 1_000_000):
        """
        초기화.

        Args:
            initial_capital: 초기 자본금 (기본 100만원)
        """
        self.initial_capital = initial_capital
        self.strategy = TIGER200Strategy()

        # 파일 경로
        self.state_file = PAPER_DIR / "state.json"
        self.trades_file = PAPER_DIR / "trades.csv"
        self.log_file = PAPER_DIR / "trading.log"

        # 상태 로드
        self.state = self._load_state()

        logger.info(f"[TIGER200] Paper Trader 초기화")
        logger.info(f"[TIGER200] 초기 자본: {self.state.cash:,.0f}원")

    def _load_state(self) -> TradingState:
        """상태 로드."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    state = TradingState(
                        initial_capital=data.get(
                            "initial_capital", self.initial_capital
                        ),
                        cash=data.get("cash", self.initial_capital),
                        position=data.get("position"),
                        total_trades=data.get("total_trades", 0),
                        winning_trades=data.get("winning_trades", 0),
                        realized_pnl=data.get("realized_pnl", 0.0),
                        max_drawdown_pct=data.get("max_drawdown_pct", 0.0),
                        peak_equity=data.get("peak_equity", self.initial_capital),
                        trades=data.get("trades", []),
                        equity_history=data.get("equity_history", [])[-100:],
                        signal_history=data.get("signal_history", [])[-100:],
                        last_update=data.get("last_update", ""),
                        last_action=data.get("last_action", "HOLD"),
                        last_composite=data.get("last_composite", 0.0),
                    )
                    return state
            except Exception as e:
                logger.warning(f"상태 로드 실패: {e}")

        return TradingState(
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
            peak_equity=self.initial_capital,
        )

    def _save_state(self):
        """상태 저장."""
        try:
            data = {
                "initial_capital": self.state.initial_capital,
                "cash": self.state.cash,
                "position": self.state.position,
                "total_trades": self.state.total_trades,
                "winning_trades": self.state.winning_trades,
                "realized_pnl": self.state.realized_pnl,
                "max_drawdown_pct": self.state.max_drawdown_pct,
                "peak_equity": self.state.peak_equity,
                "trades": self.state.trades[-50:],  # 최근 50개만
                "equity_history": self.state.equity_history[-100:],
                "signal_history": self.state.signal_history[-100:],
                "last_update": self.state.last_update,
                "last_action": self.state.last_action,
                "last_composite": self.state.last_composite,
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"상태 저장 실패: {e}")

    def _log_trade(self, trade: Trade):
        """거래 기록."""
        try:
            header = not self.trades_file.exists()
            with open(self.trades_file, "a", encoding="utf-8") as f:
                if header:
                    f.write(
                        "timestamp,action,shares,price,amount,commission,pnl,reason\n"
                    )
                f.write(
                    f"{trade.timestamp},{trade.action},{trade.shares},"
                    f"{trade.price},{trade.amount},{trade.commission},"
                    f"{trade.pnl},{trade.reason}\n"
                )
        except Exception as e:
            logger.error(f"거래 기록 실패: {e}")

    def _log_message(self, msg: str):
        """메시지 기록."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] {msg}\n")
        except:
            pass

    def _get_etf_price(self) -> float:
        """
        ETF 현재가 조회.

        실제로는 KOSPI200 지수를 기반으로 추정.
        TIGER 200 ≈ KOSPI200 × 0.052 (대략적 비율)
        """
        if self.strategy._kospi_data is not None:
            kospi = self.strategy._kospi_data["close"].iloc[-1]
            # TIGER 200은 약 KOSPI200 × 0.052 수준
            # 765 포인트 → 약 40,000원
            etf_price = kospi * 52.2  # 약 40,000원 수준
            return round(etf_price, 0)
        return 40000  # 기본값

    def _calc_equity(self, etf_price: float) -> float:
        """총 자산 계산."""
        equity = self.state.cash

        if self.state.position:
            shares = self.state.position["shares"]
            avg_price = self.state.position["avg_price"]
            position_value = shares * etf_price
            equity += position_value

            # 미실현 손익 업데이트
            self.state.position["current_price"] = etf_price
            self.state.position["unrealized_pnl"] = (etf_price - avg_price) * shares
            self.state.position["unrealized_pnl_pct"] = (
                (etf_price - avg_price) / avg_price * 100 if avg_price > 0 else 0
            )

        return equity

    def _execute_buy(
        self, etf_price: float, composite: float, signals: dict, reason: str
    ):
        """매수 실행."""
        if self.state.position:
            logger.info("[TIGER200] 이미 포지션 보유 중, 매수 스킵")
            return

        # 주문 가능 금액 (수수료 고려)
        available = self.state.cash * 0.999  # 0.1% 여유

        if available < self.MIN_ORDER_AMOUNT:
            logger.warning(f"[TIGER200] 주문 금액 부족: {available:,.0f}원")
            return

        # 매수 수량 계산 (전액 매수)
        shares = int(available / etf_price)
        if shares <= 0:
            logger.warning("[TIGER200] 매수 가능 수량 없음")
            return

        # 거래 실행
        amount = shares * etf_price
        commission = amount * self.COMMISSION_RATE

        self.state.cash -= amount + commission
        self.state.position = {
            "shares": shares,
            "avg_price": etf_price,
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
            "current_price": etf_price,
            "unrealized_pnl": 0,
            "unrealized_pnl_pct": 0,
        }

        trade = Trade(
            timestamp=datetime.now().isoformat(),
            action="BUY",
            shares=shares,
            price=etf_price,
            amount=amount,
            commission=commission,
            reason=reason,
        )

        self.state.trades.append(asdict(trade))
        self.state.total_trades += 1
        self._log_trade(trade)

        logger.info(
            f"[TIGER200] 매수 체결: {shares}주 × {etf_price:,.0f}원 = {amount:,.0f}원"
        )
        self._log_message(f"BUY: {shares}주 @ {etf_price:,.0f}원, {reason}")

    def _execute_sell(
        self, etf_price: float, composite: float, signals: dict, reason: str
    ):
        """매도 실행."""
        if not self.state.position:
            logger.info("[TIGER200] 보유 포지션 없음, 매도 스킵")
            return

        shares = self.state.position["shares"]
        avg_price = self.state.position["avg_price"]

        # 거래 실행
        amount = shares * etf_price
        commission = amount * self.COMMISSION_RATE

        # 손익 계산
        gross_pnl = (etf_price - avg_price) * shares
        net_pnl = gross_pnl - commission
        pnl_pct = (etf_price - avg_price) / avg_price * 100

        self.state.cash += amount - commission
        self.state.realized_pnl += net_pnl

        if net_pnl > 0:
            self.state.winning_trades += 1

        trade = Trade(
            timestamp=datetime.now().isoformat(),
            action="SELL",
            shares=shares,
            price=etf_price,
            amount=amount,
            commission=commission,
            reason=reason,
            pnl=net_pnl,
        )

        self.state.trades.append(asdict(trade))
        self.state.total_trades += 1
        self._log_trade(trade)

        logger.info(
            f"[TIGER200] 매도 체결: {shares}주 × {etf_price:,.0f}원, "
            f"손익: {net_pnl:+,.0f}원 ({pnl_pct:+.2f}%)"
        )
        self._log_message(
            f"SELL: {shares}주 @ {etf_price:,.0f}원, P&L: {net_pnl:+,.0f}원"
        )

        # 포지션 초기화
        self.state.position = None

    def run_once(self) -> dict:
        """1회 실행."""
        logger.info("=" * 60)
        logger.info(f"[TIGER200] Paper Trading 실행")
        logger.info("=" * 60)

        # 데이터 로드
        if not self.strategy.load_data():
            logger.error("[TIGER200] 데이터 로드 실패")
            return {"error": "데이터 로드 실패"}

        # ETF 가격 조회
        etf_price = self._get_etf_price()
        logger.info(f"[TIGER200] {self.ETF_NAME} 추정가: {etf_price:,.0f}원")

        # 시그널 계산
        composite, signals, action = self.strategy.calculate_signal()

        logger.info(f"[TIGER200] Composite: {composite:.0%}")
        for name, sig in signals.items():
            status = "LONG" if sig == 1 else "CASH"
            logger.info(f"[TIGER200]   {name}: {status}")

        # 행동 실행
        has_position = self.state.position is not None

        if action == "BUY" and not has_position:
            self._execute_buy(
                etf_price, composite, signals, f"Composite {composite:.0%}"
            )
            executed_action = "BUY"
        elif action == "SELL" and has_position:
            self._execute_sell(
                etf_price, composite, signals, f"Composite {composite:.0%}"
            )
            executed_action = "SELL"
        else:
            executed_action = "HOLD"
            logger.info(
                f"[TIGER200] HOLD (Composite: {composite:.0%}, Position: {has_position})"
            )

        # 자산 계산
        equity = self._calc_equity(etf_price)

        # 드로다운 계산
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        dd_pct = (equity - self.state.peak_equity) / self.state.peak_equity * 100
        if dd_pct < self.state.max_drawdown_pct:
            self.state.max_drawdown_pct = dd_pct

        # 상태 업데이트
        self.state.last_update = datetime.now().isoformat()
        self.state.last_action = executed_action
        self.state.last_composite = composite

        # 기록
        self.state.equity_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "equity": equity,
                "price": etf_price,
            }
        )
        self.state.signal_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "composite": composite,
                "signals": signals,
                "action": executed_action,
            }
        )

        # 저장
        self._save_state()

        # 결과 출력
        total_return = (
            (equity - self.state.initial_capital) / self.state.initial_capital * 100
        )
        win_rate = (
            self.state.winning_trades / self.state.total_trades * 100
            if self.state.total_trades > 0
            else 0
        )

        logger.info("-" * 60)
        logger.info("[TIGER200] 현재 상태")
        logger.info("-" * 60)
        logger.info(f"  총 자산: {equity:,.0f}원")
        logger.info(f"  수익률: {total_return:+.2f}%")
        logger.info(f"  실현 손익: {self.state.realized_pnl:+,.0f}원")
        logger.info(f"  최대 낙폭: {self.state.max_drawdown_pct:.2f}%")
        logger.info(f"  총 거래: {self.state.total_trades}회")
        logger.info(f"  승률: {win_rate:.1f}%")

        if self.state.position:
            pos = self.state.position
            logger.info(f"  포지션: {pos['shares']}주 @ {pos['avg_price']:,.0f}원")
            logger.info(
                f"  미실현 손익: {pos['unrealized_pnl']:+,.0f}원 ({pos['unrealized_pnl_pct']:+.2f}%)"
            )
        else:
            logger.info("  포지션: 없음 (현금)")

        logger.info("=" * 60)

        return {
            "timestamp": self.state.last_update,
            "action": executed_action,
            "composite": composite,
            "signals": signals,
            "etf_price": etf_price,
            "equity": equity,
            "return_pct": total_return,
            "position": self.state.position,
        }

    def run_daemon(self, interval_minutes: int = 60):
        """데몬 모드 실행."""
        logger.info(f"[TIGER200] 데몬 모드 시작 (간격: {interval_minutes}분)")
        self._log_message(f"데몬 시작: {interval_minutes}분 간격")

        while True:
            try:
                self.run_once()
                logger.info(f"[TIGER200] 다음 실행: {interval_minutes}분 후...")
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                logger.info("[TIGER200] 데몬 종료")
                self._log_message("데몬 종료")
                break
            except Exception as e:
                logger.error(f"[TIGER200] 오류: {e}")
                time.sleep(60)

    def get_status(self) -> dict:
        """현재 상태 조회."""
        if not self.strategy.load_data():
            return {"error": "데이터 로드 실패"}

        etf_price = self._get_etf_price()
        equity = self._calc_equity(etf_price)
        total_return = (
            (equity - self.state.initial_capital) / self.state.initial_capital * 100
        )

        indicators = self.strategy.get_indicators()

        return {
            "etf": self.ETF_NAME,
            "etf_code": self.ETF_CODE,
            "etf_price": etf_price,
            "last_update": self.state.last_update,
            "last_action": self.state.last_action,
            "last_composite": self.state.last_composite,
            "equity": equity,
            "cash": self.state.cash,
            "return_pct": total_return,
            "realized_pnl": self.state.realized_pnl,
            "max_drawdown_pct": self.state.max_drawdown_pct,
            "total_trades": self.state.total_trades,
            "win_rate": (
                self.state.winning_trades / self.state.total_trades * 100
                if self.state.total_trades > 0
                else 0
            ),
            "position": self.state.position,
            "indicators": indicators,
        }


def main():
    parser = argparse.ArgumentParser(description="TIGER 200 ETF Paper Trading")
    parser.add_argument("--once", action="store_true", help="1회 실행")
    parser.add_argument("--daemon", action="store_true", help="데몬 모드")
    parser.add_argument("--status", action="store_true", help="상태 확인")
    parser.add_argument("--interval", type=int, default=60, help="간격 (분)")
    parser.add_argument(
        "--capital", type=float, default=1_000_000, help="초기 자본 (원)"
    )

    args = parser.parse_args()

    trader = TIGER200PaperTrader(initial_capital=args.capital)

    if args.status:
        status = trader.get_status()
        print("\n" + "=" * 60)
        print("TIGER 200 ETF Paper Trading 상태")
        print("=" * 60)

        # 기본 정보
        print(f"\n[ETF 정보]")
        print(f"  종목: {status.get('etf')} ({status.get('etf_code')})")
        print(f"  현재가: {status.get('etf_price', 0):,.0f}원")

        # 계좌 정보
        print(f"\n[계좌 현황]")
        print(f"  총 자산: {status.get('equity', 0):,.0f}원")
        print(f"  현금: {status.get('cash', 0):,.0f}원")
        print(f"  수익률: {status.get('return_pct', 0):+.2f}%")
        print(f"  실현 손익: {status.get('realized_pnl', 0):+,.0f}원")
        print(f"  최대 낙폭: {status.get('max_drawdown_pct', 0):.2f}%")

        # 거래 통계
        print(f"\n[거래 통계]")
        print(f"  총 거래: {status.get('total_trades', 0)}회")
        print(f"  승률: {status.get('win_rate', 0):.1f}%")

        # 포지션
        print(f"\n[포지션]")
        pos = status.get("position")
        if pos:
            print(f"  보유: {pos['shares']}주 @ {pos['avg_price']:,.0f}원")
            print(f"  현재가: {pos['current_price']:,.0f}원")
            print(
                f"  미실현 손익: {pos['unrealized_pnl']:+,.0f}원 ({pos['unrealized_pnl_pct']:+.2f}%)"
            )
        else:
            print("  없음 (현금 보유)")

        # 시그널
        print(f"\n[마지막 시그널]")
        print(f"  Composite: {status.get('last_composite', 0):.0%}")
        print(f"  Action: {status.get('last_action', 'N/A')}")
        print(f"  시간: {status.get('last_update', 'N/A')}")

        # 지표
        print(f"\n[현재 지표]")
        indicators = status.get("indicators", {})
        for key, value in indicators.items():
            if isinstance(value, float):
                if "foreign" in key:
                    print(f"  {key}: {value/1e8:,.0f}억원")
                else:
                    print(f"  {key}: {value:,.2f}")
            else:
                print(f"  {key}: {value}")

        print("=" * 60 + "\n")

    elif args.daemon:
        trader.run_daemon(interval_minutes=args.interval)

    else:
        result = trader.run_once()
        print(
            f"\n결과: {json.dumps(result, indent=2, default=str, ensure_ascii=False)}"
        )


if __name__ == "__main__":
    main()
