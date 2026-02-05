"""
Daily Signal Alert Service

Sept_v3_RSI50_Gate 전략 시그널을 계산하고 텔레그램으로 알림을 보냅니다.

전략: 7중 OR 시그널 + 거래량 상위 30% 필터 + BTC Gate

시그널 (7개 중 1개 이상):
    1. KAMA(5): price > KAMA
    2. TSMOM(90): price > price[90]
    3. EMA Cross: EMA12 > EMA26
    4. Momentum(20): price > price[20]
    5. SMA Cross: SMA20 > SMA50
    6. RSI(14) > 50
    7. Higher Low: price > min(price[1:20])

사용법:
    python -m services.daily_signal_alert

환경변수:
    TELEGRAM_BOT_TOKEN: 텔레그램 봇 토큰
    TELEGRAM_CHAT_ID: 알림을 받을 채팅 ID
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

from libs.notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

# Data paths
DATA_ROOT = Path(os.environ.get("CRYPTO_DATA_ROOT", "/app/data/crypto_ohlcv"))


class DailySignalAlertService:
    """
    일간 시그널 알림 서비스

    Sept_v3_RSI50_Gate 전략의 시그널을 계산하고
    텔레그램으로 매일 알림을 전송합니다.

    7중 OR 시그널 + 거래량 상위 30% 필터 + BTC Gate
    """

    # 전략 정보
    STRATEGY_NAME = "Sept-v3-RSI50-Gate"
    STRATEGY_VERSION = "3.0"

    def __init__(
        self,
        # BTC Gate
        btc_ma_period: int = 30,
        # 7 Signals
        kama_period: int = 5,
        tsmom_lookback: int = 90,
        ema_fast: int = 12,
        ema_slow: int = 26,
        momentum_period: int = 20,
        sma_fast: int = 20,
        sma_slow: int = 50,
        rsi_period: int = 14,
        rsi_threshold: int = 50,
        higher_low_period: int = 20,
        # v3 settings
        min_signals: int = 1,  # 7중 OR
        volume_filter_pct: float = 0.30,  # 상위 30%
        top_n: int = 30,
        exchange: str = "upbit",
    ):
        # BTC Gate
        self.btc_ma_period = btc_ma_period

        # 7 Signal parameters
        self.kama_period = kama_period
        self.tsmom_lookback = tsmom_lookback
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.momentum_period = momentum_period
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.higher_low_period = higher_low_period

        # v3 settings
        self.min_signals = min_signals
        self.volume_filter_pct = volume_filter_pct
        self.top_n = top_n
        self.exchange = exchange

        self.notifier = TelegramNotifier()

        logger.info(
            f"[DailySignalAlert] Initialized: {self.STRATEGY_NAME} v{self.STRATEGY_VERSION}"
        )
        logger.info(
            f"  7중 OR (min={min_signals}), Volume Top {volume_filter_pct*100:.0f}%, BTC Gate MA{btc_ma_period}"
        )

    # ===== 지표 계산 함수들 =====

    def _calc_sma(self, prices: np.ndarray, period: int) -> float:
        """SMA 계산 (마지막 값)"""
        if len(prices) < period:
            return np.nan
        return np.mean(prices[-period:])

    def _calc_ema(self, prices: np.ndarray, period: int) -> float:
        """EMA 계산 (마지막 값)"""
        if len(prices) < period:
            return np.nan
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _calc_kama(self, prices: np.ndarray, period: int = 5) -> float:
        """KAMA 계산 (마지막 값)"""
        n = len(prices)
        if n < period + 1:
            return np.nan

        kama = np.mean(prices[:period])
        fast_sc = 2 / 3  # fast=2
        slow_sc = 2 / 31  # slow=30

        for i in range(period, n):
            change = abs(prices[i] - prices[i - period])
            volatility = np.sum(np.abs(np.diff(prices[i - period : i + 1])))
            er = change / volatility if volatility > 0 else 0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama = kama + sc * (prices[i] - kama)

        return kama

    def _calc_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI 계산 (마지막 값)"""
        if len(prices) < period + 1:
            return np.nan

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # ===== 7개 시그널 체크 =====

    def _check_kama_signal(self, prices: np.ndarray) -> bool:
        """Signal 1: KAMA(5) - price > KAMA"""
        if len(prices) < self.kama_period + 1:
            return False
        kama = self._calc_kama(prices, self.kama_period)
        return prices[-1] > kama if not np.isnan(kama) else False

    def _check_tsmom_signal(self, prices: np.ndarray) -> bool:
        """Signal 2: TSMOM(90) - price > price[90]"""
        if len(prices) <= self.tsmom_lookback:
            return False
        return prices[-1] > prices[-(self.tsmom_lookback + 1)]

    def _check_ema_cross_signal(self, prices: np.ndarray) -> bool:
        """Signal 3: EMA Cross - EMA12 > EMA26"""
        if len(prices) < self.ema_slow:
            return False
        ema_fast = self._calc_ema(prices, self.ema_fast)
        ema_slow = self._calc_ema(prices, self.ema_slow)
        return ema_fast > ema_slow

    def _check_momentum_signal(self, prices: np.ndarray) -> bool:
        """Signal 4: Momentum(20) - price > price[20]"""
        if len(prices) <= self.momentum_period:
            return False
        return prices[-1] > prices[-(self.momentum_period + 1)]

    def _check_sma_cross_signal(self, prices: np.ndarray) -> bool:
        """Signal 5: SMA Cross - SMA20 > SMA50"""
        if len(prices) < self.sma_slow:
            return False
        sma_fast = self._calc_sma(prices, self.sma_fast)
        sma_slow = self._calc_sma(prices, self.sma_slow)
        return sma_fast > sma_slow

    def _check_rsi_signal(self, prices: np.ndarray) -> bool:
        """Signal 6: RSI(14) > 50"""
        if len(prices) < self.rsi_period + 1:
            return False
        rsi = self._calc_rsi(prices, self.rsi_period)
        return rsi > self.rsi_threshold if not np.isnan(rsi) else False

    def _check_higher_low_signal(self, prices: np.ndarray) -> bool:
        """Signal 7: Higher Low - price > min(price[1:20])"""
        if len(prices) <= self.higher_low_period:
            return False
        return prices[-1] > min(prices[-(self.higher_low_period + 1) : -1])

    def _count_signals(self, prices: np.ndarray) -> Tuple[int, Dict[str, bool]]:
        """7개 시그널 카운트 및 상세 정보 반환"""
        signal_details = {
            "kama": self._check_kama_signal(prices),
            "tsmom": self._check_tsmom_signal(prices),
            "ema_cross": self._check_ema_cross_signal(prices),
            "momentum": self._check_momentum_signal(prices),
            "sma_cross": self._check_sma_cross_signal(prices),
            "rsi": self._check_rsi_signal(prices),
            "higher_low": self._check_higher_low_signal(prices),
        }
        count = sum(signal_details.values())
        return count, signal_details

    # ===== 데이터 로딩 =====

    def load_ohlcv(self, min_days: int = 100) -> Dict[str, pd.DataFrame]:
        """OHLCV 데이터 로드 (API 우선, 로컬 파일 폴백)"""
        data = self._load_from_api(min_days)
        if data:
            logger.info(f"[DailySignalAlert] Loaded {len(data)} symbols from API")
            return data

        data = self._load_from_file(min_days)
        if data:
            logger.info(
                f"[DailySignalAlert] Loaded {len(data)} symbols from local files"
            )
            return data

        logger.warning("[DailySignalAlert] No data loaded")
        return {}

    def _load_from_api(self, min_days: int = 100) -> Dict[str, pd.DataFrame]:
        """Upbit API에서 OHLCV 데이터 로드"""
        try:
            import pyupbit

            tickers = pyupbit.get_tickers(fiat="KRW")
            if not tickers:
                logger.warning("[DailySignalAlert] No tickers from API")
                return {}

            data = {}
            # BTC를 항상 포함
            btc_ticker = "KRW-BTC"
            target_tickers = [btc_ticker] if btc_ticker in tickers else []
            target_tickers += [t for t in tickers[: self.top_n] if t != btc_ticker]

            for ticker in target_tickers:
                try:
                    df = pyupbit.get_ohlcv(ticker, interval="day", count=min_days + 10)
                    if df is None or df.empty or len(df) < min_days:
                        continue

                    df = df.rename(
                        columns={
                            "open": "open",
                            "high": "high",
                            "low": "low",
                            "close": "close",
                            "volume": "volume",
                        }
                    )

                    required = ["open", "high", "low", "close", "volume"]
                    if all(c in df.columns for c in required):
                        df = df[required].dropna()
                        if len(df) >= min_days:
                            data[ticker] = df

                except Exception as e:
                    logger.debug(f"[DailySignalAlert] Failed to load {ticker}: {e}")
                    continue

            return data

        except ImportError:
            logger.warning("[DailySignalAlert] pyupbit not installed")
            return {}
        except Exception as e:
            logger.warning(f"[DailySignalAlert] API load failed: {e}")
            return {}

    def _load_from_file(self, min_days: int = 100) -> Dict[str, pd.DataFrame]:
        """로컬 CSV 파일에서 OHLCV 데이터 로드"""
        folder = DATA_ROOT / f"{self.exchange}_1d"
        if not folder.exists():
            logger.debug(f"[DailySignalAlert] Data folder not found: {folder}")
            return {}

        data = {}
        for f in folder.glob("*.csv"):
            try:
                df = pd.read_csv(f)
                date_col = [
                    c for c in df.columns if "date" in c.lower() or "time" in c.lower()
                ]
                if not date_col:
                    continue

                df["date"] = pd.to_datetime(df[date_col[0]]).dt.normalize()
                df = df.set_index("date").sort_index()
                df = df[~df.index.duplicated(keep="last")]

                required = ["open", "high", "low", "close", "volume"]
                if not all(c in df.columns for c in required):
                    continue

                df = df[required]
                if len(df) >= min_days:
                    data[f.stem] = df
            except Exception as e:
                logger.debug(f"[DailySignalAlert] Failed to load {f}: {e}")
                continue

        return data

    def _get_volume_rank(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """거래대금 기준 랭킹"""
        vols = {}
        for symbol, df in data.items():
            avg_volume = (df["close"] * df["volume"]).tail(20).mean()
            vols[symbol] = avg_volume
        return vols

    # ===== 시그널 계산 =====

    def calculate_signals(self) -> Dict[str, dict]:
        """모든 심볼에 대한 시그널 계산"""
        data = self.load_ohlcv()
        if not data:
            logger.error("[DailySignalAlert] No data loaded")
            return {}

        # BTC 데이터 찾기
        btc_data = None
        for k, v in data.items():
            if "BTC" in k.upper() and "DOWN" not in k.upper():
                btc_data = v
                break

        if btc_data is None:
            logger.error("[DailySignalAlert] BTC data not found")
            return {}

        # BTC Gate 계산
        btc_prices = btc_data["close"].values
        btc_ma = self._calc_sma(btc_prices, self.btc_ma_period)
        btc_gate = btc_prices[-1] > btc_ma if not np.isnan(btc_ma) else False

        # 거래대금 랭킹
        volume_ranks = self._get_volume_rank(data)
        sorted_by_volume = sorted(
            volume_ranks.items(), key=lambda x: x[1], reverse=True
        )

        # 시그널 계산 (BTC 제외)
        candidates = []
        all_signals = {}

        for symbol, df in data.items():
            if "BTC" in symbol.upper():
                continue

            prices = df["close"].values
            min_len = (
                max(self.tsmom_lookback, self.sma_slow, self.higher_low_period) + 10
            )

            if len(prices) < min_len:
                continue

            signal_count, signal_details = self._count_signals(prices)

            price_change_1d = (
                (prices[-1] - prices[-2]) / prices[-2] * 100 if len(prices) > 1 else 0
            )
            price_change_7d = (
                (prices[-1] - prices[-7]) / prices[-7] * 100 if len(prices) > 7 else 0
            )

            all_signals[symbol] = {
                "price": prices[-1],
                "signal_count": signal_count,
                "signals": signal_details,
                "change_1d": price_change_1d,
                "change_7d": price_change_7d,
                "volume": volume_ranks.get(symbol, 0),
            }

            # 시그널 조건 충족 시 후보에 추가
            if signal_count >= self.min_signals:
                candidates.append((symbol, signal_count, volume_ranks.get(symbol, 0)))

        # v3: 거래량 상위 30% 필터
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            cutoff = max(1, int(len(candidates) * self.volume_filter_pct))
            top_candidates = set(s for s, _, _ in candidates[:cutoff])
        else:
            top_candidates = set()

        # Entry 시그널 결정
        for symbol in all_signals:
            is_entry = (
                btc_gate
                and all_signals[symbol]["signal_count"] >= self.min_signals
                and symbol in top_candidates
            )
            all_signals[symbol]["entry_signal"] = is_entry
            all_signals[symbol]["in_volume_top"] = symbol in top_candidates

        # 메타 정보 추가
        all_signals["_META"] = {
            "btc_gate": btc_gate,
            "btc_price": btc_prices[-1],
            "btc_ma": btc_ma,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "strategy": self.STRATEGY_NAME,
            "version": self.STRATEGY_VERSION,
            "min_signals": self.min_signals,
            "volume_filter_pct": self.volume_filter_pct,
            "btc_ma_period": self.btc_ma_period,
            "total_candidates": len(candidates),
            "filtered_candidates": len(top_candidates),
        }

        return all_signals

    def format_telegram_message(self, signals: Dict[str, dict]) -> str:
        """텔레그램 메시지 포맷팅"""
        if not signals or "_META" not in signals:
            return "<b>[MASP] 시그널 오류</b>\n시그널을 계산할 수 없습니다"

        meta = signals["_META"]

        lines = [
            f"<b>[MASP] {meta['strategy']} v{meta['version']}</b>",
            f"시간: {meta['timestamp']}",
            f"조건: 7중 OR (min {meta['min_signals']}) + Vol Top {meta['volume_filter_pct']*100:.0f}%",
            "",
            f"<b>BTC Gate: {'통과' if meta['btc_gate'] else '실패'}</b>",
            f"BTC: {meta['btc_price']:,.0f}원 {'(MA 위)' if meta['btc_gate'] else '(MA 아래)'}",
            f"MA{meta['btc_ma_period']}: {meta['btc_ma']:,.0f}원",
            "",
        ]

        # Entry signals
        entry_signals = [
            (s, d) for s, d in signals.items() if s != "_META" and d.get("entry_signal")
        ]
        entry_signals.sort(key=lambda x: x[1]["change_7d"], reverse=True)

        if entry_signals:
            lines.append(f"<b>매수 시그널 ({len(entry_signals)}개):</b>")
            for symbol, data in entry_signals[:10]:
                active = [k[0].upper() for k, v in data["signals"].items() if v]
                sig_str = "".join(active[:3]) + ("+" if len(active) > 3 else "")
                lines.append(
                    f"  [{data['signal_count']}/7 {sig_str}] {symbol}: "
                    f"{data['price']:,.0f}원 (7D {data['change_7d']:+.1f}%)"
                )
        else:
            lines.append("<b>매수 시그널: 없음</b>")

        lines.append("")

        if not meta["btc_gate"]:
            lines.append("<b>** BTC Gate 실패 - 모든 보유 종목 청산 권고 **</b>")
        else:
            # 시그널은 있지만 볼륨 필터 미통과
            volume_filtered = [
                (s, d)
                for s, d in signals.items()
                if s != "_META"
                and d.get("signal_count", 0) >= meta["min_signals"]
                and not d.get("in_volume_top")
            ]
            if volume_filtered:
                lines.append(f"<b>볼륨 필터 미통과 ({len(volume_filtered)}개):</b>")
                for symbol, data in volume_filtered[:5]:
                    lines.append(
                        f"  [{data['signal_count']}/7] {symbol}: {data['price']:,.0f}원"
                    )

        return "\n".join(lines)

    def send_daily_alert(self) -> bool:
        """일간 시그널 알림 전송"""
        logger.info("[DailySignalAlert] Calculating signals...")

        signals = self.calculate_signals()

        if not signals:
            logger.error("[DailySignalAlert] No signals to send")
            return False

        message = self.format_telegram_message(signals)

        logger.info("[DailySignalAlert] Sending Telegram notification...")
        success = self.notifier.send_message_sync(message)

        if success:
            logger.info("[DailySignalAlert] Alert sent successfully")
        else:
            logger.warning("[DailySignalAlert] Failed to send alert")

        return success

    def get_signal_summary(self) -> dict:
        """시그널 요약 정보 반환 (대시보드용)"""
        signals = self.calculate_signals()

        if not signals or "_META" not in signals:
            return {"error": True, "message": "No signals calculated"}

        meta = signals["_META"]
        entry_count = sum(
            1 for s, d in signals.items() if s != "_META" and d.get("entry_signal")
        )

        return {
            "error": False,
            "timestamp": meta["timestamp"],
            "btc_gate": meta["btc_gate"],
            "btc_price": meta["btc_price"],
            "btc_ma": meta["btc_ma"],
            "total_symbols": len(signals) - 1,
            "entry_signals": entry_count,
            "strategy": f"{meta['strategy']} v{meta['version']}",
            "min_signals": meta["min_signals"],
            "volume_filter_pct": meta["volume_filter_pct"],
            "signals": {s: d for s, d in signals.items() if s != "_META"},
        }


def main():
    """CLI 실행"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("MASP Daily Signal Alert Service")
    print("Sept_v3_RSI50_Gate Strategy")
    print("=" * 60)

    service = DailySignalAlertService()

    summary = service.get_signal_summary()

    if summary.get("error"):
        print(f"Error: {summary.get('message')}")
        return

    print(f"\nTimestamp: {summary['timestamp']}")
    print(f"Strategy: {summary['strategy']}")
    print(
        f"Settings: 7중 OR (min {summary['min_signals']}), Vol Top {summary['volume_filter_pct']*100:.0f}%"
    )
    print(f"\nBTC Gate: {'PASS' if summary['btc_gate'] else 'FAIL'}")
    print(f"BTC Price: {summary['btc_price']:,.0f}")
    print(f"BTC MA30: {summary['btc_ma']:,.0f}")
    print(f"\nTotal Symbols: {summary['total_symbols']}")
    print(f"Entry Signals: {summary['entry_signals']}")

    print("\n" + "-" * 60)
    print("Entry Signals (7중 OR + Vol Top 30%):")
    print("-" * 60)

    entry_list = [
        (s, d) for s, d in summary["signals"].items() if d.get("entry_signal")
    ]
    entry_list.sort(key=lambda x: x[1]["change_7d"], reverse=True)

    for symbol, data in entry_list:
        active = [k for k, v in data["signals"].items() if v]
        print(
            f"  {symbol:15} | {data['price']:>12,.0f} | "
            f"{data['signal_count']}/7 signals | "
            f"7D: {data['change_7d']:+6.1f}%"
        )
        print(f"    Active: {', '.join(active)}")

    print("\n" + "=" * 60)
    if service.notifier.enabled:
        response = input("Send Telegram notification? (y/n): ")
        if response.lower() == "y":
            success = service.send_daily_alert()
            print(f"Telegram sent: {success}")
    else:
        print("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")


if __name__ == "__main__":
    main()
