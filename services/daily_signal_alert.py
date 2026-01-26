"""
Daily Signal Alert Service

매일 KAMA5/TSMOM90/MA30 전략 시그널을 계산하고 텔레그램으로 알림을 보냅니다.

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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / '.env')

from libs.notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

# Data paths
DATA_ROOT = Path('E:/data/crypto_ohlcv')


class DailySignalAlertService:
    """
    일간 시그널 알림 서비스

    KAMA5/TSMOM90/MA30 OR_LOOSE 전략의 시그널을 계산하고
    텔레그램으로 매일 알림을 전송합니다.
    """

    def __init__(
        self,
        kama_period: int = 5,
        tsmom_lookback: int = 90,
        btc_ma_period: int = 30,
        top_n: int = 20,
        exchange: str = "upbit",
    ):
        self.kama_period = kama_period
        self.tsmom_lookback = tsmom_lookback
        self.btc_ma_period = btc_ma_period
        self.top_n = top_n
        self.exchange = exchange

        self.notifier = TelegramNotifier()

        logger.info(
            f"[DailySignalAlert] Initialized: KAMA{kama_period}/TSMOM{tsmom_lookback}/MA{btc_ma_period}"
        )

    def _calc_kama(self, prices: np.ndarray, period: int = 5) -> np.ndarray:
        """KAMA 계산"""
        n = len(prices)
        kama = np.full(n, np.nan)

        if n < period + 1:
            return kama

        kama[period - 1] = np.mean(prices[:period])
        fast_sc = 2 / 3  # fast=2
        slow_sc = 2 / 31  # slow=30

        for i in range(period, n):
            change = abs(prices[i] - prices[i - period])
            volatility = np.sum(np.abs(np.diff(prices[i - period:i + 1])))
            er = change / volatility if volatility > 0 else 0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (prices[i] - kama[i - 1])

        return kama

    def _calc_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """SMA 계산"""
        result = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i - period + 1:i + 1])
        return result

    def _calc_tsmom(self, prices: np.ndarray, period: int) -> np.ndarray:
        """TSMOM 시그널 계산"""
        n = len(prices)
        signal = np.zeros(n, dtype=bool)
        for i in range(period, n):
            signal[i] = prices[i] > prices[i - period]
        return signal

    def load_ohlcv(self, min_days: int = 100) -> Dict[str, pd.DataFrame]:
        """OHLCV 데이터 로드"""
        folder = DATA_ROOT / f'{self.exchange}_1d'
        if not folder.exists():
            logger.warning(f"[DailySignalAlert] Data folder not found: {folder}")
            return {}

        data = {}
        for f in folder.glob('*.csv'):
            try:
                df = pd.read_csv(f)
                date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
                if not date_col:
                    continue

                df['date'] = pd.to_datetime(df[date_col[0]]).dt.normalize()
                df = df.set_index('date').sort_index()
                df = df[~df.index.duplicated(keep='last')]

                required = ['open', 'high', 'low', 'close', 'volume']
                if not all(c in df.columns for c in required):
                    continue

                df = df[required]
                if len(df) >= min_days:
                    data[f.stem] = df
            except Exception as e:
                logger.debug(f"[DailySignalAlert] Failed to load {f}: {e}")
                continue

        return data

    def get_top_symbols(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """거래대금 기준 상위 N개 심볼 선택"""
        vols = [(s, (df['close'] * df['volume']).mean()) for s, df in data.items()]
        vols.sort(key=lambda x: x[1], reverse=True)
        return {s: data[s] for s, _ in vols[:self.top_n]}

    def calculate_signals(self) -> Dict[str, dict]:
        """
        모든 심볼에 대한 시그널 계산

        Returns:
            심볼별 시그널 정보 딕셔너리
        """
        # 데이터 로드
        data = self.load_ohlcv()
        if not data:
            logger.error("[DailySignalAlert] No data loaded")
            return {}

        # Top N 유니버스
        filtered = self.get_top_symbols(data)

        # BTC 데이터 찾기
        btc_data = None
        for k, v in data.items():
            if 'BTC' in k.upper() and 'DOWN' not in k.upper():
                btc_data = v
                break

        if btc_data is None:
            logger.error("[DailySignalAlert] BTC data not found")
            return {}

        # BTC Gate 계산
        btc_prices = btc_data['close'].values
        btc_ma = self._calc_sma(btc_prices, self.btc_ma_period)
        btc_gate = btc_prices[-1] > btc_ma[-1] if not np.isnan(btc_ma[-1]) else False

        signals = {}

        for symbol, df in filtered.items():
            prices = df['close'].values

            if len(prices) < max(self.kama_period + 10, self.tsmom_lookback + 1):
                continue

            # KAMA 시그널
            kama = self._calc_kama(prices, self.kama_period)
            kama_signal = prices[-1] > kama[-1] if not np.isnan(kama[-1]) else False

            # TSMOM 시그널
            tsmom_signal = prices[-1] > prices[-self.tsmom_lookback - 1] if len(prices) > self.tsmom_lookback else False

            # OR_LOOSE: (KAMA OR TSMOM) AND BTC_GATE
            entry_signal = (kama_signal or tsmom_signal) and btc_gate

            # 추가 정보
            price_change_1d = (prices[-1] - prices[-2]) / prices[-2] * 100 if len(prices) > 1 else 0
            price_change_7d = (prices[-1] - prices[-7]) / prices[-7] * 100 if len(prices) > 7 else 0

            signals[symbol] = {
                'price': prices[-1],
                'kama': kama[-1] if not np.isnan(kama[-1]) else 0,
                'kama_signal': kama_signal,
                'tsmom_signal': tsmom_signal,
                'entry_signal': entry_signal,
                'change_1d': price_change_1d,
                'change_7d': price_change_7d,
                'volume': df['volume'].iloc[-1],
            }

        # BTC Gate 정보 추가
        signals['_META'] = {
            'btc_gate': btc_gate,
            'btc_price': btc_prices[-1],
            'btc_ma': btc_ma[-1],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'kama_period': self.kama_period,
            'tsmom_lookback': self.tsmom_lookback,
            'btc_ma_period': self.btc_ma_period,
        }

        return signals

    def format_telegram_message(self, signals: Dict[str, dict]) -> str:
        """텔레그램 메시지 포맷팅 (한글)"""
        if not signals or '_META' not in signals:
            return "<b>[MASP] 시그널 오류</b>\n시그널을 계산할 수 없습니다"

        meta = signals['_META']

        # Header
        lines = [
            f"<b>[MASP] 일간 시그널 리포트</b>",
            f"시간: {meta['timestamp']}",
            f"전략: KAMA{meta['kama_period']}/TSMOM{meta['tsmom_lookback']}/MA{meta['btc_ma_period']}",
            "",
            f"<b>BTC Gate: {'통과' if meta['btc_gate'] else '실패'}</b>",
            f"BTC: {meta['btc_price']:,.0f}원 {'(MA 위)' if meta['btc_gate'] else '(MA 아래)'}",
            f"MA{meta['btc_ma_period']}: {meta['btc_ma']:,.0f}원",
            "",
        ]

        # Entry signals (buy)
        entry_signals = [
            (s, d) for s, d in signals.items()
            if s != '_META' and d['entry_signal']
        ]
        entry_signals.sort(key=lambda x: x[1]['change_7d'], reverse=True)

        if entry_signals:
            lines.append(f"<b>매수 시그널 ({len(entry_signals)}개):</b>")
            for symbol, data in entry_signals[:10]:  # Top 10
                kama_mark = "K" if data['kama_signal'] else ""
                tsmom_mark = "T" if data['tsmom_signal'] else ""
                signal_type = f"[{kama_mark}{tsmom_mark}]"
                lines.append(
                    f"  {signal_type} {symbol}: {data['price']:,.0f}원 "
                    f"(1일 {data['change_1d']:+.1f}% / 7일 {data['change_7d']:+.1f}%)"
                )
        else:
            lines.append("<b>매수 시그널: 없음</b>")

        lines.append("")

        # Exit candidates (no signal)
        if meta['btc_gate']:
            no_signal = [
                (s, d) for s, d in signals.items()
                if s != '_META' and not d['entry_signal']
            ]
            no_signal.sort(key=lambda x: x[1]['change_7d'])

            if no_signal:
                lines.append(f"<b>시그널 없음 ({len(no_signal)}개):</b>")
                for symbol, data in no_signal[:5]:  # Bottom 5
                    lines.append(
                        f"  {symbol}: {data['price']:,.0f}원 "
                        f"(1일 {data['change_1d']:+.1f}% / 7일 {data['change_7d']:+.1f}%)"
                    )
        else:
            lines.append("<b>** BTC Gate 실패 - 모든 보유 종목 청산 권고 **</b>")

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

        if not signals or '_META' not in signals:
            return {
                'error': True,
                'message': 'No signals calculated'
            }

        meta = signals['_META']
        entry_count = sum(1 for s, d in signals.items() if s != '_META' and d['entry_signal'])

        return {
            'error': False,
            'timestamp': meta['timestamp'],
            'btc_gate': meta['btc_gate'],
            'btc_price': meta['btc_price'],
            'btc_ma': meta['btc_ma'],
            'total_symbols': len(signals) - 1,
            'entry_signals': entry_count,
            'strategy': f"KAMA{meta['kama_period']}/TSMOM{meta['tsmom_lookback']}/MA{meta['btc_ma_period']}",
            'signals': {
                s: d for s, d in signals.items() if s != '_META'
            }
        }


def main():
    """CLI 실행"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("MASP Daily Signal Alert Service")
    print("=" * 60)

    service = DailySignalAlertService(
        kama_period=5,
        tsmom_lookback=90,
        btc_ma_period=30,
        top_n=20,
        exchange="upbit"
    )

    # 시그널 계산 및 출력
    summary = service.get_signal_summary()

    if summary.get('error'):
        print(f"Error: {summary.get('message')}")
        return

    print(f"\nTimestamp: {summary['timestamp']}")
    print(f"Strategy: {summary['strategy']}")
    print(f"\nBTC Gate: {'PASS' if summary['btc_gate'] else 'FAIL'}")
    print(f"BTC Price: {summary['btc_price']:,.0f}")
    print(f"BTC MA30: {summary['btc_ma']:,.0f}")
    print(f"\nTotal Symbols: {summary['total_symbols']}")
    print(f"Entry Signals: {summary['entry_signals']}")

    # Entry signals 출력
    print("\n" + "-" * 60)
    print("Entry Signals:")
    print("-" * 60)

    entry_list = [
        (s, d) for s, d in summary['signals'].items() if d['entry_signal']
    ]
    entry_list.sort(key=lambda x: x[1]['change_7d'], reverse=True)

    for symbol, data in entry_list:
        kama = "KAMA" if data['kama_signal'] else ""
        tsmom = "TSMOM" if data['tsmom_signal'] else ""
        signals_str = "+".join(filter(None, [kama, tsmom]))
        print(f"  {symbol:15} | {data['price']:>12,.0f} | {signals_str:12} | "
              f"1D: {data['change_1d']:+6.1f}% | 7D: {data['change_7d']:+6.1f}%")

    # 텔레그램 전송 확인
    print("\n" + "=" * 60)
    if service.notifier.enabled:
        response = input("Send Telegram notification? (y/n): ")
        if response.lower() == 'y':
            success = service.send_daily_alert()
            print(f"Telegram sent: {success}")
    else:
        print("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")


if __name__ == "__main__":
    main()
