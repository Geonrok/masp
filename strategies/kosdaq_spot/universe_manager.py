# -*- coding: utf-8 -*-
"""
코스닥 현물 전략 - 유니버스 관리 모듈
시총 상위 1/8 종목 선정 및 관리
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class UniverseConfig:
    """유니버스 설정"""

    ohlcv_path: str = "E:/투자/data/kr_stock/kosdaq_ohlcv"
    foreign_path: str = "E:/투자/data/kosdaq_foreign"
    liquidity_percentile: float = 0.5  # 상위 50%
    marketcap_percentile: float = 0.125  # 상위 1/8
    min_data_days: int = 200


class KOSDAQUniverseManager:
    """
    코스닥 유니버스 관리자

    사용법:
        manager = KOSDAQUniverseManager()
        universe = manager.get_universe()
        manager.print_universe()
    """

    def __init__(self, config: Optional[UniverseConfig] = None):
        self.config = config or UniverseConfig()
        self._universe = None
        self._stock_info = None
        self._last_update = None

    def load_stock_info(self) -> Dict:
        """모든 종목 정보 로드"""
        if self._stock_info is not None:
            return self._stock_info

        self._stock_info = {}
        foreign_files = [
            f
            for f in os.listdir(self.config.foreign_path)
            if f.endswith("_foreign.csv")
        ]

        for f in foreign_files:
            ticker = f.replace("_foreign.csv", "")
            ohlcv_file = os.path.join(self.config.ohlcv_path, f"{ticker}.csv")

            if not os.path.exists(ohlcv_file):
                continue

            try:
                # 외국인 데이터 로드
                foreign_df = pd.read_csv(os.path.join(self.config.foreign_path, f))
                foreign_df["date"] = pd.to_datetime(foreign_df["dt"], format="%Y%m%d")
                foreign_df = foreign_df.sort_values("date").reset_index(drop=True)

                # OHLCV 로드
                ohlcv_df = pd.read_csv(ohlcv_file)
                ohlcv_df["date"] = pd.to_datetime(ohlcv_df["Date"])
                ohlcv_df = ohlcv_df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )

                # 병합
                merged = pd.merge(
                    ohlcv_df,
                    foreign_df[["date", "chg_qty", "wght", "poss_stkcnt"]],
                    on="date",
                    how="inner",
                )
                merged = merged.sort_values("date").reset_index(drop=True)
                merged = merged[merged["date"] >= "2018-01-01"].copy()

                if merged["wght"].dtype == object:
                    merged["wght"] = merged["wght"].str.replace("+", "").astype(float)

                if len(merged) >= self.config.min_data_days:
                    # 최근 1년 평균 거래대금 (유동성)
                    recent = merged[
                        merged["date"] >= merged["date"].max() - pd.Timedelta(days=365)
                    ]
                    avg_turnover = recent["volume"].mean() * recent["close"].mean()

                    # 시총 추정
                    latest = merged.iloc[-1]
                    if latest["wght"] > 0:
                        est_shares = latest["poss_stkcnt"] / (latest["wght"] / 100)
                        est_marketcap = est_shares * latest["close"]
                    else:
                        est_marketcap = 0

                    self._stock_info[ticker] = {
                        "liquidity": avg_turnover,
                        "marketcap": est_marketcap,
                        "latest_price": latest["close"],
                        "latest_wght": latest["wght"],
                        "data_days": len(merged),
                    }

            except Exception:
                pass

        self._last_update = datetime.now()
        return self._stock_info

    def get_universe(self, force_refresh: bool = False) -> List[str]:
        """
        유니버스 종목 리스트 반환

        Returns:
            시총 상위 1/8 종목 티커 리스트
        """
        if self._universe is not None and not force_refresh:
            return self._universe

        stock_info = self.load_stock_info()

        # 유동성 기준 상위 50%
        sorted_by_liquidity = sorted(
            stock_info.items(), key=lambda x: x[1]["liquidity"], reverse=True
        )
        cutoff = int(len(sorted_by_liquidity) * self.config.liquidity_percentile)
        liquid_tickers = [t[0] for t in sorted_by_liquidity[:cutoff]]

        # 시총 기준 상위 1/8 (유동성 필터 내에서)
        liquid_info = {
            t: stock_info[t]
            for t in liquid_tickers
            if t in stock_info and stock_info[t]["marketcap"] > 0
        }
        sorted_by_mcap = sorted(
            liquid_info.items(), key=lambda x: x[1]["marketcap"], reverse=True
        )
        mcap_cutoff = int(len(sorted_by_mcap) * self.config.marketcap_percentile)
        self._universe = [t[0] for t in sorted_by_mcap[:mcap_cutoff]]

        return self._universe

    def get_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """개별 종목 데이터 로드"""
        ohlcv_file = os.path.join(self.config.ohlcv_path, f"{ticker}.csv")
        foreign_file = os.path.join(self.config.foreign_path, f"{ticker}_foreign.csv")

        if not os.path.exists(ohlcv_file) or not os.path.exists(foreign_file):
            return None

        try:
            foreign_df = pd.read_csv(foreign_file)
            foreign_df["date"] = pd.to_datetime(foreign_df["dt"], format="%Y%m%d")
            foreign_df = foreign_df.sort_values("date").reset_index(drop=True)

            ohlcv_df = pd.read_csv(ohlcv_file)
            ohlcv_df["date"] = pd.to_datetime(ohlcv_df["Date"])
            ohlcv_df = ohlcv_df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            merged = pd.merge(
                ohlcv_df,
                foreign_df[["date", "chg_qty", "wght", "poss_stkcnt"]],
                on="date",
                how="inner",
            )
            merged = merged.sort_values("date").reset_index(drop=True)

            if merged["wght"].dtype == object:
                merged["wght"] = merged["wght"].str.replace("+", "").astype(float)

            return merged

        except Exception:
            return None

    def get_universe_stats(self) -> Dict:
        """유니버스 통계"""
        universe = self.get_universe()
        stock_info = self.load_stock_info()

        stats = {
            "total_stocks": len(universe),
            "last_update": str(self._last_update) if self._last_update else None,
            "avg_marketcap": 0,
            "avg_liquidity": 0,
            "avg_foreign_wght": 0,
        }

        if universe:
            mcaps = [stock_info[t]["marketcap"] for t in universe if t in stock_info]
            liquids = [stock_info[t]["liquidity"] for t in universe if t in stock_info]
            wghts = [stock_info[t]["latest_wght"] for t in universe if t in stock_info]

            stats["avg_marketcap"] = sum(mcaps) / len(mcaps) if mcaps else 0
            stats["avg_liquidity"] = sum(liquids) / len(liquids) if liquids else 0
            stats["avg_foreign_wght"] = sum(wghts) / len(wghts) if wghts else 0

        return stats

    def print_universe(self):
        """유니버스 출력"""
        universe = self.get_universe()
        stock_info = self.load_stock_info()
        stats = self.get_universe_stats()

        print("=" * 70)
        print("코스닥 현물 전략 유니버스")
        print("기준: 유동성 상위 50% 내 시총 상위 1/8")
        print("=" * 70)

        print(f"\n총 종목 수: {stats['total_stocks']}개")
        print(f"평균 시가총액: {stats['avg_marketcap']/1e9:.0f}억원")
        print(f"평균 외국인 비중: {stats['avg_foreign_wght']:.1f}%")

        print(f"\n{'순위':<4} {'티커':<10} {'시총(억)':<12} {'외국인비중':<10}")
        print("-" * 45)

        for i, ticker in enumerate(universe[:20], 1):
            info = stock_info.get(ticker, {})
            mcap = info.get("marketcap", 0) / 1e8
            wght = info.get("latest_wght", 0)
            print(f"{i:<4} {ticker:<10} {mcap:<12,.0f} {wght:<10.1f}%")

        if len(universe) > 20:
            print(f"... 외 {len(universe) - 20}개 종목")

        print("=" * 70)

    def save_universe(self, output_path: Optional[str] = None) -> str:
        """유니버스를 JSON으로 저장"""
        universe = self.get_universe()
        stock_info = self.load_stock_info()
        stats = self.get_universe_stats()

        data = {
            "created": datetime.now().isoformat(),
            "config": {
                "liquidity_percentile": self.config.liquidity_percentile,
                "marketcap_percentile": self.config.marketcap_percentile,
            },
            "stats": stats,
            "universe": [
                {
                    "ticker": t,
                    "marketcap": stock_info[t]["marketcap"],
                    "foreign_wght": stock_info[t]["latest_wght"],
                }
                for t in universe
                if t in stock_info
            ],
        }

        if output_path is None:
            output_path = "E:/투자/Multi-Asset Strategy Platform/strategies/kosdaq_spot/universe.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return output_path


def main():
    """테스트 실행"""
    manager = KOSDAQUniverseManager()
    manager.print_universe()
    manager.save_universe()


if __name__ == "__main__":
    main()
