"""
FinanceDataReader 투자자 데이터 테스트
"""

import logging

import FinanceDataReader as fdr
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_fdr_investor():
    """FDR 투자자 데이터 함수 테스트"""

    logger.info("=== FinanceDataReader 투자자 데이터 테스트 ===")

    # 1. StockListing으로 종목 정보 확인
    logger.info("\n1. 코스닥 종목 리스트")
    try:
        kosdaq = fdr.StockListing("KOSDAQ")
        logger.info(f"코스닥 종목 수: {len(kosdaq)}")
        logger.info(f"컬럼: {list(kosdaq.columns)}")

        # 에코프로비엠 찾기
        if "Name" in kosdaq.columns:
            eco = kosdaq[kosdaq["Name"].str.contains("에코프로", na=False)]
            if len(eco) > 0:
                print(eco.to_string())
    except Exception as e:
        logger.error(f"StockListing 오류: {e}")

    # 2. 개별 종목 데이터 조회
    logger.info("\n2. 에코프로비엠 (247540) OHLCV")
    try:
        df = fdr.DataReader("247540", "2025-01-01", "2026-01-28")
        logger.info(f"데이터 수: {len(df)}")
        logger.info(f"컬럼: {list(df.columns)}")
        if len(df) > 0:
            print(df.tail(10).to_string())
    except Exception as e:
        logger.error(f"DataReader 오류: {e}")

    # 3. 투자자별 매매동향 - KRX 크롤링 시도
    logger.info("\n3. KRX 투자자별 매매동향 시도")

    # fdr에는 투자자별 매매동향 함수가 따로 없음
    # 대신 KRX 데이터를 직접 가져오는 방법 시도

    # 4. ETF/ETN에서 투자자 데이터 확인
    logger.info("\n4. ETF 리스트")
    try:
        etf = fdr.StockListing("ETF/KR")
        logger.info(f"ETF 수: {len(etf)}")
        logger.info(f"컬럼: {list(etf.columns)}")
    except Exception as e:
        logger.error(f"ETF 리스트 오류: {e}")

    # 5. 인덱스 데이터 (코스닥 지수)
    logger.info("\n5. 코스닥 지수")
    try:
        kosdaq_idx = fdr.DataReader("KQ11", "2025-01-01", "2026-01-28")
        logger.info(f"코스닥 지수 데이터: {len(kosdaq_idx)}")
        logger.info(f"컬럼: {list(kosdaq_idx.columns)}")
        if len(kosdaq_idx) > 0:
            print(kosdaq_idx.tail(5).to_string())
    except Exception as e:
        logger.error(f"코스닥 지수 오류: {e}")

    logger.info("\n테스트 완료!")


if __name__ == "__main__":
    test_fdr_investor()
