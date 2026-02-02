"""
이베스트 OpenAPI로 수급 데이터 다운로드
- 외국인/기관 매매동향
"""

import asyncio
from ebest import OpenApi
import pandas as pd
import logging
from datetime import datetime, timedelta
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API 키
APPKEY = "PSksZHwKhxHl46ptBNkQOtlTKbBKjpnuHS9g"
APPSECRET = "XEbkKJLa9GgounyGHswizjdahW9t3AzW"

# 저장 경로
SAVE_PATH = "E:/투자/data/kosdaq_investor_data"
os.makedirs(SAVE_PATH, exist_ok=True)


async def test_connection():
    """API 연결 테스트"""
    logger.info("이베스트 OpenAPI 연결 테스트...")

    api = OpenApi()

    # 로그인 (실거래)
    login_result = await api.login(APPKEY, APPSECRET)

    if login_result:
        logger.info("로그인 성공!")
        logger.info(f"모의투자 여부: {api.is_simulation}")
        return api
    else:
        logger.error(f"로그인 실패: {api.last_message}")
        return None


async def get_foreign_trend(api, ticker, start_date, end_date):
    """
    t1702: 외국인기관별종목별동향
    외국인/기관의 일별 순매수 데이터
    """
    try:
        # t1702 TR 요청
        response = await api.request(
            "t1702",
            {
                "shcode": ticker,
                "todt": end_date.replace('-', ''),
                "frdt": start_date.replace('-', ''),
                "gubun": "0",  # 0:일간
                "type": "0",   # 0:금액
            }
        )

        if response and 't1702OutBlock1' in response:
            data = response['t1702OutBlock1']
            df = pd.DataFrame(data)
            logger.info(f"{ticker}: {len(df)}일 데이터 수신")
            return df
        else:
            logger.warning(f"{ticker}: 데이터 없음")
            return None

    except Exception as e:
        logger.error(f"{ticker} 오류: {e}")
        return None


async def get_investor_ranking(api, market="2", investor="9000"):
    """
    t1717: 외국인기관 순매매 상위종목
    market: 1=코스피, 2=코스닥
    investor: 9000=외국인, 9001=기관
    """
    try:
        response = await api.request(
            "t1717",
            {
                "gubun": market,
                "fromdt": (datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
                "todt": datetime.now().strftime("%Y%m%d"),
                "pression": investor,
                "cnt": "100",
            }
        )

        if response and 't1717OutBlock' in response:
            data = response['t1717OutBlock']
            df = pd.DataFrame(data)
            logger.info(f"순매매 상위 {len(df)}개 종목")
            return df

    except Exception as e:
        logger.error(f"t1717 오류: {e}")

    return None


async def get_stock_chart(api, ticker, start_date, end_date):
    """
    t8413: 주식챠트 (일봉)
    """
    try:
        response = await api.request(
            "t8413",
            {
                "shcode": ticker,
                "gubun": "2",  # 2=일봉
                "sdate": start_date.replace('-', ''),
                "edate": end_date.replace('-', ''),
                "comp_yn": "N",
            }
        )

        if response and 't8413OutBlock1' in response:
            data = response['t8413OutBlock1']
            df = pd.DataFrame(data)
            return df

    except Exception as e:
        logger.error(f"t8413 오류: {e}")

    return None


async def main():
    logger.info("=" * 60)
    logger.info("이베스트 OpenAPI 수급 데이터 다운로드")
    logger.info("=" * 60)

    # 연결
    api = await test_connection()
    if api is None:
        return

    # 테스트 종목
    test_tickers = [
        ('005930', '삼성전자'),
        ('247540', '에코프로비엠'),
        ('035720', '카카오'),
    ]

    # 기간 설정
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    logger.info(f"\n기간: {start_date} ~ {end_date}")

    # 1. 외국인기관 순매매 상위 (코스닥)
    logger.info("\n=== 외국인 순매매 상위 (코스닥) ===")
    foreign_top = await get_investor_ranking(api, market="2", investor="9000")
    if foreign_top is not None:
        print(foreign_top.head(10).to_string())
        foreign_top.to_csv(f"{SAVE_PATH}/kosdaq_foreign_top.csv", index=False, encoding='utf-8-sig')

    await asyncio.sleep(1)

    # 2. 기관 순매매 상위 (코스닥)
    logger.info("\n=== 기관 순매매 상위 (코스닥) ===")
    inst_top = await get_investor_ranking(api, market="2", investor="9001")
    if inst_top is not None:
        print(inst_top.head(10).to_string())
        inst_top.to_csv(f"{SAVE_PATH}/kosdaq_inst_top.csv", index=False, encoding='utf-8-sig')

    await asyncio.sleep(1)

    # 3. 개별 종목 외국인/기관 동향
    logger.info("\n=== 개별 종목 외국인/기관 동향 ===")
    for ticker, name in test_tickers:
        logger.info(f"\n--- {name} ({ticker}) ---")

        df = await get_foreign_trend(api, ticker, start_date, end_date)
        if df is not None and len(df) > 0:
            print(f"컬럼: {list(df.columns)}")
            print(df.head(5).to_string())
            df.to_csv(f"{SAVE_PATH}/{ticker}_investor.csv", index=False, encoding='utf-8-sig')

        await asyncio.sleep(0.5)

    # 4. 주식 차트 테스트
    logger.info("\n=== 주식 차트 테스트 ===")
    chart = await get_stock_chart(api, '005930', start_date, end_date)
    if chart is not None:
        print(f"차트 컬럼: {list(chart.columns)}")
        print(chart.head(3).to_string())

    logger.info("\n다운로드 완료!")
    logger.info(f"저장 위치: {SAVE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
