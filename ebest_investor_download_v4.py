"""
이베스트 OpenAPI로 수급 데이터 다운로드 (v4)
- 날짜 수정 (2026년)
- t1702 개별종목 + t1601 시장전체
"""

import asyncio
from ebest import OpenApi
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# API 키
APPKEY = "PSksZHwKhxHl46ptBNkQOtlTKbBKjpnuHS9g"
APPSECRET = "XEbkKJLa9GgounyGHswizjdahW9t3AzW"

# 저장 경로
SAVE_PATH = "E:/투자/data/kosdaq_investor_data"
os.makedirs(SAVE_PATH, exist_ok=True)

# 코스닥 주요 종목
KOSDAQ_TICKERS = [
    ("247540", "에코프로비엠"),
    ("196170", "알테오젠"),
    ("263750", "펄어비스"),
    ("091990", "셀트리온헬스케어"),
    ("293490", "카카오게임즈"),
    ("068270", "셀트리온제약"),
    ("028300", "HLB"),
    ("042700", "한미반도체"),
    ("095340", "ISC"),
    ("086520", "에코프로"),
    ("251270", "넷마블"),
    ("112040", "위메이드"),
    ("141080", "레고켐바이오"),
    ("039030", "이오테크닉스"),
    ("036540", "SFA반도체"),
    ("058470", "리노공업"),
    ("041510", "에스엠"),
    ("053800", "안랩"),
    ("214150", "클래시스"),
    ("145020", "휴젤"),
    ("060310", "3S"),
    ("035900", "JYP Ent"),
    ("066970", "엘앤에프"),
    ("053610", "프로텍"),
    ("034230", "파라다이스"),
    ("215200", "메가스터디교육"),
    ("042660", "대우조선해양"),
    ("067160", "아프리카TV"),
    ("064760", "티씨케이"),
    ("137310", "에스디바이오센서"),
]


async def get_investor_trend_t1702(api, ticker, start_date, end_date):
    """
    t1702: 외국인기관별종목별동향
    """
    try:
        response = await api.request(
            "t1702",
            {
                "shcode": ticker,
                "todt": end_date,
                "frdt": start_date,
                "gubun": "0",  # 0:일간
                "type": "0",  # 0:금액
            },
        )

        if response and hasattr(response, "body"):
            body = response.body
            if "t1702OutBlock1" in body:
                data = body["t1702OutBlock1"]
                if data and len(data) > 0:
                    # 유효한 데이터인지 확인
                    first = data[0]
                    if first.get("date", "") and first.get("date", "") != "":
                        df = pd.DataFrame(data)
                        return df

    except Exception as e:
        logger.debug(f"t1702 오류 ({ticker}): {e}")

    return None


async def get_foreign_trend_t1764(api, ticker, start_date, end_date):
    """
    t1764: 외국인 순매수 추이
    """
    try:
        response = await api.request(
            "t1764",
            {
                "shcode": ticker,
                "gubun1": "0",  # 0:일
                "gubun2": "0",  # 0:순매수량
                "gubun3": "1",  # 1:외국인
                "fromdt": start_date,
                "todt": end_date,
            },
        )

        if response and hasattr(response, "body"):
            body = response.body
            if "t1764OutBlock1" in body:
                data = body["t1764OutBlock1"]
                if data and len(data) > 0:
                    df = pd.DataFrame(data)
                    return df

    except Exception as e:
        logger.debug(f"t1764 오류 ({ticker}): {e}")

    return None


async def get_market_investor_t1601(api, market="1"):
    """
    t1601: 투자자별 종합 (시장 전체)
    market: 0=코스피, 1=코스닥
    """
    try:
        response = await api.request(
            "t1601",
            {
                "gubun": market,  # 1:코스닥
                "fession": "0",  # 0:수량
            },
        )

        if response and hasattr(response, "body"):
            body = response.body
            result = {}

            for key in body:
                if "OutBlock" in key:
                    result[key] = body[key]

            return result

    except Exception as e:
        logger.error(f"t1601 오류: {e}")

    return None


async def get_stock_chart_t8413(api, ticker, start_date, end_date):
    """
    t8413: 주식차트 (일봉) - 가격 데이터
    """
    try:
        response = await api.request(
            "t8413",
            {
                "shcode": ticker,
                "gubun": "2",  # 2:일봉
                "qrycnt": 500,
                "sdate": start_date,
                "edate": end_date,
                "comp_yn": "N",
            },
        )

        if response and hasattr(response, "body"):
            body = response.body
            if "t8413OutBlock1" in body:
                data = body["t8413OutBlock1"]
                if data and len(data) > 0:
                    df = pd.DataFrame(data)
                    return df

    except Exception as e:
        logger.debug(f"t8413 오류 ({ticker}): {e}")

    return None


async def main():
    logger.info("=" * 60)
    logger.info("이베스트 OpenAPI 수급 데이터 다운로드 v4")
    logger.info("=" * 60)

    api = OpenApi()

    # 로그인
    login_result = await api.login(APPKEY, APPSECRET)
    if not login_result:
        logger.error(f"로그인 실패: {api.last_message}")
        return

    logger.info("로그인 성공!")

    # 기간 설정 (2026년 - 현재 날짜 기준)
    today = datetime.now()
    end_date = (today - timedelta(days=1)).strftime("%Y%m%d")  # 어제
    start_date = (today - timedelta(days=365)).strftime("%Y%m%d")  # 1년 전

    logger.info(f"기간: {start_date} ~ {end_date}")

    # 1. 시장 전체 투자자 동향 (코스닥)
    logger.info("\n=== 코스닥 시장 전체 투자자 동향 (t1601) ===")
    market_data = await get_market_investor_t1601(api, market="1")

    if market_data:
        for key, value in market_data.items():
            logger.info(f"{key}: {value}")

        # 저장
        import json

        with open(
            f"{SAVE_PATH}/kosdaq_market_investor.json", "w", encoding="utf-8"
        ) as f:
            json.dump(market_data, f, ensure_ascii=False, indent=2)
        logger.info(f"시장 데이터 저장: {SAVE_PATH}/kosdaq_market_investor.json")

    await asyncio.sleep(0.5)

    # 2. 개별 종목 테스트 (첫 3개만)
    logger.info("\n=== 개별 종목 테스트 ===")
    test_tickers = KOSDAQ_TICKERS[:3]

    for ticker, name in test_tickers:
        logger.info(f"\n--- {name} ({ticker}) ---")

        # t1702 시도
        df = await get_investor_trend_t1702(api, ticker, start_date, end_date)
        if df is not None:
            logger.info(f"t1702 성공: {len(df)}일 데이터")
            logger.info(f"컬럼: {list(df.columns)}")
            print(df.head(3).to_string())
        else:
            logger.info("t1702: 데이터 없음")

            # t1764 시도
            df = await get_foreign_trend_t1764(api, ticker, start_date, end_date)
            if df is not None:
                logger.info(f"t1764 성공: {len(df)}일 데이터")
                print(df.head(3).to_string())
            else:
                logger.info("t1764: 데이터 없음")

        # 차트 데이터
        chart = await get_stock_chart_t8413(api, ticker, start_date, end_date)
        if chart is not None:
            logger.info(f"t8413 차트: {len(chart)}일 데이터")
        else:
            logger.info("t8413: 데이터 없음")

        await asyncio.sleep(0.3)

    # 3. 전체 종목 다운로드 (성공 시)
    logger.info("\n=== 전체 종목 다운로드 ===")
    all_data = {}
    success_count = 0

    for ticker, name in tqdm(KOSDAQ_TICKERS, desc="다운로드"):
        df = await get_investor_trend_t1702(api, ticker, start_date, end_date)

        if df is None:
            df = await get_foreign_trend_t1764(api, ticker, start_date, end_date)

        if df is not None and len(df) > 0:
            all_data[ticker] = df
            success_count += 1
            df.to_csv(
                f"{SAVE_PATH}/{ticker}_investor.csv", index=False, encoding="utf-8-sig"
            )

        await asyncio.sleep(0.2)

    # 결과
    logger.info("\n" + "=" * 60)
    logger.info("다운로드 결과")
    logger.info("=" * 60)
    logger.info(f"성공: {success_count}개 / {len(KOSDAQ_TICKERS)}개")

    if success_count > 0:
        # 전체 합치기
        combined = []
        for ticker, df in all_data.items():
            df["ticker"] = ticker
            combined.append(df)

        if combined:
            combined_df = pd.concat(combined, ignore_index=True)
            combined_df.to_csv(
                f"{SAVE_PATH}/all_investor_data.csv", index=False, encoding="utf-8-sig"
            )
            logger.info(f"전체 데이터 저장: {SAVE_PATH}/all_investor_data.csv")
            logger.info(f"총 레코드: {len(combined_df)}")

    logger.info(f"\n저장 위치: {SAVE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
