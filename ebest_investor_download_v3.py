"""
이베스트 OpenAPI로 수급 데이터 다운로드 (v3)
- 코스닥 종목 외국인/기관 매매동향
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd
from ebest import OpenApi
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

# 코스닥 주요 종목 (테스트용 30개)
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
    ("042660", "한화오션"),
    ("067160", "아프리카TV"),
    ("064760", "티씨케이"),
    ("137310", "에스디바이오센서"),
]


async def get_investor_trend(api, ticker, start_date, end_date):
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
                if data and len(data) > 0 and data[0].get("date", ""):
                    df = pd.DataFrame(data)
                    return df

    except Exception as e:
        pass

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
        pass

    return None


async def main():
    logger.info("=" * 60)
    logger.info("이베스트 OpenAPI 수급 데이터 다운로드")
    logger.info("=" * 60)

    api = OpenApi()

    # 로그인
    login_result = await api.login(APPKEY, APPSECRET)
    if not login_result:
        logger.error(f"로그인 실패: {api.last_message}")
        return

    logger.info("로그인 성공!")

    # 기간 설정 (최근 1년)
    end_date = "20250128"  # 어제 날짜 (과거 데이터)
    start_date = "20240101"

    logger.info(f"기간: {start_date} ~ {end_date}")
    logger.info(f"종목 수: {len(KOSDAQ_TICKERS)}개")

    # 결과 저장
    all_data = {}
    success_count = 0

    for ticker, name in tqdm(KOSDAQ_TICKERS, desc="수급 데이터 다운로드"):
        # t1702 시도
        df = await get_investor_trend(api, ticker, start_date, end_date)

        if df is not None and len(df) > 0:
            all_data[ticker] = df
            success_count += 1
            logger.info(f"{name}({ticker}): {len(df)}일 데이터")

            # 파일 저장
            df.to_csv(
                f"{SAVE_PATH}/{ticker}_investor.csv", index=False, encoding="utf-8-sig"
            )
        else:
            # t1764 시도
            df = await get_foreign_trend_t1764(api, ticker, start_date, end_date)
            if df is not None and len(df) > 0:
                all_data[ticker] = df
                success_count += 1
                df.to_csv(
                    f"{SAVE_PATH}/{ticker}_foreign.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

        # API 속도 제한
        await asyncio.sleep(0.3)

    # 결과 출력
    logger.info("\n" + "=" * 60)
    logger.info("다운로드 결과")
    logger.info("=" * 60)
    logger.info(f"성공: {success_count}개 / {len(KOSDAQ_TICKERS)}개")

    if success_count > 0:
        # 샘플 데이터 출력
        sample_ticker = list(all_data.keys())[0]
        sample_df = all_data[sample_ticker]
        logger.info(f"\n샘플 데이터 ({sample_ticker}):")
        logger.info(f"컬럼: {list(sample_df.columns)}")
        print(sample_df.head(10).to_string())

        # 전체 데이터 합치기
        combined = []
        for ticker, df in all_data.items():
            df["ticker"] = ticker
            combined.append(df)

        if combined:
            combined_df = pd.concat(combined, ignore_index=True)
            combined_df.to_csv(
                f"{SAVE_PATH}/all_investor_data.csv", index=False, encoding="utf-8-sig"
            )
            logger.info(f"\n전체 데이터 저장: {SAVE_PATH}/all_investor_data.csv")

    logger.info(f"\n저장 위치: {SAVE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
