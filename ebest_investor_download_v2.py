"""
이베스트 OpenAPI로 수급 데이터 다운로드 (v2)
- 응답 구조 확인 후 수정
"""

import asyncio
from ebest import OpenApi
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API 키
APPKEY = "PSksZHwKhxHl46ptBNkQOtlTKbBKjpnuHS9g"
APPSECRET = "XEbkKJLa9GgounyGHswizjdahW9t3AzW"

# 저장 경로
SAVE_PATH = "E:/투자/data/kosdaq_investor_data"
os.makedirs(SAVE_PATH, exist_ok=True)


async def main():
    logger.info("=" * 60)
    logger.info("이베스트 OpenAPI 수급 데이터 다운로드 v2")
    logger.info("=" * 60)

    api = OpenApi()

    # 로그인
    login_result = await api.login(APPKEY, APPSECRET)
    if not login_result:
        logger.error(f"로그인 실패: {api.last_message}")
        return

    logger.info("로그인 성공!")

    # 테스트 1: t1702 외국인기관별종목별동향
    logger.info("\n=== t1702 테스트 (삼성전자) ===")
    try:
        response = await api.request(
            "t1702",
            {
                "shcode": "005930",
                "todt": "20260129",
                "frdt": "20260101",
                "gubun": "0",
                "type": "0",
            }
        )

        # 응답 객체 확인
        logger.info(f"응답 타입: {type(response)}")
        logger.info(f"응답 내용: {response}")

        # ResponseValue 객체 속성 확인
        if response:
            logger.info(f"응답 속성: {dir(response)}")

            # body 속성이 있는지 확인
            if hasattr(response, 'body'):
                logger.info(f"body: {response.body}")

            # json이나 data 속성 확인
            if hasattr(response, 'json'):
                logger.info(f"json: {response.json}")

            if hasattr(response, 'data'):
                logger.info(f"data: {response.data}")

            # 딕셔너리로 변환 시도
            try:
                data_dict = dict(response)
                logger.info(f"dict 변환: {data_dict}")
            except:
                pass

            # 인덱싱 시도
            try:
                for key in ['t1702OutBlock', 't1702OutBlock1', 'OutBlock', 'OutBlock1']:
                    if hasattr(response, key):
                        logger.info(f"{key}: {getattr(response, key)}")
            except:
                pass

    except Exception as e:
        logger.error(f"t1702 오류: {e}")
        import traceback
        traceback.print_exc()

    # 테스트 2: t8413 주식차트 (일봉)
    logger.info("\n=== t8413 테스트 (삼성전자 일봉) ===")
    try:
        response = await api.request(
            "t8413",
            {
                "shcode": "005930",
                "gubun": "2",
                "sdate": "20260101",
                "edate": "20260129",
                "comp_yn": "N",
            }
        )

        logger.info(f"응답 타입: {type(response)}")

        if response:
            # 속성 확인
            for attr in ['body', 'data', 'json', 't8413OutBlock', 't8413OutBlock1']:
                if hasattr(response, attr):
                    val = getattr(response, attr)
                    logger.info(f"{attr}: {val}")

    except Exception as e:
        logger.error(f"t8413 오류: {e}")
        import traceback
        traceback.print_exc()

    # 테스트 3: t1764 외국인 순매수 추이
    logger.info("\n=== t1764 테스트 (외국인 순매수 추이) ===")
    try:
        response = await api.request(
            "t1764",
            {
                "shcode": "005930",
                "gubun1": "0",  # 0:일
                "gubun2": "0",  # 0:순매수량
                "gubun3": "1",  # 1:외국인
                "fromdt": "20260101",
                "todt": "20260129",
            }
        )

        logger.info(f"응답 타입: {type(response)}")
        logger.info(f"응답: {response}")

        if response and hasattr(response, 't1764OutBlock1'):
            data = response.t1764OutBlock1
            logger.info(f"데이터: {data}")

    except Exception as e:
        logger.error(f"t1764 오류: {e}")

    # 세션 종료
    await api.close() if hasattr(api, 'close') else None

    logger.info("\n테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())
