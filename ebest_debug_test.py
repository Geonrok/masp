"""
이베스트 API 상세 디버그 테스트
"""

import asyncio
import logging

from ebest import OpenApi

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

APPKEY = "PSksZHwKhxHl46ptBNkQOtlTKbBKjpnuHS9g"
APPSECRET = "XEbkKJLa9GgounyGHswizjdahW9t3AzW"


async def main():
    api = OpenApi()

    login_result = await api.login(APPKEY, APPSECRET)
    logger.info(f"로그인: {login_result}")

    # 테스트 1: 삼성전자 (코스피)
    logger.info("\n=== 삼성전자 (005930) t1702 ===")
    response = await api.request(
        "t1702",
        {
            "shcode": "005930",
            "todt": "20250127",
            "frdt": "20250120",
            "gubun": "0",
            "type": "0",
        },
    )

    if response and hasattr(response, "body"):
        body = response.body
        logger.info(f"응답 코드: {body.get('rsp_cd')}")
        logger.info(f"응답 메시지: {body.get('rsp_msg')}")

        if "t1702OutBlock1" in body:
            data = body["t1702OutBlock1"]
            logger.info(f"데이터 수: {len(data)}")
            if data:
                logger.info(f"첫 번째 데이터: {data[0]}")

    await asyncio.sleep(0.5)

    # 테스트 2: 다른 TR (t8413 일봉)
    logger.info("\n=== 삼성전자 일봉 (t8413) ===")
    response = await api.request(
        "t8413",
        {
            "shcode": "005930",
            "gubun": "2",
            "qrycnt": 10,
            "sdate": "20250120",
            "edate": "20250127",
            "comp_yn": "N",
        },
    )

    if response and hasattr(response, "body"):
        body = response.body
        logger.info(f"body 키: {body.keys()}")

        for key in body:
            if "OutBlock" in key:
                data = body[key]
                logger.info(f"{key}: {len(data) if isinstance(data, list) else data}")
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"첫 번째: {data[0]}")

    await asyncio.sleep(0.5)

    # 테스트 3: t1305 기간별 시세
    logger.info("\n=== 기간별 시세 (t1305) ===")
    response = await api.request(
        "t1305",
        {
            "shcode": "005930",
            "dwmcode": "1",  # 1:일
            "cnt": 10,
        },
    )

    if response and hasattr(response, "body"):
        body = response.body
        logger.info(f"body 키: {body.keys()}")

        for key in body:
            if "OutBlock" in key:
                data = body[key]
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"{key}: {len(data)}개 데이터")
                    logger.info(f"컬럼: {data[0].keys() if data else '없음'}")
                    logger.info(f"첫 번째: {data[0]}")

    await asyncio.sleep(0.5)

    # 테스트 4: t1601 투자자별 종합
    logger.info("\n=== 투자자별 종합 (t1601) ===")
    response = await api.request(
        "t1601",
        {
            "gubun": "0",  # 0:코스피, 1:코스닥
            "fession": "0",  # 0:수량, 1:금액
        },
    )

    if response and hasattr(response, "body"):
        body = response.body
        logger.info(f"body 키: {body.keys()}")

        for key in body:
            if "OutBlock" in key:
                data = body[key]
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"{key}: {len(data)}개 데이터")
                    logger.info(f"첫 번째: {data[0]}")
                elif isinstance(data, dict):
                    logger.info(f"{key}: {data}")

    logger.info("\n디버그 완료!")


if __name__ == "__main__":
    asyncio.run(main())
