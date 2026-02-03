"""
이베스트 간단 테스트 - 작동하는 TR 찾기
"""

import asyncio
import logging

from ebest import OpenApi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

APPKEY = "PSksZHwKhxHl46ptBNkQOtlTKbBKjpnuHS9g"
APPSECRET = "XEbkKJLa9GgounyGHswizjdahW9t3AzW"


async def main():
    api = OpenApi()

    login_result = await api.login(APPKEY, APPSECRET)
    logger.info(f"로그인: {login_result}")
    logger.info(
        f"모의투자: {api.is_simulation if hasattr(api, 'is_simulation') else 'unknown'}"
    )

    # 1. t8436 - 주식종목조회 API (전종목)
    logger.info("\n=== t8436 주식종목조회 ===")
    response = await api.request("t8436", {"gubun": "1"})  # 1:코스닥
    if response and hasattr(response, "body"):
        body = response.body
        if "t8436OutBlock" in body:
            data = body["t8436OutBlock"]
            logger.info(f"코스닥 종목 수: {len(data)}")
            if len(data) > 0:
                logger.info(f"첫 번째: {data[0]}")
                logger.info(f"마지막: {data[-1]}")

    await asyncio.sleep(0.5)

    # 2. t1101 - 주식현재가호가 (실시간 데이터가 필요 없는)
    logger.info("\n=== t1101 주식현재가호가 (삼성전자) ===")
    response = await api.request("t1101", {"shcode": "005930"})
    if response and hasattr(response, "body"):
        body = response.body
        for key in body:
            if "OutBlock" in key:
                logger.info(f"{key}: {body[key]}")

    await asyncio.sleep(0.5)

    # 3. t1102 - 주식현재가 (삼성전자)
    logger.info("\n=== t1102 주식현재가 (삼성전자) ===")
    response = await api.request("t1102", {"shcode": "005930"})
    if response and hasattr(response, "body"):
        body = response.body
        for key in body:
            if "OutBlock" in key:
                data = body[key]
                if isinstance(data, dict):
                    # 주요 필드만 출력
                    important = {
                        k: v
                        for k, v in data.items()
                        if k
                        in [
                            "hname",
                            "price",
                            "sign",
                            "change",
                            "diff",
                            "volume",
                            "jnilclose",
                            "offerho1",
                            "bidho1",
                            "high",
                            "low",
                            "open",
                            "shcode",
                            "exhratio",
                        ]
                    }
                    logger.info(f"{key}: {important}")

    await asyncio.sleep(0.5)

    # 4. t1104 - 종목현재가 다중조회
    logger.info("\n=== t1104 멀티 현재가 ===")
    response = await api.request(
        "t1104",
        {
            "code1": "247540",  # 에코프로비엠
            "code2": "196170",  # 알테오젠
            "code3": "263750",  # 펄어비스
        },
    )
    if response and hasattr(response, "body"):
        body = response.body
        for key in body:
            if "OutBlock" in key:
                data = body[key]
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"{key}: {len(data)}개")
                    for item in data:
                        if item.get("hname"):
                            logger.info(f"  {item.get('hname')}: {item.get('price')}")

    await asyncio.sleep(0.5)

    # 5. t8424 - 업종전종목조회
    logger.info("\n=== t8424 업종전종목 (코스닥) ===")
    response = await api.request("t8424", {"gubun1": "1"})  # 1:코스닥
    if response and hasattr(response, "body"):
        body = response.body
        if "t8424OutBlock" in body:
            data = body["t8424OutBlock"]
            logger.info(f"종목 수: {len(data)}")
            if len(data) > 0:
                # 에코프로비엠 찾기
                for item in data:
                    if "에코프로" in item.get("hname", ""):
                        logger.info(f"  {item}")

    await asyncio.sleep(0.5)

    # 6. t1514 - 업종기간별추이 (투자자별 매매동향 포함)
    logger.info("\n=== t1514 업종기간별추이 (코스닥) ===")
    response = await api.request(
        "t1514",
        {
            "upcode": "001",  # 코스닥 지수
            "gubun2": "2",  # 2:일별
            "cts_date": "",
            "cnt": "20",
            "gubun3": "0",  # 0:전체
        },
    )
    if response and hasattr(response, "body"):
        body = response.body
        for key in body:
            if "OutBlock" in key:
                data = body[key]
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"{key}: {len(data)}개")
                    logger.info(f"  첫 번째: {data[0]}")
                elif isinstance(data, dict) and any(v for v in data.values() if v):
                    logger.info(f"{key}: {data}")

    await asyncio.sleep(0.5)

    # 7. t1516 - 업종별종목시세 (투자자별 정보 포함?)
    logger.info("\n=== t1516 업종별종목시세 ===")
    response = await api.request(
        "t1516",
        {
            "upcode": "001",  # 코스닥
            "gubun": "1",  # 1:시가총액
            "shcode": "",
        },
    )
    if response and hasattr(response, "body"):
        body = response.body
        for key in body:
            if "OutBlock" in key:
                data = body[key]
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"{key}: {len(data)}개")
                    logger.info(f"  컬럼: {list(data[0].keys())}")
                    logger.info(f"  첫 번째: {data[0]}")

    # 8. KOSDAQ/KOSPI 가격 데이터 - t9945
    logger.info("\n=== t9945 코스닥100 종목 ===")
    response = await api.request("t9945", {"gubun": "K"})  # K:코스닥100
    if response and hasattr(response, "body"):
        body = response.body
        if "t9945OutBlock" in body:
            data = body["t9945OutBlock"]
            logger.info(f"종목 수: {len(data)}")
            if len(data) > 0:
                logger.info(f"  첫 번째: {data[0]}")

    logger.info("\n테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())
