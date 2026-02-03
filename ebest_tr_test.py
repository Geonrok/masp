"""
이베스트 TR 코드 테스트
- 다양한 TR 코드로 수급 데이터 조회
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


async def test_tr(api, tr_code, params, description):
    """TR 코드 테스트"""
    logger.info(f"\n=== {description} ({tr_code}) ===")
    try:
        response = await api.request(tr_code, params)

        if response and hasattr(response, "body"):
            body = response.body
            logger.info(f"응답 키: {list(body.keys())}")

            for key in body:
                if "OutBlock" in key:
                    data = body[key]
                    if isinstance(data, list):
                        logger.info(f"{key}: {len(data)}개 데이터")
                        if len(data) > 0:
                            logger.info(f"  첫 번째: {data[0]}")
                            return True
                    elif isinstance(data, dict):
                        # 값이 있는지 확인
                        has_value = any(
                            v for v in data.values() if v and v != "" and v != 0
                        )
                        if has_value:
                            logger.info(f"{key}: {data}")
                            return True
                        else:
                            logger.info(f"{key}: (빈 데이터)")
        else:
            logger.info("응답 없음")

    except Exception as e:
        logger.error(f"오류: {e}")

    return False


async def main():
    api = OpenApi()

    login_result = await api.login(APPKEY, APPSECRET)
    if not login_result:
        logger.error("로그인 실패")
        return

    logger.info("로그인 성공!")

    # 테스트할 TR 코드들
    tests = [
        # 1. t1717 - 외국인기관 순매매 상위종목
        (
            "t1717",
            {
                "gubun": "1",  # 1:코스닥
                "fromdt": "20260101",
                "todt": "20260128",
                "pression": "1",  # 1:외국인
                "datatp": "0",  # 0:순매수
                "gubun1": "1",  # 1:금액
            },
            "외국인 순매수 상위 (코스닥)",
        ),
        # 2. t1603 - 시간별 투자자 매매추이
        (
            "t1603",
            {
                "gubun1": "1",  # 1:코스닥
                "gubun2": "0",  # 0:전체
            },
            "시간별 투자자 매매추이",
        ),
        # 3. t1664 - 투자자매매종합 (일자별)
        (
            "t1664",
            {
                "mgubun": "1",  # 1:코스닥
                "vagession": "0",  # 0:순매수
                "fumession": "0",  # 0:외국인
                "date": "20260128",
            },
            "투자자매매종합 일자별",
        ),
        # 4. t1665 - 투자자매매종합 (기간)
        (
            "t1665",
            {
                "market": "1",  # 1:코스닥
                "upcode": "",
                "gubun": "1",  # 1:일별
                "edate": "20260128",
                "sdate": "20260101",
            },
            "투자자매매종합 기간",
        ),
        # 5. t8425 - 전체 투자자매매동향 (일자)
        (
            "t8425",
            {
                "jgubun": "3",  # 3:코스닥
                "tjgubun": "1",  # 1:투자자
                "cts_dt": "",
            },
            "전체 투자자매매동향",
        ),
        # 6. t1637 - 종목별 프로그램매매추이
        (
            "t1637",
            {
                "shcode": "247540",  # 에코프로비엠
                "gubun": "0",  # 0:당일
            },
            "종목별 프로그램매매추이",
        ),
        # 7. t1633 - 기간별 프로그램매매추이
        (
            "t1633",
            {
                "gubun": "0",  # 0:코스피
                "gubun1": "1",  # 1:일별
                "gubun2": "0",  # 0:체결기준
                "gubun3": "0",  # 0:전체
                "sdate": "20260101",
                "edate": "20260128",
            },
            "기간별 프로그램매매추이",
        ),
        # 8. t1302 - 주식현재가 (시세정보)
        (
            "t1302",
            {
                "shcode": "247540",
                "gubun": "0",
            },
            "현재가 시세",
        ),
        # 9. t8407 - API용 멀티현재가
        (
            "t8407",
            {
                "nrec": 1,
                "shcode": "247540",
            },
            "멀티현재가",
        ),
        # 10. FOCCQ33600 - 주식매도주문체결내역 (체결 내역)
        # 주문관련은 skip
        # 11. t1305 - 기간별 시세
        (
            "t1305",
            {
                "shcode": "247540",
                "dwmcode": "1",  # 1:일
                "date": "",
                "idx": "",
                "cnt": 20,
            },
            "기간별 시세 (에코프로비엠)",
        ),
    ]

    success_count = 0
    for tr_code, params, desc in tests:
        result = await test_tr(api, tr_code, params, desc)
        if result:
            success_count += 1
        await asyncio.sleep(0.3)

    logger.info(f"\n=== 테스트 결과: {success_count}/{len(tests)} 성공 ===")


if __name__ == "__main__":
    asyncio.run(main())
